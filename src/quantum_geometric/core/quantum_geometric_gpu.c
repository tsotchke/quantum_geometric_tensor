#include "quantum_geometric/core/quantum_geometric_gpu.h"
#include <stdlib.h>
#include <string.h>

qgt_error_t gpu_malloc(void** ptr, size_t size) {
    if (!ptr || size == 0) {
        return QGT_ERROR_INVALID_PARAMETER;
    }

    #ifdef QGT_ENABLE_METAL
    MTLDevice* device = MTLCreateSystemDefaultDevice();
    if (device) {
        *ptr = [device newBufferWithLength:size options:MTLResourceStorageModeShared].contents;
        if (!*ptr) {
            return QGT_ERROR_GPU_OUT_OF_MEMORY;
        }
        return QGT_SUCCESS;
    }
    #endif

    #ifdef QGT_ENABLE_CUDA
    cudaError_t error = cudaMalloc(ptr, size);
    if (error != cudaSuccess) {
        return QGT_ERROR_GPU_OUT_OF_MEMORY;
    }
    return QGT_SUCCESS;
    #endif

    // No GPU available, use CPU memory as fallback with warning
    *ptr = malloc(size);
    if (!*ptr) {
        return QGT_ERROR_GPU_OUT_OF_MEMORY;
    }
    
    return QGT_SUCCESS;
}

qgt_error_t gpu_free(void* ptr) {
    if (!ptr) {
        return QGT_ERROR_INVALID_PARAMETER;
    }

    #ifdef QGT_ENABLE_METAL
    MTLDevice* device = MTLCreateSystemDefaultDevice();
    if (device) {
        MTLBuffer* buffer = (__bridge MTLBuffer*)ptr;
        [buffer release];
        return QGT_SUCCESS;
    }
    #endif

    #ifdef QGT_ENABLE_CUDA
    cudaError_t error = cudaFree(ptr);
    if (error != cudaSuccess) {
        return QGT_ERROR_GPU_INTERNAL;
    }
    return QGT_SUCCESS;
    #endif

    // CPU memory fallback
    free(ptr);
    return QGT_SUCCESS;
}

void gpu_free_pooled(MemoryPool* pool, void* ptr) {
    if (!pool || !ptr) return;

    #ifdef QGT_ENABLE_METAL
    MTLDevice* device = MTLCreateSystemDefaultDevice();
    if (device) {
        MTLBuffer* buffer = (__bridge MTLBuffer*)ptr;
        [buffer release];
        return;
    }
    #endif

    #ifdef QGT_ENABLE_CUDA
    cudaFree(ptr);
    return;
    #endif

    // CPU memory fallback
    free(ptr);
}

qgt_error_t gpu_memcpy_host_to_device(void* dst, const void* src, size_t size) {
    if (!dst || !src || size == 0) {
        return QGT_ERROR_INVALID_PARAMETER;
    }

    #ifdef QGT_ENABLE_METAL
    MTLDevice* device = MTLCreateSystemDefaultDevice();
    if (device) {
        MTLBuffer* buffer = (__bridge MTLBuffer*)dst;
        void* contents = [buffer contents];
        memcpy(contents, src, size);
        return QGT_SUCCESS;
    }
    #endif

    #ifdef QGT_ENABLE_CUDA
    cudaError_t error = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        return QGT_ERROR_GPU_INTERNAL;
    }
    return QGT_SUCCESS;
    #endif

    // CPU memory fallback
    memcpy(dst, src, size);
    return QGT_SUCCESS;
}

qgt_error_t gpu_memcpy_device_to_host(void* dst, const void* src, size_t size) {
    if (!dst || !src || size == 0) {
        return QGT_ERROR_INVALID_PARAMETER;
    }

    #ifdef QGT_ENABLE_METAL
    MTLDevice* device = MTLCreateSystemDefaultDevice();
    if (device) {
        MTLBuffer* buffer = (__bridge MTLBuffer*)src;
        void* contents = [buffer contents];
        memcpy(dst, contents, size);
        return QGT_SUCCESS;
    }
    #endif

    #ifdef QGT_ENABLE_CUDA
    cudaError_t error = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        return QGT_ERROR_GPU_INTERNAL;
    }
    return QGT_SUCCESS;
    #endif

    // CPU memory fallback
    memcpy(dst, src, size);
    return QGT_SUCCESS;
}

// Global state
static struct {
    bool initialized;
    gpu_device_state_t* devices;
    int num_devices;
    gpu_memory_pool_t* memory_pools;
    kernel_launch_info_t kernel_queue[MAX_QUEUED_KERNELS];
    size_t queue_head;
    size_t queue_tail;
    pthread_mutex_t queue_lock;
    gpu_error_t last_error;
} gpu_state = {0};

// Forward declarations of static functions
static void* allocate_from_pool(gpu_memory_pool_t* pool, size_t size);
static int queue_kernel(const char* name, const void* args, size_t args_size,
                       dim3 grid, dim3 block, size_t shared_memory,
                       int stream_id);
static void process_kernel_queue(void);
static int get_optimal_device(size_t memory_required, int compute_capability);

// Implementation of public functions
int qg_gpu_init(void) {
    if (gpu_state.initialized) return QG_GPU_SUCCESS;

    pthread_mutex_init(&gpu_state.queue_lock, NULL);

    // Initialize Metal
#ifdef QGT_ENABLE_METAL
    MTLDevice* device = MTLCreateSystemDefaultDevice();
    if (device) {
        gpu_state.num_devices = 1;
        gpu_state.devices = calloc(1, sizeof(gpu_device_state_t));
        if (!gpu_state.devices) {
            gpu_state.last_error = QG_GPU_ERROR_OUT_OF_MEMORY;
            return QG_GPU_ERROR_OUT_OF_MEMORY;
        }

        // Get device capabilities
        gpu_state.devices[0].device_id = 0;
        gpu_state.devices[0].total_memory = [device maxBufferLength];
        gpu_state.devices[0].free_memory = gpu_state.devices[0].total_memory;
        gpu_state.devices[0].compute_capability = 2;  // Metal 2
        gpu_state.devices[0].num_multiprocessors = 
            device.maxThreadgroupsPerThreadblock;
        gpu_state.devices[0].max_threads_per_block = 
            device.maxThreadsPerThreadgroup.width;
        gpu_state.devices[0].warp_size = WARP_SIZE;
        strncpy(gpu_state.devices[0].name, device.name.UTF8String, 255);
        gpu_state.devices[0].unified_memory = true;
        gpu_state.devices[0].concurrent_kernels = true;

        // Initialize memory pool
        gpu_state.memory_pools = calloc(1, sizeof(gpu_memory_pool_t));
        if (!gpu_state.memory_pools) {
            free(gpu_state.devices);
            gpu_state.last_error = QG_GPU_ERROR_OUT_OF_MEMORY;
            return QG_GPU_ERROR_OUT_OF_MEMORY;
        }

        pthread_mutex_init(&gpu_state.memory_pools[0].lock, NULL);
        gpu_state.memory_pools[0].block_size = MEMORY_POOL_BLOCK_SIZE;
        gpu_state.memory_pools[0].uses_unified_memory = true;

        gpu_state.initialized = true;
        return QG_GPU_SUCCESS;
    }
#endif

    // Initialize CUDA
#ifdef QGT_ENABLE_CUDA
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count > 0) {
        gpu_state.num_devices = device_count;
        gpu_state.devices = calloc(device_count, sizeof(gpu_device_state_t));
        gpu_state.memory_pools = calloc(device_count, sizeof(gpu_memory_pool_t));
        
        if (!gpu_state.devices || !gpu_state.memory_pools) {
            free(gpu_state.devices);
            free(gpu_state.memory_pools);
            gpu_state.last_error = QG_GPU_ERROR_OUT_OF_MEMORY;
            return QG_GPU_ERROR_OUT_OF_MEMORY;
        }

        // Initialize each device
        for (int i = 0; i < device_count; i++) {
            cudaDeviceProp props;
            cudaGetDeviceProperties(&props, i);
            
            gpu_state.devices[i].device_id = i;
            gpu_state.devices[i].total_memory = props.totalGlobalMem;
            gpu_state.devices[i].compute_capability = 
                props.major * 10 + props.minor;
            gpu_state.devices[i].num_multiprocessors = props.multiProcessorCount;
            gpu_state.devices[i].max_threads_per_block = 
                props.maxThreadsPerBlock;
            gpu_state.devices[i].warp_size = props.warpSize;
            strncpy(gpu_state.devices[i].name, props.name, 255);
            gpu_state.devices[i].unified_memory = props.unifiedAddressing;
            gpu_state.devices[i].concurrent_kernels = props.concurrentKernels;

            // Initialize memory pool
            pthread_mutex_init(&gpu_state.memory_pools[i].lock, NULL);
            gpu_state.memory_pools[i].block_size = MEMORY_POOL_BLOCK_SIZE;
            gpu_state.memory_pools[i].uses_unified_memory = 
                props.unifiedAddressing;
        }

        gpu_state.initialized = true;
        return QG_GPU_SUCCESS;
    }
#endif

    gpu_state.last_error = QG_GPU_ERROR_NO_DEVICE;
    return QG_GPU_ERROR_NO_DEVICE;
}

void qg_gpu_cleanup(void) {
    if (!gpu_state.initialized) return;

    // Clean up memory pools
    for (int i = 0; i < gpu_state.num_devices; i++) {
        pthread_mutex_lock(&gpu_state.memory_pools[i].lock);
        
        if (gpu_state.memory_pools[i].base_ptr) {
            if (gpu_state.memory_pools[i].uses_unified_memory) {
                #ifdef QGT_ENABLE_CUDA
                cudaFreeManaged(gpu_state.memory_pools[i].base_ptr);
                #endif
            } else {
                #ifdef QGT_ENABLE_METAL
                qgt_metal_free_pool(gpu_state.memory_pools[i].base_ptr);
                #endif
                #ifdef QGT_ENABLE_CUDA
                cudaFree(gpu_state.memory_pools[i].base_ptr);
                #endif
            }
        }
        
        pthread_mutex_unlock(&gpu_state.memory_pools[i].lock);
        pthread_mutex_destroy(&gpu_state.memory_pools[i].lock);
    }

    // Clean up kernel queue
    pthread_mutex_lock(&gpu_state.queue_lock);
    for (size_t i = gpu_state.queue_head; i != gpu_state.queue_tail; 
         i = (i + 1) % MAX_QUEUED_KERNELS) {
        free(gpu_state.kernel_queue[i].args);
    }
    pthread_mutex_unlock(&gpu_state.queue_lock);
    pthread_mutex_destroy(&gpu_state.queue_lock);

    // Clean up device resources
    #ifdef QGT_ENABLE_METAL
    qgt_metal_cleanup();
    #endif

    #ifdef QGT_ENABLE_CUDA
    for (int i = 0; i < gpu_state.num_devices; i++) {
        cudaSetDevice(i);
        cudaDeviceReset();
    }
    #endif

    // Free allocated memory
    free(gpu_state.devices);
    free(gpu_state.memory_pools);
    
    // Reset state
    memset(&gpu_state, 0, sizeof(gpu_state));
}

int qg_gpu_get_device_count(int* count) {
    if (!gpu_state.initialized || !count) {
        gpu_state.last_error = QG_GPU_ERROR_NOT_INITIALIZED;
        return QG_GPU_ERROR_NOT_INITIALIZED;
    }

    *count = gpu_state.num_devices;
    return QG_GPU_SUCCESS;
}

int qg_gpu_get_device_info(int device_id, gpu_device_info_t* info) {
    if (!gpu_state.initialized || !info) {
        gpu_state.last_error = QG_GPU_ERROR_NOT_INITIALIZED;
        return QG_GPU_ERROR_NOT_INITIALIZED;
    }

    if (device_id < 0 || device_id >= gpu_state.num_devices) {
        gpu_state.last_error = QG_GPU_ERROR_INVALID_DEVICE;
        return QG_GPU_ERROR_INVALID_DEVICE;
    }

    gpu_device_state_t* device = &gpu_state.devices[device_id];
    info->device_id = device->device_id;
    info->total_memory = device->total_memory;
    info->free_memory = device->free_memory;
    info->compute_capability_major = device->compute_capability / 10;
    info->compute_capability_minor = device->compute_capability % 10;
    info->max_threads_per_block = device->max_threads_per_block;
    memcpy(info->max_block_dimensions, device->max_block_dimensions, 
           sizeof(info->max_block_dimensions));
    memcpy(info->max_grid_dimensions, device->max_grid_dimensions,
           sizeof(info->max_grid_dimensions));

    return QG_GPU_SUCCESS;
}

int qg_gpu_set_device(int device_id) {
    if (!gpu_state.initialized) {
        gpu_state.last_error = QG_GPU_ERROR_NOT_INITIALIZED;
        return QG_GPU_ERROR_NOT_INITIALIZED;
    }

    if (device_id < 0 || device_id >= gpu_state.num_devices) {
        gpu_state.last_error = QG_GPU_ERROR_INVALID_DEVICE;
        return QG_GPU_ERROR_INVALID_DEVICE;
    }

#ifdef QGT_ENABLE_METAL
    return qgt_metal_set_device(device_id);
#endif

#ifdef QGT_ENABLE_CUDA
    cudaError_t error = cudaSetDevice(device_id);
    if (error != cudaSuccess) {
        gpu_state.last_error = QG_GPU_ERROR_INVALID_DEVICE;
        return QG_GPU_ERROR_INVALID_DEVICE;
    }
#endif

    return QG_GPU_SUCCESS;
}

int qg_gpu_allocate(gpu_buffer_t* buffer, size_t size) {
    if (!gpu_state.initialized || !buffer || size == 0) {
        gpu_state.last_error = QG_GPU_ERROR_INVALID_VALUE;
        return QG_GPU_ERROR_INVALID_VALUE;
    }

    buffer->size = size;
    buffer->is_pinned = false;

    // Try to allocate from pool first
    int device_id;
    cudaGetDevice(&device_id);
    buffer->device_ptr = allocate_from_pool(&gpu_state.memory_pools[device_id], size);
    
    if (buffer->device_ptr) {
        return QG_GPU_SUCCESS;
    }

    // Fall back to direct allocation
#ifdef QGT_ENABLE_METAL
    buffer->device_ptr = qgt_metal_allocate(size);
#endif

#ifdef QGT_ENABLE_CUDA
    cudaError_t error = cudaMalloc(&buffer->device_ptr, size);
    if (error != cudaSuccess) {
        buffer->device_ptr = NULL;
    }
#endif

    if (!buffer->device_ptr) {
        gpu_state.last_error = QG_GPU_ERROR_OUT_OF_MEMORY;
        return QG_GPU_ERROR_OUT_OF_MEMORY;
    }

    return QG_GPU_SUCCESS;
}

int qg_gpu_allocate_pinned(gpu_buffer_t* buffer, size_t size) {
    if (!gpu_state.initialized || !buffer || size == 0) {
        gpu_state.last_error = QG_GPU_ERROR_INVALID_VALUE;
        return QG_GPU_ERROR_INVALID_VALUE;
    }

    buffer->size = size;
    buffer->is_pinned = true;

#ifdef QGT_ENABLE_METAL
    buffer->device_ptr = qgt_metal_allocate_pinned(size);
#endif

#ifdef QGT_ENABLE_CUDA
    cudaError_t error = cudaMallocHost(&buffer->device_ptr, size);
    if (error != cudaSuccess) {
        buffer->device_ptr = NULL;
    }
#endif

    if (!buffer->device_ptr) {
        gpu_state.last_error = QG_GPU_ERROR_OUT_OF_MEMORY;
        return QG_GPU_ERROR_OUT_OF_MEMORY;
    }

    return QG_GPU_SUCCESS;
}

int qg_gpu_free(gpu_buffer_t* buffer) {
    if (!gpu_state.initialized || !buffer || !buffer->device_ptr) {
        gpu_state.last_error = QG_GPU_ERROR_INVALID_VALUE;
        return QG_GPU_ERROR_INVALID_VALUE;
    }

    if (buffer->is_pinned) {
#ifdef QGT_ENABLE_METAL
        qgt_metal_free_pinned(buffer->device_ptr);
#endif

#ifdef QGT_ENABLE_CUDA
        cudaFreeHost(buffer->device_ptr);
#endif
    } else {
#ifdef QGT_ENABLE_METAL
        qgt_metal_free(buffer->device_ptr);
#endif

#ifdef QGT_ENABLE_CUDA
        cudaFree(buffer->device_ptr);
#endif
    }

    buffer->device_ptr = NULL;
    buffer->size = 0;
    return QG_GPU_SUCCESS;
}

int qg_gpu_memcpy_to_device(gpu_buffer_t* dst, const void* src, size_t size) {
    if (!gpu_state.initialized || !dst || !src || !dst->device_ptr || 
        size > dst->size) {
        gpu_state.last_error = QG_GPU_ERROR_INVALID_VALUE;
        return QG_GPU_ERROR_INVALID_VALUE;
    }

#ifdef QGT_ENABLE_METAL
    if (qgt_metal_upload(src, dst->device_ptr, size) != 0) {
        gpu_state.last_error = QG_GPU_ERROR_LAUNCH_FAILED;
        return QG_GPU_ERROR_LAUNCH_FAILED;
    }
#endif

#ifdef QGT_ENABLE_CUDA
    cudaError_t error = cudaMemcpy(dst->device_ptr, src, size, 
                                  cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        gpu_state.last_error = QG_GPU_ERROR_LAUNCH_FAILED;
        return QG_GPU_ERROR_LAUNCH_FAILED;
    }
#endif

    return QG_GPU_SUCCESS;
}

int qg_gpu_memcpy_to_host(void* dst, const gpu_buffer_t* src, size_t size) {
    if (!gpu_state.initialized || !dst || !src || !src->device_ptr || 
        size > src->size) {
        gpu_state.last_error = QG_GPU_ERROR_INVALID_VALUE;
        return QG_GPU_ERROR_INVALID_VALUE;
    }

#ifdef QGT_ENABLE_METAL
    if (qgt_metal_download(src->device_ptr, dst, size) != 0) {
        gpu_state.last_error = QG_GPU_ERROR_LAUNCH_FAILED;
        return QG_GPU_ERROR_LAUNCH_FAILED;
    }
#endif

#ifdef QGT_ENABLE_CUDA
    cudaError_t error = cudaMemcpy(dst, src->device_ptr, size, 
                                  cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        gpu_state.last_error = QG_GPU_ERROR_LAUNCH_FAILED;
        return QG_GPU_ERROR_LAUNCH_FAILED;
    }
#endif

    return QG_GPU_SUCCESS;
}

int qg_gpu_create_stream(int* stream_id) {
    if (!gpu_state.initialized || !stream_id) {
        gpu_state.last_error = QG_GPU_ERROR_NOT_INITIALIZED;
        return QG_GPU_ERROR_NOT_INITIALIZED;
    }

#ifdef QGT_ENABLE_METAL
    MTLCommandQueue* queue = qgt_metal_create_command_queue();
    if (!queue) {
        gpu_state.last_error = QG_GPU_ERROR_LAUNCH_FAILED;
        return QG_GPU_ERROR_LAUNCH_FAILED;
    }
    *stream_id = qgt_metal_register_command_queue(queue);
#endif

#ifdef QGT_ENABLE_CUDA
    cudaStream_t stream;
    cudaError_t error = cudaStreamCreate(&stream);
    if (error != cudaSuccess) {
        gpu_state.last_error = QG_GPU_ERROR_LAUNCH_FAILED;
        return QG_GPU_ERROR_LAUNCH_FAILED;
    }
    *stream_id = qgt_cuda_register_stream(stream);
#endif

    return QG_GPU_SUCCESS;
}

int qg_gpu_destroy_stream(int stream_id) {
    if (!gpu_state.initialized) {
        gpu_state.last_error = QG_GPU_ERROR_NOT_INITIALIZED;
        return QG_GPU_ERROR_NOT_INITIALIZED;
    }

#ifdef QGT_ENABLE_METAL
    MTLCommandQueue* queue = qgt_metal_get_command_queue(stream_id);
    if (!queue) {
        gpu_state.last_error = QG_GPU_ERROR_INVALID_VALUE;
        return QG_GPU_ERROR_INVALID_VALUE;
    }
    qgt_metal_unregister_command_queue(stream_id);
#endif

#ifdef QGT_ENABLE_CUDA
    cudaStream_t stream = qgt_cuda_get_stream(stream_id);
    if (!stream) {
        gpu_state.last_error = QG_GPU_ERROR_INVALID_VALUE;
        return QG_GPU_ERROR_INVALID_VALUE;
    }
    cudaStreamDestroy(stream);
    qgt_cuda_unregister_stream(stream_id);
#endif

    return QG_GPU_SUCCESS;
}

int qg_gpu_synchronize_stream(int stream_id) {
    if (!gpu_state.initialized) {
        gpu_state.last_error = QG_GPU_ERROR_NOT_INITIALIZED;
        return QG_GPU_ERROR_NOT_INITIALIZED;
    }

#ifdef QGT_ENABLE_METAL
    MTLCommandQueue* queue = qgt_metal_get_command_queue(stream_id);
    if (!queue) {
        gpu_state.last_error = QG_GPU_ERROR_INVALID_VALUE;
        return QG_GPU_ERROR_INVALID_VALUE;
    }
    [queue waitUntilCompleted];
#endif

#ifdef QGT_ENABLE_CUDA
    cudaStream_t stream = qgt_cuda_get_stream(stream_id);
    if (!stream) {
        gpu_state.last_error = QG_GPU_ERROR_INVALID_VALUE;
        return QG_GPU_ERROR_INVALID_VALUE;
    }
    cudaError_t error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess) {
        gpu_state.last_error = QG_GPU_ERROR_LAUNCH_FAILED;
        return QG_GPU_ERROR_LAUNCH_FAILED;
    }
#endif

    return QG_GPU_SUCCESS;
}

const char* qg_gpu_get_error_string(gpu_error_t error) {
    switch (error) {
        case QG_GPU_SUCCESS:
            return "Success";
        case QG_GPU_ERROR_NO_DEVICE:
            return "No GPU device available";
        case QG_GPU_ERROR_OUT_OF_MEMORY:
            return "Out of memory";
        case QG_GPU_ERROR_INVALID_DEVICE:
            return "Invalid device";
        case QG_GPU_ERROR_LAUNCH_FAILED:
            return "Kernel launch failed";
        case QG_GPU_ERROR_INVALID_VALUE:
            return "Invalid value";
        case QG_GPU_ERROR_NOT_INITIALIZED:
            return "GPU not initialized";
        default:
            return "Unknown error";
    }
}

gpu_error_t qg_gpu_get_last_error(void) {
    return gpu_state.last_error;
}

// Static helper functions
static void* allocate_from_pool(gpu_memory_pool_t* pool, size_t size) {
    pthread_mutex_lock(&pool->lock);
    
    // Round up to block size
    size_t aligned_size = (size + pool->block_size - 1) & ~(pool->block_size - 1);
    
    // Check if we need to grow the pool
    if (pool->used_size + aligned_size > pool->total_size) {
        size_t new_size = pool->total_size * 2;
        while (new_size < pool->used_size + aligned_size) {
            new_size *= 2;
        }
        
        void* new_base = NULL;
        if (pool->uses_unified_memory) {
            #ifdef QGT_ENABLE_CUDA
            cudaMallocManaged(&new_base, new_size);
            #endif
        } else {
            #ifdef QGT_ENABLE_METAL
            new_base = qgt_metal_allocate_pool(new_size);
            #endif
            #ifdef QGT_ENABLE_CUDA
            cudaMalloc(&new_base, new_size);
            #endif
        }
        
        if (!new_base) {
            pthread_mutex_unlock(&pool->lock);
            return NULL;
        }
        
        // Copy existing data
        if (pool->base_ptr) {
            if (pool->uses_unified_memory) {
                memcpy(new_base, pool->base_ptr, pool->used_size);
            } else {
                #ifdef QGT_ENABLE_METAL
                qgt_metal_memcpy(new_base, pool->base_ptr, pool->used_size);
                #endif
                #ifdef QGT_ENABLE_CUDA
                cudaMemcpy(new_base, pool->base_ptr, pool->used_size,
                          cudaMemcpyDeviceToDevice);
                #endif
            }
            
            // Free old memory
            if (pool->uses_unified_memory) {
                #ifdef QGT_ENABLE_CUDA
                cudaFreeManaged(pool->base_ptr);
                #endif
            } else {
                #ifdef QGT_ENABLE_METAL
                qgt_metal_free_pool(pool->base_ptr);
                #endif
                #ifdef QGT_ENABLE_CUDA
                cudaFree(pool->base_ptr);
                #endif
            }
        }
        
        pool->base_ptr = new_base;
        pool->total_size = new_size;
    }
    
    void* ptr = (char*)pool->base_ptr + pool->used_size;
    pool->used_size += aligned_size;
    
    pthread_mutex_unlock(&pool->lock);
    return ptr;
}

static int queue_kernel(const char* name, const void* args, size_t args_size,
                       dim3 grid, dim3 block, size_t shared_memory,
                       int stream_id) {
    pthread_mutex_lock(&gpu_state.queue_lock);
    
    // Check if queue is full
    size_t next_tail = (gpu_state.queue_tail + 1) % MAX_QUEUED_KERNELS;
    if (next_tail == gpu_state.queue_head) {
        pthread_mutex_unlock(&gpu_state.queue_lock);
        return QG_GPU_ERROR_LAUNCH_FAILED;
    }
    
    // Copy kernel arguments
    kernel_launch_info_t* info = &gpu_state.kernel_queue[gpu_state.queue_tail];
    info->name = name;
    info->args = malloc(args_size);
    if (!info->args) {
        pthread_mutex_unlock(&gpu_state.queue_lock);
        return QG_GPU_ERROR_OUT_OF_MEMORY;
    }
    memcpy(info->args, args, args_size);
    
    info->args_size = args_size;
    info->grid = grid;
    info->block = block;
    info->shared_memory = shared_memory;
    info->stream_id = stream_id;
    
    gpu_state.queue_tail = next_tail;
    
    pthread_mutex_unlock(&gpu_state.queue_lock);
    return QG_GPU_SUCCESS;
}

static void process_kernel_queue(void) {
    pthread_mutex_lock(&gpu_state.queue_lock);
    
    while (gpu_state.queue_head != gpu_state.queue_tail) {
        kernel_launch_info_t* info = 
            &gpu_state.kernel_queue[gpu_state.queue_head];
            
        #ifdef QGT_ENABLE_METAL
        qgt_metal_launch_kernel(info->name, info->args, info->args_size,
                              info->grid, info->block, info->shared_memory,
                              info->stream_id);
        #endif
        
        #ifdef QGT_ENABLE_CUDA
        qgt_cuda_launch_kernel(info->name, info->args, info->args_size,
                             info->grid, info->block, info->shared_memory,
                             info->stream_id);
        #endif
        
        free(info->args);
        gpu_state.queue_head = (gpu_state.queue_head + 1) % MAX_QUEUED_KERNELS;
    }
    
    pthread_mutex_unlock(&gpu_state.queue_lock);
}

static int get_optimal_device(size_t memory_required, int compute_capability) {
    int best_device = -1;
    float best_score = -1.0f;
    
    for (int i = 0; i < gpu_state.num_devices; i++) {
        gpu_device_state_t* device = &gpu_state.devices[i];
        
        // Skip if device doesn't meet minimum requirements
        if (device->compute_capability < compute_capability) continue;
        
        // Calculate device score based on:
        // - Available memory
        // - Compute capability
        // - Number of multiprocessors
        // - Current workload
        float memory_score = (float)device->free_memory / memory_required;
        float compute_score = (float)device->compute_capability / 100.0f;
        float sm_score = (float)device->num_multiprocessors / 100.0f;
        float workload_score = 1.0f - ((float)device->used_memory / 
                                     device->total_memory);
        
        float total_score = memory_score * 0.4f +
                           compute_score * 0.2f +
                           sm_score * 0.2f +
                           workload_score * 0.2f;
        
        if (total_score > best_score) {
            best_score = total_score;
            best_device = i;
        }
    }
    
    return best_device;
}
