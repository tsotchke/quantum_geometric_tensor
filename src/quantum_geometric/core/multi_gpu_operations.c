#include "quantum_geometric/core/multi_gpu_operations.h"
#include <stdlib.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#ifndef NO_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>

// Multi-GPU parameters
#define MAX_GPUS 8
#define CHUNK_SIZE (1024 * 1024)  // 1MB chunks
#define PIPELINE_DEPTH 3

// GPU context
typedef struct {
    int device_id;
    cudaStream_t stream;
    ncclComm_t nccl_comm;
    void* device_memory;
    size_t memory_size;
    bool is_active;
} GPUContext;

// Multi-GPU manager implementation
struct MultiGPUManager {
    GPUContext contexts[MAX_GPUS];
    size_t num_gpus;
    ncclUniqueId nccl_id;
    cudaEvent_t events[PIPELINE_DEPTH];
    bool initialized;
};

#else

// CPU fallback implementation when CUDA is disabled
// Provides single-device CPU emulation for multi-GPU operations

#define MAX_GPUS 1  // CPU fallback uses single "device"

typedef struct {
    int device_id;
    void* host_memory;      // CPU memory buffer
    size_t memory_size;
    size_t element_count;
    DataType data_type;
    bool is_active;
} CPUContext;

struct MultiGPUManager {
    CPUContext context;     // Single CPU context
    size_t num_devices;     // Always 1 for CPU fallback
    bool initialized;
};

#endif

// Initialize multi-GPU manager
MultiGPUManager* init_multi_gpu_manager(void) {
#ifndef NO_CUDA
    MultiGPUManager* manager = malloc(sizeof(MultiGPUManager));
    if (!manager) return NULL;
    
    manager->initialized = false;
    manager->num_gpus = 0;
    
    // Get number of available GPUs
    int device_count;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess) {
        free(manager);
        return NULL;
    }
    
    manager->num_gpus = min(device_count, MAX_GPUS);
    
    // Initialize NCCL
    if (ncclGetUniqueId(&manager->nccl_id) != ncclSuccess) {
        free(manager);
        return NULL;
    }
    
    // Initialize each GPU context
    for (size_t i = 0; i < manager->num_gpus; i++) {
        GPUContext* ctx = &manager->contexts[i];
        ctx->device_id = i;
        
        // Set device
        if (cudaSetDevice(i) != cudaSuccess) {
            cleanup_multi_gpu_manager(manager);
            return NULL;
        }
        
        // Create CUDA stream
        if (cudaStreamCreate(&ctx->stream) != cudaSuccess) {
            cleanup_multi_gpu_manager(manager);
            return NULL;
        }
        
        // Initialize NCCL communicator
        if (ncclCommInitRank(&ctx->nccl_comm,
                           manager->num_gpus,
                           manager->nccl_id,
                           i) != ncclSuccess) {
            cleanup_multi_gpu_manager(manager);
            return NULL;
        }
        
        ctx->is_active = true;
    }
    
    // Create events for synchronization
    for (size_t i = 0; i < PIPELINE_DEPTH; i++) {
        if (cudaEventCreate(&manager->events[i]) != cudaSuccess) {
            cleanup_multi_gpu_manager(manager);
            return NULL;
        }
    }
    
    manager->initialized = true;
    return manager;
#else
    // CPU fallback implementation
    MultiGPUManager* manager = malloc(sizeof(MultiGPUManager));
    if (!manager) return NULL;

    // Initialize single CPU context
    manager->num_devices = 1;
    manager->context.device_id = 0;
    manager->context.host_memory = NULL;
    manager->context.memory_size = 0;
    manager->context.element_count = 0;
    manager->context.is_active = true;
    manager->initialized = true;

    return manager;
#endif
}

// Distribute tensor across GPUs
int distribute_tensor(MultiGPUManager* manager,
                     const void* host_data,
                     size_t size,
                     DataType dtype) {
#ifndef NO_CUDA
    if (!manager || !manager->initialized || !host_data) return -1;
    
    size_t element_size;
    ncclDataType_t nccl_dtype;
    
    switch (dtype) {
        case TYPE_FLOAT:
            element_size = sizeof(float);
            nccl_dtype = ncclFloat;
            break;
        case TYPE_DOUBLE:
            element_size = sizeof(double);
            nccl_dtype = ncclDouble;
            break;
        default:
            return -1;
    }
    
    // Calculate chunk size per GPU
    size_t total_elements = size / element_size;
    size_t elements_per_gpu = (total_elements + manager->num_gpus - 1) /
                             manager->num_gpus;
    
    // Distribute data to each GPU
    for (size_t i = 0; i < manager->num_gpus; i++) {
        GPUContext* ctx = &manager->contexts[i];
        
        // Calculate chunk for this GPU
        size_t start = i * elements_per_gpu;
        size_t count = min(elements_per_gpu,
                          total_elements - start);
        size_t bytes = count * element_size;
        
        // Allocate GPU memory
        if (cudaSetDevice(ctx->device_id) != cudaSuccess) return -1;
        
        if (cudaMalloc(&ctx->device_memory, bytes) != cudaSuccess) {
            return -1;
        }
        
        ctx->memory_size = bytes;
        
        // Copy data to GPU
        if (cudaMemcpyAsync(ctx->device_memory,
                           (char*)host_data + start * element_size,
                           bytes,
                           cudaMemcpyHostToDevice,
                           ctx->stream) != cudaSuccess) {
            return -1;
        }
    }
    
    // Synchronize all GPUs
    for (size_t i = 0; i < manager->num_gpus; i++) {
        if (cudaStreamSynchronize(manager->contexts[i].stream) !=
            cudaSuccess) {
            return -1;
        }
    }
    
    return 0;
#else
    // CPU fallback: allocate host memory and copy data
    if (!manager || !manager->initialized || !host_data || size == 0) return -1;

    size_t element_size;
    switch (dtype) {
        case TYPE_FLOAT:
            element_size = sizeof(float);
            break;
        case TYPE_DOUBLE:
            element_size = sizeof(double);
            break;
        default:
            return -1;
    }

    // Free any existing memory
    if (manager->context.host_memory) {
        free(manager->context.host_memory);
    }

    // Allocate CPU memory
    manager->context.host_memory = malloc(size);
    if (!manager->context.host_memory) {
        return -1;
    }

    // Copy data to our buffer
    memcpy(manager->context.host_memory, host_data, size);
    manager->context.memory_size = size;
    manager->context.element_count = size / element_size;
    manager->context.data_type = dtype;

    return 0;
#endif
}

// All-reduce across GPUs
int all_reduce(MultiGPUManager* manager,
              void* data,
              size_t count,
              DataType dtype,
              ReduceOp op) {
#ifndef NO_CUDA
    if (!manager || !manager->initialized || !data) return -1;
    
    ncclDataType_t nccl_dtype;
    ncclRedOp_t nccl_op;
    
    // Map data type
    switch (dtype) {
        case TYPE_FLOAT:
            nccl_dtype = ncclFloat;
            break;
        case TYPE_DOUBLE:
            nccl_dtype = ncclDouble;
            break;
        default:
            return -1;
    }
    
    // Map reduction operation
    switch (op) {
        case REDUCE_SUM:
            nccl_op = ncclSum;
            break;
        case REDUCE_PROD:
            nccl_op = ncclProd;
            break;
        case REDUCE_MAX:
            nccl_op = ncclMax;
            break;
        case REDUCE_MIN:
            nccl_op = ncclMin;
            break;
        default:
            return -1;
    }
    
    // Perform all-reduce
    ncclGroupStart();
    
    for (size_t i = 0; i < manager->num_gpus; i++) {
        GPUContext* ctx = &manager->contexts[i];
        if (ncclAllReduce(ctx->device_memory,
                         ctx->device_memory,
                         count,
                         nccl_dtype,
                         nccl_op,
                         ctx->nccl_comm,
                         ctx->stream) != ncclSuccess) {
            return -1;
        }
    }
    
    ncclGroupEnd();
    
    // Synchronize streams
    for (size_t i = 0; i < manager->num_gpus; i++) {
        if (cudaStreamSynchronize(manager->contexts[i].stream) !=
            cudaSuccess) {
            return -1;
        }
    }
    
    return 0;
#else
    // CPU fallback: perform reduction on single CPU buffer
    if (!manager || !manager->initialized || !data || count == 0) return -1;

    // For single-device CPU, all-reduce with external data means:
    // reduce data into internal buffer, then copy result back
    void* internal = manager->context.host_memory;
    if (!internal) {
        // No internal buffer - just return success (data is already reduced with itself)
        return 0;
    }

    size_t element_size;
    switch (dtype) {
        case TYPE_FLOAT:
            element_size = sizeof(float);
            break;
        case TYPE_DOUBLE:
            element_size = sizeof(double);
            break;
        default:
            return -1;
    }

    // Perform element-wise reduction between external data and internal buffer
    size_t max_count = manager->context.element_count < count ?
                       manager->context.element_count : count;

    if (dtype == TYPE_FLOAT) {
        float* ext = (float*)data;
        float* intl = (float*)internal;
        for (size_t i = 0; i < max_count; i++) {
            switch (op) {
                case REDUCE_SUM:
                    ext[i] = ext[i] + intl[i];
                    break;
                case REDUCE_PROD:
                    ext[i] = ext[i] * intl[i];
                    break;
                case REDUCE_MAX:
                    ext[i] = ext[i] > intl[i] ? ext[i] : intl[i];
                    break;
                case REDUCE_MIN:
                    ext[i] = ext[i] < intl[i] ? ext[i] : intl[i];
                    break;
                default:
                    return -1;
            }
        }
    } else {  // TYPE_DOUBLE
        double* ext = (double*)data;
        double* intl = (double*)internal;
        for (size_t i = 0; i < max_count; i++) {
            switch (op) {
                case REDUCE_SUM:
                    ext[i] = ext[i] + intl[i];
                    break;
                case REDUCE_PROD:
                    ext[i] = ext[i] * intl[i];
                    break;
                case REDUCE_MAX:
                    ext[i] = ext[i] > intl[i] ? ext[i] : intl[i];
                    break;
                case REDUCE_MIN:
                    ext[i] = ext[i] < intl[i] ? ext[i] : intl[i];
                    break;
                default:
                    return -1;
            }
        }
    }

    // Update internal buffer with reduced result
    memcpy(internal, data, max_count * element_size);

    return 0;
#endif
}

// Gather results from GPUs
int gather_results(MultiGPUManager* manager,
                  void* host_data,
                  size_t size) {
#ifndef NO_CUDA
    if (!manager || !manager->initialized || !host_data) return -1;
    
    // Copy data back from each GPU
    for (size_t i = 0; i < manager->num_gpus; i++) {
        GPUContext* ctx = &manager->contexts[i];
        
        if (cudaSetDevice(ctx->device_id) != cudaSuccess) return -1;
        
        if (cudaMemcpyAsync((char*)host_data + i * ctx->memory_size,
                           ctx->device_memory,
                           ctx->memory_size,
                           cudaMemcpyDeviceToHost,
                           ctx->stream) != cudaSuccess) {
            return -1;
        }
    }
    
    // Synchronize all transfers
    for (size_t i = 0; i < manager->num_gpus; i++) {
        if (cudaStreamSynchronize(manager->contexts[i].stream) !=
            cudaSuccess) {
            return -1;
        }
    }
    
    return 0;
#else
    // CPU fallback: copy internal buffer to host_data
    if (!manager || !manager->initialized || !host_data) return -1;

    if (!manager->context.host_memory || manager->context.memory_size == 0) {
        // No data to gather
        return 0;
    }

    // Copy from internal buffer to output
    size_t copy_size = size < manager->context.memory_size ?
                       size : manager->context.memory_size;
    memcpy(host_data, manager->context.host_memory, copy_size);

    return 0;
#endif
}

// Execute kernel on multiple GPUs
int execute_multi_gpu_kernel(MultiGPUManager* manager,
                           KernelFunction kernel,
                           void* args,
                           const KernelConfig* config) {
#ifndef NO_CUDA
    if (!manager || !manager->initialized || !kernel || !config) return -1;

    size_t grid_size = config->grid_size;
    size_t block_size = config->block_size;

    // Launch kernel on each GPU
    for (size_t i = 0; i < manager->num_gpus; i++) {
        GPUContext* ctx = &manager->contexts[i];

        if (cudaSetDevice(ctx->device_id) != cudaSuccess) return -1;

        // Calculate device-specific grid size
        size_t device_grid = (grid_size + manager->num_gpus - 1) /
                            manager->num_gpus;

        // Launch kernel
        kernel<<<device_grid, block_size, 0, ctx->stream>>>(
            ctx->device_memory,
            args);
        
        if (cudaGetLastError() != cudaSuccess) return -1;
    }
    
    // Synchronize all kernels
    for (size_t i = 0; i < manager->num_gpus; i++) {
        if (cudaStreamSynchronize(manager->contexts[i].stream) !=
            cudaSuccess) {
            return -1;
        }
    }
    
    return 0;
#else
    // CPU fallback: execute callback function if provided
    // Note: KernelFunction in CPU mode should be a CPU function pointer
    if (!manager || !manager->initialized) return -1;

    // For CPU fallback, we interpret KernelFunction as a CPU callback
    // that takes (data, args) parameters
    if (kernel && manager->context.host_memory) {
        // Execute the "kernel" as a CPU function on our data
        // The kernel function signature must match: void (*)(void* data, void* args)
        typedef void (*CPUKernelFunction)(void*, void*);
        CPUKernelFunction cpu_kernel = (CPUKernelFunction)kernel;
        cpu_kernel(manager->context.host_memory, args);
    }

    return 0;
#endif
}

// Clean up multi-GPU manager
void cleanup_multi_gpu_manager(MultiGPUManager* manager) {
#ifndef NO_CUDA
    if (!manager) return;
    
    for (size_t i = 0; i < manager->num_gpus; i++) {
        GPUContext* ctx = &manager->contexts[i];
        
        if (ctx->is_active) {
            cudaSetDevice(ctx->device_id);
            cudaStreamDestroy(ctx->stream);
            ncclCommDestroy(ctx->nccl_comm);
            cudaFree(ctx->device_memory);
        }
    }
    
    for (size_t i = 0; i < PIPELINE_DEPTH; i++) {
        cudaEventDestroy(manager->events[i]);
    }
    
    free(manager);
#else
    // CPU fallback cleanup
    if (!manager) return;

    // Free allocated host memory
    if (manager->context.host_memory) {
        free(manager->context.host_memory);
        manager->context.host_memory = NULL;
    }

    free(manager);
#endif
}
