/**
 * compute_cuda.cu - CUDA backend implementation
 *
 * This backend provides GPU-accelerated compute using:
 * - CUDA kernels for quantum operations
 * - cuBLAS for dense linear algebra
 * - NCCL for multi-GPU communication (optional)
 * - MPI for multi-node communication (optional)
 *
 * Priority: 100 (highest, preferred over other backends when available)
 */

#include "quantum_geometric/supercomputer/compute_backend.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#if COMPUTE_HAS_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuComplex.h>

#if COMPUTE_HAS_NCCL
#include <nccl.h>
#endif

#if COMPUTE_HAS_MPI
#include <mpi.h>
#endif

// ============================================================================
// Constants
// ============================================================================

#define CUDA_BLOCK_SIZE 256
#define CUDA_TILE_SIZE 16
#define CUDA_WARP_SIZE 32
#define CUDA_MAX_STREAMS 8
#define CUDA_MIN_SIZE_FOR_GPU 1024

// ============================================================================
// Error Handling Macros
// ============================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        return COMPUTE_ERROR_KERNEL_FAILED; \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = (call); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
        return COMPUTE_ERROR_KERNEL_FAILED; \
    } \
} while(0)

// ============================================================================
// CUDA Backend Context
// ============================================================================

typedef struct CUDABackendContext {
    // Device info
    int device_id;
    int num_devices;
    cudaDeviceProp device_props;

    // cuBLAS handle
    cublasHandle_t cublas_handle;

    // Streams for async operations
    cudaStream_t streams[CUDA_MAX_STREAMS];
    int num_streams;
    int current_stream;

    // Memory pool
    cudaMemPool_t mem_pool;
    bool pool_supported;

    // MPI state
    int node_rank;
    int num_nodes;
#if COMPUTE_HAS_MPI
    MPI_Comm comm;
    bool mpi_initialized_by_us;
#endif

#if COMPUTE_HAS_NCCL
    ncclComm_t nccl_comm;
    bool nccl_initialized;
#endif

    // Memory tracking
    size_t total_allocated;
    size_t peak_allocated;

    // Performance metrics
    ComputeMetrics metrics;

} CUDABackendContext;

// ============================================================================
// CUDA Kernels - Quantum Operations
// ============================================================================

// Warp-level reduction for float
__device__ __forceinline__ float warpReduceSumFloat(float val) {
    for (int offset = CUDA_WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction for float
__device__ float blockReduceSumFloat(float val) {
    __shared__ float shared[32];

    int lane = threadIdx.x % CUDA_WARP_SIZE;
    int wid = threadIdx.x / CUDA_WARP_SIZE;

    val = warpReduceSumFloat(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    int num_warps = (blockDim.x + CUDA_WARP_SIZE - 1) / CUDA_WARP_SIZE;
    val = (threadIdx.x < num_warps) ? shared[lane] : 0.0f;

    if (wid == 0) val = warpReduceSumFloat(val);

    return val;
}

// Quantum state normalization kernel
__global__ void cuda_quantum_normalize_kernel(float* state, size_t size, float* norm_squared) {
    __shared__ float shared_sum[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float local_sum = 0.0f;
    if (idx < size) {
        float real = state[2 * idx];
        float imag = state[2 * idx + 1];
        local_sum = real * real + imag * imag;
    }

    shared_sum[tid] = local_sum;
    __syncthreads();

    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(norm_squared, shared_sum[0]);
    }
}

// Scale state kernel
__global__ void cuda_scale_kernel(float* state, size_t size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        state[2 * idx] *= scale;
        state[2 * idx + 1] *= scale;
    }
}

// Inner product kernel: <a|b> = sum(conj(a) * b)
__global__ void cuda_inner_product_kernel(const float* state_a, const float* state_b,
                                           float* partial_real, float* partial_imag,
                                           size_t size) {
    __shared__ float shared_real[256];
    __shared__ float shared_imag[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float local_real = 0.0f;
    float local_imag = 0.0f;

    if (idx < size) {
        float ar = state_a[2 * idx];
        float ai = state_a[2 * idx + 1];
        float br = state_b[2 * idx];
        float bi = state_b[2 * idx + 1];

        // conj(a) * b = (ar - i*ai) * (br + i*bi) = (ar*br + ai*bi) + i*(ar*bi - ai*br)
        local_real = ar * br + ai * bi;
        local_imag = ar * bi - ai * br;
    }

    shared_real[tid] = local_real;
    shared_imag[tid] = local_imag;
    __syncthreads();

    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_real[tid] += shared_real[tid + s];
            shared_imag[tid] += shared_imag[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_real[blockIdx.x] = shared_real[0];
        partial_imag[blockIdx.x] = shared_imag[0];
    }
}

// Expectation value kernel (diagonal observable)
__global__ void cuda_expectation_kernel(const float* state, const float* observable,
                                         float* partial_sum, size_t size) {
    __shared__ float shared_sum[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float local_sum = 0.0f;
    if (idx < size) {
        float real = state[2 * idx];
        float imag = state[2 * idx + 1];
        float prob = real * real + imag * imag;
        local_sum = prob * observable[idx];
    }

    shared_sum[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sum[blockIdx.x] = shared_sum[0];
    }
}

// Gradient kernel (parameter-shift gradient approximation)
__global__ void cuda_gradient_kernel(const float* forward_state,
                                      const float* backward_state,
                                      float* gradients,
                                      size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float fr = forward_state[2 * idx];
        float fi = forward_state[2 * idx + 1];
        float br = backward_state[2 * idx];
        float bi = backward_state[2 * idx + 1];

        // Gradient is real part of <backward|forward>
        gradients[2 * idx] = br * fr + bi * fi;
        gradients[2 * idx + 1] = br * fi - bi * fr;
    }
}

// Complex matrix-vector multiplication kernel
__global__ void cuda_complex_matvec_kernel(const float* matrix, const float* vector,
                                            float* output, size_t n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n) {
        float sum_real = 0.0f;
        float sum_imag = 0.0f;

        for (size_t col = 0; col < n; col++) {
            // Matrix is stored row-major, complex interleaved
            float mr = matrix[2 * (row * n + col)];
            float mi = matrix[2 * (row * n + col) + 1];
            float vr = vector[2 * col];
            float vi = vector[2 * col + 1];

            // Complex multiply-add
            sum_real += mr * vr - mi * vi;
            sum_imag += mr * vi + mi * vr;
        }

        output[2 * row] = sum_real;
        output[2 * row + 1] = sum_imag;
    }
}

// Tiled complex matrix multiplication kernel
__global__ void cuda_complex_matmul_kernel(const float* A, const float* B, float* C,
                                            size_t M, size_t N, size_t K) {
    __shared__ float As_real[CUDA_TILE_SIZE][CUDA_TILE_SIZE + 1];
    __shared__ float As_imag[CUDA_TILE_SIZE][CUDA_TILE_SIZE + 1];
    __shared__ float Bs_real[CUDA_TILE_SIZE][CUDA_TILE_SIZE + 1];
    __shared__ float Bs_imag[CUDA_TILE_SIZE][CUDA_TILE_SIZE + 1];

    int bx = blockIdx.x * CUDA_TILE_SIZE;
    int by = blockIdx.y * CUDA_TILE_SIZE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = bx + tx;
    int col = by + ty;

    float sum_real = 0.0f;
    float sum_imag = 0.0f;

    for (int tile = 0; tile < (K + CUDA_TILE_SIZE - 1) / CUDA_TILE_SIZE; tile++) {
        int k_offset = tile * CUDA_TILE_SIZE;

        // Load tiles into shared memory
        if (row < M && k_offset + ty < K) {
            int a_idx = row * K + k_offset + ty;
            As_real[tx][ty] = A[2 * a_idx];
            As_imag[tx][ty] = A[2 * a_idx + 1];
        } else {
            As_real[tx][ty] = 0.0f;
            As_imag[tx][ty] = 0.0f;
        }

        if (k_offset + tx < K && col < N) {
            int b_idx = (k_offset + tx) * N + col;
            Bs_real[tx][ty] = B[2 * b_idx];
            Bs_imag[tx][ty] = B[2 * b_idx + 1];
        } else {
            Bs_real[tx][ty] = 0.0f;
            Bs_imag[tx][ty] = 0.0f;
        }

        __syncthreads();

        // Compute partial sum for this tile
        #pragma unroll
        for (int k = 0; k < CUDA_TILE_SIZE; k++) {
            float ar = As_real[tx][k];
            float ai = As_imag[tx][k];
            float br = Bs_real[k][ty];
            float bi = Bs_imag[k][ty];

            sum_real += ar * br - ai * bi;
            sum_imag += ar * bi + ai * br;
        }

        __syncthreads();
    }

    // Store result
    if (row < M && col < N) {
        int c_idx = row * N + col;
        C[2 * c_idx] = sum_real;
        C[2 * c_idx + 1] = sum_imag;
    }
}

// ============================================================================
// Lifecycle Operations
// ============================================================================

static ComputeBackend* cuda_init(const ComputeDistributedConfig* config) {
    CUDABackendContext* ctx = (CUDABackendContext*)calloc(1, sizeof(CUDABackendContext));
    if (!ctx) return NULL;

    cudaError_t err;

    // Get device count
    err = cudaGetDeviceCount(&ctx->num_devices);
    if (err != cudaSuccess || ctx->num_devices == 0) {
        free(ctx);
        return NULL;
    }

    // Select device
    ctx->device_id = config->local_rank % ctx->num_devices;
    err = cudaSetDevice(ctx->device_id);
    if (err != cudaSuccess) {
        free(ctx);
        return NULL;
    }

    // Get device properties
    cudaGetDeviceProperties(&ctx->device_props, ctx->device_id);

    // Create cuBLAS handle
    cublasStatus_t cublas_status = cublasCreate(&ctx->cublas_handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        free(ctx);
        return NULL;
    }

    // Enable tensor cores if available
    cublasSetMathMode(ctx->cublas_handle, CUBLAS_TENSOR_OP_MATH);
    cublasSetAtomicsMode(ctx->cublas_handle, CUBLAS_ATOMICS_ALLOWED);

    // Create streams
    ctx->num_streams = config->num_streams > 0 ?
                       (config->num_streams < CUDA_MAX_STREAMS ? config->num_streams : CUDA_MAX_STREAMS) : 4;
    for (int i = 0; i < ctx->num_streams; i++) {
        cudaStreamCreateWithFlags(&ctx->streams[i], cudaStreamNonBlocking);
    }
    ctx->current_stream = 0;

    // Try to create memory pool (CUDA 11.2+)
    ctx->pool_supported = false;
    cudaMemPoolProps pool_props = {};
    pool_props.allocType = cudaMemAllocationTypePinned;
    pool_props.handleTypes = cudaMemHandleTypeNone;
    pool_props.location.type = cudaMemLocationTypeDevice;
    pool_props.location.id = ctx->device_id;

    err = cudaMemPoolCreate(&ctx->mem_pool, &pool_props);
    if (err == cudaSuccess) {
        ctx->pool_supported = true;
    }

    // Initialize MPI if needed
#if COMPUTE_HAS_MPI
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);

    if (!mpi_initialized && config->num_nodes > 1) {
        int provided;
        MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);
        ctx->mpi_initialized_by_us = true;
    }

    if (mpi_initialized || ctx->mpi_initialized_by_us) {
        MPI_Comm_dup(MPI_COMM_WORLD, &ctx->comm);
        MPI_Comm_rank(ctx->comm, &ctx->node_rank);
        MPI_Comm_size(ctx->comm, &ctx->num_nodes);
    } else {
        ctx->node_rank = 0;
        ctx->num_nodes = 1;
    }
#else
    ctx->node_rank = 0;
    ctx->num_nodes = 1;
#endif

    // Initialize NCCL if available
#if COMPUTE_HAS_NCCL
    if (ctx->num_devices > 1 || ctx->num_nodes > 1) {
        ncclUniqueId nccl_id;
        if (ctx->node_rank == 0) {
            ncclGetUniqueId(&nccl_id);
        }
#if COMPUTE_HAS_MPI
        MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, ctx->comm);
#endif
        ncclResult_t nccl_result = ncclCommInitRank(&ctx->nccl_comm,
                                                     ctx->num_nodes * ctx->num_devices,
                                                     nccl_id,
                                                     ctx->node_rank * ctx->num_devices + ctx->device_id);
        ctx->nccl_initialized = (nccl_result == ncclSuccess);
    }
#endif

    return (ComputeBackend*)ctx;
}

static void cuda_cleanup(ComputeBackend* backend) {
    CUDABackendContext* ctx = (CUDABackendContext*)backend;
    if (!ctx) return;

#if COMPUTE_HAS_NCCL
    if (ctx->nccl_initialized) {
        ncclCommDestroy(ctx->nccl_comm);
    }
#endif

    // Destroy streams
    for (int i = 0; i < ctx->num_streams; i++) {
        cudaStreamDestroy(ctx->streams[i]);
    }

    // Destroy memory pool
    if (ctx->pool_supported) {
        cudaMemPoolDestroy(ctx->mem_pool);
    }

    // Destroy cuBLAS handle
    if (ctx->cublas_handle) {
        cublasDestroy(ctx->cublas_handle);
    }

#if COMPUTE_HAS_MPI
    if (ctx->comm != MPI_COMM_NULL) {
        MPI_Comm_free(&ctx->comm);
    }
    if (ctx->mpi_initialized_by_us) {
        MPI_Finalize();
    }
#endif

    free(ctx);
}

static bool cuda_probe(void) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
}

static ComputeResult cuda_get_capabilities(ComputeBackend* backend,
                                            int* num_devices,
                                            size_t* total_memory) {
    CUDABackendContext* ctx = (CUDABackendContext*)backend;
    if (!ctx) return COMPUTE_ERROR_INVALID_ARGUMENT;

    if (num_devices) *num_devices = ctx->num_devices;
    if (total_memory) *total_memory = ctx->device_props.totalGlobalMem;

    return COMPUTE_SUCCESS;
}

// ============================================================================
// Memory Management
// ============================================================================

static void* cuda_alloc(ComputeBackend* backend, size_t size, ComputeMemType mem_type) {
    CUDABackendContext* ctx = (CUDABackendContext*)backend;
    void* ptr = NULL;
    cudaError_t err;

    switch (mem_type) {
        case COMPUTE_MEM_HOST:
            ptr = malloc(size);
            break;

        case COMPUTE_MEM_DEVICE:
            err = cudaMalloc(&ptr, size);
            if (err != cudaSuccess) ptr = NULL;
            break;

        case COMPUTE_MEM_UNIFIED:
            err = cudaMallocManaged(&ptr, size);
            if (err != cudaSuccess) ptr = NULL;
            break;

        case COMPUTE_MEM_PINNED:
            err = cudaMallocHost(&ptr, size);
            if (err != cudaSuccess) ptr = NULL;
            break;
    }

    if (ptr && ctx) {
        ctx->total_allocated += size;
        if (ctx->total_allocated > ctx->peak_allocated) {
            ctx->peak_allocated = ctx->total_allocated;
        }
    }

    return ptr;
}

static void cuda_free(ComputeBackend* backend, void* ptr, ComputeMemType mem_type) {
    (void)backend;
    if (!ptr) return;

    switch (mem_type) {
        case COMPUTE_MEM_HOST:
            free(ptr);
            break;

        case COMPUTE_MEM_DEVICE:
        case COMPUTE_MEM_UNIFIED:
            cudaFree(ptr);
            break;

        case COMPUTE_MEM_PINNED:
            cudaFreeHost(ptr);
            break;
    }
}

static ComputeResult cuda_memcpy(ComputeBackend* backend,
                                  void* dst, ComputeMemType dst_type,
                                  const void* src, ComputeMemType src_type,
                                  size_t size, ComputeStream* stream) {
    CUDABackendContext* ctx = (CUDABackendContext*)backend;
    cudaMemcpyKind kind;

    if (src_type == COMPUTE_MEM_HOST && dst_type == COMPUTE_MEM_DEVICE) {
        kind = cudaMemcpyHostToDevice;
    } else if (src_type == COMPUTE_MEM_DEVICE && dst_type == COMPUTE_MEM_HOST) {
        kind = cudaMemcpyDeviceToHost;
    } else if (src_type == COMPUTE_MEM_DEVICE && dst_type == COMPUTE_MEM_DEVICE) {
        kind = cudaMemcpyDeviceToDevice;
    } else {
        kind = cudaMemcpyDefault;
    }

    cudaStream_t cuda_stream = stream ? (cudaStream_t)stream : ctx->streams[ctx->current_stream];
    cudaError_t err = cudaMemcpyAsync(dst, src, size, kind, cuda_stream);

    return (err == cudaSuccess) ? COMPUTE_SUCCESS : COMPUTE_ERROR_KERNEL_FAILED;
}

static ComputeResult cuda_memset(ComputeBackend* backend,
                                  void* ptr, int value, size_t size,
                                  ComputeStream* stream) {
    CUDABackendContext* ctx = (CUDABackendContext*)backend;
    cudaStream_t cuda_stream = stream ? (cudaStream_t)stream : ctx->streams[ctx->current_stream];

    cudaError_t err = cudaMemsetAsync(ptr, value, size, cuda_stream);
    return (err == cudaSuccess) ? COMPUTE_SUCCESS : COMPUTE_ERROR_KERNEL_FAILED;
}

// ============================================================================
// Stream Management
// ============================================================================

static ComputeStream* cuda_create_stream(ComputeBackend* backend) {
    (void)backend;
    cudaStream_t stream;
    cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    return (err == cudaSuccess) ? (ComputeStream*)stream : NULL;
}

static void cuda_destroy_stream(ComputeBackend* backend, ComputeStream* stream) {
    (void)backend;
    if (stream) {
        cudaStreamDestroy((cudaStream_t)stream);
    }
}

static ComputeResult cuda_synchronize_stream(ComputeBackend* backend, ComputeStream* stream) {
    CUDABackendContext* ctx = (CUDABackendContext*)backend;
    cudaError_t err;

    if (stream) {
        err = cudaStreamSynchronize((cudaStream_t)stream);
    } else {
        // Synchronize all streams
        for (int i = 0; i < ctx->num_streams; i++) {
            err = cudaStreamSynchronize(ctx->streams[i]);
            if (err != cudaSuccess) break;
        }
    }

    return (err == cudaSuccess) ? COMPUTE_SUCCESS : COMPUTE_ERROR_SYNCHRONIZATION_FAILED;
}

static ComputeEvent* cuda_create_event(ComputeBackend* backend) {
    (void)backend;
    cudaEvent_t event;
    cudaError_t err = cudaEventCreate(&event);
    return (err == cudaSuccess) ? (ComputeEvent*)event : NULL;
}

static void cuda_destroy_event(ComputeBackend* backend, ComputeEvent* event) {
    (void)backend;
    if (event) {
        cudaEventDestroy((cudaEvent_t)event);
    }
}

static ComputeResult cuda_record_event(ComputeBackend* backend,
                                        ComputeEvent* event,
                                        ComputeStream* stream) {
    CUDABackendContext* ctx = (CUDABackendContext*)backend;
    cudaStream_t cuda_stream = stream ? (cudaStream_t)stream : ctx->streams[ctx->current_stream];

    cudaError_t err = cudaEventRecord((cudaEvent_t)event, cuda_stream);
    return (err == cudaSuccess) ? COMPUTE_SUCCESS : COMPUTE_ERROR_KERNEL_FAILED;
}

static ComputeResult cuda_wait_event(ComputeBackend* backend,
                                      ComputeStream* stream,
                                      ComputeEvent* event) {
    CUDABackendContext* ctx = (CUDABackendContext*)backend;
    cudaStream_t cuda_stream = stream ? (cudaStream_t)stream : ctx->streams[ctx->current_stream];

    cudaError_t err = cudaStreamWaitEvent(cuda_stream, (cudaEvent_t)event, 0);
    return (err == cudaSuccess) ? COMPUTE_SUCCESS : COMPUTE_ERROR_SYNCHRONIZATION_FAILED;
}

// ============================================================================
// Quantum Operations
// ============================================================================

static ComputeResult cuda_quantum_unitary(ComputeBackend* backend,
                                           float* state, size_t state_size,
                                           const float* unitary, size_t unitary_size,
                                           ComputeStream* stream) {
    CUDABackendContext* ctx = (CUDABackendContext*)backend;
    if (!ctx || !state || !unitary || state_size == 0) {
        return COMPUTE_ERROR_INVALID_ARGUMENT;
    }

    cudaStream_t cuda_stream = stream ? (cudaStream_t)stream : ctx->streams[ctx->current_stream];
    size_t bytes = 2 * state_size * sizeof(float);
    size_t unitary_bytes = 2 * unitary_size * unitary_size * sizeof(float);

    // Allocate device memory
    float *d_state, *d_unitary, *d_output;
    cudaMalloc(&d_state, bytes);
    cudaMalloc(&d_unitary, unitary_bytes);
    cudaMalloc(&d_output, bytes);

    // Copy data to device
    cudaMemcpyAsync(d_state, state, bytes, cudaMemcpyHostToDevice, cuda_stream);
    cudaMemcpyAsync(d_unitary, unitary, unitary_bytes, cudaMemcpyHostToDevice, cuda_stream);

    // Use cuBLAS for matrix-vector multiplication
    cuComplex alpha = make_cuComplex(1.0f, 0.0f);
    cuComplex beta = make_cuComplex(0.0f, 0.0f);

    cublasSetStream(ctx->cublas_handle, cuda_stream);
    cublasStatus_t status = cublasCgemv(ctx->cublas_handle, CUBLAS_OP_N,
                                         (int)state_size, (int)unitary_size,
                                         &alpha,
                                         (cuComplex*)d_unitary, (int)state_size,
                                         (cuComplex*)d_state, 1,
                                         &beta,
                                         (cuComplex*)d_output, 1);

    // Copy result back
    cudaMemcpyAsync(state, d_output, bytes, cudaMemcpyDeviceToHost, cuda_stream);
    cudaStreamSynchronize(cuda_stream);

    // Cleanup
    cudaFree(d_state);
    cudaFree(d_unitary);
    cudaFree(d_output);

    return (status == CUBLAS_STATUS_SUCCESS) ? COMPUTE_SUCCESS : COMPUTE_ERROR_KERNEL_FAILED;
}

static ComputeResult cuda_quantum_normalize(ComputeBackend* backend,
                                             float* state, size_t size,
                                             ComputeStream* stream) {
    CUDABackendContext* ctx = (CUDABackendContext*)backend;
    if (!ctx || !state || size == 0) {
        return COMPUTE_ERROR_INVALID_ARGUMENT;
    }

    cudaStream_t cuda_stream = stream ? (cudaStream_t)stream : ctx->streams[ctx->current_stream];
    size_t bytes = 2 * size * sizeof(float);

    // For small sizes, use CPU
    if (size < CUDA_MIN_SIZE_FOR_GPU) {
        float norm_sq = 0.0f;
        for (size_t i = 0; i < size; i++) {
            norm_sq += state[2*i] * state[2*i] + state[2*i+1] * state[2*i+1];
        }
        float norm = sqrtf(norm_sq);
        if (norm > 1e-10f) {
            float scale = 1.0f / norm;
            for (size_t i = 0; i < size; i++) {
                state[2*i] *= scale;
                state[2*i+1] *= scale;
            }
        }
        return COMPUTE_SUCCESS;
    }

    // Allocate device memory
    float *d_state, *d_norm_sq;
    cudaMalloc(&d_state, bytes);
    cudaMalloc(&d_norm_sq, sizeof(float));
    cudaMemset(d_norm_sq, 0, sizeof(float));

    // Copy state to device
    cudaMemcpyAsync(d_state, state, bytes, cudaMemcpyHostToDevice, cuda_stream);

    // Compute norm squared
    int blocks = (size + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    cuda_quantum_normalize_kernel<<<blocks, CUDA_BLOCK_SIZE, 0, cuda_stream>>>(
        d_state, size, d_norm_sq);

    // Get norm and scale
    float norm_sq;
    cudaMemcpy(&norm_sq, d_norm_sq, sizeof(float), cudaMemcpyDeviceToHost);

    float norm = sqrtf(norm_sq);
    if (norm > 1e-10f) {
        float scale = 1.0f / norm;
        cuda_scale_kernel<<<blocks, CUDA_BLOCK_SIZE, 0, cuda_stream>>>(d_state, size, scale);
    }

    // Copy back
    cudaMemcpyAsync(state, d_state, bytes, cudaMemcpyDeviceToHost, cuda_stream);
    cudaStreamSynchronize(cuda_stream);

    cudaFree(d_state);
    cudaFree(d_norm_sq);

    return COMPUTE_SUCCESS;
}

static ComputeResult cuda_quantum_tensor_contract(ComputeBackend* backend,
                                                   float* result,
                                                   const float* a, const float* b,
                                                   size_t m, size_t n, size_t k,
                                                   ComputeStream* stream) {
    CUDABackendContext* ctx = (CUDABackendContext*)backend;
    if (!ctx || !result || !a || !b) {
        return COMPUTE_ERROR_INVALID_ARGUMENT;
    }

    cudaStream_t cuda_stream = stream ? (cudaStream_t)stream : ctx->streams[ctx->current_stream];

    size_t a_bytes = 2 * m * k * sizeof(float);
    size_t b_bytes = 2 * k * n * sizeof(float);
    size_t c_bytes = 2 * m * n * sizeof(float);

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, a_bytes);
    cudaMalloc(&d_b, b_bytes);
    cudaMalloc(&d_c, c_bytes);

    // Copy data to device
    cudaMemcpyAsync(d_a, a, a_bytes, cudaMemcpyHostToDevice, cuda_stream);
    cudaMemcpyAsync(d_b, b, b_bytes, cudaMemcpyHostToDevice, cuda_stream);

    // Use cuBLAS for large matrices
    if (m >= 64 && n >= 64 && k >= 64) {
        cuComplex alpha = make_cuComplex(1.0f, 0.0f);
        cuComplex beta = make_cuComplex(0.0f, 0.0f);

        cublasSetStream(ctx->cublas_handle, cuda_stream);
        cublasStatus_t status = cublasCgemm(ctx->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                             (int)n, (int)m, (int)k,
                                             &alpha,
                                             (cuComplex*)d_b, (int)n,
                                             (cuComplex*)d_a, (int)k,
                                             &beta,
                                             (cuComplex*)d_c, (int)n);

        if (status != CUBLAS_STATUS_SUCCESS) {
            cudaFree(d_a);
            cudaFree(d_b);
            cudaFree(d_c);
            return COMPUTE_ERROR_KERNEL_FAILED;
        }
    } else {
        // Use custom kernel for smaller matrices
        dim3 block(CUDA_TILE_SIZE, CUDA_TILE_SIZE);
        dim3 grid((m + CUDA_TILE_SIZE - 1) / CUDA_TILE_SIZE,
                  (n + CUDA_TILE_SIZE - 1) / CUDA_TILE_SIZE);

        cuda_complex_matmul_kernel<<<grid, block, 0, cuda_stream>>>(d_a, d_b, d_c, m, n, k);
    }

    // Copy result back
    cudaMemcpyAsync(result, d_c, c_bytes, cudaMemcpyDeviceToHost, cuda_stream);
    cudaStreamSynchronize(cuda_stream);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return COMPUTE_SUCCESS;
}

static ComputeResult cuda_quantum_gradient(ComputeBackend* backend,
                                            float* gradients,
                                            const float* forward_state,
                                            const float* backward_state,
                                            size_t size,
                                            ComputeStream* stream) {
    CUDABackendContext* ctx = (CUDABackendContext*)backend;
    if (!ctx || !gradients || !forward_state || !backward_state) {
        return COMPUTE_ERROR_INVALID_ARGUMENT;
    }

    cudaStream_t cuda_stream = stream ? (cudaStream_t)stream : ctx->streams[ctx->current_stream];
    size_t bytes = 2 * size * sizeof(float);

    // For small sizes, use CPU
    if (size < CUDA_MIN_SIZE_FOR_GPU) {
        float real_sum = 0.0f, imag_sum = 0.0f;
        for (size_t i = 0; i < size; i++) {
            float br = backward_state[2*i];
            float bi = backward_state[2*i+1];
            float fr = forward_state[2*i];
            float fi = forward_state[2*i+1];
            real_sum += br * fr + bi * fi;
            imag_sum += br * fi - bi * fr;
        }
        gradients[0] = real_sum;
        gradients[1] = imag_sum;
        return COMPUTE_SUCCESS;
    }

    float *d_fwd, *d_bwd, *d_grad;
    cudaMalloc(&d_fwd, bytes);
    cudaMalloc(&d_bwd, bytes);
    cudaMalloc(&d_grad, bytes);

    cudaMemcpyAsync(d_fwd, forward_state, bytes, cudaMemcpyHostToDevice, cuda_stream);
    cudaMemcpyAsync(d_bwd, backward_state, bytes, cudaMemcpyHostToDevice, cuda_stream);

    int blocks = (size + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    cuda_gradient_kernel<<<blocks, CUDA_BLOCK_SIZE, 0, cuda_stream>>>(d_fwd, d_bwd, d_grad, size);

    cudaMemcpyAsync(gradients, d_grad, bytes, cudaMemcpyDeviceToHost, cuda_stream);
    cudaStreamSynchronize(cuda_stream);

    cudaFree(d_fwd);
    cudaFree(d_bwd);
    cudaFree(d_grad);

    return COMPUTE_SUCCESS;
}

static ComputeResult cuda_quantum_inner_product(ComputeBackend* backend,
                                                 float* result,
                                                 const float* state_a,
                                                 const float* state_b,
                                                 size_t size,
                                                 ComputeStream* stream) {
    CUDABackendContext* ctx = (CUDABackendContext*)backend;
    if (!ctx || !result || !state_a || !state_b) {
        return COMPUTE_ERROR_INVALID_ARGUMENT;
    }

    cudaStream_t cuda_stream = stream ? (cudaStream_t)stream : ctx->streams[ctx->current_stream];

    // For small sizes, use CPU
    if (size < CUDA_MIN_SIZE_FOR_GPU) {
        float real_sum = 0.0f, imag_sum = 0.0f;
        for (size_t i = 0; i < size; i++) {
            float ar = state_a[2*i], ai = state_a[2*i+1];
            float br = state_b[2*i], bi = state_b[2*i+1];
            real_sum += ar * br + ai * bi;
            imag_sum += ar * bi - ai * br;
        }
        result[0] = real_sum;
        result[1] = imag_sum;
        return COMPUTE_SUCCESS;
    }

    size_t bytes = 2 * size * sizeof(float);
    int blocks = (size + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;

    float *d_a, *d_b, *d_partial_real, *d_partial_imag;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_partial_real, blocks * sizeof(float));
    cudaMalloc(&d_partial_imag, blocks * sizeof(float));

    cudaMemcpyAsync(d_a, state_a, bytes, cudaMemcpyHostToDevice, cuda_stream);
    cudaMemcpyAsync(d_b, state_b, bytes, cudaMemcpyHostToDevice, cuda_stream);

    cuda_inner_product_kernel<<<blocks, CUDA_BLOCK_SIZE, 0, cuda_stream>>>(
        d_a, d_b, d_partial_real, d_partial_imag, size);

    // Sum partial results on CPU
    float* h_partial_real = (float*)malloc(blocks * sizeof(float));
    float* h_partial_imag = (float*)malloc(blocks * sizeof(float));

    cudaMemcpy(h_partial_real, d_partial_real, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_partial_imag, d_partial_imag, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float real_sum = 0.0f, imag_sum = 0.0f;
    for (int i = 0; i < blocks; i++) {
        real_sum += h_partial_real[i];
        imag_sum += h_partial_imag[i];
    }
    result[0] = real_sum;
    result[1] = imag_sum;

    free(h_partial_real);
    free(h_partial_imag);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_partial_real);
    cudaFree(d_partial_imag);

    return COMPUTE_SUCCESS;
}

static ComputeResult cuda_quantum_expectation(ComputeBackend* backend,
                                               float* result,
                                               const float* state,
                                               const float* observable,
                                               size_t size,
                                               ComputeStream* stream) {
    CUDABackendContext* ctx = (CUDABackendContext*)backend;
    if (!ctx || !result || !state || !observable) {
        return COMPUTE_ERROR_INVALID_ARGUMENT;
    }

    cudaStream_t cuda_stream = stream ? (cudaStream_t)stream : ctx->streams[ctx->current_stream];

    // For small sizes, use CPU
    if (size < CUDA_MIN_SIZE_FOR_GPU) {
        float sum = 0.0f;
        for (size_t i = 0; i < size; i++) {
            float real = state[2*i], imag = state[2*i+1];
            float prob = real * real + imag * imag;
            sum += prob * observable[i];
        }
        *result = sum;
        return COMPUTE_SUCCESS;
    }

    size_t state_bytes = 2 * size * sizeof(float);
    size_t obs_bytes = size * sizeof(float);
    int blocks = (size + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;

    float *d_state, *d_obs, *d_partial;
    cudaMalloc(&d_state, state_bytes);
    cudaMalloc(&d_obs, obs_bytes);
    cudaMalloc(&d_partial, blocks * sizeof(float));

    cudaMemcpyAsync(d_state, state, state_bytes, cudaMemcpyHostToDevice, cuda_stream);
    cudaMemcpyAsync(d_obs, observable, obs_bytes, cudaMemcpyHostToDevice, cuda_stream);

    cuda_expectation_kernel<<<blocks, CUDA_BLOCK_SIZE, 0, cuda_stream>>>(
        d_state, d_obs, d_partial, size);

    float* h_partial = (float*)malloc(blocks * sizeof(float));
    cudaMemcpy(h_partial, d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float sum = 0.0f;
    for (int i = 0; i < blocks; i++) {
        sum += h_partial[i];
    }
    *result = sum;

    free(h_partial);
    cudaFree(d_state);
    cudaFree(d_obs);
    cudaFree(d_partial);

    return COMPUTE_SUCCESS;
}

// ============================================================================
// Collective Communication
// ============================================================================

static ComputeResult cuda_barrier(ComputeBackend* backend) {
#if COMPUTE_HAS_MPI
    CUDABackendContext* ctx = (CUDABackendContext*)backend;
    if (ctx && ctx->num_nodes > 1) {
        MPI_Barrier(ctx->comm);
    }
#else
    (void)backend;
#endif
    return COMPUTE_SUCCESS;
}

static ComputeResult cuda_broadcast(ComputeBackend* backend,
                                     void* data, size_t size,
                                     ComputeDataType dtype, int root) {
#if COMPUTE_HAS_NCCL
    CUDABackendContext* ctx = (CUDABackendContext*)backend;
    if (ctx && ctx->nccl_initialized) {
        ncclDataType_t nccl_type;
        switch (dtype) {
            case COMPUTE_DTYPE_FLOAT32:   nccl_type = ncclFloat32; break;
            case COMPUTE_DTYPE_FLOAT64:   nccl_type = ncclFloat64; break;
            case COMPUTE_DTYPE_INT32:     nccl_type = ncclInt32; break;
            case COMPUTE_DTYPE_INT64:     nccl_type = ncclInt64; break;
            default:                      nccl_type = ncclChar; break;
        }
        ncclBroadcast(data, data, size, nccl_type, root, ctx->nccl_comm, ctx->streams[0]);
        cudaStreamSynchronize(ctx->streams[0]);
        return COMPUTE_SUCCESS;
    }
#endif

#if COMPUTE_HAS_MPI
    CUDABackendContext* ctx = (CUDABackendContext*)backend;
    if (ctx && ctx->num_nodes > 1) {
        MPI_Datatype mpi_type;
        switch (dtype) {
            case COMPUTE_DTYPE_FLOAT32:   mpi_type = MPI_FLOAT; break;
            case COMPUTE_DTYPE_FLOAT64:   mpi_type = MPI_DOUBLE; break;
            case COMPUTE_DTYPE_INT32:     mpi_type = MPI_INT; break;
            case COMPUTE_DTYPE_INT64:     mpi_type = MPI_LONG_LONG; break;
            default:                      mpi_type = MPI_BYTE; break;
        }
        MPI_Bcast(data, (int)size, mpi_type, root, ctx->comm);
    }
#else
    (void)backend;
    (void)data;
    (void)size;
    (void)dtype;
    (void)root;
#endif
    return COMPUTE_SUCCESS;
}

static ComputeResult cuda_allreduce(ComputeBackend* backend,
                                     const void* send_data, void* recv_data,
                                     size_t count, ComputeDataType dtype,
                                     ComputeReduceOp op) {
#if COMPUTE_HAS_NCCL
    CUDABackendContext* ctx = (CUDABackendContext*)backend;
    if (ctx && ctx->nccl_initialized) {
        ncclDataType_t nccl_type;
        ncclRedOp_t nccl_op;

        switch (dtype) {
            case COMPUTE_DTYPE_FLOAT32:   nccl_type = ncclFloat32; break;
            case COMPUTE_DTYPE_FLOAT64:   nccl_type = ncclFloat64; break;
            case COMPUTE_DTYPE_INT32:     nccl_type = ncclInt32; break;
            case COMPUTE_DTYPE_INT64:     nccl_type = ncclInt64; break;
            default:                      nccl_type = ncclChar; break;
        }

        switch (op) {
            case COMPUTE_REDUCE_SUM:  nccl_op = ncclSum; break;
            case COMPUTE_REDUCE_PROD: nccl_op = ncclProd; break;
            case COMPUTE_REDUCE_MIN:  nccl_op = ncclMin; break;
            case COMPUTE_REDUCE_MAX:  nccl_op = ncclMax; break;
            default:                  nccl_op = ncclSum; break;
        }

        ncclAllReduce(send_data, recv_data, count, nccl_type, nccl_op, ctx->nccl_comm, ctx->streams[0]);
        cudaStreamSynchronize(ctx->streams[0]);
        return COMPUTE_SUCCESS;
    }
#endif

#if COMPUTE_HAS_MPI
    CUDABackendContext* ctx = (CUDABackendContext*)backend;
    if (ctx && ctx->num_nodes > 1) {
        MPI_Datatype mpi_type;
        MPI_Op mpi_op;

        switch (dtype) {
            case COMPUTE_DTYPE_FLOAT32:   mpi_type = MPI_FLOAT; break;
            case COMPUTE_DTYPE_FLOAT64:   mpi_type = MPI_DOUBLE; break;
            case COMPUTE_DTYPE_INT32:     mpi_type = MPI_INT; break;
            case COMPUTE_DTYPE_INT64:     mpi_type = MPI_LONG_LONG; break;
            default:                      mpi_type = MPI_BYTE; break;
        }

        switch (op) {
            case COMPUTE_REDUCE_SUM:  mpi_op = MPI_SUM; break;
            case COMPUTE_REDUCE_PROD: mpi_op = MPI_PROD; break;
            case COMPUTE_REDUCE_MIN:  mpi_op = MPI_MIN; break;
            case COMPUTE_REDUCE_MAX:  mpi_op = MPI_MAX; break;
            default:                  mpi_op = MPI_SUM; break;
        }

        MPI_Allreduce(send_data, recv_data, (int)count, mpi_type, mpi_op, ctx->comm);
    } else
#endif
    {
        size_t elem_size = compute_dtype_size(dtype);
        memcpy(recv_data, send_data, count * elem_size);
        (void)backend;
        (void)op;
    }
    return COMPUTE_SUCCESS;
}

static ComputeResult cuda_scatter(ComputeBackend* backend,
                                   const void* send_data, void* recv_data,
                                   size_t count, ComputeDataType dtype, int root) {
#if COMPUTE_HAS_MPI
    CUDABackendContext* ctx = (CUDABackendContext*)backend;
    if (ctx && ctx->num_nodes > 1) {
        MPI_Datatype mpi_type;
        switch (dtype) {
            case COMPUTE_DTYPE_FLOAT32:   mpi_type = MPI_FLOAT; break;
            case COMPUTE_DTYPE_FLOAT64:   mpi_type = MPI_DOUBLE; break;
            case COMPUTE_DTYPE_INT32:     mpi_type = MPI_INT; break;
            case COMPUTE_DTYPE_INT64:     mpi_type = MPI_LONG_LONG; break;
            default:                      mpi_type = MPI_BYTE; break;
        }
        MPI_Scatter(send_data, (int)count, mpi_type,
                    recv_data, (int)count, mpi_type, root, ctx->comm);
    } else
#endif
    {
        size_t elem_size = compute_dtype_size(dtype);
        memcpy(recv_data, send_data, count * elem_size);
        (void)backend;
        (void)root;
    }
    return COMPUTE_SUCCESS;
}

static ComputeResult cuda_gather(ComputeBackend* backend,
                                  const void* send_data, void* recv_data,
                                  size_t count, ComputeDataType dtype, int root) {
#if COMPUTE_HAS_MPI
    CUDABackendContext* ctx = (CUDABackendContext*)backend;
    if (ctx && ctx->num_nodes > 1) {
        MPI_Datatype mpi_type;
        switch (dtype) {
            case COMPUTE_DTYPE_FLOAT32:   mpi_type = MPI_FLOAT; break;
            case COMPUTE_DTYPE_FLOAT64:   mpi_type = MPI_DOUBLE; break;
            case COMPUTE_DTYPE_INT32:     mpi_type = MPI_INT; break;
            case COMPUTE_DTYPE_INT64:     mpi_type = MPI_LONG_LONG; break;
            default:                      mpi_type = MPI_BYTE; break;
        }
        MPI_Gather(send_data, (int)count, mpi_type,
                   recv_data, (int)count, mpi_type, root, ctx->comm);
    } else
#endif
    {
        size_t elem_size = compute_dtype_size(dtype);
        memcpy(recv_data, send_data, count * elem_size);
        (void)backend;
        (void)root;
    }
    return COMPUTE_SUCCESS;
}

static ComputeResult cuda_allgather(ComputeBackend* backend,
                                     const void* send_data, void* recv_data,
                                     size_t count, ComputeDataType dtype) {
#if COMPUTE_HAS_NCCL
    CUDABackendContext* ctx = (CUDABackendContext*)backend;
    if (ctx && ctx->nccl_initialized) {
        ncclDataType_t nccl_type;
        switch (dtype) {
            case COMPUTE_DTYPE_FLOAT32:   nccl_type = ncclFloat32; break;
            case COMPUTE_DTYPE_FLOAT64:   nccl_type = ncclFloat64; break;
            case COMPUTE_DTYPE_INT32:     nccl_type = ncclInt32; break;
            case COMPUTE_DTYPE_INT64:     nccl_type = ncclInt64; break;
            default:                      nccl_type = ncclChar; break;
        }
        ncclAllGather(send_data, recv_data, count, nccl_type, ctx->nccl_comm, ctx->streams[0]);
        cudaStreamSynchronize(ctx->streams[0]);
        return COMPUTE_SUCCESS;
    }
#endif

#if COMPUTE_HAS_MPI
    CUDABackendContext* ctx = (CUDABackendContext*)backend;
    if (ctx && ctx->num_nodes > 1) {
        MPI_Datatype mpi_type;
        switch (dtype) {
            case COMPUTE_DTYPE_FLOAT32:   mpi_type = MPI_FLOAT; break;
            case COMPUTE_DTYPE_FLOAT64:   mpi_type = MPI_DOUBLE; break;
            case COMPUTE_DTYPE_INT32:     mpi_type = MPI_INT; break;
            case COMPUTE_DTYPE_INT64:     mpi_type = MPI_LONG_LONG; break;
            default:                      mpi_type = MPI_BYTE; break;
        }
        MPI_Allgather(send_data, (int)count, mpi_type,
                      recv_data, (int)count, mpi_type, ctx->comm);
    } else
#endif
    {
        size_t elem_size = compute_dtype_size(dtype);
        memcpy(recv_data, send_data, count * elem_size);
        (void)backend;
    }
    return COMPUTE_SUCCESS;
}

static ComputeResult cuda_reduce_scatter(ComputeBackend* backend,
                                          const void* send_data, void* recv_data,
                                          size_t count, ComputeDataType dtype,
                                          ComputeReduceOp op) {
#if COMPUTE_HAS_NCCL
    CUDABackendContext* ctx = (CUDABackendContext*)backend;
    if (ctx && ctx->nccl_initialized) {
        ncclDataType_t nccl_type;
        ncclRedOp_t nccl_op;

        switch (dtype) {
            case COMPUTE_DTYPE_FLOAT32:   nccl_type = ncclFloat32; break;
            case COMPUTE_DTYPE_FLOAT64:   nccl_type = ncclFloat64; break;
            case COMPUTE_DTYPE_INT32:     nccl_type = ncclInt32; break;
            case COMPUTE_DTYPE_INT64:     nccl_type = ncclInt64; break;
            default:                      nccl_type = ncclChar; break;
        }

        switch (op) {
            case COMPUTE_REDUCE_SUM:  nccl_op = ncclSum; break;
            case COMPUTE_REDUCE_PROD: nccl_op = ncclProd; break;
            case COMPUTE_REDUCE_MIN:  nccl_op = ncclMin; break;
            case COMPUTE_REDUCE_MAX:  nccl_op = ncclMax; break;
            default:                  nccl_op = ncclSum; break;
        }

        ncclReduceScatter(send_data, recv_data, count, nccl_type, nccl_op, ctx->nccl_comm, ctx->streams[0]);
        cudaStreamSynchronize(ctx->streams[0]);
        return COMPUTE_SUCCESS;
    }
#endif

#if COMPUTE_HAS_MPI
    CUDABackendContext* ctx = (CUDABackendContext*)backend;
    if (ctx && ctx->num_nodes > 1) {
        MPI_Datatype mpi_type;
        MPI_Op mpi_op;

        switch (dtype) {
            case COMPUTE_DTYPE_FLOAT32:   mpi_type = MPI_FLOAT; break;
            case COMPUTE_DTYPE_FLOAT64:   mpi_type = MPI_DOUBLE; break;
            case COMPUTE_DTYPE_INT32:     mpi_type = MPI_INT; break;
            case COMPUTE_DTYPE_INT64:     mpi_type = MPI_LONG_LONG; break;
            default:                      mpi_type = MPI_BYTE; break;
        }

        switch (op) {
            case COMPUTE_REDUCE_SUM:  mpi_op = MPI_SUM; break;
            case COMPUTE_REDUCE_PROD: mpi_op = MPI_PROD; break;
            case COMPUTE_REDUCE_MIN:  mpi_op = MPI_MIN; break;
            case COMPUTE_REDUCE_MAX:  mpi_op = MPI_MAX; break;
            default:                  mpi_op = MPI_SUM; break;
        }

        int* recvcounts = (int*)calloc(ctx->num_nodes, sizeof(int));
        for (int i = 0; i < ctx->num_nodes; i++) {
            recvcounts[i] = (int)count;
        }
        MPI_Reduce_scatter(send_data, recv_data, recvcounts, mpi_type, mpi_op, ctx->comm);
        free(recvcounts);
    } else
#endif
    {
        size_t elem_size = compute_dtype_size(dtype);
        memcpy(recv_data, send_data, count * elem_size);
        (void)backend;
        (void)op;
    }
    return COMPUTE_SUCCESS;
}

// ============================================================================
// Execution & Scheduling
// ============================================================================

static ComputeResult cuda_execute(ComputeBackend* backend,
                                   const ComputeQuantumOp* op,
                                   const ComputeExecutionPlan* plan,
                                   ComputeStream* stream) {
    if (!backend || !op) {
        return COMPUTE_ERROR_INVALID_ARGUMENT;
    }

    CUDABackendContext* ctx = (CUDABackendContext*)backend;
    ComputeResult result = COMPUTE_SUCCESS;

    switch (op->type) {
        case QUANTUM_OP_UNITARY:
            result = cuda_quantum_unitary(backend,
                                          (float*)op->output_data, op->output_size,
                                          (const float*)op->parameters, op->param_size,
                                          stream);
            break;

        case QUANTUM_OP_NORMALIZE:
            result = cuda_quantum_normalize(backend,
                                            (float*)op->output_data, op->output_size,
                                            stream);
            break;

        case QUANTUM_OP_TENSOR_CONTRACT:
            if (op->num_dims >= 3) {
                result = cuda_quantum_tensor_contract(backend,
                                                      (float*)op->output_data,
                                                      (const float*)op->input_data,
                                                      (const float*)op->parameters,
                                                      op->dims[0], op->dims[1], op->dims[2],
                                                      stream);
            }
            break;

        case QUANTUM_OP_GRADIENT:
            result = cuda_quantum_gradient(backend,
                                           (float*)op->output_data,
                                           (const float*)op->input_data,
                                           (const float*)op->parameters,
                                           op->input_size,
                                           stream);
            break;

        case QUANTUM_OP_INNER_PRODUCT:
            result = cuda_quantum_inner_product(backend,
                                                (float*)op->output_data,
                                                (const float*)op->input_data,
                                                (const float*)op->parameters,
                                                op->input_size,
                                                stream);
            break;

        case QUANTUM_OP_EXPECTATION:
            result = cuda_quantum_expectation(backend,
                                              (float*)op->output_data,
                                              (const float*)op->input_data,
                                              (const float*)op->parameters,
                                              op->input_size,
                                              stream);
            break;

        default:
            result = COMPUTE_ERROR_NOT_IMPLEMENTED;
            break;
    }

    ctx->metrics.operations_per_second += 1.0;
    (void)plan;

    return result;
}

static ComputeExecutionPlan* cuda_create_plan(ComputeBackend* backend,
                                               const ComputeQuantumOp* op) {
    CUDABackendContext* ctx = (CUDABackendContext*)backend;
    if (!ctx || !op) return NULL;

    ComputeExecutionPlan* plan = (ComputeExecutionPlan*)calloc(1, sizeof(ComputeExecutionPlan));
    if (!plan) return NULL;

    // Distribute across GPUs and nodes
    int total_workers = ctx->num_nodes * ctx->num_devices;
    plan->num_partitions = total_workers;
    plan->partition_size = op->input_size / total_workers;

    plan->node_assignments = (int*)calloc(plan->num_partitions, sizeof(int));
    plan->offsets = (size_t*)calloc(plan->num_partitions, sizeof(size_t));
    plan->sizes = (size_t*)calloc(plan->num_partitions, sizeof(size_t));

    if (!plan->node_assignments || !plan->offsets || !plan->sizes) {
        free(plan->node_assignments);
        free(plan->offsets);
        free(plan->sizes);
        free(plan);
        return NULL;
    }

    for (size_t i = 0; i < plan->num_partitions; i++) {
        plan->node_assignments[i] = (int)(i / ctx->num_devices);
        plan->offsets[i] = i * plan->partition_size;
        plan->sizes[i] = (i == plan->num_partitions - 1) ?
                         (op->input_size - i * plan->partition_size) :
                         plan->partition_size;
    }

    return plan;
}

static void cuda_destroy_plan(ComputeBackend* backend, ComputeExecutionPlan* plan) {
    (void)backend;
    if (!plan) return;

    free(plan->node_assignments);
    free(plan->offsets);
    free(plan->sizes);
    free(plan->send_targets);
    free(plan->recv_sources);
    free(plan->workspace);
    free(plan);
}

// ============================================================================
// Performance Monitoring
// ============================================================================

static ComputeResult cuda_get_metrics(ComputeBackend* backend, ComputeMetrics* metrics) {
    CUDABackendContext* ctx = (CUDABackendContext*)backend;
    if (!ctx || !metrics) return COMPUTE_ERROR_INVALID_ARGUMENT;

    *metrics = ctx->metrics;
    metrics->peak_memory_bytes = ctx->peak_allocated;
    metrics->current_memory_bytes = ctx->total_allocated;

    return COMPUTE_SUCCESS;
}

static ComputeResult cuda_reset_metrics(ComputeBackend* backend) {
    CUDABackendContext* ctx = (CUDABackendContext*)backend;
    if (!ctx) return COMPUTE_ERROR_INVALID_ARGUMENT;

    memset(&ctx->metrics, 0, sizeof(ComputeMetrics));
    return COMPUTE_SUCCESS;
}

// ============================================================================
// Backend Registration
// ============================================================================

static const ComputeBackendOps cuda_ops = {
    // Lifecycle
    .init = cuda_init,
    .cleanup = cuda_cleanup,
    .probe = cuda_probe,
    .get_capabilities = cuda_get_capabilities,

    // Memory
    .alloc = cuda_alloc,
    .free = cuda_free,
    .memcpy = cuda_memcpy,
    .memset = cuda_memset,

    // Streams
    .create_stream = cuda_create_stream,
    .destroy_stream = cuda_destroy_stream,
    .synchronize_stream = cuda_synchronize_stream,
    .create_event = cuda_create_event,
    .destroy_event = cuda_destroy_event,
    .record_event = cuda_record_event,
    .wait_event = cuda_wait_event,

    // Quantum operations
    .quantum_unitary = cuda_quantum_unitary,
    .quantum_normalize = cuda_quantum_normalize,
    .quantum_tensor_contract = cuda_quantum_tensor_contract,
    .quantum_gradient = cuda_quantum_gradient,
    .quantum_inner_product = cuda_quantum_inner_product,
    .quantum_expectation = cuda_quantum_expectation,

    // Collective communication
    .barrier = cuda_barrier,
    .broadcast = cuda_broadcast,
    .allreduce = cuda_allreduce,
    .scatter = cuda_scatter,
    .gather = cuda_gather,
    .allgather = cuda_allgather,
    .reduce_scatter = cuda_reduce_scatter,

    // Execution
    .execute = cuda_execute,
    .create_plan = cuda_create_plan,
    .destroy_plan = cuda_destroy_plan,

    // Metrics
    .get_metrics = cuda_get_metrics,
    .reset_metrics = cuda_reset_metrics,
};

// Register the CUDA backend at library load time
// Priority 100 = highest, preferred over CPU (10), OpenCL (50), Metal (80)
COMPUTE_REGISTER_BACKEND(COMPUTE_BACKEND_CUDA, "CUDA (NVIDIA GPU)", "1.0.0", 100, cuda_ops)

#endif // COMPUTE_HAS_CUDA
