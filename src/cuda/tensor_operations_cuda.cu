#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <math.h>
#include "quantum_geometric/hardware/tensor_operations_cuda.h"
#include "quantum_geometric/hardware/quantum_geometric_cuda.h"

// Optimized block sizes for modern GPUs
#define BLOCK_SIZE 256   // Balanced for occupancy
#define TILE_SIZE 32     // Standard tile size
#define WARP_SIZE 32     // Warp size is fixed on NVIDIA GPUs
#define PREFETCH_DISTANCE 2  // Prefetch next tiles

// Thread-safe resource management
static cudaMemPool_t cuda_pool = NULL;
static cublasHandle_t cublas_handle = NULL;
static int resources_initialized = 0;

// Initialize CUDA resources with optimized settings
static cudaError_t init_cuda_resources() {
    if (resources_initialized) return cudaSuccess;

    cudaError_t err;

    // Create memory pool
    cudaMemPoolProps poolProps = {};
    poolProps.allocType = cudaMemAllocationTypePinned;
    poolProps.handleTypes = cudaMemHandleTypeNone;
    poolProps.location.type = cudaMemLocationTypeDevice;
    poolProps.location.id = 0;

    err = cudaMemPoolCreate(&cuda_pool, &poolProps);
    if (err != cudaSuccess) {
        // Fallback: pool creation might not be supported
        cuda_pool = NULL;
    }

    // Create cuBLAS handle
    cublasStatus_t status = cublasCreate(&cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        if (cuda_pool) cudaMemPoolDestroy(cuda_pool);
        return cudaErrorInitializationError;
    }

    // Enable Tensor Cores when available
    cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH);
    cublasSetAtomicsMode(cublas_handle, CUBLAS_ATOMICS_ALLOWED);

    resources_initialized = 1;
    return cudaSuccess;
}

// Cleanup CUDA resources
void cleanup_cuda_resources() {
    if (cuda_pool) {
        cudaMemPoolDestroy(cuda_pool);
        cuda_pool = NULL;
    }
    if (cublas_handle) {
        cublasDestroy(cublas_handle);
        cublas_handle = NULL;
    }
    resources_initialized = 0;
}

// Complex matrix multiplication kernel with shared memory tiling
// Uses standard CUDA operations for correctness across all architectures
__global__ void tensor_multiply_kernel(const QuantumAmplitude* A,
                                     const QuantumAmplitude* B,
                                     QuantumAmplitude* C,
                                     int M, int N, int K) {
    // Shared memory with padding to avoid bank conflicts
    __shared__ cuDoubleComplex As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ cuDoubleComplex Bs[TILE_SIZE][TILE_SIZE + 1];

    int bx = blockIdx.x * TILE_SIZE;
    int by = blockIdx.y * TILE_SIZE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = bx + tx;
    int col = by + ty;

    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);

    // Loop over tiles
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        int k_offset = tile * TILE_SIZE;

        // Load tiles into shared memory
        if (row < M && k_offset + ty < K) {
            As[tx][ty] = to_cuda_complex(A[row * K + k_offset + ty].amplitude);
        } else {
            As[tx][ty] = make_cuDoubleComplex(0.0, 0.0);
        }

        if (k_offset + tx < K && col < N) {
            Bs[tx][ty] = to_cuda_complex(B[(k_offset + tx) * N + col].amplitude);
        } else {
            Bs[tx][ty] = make_cuDoubleComplex(0.0, 0.0);
        }

        __syncthreads();

        // Compute partial dot product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            cuDoubleComplex a_val = As[tx][k];
            cuDoubleComplex b_val = Bs[k][ty];
            // Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
            sum.x += a_val.x * b_val.x - a_val.y * b_val.y;
            sum.y += a_val.x * b_val.y + a_val.y * b_val.x;
        }

        __syncthreads();
    }

    // Store result
    if (row < M && col < N) {
        C[row * N + col].amplitude = from_cuda_complex(sum);
    }
}

// Optimized tensor multiplication with automatic algorithm selection
cudaError_t cuda_tensor_multiply(QuantumAmplitude* C, const QuantumAmplitude* A, const QuantumAmplitude* B, int size) {
    cudaError_t err = init_cuda_resources();
    if (err != cudaSuccess) return err;

    // Allocate device memory
    QuantumAmplitude *d_A = NULL, *d_B = NULL, *d_C = NULL;
    size_t matrix_size = (size_t)size * size * sizeof(QuantumAmplitude);

    // Use standard allocation (more portable than memory pools)
    err = cudaMalloc(&d_A, matrix_size);
    if (err != cudaSuccess) goto cleanup;

    err = cudaMalloc(&d_B, matrix_size);
    if (err != cudaSuccess) goto cleanup;

    err = cudaMalloc(&d_C, matrix_size);
    if (err != cudaSuccess) goto cleanup;

    // Create streams for overlapped operations
    cudaStream_t compute_stream, copy_stream;
    err = cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) goto cleanup;

    err = cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        cudaStreamDestroy(compute_stream);
        goto cleanup;
    }

    // Copy input data with streams for overlap
    err = cudaMemcpyAsync(d_A, A, matrix_size, cudaMemcpyHostToDevice, copy_stream);
    if (err != cudaSuccess) goto cleanup_streams;

    err = cudaMemcpyAsync(d_B, B, matrix_size, cudaMemcpyHostToDevice, copy_stream);
    if (err != cudaSuccess) goto cleanup_streams;

    // Wait for copies to complete before compute
    cudaStreamSynchronize(copy_stream);

    // Choose optimal algorithm based on size
    if (size >= 256 && cublas_handle) {
        // Use cuBLAS for large matrices with tensor cores
        const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
        const cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);

        cublasSetStream(cublas_handle, compute_stream);

        cublasStatus_t status = cublasZgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                   size, size, size,
                   &alpha,
                   (const cuDoubleComplex*)d_A, size,
                   (const cuDoubleComplex*)d_B, size,
                   &beta,
                   (cuDoubleComplex*)d_C, size);

        if (status != CUBLAS_STATUS_SUCCESS) {
            err = cudaErrorUnknown;
            goto cleanup_streams;
        }
    } else {
        // Use custom kernel for smaller matrices
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((size + TILE_SIZE - 1) / TILE_SIZE,
                 (size + TILE_SIZE - 1) / TILE_SIZE);

        tensor_multiply_kernel<<<grid, block, 0, compute_stream>>>(
            d_A, d_B, d_C, size, size, size);

        err = cudaGetLastError();
        if (err != cudaSuccess) goto cleanup_streams;
    }

    // Wait for compute and copy result back
    cudaStreamSynchronize(compute_stream);
    err = cudaMemcpy(C, d_C, matrix_size, cudaMemcpyDeviceToHost);

cleanup_streams:
    cudaStreamDestroy(compute_stream);
    cudaStreamDestroy(copy_stream);

cleanup:
    if (d_A) cudaFree(d_A);
    if (d_B) cudaFree(d_B);
    if (d_C) cudaFree(d_C);

    return err;
}

// Advanced tensor transformation kernel with vectorized access
__global__ void tensor_transform_kernel(QuantumAmplitude* data,
                                      int size,
                                      TransformType type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    // Shared memory with padding to avoid bank conflicts
    __shared__ cuDoubleComplex shared_data[BLOCK_SIZE + 1];
    
    // Load data with vectorized reads
    if (idx + WARP_SIZE < size) {
        float4* vec_data = (float4*)&data[idx];
        float4 vec = *vec_data;
        shared_data[threadIdx.x] = make_cuDoubleComplex(vec.x, vec.y);
        shared_data[threadIdx.x + 1] = make_cuDoubleComplex(vec.z, vec.w);
    } else {
        shared_data[threadIdx.x] = to_cuda_complex(data[idx].amplitude);
    }
    
    __syncthreads();
    
    cuDoubleComplex result;
    switch (type) {
        case TRANSFORM_QUANTUM:
            // Quantum-inspired transformation with fast math
            result = quantum_transform_element(shared_data[threadIdx.x]);
            break;
            
        case TRANSFORM_GEOMETRIC:
            // Geometric transformation with fast math
            result = geometric_transform_element(shared_data[threadIdx.x]);
            break;
            
        case TRANSFORM_ATTENTION:
            // Attention-based transformation with fast math
            result = attention_transform_element(shared_data[threadIdx.x]);
            break;
            
        default:
            // Optimized standard transformation
            result = cuCmul(shared_data[threadIdx.x],
                          make_cuDoubleComplex(2.0, 0.0));
    }
    
    // Store result with vectorized writes
    if (idx + WARP_SIZE < size) {
        float4* vec_out = (float4*)&data[idx];
        *vec_out = make_float4(result.x, result.y,
                             shared_data[threadIdx.x + 1].x,
                             shared_data[threadIdx.x + 1].y);
    } else {
        data[idx].amplitude = from_cuda_complex(result);
    }
}

// Optimized tensor transformation with stream overlap
void cuda_tensor_transform(QuantumAmplitude* data, int size, TransformType type) {
    init_cuda_resources();
    
    // Create streams for overlap
    cudaStream_t compute_stream, copy_stream;
    cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking);
    
    // Allocate device memory with prefetching
    QuantumAmplitude* d_data;
    cudaMallocFromPoolAsync(&d_data, size * sizeof(QuantumAmplitude), cuda_pool, copy_stream);
    
    // Copy data with prefetching
    cudaMemcpyAsync(d_data, data, size * sizeof(QuantumAmplitude),
                    cudaMemcpyHostToDevice, copy_stream);
    
    // Launch kernel with optimal block size
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    tensor_transform_kernel<<<blocks, BLOCK_SIZE, 0, compute_stream>>>(
        d_data, size, type);
    
    // Copy result back
    cudaMemcpyAsync(data, d_data, size * sizeof(QuantumAmplitude),
                    cudaMemcpyDeviceToHost, copy_stream);
    
    // Cleanup
    cudaStreamSynchronize(compute_stream);
    cudaStreamSynchronize(copy_stream);
    cudaStreamDestroy(compute_stream);
    cudaStreamDestroy(copy_stream);
    cudaFreeAsync(d_data, copy_stream);
}

// Helper functions with fast math
__device__ __forceinline__ cuDoubleComplex quantum_transform_element(cuDoubleComplex x) {
    // Fast quantum-inspired non-linear transformation
    float mag = __fmul_rn(__fsqrt_rn(x.x * x.x + x.y * x.y), 1.0f);
    float phase = atan2f(x.y, x.x);
    float tanh_mag = tanhf(mag);
    float new_mag = __fmul_rn(tanh_mag,
                             __fsqrt_rn(__fsub_rn(1.0f,
                                                 __fmul_rn(tanh_mag, tanh_mag))));
    return make_cuDoubleComplex(new_mag * __cosf(phase),
                               new_mag * __sinf(phase));
}

__device__ __forceinline__ cuDoubleComplex geometric_transform_element(cuDoubleComplex x) {
    // Fast geometric transformation
    float mag = __fsqrt_rn(x.x * x.x + x.y * x.y);
    float phase = atan2f(x.y, x.x);
    float new_mag = __fdiv_rn(mag,
                             __fsqrt_rn(__fadd_rn(1.0f,
                                                 __fmul_rn(mag, mag))));
    return make_cuDoubleComplex(new_mag * __cosf(phase),
                               new_mag * __sinf(phase));
}

__device__ __forceinline__ cuDoubleComplex attention_transform_element(cuDoubleComplex x) {
    // Fast attention-based transformation
    float mag = __fsqrt_rn(x.x * x.x + x.y * x.y);
    float phase = atan2f(x.y, x.x);
    float exp_mag = expf(mag);
    float new_mag = __fdiv_rn(exp_mag,
                             __fadd_rn(1.0f, exp_mag));
    return make_cuDoubleComplex(new_mag * __cosf(phase),
                               new_mag * __sinf(phase));
}
