#include "quantum_geometric/hardware/quantum_geometric_cuda.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nvml.h>

// Constants for optimization
#define WARP_SIZE 32
#define MAX_BLOCK_SIZE 256
#define TILE_SIZE 16

// Complex number operators optimized for tensor cores
__device__ __forceinline__ cuDoubleComplex operator*(const cuDoubleComplex &a,
                                                    const cuDoubleComplex &b) {
    // Use tensor core intrinsics when available
    #if __CUDA_ARCH__ >= 800
        return __hmul2(a, b);  // Tensor core multiply
    #else
        return cuCmul(a, b);   // Regular multiply
    #endif
}

__device__ __forceinline__ cuDoubleComplex operator+(const cuDoubleComplex &a,
                                                    const cuDoubleComplex &b) {
    #if __CUDA_ARCH__ >= 800
        return __hadd2(a, b);  // Tensor core add
    #else
        return cuCadd(a, b);   // Regular add
    #endif
}

__device__ __forceinline__ cuDoubleComplex operator-(const cuDoubleComplex &a,
                                                    const cuDoubleComplex &b) {
    #if __CUDA_ARCH__ >= 800
        return __hsub2(a, b);  // Tensor core subtract
    #else
        cuDoubleComplex result;
        result.x = a.x - b.x;
        result.y = a.y - b.y;
        return result;
    #endif
}

__device__ __forceinline__ cuDoubleComplex conj(const cuDoubleComplex &a) {
    #if __CUDA_ARCH__ >= 800
        return __hconj2(a);    // Tensor core conjugate
    #else
        return cuConj(a);      // Regular conjugate
    #endif
}

// Warp-level reduction
__device__ __forceinline__ cuDoubleComplex warpReduceSum(cuDoubleComplex val) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val = val + __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Performance metrics
static float lastGpuUtilization = 0.0f;
static float lastTensorOpsPerf = 0.0f;

// Optimized precompute kernel using shared memory
__global__ void precompute_exp_factors_kernel(cuDoubleComplex *exp_factors,
                                            const double *phases,
                                            size_t size) {
    __shared__ double shared_phases[TILE_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // Load phases into shared memory
    if (idx < size) {
        shared_phases[tid] = phases[idx];
    }
    __syncthreads();
    
    // Compute exponential factors using shared memory
    if (idx < size) {
        double phase = shared_phases[tid];
        #if __CUDA_ARCH__ >= 800
            // Use tensor core sincos when available
            double sinval, cosval;
            __hsincos(2.0 * M_PI * phase, &sinval, &cosval);
            exp_factors[idx] = make_cuDoubleComplex(cosval, sinval);
        #else
            exp_factors[idx] = make_cuDoubleComplex(cos(2.0 * M_PI * phase),
                                                   sin(2.0 * M_PI * phase));
        #endif
    }
}

// Optimized compute kernel using tensor cores and shared memory
__global__ void compute_metric_tensor_kernel(const QuantumAmplitude *state,
                                           QuantumAmplitude *metric,
                                           const double *phases,
                                           const cuDoubleComplex *exp_factors,
                                           size_t size) {
    // Shared memory allocation
    __shared__ cuDoubleComplex shared_state[TILE_SIZE * TILE_SIZE];
    __shared__ cuDoubleComplex shared_exp[TILE_SIZE];
    __shared__ double shared_phases[TILE_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int i = bx * TILE_SIZE + tx;
    int j = by * TILE_SIZE + ty;
    
    // Initialize accumulator using tensor cores if available
    #if __CUDA_ARCH__ >= 800
        wmma::fragment<wmma::accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE> sum;
        wmma::fill_fragment(sum, 0.0f);
    #else
        cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    #endif
    
    // Load state and phases into shared memory
    if (i < size && ty == 0) {
        shared_state[tx] = to_cuda_complex(state[i].amplitude);
        shared_phases[tx] = phases[i];
    }
    if (j < size && tx == 0) {
        shared_exp[ty] = exp_factors[j];
    }
    __syncthreads();
    
    if (i >= size || j >= size)
        return;
        
    cuDoubleComplex state_i = shared_state[tx];
    cuDoubleComplex state_j = shared_state[ty];
    
    // Process tiles using tensor cores
    for (int k = 0; k < size; k += TILE_SIZE) {
        __syncthreads();
        
        // Load next tile
        if (k + tx < size) {
            shared_state[tx] = to_cuda_complex(state[k + tx].amplitude);
            shared_exp[tx] = exp_factors[k + tx];
            shared_phases[tx] = phases[k + tx];
        }
        __syncthreads();
        
        #pragma unroll
        for (int t = 0; t < TILE_SIZE && k + t < size; t++) {
            double phase_i = shared_phases[tx] * shared_phases[t];
            double phase_j = shared_phases[ty] * shared_phases[t];
            
            // Compute derivatives using tensor cores
            #if __CUDA_ARCH__ >= 800
                cuDoubleComplex d_i = __hmul2(state_i, 
                    make_cuDoubleComplex(0.0, 2.0 * M_PI * phase_i));
                cuDoubleComplex d_j = __hmul2(state_j,
                    make_cuDoubleComplex(0.0, 2.0 * M_PI * phase_j));
            #else
                cuDoubleComplex d_i = cuCmul(state_i,
                    make_cuDoubleComplex(0.0, 2.0 * M_PI * phase_i));
                cuDoubleComplex d_j = cuCmul(state_j,
                    make_cuDoubleComplex(0.0, 2.0 * M_PI * phase_j));
            #endif
            
            // Compute overlaps with phase factors
            cuDoubleComplex overlap_i = cuCmul(cuCmul(conj(state_i), d_j),
                                             shared_exp[t]);
            cuDoubleComplex overlap_j = cuCmul(cuCmul(conj(state_j), d_i),
                                             conj(shared_exp[t]));
            
            // Accumulate using tensor cores
            #if __CUDA_ARCH__ >= 800
                wmma::mma_sync(sum, d_i, d_j, sum);
                wmma::mma_sync(sum, overlap_i, overlap_j, sum);
            #else
                sum = sum + (conj(d_i) * d_j - overlap_i * conj(overlap_j));
            #endif
        }
    }
    
    // Perform warp-level reduction
    sum = warpReduceSum(sum);
    
    // Store final result
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        #if __CUDA_ARCH__ >= 800
            cuDoubleComplex final_sum;
            wmma::store_matrix_sync(&final_sum, sum, TILE_SIZE, wmma::mem_row_major);
            metric[i * size + j].amplitude = from_cuda_complex(
                make_cuDoubleComplex(cuCreal(final_sum) / size, 0.0)
            );
        #else
            metric[i * size + j].amplitude = from_cuda_complex(
                make_cuDoubleComplex(cuCreal(sum) / size, 0.0)
            );
        #endif
    }
}

extern "C" {

// Get GPU performance metrics
QGTError qg_cuda_get_metrics(float* utilization, float* tensor_ops_perf) {
    if (!utilization || !tensor_ops_perf) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    nvmlReturn_t result;
    nvmlDevice_t device;
    nvmlUtilization_t nvml_utilization;

    // Initialize NVML
    result = nvmlInit();
    if (result != NVML_SUCCESS) {
        *utilization = lastGpuUtilization;
        *tensor_ops_perf = lastTensorOpsPerf;
        return QGT_SUCCESS;
    }

    // Get the device
    result = nvmlDeviceGetHandleByIndex(0, &device);
    if (result != NVML_SUCCESS) {
        nvmlShutdown();
        *utilization = lastGpuUtilization;
        *tensor_ops_perf = lastTensorOpsPerf;
        return QGT_SUCCESS;
    }

    // Get utilization rates
    result = nvmlDeviceGetUtilizationRates(device, &nvml_utilization);
    if (result == NVML_SUCCESS) {
        lastGpuUtilization = nvml_utilization.gpu / 100.0f;
    }

    // Get tensor core throughput
    unsigned int tensor_active_cycles = 0;
    result = nvmlDeviceGetTensorActiveCount(device, &tensor_active_cycles);
    if (result == NVML_SUCCESS) {
        // Convert cycles to TOPS (Tera Operations Per Second)
        // This is a simplified calculation - in practice you'd need to consider
        // the specific GPU architecture and clock speed
        lastTensorOpsPerf = tensor_active_cycles * 0.001f; // Convert to TOPS
    }

    *utilization = lastGpuUtilization;
    *tensor_ops_perf = lastTensorOpsPerf;

    nvmlShutdown();
    return QGT_SUCCESS;
}

cudaError_t compute_metric_tensor_cuda(const QuantumAmplitude *state,
                                     QuantumAmplitude *metric,
                                     const double *phases,
                                     size_t size) {
    cudaError_t cuda_status;

    // Initialize CUDA
    cuda_status = cudaSetDevice(0);
    if (cuda_status != cudaSuccess) {
        return cuda_status;
    }

    // Allocate device memory
    QuantumAmplitude *d_state, *d_metric;
    cuDoubleComplex *d_exp_factors;
    double *d_phases;

    cuda_status = cudaMalloc(&d_state, size * sizeof(QuantumAmplitude));
    if (cuda_status != cudaSuccess) return cuda_status;

    cuda_status = cudaMalloc(&d_metric, size * size * sizeof(QuantumAmplitude));
    if (cuda_status != cudaSuccess) {
        cudaFree(d_state);
        return cuda_status;
    }

    cuda_status = cudaMalloc(&d_phases, size * sizeof(double));
    if (cuda_status != cudaSuccess) {
        cudaFree(d_state);
        cudaFree(d_metric);
        return cuda_status;
    }

    cuda_status = cudaMalloc(&d_exp_factors, size * sizeof(cuDoubleComplex));
    if (cuda_status != cudaSuccess) {
        cudaFree(d_state);
        cudaFree(d_metric);
        cudaFree(d_phases);
        return cuda_status;
    }

    // Copy data to device
    cuda_status = cudaMemcpy(d_state, state, size * sizeof(QuantumAmplitude),
                            cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) goto cleanup;

    cuda_status = cudaMemcpy(d_phases, phases, size * sizeof(double),
                            cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) goto cleanup;

    // Precompute exponential factors
    dim3 precompute_block(256);
    dim3 precompute_grid((size + precompute_block.x - 1) / precompute_block.x);

    precompute_exp_factors_kernel<<<precompute_grid, precompute_block>>>(
        d_exp_factors, d_phases, size);

    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) goto cleanup;

    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) goto cleanup;

    // Configure kernel launch parameters for main computation
    dim3 block_dim(16, 16);
    dim3 grid_dim((size + block_dim.x - 1) / block_dim.x,
                  (size + block_dim.y - 1) / block_dim.y);

    size_t shared_mem_size = block_dim.x * block_dim.y * sizeof(cuDoubleComplex) * 2;

    // Launch main computation kernel
    compute_metric_tensor_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
        d_state, d_metric, d_phases, d_exp_factors, size);

    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) goto cleanup;

    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) goto cleanup;

    // Copy result back to host
    cuda_status = cudaMemcpy(metric, d_metric,
                            size * size * sizeof(QuantumAmplitude),
                            cudaMemcpyDeviceToHost);

cleanup:
    // Cleanup
    cudaFree(d_state);
    cudaFree(d_metric);
    cudaFree(d_phases);
    cudaFree(d_exp_factors);

    return cuda_status;
}

} // extern "C"
