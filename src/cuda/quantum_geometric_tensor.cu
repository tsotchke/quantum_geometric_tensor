#include "quantum_geometric/hardware/quantum_geometric_cuda.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nvml.h>
#include <math.h>

// Include tensor core headers for supported architectures
#if __CUDA_ARCH__ >= 700
#include <mma.h>
using namespace nvcuda;
#endif

// Constants for optimization
#define WARP_SIZE 32
#define MAX_BLOCK_SIZE 256
#define TILE_SIZE 16
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Complex number operators - always use cuBLAS complex operations
// Tensor cores work on matrix fragments, not individual complex ops
__device__ __forceinline__ cuDoubleComplex operator*(const cuDoubleComplex &a,
                                                    const cuDoubleComplex &b) {
    return cuCmul(a, b);
}

__device__ __forceinline__ cuDoubleComplex operator+(const cuDoubleComplex &a,
                                                    const cuDoubleComplex &b) {
    return cuCadd(a, b);
}

__device__ __forceinline__ cuDoubleComplex operator-(const cuDoubleComplex &a,
                                                    const cuDoubleComplex &b) {
    cuDoubleComplex result;
    result.x = a.x - b.x;
    result.y = a.y - b.y;
    return result;
}

__device__ __forceinline__ cuDoubleComplex conj(const cuDoubleComplex &a) {
    return cuConj(a);
}

// Warp-level reduction for complex numbers
// Need to reduce real and imaginary parts separately
__device__ __forceinline__ cuDoubleComplex warpReduceSum(cuDoubleComplex val) {
    // Shuffle real and imaginary parts separately
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        double real_part = __shfl_down_sync(0xffffffff, val.x, offset);
        double imag_part = __shfl_down_sync(0xffffffff, val.y, offset);
        val.x += real_part;
        val.y += imag_part;
    }
    return val;
}

// Block-level reduction for complex numbers
__device__ __forceinline__ cuDoubleComplex blockReduceSum(cuDoubleComplex val) {
    __shared__ double shared_real[32];
    __shared__ double shared_imag[32];

    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    // Warp-level reduction
    val = warpReduceSum(val);

    // Write reduced warp value
    if (lane == 0) {
        shared_real[wid] = val.x;
        shared_imag[wid] = val.y;
    }
    __syncthreads();

    // Final reduction in first warp
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    if (threadIdx.x < num_warps) {
        val.x = shared_real[lane];
        val.y = shared_imag[lane];
    } else {
        val.x = 0.0;
        val.y = 0.0;
    }

    if (wid == 0) {
        val = warpReduceSum(val);
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
    __shared__ double shared_phases[256];  // Match max block size

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load phases into shared memory
    if (idx < size && tid < 256) {
        shared_phases[tid] = phases[idx];
    }
    __syncthreads();

    // Compute exponential factors using shared memory
    if (idx < size) {
        double phase = shared_phases[tid];
        double angle = 2.0 * M_PI * phase;
        // Use sincos for better performance (single instruction on GPU)
        double sinval, cosval;
        sincos(angle, &sinval, &cosval);
        exp_factors[idx] = make_cuDoubleComplex(cosval, sinval);
    }
}

// Optimized compute kernel for quantum geometric metric tensor
// Uses shared memory tiling and efficient memory access patterns
__global__ void compute_metric_tensor_kernel(const QuantumAmplitude *state,
                                           QuantumAmplitude *metric,
                                           const double *phases,
                                           const cuDoubleComplex *exp_factors,
                                           size_t size) {
    // Shared memory allocation
    __shared__ cuDoubleComplex shared_state[TILE_SIZE];
    __shared__ cuDoubleComplex shared_exp[TILE_SIZE];
    __shared__ double shared_phases[TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int i = bx * TILE_SIZE + tx;
    int j = by * TILE_SIZE + ty;

    // Initialize accumulator
    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);

    if (i >= size || j >= size)
        return;

    // Load initial state elements for this thread
    cuDoubleComplex state_i = to_cuda_complex(state[i].amplitude);
    cuDoubleComplex state_j = to_cuda_complex(state[j].amplitude);
    double phase_i_base = phases[i];
    double phase_j_base = phases[j];

    // Process tiles
    for (int k = 0; k < size; k += TILE_SIZE) {
        // Load tile into shared memory (first warp does the loading)
        if (ty == 0 && k + tx < size) {
            shared_state[tx] = to_cuda_complex(state[k + tx].amplitude);
            shared_exp[tx] = exp_factors[k + tx];
            shared_phases[tx] = phases[k + tx];
        }
        __syncthreads();

        // Process elements in this tile
        int tile_end = min(TILE_SIZE, (int)(size - k));
        #pragma unroll 4
        for (int t = 0; t < tile_end; t++) {
            double phase_k = shared_phases[t];
            double phase_i = phase_i_base * phase_k;
            double phase_j = phase_j_base * phase_k;

            // Compute derivatives: d/dθ |ψ⟩ = i * phase * |ψ⟩
            cuDoubleComplex deriv_factor_i = make_cuDoubleComplex(0.0, 2.0 * M_PI * phase_i);
            cuDoubleComplex deriv_factor_j = make_cuDoubleComplex(0.0, 2.0 * M_PI * phase_j);

            cuDoubleComplex d_i = cuCmul(state_i, deriv_factor_i);
            cuDoubleComplex d_j = cuCmul(state_j, deriv_factor_j);

            // Compute overlaps with phase factors
            // ⟨∂_i ψ | ∂_j ψ⟩ - ⟨∂_i ψ | ψ⟩⟨ψ | ∂_j ψ⟩
            cuDoubleComplex overlap_i = cuCmul(cuCmul(conj(state_i), d_j), shared_exp[t]);
            cuDoubleComplex overlap_j = cuCmul(cuCmul(conj(state_j), d_i), conj(shared_exp[t]));

            // Accumulate metric tensor element
            cuDoubleComplex term1 = cuCmul(conj(d_i), d_j);
            cuDoubleComplex term2 = cuCmul(overlap_i, conj(overlap_j));
            sum = sum + (term1 - term2);
        }
        __syncthreads();
    }

    // Use block-level reduction for better accuracy
    sum = blockReduceSum(sum);

    // Store final result (only one thread per output element)
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        // Real part is the metric tensor element
        metric[i * size + j].amplitude = from_cuda_complex(
            make_cuDoubleComplex(sum.x / (double)size, 0.0)
        );
    }
}

// High-performance quantum state normalization kernel
__global__ void quantum_normalize_kernel(cuDoubleComplex *state,
                                         double *norm_squared,
                                         size_t size) {
    __shared__ double shared_sum[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Compute local norm squared
    double local_sum = 0.0;
    if (idx < size) {
        cuDoubleComplex amp = state[idx];
        local_sum = amp.x * amp.x + amp.y * amp.y;
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

    // Write block result
    if (tid == 0) {
        atomicAdd(norm_squared, shared_sum[0]);
    }
}

// Apply normalization factor
__global__ void quantum_scale_kernel(cuDoubleComplex *state,
                                     double scale,
                                     size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        state[idx].x *= scale;
        state[idx].y *= scale;
    }
}

// Inner product kernel: <a|b>
__global__ void quantum_inner_product_kernel(const cuDoubleComplex *state_a,
                                              const cuDoubleComplex *state_b,
                                              cuDoubleComplex *result,
                                              size_t size) {
    __shared__ double shared_real[256];
    __shared__ double shared_imag[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    double local_real = 0.0;
    double local_imag = 0.0;

    if (idx < size) {
        cuDoubleComplex a = state_a[idx];
        cuDoubleComplex b = state_b[idx];
        // <a|b> = conj(a) * b
        local_real = a.x * b.x + a.y * b.y;
        local_imag = a.x * b.y - a.y * b.x;
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
        atomicAdd(&result->x, shared_real[0]);
        atomicAdd(&result->y, shared_imag[0]);
    }
}

// Expectation value for diagonal observable
__global__ void quantum_expectation_kernel(const cuDoubleComplex *state,
                                           const double *observable,
                                           double *result,
                                           size_t size) {
    __shared__ double shared_sum[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    double local_sum = 0.0;
    if (idx < size) {
        cuDoubleComplex amp = state[idx];
        double prob = amp.x * amp.x + amp.y * amp.y;
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
        atomicAdd(result, shared_sum[0]);
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
