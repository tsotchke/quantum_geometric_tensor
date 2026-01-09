#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>
#include <stdio.h>
#include "quantum_geometric/core/error_codes.h"
#include "quantum_geometric/core/hierarchical_matrix.h"

// =============================================================================
// CUDA Device Constants
// =============================================================================

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// =============================================================================
// CUDA Kernels
// =============================================================================

/**
 * @brief Apply phase rotation to quantum state amplitudes
 *
 * Each thread processes one amplitude, applying e^(i*angle) rotation
 * to states where the target qubit is |1⟩.
 */
__global__ void phase_estimation_kernel(
    cuDoubleComplex* state,
    const unsigned long long qubit_mask,
    const double cos_angle,
    const double sin_angle,
    const size_t dim)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;

    // Only apply rotation when qubit is |1⟩
    if (idx & qubit_mask) {
        cuDoubleComplex amp = state[idx];
        cuDoubleComplex rotated;

        // Multiply by e^(i*angle) = cos(angle) + i*sin(angle)
        rotated.x = amp.x * cos_angle - amp.y * sin_angle;
        rotated.y = amp.x * sin_angle + amp.y * cos_angle;

        state[idx] = rotated;
    }
}

/**
 * @brief Compute norm squared with block-level reduction
 */
__global__ void compute_norm_squared_kernel(
    const cuDoubleComplex* state,
    double* partial_sums,
    const size_t dim)
{
    __shared__ double sdata[BLOCK_SIZE];

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t tid = threadIdx.x;

    // Load and compute local contribution
    double local_sum = 0.0;
    if (idx < dim) {
        cuDoubleComplex amp = state[idx];
        local_sum = amp.x * amp.x + amp.y * amp.y;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    // Block reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

/**
 * @brief Final reduction for norm squared
 */
__global__ void final_reduction_kernel(
    const double* partial_sums,
    double* result,
    const size_t num_blocks)
{
    __shared__ double sdata[BLOCK_SIZE];
    size_t tid = threadIdx.x;

    double local_sum = 0.0;
    for (size_t i = tid; i < num_blocks; i += blockDim.x) {
        local_sum += partial_sums[i];
    }
    sdata[tid] = local_sum;
    __syncthreads();

    // Block reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *result = sdata[0];
    }
}

/**
 * @brief Normalize state vector
 */
__global__ void normalize_state_kernel(
    cuDoubleComplex* state,
    const double inv_norm,
    const size_t dim)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;

    state[idx].x *= inv_norm;
    state[idx].y *= inv_norm;
}

/**
 * @brief Compute Berry phase gradient contributions
 */
__global__ void berry_phase_gradient_kernel(
    const cuDoubleComplex* data,
    cuDoubleComplex* grad,
    const size_t rows,
    const size_t cols)
{
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= rows || j >= cols) return;

    size_t idx = i * cols + j;
    cuDoubleComplex psi = data[idx];
    double phase_contrib = 0.0;

    // Berry phase from row neighbor
    if (i > 0) {
        cuDoubleComplex psi_prev = data[(i - 1) * cols + j];
        // overlap = conj(psi_prev) * psi
        cuDoubleComplex overlap;
        overlap.x = psi_prev.x * psi.x + psi_prev.y * psi.y;
        overlap.y = psi_prev.x * psi.y - psi_prev.y * psi.x;

        double overlap_abs = sqrt(overlap.x * overlap.x + overlap.y * overlap.y);
        if (overlap_abs > 1e-15) {
            // Normalize and take log
            double norm_re = overlap.x / overlap_abs;
            double norm_im = overlap.y / overlap_abs;
            phase_contrib += atan2(norm_im, norm_re);
        }
    }

    // Berry phase from column neighbor
    if (j > 0) {
        cuDoubleComplex psi_prev = data[i * cols + j - 1];
        cuDoubleComplex overlap;
        overlap.x = psi_prev.x * psi.x + psi_prev.y * psi.y;
        overlap.y = psi_prev.x * psi.y - psi_prev.y * psi.x;

        double overlap_abs = sqrt(overlap.x * overlap.x + overlap.y * overlap.y);
        if (overlap_abs > 1e-15) {
            double norm_re = overlap.x / overlap_abs;
            double norm_im = overlap.y / overlap_abs;
            phase_contrib += atan2(norm_im, norm_re);
        }
    }

    // Add phase contribution to gradient
    grad[idx].x += phase_contrib * psi.x;
    grad[idx].y += phase_contrib * psi.y;
}

/**
 * @brief Compute quantum metric tensor
 */
__global__ void quantum_metric_kernel(
    const cuDoubleComplex* data,
    double* metric,
    const size_t rows,
    const size_t cols)
{
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= rows || j >= cols) return;

    size_t idx = i * cols + j;
    cuDoubleComplex psi = data[idx];
    cuDoubleComplex dpsi_x = make_cuDoubleComplex(0.0, 0.0);
    cuDoubleComplex dpsi_y = make_cuDoubleComplex(0.0, 0.0);

    // Compute x-derivative (central difference)
    if (i > 0 && i < rows - 1) {
        cuDoubleComplex next = data[(i + 1) * cols + j];
        cuDoubleComplex prev = data[(i - 1) * cols + j];
        dpsi_x.x = (next.x - prev.x) * 0.5;
        dpsi_x.y = (next.y - prev.y) * 0.5;
    } else if (i == 0 && rows > 1) {
        cuDoubleComplex next = data[cols + j];
        dpsi_x.x = next.x - psi.x;
        dpsi_x.y = next.y - psi.y;
    } else if (i == rows - 1 && rows > 1) {
        cuDoubleComplex prev = data[(rows - 2) * cols + j];
        dpsi_x.x = psi.x - prev.x;
        dpsi_x.y = psi.y - prev.y;
    }

    // Compute y-derivative (central difference)
    if (j > 0 && j < cols - 1) {
        cuDoubleComplex next = data[i * cols + j + 1];
        cuDoubleComplex prev = data[i * cols + j - 1];
        dpsi_y.x = (next.x - prev.x) * 0.5;
        dpsi_y.y = (next.y - prev.y) * 0.5;
    } else if (j == 0 && cols > 1) {
        cuDoubleComplex next = data[i * cols + 1];
        dpsi_y.x = next.x - psi.x;
        dpsi_y.y = next.y - psi.y;
    } else if (j == cols - 1 && cols > 1) {
        cuDoubleComplex prev = data[i * cols + cols - 2];
        dpsi_y.x = psi.x - prev.x;
        dpsi_y.y = psi.y - prev.y;
    }

    // Quantum metric: g = Re[<d_psi|d_psi>] = |d_psi_x|^2 + |d_psi_y|^2
    double metric_val = dpsi_x.x * dpsi_x.x + dpsi_x.y * dpsi_x.y +
                        dpsi_y.x * dpsi_y.x + dpsi_y.y * dpsi_y.y;
    metric[idx] = metric_val;
}

/**
 * @brief Compute Berry curvature using Wilson loop (plaquette method)
 */
__global__ void berry_curvature_kernel(
    const cuDoubleComplex* data,
    double* curvature,
    const size_t rows,
    const size_t cols)
{
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= rows || j >= cols) return;

    size_t idx = i * cols + j;

    // Initialize to zero for boundary cases
    if (i >= rows - 1 || j >= cols - 1) {
        curvature[idx] = 0.0;
        return;
    }

    // Plaquette corners
    cuDoubleComplex psi_00 = data[i * cols + j];
    cuDoubleComplex psi_10 = data[(i + 1) * cols + j];
    cuDoubleComplex psi_01 = data[i * cols + j + 1];
    cuDoubleComplex psi_11 = data[(i + 1) * cols + j + 1];

    // Wilson loop links U_ij = <psi_i | psi_j> = conj(psi_i) * psi_j
    // U_01 = <psi_00 | psi_10>
    cuDoubleComplex U_01;
    U_01.x = psi_00.x * psi_10.x + psi_00.y * psi_10.y;
    U_01.y = psi_00.x * psi_10.y - psi_00.y * psi_10.x;

    // U_12 = <psi_10 | psi_11>
    cuDoubleComplex U_12;
    U_12.x = psi_10.x * psi_11.x + psi_10.y * psi_11.y;
    U_12.y = psi_10.x * psi_11.y - psi_10.y * psi_11.x;

    // U_23 = <psi_11 | psi_01>
    cuDoubleComplex U_23;
    U_23.x = psi_11.x * psi_01.x + psi_11.y * psi_01.y;
    U_23.y = psi_11.x * psi_01.y - psi_11.y * psi_01.x;

    // U_30 = <psi_01 | psi_00>
    cuDoubleComplex U_30;
    U_30.x = psi_01.x * psi_00.x + psi_01.y * psi_00.y;
    U_30.y = psi_01.x * psi_00.y - psi_01.y * psi_00.x;

    // Wilson loop product W = U_01 * U_12 * U_23 * U_30
    cuDoubleComplex W1, W2, W;
    W1.x = U_01.x * U_12.x - U_01.y * U_12.y;
    W1.y = U_01.x * U_12.y + U_01.y * U_12.x;

    W2.x = U_23.x * U_30.x - U_23.y * U_30.y;
    W2.y = U_23.x * U_30.y + U_23.y * U_30.x;

    W.x = W1.x * W2.x - W1.y * W2.y;
    W.y = W1.x * W2.y + W1.y * W2.x;

    // Berry curvature = Im[log(W)] = arg(W)
    double W_abs = sqrt(W.x * W.x + W.y * W.y);
    if (W_abs > 1e-15) {
        curvature[idx] = atan2(W.y, W.x);
    } else {
        curvature[idx] = 0.0;
    }
}

// =============================================================================
// C Interface Functions
// =============================================================================

extern "C" {

int cuda_phase_estimation_dispatch(
    void* device_handle,
    void* stream,
    double _Complex* state,
    size_t q,
    size_t dim)
{
    if (!state || dim == 0) {
        return QGT_ERROR_INVALID_PARAMETER;
    }

    cudaStream_t cuda_stream = stream ? (cudaStream_t)stream : 0;
    cudaError_t err;

    // Allocate device memory
    cuDoubleComplex* d_state;
    size_t buffer_size = dim * sizeof(cuDoubleComplex);
    err = cudaMalloc(&d_state, buffer_size);
    if (err != cudaSuccess) {
        return QGT_ERROR_OUT_OF_MEMORY;
    }

    // Copy state to device (cast is safe since double complex is binary compatible)
    err = cudaMemcpyAsync(d_state, state, buffer_size, cudaMemcpyHostToDevice, cuda_stream);
    if (err != cudaSuccess) {
        cudaFree(d_state);
        return QGT_ERROR_RUNTIME;
    }

    // Calculate phase parameters
    unsigned long long qubit_mask = 1ULL << q;
    double angle = 2.0 * M_PI / (double)(1ULL << (q + 1));
    double cos_angle = cos(angle);
    double sin_angle = sin(angle);

    // Launch kernel
    size_t num_blocks = (dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    phase_estimation_kernel<<<num_blocks, BLOCK_SIZE, 0, cuda_stream>>>(
        d_state, qubit_mask, cos_angle, sin_angle, dim);

    // Check for kernel errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_state);
        return QGT_ERROR_RUNTIME;
    }

    // Copy result back
    err = cudaMemcpyAsync(state, d_state, buffer_size, cudaMemcpyDeviceToHost, cuda_stream);
    if (err != cudaSuccess) {
        cudaFree(d_state);
        return QGT_ERROR_RUNTIME;
    }

    // Synchronize stream
    cudaStreamSynchronize(cuda_stream);

    // Cleanup
    cudaFree(d_state);

    return QGT_SUCCESS;
}

int cuda_hmatrix_update_dispatch(
    void* device_handle,
    void* stream,
    HierarchicalMatrix* mat)
{
    if (!mat || !mat->data) {
        return QGT_ERROR_INVALID_PARAMETER;
    }

    cudaStream_t cuda_stream = stream ? (cudaStream_t)stream : 0;
    cudaError_t err;

    size_t size = mat->rows * mat->cols;
    if (size == 0) return QGT_SUCCESS;

    size_t buffer_size = size * sizeof(cuDoubleComplex);

    // Allocate device memory
    cuDoubleComplex* d_data;
    err = cudaMalloc(&d_data, buffer_size);
    if (err != cudaSuccess) {
        return QGT_ERROR_OUT_OF_MEMORY;
    }

    // Copy data to device
    err = cudaMemcpyAsync(d_data, mat->data, buffer_size, cudaMemcpyHostToDevice, cuda_stream);
    if (err != cudaSuccess) {
        cudaFree(d_data);
        return QGT_ERROR_RUNTIME;
    }

    // Compute norm squared with reduction
    size_t num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    double* d_partial_sums;
    double* d_norm_sq;

    err = cudaMalloc(&d_partial_sums, num_blocks * sizeof(double));
    if (err != cudaSuccess) {
        cudaFree(d_data);
        return QGT_ERROR_OUT_OF_MEMORY;
    }

    err = cudaMalloc(&d_norm_sq, sizeof(double));
    if (err != cudaSuccess) {
        cudaFree(d_data);
        cudaFree(d_partial_sums);
        return QGT_ERROR_OUT_OF_MEMORY;
    }

    // First pass: block-level reduction
    compute_norm_squared_kernel<<<num_blocks, BLOCK_SIZE, 0, cuda_stream>>>(
        d_data, d_partial_sums, size);

    // Second pass: final reduction
    final_reduction_kernel<<<1, BLOCK_SIZE, 0, cuda_stream>>>(
        d_partial_sums, d_norm_sq, num_blocks);

    // Copy norm back
    double norm_sq;
    cudaMemcpyAsync(&norm_sq, d_norm_sq, sizeof(double), cudaMemcpyDeviceToHost, cuda_stream);
    cudaStreamSynchronize(cuda_stream);

    // Normalize if needed
    if (norm_sq > 1e-15 && fabs(norm_sq - 1.0) > 1e-10) {
        double inv_norm = 1.0 / sqrt(norm_sq);
        normalize_state_kernel<<<num_blocks, BLOCK_SIZE, 0, cuda_stream>>>(
            d_data, inv_norm, size);
    }

    // Compute Berry phase gradient if gradient exists
    if (mat->grad) {
        cuDoubleComplex* d_grad;
        err = cudaMalloc(&d_grad, buffer_size);
        if (err == cudaSuccess) {
            cudaMemcpyAsync(d_grad, mat->grad, buffer_size, cudaMemcpyHostToDevice, cuda_stream);

            dim3 blockDim(16, 16);
            dim3 gridDim((mat->cols + blockDim.x - 1) / blockDim.x,
                        (mat->rows + blockDim.y - 1) / blockDim.y);

            berry_phase_gradient_kernel<<<gridDim, blockDim, 0, cuda_stream>>>(
                d_data, d_grad, mat->rows, mat->cols);

            cudaMemcpyAsync(mat->grad, d_grad, buffer_size, cudaMemcpyDeviceToHost, cuda_stream);
            cudaFree(d_grad);
        }
    }

    // Copy data back
    cudaMemcpyAsync(mat->data, d_data, buffer_size, cudaMemcpyDeviceToHost, cuda_stream);
    cudaStreamSynchronize(cuda_stream);

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_partial_sums);
    cudaFree(d_norm_sq);

    return QGT_SUCCESS;
}

int cuda_quantum_metric_dispatch(
    void* device_handle,
    void* stream,
    const HierarchicalMatrix* mat,
    double* metric)
{
    if (!mat || !mat->data || !metric) {
        return QGT_ERROR_INVALID_PARAMETER;
    }

    cudaStream_t cuda_stream = stream ? (cudaStream_t)stream : 0;
    cudaError_t err;

    size_t size = mat->rows * mat->cols;
    if (size == 0) return QGT_SUCCESS;

    size_t data_buffer_size = size * sizeof(cuDoubleComplex);
    size_t metric_buffer_size = size * sizeof(double);

    // Allocate device memory
    cuDoubleComplex* d_data;
    double* d_metric;

    err = cudaMalloc(&d_data, data_buffer_size);
    if (err != cudaSuccess) {
        return QGT_ERROR_OUT_OF_MEMORY;
    }

    err = cudaMalloc(&d_metric, metric_buffer_size);
    if (err != cudaSuccess) {
        cudaFree(d_data);
        return QGT_ERROR_OUT_OF_MEMORY;
    }

    // Copy data to device
    err = cudaMemcpyAsync(d_data, mat->data, data_buffer_size, cudaMemcpyHostToDevice, cuda_stream);
    if (err != cudaSuccess) {
        cudaFree(d_data);
        cudaFree(d_metric);
        return QGT_ERROR_RUNTIME;
    }

    // Launch quantum metric kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((mat->cols + blockDim.x - 1) / blockDim.x,
                 (mat->rows + blockDim.y - 1) / blockDim.y);

    quantum_metric_kernel<<<gridDim, blockDim, 0, cuda_stream>>>(
        d_data, d_metric, mat->rows, mat->cols);

    // Check for errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_data);
        cudaFree(d_metric);
        return QGT_ERROR_RUNTIME;
    }

    // Copy result back
    err = cudaMemcpyAsync(metric, d_metric, metric_buffer_size, cudaMemcpyDeviceToHost, cuda_stream);
    if (err != cudaSuccess) {
        cudaFree(d_data);
        cudaFree(d_metric);
        return QGT_ERROR_RUNTIME;
    }

    cudaStreamSynchronize(cuda_stream);

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_metric);

    return QGT_SUCCESS;
}

int cuda_berry_curvature_dispatch(
    void* device_handle,
    void* stream,
    const HierarchicalMatrix* mat,
    double* curvature)
{
    if (!mat || !mat->data || !curvature) {
        return QGT_ERROR_INVALID_PARAMETER;
    }

    cudaStream_t cuda_stream = stream ? (cudaStream_t)stream : 0;
    cudaError_t err;

    size_t size = mat->rows * mat->cols;
    if (size == 0) return QGT_SUCCESS;

    size_t data_buffer_size = size * sizeof(cuDoubleComplex);
    size_t curvature_buffer_size = size * sizeof(double);

    // Allocate device memory
    cuDoubleComplex* d_data;
    double* d_curvature;

    err = cudaMalloc(&d_data, data_buffer_size);
    if (err != cudaSuccess) {
        return QGT_ERROR_OUT_OF_MEMORY;
    }

    err = cudaMalloc(&d_curvature, curvature_buffer_size);
    if (err != cudaSuccess) {
        cudaFree(d_data);
        return QGT_ERROR_OUT_OF_MEMORY;
    }

    // Initialize curvature to zero
    cudaMemsetAsync(d_curvature, 0, curvature_buffer_size, cuda_stream);

    // Copy data to device
    err = cudaMemcpyAsync(d_data, mat->data, data_buffer_size, cudaMemcpyHostToDevice, cuda_stream);
    if (err != cudaSuccess) {
        cudaFree(d_data);
        cudaFree(d_curvature);
        return QGT_ERROR_RUNTIME;
    }

    // Launch Berry curvature kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((mat->cols + blockDim.x - 1) / blockDim.x,
                 (mat->rows + blockDim.y - 1) / blockDim.y);

    berry_curvature_kernel<<<gridDim, blockDim, 0, cuda_stream>>>(
        d_data, d_curvature, mat->rows, mat->cols);

    // Check for errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_data);
        cudaFree(d_curvature);
        return QGT_ERROR_RUNTIME;
    }

    // Copy result back
    err = cudaMemcpyAsync(curvature, d_curvature, curvature_buffer_size, cudaMemcpyDeviceToHost, cuda_stream);
    if (err != cudaSuccess) {
        cudaFree(d_data);
        cudaFree(d_curvature);
        return QGT_ERROR_RUNTIME;
    }

    cudaStreamSynchronize(cuda_stream);

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_curvature);

    return QGT_SUCCESS;
}

} // extern "C"
