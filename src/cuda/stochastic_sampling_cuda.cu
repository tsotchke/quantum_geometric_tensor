/**
 * @file stochastic_sampling_cuda.cu
 * @brief CUDA implementation of neural network operations for stochastic sampling
 */

#include "quantum_geometric/learning/stochastic_sampling.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA kernel for GELU activation
__device__ double gelu(double x) {
    const double sqrt_2_pi = 2.506628275;
    return x * 0.5 * (1.0 + tanh(sqrt(2.0/M_PI) * (x + 0.044715 * pow(x, 3))));
}

// CUDA kernel for forward pass
__global__ void forward_kernel(
    const double* input,
    const double* weights,
    const double* biases,
    double* output,
    size_t input_dim,
    size_t output_dim,
    size_t batch_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size * output_dim) return;
    
    int batch_idx = tid / output_dim;
    int output_idx = tid % output_dim;
    
    double sum = biases[output_idx];
    for (int i = 0; i < input_dim; i++) {
        sum += weights[output_idx * input_dim + i] * input[batch_idx * input_dim + i];
    }
    output[tid] = gelu(sum);
}

// CUDA kernel for gradient computation
__global__ void gradient_kernel(
    const double* input,
    const double* weights,
    const double* d_output,
    double* d_input,
    double* d_weights,
    double* d_biases,
    size_t input_dim,
    size_t output_dim,
    size_t batch_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size * output_dim) return;
    
    int batch_idx = tid / output_dim;
    int output_idx = tid % output_dim;
    
    // Compute gradients for weights and biases
    for (int i = 0; i < input_dim; i++) {
        atomicAdd(&d_weights[output_idx * input_dim + i],
                 d_output[tid] * input[batch_idx * input_dim + i]);
    }
    atomicAdd(&d_biases[output_idx], d_output[tid]);
    
    // Compute gradients for inputs
    for (int i = 0; i < input_dim; i++) {
        atomicAdd(&d_input[batch_idx * input_dim + i],
                 d_output[tid] * weights[output_idx * input_dim + i]);
    }
}

// CUDA implementation of neural network forward pass
extern "C" int cuda_forward(
    const double* input,
    const double* weights,
    const double* biases,
    double* output,
    size_t input_dim,
    size_t output_dim,
    size_t batch_size
) {
    // Allocate device memory
    double *d_input, *d_weights, *d_biases, *d_output;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(double));
    cudaMalloc(&d_weights, output_dim * input_dim * sizeof(double));
    cudaMalloc(&d_biases, output_dim * sizeof(double));
    cudaMalloc(&d_output, batch_size * output_dim * sizeof(double));
    
    // Copy data to device
    cudaMemcpy(d_input, input, batch_size * input_dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, output_dim * input_dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, biases, output_dim * sizeof(double), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int block_size = 256;
    int num_blocks = (batch_size * output_dim + block_size - 1) / block_size;
    forward_kernel<<<num_blocks, block_size>>>(
        d_input, d_weights, d_biases, d_output,
        input_dim, output_dim, batch_size
    );
    
    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * output_dim * sizeof(double), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_biases);
    cudaFree(d_output);
    
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// CUDA implementation of gradient computation
extern "C" int cuda_backward(
    const double* input,
    const double* weights,
    const double* d_output,
    double* d_input,
    double* d_weights,
    double* d_biases,
    size_t input_dim,
    size_t output_dim,
    size_t batch_size
) {
    // Allocate device memory
    double *d_input_gpu, *d_weights_gpu, *d_output_gpu;
    double *d_input_grad_gpu, *d_weights_grad_gpu, *d_biases_grad_gpu;
    
    cudaMalloc(&d_input_gpu, batch_size * input_dim * sizeof(double));
    cudaMalloc(&d_weights_gpu, output_dim * input_dim * sizeof(double));
    cudaMalloc(&d_output_gpu, batch_size * output_dim * sizeof(double));
    cudaMalloc(&d_input_grad_gpu, batch_size * input_dim * sizeof(double));
    cudaMalloc(&d_weights_grad_gpu, output_dim * input_dim * sizeof(double));
    cudaMalloc(&d_biases_grad_gpu, output_dim * sizeof(double));
    
    // Copy data to device
    cudaMemcpy(d_input_gpu, input, batch_size * input_dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights_gpu, weights, output_dim * input_dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_gpu, d_output, batch_size * output_dim * sizeof(double), cudaMemcpyHostToDevice);
    
    // Initialize gradients to zero
    cudaMemset(d_input_grad_gpu, 0, batch_size * input_dim * sizeof(double));
    cudaMemset(d_weights_grad_gpu, 0, output_dim * input_dim * sizeof(double));
    cudaMemset(d_biases_grad_gpu, 0, output_dim * sizeof(double));
    
    // Launch kernel
    int block_size = 256;
    int num_blocks = (batch_size * output_dim + block_size - 1) / block_size;
    gradient_kernel<<<num_blocks, block_size>>>(
        d_input_gpu, d_weights_gpu, d_output_gpu,
        d_input_grad_gpu, d_weights_grad_gpu, d_biases_grad_gpu,
        input_dim, output_dim, batch_size
    );
    
    // Copy results back to host
    cudaMemcpy(d_input, d_input_grad_gpu, batch_size * input_dim * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(d_weights, d_weights_grad_gpu, output_dim * input_dim * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(d_biases, d_biases_grad_gpu, output_dim * sizeof(double), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_input_gpu);
    cudaFree(d_weights_gpu);
    cudaFree(d_output_gpu);
    cudaFree(d_input_grad_gpu);
    cudaFree(d_weights_grad_gpu);
    cudaFree(d_biases_grad_gpu);
    
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
