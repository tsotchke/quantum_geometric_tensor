#include "quantum_geometric/core/differential_transformer.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Constants for CUDA kernels
#define BLOCK_SIZE 256
#define MAX_THREADS_PER_BLOCK 1024
#define WARP_SIZE 32
#define SHARED_MEMORY_SIZE 48000

// Device constants
__constant__ double d_epsilon;
__constant__ double d_max_grad_norm;

// Handle for cuBLAS
static cublasHandle_t cublas_handle;

// Error checking macro
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// CUDA kernel for computing token derivatives
__global__ void compute_token_derivatives_kernel(
    const double* values,
    double* derivatives,
    size_t seq_length,
    size_t hidden_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_length * hidden_dim) return;
    
    size_t i = idx / hidden_dim;
    size_t j = idx % hidden_dim;
    
    // Central difference approximation
    double h = max(fabs(values[idx]) * 1e-4, d_epsilon);
    derivatives[idx] = (values[idx + 1] - values[idx - 1]) / (2.0 * h);
}

// CUDA kernel for differential attention scores
__global__ void differential_attention_scores_kernel(
    const double* query,
    const double* key,
    const double* query_deriv,
    const double* key_deriv,
    double* scores,
    double* score_derivs,
    size_t seq_length,
    size_t head_dim
) {
    // Get block indices
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= seq_length || col >= seq_length) return;
    
    // Shared memory for query and key vectors
    extern __shared__ double shared_mem[];
    double* shared_query = shared_mem;
    double* shared_key = shared_mem + head_dim;
    
    // Load query and key vectors into shared memory
    if (threadIdx.x < head_dim) {
        shared_query[threadIdx.x] = query[row * head_dim + threadIdx.x];
        shared_key[threadIdx.x] = key[col * head_dim + threadIdx.x];
    }
    __syncthreads();
    
    // Compute attention score and derivative
    double score = 0.0;
    double score_deriv = 0.0;
    
    for (int k = 0; k < head_dim; k++) {
        double q = shared_query[k];
        double k_val = shared_key[k];
        double q_deriv = query_deriv[row * head_dim + k];
        double k_deriv = key_deriv[col * head_dim + k];
        
        score += q * k_val;
        score_deriv += q_deriv * k_val + q * k_deriv;
    }
    
    // Scale and store results
    double scale = 1.0 / sqrt(head_dim);
    scores[row * seq_length + col] = score * scale;
    score_derivs[row * seq_length + col] = score_deriv * scale;
}

// CUDA kernel for differential softmax
__global__ void differential_softmax_kernel(
    double* values,
    double* derivatives,
    size_t seq_length
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    // Shared memory for reductions
    extern __shared__ double shared_mem[];
    double* shared_max = shared_mem;
    double* shared_sum = shared_mem + blockDim.x;
    double* shared_deriv_sum = shared_mem + 2 * blockDim.x;
    
    // Find max value for numerical stability
    double max_val = -INFINITY;
    for (int j = tid; j < seq_length; j += blockDim.x) {
        max_val = max(max_val, values[row * seq_length + j]);
    }
    shared_max[tid] = max_val;
    __syncthreads();
    
    // Reduce to find global max
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }
    max_val = shared_max[0];
    __syncthreads();
    
    // Compute exp and sums
    double local_sum = 0.0;
    double local_deriv_sum = 0.0;
    for (int j = tid; j < seq_length; j += blockDim.x) {
        int idx = row * seq_length + j;
        double exp_val = exp(values[idx] - max_val);
        values[idx] = exp_val;
        local_sum += exp_val;
        local_deriv_sum += derivatives[idx] * exp_val;
    }
    shared_sum[tid] = local_sum;
    shared_deriv_sum[tid] = local_deriv_sum;
    __syncthreads();
    
    // Reduce sums
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_deriv_sum[tid] += shared_deriv_sum[tid + stride];
        }
        __syncthreads();
    }
    double sum = shared_sum[0];
    double deriv_sum = shared_deriv_sum[0];
    __syncthreads();
    
    // Normalize values and compute derivatives
    double inv_sum = 1.0 / sum;
    for (int j = tid; j < seq_length; j += blockDim.x) {
        int idx = row * seq_length + j;
        double softmax_val = values[idx] * inv_sum;
        double softmax_deriv = derivatives[idx] * softmax_val - 
                             softmax_val * deriv_sum * inv_sum;
        
        values[idx] = softmax_val;
        derivatives[idx] = softmax_deriv;
    }
}

// Initialize CUDA resources
extern "C" void cuda_init_differential() {
    double h_epsilon = 1e-6;
    double h_max_grad_norm = 1.0;
    
    CHECK_CUDA(cudaMemcpyToSymbol(d_epsilon, &h_epsilon, sizeof(double)));
    CHECK_CUDA(cudaMemcpyToSymbol(d_max_grad_norm, &h_max_grad_norm, sizeof(double)));
    
    CHECK_CUDA(cublasCreate(&cublas_handle));
}

// Clean up CUDA resources
extern "C" void cuda_cleanup_differential() {
    cublasDestroy(cublas_handle);
}

// GPU-accelerated differential transformer forward pass
extern "C" void cuda_diff_transformer_forward(
    DiffTransformerState* state,
    const double* input,
    double* output
) {
    size_t seq_len = state->seq_length;
    size_t hidden_dim = state->hidden_dim;
    size_t total_size = seq_len * hidden_dim;
    
    // Allocate device memory
    double *d_input, *d_output, *d_values, *d_derivatives;
    CHECK_CUDA(cudaMalloc(&d_input, total_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_output, total_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_values, total_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_derivatives, total_size * sizeof(double)));
    
    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_input, input, total_size * sizeof(double), 
                         cudaMemcpyHostToDevice));
    
    // Launch kernels
    dim3 block(BLOCK_SIZE);
    dim3 grid((total_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    compute_token_derivatives_kernel<<<grid, block>>>(
        d_input, d_derivatives, seq_len, hidden_dim
    );
    
    // Process attention layers
    size_t head_dim = hidden_dim / state->num_heads;
    dim3 attn_block(BLOCK_SIZE);
    dim3 attn_grid(seq_len, seq_len);
    size_t shared_mem_size = 2 * head_dim * sizeof(double);
    
    differential_attention_scores_kernel<<<attn_grid, attn_block, shared_mem_size>>>(
        d_values, d_values,  // Using values as both query and key
        d_derivatives, d_derivatives,
        d_output, d_derivatives,
        seq_len, head_dim
    );
    
    // Apply softmax
    differential_softmax_kernel<<<seq_len, BLOCK_SIZE, 3 * BLOCK_SIZE * sizeof(double)>>>(
        d_output, d_derivatives, seq_len
    );
    
    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(output, d_output, total_size * sizeof(double),
                         cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(state->derivatives, d_derivatives, total_size * sizeof(double),
                         cudaMemcpyDeviceToHost));
    
    // Cleanup
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_values));
    CHECK_CUDA(cudaFree(d_derivatives));
}
