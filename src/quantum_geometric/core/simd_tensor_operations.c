/**
 * @file simd_tensor_operations.c
 * @brief High-performance SIMD-accelerated tensor operations
 */

#include "quantum_geometric/core/simd_tensor_operations.h"
#include "quantum_geometric/core/error_handling.h"
#include "quantum_geometric/hardware/hardware_capabilities.h"
#include <stdlib.h>
#include <string.h>

#ifdef __x86_64__
#include <immintrin.h>
#else
#include <arm_neon.h>
#endif

#define MAX_TENSOR_RANK 16
#define min(a,b) ((a) < (b) ? (a) : (b))

// Default cache parameters if none provided
static const CacheOptParams DEFAULT_PARAMS = {
    .block_size = BLOCK_SIZE_L2,
    .prefetch_dist = 16,
    .vector_width = 8,
    .align_size = 32
};

/**
 * @brief Get cache parameters, using defaults if NULL
 */
static inline const CacheOptParams* get_params(const CacheOptParams* params) {
    return params ? params : &DEFAULT_PARAMS;
}

int matrix_multiply_avx512(const double* A,
                         const double* B,
                         double* C,
                         size_t M,
                         size_t N,
                         size_t K,
                         const CacheOptParams* params) {
    // Input validation
    if (!A || !B || !C) {
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    if (M == 0 || N == 0 || K == 0) {
        return QGT_ERROR_INVALID_PARAMETER;
    }

    // Get cache parameters
    const CacheOptParams* cache_params = get_params(params);
    const size_t block_size = cache_params->block_size;
    const size_t prefetch_dist = cache_params->prefetch_dist;
    const size_t vector_width = cache_params->vector_width;

    // Initialize output matrix
    memset(C, 0, M * N * sizeof(double));

    // Cache blocking
    for (size_t ii = 0; ii < M; ii += block_size) {
        for (size_t jj = 0; jj < N; jj += block_size) {
            for (size_t kk = 0; kk < K; kk += block_size) {
                size_t max_i = min(ii + block_size, M);
                size_t max_j = min(jj + block_size, N);
                size_t max_k = min(kk + block_size, K);

                // Process block
                for (size_t i = ii; i < max_i; i += vector_width) {
                    for (size_t j = jj; j < max_j; j += vector_width) {
                        #ifdef __x86_64__
                        __m512d c[vector_width];
                        
                        // Load accumulators
                        for (size_t u = 0; u < vector_width; u++) {
                            c[u] = _mm512_load_pd(&C[(i + u) * N + j]);
                        }
                        
                        // Compute block
                        for (size_t k = kk; k < max_k; k++) {
                            _mm_prefetch(&A[(i + vector_width) * K + k + prefetch_dist],
                                       _MM_HINT_T0);
                            _mm_prefetch(&B[(k + prefetch_dist) * N + j],
                                       _MM_HINT_T0);
                            
                            __m512d b = _mm512_load_pd(&B[k * N + j]);
                            
                            for (size_t u = 0; u < vector_width; u++) {
                                __m512d a = _mm512_set1_pd(A[(i + u) * K + k]);
                                c[u] = _mm512_fmadd_pd(a, b, c[u]);
                            }
                        }
                        
                        // Store results
                        for (size_t u = 0; u < vector_width; u++) {
                            _mm512_store_pd(&C[(i + u) * N + j], c[u]);
                        }
                        #else
                        // ARM NEON implementation
                        float64x2_t c[vector_width];
                        
                        // Load accumulators
                        for (size_t u = 0; u < vector_width; u++) {
                            c[u] = vld1q_f64(&C[(i + u) * N + j]);
                        }
                        
                        // Compute block
                        for (size_t k = kk; k < max_k; k++) {
                            __builtin_prefetch(&A[(i + vector_width) * K + k + prefetch_dist]);
                            __builtin_prefetch(&B[(k + prefetch_dist) * N + j]);
                            
                            float64x2_t b = vld1q_f64(&B[k * N + j]);
                            
                            for (size_t u = 0; u < vector_width; u++) {
                                float64x2_t a = vdupq_n_f64(A[(i + u) * K + k]);
                                c[u] = vfmaq_f64(c[u], a, b);
                            }
                        }
                        
                        // Store results
                        for (size_t u = 0; u < vector_width; u++) {
                            vst1q_f64(&C[(i + u) * N + j], c[u]);
                        }
                        #endif
                    }
                }
            }
        }
    }

    return QGT_SUCCESS;
}

int tensor_contract_avx512(const double* A,
                         const double* B,
                         double* C,
                         const size_t* dims,
                         size_t rank,
                         const CacheOptParams* params) {
    // Input validation
    if (!A || !B || !C || !dims) {
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    if (rank == 0 || rank > MAX_TENSOR_RANK) {
        return QGT_ERROR_INVALID_PARAMETER;
    }

    // Get cache parameters
    const CacheOptParams* cache_params = get_params(params);
    const size_t block_size = cache_params->block_size;
    const size_t vector_width = cache_params->vector_width;

    // Compute sizes
    size_t total_size = 1;
    size_t contract_size = 1;
    for (size_t i = 0; i < rank; i++) {
        total_size *= dims[i];
        if (i < rank/2) {
            contract_size *= dims[i];
        }
    }
    
    size_t batch_size = total_size / (contract_size * contract_size);
    
    // Process in blocks
    for (size_t batch = 0; batch < batch_size; batch++) {
        for (size_t block_start = 0; block_start < contract_size; 
             block_start += block_size) {
            
            size_t current_block_size = min(block_size,
                                          contract_size - block_start);
            size_t offset = batch * contract_size * contract_size + block_start;
            
            // Process block
            #ifdef __x86_64__
            for (size_t i = 0; i < current_block_size; i += vector_width) {
                __m512d sum = _mm512_setzero_pd();
                
                if (i + vector_width < current_block_size) {
                    _mm_prefetch(&A[offset + i + vector_width], _MM_HINT_T0);
                    _mm_prefetch(&B[offset + i + vector_width], _MM_HINT_T0);
                }
                
                __m512d a_vec = _mm512_load_pd(&A[offset + i]);
                __m512d b_vec = _mm512_load_pd(&B[offset + i]);
                sum = _mm512_fmadd_pd(a_vec, b_vec, sum);
                
                _mm512_store_pd(&C[offset + i], sum);
            }
            #else
            for (size_t i = 0; i < current_block_size; i += vector_width) {
                float64x2_t sum = vdupq_n_f64(0.0);
                
                if (i + vector_width < current_block_size) {
                    __builtin_prefetch(&A[offset + i + vector_width]);
                    __builtin_prefetch(&B[offset + i + vector_width]);
                }
                
                float64x2_t a_vec = vld1q_f64(&A[offset + i]);
                float64x2_t b_vec = vld1q_f64(&B[offset + i]);
                sum = vfmaq_f64(sum, a_vec, b_vec);
                
                vst1q_f64(&C[offset + i], sum);
            }
            #endif
        }
    }
    
    return QGT_SUCCESS;
}

int tensor_elementwise_avx512(const double* A,
                            const double* B,
                            double* C,
                            size_t size,
                            ElementwiseOp op,
                            const CacheOptParams* params) {
    // Input validation
    if (!A || !B || !C) {
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    if (size == 0) {
        return QGT_ERROR_INVALID_PARAMETER;
    }

    // Get cache parameters
    const CacheOptParams* cache_params = get_params(params);
    const size_t vector_width = cache_params->vector_width;
    
    size_t vec_size = size - (size % vector_width);
    
    for (size_t i = 0; i < vec_size; i += vector_width) {
        #ifdef __x86_64__
        __m512d a = _mm512_load_pd(&A[i]);
        __m512d b = _mm512_load_pd(&B[i]);
        __m512d result;
        
        switch (op) {
            case OP_ADD:
                result = _mm512_add_pd(a, b);
                break;
            case OP_SUBTRACT:
                result = _mm512_sub_pd(a, b);
                break;
            case OP_MULTIPLY:
                result = _mm512_mul_pd(a, b);
                break;
            case OP_DIVIDE:
                result = _mm512_div_pd(a, b);
                break;
            case OP_MAXIMUM:
                result = _mm512_max_pd(a, b);
                break;
            case OP_MINIMUM:
                result = _mm512_min_pd(a, b);
                break;
        }
        
        _mm512_store_pd(&C[i], result);
        #else
        float64x2_t a = vld1q_f64(&A[i]);
        float64x2_t b = vld1q_f64(&B[i]);
        float64x2_t result;
        
        switch (op) {
            case OP_ADD:
                result = vaddq_f64(a, b);
                break;
            case OP_SUBTRACT:
                result = vsubq_f64(a, b);
                break;
            case OP_MULTIPLY:
                result = vmulq_f64(a, b);
                break;
            case OP_DIVIDE:
                result = vdivq_f64(a, b);
                break;
            case OP_MAXIMUM:
                result = vmaxq_f64(a, b);
                break;
            case OP_MINIMUM:
                result = vminq_f64(a, b);
                break;
        }
        
        vst1q_f64(&C[i], result);
        #endif
    }
    
    // Handle remaining elements
    for (size_t i = vec_size; i < size; i++) {
        switch (op) {
            case OP_ADD:
                C[i] = A[i] + B[i];
                break;
            case OP_SUBTRACT:
                C[i] = A[i] - B[i];
                break;
            case OP_MULTIPLY:
                C[i] = A[i] * B[i];
                break;
            case OP_DIVIDE:
                C[i] = A[i] / B[i];
                break;
            case OP_MAXIMUM:
                C[i] = A[i] > B[i] ? A[i] : B[i];
                break;
            case OP_MINIMUM:
                C[i] = A[i] < B[i] ? A[i] : B[i];
                break;
        }
    }
    return QGT_SUCCESS;
}

int tensor_reduce_avx512(const double* A,
                        size_t size,
                        ReductionOp op,
                        double* result,
                        const CacheOptParams* params) {
    // Input validation
    if (!A || !result) {
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    if (size == 0) {
        return QGT_ERROR_INVALID_PARAMETER;
    }

    // Get cache parameters
    const CacheOptParams* cache_params = get_params(params);
    const size_t vector_width = cache_params->vector_width;

    // Allocate temporary buffer
    double* temp = (double*)malloc(size * sizeof(double));
    if (!temp) {
        return QGT_ERROR_MEMORY_ALLOCATION;
    }
    memcpy(temp, A, size * sizeof(double));

    // Hierarchical reduction
    size_t chunk_size = size;
    while (chunk_size > 1) {
        size_t new_size = chunk_size / 2;
        
        for (size_t i = 0; i < new_size; i += vector_width) {
            #ifdef __x86_64__
            __m512d a = _mm512_load_pd(&temp[i * 2]);
            __m512d b = _mm512_load_pd(&temp[i * 2 + vector_width]);
            __m512d r;

            switch (op) {
                case REDUCE_SUM:
                    r = _mm512_add_pd(a, b);
                    break;
                case REDUCE_PRODUCT:
                    r = _mm512_mul_pd(a, b);
                    break;
                case REDUCE_MAXIMUM:
                    r = _mm512_max_pd(a, b);
                    break;
                case REDUCE_MINIMUM:
                    r = _mm512_min_pd(a, b);
                    break;
            }
            
            _mm512_store_pd(&temp[i], r);
            #else
            float64x2_t a = vld1q_f64(&temp[i * 2]);
            float64x2_t b = vld1q_f64(&temp[i * 2 + vector_width]);
            float64x2_t r;

            switch (op) {
                case REDUCE_SUM:
                    r = vaddq_f64(a, b);
                    break;
                case REDUCE_PRODUCT:
                    r = vmulq_f64(a, b);
                    break;
                case REDUCE_MAXIMUM:
                    r = vmaxq_f64(a, b);
                    break;
                case REDUCE_MINIMUM:
                    r = vminq_f64(a, b);
                    break;
            }
            
            vst1q_f64(&temp[i], r);
            #endif
        }
        
        chunk_size = new_size;
    }

    *result = temp[0];
    free(temp);
    return QGT_SUCCESS;
}

int tensor_conv_avx512(const double* input,
                      const double* kernel,
                      double* output,
                      const size_t* input_dims,
                      const size_t* kernel_dims,
                      size_t rank,
                      const CacheOptParams* params) {
    // Input validation
    if (!input || !kernel || !output || !input_dims || !kernel_dims) {
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    if (rank == 0 || rank > MAX_TENSOR_RANK) {
        return QGT_ERROR_INVALID_PARAMETER;
    }

    // Get cache parameters
    const CacheOptParams* cache_params = get_params(params);
    const size_t block_size = cache_params->block_size;
    const size_t vector_width = cache_params->vector_width;

    // Compute output dimensions
    size_t output_dims[MAX_TENSOR_RANK];
    for (size_t i = 0; i < rank; i++) {
        output_dims[i] = input_dims[i] - kernel_dims[i] + 1;
    }
    
    // Compute total sizes
    size_t total_input = 1;
    size_t total_kernel = 1;
    size_t total_output = 1;
    for (size_t i = 0; i < rank; i++) {
        total_input *= input_dims[i];
        total_kernel *= kernel_dims[i];
        total_output *= output_dims[i];
    }

    // Process in blocks
    for (size_t out_pos = 0; out_pos < total_output; out_pos += block_size) {
        size_t current_block = min(block_size, total_output - out_pos);
        
        for (size_t k = 0; k < current_block; k += vector_width) {
            #ifdef __x86_64__
            __m512d sum = _mm512_setzero_pd();
            
            for (size_t i = 0; i < total_kernel; i++) {
                __m512d kern = _mm512_set1_pd(kernel[i]);
                __m512d in = _mm512_load_pd(&input[out_pos + k + i]);
                sum = _mm512_fmadd_pd(kern, in, sum);
            }
            
            _mm512_store_pd(&output[out_pos + k], sum);
            #else
            float64x2_t sum = vdupq_n_f64(0.0);
            
            for (size_t i = 0; i < total_kernel; i++) {
                float64x2_t kern = vdupq_n_f64(kernel[i]);
                float64x2_t in = vld1q_f64(&input[out_pos + k + i]);
                sum = vfmaq_f64(sum, kern, in);
            }
            
            vst1q_f64(&output[out_pos + k], sum);
            #endif
        }
    }

    return QGT_SUCCESS;
}

CacheOptParams get_optimal_cache_params(size_t total_size, size_t vector_size) {
    CacheOptParams params = DEFAULT_PARAMS;
    
    // Adjust block size based on problem size
    if (total_size <= 32 * 1024) { // L1 cache
        params.block_size = BLOCK_SIZE_L1;
    } else if (total_size <= 256 * 1024) { // L2 cache
        params.block_size = BLOCK_SIZE_L2;
    } else { // L3 cache
        params.block_size = BLOCK_SIZE_L3;
    }
    
    params.vector_width = vector_size;
    return params;
}

CacheOptParams init_default_cache_params(void) {
    return DEFAULT_PARAMS;
}
