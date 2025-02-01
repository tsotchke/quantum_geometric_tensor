#ifndef SIMD_TENSOR_OPERATIONS_H
#define SIMD_TENSOR_OPERATIONS_H

#include <stddef.h>
#include "quantum_geometric/core/quantum_geometric_types.h"

// Block size tuning parameters
#define BLOCK_SIZE_L1 32  // L1 cache block size
#define BLOCK_SIZE_L2 64  // L2 cache block size
#define BLOCK_SIZE_L3 128 // L3 cache block size

// Operation types for elementwise operations
typedef enum {
    OP_ADD,
    OP_SUBTRACT,
    OP_MULTIPLY,
    OP_DIVIDE,
    OP_MAXIMUM,
    OP_MINIMUM
} ElementwiseOp;

// Operation types for reduction operations
typedef enum {
    REDUCE_SUM,
    REDUCE_PRODUCT,
    REDUCE_MAXIMUM,
    REDUCE_MINIMUM
} ReductionOp;

// Cache optimization parameters
typedef struct {
    size_t block_size;      // Block size for tiling
    size_t prefetch_dist;   // Prefetch distance in elements
    size_t vector_width;    // SIMD vector width
    size_t align_size;      // Memory alignment size
} CacheOptParams;

/**
 * @brief Matrix multiplication using AVX-512 with block processing and prefetching
 * 
 * @param A Input matrix A
 * @param B Input matrix B
 * @param C Output matrix C = A * B
 * @param M Number of rows in A
 * @param N Number of columns in B
 * @param K Number of columns in A / rows in B
 * @param params Cache optimization parameters
 * @return int Error code (0 on success)
 */
int matrix_multiply_avx512(const double* A,
                         const double* B,
                         double* C,
                         size_t M,
                         size_t N,
                         size_t K,
                         const CacheOptParams* params);

/**
 * @brief Tensor contraction using hybrid quantum-classical execution with block processing
 * 
 * @param A Input tensor A
 * @param B Input tensor B
 * @param C Output tensor C
 * @param dims Array of dimension sizes
 * @param rank Tensor rank
 * @param params Cache optimization parameters
 * @return int Error code (0 on success)
 */
int tensor_contract_avx512(const double* A,
                         const double* B,
                         double* C,
                         const size_t* dims,
                         size_t rank,
                         const CacheOptParams* params);

/**
 * @brief Elementwise operations using AVX-512 with block processing
 * 
 * @param A Input tensor A
 * @param B Input tensor B
 * @param C Output tensor C
 * @param size Number of elements
 * @param op Operation type
 * @param params Cache optimization parameters
 * @return int Error code (0 on success)
 */
int tensor_elementwise_avx512(const double* A,
                            const double* B,
                            double* C,
                            size_t size,
                            ElementwiseOp op,
                            const CacheOptParams* params);

/**
 * @brief Hierarchical reduction using GPU and quantum-inspired techniques with block processing
 * 
 * @param A Input tensor
 * @param size Number of elements
 * @param op Reduction operation
 * @param result Pointer to store result
 * @param params Cache optimization parameters
 * @return int Error code (0 on success)
 */
int tensor_reduce_avx512(const double* A,
                        size_t size,
                        ReductionOp op,
                        double* result,
                        const CacheOptParams* params);

/**
 * @brief Convolution using hybrid quantum-classical execution with block processing
 * 
 * @param input Input tensor
 * @param kernel Convolution kernel
 * @param output Output tensor
 * @param input_dims Input dimensions
 * @param kernel_dims Kernel dimensions
 * @param rank Tensor rank
 * @param params Cache optimization parameters
 * @return int Error code (0 on success)
 */
int tensor_conv_avx512(const double* input,
                      const double* kernel,
                      double* output,
                      const size_t* input_dims,
                      const size_t* kernel_dims,
                      size_t rank,
                      const CacheOptParams* params);

/**
 * @brief Get optimal cache parameters based on problem size and hardware
 * 
 * @param total_size Total size of data in elements
 * @param vector_size Size of SIMD vector in elements
 * @return CacheOptParams Optimized parameters for cache usage
 */
CacheOptParams get_optimal_cache_params(size_t total_size, size_t vector_size);

/**
 * @brief Initialize cache optimization parameters with default values
 * 
 * @return CacheOptParams Default cache optimization parameters
 */
CacheOptParams init_default_cache_params(void);

#endif // SIMD_TENSOR_OPERATIONS_H
