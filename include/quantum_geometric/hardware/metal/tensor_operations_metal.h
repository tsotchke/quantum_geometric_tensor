#ifndef TENSOR_OPERATIONS_METAL_H
#define TENSOR_OPERATIONS_METAL_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ===========================================================================
// Metal Tensor Error Types
// ===========================================================================

typedef int32_t metal_tensor_error_t;

#define METAL_TENSOR_SUCCESS              0
#define METAL_TENSOR_ERROR_INVALID_ARGS   -1
#define METAL_TENSOR_ERROR_NO_DEVICE      -2
#define METAL_TENSOR_ERROR_INIT_FAILED    -3
#define METAL_TENSOR_ERROR_OUT_OF_MEMORY  -4
#define METAL_TENSOR_ERROR_EXECUTION_FAILED -5
#define METAL_TENSOR_ERROR_NOT_SUPPORTED  -6
#define METAL_TENSOR_ERROR_BUFFER_FAILED  -7

// ===========================================================================
// Metal Tensor Device Functions
// ===========================================================================

/**
 * Check if Metal is available on this system
 * @return true if Metal is available, false otherwise
 */
bool metal_is_available(void);

/**
 * Initialize Metal tensor operations
 * @return METAL_TENSOR_SUCCESS on success, error code on failure
 */
metal_tensor_error_t metal_tensor_initialize(void);

/**
 * Cleanup Metal tensor operations
 */
void metal_tensor_cleanup(void);

// ===========================================================================
// Metal Matrix Operations
// ===========================================================================

/**
 * Perform complex matrix multiplication C = A * B on Metal GPU
 * Supports both real and complex matrices (pass NULL for imaginary if real-only)
 * @param C_real Output matrix real part (M x N)
 * @param C_imag Output matrix imaginary part (M x N), can be NULL
 * @param A_real Input matrix A real part (M x K)
 * @param A_imag Input matrix A imaginary part (M x K), can be NULL
 * @param B_real Input matrix B real part (K x N)
 * @param B_imag Input matrix B imaginary part (K x N), can be NULL
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A / rows in B
 * @return METAL_TENSOR_SUCCESS on success, error code on failure
 */
metal_tensor_error_t metal_matrix_multiply(float* C_real, float* C_imag,
                                           const float* A_real, const float* A_imag,
                                           const float* B_real, const float* B_imag,
                                           uint32_t M, uint32_t N, uint32_t K);

/**
 * Perform batched matrix multiplication on Metal GPU
 * @param C_real Output batch of matrices real parts
 * @param C_imag Output batch of matrices imaginary parts, can be NULL
 * @param A_real Input batch A real parts
 * @param A_imag Input batch A imaginary parts, can be NULL
 * @param B_real Input batch B real parts
 * @param B_imag Input batch B imaginary parts, can be NULL
 * @param M Number of rows in each A and C matrix
 * @param N Number of columns in each B and C matrix
 * @param K Number of columns in A / rows in B
 * @param batch_size Number of matrix pairs to multiply
 * @return METAL_TENSOR_SUCCESS on success, error code on failure
 */
metal_tensor_error_t metal_batched_matrix_multiply(float* C_real, float* C_imag,
                                                   const float* A_real, const float* A_imag,
                                                   const float* B_real, const float* B_imag,
                                                   uint32_t M, uint32_t N, uint32_t K,
                                                   uint32_t batch_size);

// ===========================================================================
// Metal Tensor Contraction Operations
// ===========================================================================

/**
 * Perform tensor contraction on Metal GPU
 * @param output Output tensor data
 * @param input_a First input tensor data
 * @param input_b Second input tensor data
 * @param dims_a Dimensions of first tensor
 * @param num_dims_a Number of dimensions in first tensor
 * @param dims_b Dimensions of second tensor
 * @param num_dims_b Number of dimensions in second tensor
 * @param contract_a Indices to contract in first tensor
 * @param contract_b Indices to contract in second tensor
 * @param num_contract Number of indices to contract
 * @return METAL_TENSOR_SUCCESS on success, error code on failure
 */
metal_tensor_error_t metal_tensor_contract(float* output,
                                           const float* input_a,
                                           const float* input_b,
                                           const uint32_t* dims_a,
                                           uint32_t num_dims_a,
                                           const uint32_t* dims_b,
                                           uint32_t num_dims_b,
                                           const uint32_t* contract_a,
                                           const uint32_t* contract_b,
                                           uint32_t num_contract);

// ===========================================================================
// Metal Element-wise Operations
// ===========================================================================

/**
 * Perform element-wise addition on Metal GPU
 * @param output Output array
 * @param input_a First input array
 * @param input_b Second input array
 * @param count Number of elements
 * @return METAL_TENSOR_SUCCESS on success, error code on failure
 */
metal_tensor_error_t metal_elementwise_add(float* output,
                                           const float* input_a,
                                           const float* input_b,
                                           size_t count);

/**
 * Perform element-wise multiplication on Metal GPU
 * @param output Output array
 * @param input_a First input array
 * @param input_b Second input array
 * @param count Number of elements
 * @return METAL_TENSOR_SUCCESS on success, error code on failure
 */
metal_tensor_error_t metal_elementwise_multiply(float* output,
                                                const float* input_a,
                                                const float* input_b,
                                                size_t count);

/**
 * Apply scalar multiplication on Metal GPU
 * @param output Output array
 * @param input Input array
 * @param scalar Scalar multiplier
 * @param count Number of elements
 * @return METAL_TENSOR_SUCCESS on success, error code on failure
 */
metal_tensor_error_t metal_scalar_multiply(float* output,
                                           const float* input,
                                           float scalar,
                                           size_t count);

// ===========================================================================
// Metal Reduction Operations
// ===========================================================================

/**
 * Compute sum of array elements on Metal GPU
 * @param result Output sum
 * @param input Input array
 * @param count Number of elements
 * @return METAL_TENSOR_SUCCESS on success, error code on failure
 */
metal_tensor_error_t metal_reduce_sum(float* result,
                                      const float* input,
                                      size_t count);

/**
 * Compute maximum of array elements on Metal GPU
 * @param result Output maximum
 * @param input Input array
 * @param count Number of elements
 * @return METAL_TENSOR_SUCCESS on success, error code on failure
 */
metal_tensor_error_t metal_reduce_max(float* result,
                                      const float* input,
                                      size_t count);

/**
 * Compute L2 norm on Metal GPU
 * @param result Output norm
 * @param input Input array
 * @param count Number of elements
 * @return METAL_TENSOR_SUCCESS on success, error code on failure
 */
metal_tensor_error_t metal_reduce_norm(float* result,
                                       const float* input,
                                       size_t count);

// ===========================================================================
// Metal Transpose Operations
// ===========================================================================

/**
 * Transpose a 2D matrix on Metal GPU
 * @param output Output transposed matrix (N x M)
 * @param input Input matrix (M x N)
 * @param M Number of rows in input
 * @param N Number of columns in input
 * @return METAL_TENSOR_SUCCESS on success, error code on failure
 */
metal_tensor_error_t metal_transpose_2d(float* output,
                                        const float* input,
                                        uint32_t M,
                                        uint32_t N);

#ifdef __cplusplus
}
#endif

#endif // TENSOR_OPERATIONS_METAL_H
