#ifndef TENSOR_OPERATIONS_H
#define TENSOR_OPERATIONS_H

#include <stdbool.h>
#include <stddef.h>

#include "quantum_geometric/core/tensor_types.h"

/**
 * @brief Initialize a tensor with given dimensions
 * 
 * @param tensor Tensor to initialize
 * @param dimensions Array of dimension sizes
 * @param rank Number of dimensions
 * @return true on success, false on failure
 */
bool qg_tensor_init(tensor_t* tensor, const size_t* dimensions, size_t rank);

/**
 * @brief Free tensor resources
 * 
 * @param tensor Tensor to clean up
 */
void qg_tensor_cleanup(tensor_t* tensor);

/**
 * @brief Contract two tensors along specified dimensions
 * 
 * @param result Output tensor
 * @param a First input tensor
 * @param b Second input tensor
 * @param dimensions_a Dimensions of first tensor
 * @param dimensions_b Dimensions of second tensor
 * @param rank_a Number of dimensions in first tensor
 * @param rank_b Number of dimensions in second tensor
 * @param contract_a Contraction indices for first tensor
 * @param contract_b Contraction indices for second tensor
 * @param num_contract Number of dimensions to contract
 * @return true on success, false on failure
 */
bool qg_tensor_contract(ComplexFloat* result,
                       const ComplexFloat* a,
                       const ComplexFloat* b,
                       const size_t* dimensions_a,
                       const size_t* dimensions_b,
                       size_t rank_a,
                       size_t rank_b,
                       const size_t* contract_a,
                       const size_t* contract_b,
                       size_t num_contract);

/**
 * @brief Decompose tensor using SVD
 * 
 * @param u Output U matrix
 * @param s Output singular values
 * @param v Output V matrix
 * @param tensor Input tensor
 * @param dimensions Tensor dimensions
 * @param rank Number of dimensions
 * @return true on success, false on failure
 */
bool qg_tensor_decompose_svd(ComplexFloat* u,
                            ComplexFloat* s,
                            ComplexFloat* v,
                            const ComplexFloat* tensor,
                            const size_t* dimensions,
                            size_t rank);

/**
 * @brief Reshape tensor to new dimensions
 * 
 * @param result Output tensor
 * @param tensor Input tensor
 * @param new_dimensions New dimensions
 * @param new_rank Number of new dimensions
 * @return true on success, false on failure
 */
bool qg_tensor_reshape(ComplexFloat* result,
                      const ComplexFloat* tensor,
                      const size_t* new_dimensions,
                      size_t new_rank);

/**
 * @brief Transpose tensor by permuting dimensions
 * 
 * @param result Output tensor
 * @param tensor Input tensor
 * @param dimensions Tensor dimensions
 * @param rank Number of dimensions
 * @param perm Permutation array
 * @return true on success, false on failure
 */
bool qg_tensor_transpose(ComplexFloat* result,
                        const ComplexFloat* tensor,
                        const size_t* dimensions,
                        size_t rank,
                        const size_t* perm);

/**
 * @brief Scale tensor by scalar value
 * 
 * @param tensor Tensor to scale
 * @param scalar Scaling factor
 * @return true on success, false on failure
 */
bool qg_tensor_scale(tensor_t* tensor, float scalar);

/**
 * @brief Add two tensors elementwise
 * 
 * @param result Output tensor
 * @param a First input tensor
 * @param b Second input tensor
 * @return true on success, false on failure
 */
bool qg_tensor_add(tensor_t* result,
                  const tensor_t* a,
                  const tensor_t* b);

/**
 * @brief Get total size of tensor from dimensions
 * 
 * @param dimensions Dimension array
 * @param rank Number of dimensions
 * @return Total size in elements
 */
size_t qg_tensor_get_size(const size_t* dimensions, size_t rank);

#endif // TENSOR_OPERATIONS_H
