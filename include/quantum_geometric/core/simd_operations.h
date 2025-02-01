#ifndef SIMD_OPERATIONS_H
#define SIMD_OPERATIONS_H

#include "quantum_geometric/core/quantum_complex.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include <stddef.h>

// SIMD complex number operations
void simd_complex_copy(ComplexFloat* dest,
                      const ComplexFloat* src,
                      size_t count);

void simd_complex_multiply_accumulate(ComplexFloat* result,
                                    const ComplexFloat* a,
                                    const ComplexFloat* b,
                                    size_t count);

void simd_complex_scale(ComplexFloat* result,
                       const ComplexFloat* input,
                       ComplexFloat scalar,
                       size_t count);

double simd_complex_norm(const ComplexFloat* input,
                        size_t count);

void simd_complex_add(ComplexFloat* result,
                     const ComplexFloat* a,
                     const ComplexFloat* b,
                     size_t count);

void simd_complex_subtract(ComplexFloat* result,
                         const ComplexFloat* a,
                         const ComplexFloat* b,
                         size_t count);

void simd_complex_multiply(ComplexFloat* result,
                         const ComplexFloat* a,
                         const ComplexFloat* b,
                         size_t count);

void simd_complex_divide(ComplexFloat* result,
                        const ComplexFloat* a,
                        const ComplexFloat* b,
                        size_t count);

// SIMD tensor operations
void simd_tensor_add(ComplexFloat* result,
                    const ComplexFloat* a,
                    const ComplexFloat* b,
                    size_t total_elements);

void simd_tensor_subtract(ComplexFloat* result,
                         const ComplexFloat* a,
                         const ComplexFloat* b,
                         size_t total_elements);

void simd_tensor_multiply(ComplexFloat* result,
                         const ComplexFloat* a,
                         const ComplexFloat* b,
                         const size_t* dimensions,
                         size_t rank);

void simd_tensor_contract(ComplexFloat* result,
                         const ComplexFloat* a,
                         const ComplexFloat* b,
                         const size_t* dimensions_a,
                         const size_t* dimensions_b,
                         const size_t* contract_indices,
                         size_t num_indices,
                         size_t rank_a,
                         size_t rank_b);

void simd_tensor_contract_block(ComplexFloat* result,
                              const ComplexFloat* a,
                              const ComplexFloat* b,
                              size_t block_size,
                              size_t free_dims_a,
                              size_t free_dims_b);

void simd_tensor_scale(ComplexFloat* result,
                      const ComplexFloat* input,
                      ComplexFloat scalar,
                      size_t total_elements);

double simd_tensor_norm(const ComplexFloat* input,
                       size_t total_elements);

void simd_tensor_conjugate(ComplexFloat* result,
                          const ComplexFloat* input,
                          size_t total_elements);

void simd_tensor_transpose(ComplexFloat* result,
                          const ComplexFloat* input,
                          const size_t* dimensions,
                          const size_t* permutation,
                          size_t rank);

#endif // SIMD_OPERATIONS_H
