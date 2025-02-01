#ifndef QUANTUM_GEOMETRIC_TENSOR_H
#define QUANTUM_GEOMETRIC_TENSOR_H

#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_complex.h"
#include "quantum_geometric/core/quantum_state_types.h"
#include "quantum_geometric/core/quantum_state.h"
#include "quantum_geometric/core/numeric_utils.h"
#include <stddef.h>
#include <stdbool.h>

// Tensor creation/destruction
qgt_error_t geometric_tensor_create(quantum_geometric_tensor_t** tensor,
                                  geometric_tensor_type_t type,
                                  const size_t* dimensions,
                                  size_t rank);
void geometric_tensor_destroy(quantum_geometric_tensor_t* tensor);
qgt_error_t geometric_tensor_clone(quantum_geometric_tensor_t** dest,
                                 const quantum_geometric_tensor_t* src);

// Tensor initialization
qgt_error_t geometric_tensor_initialize(quantum_geometric_tensor_t* tensor,
                                      const ComplexFloat* data);
qgt_error_t geometric_tensor_initialize_zero(quantum_geometric_tensor_t* tensor);
qgt_error_t geometric_tensor_initialize_random(quantum_geometric_tensor_t* tensor);
qgt_error_t geometric_tensor_initialize_identity(quantum_geometric_tensor_t* tensor);

// Tensor operations
qgt_error_t geometric_tensor_add(quantum_geometric_tensor_t* result,
                                const quantum_geometric_tensor_t* a,
                                const quantum_geometric_tensor_t* b);
qgt_error_t geometric_tensor_subtract(quantum_geometric_tensor_t* result,
                                     const quantum_geometric_tensor_t* a,
                                     const quantum_geometric_tensor_t* b);
qgt_error_t geometric_tensor_multiply(quantum_geometric_tensor_t* result,
                                     const quantum_geometric_tensor_t* a,
                                     const quantum_geometric_tensor_t* b);
qgt_error_t geometric_tensor_divide(quantum_geometric_tensor_t* result,
                                   const quantum_geometric_tensor_t* a,
                                   const quantum_geometric_tensor_t* b);

// Tensor contractions
qgt_error_t geometric_tensor_contract(quantum_geometric_tensor_t* result,
                                     const quantum_geometric_tensor_t* a,
                                     const quantum_geometric_tensor_t* b,
                                     const size_t* indices_a,
                                     const size_t* indices_b,
                                     size_t num_indices);
qgt_error_t geometric_tensor_outer_product(quantum_geometric_tensor_t* result,
                                          const quantum_geometric_tensor_t* a,
                                          const quantum_geometric_tensor_t* b);
qgt_error_t geometric_tensor_inner_product(ComplexFloat* result,
                                          const quantum_geometric_tensor_t* a,
                                          const quantum_geometric_tensor_t* b);

// Tensor transformations
qgt_error_t geometric_tensor_transpose(quantum_geometric_tensor_t* result,
                                      const quantum_geometric_tensor_t* tensor,
                                      const size_t* permutation);
qgt_error_t geometric_tensor_conjugate(quantum_geometric_tensor_t* result,
                                      const quantum_geometric_tensor_t* tensor);
qgt_error_t geometric_tensor_scale(quantum_geometric_tensor_t* result,
                                  const quantum_geometric_tensor_t* tensor,
                                  ComplexFloat scalar);
qgt_error_t geometric_tensor_adjoint(quantum_geometric_tensor_t* result,
                                    const quantum_geometric_tensor_t* tensor);

// Tensor decompositions
qgt_error_t geometric_tensor_svd(quantum_geometric_tensor_t* u,
                                quantum_geometric_tensor_t* s,
                                quantum_geometric_tensor_t* v,
                                const quantum_geometric_tensor_t* tensor);
qgt_error_t geometric_tensor_eigendecomposition(quantum_geometric_tensor_t* eigenvectors,
                                               ComplexFloat* eigenvalues,
                                               const quantum_geometric_tensor_t* tensor);
qgt_error_t geometric_tensor_qr(quantum_geometric_tensor_t* q,
                               quantum_geometric_tensor_t* r,
                               const quantum_geometric_tensor_t* tensor);

// Tensor properties
qgt_error_t geometric_tensor_norm(float* norm,
                                 const quantum_geometric_tensor_t* tensor);
qgt_error_t geometric_tensor_trace(ComplexFloat* trace,
                                  const quantum_geometric_tensor_t* tensor);
qgt_error_t geometric_tensor_determinant(ComplexFloat* determinant,
                                        const quantum_geometric_tensor_t* tensor);
bool geometric_tensor_is_hermitian(const quantum_geometric_tensor_t* tensor);
bool geometric_tensor_is_unitary(const quantum_geometric_tensor_t* tensor);
bool geometric_tensor_is_positive_definite(const quantum_geometric_tensor_t* tensor);

// Tensor validation
qgt_error_t geometric_tensor_validate(const quantum_geometric_tensor_t* tensor);
qgt_error_t geometric_tensor_validate_dimensions(const quantum_geometric_tensor_t* tensor);
qgt_error_t geometric_tensor_validate_compatibility(const quantum_geometric_tensor_t* a,
                                                   const quantum_geometric_tensor_t* b);

// Hardware operations
qgt_error_t geometric_tensor_to_device(quantum_geometric_tensor_t* tensor,
                                      quantum_hardware_t hardware);
qgt_error_t geometric_tensor_from_device(quantum_geometric_tensor_t* tensor,
                                        quantum_hardware_t hardware);
bool geometric_tensor_is_on_device(const quantum_geometric_tensor_t* tensor,
                                  quantum_hardware_t hardware);

// Geometric metric operations
qgt_error_t geometric_create_metric(quantum_geometric_metric_t** metric,
                                  geometric_metric_type_t type,
                                  size_t dimension);
qgt_error_t geometric_compute_metric(quantum_geometric_metric_t* metric,
                                   const quantum_state_t* state);
void geometric_destroy_metric(quantum_geometric_metric_t* metric);

// Geometric connection operations
qgt_error_t geometric_create_connection(quantum_geometric_connection_t** connection,
                                      geometric_connection_type_t type,
                                      size_t dimension);
qgt_error_t geometric_compute_connection(quantum_geometric_connection_t* connection,
                                       const quantum_geometric_metric_t* metric);
void geometric_destroy_connection(quantum_geometric_connection_t* connection);

// Geometric curvature operations
qgt_error_t geometric_create_curvature(quantum_geometric_curvature_t** curvature,
                                     geometric_curvature_type_t type,
                                     size_t dimension);
qgt_error_t geometric_compute_curvature(quantum_geometric_curvature_t* curvature,
                                      const quantum_geometric_connection_t* connection);
void geometric_destroy_curvature(quantum_geometric_curvature_t* curvature);

// Geometric phase operations
qgt_error_t geometric_compute_phase(ComplexFloat* phase,
                                  const quantum_state_t* state,
                                  const quantum_geometric_connection_t* connection);

// Resource management
qgt_error_t geometric_tensor_estimate_resources(const quantum_geometric_tensor_t* tensor,
                                               size_t* memory,
                                               size_t* operations);
qgt_error_t geometric_tensor_optimize_resources(quantum_geometric_tensor_t* tensor);
qgt_error_t geometric_tensor_validate_resources(const quantum_geometric_tensor_t* tensor);

// Utility functions
qgt_error_t geometric_tensor_print(const quantum_geometric_tensor_t* tensor);
qgt_error_t geometric_tensor_save(const quantum_geometric_tensor_t* tensor,
                                 const char* filename);
qgt_error_t geometric_tensor_load(quantum_geometric_tensor_t** tensor,
                                 const char* filename);

#endif // QUANTUM_GEOMETRIC_TENSOR_H
