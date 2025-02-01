#ifndef QUANTUM_OPERATIONS_H
#define QUANTUM_OPERATIONS_H

#include "quantum_geometric/core/quantum_types.h"
#include "quantum_geometric/core/quantum_complex.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_state.h"
#include <stddef.h>
#include <stdbool.h>

// Operator creation/destruction
qgt_error_t quantum_operator_create(quantum_operator_t** operator,
                                  quantum_operator_type_t type,
                                  size_t dimension);
void quantum_operator_destroy(quantum_operator_t* operator);
qgt_error_t quantum_operator_clone(quantum_operator_t** dest,
                                 const quantum_operator_t* src);

// Operator initialization
qgt_error_t quantum_operator_initialize(quantum_operator_t* operator,
                                      const ComplexFloat* matrix);
qgt_error_t quantum_operator_initialize_identity(quantum_operator_t* operator);
qgt_error_t quantum_operator_initialize_zero(quantum_operator_t* operator);
qgt_error_t quantum_operator_initialize_random(quantum_operator_t* operator);

// Standard quantum gates
qgt_error_t quantum_operator_hadamard(quantum_operator_t* operator,
                                     size_t qubit);
qgt_error_t quantum_operator_pauli_x(quantum_operator_t* operator,
                                    size_t qubit);
qgt_error_t quantum_operator_pauli_y(quantum_operator_t* operator,
                                    size_t qubit);
qgt_error_t quantum_operator_pauli_z(quantum_operator_t* operator,
                                    size_t qubit);
qgt_error_t quantum_operator_phase(quantum_operator_t* operator,
                                  size_t qubit,
                                  float angle);
qgt_error_t quantum_operator_cnot(quantum_operator_t* operator,
                                 size_t control,
                                 size_t target);
qgt_error_t quantum_operator_swap(quantum_operator_t* operator,
                                 size_t qubit1,
                                 size_t qubit2);

// Operator operations
qgt_error_t quantum_operator_apply(quantum_operator_t* operator,
                                  quantum_state_t* state);
qgt_error_t quantum_operator_add(quantum_operator_t* result,
                                const quantum_operator_t* a,
                                const quantum_operator_t* b);
qgt_error_t quantum_operator_subtract(quantum_operator_t* result,
                                     const quantum_operator_t* a,
                                     const quantum_operator_t* b);
qgt_error_t quantum_operator_multiply(quantum_operator_t* result,
                                     const quantum_operator_t* a,
                                     const quantum_operator_t* b);
qgt_error_t quantum_operator_tensor_product(quantum_operator_t* result,
                                           const quantum_operator_t* a,
                                           const quantum_operator_t* b);

// Operator transformations
qgt_error_t quantum_operator_adjoint(quantum_operator_t* result,
                                    const quantum_operator_t* operator);
qgt_error_t quantum_operator_transpose(quantum_operator_t* result,
                                      const quantum_operator_t* operator);
qgt_error_t quantum_operator_conjugate(quantum_operator_t* result,
                                      const quantum_operator_t* operator);
qgt_error_t quantum_operator_exponential(quantum_operator_t* result,
                                        const quantum_operator_t* operator);

// Operator properties
qgt_error_t quantum_operator_trace(ComplexFloat* trace,
                                  const quantum_operator_t* operator);
qgt_error_t quantum_operator_determinant(ComplexFloat* determinant,
                                        const quantum_operator_t* operator);
qgt_error_t quantum_operator_eigenvalues(ComplexFloat* eigenvalues,
                                        const quantum_operator_t* operator);
qgt_error_t quantum_operator_eigenvectors(quantum_operator_t* eigenvectors,
                                         const quantum_operator_t* operator);

// Operator validation
qgt_error_t quantum_operator_validate(const quantum_operator_t* operator);
bool quantum_operator_is_hermitian(const quantum_operator_t* operator);
bool quantum_operator_is_unitary(const quantum_operator_t* operator);
bool quantum_operator_is_positive(const quantum_operator_t* operator);

// Hardware operations
qgt_error_t quantum_operator_to_device(quantum_operator_t* operator,
                                      quantum_hardware_t hardware);
qgt_error_t quantum_operator_from_device(quantum_operator_t* operator,
                                        quantum_hardware_t hardware);
bool quantum_operator_is_on_device(const quantum_operator_t* operator,
                                  quantum_hardware_t hardware);

// Resource management
qgt_error_t quantum_operator_estimate_resources(const quantum_operator_t* operator,
                                               size_t* memory,
                                               size_t* operations);
qgt_error_t quantum_operator_optimize_resources(quantum_operator_t* operator);
qgt_error_t quantum_operator_validate_resources(const quantum_operator_t* operator);

// Error correction
qgt_error_t quantum_operator_encode(quantum_operator_t* encoded,
                                   const quantum_operator_t* operator,
                                   quantum_error_code_t code);
qgt_error_t quantum_operator_decode(quantum_operator_t* decoded,
                                   const quantum_operator_t* operator,
                                   quantum_error_code_t code);
qgt_error_t quantum_operator_correct(quantum_operator_t* operator,
                                    quantum_error_code_t code);

// Utility functions
qgt_error_t quantum_operator_print(const quantum_operator_t* operator);
qgt_error_t quantum_operator_save(const quantum_operator_t* operator,
                                 const char* filename);
qgt_error_t quantum_operator_load(quantum_operator_t** operator,
                                 const char* filename);

#endif // QUANTUM_OPERATIONS_H
