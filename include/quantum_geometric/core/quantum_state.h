#ifndef QUANTUM_STATE_H
#define QUANTUM_STATE_H

#include <stddef.h>
#include <stdbool.h>
#include "quantum_geometric/core/quantum_complex.h"
#include "quantum_geometric/core/quantum_state_types.h"
#include "quantum_geometric/core/error_codes.h"
#include "quantum_geometric/hardware/quantum_hardware_types.h"
#include "quantum_geometric/hardware/quantum_error_constants.h"

// Quantum state structure is defined in quantum_types.h

// State creation/destruction
qgt_error_t quantum_state_create(quantum_state_t** state,
                               quantum_state_type_t type,
                               size_t dimension);
void quantum_state_destroy(quantum_state_t* state);
qgt_error_t quantum_state_clone(quantum_state_t** dest,
                              const quantum_state_t* src);

// State initialization
qgt_error_t quantum_state_initialize(quantum_state_t* state,
                                   const ComplexFloat* amplitudes);
qgt_error_t quantum_state_initialize_zero(quantum_state_t* state);
qgt_error_t quantum_state_initialize_basis(quantum_state_t* state,
                                         size_t basis_index);
qgt_error_t quantum_state_initialize_random(quantum_state_t* state);

// State operations
qgt_error_t quantum_state_add(quantum_state_t* result,
                             const quantum_state_t* a,
                             const quantum_state_t* b);
qgt_error_t quantum_state_subtract(quantum_state_t* result,
                                  const quantum_state_t* a,
                                  const quantum_state_t* b);
qgt_error_t quantum_state_multiply(quantum_state_t* result,
                                  const quantum_state_t* state,
                                  ComplexFloat scalar);
qgt_error_t quantum_state_tensor_product(quantum_state_t* result,
                                        const quantum_state_t* a,
                                        const quantum_state_t* b);

// State transformations
qgt_error_t quantum_state_apply_operator(quantum_state_t* result,
                                        const quantum_state_t* state,
                                        const quantum_operator_t* operator);
qgt_error_t quantum_state_evolve(quantum_state_t* result,
                                const quantum_state_t* state,
                                const quantum_operator_t* hamiltonian,
                                float time);
qgt_error_t quantum_state_measure(size_t* outcome,
                                 quantum_state_t* post_state,
                                 const quantum_state_t* state,
                                 const quantum_measurement_t* measurement);

// State properties
qgt_error_t quantum_state_inner_product(ComplexFloat* result,
                                       const quantum_state_t* a,
                                       const quantum_state_t* b);
qgt_error_t quantum_state_norm(float* norm,
                              const quantum_state_t* state);
qgt_error_t quantum_state_normalize(quantum_state_t* state);
qgt_error_t quantum_state_trace(ComplexFloat* trace,
                               const quantum_state_t* state);
qgt_error_t quantum_state_purity(float* purity,
                                const quantum_state_t* state);
qgt_error_t quantum_state_fidelity(float* fidelity,
                                  const quantum_state_t* a,
                                  const quantum_state_t* b);

// State analysis
qgt_error_t quantum_state_expectation_value(ComplexFloat* result,
                                           const quantum_state_t* state,
                                           const quantum_operator_t* operator);
qgt_error_t quantum_state_reduced_density_matrix(quantum_state_t* result,
                                                const quantum_state_t* state,
                                                const size_t* subsystem_qubits,
                                                size_t num_qubits);
qgt_error_t quantum_state_entropy(float* entropy,
                                 const quantum_state_t* state);
qgt_error_t quantum_state_concurrence(float* concurrence,
                                     const quantum_state_t* state);

// State validation
qgt_error_t quantum_state_validate(const quantum_state_t* state);
bool quantum_state_is_pure(const quantum_state_t* state);
bool quantum_state_is_mixed(const quantum_state_t* state);
bool quantum_state_is_normalized(const quantum_state_t* state);
bool quantum_state_is_entangled(const quantum_state_t* state);

// Hardware operations
qgt_error_t quantum_state_to_device(quantum_state_t* state,
                                   quantum_hardware_t hardware);
qgt_error_t quantum_state_from_device(quantum_state_t* state,
                                     quantum_hardware_t hardware);
bool quantum_state_is_on_device(const quantum_state_t* state,
                               quantum_hardware_t hardware);

// Resource management
qgt_error_t quantum_state_estimate_resources(const quantum_state_t* state,
                                            size_t* memory,
                                            size_t* operations);
qgt_error_t quantum_state_optimize_resources(quantum_state_t* state);
qgt_error_t quantum_state_validate_resources(const quantum_state_t* state);

// Error correction
qgt_error_t quantum_state_encode(quantum_state_t* encoded,
                                const quantum_state_t* state,
                                quantum_error_code_t code);
qgt_error_t quantum_state_decode(quantum_state_t* decoded,
                                const quantum_state_t* state,
                                quantum_error_code_t code);
qgt_error_t quantum_state_correct(quantum_state_t* state,
                                 quantum_error_code_t code);

// Utility functions
qgt_error_t quantum_state_print(const quantum_state_t* state);
qgt_error_t quantum_state_save(const quantum_state_t* state,
                              const char* filename);
qgt_error_t quantum_state_load(quantum_state_t** state,
                              const char* filename);

#endif // QUANTUM_STATE_H
