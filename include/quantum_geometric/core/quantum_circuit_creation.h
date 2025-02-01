#ifndef QUANTUM_CIRCUIT_CREATION_H
#define QUANTUM_CIRCUIT_CREATION_H

#include "quantum_geometric/core/quantum_types.h"
#include "quantum_geometric/core/quantum_circuit.h"

// Circuit creation functions
quantum_circuit_t* quantum_create_inversion_circuit(size_t num_qubits, int flags);
quantum_circuit_t* quantum_create_gradient_circuit(size_t num_qubits, int flags);
quantum_circuit_t* quantum_create_hessian_circuit(size_t num_qubits, int flags);

// Quantum operations
void quantum_compute_gradient(quantum_register_t* reg_state,
                            quantum_register_t* reg_observable,
                            quantum_register_t* reg_gradient,
                            quantum_system_t* system,
                            quantum_circuit_t* circuit,
                            const quantum_phase_config_t* config);

void quantum_compute_hessian_hierarchical(quantum_register_t* reg_state,
                                        quantum_register_t* reg_observable,
                                        quantum_register_t* reg_gradient,
                                        quantum_register_t* reg_hessian,
                                        quantum_system_t* system,
                                        quantum_circuit_t* circuit,
                                        const quantum_phase_config_t* config);

void quantum_apply_threshold(double complex* data,
                           size_t size,
                           double threshold,
                           quantum_system_t* system,
                           quantum_circuit_t* circuit,
                           const quantum_phase_config_t* config);

void quantum_apply_matrix_threshold(double complex* matrix,
                                  size_t dim,
                                  double threshold,
                                  quantum_system_t* system,
                                  quantum_circuit_t* circuit,
                                  const quantum_phase_config_t* config);

void qgt_normalize_state(double complex* state, size_t dim);
void qgt_complex_matrix_multiply(const double complex* a,
                               const double complex* b,
                               double complex* c,
                               size_t m, size_t n, size_t p);

#endif // QUANTUM_CIRCUIT_CREATION_H
