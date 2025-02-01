#ifndef QUANTUM_REGISTER_H
#define QUANTUM_REGISTER_H

#include <stddef.h>
#include <complex.h>
#include "quantum_geometric/core/quantum_types.h"
#include "quantum_geometric/core/quantum_geometric_types.h"

// Creation and destruction
quantum_register_t* quantum_register_create(size_t num_qubits, int flags);
quantum_register_t* quantum_register_create_state(complex double* amplitudes, size_t size, quantum_system_t* system);
void quantum_register_destroy(quantum_register_t* reg);

// State initialization
qgt_error_t quantum_register_initialize(quantum_register_t* reg, const complex double* initial_state);
qgt_error_t quantum_register_reset(quantum_register_t* reg);

// State manipulation
qgt_error_t quantum_register_apply_gate(quantum_register_t* reg, const quantum_operator_t* gate, size_t target);
qgt_error_t quantum_register_apply_controlled_gate(quantum_register_t* reg, const quantum_operator_t* gate, 
                                                size_t control, size_t target);

// Measurement operations
qgt_error_t quantum_register_measure_qubit(quantum_register_t* reg, size_t qubit, int* result);
qgt_error_t quantum_register_measure_all(quantum_register_t* reg, size_t* results);
qgt_error_t quantum_register_get_probabilities(quantum_register_t* reg, double* probabilities);

// State analysis
double quantum_register_fidelity(const quantum_register_t* reg1, const quantum_register_t* reg2);
double quantum_register_trace_distance(const quantum_register_t* reg1, const quantum_register_t* reg2);
complex double quantum_register_expectation_value(const quantum_register_t* reg, const quantum_operator_t* operator);

// Error correction
qgt_error_t quantum_register_apply_error_correction(quantum_register_t* reg);
qgt_error_t quantum_register_syndrome_measurement(quantum_register_t* reg, double* syndrome);

// Device management
qgt_error_t quantum_register_to_device(quantum_register_t* reg, int device_type);
qgt_error_t quantum_register_from_device(quantum_register_t* reg);

// Utility functions
void quantum_register_print_state(const quantum_register_t* reg);
int quantum_register_verify_state(const quantum_register_t* reg);
size_t quantum_register_memory_size(const quantum_register_t* reg);

#endif // QUANTUM_REGISTER_H
