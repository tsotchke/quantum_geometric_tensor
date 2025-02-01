#ifndef QUANTUM_CIRCUIT_OPERATIONS_H
#define QUANTUM_CIRCUIT_OPERATIONS_H

#include <complex.h>
#include <stdbool.h>
#include "quantum_geometric/core/quantum_types.h"
#include "quantum_geometric/core/quantum_state.h"

// Pauli operator types
typedef enum {
    PAULI_X,
    PAULI_Y,
    PAULI_Z
} pauli_type;

// Circuit creation and management
quantum_circuit_t* quantum_circuit_create(size_t num_qubits);
void quantum_circuit_destroy(quantum_circuit_t* circuit);
void quantum_circuit_reset(quantum_circuit_t* circuit);

// Single-qubit gates
qgt_error_t quantum_circuit_hadamard(quantum_circuit_t* circuit, size_t qubit);
qgt_error_t quantum_circuit_pauli_x(quantum_circuit_t* circuit, size_t qubit);
qgt_error_t quantum_circuit_pauli_y(quantum_circuit_t* circuit, size_t qubit);
qgt_error_t quantum_circuit_pauli_z(quantum_circuit_t* circuit, size_t qubit);
qgt_error_t quantum_circuit_phase(quantum_circuit_t* circuit, size_t qubit, double angle);
qgt_error_t quantum_circuit_rotation(quantum_circuit_t* circuit, 
                                   size_t qubit, 
                                   double angle, 
                                   pauli_type axis);

// Two-qubit gates
qgt_error_t quantum_circuit_cnot(quantum_circuit_t* circuit, size_t control, size_t target);
qgt_error_t quantum_circuit_cz(quantum_circuit_t* circuit, size_t control, size_t target);
qgt_error_t quantum_circuit_swap(quantum_circuit_t* circuit, size_t qubit1, size_t qubit2);

// Circuit execution
qgt_error_t quantum_circuit_execute(quantum_circuit_t* circuit, quantum_state_t* state);
qgt_error_t quantum_circuit_measure(quantum_circuit_t* circuit, quantum_state_t* state, size_t* results);
qgt_error_t quantum_circuit_measure_all(quantum_circuit_t* circuit, quantum_state_t* state, size_t* results);

// Circuit optimization
qgt_error_t quantum_circuit_optimize(quantum_circuit_t* circuit, int optimization_level);
qgt_error_t quantum_circuit_decompose(quantum_circuit_t* circuit);
qgt_error_t quantum_circuit_validate(quantum_circuit_t* circuit);

// Circuit analysis
size_t quantum_circuit_depth(const quantum_circuit_t* circuit);
size_t quantum_circuit_gate_count(const quantum_circuit_t* circuit);
bool quantum_circuit_is_unitary(const quantum_circuit_t* circuit);

#endif // QUANTUM_CIRCUIT_OPERATIONS_H
