#ifndef QUANTUM_CIRCUIT_OPERATIONS_H
#define QUANTUM_CIRCUIT_OPERATIONS_H

#include <complex.h>
#include <stdbool.h>
#include "quantum_geometric/core/quantum_types.h"
#include "quantum_geometric/core/quantum_state.h"
#include "quantum_geometric/core/quantum_state_types.h"

// Pauli types (pauli_type, rotation_axis_t, PAULI_X/Y/Z) are defined
// in quantum_types.h which is included above

// Convenience type alias for tests - QuantumState has 'amplitudes' field
typedef QuantumState quantum_state;

// Convenience state creation/reset functions for tests
quantum_state* init_quantum_state(size_t num_qubits);
void quantum_state_reset(quantum_state* state);
void quantum_state_cleanup(quantum_state* state);

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
// Note: Uses quantum_state (QuantumState) which has amplitudes field
// For quantum_state_t integration, use quantum_operator_apply() from quantum_operations.h
qgt_error_t quantum_circuit_execute(quantum_circuit_t* circuit, quantum_state* state);
qgt_error_t quantum_circuit_measure(quantum_circuit_t* circuit, quantum_state* state, size_t* results);
qgt_error_t quantum_circuit_measure_all(quantum_circuit_t* circuit, quantum_state* state, size_t* results);

// Circuit optimization
qgt_error_t quantum_circuit_optimize(quantum_circuit_t* circuit, int optimization_level);
qgt_error_t quantum_circuit_decompose(quantum_circuit_t* circuit);
qgt_error_t quantum_circuit_validate(quantum_circuit_t* circuit);

// Circuit analysis
size_t quantum_circuit_depth(const quantum_circuit_t* circuit);
size_t quantum_circuit_gate_count(const quantum_circuit_t* circuit);
bool quantum_circuit_is_unitary(const quantum_circuit_t* circuit);

// Forward declaration for HierarchicalMatrix
struct HierarchicalMatrix;
typedef struct HierarchicalMatrix HierarchicalMatrix;

// Quantum-accelerated matrix operations
void quantum_encode_matrix(QuantumState* state, const HierarchicalMatrix* mat);
void quantum_decode_matrix(HierarchicalMatrix* mat, const QuantumState* state);
void quantum_circuit_multiply(QuantumState* a, QuantumState* b);
void quantum_compress_circuit(QuantumState* state, size_t target_qubits);

#endif // QUANTUM_CIRCUIT_OPERATIONS_H
