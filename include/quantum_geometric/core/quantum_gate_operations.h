#ifndef QUANTUM_GATE_OPERATIONS_H
#define QUANTUM_GATE_OPERATIONS_H

#include "quantum_geometric/core/quantum_geometric_types.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Gate types
typedef enum {
    GATE_TYPE_I = 0,   // Identity
    GATE_TYPE_X = 1,   // Pauli-X
    GATE_TYPE_Y = 2,   // Pauli-Y
    GATE_TYPE_Z = 3,   // Pauli-Z
    GATE_TYPE_H = 4,   // Hadamard
    GATE_TYPE_S = 5,   // Phase
    GATE_TYPE_T = 6,   // T gate
    GATE_TYPE_RX = 7,  // Rotation around X
    GATE_TYPE_RY = 8,  // Rotation around Y
    GATE_TYPE_RZ = 9,  // Rotation around Z
    GATE_TYPE_CNOT = 10, // Controlled-NOT
    GATE_TYPE_CZ = 11,   // Controlled-Z
    GATE_TYPE_SWAP = 12, // SWAP
    GATE_TYPE_CUSTOM = 13 // Custom unitary
} gate_type_t;

/**
 * @brief Create a new quantum gate
 * 
 * @param type Gate type from gate_type_t enum
 * @param qubits Array of qubit indices the gate acts on
 * @param num_qubits Number of qubits
 * @param parameters Array of gate parameters (e.g. rotation angles)
 * @param num_parameters Number of parameters
 * @return quantum_gate_t* Pointer to created gate, NULL on failure
 */
quantum_gate_t* create_quantum_gate(
    gate_type_t type,
    const size_t* qubits,
    size_t num_qubits,
    const double* parameters,
    size_t num_parameters);

/**
 * @brief Update gate parameters
 * 
 * @param gate Gate to update
 * @param parameters New parameter values
 * @param num_parameters Number of parameters
 * @return true if successful, false otherwise
 */
bool update_gate_parameters(
    quantum_gate_t* gate,
    const double* parameters,
    size_t num_parameters);

/**
 * @brief Shift a gate parameter by given amount
 * 
 * @param gate Gate to modify
 * @param param_idx Index of parameter to shift
 * @param shift_amount Amount to shift parameter by
 * @return true if successful, false otherwise
 */
bool shift_gate_parameters(
    quantum_gate_t* gate,
    size_t param_idx,
    double shift_amount);

/**
 * @brief Destroy a quantum gate and free resources
 * 
 * @param gate Gate to destroy
 */
void destroy_quantum_gate(quantum_gate_t* gate);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_GATE_OPERATIONS_H
