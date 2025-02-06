#ifndef QUANTUM_GATE_OPERATIONS_H
#define QUANTUM_GATE_OPERATIONS_H

#include "quantum_geometric/core/quantum_types.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

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
 * @brief Create a deep copy of a quantum gate
 * 
 * @param gate Gate to copy
 * @return quantum_gate_t* Pointer to copied gate, NULL on failure
 */
quantum_gate_t* copy_quantum_gate(const quantum_gate_t* gate);

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
