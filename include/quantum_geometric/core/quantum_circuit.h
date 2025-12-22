/**
 * @file quantum_circuit.h
 * @brief Quantum circuit representation and operations
 */

#ifndef QUANTUM_GEOMETRIC_CIRCUIT_H
#define QUANTUM_GEOMETRIC_CIRCUIT_H

#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_circuit_types.h"
#include "quantum_geometric/hardware/quantum_hardware_types.h"
#include <stdbool.h>

// Include the canonical quantum_result definition
#include "quantum_geometric/core/quantum_result.h"

// Gate types and aliases are defined in quantum_base_types.h (included via quantum_types.h)
// gate_type_t enum and GATE_* aliases are available from there

// Gate structure is defined in quantum_types.h

// quantum_result is defined in quantum_result.h - do not redefine here

// Circuit creation and management
quantum_circuit* create_quantum_circuit(size_t num_qubits);
void cleanup_quantum_circuit(quantum_circuit* circuit);

// Gate operations
qgt_error_t add_gate(quantum_circuit* circuit,
                     gate_type_t type,
                     size_t* qubits,
                     size_t num_qubits,
                     double* parameters,
                     size_t num_parameters);

qgt_error_t remove_gate(quantum_circuit* circuit, size_t index);

// Circuit optimization
qgt_error_t optimize_circuit_depth(quantum_circuit* circuit);
qgt_error_t optimize_gate_count(quantum_circuit* circuit);
qgt_error_t transpile_circuit(quantum_circuit* circuit,
                             const quantum_hardware_t* hardware);

// Circuit validation
qgt_error_t validate_circuit(const quantum_circuit* circuit);
bool is_valid_gate_sequence(const quantum_circuit* circuit);
bool check_qubit_connectivity(const quantum_circuit* circuit,
                            const quantum_hardware_t* hardware);

// Circuit analysis
size_t get_circuit_depth(const quantum_circuit* circuit);
size_t get_gate_count(const quantum_circuit* circuit);
double estimate_execution_time(const quantum_circuit* circuit,
                             const quantum_hardware_t* hardware);

// Result management
quantum_result* create_quantum_result(void);
void cleanup_quantum_result(quantum_result* result);
qgt_error_t process_measurement_results(quantum_result* result,
                                      const double* raw_measurements,
                                      size_t num_measurements);

#endif // QUANTUM_GEOMETRIC_CIRCUIT_H
