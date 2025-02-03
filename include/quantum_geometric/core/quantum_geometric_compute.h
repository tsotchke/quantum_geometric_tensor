#ifndef QUANTUM_GEOMETRIC_COMPUTE_H
#define QUANTUM_GEOMETRIC_COMPUTE_H

#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_complex.h"
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque types
typedef struct quantum_circuit quantum_circuit_t;

// Circuit creation and destruction
quantum_circuit_t* quantum_circuit_create(size_t num_qubits);
void quantum_circuit_destroy(quantum_circuit_t* circuit);

// Circuit execution
bool quantum_circuit_execute(quantum_circuit_t* circuit);

// State access
const quantum_geometric_state_t* quantum_circuit_get_state(const quantum_circuit_t* circuit);

// Common quantum gates
bool quantum_circuit_add_hadamard(quantum_circuit_t* circuit, size_t qubit);
bool quantum_circuit_add_cnot(quantum_circuit_t* circuit, size_t control, size_t target);
bool quantum_circuit_add_phase(quantum_circuit_t* circuit, size_t qubit, float phase);
bool quantum_circuit_add_measurement(quantum_circuit_t* circuit, size_t qubit);

// Advanced operations
bool quantum_circuit_add_tensor_product(quantum_circuit_t* circuit,
                                      const size_t* qubits,
                                      size_t num_qubits);

bool quantum_circuit_add_partial_trace(quantum_circuit_t* circuit,
                                     const size_t* qubits,
                                     size_t num_qubits);

bool quantum_circuit_add_quantum_fourier(quantum_circuit_t* circuit,
                                       const size_t* qubits,
                                       size_t num_qubits);

// Custom operations
bool quantum_circuit_add_custom_unitary(quantum_circuit_t* circuit,
                                      const ComplexFloat* matrix,
                                      const size_t* qubits,
                                      size_t num_qubits);

// Error handling
typedef enum {
    QUANTUM_CIRCUIT_SUCCESS = 0,
    QUANTUM_CIRCUIT_ERROR_INVALID_ARGUMENT = -1,
    QUANTUM_CIRCUIT_ERROR_MEMORY = -2,
    QUANTUM_CIRCUIT_ERROR_EXECUTION = -3,
    QUANTUM_CIRCUIT_ERROR_INVALID_STATE = -4
} quantum_circuit_error_t;

const char* quantum_circuit_get_error_string(quantum_circuit_error_t error);

// Performance monitoring
typedef struct {
    double execution_time;    // Total execution time in seconds
    size_t gate_count;       // Number of gates executed
    size_t memory_used;      // Peak memory usage in bytes
    double fidelity;         // State fidelity (if available)
} quantum_circuit_metrics_t;

bool quantum_circuit_get_metrics(const quantum_circuit_t* circuit,
                               quantum_circuit_metrics_t* metrics);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_GEOMETRIC_COMPUTE_H
