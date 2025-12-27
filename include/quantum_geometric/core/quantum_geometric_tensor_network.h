#ifndef QUANTUM_GEOMETRIC_TENSOR_NETWORK_H
#define QUANTUM_GEOMETRIC_TENSOR_NETWORK_H

#include "quantum_geometric/core/tensor_network_operations.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/numerical_backend.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Quantum geometric tensor network structure
// Quantum backend types
typedef enum {
    QGTN_BACKEND_SIMULATOR,    // Local quantum simulator
    QGTN_BACKEND_IBM,         // IBM Quantum hardware
    QGTN_BACKEND_RIGETTI,     // Rigetti quantum hardware
    QGTN_BACKEND_DWAVE,       // D-Wave quantum annealer
    QGTN_BACKEND_IONQ,        // IonQ trapped-ion hardware
    QGTN_BACKEND_XANADU,      // Xanadu photonic hardware
    QGTN_BACKEND_CUSTOM       // Custom quantum backend
} qgtn_backend_type_t;

// Hardware configuration
typedef struct {
    qgtn_backend_type_t type;
    void* backend_specific;    // Backend-specific configuration
    bool supports_gradients;   // Whether backend supports gradient computation
    bool supports_hybrid;      // Whether backend supports hybrid computation
} qgtn_hardware_config_t;

typedef struct quantum_geometric_tensor_network {
    tensor_network_t* network;
    quantum_circuit_t* circuit;
    size_t num_qubits;
    size_t num_layers;
    bool is_distributed;
    bool use_hardware_acceleration;
    qgtn_hardware_config_t hardware_config;
    void* backend_state;       // Backend-specific state
} quantum_geometric_tensor_network_t;

// Creation and destruction
quantum_geometric_tensor_network_t* create_quantum_geometric_tensor_network(
    size_t num_qubits,
    size_t num_layers,
    bool is_distributed,
    bool use_hardware_acceleration
);

void destroy_quantum_geometric_tensor_network(
    quantum_geometric_tensor_network_t* qgtn
);

/**
 * @brief Create a deep copy of a quantum geometric tensor network
 * 
 * Creates a new quantum geometric tensor network that is an exact copy
 * of the input network, including all internal state.
 * 
 * @param qgtn The quantum geometric tensor network to copy
 * @return A pointer to the new copy, or NULL if allocation failed
 */
quantum_geometric_tensor_network_t* copy_quantum_geometric_tensor_network(
    const quantum_geometric_tensor_network_t* qgtn
);

// Quantum operations
/**
 * @brief Create a deep copy of a quantum gate
 * 
 * Creates a new quantum gate that is an exact copy of the input gate,
 * including all internal state.
 * 
 * @param gate The quantum gate to copy
 * @return A pointer to the new copy, or NULL if allocation failed
 */
quantum_gate_t* copy_quantum_gate(const quantum_gate_t* gate);

bool apply_quantum_gate(
    quantum_geometric_tensor_network_t* qgtn,
    const quantum_gate_t* gate,
    const size_t* qubits,
    size_t num_qubits
);

bool apply_quantum_circuit(
    quantum_geometric_tensor_network_t* qgtn,
    const quantum_circuit_t* circuit
);

bool qgtn_measure_qubit(
    const quantum_geometric_tensor_network_t* qgtn,
    size_t qubit,
    double* probability_zero,
    double* probability_one
);

bool get_quantum_state(
    const quantum_geometric_tensor_network_t* qgtn,
    ComplexFloat** state_vector,
    size_t* dimension
);

// Geometric operations
bool compute_quantum_geometric_tensor(
    const quantum_geometric_tensor_network_t* qgtn,
    size_t param_i,
    size_t param_j,
    ComplexFloat* result
);

bool compute_quantum_metric(
    const quantum_geometric_tensor_network_t* qgtn,
    size_t param_i,
    size_t param_j,
    double* result
);

bool compute_berry_curvature(
    const quantum_geometric_tensor_network_t* qgtn,
    size_t param_i,
    size_t param_j,
    double* result
);

// Distributed operations
bool distribute_computation(
    quantum_geometric_tensor_network_t* qgtn,
    const size_t* device_ids,
    size_t num_devices
);

bool synchronize_distributed_state(
    quantum_geometric_tensor_network_t* qgtn
);

// Hardware acceleration
bool enable_hardware_acceleration(
    quantum_geometric_tensor_network_t* qgtn,
    HardwareType type
);

bool disable_hardware_acceleration(
    quantum_geometric_tensor_network_t* qgtn
);

// Error handling
const char* get_quantum_geometric_tensor_network_error(void);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_GEOMETRIC_TENSOR_NETWORK_H
