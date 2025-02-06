#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "quantum_geometric/core/quantum_geometric_tensor_network.h"
#include "quantum_geometric/core/quantum_gate_operations.h"
#include "quantum_geometric/core/numerical_backend.h"

#define NUM_QUBITS 2
#define NUM_LAYERS 1

int main(void) {
    printf("Testing quantum geometric tensor...\n");

    // Initialize numerical backend
    numerical_config_t config = {
        .type = NUMERICAL_BACKEND_CPU,
        .max_threads = 1
    };
    bool success = initialize_numerical_backend(&config);
    assert(success && "Failed to initialize numerical backend");

    // Create quantum geometric tensor network
    quantum_geometric_tensor_network_t* qgtn = create_quantum_geometric_tensor_network(
        NUM_QUBITS, NUM_LAYERS, false, false
    );
    assert(qgtn != NULL && "Failed to create quantum geometric tensor network");

    // Add a parameterized gate (RX gate)
    size_t qubits[] = {0};  // Apply to first qubit
    double params[] = {0.5}; // Initial angle
    quantum_gate_t* gate = create_quantum_gate(GATE_TYPE_RX, qubits, 1, params, 1);
    assert(gate != NULL && "Failed to create quantum gate");

    // Apply the gate to the network
    success = apply_quantum_gate(qgtn, gate, qubits, 1);
    assert(success && "Failed to apply quantum gate");

    // Compute quantum geometric tensor
    ComplexFloat result;
    success = compute_quantum_geometric_tensor(qgtn, 0, 0, &result);
    assert(success && "Failed to compute quantum geometric tensor");

    printf("Quantum geometric tensor[0,0] = %f + %fi\n", result.real, result.imag);

    // Cleanup
    destroy_quantum_gate(gate);
    destroy_quantum_geometric_tensor_network(qgtn);
    printf("Test completed successfully!\n");
    return 0;
}
