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
        .max_threads = 1,
        .use_fma = false,
        .use_avx = false,
        .use_neon = false,
        .cache_size = 0,
        .backend_specific = NULL
    };
    
    numerical_error_t error = initialize_numerical_backend(&config);
    if (error != NUMERICAL_SUCCESS) {
        printf("Failed to initialize numerical backend: %s\n", 
               get_numerical_error_string(error));
        return 1;
    }

    // Create quantum geometric tensor network
    quantum_geometric_tensor_network_t* qgtn = create_quantum_geometric_tensor_network(
        NUM_QUBITS, NUM_LAYERS, false, false
    );
    if (!qgtn) {
        printf("Failed to create quantum geometric tensor network\n");
        shutdown_numerical_backend();
        return 1;
    }

    // Add a parameterized gate (RX gate)
    size_t qubits[] = {0};  // Apply to first qubit
    double params[] = {0.5}; // Initial angle
    quantum_gate_t* gate = create_quantum_gate(GATE_TYPE_RX, qubits, 1, params, 1);
    if (!gate) {
        printf("Failed to create quantum gate\n");
        destroy_quantum_geometric_tensor_network(qgtn);
        shutdown_numerical_backend();
        return 1;
    }

    // Apply the gate to the network
    error = apply_quantum_gate(qgtn, gate, qubits, 1);
    if (error != NUMERICAL_SUCCESS) {
        printf("Failed to apply quantum gate: %s\n",
               get_numerical_error_string(error));
        destroy_quantum_gate(gate);
        destroy_quantum_geometric_tensor_network(qgtn);
        shutdown_numerical_backend();
        return 1;
    }

    // Compute quantum geometric tensor
    ComplexFloat result;
    error = compute_quantum_geometric_tensor(qgtn, 0, 0, &result);
    if (error != NUMERICAL_SUCCESS) {
        printf("Failed to compute quantum geometric tensor: %s\n",
               get_numerical_error_string(error));
        destroy_quantum_gate(gate);
        destroy_quantum_geometric_tensor_network(qgtn);
        shutdown_numerical_backend();
        return 1;
    }

    printf("Quantum geometric tensor[0,0] = %f + %fi\n", result.real, result.imag);

    // Cleanup
    destroy_quantum_gate(gate);
    destroy_quantum_geometric_tensor_network(qgtn);
    shutdown_numerical_backend();
    
    printf("Test completed successfully!\n");
    return 0;
}
