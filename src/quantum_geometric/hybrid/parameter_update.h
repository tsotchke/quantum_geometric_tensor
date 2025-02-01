#ifndef PARAMETER_UPDATE_H
#define PARAMETER_UPDATE_H

#include "quantum_geometric/hybrid/classical_optimization_engine.h"
#include "quantum_geometric/hybrid/quantum_machine_learning.h"

// Update classical network parameters using optimizer
static inline void update_classical_parameters(ClassicalNetwork* network,
                                            OptimizationContext* optimizer) {
    if (!network || !optimizer) return;
    
    size_t param_offset = 0;
    
    // Update weights and biases for each layer
    for (size_t i = 0; i < network->num_layers; i++) {
        // Calculate layer sizes based on MNIST dimensions
        size_t curr_size = (i == network->num_layers - 1) ? 
                          10 : 32;  // Output size = 10, Hidden size = 32
        size_t prev_size = (i == 0) ? 
                          784 : 32;  // Input size = 784, Hidden size = 32
        
        // Update weights
        size_t num_weights = prev_size * curr_size;
        for (size_t j = 0; j < num_weights; j++) {
            network->weights[i][j] -= optimizer->learning_rate * 
                                    optimizer->gradients[param_offset + j];
        }
        param_offset += num_weights;
        
        // Update biases
        for (size_t j = 0; j < curr_size; j++) {
            network->biases[i][j] -= optimizer->learning_rate * 
                                   optimizer->gradients[param_offset + j];
        }
        param_offset += curr_size;
    }
}

// Update quantum circuit parameters
static inline void update_quantum_parameters(quantum_circuit* circuit,
                                          OptimizationContext* optimizer) {
    // Implementation depends on quantum_circuit structure
    // This is a placeholder that would be implemented based on the
    // actual quantum circuit parameter update requirements
}

#endif // PARAMETER_UPDATE_H
