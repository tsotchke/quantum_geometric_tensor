#ifndef PARAMETER_UPDATE_H
#define PARAMETER_UPDATE_H

#include "quantum_geometric/hybrid/classical_optimization_engine.h"
#include "quantum_geometric/hybrid/quantum_machine_learning.h"
#include <math.h>

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

// Update quantum circuit parameters using Adam-style optimization
// Applies gradients stored in optimizer to circuit gate parameters
static inline void update_circuit_quantum_parameters(quantum_circuit* circuit,
                                                     OptimizationContext* optimizer) {
    if (!circuit || !optimizer) return;

    // Update parameters stored in the circuit's gates
    // Each gate may have rotation parameters that need to be optimized
    if (circuit->num_gates > 0 && circuit->gates && optimizer->gradients) {
        size_t param_idx = 0;
        for (size_t g = 0; g < circuit->num_gates && param_idx < optimizer->num_parameters; g++) {
            quantum_gate_t* gate = circuit->gates[g];
            if (gate && gate->num_parameters > 0 && gate->parameters) {
                for (size_t p = 0; p < gate->num_parameters && param_idx < optimizer->num_parameters; p++) {
                    double grad = optimizer->gradients[param_idx];

                    // Apply Adam-style update for adaptive learning rates
                    double m_hat = grad;
                    double v_hat = 1.0;

                    // First moment (momentum) update
                    if (optimizer->momentum) {
                        optimizer->momentum[param_idx] = optimizer->beta1 * optimizer->momentum[param_idx]
                                                       + (1.0 - optimizer->beta1) * grad;
                        m_hat = optimizer->momentum[param_idx] / (1.0 - optimizer->beta1);
                    }

                    // Second moment (velocity/variance) update
                    if (optimizer->velocity) {
                        optimizer->velocity[param_idx] = optimizer->beta2 * optimizer->velocity[param_idx]
                                                       + (1.0 - optimizer->beta2) * grad * grad;
                        v_hat = optimizer->velocity[param_idx] / (1.0 - optimizer->beta2);
                    }

                    // Apply adaptive gradient update: θ = θ - lr * m_hat / (sqrt(v_hat) + ε)
                    gate->parameters[p] -= optimizer->learning_rate * m_hat / (sqrt(v_hat) + optimizer->epsilon);
                    param_idx++;
                }
            }
        }
    }
}

#endif // PARAMETER_UPDATE_H
