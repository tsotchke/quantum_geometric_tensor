#include "quantum_geometric/core/quantum_geometric_gradient.h"
#include "quantum_geometric/core/quantum_gate_operations.h"
#include "quantum_geometric/core/numerical_backend.h"
#include "quantum_geometric/core/error_handling.h"
#include "quantum_geometric/core/advanced_memory_system.h"
#include "quantum_geometric/core/quantum_geometric_compute.h"
#include "quantum_geometric/core/geometric_processor.h"
#include "quantum_geometric/core/computational_graph.h"
#include "quantum_geometric/core/quantum_geometric_tensor_network.h"
#include "quantum_geometric/core/quantum_parameter_shift.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stddef.h>

// Helper function to find gate containing parameter
static quantum_gate_t* find_parameterized_gate(
    quantum_geometric_tensor_network_t* qgtn,
    size_t param_idx,
    size_t* layer_idx,
    size_t* gate_idx) {
    
    printf("DEBUG: Starting find_parameterized_gate\n");
    printf("DEBUG: Looking for param_idx=%zu\n", param_idx);
    
    if (!qgtn || !qgtn->circuit) {
        printf("DEBUG: Invalid qgtn or circuit\n");
        return NULL;
    }
    
    printf("DEBUG: Circuit has %zu layers\n", qgtn->num_layers);
    printf("DEBUG: Circuit state: %p\n", (void*)qgtn->circuit->state);
    if (qgtn->circuit->state) {
        printf("DEBUG: Circuit state dimension: %zu\n", qgtn->circuit->state->dimension);
    }
    
    size_t current_param = 0;
    
    for (size_t l = 0; l < qgtn->num_layers; l++) {
        circuit_layer_t* layer = qgtn->circuit->layers[l];
        if (!layer) {
            printf("DEBUG: Layer %zu is NULL\n", l);
            continue;
        }
        printf("DEBUG: Layer %zu has %zu gates, is_parameterized=%d\n", 
               l, layer->num_gates, layer->is_parameterized);
        
        for (size_t g = 0; g < layer->num_gates; g++) {
            quantum_gate_t* gate = layer->gates[g];
            if (!gate) {
                printf("DEBUG: Gate %zu in layer %zu is NULL\n", g, l);
                continue;
            }
            printf("DEBUG: Gate %zu in layer %zu: type=%d, is_parameterized=%d, num_params=%zu\n", 
                   g, l, gate->type, gate->is_parameterized, gate->num_parameters);
            
            if (gate->is_parameterized && gate->parameters) {
                printf("DEBUG: Gate parameters: [%.6f]\n", gate->parameters[0]);
            }
            
            if (gate->is_parameterized) {
                printf("DEBUG: Found parameterized gate, current_param=%zu\n", current_param);
                if (current_param == param_idx) {
                    printf("DEBUG: Found target gate at layer %zu, index %zu\n", l, g);
                    *layer_idx = l;
                    *gate_idx = g;
                    return gate;
                }
                current_param++;
            }
        }
    }
    
    printf("DEBUG: No matching parameterized gate found\n");
    return NULL;
}

bool shift_parameter(
    quantum_geometric_tensor_network_t* qgtn,
    size_t param_idx,
    double shift_amount) {
    
    printf("DEBUG: Starting shift_parameter\n");
    printf("DEBUG: param_idx=%zu, shift_amount=%.6f\n", param_idx, shift_amount);
    
    // Find gate containing parameter
    size_t layer_idx, gate_idx;
    quantum_gate_t* gate = find_parameterized_gate(qgtn, param_idx, &layer_idx, &gate_idx);
    if (!gate) {
        printf("DEBUG: Failed to find parameterized gate\n");
        return false;
    }
    printf("DEBUG: Found gate at layer %zu, index %zu\n", layer_idx, gate_idx);
    printf("DEBUG: Gate type=%d, num_qubits=%zu, is_parameterized=%d\n", 
           gate->type, gate->num_qubits, gate->is_parameterized);
    
    // Store original parameter
    double original_param = gate->parameters[0];
    printf("DEBUG: Original parameter=%.6f\n", original_param);
    
    // Apply shift
    gate->parameters[0] += shift_amount;
    printf("DEBUG: New parameter=%.6f\n", gate->parameters[0]);
    
    // Update gate matrix based on type
    ComplexFloat new_matrix[4];
    printf("DEBUG: Updating gate matrix for type %d\n", gate->type);
    switch (gate->type) {
        case GATE_TYPE_RX:
            {
                double cos_half = cos(gate->parameters[0] / 2.0);
                double sin_half = sin(gate->parameters[0] / 2.0);
                // Rx = [cos(θ/2)    -i*sin(θ/2)]
                //      [-i*sin(θ/2)   cos(θ/2) ]
                new_matrix[0] = (ComplexFloat){cos_half, 0.0};
                new_matrix[1] = (ComplexFloat){0.0, -sin_half};
                new_matrix[2] = (ComplexFloat){0.0, -sin_half};
                new_matrix[3] = (ComplexFloat){cos_half, 0.0};
            }
            break;
            
        case GATE_TYPE_RY:
            {
                double cos_half = cos(gate->parameters[0] / 2.0);
                double sin_half = sin(gate->parameters[0] / 2.0);
                // Ry = [cos(θ/2)    -sin(θ/2)]
                //      [sin(θ/2)     cos(θ/2)]
                new_matrix[0] = (ComplexFloat){cos_half, 0.0};
                new_matrix[1] = (ComplexFloat){-sin_half, 0.0};
                new_matrix[2] = (ComplexFloat){sin_half, 0.0};
                new_matrix[3] = (ComplexFloat){cos_half, 0.0};
            }
            break;
            
        case GATE_TYPE_RZ:
            {
                double cos_half = cos(gate->parameters[0] / 2.0);
                double sin_half = sin(gate->parameters[0] / 2.0);
                // Rz = [e^(-iθ/2)    0        ]
                //      [0            e^(iθ/2)  ]
                new_matrix[0] = (ComplexFloat){cos_half, -sin_half};
                new_matrix[1] = (ComplexFloat){0.0, 0.0};
                new_matrix[2] = (ComplexFloat){0.0, 0.0};
                new_matrix[3] = (ComplexFloat){cos_half, sin_half};
            }
            break;
            
        default:
            gate->parameters[0] = original_param;
            return false;
    }
    
    // Update gate matrix
    memcpy(gate->matrix, new_matrix, 4 * sizeof(ComplexFloat));
    printf("DEBUG: Updated gate matrix:\n");
    for (int i = 0; i < 4; i++) {
        printf("  [%d]: (%.6f,%.6f)\n", i, new_matrix[i].real, new_matrix[i].imag);
    }
    
    return true;
}

bool compute_shifted_states(
    quantum_geometric_tensor_network_t* qgtn,
    size_t param_idx,
    double shift_amount,
    ComplexFloat** forward_state,
    ComplexFloat** backward_state,
    size_t* dimension) {
    
    printf("DEBUG: Starting compute_shifted_states\n");
    printf("DEBUG: param_idx=%zu, shift_amount=%.6f\n", param_idx, shift_amount);
    
    // Get state dimension
    size_t state_dim = 1 << qgtn->num_qubits;
    printf("DEBUG: state_dim=%zu (num_qubits=%zu)\n", state_dim, qgtn->num_qubits);
    
    // Allocate states
    *forward_state = calloc(state_dim, sizeof(ComplexFloat));
    *backward_state = calloc(state_dim, sizeof(ComplexFloat));
    if (!*forward_state || !*backward_state) {
        free(*forward_state);
        free(*backward_state);
        return false;
    }
    
    // Save original state coordinates
    ComplexFloat* original_coordinates = NULL;
    if (qgtn->circuit->state && qgtn->circuit->state->coordinates) {
        original_coordinates = malloc(state_dim * sizeof(ComplexFloat));
        if (!original_coordinates) {
            free(*forward_state);
            free(*backward_state);
            return false;
        }
        memcpy(original_coordinates, qgtn->circuit->state->coordinates, 
               state_dim * sizeof(ComplexFloat));
    }
    
    // Initialize quantum state in tensor network
    if (!qgtn->network || !qgtn->network->nodes || qgtn->network->num_nodes < 1) {
        printf("DEBUG: Invalid tensor network state\n");
        free(original_coordinates);
        free(*forward_state);
        free(*backward_state);
        return false;
    }
    printf("DEBUG: Tensor network state valid\n");

    // Initialize state vector with |0> state
    ComplexFloat* init_state = calloc(state_dim, sizeof(ComplexFloat));
    if (!init_state) {
        printf("DEBUG: Failed to allocate init_state\n");
        free(original_coordinates);
        free(*forward_state);
        free(*backward_state);
        return false;
    }
    init_state[0] = (ComplexFloat){1.0f, 0.0f};  // |0> state
    printf("DEBUG: Initialized |0> state\n");

    // Set initial state in tensor network
    if (!qgtn->network->nodes[0] || !qgtn->network->nodes[0]->data) {
        printf("DEBUG: Invalid tensor network node or data\n");
        free(init_state);
        free(original_coordinates);
        free(*forward_state);
        free(*backward_state);
        return false;
    }
    memcpy(qgtn->network->nodes[0]->data, init_state, state_dim * sizeof(ComplexFloat));
    free(init_state);
    printf("DEBUG: Set initial state in tensor network\n");

    // Forward shift
    printf("DEBUG: Applying forward shift (amount=%.6f)\n", shift_amount);
    if (!shift_parameter(qgtn, param_idx, shift_amount)) {
        printf("DEBUG: Forward shift_parameter failed\n");
        free(original_coordinates);
        free(*forward_state);
        free(*backward_state);
        return false;
    }
    printf("DEBUG: Forward shift applied successfully\n");

    // Apply circuit to get forward state
    printf("DEBUG: Applying quantum circuit for forward state\n");
    if (!apply_quantum_circuit(qgtn, qgtn->circuit)) {
        printf("DEBUG: Forward apply_quantum_circuit failed\n");
        shift_parameter(qgtn, param_idx, -shift_amount); // Restore parameter
        free(original_coordinates);
        free(*forward_state);
        free(*backward_state);
        return false;
    }
    printf("DEBUG: Forward quantum circuit applied successfully\n");
    
    // Copy forward state
    memcpy(*forward_state, qgtn->circuit->state->coordinates, 
           state_dim * sizeof(ComplexFloat));
    
    // Reset state to |0> for backward shift
    printf("DEBUG: Resetting state for backward shift\n");
    init_state = calloc(state_dim, sizeof(ComplexFloat));
    if (!init_state) {
        printf("DEBUG: Failed to allocate init_state for backward shift\n");
        free(original_coordinates);
        free(*forward_state);
        free(*backward_state);
        return false;
    }
    init_state[0] = (ComplexFloat){1.0f, 0.0f};  // |0> state
    memcpy(qgtn->network->nodes[0]->data, init_state, state_dim * sizeof(ComplexFloat));
    free(init_state);
    printf("DEBUG: Reset state to |0> for backward shift\n");

    // Backward shift
    printf("DEBUG: Applying backward shift (amount=%.6f)\n", -2 * shift_amount);
    if (!shift_parameter(qgtn, param_idx, -2 * shift_amount)) {  // -2x to go back from +x to -x
        printf("DEBUG: Backward shift_parameter failed\n");
        free(original_coordinates);
        free(*forward_state);
        free(*backward_state);
        return false;
    }
    printf("DEBUG: Backward shift applied successfully\n");

    // Apply circuit to get backward state
    printf("DEBUG: Applying quantum circuit for backward state\n");
    if (!apply_quantum_circuit(qgtn, qgtn->circuit)) {
        printf("DEBUG: Backward apply_quantum_circuit failed\n");
        shift_parameter(qgtn, param_idx, shift_amount); // Restore parameter
        free(original_coordinates);
        free(*forward_state);
        free(*backward_state);
        return false;
    }
    printf("DEBUG: Backward quantum circuit applied successfully\n");
    
    // Copy backward state
    memcpy(*backward_state, qgtn->circuit->state->coordinates,
           state_dim * sizeof(ComplexFloat));
    
    // Restore original state if it existed
    if (original_coordinates) {
        if (qgtn->circuit->state->coordinates) {
            free(qgtn->circuit->state->coordinates);
        }
        qgtn->circuit->state->coordinates = original_coordinates;
    }
    
    *dimension = state_dim;
    return true;
}
