#include "quantum_geometric/core/quantum_geometric_gradient.h"
#include "quantum_geometric/core/quantum_gate_operations.h"
#include "quantum_geometric/core/numerical_backend.h"
#include "quantum_geometric/core/error_handling.h"
#include "quantum_geometric/core/advanced_memory_system.h"
#include "quantum_geometric/core/memory_singleton.h"
#include "quantum_geometric/core/quantum_geometric_compute.h"
#include "quantum_geometric/core/geometric_processor.h"
#include "quantum_geometric/core/computational_graph.h"
#include "quantum_geometric/core/quantum_geometric_tensor_network.h"
#include "quantum_geometric/core/quantum_parameter_shift.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stddef.h>

bool compute_gradient_with_error(
    const quantum_geometric_tensor_network_t* qgtn,
    size_t param_idx,
    ComplexFloat** gradient,
    double* error_estimate,
    size_t* dimension) {
    
    printf("DEBUG: Starting compute_gradient_with_error\n");
    printf("DEBUG: param_idx=%zu\n", param_idx);
    
    // Use adaptive step sizes for Richardson extrapolation
    // We'll use 4 different step sizes in a geometric sequence
    const double base_step = M_PI / 16.0;  // Smaller base step size for better accuracy
    const int num_steps = 4;               // Number of step sizes to use
    const double step_ratio = 2.0;         // Ratio between consecutive step sizes
    
    // Allocate arrays for multiple gradients
    ComplexFloat** gradients = malloc(num_steps * sizeof(ComplexFloat*));
    size_t* dims = malloc(num_steps * sizeof(size_t));
    
    if (!gradients || !dims) {
        printf("DEBUG: Failed to allocate memory for gradient arrays\n");
        free(gradients);
        free(dims);
        return false;
    }
    
    // Initialize gradient pointers to NULL
    for (int i = 0; i < num_steps; i++) {
        gradients[i] = NULL;
    }
    
    // Compute gradients with different step sizes
    bool success = true;
    size_t common_dim = 0;
    
    for (int i = 0; i < num_steps; i++) {
        double step_size = base_step * pow(step_ratio, i);
        printf("DEBUG: Computing gradient with step size %.6f\n", step_size);
        
        if (!compute_centered_difference_gradient(qgtn, param_idx, step_size, &gradients[i], &dims[i])) {
            printf("DEBUG: Failed to compute gradient with step size %.6f\n", step_size);
            success = false;
            break;
        }
        
        // Check dimension consistency
        if (i == 0) {
            common_dim = dims[i];
        } else if (dims[i] != common_dim) {
            printf("DEBUG: Dimension mismatch between gradients: %zu vs %zu\n", dims[i], common_dim);
            success = false;
            break;
        }
    }
    
    // If any gradient computation failed, clean up and return
    if (!success) {
        for (int i = 0; i < num_steps; i++) {
            if (gradients[i]) free(gradients[i]);
        }
        free(gradients);
        free(dims);
        return false;
    }
    
    // Allocate output gradient array
    *gradient = malloc(common_dim * sizeof(ComplexFloat));
    if (!*gradient) {
        printf("DEBUG: Failed to allocate gradient array\n");
        for (int i = 0; i < num_steps; i++) {
            free(gradients[i]);
        }
        free(gradients);
        free(dims);
        return false;
    }
    
    // Perform Richardson extrapolation to get higher-order accuracy
    printf("DEBUG: Performing Richardson extrapolation\n");
    
    // Initialize with the finest step size gradient
    for (size_t i = 0; i < common_dim; i++) {
        (*gradient)[i] = gradients[0][i];
    }
    
    // Apply Richardson extrapolation formula
    for (int k = 1; k < num_steps; k++) {
        double factor = pow(step_ratio, 2 * k);
        double weight = factor / (factor - 1.0);
        
        for (size_t i = 0; i < common_dim; i++) {
            // Extrapolated value = weight * fine_step - (weight-1) * coarse_step
            (*gradient)[i].real = weight * (*gradient)[i].real - (weight - 1.0) * gradients[k][i].real;
            (*gradient)[i].imag = weight * (*gradient)[i].imag - (weight - 1.0) * gradients[k][i].imag;
        }
    }
    
    // Compute error estimate using the difference between the extrapolated result
    // and the finest step size gradient
    double total_error = 0.0;
    printf("DEBUG: Computing error estimate\n");
    
    for (size_t i = 0; i < common_dim; i++) {
        double real_diff = (*gradient)[i].real - gradients[0][i].real;
        double imag_diff = (*gradient)[i].imag - gradients[0][i].imag;
        
        // Add to total error (using L2 norm of differences)
        total_error += real_diff * real_diff + imag_diff * imag_diff;
        
        if (i < 4) {
            printf("DEBUG: Gradient[%zu]: (%.6f,%.6f)\n", 
                   i, (*gradient)[i].real, (*gradient)[i].imag);
            printf("DEBUG: Difference[%zu]: (%.6f,%.6f)\n",
                   i, real_diff, imag_diff);
        }
    }
    
    // Compute RMS error
    *error_estimate = sqrt(total_error / common_dim);
    printf("DEBUG: Error estimate: %.6f\n", *error_estimate);
    
    *dimension = common_dim;
    
    // Clean up
    for (int i = 0; i < num_steps; i++) {
        free(gradients[i]);
    }
    free(gradients);
    free(dims);
    
    return true;
}

bool compute_centered_difference_gradient(
    const quantum_geometric_tensor_network_t* qgtn,
    size_t param_idx,
    double step_size,
    ComplexFloat** gradient,
    size_t* dimension) {
    
    printf("DEBUG: Starting compute_centered_difference_gradient\n");
    printf("DEBUG: param_idx=%zu, step_size=%.6f\n", param_idx, step_size);
    
    // Create a copy of qgtn since we need to modify it
    quantum_geometric_tensor_network_t* qgtn_copy = copy_quantum_geometric_tensor_network(qgtn);
    if (!qgtn_copy) {
        printf("DEBUG: Failed to create copy of quantum geometric tensor network\n");
        return false;
    }
    
    // Get forward and backward shifted states
    ComplexFloat* forward_state = NULL;
    ComplexFloat* backward_state = NULL;
    size_t state_dim;
    
    printf("DEBUG: Computing shifted states\n");
    if (!compute_shifted_states(qgtn_copy, param_idx, step_size,
                              &forward_state, &backward_state, &state_dim)) {
        printf("DEBUG: compute_shifted_states failed\n");
        destroy_quantum_geometric_tensor_network(qgtn_copy);
        return false;
    }
    printf("DEBUG: Shifted states computed successfully\n");
    
    // Allocate gradient array
    *gradient = malloc(state_dim * sizeof(ComplexFloat));
    if (!*gradient) {
        printf("DEBUG: Failed to allocate gradient array\n");
        free(forward_state);
        free(backward_state);
        destroy_quantum_geometric_tensor_network(qgtn_copy);
        return false;
    }
    
    // Compute gradient using centered difference formula
    printf("DEBUG: Computing gradient using centered difference formula\n");
    for (size_t i = 0; i < state_dim; i++) {
        (*gradient)[i].real = (forward_state[i].real - backward_state[i].real) / (2.0 * step_size);
        (*gradient)[i].imag = (forward_state[i].imag - backward_state[i].imag) / (2.0 * step_size);
        
        if (i < 4) {
            printf("DEBUG: Gradient[%zu]: (%.6f,%.6f)\n", 
                   i, (*gradient)[i].real, (*gradient)[i].imag);
        }
    }
    
    *dimension = state_dim;
    
    // Clean up
    free(forward_state);
    free(backward_state);
    destroy_quantum_geometric_tensor_network(qgtn_copy);
    
    return true;
}

bool compute_parameter_shift_gradient(
    const quantum_geometric_tensor_network_t* qgtn,
    size_t param_idx,
    double shift_amount,
    ComplexFloat** gradient,
    size_t* dimension) {
    
    printf("DEBUG: Starting compute_parameter_shift_gradient\n");
    printf("DEBUG: param_idx=%zu, shift_amount=%.6f\n", param_idx, shift_amount);
    
    // Create a copy of qgtn since we need to modify it
    quantum_geometric_tensor_network_t* qgtn_copy = copy_quantum_geometric_tensor_network(qgtn);
    if (!qgtn_copy) {
        printf("DEBUG: Failed to create copy of quantum geometric tensor network\n");
        return false;
    }
    
    // Get forward and backward shifted states
    ComplexFloat* forward_state = NULL;
    ComplexFloat* backward_state = NULL;
    size_t state_dim;
    
    printf("DEBUG: Computing shifted states\n");
    if (!compute_shifted_states(qgtn_copy, param_idx, shift_amount,
                              &forward_state, &backward_state, &state_dim)) {
        printf("DEBUG: compute_shifted_states failed\n");
        destroy_quantum_geometric_tensor_network(qgtn_copy);
        return false;
    }
    printf("DEBUG: Shifted states computed successfully\n");
    
    // Allocate gradient array
    *gradient = malloc(state_dim * sizeof(ComplexFloat));
    if (!*gradient) {
        printf("DEBUG: Failed to allocate gradient array\n");
        free(forward_state);
        free(backward_state);
        destroy_quantum_geometric_tensor_network(qgtn_copy);
        return false;
    }
    
    // Compute gradient using parameter shift rule
    printf("DEBUG: Computing gradient using parameter shift rule\n");
    for (size_t i = 0; i < state_dim; i++) {
        (*gradient)[i].real = (forward_state[i].real - backward_state[i].real) / (2.0 * shift_amount);
        (*gradient)[i].imag = (forward_state[i].imag - backward_state[i].imag) / (2.0 * shift_amount);
        
        if (i < 4) {
            printf("DEBUG: Gradient[%zu]: (%.6f,%.6f)\n", 
                   i, (*gradient)[i].real, (*gradient)[i].imag);
        }
    }
    
    *dimension = state_dim;
    
    // Clean up
    free(forward_state);
    free(backward_state);
    destroy_quantum_geometric_tensor_network(qgtn_copy);
    
    return true;
}

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
    
    // Get global memory system
    advanced_memory_system_t* memory = get_global_memory_system();
    if (!memory) {
        // Create memory system if it doesn't exist
        memory_system_config_t mem_config = {
            .type = MEM_SYSTEM_QUANTUM,
            .strategy = ALLOC_STRATEGY_BUDDY,
            .optimization = MEM_OPT_ADVANCED,
            .alignment = sizeof(ComplexFloat),
            .enable_monitoring = true,
            .enable_defragmentation = true
        };
        memory = create_memory_system(&mem_config);
        if (!memory) {
            printf("DEBUG: Failed to create memory system\n");
            return false;
        }
    }
    
    // Allocate states using safe memory allocation
    *forward_state = safe_memory_allocate(memory, state_dim * sizeof(ComplexFloat), sizeof(ComplexFloat));
    *backward_state = safe_memory_allocate(memory, state_dim * sizeof(ComplexFloat), sizeof(ComplexFloat));
    if (!*forward_state || !*backward_state) {
        if (*forward_state) safe_memory_free(memory, *forward_state);
        if (*backward_state) safe_memory_free(memory, *backward_state);
        return false;
    }
    
    // Initialize to zero
    memset(*forward_state, 0, state_dim * sizeof(ComplexFloat));
    memset(*backward_state, 0, state_dim * sizeof(ComplexFloat));
    
    // Save original state coordinates
    ComplexFloat* original_coordinates = NULL;
    if (qgtn->circuit->state && qgtn->circuit->state->coordinates) {
        original_coordinates = safe_memory_allocate(memory, state_dim * sizeof(ComplexFloat), sizeof(ComplexFloat));
        if (!original_coordinates) {
            safe_memory_free(memory, *forward_state);
            safe_memory_free(memory, *backward_state);
            return false;
        }
        memcpy(original_coordinates, qgtn->circuit->state->coordinates, 
               state_dim * sizeof(ComplexFloat));
    }
    
    // Initialize quantum state in tensor network
    if (!qgtn->network || !qgtn->network->nodes || qgtn->network->num_nodes < 1) {
        printf("DEBUG: Invalid tensor network state\n");
        if (original_coordinates) safe_memory_free(memory, original_coordinates);
        safe_memory_free(memory, *forward_state);
        safe_memory_free(memory, *backward_state);
        return false;
    }
    printf("DEBUG: Tensor network state valid\n");

    // Initialize state vector with |0> state
    ComplexFloat* init_state = safe_memory_allocate(memory, state_dim * sizeof(ComplexFloat), sizeof(ComplexFloat));
    if (!init_state) {
        printf("DEBUG: Failed to allocate init_state\n");
        if (original_coordinates) safe_memory_free(memory, original_coordinates);
        safe_memory_free(memory, *forward_state);
        safe_memory_free(memory, *backward_state);
        return false;
    }
    memset(init_state, 0, state_dim * sizeof(ComplexFloat));
    init_state[0] = (ComplexFloat){1.0f, 0.0f};  // |0> state
    printf("DEBUG: Initialized |0> state\n");

    // Set initial state in tensor network
    if (!qgtn->network->nodes[0] || !qgtn->network->nodes[0]->data) {
        printf("DEBUG: Invalid tensor network node or data\n");
        safe_memory_free(memory, init_state);
        if (original_coordinates) safe_memory_free(memory, original_coordinates);
        safe_memory_free(memory, *forward_state);
        safe_memory_free(memory, *backward_state);
        return false;
    }
    memcpy(qgtn->network->nodes[0]->data, init_state, state_dim * sizeof(ComplexFloat));
    safe_memory_free(memory, init_state);
    printf("DEBUG: Set initial state in tensor network\n");

    // Forward shift
    printf("DEBUG: Applying forward shift (amount=%.6f)\n", shift_amount);
    if (!shift_parameter(qgtn, param_idx, shift_amount)) {
        printf("DEBUG: Forward shift_parameter failed\n");
        if (original_coordinates) safe_memory_free(memory, original_coordinates);
        safe_memory_free(memory, *forward_state);
        safe_memory_free(memory, *backward_state);
        return false;
    }
    printf("DEBUG: Forward shift applied successfully\n");

    // Apply circuit to get forward state
    printf("DEBUG: Applying quantum circuit for forward state\n");
    if (!apply_quantum_circuit(qgtn, qgtn->circuit)) {
        printf("DEBUG: Forward apply_quantum_circuit failed\n");
        shift_parameter(qgtn, param_idx, -shift_amount); // Restore parameter
        if (original_coordinates) {
            safe_memory_free(memory, original_coordinates);
        }
        safe_memory_free(memory, *forward_state);
        safe_memory_free(memory, *backward_state);
        return false;
    }
    printf("DEBUG: Forward quantum circuit applied successfully\n");
    
    // Copy forward state
    if (qgtn->circuit->state && qgtn->circuit->state->coordinates) {
        memcpy(*forward_state, qgtn->circuit->state->coordinates, 
               state_dim * sizeof(ComplexFloat));
    } else {
        printf("DEBUG: Forward state coordinates are NULL\n");
        if (original_coordinates) {
            safe_memory_free(memory, original_coordinates);
        }
        safe_memory_free(memory, *forward_state);
        safe_memory_free(memory, *backward_state);
        return false;
    }
    
    // Reset state to |0> for backward shift
    printf("DEBUG: Resetting state for backward shift\n");
    init_state = safe_memory_allocate(memory, state_dim * sizeof(ComplexFloat), sizeof(ComplexFloat));
    if (!init_state) {
        printf("DEBUG: Failed to allocate init_state for backward shift\n");
        if (original_coordinates) {
            safe_memory_free(memory, original_coordinates);
        }
        safe_memory_free(memory, *forward_state);
        safe_memory_free(memory, *backward_state);
        return false;
    }
    memset(init_state, 0, state_dim * sizeof(ComplexFloat));
    init_state[0] = (ComplexFloat){1.0f, 0.0f};  // |0> state
    if (qgtn->network && qgtn->network->nodes && qgtn->network->nodes[0] && qgtn->network->nodes[0]->data) {
        memcpy(qgtn->network->nodes[0]->data, init_state, state_dim * sizeof(ComplexFloat));
    } else {
        printf("DEBUG: Network nodes data is NULL\n");
        safe_memory_free(memory, init_state);
        if (original_coordinates) {
            safe_memory_free(memory, original_coordinates);
        }
        safe_memory_free(memory, *forward_state);
        safe_memory_free(memory, *backward_state);
        return false;
    }
    safe_memory_free(memory, init_state);
    printf("DEBUG: Reset state to |0> for backward shift\n");

    // Backward shift
    printf("DEBUG: Applying backward shift (amount=%.6f)\n", -shift_amount);
    if (!shift_parameter(qgtn, param_idx, -shift_amount)) {  // Apply negative shift
        printf("DEBUG: Backward shift_parameter failed\n");
        if (original_coordinates) safe_memory_free(memory, original_coordinates);
        safe_memory_free(memory, *forward_state);
        safe_memory_free(memory, *backward_state);
        return false;
    }
    printf("DEBUG: Backward shift applied successfully\n");

    // Apply circuit to get backward state
    printf("DEBUG: Applying quantum circuit for backward state\n");
    if (!apply_quantum_circuit(qgtn, qgtn->circuit)) {
        printf("DEBUG: Backward apply_quantum_circuit failed\n");
        shift_parameter(qgtn, param_idx, shift_amount); // Restore parameter
        if (original_coordinates) {
            safe_memory_free(memory, original_coordinates);
        }
        safe_memory_free(memory, *forward_state);
        safe_memory_free(memory, *backward_state);
        return false;
    }
    printf("DEBUG: Backward quantum circuit applied successfully\n");
    
    // Copy backward state
    if (qgtn->circuit->state && qgtn->circuit->state->coordinates) {
        memcpy(*backward_state, qgtn->circuit->state->coordinates,
               state_dim * sizeof(ComplexFloat));
    } else {
        printf("DEBUG: Backward state coordinates are NULL\n");
        if (original_coordinates) {
            safe_memory_free(memory, original_coordinates);
        }
        safe_memory_free(memory, *forward_state);
        safe_memory_free(memory, *backward_state);
        return false;
    }
    
    // Restore parameter to original value
    if (!shift_parameter(qgtn, param_idx, shift_amount)) {  // Shift back to original
        printf("DEBUG: Failed to restore parameter to original value\n");
        if (original_coordinates) safe_memory_free(memory, original_coordinates);
        safe_memory_free(memory, *forward_state);
        safe_memory_free(memory, *backward_state);
        return false;
    }
    printf("DEBUG: Parameter restored to original value\n");
    
    // Restore original state if it existed
    printf("DEBUG: Restoring original state, original_coordinates=%p\n", (void*)original_coordinates);
    if (original_coordinates) {
        if (qgtn->circuit->state && qgtn->circuit->state->coordinates) {
            printf("DEBUG: Circuit state coordinates exist at %p\n", (void*)qgtn->circuit->state->coordinates);
            // Simply copy the original coordinates back to the existing buffer
            // This avoids unnecessary free/malloc cycles and potential memory leaks
            printf("DEBUG: Copying original coordinates back to circuit state\n");
            memcpy(qgtn->circuit->state->coordinates, original_coordinates, 
                   state_dim * sizeof(ComplexFloat));
            printf("DEBUG: Original coordinates copied successfully\n");
        } else {
            printf("DEBUG: Circuit state coordinates are NULL, cannot restore\n");
        }
        
        // Free our backup of the original coordinates
        printf("DEBUG: Freeing original coordinates backup\n");
        safe_memory_free(memory, original_coordinates);
        printf("DEBUG: Original coordinates freed successfully\n");
    } else {
        printf("DEBUG: No original coordinates to restore\n");
    }
    
    *dimension = state_dim;
    return true;
}
