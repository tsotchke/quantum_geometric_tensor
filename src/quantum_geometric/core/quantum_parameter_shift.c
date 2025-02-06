#include "quantum_geometric/core/quantum_geometric_gradient.h"
#include "quantum_geometric/core/quantum_gate_operations.h"
#include "quantum_geometric/core/numerical_backend.h"
#include "quantum_geometric/core/error_handling.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Helper function to find gate containing parameter
static quantum_gate_t* find_parameterized_gate(
    quantum_geometric_tensor_network_t* qgtn,
    size_t param_idx,
    size_t* layer_idx,
    size_t* gate_idx) {
    
    size_t current_param = 0;
    
    for (size_t l = 0; l < qgtn->num_layers; l++) {
        circuit_layer_t* layer = qgtn->circuit->layers[l];
        for (size_t g = 0; g < layer->num_gates; g++) {
            quantum_gate_t* gate = layer->gates[g];
            if (gate->is_parameterized) {
                if (current_param == param_idx) {
                    *layer_idx = l;
                    *gate_idx = g;
                    return gate;
                }
                current_param++;
            }
        }
    }
    
    return NULL;
}

// Helper function to shift a parameter in the circuit
static bool shift_parameter(
    quantum_geometric_tensor_network_t* qgtn,
    size_t param_idx,
    double shift_amount) {
    
    // Find gate containing parameter
    size_t layer_idx, gate_idx;
    quantum_gate_t* gate = find_parameterized_gate(qgtn, param_idx, &layer_idx, &gate_idx);
    if (!gate) {
        return false;
    }
    
    // Store original parameter
    double original_param = gate->parameters[0];
    
    // Apply shift
    gate->parameters[0] += shift_amount;
    
    // Update gate matrix based on type
    ComplexFloat new_matrix[4];
    switch (gate->type) {
        case GATE_TYPE_RX:
            // Rx = [cos(θ/2)   -i*sin(θ/2)]
            //      [-i*sin(θ/2)  cos(θ/2) ]
            {
                double cos_half = cos(gate->parameters[0] / 2.0);
                double sin_half = sin(gate->parameters[0] / 2.0);
                new_matrix[0] = (ComplexFloat){cos_half, 0};
                new_matrix[1] = (ComplexFloat){0, -sin_half};
                new_matrix[2] = (ComplexFloat){0, -sin_half};
                new_matrix[3] = (ComplexFloat){cos_half, 0};
            }
            break;
            
        case GATE_TYPE_RY:
            // Ry = [cos(θ/2)   -sin(θ/2)]
            //      [sin(θ/2)    cos(θ/2)]
            {
                double cos_half = cos(gate->parameters[0] / 2.0);
                double sin_half = sin(gate->parameters[0] / 2.0);
                new_matrix[0] = (ComplexFloat){cos_half, 0};
                new_matrix[1] = (ComplexFloat){-sin_half, 0};
                new_matrix[2] = (ComplexFloat){sin_half, 0};
                new_matrix[3] = (ComplexFloat){cos_half, 0};
            }
            break;
            
        case GATE_TYPE_RZ:
            // Rz = [e^(-iθ/2)      0    ]
            //      [    0      e^(iθ/2) ]
            {
                double cos_half = cos(gate->parameters[0] / 2.0);
                double sin_half = sin(gate->parameters[0] / 2.0);
                new_matrix[0] = (ComplexFloat){cos_half, -sin_half};
                new_matrix[1] = (ComplexFloat){0, 0};
                new_matrix[2] = (ComplexFloat){0, 0};
                new_matrix[3] = (ComplexFloat){cos_half, sin_half};
            }
            break;
            
        default:
            gate->parameters[0] = original_param;
            return false;
    }
    
    // Update gate matrix
    memcpy(gate->matrix, new_matrix, 4 * sizeof(ComplexFloat));
    
    // Rebuild tensor network
    destroy_tensor_network(qgtn->network);
    qgtn->network = create_tensor_network();
    if (!qgtn->network) {
        gate->parameters[0] = original_param;
        return false;
    }
    
    // Reapply circuit
    for (size_t l = 0; l < qgtn->circuit->num_layers; l++) {
        circuit_layer_t* layer = qgtn->circuit->layers[l];
        if (layer) {
            for (size_t g = 0; g < layer->num_gates; g++) {
                quantum_gate_t* current_gate = layer->gates[g];
                size_t node_id;
                if (!add_tensor_node(qgtn->network, current_gate->matrix,
                                   &current_gate->num_qubits, 1, &node_id)) {
                    gate->parameters[0] = original_param;
                    return false;
                }
                
                for (size_t i = 0; i < current_gate->num_qubits; i++) {
                    if (!connect_tensor_nodes(qgtn->network, node_id, i,
                                            current_gate->target_qubits[i], 0)) {
                        gate->parameters[0] = original_param;
                        return false;
                    }
                }
            }
        }
    }
    
    return true;
}

// Helper function to compute forward and backward shifted states
static bool compute_shifted_states(
    quantum_geometric_tensor_network_t* qgtn,
    size_t param_idx,
    double shift_amount,
    ComplexFloat** forward_state,
    ComplexFloat** backward_state,
    size_t* dimension) {
    
    // Get original state
    ComplexFloat* original_state;
    size_t state_dim;
    if (!get_quantum_state(qgtn, &original_state, &state_dim)) {
        return false;
    }
    free(original_state);  // We don't need the original state
    
    // Allocate states
    *forward_state = malloc(state_dim * sizeof(ComplexFloat));
    *backward_state = malloc(state_dim * sizeof(ComplexFloat));
    if (!*forward_state || !*backward_state) {
        free(*forward_state);
        free(*backward_state);
        return false;
    }
    
    // Forward shift
    if (!shift_parameter(qgtn, param_idx, shift_amount)) {
        free(*forward_state);
        free(*backward_state);
        return false;
    }
    
    if (!get_quantum_state(qgtn, forward_state, &state_dim)) {
        free(*forward_state);
        free(*backward_state);
        return false;
    }
    
    // Backward shift
    if (!shift_parameter(qgtn, param_idx, -2 * shift_amount)) {  // -2x to go back from +x to -x
        free(*forward_state);
        free(*backward_state);
        return false;
    }
    
    if (!get_quantum_state(qgtn, backward_state, &state_dim)) {
        free(*forward_state);
        free(*backward_state);
        return false;
    }
    
    // Reset parameter
    if (!shift_parameter(qgtn, param_idx, shift_amount)) {  // +x to go back to original
        free(*forward_state);
        free(*backward_state);
        return false;
    }
    
    *dimension = state_dim;
    return true;
}

// Compute gradient using parameter shift rule
// Helper function to create a mutable copy of a quantum geometric tensor network
static quantum_geometric_tensor_network_t* copy_quantum_geometric_tensor_network(
    const quantum_geometric_tensor_network_t* qgtn) {
    
    quantum_geometric_tensor_network_t* copy = malloc(sizeof(quantum_geometric_tensor_network_t));
    if (!copy) return NULL;
    
    // Copy basic fields
    copy->num_qubits = qgtn->num_qubits;
    copy->num_layers = qgtn->num_layers;
    copy->is_distributed = qgtn->is_distributed;
    copy->use_hardware_acceleration = qgtn->use_hardware_acceleration;
    
    // Create new tensor network
    copy->network = create_tensor_network();
    if (!copy->network) {
        free(copy);
        return NULL;
    }
    
    // Copy circuit
    copy->circuit = malloc(sizeof(quantum_circuit_t));
    if (!copy->circuit) {
        destroy_tensor_network(copy->network);
        free(copy);
        return NULL;
    }
    
    copy->circuit->layers = malloc(qgtn->num_layers * sizeof(circuit_layer_t*));
    if (!copy->circuit->layers) {
        free(copy->circuit);
        destroy_tensor_network(copy->network);
        free(copy);
        return NULL;
    }
    
    copy->circuit->num_layers = qgtn->num_layers;
    copy->circuit->num_qubits = qgtn->num_qubits;
    copy->circuit->is_parameterized = qgtn->circuit->is_parameterized;
    
    // Copy each layer
    for (size_t l = 0; l < qgtn->num_layers; l++) {
        if (qgtn->circuit->layers[l]) {
            circuit_layer_t* layer = malloc(sizeof(circuit_layer_t));
            if (!layer) {
                // Clean up on failure
                for (size_t j = 0; j < l; j++) {
                    if (copy->circuit->layers[j]) {
                        for (size_t g = 0; g < copy->circuit->layers[j]->num_gates; g++) {
                            free(copy->circuit->layers[j]->gates[g]);
                        }
                        free(copy->circuit->layers[j]->gates);
                        free(copy->circuit->layers[j]);
                    }
                }
                free(copy->circuit->layers);
                free(copy->circuit);
                destroy_tensor_network(copy->network);
                free(copy);
                return NULL;
            }
            
            layer->num_gates = qgtn->circuit->layers[l]->num_gates;
            layer->is_parameterized = qgtn->circuit->layers[l]->is_parameterized;
            
            layer->gates = malloc(layer->num_gates * sizeof(quantum_gate_t*));
            if (!layer->gates) {
                free(layer);
                for (size_t j = 0; j < l; j++) {
                    if (copy->circuit->layers[j]) {
                        for (size_t g = 0; g < copy->circuit->layers[j]->num_gates; g++) {
                            free(copy->circuit->layers[j]->gates[g]);
                        }
                        free(copy->circuit->layers[j]->gates);
                        free(copy->circuit->layers[j]);
                    }
                }
                free(copy->circuit->layers);
                free(copy->circuit);
                destroy_tensor_network(copy->network);
                free(copy);
                return NULL;
            }
            
            // Copy each gate
            for (size_t g = 0; g < layer->num_gates; g++) {
                layer->gates[g] = copy_quantum_gate(qgtn->circuit->layers[l]->gates[g]);
                if (!layer->gates[g]) {
                    // Clean up on failure
                    for (size_t h = 0; h < g; h++) {
                        free(layer->gates[h]);
                    }
                    free(layer->gates);
                    free(layer);
                    for (size_t j = 0; j < l; j++) {
                        if (copy->circuit->layers[j]) {
                            for (size_t h = 0; h < copy->circuit->layers[j]->num_gates; h++) {
                                free(copy->circuit->layers[j]->gates[h]);
                            }
                            free(copy->circuit->layers[j]->gates);
                            free(copy->circuit->layers[j]);
                        }
                    }
                    free(copy->circuit->layers);
                    free(copy->circuit);
                    destroy_tensor_network(copy->network);
                    free(copy);
                    return NULL;
                }
            }
            
            copy->circuit->layers[l] = layer;
        } else {
            copy->circuit->layers[l] = NULL;
        }
    }
    
    return copy;
}

bool compute_parameter_shift_gradient(
    const quantum_geometric_tensor_network_t* qgtn,
    size_t param_idx,
    double shift_amount,
    ComplexFloat** gradient,
    size_t* dimension) {
    
    if (!qgtn || !gradient || !dimension || shift_amount <= 0.0) {
        return false;
    }
    
    // Create mutable copy
    quantum_geometric_tensor_network_t* qgtn_copy = copy_quantum_geometric_tensor_network(qgtn);
    if (!qgtn_copy) {
        return false;
    }
    
    // Get shifted states
    ComplexFloat* forward_state;
    ComplexFloat* backward_state;
    size_t state_dim;
    if (!compute_shifted_states(qgtn_copy, param_idx, shift_amount,
                              &forward_state, &backward_state, &state_dim)) {
        destroy_quantum_geometric_tensor_network(qgtn_copy);
        return false;
    }
    
    // Allocate gradient
    *gradient = malloc(state_dim * sizeof(ComplexFloat));
    if (!*gradient) {
        free(forward_state);
        free(backward_state);
        return false;
    }
    
    // Compute gradient using parameter shift rule:
    // df/dθ = [f(θ + r) - f(θ - r)]/(2r)
    // where r is the shift amount
    for (size_t i = 0; i < state_dim; i++) {
        (*gradient)[i].real = (forward_state[i].real - backward_state[i].real) / (2 * shift_amount);
        (*gradient)[i].imag = (forward_state[i].imag - backward_state[i].imag) / (2 * shift_amount);
    }
    
    *dimension = state_dim;
    
    free(forward_state);
    free(backward_state);
    
    return true;
}

// Helper function for Richardson extrapolation
static void richardson_extrapolate(
    ComplexFloat** gradients,
    size_t num_gradients,
    size_t dimension,
    const double* step_sizes,
    ComplexFloat* result) {
    
    // Richardson extrapolation formula:
    // f'(0) ≈ [4f'(h/2) - f'(h)]/3
    // where f'(h) is the finite difference approximation with step size h
    
    // For multiple steps, we use higher order formulas
    double* coefficients = malloc(num_gradients * sizeof(double));
    if (!coefficients) return;
    
    // Compute Richardson coefficients
    for (size_t i = 0; i < num_gradients; i++) {
        double power = 1.0;
        for (size_t j = 0; j < i; j++) {
            power *= 4.0;  // Each level multiplies by 4
        }
        coefficients[i] = power / (power - 1.0);
    }
    
    // Apply Richardson extrapolation
    for (size_t d = 0; d < dimension; d++) {
        result[d].real = 0;
        result[d].imag = 0;
        
        for (size_t i = 0; i < num_gradients; i++) {
            result[d].real += coefficients[i] * gradients[i][d].real;
            result[d].imag += coefficients[i] * gradients[i][d].imag;
        }
    }
    
    free(coefficients);
}

// Compute higher order gradient using multiple shifts
bool compute_higher_order_gradient(
    const quantum_geometric_tensor_network_t* qgtn,
    size_t param_idx,
    const double* shift_amounts,
    size_t num_shifts,
    ComplexFloat** gradient,
    size_t* dimension) {
    
    if (!qgtn || !gradient || !dimension || !shift_amounts || num_shifts < 2) {
        return false;
    }
    
    // Allocate array to store gradients at different step sizes
    ComplexFloat** gradients = malloc(num_shifts * sizeof(ComplexFloat*));
    size_t* dimensions = malloc(num_shifts * sizeof(size_t));
    if (!gradients || !dimensions) {
        free(gradients);
        free(dimensions);
        return false;
    }
    
    // Compute gradients at each step size
    for (size_t i = 0; i < num_shifts; i++) {
        if (!compute_parameter_shift_gradient(qgtn, param_idx, shift_amounts[i],
                                           &gradients[i], &dimensions[i])) {
            // Clean up on failure
            for (size_t j = 0; j < i; j++) {
                free(gradients[j]);
            }
            free(gradients);
            free(dimensions);
            return false;
        }
        
        // Verify dimensions match
        if (i > 0 && dimensions[i] != dimensions[0]) {
            // Clean up on dimension mismatch
            for (size_t j = 0; j <= i; j++) {
                free(gradients[j]);
            }
            free(gradients);
            free(dimensions);
            return false;
        }
    }
    
    // Allocate result gradient
    *gradient = malloc(dimensions[0] * sizeof(ComplexFloat));
    if (!*gradient) {
        for (size_t i = 0; i < num_shifts; i++) {
            free(gradients[i]);
        }
        free(gradients);
        free(dimensions);
        return false;
    }
    
    // Perform Richardson extrapolation
    richardson_extrapolate(gradients, num_shifts, dimensions[0], shift_amounts, *gradient);
    
    // Clean up
    for (size_t i = 0; i < num_shifts; i++) {
        free(gradients[i]);
    }
    free(gradients);
    *dimension = dimensions[0];
    free(dimensions);
    
    return true;
}

// Compute centered finite difference gradient
bool compute_centered_difference_gradient(
    const quantum_geometric_tensor_network_t* qgtn,
    size_t param_idx,
    double step_size,
    ComplexFloat** gradient,
    size_t* dimension) {
    
    // Centered difference is just parameter shift with step_size
    return compute_parameter_shift_gradient(qgtn, param_idx, step_size,
                                         gradient, dimension);
}

// Helper function to estimate gradient error
static double estimate_gradient_error(
    const ComplexFloat* gradient1,
    const ComplexFloat* gradient2,
    size_t dimension) {
    
    double error = 0.0;
    for (size_t i = 0; i < dimension; i++) {
        double real_diff = gradient1[i].real - gradient2[i].real;
        double imag_diff = gradient1[i].imag - gradient2[i].imag;
        error += real_diff * real_diff + imag_diff * imag_diff;
    }
    return sqrt(error / dimension);
}

// Compute gradient with error estimation
bool compute_gradient_with_error(
    const quantum_geometric_tensor_network_t* qgtn,
    size_t param_idx,
    ComplexFloat** gradient,
    double* error_estimate,
    size_t* dimension) {
    
    if (!qgtn || !gradient || !error_estimate || !dimension) {
        return false;
    }
    
    // Compute gradients at two different step sizes
    ComplexFloat* gradient1;
    ComplexFloat* gradient2;
    size_t dim1, dim2;
    
    if (!compute_parameter_shift_gradient(qgtn, param_idx, M_PI_2,
                                       &gradient1, &dim1)) {
        return false;
    }
    
    if (!compute_parameter_shift_gradient(qgtn, param_idx, M_PI_4,
                                       &gradient2, &dim2)) {
        free(gradient1);
        return false;
    }
    
    if (dim1 != dim2) {
        free(gradient1);
        free(gradient2);
        return false;
    }
    
    // Use the more accurate (smaller step size) gradient
    *gradient = gradient2;
    *dimension = dim2;
    
    // Estimate error from difference between gradients
    *error_estimate = estimate_gradient_error(gradient1, gradient2, dim1);
    
    free(gradient1);
    return true;
}
