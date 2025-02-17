#include "quantum_geometric/core/quantum_geometric_gradient.h"
#include "quantum_geometric/core/quantum_parameter_shift.h"
#include "quantum_geometric/core/advanced_memory_system.h"
#include "quantum_geometric/core/numerical_backend.h"
#include "quantum_geometric/core/error_handling.h"
#include "quantum_geometric/core/complex_arithmetic.h"
#include "quantum_geometric/core/quantum_complex.h"
#include "quantum_geometric/core/matrix_operations.h"
#include "quantum_geometric/core/quantum_gate_operations.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Global gradient options
static gradient_options_t g_gradient_options = {
    .shift_amount = M_PI_2,
    .use_centered_difference = true,
    .use_higher_order = false,
    .num_points = 2,
    .accumulate_gradients = false,
    .custom_options = NULL
};

// Helper function to compute parameter shift
static void compute_parameter_shift(
    const ComplexFloat* state,
    ComplexFloat* gradient,
    size_t dimension) {
    
    printf("DEBUG: Starting parameter shift computation for dimension %zu\n", dimension);
    
    // Print input state values
    printf("DEBUG: Input state values:\n");
    for (size_t i = 0; i < dimension && i < 3; i++) {
        printf("DEBUG: state[%zu]: (%.6f, %.6f)\n", 
               i, state[i].real, state[i].imag);
    }
    
    // Parameter shift rule for quantum gradients:
    // df/dθ = [f(θ + π/2) - f(θ - π/2)]/2
    
    // Compute forward and backward shifted states
    ComplexFloat* forward_state = NULL;
    ComplexFloat* backward_state = NULL;
    
    printf("DEBUG: Allocating forward and backward states\n");
    
    forward_state = malloc(dimension * sizeof(ComplexFloat));
    if (!forward_state) {
        return;
    }
    
    backward_state = malloc(dimension * sizeof(ComplexFloat));
    if (!backward_state) {
        free(forward_state);
        return;
    }
    
    printf("DEBUG: Computing forward shift\n");
    // Create RY rotation gates for forward and backward shifts
    quantum_gate_t ry_forward = {
        .type = GATE_TYPE_RY,
        .num_qubits = 1,
        .target_qubits = malloc(sizeof(size_t)),
        .control_qubits = NULL,
        .num_controls = 0,
        .is_controlled = false,
        .parameters = malloc(sizeof(double)),
        .num_parameters = 1,
        .is_parameterized = true,
        .matrix = malloc(4 * sizeof(ComplexFloat))
    };
    
    quantum_gate_t ry_backward = {
        .type = GATE_TYPE_RY,
        .num_qubits = 1,
        .target_qubits = malloc(sizeof(size_t)),
        .control_qubits = NULL,
        .num_controls = 0,
        .is_controlled = false,
        .parameters = malloc(sizeof(double)),
        .num_parameters = 1,
        .is_parameterized = true,
        .matrix = malloc(4 * sizeof(ComplexFloat))
    };
    
    if (!ry_forward.target_qubits || !ry_forward.parameters || !ry_forward.matrix ||
        !ry_backward.target_qubits || !ry_backward.parameters || !ry_backward.matrix) {
        free(ry_forward.target_qubits);
        free(ry_forward.parameters);
        free(ry_forward.matrix);
        free(ry_backward.target_qubits);
        free(ry_backward.parameters);
        free(ry_backward.matrix);
        free(forward_state);
        free(backward_state);
        return;
    }
    
    // Copy initial state
    memcpy(forward_state, state, dimension * sizeof(ComplexFloat));
    memcpy(backward_state, state, dimension * sizeof(ComplexFloat));
    
    // Create quantum circuit for forward and backward states
    quantum_circuit_t* forward_circuit = malloc(sizeof(quantum_circuit_t));
    quantum_circuit_t* backward_circuit = malloc(sizeof(quantum_circuit_t));
    if (!forward_circuit || !backward_circuit) {
        free(forward_circuit);
        free(backward_circuit);
        return;
    }

    // Initialize circuits
    forward_circuit->num_qubits = (size_t)log2(dimension);
    backward_circuit->num_qubits = forward_circuit->num_qubits;
    forward_circuit->num_layers = 1;
    backward_circuit->num_layers = 1;
    forward_circuit->is_parameterized = true;
    backward_circuit->is_parameterized = true;

    // Create layers
    forward_circuit->layers = malloc(sizeof(circuit_layer_t*));
    backward_circuit->layers = malloc(sizeof(circuit_layer_t*));
    if (!forward_circuit->layers || !backward_circuit->layers) {
        free(forward_circuit->layers);
        free(backward_circuit->layers);
        free(forward_circuit);
        free(backward_circuit);
        return;
    }

    forward_circuit->layers[0] = malloc(sizeof(circuit_layer_t));
    backward_circuit->layers[0] = malloc(sizeof(circuit_layer_t));
    if (!forward_circuit->layers[0] || !backward_circuit->layers[0]) {
        free(forward_circuit->layers[0]);
        free(backward_circuit->layers[0]);
        free(forward_circuit->layers);
        free(backward_circuit->layers);
        free(forward_circuit);
        free(backward_circuit);
        return;
    }

    // Initialize layers
    forward_circuit->layers[0]->num_gates = forward_circuit->num_qubits;
    backward_circuit->layers[0]->num_gates = backward_circuit->num_qubits;
    forward_circuit->layers[0]->is_parameterized = true;
    backward_circuit->layers[0]->is_parameterized = true;

    // Create gates arrays
    forward_circuit->layers[0]->gates = malloc(forward_circuit->num_qubits * sizeof(quantum_gate_t*));
    backward_circuit->layers[0]->gates = malloc(backward_circuit->num_qubits * sizeof(quantum_gate_t*));
    if (!forward_circuit->layers[0]->gates || !backward_circuit->layers[0]->gates) {
        free(forward_circuit->layers[0]->gates);
        free(backward_circuit->layers[0]->gates);
        free(forward_circuit->layers[0]);
        free(backward_circuit->layers[0]);
        free(forward_circuit->layers);
        free(backward_circuit->layers);
        free(forward_circuit);
        free(backward_circuit);
        return;
    }

    // Create RY gates for each qubit
    printf("DEBUG: Creating RY gates for %zu qubits\n", forward_circuit->num_qubits);
    for (size_t qubit = 0; qubit < forward_circuit->num_qubits; qubit++) {
        // Forward gate
        quantum_gate_t* forward_gate = malloc(sizeof(quantum_gate_t));
        if (!forward_gate) {
            // Clean up and return
            for (size_t i = 0; i < qubit; i++) {
                free(forward_circuit->layers[0]->gates[i]);
                free(backward_circuit->layers[0]->gates[i]);
            }
            free(forward_circuit->layers[0]->gates);
            free(backward_circuit->layers[0]->gates);
            free(forward_circuit->layers[0]);
            free(backward_circuit->layers[0]);
            free(forward_circuit->layers);
            free(backward_circuit->layers);
            free(forward_circuit);
            free(backward_circuit);
            return;
        }

        forward_gate->type = GATE_TYPE_RY;
        forward_gate->num_qubits = 1;
        forward_gate->target_qubits = malloc(sizeof(size_t));
        forward_gate->target_qubits[0] = qubit;
        forward_gate->control_qubits = NULL;
        forward_gate->num_controls = 0;
        forward_gate->is_controlled = false;
        forward_gate->parameters = malloc(sizeof(double));
        forward_gate->parameters[0] = M_PI_2;  // +π/2
        forward_gate->num_parameters = 1;
        forward_gate->is_parameterized = true;
        forward_gate->matrix = malloc(4 * sizeof(ComplexFloat));

        // Compute forward gate matrix
        double cos_half = cos(forward_gate->parameters[0] / 2.0);
        double sin_half = sin(forward_gate->parameters[0] / 2.0);
        forward_gate->matrix[0] = (ComplexFloat){cos_half, 0};
        forward_gate->matrix[1] = (ComplexFloat){-sin_half, 0};
        forward_gate->matrix[2] = (ComplexFloat){sin_half, 0};
        forward_gate->matrix[3] = (ComplexFloat){cos_half, 0};

        forward_circuit->layers[0]->gates[qubit] = forward_gate;

        // Backward gate
        quantum_gate_t* backward_gate = malloc(sizeof(quantum_gate_t));
        if (!backward_gate) {
            // Clean up and return
            free(forward_gate->target_qubits);
            free(forward_gate->parameters);
            free(forward_gate->matrix);
            free(forward_gate);
            for (size_t i = 0; i < qubit; i++) {
                free(forward_circuit->layers[0]->gates[i]);
                free(backward_circuit->layers[0]->gates[i]);
            }
            free(forward_circuit->layers[0]->gates);
            free(backward_circuit->layers[0]->gates);
            free(forward_circuit->layers[0]);
            free(backward_circuit->layers[0]);
            free(forward_circuit->layers);
            free(backward_circuit->layers);
            free(forward_circuit);
            free(backward_circuit);
            return;
        }

        backward_gate->type = GATE_TYPE_RY;
        backward_gate->num_qubits = 1;
        backward_gate->target_qubits = malloc(sizeof(size_t));
        backward_gate->target_qubits[0] = qubit;
        backward_gate->control_qubits = NULL;
        backward_gate->num_controls = 0;
        backward_gate->is_controlled = false;
        backward_gate->parameters = malloc(sizeof(double));
        backward_gate->parameters[0] = -M_PI_2;  // -π/2
        backward_gate->num_parameters = 1;
        backward_gate->is_parameterized = true;
        backward_gate->matrix = malloc(4 * sizeof(ComplexFloat));

        // Compute backward gate matrix
        cos_half = cos(backward_gate->parameters[0] / 2.0);
        sin_half = sin(backward_gate->parameters[0] / 2.0);
        backward_gate->matrix[0] = (ComplexFloat){cos_half, 0};
        backward_gate->matrix[1] = (ComplexFloat){-sin_half, 0};
        backward_gate->matrix[2] = (ComplexFloat){sin_half, 0};
        backward_gate->matrix[3] = (ComplexFloat){cos_half, 0};

        backward_circuit->layers[0]->gates[qubit] = backward_gate;
    }

    // Create quantum geometric tensor networks for forward and backward states
    quantum_geometric_tensor_network_t* forward_qgtn = create_quantum_geometric_tensor_network(
        forward_circuit->num_qubits, 1, false, false);
    quantum_geometric_tensor_network_t* backward_qgtn = create_quantum_geometric_tensor_network(
        backward_circuit->num_qubits, 1, false, false);

    if (!forward_qgtn || !backward_qgtn) {
        // Clean up and return
        for (size_t i = 0; i < forward_circuit->num_qubits; i++) {
            free(forward_circuit->layers[0]->gates[i]->target_qubits);
            free(forward_circuit->layers[0]->gates[i]->parameters);
            free(forward_circuit->layers[0]->gates[i]->matrix);
            free(forward_circuit->layers[0]->gates[i]);
            free(backward_circuit->layers[0]->gates[i]->target_qubits);
            free(backward_circuit->layers[0]->gates[i]->parameters);
            free(backward_circuit->layers[0]->gates[i]->matrix);
            free(backward_circuit->layers[0]->gates[i]);
        }
        free(forward_circuit->layers[0]->gates);
        free(backward_circuit->layers[0]->gates);
        free(forward_circuit->layers[0]);
        free(backward_circuit->layers[0]);
        free(forward_circuit->layers);
        free(backward_circuit->layers);
        free(forward_circuit);
        free(backward_circuit);
        destroy_quantum_geometric_tensor_network(forward_qgtn);
        destroy_quantum_geometric_tensor_network(backward_qgtn);
        return;
    }

    // Initialize states
    memcpy(forward_state, state, dimension * sizeof(ComplexFloat));
    memcpy(backward_state, state, dimension * sizeof(ComplexFloat));

    // Apply circuits
    if (!apply_quantum_circuit(forward_qgtn, forward_circuit) ||
        !apply_quantum_circuit(backward_qgtn, backward_circuit)) {
        printf("DEBUG: Failed to apply quantum circuits\n");
    }

    // Copy final states
    if (forward_qgtn->circuit->state && forward_qgtn->circuit->state->coordinates) {
        memcpy(forward_state, forward_qgtn->circuit->state->coordinates, dimension * sizeof(ComplexFloat));
    }
    if (backward_qgtn->circuit->state && backward_qgtn->circuit->state->coordinates) {
        memcpy(backward_state, backward_qgtn->circuit->state->coordinates, dimension * sizeof(ComplexFloat));
    }

    // Clean up
    for (size_t i = 0; i < forward_circuit->num_qubits; i++) {
        free(forward_circuit->layers[0]->gates[i]->target_qubits);
        free(forward_circuit->layers[0]->gates[i]->parameters);
        free(forward_circuit->layers[0]->gates[i]->matrix);
        free(forward_circuit->layers[0]->gates[i]);
        free(backward_circuit->layers[0]->gates[i]->target_qubits);
        free(backward_circuit->layers[0]->gates[i]->parameters);
        free(backward_circuit->layers[0]->gates[i]->matrix);
        free(backward_circuit->layers[0]->gates[i]);
    }
    free(forward_circuit->layers[0]->gates);
    free(backward_circuit->layers[0]->gates);
    free(forward_circuit->layers[0]);
    free(backward_circuit->layers[0]);
    free(forward_circuit->layers);
    free(backward_circuit->layers);
    free(forward_circuit);
    free(backward_circuit);
    destroy_quantum_geometric_tensor_network(forward_qgtn);
    destroy_quantum_geometric_tensor_network(backward_qgtn);
    
    // Print transformed states
    printf("DEBUG: Forward state after RY rotations:\n");
    for (size_t i = 0; i < dimension && i < 4; i++) {
        printf("  |%zu>: (%.6f,%.6f)\n", i, forward_state[i].real, forward_state[i].imag);
    }
    
    printf("DEBUG: Backward state after RY rotations:\n");
    for (size_t i = 0; i < dimension && i < 4; i++) {
        printf("  |%zu>: (%.6f,%.6f)\n", i, backward_state[i].real, backward_state[i].imag);
    }
    
    // Clean up gates
    free(ry_forward.target_qubits);
    free(ry_forward.parameters);
    free(ry_forward.matrix);
    free(ry_backward.target_qubits);
    free(ry_backward.parameters);
    free(ry_backward.matrix);
    
    printf("DEBUG: Computing raw gradients\n");
    // Compute gradient using parameter shift rule with complex L2 normalization
    float sum_squares = 0.0f;
    
    // Print first few raw gradients
    for (size_t i = 0; i < dimension && i < 3; i++) {
        float real_grad = (forward_state[i].real - backward_state[i].real) / 2.0f;
        float imag_grad = (forward_state[i].imag - backward_state[i].imag) / 2.0f;
        printf("DEBUG: Raw gradient[%zu]: (%.6f, %.6f)\n", i, real_grad, imag_grad);
    }
    
    // First pass: compute raw gradients and sum of squares of complex magnitudes
    for (size_t i = 0; i < dimension; i++) {
        float real_grad = (forward_state[i].real - backward_state[i].real) / 2.0f;
        float imag_grad = (forward_state[i].imag - backward_state[i].imag) / 2.0f;
        
        // Compute squared magnitude of complex gradient
        sum_squares += real_grad * real_grad + imag_grad * imag_grad;
        
        gradient[i].real = real_grad;
        gradient[i].imag = imag_grad;
    }
    
    // Compute L2 norm of complex vector
    float norm = sqrtf(sum_squares);
    
    // Prevent division by zero with a minimum norm
    const float min_norm = 1e-6f;
    norm = fmaxf(norm, min_norm);
    
    printf("DEBUG: Normalizing gradients with norm=%.6f\n", norm);
    // Scale gradients to have unit norm while preserving complex ratios
    const float scale = 1.0f / norm;
    for (size_t i = 0; i < dimension; i++) {
        gradient[i].real *= scale;
        gradient[i].imag *= scale;
        
        if (i < 3) {
            printf("DEBUG: Normalized gradient[%zu]: (%.6f, %.6f)\n", 
                   i, gradient[i].real, gradient[i].imag);
        }
    }
    
    if (forward_state) free(forward_state);
    if (backward_state) free(backward_state);
}

// Helper function to compute natural gradient
static void compute_natural_gradient(
    const ComplexFloat* gradient,
    const ComplexFloat* metric,
    ComplexFloat* natural_gradient,
    size_t dimension) {
    
    // Natural gradient is computed by:
    // g_nat = G^{-1} g where G is the metric tensor
    
    // First compute metric tensor inverse
    ComplexFloat* metric_inverse = malloc(dimension * dimension * sizeof(ComplexFloat));
    if (!metric_inverse) {
        return;
    }
    
    // Use matrix_inverse from matrix_operations.h
    if (!matrix_inverse(metric, metric_inverse, dimension)) {
        free(metric_inverse);
        return;
    }
    
    // Multiply inverse metric with gradient
    for (size_t i = 0; i < dimension; i++) {
        natural_gradient[i] = (ComplexFloat){0.0f, 0.0f};
        for (size_t j = 0; j < dimension; j++) {
            ComplexFloat prod = complex_multiply(metric_inverse[i * dimension + j], gradient[j]);
            natural_gradient[i] = complex_add(natural_gradient[i], prod);
        }
    }
    
    free(metric_inverse);
}

// Compute gradient of quantum state with respect to parameter
bool compute_quantum_gradient(
    const quantum_geometric_tensor_network_t* qgtn,
    size_t param_idx,
    ComplexFloat** gradient,
    size_t* dimension) {
    
    printf("DEBUG: Starting compute_quantum_gradient\n");
    printf("DEBUG: param_idx=%zu\n", param_idx);
    
    if (!qgtn || !gradient || !dimension) {
        printf("DEBUG: Invalid arguments to compute_quantum_gradient\n");
        return false;
    }
    
    // For gradient computation, we can work directly with the tensor network data
    if (!qgtn || !qgtn->network || !qgtn->network->nodes || 
        qgtn->network->num_nodes < 1) {
        printf("DEBUG: Invalid tensor network state\n");
        return false;
    }
    
    tensor_node_t* node = qgtn->network->nodes[0];
    if (!node || !node->data) {
        printf("DEBUG: Invalid tensor node or data\n");
        return false;
    }
    printf("DEBUG: Tensor network state valid\n");
    
    // Initialize memory system
    printf("DEBUG: Initializing memory system\n");
    memory_system_config_t mem_config = {
        .type = MEM_SYSTEM_QUANTUM,
        .strategy = ALLOC_STRATEGY_BUDDY,
        .optimization = MEM_OPT_ADVANCED,
        .alignment = sizeof(ComplexFloat),
        .enable_monitoring = true,
        .enable_defragmentation = true
    };
    
    advanced_memory_system_t* memory = create_memory_system(&mem_config);
    if (!memory) {
        printf("DEBUG: Failed to create memory system\n");
        return false;
    }
    printf("DEBUG: Memory system initialized\n");
    
    // Get state dimension from tensor network
    size_t state_dim = 1 << qgtn->num_qubits; // 2^n for n qubits
    printf("DEBUG: State dimension=%zu (num_qubits=%zu)\n", state_dim, qgtn->num_qubits);
    
    // Allocate state vector
    printf("DEBUG: Allocating state vector\n");
    ComplexFloat* state = memory_allocate(memory, 
                                        state_dim * sizeof(ComplexFloat),
                                        sizeof(ComplexFloat));
    if (!state) {
        printf("DEBUG: Failed to allocate state vector\n");
        destroy_memory_system(memory);
        return false;
    }
    printf("DEBUG: State vector allocated\n");
    
    // Initialize quantum state
    if (!node->data) {
        printf("DEBUG: Node data is NULL\n");
        memory_free(memory, state);
        destroy_memory_system(memory);
        return false;
    }
    
    // Initialize state to |0>
    printf("DEBUG: Initializing state to |0>\n");
    memset(state, 0, state_dim * sizeof(ComplexFloat));
    state[0].real = 1.0f;
    state[0].imag = 0.0f;
    
    // Copy initial state
    printf("DEBUG: Copying initial state\n");
    memcpy(state, node->data, state_dim * sizeof(ComplexFloat));
    printf("DEBUG: Initial state copied from node data\n");
    printf("DEBUG: First few state values:\n");
    for (size_t i = 0; i < state_dim && i < 4; i++) {
        printf("  |%zu>: (%.6f,%.6f)\n", i, state[i].real, state[i].imag);
    }
    
    // Initialize quantum state in circuit
    printf("DEBUG: Initializing circuit state\n");
    if (!qgtn->circuit->state) {
        printf("DEBUG: Creating new circuit state\n");
        qgtn->circuit->state = malloc(sizeof(quantum_geometric_state_t));
        if (!qgtn->circuit->state) {
            printf("DEBUG: Failed to allocate circuit state\n");
            memory_free(memory, state);
            destroy_memory_system(memory);
            return false;
        }
    }
    printf("DEBUG: Circuit state initialized\n");
    
    // Set state coordinates
    printf("DEBUG: Setting circuit state coordinates\n");
    if (qgtn->circuit->state->coordinates) {
        printf("DEBUG: Freeing existing coordinates\n");
        free(qgtn->circuit->state->coordinates);
    }
    qgtn->circuit->state->coordinates = malloc(state_dim * sizeof(ComplexFloat));
    if (!qgtn->circuit->state->coordinates) {
        printf("DEBUG: Failed to allocate coordinates\n");
        memory_free(memory, state);
        destroy_memory_system(memory);
        return false;
    }
    memcpy(qgtn->circuit->state->coordinates, state, state_dim * sizeof(ComplexFloat));
    qgtn->circuit->state->dimension = state_dim;
    printf("DEBUG: Circuit state coordinates set\n");
    
    // Allocate gradient array
    printf("DEBUG: Allocating gradient array\n");
    *gradient = memory_allocate(memory,
                              state_dim * sizeof(ComplexFloat),
                              sizeof(ComplexFloat));
    if (!*gradient) {
        printf("DEBUG: Failed to allocate gradient array\n");
        memory_free(memory, state);
        destroy_memory_system(memory);
        return false;
    }
    printf("DEBUG: Gradient array allocated\n");
    
    // Initialize gradient array to zero
    memset(*gradient, 0, state_dim * sizeof(ComplexFloat));
    printf("DEBUG: Gradient array initialized to zero\n");
    
    printf("DEBUG: Computing gradient for state dimension %zu\n", state_dim);
    
    // Compute gradient using parameter shift rule
    printf("DEBUG: Calling compute_parameter_shift\n");
    compute_parameter_shift(state, *gradient, state_dim);
    printf("DEBUG: Parameter shift computation completed\n");
    
    printf("DEBUG: Checking gradient validity\n");
    
    // Verify gradient computation succeeded
    bool has_valid_gradient = false;
    printf("DEBUG: Gradient components:\n");
    for (size_t i = 0; i < state_dim && i < 4; i++) {
        printf("  [%zu]: (%.6f,%.6f)\n", i, (*gradient)[i].real, (*gradient)[i].imag);
        if ((*gradient)[i].real != 0.0f || (*gradient)[i].imag != 0.0f) {
            has_valid_gradient = true;
        }
    }
    
    if (!has_valid_gradient) {
        printf("DEBUG: No valid gradient components found\n");
        memory_free(memory, *gradient);
        memory_free(memory, state);
        destroy_memory_system(memory);
        return false;
    }
    printf("DEBUG: Valid gradient components found\n");
    
    printf("DEBUG: Valid gradient computed\n");
    
    *dimension = state_dim;
    
    // Clean up
    memory_free(memory, state);
    destroy_memory_system(memory);
    
    return true;
}

// Compute natural gradient using quantum geometric tensor
bool compute_quantum_natural_gradient(
    const quantum_geometric_tensor_network_t* qgtn,
    size_t param_idx,
    ComplexFloat** natural_gradient,
    size_t* dimension) {
    
    if (!qgtn || !natural_gradient || !dimension) {
        return false;
    }
    
    // Get regular gradient first
    ComplexFloat* gradient;
    size_t grad_dim;
    if (!compute_quantum_gradient(qgtn, param_idx, &gradient, &grad_dim)) {
        return false;
    }
    
    // Get quantum geometric tensor to use as metric
    ComplexFloat* metric = malloc(grad_dim * grad_dim * sizeof(ComplexFloat));
    if (!metric) {
        free(gradient);
        return false;
    }
    
    // Compute metric tensor components
    for (size_t i = 0; i < grad_dim; i++) {
        for (size_t j = 0; j < grad_dim; j++) {
            if (!compute_quantum_geometric_tensor(qgtn, i, j, 
                &metric[i * grad_dim + j])) {
                free(gradient);
                free(metric);
                return false;
            }
        }
    }
    
    // Allocate natural gradient array
    *natural_gradient = malloc(grad_dim * sizeof(ComplexFloat));
    if (!*natural_gradient) {
        free(gradient);
        free(metric);
        return false;
    }
    
    // Compute natural gradient
    compute_natural_gradient(gradient, metric, *natural_gradient, grad_dim);
    
    *dimension = grad_dim;
    
    free(gradient);
    free(metric);
    
    return true;
}

// Compute gradient of expectation value
bool compute_expectation_gradient(
    const quantum_geometric_tensor_network_t* qgtn,
    const quantum_geometric_operator_t* op,
    size_t param_idx,
    double* gradient) {
    
    if (!qgtn || !op || !gradient) {
        return false;
    }
    
    // Get quantum state gradient
    ComplexFloat* state_gradient;
    size_t grad_dim;
    if (!compute_quantum_gradient(qgtn, param_idx, &state_gradient, &grad_dim)) {
        return false;
    }
    
    // Get current quantum state
    ComplexFloat* state;
    size_t state_dim;
    if (!get_quantum_state(qgtn, &state, &state_dim)) {
        free(state_gradient);
        return false;
    }
    
    if (grad_dim != state_dim) {
        free(state_gradient);
        free(state);
        return false;
    }
    
    // Compute gradient of expectation value:
    // d<O>/dθ = <dψ/dθ|O|ψ> + <ψ|O|dψ/dθ>
    
    // First compute O|ψ>
    ComplexFloat* op_state = malloc(state_dim * sizeof(ComplexFloat));
    if (!op_state) {
        free(state_gradient);
        free(state);
        return false;
    }
    
    // Create temporary qubit array
    size_t* qubits = malloc(state_dim * sizeof(size_t));
    if (!qubits) {
        free(op_state);
        free(state_gradient);
        free(state);
        return false;
    }
    for (size_t i = 0; i < state_dim; i++) {
        qubits[i] = i;
    }
    
    // Create temporary quantum gate from operator
    quantum_gate_t gate;
    gate.num_qubits = state_dim;
    gate.matrix = (ComplexFloat*)op;
    gate.target_qubits = qubits;  // Use all qubits as targets
    gate.control_qubits = NULL;
    gate.num_controls = 0;
    gate.is_controlled = false;
    gate.type = GATE_TYPE_CUSTOM;
    gate.parameters = NULL;
    gate.num_parameters = 0;
    gate.is_parameterized = false;
    
    // Apply operator
    quantum_geometric_tensor_network_t* qgtn_copy = (quantum_geometric_tensor_network_t*)qgtn;
    if (!apply_quantum_gate(qgtn_copy, &gate, qubits, state_dim)) {
        free(qubits);
        free(op_state);
        free(state_gradient);
        free(state);
        return false;
    }
    
    // Compute <dψ/dθ|O|ψ>
    ComplexFloat term1 = {0.0f, 0.0f};
    for (size_t i = 0; i < state_dim; i++) {
        ComplexFloat conj_grad = complex_conjugate(state_gradient[i]);
        ComplexFloat prod = complex_multiply(conj_grad, op_state[i]);
        term1 = complex_add(term1, prod);
    }
    
    // Compute <ψ|O|dψ/dθ>
    ComplexFloat term2 = {0.0f, 0.0f};
    ComplexFloat* op_grad = malloc(state_dim * sizeof(ComplexFloat));
    if (!op_grad) {
        free(qubits);
        free(op_state);
        free(state_gradient);
        free(state);
        return false;
    }
    
    // Apply operator to gradient state
    if (!apply_quantum_gate(qgtn_copy, &gate, qubits, state_dim)) {
        free(qubits);
        free(op_grad);
        free(op_state);
        free(state_gradient);
        free(state);
        return false;
    }
    
    free(qubits);
    
    for (size_t i = 0; i < state_dim; i++) {
        ComplexFloat conj_state = complex_conjugate(state[i]);
        ComplexFloat prod = complex_multiply(conj_state, op_grad[i]);
        term2 = complex_add(term2, prod);
    }
    
    // Sum the terms and take real part for final gradient
    *gradient = term1.real + term2.real;
    
    free(op_state);
    free(op_grad);
    free(state_gradient);
    free(state);
    
    return true;
}

// Set gradient computation options
bool set_gradient_options(const gradient_options_t* options) {
    if (!options) {
        return false;
    }
    
    // Validate options
    if (options->shift_amount <= 0.0 ||
        options->num_points < 2) {
        return false;
    }
    
    // Copy options
    memcpy(&g_gradient_options, options, sizeof(gradient_options_t));
    return true;
}

// Get current gradient computation options
bool get_gradient_options(gradient_options_t* options) {
    if (!options) {
        return false;
    }
    
    // Copy options
    memcpy(options, &g_gradient_options, sizeof(gradient_options_t));
    return true;
}

bool compute_higher_order_gradient(
    const quantum_geometric_tensor_network_t* qgtn,
    size_t param_idx,
    const double* shift_amounts,
    size_t num_shifts,
    ComplexFloat** gradient,
    size_t* dimension) {
    
    if (!qgtn || !shift_amounts || !gradient || !dimension || num_shifts < 2) {
        return false;
    }
    
    // Get state dimension
    size_t state_dim = 1 << qgtn->num_qubits;
    
    // Allocate arrays for shifted states
    ComplexFloat** shifted_states = malloc(num_shifts * sizeof(ComplexFloat*));
    if (!shifted_states) {
        return false;
    }
    
    // Initialize shifted states
    for (size_t i = 0; i < num_shifts; i++) {
        shifted_states[i] = calloc(state_dim, sizeof(ComplexFloat));
        if (!shifted_states[i]) {
            for (size_t j = 0; j < i; j++) {
                free(shifted_states[j]);
            }
            free(shifted_states);
            return false;
        }
    }
    
    // Save original state
    ComplexFloat* original_state = NULL;
    if (qgtn->circuit->state && qgtn->circuit->state->coordinates) {
        original_state = malloc(state_dim * sizeof(ComplexFloat));
        if (!original_state) {
            for (size_t i = 0; i < num_shifts; i++) {
                free(shifted_states[i]);
            }
            free(shifted_states);
            return false;
        }
        memcpy(original_state, qgtn->circuit->state->coordinates, 
               state_dim * sizeof(ComplexFloat));
    }
    
    printf("DEBUG: Computing higher order gradient with %zu shifts\n", num_shifts);
    printf("DEBUG: Shift amounts: ");
    for (size_t i = 0; i < num_shifts; i++) {
        printf("%.6f ", shift_amounts[i]);
    }
    printf("\n");

    // Compute states at each shift point
    for (size_t i = 0; i < num_shifts; i++) {
        ComplexFloat *forward = NULL, *backward = NULL;
        size_t dim;
        
        printf("DEBUG: Computing shifted states for shift %zu (amount=%.6f)\n", i, shift_amounts[i]);
        
        // Create a copy of qgtn since compute_shifted_states needs to modify it
        quantum_geometric_tensor_network_t* qgtn_copy = copy_quantum_geometric_tensor_network(qgtn);
        if (!qgtn_copy) {
            printf("DEBUG: Failed to create copy of quantum geometric tensor network\n");
            for (size_t j = 0; j < num_shifts; j++) {
                free(shifted_states[j]);
            }
            free(shifted_states);
            free(original_state);
            return false;
        }

        printf("DEBUG: Calling compute_shifted_states for param_idx=%zu\n", param_idx);
        if (!compute_shifted_states(qgtn_copy, param_idx, shift_amounts[i],
                                  &forward, &backward, &dim)) {
            printf("DEBUG: compute_shifted_states failed\n");
            destroy_quantum_geometric_tensor_network(qgtn_copy);
            for (size_t j = 0; j < num_shifts; j++) {
                free(shifted_states[j]);
            }
            free(shifted_states);
            free(original_state);
            return false;
        }
        printf("DEBUG: compute_shifted_states succeeded\n");
        
        printf("DEBUG: Forward state after shift %zu:\n", i);
        for (size_t j = 0; j < dim && j < 4; j++) {
            printf("  |%zu>: (%.6f,%.6f)\n", j, forward[j].real, forward[j].imag);
        }

        // Store forward shifted state
        memcpy(shifted_states[i], forward, state_dim * sizeof(ComplexFloat));
        
        free(forward);
        free(backward);
        destroy_quantum_geometric_tensor_network(qgtn_copy);
        printf("DEBUG: Completed shift %zu\n", i);
    }

    printf("DEBUG: Computing gradient using shifted states\n");
    
    // Allocate gradient array
    *gradient = malloc(state_dim * sizeof(ComplexFloat));
    if (!*gradient) {
        for (size_t i = 0; i < num_shifts; i++) {
            free(shifted_states[i]);
        }
        free(shifted_states);
        free(original_state);
        return false;
    }
    
    // Compute higher order gradient using finite difference
    printf("DEBUG: Using %s difference method\n", 
           num_shifts >= 5 ? "4th order central" : 
           num_shifts >= 3 ? "2nd order central" : "forward");

    // Here using 4th order central difference
    if (num_shifts >= 5) {
        for (size_t i = 0; i < state_dim; i++) {
            (*gradient)[i].real = (shifted_states[0][i].real - 8*shifted_states[1][i].real + 
                                 8*shifted_states[3][i].real - shifted_states[4][i].real) / 12.0;
            (*gradient)[i].imag = (shifted_states[0][i].imag - 8*shifted_states[1][i].imag + 
                                 8*shifted_states[3][i].imag - shifted_states[4][i].imag) / 12.0;
            
            if (i < 4) {
                printf("DEBUG: Gradient[%zu]: (%.6f,%.6f)\n", 
                       i, (*gradient)[i].real, (*gradient)[i].imag);
            }
        }
    }
    // Fall back to 2nd order central difference
    else if (num_shifts >= 3) {
        for (size_t i = 0; i < state_dim; i++) {
            (*gradient)[i].real = (shifted_states[2][i].real - shifted_states[0][i].real) / 2.0;
            (*gradient)[i].imag = (shifted_states[2][i].imag - shifted_states[0][i].imag) / 2.0;
            
            if (i < 4) {
                printf("DEBUG: Gradient[%zu]: (%.6f,%.6f)\n", 
                       i, (*gradient)[i].real, (*gradient)[i].imag);
            }
        }
    }
    // Fall back to forward difference
    else {
        for (size_t i = 0; i < state_dim; i++) {
            (*gradient)[i].real = shifted_states[1][i].real - shifted_states[0][i].real;
            (*gradient)[i].imag = shifted_states[1][i].imag - shifted_states[0][i].imag;
            
            if (i < 4) {
                printf("DEBUG: Gradient[%zu]: (%.6f,%.6f)\n", 
                       i, (*gradient)[i].real, (*gradient)[i].imag);
            }
        }
    }

    printf("DEBUG: Gradient computation completed\n");
    
    *dimension = state_dim;
    
    // Clean up
    for (size_t i = 0; i < num_shifts; i++) {
        free(shifted_states[i]);
    }
    free(shifted_states);
    
    // Restore original state if it existed
    if (original_state) {
        if (qgtn->circuit->state->coordinates) {
            free(qgtn->circuit->state->coordinates);
        }
        qgtn->circuit->state->coordinates = original_state;
    }
    
    return true;
}
