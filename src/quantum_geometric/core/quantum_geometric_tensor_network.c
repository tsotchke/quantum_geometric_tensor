#include "quantum_geometric/core/quantum_geometric_tensor_network.h"
#include "quantum_geometric/core/quantum_geometric_gradient.h"
#include "quantum_geometric/core/tensor_network_operations.h"
#include "quantum_geometric/core/numerical_backend.h"
#include "quantum_geometric/core/error_handling.h"
#include "quantum_geometric/hardware/quantum_hardware_types.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Error handling
static const char* g_last_error = NULL;

static void set_error(const char* error) {
    g_last_error = error;
}

const char* get_quantum_geometric_tensor_network_error(void) {
    return g_last_error ? g_last_error : "No error";
}

// Creation and destruction
quantum_geometric_tensor_network_t* create_quantum_geometric_tensor_network(
    size_t num_qubits,
    size_t num_layers,
    bool is_distributed,
    bool use_hardware_acceleration) {
    
    quantum_geometric_tensor_network_t* qgtn = malloc(sizeof(quantum_geometric_tensor_network_t));
    if (!qgtn) {
        set_error("Failed to allocate quantum geometric tensor network");
        return NULL;
    }
    
    qgtn->network = create_tensor_network();
    if (!qgtn->network) {
        free(qgtn);
        set_error("Failed to create tensor network");
        return NULL;
    }

    // Initialize with |0> state for each qubit
    ComplexFloat zero_state[2] = {{1.0f, 0.0f}, {0.0f, 0.0f}};
    size_t dim = 2;
    for (size_t i = 0; i < num_qubits; i++) {
        size_t node_id;
        if (!add_tensor_node(qgtn->network, zero_state, &dim, 1, &node_id)) {
            destroy_tensor_network(qgtn->network);
            free(qgtn);
            set_error("Failed to initialize qubit state");
            return NULL;
        }
    }

    // Initialize empty quantum circuit
    qgtn->circuit = malloc(sizeof(quantum_circuit_t));
    if (!qgtn->circuit) {
        destroy_tensor_network(qgtn->network);
        free(qgtn);
        set_error("Failed to allocate quantum circuit");
        return NULL;
    }

    qgtn->circuit->layers = malloc(num_layers * sizeof(circuit_layer_t*));
    if (!qgtn->circuit->layers) {
        free(qgtn->circuit);
        destroy_tensor_network(qgtn->network);
        free(qgtn);
        set_error("Failed to allocate circuit layers");
        return NULL;
    }

    for (size_t i = 0; i < num_layers; i++) {
        qgtn->circuit->layers[i] = NULL;
    }

    qgtn->circuit->num_layers = num_layers;
    qgtn->circuit->num_qubits = num_qubits;
    qgtn->circuit->is_parameterized = false;
    
    qgtn->num_qubits = num_qubits;
    qgtn->num_layers = num_layers;
    qgtn->is_distributed = is_distributed;
    qgtn->use_hardware_acceleration = use_hardware_acceleration;
    
    return qgtn;
}

void destroy_quantum_geometric_tensor_network(quantum_geometric_tensor_network_t* qgtn) {
    if (!qgtn) return;
    
    if (qgtn->network) {
        destroy_tensor_network(qgtn->network);
    }

    if (qgtn->circuit) {
        // Free each layer
        for (size_t l = 0; l < qgtn->circuit->num_layers; l++) {
            circuit_layer_t* layer = qgtn->circuit->layers[l];
            if (layer) {
                // Free each gate in the layer
                for (size_t g = 0; g < layer->num_gates; g++) {
                    quantum_gate_t* gate = layer->gates[g];
                    if (gate) {
                        free(gate->matrix);
                        free(gate->target_qubits);
                        free(gate->control_qubits);
                        free(gate->parameters);
                        free(gate);
                    }
                }
                free(layer->gates);
                free(layer);
            }
        }
        free(qgtn->circuit->layers);
        free(qgtn->circuit);
    }
    
    free(qgtn);
}

// Quantum operations
// Helper function to create a new circuit layer
static circuit_layer_t* create_circuit_layer(void) {
    circuit_layer_t* layer = malloc(sizeof(circuit_layer_t));
    if (!layer) return NULL;
    
    layer->gates = malloc(16 * sizeof(quantum_gate_t*));  // Initial capacity of 16 gates
    if (!layer->gates) {
        free(layer);
        return NULL;
    }
    
    layer->num_gates = 0;
    layer->is_parameterized = false;
    return layer;
}

// Helper function to copy a quantum gate
static quantum_gate_t* copy_quantum_gate(const quantum_gate_t* gate) {
    quantum_gate_t* new_gate = malloc(sizeof(quantum_gate_t));
    if (!new_gate) return NULL;
    
    // Copy basic fields
    new_gate->num_qubits = gate->num_qubits;
    new_gate->num_controls = gate->num_controls;
    new_gate->is_controlled = gate->is_controlled;
    new_gate->type = gate->type;
    new_gate->num_parameters = gate->num_parameters;
    new_gate->is_parameterized = gate->is_parameterized;
    
    // Allocate and copy arrays
    size_t matrix_size = 1 << (2 * gate->num_qubits);  // 2^n x 2^n matrix
    new_gate->matrix = malloc(matrix_size * sizeof(ComplexFloat));
    new_gate->target_qubits = malloc(gate->num_qubits * sizeof(size_t));
    new_gate->control_qubits = gate->num_controls ? malloc(gate->num_controls * sizeof(size_t)) : NULL;
    new_gate->parameters = gate->num_parameters ? malloc(gate->num_parameters * sizeof(double)) : NULL;
    
    if (!new_gate->matrix || !new_gate->target_qubits || 
        (gate->num_controls && !new_gate->control_qubits) ||
        (gate->num_parameters && !new_gate->parameters)) {
        free(new_gate->matrix);
        free(new_gate->target_qubits);
        free(new_gate->control_qubits);
        free(new_gate->parameters);
        free(new_gate);
        return NULL;
    }
    
    // Copy data
    memcpy(new_gate->matrix, gate->matrix, matrix_size * sizeof(ComplexFloat));
    memcpy(new_gate->target_qubits, gate->target_qubits, gate->num_qubits * sizeof(size_t));
    if (gate->num_controls) {
        memcpy(new_gate->control_qubits, gate->control_qubits, gate->num_controls * sizeof(size_t));
    }
    if (gate->num_parameters) {
        memcpy(new_gate->parameters, gate->parameters, gate->num_parameters * sizeof(double));
    }
    
    return new_gate;
}

bool apply_quantum_gate(
    quantum_geometric_tensor_network_t* qgtn,
    const quantum_gate_t* gate,
    const size_t* qubits,
    size_t num_qubits) {
    
    if (!qgtn || !gate || !qubits || num_qubits == 0) {
        set_error("Invalid arguments to apply_quantum_gate");
        return false;
    }
    
    // Find or create layer for this gate
    size_t layer_idx = qgtn->num_layers - 1;  // Add to last layer
    if (!qgtn->circuit->layers[layer_idx]) {
        qgtn->circuit->layers[layer_idx] = create_circuit_layer();
        if (!qgtn->circuit->layers[layer_idx]) {
            set_error("Failed to create circuit layer");
            return false;
        }
    }
    
    // Copy gate to circuit
    quantum_gate_t* circuit_gate = copy_quantum_gate(gate);
    if (!circuit_gate) {
        set_error("Failed to copy quantum gate");
        return false;
    }
    
    // Add gate to layer
    circuit_layer_t* layer = qgtn->circuit->layers[layer_idx];
    layer->gates[layer->num_gates++] = circuit_gate;
    
    // Update layer parameterization status
    if (circuit_gate->is_parameterized) {
        layer->is_parameterized = true;
        qgtn->circuit->is_parameterized = true;
    }
    
    // Create tensor node for gate
    size_t node_id;
    if (!add_tensor_node(qgtn->network, gate->matrix,
                        &gate->num_qubits, 1, &node_id)) {
        set_error("Failed to add gate tensor node");
        return false;
    }
    
    // Connect gate tensor to qubit tensors
    for (size_t i = 0; i < num_qubits; i++) {
        if (!connect_tensor_nodes(qgtn->network, node_id, i,
                                qubits[i], 0)) {
            set_error("Failed to connect gate tensor to qubit");
            return false;
        }
    }
    
    return true;
}

bool apply_quantum_circuit(
    quantum_geometric_tensor_network_t* qgtn,
    const quantum_circuit_t* circuit) {
    
    if (!qgtn || !circuit) {
        set_error("Invalid arguments to apply_quantum_circuit");
        return false;
    }
    
    // Apply each layer
    for (size_t l = 0; l < circuit->num_layers; l++) {
        circuit_layer_t* layer = circuit->layers[l];
        
        // Apply gates in layer
        for (size_t g = 0; g < layer->num_gates; g++) {
            quantum_gate_t* gate = layer->gates[g];
            if (!apply_quantum_gate(qgtn, gate,
                                  gate->target_qubits,
                                  gate->num_qubits)) {
                return false;
            }
        }
    }
    
    return true;
}

bool measure_quantum_state(
    const quantum_geometric_tensor_network_t* qgtn,
    size_t qubit,
    double* probability_zero,
    double* probability_one) {
    
    if (!qgtn || !probability_zero || !probability_one) {
        set_error("Invalid arguments to measure_quantum_state");
        return false;
    }
    
    // Contract network to get amplitudes
    ComplexFloat* state;
    size_t dims[1];
    size_t num_dims;
    
    if (!contract_full_network(qgtn->network, &state, dims, &num_dims)) {
        set_error("Failed to contract network for measurement");
        return false;
    }
    
    // Calculate probabilities
    *probability_zero = state[0].real * state[0].real + 
                       state[0].imag * state[0].imag;
    *probability_one = state[1].real * state[1].real +
                      state[1].imag * state[1].imag;
    
    free(state);
    return true;
}

bool get_quantum_state(
    const quantum_geometric_tensor_network_t* qgtn,
    ComplexFloat** state_vector,
    size_t* dimension) {
    
    if (!qgtn || !state_vector || !dimension) {
        set_error("Invalid arguments to get_quantum_state");
        return false;
    }
    
    // Contract network to get state vector
    size_t dims[1];
    size_t num_dims;
    
    if (!contract_full_network(qgtn->network, state_vector, dims, &num_dims)) {
        set_error("Failed to contract network for state vector");
        return false;
    }
    
    *dimension = dims[0];
    return true;
}

// Geometric operations
bool compute_quantum_geometric_tensor(
    const quantum_geometric_tensor_network_t* qgtn,
    size_t param_i,
    size_t param_j,
    ComplexFloat* result) {
    
    if (!qgtn || !result) {
        set_error("Invalid arguments to compute_quantum_geometric_tensor");
        return false;
    }

    // Initialize numerical backend if needed
    numerical_config_t config = {
        .type = NUMERICAL_BACKEND_CPU,
        .max_threads = 1,
        .use_fma = true,
        .use_avx = true,
        .use_neon = true,
        .cache_size = 32 * 1024 * 1024
    };
    
    if (!initialize_numerical_backend(&config)) {
        set_error("Failed to initialize numerical backend");
        return false;
    }
    
    // Get quantum state
    ComplexFloat* state;
    size_t dim;
    if (!get_quantum_state(qgtn, &state, &dim)) {
        return false;
    }
    
    // Compute gradients using higher order method
    ComplexFloat* grad_i;
    ComplexFloat* grad_j;
    size_t dim_i, dim_j;
    
    // Use multiple shift amounts for higher accuracy
    // M_PI_4 = π/4, M_PI/8 = π/8, M_PI/16 = π/16
    double shift_amounts[] = {M_PI_4, M_PI/8.0, M_PI/16.0};
    
    if (!compute_higher_order_gradient((quantum_geometric_tensor_network_t*)qgtn, 
                                     param_i, shift_amounts, 3, &grad_i, &dim_i)) {
        free(state);
        return false;
    }
    
    if (!compute_higher_order_gradient((quantum_geometric_tensor_network_t*)qgtn,
                                     param_j, shift_amounts, 3, &grad_j, &dim_j)) {
        free(state);
        free(grad_i);
        return false;
    }
    
    if (dim_i != dim_j || dim_i != dim) {
        free(state);
        free(grad_i);
        free(grad_j);
        return false;
    }
    
    // Compute geometric tensor components:
    // g_ij = <∂ψ/∂θi|∂ψ/∂θj>
    result->real = 0;
    result->imag = 0;
    
    for (size_t k = 0; k < dim; k++) {
        // Complex conjugate of grad_i
        ComplexFloat grad_i_conj = {grad_i[k].real, -grad_i[k].imag};
        
        // Inner product
        result->real += grad_i_conj.real * grad_j[k].real - 
                       grad_i_conj.imag * grad_j[k].imag;
        result->imag += grad_i_conj.real * grad_j[k].imag + 
                       grad_i_conj.imag * grad_j[k].real;
    }
    
    free(state);
    free(grad_i);
    free(grad_j);
    return true;
}

bool compute_quantum_metric(
    const quantum_geometric_tensor_network_t* qgtn,
    size_t param_i,
    size_t param_j,
    double* result) {
    
    if (!qgtn || !result) {
        set_error("Invalid arguments to compute_quantum_metric");
        return false;
    }
    
    // Compute geometric tensor
    ComplexFloat tensor;
    if (!compute_quantum_geometric_tensor(qgtn, param_i, param_j, &tensor)) {
        return false;
    }
    
    // Metric is real part of geometric tensor
    *result = tensor.real;
    return true;
}

bool compute_berry_curvature(
    const quantum_geometric_tensor_network_t* qgtn,
    size_t param_i,
    size_t param_j,
    double* result) {
    
    if (!qgtn || !result) {
        set_error("Invalid arguments to compute_berry_curvature");
        return false;
    }
    
    // Compute geometric tensor
    ComplexFloat tensor;
    if (!compute_quantum_geometric_tensor(qgtn, param_i, param_j, &tensor)) {
        return false;
    }
    
    // Berry curvature is imaginary part of geometric tensor
    *result = tensor.imag;
    return true;
}

// Distributed operations
bool distribute_computation(
    quantum_geometric_tensor_network_t* qgtn,
    const size_t* device_ids,
    size_t num_devices) {
    
    if (!qgtn || !device_ids || num_devices == 0) {
        set_error("Invalid arguments to distribute_computation");
        return false;
    }
    
    if (!qgtn->is_distributed) {
        set_error("Network not configured for distributed computation");
        return false;
    }
    
    // TODO: Implement actual distribution logic
    return true;
}

bool synchronize_distributed_state(
    quantum_geometric_tensor_network_t* qgtn) {
    
    if (!qgtn) {
        set_error("Invalid arguments to synchronize_distributed_state");
        return false;
    }
    
    if (!qgtn->is_distributed) {
        set_error("Network not configured for distributed computation");
        return false;
    }
    
    // TODO: Implement actual synchronization logic
    return true;
}

// Hardware acceleration
bool enable_hardware_acceleration(
    quantum_geometric_tensor_network_t* qgtn,
    HardwareType type) {
    
    if (!qgtn) {
        set_error("Invalid arguments to enable_hardware_acceleration");
        return false;
    }
    
    if (!qgtn->use_hardware_acceleration) {
        set_error("Network not configured for hardware acceleration");
        return false;
    }
    
    // TODO: Implement actual hardware acceleration logic
    qgtn->use_hardware_acceleration = true;
    return true;
}

bool disable_hardware_acceleration(
    quantum_geometric_tensor_network_t* qgtn) {
    
    if (!qgtn) {
        set_error("Invalid arguments to disable_hardware_acceleration");
        return false;
    }
    
    qgtn->use_hardware_acceleration = false;
    return true;
}
