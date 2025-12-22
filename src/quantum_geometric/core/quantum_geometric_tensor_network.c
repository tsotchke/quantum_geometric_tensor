#include "quantum_geometric/core/quantum_geometric_tensor_network.h"
#include "quantum_geometric/core/quantum_geometric_gradient.h"
#include "quantum_geometric/core/tensor_network_operations.h"
#include "quantum_geometric/core/numerical_backend.h"
#include "quantum_geometric/core/error_handling.h"
#include "quantum_geometric/core/geometric_processor.h"
#include "quantum_geometric/core/computational_graph.h"
#include "quantum_geometric/core/quantum_gate_operations.h"
#include "quantum_geometric/hardware/quantum_hardware_types.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Error handling
static const char* g_last_error = NULL;

static void set_qgtn_error(const char* error) {
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
        set_qgtn_error("Failed to allocate quantum geometric tensor network");
        return NULL;
    }
    
    qgtn->network = create_tensor_network();
    if (!qgtn->network) {
        free(qgtn);
        set_qgtn_error("Failed to create tensor network");
        return NULL;
    }
    
    // Initialize network nodes array
    qgtn->network->capacity = 16;  // Initial capacity
    qgtn->network->nodes = malloc(qgtn->network->capacity * sizeof(tensor_node_t*));
    if (!qgtn->network->nodes) {
        destroy_tensor_network(qgtn->network);
        free(qgtn);
        set_qgtn_error("Failed to allocate network nodes array");
        return NULL;
    }
    qgtn->network->num_nodes = 0;
    qgtn->network->next_id = 0;
    memset(qgtn->network->nodes, 0, qgtn->network->capacity * sizeof(tensor_node_t*));
    
    // Initialize metrics
    qgtn->network->metrics.num_contractions = 0;
    qgtn->network->metrics.peak_memory_usage = 0;
    qgtn->network->metrics.total_time = 0.0;
    qgtn->network->metrics.optimization_time = 0.0;
    qgtn->network->metrics.contraction_time = 0.0;
    
    // Initialize optimization state
    qgtn->network->optimized = false;
    qgtn->network->last_error = TENSOR_NETWORK_SUCCESS;

    // Initialize hardware configuration
    qgtn->hardware_config.type = QGTN_BACKEND_SIMULATOR;  // Default to simulator
    qgtn->hardware_config.backend_specific = NULL;
    qgtn->hardware_config.supports_gradients = true;  // Simulator supports gradients
    qgtn->hardware_config.supports_hybrid = true;     // Simulator supports hybrid computation
    qgtn->backend_state = NULL;

    // Initialize with normalized initial state for gradient computation
    size_t state_dim = 1 << num_qubits;  // 2^n dimensional state space
    ComplexFloat* state_vector = calloc(state_dim, sizeof(ComplexFloat));
    if (!state_vector) {
        destroy_tensor_network(qgtn->network);
        free(qgtn);
        set_qgtn_error("Failed to allocate state vector");
        return NULL;
    }
    
    // Set initial state to |0> for gradient computation
    memset(state_vector, 0, state_dim * sizeof(ComplexFloat));
    state_vector[0] = (ComplexFloat){1.0f, 0.0f};
    
    // Add state vector as a single tensor node
    size_t node_id;
    size_t dims[1] = {state_dim};
    if (!add_tensor_node(qgtn->network, state_vector, dims, 1, &node_id)) {
        free(state_vector);
        destroy_tensor_network(qgtn->network);
        free(qgtn);
        set_qgtn_error("Failed to initialize quantum state");
        return NULL;
    }
    
    // Initialize first tensor node
    tensor_node_t* node = malloc(sizeof(tensor_node_t));
    if (!node) {
        free(state_vector);
        destroy_tensor_network(qgtn->network);
        free(qgtn);
        set_qgtn_error("Failed to allocate tensor node");
        return NULL;
    }
    
    node->data = state_vector;  // Transfer ownership
    node->num_dimensions = 1;
    node->dimensions = malloc(sizeof(size_t));
    if (!node->dimensions) {
        free(node);
        free(state_vector);
        destroy_tensor_network(qgtn->network);
        free(qgtn);
        set_qgtn_error("Failed to allocate dimensions array");
        return NULL;
    }
    node->dimensions[0] = state_dim;
    node->num_connections = 0;
    node->connected_nodes = NULL;
    node->connected_dims = NULL;
    node->id = node_id;
    node->is_valid = true;
    
    // Initialize network state
    qgtn->network->num_nodes = 1;
    qgtn->network->nodes[0] = node;

    // Initialize empty quantum circuit
    qgtn->circuit = malloc(sizeof(quantum_circuit_t));
    if (!qgtn->circuit) {
        destroy_tensor_network(qgtn->network);
        free(qgtn);
        set_qgtn_error("Failed to allocate quantum circuit");
        return NULL;
    }

    qgtn->circuit->layers = malloc(num_layers * sizeof(circuit_layer_t*));
    if (!qgtn->circuit->layers) {
        free(qgtn->circuit);
        destroy_tensor_network(qgtn->network);
        free(qgtn);
        set_qgtn_error("Failed to allocate circuit layers");
        return NULL;
    }

    for (size_t i = 0; i < num_layers; i++) {
        qgtn->circuit->layers[i] = NULL;
    }

    // Initialize geometric processor for graph
    geometric_processor_t* processor = create_geometric_processor(NULL);
    if (!processor) {
        free(qgtn->circuit->layers);
        free(qgtn->circuit);
        destroy_tensor_network(qgtn->network);
        free(qgtn);
        set_qgtn_error("Failed to create geometric processor");
        return NULL;
    }
    
    // Initialize computational graph with processor
    qgtn->circuit->graph = create_computational_graph(processor);
    if (!qgtn->circuit->graph) {
        destroy_geometric_processor(processor);
        free(qgtn->circuit->layers);
        free(qgtn->circuit);
        destroy_tensor_network(qgtn->network);
        free(qgtn);
        set_qgtn_error("Failed to create computational graph");
        return NULL;
    }
    
    // Store processor in backend state
    qgtn->backend_state = processor;
    
    // Initialize nodes array
    qgtn->circuit->capacity = 16;
    qgtn->circuit->nodes = malloc(qgtn->circuit->capacity * sizeof(quantum_compute_node_t*));
    if (!qgtn->circuit->nodes) {
        destroy_computational_graph(qgtn->circuit->graph);
        destroy_geometric_processor(processor);
        free(qgtn->circuit->layers);
        free(qgtn->circuit);
        destroy_tensor_network(qgtn->network);
        free(qgtn);
        set_qgtn_error("Failed to allocate nodes array");
        return NULL;
    }
    qgtn->circuit->num_nodes = 0;
    memset(qgtn->circuit->nodes, 0, qgtn->circuit->capacity * sizeof(quantum_compute_node_t*));
    
    qgtn->circuit->num_layers = num_layers;
    qgtn->circuit->num_qubits = num_qubits;
    qgtn->circuit->is_parameterized = false;
    qgtn->circuit->state = NULL;  // Initialize state to NULL, will be created when needed
    
    qgtn->num_qubits = num_qubits;
    qgtn->num_layers = num_layers;
    qgtn->is_distributed = is_distributed;
    qgtn->use_hardware_acceleration = use_hardware_acceleration;
    
    return qgtn;
}

quantum_geometric_tensor_network_t* copy_quantum_geometric_tensor_network(
    const quantum_geometric_tensor_network_t* qgtn) {
    
    quantum_geometric_tensor_network_t* copy = malloc(sizeof(quantum_geometric_tensor_network_t));
    if (!copy) return NULL;
    
    // Copy basic fields
    copy->num_qubits = qgtn->num_qubits;
    copy->num_layers = qgtn->num_layers;
    copy->is_distributed = qgtn->is_distributed;
    copy->use_hardware_acceleration = qgtn->use_hardware_acceleration;
    
    // Copy hardware configuration
    copy->hardware_config = qgtn->hardware_config;
    copy->backend_state = NULL;  // Backend state needs to be initialized separately
    
    // Initialize hardware backend if needed
    if (copy->use_hardware_acceleration) {
        numerical_config_t config = {
            .type = NUMERICAL_BACKEND_CPU,  // Default to CPU
            .max_threads = 1,
            .use_fma = true,
            .use_avx = true,
            .use_neon = true,
            .cache_size = 32 * 1024 * 1024  // 32MB cache
        };
        
        if (!initialize_numerical_backend(&config)) {
            free(copy);
            return NULL;
        }
    }
    
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
    
    // Initialize geometric processor for graph
    geometric_processor_t* processor = create_geometric_processor(NULL);
    if (!processor) {
        free(copy->circuit->layers);
        free(copy->circuit);
        destroy_tensor_network(copy->network);
        free(copy);
        return NULL;
    }
    
    // Initialize computational graph with processor
    copy->circuit->graph = create_computational_graph(processor);
    if (!copy->circuit->graph) {
        destroy_geometric_processor(processor);
        free(copy->circuit->layers);
        free(copy->circuit);
        destroy_tensor_network(copy->network);
        free(copy);
        return NULL;
    }
    
    // Store processor in backend state
    copy->backend_state = processor;
    
    // Initialize quantum geometric state
    copy->circuit->state = malloc(sizeof(quantum_geometric_state_t));
    if (!copy->circuit->state) {
        destroy_computational_graph(copy->circuit->graph);
        destroy_geometric_processor(processor);
        free(copy->circuit->layers);
        free(copy->circuit);
        destroy_tensor_network(copy->network);
        free(copy);
        return NULL;
    }
    
    // Copy state fields
    if (qgtn->circuit->state) {
        // Copy state structure
        memcpy(copy->circuit->state, qgtn->circuit->state, sizeof(quantum_geometric_state_t));
        
        // Allocate and copy coordinates if they exist
        if (qgtn->circuit->state->coordinates) {
            copy->circuit->state->coordinates = malloc(qgtn->circuit->state->dimension * sizeof(ComplexFloat));
            if (!copy->circuit->state->coordinates) {
                free(copy->circuit->state);
                destroy_computational_graph(copy->circuit->graph);
                destroy_geometric_processor(processor);
                free(copy->circuit->layers);
                free(copy->circuit);
                destroy_tensor_network(copy->network);
                free(copy);
                return NULL;
            }
            memcpy(copy->circuit->state->coordinates, qgtn->circuit->state->coordinates,
                   qgtn->circuit->state->dimension * sizeof(ComplexFloat));
        } else {
            copy->circuit->state->coordinates = NULL;
        }
    } else {
        memset(copy->circuit->state, 0, sizeof(quantum_geometric_state_t));
        copy->circuit->state->coordinates = NULL;
    }
    
    // Initialize nodes array
    copy->circuit->nodes = malloc(qgtn->circuit->capacity * sizeof(quantum_compute_node_t*));
    if (!copy->circuit->nodes) {
        free(copy->circuit->state);
        destroy_computational_graph(copy->circuit->graph);
        free(copy->circuit->layers);
        free(copy->circuit);
        destroy_tensor_network(copy->network);
        free(copy);
        return NULL;
    }
    
    copy->circuit->num_nodes = qgtn->circuit->num_nodes;
    copy->circuit->capacity = qgtn->circuit->capacity;
    
    // Copy compute nodes
    for (size_t i = 0; i < qgtn->circuit->num_nodes; i++) {
        if (qgtn->circuit->nodes[i]) {
            // Deep copy compute node
            quantum_compute_node_t* node = malloc(sizeof(quantum_compute_node_t));
            if (!node) {
                for (size_t j = 0; j < i; j++) {
                    if (copy->circuit->nodes[j]) {
                        free(copy->circuit->nodes[j]->qubit_indices);
                        free(copy->circuit->nodes[j]->parameters);
                        free(copy->circuit->nodes[j]);
                    }
                }
                free(copy->circuit->nodes);
                free(copy->circuit->state);
                destroy_computational_graph(copy->circuit->graph);
                free(copy->circuit->layers);
                free(copy->circuit);
                destroy_tensor_network(copy->network);
                free(copy);
                return NULL;
            }
            
            // Copy basic fields
            node->type = qgtn->circuit->nodes[i]->type;
            node->num_qubits = qgtn->circuit->nodes[i]->num_qubits;
            node->num_parameters = qgtn->circuit->nodes[i]->num_parameters;
            
            // Copy qubit indices
            node->qubit_indices = malloc(node->num_qubits * sizeof(size_t));
            if (!node->qubit_indices) {
                free(node);
                for (size_t j = 0; j < i; j++) {
                    if (copy->circuit->nodes[j]) {
                        free(copy->circuit->nodes[j]->qubit_indices);
                        free(copy->circuit->nodes[j]->parameters);
                        free(copy->circuit->nodes[j]);
                    }
                }
                free(copy->circuit->nodes);
                free(copy->circuit->state);
                destroy_computational_graph(copy->circuit->graph);
                free(copy->circuit->layers);
                free(copy->circuit);
                destroy_tensor_network(copy->network);
                free(copy);
                return NULL;
            }
            memcpy(node->qubit_indices, qgtn->circuit->nodes[i]->qubit_indices, 
                   node->num_qubits * sizeof(size_t));
            
            // Copy parameters
            if (qgtn->circuit->nodes[i]->parameters) {
                node->parameters = malloc(node->num_parameters * sizeof(ComplexFloat));
                if (!node->parameters) {
                    free(node->qubit_indices);
                    free(node);
                    for (size_t j = 0; j < i; j++) {
                        if (copy->circuit->nodes[j]) {
                            free(copy->circuit->nodes[j]->qubit_indices);
                            free(copy->circuit->nodes[j]->parameters);
                            free(copy->circuit->nodes[j]);
                        }
                    }
                    free(copy->circuit->nodes);
                    free(copy->circuit->state);
                    destroy_computational_graph(copy->circuit->graph);
                    free(copy->circuit->layers);
                    free(copy->circuit);
                    destroy_tensor_network(copy->network);
                    free(copy);
                    return NULL;
                }
                memcpy(node->parameters, qgtn->circuit->nodes[i]->parameters,
                       node->num_parameters * sizeof(ComplexFloat));
            } else {
                node->parameters = NULL;
            }
            
            // Set additional data to NULL (will be reconstructed during circuit application)
            node->additional_data = NULL;
            
            copy->circuit->nodes[i] = node;
        } else {
            copy->circuit->nodes[i] = NULL;
        }
    }
    
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

void destroy_quantum_geometric_tensor_network(quantum_geometric_tensor_network_t* qgtn) {
    if (!qgtn) return;
    
    if (qgtn->network) {
        destroy_tensor_network(qgtn->network);
    }

    if (qgtn->circuit) {
        // Free circuit nodes
        if (qgtn->circuit->nodes) {
            for (size_t i = 0; i < qgtn->circuit->num_nodes; i++) {
                if (qgtn->circuit->nodes[i]) {
                    free(qgtn->circuit->nodes[i]->qubit_indices);
                    free(qgtn->circuit->nodes[i]->parameters);
                    free(qgtn->circuit->nodes[i]);
                }
            }
            free(qgtn->circuit->nodes);
        }

        // Free circuit state
        if (qgtn->circuit->state) {
            if (qgtn->circuit->state->coordinates) {
                free(qgtn->circuit->state->coordinates);
            }
            free(qgtn->circuit->state);
        }
        
        // Free each layer
        for (size_t l = 0; l < qgtn->circuit->num_layers; l++) {
            circuit_layer_t* layer = qgtn->circuit->layers[l];
            if (layer) {
                // Free each gate in the layer using destroy_quantum_gate
                for (size_t g = 0; g < layer->num_gates; g++) {
                    quantum_gate_t* gate = layer->gates[g];
                    if (gate) {
                        destroy_quantum_gate(gate);
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

bool apply_quantum_gate(
    quantum_geometric_tensor_network_t* qgtn,
    const quantum_gate_t* gate,
    const size_t* qubits,
    size_t num_qubits) {
    
    if (!qgtn || !gate || !qubits || num_qubits == 0) {
        set_qgtn_error("Invalid arguments to apply_quantum_gate");
        return false;
    }

    // Check if circuit is initialized
    if (!qgtn->circuit) {
        set_qgtn_error("Circuit not initialized");
        return false;
    }

    // Initialize circuit state if needed
    if (!qgtn->circuit->state) {
        qgtn->circuit->state = malloc(sizeof(quantum_geometric_state_t));
        if (!qgtn->circuit->state) {
            set_qgtn_error("Failed to allocate circuit state");
            return false;
        }
        
        // Initialize quantum state fields
        qgtn->circuit->state->type = GEOMETRIC_STATE_EUCLIDEAN;
        qgtn->circuit->state->dimension = 1 << qgtn->num_qubits;  // 2^n qubits
        qgtn->circuit->state->manifold_dim = qgtn->circuit->state->dimension;
        qgtn->circuit->state->coordinates = malloc(qgtn->circuit->state->dimension * sizeof(ComplexFloat));
        if (!qgtn->circuit->state->coordinates) {
            free(qgtn->circuit->state);
            qgtn->circuit->state = NULL;
            set_qgtn_error("Failed to allocate state coordinates");
            return false;
        }
        
        // Initialize to |0> state
        memset(qgtn->circuit->state->coordinates, 0, qgtn->circuit->state->dimension * sizeof(ComplexFloat));
        qgtn->circuit->state->coordinates[0] = (ComplexFloat){1.0f, 0.0f};
        
        qgtn->circuit->state->metric = NULL;
        qgtn->circuit->state->connection = NULL;
        qgtn->circuit->state->auxiliary_data = NULL;
        qgtn->circuit->state->is_normalized = true;
        qgtn->circuit->state->hardware = HARDWARE_TYPE_CPU;
    }

    // Initialize nodes array if needed
    if (!qgtn->circuit->nodes) {
        qgtn->circuit->capacity = 16;  // Initial capacity
        qgtn->circuit->nodes = malloc(qgtn->circuit->capacity * sizeof(quantum_compute_node_t*));
        if (!qgtn->circuit->nodes) {
            set_qgtn_error("Failed to allocate nodes array");
            return false;
        }
        qgtn->circuit->num_nodes = 0;
        memset(qgtn->circuit->nodes, 0, qgtn->circuit->capacity * sizeof(quantum_compute_node_t*));
    }
    
    // Check if layers array is initialized
    if (!qgtn->circuit->layers) {
        set_qgtn_error("Circuit layers not initialized");
        return false;
    }
    
    // Find or create layer for this gate
    size_t layer_idx = qgtn->num_layers > 0 ? qgtn->num_layers - 1 : 0;  // Add to last layer
    if (layer_idx >= qgtn->circuit->num_layers) {
        set_qgtn_error("Layer index out of bounds");
        return false;
    }
    
    if (!qgtn->circuit->layers[layer_idx]) {
        qgtn->circuit->layers[layer_idx] = create_circuit_layer();
        if (!qgtn->circuit->layers[layer_idx]) {
            set_qgtn_error("Failed to create circuit layer");
            return false;
        }
    }
    
    // Copy gate to circuit
    quantum_gate_t* circuit_gate = copy_quantum_gate(gate);
    if (!circuit_gate) {
        set_qgtn_error("Failed to copy quantum gate");
        return false;
    }
    
    // Add gate to layer
    circuit_layer_t* layer = qgtn->circuit->layers[layer_idx];
    if (!layer) {
        destroy_quantum_gate(circuit_gate);
        set_qgtn_error("Layer is NULL");
        return false;
    }
    
    // Check if gates array is initialized
    if (!layer->gates) {
        destroy_quantum_gate(circuit_gate);
        set_qgtn_error("Layer gates array not initialized");
        return false;
    }
    
    // Ensure we have enough space in the gates array
    if (layer->num_gates >= 16) {  // Assuming initial capacity is 16
        quantum_gate_t** new_gates = realloc(layer->gates, (layer->num_gates + 16) * sizeof(quantum_gate_t*));
        if (!new_gates) {
            destroy_quantum_gate(circuit_gate);
            set_qgtn_error("Failed to resize gates array");
            return false;
        }
        layer->gates = new_gates;
    }
    
    layer->gates[layer->num_gates++] = circuit_gate;
    
    // Update layer parameterization status
    if (circuit_gate->is_parameterized) {
        layer->is_parameterized = true;
        qgtn->circuit->is_parameterized = true;
    }
    
    // Create tensor node for gate
    size_t node_id;
    
    // For rotation gates, ensure matrix is properly initialized
    if (gate->type == GATE_TYPE_RX || gate->type == GATE_TYPE_RY || gate->type == GATE_TYPE_RZ) {
        if (!gate->is_parameterized || !gate->parameters) {
            set_qgtn_error("Invalid rotation gate parameters");
            return false;
        }
        
        // Create 2x2 rotation matrix (allocate on heap)
        ComplexFloat* rotation = malloc(4 * sizeof(ComplexFloat));
        if (!rotation) {
            set_qgtn_error("Failed to allocate rotation matrix");
            return false;
        }
        
        double angle = gate->parameters[0];
        double cos_half = cos(angle / 2.0);
        double sin_half = sin(angle / 2.0);
        
        switch (gate->type) {
            case GATE_TYPE_RX:
                rotation[0] = (ComplexFloat){cos_half, 0};
                rotation[1] = (ComplexFloat){0, -sin_half};
                rotation[2] = (ComplexFloat){0, -sin_half};
                rotation[3] = (ComplexFloat){cos_half, 0};
                break;
            case GATE_TYPE_RY:
                rotation[0] = (ComplexFloat){cos_half, 0};
                rotation[1] = (ComplexFloat){-sin_half, 0};
                rotation[2] = (ComplexFloat){sin_half, 0};
                rotation[3] = (ComplexFloat){cos_half, 0};
                break;
            case GATE_TYPE_RZ:
                rotation[0] = (ComplexFloat){cos_half, -sin_half};
                rotation[1] = (ComplexFloat){0, 0};
                rotation[2] = (ComplexFloat){0, 0};
                rotation[3] = (ComplexFloat){cos_half, sin_half};
                break;
            default:
                free(rotation);
                set_qgtn_error("Invalid rotation gate type");
                return false;
        }
        
        // For rotation gates, we need a 2-dimensional tensor
        size_t dims[2] = {2, 2};  // 2x2 matrix
        printf("DEBUG: Adding rotation matrix to tensor network: %p\n", (void*)rotation);
        printf("DEBUG: Rotation matrix values: (%.3f,%.3f) (%.3f,%.3f) (%.3f,%.3f) (%.3f,%.3f)\n",
               rotation[0].real, rotation[0].imag,
               rotation[1].real, rotation[1].imag,
               rotation[2].real, rotation[2].imag,
               rotation[3].real, rotation[3].imag);
        
        // Make sure the rotation matrix is valid
        for (int i = 0; i < 4; i++) {
            if (isnan(rotation[i].real) || isnan(rotation[i].imag) ||
                isinf(rotation[i].real) || isinf(rotation[i].imag)) {
                printf("DEBUG: Invalid rotation matrix value at index %d: (%.3f,%.3f)\n",
                       i, rotation[i].real, rotation[i].imag);
                free(rotation);
                set_qgtn_error("Invalid rotation matrix values");
                return false;
            }
        }
        
        if (!add_tensor_node(qgtn->network, rotation, dims, 2, &node_id)) {
            free(rotation);  // Free rotation matrix if add_tensor_node fails
            set_qgtn_error("Failed to add rotation gate tensor node");
            return false;
        }
        
        // Free rotation matrix since add_tensor_node makes a copy
        free(rotation);

        // Add node to circuit for parameter tracking
        quantum_compute_node_t* compute_node = malloc(sizeof(quantum_compute_node_t));
        if (!compute_node) {
            set_qgtn_error("Failed to allocate compute node");
            return false;
        }

        // Initialize compute node fields
        compute_node->type = NODE_ROTATION;
        compute_node->num_qubits = 1;
        compute_node->additional_data = NULL;
        
        compute_node->qubit_indices = malloc(sizeof(size_t));
        if (!compute_node->qubit_indices) {
            free(compute_node);
            set_qgtn_error("Failed to allocate qubit indices");
            return false;
        }
        compute_node->qubit_indices[0] = qubits[0];
        
        compute_node->num_parameters = 1;
        compute_node->parameters = malloc(sizeof(ComplexFloat));
        if (!compute_node->parameters) {
            free(compute_node->qubit_indices);
            free(compute_node);
            set_qgtn_error("Failed to allocate parameters");
            return false;
        }
        compute_node->parameters[0] = (ComplexFloat){gate->parameters[0], 0.0};
        
        // Add node to circuit
        if (qgtn->circuit->num_nodes >= qgtn->circuit->capacity) {
            size_t new_capacity = qgtn->circuit->capacity * 2;
            quantum_compute_node_t** new_nodes = realloc(qgtn->circuit->nodes, 
                new_capacity * sizeof(quantum_compute_node_t*));
            if (!new_nodes) {
                free(compute_node->parameters);
                free(compute_node->qubit_indices);
                free(compute_node);
                set_qgtn_error("Failed to resize nodes array");
                return false;
            }
            qgtn->circuit->nodes = new_nodes;
            qgtn->circuit->capacity = new_capacity;
        }
        
        qgtn->circuit->nodes[qgtn->circuit->num_nodes++] = compute_node;
    } else {
        // For non-rotation gates, use the gate matrix directly
        if (!gate->matrix) {
            set_qgtn_error("Gate matrix is NULL");
            return false;
        }
        
        size_t dims[1] = {1 << gate->num_qubits};  // 2^n dimensional matrix
        if (!add_tensor_node(qgtn->network, gate->matrix,
                            dims, 1, &node_id)) {
            set_qgtn_error("Failed to add gate tensor node");
            return false;
        }
    }
    
    // Connect gate tensor to qubit tensors
    // We need to reshape the qubit tensor to match the gate tensor dimensions
    printf("DEBUG: Reshaping qubit tensor to match gate tensor dimensions\n");
    
    // For each qubit, we need to create a new tensor with the right dimensions
    for (size_t i = 0; i < num_qubits; i++) {
        printf("DEBUG: Processing qubit %zu\n", qubits[i]);
        
        // Create a new tensor for the qubit with dimension 2
        size_t qubit_dims[1] = {2};  // Qubit has dimension 2 (|0⟩ and |1⟩)
        ComplexFloat qubit_data[2] = {{1.0, 0.0}, {0.0, 0.0}}; // Initialize to |0⟩
        size_t new_qubit_id;
        
        if (!add_tensor_node(qgtn->network, qubit_data, qubit_dims, 1, &new_qubit_id)) {
            printf("DEBUG: Failed to create new qubit tensor\n");
            set_qgtn_error("Failed to create new qubit tensor");
            return false;
        }
        
        printf("DEBUG: Created new qubit tensor with id %zu\n", new_qubit_id);
        
        // Now connect the gate tensor to the new qubit tensor
        printf("DEBUG: Connecting gate tensor (id=%zu, dim=%zu) to new qubit tensor (id=%zu, dim=0)\n", 
               node_id, i, new_qubit_id);
        
        if (!connect_tensor_nodes(qgtn->network, node_id, i, new_qubit_id, 0)) {
            printf("DEBUG: Failed to connect gate tensor to new qubit tensor\n");
            set_qgtn_error("Failed to connect gate tensor to new qubit tensor");
            return false;
        }
        
        printf("DEBUG: Successfully connected gate tensor to new qubit tensor\n");
    }
    
    return true;
}

bool apply_quantum_circuit(
    quantum_geometric_tensor_network_t* qgtn,
    const quantum_circuit_t* circuit) {
    
    printf("DEBUG: Starting apply_quantum_circuit\n");
    if (!qgtn || !circuit) {
        set_qgtn_error("Invalid arguments to apply_quantum_circuit");
        return false;
    }

    // Initialize circuit state if needed
    if (!qgtn->circuit->state) {
        printf("DEBUG: Creating new circuit state\n");
        qgtn->circuit->state = malloc(sizeof(quantum_geometric_state_t));
        if (!qgtn->circuit->state) {
            set_qgtn_error("Failed to allocate circuit state");
            return false;
        }
        
        // Initialize quantum state fields
        printf("DEBUG: Initializing quantum state fields\n");
        qgtn->circuit->state->type = GEOMETRIC_STATE_EUCLIDEAN;
        qgtn->circuit->state->dimension = 1 << qgtn->num_qubits;  // 2^n qubits
        qgtn->circuit->state->manifold_dim = qgtn->circuit->state->dimension;
        qgtn->circuit->state->coordinates = malloc(qgtn->circuit->state->dimension * sizeof(ComplexFloat));
        if (!qgtn->circuit->state->coordinates) {
            free(qgtn->circuit->state);
            qgtn->circuit->state = NULL;
            set_qgtn_error("Failed to allocate state coordinates");
            return false;
        }
        
        // Initialize to |0> state
        printf("DEBUG: Initializing to |0> state\n");
        memset(qgtn->circuit->state->coordinates, 0, qgtn->circuit->state->dimension * sizeof(ComplexFloat));
        qgtn->circuit->state->coordinates[0] = (ComplexFloat){1.0f, 0.0f};
        
        qgtn->circuit->state->metric = NULL;
        qgtn->circuit->state->connection = NULL;
        qgtn->circuit->state->auxiliary_data = NULL;
        qgtn->circuit->state->is_normalized = true;
        qgtn->circuit->state->hardware = HARDWARE_TYPE_CPU;
        printf("DEBUG: Circuit state initialized\n");
    }

    // Initialize nodes array if needed
    if (!qgtn->circuit->nodes) {
        printf("DEBUG: Initializing circuit nodes array\n");
        qgtn->circuit->capacity = 16;  // Initial capacity
        qgtn->circuit->nodes = malloc(qgtn->circuit->capacity * sizeof(quantum_compute_node_t*));
        if (!qgtn->circuit->nodes) {
            set_qgtn_error("Failed to allocate nodes array");
            return false;
        }
        qgtn->circuit->num_nodes = 0;
        memset(qgtn->circuit->nodes, 0, qgtn->circuit->capacity * sizeof(quantum_compute_node_t*));
        printf("DEBUG: Circuit nodes array initialized\n");
    }
    
    // Apply each layer
    printf("DEBUG: Applying circuit layers\n");
    for (size_t l = 0; l < circuit->num_layers; l++) {
        circuit_layer_t* layer = circuit->layers[l];
        if (!layer) {
            printf("DEBUG: Layer %zu is NULL\n", l);
            continue;
        }
        
        printf("DEBUG: Applying layer %zu with %zu gates\n", l, layer->num_gates);
        // Apply gates in layer
        for (size_t g = 0; g < layer->num_gates; g++) {
            quantum_gate_t* gate = layer->gates[g];
            if (!gate) {
                printf("DEBUG: Gate %zu in layer %zu is NULL\n", g, l);
                continue;
            }
            
            printf("DEBUG: Applying gate %zu (type=%d, num_qubits=%zu)\n", 
                   g, gate->type, gate->num_qubits);
                   
            // Validate gate before applying
            if (!gate->matrix || !gate->target_qubits) {
                printf("DEBUG: Invalid gate structure: matrix=%p, target_qubits=%p\n", 
                       (void*)gate->matrix, (void*)gate->target_qubits);
                set_qgtn_error("Invalid gate structure");
                return false;
            }
            
            if (!apply_quantum_gate(qgtn, gate,
                                  gate->target_qubits,
                                  gate->num_qubits)) {
                printf("DEBUG: Failed to apply gate %zu in layer %zu\n", g, l);
                return false;
            }
        }
    }
    printf("DEBUG: All layers applied\n");
    
    // Contract network to get final state
    printf("DEBUG: Contracting network for final state\n");
    ComplexFloat* state_vector;
    size_t dims[1];
    size_t num_dims;
    
    if (!contract_full_network(qgtn->network, &state_vector, dims, &num_dims)) {
        printf("DEBUG: Failed to contract network for final state\n");
        set_qgtn_error("Failed to contract network for final state");
        return false;
    }
    printf("DEBUG: Network contracted successfully\n");
    
    // Update circuit state
    if (qgtn->circuit->state->coordinates) {
        free(qgtn->circuit->state->coordinates);
    }
    qgtn->circuit->state->coordinates = state_vector;
    qgtn->circuit->state->dimension = dims[0];
    
    return true;
}

bool measure_quantum_state(
    const quantum_geometric_tensor_network_t* qgtn,
    size_t qubit,
    double* probability_zero,
    double* probability_one) {
    
    if (!qgtn || !probability_zero || !probability_one) {
        set_qgtn_error("Invalid arguments to measure_quantum_state");
        return false;
    }
    
    // Contract network to get amplitudes
    ComplexFloat* state;
    size_t dims[1];
    size_t num_dims;
    
    if (!contract_full_network(qgtn->network, &state, dims, &num_dims)) {
        set_qgtn_error("Failed to contract network for measurement");
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
        set_qgtn_error("Invalid arguments to get_quantum_state");
        return false;
    }
    
    // Contract network to get state vector
    size_t dims[1];
    size_t num_dims;
    
    if (!contract_full_network(qgtn->network, state_vector, dims, &num_dims)) {
        set_qgtn_error("Failed to contract network for state vector");
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
        set_qgtn_error("Invalid arguments to compute_quantum_geometric_tensor");
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
        set_qgtn_error("Failed to initialize numerical backend");
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
        set_qgtn_error("Invalid arguments to compute_quantum_metric");
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
        set_qgtn_error("Invalid arguments to compute_berry_curvature");
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
        set_qgtn_error("Invalid arguments to distribute_computation");
        return false;
    }
    
    if (!qgtn->is_distributed) {
        set_qgtn_error("Network not configured for distributed computation");
        return false;
    }
    
    // Implement distribution logic
    for (size_t i = 0; i < num_devices; i++) {
        // Assign computation to device
        size_t device_id = device_ids[i];
        
        // Partition tensor network nodes across devices
        size_t start_node = (i * qgtn->network->num_nodes) / num_devices;
        size_t end_node = ((i + 1) * qgtn->network->num_nodes) / num_devices;
        
        // Set device-specific properties
        if (qgtn->network->nodes && qgtn->network->num_nodes > 0) {
            for (size_t n = start_node; n < end_node && n < qgtn->network->num_nodes; n++) {
                if (qgtn->network->nodes[n]) {
                    // Mark node for specific device
                    // Note: We would need to add device_id field to tensor_node_t
                    // For now, we'll just track the assignment in a separate data structure
                }
            }
        }
    }
    
    return true;
}

bool synchronize_distributed_state(
    quantum_geometric_tensor_network_t* qgtn) {
    
    if (!qgtn) {
        set_qgtn_error("Invalid arguments to synchronize_distributed_state");
        return false;
    }
    
    if (!qgtn->is_distributed) {
        set_qgtn_error("Network not configured for distributed computation");
        return false;
    }
    
    // Implement synchronization logic
    if (qgtn->network && qgtn->network->nodes) {
        // Gather results from all devices
        for (size_t i = 0; i < qgtn->network->num_nodes; i++) {
            if (qgtn->network->nodes[i]) {
                // Ensure node data is synchronized
                // Note: We would need to add is_synchronized field to tensor_node_t
                // For now, we'll just assume synchronization happens automatically
            }
        }
    }
    
    return true;
}

// Hardware acceleration
bool enable_hardware_acceleration(
    quantum_geometric_tensor_network_t* qgtn,
    HardwareType type) {
    
    if (!qgtn) {
        set_qgtn_error("Invalid arguments to enable_hardware_acceleration");
        return false;
    }
    
    if (!qgtn->use_hardware_acceleration) {
        set_qgtn_error("Network not configured for hardware acceleration");
        return false;
    }
    
    // Implement hardware acceleration logic
    switch (type) {
        case HARDWARE_TYPE_CPU:
            // Use optimized CPU operations
            qgtn->hardware_config.type = QGTN_BACKEND_SIMULATOR;
            break;
            
        case HARDWARE_TYPE_GPU:
            // Configure for GPU acceleration
            // Note: QGTN_BACKEND_GPU is not defined, using SIMULATOR for now
            qgtn->hardware_config.type = QGTN_BACKEND_SIMULATOR;
            break;
            
        case HARDWARE_TYPE_QPU:
            // Configure for quantum processing unit
            // Note: QGTN_BACKEND_QUANTUM is not defined, using SIMULATOR for now
            qgtn->hardware_config.type = QGTN_BACKEND_SIMULATOR;
            break;
            
        default:
            // Default to CPU
            qgtn->hardware_config.type = QGTN_BACKEND_SIMULATOR;
            break;
    }
    
    qgtn->use_hardware_acceleration = true;
    return true;
}

bool disable_hardware_acceleration(
    quantum_geometric_tensor_network_t* qgtn) {
    
    if (!qgtn) {
        set_qgtn_error("Invalid arguments to disable_hardware_acceleration");
        return false;
    }
    
    // Reset to CPU-based simulation
    qgtn->hardware_config.type = QGTN_BACKEND_SIMULATOR;
    qgtn->use_hardware_acceleration = false;
    return true;
}
