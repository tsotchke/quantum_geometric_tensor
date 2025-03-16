#include "quantum_geometric/core/quantum_geometric_compute.h"
#include "quantum_geometric/core/quantum_geometric_operations.h"
#include "quantum_geometric/core/computational_graph.h"
#include "quantum_geometric/core/numerical_backend.h"
#include "quantum_geometric/core/quantum_scheduler.h"
#include "quantum_geometric/core/operation_fusion.h"
#include "quantum_geometric/core/geometric_processor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Add fields to existing quantum_circuit_t from quantum_types.h
static bool extend_quantum_circuit(quantum_circuit_t* circuit) {
    circuit->graph = NULL;
    circuit->state = NULL;
    circuit->nodes = NULL;
    circuit->num_nodes = 0;
    circuit->capacity = 0;
    return true;
}

// Initialize quantum circuit
quantum_circuit_t* quantum_circuit_create(size_t num_qubits) {
    quantum_circuit_t* circuit = malloc(sizeof(quantum_circuit_t));
    if (!circuit) return NULL;
    
    // Initialize computational graph
    geometric_processor_t* processor = create_geometric_processor(NULL);
    if (!processor) {
        free(circuit);
        return NULL;
    }
    
    circuit->graph = create_computational_graph(processor);
    if (!circuit->graph) {
        destroy_geometric_processor(processor);
        free(circuit);
        return NULL;
    }
    
    // Initialize quantum state
    qgt_error_t err = geometric_create_state(&circuit->state,
                                           GEOMETRIC_STATE_EUCLIDEAN,
                                           1 << num_qubits,
                                           HARDWARE_TYPE_CPU);
    if (err != QGT_SUCCESS) {
        destroy_computational_graph(circuit->graph);
        destroy_geometric_processor(processor);
        free(circuit);
        return NULL;
    }
    
    // Initialize to |0...0⟩ state
    memset(circuit->state->coordinates, 0, (1 << num_qubits) * sizeof(ComplexFloat));
    circuit->state->coordinates[0] = (ComplexFloat){1.0f, 0.0f};
    circuit->state->is_normalized = true;
    
    // Initialize node storage
    circuit->nodes = malloc(16 * sizeof(quantum_compute_node_t*));
    if (!circuit->nodes) {
        geometric_destroy_state(circuit->state);
        destroy_computational_graph(circuit->graph);
        destroy_geometric_processor(processor);
        free(circuit);
        return NULL;
    }
    
    circuit->num_qubits = num_qubits;
    circuit->num_nodes = 0;
    circuit->capacity = 16;
    
    return circuit;
}

// Add quantum operation node
bool quantum_circuit_add_operation(quantum_circuit_t* circuit,
                                 quantum_node_type_t type,
                                 const size_t* qubits,
                                 size_t num_qubits,
                                 const ComplexFloat* params,
                                 size_t num_params) {
    if (!circuit || !qubits || (num_params > 0 && !params)) return false;
    
    // Resize node array if needed
    if (circuit->num_nodes >= circuit->capacity) {
        size_t new_capacity = circuit->capacity * 2;
        quantum_compute_node_t** new_nodes = realloc(circuit->nodes,
            new_capacity * sizeof(quantum_compute_node_t*));
        if (!new_nodes) return false;
        circuit->nodes = new_nodes;
        circuit->capacity = new_capacity;
    }
    
    // Create new node
    quantum_compute_node_t* node = malloc(sizeof(quantum_compute_node_t));
    if (!node) return false;
    
    node->type = type;
    node->num_qubits = num_qubits;
    node->num_parameters = num_params;
    
    // Copy qubit indices
    node->qubit_indices = malloc(num_qubits * sizeof(size_t));
    if (!node->qubit_indices) {
        free(node);
        return false;
    }
    memcpy(node->qubit_indices, qubits, num_qubits * sizeof(size_t));
    
    // Copy parameters if any
    if (num_params > 0) {
        node->parameters = malloc(num_params * sizeof(ComplexFloat));
        if (!node->parameters) {
            free(node->qubit_indices);
            free(node);
            return false;
        }
        memcpy(node->parameters, params, num_params * sizeof(ComplexFloat));
    } else {
        node->parameters = NULL;
    }
    
    node->additional_data = NULL;
    
    // Add node to circuit
    circuit->nodes[circuit->num_nodes++] = node;
    
    // Add to computational graph
    computation_node_t* comp_node = add_node(circuit->graph, NODE_OPERATION, OP_QUANTUM, node);
    if (!comp_node) {
        free(node->parameters);
        free(node->qubit_indices);
        free(node);
        circuit->num_nodes--;
        return false;
    }
    
    return true;
}

// Execute quantum circuit
bool quantum_circuit_execute(quantum_circuit_t* circuit) {
    if (!circuit) return false;
    
    // Initialize numerical backend if not already done
    numerical_config_t config = {
        .type = NUMERICAL_BACKEND_CPU,
        .max_threads = 8,
        .use_fma = true,
        .use_avx = true,
        .use_neon = true,
        .cache_size = 32 * 1024 * 1024
    };
    
    if (!initialize_numerical_backend(&config)) return false;
    
    // Execute computational graph
    if (!execute_graph(circuit->graph)) return false;
    
    // Process each node
    for (size_t i = 0; i < circuit->num_nodes; i++) {
        quantum_compute_node_t* node = circuit->nodes[i];
        
        switch (node->type) {
            case NODE_UNITARY: {
                // Apply unitary operation using numerical backend
                size_t dim = 1 << node->num_qubits;
                ComplexFloat* unitary = node->parameters;
                ComplexFloat* temp = malloc(dim * sizeof(ComplexFloat));
                if (!temp) return false;
                
                if (!numerical_matrix_multiply(unitary,
                                            circuit->state->coordinates,
                                            temp,
                                            dim, dim, 1,
                                            false, false)) {
                    free(temp);
                    return false;
                }
                
                memcpy(circuit->state->coordinates, temp, dim * sizeof(ComplexFloat));
                free(temp);
                break;
            }

            case NODE_ROTATION: {
                // Handle rotation gates (RX, RY, RZ)
                size_t target = node->qubit_indices[0];
                float angle = node->parameters[0].real;
                size_t dim = 1 << circuit->num_qubits;
                
                // Create 2x2 rotation matrix
                ComplexFloat rotation[4];
                switch (node->additional_data ? *(gate_type_t*)node->additional_data : GATE_TYPE_RY) {
                    case GATE_TYPE_RX:
                        rotation[0] = (ComplexFloat){cosf(angle/2), 0};
                        rotation[1] = (ComplexFloat){0, -sinf(angle/2)};
                        rotation[2] = (ComplexFloat){0, -sinf(angle/2)};
                        rotation[3] = (ComplexFloat){cosf(angle/2), 0};
                        break;
                    case GATE_TYPE_RY:
                        rotation[0] = (ComplexFloat){cosf(angle/2), 0};
                        rotation[1] = (ComplexFloat){-sinf(angle/2), 0};
                        rotation[2] = (ComplexFloat){sinf(angle/2), 0};
                        rotation[3] = (ComplexFloat){cosf(angle/2), 0};
                        break;
                    case GATE_TYPE_RZ:
                        rotation[0] = (ComplexFloat){cosf(angle/2), -sinf(angle/2)};
                        rotation[1] = (ComplexFloat){0, 0};
                        rotation[2] = (ComplexFloat){0, 0};
                        rotation[3] = (ComplexFloat){cosf(angle/2), sinf(angle/2)};
                        break;
                    default:
                        return false;
                }
                
                // Apply rotation to target qubit
                ComplexFloat* temp = malloc(dim * sizeof(ComplexFloat));
                if (!temp) return false;
                memcpy(temp, circuit->state->coordinates, dim * sizeof(ComplexFloat));
                
                for (size_t i = 0; i < dim; i++) {
                    size_t i0 = i & ~(1ULL << target);  // i with target bit = 0
                    size_t i1 = i | (1ULL << target);   // i with target bit = 1
                    
                    if (i == i0) {  // Only process when target bit is 0
                        circuit->state->coordinates[i0] = complex_float_add(
                            complex_float_multiply(rotation[0], temp[i0]),
                            complex_float_multiply(rotation[1], temp[i1])
                        );
                        circuit->state->coordinates[i1] = complex_float_add(
                            complex_float_multiply(rotation[2], temp[i0]),
                            complex_float_multiply(rotation[3], temp[i1])
                        );
                    }
                }
                
                free(temp);
                break;
            }
            
            case NODE_MEASUREMENT: {
                // Perform measurement and collapse state
                size_t target = node->qubit_indices[0];
                size_t dim = 1 << circuit->num_qubits;
                ComplexFloat* new_state = malloc(dim * sizeof(ComplexFloat));
                if (!new_state) return false;
                
                // Calculate probabilities
                double p0 = 0.0, p1 = 0.0;
                for (size_t j = 0; j < dim; j++) {
                    if ((j & (1 << target)) == 0) {
                        p0 += complex_float_abs_squared(circuit->state->coordinates[j]);
                    } else {
                        p1 += complex_float_abs_squared(circuit->state->coordinates[j]);
                    }
                }
                
                // Collapse state based on measurement
                double rand_val = (double)rand() / RAND_MAX;
                bool result = rand_val < p0;
                double norm = sqrt(result ? p0 : p1);
                
                for (size_t j = 0; j < dim; j++) {
                    if (((j & (1 << target)) == 0) == result) {
                        new_state[j] = complex_float_multiply_real(
                            circuit->state->coordinates[j],
                            1.0f / norm
                        );
                    } else {
                        new_state[j] = COMPLEX_FLOAT_ZERO;
                    }
                }
                
                memcpy(circuit->state->coordinates, new_state, dim * sizeof(ComplexFloat));
                free(new_state);
                break;
            }
            
            case NODE_TENSOR_PRODUCT: {
                // Implement tensor product operation
                // TODO: Implement tensor product
                break;
            }
            
            case NODE_PARTIAL_TRACE: {
                // Implement partial trace operation
                // TODO: Implement partial trace
                break;
            }
            
            case NODE_QUANTUM_FOURIER: {
                // Implement quantum Fourier transform
                size_t dim = 1 << node->num_qubits;
                ComplexFloat* qft = malloc(dim * dim * sizeof(ComplexFloat));
                if (!qft) return false;
                
                // Build QFT matrix
                for (size_t j = 0; j < dim; j++) {
                    for (size_t k = 0; k < dim; k++) {
                        float angle = 2.0f * M_PI * j * k / dim;
                        qft[j * dim + k].real = cosf(angle) / sqrtf(dim);
                        qft[j * dim + k].imag = sinf(angle) / sqrtf(dim);
                    }
                }
                
                // Apply QFT
                ComplexFloat* temp = malloc(dim * sizeof(ComplexFloat));
                if (!temp) {
                    free(qft);
                    return false;
                }
                
                if (!numerical_matrix_multiply(qft,
                                            circuit->state->coordinates,
                                            temp,
                                            dim, dim, 1,
                                            false, false)) {
                    free(temp);
                    free(qft);
                    return false;
                }
                
                memcpy(circuit->state->coordinates, temp, dim * sizeof(ComplexFloat));
                free(temp);
                free(qft);
                break;
            }
            
            case NODE_QUANTUM_PHASE: {
                // Apply phase operation
                size_t target = node->qubit_indices[0];
                float phase = node->parameters[0].real;
                size_t dim = 1 << circuit->num_qubits;
                
                for (size_t j = 0; j < dim; j++) {
                    if (j & (1 << target)) {
                        float angle = phase;
                        ComplexFloat phase_factor = {cosf(angle), sinf(angle)};
                        circuit->state->coordinates[j] = complex_float_multiply(
                            circuit->state->coordinates[j],
                            phase_factor
                        );
                    }
                }
                break;
            }
        }
    }
    
    return true;
}

// Clean up quantum circuit
void quantum_circuit_destroy(quantum_circuit_t* circuit) {
    if (!circuit) return;
    
    // Clean up nodes
    for (size_t i = 0; i < circuit->num_nodes; i++) {
        quantum_compute_node_t* node = circuit->nodes[i];
        free(node->qubit_indices);
        free(node->parameters);
        free(node->additional_data);
        free(node);
    }
    free(circuit->nodes);
    
    // Clean up state and graph
    geometric_destroy_state(circuit->state);
    destroy_computational_graph(circuit->graph);
    
    free(circuit);
}

// Get quantum state
const quantum_geometric_state_t* quantum_circuit_get_state(const quantum_circuit_t* circuit) {
    return circuit ? circuit->state : NULL;
}

// Add common quantum gates
bool quantum_circuit_add_hadamard(quantum_circuit_t* circuit, size_t qubit) {
    if (!circuit) return false;
    
    ComplexFloat h[4] = {
        {1.0f / sqrtf(2.0f), 0.0f},
        {1.0f / sqrtf(2.0f), 0.0f},
        {1.0f / sqrtf(2.0f), 0.0f},
        {-1.0f / sqrtf(2.0f), 0.0f}
    };
    
    return quantum_circuit_add_operation(circuit, NODE_UNITARY,
                                       &qubit, 1, h, 4);
}

bool quantum_circuit_add_cnot(quantum_circuit_t* circuit,
                            size_t control,
                            size_t target) {
    if (!circuit) return false;
    
    size_t qubits[2] = {control, target};
    ComplexFloat cnot[16] = {
        {1.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f},
        {0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f},
        {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {1.0f, 0.0f},
        {0.0f, 0.0f}, {0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 0.0f}
    };
    
    return quantum_circuit_add_operation(circuit, NODE_UNITARY,
                                       qubits, 2, cnot, 16);
}

bool quantum_circuit_add_phase(quantum_circuit_t* circuit,
                             size_t qubit,
                             float phase) {
    if (!circuit) return false;
    
    ComplexFloat param = {phase, 0.0f};
    return quantum_circuit_add_operation(circuit, NODE_QUANTUM_PHASE,
                                       &qubit, 1, &param, 1);
}

bool quantum_circuit_add_measurement(quantum_circuit_t* circuit,
                                   size_t qubit) {
    if (!circuit) return false;
    
    return quantum_circuit_add_operation(circuit, NODE_MEASUREMENT,
                                       &qubit, 1, NULL, 0);
}

bool quantum_circuit_add_rotation(quantum_circuit_t* circuit,
                                size_t qubit,
                                gate_type_t rotation_type,
                                float angle) {
    if (!circuit || (rotation_type != GATE_TYPE_RX && 
                    rotation_type != GATE_TYPE_RY && 
                    rotation_type != GATE_TYPE_RZ)) {
        return false;
    }
    
    // Create parameter
    ComplexFloat param = {angle, 0.0f};
    
    // Create node
    quantum_compute_node_t* node = malloc(sizeof(quantum_compute_node_t));
    if (!node) return false;
    
    node->type = NODE_ROTATION;
    node->num_qubits = 1;
    node->qubit_indices = malloc(sizeof(size_t));
    if (!node->qubit_indices) {
        free(node);
        return false;
    }
    node->qubit_indices[0] = qubit;
    
    node->parameters = malloc(sizeof(ComplexFloat));
    if (!node->parameters) {
        free(node->qubit_indices);
        free(node);
        return false;
    }
    node->parameters[0] = param;
    node->num_parameters = 1;
    
    // Store rotation type in additional_data
    gate_type_t* type_ptr = malloc(sizeof(gate_type_t));
    if (!type_ptr) {
        free(node->parameters);
        free(node->qubit_indices);
        free(node);
        return false;
    }
    *type_ptr = rotation_type;
    node->additional_data = type_ptr;
    
    // Add node to circuit
    if (circuit->num_nodes >= circuit->capacity) {
        size_t new_capacity = circuit->capacity * 2;
        quantum_compute_node_t** new_nodes = realloc(circuit->nodes,
            new_capacity * sizeof(quantum_compute_node_t*));
        if (!new_nodes) {
            free(type_ptr);
            free(node->parameters);
            free(node->qubit_indices);
            free(node);
            return false;
        }
        circuit->nodes = new_nodes;
        circuit->capacity = new_capacity;
    }
    
    circuit->nodes[circuit->num_nodes++] = node;
    
    // Add to computational graph
    computation_node_t* comp_node = add_node(circuit->graph, NODE_OPERATION, OP_QUANTUM, node);
    if (!comp_node) {
        free(type_ptr);
        free(node->parameters);
        free(node->qubit_indices);
        free(node);
        circuit->num_nodes--;
        return false;
    }
    
    return true;
}
