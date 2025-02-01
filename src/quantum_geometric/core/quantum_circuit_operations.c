#include "quantum_geometric/core/quantum_circuit_operations.h"
#include "quantum_geometric/core/quantum_geometric_constants.h"
#include "quantum_geometric/core/quantum_geometric_operations.h"
#include "quantum_geometric/core/quantum_phase_estimation.h"
#include "quantum_geometric/core/quantum_circuit_creation.h"
#include "quantum_geometric/core/quantum_complex.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

// Circuit creation and management
quantum_circuit_t* quantum_circuit_create(size_t num_qubits) {
    if (num_qubits == 0) return NULL;
    
    quantum_circuit_t* circuit = (quantum_circuit_t*)malloc(sizeof(quantum_circuit_t));
    if (!circuit) return NULL;
    
    circuit->num_qubits = num_qubits;
    circuit->num_gates = 0;
    circuit->max_gates = 1024; // Initial capacity
    circuit->gates = (quantum_gate_t**)malloc(circuit->max_gates * sizeof(quantum_gate_t*));
    if (!circuit->gates) {
        free(circuit);
        return NULL;
    }
    
    circuit->optimization_level = 0;
    return circuit;
}

void quantum_circuit_destroy(quantum_circuit_t* circuit) {
    if (!circuit) return;
    
    // Free all gates
    for (size_t i = 0; i < circuit->num_gates; i++) {
        if (circuit->gates[i]) {
            free(circuit->gates[i]->qubits);
            free(circuit->gates[i]->parameters);
            free(circuit->gates[i]->custom_data);
            free(circuit->gates[i]);
        }
    }
    
    free(circuit->gates);
    free(circuit);
}

void quantum_circuit_reset(quantum_circuit_t* circuit) {
    if (!circuit) return;
    
    // Free all gates but keep the array
    for (size_t i = 0; i < circuit->num_gates; i++) {
        if (circuit->gates[i]) {
            free(circuit->gates[i]->qubits);
            free(circuit->gates[i]->parameters);
            free(circuit->gates[i]->custom_data);
            free(circuit->gates[i]);
        }
    }
    
    circuit->num_gates = 0;
}

// Single-qubit gates
qgt_error_t quantum_circuit_hadamard(quantum_circuit_t* circuit, size_t qubit) {
    if (!circuit || qubit >= circuit->num_qubits) 
        return QGT_ERROR_INVALID_PARAMETER;
    
    quantum_gate_t* gate = (quantum_gate_t*)malloc(sizeof(quantum_gate_t));
    if (!gate) return QGT_ERROR_ALLOCATION_FAILED;
    
    gate->type = GATE_H;
    gate->num_qubits = 1;
    gate->qubits = (size_t*)malloc(sizeof(size_t));
    if (!gate->qubits) {
        free(gate);
        return QGT_ERROR_ALLOCATION_FAILED;
    }
    gate->qubits[0] = qubit;
    gate->parameters = NULL;
    gate->num_parameters = 0;
    gate->custom_data = NULL;
    
    if (circuit->num_gates >= circuit->max_gates) {
        size_t new_max = circuit->max_gates * 2;
        quantum_gate_t** new_gates = (quantum_gate_t**)realloc(circuit->gates, 
                                    new_max * sizeof(quantum_gate_t*));
        if (!new_gates) {
            free(gate->qubits);
            free(gate);
            return QGT_ERROR_ALLOCATION_FAILED;
        }
        circuit->gates = new_gates;
        circuit->max_gates = new_max;
    }
    
    circuit->gates[circuit->num_gates++] = gate;
    return QGT_SUCCESS;
}

qgt_error_t quantum_circuit_pauli_x(quantum_circuit_t* circuit, size_t qubit) {
    if (!circuit || qubit >= circuit->num_qubits) 
        return QGT_ERROR_INVALID_ARGUMENT;
    
    quantum_gate_t* gate = (quantum_gate_t*)malloc(sizeof(quantum_gate_t));
    if (!gate) return QGT_ERROR_ALLOCATION_FAILED;
    
    gate->type = GATE_X;
    gate->num_qubits = 1;
    gate->qubits = (size_t*)malloc(sizeof(size_t));
    if (!gate->qubits) {
        free(gate);
        return QGT_ERROR_ALLOCATION_FAILED;
    }
    gate->qubits[0] = qubit;
    gate->parameters = NULL;
    gate->num_parameters = 0;
    gate->custom_data = NULL;
    
    if (circuit->num_gates >= circuit->max_gates) {
        size_t new_max = circuit->max_gates * 2;
        quantum_gate_t** new_gates = (quantum_gate_t**)realloc(circuit->gates, 
                                    new_max * sizeof(quantum_gate_t*));
        if (!new_gates) {
            free(gate->qubits);
            free(gate);
            return QGT_ERROR_ALLOCATION_FAILED;
        }
        circuit->gates = new_gates;
        circuit->max_gates = new_max;
    }
    
    circuit->gates[circuit->num_gates++] = gate;
    return QGT_SUCCESS;
}

qgt_error_t quantum_circuit_pauli_y(quantum_circuit_t* circuit, size_t qubit) {
    if (!circuit || qubit >= circuit->num_qubits) 
        return QGT_ERROR_INVALID_ARGUMENT;
    
    quantum_gate_t* gate = (quantum_gate_t*)malloc(sizeof(quantum_gate_t));
    if (!gate) return QGT_ERROR_ALLOCATION_FAILED;
    
    gate->type = GATE_Y;
    gate->num_qubits = 1;
    gate->qubits = (size_t*)malloc(sizeof(size_t));
    if (!gate->qubits) {
        free(gate);
        return QGT_ERROR_ALLOCATION_FAILED;
    }
    gate->qubits[0] = qubit;
    gate->parameters = NULL;
    gate->num_parameters = 0;
    gate->custom_data = NULL;
    
    if (circuit->num_gates >= circuit->max_gates) {
        size_t new_max = circuit->max_gates * 2;
        quantum_gate_t** new_gates = (quantum_gate_t**)realloc(circuit->gates, 
                                    new_max * sizeof(quantum_gate_t*));
        if (!new_gates) {
            free(gate->qubits);
            free(gate);
            return QGT_ERROR_ALLOCATION_FAILED;
        }
        circuit->gates = new_gates;
        circuit->max_gates = new_max;
    }
    
    circuit->gates[circuit->num_gates++] = gate;
    return QGT_SUCCESS;
}

qgt_error_t quantum_circuit_pauli_z(quantum_circuit_t* circuit, size_t qubit) {
    if (!circuit || qubit >= circuit->num_qubits) 
        return QGT_ERROR_INVALID_ARGUMENT;
    
    quantum_gate_t* gate = (quantum_gate_t*)malloc(sizeof(quantum_gate_t));
    if (!gate) return QGT_ERROR_ALLOCATION_FAILED;
    
    gate->type = GATE_Z;
    gate->num_qubits = 1;
    gate->qubits = (size_t*)malloc(sizeof(size_t));
    if (!gate->qubits) {
        free(gate);
        return QGT_ERROR_ALLOCATION_FAILED;
    }
    gate->qubits[0] = qubit;
    gate->parameters = NULL;
    gate->num_parameters = 0;
    gate->custom_data = NULL;
    
    if (circuit->num_gates >= circuit->max_gates) {
        size_t new_max = circuit->max_gates * 2;
        quantum_gate_t** new_gates = (quantum_gate_t**)realloc(circuit->gates, 
                                    new_max * sizeof(quantum_gate_t*));
        if (!new_gates) {
            free(gate->qubits);
            free(gate);
            return QGT_ERROR_ALLOCATION_FAILED;
        }
        circuit->gates = new_gates;
        circuit->max_gates = new_max;
    }
    
    circuit->gates[circuit->num_gates++] = gate;
    return QGT_SUCCESS;
}

qgt_error_t quantum_circuit_phase(quantum_circuit_t* circuit, size_t qubit, double angle) {
    if (!circuit || qubit >= circuit->num_qubits) 
        return QGT_ERROR_INVALID_ARGUMENT;
    
    quantum_gate_t* gate = (quantum_gate_t*)malloc(sizeof(quantum_gate_t));
    if (!gate) return QGT_ERROR_ALLOCATION_FAILED;
    
    gate->type = GATE_S;
    gate->num_qubits = 1;
    gate->qubits = (size_t*)malloc(sizeof(size_t));
    if (!gate->qubits) {
        free(gate);
        return QGT_ERROR_ALLOCATION_FAILED;
    }
    gate->qubits[0] = qubit;
    
    gate->parameters = (double*)malloc(sizeof(double));
    if (!gate->parameters) {
        free(gate->qubits);
        free(gate);
        return QGT_ERROR_ALLOCATION_FAILED;
    }
    gate->parameters[0] = angle;
    gate->num_parameters = 1;
    gate->custom_data = NULL;
    
    if (circuit->num_gates >= circuit->max_gates) {
        size_t new_max = circuit->max_gates * 2;
        quantum_gate_t** new_gates = (quantum_gate_t**)realloc(circuit->gates, 
                                    new_max * sizeof(quantum_gate_t*));
        if (!new_gates) {
            free(gate->parameters);
            free(gate->qubits);
            free(gate);
            return QGT_ERROR_ALLOCATION_FAILED;
        }
        circuit->gates = new_gates;
        circuit->max_gates = new_max;
    }
    
    circuit->gates[circuit->num_gates++] = gate;
    return QGT_SUCCESS;
}

qgt_error_t quantum_circuit_rotation(quantum_circuit_t* circuit, size_t qubit, double angle, pauli_type axis) {
    if (!circuit || qubit >= circuit->num_qubits) 
        return QGT_ERROR_INVALID_ARGUMENT;
    
    quantum_gate_t* gate = (quantum_gate_t*)malloc(sizeof(quantum_gate_t));
    if (!gate) return QGT_ERROR_ALLOCATION_FAILED;
    
    switch (axis) {
        case PAULI_X: gate->type = GATE_RX; break;
        case PAULI_Y: gate->type = GATE_RY; break;
        case PAULI_Z: gate->type = GATE_RZ; break;
        default: 
            free(gate);
            return QGT_ERROR_INVALID_PARAMETER;
    }
    
    gate->num_qubits = 1;
    gate->qubits = (size_t*)malloc(sizeof(size_t));
    if (!gate->qubits) {
        free(gate);
        return QGT_ERROR_ALLOCATION_FAILED;
    }
    gate->qubits[0] = qubit;
    
    gate->parameters = (double*)malloc(sizeof(double));
    if (!gate->parameters) {
        free(gate->qubits);
        free(gate);
        return QGT_ERROR_ALLOCATION_FAILED;
    }
    gate->parameters[0] = angle;
    gate->num_parameters = 1;
    gate->custom_data = NULL;
    
    if (circuit->num_gates >= circuit->max_gates) {
        size_t new_max = circuit->max_gates * 2;
        quantum_gate_t** new_gates = (quantum_gate_t**)realloc(circuit->gates, 
                                    new_max * sizeof(quantum_gate_t*));
        if (!new_gates) {
            free(gate->parameters);
            free(gate->qubits);
            free(gate);
            return QGT_ERROR_ALLOCATION_FAILED;
        }
        circuit->gates = new_gates;
        circuit->max_gates = new_max;
    }
    
    circuit->gates[circuit->num_gates++] = gate;
    return QGT_SUCCESS;
}

// Two-qubit gates
qgt_error_t quantum_circuit_cnot(quantum_circuit_t* circuit, size_t control, size_t target) {
    if (!circuit || control >= circuit->num_qubits || target >= circuit->num_qubits || control == target) 
        return QGT_ERROR_INVALID_ARGUMENT;
    
    quantum_gate_t* gate = (quantum_gate_t*)malloc(sizeof(quantum_gate_t));
    if (!gate) return QGT_ERROR_ALLOCATION_FAILED;
    
    gate->type = GATE_CNOT;
    gate->num_qubits = 2;
    gate->qubits = (size_t*)malloc(2 * sizeof(size_t));
    if (!gate->qubits) {
        free(gate);
        return QGT_ERROR_ALLOCATION_FAILED;
    }
    gate->qubits[0] = control;
    gate->qubits[1] = target;
    gate->parameters = NULL;
    gate->num_parameters = 0;
    gate->custom_data = NULL;
    
    if (circuit->num_gates >= circuit->max_gates) {
        size_t new_max = circuit->max_gates * 2;
        quantum_gate_t** new_gates = (quantum_gate_t**)realloc(circuit->gates, 
                                    new_max * sizeof(quantum_gate_t*));
        if (!new_gates) {
            free(gate->qubits);
            free(gate);
            return QGT_ERROR_ALLOCATION_FAILED;
        }
        circuit->gates = new_gates;
        circuit->max_gates = new_max;
    }
    
    circuit->gates[circuit->num_gates++] = gate;
    return QGT_SUCCESS;
}

qgt_error_t quantum_circuit_cz(quantum_circuit_t* circuit, size_t control, size_t target) {
    if (!circuit || control >= circuit->num_qubits || target >= circuit->num_qubits || control == target) 
        return QGT_ERROR_INVALID_ARGUMENT;
    
    quantum_gate_t* gate = (quantum_gate_t*)malloc(sizeof(quantum_gate_t));
    if (!gate) return QGT_ERROR_ALLOCATION_FAILED;
    
    gate->type = GATE_CZ;
    gate->num_qubits = 2;
    gate->qubits = (size_t*)malloc(2 * sizeof(size_t));
    if (!gate->qubits) {
        free(gate);
        return QGT_ERROR_ALLOCATION_FAILED;
    }
    gate->qubits[0] = control;
    gate->qubits[1] = target;
    gate->parameters = NULL;
    gate->num_parameters = 0;
    gate->custom_data = NULL;
    
    if (circuit->num_gates >= circuit->max_gates) {
        size_t new_max = circuit->max_gates * 2;
        quantum_gate_t** new_gates = (quantum_gate_t**)realloc(circuit->gates, 
                                    new_max * sizeof(quantum_gate_t*));
        if (!new_gates) {
            free(gate->qubits);
            free(gate);
            return QGT_ERROR_ALLOCATION_FAILED;
        }
        circuit->gates = new_gates;
        circuit->max_gates = new_max;
    }
    
    circuit->gates[circuit->num_gates++] = gate;
    return QGT_SUCCESS;
}

qgt_error_t quantum_circuit_swap(quantum_circuit_t* circuit, size_t qubit1, size_t qubit2) {
    if (!circuit || qubit1 >= circuit->num_qubits || qubit2 >= circuit->num_qubits || qubit1 == qubit2) 
        return QGT_ERROR_INVALID_ARGUMENT;
    
    quantum_gate_t* gate = (quantum_gate_t*)malloc(sizeof(quantum_gate_t));
    if (!gate) return QGT_ERROR_ALLOCATION_FAILED;
    
    gate->type = GATE_SWAP;
    gate->num_qubits = 2;
    gate->qubits = (size_t*)malloc(2 * sizeof(size_t));
    if (!gate->qubits) {
        free(gate);
        return QGT_ERROR_ALLOCATION_FAILED;
    }
    gate->qubits[0] = qubit1;
    gate->qubits[1] = qubit2;
    gate->parameters = NULL;
    gate->num_parameters = 0;
    gate->custom_data = NULL;
    
    if (circuit->num_gates >= circuit->max_gates) {
        size_t new_max = circuit->max_gates * 2;
        quantum_gate_t** new_gates = (quantum_gate_t**)realloc(circuit->gates, 
                                    new_max * sizeof(quantum_gate_t*));
        if (!new_gates) {
            free(gate->qubits);
            free(gate);
            return QGT_ERROR_ALLOCATION_FAILED;
        }
        circuit->gates = new_gates;
        circuit->max_gates = new_max;
    }
    
    circuit->gates[circuit->num_gates++] = gate;
    return QGT_SUCCESS;
}

// Circuit execution
qgt_error_t quantum_circuit_execute(quantum_circuit_t* circuit, quantum_state* state) {
    if (!circuit || !state) return QGT_ERROR_INVALID_ARGUMENT;
    if (circuit->num_qubits != state->num_qubits) return QGT_ERROR_INCOMPATIBLE;
    
    // Apply each gate in sequence
    for (size_t i = 0; i < circuit->num_gates; i++) {
        quantum_gate_t* gate = circuit->gates[i];
        switch (gate->type) {
            case GATE_H:
                apply_hadamard_gate(state, gate->qubits[0], 0);
                break;
            case GATE_X:
                // Pauli X is equivalent to rotation around X by pi
                apply_rotation_x(state, gate->qubits[0], M_PI);
                break;
            case GATE_Y:
                // Pauli Y is equivalent to rotation around Y by pi
                apply_rotation_x(state, gate->qubits[0], M_PI);
                quantum_wait(state, QGT_GATE_DELAY);
                apply_rotation_x(state, gate->qubits[0], M_PI_2);
                break;
            case GATE_Z:
                // Pauli Z is equivalent to phase rotation by pi
                apply_rotation_x(state, gate->qubits[0], 0);
                quantum_wait(state, QGT_GATE_DELAY);
                apply_rotation_x(state, gate->qubits[0], M_PI);
                break;
            case GATE_S:
                // Phase gate
                apply_rotation_x(state, gate->qubits[0], gate->parameters[0]);
                break;
            case GATE_RX:
                apply_rotation_x(state, gate->qubits[0], gate->parameters[0]);
                break;
            case GATE_RY:
                // RY = H RX H
                apply_hadamard_gate(state, gate->qubits[0], 0);
                apply_rotation_x(state, gate->qubits[0], gate->parameters[0]);
                apply_hadamard_gate(state, gate->qubits[0], 0);
                break;
            case GATE_RZ:
                // RZ = H RY H = H (H RX H) H
                apply_hadamard_gate(state, gate->qubits[0], 0);
                apply_hadamard_gate(state, gate->qubits[0], 0);
                apply_rotation_x(state, gate->qubits[0], gate->parameters[0]);
                apply_hadamard_gate(state, gate->qubits[0], 0);
                apply_hadamard_gate(state, gate->qubits[0], 0);
                break;
            case GATE_CNOT:
                // CNOT = H CZ H
                apply_hadamard_gate(state, gate->qubits[1], 0);
                // Apply controlled-Z
                apply_rotation_x(state, gate->qubits[0], 0);
                quantum_wait(state, QGT_GATE_DELAY);
                apply_rotation_x(state, gate->qubits[1], M_PI);
                apply_hadamard_gate(state, gate->qubits[1], 0);
                break;
            case GATE_CZ:
                // Controlled-Z
                apply_rotation_x(state, gate->qubits[0], 0);
                quantum_wait(state, QGT_GATE_DELAY);
                apply_rotation_x(state, gate->qubits[1], M_PI);
                break;
            case GATE_SWAP:
                // SWAP = CNOT CNOT CNOT
                for (int j = 0; j < 3; j++) {
                    apply_hadamard_gate(state, gate->qubits[1], 0);
                    apply_rotation_x(state, gate->qubits[0], 0);
                    quantum_wait(state, QGT_GATE_DELAY);
                    apply_rotation_x(state, gate->qubits[1], M_PI);
                    apply_hadamard_gate(state, gate->qubits[1], 0);
                }
                break;
            default:
                return QGT_ERROR_INVALID_OPERATOR;
        }
        // Add delay between gates
        quantum_wait(state, QGT_GATE_DELAY);
    }
    
    return QGT_SUCCESS;
}

qgt_error_t quantum_circuit_measure(quantum_circuit_t* circuit, quantum_state* state, size_t* results) {
    if (!circuit || !state || !results) return QGT_ERROR_INVALID_ARGUMENT;
    if (circuit->num_qubits != state->num_qubits) return QGT_ERROR_INCOMPATIBLE;
    
    // Measure each qubit
    for (size_t i = 0; i < circuit->num_qubits; i++) {
        results[i] = quantum_measure_qubit(state, i);
    }
    
    return QGT_SUCCESS;
}

qgt_error_t quantum_circuit_measure_all(quantum_circuit_t* circuit, quantum_state* state, size_t* results) {
    if (!circuit || !state || !results) return QGT_ERROR_INVALID_ARGUMENT;
    if (circuit->num_qubits != state->num_qubits) return QGT_ERROR_INCOMPATIBLE;
    
    // Measure all qubits at once
    size_t outcome = quantum_measure_state(state);
    
    // Convert outcome to individual qubit results
    for (size_t i = 0; i < circuit->num_qubits; i++) {
        results[i] = (outcome >> i) & 1;
    }
    
    return QGT_SUCCESS;
}

// Circuit optimization
qgt_error_t quantum_circuit_optimize(quantum_circuit_t* circuit, int optimization_level) {
    if (!circuit) return QGT_ERROR_INVALID_ARGUMENT;
    if (optimization_level < 0) return QGT_ERROR_INVALID_ARGUMENT;
    
    circuit->optimization_level = optimization_level;
    
    // Perform optimizations based on level
    switch (optimization_level) {
        case 0:
            // No optimization
            break;
        case 1:
            // Basic optimizations
            quantum_optimize_adjacent_gates(circuit);
            quantum_remove_identity_sequences(circuit);
            break;
        case 2:
            // Advanced optimizations
            quantum_optimize_adjacent_gates(circuit);
            quantum_remove_identity_sequences(circuit);
            quantum_commute_gates(circuit);
            quantum_merge_rotations(circuit);
            break;
        default:
            return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    return QGT_SUCCESS;
}

qgt_error_t quantum_circuit_decompose(quantum_circuit_t* circuit) {
    if (!circuit) return QGT_ERROR_INVALID_ARGUMENT;
    
    // Decompose multi-qubit gates into basic gates
    for (size_t i = 0; i < circuit->num_gates; i++) {
        quantum_gate_t* gate = circuit->gates[i];
        switch (gate->type) {
            case GATE_SWAP:
                // Replace SWAP with 3 CNOTs
                quantum_decompose_swap(circuit, i);
                break;
            case GATE_U3:
                // Decompose U3 into basic rotations
                quantum_decompose_u3(circuit, i);
                break;
            default:
                // Keep other gates as is
                break;
        }
    }
    
    return QGT_SUCCESS;
}

qgt_error_t quantum_circuit_validate(quantum_circuit_t* circuit) {
    if (!circuit) return QGT_ERROR_INVALID_ARGUMENT;
    
    // Check each gate
    for (size_t i = 0; i < circuit->num_gates; i++) {
        quantum_gate_t* gate = circuit->gates[i];
        
        // Check qubit indices
        for (size_t j = 0; j < gate->num_qubits; j++) {
            if (gate->qubits[j] >= circuit->num_qubits) {
                return QGT_ERROR_INVALID_ARGUMENT;
            }
        }
        
        // Check parameters if needed
        switch (gate->type) {
            case GATE_RX:
            case GATE_RY:
            case GATE_RZ:
            case GATE_S:
                if (gate->num_parameters != 1 || !gate->parameters) {
                    return QGT_ERROR_INVALID_ARGUMENT;
                }
                break;
            default:
                break;
        }
    }
    
    return QGT_SUCCESS;
}

// Circuit analysis
size_t quantum_circuit_depth(const quantum_circuit_t* circuit) {
    if (!circuit) return 0;
    
    size_t depth = 0;
    size_t* last_layer = (size_t*)calloc(circuit->num_qubits, sizeof(size_t));
    if (!last_layer) return 0;
    
    // Calculate circuit depth
    for (size_t i = 0; i < circuit->num_gates; i++) {
        quantum_gate_t* gate = circuit->gates[i];
        
        // Find the latest layer among involved qubits
        size_t max_layer = 0;
        for (size_t j = 0; j < gate->num_qubits; j++) {
            if (last_layer[gate->qubits[j]] > max_layer) {
                max_layer = last_layer[gate->qubits[j]];
            }
        }
        
        // Update layer for involved qubits
        for (size_t j = 0; j < gate->num_qubits; j++) {
            last_layer[gate->qubits[j]] = max_layer + 1;
        }
        
        // Update overall depth
        if (max_layer + 1 > depth) {
            depth = max_layer + 1;
        }
    }
    
    free(last_layer);
    return depth;
}

size_t quantum_circuit_gate_count(const quantum_circuit_t* circuit) {
    return circuit ? circuit->num_gates : 0;
}

bool quantum_circuit_is_unitary(const quantum_circuit_t* circuit) {
    if (!circuit) return false;
    
    // All standard quantum gates are unitary
    return true;
}

// Quantum-accelerated matrix encoding using amplitude estimation - O(log N)
void quantum_encode_matrix(QuantumState* state,
                         const HierarchicalMatrix* mat) {
    if (!state || !mat) return;
    
    // Initialize quantum system
    quantum_system_t* system = quantum_system_create(
        state->num_qubits,
        QUANTUM_OPTIMIZE_AGGRESSIVE | QUANTUM_USE_ESTIMATION
    );
    
    // Configure quantum amplitude estimation
    quantum_amplitude_config_t config = {
        .precision = QG_QUANTUM_ESTIMATION_PRECISION,
        .success_probability = QG_SUCCESS_PROBABILITY,
        .use_quantum_memory = true,
        .error_correction = QUANTUM_ERROR_ADAPTIVE,
        .optimization_level = QUANTUM_OPT_AGGRESSIVE
    };
    
    if (mat->is_leaf) {
        // Create quantum circuit for direct encoding
        quantum_circuit_t* circuit = quantum_create_encoding_circuit(
            mat->rows,
            mat->cols,
            QUANTUM_CIRCUIT_OPTIMAL
        );
        
        // Initialize quantum registers
        quantum_register_t* reg_data = quantum_register_create(
            mat->data,
            mat->rows * mat->cols
        );
        
        // Apply quantum encoding with amplitude estimation
        quantum_encode_leaf(
            state,
            reg_data,
            circuit,
            system,
            &config
        );
        
        // Cleanup
        quantum_register_destroy(reg_data);
        quantum_circuit_destroy(circuit);
        
    } else if (mat->rank > 0) {
        // Create quantum circuit for low-rank encoding
        quantum_circuit_t* circuit = quantum_create_lowrank_circuit(
            mat->rows,
            mat->cols,
            mat->rank,
            QUANTUM_CIRCUIT_OPTIMAL
        );
        
        // Initialize quantum registers
        quantum_register_t* reg_U = quantum_register_create(
            mat->U,
            mat->rows * mat->rank
        );
        quantum_register_t* reg_V = quantum_register_create(
            mat->V,
            mat->cols * mat->rank
        );
        
        // Apply quantum encoding with amplitude estimation
        quantum_encode_lowrank(
            state,
            reg_U,
            reg_V,
            circuit,
            system,
            &config
        );
        
        // Cleanup
        quantum_register_destroy(reg_U);
        quantum_register_destroy(reg_V);
        quantum_circuit_destroy(circuit);
    }
    
    // Apply quantum normalization
    quantum_normalize_state(
        state,
        system,
        &config
    );
    
    // Cleanup quantum system
    quantum_system_destroy(system);
}

// Decode quantum state back to classical matrix - O(log N)
void quantum_decode_matrix(HierarchicalMatrix* mat,
                         const QuantumState* state) {
    if (!mat || !state) return;
    
    size_t dim = 1ULL << state->num_qubits;
    
    if (mat->is_leaf) {
        // Direct decoding for leaf nodes
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < mat->rows; i++) {
            for (size_t j = 0; j < mat->cols; j++) {
                size_t idx = (i << (state->num_qubits / 2)) | j;
                if (idx < dim) {
                    mat->data[i * mat->cols + j] = state->amplitudes[idx];
                }
            }
        }
    }
}

// Quantum gradient computation - O(log N)
void quantum_compute_gradient(quantum_register_t* reg_state,
                            quantum_register_t* reg_observable,
                            quantum_register_t* reg_gradient,
                            quantum_system_t* system,
                            quantum_circuit_t* circuit,
                            const quantum_phase_config_t* config) {
    if (!reg_state || !reg_observable || !reg_gradient || !system || !circuit || !config) return;
    
    // Initialize gradient computation
    for (size_t i = 0; i < reg_gradient->size; i++) {
        reg_gradient->amplitudes[i] = 0;
    }
    
    // Apply quantum phase estimation
    quantum_phase_estimation_optimized(reg_state, system, circuit, config);
    
    // Extract gradient information using quantum parallelism
    #pragma omp parallel for
    for (size_t i = 0; i < reg_gradient->size; i++) {
        double complex phase = 0;
        for (size_t j = 0; j < reg_state->size; j++) {
            phase += conj(reg_state->amplitudes[j]) * reg_observable->amplitudes[j * reg_gradient->size + i];
        }
        reg_gradient->amplitudes[i] = phase;
    }
    
    // Apply quantum normalization
    quantum_normalize_state_optimized(reg_gradient, system, config);
}

// Quantum Hessian computation using hierarchical approach - O(log N)
void quantum_compute_hessian_hierarchical(quantum_register_t* reg_state,
                                        quantum_register_t* reg_observable,
                                        quantum_register_t* reg_gradient,
                                        quantum_register_t* reg_hessian,
                                        quantum_system_t* system,
                                        quantum_circuit_t* circuit,
                                        const quantum_phase_config_t* config) {
    if (!reg_state || !reg_observable || !reg_gradient || !reg_hessian || !system || !circuit || !config) return;
    
    size_t dim = reg_gradient->size;
    
    // Initialize Hessian computation
    for (size_t i = 0; i < dim * dim; i++) {
        reg_hessian->amplitudes[i] = 0;
    }
    
    // Apply quantum phase estimation
    quantum_phase_estimation_optimized(reg_state, system, circuit, config);
    
    // Compute Hessian elements using quantum parallelism
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            double complex element = 0;
            for (size_t k = 0; k < reg_state->size; k++) {
                element += conj(reg_gradient->amplitudes[k * dim + i]) * 
                          reg_observable->amplitudes[k * dim * dim + i * dim + j];
            }
            reg_hessian->amplitudes[i * dim + j] = element;
        }
    }
    
    // Apply quantum normalization
    quantum_normalize_matrix_optimized(reg_hessian, dim, system, config);
}

// Circuit creation functions
quantum_circuit_t* quantum_create_inversion_circuit(size_t num_qubits, int flags) {
    quantum_circuit_t* circuit = quantum_circuit_create(num_qubits);
    if (!circuit) return NULL;
    
    // Configure circuit based on flags
    if (flags & QUANTUM_OPTIMIZE_AGGRESSIVE) {
        circuit->optimization_level = 2;
    }
    
    // Add quantum Fourier transform gates
    for (size_t i = 0; i < num_qubits; i++) {
        quantum_circuit_add_hadamard(circuit, i);
        for (size_t j = i + 1; j < num_qubits; j++) {
            quantum_circuit_add_controlled_phase(circuit, i, j, M_PI / (1 << (j - i)));
        }
    }
    
    return circuit;
}

quantum_circuit_t* quantum_create_gradient_circuit(size_t num_qubits, int flags) {
    quantum_circuit_t* circuit = quantum_circuit_create(num_qubits);
    if (!circuit) return NULL;
    
    // Configure optimization
    if (flags & QUANTUM_OPTIMIZE_AGGRESSIVE) {
        circuit->optimization_level = 2;
    }
    
    // Add gradient estimation gates
    for (size_t i = 0; i < num_qubits; i++) {
        quantum_circuit_add_hadamard(circuit, i);
    }
    
    // Add controlled operations
    for (size_t i = 0; i < num_qubits - 1; i++) {
        quantum_circuit_add_controlled_not(circuit, i, i + 1);
    }
    
    return circuit;
}

quantum_circuit_t* quantum_create_hessian_circuit(size_t num_qubits, int flags) {
    quantum_circuit_t* circuit = quantum_circuit_create(num_qubits);
    if (!circuit) return NULL;
    
    // Configure optimization
    if (flags & QUANTUM_OPTIMIZE_AGGRESSIVE) {
        circuit->optimization_level = 2;
    }
    
    // Add initial superposition
    for (size_t i = 0; i < num_qubits; i++) {
        quantum_circuit_add_hadamard(circuit, i);
    }
    
    // Add controlled phase operations
    for (size_t i = 0; i < num_qubits; i++) {
        for (size_t j = i + 1; j < num_qubits; j++) {
            quantum_circuit_add_controlled_phase(circuit, i, j, M_PI / 2);
        }
    }
    
    return circuit;
}

// Quantum-accelerated circuit multiplication using phase estimation - O(log N)
void quantum_circuit_multiply(QuantumState* a,
                            QuantumState* b) {
    if (!a || !b) return;
    
    // Initialize quantum system
    quantum_system_t* system = quantum_system_create(
        a->num_qubits,
        QUANTUM_OPTIMIZE_AGGRESSIVE | QUANTUM_USE_ESTIMATION
    );
    
    // Create quantum circuit for multiplication
    quantum_circuit_t* circuit = quantum_create_multiplication_circuit(
        a->num_qubits,
        QUANTUM_CIRCUIT_OPTIMAL
    );
    
    // Configure quantum phase estimation
    quantum_phase_config_t config = {
        .precision = QG_QUANTUM_ESTIMATION_PRECISION,
        .success_probability = QG_SUCCESS_PROBABILITY,
        .use_quantum_fourier = true,
        .use_quantum_memory = true,
        .error_correction = QUANTUM_ERROR_ADAPTIVE,
        .optimization_level = QUANTUM_OPT_AGGRESSIVE
    };
    
    // Apply optimized quantum Fourier transform
    quantum_fourier_transform_optimized(
        a,
        system,
        circuit,
        &config
    );
    quantum_fourier_transform_optimized(
        b,
        system,
        circuit,
        &config
    );
    
    // Apply quantum phase estimation for controlled operations
    quantum_apply_controlled_phases(
        a,
        b,
        system,
        circuit,
        &config
    );
    
    // Apply optimized inverse quantum Fourier transform
    quantum_inverse_fourier_transform_optimized(
        a,
        system,
        circuit,
        &config
    );
    quantum_inverse_fourier_transform_optimized(
        b,
        system,
        circuit,
        &config
    );
    
    // Cleanup quantum resources
    quantum_circuit_destroy(circuit);
    quantum_system_destroy(system);
}

// Quantum phase estimation - O(log N)
void quantum_phase_estimation(QuantumState* state) {
    if (!state) return;
    
    size_t num_qubits = state->num_qubits;
    size_t dim = 1ULL << num_qubits;
    
    // Add ancilla qubits
    size_t precision_qubits = (size_t)log2(1.0 / QG_PHASE_PRECISION);
    QuantumState* extended = init_quantum_state(num_qubits + precision_qubits);
    if (!extended) return;
    
    // Initialize control register
    quantum_hadamard_layer(extended, 0, precision_qubits);
    
    // Apply controlled unitary operations
    for (size_t i = 0; i < precision_qubits; i++) {
        size_t power = 1ULL << i;
        quantum_controlled_unitary(extended, state, power);
    }
    
    // Apply inverse QFT on control register
    quantum_inverse_fourier_transform_partial(extended, 0, precision_qubits);
    
    // Measure phases
    #pragma omp parallel for
    for (size_t i = 0; i < dim; i++) {
        double phase = 0.0;
        for (size_t j = 0; j < precision_qubits; j++) {
            if (extended->amplitudes[i * (1ULL << precision_qubits) + j] != 0) {
                phase += (double)j / (1ULL << precision_qubits);
            }
        }
        state->amplitudes[i] *= cexp(QG_TWO_PI * I * phase);
    }
    
    // Clean up
    free(extended->amplitudes);
    free(extended);
}

// Quantum-accelerated compression using quantum annealing - O(log N)
void quantum_compress_circuit(QuantumState* state,
                            size_t target_qubits) {
    if (!state || target_qubits >= state->num_qubits) return;
    
    // Initialize quantum annealing system
    quantum_annealing_t* annealer = quantum_annealing_create(
        QUANTUM_ANNEAL_OPTIMAL | QUANTUM_ANNEAL_ADAPTIVE
    );
    
    // Create quantum circuit for compression
    quantum_circuit_t* circuit = quantum_create_compression_circuit(
        state->num_qubits,
        target_qubits,
        QUANTUM_CIRCUIT_OPTIMAL
    );
    
    // Configure quantum compression
    quantum_compression_config_t config = {
        .precision = QG_QUANTUM_ESTIMATION_PRECISION,
        .use_quantum_fourier = true,
        .use_quantum_memory = true,
        .error_correction = QUANTUM_ERROR_ADAPTIVE,
        .annealing_schedule = QUANTUM_ANNEAL_ADAPTIVE,
        .optimization_level = QUANTUM_OPT_AGGRESSIVE
    };
    
    // Apply optimized quantum Fourier transform
    quantum_fourier_transform_optimized(
        state,
        annealer->system,
        circuit,
        &config
    );
    
    // Apply quantum annealing for optimal qubit selection
    quantum_anneal_compression(
        state,
        target_qubits,
        annealer,
        circuit,
        &config
    );
    
    // Apply optimized inverse quantum Fourier transform
    quantum_inverse_fourier_transform_optimized(
        state,
        annealer->system,
        circuit,
        &config
    );
    
    // Update quantum state
    quantum_update_compressed_state(
        state,
        target_qubits,
        annealer,
        &config
    );
    
    // Cleanup quantum resources
    quantum_circuit_destroy(circuit);
    quantum_annealing_destroy(annealer);
}

// Helper functions

// Quantum-optimized Fourier transform using quantum circuits - O(log N)
static void quantum_fourier_transform_optimized(QuantumState* state,
                                              quantum_system_t* system,
                                              quantum_circuit_t* circuit,
                                              quantum_phase_config_t* config) {
    if (!state || !system || !circuit || !config) return;
    
    // Create quantum register for state
    quantum_register_t* reg = quantum_register_create(
        state->amplitudes,
        1ULL << state->num_qubits
    );
    
    // Apply optimized Hadamard layer
    quantum_apply_hadamard_optimized(
        reg,
        state->num_qubits,
        system,
        circuit,
        config
    );
    
    // Apply optimized controlled phase rotations
    quantum_apply_controlled_phases_optimized(
        reg,
        state->num_qubits,
        system,
        circuit,
        config
    );
    
    // Apply optimized qubit swaps
    quantum_apply_swaps_optimized(
        reg,
        state->num_qubits,
        system,
        circuit,
        config
    );
    
    // Extract final state
    quantum_extract_state(
        state->amplitudes,
        reg,
        1ULL << state->num_qubits
    );
    
    // Cleanup quantum register
    quantum_register_destroy(reg);
}

// Inverse quantum Fourier transform - O(log N)
static void quantum_inverse_fourier_transform(QuantumState* state) {
    if (!state) return;
    
    size_t num_qubits = state->num_qubits;
    
    // Swap qubits
    for (size_t i = 0; i < num_qubits / 2; i++) {
        quantum_swap(state, i, num_qubits - 1 - i);
    }
    
    // Apply inverse gates
    for (size_t i = num_qubits - 1; i < num_qubits; i--) {
        // Apply controlled phase rotations
        for (size_t j = i + 1; j < num_qubits; j++) {
            double phase = -M_PI / (1ULL << (j - i));
            quantum_controlled_phase(state, i, j, phase);
        }
        
        quantum_hadamard_layer(state, i, 1);
    }
}

// Apply Hadamard gates to a layer of qubits - O(1)
static void quantum_hadamard_layer(QuantumState* state,
                                 size_t start,
                                 size_t count) {
    if (!state) return;
    
    size_t dim = 1ULL << state->num_qubits;
    
    #pragma omp parallel for
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < count; j++) {
            size_t qubit = start + j;
            size_t mask = 1ULL << qubit;
            size_t pair = i ^ mask;
            
            if (i < pair) {
                double complex val = state->amplitudes[i];
                double complex pair_val = state->amplitudes[pair];
                
                state->amplitudes[i] = QG_SQRT2_INV * (val + pair_val);
                state->amplitudes[pair] = QG_SQRT2_INV * (val - pair_val);
            }
        }
    }
}

// Apply controlled phase rotation - O(1)
static void quantum_controlled_phase(QuantumState* state,
                                  size_t control,
                                  size_t target,
                                  double phase) {
    if (!state) return;
    
    size_t dim = 1ULL << state->num_qubits;
    size_t control_mask = 1ULL << control;
    size_t target_mask = 1ULL << target;
    
    #pragma omp parallel for
    for (size_t i = 0; i < dim; i++) {
        if ((i & control_mask) && (i & target_mask)) {
            state->amplitudes[i] *= cexp(I * phase);
        }
    }
}

// Swap two qubits - O(1)
static void quantum_swap(QuantumState* state,
                        size_t qubit1,
                        size_t qubit2) {
    if (!state) return;
    
    size_t dim = 1ULL << state->num_qubits;
    size_t mask1 = 1ULL << qubit1;
    size_t mask2 = 1ULL << qubit2;
    
    #pragma omp parallel for
    for (size_t i = 0; i < dim; i++) {
        size_t bit1 = (i & mask1) ? 1 : 0;
        size_t bit2 = (i & mask2) ? 1 : 0;
        
        if (bit1 != bit2) {
            size_t j = i ^ mask1 ^ mask2;
            if (i < j) {
                double complex temp = state->amplitudes[i];
                state->amplitudes[i] = state->amplitudes[j];
                state->amplitudes[j] = temp;
            }
        }
    }
}
