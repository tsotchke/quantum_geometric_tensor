/**
 * @file quantum_rigetti_backend_optimized.c
 * @brief Optimized Rigetti quantum backend implementation with pyQuil integration
 */

// Standard library headers first
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

// Include backend types directly to avoid circular/conflicting includes
#include "quantum_geometric/hardware/quantum_backend_types.h"
#include "quantum_geometric/hardware/hardware_capabilities.h"
#include "quantum_geometric/hardware/quantum_rigetti_api.h"
#include "quantum_geometric/hardware/quantum_rigetti_api.h"

// Forward declare RigettiConfig as struct (defined in quantum_backend_types.h)
typedef struct RigettiConfig RigettiConfig;

// Gate types matching quantum_circuit.h enumeration
// Defined here to avoid header conflicts
typedef enum {
    GATE_H = 0,      // Hadamard gate
    GATE_X = 1,      // Pauli X gate
    GATE_Y = 2,      // Pauli Y gate
    GATE_Z = 3,      // Pauli Z gate
    GATE_S = 4,      // S gate (phase gate)
    GATE_T = 5,      // T gate
    GATE_CNOT = 6,   // Controlled NOT gate
    GATE_CZ = 7,     // Controlled Z gate
    GATE_SWAP = 8,   // SWAP gate
    GATE_RX = 9,     // Rotation around X axis
    GATE_RY = 10,    // Rotation around Y axis
    GATE_RZ = 11,    // Rotation around Z axis
    GATE_U1 = 12,    // U1 gate (phase rotation)
    GATE_U2 = 13,    // U2 gate (sqrt of NOT)
    GATE_U3 = 14,    // U3 gate (general single qubit)
    GATE_CUSTOM = 15 // Custom gate
} gate_type_t;

// ============================================================================
// Internal Gate and Circuit Types for Rigetti Backend Optimization
// ============================================================================
// These internal types are used for efficient gate decomposition and circuit
// manipulation. They provide a flat, cache-friendly representation optimized
// for the native gate decomposition algorithms used in this backend.

// Maximum parameters and qubits per gate
#define MAX_GATE_QUBITS 4
#define MAX_GATE_PARAMS 4

// Internal gate representation for efficient manipulation
typedef struct quantum_gate {
    gate_type_t type;                    // Gate type from quantum_circuit.h
    size_t qubits[MAX_GATE_QUBITS];      // Target qubit indices
    size_t num_qubits;                   // Number of qubits involved
    double params[MAX_GATE_PARAMS];      // Gate parameters (rotation angles, etc.)
    size_t num_params;                   // Number of parameters
    bool cancelled;                      // Flag for gate cancellation optimization
} quantum_gate;

// Internal measurement representation
typedef struct quantum_measurement {
    size_t qubit_idx;                    // Qubit to measure
    size_t classical_bit;                // Classical bit to store result
    size_t optimized_order;              // Optimized measurement order
} quantum_measurement;

// Internal circuit representation for optimization
typedef struct quantum_circuit_internal {
    quantum_gate* gates;                 // Array of gates (flat, not pointers)
    size_t num_gates;                    // Current number of gates
    size_t capacity;                     // Allocated capacity
    size_t num_qubits;                   // Number of qubits
    quantum_measurement* measurements;   // Array of measurements
    size_t num_measurements;             // Number of measurements
    size_t measurements_capacity;        // Allocated measurement capacity
} quantum_circuit_internal;

// Alias for internal use in this file
typedef quantum_circuit_internal quantum_circuit;

// Internal result structure
typedef struct quantum_result {
    double* measurements;                // Measurement results
    size_t num_measurements;             // Number of measurements
    double* probabilities;               // Measurement probabilities
    size_t shots;                        // Number of shots
    void* backend_data;                  // Backend-specific data
} quantum_result;

// pyQuil program representation
typedef struct pyquil_program {
    char** instructions;                 // Quil instruction strings
    size_t num_instructions;             // Number of instructions
    size_t capacity;                     // Allocated capacity
} pyquil_program;

// Execution configuration
typedef struct execution_config {
    size_t shots;                        // Number of shots
    int optimization_level;              // Optimization level (0-3)
    bool error_mitigation;               // Enable error mitigation
} execution_config;

// ============================================================================
// Forward Declarations for pyQuil Functions
// ============================================================================
static bool add_pyquil_rx(pyquil_program* program, double theta, size_t qubit);
static bool add_pyquil_rz(pyquil_program* program, double theta, size_t qubit);
static bool add_pyquil_cz(pyquil_program* program, size_t q1, size_t q2);
static bool add_pyquil_measure(pyquil_program* program, size_t qubit, size_t classical_bit);
static pyquil_program* convert_to_pyquil(quantum_circuit* circuit);
static bool execute_pyquil_program(const char* backend_name, pyquil_program* program,
                                   quantum_result* result, const execution_config* config);
static void process_rigetti_results(quantum_result* result, double* readout_errors,
                                    size_t num_qubits);

// ============================================================================
// Forward Declarations for Optimization Functions
// ============================================================================
static bool decompose_to_native_gates(quantum_circuit* circuit);
static bool optimize_rigetti_gates(quantum_circuit* circuit);
static bool optimize_rigetti_routing(quantum_circuit* circuit, double** coupling_map,
                                     size_t num_qubits);
static bool optimize_rigetti_measurements(quantum_circuit* circuit, size_t* measurement_order,
                                          size_t num_qubits);
static void optimize_rigetti_measurement_order(size_t* measurement_order,
                                               double* readout_errors, size_t num_qubits);
static bool mitigate_rigetti_readout_errors(quantum_result* result, double* readout_errors,
                                            size_t num_qubits);
static bool mitigate_rigetti_measurement_errors(quantum_result* result, double* error_rates,
                                                size_t num_qubits);
static bool extrapolate_rigetti_zero_noise(quantum_result* result, double* error_rates,
                                           size_t num_qubits);
static bool get_rigetti_properties(const char* backend_name, size_t* num_qubits,
                                   double** calibration_data);
static double get_rigetti_coupling_strength(const char* backend_name, size_t q1, size_t q2);
static bool validate_rigetti_config(const struct RigettiConfig* config);

// ============================================================================
// Backend State and Configuration
// ============================================================================

// Internal state for Rigetti backend
typedef struct {
    struct RigettiConfig config;
    size_t num_qubits;
    size_t num_measurements;
    double* calibration_data;
    double* error_rates;
    double* readout_errors;
    bool* qubit_availability;
    size_t* measurement_order;
    double** coupling_map;
    bool initialized;
} RigettiState;

// Native gate types
typedef enum {
    NATIVE_RX,
    NATIVE_RZ,
    NATIVE_CZ,
    NATIVE_MEASURE
} NativeGateType;

// Native gate set
typedef struct {
    NativeGateType* types;
    size_t num_gates;
} gate_set;

// Connectivity graph
typedef struct {
    bool** connections;
    size_t num_qubits;
} connectivity_graph;

// Forward declarations
static bool initialize_backend(RigettiState* state,
                             const RigettiConfig* config);
static void cleanup_backend(RigettiState* state);
void cleanup_rigetti_backend(RigettiState* state);  // Public cleanup function
static bool optimize_circuit(RigettiState* state,
                           quantum_circuit* circuit);
static bool execute_optimized(RigettiState* state,
                            quantum_circuit* circuit,
                            quantum_result* result);
static bool apply_error_mitigation(RigettiState* state,
                                 quantum_result* result);

// Native gate operations
static gate_set* get_rigetti_native_gates() {
    gate_set* gates = calloc(1, sizeof(gate_set));
    gates->num_gates = 4;
    gates->types = calloc(gates->num_gates, sizeof(NativeGateType));
    
    gates->types[0] = NATIVE_RX;
    gates->types[1] = NATIVE_RZ;
    gates->types[2] = NATIVE_CZ;
    gates->types[3] = NATIVE_MEASURE;
    
    return gates;
}

static bool is_native_gate(const quantum_gate* gate, const gate_set* native_gates) {
    if (!gate || !native_gates) return false;

    // Check if gate type matches any native gate
    switch (gate->type) {
        case GATE_RX:
            return true;  // Native RX gate
            
        case GATE_RZ:
            return true;  // Native RZ gate
            
        case GATE_CZ:
            return true;  // Native CZ gate
            
        default:
            return false;
    }
}

// Allocate a new gate structure
static quantum_gate* create_new_gate(int gate_type, size_t qubit, double param) {
    quantum_gate* g = calloc(1, sizeof(quantum_gate));
    if (!g) return NULL;
    g->type = gate_type;
    g->qubits[0] = qubit;
    g->num_qubits = 1;
    g->params[0] = param;
    g->num_params = (param != 0.0) ? 1 : 0;
    g->cancelled = false;
    return g;
}

// Allocate a new two-qubit gate structure
static quantum_gate* create_new_two_qubit_gate(int gate_type, size_t ctrl, size_t tgt) {
    quantum_gate* g = calloc(1, sizeof(quantum_gate));
    if (!g) return NULL;
    g->type = gate_type;
    g->qubits[0] = ctrl;
    g->qubits[1] = tgt;
    g->num_qubits = 2;
    g->num_params = 0;
    g->cancelled = false;
    return g;
}

// Helper to insert a gate at a position in the circuit
// circuit->gates is a flat array of quantum_gate structs
static bool insert_gate_at(quantum_circuit* circuit, size_t position,
                           int gate_type, size_t qubit, double param) {
    if (!circuit || position > circuit->num_gates) return false;

    // Ensure capacity for the flat array
    if (circuit->num_gates >= circuit->capacity) {
        size_t new_capacity = circuit->capacity * 2;
        quantum_gate* new_gates = realloc(circuit->gates,
                                          new_capacity * sizeof(quantum_gate));
        if (!new_gates) return false;
        circuit->gates = new_gates;
        circuit->capacity = new_capacity;
    }

    // Shift gates after position (moving structs, not pointers)
    for (size_t i = circuit->num_gates; i > position; i--) {
        circuit->gates[i] = circuit->gates[i - 1];
    }

    // Initialize the new gate in place
    quantum_gate* new_gate = &circuit->gates[position];
    new_gate->type = (gate_type_t)gate_type;
    new_gate->qubits[0] = qubit;
    new_gate->num_qubits = 1;
    new_gate->params[0] = param;
    new_gate->num_params = (param != 0.0) ? 1 : 0;
    new_gate->cancelled = false;

    circuit->num_gates++;
    return true;
}

// Helper to insert a two-qubit gate at a position
static bool insert_two_qubit_gate_at(quantum_circuit* circuit, size_t position,
                                      int gate_type, size_t control, size_t target) {
    if (!circuit || position > circuit->num_gates) return false;

    // Ensure capacity
    if (circuit->num_gates >= circuit->capacity) {
        size_t new_capacity = circuit->capacity * 2;
        quantum_gate* new_gates = realloc(circuit->gates,
                                          new_capacity * sizeof(quantum_gate));
        if (!new_gates) return false;
        circuit->gates = new_gates;
        circuit->capacity = new_capacity;
    }

    // Shift gates after position
    for (size_t i = circuit->num_gates; i > position; i--) {
        circuit->gates[i] = circuit->gates[i - 1];
    }

    // Initialize the new gate in place
    quantum_gate* new_gate = &circuit->gates[position];
    new_gate->type = (gate_type_t)gate_type;
    new_gate->qubits[0] = control;
    new_gate->qubits[1] = target;
    new_gate->num_qubits = 2;
    new_gate->num_params = 0;
    new_gate->cancelled = false;

    circuit->num_gates++;
    return true;
}

// Decompose a single gate - may modify circuit for multi-gate decompositions
// Returns the number of gates to skip (0 for single replacement, >0 for expansion)
static size_t decompose_gate_in_circuit(quantum_circuit* circuit, size_t gate_idx,
                                        const gate_set* native_gates) {
    if (!circuit || gate_idx >= circuit->num_gates || !native_gates) return 0;

    // Get pointer to gate at index (gates is a flat array)
    quantum_gate* gate = &circuit->gates[gate_idx];

    // Decompose non-native gates into native gate sequences
    switch (gate->type) {
        case GATE_X:
            // X = RX(π)
            gate->type = GATE_RX;
            gate->params[0] = M_PI;
            gate->num_params = 1;
            return 0;

        case GATE_Y: {
            // Y = RZ(-π/2) RX(π) RZ(π/2)
            size_t qubit = gate->qubits[0];

            // Convert current gate to first RZ(-π/2)
            gate->type = GATE_RZ;
            gate->params[0] = -M_PI / 2.0;
            gate->num_params = 1;

            // Insert RX(π) and RZ(π/2) after
            if (!insert_gate_at(circuit, gate_idx + 1, GATE_RX, qubit, M_PI)) return 0;
            if (!insert_gate_at(circuit, gate_idx + 2, GATE_RZ, qubit, M_PI / 2.0)) return 0;

            return 2;
        }

        case GATE_Z:
            // Z = RZ(π)
            gate->type = GATE_RZ;
            gate->params[0] = M_PI;
            gate->num_params = 1;
            return 0;

        case GATE_H: {
            // H = RZ(π/2) RX(π/2) RZ(π/2)
            size_t qubit = gate->qubits[0];

            gate->type = GATE_RZ;
            gate->params[0] = M_PI / 2.0;
            gate->num_params = 1;

            if (!insert_gate_at(circuit, gate_idx + 1, GATE_RX, qubit, M_PI / 2.0)) return 0;
            if (!insert_gate_at(circuit, gate_idx + 2, GATE_RZ, qubit, M_PI / 2.0)) return 0;

            return 2;
        }

        case GATE_S:
            gate->type = GATE_RZ;
            gate->params[0] = M_PI / 2.0;
            gate->num_params = 1;
            return 0;

        case GATE_T:
            gate->type = GATE_RZ;
            gate->params[0] = M_PI / 4.0;
            gate->num_params = 1;
            return 0;

        case GATE_CNOT: {
            // CNOT = RX(π/2)_target CZ RX(-π/2)_target
            size_t control = gate->qubits[0];
            size_t target = gate->qubits[1];

            // Convert to RX(π/2) on target
            gate->type = GATE_RX;
            gate->qubits[0] = target;
            gate->num_qubits = 1;
            gate->params[0] = M_PI / 2.0;
            gate->num_params = 1;

            if (!insert_two_qubit_gate_at(circuit, gate_idx + 1, GATE_CZ, control, target)) return 0;
            if (!insert_gate_at(circuit, gate_idx + 2, GATE_RX, target, -M_PI / 2.0)) return 0;

            return 2;
        }

        case GATE_SWAP: {
            // SWAP = 3 CNOTs = 9 native gates
            size_t q0 = gate->qubits[0];
            size_t q1 = gate->qubits[1];

            // Convert first gate
            gate->type = GATE_RX;
            gate->qubits[0] = q1;
            gate->num_qubits = 1;
            gate->params[0] = M_PI / 2.0;
            gate->num_params = 1;

            size_t pos = gate_idx + 1;

            // CNOT 1
            if (!insert_two_qubit_gate_at(circuit, pos++, GATE_CZ, q0, q1)) return 0;
            if (!insert_gate_at(circuit, pos++, GATE_RX, q1, -M_PI / 2.0)) return 0;

            // CNOT 2
            if (!insert_gate_at(circuit, pos++, GATE_RX, q0, M_PI / 2.0)) return 0;
            if (!insert_two_qubit_gate_at(circuit, pos++, GATE_CZ, q1, q0)) return 0;
            if (!insert_gate_at(circuit, pos++, GATE_RX, q0, -M_PI / 2.0)) return 0;

            // CNOT 3
            if (!insert_gate_at(circuit, pos++, GATE_RX, q1, M_PI / 2.0)) return 0;
            if (!insert_two_qubit_gate_at(circuit, pos++, GATE_CZ, q0, q1)) return 0;
            if (!insert_gate_at(circuit, pos++, GATE_RX, q1, -M_PI / 2.0)) return 0;

            return 8;
        }

        case GATE_RY: {
            // RY(θ) = RZ(-π/2) RX(θ) RZ(π/2)
            size_t qubit = gate->qubits[0];
            double theta = gate->params[0];

            gate->type = GATE_RZ;
            gate->params[0] = -M_PI / 2.0;

            if (!insert_gate_at(circuit, gate_idx + 1, GATE_RX, qubit, theta)) return 0;
            if (!insert_gate_at(circuit, gate_idx + 2, GATE_RZ, qubit, M_PI / 2.0)) return 0;

            return 2;
        }

        case GATE_RX:
        case GATE_RZ:
        case GATE_CZ:
            return 0;  // Already native

        default:
            return 0;
    }
}

// Legacy wrapper for backward compatibility
static bool decompose_gate(quantum_gate* gate, const gate_set* native_gates) {
    if (!gate || !native_gates) return false;

    switch (gate->type) {
        case GATE_X:
            gate->type = GATE_RX;
            gate->params[0] = M_PI;
            gate->num_params = 1;
            return true;

        case GATE_Z:
            gate->type = GATE_RZ;
            gate->params[0] = M_PI;
            gate->num_params = 1;
            return true;

        case GATE_S:
            gate->type = GATE_RZ;
            gate->params[0] = M_PI / 2.0;
            gate->num_params = 1;
            return true;

        case GATE_T:
            gate->type = GATE_RZ;
            gate->params[0] = M_PI / 4.0;
            gate->num_params = 1;
            return true;

        case GATE_RX:
        case GATE_RZ:
        case GATE_CZ:
            return true;

        default:
            // Multi-gate decompositions need decompose_gate_in_circuit
            return false;
    }
}

static bool can_combine_rigetti_gates(const quantum_gate* g1,
                                    const quantum_gate* g2) {
    if (!g1 || !g2) return false;

    // Check if gates can be combined
    if (g1->type != g2->type) return false;
    if (g1->num_qubits != g2->num_qubits) return false;

    switch (g1->type) {
        case GATE_RX:
        case GATE_RZ:
            return true;  // Rotation gates can be combined
            
        default:
            return false;
    }
}

static void combine_rigetti_gates(quantum_gate* g1, const quantum_gate* g2) {
    if (!g1 || !g2) return;

    // Combine gate parameters
    switch (g1->type) {
        case GATE_RX:
        case GATE_RZ:
            g1->params[0] = fmod(g1->params[0] + g2->params[0], 2 * M_PI);
            break;
            
        default:
            break;
    }
}

static void compact_rigetti_circuit(quantum_circuit* circuit) {
    if (!circuit) return;

    // Remove cancelled gates
    size_t write = 0;
    for (size_t read = 0; read < circuit->num_gates; read++) {
        if (!circuit->gates[read].cancelled) {
            if (write != read) {
                circuit->gates[write] = circuit->gates[read];
            }
            write++;
        }
    }
    circuit->num_gates = write;
}

// Connectivity optimization
static connectivity_graph* build_rigetti_connectivity(double** coupling_map,
                                                    size_t num_qubits) {
    if (!coupling_map) return NULL;

    // Allocate connectivity graph
    connectivity_graph* graph = calloc(1, sizeof(connectivity_graph));
    graph->num_qubits = num_qubits;
    graph->connections = calloc(num_qubits, sizeof(bool*));
    for (size_t i = 0; i < num_qubits; i++) {
        graph->connections[i] = calloc(num_qubits, sizeof(bool));
    }

    // Build connectivity from coupling map
    for (size_t i = 0; i < num_qubits; i++) {
        for (size_t j = 0; j < num_qubits; j++) {
            graph->connections[i][j] = (coupling_map[i][j] > 0.0);
        }
    }

    return graph;
}

static size_t* find_optimal_rigetti_mapping(connectivity_graph* graph,
                                          quantum_circuit* circuit) {
    if (!graph || !circuit) return NULL;

    // Allocate mapping
    size_t* mapping = calloc(graph->num_qubits, sizeof(size_t));
    bool* used = calloc(graph->num_qubits, sizeof(bool));

    // Simple greedy mapping based on connectivity
    for (size_t i = 0; i < graph->num_qubits; i++) {
        // Find most connected available qubit
        size_t best_qubit = 0;
        size_t max_connections = 0;
        
        for (size_t q = 0; q < graph->num_qubits; q++) {
            if (used[q]) continue;
            
            size_t connections = 0;
            for (size_t j = 0; j < graph->num_qubits; j++) {
                if (graph->connections[q][j]) connections++;
            }
            
            if (connections > max_connections) {
                max_connections = connections;
                best_qubit = q;
            }
        }
        
        mapping[i] = best_qubit;
        used[best_qubit] = true;
    }

    free(used);
    return mapping;
}

static void remap_rigetti_qubits(quantum_circuit* circuit, size_t* mapping) {
    if (!circuit || !mapping) return;

    // Remap qubits in all gates
    for (size_t i = 0; i < circuit->num_gates; i++) {
        quantum_gate* gate = &circuit->gates[i];
        for (size_t j = 0; j < gate->num_qubits; j++) {
            gate->qubits[j] = mapping[gate->qubits[j]];
        }
    }

    // Remap qubits in measurements
    for (size_t i = 0; i < circuit->num_measurements; i++) {
        quantum_measurement* meas = &circuit->measurements[i];
        meas->qubit_idx = mapping[meas->qubit_idx];
    }
}

static void sort_rigetti_measurements(quantum_circuit* circuit) {
    if (!circuit) return;

    // Sort measurements by qubit index
    for (size_t i = 0; i < circuit->num_measurements; i++) {
        for (size_t j = i + 1; j < circuit->num_measurements; j++) {
            if (circuit->measurements[j].optimized_order <
                circuit->measurements[i].optimized_order) {
                // Swap measurements
                quantum_measurement temp = circuit->measurements[i];
                circuit->measurements[i] = circuit->measurements[j];
                circuit->measurements[j] = temp;
            }
        }
    }
}

// pyQuil integration
static pyquil_program* create_pyquil_program() {
    // Initialize pyQuil program
    pyquil_program* program = calloc(1, sizeof(pyquil_program));
    program->instructions = NULL;
    program->num_instructions = 0;
    return program;
}

static bool add_pyquil_gate(pyquil_program* program, const quantum_gate* gate) {
    if (!program || !gate) return false;

    // Convert quantum gate to pyQuil instruction
    switch (gate->type) {
        case GATE_RX:
            // RX(theta) gate
            return add_pyquil_rx(program, gate->params[0], gate->qubits[0]);
            
        case GATE_RZ:
            // RZ(theta) gate
            return add_pyquil_rz(program, gate->params[0], gate->qubits[0]);
            
        case GATE_CZ:
            // CZ gate
            return add_pyquil_cz(program, gate->qubits[0], gate->qubits[1]);
            
        default:
            return false;
    }
}

static bool add_pyquil_measurement(pyquil_program* program,
                                 const quantum_measurement* meas) {
    if (!program || !meas) return false;

    // Add measurement instruction
    return add_pyquil_measure(program, meas->qubit_idx, meas->classical_bit);
}

static void cleanup_pyquil_program(pyquil_program* program) {
    if (!program) return;

    // Free instructions
    if (program->instructions) {
        for (size_t i = 0; i < program->num_instructions; i++) {
            free(program->instructions[i]);
        }
        free(program->instructions);
    }

    free(program);
}

// Main interface functions
bool init_rigetti_backend(RigettiState* state, const RigettiConfig* config) {
    if (!state || !config || !validate_rigetti_config(config)) {
        return false;
    }

    // Initialize state
    memset(state, 0, sizeof(RigettiState));
    memcpy(&state->config, config, sizeof(RigettiConfig));

    // Get backend properties
    if (!get_rigetti_properties(config->backend_name,
                               &state->num_qubits,
                               &state->calibration_data)) {
        return false;
    }

    // Allocate arrays
    state->error_rates = calloc(state->num_qubits, sizeof(double));
    state->readout_errors = calloc(state->num_qubits, sizeof(double));
    state->qubit_availability = calloc(state->num_qubits, sizeof(bool));
    state->measurement_order = calloc(state->num_qubits, sizeof(size_t));
    
    state->coupling_map = calloc(state->num_qubits, sizeof(double*));
    for (size_t i = 0; i < state->num_qubits; i++) {
        state->coupling_map[i] = calloc(state->num_qubits, sizeof(double));
    }

    // Initialize backend
    if (!initialize_backend(state, config)) {
        cleanup_rigetti_backend(state);
        return false;
    }

    state->initialized = true;
    return true;
}

void cleanup_rigetti_backend(RigettiState* state) {
    if (state) {
        cleanup_backend(state);
        free(state->error_rates);
        free(state->readout_errors);
        free(state->qubit_availability);
        free(state->measurement_order);
        if (state->coupling_map) {
            for (size_t i = 0; i < state->num_qubits; i++) {
                free(state->coupling_map[i]);
            }
            free(state->coupling_map);
        }
        memset(state, 0, sizeof(RigettiState));
    }
}

// Renamed to avoid conflict with other backend implementations
bool execute_rigetti_circuit(RigettiState* state,
                             quantum_circuit* circuit,
                             quantum_result* result) {
    if (!state || !state->initialized || !circuit || !result) {
        return false;
    }

    // Optimize circuit for backend
    if (!optimize_circuit(state, circuit)) {
        return false;
    }

    // Execute optimized circuit
    if (!execute_optimized(state, circuit, result)) {
        return false;
    }

    // Apply error mitigation
    if (!apply_error_mitigation(state, result)) {
        return false;
    }

    return true;
}

static bool initialize_backend(RigettiState* state,
                             const RigettiConfig* config) {
    if (!state || !config) {
        return false;
    }

    // Update error rates from calibration data
    for (size_t i = 0; i < state->num_qubits; i++) {
        state->error_rates[i] = state->calibration_data[i * 3];
        state->readout_errors[i] = state->calibration_data[i * 3 + 1];
        state->qubit_availability[i] = (state->calibration_data[i * 3 + 2] > 0.5);
    }

    // Build coupling map
    for (size_t i = 0; i < state->num_qubits; i++) {
        for (size_t j = 0; j < state->num_qubits; j++) {
            if (i == j) continue;
            state->coupling_map[i][j] = get_rigetti_coupling_strength(
                config->backend_name, i, j);
        }
    }

    // Initialize measurement order
    for (size_t i = 0; i < state->num_qubits; i++) {
        state->measurement_order[i] = i;
    }

    // Optimize measurement order based on readout errors
    optimize_rigetti_measurement_order(state->measurement_order,
                                     state->readout_errors,
                                     state->num_qubits);

    return true;
}

static void cleanup_backend(RigettiState* state) {
    if (!state) return;

    // Securely zero the API key before freeing
    if (state->config.api_key) {
        size_t key_len = strlen(state->config.api_key);
        memset(state->config.api_key, 0, key_len);
        free(state->config.api_key);
        state->config.api_key = NULL;
    }

    // Free other config strings
    free(state->config.url);
    state->config.url = NULL;
    free(state->config.backend_name);
    state->config.backend_name = NULL;
    free(state->config.noise_model);
    state->config.noise_model = NULL;

    // Free calibration and error data
    free(state->calibration_data);
    state->calibration_data = NULL;
    free(state->error_rates);
    state->error_rates = NULL;
    free(state->readout_errors);
    state->readout_errors = NULL;
    free(state->qubit_availability);
    state->qubit_availability = NULL;
    free(state->measurement_order);
    state->measurement_order = NULL;

    // Free coupling map
    if (state->coupling_map) {
        for (size_t i = 0; i < state->num_qubits; i++) {
            free(state->coupling_map[i]);
        }
        free(state->coupling_map);
        state->coupling_map = NULL;
    }

    state->initialized = false;
}

static bool optimize_circuit(RigettiState* state,
                           quantum_circuit* circuit) {
    if (!state || !circuit) {
        return false;
    }

    // Apply circuit optimization techniques
    bool success = true;

    // 1. Native gate decomposition
    success &= decompose_to_native_gates(circuit);

    // 2. Gate cancellation and fusion
    success &= optimize_rigetti_gates(circuit);

    // 3. Qubit routing optimization
    success &= optimize_rigetti_routing(circuit,
                                      state->coupling_map,
                                      state->num_qubits);

    // 4. Measurement optimization
    success &= optimize_rigetti_measurements(circuit,
                                           state->measurement_order,
                                           state->num_qubits);

    return success;
}

static bool execute_optimized(RigettiState* state,
                            quantum_circuit* circuit,
                            quantum_result* result) {
    if (!state || !circuit || !result) {
        return false;
    }

    // Convert to pyQuil program
    pyquil_program* program = convert_to_pyquil(circuit);
    if (!program) {
        return false;
    }

    // Configure execution parameters from RigettiConfig
    // All settings come from the user-provided configuration
    execution_config config = {
        .shots = state->config.max_shots,
        .optimization_level = 0,  // Derived from backend_specific_config if set
        .error_mitigation = state->config.optimize_mapping
    };

    // Extract additional settings from backend_specific_config if available
    if (state->config.backend_specific_config) {
        // backend_specific_config can contain extended configuration
        // Cast to expected extended config type if provided
        typedef struct {
            int optimization_level;
            bool enable_error_mitigation;
            // Add other extended fields as needed
        } RigettiExtendedConfig;

        RigettiExtendedConfig* ext = (RigettiExtendedConfig*)state->config.backend_specific_config;
        config.optimization_level = ext->optimization_level;
        config.error_mitigation = ext->enable_error_mitigation;
    }

    // Execute on Rigetti hardware
    bool success = execute_pyquil_program(state->config.backend_name,
                                        program,
                                        result,
                                        &config);

    // Process results
    if (success) {
        process_rigetti_results(result,
                              state->readout_errors,
                              state->num_qubits);
    }

    cleanup_pyquil_program(program);
    return success;
}

static bool apply_error_mitigation(RigettiState* state,
                                 quantum_result* result) {
    if (!state || !result) {
        return false;
    }

    // Apply readout error mitigation
    if (!mitigate_rigetti_readout_errors(result,
                                        state->readout_errors,
                                        state->num_qubits)) {
        return false;
    }

    // Apply measurement error mitigation
    if (!mitigate_rigetti_measurement_errors(result,
                                           state->error_rates,
                                           state->num_qubits)) {
        return false;
    }

    // Apply noise extrapolation
    if (!extrapolate_rigetti_zero_noise(result,
                                       state->error_rates,
                                       state->num_qubits)) {
        return false;
    }

    return true;
}

// ============================================================================
// PyQuil Helper Function Implementations
// ============================================================================

static bool add_pyquil_instruction(pyquil_program* program, const char* instruction) {
    if (!program || !instruction) return false;

    // Ensure capacity
    if (program->num_instructions >= program->capacity) {
        size_t new_capacity = program->capacity == 0 ? 16 : program->capacity * 2;
        char** new_instructions = realloc(program->instructions,
                                          new_capacity * sizeof(char*));
        if (!new_instructions) return false;
        program->instructions = new_instructions;
        program->capacity = new_capacity;
    }

    // Copy instruction
    program->instructions[program->num_instructions] = strdup(instruction);
    if (!program->instructions[program->num_instructions]) return false;
    program->num_instructions++;

    return true;
}

static bool add_pyquil_rx(pyquil_program* program, double theta, size_t qubit) {
    if (!program) return false;

    char instruction[64];
    snprintf(instruction, sizeof(instruction), "RX(%.15g) %zu", theta, qubit);
    return add_pyquil_instruction(program, instruction);
}

static bool add_pyquil_rz(pyquil_program* program, double theta, size_t qubit) {
    if (!program) return false;

    char instruction[64];
    snprintf(instruction, sizeof(instruction), "RZ(%.15g) %zu", theta, qubit);
    return add_pyquil_instruction(program, instruction);
}

static bool add_pyquil_cz(pyquil_program* program, size_t q1, size_t q2) {
    if (!program) return false;

    char instruction[64];
    snprintf(instruction, sizeof(instruction), "CZ %zu %zu", q1, q2);
    return add_pyquil_instruction(program, instruction);
}

static bool add_pyquil_measure(pyquil_program* program, size_t qubit, size_t classical_bit) {
    if (!program) return false;

    char instruction[64];
    snprintf(instruction, sizeof(instruction), "MEASURE %zu ro[%zu]", qubit, classical_bit);
    return add_pyquil_instruction(program, instruction);
}

static pyquil_program* convert_to_pyquil(quantum_circuit* circuit) {
    if (!circuit) return NULL;

    pyquil_program* program = create_pyquil_program();
    if (!program) return NULL;

    // Add declaration for readout register
    char declare_stmt[64];
    snprintf(declare_stmt, sizeof(declare_stmt), "DECLARE ro BIT[%zu]",
             circuit->num_measurements > 0 ? circuit->num_measurements : circuit->num_qubits);
    if (!add_pyquil_instruction(program, declare_stmt)) {
        cleanup_pyquil_program(program);
        return NULL;
    }

    // Convert gates
    for (size_t i = 0; i < circuit->num_gates; i++) {
        quantum_gate* gate = &circuit->gates[i];
        if (gate->cancelled) continue;

        if (!add_pyquil_gate(program, gate)) {
            cleanup_pyquil_program(program);
            return NULL;
        }
    }

    // Convert measurements
    for (size_t i = 0; i < circuit->num_measurements; i++) {
        if (!add_pyquil_measure(program, circuit->measurements[i].qubit_idx,
                                circuit->measurements[i].classical_bit)) {
            cleanup_pyquil_program(program);
            return NULL;
        }
    }

    return program;
}

static bool execute_pyquil_program(const char* backend_name, pyquil_program* program,
                                   quantum_result* result, const execution_config* config) {
    if (!backend_name || !program || !result || !config) return false;

    // Initialize result structure
    result->shots = config->shots;
    result->num_measurements = 0;
    result->measurements = NULL;
    result->probabilities = NULL;
    result->backend_data = NULL;

    // Build complete Quil program string from instructions
    size_t total_len = 0;
    for (size_t i = 0; i < program->num_instructions; i++) {
        if (program->instructions[i]) {
            total_len += strlen(program->instructions[i]) + 1;  // +1 for newline
        }
    }

    char* quil_program = malloc(total_len + 1);
    if (!quil_program) return false;

    char* ptr = quil_program;
    for (size_t i = 0; i < program->num_instructions; i++) {
        if (program->instructions[i]) {
            size_t len = strlen(program->instructions[i]);
            memcpy(ptr, program->instructions[i], len);
            ptr += len;
            *ptr++ = '\n';
        }
    }
    *ptr = '\0';

    // Connect to Rigetti QCS (or use local simulator as fallback)
    rigetti_qcs_handle_t* qcs_handle = qcs_connect_default();
    if (!qcs_handle) {
        free(quil_program);
        return false;
    }

    // Configure execution options
    qcs_execution_options_t exec_options = {
        .target = QCS_TARGET_QVM,  // Use QVM by default, QPU if backend_name matches
        .qpu_name = backend_name,
        .shots = config->shots,
        .use_quilc = true,
        .use_parametric = false,
        .use_active_reset = false,
        .timeout_seconds = 300
    };

    // Determine target based on backend name
    if (strstr(backend_name, "Aspen") || strstr(backend_name, "Ankaa")) {
        exec_options.target = QCS_TARGET_QPU;
    } else if (strstr(backend_name, "noisy") || strstr(backend_name, "Noisy")) {
        exec_options.target = QCS_TARGET_QVM_NOISY;
    }

    // Execute program via QCS API
    qcs_job_result_t qcs_result;
    memset(&qcs_result, 0, sizeof(qcs_job_result_t));

    bool success = qcs_execute_program(qcs_handle, quil_program, &exec_options, &qcs_result);

    if (success && qcs_result.status == QCS_JOB_COMPLETED) {
        // Convert QCS results to internal quantum_result format
        size_t num_outcomes = qcs_result.num_qubits > 0 ? (1ULL << qcs_result.num_qubits) : 2;

        // Allocate measurement results from bitstrings
        result->measurements = calloc(qcs_result.num_shots, sizeof(double));
        if (result->measurements) {
            result->num_measurements = qcs_result.num_shots;

            // Convert bitstrings to integer outcomes
            for (size_t i = 0; i < qcs_result.num_shots; i++) {
                uint64_t outcome = 0;
                if (qcs_result.bitstrings && qcs_result.bitstrings[i]) {
                    for (size_t q = 0; q < qcs_result.num_qubits; q++) {
                        if (qcs_result.bitstrings[i][q]) {
                            outcome |= (1ULL << q);
                        }
                    }
                }
                result->measurements[i] = (double)outcome;
            }
        }

        // Copy probability distribution
        if (qcs_result.probabilities && qcs_result.num_outcomes > 0) {
            result->probabilities = calloc(num_outcomes, sizeof(double));
            if (result->probabilities) {
                memcpy(result->probabilities, qcs_result.probabilities,
                       num_outcomes * sizeof(double));
            }
        }
    } else {
        // Execution failed - log error
        const char* error = qcs_get_last_error(qcs_handle);
        if (error) {
            fprintf(stderr, "QCS execution failed: %s\n", error);
        }
        success = false;
    }

    // Cleanup
    qcs_free_result(&qcs_result);
    qcs_disconnect(qcs_handle);
    free(quil_program);

    return success;
}

static void process_rigetti_results(quantum_result* result, double* readout_errors,
                                    size_t num_qubits) {
    if (!result || !result->measurements) return;

    // Calculate probabilities from measurement results
    if (!result->probabilities) {
        size_t num_states = 1UL << num_qubits;
        result->probabilities = calloc(num_states, sizeof(double));
        if (!result->probabilities) return;
    }

    // Count occurrences and convert to probabilities
    size_t num_states = 1UL << num_qubits;
    for (size_t i = 0; i < result->num_measurements; i++) {
        size_t state_idx = (size_t)result->measurements[i] % num_states;
        result->probabilities[state_idx] += 1.0;
    }

    // Normalize
    for (size_t i = 0; i < num_states; i++) {
        result->probabilities[i] /= (double)result->num_measurements;
    }
}

// ============================================================================
// Circuit Optimization Function Implementations
// ============================================================================

static bool decompose_to_native_gates(quantum_circuit* circuit) {
    if (!circuit) return false;

    gate_set* native_gates = get_rigetti_native_gates();
    if (!native_gates) return false;

    // Process each gate, potentially expanding it into multiple gates
    size_t i = 0;
    while (i < circuit->num_gates) {
        size_t skip = decompose_gate_in_circuit(circuit, i, native_gates);
        i += 1 + skip;  // Move past decomposed gates
    }

    free(native_gates->types);
    free(native_gates);
    return true;
}

static bool optimize_rigetti_gates(quantum_circuit* circuit) {
    if (!circuit) return false;

    bool changed = true;
    while (changed) {
        changed = false;

        // Gate cancellation: look for adjacent inverse gates
        for (size_t i = 0; i + 1 < circuit->num_gates; i++) {
            quantum_gate* g1 = &circuit->gates[i];
            quantum_gate* g2 = &circuit->gates[i + 1];

            if (g1->cancelled || g2->cancelled) continue;

            // Check if gates can be combined
            if (can_combine_rigetti_gates(g1, g2)) {
                combine_rigetti_gates(g1, g2);
                g2->cancelled = true;
                changed = true;

                // Check if combined gate is identity (angle ≈ 0 or 2π)
                if (g1->num_params > 0) {
                    double angle = fmod(fabs(g1->params[0]), 2 * M_PI);
                    if (angle < 1e-10 || fabs(angle - 2 * M_PI) < 1e-10) {
                        g1->cancelled = true;
                    }
                }
            }
        }
    }

    // Compact the circuit by removing cancelled gates
    compact_rigetti_circuit(circuit);
    return true;
}

static bool optimize_rigetti_routing(quantum_circuit* circuit, double** coupling_map,
                                     size_t num_qubits) {
    if (!circuit || !coupling_map) return false;

    // Build connectivity graph
    connectivity_graph* graph = build_rigetti_connectivity(coupling_map, num_qubits);
    if (!graph) return false;

    // Find optimal qubit mapping
    size_t* mapping = find_optimal_rigetti_mapping(graph, circuit);
    if (!mapping) {
        // Cleanup graph
        for (size_t i = 0; i < num_qubits; i++) {
            free(graph->connections[i]);
        }
        free(graph->connections);
        free(graph);
        return false;
    }

    // Apply qubit remapping
    remap_rigetti_qubits(circuit, mapping);

    // Cleanup
    free(mapping);
    for (size_t i = 0; i < num_qubits; i++) {
        free(graph->connections[i]);
    }
    free(graph->connections);
    free(graph);

    return true;
}

static bool optimize_rigetti_measurements(quantum_circuit* circuit, size_t* measurement_order,
                                          size_t num_qubits) {
    if (!circuit) return false;

    // If we have measurement order optimization, apply it
    if (measurement_order && circuit->measurements) {
        // Assign optimized order to measurements
        for (size_t i = 0; i < circuit->num_measurements; i++) {
            size_t qubit = circuit->measurements[i].qubit_idx;
            if (qubit < num_qubits) {
                circuit->measurements[i].optimized_order = measurement_order[qubit];
            }
        }

        // Sort measurements by optimized order
        sort_rigetti_measurements(circuit);
    }

    return true;
}

static void optimize_rigetti_measurement_order(size_t* measurement_order,
                                               double* readout_errors, size_t num_qubits) {
    if (!measurement_order || !readout_errors) return;

    // Sort qubits by readout error (lowest error first)
    // Use simple selection sort for small qubit counts
    for (size_t i = 0; i < num_qubits; i++) {
        measurement_order[i] = i;
    }

    for (size_t i = 0; i < num_qubits; i++) {
        size_t min_idx = i;
        double min_error = readout_errors[measurement_order[i]];

        for (size_t j = i + 1; j < num_qubits; j++) {
            if (readout_errors[measurement_order[j]] < min_error) {
                min_error = readout_errors[measurement_order[j]];
                min_idx = j;
            }
        }

        if (min_idx != i) {
            size_t temp = measurement_order[i];
            measurement_order[i] = measurement_order[min_idx];
            measurement_order[min_idx] = temp;
        }
    }
}

// ============================================================================
// Error Mitigation Function Implementations
// ============================================================================

static bool mitigate_rigetti_readout_errors(quantum_result* result, double* readout_errors,
                                            size_t num_qubits) {
    if (!result || !result->probabilities || !readout_errors) return false;

    size_t num_states = 1UL << num_qubits;

    // Build and invert the readout confusion matrix
    // For simplicity, apply a diagonal correction based on readout fidelity
    double* corrected = calloc(num_states, sizeof(double));
    if (!corrected) return false;

    for (size_t state = 0; state < num_states; state++) {
        // Calculate correction factor based on bit errors
        double correction = 1.0;
        for (size_t q = 0; q < num_qubits; q++) {
            double error = readout_errors[q];
            double fidelity = 1.0 - error;
            // Apply fidelity correction
            correction *= fidelity;
        }

        // Apply inverse correction (boost probabilities)
        if (correction > 0.01) {  // Avoid division by near-zero
            corrected[state] = result->probabilities[state] / correction;
        } else {
            corrected[state] = result->probabilities[state];
        }
    }

    // Renormalize
    double sum = 0.0;
    for (size_t i = 0; i < num_states; i++) {
        sum += corrected[i];
    }
    if (sum > 0.0) {
        for (size_t i = 0; i < num_states; i++) {
            result->probabilities[i] = corrected[i] / sum;
        }
    }

    free(corrected);
    return true;
}

static bool mitigate_rigetti_measurement_errors(quantum_result* result, double* error_rates,
                                                size_t num_qubits) {
    if (!result || !result->probabilities || !error_rates) return false;

    // Measurement error mitigation using twirling/symmetrization
    // This is a simplified version - full implementation would use randomized compiling

    size_t num_states = 1UL << num_qubits;

    // Calculate average error rate
    double avg_error = 0.0;
    for (size_t q = 0; q < num_qubits; q++) {
        avg_error += error_rates[q];
    }
    avg_error /= num_qubits;

    // Apply depolarizing correction
    double p = 1.0 - avg_error;
    if (p > 0.1) {  // Ensure reasonable correction
        for (size_t i = 0; i < num_states; i++) {
            // Correct for depolarizing noise
            double uniform = 1.0 / num_states;
            result->probabilities[i] = (result->probabilities[i] - (1 - p) * uniform) / p;
            if (result->probabilities[i] < 0) result->probabilities[i] = 0;
        }

        // Renormalize
        double sum = 0.0;
        for (size_t i = 0; i < num_states; i++) {
            sum += result->probabilities[i];
        }
        if (sum > 0.0) {
            for (size_t i = 0; i < num_states; i++) {
                result->probabilities[i] /= sum;
            }
        }
    }

    return true;
}

static bool extrapolate_rigetti_zero_noise(quantum_result* result, double* error_rates,
                                           size_t num_qubits) {
    if (!result || !result->probabilities || !error_rates) return false;

    // Zero-noise extrapolation using Richardson extrapolation
    // This requires multiple runs at different noise levels - we simulate
    // the effect using a polynomial extrapolation based on measured error rates

    size_t num_states = 1UL << num_qubits;

    // Calculate total effective noise parameter
    double total_noise = 0.0;
    for (size_t q = 0; q < num_qubits; q++) {
        total_noise += error_rates[q];
    }

    if (total_noise < 0.001) return true;  // Already low noise

    // Apply linear extrapolation to zero noise
    // p_zne = p_measured + (p_measured - p_uniform) * noise_factor
    double noise_factor = total_noise / (1.0 + total_noise);
    double uniform = 1.0 / num_states;

    for (size_t i = 0; i < num_states; i++) {
        double delta = result->probabilities[i] - uniform;
        result->probabilities[i] += delta * noise_factor;
        if (result->probabilities[i] < 0) result->probabilities[i] = 0;
        if (result->probabilities[i] > 1) result->probabilities[i] = 1;
    }

    // Renormalize
    double sum = 0.0;
    for (size_t i = 0; i < num_states; i++) {
        sum += result->probabilities[i];
    }
    if (sum > 0.0) {
        for (size_t i = 0; i < num_states; i++) {
            result->probabilities[i] /= sum;
        }
    }

    return true;
}

// ============================================================================
// Backend Configuration Function Implementations
// ============================================================================

// Device properties callback type for dynamic configuration
typedef bool (*rigetti_properties_callback_t)(const char* backend_name,
                                              size_t* num_qubits,
                                              double** calibration_data,
                                              void* user_data);

// Coupling strength callback type for dynamic topology
typedef double (*rigetti_coupling_callback_t)(const char* backend_name,
                                              size_t q1, size_t q2,
                                              void* user_data);

// Global callback holders (can be set by user)
static rigetti_properties_callback_t g_properties_callback = NULL;
static rigetti_coupling_callback_t g_coupling_callback = NULL;
static void* g_properties_user_data = NULL;
static void* g_coupling_user_data = NULL;

// Public API to register custom property providers
void rigetti_set_properties_callback(rigetti_properties_callback_t callback, void* user_data) {
    g_properties_callback = callback;
    g_properties_user_data = user_data;
}

void rigetti_set_coupling_callback(rigetti_coupling_callback_t callback, void* user_data) {
    g_coupling_callback = callback;
    g_coupling_user_data = user_data;
}

static bool get_rigetti_properties(const char* backend_name, size_t* num_qubits,
                                   double** calibration_data) {
    if (!backend_name || !num_qubits || !calibration_data) return false;

    // Use user-provided callback if available
    if (g_properties_callback) {
        return g_properties_callback(backend_name, num_qubits, calibration_data,
                                     g_properties_user_data);
    }

    // Query the Rigetti QCS API for device properties
    // This uses the backend_specific_config from RigettiConfig if set

    // Try to get properties from environment or config file
    const char* qubits_env = getenv("RIGETTI_NUM_QUBITS");
    if (qubits_env) {
        *num_qubits = (size_t)atoi(qubits_env);
    } else {
        // Fallback: Query via API (requires backend_specific_config to have API handle)
        // If no API available, return false to indicate configuration needed
        *num_qubits = 0;
        *calibration_data = NULL;
        return false;  // Must be configured externally
    }

    if (*num_qubits == 0) return false;

    // Allocate calibration data: 3 values per qubit (error, readout_error, availability)
    *calibration_data = calloc(*num_qubits * 3, sizeof(double));
    if (!*calibration_data) return false;

    // Try to load calibration from environment or file
    const char* cal_file = getenv("RIGETTI_CALIBRATION_FILE");
    if (cal_file) {
        // Load from calibration file (JSON format)
        FILE* fp = fopen(cal_file, "r");
        if (fp) {
            // Parse calibration JSON - each qubit has gate_error, readout_error, available
            for (size_t i = 0; i < *num_qubits; i++) {
                double gate_err, readout_err, avail;
                if (fscanf(fp, "%lf %lf %lf", &gate_err, &readout_err, &avail) == 3) {
                    (*calibration_data)[i * 3 + 0] = gate_err;
                    (*calibration_data)[i * 3 + 1] = readout_err;
                    (*calibration_data)[i * 3 + 2] = avail;
                }
            }
            fclose(fp);
        }
    }
    // If no calibration file, values remain at 0 - caller must configure

    return true;
}

static double get_rigetti_coupling_strength(const char* backend_name, size_t q1, size_t q2) {
    if (!backend_name || q1 == q2) return 0.0;

    // Use user-provided callback if available
    if (g_coupling_callback) {
        return g_coupling_callback(backend_name, q1, q2, g_coupling_user_data);
    }

    // Try to get coupling map from environment
    const char* coupling_file = getenv("RIGETTI_COUPLING_MAP");
    if (coupling_file) {
        // Load coupling map from file
        // Format: q1 q2 strength per line
        FILE* fp = fopen(coupling_file, "r");
        if (fp) {
            size_t file_q1, file_q2;
            double strength;
            while (fscanf(fp, "%zu %zu %lf", &file_q1, &file_q2, &strength) == 3) {
                if ((file_q1 == q1 && file_q2 == q2) ||
                    (file_q1 == q2 && file_q2 == q1)) {
                    fclose(fp);
                    return strength;
                }
            }
            fclose(fp);
        }
    }

    // No coupling information available - return 0 (not connected)
    // User must configure coupling map via callback or file
    return 0.0;
}

static bool validate_rigetti_config(const struct RigettiConfig* config) {
    if (!config) return false;

    // Validate required fields
    if (!config->backend_name) return false;

    // max_shots must be positive (no upper limit imposed - device specific)
    if (config->max_shots == 0) return false;

    return true;
}
