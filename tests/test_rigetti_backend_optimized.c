/**
 * @file test_rigetti_backend_optimized.c
 * @brief Tests for optimized Rigetti quantum backend
 */

#include "quantum_geometric/hardware/quantum_rigetti_backend.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

// Test helper functions
static quantum_circuit* create_test_circuit() {
    quantum_circuit* circuit = malloc(sizeof(quantum_circuit));
    circuit->num_qubits = 4;
    circuit->num_gates = 0;
    circuit->max_gates = 100;
    circuit->gates = calloc(circuit->max_gates, sizeof(quantum_gate));
    circuit->num_measurements = 0;
    circuit->max_measurements = 10;
    circuit->measurements = calloc(circuit->max_measurements, 
                                 sizeof(quantum_measurement));
    return circuit;
}

static void cleanup_test_circuit(quantum_circuit* circuit) {
    if (circuit) {
        free(circuit->gates);
        free(circuit->measurements);
        free(circuit);
    }
}

static void test_initialization() {
    printf("Testing Rigetti backend initialization...\n");

    // Setup config
    RigettiConfig config = {
        .backend_name = "Aspen-9",
        .num_shots = 1000,
        .optimization_level = 3,
        .error_mitigation = true,
        .native_gates_only = true
    };

    // Initialize backend
    RigettiState state;
    bool success = init_rigetti_backend(&state, &config);
    assert(success && "Failed to initialize backend");

    // Verify initialization
    assert(state.initialized && "Backend not marked as initialized");
    assert(state.num_qubits > 0 && "Invalid number of qubits");
    assert(state.error_rates && "Error rates not allocated");
    assert(state.readout_errors && "Readout errors not allocated");
    assert(state.qubit_availability && "Qubit availability not allocated");
    assert(state.measurement_order && "Measurement order not allocated");
    assert(state.coupling_map && "Coupling map not allocated");

    cleanup_rigetti_backend(&state);
    printf("Initialization test passed\n");
}

static void test_native_gates() {
    printf("Testing native gate decomposition...\n");

    // Setup backend
    RigettiConfig config = {
        .backend_name = "Aspen-9",
        .num_shots = 1000,
        .optimization_level = 3,
        .error_mitigation = true,
        .native_gates_only = true
    };

    RigettiState state;
    bool success = init_rigetti_backend(&state, &config);
    assert(success && "Failed to initialize backend");

    // Create test circuit with non-native gates
    quantum_circuit* circuit = create_test_circuit();
    
    // Add non-native gate (e.g., T gate)
    quantum_gate t = {.type = GATE_T, .qubit = 0};
    circuit->gates[circuit->num_gates++] = t;

    // Add non-native two-qubit gate
    quantum_gate swap = {.type = GATE_SWAP, .qubit = 0, .target = 1};
    circuit->gates[circuit->num_gates++] = swap;

    // Optimize circuit (includes decomposition)
    success = optimize_circuit(&state, circuit);
    assert(success && "Circuit optimization failed");

    // Verify decomposition
    bool all_native = true;
    for (size_t i = 0; i < circuit->num_gates; i++) {
        quantum_gate* g = &circuit->gates[i];
        if (!is_native_gate(g, get_rigetti_native_gates())) {
            all_native = false;
            break;
        }
    }
    assert(all_native && "Not all gates decomposed to native gates");

    cleanup_test_circuit(circuit);
    cleanup_rigetti_backend(&state);
    printf("Native gate decomposition test passed\n");
}

static void test_pyquil_integration() {
    printf("Testing pyQuil integration...\n");

    // Setup backend
    RigettiConfig config = {
        .backend_name = "Aspen-9",
        .num_shots = 1000,
        .optimization_level = 3,
        .error_mitigation = true,
        .native_gates_only = true
    };

    RigettiState state;
    bool success = init_rigetti_backend(&state, &config);
    assert(success && "Failed to initialize backend");

    // Create test circuit
    quantum_circuit* circuit = create_test_circuit();
    
    // Add some gates
    quantum_gate h = {.type = GATE_H, .qubit = 0};
    quantum_gate cnot = {.type = GATE_CNOT, .qubit = 0, .target = 1};
    circuit->gates[circuit->num_gates++] = h;
    circuit->gates[circuit->num_gates++] = cnot;

    // Add measurement
    quantum_measurement m = {.qubit_idx = 1};
    circuit->measurements[circuit->num_measurements++] = m;

    // Convert to pyQuil program
    pyquil_program* program = convert_to_pyquil(circuit);
    assert(program && "Failed to convert to pyQuil program");

    // Verify program structure
    assert(program->num_instructions > 0 && 
           "No instructions in pyQuil program");
    assert(program->num_measurements > 0 && 
           "No measurements in pyQuil program");

    cleanup_pyquil_program(program);
    cleanup_test_circuit(circuit);
    cleanup_rigetti_backend(&state);
    printf("pyQuil integration test passed\n");
}

static void test_qubit_routing() {
    printf("Testing qubit routing optimization...\n");

    // Setup backend
    RigettiConfig config = {
        .backend_name = "Aspen-9",
        .num_shots = 1000,
        .optimization_level = 3,
        .error_mitigation = true,
        .native_gates_only = true
    };

    RigettiState state;
    bool success = init_rigetti_backend(&state, &config);
    assert(success && "Failed to initialize backend");

    // Create test circuit with non-adjacent qubit operations
    quantum_circuit* circuit = create_test_circuit();
    
    // Add CNOT between non-adjacent qubits
    quantum_gate cnot = {.type = GATE_CNOT, .qubit = 0, .target = 3};
    circuit->gates[circuit->num_gates++] = cnot;

    // Get initial path length
    size_t initial_length = circuit->num_gates;

    // Optimize routing
    success = optimize_rigetti_routing(circuit,
                                     state.coupling_map,
                                     state.num_qubits);
    assert(success && "Routing optimization failed");

    // Verify routing
    assert(circuit->num_gates > initial_length && 
           "No SWAP gates inserted for routing");
    
    bool valid_routing = true;
    for (size_t i = 0; i < circuit->num_gates; i++) {
        quantum_gate* g = &circuit->gates[i];
        if (g->type == GATE_CNOT) {
            // Check if qubits are adjacent in coupling map
            if (!are_qubits_adjacent(state.coupling_map,
                                   g->qubit,
                                   g->target)) {
                valid_routing = false;
                break;
            }
        }
    }
    assert(valid_routing && "Invalid qubit routing");

    cleanup_test_circuit(circuit);
    cleanup_rigetti_backend(&state);
    printf("Qubit routing test passed\n");
}

static void test_error_mitigation() {
    printf("Testing Rigetti error mitigation...\n");

    // Setup backend
    RigettiConfig config = {
        .backend_name = "Aspen-9",
        .num_shots = 1000,
        .optimization_level = 3,
        .error_mitigation = true,
        .native_gates_only = true
    };

    RigettiState state;
    bool success = init_rigetti_backend(&state, &config);
    assert(success && "Failed to initialize backend");

    // Create test circuit
    quantum_circuit* circuit = create_test_circuit();
    
    // Add gates and measurements
    quantum_gate h = {.type = GATE_H, .qubit = 0};
    circuit->gates[circuit->num_gates++] = h;
    
    quantum_measurement m = {.qubit_idx = 0};
    circuit->measurements[circuit->num_measurements++] = m;

    // Execute circuit multiple times to build statistics
    quantum_result results[10];
    for (size_t i = 0; i < 10; i++) {
        success = execute_circuit(&state, circuit, &results[i]);
        assert(success && "Circuit execution failed");
    }

    // Verify error mitigation
    double raw_error_rate = 0.0;
    double mitigated_error_rate = 0.0;

    for (size_t i = 0; i < 10; i++) {
        raw_error_rate += results[i].raw_error_rate;
        mitigated_error_rate += results[i].mitigated_error_rate;
    }
    raw_error_rate /= 10;
    mitigated_error_rate /= 10;

    assert(mitigated_error_rate < raw_error_rate && 
           "Error mitigation not effective");
    assert(mitigated_error_rate < state.config.error_threshold && 
           "Error rate above threshold after mitigation");

    cleanup_test_circuit(circuit);
    cleanup_rigetti_backend(&state);
    printf("Error mitigation test passed\n");
}

static void test_error_handling() {
    printf("Testing error handling...\n");

    // Test null pointers
    bool success = init_rigetti_backend(NULL, NULL);
    assert(!success && "Should fail with null pointers");

    // Test invalid config
    RigettiConfig invalid_config = {
        .backend_name = NULL,
        .num_shots = 0,
        .optimization_level = 99,
        .error_mitigation = true,
        .native_gates_only = true
    };

    RigettiState state;
    success = init_rigetti_backend(&state, &invalid_config);
    assert(!success && "Should fail with invalid config");

    // Test invalid circuit
    RigettiConfig valid_config = {
        .backend_name = "Aspen-9",
        .num_shots = 1000,
        .optimization_level = 3,
        .error_mitigation = true,
        .native_gates_only = true
    };

    success = init_rigetti_backend(&state, &valid_config);
    assert(success && "Failed to initialize with valid config");

    quantum_result result;
    success = execute_circuit(&state, NULL, &result);
    assert(!success && "Should fail with null circuit");

    // Test circuit with too many qubits
    quantum_circuit* large_circuit = create_test_circuit();
    large_circuit->num_qubits = 9999;
    
    success = execute_circuit(&state, large_circuit, &result);
    assert(!success && "Should fail with too many qubits");

    cleanup_test_circuit(large_circuit);
    cleanup_rigetti_backend(&state);
    printf("Error handling test passed\n");
}

int main() {
    printf("Running Rigetti backend tests...\n\n");

    test_initialization();
    test_native_gates();
    test_pyquil_integration();
    test_qubit_routing();
    test_error_mitigation();
    test_error_handling();

    printf("\nAll Rigetti backend tests passed!\n");
    return 0;
}
