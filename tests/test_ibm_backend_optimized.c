/**
 * @file test_ibm_backend_optimized.c
 * @brief Tests for optimized IBM quantum backend
 */

#include "quantum_geometric/hardware/quantum_ibm_backend.h"
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
    printf("Testing IBM backend initialization...\n");

    // Setup config
    IBMConfig config = {
        .backend_name = "ibmq_test",
        .num_shots = 1000,
        .optimization_level = 3,
        .error_mitigation = true,
        .fast_feedback = true
    };

    // Initialize backend
    IBMState state;
    bool success = init_ibm_backend(&state, &config);
    assert(success && "Failed to initialize backend");

    // Verify initialization
    assert(state.initialized && "Backend not marked as initialized");
    assert(state.num_qubits > 0 && "Invalid number of qubits");
    assert(state.error_rates && "Error rates not allocated");
    assert(state.readout_errors && "Readout errors not allocated");
    assert(state.qubit_availability && "Qubit availability not allocated");
    assert(state.measurement_order && "Measurement order not allocated");
    assert(state.coupling_map && "Coupling map not allocated");

    cleanup_ibm_backend(&state);
    printf("Initialization test passed\n");
}

static void test_circuit_optimization() {
    printf("Testing circuit optimization...\n");

    // Setup backend
    IBMConfig config = {
        .backend_name = "ibmq_test",
        .num_shots = 1000,
        .optimization_level = 3,
        .error_mitigation = true,
        .fast_feedback = true
    };

    IBMState state;
    bool success = init_ibm_backend(&state, &config);
    assert(success && "Failed to initialize backend");

    // Create test circuit with redundant gates
    quantum_circuit* circuit = create_test_circuit();
    
    // Add redundant X gates (should cancel)
    quantum_gate x1 = {.type = GATE_X, .qubit = 0};
    quantum_gate x2 = {.type = GATE_X, .qubit = 0};
    circuit->gates[circuit->num_gates++] = x1;
    circuit->gates[circuit->num_gates++] = x2;

    // Add fusible rotation gates
    quantum_gate r1 = {.type = GATE_RZ, .qubit = 1, .params[0] = 0.1};
    quantum_gate r2 = {.type = GATE_RZ, .qubit = 1, .params[0] = 0.2};
    circuit->gates[circuit->num_gates++] = r1;
    circuit->gates[circuit->num_gates++] = r2;

    // Optimize circuit
    success = optimize_circuit(&state, circuit);
    assert(success && "Circuit optimization failed");

    // Verify optimizations
    size_t original_gates = circuit->num_gates;
    bool found_cancelled = false;
    bool found_fused = false;

    for (size_t i = 0; i < circuit->num_gates; i++) {
        quantum_gate* g = &circuit->gates[i];
        if (g->cancelled) {
            found_cancelled = true;
        }
        if (g->type == GATE_RZ && fabs(g->params[0] - 0.3) < 1e-6) {
            found_fused = true;
        }
    }

    assert(found_cancelled && "Gate cancellation failed");
    assert(found_fused && "Gate fusion failed");
    assert(circuit->num_gates < original_gates && 
           "Circuit not reduced after optimization");

    cleanup_test_circuit(circuit);
    cleanup_ibm_backend(&state);
    printf("Circuit optimization test passed\n");
}

static void test_parallel_execution() {
    printf("Testing parallel execution...\n");

    // Setup backend
    IBMConfig config = {
        .backend_name = "ibmq_test",
        .num_shots = 1000,
        .optimization_level = 3,
        .error_mitigation = true,
        .fast_feedback = true
    };

    IBMState state;
    bool success = init_ibm_backend(&state, &config);
    assert(success && "Failed to initialize backend");

    // Create test circuit with parallel gates
    quantum_circuit* circuit = create_test_circuit();
    
    // Add independent gates that can run in parallel
    quantum_gate h1 = {.type = GATE_H, .qubit = 0};
    quantum_gate h2 = {.type = GATE_H, .qubit = 2};
    circuit->gates[circuit->num_gates++] = h1;
    circuit->gates[circuit->num_gates++] = h2;

    // Add measurements
    quantum_measurement m1 = {.qubit_idx = 0};
    quantum_measurement m2 = {.qubit_idx = 2};
    circuit->measurements[circuit->num_measurements++] = m1;
    circuit->measurements[circuit->num_measurements++] = m2;

    // Execute circuit
    quantum_result result;
    success = execute_circuit(&state, circuit, &result);
    assert(success && "Circuit execution failed");

    // Verify parallel execution
    assert(result.parallel_groups > 0 && 
           "No parallel execution groups created");
    assert(result.execution_time < 2 * result.gate_time && 
           "Parallel execution not faster than serial");

    cleanup_test_circuit(circuit);
    cleanup_ibm_backend(&state);
    printf("Parallel execution test passed\n");
}

static void test_error_mitigation() {
    printf("Testing error mitigation...\n");

    // Setup backend
    IBMConfig config = {
        .backend_name = "ibmq_test",
        .num_shots = 1000,
        .optimization_level = 3,
        .error_mitigation = true,
        .fast_feedback = true
    };

    IBMState state;
    bool success = init_ibm_backend(&state, &config);
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
    cleanup_ibm_backend(&state);
    printf("Error mitigation test passed\n");
}

static void test_fast_feedback() {
    printf("Testing fast feedback...\n");

    // Setup backend
    IBMConfig config = {
        .backend_name = "ibmq_test",
        .num_shots = 1000,
        .optimization_level = 3,
        .error_mitigation = true,
        .fast_feedback = true
    };

    IBMState state;
    bool success = init_ibm_backend(&state, &config);
    assert(success && "Failed to initialize backend");

    // Create test circuit with conditional operations
    quantum_circuit* circuit = create_test_circuit();
    
    // Add measurement-based feedback
    quantum_gate h = {.type = GATE_H, .qubit = 0};
    quantum_measurement m = {.qubit_idx = 0};
    quantum_gate x = {
        .type = GATE_X,
        .qubit = 1,
        .conditional = true,
        .condition_qubit = 0,
        .condition_value = 1
    };

    circuit->gates[circuit->num_gates++] = h;
    circuit->measurements[circuit->num_measurements++] = m;
    circuit->gates[circuit->num_gates++] = x;

    // Execute circuit
    quantum_result result;
    success = execute_circuit(&state, circuit, &result);
    assert(success && "Circuit execution failed");

    // Verify fast feedback
    assert(result.feedback_latency < 1e-6 && 
           "Feedback latency too high");
    assert(result.conditional_success_rate > 0.99 && 
           "Conditional operations not reliable");

    cleanup_test_circuit(circuit);
    cleanup_ibm_backend(&state);
    printf("Fast feedback test passed\n");
}

static void test_error_handling() {
    printf("Testing error handling...\n");

    // Test null pointers
    bool success = init_ibm_backend(NULL, NULL);
    assert(!success && "Should fail with null pointers");

    // Test invalid config
    IBMConfig invalid_config = {
        .backend_name = NULL,
        .num_shots = 0,
        .optimization_level = 99,
        .error_mitigation = true,
        .fast_feedback = true
    };

    IBMState state;
    success = init_ibm_backend(&state, &invalid_config);
    assert(!success && "Should fail with invalid config");

    // Test invalid circuit
    IBMConfig valid_config = {
        .backend_name = "ibmq_test",
        .num_shots = 1000,
        .optimization_level = 3,
        .error_mitigation = true,
        .fast_feedback = true
    };

    success = init_ibm_backend(&state, &valid_config);
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
    cleanup_ibm_backend(&state);
    printf("Error handling test passed\n");
}

int main() {
    printf("Running IBM backend tests...\n\n");

    test_initialization();
    test_circuit_optimization();
    test_parallel_execution();
    test_error_mitigation();
    test_fast_feedback();
    test_error_handling();

    printf("\nAll IBM backend tests passed!\n");
    return 0;
}
