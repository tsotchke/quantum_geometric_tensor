/**
 * @file test_ibm_backend_optimized.c
 * @brief Tests for optimized IBM quantum backend
 */

#include "quantum_geometric/hardware/quantum_ibm_backend.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/core/quantum_circuit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

// Helper to create test circuit with gates
static quantum_circuit_t* create_test_circuit_with_gates() {
    quantum_circuit_t* circuit = malloc(sizeof(quantum_circuit_t));
    memset(circuit, 0, sizeof(quantum_circuit_t));
    circuit->num_qubits = 4;
    circuit->num_gates = 0;
    circuit->gates = calloc(100, sizeof(quantum_gate_t*));
    return circuit;
}

// Helper to add a gate to circuit
static void add_test_gate(quantum_circuit_t* circuit, gate_type_t type, size_t qubit, double* params, size_t num_params) {
    quantum_gate_t* gate = calloc(1, sizeof(quantum_gate_t));
    gate->type = type;
    gate->num_qubits = 1;
    gate->target_qubits = malloc(sizeof(size_t));
    gate->target_qubits[0] = qubit;

    if (num_params > 0 && params) {
        gate->parameters = malloc(num_params * sizeof(double));
        memcpy(gate->parameters, params, num_params * sizeof(double));
        gate->num_parameters = num_params;
    }

    circuit->gates[circuit->num_gates++] = gate;
}

static void test_initialization() {
    printf("Testing IBM backend initialization...\n");

    // Setup config using actual IBMBackendConfig type
    IBMBackendConfig config = {0};
    config.backend_name = strdup("ibmq_qasm_simulator");
    config.optimization_level = 3;
    config.error_mitigation = true;
    config.dynamic_decoupling = false;
    config.readout_error_mitigation = true;
    config.measurement_error_mitigation = true;

    // Initialize backend
    IBMBackendState state = {0};
    qgt_error_t err = init_ibm_backend(&state, &config);
    assert(err == QGT_SUCCESS && "Failed to initialize backend");

    // Verify initialization
    assert(state.initialized && "Backend not marked as initialized");
    assert(state.num_qubits > 0 && "Invalid number of qubits");
    assert(state.error_rates && "Error rates not allocated");
    assert(state.readout_errors && "Readout errors not allocated");
    assert(state.qubit_availability && "Qubit availability not allocated");
    assert(state.measurement_order && "Measurement order not allocated");
    assert(state.coupling_map && "Coupling map not allocated");

    cleanup_ibm_config(&config);
    printf("Initialization test passed\n");
}

static void test_circuit_optimization() {
    printf("Testing circuit optimization...\n");

    // Setup backend
    IBMBackendConfig config = {0};
    config.backend_name = strdup("ibmq_qasm_simulator");
    config.optimization_level = 3;
    config.error_mitigation = true;

    IBMBackendState state = {0};
    qgt_error_t err = init_ibm_backend(&state, &config);
    assert(err == QGT_SUCCESS && "Failed to initialize backend");

    // Create test circuit with redundant gates
    quantum_circuit_t* circuit = create_test_circuit_with_gates();

    // Add redundant X gates (should cancel)
    add_test_gate(circuit, GATE_X, 0, NULL, 0);
    add_test_gate(circuit, GATE_X, 0, NULL, 0);

    // Add fusible rotation gates
    double angle1 = 0.1;
    double angle2 = 0.2;
    add_test_gate(circuit, GATE_RZ, 1, &angle1, 1);
    add_test_gate(circuit, GATE_RZ, 1, &angle2, 1);

    // Optimize circuit
    bool success = optimize_circuit(&state, circuit);
    assert(success && "Circuit optimization failed");

    // Verify optimizations - check for cancelled and fused gates
    bool found_cancelled = false;
    bool found_fused = false;

    for (size_t i = 0; i < circuit->num_gates; i++) {
        quantum_gate_t* g = circuit->gates[i];
        if (g->custom_data) {
            found_cancelled = true;
        }
        if (g->type == GATE_RZ && g->parameters && fabs(g->parameters[0] - 0.3) < 1e-6) {
            found_fused = true;
        }
    }

    assert(found_cancelled && "Gate cancellation failed");
    assert(found_fused && "Gate fusion failed");

    cleanup_circuit(circuit);
    cleanup_ibm_config(&config);
    printf("Circuit optimization test passed\n");
}

static void test_circuit_execution() {
    printf("Testing circuit execution...\n");

    // Setup backend
    IBMBackendConfig config = {0};
    config.backend_name = strdup("ibmq_qasm_simulator");
    config.optimization_level = 3;
    config.error_mitigation = true;

    IBMBackendState state = {0};
    qgt_error_t err = init_ibm_backend(&state, &config);
    assert(err == QGT_SUCCESS && "Failed to initialize backend");

    // Create test circuit
    quantum_circuit_t* circuit = create_test_circuit_with_gates();
    add_test_gate(circuit, GATE_H, 0, NULL, 0);
    add_test_gate(circuit, GATE_H, 2, NULL, 0);

    // Execute circuit
    quantum_result result = {0};
    bool success = execute_circuit(&state, circuit, &result);
    assert(success && "Circuit execution failed");

    // Verify execution metrics
    assert(result.parallel_groups > 0 && "No parallel execution groups created");
    assert(result.execution_time >= 0 && "Invalid execution time");
    assert(result.num_measurements > 0 && "No measurements returned");
    assert(result.measurements && "Measurements not allocated");
    assert(result.probabilities && "Probabilities not allocated");

    // Cleanup
    free(result.measurements);
    free(result.probabilities);
    cleanup_circuit(circuit);
    cleanup_ibm_config(&config);
    printf("Circuit execution test passed\n");
}

static void test_error_mitigation() {
    printf("Testing error mitigation...\n");

    // Setup backend with error mitigation enabled
    IBMBackendConfig config = {0};
    config.backend_name = strdup("ibmq_qasm_simulator");
    config.optimization_level = 3;
    config.error_mitigation = true;

    IBMBackendState state = {0};
    qgt_error_t err = init_ibm_backend(&state, &config);
    assert(err == QGT_SUCCESS && "Failed to initialize backend");

    // Create test circuit with multiple gates
    quantum_circuit_t* circuit = create_test_circuit_with_gates();
    add_test_gate(circuit, GATE_H, 0, NULL, 0);
    add_test_gate(circuit, GATE_X, 1, NULL, 0);
    add_test_gate(circuit, GATE_Z, 2, NULL, 0);

    // Execute circuit
    quantum_result result = {0};
    bool success = execute_circuit(&state, circuit, &result);
    assert(success && "Circuit execution failed");

    // Verify error mitigation
    assert(result.mitigated_error_rate < result.raw_error_rate && "Error mitigation not effective");

    // Cleanup
    free(result.measurements);
    free(result.probabilities);
    cleanup_circuit(circuit);
    cleanup_ibm_config(&config);
    printf("Error mitigation test passed\n");
}

static void test_fast_feedback() {
    printf("Testing fast feedback...\n");

    // Setup backend
    IBMBackendConfig config = {0};
    config.backend_name = strdup("ibmq_qasm_simulator");
    config.optimization_level = 3;
    config.error_mitigation = true;

    IBMBackendState state = {0};
    qgt_error_t err = init_ibm_backend(&state, &config);
    assert(err == QGT_SUCCESS && "Failed to initialize backend");

    // Create simple circuit
    quantum_circuit_t* circuit = create_test_circuit_with_gates();
    add_test_gate(circuit, GATE_H, 0, NULL, 0);

    // Execute circuit
    quantum_result result = {0};
    bool success = execute_circuit(&state, circuit, &result);
    assert(success && "Circuit execution failed");

    // Verify fast feedback metrics
    assert(result.feedback_latency < 1e-3 && "Feedback latency too high");
    assert(result.conditional_success_rate > 0.99 && "Conditional operations not reliable");

    // Cleanup
    free(result.measurements);
    free(result.probabilities);
    cleanup_circuit(circuit);
    cleanup_ibm_config(&config);
    printf("Fast feedback test passed\n");
}

static void test_error_handling() {
    printf("Testing error handling...\n");

    // Test null pointers
    qgt_error_t err = init_ibm_backend(NULL, NULL);
    assert(err != QGT_SUCCESS && "Should fail with null pointers");

    // Test invalid config
    IBMBackendConfig invalid_config = {0};
    invalid_config.backend_name = NULL;

    IBMBackendState state = {0};
    err = init_ibm_backend(&state, &invalid_config);
    assert(err != QGT_SUCCESS && "Should fail with invalid config");

    // Test invalid circuit
    IBMBackendConfig valid_config = {0};
    valid_config.backend_name = strdup("ibmq_qasm_simulator");
    valid_config.optimization_level = 3;
    valid_config.error_mitigation = true;

    err = init_ibm_backend(&state, &valid_config);
    assert(err == QGT_SUCCESS && "Failed to initialize with valid config");

    quantum_result result = {0};
    bool success = execute_circuit(&state, NULL, &result);
    assert(!success && "Should fail with null circuit");

    cleanup_ibm_config(&valid_config);
    printf("Error handling test passed\n");
}

int main() {
    printf("Running IBM backend tests...\n\n");

    test_initialization();
    test_circuit_optimization();
    test_circuit_execution();
    test_error_mitigation();
    test_fast_feedback();
    test_error_handling();

    printf("\nAll IBM backend tests passed!\n");
    return 0;
}
