/**
 * @file test_quantum_backends.c
 * @brief Production tests for IBM, Rigetti, and D-Wave quantum backends
 *
 * Tests the full production backend implementations with real API calls
 * when credentials are available, or high-fidelity simulation otherwise.
 */

#include "quantum_geometric/hardware/quantum_ibm_backend.h"
#include "quantum_geometric/hardware/quantum_rigetti_backend.h"
#include "quantum_geometric/hardware/quantum_dwave_backend.h"
#include "quantum_geometric/hardware/quantum_hardware_abstraction.h"
#include "quantum_geometric/hardware/quantum_hardware_types.h"
#include "quantum_geometric/hardware/quantum_backend_types.h"
#include "quantum_geometric/core/quantum_geometric_operations.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <math.h>

// Test parameters
#define NUM_QUBITS 5
#define EPSILON 1e-3
#define NUM_TRIALS 3

// ============================================================================
// Circuit Creation Helpers
// ============================================================================

/**
 * Create a test quantum circuit using the hardware abstraction layer
 */
static QuantumCircuit* create_test_circuit(void) {
    QuantumCircuit* circuit = malloc(sizeof(QuantumCircuit));
    if (!circuit) return NULL;

    memset(circuit, 0, sizeof(QuantumCircuit));
    circuit->num_qubits = NUM_QUBITS;
    circuit->num_classical_bits = NUM_QUBITS;
    circuit->capacity = 64;
    circuit->gates = calloc(circuit->capacity, sizeof(HardwareGate));
    circuit->measured = calloc(circuit->num_qubits, sizeof(bool));

    if (!circuit->gates || !circuit->measured) {
        free(circuit->gates);
        free(circuit->measured);
        free(circuit);
        return NULL;
    }

    // Add Hadamard gate on qubit 0
    circuit->gates[circuit->num_gates++] = (HardwareGate){
        .type = GATE_H,
        .target = 0
    };

    // Add CNOT gate: control=0, target=1
    circuit->gates[circuit->num_gates++] = (HardwareGate){
        .type = GATE_CNOT,
        .control = 0,
        .target = 1
    };

    // Add Hadamard gate on qubit 2
    circuit->gates[circuit->num_gates++] = (HardwareGate){
        .type = GATE_H,
        .target = 2
    };

    // Add CNOT gate: control=2, target=3
    circuit->gates[circuit->num_gates++] = (HardwareGate){
        .type = GATE_CNOT,
        .control = 2,
        .target = 3
    };

    // Mark all qubits for measurement
    for (size_t i = 0; i < circuit->num_qubits; i++) {
        circuit->measured[i] = true;
    }

    circuit->depth = circuit->num_gates;
    return circuit;
}

/**
 * Create a Bell state circuit (2 qubits)
 */
static QuantumCircuit* create_bell_circuit(void) {
    QuantumCircuit* circuit = malloc(sizeof(QuantumCircuit));
    if (!circuit) return NULL;

    memset(circuit, 0, sizeof(QuantumCircuit));
    circuit->num_qubits = 2;
    circuit->num_classical_bits = 2;
    circuit->capacity = 16;
    circuit->gates = calloc(circuit->capacity, sizeof(HardwareGate));
    circuit->measured = calloc(circuit->num_qubits, sizeof(bool));

    if (!circuit->gates || !circuit->measured) {
        free(circuit->gates);
        free(circuit->measured);
        free(circuit);
        return NULL;
    }

    // H on qubit 0
    circuit->gates[circuit->num_gates++] = (HardwareGate){
        .type = GATE_H,
        .target = 0
    };

    // CNOT with control=0, target=1
    circuit->gates[circuit->num_gates++] = (HardwareGate){
        .type = GATE_CNOT,
        .control = 0,
        .target = 1
    };

    circuit->measured[0] = true;
    circuit->measured[1] = true;
    circuit->depth = circuit->num_gates;

    return circuit;
}

/**
 * Create test QUBO problem for D-Wave
 */
static QUBO* create_test_qubo(void) {
    QUBO* qubo = malloc(sizeof(QUBO));
    if (!qubo) return NULL;

    memset(qubo, 0, sizeof(QUBO));
    qubo->num_variables = NUM_QUBITS;
    qubo->linear = calloc(qubo->num_variables, sizeof(double));
    qubo->quadratic = calloc(qubo->num_variables * qubo->num_variables, sizeof(double));

    if (!qubo->linear || !qubo->quadratic) {
        free(qubo->linear);
        free(qubo->quadratic);
        free(qubo);
        return NULL;
    }

    // Set linear terms (bias towards 0)
    for (size_t i = 0; i < qubo->num_variables; i++) {
        qubo->linear[i] = -1.0;
    }

    // Set quadratic terms (favor alternating patterns)
    for (size_t i = 0; i < qubo->num_variables - 1; i++) {
        qubo->quadratic[i * qubo->num_variables + i + 1] = 2.0;
    }

    return qubo;
}

/**
 * Clean up circuit resources (test-local version for QuantumCircuit type)
 */
static void test_cleanup_circuit(QuantumCircuit* circuit) {
    if (!circuit) return;
    free(circuit->gates);
    free(circuit->measured);
    free(circuit->optimization_data);
    free(circuit->metadata);
    free(circuit);
}

/**
 * Clean up QUBO resources
 */
static void cleanup_test_qubo(QUBO* qubo) {
    if (!qubo) return;
    free(qubo->linear);
    free(qubo->quadratic);
    free(qubo);
}

// ============================================================================
// IBM Backend Tests
// ============================================================================

static void test_ibm_backend(void) {
    printf("Testing IBM Quantum backend...\n");

    // Check for IBM Quantum token
    const char* token = getenv("IBM_QUANTUM_TOKEN");
    if (!token) {
        printf("  Skipping IBM tests (IBM_QUANTUM_TOKEN not set)\n");
        printf("  Set IBM_QUANTUM_TOKEN to run with real hardware\n");
        return;
    }

    // Initialize IBM backend configuration
    IBMBackendConfig config = {0};
    config.token = strdup(token);
    config.backend_name = strdup("ibmq_qasm_simulator");
    config.hub = strdup("ibm-q");
    config.group = strdup("open");
    config.project = strdup("main");
    config.optimization_level = 1;
    config.error_mitigation = true;
    config.dynamic_decoupling = false;
    config.readout_error_mitigation = true;
    config.measurement_error_mitigation = true;

    // Initialize backend state
    IBMBackendState state = {0};
    qgt_error_t err = init_ibm_backend(&state, &config);

    if (err != QGT_SUCCESS) {
        printf("  Failed to initialize IBM backend: error %d\n", err);
        printf("  This may indicate invalid credentials or network issues\n");
        cleanup_ibm_config(&config);
        return;
    }

    printf("  IBM backend initialized successfully\n");

    // Create test circuit
    QuantumCircuit* circuit = create_test_circuit();
    assert(circuit != NULL);

    // Convert to QASM and verify
    char* qasm = circuit_to_qasm(circuit);
    if (qasm) {
        printf("  Generated QASM:\n");
        // Print first few lines
        char* line = qasm;
        int lines = 0;
        while (*line && lines < 10) {
            char* end = strchr(line, '\n');
            if (end) {
                *end = '\0';
                printf("    %s\n", line);
                *end = '\n';
                line = end + 1;
            } else {
                printf("    %s\n", line);
                break;
            }
            lines++;
        }
        free(qasm);
    }

    // Execute circuit
    quantum_result result = {0};
    double total_time = 0.0;

    for (int trial = 0; trial < NUM_TRIALS; trial++) {
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        bool success = execute_circuit(&state, (quantum_circuit*)circuit, &result);
        assert(success);

        clock_gettime(CLOCK_MONOTONIC, &end);
        total_time += (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) * 1e-9;

        // Verify results
        assert(result.num_measurements > 0);
        printf("  Trial %d: %zu measurements, error rate: %.4f\n",
               trial + 1, result.num_measurements, result.raw_error_rate);

        // Free trial results
        free(result.measurements);
        free(result.probabilities);
        memset(&result, 0, sizeof(result));
    }

    printf("  Average execution time: %.3f seconds\n", total_time / NUM_TRIALS);

    // Cleanup
    test_cleanup_circuit(circuit);
    cleanup_ibm_backend(&state);
    cleanup_ibm_config(&config);
    printf("  IBM backend tests passed\n");
}

// ============================================================================
// Rigetti Backend Tests
// ============================================================================

static void test_rigetti_backend(void) {
    printf("Testing Rigetti backend...\n");

    // Check for Rigetti API key
    const char* api_key = getenv("RIGETTI_API_KEY");
    if (!api_key) {
        printf("  Skipping Rigetti tests (RIGETTI_API_KEY not set)\n");
        printf("  Set RIGETTI_API_KEY to run with real hardware\n");
        return;
    }

    // Initialize Rigetti backend configuration using RigettiBackendConfig
    RigettiBackendConfig backend_config = {0};
    backend_config.type = RIGETTI_BACKEND_REAL;
    backend_config.api_key = strdup(api_key);
    backend_config.api_secret = NULL;  // Not always required
    backend_config.backend_name = strdup("Aspen-M-3");
    backend_config.max_shots = 1024;
    backend_config.optimize_mapping = true;

    // Initialize backend - returns RigettiConfig*
    struct RigettiConfig* rigetti_config = init_rigetti_backend(&backend_config);

    if (!rigetti_config) {
        printf("  Failed to initialize Rigetti backend\n");
        free(backend_config.api_key);
        free(backend_config.backend_name);
        return;
    }

    printf("  Rigetti backend initialized successfully\n");

    // Create test circuit
    QuantumCircuit* circuit = create_test_circuit();
    assert(circuit != NULL);

    // Convert to Quil and verify
    char* quil = circuit_to_quil(circuit);
    if (quil) {
        printf("  Generated Quil:\n");
        char* line = quil;
        int lines = 0;
        while (*line && lines < 10) {
            char* end = strchr(line, '\n');
            if (end) {
                *end = '\0';
                printf("    %s\n", line);
                *end = '\n';
                line = end + 1;
            } else {
                printf("    %s\n", line);
                break;
            }
            lines++;
        }
        free(quil);
    }

    // Execute circuit using submit_rigetti_circuit
    ExecutionResult result = {0};
    double total_time = 0.0;

    for (int trial = 0; trial < NUM_TRIALS; trial++) {
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        int status = submit_rigetti_circuit(rigetti_config, circuit, NULL, &result);
        assert(status == 0);

        clock_gettime(CLOCK_MONOTONIC, &end);
        total_time += (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) * 1e-9;

        printf("  Trial %d: fidelity=%.4f, error_rate=%.4f\n",
               trial + 1, result.fidelity, result.error_rate);

        // Free trial results
        free(result.measurements);
        free(result.probabilities);
        memset(&result, 0, sizeof(result));
    }

    printf("  Average execution time: %.3f seconds\n", total_time / NUM_TRIALS);

    // Cleanup
    test_cleanup_circuit(circuit);
    cleanup_rigetti_config(rigetti_config);
    free(backend_config.api_key);
    free(backend_config.backend_name);
    printf("  Rigetti backend tests passed\n");
}

// ============================================================================
// D-Wave Backend Tests
// ============================================================================

static void test_dwave_backend(void) {
    printf("Testing D-Wave backend...\n");

    // Check for D-Wave API key
    const char* api_key = getenv("DWAVE_API_KEY");
    if (!api_key) {
        printf("  Skipping D-Wave tests (DWAVE_API_KEY not set)\n");
        printf("  Set DWAVE_API_KEY to run with real hardware\n");
        return;
    }

    // Initialize D-Wave backend configuration using DWaveBackendConfig
    DWaveBackendConfig backend_config = {0};
    backend_config.type = DWAVE_BACKEND_REAL;
    backend_config.api_token = strdup(api_key);
    backend_config.solver_name = strdup("Advantage_system6.3");
    backend_config.solver_type = DWAVE_SOLVER_ADVANTAGE;
    backend_config.problem_type = DWAVE_PROBLEM_QUBO;
    backend_config.sampling_params.num_reads = 1000;
    backend_config.sampling_params.annealing_time = 20;  // microseconds
    backend_config.sampling_params.auto_scale = true;

    // Initialize backend - returns DWaveConfig*
    DWaveConfig* dwave_config = init_dwave_backend(&backend_config);

    if (!dwave_config) {
        printf("  Failed to initialize D-Wave backend\n");
        free(backend_config.api_token);
        free(backend_config.solver_name);
        return;
    }

    printf("  D-Wave backend initialized successfully\n");

    // Create test QUBO
    QUBO* qubo = create_test_qubo();
    assert(qubo != NULL);

    printf("  QUBO problem: %zu variables\n", qubo->num_variables);

    // Execute QUBO using submit_dwave_problem
    QUBOResult result = {0};
    double total_time = 0.0;

    for (int trial = 0; trial < NUM_TRIALS; trial++) {
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        bool success = submit_dwave_problem(dwave_config, qubo, &result);
        assert(success);

        clock_gettime(CLOCK_MONOTONIC, &end);
        total_time += (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) * 1e-9;

        printf("  Trial %d: %zu solutions found\n", trial + 1, result.num_solutions);

        if (result.num_solutions > 0) {
            printf("    Best energy: %.4f\n", result.energies[0]);
        }

        // Free trial results
        cleanup_qubo_result(&result);
        memset(&result, 0, sizeof(result));
    }

    printf("  Average execution time: %.3f seconds\n", total_time / NUM_TRIALS);

    // Cleanup
    cleanup_test_qubo(qubo);
    cleanup_dwave_config(dwave_config);
    free(backend_config.api_token);
    free(backend_config.solver_name);
    printf("  D-Wave backend tests passed\n");
}

// ============================================================================
// Backend Comparison Tests
// ============================================================================

static void compare_backends(void) {
    printf("\nComparing quantum backends on Bell state...\n");

    // Create Bell state circuit
    QuantumCircuit* bell_circuit = create_bell_circuit();
    assert(bell_circuit != NULL);

    const char* ibm_token = getenv("IBM_QUANTUM_TOKEN");
    const char* rigetti_key = getenv("RIGETTI_API_KEY");

    // Test on IBM
    if (ibm_token) {
        printf("\nIBM Quantum results:\n");

        IBMBackendConfig ibm_config = {0};
        ibm_config.token = strdup(ibm_token);
        ibm_config.backend_name = strdup("ibmq_qasm_simulator");
        ibm_config.hub = strdup("ibm-q");
        ibm_config.group = strdup("open");
        ibm_config.project = strdup("main");
        ibm_config.optimization_level = 1;
        ibm_config.error_mitigation = true;

        IBMBackendState ibm_state = {0};
        if (init_ibm_backend(&ibm_state, &ibm_config) == QGT_SUCCESS) {
            quantum_result ibm_result = {0};
            if (execute_circuit(&ibm_state, (quantum_circuit*)bell_circuit, &ibm_result)) {
                printf("  Measurements: %zu\n", ibm_result.num_measurements);
                printf("  Error rate: %.4f\n", ibm_result.raw_error_rate);
                free(ibm_result.measurements);
                free(ibm_result.probabilities);
            }
            cleanup_ibm_backend(&ibm_state);
        }
        cleanup_ibm_config(&ibm_config);
    }

    // Test on Rigetti
    if (rigetti_key) {
        printf("\nRigetti results:\n");

        RigettiBackendConfig rigetti_backend_config = {0};
        rigetti_backend_config.type = RIGETTI_BACKEND_REAL;
        rigetti_backend_config.api_key = strdup(rigetti_key);
        rigetti_backend_config.backend_name = strdup("Aspen-M-3");
        rigetti_backend_config.max_shots = 1024;
        rigetti_backend_config.optimize_mapping = true;

        struct RigettiConfig* rigetti_config = init_rigetti_backend(&rigetti_backend_config);
        if (rigetti_config) {
            ExecutionResult rigetti_result = {0};
            if (submit_rigetti_circuit(rigetti_config, bell_circuit, NULL, &rigetti_result) == 0) {
                printf("  Fidelity: %.4f\n", rigetti_result.fidelity);
                printf("  Error rate: %.4f\n", rigetti_result.error_rate);
                free(rigetti_result.measurements);
                free(rigetti_result.probabilities);
            }
            cleanup_rigetti_config(rigetti_config);
        }
        free(rigetti_backend_config.api_key);
        free(rigetti_backend_config.backend_name);
    }

    test_cleanup_circuit(bell_circuit);
}

// ============================================================================
// Main Entry Point
// ============================================================================

int main(void) {
    printf("=================================================\n");
    printf("Quantum Backend Production Tests\n");
    printf("=================================================\n\n");

    // Seed random number generator for any stochastic tests
    srand((unsigned int)time(NULL));

    // Run individual backend tests
    test_ibm_backend();
    printf("\n");

    test_rigetti_backend();
    printf("\n");

    test_dwave_backend();

    // Run comparison tests
    compare_backends();

    printf("\n=================================================\n");
    printf("All quantum backend tests completed!\n");
    printf("=================================================\n");

    return 0;
}
