/**
 * @file test_z_stabilizer_operations.c
 * @brief Tests for Z-stabilizer measurement and optimization operations
 */

#include "quantum_geometric/physics/z_stabilizer_operations.h"
#include "quantum_geometric/core/error_codes.h"
#include "quantum_geometric/core/quantum_complex.h"
#include "quantum_geometric/physics/stabilizer_types.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <time.h>

// Error code constants
#define QGT_SUCCESS 0

// Test helper functions
static void test_initialization(void);
static void test_phase_error_mitigation(void);
static void test_z_measurement_correlation(void);
static void test_parallel_measurement(void);
static void test_phase_tracking(void);
static void test_hardware_optimization(void);
static void test_error_cases(void);
static void test_performance_requirements(void);
static void test_z_stabilizer_measurement_init(void);

// Helper to create test quantum state
static quantum_geometric_state_t* create_test_state(size_t num_qubits) {
    quantum_geometric_state_t* state = malloc(sizeof(quantum_geometric_state_t));
    state->dimension = num_qubits;
    state->manifold_dim = num_qubits;
    state->coordinates = calloc(num_qubits * 2, sizeof(ComplexFloat));
    state->metric = calloc(num_qubits * num_qubits, sizeof(ComplexFloat));
    state->connection = calloc(num_qubits * num_qubits * num_qubits, sizeof(ComplexFloat));
    state->is_normalized = true;
    
    // Initialize to |0⟩ state
    for (size_t i = 0; i < num_qubits; i++) {
        state->coordinates[2 * i].real = 1.0f;
        state->coordinates[2 * i].imag = 0.0f;
    }
    return state;
}

// Helper to create test stabilizer
static quantum_stabilizer_t* create_test_stabilizer(size_t num_qubits) {
    quantum_stabilizer_t* stabilizer = malloc(sizeof(quantum_stabilizer_t));
    stabilizer->type = STABILIZER_Z;
    stabilizer->dimension = num_qubits;
    stabilizer->num_qubits = num_qubits;
    stabilizer->qubit_indices = calloc(num_qubits, sizeof(size_t));
    stabilizer->coefficients = calloc(num_qubits, sizeof(ComplexFloat));
    stabilizer->is_hermitian = true;
    
    // Initialize to all Z operators
    for (size_t i = 0; i < num_qubits; i++) {
        stabilizer->qubit_indices[i] = i;
        stabilizer->coefficients[i].real = 1.0f;
        stabilizer->coefficients[i].imag = 0.0f;
    }
    return stabilizer;
}

// Helper to cleanup test state
static void cleanup_test_state(quantum_geometric_state_t* state) {
    if (state) {
        free(state->coordinates);
        free(state->metric);
        free(state->connection);
        free(state);
    }
}

// Helper to cleanup test stabilizer
static void cleanup_test_stabilizer(quantum_stabilizer_t* stabilizer) {
    if (stabilizer) {
        free(stabilizer->qubit_indices);
        free(stabilizer->coefficients);
        free(stabilizer);
    }
}

int main(void) {
    printf("Running Z-stabilizer operations tests...\n");

    // Run all tests
    test_initialization();
    test_phase_error_mitigation();
    test_z_measurement_correlation();
    test_parallel_measurement();
    test_phase_tracking();
    test_hardware_optimization();
    test_error_cases();
    test_performance_requirements();

    // Test Z stabilizer measurement initialization
    test_z_stabilizer_measurement_init();

    printf("All Z-stabilizer operations tests passed!\n");
    return 0;
}

static void test_z_stabilizer_measurement_init(void) {
    printf("Testing Z stabilizer measurement initialization...\n");

    // Test configuration
    ZStabilizerConfig config = {
        .repetition_count = 100,
        .error_threshold = 0.01,
        .confidence_threshold = 0.95,
        .history_capacity = 1000,
        .phase_calibration = 1.0,
        .echo_sequence_length = 8,
        .dynamic_phase_correction = true,
        .enable_z_optimization = true
    };

    ZHardwareConfig hardware = {
        .phase_calibration = 1.0,
        .z_gate_fidelity = 0.99,
        .measurement_fidelity = 0.98,
        .echo_sequence_length = 8,
        .dynamic_phase_correction = true
    };

    // Test initialization
    ZStabilizerState* state = init_z_stabilizer_measurement(&config, &hardware);
    assert(state != NULL);
    assert(state->config.repetition_count == config.repetition_count);
    assert(state->config.error_threshold == config.error_threshold);
    assert(state->config.confidence_threshold == config.confidence_threshold);
    assert(state->config.history_capacity == config.history_capacity);
    assert(state->config.phase_calibration == config.phase_calibration);
    assert(state->config.echo_sequence_length == config.echo_sequence_length);
    assert(state->config.dynamic_phase_correction == config.dynamic_phase_correction);
    assert(state->config.enable_z_optimization == config.enable_z_optimization);

    // Test array allocations
    assert(state->phase_correlations != NULL);
    assert(state->measurement_confidences != NULL);
    assert(state->measurement_history != NULL);
    assert(state->stabilizer_values != NULL);

    // Test initial values
    assert(state->history_size == 0);
    assert(state->phase_error_rate == 0.0);

    // Test cleanup
    cleanup_z_stabilizer_measurement(state);

    // Test error cases
    assert(init_z_stabilizer_measurement(NULL, &hardware) == NULL);
    assert(init_z_stabilizer_measurement(&config, NULL) == NULL);

    printf("Z stabilizer measurement initialization tests passed\n");
}

static void test_initialization(void) {
    printf("Testing initialization...\n");

    // Test valid initialization
    z_measurement_t* measurement = NULL;
    qgt_error_t err = z_measurement_create(&measurement);
    assert(err == QGT_SUCCESS);
    assert(measurement != NULL);
    assert(measurement->value == 1);
    assert(measurement->confidence == 1.0);
    assert(measurement->error_rate == 0.0);
    assert(measurement->needs_correction == false);
    assert(measurement->auxiliary_data == NULL);

    z_measurement_destroy(measurement);

    // Test invalid parameters
    err = z_measurement_create(NULL);
    assert(err == QGT_ERROR_INVALID_ARGUMENT);

    printf("Initialization tests passed\n");
}

static void test_phase_error_mitigation(void) {
    printf("Testing phase error mitigation...\n");

    // Create test state and stabilizer
    quantum_geometric_state_t* state = create_test_state(4);
    quantum_stabilizer_t* stabilizer = create_test_stabilizer(4);
    z_measurement_t* measurement = NULL;
    
    qgt_error_t err = z_measurement_create(&measurement);
    assert(err == QGT_SUCCESS);

    // Test measurement with error mitigation
    err = z_stabilizer_measure(measurement, stabilizer, state);
    assert(err == QGT_SUCCESS);
    
    // Verify error mitigation effects
    assert(measurement->error_rate < 0.2);  // Error should be reduced
    assert(measurement->confidence > 0.8);  // Confidence should be high

    // Cleanup
    z_measurement_destroy(measurement);
    cleanup_test_state(state);
    cleanup_test_stabilizer(stabilizer);
    
    printf("Phase error mitigation tests passed\n");
}

static void test_z_measurement_correlation(void) {
    printf("Testing Z-measurement correlation...\n");

    // Create test states and stabilizers
    quantum_geometric_state_t* state = create_test_state(4);
    quantum_stabilizer_t* stabilizer1 = create_test_stabilizer(2);
    quantum_stabilizer_t* stabilizer2 = create_test_stabilizer(2);
    double correlation = 0.0;

    // Test correlation between nearby stabilizers
    qgt_error_t err = z_stabilizer_correlation(&correlation, stabilizer1, stabilizer2, state);
    assert(err == QGT_SUCCESS);
    assert(fabs(correlation) <= 1.0);

    // Test commutation
    bool commute = false;
    err = z_stabilizer_commute(&commute, stabilizer1, stabilizer2);
    assert(err == QGT_SUCCESS);
    assert(commute == true);  // Z stabilizers always commute

    // Cleanup
    cleanup_test_state(state);
    cleanup_test_stabilizer(stabilizer1);
    cleanup_test_stabilizer(stabilizer2);
    
    printf("Z-measurement correlation tests passed\n");
}

static void test_parallel_measurement(void) {
    printf("Testing parallel measurement...\n");

    // Create test state and stabilizers
    quantum_geometric_state_t* state = create_test_state(8);
    quantum_stabilizer_t* stabilizers[4];
    z_measurement_t* measurements[4] = {NULL};

    // Initialize stabilizers and measurements
    for (int i = 0; i < 4; i++) {
        stabilizers[i] = create_test_stabilizer(2);
        qgt_error_t err = z_measurement_create(&measurements[i]);
        assert(err == QGT_SUCCESS);
    }

    // Perform parallel measurements
    for (int i = 0; i < 4; i++) {
        qgt_error_t err = z_stabilizer_measure(measurements[i], stabilizers[i], state);
        assert(err == QGT_SUCCESS);
        assert(abs(measurements[i]->value) == 1);
    }

    // Cleanup
    for (int i = 0; i < 4; i++) {
        z_measurement_destroy(measurements[i]);
        cleanup_test_stabilizer(stabilizers[i]);
    }
    cleanup_test_state(state);
    
    printf("Parallel measurement tests passed\n");
}

static void test_phase_tracking(void) {
    printf("Testing phase tracking...\n");

    // Create test state and stabilizer
    quantum_geometric_state_t* state = create_test_state(4);
    quantum_stabilizer_t* stabilizer = create_test_stabilizer(4);
    z_measurement_t* measurement = NULL;
    
    qgt_error_t err = z_measurement_create(&measurement);
    assert(err == QGT_SUCCESS);

    // Test phase tracking through multiple measurements
    for (int i = 0; i < 10; i++) {
        err = z_stabilizer_measure(measurement, stabilizer, state);
        assert(err == QGT_SUCCESS);
        
        // Verify measurement properties
        assert(measurement->value == 1 || measurement->value == -1);
        assert(measurement->confidence >= 0.0 && measurement->confidence <= 1.0);
        assert(measurement->error_rate >= 0.0 && measurement->error_rate <= 1.0);
    }

    // Cleanup
    z_measurement_destroy(measurement);
    cleanup_test_state(state);
    cleanup_test_stabilizer(stabilizer);
    
    printf("Phase tracking tests passed\n");
}

static void test_hardware_optimization(void) {
    printf("Testing hardware optimization...\n");

    // Create test state and stabilizer
    quantum_geometric_state_t* state = create_test_state(4);
    quantum_stabilizer_t* stabilizer = create_test_stabilizer(4);
    z_measurement_t* measurement = NULL;
    
    qgt_error_t err = z_measurement_create(&measurement);
    assert(err == QGT_SUCCESS);

    // Test measurement with hardware optimization
    err = z_stabilizer_measure(measurement, stabilizer, state);
    assert(err == QGT_SUCCESS);

    // Verify optimized measurement quality
    assert(measurement->confidence > 0.95);  // Should be very high with optimization
    assert(measurement->error_rate < 0.05);  // Should be very low with optimization

    // Cleanup
    z_measurement_destroy(measurement);
    cleanup_test_state(state);
    cleanup_test_stabilizer(stabilizer);
    
    printf("Hardware optimization tests passed\n");
}

static void test_error_cases(void) {
    printf("Testing error cases...\n");

    // Test NULL parameters
    qgt_error_t err = z_measurement_create(NULL);
    assert(err == QGT_ERROR_INVALID_ARGUMENT);

    z_measurement_t* measurement = NULL;
    err = z_measurement_create(&measurement);
    assert(err == QGT_SUCCESS);

    // Test invalid measurement parameters
    err = z_stabilizer_measure(NULL, NULL, NULL);
    assert(err == QGT_ERROR_INVALID_ARGUMENT);

    err = z_stabilizer_measure(measurement, NULL, NULL);
    assert(err == QGT_ERROR_INVALID_ARGUMENT);

    // Test invalid correlation parameters
    double correlation;
    err = z_stabilizer_correlation(NULL, NULL, NULL, NULL);
    assert(err == QGT_ERROR_INVALID_ARGUMENT);

    // Test invalid validation parameters
    err = z_measurement_validate(NULL);
    assert(err == QGT_ERROR_INVALID_ARGUMENT);

    // Test invalid state
    measurement->value = 0;  // Invalid value
    err = z_measurement_validate(measurement);
    assert(err == QGT_ERROR_INVALID_STATE);

    // Cleanup
    z_measurement_destroy(measurement);
    
    printf("Error case tests passed\n");
}

static void test_performance_requirements(void) {
    printf("Testing performance requirements...\n");

    // Test creation performance
    #include <time.h>
    
    clock_t start = clock();
    z_measurement_t* measurement = NULL;
    qgt_error_t err = z_measurement_create(&measurement);
    clock_t end = clock();
    double init_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    assert(err == QGT_SUCCESS);
    assert(init_time < 0.001);  // Should initialize quickly

    // Test measurement performance
    quantum_geometric_state_t* state = create_test_state(4);
    quantum_stabilizer_t* stabilizer = create_test_stabilizer(4);
    
    start = clock();
    err = z_stabilizer_measure(measurement, stabilizer, state);
    end = clock();
    double measurement_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    assert(err == QGT_SUCCESS);
    assert(measurement_time < 0.0001);  // Should measure quickly

    // Verify memory usage
    size_t measurement_size = sizeof(z_measurement_t);
    assert(measurement_size < 1024);  // Should be small

    // Cleanup
    z_measurement_destroy(measurement);
    cleanup_test_state(state);
    cleanup_test_stabilizer(stabilizer);
    
    printf("Performance requirement tests passed\n");
}
