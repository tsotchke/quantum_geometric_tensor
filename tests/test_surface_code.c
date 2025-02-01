/**
 * @file test_surface_code.c
 * @brief Test suite for surface code implementation
 */

#include "quantum_geometric/physics/surface_code.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

// Test configurations
static const size_t TEST_DISTANCE = 3;
static const double TEST_THRESHOLD = 0.1;

// Helper function to create test configuration
static SurfaceConfig create_test_config(void) {
    SurfaceConfig config = {
        .type = SURFACE_CODE_STANDARD,
        .distance = TEST_DISTANCE,
        .width = TEST_DISTANCE * 2 + 1,
        .height = TEST_DISTANCE * 2 + 1,
        .threshold = TEST_THRESHOLD,
        .measurement_error_rate = 0.01,
        .error_weight_factor = 0.5,
        .correlation_factor = 0.8,
        .use_metal_acceleration = true,
        .time_steps = 4
    };
    return config;
}

// Test surface code initialization
static void test_initialization(void) {
    printf("Testing surface code initialization...\n");

    SurfaceConfig config = create_test_config();
    SurfaceCode* state = init_surface_code(&config);
    assert(state != NULL);
    assert(state->initialized);
    assert(state->num_stabilizers > 0);
    assert(state->num_stabilizers <= MAX_STABILIZERS);
    assert(state->num_logical_qubits == 0);

    // Verify configuration copied correctly
    assert(state->config.type == config.type);
    assert(state->config.distance == config.distance);
    assert(state->config.width == config.width);
    assert(state->config.height == config.height);
    assert(fabs(state->config.threshold - config.threshold) < 1e-6);

    cleanup_surface_code(state);
    printf("Surface code initialization test passed\n");
}

// Test stabilizer measurement
static void test_stabilizer_measurement(void) {
    printf("Testing stabilizer measurement...\n");

    SurfaceConfig config = create_test_config();
    SurfaceCode* state = init_surface_code(&config);
    assert(state != NULL);

    // Allocate results array
    StabilizerResult* results = calloc(state->num_stabilizers, sizeof(StabilizerResult));
    assert(results != NULL);

    // Perform measurements
    size_t num_measurements = measure_stabilizers(state, results);
    assert(num_measurements == state->num_stabilizers);

    // Verify results
    for (size_t i = 0; i < num_measurements; i++) {
        assert(results[i].value == 1 || results[i].value == -1);
        assert(results[i].confidence >= 0.0 && results[i].confidence <= 1.0);
        
        // Verify stabilizer state updated
        const Stabilizer* stabilizer = get_stabilizer(state, i);
        assert(stabilizer != NULL);
        assert(stabilizer->result.value == results[i].value);
        assert(fabs(stabilizer->result.confidence - results[i].confidence) < 1e-6);
    }

    free(results);
    cleanup_surface_code(state);
    printf("Stabilizer measurement test passed\n");
}

// Test error correction
static void test_error_correction(void) {
    printf("Testing error correction...\n");

    SurfaceConfig config = create_test_config();
    SurfaceCode* state = init_surface_code(&config);
    assert(state != NULL);

    // Create test syndromes
    SyndromeVertex syndromes[4] = {
        {.x = 1.0, .y = 1.0, .z = 0.0, .weight = 1.0, .matched = false, .time = 0},
        {.x = 3.0, .y = 1.0, .z = 0.0, .weight = 1.0, .matched = false, .time = 0},
        {.x = 1.0, .y = 3.0, .z = 0.0, .weight = 1.0, .matched = false, .time = 0},
        {.x = 3.0, .y = 3.0, .z = 0.0, .weight = 1.0, .matched = false, .time = 0}
    };

    // Apply corrections
    size_t corrections = apply_corrections(state, syndromes, 4);
    assert(corrections > 0);

    // Verify error rates updated
    assert(state->total_error_rate >= 0.0 && state->total_error_rate <= 1.0);

    // Verify stabilizer states
    for (size_t i = 0; i < state->num_stabilizers; i++) {
        const Stabilizer* stabilizer = get_stabilizer(state, i);
        assert(stabilizer != NULL);
        assert(stabilizer->error_rate >= 0.0 && stabilizer->error_rate <= 1.0);
    }

    cleanup_surface_code(state);
    printf("Error correction test passed\n");
}

// Test logical qubit operations
static void test_logical_qubit_operations(void) {
    printf("Testing logical qubit operations...\n");

    SurfaceConfig config = create_test_config();
    SurfaceCode* state = init_surface_code(&config);
    assert(state != NULL);

    // Create test data qubits
    size_t data_qubits[] = {0, 1, 2, 3, 4};
    int logical_idx = encode_logical_qubit(state, data_qubits, 5);
    assert(logical_idx >= 0);
    assert(logical_idx < MAX_LOGICAL_QUBITS);

    // Verify logical qubit state
    const LogicalQubit* logical = get_logical_qubit(state, logical_idx);
    assert(logical != NULL);
    assert(logical->num_data_qubits == 5);
    assert(logical->num_stabilizers > 0);
    assert(logical->logical_error_rate >= 0.0 && logical->logical_error_rate <= 1.0);

    // Measure logical qubit
    StabilizerResult result;
    bool measure_success = measure_logical_qubit(state, logical_idx, &result);
    assert(measure_success);
    assert(result.value == 1 || result.value == -1);
    assert(result.confidence >= 0.0 && result.confidence <= 1.0);

    cleanup_surface_code(state);
    printf("Logical qubit operations test passed\n");
}

// Test error rate tracking
static void test_error_rate_tracking(void) {
    printf("Testing error rate tracking...\n");

    SurfaceConfig config = create_test_config();
    SurfaceCode* state = init_surface_code(&config);
    assert(state != NULL);

    // Create test measurements
    StabilizerResult* measurements = calloc(state->num_stabilizers, sizeof(StabilizerResult));
    assert(measurements != NULL);

    for (size_t i = 0; i < state->num_stabilizers; i++) {
        measurements[i].value = 1;
        measurements[i].confidence = 0.9;
        measurements[i].needs_correction = false;
    }

    // Update error rates
    double total_rate = update_error_rates(state, measurements, state->num_stabilizers);
    assert(total_rate >= 0.0 && total_rate <= 1.0);
    assert(fabs(total_rate - state->total_error_rate) < 1e-6);

    // Verify threshold check
    bool below_threshold = check_error_threshold(state);
    assert(below_threshold == (total_rate < config.threshold));

    free(measurements);
    cleanup_surface_code(state);
    printf("Error rate tracking test passed\n");
}

// Test different lattice types
static void test_lattice_types(void) {
    printf("Testing different lattice types...\n");

    SurfaceConfig config = create_test_config();
    SurfaceCode* state;

    // Test rotated lattice
    config.type = SURFACE_CODE_ROTATED;
    state = init_surface_code(&config);
    assert(state != NULL);
    assert(state->initialized);
    cleanup_surface_code(state);

    // Test heavy hex lattice
    config.type = SURFACE_CODE_HEAVY_HEX;
    state = init_surface_code(&config);
    assert(state != NULL);
    assert(state->initialized);
    cleanup_surface_code(state);

    // Test Floquet lattice
    config.type = SURFACE_CODE_FLOQUET;
    state = init_surface_code(&config);
    assert(state != NULL);
    assert(state->initialized);
    cleanup_surface_code(state);

    printf("Lattice types test passed\n");
}

// Test Metal acceleration
static void test_metal_acceleration(void) {
    printf("Testing Metal acceleration...\n");

    SurfaceConfig config = create_test_config();
    config.use_metal_acceleration = true;
    SurfaceCode* state = init_surface_code(&config);
    assert(state != NULL);

    // Perform measurements with Metal acceleration
    StabilizerResult* results = calloc(state->num_stabilizers, sizeof(StabilizerResult));
    assert(results != NULL);

    size_t num_measurements = measure_stabilizers(state, results);
    assert(num_measurements == state->num_stabilizers);

    // Verify results still valid with Metal
    for (size_t i = 0; i < num_measurements; i++) {
        assert(results[i].value == 1 || results[i].value == -1);
        assert(results[i].confidence >= 0.0 && results[i].confidence <= 1.0);
    }

    free(results);
    cleanup_surface_code(state);
    printf("Metal acceleration test passed\n");
}

// Test configuration validation
static void test_config_validation(void) {
    printf("Testing configuration validation...\n");

    // Test invalid distance
    SurfaceConfig invalid_config = create_test_config();
    invalid_config.distance = 2; // Must be odd and >= 3
    SurfaceCode* state = init_surface_code(&invalid_config);
    assert(state == NULL);

    // Test invalid dimensions
    invalid_config = create_test_config();
    invalid_config.width = MAX_SURFACE_SIZE + 1;
    state = init_surface_code(&invalid_config);
    assert(state == NULL);

    // Test invalid threshold
    invalid_config = create_test_config();
    invalid_config.threshold = 1.5; // Must be between 0 and 1
    state = init_surface_code(&invalid_config);
    assert(state == NULL);

    // Test invalid error rates
    invalid_config = create_test_config();
    invalid_config.measurement_error_rate = -0.1;
    state = init_surface_code(&invalid_config);
    assert(state == NULL);

    // Test valid config
    SurfaceConfig valid_config = create_test_config();
    state = init_surface_code(&valid_config);
    assert(state != NULL);
    cleanup_surface_code(state);

    printf("Configuration validation test passed\n");
}

// Test stabilizer configuration
static void test_stabilizer_configuration(void) {
    printf("Testing stabilizer configuration...\n");

    SurfaceConfig config = create_test_config();
    SurfaceCode* state = init_surface_code(&config);
    assert(state != NULL);

    // Verify standard lattice setup
    for (size_t i = 0; i < state->num_stabilizers; i++) {
        const Stabilizer* stabilizer = get_stabilizer(state, i);
        assert(stabilizer != NULL);
        assert(stabilizer->num_qubits >= 2 && stabilizer->num_qubits <= 4);
        assert(stabilizer->type == STABILIZER_X || stabilizer->type == STABILIZER_Z);

        // Verify qubit indices
        for (size_t j = 0; j < stabilizer->num_qubits; j++) {
            assert(stabilizer->qubits[j] < config.width * config.height);
        }

        // Verify initial state
        assert(stabilizer->error_rate >= 0.0 && stabilizer->error_rate <= 1.0);
        assert(stabilizer->result.confidence >= 0.0 && stabilizer->result.confidence <= 1.0);
    }

    cleanup_surface_code(state);
    printf("Stabilizer configuration test passed\n");
}

int main(void) {
    printf("Running surface code tests...\n\n");

    test_initialization();
    test_stabilizer_measurement();
    test_error_correction();
    test_logical_qubit_operations();
    test_error_rate_tracking();
    test_lattice_types();
    test_metal_acceleration();
    test_config_validation();
    test_stabilizer_configuration();

    printf("\nAll surface code tests passed!\n");
    return 0;
}
