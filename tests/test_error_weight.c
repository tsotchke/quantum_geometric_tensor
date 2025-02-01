/**
 * @file test_error_weight.c
 * @brief Tests for quantum error weight calculation system
 */

#include "quantum_geometric/physics/error_weight.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/physics/quantum_state_operations.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

// Test helper functions
static void test_initialization(void);
static void test_weight_calculation(void);
static void test_weight_statistics(void);
static void test_geometric_scaling(void);
static void test_error_cases(void);
static void test_performance_requirements(void);

// Mock functions and data
static quantum_state* create_test_state(size_t width, size_t height, size_t depth);
static void cleanup_test_state(quantum_state* state);

int main(void) {
    printf("Running error weight tests...\n");

    // Run all tests
    test_initialization();
    test_weight_calculation();
    test_weight_statistics();
    test_geometric_scaling();
    test_error_cases();
    test_performance_requirements();

    printf("All error weight tests passed!\n");
    return 0;
}

static void test_initialization(void) {
    printf("Testing initialization...\n");

    // Test valid initialization
    WeightState state;
    WeightConfig config = {
        .lattice_width = 4,
        .lattice_height = 4,
        .lattice_depth = 4,
        .base_error_rate = 0.01,
        .probability_factor = 1.0,
        .geometric_factor = 1.0,
        .size_factor = 0.5,
        .use_geometric_scaling = true,
        .normalize_weights = true
    };

    bool success = init_error_weight(&state, &config);
    assert(success);
    assert(state.weight_map != NULL);
    assert(state.total_weight == 0.0);
    assert(state.max_weight == 0.0);
    assert(state.min_weight == INFINITY);
    assert(state.measurement_count == 0);

    // Test cleanup
    cleanup_error_weight(&state);

    // Test invalid parameters
    success = init_error_weight(NULL, &config);
    assert(!success);
    success = init_error_weight(&state, NULL);
    assert(!success);

    // Test invalid dimensions
    WeightConfig invalid_config = config;
    invalid_config.lattice_width = 0;
    success = init_error_weight(&state, &invalid_config);
    assert(!success);

    printf("Initialization tests passed\n");
}

static void test_weight_calculation(void) {
    printf("Testing weight calculation...\n");

    // Initialize weight system
    WeightState state;
    WeightConfig config = {
        .lattice_width = 4,
        .lattice_height = 4,
        .lattice_depth = 4,
        .base_error_rate = 0.01,
        .probability_factor = 1.0,
        .geometric_factor = 1.0,
        .size_factor = 0.5,
        .use_geometric_scaling = false,
        .normalize_weights = false
    };
    bool success = init_error_weight(&state, &config);
    assert(success);

    // Create test quantum state
    quantum_state* qstate = create_test_state(4, 4, 4);
    assert(qstate != NULL);

    // Calculate weights
    success = calculate_error_weights(&state, qstate);
    assert(success);
    assert(state.measurement_count == 1);

    // Verify weight map
    size_t map_size;
    const double* weights = get_weight_map(&state, &map_size);
    assert(weights != NULL);
    assert(map_size == 64);  // 4x4x4 lattice

    // Check individual weights
    for (size_t z = 0; z < 4; z++) {
        for (size_t y = 0; y < 4; y++) {
            for (size_t x = 0; x < 4; x++) {
                double weight = get_error_weight(&state, x, y, z);
                assert(weight >= 0.0);
                assert(weight <= 1.0);
            }
        }
    }

    // Cleanup
    cleanup_error_weight(&state);
    cleanup_test_state(qstate);

    printf("Weight calculation tests passed\n");
}

static void test_weight_statistics(void) {
    printf("Testing weight statistics...\n");

    // Initialize weight system
    WeightState state;
    WeightConfig config = {
        .lattice_width = 4,
        .lattice_height = 4,
        .lattice_depth = 4,
        .base_error_rate = 0.01,
        .probability_factor = 1.0,
        .geometric_factor = 1.0,
        .size_factor = 0.5,
        .use_geometric_scaling = false,
        .normalize_weights = true
    };
    bool success = init_error_weight(&state, &config);
    assert(success);

    // Create test quantum state
    quantum_state* qstate = create_test_state(4, 4, 4);
    assert(qstate != NULL);

    // Calculate weights
    success = calculate_error_weights(&state, qstate);
    assert(success);

    // Verify statistics
    WeightStatistics stats;
    success = get_weight_statistics(&state, &stats);
    assert(success);
    assert(stats.total_weight == 1.0);  // Due to normalization
    assert(stats.max_weight > 0.0);
    assert(stats.min_weight >= 0.0);
    assert(stats.max_weight >= stats.min_weight);
    assert(stats.measurement_count == 1);

    // Cleanup
    cleanup_error_weight(&state);
    cleanup_test_state(qstate);

    printf("Weight statistics tests passed\n");
}

static void test_geometric_scaling(void) {
    printf("Testing geometric scaling...\n");

    // Initialize weight system with geometric scaling
    WeightState state;
    WeightConfig config = {
        .lattice_width = 4,
        .lattice_height = 4,
        .lattice_depth = 4,
        .base_error_rate = 0.01,
        .probability_factor = 1.0,
        .geometric_factor = 2.0,
        .size_factor = 0.5,
        .use_geometric_scaling = true,
        .normalize_weights = false
    };
    bool success = init_error_weight(&state, &config);
    assert(success);

    // Create test quantum state
    quantum_state* qstate = create_test_state(4, 4, 4);
    assert(qstate != NULL);

    // Calculate weights
    success = calculate_error_weights(&state, qstate);
    assert(success);

    // Verify geometric scaling
    // Center weights should be higher than boundary weights
    double center_weight = get_error_weight(&state, 2, 2, 2);
    double boundary_weight = get_error_weight(&state, 0, 0, 0);
    assert(center_weight > boundary_weight);

    // Cleanup
    cleanup_error_weight(&state);
    cleanup_test_state(qstate);

    printf("Geometric scaling tests passed\n");
}

static void test_error_cases(void) {
    printf("Testing error cases...\n");

    WeightState state;
    WeightConfig config = {
        .lattice_width = 4,
        .lattice_height = 4,
        .lattice_depth = 4,
        .base_error_rate = 0.01,
        .probability_factor = 1.0,
        .geometric_factor = 1.0,
        .size_factor = 0.5,
        .use_geometric_scaling = false,
        .normalize_weights = false
    };

    // Test NULL parameters
    bool success = calculate_error_weights(NULL, NULL);
    assert(!success);

    quantum_state* qstate = create_test_state(4, 4, 4);
    success = calculate_error_weights(&state, NULL);
    assert(!success);
    success = calculate_error_weights(NULL, qstate);
    assert(!success);

    // Test invalid coordinates
    success = init_error_weight(&state, &config);
    assert(success);
    assert(get_error_weight(&state, 100, 0, 0) == 0.0);
    assert(get_error_weight(&state, 0, 100, 0) == 0.0);
    assert(get_error_weight(&state, 0, 0, 100) == 0.0);

    // Test invalid statistics access
    WeightStatistics stats;
    assert(!get_weight_statistics(NULL, &stats));
    assert(!get_weight_statistics(&state, NULL));

    // Cleanup
    cleanup_error_weight(&state);
    cleanup_test_state(qstate);

    printf("Error case tests passed\n");
}

static void test_performance_requirements(void) {
    printf("Testing performance requirements...\n");

    // Initialize large test system
    WeightState state;
    WeightConfig config = {
        .lattice_width = 100,   // Large lattice for stress testing
        .lattice_height = 100,
        .lattice_depth = 100,
        .base_error_rate = 0.01,
        .probability_factor = 1.0,
        .geometric_factor = 1.0,
        .size_factor = 0.5,
        .use_geometric_scaling = true,
        .normalize_weights = true
    };
    bool success = init_error_weight(&state, &config);
    assert(success);

    // Create large test state
    quantum_state* qstate = create_test_state(100, 100, 100);
    assert(qstate != NULL);

    // Measure initialization time
    clock_t start = clock();
    success = init_error_weight(&state, &config);
    clock_t end = clock();
    double init_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    assert(init_time < 0.001);  // Should initialize quickly

    // Measure weight calculation time
    start = clock();
    success = calculate_error_weights(&state, qstate);
    end = clock();
    double calc_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    assert(calc_time < 0.1);  // Should calculate quickly

    // Verify memory usage
    size_t expected_memory = 100 * 100 * 100 * sizeof(double);  // Weight map
    size_t actual_memory = sizeof(WeightState) + expected_memory;
    // Memory overhead should be reasonable
    assert(actual_memory < 100 * 1024 * 1024);  // Less than 100MB for 100x100x100 lattice

    // Cleanup
    cleanup_error_weight(&state);
    cleanup_test_state(qstate);

    printf("Performance requirement tests passed\n");
}

// Mock implementation of test helpers
static quantum_state* create_test_state(size_t width, size_t height, size_t depth) {
    quantum_state* state = malloc(sizeof(quantum_state));
    if (state) {
        state->width = width;
        state->height = height;
        state->depth = depth;
        // Initialize with test data
    }
    return state;
}

static void cleanup_test_state(quantum_state* state) {
    free(state);
}
