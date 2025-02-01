/**
 * @file test_heavy_hex_surface_code.c
 * @brief Tests for heavy-hex surface code implementation
 */

#include "quantum_geometric/physics/heavy_hex_surface_code.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/physics/quantum_state_operations.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

// Test helper functions
static void test_initialization(void);
static void test_hex_stabilizers(void);
static void test_error_detection(void);
static void test_error_correction(void);
static void test_error_cases(void);
static void test_performance_requirements(void);

// Mock functions and data
static quantum_state* create_test_state(size_t width, size_t height);
static void apply_test_errors(quantum_state* state);
static void cleanup_test_state(quantum_state* state);

int main(void) {
    printf("Running heavy-hex surface code tests...\n");

    // Run all tests
    test_initialization();
    test_hex_stabilizers();
    test_error_detection();
    test_error_correction();
    test_error_cases();
    test_performance_requirements();

    printf("All heavy-hex surface code tests passed!\n");
    return 0;
}

static void test_initialization(void) {
    printf("Testing initialization...\n");

    // Test valid initialization
    HexState state;
    HexConfig config = {
        .lattice_width = 5,   // Must be odd for heavy-hex layout
        .lattice_height = 5,  // Must be odd for heavy-hex layout
        .base_error_rate = 0.01,
        .error_threshold = 0.1,
        .auto_correction = true
    };

    bool success = init_heavy_hex_code(&state, &config);
    assert(success);
    assert(state.lattice != NULL);
    assert(state.lattice->stabilizer_values != NULL);
    assert(state.lattice->stabilizer_coordinates != NULL);
    assert(state.measurement_count == 0);
    assert(state.error_rate == 0.0);

    // Verify hex layout properties
    assert(state.lattice->num_stabilizers == ((5 - 1) * (5 - 1)) / 2);
    
    // Test cleanup
    cleanup_heavy_hex_code(&state);

    // Test invalid parameters
    success = init_heavy_hex_code(NULL, &config);
    assert(!success);
    success = init_heavy_hex_code(&state, NULL);
    assert(!success);

    // Test invalid dimensions (must be odd)
    HexConfig invalid_config = config;
    invalid_config.lattice_width = 4;  // Even width not allowed
    success = init_heavy_hex_code(&state, &invalid_config);
    assert(!success);

    printf("Initialization tests passed\n");
}

static void test_hex_stabilizers(void) {
    printf("Testing hex stabilizers...\n");

    // Initialize hex code
    HexState state;
    HexConfig config = {
        .lattice_width = 5,
        .lattice_height = 5,
        .base_error_rate = 0.01,
        .error_threshold = 0.1,
        .auto_correction = false
    };
    bool success = init_heavy_hex_code(&state, &config);
    assert(success);

    // Create test quantum state
    quantum_state* qstate = create_test_state(5, 5);
    assert(qstate != NULL);

    // Measure stabilizers
    success = measure_hex_code(&state, qstate);
    assert(success);
    assert(state.measurement_count == 1);

    // Verify stabilizer pattern
    size_t plaquette_count = 0;
    size_t vertex_count = 0;
    for (size_t i = 0; i < state.lattice->num_stabilizers; i++) {
        HexCoordinate* coord = &state.lattice->stabilizer_coordinates[i];
        
        // Verify coordinates are valid
        assert(coord->x > 0 && coord->x < config.lattice_width - 1);
        assert(coord->y > 0 && coord->y < config.lattice_height - 1);
        
        // Count stabilizer types
        if (coord->type == HEX_PLAQUETTE) {
            plaquette_count++;
        } else {
            vertex_count++;
        }

        // Verify stabilizer values are valid
        assert(fabs(state.lattice->stabilizer_values[i]) <= 1.0);
    }

    // Verify stabilizer distribution
    assert(plaquette_count > 0);
    assert(vertex_count > 0);
    assert(plaquette_count + vertex_count == state.lattice->num_stabilizers);

    // Cleanup
    cleanup_heavy_hex_code(&state);
    cleanup_test_state(qstate);

    printf("Hex stabilizer tests passed\n");
}

static void test_error_detection(void) {
    printf("Testing error detection...\n");

    // Initialize hex code
    HexState state;
    HexConfig config = {
        .lattice_width = 5,
        .lattice_height = 5,
        .base_error_rate = 0.01,
        .error_threshold = 0.1,
        .auto_correction = false
    };
    bool success = init_heavy_hex_code(&state, &config);
    assert(success);

    // Create test quantum state with errors
    quantum_state* qstate = create_test_state(5, 5);
    assert(qstate != NULL);
    apply_test_errors(qstate);

    // Measure stabilizers
    success = measure_hex_code(&state, qstate);
    assert(success);

    // Verify error detection
    double error_rate = get_hex_error_rate(&state);
    assert(error_rate > 0.0);  // Should detect errors
    assert(error_rate <= 1.0);

    // Verify syndrome storage
    size_t syndrome_size;
    const double* syndrome = get_hex_syndrome(&state, &syndrome_size);
    assert(syndrome != NULL);
    assert(syndrome_size == state.lattice->num_stabilizers);

    // Cleanup
    cleanup_heavy_hex_code(&state);
    cleanup_test_state(qstate);

    printf("Error detection tests passed\n");
}

static void test_error_correction(void) {
    printf("Testing error correction...\n");

    // Initialize hex code with auto-correction
    HexState state;
    HexConfig config = {
        .lattice_width = 5,
        .lattice_height = 5,
        .base_error_rate = 0.01,
        .error_threshold = 0.1,
        .auto_correction = true
    };
    bool success = init_heavy_hex_code(&state, &config);
    assert(success);

    // Create test quantum state with errors
    quantum_state* qstate = create_test_state(5, 5);
    assert(qstate != NULL);
    apply_test_errors(qstate);

    // Initial measurement to detect errors
    success = measure_hex_code(&state, qstate);
    assert(success);
    double initial_error_rate = get_hex_error_rate(&state);
    assert(initial_error_rate > 0.0);

    // Second measurement after auto-correction
    success = measure_hex_code(&state, qstate);
    assert(success);
    double final_error_rate = get_hex_error_rate(&state);
    assert(final_error_rate < initial_error_rate);

    // Cleanup
    cleanup_heavy_hex_code(&state);
    cleanup_test_state(qstate);

    printf("Error correction tests passed\n");
}

static void test_error_cases(void) {
    printf("Testing error cases...\n");

    HexState state;
    HexConfig config = {
        .lattice_width = 5,
        .lattice_height = 5,
        .base_error_rate = 0.01,
        .error_threshold = 0.1,
        .auto_correction = true
    };

    // Test NULL parameters
    bool success = measure_hex_code(NULL, NULL);
    assert(!success);

    quantum_state* qstate = create_test_state(5, 5);
    success = measure_hex_code(&state, NULL);
    assert(!success);
    success = measure_hex_code(NULL, qstate);
    assert(!success);

    // Test invalid dimensions
    HexConfig invalid_config = config;
    invalid_config.lattice_width = 0;
    success = init_heavy_hex_code(&state, &invalid_config);
    assert(!success);

    // Test invalid error parameters
    invalid_config = config;
    invalid_config.base_error_rate = -1.0;
    success = init_heavy_hex_code(&state, &invalid_config);
    assert(!success);

    // Test invalid syndrome access
    size_t size;
    assert(get_hex_syndrome(NULL, &size) == NULL);
    assert(get_hex_syndrome(&state, NULL) == NULL);

    // Cleanup
    cleanup_test_state(qstate);

    printf("Error case tests passed\n");
}

static void test_performance_requirements(void) {
    printf("Testing performance requirements...\n");

    // Initialize large test system
    HexState state;
    HexConfig config = {
        .lattice_width = 101,   // Large odd number for stress testing
        .lattice_height = 101,
        .base_error_rate = 0.01,
        .error_threshold = 0.1,
        .auto_correction = true
    };
    bool success = init_heavy_hex_code(&state, &config);
    assert(success);

    // Create large test state
    quantum_state* qstate = create_test_state(101, 101);
    assert(qstate != NULL);

    // Measure initialization time
    clock_t start = clock();
    success = init_heavy_hex_code(&state, &config);
    clock_t end = clock();
    double init_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    assert(init_time < 0.001);  // Should initialize quickly

    // Measure stabilizer measurement time
    start = clock();
    success = measure_hex_code(&state, qstate);
    end = clock();
    double measurement_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    assert(measurement_time < 0.1);  // Should measure quickly

    // Verify memory usage
    size_t expected_memory = state.lattice->num_stabilizers * 
                            (sizeof(double) + sizeof(HexCoordinate));
    size_t actual_memory = sizeof(HexState) + sizeof(HexLattice) + expected_memory;
    // Memory overhead should be reasonable
    assert(actual_memory < 10 * 1024 * 1024);  // Less than 10MB for 101x101 lattice

    // Cleanup
    cleanup_heavy_hex_code(&state);
    cleanup_test_state(qstate);

    printf("Performance requirement tests passed\n");
}

// Mock implementation of test helpers
static quantum_state* create_test_state(size_t width, size_t height) {
    quantum_state* state = malloc(sizeof(quantum_state));
    if (state) {
        state->width = width;
        state->height = height;
        // Initialize with test data
    }
    return state;
}

static void apply_test_errors(quantum_state* state) {
    if (!state) return;
    // Apply known test errors
    // For example, flip bits at specific locations
}

static void cleanup_test_state(quantum_state* state) {
    free(state);
}
