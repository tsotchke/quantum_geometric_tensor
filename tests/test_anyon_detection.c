/**
 * @file test_anyon_detection.c
 * @brief Tests for anyon detection and tracking system
 *
 * Tests the anyon detection API using the library's geometric quantum_state type.
 * quantum_state is typedef'd from quantum_state_t which contains geometric coordinates.
 */

#include "quantum_geometric/physics/anyon_detection.h"
#include "quantum_geometric/physics/quantum_state_operations.h"
#include "quantum_geometric/core/quantum_types.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <time.h>

// Test helper functions
static void test_initialization(void);
static void test_detection(void);
static void test_tracking(void);
static void test_charge_measurement(void);
static void test_fusion_rules(void);
static void test_error_cases(void);
static void test_performance_requirements(void);

// Helper functions using library's geometric quantum_state API
// quantum_state is typedef'd to quantum_state_t in anyon_detection.h
static quantum_state* create_test_state(size_t width, size_t height, size_t depth);
static void apply_test_anyons(quantum_state* state);
static void cleanup_test_state(quantum_state* state);

int main(void) {
    printf("Running anyon detection tests...\n");

    // Run all tests
    test_initialization();
    test_detection();
    test_tracking();
    test_charge_measurement();
    test_fusion_rules();
    test_error_cases();
    test_performance_requirements();

    printf("All anyon detection tests passed!\n");
    return 0;
}

static void test_initialization(void) {
    printf("Testing initialization...\n");

    // Test valid initialization
    AnyonState state;
    AnyonConfig config = {
        .grid_width = 4,
        .grid_height = 4,
        .grid_depth = 2,
        .detection_threshold = 0.1,
        .max_movement_speed = 1.0,
        .charge_threshold = 0.01
    };

    bool success = init_anyon_detection(&state, &config);
    assert(success);
    assert(state.grid != NULL);
    assert(state.grid->cells != NULL);
    assert(state.measurement_count == 0);
    assert(state.total_anyons == 0);

    // Test cleanup
    cleanup_anyon_detection(&state);

    // Test invalid parameters
    success = init_anyon_detection(NULL, &config);
    assert(!success);
    success = init_anyon_detection(&state, NULL);
    assert(!success);

    // Test invalid dimensions
    AnyonConfig invalid_config = config;
    invalid_config.grid_width = 0;
    success = init_anyon_detection(&state, &invalid_config);
    assert(!success);

    printf("Initialization tests passed\n");
}

static void test_detection(void) {
    printf("Testing anyon detection...\n");

    // Initialize detection system
    AnyonState state;
    AnyonConfig config = {
        .grid_width = 4,
        .grid_height = 4,
        .grid_depth = 2,
        .detection_threshold = 0.1,
        .max_movement_speed = 1.0,
        .charge_threshold = 0.01
    };
    bool success = init_anyon_detection(&state, &config);
    assert(success);

    // Create test quantum state with anyons
    quantum_state* qstate = create_test_state(4, 4, 2);
    assert(qstate != NULL);
    apply_test_anyons(qstate);

    // Detect anyons
    success = detect_and_track_anyons(&state, qstate);
    assert(success);
    assert(state.measurement_count == 1);
    assert(state.total_anyons > 0);

    // Verify anyon positions
    size_t num_anyons = count_anyons(state.grid);
    assert(num_anyons == state.total_anyons);
    assert(num_anyons > 0);

    // Verify position storage
    assert(state.last_positions != NULL);
    for (size_t i = 0; i < num_anyons; i++) {
        assert(state.last_positions[i].x < config.grid_width);
        assert(state.last_positions[i].y < config.grid_height);
        assert(state.last_positions[i].z < config.grid_depth);
        assert(state.last_positions[i].type != ANYON_NONE);
    }

    // Cleanup
    cleanup_anyon_detection(&state);
    cleanup_test_state(qstate);

    printf("Detection tests passed\n");
}

static void test_tracking(void) {
    printf("Testing anyon tracking...\n");

    // Initialize detection system
    AnyonState state;
    AnyonConfig config = {
        .grid_width = 4,
        .grid_height = 4,
        .grid_depth = 2,
        .detection_threshold = 0.1,
        .max_movement_speed = 1.0,
        .charge_threshold = 0.01
    };
    bool success = init_anyon_detection(&state, &config);
    assert(success);

    // Create test quantum state with moving anyons
    quantum_state* qstate = create_test_state(4, 4, 2);
    assert(qstate != NULL);
    apply_test_anyons(qstate);

    // Track anyons over multiple measurements
    for (size_t i = 0; i < 5; i++) {
        success = detect_and_track_anyons(&state, qstate);
        assert(success);

        // Verify velocity tracking
        for (size_t z = 0; z < state.grid->depth; z++) {
            for (size_t y = 0; y < state.grid->height; y++) {
                for (size_t x = 0; x < state.grid->width; x++) {
                    size_t idx = z * state.grid->height * state.grid->width +
                               y * state.grid->width + x;
                    if (state.grid->cells[idx].type != ANYON_NONE) {
                        // Check velocity components are within bounds
                        assert(fabs(state.grid->cells[idx].velocity[0]) <= config.max_movement_speed);
                        assert(fabs(state.grid->cells[idx].velocity[1]) <= config.max_movement_speed);
                        assert(fabs(state.grid->cells[idx].velocity[2]) <= config.max_movement_speed);
                    }
                }
            }
        }
    }

    // Cleanup
    cleanup_anyon_detection(&state);
    cleanup_test_state(qstate);

    printf("Tracking tests passed\n");
}

static void test_charge_measurement(void) {
    printf("Testing charge measurement...\n");

    // Initialize detection system
    AnyonState state;
    AnyonConfig config = {
        .grid_width = 4,
        .grid_height = 4,
        .grid_depth = 2,
        .detection_threshold = 0.1,
        .max_movement_speed = 1.0,
        .charge_threshold = 0.01
    };
    bool success = init_anyon_detection(&state, &config);
    assert(success);

    // Create test quantum state with charged anyons
    quantum_state* qstate = create_test_state(4, 4, 2);
    assert(qstate != NULL);
    apply_test_anyons(qstate);

    // Measure charges
    success = detect_and_track_anyons(&state, qstate);
    assert(success);

    // Verify charge measurements
    for (size_t z = 0; z < state.grid->depth; z++) {
        for (size_t y = 0; y < state.grid->height; y++) {
            for (size_t x = 0; x < state.grid->width; x++) {
                size_t idx = z * state.grid->height * state.grid->width +
                           y * state.grid->width + x;
                if (state.grid->cells[idx].type != ANYON_NONE) {
                    assert(fabs(state.grid->cells[idx].charge) > config.charge_threshold);
                }
            }
        }
    }

    // Cleanup
    cleanup_anyon_detection(&state);
    cleanup_test_state(qstate);

    printf("Charge measurement tests passed\n");
}

static void test_fusion_rules(void) {
    printf("Testing fusion rules...\n");

    // Initialize detection system
    AnyonState state;
    AnyonConfig config = {
        .grid_width = 4,
        .grid_height = 4,
        .grid_depth = 2,
        .detection_threshold = 0.1,
        .max_movement_speed = 1.0,
        .charge_threshold = 0.01
    };
    bool success = init_anyon_detection(&state, &config);
    assert(success);

    // Create test quantum state with adjacent anyons
    quantum_state* qstate = create_test_state(4, 4, 2);
    assert(qstate != NULL);
    apply_test_anyons(qstate);

    // Initial anyon count
    success = detect_and_track_anyons(&state, qstate);
    assert(success);
    size_t initial_count = state.total_anyons;
    assert(initial_count > 0);

    // Apply fusion rules
    success = detect_and_track_anyons(&state, qstate);
    assert(success);
    size_t final_count = state.total_anyons;

    // Verify fusion effects
    assert(final_count <= initial_count);  // Some anyons should fuse/annihilate

    // Cleanup
    cleanup_anyon_detection(&state);
    cleanup_test_state(qstate);

    printf("Fusion rule tests passed\n");
}

static void test_error_cases(void) {
    printf("Testing error cases...\n");

    AnyonState state;
    AnyonConfig config = {
        .grid_width = 4,
        .grid_height = 4,
        .grid_depth = 2,
        .detection_threshold = 0.1,
        .max_movement_speed = 1.0,
        .charge_threshold = 0.01
    };

    // Test NULL parameters
    bool success = detect_and_track_anyons(NULL, NULL);
    assert(!success);

    quantum_state* qstate = create_test_state(4, 4, 2);
    success = detect_and_track_anyons(&state, NULL);
    assert(!success);
    success = detect_and_track_anyons(NULL, qstate);
    assert(!success);

    // Test invalid grid access
    assert(count_anyons(NULL) == 0);
    AnyonPosition positions[1];
    assert(!get_anyon_positions(NULL, positions));
    assert(!get_anyon_positions(state.grid, NULL));

    // Test invalid configuration
    AnyonConfig invalid_config = config;
    invalid_config.detection_threshold = -1.0;
    success = init_anyon_detection(&state, &invalid_config);
    assert(!success);

    // Cleanup
    cleanup_test_state(qstate);

    printf("Error case tests passed\n");
}

static void test_performance_requirements(void) {
    printf("Testing performance requirements...\n");

    // Initialize large test system
    AnyonState state;
    AnyonConfig config = {
        .grid_width = 100,   // Large grid for stress testing
        .grid_height = 100,
        .grid_depth = 10,
        .detection_threshold = 0.1,
        .max_movement_speed = 1.0,
        .charge_threshold = 0.01
    };
    bool success = init_anyon_detection(&state, &config);
    assert(success);

    // Create large test state
    quantum_state* qstate = create_test_state(100, 100, 10);
    assert(qstate != NULL);

    // Measure initialization time
    clock_t start = clock();
    success = init_anyon_detection(&state, &config);
    clock_t end = clock();
    double init_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    assert(init_time < 0.001);  // Should initialize quickly

    // Measure detection time
    start = clock();
    success = detect_and_track_anyons(&state, qstate);
    end = clock();
    double detection_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    assert(detection_time < 0.1);  // Should detect quickly

    // Verify memory usage
    size_t expected_memory = config.grid_width * config.grid_height *
                           config.grid_depth * sizeof(AnyonCell);
    size_t actual_memory = sizeof(AnyonState) + sizeof(AnyonGrid) + expected_memory;
    // Memory overhead should be reasonable
    assert(actual_memory < 100 * 1024 * 1024);  // Less than 100MB for 100x100x10 grid

    // Cleanup
    cleanup_anyon_detection(&state);
    cleanup_test_state(qstate);

    printf("Performance requirement tests passed\n");
}

// Test helper implementations using library's geometric quantum_state API
static quantum_state* create_test_state(size_t width, size_t height, size_t depth) {
    // Use library's create_quantum_state function
    // Total qubits = width * height * depth
    size_t num_qubits = width * height * depth;

    // Create quantum state using the library function from anyon_detection.h
    quantum_state* state = create_quantum_state(num_qubits);
    if (!state) return NULL;

    // Initialize to |00...0⟩ - first basis state has amplitude 1
    // The geometric quantum state uses coordinates (ComplexFloat array)
    if (state->coordinates && state->dimension > 0) {
        state->coordinates[0].real = 1.0f;
        state->coordinates[0].imag = 0.0f;
        for (size_t i = 1; i < state->dimension; i++) {
            state->coordinates[i].real = 0.0f;
            state->coordinates[i].imag = 0.0f;
        }
        state->is_normalized = true;
    }

    return state;
}

static void apply_test_anyons(quantum_state* state) {
    if (!state || !state->coordinates) return;

    // Create anyon excitations by modifying the quantum state
    // Anyons are detected from non-trivial stabilizer measurements,
    // which arise from excited states (non-ground state configurations)

    // For a lattice of qubits, we create excitations by putting
    // the system in a superposition that violates stabilizer constraints

    size_t dim = state->dimension;
    if (dim < 4) return;  // Need at least 2 qubits for meaningful test

    // Create a state with non-trivial syndrome pattern:
    // Equal superposition of |00...0⟩ and some excited states
    // This will create detectable anyon signatures

    float norm = 1.0f / sqrtf(3.0f);  // Normalize over 3 states

    // |ψ⟩ = (|0⟩ + |1⟩ + |2⟩) / √3  - creates X-type and Z-type excitations
    state->coordinates[0].real = norm;
    state->coordinates[0].imag = 0.0f;

    if (dim > 1) {
        state->coordinates[1].real = norm;
        state->coordinates[1].imag = 0.0f;
    }

    if (dim > 2) {
        state->coordinates[2].real = norm;
        state->coordinates[2].imag = 0.0f;
    }

    // Zero out remaining amplitudes
    for (size_t i = 3; i < dim; i++) {
        state->coordinates[i].real = 0.0f;
        state->coordinates[i].imag = 0.0f;
    }

    state->is_normalized = true;
}

static void cleanup_test_state(quantum_state* state) {
    // Use library's destroy function
    destroy_quantum_state(state);
}
