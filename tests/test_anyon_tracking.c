/**
 * @file test_anyon_tracking.c
 * @brief Tests for anyon position tracking system
 */

#include "quantum_geometric/physics/anyon_tracking.h"
#include "quantum_geometric/physics/syndrome_extraction.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

// Test helper functions
static ErrorSyndrome* create_test_syndrome(size_t num_errors) {
    ErrorSyndrome* syndrome = malloc(sizeof(ErrorSyndrome));
    syndrome->num_errors = num_errors;
    syndrome->total_weight = 0.0;

    for (size_t i = 0; i < num_errors; i++) {
        syndrome->error_locations[i] = i * 2;  // Spread errors out
        syndrome->error_types[i] = (i % 2) ? ERROR_X : ERROR_Z;
        syndrome->error_weights[i] = 1.0;
        syndrome->total_weight += 1.0;
    }

    return syndrome;
}

static void cleanup_test_syndrome(ErrorSyndrome* syndrome) {
    free(syndrome);
}

// Test cases
static void test_initialization() {
    printf("Testing initialization...\n");

    TrackingConfig config = {
        .lattice_width = 4,
        .lattice_height = 4,
        .history_length = 100,
        .move_threshold = 0.01,
        .track_charges = true
    };

    TrackingState state;
    bool success = init_anyon_tracking(&state, &config);
    assert(success && "Failed to initialize tracking state");
    assert(state.cache != NULL && "Cache not allocated");
    assert(state.cache->occupation_map != NULL && "Occupation map not allocated");
    assert(state.total_anyons == 0 && "Initial anyons not zero");
    assert(fabs(state.stability_score - 1.0) < 1e-6 &&
           "Initial stability not 1.0");

    cleanup_anyon_tracking(&state);
    printf("Initialization test passed\n");
}

static void test_position_tracking() {
    printf("Testing position tracking...\n");

    // Setup
    TrackingConfig config = {
        .lattice_width = 4,
        .lattice_height = 4,
        .history_length = 100,
        .move_threshold = 0.01,
        .track_charges = true
    };

    TrackingState state;
    bool success = init_anyon_tracking(&state, &config);
    assert(success && "Failed to initialize tracking state");

    // Create test syndrome with 3 errors
    ErrorSyndrome* syndrome = create_test_syndrome(3);
    AnyonPositions positions;

    // Track anyons
    success = track_anyons(&state, syndrome, &positions);
    assert(success && "Failed to track anyons");
    assert(positions.num_anyons == 3 && "Incorrect number of anyons");

    // Verify positions
    for (size_t i = 0; i < positions.num_anyons; i++) {
        size_t expected_x = (syndrome->error_locations[i]) % config.lattice_width;
        size_t expected_y = (syndrome->error_locations[i]) / config.lattice_width;
        assert(positions.x_coords[i] == expected_x && "Incorrect X coordinate");
        assert(positions.y_coords[i] == expected_y && "Incorrect Y coordinate");
    }

    cleanup_anyon_tracking(&state);
    cleanup_test_syndrome(syndrome);
    printf("Position tracking test passed\n");
}

static void test_movement_detection() {
    printf("Testing movement detection...\n");

    TrackingConfig config = {
        .lattice_width = 4,
        .lattice_height = 4,
        .history_length = 10,
        .move_threshold = 0.01,
        .track_charges = true
    };

    TrackingState state;
    bool success = init_anyon_tracking(&state, &config);
    assert(success && "Failed to initialize tracking state");

    // Create sequence of syndromes with moving anyons
    ErrorSyndrome* syndromes[3];
    AnyonPositions positions;

    // First position
    syndromes[0] = create_test_syndrome(1);
    syndromes[0]->error_locations[0] = 0;  // (0,0)
    success = track_anyons(&state, syndromes[0], &positions);
    assert(success && "Failed to track initial position");

    // Move anyon
    syndromes[1] = create_test_syndrome(1);
    syndromes[1]->error_locations[0] = 1;  // (1,0)
    success = track_anyons(&state, syndromes[1], &positions);
    assert(success && "Failed to track movement");

    // Final position
    syndromes[2] = create_test_syndrome(1);
    syndromes[2]->error_locations[0] = 2;  // (2,0)
    success = track_anyons(&state, syndromes[2], &positions);
    assert(success && "Failed to track final position");

    // Verify movement detection
    assert(state.total_movements > 0 && "No movements detected");
    assert(state.stability_score < 1.0 && "Stability not affected by movement");

    // Cleanup
    cleanup_anyon_tracking(&state);
    for (size_t i = 0; i < 3; i++) {
        cleanup_test_syndrome(syndromes[i]);
    }
    printf("Movement detection test passed\n");
}

static void test_charge_tracking() {
    printf("Testing charge tracking...\n");

    TrackingConfig config = {
        .lattice_width = 4,
        .lattice_height = 4,
        .history_length = 100,
        .move_threshold = 0.01,
        .track_charges = true
    };

    TrackingState state;
    bool success = init_anyon_tracking(&state, &config);
    assert(success && "Failed to initialize tracking state");

    // Create syndrome with different types of errors
    ErrorSyndrome* syndrome = create_test_syndrome(2);
    syndrome->error_types[0] = ERROR_X;  // +1 charge
    syndrome->error_types[1] = ERROR_Z;  // -1 charge
    
    AnyonPositions positions;
    success = track_anyons(&state, syndrome, &positions);
    assert(success && "Failed to track anyons");

    // Verify charges
    assert(positions.num_anyons == 2 && "Incorrect number of anyons");
    bool found_positive = false;
    bool found_negative = false;
    
    for (size_t i = 0; i < positions.num_anyons; i++) {
        if (positions.charges[i] == 1) found_positive = true;
        if (positions.charges[i] == -1) found_negative = true;
    }
    
    assert(found_positive && found_negative && "Missing charge types");

    cleanup_anyon_tracking(&state);
    cleanup_test_syndrome(syndrome);
    printf("Charge tracking test passed\n");
}

static void test_metrics_update() {
    printf("Testing metrics update...\n");

    TrackingConfig config = {
        .lattice_width = 4,
        .lattice_height = 4,
        .history_length = 10,
        .move_threshold = 0.01,
        .track_charges = true
    };

    TrackingState state;
    bool success = init_anyon_tracking(&state, &config);
    assert(success && "Failed to initialize tracking state");

    // Track sequence of anyons
    ErrorSyndrome* syndrome = create_test_syndrome(3);
    AnyonPositions positions;

    for (size_t i = 0; i < 5; i++) {
        // Modify syndrome to simulate movement
        for (size_t j = 0; j < syndrome->num_errors; j++) {
            syndrome->error_locations[j] += i;
        }
        
        success = track_anyons(&state, syndrome, &positions);
        assert(success && "Failed to track anyons");
    }

    // Update metrics
    success = update_tracking_metrics(&state);
    assert(success && "Failed to update metrics");

    // Verify metrics
    assert(state.total_anyons == 3 && "Incorrect anyon count");
    assert(state.total_movements > 0 && "No movements recorded");
    assert(state.stability_score >= 0.0 && state.stability_score <= 1.0 &&
           "Invalid stability score");

    cleanup_anyon_tracking(&state);
    cleanup_test_syndrome(syndrome);
    printf("Metrics update test passed\n");
}

static void test_error_handling() {
    printf("Testing error handling...\n");

    // Test null pointers
    bool success = init_anyon_tracking(NULL, NULL);
    assert(!success && "Should fail with null pointers");

    TrackingConfig config = {
        .lattice_width = 4,
        .lattice_height = 4,
        .history_length = 100,
        .move_threshold = 0.01,
        .track_charges = true
    };

    TrackingState state;
    success = init_anyon_tracking(&state, &config);
    assert(success && "Failed to initialize valid state");

    // Test invalid syndrome
    AnyonPositions positions;
    success = track_anyons(&state, NULL, &positions);
    assert(!success && "Should fail with null syndrome");

    // Test invalid positions pointer
    ErrorSyndrome* syndrome = create_test_syndrome(1);
    success = track_anyons(&state, syndrome, NULL);
    assert(!success && "Should fail with null positions");

    // Test invalid lattice dimensions
    TrackingConfig invalid_config = {
        .lattice_width = 0,
        .lattice_height = 0,
        .history_length = 100,
        .move_threshold = 0.01,
        .track_charges = true
    };

    TrackingState invalid_state;
    success = init_anyon_tracking(&invalid_state, &invalid_config);
    assert(!success && "Should fail with invalid dimensions");

    cleanup_anyon_tracking(&state);
    cleanup_test_syndrome(syndrome);
    printf("Error handling test passed\n");
}

int main() {
    printf("Running anyon tracking tests...\n\n");

    test_initialization();
    test_position_tracking();
    test_movement_detection();
    test_charge_tracking();
    test_metrics_update();
    test_error_handling();

    printf("\nAll anyon tracking tests passed!\n");
    return 0;
}
