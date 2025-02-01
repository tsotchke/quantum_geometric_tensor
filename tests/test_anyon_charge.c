/**
 * @file test_anyon_charge.c
 * @brief Tests for anyon charge measurement system
 */

#include "quantum_geometric/physics/anyon_charge.h"
#include "quantum_geometric/physics/anyon_tracking.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

// Test helper functions
static AnyonPositions* create_test_positions(size_t num_anyons) {
    AnyonPositions* positions = malloc(sizeof(AnyonPositions));
    positions->num_anyons = num_anyons;

    for (size_t i = 0; i < num_anyons; i++) {
        positions->x_coords[i] = i % 4;  // Spread across width
        positions->y_coords[i] = i / 4;  // Spread across height
        positions->charges[i] = (i % 2) ? 1 : -1;  // Alternate charges
    }

    return positions;
}

static void cleanup_test_positions(AnyonPositions* positions) {
    free(positions);
}

// Test cases
static void test_initialization() {
    printf("Testing initialization...\n");

    ChargeConfig config = {
        .lattice_width = 4,
        .lattice_height = 4,
        .history_length = 100,
        .fusion_threshold = 0.01,
        .track_correlations = true
    };

    ChargeState state;
    bool success = init_charge_measurement(&state, &config);
    assert(success && "Failed to initialize charge state");
    assert(state.cache != NULL && "Cache not allocated");
    assert(state.cache->charge_map != NULL && "Charge map not allocated");
    assert(state.total_measurements == 0 && "Initial measurements not zero");
    assert(fabs(state.conservation_score - 1.0) < 1e-6 &&
           "Initial conservation not 1.0");

    cleanup_charge_measurement(&state);
    printf("Initialization test passed\n");
}

static void test_charge_measurement() {
    printf("Testing charge measurement...\n");

    // Setup
    ChargeConfig config = {
        .lattice_width = 4,
        .lattice_height = 4,
        .history_length = 100,
        .fusion_threshold = 0.01,
        .track_correlations = true
    };

    ChargeState state;
    bool success = init_charge_measurement(&state, &config);
    assert(success && "Failed to initialize charge state");

    // Create test positions with 3 anyons
    AnyonPositions* positions = create_test_positions(3);
    ChargeMeasurements measurements;

    // Measure charges
    success = measure_charges(&state, positions, &measurements);
    assert(success && "Failed to measure charges");
    assert(measurements.num_measurements == 3 && "Incorrect number of measurements");

    // Verify measurements
    for (size_t i = 0; i < measurements.num_measurements; i++) {
        size_t expected_location = positions->y_coords[i] * config.lattice_width +
                                 positions->x_coords[i];
        assert(measurements.locations[i] == expected_location &&
               "Incorrect measurement location");
        assert(measurements.charges[i] == positions->charges[i] &&
               "Incorrect measured charge");
        assert(measurements.probabilities[i] >= 0.0 &&
               measurements.probabilities[i] <= 1.0 &&
               "Invalid measurement probability");
    }

    cleanup_charge_measurement(&state);
    cleanup_test_positions(positions);
    printf("Charge measurement test passed\n");
}

static void test_fusion_detection() {
    printf("Testing fusion detection...\n");

    ChargeConfig config = {
        .lattice_width = 4,
        .lattice_height = 4,
        .history_length = 10,
        .fusion_threshold = 0.01,
        .track_correlations = true
    };

    ChargeState state;
    bool success = init_charge_measurement(&state, &config);
    assert(success && "Failed to initialize charge state");

    // Create sequence of positions with fusing anyons
    AnyonPositions* positions[3];
    ChargeMeasurements measurements;

    // First configuration
    positions[0] = create_test_positions(2);
    positions[0]->x_coords[0] = 0;  // First anyon at (0,0)
    positions[0]->y_coords[0] = 0;
    positions[0]->charges[0] = 1;
    positions[0]->x_coords[1] = 1;  // Second anyon at (1,0)
    positions[0]->y_coords[1] = 0;
    positions[0]->charges[1] = -1;

    success = measure_charges(&state, positions[0], &measurements);
    assert(success && "Failed to measure initial charges");

    // Move anyons closer
    positions[1] = create_test_positions(2);
    positions[1]->x_coords[0] = 0;
    positions[1]->y_coords[0] = 0;
    positions[1]->charges[0] = 1;
    positions[1]->x_coords[1] = 0;  // Move second anyon to (0,0)
    positions[1]->y_coords[1] = 0;
    positions[1]->charges[1] = -1;

    success = measure_charges(&state, positions[1], &measurements);
    assert(success && "Failed to measure fusion");

    // Verify fusion detection
    assert(state.total_fusions > 0 && "No fusions detected");

    // Cleanup
    cleanup_charge_measurement(&state);
    for (size_t i = 0; i < 2; i++) {
        cleanup_test_positions(positions[i]);
    }
    printf("Fusion detection test passed\n");
}

static void test_correlation_tracking() {
    printf("Testing correlation tracking...\n");

    ChargeConfig config = {
        .lattice_width = 4,
        .lattice_height = 4,
        .history_length = 10,
        .fusion_threshold = 0.01,
        .track_correlations = true
    };

    ChargeState state;
    bool success = init_charge_measurement(&state, &config);
    assert(success && "Failed to initialize charge state");

    // Create test positions with correlated charges
    AnyonPositions* positions = create_test_positions(4);
    positions->charges[0] = 1;   // Create charge pairs
    positions->charges[1] = -1;
    positions->charges[2] = 1;
    positions->charges[3] = -1;

    ChargeMeasurements measurements;

    // Measure multiple times to build correlation data
    for (size_t i = 0; i < 5; i++) {
        success = measure_charges(&state, positions, &measurements);
        assert(success && "Failed to measure charges");
    }

    // Verify correlations
    size_t total_sites = config.lattice_width * config.lattice_height;
    bool found_correlation = false;
    
    for (size_t i = 0; i < total_sites; i++) {
        for (size_t j = i + 1; j < total_sites; j++) {
            if (fabs(state.cache->correlation_matrix[i * total_sites + j]) > 0.5) {
                found_correlation = true;
                break;
            }
        }
    }
    
    assert(found_correlation && "No charge correlations detected");

    cleanup_charge_measurement(&state);
    cleanup_test_positions(positions);
    printf("Correlation tracking test passed\n");
}

static void test_metrics_update() {
    printf("Testing metrics update...\n");

    ChargeConfig config = {
        .lattice_width = 4,
        .lattice_height = 4,
        .history_length = 10,
        .fusion_threshold = 0.01,
        .track_correlations = true
    };

    ChargeState state;
    bool success = init_charge_measurement(&state, &config);
    assert(success && "Failed to initialize charge state");

    // Create test positions with unbalanced charges
    AnyonPositions* positions = create_test_positions(3);
    positions->charges[0] = 1;
    positions->charges[1] = 1;
    positions->charges[2] = 1;  // Net positive charge

    ChargeMeasurements measurements;
    success = measure_charges(&state, positions, &measurements);
    assert(success && "Failed to measure charges");

    // Update metrics
    success = update_charge_metrics(&state);
    assert(success && "Failed to update metrics");

    // Verify metrics
    assert(state.total_measurements == 1 && "Incorrect measurement count");
    assert(state.conservation_score < 1.0 &&
           "Conservation score not affected by charge imbalance");

    cleanup_charge_measurement(&state);
    cleanup_test_positions(positions);
    printf("Metrics update test passed\n");
}

static void test_error_handling() {
    printf("Testing error handling...\n");

    // Test null pointers
    bool success = init_charge_measurement(NULL, NULL);
    assert(!success && "Should fail with null pointers");

    ChargeConfig config = {
        .lattice_width = 4,
        .lattice_height = 4,
        .history_length = 100,
        .fusion_threshold = 0.01,
        .track_correlations = true
    };

    ChargeState state;
    success = init_charge_measurement(&state, &config);
    assert(success && "Failed to initialize valid state");

    // Test invalid positions
    ChargeMeasurements measurements;
    success = measure_charges(&state, NULL, &measurements);
    assert(!success && "Should fail with null positions");

    // Test invalid measurements pointer
    AnyonPositions* positions = create_test_positions(1);
    success = measure_charges(&state, positions, NULL);
    assert(!success && "Should fail with null measurements");

    // Test invalid lattice dimensions
    ChargeConfig invalid_config = {
        .lattice_width = 0,
        .lattice_height = 0,
        .history_length = 100,
        .fusion_threshold = 0.01,
        .track_correlations = true
    };

    ChargeState invalid_state;
    success = init_charge_measurement(&invalid_state, &invalid_config);
    assert(!success && "Should fail with invalid dimensions");

    cleanup_charge_measurement(&state);
    cleanup_test_positions(positions);
    printf("Error handling test passed\n");
}

int main() {
    printf("Running anyon charge measurement tests...\n\n");

    test_initialization();
    test_charge_measurement();
    test_fusion_detection();
    test_correlation_tracking();
    test_metrics_update();
    test_error_handling();

    printf("\nAll anyon charge measurement tests passed!\n");
    return 0;
}
