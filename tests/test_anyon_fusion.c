/**
 * @file test_anyon_fusion.c
 * @brief Tests for anyon fusion rules system
 */

#include "quantum_geometric/physics/anyon_fusion.h"
#include "quantum_geometric/physics/anyon_charge.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

// Test helper functions
static ChargeMeasurements* create_test_measurements(size_t num_measurements) {
    ChargeMeasurements* measurements = malloc(sizeof(ChargeMeasurements));
    measurements->num_measurements = num_measurements;

    for (size_t i = 0; i < num_measurements; i++) {
        measurements->locations[i] = i * 2;  // Spread out
        measurements->charges[i] = (i % 2) ? 1 : -1;  // Alternate charges
        measurements->probabilities[i] = 0.9;  // High confidence
    }

    return measurements;
}

static void cleanup_test_measurements(ChargeMeasurements* measurements) {
    free(measurements);
}

// Test cases
static void test_initialization() {
    printf("Testing initialization...\n");

    FusionConfig config = {
        .lattice_width = 4,
        .lattice_height = 4,
        .history_length = 100,
        .fusion_threshold = 0.01,
        .track_statistics = true
    };

    FusionState state;
    bool success = init_fusion_rules(&state, &config);
    assert(success && "Failed to initialize fusion state");
    assert(state.cache != NULL && "Cache not allocated");
    assert(state.cache->channel_map != NULL && "Channel map not allocated");
    assert(state.total_fusions == 0 && "Initial fusions not zero");
    assert(fabs(state.consistency_score - 1.0) < 1e-6 &&
           "Initial consistency not 1.0");

    cleanup_fusion_rules(&state);
    printf("Initialization test passed\n");
}

static void test_fusion_channels() {
    printf("Testing fusion channels...\n");

    // Setup
    FusionConfig config = {
        .lattice_width = 4,
        .lattice_height = 4,
        .history_length = 100,
        .fusion_threshold = 0.01,
        .track_statistics = true
    };

    FusionState state;
    bool success = init_fusion_rules(&state, &config);
    assert(success && "Failed to initialize fusion state");

    // Create test measurements with adjacent charges
    ChargeMeasurements* measurements = create_test_measurements(2);
    measurements->locations[0] = 0;  // First charge at (0,0)
    measurements->locations[1] = 1;  // Second charge at (1,0)
    measurements->charges[0] = 1;    // Positive charge
    measurements->charges[1] = -1;   // Negative charge

    FusionChannels channels;
    success = determine_fusion_channels(&state, measurements, &channels);
    assert(success && "Failed to determine fusion channels");

    // Verify fusion channel detection
    assert(channels.num_channels > 0 && "No fusion channels detected");
    bool found_channel = false;
    
    for (size_t i = 0; i < channels.num_channels; i++) {
        if (channels.locations[i] == 0) {  // Channel at fusion site
            assert(channels.input_charges[i][0] == 1 &&
                   channels.input_charges[i][1] == -1 &&
                   "Incorrect input charges");
            assert(channels.output_charge[i] == 0 &&
                   "Incorrect fusion output");
            assert(channels.probabilities[i] > 0.8 &&
                   "Low fusion probability");
            found_channel = true;
            break;
        }
    }
    
    assert(found_channel && "Failed to find expected fusion channel");

    cleanup_fusion_rules(&state);
    cleanup_test_measurements(measurements);
    printf("Fusion channels test passed\n");
}

static void test_braiding_statistics() {
    printf("Testing braiding statistics...\n");

    FusionConfig config = {
        .lattice_width = 4,
        .lattice_height = 4,
        .history_length = 10,
        .fusion_threshold = 0.01,
        .track_statistics = true
    };

    FusionState state;
    bool success = init_fusion_rules(&state, &config);
    assert(success && "Failed to initialize fusion state");

    // Create sequence of measurements with braiding pattern
    ChargeMeasurements* measurements[3];
    FusionChannels channels;

    // Initial configuration
    measurements[0] = create_test_measurements(2);
    measurements[0]->locations[0] = 0;  // First anyon at (0,0)
    measurements[0]->locations[1] = 2;  // Second anyon at (2,0)
    measurements[0]->charges[0] = 1;
    measurements[0]->charges[1] = -1;

    success = determine_fusion_channels(&state, measurements[0], &channels);
    assert(success && "Failed to determine initial channels");

    // Move anyons to create braiding
    measurements[1] = create_test_measurements(2);
    measurements[1]->locations[0] = 2;  // First anyon moves right
    measurements[1]->locations[1] = 0;  // Second anyon moves left
    measurements[1]->charges[0] = 1;
    measurements[1]->charges[1] = -1;

    success = determine_fusion_channels(&state, measurements[1], &channels);
    assert(success && "Failed to determine braiding channels");

    // Verify braiding detection
    assert(state.total_braidings > 0 && "No braiding events detected");

    // Cleanup
    cleanup_fusion_rules(&state);
    for (size_t i = 0; i < 2; i++) {
        cleanup_test_measurements(measurements[i]);
    }
    printf("Braiding statistics test passed\n");
}

static void test_consistency_metrics() {
    printf("Testing consistency metrics...\n");

    FusionConfig config = {
        .lattice_width = 4,
        .lattice_height = 4,
        .history_length = 10,
        .fusion_threshold = 0.01,
        .track_statistics = true
    };

    FusionState state;
    bool success = init_fusion_rules(&state, &config);
    assert(success && "Failed to initialize fusion state");

    // Create test measurements with consistent fusion rules
    ChargeMeasurements* measurements = create_test_measurements(4);
    measurements->charges[0] = 1;   // Two pairs of opposite charges
    measurements->charges[1] = -1;
    measurements->charges[2] = 1;
    measurements->charges[3] = -1;

    FusionChannels channels;
    for (size_t i = 0; i < 5; i++) {
        success = determine_fusion_channels(&state, measurements, &channels);
        assert(success && "Failed to determine channels");
    }

    // Update metrics
    success = update_fusion_metrics(&state);
    assert(success && "Failed to update metrics");

    // Verify metrics
    assert(state.total_fusions == 5 && "Incorrect fusion count");
    assert(state.consistency_score > 0.9 &&
           "Low consistency score for valid fusions");

    cleanup_fusion_rules(&state);
    cleanup_test_measurements(measurements);
    printf("Consistency metrics test passed\n");
}

static void test_error_handling() {
    printf("Testing error handling...\n");

    // Test null pointers
    bool success = init_fusion_rules(NULL, NULL);
    assert(!success && "Should fail with null pointers");

    FusionConfig config = {
        .lattice_width = 4,
        .lattice_height = 4,
        .history_length = 100,
        .fusion_threshold = 0.01,
        .track_statistics = true
    };

    FusionState state;
    success = init_fusion_rules(&state, &config);
    assert(success && "Failed to initialize valid state");

    // Test invalid measurements
    FusionChannels channels;
    success = determine_fusion_channels(&state, NULL, &channels);
    assert(!success && "Should fail with null measurements");

    // Test invalid channels pointer
    ChargeMeasurements* measurements = create_test_measurements(1);
    success = determine_fusion_channels(&state, measurements, NULL);
    assert(!success && "Should fail with null channels");

    // Test invalid lattice dimensions
    FusionConfig invalid_config = {
        .lattice_width = 0,
        .lattice_height = 0,
        .history_length = 100,
        .fusion_threshold = 0.01,
        .track_statistics = true
    };

    FusionState invalid_state;
    success = init_fusion_rules(&invalid_state, &invalid_config);
    assert(!success && "Should fail with invalid dimensions");

    cleanup_fusion_rules(&state);
    cleanup_test_measurements(measurements);
    printf("Error handling test passed\n");
}

int main() {
    printf("Running anyon fusion tests...\n\n");

    test_initialization();
    test_fusion_channels();
    test_braiding_statistics();
    test_consistency_metrics();
    test_error_handling();

    printf("\nAll anyon fusion tests passed!\n");
    return 0;
}
