/**
 * @file test_anyon_correction.c
 * @brief Tests for anyon-based error correction system
 */

#include "quantum_geometric/physics/anyon_correction.h"
#include "quantum_geometric/physics/anyon_fusion.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

// Test helper functions
static FusionChannels* create_test_channels(size_t num_channels) {
    FusionChannels* channels = malloc(sizeof(FusionChannels));
    channels->num_channels = num_channels;

    for (size_t i = 0; i < num_channels; i++) {
        channels->locations[i] = i * 2;  // Spread out
        channels->input_charges[i][0] = 1;   // First input charge
        channels->input_charges[i][1] = -1;  // Second input charge
        channels->output_charge[i] = 0;      // Expected fusion outcome
        channels->probabilities[i] = 0.9;    // High confidence
    }

    return channels;
}

static void cleanup_test_channels(FusionChannels* channels) {
    free(channels);
}

// Test cases
static void test_initialization() {
    printf("Testing initialization...\n");

    CorrectionConfig config = {
        .lattice_width = 4,
        .lattice_height = 4,
        .history_length = 100,
        .error_threshold = 0.9,
        .track_success = true
    };

    CorrectionState state;
    bool success = init_error_correction(&state, &config);
    assert(success && "Failed to initialize correction state");
    assert(state.cache != NULL && "Cache not allocated");
    assert(state.cache->correction_map != NULL && "Correction map not allocated");
    assert(state.total_corrections == 0 && "Initial corrections not zero");
    assert(fabs(state.success_rate - 1.0) < 1e-6 &&
           "Initial success rate not 1.0");

    cleanup_error_correction(&state);
    printf("Initialization test passed\n");
}

static void test_correction_operations() {
    printf("Testing correction operations...\n");

    // Setup
    CorrectionConfig config = {
        .lattice_width = 4,
        .lattice_height = 4,
        .history_length = 100,
        .error_threshold = 0.9,
        .track_success = true
    };

    CorrectionState state;
    bool success = init_error_correction(&state, &config);
    assert(success && "Failed to initialize correction state");

    // Create test fusion channels with error
    FusionChannels* channels = create_test_channels(2);
    channels->output_charge[0] = 1;  // Unexpected fusion outcome
    channels->output_charge[1] = -1; // Another error

    CorrectionOperations operations;
    success = determine_corrections(&state, channels, &operations);
    assert(success && "Failed to determine corrections");

    // Verify correction operations
    assert(operations.num_corrections > 0 && "No corrections generated");
    bool found_correction = false;
    
    for (size_t i = 0; i < operations.num_corrections; i++) {
        if (operations.locations[i] == 0) {  // First error site
            assert(operations.types[i] == CORRECTION_X &&
                   "Incorrect correction type");
            assert(operations.weights[i] > 0.8 &&
                   "Low correction weight");
            found_correction = true;
            break;
        }
    }
    
    assert(found_correction && "Failed to find expected correction");

    cleanup_error_correction(&state);
    cleanup_test_channels(channels);
    printf("Correction operations test passed\n");
}

static void test_success_tracking() {
    printf("Testing success tracking...\n");

    CorrectionConfig config = {
        .lattice_width = 4,
        .lattice_height = 4,
        .history_length = 10,
        .error_threshold = 0.9,
        .track_success = true
    };

    CorrectionState state;
    bool success = init_error_correction(&state, &config);
    assert(success && "Failed to initialize correction state");

    // Create sequence of successful corrections
    FusionChannels* channels = create_test_channels(4);
    CorrectionOperations operations;

    for (size_t i = 0; i < 5; i++) {
        success = determine_corrections(&state, channels, &operations);
        assert(success && "Failed to determine corrections");
    }

    // Verify success metrics
    assert(state.total_corrections == 5 && "Incorrect correction count");
    assert(state.success_rate > 0.9 &&
           "Low success rate for valid corrections");

    cleanup_error_correction(&state);
    cleanup_test_channels(channels);
    printf("Success tracking test passed\n");
}

static void test_pattern_detection() {
    printf("Testing pattern detection...\n");

    CorrectionConfig config = {
        .lattice_width = 4,
        .lattice_height = 4,
        .history_length = 10,
        .error_threshold = 0.9,
        .track_success = true
    };

    CorrectionState state;
    bool success = init_error_correction(&state, &config);
    assert(success && "Failed to initialize correction state");

    // Create repeating error pattern
    FusionChannels* channels = create_test_channels(2);
    channels->output_charge[0] = 1;  // Consistent error at first site
    channels->output_charge[1] = -1; // Consistent error at second site

    CorrectionOperations operations;
    for (size_t i = 0; i < 5; i++) {
        success = determine_corrections(&state, channels, &operations);
        assert(success && "Failed to determine corrections");
    }

    // Verify pattern detection
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
    
    assert(found_correlation && "No error patterns detected");

    cleanup_error_correction(&state);
    cleanup_test_channels(channels);
    printf("Pattern detection test passed\n");
}

static void test_error_handling() {
    printf("Testing error handling...\n");

    // Test null pointers
    bool success = init_error_correction(NULL, NULL);
    assert(!success && "Should fail with null pointers");

    CorrectionConfig config = {
        .lattice_width = 4,
        .lattice_height = 4,
        .history_length = 100,
        .error_threshold = 0.9,
        .track_success = true
    };

    CorrectionState state;
    success = init_error_correction(&state, &config);
    assert(success && "Failed to initialize valid state");

    // Test invalid fusion channels
    CorrectionOperations operations;
    success = determine_corrections(&state, NULL, &operations);
    assert(!success && "Should fail with null channels");

    // Test invalid operations pointer
    FusionChannels* channels = create_test_channels(1);
    success = determine_corrections(&state, channels, NULL);
    assert(!success && "Should fail with null operations");

    // Test invalid lattice dimensions
    CorrectionConfig invalid_config = {
        .lattice_width = 0,
        .lattice_height = 0,
        .history_length = 100,
        .error_threshold = 0.9,
        .track_success = true
    };

    CorrectionState invalid_state;
    success = init_error_correction(&invalid_state, &invalid_config);
    assert(!success && "Should fail with invalid dimensions");

    cleanup_error_correction(&state);
    cleanup_test_channels(channels);
    printf("Error handling test passed\n");
}

int main() {
    printf("Running anyon correction tests...\n\n");

    test_initialization();
    test_correction_operations();
    test_success_tracking();
    test_pattern_detection();
    test_error_handling();

    printf("\nAll anyon correction tests passed!\n");
    return 0;
}
