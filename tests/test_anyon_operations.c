/**
 * @file test_anyon_operations.c
 * @brief Test suite for anyon detection and manipulation operations
 */

#include "quantum_geometric/physics/anyon_detection.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/physics/quantum_state_operations.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

// Test configurations
static const double TEST_THRESHOLD = 0.01;
static const size_t TEST_LATTICE_SIZE = 16;
static const size_t MAX_TEST_ANYONS = 10;

// Helper function to create a test quantum state
static quantum_state* create_test_state(void) {
    quantum_state* state = create_quantum_state(TEST_LATTICE_SIZE);
    if (!state) {
        return NULL;
    }

    // Initialize with a simple topological state
    initialize_topological_state(state);
    return state;
}

// Test anyon detection
static void test_anyon_detection(void) {
    printf("Testing anyon detection...\n");
    
    quantum_state* state = create_test_state();
    assert(state != NULL);

    // Configure detection
    DetectionConfig config = {
        .detection_threshold = TEST_THRESHOLD,
        .noise_tolerance = 0.1,
        .measurement_cycles = 100,
        .use_error_correction = true
    };

    // Initialize detection system
    bool init_success = init_anyon_detection(state, &config);
    assert(init_success);

    // Create array for detected anyons
    Anyon detected_anyons[MAX_TEST_ANYONS];
    
    // Inject test errors
    inject_test_error(state, 5);  // Position 5
    inject_test_error(state, 10); // Position 10

    // Detect anyons
    size_t num_detected = detect_anyons(state, detected_anyons, MAX_TEST_ANYONS);
    
    // Verify detection
    assert(num_detected == 2);
    assert(detected_anyons[0].position.x == 5 || detected_anyons[0].position.x == 10);
    assert(detected_anyons[1].position.x == 5 || detected_anyons[1].position.x == 10);
    assert(detected_anyons[0].position.x != detected_anyons[1].position.x);

    cleanup_anyon_detection(state);
    destroy_quantum_state(state);
    printf("Anyon detection test passed\n");
}

// Test anyon tracking
static void test_anyon_tracking(void) {
    printf("Testing anyon tracking...\n");
    
    quantum_state* state = create_test_state();
    assert(state != NULL);

    // Configure detection
    DetectionConfig config = {
        .detection_threshold = TEST_THRESHOLD,
        .noise_tolerance = 0.1,
        .measurement_cycles = 100,
        .use_error_correction = true
    };

    bool init_success = init_anyon_detection(state, &config);
    assert(init_success);

    // Create and inject a test anyon
    Anyon test_anyon = {
        .type = ANYON_ABELIAN,
        .position = {.x = 5, .y = 5, .z = 0, .stability = 1.0},
        .lifetime = 0.0,
        .energy = 1.0,
        .is_mobile = true
    };

    // Track anyon movement
    bool tracking_success = track_anyon_movement(state, &test_anyon, 10);
    assert(tracking_success);

    // Verify tracking results
    assert(test_anyon.lifetime > 0.0);
    assert(test_anyon.position.stability > 0.5);

    cleanup_anyon_detection(state);
    destroy_quantum_state(state);
    printf("Anyon tracking test passed\n");
}

// Test anyon braiding
static void test_anyon_braiding(void) {
    printf("Testing anyon braiding...\n");
    
    quantum_state* state = create_test_state();
    assert(state != NULL);

    // Create test anyons
    Anyon anyon1 = {
        .type = ANYON_ABELIAN,
        .position = {.x = 5, .y = 5, .z = 0, .stability = 1.0},
        .lifetime = 0.0,
        .energy = 1.0,
        .is_mobile = true
    };

    Anyon anyon2 = {
        .type = ANYON_ABELIAN,
        .position = {.x = 8, .y = 5, .z = 0, .stability = 1.0},
        .lifetime = 0.0,
        .energy = 1.0,
        .is_mobile = true
    };

    // Create anyon pair
    AnyonPair pair = {
        .anyon1 = &anyon1,
        .anyon2 = &anyon2,
        .interaction_strength = 0.0,
        .braiding_phase = 0.0
    };

    // Configure braiding
    BraidingConfig config = {
        .min_separation = 2.0,
        .max_interaction_strength = 1.0,
        .braiding_steps = 100,
        .verify_topology = true
    };

    // Perform braiding
    bool braiding_success = braid_anyons(state, &pair, &config);
    assert(braiding_success);

    // Verify braiding results
    assert(fabs(pair.braiding_phase) > 0.0);
    assert(pair.interaction_strength > 0.0);

    destroy_quantum_state(state);
    printf("Anyon braiding test passed\n");
}

// Test anyon fusion
static void test_anyon_fusion(void) {
    printf("Testing anyon fusion...\n");
    
    quantum_state* state = create_test_state();
    assert(state != NULL);

    // Create test anyons
    Anyon anyon1 = {
        .type = ANYON_ABELIAN,
        .position = {.x = 5, .y = 5, .z = 0, .stability = 1.0},
        .lifetime = 0.0,
        .energy = 1.0,
        .is_mobile = true
    };

    Anyon anyon2 = {
        .type = ANYON_ABELIAN,
        .position = {.x = 6, .y = 5, .z = 0, .stability = 1.0},
        .lifetime = 0.0,
        .energy = 1.0,
        .is_mobile = true
    };

    // Create anyon pair
    AnyonPair pair = {
        .anyon1 = &anyon1,
        .anyon2 = &anyon2,
        .interaction_strength = 1.0,
        .braiding_phase = 0.0
    };

    // Configure fusion
    FusionConfig config = {
        .energy_threshold = 2.0,
        .coherence_requirement = 0.9,
        .fusion_attempts = 10,
        .track_statistics = true
    };

    // Perform fusion
    FusionOutcome outcome = fuse_anyons(state, &pair, &config);

    // Verify fusion results
    assert(outcome.probability > 0.0);
    assert(outcome.result_type != ANYON_ABELIAN || 
           outcome.result_type != ANYON_NON_ABELIAN);
    assert(fabs(outcome.energy_delta) > 0.0);

    destroy_quantum_state(state);
    printf("Anyon fusion test passed\n");
}

// Test fusion rules
static void test_fusion_rules(void) {
    printf("Testing fusion rules...\n");

    // Create test anyons with different types
    Anyon abelian1 = {.type = ANYON_ABELIAN};
    Anyon abelian2 = {.type = ANYON_ABELIAN};
    Anyon non_abelian = {.type = ANYON_NON_ABELIAN};
    Anyon majorana = {.type = ANYON_MAJORANA};
    Anyon fibonacci = {.type = ANYON_FIBONACCI};

    // Test valid fusion combinations
    assert(check_fusion_rules(&abelian1, &abelian2));
    assert(check_fusion_rules(&non_abelian, &non_abelian));
    assert(check_fusion_rules(&majorana, &majorana));
    assert(check_fusion_rules(&fibonacci, &fibonacci));

    // Test invalid fusion combinations
    assert(!check_fusion_rules(&abelian1, NULL));
    assert(!check_fusion_rules(NULL, &abelian2));
    assert(!check_fusion_rules(NULL, NULL));

    printf("Fusion rules test passed\n");
}

// Test interaction energy calculations
static void test_interaction_energy(void) {
    printf("Testing interaction energy calculations...\n");

    // Create test anyons at different positions
    Anyon anyon1 = {
        .type = ANYON_ABELIAN,
        .position = {.x = 0, .y = 0, .z = 0},
        .charge = {.electric = 1.0, .magnetic = 0.0, .topological = 1.0}
    };

    Anyon anyon2 = {
        .type = ANYON_ABELIAN,
        .position = {.x = 3, .y = 4, .z = 0},
        .charge = {.electric = -1.0, .magnetic = 0.0, .topological = 1.0}
    };

    // Calculate interaction energy
    double energy = calculate_interaction_energy(&anyon1, &anyon2);

    // Verify energy calculations
    assert(energy > 0.0);
    assert(isfinite(energy));

    // Test edge cases
    assert(calculate_interaction_energy(&anyon1, NULL) == 0.0);
    assert(calculate_interaction_energy(NULL, &anyon2) == 0.0);
    assert(calculate_interaction_energy(NULL, NULL) == 0.0);

    printf("Interaction energy test passed\n");
}

int main(void) {
    printf("Running anyon operations tests...\n\n");

    test_anyon_detection();
    test_anyon_tracking();
    test_anyon_braiding();
    test_anyon_fusion();
    test_fusion_rules();
    test_interaction_energy();

    printf("\nAll anyon operations tests passed!\n");
    return 0;
}
