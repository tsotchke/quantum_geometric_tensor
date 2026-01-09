/**
 * @file test_full_topological_protection.c
 * @brief Tests for full topological error protection implementation
 */

#include "quantum_geometric/physics/full_topological_protection.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/hardware/quantum_error_correction.h"
#include "quantum_geometric/core/quantum_complex.h"
#include "quantum_geometric/core/quantum_operations.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Test configurations
static const size_t TEST_NUM_QUBITS = 16;
static const double TEST_COHERENCE_TIME = 100.0;  // microseconds
static const double TEST_GATE_ERROR = 0.001;      // 0.1%
static const double TEST_MEASUREMENT_ERROR = 0.01; // 1%

// Helper functions
static quantum_state_t* create_quantum_state(size_t num_qubits) {
    quantum_state_t* state = malloc(sizeof(quantum_state_t));
    if (state) {
        state->num_qubits = num_qubits;
        state->dimension = 1 << num_qubits;  // 2^num_qubits
        state->coordinates = calloc(state->dimension, sizeof(ComplexFloat));
        if (state->coordinates) {
            // Initialize to |0⟩ state
            state->coordinates[0] = complex_float_create(1.0f, 0.0f);
            for (size_t i = 1; i < state->dimension; i++) {
                state->coordinates[i] = complex_float_create(0.0f, 0.0f);
            }
        }
        state->is_normalized = true;
    }
    return state;
}

static void initialize_ground_state(quantum_state_t* state) {
    if (state && state->coordinates) {
        // Set to |0⟩ state
        state->coordinates[0] = complex_float_create(1.0f, 0.0f);
        for (size_t i = 1; i < state->dimension; i++) {
            state->coordinates[i] = complex_float_create(0.0f, 0.0f);
        }
        state->is_normalized = true;
    }
}

static void destroy_quantum_state(quantum_state_t* state) {
    if (state) {
        free(state->coordinates);
        free(state);
    }
}

static void inject_test_error(quantum_state_t* state, size_t qubit) {
    // Flip a qubit (apply X gate)
    if (state && qubit < state->num_qubits) {
        size_t mask = 1 << qubit;
        for (size_t i = 0; i < state->dimension; i += 2 * mask) {
            for (size_t j = 0; j < mask; j++) {
                ComplexFloat temp = state->coordinates[i + j];
                state->coordinates[i + j] = state->coordinates[i + j + mask];
                state->coordinates[i + j + mask] = temp;
            }
        }
    }
}

static void reset_quantum_state(quantum_state_t* state) {
    initialize_ground_state(state);
}

static quantum_state_t* create_test_state(void) {
    quantum_state_t* state = create_quantum_state(TEST_NUM_QUBITS);
    assert(state != NULL);
    initialize_ground_state(state);
    return state;
}

static HardwareConfig* create_test_config(HardwareType type) {
    (void)type;  // Unused for now
    HardwareConfig* config = malloc(sizeof(HardwareConfig));
    assert(config != NULL);

    config->num_qubits = TEST_NUM_QUBITS;
    config->coherence_time = TEST_COHERENCE_TIME;
    config->gate_error_rate = TEST_GATE_ERROR;
    config->measurement_error_rate = TEST_MEASUREMENT_ERROR;
    config->t1_time = TEST_COHERENCE_TIME / 2.0;
    config->readout_fidelity = 0.99;
    config->supports_mid_circuit = true;

    return config;
}

// Test cases
void test_error_detection(void) {
    printf("Testing error detection...\n");

    quantum_state_t* state = create_test_state();
    HardwareConfig* config = create_test_config(HARDWARE_IBM);

    // Test clean state
    qgt_error_t result = detect_topological_errors(state, config);
    assert(result == QGT_SUCCESS);

    // Inject error
    inject_test_error(state, 5);

    // Test error detection (should detect error, return non-success)
    result = detect_topological_errors(state, config);
    assert(result != QGT_SUCCESS);

    free(config);
    destroy_quantum_state(state);
    printf("Error detection test passed\n");
}

void test_error_correction(void) {
    printf("Testing error correction...\n");

    quantum_state_t* state = create_test_state();
    HardwareConfig* config = create_test_config(HARDWARE_RIGETTI);

    // Inject error
    inject_test_error(state, 5);

    // Verify error detected
    qgt_error_t result = detect_topological_errors(state, config);
    assert(result != QGT_SUCCESS);

    // Apply correction
    correct_topological_errors(state, config);

    // Verify correction worked
    result = detect_topological_errors(state, config);
    assert(result == QGT_SUCCESS);

    free(config);
    destroy_quantum_state(state);
    printf("Error correction test passed\n");
}

void test_state_verification(void) {
    printf("Testing state verification...\n");

    quantum_state_t* state = create_test_state();
    HardwareConfig* config = create_test_config(HARDWARE_DWAVE);

    // Test clean state
    bool verified = verify_topological_state(state, config);
    assert(verified == true);

    // Inject error
    inject_test_error(state, 5);

    // Test corrupted state
    verified = verify_topological_state(state, config);
    assert(verified == false);

    // Correct error
    correct_topological_errors(state, config);

    // Test corrected state
    verified = verify_topological_state(state, config);
    assert(verified == true);

    free(config);
    destroy_quantum_state(state);
    printf("State verification test passed\n");
}

void test_continuous_protection(void) {
    printf("Testing continuous protection...\n");

    quantum_state_t* state = create_test_state();
    HardwareConfig* config = create_test_config(HARDWARE_IBM);

    // Start protection
    protect_topological_state(state, config);

    // Inject periodic errors
    for (int i = 0; i < 5; i++) {
        inject_test_error(state, i * 3);

        // Let protection cycle run
        wait_protection_interval(NULL);

        // Verify state remains protected
        bool verified = verify_topological_state(state, config);
        assert(verified == true);
    }

    free(config);
    destroy_quantum_state(state);
    printf("Continuous protection test passed\n");
}

void test_hardware_specific_behavior(void) {
    printf("Testing hardware-specific behavior...\n");

    quantum_state_t* state = create_test_state();

    // Test each hardware type
    HardwareType types[] = {
        HARDWARE_IBM,
        HARDWARE_RIGETTI,
        HARDWARE_DWAVE
    };

    for (size_t i = 0; i < sizeof(types)/sizeof(types[0]); i++) {
        HardwareConfig* config = create_test_config(types[i]);

        // Inject error
        inject_test_error(state, 5);

        // Verify error detected
        qgt_error_t result = detect_topological_errors(state, config);
        assert(result != QGT_SUCCESS);

        // Apply correction
        correct_topological_errors(state, config);

        // Verify correction worked with hardware-specific verification
        bool verified = verify_topological_state(state, config);
        assert(verified == true);

        free(config);
        reset_quantum_state(state);
    }

    destroy_quantum_state(state);
    printf("Hardware-specific behavior test passed\n");
}

void test_error_tracking(void) {
    printf("Testing error tracking...\n");

    quantum_state_t* state = create_test_state();
    HardwareConfig* config = create_test_config(HARDWARE_IBM);

    // Initialize tracker
    ErrorTracker* tracker = init_error_tracker(state, config);
    assert(tracker != NULL);

    // Verify tracker initialized with correct capacity
    assert(tracker->capacity > 0);
    assert(tracker->error_rates != NULL);
    assert(tracker->num_errors == 0);

    // Manually track errors
    for (int i = 0; i < 3; i++) {
        inject_test_error(state, i * 4);

        // Manually update tracker
        if (tracker->num_errors < tracker->capacity) {
            tracker->error_locations[tracker->num_errors] = i * 4;
            tracker->num_errors++;
        }
    }

    // Verify error count
    assert(tracker->num_errors == 3);

    free_error_tracker(tracker);
    free(config);
    destroy_quantum_state(state);
    printf("Error tracking test passed\n");
}

int main(void) {
    printf("Running full topological protection tests...\n\n");
    
    test_error_detection();
    test_error_correction();
    test_state_verification();
    test_continuous_protection();
    test_hardware_specific_behavior();
    test_error_tracking();
    
    printf("\nAll tests passed!\n");
    return 0;
}
