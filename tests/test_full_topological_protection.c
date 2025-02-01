/**
 * @file test_full_topological_protection.c
 * @brief Tests for full topological error protection implementation
 */

#include "quantum_geometric/physics/full_topological_protection.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/hardware/quantum_error_correction.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

// Test configurations
static const size_t TEST_NUM_QUBITS = 16;
static const double TEST_COHERENCE_TIME = 100.0;  // microseconds
static const double TEST_GATE_ERROR = 0.001;      // 0.1%
static const double TEST_MEASUREMENT_ERROR = 0.01; // 1%

// Helper functions
static quantum_state* create_test_state(void) {
    quantum_state* state = create_quantum_state(TEST_NUM_QUBITS);
    assert(state != NULL);
    initialize_ground_state(state);
    return state;
}

static HardwareConfig* create_test_config(HardwareType type) {
    HardwareConfig* config = malloc(sizeof(HardwareConfig));
    assert(config != NULL);
    
    config->hardware_type = type;
    config->num_qubits = TEST_NUM_QUBITS;
    config->coherence_time = TEST_COHERENCE_TIME;
    config->gate_error_rate = TEST_GATE_ERROR;
    config->measurement_error_rate = TEST_MEASUREMENT_ERROR;
    config->supports_parallel_measurement = true;
    config->supports_fast_feedback = true;
    
    return config;
}

// Test cases
void test_error_detection(void) {
    printf("Testing error detection...\n");
    
    quantum_state* state = create_test_state();
    HardwareConfig* config = create_test_config(HARDWARE_IBM);
    
    // Test clean state
    ErrorCode result = detect_topological_errors(state, config);
    assert(result == NO_ERROR);
    
    // Inject error
    inject_test_error(state, 5);
    
    // Test error detection
    result = detect_topological_errors(state, config);
    assert(result == ERROR_DETECTED);
    
    free(config);
    destroy_quantum_state(state);
    printf("Error detection test passed\n");
}

void test_error_correction(void) {
    printf("Testing error correction...\n");
    
    quantum_state* state = create_test_state();
    HardwareConfig* config = create_test_config(HARDWARE_RIGETTI);
    
    // Inject error
    inject_test_error(state, 5);
    
    // Verify error detected
    ErrorCode result = detect_topological_errors(state, config);
    assert(result == ERROR_DETECTED);
    
    // Apply correction
    correct_topological_errors(state, config);
    
    // Verify correction worked
    result = detect_topological_errors(state, config);
    assert(result == NO_ERROR);
    
    free(config);
    destroy_quantum_state(state);
    printf("Error correction test passed\n");
}

void test_state_verification(void) {
    printf("Testing state verification...\n");
    
    quantum_state* state = create_test_state();
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
    
    quantum_state* state = create_test_state();
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
    
    quantum_state* state = create_test_state();
    
    // Test each hardware type
    HardwareType types[] = {
        HARDWARE_IBM,
        HARDWARE_RIGETTI,
        HARDWARE_DWAVE,
        HARDWARE_GENERIC
    };
    
    for (size_t i = 0; i < sizeof(types)/sizeof(types[0]); i++) {
        HardwareConfig* config = create_test_config(types[i]);
        
        // Inject error
        inject_test_error(state, 5);
        
        // Verify error detected
        ErrorCode result = detect_topological_errors(state, config);
        assert(result == ERROR_DETECTED);
        
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
    
    quantum_state* state = create_test_state();
    HardwareConfig* config = create_test_config(HARDWARE_IBM);
    
    // Initialize tracker
    ErrorTracker* tracker = init_error_tracker(state, config);
    assert(tracker != NULL);
    
    // Track multiple errors
    for (int i = 0; i < 3; i++) {
        inject_test_error(state, i * 4);
        update_error_statistics(tracker);
        
        // Verify error count increases
        assert(tracker->num_errors == (size_t)(i + 1));
    }
    
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
