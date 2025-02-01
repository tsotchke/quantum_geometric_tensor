/**
 * @file test_basic_topological_protection.c
 * @brief Tests for basic topological error protection
 */

#include "quantum_geometric/physics/basic_topological_protection.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include <assert.h>
#include <stdio.h>

// Test helper functions
static quantum_state* create_test_state(void) {
    // Create a 4x4 lattice for testing
    quantum_state* state = create_quantum_state(16); // 16 qubits
    if (!state) return NULL;
    
    // Initialize in ground state
    initialize_ground_state(state);
    
    return state;
}

static void inject_test_error(quantum_state* state) {
    // Apply a single X error
    apply_pauli_x(state, 5); // Error on qubit 5
}

// Test cases
void test_error_detection(void) {
    printf("Testing error detection...\n");
    
    quantum_state* state = create_test_state();
    assert(state != NULL);
    
    // Initially no errors
    ErrorCode result = detect_basic_errors(state);
    assert(result == NO_ERROR);
    
    // Inject error
    inject_test_error(state);
    
    // Should detect error
    result = detect_basic_errors(state);
    assert(result == ERROR_DETECTED);
    
    destroy_quantum_state(state);
    printf("Error detection test passed\n");
}

void test_error_correction(void) {
    printf("Testing error correction...\n");
    
    quantum_state* state = create_test_state();
    assert(state != NULL);
    
    // Inject error
    inject_test_error(state);
    
    // Verify error detected
    ErrorCode result = detect_basic_errors(state);
    assert(result == ERROR_DETECTED);
    
    // Apply correction
    correct_basic_errors(state);
    
    // Verify error corrected
    result = detect_basic_errors(state);
    assert(result == NO_ERROR);
    
    destroy_quantum_state(state);
    printf("Error correction test passed\n");
}

void test_state_verification(void) {
    printf("Testing state verification...\n");
    
    quantum_state* state = create_test_state();
    assert(state != NULL);
    
    // Initially stable
    bool stable = verify_basic_state(state);
    assert(stable == true);
    
    // Inject error
    inject_test_error(state);
    
    // Should be unstable
    stable = verify_basic_state(state);
    assert(stable == false);
    
    // Correct error
    correct_basic_errors(state);
    
    // Should be stable again
    stable = verify_basic_state(state);
    assert(stable == true);
    
    destroy_quantum_state(state);
    printf("State verification test passed\n");
}

void test_protection_cycle(void) {
    printf("Testing protection cycle...\n");
    
    quantum_state* state = create_test_state();
    assert(state != NULL);
    
    // Run protection cycle on clean state
    protect_basic_state(state);
    assert(verify_basic_state(state) == true);
    
    // Inject error and protect
    inject_test_error(state);
    protect_basic_state(state);
    assert(verify_basic_state(state) == true);
    
    destroy_quantum_state(state);
    printf("Protection cycle test passed\n");
}

int main(void) {
    printf("Running basic topological protection tests...\n\n");
    
    test_error_detection();
    test_error_correction();
    test_state_verification();
    test_protection_cycle();
    
    printf("\nAll tests passed!\n");
    return 0;
}
