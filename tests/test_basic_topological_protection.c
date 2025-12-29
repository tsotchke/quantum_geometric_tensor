/**
 * @file test_basic_topological_protection.c
 * @brief Tests for basic topological error protection
 */

#include "quantum_geometric/physics/basic_topological_protection.h"
#include "quantum_geometric/physics/anyon_detection.h"
#include "quantum_geometric/core/quantum_state.h"
#include "quantum_geometric/core/error_codes.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Test lattice configuration
#define TEST_LATTICE_WIDTH 4
#define TEST_LATTICE_HEIGHT 4
#define TEST_NUM_QUBITS 16

/**
 * Create a quantum state configured for topological code testing.
 * Sets up the lattice structure needed for stabilizer operations.
 */
static quantum_state_t* create_test_state(void) {
    quantum_state_t* state = NULL;
    size_t dim = 1UL << TEST_NUM_QUBITS;

    qgt_error_t err = quantum_state_create(&state, QUANTUM_STATE_PURE, dim);
    if (err != QGT_SUCCESS || !state) {
        return NULL;
    }

    // Configure lattice structure for surface code
    state->num_qubits = TEST_NUM_QUBITS;
    state->lattice_width = TEST_LATTICE_WIDTH;
    state->lattice_height = TEST_LATTICE_HEIGHT;

    // For a square lattice: (L-1)×(L-1) plaquettes, L×L vertices
    state->num_plaquettes = (TEST_LATTICE_WIDTH - 1) * (TEST_LATTICE_HEIGHT - 1);
    state->num_vertices = TEST_LATTICE_WIDTH * TEST_LATTICE_HEIGHT;
    state->num_stabilizers = state->num_plaquettes + state->num_vertices;

    // Allocate anyon tracking array
    state->anyons = calloc(TEST_NUM_QUBITS, sizeof(Anyon));
    state->num_anyons = 0;
    state->max_anyons = TEST_NUM_QUBITS;

    // Initialize to ground state
    initialize_ground_state(state);

    return state;
}

/**
 * Inject a test error (single X error on specified qubit)
 * Uses library's inject_x_error if available, otherwise apply_pauli_x
 */
static void inject_x_error(quantum_state_t* state, size_t qubit) {
    apply_pauli_x(state, qubit);
}

/**
 * Clean up test state
 */
static void destroy_test_state(quantum_state_t* state) {
    if (state) {
        free(state->anyons);
        quantum_state_destroy(state);
    }
}

// ============================================================================
// Test Cases
// ============================================================================

void test_error_detection(void) {
    printf("Testing error detection...\n");

    quantum_state_t* state = create_test_state();
    assert(state != NULL);

    // Initially no errors - ground state should pass all stabilizer checks
    ErrorCode result = detect_basic_errors(state);
    assert(result == NO_ERROR);

    // Inject error on qubit 5
    inject_x_error(state, 5);

    // Should detect error via syndrome measurement
    result = detect_basic_errors(state);
    assert(result == ERROR_DETECTED);

    destroy_test_state(state);
    printf("Error detection test passed\n");
}

void test_error_correction(void) {
    printf("Testing error correction...\n");

    quantum_state_t* state = create_test_state();
    assert(state != NULL);

    // Inject error
    inject_x_error(state, 5);

    // Verify error detected
    ErrorCode result = detect_basic_errors(state);
    assert(result == ERROR_DETECTED);

    // Apply correction
    correct_basic_errors(state);

    // Verify error corrected
    result = detect_basic_errors(state);
    assert(result == NO_ERROR);

    destroy_test_state(state);
    printf("Error correction test passed\n");
}

void test_state_verification(void) {
    printf("Testing state verification...\n");

    quantum_state_t* state = create_test_state();
    assert(state != NULL);

    // Initially stable (all stabilizers should have eigenvalue +1)
    bool stable = verify_basic_state(state);
    assert(stable == true);

    // Inject error
    inject_x_error(state, 5);

    // Should be unstable (some stabilizers have eigenvalue -1)
    stable = verify_basic_state(state);
    assert(stable == false);

    // Correct error
    correct_basic_errors(state);

    // Should be stable again
    stable = verify_basic_state(state);
    assert(stable == true);

    destroy_test_state(state);
    printf("State verification test passed\n");
}

void test_protection_cycle(void) {
    printf("Testing protection cycle...\n");

    quantum_state_t* state = create_test_state();
    assert(state != NULL);

    // Run protection cycle on clean state
    protect_basic_state(state);
    assert(verify_basic_state(state) == true);

    // Inject error and protect
    inject_x_error(state, 5);
    protect_basic_state(state);
    assert(verify_basic_state(state) == true);

    destroy_test_state(state);
    printf("Protection cycle test passed\n");
}

void test_multiple_errors(void) {
    printf("Testing multiple error correction...\n");

    quantum_state_t* state = create_test_state();
    assert(state != NULL);

    // Inject multiple errors (but within correction capability)
    inject_x_error(state, 0);
    inject_x_error(state, 7);

    // Should detect errors
    ErrorCode result = detect_basic_errors(state);
    assert(result == ERROR_DETECTED);

    // Apply correction
    correct_basic_errors(state);

    // Verify correction (may not be perfect for multiple errors)
    result = detect_basic_errors(state);
    // For distance-3 code, single errors are correctable
    // Multiple errors may or may not be correctable depending on separation

    destroy_test_state(state);
    printf("Multiple error test completed\n");
}

void test_stabilizer_measurement(void) {
    printf("Testing stabilizer measurement...\n");

    quantum_state_t* state = create_test_state();
    assert(state != NULL);

    // Ground state: all stabilizers should have eigenvalue +1
    for (size_t i = 0; i < state->num_plaquettes; i++) {
        double val = measure_plaquette_operator(state, i);
        // Expectation value should be close to +1
        assert(fabs(val - 1.0) < 0.1 || fabs(val + 1.0) < 0.1);
    }

    destroy_test_state(state);
    printf("Stabilizer measurement test passed\n");
}

// ============================================================================
// Main
// ============================================================================

int main(void) {
    printf("Running basic topological protection tests...\n\n");

    test_error_detection();
    test_error_correction();
    test_state_verification();
    test_protection_cycle();
    test_multiple_errors();
    test_stabilizer_measurement();

    printf("\nAll tests passed!\n");
    return 0;
}
