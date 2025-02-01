/**
 * @file test_error_syndrome.c
 * @brief Tests for error syndrome detection and correction
 */

#include "quantum_geometric/physics/error_syndrome.h"
#include "quantum_geometric/physics/quantum_state_operations.h"
#include "quantum_geometric/core/quantum_state.h"
#include "quantum_geometric/core/quantum_operations.h"
#include "quantum_geometric/physics/error_types.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <complex.h>

// Test helper functions
static void test_init_cleanup(void);
static void test_error_detection(void);
static void test_error_classification(void);
static void test_error_correction(void);
static void test_boundary_cases(void);
static void test_error_chains(void);
static void test_error_patterns(void);
static void test_error_correlations(void);

// Helper functions
static quantum_state_t* create_test_state(void);
static void verify_syndrome_state(const ErrorSyndrome* syndrome);
static void verify_correction_result(const quantum_state_t* state);

static void inject_error(quantum_state_t* state, size_t location, error_type_t type) {
    qgt_error_t err;
    quantum_operator_t* op;
    err = quantum_operator_create(&op, QUANTUM_OPERATOR_PAULI, state->dimension);
    assert(err == QGT_SUCCESS);

    switch (type) {
        case ERROR_X:
            err = quantum_operator_pauli_x(op, location);
            break;
        case ERROR_Z:
            err = quantum_operator_pauli_z(op, location);
            break;
        case ERROR_Y:
            err = quantum_operator_pauli_y(op, location);
            break;
    }
    assert(err == QGT_SUCCESS);

    // Apply the error operator to the state
    err = quantum_operator_apply(op, state);
    assert(err == QGT_SUCCESS);

    // Clean up
    quantum_operator_destroy(op);
}

int main(void) {
    printf("Running error syndrome tests...\n");

    test_init_cleanup();
    test_error_detection();
    test_error_classification();
    test_error_correction();
    test_boundary_cases();
    test_error_chains();
    test_error_patterns();
    test_error_correlations();

    printf("All error syndrome tests passed!\n");
    return 0;
}

static void test_init_cleanup(void) {
    printf("Testing initialization and cleanup...\n");

    ErrorSyndrome syndrome;
    qgt_error_t err = init_error_syndrome(&syndrome, 16);
    assert(err == QGT_SUCCESS);

    // Verify initialization
    assert(syndrome.num_errors == 0);
    assert(syndrome.max_errors == 16);
    assert(syndrome.error_locations != NULL);
    assert(syndrome.error_types != NULL);
    assert(syndrome.error_weights != NULL);

    cleanup_error_syndrome(&syndrome);
    printf("Initialization test passed\n");
}

static void test_error_detection(void) {
    printf("Testing error detection...\n");

    quantum_state_t* state = create_test_state();
    ErrorSyndrome syndrome;
    init_error_syndrome(&syndrome, 16);

    // Test single error detection
    inject_error(state, 0, ERROR_X);
    qgt_error_t err = detect_errors(state, &syndrome);
    assert(err == QGT_SUCCESS);
    assert(syndrome.num_errors == 1);
    assert(syndrome.error_locations[0] == 0);
    assert(syndrome.error_types[0] == ERROR_X);

    // Test multiple error detection
    inject_error(state, 5, ERROR_Z);
    inject_error(state, 10, ERROR_Y);
    err = detect_errors(state, &syndrome);
    assert(err == QGT_SUCCESS);
    assert(syndrome.num_errors == 3);

    cleanup_error_syndrome(&syndrome);
    cleanup_test_state(state);
    printf("Error detection test passed\n");
}

static void test_error_classification(void) {
    printf("Testing error classification...\n");

    quantum_state_t* state = create_test_state();
    ErrorSyndrome syndrome;
    init_error_syndrome(&syndrome, 16);

    // Test X error classification
    inject_error(state, 0, ERROR_X);
    qgt_error_t err = detect_errors(state, &syndrome);
    assert(err == QGT_SUCCESS);
    assert(syndrome.error_types[0] == ERROR_X);

    // Test Z error classification
    inject_error(state, 5, ERROR_Z);
    err = detect_errors(state, &syndrome);
    assert(err == QGT_SUCCESS);
    bool found_z = false;
    for (size_t i = 0; i < syndrome.num_errors; i++) {
        if (syndrome.error_types[i] == ERROR_Z) {
            found_z = true;
            break;
        }
    }
    assert(found_z);

    // Test Y error classification
    inject_error(state, 10, ERROR_Y);
    err = detect_errors(state, &syndrome);
    assert(err == QGT_SUCCESS);
    bool found_y = false;
    for (size_t i = 0; i < syndrome.num_errors; i++) {
        if (syndrome.error_types[i] == ERROR_Y) {
            found_y = true;
            break;
        }
    }
    assert(found_y);

    cleanup_error_syndrome(&syndrome);
    cleanup_test_state(state);
    printf("Error classification test passed\n");
}

static void test_error_correction(void) {
    printf("Testing error correction...\n");

    quantum_state_t* state = create_test_state();
    ErrorSyndrome syndrome;
    init_error_syndrome(&syndrome, 16);

    // Create test error pattern
    inject_error(state, 0, ERROR_X);
    inject_error(state, 1, ERROR_X);
    inject_error(state, 4, ERROR_Z);
    inject_error(state, 8, ERROR_Z);

    // Detect and correct errors
    qgt_error_t err = detect_errors(state, &syndrome);
    assert(err == QGT_SUCCESS);
    err = correct_errors(state, &syndrome);
    assert(err == QGT_SUCCESS);

    // Verify correction
    verify_correction_result(state);

    cleanup_error_syndrome(&syndrome);
    cleanup_test_state(state);
    printf("Error correction test passed\n");
}

static void test_boundary_cases(void) {
    printf("Testing boundary cases...\n");

    // Test NULL state
    ErrorSyndrome syndrome;
    init_error_syndrome(&syndrome, 16);
    qgt_error_t err = detect_errors(NULL, &syndrome);
    assert(err == QGT_ERROR_INVALID_ARGUMENT);

    // Test NULL syndrome
    quantum_state_t* state = create_test_state();
    err = detect_errors(state, NULL);
    assert(err == QGT_ERROR_INVALID_ARGUMENT);

    // Test zero max errors
    ErrorSyndrome zero_syndrome;
    err = init_error_syndrome(&zero_syndrome, 0);
    assert(err == QGT_ERROR_INVALID_PARAMETER);

    cleanup_test_state(state);
    cleanup_error_syndrome(&syndrome);
    printf("Boundary case tests passed\n");
}

static void test_error_chains(void) {
    printf("Testing error chains...\n");

    quantum_state_t* state = create_test_state();
    ErrorSyndrome syndrome;
    init_error_syndrome(&syndrome, 16);

    // Create chain of errors
    for (size_t i = 0; i < 4; i++) {
        inject_error(state, i, ERROR_X);
    }

    qgt_error_t err = detect_errors(state, &syndrome);
    assert(err == QGT_SUCCESS);

    // Verify chain detection
    bool found_chain = false;
    for (size_t i = 1; i < syndrome.num_errors; i++) {
        if (syndrome.error_locations[i] == syndrome.error_locations[i-1] + 1) {
            found_chain = true;
            break;
        }
    }
    assert(found_chain);

    cleanup_error_syndrome(&syndrome);
    cleanup_test_state(state);
    printf("Error chain test passed\n");
}

static void test_error_patterns(void) {
    printf("Testing error patterns...\n");

    quantum_state_t* state = create_test_state();
    ErrorSyndrome syndrome;
    init_error_syndrome(&syndrome, 16);

    // Create repeating error pattern
    for (size_t i = 0; i < 3; i++) {
        size_t base = i * 4;
        inject_error(state, base, ERROR_X);
        inject_error(state, base + 1, ERROR_Z);
    }

    qgt_error_t err = detect_errors(state, &syndrome);
    assert(err == QGT_SUCCESS);

    // Verify pattern detection
    bool found_pattern = false;
    for (size_t i = 4; i < syndrome.num_errors; i++) {
        if (syndrome.error_types[i] == syndrome.error_types[i-4] &&
            syndrome.error_types[i+1] == syndrome.error_types[i-3]) {
            found_pattern = true;
            break;
        }
    }
    assert(found_pattern);

    cleanup_error_syndrome(&syndrome);
    cleanup_test_state(state);
    printf("Error pattern test passed\n");
}

static void test_error_correlations(void) {
    printf("Testing error correlations...\n");

    quantum_state_t* state = create_test_state();
    ErrorSyndrome syndrome;
    init_error_syndrome(&syndrome, 16);

    // Create correlated errors
    inject_error(state, 0, ERROR_X);
    inject_error(state, 1, ERROR_X);
    inject_error(state, 4, ERROR_Z);
    inject_error(state, 8, ERROR_Z);

    qgt_error_t err = detect_errors(state, &syndrome);
    assert(err == QGT_SUCCESS);

    // Verify correlations
    bool found_correlation = false;
    for (size_t i = 0; i < syndrome.num_errors; i++) {
        for (size_t j = i + 1; j < syndrome.num_errors; j++) {
            if (syndrome.error_types[i] == syndrome.error_types[j]) {
                found_correlation = true;
                break;
            }
        }
    }
    assert(found_correlation);

    cleanup_error_syndrome(&syndrome);
    cleanup_test_state(state);
    printf("Error correlation test passed\n");
}

static quantum_state_t* create_test_state(void) {
    quantum_state_t* state = malloc(sizeof(quantum_state_t));
    state->dimension = 32;  // 4x4 lattice with 2 qubits per site
    state->coordinates = calloc(state->dimension * 2, sizeof(double));
    
    // Initialize to |0⟩ state
    for (size_t i = 0; i < state->dimension; i++) {
        state->coordinates[i * 2] = 1.0;
        state->coordinates[i * 2 + 1] = 0.0;
    }
    
    return state;
}

static void verify_syndrome_state(const ErrorSyndrome* syndrome) {
    assert(syndrome->num_errors <= syndrome->max_errors);
    for (size_t i = 0; i < syndrome->num_errors; i++) {
        assert(syndrome->error_weights[i] >= 0.0 && syndrome->error_weights[i] <= 1.0);
    }
}

static void verify_correction_result(const quantum_state_t* state) {
    // Calculate fidelity with |0⟩ state
    double fidelity = 0.0;
    for (size_t i = 0; i < state->dimension; i++) {
        fidelity += state->coordinates[i * 2] * state->coordinates[i * 2];
    }
    assert(fidelity > 0.9);  // 90% fidelity threshold
}
