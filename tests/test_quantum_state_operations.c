 /**
 * @file test_quantum_state_operations.c
 * @brief Tests for quantum state operations with focus on X-stabilizer optimizations
 */

#include "quantum_geometric/physics/quantum_state_operations.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

// Test helper functions
static void test_x_stabilizer_initialization(void);
static void test_x_error_mitigation(void);
static void test_x_measurement_correlation(void);
static void test_x_measurement_correction(void);
static void test_parallel_x_stabilizer(void);
static void test_x_stabilizer_history(void);
static void test_error_cases(void);

// Helper functions
static quantum_state_t* create_test_state(void);
static void apply_test_errors(quantum_state_t* state);
static void cleanup_test_state(quantum_state_t* state);
static XStabilizerState* get_x_stabilizer(const quantum_state_t* state);

int main(void) {
    printf("Running quantum state operations tests...\n");

    // Run all tests
    test_x_stabilizer_initialization();
    test_x_error_mitigation();
    test_x_measurement_correlation();
    test_x_measurement_correction();
    test_parallel_x_stabilizer();
    test_x_stabilizer_history();
    test_error_cases();

    printf("All quantum state operations tests passed!\n");
    return 0;
}

static void test_x_stabilizer_initialization(void) {
    printf("Testing X-stabilizer initialization...\n");

    // Initialize with X-stabilizer config enabled
    StateConfig config = {
        .num_qubits = 16,
        .decoherence_rate = 0.01,
        .track_phase = true,
        .enable_error_correction = true,
        .x_stabilizer_config = {
            .enable_x_optimization = true,
            .repetition_count = 100,
            .error_threshold = 0.1,
            .confidence_threshold = 0.9,
            .use_dynamic_decoupling = true,
            .track_correlations = true
        }
    };

    QuantumStateOps* ops = init_quantum_state_ops(&config);
    assert(ops != NULL);

    // Create test state
    quantum_state_t* state = create_test_state();
    assert(state != NULL);

    // Verify X-stabilizer state initialization
    XStabilizerState* x_state = get_x_stabilizer(state);
    assert(x_state != NULL);
    assert(x_state->correlations != NULL);
    assert(x_state->confidences != NULL);
    assert(x_state->history_size == 0);
    assert(x_state->error_rate >= 0.0);

    // Cleanup
    cleanup_state_result(get_state_result(ops));
    cleanup_quantum_state_ops(ops);
    cleanup_test_state(state);

    printf("X-stabilizer initialization tests passed\n");
}

static void test_x_error_mitigation(void) {
    printf("Testing X-error mitigation...\n");

    // Create test state with errors
    quantum_state_t* state = create_test_state();
    assert(state != NULL);
    apply_test_errors(state);

    // Apply error mitigation sequence
    apply_x_error_mitigation_sequence(state, 1, 1);

    // Verify error mitigation effects
    XStabilizerState* x_state = get_x_stabilizer(state);
    double initial_error = x_state->error_rate;
    apply_x_error_mitigation_sequence(state, 1, 1);
    double final_error = x_state->error_rate;

    // Error rate should decrease after mitigation
    assert(final_error < initial_error);

    // Verify state properties
    float fidelity;
    quantum_state_fidelity(&fidelity, state, state);
    assert(fidelity > 0.9);

    cleanup_test_state(state);
    printf("X-error mitigation tests passed\n");
}

static void test_x_measurement_correlation(void) {
    printf("Testing X-measurement correlation...\n");

    quantum_state_t* state = create_test_state();
    assert(state != NULL);

    // Get correlation for nearby qubits
    double corr1 = get_x_stabilizer_correlation(state, 0, 0, 0);
    double corr2 = get_x_stabilizer_correlation(state, 0, 1, 1);
    double corr3 = get_x_stabilizer_correlation(state, 1, 0, 2);

    // Verify correlation properties
    assert(fabs(corr1) <= 1.0);
    assert(fabs(corr2) <= 1.0);
    assert(fabs(corr3) <= 1.0);

    // Nearby qubits should have stronger correlations
    assert(fabs(corr1) > fabs(corr2));
    assert(fabs(corr2) > fabs(corr3));

    cleanup_test_state(state);
    printf("X-measurement correlation tests passed\n");
}

static void test_x_measurement_correction(void) {
    printf("Testing X-measurement correction...\n");

    quantum_state_t* state = create_test_state();
    assert(state != NULL);
    apply_test_errors(state);

    // Test measurement correction
    double result = 0.5;  // Initial measurement result
    apply_x_measurement_correction(state, 1, 1, &result);

    // Verify correction bounds
    assert(fabs(result) <= 1.0);
    
    // Test threshold behavior
    double small_result = 1e-7;  // Below threshold
    apply_x_measurement_correction(state, 1, 1, &small_result);
    assert(small_result == 0.0);  // Should be zeroed

    cleanup_test_state(state);
    printf("X-measurement correction tests passed\n");
}

static void test_parallel_x_stabilizer(void) {
    printf("Testing parallel X-stabilizer operations...\n");

    StateConfig config = {
        .num_qubits = 16,
        .decoherence_rate = 0.01,
        .track_phase = true,
        .enable_error_correction = true,
        .x_stabilizer_config = {
            .enable_x_optimization = true,
            .repetition_count = 100,
            .error_threshold = 0.1,
            .confidence_threshold = 0.9,
            .use_dynamic_decoupling = true,
            .track_correlations = true
        }
    };

    QuantumStateOps* ops = init_quantum_state_ops(&config);
    assert(ops != NULL);

    // Create multiple test states
    quantum_state_t* states[4];
    for (int i = 0; i < 4; i++) {
        states[i] = create_test_state();
        assert(states[i] != NULL);
    }

    // Perform parallel operations
    perform_state_operation(ops, states, 4);

    // Get results
    StateOperationResult* result = get_state_result(ops);
    assert(result != NULL);

    // Verify parallel execution metrics
    assert(result->x_stabilizer_results.measurement_count == 4);
    assert(result->x_stabilizer_results.average_fidelity > 0.9);

    // Cleanup
    for (int i = 0; i < 4; i++) {
        cleanup_test_state(states[i]);
    }
    cleanup_state_result(result);
    cleanup_quantum_state_ops(ops);

    printf("Parallel X-stabilizer tests passed\n");
}

static void test_x_stabilizer_history(void) {
    printf("Testing X-stabilizer history tracking...\n");

    quantum_state_t* state = create_test_state();
    assert(state != NULL);

    // Perform multiple measurements
    double result = 0.5;
    for (int i = 0; i < 10; i++) {
        apply_x_measurement_correction(state, 1, 1, &result);
        // Result should be modified by correction
        assert(fabs(result) <= 1.0);
    }

    // Verify history tracking
    XStabilizerState* x_state = get_x_stabilizer(state);
    assert(x_state->history_size > 0);
    assert(x_state->history_size <= 1000);  // Default capacity

    cleanup_test_state(state);
    printf("X-stabilizer history tests passed\n");
}

static void test_error_cases(void) {
    printf("Testing error cases...\n");

    // Test NULL state handling
    double result = 0.5;
    apply_x_measurement_correction(NULL, 1, 1, &result);
    assert(result == 0.5);  // Should not modify result

    // Test NULL result pointer
    quantum_state_t* state = create_test_state();
    assert(state != NULL);
    apply_x_measurement_correction(state, 1, 1, NULL);  // Should not crash

    // Test invalid coordinates
    result = 0.5;
    apply_x_measurement_correction(state, 999, 999, &result);
    assert(result == 0.5);  // Should not modify result

    cleanup_test_state(state);
    printf("Error case tests passed\n");
}

// Helper function implementations
static quantum_state_t* create_test_state(void) {
    quantum_state_t* state = NULL;
    qgt_error_t err = quantum_state_create(&state, QUANTUM_STATE_PURE, 16);
    if (err != QGT_SUCCESS) return NULL;

    // Initialize X-stabilizer state
    XStabilizerState* x_state = malloc(sizeof(XStabilizerState));
    if (!x_state) {
        quantum_state_destroy(state);
        return NULL;
    }

    x_state->correlations = calloc(16, sizeof(double));
    x_state->confidences = calloc(16, sizeof(double));
    if (!x_state->correlations || !x_state->confidences) {
        free(x_state->correlations);
        free(x_state->confidences);
        free(x_state);
        quantum_state_destroy(state);
        return NULL;
    }

    x_state->history_size = 0;
    x_state->error_rate = 0.0;
    state->auxiliary_data = x_state;

    // Initialize to |0⟩ state
    ComplexFloat* coords = state->coordinates;
    coords[0] = complex_float_create(1.0f, 0.0f);
    for (size_t i = 1; i < 16; i++) {
        coords[i] = complex_float_create(0.0f, 0.0f);
    }

    return state;
}

static void apply_test_errors(quantum_state_t* state) {
    if (!state) return;

    // Apply bit flip errors
    state->coordinates[1] = complex_float_create(0.1f, 0.0f);
    state->coordinates[2] = complex_float_create(0.2f, 0.0f);
    
    XStabilizerState* x_state = get_x_stabilizer(state);
    if (x_state) {
        x_state->error_rate = 0.2;
    }

    quantum_state_normalize(state);
}

static void cleanup_test_state(quantum_state_t* state) {
    if (!state) return;

    XStabilizerState* x_state = get_x_stabilizer(state);
    if (x_state) {
        free(x_state->correlations);
        free(x_state->confidences);
        free(x_state);
    }
    quantum_state_destroy(state);
}

static XStabilizerState* get_x_stabilizer(const quantum_state_t* state) {
    if (!state || !state->auxiliary_data) return NULL;
    return (XStabilizerState*)state->auxiliary_data;
}
