/**
 * @file test_parallel_stabilizer.c
 * @brief Tests for parallel stabilizer measurement system
 */

#include "quantum_geometric/physics/parallel_stabilizer.h"
#include "quantum_geometric/physics/quantum_stabilizer.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <time.h>

// Test helper functions
static quantum_state* create_test_state(size_t num_qubits) {
    quantum_state* state = malloc(sizeof(quantum_state));
    if (!state) return NULL;

    state->num_qubits = num_qubits;
    state->dimension = 1UL << num_qubits;  // 2^num_qubits

    // Allocate coordinates as ComplexFloat array
    state->coordinates = calloc(state->dimension, sizeof(ComplexFloat));
    if (!state->coordinates) {
        free(state);
        return NULL;
    }

    // Initialize to |0⟩^n state (all qubits in |0⟩)
    // |0...0⟩ basis state has index 0
    state->coordinates[0].real = 1.0f;
    state->coordinates[0].imag = 0.0f;

    return state;
}

static void cleanup_test_state(quantum_state* state) {
    if (state) {
        free(state->coordinates);
        free(state);
    }
}

static size_t* create_test_indices(size_t num_qubits) {
    size_t* indices = malloc(num_qubits * sizeof(size_t));
    for (size_t i = 0; i < num_qubits; i++) {
        indices[i] = i;
    }
    return indices;
}

static bool compare_results(const double* parallel_results,
                          const double* serial_results,
                          size_t num_results,
                          double tolerance) {
    for (size_t i = 0; i < num_results; i++) {
        if (fabs(parallel_results[i] - serial_results[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

// Test cases
static void test_basic_parallel_measurement() {
    printf("Testing basic parallel measurement...\n");

    size_t num_qubits = 16;
    size_t num_threads = 4;
    quantum_state* state = create_test_state(num_qubits);
    assert(state && "Failed to create test state");

    size_t* indices = create_test_indices(num_qubits);
    assert(indices && "Failed to create test indices");

    // Allocate results arrays
    size_t num_stabilizers = num_qubits / 4;  // 4 qubits per stabilizer
    double* parallel_results = calloc(num_stabilizers, sizeof(double));
    double* serial_results = calloc(num_stabilizers, sizeof(double));

    // Perform parallel measurement
    bool success = measure_stabilizers_parallel(state, indices, num_qubits,
                                             STABILIZER_PLAQUETTE,
                                             num_threads, parallel_results, NULL);
    assert(success && "Parallel measurement failed");

    // Perform serial measurements for comparison
    for (size_t i = 0; i < num_stabilizers; i++) {
        size_t base_idx = i * 4;
        size_t stabilizer_indices[4] = {
            indices[base_idx],
            indices[base_idx + 1],
            indices[base_idx + 2],
            indices[base_idx + 3]
        };
        success = measure_stabilizer(state, stabilizer_indices, 4,
                                   STABILIZER_PLAQUETTE, &serial_results[i]);
        assert(success && "Serial measurement failed");
    }

    // Compare results
    assert(compare_results(parallel_results, serial_results,
                         num_stabilizers, 1e-6) &&
           "Parallel and serial results differ");

    // Cleanup
    cleanup_test_state(state);
    free(indices);
    free(parallel_results);
    free(serial_results);
    printf("Basic parallel measurement test passed\n");
}

static void test_thread_scaling() {
    printf("Testing thread scaling...\n");

    size_t num_qubits = 20;  // 2^20 = 1M states (reasonable for testing)
    quantum_state* state = create_test_state(num_qubits);
    if (!state) {
        printf("  SKIP: Could not allocate state for %zu qubits\n", num_qubits);
        return;
    }

    size_t* indices = create_test_indices(num_qubits);
    size_t num_stabilizers = num_qubits / 4;
    double* results = calloc(num_stabilizers, sizeof(double));

    // Test with different thread counts
    size_t thread_counts[] = {1, 2, 4, 8};
    clock_t times[4];

    for (size_t i = 0; i < 4; i++) {
        clock_t start = clock();
        bool success = measure_stabilizers_parallel(state, indices, num_qubits,
                                                 STABILIZER_PLAQUETTE,
                                                 thread_counts[i], results, NULL);
        clock_t end = clock();
        times[i] = end - start;

        assert(success && "Thread scaling measurement failed");

        // Verify scaling (should see improvement up to hardware thread count)
        if (i > 0 && times[0] > 0) {
            double speedup = (double)times[0] / times[i];
            printf("  Speedup with %zu threads: %.2fx\n",
                   thread_counts[i], speedup);
            // Allow for overhead - just verify we're not getting worse
            assert(speedup > 0.5 && "Insufficient scaling");
        }
    }

    cleanup_test_state(state);
    free(indices);
    free(results);
    printf("Thread scaling test passed\n");
}

static void test_error_handling() {
    printf("Testing error handling...\n");

    // Test null pointers
    bool success = measure_stabilizers_parallel(NULL, NULL, 0,
                                             STABILIZER_PLAQUETTE,
                                             1, NULL, NULL);
    assert(!success && "Should fail with null pointers");

    // Test invalid thread count
    quantum_state* state = create_test_state(4);
    assert(state && "Failed to create test state");

    size_t* indices = create_test_indices(4);
    double results[1];
    success = measure_stabilizers_parallel(state, indices, 4,
                                        STABILIZER_PLAQUETTE,
                                        0, results, NULL);
    assert(!success && "Should fail with zero threads");

    // Test invalid qubit count
    success = measure_stabilizers_parallel(state, indices, 0,
                                        STABILIZER_PLAQUETTE,
                                        1, results, NULL);
    assert(!success && "Should fail with zero qubits");

    // Test invalid stabilizer type
    success = measure_stabilizers_parallel(state, indices, 4,
                                        (StabilizerType)999,
                                        1, results, NULL);
    assert(!success && "Should fail with invalid stabilizer type");

    cleanup_test_state(state);
    free(indices);
    printf("Error handling test passed\n");
}

static void test_workload_distribution() {
    printf("Testing workload distribution...\n");

    size_t num_qubits = 32;
    size_t num_threads = 3;  // Non-power-of-2 for uneven distribution
    quantum_state* state = create_test_state(num_qubits);
    if (!state) {
        printf("  SKIP: Could not allocate state for %zu qubits\n", num_qubits);
        return;
    }

    size_t* indices = create_test_indices(num_qubits);
    size_t num_stabilizers = num_qubits / 4;
    double* results = calloc(num_stabilizers, sizeof(double));

    bool success = measure_stabilizers_parallel(state, indices, num_qubits,
                                             STABILIZER_PLAQUETTE,
                                             num_threads, results, NULL);
    assert(success && "Workload distribution measurement failed");

    // Verify results are complete (all should be +1 for |0...0⟩ state with Z stabilizers)
    for (size_t i = 0; i < num_stabilizers; i++) {
        // For |0...0⟩ state, all Z-stabilizer measurements should give +1
        // (times confidence weighting which is near 1.0)
        assert(results[i] > 0.5 && "Unexpected measurement result");
    }

    cleanup_test_state(state);
    free(indices);
    free(results);
    printf("Workload distribution test passed\n");
}

static void test_vertex_stabilizers() {
    printf("Testing vertex stabilizers...\n");

    size_t num_qubits = 16;
    size_t num_threads = 4;
    quantum_state* state = create_test_state(num_qubits);
    assert(state && "Failed to create test state");

    size_t* indices = create_test_indices(num_qubits);
    size_t num_stabilizers = num_qubits / 4;
    double* results = calloc(num_stabilizers, sizeof(double));

    bool success = measure_stabilizers_parallel(state, indices, num_qubits,
                                             STABILIZER_VERTEX,
                                             num_threads, results, NULL);
    assert(success && "Vertex stabilizer measurement failed");

    // Verify X-basis measurements are valid
    // For |0...0⟩ state, X-stabilizer (X⊗X⊗X⊗X) measurements
    // give expectation value that depends on the state structure
    for (size_t i = 0; i < num_stabilizers; i++) {
        // Result should be in [-1, 1] range (possibly scaled by confidence)
        assert(fabs(results[i]) <= 1.1 && "Invalid X measurement");
    }

    cleanup_test_state(state);
    free(indices);
    free(results);
    printf("Vertex stabilizers test passed\n");
}

static void test_single_stabilizer_measurement() {
    printf("Testing single stabilizer measurement...\n");

    size_t num_qubits = 4;
    quantum_state* state = create_test_state(num_qubits);
    assert(state && "Failed to create test state");

    size_t indices[4] = {0, 1, 2, 3};
    double result;

    // Test Z-stabilizer on |0000⟩
    bool success = measure_stabilizer(state, indices, 4, STABILIZER_PLAQUETTE, &result);
    assert(success && "Z-stabilizer measurement failed");
    // Z⊗Z⊗Z⊗Z |0000⟩ = +|0000⟩, eigenvalue = +1
    assert(fabs(result - 1.0) < 1e-6 && "Z-stabilizer should give +1 for |0000⟩");

    // Test X-stabilizer on |0000⟩
    success = measure_stabilizer(state, indices, 4, STABILIZER_VERTEX, &result);
    assert(success && "X-stabilizer measurement failed");
    // X⊗X⊗X⊗X |0000⟩ = |1111⟩, <0000|X^4|0000> = 0 unless we have the superposition
    // For |0000⟩, <X^4> = <0000|1111> = 0
    assert(fabs(result) < 1e-6 && "X-stabilizer should give 0 for |0000⟩");

    // Test with GHZ-like state: (|0000⟩ + |1111⟩)/√2
    state->coordinates[0].real = 1.0f / sqrtf(2.0f);
    state->coordinates[0].imag = 0.0f;
    state->coordinates[15].real = 1.0f / sqrtf(2.0f);  // |1111⟩ is index 15
    state->coordinates[15].imag = 0.0f;

    // Z-stabilizer: Z^4 |0000⟩ = +|0000⟩, Z^4 |1111⟩ = +|1111⟩
    // So <Z^4> = 1 for GHZ state
    success = measure_stabilizer(state, indices, 4, STABILIZER_PLAQUETTE, &result);
    assert(success && "Z-stabilizer measurement on GHZ failed");
    assert(fabs(result - 1.0) < 1e-6 && "Z-stabilizer should give +1 for GHZ state");

    // X-stabilizer: X^4 |0000⟩ = |1111⟩, X^4 |1111⟩ = |0000⟩
    // <GHZ|X^4|GHZ> = (1/2)(<0000| + <1111|)(|1111⟩ + |0000⟩) = 1
    success = measure_stabilizer(state, indices, 4, STABILIZER_VERTEX, &result);
    assert(success && "X-stabilizer measurement on GHZ failed");
    assert(fabs(result - 1.0) < 1e-6 && "X-stabilizer should give +1 for GHZ state");

    cleanup_test_state(state);
    printf("Single stabilizer measurement test passed\n");
}

int main() {
    printf("Running parallel stabilizer tests...\n\n");

    test_basic_parallel_measurement();
    test_thread_scaling();
    test_error_handling();
    test_workload_distribution();
    test_vertex_stabilizers();
    test_single_stabilizer_measurement();

    printf("\nAll parallel stabilizer tests passed!\n");
    return 0;
}
