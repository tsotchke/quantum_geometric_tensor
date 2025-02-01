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
    state->num_qubits = num_qubits;
    state->amplitudes = calloc(num_qubits * 2, sizeof(double));
    // Initialize to |0⟩ state
    for (size_t i = 0; i < num_qubits; i++) {
        state->amplitudes[i * 2] = 1.0;
    }
    return state;
}

static void cleanup_test_state(quantum_state* state) {
    if (state) {
        free(state->amplitudes);
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
    size_t* indices = create_test_indices(num_qubits);
    
    // Allocate results arrays
    size_t num_stabilizers = num_qubits / 4;  // 4 qubits per stabilizer
    double* parallel_results = calloc(num_stabilizers, sizeof(double));
    double* serial_results = calloc(num_stabilizers, sizeof(double));

    // Perform parallel measurement
    bool success = measure_stabilizers_parallel(state, indices, num_qubits,
                                             STABILIZER_PLAQUETTE,
                                             num_threads, parallel_results);
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

    size_t num_qubits = 1024;  // Large enough to see scaling effects
    quantum_state* state = create_test_state(num_qubits);
    size_t* indices = create_test_indices(num_qubits);
    size_t num_stabilizers = num_qubits / 4;
    double* results = calloc(num_stabilizers, sizeof(double));

    // Test with different thread counts
    size_t thread_counts[] = {1, 2, 4, 8, 16};
    clock_t times[5];

    for (size_t i = 0; i < 5; i++) {
        clock_t start = clock();
        bool success = measure_stabilizers_parallel(state, indices, num_qubits,
                                                 STABILIZER_PLAQUETTE,
                                                 thread_counts[i], results);
        clock_t end = clock();
        times[i] = end - start;
        
        assert(success && "Thread scaling measurement failed");
        
        // Verify scaling (should see improvement up to hardware thread count)
        if (i > 0) {
            double speedup = (double)times[0] / times[i];
            printf("Speedup with %zu threads: %.2fx\n",
                   thread_counts[i], speedup);
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
                                             1, NULL);
    assert(!success && "Should fail with null pointers");

    // Test invalid thread count
    quantum_state* state = create_test_state(4);
    size_t* indices = create_test_indices(4);
    double results[1];
    success = measure_stabilizers_parallel(state, indices, 4,
                                        STABILIZER_PLAQUETTE,
                                        0, results);
    assert(!success && "Should fail with zero threads");

    // Test invalid qubit count
    success = measure_stabilizers_parallel(state, indices, 0,
                                        STABILIZER_PLAQUETTE,
                                        1, results);
    assert(!success && "Should fail with zero qubits");

    // Test invalid stabilizer type
    success = measure_stabilizers_parallel(state, indices, 4,
                                        (StabilizerType)999,
                                        1, results);
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
    size_t* indices = create_test_indices(num_qubits);
    size_t num_stabilizers = num_qubits / 4;
    double* results = calloc(num_stabilizers, sizeof(double));

    bool success = measure_stabilizers_parallel(state, indices, num_qubits,
                                             STABILIZER_PLAQUETTE,
                                             num_threads, results);
    assert(success && "Workload distribution measurement failed");

    // Verify results are complete (no gaps)
    for (size_t i = 0; i < num_stabilizers; i++) {
        assert(fabs(results[i]) > 0.0 && "Missing measurement result");
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
    size_t* indices = create_test_indices(num_qubits);
    size_t num_stabilizers = num_qubits / 4;
    double* results = calloc(num_stabilizers, sizeof(double));

    bool success = measure_stabilizers_parallel(state, indices, num_qubits,
                                             STABILIZER_VERTEX,
                                             num_threads, results);
    assert(success && "Vertex stabilizer measurement failed");

    // Verify X-basis measurements
    for (size_t i = 0; i < num_stabilizers; i++) {
        assert(fabs(results[i]) <= 1.0 && "Invalid X measurement");
    }

    cleanup_test_state(state);
    free(indices);
    free(results);
    printf("Vertex stabilizers test passed\n");
}

int main() {
    printf("Running parallel stabilizer tests...\n\n");

    test_basic_parallel_measurement();
    test_thread_scaling();
    test_error_handling();
    test_workload_distribution();
    test_vertex_stabilizers();

    printf("\nAll parallel stabilizer tests passed!\n");
    return 0;
}
