#include "quantum_geometric/core/quantum_geometric_operations.h"
#include "mocks/mock_quantum_state.h"
#include "test_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <complex.h>
#include <time.h>

/* Test macros */
#define ASSERT_NEAR(x, y, tol) assert(fabs((x) - (y)) < (tol))
#define ASSERT_SUCCESS(x) assert((x) == QGT_SUCCESS)
#define ASSERT_ERROR(x, expected) assert((x) == (expected))

/* Test error correction */
static void test_error_correction(qgt_context_t* ctx) {
    printf("Testing error correction...\n");
    
    // Create entangled state
    qgt_state_t* state;
    ASSERT_SUCCESS(mock_create_entangled_state(ctx, 0, TEST_NUM_QUBITS, &state));
    
    // Apply error channel
    double error_rate = 0.01;
    ASSERT_SUCCESS(qgt_apply_error_channel(ctx, state, error_rate));
    
    // Apply error correction
    ASSERT_SUCCESS(qgt_apply_error_correction(ctx, state));
    
    // Verify state fidelity after correction
    qgt_state_t* reference;
    ASSERT_SUCCESS(mock_create_entangled_state(ctx, 0, TEST_NUM_QUBITS, &reference));
    
    double fidelity;
    ASSERT_SUCCESS(qgt_geometric_measure_fidelity(ctx, state, reference, &fidelity));
    ASSERT_NEAR(fidelity, 1.0, 0.1); // Allow some tolerance due to imperfect correction
    
    mock_destroy_state(ctx, reference);
    mock_destroy_state(ctx, state);
    printf("✓ Error correction tests passed\n");
}

/* Test fault tolerance */
static void test_fault_tolerance(qgt_context_t* ctx) {
    printf("Testing fault tolerance...\n");
    
    // Create logical qubit state
    qgt_state_t* logical_state;
    ASSERT_SUCCESS(qgt_create_logical_state(ctx, TEST_NUM_QUBITS, &logical_state));
    
    // Apply noisy operations
    double noise_strength = 0.01;
    ASSERT_SUCCESS(qgt_enable_noise_model(ctx, noise_strength));
    
    double axis[3] = {1/sqrt(3), 1/sqrt(3), 1/sqrt(3)};
    ASSERT_SUCCESS(qgt_geometric_rotate(ctx, logical_state, M_PI/4, axis));
    
    // Apply error correction cycle
    ASSERT_SUCCESS(qgt_apply_error_correction_cycle(ctx, logical_state));
    
    // Verify logical state preserved
    double logical_fidelity;
    ASSERT_SUCCESS(qgt_measure_logical_fidelity(ctx, logical_state, &logical_fidelity));
    ASSERT_NEAR(logical_fidelity, 1.0, 0.1);
    
    qgt_destroy_state(ctx, logical_state);
    printf("✓ Fault tolerance tests passed\n");
}

/* Test distributed execution */
static void test_distributed_execution(qgt_context_t* ctx) {
    printf("Testing distributed execution...\n");
    
    // Create distributed context
    qgt_distributed_context_t* dist_ctx;
    ASSERT_SUCCESS(qgt_create_distributed_context(ctx, 4, &dist_ctx)); // 4 nodes
    
    // Create distributed state
    qgt_state_t* state;
    ASSERT_SUCCESS(mock_create_entangled_state(ctx, 1, TEST_NUM_QUBITS, &state)); // GHZ state
    
    // Distribute state across nodes
    ASSERT_SUCCESS(qgt_distribute_state(dist_ctx, state));
    
    // Perform distributed geometric operations
    double axis[3] = {0, 0, 1};
    ASSERT_SUCCESS(qgt_distributed_geometric_rotate(dist_ctx, state, M_PI/2, axis));
    
    // Test distributed parallel transport
    double* path = malloc(3 * QGT_TEST_CIRCLE_PATH_POINTS * sizeof(double));
    for (size_t i = 0; i < QGT_TEST_CIRCLE_PATH_POINTS; i++) {
        double angle = 2.0 * M_PI * i / (QGT_TEST_CIRCLE_PATH_POINTS - 1);
        path[3*i] = cos(angle);
        path[3*i + 1] = sin(angle);
        path[3*i + 2] = 0;
    }
    
    ASSERT_SUCCESS(qgt_distributed_geometric_parallel_transport(dist_ctx, state, path, QGT_TEST_CIRCLE_PATH_POINTS));
    
    // Gather state back
    ASSERT_SUCCESS(qgt_gather_state(dist_ctx, state));
    
    // Verify state remains valid
    double metric[TEST_DIMENSION * TEST_DIMENSION];
    ASSERT_SUCCESS(qgt_geometric_compute_metric(ctx, state, metric));
    
    free(path);
    qgt_destroy_distributed_context(dist_ctx);
    mock_destroy_state(ctx, state);
    printf("✓ Distributed execution tests passed\n");
}

/* Test distributed error handling */
static void test_distributed_error_handling(qgt_context_t* ctx) {
    printf("Testing distributed error handling...\n");
    
    // Test invalid arguments
    ASSERT_ERROR(qgt_create_distributed_context(ctx, 0, NULL),
                QGT_ERROR_INVALID_ARGUMENT);
    
    qgt_distributed_context_t* dist_ctx;
    ASSERT_SUCCESS(qgt_create_distributed_context(ctx, 2, &dist_ctx));
    
    // Test node failures
    ASSERT_SUCCESS(qgt_simulate_node_failure(dist_ctx, 1));
    
    qgt_state_t* state;
    ASSERT_SUCCESS(mock_create_state(ctx, TEST_NUM_QUBITS, &state));
    
    // Verify operations fail gracefully with node failure
    double axis[3] = {1, 0, 0};
    ASSERT_ERROR(qgt_distributed_geometric_rotate(dist_ctx, state, M_PI, axis),
                QGT_ERROR_NODE_FAILURE);
    
    mock_destroy_state(ctx, state);
    qgt_destroy_distributed_context(dist_ctx);
    printf("✓ Distributed error handling tests passed\n");
}

/* Test distributed performance */
static void test_distributed_performance(qgt_context_t* ctx) {
    printf("Testing distributed performance...\n");
    
    qgt_state_t* state;
    ASSERT_SUCCESS(mock_create_state(ctx, TEST_NUM_QUBITS, &state));
    
    // Create distributed contexts with different numbers of nodes
    const int max_nodes = 4;
    clock_t times[max_nodes];
    
    double axis[3] = {1/sqrt(3), 1/sqrt(3), 1/sqrt(3)};
    const int num_iterations = 1000;
    
    // Single node baseline
    clock_t start = clock();
    for (int i = 0; i < num_iterations; i++) {
        ASSERT_SUCCESS(qgt_geometric_rotate(ctx, state, M_PI/4, axis));
    }
    times[0] = clock() - start;
    
    // Test with increasing number of nodes
    for (int num_nodes = 2; num_nodes <= max_nodes; num_nodes++) {
        qgt_distributed_context_t* dist_ctx;
        ASSERT_SUCCESS(qgt_create_distributed_context(ctx, num_nodes, &dist_ctx));
        
        ASSERT_SUCCESS(qgt_distribute_state(dist_ctx, state));
        
        start = clock();
        for (int i = 0; i < num_iterations; i++) {
            ASSERT_SUCCESS(qgt_distributed_geometric_rotate(dist_ctx, state, M_PI/4, axis));
        }
        times[num_nodes-1] = clock() - start;
        
        ASSERT_SUCCESS(qgt_gather_state(dist_ctx, state));
        qgt_destroy_distributed_context(dist_ctx);
        
        printf("  %d nodes speedup: %.2fx\n", num_nodes, 
               (double)times[0] / times[num_nodes-1]);
    }
    
    mock_destroy_state(ctx, state);
    printf("✓ Distributed performance tests passed\n");
}

/* Main test runner */
int main() {
    printf("\nQuantum Geometric Advanced Features Test Suite\n");
    printf("==========================================\n\n");
    
    // Initialize random seed
    srand(time(NULL));
    
    // Create context
    qgt_context_t* ctx;
    ASSERT_SUCCESS(qgt_create_context(&ctx));
    
    // Run tests
    test_error_correction(ctx);
    test_fault_tolerance(ctx);
    test_distributed_execution(ctx);
    test_distributed_error_handling(ctx);
    test_distributed_performance(ctx);
    
    // Cleanup
    qgt_destroy_context(ctx);
    
    printf("\nAll advanced feature tests passed successfully!\n");
    return 0;
}
