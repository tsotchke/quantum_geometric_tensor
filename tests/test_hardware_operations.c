#include "../include/quantum_geometric/core/quantum_geometric_operations.h"
#include "../include/quantum_geometric/core/quantum_geometric_gpu.h"
#include "../include/quantum_geometric/hardware/metal/quantum_geometric_metal.h"
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

/* Test GPU operations */
static void test_gpu_operations(qgt_context_t* ctx) {
    printf("Testing GPU operations...\n");
    
    #ifdef QGT_ENABLE_GPU
    qgt_state_t* state;
    ASSERT_SUCCESS(qgt_create_state(ctx, TEST_NUM_QUBITS, &state));
    
    // Test GPU state transfer
    ASSERT_SUCCESS(qgt_gpu_transfer_state(ctx, state));
    
    // Test GPU geometric operations
    double axis[3] = {1/sqrt(3), 1/sqrt(3), 1/sqrt(3)};
    ASSERT_SUCCESS(qgt_gpu_geometric_rotate(ctx, state, M_PI/4, axis));
    
    // Test parallel transport on GPU
    double* path = malloc(3 * QGT_TEST_CIRCLE_PATH_POINTS * sizeof(double));
    for (size_t i = 0; i < QGT_TEST_CIRCLE_PATH_POINTS; i++) {
        double angle = 2.0 * M_PI * i / (QGT_TEST_CIRCLE_PATH_POINTS - 1);
        path[3*i] = cos(angle);
        path[3*i + 1] = sin(angle);
        path[3*i + 2] = 0;
    }
    
    ASSERT_SUCCESS(qgt_gpu_geometric_parallel_transport(ctx, state, path, QGT_TEST_CIRCLE_PATH_POINTS));
    
    // Test GPU memory management
    ASSERT_SUCCESS(qgt_gpu_sync(ctx));
    ASSERT_SUCCESS(qgt_gpu_retrieve_state(ctx, state));
    
    free(path);
    qgt_destroy_state(ctx, state);
    #else
    printf("  GPU support not enabled\n");
    #endif
    
    printf("✓ GPU operation tests passed\n");
}

/* Test Metal operations */
static void test_metal_operations(qgt_context_t* ctx) {
    printf("Testing Metal operations...\n");
    
    #if defined(QGT_ENABLE_METAL) && defined(__APPLE__)
    qgt_state_t* state;
    ASSERT_SUCCESS(qgt_create_state(ctx, TEST_NUM_QUBITS, &state));
    
    // Test Metal state transfer
    ASSERT_SUCCESS(qgt_metal_transfer_state(ctx, state));
    
    // Test Metal geometric operations
    double axis[3] = {1/sqrt(3), 1/sqrt(3), 1/sqrt(3)};
    ASSERT_SUCCESS(qgt_metal_geometric_rotate(ctx, state, M_PI/4, axis));
    
    // Test parallel transport on Metal
    double* path = malloc(3 * QGT_TEST_CIRCLE_PATH_POINTS * sizeof(double));
    for (size_t i = 0; i < QGT_TEST_CIRCLE_PATH_POINTS; i++) {
        double angle = 2.0 * M_PI * i / (QGT_TEST_CIRCLE_PATH_POINTS - 1);
        path[3*i] = cos(angle);
        path[3*i + 1] = sin(angle);
        path[3*i + 2] = 0;
    }
    
    ASSERT_SUCCESS(qgt_metal_geometric_parallel_transport(ctx, state, path, QGT_TEST_CIRCLE_PATH_POINTS));
    
    // Test Metal memory management
    ASSERT_SUCCESS(qgt_metal_sync(ctx));
    ASSERT_SUCCESS(qgt_metal_retrieve_state(ctx, state));
    
    free(path);
    qgt_destroy_state(ctx, state);
    #else
    printf("  Metal support not enabled or not on macOS\n");
    #endif
    
    printf("✓ Metal operation tests passed\n");
}

/* Test hardware error handling */
static void test_hardware_error_handling(qgt_context_t* ctx) {
    printf("Testing hardware error handling...\n");
    
    #ifdef QGT_ENABLE_GPU
    // Test GPU error handling
    ASSERT_ERROR(qgt_gpu_transfer_state(ctx, NULL), QGT_ERROR_INVALID_ARGUMENT);
    ASSERT_ERROR(qgt_gpu_retrieve_state(ctx, NULL), QGT_ERROR_INVALID_ARGUMENT);
    #endif
    
    #if defined(QGT_ENABLE_METAL) && defined(__APPLE__)
    // Test Metal error handling
    ASSERT_ERROR(qgt_metal_transfer_state(ctx, NULL), QGT_ERROR_INVALID_ARGUMENT);
    ASSERT_ERROR(qgt_metal_retrieve_state(ctx, NULL), QGT_ERROR_INVALID_ARGUMENT);
    #endif
    
    printf("✓ Hardware error handling tests passed\n");
}

/* Test hardware performance */
static void test_hardware_performance(qgt_context_t* ctx) {
    printf("Testing hardware performance...\n");
    
    qgt_state_t* state;
    ASSERT_SUCCESS(qgt_create_state(ctx, TEST_NUM_QUBITS, &state));
    
    double axis[3] = {1/sqrt(3), 1/sqrt(3), 1/sqrt(3)};
    const int num_iterations = 1000;
    
    // CPU baseline
    clock_t start = clock();
    for (int i = 0; i < num_iterations; i++) {
        ASSERT_SUCCESS(qgt_geometric_rotate(ctx, state, M_PI/4, axis));
    }
    clock_t cpu_time = clock() - start;
    
    #ifdef QGT_ENABLE_GPU
    // GPU performance
    ASSERT_SUCCESS(qgt_gpu_transfer_state(ctx, state));
    start = clock();
    for (int i = 0; i < num_iterations; i++) {
        ASSERT_SUCCESS(qgt_gpu_geometric_rotate(ctx, state, M_PI/4, axis));
    }
    ASSERT_SUCCESS(qgt_gpu_sync(ctx));
    clock_t gpu_time = clock() - start;
    printf("  GPU speedup: %.2fx\n", (double)cpu_time / gpu_time);
    #endif
    
    #if defined(QGT_ENABLE_METAL) && defined(__APPLE__)
    // Metal performance
    ASSERT_SUCCESS(qgt_metal_transfer_state(ctx, state));
    start = clock();
    for (int i = 0; i < num_iterations; i++) {
        ASSERT_SUCCESS(qgt_metal_geometric_rotate(ctx, state, M_PI/4, axis));
    }
    ASSERT_SUCCESS(qgt_metal_sync(ctx));
    clock_t metal_time = clock() - start;
    printf("  Metal speedup: %.2fx\n", (double)cpu_time / metal_time);
    #endif
    
    qgt_destroy_state(ctx, state);
    printf("✓ Hardware performance tests passed\n");
}

/* Main test runner */
int main() {
    printf("\nQuantum Geometric Hardware Operations Test Suite\n");
    printf("============================================\n\n");
    
    // Initialize random seed
    srand(time(NULL));
    
    // Create context
    qgt_context_t* ctx;
    ASSERT_SUCCESS(qgt_create_context(&ctx));
    
    // Run tests
    test_gpu_operations(ctx);
    test_metal_operations(ctx);
    test_hardware_error_handling(ctx);
    test_hardware_performance(ctx);
    
    // Cleanup
    qgt_destroy_context(ctx);
    
    printf("\nAll hardware tests passed successfully!\n");
    return 0;
}
