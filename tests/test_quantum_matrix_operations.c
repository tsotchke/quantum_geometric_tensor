#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "quantum_geometric/core/quantum_matrix_operations.h"
#include "quantum_geometric/core/quantum_geometric_logging.h"
#include "quantum_geometric/core/tensor_network.h"
#include "quantum_geometric/core/hierarchical_matrix.h"

#define EPSILON 1e-6
#define TEST_SIZE 4

static int tests_run = 0;
static int tests_passed = 0;

static double get_time_ms(struct timespec *start, struct timespec *end) {
    return (end->tv_sec - start->tv_sec) * 1000.0 + (end->tv_nsec - start->tv_nsec) / 1000000.0;
}

static void print_matrix(const float* matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%8.3f ", matrix[i * size + j]);
        }
        printf("\n");
    }
    printf("\n");
}

static bool test_matrix_decomposition() {
    printf("Testing matrix decomposition...\n");
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    tests_run++;

    const int size = TEST_SIZE;
    float* matrix = (float*)malloc(size * size * sizeof(float));
    float* U = (float*)malloc(size * size * sizeof(float));
    float* V = (float*)malloc(size * size * sizeof(float));

    if (!matrix || !U || !V) {
        printf("Memory allocation failed\n");
        free(matrix);
        free(U);
        free(V);
        return false;
    }

    // Initialize test matrix with some values
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i * size + j] = (float)(i + j + 1);
        }
    }

    printf("Original matrix:\n");
    print_matrix(matrix, size);

    struct timespec op_start, op_end;
    clock_gettime(CLOCK_MONOTONIC, &op_start);
    bool result = quantum_decompose_matrix(matrix, size, U, V);
    clock_gettime(CLOCK_MONOTONIC, &op_end);
    printf("  Operation time (matrix decomposition): %.3f ms\n", get_time_ms(&op_start, &op_end));

    if (!result) {
        printf("Matrix decomposition failed\n");
        free(matrix);
        free(U);
        free(V);
        return false;
    }

    printf("U matrix:\n");
    print_matrix(U, size);
    printf("V matrix:\n");
    print_matrix(V, size);

    // Verify decomposition by multiplying U and V
    float* reconstructed = (float*)malloc(size * size * sizeof(float));
    if (!reconstructed) {
        printf("Memory allocation failed for reconstruction\n");
        free(matrix);
        free(U);
        free(V);
        return false;
    }

    // Multiply U and V to reconstruct original matrix
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            float sum = 0;
            for (int k = 0; k < size; k++) {
                sum += U[i * size + k] * V[k * size + j];
            }
            reconstructed[i * size + j] = sum;
        }
    }

    // Check if reconstruction matches original within epsilon
    bool success = true;
    for (int i = 0; i < size * size; i++) {
        if (fabs(matrix[i] - reconstructed[i]) > EPSILON) {
            printf("Reconstruction mismatch at index %d: got %f, expected %f\n",
                   i, reconstructed[i], matrix[i]);
            success = false;
            break;
        }
    }

    free(matrix);
    free(U);
    free(V);
    free(reconstructed);

    if (success) {
        tests_passed++;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("Time taken: %.3f ms\n", get_time_ms(&start, &end));
    return success;
}

static bool test_condition_number() {
    printf("Testing condition number computation...\n");
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    tests_run++;

    const int size = TEST_SIZE;
    float** matrix = (float**)malloc(size * sizeof(float*));
    for (int i = 0; i < size; i++) {
        matrix[i] = (float*)malloc(size * sizeof(float));
    }

    // Initialize with a well-conditioned matrix
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i][j] = (i == j) ? 1.0f : 0.1f;  // Diagonal dominant
        }
    }

    float condition_number;
    struct timespec op_start, op_end;
    clock_gettime(CLOCK_MONOTONIC, &op_start);
    bool result = quantum_compute_condition_number(matrix, size, &condition_number);
    clock_gettime(CLOCK_MONOTONIC, &op_end);
    printf("  Operation time (condition number): %.3f ms\n", get_time_ms(&op_start, &op_end));

    if (!result) {
        printf("Condition number computation failed\n");
        for (int i = 0; i < size; i++) {
            free(matrix[i]);
        }
        free(matrix);
        return false;
    }

    printf("Condition number: %f\n", condition_number);

    // For a well-conditioned matrix, condition number should be relatively small
    bool success = condition_number < 100.0f;  // Reasonable threshold for test matrix

    for (int i = 0; i < size; i++) {
        free(matrix[i]);
    }
    free(matrix);

    if (success) {
        tests_passed++;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("Time taken: %.3f ms\n", get_time_ms(&start, &end));
    return success;
}

static bool test_tensor_network_conversion() {
    printf("Testing matrix to tensor network conversion...\n");
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    tests_run++;

    const int size = TEST_SIZE;
    float* matrix = (float*)malloc(size * size * sizeof(float));
    tensor_network_t network;

    if (!matrix) {
        printf("Memory allocation failed\n");
        return false;
    }

    // Initialize test matrix
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i * size + j] = (float)(i + j + 1);
        }
    }

    printf("Original matrix:\n");
    print_matrix(matrix, size);

    struct timespec op_start, op_end;
    clock_gettime(CLOCK_MONOTONIC, &op_start);
    bool result = quantum_matrix_to_tensor_network(matrix, size, &network);
    clock_gettime(CLOCK_MONOTONIC, &op_end);
    printf("  Operation time (tensor network conversion): %.3f ms\n", get_time_ms(&op_start, &op_end));

    if (!result) {
        printf("Tensor network conversion failed\n");
        free(matrix);
        return false;
    }

    // Basic validation of tensor network properties
    bool success = (network.num_tensors > 0);

    free(matrix);
    // Cleanup tensor network (assuming there's a cleanup function)
    // qg_tensor_network_cleanup(&network);

    if (success) {
        tests_passed++;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("Time taken: %.3f ms\n", get_time_ms(&start, &end));
    return success;
}

static bool test_hierarchical_conversion() {
    printf("Testing matrix to hierarchical conversion...\n");
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    tests_run++;

    const int size = TEST_SIZE;
    float* matrix = (float*)malloc(size * size * sizeof(float));
    HierarchicalMatrix hmatrix;

    if (!matrix) {
        printf("Memory allocation failed\n");
        return false;
    }

    // Initialize test matrix
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i * size + j] = (float)(i + j + 1);
        }
    }

    printf("Original matrix:\n");
    print_matrix(matrix, size);

    struct timespec op_start, op_end;
    clock_gettime(CLOCK_MONOTONIC, &op_start);
    bool result = quantum_matrix_to_hierarchical(matrix, size, &hmatrix);
    clock_gettime(CLOCK_MONOTONIC, &op_end);
    printf("  Operation time (hierarchical conversion): %.3f ms\n", get_time_ms(&op_start, &op_end));

    if (!result) {
        printf("Hierarchical conversion failed\n");
        free(matrix);
        return false;
    }

    // Basic validation of hierarchical matrix properties
    bool success = (hmatrix.size == size);

    free(matrix);
    // Cleanup hierarchical matrix (assuming there's a cleanup function)
    // hierarchical_matrix_cleanup(&hmatrix);

    if (success) {
        tests_passed++;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("Time taken: %.3f ms\n", get_time_ms(&start, &end));
    return success;
}

static bool test_decomposition_optimization() {
    printf("Testing decomposition optimization...\n");
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    tests_run++;

    const int size = TEST_SIZE;
    float* U = (float*)malloc(size * size * sizeof(float));
    float* V = (float*)malloc(size * size * sizeof(float));

    if (!U || !V) {
        printf("Memory allocation failed\n");
        free(U);
        free(V);
        return false;
    }

    // Initialize test matrices
    for (int i = 0; i < size * size; i++) {
        U[i] = (float)(i + 1);
        V[i] = (float)(size * size - i);
    }

    printf("Original U matrix:\n");
    print_matrix(U, size);
    printf("Original V matrix:\n");
    print_matrix(V, size);

    const float tolerance = 0.01f;
    struct timespec op_start, op_end;
    clock_gettime(CLOCK_MONOTONIC, &op_start);
    bool result = quantum_optimize_decomposition(U, V, size, tolerance);
    clock_gettime(CLOCK_MONOTONIC, &op_end);
    printf("  Operation time (decomposition optimization): %.3f ms\n", get_time_ms(&op_start, &op_end));

    if (!result) {
        printf("Decomposition optimization failed\n");
        free(U);
        free(V);
        return false;
    }

    printf("Optimized U matrix:\n");
    print_matrix(U, size);
    printf("Optimized V matrix:\n");
    print_matrix(V, size);

    // Basic validation that matrices changed
    bool success = true;
    for (int i = 0; i < size * size; i++) {
        if (U[i] == (float)(i + 1) && V[i] == (float)(size * size - i)) {
            // If both matrices are unchanged, optimization might have failed
            success = false;
            break;
        }
    }

    free(U);
    free(V);

    if (success) {
        tests_passed++;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("Time taken: %.3f ms\n", get_time_ms(&start, &end));
    return success;
}

static bool test_error_string() {
    printf("Testing error string retrieval...\n");
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    tests_run++;

    bool success = true;
    const char* error_str;

    // Test all error codes
    error_str = quantum_matrix_get_error_string(QUANTUM_MATRIX_SUCCESS);
    if (!error_str || strlen(error_str) == 0) {
        printf("Failed to get string for QUANTUM_MATRIX_SUCCESS\n");
        success = false;
    }

    error_str = quantum_matrix_get_error_string(QUANTUM_MATRIX_INVALID_INPUT);
    if (!error_str || strlen(error_str) == 0) {
        printf("Failed to get string for QUANTUM_MATRIX_INVALID_INPUT\n");
        success = false;
    }

    error_str = quantum_matrix_get_error_string(QUANTUM_MATRIX_DECOMPOSITION_FAILED);
    if (!error_str || strlen(error_str) == 0) {
        printf("Failed to get string for QUANTUM_MATRIX_DECOMPOSITION_FAILED\n");
        success = false;
    }

    error_str = quantum_matrix_get_error_string(QUANTUM_MATRIX_MEMORY_ERROR);
    if (!error_str || strlen(error_str) == 0) {
        printf("Failed to get string for QUANTUM_MATRIX_MEMORY_ERROR\n");
        success = false;
    }

    error_str = quantum_matrix_get_error_string(QUANTUM_MATRIX_NUMERICAL_ERROR);
    if (!error_str || strlen(error_str) == 0) {
        printf("Failed to get string for QUANTUM_MATRIX_NUMERICAL_ERROR\n");
        success = false;
    }

    if (success) {
        tests_passed++;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("Time taken: %.3f ms\n", get_time_ms(&start, &end));
    return success;
}

int main() {
    printf("Running quantum matrix operations tests...\n\n");
    struct timespec suite_start, suite_end;
    clock_gettime(CLOCK_MONOTONIC, &suite_start);

    test_matrix_decomposition();
    test_condition_number();
    test_tensor_network_conversion();
    test_hierarchical_conversion();
    test_decomposition_optimization();
    test_error_string();

    clock_gettime(CLOCK_MONOTONIC, &suite_end);
    printf("\nTest summary: %d/%d tests passed\n", tests_passed, tests_run);
    printf("Total time: %.3f ms\n", get_time_ms(&suite_start, &suite_end));
    return tests_passed == tests_run ? 0 : 1;
}
