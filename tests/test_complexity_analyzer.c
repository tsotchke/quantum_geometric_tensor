#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "quantum_geometric/core/complexity_analyzer.h"
#include "quantum_geometric/core/tensor_operations.h"

// Test implementations with different complexities
static void cubic_algorithm(void* data, int size) {
    float* arr = (float*)data;
    // O(n³) implementation
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < size; k++) {
                arr[i * size + j] += k;
            }
        }
    }
}

static void quadratic_algorithm(void* data, int size) {
    float* arr = (float*)data;
    // O(n²) implementation
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            arr[i * size + j] = i + j;
        }
    }
}

static void linearithmic_algorithm(void* data, int size) {
    float* arr = (float*)data;
    // O(n log n) implementation - simulate sorting
    for (int i = 0; i < size; i++) {
        int log_steps = (int)(log2(size));
        for (int j = 0; j < log_steps; j++) {
            arr[i] += j;
        }
    }
}

static void linear_algorithm(void* data, int size) {
    float* arr = (float*)data;
    // O(n) implementation
    for (int i = 0; i < size; i++) {
        arr[i] = i;
    }
}

// Test matrix multiplication implementations
static void test_matrix_operations() {
    printf("\nTesting Matrix Multiplication Implementations:\n");
    printf("=============================================\n");

    // Create test matrices
    int test_sizes[] = {128, 256, 512, 1024};
    int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    printf("\nBaseline vs Hardware-Accelerated Implementations:\n");
    for (int i = 0; i < num_sizes; i++) {
        int size = test_sizes[i];
        float* A = tensor_create_random(size, size);
        float* B = tensor_create_random(size, size);
        float* C = tensor_create_zero(size, size);
        
        printf("\nMatrix size: %dx%d\n", size, size);
        
        // Time baseline implementation
        clock_t start = clock();
        cubic_algorithm(C, size);  // Use cubic as baseline
        clock_t end = clock();
        double baseline_time = ((double)(end - start)) / CLOCKS_PER_SEC;
        
        // Time optimized implementation
        start = clock();
        tensor_matmul(A, B, C, size);
        end = clock();
        double optimized_time = ((double)(end - start)) / CLOCKS_PER_SEC;
        
        printf("Baseline time: %.3f seconds\n", baseline_time);
        printf("Optimized time: %.3f seconds\n", optimized_time);
        printf("Speedup: %.2fx\n", baseline_time / optimized_time);
        
        tensor_free(A);
        tensor_free(B);
        tensor_free(C);
    }
}

// Test complexity analyzer accuracy
static void test_complexity_analyzer_accuracy() {
    printf("\nTesting Complexity Analyzer Accuracy:\n");
    printf("===================================\n");
    
    // Test with known complexity implementations
    verify_optimization_target("Linear Algorithm", linear_algorithm, 
                             100, 10000, 10, 1.0);
    
    verify_optimization_target("Linearithmic Algorithm", linearithmic_algorithm,
                             100, 10000, 10, 1.2);
    
    verify_optimization_target("Quadratic Algorithm", quadratic_algorithm,
                             100, 10000, 10, 2.0);
    
    verify_optimization_target("Cubic Algorithm", cubic_algorithm,
                             100, 10000, 10, 3.0);
}

// Test hardware acceleration impact
static void test_hardware_acceleration() {
    printf("\nTesting Hardware Acceleration Impact:\n");
    printf("===================================\n");
    
    // Compare implementations
    compare_implementations("Matrix Multiplication",
                          cubic_algorithm,  // Baseline cubic implementation
                          tensor_matmul,    // Optimized implementation
                          128, 1024, 5);
    
    // Verify optimized implementation meets target
    verify_optimization_target("Hardware Accelerated Matrix Multiplication",
                             tensor_matmul,
                             128, 1024, 5,
                             2.0);  // Target: O(n²) or better
}

int main() {
    printf("Running Complexity Analyzer Tests\n");
    printf("================================\n");
    
    // Test complexity analyzer accuracy
    test_complexity_analyzer_accuracy();
    
    // Test matrix operations
    test_matrix_operations();
    
    // Test hardware acceleration
    test_hardware_acceleration();
    
    printf("\nAll tests completed!\n");
    return 0;
}
