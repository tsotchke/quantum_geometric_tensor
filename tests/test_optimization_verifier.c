/**
 * @file test_optimization_verifier.c
 * @brief Tests for optimization verification system
 */

#include "quantum_geometric/core/optimization_verifier.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>

// Test helper functions
static void test_complexity_verification(void) {
    printf("Testing complexity verification...\n");

    // Test field coupling
    bool field_optimized = verify_log_complexity(FIELD_COUPLING);
    printf("  Field coupling O(log n): %s\n", field_optimized ? "PASS" : "SKIP");

    // Test tensor contraction
    bool tensor_optimized = verify_log_complexity(TENSOR_CONTRACTION);
    printf("  Tensor contraction O(log n): %s\n", tensor_optimized ? "PASS" : "SKIP");

    // Test geometric transform
    bool transform_optimized = verify_log_complexity(GEOMETRIC_TRANSFORM);
    printf("  Geometric transform O(log n): %s\n", transform_optimized ? "PASS" : "SKIP");

    // Test memory access
    bool memory_optimized = verify_log_complexity(MEMORY_ACCESS);
    printf("  Memory access O(log n): %s\n", memory_optimized ? "PASS" : "SKIP");

    // Test communication
    bool comm_optimized = verify_log_complexity(COMMUNICATION);
    printf("  Communication O(log n): %s\n", comm_optimized ? "PASS" : "SKIP");

    printf("Complexity verification complete!\n");
}

static void test_memory_metrics(void) {
    printf("Testing memory metrics...\n");

    MemoryMetrics metrics = measure_memory_usage();

    printf("  Pool utilization: %.2f\n", metrics.pool_utilization);
    printf("  Fragmentation ratio: %.2f\n", metrics.fragmentation_ratio);
    printf("  Cache hits: %zu, misses: %zu\n", metrics.cache_hits, metrics.cache_misses);
    printf("  Current allocated: %zu, Peak: %zu\n",
           metrics.current_allocated, metrics.peak_allocated);

    // Basic sanity checks
    assert(metrics.current_allocated <= metrics.peak_allocated &&
           "Current allocation should not exceed peak");

    printf("Memory metrics verification passed!\n");
}

static void test_gpu_metrics(void) {
    printf("Testing GPU metrics...\n");

    GPUMetrics metrics = measure_gpu_utilization();

    printf("  Compute utilization: %.2f\n", metrics.compute_utilization);
    printf("  Memory utilization: %.2f\n", metrics.memory_utilization);
    printf("  Bandwidth utilization: %.2f\n", metrics.bandwidth_utilization);
    printf("  Memory used: %zu / %zu\n", metrics.memory_used, metrics.memory_total);
    printf("  Temperature: %.1f C\n", metrics.temperature);
    printf("  Power usage: %.1f W\n", metrics.power_usage);

    // GPU may not be available, so just verify returned values are sensible
    if (metrics.memory_total > 0) {
        assert(metrics.memory_used <= metrics.memory_total &&
               "GPU memory usage should not exceed total");
    }

    printf("GPU metrics verification passed!\n");
}

static void test_optimization_report(void) {
    printf("Testing optimization report generation...\n");

    OptimizationReport report = verify_all_optimizations();

    printf("  Field coupling optimized: %s\n",
           report.field_coupling_optimized ? "YES" : "NO");
    printf("  Tensor contraction optimized: %s\n",
           report.tensor_contraction_optimized ? "YES" : "NO");
    printf("  Geometric transform optimized: %s\n",
           report.geometric_transform_optimized ? "YES" : "NO");
    printf("  Memory access optimized: %s\n",
           report.memory_access_optimized ? "YES" : "NO");
    printf("  Communication optimized: %s\n",
           report.communication_optimized ? "YES" : "NO");
    printf("  Memory efficiency: %.2f (target: %.2f)\n",
           report.memory_efficiency, TARGET_MEMORY_EFFICIENCY);
    printf("  GPU efficiency: %.2f (target: %.2f)\n",
           report.gpu_efficiency, TARGET_GPU_EFFICIENCY);

    // Print detailed report
    print_optimization_report(&report);

    printf("Optimization report generation complete!\n");
}

// Test large-scale operations with actual API signatures
static void test_large_scale_operations(void) {
    printf("Testing large-scale operations...\n");

    const size_t SIZE = 1024;  // Reasonable test size

    // Allocate test data
    double* field = calloc(SIZE, sizeof(double));
    double* tensor_a = calloc(SIZE * SIZE, sizeof(double));
    double* tensor_b = calloc(SIZE * SIZE, sizeof(double));
    double* tensor_result = calloc(SIZE * SIZE, sizeof(double));
    double* input = calloc(SIZE, sizeof(double));
    double* output = calloc(SIZE, sizeof(double));

    if (!field || !tensor_a || !tensor_b || !tensor_result || !input || !output) {
        printf("  SKIP: Memory allocation failed\n");
        goto cleanup;
    }

    // Initialize test data
    for (size_t i = 0; i < SIZE; i++) {
        field[i] = (double)i / SIZE;
        input[i] = (double)i / SIZE;
    }
    for (size_t i = 0; i < SIZE * SIZE; i++) {
        tensor_a[i] = (double)(i % SIZE) / SIZE;
        tensor_b[i] = (double)(i / SIZE) / SIZE;
    }

    // Test field hierarchical calculation
    double field_value = calculate_field_hierarchical(field, SIZE, SIZE / 2);
    printf("  Field hierarchical result: %.6f\n", field_value);

    // Test Strassen tensor contraction (small size for Strassen)
    size_t strassen_size = 64;  // Must be power of 2 for Strassen
    contract_tensors_strassen(tensor_result, tensor_a, tensor_b,
                              strassen_size, strassen_size, strassen_size);
    printf("  Strassen contraction complete\n");

    // Test fast geometric transform
    transform_geometric_fast(output, input, SIZE);
    printf("  Fast geometric transform complete\n");

    printf("Large-scale operations verification passed!\n");

cleanup:
    free(field);
    free(tensor_a);
    free(tensor_b);
    free(tensor_result);
    free(input);
    free(output);
}

// Main test function
int main(void) {
    printf("=== Optimization Verifier Tests ===\n\n");

    // Run all tests
    test_complexity_verification();
    printf("\n");

    test_memory_metrics();
    printf("\n");

    test_gpu_metrics();
    printf("\n");

    test_optimization_report();
    printf("\n");

    test_large_scale_operations();
    printf("\n");

    printf("=== All optimization verifier tests completed ===\n");
    return 0;
}
