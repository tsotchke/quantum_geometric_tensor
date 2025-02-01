#include "quantum_geometric/core/optimization_verifier.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

// Test helper functions
static void test_complexity_verification(void) {
    printf("Testing complexity verification...\n");
    
    // Test field coupling
    bool field_optimized = verify_log_complexity(FIELD_COUPLING);
    assert(field_optimized && "Field coupling should be O(log n)");
    
    // Test tensor contraction
    bool tensor_optimized = verify_log_complexity(TENSOR_CONTRACTION);
    assert(tensor_optimized && "Tensor contraction should be O(log n)");
    
    // Test geometric transform
    bool transform_optimized = verify_log_complexity(GEOMETRIC_TRANSFORM);
    assert(transform_optimized && "Geometric transform should be O(log n)");
    
    // Test memory access
    bool memory_optimized = verify_log_complexity(MEMORY_ACCESS);
    assert(memory_optimized && "Memory access should be O(log n)");
    
    // Test communication
    bool comm_optimized = verify_log_complexity(COMMUNICATION);
    assert(comm_optimized && "Communication should be O(log n)");
    
    printf("All complexity verifications passed!\n");
}

static void test_memory_metrics(void) {
    printf("Testing memory metrics...\n");
    
    MemoryMetrics metrics = measure_memory_usage();
    
    // Verify memory pool efficiency
    assert(metrics.pool_utilization > 0 && "Memory pool should be utilized");
    assert(metrics.fragmentation_ratio < 0.2 && "Memory fragmentation should be low");
    
    // Verify cache performance
    double cache_hit_ratio = (double)metrics.cache_hits / 
                            (metrics.cache_hits + metrics.cache_misses);
    assert(cache_hit_ratio > 0.9 && "Cache hit ratio should be high");
    
    // Verify memory allocation
    assert(metrics.current_allocated <= metrics.peak_allocated && 
           "Current allocation should not exceed peak");
    
    printf("Memory metrics verification passed!\n");
}

static void test_gpu_metrics(void) {
    printf("Testing GPU metrics...\n");
    
    GPUMetrics metrics = measure_gpu_utilization();
    
    // Verify GPU utilization
    assert(metrics.compute_utilization > 0.8 && "GPU compute utilization should be high");
    assert(metrics.memory_utilization > 0.7 && "GPU memory utilization should be high");
    assert(metrics.bandwidth_utilization > 0.7 && "GPU bandwidth utilization should be high");
    
    // Verify memory usage
    assert(metrics.memory_used < metrics.memory_total && 
           "GPU memory usage should not exceed total");
    
    // Verify temperature and power
    assert(metrics.temperature < 85 && "GPU temperature should be within limits");
    assert(metrics.power_usage > 0 && "GPU power usage should be measurable");
    
    printf("GPU metrics verification passed!\n");
}

static void test_optimization_report(void) {
    printf("Testing optimization report generation...\n");
    
    OptimizationReport report = verify_all_optimizations();
    
    // Verify all optimizations are applied
    assert(report.field_coupling_optimized && 
           "Field coupling should be optimized");
    assert(report.tensor_contraction_optimized && 
           "Tensor contraction should be optimized");
    assert(report.geometric_transform_optimized && 
           "Geometric transform should be optimized");
    assert(report.memory_access_optimized && 
           "Memory access should be optimized");
    assert(report.communication_optimized && 
           "Communication should be optimized");
    
    // Verify resource efficiency
    assert(report.memory_efficiency > TARGET_MEMORY_EFFICIENCY && 
           "Memory efficiency should meet target");
    assert(report.gpu_efficiency > TARGET_GPU_EFFICIENCY && 
           "GPU efficiency should meet target");
    
    // Print report for inspection
    print_optimization_report(&report);
    
    printf("Optimization report verification passed!\n");
}

static void test_error_handling(void) {
    printf("Testing error handling...\n");
    
    // Test error string retrieval
    const char* error_str = get_verification_error_string(VERIFY_ERROR_COMPLEXITY);
    assert(error_str != NULL && "Error string should not be NULL");
    
    // Test last error tracking
    VerificationError last_error = get_last_verification_error();
    assert(last_error == VERIFY_SUCCESS || 
           last_error == VERIFY_ERROR_COMPLEXITY ||
           last_error == VERIFY_ERROR_MEMORY ||
           last_error == VERIFY_ERROR_GPU ||
           last_error == VERIFY_ERROR_COMMUNICATION);
    
    printf("Error handling verification passed!\n");
}

// Stress test with large data sizes
static void test_large_scale_operations(void) {
    printf("Testing large-scale operations...\n");
    
    const size_t LARGE_SIZE = 1024 * 1024;  // 1M elements
    
    // Test field operations
    calculate_field_hierarchical(LARGE_SIZE);
    
    // Test tensor operations
    contract_tensors_strassen(LARGE_SIZE);
    
    // Test geometric operations
    transform_geometric_fast(LARGE_SIZE);
    
    // Test memory patterns
    access_memory_pattern(LARGE_SIZE);
    
    // Test communication
    measure_communication(LARGE_SIZE);
    
    printf("Large-scale operations verification passed!\n");
}

// Main test function
int main(void) {
    printf("Running optimization verifier tests...\n\n");
    
    // Run all tests
    test_complexity_verification();
    printf("\n");
    
    test_memory_metrics();
    printf("\n");
    
    test_gpu_metrics();
    printf("\n");
    
    test_optimization_report();
    printf("\n");
    
    test_error_handling();
    printf("\n");
    
    test_large_scale_operations();
    printf("\n");
    
    printf("All optimization verifier tests passed successfully!\n");
    return 0;
}
