#include "quantum_geometric/core/optimization_verifier.h"
#include "quantum_geometric/core/performance_monitor.h"
#include "quantum_geometric/core/bottleneck_detector.h"
#include <math.h>
#include <stdio.h>

// Verify algorithmic complexity
static double measure_complexity(size_t size, OperationType op_type) {
    double start_time = get_time();
    
    switch (op_type) {
        case FIELD_COUPLING:
            calculate_field_hierarchical(size);
            break;
            
        case TENSOR_CONTRACTION:
            contract_tensors_strassen(size);
            break;
            
        case GEOMETRIC_TRANSFORM:
            transform_geometric_fast(size);
            break;
            
        case MEMORY_ACCESS:
            access_memory_pattern(size);
            break;
            
        case COMMUNICATION:
            measure_communication(size);
            break;
    }
    
    double end_time = get_time();
    return end_time - start_time;
}

// Verify O(log n) complexity
bool verify_log_complexity(OperationType op_type) {
    const size_t NUM_SAMPLES = 5;
    const size_t BASE_SIZE = 16;
    
    double times[NUM_SAMPLES];
    double ratios[NUM_SAMPLES - 1];
    
    // Measure execution times for different input sizes
    for (size_t i = 0; i < NUM_SAMPLES; i++) {
        size_t size = BASE_SIZE * (1 << i);  // 16, 32, 64, 128, 256
        times[i] = measure_complexity(size, op_type);
    }
    
    // Calculate ratios between consecutive measurements
    for (size_t i = 0; i < NUM_SAMPLES - 1; i++) {
        ratios[i] = times[i + 1] / times[i];
    }
    
    // For O(log n), ratio should be approximately constant and close to 1
    // since log(2n)/log(n) approaches 1 as n grows
    double avg_ratio = 0;
    for (size_t i = 0; i < NUM_SAMPLES - 1; i++) {
        avg_ratio += ratios[i];
    }
    avg_ratio /= (NUM_SAMPLES - 1);
    
    // Check if ratios are consistent with O(log n)
    // Theoretical ratio for O(log n) should be around 1.1
    const double EXPECTED_RATIO = 1.1;
    const double TOLERANCE = 0.2;
    
    bool is_log_complexity = fabs(avg_ratio - EXPECTED_RATIO) < TOLERANCE;
    
    // Print verification results
    printf("Complexity verification for operation type %d:\n", op_type);
    printf("Average ratio: %.2f (expected: %.2f ± %.2f)\n", 
           avg_ratio, EXPECTED_RATIO, TOLERANCE);
    printf("Complexity appears to be: %s\n", 
           is_log_complexity ? "O(log n)" : "NOT O(log n)");
    
    return is_log_complexity;
}

// Verify all optimizations
OptimizationReport verify_all_optimizations() {
    OptimizationReport report = {0};
    
    // Check each operation type
    report.field_coupling_optimized = verify_log_complexity(FIELD_COUPLING);
    report.tensor_contraction_optimized = verify_log_complexity(TENSOR_CONTRACTION);
    report.geometric_transform_optimized = verify_log_complexity(GEOMETRIC_TRANSFORM);
    report.memory_access_optimized = verify_log_complexity(MEMORY_ACCESS);
    report.communication_optimized = verify_log_complexity(COMMUNICATION);
    
    // Check memory usage
    MemoryMetrics mem_metrics = measure_memory_usage();
    report.memory_efficiency = calculate_memory_efficiency(&mem_metrics);
    
    // Check GPU utilization
    GPUMetrics gpu_metrics = measure_gpu_utilization();
    report.gpu_efficiency = calculate_gpu_efficiency(&gpu_metrics);
    
    // Generate optimization suggestions
    if (!report.field_coupling_optimized) {
        add_suggestion(&report, "Field coupling operations need hierarchical optimization");
    }
    if (!report.tensor_contraction_optimized) {
        add_suggestion(&report, "Tensor contractions need Strassen algorithm optimization");
    }
    if (!report.geometric_transform_optimized) {
        add_suggestion(&report, "Geometric transformations need fast multipole optimization");
    }
    if (!report.memory_access_optimized) {
        add_suggestion(&report, "Memory access patterns need cache-aware optimization");
    }
    if (!report.communication_optimized) {
        add_suggestion(&report, "Communication patterns need hierarchical optimization");
    }
    
    // Check for remaining bottlenecks
    BottleneckReport bottleneck_report = detect_bottlenecks();
    for (int i = 0; i < bottleneck_report.num_bottlenecks; i++) {
        add_suggestion(&report, bottleneck_report.suggestions[i]);
    }
    
    return report;
}

// Print optimization report
void print_optimization_report(const OptimizationReport* report) {
    printf("\nOptimization Verification Report\n");
    printf("===============================\n\n");
    
    printf("1. Algorithm Complexity\n");
    printf("   - Field Coupling: %s\n", 
           report->field_coupling_optimized ? "O(log n)" : "Needs optimization");
    printf("   - Tensor Contraction: %s\n", 
           report->tensor_contraction_optimized ? "O(log n)" : "Needs optimization");
    printf("   - Geometric Transform: %s\n", 
           report->geometric_transform_optimized ? "O(log n)" : "Needs optimization");
    printf("   - Memory Access: %s\n", 
           report->memory_access_optimized ? "O(log n)" : "Needs optimization");
    printf("   - Communication: %s\n", 
           report->communication_optimized ? "O(log n)" : "Needs optimization");
    
    printf("\n2. Resource Utilization\n");
    printf("   - Memory Efficiency: %.1f%%\n", report->memory_efficiency * 100);
    printf("   - GPU Efficiency: %.1f%%\n", report->gpu_efficiency * 100);
    
    printf("\n3. Optimization Suggestions\n");
    for (int i = 0; i < report->num_suggestions; i++) {
        printf("   - %s\n", report->suggestions[i]);
    }
    
    printf("\nOverall Status: %s\n", 
           (report->field_coupling_optimized &&
            report->tensor_contraction_optimized &&
            report->geometric_transform_optimized &&
            report->memory_access_optimized &&
            report->communication_optimized &&
            report->memory_efficiency > 0.9 &&
            report->gpu_efficiency > 0.9)
           ? "Fully Optimized"
           : "Needs Optimization");
}
