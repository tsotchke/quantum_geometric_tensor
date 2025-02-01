#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "quantum_geometric/core/complexity_analyzer.h"

// Structure to store timing data points
typedef struct {
    int size;
    double time;
} timing_point;

// Structure to store analysis results
typedef struct {
    double complexity_exponent;  // For O(n^x), this is x
    double r_squared;           // R² value for fit quality
    char* algorithm_name;
    int data_points;
    timing_point* measurements;
} complexity_analysis;

// Helper function to perform linear regression on log-transformed data
static void calculate_complexity(timing_point* points, int n, complexity_analysis* result) {
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
    double* log_sizes = malloc(n * sizeof(double));
    double* log_times = malloc(n * sizeof(double));
    
    // Transform data to log space
    for (int i = 0; i < n; i++) {
        log_sizes[i] = log2(points[i].size);
        log_times[i] = log2(points[i].time);
        sum_x += log_sizes[i];
        sum_y += log_times[i];
        sum_xy += log_sizes[i] * log_times[i];
        sum_xx += log_sizes[i] * log_sizes[i];
    }
    
    // Calculate linear regression coefficients
    double mean_x = sum_x / n;
    double mean_y = sum_y / n;
    result->complexity_exponent = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    
    // Calculate R² value
    double intercept = mean_y - result->complexity_exponent * mean_x;
    double ss_tot = 0, ss_res = 0;
    for (int i = 0; i < n; i++) {
        double predicted = result->complexity_exponent * log_sizes[i] + intercept;
        ss_res += (log_times[i] - predicted) * (log_times[i] - predicted);
        ss_tot += (log_times[i] - mean_y) * (log_times[i] - mean_y);
    }
    result->r_squared = 1 - (ss_res / ss_tot);
    
    free(log_sizes);
    free(log_times);
}

complexity_analysis* analyze_algorithm(const char* name, void (*func)(void*, int), int min_size, int max_size, int steps) {
    complexity_analysis* result = malloc(sizeof(complexity_analysis));
    result->algorithm_name = strdup(name);
    result->data_points = steps;
    result->measurements = malloc(steps * sizeof(timing_point));
    
    // Generate test sizes exponentially distributed between min_size and max_size
    double size_ratio = pow(((double)max_size / min_size), 1.0 / (steps - 1));
    
    for (int i = 0; i < steps; i++) {
        int size = (int)(min_size * pow(size_ratio, i));
        void* data = malloc(size * sizeof(float));  // Allocate test data
        
        // Time the algorithm
        clock_t start = clock();
        func(data, size);
        clock_t end = clock();
        
        result->measurements[i].size = size;
        result->measurements[i].time = ((double)(end - start)) / CLOCKS_PER_SEC;
        
        free(data);
    }
    
    // Calculate complexity metrics
    calculate_complexity(result->measurements, steps, result);
    
    return result;
}

void print_analysis(const complexity_analysis* analysis) {
    printf("\nComplexity Analysis for %s:\n", analysis->algorithm_name);
    printf("Estimated complexity: O(n^%.2f)\n", analysis->complexity_exponent);
    printf("R² value: %.4f\n", analysis->r_squared);
    printf("\nMeasurements:\n");
    printf("Size\t\tTime (s)\n");
    for (int i = 0; i < analysis->data_points; i++) {
        printf("%d\t\t%.6f\n", 
               analysis->measurements[i].size, 
               analysis->measurements[i].time);
    }
    printf("\n");
}

void free_analysis(complexity_analysis* analysis) {
    free(analysis->algorithm_name);
    free(analysis->measurements);
    free(analysis);
}

const char* get_complexity_class(double exponent) {
    if (exponent <= 1.1) return "Linear - O(n)";
    if (exponent <= 1.3) return "Linearithmic - O(n log n)";
    if (exponent <= 2.2) return "Quadratic - O(n²)";
    if (exponent <= 2.5) return "Between quadratic and cubic";
    if (exponent <= 3.2) return "Cubic - O(n³)";
    return "Higher than cubic";
}

void compare_implementations(const char* name,
                           void (*baseline_func)(void*, int),
                           void (*optimized_func)(void*, int),
                           int min_size, int max_size, int steps) {
    printf("\nComparing implementations of %s:\n", name);
    
    complexity_analysis* baseline = analyze_algorithm("Baseline", baseline_func, min_size, max_size, steps);
    complexity_analysis* optimized = analyze_algorithm("Optimized", optimized_func, min_size, max_size, steps);
    
    printf("\nBaseline implementation:\n");
    printf("Complexity class: %s\n", get_complexity_class(baseline->complexity_exponent));
    printf("Exact exponent: %.2f (R² = %.4f)\n", baseline->complexity_exponent, baseline->r_squared);
    
    printf("\nOptimized implementation:\n");
    printf("Complexity class: %s\n", get_complexity_class(optimized->complexity_exponent));
    printf("Exact exponent: %.2f (R² = %.4f)\n", optimized->complexity_exponent, optimized->r_squared);
    
    // Calculate and print speedup for each size
    printf("\nSpeedup analysis:\n");
    printf("Size\t\tSpeedup\n");
    for (int i = 0; i < steps; i++) {
        double speedup = baseline->measurements[i].time / optimized->measurements[i].time;
        printf("%d\t\t%.2fx\n", baseline->measurements[i].size, speedup);
    }
    
    free_analysis(baseline);
    free_analysis(optimized);
}

void verify_optimization_target(const char* name, void (*func)(void*, int),
                              int min_size, int max_size, int steps,
                              double target_exponent) {
    complexity_analysis* analysis = analyze_algorithm(name, func, min_size, max_size, steps);
    
    printf("\nOptimization target verification for %s:\n", name);
    printf("Target complexity: O(n^%.2f)\n", target_exponent);
    printf("Actual complexity: O(n^%.2f)\n", analysis->complexity_exponent);
    
    if (analysis->complexity_exponent <= target_exponent + 0.1) {
        printf("✅ Target achieved!\n");
    } else {
        printf("❌ Target not met. Current implementation is %.2f orders higher than target.\n",
               analysis->complexity_exponent - target_exponent);
    }
    
    free_analysis(analysis);
}
