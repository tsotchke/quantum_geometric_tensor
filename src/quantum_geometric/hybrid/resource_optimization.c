#include "quantum_geometric/hybrid/resource_optimization.h"
#include "quantum_geometric/hybrid/performance_monitoring.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Optimization parameters
#define MIN_QUANTUM_RATIO 0.1
#define MAX_QUANTUM_RATIO 0.9
#define OPTIMIZATION_INTERVAL 100
#define LEARNING_RATE 0.01

// Resource optimizer
typedef struct {
    double quantum_ratio;
    double classical_ratio;
    double* performance_history;
    size_t history_size;
    size_t history_capacity;
    OptimizationStrategy strategy;
    bool auto_tuning;
} ResourceOptimizer;

// Initialize resource optimizer
ResourceOptimizer* init_resource_optimizer(
    OptimizationStrategy strategy) {
    ResourceOptimizer* optimizer = malloc(sizeof(ResourceOptimizer));
    if (!optimizer) return NULL;
    
    optimizer->quantum_ratio = 0.5;  // Start with even split
    optimizer->classical_ratio = 0.5;
    optimizer->strategy = strategy;
    optimizer->auto_tuning = true;
    
    // Initialize performance history
    optimizer->history_capacity = 1000;
    optimizer->performance_history = malloc(
        optimizer->history_capacity * sizeof(double));
    
    if (!optimizer->performance_history) {
        free(optimizer);
        return NULL;
    }
    
    optimizer->history_size = 0;
    
    return optimizer;
}

// Update resource allocation
void update_resource_allocation(ResourceOptimizer* optimizer,
                              const PerformanceMetrics* metrics) {
    if (!optimizer || !metrics || !optimizer->auto_tuning) return;
    
    // Record performance
    double performance = compute_performance_score(metrics);
    if (optimizer->history_size < optimizer->history_capacity) {
        optimizer->performance_history[optimizer->history_size++] =
            performance;
    }
    
    // Update allocation based on strategy
    switch (optimizer->strategy) {
        case STRATEGY_PERFORMANCE:
            optimize_for_performance(optimizer, metrics);
            break;
            
        case STRATEGY_ENERGY:
            optimize_for_energy(optimizer, metrics);
            break;
            
        case STRATEGY_BALANCED:
            optimize_balanced(optimizer, metrics);
            break;
            
        case STRATEGY_ADAPTIVE:
            optimize_adaptive(optimizer, metrics);
            break;
    }
    
    // Ensure ratios stay within bounds
    optimizer->quantum_ratio = clamp(optimizer->quantum_ratio,
                                   MIN_QUANTUM_RATIO,
                                   MAX_QUANTUM_RATIO);
    optimizer->classical_ratio = 1.0 - optimizer->quantum_ratio;
}

// Optimize for maximum performance
static void optimize_for_performance(ResourceOptimizer* optimizer,
                                  const PerformanceMetrics* metrics) {
    // Calculate performance gradient
    double gradient = compute_performance_gradient(
        optimizer->performance_history,
        optimizer->history_size);
    
    // Adjust quantum ratio based on gradient
    if (gradient > 0) {
        // Performance improving with more quantum
        optimizer->quantum_ratio += LEARNING_RATE * gradient;
    } else {
        // Performance improving with more classical
        optimizer->quantum_ratio += LEARNING_RATE * gradient;
    }
}

// Optimize for minimum energy consumption
static void optimize_for_energy(ResourceOptimizer* optimizer,
                              const PerformanceMetrics* metrics) {
    double quantum_energy = metrics->quantum_energy_consumption;
    double classical_energy = metrics->classical_energy_consumption;
    
    // Compare energy efficiency
    double quantum_efficiency = quantum_energy /
        metrics->quantum_execution_time;
    double classical_efficiency = classical_energy /
        metrics->classical_execution_time;
    
    // Adjust ratio based on efficiency
    if (quantum_efficiency < classical_efficiency) {
        optimizer->quantum_ratio += LEARNING_RATE;
    } else {
        optimizer->quantum_ratio -= LEARNING_RATE;
    }
}

// Optimize for balanced resource usage
static void optimize_balanced(ResourceOptimizer* optimizer,
                            const PerformanceMetrics* metrics) {
    double quantum_util = metrics->quantum_resource_usage;
    double classical_util = (metrics->cpu_utilization +
                           metrics->gpu_utilization) / 2.0;
    
    // Balance utilization
    double util_diff = quantum_util - classical_util;
    optimizer->quantum_ratio -= LEARNING_RATE * util_diff;
}

// Adaptive optimization based on multiple factors
static void optimize_adaptive(ResourceOptimizer* optimizer,
                            const PerformanceMetrics* metrics) {
    // Weight different factors
    double performance_weight = 0.4;
    double energy_weight = 0.3;
    double balance_weight = 0.3;
    
    // Performance factor
    double perf_gradient = compute_performance_gradient(
        optimizer->performance_history,
        optimizer->history_size);
    
    // Energy factor
    double quantum_efficiency = metrics->quantum_energy_consumption /
        metrics->quantum_execution_time;
    double classical_efficiency = metrics->classical_energy_consumption /
        metrics->classical_execution_time;
    double energy_factor = (classical_efficiency - quantum_efficiency) /
        (classical_efficiency + quantum_efficiency);
    
    // Balance factor
    double quantum_util = metrics->quantum_resource_usage;
    double classical_util = (metrics->cpu_utilization +
                           metrics->gpu_utilization) / 2.0;
    double balance_factor = classical_util - quantum_util;
    
    // Combine factors
    double adjustment = performance_weight * perf_gradient +
                       energy_weight * energy_factor +
                       balance_weight * balance_factor;
    
    optimizer->quantum_ratio += LEARNING_RATE * adjustment;
}

// Get current resource allocation
ResourceAllocation get_resource_allocation(
    const ResourceOptimizer* optimizer) {
    ResourceAllocation allocation = {
        .quantum_ratio = optimizer->quantum_ratio,
        .classical_ratio = optimizer->classical_ratio
    };
    return allocation;
}

// Helper functions

static double compute_performance_score(
    const PerformanceMetrics* metrics) {
    // Weighted combination of metrics
    double time_score = 1.0 / (metrics->quantum_execution_time +
                              metrics->classical_execution_time +
                              metrics->communication_overhead);
    
    double error_score = metrics->total_fidelity;
    
    double resource_score = 1.0 - (
        metrics->quantum_resource_usage +
        metrics->cpu_utilization +
        metrics->gpu_utilization) / 3.0;
    
    return 0.5 * time_score +
           0.3 * error_score +
           0.2 * resource_score;
}

static double compute_performance_gradient(const double* history,
                                        size_t size) {
    if (size < 2) return 0.0;
    
    // Simple gradient over recent history
    size_t window = min(size, 10);
    double recent_avg = 0.0;
    double old_avg = 0.0;
    
    for (size_t i = 0; i < window; i++) {
        recent_avg += history[size - 1 - i];
        old_avg += history[size - 1 - window - i];
    }
    
    recent_avg /= window;
    old_avg /= window;
    
    return (recent_avg - old_avg) / window;
}

static double clamp(double value, double min_val, double max_val) {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

// Clean up resource optimizer
void cleanup_resource_optimizer(ResourceOptimizer* optimizer) {
    if (!optimizer) return;
    
    free(optimizer->performance_history);
    free(optimizer);
}
