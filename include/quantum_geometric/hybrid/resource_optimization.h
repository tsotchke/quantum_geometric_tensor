#ifndef RESOURCE_OPTIMIZATION_H
#define RESOURCE_OPTIMIZATION_H

#include <stddef.h>
#include <stdbool.h>
#include "quantum_geometric/hybrid/performance_monitoring.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Optimization Strategy
// ============================================================================

/**
 * @brief Optimization strategies for quantum/classical resource allocation
 *
 * These strategies determine how the optimizer adjusts the ratio of work
 * between quantum and classical processors.
 */
typedef enum {
    STRATEGY_PERFORMANCE,   // Maximize throughput and minimize latency
    STRATEGY_ENERGY,        // Minimize total energy consumption
    STRATEGY_BALANCED,      // Balance resource utilization across processors
    STRATEGY_ADAPTIVE       // Dynamically adapt based on multiple factors
} OptimizationStrategy;

// ============================================================================
// Resource Allocation
// ============================================================================

/**
 * @brief Resource allocation between quantum and classical processors
 */
typedef struct {
    double quantum_ratio;     // Ratio of work allocated to quantum (0.0-1.0)
    double classical_ratio;   // Ratio of work allocated to classical (0.0-1.0)
} ResourceAllocation;

// ============================================================================
// Resource Optimizer
// ============================================================================

/**
 * @brief Resource optimizer for quantum/classical workload distribution
 *
 * This optimizer continuously adjusts the allocation of work between
 * quantum and classical processors based on performance feedback.
 */
typedef struct {
    double quantum_ratio;           // Current quantum allocation ratio
    double classical_ratio;         // Current classical allocation ratio
    double* performance_history;    // History of performance scores
    size_t history_size;            // Current history size
    size_t history_capacity;        // Maximum history capacity
    OptimizationStrategy strategy;  // Current optimization strategy
    bool auto_tuning;               // Whether auto-tuning is enabled
} ResourceOptimizer;

// ============================================================================
// Public API
// ============================================================================

/**
 * @brief Initialize resource optimizer with specified strategy
 * @param strategy The optimization strategy to use
 * @return Pointer to new optimizer, or NULL on failure
 */
ResourceOptimizer* init_resource_optimizer(OptimizationStrategy strategy);

/**
 * @brief Update resource allocation based on current performance metrics
 * @param optimizer The optimizer to update
 * @param metrics Current performance metrics from the monitoring system
 */
void update_resource_allocation(ResourceOptimizer* optimizer,
                                const MonitoringMetrics* metrics);

/**
 * @brief Get current resource allocation ratios
 * @param optimizer The optimizer to query
 * @return Current resource allocation
 */
ResourceAllocation get_resource_allocation(const ResourceOptimizer* optimizer);

/**
 * @brief Clean up resource optimizer
 * @param optimizer The optimizer to clean up
 */
void cleanup_resource_optimizer(ResourceOptimizer* optimizer);

#ifdef __cplusplus
}
#endif

#endif // RESOURCE_OPTIMIZATION_H
