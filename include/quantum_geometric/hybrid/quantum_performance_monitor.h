#ifndef QUANTUM_PERFORMANCE_MONITOR_H
#define QUANTUM_PERFORMANCE_MONITOR_H

#include <stddef.h>
#include <stdbool.h>
#include "quantum_geometric/hardware/quantum_hardware_abstraction.h"

#ifdef __cplusplus
extern "C" {
#endif

// Optimization strategy
typedef enum {
    OPTIMIZE_NONE,
    OPTIMIZE_MEMORY,
    OPTIMIZE_NETWORK,
    OPTIMIZE_COMPUTE,
    OPTIMIZE_ALL
} OptimizationStrategy;

// Forward declarations for internal structures
typedef struct MetricsTracker MetricsTracker;
typedef struct SystemAnalyzer SystemAnalyzer;
typedef struct OptimizationEngine OptimizationEngine;
typedef struct PerformanceState PerformanceState;

// Monitor configuration
typedef struct {
    double analyze_interval;        // Analysis interval in ms
    double threshold;               // Performance threshold
    OptimizationStrategy strategy;  // Initial optimization strategy
    bool aggressive;                // Aggressive optimization mode
} MonitorConfig;

// Performance monitor structure
typedef struct PerformanceMonitor {
    MetricsTracker* metrics;
    SystemAnalyzer* analyzer;
    OptimizationEngine* optimizer;
    PerformanceState* state;
} PerformanceMonitor;

// Performance metrics (public version for queries)
typedef struct {
    double total_runtime;
    double compute_time;
    double network_time;
    double memory_time;
    double throughput;
    double latency;
    double efficiency;
    double overhead;
} PerformanceSnapshot;

// Core functions
PerformanceMonitor* qpm_init(const MonitorConfig* config);
void qpm_cleanup(PerformanceMonitor* monitor);

int qpm_update(PerformanceMonitor* monitor, const QuantumOperation* op);
PerformanceSnapshot qpm_get_snapshot(const PerformanceMonitor* monitor);
OptimizationStrategy qpm_get_strategy(const PerformanceMonitor* monitor);
bool qpm_needs_optimization(const PerformanceMonitor* monitor);

// Internal component initialization (used by implementation)
MetricsTracker* init_metrics_tracker(size_t history_size, size_t update_interval);
void cleanup_metrics_tracker(MetricsTracker* tracker);

SystemAnalyzer* init_system_analyzer(double interval, double threshold);
void cleanup_system_analyzer(SystemAnalyzer* analyzer);

OptimizationEngine* init_optimization_engine(OptimizationStrategy strategy, bool aggressive);
void cleanup_optimization_engine(OptimizationEngine* engine);

PerformanceState* init_performance_state(void);
void cleanup_performance_state(PerformanceState* state);

// System resource queries
double get_total_memory(void);
double get_network_bandwidth(void);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_PERFORMANCE_MONITOR_H
