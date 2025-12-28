#ifndef PERFORMANCE_MONITORING_H
#define PERFORMANCE_MONITORING_H

#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

// Operation types for monitoring (prefixed to avoid conflicts)
typedef enum {
    MONITOR_OP_QUANTUM,
    MONITOR_OP_CLASSICAL,
    MONITOR_OP_COMMUNICATION
} MonitoringOperationType;

// Performance summary structure
typedef struct {
    double avg_quantum_time;
    double avg_classical_time;
    double avg_communication_time;
    double avg_error_rate;
    double avg_fidelity;
    double peak_memory;
    double peak_quantum_resources;
    double total_energy;
} PerformanceSummary;

// Monitoring metrics (prefixed to avoid conflicts with PerformanceMetrics in hardware types)
typedef struct {
    // Timing metrics
    double quantum_execution_time;
    double classical_execution_time;
    double communication_overhead;

    // Resource usage
    double cpu_utilization;
    double gpu_utilization;
    double memory_usage;
    double quantum_resource_usage;

    // Error metrics
    double quantum_error_rate;
    double classical_error_rate;
    double total_fidelity;

    // Energy metrics
    double quantum_energy_consumption;
    double classical_energy_consumption;

    // Performance counters
    size_t num_quantum_operations;
    size_t num_classical_operations;
    size_t num_communications;

    // Timestamps
    struct timespec start_time;
    struct timespec last_update;
} MonitoringMetrics;

// Performance monitor
typedef struct PerformanceMonitor {
    MonitoringMetrics current;
    MonitoringMetrics* history;
    size_t history_size;
    size_t history_capacity;
    bool monitoring_enabled;
    FILE* log_file;
} PerformanceMonitor;

// Configuration constants
#define MAX_QUANTUM_OPERATIONS 10000
#define QUANTUM_POWER_CONSUMPTION 1e-6  // Watts per operation
#define CPU_POWER_CONSUMPTION 100.0     // Watts at 100% utilization
#define GPU_POWER_CONSUMPTION 300.0     // Watts at 100% utilization

// Core functions
PerformanceMonitor* init_performance_monitor(void);
void cleanup_performance_monitor(PerformanceMonitor* monitor);

// Monitoring operations
void start_operation(PerformanceMonitor* monitor, MonitoringOperationType type);
void end_operation(PerformanceMonitor* monitor, MonitoringOperationType type);

// Analysis
PerformanceSummary get_performance_summary(const PerformanceMonitor* monitor);

// Quantum hardware state updates (call from quantum execution code)
void perf_update_quantum_state(size_t active_qubits, size_t circuit_depth,
                               size_t gates_executed, double coherence_time_used);
void perf_set_quantum_hardware(size_t max_qubits, double max_coherence_time,
                               double cryostat_power, double control_power);
void perf_record_error_rate(double error_rate);
void perf_record_fidelity(double fidelity);

// Aggregate metrics
double perf_get_average_error_rate(void);
double perf_get_average_fidelity(void);
double perf_get_network_bandwidth_mbps(void);

#ifdef __cplusplus
}
#endif

#endif // PERFORMANCE_MONITORING_H
