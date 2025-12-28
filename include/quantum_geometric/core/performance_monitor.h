/**
 * @file performance_monitor.h
 * @brief Performance monitoring system for quantum geometric computations
 *
 * Provides hardware performance counters, metrics collection, and monitoring
 * for both classical and quantum computing operations.
 */

#ifndef PERFORMANCE_MONITOR_H
#define PERFORMANCE_MONITOR_H

#include <stdint.h>
#include <stdbool.h>
#include "quantum_geometric/core/performance_operations.h"
#include "quantum_geometric/hardware/quantum_hardware_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Hardware performance counters
uint64_t get_page_faults(void);
uint64_t get_cache_misses(void);
uint64_t get_tlb_misses(void);

// Initialize performance monitoring (simple version)
void init_performance_monitor(void);

// Initialize performance monitoring with configuration
int initialize_performance_monitor(const char* config_path, const char* metrics_path);

// Get current performance metrics (uses shared performance_metrics_t)
performance_metrics_t get_current_performance_metrics(void);

// Get hardware metrics (legacy - uses PerformanceMetrics from hardware types)
PerformanceMetrics get_performance_metrics(void);

// Update performance metrics
void update_performance_metrics(PerformanceMetrics* metrics);

// Record metric by name
int register_performance_metric(const char* name, int type, void* parameters, size_t count);
int record_metric_value(const char* name, double value);

// Performance analysis
void analyze_performance(const performance_metrics_t* metrics);
void generate_recommendations(void);

// Quantum-specific performance measurement
double measure_flops(void);
double measure_memory_bandwidth(void);
double measure_cache_performance(void);
double measure_quantum_error_rate(void);
double measure_quantum_fidelity(void);
double measure_entanglement_fidelity(void);
double measure_gate_error_rate(void);

// Reset performance counters
void reset_performance_counters(void);

// Clean up performance monitoring
void cleanup_performance_monitor(void);

// Allocation statistics tracking (call from memory pool)
void update_allocation_stats(size_t requested, size_t allocated, bool cache_hit);

// Thread workload tracking (call from worker threads)
void register_thread_work(uint64_t work_units, uint64_t active_time_ns);

// Quantum metrics update API (call from error mitigation module)
void set_quantum_error_rate(double rate);
void set_quantum_fidelity(double fidelity);
void set_entanglement_fidelity(double fidelity);
void set_gate_error_rate(double rate);
void update_quantum_metrics(double error_rate, double fidelity,
                            double entanglement_fidelity, double gate_error_rate);

// Optimization parameter accessors
int get_recommended_thread_count(void);
void get_memory_optimization_params(size_t* block_size, size_t* prefetch_distance,
                                    bool* prefetch_enabled);
int get_numa_preferred_node(void);
void get_resource_allocation_params(double* cpu_weight, double* memory_weight,
                                    double* io_weight, size_t* max_memory,
                                    int* max_threads);

// Performance monitoring control
int start_performance_monitoring(void);
int stop_performance_monitoring(void);
bool is_performance_monitoring_active(void);

// Report generation
int generate_performance_report_txt(const char* filename);
int export_metrics_json(const char* filename);

#ifdef __cplusplus
}
#endif

#endif // PERFORMANCE_MONITOR_H
