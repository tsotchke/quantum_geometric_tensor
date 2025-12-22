/**
 * @file performance_operations.h
 * @brief Cross-platform performance monitoring and profiling
 *
 * Provides high-precision timing, resource tracking, and performance
 * analysis capabilities that work across macOS (ARM/x86), Linux, and Windows.
 */

#ifndef PERFORMANCE_OPERATIONS_H
#define PERFORMANCE_OPERATIONS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

// Error codes for performance operations
typedef enum {
    QG_PERFORMANCE_SUCCESS = 0,
    QG_PERFORMANCE_ERROR_INVALID_PARAMETER = -1,
    QG_PERFORMANCE_ERROR_NOT_INITIALIZED = -2,
    QG_PERFORMANCE_ERROR_ALREADY_RUNNING = -3,
    QG_PERFORMANCE_ERROR_NOT_RUNNING = -4,
    QG_PERFORMANCE_ERROR_FILE_IO = -5,
    QG_PERFORMANCE_ERROR_INTERNAL = -6
} performance_error_t;

// Performance timer structure
typedef struct {
    const char* label;
    struct timespec start_time;
    struct timespec end_time;
    double elapsed_time;
    int is_running;
} performance_timer_t;

// Comprehensive performance metrics structure for classical and quantum computing
typedef struct {
    // Basic timing metrics
    double execution_time;      // Execution time in seconds
    size_t cpu_cycles;          // CPU cycles (if available)
    size_t instructions;        // Instruction count (if available)

    // Memory metrics
    size_t memory_usage;        // Current memory usage in bytes
    size_t peak_memory;         // Peak memory usage in bytes
    size_t page_faults;         // Page fault count

    // Computational metrics
    size_t flop_count;          // Number of floating-point operations
    double flops;               // FLOPS achieved
    double memory_bandwidth;    // Memory bandwidth (GB/s)
    size_t cache_misses;        // Cache miss count (if available)
    double cache_hit_rate;      // Cache hit rate (0.0-1.0)

    // Quantum computation metrics
    double quantum_error_rate;     // Overall quantum error rate
    double quantum_fidelity;       // Quantum state fidelity
    double entanglement_fidelity;  // Entanglement fidelity
    double gate_error_rate;        // Gate error rate

    // Resource utilization metrics
    double cpu_utilization;     // CPU utilization (0.0-1.0)
    double gpu_utilization;     // GPU utilization (0.0-1.0)

    // Communication metrics (MPI/distributed)
    double mpi_time;            // Time spent in MPI calls
    double network_bandwidth;   // Network bandwidth (GB/s)
    double latency;             // Communication latency
    double numa_local_ratio;    // NUMA local access ratio

    // Performance quality metrics
    double throughput;          // Operations per second
    double response_time;       // Response time
    double avg_latency;         // Average latency
    double error_rate;          // Error rate
    double queue_length;        // Queue length
    double wait_time;           // Wait time in queue

    // Resource allocation metrics
    double allocation_efficiency;  // Memory allocation efficiency
    double resource_contention;    // Resource contention level
    double load_balance;           // Load balance factor
    double resource_utilization;   // Overall resource utilization
} performance_metrics_t;

// Performance configuration
typedef struct {
    const char* log_file;       // Log file path (NULL for no logging)
    int log_level;              // Logging verbosity level
    int enable_profiling;       // Enable detailed profiling
    int collect_memory_stats;   // Track memory usage
    int collect_cache_stats;    // Track cache performance
    int collect_flops;          // Count floating-point operations
    int enable_visualization;   // Enable visualization output
} performance_config_t;

// Initialize performance monitoring
int qg_performance_init(const performance_config_t* config);

// Cleanup performance monitoring
void qg_performance_cleanup(void);

// Timer operations
int qg_timer_start(performance_timer_t* timer, const char* label);
int qg_timer_stop(performance_timer_t* timer);
double qg_timer_get_elapsed(const performance_timer_t* timer);
int qg_timer_reset(performance_timer_t* timer);

// Section-based monitoring
int qg_start_monitoring(const char* section_name);
int qg_stop_monitoring(const char* section_name);
int qg_get_performance_metrics(const char* section_name, performance_metrics_t* metrics);

// Memory tracking
size_t qg_get_current_memory_usage(void);
size_t qg_get_peak_memory_usage(void);
int qg_reset_memory_stats(void);

// Performance logging and reporting
int qg_log_performance_event(const char* event_name, const performance_metrics_t* metrics);
int qg_generate_performance_report(const char* filename);

// Error handling
const char* qg_performance_get_error_string(performance_error_t error);

// Utility functions
const char* qg_performance_get_version(void);
int qg_performance_set_log_level(int level);
int qg_performance_enable_feature(const char* feature_name);
int qg_performance_disable_feature(const char* feature_name);

// High-precision timing utilities
uint64_t qg_get_timestamp_ns(void);
double qg_get_time_seconds(void);

#ifdef __cplusplus
}
#endif

#endif // PERFORMANCE_OPERATIONS_H
