#ifndef EXECUTION_ANALYZER_H
#define EXECUTION_ANALYZER_H

#include <stdbool.h>
#include <stddef.h>
#include <time.h>

// Execution types
typedef enum {
    EXEC_TYPE_QUANTUM,        // Quantum execution
    EXEC_TYPE_CLASSICAL,      // Classical execution
    EXEC_TYPE_HYBRID,         // Hybrid execution
    EXEC_TYPE_DISTRIBUTED,    // Distributed execution
    EXEC_TYPE_PARALLEL       // Parallel execution
} execution_type_t;

// Analysis modes
typedef enum {
    ANALYZE_RUNTIME,          // Runtime analysis
    ANALYZE_PROFILING,        // Profiling analysis
    ANALYZE_TRACING,          // Execution tracing
    ANALYZE_SAMPLING         // Sampling analysis
} analysis_mode_t;

// Execution phases
typedef enum {
    PHASE_INITIALIZATION,     // Initialization phase
    PHASE_COMPUTATION,        // Computation phase
    PHASE_COMMUNICATION,      // Communication phase
    PHASE_SYNCHRONIZATION,    // Synchronization phase
    PHASE_FINALIZATION      // Finalization phase
} execution_phase_t;

// Performance levels
typedef enum {
    PERF_LEVEL_OPTIMAL,      // Optimal performance
    PERF_LEVEL_GOOD,         // Good performance
    PERF_LEVEL_ACCEPTABLE,   // Acceptable performance
    PERF_LEVEL_POOR,         // Poor performance
    PERF_LEVEL_CRITICAL     // Critical performance
} performance_level_t;

// Analyzer configuration
typedef struct {
    analysis_mode_t mode;            // Analysis mode
    bool track_phases;               // Track execution phases
    bool enable_profiling;           // Enable profiling
    bool collect_metrics;            // Collect metrics
    size_t sample_interval;          // Sampling interval
    double threshold;               // Performance threshold
} analyzer_config_t;

// Execution metrics
typedef struct {
    execution_type_t type;           // Execution type
    execution_phase_t phase;         // Current phase
    double duration;                 // Execution duration
    double cpu_usage;                // CPU usage
    double memory_usage;             // Memory usage
    size_t operation_count;         // Operation count
} execution_metrics_t;

// Performance metrics
typedef struct {
    performance_level_t level;       // Performance level
    double throughput;               // Operation throughput
    double latency;                  // Operation latency
    double efficiency;               // Execution efficiency
    double utilization;              // Resource utilization
    size_t error_count;             // Error count
} performance_metrics_t;

// Resource utilization
typedef struct {
    double cpu_utilization;          // CPU utilization
    double memory_utilization;       // Memory utilization
    double network_utilization;      // Network utilization
    double storage_utilization;      // Storage utilization
    double quantum_utilization;      // Quantum resource utilization
    size_t active_resources;        // Active resources
} resource_utilization_t;

// Execution profile
typedef struct {
    execution_metrics_t metrics;     // Execution metrics
    performance_metrics_t performance; // Performance metrics
    resource_utilization_t resources; // Resource utilization
    struct timespec start_time;      // Start timestamp
    struct timespec end_time;        // End timestamp
    void* profile_data;             // Additional data
} execution_profile_t;

// Opaque analyzer handle
typedef struct execution_analyzer_t execution_analyzer_t;

// Core functions
execution_analyzer_t* create_execution_analyzer(const analyzer_config_t* config);
void destroy_execution_analyzer(execution_analyzer_t* analyzer);

// Analysis functions
bool analyze_execution(execution_analyzer_t* analyzer,
                      execution_type_t type,
                      execution_metrics_t* metrics);
bool analyze_performance(execution_analyzer_t* analyzer,
                        performance_metrics_t* metrics);
bool analyze_resource_utilization(execution_analyzer_t* analyzer,
                                resource_utilization_t* utilization);

// Profiling functions
bool start_profiling(execution_analyzer_t* analyzer,
                    execution_type_t type);
bool stop_profiling(execution_analyzer_t* analyzer,
                   execution_profile_t* profile);
bool collect_profile_data(execution_analyzer_t* analyzer,
                         execution_profile_t* profile);

// Phase tracking
bool track_execution_phase(execution_analyzer_t* analyzer,
                          execution_phase_t phase,
                          execution_metrics_t* metrics);
bool get_phase_metrics(const execution_analyzer_t* analyzer,
                      execution_phase_t phase,
                      execution_metrics_t* metrics);
bool validate_phase_transition(execution_analyzer_t* analyzer,
                             execution_phase_t from,
                             execution_phase_t to);

// Performance monitoring
bool monitor_performance(execution_analyzer_t* analyzer,
                        performance_metrics_t* metrics);
bool evaluate_performance(execution_analyzer_t* analyzer,
                         const performance_metrics_t* metrics,
                         performance_level_t* level);
bool detect_performance_issues(execution_analyzer_t* analyzer,
                             performance_metrics_t* issues);

// Resource monitoring
bool monitor_resources(execution_analyzer_t* analyzer,
                      resource_utilization_t* utilization);
bool track_resource_usage(execution_analyzer_t* analyzer,
                         const resource_utilization_t* usage);
bool detect_resource_bottlenecks(execution_analyzer_t* analyzer,
                                resource_utilization_t* bottlenecks);

// Quantum-specific functions
bool analyze_quantum_execution(execution_analyzer_t* analyzer,
                             execution_metrics_t* metrics);
bool profile_quantum_operations(execution_analyzer_t* analyzer,
                              execution_profile_t* profile);
bool validate_quantum_performance(execution_analyzer_t* analyzer,
                                const performance_metrics_t* metrics);

// Utility functions
bool export_analyzer_data(const execution_analyzer_t* analyzer,
                         const char* filename);
bool import_analyzer_data(execution_analyzer_t* analyzer,
                         const char* filename);
void free_execution_profile(execution_profile_t* profile);

#endif // EXECUTION_ANALYZER_H
