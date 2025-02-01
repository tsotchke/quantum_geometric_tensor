#ifndef EFFICIENCY_ANALYZER_H
#define EFFICIENCY_ANALYZER_H

#include <stdbool.h>
#include <stddef.h>
#include <time.h>

// Efficiency types
typedef enum {
    EFFICIENCY_COMPUTATIONAL,  // Computational efficiency
    EFFICIENCY_MEMORY,        // Memory efficiency
    EFFICIENCY_QUANTUM,       // Quantum efficiency
    EFFICIENCY_ENERGY,        // Energy efficiency
    EFFICIENCY_RESOURCE      // Resource efficiency
} efficiency_type_t;

// Analysis modes
typedef enum {
    ANALYZE_REALTIME,        // Real-time analysis
    ANALYZE_HISTORICAL,      // Historical analysis
    ANALYZE_PREDICTIVE,      // Predictive analysis
    ANALYZE_COMPARATIVE     // Comparative analysis
} analysis_mode_t;

// Optimization levels
typedef enum {
    OPT_LEVEL_NONE,         // No optimization
    OPT_LEVEL_BASIC,        // Basic optimization
    OPT_LEVEL_ADVANCED,     // Advanced optimization
    OPT_LEVEL_AGGRESSIVE   // Aggressive optimization
} optimization_level_t;

// Resource types
typedef enum {
    RESOURCE_CPU,           // CPU resources
    RESOURCE_GPU,           // GPU resources
    RESOURCE_QPU,           // Quantum processing unit
    RESOURCE_MEMORY,        // Memory resources
    RESOURCE_NETWORK       // Network resources
} resource_type_t;

// Analyzer configuration
typedef struct {
    analysis_mode_t mode;           // Analysis mode
    optimization_level_t opt_level; // Optimization level
    bool track_history;             // Track efficiency history
    bool enable_prediction;         // Enable prediction
    size_t window_size;            // Analysis window size
    double threshold;              // Efficiency threshold
} analyzer_config_t;

// Efficiency metrics
typedef struct {
    efficiency_type_t type;         // Efficiency type
    double score;                   // Efficiency score
    double utilization;             // Resource utilization
    double throughput;              // Processing throughput
    double overhead;                // Processing overhead
    double energy_usage;            // Energy usage
} efficiency_metrics_t;

// Resource metrics
typedef struct {
    resource_type_t type;           // Resource type
    double usage;                   // Resource usage
    double availability;            // Resource availability
    double efficiency;              // Resource efficiency
    double contention;              // Resource contention
    size_t active_tasks;           // Active tasks
} resource_metrics_t;

// Performance profile
typedef struct {
    double processing_speed;        // Processing speed
    double response_time;           // Response time
    double latency;                 // Operation latency
    double bandwidth;               // Data bandwidth
    double error_rate;             // Error rate
    size_t operations_count;       // Operation count
} performance_profile_t;

// Optimization result
typedef struct {
    bool success;                   // Optimization success
    double improvement;             // Efficiency improvement
    double cost_savings;            // Cost savings
    size_t optimizations;           // Number of optimizations
    char* description;             // Result description
    void* result_data;            // Additional data
} optimization_result_t;

// Opaque analyzer handle
typedef struct efficiency_analyzer_t efficiency_analyzer_t;

// Core functions
efficiency_analyzer_t* create_efficiency_analyzer(const analyzer_config_t* config);
void destroy_efficiency_analyzer(efficiency_analyzer_t* analyzer);

// Analysis functions
bool analyze_efficiency(efficiency_analyzer_t* analyzer,
                       efficiency_type_t type,
                       efficiency_metrics_t* metrics);
bool analyze_resource_efficiency(efficiency_analyzer_t* analyzer,
                               resource_type_t resource,
                               resource_metrics_t* metrics);
bool analyze_performance_efficiency(efficiency_analyzer_t* analyzer,
                                  performance_profile_t* profile);

// Monitoring functions
bool monitor_efficiency(efficiency_analyzer_t* analyzer,
                       efficiency_type_t type,
                       efficiency_metrics_t* metrics);
bool track_resource_usage(efficiency_analyzer_t* analyzer,
                         resource_type_t resource,
                         resource_metrics_t* metrics);
bool get_efficiency_history(const efficiency_analyzer_t* analyzer,
                          efficiency_metrics_t* history,
                          size_t* num_entries);

// Optimization functions
bool optimize_efficiency(efficiency_analyzer_t* analyzer,
                        efficiency_type_t type,
                        optimization_result_t* result);
bool optimize_resource_usage(efficiency_analyzer_t* analyzer,
                           resource_type_t resource,
                           optimization_result_t* result);
bool validate_optimization(efficiency_analyzer_t* analyzer,
                         const optimization_result_t* result);

// Prediction functions
bool predict_efficiency(efficiency_analyzer_t* analyzer,
                       efficiency_type_t type,
                       efficiency_metrics_t* prediction);
bool predict_resource_usage(efficiency_analyzer_t* analyzer,
                          resource_type_t resource,
                          resource_metrics_t* prediction);
bool validate_predictions(efficiency_analyzer_t* analyzer,
                         const efficiency_metrics_t* predicted,
                         const efficiency_metrics_t* actual);

// Quantum-specific functions
bool analyze_quantum_efficiency(efficiency_analyzer_t* analyzer,
                              efficiency_metrics_t* metrics);
bool optimize_quantum_operations(efficiency_analyzer_t* analyzer,
                               optimization_result_t* result);
bool validate_quantum_efficiency(efficiency_analyzer_t* analyzer,
                               const efficiency_metrics_t* metrics);

// Utility functions
bool export_analyzer_data(const efficiency_analyzer_t* analyzer,
                         const char* filename);
bool import_analyzer_data(efficiency_analyzer_t* analyzer,
                         const char* filename);
void free_optimization_result(optimization_result_t* result);

#endif // EFFICIENCY_ANALYZER_H
