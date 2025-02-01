#ifndef METHOD_ANALYZER_H
#define METHOD_ANALYZER_H

#include <stdbool.h>
#include <stddef.h>
#include <time.h>

// Method types
typedef enum {
    METHOD_QUANTUM,         // Quantum method
    METHOD_CLASSICAL,       // Classical method
    METHOD_HYBRID,          // Hybrid method
    METHOD_GEOMETRIC,       // Geometric method
    METHOD_ADAPTIVE        // Adaptive method
} method_type_t;

// Analysis modes
typedef enum {
    ANALYZE_STATIC,        // Static analysis
    ANALYZE_DYNAMIC,       // Dynamic analysis
    ANALYZE_PROFILING,     // Profiling analysis
    ANALYZE_COMPARATIVE   // Comparative analysis
} analysis_mode_t;

// Performance metrics
typedef enum {
    METRIC_EXECUTION_TIME, // Execution time
    METRIC_COMPLEXITY,     // Computational complexity
    METRIC_ACCURACY,       // Method accuracy
    METRIC_EFFICIENCY,     // Resource efficiency
    METRIC_RELIABILITY    // Method reliability
} performance_metric_t;

// Optimization levels
typedef enum {
    OPT_LEVEL_NONE,       // No optimization
    OPT_LEVEL_BASIC,      // Basic optimization
    OPT_LEVEL_ADVANCED,   // Advanced optimization
    OPT_LEVEL_AGGRESSIVE // Aggressive optimization
} optimization_level_t;

// Analyzer configuration
typedef struct {
    analysis_mode_t mode;          // Analysis mode
    bool track_history;            // Track history
    bool enable_profiling;         // Enable profiling
    bool monitor_performance;      // Monitor performance
    size_t sample_size;           // Sample size
    double threshold;             // Analysis threshold
} analyzer_config_t;

// Method metrics
typedef struct {
    method_type_t type;            // Method type
    double execution_time;         // Execution time
    double accuracy;               // Method accuracy
    double efficiency;             // Resource efficiency
    size_t call_count;            // Call count
    void* method_data;           // Additional data
} method_metrics_t;

// Performance profile
typedef struct {
    performance_metric_t metric;   // Performance metric
    double value;                  // Metric value
    double baseline;               // Baseline value
    double improvement;            // Improvement ratio
    struct timespec timestamp;     // Profile timestamp
    void* profile_data;          // Additional data
} performance_profile_t;

// Optimization result
typedef struct {
    optimization_level_t level;    // Optimization level
    double improvement;            // Performance improvement
    double overhead;               // Optimization overhead
    bool is_stable;               // Stability flag
    char* description;           // Result description
    void* result_data;          // Additional data
} optimization_result_t;

// Opaque analyzer handle
typedef struct method_analyzer_t method_analyzer_t;

// Core functions
method_analyzer_t* create_method_analyzer(const analyzer_config_t* config);
void destroy_method_analyzer(method_analyzer_t* analyzer);

// Analysis functions
bool analyze_method(method_analyzer_t* analyzer,
                   method_type_t type,
                   method_metrics_t* metrics);
bool analyze_performance(method_analyzer_t* analyzer,
                        performance_metric_t metric,
                        performance_profile_t* profile);
bool analyze_complexity(method_analyzer_t* analyzer,
                       const method_metrics_t* metrics,
                       double* complexity);

// Profiling functions
bool start_profiling(method_analyzer_t* analyzer,
                    method_type_t type);
bool stop_profiling(method_analyzer_t* analyzer,
                   method_metrics_t* metrics);
bool collect_profile_data(method_analyzer_t* analyzer,
                         performance_profile_t* profile);

// Optimization functions
bool optimize_method(method_analyzer_t* analyzer,
                    method_type_t type,
                    optimization_result_t* result);
bool validate_optimization(method_analyzer_t* analyzer,
                         const optimization_result_t* result);
bool apply_optimization(method_analyzer_t* analyzer,
                       const optimization_result_t* result);

// Comparison functions
bool compare_methods(method_analyzer_t* analyzer,
                    const method_metrics_t* method1,
                    const method_metrics_t* method2,
                    int* comparison);
bool rank_methods(method_analyzer_t* analyzer,
                 method_metrics_t* methods,
                 size_t num_methods,
                 size_t* rankings);
bool select_best_method(method_analyzer_t* analyzer,
                       method_metrics_t* methods,
                       size_t num_methods,
                       method_type_t* best_method);

// Monitoring functions
bool monitor_method(method_analyzer_t* analyzer,
                   method_type_t type,
                   method_metrics_t* metrics);
bool track_performance(method_analyzer_t* analyzer,
                      performance_metric_t metric,
                      performance_profile_t* profile);
bool get_method_history(const method_analyzer_t* analyzer,
                       method_metrics_t* history,
                       size_t* num_entries);

// Quantum-specific functions
bool analyze_quantum_method(method_analyzer_t* analyzer,
                          method_metrics_t* metrics);
bool optimize_quantum_method(method_analyzer_t* analyzer,
                           optimization_result_t* result);
bool validate_quantum_performance(method_analyzer_t* analyzer,
                                const performance_profile_t* profile);

// Utility functions
bool export_analyzer_data(const method_analyzer_t* analyzer,
                         const char* filename);
bool import_analyzer_data(method_analyzer_t* analyzer,
                         const char* filename);
void free_method_metrics(method_metrics_t* metrics);

#endif // METHOD_ANALYZER_H
