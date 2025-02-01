#ifndef LOAD_ANALYZER_H
#define LOAD_ANALYZER_H

#include <stdbool.h>
#include <stddef.h>
#include <time.h>

// Load types
typedef enum {
    LOAD_CPU,              // CPU load
    LOAD_MEMORY,           // Memory load
    LOAD_QUANTUM,          // Quantum load
    LOAD_NETWORK,          // Network load
    LOAD_IO               // I/O load
} load_type_t;

// Analysis modes
typedef enum {
    ANALYZE_REALTIME,      // Real-time analysis
    ANALYZE_HISTORICAL,    // Historical analysis
    ANALYZE_PREDICTIVE,    // Predictive analysis
    ANALYZE_ADAPTIVE      // Adaptive analysis
} analysis_mode_t;

// Load levels
typedef enum {
    LEVEL_LIGHT,          // Light load
    LEVEL_MODERATE,       // Moderate load
    LEVEL_HEAVY,          // Heavy load
    LEVEL_CRITICAL,       // Critical load
    LEVEL_OVERLOAD       // Overload condition
} load_level_t;

// Resource types
typedef enum {
    RESOURCE_PROCESSOR,    // Processor resources
    RESOURCE_MEMORY,       // Memory resources
    RESOURCE_QUANTUM,      // Quantum resources
    RESOURCE_NETWORK      // Network resources
} resource_type_t;

// Analyzer configuration
typedef struct {
    analysis_mode_t mode;          // Analysis mode
    bool track_history;            // Track load history
    bool enable_prediction;        // Enable prediction
    bool monitor_thresholds;       // Monitor thresholds
    size_t window_size;           // Analysis window
    double threshold;             // Load threshold
} analyzer_config_t;

// Load metrics
typedef struct {
    load_type_t type;             // Load type
    load_level_t level;           // Load level
    double utilization;            // Resource utilization
    double throughput;             // Processing throughput
    double latency;                // Processing latency
    size_t queue_length;          // Queue length
} load_metrics_t;

// Resource metrics
typedef struct {
    resource_type_t type;         // Resource type
    double capacity;               // Resource capacity
    double usage;                  // Current usage
    double availability;           // Resource availability
    size_t active_tasks;          // Active tasks
    double efficiency;            // Resource efficiency
} resource_metrics_t;

// Load prediction
typedef struct {
    load_type_t type;             // Load type
    load_level_t predicted_level;  // Predicted level
    double predicted_value;        // Predicted value
    struct timespec timestamp;     // Prediction time
    double confidence;            // Prediction confidence
    void* prediction_data;       // Additional data
} load_prediction_t;

// Opaque analyzer handle
typedef struct load_analyzer_t load_analyzer_t;

// Core functions
load_analyzer_t* create_load_analyzer(const analyzer_config_t* config);
void destroy_load_analyzer(load_analyzer_t* analyzer);

// Analysis functions
bool analyze_load(load_analyzer_t* analyzer,
                 load_type_t type,
                 load_metrics_t* metrics);
bool analyze_resource_load(load_analyzer_t* analyzer,
                         resource_type_t resource,
                         resource_metrics_t* metrics);
bool analyze_system_load(load_analyzer_t* analyzer,
                        load_metrics_t* metrics);

// Monitoring functions
bool monitor_load(load_analyzer_t* analyzer,
                 load_type_t type,
                 load_metrics_t* metrics);
bool track_load_changes(load_analyzer_t* analyzer,
                       const load_metrics_t* metrics,
                       bool* significant_change);
bool get_load_history(const load_analyzer_t* analyzer,
                     load_metrics_t* history,
                     size_t* num_entries);

// Threshold functions
bool set_load_threshold(load_analyzer_t* analyzer,
                       load_type_t type,
                       double threshold);
bool check_threshold(load_analyzer_t* analyzer,
                    const load_metrics_t* metrics,
                    bool* threshold_exceeded);
bool adjust_thresholds(load_analyzer_t* analyzer,
                      const load_metrics_t* metrics);

// Prediction functions
bool predict_load(load_analyzer_t* analyzer,
                 load_type_t type,
                 load_prediction_t* prediction);
bool validate_prediction(load_analyzer_t* analyzer,
                        const load_prediction_t* prediction,
                        const load_metrics_t* actual);
bool update_prediction_model(load_analyzer_t* analyzer,
                           const load_metrics_t* metrics);

// Resource management
bool monitor_resource_usage(load_analyzer_t* analyzer,
                          resource_type_t resource,
                          resource_metrics_t* metrics);
bool optimize_resource_usage(load_analyzer_t* analyzer,
                           resource_type_t resource,
                           resource_metrics_t* metrics);
bool validate_resource_state(load_analyzer_t* analyzer,
                           const resource_metrics_t* metrics);

// Quantum-specific functions
bool analyze_quantum_load(load_analyzer_t* analyzer,
                        load_metrics_t* metrics);
bool predict_quantum_load(load_analyzer_t* analyzer,
                        load_prediction_t* prediction);
bool optimize_quantum_resources(load_analyzer_t* analyzer,
                              resource_metrics_t* metrics);

// Utility functions
bool export_analyzer_data(const load_analyzer_t* analyzer,
                         const char* filename);
bool import_analyzer_data(load_analyzer_t* analyzer,
                         const char* filename);
void free_load_metrics(load_metrics_t* metrics);

#endif // LOAD_ANALYZER_H
