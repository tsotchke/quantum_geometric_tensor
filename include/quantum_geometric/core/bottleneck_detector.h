#ifndef BOTTLENECK_DETECTOR_H
#define BOTTLENECK_DETECTOR_H

#include <stdbool.h>
#include <stddef.h>
#include <time.h>

// Bottleneck types
typedef enum {
    BOTTLENECK_CPU,           // CPU bottleneck
    BOTTLENECK_MEMORY,        // Memory bottleneck
    BOTTLENECK_IO,            // I/O bottleneck
    BOTTLENECK_NETWORK,       // Network bottleneck
    BOTTLENECK_QUANTUM       // Quantum resource bottleneck
} bottleneck_type_t;

// Detection modes
typedef enum {
    DETECT_REALTIME,          // Real-time detection
    DETECT_PERIODIC,          // Periodic detection
    DETECT_THRESHOLD,         // Threshold-based detection
    DETECT_PREDICTIVE        // Predictive detection
} detection_mode_t;

// Severity levels
typedef enum {
    SEVERITY_LOW,             // Low severity
    SEVERITY_MEDIUM,          // Medium severity
    SEVERITY_HIGH,            // High severity
    SEVERITY_CRITICAL        // Critical severity
} severity_level_t;

// Resource types
typedef enum {
    RESOURCE_COMPUTE,         // Compute resources
    RESOURCE_STORAGE,         // Storage resources
    RESOURCE_BANDWIDTH,       // Bandwidth resources
    RESOURCE_QUBITS          // Quantum resources
} resource_type_t;

// Detector configuration
typedef struct {
    detection_mode_t mode;            // Detection mode
    size_t interval;                  // Detection interval
    double threshold;                 // Detection threshold
    bool enable_prediction;           // Enable prediction
    bool enable_mitigation;           // Enable mitigation
    bool track_history;              // Track history
    size_t history_size;             // History size
} detector_config_t;

// Bottleneck description
typedef struct {
    bottleneck_type_t type;          // Bottleneck type
    severity_level_t severity;        // Severity level
    resource_type_t resource;         // Affected resource
    double impact_score;              // Impact score
    struct timespec detection_time;   // Detection time
    char* description;               // Description
} bottleneck_desc_t;

// Resource metrics
typedef struct {
    resource_type_t type;            // Resource type
    double utilization;              // Resource utilization
    double saturation;               // Resource saturation
    double efficiency;               // Resource efficiency
    size_t queue_length;             // Queue length
    double response_time;            // Response time
} resource_metrics_t;

// Performance impact
typedef struct {
    double throughput_impact;         // Impact on throughput
    double latency_impact;            // Impact on latency
    double efficiency_impact;         // Impact on efficiency
    double resource_impact;           // Impact on resources
    size_t affected_components;       // Number of affected components
    double overall_impact;            // Overall impact score
} performance_impact_t;

// Mitigation strategy
typedef struct {
    bottleneck_type_t target;        // Target bottleneck
    resource_type_t resource;         // Target resource
    double expected_improvement;      // Expected improvement
    double confidence;               // Strategy confidence
    char* action_plan;              // Action plan
    void* strategy_data;            // Strategy-specific data
} mitigation_strategy_t;

// Opaque detector handle
typedef struct bottleneck_detector_t bottleneck_detector_t;

// Core functions
bottleneck_detector_t* create_bottleneck_detector(const detector_config_t* config);
void destroy_bottleneck_detector(bottleneck_detector_t* detector);

// Detection functions
bool detect_bottlenecks(bottleneck_detector_t* detector,
                       bottleneck_desc_t* bottlenecks,
                       size_t* num_bottlenecks);
bool analyze_bottleneck(bottleneck_detector_t* detector,
                       const bottleneck_desc_t* bottleneck,
                       performance_impact_t* impact);
bool validate_bottleneck(bottleneck_detector_t* detector,
                        const bottleneck_desc_t* bottleneck);

// Resource monitoring
bool monitor_resource(bottleneck_detector_t* detector,
                     resource_type_t resource,
                     resource_metrics_t* metrics);
bool update_resource_metrics(bottleneck_detector_t* detector,
                           const resource_metrics_t* metrics);
bool get_resource_metrics(const bottleneck_detector_t* detector,
                         resource_type_t resource,
                         resource_metrics_t* metrics);

// Impact analysis
bool analyze_performance_impact(bottleneck_detector_t* detector,
                              const bottleneck_desc_t* bottleneck,
                              performance_impact_t* impact);
bool predict_impact(bottleneck_detector_t* detector,
                   const bottleneck_desc_t* bottleneck,
                   performance_impact_t* predicted_impact);
bool validate_impact(bottleneck_detector_t* detector,
                    const performance_impact_t* predicted,
                    const performance_impact_t* actual);

// Mitigation functions
bool suggest_mitigation(bottleneck_detector_t* detector,
                       const bottleneck_desc_t* bottleneck,
                       mitigation_strategy_t* strategy);
bool apply_mitigation(bottleneck_detector_t* detector,
                     const mitigation_strategy_t* strategy);
bool validate_mitigation(bottleneck_detector_t* detector,
                        const mitigation_strategy_t* strategy,
                        performance_impact_t* improvement);

// History functions
bool get_bottleneck_history(const bottleneck_detector_t* detector,
                           bottleneck_desc_t* history,
                           size_t* num_entries);
bool get_impact_history(const bottleneck_detector_t* detector,
                       performance_impact_t* history,
                       size_t* num_entries);
bool clear_history(bottleneck_detector_t* detector);

// Utility functions
bool export_detector_data(const bottleneck_detector_t* detector,
                         const char* filename);
bool import_detector_data(bottleneck_detector_t* detector,
                         const char* filename);
void free_bottleneck_description(bottleneck_desc_t* desc);

#endif // BOTTLENECK_DETECTOR_H
