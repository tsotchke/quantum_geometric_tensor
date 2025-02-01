#ifndef IMPACT_ANALYZER_H
#define IMPACT_ANALYZER_H

#include <stdbool.h>
#include <stddef.h>
#include <time.h>

// Impact types
typedef enum {
    IMPACT_PERFORMANCE,     // Performance impact
    IMPACT_RESOURCE,        // Resource impact
    IMPACT_QUANTUM,         // Quantum impact
    IMPACT_SYSTEM,          // System impact
    IMPACT_GEOMETRIC       // Geometric impact
} impact_type_t;

// Analysis modes
typedef enum {
    ANALYZE_IMMEDIATE,      // Immediate analysis
    ANALYZE_CUMULATIVE,     // Cumulative analysis
    ANALYZE_PREDICTIVE,     // Predictive analysis
    ANALYZE_HISTORICAL     // Historical analysis
} analysis_mode_t;

// Severity levels
typedef enum {
    SEVERITY_NEGLIGIBLE,    // Negligible impact
    SEVERITY_MINOR,         // Minor impact
    SEVERITY_MODERATE,      // Moderate impact
    SEVERITY_MAJOR,         // Major impact
    SEVERITY_CRITICAL      // Critical impact
} severity_level_t;

// Scope types
typedef enum {
    SCOPE_LOCAL,           // Local scope
    SCOPE_COMPONENT,       // Component scope
    SCOPE_SYSTEM,          // System scope
    SCOPE_GLOBAL          // Global scope
} scope_type_t;

// Analyzer configuration
typedef struct {
    analysis_mode_t mode;          // Analysis mode
    bool track_history;            // Track impact history
    bool enable_prediction;        // Enable prediction
    bool monitor_cascading;        // Monitor cascading effects
    size_t window_size;           // Analysis window size
    double threshold;             // Impact threshold
} analyzer_config_t;

// Impact metrics
typedef struct {
    impact_type_t type;           // Impact type
    severity_level_t severity;     // Impact severity
    scope_type_t scope;           // Impact scope
    double magnitude;              // Impact magnitude
    double duration;               // Impact duration
    size_t affected_components;   // Affected components
} impact_metrics_t;

// Cascade effects
typedef struct {
    impact_metrics_t* impacts;     // Chain of impacts
    size_t num_impacts;            // Number of impacts
    double total_magnitude;        // Total magnitude
    double propagation_speed;      // Propagation speed
    bool is_contained;            // Containment status
    void* cascade_data;          // Additional data
} cascade_effects_t;

// Impact prediction
typedef struct {
    impact_type_t type;           // Predicted impact type
    severity_level_t severity;     // Predicted severity
    double probability;            // Occurrence probability
    struct timespec expected_time; // Expected time
    char* description;           // Impact description
    void* prediction_data;       // Additional data
} impact_prediction_t;

// Opaque analyzer handle
typedef struct impact_analyzer_t impact_analyzer_t;

// Core functions
impact_analyzer_t* create_impact_analyzer(const analyzer_config_t* config);
void destroy_impact_analyzer(impact_analyzer_t* analyzer);

// Analysis functions
bool analyze_impact(impact_analyzer_t* analyzer,
                   impact_type_t type,
                   impact_metrics_t* metrics);
bool analyze_cascade_effects(impact_analyzer_t* analyzer,
                           const impact_metrics_t* impact,
                           cascade_effects_t* effects);
bool analyze_system_impact(impact_analyzer_t* analyzer,
                         impact_metrics_t* metrics);

// Monitoring functions
bool monitor_impact(impact_analyzer_t* analyzer,
                   impact_type_t type,
                   impact_metrics_t* metrics);
bool track_cascade_effects(impact_analyzer_t* analyzer,
                         cascade_effects_t* effects);
bool get_impact_history(const impact_analyzer_t* analyzer,
                       impact_metrics_t* history,
                       size_t* num_entries);

// Assessment functions
bool assess_severity(impact_analyzer_t* analyzer,
                    const impact_metrics_t* metrics,
                    severity_level_t* level);
bool evaluate_scope(impact_analyzer_t* analyzer,
                   const impact_metrics_t* metrics,
                   scope_type_t* scope);
bool measure_magnitude(impact_analyzer_t* analyzer,
                      const impact_metrics_t* metrics,
                      double* magnitude);

// Prediction functions
bool predict_impact(impact_analyzer_t* analyzer,
                   impact_type_t type,
                   impact_prediction_t* prediction);
bool predict_cascade_effects(impact_analyzer_t* analyzer,
                           const impact_metrics_t* impact,
                           cascade_effects_t* effects);
bool validate_predictions(impact_analyzer_t* analyzer,
                         const impact_prediction_t* prediction,
                         const impact_metrics_t* actual);

// Mitigation functions
bool suggest_mitigation(impact_analyzer_t* analyzer,
                       const impact_metrics_t* metrics,
                       char** mitigation_plan);
bool evaluate_mitigation(impact_analyzer_t* analyzer,
                        const char* mitigation_plan,
                        impact_metrics_t* residual_impact);
bool apply_mitigation(impact_analyzer_t* analyzer,
                     const char* mitigation_plan);

// Quantum-specific functions
bool analyze_quantum_impact(impact_analyzer_t* analyzer,
                          impact_metrics_t* metrics);
bool predict_quantum_effects(impact_analyzer_t* analyzer,
                           impact_prediction_t* prediction);
bool validate_quantum_impact(impact_analyzer_t* analyzer,
                           const impact_metrics_t* metrics);

// Utility functions
bool export_analyzer_data(const impact_analyzer_t* analyzer,
                         const char* filename);
bool import_analyzer_data(impact_analyzer_t* analyzer,
                         const char* filename);
void free_cascade_effects(cascade_effects_t* effects);

#endif // IMPACT_ANALYZER_H
