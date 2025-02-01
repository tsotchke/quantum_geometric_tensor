#ifndef FLEXIBILITY_ANALYZER_H
#define FLEXIBILITY_ANALYZER_H

#include <stdbool.h>
#include <stddef.h>
#include <time.h>

// Flexibility types
typedef enum {
    FLEX_COMPUTATIONAL,     // Computational flexibility
    FLEX_ARCHITECTURAL,     // Architectural flexibility
    FLEX_OPERATIONAL,       // Operational flexibility
    FLEX_RESOURCE,         // Resource flexibility
    FLEX_QUANTUM          // Quantum flexibility
} flexibility_type_t;

// Analysis modes
typedef enum {
    ANALYZE_STATIC,        // Static analysis
    ANALYZE_DYNAMIC,       // Dynamic analysis
    ANALYZE_ADAPTIVE,      // Adaptive analysis
    ANALYZE_PREDICTIVE    // Predictive analysis
} analysis_mode_t;

// Flexibility levels
typedef enum {
    LEVEL_RIGID,          // Rigid system
    LEVEL_LIMITED,        // Limited flexibility
    LEVEL_MODERATE,       // Moderate flexibility
    LEVEL_HIGH,           // High flexibility
    LEVEL_DYNAMIC        // Dynamic flexibility
} flexibility_level_t;

// Adaptation types
typedef enum {
    ADAPT_STRUCTURAL,      // Structural adaptation
    ADAPT_BEHAVIORAL,      // Behavioral adaptation
    ADAPT_RESOURCE,        // Resource adaptation
    ADAPT_QUANTUM         // Quantum adaptation
} adaptation_type_t;

// Analyzer configuration
typedef struct {
    analysis_mode_t mode;          // Analysis mode
    bool track_changes;            // Track flexibility changes
    bool enable_prediction;        // Enable prediction
    bool monitor_adaptations;      // Monitor adaptations
    size_t history_size;          // History size
    double threshold;             // Flexibility threshold
} analyzer_config_t;

// Flexibility metrics
typedef struct {
    flexibility_type_t type;       // Flexibility type
    flexibility_level_t level;     // Flexibility level
    double score;                  // Flexibility score
    double adaptability;           // Adaptability score
    double stability;              // Stability score
    size_t adaptation_count;      // Number of adaptations
} flexibility_metrics_t;

// Adaptation metrics
typedef struct {
    adaptation_type_t type;        // Adaptation type
    double success_rate;           // Success rate
    double cost;                   // Adaptation cost
    double impact;                 // System impact
    double recovery_time;          // Recovery time
    size_t attempt_count;         // Attempt count
} adaptation_metrics_t;

// System state
typedef struct {
    flexibility_level_t level;     // Current flexibility level
    double resource_availability;  // Resource availability
    double performance_impact;     // Performance impact
    double stability_index;        // Stability index
    bool requires_adaptation;      // Adaptation requirement
    void* state_data;            // Additional data
} system_state_t;

// Opaque analyzer handle
typedef struct flexibility_analyzer_t flexibility_analyzer_t;

// Core functions
flexibility_analyzer_t* create_flexibility_analyzer(const analyzer_config_t* config);
void destroy_flexibility_analyzer(flexibility_analyzer_t* analyzer);

// Analysis functions
bool analyze_flexibility(flexibility_analyzer_t* analyzer,
                        flexibility_type_t type,
                        flexibility_metrics_t* metrics);
bool analyze_adaptability(flexibility_analyzer_t* analyzer,
                         adaptation_type_t type,
                         adaptation_metrics_t* metrics);
bool analyze_system_state(flexibility_analyzer_t* analyzer,
                         system_state_t* state);

// Monitoring functions
bool monitor_flexibility(flexibility_analyzer_t* analyzer,
                        flexibility_type_t type,
                        flexibility_metrics_t* metrics);
bool track_adaptations(flexibility_analyzer_t* analyzer,
                      adaptation_type_t type,
                      adaptation_metrics_t* metrics);
bool get_flexibility_history(const flexibility_analyzer_t* analyzer,
                           flexibility_metrics_t* history,
                           size_t* num_entries);

// Assessment functions
bool assess_flexibility_level(flexibility_analyzer_t* analyzer,
                            flexibility_type_t type,
                            flexibility_level_t* level);
bool evaluate_adaptation_capability(flexibility_analyzer_t* analyzer,
                                  adaptation_type_t type,
                                  double* capability_score);
bool measure_system_flexibility(flexibility_analyzer_t* analyzer,
                              system_state_t* state,
                              double* flexibility_score);

// Prediction functions
bool predict_flexibility_needs(flexibility_analyzer_t* analyzer,
                             flexibility_type_t type,
                             flexibility_metrics_t* prediction);
bool predict_adaptation_impact(flexibility_analyzer_t* analyzer,
                             adaptation_type_t type,
                             adaptation_metrics_t* impact);
bool validate_predictions(flexibility_analyzer_t* analyzer,
                         const flexibility_metrics_t* predicted,
                         const flexibility_metrics_t* actual);

// Optimization functions
bool optimize_flexibility(flexibility_analyzer_t* analyzer,
                        flexibility_type_t type,
                        flexibility_metrics_t* result);
bool optimize_adaptations(flexibility_analyzer_t* analyzer,
                         adaptation_type_t type,
                         adaptation_metrics_t* result);
bool validate_optimization(flexibility_analyzer_t* analyzer,
                         const flexibility_metrics_t* metrics);

// Quantum-specific functions
bool analyze_quantum_flexibility(flexibility_analyzer_t* analyzer,
                               flexibility_metrics_t* metrics);
bool optimize_quantum_adaptations(flexibility_analyzer_t* analyzer,
                                adaptation_metrics_t* metrics);
bool validate_quantum_flexibility(flexibility_analyzer_t* analyzer,
                                const flexibility_metrics_t* metrics);

// Utility functions
bool export_analyzer_data(const flexibility_analyzer_t* analyzer,
                         const char* filename);
bool import_analyzer_data(flexibility_analyzer_t* analyzer,
                         const char* filename);
void free_flexibility_metrics(flexibility_metrics_t* metrics);

#endif // FLEXIBILITY_ANALYZER_H
