#ifndef EFFECT_ANALYZER_H
#define EFFECT_ANALYZER_H

#include <stdbool.h>
#include <stddef.h>
#include <time.h>

// Effect types
typedef enum {
    EFFECT_QUANTUM,          // Quantum effects
    EFFECT_GEOMETRIC,        // Geometric effects
    EFFECT_PERFORMANCE,      // Performance effects
    EFFECT_RESOURCE,         // Resource effects
    EFFECT_SYSTEM          // System-wide effects
} effect_type_t;

// Analysis modes
typedef enum {
    ANALYZE_IMMEDIATE,       // Immediate effects
    ANALYZE_PROPAGATED,      // Propagated effects
    ANALYZE_CUMULATIVE,      // Cumulative effects
    ANALYZE_PREDICTIVE      // Predicted effects
} analysis_mode_t;

// Effect severity
typedef enum {
    SEVERITY_NEGLIGIBLE,     // Negligible effect
    SEVERITY_MINOR,          // Minor effect
    SEVERITY_MODERATE,       // Moderate effect
    SEVERITY_MAJOR,          // Major effect
    SEVERITY_CRITICAL       // Critical effect
} effect_severity_t;

// Propagation patterns
typedef enum {
    PROP_LINEAR,            // Linear propagation
    PROP_EXPONENTIAL,       // Exponential propagation
    PROP_GEOMETRIC,         // Geometric propagation
    PROP_QUANTUM           // Quantum propagation
} propagation_pattern_t;

// Analyzer configuration
typedef struct {
    analysis_mode_t mode;           // Analysis mode
    bool track_propagation;         // Track effect propagation
    bool enable_prediction;         // Enable effect prediction
    bool monitor_interactions;      // Monitor effect interactions
    size_t history_size;           // Effect history size
    double threshold;              // Effect threshold
} analyzer_config_t;

// Effect descriptor
typedef struct {
    effect_type_t type;            // Effect type
    effect_severity_t severity;     // Effect severity
    propagation_pattern_t pattern;  // Propagation pattern
    struct timespec timestamp;      // Effect timestamp
    void* source;                  // Effect source
    void* target;                  // Effect target
} effect_desc_t;

// Effect metrics
typedef struct {
    double magnitude;              // Effect magnitude
    double duration;               // Effect duration
    double propagation_speed;      // Propagation speed
    size_t affected_components;    // Affected components
    double impact_score;           // Impact score
    double confidence;             // Confidence level
} effect_metrics_t;

// Propagation chain
typedef struct {
    effect_desc_t* effects;        // Chain of effects
    size_t num_effects;            // Number of effects
    double total_impact;           // Total impact
    struct timespec start_time;    // Chain start time
    struct timespec end_time;      // Chain end time
    bool is_active;               // Chain activity status
} propagation_chain_t;

// Effect prediction
typedef struct {
    effect_type_t predicted_type;  // Predicted effect type
    effect_severity_t severity;     // Predicted severity
    double probability;            // Occurrence probability
    struct timespec expected_time; // Expected time
    void* prediction_data;         // Additional data
} effect_prediction_t;

// Opaque analyzer handle
typedef struct effect_analyzer_t effect_analyzer_t;

// Core functions
effect_analyzer_t* create_effect_analyzer(const analyzer_config_t* config);
void destroy_effect_analyzer(effect_analyzer_t* analyzer);

// Analysis functions
bool analyze_effect(effect_analyzer_t* analyzer,
                   const effect_desc_t* effect,
                   effect_metrics_t* metrics);
bool analyze_propagation(effect_analyzer_t* analyzer,
                        const effect_desc_t* effect,
                        propagation_chain_t* chain);
bool analyze_interactions(effect_analyzer_t* analyzer,
                         const effect_desc_t* effects,
                         size_t num_effects);

// Tracking functions
bool track_effect(effect_analyzer_t* analyzer,
                 const effect_desc_t* effect);
bool update_effect_metrics(effect_analyzer_t* analyzer,
                          const effect_metrics_t* metrics);
bool get_effect_history(const effect_analyzer_t* analyzer,
                       effect_desc_t* history,
                       size_t* num_entries);

// Propagation functions
bool predict_propagation(effect_analyzer_t* analyzer,
                        const effect_desc_t* effect,
                        propagation_chain_t* chain);
bool validate_propagation(effect_analyzer_t* analyzer,
                         const propagation_chain_t* chain);
bool stop_propagation(effect_analyzer_t* analyzer,
                     const effect_desc_t* effect);

// Prediction functions
bool predict_effects(effect_analyzer_t* analyzer,
                    const effect_desc_t* current,
                    effect_prediction_t* prediction);
bool validate_prediction(effect_analyzer_t* analyzer,
                        const effect_prediction_t* prediction,
                        const effect_metrics_t* actual);
bool update_prediction_model(effect_analyzer_t* analyzer,
                           const effect_metrics_t* metrics);

// Quantum-specific functions
bool analyze_quantum_effects(effect_analyzer_t* analyzer,
                           const effect_desc_t* effect,
                           effect_metrics_t* metrics);
bool predict_quantum_effects(effect_analyzer_t* analyzer,
                           const effect_desc_t* current,
                           effect_prediction_t* prediction);
bool validate_quantum_effects(effect_analyzer_t* analyzer,
                            const effect_metrics_t* metrics);

// Utility functions
bool export_analyzer_data(const effect_analyzer_t* analyzer,
                         const char* filename);
bool import_analyzer_data(effect_analyzer_t* analyzer,
                         const char* filename);
void free_propagation_chain(propagation_chain_t* chain);

#endif // EFFECT_ANALYZER_H
