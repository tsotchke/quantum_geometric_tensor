#ifndef FAILURE_PREDICTOR_H
#define FAILURE_PREDICTOR_H

#include <stdbool.h>
#include <stddef.h>
#include <time.h>

// Failure types
typedef enum {
    FAILURE_QUANTUM,         // Quantum failure
    FAILURE_HARDWARE,        // Hardware failure
    FAILURE_SOFTWARE,        // Software failure
    FAILURE_NETWORK,         // Network failure
    FAILURE_RESOURCE        // Resource failure
} failure_type_t;

// Prediction modes
typedef enum {
    PREDICT_REALTIME,       // Real-time prediction
    PREDICT_PROACTIVE,      // Proactive prediction
    PREDICT_HISTORICAL,     // Historical prediction
    PREDICT_HYBRID         // Hybrid prediction
} prediction_mode_t;

// Risk levels
typedef enum {
    RISK_NEGLIGIBLE,        // Negligible risk
    RISK_LOW,               // Low risk
    RISK_MODERATE,          // Moderate risk
    RISK_HIGH,              // High risk
    RISK_CRITICAL          // Critical risk
} risk_level_t;

// Failure sources
typedef enum {
    SOURCE_DECOHERENCE,     // Quantum decoherence
    SOURCE_ERROR_RATE,      // Error rates
    SOURCE_RESOURCE,        // Resource exhaustion
    SOURCE_OVERLOAD,        // System overload
    SOURCE_EXTERNAL        // External factors
} failure_source_t;

// Predictor configuration
typedef struct {
    prediction_mode_t mode;         // Prediction mode
    bool track_history;             // Track failure history
    bool enable_learning;           // Enable learning
    bool monitor_sources;           // Monitor failure sources
    size_t window_size;            // Analysis window size
    double threshold;              // Risk threshold
} predictor_config_t;

// Failure metrics
typedef struct {
    failure_type_t type;           // Failure type
    risk_level_t risk;             // Risk level
    failure_source_t source;       // Failure source
    double probability;            // Failure probability
    double impact;                 // Failure impact
    size_t occurrence_count;       // Occurrence count
} failure_metrics_t;

// Risk assessment
typedef struct {
    risk_level_t level;            // Risk level
    double probability;            // Risk probability
    double severity;               // Risk severity
    double mitigation_factor;      // Mitigation factor
    size_t affected_components;    // Affected components
    bool requires_action;         // Action required flag
} risk_assessment_t;

// Failure prediction
typedef struct {
    failure_type_t type;           // Predicted failure type
    struct timespec expected_time; // Expected failure time
    double confidence;             // Prediction confidence
    double time_to_failure;        // Time until failure
    char* description;            // Failure description
    void* prediction_data;        // Additional data
} failure_prediction_t;

// Opaque predictor handle
typedef struct failure_predictor_t failure_predictor_t;

// Core functions
failure_predictor_t* create_failure_predictor(const predictor_config_t* config);
void destroy_failure_predictor(failure_predictor_t* predictor);

// Prediction functions
bool predict_failures(failure_predictor_t* predictor,
                     failure_prediction_t* predictions,
                     size_t* num_predictions);
bool assess_risk(failure_predictor_t* predictor,
                failure_type_t type,
                risk_assessment_t* assessment);
bool validate_prediction(failure_predictor_t* predictor,
                        const failure_prediction_t* prediction);

// Analysis functions
bool analyze_failure_patterns(failure_predictor_t* predictor,
                            failure_metrics_t* metrics);
bool analyze_risk_factors(failure_predictor_t* predictor,
                         failure_type_t type,
                         risk_assessment_t* assessment);
bool analyze_failure_sources(failure_predictor_t* predictor,
                           failure_source_t* sources,
                           size_t* num_sources);

// Monitoring functions
bool monitor_failure_indicators(failure_predictor_t* predictor,
                              failure_metrics_t* metrics);
bool track_failure_events(failure_predictor_t* predictor,
                         const failure_metrics_t* metrics);
bool get_failure_history(const failure_predictor_t* predictor,
                        failure_metrics_t* history,
                        size_t* num_entries);

// Risk management
bool evaluate_risk_level(failure_predictor_t* predictor,
                        const failure_metrics_t* metrics,
                        risk_level_t* level);
bool suggest_mitigation(failure_predictor_t* predictor,
                       const risk_assessment_t* assessment,
                       char** mitigation_plan);
bool validate_mitigation(failure_predictor_t* predictor,
                        const char* mitigation_plan);

// Learning functions
bool update_prediction_model(failure_predictor_t* predictor,
                           const failure_metrics_t* metrics);
bool train_predictor(failure_predictor_t* predictor,
                    const failure_metrics_t* training_data,
                    size_t num_samples);
bool validate_model(failure_predictor_t* predictor,
                   const failure_metrics_t* validation_data,
                   size_t num_samples);

// Quantum-specific functions
bool predict_quantum_failures(failure_predictor_t* predictor,
                            failure_prediction_t* predictions,
                            size_t* num_predictions);
bool assess_quantum_risk(failure_predictor_t* predictor,
                        risk_assessment_t* assessment);
bool analyze_decoherence_patterns(failure_predictor_t* predictor,
                                failure_metrics_t* metrics);

// Utility functions
bool export_predictor_data(const failure_predictor_t* predictor,
                          const char* filename);
bool import_predictor_data(failure_predictor_t* predictor,
                          const char* filename);
void free_prediction(failure_prediction_t* prediction);

#endif // FAILURE_PREDICTOR_H
