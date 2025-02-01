#ifndef BEHAVIOR_ANALYZER_H
#define BEHAVIOR_ANALYZER_H

#include <stdbool.h>
#include <stddef.h>
#include <time.h>

// Behavior types
typedef enum {
    BEHAVIOR_NORMAL,           // Normal system behavior
    BEHAVIOR_ANOMALOUS,       // Anomalous behavior
    BEHAVIOR_TRANSITIONAL,    // Transitional behavior
    BEHAVIOR_UNSTABLE        // Unstable behavior
} behavior_type_t;

// Analysis modes
typedef enum {
    ANALYSIS_CONTINUOUS,      // Continuous analysis
    ANALYSIS_SNAPSHOT,        // Snapshot analysis
    ANALYSIS_HISTORICAL,      // Historical analysis
    ANALYSIS_PREDICTIVE      // Predictive analysis
} analysis_mode_t;

// Pattern types
typedef enum {
    PATTERN_CYCLIC,          // Cyclic patterns
    PATTERN_TRENDING,        // Trending patterns
    PATTERN_SEASONAL,        // Seasonal patterns
    PATTERN_RANDOM          // Random patterns
} pattern_type_t;

// Anomaly types
typedef enum {
    ANOMALY_POINT,           // Point anomalies
    ANOMALY_CONTEXTUAL,      // Contextual anomalies
    ANOMALY_COLLECTIVE,      // Collective anomalies
    ANOMALY_SYSTEMIC        // Systemic anomalies
} anomaly_type_t;

// Analyzer configuration
typedef struct {
    analysis_mode_t mode;            // Analysis mode
    size_t window_size;             // Analysis window size
    size_t sample_rate;             // Sampling rate
    double confidence_threshold;     // Confidence threshold
    bool track_patterns;            // Enable pattern tracking
    bool track_anomalies;           // Enable anomaly tracking
    bool enable_prediction;         // Enable prediction
} analyzer_config_t;

// Behavior metrics
typedef struct {
    behavior_type_t type;           // Behavior type
    double stability_score;         // Stability score
    double predictability_score;    // Predictability score
    double anomaly_score;          // Anomaly score
    size_t pattern_count;          // Pattern count
    size_t anomaly_count;          // Anomaly count
} behavior_metrics_t;

// Pattern description
typedef struct {
    pattern_type_t type;           // Pattern type
    size_t frequency;              // Pattern frequency
    double confidence;             // Pattern confidence
    double significance;           // Pattern significance
    struct timespec start_time;    // Pattern start time
    struct timespec end_time;      // Pattern end time
} pattern_desc_t;

// Anomaly description
typedef struct {
    anomaly_type_t type;          // Anomaly type
    double severity;              // Anomaly severity
    double confidence;            // Detection confidence
    void* context;               // Anomaly context
    struct timespec timestamp;    // Detection timestamp
    char* description;           // Anomaly description
} anomaly_desc_t;

// Behavior prediction
typedef struct {
    behavior_type_t predicted_type;    // Predicted behavior
    double confidence;                 // Prediction confidence
    struct timespec prediction_time;   // Prediction timestamp
    void* prediction_data;             // Additional data
} behavior_prediction_t;

// Opaque analyzer handle
typedef struct behavior_analyzer_t behavior_analyzer_t;

// Core functions
behavior_analyzer_t* create_behavior_analyzer(const analyzer_config_t* config);
void destroy_behavior_analyzer(behavior_analyzer_t* analyzer);

// Analysis functions
bool analyze_current_behavior(behavior_analyzer_t* analyzer,
                            behavior_metrics_t* metrics);
bool analyze_behavior_pattern(behavior_analyzer_t* analyzer,
                            pattern_desc_t* pattern);
bool analyze_behavior_anomalies(behavior_analyzer_t* analyzer,
                              anomaly_desc_t* anomalies,
                              size_t* num_anomalies);

// Pattern functions
bool detect_patterns(behavior_analyzer_t* analyzer,
                    pattern_desc_t* patterns,
                    size_t* num_patterns);
bool validate_pattern(behavior_analyzer_t* analyzer,
                     const pattern_desc_t* pattern);
bool track_pattern_evolution(behavior_analyzer_t* analyzer,
                           const pattern_desc_t* pattern,
                           pattern_desc_t* evolution);

// Anomaly functions
bool detect_anomalies(behavior_analyzer_t* analyzer,
                     anomaly_desc_t* anomalies,
                     size_t* num_anomalies);
bool classify_anomaly(behavior_analyzer_t* analyzer,
                     const anomaly_desc_t* anomaly,
                     anomaly_type_t* type);
bool validate_anomaly(behavior_analyzer_t* analyzer,
                     const anomaly_desc_t* anomaly);

// Prediction functions
bool predict_behavior(behavior_analyzer_t* analyzer,
                     behavior_prediction_t* prediction);
bool validate_prediction(behavior_analyzer_t* analyzer,
                        const behavior_prediction_t* prediction,
                        const behavior_metrics_t* actual);
bool update_prediction_model(behavior_analyzer_t* analyzer,
                           const behavior_metrics_t* metrics);

// Monitoring functions
bool update_behavior_metrics(behavior_analyzer_t* analyzer,
                           const behavior_metrics_t* metrics);
bool get_behavior_metrics(const behavior_analyzer_t* analyzer,
                         behavior_metrics_t* metrics);
bool get_behavior_history(const behavior_analyzer_t* analyzer,
                         behavior_metrics_t* history,
                         size_t* num_entries);

// Utility functions
bool reset_analyzer_state(behavior_analyzer_t* analyzer);
bool export_behavior_data(const behavior_analyzer_t* analyzer,
                         const char* filename);
bool import_behavior_data(behavior_analyzer_t* analyzer,
                         const char* filename);

#endif // BEHAVIOR_ANALYZER_H
