#ifndef FEATURE_ANALYZER_H
#define FEATURE_ANALYZER_H

#include <stdbool.h>
#include <stddef.h>
#include <time.h>

// Feature types
typedef enum {
    FEATURE_QUANTUM,         // Quantum features
    FEATURE_GEOMETRIC,       // Geometric features
    FEATURE_TOPOLOGICAL,     // Topological features
    FEATURE_STATISTICAL,     // Statistical features
    FEATURE_TEMPORAL       // Temporal features
} feature_type_t;

// Analysis modes
typedef enum {
    ANALYZE_STATIC,         // Static analysis
    ANALYZE_DYNAMIC,        // Dynamic analysis
    ANALYZE_ADAPTIVE,       // Adaptive analysis
    ANALYZE_QUANTUM        // Quantum analysis
} analysis_mode_t;

// Feature importance
typedef enum {
    IMPORTANCE_CRITICAL,    // Critical features
    IMPORTANCE_HIGH,        // High importance
    IMPORTANCE_MEDIUM,      // Medium importance
    IMPORTANCE_LOW,         // Low importance
    IMPORTANCE_NEGLIGIBLE  // Negligible importance
} feature_importance_t;

// Extraction methods
typedef enum {
    METHOD_DIRECT,          // Direct extraction
    METHOD_TRANSFORM,       // Transform-based
    METHOD_LEARNING,        // Learning-based
    METHOD_HYBRID         // Hybrid methods
} extraction_method_t;

// Analyzer configuration
typedef struct {
    analysis_mode_t mode;           // Analysis mode
    extraction_method_t method;     // Extraction method
    bool enable_learning;           // Enable learning
    bool track_history;             // Track history
    size_t max_features;           // Maximum features
    double threshold;              // Feature threshold
} analyzer_config_t;

// Feature descriptor
typedef struct {
    feature_type_t type;           // Feature type
    feature_importance_t importance; // Feature importance
    double value;                   // Feature value
    double confidence;              // Feature confidence
    char* name;                    // Feature name
    void* feature_data;           // Additional data
} feature_desc_t;

// Feature metrics
typedef struct {
    double relevance;              // Feature relevance
    double distinctiveness;         // Feature distinctiveness
    double stability;               // Feature stability
    double correlation;             // Feature correlation
    size_t occurrence_count;        // Occurrence count
    double variance;               // Feature variance
} feature_metrics_t;

// Feature set
typedef struct {
    feature_desc_t* features;      // Feature array
    size_t num_features;           // Number of features
    double total_importance;        // Total importance
    struct timespec timestamp;      // Extraction time
    bool is_normalized;            // Normalization flag
    void* set_data;               // Additional data
} feature_set_t;

// Opaque analyzer handle
typedef struct feature_analyzer_t feature_analyzer_t;

// Core functions
feature_analyzer_t* create_feature_analyzer(const analyzer_config_t* config);
void destroy_feature_analyzer(feature_analyzer_t* analyzer);

// Analysis functions
bool analyze_features(feature_analyzer_t* analyzer,
                     const void* data,
                     feature_set_t* features);
bool analyze_feature_importance(feature_analyzer_t* analyzer,
                              const feature_desc_t* feature,
                              feature_importance_t* importance);
bool analyze_feature_relationships(feature_analyzer_t* analyzer,
                                 const feature_set_t* features,
                                 double* correlation_matrix);

// Extraction functions
bool extract_features(feature_analyzer_t* analyzer,
                     const void* data,
                     feature_set_t* features);
bool extract_specific_feature(feature_analyzer_t* analyzer,
                            const void* data,
                            feature_type_t type,
                            feature_desc_t* feature);
bool validate_extraction(feature_analyzer_t* analyzer,
                        const feature_set_t* features);

// Feature selection
bool select_features(feature_analyzer_t* analyzer,
                    const feature_set_t* input,
                    feature_set_t* selected,
                    size_t max_features);
bool rank_features(feature_analyzer_t* analyzer,
                  const feature_set_t* features,
                  size_t* rankings);
bool filter_features(feature_analyzer_t* analyzer,
                    const feature_set_t* input,
                    feature_set_t* filtered,
                    double threshold);

// Metrics computation
bool compute_feature_metrics(feature_analyzer_t* analyzer,
                           const feature_desc_t* feature,
                           feature_metrics_t* metrics);
bool evaluate_feature_quality(feature_analyzer_t* analyzer,
                            const feature_desc_t* feature,
                            double* quality_score);
bool measure_feature_stability(feature_analyzer_t* analyzer,
                             const feature_desc_t* feature,
                             double* stability_score);

// Learning functions
bool train_feature_extractor(feature_analyzer_t* analyzer,
                           const void* training_data,
                           size_t num_samples);
bool update_feature_model(feature_analyzer_t* analyzer,
                         const feature_set_t* features);
bool validate_feature_model(feature_analyzer_t* analyzer,
                          const void* validation_data,
                          size_t num_samples);

// Quantum-specific functions
bool analyze_quantum_features(feature_analyzer_t* analyzer,
                            const void* quantum_state,
                            feature_set_t* features);
bool extract_quantum_features(feature_analyzer_t* analyzer,
                            const void* quantum_state,
                            feature_set_t* features);
bool validate_quantum_features(feature_analyzer_t* analyzer,
                             const feature_set_t* features);

// Utility functions
bool export_analyzer_data(const feature_analyzer_t* analyzer,
                         const char* filename);
bool import_analyzer_data(feature_analyzer_t* analyzer,
                         const char* filename);
void free_feature_set(feature_set_t* feature_set);

#endif // FEATURE_ANALYZER_H
