#ifndef PATTERN_ANALYZER_H
#define PATTERN_ANALYZER_H

#include <stdbool.h>
#include <stddef.h>
#include <time.h>
#include "quantum_geometric/core/error_codes.h"

// Pattern types
typedef enum {
    PATTERN_TEMPORAL,        // Temporal patterns
    PATTERN_SPATIAL,         // Spatial patterns
    PATTERN_QUANTUM,         // Quantum patterns
    PATTERN_GEOMETRIC,       // Geometric patterns
    PATTERN_STATISTICAL     // Statistical patterns
} pattern_type_t;

// Analysis modes
typedef enum {
    MODE_REALTIME,          // Real-time analysis
    MODE_BATCH,             // Batch analysis
    MODE_INCREMENTAL,       // Incremental analysis
    MODE_PREDICTIVE,        // Predictive analysis
    MODE_ADAPTIVE          // Adaptive analysis
} analysis_mode_t;

// Pattern features
typedef enum {
    FEATURE_FREQUENCY,      // Pattern frequency
    FEATURE_DURATION,       // Pattern duration
    FEATURE_CORRELATION,    // Pattern correlation
    FEATURE_ENTROPY,        // Pattern entropy
    FEATURE_COMPLEXITY     // Pattern complexity
} pattern_feature_t;

// Detection methods
typedef enum {
    METHOD_STATISTICAL,     // Statistical methods
    METHOD_MACHINE_LEARNING, // Machine learning methods
    METHOD_QUANTUM,         // Quantum methods
    METHOD_HYBRID,          // Hybrid methods
    METHOD_HEURISTIC       // Heuristic methods
} detection_method_t;

// Analyzer configuration
typedef struct {
    analysis_mode_t mode;           // Analysis mode
    detection_method_t method;      // Detection method
    bool enable_learning;           // Enable learning
    bool track_history;             // Track pattern history
    size_t window_size;            // Analysis window size
    double threshold;              // Detection threshold
    void* config_data;            // Additional config data
} analyzer_config_t;

// Pattern descriptor
typedef struct {
    pattern_type_t type;           // Pattern type
    size_t length;                 // Pattern length
    double* values;                // Pattern values
    double frequency;              // Pattern frequency
    double confidence;             // Detection confidence
    struct timespec timestamp;     // Detection time
    void* pattern_data;           // Additional pattern data
} pattern_descriptor_t;

// Pattern metrics
typedef struct {
    double significance;           // Pattern significance
    double reliability;            // Pattern reliability
    double stability;              // Pattern stability
    double complexity;             // Pattern complexity
    size_t occurrences;           // Number of occurrences
    void* metric_data;           // Additional metrics
} pattern_metrics_t;

// Analysis results
typedef struct {
    size_t num_patterns;           // Number of patterns
    pattern_descriptor_t* patterns; // Detected patterns
    pattern_metrics_t* metrics;    // Pattern metrics
    struct timespec timestamp;     // Analysis time
    void* result_data;           // Additional results
} analysis_results_t;

// Opaque analyzer handle
typedef struct pattern_analyzer_t pattern_analyzer_t;

// Core functions
pattern_analyzer_t* create_pattern_analyzer(const analyzer_config_t* config);
void destroy_pattern_analyzer(pattern_analyzer_t* analyzer);

// Analysis functions
qgt_error_t analyze_patterns(pattern_analyzer_t* analyzer,
                           const void* data,
                           size_t size,
                           analysis_results_t* results);
qgt_error_t detect_pattern(pattern_analyzer_t* analyzer,
                         const void* data,
                         size_t size,
                         pattern_type_t type,
                         pattern_descriptor_t* pattern);
qgt_error_t classify_pattern(pattern_analyzer_t* analyzer,
                           const pattern_descriptor_t* pattern,
                           pattern_type_t* type);

// Feature extraction
qgt_error_t extract_features(pattern_analyzer_t* analyzer,
                           const pattern_descriptor_t* pattern,
                           pattern_feature_t* features,
                           size_t* num_features);
qgt_error_t compute_metrics(pattern_analyzer_t* analyzer,
                          const pattern_descriptor_t* pattern,
                          pattern_metrics_t* metrics);
qgt_error_t evaluate_significance(pattern_analyzer_t* analyzer,
                                const pattern_descriptor_t* pattern,
                                double* significance);

// Pattern matching
qgt_error_t match_pattern(pattern_analyzer_t* analyzer,
                        const pattern_descriptor_t* pattern,
                        const void* data,
                        size_t size,
                        size_t* matches);
qgt_error_t find_similar_patterns(pattern_analyzer_t* analyzer,
                                const pattern_descriptor_t* pattern,
                                pattern_descriptor_t* similar,
                                size_t* num_similar);
qgt_error_t validate_pattern(pattern_analyzer_t* analyzer,
                           const pattern_descriptor_t* pattern);

// Quantum-specific functions
qgt_error_t analyze_quantum_patterns(pattern_analyzer_t* analyzer,
                                   const void* quantum_state,
                                   size_t num_qubits,
                                   analysis_results_t* results);
qgt_error_t detect_quantum_correlations(pattern_analyzer_t* analyzer,
                                      const void* quantum_state,
                                      pattern_descriptor_t* correlations);
qgt_error_t analyze_entanglement_patterns(pattern_analyzer_t* analyzer,
                                        const void* quantum_state,
                                        pattern_metrics_t* metrics);

// History management
qgt_error_t store_pattern(pattern_analyzer_t* analyzer,
                        const pattern_descriptor_t* pattern);
qgt_error_t retrieve_pattern_history(pattern_analyzer_t* analyzer,
                                   pattern_type_t type,
                                   pattern_descriptor_t* history,
                                   size_t* num_patterns);
qgt_error_t clear_pattern_history(pattern_analyzer_t* analyzer);

// Utility functions
void free_pattern_descriptor(pattern_descriptor_t* pattern);
void free_analysis_results(analysis_results_t* results);
const char* get_pattern_description(const pattern_descriptor_t* pattern);

#endif // PATTERN_ANALYZER_H
