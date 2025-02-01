#ifndef PATTERN_METRICS_H
#define PATTERN_METRICS_H

#include <stdbool.h>
#include <stddef.h>
#include <time.h>
#include "quantum_geometric/core/error_codes.h"

// Metric types
typedef enum {
    METRIC_FREQUENCY,       // Pattern frequency
    METRIC_DURATION,        // Pattern duration
    METRIC_COMPLEXITY,      // Pattern complexity
    METRIC_ENTROPY,         // Pattern entropy
    METRIC_CORRELATION     // Pattern correlation
} metric_type_t;

// Aggregation modes
typedef enum {
    AGGREGATE_SUM,         // Sum of metrics
    AGGREGATE_AVERAGE,     // Average of metrics
    AGGREGATE_WEIGHTED,    // Weighted average
    AGGREGATE_MAXIMUM,     // Maximum value
    AGGREGATE_MINIMUM     // Minimum value
} aggregation_mode_t;

// Normalization types
typedef enum {
    NORMALIZE_NONE,        // No normalization
    NORMALIZE_MINMAX,      // Min-max normalization
    NORMALIZE_ZSCORE,      // Z-score normalization
    NORMALIZE_ROBUST,      // Robust scaling
    NORMALIZE_QUANTUM     // Quantum normalization
} normalization_type_t;

// Metric configuration
typedef struct {
    metric_type_t type;            // Metric type
    aggregation_mode_t mode;       // Aggregation mode
    normalization_type_t norm;     // Normalization type
    bool enable_weighting;         // Enable weighting
    double threshold;              // Metric threshold
    void* config_data;           // Additional config
} metric_config_t;

// Basic metrics
typedef struct {
    double value;                  // Metric value
    double weight;                 // Metric weight
    double confidence;             // Confidence score
    double significance;           // Significance score
    struct timespec timestamp;     // Measurement time
    void* metric_data;           // Additional data
} basic_metrics_t;

// Pattern metrics
typedef struct {
    size_t pattern_id;             // Pattern identifier
    metric_type_t type;            // Metric type
    double* values;                // Metric values
    size_t num_values;             // Number of values
    double aggregated_value;       // Aggregated value
    void* pattern_data;           // Additional data
} pattern_metrics_t;

// Statistical metrics
typedef struct {
    double mean;                   // Mean value
    double variance;               // Variance
    double skewness;               // Skewness
    double kurtosis;               // Kurtosis
    double* moments;               // Higher moments
    size_t num_moments;           // Number of moments
} statistical_metrics_t;

// Quantum metrics
typedef struct {
    double fidelity;              // Quantum fidelity
    double coherence;             // Quantum coherence
    double entanglement;          // Entanglement measure
    double purity;                // State purity
    void* quantum_data;          // Additional data
} quantum_metrics_t;

// Opaque metrics handle
typedef struct pattern_metrics_calculator_t pattern_metrics_calculator_t;

// Core functions
pattern_metrics_calculator_t* create_metrics_calculator(const metric_config_t* config);
void destroy_metrics_calculator(pattern_metrics_calculator_t* calculator);

// Computation functions
qgt_error_t compute_metrics(pattern_metrics_calculator_t* calculator,
                          const void* data,
                          size_t size,
                          pattern_metrics_t* metrics);
qgt_error_t compute_basic_metrics(pattern_metrics_calculator_t* calculator,
                                const void* data,
                                size_t size,
                                basic_metrics_t* metrics);
qgt_error_t compute_statistical_metrics(pattern_metrics_calculator_t* calculator,
                                      const void* data,
                                      size_t size,
                                      statistical_metrics_t* metrics);

// Aggregation functions
qgt_error_t aggregate_metrics(pattern_metrics_calculator_t* calculator,
                            const pattern_metrics_t* metrics,
                            size_t num_metrics,
                            double* result);
qgt_error_t compute_weighted_metrics(pattern_metrics_calculator_t* calculator,
                                   const pattern_metrics_t* metrics,
                                   const double* weights,
                                   size_t num_metrics,
                                   double* result);
qgt_error_t normalize_metrics(pattern_metrics_calculator_t* calculator,
                            pattern_metrics_t* metrics,
                            normalization_type_t type);

// Analysis functions
qgt_error_t analyze_metrics_distribution(pattern_metrics_calculator_t* calculator,
                                       const pattern_metrics_t* metrics,
                                       statistical_metrics_t* stats);
qgt_error_t compute_metrics_correlation(pattern_metrics_calculator_t* calculator,
                                      const pattern_metrics_t* metrics1,
                                      const pattern_metrics_t* metrics2,
                                      double* correlation);
qgt_error_t evaluate_metrics_significance(pattern_metrics_calculator_t* calculator,
                                        const pattern_metrics_t* metrics,
                                        double* significance);

// Quantum-specific functions
qgt_error_t compute_quantum_metrics(pattern_metrics_calculator_t* calculator,
                                  const void* quantum_state,
                                  quantum_metrics_t* metrics);
qgt_error_t analyze_quantum_correlations(pattern_metrics_calculator_t* calculator,
                                       const quantum_metrics_t* metrics,
                                       double* correlations);
qgt_error_t validate_quantum_metrics(pattern_metrics_calculator_t* calculator,
                                   const quantum_metrics_t* metrics);

// Utility functions
void free_pattern_metrics(pattern_metrics_t* metrics);
void free_statistical_metrics(statistical_metrics_t* metrics);
void free_quantum_metrics(quantum_metrics_t* metrics);

#endif // PATTERN_METRICS_H
