#ifndef BOTTLENECK_DETECTOR_H
#define BOTTLENECK_DETECTOR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================

#define MAX_METRICS 16
#define MAX_SUGGESTIONS 32

// ============================================================================
// Type Definitions
// ============================================================================

/**
 * @brief Bottleneck types
 */
typedef enum {
    COMPUTE_BOUND,
    MEMORY_BOUND,
    IO_BOUND,
    COMMUNICATION_BOUND,
    QUANTUM_BOUND,
    NO_BOTTLENECK
} BottleneckType;

/**
 * @brief System metrics for analysis
 */
typedef struct SystemMetrics {
    double cpu_usage;           // CPU utilization (0.0-1.0)
    double gpu_usage;           // GPU utilization (0.0-1.0)
    double memory_usage;        // Memory utilization (0.0-1.0)
    double quantum_usage;       // Quantum resource utilization
    double network_bandwidth;   // Network bandwidth utilization
    double disk_io;             // Disk I/O utilization
    double latency;             // System latency
    double throughput;          // System throughput
    size_t timestamp;           // Measurement timestamp
} SystemMetrics;

/**
 * @brief Optimization suggestion
 */
typedef struct OptimizationSuggestion {
    char* description;          // Description of suggestion
    double expected_improvement; // Expected improvement (0.0-1.0)
    double confidence;          // Confidence in suggestion
    BottleneckType target;      // Bottleneck this addresses
    bool requires_restart;      // Whether restart is needed
    int priority;               // Priority (1-10)
} OptimizationSuggestion;

/**
 * @brief Performance pattern for analysis
 */
typedef struct PerformancePattern {
    double* values;
    size_t size;
    size_t capacity;
    double mean;
    double std_dev;
    bool is_anomaly;
} PerformancePattern;

/**
 * @brief Simple ML model for bottleneck prediction
 */
typedef struct MLModel {
    double* weights;
    size_t num_features;
    double bias;
    double learning_rate;
    size_t training_samples;
} MLModel;

/**
 * @brief ML prediction result
 */
typedef struct MLPrediction {
    BottleneckType bottleneck_type;
    double confidence;
    double* feature_contributions;
} MLPrediction;

/**
 * @brief Bottleneck detector structure
 */
typedef struct BottleneckDetector {
    // Pattern analysis
    PerformancePattern** patterns;
    size_t num_patterns;

    // ML model
    MLModel* model;
    double* feature_importance;

    // Detection state
    BottleneckType current_bottleneck;
    double confidence;

    // History
    SystemMetrics* metrics_history;
    size_t history_index;
    size_t history_size;

    // Optimization suggestions
    OptimizationSuggestion* suggestions;
    size_t num_suggestions;
} BottleneckDetector;

// ============================================================================
// Public API
// ============================================================================

/**
 * @brief Initialize bottleneck detector
 */
BottleneckDetector* init_bottleneck_detector(void);

/**
 * @brief Analyze system metrics
 */
void analyze_system_metrics(
    BottleneckDetector* detector,
    const SystemMetrics* metrics);

/**
 * @brief Get optimization suggestions
 */
const OptimizationSuggestion* get_suggestions(
    BottleneckDetector* detector,
    size_t* num_suggestions);

/**
 * @brief Clean up bottleneck detector
 */
void cleanup_bottleneck_detector(BottleneckDetector* detector);

// ============================================================================
// Internal Helper Functions
// ============================================================================

// ML model functions
MLModel* init_ml_model(void);
MLPrediction run_ml_model(MLModel* model, const double* features);
void update_feature_importance(MLModel* model, double* importance);
void cleanup_ml_model(MLModel* model);

// Pattern functions
void update_pattern(PerformancePattern* pattern, double value);
void update_pattern_statistics(PerformancePattern* pattern);
bool detect_pattern_anomaly(const PerformancePattern* pattern);
void cleanup_pattern(PerformancePattern* pattern);

// History and analysis
void store_metrics_history(BottleneckDetector* detector, const SystemMetrics* metrics);
void update_performance_patterns(BottleneckDetector* detector, const SystemMetrics* metrics);
void detect_anomalies(BottleneckDetector* detector);
void identify_bottleneck(BottleneckDetector* detector);
double* extract_features(BottleneckDetector* detector);

// Suggestion generators
void generate_suggestions(BottleneckDetector* detector);
void generate_compute_suggestions(BottleneckDetector* detector);
void generate_memory_suggestions(BottleneckDetector* detector);
void generate_io_suggestions(BottleneckDetector* detector);
void generate_communication_suggestions(BottleneckDetector* detector);
void generate_quantum_suggestions(BottleneckDetector* detector);

#ifdef __cplusplus
}
#endif

#endif // BOTTLENECK_DETECTOR_H
