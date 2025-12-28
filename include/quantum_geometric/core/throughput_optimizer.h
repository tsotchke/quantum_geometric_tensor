/**
 * @file throughput_optimizer.h
 * @brief Throughput Optimization for Quantum Geometric Operations
 *
 * Provides throughput analysis and optimization including:
 * - Throughput measurement and tracking
 * - Bottleneck identification
 * - Queue management optimization
 * - Batch size optimization
 * - Pipeline efficiency analysis
 * - Throughput prediction
 *
 * Part of the QGTL Monitoring Framework.
 */

#ifndef THROUGHPUT_OPTIMIZER_H
#define THROUGHPUT_OPTIMIZER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================

#define THROUGHPUT_MAX_NAME_LENGTH 128
#define THROUGHPUT_MAX_PIPELINES 64
#define THROUGHPUT_MAX_STAGES 32
#define THROUGHPUT_HISTORY_SIZE 10000

// ============================================================================
// Enumerations
// ============================================================================

/**
 * Throughput optimization strategies
 */
typedef enum {
    THROUGHPUT_STRATEGY_BATCHING,     // Increase batch size
    THROUGHPUT_STRATEGY_PIPELINING,   // Add pipeline stages
    THROUGHPUT_STRATEGY_PARALLELISM,  // Increase parallelism
    THROUGHPUT_STRATEGY_CACHING,      // Add caching layers
    THROUGHPUT_STRATEGY_PREFETCHING,  // Prefetch data
    THROUGHPUT_STRATEGY_COMPRESSION,  // Compress data transfers
    THROUGHPUT_STRATEGY_LOAD_BALANCE, // Balance workload
    THROUGHPUT_STRATEGY_CUSTOM        // Custom strategy
} throughput_strategy_t;

/**
 * Pipeline stage types
 */
typedef enum {
    THROUGHPUT_STAGE_INPUT,           // Input/ingestion
    THROUGHPUT_STAGE_PREPROCESSING,   // Data preprocessing
    THROUGHPUT_STAGE_COMPUTE,         // Main computation
    THROUGHPUT_STAGE_POSTPROCESSING,  // Result processing
    THROUGHPUT_STAGE_OUTPUT,          // Output/storage
    THROUGHPUT_STAGE_CUSTOM           // Custom stage
} throughput_stage_type_t;

/**
 * Throughput health status
 */
typedef enum {
    THROUGHPUT_HEALTH_OPTIMAL,        // Meeting targets
    THROUGHPUT_HEALTH_GOOD,           // Near targets
    THROUGHPUT_HEALTH_DEGRADED,       // Below targets
    THROUGHPUT_HEALTH_CRITICAL,       // Significantly below
    THROUGHPUT_HEALTH_UNKNOWN         // Cannot determine
} throughput_health_t;

/**
 * Bottleneck severity
 */
typedef enum {
    BOTTLENECK_SEVERITY_NONE,
    BOTTLENECK_SEVERITY_MINOR,
    BOTTLENECK_SEVERITY_MODERATE,
    BOTTLENECK_SEVERITY_SEVERE,
    BOTTLENECK_SEVERITY_CRITICAL
} bottleneck_severity_t;

// ============================================================================
// Data Structures
// ============================================================================

/**
 * Throughput measurement
 */
typedef struct {
    uint64_t timestamp_ns;
    double items_per_second;
    double bytes_per_second;
    uint64_t items_processed;
    uint64_t bytes_processed;
    uint64_t duration_ns;
} throughput_measurement_t;

/**
 * Pipeline stage metrics
 */
typedef struct {
    char name[THROUGHPUT_MAX_NAME_LENGTH];
    throughput_stage_type_t type;
    double throughput_items_per_sec;
    double throughput_bytes_per_sec;
    uint64_t avg_latency_ns;
    uint64_t queue_depth;
    uint64_t items_processed;
    double utilization;
    bool is_bottleneck;
} throughput_stage_metrics_t;

/**
 * Pipeline metrics
 */
typedef struct {
    char name[THROUGHPUT_MAX_NAME_LENGTH];
    double overall_throughput;
    double theoretical_max_throughput;
    double efficiency;                // actual/theoretical
    size_t num_stages;
    throughput_stage_metrics_t stages[THROUGHPUT_MAX_STAGES];
    size_t bottleneck_stage_index;
    uint64_t total_latency_ns;
    throughput_health_t health;
} throughput_pipeline_metrics_t;

/**
 * Batch optimization result
 */
typedef struct {
    size_t current_batch_size;
    size_t optimal_batch_size;
    double current_throughput;
    double predicted_throughput;
    double improvement_percent;
    char rationale[256];
} throughput_batch_optimization_t;

/**
 * Bottleneck analysis
 */
typedef struct {
    char component[THROUGHPUT_MAX_NAME_LENGTH];
    bottleneck_severity_t severity;
    double impact_percent;            // Impact on throughput
    char cause[256];
    char recommendation[256];
    throughput_strategy_t suggested_strategy;
} throughput_bottleneck_t;

/**
 * Throughput prediction
 */
typedef struct {
    double predicted_throughput;
    double confidence_low;            // Lower bound
    double confidence_high;           // Upper bound
    double confidence_level;          // Confidence percentage
    uint64_t prediction_horizon_ns;
    char model_used[64];
} throughput_prediction_t;

/**
 * Queue statistics
 */
typedef struct {
    char queue_name[THROUGHPUT_MAX_NAME_LENGTH];
    uint64_t current_depth;
    uint64_t max_depth;
    double avg_depth;
    double avg_wait_time_ns;
    uint64_t enqueue_rate;
    uint64_t dequeue_rate;
    uint64_t drops;
    double utilization;
} throughput_queue_stats_t;

/**
 * Optimization suggestion
 */
typedef struct {
    throughput_strategy_t strategy;
    char description[256];
    double estimated_improvement;     // Percentage
    double implementation_cost;       // Relative cost 0-1
    double confidence;                // Confidence in estimate
    char prerequisites[256];
} throughput_suggestion_t;

/**
 * Overall metrics
 */
typedef struct {
    double current_throughput;
    double peak_throughput;
    double avg_throughput;
    double min_throughput;
    double target_throughput;
    double throughput_variance;
    throughput_health_t health;
    uint64_t total_items_processed;
    uint64_t total_bytes_processed;
    uint64_t measurement_duration_ns;
} throughput_overall_metrics_t;

/**
 * Optimizer configuration
 */
typedef struct {
    double target_throughput;         // Target items/sec
    size_t history_size;
    bool enable_prediction;
    bool enable_auto_optimization;
    double bottleneck_threshold;      // Utilization threshold
    size_t min_batch_size;
    size_t max_batch_size;
    uint64_t measurement_interval_ns;
    double smoothing_factor;          // For EMA calculations
} throughput_optimizer_config_t;

/**
 * Opaque optimizer handle
 */
typedef struct throughput_optimizer throughput_optimizer_t;

// ============================================================================
// Initialization and Configuration
// ============================================================================

/**
 * Create throughput optimizer with default configuration
 */
throughput_optimizer_t* throughput_optimizer_create(void);

/**
 * Create with custom configuration
 */
throughput_optimizer_t* throughput_optimizer_create_with_config(
    const throughput_optimizer_config_t* config);

/**
 * Get default configuration
 */
throughput_optimizer_config_t throughput_optimizer_default_config(void);

/**
 * Destroy throughput optimizer
 */
void throughput_optimizer_destroy(throughput_optimizer_t* optimizer);

/**
 * Reset all statistics
 */
bool throughput_optimizer_reset(throughput_optimizer_t* optimizer);

/**
 * Set target throughput
 */
bool throughput_set_target(
    throughput_optimizer_t* optimizer,
    double target_items_per_sec);

// ============================================================================
// Pipeline Management
// ============================================================================

/**
 * Register a pipeline
 */
bool throughput_register_pipeline(
    throughput_optimizer_t* optimizer,
    const char* name,
    double theoretical_max);

/**
 * Add stage to pipeline
 */
bool throughput_add_stage(
    throughput_optimizer_t* optimizer,
    const char* pipeline_name,
    const char* stage_name,
    throughput_stage_type_t type);

/**
 * Remove pipeline
 */
bool throughput_remove_pipeline(
    throughput_optimizer_t* optimizer,
    const char* name);

// ============================================================================
// Measurement Recording
// ============================================================================

/**
 * Record throughput measurement
 */
bool throughput_record_measurement(
    throughput_optimizer_t* optimizer,
    const char* pipeline_name,
    uint64_t items_processed,
    uint64_t bytes_processed,
    uint64_t duration_ns);

/**
 * Record stage throughput
 */
bool throughput_record_stage(
    throughput_optimizer_t* optimizer,
    const char* pipeline_name,
    const char* stage_name,
    uint64_t items_processed,
    uint64_t latency_ns);

/**
 * Record queue depth
 */
bool throughput_record_queue_depth(
    throughput_optimizer_t* optimizer,
    const char* queue_name,
    uint64_t depth);

/**
 * Start measurement period
 * Returns: measurement handle
 */
uint64_t throughput_start_measurement(
    throughput_optimizer_t* optimizer,
    const char* pipeline_name);

/**
 * End measurement period
 */
bool throughput_end_measurement(
    throughput_optimizer_t* optimizer,
    uint64_t handle,
    uint64_t items_processed);

// ============================================================================
// Analysis
// ============================================================================

/**
 * Get overall metrics
 */
bool throughput_get_overall_metrics(
    throughput_optimizer_t* optimizer,
    throughput_overall_metrics_t* metrics);

/**
 * Get pipeline metrics
 */
bool throughput_get_pipeline_metrics(
    throughput_optimizer_t* optimizer,
    const char* pipeline_name,
    throughput_pipeline_metrics_t* metrics);

/**
 * Get all pipeline metrics
 */
bool throughput_get_all_pipelines(
    throughput_optimizer_t* optimizer,
    throughput_pipeline_metrics_t** metrics,
    size_t* count);

/**
 * Identify bottlenecks
 */
bool throughput_identify_bottlenecks(
    throughput_optimizer_t* optimizer,
    throughput_bottleneck_t** bottlenecks,
    size_t* count);

/**
 * Get queue statistics
 */
bool throughput_get_queue_stats(
    throughput_optimizer_t* optimizer,
    const char* queue_name,
    throughput_queue_stats_t* stats);

/**
 * Get throughput history
 */
bool throughput_get_history(
    throughput_optimizer_t* optimizer,
    const char* pipeline_name,
    throughput_measurement_t** measurements,
    size_t* count);

// ============================================================================
// Optimization
// ============================================================================

/**
 * Optimize batch size
 */
bool throughput_optimize_batch_size(
    throughput_optimizer_t* optimizer,
    const char* pipeline_name,
    throughput_batch_optimization_t* result);

/**
 * Get optimization suggestions
 */
bool throughput_get_suggestions(
    throughput_optimizer_t* optimizer,
    throughput_suggestion_t** suggestions,
    size_t* count);

/**
 * Apply optimization strategy
 */
bool throughput_apply_strategy(
    throughput_optimizer_t* optimizer,
    const char* pipeline_name,
    throughput_strategy_t strategy);

/**
 * Auto-optimize (if enabled)
 */
bool throughput_auto_optimize(throughput_optimizer_t* optimizer);

// ============================================================================
// Prediction
// ============================================================================

/**
 * Predict throughput
 */
bool throughput_predict(
    throughput_optimizer_t* optimizer,
    const char* pipeline_name,
    uint64_t horizon_ns,
    throughput_prediction_t* prediction);

/**
 * Predict with workload change
 */
bool throughput_predict_with_workload(
    throughput_optimizer_t* optimizer,
    const char* pipeline_name,
    double workload_multiplier,
    throughput_prediction_t* prediction);

// ============================================================================
// Reporting
// ============================================================================

/**
 * Generate throughput report
 */
char* throughput_generate_report(throughput_optimizer_t* optimizer);

/**
 * Export to JSON
 */
char* throughput_export_json(throughput_optimizer_t* optimizer);

/**
 * Export to file
 */
bool throughput_export_to_file(
    throughput_optimizer_t* optimizer,
    const char* filename);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Get strategy name
 */
const char* throughput_strategy_name(throughput_strategy_t strategy);

/**
 * Get stage type name
 */
const char* throughput_stage_type_name(throughput_stage_type_t type);

/**
 * Get health status name
 */
const char* throughput_health_name(throughput_health_t health);

/**
 * Get bottleneck severity name
 */
const char* throughput_bottleneck_severity_name(bottleneck_severity_t severity);

/**
 * Free pipeline metrics array
 */
void throughput_free_pipeline_metrics(throughput_pipeline_metrics_t* metrics, size_t count);

/**
 * Free bottleneck array
 */
void throughput_free_bottlenecks(throughput_bottleneck_t* bottlenecks, size_t count);

/**
 * Free suggestions array
 */
void throughput_free_suggestions(throughput_suggestion_t* suggestions, size_t count);

/**
 * Free measurements array
 */
void throughput_free_measurements(throughput_measurement_t* measurements, size_t count);

/**
 * Get last error message
 */
const char* throughput_get_last_error(throughput_optimizer_t* optimizer);

#ifdef __cplusplus
}
#endif

#endif // THROUGHPUT_OPTIMIZER_H
