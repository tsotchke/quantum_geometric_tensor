/**
 * @file quantum_geometric_tensor_perf.h
 * @brief Performance Tracking for Quantum Geometric Tensor Operations
 *
 * Provides comprehensive performance monitoring including:
 * - Operation timing and profiling
 * - FLOPS measurement
 * - Memory bandwidth tracking
 * - GPU kernel profiling
 * - Performance regression detection
 * - Optimization recommendations
 *
 * Part of the QGTL Hardware Acceleration Framework.
 */

#ifndef QUANTUM_GEOMETRIC_TENSOR_PERF_H
#define QUANTUM_GEOMETRIC_TENSOR_PERF_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================

#define PERF_MAX_NAME_LENGTH 256
#define PERF_MAX_OPERATIONS 4096
#define PERF_MAX_MARKERS 256
#define PERF_HISTORY_SIZE 10000

// ============================================================================
// Enumerations
// ============================================================================

/**
 * Performance metric types
 */
typedef enum {
    PERF_METRIC_TIME,                 // Execution time
    PERF_METRIC_FLOPS,                // Floating point operations
    PERF_METRIC_MEMORY_READ,          // Memory bytes read
    PERF_METRIC_MEMORY_WRITE,         // Memory bytes written
    PERF_METRIC_MEMORY_TRANSFER,      // Host-device transfer
    PERF_METRIC_GPU_OCCUPANCY,        // GPU occupancy
    PERF_METRIC_CACHE_HIT,            // Cache hit rate
    PERF_METRIC_INSTRUCTIONS,         // Instructions executed
    PERF_METRIC_BRANCHES,             // Branch count
    PERF_METRIC_CUSTOM                // Custom metric
} perf_metric_type_t;

/**
 * Profiling levels
 */
typedef enum {
    PERF_LEVEL_OFF,                   // No profiling
    PERF_LEVEL_MINIMAL,               // Basic timing only
    PERF_LEVEL_STANDARD,              // Standard metrics
    PERF_LEVEL_DETAILED,              // Detailed profiling
    PERF_LEVEL_FULL                   // Full profiling with GPU counters
} perf_level_t;

/**
 * Performance trend
 */
typedef enum {
    PERF_TREND_IMPROVING,             // Getting faster
    PERF_TREND_STABLE,                // No significant change
    PERF_TREND_DEGRADING,             // Getting slower
    PERF_TREND_VOLATILE,              // High variance
    PERF_TREND_UNKNOWN                // Not enough data
} perf_trend_t;

/**
 * Operation categories
 */
typedef enum {
    PERF_CAT_TENSOR_CREATE,           // Tensor creation
    PERF_CAT_TENSOR_DESTROY,          // Tensor destruction
    PERF_CAT_TENSOR_COPY,             // Tensor copy/transfer
    PERF_CAT_TENSOR_ARITHMETIC,       // Basic arithmetic
    PERF_CAT_MATRIX_MULTIPLY,         // Matrix multiplication
    PERF_CAT_TENSOR_CONTRACT,         // Tensor contraction
    PERF_CAT_DECOMPOSITION,           // SVD, QR, etc.
    PERF_CAT_QUANTUM_GATE,            // Quantum gate application
    PERF_CAT_MEASUREMENT,             // Quantum measurement
    PERF_CAT_FFT,                     // Fourier transform
    PERF_CAT_COMMUNICATION,           // MPI/distributed comm
    PERF_CAT_SYNCHRONIZATION,         // Barriers, syncs
    PERF_CAT_OTHER                    // Other operations
} perf_category_t;

// ============================================================================
// Data Structures
// ============================================================================

/**
 * Performance sample
 */
typedef struct {
    char operation_name[PERF_MAX_NAME_LENGTH];
    perf_category_t category;
    uint64_t timestamp_ns;
    uint64_t duration_ns;
    uint64_t flops;
    size_t memory_bytes;
    double gpu_occupancy;
    int device_id;
    int stream_id;
    void* extra_data;
} perf_sample_t;

/**
 * Operation statistics
 */
typedef struct {
    char operation_name[PERF_MAX_NAME_LENGTH];
    perf_category_t category;
    uint64_t call_count;
    uint64_t total_time_ns;
    uint64_t min_time_ns;
    uint64_t max_time_ns;
    double mean_time_ns;
    double std_dev_ns;
    double median_ns;
    double p95_ns;
    double p99_ns;
    uint64_t total_flops;
    double gflops_per_sec;
    size_t total_memory;
    double memory_bandwidth_gbps;
    double avg_gpu_occupancy;
    perf_trend_t trend;
} perf_operation_stats_t;

/**
 * Category statistics
 */
typedef struct {
    perf_category_t category;
    uint64_t total_operations;
    uint64_t total_time_ns;
    double percentage_of_total;
    double avg_time_ns;
    double total_gflops;
} perf_category_stats_t;

/**
 * Roofline model data
 */
typedef struct {
    double peak_flops;                // Peak GFLOPS
    double peak_memory_bandwidth;     // Peak GB/s
    double ridge_point;               // FLOPS/byte at ridge
    double achieved_flops;
    double achieved_bandwidth;
    double operational_intensity;     // FLOPS/byte
    double efficiency;                // % of peak
    bool is_compute_bound;
} perf_roofline_t;

/**
 * Performance baseline
 */
typedef struct {
    char operation_name[PERF_MAX_NAME_LENGTH];
    double baseline_time_ns;
    double baseline_flops;
    uint64_t baseline_timestamp_ns;
    char description[PERF_MAX_NAME_LENGTH];
} perf_baseline_t;

/**
 * Regression detection result
 */
typedef struct {
    char operation_name[PERF_MAX_NAME_LENGTH];
    double baseline_value;
    double current_value;
    double regression_percent;
    double confidence;
    bool is_significant;
    char recommendation[512];
} perf_regression_t;

/**
 * Performance marker
 */
typedef struct {
    char name[PERF_MAX_NAME_LENGTH];
    uint64_t timestamp_ns;
    int color_id;                     // For visualization
    char annotation[PERF_MAX_NAME_LENGTH];
} perf_marker_t;

/**
 * Time range for analysis
 */
typedef struct {
    uint64_t start_ns;
    uint64_t end_ns;
} perf_time_range_t;

/**
 * Overall performance summary
 */
typedef struct {
    uint64_t total_operations;
    uint64_t total_time_ns;
    double total_gflops;
    double avg_gflops_per_sec;
    double peak_gflops_per_sec;
    size_t total_memory_transferred;
    double avg_memory_bandwidth_gbps;
    double avg_gpu_utilization;
    perf_category_stats_t category_stats[13]; // One per category
    uint64_t analysis_start_ns;
    uint64_t analysis_end_ns;
} perf_summary_t;

/**
 * Optimization suggestion
 */
typedef struct {
    char operation_name[PERF_MAX_NAME_LENGTH];
    char suggestion[512];
    double estimated_improvement;
    uint32_t priority;                // 1 = highest
    char category[64];
} perf_suggestion_t;

/**
 * Profiler configuration
 */
typedef struct {
    perf_level_t level;
    bool enable_gpu_counters;
    bool enable_memory_tracking;
    bool enable_regression_detection;
    double regression_threshold;      // % change to flag
    size_t history_size;
    bool enable_roofline;
    bool enable_markers;
    uint64_t warmup_samples;
} perf_config_t;

/**
 * Opaque profiler handle
 */
typedef struct perf_profiler perf_profiler_t;

// ============================================================================
// Initialization
// ============================================================================

/**
 * Create performance profiler
 */
perf_profiler_t* perf_profiler_create(void);

/**
 * Create with configuration
 */
perf_profiler_t* perf_profiler_create_with_config(
    const perf_config_t* config);

/**
 * Get default configuration
 */
perf_config_t perf_default_config(void);

/**
 * Destroy profiler
 */
void perf_profiler_destroy(perf_profiler_t* profiler);

/**
 * Enable/disable profiling
 */
void perf_profiler_enable(perf_profiler_t* profiler, bool enable);

/**
 * Check if profiling is enabled
 */
bool perf_profiler_is_enabled(perf_profiler_t* profiler);

/**
 * Set profiling level
 */
void perf_profiler_set_level(perf_profiler_t* profiler, perf_level_t level);

// ============================================================================
// Timing Operations
// ============================================================================

/**
 * Start timing an operation
 * Returns: handle for stop_timing
 */
uint64_t perf_start_timing(
    perf_profiler_t* profiler,
    const char* operation_name,
    perf_category_t category);

/**
 * Stop timing an operation
 */
void perf_stop_timing(
    perf_profiler_t* profiler,
    uint64_t handle);

/**
 * Stop timing with additional metrics
 */
void perf_stop_timing_with_metrics(
    perf_profiler_t* profiler,
    uint64_t handle,
    uint64_t flops,
    size_t memory_bytes);

/**
 * Record complete sample
 */
void perf_record_sample(
    perf_profiler_t* profiler,
    const perf_sample_t* sample);

// Convenience macro for scoped timing
#define PERF_TIMED_BLOCK(profiler, name, category) \
    for (uint64_t _ph = perf_start_timing(profiler, name, category), \
         *_po = &_ph; _po; perf_stop_timing(profiler, _ph), _po = NULL)

// ============================================================================
// Markers and Annotations
// ============================================================================

/**
 * Add marker at current time
 */
bool perf_add_marker(
    perf_profiler_t* profiler,
    const char* name,
    const char* annotation);

/**
 * Add marker with color
 */
bool perf_add_marker_colored(
    perf_profiler_t* profiler,
    const char* name,
    const char* annotation,
    int color_id);

/**
 * Get markers in range
 */
bool perf_get_markers(
    perf_profiler_t* profiler,
    const perf_time_range_t* range,
    perf_marker_t** markers,
    size_t* count);

// ============================================================================
// Statistics
// ============================================================================

/**
 * Get operation statistics
 */
bool perf_get_operation_stats(
    perf_profiler_t* profiler,
    const char* operation_name,
    perf_operation_stats_t* stats);

/**
 * Get all operation statistics
 */
bool perf_get_all_stats(
    perf_profiler_t* profiler,
    perf_operation_stats_t** stats,
    size_t* count);

/**
 * Get category statistics
 */
bool perf_get_category_stats(
    perf_profiler_t* profiler,
    perf_category_t category,
    perf_category_stats_t* stats);

/**
 * Get performance summary
 */
bool perf_get_summary(
    perf_profiler_t* profiler,
    perf_summary_t* summary);

/**
 * Get summary for time range
 */
bool perf_get_summary_range(
    perf_profiler_t* profiler,
    const perf_time_range_t* range,
    perf_summary_t* summary);

/**
 * Get hottest operations
 */
bool perf_get_hottest_operations(
    perf_profiler_t* profiler,
    size_t n,
    perf_operation_stats_t** stats,
    size_t* count);

/**
 * Get sample history
 */
bool perf_get_history(
    perf_profiler_t* profiler,
    const char* operation_name,
    perf_sample_t** samples,
    size_t* count);

// ============================================================================
// Roofline Analysis
// ============================================================================

/**
 * Perform roofline analysis
 */
bool perf_roofline_analyze(
    perf_profiler_t* profiler,
    int device_id,
    perf_roofline_t* roofline);

/**
 * Get roofline for specific operation
 */
bool perf_roofline_operation(
    perf_profiler_t* profiler,
    const char* operation_name,
    perf_roofline_t* roofline);

/**
 * Set device peak performance
 */
void perf_roofline_set_peak(
    perf_profiler_t* profiler,
    int device_id,
    double peak_gflops,
    double peak_bandwidth_gbps);

// ============================================================================
// Baseline and Regression
// ============================================================================

/**
 * Set baseline for operation
 */
bool perf_set_baseline(
    perf_profiler_t* profiler,
    const char* operation_name,
    const char* description);

/**
 * Set baseline from current stats
 */
bool perf_set_baseline_current(
    perf_profiler_t* profiler,
    const char* operation_name);

/**
 * Load baselines from file
 */
bool perf_load_baselines(
    perf_profiler_t* profiler,
    const char* filename);

/**
 * Save baselines to file
 */
bool perf_save_baselines(
    perf_profiler_t* profiler,
    const char* filename);

/**
 * Detect regressions
 */
bool perf_detect_regressions(
    perf_profiler_t* profiler,
    perf_regression_t** regressions,
    size_t* count);

/**
 * Compare against baseline
 */
bool perf_compare_baseline(
    perf_profiler_t* profiler,
    const char* operation_name,
    perf_regression_t* result);

// ============================================================================
// Optimization Suggestions
// ============================================================================

/**
 * Generate optimization suggestions
 */
bool perf_get_suggestions(
    perf_profiler_t* profiler,
    perf_suggestion_t** suggestions,
    size_t* count);

/**
 * Get suggestions for operation
 */
bool perf_get_operation_suggestions(
    perf_profiler_t* profiler,
    const char* operation_name,
    perf_suggestion_t** suggestions,
    size_t* count);

// ============================================================================
// Reset and Clear
// ============================================================================

/**
 * Reset all statistics
 */
void perf_reset_stats(perf_profiler_t* profiler);

/**
 * Reset statistics for operation
 */
void perf_reset_operation_stats(
    perf_profiler_t* profiler,
    const char* operation_name);

/**
 * Clear history
 */
void perf_clear_history(perf_profiler_t* profiler);

/**
 * Clear markers
 */
void perf_clear_markers(perf_profiler_t* profiler);

// ============================================================================
// Reporting
// ============================================================================

/**
 * Generate performance report
 */
char* perf_generate_report(perf_profiler_t* profiler);

/**
 * Generate detailed operation report
 */
char* perf_generate_operation_report(
    perf_profiler_t* profiler,
    const char* operation_name);

/**
 * Export to JSON
 */
char* perf_export_json(perf_profiler_t* profiler);

/**
 * Export to Chrome trace format
 */
bool perf_export_chrome_trace(
    perf_profiler_t* profiler,
    const char* filename);

/**
 * Export to file
 */
bool perf_export_to_file(
    perf_profiler_t* profiler,
    const char* filename);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Get metric type name
 */
const char* perf_metric_name(perf_metric_type_t type);

/**
 * Get profiling level name
 */
const char* perf_level_name(perf_level_t level);

/**
 * Get trend name
 */
const char* perf_trend_name(perf_trend_t trend);

/**
 * Get category name
 */
const char* perf_category_name(perf_category_t category);

/**
 * Get current timestamp
 */
uint64_t perf_get_timestamp_ns(void);

/**
 * Format duration
 */
char* perf_format_duration(uint64_t duration_ns);

/**
 * Format throughput
 */
char* perf_format_throughput(double gflops);

/**
 * Format bandwidth
 */
char* perf_format_bandwidth(double gbps);

/**
 * Free operation stats array
 */
void perf_free_operation_stats(perf_operation_stats_t* stats, size_t count);

/**
 * Free samples array
 */
void perf_free_samples(perf_sample_t* samples, size_t count);

/**
 * Free markers array
 */
void perf_free_markers(perf_marker_t* markers, size_t count);

/**
 * Free regressions array
 */
void perf_free_regressions(perf_regression_t* regressions, size_t count);

/**
 * Free suggestions array
 */
void perf_free_suggestions(perf_suggestion_t* suggestions, size_t count);

/**
 * Get last error message
 */
const char* perf_get_last_error(perf_profiler_t* profiler);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_GEOMETRIC_TENSOR_PERF_H
