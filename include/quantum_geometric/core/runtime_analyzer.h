/**
 * @file runtime_analyzer.h
 * @brief Runtime Performance Analysis for Quantum Geometric Operations
 *
 * Provides comprehensive runtime analysis including:
 * - Operation timing and profiling
 * - Latency distribution analysis
 * - Throughput measurement
 * - Performance regression detection
 * - Bottleneck identification
 * - Execution path analysis
 *
 * Part of the QGTL Monitoring Framework.
 */

#ifndef RUNTIME_ANALYZER_H
#define RUNTIME_ANALYZER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================

#define RUNTIME_MAX_OPERATION_NAME 128
#define RUNTIME_MAX_STACK_DEPTH 32
#define RUNTIME_MAX_HISTOGRAM_BINS 100
#define RUNTIME_MAX_OPERATIONS 10000

// ============================================================================
// Enumerations
// ============================================================================

/**
 * Operation categories for classification
 */
typedef enum {
    RUNTIME_OP_GATE_APPLICATION,      // Single/multi-qubit gate operations
    RUNTIME_OP_STATE_PREPARATION,     // State initialization/preparation
    RUNTIME_OP_MEASUREMENT,           // Measurement operations
    RUNTIME_OP_TENSOR_CONTRACTION,    // Tensor network contractions
    RUNTIME_OP_MATRIX_MULTIPLY,       // Matrix multiplications
    RUNTIME_OP_FFT,                   // Fourier transforms
    RUNTIME_OP_DECOMPOSITION,         // SVD, QR, etc.
    RUNTIME_OP_ERROR_CORRECTION,      // QEC operations
    RUNTIME_OP_MEMORY_ALLOCATION,     // Memory operations
    RUNTIME_OP_IO,                    // I/O operations
    RUNTIME_OP_COMMUNICATION,         // Inter-process communication
    RUNTIME_OP_GPU_KERNEL,            // GPU kernel execution
    RUNTIME_OP_SYNCHRONIZATION,       // Synchronization primitives
    RUNTIME_OP_CUSTOM,                // User-defined operations
    RUNTIME_OP_CATEGORY_COUNT
} runtime_op_category_t;

/**
 * Performance trend indicators
 */
typedef enum {
    RUNTIME_TREND_IMPROVING,          // Performance getting better
    RUNTIME_TREND_STABLE,             // Performance stable
    RUNTIME_TREND_DEGRADING,          // Performance getting worse
    RUNTIME_TREND_VOLATILE,           // High variance
    RUNTIME_TREND_UNKNOWN             // Insufficient data
} runtime_trend_t;

/**
 * Bottleneck types
 */
typedef enum {
    BOTTLENECK_NONE,                  // No bottleneck detected
    BOTTLENECK_CPU_BOUND,             // CPU-limited
    BOTTLENECK_MEMORY_BOUND,          // Memory bandwidth limited
    BOTTLENECK_MEMORY_LATENCY,        // Memory latency limited
    BOTTLENECK_GPU_COMPUTE,           // GPU compute limited
    BOTTLENECK_GPU_MEMORY,            // GPU memory limited
    BOTTLENECK_IO_BOUND,              // I/O limited
    BOTTLENECK_NETWORK_BOUND,         // Network limited
    BOTTLENECK_SYNCHRONIZATION,       // Lock contention
    BOTTLENECK_CACHE_MISS             // Cache efficiency issue
} runtime_bottleneck_t;

// ============================================================================
// Data Structures
// ============================================================================

/**
 * Single timing sample
 */
typedef struct {
    uint64_t start_ns;                // Start timestamp in nanoseconds
    uint64_t end_ns;                  // End timestamp in nanoseconds
    uint64_t duration_ns;             // Duration in nanoseconds
    uint32_t thread_id;               // Thread that executed
    int cpu_core;                     // CPU core used (-1 if unknown)
} runtime_timing_t;

/**
 * Timing statistics for an operation
 */
typedef struct {
    uint64_t count;                   // Number of invocations
    uint64_t total_ns;                // Total time in nanoseconds
    uint64_t min_ns;                  // Minimum duration
    uint64_t max_ns;                  // Maximum duration
    double mean_ns;                   // Mean duration
    double variance_ns;               // Variance in nanoseconds
    double std_dev_ns;                // Standard deviation
    double median_ns;                 // Median (50th percentile)
    double p90_ns;                    // 90th percentile
    double p95_ns;                    // 95th percentile
    double p99_ns;                    // 99th percentile
    double throughput_per_sec;        // Operations per second
} runtime_stats_t;

/**
 * Latency histogram for distribution analysis
 */
typedef struct {
    uint64_t bin_edges_ns[RUNTIME_MAX_HISTOGRAM_BINS + 1];  // Bin boundaries
    uint64_t bin_counts[RUNTIME_MAX_HISTOGRAM_BINS];        // Count per bin
    size_t num_bins;                  // Number of bins
    uint64_t min_observed_ns;         // Minimum observed value
    uint64_t max_observed_ns;         // Maximum observed value
    uint64_t total_samples;           // Total samples
    uint64_t outliers_low;            // Samples below first bin
    uint64_t outliers_high;           // Samples above last bin
} runtime_histogram_t;

/**
 * Operation timing entry
 */
typedef struct {
    char name[RUNTIME_MAX_OPERATION_NAME];
    runtime_op_category_t category;
    runtime_stats_t stats;
    runtime_histogram_t histogram;
    runtime_trend_t trend;
    uint64_t first_seen_ns;           // First occurrence timestamp
    uint64_t last_seen_ns;            // Last occurrence timestamp
    bool is_active;                   // Currently being timed
} runtime_operation_t;

/**
 * Execution span for hierarchical timing
 */
typedef struct {
    char name[RUNTIME_MAX_OPERATION_NAME];
    uint64_t start_ns;
    uint64_t end_ns;
    runtime_op_category_t category;
    uint32_t depth;                   // Nesting depth
    uint32_t parent_id;               // Parent span ID
    uint32_t span_id;                 // This span's ID
} runtime_span_t;

/**
 * Bottleneck analysis result
 */
typedef struct {
    runtime_bottleneck_t type;
    double severity;                  // 0.0 (none) to 1.0 (severe)
    char description[256];
    char suggestion[512];             // Suggested optimization
    const char* operation_name;       // Associated operation
} runtime_bottleneck_info_t;

/**
 * Performance regression detection
 */
typedef struct {
    const char* operation_name;
    double baseline_mean_ns;          // Baseline mean duration
    double current_mean_ns;           // Current mean duration
    double regression_percent;        // Percent change (positive = slower)
    double confidence;                // Statistical confidence (0-1)
    bool is_significant;              // Statistically significant
} runtime_regression_t;

/**
 * Analyzer configuration
 */
typedef struct {
    bool enable_histogram;            // Enable latency histograms
    bool enable_hierarchical;         // Enable span tracking
    bool enable_regression_detection; // Enable regression detection
    bool enable_bottleneck_analysis;  // Enable bottleneck detection
    size_t max_operations;            // Max operations to track
    size_t histogram_bins;            // Number of histogram bins
    double regression_threshold;      // Regression detection threshold (%)
    double outlier_threshold;         // Outlier threshold (std devs)
    uint64_t warmup_samples;          // Samples to skip for warmup
} runtime_analyzer_config_t;

/**
 * Opaque analyzer handle
 */
typedef struct runtime_analyzer runtime_analyzer_t;

// ============================================================================
// Initialization and Configuration
// ============================================================================

/**
 * Create runtime analyzer with default configuration
 */
runtime_analyzer_t* runtime_analyzer_create(void);

/**
 * Create runtime analyzer with custom configuration
 */
runtime_analyzer_t* runtime_analyzer_create_with_config(
    const runtime_analyzer_config_t* config);

/**
 * Get default configuration
 */
runtime_analyzer_config_t runtime_analyzer_default_config(void);

/**
 * Destroy runtime analyzer
 */
void runtime_analyzer_destroy(runtime_analyzer_t* analyzer);

/**
 * Reset all statistics
 */
bool runtime_analyzer_reset(runtime_analyzer_t* analyzer);

/**
 * Enable/disable collection
 */
void runtime_analyzer_enable(runtime_analyzer_t* analyzer, bool enable);

/**
 * Check if collection is enabled
 */
bool runtime_analyzer_is_enabled(runtime_analyzer_t* analyzer);

// ============================================================================
// Basic Timing Operations
// ============================================================================

/**
 * Start timing an operation
 * Returns: timing handle (non-zero on success)
 */
uint64_t runtime_start_timing(runtime_analyzer_t* analyzer,
                               const char* operation_name,
                               runtime_op_category_t category);

/**
 * Stop timing an operation
 */
void runtime_stop_timing(runtime_analyzer_t* analyzer, uint64_t handle);

/**
 * Record a pre-measured duration
 */
void runtime_record_duration(runtime_analyzer_t* analyzer,
                             const char* operation_name,
                             runtime_op_category_t category,
                             uint64_t duration_ns);

/**
 * Get current timestamp in nanoseconds
 */
uint64_t runtime_get_timestamp_ns(void);

// ============================================================================
// Scoped Timing (RAII-style)
// ============================================================================

/**
 * Scoped timer that stops on scope exit
 */
typedef struct {
    runtime_analyzer_t* analyzer;
    uint64_t handle;
} runtime_scoped_timer_t;

/**
 * Create scoped timer
 */
runtime_scoped_timer_t runtime_scoped_begin(runtime_analyzer_t* analyzer,
                                             const char* operation_name,
                                             runtime_op_category_t category);

/**
 * End scoped timer
 */
void runtime_scoped_end(runtime_scoped_timer_t* timer);

// Convenience macro for C
#define RUNTIME_TIMED_BLOCK(analyzer, name, category) \
    for (runtime_scoped_timer_t _timer = runtime_scoped_begin(analyzer, name, category), \
         *_once = &_timer; _once; runtime_scoped_end(&_timer), _once = NULL)

// ============================================================================
// Hierarchical/Span Timing
// ============================================================================

/**
 * Begin a timing span (for hierarchical profiling)
 */
uint32_t runtime_begin_span(runtime_analyzer_t* analyzer,
                            const char* name,
                            runtime_op_category_t category);

/**
 * End a timing span
 */
void runtime_end_span(runtime_analyzer_t* analyzer, uint32_t span_id);

/**
 * Get spans for visualization
 */
bool runtime_get_spans(runtime_analyzer_t* analyzer,
                       runtime_span_t** spans,
                       size_t* count);

/**
 * Export spans to Chrome trace format
 */
bool runtime_export_trace_json(runtime_analyzer_t* analyzer,
                               const char* filename);

// ============================================================================
// Statistics and Analysis
// ============================================================================

/**
 * Get statistics for a specific operation
 */
bool runtime_get_operation_stats(runtime_analyzer_t* analyzer,
                                 const char* operation_name,
                                 runtime_stats_t* stats);

/**
 * Get statistics for all operations
 */
bool runtime_get_all_stats(runtime_analyzer_t* analyzer,
                           runtime_operation_t** operations,
                           size_t* count);

/**
 * Get statistics for a category
 */
bool runtime_get_category_stats(runtime_analyzer_t* analyzer,
                                runtime_op_category_t category,
                                runtime_stats_t* stats);

/**
 * Get histogram for an operation
 */
bool runtime_get_histogram(runtime_analyzer_t* analyzer,
                           const char* operation_name,
                           runtime_histogram_t* histogram);

/**
 * Get top N slowest operations
 */
bool runtime_get_slowest_operations(runtime_analyzer_t* analyzer,
                                    size_t n,
                                    runtime_operation_t** operations,
                                    size_t* count);

/**
 * Get most frequent operations
 */
bool runtime_get_hottest_operations(runtime_analyzer_t* analyzer,
                                    size_t n,
                                    runtime_operation_t** operations,
                                    size_t* count);

// ============================================================================
// Trend and Regression Analysis
// ============================================================================

/**
 * Get performance trend for an operation
 */
runtime_trend_t runtime_get_trend(runtime_analyzer_t* analyzer,
                                  const char* operation_name);

/**
 * Detect performance regressions
 */
bool runtime_detect_regressions(runtime_analyzer_t* analyzer,
                                runtime_regression_t** regressions,
                                size_t* count);

/**
 * Set baseline for regression detection
 */
bool runtime_set_baseline(runtime_analyzer_t* analyzer,
                          const char* operation_name);

/**
 * Set all current stats as baseline
 */
bool runtime_set_all_baselines(runtime_analyzer_t* analyzer);

/**
 * Clear baseline
 */
void runtime_clear_baseline(runtime_analyzer_t* analyzer,
                            const char* operation_name);

// ============================================================================
// Bottleneck Analysis
// ============================================================================

/**
 * Analyze system for bottlenecks
 */
bool runtime_analyze_bottlenecks(runtime_analyzer_t* analyzer,
                                 runtime_bottleneck_info_t** bottlenecks,
                                 size_t* count);

/**
 * Get bottleneck type name
 */
const char* runtime_bottleneck_name(runtime_bottleneck_t type);

/**
 * Get suggested optimizations
 */
bool runtime_get_optimization_suggestions(runtime_analyzer_t* analyzer,
                                          char*** suggestions,
                                          size_t* count);

// ============================================================================
// Export and Reporting
// ============================================================================

/**
 * Export all data to JSON
 */
char* runtime_export_json(runtime_analyzer_t* analyzer);

/**
 * Export to file
 */
bool runtime_export_to_file(runtime_analyzer_t* analyzer,
                            const char* filename);

/**
 * Generate human-readable report
 */
char* runtime_generate_report(runtime_analyzer_t* analyzer);

/**
 * Generate flamegraph data (folded stacks format)
 */
bool runtime_export_flamegraph(runtime_analyzer_t* analyzer,
                               const char* filename);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Get operation category name
 */
const char* runtime_category_name(runtime_op_category_t category);

/**
 * Get trend name
 */
const char* runtime_trend_name(runtime_trend_t trend);

/**
 * Format duration to human-readable string
 */
char* runtime_format_duration(uint64_t duration_ns);

/**
 * Free allocated operation array
 */
void runtime_free_operations(runtime_operation_t* operations, size_t count);

/**
 * Free allocated regression array
 */
void runtime_free_regressions(runtime_regression_t* regressions, size_t count);

/**
 * Free allocated bottleneck array
 */
void runtime_free_bottlenecks(runtime_bottleneck_info_t* bottlenecks, size_t count);

/**
 * Free allocated string array
 */
void runtime_free_strings(char** strings, size_t count);

/**
 * Get last error message
 */
const char* runtime_get_last_error(runtime_analyzer_t* analyzer);

#ifdef __cplusplus
}
#endif

#endif // RUNTIME_ANALYZER_H
