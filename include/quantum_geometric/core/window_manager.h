/**
 * @file window_manager.h
 * @brief Sliding Window Operations for Time-Series Analysis
 *
 * Provides sliding window data management including:
 * - Fixed and variable window sizes
 * - Time-based and count-based windows
 * - Window aggregation functions
 * - Streaming data support
 * - Multi-dimensional windows
 * - Window-based statistics
 *
 * Part of the QGTL Monitoring Framework.
 */

#ifndef WINDOW_MANAGER_H
#define WINDOW_MANAGER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================

#define WINDOW_MAX_NAME_LENGTH 128
#define WINDOW_MAX_WINDOWS 256
#define WINDOW_MAX_DIMENSIONS 16
#define WINDOW_DEFAULT_SIZE 1000

// ============================================================================
// Enumerations
// ============================================================================

/**
 * Window types
 */
typedef enum {
    WINDOW_TYPE_TUMBLING,             // Non-overlapping fixed windows
    WINDOW_TYPE_SLIDING,              // Overlapping sliding windows
    WINDOW_TYPE_HOPPING,              // Fixed hop interval
    WINDOW_TYPE_SESSION,              // Session-based windows
    WINDOW_TYPE_GLOBAL,               // Single global window
    WINDOW_TYPE_EXPONENTIAL           // Exponentially weighted
} window_type_t;

/**
 * Window boundary types
 */
typedef enum {
    WINDOW_BOUNDARY_COUNT,            // Count-based boundary
    WINDOW_BOUNDARY_TIME,             // Time-based boundary
    WINDOW_BOUNDARY_SIZE,             // Size-based boundary
    WINDOW_BOUNDARY_CUSTOM            // Custom boundary function
} window_boundary_t;

/**
 * Aggregation functions
 */
typedef enum {
    WINDOW_AGG_SUM,                   // Sum of values
    WINDOW_AGG_MEAN,                  // Arithmetic mean
    WINDOW_AGG_MEDIAN,                // Median value
    WINDOW_AGG_MIN,                   // Minimum value
    WINDOW_AGG_MAX,                   // Maximum value
    WINDOW_AGG_COUNT,                 // Count of values
    WINDOW_AGG_VARIANCE,              // Variance
    WINDOW_AGG_STD_DEV,               // Standard deviation
    WINDOW_AGG_FIRST,                 // First value in window
    WINDOW_AGG_LAST,                  // Last value in window
    WINDOW_AGG_RANGE,                 // Max - min
    WINDOW_AGG_PERCENTILE,            // Nth percentile
    WINDOW_AGG_EMA,                   // Exponential moving average
    WINDOW_AGG_CUSTOM                 // Custom aggregation
} window_aggregation_t;

/**
 * Window state
 */
typedef enum {
    WINDOW_STATE_EMPTY,               // No data in window
    WINDOW_STATE_FILLING,             // Window accumulating data
    WINDOW_STATE_FULL,                // Window at capacity
    WINDOW_STATE_READY,               // Ready for processing
    WINDOW_STATE_CLOSED               // Window closed
} window_state_t;

// ============================================================================
// Data Structures
// ============================================================================

/**
 * Single data point
 */
typedef struct {
    double value;
    uint64_t timestamp_ns;
    uint32_t dimension;               // For multi-dimensional data
    void* metadata;                   // Optional metadata
} window_data_point_t;

/**
 * Window statistics
 */
typedef struct {
    size_t count;
    double sum;
    double mean;
    double variance;
    double std_dev;
    double min;
    double max;
    double median;
    double p25;
    double p75;
    double p95;
    double p99;
    double ema;                       // Exponential moving average
    uint64_t first_timestamp_ns;
    uint64_t last_timestamp_ns;
} window_stats_t;

/**
 * Window configuration
 */
typedef struct {
    window_type_t type;
    window_boundary_t boundary;
    size_t size;                      // Count or size limit
    uint64_t duration_ns;             // Time duration for time-based
    uint64_t hop_interval_ns;         // For hopping windows
    uint64_t session_gap_ns;          // For session windows
    double ema_alpha;                 // EMA smoothing factor (0-1)
    bool keep_all_data;               // Keep all data points
    bool auto_close;                  // Auto-close when full
} window_config_t;

/**
 * Window snapshot
 */
typedef struct {
    char name[WINDOW_MAX_NAME_LENGTH];
    window_type_t type;
    window_state_t state;
    size_t current_count;
    size_t capacity;
    window_stats_t stats;
    uint64_t window_start_ns;
    uint64_t window_end_ns;
    double* values;                   // Current values (if kept)
    size_t values_count;
} window_snapshot_t;

/**
 * Aggregation result
 */
typedef struct {
    window_aggregation_t type;
    double result;
    size_t samples_used;
    bool is_valid;
    char details[128];
} window_aggregation_result_t;

/**
 * Multi-window analysis result
 */
typedef struct {
    size_t num_windows;
    window_snapshot_t* windows;
    double correlation;               // Inter-window correlation
    double trend;                     // Overall trend
    bool anomaly_detected;
    char anomaly_description[256];
} window_analysis_result_t;

/**
 * Custom boundary function type
 */
typedef bool (*window_boundary_func_t)(
    const window_data_point_t* point,
    const window_stats_t* current_stats,
    void* user_data);

/**
 * Custom aggregation function type
 */
typedef double (*window_aggregation_func_t)(
    const double* values,
    size_t count,
    void* user_data);

/**
 * Manager configuration
 */
typedef struct {
    size_t max_windows;
    size_t default_window_size;
    uint64_t default_duration_ns;
    double default_ema_alpha;
    bool enable_statistics;
    bool enable_anomaly_detection;
    double anomaly_threshold;         // Std devs for anomaly
} window_manager_config_t;

/**
 * Opaque manager handle
 */
typedef struct window_manager window_manager_t;

/**
 * Opaque window handle
 */
typedef struct window_handle window_handle_t;

// ============================================================================
// Initialization and Configuration
// ============================================================================

/**
 * Create window manager with default configuration
 */
window_manager_t* window_manager_create(void);

/**
 * Create with custom configuration
 */
window_manager_t* window_manager_create_with_config(
    const window_manager_config_t* config);

/**
 * Get default configuration
 */
window_manager_config_t window_manager_default_config(void);

/**
 * Destroy window manager
 */
void window_manager_destroy(window_manager_t* manager);

/**
 * Reset all windows
 */
bool window_manager_reset(window_manager_t* manager);

// ============================================================================
// Window Creation and Management
// ============================================================================

/**
 * Create a new window
 */
window_handle_t* window_create(
    window_manager_t* manager,
    const char* name,
    const window_config_t* config);

/**
 * Create tumbling window (convenience)
 */
window_handle_t* window_create_tumbling(
    window_manager_t* manager,
    const char* name,
    size_t size);

/**
 * Create sliding window (convenience)
 */
window_handle_t* window_create_sliding(
    window_manager_t* manager,
    const char* name,
    size_t size);

/**
 * Create time-based window
 */
window_handle_t* window_create_time_based(
    window_manager_t* manager,
    const char* name,
    uint64_t duration_ns);

/**
 * Create exponential moving average window
 */
window_handle_t* window_create_ema(
    window_manager_t* manager,
    const char* name,
    double alpha);

/**
 * Get window by name
 */
window_handle_t* window_get_by_name(
    window_manager_t* manager,
    const char* name);

/**
 * Destroy a window
 */
bool window_destroy(
    window_manager_t* manager,
    window_handle_t* window);

/**
 * Close a window (no more data)
 */
bool window_close(window_handle_t* window);

/**
 * Reset a window
 */
bool window_reset(window_handle_t* window);

// ============================================================================
// Data Operations
// ============================================================================

/**
 * Add value to window
 */
bool window_add_value(
    window_handle_t* window,
    double value);

/**
 * Add timestamped value
 */
bool window_add_timestamped(
    window_handle_t* window,
    double value,
    uint64_t timestamp_ns);

/**
 * Add data point with metadata
 */
bool window_add_point(
    window_handle_t* window,
    const window_data_point_t* point);

/**
 * Add batch of values
 */
bool window_add_batch(
    window_handle_t* window,
    const double* values,
    size_t count);

/**
 * Get current window state
 */
window_state_t window_get_state(const window_handle_t* window);

/**
 * Get current count
 */
size_t window_get_count(const window_handle_t* window);

/**
 * Check if window is full
 */
bool window_is_full(const window_handle_t* window);

/**
 * Check if window is ready
 */
bool window_is_ready(const window_handle_t* window);

// ============================================================================
// Statistics and Aggregation
// ============================================================================

/**
 * Get window statistics
 */
bool window_get_stats(
    const window_handle_t* window,
    window_stats_t* stats);

/**
 * Get specific aggregation
 */
window_aggregation_result_t window_aggregate(
    const window_handle_t* window,
    window_aggregation_t aggregation);

/**
 * Get percentile
 */
double window_get_percentile(
    const window_handle_t* window,
    double percentile);

/**
 * Apply custom aggregation
 */
double window_aggregate_custom(
    const window_handle_t* window,
    window_aggregation_func_t func,
    void* user_data);

/**
 * Get all values in window
 */
bool window_get_values(
    const window_handle_t* window,
    double** values,
    size_t* count);

/**
 * Get window snapshot
 */
bool window_get_snapshot(
    const window_handle_t* window,
    window_snapshot_t* snapshot);

// ============================================================================
// Advanced Operations
// ============================================================================

/**
 * Set custom boundary function
 */
bool window_set_boundary_func(
    window_handle_t* window,
    window_boundary_func_t func,
    void* user_data);

/**
 * Detect anomalies in window
 */
bool window_detect_anomalies(
    const window_handle_t* window,
    double threshold,
    size_t** anomaly_indices,
    size_t* count);

/**
 * Calculate trend
 */
double window_calculate_trend(const window_handle_t* window);

/**
 * Compare two windows
 */
double window_compare(
    const window_handle_t* window1,
    const window_handle_t* window2);

/**
 * Merge windows
 */
window_handle_t* window_merge(
    window_manager_t* manager,
    const char* name,
    const window_handle_t* window1,
    const window_handle_t* window2);

/**
 * Analyze multiple windows
 */
bool window_analyze_multi(
    window_manager_t* manager,
    const window_handle_t** windows,
    size_t count,
    window_analysis_result_t* result);

// ============================================================================
// Reporting
// ============================================================================

/**
 * Generate window report
 */
char* window_generate_report(const window_handle_t* window);

/**
 * Generate manager report (all windows)
 */
char* window_manager_generate_report(window_manager_t* manager);

/**
 * Export to JSON
 */
char* window_export_json(const window_handle_t* window);

/**
 * Export manager to JSON
 */
char* window_manager_export_json(window_manager_t* manager);

/**
 * Export to file
 */
bool window_export_to_file(
    const window_handle_t* window,
    const char* filename);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Get window type name
 */
const char* window_type_name(window_type_t type);

/**
 * Get boundary type name
 */
const char* window_boundary_name(window_boundary_t boundary);

/**
 * Get aggregation name
 */
const char* window_aggregation_name(window_aggregation_t aggregation);

/**
 * Get state name
 */
const char* window_state_name(window_state_t state);

/**
 * Get default window configuration
 */
window_config_t window_default_config(void);

/**
 * Free snapshot resources
 */
void window_free_snapshot(window_snapshot_t* snapshot);

/**
 * Free analysis result resources
 */
void window_free_analysis_result(window_analysis_result_t* result);

/**
 * Free values array
 */
void window_free_values(double* values, size_t count);

/**
 * Get last error message
 */
const char* window_get_last_error(window_manager_t* manager);

#ifdef __cplusplus
}
#endif

#endif // WINDOW_MANAGER_H
