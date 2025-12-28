/**
 * @file usage_analyzer.h
 * @brief Usage Pattern Analysis for Quantum Geometric Operations
 *
 * Provides usage tracking and pattern analysis including:
 * - API call frequency tracking
 * - Usage pattern detection
 * - Feature utilization analysis
 * - User behavior modeling
 * - Deprecation tracking
 * - Usage recommendations
 *
 * Part of the QGTL Monitoring Framework.
 */

#ifndef USAGE_ANALYZER_H
#define USAGE_ANALYZER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================

#define USAGE_MAX_API_NAME_LENGTH 128
#define USAGE_MAX_APIS 1024
#define USAGE_MAX_PATTERNS 256
#define USAGE_MAX_SESSIONS 100
#define USAGE_HISTORY_SIZE 10000

// ============================================================================
// Enumerations
// ============================================================================

/**
 * API categories for usage tracking
 */
typedef enum {
    USAGE_CATEGORY_CORE,              // Core quantum operations
    USAGE_CATEGORY_GATES,             // Gate operations
    USAGE_CATEGORY_MEASUREMENT,       // Measurement operations
    USAGE_CATEGORY_STATE_PREP,        // State preparation
    USAGE_CATEGORY_TENSOR,            // Tensor operations
    USAGE_CATEGORY_ERROR_CORRECTION,  // QEC operations
    USAGE_CATEGORY_OPTIMIZATION,      // Optimization routines
    USAGE_CATEGORY_SIMULATION,        // Simulation functions
    USAGE_CATEGORY_HARDWARE,          // Hardware interface
    USAGE_CATEGORY_UTILITY,           // Utility functions
    USAGE_CATEGORY_DEPRECATED,        // Deprecated APIs
    USAGE_CATEGORY_CUSTOM,            // Custom user APIs
    USAGE_CATEGORY_COUNT
} usage_category_t;

/**
 * Usage pattern types
 */
typedef enum {
    USAGE_PATTERN_SEQUENTIAL,         // APIs called in sequence
    USAGE_PATTERN_PARALLEL,           // APIs called concurrently
    USAGE_PATTERN_ITERATIVE,          // Repeated API calls
    USAGE_PATTERN_CONDITIONAL,        // Branching based on results
    USAGE_PATTERN_PIPELINE,           // Data flowing through APIs
    USAGE_PATTERN_BATCH,              // Batch processing
    USAGE_PATTERN_STREAMING,          // Stream processing
    USAGE_PATTERN_INTERACTIVE,        // User-interactive pattern
    USAGE_PATTERN_CUSTOM              // Custom pattern
} usage_pattern_type_t;

/**
 * Usage frequency classification
 */
typedef enum {
    USAGE_FREQUENCY_NEVER,            // 0 calls
    USAGE_FREQUENCY_RARE,             // < 1 call/hour
    USAGE_FREQUENCY_OCCASIONAL,       // 1-10 calls/hour
    USAGE_FREQUENCY_REGULAR,          // 10-100 calls/hour
    USAGE_FREQUENCY_FREQUENT,         // 100-1000 calls/hour
    USAGE_FREQUENCY_HEAVY             // > 1000 calls/hour
} usage_frequency_t;

/**
 * Deprecation status
 */
typedef enum {
    DEPRECATION_NONE,                 // Not deprecated
    DEPRECATION_PLANNED,              // Deprecation planned
    DEPRECATION_WARNING,              // Deprecated with warning
    DEPRECATION_ERROR,                // Deprecated, returns error
    DEPRECATION_REMOVED               // Removed from API
} deprecation_status_t;

// ============================================================================
// Data Structures
// ============================================================================

/**
 * Single API call record
 */
typedef struct {
    char api_name[USAGE_MAX_API_NAME_LENGTH];
    usage_category_t category;
    uint64_t timestamp_ns;
    uint64_t duration_ns;
    bool success;
    uint32_t thread_id;
    uint32_t session_id;
} usage_call_record_t;

/**
 * API usage statistics
 */
typedef struct {
    char api_name[USAGE_MAX_API_NAME_LENGTH];
    usage_category_t category;
    uint64_t total_calls;
    uint64_t successful_calls;
    uint64_t failed_calls;
    double success_rate;
    uint64_t total_duration_ns;
    double avg_duration_ns;
    double min_duration_ns;
    double max_duration_ns;
    usage_frequency_t frequency;
    deprecation_status_t deprecation;
    uint64_t first_call_ns;
    uint64_t last_call_ns;
} usage_api_stats_t;

/**
 * Category usage statistics
 */
typedef struct {
    usage_category_t category;
    uint64_t total_calls;
    double percentage_of_total;
    size_t unique_apis_used;
    size_t total_apis_available;
    double coverage_percent;
} usage_category_stats_t;

/**
 * Detected usage pattern
 */
typedef struct {
    usage_pattern_type_t type;
    char description[256];
    char* api_sequence;                // NULL-separated API names
    size_t sequence_length;
    uint64_t occurrence_count;
    double confidence;                 // 0.0 to 1.0
    uint64_t avg_duration_ns;
    bool is_optimal;                   // Optimal pattern flag
} usage_pattern_t;

/**
 * Session usage summary
 */
typedef struct {
    uint32_t session_id;
    uint64_t start_time_ns;
    uint64_t end_time_ns;
    uint64_t total_calls;
    size_t unique_apis_used;
    usage_category_stats_t category_breakdown[USAGE_CATEGORY_COUNT];
    usage_pattern_t* detected_patterns;
    size_t pattern_count;
} usage_session_summary_t;

/**
 * Usage recommendation
 */
typedef struct {
    char current_api[USAGE_MAX_API_NAME_LENGTH];
    char recommended_api[USAGE_MAX_API_NAME_LENGTH];
    char reason[256];
    double improvement_estimate;       // Estimated improvement factor
    deprecation_status_t deprecation_reason;
} usage_recommendation_t;

/**
 * Overall usage metrics
 */
typedef struct {
    uint64_t total_api_calls;
    uint64_t total_duration_ns;
    size_t unique_apis_used;
    size_t total_apis_available;
    double api_coverage_percent;
    size_t deprecated_api_calls;
    usage_category_stats_t category_stats[USAGE_CATEGORY_COUNT];
    size_t active_sessions;
    uint64_t analysis_timestamp_ns;
} usage_metrics_t;

/**
 * Analyzer configuration
 */
typedef struct {
    bool track_call_sequences;
    bool detect_patterns;
    bool track_deprecated_usage;
    bool generate_recommendations;
    size_t history_size;
    size_t max_pattern_length;
    double pattern_confidence_threshold;
    bool per_session_tracking;
    bool per_thread_tracking;
} usage_analyzer_config_t;

/**
 * Opaque analyzer handle
 */
typedef struct usage_analyzer usage_analyzer_t;

// ============================================================================
// Initialization and Configuration
// ============================================================================

/**
 * Create usage analyzer with default configuration
 */
usage_analyzer_t* usage_analyzer_create(void);

/**
 * Create with custom configuration
 */
usage_analyzer_t* usage_analyzer_create_with_config(
    const usage_analyzer_config_t* config);

/**
 * Get default configuration
 */
usage_analyzer_config_t usage_analyzer_default_config(void);

/**
 * Destroy usage analyzer
 */
void usage_analyzer_destroy(usage_analyzer_t* analyzer);

/**
 * Reset all statistics
 */
bool usage_analyzer_reset(usage_analyzer_t* analyzer);

// ============================================================================
// API Registration
// ============================================================================

/**
 * Register an API for tracking
 */
bool usage_register_api(
    usage_analyzer_t* analyzer,
    const char* api_name,
    usage_category_t category,
    deprecation_status_t deprecation);

/**
 * Set deprecation status for API
 */
bool usage_set_deprecation(
    usage_analyzer_t* analyzer,
    const char* api_name,
    deprecation_status_t status,
    const char* replacement_api);

/**
 * Unregister an API
 */
bool usage_unregister_api(
    usage_analyzer_t* analyzer,
    const char* api_name);

// ============================================================================
// Usage Recording
// ============================================================================

/**
 * Record API call start
 * Returns: call handle (non-zero on success)
 */
uint64_t usage_record_call_start(
    usage_analyzer_t* analyzer,
    const char* api_name);

/**
 * Record API call end
 */
bool usage_record_call_end(
    usage_analyzer_t* analyzer,
    uint64_t call_handle,
    bool success);

/**
 * Record complete API call
 */
bool usage_record_call(
    usage_analyzer_t* analyzer,
    const char* api_name,
    uint64_t duration_ns,
    bool success);

/**
 * Start new session
 * Returns: session ID
 */
uint32_t usage_start_session(usage_analyzer_t* analyzer);

/**
 * End current session
 */
bool usage_end_session(usage_analyzer_t* analyzer, uint32_t session_id);

// ============================================================================
// Analysis
// ============================================================================

/**
 * Get statistics for specific API
 */
bool usage_get_api_stats(
    usage_analyzer_t* analyzer,
    const char* api_name,
    usage_api_stats_t* stats);

/**
 * Get statistics for all APIs
 */
bool usage_get_all_api_stats(
    usage_analyzer_t* analyzer,
    usage_api_stats_t** stats,
    size_t* count);

/**
 * Get category statistics
 */
bool usage_get_category_stats(
    usage_analyzer_t* analyzer,
    usage_category_t category,
    usage_category_stats_t* stats);

/**
 * Get overall metrics
 */
bool usage_get_metrics(
    usage_analyzer_t* analyzer,
    usage_metrics_t* metrics);

/**
 * Get session summary
 */
bool usage_get_session_summary(
    usage_analyzer_t* analyzer,
    uint32_t session_id,
    usage_session_summary_t* summary);

/**
 * Detect usage patterns
 */
bool usage_detect_patterns(
    usage_analyzer_t* analyzer,
    usage_pattern_t** patterns,
    size_t* count);

/**
 * Get most used APIs
 */
bool usage_get_top_apis(
    usage_analyzer_t* analyzer,
    size_t n,
    usage_api_stats_t** stats,
    size_t* count);

/**
 * Get unused APIs
 */
bool usage_get_unused_apis(
    usage_analyzer_t* analyzer,
    char*** api_names,
    size_t* count);

/**
 * Get deprecated API usage
 */
bool usage_get_deprecated_usage(
    usage_analyzer_t* analyzer,
    usage_api_stats_t** stats,
    size_t* count);

// ============================================================================
// Recommendations
// ============================================================================

/**
 * Generate usage recommendations
 */
bool usage_generate_recommendations(
    usage_analyzer_t* analyzer,
    usage_recommendation_t** recommendations,
    size_t* count);

/**
 * Get migration path for deprecated API
 */
bool usage_get_migration_path(
    usage_analyzer_t* analyzer,
    const char* deprecated_api,
    char** replacement_api,
    char** migration_guide);

// ============================================================================
// Reporting
// ============================================================================

/**
 * Generate usage report
 */
char* usage_generate_report(usage_analyzer_t* analyzer);

/**
 * Export to JSON
 */
char* usage_export_json(usage_analyzer_t* analyzer);

/**
 * Export to file
 */
bool usage_export_to_file(
    usage_analyzer_t* analyzer,
    const char* filename);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Get category name
 */
const char* usage_category_name(usage_category_t category);

/**
 * Get frequency name
 */
const char* usage_frequency_name(usage_frequency_t frequency);

/**
 * Get pattern type name
 */
const char* usage_pattern_type_name(usage_pattern_type_t type);

/**
 * Get deprecation status name
 */
const char* usage_deprecation_name(deprecation_status_t status);

/**
 * Free allocated stats array
 */
void usage_free_api_stats(usage_api_stats_t* stats, size_t count);

/**
 * Free allocated patterns array
 */
void usage_free_patterns(usage_pattern_t* patterns, size_t count);

/**
 * Free allocated recommendations array
 */
void usage_free_recommendations(usage_recommendation_t* recs, size_t count);

/**
 * Free session summary resources
 */
void usage_free_session_summary(usage_session_summary_t* summary);

/**
 * Get last error message
 */
const char* usage_get_last_error(usage_analyzer_t* analyzer);

#ifdef __cplusplus
}
#endif

#endif // USAGE_ANALYZER_H
