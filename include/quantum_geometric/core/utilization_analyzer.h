/**
 * @file utilization_analyzer.h
 * @brief Resource Utilization Analysis for Quantum Geometric Operations
 *
 * Provides utilization tracking and analysis including:
 * - CPU/GPU utilization monitoring
 * - Memory utilization tracking
 * - Qubit utilization efficiency
 * - Gate operation efficiency
 * - Resource contention detection
 *
 * Part of the QGTL Monitoring Framework.
 */

#ifndef UTILIZATION_ANALYZER_H
#define UTILIZATION_ANALYZER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================

#define UTILIZATION_MAX_RESOURCES 64
#define UTILIZATION_HISTORY_SIZE 1000
#define UTILIZATION_MAX_NAME_LENGTH 128

// ============================================================================
// Enumerations
// ============================================================================

/**
 * Resource types for utilization tracking
 */
typedef enum {
    UTILIZATION_RESOURCE_CPU,
    UTILIZATION_RESOURCE_GPU,
    UTILIZATION_RESOURCE_MEMORY,
    UTILIZATION_RESOURCE_GPU_MEMORY,
    UTILIZATION_RESOURCE_QUBIT,
    UTILIZATION_RESOURCE_GATE,
    UTILIZATION_RESOURCE_NETWORK,
    UTILIZATION_RESOURCE_DISK,
    UTILIZATION_RESOURCE_CUSTOM
} utilization_resource_type_t;

/**
 * Utilization level classification
 */
typedef enum {
    UTILIZATION_LEVEL_IDLE,           // < 10%
    UTILIZATION_LEVEL_LOW,            // 10-30%
    UTILIZATION_LEVEL_MODERATE,       // 30-60%
    UTILIZATION_LEVEL_HIGH,           // 60-85%
    UTILIZATION_LEVEL_SATURATED,      // 85-95%
    UTILIZATION_LEVEL_OVERLOADED      // > 95%
} utilization_level_t;

// ============================================================================
// Data Structures
// ============================================================================

/**
 * Single utilization sample
 */
typedef struct {
    uint64_t timestamp_ns;
    double utilization_percent;       // 0.0 to 100.0
    double capacity_used;             // Absolute value used
    double capacity_total;            // Total capacity
} utilization_sample_t;

/**
 * Resource utilization statistics
 */
typedef struct {
    utilization_resource_type_t type;
    char name[UTILIZATION_MAX_NAME_LENGTH];
    double current_percent;
    double average_percent;
    double min_percent;
    double max_percent;
    double std_dev_percent;
    utilization_level_t current_level;
    uint64_t sample_count;
    uint64_t time_idle_ns;
    uint64_t time_active_ns;
    uint64_t time_saturated_ns;
} utilization_resource_stats_t;

/**
 * Efficiency metrics
 */
typedef struct {
    double overall_efficiency;        // 0.0 to 1.0
    double compute_efficiency;
    double memory_efficiency;
    double qubit_efficiency;
    double gate_efficiency;
    double parallelism_ratio;
    uint64_t wasted_cycles;
    uint64_t useful_cycles;
} utilization_efficiency_t;

/**
 * Contention detection result
 */
typedef struct {
    bool detected;
    utilization_resource_type_t resource;
    double severity;                  // 0.0 to 1.0
    char description[256];
    char recommendation[256];
} utilization_contention_t;

/**
 * Analyzer configuration
 */
typedef struct {
    bool enable_continuous_monitoring;
    bool enable_contention_detection;
    double sampling_interval_ms;
    size_t history_size;
    double idle_threshold;            // Below this = idle
    double saturated_threshold;       // Above this = saturated
} utilization_analyzer_config_t;

/**
 * Opaque analyzer handle
 */
typedef struct utilization_analyzer utilization_analyzer_t;

// ============================================================================
// Initialization and Configuration
// ============================================================================

/**
 * Create utilization analyzer with default configuration
 */
utilization_analyzer_t* utilization_analyzer_create(void);

/**
 * Create with custom configuration
 */
utilization_analyzer_t* utilization_analyzer_create_with_config(
    const utilization_analyzer_config_t* config);

/**
 * Get default configuration
 */
utilization_analyzer_config_t utilization_analyzer_default_config(void);

/**
 * Destroy utilization analyzer
 */
void utilization_analyzer_destroy(utilization_analyzer_t* analyzer);

/**
 * Reset all statistics
 */
bool utilization_analyzer_reset(utilization_analyzer_t* analyzer);

// ============================================================================
// Resource Registration
// ============================================================================

/**
 * Register a resource for monitoring
 */
bool utilization_register_resource(
    utilization_analyzer_t* analyzer,
    const char* name,
    utilization_resource_type_t type,
    double total_capacity);

/**
 * Unregister a resource
 */
bool utilization_unregister_resource(
    utilization_analyzer_t* analyzer,
    const char* name);

// ============================================================================
// Utilization Recording
// ============================================================================

/**
 * Record utilization sample
 */
bool utilization_record_sample(
    utilization_analyzer_t* analyzer,
    const char* resource_name,
    double used,
    double total);

/**
 * Record utilization as percentage
 */
bool utilization_record_percent(
    utilization_analyzer_t* analyzer,
    const char* resource_name,
    double percent);

/**
 * Record qubit utilization
 */
bool utilization_record_qubit_usage(
    utilization_analyzer_t* analyzer,
    size_t qubits_used,
    size_t qubits_available);

/**
 * Record gate utilization
 */
bool utilization_record_gate_usage(
    utilization_analyzer_t* analyzer,
    size_t gates_executed,
    size_t max_gates_possible);

// ============================================================================
// Analysis
// ============================================================================

/**
 * Get current utilization for a resource
 */
double utilization_get_current(
    utilization_analyzer_t* analyzer,
    const char* resource_name);

/**
 * Get statistics for a resource
 */
bool utilization_get_resource_stats(
    utilization_analyzer_t* analyzer,
    const char* resource_name,
    utilization_resource_stats_t* stats);

/**
 * Get statistics for all resources
 */
bool utilization_get_all_stats(
    utilization_analyzer_t* analyzer,
    utilization_resource_stats_t** stats,
    size_t* count);

/**
 * Calculate efficiency metrics
 */
bool utilization_calculate_efficiency(
    utilization_analyzer_t* analyzer,
    utilization_efficiency_t* efficiency);

/**
 * Detect resource contention
 */
bool utilization_detect_contention(
    utilization_analyzer_t* analyzer,
    utilization_contention_t** contentions,
    size_t* count);

/**
 * Get utilization history for a resource
 */
bool utilization_get_history(
    utilization_analyzer_t* analyzer,
    const char* resource_name,
    utilization_sample_t** samples,
    size_t* count);

// ============================================================================
// Reporting
// ============================================================================

/**
 * Generate utilization report
 */
char* utilization_generate_report(utilization_analyzer_t* analyzer);

/**
 * Export to JSON
 */
char* utilization_export_json(utilization_analyzer_t* analyzer);

/**
 * Export to file
 */
bool utilization_export_to_file(
    utilization_analyzer_t* analyzer,
    const char* filename);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Get resource type name
 */
const char* utilization_resource_type_name(utilization_resource_type_t type);

/**
 * Get level name
 */
const char* utilization_level_name(utilization_level_t level);

/**
 * Classify utilization level
 */
utilization_level_t utilization_classify_level(double percent);

/**
 * Free allocated stats array
 */
void utilization_free_stats(utilization_resource_stats_t* stats, size_t count);

/**
 * Free allocated samples array
 */
void utilization_free_samples(utilization_sample_t* samples, size_t count);

/**
 * Free allocated contentions array
 */
void utilization_free_contentions(utilization_contention_t* contentions, size_t count);

/**
 * Get last error message
 */
const char* utilization_get_last_error(utilization_analyzer_t* analyzer);

#ifdef __cplusplus
}
#endif

#endif // UTILIZATION_ANALYZER_H
