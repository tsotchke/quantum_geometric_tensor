/**
 * @file quantum_field_gpu_monitor.h
 * @brief GPU Monitoring for Quantum Field Operations
 *
 * Provides comprehensive GPU monitoring including:
 * - GPU utilization tracking
 * - Memory usage monitoring
 * - Temperature and power monitoring
 * - Kernel execution profiling
 * - Multi-GPU coordination
 * - Performance alerts
 *
 * Part of the QGTL Hardware Acceleration Framework.
 */

#ifndef QUANTUM_FIELD_GPU_MONITOR_H
#define QUANTUM_FIELD_GPU_MONITOR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================

#define GPU_MONITOR_MAX_GPUS 16
#define GPU_MONITOR_MAX_NAME_LENGTH 256
#define GPU_MONITOR_HISTORY_SIZE 1000
#define GPU_MONITOR_MAX_STREAMS 64
#define GPU_MONITOR_MAX_KERNELS 256

// ============================================================================
// Enumerations
// ============================================================================

/**
 * GPU health status
 */
typedef enum {
    GPU_HEALTH_HEALTHY,               // Normal operation
    GPU_HEALTH_WARNING,               // Minor issues
    GPU_HEALTH_CRITICAL,              // Severe issues
    GPU_HEALTH_FAILED,                // GPU failed
    GPU_HEALTH_UNKNOWN                // Cannot determine
} gpu_health_status_t;

/**
 * GPU throttle reason
 */
typedef enum {
    GPU_THROTTLE_NONE,                // No throttling
    GPU_THROTTLE_POWER,               // Power limit
    GPU_THROTTLE_THERMAL,             // Temperature limit
    GPU_THROTTLE_SYNC,                // Sync boost
    GPU_THROTTLE_BOARD_LIMIT,         // Board design limit
    GPU_THROTTLE_LOW_UTIL,            // Low utilization
    GPU_THROTTLE_UNKNOWN              // Unknown reason
} gpu_throttle_reason_t;

/**
 * Memory type
 */
typedef enum {
    GPU_MEMORY_GLOBAL,                // Global memory
    GPU_MEMORY_SHARED,                // Shared memory
    GPU_MEMORY_LOCAL,                 // Local memory
    GPU_MEMORY_CONSTANT,              // Constant memory
    GPU_MEMORY_TEXTURE,               // Texture memory
    GPU_MEMORY_L1_CACHE,              // L1 cache
    GPU_MEMORY_L2_CACHE               // L2 cache
} gpu_memory_type_t;

/**
 * Monitor event types
 */
typedef enum {
    GPU_EVENT_KERNEL_LAUNCH,          // Kernel launched
    GPU_EVENT_KERNEL_COMPLETE,        // Kernel completed
    GPU_EVENT_MEMORY_ALLOC,           // Memory allocated
    GPU_EVENT_MEMORY_FREE,            // Memory freed
    GPU_EVENT_MEMORY_COPY,            // Memory copy
    GPU_EVENT_SYNC,                   // Synchronization
    GPU_EVENT_THROTTLE_START,         // Throttling started
    GPU_EVENT_THROTTLE_END,           // Throttling ended
    GPU_EVENT_TEMP_WARNING,           // Temperature warning
    GPU_EVENT_MEMORY_WARNING,         // Memory low warning
    GPU_EVENT_ERROR                   // Error occurred
} gpu_event_type_t;

// ============================================================================
// Data Structures
// ============================================================================

/**
 * GPU device information
 */
typedef struct {
    int device_id;
    char name[GPU_MONITOR_MAX_NAME_LENGTH];
    int compute_capability_major;
    int compute_capability_minor;
    size_t total_memory;
    int multiprocessor_count;
    int max_threads_per_mp;
    int warp_size;
    int max_block_dim[3];
    int max_grid_dim[3];
    int clock_rate_khz;
    int memory_clock_rate_khz;
    int memory_bus_width;
    int l2_cache_size;
    bool unified_addressing;
    bool concurrent_kernels;
    bool ecc_enabled;
} gpu_device_info_t;

/**
 * GPU utilization metrics
 */
typedef struct {
    int device_id;
    uint64_t timestamp_ns;
    double gpu_utilization;           // 0.0 to 1.0
    double memory_utilization;        // 0.0 to 1.0
    double encoder_utilization;       // 0.0 to 1.0
    double decoder_utilization;       // 0.0 to 1.0
    size_t memory_used;
    size_t memory_free;
    double sm_clock_mhz;
    double memory_clock_mhz;
    double power_watts;
    double temperature_celsius;
    gpu_throttle_reason_t throttle_reason;
} gpu_utilization_t;

/**
 * Kernel execution info
 */
typedef struct {
    char kernel_name[GPU_MONITOR_MAX_NAME_LENGTH];
    int device_id;
    int stream_id;
    uint64_t start_ns;
    uint64_t end_ns;
    uint64_t duration_ns;
    int grid_dim[3];
    int block_dim[3];
    size_t shared_memory;
    size_t registers_per_thread;
    double occupancy;
    uint64_t flops;                   // Estimated FLOPS
    size_t bytes_read;
    size_t bytes_written;
} gpu_kernel_info_t;

/**
 * Memory operation info
 */
typedef struct {
    int src_device;                   // -1 for host
    int dst_device;                   // -1 for host
    size_t bytes;
    uint64_t start_ns;
    uint64_t end_ns;
    double bandwidth_gbps;
    bool async;
    int stream_id;
} gpu_memory_op_t;

/**
 * Stream statistics
 */
typedef struct {
    int stream_id;
    int device_id;
    uint64_t kernel_count;
    uint64_t memory_op_count;
    uint64_t total_kernel_time_ns;
    uint64_t total_idle_time_ns;
    double avg_occupancy;
} gpu_stream_stats_t;

/**
 * Performance alert
 */
typedef struct {
    gpu_event_type_t event_type;
    int device_id;
    uint64_t timestamp_ns;
    char message[512];
    double value;                     // Relevant metric value
    double threshold;                 // Threshold that triggered alert
    bool acknowledged;
} gpu_alert_t;

/**
 * GPU performance summary
 */
typedef struct {
    int device_id;
    uint64_t observation_period_ns;
    double avg_gpu_utilization;
    double avg_memory_utilization;
    double peak_gpu_utilization;
    double peak_memory_utilization;
    size_t peak_memory_used;
    double avg_temperature;
    double max_temperature;
    double avg_power;
    double max_power;
    uint64_t total_kernels;
    uint64_t total_memory_ops;
    double total_compute_time_ns;
    double total_transfer_time_ns;
    double compute_efficiency;        // Useful compute / total time
    gpu_health_status_t health;
} gpu_performance_summary_t;

/**
 * Monitor configuration
 */
typedef struct {
    bool enable_utilization_tracking;
    bool enable_kernel_profiling;
    bool enable_memory_tracking;
    bool enable_power_monitoring;
    bool enable_temperature_monitoring;
    bool enable_alerts;
    uint64_t polling_interval_ms;
    double temp_warning_threshold;
    double temp_critical_threshold;
    double memory_warning_threshold;  // Fraction 0.0-1.0
    size_t history_size;
} gpu_monitor_config_t;

/**
 * Opaque monitor handle
 */
typedef struct gpu_monitor gpu_monitor_t;

// ============================================================================
// Initialization
// ============================================================================

/**
 * Create GPU monitor
 */
gpu_monitor_t* gpu_monitor_create(void);

/**
 * Create with configuration
 */
gpu_monitor_t* gpu_monitor_create_with_config(
    const gpu_monitor_config_t* config);

/**
 * Get default configuration
 */
gpu_monitor_config_t gpu_monitor_default_config(void);

/**
 * Destroy monitor
 */
void gpu_monitor_destroy(gpu_monitor_t* monitor);

/**
 * Start monitoring
 */
bool gpu_monitor_start(gpu_monitor_t* monitor);

/**
 * Stop monitoring
 */
bool gpu_monitor_stop(gpu_monitor_t* monitor);

/**
 * Check if monitoring is active
 */
bool gpu_monitor_is_active(gpu_monitor_t* monitor);

// ============================================================================
// Device Information
// ============================================================================

/**
 * Get number of available GPUs
 */
int gpu_monitor_get_device_count(gpu_monitor_t* monitor);

/**
 * Get device information
 */
bool gpu_monitor_get_device_info(
    gpu_monitor_t* monitor,
    int device_id,
    gpu_device_info_t* info);

/**
 * Get all device information
 */
bool gpu_monitor_get_all_devices(
    gpu_monitor_t* monitor,
    gpu_device_info_t** devices,
    int* count);

// ============================================================================
// Real-time Monitoring
// ============================================================================

/**
 * Get current utilization
 */
bool gpu_monitor_get_utilization(
    gpu_monitor_t* monitor,
    int device_id,
    gpu_utilization_t* utilization);

/**
 * Get all GPU utilizations
 */
bool gpu_monitor_get_all_utilizations(
    gpu_monitor_t* monitor,
    gpu_utilization_t** utilizations,
    int* count);

/**
 * Get utilization history
 */
bool gpu_monitor_get_utilization_history(
    gpu_monitor_t* monitor,
    int device_id,
    gpu_utilization_t** history,
    size_t* count);

/**
 * Get current temperature
 */
bool gpu_monitor_get_temperature(
    gpu_monitor_t* monitor,
    int device_id,
    double* temperature_celsius);

/**
 * Get current power usage
 */
bool gpu_monitor_get_power(
    gpu_monitor_t* monitor,
    int device_id,
    double* power_watts);

/**
 * Get memory info
 */
bool gpu_monitor_get_memory_info(
    gpu_monitor_t* monitor,
    int device_id,
    size_t* used,
    size_t* free,
    size_t* total);

// ============================================================================
// Kernel Profiling
// ============================================================================

/**
 * Start kernel profiling
 */
bool gpu_monitor_start_kernel_profiling(
    gpu_monitor_t* monitor,
    int device_id);

/**
 * Stop kernel profiling
 */
bool gpu_monitor_stop_kernel_profiling(
    gpu_monitor_t* monitor,
    int device_id);

/**
 * Record kernel execution
 */
bool gpu_monitor_record_kernel(
    gpu_monitor_t* monitor,
    const gpu_kernel_info_t* kernel_info);

/**
 * Get kernel statistics
 */
bool gpu_monitor_get_kernel_stats(
    gpu_monitor_t* monitor,
    const char* kernel_name,
    uint64_t* total_calls,
    double* avg_duration_ns,
    double* total_time_ns);

/**
 * Get hottest kernels
 */
bool gpu_monitor_get_hottest_kernels(
    gpu_monitor_t* monitor,
    int device_id,
    size_t n,
    gpu_kernel_info_t** kernels,
    size_t* count);

/**
 * Get kernel history
 */
bool gpu_monitor_get_kernel_history(
    gpu_monitor_t* monitor,
    int device_id,
    gpu_kernel_info_t** kernels,
    size_t* count);

// ============================================================================
// Memory Tracking
// ============================================================================

/**
 * Record memory allocation
 */
bool gpu_monitor_record_alloc(
    gpu_monitor_t* monitor,
    int device_id,
    size_t bytes,
    void* ptr);

/**
 * Record memory free
 */
bool gpu_monitor_record_free(
    gpu_monitor_t* monitor,
    int device_id,
    void* ptr);

/**
 * Record memory transfer
 */
bool gpu_monitor_record_transfer(
    gpu_monitor_t* monitor,
    const gpu_memory_op_t* op);

/**
 * Get memory operation history
 */
bool gpu_monitor_get_memory_history(
    gpu_monitor_t* monitor,
    int device_id,
    gpu_memory_op_t** ops,
    size_t* count);

/**
 * Detect memory leaks
 */
bool gpu_monitor_detect_leaks(
    gpu_monitor_t* monitor,
    int device_id,
    void*** leaked_ptrs,
    size_t** leaked_sizes,
    size_t* count);

// ============================================================================
// Stream Monitoring
// ============================================================================

/**
 * Get stream statistics
 */
bool gpu_monitor_get_stream_stats(
    gpu_monitor_t* monitor,
    int device_id,
    int stream_id,
    gpu_stream_stats_t* stats);

/**
 * Get all stream statistics
 */
bool gpu_monitor_get_all_stream_stats(
    gpu_monitor_t* monitor,
    int device_id,
    gpu_stream_stats_t** stats,
    size_t* count);

// ============================================================================
// Alerts
// ============================================================================

/**
 * Check for alerts
 */
bool gpu_monitor_check_alerts(
    gpu_monitor_t* monitor,
    gpu_alert_t** alerts,
    size_t* count);

/**
 * Acknowledge alert
 */
bool gpu_monitor_acknowledge_alert(
    gpu_monitor_t* monitor,
    size_t alert_index);

/**
 * Clear all alerts
 */
void gpu_monitor_clear_alerts(gpu_monitor_t* monitor);

/**
 * Set alert thresholds
 */
bool gpu_monitor_set_thresholds(
    gpu_monitor_t* monitor,
    double temp_warning,
    double temp_critical,
    double memory_warning);

// ============================================================================
// Performance Summary
// ============================================================================

/**
 * Get performance summary
 */
bool gpu_monitor_get_summary(
    gpu_monitor_t* monitor,
    int device_id,
    gpu_performance_summary_t* summary);

/**
 * Get health status
 */
gpu_health_status_t gpu_monitor_get_health(
    gpu_monitor_t* monitor,
    int device_id);

/**
 * Reset statistics
 */
void gpu_monitor_reset_stats(
    gpu_monitor_t* monitor,
    int device_id);

/**
 * Reset all statistics
 */
void gpu_monitor_reset_all_stats(gpu_monitor_t* monitor);

// ============================================================================
// Reporting
// ============================================================================

/**
 * Generate monitor report
 */
char* gpu_monitor_generate_report(gpu_monitor_t* monitor);

/**
 * Generate device report
 */
char* gpu_monitor_generate_device_report(
    gpu_monitor_t* monitor,
    int device_id);

/**
 * Export to JSON
 */
char* gpu_monitor_export_json(gpu_monitor_t* monitor);

/**
 * Export to file
 */
bool gpu_monitor_export_to_file(
    gpu_monitor_t* monitor,
    const char* filename);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Get health status name
 */
const char* gpu_health_name(gpu_health_status_t status);

/**
 * Get throttle reason name
 */
const char* gpu_throttle_name(gpu_throttle_reason_t reason);

/**
 * Get memory type name
 */
const char* gpu_memory_type_name(gpu_memory_type_t type);

/**
 * Get event type name
 */
const char* gpu_event_name(gpu_event_type_t event);

/**
 * Free device info array
 */
void gpu_monitor_free_devices(gpu_device_info_t* devices);

/**
 * Free utilization array
 */
void gpu_monitor_free_utilizations(gpu_utilization_t* utilizations);

/**
 * Free kernel info array
 */
void gpu_monitor_free_kernels(gpu_kernel_info_t* kernels);

/**
 * Free memory op array
 */
void gpu_monitor_free_memory_ops(gpu_memory_op_t* ops);

/**
 * Free stream stats array
 */
void gpu_monitor_free_stream_stats(gpu_stream_stats_t* stats);

/**
 * Free alerts array
 */
void gpu_monitor_free_alerts(gpu_alert_t* alerts);

/**
 * Get last error message
 */
const char* gpu_monitor_get_last_error(gpu_monitor_t* monitor);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_FIELD_GPU_MONITOR_H
