#ifndef DISTRIBUTED_PERFORMANCE_MONITOR_H
#define DISTRIBUTED_PERFORMANCE_MONITOR_H

/**
 * @file performance_monitor.h
 * @brief Distributed system performance monitoring
 *
 * Provides real-time monitoring of system performance including CPU, GPU,
 * memory, network, and quantum device metrics for distributed training.
 */

#include <stddef.h>
#include <stdbool.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

// Configuration constants
#define PERF_MON_MAX_DEVICES 16
#define PERF_MON_HISTORY_LENGTH 1000
#define PERF_MON_SAMPLING_INTERVAL_MS 100
#define PERF_MON_ALERT_THRESHOLD 0.9

// Device state enumeration
typedef enum {
    DEVICE_STATE_UNKNOWN = 0,
    DEVICE_STATE_IDLE,
    DEVICE_STATE_ACTIVE,
    DEVICE_STATE_BUSY,
    DEVICE_STATE_OVERLOADED,
    DEVICE_STATE_ERROR,
    DEVICE_STATE_OFFLINE
} DeviceState;

// Alert type enumeration
typedef enum {
    ALERT_TYPE_CPU,
    ALERT_TYPE_MEMORY,
    ALERT_TYPE_GPU,
    ALERT_TYPE_NETWORK,
    ALERT_TYPE_TEMPERATURE,
    ALERT_TYPE_QUANTUM_ERROR,
    ALERT_TYPE_BOTTLENECK
} AlertType;

// System metrics snapshot
typedef struct {
    double cpu_usage;
    double gpu_usage;
    double memory_usage;
    double quantum_usage;
    double network_bandwidth;
    double disk_io;
    struct timespec timestamp;
} SystemMetrics;

// Device metrics
typedef struct {
    double utilization;
    double temperature;
    double power_usage;
    double memory_used;
    double error_rate;
    DeviceState state;
    int device_id;
} DeviceMetrics;

// Alert configuration
typedef struct {
    double cpu_threshold;
    double memory_threshold;
    double gpu_threshold;
    double temp_threshold;
    double error_rate_threshold;
    bool enable_alerts;
} AlertConfig;

// Alert information
typedef struct {
    AlertType type;
    int device_id;
    double value;
    double threshold;
    struct timespec timestamp;
    char message[256];
} AlertInfo;

// Alert callback function type
typedef void (*AlertCallback)(const AlertInfo* alert, void* user_data);

// Monitor configuration
typedef struct {
    int num_devices;
    size_t history_length;
    size_t sampling_interval_ms;
    AlertConfig alert_config;
    AlertCallback alert_callback;
    void* callback_user_data;
    bool enable_threading;
} MonitorConfig;

// Performance monitor (opaque)
typedef struct PerformanceMonitorImpl PerformanceMonitor;

// Initialize performance monitor
PerformanceMonitor* init_performance_monitor(const MonitorConfig* config);

// Start monitoring (background thread)
void perf_monitor_start(PerformanceMonitor* monitor);

// Stop monitoring
void perf_monitor_stop(PerformanceMonitor* monitor);

// Manual update (for non-threaded mode)
void perf_monitor_update(PerformanceMonitor* monitor);

// Get current system metrics
void perf_monitor_get_system_metrics(
    PerformanceMonitor* monitor,
    SystemMetrics* metrics);

// Get device metrics
void perf_monitor_get_device_metrics(
    PerformanceMonitor* monitor,
    int device_id,
    DeviceMetrics* metrics);

// Get metrics history
const SystemMetrics* perf_monitor_get_history(
    const PerformanceMonitor* monitor,
    size_t* count);

// Check if bottleneck detected
bool perf_monitor_has_bottleneck(const PerformanceMonitor* monitor);

// Get current alert count
size_t perf_monitor_get_alert_count(const PerformanceMonitor* monitor);

// Set alert callback
void perf_monitor_set_callback(
    PerformanceMonitor* monitor,
    AlertCallback callback,
    void* user_data);

// Reset statistics
void perf_monitor_reset(PerformanceMonitor* monitor);

// Clean up performance monitor
void cleanup_performance_monitor(PerformanceMonitor* monitor);

#ifdef __cplusplus
}
#endif

#endif // DISTRIBUTED_PERFORMANCE_MONITOR_H
