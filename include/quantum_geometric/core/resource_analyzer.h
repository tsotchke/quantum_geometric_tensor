#ifndef RESOURCE_ANALYZER_H
#define RESOURCE_ANALYZER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Resource Analyzer Types
// ============================================================================

// Resource types
typedef enum {
    RESOURCE_MEMORY,
    RESOURCE_CPU,
    RESOURCE_GPU,
    RESOURCE_DISK,
    RESOURCE_NETWORK,
    RESOURCE_FILE_DESCRIPTORS,
    RESOURCE_THREADS,
    RESOURCE_TYPE_COUNT
} resource_type_t;

// Alert severity levels
typedef enum {
    ALERT_INFO,
    ALERT_WARNING,
    ALERT_CRITICAL,
    ALERT_EMERGENCY
} alert_severity_t;

// Memory allocation tracking entry
typedef struct {
    void* address;
    size_t size;
    const char* file;
    int line;
    const char* function;
    uint64_t timestamp_ns;
    uint32_t thread_id;
    bool is_freed;
} memory_allocation_t;

// Memory statistics
typedef struct {
    size_t current_allocated;       // Currently allocated bytes
    size_t peak_allocated;          // Peak allocation in bytes
    size_t total_allocated;         // Total bytes ever allocated
    size_t total_freed;             // Total bytes freed
    uint64_t allocation_count;      // Number of allocations
    uint64_t free_count;            // Number of frees
    double fragmentation_ratio;     // 0.0 (no fragmentation) - 1.0 (severe)
    size_t largest_free_block;      // Largest contiguous free block
    size_t virtual_memory_used;     // Virtual memory usage
    size_t resident_memory;         // Resident set size (RSS)
    size_t shared_memory;           // Shared memory usage
} memory_stats_t;

// CPU utilization per thread
typedef struct {
    uint32_t thread_id;
    char thread_name[64];
    double user_time_percent;       // User-mode CPU time
    double system_time_percent;     // Kernel-mode CPU time
    double total_percent;           // Total CPU usage
    uint64_t cpu_cycles;            // CPU cycles consumed
    uint64_t context_switches;      // Context switches
    int cpu_affinity;               // CPU core affinity (-1 for any)
} thread_cpu_stats_t;

// Overall CPU statistics
typedef struct {
    double overall_usage_percent;    // Overall CPU usage
    double user_percent;             // User-mode percentage
    double system_percent;           // System/kernel percentage
    double idle_percent;             // Idle percentage
    double iowait_percent;           // I/O wait percentage
    size_t num_cores;                // Number of CPU cores
    double per_core_usage[128];      // Per-core usage (up to 128 cores)
    double load_average_1min;        // 1-minute load average
    double load_average_5min;        // 5-minute load average
    double load_average_15min;       // 15-minute load average
} cpu_stats_t;

// GPU statistics
typedef struct {
    int device_id;
    char device_name[256];
    size_t memory_total;             // Total GPU memory
    size_t memory_used;              // Used GPU memory
    size_t memory_free;              // Free GPU memory
    double memory_usage_percent;     // Memory usage percentage
    double compute_usage_percent;    // Compute utilization
    double temperature_celsius;      // GPU temperature
    double power_watts;              // Power consumption
    uint64_t active_kernels;         // Active kernel count
} gpu_stats_t;

// Disk I/O statistics
typedef struct {
    char device_name[64];
    uint64_t bytes_read;             // Total bytes read
    uint64_t bytes_written;          // Total bytes written
    uint64_t read_ops;               // Read operations
    uint64_t write_ops;              // Write operations
    double read_bandwidth_mbps;      // Read bandwidth MB/s
    double write_bandwidth_mbps;     // Write bandwidth MB/s
    double avg_read_latency_ms;      // Average read latency
    double avg_write_latency_ms;     // Average write latency
    double io_utilization_percent;   // I/O utilization
} disk_stats_t;

// Network statistics
typedef struct {
    char interface_name[64];
    uint64_t bytes_sent;             // Total bytes sent
    uint64_t bytes_received;         // Total bytes received
    uint64_t packets_sent;           // Packets sent
    uint64_t packets_received;       // Packets received
    uint64_t errors_in;              // Receive errors
    uint64_t errors_out;             // Transmit errors
    double send_bandwidth_mbps;      // Send bandwidth MB/s
    double recv_bandwidth_mbps;      // Receive bandwidth MB/s
    uint32_t active_connections;     // Active TCP connections
} network_stats_t;

// File descriptor statistics
typedef struct {
    uint32_t open_fds;               // Currently open file descriptors
    uint32_t max_fds;                // Maximum allowed
    uint32_t socket_count;           // Open sockets
    uint32_t pipe_count;             // Open pipes
    uint32_t file_count;             // Open regular files
    double usage_percent;            // FD usage percentage
} fd_stats_t;

// Resource threshold configuration
typedef struct {
    resource_type_t resource;
    double warning_threshold;        // Warning level (0.0-1.0)
    double critical_threshold;       // Critical level (0.0-1.0)
    double emergency_threshold;      // Emergency level (0.0-1.0)
    bool enabled;                    // Whether alerting is enabled
} resource_threshold_t;

// Resource alert
typedef struct {
    resource_type_t resource;
    alert_severity_t severity;
    char message[512];
    double current_value;
    double threshold_value;
    uint64_t timestamp_ns;
    bool acknowledged;
} resource_alert_t;

// Potential resource leak
typedef struct {
    void* address;
    size_t size;
    const char* allocation_site;     // File:line where allocated
    uint64_t age_ms;                 // Time since allocation
    double leak_probability;         // Estimated probability this is a leak
    const char* suggested_fix;       // Suggested fix
} potential_leak_t;

// Resource analyzer configuration
typedef struct {
    bool track_allocations;          // Track individual allocations
    bool track_per_thread_cpu;       // Track per-thread CPU
    bool monitor_gpu;                // Monitor GPU if available
    bool monitor_network;            // Monitor network interfaces
    bool enable_leak_detection;      // Enable leak detection
    size_t allocation_history_size;  // Max allocations to track
    double sampling_interval_ms;     // Sampling interval
    size_t alert_buffer_size;        // Max alerts to store
} resource_analyzer_config_t;

// Opaque handle
typedef struct resource_analyzer resource_analyzer_t;

// ============================================================================
// Initialization and Configuration
// ============================================================================

// Create resource analyzer with default settings
resource_analyzer_t* resource_analyzer_create(void);

// Create with custom configuration
resource_analyzer_t* resource_analyzer_create_with_config(
    const resource_analyzer_config_t* config);

// Get default configuration
resource_analyzer_config_t resource_analyzer_default_config(void);

// Destroy resource analyzer
void resource_analyzer_destroy(resource_analyzer_t* analyzer);

// Reset all statistics
bool resource_analyzer_reset(resource_analyzer_t* analyzer);

// ============================================================================
// Memory Tracking
// ============================================================================

// Track a memory allocation
void resource_track_allocation(resource_analyzer_t* analyzer,
                               void* ptr,
                               size_t size,
                               const char* file,
                               int line,
                               const char* function);

// Track a memory free
void resource_track_free(resource_analyzer_t* analyzer, void* ptr);

// Get memory statistics
bool resource_get_memory_stats(resource_analyzer_t* analyzer,
                               memory_stats_t* stats);

// Get allocation history
bool resource_get_allocations(resource_analyzer_t* analyzer,
                              memory_allocation_t** allocations,
                              size_t* count);

// Find allocation by address
bool resource_find_allocation(resource_analyzer_t* analyzer,
                              void* address,
                              memory_allocation_t* allocation);

// Calculate fragmentation ratio
double resource_calculate_fragmentation(resource_analyzer_t* analyzer);

// Convenience macros for tracking
#define RESOURCE_TRACK_ALLOC(analyzer, ptr, size) \
    resource_track_allocation(analyzer, ptr, size, __FILE__, __LINE__, __func__)
#define RESOURCE_TRACK_FREE(analyzer, ptr) \
    resource_track_free(analyzer, ptr)

// ============================================================================
// CPU Monitoring
// ============================================================================

// Get overall CPU statistics
bool resource_get_cpu_stats(resource_analyzer_t* analyzer,
                            cpu_stats_t* stats);

// Get per-thread CPU statistics
bool resource_get_thread_cpu_stats(resource_analyzer_t* analyzer,
                                   thread_cpu_stats_t** stats,
                                   size_t* count);

// Get CPU stats for specific thread
bool resource_get_thread_cpu_by_id(resource_analyzer_t* analyzer,
                                   uint32_t thread_id,
                                   thread_cpu_stats_t* stats);

// Set CPU affinity for current thread
bool resource_set_cpu_affinity(int cpu_core);

// Get number of available CPU cores
size_t resource_get_cpu_core_count(void);

// ============================================================================
// GPU Monitoring
// ============================================================================

// Get number of GPUs
size_t resource_get_gpu_count(resource_analyzer_t* analyzer);

// Get GPU statistics by device ID
bool resource_get_gpu_stats(resource_analyzer_t* analyzer,
                            int device_id,
                            gpu_stats_t* stats);

// Get all GPU statistics
bool resource_get_all_gpu_stats(resource_analyzer_t* analyzer,
                                gpu_stats_t** stats,
                                size_t* count);

// Check if GPU is available
bool resource_gpu_available(void);

// ============================================================================
// Disk I/O Monitoring
// ============================================================================

// Get disk I/O statistics
bool resource_get_disk_stats(resource_analyzer_t* analyzer,
                             const char* device,
                             disk_stats_t* stats);

// Get all disk statistics
bool resource_get_all_disk_stats(resource_analyzer_t* analyzer,
                                 disk_stats_t** stats,
                                 size_t* count);

// ============================================================================
// Network Monitoring
// ============================================================================

// Get network statistics for interface
bool resource_get_network_stats(resource_analyzer_t* analyzer,
                                const char* interface,
                                network_stats_t* stats);

// Get all network interface statistics
bool resource_get_all_network_stats(resource_analyzer_t* analyzer,
                                    network_stats_t** stats,
                                    size_t* count);

// ============================================================================
// File Descriptor Tracking
// ============================================================================

// Get file descriptor statistics
bool resource_get_fd_stats(resource_analyzer_t* analyzer,
                           fd_stats_t* stats);

// List open file descriptors
bool resource_list_open_fds(resource_analyzer_t* analyzer,
                            int** fds,
                            size_t* count);

// ============================================================================
// Resource Leak Detection
// ============================================================================

// Run leak detection analysis
bool resource_detect_leaks(resource_analyzer_t* analyzer,
                           potential_leak_t** leaks,
                           size_t* count);

// Check for leaked allocations (allocations without matching frees)
bool resource_check_unfreed_allocations(resource_analyzer_t* analyzer,
                                        memory_allocation_t** unfreed,
                                        size_t* count);

// Mark allocation as intentionally long-lived (not a leak)
void resource_mark_intentional(resource_analyzer_t* analyzer, void* ptr);

// Get leak detection summary
bool resource_get_leak_summary(resource_analyzer_t* analyzer,
                               size_t* total_leaked_bytes,
                               size_t* leak_count);

// Free leak detection results
void resource_free_leak_results(potential_leak_t* leaks, size_t count);

// ============================================================================
// Threshold-Based Alerting
// ============================================================================

// Set threshold for a resource type
bool resource_set_threshold(resource_analyzer_t* analyzer,
                            const resource_threshold_t* threshold);

// Get current threshold for a resource type
bool resource_get_threshold(resource_analyzer_t* analyzer,
                            resource_type_t resource,
                            resource_threshold_t* threshold);

// Check all thresholds and generate alerts
size_t resource_check_thresholds(resource_analyzer_t* analyzer);

// Get pending alerts
bool resource_get_alerts(resource_analyzer_t* analyzer,
                         resource_alert_t** alerts,
                         size_t* count);

// Clear all alerts
void resource_clear_alerts(resource_analyzer_t* analyzer);

// Acknowledge an alert
void resource_acknowledge_alert(resource_analyzer_t* analyzer,
                                uint64_t alert_timestamp);

// Set alert callback
typedef void (*resource_alert_callback_t)(const resource_alert_t* alert,
                                          void* user_data);
void resource_set_alert_callback(resource_analyzer_t* analyzer,
                                 resource_alert_callback_t callback,
                                 void* user_data);

// ============================================================================
// Resource History and Trends
// ============================================================================

// Start recording resource history
bool resource_start_recording(resource_analyzer_t* analyzer,
                              resource_type_t resource,
                              double interval_ms);

// Stop recording resource history
void resource_stop_recording(resource_analyzer_t* analyzer,
                             resource_type_t resource);

// Get resource history
bool resource_get_history(resource_analyzer_t* analyzer,
                          resource_type_t resource,
                          double** values,
                          uint64_t** timestamps,
                          size_t* count);

// Get resource trend (positive = increasing, negative = decreasing)
double resource_get_trend(resource_analyzer_t* analyzer,
                          resource_type_t resource);

// ============================================================================
// Export and Reporting
// ============================================================================

// Export resource data to JSON
char* resource_export_json(resource_analyzer_t* analyzer);

// Export resource data to file
bool resource_export_to_file(resource_analyzer_t* analyzer,
                             const char* filename);

// Generate resource usage report
char* resource_generate_report(resource_analyzer_t* analyzer);

// ============================================================================
// Utility Functions
// ============================================================================

// Get resource type name
const char* resource_type_name(resource_type_t type);

// Get alert severity name
const char* resource_alert_severity_name(alert_severity_t severity);

// Format bytes to human-readable string
char* resource_format_bytes(size_t bytes);

// Get last error message
const char* resource_get_last_error(void);

// Get system page size
size_t resource_get_page_size(void);

// Get total system memory
size_t resource_get_total_memory(void);

// Get available system memory
size_t resource_get_available_memory(void);

#ifdef __cplusplus
}
#endif

#endif // RESOURCE_ANALYZER_H
