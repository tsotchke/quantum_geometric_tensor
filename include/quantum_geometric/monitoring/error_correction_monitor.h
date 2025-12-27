#ifndef ERROR_CORRECTION_MONITOR_H
#define ERROR_CORRECTION_MONITOR_H

#include <stdbool.h>
#include <stddef.h>
#include <time.h>
#include <stdio.h>

// Maximum history buffer length
#define MAX_HISTORY_LENGTH 1024

// Error types for pattern detection
typedef enum {
    ERROR_X,
    ERROR_Y,
    ERROR_Z
} error_type_t;

// Performance trend direction
typedef enum {
    TREND_IMPROVING,
    TREND_STABLE,
    TREND_DECLINING
} trend_direction_t;

// Alert levels for monitoring
typedef enum {
    ALERT_INFO,
    ALERT_WARNING,
    ALERT_ERROR,
    ALERT_CRITICAL
} AlertLevel;

// Configuration for the error correction monitor
typedef struct {
    size_t history_length;        // Number of historical data points to maintain
    double alert_threshold;       // Threshold for generating alerts
    bool log_to_file;            // Whether to log to file
    const char* log_path;        // Path for log file
    bool real_time_alerts;       // Enable real-time alerting
    bool track_resources;        // Enable resource tracking
    bool pattern_detection;      // Enable error pattern detection
    size_t update_interval_ms;   // Update interval for real-time monitoring
} MonitorConfig;

// State of a correction operation
typedef struct {
    double success_rate;          // Current success rate
    size_t total_corrections;     // Total number of corrections attempted
    size_t total_successes;       // Total number of successful corrections
} CorrectionState;

// Metrics for correction performance
typedef struct {
    time_t timestamp;             // Timestamp of the metrics
    double success_rate;          // Success rate for the period
    double error_rate;            // Error rate for the period
    double avg_correction_time;   // Average correction time
    double latency;               // Average correction latency
    size_t correction_count;      // Number of corrections in period
    size_t total_corrections;     // Total number of corrections
    size_t failed_corrections;    // Number of failed corrections
} CorrectionMetrics;

// Performance trend analysis
typedef struct {
    trend_direction_t direction;  // Trend direction
    double rate;                 // Rate of change
    double confidence;           // Confidence in trend analysis
} PerformanceTrend;

// Resource utilization metrics
typedef struct {
    double cpu_usage;            // CPU utilization (0-1)
    double memory_usage;         // Memory utilization (0-1)
    double gpu_usage;            // GPU utilization (0-1)
    double network_bandwidth;    // Network bandwidth utilization (0-1)
} ResourceMetrics;

// Resource utilization statistics
typedef struct {
    double peak_cpu_usage;       // Peak CPU usage
    double peak_memory_usage;    // Peak memory usage
    double avg_cpu_usage;        // Average CPU usage
    double avg_memory_usage;     // Average memory usage
} ResourceStats;

// Error pattern structure
typedef struct {
    size_t locations[16];        // Error locations
    error_type_t types[16];      // Error types
    size_t size;                // Number of errors in pattern
    double frequency;           // Pattern frequency
    double confidence;          // Confidence in pattern detection
} ErrorPattern;

// Real-time monitoring statistics
typedef struct {
    size_t update_count;         // Number of monitoring updates
    time_t last_update_time;     // Timestamp of last update
    double avg_update_interval;  // Average interval between updates
} MonitoringStats;

// Pipeline statistics
typedef struct {
    size_t total_cycles;         // Total correction cycles
    double success_rate;         // Overall success rate
    double avg_cycle_time;       // Average cycle time
} PipelineStats;

// Alert information structure
typedef struct {
    AlertLevel level;            // Alert severity level
    time_t timestamp;            // When alert was generated
    CorrectionMetrics metrics;   // Metrics at time of alert
    const char* message;         // Alert message
} AlertInfo;

// Full monitor state structure
typedef struct MonitorState {
    MonitorConfig config;                  // Configuration
    CorrectionMetrics current;             // Current metrics
    CorrectionMetrics* history;            // History buffer
    size_t history_count;                  // Number of history entries
    double cumulative_success_rate;        // Cumulative success rate
    double peak_correction_time;           // Peak correction time observed
    time_t start_time;                     // Start time of monitoring
    FILE* log_file;                        // Log file handle
    bool real_time_active;                 // Real-time monitoring active
    ResourceMetrics resource_history[MAX_HISTORY_LENGTH];  // Resource history
    size_t resource_history_count;         // Resource history count
    ErrorPattern detected_patterns[64];    // Detected error patterns
    size_t pattern_count;                  // Number of detected patterns
} MonitorState;

// Core monitoring functions
bool init_correction_monitor(MonitorState* state, const MonitorConfig* config);
void cleanup_correction_monitor(MonitorState* state);
bool record_correction_metrics(MonitorState* state, const CorrectionState* correction_state, double latency);

// Alert functions
bool generate_correction_alert(const MonitorState* state, AlertInfo* alert);
bool generate_correction_report(const MonitorState* state, time_t start_time, time_t end_time, const char* report_path);
bool get_current_metrics(const MonitorState* state, CorrectionMetrics* metrics);
bool check_correction_health(const MonitorState* state);

// Performance analysis functions
bool analyze_performance_trend(const MonitorState* state, PerformanceTrend* trend);
bool detect_performance_degradation(const MonitorState* state);
bool detect_performance_improvement(const MonitorState* state);

// Resource monitoring functions
bool record_resource_metrics(MonitorState* state, const ResourceMetrics* metrics);
bool get_resource_statistics(const MonitorState* state, ResourceStats* stats);
bool check_resource_thresholds(const MonitorState* state);

// Error pattern analysis functions
bool record_error_pattern(MonitorState* state, const ErrorPattern* pattern);
ErrorPattern* detect_error_patterns(const MonitorState* state, size_t* num_patterns);
bool match_error_pattern(const MonitorState* state, const ErrorPattern* pattern);

// Real-time monitoring functions
bool start_real_time_monitoring(MonitorState* state);
bool stop_real_time_monitoring(MonitorState* state);
bool get_monitoring_stats(const MonitorState* state, MonitoringStats* stats);

// Pipeline integration functions
bool get_pipeline_statistics(const MonitorState* state, PipelineStats* stats);

#endif // ERROR_CORRECTION_MONITOR_H
