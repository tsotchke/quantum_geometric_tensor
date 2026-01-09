/**
 * @file error_correction_monitor.c
 * @brief Implementation of error correction monitoring system
 */

#include "quantum_geometric/monitoring/error_correction_monitor.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

// Helper functions
static bool write_metrics_to_log(FILE* log_file,
                               const CorrectionMetrics* metrics) {
    if (!log_file || !metrics) {
        return false;
    }

    char timestamp[32];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S",
             localtime(&metrics->timestamp));

    return fprintf(log_file,
                  "%s,%.4f,%.6f,%.4f,%zu,%zu\n",
                  timestamp,
                  metrics->success_rate,
                  metrics->avg_correction_time,
                  metrics->error_rate,
                  metrics->total_corrections,
                  metrics->failed_corrections) > 0;
}

static bool analyze_metrics_trend(const MonitorState* state,
                                double* trend_score) {
    if (!state || !trend_score || state->history_count < 2) {
        return false;
    }

    // Calculate trend in success rate
    double total_delta = 0.0;
    size_t count = 0;
    
    for (size_t i = 1; i < state->history_count; i++) {
        double delta = state->history[i].success_rate -
                      state->history[i-1].success_rate;
        total_delta += delta;
        count++;
    }

    *trend_score = count > 0 ? total_delta / count : 0.0;
    return true;
}

static AlertLevel determine_alert_level(const MonitorState* state) {
    if (!state) {
        return ALERT_ERROR;
    }

    double trend_score;
    if (!analyze_metrics_trend(state, &trend_score)) {
        return ALERT_ERROR;
    }

    if (state->current.success_rate < state->config.alert_threshold) {
        if (trend_score < -0.1) {
            return ALERT_CRITICAL;
        }
        return ALERT_ERROR;
    }

    if (trend_score < -0.05) {
        return ALERT_WARNING;
    }

    return ALERT_INFO;
}

bool init_correction_monitor(MonitorState* state,
                           const MonitorConfig* config) {
    if (!state || !config) {
        return false;
    }

    // Initialize state
    memset(state, 0, sizeof(MonitorState));
    memcpy(&state->config, config, sizeof(MonitorConfig));

    // Allocate history buffer
    size_t history_length = config->history_length;
    if (history_length > MAX_HISTORY_LENGTH) {
        history_length = MAX_HISTORY_LENGTH;
    }

    state->history = calloc(history_length, sizeof(CorrectionMetrics));
    if (!state->history) {
        return false;
    }

    // Open log file if enabled
    if (config->log_to_file && config->log_path) {
        state->log_file = fopen(config->log_path, "a");
        if (!state->log_file) {
            free(state->history);
            return false;
        }

        // Write header if new file
        if (ftell(state->log_file) == 0) {
            fprintf(state->log_file,
                    "timestamp,success_rate,avg_time,error_rate,"
                    "total_corrections,failed_corrections\n");
        }
    }

    state->start_time = time(NULL);
    return true;
}

void cleanup_correction_monitor(MonitorState* state) {
    if (state) {
        if (state->history) {
            free(state->history);
        }
        if (state->log_file) {
            fclose(state->log_file);
        }
        memset(state, 0, sizeof(MonitorState));
    }
}

bool record_correction_metrics(MonitorState* state,
                             const CorrectionState* correction_state,
                             double correction_time) {
    if (!state || !correction_state) {
        return false;
    }

    // Update current metrics
    state->current.timestamp = time(NULL);
    state->current.success_rate = correction_state->success_rate;
    state->current.avg_correction_time = correction_time;
    state->current.error_rate = 1.0 - correction_state->success_rate;
    state->current.total_corrections = correction_state->total_corrections;
    state->current.failed_corrections = correction_state->total_corrections -
                                      correction_state->total_successes;

    // Update peak correction time
    if (correction_time > state->peak_correction_time) {
        state->peak_correction_time = correction_time;
    }

    // Update cumulative success rate
    size_t total = state->current.total_corrections;
    if (total > 0) {
        state->cumulative_success_rate =
            ((state->cumulative_success_rate * (total - 1)) +
             state->current.success_rate) / total;
    }

    // Add to history buffer
    if (state->history_count < state->config.history_length) {
        memcpy(&state->history[state->history_count],
               &state->current,
               sizeof(CorrectionMetrics));
        state->history_count++;
    } else {
        // Shift history and add new entry
        memmove(state->history,
                &state->history[1],
                (state->config.history_length - 1) * sizeof(CorrectionMetrics));
        memcpy(&state->history[state->config.history_length - 1],
               &state->current,
               sizeof(CorrectionMetrics));
    }

    // Write to log file if enabled
    if (state->log_file) {
        if (!write_metrics_to_log(state->log_file, &state->current)) {
            return false;
        }
        fflush(state->log_file);
    }

    return true;
}

bool generate_correction_alert(const MonitorState* state,
                             AlertInfo* alert) {
    if (!state || !alert) {
        return false;
    }

    AlertLevel level = determine_alert_level(state);
    if (level == ALERT_INFO && !state->config.real_time_alerts) {
        return false;
    }

    alert->level = level;
    alert->timestamp = time(NULL);
    memcpy(&alert->metrics, &state->current, sizeof(CorrectionMetrics));

    // Generate alert message
    static char message[256];
    switch (level) {
        case ALERT_CRITICAL:
            snprintf(message, sizeof(message),
                    "Critical: Success rate %.2f%% below threshold %.2f%% "
                    "with negative trend",
                    state->current.success_rate * 100,
                    state->config.alert_threshold * 100);
            break;
        case ALERT_ERROR:
            snprintf(message, sizeof(message),
                    "Error: Success rate %.2f%% below threshold %.2f%%",
                    state->current.success_rate * 100,
                    state->config.alert_threshold * 100);
            break;
        case ALERT_WARNING:
            snprintf(message, sizeof(message),
                    "Warning: Declining success rate trend detected");
            break;
        case ALERT_INFO:
            snprintf(message, sizeof(message),
                    "Info: System operating normally, success rate %.2f%%",
                    state->current.success_rate * 100);
            break;
    }
    alert->message = message;

    return true;
}

bool generate_correction_report(const MonitorState* state,
                              time_t start_time,
                              time_t end_time,
                              const char* report_path) {
    if (!state || !report_path) {
        return false;
    }

    FILE* report_file = fopen(report_path, "w");
    if (!report_file) {
        return false;
    }

    // Write report header
    fprintf(report_file, "Error Correction Performance Report\n");
    fprintf(report_file, "Period: %s to %s\n",
            ctime(&start_time), ctime(&end_time));
    fprintf(report_file, "\nSummary Metrics:\n");
    fprintf(report_file, "Total Corrections: %zu\n",
            state->current.total_corrections);
    fprintf(report_file, "Failed Corrections: %zu\n",
            state->current.failed_corrections);
    fprintf(report_file, "Current Success Rate: %.2f%%\n",
            state->current.success_rate * 100);
    fprintf(report_file, "Cumulative Success Rate: %.2f%%\n",
            state->cumulative_success_rate * 100);
    fprintf(report_file, "Average Correction Time: %.6f s\n",
            state->current.avg_correction_time);
    fprintf(report_file, "Peak Correction Time: %.6f s\n",
            state->peak_correction_time);

    // Write historical data
    fprintf(report_file, "\nHistorical Data:\n");
    fprintf(report_file,
            "Timestamp,Success Rate,Avg Time,Error Rate,"
            "Total,Failed\n");

    for (size_t i = 0; i < state->history_count; i++) {
        const CorrectionMetrics* metrics = &state->history[i];
        if (metrics->timestamp >= start_time &&
            metrics->timestamp <= end_time) {
            write_metrics_to_log(report_file, metrics);
        }
    }

    fclose(report_file);
    return true;
}

bool get_current_metrics(const MonitorState* state,
                        CorrectionMetrics* metrics) {
    if (!state || !metrics) {
        return false;
    }

    memcpy(metrics, &state->current, sizeof(CorrectionMetrics));
    return true;
}

bool check_correction_health(const MonitorState* state) {
    if (!state) {
        return false;
    }

    // Check current success rate
    if (state->current.success_rate < state->config.alert_threshold) {
        return false;
    }

    // Check success rate trend
    double trend_score;
    if (!analyze_metrics_trend(state, &trend_score)) {
        return false;
    }

    if (trend_score < -0.1) {
        return false;
    }

    // Check correction time stability
    if (state->current.avg_correction_time > state->peak_correction_time * 0.9) {
        return false;
    }

    return true;
}

// Performance analysis functions
bool analyze_performance_trend(const MonitorState* state, PerformanceTrend* trend) {
    if (!state || !trend || state->history_count < 5) {
        return false;
    }

    // Linear regression on success rate over time
    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_x2 = 0.0;
    size_t n = state->history_count;

    for (size_t i = 0; i < n; i++) {
        double x = (double)i;
        double y = state->history[i].success_rate;
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_x2 += x * x;
    }

    double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    trend->rate = slope;

    // Determine direction
    if (slope > 0.001) {
        trend->direction = TREND_IMPROVING;
    } else if (slope < -0.001) {
        trend->direction = TREND_DECLINING;
    } else {
        trend->direction = TREND_STABLE;
    }

    // Calculate confidence (R-squared)
    double mean_y = sum_y / n;
    double ss_tot = 0.0, ss_res = 0.0;
    for (size_t i = 0; i < n; i++) {
        double y = state->history[i].success_rate;
        double y_pred = slope * i + (mean_y - slope * sum_x / n);
        ss_tot += (y - mean_y) * (y - mean_y);
        ss_res += (y - y_pred) * (y - y_pred);
    }
    trend->confidence = 1.0 - (ss_res / (ss_tot + 1e-10));

    return true;
}

bool monitor_detect_performance_degradation(const MonitorState* state) {
    PerformanceTrend trend;
    if (!analyze_performance_trend(state, &trend)) {
        return false;
    }
    return trend.direction == TREND_DECLINING && trend.confidence > 0.7;
}

bool monitor_detect_performance_improvement(const MonitorState* state) {
    PerformanceTrend trend;
    if (!analyze_performance_trend(state, &trend)) {
        return false;
    }
    return trend.direction == TREND_IMPROVING && trend.confidence > 0.7;
}

// Note: detect_performance_degradation is defined in workload_balancer.c with different signature
// Note: detect_performance_improvement is used in monitoring context only

// Resource monitoring functions
bool record_resource_metrics(MonitorState* state, const ResourceMetrics* metrics) {
    if (!state || !metrics) {
        return false;
    }

    if (state->resource_history_count < MAX_HISTORY_LENGTH) {
        state->resource_history[state->resource_history_count] = *metrics;
        state->resource_history_count++;
    } else {
        // Shift and add new
        memmove(state->resource_history, &state->resource_history[1],
                (MAX_HISTORY_LENGTH - 1) * sizeof(ResourceMetrics));
        state->resource_history[MAX_HISTORY_LENGTH - 1] = *metrics;
    }

    return true;
}

bool get_resource_statistics(const MonitorState* state, ResourceStats* stats) {
    if (!state || !stats || state->resource_history_count == 0) {
        return false;
    }

    stats->peak_cpu_usage = 0.0;
    stats->peak_memory_usage = 0.0;
    stats->avg_cpu_usage = 0.0;
    stats->avg_memory_usage = 0.0;

    for (size_t i = 0; i < state->resource_history_count; i++) {
        const ResourceMetrics* m = &state->resource_history[i];
        if (m->cpu_usage > stats->peak_cpu_usage) {
            stats->peak_cpu_usage = m->cpu_usage;
        }
        if (m->memory_usage > stats->peak_memory_usage) {
            stats->peak_memory_usage = m->memory_usage;
        }
        stats->avg_cpu_usage += m->cpu_usage;
        stats->avg_memory_usage += m->memory_usage;
    }

    stats->avg_cpu_usage /= state->resource_history_count;
    stats->avg_memory_usage /= state->resource_history_count;

    return true;
}

bool check_resource_thresholds(const MonitorState* state) {
    if (!state || state->resource_history_count == 0) {
        return false;
    }

    const ResourceMetrics* latest = &state->resource_history[state->resource_history_count - 1];
    return (latest->cpu_usage > 0.9 || latest->memory_usage > 0.9 ||
            latest->gpu_usage > 0.9);
}

// Error pattern analysis functions
bool record_error_pattern(MonitorState* state, const ErrorPattern* pattern) {
    if (!state || !pattern) {
        return false;
    }

    // Search for existing pattern
    for (size_t i = 0; i < state->pattern_count; i++) {
        ErrorPattern* existing = &state->detected_patterns[i];
        if (existing->size == pattern->size) {
            bool match = true;
            for (size_t j = 0; j < pattern->size; j++) {
                if (existing->locations[j] != pattern->locations[j] ||
                    existing->types[j] != pattern->types[j]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                existing->frequency += 0.01;  // Increment frequency
                return true;
            }
        }
    }

    // Add new pattern
    if (state->pattern_count < 64) {
        state->detected_patterns[state->pattern_count] = *pattern;
        state->detected_patterns[state->pattern_count].frequency = 0.01;
        state->detected_patterns[state->pattern_count].confidence = 0.5;
        state->pattern_count++;
    }

    return true;
}

ErrorPattern* monitor_detect_error_patterns(const MonitorState* state, size_t* num_patterns) {
    if (!state || !num_patterns) {
        return NULL;
    }

    // Return copy of detected patterns with frequency > threshold
    size_t count = 0;
    for (size_t i = 0; i < state->pattern_count; i++) {
        if (state->detected_patterns[i].frequency > 0.1) {
            count++;
        }
    }

    if (count == 0) {
        *num_patterns = 0;
        return NULL;
    }

    ErrorPattern* result = malloc(count * sizeof(ErrorPattern));
    if (!result) {
        return NULL;
    }

    size_t idx = 0;
    for (size_t i = 0; i < state->pattern_count; i++) {
        if (state->detected_patterns[i].frequency > 0.1) {
            result[idx] = state->detected_patterns[i];
            result[idx].confidence = fmin(state->detected_patterns[i].frequency * 10.0, 1.0);
            idx++;
        }
    }

    *num_patterns = count;
    return result;
}

// Note: detect_error_patterns is defined in error_patterns.c with different signature
// Use monitor_detect_error_patterns for monitoring context

bool match_error_pattern(const MonitorState* state, const ErrorPattern* pattern) {
    if (!state || !pattern) {
        return false;
    }

    for (size_t i = 0; i < state->pattern_count; i++) {
        const ErrorPattern* existing = &state->detected_patterns[i];
        if (existing->size == pattern->size && existing->frequency > 0.1) {
            bool match = true;
            for (size_t j = 0; j < pattern->size; j++) {
                if (existing->locations[j] != pattern->locations[j]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                return true;
            }
        }
    }

    return false;
}

// Real-time monitoring functions
bool start_real_time_monitoring(MonitorState* state) {
    if (!state) {
        return false;
    }
    state->real_time_active = true;
    return true;
}

bool stop_real_time_monitoring(MonitorState* state) {
    if (!state) {
        return false;
    }
    state->real_time_active = false;
    return true;
}

bool get_monitoring_stats(const MonitorState* state, MonitoringStats* stats) {
    if (!state || !stats) {
        return false;
    }

    stats->update_count = state->history_count;
    stats->last_update_time = state->current.timestamp;

    // Calculate average interval
    if (state->history_count > 1) {
        time_t total_time = state->history[state->history_count - 1].timestamp -
                           state->history[0].timestamp;
        stats->avg_update_interval = (double)total_time / (state->history_count - 1);
    } else {
        stats->avg_update_interval = 0.0;
    }

    return true;
}

// Pipeline integration functions
bool get_pipeline_statistics(const MonitorState* state, PipelineStats* stats) {
    if (!state || !stats) {
        return false;
    }

    stats->total_cycles = state->current.total_corrections;
    stats->success_rate = state->cumulative_success_rate;
    stats->avg_cycle_time = state->current.avg_correction_time;

    return true;
}

// Additional monitoring functions
Alert* get_pending_alerts(const MonitorState* state, size_t* num_alerts) {
    if (!state || !num_alerts) {
        return NULL;
    }

    // Generate alert if current state warrants it
    AlertLevel level = determine_alert_level(state);

    if (level == ALERT_WARNING || level == ALERT_ERROR || level == ALERT_CRITICAL) {
        Alert* alert = malloc(sizeof(Alert));
        if (!alert) {
            return NULL;
        }

        alert->level = level;
        alert->timestamp = time(NULL);
        memcpy(&alert->metrics, &state->current, sizeof(CorrectionMetrics));

        static char message[256];
        switch (level) {
            case ALERT_CRITICAL:
                snprintf(message, sizeof(message),
                        "Critical: Success rate %.2f%% with declining trend",
                        state->current.success_rate * 100);
                break;
            case ALERT_ERROR:
                snprintf(message, sizeof(message),
                        "Error: Success rate %.2f%% below threshold",
                        state->current.success_rate * 100);
                break;
            case ALERT_WARNING:
                snprintf(message, sizeof(message),
                        "Warning: Performance degradation detected");
                break;
            default:
                snprintf(message, sizeof(message), "Unknown alert");
                break;
        }
        alert->message = message;

        *num_alerts = 1;
        return alert;
    }

    *num_alerts = 0;
    return NULL;
}

HealthStatus check_system_health(const MonitorState* state) {
    if (!state) {
        return HEALTH_CRITICAL;
    }

    double success_rate = state->current.success_rate;

    if (success_rate >= 0.98) {
        return HEALTH_EXCELLENT;
    } else if (success_rate >= 0.95) {
        return HEALTH_GOOD;
    } else if (success_rate >= 0.90) {
        return HEALTH_FAIR;
    } else if (success_rate >= 0.80) {
        return HEALTH_POOR;
    } else {
        return HEALTH_CRITICAL;
    }
}

char* generate_monitoring_report(const MonitorState* state) {
    if (!state) {
        return NULL;
    }

    // Allocate buffer for report
    size_t buffer_size = 4096;
    char* report = malloc(buffer_size);
    if (!report) {
        return NULL;
    }

    // Build report
    int offset = 0;
    offset += snprintf(report + offset, buffer_size - offset,
                      "=== Error Correction Monitoring Report ===\n\n");

    offset += snprintf(report + offset, buffer_size - offset,
                      "Current Metrics:\n");
    offset += snprintf(report + offset, buffer_size - offset,
                      "  Success Rate: %.2f%%\n",
                      state->current.success_rate * 100);
    offset += snprintf(report + offset, buffer_size - offset,
                      "  Total Corrections: %zu\n",
                      state->current.total_corrections);
    offset += snprintf(report + offset, buffer_size - offset,
                      "  Failed Corrections: %zu\n",
                      state->current.failed_corrections);
    offset += snprintf(report + offset, buffer_size - offset,
                      "  Avg Correction Time: %.6f s\n",
                      state->current.avg_correction_time);

    offset += snprintf(report + offset, buffer_size - offset,
                      "\nSystem Health: ");
    HealthStatus health = check_system_health(state);
    const char* health_str[] = {"EXCELLENT", "GOOD", "FAIR", "POOR", "CRITICAL"};
    offset += snprintf(report + offset, buffer_size - offset,
                      "%s\n", health_str[health]);

    offset += snprintf(report + offset, buffer_size - offset,
                      "\nHistory: %zu entries\n", state->history_count);

    return report;
}
