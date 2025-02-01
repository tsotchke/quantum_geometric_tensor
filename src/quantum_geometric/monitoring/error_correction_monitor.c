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
