#ifndef PROGRESS_ANALYTICS_H
#define PROGRESS_ANALYTICS_H

/**
 * @file progress_analytics.h
 * @brief Progress analytics for distributed training
 *
 * Provides pattern analysis, trend detection, resource monitoring,
 * and completion time prediction for distributed training tasks.
 */

#include <stddef.h>
#include <stdbool.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

// Configuration constants
#define ANALYTICS_MAX_PATTERNS 100
#define ANALYTICS_MAX_TRENDS 100
#define ANALYTICS_HISTORY_SIZE 1000
#define ANALYTICS_MIN_SAMPLES 10
#define ANALYTICS_TREND_WINDOW 100

// Progress pattern
typedef struct {
    double* values;
    size_t length;
    size_t capacity;
    double mean;
    double std_dev;
    double min_val;
    double max_val;
    bool is_significant;
} ProgressPattern;

// Performance trend
typedef struct {
    double slope;
    double intercept;
    double r_squared;
    bool is_accelerating;
    double confidence;
    time_t start_time;
} PerformanceTrend;

// Resource utilization snapshot
typedef struct {
    double cpu_usage;
    double memory_usage;
    double network_usage;
    double quantum_usage;
    time_t timestamp;
} ResourceUtilization;

// Plan progress information
typedef struct {
    double overall_progress;      // 0.0 to 1.0
    size_t completed_steps;
    size_t total_steps;
    time_t start_time;
    time_t last_update_time;
    double estimated_remaining;   // seconds
    double average_step_time;     // seconds
} PlanProgress;

// Analytics configuration
typedef struct {
    size_t max_patterns;
    size_t max_trends;
    size_t history_size;
    size_t min_samples;
    size_t trend_window;
    double confidence_threshold;
    bool enable_prediction;
    bool enable_recommendations;
} AnalyticsConfig;

// Analytics report
typedef struct {
    // Pattern summary
    size_t num_patterns_detected;
    double pattern_consistency;

    // Trend summary
    double overall_trend_slope;
    bool is_accelerating;
    double trend_confidence;

    // Resource summary
    double avg_cpu_usage;
    double avg_memory_usage;
    double peak_cpu_usage;
    double peak_memory_usage;

    // Predictions
    time_t predicted_completion;
    double prediction_confidence;

    // Recommendations
    char recommendations[1024];
} AnalyticsReport;

// Progress analytics (opaque)
typedef struct ProgressAnalyticsImpl ProgressAnalytics;

// Initialize progress analytics
ProgressAnalytics* init_progress_analytics(const AnalyticsConfig* config);

// Update with new progress data
void analytics_update(
    ProgressAnalytics* analytics,
    const PlanProgress* progress);

// Update resource utilization
void analytics_update_resources(
    ProgressAnalytics* analytics,
    const ResourceUtilization* resources);

// Predict completion time
time_t analytics_predict_completion(
    ProgressAnalytics* analytics,
    const PlanProgress* progress);

// Get current trend
const PerformanceTrend* analytics_get_trend(
    const ProgressAnalytics* analytics);

// Get pattern count
size_t analytics_get_pattern_count(const ProgressAnalytics* analytics);

// Generate analytics report
void analytics_generate_report(
    ProgressAnalytics* analytics,
    const PlanProgress* progress,
    AnalyticsReport* report);

// Reset analytics
void analytics_reset(ProgressAnalytics* analytics);

// Clean up progress analytics
void cleanup_progress_analytics(ProgressAnalytics* analytics);

#ifdef __cplusplus
}
#endif

#endif // PROGRESS_ANALYTICS_H
