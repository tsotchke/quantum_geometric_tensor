/**
 * @file progress_analytics.c
 * @brief Progress analytics implementation
 *
 * Implements pattern analysis, trend detection, resource monitoring,
 * and completion time prediction for distributed training tasks.
 */

#include "quantum_geometric/distributed/progress_analytics.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

// Internal constants
#define PATTERN_INITIAL_CAPACITY 100
#define MIN_CONFIDENCE 0.5
#define PATTERN_CHANGE_THRESHOLD 2.0  // Standard deviations

// Progress analytics - internal structure
struct ProgressAnalyticsImpl {
    // Pattern analysis
    ProgressPattern** patterns;
    size_t num_patterns;
    size_t pattern_capacity;

    // Trend tracking
    PerformanceTrend* trends;
    size_t num_trends;
    size_t trend_capacity;

    // Resource monitoring
    ResourceUtilization* resource_history;
    size_t history_count;
    size_t history_index;
    size_t history_capacity;

    // Statistics
    double peak_cpu;
    double peak_memory;
    double total_cpu;
    double total_memory;
    size_t resource_samples;

    // Configuration
    AnalyticsConfig config;
};

// Forward declarations
static ProgressPattern* create_progress_pattern(size_t capacity);
static void cleanup_progress_pattern(ProgressPattern* pattern);
static void add_progress_value(ProgressPattern* pattern, double value);
static void update_pattern_statistics(ProgressPattern* pattern);
static bool detect_pattern_change(const ProgressPattern* pattern, double new_value);
static void compute_trend_metrics(PerformanceTrend* trend, const ProgressPattern* pattern);
static double compute_trend_confidence(const PerformanceTrend* trend, size_t sample_count);

// Initialize progress analytics
ProgressAnalytics* init_progress_analytics(const AnalyticsConfig* config) {
    ProgressAnalytics* analytics = calloc(1, sizeof(ProgressAnalytics));
    if (!analytics) return NULL;

    // Store configuration
    if (config) {
        analytics->config = *config;
    } else {
        // Default configuration
        analytics->config.max_patterns = ANALYTICS_MAX_PATTERNS;
        analytics->config.max_trends = ANALYTICS_MAX_TRENDS;
        analytics->config.history_size = ANALYTICS_HISTORY_SIZE;
        analytics->config.min_samples = ANALYTICS_MIN_SAMPLES;
        analytics->config.trend_window = ANALYTICS_TREND_WINDOW;
        analytics->config.confidence_threshold = 0.8;
        analytics->config.enable_prediction = true;
        analytics->config.enable_recommendations = true;
    }

    // Initialize pattern storage
    analytics->pattern_capacity = analytics->config.max_patterns;
    analytics->patterns = calloc(analytics->pattern_capacity, sizeof(ProgressPattern*));
    if (!analytics->patterns) {
        free(analytics);
        return NULL;
    }
    analytics->num_patterns = 0;

    // Create initial pattern
    ProgressPattern* initial = create_progress_pattern(PATTERN_INITIAL_CAPACITY);
    if (initial) {
        analytics->patterns[analytics->num_patterns++] = initial;
    }

    // Initialize trend tracking
    analytics->trend_capacity = analytics->config.max_trends;
    analytics->trends = calloc(analytics->trend_capacity, sizeof(PerformanceTrend));
    if (!analytics->trends) {
        cleanup_progress_pattern(initial);
        free(analytics->patterns);
        free(analytics);
        return NULL;
    }
    analytics->num_trends = 0;

    // Add initial trend
    if (analytics->trend_capacity > 0) {
        analytics->trends[0].slope = 0.0;
        analytics->trends[0].intercept = 0.0;
        analytics->trends[0].r_squared = 0.0;
        analytics->trends[0].is_accelerating = false;
        analytics->trends[0].confidence = 0.0;
        analytics->trends[0].start_time = time(NULL);
        analytics->num_trends = 1;
    }

    // Initialize resource history
    analytics->history_capacity = analytics->config.history_size;
    analytics->resource_history = calloc(analytics->history_capacity, sizeof(ResourceUtilization));
    if (!analytics->resource_history) {
        free(analytics->trends);
        cleanup_progress_pattern(initial);
        free(analytics->patterns);
        free(analytics);
        return NULL;
    }
    analytics->history_count = 0;
    analytics->history_index = 0;

    // Initialize statistics
    analytics->peak_cpu = 0.0;
    analytics->peak_memory = 0.0;
    analytics->total_cpu = 0.0;
    analytics->total_memory = 0.0;
    analytics->resource_samples = 0;

    return analytics;
}

// Create progress pattern
static ProgressPattern* create_progress_pattern(size_t capacity) {
    ProgressPattern* pattern = calloc(1, sizeof(ProgressPattern));
    if (!pattern) return NULL;

    pattern->values = calloc(capacity, sizeof(double));
    if (!pattern->values) {
        free(pattern);
        return NULL;
    }

    pattern->capacity = capacity;
    pattern->length = 0;
    pattern->mean = 0.0;
    pattern->std_dev = 0.0;
    pattern->min_val = 0.0;
    pattern->max_val = 0.0;
    pattern->is_significant = false;

    return pattern;
}

// Cleanup progress pattern
static void cleanup_progress_pattern(ProgressPattern* pattern) {
    if (!pattern) return;
    free(pattern->values);
    free(pattern);
}

// Add progress value to pattern
static void add_progress_value(ProgressPattern* pattern, double value) {
    if (!pattern || !pattern->values) return;

    // Expand if needed
    if (pattern->length >= pattern->capacity) {
        size_t new_capacity = pattern->capacity * 2;
        double* new_values = realloc(pattern->values, new_capacity * sizeof(double));
        if (!new_values) return;
        pattern->values = new_values;
        pattern->capacity = new_capacity;
    }

    pattern->values[pattern->length++] = value;
}

// Update pattern statistics
static void update_pattern_statistics(ProgressPattern* pattern) {
    if (!pattern || pattern->length == 0) return;

    // Compute mean
    double sum = 0.0;
    pattern->min_val = pattern->values[0];
    pattern->max_val = pattern->values[0];

    for (size_t i = 0; i < pattern->length; i++) {
        sum += pattern->values[i];
        if (pattern->values[i] < pattern->min_val) {
            pattern->min_val = pattern->values[i];
        }
        if (pattern->values[i] > pattern->max_val) {
            pattern->max_val = pattern->values[i];
        }
    }
    pattern->mean = sum / (double)pattern->length;

    // Compute standard deviation
    double variance = 0.0;
    for (size_t i = 0; i < pattern->length; i++) {
        double diff = pattern->values[i] - pattern->mean;
        variance += diff * diff;
    }
    variance /= (double)pattern->length;
    pattern->std_dev = sqrt(variance);

    // Check significance
    pattern->is_significant = (pattern->std_dev > 0.01 && pattern->length >= 10);
}

// Detect pattern change
static bool detect_pattern_change(const ProgressPattern* pattern, double new_value) {
    if (!pattern || pattern->length < 10 || pattern->std_dev < 1e-10) {
        return false;
    }

    double z_score = fabs(new_value - pattern->mean) / pattern->std_dev;
    return (z_score > PATTERN_CHANGE_THRESHOLD);
}

// Compute trend metrics using linear regression
static void compute_trend_metrics(PerformanceTrend* trend, const ProgressPattern* pattern) {
    if (!trend || !pattern || pattern->length < 2) return;

    // Use last N values for trend
    size_t n = pattern->length;
    if (n > ANALYTICS_TREND_WINDOW) {
        n = ANALYTICS_TREND_WINDOW;
    }

    size_t start = pattern->length - n;

    // Linear regression: y = slope * x + intercept
    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_xx = 0.0, sum_yy = 0.0;

    for (size_t i = 0; i < n; i++) {
        double x = (double)i;
        double y = pattern->values[start + i];
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_xx += x * x;
        sum_yy += y * y;
    }

    double mean_x = sum_x / (double)n;
    double mean_y = sum_y / (double)n;

    double denominator = sum_xx - (double)n * mean_x * mean_x;
    if (fabs(denominator) < 1e-10) {
        trend->slope = 0.0;
        trend->intercept = mean_y;
        trend->r_squared = 0.0;
    } else {
        trend->slope = (sum_xy - (double)n * mean_x * mean_y) / denominator;
        trend->intercept = mean_y - trend->slope * mean_x;

        // R-squared using efficient formula from sums (avoids second loop)
        // R^2 = [n * sum_xy - sum_x * sum_y]^2 / [(n * sum_xx - sum_x^2) * (n * sum_yy - sum_y^2)]
        double ss_xx = (double)n * sum_xx - sum_x * sum_x;
        double ss_yy = (double)n * sum_yy - sum_y * sum_y;
        double ss_xy = (double)n * sum_xy - sum_x * sum_y;
        if (fabs(ss_xx * ss_yy) > 1e-20) {
            trend->r_squared = (ss_xy * ss_xy) / (ss_xx * ss_yy);
        } else {
            trend->r_squared = 0.0;
        }
    }

    // Check for acceleration (positive second derivative)
    if (n >= 3) {
        double first_half_slope = 0.0, second_half_slope = 0.0;
        size_t half = n / 2;

        // Compute slopes for each half
        double sum1_x = 0.0, sum1_y = 0.0, sum1_xy = 0.0, sum1_xx = 0.0;
        for (size_t i = 0; i < half; i++) {
            double x = (double)i;
            double y = pattern->values[start + i];
            sum1_x += x;
            sum1_y += y;
            sum1_xy += x * y;
            sum1_xx += x * x;
        }
        double denom1 = sum1_xx - (double)half * (sum1_x / half) * (sum1_x / half);
        if (fabs(denom1) > 1e-10) {
            first_half_slope = (sum1_xy - (double)half * (sum1_x / half) * (sum1_y / half)) / denom1;
        }

        double sum2_x = 0.0, sum2_y = 0.0, sum2_xy = 0.0, sum2_xx = 0.0;
        for (size_t i = half; i < n; i++) {
            double x = (double)(i - half);
            double y = pattern->values[start + i];
            sum2_x += x;
            sum2_y += y;
            sum2_xy += x * y;
            sum2_xx += x * x;
        }
        size_t n2 = n - half;
        double denom2 = sum2_xx - (double)n2 * (sum2_x / n2) * (sum2_x / n2);
        if (fabs(denom2) > 1e-10) {
            second_half_slope = (sum2_xy - (double)n2 * (sum2_x / n2) * (sum2_y / n2)) / denom2;
        }

        trend->is_accelerating = (second_half_slope > first_half_slope);
    }
}

// Compute trend confidence
static double compute_trend_confidence(const PerformanceTrend* trend, size_t sample_count) {
    if (!trend || sample_count < 2) return 0.0;

    // Confidence based on R-squared and sample size
    double r_factor = trend->r_squared;
    double n_factor = 1.0 - exp(-(double)sample_count / 20.0);  // Asymptotic to 1

    return r_factor * n_factor;
}

// Update with new progress data
void analytics_update(ProgressAnalytics* analytics, const PlanProgress* progress) {
    if (!analytics || !progress) return;

    // Get current pattern
    if (analytics->num_patterns == 0) {
        ProgressPattern* pattern = create_progress_pattern(PATTERN_INITIAL_CAPACITY);
        if (pattern) {
            analytics->patterns[analytics->num_patterns++] = pattern;
        }
    }

    ProgressPattern* current = analytics->patterns[analytics->num_patterns - 1];
    if (!current) return;

    // Check for pattern change before adding
    bool changed = detect_pattern_change(current, progress->overall_progress);

    // Add progress value
    add_progress_value(current, progress->overall_progress);

    // Update statistics
    update_pattern_statistics(current);

    // If pattern changed, start a new one
    if (changed && analytics->num_patterns < analytics->pattern_capacity) {
        ProgressPattern* new_pattern = create_progress_pattern(PATTERN_INITIAL_CAPACITY);
        if (new_pattern) {
            add_progress_value(new_pattern, progress->overall_progress);
            analytics->patterns[analytics->num_patterns++] = new_pattern;
        }
    }

    // Update trends
    if (analytics->num_trends > 0) {
        PerformanceTrend* trend = &analytics->trends[analytics->num_trends - 1];
        compute_trend_metrics(trend, current);
        trend->confidence = compute_trend_confidence(trend, current->length);
    }
}

// Update resource utilization
void analytics_update_resources(ProgressAnalytics* analytics, const ResourceUtilization* resources) {
    if (!analytics || !resources) return;

    // Store in circular buffer
    analytics->resource_history[analytics->history_index] = *resources;
    analytics->history_index = (analytics->history_index + 1) % analytics->history_capacity;
    if (analytics->history_count < analytics->history_capacity) {
        analytics->history_count++;
    }

    // Update statistics
    analytics->resource_samples++;
    analytics->total_cpu += resources->cpu_usage;
    analytics->total_memory += resources->memory_usage;

    if (resources->cpu_usage > analytics->peak_cpu) {
        analytics->peak_cpu = resources->cpu_usage;
    }
    if (resources->memory_usage > analytics->peak_memory) {
        analytics->peak_memory = resources->memory_usage;
    }
}

// Predict completion time
time_t analytics_predict_completion(ProgressAnalytics* analytics, const PlanProgress* progress) {
    if (!analytics || !progress) return 0;

    // Simple linear extrapolation
    if (progress->overall_progress < 0.01) {
        // Not enough data
        return 0;
    }

    time_t now = time(NULL);
    double elapsed = difftime(now, progress->start_time);
    double progress_rate = progress->overall_progress / elapsed;

    if (progress_rate < 1e-10) {
        return 0;  // No progress
    }

    double remaining_progress = 1.0 - progress->overall_progress;
    double remaining_time = remaining_progress / progress_rate;

    // Adjust for trends if available
    if (analytics->num_trends > 0 && analytics->num_patterns > 0) {
        PerformanceTrend* trend = &analytics->trends[analytics->num_trends - 1];

        if (trend->confidence > MIN_CONFIDENCE) {
            // If accelerating, reduce remaining time estimate
            if (trend->is_accelerating && trend->slope > 0) {
                remaining_time *= 0.9;  // 10% faster
            } else if (!trend->is_accelerating && trend->slope < 0) {
                remaining_time *= 1.1;  // 10% slower
            }
        }
    }

    return now + (time_t)remaining_time;
}

// Get current trend
const PerformanceTrend* analytics_get_trend(const ProgressAnalytics* analytics) {
    if (!analytics || analytics->num_trends == 0) return NULL;
    return &analytics->trends[analytics->num_trends - 1];
}

// Get pattern count
size_t analytics_get_pattern_count(const ProgressAnalytics* analytics) {
    return analytics ? analytics->num_patterns : 0;
}

// Generate analytics report
void analytics_generate_report(ProgressAnalytics* analytics, const PlanProgress* progress, AnalyticsReport* report) {
    if (!analytics || !report) return;

    memset(report, 0, sizeof(AnalyticsReport));

    // Pattern summary
    report->num_patterns_detected = analytics->num_patterns;

    if (analytics->num_patterns > 0) {
        ProgressPattern* current = analytics->patterns[analytics->num_patterns - 1];
        if (current && current->std_dev > 0) {
            // Consistency: inverse of coefficient of variation
            report->pattern_consistency = 1.0 / (1.0 + current->std_dev / (fabs(current->mean) + 1e-10));
        } else {
            report->pattern_consistency = 1.0;
        }
    }

    // Trend summary
    if (analytics->num_trends > 0) {
        PerformanceTrend* trend = &analytics->trends[analytics->num_trends - 1];
        report->overall_trend_slope = trend->slope;
        report->is_accelerating = trend->is_accelerating;
        report->trend_confidence = trend->confidence;
    }

    // Resource summary
    if (analytics->resource_samples > 0) {
        report->avg_cpu_usage = analytics->total_cpu / (double)analytics->resource_samples;
        report->avg_memory_usage = analytics->total_memory / (double)analytics->resource_samples;
        report->peak_cpu_usage = analytics->peak_cpu;
        report->peak_memory_usage = analytics->peak_memory;
    }

    // Predictions
    if (progress && analytics->config.enable_prediction) {
        report->predicted_completion = analytics_predict_completion(analytics, progress);
        report->prediction_confidence = (analytics->num_trends > 0) ?
            analytics->trends[analytics->num_trends - 1].confidence : 0.5;
    }

    // Recommendations
    if (analytics->config.enable_recommendations) {
        report->recommendations[0] = '\0';

        if (report->avg_cpu_usage > 0.9) {
            strncat(report->recommendations, "Consider reducing parallelism due to high CPU usage. ",
                    sizeof(report->recommendations) - strlen(report->recommendations) - 1);
        }

        if (report->avg_memory_usage > 0.85) {
            strncat(report->recommendations, "Memory pressure detected, consider batch size reduction. ",
                    sizeof(report->recommendations) - strlen(report->recommendations) - 1);
        }

        if (report->is_accelerating && report->trend_confidence > 0.7) {
            strncat(report->recommendations, "Progress is accelerating, continue current configuration. ",
                    sizeof(report->recommendations) - strlen(report->recommendations) - 1);
        }

        if (!report->is_accelerating && report->overall_trend_slope < 0 && report->trend_confidence > 0.7) {
            strncat(report->recommendations, "Progress is slowing, consider optimization review. ",
                    sizeof(report->recommendations) - strlen(report->recommendations) - 1);
        }
    }
}

// Reset analytics
void analytics_reset(ProgressAnalytics* analytics) {
    if (!analytics) return;

    // Clear patterns (keep first one)
    for (size_t i = 1; i < analytics->num_patterns; i++) {
        cleanup_progress_pattern(analytics->patterns[i]);
        analytics->patterns[i] = NULL;
    }
    if (analytics->num_patterns > 0 && analytics->patterns[0]) {
        analytics->patterns[0]->length = 0;
        analytics->patterns[0]->mean = 0.0;
        analytics->patterns[0]->std_dev = 0.0;
    }
    analytics->num_patterns = (analytics->num_patterns > 0) ? 1 : 0;

    // Reset trends
    if (analytics->num_trends > 0) {
        memset(&analytics->trends[0], 0, sizeof(PerformanceTrend));
        analytics->trends[0].start_time = time(NULL);
    }
    analytics->num_trends = 1;

    // Clear resource history
    memset(analytics->resource_history, 0, analytics->history_capacity * sizeof(ResourceUtilization));
    analytics->history_count = 0;
    analytics->history_index = 0;

    // Reset statistics
    analytics->peak_cpu = 0.0;
    analytics->peak_memory = 0.0;
    analytics->total_cpu = 0.0;
    analytics->total_memory = 0.0;
    analytics->resource_samples = 0;
}

// Clean up progress analytics
void cleanup_progress_analytics(ProgressAnalytics* analytics) {
    if (!analytics) return;

    // Clean up patterns
    for (size_t i = 0; i < analytics->num_patterns; i++) {
        cleanup_progress_pattern(analytics->patterns[i]);
    }
    free(analytics->patterns);

    // Clean up trends
    free(analytics->trends);

    // Clean up resource history
    free(analytics->resource_history);

    free(analytics);
}
