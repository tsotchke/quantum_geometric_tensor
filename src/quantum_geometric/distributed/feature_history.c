#include "quantum_geometric/distributed/feature_history.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

// History parameters
#define MAX_HISTORY_LENGTH 10000
#define MIN_PATTERN_LENGTH 10
#define ANOMALY_THRESHOLD 3.0
#define IMPORTANCE_THRESHOLD 0.1
#define MIN_VARIANCE 1e-10

// Pattern type
typedef enum {
    PATTERN_TREND,
    PATTERN_CYCLE,
    PATTERN_SPIKE,
    PATTERN_DRIFT,
    PATTERN_OSCILLATION
} PatternType;

// Feature pattern
typedef struct FeaturePattern {
    PatternType type;
    double* values;
    size_t length;
    double confidence;
    time_t detection_time;
    size_t feature_index;
} FeaturePattern;

// Temporal anomaly
typedef struct {
    double value;
    double severity;
    time_t timestamp;
    char* description;
    bool is_resolved;
    size_t feature_index;
} TemporalAnomaly;

// Feature importance
typedef struct {
    double correlation_score;
    double predictive_power;
    double stability_score;
    double overall_importance;
} FeatureImportance;

// Feature history tracker - named to match header typedef
struct FeatureHistoryTrackerImpl {
    // History storage
    double** feature_history;
    time_t** timestamps;
    size_t* history_lengths;
    size_t num_features;

    // Pattern detection
    FeaturePattern** detected_patterns;
    size_t num_patterns;
    size_t pattern_capacity;

    // Anomaly tracking
    TemporalAnomaly** anomalies;
    size_t num_anomalies;
    size_t anomaly_capacity;

    // Feature importance
    FeatureImportance* importance_scores;

    // Configuration
    HistoryConfig config;
};

// Forward declarations for static functions
static void detect_patterns(FeatureHistoryTracker* tracker);
static void detect_anomalies(FeatureHistoryTracker* tracker);
static void update_importance_scores(FeatureHistoryTracker* tracker);
static void shift_history(FeatureHistoryTracker* tracker, size_t feature_index);
static void detect_trend_patterns(FeatureHistoryTracker* tracker, size_t feature_index);
static void detect_cycle_patterns(FeatureHistoryTracker* tracker, size_t feature_index);
static void detect_complex_patterns(FeatureHistoryTracker* tracker, size_t feature_index);
static void detect_statistical_anomalies(FeatureHistoryTracker* tracker, size_t feature_index);
static void detect_pattern_anomalies(FeatureHistoryTracker* tracker, size_t feature_index);
static void detect_correlation_anomalies(FeatureHistoryTracker* tracker, size_t feature_index);
static double compute_correlation_score(FeatureHistoryTracker* tracker, size_t feature_index);
static double compute_predictive_power(FeatureHistoryTracker* tracker, size_t feature_index);
static double compute_stability_score(FeatureHistoryTracker* tracker, size_t feature_index);
static double compute_overall_importance(const FeatureImportance* importance);
static FeaturePattern** filter_patterns(FeatureHistoryTracker* tracker, size_t feature_index);
static size_t count_feature_patterns(FeatureHistoryTracker* tracker, size_t feature_index);
static void cleanup_feature_pattern(FeaturePattern* pattern);
static void cleanup_temporal_anomaly(TemporalAnomaly* anomaly);
static double compute_mean(const double* values, size_t length);
static double compute_std_dev(const double* values, size_t length, double mean);
static double compute_autocorrelation(const double* values, size_t length, size_t lag);

// Initialize feature history tracker
FeatureHistoryTracker* init_feature_history_tracker(
    const HistoryConfig* config) {

    if (!config || config->num_features == 0) return NULL;

    FeatureHistoryTracker* tracker = calloc(1, sizeof(FeatureHistoryTracker));
    if (!tracker) return NULL;

    tracker->num_features = config->num_features;
    tracker->config = *config;

    // Initialize history storage
    tracker->feature_history = calloc(config->num_features, sizeof(double*));
    tracker->timestamps = calloc(config->num_features, sizeof(time_t*));
    tracker->history_lengths = calloc(config->num_features, sizeof(size_t));

    if (!tracker->feature_history || !tracker->timestamps || !tracker->history_lengths) {
        cleanup_feature_history_tracker(tracker);
        return NULL;
    }

    for (size_t i = 0; i < config->num_features; i++) {
        tracker->feature_history[i] = calloc(MAX_HISTORY_LENGTH, sizeof(double));
        tracker->timestamps[i] = calloc(MAX_HISTORY_LENGTH, sizeof(time_t));
        tracker->history_lengths[i] = 0;
        if (!tracker->feature_history[i] || !tracker->timestamps[i]) {
            cleanup_feature_history_tracker(tracker);
            return NULL;
        }
    }

    // Initialize pattern storage
    tracker->pattern_capacity = MAX_PATTERNS;
    tracker->detected_patterns = calloc(MAX_PATTERNS, sizeof(FeaturePattern*));
    tracker->num_patterns = 0;

    // Initialize anomaly tracking
    tracker->anomaly_capacity = MAX_ANOMALIES;
    tracker->anomalies = calloc(MAX_ANOMALIES, sizeof(TemporalAnomaly*));
    tracker->num_anomalies = 0;

    // Initialize importance tracking
    tracker->importance_scores = calloc(config->num_features, sizeof(FeatureImportance));

    return tracker;
}

// Update feature history
void update_history(
    FeatureHistoryTracker* tracker,
    const double* features,
    time_t timestamp) {
    
    // Add new values to history
    for (size_t i = 0; i < tracker->num_features; i++) {
        size_t idx = tracker->history_lengths[i];
        if (idx >= MAX_HISTORY_LENGTH) {
            shift_history(tracker, i);
            idx = MAX_HISTORY_LENGTH - 1;
        }
        
        tracker->feature_history[i][idx] = features[i];
        tracker->timestamps[i][idx] = timestamp;
        tracker->history_lengths[i]++;
    }
    
    // Detect patterns
    detect_patterns(tracker);
    
    // Check for anomalies
    detect_anomalies(tracker);
    
    // Update importance scores
    update_importance_scores(tracker);
}

// Detect temporal patterns
static void detect_patterns(FeatureHistoryTracker* tracker) {
    for (size_t i = 0; i < tracker->num_features; i++) {
        if (tracker->history_lengths[i] < MIN_PATTERN_LENGTH) {
            continue;
        }
        
        // Detect trends
        detect_trend_patterns(tracker, i);
        
        // Detect cycles
        detect_cycle_patterns(tracker, i);
        
        // Detect other patterns
        detect_complex_patterns(tracker, i);
    }
}

// Detect anomalies
static void detect_anomalies(FeatureHistoryTracker* tracker) {
    for (size_t i = 0; i < tracker->num_features; i++) {
        if (tracker->history_lengths[i] < MIN_PATTERN_LENGTH) {
            continue;
        }
        
        // Statistical anomalies
        detect_statistical_anomalies(tracker, i);
        
        // Pattern-based anomalies
        detect_pattern_anomalies(tracker, i);
        
        // Correlation anomalies
        detect_correlation_anomalies(tracker, i);
    }
}

// Update feature importance scores
static void update_importance_scores(
    FeatureHistoryTracker* tracker) {
    
    for (size_t i = 0; i < tracker->num_features; i++) {
        FeatureImportance* importance = &tracker->importance_scores[i];
        
        // Compute correlation score
        importance->correlation_score = compute_correlation_score(
            tracker, i);
        
        // Compute predictive power
        importance->predictive_power = compute_predictive_power(
            tracker, i);
        
        // Compute stability score
        importance->stability_score = compute_stability_score(
            tracker, i);
        
        // Compute overall importance
        importance->overall_importance = compute_overall_importance(
            importance);
    }
}

// Get feature patterns
const FeaturePattern** get_patterns(
    FeatureHistoryTracker* tracker,
    size_t feature_index,
    size_t* num_patterns) {
    
    // Filter patterns for feature
    FeaturePattern** patterns = filter_patterns(
        tracker, feature_index);
    
    *num_patterns = count_feature_patterns(
        tracker, feature_index);
    
    return (const FeaturePattern**)patterns;
}

// Clean up
void cleanup_feature_history_tracker(
    FeatureHistoryTracker* tracker) {
    
    if (!tracker) return;
    
    // Clean up history storage
    for (size_t i = 0; i < tracker->num_features; i++) {
        free(tracker->feature_history[i]);
        free(tracker->timestamps[i]);
    }
    free(tracker->feature_history);
    free(tracker->timestamps);
    free(tracker->history_lengths);
    
    // Clean up patterns
    for (size_t i = 0; i < tracker->num_patterns; i++) {
        cleanup_feature_pattern(tracker->detected_patterns[i]);
    }
    free(tracker->detected_patterns);
    
    // Clean up anomalies
    for (size_t i = 0; i < tracker->num_anomalies; i++) {
        cleanup_temporal_anomaly(tracker->anomalies[i]);
    }
    free(tracker->anomalies);
    
    // Clean up importance scores
    free(tracker->importance_scores);

    free(tracker);
}

// ============================================================================
// Helper function implementations
// ============================================================================

// Compute mean of values
static double compute_mean(const double* values, size_t length) {
    if (!values || length == 0) return 0.0;

    double sum = 0.0;
    for (size_t i = 0; i < length; i++) {
        sum += values[i];
    }
    return sum / (double)length;
}

// Compute standard deviation
static double compute_std_dev(const double* values, size_t length, double mean) {
    if (!values || length < 2) return 0.0;

    double sum_sq = 0.0;
    for (size_t i = 0; i < length; i++) {
        double diff = values[i] - mean;
        sum_sq += diff * diff;
    }
    return sqrt(sum_sq / (double)(length - 1));
}

// Compute autocorrelation at a given lag
static double compute_autocorrelation(const double* values, size_t length, size_t lag) {
    if (!values || length <= lag || lag == 0) return 0.0;

    double mean = compute_mean(values, length);
    double variance = 0.0;
    double covariance = 0.0;

    for (size_t i = 0; i < length; i++) {
        double diff = values[i] - mean;
        variance += diff * diff;
    }

    if (variance < MIN_VARIANCE) return 0.0;

    for (size_t i = 0; i < length - lag; i++) {
        covariance += (values[i] - mean) * (values[i + lag] - mean);
    }

    return covariance / variance;
}

// Shift history to make room for new values (circular buffer behavior)
static void shift_history(FeatureHistoryTracker* tracker, size_t feature_index) {
    if (!tracker || feature_index >= tracker->num_features) return;

    // Shift all values left by 1
    memmove(tracker->feature_history[feature_index],
            tracker->feature_history[feature_index] + 1,
            (MAX_HISTORY_LENGTH - 1) * sizeof(double));
    memmove(tracker->timestamps[feature_index],
            tracker->timestamps[feature_index] + 1,
            (MAX_HISTORY_LENGTH - 1) * sizeof(time_t));

    tracker->history_lengths[feature_index] = MAX_HISTORY_LENGTH - 1;
}

// Add a pattern to the tracker
static void add_pattern(FeatureHistoryTracker* tracker, FeaturePattern* pattern) {
    if (!tracker || !pattern) return;

    if (tracker->num_patterns >= tracker->pattern_capacity) {
        // Remove oldest pattern
        cleanup_feature_pattern(tracker->detected_patterns[0]);
        memmove(tracker->detected_patterns, tracker->detected_patterns + 1,
                (tracker->pattern_capacity - 1) * sizeof(FeaturePattern*));
        tracker->num_patterns--;
    }

    tracker->detected_patterns[tracker->num_patterns++] = pattern;
}

// Create a new pattern
static FeaturePattern* create_pattern(PatternType type, size_t feature_index,
                                       const double* values, size_t length,
                                       double confidence) {
    FeaturePattern* pattern = calloc(1, sizeof(FeaturePattern));
    if (!pattern) return NULL;

    pattern->type = type;
    pattern->feature_index = feature_index;
    pattern->confidence = confidence;
    pattern->detection_time = time(NULL);
    pattern->length = length;

    if (length > 0 && values) {
        pattern->values = calloc(length, sizeof(double));
        if (pattern->values) {
            memcpy(pattern->values, values, length * sizeof(double));
        }
    }

    return pattern;
}

// Detect trend patterns using linear regression
static void detect_trend_patterns(FeatureHistoryTracker* tracker, size_t feature_index) {
    if (!tracker || feature_index >= tracker->num_features) return;

    size_t len = tracker->history_lengths[feature_index];
    if (len < MIN_PATTERN_LENGTH) return;

    double* values = tracker->feature_history[feature_index];

    // Compute linear regression slope
    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_xx = 0.0;
    for (size_t i = 0; i < len; i++) {
        double x = (double)i;
        sum_x += x;
        sum_y += values[i];
        sum_xy += x * values[i];
        sum_xx += x * x;
    }

    double n = (double)len;
    double denom = n * sum_xx - sum_x * sum_x;
    if (fabs(denom) < MIN_VARIANCE) return;

    double slope = (n * sum_xy - sum_x * sum_y) / denom;

    // Compute R-squared as confidence
    double mean_y = sum_y / n;
    double ss_tot = 0.0, ss_res = 0.0;
    double intercept = (sum_y - slope * sum_x) / n;

    for (size_t i = 0; i < len; i++) {
        double y_pred = slope * (double)i + intercept;
        double y_diff = values[i] - mean_y;
        double res = values[i] - y_pred;
        ss_tot += y_diff * y_diff;
        ss_res += res * res;
    }

    double r_squared = (ss_tot > MIN_VARIANCE) ? 1.0 - (ss_res / ss_tot) : 0.0;

    // Only add pattern if trend is significant
    double std_dev = compute_std_dev(values, len, mean_y);
    double normalized_slope = (std_dev > MIN_VARIANCE) ? fabs(slope) / std_dev : 0.0;

    if (normalized_slope > 0.1 && r_squared > 0.5) {
        FeaturePattern* pattern = create_pattern(PATTERN_TREND, feature_index,
                                                  values, len, r_squared);
        if (pattern) {
            add_pattern(tracker, pattern);
        }
    }
}

// Detect cycle patterns using autocorrelation
static void detect_cycle_patterns(FeatureHistoryTracker* tracker, size_t feature_index) {
    if (!tracker || feature_index >= tracker->num_features) return;

    size_t len = tracker->history_lengths[feature_index];
    if (len < MIN_PATTERN_LENGTH * 2) return;

    double* values = tracker->feature_history[feature_index];

    // Find peak autocorrelation lag
    double max_autocorr = 0.0;
    size_t best_lag = 0;

    for (size_t lag = MIN_PATTERN_LENGTH / 2; lag < len / 2; lag++) {
        double autocorr = compute_autocorrelation(values, len, lag);
        if (autocorr > max_autocorr) {
            max_autocorr = autocorr;
            best_lag = lag;
        }
    }

    // Add cycle pattern if significant autocorrelation found
    if (max_autocorr > 0.5 && best_lag > 0) {
        FeaturePattern* pattern = create_pattern(PATTERN_CYCLE, feature_index,
                                                  values, best_lag, max_autocorr);
        if (pattern) {
            add_pattern(tracker, pattern);
        }
    }
}

// Detect complex patterns (spikes, drifts, oscillations)
static void detect_complex_patterns(FeatureHistoryTracker* tracker, size_t feature_index) {
    if (!tracker || feature_index >= tracker->num_features) return;

    size_t len = tracker->history_lengths[feature_index];
    if (len < MIN_PATTERN_LENGTH) return;

    double* values = tracker->feature_history[feature_index];
    double mean = compute_mean(values, len);
    double std_dev = compute_std_dev(values, len, mean);

    if (std_dev < MIN_VARIANCE) return;

    // Detect spikes (values > 3 sigma from mean)
    size_t spike_count = 0;
    for (size_t i = 0; i < len; i++) {
        if (fabs(values[i] - mean) > 3.0 * std_dev) {
            spike_count++;
        }
    }

    if (spike_count > 0 && spike_count < len / 10) {
        double confidence = 1.0 - ((double)spike_count / (double)len);
        FeaturePattern* pattern = create_pattern(PATTERN_SPIKE, feature_index,
                                                  values, len, confidence);
        if (pattern) {
            add_pattern(tracker, pattern);
        }
    }

    // Detect oscillation using sign changes in derivative
    size_t sign_changes = 0;
    for (size_t i = 1; i < len - 1; i++) {
        double d1 = values[i] - values[i-1];
        double d2 = values[i+1] - values[i];
        if ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) {
            sign_changes++;
        }
    }

    double oscillation_rate = (double)sign_changes / (double)(len - 2);
    if (oscillation_rate > 0.3) {
        FeaturePattern* pattern = create_pattern(PATTERN_OSCILLATION, feature_index,
                                                  values, len, oscillation_rate);
        if (pattern) {
            add_pattern(tracker, pattern);
        }
    }
}

// Add anomaly to tracker
static void add_anomaly(FeatureHistoryTracker* tracker, TemporalAnomaly* anomaly) {
    if (!tracker || !anomaly) return;

    if (tracker->num_anomalies >= tracker->anomaly_capacity) {
        // Remove oldest anomaly
        cleanup_temporal_anomaly(tracker->anomalies[0]);
        memmove(tracker->anomalies, tracker->anomalies + 1,
                (tracker->anomaly_capacity - 1) * sizeof(TemporalAnomaly*));
        tracker->num_anomalies--;
    }

    tracker->anomalies[tracker->num_anomalies++] = anomaly;
}

// Create a new anomaly
static TemporalAnomaly* create_anomaly(double value, double severity,
                                        size_t feature_index, const char* description) {
    TemporalAnomaly* anomaly = calloc(1, sizeof(TemporalAnomaly));
    if (!anomaly) return NULL;

    anomaly->value = value;
    anomaly->severity = severity;
    anomaly->timestamp = time(NULL);
    anomaly->is_resolved = false;
    anomaly->feature_index = feature_index;

    if (description) {
        anomaly->description = strdup(description);
    }

    return anomaly;
}

// Detect statistical anomalies using z-score
static void detect_statistical_anomalies(FeatureHistoryTracker* tracker, size_t feature_index) {
    if (!tracker || feature_index >= tracker->num_features) return;

    size_t len = tracker->history_lengths[feature_index];
    if (len < MIN_PATTERN_LENGTH) return;

    double* values = tracker->feature_history[feature_index];
    double mean = compute_mean(values, len);
    double std_dev = compute_std_dev(values, len, mean);

    if (std_dev < MIN_VARIANCE) return;

    double threshold = tracker->config.anomaly_threshold > 0 ?
                       tracker->config.anomaly_threshold : ANOMALY_THRESHOLD;

    // Check most recent value
    double latest = values[len - 1];
    double z_score = (latest - mean) / std_dev;

    if (fabs(z_score) > threshold) {
        char desc[128];
        snprintf(desc, sizeof(desc), "Statistical anomaly: z-score=%.2f", z_score);
        TemporalAnomaly* anomaly = create_anomaly(latest, fabs(z_score) / threshold,
                                                   feature_index, desc);
        if (anomaly) {
            add_anomaly(tracker, anomaly);
        }
    }
}

// Detect pattern-based anomalies
static void detect_pattern_anomalies(FeatureHistoryTracker* tracker, size_t feature_index) {
    if (!tracker || feature_index >= tracker->num_features) return;

    size_t len = tracker->history_lengths[feature_index];
    if (len < MIN_PATTERN_LENGTH * 2) return;

    double* values = tracker->feature_history[feature_index];

    // Compare recent window to historical pattern
    size_t window_size = MIN_PATTERN_LENGTH;
    double* recent = values + len - window_size;
    double* historical = values + len - 2 * window_size;

    double recent_mean = compute_mean(recent, window_size);
    double hist_mean = compute_mean(historical, window_size);
    double hist_std = compute_std_dev(historical, window_size, hist_mean);

    if (hist_std < MIN_VARIANCE) return;

    double deviation = fabs(recent_mean - hist_mean) / hist_std;
    if (deviation > ANOMALY_THRESHOLD) {
        char desc[128];
        snprintf(desc, sizeof(desc), "Pattern deviation: %.2f sigma", deviation);
        TemporalAnomaly* anomaly = create_anomaly(recent_mean, deviation / ANOMALY_THRESHOLD,
                                                   feature_index, desc);
        if (anomaly) {
            add_anomaly(tracker, anomaly);
        }
    }
}

// Detect correlation anomalies
static void detect_correlation_anomalies(FeatureHistoryTracker* tracker, size_t feature_index) {
    if (!tracker || feature_index >= tracker->num_features) return;
    if (tracker->num_features < 2) return;

    size_t len = tracker->history_lengths[feature_index];
    if (len < MIN_PATTERN_LENGTH) return;

    double* values = tracker->feature_history[feature_index];

    // Compare with other features for correlation breakdown
    for (size_t j = 0; j < tracker->num_features; j++) {
        if (j == feature_index) continue;
        if (tracker->history_lengths[j] < len) continue;

        double* other = tracker->feature_history[j];

        // Compute correlation
        double mean1 = compute_mean(values, len);
        double mean2 = compute_mean(other, len);
        double cov = 0.0, var1 = 0.0, var2 = 0.0;

        for (size_t i = 0; i < len; i++) {
            double d1 = values[i] - mean1;
            double d2 = other[i] - mean2;
            cov += d1 * d2;
            var1 += d1 * d1;
            var2 += d2 * d2;
        }

        if (var1 < MIN_VARIANCE || var2 < MIN_VARIANCE) continue;

        double correlation = cov / sqrt(var1 * var2);

        // Check if correlation has changed significantly in recent window
        if (len >= MIN_PATTERN_LENGTH * 2) {
            size_t half = len / 2;
            double early_corr = 0.0, late_corr = 0.0;

            // Early half correlation
            double em1 = compute_mean(values, half);
            double em2 = compute_mean(other, half);
            double ecov = 0.0, ev1 = 0.0, ev2 = 0.0;
            for (size_t i = 0; i < half; i++) {
                double d1 = values[i] - em1;
                double d2 = other[i] - em2;
                ecov += d1 * d2;
                ev1 += d1 * d1;
                ev2 += d2 * d2;
            }
            if (ev1 > MIN_VARIANCE && ev2 > MIN_VARIANCE) {
                early_corr = ecov / sqrt(ev1 * ev2);
            }

            // Late half correlation
            double lm1 = compute_mean(values + half, len - half);
            double lm2 = compute_mean(other + half, len - half);
            double lcov = 0.0, lv1 = 0.0, lv2 = 0.0;
            for (size_t i = half; i < len; i++) {
                double d1 = values[i] - lm1;
                double d2 = other[i] - lm2;
                lcov += d1 * d2;
                lv1 += d1 * d1;
                lv2 += d2 * d2;
            }
            if (lv1 > MIN_VARIANCE && lv2 > MIN_VARIANCE) {
                late_corr = lcov / sqrt(lv1 * lv2);
            }

            double corr_change = fabs(late_corr - early_corr);
            if (corr_change > 0.5) {
                char desc[128];
                snprintf(desc, sizeof(desc), "Correlation change with feature %zu: %.2f",
                         j, corr_change);
                TemporalAnomaly* anomaly = create_anomaly(correlation, corr_change,
                                                           feature_index, desc);
                if (anomaly) {
                    add_anomaly(tracker, anomaly);
                }
            }
        }
    }
}

// Compute correlation score for importance
static double compute_correlation_score(FeatureHistoryTracker* tracker, size_t feature_index) {
    if (!tracker || feature_index >= tracker->num_features) return 0.0;
    if (tracker->num_features < 2) return 0.5;

    size_t len = tracker->history_lengths[feature_index];
    if (len < MIN_PATTERN_LENGTH) return 0.0;

    double* values = tracker->feature_history[feature_index];
    double max_corr = 0.0;

    // Find maximum absolute correlation with other features
    for (size_t j = 0; j < tracker->num_features; j++) {
        if (j == feature_index) continue;
        if (tracker->history_lengths[j] < len) continue;

        double* other = tracker->feature_history[j];
        double mean1 = compute_mean(values, len);
        double mean2 = compute_mean(other, len);
        double cov = 0.0, var1 = 0.0, var2 = 0.0;

        for (size_t i = 0; i < len; i++) {
            double d1 = values[i] - mean1;
            double d2 = other[i] - mean2;
            cov += d1 * d2;
            var1 += d1 * d1;
            var2 += d2 * d2;
        }

        if (var1 > MIN_VARIANCE && var2 > MIN_VARIANCE) {
            double corr = fabs(cov / sqrt(var1 * var2));
            if (corr > max_corr) max_corr = corr;
        }
    }

    return max_corr;
}

// Compute predictive power
static double compute_predictive_power(FeatureHistoryTracker* tracker, size_t feature_index) {
    if (!tracker || feature_index >= tracker->num_features) return 0.0;

    size_t len = tracker->history_lengths[feature_index];
    if (len < MIN_PATTERN_LENGTH * 2) return 0.0;

    double* values = tracker->feature_history[feature_index];

    // Use lag-1 autocorrelation as proxy for predictive power
    double autocorr = compute_autocorrelation(values, len, 1);
    return fabs(autocorr);
}

// Compute stability score
static double compute_stability_score(FeatureHistoryTracker* tracker, size_t feature_index) {
    if (!tracker || feature_index >= tracker->num_features) return 0.0;

    size_t len = tracker->history_lengths[feature_index];
    if (len < MIN_PATTERN_LENGTH) return 0.0;

    double* values = tracker->feature_history[feature_index];
    double mean = compute_mean(values, len);
    double std_dev = compute_std_dev(values, len, mean);

    if (fabs(mean) < MIN_VARIANCE) return 0.0;

    // Coefficient of variation (inverted so higher = more stable)
    double cv = std_dev / fabs(mean);
    return 1.0 / (1.0 + cv);
}

// Compute overall importance
static double compute_overall_importance(const FeatureImportance* importance) {
    if (!importance) return 0.0;

    // Weighted combination of importance factors
    return 0.4 * importance->correlation_score +
           0.3 * importance->predictive_power +
           0.3 * importance->stability_score;
}

// Filter patterns by feature index
static FeaturePattern** filter_patterns(FeatureHistoryTracker* tracker, size_t feature_index) {
    if (!tracker) return NULL;

    size_t count = count_feature_patterns(tracker, feature_index);
    if (count == 0) return NULL;

    FeaturePattern** filtered = calloc(count, sizeof(FeaturePattern*));
    if (!filtered) return NULL;

    size_t idx = 0;
    for (size_t i = 0; i < tracker->num_patterns && idx < count; i++) {
        if (tracker->detected_patterns[i] &&
            tracker->detected_patterns[i]->feature_index == feature_index) {
            filtered[idx++] = tracker->detected_patterns[i];
        }
    }

    return filtered;
}

// Count patterns for a specific feature
static size_t count_feature_patterns(FeatureHistoryTracker* tracker, size_t feature_index) {
    if (!tracker) return 0;

    size_t count = 0;
    for (size_t i = 0; i < tracker->num_patterns; i++) {
        if (tracker->detected_patterns[i] &&
            tracker->detected_patterns[i]->feature_index == feature_index) {
            count++;
        }
    }
    return count;
}

// Cleanup feature pattern
static void cleanup_feature_pattern(FeaturePattern* pattern) {
    if (!pattern) return;
    free(pattern->values);
    free(pattern);
}

// Cleanup temporal anomaly
static void cleanup_temporal_anomaly(TemporalAnomaly* anomaly) {
    if (!anomaly) return;
    free(anomaly->description);
    free(anomaly);
}
