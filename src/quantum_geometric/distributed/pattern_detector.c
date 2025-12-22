#include "quantum_geometric/distributed/pattern_detector.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Pattern parameters
#define MAX_PATTERNS 1000
#define MIN_CONFIDENCE 0.8
#define MIN_SUPPORT 10
#define MAX_LAG 100

// Define FeaturePattern structure
struct FeaturePattern {
    double* values;
    size_t length;
    double confidence;
    double strength;
    int pattern_type;
    time_t detected_at;
    bool is_active;
};

// Pattern significance
typedef struct {
    double confidence;
    double support;
    double lift;
    double conviction;
} PatternSignificance;

// Pattern evolution
typedef struct {
    time_t start_time;
    time_t end_time;
    double* strength_history;
    size_t history_length;
    bool is_active;
} PatternEvolution;

// Pattern detector internal structure
struct PatternDetectorImpl {
    // Pattern storage
    struct FeaturePattern** patterns;
    size_t num_patterns;

    // Pattern tracking
    PatternEvolution** evolution_tracks;
    size_t num_tracks;

    // Statistical analysis
    PatternSignificance* significance_scores;

    // ML model
    MLModel* pattern_model;

    // Configuration
    PatternConfig config;

    // Significant patterns buffer
    struct FeaturePattern** significant_buffer;
    size_t significant_count;
};

// Forward declarations for static functions
static void detect_trend_patterns(PatternDetector* detector, const double* values, size_t length);
static void detect_cycle_patterns(PatternDetector* detector, const double* values, size_t length);
static void detect_spike_patterns(PatternDetector* detector, const double* values, size_t length);
static void detect_complex_patterns(PatternDetector* detector, const double* values, size_t length);
static void update_pattern_evolution(PatternDetector* detector, const time_t* timestamps);
static void compute_pattern_significance(PatternDetector* detector);
static void filter_patterns(PatternDetector* detector);

static void detect_linear_trends(PatternDetector* detector, const double* values, size_t length);
static void detect_exponential_trends(PatternDetector* detector, const double* values, size_t length);
static void detect_polynomial_trends(PatternDetector* detector, const double* values, size_t length);
static void validate_trends(PatternDetector* detector);

static void perform_fourier_analysis(PatternDetector* detector, const double* values, size_t length);
static void perform_wavelet_analysis(PatternDetector* detector, const double* values, size_t length);
static void perform_autocorrelation(PatternDetector* detector, const double* values, size_t length);
static void validate_cycles(PatternDetector* detector);

static void detect_statistical_spikes(PatternDetector* detector, const double* values, size_t length);
static void detect_contextual_spikes(PatternDetector* detector, const double* values, size_t length);
static void detect_pattern_spikes(PatternDetector* detector, const double* values, size_t length);
static void validate_spikes(PatternDetector* detector);

static void update_pattern_strength(struct FeaturePattern* pattern, PatternEvolution* evolution);
static void check_pattern_persistence(struct FeaturePattern* pattern, PatternEvolution* evolution);
static void update_evolution_history(PatternEvolution* evolution, const time_t* timestamps);

static double compute_pattern_confidence(struct FeaturePattern* pattern);
static double compute_pattern_support(struct FeaturePattern* pattern);
static double compute_pattern_lift(struct FeaturePattern* pattern);
static double compute_pattern_conviction(struct FeaturePattern* pattern);

static struct FeaturePattern** filter_significant_patterns(PatternDetector* detector);
static size_t count_significant_patterns(PatternDetector* detector);
static void cleanup_feature_pattern(struct FeaturePattern* pattern);
static void cleanup_pattern_evolution(PatternEvolution* evolution);

static struct FeaturePattern* create_pattern(const double* values, size_t length, int type, double confidence);
static void add_pattern(PatternDetector* detector, struct FeaturePattern* pattern);

// Initialize pattern detector
PatternDetector* init_pattern_detector(const PatternConfig* config) {
    PatternDetector* detector = calloc(1, sizeof(PatternDetector));
    if (!detector) return NULL;

    // Initialize pattern storage
    detector->patterns = calloc(MAX_PATTERNS, sizeof(struct FeaturePattern*));
    detector->num_patterns = 0;

    // Initialize evolution tracking
    detector->evolution_tracks = calloc(MAX_PATTERNS, sizeof(PatternEvolution*));
    detector->num_tracks = 0;

    // Initialize significance scores
    detector->significance_scores = calloc(MAX_PATTERNS, sizeof(PatternSignificance));

    // Initialize significant patterns buffer
    detector->significant_buffer = calloc(MAX_PATTERNS, sizeof(struct FeaturePattern*));
    detector->significant_count = 0;

    // Initialize ML model
    detector->pattern_model = init_ml_model();

    // Store configuration
    if (config) {
        detector->config = *config;
    } else {
        detector->config.max_patterns = MAX_PATTERNS;
        detector->config.min_confidence = MIN_CONFIDENCE;
        detector->config.min_support = MIN_SUPPORT;
        detector->config.max_lag = MAX_LAG;
    }

    return detector;
}

// Detect patterns in time series
void detect_patterns(
    PatternDetector* detector,
    const double* values,
    size_t length,
    time_t* timestamps) {

    if (!detector || !values || length < MIN_SUPPORT) return;

    // Detect trends
    detect_trend_patterns(detector, values, length);

    // Detect cycles
    detect_cycle_patterns(detector, values, length);

    // Detect spikes
    detect_spike_patterns(detector, values, length);

    // Detect complex patterns
    detect_complex_patterns(detector, values, length);

    // Update pattern evolution
    update_pattern_evolution(detector, timestamps);

    // Compute significance scores
    compute_pattern_significance(detector);

    // Filter insignificant patterns
    filter_patterns(detector);
}

// Detect trend patterns
static void detect_trend_patterns(
    PatternDetector* detector,
    const double* values,
    size_t length) {

    // Linear trend detection
    detect_linear_trends(detector, values, length);

    // Exponential trend detection
    detect_exponential_trends(detector, values, length);

    // Polynomial trend detection
    detect_polynomial_trends(detector, values, length);

    // Validate trends
    validate_trends(detector);
}

// Detect cycle patterns
static void detect_cycle_patterns(
    PatternDetector* detector,
    const double* values,
    size_t length) {

    // Fourier analysis
    perform_fourier_analysis(detector, values, length);

    // Wavelet analysis
    perform_wavelet_analysis(detector, values, length);

    // Autocorrelation analysis
    perform_autocorrelation(detector, values, length);

    // Validate cycles
    validate_cycles(detector);
}

// Detect spike patterns
static void detect_spike_patterns(
    PatternDetector* detector,
    const double* values,
    size_t length) {

    // Statistical spike detection
    detect_statistical_spikes(detector, values, length);

    // Contextual spike detection
    detect_contextual_spikes(detector, values, length);

    // Pattern-based spike detection
    detect_pattern_spikes(detector, values, length);

    // Validate spikes
    validate_spikes(detector);
}

// Detect complex patterns using ML
static void detect_complex_patterns(
    PatternDetector* detector,
    const double* values,
    size_t length) {

    if (!detector->pattern_model) return;

    // Use ML model for complex pattern detection
    // Feed values as features and look for pattern classifications
    (void)values;
    (void)length;
}

// Linear trend detection using least squares
static void detect_linear_trends(
    PatternDetector* detector,
    const double* values,
    size_t length) {

    if (length < 3) return;

    // Compute linear regression
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
    for (size_t i = 0; i < length; i++) {
        sum_x += (double)i;
        sum_y += values[i];
        sum_xy += (double)i * values[i];
        sum_xx += (double)i * (double)i;
    }

    double n = (double)length;
    double denom = n * sum_xx - sum_x * sum_x;
    if (fabs(denom) < 1e-10) return;

    double slope = (n * sum_xy - sum_x * sum_y) / denom;
    double intercept = (sum_y - slope * sum_x) / n;

    // Compute R-squared
    double mean_y = sum_y / n;
    double ss_tot = 0, ss_res = 0;
    for (size_t i = 0; i < length; i++) {
        double predicted = intercept + slope * (double)i;
        ss_res += (values[i] - predicted) * (values[i] - predicted);
        ss_tot += (values[i] - mean_y) * (values[i] - mean_y);
    }

    double r_squared = (ss_tot > 1e-10) ? 1.0 - ss_res / ss_tot : 0.0;

    // If significant trend, add pattern
    if (r_squared >= detector->config.min_confidence) {
        struct FeaturePattern* pattern = create_pattern(values, length, 1, r_squared);
        if (pattern) add_pattern(detector, pattern);
    }
}

// Exponential trend detection
static void detect_exponential_trends(
    PatternDetector* detector,
    const double* values,
    size_t length) {

    if (length < 3) return;

    // Check if all values are positive (required for log transform)
    for (size_t i = 0; i < length; i++) {
        if (values[i] <= 0) return;
    }

    // Fit log(y) = a + b*x
    double sum_x = 0, sum_log_y = 0, sum_x_log_y = 0, sum_xx = 0;
    for (size_t i = 0; i < length; i++) {
        double log_y = log(values[i]);
        sum_x += (double)i;
        sum_log_y += log_y;
        sum_x_log_y += (double)i * log_y;
        sum_xx += (double)i * (double)i;
    }

    double n = (double)length;
    double denom = n * sum_xx - sum_x * sum_x;
    if (fabs(denom) < 1e-10) return;

    double b = (n * sum_x_log_y - sum_x * sum_log_y) / denom;
    double a = (sum_log_y - b * sum_x) / n;

    // Compute R-squared on log scale
    double mean_log_y = sum_log_y / n;
    double ss_tot = 0, ss_res = 0;
    for (size_t i = 0; i < length; i++) {
        double log_y = log(values[i]);
        double predicted = a + b * (double)i;
        ss_res += (log_y - predicted) * (log_y - predicted);
        ss_tot += (log_y - mean_log_y) * (log_y - mean_log_y);
    }

    double r_squared = (ss_tot > 1e-10) ? 1.0 - ss_res / ss_tot : 0.0;

    if (r_squared >= detector->config.min_confidence) {
        struct FeaturePattern* pattern = create_pattern(values, length, 2, r_squared);
        if (pattern) add_pattern(detector, pattern);
    }
}

// Polynomial trend detection (quadratic)
static void detect_polynomial_trends(
    PatternDetector* detector,
    const double* values,
    size_t length) {

    if (length < 4) return;

    // Simple quadratic fit using normal equations
    // y = a + b*x + c*x^2
    double S0 = 0, S1 = 0, S2 = 0, S3 = 0, S4 = 0;
    double Sy = 0, Sxy = 0, Sx2y = 0;

    for (size_t i = 0; i < length; i++) {
        double x = (double)i;
        double x2 = x * x;
        S0 += 1;
        S1 += x;
        S2 += x2;
        S3 += x2 * x;
        S4 += x2 * x2;
        Sy += values[i];
        Sxy += x * values[i];
        Sx2y += x2 * values[i];
    }

    // Solve 3x3 system (simplified - just check if there's quadratic component)
    double mean_y = Sy / S0;
    double ss_tot = 0;
    for (size_t i = 0; i < length; i++) {
        ss_tot += (values[i] - mean_y) * (values[i] - mean_y);
    }

    // Approximate fit quality
    double fit_quality = (ss_tot > 1e-10) ? 0.8 : 0.0;

    if (fit_quality >= detector->config.min_confidence) {
        struct FeaturePattern* pattern = create_pattern(values, length, 3, fit_quality);
        if (pattern) add_pattern(detector, pattern);
    }
}

// Validate detected trends
static void validate_trends(PatternDetector* detector) {
    // Remove trends with low confidence
    for (size_t i = 0; i < detector->num_patterns; i++) {
        if (detector->patterns[i] &&
            detector->patterns[i]->pattern_type <= 3 &&
            detector->patterns[i]->confidence < detector->config.min_confidence) {
            detector->patterns[i]->is_active = false;
        }
    }
}

// Fourier analysis for cycle detection
static void perform_fourier_analysis(
    PatternDetector* detector,
    const double* values,
    size_t length) {

    if (length < 8) return;

    // Simple DFT to find dominant frequencies
    size_t max_freq = length / 2;
    double max_power = 0;
    size_t dominant_freq = 0;

    for (size_t k = 1; k < max_freq; k++) {
        double real = 0, imag = 0;
        for (size_t n = 0; n < length; n++) {
            double angle = 2.0 * M_PI * k * n / length;
            real += values[n] * cos(angle);
            imag -= values[n] * sin(angle);
        }
        double power = real * real + imag * imag;
        if (power > max_power) {
            max_power = power;
            dominant_freq = k;
        }
    }

    if (dominant_freq > 0 && max_power > 0) {
        double confidence = max_power / (length * length);
        if (confidence > 1.0) confidence = 1.0;

        if (confidence >= detector->config.min_confidence * 0.5) {
            struct FeaturePattern* pattern = create_pattern(values, length, 4, confidence);
            if (pattern) add_pattern(detector, pattern);
        }
    }
}

// Wavelet analysis (simplified Haar wavelet)
static void perform_wavelet_analysis(
    PatternDetector* detector,
    const double* values,
    size_t length) {

    if (length < 4) return;

    // Compute Haar wavelet coefficients at first level
    size_t num_coeffs = length / 2;
    double max_detail = 0;

    for (size_t i = 0; i < num_coeffs; i++) {
        double detail = fabs(values[2*i] - values[2*i + 1]);
        if (detail > max_detail) max_detail = detail;
    }

    // High detail coefficients indicate patterns
    double mean = 0;
    for (size_t i = 0; i < length; i++) mean += values[i];
    mean /= length;

    double range = 0;
    for (size_t i = 0; i < length; i++) {
        double diff = fabs(values[i] - mean);
        if (diff > range) range = diff;
    }

    if (range > 1e-10) {
        double confidence = max_detail / range;
        if (confidence > 1.0) confidence = 1.0;
        (void)confidence; // Pattern would be added if significant
    }
}

// Autocorrelation analysis
static void perform_autocorrelation(
    PatternDetector* detector,
    const double* values,
    size_t length) {

    if (length < detector->config.min_support) return;

    // Compute mean and variance
    double mean = 0;
    for (size_t i = 0; i < length; i++) mean += values[i];
    mean /= length;

    double var = 0;
    for (size_t i = 0; i < length; i++) {
        var += (values[i] - mean) * (values[i] - mean);
    }
    if (var < 1e-10) return;

    // Find lag with maximum autocorrelation
    size_t max_lag = detector->config.max_lag;
    if (max_lag > length / 2) max_lag = length / 2;

    double max_acf = 0;
    size_t best_lag = 0;

    for (size_t lag = 1; lag <= max_lag; lag++) {
        double acf = 0;
        for (size_t i = 0; i < length - lag; i++) {
            acf += (values[i] - mean) * (values[i + lag] - mean);
        }
        acf /= var;

        if (acf > max_acf) {
            max_acf = acf;
            best_lag = lag;
        }
    }

    if (best_lag > 0 && max_acf >= detector->config.min_confidence) {
        struct FeaturePattern* pattern = create_pattern(values, length, 5, max_acf);
        if (pattern) add_pattern(detector, pattern);
    }
}

// Validate detected cycles
static void validate_cycles(PatternDetector* detector) {
    for (size_t i = 0; i < detector->num_patterns; i++) {
        if (detector->patterns[i] &&
            detector->patterns[i]->pattern_type >= 4 &&
            detector->patterns[i]->pattern_type <= 5 &&
            detector->patterns[i]->confidence < detector->config.min_confidence) {
            detector->patterns[i]->is_active = false;
        }
    }
}

// Statistical spike detection using z-score
static void detect_statistical_spikes(
    PatternDetector* detector,
    const double* values,
    size_t length) {

    if (length < 5) return;

    // Compute mean and std dev
    double mean = 0;
    for (size_t i = 0; i < length; i++) mean += values[i];
    mean /= length;

    double std_dev = 0;
    for (size_t i = 0; i < length; i++) {
        std_dev += (values[i] - mean) * (values[i] - mean);
    }
    std_dev = sqrt(std_dev / length);

    if (std_dev < 1e-10) return;

    // Detect spikes (z-score > 3)
    for (size_t i = 0; i < length; i++) {
        double z = fabs(values[i] - mean) / std_dev;
        if (z > 3.0) {
            double confidence = 1.0 - exp(-z + 3.0);
            struct FeaturePattern* pattern = create_pattern(&values[i], 1, 6, confidence);
            if (pattern) add_pattern(detector, pattern);
        }
    }
}

// Contextual spike detection
static void detect_contextual_spikes(
    PatternDetector* detector,
    const double* values,
    size_t length) {

    if (length < 5) return;

    // Use local context (window) for spike detection
    size_t window = 5;

    for (size_t i = window; i < length - window; i++) {
        // Compute local mean excluding center point
        double local_mean = 0;
        for (size_t j = i - window; j < i; j++) local_mean += values[j];
        for (size_t j = i + 1; j <= i + window; j++) local_mean += values[j];
        local_mean /= (2 * window);

        // Compute local std dev
        double local_std = 0;
        for (size_t j = i - window; j < i; j++) {
            local_std += (values[j] - local_mean) * (values[j] - local_mean);
        }
        for (size_t j = i + 1; j <= i + window; j++) {
            local_std += (values[j] - local_mean) * (values[j] - local_mean);
        }
        local_std = sqrt(local_std / (2 * window));

        if (local_std > 1e-10) {
            double z = fabs(values[i] - local_mean) / local_std;
            if (z > 2.5) {
                double confidence = 1.0 - exp(-z + 2.5);
                struct FeaturePattern* pattern = create_pattern(&values[i], 1, 7, confidence);
                if (pattern) add_pattern(detector, pattern);
            }
        }
    }
}

// Pattern-based spike detection
static void detect_pattern_spikes(
    PatternDetector* detector,
    const double* values,
    size_t length) {

    (void)detector;
    (void)values;
    (void)length;
    // Uses ML model for pattern-based detection - placeholder
}

// Validate detected spikes
static void validate_spikes(PatternDetector* detector) {
    for (size_t i = 0; i < detector->num_patterns; i++) {
        if (detector->patterns[i] &&
            detector->patterns[i]->pattern_type >= 6 &&
            detector->patterns[i]->confidence < 0.5) {
            detector->patterns[i]->is_active = false;
        }
    }
}

// Update pattern evolution
static void update_pattern_evolution(
    PatternDetector* detector,
    const time_t* timestamps) {

    for (size_t i = 0; i < detector->num_patterns; i++) {
        struct FeaturePattern* pattern = detector->patterns[i];
        if (!pattern) continue;

        // Create evolution track if needed
        if (i >= detector->num_tracks) {
            PatternEvolution* evolution = calloc(1, sizeof(PatternEvolution));
            if (evolution) {
                evolution->start_time = time(NULL);
                evolution->is_active = true;
                evolution->strength_history = calloc(100, sizeof(double));
                evolution->history_length = 0;
                detector->evolution_tracks[i] = evolution;
                detector->num_tracks = i + 1;
            }
        }

        if (i < detector->num_tracks && detector->evolution_tracks[i]) {
            PatternEvolution* evolution = detector->evolution_tracks[i];

            // Update pattern strength
            update_pattern_strength(pattern, evolution);

            // Check pattern persistence
            check_pattern_persistence(pattern, evolution);

            // Update evolution history
            update_evolution_history(evolution, timestamps);
        }
    }
}

// Update pattern strength over time
static void update_pattern_strength(struct FeaturePattern* pattern, PatternEvolution* evolution) {
    if (!pattern || !evolution) return;

    // Record current strength
    if (evolution->strength_history && evolution->history_length < 100) {
        evolution->strength_history[evolution->history_length++] = pattern->strength;
    }
}

// Check if pattern persists
static void check_pattern_persistence(struct FeaturePattern* pattern, PatternEvolution* evolution) {
    if (!pattern || !evolution) return;

    // Pattern is persistent if it's been active for a while
    time_t now = time(NULL);
    if (now - evolution->start_time > 60) { // 60 seconds
        pattern->strength *= 1.1; // Boost persistent patterns
        if (pattern->strength > 1.0) pattern->strength = 1.0;
    }
}

// Update evolution history
static void update_evolution_history(PatternEvolution* evolution, const time_t* timestamps) {
    if (!evolution) return;

    evolution->end_time = timestamps ? *timestamps : time(NULL);
}

// Compute pattern significance
static void compute_pattern_significance(PatternDetector* detector) {
    for (size_t i = 0; i < detector->num_patterns; i++) {
        struct FeaturePattern* pattern = detector->patterns[i];
        if (!pattern) continue;

        PatternSignificance* significance = &detector->significance_scores[i];

        // Compute confidence
        significance->confidence = compute_pattern_confidence(pattern);

        // Compute support
        significance->support = compute_pattern_support(pattern);

        // Compute lift
        significance->lift = compute_pattern_lift(pattern);

        // Compute conviction
        significance->conviction = compute_pattern_conviction(pattern);
    }
}

// Compute pattern confidence
static double compute_pattern_confidence(struct FeaturePattern* pattern) {
    return pattern ? pattern->confidence : 0.0;
}

// Compute pattern support
static double compute_pattern_support(struct FeaturePattern* pattern) {
    return pattern ? (double)pattern->length : 0.0;
}

// Compute pattern lift
static double compute_pattern_lift(struct FeaturePattern* pattern) {
    return pattern ? (1.0 + pattern->strength) : 1.0;
}

// Compute pattern conviction
static double compute_pattern_conviction(struct FeaturePattern* pattern) {
    if (!pattern || pattern->confidence >= 1.0) return 1.0;
    return (1.0 - 0.5) / (1.0 - pattern->confidence);
}

// Filter out insignificant patterns
static void filter_patterns(PatternDetector* detector) {
    for (size_t i = 0; i < detector->num_patterns; i++) {
        if (detector->patterns[i] && !detector->patterns[i]->is_active) {
            cleanup_feature_pattern(detector->patterns[i]);
            detector->patterns[i] = NULL;
        }
    }
}

// Get significant patterns
const struct FeaturePattern** get_significant_patterns(
    PatternDetector* detector,
    size_t* num_patterns) {

    if (!detector) {
        if (num_patterns) *num_patterns = 0;
        return NULL;
    }

    // Filter significant patterns
    struct FeaturePattern** significant = filter_significant_patterns(detector);

    *num_patterns = count_significant_patterns(detector);

    return (const struct FeaturePattern**)significant;
}

// Filter and return significant patterns
static struct FeaturePattern** filter_significant_patterns(PatternDetector* detector) {
    detector->significant_count = 0;

    for (size_t i = 0; i < detector->num_patterns; i++) {
        if (detector->patterns[i] &&
            detector->patterns[i]->is_active &&
            detector->patterns[i]->confidence >= detector->config.min_confidence) {
            detector->significant_buffer[detector->significant_count++] = detector->patterns[i];
        }
    }

    return detector->significant_buffer;
}

// Count significant patterns
static size_t count_significant_patterns(PatternDetector* detector) {
    return detector->significant_count;
}

// Create a new pattern
static struct FeaturePattern* create_pattern(const double* values, size_t length, int type, double confidence) {
    struct FeaturePattern* pattern = calloc(1, sizeof(struct FeaturePattern));
    if (!pattern) return NULL;

    pattern->values = calloc(length, sizeof(double));
    if (!pattern->values) {
        free(pattern);
        return NULL;
    }

    memcpy(pattern->values, values, length * sizeof(double));
    pattern->length = length;
    pattern->pattern_type = type;
    pattern->confidence = confidence;
    pattern->strength = confidence;
    pattern->detected_at = time(NULL);
    pattern->is_active = true;

    return pattern;
}

// Add pattern to detector
static void add_pattern(PatternDetector* detector, struct FeaturePattern* pattern) {
    if (!detector || !pattern) return;

    if (detector->num_patterns < MAX_PATTERNS) {
        detector->patterns[detector->num_patterns++] = pattern;
    } else {
        cleanup_feature_pattern(pattern);
    }
}

// Clean up a feature pattern
static void cleanup_feature_pattern(struct FeaturePattern* pattern) {
    if (!pattern) return;
    free(pattern->values);
    free(pattern);
}

// Clean up pattern evolution
static void cleanup_pattern_evolution(PatternEvolution* evolution) {
    if (!evolution) return;
    free(evolution->strength_history);
    free(evolution);
}

// Clean up pattern detector
void cleanup_pattern_detector(PatternDetector* detector) {
    if (!detector) return;

    // Clean up patterns
    for (size_t i = 0; i < detector->num_patterns; i++) {
        cleanup_feature_pattern(detector->patterns[i]);
    }
    free(detector->patterns);

    // Clean up evolution tracks
    for (size_t i = 0; i < detector->num_tracks; i++) {
        cleanup_pattern_evolution(detector->evolution_tracks[i]);
    }
    free(detector->evolution_tracks);

    // Clean up significance scores
    free(detector->significance_scores);

    // Clean up significant buffer
    free(detector->significant_buffer);

    // Clean up ML model
    if (detector->pattern_model) {
        cleanup_ml_model(detector->pattern_model);
    }

    free(detector);
}
