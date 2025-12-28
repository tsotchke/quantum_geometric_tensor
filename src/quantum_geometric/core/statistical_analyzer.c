/**
 * @file statistical_analyzer.c
 * @brief Implementation of Statistical Analysis for Quantum Operations
 *
 * Provides rigorous statistical analysis including hypothesis testing,
 * distribution fitting, Bayesian inference, and QEC-specific statistics.
 */

#include "quantum_geometric/core/statistical_analyzer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>
#include <float.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_E
#define M_E 2.71828182845904523536
#endif

// ============================================================================
// Constants
// ============================================================================

#define HASH_TABLE_SIZE 256
#define MAX_SERIES_NAME 128
#define DEFAULT_SERIES_SIZE 10000

// ============================================================================
// Internal Structures
// ============================================================================

// Data series entry
typedef struct series_entry {
    char name[MAX_SERIES_NAME];
    double* data;
    size_t count;
    size_t capacity;
    struct series_entry* next;
} series_entry_t;

// QEC tracking state
typedef struct {
    size_t total_cycles;
    size_t total_syndromes;
    size_t total_corrections;
    size_t logical_errors;
    double* cycle_syndrome_rates;
    size_t cycle_capacity;
} qec_tracker_t;

// Main analyzer structure
struct stat_analyzer {
    stat_analyzer_config_t config;

    // Series hash table
    series_entry_t* series[HASH_TABLE_SIZE];
    size_t series_count;

    // QEC tracking
    qec_tracker_t qec;

    // Thread safety
    pthread_mutex_t mutex;

    // Error handling
    char last_error[256];

    // Random state
    uint64_t rng_state;
};

// Thread-local error storage
static __thread char tls_error[256] = {0};

// ============================================================================
// Internal Helper Functions
// ============================================================================

static void set_error(stat_analyzer_t* analyzer, const char* msg) {
    if (analyzer) {
        strncpy(analyzer->last_error, msg, sizeof(analyzer->last_error) - 1);
        analyzer->last_error[sizeof(analyzer->last_error) - 1] = '\0';
    }
    strncpy(tls_error, msg, sizeof(tls_error) - 1);
    tls_error[sizeof(tls_error) - 1] = '\0';
}

static size_t hash_string(const char* str) {
    size_t hash = 5381;
    int c;
    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash % HASH_TABLE_SIZE;
}

// Simple xorshift64 RNG
static uint64_t xorshift64(uint64_t* state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static double random_uniform(stat_analyzer_t* analyzer) {
    return (double)xorshift64(&analyzer->rng_state) / (double)UINT64_MAX;
}

// Box-Muller transform for normal distribution
static double random_normal(stat_analyzer_t* analyzer, double mean, double std_dev) {
    double u1 = random_uniform(analyzer);
    double u2 = random_uniform(analyzer);

    // Avoid log(0)
    while (u1 == 0.0) u1 = random_uniform(analyzer);

    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return mean + std_dev * z;
}

// Comparison function for qsort
static int compare_doubles(const void* a, const void* b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

// ============================================================================
// Series Management
// ============================================================================

static series_entry_t* find_series(stat_analyzer_t* analyzer, const char* name) {
    size_t idx = hash_string(name);
    series_entry_t* entry = analyzer->series[idx];

    while (entry) {
        if (strcmp(entry->name, name) == 0) {
            return entry;
        }
        entry = entry->next;
    }
    return NULL;
}

static series_entry_t* create_series(stat_analyzer_t* analyzer,
                                      const char* name,
                                      size_t capacity) {
    series_entry_t* entry = calloc(1, sizeof(series_entry_t));
    if (!entry) return NULL;

    strncpy(entry->name, name, MAX_SERIES_NAME - 1);
    entry->capacity = capacity;
    entry->data = calloc(capacity, sizeof(double));

    if (!entry->data) {
        free(entry);
        return NULL;
    }

    size_t idx = hash_string(name);
    entry->next = analyzer->series[idx];
    analyzer->series[idx] = entry;
    analyzer->series_count++;

    return entry;
}

// ============================================================================
// Configuration and Initialization
// ============================================================================

stat_analyzer_config_t stat_analyzer_default_config(void) {
    return (stat_analyzer_config_t){
        .default_confidence = STAT_CONFIDENCE_95,
        .significance_alpha = 0.05,
        .bootstrap_iterations = STAT_BOOTSTRAP_ITERATIONS,
        .use_exact_tests = true,
        .correct_multiple_comparisons = false,
        .histogram_bins = 50,
        .outlier_threshold = 1.5
    };
}

stat_analyzer_t* stat_analyzer_create(void) {
    stat_analyzer_config_t config = stat_analyzer_default_config();
    return stat_analyzer_create_with_config(&config);
}

stat_analyzer_t* stat_analyzer_create_with_config(
    const stat_analyzer_config_t* config) {

    if (!config) {
        set_error(NULL, "NULL config provided");
        return NULL;
    }

    stat_analyzer_t* analyzer = calloc(1, sizeof(stat_analyzer_t));
    if (!analyzer) {
        set_error(NULL, "Failed to allocate analyzer");
        return NULL;
    }

    analyzer->config = *config;
    pthread_mutex_init(&analyzer->mutex, NULL);

    // Initialize RNG with time-based seed
    analyzer->rng_state = (uint64_t)time(NULL) ^ 0xDEADBEEFCAFEBABEULL;

    // Initialize QEC tracker
    analyzer->qec.cycle_capacity = 10000;
    analyzer->qec.cycle_syndrome_rates = calloc(analyzer->qec.cycle_capacity,
                                                  sizeof(double));

    return analyzer;
}

void stat_analyzer_destroy(stat_analyzer_t* analyzer) {
    if (!analyzer) return;

    pthread_mutex_lock(&analyzer->mutex);

    // Free series
    for (size_t i = 0; i < HASH_TABLE_SIZE; i++) {
        series_entry_t* entry = analyzer->series[i];
        while (entry) {
            series_entry_t* next = entry->next;
            free(entry->data);
            free(entry);
            entry = next;
        }
    }

    // Free QEC data
    free(analyzer->qec.cycle_syndrome_rates);

    pthread_mutex_unlock(&analyzer->mutex);
    pthread_mutex_destroy(&analyzer->mutex);

    free(analyzer);
}

bool stat_analyzer_reset(stat_analyzer_t* analyzer) {
    if (!analyzer) return false;

    pthread_mutex_lock(&analyzer->mutex);

    // Clear all series
    for (size_t i = 0; i < HASH_TABLE_SIZE; i++) {
        series_entry_t* entry = analyzer->series[i];
        while (entry) {
            series_entry_t* next = entry->next;
            free(entry->data);
            free(entry);
            entry = next;
        }
        analyzer->series[i] = NULL;
    }
    analyzer->series_count = 0;

    // Reset QEC
    memset(&analyzer->qec, 0, sizeof(qec_tracker_t));
    analyzer->qec.cycle_capacity = 10000;
    analyzer->qec.cycle_syndrome_rates = calloc(analyzer->qec.cycle_capacity,
                                                  sizeof(double));

    pthread_mutex_unlock(&analyzer->mutex);
    return true;
}

// ============================================================================
// Data Management
// ============================================================================

bool stat_create_series(stat_analyzer_t* analyzer,
                        const char* name,
                        size_t max_samples) {
    if (!analyzer || !name) return false;

    pthread_mutex_lock(&analyzer->mutex);

    if (find_series(analyzer, name)) {
        pthread_mutex_unlock(&analyzer->mutex);
        set_error(analyzer, "Series already exists");
        return false;
    }

    series_entry_t* entry = create_series(analyzer, name, max_samples);

    pthread_mutex_unlock(&analyzer->mutex);
    return entry != NULL;
}

bool stat_add_sample(stat_analyzer_t* analyzer,
                     const char* series_name,
                     double value) {
    if (!analyzer || !series_name) return false;

    pthread_mutex_lock(&analyzer->mutex);

    series_entry_t* series = find_series(analyzer, series_name);
    if (!series) {
        // Auto-create series
        series = create_series(analyzer, series_name, DEFAULT_SERIES_SIZE);
        if (!series) {
            pthread_mutex_unlock(&analyzer->mutex);
            return false;
        }
    }

    if (series->count >= series->capacity) {
        // Grow capacity
        size_t new_capacity = series->capacity * 2;
        double* new_data = realloc(series->data, new_capacity * sizeof(double));
        if (!new_data) {
            pthread_mutex_unlock(&analyzer->mutex);
            return false;
        }
        series->data = new_data;
        series->capacity = new_capacity;
    }

    series->data[series->count++] = value;

    pthread_mutex_unlock(&analyzer->mutex);
    return true;
}

bool stat_add_samples(stat_analyzer_t* analyzer,
                      const char* series_name,
                      const double* values,
                      size_t count) {
    if (!analyzer || !series_name || !values || count == 0) return false;

    for (size_t i = 0; i < count; i++) {
        if (!stat_add_sample(analyzer, series_name, values[i])) {
            return false;
        }
    }
    return true;
}

bool stat_get_samples(stat_analyzer_t* analyzer,
                      const char* series_name,
                      double** values,
                      size_t* count) {
    if (!analyzer || !series_name || !values || !count) return false;

    pthread_mutex_lock(&analyzer->mutex);

    series_entry_t* series = find_series(analyzer, series_name);
    if (!series) {
        pthread_mutex_unlock(&analyzer->mutex);
        set_error(analyzer, "Series not found");
        return false;
    }

    *count = series->count;
    if (*count == 0) {
        *values = NULL;
        pthread_mutex_unlock(&analyzer->mutex);
        return true;
    }

    *values = malloc(*count * sizeof(double));
    if (!*values) {
        pthread_mutex_unlock(&analyzer->mutex);
        return false;
    }

    memcpy(*values, series->data, *count * sizeof(double));

    pthread_mutex_unlock(&analyzer->mutex);
    return true;
}

bool stat_clear_series(stat_analyzer_t* analyzer,
                       const char* series_name) {
    if (!analyzer || !series_name) return false;

    pthread_mutex_lock(&analyzer->mutex);

    series_entry_t* series = find_series(analyzer, series_name);
    if (series) {
        series->count = 0;
    }

    pthread_mutex_unlock(&analyzer->mutex);
    return series != NULL;
}

// ============================================================================
// Descriptive Statistics
// ============================================================================

bool stat_descriptive_from_data(const double* data,
                                size_t count,
                                stat_descriptive_t* stats) {
    if (!data || !stats || count == 0) return false;

    memset(stats, 0, sizeof(stat_descriptive_t));
    stats->count = count;

    // First pass: sum, min, max
    stats->min = DBL_MAX;
    stats->max = -DBL_MAX;
    double log_sum = 0;
    double reciprocal_sum = 0;
    size_t valid_log = 0;
    size_t valid_recip = 0;

    for (size_t i = 0; i < count; i++) {
        stats->sum += data[i];
        if (data[i] < stats->min) stats->min = data[i];
        if (data[i] > stats->max) stats->max = data[i];

        if (data[i] > 0) {
            log_sum += log(data[i]);
            valid_log++;
        }
        if (data[i] != 0) {
            reciprocal_sum += 1.0 / data[i];
            valid_recip++;
        }
    }

    stats->mean = stats->sum / count;
    stats->range = stats->max - stats->min;

    // Geometric and harmonic means
    if (valid_log == count) {
        stats->geometric_mean = exp(log_sum / count);
    }
    if (valid_recip == count && reciprocal_sum != 0) {
        stats->harmonic_mean = count / reciprocal_sum;
    }

    // Second pass: variance, skewness, kurtosis
    double m2 = 0, m3 = 0, m4 = 0;
    for (size_t i = 0; i < count; i++) {
        double diff = data[i] - stats->mean;
        m2 += diff * diff;
        m3 += diff * diff * diff;
        m4 += diff * diff * diff * diff;
    }

    if (count > 1) {
        stats->variance = m2 / (count - 1);  // Sample variance
        stats->std_dev = sqrt(stats->variance);
        stats->std_error = stats->std_dev / sqrt((double)count);
    }

    if (stats->std_dev > 0 && count > 2) {
        double n = (double)count;
        stats->skewness = (m3 / count) / pow(stats->std_dev, 3) *
                          (n * n) / ((n - 1) * (n - 2));
    }

    if (stats->variance > 0 && count > 3) {
        double n = (double)count;
        double excess_kurtosis = ((m4 / count) / (stats->variance * stats->variance)) - 3.0;
        // Bias correction
        stats->kurtosis = ((n - 1) / ((n - 2) * (n - 3))) *
                          ((n + 1) * excess_kurtosis + 6);
    }

    // Quantiles (need sorted data)
    double* sorted = malloc(count * sizeof(double));
    if (sorted) {
        memcpy(sorted, data, count * sizeof(double));
        qsort(sorted, count, sizeof(double), compare_doubles);

        stats->median = stat_percentile(sorted, count, 50.0);
        stats->q1 = stat_percentile(sorted, count, 25.0);
        stats->q3 = stat_percentile(sorted, count, 75.0);
        stats->iqr = stats->q3 - stats->q1;

        // Median absolute deviation
        double* abs_devs = malloc(count * sizeof(double));
        if (abs_devs) {
            for (size_t i = 0; i < count; i++) {
                abs_devs[i] = fabs(data[i] - stats->median);
            }
            qsort(abs_devs, count, sizeof(double), compare_doubles);
            stats->mad = stat_percentile(abs_devs, count, 50.0);
            free(abs_devs);
        }

        free(sorted);
    }

    return true;
}

bool stat_calculate_descriptive(stat_analyzer_t* analyzer,
                                const char* series_name,
                                stat_descriptive_t* stats) {
    if (!analyzer || !series_name || !stats) return false;

    pthread_mutex_lock(&analyzer->mutex);

    series_entry_t* series = find_series(analyzer, series_name);
    if (!series || series->count == 0) {
        pthread_mutex_unlock(&analyzer->mutex);
        set_error(analyzer, "Series not found or empty");
        return false;
    }

    bool result = stat_descriptive_from_data(series->data, series->count, stats);

    pthread_mutex_unlock(&analyzer->mutex);
    return result;
}

double stat_percentile(const double* sorted_data, size_t count, double percentile) {
    if (!sorted_data || count == 0 || percentile < 0 || percentile > 100) {
        return NAN;
    }

    if (count == 1) return sorted_data[0];

    double rank = (percentile / 100.0) * (count - 1);
    size_t lower = (size_t)floor(rank);
    size_t upper = (size_t)ceil(rank);

    if (lower == upper || upper >= count) {
        return sorted_data[lower];
    }

    double frac = rank - lower;
    return sorted_data[lower] * (1 - frac) + sorted_data[upper] * frac;
}

bool stat_detect_outliers(const double* data,
                          size_t count,
                          double iqr_multiplier,
                          size_t** outlier_indices,
                          size_t* outlier_count) {
    if (!data || !outlier_indices || !outlier_count || count < 4) {
        return false;
    }

    // Sort data to get quartiles
    double* sorted = malloc(count * sizeof(double));
    if (!sorted) return false;
    memcpy(sorted, data, count * sizeof(double));
    qsort(sorted, count, sizeof(double), compare_doubles);

    double q1 = stat_percentile(sorted, count, 25.0);
    double q3 = stat_percentile(sorted, count, 75.0);
    double iqr = q3 - q1;
    double lower_fence = q1 - iqr_multiplier * iqr;
    double upper_fence = q3 + iqr_multiplier * iqr;

    free(sorted);

    // Count outliers first
    size_t num_outliers = 0;
    for (size_t i = 0; i < count; i++) {
        if (data[i] < lower_fence || data[i] > upper_fence) {
            num_outliers++;
        }
    }

    if (num_outliers == 0) {
        *outlier_indices = NULL;
        *outlier_count = 0;
        return true;
    }

    *outlier_indices = malloc(num_outliers * sizeof(size_t));
    if (!*outlier_indices) return false;

    size_t idx = 0;
    for (size_t i = 0; i < count; i++) {
        if (data[i] < lower_fence || data[i] > upper_fence) {
            (*outlier_indices)[idx++] = i;
        }
    }

    *outlier_count = num_outliers;
    return true;
}

// ============================================================================
// Confidence Intervals
// ============================================================================

double stat_confidence_value(stat_confidence_level_t level) {
    switch (level) {
        case STAT_CONFIDENCE_90: return 0.90;
        case STAT_CONFIDENCE_95: return 0.95;
        case STAT_CONFIDENCE_99: return 0.99;
        case STAT_CONFIDENCE_999: return 0.999;
        default: return 0.95;
    }
}

double stat_z_critical(double alpha, bool two_tailed) {
    // Standard normal critical values (approximate)
    double effective_alpha = two_tailed ? alpha / 2 : alpha;

    // Common values
    if (fabs(effective_alpha - 0.025) < 0.001) return 1.96;
    if (fabs(effective_alpha - 0.05) < 0.001) return 1.645;
    if (fabs(effective_alpha - 0.005) < 0.001) return 2.576;
    if (fabs(effective_alpha - 0.0005) < 0.001) return 3.291;

    // Approximation using rational function
    double p = effective_alpha;
    if (p > 0.5) p = 1.0 - p;

    double t = sqrt(-2.0 * log(p));
    double c0 = 2.515517, c1 = 0.802853, c2 = 0.010328;
    double d1 = 1.432788, d2 = 0.189269, d3 = 0.001308;

    double z = t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t);

    return z;
}

double stat_t_critical(double alpha, size_t df, bool two_tailed) {
    // For large df, use z approximation
    if (df > 120) {
        return stat_z_critical(alpha, two_tailed);
    }

    // Lookup table for common values (df: 1-30, 40, 60, 120)
    // Alpha = 0.05 two-tailed
    static const double t_05_2tail[] = {
        12.706, 4.303, 3.182, 2.776, 2.571, // df 1-5
        2.447, 2.365, 2.306, 2.262, 2.228,   // df 6-10
        2.201, 2.179, 2.160, 2.145, 2.131,   // df 11-15
        2.120, 2.110, 2.101, 2.093, 2.086,   // df 16-20
        2.080, 2.074, 2.069, 2.064, 2.060,   // df 21-25
        2.056, 2.052, 2.048, 2.045, 2.042    // df 26-30
    };

    double effective_alpha = two_tailed ? alpha : alpha * 2;

    if (df <= 30 && fabs(effective_alpha - 0.05) < 0.001) {
        return t_05_2tail[df - 1];
    }

    // General approximation
    double z = stat_z_critical(alpha, two_tailed);
    double g1 = (z * z * z + z) / 4;
    double g2 = (5 * z * z * z * z * z + 16 * z * z * z + 3 * z) / 96;

    return z + g1 / df + g2 / (df * df);
}

bool stat_confidence_interval_mean(stat_analyzer_t* analyzer,
                                   const char* series_name,
                                   stat_confidence_level_t level,
                                   stat_confidence_interval_t* ci) {
    if (!analyzer || !series_name || !ci) return false;

    stat_descriptive_t stats;
    if (!stat_calculate_descriptive(analyzer, series_name, &stats)) {
        return false;
    }

    if (stats.count < 2) {
        set_error(analyzer, "Need at least 2 samples for CI");
        return false;
    }

    double conf_level = stat_confidence_value(level);
    double alpha = 1.0 - conf_level;
    double t_crit = stat_t_critical(alpha, stats.count - 1, true);

    ci->point_estimate = stats.mean;
    ci->margin_of_error = t_crit * stats.std_error;
    ci->lower_bound = stats.mean - ci->margin_of_error;
    ci->upper_bound = stats.mean + ci->margin_of_error;
    ci->confidence_level = conf_level;
    ci->level = level;

    return true;
}

bool stat_confidence_interval_proportion(size_t successes,
                                          size_t total,
                                          stat_confidence_level_t level,
                                          stat_confidence_interval_t* ci) {
    if (!ci || total == 0) return false;

    double p = (double)successes / (double)total;
    double conf_level = stat_confidence_value(level);
    double alpha = 1.0 - conf_level;
    double z = stat_z_critical(alpha, true);

    // Wilson score interval (better for small samples)
    double n = (double)total;
    double z2 = z * z;

    double denom = 1 + z2 / n;
    double center = (p + z2 / (2 * n)) / denom;
    double width = z * sqrt((p * (1 - p) + z2 / (4 * n)) / n) / denom;

    ci->point_estimate = p;
    ci->lower_bound = fmax(0.0, center - width);
    ci->upper_bound = fmin(1.0, center + width);
    ci->margin_of_error = width;
    ci->confidence_level = conf_level;
    ci->level = level;

    return true;
}

bool stat_confidence_interval_bootstrap(const double* data,
                                        size_t count,
                                        double (*statistic)(const double*, size_t),
                                        stat_confidence_level_t level,
                                        size_t iterations,
                                        stat_confidence_interval_t* ci) {
    if (!data || !statistic || !ci || count < 2) return false;

    // Create temporary analyzer for RNG
    stat_analyzer_t* temp = stat_analyzer_create();
    if (!temp) return false;

    double* bootstrap_stats = malloc(iterations * sizeof(double));
    double* sample = malloc(count * sizeof(double));

    if (!bootstrap_stats || !sample) {
        free(bootstrap_stats);
        free(sample);
        stat_analyzer_destroy(temp);
        return false;
    }

    // Generate bootstrap samples
    for (size_t i = 0; i < iterations; i++) {
        // Resample with replacement
        for (size_t j = 0; j < count; j++) {
            size_t idx = (size_t)(random_uniform(temp) * count);
            if (idx >= count) idx = count - 1;
            sample[j] = data[idx];
        }
        bootstrap_stats[i] = statistic(sample, count);
    }

    // Sort bootstrap statistics
    qsort(bootstrap_stats, iterations, sizeof(double), compare_doubles);

    // Calculate percentile CI
    double alpha = 1.0 - stat_confidence_value(level);
    size_t lower_idx = (size_t)(alpha / 2 * iterations);
    size_t upper_idx = (size_t)((1.0 - alpha / 2) * iterations);

    if (upper_idx >= iterations) upper_idx = iterations - 1;

    ci->point_estimate = statistic(data, count);
    ci->lower_bound = bootstrap_stats[lower_idx];
    ci->upper_bound = bootstrap_stats[upper_idx];
    ci->margin_of_error = (ci->upper_bound - ci->lower_bound) / 2;
    ci->confidence_level = stat_confidence_value(level);
    ci->level = level;

    free(bootstrap_stats);
    free(sample);
    stat_analyzer_destroy(temp);

    return true;
}

// ============================================================================
// Correlation Analysis
// ============================================================================

bool stat_correlation_from_data(const double* x,
                                const double* y,
                                size_t count,
                                stat_correlation_type_t type,
                                stat_correlation_result_t* result) {
    if (!x || !y || !result || count < 3) return false;

    memset(result, 0, sizeof(stat_correlation_result_t));
    result->type = type;
    result->sample_size = count;

    // Calculate means
    double mean_x = 0, mean_y = 0;
    for (size_t i = 0; i < count; i++) {
        mean_x += x[i];
        mean_y += y[i];
    }
    mean_x /= count;
    mean_y /= count;

    if (type == STAT_CORR_PEARSON || type == STAT_CORR_PARTIAL) {
        // Pearson correlation
        double sum_xy = 0, sum_xx = 0, sum_yy = 0;
        for (size_t i = 0; i < count; i++) {
            double dx = x[i] - mean_x;
            double dy = y[i] - mean_y;
            sum_xy += dx * dy;
            sum_xx += dx * dx;
            sum_yy += dy * dy;
        }

        if (sum_xx > 0 && sum_yy > 0) {
            result->coefficient = sum_xy / sqrt(sum_xx * sum_yy);
        }
    } else if (type == STAT_CORR_SPEARMAN) {
        // Rank-based correlation
        // Create rank arrays
        double* rank_x = malloc(count * sizeof(double));
        double* rank_y = malloc(count * sizeof(double));

        if (!rank_x || !rank_y) {
            free(rank_x);
            free(rank_y);
            return false;
        }

        // Create index arrays for sorting
        size_t* idx_x = malloc(count * sizeof(size_t));
        size_t* idx_y = malloc(count * sizeof(size_t));

        for (size_t i = 0; i < count; i++) {
            idx_x[i] = i;
            idx_y[i] = i;
        }

        // Simple ranking (not handling ties optimally)
        for (size_t i = 0; i < count; i++) {
            rank_x[i] = 1;
            rank_y[i] = 1;
            for (size_t j = 0; j < count; j++) {
                if (x[j] < x[i]) rank_x[i]++;
                if (y[j] < y[i]) rank_y[i]++;
            }
        }

        // Pearson on ranks
        double sum_d2 = 0;
        for (size_t i = 0; i < count; i++) {
            double d = rank_x[i] - rank_y[i];
            sum_d2 += d * d;
        }

        double n = (double)count;
        result->coefficient = 1.0 - (6.0 * sum_d2) / (n * (n * n - 1));

        free(rank_x);
        free(rank_y);
        free(idx_x);
        free(idx_y);
    }

    result->r_squared = result->coefficient * result->coefficient;

    // Calculate p-value using t-test approximation
    if (count > 2 && fabs(result->coefficient) < 1.0) {
        double t = result->coefficient * sqrt((count - 2) / (1 - result->r_squared));
        // Approximate p-value
        double df = count - 2;
        double x_stat = df / (df + t * t);
        // Incomplete beta approximation (simplified)
        result->p_value = 2.0 * (1.0 - 0.5 * (1.0 + erf(fabs(t) / sqrt(2.0))));
    }

    result->is_significant = result->p_value < 0.05;

    // Confidence interval using Fisher z-transform
    double z = 0.5 * log((1 + result->coefficient) / (1 - result->coefficient));
    double se_z = 1.0 / sqrt(count - 3);
    double z_crit = 1.96;  // 95% CI

    double z_lower = z - z_crit * se_z;
    double z_upper = z + z_crit * se_z;

    result->confidence_lower = (exp(2 * z_lower) - 1) / (exp(2 * z_lower) + 1);
    result->confidence_upper = (exp(2 * z_upper) - 1) / (exp(2 * z_upper) + 1);

    return true;
}

bool stat_correlation(stat_analyzer_t* analyzer,
                      const char* series1_name,
                      const char* series2_name,
                      stat_correlation_type_t type,
                      stat_correlation_result_t* result) {
    if (!analyzer || !series1_name || !series2_name || !result) return false;

    double* x = NULL;
    double* y = NULL;
    size_t count1 = 0, count2 = 0;

    if (!stat_get_samples(analyzer, series1_name, &x, &count1) ||
        !stat_get_samples(analyzer, series2_name, &y, &count2)) {
        free(x);
        free(y);
        return false;
    }

    if (count1 != count2) {
        set_error(analyzer, "Series have different lengths");
        free(x);
        free(y);
        return false;
    }

    bool success = stat_correlation_from_data(x, y, count1, type, result);

    free(x);
    free(y);
    return success;
}

bool stat_autocorrelation(const double* data,
                          size_t count,
                          size_t max_lag,
                          double* acf) {
    if (!data || !acf || count < 4 || max_lag == 0) return false;

    // Calculate mean
    double mean = 0;
    for (size_t i = 0; i < count; i++) {
        mean += data[i];
    }
    mean /= count;

    // Calculate variance
    double var = 0;
    for (size_t i = 0; i < count; i++) {
        double diff = data[i] - mean;
        var += diff * diff;
    }
    var /= count;

    if (var == 0) return false;

    // Calculate autocorrelation for each lag
    for (size_t lag = 0; lag <= max_lag && lag < count; lag++) {
        double sum = 0;
        for (size_t i = 0; i < count - lag; i++) {
            sum += (data[i] - mean) * (data[i + lag] - mean);
        }
        acf[lag] = sum / (count * var);
    }

    return true;
}

// ============================================================================
// QEC Statistics
// ============================================================================

bool stat_track_qec_cycle(stat_analyzer_t* analyzer,
                          size_t syndrome_count,
                          size_t correction_count,
                          bool logical_error) {
    if (!analyzer) return false;

    pthread_mutex_lock(&analyzer->mutex);

    qec_tracker_t* qec = &analyzer->qec;

    // Grow if needed
    if (qec->total_cycles >= qec->cycle_capacity) {
        size_t new_cap = qec->cycle_capacity * 2;
        double* new_rates = realloc(qec->cycle_syndrome_rates, new_cap * sizeof(double));
        if (!new_rates) {
            pthread_mutex_unlock(&analyzer->mutex);
            return false;
        }
        qec->cycle_syndrome_rates = new_rates;
        qec->cycle_capacity = new_cap;
    }

    qec->total_cycles++;
    qec->total_syndromes += syndrome_count;
    qec->total_corrections += correction_count;
    if (logical_error) {
        qec->logical_errors++;
    }

    pthread_mutex_unlock(&analyzer->mutex);
    return true;
}

bool stat_get_error_rates(stat_analyzer_t* analyzer,
                          stat_error_rates_t* rates) {
    if (!analyzer || !rates) return false;

    pthread_mutex_lock(&analyzer->mutex);

    memset(rates, 0, sizeof(stat_error_rates_t));

    qec_tracker_t* qec = &analyzer->qec;

    if (qec->total_cycles == 0) {
        pthread_mutex_unlock(&analyzer->mutex);
        return true;
    }

    rates->total_cycles = qec->total_cycles;
    rates->total_errors = qec->logical_errors;
    rates->corrected_errors = qec->total_corrections;

    rates->logical_error_rate = (double)qec->logical_errors / (double)qec->total_cycles;
    rates->syndrome_detection_rate = (double)qec->total_syndromes / (double)qec->total_cycles;

    if (qec->total_syndromes > 0) {
        rates->correction_success_rate = 1.0 - rates->logical_error_rate;
    }

    // Calculate confidence intervals
    stat_confidence_interval_proportion(qec->logical_errors, qec->total_cycles,
                                         STAT_CONFIDENCE_95, &rates->logical_ci);

    pthread_mutex_unlock(&analyzer->mutex);
    return true;
}

double stat_error_suppression(double physical_rate,
                              double logical_rate,
                              size_t code_distance) {
    if (physical_rate <= 0 || logical_rate <= 0 || code_distance == 0) {
        return 0;
    }

    // Lambda = -log(logical_rate) / log(physical_rate^(d+1)/2)
    // Simplified: Lambda ≈ d * log(p_phys) / log(p_log)

    return log(physical_rate) / log(logical_rate) * code_distance;
}

// ============================================================================
// Bayesian Inference
// ============================================================================

bool stat_bayesian_error_rate(double prior_alpha,
                              double prior_beta,
                              size_t successes,
                              size_t failures,
                              stat_bayesian_result_t* result) {
    if (!result) return false;

    memset(result, 0, sizeof(stat_bayesian_result_t));

    // Posterior parameters (Beta-Binomial conjugate)
    double post_alpha = prior_alpha + (double)successes;
    double post_beta = prior_beta + (double)failures;

    // Prior mean and variance (Beta distribution)
    result->prior_mean = prior_alpha / (prior_alpha + prior_beta);
    result->prior_variance = (prior_alpha * prior_beta) /
                             ((prior_alpha + prior_beta) * (prior_alpha + prior_beta) *
                              (prior_alpha + prior_beta + 1));

    // Posterior mean and variance
    result->posterior_mean = post_alpha / (post_alpha + post_beta);
    result->posterior_variance = (post_alpha * post_beta) /
                                  ((post_alpha + post_beta) * (post_alpha + post_beta) *
                                   (post_alpha + post_beta + 1));

    result->sample_size = successes + failures;

    // Credible interval (approximate using normal)
    double std_dev = sqrt(result->posterior_variance);
    result->credible_interval.point_estimate = result->posterior_mean;
    result->credible_interval.lower_bound = fmax(0, result->posterior_mean - 1.96 * std_dev);
    result->credible_interval.upper_bound = fmin(1, result->posterior_mean + 1.96 * std_dev);
    result->credible_interval.confidence_level = 0.95;

    return true;
}

// ============================================================================
// Utility Functions
// ============================================================================

const char* stat_distribution_name(stat_distribution_type_t type) {
    static const char* names[] = {
        "normal", "poisson", "binomial", "exponential", "uniform",
        "chi_squared", "student_t", "f_distribution", "beta", "gamma",
        "weibull", "log_normal", "empirical"
    };
    if (type >= 0 && type < STAT_DIST_TYPE_COUNT) {
        return names[type];
    }
    return "unknown";
}

const char* stat_test_name(stat_test_type_t type) {
    static const char* names[] = {
        "z_test", "t_test", "paired_t_test", "welch_t_test", "chi_squared",
        "f_test", "kolmogorov_smirnov", "shapiro_wilk", "anderson_darling",
        "mann_whitney", "wilcoxon_signed", "fisher_exact"
    };
    if (type >= 0 && type < STAT_TEST_TYPE_COUNT) {
        return names[type];
    }
    return "unknown";
}

const char* stat_correlation_name(stat_correlation_type_t type) {
    static const char* names[] = {
        "pearson", "spearman", "kendall", "point_biserial",
        "autocorrelation", "cross_correlation", "partial"
    };
    if (type >= 0 && type < STAT_CORR_TYPE_COUNT) {
        return names[type];
    }
    return "unknown";
}

const char* stat_get_last_error(stat_analyzer_t* analyzer) {
    if (analyzer && analyzer->last_error[0]) {
        return analyzer->last_error;
    }
    return tls_error;
}

void stat_free_samples(double* samples) {
    free(samples);
}

void stat_free_indices(size_t* indices) {
    free(indices);
}

void stat_free_matrix(double* matrix, size_t rows) {
    (void)rows;
    free(matrix);
}

char* stat_export_json(stat_analyzer_t* analyzer) {
    if (!analyzer) return NULL;

    pthread_mutex_lock(&analyzer->mutex);

    size_t buf_size = 4096 + analyzer->series_count * 512;
    char* json = malloc(buf_size);
    if (!json) {
        pthread_mutex_unlock(&analyzer->mutex);
        return NULL;
    }

    char* p = json;
    p += sprintf(p, "{\n  \"series\": [\n");

    bool first = true;
    for (size_t i = 0; i < HASH_TABLE_SIZE; i++) {
        series_entry_t* entry = analyzer->series[i];
        while (entry) {
            if (!first) p += sprintf(p, ",\n");
            first = false;

            stat_descriptive_t stats;
            stat_descriptive_from_data(entry->data, entry->count, &stats);

            p += sprintf(p,
                "    {\n"
                "      \"name\": \"%s\",\n"
                "      \"count\": %zu,\n"
                "      \"mean\": %.6f,\n"
                "      \"std_dev\": %.6f,\n"
                "      \"min\": %.6f,\n"
                "      \"max\": %.6f,\n"
                "      \"median\": %.6f\n"
                "    }",
                entry->name,
                stats.count,
                stats.mean,
                stats.std_dev,
                stats.min,
                stats.max,
                stats.median);

            entry = entry->next;
        }
    }

    p += sprintf(p, "\n  ],\n");

    // Add QEC stats
    p += sprintf(p, "  \"qec\": {\n");
    p += sprintf(p, "    \"total_cycles\": %zu,\n", analyzer->qec.total_cycles);
    p += sprintf(p, "    \"logical_errors\": %zu,\n", analyzer->qec.logical_errors);
    if (analyzer->qec.total_cycles > 0) {
        p += sprintf(p, "    \"logical_error_rate\": %.6e\n",
                     (double)analyzer->qec.logical_errors / (double)analyzer->qec.total_cycles);
    } else {
        p += sprintf(p, "    \"logical_error_rate\": null\n");
    }
    p += sprintf(p, "  }\n}\n");

    pthread_mutex_unlock(&analyzer->mutex);
    return json;
}

bool stat_export_to_file(stat_analyzer_t* analyzer, const char* filename) {
    if (!analyzer || !filename) return false;

    char* json = stat_export_json(analyzer);
    if (!json) return false;

    FILE* f = fopen(filename, "w");
    if (!f) {
        free(json);
        return false;
    }

    fputs(json, f);
    fclose(f);
    free(json);

    return true;
}

char* stat_generate_report(stat_analyzer_t* analyzer) {
    if (!analyzer) return NULL;

    pthread_mutex_lock(&analyzer->mutex);

    size_t buf_size = 8192;
    char* report = malloc(buf_size);
    if (!report) {
        pthread_mutex_unlock(&analyzer->mutex);
        return NULL;
    }

    char* p = report;
    p += sprintf(p, "=== Statistical Analysis Report ===\n\n");
    p += sprintf(p, "Data Series: %zu\n\n", analyzer->series_count);

    for (size_t i = 0; i < HASH_TABLE_SIZE; i++) {
        series_entry_t* entry = analyzer->series[i];
        while (entry) {
            stat_descriptive_t stats;
            stat_descriptive_from_data(entry->data, entry->count, &stats);

            p += sprintf(p, "Series: %s (n=%zu)\n", entry->name, stats.count);
            p += sprintf(p, "  Mean: %.4f ± %.4f (SE)\n", stats.mean, stats.std_error);
            p += sprintf(p, "  Median: %.4f, IQR: [%.4f, %.4f]\n",
                        stats.median, stats.q1, stats.q3);
            p += sprintf(p, "  Range: [%.4f, %.4f]\n", stats.min, stats.max);
            p += sprintf(p, "  Skewness: %.4f, Kurtosis: %.4f\n\n",
                        stats.skewness, stats.kurtosis);

            entry = entry->next;
        }
    }

    // QEC summary
    if (analyzer->qec.total_cycles > 0) {
        p += sprintf(p, "=== QEC Statistics ===\n");
        p += sprintf(p, "Total cycles: %zu\n", analyzer->qec.total_cycles);
        p += sprintf(p, "Logical errors: %zu\n", analyzer->qec.logical_errors);
        p += sprintf(p, "Logical error rate: %.2e\n",
                     (double)analyzer->qec.logical_errors / (double)analyzer->qec.total_cycles);
    }

    pthread_mutex_unlock(&analyzer->mutex);
    return report;
}
