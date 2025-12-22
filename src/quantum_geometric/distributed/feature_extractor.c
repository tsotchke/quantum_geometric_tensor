/**
 * @file feature_extractor.c
 * @brief Feature extraction for distributed training optimization
 */

#include "quantum_geometric/distributed/feature_extractor.h"
#include <math.h>
#include <string.h>

// Minimum variance for normalization
#define MIN_VARIANCE 1e-6
#define MIN_HISTORY_SAMPLES 10

// Forward declarations for static functions
static void extract_raw_features(FeatureExtractor* extractor,
                                const SystemMetrics* metrics, double* features);
static void compute_derived_features(FeatureExtractor* extractor,
                                    const SystemMetrics* metrics, double* features);
static void update_feature_statistics(FeatureExtractor* extractor,
                                     const double* features);
static void normalize_features(FeatureExtractor* extractor, double* features);
static double compute_trend(const FeatureHistory* history);

// Feature extractor structure
struct FeatureExtractor {
    FeatureDescriptor* features;
    size_t num_features;
    FeatureStats* statistics;
    FeatureHistory** history;
    double* normalization_params;
    bool* requires_scaling;
    ExtractorConfig config;
};

// Create feature descriptors
FeatureDescriptor* create_feature_descriptors(void) {
    size_t total = MAX_RAW_FEATURES + MAX_DERIVED_FEATURES;
    FeatureDescriptor* descriptors = malloc(total * sizeof(FeatureDescriptor));
    if (!descriptors) return NULL;

    memset(descriptors, 0, total * sizeof(FeatureDescriptor));

    for (size_t i = 0; i < total; i++) {
        descriptors[i].weight = 1.0;
        descriptors[i].is_normalized = false;
        descriptors[i].requires_history = (i < MAX_RAW_FEATURES);
        descriptors[i].metric_type = (int)(i % 8);
    }

    return descriptors;
}

// Cleanup feature descriptors
void cleanup_feature_descriptors(FeatureDescriptor* descriptors, size_t count) {
    if (!descriptors) return;
    for (size_t i = 0; i < count; i++) {
        free(descriptors[i].name);
    }
    free(descriptors);
}

// Create feature history
FeatureHistory* create_feature_history(void) {
    FeatureHistory* history = malloc(sizeof(FeatureHistory));
    if (!history) return NULL;

    history->capacity = HISTORY_WINDOW;
    history->length = 0;
    history->values = malloc(HISTORY_WINDOW * sizeof(double));
    history->timestamps = malloc(HISTORY_WINDOW * sizeof(time_t));

    if (!history->values || !history->timestamps) {
        free(history->values);
        free(history->timestamps);
        free(history);
        return NULL;
    }

    return history;
}

// Cleanup feature history
void cleanup_feature_history(FeatureHistory* history) {
    if (!history) return;
    free(history->values);
    free(history->timestamps);
    free(history);
}

// Update feature history (circular buffer)
void update_feature_history(FeatureHistory* history, double value) {
    if (!history) return;

    if (history->length < history->capacity) {
        history->values[history->length] = value;
        history->timestamps[history->length] = time(NULL);
        history->length++;
    } else {
        memmove(history->values, history->values + 1,
                (history->capacity - 1) * sizeof(double));
        memmove(history->timestamps, history->timestamps + 1,
                (history->capacity - 1) * sizeof(time_t));
        history->values[history->capacity - 1] = value;
        history->timestamps[history->capacity - 1] = time(NULL);
    }
}

// Update running statistics using Welford's online algorithm
void update_running_stats(FeatureStats* stats, double value) {
    if (!stats) return;

    stats->num_samples++;

    if (stats->num_samples == 1) {
        stats->mean = value;
        stats->m2 = 0.0;
        stats->min_value = value;
        stats->max_value = value;
    } else {
        double delta = value - stats->mean;
        stats->mean += delta / (double)stats->num_samples;
        double delta2 = value - stats->mean;
        stats->m2 += delta * delta2;

        if (value < stats->min_value) stats->min_value = value;
        if (value > stats->max_value) stats->max_value = value;
    }

    if (stats->num_samples > 1) {
        stats->std_dev = sqrt(stats->m2 / (double)(stats->num_samples - 1));
    }
}

// Check if extractor has sufficient history
bool has_sufficient_history(const FeatureExtractor* extractor) {
    if (!extractor || !extractor->history) return false;

    for (size_t i = 0; i < MAX_RAW_FEATURES; i++) {
        if (extractor->history[i] &&
            extractor->history[i]->length >= MIN_HISTORY_SAMPLES) {
            return true;
        }
    }
    return false;
}

// Compute CPU efficiency
double compute_cpu_efficiency(const SystemMetrics* metrics) {
    if (!metrics) return 0.0;
    double temp_factor = 1.0 - (metrics->cpu_temperature / 100.0);
    return metrics->cpu_usage * fmax(temp_factor, 0.1);
}

// Compute memory efficiency
double compute_memory_efficiency(const SystemMetrics* metrics) {
    if (!metrics) return 0.0;
    double bandwidth_efficiency = metrics->memory_bandwidth /
                                 fmax(metrics->memory_usage, MIN_VARIANCE);
    return fmin(bandwidth_efficiency, 1.0);
}

// Compute network efficiency
double compute_network_efficiency(const SystemMetrics* metrics) {
    if (!metrics) return 0.0;
    double loss_factor = 1.0 - metrics->packet_loss;
    double latency_factor = 1.0 / fmax(metrics->latency, MIN_VARIANCE);
    return metrics->network_throughput * loss_factor * latency_factor * 0.001;
}

// Compute quantum efficiency
double compute_quantum_efficiency(const SystemMetrics* metrics) {
    if (!metrics) return 0.0;
    double error_factor = 1.0 - metrics->qubit_error_rate;
    return metrics->quantum_utilization * error_factor * metrics->quantum_coherence;
}

// Compute memory pressure
double compute_memory_pressure(const SystemMetrics* metrics) {
    if (!metrics) return 0.0;
    double usage_pressure = metrics->memory_usage / 100.0;
    double fault_pressure = fmin(metrics->page_faults / 1000.0, 1.0);
    return (usage_pressure + fault_pressure) / 2.0;
}

// Compute I/O pressure
double compute_io_pressure(const SystemMetrics* metrics) {
    if (!metrics) return 0.0;
    double io_rate = (metrics->disk_read_rate + metrics->disk_write_rate) / 200.0;
    return fmin((metrics->io_wait / 100.0 + io_rate) / 2.0, 1.0);
}

// Compute network pressure
double compute_network_pressure(const SystemMetrics* metrics) {
    if (!metrics) return 0.0;
    double latency_pressure = fmin(metrics->latency / 100.0, 1.0);
    double loss_pressure = metrics->packet_loss;
    return (latency_pressure + loss_pressure) / 2.0;
}

// Compute trend from history using linear regression
static double compute_trend(const FeatureHistory* history) {
    if (!history || history->length < 2) return 0.0;

    size_t n = history->length;
    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_xx = 0.0;

    for (size_t i = 0; i < n; i++) {
        double x = (double)i;
        double y = history->values[i];
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_xx += x * x;
    }

    double denom = (double)n * sum_xx - sum_x * sum_x;
    if (fabs(denom) < MIN_VARIANCE) return 0.0;

    return ((double)n * sum_xy - sum_x * sum_y) / denom;
}

// Compute CPU trend
double compute_cpu_trend(const FeatureExtractor* extractor) {
    if (!extractor || !extractor->history || !extractor->history[0]) return 0.0;
    return compute_trend(extractor->history[0]);
}

// Compute memory trend
double compute_memory_trend(const FeatureExtractor* extractor) {
    if (!extractor || !extractor->history || !extractor->history[3]) return 0.0;
    return compute_trend(extractor->history[3]);
}

// Compute quantum trend
double compute_quantum_trend(const FeatureExtractor* extractor) {
    if (!extractor || !extractor->history || !extractor->history[12]) return 0.0;
    return compute_trend(extractor->history[12]);
}

// Initialize feature extractor
FeatureExtractor* init_feature_extractor(const ExtractorConfig* config) {
    FeatureExtractor* extractor = malloc(sizeof(FeatureExtractor));
    if (!extractor) return NULL;

    memset(extractor, 0, sizeof(FeatureExtractor));

    extractor->features = create_feature_descriptors();
    extractor->num_features = MAX_RAW_FEATURES + MAX_DERIVED_FEATURES;

    extractor->statistics = calloc(MAX_RAW_FEATURES, sizeof(FeatureStats));

    extractor->history = malloc(MAX_RAW_FEATURES * sizeof(FeatureHistory*));
    if (extractor->history) {
        for (size_t i = 0; i < MAX_RAW_FEATURES; i++) {
            extractor->history[i] = create_feature_history();
        }
    }

    extractor->normalization_params = calloc(MAX_RAW_FEATURES, sizeof(double));
    extractor->requires_scaling = calloc(MAX_RAW_FEATURES, sizeof(bool));
    for (size_t i = 0; i < MAX_RAW_FEATURES; i++) {
        extractor->requires_scaling[i] = true;
    }

    if (config) {
        extractor->config = *config;
    } else {
        extractor->config.normalize_features = true;
        extractor->config.track_history = true;
        extractor->config.history_window = HISTORY_WINDOW;
        extractor->config.outlier_threshold = 3.0;
        extractor->config.compute_trends = true;
    }

    return extractor;
}

// Extract raw features
static void extract_raw_features(FeatureExtractor* extractor,
                                const SystemMetrics* metrics,
                                double* features) {
    if (!extractor || !metrics || !features) return;
    (void)extractor;

    size_t idx = 0;

    features[idx++] = metrics->cpu_usage;
    features[idx++] = metrics->cpu_frequency;
    features[idx++] = metrics->cpu_temperature;

    features[idx++] = metrics->memory_usage;
    features[idx++] = metrics->memory_bandwidth;
    features[idx++] = metrics->page_faults;

    features[idx++] = metrics->network_throughput;
    features[idx++] = metrics->packet_loss;
    features[idx++] = metrics->latency;

    features[idx++] = metrics->gpu_utilization;
    features[idx++] = metrics->gpu_memory_usage;
    features[idx++] = metrics->gpu_temperature;

    features[idx++] = metrics->quantum_utilization;
    features[idx++] = metrics->qubit_error_rate;
    features[idx++] = metrics->quantum_coherence;
}

// Compute derived features
static void compute_derived_features(FeatureExtractor* extractor,
                                    const SystemMetrics* metrics,
                                    double* features) {
    if (!extractor || !metrics || !features) return;

    size_t base_idx = MAX_RAW_FEATURES;

    features[base_idx++] = compute_cpu_efficiency(metrics);
    features[base_idx++] = compute_memory_efficiency(metrics);
    features[base_idx++] = compute_network_efficiency(metrics);
    features[base_idx++] = compute_quantum_efficiency(metrics);

    features[base_idx++] = compute_memory_pressure(metrics);
    features[base_idx++] = compute_io_pressure(metrics);
    features[base_idx++] = compute_network_pressure(metrics);

    if (extractor->config.compute_trends && has_sufficient_history(extractor)) {
        features[base_idx++] = compute_cpu_trend(extractor);
        features[base_idx++] = compute_memory_trend(extractor);
        features[base_idx++] = compute_quantum_trend(extractor);
    } else {
        features[base_idx++] = 0.0;
        features[base_idx++] = 0.0;
        features[base_idx++] = 0.0;
    }
}

// Update feature statistics
static void update_feature_statistics(FeatureExtractor* extractor,
                                     const double* features) {
    if (!extractor || !features) return;

    for (size_t i = 0; i < MAX_RAW_FEATURES && i < extractor->num_features; i++) {
        update_running_stats(&extractor->statistics[i], features[i]);

        if (extractor->config.track_history && extractor->history[i]) {
            update_feature_history(extractor->history[i], features[i]);
        }
    }
}

// Normalize features
static void normalize_features(FeatureExtractor* extractor, double* features) {
    if (!extractor || !features || !extractor->config.normalize_features) return;

    for (size_t i = 0; i < MAX_RAW_FEATURES; i++) {
        if (!extractor->requires_scaling[i]) continue;

        double std_dev = extractor->statistics[i].std_dev;
        if (std_dev < MIN_VARIANCE) std_dev = MIN_VARIANCE;

        features[i] = (features[i] - extractor->statistics[i].mean) / std_dev;
    }
}

// Extract features
double* extract_features(FeatureExtractor* extractor, const SystemMetrics* metrics) {
    if (!extractor || !metrics) return NULL;

    double* features = calloc(extractor->num_features, sizeof(double));
    if (!features) return NULL;

    extract_raw_features(extractor, metrics, features);
    compute_derived_features(extractor, metrics, features);
    update_feature_statistics(extractor, features);
    normalize_features(extractor, features);

    return features;
}

// Cleanup feature extractor
void cleanup_feature_extractor(FeatureExtractor* extractor) {
    if (!extractor) return;

    cleanup_feature_descriptors(extractor->features, extractor->num_features);
    free(extractor->statistics);

    if (extractor->history) {
        for (size_t i = 0; i < MAX_RAW_FEATURES; i++) {
            cleanup_feature_history(extractor->history[i]);
        }
        free(extractor->history);
    }

    free(extractor->normalization_params);
    free(extractor->requires_scaling);
    free(extractor);
}
