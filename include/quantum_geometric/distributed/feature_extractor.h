#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

// Feature parameters
#define MAX_RAW_FEATURES 64
#define MAX_DERIVED_FEATURES 32
#define HISTORY_WINDOW 100

// System metrics structure
typedef struct SystemMetrics {
    // CPU metrics
    double cpu_usage;
    double cpu_frequency;
    double cpu_temperature;

    // Memory metrics
    double memory_usage;
    double memory_bandwidth;
    double page_faults;

    // Network metrics
    double network_throughput;
    double packet_loss;
    double latency;

    // GPU metrics
    double gpu_utilization;
    double gpu_memory_usage;
    double gpu_temperature;

    // Quantum metrics
    double quantum_utilization;
    double qubit_error_rate;
    double quantum_coherence;

    // Disk metrics
    double disk_read_rate;
    double disk_write_rate;
    double io_wait;
} SystemMetrics;

// Extractor configuration
typedef struct ExtractorConfig {
    bool normalize_features;
    bool track_history;
    size_t history_window;
    double outlier_threshold;
    bool compute_trends;
} ExtractorConfig;

// Feature descriptor
typedef struct FeatureDescriptor {
    char* name;
    double weight;
    bool is_normalized;
    bool requires_history;
    int metric_type;
} FeatureDescriptor;

// Feature statistics
typedef struct FeatureStats {
    double mean;
    double std_dev;
    double min_value;
    double max_value;
    size_t num_samples;
    double m2;  // For Welford's online variance
} FeatureStats;

// Feature history
typedef struct FeatureHistory {
    double* values;
    size_t length;
    size_t capacity;
    time_t* timestamps;
} FeatureHistory;

// Forward declaration for opaque type
typedef struct FeatureExtractor FeatureExtractor;

// Lifecycle functions
FeatureExtractor* init_feature_extractor(const ExtractorConfig* config);
void cleanup_feature_extractor(FeatureExtractor* extractor);

// Feature extraction
double* extract_features(FeatureExtractor* extractor, const SystemMetrics* metrics);

// Helper functions
FeatureDescriptor* create_feature_descriptors(void);
void cleanup_feature_descriptors(FeatureDescriptor* descriptors, size_t count);

FeatureHistory* create_feature_history(void);
void cleanup_feature_history(FeatureHistory* history);
void update_feature_history(FeatureHistory* history, double value);

// Statistics functions
void update_running_stats(FeatureStats* stats, double value);
bool has_sufficient_history(const FeatureExtractor* extractor);

// Efficiency computation
double compute_cpu_efficiency(const SystemMetrics* metrics);
double compute_memory_efficiency(const SystemMetrics* metrics);
double compute_network_efficiency(const SystemMetrics* metrics);
double compute_quantum_efficiency(const SystemMetrics* metrics);

// Pressure computation
double compute_memory_pressure(const SystemMetrics* metrics);
double compute_io_pressure(const SystemMetrics* metrics);
double compute_network_pressure(const SystemMetrics* metrics);

// Trend computation
double compute_cpu_trend(const FeatureExtractor* extractor);
double compute_memory_trend(const FeatureExtractor* extractor);
double compute_quantum_trend(const FeatureExtractor* extractor);

#ifdef __cplusplus
}
#endif

#endif // FEATURE_EXTRACTOR_H
