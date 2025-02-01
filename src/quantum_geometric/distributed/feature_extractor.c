#include "quantum_geometric/distributed/feature_extractor.h"
#include "quantum_geometric/core/performance_operations.h"
#include <math.h>

// Feature parameters
#define MAX_RAW_FEATURES 64
#define MAX_DERIVED_FEATURES 32
#define HISTORY_WINDOW 100
#define MIN_VARIANCE 1e-6

// Raw metric type
typedef enum {
    CPU_UTILIZATION,
    MEMORY_USAGE,
    NETWORK_BANDWIDTH,
    DISK_IO,
    GPU_UTILIZATION,
    QUANTUM_USAGE,
    POWER_CONSUMPTION,
    TEMPERATURE
} MetricType;

// Feature descriptor
typedef struct {
    char* name;
    double weight;
    bool is_normalized;
    bool requires_history;
    MetricType metric_type;
} FeatureDescriptor;

// Feature statistics
typedef struct {
    double mean;
    double std_dev;
    double min_value;
    double max_value;
    size_t num_samples;
} FeatureStats;

// Feature history
typedef struct {
    double* values;
    size_t length;
    size_t capacity;
    time_t* timestamps;
} FeatureHistory;

// Feature extractor
typedef struct {
    // Feature definitions
    FeatureDescriptor* features;
    size_t num_features;
    
    // Statistics tracking
    FeatureStats* statistics;
    FeatureHistory** history;
    
    // Preprocessing
    double* normalization_params;
    bool* requires_scaling;
    
    // Configuration
    ExtractorConfig config;
} FeatureExtractor;

// Initialize feature extractor
FeatureExtractor* init_feature_extractor(
    const ExtractorConfig* config) {
    
    FeatureExtractor* extractor = aligned_alloc(64,
        sizeof(FeatureExtractor));
    if (!extractor) return NULL;
    
    // Initialize feature definitions
    extractor->features = create_feature_descriptors();
    extractor->num_features = 0;
    
    // Initialize statistics
    extractor->statistics = aligned_alloc(64,
        MAX_RAW_FEATURES * sizeof(FeatureStats));
    
    // Initialize history tracking
    extractor->history = aligned_alloc(64,
        MAX_RAW_FEATURES * sizeof(FeatureHistory*));
    for (size_t i = 0; i < MAX_RAW_FEATURES; i++) {
        extractor->history[i] = create_feature_history();
    }
    
    // Initialize preprocessing
    extractor->normalization_params = aligned_alloc(64,
        MAX_RAW_FEATURES * sizeof(double));
    extractor->requires_scaling = aligned_alloc(64,
        MAX_RAW_FEATURES * sizeof(bool));
    
    // Store configuration
    extractor->config = *config;
    
    return extractor;
}

// Extract features from metrics
double* extract_features(
    FeatureExtractor* extractor,
    const SystemMetrics* metrics) {
    
    // Allocate feature vector
    double* features = aligned_alloc(64,
        extractor->num_features * sizeof(double));
    
    // Extract raw features
    extract_raw_features(extractor, metrics, features);
    
    // Compute derived features
    compute_derived_features(extractor, metrics, features);
    
    // Update statistics
    update_feature_statistics(extractor, features);
    
    // Normalize features
    normalize_features(extractor, features);
    
    return features;
}

// Extract raw features
static void extract_raw_features(
    FeatureExtractor* extractor,
    const SystemMetrics* metrics,
    double* features) {
    
    size_t feature_idx = 0;
    
    // CPU features
    features[feature_idx++] = metrics->cpu_usage;
    features[feature_idx++] = metrics->cpu_frequency;
    features[feature_idx++] = metrics->cpu_temperature;
    
    // Memory features
    features[feature_idx++] = metrics->memory_usage;
    features[feature_idx++] = metrics->memory_bandwidth;
    features[feature_idx++] = metrics->page_faults;
    
    // Network features
    features[feature_idx++] = metrics->network_throughput;
    features[feature_idx++] = metrics->packet_loss;
    features[feature_idx++] = metrics->latency;
    
    // GPU features
    features[feature_idx++] = metrics->gpu_utilization;
    features[feature_idx++] = metrics->gpu_memory_usage;
    features[feature_idx++] = metrics->gpu_temperature;
    
    // Quantum features
    features[feature_idx++] = metrics->quantum_utilization;
    features[feature_idx++] = metrics->qubit_error_rate;
    features[feature_idx++] = metrics->quantum_coherence;
}

// Compute derived features
static void compute_derived_features(
    FeatureExtractor* extractor,
    const SystemMetrics* metrics,
    double* features) {
    
    size_t base_idx = extractor->num_features - MAX_DERIVED_FEATURES;
    
    // Compute efficiency metrics
    features[base_idx++] = compute_cpu_efficiency(metrics);
    features[base_idx++] = compute_memory_efficiency(metrics);
    features[base_idx++] = compute_network_efficiency(metrics);
    features[base_idx++] = compute_quantum_efficiency(metrics);
    
    // Compute resource pressure
    features[base_idx++] = compute_memory_pressure(metrics);
    features[base_idx++] = compute_io_pressure(metrics);
    features[base_idx++] = compute_network_pressure(metrics);
    
    // Compute trend features
    if (has_sufficient_history(extractor)) {
        features[base_idx++] = compute_cpu_trend(extractor);
        features[base_idx++] = compute_memory_trend(extractor);
        features[base_idx++] = compute_quantum_trend(extractor);
    }
}

// Update feature statistics
static void update_feature_statistics(
    FeatureExtractor* extractor,
    const double* features) {
    
    for (size_t i = 0; i < extractor->num_features; i++) {
        FeatureStats* stats = &extractor->statistics[i];
        
        // Update running statistics
        update_running_stats(stats, features[i]);
        
        // Update history if needed
        if (extractor->features[i].requires_history) {
            update_feature_history(extractor->history[i],
                                 features[i]);
        }
    }
}

// Normalize features
static void normalize_features(
    FeatureExtractor* extractor,
    double* features) {
    
    for (size_t i = 0; i < extractor->num_features; i++) {
        if (!extractor->requires_scaling[i]) continue;
        
        // Apply normalization
        features[i] = (features[i] - extractor->statistics[i].mean) /
                     (extractor->statistics[i].std_dev + MIN_VARIANCE);
    }
}

// Clean up
void cleanup_feature_extractor(FeatureExtractor* extractor) {
    if (!extractor) return;
    
    // Clean up feature definitions
    cleanup_feature_descriptors(extractor->features,
                              extractor->num_features);
    
    // Clean up statistics
    free(extractor->statistics);
    
    // Clean up history
    for (size_t i = 0; i < MAX_RAW_FEATURES; i++) {
        cleanup_feature_history(extractor->history[i]);
    }
    free(extractor->history);
    
    // Clean up preprocessing
    free(extractor->normalization_params);
    free(extractor->requires_scaling);
    
    free(extractor);
}
