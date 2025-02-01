#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <stdbool.h>
#include <stddef.h>
#include "quantum_geometric/hardware/quantum_hardware_types.h"
#include "quantum_geometric/core/quantum_complex.h"

// Performance metrics structure
typedef struct {
    double load_time;        // Time taken to load data
    size_t memory_usage;     // Memory usage in bytes
    double throughput;       // Data throughput in samples/second
} performance_metrics_t;

// Dataset format types
typedef enum {
    DATA_FORMAT_CSV,
    DATA_FORMAT_NUMPY,
    DATA_FORMAT_HDF5,
    DATA_FORMAT_IMAGE
} data_format_t;

// Normalization types
typedef enum {
    NORMALIZATION_NONE,
    NORMALIZATION_MINMAX,
    NORMALIZATION_ZSCORE
} normalization_t;

// Synthetic data types
typedef enum {
    SYNTHETIC_DATA_CLASSIFICATION,
    SYNTHETIC_DATA_REGRESSION
} synthetic_data_type_t;

// Dataset configuration
typedef struct {
    data_format_t format;
    bool has_header;
    const char* delimiter;
    bool normalize;
    normalization_t normalization_method;
} dataset_config_t;

// Memory configuration
typedef struct {
    bool use_mmap;
    bool gpu_cache;
    size_t max_memory;
    bool streaming;
    size_t chunk_size;
    bool compress;
} memory_config_t;

// Performance configuration
typedef struct {
    size_t cache_size;
    size_t prefetch_size;
    size_t num_workers;
    bool profile;
} performance_config_t;

// Dataset structure
typedef struct {
    ComplexFloat** features;
    ComplexFloat* labels;
    size_t num_samples;
    size_t feature_dim;
    size_t num_classes;
    char** feature_names;
    char** class_names;
} dataset_t;

// Dataset split structure
typedef struct {
    dataset_t* train_data;
    dataset_t* val_data;
    dataset_t* test_data;
} dataset_split_t;

// Function declarations
dataset_t* quantum_load_dataset(const char* path, dataset_config_t config);
dataset_t* quantum_create_synthetic_data(size_t num_samples, size_t feature_dim, size_t num_classes, int type);
dataset_split_t quantum_split_dataset(dataset_t* dataset, float train_ratio, float val_ratio, bool shuffle, bool stratify);
bool quantum_normalize_data(dataset_t* dataset, normalization_t method);
void quantum_dataset_destroy(dataset_t* dataset);
void quantum_dataset_split_destroy(dataset_split_t* split);
bool quantum_configure_performance(performance_config_t config);
bool quantum_configure_memory(memory_config_t config);
bool quantum_get_performance_metrics(performance_metrics_t* metrics);
bool quantum_reset_performance_metrics(void);

// Additional quantum matrix operations
bool quantum_decompose_matrix(const float* matrix, size_t size, float* U, float* V);
bool quantum_compute_condition_number(const float** matrix, size_t size, float* condition_number);

// CSV configuration
typedef struct {
    const char* delimiter;
    bool has_header;
    size_t skip_rows;
    size_t skip_cols;
} csv_config_t;

#endif // DATA_LOADER_H
