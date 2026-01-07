#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <stdbool.h>
#include <stddef.h>
#include "quantum_geometric/hardware/quantum_hardware_types.h"
#include "quantum_geometric/core/quantum_complex.h"

// Data loader specific performance metrics
typedef struct {
    double load_time;        // Time taken to load data
    size_t memory_usage;     // Memory usage in bytes
    double throughput;       // Data throughput in samples/second
} data_loader_metrics_t;

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

// Data loader performance configuration
typedef struct {
    size_t cache_size;
    size_t prefetch_size;
    size_t num_workers;
    bool profile;
} data_performance_config_t;

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
dataset_t* allocate_dataset(size_t num_samples, size_t feature_dim, size_t num_classes, memory_config_t* memory_config);
dataset_t* quantum_load_dataset(const char* path, dataset_config_t config);
dataset_t* quantum_create_synthetic_data(size_t num_samples, size_t feature_dim, size_t num_classes, int type);
dataset_split_t quantum_split_dataset(dataset_t* dataset, float train_ratio, float val_ratio, bool shuffle, bool stratify);
bool quantum_normalize_data(dataset_t* dataset, normalization_t method);
void quantum_dataset_destroy(dataset_t* dataset);
void quantum_dataset_split_destroy(dataset_split_t* split);
bool quantum_configure_performance(data_performance_config_t config);
bool quantum_configure_memory(memory_config_t config);
bool quantum_get_data_loader_metrics(data_loader_metrics_t* metrics);
bool quantum_reset_performance_metrics(void);

// Additional quantum matrix operations
bool quantum_decompose_matrix(const float* matrix, size_t size, float* U, float* V);
bool quantum_compute_condition_number(const float** matrix, size_t size, float* condition_number);

// Standard dataset loaders (download from official sources)
dataset_t* quantum_load_mnist(dataset_config_t config);
dataset_t* quantum_load_cifar10(dataset_config_t config);
dataset_t* quantum_load_uci(const char* name, dataset_config_t config);

// CSV configuration
typedef struct {
    const char* delimiter;
    bool has_header;
    size_t skip_rows;
    size_t skip_cols;
} csv_config_t;

// ============================================================================
// Simplified Data Loader API for Image Datasets (MNIST, CIFAR-10, etc.)
// ============================================================================

// Forward declaration of tensor_t (defined in tensor_types.h)
struct tensor_t;

/**
 * @brief Simple data loader for image datasets
 *
 * This provides a simplified interface for loading batch image data
 * for training quantum machine learning models.
 */
typedef struct DataLoader {
    size_t image_width;         /**< Width of images in pixels */
    size_t image_height;        /**< Height of images in pixels */
    size_t num_channels;        /**< Number of color channels (1=grayscale, 3=RGB) */
    size_t batch_size;          /**< Current batch size */
    size_t num_classes;         /**< Number of classification classes */
    void* data_handle;          /**< Internal handle for data management */
} DataLoader;

/**
 * @brief Initialize a data loader for a named dataset
 *
 * @param loader Pointer to DataLoader structure to initialize
 * @param dataset_name Name of dataset ("cifar10", "mnist", etc.)
 * @param batch_size Number of samples per batch
 * @return true if initialization successful, false otherwise
 */
bool init_data_loader(DataLoader* loader, const char* dataset_name, size_t batch_size);

/**
 * @brief Clean up data loader resources
 *
 * @param loader Pointer to DataLoader to clean up
 */
void cleanup_data_loader(DataLoader* loader);

/**
 * @brief Load the next batch of images and labels
 *
 * @param loader Pointer to initialized DataLoader
 * @param images Output tensor for image data [batch_size x height x width x channels]
 * @param labels Output tensor for labels [batch_size x num_classes] (one-hot encoded)
 * @return true if batch loaded successfully, false otherwise
 */
bool load_next_batch(DataLoader* loader, struct tensor_t* images, struct tensor_t* labels);

#endif // DATA_LOADER_H
