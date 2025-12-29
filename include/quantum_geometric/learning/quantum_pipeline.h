#ifndef QUANTUM_PIPELINE_H
#define QUANTUM_PIPELINE_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Success/Error codes
#define QG_SUCCESS 0
#define QG_ERROR_INVALID_ARGUMENT 1
#define QG_ERROR_MEMORY_ALLOCATION 2
#define QG_ERROR_INITIALIZATION 3
#define QG_ERROR_RUNTIME 4
#define QG_ERROR_GPU_UNAVAILABLE 5
#define QG_ERROR_INVALID_STATE 6

// Pipeline configuration indices
#define QG_CONFIG_INPUT_DIM 0
#define QG_CONFIG_LATENT_DIM 1
#define QG_CONFIG_NUM_CLUSTERS 2
#define QG_CONFIG_NUM_CLASSES 3
#define QG_CONFIG_BATCH_SIZE 4
#define QG_CONFIG_LEARNING_RATE 5
#define QG_CONFIG_USE_GPU 6
#define QG_CONFIG_NUM_QUBITS 7
#define QG_CONFIG_NUM_LAYERS 8
#define QG_CONFIG_SIZE 9

// Opaque handle to quantum pipeline
typedef void* quantum_pipeline_handle_t;

// Pipeline configuration validation
bool quantum_pipeline_validate_config(const float* config);

// Create a quantum pipeline with given configuration
// config: Array of float parameters [input_dim, latent_dim, num_clusters, num_classes, batch_size, learning_rate, use_gpu]
// Returns: Handle to created pipeline or NULL on error
quantum_pipeline_handle_t quantum_pipeline_create(const float* config);

// Load a pre-trained pipeline from file
// filename: Path to pipeline file
// Returns: Handle to loaded pipeline or NULL on error
quantum_pipeline_handle_t quantum_pipeline_load(const char* filename);

// Train the quantum pipeline on given data
// pipeline: Pipeline handle from quantum_pipeline_create
// data: Input data array [num_samples x input_dim]
// labels: Target labels array [num_samples]
// num_samples: Number of training samples
// Returns: QG_SUCCESS on success, error code on failure
int quantum_pipeline_train(quantum_pipeline_handle_t pipeline,
                         const float* data,
                         const int* labels,
                         size_t num_samples);

// Get current training progress
// pipeline: Pipeline handle from quantum_pipeline_create
// epoch: Current epoch number
// loss: Current loss value
// accuracy: Current accuracy
// Returns: QG_SUCCESS on success, error code on failure
int quantum_pipeline_get_progress(quantum_pipeline_handle_t pipeline,
                                size_t* epoch,
                                float* loss,
                                float* accuracy);

// Evaluate the quantum pipeline on test data
// pipeline: Pipeline handle from quantum_pipeline_create
// data: Input data array [num_samples x input_dim]
// labels: Target labels array [num_samples]
// num_samples: Number of test samples
// results: Output array for metrics [accuracy, time_ms, memory_mb]
// Returns: QG_SUCCESS on success, error code on failure
int quantum_pipeline_evaluate(quantum_pipeline_handle_t pipeline,
                           const float* data,
                           const int* labels,
                           size_t num_samples,
                           float* results);

// Make predictions using the pipeline
// pipeline: Pipeline handle from quantum_pipeline_create
// data: Input data array [input_dim]
// prediction: Output prediction
// Returns: QG_SUCCESS on success, error code on failure
int quantum_pipeline_predict(quantum_pipeline_handle_t pipeline,
                           const float* data,
                           int* prediction);

// Compare with classical pipeline
// pipeline: Pipeline handle from quantum_pipeline_create
// data: Input data array [num_samples x input_dim]
// labels: Target labels array [num_samples]
// num_samples: Number of test samples
// results: Output array for classical metrics [accuracy, time_ms, memory_mb]
// Returns: QG_SUCCESS on success, error code on failure
int quantum_pipeline_compare_classical(quantum_pipeline_handle_t pipeline,
                                    const float* data,
                                    const int* labels,
                                    size_t num_samples,
                                    float* results);

// Save trained pipeline to file
// pipeline: Pipeline handle from quantum_pipeline_create
// filename: Path to save file
// Returns: QG_SUCCESS on success, error code on failure
int quantum_pipeline_save(quantum_pipeline_handle_t pipeline,
                        const char* filename);

// Get pipeline performance metrics
// pipeline: Pipeline handle from quantum_pipeline_create
// gpu_utilization: GPU utilization percentage
// memory_usage: Memory usage in bytes
// throughput: Samples processed per second
// Returns: QG_SUCCESS on success, error code on failure
int quantum_pipeline_get_metrics(quantum_pipeline_handle_t pipeline,
                               float* gpu_utilization,
                               size_t* memory_usage,
                               float* throughput);

// Enable/disable pipeline features
int quantum_pipeline_enable_feature(quantum_pipeline_handle_t pipeline,
                                  const char* feature_name);
int quantum_pipeline_disable_feature(quantum_pipeline_handle_t pipeline,
                                   const char* feature_name);

// Get last error message
const char* quantum_pipeline_get_error(quantum_pipeline_handle_t pipeline);

// Destroy pipeline and free resources
// pipeline: Pipeline handle from quantum_pipeline_create
void quantum_pipeline_destroy(quantum_pipeline_handle_t pipeline);

// ============================================================================
// Simplified Quantum Pipeline API for Training and Inference
// ============================================================================

// Forward declaration of tensor_t (defined in tensor_types.h)
struct tensor_t;

/**
 * @brief Simplified quantum pipeline for training and inference
 *
 * This provides a simple interface for quantum machine learning pipelines
 * that abstracts the internal complexity of the quantum circuit.
 */
typedef struct QuantumPipeline {
    size_t input_size;          /**< Input dimension (flattened) */
    size_t output_size;         /**< Output dimension (number of classes) */
    float learning_rate;        /**< Current learning rate */
    void* model_handle;         /**< Internal handle for quantum model */
} QuantumPipeline;

/**
 * @brief Initialize a quantum pipeline
 *
 * @param pipeline Pointer to QuantumPipeline structure to initialize
 * @param input_size Input dimension (e.g., image_width * image_height * channels)
 * @param output_size Output dimension (number of classes for classification)
 * @param learning_rate Learning rate for training
 * @return true if initialization successful, false otherwise
 */
bool init_quantum_pipeline(QuantumPipeline* pipeline, size_t input_size,
                          size_t output_size, float learning_rate);

/**
 * @brief Clean up quantum pipeline resources
 *
 * @param pipeline Pointer to QuantumPipeline to clean up
 */
void cleanup_quantum_pipeline(QuantumPipeline* pipeline);

/**
 * @brief Perform a single training step
 *
 * @param pipeline Pointer to initialized QuantumPipeline
 * @param input Input tensor [batch_size x input_size]
 * @param target Target tensor [batch_size x output_size] (one-hot encoded)
 * @param loss Output: computed loss value
 * @return true if training step successful, false otherwise
 */
bool train_step(QuantumPipeline* pipeline, struct tensor_t* input,
               struct tensor_t* target, float* loss);

/**
 * @brief Perform inference on input data
 *
 * @param pipeline Pointer to initialized QuantumPipeline
 * @param input Input tensor [batch_size x input_size]
 * @param output Output tensor (will be allocated) [batch_size x output_size]
 * @return true if inference successful, false otherwise
 */
bool inference(QuantumPipeline* pipeline, struct tensor_t* input,
              struct tensor_t* output);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_PIPELINE_H
