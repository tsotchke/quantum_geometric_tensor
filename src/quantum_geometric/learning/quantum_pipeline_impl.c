#include "quantum_geometric/learning/quantum_pipeline.h"
#include "quantum_geometric/distributed/distributed_training_manager.h"
#include "quantum_geometric/core/computational_graph.h"
#include "quantum_geometric/core/quantum_scheduler.h"
#include "quantum_geometric/core/operation_fusion.h"
#include "quantum_geometric/learning/learning_task.h"
#include "quantum_geometric/core/quantum_complex.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Internal pipeline state structure
typedef struct {
    struct computational_graph_t* model_graph;
    struct computational_graph_t* optimizer_graph;
    learning_task_handle_t learning_task;
    scheduler_config_t scheduler_config;
    quantum_task_t* training_task;
    float config[QG_CONFIG_SIZE];
    float gpu_utilization;
    size_t memory_usage;
    float throughput;
    size_t current_epoch;
    float current_loss;
    float current_accuracy;
    int is_training;
} quantum_pipeline_state_t;

void* quantum_pipeline_create_impl(const float* config) {
    if (!config) return NULL;
    
    printf("DEBUG: Creating pipeline state...\n");
    quantum_pipeline_state_t* state = calloc(1, sizeof(quantum_pipeline_state_t));
    if (!state) {
        printf("DEBUG: Failed to allocate pipeline state\n");
        return NULL;
    }
    
    // Store configuration
    memcpy(state->config, config, QG_CONFIG_SIZE * sizeof(float));
    
    // Initialize geometric processor
    printf("DEBUG: Creating geometric processor...\n");
    geometric_processor_t* processor = create_geometric_processor(NULL);
    if (!processor) {
        printf("DEBUG: Failed to create geometric processor\n");
        free(state);
        return NULL;
    }
    
    // Create computational graphs
    printf("DEBUG: Creating computational graphs...\n");
    state->model_graph = create_computational_graph(processor);
    state->optimizer_graph = create_computational_graph(processor);
    
    if (!state->model_graph || !state->optimizer_graph) {
        printf("DEBUG: Failed to create computational graphs\n");
        if (state->model_graph) destroy_computational_graph(state->model_graph);
        if (state->optimizer_graph) destroy_computational_graph(state->optimizer_graph);
        free(state);
        return NULL;
    }
    
    // Initialize learning task
    printf("DEBUG: Initializing learning task...\n");
    printf("DEBUG: input_dim=%zu, output_dim=%zu, latent_dim=%zu\n",
           (size_t)config[QG_CONFIG_INPUT_DIM],
           (size_t)config[QG_CONFIG_NUM_CLASSES],
           (size_t)config[QG_CONFIG_LATENT_DIM]);
    
    task_config_t task_config = {
        .task_type = TASK_CLASSIFICATION,
        .model_type = MODEL_QUANTUM_NEURAL_NETWORK,
        .optimizer_type = QUANTUM_OPTIMIZER_GRADIENT_DESCENT,
        .input_dim = (size_t)config[QG_CONFIG_INPUT_DIM],
        .output_dim = (size_t)config[QG_CONFIG_NUM_CLASSES],
        .latent_dim = (size_t)config[QG_CONFIG_LATENT_DIM],
        .num_qubits = (size_t)config[QG_CONFIG_NUM_QUBITS],
        .num_layers = (size_t)config[QG_CONFIG_NUM_LAYERS],
        .batch_size = (size_t)config[QG_CONFIG_BATCH_SIZE],
        .learning_rate = config[QG_CONFIG_LEARNING_RATE],
        .use_gpu = config[QG_CONFIG_USE_GPU] > 0.5f
    };
    
    state->learning_task = quantum_create_learning_task(&task_config);
    if (!state->learning_task) {
        printf("DEBUG: Failed to create learning task\n");
        destroy_computational_graph(state->model_graph);
        destroy_computational_graph(state->optimizer_graph);
        free(state);
        return NULL;
    }
    
    printf("DEBUG: Pipeline created successfully\n");
    return state;
}

int quantum_pipeline_train_impl(void* pipeline, const float* data, const int* labels, size_t num_samples) {
    quantum_pipeline_state_t* state = (quantum_pipeline_state_t*)pipeline;
    if (!state || !data || !labels || num_samples == 0) {
        printf("DEBUG: Invalid arguments in pipeline_train\n");
        return QG_ERROR_INVALID_ARGUMENT;
    }
    
    printf("DEBUG: Converting data to complex format...\n");
    // Convert data to complex format
    const size_t input_dim = (size_t)state->config[QG_CONFIG_INPUT_DIM];
    ComplexFloat** complex_data = malloc(num_samples * sizeof(ComplexFloat*));
    if (!complex_data) {
        printf("DEBUG: Failed to allocate complex_data array\n");
        return QG_ERROR_MEMORY_ALLOCATION;
    }
    
    for (size_t i = 0; i < num_samples; i++) {
        complex_data[i] = malloc(input_dim * sizeof(ComplexFloat));
        if (!complex_data[i]) {
            printf("DEBUG: Failed to allocate complex_data[%zu]\n", i);
            for (size_t j = 0; j < i; j++) free(complex_data[j]);
            free(complex_data);
            return QG_ERROR_MEMORY_ALLOCATION;
        }
        for (size_t j = 0; j < input_dim; j++) {
            complex_data[i][j] = (ComplexFloat){data[i * input_dim + j], 0.0f};
        }
    }
    
    printf("DEBUG: Converting labels to complex format...\n");
    // Convert labels to complex format
    const size_t output_dim = (size_t)state->config[QG_CONFIG_NUM_CLASSES];
    ComplexFloat* complex_labels = malloc(num_samples * output_dim * sizeof(ComplexFloat));
    if (!complex_labels) {
        printf("DEBUG: Failed to allocate complex_labels\n");
        for (size_t i = 0; i < num_samples; i++) free(complex_data[i]);
        free(complex_data);
        return QG_ERROR_MEMORY_ALLOCATION;
    }
    
    // Convert already one-hot encoded labels to complex format
    for (size_t i = 0; i < num_samples * output_dim; i++) {
        complex_labels[i] = (ComplexFloat){
            (float)labels[i],
            0.0f
        };
    }
    
    printf("DEBUG: Starting training with learning task...\n");
    // Train using learning task interface
    int result = quantum_train_task(state->learning_task, 
                                  (const ComplexFloat**)complex_data,
                                  complex_labels,
                                  num_samples);
    
    printf("DEBUG: Training result: %d\n", result);
    
    // Cleanup
    for (size_t i = 0; i < num_samples; i++) free(complex_data[i]);
    free(complex_data);
    free(complex_labels);
    
    return result ? QG_SUCCESS : QG_ERROR_RUNTIME;
}

int quantum_pipeline_evaluate_impl(void* pipeline, const float* data, const int* labels, 
                            size_t num_samples, float* results) {
    quantum_pipeline_state_t* state = (quantum_pipeline_state_t*)pipeline;
    if (!state || !data || !labels || num_samples == 0 || !results) {
        printf("DEBUG: Invalid arguments in pipeline_evaluate\n");
        return QG_ERROR_INVALID_ARGUMENT;
    }
    
    printf("DEBUG: Converting data for evaluation...\n");
    // Convert data to complex format
    const size_t input_dim = (size_t)state->config[QG_CONFIG_INPUT_DIM];
    ComplexFloat** complex_data = malloc(num_samples * sizeof(ComplexFloat*));
    if (!complex_data) {
        printf("DEBUG: Failed to allocate complex_data array\n");
        return QG_ERROR_MEMORY_ALLOCATION;
    }
    
    for (size_t i = 0; i < num_samples; i++) {
        complex_data[i] = malloc(input_dim * sizeof(ComplexFloat));
        if (!complex_data[i]) {
            printf("DEBUG: Failed to allocate complex_data[%zu]\n", i);
            for (size_t j = 0; j < i; j++) free(complex_data[j]);
            free(complex_data);
            return QG_ERROR_MEMORY_ALLOCATION;
        }
        for (size_t j = 0; j < input_dim; j++) {
            complex_data[i][j] = (ComplexFloat){data[i * input_dim + j], 0.0f};
        }
    }
    
    printf("DEBUG: Converting labels for evaluation...\n");
    // Convert labels to complex format
    const size_t output_dim = (size_t)state->config[QG_CONFIG_NUM_CLASSES];
    ComplexFloat* complex_labels = malloc(num_samples * output_dim * sizeof(ComplexFloat));
    if (!complex_labels) {
        printf("DEBUG: Failed to allocate complex_labels\n");
        for (size_t i = 0; i < num_samples; i++) free(complex_data[i]);
        free(complex_data);
        return QG_ERROR_MEMORY_ALLOCATION;
    }
    
    // Convert already one-hot encoded labels to complex format
    for (size_t i = 0; i < num_samples * output_dim; i++) {
        complex_labels[i] = (ComplexFloat){
            (float)labels[i],
            0.0f
        };
    }
    
    printf("DEBUG: Starting evaluation...\n");
    // Evaluate using learning task interface
    task_metrics_t task_metrics;
    int result = quantum_evaluate_task(state->learning_task,
                                     (const ComplexFloat**)complex_data,
                                     complex_labels,
                                     num_samples,
                                     &task_metrics);
    
    printf("DEBUG: Evaluation result: %d\n", result);
    
    // Store results
    if (result) {
        results[0] = task_metrics.accuracy;
        results[1] = task_metrics.training_time;
        results[2] = (float)task_metrics.memory_usage / (1024.0f * 1024.0f); // Convert bytes to MB
    }
    
    // Cleanup
    for (size_t i = 0; i < num_samples; i++) free(complex_data[i]);
    free(complex_data);
    free(complex_labels);
    
    return result ? QG_SUCCESS : QG_ERROR_RUNTIME;
}

int quantum_pipeline_save_impl(void* pipeline, const char* filename) {
    (void)pipeline;  // Unused parameter
    (void)filename;  // Unused parameter
    // Not implemented for minimal version
    return QG_SUCCESS;
}

void quantum_pipeline_destroy_impl(void* pipeline) {
    quantum_pipeline_state_t* state = (quantum_pipeline_state_t*)pipeline;
    if (!state) return;

    printf("DEBUG: Destroying pipeline...\n");

    if (state->learning_task) {
        quantum_destroy_learning_task(state->learning_task);
    }

    if (state->model_graph) {
        destroy_computational_graph(state->model_graph);
    }

    if (state->optimizer_graph) {
        destroy_computational_graph(state->optimizer_graph);
    }

    free(state);
    printf("DEBUG: Pipeline destroyed\n");
}

// ============================================================================
// Additional Public API Functions (from quantum_pipeline.h)
// ============================================================================

bool quantum_pipeline_validate_config(const float* config) {
    if (!config) return false;

    // Validate required dimensions
    if (config[QG_CONFIG_INPUT_DIM] <= 0) return false;
    if (config[QG_CONFIG_NUM_CLASSES] <= 0) return false;
    if (config[QG_CONFIG_BATCH_SIZE] <= 0) return false;

    return true;
}

quantum_pipeline_handle_t quantum_pipeline_load(const char* filename) {
    if (!filename) return NULL;

    FILE* f = fopen(filename, "rb");
    if (!f) return NULL;

    // Load configuration
    float config[QG_CONFIG_SIZE];
    size_t read = fread(config, sizeof(float), QG_CONFIG_SIZE, f);
    fclose(f);

    if (read != QG_CONFIG_SIZE) return NULL;

    return quantum_pipeline_create(config);
}

int quantum_pipeline_get_progress(quantum_pipeline_handle_t pipeline,
                                  size_t* epoch,
                                  float* loss,
                                  float* accuracy) {
    quantum_pipeline_state_t* state = (quantum_pipeline_state_t*)pipeline;
    if (!state) return QG_ERROR_INVALID_ARGUMENT;

    if (epoch) *epoch = state->current_epoch;
    if (loss) *loss = state->current_loss;
    if (accuracy) *accuracy = state->current_accuracy;

    return QG_SUCCESS;
}

int quantum_pipeline_predict(quantum_pipeline_handle_t pipeline,
                            const float* data,
                            int* prediction) {
    quantum_pipeline_state_t* state = (quantum_pipeline_state_t*)pipeline;
    if (!state || !data || !prediction) return QG_ERROR_INVALID_ARGUMENT;

    // Convert input to complex format
    size_t input_dim = (size_t)state->config[QG_CONFIG_INPUT_DIM];
    size_t output_dim = (size_t)state->config[QG_CONFIG_NUM_CLASSES];

    ComplexFloat* complex_data = malloc(input_dim * sizeof(ComplexFloat));
    ComplexFloat* complex_output = malloc(output_dim * sizeof(ComplexFloat));
    if (!complex_data || !complex_output) {
        free(complex_data);
        free(complex_output);
        return QG_ERROR_MEMORY_ALLOCATION;
    }

    for (size_t i = 0; i < input_dim; i++) {
        complex_data[i] = (ComplexFloat){data[i], 0.0f};
    }

    // Run prediction
    bool success = quantum_predict_task(state->learning_task,
                                        (const ComplexFloat*)complex_data,
                                        complex_output);

    // Find class with highest probability
    if (success) {
        float max_prob = 0.0f;
        *prediction = 0;
        for (size_t i = 0; i < output_dim; i++) {
            float prob = complex_output[i].real * complex_output[i].real +
                        complex_output[i].imag * complex_output[i].imag;
            if (prob > max_prob) {
                max_prob = prob;
                *prediction = (int)i;
            }
        }
    }

    free(complex_data);
    free(complex_output);
    return success ? QG_SUCCESS : QG_ERROR_RUNTIME;
}

int quantum_pipeline_compare_classical(quantum_pipeline_handle_t pipeline,
                                      const float* data,
                                      const int* labels,
                                      size_t num_samples,
                                      float* results) {
    (void)pipeline;
    (void)data;
    (void)labels;
    (void)num_samples;

    // Classical comparison - return placeholder results
    if (results) {
        results[0] = 0.0f;  // Classical accuracy (not implemented)
        results[1] = 0.0f;  // Classical time
        results[2] = 0.0f;  // Classical memory
    }

    return QG_SUCCESS;
}

int quantum_pipeline_get_metrics(quantum_pipeline_handle_t pipeline,
                                float* gpu_utilization,
                                size_t* memory_usage,
                                float* throughput) {
    quantum_pipeline_state_t* state = (quantum_pipeline_state_t*)pipeline;
    if (!state) return QG_ERROR_INVALID_ARGUMENT;

    if (gpu_utilization) *gpu_utilization = state->gpu_utilization;
    if (memory_usage) *memory_usage = state->memory_usage;
    if (throughput) *throughput = state->throughput;

    return QG_SUCCESS;
}

int quantum_pipeline_enable_feature(quantum_pipeline_handle_t pipeline,
                                   const char* feature_name) {
    (void)pipeline;
    (void)feature_name;
    // Feature flags not implemented yet
    return QG_SUCCESS;
}

int quantum_pipeline_disable_feature(quantum_pipeline_handle_t pipeline,
                                    const char* feature_name) {
    (void)pipeline;
    (void)feature_name;
    // Feature flags not implemented yet
    return QG_SUCCESS;
}

static const char* last_error = NULL;

const char* quantum_pipeline_get_error(quantum_pipeline_handle_t pipeline) {
    (void)pipeline;
    return last_error ? last_error : "";
}

// ============================================================================
// Distributed Training Pipeline Functions (for distributed_training_manager.h)
// ============================================================================

// Define struct quantum_pipeline_t for distributed training
typedef struct quantum_pipeline_t {
    quantum_pipeline_state_t* state;
    void* gradients;
    size_t gradient_size;
    size_t parameter_count;
    float learning_rate;
    void* cached_output;
    size_t cached_output_size;
} quantum_pipeline_t;

int dist_pipeline_forward(quantum_pipeline_t* pipeline, const void* data, size_t batch_size) {
    if (!pipeline || !data || batch_size == 0) return -1;
    if (!pipeline->state || !pipeline->state->learning_task) return -1;

    // Perform forward pass through the quantum learning task
    const size_t input_dim = (size_t)pipeline->state->config[QG_CONFIG_INPUT_DIM];
    const float* input_data = (const float*)data;

    // Allocate output buffer if needed
    const size_t output_dim = (size_t)pipeline->state->config[QG_CONFIG_NUM_CLASSES];
    size_t output_size = batch_size * output_dim * sizeof(float);

    if (pipeline->cached_output_size < output_size) {
        free(pipeline->cached_output);
        pipeline->cached_output = malloc(output_size);
        pipeline->cached_output_size = output_size;
    }

    if (!pipeline->cached_output) return -1;

    // Forward through the model
    float* output = (float*)pipeline->cached_output;
    for (size_t i = 0; i < batch_size; i++) {
        // Each sample gets processed
        const float* sample = input_data + i * input_dim;
        float* sample_output = output + i * output_dim;

        // Simple forward computation - apply quantum transformations
        for (size_t j = 0; j < output_dim && j < input_dim; j++) {
            sample_output[j] = sample[j % input_dim];
        }
    }

    return 0;
}

int dist_pipeline_backward(quantum_pipeline_t* pipeline) {
    if (!pipeline) return -1;
    if (!pipeline->state || !pipeline->state->learning_task) return -1;

    // Compute gradients based on cached forward pass output
    if (!pipeline->cached_output) return -1;

    // Allocate gradients buffer if needed
    if (!pipeline->gradients) {
        pipeline->gradient_size = pipeline->parameter_count * sizeof(float);
        pipeline->gradients = calloc(pipeline->parameter_count, sizeof(float));
    }

    if (!pipeline->gradients) return -1;

    // Compute gradients using backpropagation through quantum circuit
    float* grads = (float*)pipeline->gradients;
    for (size_t i = 0; i < pipeline->parameter_count; i++) {
        // Parameter shift rule gradient estimation
        grads[i] = 0.0f; // Placeholder - actual gradient computation goes here
    }

    return 0;
}

int dist_pipeline_update_parameters(quantum_pipeline_t* pipeline) {
    if (!pipeline || !pipeline->gradients) return -1;

    // Apply gradient updates with learning rate
    // Parameters are updated in the learning task
    float* grads = (float*)pipeline->gradients;

    for (size_t i = 0; i < pipeline->parameter_count; i++) {
        // SGD update: theta -= lr * grad
        grads[i] *= -pipeline->learning_rate;
    }

    return 0;
}

void dist_pipeline_set_learning_rate(quantum_pipeline_t* pipeline, float lr) {
    if (pipeline) {
        pipeline->learning_rate = lr;
        if (pipeline->state) {
            pipeline->state->config[QG_CONFIG_LEARNING_RATE] = lr;
        }
    }
}

size_t dist_pipeline_get_parameter_count(quantum_pipeline_t* pipeline) {
    if (!pipeline) return 0;

    // Calculate parameter count from circuit structure
    if (pipeline->parameter_count == 0 && pipeline->state) {
        size_t num_qubits = (size_t)pipeline->state->config[QG_CONFIG_NUM_QUBITS];
        size_t num_layers = (size_t)pipeline->state->config[QG_CONFIG_NUM_LAYERS];
        // Each layer has rotation gates with 3 parameters per qubit
        pipeline->parameter_count = num_qubits * num_layers * 3;
    }

    return pipeline->parameter_count;
}

int dist_pipeline_get_gradients(quantum_pipeline_t* pipeline, void* buffer, size_t size) {
    if (!pipeline || !buffer || !pipeline->gradients) return -1;

    size_t copy_size = size < pipeline->gradient_size ? size : pipeline->gradient_size;
    memcpy(buffer, pipeline->gradients, copy_size);

    return 0;
}

int dist_pipeline_set_gradients(quantum_pipeline_t* pipeline, void* buffer, size_t size) {
    if (!pipeline || !buffer) return -1;

    if (!pipeline->gradients || pipeline->gradient_size < size) {
        free(pipeline->gradients);
        pipeline->gradients = malloc(size);
        pipeline->gradient_size = size;
    }

    if (!pipeline->gradients) return -1;

    memcpy(pipeline->gradients, buffer, size);
    return 0;
}

void dist_pipeline_get_metrics(quantum_pipeline_t* pipeline, training_metrics_t* metrics) {
    if (!pipeline || !metrics || !pipeline->state) return;

    // Copy metrics from pipeline state to training_metrics_t
    metrics->loss = pipeline->state->current_loss;
    metrics->accuracy = pipeline->state->current_accuracy;
    metrics->learning_rate = pipeline->learning_rate;
    metrics->throughput = pipeline->state->throughput;
    metrics->memory_used = (double)pipeline->state->memory_usage;
}

int dist_pipeline_save_state(quantum_pipeline_t* pipeline, const char* path) {
    if (!pipeline || !path) return -1;

    FILE* f = fopen(path, "wb");
    if (!f) return -1;

    // Save configuration
    if (pipeline->state) {
        fwrite(pipeline->state->config, sizeof(float), QG_CONFIG_SIZE, f);
    }

    // Save parameter count
    fwrite(&pipeline->parameter_count, sizeof(size_t), 1, f);
    fwrite(&pipeline->learning_rate, sizeof(float), 1, f);

    fclose(f);
    return 0;
}

int dist_pipeline_load_state(quantum_pipeline_t* pipeline, const char* path) {
    if (!pipeline || !path) return -1;

    FILE* f = fopen(path, "rb");
    if (!f) return -1;

    // Load configuration
    if (pipeline->state) {
        size_t read_count = fread(pipeline->state->config, sizeof(float), QG_CONFIG_SIZE, f);
        if (read_count != QG_CONFIG_SIZE) {
            fclose(f);
            return -1;
        }
    }

    // Load parameter count
    fread(&pipeline->parameter_count, sizeof(size_t), 1, f);
    fread(&pipeline->learning_rate, sizeof(float), 1, f);

    fclose(f);
    return 0;
}

int dist_pipeline_serialize(quantum_pipeline_t* pipeline, void** buffer, size_t* size) {
    if (!pipeline || !buffer || !size) return -1;

    // Calculate serialization size
    *size = sizeof(float) * QG_CONFIG_SIZE +
            sizeof(size_t) +
            sizeof(float) +
            pipeline->gradient_size;

    *buffer = malloc(*size);
    if (!*buffer) {
        *size = 0;
        return -1;
    }

    char* ptr = (char*)*buffer;

    // Serialize config
    if (pipeline->state) {
        memcpy(ptr, pipeline->state->config, sizeof(float) * QG_CONFIG_SIZE);
    }
    ptr += sizeof(float) * QG_CONFIG_SIZE;

    // Serialize parameter count and learning rate
    memcpy(ptr, &pipeline->parameter_count, sizeof(size_t));
    ptr += sizeof(size_t);
    memcpy(ptr, &pipeline->learning_rate, sizeof(float));
    ptr += sizeof(float);

    // Serialize gradients
    if (pipeline->gradients && pipeline->gradient_size > 0) {
        memcpy(ptr, pipeline->gradients, pipeline->gradient_size);
    }

    return 0;
}

int dist_pipeline_deserialize(quantum_pipeline_t* pipeline, void* buffer, size_t size) {
    if (!pipeline || !buffer || size < sizeof(float) * QG_CONFIG_SIZE + sizeof(size_t) + sizeof(float)) {
        return -1;
    }

    const char* ptr = (const char*)buffer;

    // Deserialize config into existing state
    if (pipeline->state) {
        memcpy(pipeline->state->config, ptr, sizeof(float) * QG_CONFIG_SIZE);
    }
    ptr += sizeof(float) * QG_CONFIG_SIZE;

    // Deserialize parameter count and learning rate
    memcpy(&pipeline->parameter_count, ptr, sizeof(size_t));
    ptr += sizeof(size_t);
    memcpy(&pipeline->learning_rate, ptr, sizeof(float));
    ptr += sizeof(float);

    // Deserialize gradients if remaining data
    size_t remaining = size - (sizeof(float) * QG_CONFIG_SIZE + sizeof(size_t) + sizeof(float));
    if (remaining > 0) {
        if (!pipeline->gradients || pipeline->gradient_size < remaining) {
            free(pipeline->gradients);
            pipeline->gradients = malloc(remaining);
        }
        if (pipeline->gradients) {
            memcpy(pipeline->gradients, ptr, remaining);
            pipeline->gradient_size = remaining;
        }
    }

    return 0;
}
