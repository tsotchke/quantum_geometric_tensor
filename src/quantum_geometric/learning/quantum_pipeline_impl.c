#include "quantum_geometric/learning/quantum_pipeline.h"
#include "quantum_geometric/distributed/distributed_training_manager.h"
#include "quantum_geometric/core/computational_graph.h"
#include "quantum_geometric/core/quantum_scheduler.h"
#include "quantum_geometric/core/operation_fusion.h"
#include "quantum_geometric/core/quantum_geometric_logging.h"
#include "quantum_geometric/learning/learning_task.h"
#include "quantum_geometric/core/quantum_complex.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <time.h>

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/task.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

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

    geometric_log_debug("Creating pipeline state...");
    quantum_pipeline_state_t* state = calloc(1, sizeof(quantum_pipeline_state_t));
    if (!state) {
        geometric_log_error("Failed to allocate pipeline state");
        return NULL;
    }
    
    // Store configuration
    memcpy(state->config, config, QG_CONFIG_SIZE * sizeof(float));
    
    // Initialize geometric processor
    geometric_log_debug("Creating geometric processor...");
    geometric_processor_t* processor = create_geometric_processor(NULL);
    if (!processor) {
        geometric_log_error("Failed to create geometric processor");
        free(state);
        return NULL;
    }
    
    // Create computational graphs
    geometric_log_debug("Creating computational graphs...");
    state->model_graph = create_computational_graph(processor);
    state->optimizer_graph = create_computational_graph(processor);

    if (!state->model_graph || !state->optimizer_graph) {
        geometric_log_error("Failed to create computational graphs");
        if (state->model_graph) destroy_computational_graph(state->model_graph);
        if (state->optimizer_graph) destroy_computational_graph(state->optimizer_graph);
        free(state);
        return NULL;
    }
    
    // Initialize learning task
    geometric_log_debug("Initializing learning task...");
    geometric_log_debug("input_dim=%zu, output_dim=%zu, latent_dim=%zu",
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
        geometric_log_error("Failed to create learning task");
        destroy_computational_graph(state->model_graph);
        destroy_computational_graph(state->optimizer_graph);
        free(state);
        return NULL;
    }

    geometric_log_info("Pipeline created successfully");
    return state;
}

int quantum_pipeline_train_impl(void* pipeline, const float* data, const int* labels, size_t num_samples) {
    quantum_pipeline_state_t* state = (quantum_pipeline_state_t*)pipeline;
    if (!state || !data || !labels || num_samples == 0) {
        geometric_log_error("Invalid arguments in pipeline_train");
        return QG_ERROR_INVALID_ARGUMENT;
    }

    geometric_log_debug("Converting data to complex format...");
    // Convert data to complex format
    const size_t input_dim = (size_t)state->config[QG_CONFIG_INPUT_DIM];
    ComplexFloat** complex_data = malloc(num_samples * sizeof(ComplexFloat*));
    if (!complex_data) {
        geometric_log_error("Failed to allocate complex_data array");
        return QG_ERROR_MEMORY_ALLOCATION;
    }

    for (size_t i = 0; i < num_samples; i++) {
        complex_data[i] = malloc(input_dim * sizeof(ComplexFloat));
        if (!complex_data[i]) {
            geometric_log_error("Failed to allocate complex_data[%zu]", i);
            for (size_t j = 0; j < i; j++) free(complex_data[j]);
            free(complex_data);
            return QG_ERROR_MEMORY_ALLOCATION;
        }
        for (size_t j = 0; j < input_dim; j++) {
            complex_data[i][j] = (ComplexFloat){data[i * input_dim + j], 0.0f};
        }
    }

    geometric_log_debug("Converting labels to complex format...");
    // Convert labels to complex format
    const size_t output_dim = (size_t)state->config[QG_CONFIG_NUM_CLASSES];
    ComplexFloat* complex_labels = malloc(num_samples * output_dim * sizeof(ComplexFloat));
    if (!complex_labels) {
        geometric_log_error("Failed to allocate complex_labels");
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

    geometric_log_debug("Starting training with learning task...");
    // Train using learning task interface
    int result = quantum_train_task(state->learning_task,
                                  (const ComplexFloat**)complex_data,
                                  complex_labels,
                                  num_samples);

    geometric_log_debug("Training result: %d", result);
    
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
        geometric_log_error("Invalid arguments in pipeline_evaluate");
        return QG_ERROR_INVALID_ARGUMENT;
    }

    geometric_log_debug("Converting data for evaluation...");
    // Convert data to complex format
    const size_t input_dim = (size_t)state->config[QG_CONFIG_INPUT_DIM];
    ComplexFloat** complex_data = malloc(num_samples * sizeof(ComplexFloat*));
    if (!complex_data) {
        geometric_log_error("Failed to allocate complex_data array");
        return QG_ERROR_MEMORY_ALLOCATION;
    }

    for (size_t i = 0; i < num_samples; i++) {
        complex_data[i] = malloc(input_dim * sizeof(ComplexFloat));
        if (!complex_data[i]) {
            geometric_log_error("Failed to allocate complex_data[%zu]", i);
            for (size_t j = 0; j < i; j++) free(complex_data[j]);
            free(complex_data);
            return QG_ERROR_MEMORY_ALLOCATION;
        }
        for (size_t j = 0; j < input_dim; j++) {
            complex_data[i][j] = (ComplexFloat){data[i * input_dim + j], 0.0f};
        }
    }

    geometric_log_debug("Converting labels for evaluation...");
    // Convert labels to complex format
    const size_t output_dim = (size_t)state->config[QG_CONFIG_NUM_CLASSES];
    ComplexFloat* complex_labels = malloc(num_samples * output_dim * sizeof(ComplexFloat));
    if (!complex_labels) {
        geometric_log_error("Failed to allocate complex_labels");
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

    geometric_log_debug("Starting evaluation...");
    // Evaluate using learning task interface
    task_metrics_t task_metrics;
    int result = quantum_evaluate_task(state->learning_task,
                                     (const ComplexFloat**)complex_data,
                                     complex_labels,
                                     num_samples,
                                     &task_metrics);

    geometric_log_debug("Evaluation result: %d", result);
    
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

// Magic number for pipeline file format: "QGTL" as uint32
#define PIPELINE_MAGIC 0x4C544751
#define PIPELINE_VERSION 1

int quantum_pipeline_save_impl(void* pipeline, const char* filename) {
    if (!pipeline || !filename) {
        geometric_log_error("Invalid arguments to pipeline save");
        return QG_ERROR_INVALID_ARGUMENT;
    }

    quantum_pipeline_state_t* state = (quantum_pipeline_state_t*)pipeline;

    FILE* f = fopen(filename, "wb");
    if (!f) {
        geometric_log_error("Failed to open file for writing: %s", filename);
        return QGT_ERROR_IO;
    }

    // Write header
    uint32_t magic = PIPELINE_MAGIC;
    uint32_t version = PIPELINE_VERSION;
    if (fwrite(&magic, sizeof(uint32_t), 1, f) != 1 ||
        fwrite(&version, sizeof(uint32_t), 1, f) != 1) {
        geometric_log_error("Failed to write header");
        fclose(f);
        return QGT_ERROR_IO;
    }

    // Write configuration
    if (fwrite(state->config, sizeof(float), QG_CONFIG_SIZE, f) != QG_CONFIG_SIZE) {
        geometric_log_error("Failed to write configuration");
        fclose(f);
        return QGT_ERROR_IO;
    }

    // Write training state
    if (fwrite(&state->current_epoch, sizeof(size_t), 1, f) != 1 ||
        fwrite(&state->current_loss, sizeof(float), 1, f) != 1 ||
        fwrite(&state->current_accuracy, sizeof(float), 1, f) != 1) {
        geometric_log_error("Failed to write training state");
        fclose(f);
        return QGT_ERROR_IO;
    }

    // Save learning task parameters if available
    if (state->learning_task) {
        size_t num_params = 0;
        float* params = NULL;

        // Get parameters from learning task
        if (quantum_get_task_parameters(state->learning_task, &params, &num_params) && params) {
            // Write parameter count and data
            if (fwrite(&num_params, sizeof(size_t), 1, f) != 1) {
                geometric_log_error("Failed to write parameter count");
                free(params);
                fclose(f);
                return QGT_ERROR_IO;
            }

            if (num_params > 0 && fwrite(params, sizeof(float), num_params, f) != num_params) {
                geometric_log_error("Failed to write parameters");
                free(params);
                fclose(f);
                return QGT_ERROR_IO;
            }
            free(params);
        } else {
            // No parameters - write zero count
            num_params = 0;
            if (fwrite(&num_params, sizeof(size_t), 1, f) != 1) {
                fclose(f);
                return QGT_ERROR_IO;
            }
        }
    } else {
        size_t zero = 0;
        if (fwrite(&zero, sizeof(size_t), 1, f) != 1) {
            fclose(f);
            return QGT_ERROR_IO;
        }
    }

    fclose(f);
    geometric_log_info("Pipeline saved to %s", filename);
    return QG_SUCCESS;
}

void quantum_pipeline_destroy_impl(void* pipeline) {
    quantum_pipeline_state_t* state = (quantum_pipeline_state_t*)pipeline;
    if (!state) return;

    geometric_log_debug("Destroying pipeline...");

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
    geometric_log_debug("Pipeline destroyed");
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
    if (!filename) {
        geometric_log_error("Null filename in pipeline load");
        return NULL;
    }

    FILE* f = fopen(filename, "rb");
    if (!f) {
        geometric_log_error("Failed to open file for reading: %s", filename);
        return NULL;
    }

    // Read and validate header
    uint32_t magic, version;
    if (fread(&magic, sizeof(uint32_t), 1, f) != 1 ||
        fread(&version, sizeof(uint32_t), 1, f) != 1) {
        geometric_log_error("Failed to read header");
        fclose(f);
        return NULL;
    }

    if (magic != PIPELINE_MAGIC) {
        geometric_log_error("Invalid file format (magic: 0x%X, expected: 0x%X)", magic, PIPELINE_MAGIC);
        fclose(f);
        return NULL;
    }

    if (version > PIPELINE_VERSION) {
        geometric_log_error("Unsupported file version: %u (max: %u)", version, PIPELINE_VERSION);
        fclose(f);
        return NULL;
    }

    // Read configuration
    float config[QG_CONFIG_SIZE];
    if (fread(config, sizeof(float), QG_CONFIG_SIZE, f) != QG_CONFIG_SIZE) {
        geometric_log_error("Failed to read configuration");
        fclose(f);
        return NULL;
    }

    // Read training state
    size_t saved_epoch;
    float saved_loss, saved_accuracy;
    if (fread(&saved_epoch, sizeof(size_t), 1, f) != 1 ||
        fread(&saved_loss, sizeof(float), 1, f) != 1 ||
        fread(&saved_accuracy, sizeof(float), 1, f) != 1) {
        geometric_log_error("Failed to read training state");
        fclose(f);
        return NULL;
    }

    // Read parameters
    size_t num_params = 0;
    float* params = NULL;
    if (fread(&num_params, sizeof(size_t), 1, f) != 1) {
        geometric_log_error("Failed to read parameter count");
        fclose(f);
        return NULL;
    }

    if (num_params > 0) {
        params = malloc(num_params * sizeof(float));
        if (!params) {
            geometric_log_error("Failed to allocate parameter buffer");
            fclose(f);
            return NULL;
        }

        if (fread(params, sizeof(float), num_params, f) != num_params) {
            geometric_log_error("Failed to read parameters");
            free(params);
            fclose(f);
            return NULL;
        }
    }

    fclose(f);

    // Create pipeline with loaded configuration
    quantum_pipeline_handle_t pipeline = quantum_pipeline_create(config);
    if (!pipeline) {
        geometric_log_error("Failed to create pipeline from loaded config");
        free(params);
        return NULL;
    }

    // Restore training state
    quantum_pipeline_state_t* state = (quantum_pipeline_state_t*)pipeline;
    state->current_epoch = saved_epoch;
    state->current_loss = saved_loss;
    state->current_accuracy = saved_accuracy;

    // Restore parameters if available
    if (params && num_params > 0 && state->learning_task) {
        quantum_set_task_parameters(state->learning_task, params, num_params);
    }
    free(params);

    geometric_log_info("Pipeline loaded from %s (epoch %zu, loss %.4f, accuracy %.4f)",
                       filename, saved_epoch, saved_loss, saved_accuracy);

    return pipeline;
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

// ============================================================================
// Classical MLP Baseline Implementation for Comparison
// ============================================================================

typedef struct {
    float** weights;      // weights[layer][input * output_dim + output]
    float** biases;       // biases[layer][neuron]
    size_t* layer_sizes;  // Number of neurons in each layer
    size_t num_layers;    // Total layers (including input/output)
    float** activations;  // Cache for forward pass
    float** gradients;    // Gradients for each layer
} ClassicalMLP;

static float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

static float relu_derivative(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

static void softmax(float* values, size_t size) {
    float max_val = values[0];
    for (size_t i = 1; i < size; i++) {
        if (values[i] > max_val) max_val = values[i];
    }

    float sum = 0.0f;
    for (size_t i = 0; i < size; i++) {
        values[i] = expf(values[i] - max_val);  // Subtract max for numerical stability
        sum += values[i];
    }

    for (size_t i = 0; i < size; i++) {
        values[i] /= sum;
    }
}

static ClassicalMLP* create_classical_mlp(size_t input_dim, size_t hidden_dim,
                                          size_t output_dim, size_t num_hidden_layers) {
    ClassicalMLP* mlp = calloc(1, sizeof(ClassicalMLP));
    if (!mlp) return NULL;

    // Total layers = input + hidden + output
    mlp->num_layers = num_hidden_layers + 2;
    mlp->layer_sizes = malloc(mlp->num_layers * sizeof(size_t));
    if (!mlp->layer_sizes) {
        free(mlp);
        return NULL;
    }

    mlp->layer_sizes[0] = input_dim;
    for (size_t i = 1; i < mlp->num_layers - 1; i++) {
        mlp->layer_sizes[i] = hidden_dim;
    }
    mlp->layer_sizes[mlp->num_layers - 1] = output_dim;

    // Allocate weights and biases (num_layers - 1 sets of weights)
    mlp->weights = calloc(mlp->num_layers - 1, sizeof(float*));
    mlp->biases = calloc(mlp->num_layers - 1, sizeof(float*));
    mlp->activations = calloc(mlp->num_layers, sizeof(float*));
    mlp->gradients = calloc(mlp->num_layers - 1, sizeof(float*));

    if (!mlp->weights || !mlp->biases || !mlp->activations || !mlp->gradients) {
        goto cleanup;
    }

    for (size_t i = 0; i < mlp->num_layers - 1; i++) {
        size_t fan_in = mlp->layer_sizes[i];
        size_t fan_out = mlp->layer_sizes[i + 1];

        mlp->weights[i] = malloc(fan_in * fan_out * sizeof(float));
        mlp->biases[i] = calloc(fan_out, sizeof(float));
        mlp->gradients[i] = calloc(fan_in * fan_out + fan_out, sizeof(float));

        if (!mlp->weights[i] || !mlp->biases[i] || !mlp->gradients[i]) {
            goto cleanup;
        }

        // Xavier/Glorot initialization
        float scale = sqrtf(6.0f / (float)(fan_in + fan_out));
        for (size_t j = 0; j < fan_in * fan_out; j++) {
            mlp->weights[i][j] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
        }
    }

    for (size_t i = 0; i < mlp->num_layers; i++) {
        mlp->activations[i] = malloc(mlp->layer_sizes[i] * sizeof(float));
        if (!mlp->activations[i]) {
            goto cleanup;
        }
    }

    return mlp;

cleanup:
    if (mlp->weights) {
        for (size_t i = 0; i < mlp->num_layers - 1; i++) {
            free(mlp->weights[i]);
        }
        free(mlp->weights);
    }
    if (mlp->biases) {
        for (size_t i = 0; i < mlp->num_layers - 1; i++) {
            free(mlp->biases[i]);
        }
        free(mlp->biases);
    }
    if (mlp->activations) {
        for (size_t i = 0; i < mlp->num_layers; i++) {
            free(mlp->activations[i]);
        }
        free(mlp->activations);
    }
    if (mlp->gradients) {
        for (size_t i = 0; i < mlp->num_layers - 1; i++) {
            free(mlp->gradients[i]);
        }
        free(mlp->gradients);
    }
    free(mlp->layer_sizes);
    free(mlp);
    return NULL;
}

static void destroy_classical_mlp(ClassicalMLP* mlp) {
    if (!mlp) return;

    for (size_t i = 0; i < mlp->num_layers - 1; i++) {
        free(mlp->weights[i]);
        free(mlp->biases[i]);
        free(mlp->gradients[i]);
    }
    for (size_t i = 0; i < mlp->num_layers; i++) {
        free(mlp->activations[i]);
    }
    free(mlp->weights);
    free(mlp->biases);
    free(mlp->activations);
    free(mlp->gradients);
    free(mlp->layer_sizes);
    free(mlp);
}

static void mlp_forward(ClassicalMLP* mlp, const float* input) {
    // Copy input to first activation
    memcpy(mlp->activations[0], input, mlp->layer_sizes[0] * sizeof(float));

    // Forward through each layer
    for (size_t l = 0; l < mlp->num_layers - 1; l++) {
        size_t in_size = mlp->layer_sizes[l];
        size_t out_size = mlp->layer_sizes[l + 1];

        for (size_t j = 0; j < out_size; j++) {
            float sum = mlp->biases[l][j];
            for (size_t i = 0; i < in_size; i++) {
                sum += mlp->activations[l][i] * mlp->weights[l][i * out_size + j];
            }
            mlp->activations[l + 1][j] = sum;
        }

        // Apply activation (ReLU for hidden, softmax for output)
        if (l < mlp->num_layers - 2) {
            for (size_t j = 0; j < out_size; j++) {
                mlp->activations[l + 1][j] = relu(mlp->activations[l + 1][j]);
            }
        } else {
            softmax(mlp->activations[l + 1], out_size);
        }
    }
}

static float mlp_backward(ClassicalMLP* mlp, const float* target, float learning_rate) {
    size_t output_size = mlp->layer_sizes[mlp->num_layers - 1];
    float* output = mlp->activations[mlp->num_layers - 1];

    // Compute cross-entropy loss and output gradients
    float loss = 0.0f;
    float* delta = malloc(output_size * sizeof(float));
    if (!delta) return loss;

    for (size_t i = 0; i < output_size; i++) {
        float pred = fmaxf(output[i], 1e-7f);
        loss -= target[i] * logf(pred);
        delta[i] = output[i] - target[i];  // Softmax + cross-entropy derivative
    }

    // Backpropagate through layers
    for (int l = (int)mlp->num_layers - 2; l >= 0; l--) {
        size_t in_size = mlp->layer_sizes[l];
        size_t out_size = mlp->layer_sizes[l + 1];

        float* next_delta = NULL;
        if (l > 0) {
            next_delta = calloc(in_size, sizeof(float));
            if (!next_delta) {
                free(delta);
                return loss;
            }
        }

        // Compute gradients and backpropagate
        for (size_t j = 0; j < out_size; j++) {
            // Bias gradient
            mlp->biases[l][j] -= learning_rate * delta[j];

            for (size_t i = 0; i < in_size; i++) {
                // Weight gradient
                float grad = delta[j] * mlp->activations[l][i];
                mlp->weights[l][i * out_size + j] -= learning_rate * grad;

                // Propagate delta to previous layer
                if (next_delta) {
                    next_delta[i] += delta[j] * mlp->weights[l][i * out_size + j];
                }
            }
        }

        // Apply ReLU derivative to delta for hidden layers
        if (next_delta && l > 0) {
            for (size_t i = 0; i < in_size; i++) {
                next_delta[i] *= relu_derivative(mlp->activations[l][i]);
            }
        }

        free(delta);
        delta = next_delta;
    }

    free(delta);
    return loss;
}

static int mlp_predict(ClassicalMLP* mlp, const float* input) {
    mlp_forward(mlp, input);

    size_t output_size = mlp->layer_sizes[mlp->num_layers - 1];
    float* output = mlp->activations[mlp->num_layers - 1];

    int prediction = 0;
    float max_prob = output[0];
    for (size_t i = 1; i < output_size; i++) {
        if (output[i] > max_prob) {
            max_prob = output[i];
            prediction = (int)i;
        }
    }

    return prediction;
}

static size_t get_memory_usage(void) {
#ifdef __APPLE__
    struct mach_task_basic_info info;
    mach_msg_type_number_t size = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &size) == KERN_SUCCESS) {
        return info.resident_size;
    }
#elif defined(__linux__)
    FILE* fp = fopen("/proc/self/statm", "r");
    if (fp) {
        unsigned long size, resident;
        if (fscanf(fp, "%lu %lu", &size, &resident) == 2) {
            fclose(fp);
            return resident * sysconf(_SC_PAGESIZE);
        }
        fclose(fp);
    }
#endif
    return 0;
}

int quantum_pipeline_compare_classical(quantum_pipeline_handle_t pipeline,
                                      const float* data,
                                      const int* labels,
                                      size_t num_samples,
                                      float* results) {
    if (!pipeline || !data || !labels || num_samples == 0 || !results) {
        return QG_ERROR_INVALID_ARGUMENT;
    }

    quantum_pipeline_state_t* state = (quantum_pipeline_state_t*)pipeline;

    // Get dimensions from pipeline config
    size_t input_dim = (size_t)state->config[QG_CONFIG_INPUT_DIM];
    size_t output_dim = (size_t)state->config[QG_CONFIG_NUM_CLASSES];
    size_t num_qubits = (size_t)state->config[QG_CONFIG_NUM_QUBITS];
    size_t num_layers = (size_t)state->config[QG_CONFIG_NUM_LAYERS];
    float learning_rate = state->config[QG_CONFIG_LEARNING_RATE];
    size_t batch_size = (size_t)state->config[QG_CONFIG_BATCH_SIZE];

    // Create a classical MLP with comparable capacity to the quantum circuit
    // Quantum circuit has ~num_qubits * num_layers * 3 parameters
    // Use hidden dimension to achieve similar parameter count
    size_t quantum_params = num_qubits * num_layers * 3;
    size_t hidden_dim = (size_t)sqrtf((float)quantum_params);
    if (hidden_dim < 32) hidden_dim = 32;
    if (hidden_dim > 256) hidden_dim = 256;

    ClassicalMLP* mlp = create_classical_mlp(input_dim, hidden_dim, output_dim, 2);
    if (!mlp) {
        geometric_log_error("Failed to create classical MLP for comparison");
        return QG_ERROR_MEMORY_ALLOCATION;
    }

    // Record start time and memory
    size_t start_memory = get_memory_usage();
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    // Convert one-hot labels back to class indices for comparison
    int* class_labels = malloc(num_samples * sizeof(int));
    if (!class_labels) {
        destroy_classical_mlp(mlp);
        return QG_ERROR_MEMORY_ALLOCATION;
    }

    for (size_t i = 0; i < num_samples; i++) {
        class_labels[i] = 0;
        for (size_t j = 0; j < output_dim; j++) {
            if (labels[i * output_dim + j] > 0) {
                class_labels[i] = (int)j;
                break;
            }
        }
    }

    // Create float target buffer
    float* target = malloc(output_dim * sizeof(float));
    if (!target) {
        free(class_labels);
        destroy_classical_mlp(mlp);
        return QG_ERROR_MEMORY_ALLOCATION;
    }

    // Training loop - use same number of epochs as quantum (default 10)
    size_t num_epochs = 10;

    for (size_t epoch = 0; epoch < num_epochs; epoch++) {
        // Mini-batch SGD
        for (size_t batch_start = 0; batch_start < num_samples; batch_start += batch_size) {
            size_t batch_end = batch_start + batch_size;
            if (batch_end > num_samples) batch_end = num_samples;

            for (size_t i = batch_start; i < batch_end; i++) {
                const float* sample = data + i * input_dim;

                // Create one-hot target
                memset(target, 0, output_dim * sizeof(float));
                target[class_labels[i]] = 1.0f;

                // Forward and backward pass
                mlp_forward(mlp, sample);
                mlp_backward(mlp, target, learning_rate);
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end_time);
    size_t end_memory = get_memory_usage();

    // Calculate training time
    double elapsed_time = (double)(end_time.tv_sec - start_time.tv_sec) +
                         (double)(end_time.tv_nsec - start_time.tv_nsec) / 1e9;

    // Evaluate accuracy on the same data (training accuracy for comparison)
    size_t correct = 0;
    for (size_t i = 0; i < num_samples; i++) {
        const float* sample = data + i * input_dim;
        int prediction = mlp_predict(mlp, sample);
        if (prediction == class_labels[i]) {
            correct++;
        }
    }
    float accuracy = (float)correct / (float)num_samples;

    // Calculate memory used (in MB)
    float memory_mb = (float)(end_memory - start_memory) / (1024.0f * 1024.0f);
    if (memory_mb < 0) memory_mb = 0;  // In case of measurement issues

    // Add estimated MLP memory: weights + biases + activations
    size_t mlp_memory = 0;
    for (size_t i = 0; i < mlp->num_layers - 1; i++) {
        mlp_memory += mlp->layer_sizes[i] * mlp->layer_sizes[i + 1] * sizeof(float);  // weights
        mlp_memory += mlp->layer_sizes[i + 1] * sizeof(float);  // biases
    }
    for (size_t i = 0; i < mlp->num_layers; i++) {
        mlp_memory += mlp->layer_sizes[i] * sizeof(float);  // activations
    }
    memory_mb += (float)mlp_memory / (1024.0f * 1024.0f);

    // Store results
    results[0] = accuracy;              // Classical accuracy
    results[1] = (float)elapsed_time;   // Training time in seconds
    results[2] = memory_mb;             // Memory usage in MB

    geometric_log_info("Classical comparison: accuracy=%.4f, time=%.3fs, memory=%.2fMB",
                       accuracy, elapsed_time, memory_mb);

    // Cleanup
    free(target);
    free(class_labels);
    destroy_classical_mlp(mlp);

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

// Types gradient_method_t and gradient_config_t are defined in
// distributed_training_manager.h which is included via quantum_pipeline.h

// Define struct quantum_pipeline_t for distributed training
typedef struct quantum_pipeline_t {
    quantum_pipeline_state_t* state;
    void* gradients;
    size_t gradient_size;
    size_t parameter_count;
    float learning_rate;
    void* cached_output;
    size_t cached_output_size;

    // Data storage for gradient computation
    float* cached_input;              // Input data from forward pass
    size_t cached_input_size;
    float* cached_labels;             // Labels/targets for loss computation
    size_t cached_labels_size;
    size_t current_batch_size;

    // Gradient estimation configuration
    gradient_config_t gradient_config;

    // Parameter buffers for gradient computation
    float* original_params;           // Original parameters before shifting
    float* shifted_params;            // Temporary buffer for shifted parameters
    size_t params_buffer_size;

    // Loss tracking
    float current_loss;
    float* loss_history;
    size_t loss_history_size;
    size_t loss_history_capacity;
} quantum_pipeline_t;

// Forward declarations for functions used before they are defined
size_t dist_pipeline_get_parameter_count(quantum_pipeline_t* pipeline);
int dist_pipeline_set_labels(quantum_pipeline_t* pipeline, const void* labels, size_t batch_size);

int dist_pipeline_forward(quantum_pipeline_t* pipeline, const void* data, size_t batch_size) {
    if (!pipeline || !data || batch_size == 0) return -1;
    if (!pipeline->state || !pipeline->state->learning_task) return -1;

    const size_t input_dim = (size_t)pipeline->state->config[QG_CONFIG_INPUT_DIM];
    const size_t output_dim = (size_t)pipeline->state->config[QG_CONFIG_NUM_CLASSES];
    const float* input_data = (const float*)data;

    // Cache input data for gradient computation
    size_t input_size = batch_size * input_dim * sizeof(float);
    if (pipeline->cached_input_size < input_size) {
        free(pipeline->cached_input);
        pipeline->cached_input = malloc(input_size);
        if (!pipeline->cached_input) {
            pipeline->cached_input_size = 0;
            return -1;
        }
        pipeline->cached_input_size = input_size;
    }
    memcpy(pipeline->cached_input, input_data, input_size);
    pipeline->current_batch_size = batch_size;

    // Allocate output buffer if needed
    size_t output_size = batch_size * output_dim * sizeof(float);
    if (pipeline->cached_output_size < output_size) {
        free(pipeline->cached_output);
        pipeline->cached_output = malloc(output_size);
        if (!pipeline->cached_output) {
            pipeline->cached_output_size = 0;
            return -1;
        }
        pipeline->cached_output_size = output_size;
    }

    // Convert input to complex format for quantum processing
    ComplexFloat* complex_input = malloc(input_dim * sizeof(ComplexFloat));
    ComplexFloat* complex_output = malloc(output_dim * sizeof(ComplexFloat));
    if (!complex_input || !complex_output) {
        free(complex_input);
        free(complex_output);
        return -1;
    }

    float* output = (float*)pipeline->cached_output;

    // Process each sample through the quantum learning task
    for (size_t i = 0; i < batch_size; i++) {
        const float* sample = input_data + i * input_dim;
        float* sample_output = output + i * output_dim;

        // Convert to complex format
        for (size_t j = 0; j < input_dim; j++) {
            complex_input[j] = (ComplexFloat){sample[j], 0.0f};
        }

        // Run quantum forward pass through the learning task
        bool success = quantum_predict_task(pipeline->state->learning_task,
                                           complex_input,
                                           complex_output);

        if (success) {
            // Convert output back to float (magnitude)
            for (size_t j = 0; j < output_dim; j++) {
                sample_output[j] = complex_output[j].real;
            }
        } else {
            // Fallback if quantum prediction fails
            for (size_t j = 0; j < output_dim; j++) {
                sample_output[j] = 0.0f;
            }
        }
    }

    free(complex_input);
    free(complex_output);

    return 0;
}

// Set labels for loss computation
int dist_pipeline_set_labels(quantum_pipeline_t* pipeline, const void* labels, size_t batch_size) {
    if (!pipeline || !labels || batch_size == 0) return -1;
    if (!pipeline->state) return -1;

    const size_t output_dim = (size_t)pipeline->state->config[QG_CONFIG_NUM_CLASSES];
    size_t labels_size = batch_size * output_dim * sizeof(float);

    if (pipeline->cached_labels_size < labels_size) {
        free(pipeline->cached_labels);
        pipeline->cached_labels = malloc(labels_size);
        if (!pipeline->cached_labels) {
            pipeline->cached_labels_size = 0;
            return -1;
        }
        pipeline->cached_labels_size = labels_size;
    }

    memcpy(pipeline->cached_labels, labels, labels_size);
    return 0;
}

// Compute MSE loss between predictions and labels
static float compute_mse_loss(const float* predictions, const float* labels,
                              size_t batch_size, size_t output_dim) {
    float total_loss = 0.0f;
    for (size_t i = 0; i < batch_size * output_dim; i++) {
        float diff = predictions[i] - labels[i];
        total_loss += diff * diff;
    }
    return total_loss / (float)(batch_size * output_dim);
}

// Compute cross-entropy loss for classification
static float compute_cross_entropy_loss(const float* predictions, const float* labels,
                                         size_t batch_size, size_t output_dim) {
    float total_loss = 0.0f;
    const float epsilon = 1e-7f;  // Numerical stability

    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < output_dim; j++) {
            size_t idx = i * output_dim + j;
            float pred = predictions[idx];
            // Clamp predictions for numerical stability
            if (pred < epsilon) pred = epsilon;
            if (pred > 1.0f - epsilon) pred = 1.0f - epsilon;

            if (labels[idx] > 0.5f) {
                total_loss -= logf(pred);
            }
        }
    }
    return total_loss / (float)batch_size;
}

// Evaluate loss with given parameters
static float evaluate_loss_with_params(quantum_pipeline_t* pipeline,
                                       const float* params,
                                       size_t num_params) {
    if (!pipeline || !params || !pipeline->cached_input || !pipeline->cached_labels) {
        return INFINITY;
    }

    // Set the parameters
    if (!quantum_set_task_parameters(pipeline->state->learning_task,
                                     params, num_params)) {
        return INFINITY;
    }

    const size_t input_dim = (size_t)pipeline->state->config[QG_CONFIG_INPUT_DIM];
    const size_t output_dim = (size_t)pipeline->state->config[QG_CONFIG_NUM_CLASSES];
    size_t batch_size = pipeline->current_batch_size;

    // Allocate temporary output buffer
    float* temp_output = malloc(batch_size * output_dim * sizeof(float));
    ComplexFloat* complex_input = malloc(input_dim * sizeof(ComplexFloat));
    ComplexFloat* complex_output = malloc(output_dim * sizeof(ComplexFloat));

    if (!temp_output || !complex_input || !complex_output) {
        free(temp_output);
        free(complex_input);
        free(complex_output);
        return INFINITY;
    }

    // Run forward pass with current parameters
    for (size_t i = 0; i < batch_size; i++) {
        const float* sample = pipeline->cached_input + i * input_dim;
        float* sample_output = temp_output + i * output_dim;

        for (size_t j = 0; j < input_dim; j++) {
            complex_input[j] = (ComplexFloat){sample[j], 0.0f};
        }

        bool success = quantum_predict_task(pipeline->state->learning_task,
                                           complex_input,
                                           complex_output);

        if (success) {
            for (size_t j = 0; j < output_dim; j++) {
                sample_output[j] = complex_output[j].real;
            }
        } else {
            for (size_t j = 0; j < output_dim; j++) {
                sample_output[j] = 0.0f;
            }
        }
    }

    // Compute loss
    float loss = compute_mse_loss(temp_output, pipeline->cached_labels,
                                  batch_size, output_dim);

    free(temp_output);
    free(complex_input);
    free(complex_output);

    return loss;
}

// Parameter shift rule gradient computation
// For each parameter θᵢ: ∂f/∂θᵢ = (f(θᵢ + π/2) - f(θᵢ - π/2)) / 2
static int compute_gradients_parameter_shift(quantum_pipeline_t* pipeline,
                                             float* gradients) {
    if (!pipeline || !gradients) return -1;

    float* params = NULL;
    size_t num_params = 0;

    // Get current parameters
    if (!quantum_get_task_parameters(pipeline->state->learning_task,
                                     &params, &num_params)) {
        return -1;
    }

    if (num_params == 0 || !params) {
        return -1;
    }

    // Ensure we have buffer space for original parameters
    if (pipeline->params_buffer_size < num_params * sizeof(float)) {
        free(pipeline->original_params);
        free(pipeline->shifted_params);
        pipeline->original_params = malloc(num_params * sizeof(float));
        pipeline->shifted_params = malloc(num_params * sizeof(float));
        pipeline->params_buffer_size = num_params * sizeof(float);
    }

    if (!pipeline->original_params || !pipeline->shifted_params) {
        free(params);
        return -1;
    }

    // Store original parameters
    memcpy(pipeline->original_params, params, num_params * sizeof(float));
    memcpy(pipeline->shifted_params, params, num_params * sizeof(float));

    const float shift = (float)(pipeline->gradient_config.shift_amount);

    // Compute gradient for each parameter using parameter shift rule
    for (size_t i = 0; i < num_params; i++) {
        // Evaluate at θᵢ + shift
        pipeline->shifted_params[i] = pipeline->original_params[i] + shift;
        float loss_plus = evaluate_loss_with_params(pipeline,
                                                    pipeline->shifted_params,
                                                    num_params);

        // Evaluate at θᵢ - shift
        pipeline->shifted_params[i] = pipeline->original_params[i] - shift;
        float loss_minus = evaluate_loss_with_params(pipeline,
                                                     pipeline->shifted_params,
                                                     num_params);

        // Restore original parameter value for next iteration
        pipeline->shifted_params[i] = pipeline->original_params[i];

        // Compute gradient: (f(θ + s) - f(θ - s)) / (2 * sin(s))
        // For shift = π/2, sin(π/2) = 1, so gradient = (f+ - f-) / 2
        float sin_shift = sinf(shift);
        if (fabsf(sin_shift) < 1e-7f) {
            sin_shift = 1.0f;  // Avoid division by zero
        }
        gradients[i] = (loss_plus - loss_minus) / (2.0f * sin_shift);
    }

    // Restore original parameters
    quantum_set_task_parameters(pipeline->state->learning_task,
                                pipeline->original_params, num_params);

    free(params);
    return 0;
}

// SPSA gradient estimation
// More efficient for many parameters: O(2) evaluations instead of O(2n)
static int compute_gradients_spsa(quantum_pipeline_t* pipeline,
                                  float* gradients) {
    if (!pipeline || !gradients) return -1;

    float* params = NULL;
    size_t num_params = 0;

    if (!quantum_get_task_parameters(pipeline->state->learning_task,
                                     &params, &num_params)) {
        return -1;
    }

    if (num_params == 0 || !params) {
        return -1;
    }

    // Ensure we have buffer space
    if (pipeline->params_buffer_size < num_params * sizeof(float)) {
        free(pipeline->original_params);
        free(pipeline->shifted_params);
        pipeline->original_params = malloc(num_params * sizeof(float));
        pipeline->shifted_params = malloc(num_params * sizeof(float));
        pipeline->params_buffer_size = num_params * sizeof(float);
    }

    if (!pipeline->original_params || !pipeline->shifted_params) {
        free(params);
        return -1;
    }

    memcpy(pipeline->original_params, params, num_params * sizeof(float));

    // Allocate perturbation vector
    float* perturbation = malloc(num_params * sizeof(float));
    float* theta_plus = malloc(num_params * sizeof(float));
    float* theta_minus = malloc(num_params * sizeof(float));

    if (!perturbation || !theta_plus || !theta_minus) {
        free(perturbation);
        free(theta_plus);
        free(theta_minus);
        free(params);
        return -1;
    }

    // Initialize gradients to zero for averaging
    memset(gradients, 0, num_params * sizeof(float));

    const float c = (float)pipeline->gradient_config.spsa_perturbation;
    size_t num_samples = pipeline->gradient_config.spsa_averaging_samples;
    if (num_samples == 0) num_samples = 1;

    // Average over multiple SPSA samples for stability
    for (size_t sample = 0; sample < num_samples; sample++) {
        // Generate random perturbation (Rademacher distribution: +1 or -1)
        for (size_t i = 0; i < num_params; i++) {
            perturbation[i] = (rand() % 2 == 0) ? 1.0f : -1.0f;
            theta_plus[i] = pipeline->original_params[i] + c * perturbation[i];
            theta_minus[i] = pipeline->original_params[i] - c * perturbation[i];
        }

        // Evaluate at perturbed points
        float loss_plus = evaluate_loss_with_params(pipeline, theta_plus, num_params);
        float loss_minus = evaluate_loss_with_params(pipeline, theta_minus, num_params);

        // Estimate gradient
        float delta_loss = loss_plus - loss_minus;
        for (size_t i = 0; i < num_params; i++) {
            gradients[i] += delta_loss / (2.0f * c * perturbation[i]);
        }
    }

    // Average the gradient estimates
    for (size_t i = 0; i < num_params; i++) {
        gradients[i] /= (float)num_samples;
    }

    // Restore original parameters
    quantum_set_task_parameters(pipeline->state->learning_task,
                                pipeline->original_params, num_params);

    free(perturbation);
    free(theta_plus);
    free(theta_minus);
    free(params);
    return 0;
}

// Finite difference gradient estimation (fallback method)
static int compute_gradients_finite_difference(quantum_pipeline_t* pipeline,
                                               float* gradients) {
    if (!pipeline || !gradients) return -1;

    float* params = NULL;
    size_t num_params = 0;

    if (!quantum_get_task_parameters(pipeline->state->learning_task,
                                     &params, &num_params)) {
        return -1;
    }

    if (num_params == 0 || !params) {
        return -1;
    }

    // Ensure buffer space
    if (pipeline->params_buffer_size < num_params * sizeof(float)) {
        free(pipeline->original_params);
        free(pipeline->shifted_params);
        pipeline->original_params = malloc(num_params * sizeof(float));
        pipeline->shifted_params = malloc(num_params * sizeof(float));
        pipeline->params_buffer_size = num_params * sizeof(float);
    }

    if (!pipeline->original_params || !pipeline->shifted_params) {
        free(params);
        return -1;
    }

    memcpy(pipeline->original_params, params, num_params * sizeof(float));
    memcpy(pipeline->shifted_params, params, num_params * sizeof(float));

    const float epsilon = (float)pipeline->gradient_config.finite_diff_epsilon;

    // Central difference for each parameter
    for (size_t i = 0; i < num_params; i++) {
        pipeline->shifted_params[i] = pipeline->original_params[i] + epsilon;
        float loss_plus = evaluate_loss_with_params(pipeline,
                                                    pipeline->shifted_params,
                                                    num_params);

        pipeline->shifted_params[i] = pipeline->original_params[i] - epsilon;
        float loss_minus = evaluate_loss_with_params(pipeline,
                                                     pipeline->shifted_params,
                                                     num_params);

        pipeline->shifted_params[i] = pipeline->original_params[i];

        gradients[i] = (loss_plus - loss_minus) / (2.0f * epsilon);
    }

    // Restore original parameters
    quantum_set_task_parameters(pipeline->state->learning_task,
                                pipeline->original_params, num_params);

    free(params);
    return 0;
}

// Apply gradient clipping
// ============================================================================
// Quantum Fisher Information Matrix (QFIM) Natural Gradient
// ============================================================================

// Compute fidelity |<ψ(θ)|ψ(θ')>|² between two parameter configurations
static float compute_state_fidelity(quantum_pipeline_t* pipeline,
                                    const float* params1, const float* params2,
                                    size_t num_params) {
    // Get output dimension
    size_t output_dim = (size_t)pipeline->state->config[QG_CONFIG_NUM_CLASSES];
    float* probs1 = calloc(output_dim, sizeof(float));
    float* probs2 = calloc(output_dim, sizeof(float));

    if (!probs1 || !probs2) {
        free(probs1);
        free(probs2);
        return 0.0f;
    }

    // Run circuit with params1 to get output probabilities
    quantum_set_task_parameters(pipeline->state->learning_task, (float*)params1, num_params);
    dist_pipeline_forward(pipeline, pipeline->cached_input, pipeline->current_batch_size);
    if (pipeline->cached_output && pipeline->cached_output_size >= output_dim * sizeof(float)) {
        memcpy(probs1, pipeline->cached_output, output_dim * sizeof(float));
    }

    // Run circuit with params2 to get output probabilities
    quantum_set_task_parameters(pipeline->state->learning_task, (float*)params2, num_params);
    dist_pipeline_forward(pipeline, pipeline->cached_input, pipeline->current_batch_size);
    if (pipeline->cached_output && pipeline->cached_output_size >= output_dim * sizeof(float)) {
        memcpy(probs2, pipeline->cached_output, output_dim * sizeof(float));
    }

    // Compute fidelity as |<ψ1|ψ2>|² ≈ Σ_i √(p1_i * p2_i)
    // This is the Bhattacharyya coefficient for probability distributions
    float fidelity = 0.0f;
    for (size_t i = 0; i < output_dim; i++) {
        if (probs1[i] > 0 && probs2[i] > 0) {
            fidelity += sqrtf(probs1[i] * probs2[i]);
        }
    }
    fidelity *= fidelity;  // Square for fidelity

    free(probs1);
    free(probs2);

    return fidelity;
}

// Compute the Quantum Fisher Information Matrix using finite differences on fidelity
// F_ij = -2 * ∂²F/∂θ_i∂θ_j where F = |<ψ(θ)|ψ(θ')>|² evaluated at θ'=θ
static int compute_qfim(quantum_pipeline_t* pipeline, float* qfim, size_t num_params) {
    if (!pipeline || !qfim || num_params == 0) return -1;

    float* params = NULL;
    size_t actual_params = 0;

    if (!quantum_get_task_parameters(pipeline->state->learning_task, &params, &actual_params)) {
        return -1;
    }

    if (actual_params != num_params) {
        free(params);
        return -1;
    }

    // Allocate shifted parameter arrays
    float* params_pi = malloc(num_params * sizeof(float));
    float* params_pj = malloc(num_params * sizeof(float));
    float* params_pipj = malloc(num_params * sizeof(float));
    float* params_minj = malloc(num_params * sizeof(float));

    if (!params_pi || !params_pj || !params_pipj || !params_minj) {
        free(params);
        free(params_pi);
        free(params_pj);
        free(params_pipj);
        free(params_minj);
        return -1;
    }

    const float epsilon = 0.01f;  // Shift amount for QFIM estimation

    // Initialize QFIM to zero
    memset(qfim, 0, num_params * num_params * sizeof(float));

    // Compute QFIM elements using finite differences on fidelity
    // F_ij = -2 * (F(θ+εi+εj) - F(θ+εi-εj) - F(θ-εi+εj) + F(θ-εi-εj)) / (4ε²)
    for (size_t i = 0; i < num_params; i++) {
        for (size_t j = i; j < num_params; j++) {
            // Reset all shifted params to original
            memcpy(params_pipj, params, num_params * sizeof(float));
            memcpy(params_pi, params, num_params * sizeof(float));
            memcpy(params_pj, params, num_params * sizeof(float));
            memcpy(params_minj, params, num_params * sizeof(float));

            // θ + ε_i + ε_j
            params_pipj[i] += epsilon;
            params_pipj[j] += epsilon;

            // θ + ε_i - ε_j
            params_pi[i] += epsilon;
            params_pi[j] -= epsilon;

            // θ - ε_i + ε_j
            params_pj[i] -= epsilon;
            params_pj[j] += epsilon;

            // θ - ε_i - ε_j
            params_minj[i] -= epsilon;
            params_minj[j] -= epsilon;

            // Compute fidelities relative to original state
            float f_pp = compute_state_fidelity(pipeline, params, params_pipj, num_params);
            float f_pm = compute_state_fidelity(pipeline, params, params_pi, num_params);
            float f_mp = compute_state_fidelity(pipeline, params, params_pj, num_params);
            float f_mm = compute_state_fidelity(pipeline, params, params_minj, num_params);

            // Second derivative via finite differences
            // ∂²F/∂θ_i∂θ_j ≈ (f_pp - f_pm - f_mp + f_mm) / (4ε²)
            float d2f = (f_pp - f_pm - f_mp + f_mm) / (4.0f * epsilon * epsilon);

            // QFIM element: F_ij = -2 * ∂²ln(F)/∂θ_i∂θ_j ≈ -2 * ∂²F/∂θ_i∂θ_j (for F near 1)
            // For numerical stability, use F_ij = 2(1 - F) expansion
            float fij = -2.0f * d2f;

            // Ensure positive semi-definiteness
            if (i == j) {
                fij = fabsf(fij) + 1e-6f;  // Diagonal must be positive
            }

            qfim[i * num_params + j] = fij;
            if (i != j) {
                qfim[j * num_params + i] = fij;  // Symmetric matrix
            }
        }
    }

    // Restore original parameters
    quantum_set_task_parameters(pipeline->state->learning_task, params, num_params);

    free(params);
    free(params_pi);
    free(params_pj);
    free(params_pipj);
    free(params_minj);

    return 0;
}

// Invert QFIM with Tikhonov regularization: (F + λI)^{-1}
static int invert_qfim_regularized(const float* qfim, float* qfim_inv,
                                    size_t n, float lambda) {
    if (!qfim || !qfim_inv || n == 0) return -1;

    // Copy QFIM and add regularization
    float* A = malloc(n * n * sizeof(float));
    float* L = calloc(n * n, sizeof(float));  // Lower triangular for Cholesky

    if (!A || !L) {
        free(A);
        free(L);
        return -1;
    }

    for (size_t i = 0; i < n * n; i++) {
        A[i] = qfim[i];
    }

    // Add Tikhonov regularization: A = F + λI
    for (size_t i = 0; i < n; i++) {
        A[i * n + i] += lambda;
    }

    // Cholesky decomposition: A = L * L^T
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j <= i; j++) {
            float sum = A[i * n + j];

            for (size_t k = 0; k < j; k++) {
                sum -= L[i * n + k] * L[j * n + k];
            }

            if (i == j) {
                if (sum <= 0) {
                    // Not positive definite - increase regularization
                    free(A);
                    free(L);
                    return invert_qfim_regularized(qfim, qfim_inv, n, lambda * 10.0f);
                }
                L[i * n + j] = sqrtf(sum);
            } else {
                L[i * n + j] = sum / L[j * n + j];
            }
        }
    }

    // Solve L * L^T * X = I for X = A^{-1}
    // First solve L * Y = I for Y
    float* Y = calloc(n * n, sizeof(float));
    if (!Y) {
        free(A);
        free(L);
        return -1;
    }

    for (size_t col = 0; col < n; col++) {
        for (size_t i = 0; i < n; i++) {
            float sum = (i == col) ? 1.0f : 0.0f;
            for (size_t j = 0; j < i; j++) {
                sum -= L[i * n + j] * Y[j * n + col];
            }
            Y[i * n + col] = sum / L[i * n + i];
        }
    }

    // Then solve L^T * X = Y for X
    for (size_t col = 0; col < n; col++) {
        for (int i = (int)n - 1; i >= 0; i--) {
            float sum = Y[i * n + col];
            for (size_t j = i + 1; j < n; j++) {
                sum -= L[j * n + i] * qfim_inv[j * n + col];
            }
            qfim_inv[i * n + col] = sum / L[i * n + i];
        }
    }

    free(A);
    free(L);
    free(Y);

    return 0;
}

// Natural gradient computation using QFIM
static int compute_gradients_natural_gradient(quantum_pipeline_t* pipeline,
                                               float* gradients) {
    if (!pipeline || !gradients) return -1;

    // First compute standard gradients using parameter shift
    int result = compute_gradients_parameter_shift(pipeline, gradients);
    if (result != 0) return result;

    size_t num_params = pipeline->parameter_count;
    if (num_params == 0) return -1;

    // Allocate QFIM and its inverse
    float* qfim = malloc(num_params * num_params * sizeof(float));
    float* qfim_inv = malloc(num_params * num_params * sizeof(float));
    float* natural_grads = malloc(num_params * sizeof(float));

    if (!qfim || !qfim_inv || !natural_grads) {
        free(qfim);
        free(qfim_inv);
        free(natural_grads);
        return -1;
    }

    // Compute the Quantum Fisher Information Matrix
    result = compute_qfim(pipeline, qfim, num_params);
    if (result != 0) {
        // Fall back to standard gradient if QFIM computation fails
        geometric_log_warning("QFIM computation failed, using standard gradient");
        free(qfim);
        free(qfim_inv);
        free(natural_grads);
        return 0;
    }

    // Invert QFIM with regularization (λ = 0.01)
    float regularization = 0.01f;
    result = invert_qfim_regularized(qfim, qfim_inv, num_params, regularization);
    if (result != 0) {
        // Fall back to standard gradient
        geometric_log_warning("QFIM inversion failed, using standard gradient");
        free(qfim);
        free(qfim_inv);
        free(natural_grads);
        return 0;
    }

    // Compute natural gradient: g_nat = F^{-1} * g
    for (size_t i = 0; i < num_params; i++) {
        natural_grads[i] = 0.0f;
        for (size_t j = 0; j < num_params; j++) {
            natural_grads[i] += qfim_inv[i * num_params + j] * gradients[j];
        }
    }

    // Copy natural gradients back
    memcpy(gradients, natural_grads, num_params * sizeof(float));

    free(qfim);
    free(qfim_inv);
    free(natural_grads);

    return 0;
}

static void clip_gradients(float* gradients, size_t num_params, float clip_value) {
    float norm = 0.0f;
    for (size_t i = 0; i < num_params; i++) {
        norm += gradients[i] * gradients[i];
    }
    norm = sqrtf(norm);

    if (norm > clip_value) {
        float scale = clip_value / norm;
        for (size_t i = 0; i < num_params; i++) {
            gradients[i] *= scale;
        }
    }
}

int dist_pipeline_backward(quantum_pipeline_t* pipeline) {
    if (!pipeline) return -1;
    if (!pipeline->state || !pipeline->state->learning_task) return -1;

    // Ensure we have cached input and labels for gradient computation
    if (!pipeline->cached_input || !pipeline->cached_labels) {
        fprintf(stderr, "ERROR: dist_pipeline_backward called without cached input or labels\n");
        return -1;
    }

    // Ensure parameter count is initialized
    if (pipeline->parameter_count == 0) {
        dist_pipeline_get_parameter_count(pipeline);
    }

    // Allocate gradients buffer if needed
    if (!pipeline->gradients || pipeline->gradient_size < pipeline->parameter_count * sizeof(float)) {
        free(pipeline->gradients);
        pipeline->gradient_size = pipeline->parameter_count * sizeof(float);
        pipeline->gradients = calloc(pipeline->parameter_count, sizeof(float));
    }

    if (!pipeline->gradients) return -1;

    float* grads = (float*)pipeline->gradients;
    int result = 0;

    // Initialize default gradient configuration if not set
    if (pipeline->gradient_config.shift_amount == 0.0) {
        pipeline->gradient_config.method = GRADIENT_METHOD_PARAMETER_SHIFT;
        pipeline->gradient_config.shift_amount = M_PI / 2.0;
        pipeline->gradient_config.finite_diff_epsilon = 1e-4;
        pipeline->gradient_config.spsa_perturbation = 0.1;
        pipeline->gradient_config.use_gradient_clipping = true;
        pipeline->gradient_config.clip_value = 1.0;
        pipeline->gradient_config.spsa_averaging_samples = 5;
    }

    // Compute gradients using the configured method
    switch (pipeline->gradient_config.method) {
        case GRADIENT_METHOD_PARAMETER_SHIFT:
            result = compute_gradients_parameter_shift(pipeline, grads);
            break;

        case GRADIENT_METHOD_SPSA:
            result = compute_gradients_spsa(pipeline, grads);
            break;

        case GRADIENT_METHOD_FINITE_DIFFERENCE:
            result = compute_gradients_finite_difference(pipeline, grads);
            break;

        case GRADIENT_METHOD_NATURAL_GRADIENT:
            // Natural gradient using Quantum Fisher Information Matrix
            result = compute_gradients_natural_gradient(pipeline, grads);
            break;

        default:
            result = compute_gradients_parameter_shift(pipeline, grads);
            break;
    }

    if (result != 0) {
        return result;
    }

    // Apply gradient clipping if enabled
    if (pipeline->gradient_config.use_gradient_clipping) {
        clip_gradients(grads, pipeline->parameter_count,
                       (float)pipeline->gradient_config.clip_value);
    }

    // Compute and store current loss
    float* params = NULL;
    size_t num_params = 0;
    if (quantum_get_task_parameters(pipeline->state->learning_task, &params, &num_params)) {
        pipeline->current_loss = evaluate_loss_with_params(pipeline, params, num_params);
        free(params);

        // Update loss history
        if (pipeline->loss_history_capacity == 0) {
            pipeline->loss_history_capacity = 100;
            pipeline->loss_history = malloc(pipeline->loss_history_capacity * sizeof(float));
        }
        if (pipeline->loss_history && pipeline->loss_history_size < pipeline->loss_history_capacity) {
            pipeline->loss_history[pipeline->loss_history_size++] = pipeline->current_loss;
        }
    }

    return 0;
}

int dist_pipeline_update_parameters(quantum_pipeline_t* pipeline) {
    if (!pipeline || !pipeline->gradients) return -1;
    if (!pipeline->state || !pipeline->state->learning_task) return -1;

    // Get current parameters from the learning task
    float* params = NULL;
    size_t num_params = 0;

    if (!quantum_get_task_parameters(pipeline->state->learning_task,
                                     &params, &num_params)) {
        return -1;
    }

    if (num_params == 0 || !params) {
        return -1;
    }

    // Apply gradient descent update: θ = θ - lr * ∇L
    const float* grads = (const float*)pipeline->gradients;
    float lr = pipeline->learning_rate;

    for (size_t i = 0; i < num_params && i < pipeline->parameter_count; i++) {
        params[i] -= lr * grads[i];
    }

    // Set the updated parameters back to the learning task
    bool success = quantum_set_task_parameters(pipeline->state->learning_task,
                                               params, num_params);

    free(params);

    return success ? 0 : -1;
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

    // Get actual parameter count from the learning task if available
    if (pipeline->parameter_count == 0 && pipeline->state && pipeline->state->learning_task) {
        float* params = NULL;
        size_t num_params = 0;

        if (quantum_get_task_parameters(pipeline->state->learning_task,
                                        &params, &num_params)) {
            pipeline->parameter_count = num_params;
            free(params);
        } else {
            // Fallback: Calculate from circuit structure
            size_t num_qubits = (size_t)pipeline->state->config[QG_CONFIG_NUM_QUBITS];
            size_t num_layers = (size_t)pipeline->state->config[QG_CONFIG_NUM_LAYERS];
            // Each layer has rotation gates with 3 parameters per qubit
            pipeline->parameter_count = num_qubits * num_layers * 3;
        }
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
    if (!pipeline || !metrics) return;

    // Copy metrics from pipeline and state
    // Use the current_loss computed during backward pass if available
    if (pipeline->current_loss != 0.0f) {
        metrics->loss = pipeline->current_loss;
    } else if (pipeline->state) {
        metrics->loss = pipeline->state->current_loss;
    } else {
        metrics->loss = 0.0f;
    }

    metrics->learning_rate = pipeline->learning_rate;

    if (pipeline->state) {
        metrics->accuracy = pipeline->state->current_accuracy;
        metrics->throughput = pipeline->state->throughput;
        metrics->memory_used = (double)pipeline->state->memory_usage;
    } else {
        metrics->accuracy = 0.0f;
        metrics->throughput = 0.0f;
        metrics->memory_used = 0.0;
    }
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

// Configure gradient estimation method
int dist_pipeline_set_gradient_config(quantum_pipeline_t* pipeline,
                                       gradient_method_t method,
                                       double shift_amount,
                                       double finite_diff_epsilon,
                                       double spsa_perturbation,
                                       bool use_gradient_clipping,
                                       double clip_value,
                                       size_t spsa_averaging_samples) {
    if (!pipeline) return -1;

    pipeline->gradient_config.method = method;
    pipeline->gradient_config.shift_amount = shift_amount > 0.0 ? shift_amount : M_PI / 2.0;
    pipeline->gradient_config.finite_diff_epsilon = finite_diff_epsilon > 0.0 ? finite_diff_epsilon : 1e-4;
    pipeline->gradient_config.spsa_perturbation = spsa_perturbation > 0.0 ? spsa_perturbation : 0.1;
    pipeline->gradient_config.use_gradient_clipping = use_gradient_clipping;
    pipeline->gradient_config.clip_value = clip_value > 0.0 ? clip_value : 1.0;
    pipeline->gradient_config.spsa_averaging_samples = spsa_averaging_samples > 0 ? spsa_averaging_samples : 5;

    return 0;
}

// Get current loss value
float dist_pipeline_get_loss(quantum_pipeline_t* pipeline) {
    if (!pipeline) return 0.0f;
    return pipeline->current_loss;
}

// Get loss history
int dist_pipeline_get_loss_history(quantum_pipeline_t* pipeline,
                                    float** history,
                                    size_t* num_entries) {
    if (!pipeline || !history || !num_entries) return -1;

    if (pipeline->loss_history && pipeline->loss_history_size > 0) {
        *history = malloc(pipeline->loss_history_size * sizeof(float));
        if (!*history) return -1;

        memcpy(*history, pipeline->loss_history, pipeline->loss_history_size * sizeof(float));
        *num_entries = pipeline->loss_history_size;
    } else {
        *history = NULL;
        *num_entries = 0;
    }

    return 0;
}

// Cleanup function for distributed pipeline (call before freeing)
void dist_pipeline_cleanup(quantum_pipeline_t* pipeline) {
    if (!pipeline) return;

    // Free all dynamically allocated memory in the pipeline struct
    free(pipeline->gradients);
    pipeline->gradients = NULL;
    pipeline->gradient_size = 0;

    free(pipeline->cached_output);
    pipeline->cached_output = NULL;
    pipeline->cached_output_size = 0;

    free(pipeline->cached_input);
    pipeline->cached_input = NULL;
    pipeline->cached_input_size = 0;

    free(pipeline->cached_labels);
    pipeline->cached_labels = NULL;
    pipeline->cached_labels_size = 0;

    free(pipeline->original_params);
    pipeline->original_params = NULL;

    free(pipeline->shifted_params);
    pipeline->shifted_params = NULL;
    pipeline->params_buffer_size = 0;

    free(pipeline->loss_history);
    pipeline->loss_history = NULL;
    pipeline->loss_history_size = 0;
    pipeline->loss_history_capacity = 0;

    pipeline->parameter_count = 0;
    pipeline->current_batch_size = 0;
    pipeline->current_loss = 0.0f;
}

// Initialize distributed pipeline with default values
int dist_pipeline_init(quantum_pipeline_t* pipeline) {
    if (!pipeline) return -1;

    // Initialize to zero/NULL
    pipeline->gradients = NULL;
    pipeline->gradient_size = 0;
    pipeline->parameter_count = 0;
    pipeline->learning_rate = 0.001f;
    pipeline->cached_output = NULL;
    pipeline->cached_output_size = 0;
    pipeline->cached_input = NULL;
    pipeline->cached_input_size = 0;
    pipeline->cached_labels = NULL;
    pipeline->cached_labels_size = 0;
    pipeline->current_batch_size = 0;
    pipeline->original_params = NULL;
    pipeline->shifted_params = NULL;
    pipeline->params_buffer_size = 0;
    pipeline->current_loss = 0.0f;
    pipeline->loss_history = NULL;
    pipeline->loss_history_size = 0;
    pipeline->loss_history_capacity = 0;

    // Set default gradient configuration
    pipeline->gradient_config.method = GRADIENT_METHOD_PARAMETER_SHIFT;
    pipeline->gradient_config.shift_amount = M_PI / 2.0;
    pipeline->gradient_config.finite_diff_epsilon = 1e-4;
    pipeline->gradient_config.spsa_perturbation = 0.1;
    pipeline->gradient_config.use_gradient_clipping = true;
    pipeline->gradient_config.clip_value = 1.0;
    pipeline->gradient_config.spsa_averaging_samples = 5;

    return 0;
}
