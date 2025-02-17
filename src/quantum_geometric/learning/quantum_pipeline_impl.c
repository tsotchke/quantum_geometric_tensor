#include "quantum_geometric/learning/quantum_pipeline.h"
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

void* quantum_pipeline_create(const float* config) {
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
        .optimizer_type = OPTIMIZER_QUANTUM_GRADIENT_DESCENT,
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

int quantum_pipeline_train(void* pipeline, const float* data, const int* labels, size_t num_samples) {
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

int quantum_pipeline_evaluate(void* pipeline, const float* data, const int* labels, 
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

int quantum_pipeline_save(void* pipeline, const char* filename) {
    (void)pipeline;  // Unused parameter
    (void)filename;  // Unused parameter
    // Not implemented for minimal version
    return QG_SUCCESS;
}

void quantum_pipeline_destroy(void* pipeline) {
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
