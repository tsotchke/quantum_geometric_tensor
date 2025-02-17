#include "quantum_geometric/learning/learning_task.h"
#include "quantum_geometric/core/numerical_backend.h"
#include "quantum_geometric/core/tensor_network_operations.h"
#include "quantum_geometric/core/hierarchical_matrix.h"
#include "quantum_geometric/core/quantum_complex.h"
#include "quantum_geometric/core/quantum_types.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/error_codes.h"
#include "quantum_geometric/core/complex_arithmetic.h"
#include "quantum_geometric/hardware/quantum_hardware_types.h"
#include "quantum_geometric/learning/quantum_stochastic_sampling.h"
#include "quantum_geometric/learning/quantum_functions.h"
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

// Default configuration values
#define DEFAULT_LEARNING_RATE 0.001
#define DEFAULT_BATCH_SIZE 32
#define DEFAULT_NUM_EPOCHS 100
#define DEFAULT_NUM_QUBITS 8
#define DEFAULT_NUM_LAYERS 4
#define DEFAULT_LATENT_DIM 32
#define DEFAULT_INPUT_DIM 784  // MNIST image size
#define DEFAULT_OUTPUT_DIM 10  // MNIST classes

// Helper function to get current time in seconds
static double get_current_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Internal state structure
struct quantum_learning_task {
    task_config_t config;           // Task configuration from learning_task.h
    training_state_t state;         // Training state from learning_task.h
    ComplexFloat* weights;          // Model weights
    void* model_state;             // HierarchicalMatrix for O(log n) operations
    void* optimizer_state;         // Optimizer state
    size_t num_weights;            // Number of model weights
    bool initialized;              // Whether model is initialized
    tensor_network_t* network;     // Reusable tensor network
    double training_start_time;    // For tracking training time
    bool (*initialize)(struct quantum_learning_task*);  // Initialize function pointer
};

// Forward declarations of advanced functionality
static bool initialize_model_state(struct quantum_learning_task* task);
static bool perform_forward_pass(struct quantum_learning_task* task,
                               const ComplexFloat** features,
                               ComplexFloat* output);
static bool perform_backward_pass(struct quantum_learning_task* task,
                                const ComplexFloat** features,
                                const ComplexFloat* labels,
                                const ComplexFloat* predictions);

learning_task_handle_t quantum_create_learning_task(const task_config_t* config) {
    if (!config) {
        printf("DEBUG: Using default config for MNIST classification\n");
        // Use default config for MNIST classification
        task_config_t default_config = {
            .task_type = TASK_CLASSIFICATION,
            .model_type = MODEL_QUANTUM_NEURAL_NETWORK,
            .optimizer_type = OPTIMIZER_QUANTUM_ADAM,
            .input_dim = DEFAULT_INPUT_DIM,
            .output_dim = DEFAULT_OUTPUT_DIM,
            .latent_dim = DEFAULT_LATENT_DIM,
            .num_qubits = DEFAULT_NUM_QUBITS,
            .num_layers = DEFAULT_NUM_LAYERS,
            .batch_size = DEFAULT_BATCH_SIZE,
            .learning_rate = DEFAULT_LEARNING_RATE,
            .use_gpu = false,
            .enable_error_mitigation = true,
            .num_shots = 1000
        };
        config = &default_config;
    }
    
    printf("DEBUG: Allocating learning task structure\n");
    struct quantum_learning_task* task = calloc(1, sizeof(struct quantum_learning_task));
    if (!task) {
        printf("DEBUG: Failed to allocate learning task\n");
        return NULL;
    }
    
    // Copy configuration
    task->config = *config;
    
    // Initialize training state
    task->state.current_epoch = 0;
    task->state.total_epochs = DEFAULT_NUM_EPOCHS;
    task->state.current_loss = 0.0;
    task->state.best_loss = INFINITY;
    task->state.learning_rate = config->learning_rate;
    task->state.iterations_without_improvement = 0;
    task->state.converged = false;
    
    // Calculate total number of weights needed
    printf("DEBUG: input_dim=%zu, output_dim=%zu, latent_dim=%zu\n", 
           config->input_dim, config->output_dim, config->latent_dim);
    
    // Calculate weights for each layer
    size_t total_weights = 0;
    size_t prev_dim = config->input_dim;
    for (size_t i = 0; i < config->num_layers + 1; i++) {
        size_t curr_dim;
        if (i == config->num_layers) {
            curr_dim = config->output_dim;  // Output layer
        } else {
            curr_dim = config->latent_dim;  // Hidden layer
        }
        total_weights += prev_dim * curr_dim;
        prev_dim = curr_dim;
    }
    
    task->num_weights = total_weights;
    printf("DEBUG: Total weights needed: %zu\n", total_weights);
    
    // Initialize weights
    task->weights = calloc(task->num_weights, sizeof(ComplexFloat));
    if (!task->weights) {
        printf("DEBUG: Failed to allocate weights\n");
        free(task);
        return NULL;
    }
    
    // Initialize weights using quantum stochastic sampling
    printf("DEBUG: Setting up stochastic sampling\n");
    DiffusionConfig diff_config = {
        .t_min = 0.0,
        .t_max = 1.0,
        .num_steps = 100,
        .beta_min = 0.1,
        .beta_max = 20.0,
        .use_cosine_schedule = true,
        .transform_type = GEOMETRIC_TRANSFORM_DIFFUSION
    };
    
    PINNConfig pinn_config = {
        .input_dim = 1,
        .hidden_dim = 32,
        .num_layers = 4,
        .learning_rate = 0.001,
        .batch_size = 32,
        .max_epochs = 100,
        .weight_decay = 0.0,
        .attention_heads = 4,
        .head_dim = 8,
        .tensor_bond_dim = 16
    };
    
    LMCConfig lmc_config = {
        .step_size = 0.1,
        .num_steps = 1000,
        .num_chains = 1,
        .adapt_step_size = true,
        .device = QUANTUM_DEVICE_CPU
    };
    
    StochasticSampler* sampler = stochastic_sampler_create(&diff_config, &pinn_config, &lmc_config);
    if (!sampler) {
        printf("DEBUG: Failed to create stochastic sampler\n");
        free(task->weights);
        free(task);
        return NULL;
    }
    
    double* samples = malloc(task->num_weights * sizeof(double));
    if (!samples) {
        printf("DEBUG: Failed to allocate samples\n");
        stochastic_sampler_free(sampler);
        free(task->weights);
        free(task);
        return NULL;
    }
    
    printf("DEBUG: Sampling weights\n");
    if (stochastic_sampler_sample(sampler, task->num_weights, samples) != 0) {
        printf("DEBUG: Failed to sample weights\n");
        free(samples);
        stochastic_sampler_free(sampler);
        free(task->weights);
        free(task);
        return NULL;
    }
    
    // Convert samples to complex weights with Xavier initialization and quantum phase
    float scale = sqrtf(2.0f / (task->config.input_dim + task->config.output_dim));
    for (size_t i = 0; i < task->num_weights; i++) {
        // Use consecutive samples for real and imaginary parts
        float real = (float)samples[i] * scale;
        float imag = (float)samples[(i + 1) % task->num_weights] * scale;
        task->weights[i] = (ComplexFloat){real, imag};
    }
    
    free(samples);
    stochastic_sampler_free(sampler);
    
    // Set up initialization function
    task->initialize = initialize_model_state;
    task->initialized = false;  // Changed to false since we haven't initialized yet
    
    // Initialize model state immediately
    printf("DEBUG: Initializing model state\n");
    if (!initialize_model_state(task)) {
        printf("DEBUG: Failed to initialize model state\n");
        free(task->weights);
        free(task);
        return NULL;
    }
    task->initialized = true;
    
    printf("DEBUG: Learning task created successfully\n");
    return task;
}

void quantum_destroy_learning_task(learning_task_handle_t task) {
    if (!task) return;
    
    printf("DEBUG: Destroying learning task\n");
    
    if (task->model_state) {
        destroy_hierarchical_matrix((HierarchicalMatrix*)task->model_state);
    }
    
    if (task->optimizer_state) {
        free(task->optimizer_state);
    }
    
    if (task->network) {
        quantum_free_tensor_network(task->network);
    }
    
    free(task->weights);
    free(task);
}

bool quantum_train_task(learning_task_handle_t task,
                       const ComplexFloat** features,
                       const ComplexFloat* labels,
                       size_t num_samples) {
    if (!task || !features || !labels || num_samples == 0) {
        printf("DEBUG: Invalid arguments in train_task\n");
        return false;
    }
    
    printf("DEBUG: Starting training with %zu samples\n", num_samples);
    
    // Record training start time
    task->training_start_time = get_current_time();
    
    // Initialize model if needed
    if (!task->initialized) {
        printf("DEBUG: Model not initialized, initializing now\n");
        if (!task->initialize(task)) {
            printf("DEBUG: Failed to initialize model state\n");
            return false;
        }
        task->initialized = true;
    }
    
    // Training loop
    for (size_t epoch = 0; epoch < task->state.total_epochs; epoch++) {
        float total_loss = 0.0f;
        
        printf("DEBUG: Starting epoch %zu\n", epoch);
        
        // Process each sample
        for (size_t i = 0; i < num_samples; i++) {
            // Forward pass
            ComplexFloat* output = malloc(task->config.output_dim * sizeof(ComplexFloat));
            if (!output) {
                printf("DEBUG: Failed to allocate output buffer\n");
                return false;
            }
            
            printf("DEBUG: Forward pass for sample %zu\n", i);
            if (!perform_forward_pass(task, &features[i], output)) {
                printf("DEBUG: Forward pass failed\n");
                free(output);
                return false;
            }
            
            // Compute loss (MSE)
            float batch_loss = 0.0f;
            for (size_t j = 0; j < task->config.output_dim; j++) {
                ComplexFloat error = complex_subtract(labels[i * task->config.output_dim + j], output[j]);
                batch_loss += complex_abs_squared(error);
            }
            total_loss += batch_loss / task->config.output_dim;
            
            printf("DEBUG: Backward pass for sample %zu\n", i);
            // Backward pass
            if (!perform_backward_pass(task, &features[i], &labels[i * task->config.output_dim], output)) {
                printf("DEBUG: Backward pass failed\n");
                free(output);
                return false;
            }
            
            free(output);
        }
        
        // Update state
        task->state.current_epoch = epoch;
        task->state.current_loss = total_loss / num_samples;
        
        printf("DEBUG: Epoch %zu complete, loss: %f\n", epoch, task->state.current_loss);
        
        // Check convergence
        if (task->state.current_loss < task->state.best_loss) {
            task->state.best_loss = task->state.current_loss;
            task->state.iterations_without_improvement = 0;
        } else {
            task->state.iterations_without_improvement++;
            if (task->state.iterations_without_improvement > 10) {
                task->state.converged = true;
                printf("DEBUG: Training converged after %zu epochs\n", epoch);
                break;
            }
        }
    }
    
    printf("DEBUG: Training completed successfully\n");
    return true;
}

// Initialize model state based on task type
static bool initialize_model_state(struct quantum_learning_task* task) {
    if (!task) return false;

    printf("DEBUG: Initializing model state for task type %d\n", task->config.task_type);

    switch (task->config.task_type) {
        case TASK_CLASSIFICATION: {
            // Store layer dimensions in matrix_data
            size_t* layer_dims = malloc((task->config.num_layers + 2) * sizeof(size_t));
            if (!layer_dims) {
                printf("DEBUG: Failed to allocate layer dimensions\n");
                return false;
            }
            
            // Fill layer dimensions array
            layer_dims[0] = task->config.input_dim;
            for (size_t i = 1; i <= task->config.num_layers; i++) {
                layer_dims[i] = task->config.latent_dim;
            }
            layer_dims[task->config.num_layers + 1] = task->config.output_dim;
            
            // Calculate total weights needed
            size_t total_weights = 0;
            for (size_t i = 0; i < task->config.num_layers + 1; i++) {
                total_weights += layer_dims[i] * layer_dims[i + 1];
            }
            
            printf("DEBUG: Creating weight matrix with total_weights=%zu\n", total_weights);
            
            HierarchicalMatrix* matrix = create_hierarchical_matrix(total_weights, 1e-6);
            if (!matrix) {
                printf("DEBUG: Failed to create hierarchical matrix\n");
                free(layer_dims);
                return false;
            }
            
            // Set matrix properties
            matrix->rows = total_weights;
            matrix->cols = 1;
            matrix->n = total_weights;
            matrix->type = MATRIX_QUANTUM;
            matrix->format = STORAGE_FULL;
            matrix->is_leaf = true;
            matrix->rank = 0;
            matrix->U = NULL;
            matrix->V = NULL;
            matrix->tolerance = 1e-6;
            matrix->matrix_data = layer_dims;
            for (int i = 0; i < 4; i++) {
                matrix->children[i] = NULL;
            }
            
            printf("DEBUG: Initializing weights with total_weights=%zu\n", total_weights);
            if (!quantum_initialize_weights(matrix, total_weights, 1)) {
                printf("DEBUG: Failed to initialize weights\n");
                destroy_hierarchical_matrix(matrix);
                return false;
            }
            
            task->model_state = matrix;
            printf("DEBUG: Model state initialized with matrix at %p\n", (void*)matrix);
            break;
        }
        case TASK_REGRESSION:
            // Similar to classification but with different output structure
            break;
        case TASK_CLUSTERING:
            // Initialize centroids and clustering state
            break;
        default:
            printf("DEBUG: Unknown task type %d\n", task->config.task_type);
            return false;
    }

    printf("DEBUG: Model state initialized successfully\n");
    return true;
}

// Forward pass using tensor networks for O(log n) operations
static bool perform_forward_pass(struct quantum_learning_task* task,
                               const ComplexFloat** features,
                               ComplexFloat* output) {
    if (!task || !features || !output) {
        printf("DEBUG: Invalid arguments in forward_pass\n");
        return false;
    }

    // Create or recreate tensor network
    if (task->network) {
        quantum_free_tensor_network(task->network);
    }
    
    printf("DEBUG: Creating tensor network\n");
    task->network = create_tensor_network();
    if (!task->network) {
        printf("DEBUG: Failed to create tensor network\n");
        return false;
    }

    // Add input features as first node
    size_t feature_node_id;
    size_t feature_dims[] = {task->config.batch_size, task->config.input_dim};
    if (!add_tensor_node(task->network, *features, feature_dims, 2, &feature_node_id)) {
        printf("DEBUG: Failed to add feature node\n");
        quantum_free_tensor_network(task->network);
        return false;
    }

    // Forward propagation through layers
    printf("DEBUG: Forward propagation through %zu layers\n", task->config.num_layers + 1);
    for (size_t i = 0; i < task->config.num_layers + 1; i++) {
        // Get layer weights from hierarchical matrix
        printf("DEBUG: Getting weights for layer %zu from model_state at %p\n", i, task->model_state);
        HierarchicalMatrix* weights = quantum_get_layer_weights(
            (HierarchicalMatrix*)task->model_state,
            i
        );
        if (!weights) {
            printf("DEBUG: Failed to get weights for layer %zu\n", i);
            return false;
        }

        // Perform O(log n) matrix multiplication
        printf("DEBUG: Matrix multiplication for layer %zu\n", i);
        if (!quantum_tensor_network_multiply(task->network, weights)) {
            printf("DEBUG: Matrix multiplication failed for layer %zu\n", i);
            return false;
        }

        // Apply activation function and normalization
        if (i < task->config.num_layers) {
            printf("DEBUG: Applying ReLU activation for layer %zu\n", i);
            quantum_apply_activation(task->network, "relu");
            
            // Normalize activations
            size_t last_id = task->network->num_nodes - 1;
            tensor_node_t* last_node = task->network->nodes[last_id];
            if (last_node && last_node->data) {
                float max_val = 0.0f;
                // Find maximum activation
                for (size_t j = 0; j < last_node->dimensions[0] * last_node->dimensions[1]; j++) {
                    float mag = sqrtf(last_node->data[j].real * last_node->data[j].real + 
                                    last_node->data[j].imag * last_node->data[j].imag);
                    if (mag > max_val) max_val = mag;
                }
                // Normalize if max value is significant
                if (max_val > 1e-6f) {
                    for (size_t j = 0; j < last_node->dimensions[0] * last_node->dimensions[1]; j++) {
                        last_node->data[j].real /= max_val;
                        last_node->data[j].imag /= max_val;
                    }
                }
            }
        }
    }

    // Extract output directly as complex values
    printf("DEBUG: Extracting output\n");
    size_t last_id = task->network->num_nodes - 1;
    tensor_node_t* last_node = task->network->nodes[last_id];
    if (!last_node || !last_node->data) {
        printf("DEBUG: Invalid last node\n");
        return false;
    }

    // Copy complex output values
    for (size_t i = 0; i < task->config.output_dim; i++) {
        output[i] = last_node->data[i];
    }
    return true;
}

// Backward pass using tensor networks for O(log n) operations
static bool perform_backward_pass(struct quantum_learning_task* task,
                                const ComplexFloat** features,
                                const ComplexFloat* labels,
                                const ComplexFloat* predictions) {
    if (!task || !features || !labels || !predictions) {
        printf("DEBUG: Invalid arguments in backward_pass\n");
        return false;
    }

    // Create gradient network directly with complex data
    printf("DEBUG: Creating gradient network\n");
    tensor_network_t* gradient_network = create_tensor_network();
    if (!gradient_network) {
        printf("DEBUG: Failed to create gradient network\n");
        return false;
    }

    // Add feature node with batch dimension
    size_t feature_node_id;
    size_t feature_dims[] = {task->config.batch_size, task->config.input_dim};
    if (!add_tensor_node(gradient_network, *features, feature_dims, 2, &feature_node_id)) {
        printf("DEBUG: Failed to add feature node\n");
        destroy_tensor_network(gradient_network);
        return false;
    }

    // Add labels node with batch dimension
    size_t labels_node_id;
    size_t label_dims[] = {task->config.batch_size, task->config.output_dim};
    if (!add_tensor_node(gradient_network, labels, label_dims, 2, &labels_node_id)) {
        printf("DEBUG: Failed to add labels node\n");
        destroy_tensor_network(gradient_network);
        return false;
    }

    // Add predictions node with batch dimension
    size_t predictions_node_id;
    if (!add_tensor_node(gradient_network, predictions, label_dims, 2, &predictions_node_id)) {
        printf("DEBUG: Failed to add predictions node\n");
        destroy_tensor_network(gradient_network);
        return false;
    }

    // Backward propagation through layers
    printf("DEBUG: Backward propagation through layers\n");
    for (int i = task->config.num_layers; i >= 0; i--) {
        // Get layer weights
        printf("DEBUG: Getting weights for layer %d\n", i);
        HierarchicalMatrix* weights = quantum_get_layer_weights(
            (HierarchicalMatrix*)task->model_state,
            i
        );
        if (!weights) {
            printf("DEBUG: Failed to get weights for layer %d\n", i);
            quantum_free_tensor_network(gradient_network);
            return false;
        }

        // Calculate gradients with O(log n) complexity
        printf("DEBUG: Calculating gradients for layer %d\n", i);
        if (!quantum_calculate_gradients(gradient_network, weights)) {
            printf("DEBUG: Failed to calculate gradients for layer %d\n", i);
            quantum_free_tensor_network(gradient_network);
            return false;
        }

        // Get clipped gradients from network
        printf("DEBUG: Getting clipped gradients for layer %d\n", i);
        size_t last_id = gradient_network->num_nodes - 1;
        tensor_node_t* clipped_node = gradient_network->nodes[last_id];
        if (!clipped_node || !clipped_node->data) {
            printf("DEBUG: Invalid clipped gradients node\n");
            quantum_free_tensor_network(gradient_network);
            return false;
        }

        // Update weights using optimizer with clipped gradients
        printf("DEBUG: Updating weights for layer %d\n", i);
        if (!quantum_update_weights(weights,
                                  gradient_network,
                                  task->config.learning_rate,
                                  task->optimizer_state,
                                  clipped_node->data)) {
            printf("DEBUG: Failed to update weights for layer %d\n", i);
            quantum_free_tensor_network(gradient_network);
            return false;
        }
    }

    quantum_free_tensor_network(gradient_network);
    printf("DEBUG: Backward pass completed successfully\n");
    return true;
}

bool quantum_evaluate_task(learning_task_handle_t task,
                         const ComplexFloat** features,
                         const ComplexFloat* labels,
                         size_t num_samples,
                         task_metrics_t* metrics) {
    if (!task || !features || !labels || num_samples == 0 || !metrics) {
        printf("DEBUG: Invalid arguments in evaluate_task\n");
        return false;
    }
    
    double total_loss = 0.0;
    size_t correct = 0;
    double start_time = get_current_time();
    
    printf("DEBUG: Starting evaluation with %zu samples\n", num_samples);
    
    // Evaluate each sample
    for (size_t i = 0; i < num_samples; i++) {
        // Forward pass
        ComplexFloat* output = malloc(task->config.output_dim * sizeof(ComplexFloat));
        if (!output) {
            printf("DEBUG: Failed to allocate output buffer\n");
            return false;
        }
        
        printf("DEBUG: Forward pass for sample %zu\n", i);
        if (!perform_forward_pass(task, &features[i], output)) {
            printf("DEBUG: Forward pass failed\n");
            free(output);
            return false;
        }
        
        // Compute metrics
        float batch_loss = 0.0f;
        for (size_t j = 0; j < task->config.output_dim; j++) {
            ComplexFloat error = complex_subtract(labels[i * task->config.output_dim + j], output[j]);
            batch_loss += complex_abs_squared(error);
        }
        total_loss += batch_loss / task->config.output_dim;
        
        // For classification, check if prediction matches label
        if (batch_loss / task->config.output_dim < 0.25) {
            correct++;
        }
        
        free(output);
    }
    
    double end_time = get_current_time();
    
    // Fill metrics
    metrics->accuracy = (double)correct / num_samples;
    metrics->mse = total_loss / num_samples;
    metrics->training_time = end_time - start_time;
    metrics->inference_time = (end_time - start_time) / num_samples;
    metrics->memory_usage = task->num_weights * sizeof(ComplexFloat);
    metrics->quantum_advantage = 1.0; // Not implemented
    
    printf("DEBUG: Evaluation completed - accuracy: %f, mse: %f\n", 
           metrics->accuracy, metrics->mse);
    
    return true;
}

bool quantum_predict_task(learning_task_handle_t task,
                        const ComplexFloat* input,
                        ComplexFloat* output) {
    if (!task || !input || !output) {
        printf("DEBUG: Invalid arguments in predict_task\n");
        return false;
    }
    return perform_forward_pass(task, &input, output);
}

bool quantum_get_training_state(learning_task_handle_t task,
                              training_state_t* state) {
    if (!task || !state) {
        printf("DEBUG: Invalid arguments in get_training_state\n");
        return false;
    }
    *state = task->state;
    return true;
}

bool quantum_update_learning_rate(learning_task_handle_t task,
                                double new_learning_rate) {
    if (!task) {
        printf("DEBUG: Invalid task in update_learning_rate\n");
        return false;
    }
    task->state.learning_rate = new_learning_rate;
    return true;
}

bool quantum_early_stop(learning_task_handle_t task) {
    if (!task) {
        printf("DEBUG: Invalid task in early_stop\n");
        return false;
    }
    task->state.converged = true;
    return true;
}

bool quantum_compare_classical(learning_task_handle_t task,
                             const ComplexFloat** features,
                             const ComplexFloat* labels,
                             size_t num_samples,
                             task_metrics_t* classical_metrics) {
    if (!task || !features || !labels || num_samples == 0 || !classical_metrics) {
        printf("DEBUG: Invalid arguments in compare_classical\n");
        return false;
    }
    
    // Run classical algorithm for comparison
    double start_time = get_current_time();
    
    double total_loss = 0.0;
    size_t correct = 0;
    
    printf("DEBUG: Running classical comparison with %zu samples\n", num_samples);
    
    // Simple classical linear model
    for (size_t i = 0; i < num_samples; i++) {
        ComplexFloat* output = malloc(task->config.output_dim * sizeof(ComplexFloat));
        if (!output) {
            printf("DEBUG: Failed to allocate output buffer\n");
            return false;
        }
        
        for (size_t j = 0; j < task->config.output_dim; j++) {
            output[j] = (ComplexFloat){0.0f, 0.0f};
            for (size_t k = 0; k < task->config.input_dim; k++) {
                output[j] = complex_add(output[j],
                                   complex_multiply(features[i][k],
                                            task->weights[j * task->config.input_dim + k]));
            }
        }
        
        float batch_loss = 0.0f;
        for (size_t j = 0; j < task->config.output_dim; j++) {
            ComplexFloat error = complex_subtract(labels[i * task->config.output_dim + j], output[j]);
            batch_loss += complex_abs_squared(error);
        }
        total_loss += batch_loss / task->config.output_dim;
        
        if (batch_loss / task->config.output_dim < 0.25) {
            correct++;
        }
        
        free(output);
    }
    
    double end_time = get_current_time();
    
    classical_metrics->accuracy = (double)correct / num_samples;
    classical_metrics->mse = total_loss / num_samples;
    classical_metrics->training_time = end_time - start_time;
    classical_metrics->memory_usage = task->num_weights * sizeof(float);
    
    printf("DEBUG: Classical comparison completed - accuracy: %f, mse: %f\n",
           classical_metrics->accuracy, classical_metrics->mse);
    
    return true;
}

bool quantum_analyze_advantage(learning_task_handle_t task,
                             const task_metrics_t* quantum_metrics,
                             const task_metrics_t* classical_metrics,
                             double* advantage_score) {
    if (!task || !quantum_metrics || !classical_metrics || !advantage_score) {
        printf("DEBUG: Invalid arguments in analyze_advantage\n");
        return false;
    }
    
    // Calculate speedup
    double speedup = classical_metrics->training_time / quantum_metrics->training_time;
    
    // Calculate accuracy improvement
    double accuracy_improvement = quantum_metrics->accuracy / classical_metrics->accuracy;
    
    // Calculate memory efficiency
    double memory_efficiency = (double)classical_metrics->memory_usage / quantum_metrics->memory_usage;
    
    // Combine factors with weights
    *advantage_score = 0.4 * speedup + 0.4 * accuracy_improvement + 0.2 * memory_efficiency;
    
    printf("DEBUG: Advantage analysis - speedup: %f, accuracy_improvement: %f, memory_efficiency: %f\n",
           speedup, accuracy_improvement, memory_efficiency);
    
    return true;
}

bool quantum_enable_error_mitigation(learning_task_handle_t task) {
    if (!task) {
        printf("DEBUG: Invalid task in enable_error_mitigation\n");
        return false;
    }
    task->config.enable_error_mitigation = true;
    return true;
}

bool quantum_disable_error_mitigation(learning_task_handle_t task) {
    if (!task) {
        printf("DEBUG: Invalid task in disable_error_mitigation\n");
        return false;
    }
    task->config.enable_error_mitigation = false;
    return true;
}

bool quantum_get_error_rates(learning_task_handle_t task,
                           double* error_rates,
                           size_t* num_rates) {
    if (!task || !error_rates || !num_rates) {
        printf("DEBUG: Invalid arguments in get_error_rates\n");
        return false;
    }
    
    // For now, return a simple error rate based on loss
    error_rates[0] = task->state.current_loss;
    *num_rates = 1;
    
    printf("DEBUG: Error rate: %f\n", error_rates[0]);
    
    return true;
}
