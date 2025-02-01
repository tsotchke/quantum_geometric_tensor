#include "quantum_geometric/learning/learning_task.h"
#include "quantum_geometric/core/quantum_matrix_operations.h"
#include "quantum_geometric/core/tensor_network_operations.h"
#include "quantum_geometric/core/performance_monitor.h"
#include "quantum_geometric/core/memory_optimization.h"
#include "quantum_geometric/distributed/training_orchestrator.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Default configurations
#define DEFAULT_BATCH_SIZE 32
#define DEFAULT_NUM_EPOCHS 100
#define DEFAULT_VAL_SPLIT 0.2f
#define DEFAULT_LEARNING_RATE 0.001f
#define DEFAULT_DROPOUT_RATE 0.5f
#define DEFAULT_CHECKPOINT_FREQ 10
#define DEFAULT_PATIENCE 10

// Helper functions for model operations
static bool initialize_model_state(qg_learning_task_t* task) {
    // Initialize based on task type
    switch (task->type) {
        case QG_TASK_CLASSIFICATION: {
            // Use hierarchical matrices for O(log n) operations
            size_t total_params = 0;
            for (size_t i = 0; i < task->model_config.num_hidden + 1; i++) {
                size_t in_dim = i == 0 ? task->model_config.input_dim : 
                               task->model_config.hidden_dims[i-1];
                size_t out_dim = i == task->model_config.num_hidden ? 
                                task->model_config.output_dim :
                                task->model_config.hidden_dims[i];
                total_params += in_dim * out_dim;
            }

            // Allocate model state
            HierarchicalMatrix* matrix = hmatrix_create(total_params, total_params, 1e-6);
            if (!matrix) return false;

            // Initialize weights using quantum operations
            if (!quantum_initialize_weights(matrix, total_params)) {
                hmatrix_destroy(matrix);
                return false;
            }
            task->model_state = matrix;
            break;
        }
        case QG_TASK_REGRESSION:
            // Similar to classification but with different output structure
            break;
        case QG_TASK_CLUSTERING:
            // Initialize centroids and clustering state
            break;
        default:
            return false;
    }

    return true;
}

static bool perform_forward_pass(qg_learning_task_t* task, float** batch_data, float** output) {
    // Use tensor networks for O(log n) matrix multiplication
    tensor_network_t* network = quantum_create_tensor_network(
        batch_data,
        task->training_config.batch_size,
        task->model_config.input_dim
    );
    if (!network) return false;

    // Forward propagation through layers
    for (size_t i = 0; i < task->model_config.num_hidden + 1; i++) {
        // Get layer weights from hierarchical matrix
        HierarchicalMatrix* weights = quantum_get_layer_weights(
            (HierarchicalMatrix*)task->model_state,
            i
        );
        if (!weights) {
            quantum_free_tensor_network(network);
            return false;
        }

        // Perform O(log n) matrix multiplication
        if (!quantum_tensor_network_multiply(network, weights)) {
            quantum_free_tensor_network(network);
            return false;
        }

        // Apply activation function if not output layer
        if (i < task->model_config.num_hidden) {
            quantum_apply_activation(network, task->model_config.activation);
        }
    }

    // Extract output
    if (!quantum_extract_output(network, output)) {
        quantum_free_tensor_network(network);
        return false;
    }

    quantum_free_tensor_network(network);
    return true;
}

static bool perform_backward_pass(qg_learning_task_t* task,
                                float** batch_data,
                                float** batch_labels,
                                float** predictions) {
    // Calculate gradients using O(log n) operations
    tensor_network_t* gradient_network = quantum_create_gradient_network(
        batch_data,
        batch_labels,
        predictions,
        task->training_config.batch_size
    );
    if (!gradient_network) return false;

    // Backward propagation through layers
    for (int i = task->model_config.num_hidden; i >= 0; i--) {
        // Get layer weights
        HierarchicalMatrix* weights = quantum_get_layer_weights(
            (HierarchicalMatrix*)task->model_state,
            i
        );
        if (!weights) {
            quantum_free_tensor_network(gradient_network);
            return false;
        }

        // Calculate gradients with O(log n) complexity
        if (!quantum_calculate_gradients(gradient_network, weights)) {
            quantum_free_tensor_network(gradient_network);
            return false;
        }

        // Update weights using optimizer
        if (!quantum_update_weights(weights,
                                  gradient_network,
                                  task->model_config.learning_rate,
                                  task->optimizer_state)) {
            quantum_free_tensor_network(gradient_network);
            return false;
        }
    }

    quantum_free_tensor_network(gradient_network);
    return true;
}

// Implementation of public functions

qg_learning_task_t* qg_create_learning_task(
    qg_task_type_t type,
    qg_model_config_t* model_config,
    qg_training_config_t* training_config
) {
    qg_learning_task_t* task = malloc(sizeof(qg_learning_task_t));
    if (!task) return NULL;

    // Initialize task properties
    task->type = type;
    if (model_config) {
        task->model_config = *model_config;
    } else {
        task->model_config = qg_default_model_config(type);
    }
    if (training_config) {
        task->training_config = *training_config;
    } else {
        task->training_config = qg_default_training_config();
    }

    // Initialize data pointers
    task->train_data = NULL;
    task->val_data = NULL;
    task->test_data = NULL;

    // Initialize state
    task->model_state = NULL;
    task->optimizer_state = NULL;
    task->current_epoch = 0;
    task->best_metric = INFINITY;
    task->converged = false;

    // Set up core operations
    task->initialize = initialize_model_state;
    task->cleanup = NULL;  // Set in specific task creation

    // Set up optional operations
    task->save_checkpoint = NULL;
    task->load_checkpoint = NULL;
    task->export_model = NULL;
    task->get_metric = NULL;

    return task;
}

bool qg_setup_task_data(
    qg_learning_task_t* task,
    qg_data_chunk_t* train_data,
    qg_data_chunk_t* val_data,
    qg_data_chunk_t* test_data
) {
    if (!task || !train_data) return false;

    // Validate dimensions
    if (train_data->num_dims != 2 || train_data->shape[1] != task->model_config.input_dim) {
        return false;
    }

    // Set data pointers
    task->train_data = train_data;
    task->val_data = val_data;
    task->test_data = test_data;

    return true;
}

bool qg_train_task(qg_learning_task_t* task) {
    if (!task || !task->train_data || !task->initialize) return false;

    // Initialize model if needed
    if (!task->model_state) {
        if (!task->initialize(task)) return false;
    }

    // Training loop
    size_t patience_counter = 0;
    for (size_t epoch = 0; epoch < task->training_config.num_epochs; epoch++) {
        task->current_epoch = epoch;

        // Training phase
        size_t num_batches = task->train_data->num_elements / 
                            task->training_config.batch_size;
        
        float epoch_loss = 0.0f;
        for (size_t batch = 0; batch < num_batches; batch++) {
            // Get batch data
            float** batch_data = quantum_get_batch(
                task->train_data,
                batch,
                task->training_config.batch_size
            );
            float** batch_labels = quantum_get_batch_labels(
                task->train_data,
                batch,
                task->training_config.batch_size
            );

            // Forward pass
            float** predictions = quantum_allocate_matrix(
                task->training_config.batch_size,
                task->model_config.output_dim
            );
            if (!perform_forward_pass(task, batch_data, predictions)) {
                quantum_free_matrix(predictions);
                quantum_free_matrix(batch_data);
                quantum_free_matrix(batch_labels);
                return false;
            }

            // Calculate loss
            float batch_loss = quantum_calculate_loss(
                predictions,
                batch_labels,
                task->training_config.batch_size,
                task->model_config.output_dim,
                task->model_config.loss
            );
            epoch_loss += batch_loss;

            // Backward pass
            if (!perform_backward_pass(task, batch_data, batch_labels, predictions)) {
                quantum_free_matrix(predictions);
                quantum_free_matrix(batch_data);
                quantum_free_matrix(batch_labels);
                return false;
            }

            // Cleanup
            quantum_free_matrix(predictions);
            quantum_free_matrix(batch_data);
            quantum_free_matrix(batch_labels);
        }
        epoch_loss /= num_batches;

        // Validation phase
        if (task->val_data) {
            float val_metric = quantum_evaluate_model(
                task,
                task->val_data
            );

            // Early stopping
            if (task->training_config.early_stopping) {
                if (val_metric < task->best_metric) {
                    task->best_metric = val_metric;
                    patience_counter = 0;
                    // Save best model if checkpointing enabled
                    if (task->save_checkpoint && task->training_config.checkpoint_dir) {
                        task->save_checkpoint(task, task->training_config.checkpoint_dir);
                    }
                } else {
                    patience_counter++;
                    if (patience_counter >= task->training_config.patience) {
                        task->converged = true;
                        break;
                    }
                }
            }
        }

        // Regular checkpointing
        if (task->save_checkpoint && task->training_config.checkpoint_dir &&
            epoch % task->training_config.checkpoint_freq == 0) {
            task->save_checkpoint(task, task->training_config.checkpoint_dir);
        }
    }

    return true;
}

bool qg_evaluate_task(
    qg_learning_task_t* task,
    const char** metrics,
    size_t num_metrics,
    float* results
) {
    if (!task || !task->test_data || !metrics || !results) return false;

    for (size_t i = 0; i < num_metrics; i++) {
        results[i] = quantum_compute_metric(
            task,
            task->test_data,
            metrics[i]
        );
    }

    return true;
}

bool qg_predict_task(
    qg_learning_task_t* task,
    float** inputs,
    size_t num_samples,
    float** outputs
) {
    if (!task || !inputs || !outputs) return false;

    // Use tensor networks for efficient prediction
    return perform_forward_pass(task, inputs, outputs);
}

void qg_cleanup_task(qg_learning_task_t* task) {
    if (!task) return;

    // Clean up model state
    if (task->model_state) {
        hmatrix_destroy((HierarchicalMatrix*)task->model_state);
    }

    // Clean up optimizer state
    if (task->optimizer_state) {
        free(task->optimizer_state);
    }

    // Note: Don't free dataset pointers as they're owned externally
    free(task);
}

qg_model_config_t qg_default_model_config(qg_task_type_t type) {
    qg_model_config_t config = {0};
    
    // Set reasonable defaults based on task type
    switch (type) {
        case QG_TASK_CLASSIFICATION:
            config.learning_rate = DEFAULT_LEARNING_RATE;
            config.dropout_rate = DEFAULT_DROPOUT_RATE;
            config.use_attention = true;
            config.use_residual = true;
            config.activation = "relu";
            config.optimizer = "adam";
            config.loss = "cross_entropy";
            break;
        case QG_TASK_REGRESSION:
            config.learning_rate = DEFAULT_LEARNING_RATE;
            config.dropout_rate = 0.0f;
            config.use_attention = false;
            config.use_residual = true;
            config.activation = "relu";
            config.optimizer = "adam";
            config.loss = "mse";
            break;
        default:
            break;
    }

    return config;
}

qg_training_config_t qg_default_training_config(void) {
    qg_training_config_t config = {
        .batch_size = DEFAULT_BATCH_SIZE,
        .num_epochs = DEFAULT_NUM_EPOCHS,
        .val_split = DEFAULT_VAL_SPLIT,
        .shuffle = true,
        .num_workers = 1,
        .distributed = false,
        .checkpoint_dir = NULL,
        .checkpoint_freq = DEFAULT_CHECKPOINT_FREQ,
        .early_stopping = true,
        .patience = DEFAULT_PATIENCE
    };
    return config;
}
