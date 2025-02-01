#ifndef LEARNING_TASK_H
#define LEARNING_TASK_H

#include <stdbool.h>
#include <stddef.h>
#include "quantum_geometric/core/quantum_complex.h"

// Learning task types
typedef enum {
    TASK_CLASSIFICATION,
    TASK_REGRESSION,
    TASK_CLUSTERING,
    TASK_DIMENSIONALITY_REDUCTION
} task_type_t;

// Model architecture types
typedef enum {
    MODEL_QUANTUM_NEURAL_NETWORK,
    MODEL_QUANTUM_KERNEL,
    MODEL_QUANTUM_BOLTZMANN,
    MODEL_QUANTUM_VARIATIONAL,
    MODEL_QUANTUM_ENSEMBLE
} model_type_t;

// Optimization methods
typedef enum {
    OPTIMIZER_QUANTUM_GRADIENT_DESCENT,
    OPTIMIZER_QUANTUM_ADAM,
    OPTIMIZER_QUANTUM_NATURAL_GRADIENT,
    OPTIMIZER_QUANTUM_EVOLUTION
} optimizer_type_t;

// Task configuration
typedef struct {
    task_type_t task_type;              // Type of learning task
    model_type_t model_type;            // Type of quantum model
    optimizer_type_t optimizer_type;     // Type of optimizer
    size_t input_dim;                   // Input dimension
    size_t output_dim;                  // Output dimension
    size_t latent_dim;                  // Latent space dimension
    size_t num_qubits;                  // Number of qubits to use
    size_t num_layers;                  // Number of quantum layers
    size_t batch_size;                  // Batch size for training
    double learning_rate;               // Learning rate
    bool use_gpu;                       // Whether to use GPU acceleration
    bool enable_error_mitigation;       // Whether to use error mitigation
    size_t num_shots;                   // Number of measurement shots
} task_config_t;

// Performance metrics
typedef struct {
    double accuracy;                    // Classification accuracy
    double precision;                   // Precision score
    double recall;                      // Recall score
    double f1_score;                    // F1 score
    double mse;                         // Mean squared error
    double mae;                         // Mean absolute error
    double training_time;               // Training time in seconds
    double inference_time;              // Inference time in seconds
    size_t memory_usage;                // Memory usage in bytes
    double quantum_advantage;           // Quantum vs classical speedup
} task_metrics_t;

// Training state
typedef struct {
    size_t current_epoch;               // Current training epoch
    size_t total_epochs;                // Total number of epochs
    double current_loss;                // Current loss value
    double best_loss;                   // Best loss achieved
    double learning_rate;               // Current learning rate
    size_t iterations_without_improvement; // Iterations without improvement
    bool converged;                     // Whether training has converged
} training_state_t;

// Opaque handle for learning task
typedef struct quantum_learning_task* learning_task_handle_t;

// Core task functions
learning_task_handle_t quantum_create_learning_task(const task_config_t* config);
void quantum_destroy_learning_task(learning_task_handle_t task);

// Training functions
bool quantum_train_task(learning_task_handle_t task, 
                       const ComplexFloat** features,
                       const ComplexFloat* labels,
                       size_t num_samples);

bool quantum_evaluate_task(learning_task_handle_t task,
                         const ComplexFloat** features,
                         const ComplexFloat* labels,
                         size_t num_samples,
                         task_metrics_t* metrics);

bool quantum_predict_task(learning_task_handle_t task,
                        const ComplexFloat* input,
                        ComplexFloat* output);

// Training control functions
bool quantum_get_training_state(learning_task_handle_t task,
                              training_state_t* state);

bool quantum_update_learning_rate(learning_task_handle_t task,
                                double new_learning_rate);

bool quantum_early_stop(learning_task_handle_t task);

// Performance comparison functions
bool quantum_compare_classical(learning_task_handle_t task,
                             const ComplexFloat** features,
                             const ComplexFloat* labels,
                             size_t num_samples,
                             task_metrics_t* classical_metrics);

bool quantum_analyze_advantage(learning_task_handle_t task,
                             const task_metrics_t* quantum_metrics,
                             const task_metrics_t* classical_metrics,
                             double* advantage_score);

// Error mitigation functions
bool quantum_enable_error_mitigation(learning_task_handle_t task);
bool quantum_disable_error_mitigation(learning_task_handle_t task);
bool quantum_get_error_rates(learning_task_handle_t task,
                           double* error_rates,
                           size_t* num_rates);

#endif // LEARNING_TASK_H
