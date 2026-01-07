#ifndef QUANTUM_MACHINE_LEARNING_H
#define QUANTUM_MACHINE_LEARNING_H

#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_circuit.h"
#include "quantum_geometric/core/quantum_state.h"
#include "quantum_geometric/core/quantum_geometric_operations.h"
#include "quantum_geometric/hybrid/classical_optimization_engine.h"
#include <stdbool.h>

// QML model types
typedef enum {
    QML_CLASSIFIER,
    QML_REGRESSOR,
    QML_AUTOENCODER
} QMLModelType;

// Activation function types
typedef enum {
    ACTIVATION_NONE,        // Linear (no activation)
    ACTIVATION_RELU,        // Rectified Linear Unit
    ACTIVATION_SIGMOID,     // Sigmoid function
    ACTIVATION_TANH,        // Hyperbolic tangent
    ACTIVATION_SOFTMAX      // Softmax (for output layer)
} ActivationType;

// Network architecture configuration
typedef struct {
    size_t input_size;
    size_t output_size;
    size_t* layer_sizes;
    size_t num_layers;
    size_t num_qubits;
    void* quantum_layers;  // Opaque pointer to quantum layer configuration
} NetworkArchitecture;

// Classical network
typedef struct {
    size_t input_size;
    size_t output_size;
    size_t num_layers;
    size_t* layer_sizes;  // Output size for each layer
    double** weights;
    double** biases;
    ActivationType* activation_functions;  // Activation function for each layer
    double** activations;  // Cached activations from forward pass for backprop
    double* last_input;    // Cached input for gradient computation
} ClassicalNetwork;

// Forward declare OptimizationContext from classical_optimization_engine.h
struct OptimizationContext;
typedef struct OptimizationContext OptimizationContext;

// Training configuration
typedef struct {
    size_t max_epochs;
    size_t batch_size;
    double learning_rate;
    double early_stopping_threshold;
    size_t patience;
    bool use_validation;
} TrainingConfig;

// Dataset structure
typedef struct {
    double* inputs;
    double* targets;
    size_t size;
    size_t input_dim;
    size_t target_dim;
} DataSet;

// QML context
typedef struct QMLContext QMLContext;

// Core functions
QMLContext* init_qml_model(QMLModelType type,
                          size_t num_qubits,
                          const NetworkArchitecture* architecture);

int train_qml_model(QMLContext* ctx,
                   const DataSet* training_data,
                   const DataSet* validation_data,
                   TrainingConfig* config);

double evaluate_model(const QMLContext* ctx,
                     const DataSet* data);

void cleanup_qml_model(QMLContext* ctx);

// Helper functions
ClassicalNetwork* create_classical_network(const NetworkArchitecture* architecture);
void cleanup_classical_network(ClassicalNetwork* network);

OptimizationContext* init_classical_optimizer(optimizer_type_t type,
                                            size_t num_parameters,
                                            bool use_gpu);
void cleanup_classical_optimizer(OptimizationContext* optimizer);

double compute_cross_entropy_loss(const double* outputs,
                                const double* targets,
                                size_t size);
double compute_mse_loss(const double* outputs,
                       const double* targets,
                       size_t size);
double compute_reconstruction_loss(const double* outputs,
                                 const double* targets,
                                 size_t size);

bool check_early_stopping(double validation_loss,
                         const TrainingConfig* config);

// ============================================================================
// High-Level Quantum ML API (for test compatibility)
// ============================================================================

// ML Backend types
typedef enum {
    ML_BACKEND_SIMULATOR,
    ML_BACKEND_IBM,
    ML_BACKEND_RIGETTI,
    ML_BACKEND_IONQ
} ml_backend_type_t;

// Convenience aliases for test compatibility
#define BACKEND_SIMULATOR ML_BACKEND_SIMULATOR
#define BACKEND_IBM ML_BACKEND_IBM
#define BACKEND_RIGETTI ML_BACKEND_RIGETTI
#define BACKEND_IONQ ML_BACKEND_IONQ

// Measurement basis
typedef enum {
    MEASUREMENT_BASIS_COMPUTATIONAL,
    MEASUREMENT_BASIS_CONTINUOUS
} measurement_basis_t;

// Loss function types
typedef enum {
    LOSS_MSE,
    LOSS_CROSS_ENTROPY,
    LOSS_HINGE
} loss_function_t;

// Training status
typedef enum {
    TRAINING_SUCCESS,
    TRAINING_FAILED,
    TRAINING_EARLY_STOPPED
} training_status_t;

// Hardware configuration for ML
typedef struct {
    ml_backend_type_t backend;
    size_t num_qubits;
    struct {
        bool circuit_optimization;
        bool error_mitigation;
        bool continuous_variable;
    } optimization;
} quantum_hardware_config_t;

// Model configuration
typedef struct {
    size_t input_dim;
    size_t output_dim;
    size_t quantum_depth;
    measurement_basis_t measurement_basis;
    struct {
        double learning_rate;
        bool geometric_enhancement;
        loss_function_t loss_function;
    } optimization;
} quantum_model_config_t;

// Training configuration
typedef struct {
    size_t num_epochs;
    size_t batch_size;
    double learning_rate;
    struct {
        bool geometric_enhancement;
        bool error_mitigation;
    } optimization;
} training_config_t;

// Training result
typedef struct {
    training_status_t status;
    double final_loss;
    double* loss_history;
    size_t num_epochs;
} training_result_t;

// Evaluation result
typedef struct {
    double mse;
    double mae;
    double r2_score;
    double accuracy;
} evaluation_result_t;

// Performance metrics (compatible with test)
typedef struct {
    double start_time;
    double end_time;
    double training_time;
    double mse;
    double mae;
    double r2_score;
    size_t memory_used;
} performance_metrics_t;

// Opaque types for ML API
typedef struct quantum_model_t quantum_model_t;
typedef struct classical_model_t classical_model_t;

// Quantum system initialization (returns CORE quantum_system_t directly)
quantum_system_t* quantum_init_system(const quantum_hardware_config_t* config);

// Quantum model functions
quantum_model_t* quantum_model_create(const quantum_model_config_t* config);
void quantum_model_destroy(quantum_model_t* model);
training_result_t quantum_train(quantum_model_t* model,
                               const float* features,
                               const float* targets,
                               size_t num_samples,
                               const training_config_t* config);
evaluation_result_t quantum_evaluate(quantum_model_t* model,
                                    const float* features,
                                    const float* targets,
                                    size_t num_samples);

// Classical model functions (for comparison)
classical_model_t* classical_model_create(size_t input_dim, size_t output_dim);
void classical_model_destroy(classical_model_t* model);
void classical_train(classical_model_t* model,
                    const float* features,
                    const float* targets,
                    size_t num_samples,
                    size_t num_epochs,
                    size_t batch_size);
evaluation_result_t classical_evaluate(classical_model_t* model,
                                      const float* features,
                                      const float* targets,
                                      size_t num_samples);

#endif // QUANTUM_MACHINE_LEARNING_H
