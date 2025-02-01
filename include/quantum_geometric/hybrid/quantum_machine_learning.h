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
    double** weights;
    double** biases;
    void* activation_functions;
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
                                const double* targets);
double compute_mse_loss(const double* outputs,
                       const double* targets);
double compute_reconstruction_loss(const double* outputs,
                                 const double* targets);

bool check_early_stopping(double validation_loss,
                         const TrainingConfig* config);

#endif // QUANTUM_MACHINE_LEARNING_H
