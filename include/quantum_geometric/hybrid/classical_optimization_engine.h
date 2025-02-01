#ifndef CLASSICAL_OPTIMIZATION_ENGINE_H
#define CLASSICAL_OPTIMIZATION_ENGINE_H

#include <stdbool.h>
#include <stddef.h>

// Forward declarations
struct OptimizationContext;
typedef struct OptimizationContext OptimizationContext;

// Optimizer types
typedef enum {
    OPTIMIZER_ADAM,
    OPTIMIZER_SGD,
    OPTIMIZER_RMSPROP,
    OPTIMIZER_ADAGRAD,
    OPTIMIZER_ADADELTA,
    OPTIMIZER_NADAM,
    OPTIMIZER_LBFGS,
    OPTIMIZER_NATURAL_GRADIENT,
    OPTIMIZER_CUSTOM
} optimizer_type_t;

// Optimization objective function type
typedef double (*objective_function_t)(const double* parameters,
                                     double* gradients,
                                     void* data);

// Optimization objective
typedef struct {
    objective_function_t function;
    void* data;
} OptimizationObjective;

// Optimization context
typedef struct OptimizationContext {
    optimizer_type_t type;
    size_t num_parameters;
    double learning_rate;
    double beta1;
    double beta2;
    double epsilon;
    bool use_gpu;
    double* gradients;
    double* momentum;
    double* velocity;
    void* optimizer_state;
} OptimizationContext;

// GPU functions
void compute_natural_gradient_gpu(const double* fisher_matrix,
                               const double* gradients,
                               double* natural_gradient,
                               size_t num_parameters);

void compute_natural_gradient_cpu(const double* fisher_matrix,
                               const double* gradients,
                               double* natural_gradient,
                               size_t num_parameters);

// Initialize optimizer
OptimizationContext* init_classical_optimizer(optimizer_type_t type,
                                           size_t num_parameters,
                                           bool use_gpu);

// Optimize parameters
int optimize_parameters(OptimizationContext* ctx,
                      OptimizationObjective objective,
                      void* objective_data);

// Cleanup optimizer
void cleanup_classical_optimizer(OptimizationContext* ctx);

#endif // CLASSICAL_OPTIMIZATION_ENGINE_H
