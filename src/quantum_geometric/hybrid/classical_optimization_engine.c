#include "quantum_geometric/hybrid/classical_optimization_engine.h"
#include "quantum_geometric/core/quantum_geometric_operations.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// OpenMP support is handled by quantum_geometric_operations.h which provides
// fallback macros when _OPENMP is not defined

// Optimization parameters
#define MAX_ITERATIONS 1000
#define CONVERGENCE_THRESHOLD 1e-6
#define LEARNING_RATE 0.01
#define MOMENTUM 0.9

// Forward declarations for GPU functions
void compute_natural_gradient_gpu(const double* fisher_matrix,
                               const double* gradients,
                               double* natural_gradient,
                               size_t num_parameters);

void compute_natural_gradient_cpu(const double* fisher_matrix,
                               const double* gradients,
                               double* natural_gradient,
                               size_t num_parameters);

// Private context extension for internal state
typedef struct {
    size_t iteration;
    bool converged;
    double* parameters;
    
    // ADAM specific parameters
    double* m;  // First moment
    double* v;  // Second moment
    
    // L-BFGS specific parameters
    double** s_vectors;  // Parameter differences
    double** y_vectors;  // Gradient differences
    size_t lbfgs_memory;
    size_t current_vector;
    
    // Natural gradient specific
    double* fisher_matrix;
    
    // Momentum specific
    double* momentum_buffer;
} OptimizationContextExt;

// Forward declarations for static functions
static bool init_adam_parameters(OptimizationContext* ctx, OptimizationContextExt* ext);
static bool init_lbfgs_parameters(OptimizationContext* ctx, OptimizationContextExt* ext);
static bool init_natural_gradient_parameters(OptimizationContext* ctx, OptimizationContextExt* ext);
static void update_adam(OptimizationContext* ctx, OptimizationContextExt* ext);
static void update_lbfgs(OptimizationContext* ctx, OptimizationContextExt* ext);
static void update_sgd(OptimizationContext* ctx, OptimizationContextExt* ext);
static void update_natural_gradient(OptimizationContext* ctx, OptimizationContextExt* ext);
static void update_rmsprop(OptimizationContext* ctx, OptimizationContextExt* ext);
static void update_adagrad(OptimizationContext* ctx, OptimizationContextExt* ext);
static void update_adadelta(OptimizationContext* ctx, OptimizationContextExt* ext);
static void update_nadam(OptimizationContext* ctx, OptimizationContextExt* ext);
static bool check_convergence(OptimizationContext* ctx, OptimizationContextExt* ext, double value);
static double dot_product(const double* a, const double* b, size_t n);
static void update_fisher_matrix(OptimizationContext* ctx, OptimizationContextExt* ext);

// Convert int optimizer type to enum
optimizer_type_t convert_optimizer_type(int optimizer_type) {
    switch (optimizer_type) {
        case 0:
            return OPTIMIZER_ADAM;
        case 1:
            return OPTIMIZER_SGD;
        case 2:
            return OPTIMIZER_RMSPROP;
        case 3:
            return OPTIMIZER_ADAGRAD;
        case 4:
            return OPTIMIZER_ADADELTA;
        case 5:
            return OPTIMIZER_NADAM;
        default:
            return OPTIMIZER_ADAM;
    }
}

// Initialize classical optimization engine
OptimizationContext* init_classical_optimizer(optimizer_type_t type,
                                           size_t num_parameters,
                                           bool use_gpu) {
    OptimizationContext* ctx = malloc(sizeof(OptimizationContext));
    OptimizationContextExt* ext = malloc(sizeof(OptimizationContextExt));
    if (!ctx || !ext) {
        free(ctx);
        free(ext);
        return NULL;
    }
    
    ctx->type = type;
    ctx->num_parameters = num_parameters;
    ctx->learning_rate = LEARNING_RATE;
    ctx->use_gpu = use_gpu;
    ctx->beta1 = 0.9;
    ctx->beta2 = 0.999;
    ctx->epsilon = 1e-8;
    
    // Initialize gradients, momentum, velocity
    ctx->gradients = aligned_alloc(64, num_parameters * sizeof(double));
    ctx->momentum = aligned_alloc(64, num_parameters * sizeof(double));
    ctx->velocity = aligned_alloc(64, num_parameters * sizeof(double));
    
    if (!ctx->gradients || !ctx->momentum || !ctx->velocity) {
        cleanup_classical_optimizer(ctx);
        free(ext);
        return NULL;
    }
    
    // Initialize extension
    ext->iteration = 0;
    ext->converged = false;
    ext->parameters = aligned_alloc(64, num_parameters * sizeof(double));
    ext->momentum_buffer = aligned_alloc(64, num_parameters * sizeof(double));
    
    if (!ext->parameters || !ext->momentum_buffer) {
        cleanup_classical_optimizer(ctx);
        free(ext->parameters);
        free(ext->momentum_buffer);
        free(ext);
        return NULL;
    }
    
    ctx->optimizer_state = ext;
    
    // Initialize algorithm-specific parameters
    switch (type) {
        case OPTIMIZER_ADAM:
            if (!init_adam_parameters(ctx, ext)) {
                cleanup_classical_optimizer(ctx);
                return NULL;
            }
            break;
            
        case OPTIMIZER_LBFGS:
            if (!init_lbfgs_parameters(ctx, ext)) {
                cleanup_classical_optimizer(ctx);
                return NULL;
            }
            break;
            
        case OPTIMIZER_NATURAL_GRADIENT:
            if (!init_natural_gradient_parameters(ctx, ext)) {
                cleanup_classical_optimizer(ctx);
                return NULL;
            }
            break;
            
        default:
            break;
    }
    
    return ctx;
}

// Initialize ADAM parameters
static bool init_adam_parameters(OptimizationContext* ctx, OptimizationContextExt* ext) {
    ext->m = aligned_alloc(64, ctx->num_parameters * sizeof(double));
    ext->v = aligned_alloc(64, ctx->num_parameters * sizeof(double));
    
    if (!ext->m || !ext->v) return false;
    
    memset(ext->m, 0, ctx->num_parameters * sizeof(double));
    memset(ext->v, 0, ctx->num_parameters * sizeof(double));
    
    return true;
}

// Initialize L-BFGS parameters
static bool init_lbfgs_parameters(OptimizationContext* ctx, OptimizationContextExt* ext) {
    ext->lbfgs_memory = 10;  // Store last 10 iterations
    ext->current_vector = 0;
    
    // Allocate memory for s and y vectors
    ext->s_vectors = malloc(ext->lbfgs_memory * sizeof(double*));
    ext->y_vectors = malloc(ext->lbfgs_memory * sizeof(double*));
    
    if (!ext->s_vectors || !ext->y_vectors) return false;
    
    for (size_t i = 0; i < ext->lbfgs_memory; i++) {
        ext->s_vectors[i] = aligned_alloc(64,
            ctx->num_parameters * sizeof(double));
        ext->y_vectors[i] = aligned_alloc(64,
            ctx->num_parameters * sizeof(double));
        
        if (!ext->s_vectors[i] || !ext->y_vectors[i]) return false;
    }
    
    return true;
}

// Initialize natural gradient parameters
static bool init_natural_gradient_parameters(OptimizationContext* ctx, OptimizationContextExt* ext) {
    size_t matrix_size = ctx->num_parameters * ctx->num_parameters;
    ext->fisher_matrix = aligned_alloc(64,
        matrix_size * sizeof(double));
    
    if (!ext->fisher_matrix) return false;
    
    // Initialize to identity matrix
    for (size_t i = 0; i < ctx->num_parameters; i++) {
        for (size_t j = 0; j < ctx->num_parameters; j++) {
            ext->fisher_matrix[i * ctx->num_parameters + j] =
                (i == j) ? 1.0 : 0.0;
        }
    }
    
    return true;
}

// Optimize parameters
int optimize_parameters(OptimizationContext* ctx,
                      OptimizationObjective objective,
                      void* objective_data) {
    if (!ctx || !objective.function) return -1;
    
    OptimizationContextExt* ext = (OptimizationContextExt*)ctx->optimizer_state;
    if (!ext) return -1;
    
    while (!ext->converged && ext->iteration < MAX_ITERATIONS) {
        // Compute objective and gradients
        double value = objective.function(ext->parameters,
                                       ctx->gradients,
                                       objective_data);
        
        // Update parameters based on algorithm
        switch (ctx->type) {
            case OPTIMIZER_ADAM:
                update_adam(ctx, ext);
                break;

            case OPTIMIZER_LBFGS:
                update_lbfgs(ctx, ext);
                break;

            case OPTIMIZER_SGD:
                update_sgd(ctx, ext);
                break;

            case OPTIMIZER_NATURAL_GRADIENT:
                update_natural_gradient(ctx, ext);
                break;

            case OPTIMIZER_RMSPROP:
                update_rmsprop(ctx, ext);
                break;

            case OPTIMIZER_ADAGRAD:
                update_adagrad(ctx, ext);
                break;

            case OPTIMIZER_ADADELTA:
                update_adadelta(ctx, ext);
                break;

            case OPTIMIZER_NADAM:
                update_nadam(ctx, ext);
                break;

            default:
                update_adam(ctx, ext);  // Default to ADAM
                break;
        }
        
        // Check convergence
        check_convergence(ctx, ext, value);
        ext->iteration++;
    }
    
    return ext->converged ? 0 : -1;
}

// Update parameters using ADAM
static void update_adam(OptimizationContext* ctx, OptimizationContextExt* ext) {
    double beta1_t = pow(ctx->beta1, ext->iteration + 1);
    double beta2_t = pow(ctx->beta2, ext->iteration + 1);
    
    #pragma omp parallel for if(ctx->num_parameters > 1000)
    for (size_t i = 0; i < ctx->num_parameters; i++) {
        // Update biased first moment estimate
        ext->m[i] = ctx->beta1 * ext->m[i] +
                   (1.0 - ctx->beta1) * ctx->gradients[i];
        
        // Update biased second raw moment estimate
        ext->v[i] = ctx->beta2 * ext->v[i] +
                   (1.0 - ctx->beta2) * ctx->gradients[i] * ctx->gradients[i];
        
        // Compute bias-corrected first moment estimate
        double m_hat = ext->m[i] / (1.0 - beta1_t);
        
        // Compute bias-corrected second raw moment estimate
        double v_hat = ext->v[i] / (1.0 - beta2_t);
        
        // Update parameters
        ext->parameters[i] -= ctx->learning_rate * m_hat /
                            (sqrt(v_hat) + ctx->epsilon);
    }
}

// Update parameters using L-BFGS
static void update_lbfgs(OptimizationContext* ctx, OptimizationContextExt* ext) {
    // Store s and y vectors
    size_t idx = ext->current_vector;
    
    #pragma omp parallel for if(ctx->num_parameters > 1000)
    for (size_t i = 0; i < ctx->num_parameters; i++) {
        ext->s_vectors[idx][i] = -ctx->learning_rate * ctx->gradients[i];
        ext->y_vectors[idx][i] = -ctx->gradients[i];
    }
    
    // Two-loop recursion
    double* q = malloc(ctx->num_parameters * sizeof(double));
    double* alpha = malloc(ext->lbfgs_memory * sizeof(double));
    
    memcpy(q, ctx->gradients, ctx->num_parameters * sizeof(double));
    
    // First loop
    for (int i = ext->lbfgs_memory - 1; i >= 0; i--) {
        double rho = 1.0 / dot_product(ext->y_vectors[i],
                                     ext->s_vectors[i],
                                     ctx->num_parameters);
        alpha[i] = rho * dot_product(ext->s_vectors[i],
                                   q,
                                   ctx->num_parameters);
        
        #pragma omp parallel for if(ctx->num_parameters > 1000)
        for (size_t j = 0; j < ctx->num_parameters; j++) {
            q[j] -= alpha[i] * ext->y_vectors[i][j];
        }
    }
    
    // Second loop
    for (size_t i = 0; i < ext->lbfgs_memory; i++) {
        double rho = 1.0 / dot_product(ext->y_vectors[i],
                                     ext->s_vectors[i],
                                     ctx->num_parameters);
        double beta = rho * dot_product(ext->y_vectors[i],
                                      q,
                                      ctx->num_parameters);
        
        #pragma omp parallel for if(ctx->num_parameters > 1000)
        for (size_t j = 0; j < ctx->num_parameters; j++) {
            q[j] += ext->s_vectors[i][j] * (alpha[i] - beta);
        }
    }
    
    // Update parameters
    #pragma omp parallel for if(ctx->num_parameters > 1000)
    for (size_t i = 0; i < ctx->num_parameters; i++) {
        ext->parameters[i] -= q[i];
    }
    
    free(q);
    free(alpha);
    
    // Update current vector index
    ext->current_vector = (ext->current_vector + 1) % ext->lbfgs_memory;
}

// Update parameters using SGD with momentum
static void update_sgd(OptimizationContext* ctx, OptimizationContextExt* ext) {
    #pragma omp parallel for if(ctx->num_parameters > 1000)
    for (size_t i = 0; i < ctx->num_parameters; i++) {
        ext->momentum_buffer[i] = MOMENTUM * ext->momentum_buffer[i] +
                                ctx->learning_rate * ctx->gradients[i];
        ext->parameters[i] -= ext->momentum_buffer[i];
    }
}

// Update parameters using natural gradient
static void update_natural_gradient(OptimizationContext* ctx, OptimizationContextExt* ext) {
    // Update Fisher information matrix
    update_fisher_matrix(ctx, ext);
    
    // Compute natural gradient
    double* natural_gradient = malloc(ctx->num_parameters * sizeof(double));
    
    if (ctx->use_gpu) {
        compute_natural_gradient_gpu(ext->fisher_matrix,
                                  ctx->gradients,
                                  natural_gradient,
                                  ctx->num_parameters);
    } else {
        compute_natural_gradient_cpu(ext->fisher_matrix,
                                  ctx->gradients,
                                  natural_gradient,
                                  ctx->num_parameters);
    }
    
    // Update parameters
    #pragma omp parallel for if(ctx->num_parameters > 1000)
    for (size_t i = 0; i < ctx->num_parameters; i++) {
        ext->parameters[i] -= ctx->learning_rate * natural_gradient[i];
    }
    
    free(natural_gradient);
}

// Check convergence
static bool check_convergence(OptimizationContext* ctx,
                            OptimizationContextExt* ext,
                            double current_value) {
    double grad_norm = 0.0;
    
    #pragma omp parallel for reduction(+:grad_norm)
    for (size_t i = 0; i < ctx->num_parameters; i++) {
        grad_norm += ctx->gradients[i] * ctx->gradients[i];
    }
    
    grad_norm = sqrt(grad_norm);
    ext->converged = grad_norm < CONVERGENCE_THRESHOLD;
    return ext->converged;
}

// Helper functions
static double dot_product(const double* a,
                        const double* b,
                        size_t n) {
    double result = 0.0;
    
    #pragma omp parallel for reduction(+:result)
    for (size_t i = 0; i < n; i++) {
        result += a[i] * b[i];
    }
    
    return result;
}

static void update_fisher_matrix(OptimizationContext* ctx, OptimizationContextExt* ext) {
    size_t n = ctx->num_parameters;

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            ext->fisher_matrix[i * n + j] =
                ctx->gradients[i] * ctx->gradients[j];
        }
    }
}

// Update parameters using RMSprop
static void update_rmsprop(OptimizationContext* ctx, OptimizationContextExt* ext) {
    double decay = 0.99;

    #pragma omp parallel for if(ctx->num_parameters > 1000)
    for (size_t i = 0; i < ctx->num_parameters; i++) {
        // Update running average of squared gradients
        ext->v[i] = decay * ext->v[i] +
                   (1.0 - decay) * ctx->gradients[i] * ctx->gradients[i];

        // Update parameters
        ext->parameters[i] -= ctx->learning_rate * ctx->gradients[i] /
                            (sqrt(ext->v[i]) + ctx->epsilon);
    }
}

// Update parameters using AdaGrad
static void update_adagrad(OptimizationContext* ctx, OptimizationContextExt* ext) {
    #pragma omp parallel for if(ctx->num_parameters > 1000)
    for (size_t i = 0; i < ctx->num_parameters; i++) {
        // Accumulate squared gradients
        ext->v[i] += ctx->gradients[i] * ctx->gradients[i];

        // Update parameters with adaptive learning rate
        ext->parameters[i] -= ctx->learning_rate * ctx->gradients[i] /
                            (sqrt(ext->v[i]) + ctx->epsilon);
    }
}

// Update parameters using AdaDelta
static void update_adadelta(OptimizationContext* ctx, OptimizationContextExt* ext) {
    double rho = 0.95;

    #pragma omp parallel for if(ctx->num_parameters > 1000)
    for (size_t i = 0; i < ctx->num_parameters; i++) {
        // Accumulate gradient squared
        ext->v[i] = rho * ext->v[i] +
                   (1.0 - rho) * ctx->gradients[i] * ctx->gradients[i];

        // Compute parameter update (using accumulated delta squared from m)
        double delta = sqrt(ext->m[i] + ctx->epsilon) /
                      sqrt(ext->v[i] + ctx->epsilon) * ctx->gradients[i];

        // Accumulate updates squared
        ext->m[i] = rho * ext->m[i] + (1.0 - rho) * delta * delta;

        // Update parameters
        ext->parameters[i] -= delta;
    }
}

// Update parameters using NAdam (Nesterov-accelerated Adam)
static void update_nadam(OptimizationContext* ctx, OptimizationContextExt* ext) {
    double beta1_t = pow(ctx->beta1, ext->iteration + 1);
    double beta2_t = pow(ctx->beta2, ext->iteration + 1);

    #pragma omp parallel for if(ctx->num_parameters > 1000)
    for (size_t i = 0; i < ctx->num_parameters; i++) {
        // Update biased first moment estimate
        ext->m[i] = ctx->beta1 * ext->m[i] +
                   (1.0 - ctx->beta1) * ctx->gradients[i];

        // Update biased second raw moment estimate
        ext->v[i] = ctx->beta2 * ext->v[i] +
                   (1.0 - ctx->beta2) * ctx->gradients[i] * ctx->gradients[i];

        // Compute bias-corrected first moment estimate
        double m_hat = ext->m[i] / (1.0 - beta1_t);

        // Compute bias-corrected second raw moment estimate
        double v_hat = ext->v[i] / (1.0 - beta2_t);

        // Nesterov momentum term
        double m_nesterov = ctx->beta1 * m_hat +
                           (1.0 - ctx->beta1) * ctx->gradients[i] / (1.0 - beta1_t);

        // Update parameters
        ext->parameters[i] -= ctx->learning_rate * m_nesterov /
                            (sqrt(v_hat) + ctx->epsilon);
    }
}

// Compute natural gradient on CPU using conjugate gradient
void compute_natural_gradient_cpu(const double* fisher_matrix,
                               const double* gradients,
                               double* natural_gradient,
                               size_t num_parameters) {
    // Solve F * ng = g using conjugate gradient method
    // where F is Fisher matrix, ng is natural gradient, g is gradient

    double* r = malloc(num_parameters * sizeof(double));
    double* p = malloc(num_parameters * sizeof(double));
    double* Ap = malloc(num_parameters * sizeof(double));

    if (!r || !p || !Ap) {
        free(r);
        free(p);
        free(Ap);
        // Fallback to regular gradient
        memcpy(natural_gradient, gradients, num_parameters * sizeof(double));
        return;
    }

    // Initialize: x = 0, r = g - F*x = g, p = r
    memset(natural_gradient, 0, num_parameters * sizeof(double));
    memcpy(r, gradients, num_parameters * sizeof(double));
    memcpy(p, gradients, num_parameters * sizeof(double));

    double rsold = dot_product(r, r, num_parameters);
    const int max_iter = 100;
    const double tol = 1e-8;

    for (int iter = 0; iter < max_iter; iter++) {
        // Compute Ap = F * p
        #pragma omp parallel for
        for (size_t i = 0; i < num_parameters; i++) {
            Ap[i] = 0.0;
            for (size_t j = 0; j < num_parameters; j++) {
                Ap[i] += fisher_matrix[i * num_parameters + j] * p[j];
            }
        }

        double pAp = dot_product(p, Ap, num_parameters);
        if (fabs(pAp) < 1e-15) break;

        double alpha = rsold / pAp;

        // Update solution and residual
        #pragma omp parallel for
        for (size_t i = 0; i < num_parameters; i++) {
            natural_gradient[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        double rsnew = dot_product(r, r, num_parameters);
        if (sqrt(rsnew) < tol) break;

        // Update search direction
        double beta = rsnew / rsold;
        #pragma omp parallel for
        for (size_t i = 0; i < num_parameters; i++) {
            p[i] = r[i] + beta * p[i];
        }

        rsold = rsnew;
    }

    free(r);
    free(p);
    free(Ap);
}

// Compute natural gradient on GPU (fallback to CPU when GPU not available)
void compute_natural_gradient_gpu(const double* fisher_matrix,
                               const double* gradients,
                               double* natural_gradient,
                               size_t num_parameters) {
    // When CUDA is not available, fall back to CPU implementation
#ifdef CUDA_AVAILABLE
    // CUDA implementation would go here using cuBLAS/cuSOLVER
    // For now, fall back to CPU
#endif
    compute_natural_gradient_cpu(fisher_matrix, gradients, natural_gradient, num_parameters);
}

// Clean up optimization engine
void cleanup_classical_optimizer(OptimizationContext* ctx) {
    if (!ctx) return;
    
    OptimizationContextExt* ext = (OptimizationContextExt*)ctx->optimizer_state;
    if (!ext) {
        free(ctx);
        return;
    }
    
    free(ctx->gradients);
    free(ctx->momentum);
    free(ctx->velocity);
    
    free(ext->parameters);
    free(ext->momentum_buffer);
    
    switch (ctx->type) {
        case OPTIMIZER_ADAM:
            free(ext->m);
            free(ext->v);
            break;
            
        case OPTIMIZER_LBFGS:
            if (ext->s_vectors) {
                for (size_t i = 0; i < ext->lbfgs_memory; i++) {
                    free(ext->s_vectors[i]);
                }
                free(ext->s_vectors);
            }
            if (ext->y_vectors) {
                for (size_t i = 0; i < ext->lbfgs_memory; i++) {
                    free(ext->y_vectors[i]);
                }
                free(ext->y_vectors);
            }
            break;
            
        case OPTIMIZER_NATURAL_GRADIENT:
            free(ext->fisher_matrix);
            break;
            
        default:
            break;
    }
    
    free(ext);
    free(ctx);
}
