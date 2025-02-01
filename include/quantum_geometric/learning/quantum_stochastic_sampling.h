#ifndef QUANTUM_STOCHASTIC_SAMPLING_H
#define QUANTUM_STOCHASTIC_SAMPLING_H

#include <complex.h>
#include <stddef.h>
#include <stdbool.h>
#include "quantum_geometric/core/hierarchical_matrix.h"
#include "quantum_geometric/core/quantum_geometric_core.h"

// Geometric transform types
typedef enum {
    GEOMETRIC_TRANSFORM_DIFFUSION,
    GEOMETRIC_TRANSFORM_FLOW,
    GEOMETRIC_TRANSFORM_SCORE
} geometric_transform_t;

// Device types
typedef enum {
    QUANTUM_DEVICE_CPU,
    QUANTUM_DEVICE_GPU
} quantum_device_t;

// Diffusion configuration
typedef struct {
    double t_min;                // Minimum time
    double t_max;                // Maximum time
    size_t num_steps;           // Number of diffusion steps
    double beta_min;            // Minimum noise level
    double beta_max;            // Maximum noise level
    bool use_cosine_schedule;   // Use cosine scheduling
    geometric_transform_t transform_type; // Type of geometric transform
} DiffusionConfig;

// Physics-informed neural network configuration
typedef struct {
    size_t input_dim;          // Input dimension
    size_t hidden_dim;         // Hidden layer dimension
    size_t num_layers;         // Number of layers
    double learning_rate;      // Learning rate
    size_t batch_size;         // Batch size
    size_t max_epochs;         // Maximum training epochs
    double weight_decay;       // Weight decay factor
    size_t attention_heads;    // Number of attention heads
    size_t head_dim;          // Dimension per head
    size_t tensor_bond_dim;    // Tensor network bond dimension
} PINNConfig;

// Langevin Monte Carlo configuration
typedef struct {
    double step_size;          // Step size for updates
    size_t num_steps;         // Number of MCMC steps
    size_t num_chains;        // Number of parallel chains
    bool adapt_step_size;     // Adaptive step size
    quantum_device_t device;  // Computation device
} LMCConfig;

// Performance metrics
typedef struct {
    double training_time;      // Training time in seconds
    double memory_usage_mb;    // Memory usage in MB
    double gpu_utilization;    // GPU utilization percentage
    double sampling_efficiency; // Effective sample size per second
} PerformanceMetrics;

// Opaque sampler handle
typedef struct StochasticSampler StochasticSampler;

// Creation and destruction
StochasticSampler* stochastic_sampler_create(const DiffusionConfig* diffusion_config,
                                            const PINNConfig* pinn_config,
                                            const LMCConfig* lmc_config);
void stochastic_sampler_free(StochasticSampler* sampler);

// Initialization and training
int stochastic_sampler_init(StochasticSampler* sampler,
                           double (*log_prob)(const double*, size_t),
                           void (*log_prob_grad)(const double*, size_t, double*),
                           size_t dim);
int stochastic_sampler_train(StochasticSampler* sampler);

// Sampling and state access
int stochastic_sampler_sample(StochasticSampler* sampler,
                             size_t num_samples,
                             double* samples);
const QuantumState* stochastic_sampler_get_state(const StochasticSampler* sampler);

// Performance monitoring
const PerformanceMetrics* stochastic_sampler_get_metrics(const StochasticSampler* sampler);

// Core mathematical operations
void forward(double complex* output,
            const double complex* input,
            const double complex* weights,
            size_t n);
void compute_residual(double complex* residual,
                     const double complex* output,
                     const double complex* target,
                     size_t n);
void generate_collocation_points(double complex* points, size_t n);

// Hierarchical matrix operations
HierarchicalMatrix* convert_to_hierarchical(const double complex* data, size_t n);
void convert_from_hierarchical(double complex* data,
                             const HierarchicalMatrix* matrix);
void forward_hierarchical(HierarchicalMatrix* output,
                         const HierarchicalMatrix* input,
                         const HierarchicalMatrix* weights);

#endif // QUANTUM_STOCHASTIC_SAMPLING_H
