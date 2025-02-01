/**
 * @file test_quantum_stochastic_sampling.c
 * @brief Tests for stochastic sampling with quantum geometric infrastructure
 */

#include "quantum_geometric/learning/stochastic_sampling.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/core/quantum_operations.h"
#include "quantum_geometric/core/performance_operations.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

// Simple Gaussian distribution for testing
static double gaussian_log_prob(const double* x, size_t dim) {
    double sum = 0.0;
    for (size_t i = 0; i < dim; i++) {
        sum += -0.5 * x[i] * x[i];
    }
    return sum - 0.5 * dim * log(2.0 * M_PI);
}

static void gaussian_log_prob_grad(const double* x, size_t dim, double* grad) {
    for (size_t i = 0; i < dim; i++) {
        grad[i] = -x[i];
    }
}

// Test helper functions
static void test_sampler_creation() {
    printf("Testing sampler creation...\n");
    
    DiffusionConfig diffusion_config = {
        .t_min = 0.001,
        .t_max = 0.999,
        .num_steps = 100,
        .beta_min = 0.1,
        .beta_max = 20.0,
        .use_cosine_schedule = true,
        .transform_type = GEOMETRIC_TRANSFORM_DIFFUSION
    };
    
    PINNConfig pinn_config = {
        .input_dim = 2,
        .hidden_dim = 64,
        .num_layers = 3,
        .learning_rate = 0.001,
        .batch_size = 64,
        .max_epochs = 100,
        .weight_decay = 0.0001,
        .attention_heads = 4,
        .head_dim = 32,
        .tensor_bond_dim = 16
    };
    
    LMCConfig lmc_config = {
        .step_size = 0.01,
        .num_steps = 50,
        .num_chains = 5,
        .adapt_step_size = true,
        .device = QUANTUM_DEVICE_GPU
    };
    
    StochasticSampler* sampler = stochastic_sampler_create(
        &diffusion_config,
        &pinn_config,
        &lmc_config
    );
    
    assert(sampler != NULL);
    stochastic_sampler_free(sampler);
}

static void test_quantum_state_integration() {
    printf("Testing quantum state integration...\n");
    
    // Create sampler with minimal configuration
    DiffusionConfig diffusion_config = {
        .t_min = 0.001,
        .t_max = 0.999,
        .num_steps = 10,
        .beta_min = 0.1,
        .beta_max = 20.0,
        .use_cosine_schedule = true,
        .transform_type = GEOMETRIC_TRANSFORM_DIFFUSION
    };
    
    PINNConfig pinn_config = {
        .input_dim = 1,
        .hidden_dim = 32,
        .num_layers = 2,
        .learning_rate = 0.001,
        .batch_size = 32,
        .max_epochs = 10,
        .weight_decay = 0.0001,
        .attention_heads = 2,
        .head_dim = 16,
        .tensor_bond_dim = 8
    };
    
    LMCConfig lmc_config = {
        .step_size = 0.01,
        .num_steps = 10,
        .num_chains = 2,
        .adapt_step_size = true,
        .device = QUANTUM_DEVICE_GPU
    };
    
    StochasticSampler* sampler = stochastic_sampler_create(
        &diffusion_config,
        &pinn_config,
        &lmc_config
    );
    
    assert(sampler != NULL);
    
    // Initialize with 1D Gaussian
    int result = stochastic_sampler_init(
        sampler,
        gaussian_log_prob,
        gaussian_log_prob_grad,
        1
    );
    assert(result == 0);
    
    // Get quantum state and verify properties
    const QuantumState* state = stochastic_sampler_get_state(sampler);
    assert(state != NULL);
    assert(quantum_state_get_dim(state) == 1);
    assert(quantum_state_get_fidelity(state) > 0.0);
    assert(quantum_state_get_purity(state) > 0.0);
    
    stochastic_sampler_free(sampler);
}

static void test_geometric_transform() {
    printf("Testing geometric transform...\n");
    
    // Create sampler with diffusion transform
    DiffusionConfig diffusion_config = {
        .t_min = 0.001,
        .t_max = 0.999,
        .num_steps = 10,
        .beta_min = 0.1,
        .beta_max = 20.0,
        .use_cosine_schedule = true,
        .transform_type = GEOMETRIC_TRANSFORM_DIFFUSION
    };
    
    PINNConfig pinn_config = {
        .input_dim = 1,
        .hidden_dim = 32,
        .num_layers = 2,
        .learning_rate = 0.001,
        .batch_size = 32,
        .max_epochs = 10,
        .weight_decay = 0.0001,
        .attention_heads = 2,
        .head_dim = 16,
        .tensor_bond_dim = 8
    };
    
    LMCConfig lmc_config = {
        .step_size = 0.01,
        .num_steps = 10,
        .num_chains = 2,
        .adapt_step_size = true,
        .device = QUANTUM_DEVICE_GPU
    };
    
    StochasticSampler* sampler = stochastic_sampler_create(
        &diffusion_config,
        &pinn_config,
        &lmc_config
    );
    
    assert(sampler != NULL);
    
    // Initialize and train briefly
    int result = stochastic_sampler_init(
        sampler,
        gaussian_log_prob,
        gaussian_log_prob_grad,
        1
    );
    assert(result == 0);
    
    result = stochastic_sampler_train(sampler);
    assert(result == 0);
    
    // Generate a few samples
    size_t num_samples = 10;
    double* samples = malloc(num_samples * sizeof(double));
    result = stochastic_sampler_sample(sampler, num_samples, samples);
    assert(result == 0);
    
    // Verify samples are finite
    for (size_t i = 0; i < num_samples; i++) {
        assert(isfinite(samples[i]));
    }
    
    free(samples);
    stochastic_sampler_free(sampler);
}

static void test_performance_metrics() {
    printf("Testing performance metrics...\n");
    
    // Create sampler with minimal configuration
    DiffusionConfig diffusion_config = {
        .t_min = 0.001,
        .t_max = 0.999,
        .num_steps = 10,
        .beta_min = 0.1,
        .beta_max = 20.0,
        .use_cosine_schedule = true,
        .transform_type = GEOMETRIC_TRANSFORM_DIFFUSION
    };
    
    PINNConfig pinn_config = {
        .input_dim = 1,
        .hidden_dim = 32,
        .num_layers = 2,
        .learning_rate = 0.001,
        .batch_size = 32,
        .max_epochs = 10,
        .weight_decay = 0.0001,
        .attention_heads = 2,
        .head_dim = 16,
        .tensor_bond_dim = 8
    };
    
    LMCConfig lmc_config = {
        .step_size = 0.01,
        .num_steps = 10,
        .num_chains = 2,
        .adapt_step_size = true,
        .device = QUANTUM_DEVICE_GPU
    };
    
    StochasticSampler* sampler = stochastic_sampler_create(
        &diffusion_config,
        &pinn_config,
        &lmc_config
    );
    
    assert(sampler != NULL);
    
    // Initialize and train
    int result = stochastic_sampler_init(
        sampler,
        gaussian_log_prob,
        gaussian_log_prob_grad,
        1
    );
    assert(result == 0);
    
    result = stochastic_sampler_train(sampler);
    assert(result == 0);
    
    // Check metrics
    const PerformanceMetrics* metrics = stochastic_sampler_get_metrics(sampler);
    assert(metrics != NULL);
    assert(metrics->training_time > 0.0);
    assert(metrics->memory_usage_mb > 0.0);
    assert(metrics->gpu_utilization >= 0.0 && metrics->gpu_utilization <= 100.0);
    assert(metrics->sampling_efficiency > 0.0);
    
    stochastic_sampler_free(sampler);
}

int main(void) {
    printf("Running quantum stochastic sampling tests...\n\n");
    
    // Initialize quantum geometric core
    QuantumGeometricCore* core = quantum_geometric_core_create();
    assert(core != NULL);
    
    // Run tests
    test_sampler_creation();
    test_quantum_state_integration();
    test_geometric_transform();
    test_performance_metrics();
    
    // Cleanup
    quantum_geometric_core_free(core);
    
    printf("\nAll tests passed successfully!\n");
    return 0;
}
