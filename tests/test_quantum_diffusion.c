#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/learning/quantum_stochastic_sampling.h"
#include "quantum_geometric/physics/quantum_state_operations.h"

/**
 * Test suite for physics-informed diffusion sampling
 * Based on Shi et al. 2023 (arXiv:2410.15336)
 */

// Test PDE residual computation
void test_pde_residual() {
    printf("Testing PDE residual computation...\n");
    
    // Create test state
    quantum_state* state = quantum_state_create(2);  // 2 qubits
    ComplexFloat amps[4] = {
        {0.5f, 0.0f},
        {0.5f, 0.0f},
        {0.5f, 0.0f},
        {0.5f, 0.0f}
    };
    quantum_state_set_amplitudes(state, amps, 4);
    
    // Test parameters
    float sigma = 0.1f;
    float t = 0.5f;
    
    // Compute residual
    float residual = compute_pde_residual(state, t, &sigma);
    
    // Verify residual is reasonable
    assert(residual >= 0.0f);
    assert(residual < 1.0f);
    
    quantum_state_destroy(state);
    printf("PDE residual test passed\n\n");
}

// Test PINN drift estimation
void test_drift_estimation() {
    printf("Testing PINN drift estimation...\n");
    
    // Create test state
    quantum_state* state = quantum_state_create(2);
    ComplexFloat amps[4] = {
        {0.5f, 0.0f},
        {0.5f, 0.0f},
        {0.5f, 0.0f},
        {0.5f, 0.0f}
    };
    quantum_state_set_amplitudes(state, amps, 4);
    
    // Create PINN config
    pinn_config config = {
        .hidden_dim = 64,
        .num_layers = 3,
        .activation = ACTIVATION_TANH,
        .learning_rate = 0.001f,
        .tolerance = 1e-6f,
        .params = &(float){0.1f}  // sigma
    };
    
    // Initialize PINN
    assert(pinn_initialize(&config));
    
    // Estimate drift
    float* drift = estimate_drift(state, 0.5f, &config);
    assert(drift != NULL);
    
    // Verify drift values
    for (int i = 0; i < 4; i++) {
        assert(!isnan(drift[i]));
        assert(!isinf(drift[i]));
    }
    
    free(drift);
    pinn_cleanup(&config);
    quantum_state_destroy(state);
    printf("Drift estimation test passed\n\n");
}

// Test state evolution
void test_state_evolution() {
    printf("Testing quantum state evolution...\n");
    
    // Create test state
    quantum_state* state = quantum_state_create(2);
    ComplexFloat amps[4] = {
        {0.5f, 0.0f},
        {0.5f, 0.0f},
        {0.5f, 0.0f},
        {0.5f, 0.0f}
    };
    quantum_state_set_amplitudes(state, amps, 4);
    
    // Evolution parameters
    float dt = 0.01f;
    float sigma = 0.1f;
    float* drift = malloc(4 * sizeof(float));
    for (int i = 0; i < 4; i++) drift[i] = 0.1f;
    
    // Compute geometric phase
    float phase = compute_geometric_phase(state);
    
    // Evolve state
    qgt_error_t err = quantum_state_update(state, drift, sigma, phase, dt);
    assert(err == QGT_SUCCESS);
    
    // Verify state is still normalized
    float norm = 0.0f;
    for (int i = 0; i < 4; i++) {
        ComplexFloat amp = quantum_state_get_amplitude(state, i);
        norm += amp.real * amp.real + amp.imag * amp.imag;
    }
    assert(fabs(norm - 1.0f) < 1e-6f);
    
    free(drift);
    quantum_state_destroy(state);
    printf("State evolution test passed\n\n");
}

// Test full diffusion process
void test_diffusion_process() {
    printf("Testing full diffusion process...\n");
    
    // Create test state
    quantum_state* state = quantum_state_create(2);
    ComplexFloat amps[4] = {
        {0.5f, 0.0f},
        {0.5f, 0.0f},
        {0.5f, 0.0f},
        {0.5f, 0.0f}
    };
    quantum_state_set_amplitudes(state, amps, 4);
    
    // Create PINN config
    pinn_config config = {
        .hidden_dim = 64,
        .num_layers = 3,
        .activation = ACTIVATION_TANH,
        .learning_rate = 0.001f,
        .tolerance = 1e-6f,
        .params = &(float){0.1f}  // sigma
    };
    
    // Initialize PINN
    assert(pinn_initialize(&config));
    
    // Run diffusion process
    const int num_steps = 100;
    const float dt = 0.01f;
    float prev_residual = INFINITY;
    
    for (int step = 0; step < num_steps; step++) {
        float t = step * dt;
        
        // Estimate drift
        float* drift = estimate_drift(state, t, &config);
        assert(drift != NULL);
        
        // Compute geometric phase
        float phase = compute_geometric_phase(state);
        
        // Update state
        qgt_error_t err = quantum_state_update(state, drift, 0.1f, phase, dt);
        assert(err == QGT_SUCCESS);
        
        // Check residual is decreasing
        float residual = compute_pde_residual(state, t, config.params);
        assert(residual <= prev_residual + 1e-6f);
        prev_residual = residual;
        
        free(drift);
    }
    
    pinn_cleanup(&config);
    quantum_state_destroy(state);
    printf("Diffusion process test passed\n\n");
}

int main() {
    printf("Running quantum diffusion tests...\n\n");
    
    test_pde_residual();
    test_drift_estimation();
    test_state_evolution();
    test_diffusion_process();
    
    printf("All tests passed!\n");
    return 0;
}
