/**
 * @file test_quantum_diffusion.c
 * @brief Test suite for quantum diffusion processes with PINN
 * Based on Shi et al. 2023 (arXiv:2410.15336)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "quantum_geometric/physics/quantum_diffusion.h"
#include "quantum_geometric/core/quantum_complex.h"
#include "test_helpers.h"

// Test PDE residual computation
void test_pde_residual(void) {
    printf("Test 1: PDE residual computation\n");

    // Create test state
    quantum_state_t* state = quantum_diffusion_create_state(2);  // 2 qubits
    TEST_ASSERT(state != NULL, "State creation failed");

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
    TEST_ASSERT(residual >= 0.0f, "Residual should be non-negative");
    TEST_ASSERT(!isnan(residual), "Residual should not be NaN");

    quantum_diffusion_destroy_state(state);
    printf("  PASSED\n\n");
}

// Test PINN initialization and drift estimation
void test_drift_estimation(void) {
    printf("Test 2: PINN drift estimation\n");

    // Create test state
    quantum_state_t* state = quantum_diffusion_create_state(2);
    TEST_ASSERT(state != NULL, "State creation failed");

    ComplexFloat amps[4] = {
        {0.5f, 0.0f},
        {0.5f, 0.0f},
        {0.5f, 0.0f},
        {0.5f, 0.0f}
    };
    quantum_state_set_amplitudes(state, amps, 4);

    // Create PINN config
    pinn_config config = {
        .input_dim = 4,
        .hidden_dim = 64,
        .num_layers = 3,
        .output_dim = 4,
        .activation = PINN_ACTIVATION_TANH,
        .learning_rate = 0.001f,
        .tolerance = 1e-6f,
        .params = NULL,
        .num_params = 0,
        .initialized = false
    };

    // Initialize PINN
    TEST_ASSERT(pinn_initialize(&config), "PINN initialization failed");
    TEST_ASSERT(config.initialized, "PINN should be marked as initialized");

    // Estimate drift
    float* drift = estimate_drift(state, 0.5f, &config);
    TEST_ASSERT(drift != NULL, "Drift estimation returned NULL");

    // Verify drift values
    for (int i = 0; i < 4; i++) {
        TEST_ASSERT(!isnan(drift[i]), "Drift should not be NaN");
        TEST_ASSERT(!isinf(drift[i]), "Drift should not be infinite");
    }

    free(drift);
    pinn_cleanup(&config);
    quantum_diffusion_destroy_state(state);
    printf("  PASSED\n\n");
}

// Test state evolution
void test_state_evolution(void) {
    printf("Test 3: Quantum state evolution\n");

    // Create test state
    quantum_state_t* state = quantum_diffusion_create_state(2);
    TEST_ASSERT(state != NULL, "State creation failed");

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
    TEST_ASSERT(drift != NULL, "Drift allocation failed");
    for (int i = 0; i < 4; i++) drift[i] = 0.1f;

    // Compute geometric phase
    float phase = compute_diffusion_geometric_phase(state);
    TEST_ASSERT(!isnan(phase), "Phase should not be NaN");

    // Evolve state
    qgt_error_t err = quantum_diffusion_state_update(state, drift, sigma, phase, dt);
    TEST_ASSERT(err == QGT_SUCCESS, "State update failed");

    // Verify state is still normalized
    quantum_diffusion_normalize(state);
    float norm = 0.0f;
    for (int i = 0; i < 4; i++) {
        ComplexFloat amp = quantum_state_get_amplitude(state, i);
        norm += amp.real * amp.real + amp.imag * amp.imag;
    }
    TEST_ASSERT(fabsf(norm - 1.0f) < 1e-4f, "State should be normalized");

    free(drift);
    quantum_diffusion_destroy_state(state);
    printf("  PASSED\n\n");
}

// Test full diffusion process
void test_diffusion_process(void) {
    printf("Test 4: Full diffusion process\n");

    // Create test state
    quantum_state_t* state = quantum_diffusion_create_state(2);
    TEST_ASSERT(state != NULL, "State creation failed");

    ComplexFloat amps[4] = {
        {0.5f, 0.0f},
        {0.5f, 0.0f},
        {0.5f, 0.0f},
        {0.5f, 0.0f}
    };
    quantum_state_set_amplitudes(state, amps, 4);

    // Create PINN config
    pinn_config pinn = {
        .input_dim = 4,
        .hidden_dim = 64,
        .num_layers = 3,
        .output_dim = 4,
        .activation = PINN_ACTIVATION_TANH,
        .learning_rate = 0.001f,
        .tolerance = 1e-6f,
        .params = NULL,
        .num_params = 0,
        .initialized = false
    };

    // Initialize PINN
    TEST_ASSERT(pinn_initialize(&pinn), "PINN initialization failed");

    // Create diffusion config
    quantum_diffusion_config_t config = {
        .num_qubits = 2,
        .process_type = DIFFUSION_BROWNIAN,
        .sigma = 0.1f,
        .dt = 0.01f,
        .use_geometric_phase = true,
        .use_error_mitigation = false,
        .pinn = &pinn
    };

    // Run diffusion evolution
    qgt_error_t err = quantum_diffusion_evolve(state, 0.0f, 1.0f, 100, &config);
    TEST_ASSERT(err == QGT_SUCCESS, "Diffusion evolution failed");

    // Verify final state is valid
    quantum_diffusion_normalize(state);
    float norm = 0.0f;
    for (int i = 0; i < 4; i++) {
        ComplexFloat amp = quantum_state_get_amplitude(state, i);
        norm += amp.real * amp.real + amp.imag * amp.imag;
    }
    TEST_ASSERT(fabsf(norm - 1.0f) < 1e-4f, "Final state should be normalized");

    pinn_cleanup(&pinn);
    quantum_diffusion_destroy_state(state);
    printf("  PASSED\n\n");
}

// Test diffusion step
void test_diffusion_step(void) {
    printf("Test 5: Single diffusion step\n");

    // Create test state
    quantum_state_t* state = quantum_diffusion_create_state(2);
    TEST_ASSERT(state != NULL, "State creation failed");

    ComplexFloat amps[4] = {
        {0.5f, 0.0f},
        {0.5f, 0.0f},
        {0.5f, 0.0f},
        {0.5f, 0.0f}
    };
    quantum_state_set_amplitudes(state, amps, 4);

    // Create PINN config
    pinn_config pinn = {
        .input_dim = 4,
        .hidden_dim = 32,
        .num_layers = 2,
        .output_dim = 4,
        .activation = PINN_ACTIVATION_RELU,
        .learning_rate = 0.001f,
        .tolerance = 1e-6f,
        .params = NULL,
        .num_params = 0,
        .initialized = false
    };
    TEST_ASSERT(pinn_initialize(&pinn), "PINN initialization failed");

    // Create diffusion config
    quantum_diffusion_config_t config = {
        .num_qubits = 2,
        .process_type = DIFFUSION_GEOMETRIC,
        .sigma = 0.05f,
        .dt = 0.01f,
        .use_geometric_phase = true,
        .use_error_mitigation = false,
        .pinn = &pinn
    };

    // Take a single step
    qgt_error_t err = quantum_diffusion_step(state, 0.5f, 0.01f, &config);
    TEST_ASSERT(err == QGT_SUCCESS, "Diffusion step failed");

    pinn_cleanup(&pinn);
    quantum_diffusion_destroy_state(state);
    printf("  PASSED\n\n");
}

int main(void) {
    printf("Running quantum diffusion tests...\n\n");

    test_pde_residual();
    test_drift_estimation();
    test_state_evolution();
    test_diffusion_process();
    test_diffusion_step();

    printf("All quantum diffusion tests passed!\n");
    return 0;
}
