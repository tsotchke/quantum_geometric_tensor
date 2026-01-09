/**
 * @file quantum_diffusion.c
 * @brief Implementation of quantum diffusion processes with PINNs
 */

#include "quantum_geometric/physics/quantum_diffusion.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Internal Helper Functions
// ============================================================================

/**
 * @brief Xavier initialization for weights
 */
static float xavier_init(size_t fan_in, size_t fan_out) {
    float scale = sqrtf(2.0f / (float)(fan_in + fan_out));
    float r = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    return r * scale;
}

/**
 * @brief Apply activation function
 */
static float apply_activation(float x, PINNActivationType activation) {
    switch (activation) {
        case PINN_ACTIVATION_TANH:
            return tanhf(x);
        case PINN_ACTIVATION_RELU:
            return fmaxf(0.0f, x);
        case PINN_ACTIVATION_SIGMOID:
            return 1.0f / (1.0f + expf(-x));
        case PINN_ACTIVATION_SWISH:
            return x / (1.0f + expf(-x));
        case PINN_ACTIVATION_GELU:
            return 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / M_PI) *
                   (x + 0.044715f * x * x * x)));
        default:
            return x;
    }
}

/**
 * @brief Compute activation derivative
 */
static float activation_derivative(float x, PINNActivationType activation) {
    float fx;
    switch (activation) {
        case PINN_ACTIVATION_TANH:
            fx = tanhf(x);
            return 1.0f - fx * fx;
        case PINN_ACTIVATION_RELU:
            return x > 0.0f ? 1.0f : 0.0f;
        case PINN_ACTIVATION_SIGMOID:
            fx = 1.0f / (1.0f + expf(-x));
            return fx * (1.0f - fx);
        case PINN_ACTIVATION_SWISH: {
            float sig = 1.0f / (1.0f + expf(-x));
            return sig + x * sig * (1.0f - sig);
        }
        default:
            return 1.0f;
    }
}

/**
 * @brief Generate Gaussian random number
 */
static float gaussian_random(void) {
    static int have_spare = 0;
    static float spare;

    if (have_spare) {
        have_spare = 0;
        return spare;
    }

    float u, v, s;
    do {
        u = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        v = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        s = u * u + v * v;
    } while (s >= 1.0f || s == 0.0f);

    s = sqrtf(-2.0f * logf(s) / s);
    spare = v * s;
    have_spare = 1;

    return u * s;
}

/**
 * @brief Extract state features for PINN input
 */
static float* extract_state_features(quantum_state_t* state, float t, size_t* feature_size) {
    if (!state || !state->coordinates) {
        *feature_size = 0;
        return NULL;
    }

    // Features: real parts, imaginary parts, time, norm
    size_t dim = state->dimension;
    *feature_size = 2 * dim + 2;

    float* features = (float*)calloc(*feature_size, sizeof(float));
    if (!features) return NULL;

    // Real parts
    for (size_t i = 0; i < dim; i++) {
        features[i] = state->coordinates[i].real;
    }

    // Imaginary parts
    for (size_t i = 0; i < dim; i++) {
        features[dim + i] = state->coordinates[i].imag;
    }

    // Time
    features[2 * dim] = t;

    // Norm
    float norm = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        norm += state->coordinates[i].real * state->coordinates[i].real +
                state->coordinates[i].imag * state->coordinates[i].imag;
    }
    features[2 * dim + 1] = sqrtf(norm);

    return features;
}

// ============================================================================
// PINN Functions
// ============================================================================

bool pinn_initialize(pinn_config* config) {
    if (!config) return false;

    // Default dimensions if not set
    if (config->hidden_dim == 0) config->hidden_dim = 64;
    if (config->num_layers == 0) config->num_layers = 3;
    if (config->input_dim == 0) config->input_dim = 8;  // Default for small states
    if (config->output_dim == 0) config->output_dim = config->input_dim;
    if (config->learning_rate == 0.0f) config->learning_rate = 0.001f;
    if (config->tolerance == 0.0f) config->tolerance = 1e-6f;

    // Calculate total parameters
    // Layer 0: input_dim -> hidden_dim (weights + biases)
    // Layer 1 to num_layers-2: hidden_dim -> hidden_dim
    // Layer num_layers-1: hidden_dim -> output_dim
    size_t num_params = 0;

    // Input layer
    num_params += config->input_dim * config->hidden_dim + config->hidden_dim;

    // Hidden layers
    for (size_t i = 1; i < config->num_layers - 1; i++) {
        num_params += config->hidden_dim * config->hidden_dim + config->hidden_dim;
    }

    // Output layer
    if (config->num_layers > 1) {
        num_params += config->hidden_dim * config->output_dim + config->output_dim;
    }

    config->num_params = num_params;
    config->params = (float*)calloc(num_params, sizeof(float));
    if (!config->params) return false;

    // Xavier initialization
    srand((unsigned int)time(NULL));

    size_t idx = 0;

    // Input layer weights
    for (size_t i = 0; i < config->input_dim * config->hidden_dim; i++) {
        config->params[idx++] = xavier_init(config->input_dim, config->hidden_dim);
    }
    // Input layer biases (zero initialized - already done by calloc)
    idx += config->hidden_dim;

    // Hidden layers
    for (size_t layer = 1; layer < config->num_layers - 1; layer++) {
        for (size_t i = 0; i < config->hidden_dim * config->hidden_dim; i++) {
            config->params[idx++] = xavier_init(config->hidden_dim, config->hidden_dim);
        }
        idx += config->hidden_dim;  // biases
    }

    // Output layer
    if (config->num_layers > 1) {
        for (size_t i = 0; i < config->hidden_dim * config->output_dim; i++) {
            config->params[idx++] = xavier_init(config->hidden_dim, config->output_dim);
        }
    }

    config->initialized = true;
    return true;
}

void pinn_cleanup(pinn_config* config) {
    if (!config) return;

    if (config->params) {
        free(config->params);
        config->params = NULL;
    }
    config->num_params = 0;
    config->initialized = false;
}

DiffusionStatus pinn_forward(const float* input, size_t input_size,
                             float* output, size_t output_size,
                             const pinn_config* config) {
    if (!input || !output || !config || !config->params) {
        return DIFFUSION_ERROR_NULL_PTR;
    }

    if (!config->initialized) {
        return DIFFUSION_ERROR_NOT_INITIALIZED;
    }

    // Allocate intermediate buffers
    float* hidden = (float*)calloc(config->hidden_dim, sizeof(float));
    float* hidden_next = (float*)calloc(config->hidden_dim, sizeof(float));
    if (!hidden || !hidden_next) {
        free(hidden);
        free(hidden_next);
        return DIFFUSION_ERROR_MEMORY;
    }

    size_t param_idx = 0;

    // Input layer
    for (size_t j = 0; j < config->hidden_dim; j++) {
        float sum = 0.0f;
        for (size_t i = 0; i < input_size && i < config->input_dim; i++) {
            sum += input[i] * config->params[param_idx + i * config->hidden_dim + j];
        }
        param_idx += config->input_dim * config->hidden_dim;
        sum += config->params[param_idx + j];  // bias
        hidden[j] = apply_activation(sum, config->activation);
    }
    param_idx += config->hidden_dim;

    // Hidden layers
    for (size_t layer = 1; layer < config->num_layers - 1; layer++) {
        for (size_t j = 0; j < config->hidden_dim; j++) {
            float sum = 0.0f;
            for (size_t i = 0; i < config->hidden_dim; i++) {
                sum += hidden[i] * config->params[param_idx + i * config->hidden_dim + j];
            }
            param_idx += config->hidden_dim * config->hidden_dim;
            sum += config->params[param_idx + j];  // bias
            hidden_next[j] = apply_activation(sum, config->activation);
        }
        param_idx += config->hidden_dim;

        // Swap buffers
        float* tmp = hidden;
        hidden = hidden_next;
        hidden_next = tmp;
    }

    // Output layer (no activation for regression output)
    if (config->num_layers > 1) {
        for (size_t j = 0; j < output_size && j < config->output_dim; j++) {
            float sum = 0.0f;
            for (size_t i = 0; i < config->hidden_dim; i++) {
                sum += hidden[i] * config->params[param_idx + i * config->output_dim + j];
            }
            param_idx += config->hidden_dim * config->output_dim;
            sum += config->params[param_idx + j];  // bias
            output[j] = sum;  // Linear output
        }
    } else {
        // Single layer: copy hidden to output
        for (size_t j = 0; j < output_size && j < config->hidden_dim; j++) {
            output[j] = hidden[j];
        }
    }

    free(hidden);
    free(hidden_next);
    return DIFFUSION_SUCCESS;
}

float* estimate_drift(quantum_state_t* state, float t, pinn_config* config) {
    if (!state || !config || !config->initialized) {
        return NULL;
    }

    // Extract features from state
    size_t feature_size;
    float* features = extract_state_features(state, t, &feature_size);
    if (!features) return NULL;

    // Allocate output
    float* drift = (float*)calloc(state->dimension * 2, sizeof(float));  // real + imag
    if (!drift) {
        free(features);
        return NULL;
    }

    // Forward pass
    DiffusionStatus status = pinn_forward(features, feature_size,
                                          drift, state->dimension * 2,
                                          config);

    free(features);

    if (status != DIFFUSION_SUCCESS) {
        free(drift);
        return NULL;
    }

    return drift;
}

DiffusionStatus pinn_train(pinn_config* config,
                           quantum_state_t** states,
                           const float* times,
                           size_t num_samples,
                           size_t num_epochs) {
    if (!config || !states || !times) {
        return DIFFUSION_ERROR_NULL_PTR;
    }

    if (!config->initialized) {
        return DIFFUSION_ERROR_NOT_INITIALIZED;
    }

    // Simple gradient descent training
    float* gradients = (float*)calloc(config->num_params, sizeof(float));
    if (!gradients) return DIFFUSION_ERROR_MEMORY;

    float epsilon = 1e-5f;

    for (size_t epoch = 0; epoch < num_epochs; epoch++) {
        memset(gradients, 0, config->num_params * sizeof(float));

        float total_loss = 0.0f;

        // Compute loss and gradients using numerical differentiation
        for (size_t s = 0; s < num_samples; s++) {
            if (!states[s]) continue;

            // Compute PDE residual as loss
            float residual = compute_pde_residual(states[s], times[s], NULL);
            total_loss += residual * residual;

            // Numerical gradient estimation (finite differences)
            for (size_t p = 0; p < config->num_params; p++) {
                float original = config->params[p];

                config->params[p] = original + epsilon;
                float loss_plus = compute_pde_residual(states[s], times[s], NULL);

                config->params[p] = original - epsilon;
                float loss_minus = compute_pde_residual(states[s], times[s], NULL);

                config->params[p] = original;

                gradients[p] += (loss_plus - loss_minus) / (2.0f * epsilon);
            }
        }

        // Apply gradients
        for (size_t p = 0; p < config->num_params; p++) {
            config->params[p] -= config->learning_rate * gradients[p] / (float)num_samples;
        }

        // Check convergence
        if (total_loss / (float)num_samples < config->tolerance) {
            break;
        }
    }

    free(gradients);
    return DIFFUSION_SUCCESS;
}

// ============================================================================
// PDE Residual Functions
// ============================================================================

float compute_pde_residual(quantum_state_t* state, float t, float* sigma) {
    if (!state || !state->coordinates) {
        return 0.0f;
    }

    float default_sigma = 0.1f;
    float sig = (sigma != NULL) ? sigma[0] : default_sigma;

    // Compute Lindblad term
    float lindblad = compute_lindblad_term(state);

    // Compute diffusion term
    float diffusion = compute_diffusion_term(state, sig);

    // PDE residual: should be zero for exact solution
    // dρ/dt - L[ρ] - D[ρ] = 0
    // We approximate dρ/dt as zero for stationary solutions
    float residual = lindblad + diffusion;

    return fabsf(residual);
}

float compute_lindblad_term(quantum_state_t* state) {
    if (!state || !state->coordinates) {
        return 0.0f;
    }

    float term = 0.0f;
    size_t dim = state->dimension;

    // Simplified Lindblad dissipator: L[ρ] = Σ_k (L_k ρ L_k† - 0.5{L_k†L_k, ρ})
    // For dephasing noise with diagonal Lindblad operators
    for (size_t i = 0; i < dim; i++) {
        float amp_sq = state->coordinates[i].real * state->coordinates[i].real +
                       state->coordinates[i].imag * state->coordinates[i].imag;

        // Dephasing contribution
        term += 0.5f * amp_sq * (1.0f - amp_sq);
    }

    return term;
}

float compute_diffusion_term(quantum_state_t* state, float sigma) {
    if (!state || !state->coordinates) {
        return 0.0f;
    }

    float term = 0.0f;
    size_t dim = state->dimension;

    // Diffusion term: D[ρ] = σ² ∇²ρ
    // Approximate using finite differences in state space
    for (size_t i = 0; i < dim; i++) {
        float amp_i = state->coordinates[i].real * state->coordinates[i].real +
                      state->coordinates[i].imag * state->coordinates[i].imag;

        // Neighbors in computational basis (simplified)
        size_t j_plus = (i + 1) % dim;
        size_t j_minus = (i + dim - 1) % dim;

        float amp_plus = state->coordinates[j_plus].real * state->coordinates[j_plus].real +
                         state->coordinates[j_plus].imag * state->coordinates[j_plus].imag;
        float amp_minus = state->coordinates[j_minus].real * state->coordinates[j_minus].real +
                          state->coordinates[j_minus].imag * state->coordinates[j_minus].imag;

        // Second derivative approximation
        float laplacian = amp_plus + amp_minus - 2.0f * amp_i;
        term += sigma * sigma * laplacian;
    }

    return term / (float)dim;
}

// ============================================================================
// Geometric Phase Functions
// ============================================================================

float compute_diffusion_geometric_phase(quantum_state_t* state) {
    if (!state || !state->coordinates) {
        return 0.0f;
    }

    float phase = 0.0f;
    size_t dim = state->dimension;

    // Berry phase: γ = -Im[∫⟨ψ|∇ψ⟩·dR]
    // For discrete state, approximate as sum over phase contributions
    for (size_t i = 0; i < dim; i++) {
        float re = state->coordinates[i].real;
        float im = state->coordinates[i].imag;

        // Local phase contribution
        float amp_sq = re * re + im * im;
        if (amp_sq > 1e-10f) {
            float local_phase = atan2f(im, re);
            phase += amp_sq * local_phase;
        }
    }

    // Normalize by total probability
    float norm = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        norm += state->coordinates[i].real * state->coordinates[i].real +
                state->coordinates[i].imag * state->coordinates[i].imag;
    }

    if (norm > 1e-10f) {
        phase /= norm;
    }

    return phase;
}

float compute_phase_rate(quantum_state_t* state, const float* drift) {
    if (!state || !state->coordinates || !drift) {
        return 0.0f;
    }

    float rate = 0.0f;
    size_t dim = state->dimension;

    // Phase rate: dγ/dt = Im[⟨ψ|d/dt|ψ⟩]
    for (size_t i = 0; i < dim; i++) {
        float re = state->coordinates[i].real;
        float im = state->coordinates[i].imag;

        // Drift contribution to phase change
        float drift_re = drift[i];
        float drift_im = drift[dim + i];

        // Im[⟨ψ|dψ⟩] = Re(ψ) * Im(dψ) - Im(ψ) * Re(dψ)
        rate += re * drift_im - im * drift_re;
    }

    return rate;
}

// ============================================================================
// State Evolution Functions
// ============================================================================

qgt_error_t quantum_diffusion_state_update(quantum_state_t* state,
                                           float* drift,
                                           float sigma,
                                           float phase,
                                           float dt) {
    if (!state || !state->coordinates) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    size_t dim = state->dimension;

    // d|ψ⟩ = drift*dt + sigma*dW + i*phase*|ψ⟩*dt
    // where dW ~ N(0, dt) is Wiener increment

    float sqrt_dt = sqrtf(dt);

    for (size_t i = 0; i < dim; i++) {
        // Drift contribution
        float drift_re = (drift != NULL) ? drift[i] * dt : 0.0f;
        float drift_im = (drift != NULL) ? drift[dim + i] * dt : 0.0f;

        // Diffusion (noise) contribution
        float noise_re = sigma * gaussian_random() * sqrt_dt;
        float noise_im = sigma * gaussian_random() * sqrt_dt;

        // Phase contribution: exp(i*phase*dt) ≈ 1 + i*phase*dt
        float phase_factor_re = 1.0f;
        float phase_factor_im = phase * dt;

        // Current amplitude
        float re = state->coordinates[i].real;
        float im = state->coordinates[i].imag;

        // Apply phase rotation
        float new_re = phase_factor_re * re - phase_factor_im * im;
        float new_im = phase_factor_im * re + phase_factor_re * im;

        // Add drift and noise
        state->coordinates[i].real = new_re + drift_re + noise_re;
        state->coordinates[i].imag = new_im + drift_im + noise_im;
    }

    // Renormalize
    quantum_diffusion_normalize(state);

    return QGT_SUCCESS;
}

qgt_error_t quantum_diffusion_step(quantum_state_t* state,
                                   float t,
                                   float dt,
                                   const quantum_diffusion_config_t* config) {
    if (!state || !config) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // Estimate drift using PINN if available
    float* drift = NULL;
    if (config->pinn && config->pinn->initialized) {
        drift = estimate_drift(state, t, config->pinn);
    }

    // Compute geometric phase if enabled
    float phase = 0.0f;
    if (config->use_geometric_phase) {
        phase = compute_diffusion_geometric_phase(state);
    }

    // Update state
    qgt_error_t err = quantum_diffusion_state_update(state, drift, config->sigma, phase, dt);

    if (drift) {
        free(drift);
    }

    return err;
}

qgt_error_t quantum_diffusion_evolve(quantum_state_t* state,
                                     float t_start,
                                     float t_end,
                                     size_t num_steps,
                                     const quantum_diffusion_config_t* config) {
    if (!state || !config || num_steps == 0) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    float dt = (t_end - t_start) / (float)num_steps;
    float t = t_start;

    for (size_t step = 0; step < num_steps; step++) {
        qgt_error_t err = quantum_diffusion_step(state, t, dt, config);
        if (err != QGT_SUCCESS) {
            return err;
        }
        t += dt;
    }

    return QGT_SUCCESS;
}

// ============================================================================
// State Management Functions
// ============================================================================

quantum_state_t* quantum_diffusion_create_state(size_t num_qubits) {
    quantum_state_t* state = (quantum_state_t*)calloc(1, sizeof(quantum_state_t));
    if (!state) return NULL;

    state->num_qubits = num_qubits;
    state->dimension = 1ULL << num_qubits;

    state->coordinates = (ComplexFloat*)calloc(state->dimension, sizeof(ComplexFloat));
    if (!state->coordinates) {
        free(state);
        return NULL;
    }

    // Initialize to |0...0⟩
    state->coordinates[0].real = 1.0f;
    state->coordinates[0].imag = 0.0f;

    return state;
}

void quantum_diffusion_destroy_state(quantum_state_t* state) {
    if (!state) return;

    if (state->coordinates) {
        free(state->coordinates);
    }
    free(state);
}

void quantum_state_set_amplitudes(quantum_state_t* state,
                                  ComplexFloat* amps,
                                  size_t num_amps) {
    if (!state || !amps) return;

    size_t copy_size = (num_amps < state->dimension) ? num_amps : state->dimension;

    for (size_t i = 0; i < copy_size; i++) {
        state->coordinates[i] = amps[i];
    }

    // Zero out remaining amplitudes
    for (size_t i = copy_size; i < state->dimension; i++) {
        state->coordinates[i].real = 0.0f;
        state->coordinates[i].imag = 0.0f;
    }
}

ComplexFloat quantum_state_get_amplitude(quantum_state_t* state, size_t index) {
    ComplexFloat zero = {0.0f, 0.0f};

    if (!state || !state->coordinates || index >= state->dimension) {
        return zero;
    }

    return state->coordinates[index];
}

void quantum_diffusion_normalize(quantum_state_t* state) {
    if (!state || !state->coordinates) return;

    // Compute norm
    float norm_sq = 0.0f;
    for (size_t i = 0; i < state->dimension; i++) {
        norm_sq += state->coordinates[i].real * state->coordinates[i].real +
                   state->coordinates[i].imag * state->coordinates[i].imag;
    }

    if (norm_sq < 1e-20f) return;  // Avoid division by zero

    float norm = sqrtf(norm_sq);

    for (size_t i = 0; i < state->dimension; i++) {
        state->coordinates[i].real /= norm;
        state->coordinates[i].imag /= norm;
    }
}

// ============================================================================
// Diffusion State Tracking
// ============================================================================

diffusion_state_t* diffusion_state_create(quantum_state_t* state,
                                          size_t max_history) {
    diffusion_state_t* dstate = (diffusion_state_t*)calloc(1, sizeof(diffusion_state_t));
    if (!dstate) return NULL;

    dstate->state = state;
    dstate->time = 0.0f;
    dstate->max_history = max_history;
    dstate->history_length = 0;

    if (max_history > 0) {
        dstate->drift_history = (float*)calloc(max_history, sizeof(float));
        dstate->phase_history = (float*)calloc(max_history, sizeof(float));

        if (!dstate->drift_history || !dstate->phase_history) {
            free(dstate->drift_history);
            free(dstate->phase_history);
            free(dstate);
            return NULL;
        }
    }

    return dstate;
}

void diffusion_state_destroy(diffusion_state_t* dstate) {
    if (!dstate) return;

    free(dstate->drift_history);
    free(dstate->phase_history);
    // Note: does not free dstate->state (not owned)
    free(dstate);
}

void diffusion_state_record(diffusion_state_t* dstate,
                            float drift,
                            float phase) {
    if (!dstate) return;

    if (dstate->history_length < dstate->max_history) {
        if (dstate->drift_history) {
            dstate->drift_history[dstate->history_length] = drift;
        }
        if (dstate->phase_history) {
            dstate->phase_history[dstate->history_length] = phase;
        }
        dstate->history_length++;
    } else if (dstate->max_history > 0) {
        // Circular buffer: shift and add
        for (size_t i = 0; i < dstate->max_history - 1; i++) {
            if (dstate->drift_history) {
                dstate->drift_history[i] = dstate->drift_history[i + 1];
            }
            if (dstate->phase_history) {
                dstate->phase_history[i] = dstate->phase_history[i + 1];
            }
        }
        if (dstate->drift_history) {
            dstate->drift_history[dstate->max_history - 1] = drift;
        }
        if (dstate->phase_history) {
            dstate->phase_history[dstate->max_history - 1] = phase;
        }
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

pinn_config pinn_create_default_config(size_t state_dim) {
    pinn_config config = {0};

    config.input_dim = 2 * state_dim + 2;  // real + imag + time + norm
    config.hidden_dim = 64;
    config.num_layers = 3;
    config.output_dim = 2 * state_dim;  // drift for real + imag
    config.activation = PINN_ACTIVATION_TANH;
    config.learning_rate = 0.001f;
    config.tolerance = 1e-6f;
    config.params = NULL;
    config.num_params = 0;
    config.initialized = false;

    return config;
}

quantum_diffusion_config_t quantum_diffusion_create_default_config(size_t num_qubits) {
    quantum_diffusion_config_t config = {0};

    config.num_qubits = num_qubits;
    config.process_type = DIFFUSION_BROWNIAN;
    config.sigma = 0.1f;
    config.dt = 0.01f;
    config.use_geometric_phase = true;
    config.use_error_mitigation = false;
    config.pinn = NULL;

    return config;
}

bool quantum_diffusion_validate_config(const quantum_diffusion_config_t* config) {
    if (!config) return false;

    if (config->num_qubits == 0 || config->num_qubits > 30) {
        return false;
    }

    if (config->sigma < 0.0f) {
        return false;
    }

    if (config->dt <= 0.0f) {
        return false;
    }

    return true;
}
