/**
 * @file quantum_autoencoder.c
 * @brief Implementation of quantum autoencoder algorithms
 */

#include "quantum_geometric/learning/quantum_autoencoder.h"
#include "quantum_geometric/core/quantum_state.h"
#include "quantum_geometric/core/quantum_complex.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdio.h>

// =============================================================================
// Internal Helper Functions
// =============================================================================

/**
 * @brief Apply variational layer to quantum state
 */
static void apply_variational_layer(quantum_state_t* state,
                                   const double* params,
                                   size_t num_params,
                                   size_t layer_idx) {
    if (!state || !params || !state->coordinates) return;

    size_t dim = state->dimension;
    size_t param_idx = layer_idx * 3;  // 3 params per layer

    // Apply rotation gates (simplified simulation)
    for (size_t i = 0; i < dim; i++) {
        if (param_idx >= num_params) break;

        double theta = params[param_idx % num_params];
        double phi = params[(param_idx + 1) % num_params];

        float cos_t = (float)cos(theta / 2.0);
        float sin_t = (float)sin(theta / 2.0);
        float cos_p = (float)cos(phi);
        float sin_p = (float)sin(phi);

        // Apply Rx(theta) * Rz(phi) rotation
        float re = state->coordinates[i].real;
        float im = state->coordinates[i].imag;

        state->coordinates[i].real = cos_t * re - sin_t * sin_p * im;
        state->coordinates[i].imag = cos_t * im + sin_t * cos_p * re;
    }

    // Normalize
    double norm = 0.0;
    for (size_t i = 0; i < dim; i++) {
        norm += state->coordinates[i].real * state->coordinates[i].real +
                state->coordinates[i].imag * state->coordinates[i].imag;
    }
    norm = sqrt(norm);
    if (norm > 1e-10) {
        for (size_t i = 0; i < dim; i++) {
            state->coordinates[i].real /= (float)norm;
            state->coordinates[i].imag /= (float)norm;
        }
    }
}

/**
 * @brief Calculate reconstruction loss
 */
static double calculate_reconstruction_loss(quantum_state_t* original,
                                           quantum_state_t* reconstructed) {
    if (!original || !reconstructed ||
        original->dimension != reconstructed->dimension) {
        return 1.0;
    }

    // Calculate fidelity F = |<original|reconstructed>|^2
    double fidelity_re = 0.0, fidelity_im = 0.0;
    for (size_t i = 0; i < original->dimension; i++) {
        fidelity_re += original->coordinates[i].real * reconstructed->coordinates[i].real +
                       original->coordinates[i].imag * reconstructed->coordinates[i].imag;
        fidelity_im += original->coordinates[i].real * reconstructed->coordinates[i].imag -
                       original->coordinates[i].imag * reconstructed->coordinates[i].real;
    }

    double fidelity = fidelity_re * fidelity_re + fidelity_im * fidelity_im;

    // Loss = 1 - fidelity
    return 1.0 - fidelity;
}

/**
 * @brief Initialize random parameters
 */
static void init_random_params(double* params, size_t num_params) {
    for (size_t i = 0; i < num_params; i++) {
        params[i] = 2.0 * M_PI * rand() / (double)RAND_MAX - M_PI;
    }
}

// =============================================================================
// Core API Implementation
// =============================================================================

quantum_autoencoder_t* quantum_autoencoder_create(const quantum_autoencoder_config_t* config) {
    if (!config || config->input_dim == 0 || config->latent_dim == 0) {
        return NULL;
    }

    quantum_autoencoder_t* model = calloc(1, sizeof(quantum_autoencoder_t));
    if (!model) return NULL;

    model->input_dim = config->input_dim;
    model->latent_dim = config->latent_dim;
    model->quantum_depth = config->quantum_depth;
    model->config = *config;
    model->is_trained = false;

    // Calculate number of parameters
    // Encoder: 3 params per qubit per layer * input_dim * depth
    model->num_encoder_params = 3 * config->input_dim * config->quantum_depth;
    model->num_decoder_params = 3 * config->input_dim * config->quantum_depth;

    // Allocate parameters
    model->encoder_params = calloc(model->num_encoder_params, sizeof(double));
    model->decoder_params = calloc(model->num_decoder_params, sizeof(double));

    if (!model->encoder_params || !model->decoder_params) {
        free(model->encoder_params);
        free(model->decoder_params);
        free(model);
        return NULL;
    }

    // Initialize parameters randomly
    static bool seeded = false;
    if (!seeded) {
        srand((unsigned int)time(NULL));
        seeded = true;
    }

    init_random_params(model->encoder_params, model->num_encoder_params);
    init_random_params(model->decoder_params, model->num_decoder_params);

    return model;
}

void quantum_autoencoder_destroy(quantum_autoencoder_t* model) {
    if (!model) return;

    free(model->encoder_params);
    free(model->decoder_params);
    free(model->internal_data);
    free(model);
}

// =============================================================================
// Encoding/Decoding Implementation
// =============================================================================

quantum_state_t* quantum_encode_state(quantum_autoencoder_t* model,
                                      quantum_state_t* state) {
    if (!model || !state) return NULL;

    // Create latent state
    size_t latent_dim_size = 1ULL << model->latent_dim;
    quantum_state_t* encoded = NULL;

    if (quantum_state_create(&encoded, QUANTUM_STATE_PURE, latent_dim_size) != QGT_SUCCESS) {
        return NULL;
    }

    // Copy and truncate/project state to latent dimension
    size_t min_dim = (state->dimension < latent_dim_size) ? state->dimension : latent_dim_size;
    for (size_t i = 0; i < min_dim; i++) {
        encoded->coordinates[i] = state->coordinates[i];
    }

    // Apply encoder variational layers
    for (size_t layer = 0; layer < model->quantum_depth; layer++) {
        apply_variational_layer(encoded, model->encoder_params,
                               model->num_encoder_params, layer);
    }

    // Normalize
    double norm = 0.0;
    for (size_t i = 0; i < latent_dim_size; i++) {
        norm += encoded->coordinates[i].real * encoded->coordinates[i].real +
                encoded->coordinates[i].imag * encoded->coordinates[i].imag;
    }
    norm = sqrt(norm);
    if (norm > 1e-10) {
        for (size_t i = 0; i < latent_dim_size; i++) {
            encoded->coordinates[i].real /= (float)norm;
            encoded->coordinates[i].imag /= (float)norm;
        }
    } else {
        encoded->coordinates[0].real = 1.0f;
    }

    encoded->num_qubits = model->latent_dim;
    encoded->is_normalized = true;

    return encoded;
}

quantum_state_t* quantum_decode_state(quantum_autoencoder_t* model,
                                      quantum_state_t* state) {
    if (!model || !state) return NULL;

    // Create output state at input dimension
    size_t input_dim_size = 1ULL << model->input_dim;
    quantum_state_t* decoded = NULL;

    if (quantum_state_create(&decoded, QUANTUM_STATE_PURE, input_dim_size) != QGT_SUCCESS) {
        return NULL;
    }

    // Copy and expand state from latent dimension
    size_t min_dim = (state->dimension < input_dim_size) ? state->dimension : input_dim_size;
    for (size_t i = 0; i < min_dim; i++) {
        decoded->coordinates[i] = state->coordinates[i];
    }

    // Apply decoder variational layers
    for (size_t layer = 0; layer < model->quantum_depth; layer++) {
        apply_variational_layer(decoded, model->decoder_params,
                               model->num_decoder_params, layer);
    }

    // Normalize
    double norm = 0.0;
    for (size_t i = 0; i < input_dim_size; i++) {
        norm += decoded->coordinates[i].real * decoded->coordinates[i].real +
                decoded->coordinates[i].imag * decoded->coordinates[i].imag;
    }
    norm = sqrt(norm);
    if (norm > 1e-10) {
        for (size_t i = 0; i < input_dim_size; i++) {
            decoded->coordinates[i].real /= (float)norm;
            decoded->coordinates[i].imag /= (float)norm;
        }
    } else {
        decoded->coordinates[0].real = 1.0f;
    }

    decoded->num_qubits = model->input_dim;
    decoded->is_normalized = true;

    return decoded;
}

// =============================================================================
// Training Implementation
// =============================================================================

training_result_t quantum_train_autoencoder_distributed(
    quantum_autoencoder_t* model,
    quantum_dataset_t* data,
    struct distributed_manager_t* manager,
    const training_config_t* config,
    void* context) {

    training_result_t result = {0};
    result.status = TRAINING_ERROR;

    if (!model || !data || !config || data->num_samples == 0) {
        return result;
    }

    // Allocate loss history
    result.loss_history = calloc(config->num_epochs, sizeof(double));
    if (!result.loss_history) {
        return result;
    }
    result.history_length = 0;

    double learning_rate = config->learning_rate;
    double best_loss = INFINITY;
    size_t patience_counter = 0;

    // Training loop
    for (size_t epoch = 0; epoch < config->num_epochs; epoch++) {
        double epoch_loss = 0.0;

        // Process each sample
        for (size_t i = 0; i < data->num_samples; i++) {
            quantum_state_t* input = data->states[i];
            if (!input) continue;

            // Forward pass: encode then decode
            quantum_state_t* encoded = quantum_encode_state(model, input);
            if (!encoded) continue;

            quantum_state_t* reconstructed = quantum_decode_state(model, encoded);
            if (!reconstructed) {
                quantum_state_destroy(encoded);
                continue;
            }

            // Calculate loss
            double sample_loss = calculate_reconstruction_loss(input, reconstructed);
            epoch_loss += sample_loss;

            // Simplified gradient update (parameter shift rule approximation)
            double epsilon = 0.01;
            for (size_t p = 0; p < model->num_encoder_params; p++) {
                // Numerical gradient approximation
                model->encoder_params[p] += epsilon;
                quantum_state_t* enc_plus = quantum_encode_state(model, input);
                quantum_state_t* dec_plus = enc_plus ? quantum_decode_state(model, enc_plus) : NULL;
                double loss_plus = dec_plus ? calculate_reconstruction_loss(input, dec_plus) : sample_loss;

                model->encoder_params[p] -= 2 * epsilon;
                quantum_state_t* enc_minus = quantum_encode_state(model, input);
                quantum_state_t* dec_minus = enc_minus ? quantum_decode_state(model, enc_minus) : NULL;
                double loss_minus = dec_minus ? calculate_reconstruction_loss(input, dec_minus) : sample_loss;

                model->encoder_params[p] += epsilon;  // Restore

                // Gradient descent step
                double gradient = (loss_plus - loss_minus) / (2 * epsilon);
                model->encoder_params[p] -= learning_rate * gradient;

                if (enc_plus) quantum_state_destroy(enc_plus);
                if (dec_plus) quantum_state_destroy(dec_plus);
                if (enc_minus) quantum_state_destroy(enc_minus);
                if (dec_minus) quantum_state_destroy(dec_minus);
            }

            quantum_state_destroy(encoded);
            quantum_state_destroy(reconstructed);
        }

        epoch_loss /= data->num_samples;
        result.loss_history[epoch] = epoch_loss;
        result.history_length = epoch + 1;
        result.epochs_completed = epoch + 1;

        // Early stopping check
        if (config->optimization.early_stopping.enabled) {
            if (epoch_loss < best_loss - config->optimization.early_stopping.min_delta) {
                best_loss = epoch_loss;
                patience_counter = 0;
            } else {
                patience_counter++;
                if (patience_counter >= config->optimization.early_stopping.patience) {
                    result.status = TRAINING_EARLY_STOPPED;
                    result.final_loss = epoch_loss;
                    result.best_loss = best_loss;
                    model->is_trained = true;
                    return result;
                }
            }
        }

        if (epoch_loss < best_loss) {
            best_loss = epoch_loss;
        }
    }

    result.status = TRAINING_SUCCESS;
    result.final_loss = result.loss_history[result.history_length - 1];
    result.best_loss = best_loss;
    model->is_trained = true;

    return result;
}

autoencoder_evaluation_result_t quantum_evaluate_autoencoder(
    quantum_autoencoder_t* model,
    quantum_dataset_t* data) {

    autoencoder_evaluation_result_t result = {0};

    if (!model || !data || data->num_samples == 0) {
        return result;
    }

    double total_error = 0.0;
    double total_fidelity = 0.0;
    size_t valid_samples = 0;

    for (size_t i = 0; i < data->num_samples; i++) {
        quantum_state_t* input = data->states[i];
        if (!input) continue;

        quantum_state_t* encoded = quantum_encode_state(model, input);
        if (!encoded) continue;

        quantum_state_t* reconstructed = quantum_decode_state(model, encoded);
        if (!reconstructed) {
            quantum_state_destroy(encoded);
            continue;
        }

        double loss = calculate_reconstruction_loss(input, reconstructed);
        double fidelity = 1.0 - loss;

        total_error += loss;
        total_fidelity += fidelity;
        valid_samples++;

        quantum_state_destroy(encoded);
        quantum_state_destroy(reconstructed);
    }

    if (valid_samples > 0) {
        result.reconstruction_error = total_error / valid_samples;
        result.avg_state_fidelity = total_fidelity / valid_samples;
    }

    // Compression ratio
    result.compression_ratio = (double)model->input_dim / model->latent_dim;

    // Latent entropy (estimated from latent representations)
    result.latent_entropy = log2((double)model->latent_dim);  // Maximum entropy estimate

    return result;
}

// =============================================================================
// Data Generation Implementation
// =============================================================================

quantum_dataset_t* quantum_generate_synthetic_states(size_t num_samples,
                                                     size_t dim,
                                                     StateType type) {
    if (num_samples == 0 || dim == 0) return NULL;

    quantum_dataset_t* data = calloc(1, sizeof(quantum_dataset_t));
    if (!data) return NULL;

    data->num_samples = num_samples;
    data->state_dim = dim;

    data->states = calloc(num_samples, sizeof(quantum_state_t*));
    if (!data->states) {
        free(data);
        return NULL;
    }

    size_t dim_size = 1ULL << dim;

    // Seed random generator
    static bool seeded = false;
    if (!seeded) {
        srand((unsigned int)time(NULL));
        seeded = true;
    }

    for (size_t i = 0; i < num_samples; i++) {
        if (quantum_state_create(&data->states[i], QUANTUM_STATE_PURE, dim_size) != QGT_SUCCESS) {
            // Cleanup on failure
            for (size_t j = 0; j < i; j++) {
                quantum_state_destroy(data->states[j]);
            }
            free(data->states);
            free(data);
            return NULL;
        }

        data->states[i]->num_qubits = dim;

        // Generate random state based on type
        switch (type) {
            case STATE_TYPE_PURE:
                // Random pure state
                {
                    double norm = 0.0;
                    for (size_t j = 0; j < dim_size; j++) {
                        data->states[i]->coordinates[j].real = (float)(rand() / (double)RAND_MAX - 0.5);
                        data->states[i]->coordinates[j].imag = (float)(rand() / (double)RAND_MAX - 0.5);
                        norm += data->states[i]->coordinates[j].real * data->states[i]->coordinates[j].real +
                                data->states[i]->coordinates[j].imag * data->states[i]->coordinates[j].imag;
                    }
                    norm = sqrt(norm);
                    for (size_t j = 0; j < dim_size; j++) {
                        data->states[i]->coordinates[j].real /= (float)norm;
                        data->states[i]->coordinates[j].imag /= (float)norm;
                    }
                }
                break;

            case STATE_TYPE_MIXED:
                // Mixed state (classical mixture of basis states)
                {
                    // Start with weighted combination
                    double norm = 0.0;
                    for (size_t j = 0; j < dim_size; j++) {
                        double weight = exp(-0.5 * (double)j);  // Exponentially decaying
                        data->states[i]->coordinates[j].real = (float)(weight * (rand() / (double)RAND_MAX - 0.5));
                        data->states[i]->coordinates[j].imag = (float)(weight * (rand() / (double)RAND_MAX - 0.5));
                        norm += data->states[i]->coordinates[j].real * data->states[i]->coordinates[j].real +
                                data->states[i]->coordinates[j].imag * data->states[i]->coordinates[j].imag;
                    }
                    norm = sqrt(norm);
                    for (size_t j = 0; j < dim_size; j++) {
                        data->states[i]->coordinates[j].real /= (float)norm;
                        data->states[i]->coordinates[j].imag /= (float)norm;
                    }
                }
                break;

            case STATE_TYPE_ENTANGLED:
                // Create entangled-like superpositions
                {
                    // Zero everything first
                    for (size_t j = 0; j < dim_size; j++) {
                        data->states[i]->coordinates[j].real = 0.0f;
                        data->states[i]->coordinates[j].imag = 0.0f;
                    }
                    // Create Bell-like state with random phases
                    if (dim_size >= 4) {
                        double phase = 2.0 * M_PI * rand() / (double)RAND_MAX;
                        data->states[i]->coordinates[0].real = (float)(M_SQRT1_2);
                        data->states[i]->coordinates[dim_size - 1].real = (float)(M_SQRT1_2 * cos(phase));
                        data->states[i]->coordinates[dim_size - 1].imag = (float)(M_SQRT1_2 * sin(phase));
                    } else {
                        data->states[i]->coordinates[0].real = 1.0f;
                    }
                }
                break;
        }

        data->states[i]->is_normalized = true;
    }

    return data;
}

// =============================================================================
// State Utility Implementation
// =============================================================================

quantum_state_t* quantum_create_bell_state(void) {
    quantum_state_t* state = NULL;

    // Bell state is 2 qubits = 4 dimensions
    if (quantum_state_create(&state, QUANTUM_STATE_PURE, 4) != QGT_SUCCESS) {
        return NULL;
    }

    // |Bell> = (|00> + |11>) / sqrt(2)
    state->coordinates[0].real = (float)M_SQRT1_2;  // |00>
    state->coordinates[0].imag = 0.0f;
    state->coordinates[1].real = 0.0f;              // |01>
    state->coordinates[1].imag = 0.0f;
    state->coordinates[2].real = 0.0f;              // |10>
    state->coordinates[2].imag = 0.0f;
    state->coordinates[3].real = (float)M_SQRT1_2;  // |11>
    state->coordinates[3].imag = 0.0f;

    state->num_qubits = 2;
    state->is_normalized = true;

    return state;
}

float quantum_autoencoder_state_fidelity(quantum_state_t* a, quantum_state_t* b) {
    if (!a || !b || a->dimension != b->dimension) return 0.0f;

    double fidelity_re = 0.0, fidelity_im = 0.0;
    for (size_t i = 0; i < a->dimension; i++) {
        fidelity_re += a->coordinates[i].real * b->coordinates[i].real +
                       a->coordinates[i].imag * b->coordinates[i].imag;
        fidelity_im += a->coordinates[i].real * b->coordinates[i].imag -
                       a->coordinates[i].imag * b->coordinates[i].real;
    }

    return (float)(fidelity_re * fidelity_re + fidelity_im * fidelity_im);
}

void quantum_destroy_state(quantum_state_t* state) {
    if (!state) return;
    if (state->coordinates) free(state->coordinates);
    if (state->metric) free(state->metric);
    if (state->connection) free(state->connection);
    if (state->stabilizers) free(state->stabilizers);
    if (state->anyons) free(state->anyons);
    if (state->syndrome_values) free(state->syndrome_values);
    free(state);
}

// =============================================================================
// Model Persistence Implementation
// =============================================================================

int quantum_save_autoencoder_model(quantum_autoencoder_t* model, const char* path) {
    if (!model || !path) return -1;

    FILE* f = fopen(path, "wb");
    if (!f) return -1;

    // Write configuration
    fwrite(&model->input_dim, sizeof(size_t), 1, f);
    fwrite(&model->latent_dim, sizeof(size_t), 1, f);
    fwrite(&model->quantum_depth, sizeof(size_t), 1, f);
    fwrite(&model->config, sizeof(quantum_autoencoder_config_t), 1, f);

    // Write parameters
    fwrite(&model->num_encoder_params, sizeof(size_t), 1, f);
    fwrite(model->encoder_params, sizeof(double), model->num_encoder_params, f);
    fwrite(&model->num_decoder_params, sizeof(size_t), 1, f);
    fwrite(model->decoder_params, sizeof(double), model->num_decoder_params, f);

    fclose(f);
    return 0;
}

quantum_autoencoder_t* quantum_load_autoencoder_model(const char* path) {
    if (!path) return NULL;

    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    size_t input_dim, latent_dim, quantum_depth;
    quantum_autoencoder_config_t config;

    fread(&input_dim, sizeof(size_t), 1, f);
    fread(&latent_dim, sizeof(size_t), 1, f);
    fread(&quantum_depth, sizeof(size_t), 1, f);
    fread(&config, sizeof(quantum_autoencoder_config_t), 1, f);

    quantum_autoencoder_t* model = quantum_autoencoder_create(&config);
    if (!model) {
        fclose(f);
        return NULL;
    }

    // Read parameters
    size_t num_enc, num_dec;
    fread(&num_enc, sizeof(size_t), 1, f);
    if (num_enc == model->num_encoder_params) {
        fread(model->encoder_params, sizeof(double), num_enc, f);
    }
    fread(&num_dec, sizeof(size_t), 1, f);
    if (num_dec == model->num_decoder_params) {
        fread(model->decoder_params, sizeof(double), num_dec, f);
    }

    model->is_trained = true;
    fclose(f);
    return model;
}

bool autoencoder_models_equal(quantum_autoencoder_t* a, quantum_autoencoder_t* b) {
    if (!a || !b) return (!a && !b);
    if (a->input_dim != b->input_dim) return false;
    if (a->latent_dim != b->latent_dim) return false;
    if (a->quantum_depth != b->quantum_depth) return false;
    if (a->num_encoder_params != b->num_encoder_params) return false;
    if (a->num_decoder_params != b->num_decoder_params) return false;

    for (size_t i = 0; i < a->num_encoder_params; i++) {
        if (fabs(a->encoder_params[i] - b->encoder_params[i]) > 1e-6) return false;
    }
    for (size_t i = 0; i < a->num_decoder_params; i++) {
        if (fabs(a->decoder_params[i] - b->decoder_params[i]) > 1e-6) return false;
    }

    return true;
}

// =============================================================================
// Test Helpers Implementation
// =============================================================================

quantum_autoencoder_t* create_test_autoencoder(size_t input_dim,
                                               size_t latent_dim,
                                               size_t quantum_depth) {
    quantum_autoencoder_config_t config = {
        .input_dim = input_dim,
        .latent_dim = latent_dim,
        .quantum_depth = quantum_depth,
        .architecture = {
            .encoder_type = ENCODER_VARIATIONAL,
            .decoder_type = DECODER_QUANTUM,
            .activation = ACTIVATION_QUANTUM_RELU
        },
        .optimization = {
            .learning_rate = 0.01,
            .geometric_enhancement = false,
            .regularization = {
                .type = REG_L2,
                .strength = 0.001
            }
        }
    };

    return quantum_autoencoder_create(&config);
}
