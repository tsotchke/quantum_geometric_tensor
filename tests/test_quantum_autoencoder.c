/**
 * @file test_quantum_autoencoder.c
 * @brief Tests for the quantum autoencoder module
 */

#include <quantum_geometric/core/quantum_geometric_core.h>
#include <quantum_geometric/learning/quantum_autoencoder.h>
#include "test_helpers.h"

// Test configurations
#define TEST_NUM_TRAIN_SAMPLES 50
#define TEST_NUM_TEST_SAMPLES 10
#define TEST_INPUT_DIM 4
#define TEST_LATENT_DIM 2
#define TEST_QUANTUM_DEPTH 2
#define TEST_BATCH_SIZE 4
#define TEST_NUM_EPOCHS 2

void test_quantum_autoencoder_creation(void) {
    // Configure autoencoder
    quantum_autoencoder_config_t config = {
        .input_dim = TEST_INPUT_DIM,
        .latent_dim = TEST_LATENT_DIM,
        .quantum_depth = TEST_QUANTUM_DEPTH,
        .architecture = {
            .encoder_type = ENCODER_VARIATIONAL,
            .decoder_type = DECODER_QUANTUM,
            .activation = ACTIVATION_QUANTUM_RELU
        },
        .optimization = {
            .learning_rate = 0.001,
            .geometric_enhancement = true,
            .regularization = {
                .type = REG_QUANTUM_ENTROPY,
                .strength = 0.1
            }
        }
    };

    // Create autoencoder
    quantum_autoencoder_t* model = quantum_autoencoder_create(&config);
    TEST_ASSERT(model != NULL, "Autoencoder creation failed");

    // Verify model parameters
    TEST_ASSERT(model->input_dim == TEST_INPUT_DIM, "Incorrect input dimension");
    TEST_ASSERT(model->latent_dim == TEST_LATENT_DIM, "Incorrect latent dimension");
    TEST_ASSERT(model->quantum_depth == TEST_QUANTUM_DEPTH, "Incorrect quantum depth");

    quantum_autoencoder_destroy(model);
}

void test_quantum_state_generation(void) {
    // Generate synthetic quantum states
    quantum_dataset_t* data = quantum_generate_synthetic_states(
        TEST_NUM_TRAIN_SAMPLES,
        TEST_INPUT_DIM,
        STATE_TYPE_MIXED
    );
    TEST_ASSERT(data != NULL, "Dataset generation failed");

    // Verify dataset properties
    TEST_ASSERT(data->num_samples == TEST_NUM_TRAIN_SAMPLES, "Incorrect number of samples");
    TEST_ASSERT(data->state_dim == TEST_INPUT_DIM, "Incorrect state dimension");
    TEST_ASSERT(data->states != NULL, "States array is NULL");

    // Verify state properties
    for (size_t i = 0; i < TEST_NUM_TRAIN_SAMPLES; i++) {
        quantum_state_t* state = data->states[i];
        TEST_ASSERT(state != NULL, "State is NULL");
    }

    quantum_destroy_quantum_dataset(data);
}

void test_encoding_decoding(void) {
    // Create autoencoder
    quantum_autoencoder_t* model = create_test_autoencoder(
        TEST_INPUT_DIM, TEST_LATENT_DIM, TEST_QUANTUM_DEPTH
    );
    TEST_ASSERT(model != NULL, "Autoencoder creation failed");

    // Create test state
    quantum_state_t* test_state = quantum_create_bell_state();
    TEST_ASSERT(test_state != NULL, "Bell state creation failed");

    // Encode state
    quantum_state_t* encoded_state = quantum_encode_state(model, test_state);
    TEST_ASSERT(encoded_state != NULL, "State encoding failed");
    TEST_ASSERT(encoded_state->num_qubits == TEST_LATENT_DIM,
                "Incorrect encoded state dimension");

    // Decode state
    quantum_state_t* decoded_state = quantum_decode_state(model, encoded_state);
    TEST_ASSERT(decoded_state != NULL, "State decoding failed");
    TEST_ASSERT(decoded_state->num_qubits == TEST_INPUT_DIM,
                "Incorrect decoded state dimension");

    // Verify reconstruction quality using autoencoder-specific fidelity
    float fidelity = quantum_autoencoder_state_fidelity(test_state, decoded_state);
    TEST_ASSERT(fidelity >= 0.0f && fidelity <= 1.0f, "Invalid fidelity value");

    // Cleanup
    quantum_destroy_state(decoded_state);
    quantum_destroy_state(encoded_state);
    quantum_destroy_state(test_state);
    quantum_autoencoder_destroy(model);
}

void test_autoencoder_training(void) {
    // Create autoencoder
    quantum_autoencoder_config_t model_config = {
        .input_dim = TEST_INPUT_DIM,
        .latent_dim = TEST_LATENT_DIM,
        .quantum_depth = TEST_QUANTUM_DEPTH,
        .architecture = {
            .encoder_type = ENCODER_VARIATIONAL,
            .decoder_type = DECODER_QUANTUM,
            .activation = ACTIVATION_QUANTUM_RELU
        },
        .optimization = {
            .learning_rate = 0.001,
            .geometric_enhancement = true,
            .regularization = {
                .type = REG_QUANTUM_ENTROPY,
                .strength = 0.1
            }
        }
    };
    quantum_autoencoder_t* model = quantum_autoencoder_create(&model_config);
    TEST_ASSERT(model != NULL, "Model creation failed");

    // Generate test data
    quantum_dataset_t* train_data = quantum_generate_synthetic_states(
        TEST_NUM_TRAIN_SAMPLES, TEST_INPUT_DIM, STATE_TYPE_MIXED
    );
    quantum_dataset_t* test_data = quantum_generate_synthetic_states(
        TEST_NUM_TEST_SAMPLES, TEST_INPUT_DIM, STATE_TYPE_MIXED
    );
    TEST_ASSERT(train_data != NULL && test_data != NULL, "Dataset generation failed");

    // Training configuration
    training_config_t train_config = {
        .num_epochs = TEST_NUM_EPOCHS,
        .batch_size = TEST_BATCH_SIZE,
        .learning_rate = 0.001,
        .optimization = {
            .geometric_enhancement = true,
            .error_mitigation = true,
            .early_stopping = {
                .enabled = true,
                .patience = 2,
                .min_delta = 1e-4
            }
        }
    };

    // Train autoencoder (without distributed manager for simplicity)
    training_result_t result = quantum_train_autoencoder_distributed(
        model, train_data, NULL, &train_config, NULL
    );
    TEST_ASSERT(result.status == TRAINING_SUCCESS || result.status == TRAINING_EARLY_STOPPED,
                "Training failed");

    // Evaluate autoencoder
    autoencoder_evaluation_result_t eval = quantum_evaluate_autoencoder(model, test_data);
    TEST_ASSERT(eval.reconstruction_error >= 0.0, "Invalid reconstruction error");
    TEST_ASSERT(eval.avg_state_fidelity >= 0.0 && eval.avg_state_fidelity <= 1.0,
               "Invalid average state fidelity");

    // Cleanup
    if (result.loss_history) free(result.loss_history);
    quantum_destroy_quantum_dataset(test_data);
    quantum_destroy_quantum_dataset(train_data);
    quantum_autoencoder_destroy(model);
}

void test_autoencoder_save_load(void) {
    // Create a model
    quantum_autoencoder_t* model = create_test_autoencoder(
        TEST_INPUT_DIM, TEST_LATENT_DIM, TEST_QUANTUM_DEPTH
    );
    TEST_ASSERT(model != NULL, "Model creation failed");

    // Save model
    const char* save_path = "/tmp/test_autoencoder.qg";
    TEST_ASSERT(quantum_save_autoencoder_model(model, save_path) == 0, "Model saving failed");

    // Load model
    quantum_autoencoder_t* loaded_model = quantum_load_autoencoder_model(save_path);
    TEST_ASSERT(loaded_model != NULL, "Model loading failed");

    // Compare models
    TEST_ASSERT(autoencoder_models_equal(model, loaded_model),
                "Loaded model differs from original");

    // Test encoding consistency
    quantum_state_t* test_state = quantum_create_bell_state();
    quantum_state_t* original_encoded = quantum_encode_state(model, test_state);
    quantum_state_t* loaded_encoded = quantum_encode_state(loaded_model, test_state);

    float encoding_fidelity = quantum_autoencoder_state_fidelity(original_encoded, loaded_encoded);
    TEST_ASSERT(encoding_fidelity > 0.99f, "Inconsistent encoding between original and loaded models");

    // Cleanup
    quantum_destroy_state(loaded_encoded);
    quantum_destroy_state(original_encoded);
    quantum_destroy_state(test_state);
    quantum_autoencoder_destroy(loaded_model);
    quantum_autoencoder_destroy(model);
}

int main(void) {
    printf("Running quantum autoencoder tests...\n\n");

    printf("Test 1: Autoencoder creation\n");
    test_quantum_autoencoder_creation();
    printf("  PASSED\n\n");

    printf("Test 2: Quantum state generation\n");
    test_quantum_state_generation();
    printf("  PASSED\n\n");

    printf("Test 3: Encoding/decoding\n");
    test_encoding_decoding();
    printf("  PASSED\n\n");

    printf("Test 4: Autoencoder training\n");
    test_autoencoder_training();
    printf("  PASSED\n\n");

    printf("Test 5: Save/load model\n");
    test_autoencoder_save_load();
    printf("  PASSED\n\n");

    printf("All quantum autoencoder tests passed!\n");
    return 0;
}
