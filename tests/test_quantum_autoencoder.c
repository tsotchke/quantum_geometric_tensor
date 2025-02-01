/**
 * @file test_quantum_autoencoder.c
 * @brief Tests for the quantum autoencoder example
 */

#include <quantum_geometric/core/quantum_geometric_core.h>
#include <quantum_geometric/learning/quantum_stochastic_sampling.h>
#include <quantum_geometric/distributed/distributed_training_manager.h>
#include <quantum_geometric/hardware/quantum_hardware_abstraction.h>
#include "test_helpers.h"

// Test configurations
#define TEST_NUM_TRAIN_SAMPLES 50
#define TEST_NUM_TEST_SAMPLES 10
#define TEST_INPUT_DIM 8
#define TEST_LATENT_DIM 2
#define TEST_QUANTUM_DEPTH 2
#define TEST_BATCH_SIZE 4
#define TEST_NUM_EPOCHS 2

void test_quantum_autoencoder_creation() {
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
    TEST_ASSERT_EQUAL(model->input_dim, TEST_INPUT_DIM, "Incorrect input dimension");
    TEST_ASSERT_EQUAL(model->latent_dim, TEST_LATENT_DIM, "Incorrect latent dimension");
    TEST_ASSERT_EQUAL(model->quantum_depth, TEST_QUANTUM_DEPTH, "Incorrect quantum depth");

    quantum_autoencoder_destroy(model);
}

void test_quantum_state_generation() {
    // Generate synthetic quantum states
    quantum_dataset_t* data = quantum_generate_synthetic_states(
        TEST_NUM_TRAIN_SAMPLES,
        TEST_INPUT_DIM,
        STATE_TYPE_MIXED
    );
    TEST_ASSERT(data != NULL, "Dataset generation failed");

    // Verify dataset properties
    TEST_ASSERT_EQUAL(data->num_samples, TEST_NUM_TRAIN_SAMPLES, "Incorrect number of samples");
    TEST_ASSERT_EQUAL(data->state_dim, TEST_INPUT_DIM, "Incorrect state dimension");
    TEST_ASSERT(data->states != NULL, "States array is NULL");

    // Verify state properties
    for (int i = 0; i < TEST_NUM_TRAIN_SAMPLES; i++) {
        quantum_state_t* state = data->states[i];
        TEST_ASSERT(state != NULL, "State is NULL");
        TEST_ASSERT_EQUAL(state->num_qubits, TEST_INPUT_DIM, "Incorrect number of qubits");
        TEST_ASSERT(quantum_is_valid_state(state), "Invalid quantum state");
        TEST_ASSERT_FLOAT_EQUAL(quantum_trace_norm(state), 1.0, 1e-6, 
                               "State trace norm not normalized");
    }

    quantum_destroy_dataset(data);
}

void test_encoding_decoding() {
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
    TEST_ASSERT_EQUAL(encoded_state->num_qubits, TEST_LATENT_DIM,
                     "Incorrect encoded state dimension");
    TEST_ASSERT(quantum_is_valid_state(encoded_state),
                "Encoded state is not valid");

    // Decode state
    quantum_state_t* decoded_state = quantum_decode_state(model, encoded_state);
    TEST_ASSERT(decoded_state != NULL, "State decoding failed");
    TEST_ASSERT_EQUAL(decoded_state->num_qubits, TEST_INPUT_DIM,
                     "Incorrect decoded state dimension");
    TEST_ASSERT(quantum_is_valid_state(decoded_state),
                "Decoded state is not valid");

    // Verify reconstruction quality
    float fidelity = quantum_state_fidelity(test_state, decoded_state);
    TEST_ASSERT(fidelity >= 0.0 && fidelity <= 1.0,
                "Invalid fidelity value");

    // Cleanup
    quantum_destroy_state(decoded_state);
    quantum_destroy_state(encoded_state);
    quantum_destroy_state(test_state);
    quantum_autoencoder_destroy(model);
}

void test_distributed_autoencoder_training() {
    // Initialize MPI
    int rank = 0, size = 1;
    #ifdef USE_MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    #endif

    // Create quantum system
    quantum_hardware_config_t hw_config = {
        .backend = BACKEND_SIMULATOR,
        .num_qubits = TEST_INPUT_DIM,
        .optimization = {
            .circuit_optimization = true,
            .error_mitigation = true,
            .noise_model = {
                .decoherence = true,
                .gate_errors = true
            }
        }
    };
    quantum_system_t* system = quantum_init_system(&hw_config);
    TEST_ASSERT(system != NULL, "System initialization failed");

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

    // Configure distributed training
    distributed_config_t dist_config = {
        .world_size = size,
        .local_rank = rank,
        .batch_size = TEST_BATCH_SIZE,
        .checkpoint_dir = "/tmp/quantum_geometric/test_checkpoints"
    };
    distributed_manager_t* manager = distributed_manager_create(&dist_config);
    TEST_ASSERT(manager != NULL, "Distributed manager creation failed");

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

    // Train autoencoder
    training_result_t result = quantum_train_distributed(
        model, train_data, manager, &train_config, NULL
    );
    TEST_ASSERT(result.status == TRAINING_SUCCESS, "Training failed");

    // Evaluate autoencoder
    if (rank == 0) {
        evaluation_result_t eval = quantum_evaluate_autoencoder(model, test_data);
        TEST_ASSERT(eval.reconstruction_error >= 0.0, "Invalid reconstruction error");
        TEST_ASSERT(eval.avg_state_fidelity >= 0.0 && eval.avg_state_fidelity <= 1.0,
                   "Invalid average state fidelity");
        TEST_ASSERT(eval.latent_entropy >= 0.0, "Invalid latent entropy");
    }

    // Cleanup
    quantum_destroy_dataset(test_data);
    quantum_destroy_dataset(train_data);
    distributed_manager_destroy(manager);
    quantum_autoencoder_destroy(model);
    quantum_system_destroy(system);

    #ifdef USE_MPI
    MPI_Finalize();
    #endif
}

void test_autoencoder_save_load() {
    // Create and train a model
    quantum_autoencoder_t* model = create_test_autoencoder(
        TEST_INPUT_DIM, TEST_LATENT_DIM, TEST_QUANTUM_DEPTH
    );
    TEST_ASSERT(model != NULL, "Model creation failed");

    // Save model
    const char* save_path = "/tmp/quantum_geometric/test_autoencoder.qg";
    TEST_ASSERT(quantum_save_model(model, save_path) == 0, "Model saving failed");

    // Load model
    quantum_autoencoder_t* loaded_model = quantum_load_model(save_path);
    TEST_ASSERT(loaded_model != NULL, "Model loading failed");

    // Compare models
    TEST_ASSERT(autoencoder_models_equal(model, loaded_model),
                "Loaded model differs from original");

    // Test encoding consistency
    quantum_state_t* test_state = quantum_create_bell_state();
    quantum_state_t* original_encoded = quantum_encode_state(model, test_state);
    quantum_state_t* loaded_encoded = quantum_encode_state(loaded_model, test_state);
    
    float encoding_fidelity = quantum_state_fidelity(original_encoded, loaded_encoded);
    TEST_ASSERT_FLOAT_EQUAL(encoding_fidelity, 1.0, 1e-6,
                           "Inconsistent encoding between original and loaded models");

    // Cleanup
    quantum_destroy_state(loaded_encoded);
    quantum_destroy_state(original_encoded);
    quantum_destroy_state(test_state);
    quantum_autoencoder_destroy(loaded_model);
    quantum_autoencoder_destroy(model);
}

int main() {
    // Register tests
    TEST_BEGIN();
    RUN_TEST(test_quantum_autoencoder_creation);
    RUN_TEST(test_quantum_state_generation);
    RUN_TEST(test_encoding_decoding);
    RUN_TEST(test_distributed_autoencoder_training);
    RUN_TEST(test_autoencoder_save_load);
    TEST_END();

    return 0;
}
