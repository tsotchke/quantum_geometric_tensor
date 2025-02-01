/**
 * @file test_quantum_regression.c
 * @brief Tests for the quantum regression example
 */

#include <quantum_geometric/core/quantum_geometric_core.h>
#include <quantum_geometric/learning/quantum_stochastic_sampling.h>
#include <quantum_geometric/distributed/distributed_training_manager.h>
#include <quantum_geometric/hardware/quantum_hardware_abstraction.h>
#include "test_helpers.h"

// Test configurations
#define TEST_NUM_TRAIN_SAMPLES 100
#define TEST_NUM_TEST_SAMPLES 20
#define TEST_INPUT_DIM 4
#define TEST_OUTPUT_DIM 1
#define TEST_QUANTUM_DEPTH 2
#define TEST_BATCH_SIZE 8
#define TEST_NUM_EPOCHS 2

void test_quantum_regression_model_creation() {
    // Configure model
    quantum_model_config_t config = {
        .input_dim = TEST_INPUT_DIM,
        .output_dim = TEST_OUTPUT_DIM,
        .quantum_depth = TEST_QUANTUM_DEPTH,
        .measurement_basis = MEASUREMENT_BASIS_CONTINUOUS,
        .optimization = {
            .learning_rate = 0.001,
            .geometric_enhancement = true,
            .loss_function = LOSS_MSE
        }
    };

    // Create model
    quantum_model_t* model = quantum_model_create(&config);
    TEST_ASSERT(model != NULL, "Model creation failed");

    // Verify model parameters
    TEST_ASSERT_EQUAL(model->input_dim, TEST_INPUT_DIM, "Incorrect input dimension");
    TEST_ASSERT_EQUAL(model->output_dim, TEST_OUTPUT_DIM, "Incorrect output dimension");
    TEST_ASSERT_EQUAL(model->quantum_depth, TEST_QUANTUM_DEPTH, "Incorrect quantum depth");
    TEST_ASSERT_EQUAL(model->measurement_basis, MEASUREMENT_BASIS_CONTINUOUS, 
                     "Incorrect measurement basis");

    quantum_model_destroy(model);
}

void test_regression_data_generation() {
    // Generate synthetic dataset
    dataset_t* data = quantum_generate_synthetic_data(
        TEST_NUM_TRAIN_SAMPLES,
        TEST_INPUT_DIM,
        REGRESSION_CONTINUOUS
    );
    TEST_ASSERT(data != NULL, "Dataset generation failed");

    // Verify dataset properties
    TEST_ASSERT_EQUAL(data->num_samples, TEST_NUM_TRAIN_SAMPLES, "Incorrect number of samples");
    TEST_ASSERT_EQUAL(data->feature_dim, TEST_INPUT_DIM, "Incorrect feature dimension");
    TEST_ASSERT(data->targets != NULL, "Targets array is NULL");
    TEST_ASSERT(data->features != NULL, "Features array is NULL");

    // Verify data ranges
    for (int i = 0; i < TEST_NUM_TRAIN_SAMPLES; i++) {
        TEST_ASSERT(data->targets[i] >= -10.0 && data->targets[i] <= 10.0,
                   "Target value out of expected range");
        for (int j = 0; j < TEST_INPUT_DIM; j++) {
            TEST_ASSERT(data->features[i][j] >= -1.0 && data->features[i][j] <= 1.0,
                       "Feature value out of range");
        }
    }

    quantum_destroy_dataset(data);
}

void test_distributed_regression_training() {
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
        .num_qubits = TEST_INPUT_DIM + 2,
        .optimization = {
            .circuit_optimization = true,
            .error_mitigation = true,
            .continuous_variable = true
        }
    };
    quantum_system_t* system = quantum_init_system(&hw_config);
    TEST_ASSERT(system != NULL, "System initialization failed");

    // Create model
    quantum_model_config_t model_config = {
        .input_dim = TEST_INPUT_DIM,
        .output_dim = TEST_OUTPUT_DIM,
        .quantum_depth = TEST_QUANTUM_DEPTH,
        .measurement_basis = MEASUREMENT_BASIS_CONTINUOUS,
        .optimization = {
            .learning_rate = 0.001,
            .geometric_enhancement = true,
            .loss_function = LOSS_MSE
        }
    };
    quantum_model_t* model = quantum_model_create(&model_config);
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
    dataset_t* train_data = quantum_generate_synthetic_data(
        TEST_NUM_TRAIN_SAMPLES, TEST_INPUT_DIM, REGRESSION_CONTINUOUS
    );
    dataset_t* test_data = quantum_generate_synthetic_data(
        TEST_NUM_TEST_SAMPLES, TEST_INPUT_DIM, REGRESSION_CONTINUOUS
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

    // Train model
    training_result_t result = quantum_train_distributed(
        model, train_data, manager, &train_config, NULL
    );
    TEST_ASSERT(result.status == TRAINING_SUCCESS, "Training failed");

    // Evaluate model
    if (rank == 0) {
        evaluation_result_t eval = quantum_evaluate(model, test_data);
        TEST_ASSERT(eval.mse >= 0.0, "Invalid MSE value");
        TEST_ASSERT(eval.mae >= 0.0, "Invalid MAE value");
        TEST_ASSERT(eval.r2_score <= 1.0 && eval.r2_score >= -1.0, "Invalid R² score");
    }

    // Cleanup
    quantum_destroy_dataset(test_data);
    quantum_destroy_dataset(train_data);
    distributed_manager_destroy(manager);
    quantum_model_destroy(model);
    quantum_system_destroy(system);

    #ifdef USE_MPI
    MPI_Finalize();
    #endif
}

void test_regression_model_save_load() {
    // Create and train a model
    quantum_model_t* model = create_test_regression_model(
        TEST_INPUT_DIM, TEST_OUTPUT_DIM, TEST_QUANTUM_DEPTH
    );
    TEST_ASSERT(model != NULL, "Model creation failed");

    // Save model
    const char* save_path = "/tmp/quantum_geometric/test_regression_model.qg";
    TEST_ASSERT(quantum_save_model(model, save_path) == 0, "Model saving failed");

    // Load model
    quantum_model_t* loaded_model = quantum_load_model(save_path);
    TEST_ASSERT(loaded_model != NULL, "Model loading failed");

    // Compare models
    TEST_ASSERT(regression_models_equal(model, loaded_model), 
                "Loaded model differs from original");

    // Test prediction consistency
    float test_input[TEST_INPUT_DIM] = {0.1, 0.2, 0.3, 0.4};
    float original_pred = quantum_predict_single(model, test_input);
    float loaded_pred = quantum_predict_single(loaded_model, test_input);
    TEST_ASSERT_FLOAT_EQUAL(original_pred, loaded_pred, 1e-6, 
                           "Predictions differ between original and loaded model");

    // Cleanup
    quantum_model_destroy(model);
    quantum_model_destroy(loaded_model);
}

void test_regression_error_metrics() {
    // Create test predictions and targets
    const int n_samples = 10;
    float predictions[10] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    float targets[10] = {1.1, 2.1, 2.9, 4.2, 5.0, 5.8, 7.2, 7.8, 9.1, 10.0};

    // Calculate metrics
    float mse = quantum_calculate_mse(predictions, targets, n_samples);
    float mae = quantum_calculate_mae(predictions, targets, n_samples);
    float r2 = quantum_calculate_r2_score(predictions, targets, n_samples);

    // Verify metrics
    TEST_ASSERT(mse >= 0.0, "MSE should be non-negative");
    TEST_ASSERT(mae >= 0.0, "MAE should be non-negative");
    TEST_ASSERT(r2 <= 1.0 && r2 >= -1.0, "R² score should be between -1 and 1");

    // Verify with known values
    TEST_ASSERT_FLOAT_EQUAL(mse, 0.0390, 1e-4, "Incorrect MSE calculation");
    TEST_ASSERT_FLOAT_EQUAL(mae, 0.1600, 1e-4, "Incorrect MAE calculation");
    TEST_ASSERT_FLOAT_EQUAL(r2, 0.9978, 1e-4, "Incorrect R² calculation");
}

int main() {
    // Register tests
    TEST_BEGIN();
    RUN_TEST(test_quantum_regression_model_creation);
    RUN_TEST(test_regression_data_generation);
    RUN_TEST(test_distributed_regression_training);
    RUN_TEST(test_regression_model_save_load);
    RUN_TEST(test_regression_error_metrics);
    TEST_END();

    return 0;
}
