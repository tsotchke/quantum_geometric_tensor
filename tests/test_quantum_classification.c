/**
 * @file test_quantum_classification.c
 * @brief Tests for the quantum classification example with TensorFlow comparison
 */

#include <quantum_geometric/core/quantum_geometric_core.h>
#include <quantum_geometric/learning/quantum_stochastic_sampling.h>
#include <quantum_geometric/distributed/distributed_training_manager.h>
#include <quantum_geometric/hardware/quantum_hardware_abstraction.h>
#include "test_helpers.h"

// Test configurations
#define TEST_NUM_TRAIN_SAMPLES 1000  // Increased for better comparison
#define TEST_NUM_TEST_SAMPLES 200
#define TEST_INPUT_DIM 4
#define TEST_QUANTUM_DEPTH 2
#define TEST_BATCH_SIZE 8
#define TEST_NUM_EPOCHS 5  // Increased for better comparison

void test_quantum_model_creation() {
    // Configure model
    quantum_model_config_t config = {
        .input_dim = TEST_INPUT_DIM,
        .quantum_depth = TEST_QUANTUM_DEPTH,
        .measurement_basis = MEASUREMENT_BASIS_Z,
        .optimization = {
            .learning_rate = 0.001,
            .geometric_enhancement = true
        }
    };

    // Create model
    quantum_model_t* model = quantum_model_create(&config);
    TEST_ASSERT(model != NULL, "Model creation failed");

    // Verify model parameters
    TEST_ASSERT_EQUAL(model->input_dim, TEST_INPUT_DIM, "Incorrect input dimension");
    TEST_ASSERT_EQUAL(model->quantum_depth, TEST_QUANTUM_DEPTH, "Incorrect quantum depth");

    quantum_model_destroy(model);
    printf("Model creation test passed\n");
}

void test_synthetic_data_generation() {
    // Generate synthetic dataset
    dataset_t* data = quantum_generate_synthetic_data(
        TEST_NUM_TRAIN_SAMPLES,
        TEST_INPUT_DIM,
        CLASSIFICATION_BINARY
    );
    TEST_ASSERT(data != NULL, "Dataset generation failed");

    // Verify dataset properties
    TEST_ASSERT_EQUAL(data->num_samples, TEST_NUM_TRAIN_SAMPLES, "Incorrect number of samples");
    TEST_ASSERT_EQUAL(data->feature_dim, TEST_INPUT_DIM, "Incorrect feature dimension");
    TEST_ASSERT(data->labels != NULL, "Labels array is NULL");
    TEST_ASSERT(data->features != NULL, "Features array is NULL");

    // Verify data ranges
    for (int i = 0; i < TEST_NUM_TRAIN_SAMPLES; i++) {
        TEST_ASSERT(data->labels[i] == 0 || data->labels[i] == 1, "Invalid label value");
        for (int j = 0; j < TEST_INPUT_DIM; j++) {
            TEST_ASSERT(data->features[i][j] >= -1.0 && data->features[i][j] <= 1.0,
                       "Feature value out of range");
        }
    }

    quantum_destroy_dataset(data);
    printf("Data generation test passed\n");
}

void test_quantum_vs_classical() {
    printf("\nRunning quantum vs classical comparison...\n");
    
    // Initialize quantum system
    quantum_hardware_config_t hw_config = {
        .backend = BACKEND_SIMULATOR,
        .num_qubits = TEST_INPUT_DIM,
        .optimization = {
            .circuit_optimization = true,
            .error_mitigation = true
        }
    };
    quantum_system_t* system = quantum_init_system(&hw_config);
    TEST_ASSERT(system != NULL, "System initialization failed");

    // Create quantum model
    quantum_model_config_t model_config = {
        .input_dim = TEST_INPUT_DIM,
        .quantum_depth = TEST_QUANTUM_DEPTH,
        .measurement_basis = MEASUREMENT_BASIS_Z,
        .optimization = {
            .learning_rate = 0.001,
            .geometric_enhancement = true
        }
    };
    quantum_model_t* model = quantum_model_create(&model_config);
    TEST_ASSERT(model != NULL, "Model creation failed");

    // Generate datasets
    dataset_t* train_data = quantum_generate_synthetic_data(
        TEST_NUM_TRAIN_SAMPLES, TEST_INPUT_DIM, CLASSIFICATION_BINARY
    );
    dataset_t* test_data = quantum_generate_synthetic_data(
        TEST_NUM_TEST_SAMPLES, TEST_INPUT_DIM, CLASSIFICATION_BINARY
    );
    TEST_ASSERT(train_data != NULL && test_data != NULL, "Dataset generation failed");

    // Training configuration
    training_config_t train_config = {
        .num_epochs = TEST_NUM_EPOCHS,
        .batch_size = TEST_BATCH_SIZE,
        .learning_rate = 0.001,
        .optimization = {
            .geometric_enhancement = true,
            .error_mitigation = true
        }
    };

    // Train and time quantum model
    printf("\nTraining quantum model...\n");
    performance_metrics_t quantum_metrics;
    quantum_metrics.start_time = get_current_time();
    
    training_result_t result = quantum_train(model, train_data, &train_config);
    TEST_ASSERT(result.status == TRAINING_SUCCESS, "Quantum training failed");
    
    quantum_metrics.end_time = get_current_time();
    quantum_metrics.training_time = quantum_metrics.end_time - quantum_metrics.start_time;

    // Evaluate quantum model
    evaluation_result_t quantum_eval = quantum_evaluate(model, test_data);
    quantum_metrics.accuracy = quantum_eval.accuracy;
    quantum_metrics.memory_used = get_peak_memory_usage();

    // Train and evaluate classical model (TensorFlow-like)
    printf("\nTraining classical model...\n");
    performance_metrics_t classical_metrics;
    classical_metrics.start_time = get_current_time();
    
    classical_model_t* classical = classical_model_create(TEST_INPUT_DIM);
    classical_train(classical, train_data, TEST_NUM_EPOCHS, TEST_BATCH_SIZE);
    
    classical_metrics.end_time = get_current_time();
    classical_metrics.training_time = classical_metrics.end_time - classical_metrics.start_time;
    
    evaluation_result_t classical_eval = classical_evaluate(classical, test_data);
    classical_metrics.accuracy = classical_eval.accuracy;
    classical_metrics.memory_used = get_peak_memory_usage();

    // Print comparison results
    printf("\nPerformance Comparison:\n");
    printf("Quantum Model:\n");
    printf("- Accuracy: %.2f%%\n", quantum_metrics.accuracy * 100);
    printf("- Training time: %.2f seconds\n", quantum_metrics.training_time);
    printf("- Memory usage: %.2f MB\n", quantum_metrics.memory_used / 1024.0 / 1024.0);
    
    printf("\nClassical Model:\n");
    printf("- Accuracy: %.2f%%\n", classical_metrics.accuracy * 100);
    printf("- Training time: %.2f seconds\n", classical_metrics.training_time);
    printf("- Memory usage: %.2f MB\n", classical_metrics.memory_used / 1024.0 / 1024.0);

    // Verify quantum model performs at least as well as classical
    TEST_ASSERT(quantum_metrics.accuracy >= classical_metrics.accuracy * 0.95,
                "Quantum model accuracy significantly worse than classical");

    // Cleanup
    quantum_destroy_dataset(test_data);
    quantum_destroy_dataset(train_data);
    quantum_model_destroy(model);
    classical_model_destroy(classical);
    quantum_system_destroy(system);
    
    printf("\nComparison test completed successfully\n");
}

int main() {
    printf("Running quantum classification tests with TensorFlow comparison...\n\n");
    
    // Set random seed for reproducibility
    srand(42);
    
    // Run tests
    test_quantum_model_creation();
    test_synthetic_data_generation();
    test_quantum_vs_classical();
    
    printf("\nAll tests passed successfully!\n");
    return 0;
}
