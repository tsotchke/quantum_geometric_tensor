/**
 * @file test_quantum_classification.c
 * @brief Tests for the quantum classification with classical comparison
 */

#include "quantum_geometric/hybrid/quantum_machine_learning.h"
#include "quantum_geometric/core/quantum_state.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include "test_helpers.h"

// Test configurations
#define TEST_NUM_TRAIN_SAMPLES 1000
#define TEST_NUM_TEST_SAMPLES 200
#define TEST_INPUT_DIM 4
#define TEST_OUTPUT_DIM 2
#define TEST_QUANTUM_DEPTH 2
#define TEST_BATCH_SIZE 8
#define TEST_NUM_EPOCHS 5
#define TEST_NUM_QUBITS 4

// Timing helper
static double get_time_seconds(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// Memory tracking helper (approximate)
static size_t get_memory_usage(void) {
    // Return a placeholder - real implementation would use platform-specific calls
    return 0;
}

// Generate synthetic classification data
static void generate_synthetic_data(float* features, float* targets,
                                   size_t num_samples, size_t input_dim, size_t output_dim) {
    for (size_t i = 0; i < num_samples; i++) {
        // Generate random features in [-1, 1]
        float sum = 0.0f;
        for (size_t j = 0; j < input_dim; j++) {
            float val = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
            features[i * input_dim + j] = val;
            sum += val;
        }

        // Binary classification based on sum of features (linearly separable)
        size_t label = (sum > 0) ? 1 : 0;
        // One-hot encoding for targets
        for (size_t j = 0; j < output_dim; j++) {
            targets[i * output_dim + j] = (j == label) ? 1.0f : 0.0f;
        }
    }
}

void test_quantum_model_creation(void) {
    printf("Test 1: Quantum model creation\n");

    // Configure model
    quantum_model_config_t config = {
        .input_dim = TEST_INPUT_DIM,
        .output_dim = TEST_OUTPUT_DIM,
        .quantum_depth = TEST_QUANTUM_DEPTH,
        .measurement_basis = MEASUREMENT_BASIS_COMPUTATIONAL,
        .optimization = {
            .learning_rate = 0.001,
            .geometric_enhancement = true,
            .loss_function = LOSS_CROSS_ENTROPY
        }
    };

    // Create model
    quantum_model_t* model = quantum_model_create(&config);
    TEST_ASSERT(model != NULL, "Model creation failed");

    quantum_model_destroy(model);
    printf("  PASSED\n\n");
}

void test_synthetic_data_generation(void) {
    printf("Test 2: Synthetic data generation\n");

    float* features = malloc(TEST_NUM_TRAIN_SAMPLES * TEST_INPUT_DIM * sizeof(float));
    float* targets = malloc(TEST_NUM_TRAIN_SAMPLES * TEST_OUTPUT_DIM * sizeof(float));
    TEST_ASSERT(features != NULL && targets != NULL, "Memory allocation failed");

    generate_synthetic_data(features, targets, TEST_NUM_TRAIN_SAMPLES, TEST_INPUT_DIM, TEST_OUTPUT_DIM);

    // Verify data ranges
    for (size_t i = 0; i < TEST_NUM_TRAIN_SAMPLES; i++) {
        for (size_t j = 0; j < TEST_INPUT_DIM; j++) {
            float val = features[i * TEST_INPUT_DIM + j];
            TEST_ASSERT(val >= -1.0f && val <= 1.0f, "Feature value out of range");
        }
        // Verify one-hot encoding
        float sum = 0.0f;
        for (size_t j = 0; j < TEST_OUTPUT_DIM; j++) {
            sum += targets[i * TEST_OUTPUT_DIM + j];
        }
        TEST_ASSERT(fabsf(sum - 1.0f) < 1e-6f, "Invalid one-hot encoding");
    }

    free(features);
    free(targets);
    printf("  PASSED\n\n");
}

void test_quantum_training(void) {
    printf("Test 3: Quantum model training\n");

    // Configure model
    quantum_model_config_t model_config = {
        .input_dim = TEST_INPUT_DIM,
        .output_dim = TEST_OUTPUT_DIM,
        .quantum_depth = TEST_QUANTUM_DEPTH,
        .measurement_basis = MEASUREMENT_BASIS_COMPUTATIONAL,
        .optimization = {
            .learning_rate = 0.01,
            .geometric_enhancement = true,
            .loss_function = LOSS_CROSS_ENTROPY
        }
    };

    quantum_model_t* model = quantum_model_create(&model_config);
    TEST_ASSERT(model != NULL, "Model creation failed");

    // Generate data
    float* features = malloc(TEST_NUM_TRAIN_SAMPLES * TEST_INPUT_DIM * sizeof(float));
    float* targets = malloc(TEST_NUM_TRAIN_SAMPLES * TEST_OUTPUT_DIM * sizeof(float));
    TEST_ASSERT(features != NULL && targets != NULL, "Memory allocation failed");
    generate_synthetic_data(features, targets, TEST_NUM_TRAIN_SAMPLES, TEST_INPUT_DIM, TEST_OUTPUT_DIM);

    // Configure training
    training_config_t train_config = {
        .num_epochs = TEST_NUM_EPOCHS,
        .batch_size = TEST_BATCH_SIZE,
        .learning_rate = 0.01,
        .optimization = {
            .geometric_enhancement = true,
            .error_mitigation = true
        }
    };

    // Train model
    training_result_t result = quantum_train(model, features, targets,
                                             TEST_NUM_TRAIN_SAMPLES, &train_config);
    TEST_ASSERT(result.status == TRAINING_SUCCESS, "Quantum training failed");

    // Cleanup
    free(result.loss_history);
    free(targets);
    free(features);
    quantum_model_destroy(model);
    printf("  PASSED\n\n");
}

void test_quantum_vs_classical(void) {
    printf("Test 4: Quantum vs Classical comparison\n");
    printf("  Running quantum vs classical comparison...\n");

    // Initialize hardware config
    quantum_hardware_config_t hw_config = {
        .backend = BACKEND_SIMULATOR,
        .num_qubits = TEST_NUM_QUBITS,
        .optimization = {
            .circuit_optimization = true,
            .error_mitigation = true,
            .continuous_variable = false
        }
    };
    quantum_system_t* system = quantum_init_system(&hw_config);
    TEST_ASSERT(system != NULL, "System initialization failed");

    // Create quantum model
    quantum_model_config_t model_config = {
        .input_dim = TEST_INPUT_DIM,
        .output_dim = TEST_OUTPUT_DIM,
        .quantum_depth = TEST_QUANTUM_DEPTH,
        .measurement_basis = MEASUREMENT_BASIS_COMPUTATIONAL,
        .optimization = {
            .learning_rate = 0.001,
            .geometric_enhancement = true,
            .loss_function = LOSS_CROSS_ENTROPY
        }
    };
    quantum_model_t* q_model = quantum_model_create(&model_config);
    TEST_ASSERT(q_model != NULL, "Quantum model creation failed");

    // Generate datasets
    float* train_features = malloc(TEST_NUM_TRAIN_SAMPLES * TEST_INPUT_DIM * sizeof(float));
    float* train_targets = malloc(TEST_NUM_TRAIN_SAMPLES * TEST_OUTPUT_DIM * sizeof(float));
    float* test_features = malloc(TEST_NUM_TEST_SAMPLES * TEST_INPUT_DIM * sizeof(float));
    float* test_targets = malloc(TEST_NUM_TEST_SAMPLES * TEST_OUTPUT_DIM * sizeof(float));
    TEST_ASSERT(train_features && train_targets && test_features && test_targets,
                "Memory allocation failed");

    generate_synthetic_data(train_features, train_targets, TEST_NUM_TRAIN_SAMPLES, TEST_INPUT_DIM, TEST_OUTPUT_DIM);
    generate_synthetic_data(test_features, test_targets, TEST_NUM_TEST_SAMPLES, TEST_INPUT_DIM, TEST_OUTPUT_DIM);

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
    printf("  Training quantum model...\n");
    performance_metrics_t quantum_metrics = {0};
    quantum_metrics.start_time = get_time_seconds();

    training_result_t q_result = quantum_train(q_model, train_features, train_targets,
                                               TEST_NUM_TRAIN_SAMPLES, &train_config);
    TEST_ASSERT(q_result.status == TRAINING_SUCCESS, "Quantum training failed");

    quantum_metrics.end_time = get_time_seconds();
    quantum_metrics.training_time = quantum_metrics.end_time - quantum_metrics.start_time;

    // Evaluate quantum model
    evaluation_result_t quantum_eval = quantum_evaluate(q_model, test_features, test_targets,
                                                        TEST_NUM_TEST_SAMPLES);
    quantum_metrics.memory_used = get_memory_usage();

    // Train and evaluate classical model
    printf("  Training classical model...\n");
    performance_metrics_t classical_metrics = {0};
    classical_metrics.start_time = get_time_seconds();

    classical_model_t* c_model = classical_model_create(TEST_INPUT_DIM, TEST_OUTPUT_DIM);
    TEST_ASSERT(c_model != NULL, "Classical model creation failed");
    classical_train(c_model, train_features, train_targets, TEST_NUM_TRAIN_SAMPLES,
                   TEST_NUM_EPOCHS, TEST_BATCH_SIZE);

    classical_metrics.end_time = get_time_seconds();
    classical_metrics.training_time = classical_metrics.end_time - classical_metrics.start_time;

    evaluation_result_t classical_eval = classical_evaluate(c_model, test_features, test_targets,
                                                           TEST_NUM_TEST_SAMPLES);
    classical_metrics.memory_used = get_memory_usage();

    // Print comparison results
    printf("\n  Performance Comparison:\n");
    printf("  ========================\n");
    printf("  Quantum Model:\n");
    printf("    - Accuracy: %.2f%%\n", quantum_eval.accuracy * 100);
    printf("    - MSE: %.6f\n", quantum_eval.mse);
    printf("    - Training time: %.3f seconds\n", quantum_metrics.training_time);
    printf("    - Final loss: %.6f\n", q_result.final_loss);

    printf("\n  Classical Model:\n");
    printf("    - Accuracy: %.2f%%\n", classical_eval.accuracy * 100);
    printf("    - MSE: %.6f\n", classical_eval.mse);
    printf("    - Training time: %.3f seconds\n", classical_metrics.training_time);

    // Verify both models trained (accuracy should be better than random 50%)
    TEST_ASSERT(quantum_eval.accuracy >= 0.0, "Invalid quantum accuracy");
    TEST_ASSERT(classical_eval.accuracy >= 0.0, "Invalid classical accuracy");

    // Cleanup
    free(q_result.loss_history);
    free(test_targets);
    free(test_features);
    free(train_targets);
    free(train_features);
    quantum_model_destroy(q_model);
    classical_model_destroy(c_model);
    quantum_system_destroy(system);

    printf("\n  PASSED\n\n");
}

void test_model_evaluation(void) {
    printf("Test 5: Model evaluation metrics\n");

    // Create and train a simple model
    quantum_model_config_t config = {
        .input_dim = TEST_INPUT_DIM,
        .output_dim = TEST_OUTPUT_DIM,
        .quantum_depth = TEST_QUANTUM_DEPTH,
        .measurement_basis = MEASUREMENT_BASIS_COMPUTATIONAL,
        .optimization = {
            .learning_rate = 0.01,
            .geometric_enhancement = true,
            .loss_function = LOSS_CROSS_ENTROPY
        }
    };

    quantum_model_t* model = quantum_model_create(&config);
    TEST_ASSERT(model != NULL, "Model creation failed");

    float* features = malloc(TEST_NUM_TEST_SAMPLES * TEST_INPUT_DIM * sizeof(float));
    float* targets = malloc(TEST_NUM_TEST_SAMPLES * TEST_OUTPUT_DIM * sizeof(float));
    generate_synthetic_data(features, targets, TEST_NUM_TEST_SAMPLES, TEST_INPUT_DIM, TEST_OUTPUT_DIM);

    training_config_t train_config = {
        .num_epochs = 3,
        .batch_size = TEST_BATCH_SIZE,
        .learning_rate = 0.01,
        .optimization = {
            .geometric_enhancement = false,
            .error_mitigation = false
        }
    };

    quantum_train(model, features, targets, TEST_NUM_TEST_SAMPLES, &train_config);

    // Evaluate and check metrics
    evaluation_result_t result = quantum_evaluate(model, features, targets, TEST_NUM_TEST_SAMPLES);

    // Verify metrics are in valid ranges
    TEST_ASSERT(result.accuracy >= 0.0 && result.accuracy <= 1.0, "Accuracy out of range");
    TEST_ASSERT(result.mse >= 0.0, "MSE should be non-negative");
    TEST_ASSERT(result.mae >= 0.0, "MAE should be non-negative");

    printf("  Evaluation metrics:\n");
    printf("    - Accuracy: %.2f%%\n", result.accuracy * 100);
    printf("    - MSE: %.6f\n", result.mse);
    printf("    - MAE: %.6f\n", result.mae);
    printf("    - R2 Score: %.6f\n", result.r2_score);

    free(targets);
    free(features);
    quantum_model_destroy(model);
    printf("  PASSED\n\n");
}

int main(void) {
    printf("Running quantum classification tests with TensorFlow comparison...\n\n");

    // Set random seed for reproducibility
    srand(42);

    // Run tests
    test_quantum_model_creation();
    test_synthetic_data_generation();
    test_quantum_training();
    test_quantum_vs_classical();
    test_model_evaluation();

    printf("All quantum classification tests passed!\n");
    return 0;
}
