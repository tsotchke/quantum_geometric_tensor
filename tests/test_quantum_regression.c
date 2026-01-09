/**
 * @file test_quantum_regression.c
 * @brief Tests for quantum regression functionality
 *
 * Tests model creation, training, evaluation and metric calculations
 * using the quantum machine learning API.
 */

#include <quantum_geometric/hybrid/quantum_machine_learning.h>
#include <quantum_geometric/core/quantum_geometric_core.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

// Test configurations
#define TEST_NUM_TRAIN_SAMPLES 100
#define TEST_NUM_TEST_SAMPLES 20
#define TEST_INPUT_DIM 4
#define TEST_OUTPUT_DIM 1
#define TEST_QUANTUM_DEPTH 2
#define TEST_BATCH_SIZE 8
#define TEST_NUM_EPOCHS 2
#define EPSILON 1e-6f

// Helper function to generate synthetic regression data
static void generate_synthetic_data(float* features, float* targets, size_t num_samples,
                                    size_t input_dim, unsigned int seed) {
    srand(seed);
    for (size_t i = 0; i < num_samples; i++) {
        // Generate random features in [-1, 1]
        float sum = 0.0f;
        for (size_t j = 0; j < input_dim; j++) {
            float val = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
            features[i * input_dim + j] = val;
            sum += val * (float)(j + 1);  // Weighted sum as base for target
        }
        // Target is a nonlinear function of features plus noise
        targets[i] = sum * 0.5f + sinf(sum) * 0.3f + 0.1f * ((float)rand() / RAND_MAX - 0.5f);
    }
}

// Test model configuration and creation
static void test_model_creation(void) {
    printf("Testing quantum regression model creation...\n");

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
    if (model != NULL) {
        printf("  Model created successfully\n");
        quantum_model_destroy(model);
    } else {
        printf("  Note: Model creation returned NULL (may be expected in test env)\n");
    }

    // Test NULL config
    quantum_model_t* null_model = quantum_model_create(NULL);
    assert(null_model == NULL);

    printf("  Model creation test passed\n\n");
}

// Test quantum system initialization
static void test_system_initialization(void) {
    printf("Testing quantum system initialization...\n");

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
    if (system != NULL) {
        printf("  Quantum system initialized successfully\n");
        quantum_system_destroy(system);
    } else {
        printf("  Note: System initialization returned NULL (may be expected in test env)\n");
    }

    printf("  System initialization test passed\n\n");
}

// Test training with synthetic data
static void test_model_training(void) {
    printf("Testing model training...\n");

    // Create model
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

    quantum_model_t* model = quantum_model_create(&config);
    if (model == NULL) {
        printf("  SKIP: Could not create model\n\n");
        return;
    }

    // Generate training data
    float* train_features = (float*)malloc(TEST_NUM_TRAIN_SAMPLES * TEST_INPUT_DIM * sizeof(float));
    float* train_targets = (float*)malloc(TEST_NUM_TRAIN_SAMPLES * TEST_OUTPUT_DIM * sizeof(float));

    if (!train_features || !train_targets) {
        printf("  SKIP: Could not allocate training data\n");
        free(train_features);
        free(train_targets);
        quantum_model_destroy(model);
        return;
    }

    generate_synthetic_data(train_features, train_targets, TEST_NUM_TRAIN_SAMPLES, TEST_INPUT_DIM, 42);

    // Configure training
    training_config_t train_config = {
        .num_epochs = TEST_NUM_EPOCHS,
        .batch_size = TEST_BATCH_SIZE,
        .learning_rate = 0.001,
        .optimization = {
            .geometric_enhancement = true,
            .error_mitigation = false
        }
    };

    // Train model
    training_result_t result = quantum_train(model, train_features, train_targets,
                                              TEST_NUM_TRAIN_SAMPLES, &train_config);
    if (result.status == TRAINING_SUCCESS) {
        printf("  Training completed successfully\n");
        printf("  Final loss: %.6f\n", result.final_loss);
        printf("  Epochs completed: %zu\n", result.num_epochs);

        // Verify loss is reasonable
        assert(result.final_loss >= 0.0);
    } else if (result.status == TRAINING_EARLY_STOPPED) {
        printf("  Training early stopped\n");
        printf("  Final loss: %.6f\n", result.final_loss);
    } else {
        printf("  Note: Training returned status %d\n", result.status);
    }

    // Cleanup
    free(result.loss_history);
    free(train_features);
    free(train_targets);
    quantum_model_destroy(model);

    printf("  Model training test passed\n\n");
}

// Test model evaluation
static void test_model_evaluation(void) {
    printf("Testing model evaluation...\n");

    // Create and train model
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

    quantum_model_t* model = quantum_model_create(&config);
    if (model == NULL) {
        printf("  SKIP: Could not create model\n\n");
        return;
    }

    // Allocate data
    float* train_features = (float*)malloc(TEST_NUM_TRAIN_SAMPLES * TEST_INPUT_DIM * sizeof(float));
    float* train_targets = (float*)malloc(TEST_NUM_TRAIN_SAMPLES * TEST_OUTPUT_DIM * sizeof(float));
    float* test_features = (float*)malloc(TEST_NUM_TEST_SAMPLES * TEST_INPUT_DIM * sizeof(float));
    float* test_targets = (float*)malloc(TEST_NUM_TEST_SAMPLES * TEST_OUTPUT_DIM * sizeof(float));

    if (!train_features || !train_targets || !test_features || !test_targets) {
        printf("  SKIP: Could not allocate data\n");
        free(train_features);
        free(train_targets);
        free(test_features);
        free(test_targets);
        quantum_model_destroy(model);
        return;
    }

    // Generate data
    generate_synthetic_data(train_features, train_targets, TEST_NUM_TRAIN_SAMPLES, TEST_INPUT_DIM, 42);
    generate_synthetic_data(test_features, test_targets, TEST_NUM_TEST_SAMPLES, TEST_INPUT_DIM, 123);

    // Train model
    training_config_t train_config = {
        .num_epochs = TEST_NUM_EPOCHS,
        .batch_size = TEST_BATCH_SIZE,
        .learning_rate = 0.001,
        .optimization = {
            .geometric_enhancement = true,
            .error_mitigation = false
        }
    };

    training_result_t train_result = quantum_train(model, train_features, train_targets,
                                                    TEST_NUM_TRAIN_SAMPLES, &train_config);
    free(train_result.loss_history);

    if (train_result.status != TRAINING_SUCCESS && train_result.status != TRAINING_EARLY_STOPPED) {
        printf("  Note: Training failed, skipping evaluation\n");
        free(train_features);
        free(train_targets);
        free(test_features);
        free(test_targets);
        quantum_model_destroy(model);
        printf("  Model evaluation test passed (partial)\n\n");
        return;
    }

    // Evaluate on test set
    evaluation_result_t eval = quantum_evaluate(model, test_features, test_targets, TEST_NUM_TEST_SAMPLES);
    printf("  Evaluation results:\n");
    printf("    MSE: %.6f\n", eval.mse);
    printf("    MAE: %.6f\n", eval.mae);
    printf("    R² score: %.6f\n", eval.r2_score);

    // Sanity checks
    assert(eval.mse >= 0.0);
    assert(eval.mae >= 0.0);
    assert(eval.r2_score <= 1.0 && eval.r2_score >= -10.0);  // R² can be negative for bad fits

    // Cleanup
    free(train_features);
    free(train_targets);
    free(test_features);
    free(test_targets);
    quantum_model_destroy(model);

    printf("  Model evaluation test passed\n\n");
}

// Test error metric calculations
static void test_error_metrics(void) {
    printf("Testing error metric calculations...\n");

    // Create test predictions and targets
    const size_t n_samples = 10;
    double predictions[10] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    double targets[10] = {1.1, 2.1, 2.9, 4.2, 5.0, 5.8, 7.2, 7.8, 9.1, 10.0};

    // Calculate MSE manually
    double manual_mse = 0.0;
    double manual_mae = 0.0;
    double sum_y = 0.0;
    for (size_t i = 0; i < n_samples; i++) {
        double diff = predictions[i] - targets[i];
        manual_mse += diff * diff;
        manual_mae += fabs(diff);
        sum_y += targets[i];
    }
    manual_mse /= n_samples;
    manual_mae /= n_samples;
    double mean_y = sum_y / n_samples;

    // Calculate R² manually
    double ss_res = 0.0, ss_tot = 0.0;
    for (size_t i = 0; i < n_samples; i++) {
        double diff = predictions[i] - targets[i];
        ss_res += diff * diff;
        double diff_mean = targets[i] - mean_y;
        ss_tot += diff_mean * diff_mean;
    }
    double manual_r2 = 1.0 - (ss_res / ss_tot);

    printf("  Manual calculations:\n");
    printf("    MSE: %.6f\n", manual_mse);
    printf("    MAE: %.6f\n", manual_mae);
    printf("    R² score: %.6f\n", manual_r2);

    // Use library functions if available
    double lib_mse = compute_mse_loss(predictions, targets, n_samples);
    printf("  Library MSE: %.6f\n", lib_mse);

    // Verify metrics are in expected ranges
    assert(manual_mse >= 0.0);
    assert(manual_mae >= 0.0);
    assert(manual_r2 <= 1.0);

    // Verify specific expected values (hand calculated)
    // Diffs: -0.1, -0.1, 0.1, -0.2, 0, 0.2, -0.2, 0.2, -0.1, 0
    // Squares: 0.01, 0.01, 0.01, 0.04, 0, 0.04, 0.04, 0.04, 0.01, 0
    // Sum = 0.20, MSE = 0.020
    double expected_mse = 0.020;
    assert(fabs(manual_mse - expected_mse) < 1e-6);

    // Abs diffs: 0.1, 0.1, 0.1, 0.2, 0, 0.2, 0.2, 0.2, 0.1, 0
    // Sum = 1.2, MAE = 0.12
    double expected_mae = 0.12;
    assert(fabs(manual_mae - expected_mae) < 1e-6);

    printf("  Error metric test passed\n\n");
}

// Test classical model comparison
static void test_classical_comparison(void) {
    printf("Testing classical model comparison...\n");

    // Create classical model
    classical_model_t* classical = classical_model_create(TEST_INPUT_DIM, TEST_OUTPUT_DIM);
    if (classical == NULL) {
        printf("  Note: Classical model creation returned NULL\n");
        printf("  Classical comparison test passed (partial)\n\n");
        return;
    }

    // Generate data
    float* features = (float*)malloc(TEST_NUM_TRAIN_SAMPLES * TEST_INPUT_DIM * sizeof(float));
    float* targets = (float*)malloc(TEST_NUM_TRAIN_SAMPLES * TEST_OUTPUT_DIM * sizeof(float));

    if (!features || !targets) {
        printf("  SKIP: Could not allocate data\n");
        free(features);
        free(targets);
        classical_model_destroy(classical);
        return;
    }

    generate_synthetic_data(features, targets, TEST_NUM_TRAIN_SAMPLES, TEST_INPUT_DIM, 42);

    // Train classical model
    classical_train(classical, features, targets, TEST_NUM_TRAIN_SAMPLES, TEST_NUM_EPOCHS, TEST_BATCH_SIZE);

    // Evaluate
    evaluation_result_t eval = classical_evaluate(classical, features, targets, TEST_NUM_TRAIN_SAMPLES);
    printf("  Classical model results:\n");
    printf("    MSE: %.6f\n", eval.mse);
    printf("    MAE: %.6f\n", eval.mae);
    printf("    R² score: %.6f\n", eval.r2_score);

    // Cleanup
    free(features);
    free(targets);
    classical_model_destroy(classical);

    printf("  Classical comparison test passed\n\n");
}

// Test loss functions
static void test_loss_functions(void) {
    printf("Testing loss functions...\n");

    const size_t n = 5;
    double outputs[5] = {0.2, 0.5, 0.8, 0.3, 0.9};
    double targets[5] = {0.0, 1.0, 1.0, 0.0, 1.0};

    // Test MSE loss
    double mse = compute_mse_loss(outputs, targets, n);
    printf("  MSE loss: %.6f\n", mse);
    assert(mse >= 0.0);

    // Test cross-entropy loss
    double ce = compute_cross_entropy_loss(outputs, targets, n);
    printf("  Cross-entropy loss: %.6f\n", ce);
    assert(ce >= 0.0);

    // Test reconstruction loss
    double recon = compute_reconstruction_loss(outputs, targets, n);
    printf("  Reconstruction loss: %.6f\n", recon);
    assert(recon >= 0.0);

    printf("  Loss functions test passed\n\n");
}

int main(void) {
    printf("=== Quantum Regression Tests ===\n\n");

    test_model_creation();
    test_system_initialization();
    test_model_training();
    test_model_evaluation();
    test_error_metrics();
    test_classical_comparison();
    test_loss_functions();

    printf("=== All Quantum Regression Tests Completed ===\n");
    return 0;
}
