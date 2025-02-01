/**
 * @file test_classical_vs_quantum.c
 * @brief Direct performance comparison between quantum and classical approaches
 */

#include <quantum_geometric/core/quantum_geometric_core.h>
#include <quantum_geometric/learning/quantum_stochastic_sampling.h>
#include <quantum_geometric/hybrid/quantum_machine_learning.h>
#include <quantum_geometric/hybrid/quantum_classical_algorithms.h>
#include <quantum_geometric/hardware/quantum_hardware_abstraction.h>
#include "test_helpers.h"

// Test configurations
#define TEST_NUM_TRAIN_SAMPLES 1000  // Large enough for meaningful comparison
#define TEST_NUM_TEST_SAMPLES 200
#define TEST_INPUT_DIM 4
#define TEST_OUTPUT_DIM 1
#define TEST_QUANTUM_DEPTH 2
#define TEST_BATCH_SIZE 32
#define TEST_NUM_EPOCHS 10  // More epochs for better convergence

// Generate synthetic regression data with non-linear relationships
static void generate_test_data(float* features, float* targets, size_t num_samples) {
    for (size_t i = 0; i < num_samples; i++) {
        // Generate input features
        for (size_t j = 0; j < TEST_INPUT_DIM; j++) {
            features[i * TEST_INPUT_DIM + j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
        
        // Generate target using non-linear function:
        // y = sin(x1) * cos(x2) + x3 * x4 + noise
        float x1 = features[i * TEST_INPUT_DIM];
        float x2 = features[i * TEST_INPUT_DIM + 1];
        float x3 = features[i * TEST_INPUT_DIM + 2];
        float x4 = features[i * TEST_INPUT_DIM + 3];
        
        targets[i] = sinf(x1) * cosf(x2) + x3 * x4 + 0.1f * ((float)rand() / RAND_MAX - 0.5f);
    }
}

void test_quantum_vs_classical() {
    printf("\nRunning quantum vs classical regression comparison...\n");
    
    // Initialize quantum system
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

    // Create quantum model
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

    // Generate datasets
    float* train_features = malloc(TEST_NUM_TRAIN_SAMPLES * TEST_INPUT_DIM * sizeof(float));
    float* train_targets = malloc(TEST_NUM_TRAIN_SAMPLES * sizeof(float));
    float* test_features = malloc(TEST_NUM_TEST_SAMPLES * TEST_INPUT_DIM * sizeof(float));
    float* test_targets = malloc(TEST_NUM_TEST_SAMPLES * sizeof(float));
    
    TEST_ASSERT(train_features && train_targets && test_features && test_targets,
                "Memory allocation failed");
    
    generate_test_data(train_features, train_targets, TEST_NUM_TRAIN_SAMPLES);
    generate_test_data(test_features, test_targets, TEST_NUM_TEST_SAMPLES);

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
    
    training_result_t result = quantum_train(model, train_features, train_targets,
                                           TEST_NUM_TRAIN_SAMPLES, &train_config);
    TEST_ASSERT(result.status == TRAINING_SUCCESS, "Quantum training failed");
    
    quantum_metrics.end_time = get_current_time();
    quantum_metrics.training_time = quantum_metrics.end_time - quantum_metrics.start_time;

    // Evaluate quantum model
    evaluation_result_t quantum_eval = quantum_evaluate(model, test_features, test_targets,
                                                      TEST_NUM_TEST_SAMPLES);
    quantum_metrics.mse = quantum_eval.mse;
    quantum_metrics.mae = quantum_eval.mae;
    quantum_metrics.r2_score = quantum_eval.r2_score;
    quantum_metrics.memory_used = get_peak_memory_usage();

    // Train and evaluate classical model (TensorFlow-like)
    printf("\nTraining classical model...\n");
    performance_metrics_t classical_metrics;
    classical_metrics.start_time = get_current_time();
    
    classical_model_t* classical = classical_model_create(TEST_INPUT_DIM, TEST_OUTPUT_DIM);
    classical_train(classical, train_features, train_targets, TEST_NUM_TRAIN_SAMPLES,
                   TEST_NUM_EPOCHS, TEST_BATCH_SIZE);
    
    classical_metrics.end_time = get_current_time();
    classical_metrics.training_time = classical_metrics.end_time - classical_metrics.start_time;
    
    evaluation_result_t classical_eval = classical_evaluate(classical, test_features,
                                                          test_targets, TEST_NUM_TEST_SAMPLES);
    classical_metrics.mse = classical_eval.mse;
    classical_metrics.mae = classical_eval.mae;
    classical_metrics.r2_score = classical_eval.r2_score;
    classical_metrics.memory_used = get_peak_memory_usage();

    // Print comparison results
    printf("\nPerformance Comparison:\n");
    printf("Quantum Model:\n");
    printf("- MSE: %.6f\n", quantum_metrics.mse);
    printf("- MAE: %.6f\n", quantum_metrics.mae);
    printf("- R² Score: %.6f\n", quantum_metrics.r2_score);
    printf("- Training time: %.2f seconds\n", quantum_metrics.training_time);
    printf("- Memory usage: %.2f MB\n", quantum_metrics.memory_used / 1024.0 / 1024.0);
    
    printf("\nClassical Model (TensorFlow):\n");
    printf("- MSE: %.6f\n", classical_metrics.mse);
    printf("- MAE: %.6f\n", classical_metrics.mae);
    printf("- R² Score: %.6f\n", classical_metrics.r2_score);
    printf("- Training time: %.2f seconds\n", classical_metrics.training_time);
    printf("- Memory usage: %.2f MB\n", classical_metrics.memory_used / 1024.0 / 1024.0);

    // Calculate improvement percentages
    float mse_improvement = ((classical_metrics.mse - quantum_metrics.mse) / classical_metrics.mse) * 100.0f;
    float r2_improvement = ((quantum_metrics.r2_score - classical_metrics.r2_score) / classical_metrics.r2_score) * 100.0f;
    float speed_improvement = ((classical_metrics.training_time - quantum_metrics.training_time) / classical_metrics.training_time) * 100.0f;
    
    printf("\nQuantum Improvements:\n");
    printf("- MSE reduction: %.1f%%\n", mse_improvement);
    printf("- R² score improvement: %.1f%%\n", r2_improvement);
    printf("- Training speedup: %.1f%%\n", speed_improvement);

    // Verify quantum model performs at least as well as classical
    TEST_ASSERT(quantum_metrics.mse <= classical_metrics.mse * 1.1,
                "Quantum MSE significantly worse than classical");
    TEST_ASSERT(quantum_metrics.r2_score >= classical_metrics.r2_score * 0.9,
                "Quantum R² score significantly worse than classical");

    // Cleanup
    free(test_features);
    free(test_targets);
    free(train_features);
    free(train_targets);
    quantum_model_destroy(model);
    classical_model_destroy(classical);
    quantum_system_destroy(system);
    
    printf("\nComparison test completed successfully\n");
}

int main() {
    printf("Running quantum vs classical regression comparison...\n\n");
    
    // Set random seed for reproducibility
    srand(42);
    
    // Run comparison test
    test_quantum_vs_classical();
    
    printf("\nAll tests passed successfully!\n");
    return 0;
}
