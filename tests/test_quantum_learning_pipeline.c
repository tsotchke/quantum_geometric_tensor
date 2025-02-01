/**
 * Test suite for quantum learning pipeline
 * 
 * Tests:
 * 1. Pipeline initialization
 * 2. Data processing through each stage
 * 3. Integration between components
 * 4. End-to-end performance
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/core/error_codes.h"
#include "quantum_geometric/learning/quantum_pipeline.h"
#include "quantum_geometric/learning/quantum_stochastic_sampling.h"
#include "quantum_geometric/hybrid/quantum_machine_learning.h"
#include "quantum_geometric/hybrid/quantum_classical_algorithms.h"

// Pipeline configuration struct for better readability
typedef struct {
    size_t input_dim;
    size_t latent_dim;
    size_t num_clusters;
    size_t num_classes;
    size_t batch_size;
    float learning_rate;
    bool use_gpu;
} pipeline_config_t;

// Helper function to convert pipeline_config_t to float array
static void config_to_array(const pipeline_config_t* config, float* array) {
    array[0] = (float)config->input_dim;
    array[1] = (float)config->latent_dim;
    array[2] = (float)config->num_clusters;
    array[3] = (float)config->num_classes;
    array[4] = (float)config->batch_size;
    array[5] = config->learning_rate;
    array[6] = config->use_gpu ? 1.0f : 0.0f;
}

// Generate synthetic test data with known patterns
static void generate_test_data(double** data, int* labels, size_t num_samples, size_t input_dim) {
    const size_t num_classes = 2;
    const size_t points_per_class = num_samples / num_classes;
    
    // Generate two interleaved spirals
    for (size_t i = 0; i < num_samples; i++) {
        size_t class = i / points_per_class;
        double t = ((double)(i % points_per_class) / points_per_class) * 4 * M_PI;
        
        // Base spiral pattern
        double r = t / (4 * M_PI);
        double x = r * cos(t + class * M_PI);
        double y = r * sin(t + class * M_PI);
        
        // Embed in higher dimensions with structure
        for (size_t j = 0; j < input_dim; j++) {
            if (j == 0) {
                data[i][j] = x;
            } else if (j == 1) {
                data[i][j] = y;
            } else {
                // Add structured noise in higher dimensions
                data[i][j] = 0.1 * sin(x * j) * cos(y * (j-1));
            }
        }
        
        labels[i] = class;
    }
}

// Test pipeline initialization
static void test_pipeline_init(void) {
    printf("Testing pipeline initialization...\n");
    
    pipeline_config_t config = {
        .input_dim = 8,
        .latent_dim = 2,
        .num_clusters = 3,
        .num_classes = 2,
        .batch_size = 32,
        .learning_rate = 0.01f,
        .use_gpu = true
    };
    
    float config_array[7];
    config_to_array(&config, config_array);
    
    quantum_pipeline_handle_t pipeline = quantum_pipeline_create(config_array);
    assert(pipeline != NULL && "Pipeline creation failed");
    
    quantum_pipeline_destroy(pipeline);
    printf("Pipeline initialization test passed\n");
}

// Test data processing through pipeline stages
static void test_pipeline_stages(void) {
    printf("Testing pipeline stages...\n");
    
    const size_t num_samples = 100;
    const size_t input_dim = 8;
    
    // Create test data
    double** data = malloc(num_samples * sizeof(double*));
    int* labels = malloc(num_samples * sizeof(int));
    
    for (size_t i = 0; i < num_samples; i++) {
        data[i] = malloc(input_dim * sizeof(double));
    }
    
    generate_test_data(data, labels, num_samples, input_dim);
    
    // Initialize pipeline
    pipeline_config_t config = {
        .input_dim = input_dim,
        .latent_dim = 2,
        .num_clusters = 3,
        .num_classes = 2,
        .batch_size = 32,
        .learning_rate = 0.01f,
        .use_gpu = true
    };
    
    float config_array[7];
    config_to_array(&config, config_array);
    
    quantum_pipeline_handle_t pipeline = quantum_pipeline_create(config_array);
    assert(pipeline != NULL && "Pipeline creation failed");
    
    // Convert data to float array
    float* data_flat = malloc(num_samples * input_dim * sizeof(float));
    for (size_t i = 0; i < num_samples; i++) {
        for (size_t j = 0; j < input_dim; j++) {
            data_flat[i * input_dim + j] = (float)data[i][j];
        }
    }
    
    // Train pipeline
    int result = quantum_pipeline_train(pipeline, data_flat, labels, num_samples);
    assert(result == QGT_SUCCESS && "Training failed");
    
    // Evaluate results
    float metrics[3];  // accuracy, time, memory
    result = quantum_pipeline_evaluate(pipeline, data_flat, labels, num_samples, metrics);
    assert(result == QGT_SUCCESS && "Evaluation failed");
    
    // Verify metrics
    assert(metrics[0] >= 0.8f && "Poor accuracy");  // accuracy
    assert(metrics[1] > 0.0f && "Invalid time");    // time
    assert(metrics[2] > 0.0f && "Invalid memory");  // memory
    
    // Cleanup
    quantum_pipeline_destroy(pipeline);
    free(data_flat);
    for (size_t i = 0; i < num_samples; i++) {
        free(data[i]);
    }
    free(data);
    free(labels);
    
    printf("Pipeline stages test passed\n");
}

// Test end-to-end pipeline performance
static void test_pipeline_performance(void) {
    printf("Testing end-to-end pipeline performance...\n");
    
    const size_t num_train = 1000;
    const size_t num_test = 200;
    const size_t input_dim = 16;
    
    // Create datasets
    double** train_data = malloc(num_train * sizeof(double*));
    int* train_labels = malloc(num_train * sizeof(int));
    
    double** test_data = malloc(num_test * sizeof(double*));
    int* test_labels = malloc(num_test * sizeof(int));
    
    for (size_t i = 0; i < num_train; i++) {
        train_data[i] = malloc(input_dim * sizeof(double));
    }
    for (size_t i = 0; i < num_test; i++) {
        test_data[i] = malloc(input_dim * sizeof(double));
    }
    
    generate_test_data(train_data, train_labels, num_train, input_dim);
    generate_test_data(test_data, test_labels, num_test, input_dim);
    
    // Initialize pipeline
    pipeline_config_t config = {
        .input_dim = input_dim,
        .latent_dim = 4,
        .num_clusters = 3,
        .num_classes = 2,
        .batch_size = 32,
        .learning_rate = 0.01f,
        .use_gpu = true
    };
    
    float config_array[7];
    config_to_array(&config, config_array);
    
    quantum_pipeline_handle_t pipeline = quantum_pipeline_create(config_array);
    assert(pipeline != NULL && "Pipeline creation failed");
    
    // Convert data to float arrays
    float* train_data_flat = malloc(num_train * input_dim * sizeof(float));
    float* test_data_flat = malloc(num_test * input_dim * sizeof(float));
    
    for (size_t i = 0; i < num_train; i++) {
        for (size_t j = 0; j < input_dim; j++) {
            train_data_flat[i * input_dim + j] = (float)train_data[i][j];
        }
    }
    for (size_t i = 0; i < num_test; i++) {
        for (size_t j = 0; j < input_dim; j++) {
            test_data_flat[i * input_dim + j] = (float)test_data[i][j];
        }
    }
    
    // Train pipeline
    int result = quantum_pipeline_train(pipeline, train_data_flat, train_labels, num_train);
    assert(result == QGT_SUCCESS && "Training failed");
    
    // Evaluate on test set
    float metrics[3];  // accuracy, time, memory
    result = quantum_pipeline_evaluate(pipeline, test_data_flat, test_labels, num_test, metrics);
    assert(result == QGT_SUCCESS && "Evaluation failed");
    
    // Verify metrics
    assert(metrics[0] >= 0.8f && "Poor test accuracy");
    assert(metrics[1] > 0.0f && "Invalid time");
    assert(metrics[2] > 0.0f && "Invalid memory");
    
    // Compare with classical pipeline
    float classical_results[3];  // accuracy, time, memory
    result = quantum_pipeline_compare_classical(pipeline, test_data_flat, test_labels,
                                             num_test, classical_results);
    assert(result == QGT_SUCCESS && "Classical comparison failed");
    
    // Verify quantum advantage
    assert(metrics[0] >= classical_results[0] && "No accuracy advantage");
    assert(metrics[1] <= classical_results[1] && "No speed advantage");
    
    // Cleanup
    quantum_pipeline_destroy(pipeline);
    free(train_data_flat);
    free(test_data_flat);
    for (size_t i = 0; i < num_train; i++) {
        free(train_data[i]);
    }
    for (size_t i = 0; i < num_test; i++) {
        free(test_data[i]);
    }
    free(train_data);
    free(test_data);
    free(train_labels);
    free(test_labels);
    
    printf("Pipeline performance test passed\n");
}

// Test pipeline integration
static void test_pipeline_integration(void) {
    printf("Testing pipeline integration...\n");
    
    const size_t num_samples = 100;
    const size_t input_dim = 8;
    
    // Create test data
    double** data = malloc(num_samples * sizeof(double*));
    int* labels = malloc(num_samples * sizeof(int));
    
    for (size_t i = 0; i < num_samples; i++) {
        data[i] = malloc(input_dim * sizeof(double));
    }
    
    generate_test_data(data, labels, num_samples, input_dim);
    
    // Initialize pipeline
    pipeline_config_t config = {
        .input_dim = input_dim,
        .latent_dim = 2,
        .num_clusters = 3,
        .num_classes = 2,
        .batch_size = 32,
        .learning_rate = 0.01f,
        .use_gpu = true
    };
    
    float config_array[7];
    config_to_array(&config, config_array);
    
    quantum_pipeline_handle_t pipeline = quantum_pipeline_create(config_array);
    assert(pipeline != NULL && "Pipeline creation failed");
    
    // Convert data to float array
    float* data_flat = malloc(num_samples * input_dim * sizeof(float));
    for (size_t i = 0; i < num_samples; i++) {
        for (size_t j = 0; j < input_dim; j++) {
            data_flat[i * input_dim + j] = (float)data[i][j];
        }
    }
    
    // Test error handling
    int result = quantum_pipeline_train(pipeline, NULL, labels, num_samples);
    assert(result != QGT_SUCCESS && "Failed to catch null input");
    
    result = quantum_pipeline_train(pipeline, data_flat, NULL, num_samples);
    assert(result != QGT_SUCCESS && "Failed to catch null labels");
    
    result = quantum_pipeline_train(pipeline, data_flat, labels, 0);
    assert(result != QGT_SUCCESS && "Failed to catch zero samples");
    
    // Test valid training
    result = quantum_pipeline_train(pipeline, data_flat, labels, num_samples);
    assert(result == QGT_SUCCESS && "Valid training failed");
    
    // Test prediction
    int prediction;
    result = quantum_pipeline_predict(pipeline, data_flat, &prediction);
    assert(result == QGT_SUCCESS && "Prediction failed");
    assert(prediction >= 0 && prediction < 2 && "Invalid prediction");
    
    // Cleanup
    quantum_pipeline_destroy(pipeline);
    free(data_flat);
    for (size_t i = 0; i < num_samples; i++) {
        free(data[i]);
    }
    free(data);
    free(labels);
    
    printf("Pipeline integration test passed\n");
}

int main(void) {
    printf("Running quantum learning pipeline tests...\n\n");
    
    // Set random seed for reproducibility
    srand(42);
    
    // Run tests
    test_pipeline_init();
    test_pipeline_stages();
    test_pipeline_performance();
    test_pipeline_integration();
    
    printf("\nAll tests passed successfully!\n");
    return QGT_SUCCESS;
}
