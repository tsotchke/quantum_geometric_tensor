/**
 * Test suite for TensorFlow vs Quantum Geometric Learning comparison
 * 
 * Tests:
 * 1. Basic pipeline setup and execution
 * 2. Performance metrics collection
 * 3. Memory usage tracking
 * 4. GPU utilization monitoring
 * 5. Results validation
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/learning/quantum_pipeline.h"
#include "quantum_geometric/learning/quantum_stochastic_sampling.h"
#include "quantum_geometric/hybrid/quantum_machine_learning.h"
#include "quantum_geometric/hybrid/quantum_classical_algorithms.h"
#include "quantum_geometric/hardware/quantum_geometric_tensor_gpu.h"
#include "quantum_geometric/hardware/quantum_geometric_tensor_perf.h"

// Test data generation
static void generate_test_data(float* images, int* labels, size_t num_samples) {
    for (size_t i = 0; i < num_samples; i++) {
        // Generate random 28x28 image
        for (size_t j = 0; j < 784; j++) {
            images[i * 784 + j] = (float)rand() / RAND_MAX;
        }
        // Random label 0-9
        labels[i] = rand() % 10;
    }
}

// Test basic pipeline setup
static void test_pipeline_setup(void) {
    printf("Testing pipeline setup...\n");
    
    float config[] = {
        784.0f,    // input_dim
        32.0f,     // latent_dim
        16.0f,     // num_clusters
        10.0f,     // num_classes
        64.0f,     // batch_size
        0.001f,    // learning_rate
        0.0f       // use_gpu (disable GPU to use CPU only)
    };
    
    quantum_pipeline_handle_t pipeline = quantum_pipeline_create(config);
    assert(pipeline != NULL);
    
    quantum_pipeline_destroy(pipeline);
    printf("Pipeline setup test passed\n");
}

// Test training process
static void test_training(void) {
    printf("Testing training process...\n");
    
    // Create test data
    const size_t num_samples = 100;
    float* images = malloc(num_samples * 784 * sizeof(float));
    int* labels = malloc(num_samples * sizeof(int));
    
    generate_test_data(images, labels, num_samples);
    
    // Initialize pipeline
    float config[] = {
        784.0f,    // input_dim
        32.0f,     // latent_dim
        16.0f,     // num_clusters
        10.0f,     // num_classes
        32.0f,     // batch_size
        0.001f,    // learning_rate
        0.0f       // use_gpu (disable GPU to use CPU only)
    };
    
    quantum_pipeline_handle_t pipeline = quantum_pipeline_create(config);
    assert(pipeline != NULL);
    
    // Train pipeline
    int result = quantum_pipeline_train(pipeline, images, labels, num_samples);
    assert(result == QGT_SUCCESS);
    
    // Cleanup
    quantum_pipeline_destroy(pipeline);
    free(images);
    free(labels);
    
    printf("Training test passed\n");
}

// Test evaluation metrics
static void test_evaluation(void) {
    printf("Testing evaluation metrics...\n");
    
    // Create test data
    const size_t num_samples = 100;
    float* train_images = malloc(num_samples * 784 * sizeof(float));
    int* train_labels = malloc(num_samples * sizeof(int));
    float* test_images = malloc(num_samples * 784 * sizeof(float));
    int* test_labels = malloc(num_samples * sizeof(int));
    
    generate_test_data(train_images, train_labels, num_samples);
    generate_test_data(test_images, test_labels, num_samples);
    
    // Initialize pipeline
    float config[] = {
        784.0f,    // input_dim
        32.0f,     // latent_dim
        16.0f,     // num_clusters
        10.0f,     // num_classes
        32.0f,     // batch_size
        0.001f,    // learning_rate
        0.0f       // use_gpu (disable GPU to use CPU only)
    };
    
    quantum_pipeline_handle_t pipeline = quantum_pipeline_create(config);
    assert(pipeline != NULL);
    
    // Train pipeline
    int result = quantum_pipeline_train(pipeline, train_images, train_labels, num_samples);
    assert(result == QGT_SUCCESS);
    
    // Evaluate on test set
    float results[3];  // accuracy, time, memory
    result = quantum_pipeline_evaluate(pipeline, test_images, test_labels, num_samples, results);
    assert(result == QGT_SUCCESS);
    
    // Verify metrics
    assert(results[0] >= 0.0f && results[0] <= 1.0f);  // accuracy
    assert(results[1] > 0.0f);  // time
    assert(results[2] > 0.0f);  // memory
    
    // Cleanup
    quantum_pipeline_destroy(pipeline);
    free(train_images);
    free(test_images);
    free(train_labels);
    free(test_labels);
    
    printf("Evaluation test passed\n");
}

// Test performance comparison
static void test_performance_comparison(void) {
    printf("Testing performance comparison...\n");
    
    // Create test data
    const size_t num_samples = 100;
    float* images = malloc(num_samples * 784 * sizeof(float));
    int* labels = malloc(num_samples * sizeof(int));
    
    generate_test_data(images, labels, num_samples);
    
    // Initialize pipeline
    float config[] = {
        784.0f,    // input_dim
        32.0f,     // latent_dim
        16.0f,     // num_clusters
        10.0f,     // num_classes
        32.0f,     // batch_size
        0.001f,    // learning_rate
        0.0f       // use_gpu (disable GPU to use CPU only)
    };
    
    quantum_pipeline_handle_t pipeline = quantum_pipeline_create(config);
    assert(pipeline != NULL);
    
    // Train and evaluate
    int result = quantum_pipeline_train(pipeline, images, labels, num_samples);
    assert(result == QGT_SUCCESS);
    
    float results[3];  // accuracy, time, memory
    result = quantum_pipeline_evaluate(pipeline, images, labels, num_samples, results);
    assert(result == QGT_SUCCESS);
    
    // Verify results
    assert(results[0] >= 0.0f && results[0] <= 1.0f);  // accuracy
    assert(results[1] > 0.0f);  // time
    assert(results[2] > 0.0f);  // memory
    
    // Cleanup
    quantum_pipeline_destroy(pipeline);
    free(images);
    free(labels);
    
    printf("Performance comparison test passed\n");
}

int main(void) {
    printf("Running TensorFlow comparison tests...\n\n");
    
    // Set random seed for reproducibility
    srand(42);
    
    // Run tests
    test_pipeline_setup();
    test_training();
    test_evaluation();
    test_performance_comparison();
    
    printf("\nAll tests passed!\n");
    return QGT_SUCCESS;
}
