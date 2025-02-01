/**
 * Test suite for MNIST quantum learning pipeline
 * 
 * Tests:
 * 1. Data loading and preprocessing
 * 2. Pipeline initialization
 * 3. Training process
 * 4. Evaluation metrics
 * 5. Performance comparison
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/learning/quantum_pipeline.h"
#include "quantum_geometric/learning/quantum_stochastic_sampling.h"
#include "quantum_geometric/hybrid/quantum_machine_learning.h"
#include "quantum_geometric/hybrid/quantum_classical_algorithms.h"

// Test dataset parameters (using smaller subset for testing)
#define TEST_SAMPLES 1000
#define TEST_PIXELS (28 * 28)
#define TEST_CLASSES 10
#define TEST_LATENT_DIM 16
#define TEST_CLUSTERS 8

// Generate synthetic MNIST-like data for testing
static void generate_test_data(float* images, int* labels, size_t num_samples) {
    for (size_t i = 0; i < num_samples; i++) {
        // Generate digit pattern (0-9)
        int digit = i % TEST_CLASSES;
        labels[i] = digit;
        
        // Clear image
        memset(&images[i * TEST_PIXELS], 0, TEST_PIXELS * sizeof(float));
        
        // Generate simple digit pattern
        for (size_t j = 0; j < TEST_PIXELS; j++) {
            // Create distinctive patterns for each digit
            double x = (j % 28) / 27.0;
            double y = (j / 28) / 27.0;
            
            switch (digit) {
                case 0:
                    images[i * TEST_PIXELS + j] = (sqrt(pow(x-0.5, 2) + pow(y-0.5, 2)) < 0.3) ? 1.0f : 0.0f;
                    break;
                case 1:
                    images[i * TEST_PIXELS + j] = (fabs(x-0.5) < 0.1) ? 1.0f : 0.0f;
                    break;
                case 2:
                    images[i * TEST_PIXELS + j] = (y > 0.8*x) ? 1.0f : 0.0f;
                    break;
                default:
                    images[i * TEST_PIXELS + j] = (float)(sin(digit * x * M_PI) * cos(digit * y * M_PI) * 0.5 + 0.5);
            }
            
            // Add noise
            images[i * TEST_PIXELS + j] += 0.1f * ((float)rand() / RAND_MAX - 0.5f);
            images[i * TEST_PIXELS + j] = fmaxf(0.0f, fminf(1.0f, images[i * TEST_PIXELS + j]));
        }
    }
}

// Test data loading and preprocessing
static void test_data_loading(void) {
    printf("Testing data loading and preprocessing...\n");
    
    // Allocate test data
    float* images = malloc(TEST_SAMPLES * TEST_PIXELS * sizeof(float));
    int* labels = malloc(TEST_SAMPLES * sizeof(int));
    assert(images != NULL && labels != NULL);
    
    // Generate test data
    generate_test_data(images, labels, TEST_SAMPLES);
    
    // Verify data properties
    for (size_t i = 0; i < TEST_SAMPLES; i++) {
        // Check label range
        assert(labels[i] >= 0 && labels[i] < TEST_CLASSES);
        
        // Check image values
        for (size_t j = 0; j < TEST_PIXELS; j++) {
            assert(images[i * TEST_PIXELS + j] >= 0.0f && images[i * TEST_PIXELS + j] <= 1.0f);
        }
    }
    
    // Cleanup
    free(images);
    free(labels);
    
    printf("Data loading test passed\n");
}

// Test pipeline initialization
static void test_pipeline_init(void) {
    printf("Testing pipeline initialization...\n");
    
    // Configure pipeline
    float config[] = {
        (float)TEST_PIXELS,     // input_dim
        (float)TEST_LATENT_DIM, // latent_dim
        (float)TEST_CLUSTERS,   // num_clusters
        (float)TEST_CLASSES,    // num_classes
        32.0f,                  // batch_size
        0.001f,                 // learning_rate
        1.0f                    // use_gpu
    };
    
    // Initialize pipeline
    quantum_pipeline_handle_t pipeline = quantum_pipeline_create(config);
    assert(pipeline != NULL && "Pipeline creation failed");
    
    quantum_pipeline_destroy(pipeline);
    printf("Pipeline initialization test passed\n");
}

// Test training process
static void test_training(void) {
    printf("Testing training process...\n");
    
    // Create test data
    float* images = malloc(TEST_SAMPLES * TEST_PIXELS * sizeof(float));
    int* labels = malloc(TEST_SAMPLES * sizeof(int));
    assert(images != NULL && labels != NULL);
    
    generate_test_data(images, labels, TEST_SAMPLES);
    
    // Initialize pipeline
    float config[] = {
        (float)TEST_PIXELS,     // input_dim
        (float)TEST_LATENT_DIM, // latent_dim
        (float)TEST_CLUSTERS,   // num_clusters
        (float)TEST_CLASSES,    // num_classes
        32.0f,                  // batch_size
        0.001f,                 // learning_rate
        1.0f                    // use_gpu
    };
    
    quantum_pipeline_handle_t pipeline = quantum_pipeline_create(config);
    assert(pipeline != NULL && "Pipeline creation failed");
    
    // Train pipeline
    int result = quantum_pipeline_train(pipeline, images, labels, TEST_SAMPLES);
    assert(result == QG_SUCCESS && "Training failed");
    
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
    float* train_images = malloc(TEST_SAMPLES * TEST_PIXELS * sizeof(float));
    int* train_labels = malloc(TEST_SAMPLES * sizeof(int));
    float* test_images = malloc(TEST_SAMPLES * TEST_PIXELS * sizeof(float));
    int* test_labels = malloc(TEST_SAMPLES * sizeof(int));
    assert(train_images != NULL && train_labels != NULL && 
           test_images != NULL && test_labels != NULL);
    
    generate_test_data(train_images, train_labels, TEST_SAMPLES);
    generate_test_data(test_images, test_labels, TEST_SAMPLES);
    
    // Initialize pipeline
    float config[] = {
        (float)TEST_PIXELS,     // input_dim
        (float)TEST_LATENT_DIM, // latent_dim
        (float)TEST_CLUSTERS,   // num_clusters
        (float)TEST_CLASSES,    // num_classes
        32.0f,                  // batch_size
        0.001f,                 // learning_rate
        1.0f                    // use_gpu
    };
    
    quantum_pipeline_handle_t pipeline = quantum_pipeline_create(config);
    assert(pipeline != NULL && "Pipeline creation failed");
    
    // Train pipeline
    int result = quantum_pipeline_train(pipeline, train_images, train_labels, TEST_SAMPLES);
    assert(result == QG_SUCCESS && "Training failed");
    
    // Evaluate on test set
    float results[3];  // accuracy, time, memory
    result = quantum_pipeline_evaluate(pipeline, test_images, test_labels, TEST_SAMPLES, results);
    assert(result == QG_SUCCESS && "Evaluation failed");
    
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
    float* images = malloc(TEST_SAMPLES * TEST_PIXELS * sizeof(float));
    int* labels = malloc(TEST_SAMPLES * sizeof(int));
    assert(images != NULL && labels != NULL);
    
    generate_test_data(images, labels, TEST_SAMPLES);
    
    // Initialize pipeline
    float config[] = {
        (float)TEST_PIXELS,     // input_dim
        (float)TEST_LATENT_DIM, // latent_dim
        (float)TEST_CLUSTERS,   // num_clusters
        (float)TEST_CLASSES,    // num_classes
        32.0f,                  // batch_size
        0.001f,                 // learning_rate
        1.0f                    // use_gpu
    };
    
    quantum_pipeline_handle_t pipeline = quantum_pipeline_create(config);
    assert(pipeline != NULL && "Pipeline creation failed");
    
    // Train and evaluate
    int result = quantum_pipeline_train(pipeline, images, labels, TEST_SAMPLES);
    assert(result == QG_SUCCESS && "Training failed");
    
    float results[3];  // accuracy, time, memory
    result = quantum_pipeline_evaluate(pipeline, images, labels, TEST_SAMPLES, results);
    assert(result == QG_SUCCESS && "Evaluation failed");
    
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
    printf("Running MNIST quantum learning pipeline tests...\n\n");
    
    // Set random seed for reproducibility
    srand(42);
    
    // Run tests
    test_data_loading();
    test_pipeline_init();
    test_training();
    test_evaluation();
    test_performance_comparison();
    
    printf("\nAll tests passed successfully!\n");
    return QG_SUCCESS;
}
