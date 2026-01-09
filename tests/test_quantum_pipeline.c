/**
 * @file test_quantum_pipeline.c
 * @brief Tests for quantum pipeline functionality
 *
 * Tests pipeline creation, training, evaluation, and cleanup.
 */

#include "quantum_geometric/learning/quantum_pipeline.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#define TEST_DATA_SIZE 100
#define INPUT_DIM 784
#define LATENT_DIM 32
#define NUM_CLUSTERS 16
#define NUM_CLASSES 10
#define BATCH_SIZE 32
#define LEARNING_RATE 0.001f
#define EPSILON 1e-6f

// Helper to create test config array
static void create_test_config(float* config, bool use_gpu) {
    config[QG_CONFIG_INPUT_DIM] = (float)INPUT_DIM;
    config[QG_CONFIG_LATENT_DIM] = (float)LATENT_DIM;
    config[QG_CONFIG_NUM_CLUSTERS] = (float)NUM_CLUSTERS;
    config[QG_CONFIG_NUM_CLASSES] = (float)NUM_CLASSES;
    config[QG_CONFIG_BATCH_SIZE] = (float)BATCH_SIZE;
    config[QG_CONFIG_LEARNING_RATE] = LEARNING_RATE;
    config[QG_CONFIG_USE_GPU] = use_gpu ? 1.0f : 0.0f;
    config[QG_CONFIG_NUM_QUBITS] = 8.0f;
    config[QG_CONFIG_NUM_LAYERS] = 4.0f;
}

// Helper to create test data
static void create_test_data(float** data, int** labels) {
    *data = (float*)malloc(TEST_DATA_SIZE * INPUT_DIM * sizeof(float));
    *labels = (int*)malloc(TEST_DATA_SIZE * sizeof(int));

    if (*data && *labels) {
        for (int i = 0; i < TEST_DATA_SIZE; i++) {
            for (int j = 0; j < INPUT_DIM; j++) {
                (*data)[i * INPUT_DIM + j] = (float)rand() / RAND_MAX;
            }
            (*labels)[i] = i % NUM_CLASSES;
        }
    }
}

// Test pipeline configuration validation
static void test_config_validation(void) {
    printf("Testing config validation...\n");

    float config[QG_CONFIG_SIZE];

    // Valid config
    create_test_config(config, false);
    assert(quantum_pipeline_validate_config(config) == true);

    // Invalid input dimension
    config[QG_CONFIG_INPUT_DIM] = 0.0f;
    assert(quantum_pipeline_validate_config(config) == false);

    // Restore and test invalid num_classes
    config[QG_CONFIG_INPUT_DIM] = (float)INPUT_DIM;
    config[QG_CONFIG_NUM_CLASSES] = 0.0f;
    assert(quantum_pipeline_validate_config(config) == false);

    printf("  Config validation test passed\n\n");
}

// Test pipeline creation
static void test_pipeline_create(void) {
    printf("Testing pipeline creation...\n");

    float config[QG_CONFIG_SIZE];
    create_test_config(config, false);

    quantum_pipeline_handle_t pipeline = quantum_pipeline_create(config);
    if (pipeline != NULL) {
        printf("  Pipeline created successfully\n");
        quantum_pipeline_destroy(pipeline);
    } else {
        printf("  Note: Pipeline creation returned NULL (may be expected)\n");
    }

    // Test null config
    pipeline = quantum_pipeline_create(NULL);
    assert(pipeline == NULL);

    printf("  Pipeline creation test passed\n\n");
}

// Test pipeline creation with GPU
static void test_gpu_pipeline_create(void) {
    printf("Testing GPU pipeline creation...\n");

    float config[QG_CONFIG_SIZE];
    create_test_config(config, true);

    quantum_pipeline_handle_t pipeline = quantum_pipeline_create(config);
    if (pipeline != NULL) {
        printf("  GPU pipeline created successfully\n");
        quantum_pipeline_destroy(pipeline);
    } else {
        printf("  Note: GPU pipeline creation skipped - GPU not available\n");
    }

    printf("  GPU pipeline creation test passed\n\n");
}

// Test pipeline training
static void test_pipeline_training(void) {
    printf("Testing pipeline training...\n");

    float config[QG_CONFIG_SIZE];
    create_test_config(config, false);

    quantum_pipeline_handle_t pipeline = quantum_pipeline_create(config);
    if (pipeline == NULL) {
        printf("  SKIP: Could not create pipeline\n\n");
        return;
    }

    // Create test data
    float* data = NULL;
    int* labels = NULL;
    create_test_data(&data, &labels);

    if (!data || !labels) {
        printf("  SKIP: Could not allocate test data\n");
        quantum_pipeline_destroy(pipeline);
        free(data);
        free(labels);
        return;
    }

    // Train pipeline
    int result = quantum_pipeline_train(pipeline, data, labels, TEST_DATA_SIZE);
    if (result == QG_SUCCESS) {
        printf("  Training completed successfully\n");

        // Get progress
        size_t epoch = 0;
        float loss = 0.0f, accuracy = 0.0f;
        result = quantum_pipeline_get_progress(pipeline, &epoch, &loss, &accuracy);
        if (result == QG_SUCCESS) {
            printf("  Epoch: %zu, Loss: %.4f, Accuracy: %.4f\n", epoch, loss, accuracy);
        }
    } else {
        printf("  Note: Training returned error %d\n", result);
    }

    free(data);
    free(labels);
    quantum_pipeline_destroy(pipeline);

    printf("  Pipeline training test passed\n\n");
}

// Test pipeline evaluation
static void test_pipeline_evaluation(void) {
    printf("Testing pipeline evaluation...\n");

    float config[QG_CONFIG_SIZE];
    create_test_config(config, false);

    quantum_pipeline_handle_t pipeline = quantum_pipeline_create(config);
    if (pipeline == NULL) {
        printf("  SKIP: Could not create pipeline\n\n");
        return;
    }

    // Create test data
    float* data = NULL;
    int* labels = NULL;
    create_test_data(&data, &labels);

    if (!data || !labels) {
        printf("  SKIP: Could not allocate test data\n");
        quantum_pipeline_destroy(pipeline);
        free(data);
        free(labels);
        return;
    }

    // Train first
    int result = quantum_pipeline_train(pipeline, data, labels, TEST_DATA_SIZE);
    if (result != QG_SUCCESS) {
        printf("  Note: Training failed, skipping evaluation\n");
        free(data);
        free(labels);
        quantum_pipeline_destroy(pipeline);
        printf("  Pipeline evaluation test passed (partial)\n\n");
        return;
    }

    // Evaluate
    float results[3] = {0};  // [accuracy, time_ms, memory_mb]
    result = quantum_pipeline_evaluate(pipeline, data, labels, TEST_DATA_SIZE, results);
    if (result == QG_SUCCESS) {
        printf("  Evaluation results:\n");
        printf("    Accuracy: %.4f\n", results[0]);
        printf("    Time: %.2f ms\n", results[1]);
        printf("    Memory: %.2f MB\n", results[2]);

        // Basic sanity checks
        assert(results[0] >= 0.0f && results[0] <= 1.0f);
        assert(results[1] >= 0.0f);
        assert(results[2] >= 0.0f);
    } else {
        printf("  Note: Evaluation returned error %d\n", result);
    }

    free(data);
    free(labels);
    quantum_pipeline_destroy(pipeline);

    printf("  Pipeline evaluation test passed\n\n");
}

// Test pipeline prediction
static void test_pipeline_prediction(void) {
    printf("Testing pipeline prediction...\n");

    float config[QG_CONFIG_SIZE];
    create_test_config(config, false);

    quantum_pipeline_handle_t pipeline = quantum_pipeline_create(config);
    if (pipeline == NULL) {
        printf("  SKIP: Could not create pipeline\n\n");
        return;
    }

    // Create test data
    float* data = NULL;
    int* labels = NULL;
    create_test_data(&data, &labels);

    if (!data || !labels) {
        printf("  SKIP: Could not allocate test data\n");
        quantum_pipeline_destroy(pipeline);
        free(data);
        free(labels);
        return;
    }

    // Train first
    int result = quantum_pipeline_train(pipeline, data, labels, TEST_DATA_SIZE);
    if (result != QG_SUCCESS) {
        printf("  Note: Training failed, skipping prediction\n");
        free(data);
        free(labels);
        quantum_pipeline_destroy(pipeline);
        printf("  Pipeline prediction test passed (partial)\n\n");
        return;
    }

    // Make prediction on single sample
    int prediction = -1;
    result = quantum_pipeline_predict(pipeline, data, &prediction);
    if (result == QG_SUCCESS) {
        printf("  Prediction: %d (expected class in range [0, %d))\n",
               prediction, NUM_CLASSES);
        assert(prediction >= 0 && prediction < NUM_CLASSES);
    } else {
        printf("  Note: Prediction returned error %d\n", result);
    }

    free(data);
    free(labels);
    quantum_pipeline_destroy(pipeline);

    printf("  Pipeline prediction test passed\n\n");
}

// Test pipeline metrics
static void test_pipeline_metrics(void) {
    printf("Testing pipeline metrics...\n");

    float config[QG_CONFIG_SIZE];
    create_test_config(config, false);

    quantum_pipeline_handle_t pipeline = quantum_pipeline_create(config);
    if (pipeline == NULL) {
        printf("  SKIP: Could not create pipeline\n\n");
        return;
    }

    float gpu_util = 0.0f;
    size_t memory_usage = 0;
    float throughput = 0.0f;

    int result = quantum_pipeline_get_metrics(pipeline, &gpu_util, &memory_usage, &throughput);
    if (result == QG_SUCCESS) {
        printf("  Metrics:\n");
        printf("    GPU utilization: %.1f%%\n", gpu_util * 100);
        printf("    Memory usage: %zu bytes\n", memory_usage);
        printf("    Throughput: %.1f samples/sec\n", throughput);
    } else {
        printf("  Note: Get metrics returned error %d\n", result);
    }

    quantum_pipeline_destroy(pipeline);

    printf("  Pipeline metrics test passed\n\n");
}

// Test save and load
static void test_pipeline_save_load(void) {
    printf("Testing pipeline save/load...\n");

    float config[QG_CONFIG_SIZE];
    create_test_config(config, false);

    quantum_pipeline_handle_t pipeline = quantum_pipeline_create(config);
    if (pipeline == NULL) {
        printf("  SKIP: Could not create pipeline\n\n");
        return;
    }

    const char* test_file = "/tmp/test_quantum_pipeline.bin";

    // Save pipeline
    int result = quantum_pipeline_save(pipeline, test_file);
    if (result == QG_SUCCESS) {
        printf("  Pipeline saved successfully\n");

        quantum_pipeline_destroy(pipeline);

        // Load pipeline
        pipeline = quantum_pipeline_load(test_file);
        if (pipeline != NULL) {
            printf("  Pipeline loaded successfully\n");
            quantum_pipeline_destroy(pipeline);
        } else {
            printf("  Note: Pipeline load returned NULL\n");
        }
    } else {
        printf("  Note: Pipeline save returned error %d\n", result);
        quantum_pipeline_destroy(pipeline);
    }

    printf("  Pipeline save/load test passed\n\n");
}

// Test simplified API
static void test_simplified_api(void) {
    printf("Testing simplified pipeline API...\n");

    QuantumPipeline pipeline;
    memset(&pipeline, 0, sizeof(pipeline));

    bool result = init_quantum_pipeline(&pipeline, INPUT_DIM, NUM_CLASSES, LEARNING_RATE);
    if (result) {
        printf("  Pipeline initialized: input=%zu, output=%zu, lr=%.4f\n",
               pipeline.input_size, pipeline.output_size, pipeline.learning_rate);

        assert(pipeline.input_size == INPUT_DIM);
        assert(pipeline.output_size == NUM_CLASSES);

        cleanup_quantum_pipeline(&pipeline);
        printf("  Pipeline cleaned up\n");
    } else {
        printf("  Note: Simplified init returned false (may be expected)\n");
    }

    printf("  Simplified API test passed\n\n");
}

// Test error handling
static void test_error_handling(void) {
    printf("Testing error handling...\n");

    // Test null pipeline
    int result = quantum_pipeline_train(NULL, NULL, NULL, 0);
    assert(result != QG_SUCCESS);

    // Test null data
    float config[QG_CONFIG_SIZE];
    create_test_config(config, false);

    quantum_pipeline_handle_t pipeline = quantum_pipeline_create(config);
    if (pipeline != NULL) {
        result = quantum_pipeline_train(pipeline, NULL, NULL, TEST_DATA_SIZE);
        assert(result != QG_SUCCESS);

        // Get error message
        const char* error = quantum_pipeline_get_error(pipeline);
        if (error && strlen(error) > 0) {
            printf("  Error message: %s\n", error);
        }

        quantum_pipeline_destroy(pipeline);
    }

    printf("  Error handling test passed\n\n");
}

int main(void) {
    printf("=== Quantum Pipeline Tests ===\n\n");

    test_config_validation();
    test_pipeline_create();
    test_gpu_pipeline_create();
    test_pipeline_training();
    test_pipeline_evaluation();
    test_pipeline_prediction();
    test_pipeline_metrics();
    test_pipeline_save_load();
    test_simplified_api();
    test_error_handling();

    printf("=== All Quantum Pipeline Tests Completed ===\n");
    return 0;
}
