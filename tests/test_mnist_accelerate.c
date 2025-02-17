#include "quantum_geometric/core/numerical_backend.h"
#include "quantum_geometric/core/error_codes.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/learning/quantum_pipeline.h"
#include "quantum_geometric/core/quantum_complex.h"
#include "quantum_geometric/learning/learning_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define INPUT_SIZE 64   // 8x8 test image
#define OUTPUT_SIZE 10  // 10 digits
#define BATCH_SIZE DEFAULT_BATCH_SIZE
#define LEARNING_RATE DEFAULT_LEARNING_RATE

int main() {
    printf("Testing MNIST Accelerate Implementation\n");
    printf("======================================\n\n");

    // Create test input data (normalized pixel values) for batch
    float* input_data = (float*)malloc(BATCH_SIZE * INPUT_SIZE * sizeof(float));
    if (!input_data) {
        printf("Failed to allocate input data\n");
        return 1;
    }
    
    // Initialize with test patterns (simple gradients with variations)
    printf("Initializing input data for %d samples...\n", BATCH_SIZE);
    for (int b = 0; b < BATCH_SIZE; b++) {
        float offset = (float)b / BATCH_SIZE;  // Add variation between samples
        for (int i = 0; i < INPUT_SIZE; i++) {
            input_data[b * INPUT_SIZE + i] = ((float)i / INPUT_SIZE + offset) / 2.0f;
        }
    }
    
    // Print first few inputs of first batch
    printf("First few inputs of first batch: ");
    for (int i = 0; i < (5 < INPUT_SIZE ? 5 : INPUT_SIZE); i++) {
        printf("%.4f ", input_data[i]);
    }
    printf("\n");

    // Initialize numerical backend
    printf("Initializing numerical backend...\n");
    numerical_config_t backend_config = {
        .type = NUMERICAL_BACKEND_ACCELERATE,
        .max_threads = 8,  // Use all cores
        .use_fma = true,   // Use FMA instructions
        .use_avx = true,   // Use AVX if available
        .use_neon = true,  // Use NEON on ARM
        .cache_size = 32 * 1024 * 1024,  // 32MB cache
        .backend_specific = NULL
    };
    
    // First check if Accelerate is available
    numerical_error_t error = is_backend_available(NUMERICAL_BACKEND_ACCELERATE);
    if (error != NUMERICAL_SUCCESS) {
        printf("Accelerate backend is not available: %s\n",
               get_numerical_error_string(error));
        free(input_data);
        return 1;
    }

    error = initialize_numerical_backend(&backend_config);
    if (error != NUMERICAL_SUCCESS) {
        printf("Failed to initialize numerical backend: %s\n",
               get_numerical_error_string(error));
        free(input_data);
        return 1;
    }

    // Initialize quantum pipeline
    printf("Creating quantum pipeline...\n");
    float pipeline_config[] = {
        (float)INPUT_SIZE,                // input_dim
        (float)QGT_MAX_DIMENSIONS,        // latent_dim
        (float)QGT_MAX_DIMENSIONS / 2,    // num_clusters
        (float)OUTPUT_SIZE,               // num_classes
        (float)BATCH_SIZE,                // batch_size
        LEARNING_RATE,                    // learning_rate
        0.0f,                            // use_gpu (0 for CPU/Accelerate)
        (float)QGT_MAX_DIMENSIONS / 2,    // num_qubits (same as clusters)
        2.0f                             // num_layers (reduced for stability)
    };
    
    quantum_pipeline_handle_t pipeline = quantum_pipeline_create(pipeline_config);
    if (!pipeline) {
        printf("Failed to create quantum pipeline\n");
        free(input_data);
        shutdown_numerical_backend();
        return 1;
    }

    // Create test labels (one-hot encoded) for batch
    int* labels = (int*)calloc(BATCH_SIZE * OUTPUT_SIZE, sizeof(int));
    if (!labels) {
        printf("Failed to allocate labels\n");
        free(input_data);
        quantum_pipeline_destroy(pipeline);
        shutdown_numerical_backend();
        return 1;
    }
    
    // Initialize labels with random targets for each batch
    printf("Initializing labels for %d samples...\n", BATCH_SIZE);
    srand(time(NULL));  // Seed random number generator
    for (int b = 0; b < BATCH_SIZE; b++) {
        int target = rand() % OUTPUT_SIZE;  // Random target for each batch
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            labels[b * OUTPUT_SIZE + i] = (i == target) ? 1 : 0;
        }
    }
    
    // Print first few labels
    printf("First few batch targets: ");
    for (int b = 0; b < (5 < BATCH_SIZE ? 5 : BATCH_SIZE); b++) {
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            if (labels[b * OUTPUT_SIZE + i]) {
                printf("%d ", i);
                break;
            }
        }
    }
    printf("\n");
    
    // Train pipeline for multiple epochs
    const int NUM_EPOCHS = 10;
    printf("Training quantum pipeline for %d epochs...\n", NUM_EPOCHS);
    
    int result;  // Declare result variable for error handling
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        printf("\nEpoch %d/%d:\n", epoch + 1, NUM_EPOCHS);
        
        result = quantum_pipeline_train(pipeline, input_data, labels, BATCH_SIZE);
        if (result != QGT_SUCCESS) {
            printf("Pipeline training failed at epoch %d\n", epoch + 1);
            free(input_data);
            free(labels);
            quantum_pipeline_destroy(pipeline);
            shutdown_numerical_backend();
            return 1;
        }
        
        // Evaluate after each epoch
        float epoch_metrics[3];
        result = quantum_pipeline_evaluate(pipeline, input_data, labels, BATCH_SIZE, epoch_metrics);
        if (result != QGT_SUCCESS) {
            printf("Pipeline evaluation failed at epoch %d\n", epoch + 1);
            free(input_data);
            free(labels);
            quantum_pipeline_destroy(pipeline);
            shutdown_numerical_backend();
            return 1;
        }
        
        printf("Epoch %d - Accuracy: %.2f%%\n", epoch + 1, epoch_metrics[0] * 100);
    }

    // Allocate output buffer for batch predictions
    float* output_data = (float*)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    if (!output_data) {
        printf("Failed to allocate output data\n");
        free(input_data);
        free(labels);
        quantum_pipeline_destroy(pipeline);
        shutdown_numerical_backend();
        return 1;
    }

    // Evaluate pipeline on full batch
    printf("Evaluating quantum pipeline on %d samples...\n", BATCH_SIZE);
    float metrics[3];  // accuracy, time, memory
    result = quantum_pipeline_evaluate(pipeline, input_data, labels, BATCH_SIZE, metrics);
    if (result != QGT_SUCCESS) {
        printf("Pipeline evaluation failed\n");
        free(input_data);
        free(labels);
        free(output_data);
        quantum_pipeline_destroy(pipeline);
        shutdown_numerical_backend();
        return 1;
    }

    // Print first few predictions
    printf("\nFirst few predictions:\n");
    for (int b = 0; b < (3 < BATCH_SIZE ? 3 : BATCH_SIZE); b++) {
        printf("Sample %d - Target: ", b);
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            if (labels[b * OUTPUT_SIZE + i]) {
                printf("%d, ", i);
                break;
            }
        }
        printf("Predicted: ");
        float max_prob = 0.0f;
        int predicted = -1;
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            float prob = output_data[b * OUTPUT_SIZE + i];
            if (prob > max_prob) {
                max_prob = prob;
                predicted = i;
            }
        }
        printf("%d (confidence: %.2f%%)\n", predicted, max_prob * 100);
    }

    // Print results
    printf("\nResults:\n");
    printf("Accuracy: %.2f%%\n", metrics[0] * 100);
    printf("Execution time: %.2f ms\n", metrics[1]);
    printf("Memory usage: %.2f MB\n", metrics[2]);
    printf("Note: Using Apple Accelerate framework for optimized performance\n");

    // Save model
    printf("\nSaving model...\n");
    result = quantum_pipeline_save(pipeline, "mnist_model_accelerate.qg");
    if (result != QGT_SUCCESS) {
        printf("Failed to save model\n");
    } else {
        printf("Model saved successfully to mnist_model_accelerate.qg\n");
    }

    // Cleanup
    free(input_data);
    free(labels);
    free(output_data);
    quantum_pipeline_destroy(pipeline);
    shutdown_numerical_backend();

    printf("\nTest completed successfully!\n");
    return 0;
}
