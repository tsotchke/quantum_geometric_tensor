#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include "quantum_geometric/hardware/metal/mnist_metal.h"
#include "quantum_geometric/core/error_codes.h"
#include "quantum_geometric/learning/quantum_pipeline.h"

#define INPUT_SIZE 784  // 28x28 MNIST image
#define OUTPUT_SIZE 10  // 10 digits
#define LATENT_DIM 32  // Quantum geometric encoding dimension
#define NUM_CLUSTERS 16 // Number of quantum clusters
#define BATCH_SIZE 1
#define LEARNING_RATE 0.001f

int main() {
    printf("Testing MNIST Metal Implementation\n");
    printf("==================================\n\n");

    // Create test input data (normalized pixel values)
    float* input_data = (float*)malloc(INPUT_SIZE * sizeof(float));
    if (!input_data) {
        printf("Failed to allocate input data\n");
        return 1;
    }
    
    // Initialize with test pattern (simple gradient)
    printf("Initializing input data...\n");
    for (int i = 0; i < INPUT_SIZE; i++) {
        input_data[i] = (float)i / INPUT_SIZE;
    }
    printf("First few inputs: ");
    for (int i = 0; i < (5 < INPUT_SIZE ? 5 : INPUT_SIZE); i++) {
        printf("%.4f ", input_data[i]);
    }
    printf("\n");

    // Initialize quantum pipeline
    printf("Initializing quantum pipeline...\n");
    float config[] = {
        (float)INPUT_SIZE,   // input_dim
        (float)LATENT_DIM,   // latent_dim
        (float)NUM_CLUSTERS, // num_clusters
        (float)OUTPUT_SIZE,  // num_classes
        (float)BATCH_SIZE,   // batch_size
        LEARNING_RATE,       // learning_rate
        1.0f                 // use_gpu
    };
    
    quantum_pipeline_handle_t pipeline = quantum_pipeline_create(config);
    if (!pipeline) {
        printf("Failed to create quantum pipeline\n");
        free(input_data);
        return 1;
    }

    // Create test labels
    int labels[] = {5}; // Example label for test
    
    // Train pipeline (this will use quantum geometric encoding)
    printf("Training quantum pipeline...\n");
    int result = quantum_pipeline_train(pipeline, input_data, labels, 1);
    if (result != QGT_SUCCESS) {
        printf("Pipeline training failed\n");
        free(input_data);
        quantum_pipeline_destroy(pipeline);
        return 1;
    }

    // Allocate output buffer
    float* output_data = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    if (!output_data) {
        printf("Failed to allocate output data\n");
        free(input_data);
        quantum_pipeline_destroy(pipeline);
        return 1;
    }

    // Evaluate pipeline
    printf("Evaluating quantum pipeline...\n");
    float metrics[3];  // accuracy, time, memory
    result = quantum_pipeline_evaluate(pipeline, input_data, labels, 1, metrics);
    if (result != QGT_SUCCESS) {
        printf("Pipeline evaluation failed\n");
        free(input_data);
        free(output_data);
        quantum_pipeline_destroy(pipeline);
        return 1;
    }

    // Get GPU metrics
    float gpu_metrics[2];  // utilization, TFLOPS
    result = quantum_pipeline_get_gpu_metrics(pipeline, gpu_metrics);
    if (result == QGT_SUCCESS) {
        printf("\nGPU Metrics:\n");
        printf("GPU Utilization: %.1f%%\n", gpu_metrics[0] * 100);
        printf("Performance: %.2f TFLOPS\n", gpu_metrics[1]);
    }

    // Print results
    printf("\nResults:\n");
    printf("Accuracy: %.2f%%\n", metrics[0] * 100);
    printf("Execution time: %.2f ms\n", metrics[1]);
    printf("Memory usage: %.2f MB\n", metrics[2]);

    // Save model
    printf("\nSaving model...\n");
    result = quantum_pipeline_save(pipeline, "mnist_model.qg");
    if (result != QGT_SUCCESS) {
        printf("Failed to save model\n");
    } else {
        printf("Model saved successfully to mnist_model.qg\n");
    }

    // Cleanup
    free(input_data);
    free(output_data);
    quantum_pipeline_destroy(pipeline);

    printf("\nTest completed successfully!\n");
    return 0;
}
