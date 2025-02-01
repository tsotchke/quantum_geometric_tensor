#include "quantum_geometric/core/differential_transformer.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>

// Test parameters
#define TEST_SEQ_LENGTH 32
#define TEST_HIDDEN_DIM 64
#define TEST_NUM_HEADS 4
#define TEST_LEARNING_RATE 1e-3
#define EPSILON 1e-6

// Helper function to generate test data
static void generate_test_data(double* data, size_t size) {
    for (size_t i = 0; i < size; i++) {
        data[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }
}

// Helper function to verify numerical derivatives
static void test_numerical_derivatives(DiffTransformerState* state) {
    printf("Testing numerical derivatives...\n");
    
    size_t data_size = state->seq_length * state->hidden_dim;
    double* input = malloc(data_size * sizeof(double));
    double* output = malloc(data_size * sizeof(double));
    double* perturbed = malloc(data_size * sizeof(double));
    
    // Generate random input
    generate_test_data(input, data_size);
    
    // Forward pass
    diff_transformer_forward(state, input, output);
    
    // Verify derivatives using finite differences
    double max_diff = 0.0;
    for (size_t i = 0; i < data_size; i++) {
        // Compute numerical derivative
        double h = fmax(fabs(input[i]) * 1e-4, EPSILON);
        
        // f(x + h)
        memcpy(perturbed, input, data_size * sizeof(double));
        perturbed[i] += h;
        diff_transformer_forward(state, perturbed, output);
        double fplus = output[i];
        
        // f(x - h)
        memcpy(perturbed, input, data_size * sizeof(double));
        perturbed[i] -= h;
        diff_transformer_forward(state, perturbed, output);
        double fminus = output[i];
        
        // Central difference
        double numerical_deriv = (fplus - fminus) / (2.0 * h);
        double diff = fabs(numerical_deriv - state->derivatives[i]);
        max_diff = fmax(max_diff, diff);
    }
    
    printf("Maximum derivative difference: %g\n", max_diff);
    assert(max_diff < 1e-4 && "Derivatives match numerical approximation");
    
    free(input);
    free(output);
    free(perturbed);
}

// Test attention mechanism
static void test_attention_mechanism() {
    printf("Testing attention mechanism...\n");
    
    DiffTransformerState* state = create_diff_transformer(
        TEST_SEQ_LENGTH, TEST_HIDDEN_DIM, TEST_NUM_HEADS, TEST_LEARNING_RATE
    );
    
    size_t data_size = TEST_SEQ_LENGTH * TEST_HIDDEN_DIM;
    double* input = malloc(data_size * sizeof(double));
    double* output = malloc(data_size * sizeof(double));
    
    // Generate random input
    generate_test_data(input, data_size);
    
    // Forward pass
    diff_transformer_forward(state, input, output);
    
    // Verify attention properties
    double attention_sum = 0.0;
    for (size_t i = 0; i < TEST_SEQ_LENGTH; i++) {
        for (size_t j = 0; j < TEST_SEQ_LENGTH; j++) {
            attention_sum += output[i * TEST_HIDDEN_DIM + j];
        }
    }
    
    // Check if attention weights approximately sum to sequence length
    double expected_sum = TEST_SEQ_LENGTH;
    double diff = fabs(attention_sum - expected_sum);
    printf("Attention sum difference from expected: %g\n", diff);
    assert(diff < 1e-4 && "Attention weights sum to approximately sequence length");
    
    free(input);
    free(output);
    free_diff_transformer(state);
}

// Test stability and convergence
static void test_stability() {
    printf("Testing stability and convergence...\n");
    
    DiffTransformerState* state = create_diff_transformer(
        TEST_SEQ_LENGTH, TEST_HIDDEN_DIM, TEST_NUM_HEADS, TEST_LEARNING_RATE
    );
    
    size_t data_size = TEST_SEQ_LENGTH * TEST_HIDDEN_DIM;
    double* input = malloc(data_size * sizeof(double));
    double* output = malloc(data_size * sizeof(double));
    double* target = malloc(data_size * sizeof(double));
    
    // Generate random input and target
    generate_test_data(input, data_size);
    generate_test_data(target, data_size);
    
    // Training loop
    double prev_loss = INFINITY;
    int num_iterations = 100;
    
    for (int i = 0; i < num_iterations; i++) {
        // Forward pass
        diff_transformer_forward(state, input, output);
        
        // Compute loss
        double loss = 0.0;
        for (size_t j = 0; j < data_size; j++) {
            double diff = output[j] - target[j];
            loss += diff * diff;
        }
        loss /= data_size;
        
        // Verify loss is decreasing
        if (i > 0) {
            assert(loss <= prev_loss * 1.1 && "Loss is generally decreasing");
        }
        prev_loss = loss;
        
        // Backward pass
        diff_transformer_backward(state, target, NULL);
        
        // Update parameters
        ttn_update_parameters(state, TEST_LEARNING_RATE);
    }
    
    // Check final stability
    double stability = compute_differential_stability(state);
    printf("Final stability metric: %g\n", stability);
    assert(stability < 1.0 && "Model remains stable after training");
    
    free(input);
    free(output);
    free(target);
    free_diff_transformer(state);
}

// Test GPU acceleration if available
static void test_gpu_acceleration() {
    printf("Testing GPU acceleration...\n");
    
    DiffTransformerState* state = create_diff_transformer(
        TEST_SEQ_LENGTH, TEST_HIDDEN_DIM, TEST_NUM_HEADS, TEST_LEARNING_RATE
    );
    
    size_t data_size = TEST_SEQ_LENGTH * TEST_HIDDEN_DIM;
    double* input = malloc(data_size * sizeof(double));
    double* output_cpu = malloc(data_size * sizeof(double));
    double* output_gpu = malloc(data_size * sizeof(double));
    
    // Generate random input
    generate_test_data(input, data_size);
    
    // CPU forward pass
    diff_transformer_forward(state, input, output_cpu);
    
    // GPU forward pass
    #ifdef USE_CUDA
    cuda_diff_transformer_forward(state, input, output_gpu);
    #endif
    
    #ifdef USE_METAL
    metal_diff_transformer_forward(state, input, output_gpu);
    #endif
    
    // Compare results
    #if defined(USE_CUDA) || defined(USE_METAL)
    double max_diff = 0.0;
    for (size_t i = 0; i < data_size; i++) {
        max_diff = fmax(max_diff, fabs(output_cpu[i] - output_gpu[i]));
    }
    printf("Maximum CPU-GPU difference: %g\n", max_diff);
    assert(max_diff < 1e-5 && "CPU and GPU results match");
    #else
    printf("No GPU acceleration available\n");
    #endif
    
    free(input);
    free(output_cpu);
    free(output_gpu);
    free_diff_transformer(state);
}

int main() {
    printf("Running differential transformer tests\n");
    printf("=====================================\n\n");
    
    // Initialize random seed
    srand(time(NULL));
    
    // Run tests
    test_numerical_derivatives(create_diff_transformer(
        TEST_SEQ_LENGTH, TEST_HIDDEN_DIM, TEST_NUM_HEADS, TEST_LEARNING_RATE
    ));
    printf("\n");
    
    test_attention_mechanism();
    printf("\n");
    
    test_stability();
    printf("\n");
    
    test_gpu_acceleration();
    printf("\n");
    
    printf("All tests passed!\n");
    return 0;
}
