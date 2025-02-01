#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/core/quantum_geometric_tensor.h"
#include "quantum_geometric/learning/quantum_pipeline.h"
#include "quantum_geometric/learning/data_loader.h"
#include "quantum_geometric/core/quantum_geometric_error.h"
#include "quantum_geometric/core/quantum_geometric_logging.h"
#include "quantum_geometric/core/quantum_geometric_memory.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/learning/learning_task.h"
#include "quantum_geometric/core/tensor_types.h"
#include "quantum_geometric/core/quantum_complex.h"
#include "test_helpers.h"

#define EPSILON 1e-6

// Data loader structure definition
typedef struct {
    size_t image_width;
    size_t image_height;
    size_t num_channels;
    void* data_handle;  // Internal handle for data management
} DataLoader;

// Quantum pipeline structure definition
typedef struct {
    size_t input_size;
    size_t output_size;
    float learning_rate;
    void* model_handle;  // Internal handle for quantum model
} QuantumPipeline;

// Data loader function declarations
bool init_data_loader(DataLoader* loader, const char* dataset_name, size_t batch_size);
void cleanup_data_loader(DataLoader* loader);
bool load_next_batch(DataLoader* loader, tensor_t* images, tensor_t* labels);

// Quantum pipeline function declarations
bool init_quantum_pipeline(QuantumPipeline* pipeline, size_t input_size, size_t output_size, float learning_rate);
void cleanup_quantum_pipeline(QuantumPipeline* pipeline);
bool train_step(QuantumPipeline* pipeline, tensor_t* input, tensor_t* target, float* loss);
bool inference(QuantumPipeline* pipeline, tensor_t* input, tensor_t* output);

#define BATCH_SIZE 32
#define NUM_EPOCHS 5
#define LEARNING_RATE 0.001
#define NUM_CLASSES 10
#define IMAGE_SIZE 32
#define NUM_CHANNELS 3

static int tests_run = 0;
static int tests_passed = 0;

static bool test_cifar10_data_loading() {
    printf("Testing CIFAR-10 data loading...\n");
    tests_run++;

    DataLoader loader;
    if (!init_data_loader(&loader, "cifar10", BATCH_SIZE)) {
        printf("Failed to initialize data loader\n");
        return false;
    }

    // Verify data dimensions
    if (loader.image_width != IMAGE_SIZE || 
        loader.image_height != IMAGE_SIZE || 
        loader.num_channels != NUM_CHANNELS) {
        printf("Incorrect data dimensions\n");
        cleanup_data_loader(&loader);
        return false;
    }

    // Load a batch and verify its properties
    tensor_t images, labels;
    if (!load_next_batch(&loader, &images, &labels)) {
        printf("Failed to load batch\n");
        cleanup_data_loader(&loader);
        return false;
    }

    // Verify batch dimensions
    if (images.dimensions[0] != BATCH_SIZE || 
        images.dimensions[1] != IMAGE_SIZE || 
        images.dimensions[2] != IMAGE_SIZE || 
        images.dimensions[3] != NUM_CHANNELS) {
        printf("Incorrect batch dimensions\n");
        qg_tensor_cleanup(&images);
        qg_tensor_cleanup(&labels);
        cleanup_data_loader(&loader);
        return false;
    }

    // Verify label dimensions
    if (labels.dimensions[0] != BATCH_SIZE || 
        labels.dimensions[1] != NUM_CLASSES) {
        printf("Incorrect label dimensions\n");
        qg_tensor_cleanup(&images);
        qg_tensor_cleanup(&labels);
        cleanup_data_loader(&loader);
        return false;
    }

    // Verify data ranges
    for (size_t i = 0; i < images.total_size; i++) {
        if (images.data[i].real < -1.0f || images.data[i].real > 1.0f ||
            fabs(images.data[i].imag) > EPSILON) {
            printf("Image values out of expected range\n");
            qg_tensor_cleanup(&images);
            qg_tensor_cleanup(&labels);
            cleanup_data_loader(&loader);
            return false;
        }
    }

    qg_tensor_cleanup(&images);
    qg_tensor_cleanup(&labels);
    cleanup_data_loader(&loader);
    tests_passed++;
    return true;
}

static bool test_model_initialization() {
    printf("Testing quantum model initialization...\n");
    tests_run++;

    QuantumPipeline pipeline;
    if (!init_quantum_pipeline(&pipeline, 
                             IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS,  // Input size
                             NUM_CLASSES,                            // Output size
                             LEARNING_RATE)) {
        printf("Failed to initialize quantum pipeline\n");
        return false;
    }

    // Verify pipeline properties
    if (pipeline.input_size != IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS ||
        pipeline.output_size != NUM_CLASSES) {
        printf("Incorrect pipeline dimensions\n");
        cleanup_quantum_pipeline(&pipeline);
        return false;
    }

    cleanup_quantum_pipeline(&pipeline);
    tests_passed++;
    return true;
}

static bool test_training_step() {
    printf("Testing training step...\n");
    tests_run++;

    // Initialize pipeline
    QuantumPipeline pipeline;
    if (!init_quantum_pipeline(&pipeline, 
                             IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS,
                             NUM_CLASSES,
                             LEARNING_RATE)) {
        printf("Failed to initialize quantum pipeline\n");
        return false;
    }

    // Create test batch
    size_t input_dims[] = {BATCH_SIZE, IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS};
    size_t output_dims[] = {BATCH_SIZE, NUM_CLASSES};
    tensor_t input, target;

    if (!qg_tensor_init(&input, input_dims, 2, GEOMETRIC_TENSOR_HERMITIAN) ||
        !qg_tensor_init(&target, output_dims, 2, GEOMETRIC_TENSOR_HERMITIAN)) {
        printf("Failed to initialize tensors\n");
        cleanup_quantum_pipeline(&pipeline);
        return false;
    }

    // Initialize test data
    for (size_t i = 0; i < input.total_size; i++) {
        input.data[i] = complex_float_create(
            ((float)rand() / RAND_MAX) * 2.0f - 1.0f,  // [-1, 1]
            0.0f
        );
    }

    for (size_t i = 0; i < target.total_size; i++) {
        target.data[i] = complex_float_create(
            (i % NUM_CLASSES == (i / NUM_CLASSES) % NUM_CLASSES) ? 1.0f : 0.0f,
            0.0f
        );
    }

    // Perform training step
    float loss;
    if (!train_step(&pipeline, &input, &target, &loss)) {
        printf("Training step failed\n");
        qg_tensor_cleanup(&input);
        qg_tensor_cleanup(&target);
        cleanup_quantum_pipeline(&pipeline);
        return false;
    }

    // Verify loss is reasonable
    if (isnan(loss) || isinf(loss) || loss < 0.0f) {
        printf("Invalid loss value: %f\n", loss);
        qg_tensor_cleanup(&input);
        qg_tensor_cleanup(&target);
        cleanup_quantum_pipeline(&pipeline);
        return false;
    }

    qg_tensor_cleanup(&input);
    qg_tensor_cleanup(&target);
    cleanup_quantum_pipeline(&pipeline);
    tests_passed++;
    return true;
}

static bool test_inference() {
    printf("Testing inference...\n");
    tests_run++;

    // Initialize pipeline
    QuantumPipeline pipeline;
    if (!init_quantum_pipeline(&pipeline, 
                             IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS,
                             NUM_CLASSES,
                             LEARNING_RATE)) {
        printf("Failed to initialize quantum pipeline\n");
        return false;
    }

    // Create test input
    size_t input_dims[] = {1, IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS};
    tensor_t input, output;

    if (!qg_tensor_init(&input, input_dims, 2, GEOMETRIC_TENSOR_HERMITIAN)) {
        printf("Failed to initialize input tensor\n");
        cleanup_quantum_pipeline(&pipeline);
        return false;
    }

    // Initialize test data
    for (size_t i = 0; i < input.total_size; i++) {
        input.data[i] = complex_float_create(
            ((float)rand() / RAND_MAX) * 2.0f - 1.0f,  // [-1, 1]
            0.0f
        );
    }

    // Perform inference
    if (!inference(&pipeline, &input, &output)) {
        printf("Inference failed\n");
        qg_tensor_cleanup(&input);
        cleanup_quantum_pipeline(&pipeline);
        return false;
    }

    // Verify output dimensions
    if (output.dimensions[0] != 1 || output.dimensions[1] != NUM_CLASSES) {
        printf("Incorrect output dimensions\n");
        qg_tensor_cleanup(&input);
        qg_tensor_cleanup(&output);
        cleanup_quantum_pipeline(&pipeline);
        return false;
    }

    // Verify output properties
    float sum = 0.0f;
    for (size_t i = 0; i < output.total_size; i++) {
        if (output.data[i].real < 0.0f || output.data[i].real > 1.0f ||
            fabs(output.data[i].imag) > EPSILON) {
            printf("Invalid output values\n");
            qg_tensor_cleanup(&input);
            qg_tensor_cleanup(&output);
            cleanup_quantum_pipeline(&pipeline);
            return false;
        }
        sum += output.data[i].real;
    }

    // Verify probabilities sum to approximately 1
    if (fabs(sum - 1.0f) > EPSILON) {
        printf("Output probabilities don't sum to 1\n");
        qg_tensor_cleanup(&input);
        qg_tensor_cleanup(&output);
        cleanup_quantum_pipeline(&pipeline);
        return false;
    }

    qg_tensor_cleanup(&input);
    qg_tensor_cleanup(&output);
    cleanup_quantum_pipeline(&pipeline);
    tests_passed++;
    return true;
}

int main() {
    printf("Running CIFAR-10 quantum learning tests...\n\n");

    srand(42);  // For reproducibility

    test_cifar10_data_loading();
    test_model_initialization();
    test_training_step();
    test_inference();

    printf("\nTest summary: %d/%d tests passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
