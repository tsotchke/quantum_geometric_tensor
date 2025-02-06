#include "quantum_geometric/core/quantum_geometric_tensor.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/error_codes.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

// Helper function to get time in milliseconds
static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

// Test configuration structure
typedef struct {
    size_t* dimensions;
    size_t rank;
    geometric_tensor_type_t type;
    const char* name;
} tensor_test_config_t;

// Helper function to test a specific tensor configuration
void test_tensor_config(const tensor_test_config_t* config) {
    printf("\nTesting %s tensor...\n", config->name);
    double start_time, end_time;
    double total_start_time = get_time_ms();
    
    // Print tensor configuration
    printf("Rank: %zu, Dimensions: [", config->rank);
    for (size_t i = 0; i < config->rank; i++) {
        printf("%zu%s", config->dimensions[i], i < config->rank - 1 ? "x" : "");
    }
    printf("], Type: %d\n", config->type);
    
    // Create tensor
    quantum_geometric_tensor_t* tensor = NULL;
    start_time = get_time_ms();
    qgt_error_t err = geometric_tensor_create(&tensor, config->type, config->dimensions, config->rank);
    end_time = get_time_ms();
    printf("Tensor creation time: %.3f ms\n", end_time - start_time);
    
    assert(err == QGT_SUCCESS);
    assert(tensor != NULL);
    assert(tensor->rank == config->rank);
    for (size_t i = 0; i < config->rank; i++) {
        assert(tensor->dimensions[i] == config->dimensions[i]);
    }
    assert(tensor->components != NULL);
    assert(tensor->type == config->type);
    
    // Initialize with random data
    start_time = get_time_ms();
    err = geometric_tensor_initialize_random(tensor);
    end_time = get_time_ms();
    printf("Random initialization time: %.3f ms\n", end_time - start_time);
    assert(err == QGT_SUCCESS);
    
    // Validate the tensor
    start_time = get_time_ms();
    err = geometric_tensor_validate(tensor);
    end_time = get_time_ms();
    printf("Validation time: %.3f ms\n", end_time - start_time);
    assert(err == QGT_SUCCESS);
    
    // Test properties based on type
    if (config->type == GEOMETRIC_TENSOR_HERMITIAN && config->rank == 2) {
        start_time = get_time_ms();
        bool is_hermitian = geometric_tensor_is_hermitian(tensor);
        end_time = get_time_ms();
        printf("Hermitian check time: %.3f ms\n", end_time - start_time);
        assert(is_hermitian);
    }
    
    // Calculate norm
    float norm;
    start_time = get_time_ms();
    err = geometric_tensor_norm(&norm, tensor);
    end_time = get_time_ms();
    printf("Norm calculation time: %.3f ms\n", end_time - start_time);
    assert(err == QGT_SUCCESS);
    printf("Tensor norm: %f\n", norm);
    
    // Create another tensor for operations
    quantum_geometric_tensor_t* result = NULL;
    start_time = get_time_ms();
    err = geometric_tensor_create(&result, config->type, config->dimensions, config->rank);
    end_time = get_time_ms();
    printf("Second tensor creation time: %.3f ms\n", end_time - start_time);
    assert(err == QGT_SUCCESS);
    
    // Test conjugate operation
    start_time = get_time_ms();
    err = geometric_tensor_conjugate(result, tensor);
    end_time = get_time_ms();
    printf("Conjugate operation time: %.3f ms\n", end_time - start_time);
    assert(err == QGT_SUCCESS);
    
    // Verify conjugate result is valid
    start_time = get_time_ms();
    err = geometric_tensor_validate(result);
    end_time = get_time_ms();
    printf("Result validation time: %.3f ms\n", end_time - start_time);
    assert(err == QGT_SUCCESS);
    
    // Cleanup
    geometric_tensor_destroy(tensor);
    geometric_tensor_destroy(result);
    
    double total_end_time = get_time_ms();
    printf("Total operation time: %.3f ms\n", total_end_time - total_start_time);
    printf("✓ %s tensor tests passed\n", config->name);
}

void test_geometric_tensor_init() {
    printf("Testing geometric tensor initialization...\n");
    
    // Test configurations
    size_t dims_1d[] = {4};                          // Vector
    size_t dims_2d[] = {2, 2};                       // 2x2 matrix
    size_t dims_2d_large[] = {4, 4};                 // 4x4 matrix
    size_t dims_3d[] = {2, 2, 2};                    // 3D tensor
    size_t dims_4d[] = {2, 2, 2, 2};                 // 4D tensor
    size_t dims_mixed[] = {3, 4, 2};                 // Mixed dimensions
    
    tensor_test_config_t configs[] = {
        {dims_1d, 1, GEOMETRIC_TENSOR_VECTOR, "Vector"},
        {dims_2d, 2, GEOMETRIC_TENSOR_HERMITIAN, "2x2 Hermitian"},
        {dims_2d_large, 2, GEOMETRIC_TENSOR_UNITARY, "4x4 Unitary"},
        {dims_3d, 3, GEOMETRIC_TENSOR_SYMMETRIC, "3D Symmetric"},
        {dims_4d, 4, GEOMETRIC_TENSOR_CUSTOM, "4D Custom"},
        {dims_mixed, 3, GEOMETRIC_TENSOR_TRIVECTOR, "Mixed Dimensions"}
    };
    
    size_t num_configs = sizeof(configs) / sizeof(configs[0]);
    for (size_t i = 0; i < num_configs; i++) {
        test_tensor_config(&configs[i]);
    }
}

int main() {
    printf("Running quantum geometric tensor initialization tests...\n");
    test_geometric_tensor_init();
    printf("All tests passed!\n");
    return 0;
}
