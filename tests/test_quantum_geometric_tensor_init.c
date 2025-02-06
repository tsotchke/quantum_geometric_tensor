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

void test_geometric_tensor_init() {
    printf("Testing geometric tensor initialization...\n");
    double start_time, end_time;
    double total_start_time = get_time_ms();
    
    // Create a 2x2 geometric tensor
    quantum_geometric_tensor_t* tensor = NULL;
    size_t dimensions[] = {2, 2};
    
    start_time = get_time_ms();
    qgt_error_t err = geometric_tensor_create(&tensor, GEOMETRIC_TENSOR_HERMITIAN, dimensions, 2);
    end_time = get_time_ms();
    printf("Tensor creation time: %.3f ms\n", end_time - start_time);
    
    assert(err == QGT_SUCCESS);
    assert(tensor != NULL);
    assert(tensor->rank == 2);
    assert(tensor->dimensions[0] == 2);
    assert(tensor->dimensions[1] == 2);
    assert(tensor->components != NULL);
    assert(tensor->type == GEOMETRIC_TENSOR_HERMITIAN);
    assert(tensor->is_hermitian);
    
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
    
    // Test properties
    start_time = get_time_ms();
    bool is_hermitian = geometric_tensor_is_hermitian(tensor);
    end_time = get_time_ms();
    printf("Hermitian check time: %.3f ms\n", end_time - start_time);
    assert(is_hermitian);
    
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
    err = geometric_tensor_create(&result, GEOMETRIC_TENSOR_HERMITIAN, dimensions, 2);
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
    printf("✓ Geometric tensor initialization tests passed\n");
}

int main() {
    printf("Running quantum geometric tensor initialization tests...\n");
    
    test_geometric_tensor_init();
    
    printf("All tests passed!\n");
    return 0;
}
