#include "quantum_geometric/core/quantum_geometric_tensor.h"
#include "quantum_geometric/core/error_codes.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#define EPSILON 1e-6f

static void test_tensor_create() {
    quantum_geometric_tensor_t* tensor;
    size_t dimensions[] = {2, 3, 4};
    qgt_error_t err = geometric_tensor_create(&tensor, GEOMETRIC_TENSOR_SCALAR, dimensions, 3);
    
    assert(err == QGT_SUCCESS);
    assert(tensor != NULL);
    assert(tensor->rank == 3);
    assert(tensor->dimensions[0] == 2);
    assert(tensor->dimensions[1] == 3);
    assert(tensor->dimensions[2] == 4);
    assert(tensor->components != NULL);
    
    geometric_tensor_destroy(tensor);
    printf("✓ Tensor creation test passed\n");
}

static void test_tensor_operations() {
    quantum_geometric_tensor_t *a, *b, *result;
    size_t dimensions[] = {2, 2};
    
    // Create tensors
    geometric_tensor_create(&a, GEOMETRIC_TENSOR_SCALAR, dimensions, 2);
    geometric_tensor_create(&b, GEOMETRIC_TENSOR_SCALAR, dimensions, 2);
    geometric_tensor_create(&result, GEOMETRIC_TENSOR_SCALAR, dimensions, 2);
    
    // Initialize test data
    ComplexFloat data_a[] = {
        {1.0f, 0.0f}, {2.0f, 0.0f},
        {3.0f, 0.0f}, {4.0f, 0.0f}
    };
    ComplexFloat data_b[] = {
        {5.0f, 0.0f}, {6.0f, 0.0f},
        {7.0f, 0.0f}, {8.0f, 0.0f}
    };
    
    memcpy(a->components, data_a, 4 * sizeof(ComplexFloat));
    memcpy(b->components, data_b, 4 * sizeof(ComplexFloat));
    
    // Test addition
    qgt_error_t err = geometric_tensor_add(result, a, b);
    assert(err == QGT_SUCCESS);
    assert(fabsf(result->components[0].real - 6.0f) < EPSILON);
    assert(fabsf(result->components[1].real - 8.0f) < EPSILON);
    assert(fabsf(result->components[2].real - 10.0f) < EPSILON);
    assert(fabsf(result->components[3].real - 12.0f) < EPSILON);
    
    // Test subtraction
    err = geometric_tensor_subtract(result, a, b);
    assert(err == QGT_SUCCESS);
    assert(fabsf(result->components[0].real + 4.0f) < EPSILON);
    assert(fabsf(result->components[1].real + 4.0f) < EPSILON);
    assert(fabsf(result->components[2].real + 4.0f) < EPSILON);
    assert(fabsf(result->components[3].real + 4.0f) < EPSILON);
    
    // Test multiplication
    err = geometric_tensor_multiply(result, a, b);
    assert(err == QGT_SUCCESS);
    assert(fabsf(result->components[0].real - 19.0f) < EPSILON);
    assert(fabsf(result->components[1].real - 22.0f) < EPSILON);
    assert(fabsf(result->components[2].real - 43.0f) < EPSILON);
    assert(fabsf(result->components[3].real - 50.0f) < EPSILON);
    
    // Test scaling
    ComplexFloat scalar = {2.0f, 0.0f};
    err = geometric_tensor_scale(result, a, scalar);
    assert(err == QGT_SUCCESS);
    assert(fabsf(result->components[0].real - 2.0f) < EPSILON);
    assert(fabsf(result->components[1].real - 4.0f) < EPSILON);
    assert(fabsf(result->components[2].real - 6.0f) < EPSILON);
    assert(fabsf(result->components[3].real - 8.0f) < EPSILON);
    
    // Test norm
    float norm;
    err = geometric_tensor_norm(&norm, a);
    assert(err == QGT_SUCCESS);
    assert(fabsf(norm - sqrtf(30.0f)) < EPSILON);
    
    // Test conjugate
    err = geometric_tensor_conjugate(result, a);
    assert(err == QGT_SUCCESS);
    assert(fabsf(result->components[0].real - 1.0f) < EPSILON);
    assert(fabsf(result->components[1].real - 2.0f) < EPSILON);
    assert(fabsf(result->components[2].real - 3.0f) < EPSILON);
    assert(fabsf(result->components[3].real - 4.0f) < EPSILON);
    
    // Test transpose
    size_t permutation[] = {1, 0};
    err = geometric_tensor_transpose(result, a, permutation);
    assert(err == QGT_SUCCESS);
    assert(fabsf(result->components[0].real - 1.0f) < EPSILON);
    assert(fabsf(result->components[1].real - 3.0f) < EPSILON);
    assert(fabsf(result->components[2].real - 2.0f) < EPSILON);
    assert(fabsf(result->components[3].real - 4.0f) < EPSILON);
    
    geometric_tensor_destroy(a);
    geometric_tensor_destroy(b);
    geometric_tensor_destroy(result);
    printf("✓ Tensor operations test passed\n");
}

static void test_tensor_validation() {
    quantum_geometric_tensor_t* tensor;
    size_t dimensions[] = {2, 3, 4};
    geometric_tensor_create(&tensor, GEOMETRIC_TENSOR_SCALAR, dimensions, 3);
    
    qgt_error_t err = geometric_tensor_validate(tensor);
    assert(err == QGT_SUCCESS);
    
    // Test invalid state
    free(tensor->components);
    tensor->components = NULL;
    err = geometric_tensor_validate(tensor);
    assert(err == QGT_ERROR_INVALID_STATE);
    
    free(tensor->dimensions);
    free(tensor);
    printf("✓ Tensor validation test passed\n");
}

static void test_tensor_contraction() {
    quantum_geometric_tensor_t *a, *b, *result;
    size_t dims_a[] = {2, 3};
    size_t dims_b[] = {3, 2};
    size_t dims_result[] = {2, 2};
    
    geometric_tensor_create(&a, GEOMETRIC_TENSOR_SCALAR, dims_a, 2);
    geometric_tensor_create(&b, GEOMETRIC_TENSOR_SCALAR, dims_b, 2);
    geometric_tensor_create(&result, GEOMETRIC_TENSOR_SCALAR, dims_result, 2);
    
    // Initialize test data
    ComplexFloat data_a[] = {
        {1.0f, 0.0f}, {2.0f, 0.0f}, {3.0f, 0.0f},
        {4.0f, 0.0f}, {5.0f, 0.0f}, {6.0f, 0.0f}
    };
    ComplexFloat data_b[] = {
        {7.0f, 0.0f}, {8.0f, 0.0f},
        {9.0f, 0.0f}, {10.0f, 0.0f},
        {11.0f, 0.0f}, {12.0f, 0.0f}
    };
    
    memcpy(a->components, data_a, 6 * sizeof(ComplexFloat));
    memcpy(b->components, data_b, 6 * sizeof(ComplexFloat));
    
    // Test contraction
    size_t indices_a[] = {1};
    size_t indices_b[] = {0};
    qgt_error_t err = geometric_tensor_contract(result, a, b, indices_a, indices_b, 1);
    assert(err == QGT_SUCCESS);
    
    // Verify contraction result
    assert(fabsf(result->components[0].real - 58.0f) < EPSILON);
    assert(fabsf(result->components[1].real - 64.0f) < EPSILON);
    assert(fabsf(result->components[2].real - 139.0f) < EPSILON);
    assert(fabsf(result->components[3].real - 154.0f) < EPSILON);
    
    geometric_tensor_destroy(a);
    geometric_tensor_destroy(b);
    geometric_tensor_destroy(result);
    printf("✓ Tensor contraction test passed\n");
}

int main() {
    printf("Running quantum geometric tensor tests...\n");
    
    test_tensor_create();
    test_tensor_operations();
    test_tensor_validation();
    test_tensor_contraction();
    
    printf("All tests passed!\n");
    return 0;
}
