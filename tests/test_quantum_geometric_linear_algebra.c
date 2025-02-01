#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_geometric_constants.h"
#include "quantum_geometric/core/error_codes.h"
#include <assert.h>
#include <stdio.h>
#include <math.h>

// Test matrix multiplication
static void test_matrix_multiply(void) {
    qgt_error_t status;
    
    // Initialize core
    status = geometric_core_initialize();
    assert(status == QGT_SUCCESS);
    
    // Test 2x2 matrix multiplication
    float a[4] = {1.0f, 2.0f,
                  3.0f, 4.0f};
    float b[4] = {5.0f, 6.0f,
                  7.0f, 8.0f};
    float result[4];
    
    status = geometric_core_matrix_multiply(result, a, b, 2, 2, 2);
    assert(status == QGT_SUCCESS);
    
    // Expected result:
    // [19 22]
    // [43 50]
    assert(fabs(result[0] - 19.0f) < 1e-6f);
    assert(fabs(result[1] - 22.0f) < 1e-6f);
    assert(fabs(result[2] - 43.0f) < 1e-6f);
    assert(fabs(result[3] - 50.0f) < 1e-6f);
    
    geometric_core_shutdown();
}

// Test matrix transpose
static void test_matrix_transpose(void) {
    qgt_error_t status;
    
    status = geometric_core_initialize();
    assert(status == QGT_SUCCESS);
    
    // Test 2x3 matrix transpose
    float a[6] = {1.0f, 2.0f, 3.0f,
                  4.0f, 5.0f, 6.0f};
    float result[6];
    
    status = geometric_core_matrix_transpose(result, a, 2, 3);
    assert(status == QGT_SUCCESS);
    
    // Expected result:
    // [1 4]
    // [2 5]
    // [3 6]
    assert(fabs(result[0] - 1.0f) < 1e-6f);
    assert(fabs(result[1] - 4.0f) < 1e-6f);
    assert(fabs(result[2] - 2.0f) < 1e-6f);
    assert(fabs(result[3] - 5.0f) < 1e-6f);
    assert(fabs(result[4] - 3.0f) < 1e-6f);
    assert(fabs(result[5] - 6.0f) < 1e-6f);
    
    geometric_core_shutdown();
}

// Test matrix inverse
static void test_matrix_inverse(void) {
    qgt_error_t status;
    
    status = geometric_core_initialize();
    assert(status == QGT_SUCCESS);
    
    // Test 2x2 matrix inverse
    float a[4] = {4.0f, 7.0f,
                  2.0f, 6.0f};
    float result[4];
    
    status = geometric_core_matrix_inverse(result, a, 2);
    assert(status == QGT_SUCCESS);
    
    // Expected result:
    // [ 0.6 -0.7]
    // [-0.2  0.4]
    assert(fabs(result[0] - 0.6f) < 1e-6f);
    assert(fabs(result[1] + 0.7f) < 1e-6f);
    assert(fabs(result[2] + 0.2f) < 1e-6f);
    assert(fabs(result[3] - 0.4f) < 1e-6f);
    
    // Test singular matrix
    float singular[4] = {1.0f, 2.0f,
                        2.0f, 4.0f};
    status = geometric_core_matrix_inverse(result, singular, 2);
    assert(status == QGT_ERROR_INVALID_ARGUMENT);
    
    geometric_core_shutdown();
}

// Test linear system solver
static void test_linear_system(void) {
    qgt_error_t status;
    
    status = geometric_core_initialize();
    assert(status == QGT_SUCCESS);
    
    // Test 2x2 system Ax = b
    float a[4] = {3.0f, 2.0f,
                  1.0f, 1.0f};
    float b[2] = {7.0f, 3.0f};
    float x[2];
    
    status = geometric_core_solve_linear_system(x, a, b, 2);
    assert(status == QGT_SUCCESS);
    
    // Expected solution: x = [1, 2]
    assert(fabs(x[0] - 1.0f) < 1e-6f);
    assert(fabs(x[1] - 2.0f) < 1e-6f);
    
    geometric_core_shutdown();
}

// Test tensor operations
static void test_tensor_operations(void) {
    qgt_error_t status;
    
    status = geometric_core_initialize();
    assert(status == QGT_SUCCESS);
    
    // Test tensor contraction
    float a[6] = {1.0f, 2.0f, 3.0f,
                  4.0f, 5.0f, 6.0f};
    float b[6] = {7.0f, 8.0f, 9.0f,
                  10.0f, 11.0f, 12.0f};
    float result[4];
    
    size_t dims_a[2] = {2, 3};
    size_t dims_b[2] = {2, 3};
    size_t contract_a[1] = {1};
    size_t contract_b[1] = {1};
    
    status = geometric_core_tensor_contract(result, a, b, dims_a, dims_b, 2, 2, contract_a, contract_b, 1);
    assert(status == QGT_SUCCESS);
    
    // Test tensor decomposition (SVD)
    float tensor[6] = {1.0f, 2.0f, 3.0f,
                      4.0f, 5.0f, 6.0f};
    float u[4], s[2], v[6];
    size_t dims[2] = {2, 3};
    
    status = geometric_core_tensor_decompose(u, s, v, tensor, dims, 2);
    assert(status == QGT_SUCCESS);
    
    geometric_core_shutdown();
}

int main(void) {
    printf("Running quantum geometric linear algebra tests...\n");
    
    test_matrix_multiply();
    test_matrix_transpose();
    test_matrix_inverse();
    test_linear_system();
    test_tensor_operations();
    
    printf("All linear algebra tests passed!\n");
    return 0;
}
