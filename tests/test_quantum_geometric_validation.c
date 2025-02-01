#include "quantum_geometric/core/quantum_geometric_validation.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/error_codes.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

// Test validation of metric
static void test_metric_validation(void) {
    quantum_geometric_metric_t metric;
    validation_result_t result;
    
    // Initialize metric
    metric.dimension = 2;
    metric.components = (ComplexFloat*)malloc(4 * sizeof(ComplexFloat));
    
    // Test symmetric metric
    metric.components[0] = (ComplexFloat){1.0f, 0.0f};  // g_00
    metric.components[1] = (ComplexFloat){0.5f, 0.0f};  // g_01
    metric.components[2] = (ComplexFloat){0.5f, 0.0f};  // g_10
    metric.components[3] = (ComplexFloat){1.0f, 0.0f};  // g_11
    
    qgt_error_t err = geometric_validate_metric(&metric, 
        GEOMETRIC_VALIDATION_CHECK_SYMMETRY | GEOMETRIC_VALIDATION_CHECK_POSITIVE_DEFINITE,
        &result);
    assert(err == QGT_SUCCESS);
    assert(result.is_valid == true);
    
    // Test asymmetric metric
    metric.components[1] = (ComplexFloat){0.5f, 0.0f};  // g_01
    metric.components[2] = (ComplexFloat){0.7f, 0.0f};  // g_10
    
    err = geometric_validate_metric(&metric, 
        GEOMETRIC_VALIDATION_CHECK_SYMMETRY | GEOMETRIC_VALIDATION_CHECK_POSITIVE_DEFINITE,
        &result);
    assert(err == QGT_SUCCESS);
    assert(result.is_valid == false);
    assert(result.error_code == QGT_ERROR_INVALID_METRIC);
    
    // Test non-positive definite metric
    metric.components[0] = (ComplexFloat){-1.0f, 0.0f};  // g_00
    
    err = geometric_validate_metric(&metric, 
        GEOMETRIC_VALIDATION_CHECK_SYMMETRY | GEOMETRIC_VALIDATION_CHECK_POSITIVE_DEFINITE,
        &result);
    assert(err == QGT_SUCCESS);
    assert(result.is_valid == false);
    assert(result.error_code == QGT_ERROR_INVALID_METRIC);
    
    free(metric.components);
    printf("✓ Metric validation test passed\n");
}

// Test validation of connection
static void test_connection_validation(void) {
    quantum_geometric_connection_t connection;
    validation_result_t result;
    
    // Initialize connection
    connection.dimension = 2;
    connection.coefficients = (ComplexFloat*)malloc(8 * sizeof(ComplexFloat));
    connection.is_compatible = true;
    
    // Test valid connection
    qgt_error_t err = geometric_validate_connection(&connection, 
        GEOMETRIC_VALIDATION_CHECK_COMPATIBILITY,
        &result);
    assert(err == QGT_SUCCESS);
    assert(result.is_valid == true);
    
    // Test incompatible connection
    connection.is_compatible = false;
    
    err = geometric_validate_connection(&connection, 
        GEOMETRIC_VALIDATION_CHECK_COMPATIBILITY,
        &result);
    assert(err == QGT_SUCCESS);
    assert(result.is_valid == false);
    assert(result.error_code == QGT_ERROR_INCOMPATIBLE);
    
    free(connection.coefficients);
    printf("✓ Connection validation test passed\n");
}

// Test validation of curvature
static void test_curvature_validation(void) {
    quantum_geometric_curvature_t curvature;
    validation_result_t result;
    
    // Initialize curvature
    curvature.dimension = 2;
    curvature.components = (ComplexFloat*)malloc(16 * sizeof(ComplexFloat));
    
    // Test valid curvature (satisfying first Bianchi identity)
    // R_ijkl + R_jkil + R_kijl = 0
    for (size_t i = 0; i < 16; i++) {
        curvature.components[i] = (ComplexFloat){0.0f, 0.0f};
    }
    
    qgt_error_t err = geometric_validate_curvature(&curvature, 
        GEOMETRIC_VALIDATION_CHECK_BIANCHI,
        &result);
    assert(err == QGT_SUCCESS);
    assert(result.is_valid == true);
    
    // Test invalid curvature (violating first Bianchi identity)
    curvature.components[0] = (ComplexFloat){1.0f, 0.0f};
    
    err = geometric_validate_curvature(&curvature, 
        GEOMETRIC_VALIDATION_CHECK_BIANCHI,
        &result);
    assert(err == QGT_SUCCESS);
    assert(result.is_valid == false);
    assert(result.error_code == QGT_ERROR_INVALID_CURVATURE);
    
    free(curvature.components);
    printf("✓ Curvature validation test passed\n");
}

// Test validation of optimization
static void test_optimization_validation(void) {
    geometric_optimization_t optimization;
    validation_result_t result;
    
    // Initialize optimization
    optimization.dimension = 2;
    optimization.type = GEOMETRIC_OPTIMIZATION_NEWTON;
    optimization.gradient = (ComplexFloat*)malloc(2 * sizeof(ComplexFloat));
    optimization.hessian = (ComplexFloat*)malloc(4 * sizeof(ComplexFloat));
    optimization.learning_rate = 0.01f;
    optimization.convergence_threshold = 1e-6f;
    optimization.max_iterations = 1000;
    
    // Test valid optimization
    qgt_error_t err = geometric_validate_optimization(&optimization, 
        GEOMETRIC_VALIDATION_CHECK_BOUNDS | GEOMETRIC_VALIDATION_CHECK_CONVERGENCE,
        &result);
    assert(err == QGT_SUCCESS);
    assert(result.is_valid == true);
    
    // Test invalid learning rate
    optimization.learning_rate = -0.01f;
    
    err = geometric_validate_optimization(&optimization, 
        GEOMETRIC_VALIDATION_CHECK_BOUNDS | GEOMETRIC_VALIDATION_CHECK_CONVERGENCE,
        &result);
    assert(err == QGT_SUCCESS);
    assert(result.is_valid == false);
    assert(result.error_code == QGT_ERROR_INVALID_PARAMETER);
    
    // Test missing Hessian for Newton method
    optimization.learning_rate = 0.01f;
    free(optimization.hessian);
    optimization.hessian = NULL;
    
    err = geometric_validate_optimization(&optimization, 
        GEOMETRIC_VALIDATION_CHECK_BOUNDS | GEOMETRIC_VALIDATION_CHECK_CONVERGENCE,
        &result);
    assert(err == QGT_SUCCESS);
    assert(result.is_valid == false);
    assert(result.error_code == QGT_ERROR_INVALID_STATE);
    
    free(optimization.gradient);
    printf("✓ Optimization validation test passed\n");
}

int main(void) {
    printf("Running quantum geometric validation tests...\n");
    
    test_metric_validation();
    test_connection_validation();
    test_curvature_validation();
    test_optimization_validation();
    
    printf("All validation tests passed!\n");
    return 0;
}
