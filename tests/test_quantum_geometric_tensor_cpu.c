#include "quantum_geometric/core/quantum_geometric_tensor.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_complex.h"
#include "quantum_geometric/core/quantum_geometric_operations.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/core/quantum_state.h"
#include "quantum_geometric/core/error_codes.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <assert.h>

// Helper function to initialize a geometric tensor for testing
static void init_test_tensor(quantum_geometric_tensor_t** tensor, size_t dim) {
    size_t dimensions[] = {dim, dim};
    qgt_error_t err = geometric_tensor_create(tensor, GEOMETRIC_TENSOR_SCALAR, dimensions, 2);
    assert(err == QGT_SUCCESS && "Failed to create geometric tensor");
    
    err = geometric_tensor_initialize_identity(*tensor);
    assert(err == QGT_SUCCESS && "Failed to initialize tensor");
}

// Test geometric tensor creation and initialization
static void test_geometric_tensor_creation() {
    printf("Testing geometric tensor creation...\n");
    
    quantum_geometric_tensor_t* tensor = NULL;
    size_t dimensions[] = {2, 2};  // 2x2 tensor
    
    // Create tensor
    qgt_error_t err = geometric_tensor_create(&tensor, GEOMETRIC_TENSOR_SCALAR, dimensions, 2);
    assert(err == QGT_SUCCESS && "Failed to create geometric tensor");
    assert(tensor != NULL && "Tensor is NULL");
    assert(tensor->rank == 2 && "Incorrect tensor rank");
    assert(tensor->dimensions[0] == 2 && tensor->dimensions[1] == 2 && "Incorrect tensor dimensions");
    
    // Initialize with identity
    err = geometric_tensor_initialize_identity(tensor);
    assert(err == QGT_SUCCESS && "Failed to initialize tensor");
    
    // Verify identity initialization
    assert(tensor->components[0].real == 1.0f && tensor->components[0].imag == 0.0f);
    assert(tensor->components[3].real == 1.0f && tensor->components[3].imag == 0.0f);
    assert(tensor->components[1].real == 0.0f && tensor->components[1].imag == 0.0f);
    assert(tensor->components[2].real == 0.0f && tensor->components[2].imag == 0.0f);
    
    geometric_tensor_destroy(tensor);
    printf("Geometric tensor creation test passed\n");
}

// Test geometric metric computation
static void test_geometric_metric() {
    printf("Testing geometric metric computation...\n");
    
    // Create a simple 2x2 tensor
    const size_t dim = 2;
    quantum_geometric_tensor_t* tensor = NULL;
    init_test_tensor(&tensor, dim);
    
    // Create quantum state
    quantum_state_t* state = NULL;
    qgt_error_t err = quantum_state_create(&state, QUANTUM_STATE_PURE, dim);
    assert(err == QGT_SUCCESS && "Failed to create quantum state");
    
    // Initialize state with identity
    ComplexFloat amplitudes[2] = {{1.0f, 0.0f}, {0.0f, 0.0f}};
    err = quantum_state_initialize(state, amplitudes);
    assert(err == QGT_SUCCESS && "Failed to initialize quantum state");
    
    // Create and compute metric
    quantum_geometric_metric_t* metric = NULL;
    err = geometric_create_metric(&metric, GEOMETRIC_METRIC_EUCLIDEAN, dim);
    assert(err == QGT_SUCCESS && "Failed to create geometric metric");
    
    err = geometric_compute_metric(metric, state);
    assert(err == QGT_SUCCESS && "Failed to compute metric");
    
    // Verify metric is identity for Euclidean space
    for (size_t i = 0; i < dim * dim; i++) {
        if (i % (dim + 1) == 0) {
            assert(fabs(metric->components[i].real - 1.0) < 1e-6 && "Diagonal elements should be 1");
            assert(fabs(metric->components[i].imag) < 1e-6 && "Diagonal elements should be real");
        } else {
            assert(fabs(metric->components[i].real) < 1e-6 && "Off-diagonal elements should be 0");
            assert(fabs(metric->components[i].imag) < 1e-6 && "Off-diagonal elements should be 0");
        }
    }
    
    // Cleanup
    geometric_destroy_metric(metric);
    geometric_tensor_destroy(tensor);
    quantum_state_destroy(state);
    
    printf("Geometric metric test passed\n");
}

// Test geometric connection computation
static void test_geometric_connection() {
    printf("Testing geometric connection computation...\n");
    
    // Create a simple 2x2 tensor
    const size_t dim = 2;
    quantum_geometric_tensor_t* tensor = NULL;
    init_test_tensor(&tensor, dim);
    
    // Create quantum state
    quantum_state_t* state = NULL;
    qgt_error_t err = quantum_state_create(&state, QUANTUM_STATE_PURE, dim);
    assert(err == QGT_SUCCESS && "Failed to create quantum state");
    
    // Initialize state with identity
    ComplexFloat amplitudes[2] = {{1.0f, 0.0f}, {0.0f, 0.0f}};
    err = quantum_state_initialize(state, amplitudes);
    assert(err == QGT_SUCCESS && "Failed to initialize quantum state");
    
    // Create and compute metric
    quantum_geometric_metric_t* metric = NULL;
    err = geometric_create_metric(&metric, GEOMETRIC_METRIC_EUCLIDEAN, dim);
    assert(err == QGT_SUCCESS && "Failed to create geometric metric");
    
    err = geometric_compute_metric(metric, state);
    assert(err == QGT_SUCCESS && "Failed to compute metric");
    
    // Create and compute connection
    quantum_geometric_connection_t* connection = NULL;
    err = geometric_create_connection(&connection, GEOMETRIC_CONNECTION_LEVI_CIVITA, dim);
    assert(err == QGT_SUCCESS && "Failed to create geometric connection");
    
    err = geometric_compute_connection(connection, metric);
    assert(err == QGT_SUCCESS && "Failed to compute connection");
    
    // Verify connection coefficients are zero for flat space
    for (size_t i = 0; i < dim * dim * dim; i++) {
        assert(fabs(connection->coefficients[i].real) < 1e-6 && "Connection should be 0");
        assert(fabs(connection->coefficients[i].imag) < 1e-6 && "Connection should be 0");
    }
    
    // Cleanup
    geometric_destroy_connection(connection);
    geometric_destroy_metric(metric);
    geometric_tensor_destroy(tensor);
    quantum_state_destroy(state);
    
    printf("Geometric connection test passed\n");
}

// Test geometric curvature computation
static void test_geometric_curvature() {
    printf("Testing geometric curvature computation...\n");
    
    // Create a simple 2x2 tensor
    const size_t dim = 2;
    quantum_geometric_tensor_t* tensor = NULL;
    init_test_tensor(&tensor, dim);
    
    // Create quantum state
    quantum_state_t* state = NULL;
    qgt_error_t err = quantum_state_create(&state, QUANTUM_STATE_PURE, dim);
    assert(err == QGT_SUCCESS && "Failed to create quantum state");
    
    // Initialize state with identity
    ComplexFloat amplitudes[2] = {{1.0f, 0.0f}, {0.0f, 0.0f}};
    err = quantum_state_initialize(state, amplitudes);
    assert(err == QGT_SUCCESS && "Failed to initialize quantum state");
    
    // Create and compute metric
    quantum_geometric_metric_t* metric = NULL;
    err = geometric_create_metric(&metric, GEOMETRIC_METRIC_EUCLIDEAN, dim);
    assert(err == QGT_SUCCESS && "Failed to create geometric metric");
    
    err = geometric_compute_metric(metric, state);
    assert(err == QGT_SUCCESS && "Failed to compute metric");
    
    // Create and compute connection
    quantum_geometric_connection_t* connection = NULL;
    err = geometric_create_connection(&connection, GEOMETRIC_CONNECTION_LEVI_CIVITA, dim);
    assert(err == QGT_SUCCESS && "Failed to create geometric connection");
    
    err = geometric_compute_connection(connection, metric);
    assert(err == QGT_SUCCESS && "Failed to compute connection");
    
    // Create and compute curvature
    quantum_geometric_curvature_t* curvature = NULL;
    err = geometric_create_curvature(&curvature, GEOMETRIC_CURVATURE_RIEMANN, dim);
    assert(err == QGT_SUCCESS && "Failed to create geometric curvature");
    
    err = geometric_compute_curvature(curvature, connection);
    assert(err == QGT_SUCCESS && "Failed to compute curvature");
    
    // Verify curvature is zero for flat space
    assert(curvature->is_flat && "Space should be flat");
    for (size_t i = 0; i < dim * dim * dim * dim; i++) {
        assert(fabs(curvature->components[i].real) < 1e-6 && "Curvature should be 0");
        assert(fabs(curvature->components[i].imag) < 1e-6 && "Curvature should be 0");
    }
    
    // Cleanup
    geometric_destroy_curvature(curvature);
    geometric_destroy_connection(connection);
    geometric_destroy_metric(metric);
    geometric_tensor_destroy(tensor);
    quantum_state_destroy(state);
    
    printf("Geometric curvature test passed\n");
}

// Test geometric phase computation
static void test_geometric_phase() {
    printf("Testing geometric phase computation...\n");
    
    // Create a simple 2x2 tensor
    const size_t dim = 2;
    quantum_geometric_tensor_t* tensor = NULL;
    init_test_tensor(&tensor, dim);
    
    // Create quantum state
    quantum_state_t* state = NULL;
    qgt_error_t err = quantum_state_create(&state, QUANTUM_STATE_PURE, dim);
    assert(err == QGT_SUCCESS && "Failed to create quantum state");
    
    // Initialize state with identity
    ComplexFloat amplitudes[2] = {{1.0f, 0.0f}, {0.0f, 0.0f}};
    err = quantum_state_initialize(state, amplitudes);
    assert(err == QGT_SUCCESS && "Failed to initialize quantum state");
    
    // Create and compute metric
    quantum_geometric_metric_t* metric = NULL;
    err = geometric_create_metric(&metric, GEOMETRIC_METRIC_EUCLIDEAN, dim);
    assert(err == QGT_SUCCESS && "Failed to create geometric metric");
    
    err = geometric_compute_metric(metric, state);
    assert(err == QGT_SUCCESS && "Failed to compute metric");
    
    // Create and compute connection
    quantum_geometric_connection_t* connection = NULL;
    err = geometric_create_connection(&connection, GEOMETRIC_CONNECTION_LEVI_CIVITA, dim);
    assert(err == QGT_SUCCESS && "Failed to create geometric connection");
    
    err = geometric_compute_connection(connection, metric);
    assert(err == QGT_SUCCESS && "Failed to compute connection");
    
    // Compute geometric phase
    ComplexFloat phase;
    err = geometric_compute_phase(&phase, state, connection);
    assert(err == QGT_SUCCESS && "Failed to compute geometric phase");
    
    // Verify phase is trivial for flat space
    assert(fabs(phase.real - 1.0) < 1e-6 && fabs(phase.imag) < 1e-6 && 
           "Phase should be trivial in flat space");
    
    // Cleanup
    geometric_destroy_connection(connection);
    geometric_destroy_metric(metric);
    geometric_tensor_destroy(tensor);
    quantum_state_destroy(state);
    
    printf("Geometric phase test passed\n");
}

int main() {
    printf("Running quantum geometric tensor CPU tests...\n");
    
    // Initialize geometric core
    qgt_error_t err = geometric_core_initialize();
    assert(err == QGT_SUCCESS && "Failed to initialize geometric core");
    
    test_geometric_tensor_creation();
    test_geometric_metric();
    test_geometric_connection();
    test_geometric_curvature();
    test_geometric_phase();
    
    // Shutdown geometric core
    geometric_core_shutdown();
    
    printf("All tests passed!\n");
    return 0;
}
