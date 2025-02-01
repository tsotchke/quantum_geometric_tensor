#ifndef TEST_QUANTUM_GEOMETRIC_CORE_H
#define TEST_QUANTUM_GEOMETRIC_CORE_H

#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_geometric_metric.h"
#include "quantum_geometric/core/quantum_geometric_connection.h"
#include "quantum_geometric/core/quantum_geometric_curvature.h"
#include "quantum_geometric/core/quantum_geometric_optimization.h"
#include "quantum_geometric/core/quantum_geometric_validation.h"
#include "quantum_geometric/core/quantum_geometric_simulation.h"
#include "quantum_geometric/core/quantum_geometric_hardware.h"
#include "quantum_geometric/core/quantum_geometric_error.h"
#include "quantum_geometric/core/quantum_geometric_memory.h"
#include "quantum_geometric/core/quantum_geometric_profiling.h"
#include "quantum_geometric/core/quantum_geometric_logging.h"
#include "quantum_geometric/core/quantum_geometric_config.h"

// Example usage demonstrating proper initialization sequence
void test_initialization(void) {
    qgt_error_t status;
    
    // Initialize core subsystems in correct order
    status = geometric_init_logging("quantum_geometric.log");
    assert(status == QGT_SUCCESS);
    
    status = geometric_init_config("quantum_geometric.conf");
    assert(status == QGT_SUCCESS);
    
    status = geometric_init_memory();
    assert(status == QGT_SUCCESS);
    
    status = geometric_init_hardware();
    assert(status == QGT_SUCCESS);
    
    // Configure system
    quantum_geometric_config_t config = {
        .dimensions = 4,
        .precision = 1e-6,
        .use_gpu = false,
        .device_name = NULL
    };
    status = geometric_configure(&config);
    assert(status == QGT_SUCCESS);
}

// Example demonstrating metric tensor operations
void test_metric_operations(void) {
    qgt_error_t status;
    quantum_geometric_metric metric;
    
    // Create metric tensor
    status = geometric_create_metric(4, &metric);
    assert(status == QGT_SUCCESS);
    
    // Set components
    double components[16] = {
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    };
    status = geometric_set_metric_components(metric, components);
    assert(status == QGT_SUCCESS);
    
    // Calculate determinant
    double det;
    status = geometric_metric_determinant(metric, &det);
    assert(status == QGT_SUCCESS);
    assert(det == 1.0);
    
    // Clean up
    geometric_destroy_metric(metric);
}

// Example demonstrating connection operations
void test_connection_operations(void) {
    qgt_error_t status;
    quantum_geometric_connection connection;
    quantum_geometric_metric metric;
    
    // Create connection from metric
    status = geometric_create_connection_from_metric(metric, &connection);
    assert(status == QGT_SUCCESS);
    
    // Calculate Christoffel symbols
    double symbols[64];  // 4^3 components for 4D
    status = geometric_calculate_christoffel_symbols(connection, symbols);
    assert(status == QGT_SUCCESS);
    
    // Clean up
    geometric_destroy_connection(connection);
}

// Example demonstrating curvature operations
void test_curvature_operations(void) {
    qgt_error_t status;
    quantum_geometric_curvature curvature;
    quantum_geometric_connection connection;
    
    // Create curvature from connection
    status = geometric_create_curvature_from_connection(connection, &curvature);
    assert(status == QGT_SUCCESS);
    
    // Calculate Riemann tensor
    double riemann[256];  // 4^4 components for 4D
    status = geometric_calculate_riemann_tensor(curvature, riemann);
    assert(status == QGT_SUCCESS);
    
    // Calculate Ricci scalar
    double ricci_scalar;
    status = geometric_calculate_ricci_scalar(curvature, &ricci_scalar);
    assert(status == QGT_SUCCESS);
    
    // Clean up
    geometric_destroy_curvature(curvature);
}

// Example demonstrating proper cleanup sequence
void test_cleanup(void) {
    geometric_cleanup_hardware();
    geometric_cleanup_memory();
    geometric_cleanup_config();
    geometric_cleanup_logging();
}

#endif // TEST_QUANTUM_GEOMETRIC_CORE_H
