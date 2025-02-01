#include "quantum_geometric/physics/quantum_field_operations.h"
#include "quantum_geometric/physics/quantum_field_calculations.h"
#include "quantum_geometric/physics/quantum_field_helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

// Test parameters
#define TEST_LATTICE_SIZE 4
#define TEST_NUM_COMPONENTS 2
#define TEST_NUM_GENERATORS 1
#define TEST_MASS 1.0
#define TEST_COUPLING 0.1
#define TEST_FIELD_STRENGTH 1.0

void test_field_initialization() {
    printf("Testing field initialization...\n");
    
    // Initialize configurations
    FieldConfig config = {
        .lattice_size = TEST_LATTICE_SIZE,
        .num_components = TEST_NUM_COMPONENTS,
        .num_generators = TEST_NUM_GENERATORS,
        .mass = TEST_MASS,
        .coupling = TEST_COUPLING,
        .field_strength = TEST_FIELD_STRENGTH,
        .gauge_group = true
    };
    
    GeometricConfig geom = {
        .metric = NULL,
        .connection = NULL,
        .curvature = NULL
    };
    
    // Allocate geometric tensors
    size_t metric_size = SPACETIME_DIMS * SPACETIME_DIMS;
    geom.metric = calloc(metric_size, sizeof(double));
    geom.connection = calloc(metric_size * SPACETIME_DIMS, sizeof(double));
    geom.curvature = calloc(metric_size * metric_size, sizeof(double));
    
    // Set Minkowski metric
    for (size_t i = 0; i < SPACETIME_DIMS; i++) {
        geom.metric[i * SPACETIME_DIMS + i] = (i == 0) ? -1.0 : 1.0;
    }
    
    // Initialize field
    QuantumField* field = init_quantum_field(&config, &geom);
    assert(field != NULL);
    assert(field->field_tensor != NULL);
    assert(field->conjugate_momentum != NULL);
    assert(field->gauge_field != NULL);
    assert(field->gauge_generators != NULL);
    
    // Verify dimensions
    assert(field->field_tensor->dims[0] == TEST_LATTICE_SIZE);
    assert(field->field_tensor->dims[4] == TEST_NUM_COMPONENTS);
    assert(field->gauge_field->dims[4] == TEST_NUM_GENERATORS);
    
    // Clean up
    cleanup_quantum_field(field);
    free(geom.metric);
    free(geom.connection);
    free(geom.curvature);
    
    printf("Field initialization test passed\n");
}

void test_gauge_transformation() {
    printf("Testing gauge transformation...\n");
    
    // Initialize field
    FieldConfig config = {
        .lattice_size = TEST_LATTICE_SIZE,
        .num_components = TEST_NUM_COMPONENTS,
        .num_generators = TEST_NUM_GENERATORS,
        .mass = TEST_MASS,
        .coupling = TEST_COUPLING,
        .field_strength = TEST_FIELD_STRENGTH,
        .gauge_group = true
    };
    
    GeometricConfig geom = {
        .metric = calloc(SPACETIME_DIMS * SPACETIME_DIMS, sizeof(double)),
        .connection = calloc(SPACETIME_DIMS * SPACETIME_DIMS * SPACETIME_DIMS, sizeof(double)),
        .curvature = calloc(SPACETIME_DIMS * SPACETIME_DIMS * SPACETIME_DIMS * SPACETIME_DIMS, sizeof(double))
    };
    
    // Set Minkowski metric
    for (size_t i = 0; i < SPACETIME_DIMS; i++) {
        geom.metric[i * SPACETIME_DIMS + i] = (i == 0) ? -1.0 : 1.0;
    }
    
    QuantumField* field = init_quantum_field(&config, &geom);
    
    // Create gauge transformation
    size_t trans_dims[2] = {TEST_NUM_COMPONENTS, TEST_NUM_COMPONENTS};
    Tensor* transformation = init_tensor(trans_dims, 2);
    
    // Set SU(2) rotation
    double theta = M_PI / 4;
    transformation->data[0] = cos(theta) + 0.0*I;
    transformation->data[1] = -sin(theta) + 0.0*I;
    transformation->data[2] = sin(theta) + 0.0*I;
    transformation->data[3] = cos(theta) + 0.0*I;
    
    // Calculate initial energy
    double initial_energy = calculate_field_energy(field);
    
    // Apply transformation
    int result = apply_gauge_transformation(field, transformation);
    assert(result == 0);
    
    // Calculate final energy
    double final_energy = calculate_field_energy(field);
    
    // Verify gauge invariance
    assert(fabs(final_energy - initial_energy) < 1e-10);
    
    // Clean up
    cleanup_tensor(transformation);
    cleanup_quantum_field(field);
    free(geom.metric);
    free(geom.connection);
    free(geom.curvature);
    
    printf("Gauge transformation test passed\n");
}

void test_field_equations() {
    printf("Testing field equations...\n");
    
    // Initialize field
    FieldConfig config = {
        .lattice_size = TEST_LATTICE_SIZE,
        .num_components = TEST_NUM_COMPONENTS,
        .num_generators = TEST_NUM_GENERATORS,
        .mass = TEST_MASS,
        .coupling = TEST_COUPLING,
        .field_strength = TEST_FIELD_STRENGTH,
        .gauge_group = true
    };
    
    GeometricConfig geom = {
        .metric = calloc(SPACETIME_DIMS * SPACETIME_DIMS, sizeof(double)),
        .connection = calloc(SPACETIME_DIMS * SPACETIME_DIMS * SPACETIME_DIMS, sizeof(double)),
        .curvature = calloc(SPACETIME_DIMS * SPACETIME_DIMS * SPACETIME_DIMS * SPACETIME_DIMS, sizeof(double))
    };
    
    // Set Minkowski metric
    for (size_t i = 0; i < SPACETIME_DIMS; i++) {
        geom.metric[i * SPACETIME_DIMS + i] = (i == 0) ? -1.0 : 1.0;
    }
    
    QuantumField* field = init_quantum_field(&config, &geom);
    
    // Create equations tensor
    size_t eq_dims[5] = {
        TEST_LATTICE_SIZE,
        TEST_LATTICE_SIZE,
        TEST_LATTICE_SIZE,
        TEST_LATTICE_SIZE,
        TEST_NUM_COMPONENTS
    };
    
    Tensor* equations = init_tensor(eq_dims, 5);
    
    // Calculate equations
    int result = calculate_field_equations(field, equations);
    assert(result == 0);
    
    // Verify equations are non-zero
    double eq_norm = 0.0;
    for (size_t i = 0; i < equations->size; i++) {
        eq_norm += cabs(equations->data[i]);
    }
    assert(eq_norm > 0.0);
    
    // Clean up
    cleanup_tensor(equations);
    cleanup_quantum_field(field);
    free(geom.metric);
    free(geom.connection);
    free(geom.curvature);
    
    printf("Field equations test passed\n");
}

int main() {
    printf("Running quantum field tests...\n\n");
    
    test_field_initialization();
    test_gauge_transformation();
    test_field_equations();
    
    printf("\nAll tests passed!\n");
    return 0;
}
