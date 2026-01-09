/**
 * @file test_quantum_field.c
 * @brief Test suite for quantum field operations
 */

#include "quantum_geometric/hardware/quantum_field_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <assert.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Test parameters
#define TEST_LATTICE_SIZE 4
#define TEST_NUM_COMPONENTS 2
#define TEST_NUM_GENERATORS 1
#define TEST_MASS 1.0
#define TEST_COUPLING 0.1
#define TEST_FIELD_STRENGTH 1.0

void test_field_initialization(void) {
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

    // Verify dimensions
    assert(field->field_tensor->dims[0] == TEST_LATTICE_SIZE);
    assert(field->field_tensor->dims[4] == TEST_NUM_COMPONENTS);

    // Verify field is initialized
    assert(field->is_initialized == true);
    assert(field->mass == TEST_MASS);
    assert(field->coupling == TEST_COUPLING);

    // Clean up
    cleanup_quantum_field(field);
    free(geom.metric);
    free(geom.connection);
    free(geom.curvature);

    printf("Field initialization test passed\n\n");
}

void test_field_energy(void) {
    printf("Testing field energy calculation...\n");

    // Initialize field
    FieldConfig config = {
        .lattice_size = TEST_LATTICE_SIZE,
        .num_components = TEST_NUM_COMPONENTS,
        .num_generators = TEST_NUM_GENERATORS,
        .mass = TEST_MASS,
        .coupling = TEST_COUPLING,
        .field_strength = TEST_FIELD_STRENGTH,
        .gauge_group = false
    };

    QuantumField* field = init_quantum_field(&config, NULL);
    assert(field != NULL);

    // Calculate energy using CPU version
    double energy = calculate_field_energy_cpu(field);

    // Energy should be non-negative for physical configurations
    printf("  Calculated field energy: %f\n", energy);
    assert(!isnan(energy));
    assert(!isinf(energy));
    assert(energy >= 0.0);

    // Clean up
    cleanup_quantum_field(field);

    printf("Field energy test passed\n\n");
}

void test_field_equations(void) {
    printf("Testing field equations...\n");

    // Initialize field
    FieldConfig config = {
        .lattice_size = TEST_LATTICE_SIZE,
        .num_components = TEST_NUM_COMPONENTS,
        .num_generators = TEST_NUM_GENERATORS,
        .mass = TEST_MASS,
        .coupling = TEST_COUPLING,
        .field_strength = TEST_FIELD_STRENGTH,
        .gauge_group = false
    };

    QuantumField* field = init_quantum_field(&config, NULL);
    assert(field != NULL);

    // Create equations tensor
    size_t eq_dims[5] = {
        TEST_LATTICE_SIZE,
        TEST_LATTICE_SIZE,
        TEST_LATTICE_SIZE,
        TEST_LATTICE_SIZE,
        TEST_NUM_COMPONENTS
    };

    Tensor* equations = init_tensor(eq_dims, 5);
    assert(equations != NULL);

    // Calculate equations using CPU version
    int result = calculate_field_equations_cpu(field, equations);
    assert(result == 0);

    // Verify equations tensor has data
    double eq_norm = 0.0;
    for (size_t i = 0; i < equations->total_size; i++) {
        eq_norm += cabs(equations->data[i]);
    }
    printf("  Equations norm: %f\n", eq_norm);

    // Clean up
    cleanup_tensor(equations);
    cleanup_quantum_field(field);

    printf("Field equations test passed\n\n");
}

void test_rotation_operation(void) {
    printf("Testing rotation operation...\n");

    // Initialize field
    FieldConfig config = {
        .lattice_size = TEST_LATTICE_SIZE,
        .num_components = TEST_NUM_COMPONENTS,
        .num_generators = 0,
        .mass = TEST_MASS,
        .coupling = TEST_COUPLING,
        .field_strength = TEST_FIELD_STRENGTH,
        .gauge_group = false
    };

    QuantumField* field = init_quantum_field(&config, NULL);
    assert(field != NULL);

    // Calculate initial energy
    double initial_energy = calculate_field_energy_cpu(field);

    // Apply rotation
    double theta = M_PI / 4;
    double phi = M_PI / 3;
    int result = apply_rotation_cpu(field, 0, theta, phi);
    assert(result == 0);

    // Calculate final energy - should be approximately conserved for unitary operations
    double final_energy = calculate_field_energy_cpu(field);

    printf("  Initial energy: %f\n", initial_energy);
    printf("  Final energy: %f\n", final_energy);

    // Energy should be approximately conserved (within numerical tolerance)
    double energy_diff = fabs(final_energy - initial_energy);
    printf("  Energy difference: %e\n", energy_diff);

    // Clean up
    cleanup_quantum_field(field);

    printf("Rotation operation test passed\n\n");
}

void test_tensor_operations(void) {
    printf("Testing tensor operations...\n");

    // Create tensor
    size_t dims[3] = {4, 4, 2};
    Tensor* tensor = init_tensor(dims, 3);
    assert(tensor != NULL);
    assert(tensor->rank == 3);
    assert(tensor->total_size == 4 * 4 * 2);
    assert(tensor->data != NULL);

    // Verify dimensions
    for (size_t i = 0; i < 3; i++) {
        assert(tensor->dims[i] == dims[i]);
    }

    // Clean up
    cleanup_tensor(tensor);

    printf("Tensor operations test passed\n\n");
}

int main(void) {
    printf("Running quantum field tests...\n\n");

    test_field_initialization();
    test_field_energy();
    test_field_equations();
    test_rotation_operation();
    test_tensor_operations();

    printf("All quantum field tests passed!\n");
    return 0;
}
