/**
 * @file test_floquet_surface_code.c
 * @brief Test suite for Floquet surface code implementation
 */

#include "quantum_geometric/physics/floquet_surface_code.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

// Test configurations
static const size_t TEST_DISTANCE = 3;
static const size_t TEST_TIME_STEPS = 10;
static const double TEST_PERIOD = 1.0;
static const double TEST_COUPLING = 0.8;

// Helper function to create test configuration
static FloquetConfig create_test_config(void) {
    FloquetConfig config = {
        .distance = TEST_DISTANCE,
        .width = 0,  // Will be calculated
        .height = 0, // Will be calculated
        .time_steps = TEST_TIME_STEPS,
        .period = TEST_PERIOD,
        .use_boundary_stabilizers = true,
        .coupling_strength = TEST_COUPLING,
        .time_dependent_couplings = NULL
    };
    calculate_floquet_dimensions(config.distance, &config.width, &config.height);
    
    // Setup time-dependent couplings
    config.time_dependent_couplings = calloc(TEST_TIME_STEPS, sizeof(double));
    for (size_t t = 0; t < TEST_TIME_STEPS; t++) {
        config.time_dependent_couplings[t] = 0.1 * sin(2.0 * M_PI * t / TEST_TIME_STEPS);
    }
    
    return config;
}

// Test initialization
static void test_initialization(void) {
    printf("Testing Floquet lattice initialization...\n");

    FloquetConfig config = create_test_config();
    bool init_success = init_floquet_lattice(&config);
    assert(init_success);

    // Verify dimensions
    assert(config.width == TEST_DISTANCE);
    assert(config.height == TEST_DISTANCE);

    cleanup_floquet_lattice();
    free(config.time_dependent_couplings);
    printf("Floquet lattice initialization test passed\n");
}

// Test time evolution
static void test_time_evolution(void) {
    printf("Testing time evolution...\n");

    FloquetConfig config = create_test_config();
    bool init_success = init_floquet_lattice(&config);
    assert(init_success);

    // Test evolution operator between time steps
    size_t matrix_size = config.width * config.height;
    double* operator = calloc(matrix_size * matrix_size, sizeof(double));
    
    bool evolution_success = get_floquet_evolution_operator(0, TEST_TIME_STEPS/2,
                                                          operator, matrix_size * matrix_size);
    assert(evolution_success);

    // Verify operator properties
    double trace = 0.0;
    for (size_t i = 0; i < matrix_size; i++) {
        trace += operator[i * matrix_size + i];
    }
    assert(fabs(trace - matrix_size) < 1e-6); // Should be unitary

    free(operator);
    cleanup_floquet_lattice();
    free(config.time_dependent_couplings);
    printf("Time evolution test passed\n");
}

// Test coupling modulation
static void test_coupling_modulation(void) {
    printf("Testing coupling modulation...\n");

    FloquetConfig config = create_test_config();
    bool init_success = init_floquet_lattice(&config);
    assert(init_success);

    // Test coupling strength at different times
    size_t central_idx = (config.width * config.height) / 2;
    size_t qubits[4];
    size_t num_qubits = get_floquet_qubits(central_idx, 0, qubits, 4);
    assert(num_qubits > 0);

    // Verify coupling modulation
    for (size_t t = 0; t < TEST_TIME_STEPS; t++) {
        double coupling = get_floquet_coupling_strength(qubits[0], qubits[1], t);
        double expected = TEST_COUPLING * (1.0 + config.time_dependent_couplings[t]);
        assert(fabs(coupling - expected) < 1e-6);
    }

    cleanup_floquet_lattice();
    free(config.time_dependent_couplings);
    printf("Coupling modulation test passed\n");
}

// Test boundary conditions
static void test_boundary_conditions(void) {
    printf("Testing boundary conditions...\n");

    FloquetConfig config = create_test_config();
    config.use_boundary_stabilizers = true;
    bool init_success = init_floquet_lattice(&config);
    assert(init_success);

    // Test corner stabilizer at different times
    size_t corner_idx = 0;
    for (size_t t = 0; t < TEST_TIME_STEPS; t++) {
        assert(is_floquet_boundary_stabilizer(corner_idx, t));

        // Verify boundary qubit count
        size_t qubits[4];
        size_t num_boundary_qubits = get_floquet_qubits(corner_idx, t, qubits, 4);
        assert(num_boundary_qubits < 4); // Should have fewer qubits on boundary
    }

    cleanup_floquet_lattice();
    free(config.time_dependent_couplings);
    printf("Boundary conditions test passed\n");
}

// Test stabilizer types
static void test_stabilizer_types(void) {
    printf("Testing stabilizer types...\n");

    FloquetConfig config = create_test_config();
    bool init_success = init_floquet_lattice(&config);
    assert(init_success);

    // Test time-dependent stabilizer types
    size_t central_idx = (config.width * config.height) / 2;
    StabilizerType initial_type = get_floquet_stabilizer_type(central_idx, 0);

    for (size_t t = 1; t < TEST_TIME_STEPS; t++) {
        StabilizerType type = get_floquet_stabilizer_type(central_idx, t);
        double time = t * config.period / config.time_steps;
        bool should_flip = (sin(2.0 * M_PI * time / config.period) > 0);
        
        if (should_flip) {
            assert(type != initial_type);
        } else {
            assert(type == initial_type);
        }
    }

    cleanup_floquet_lattice();
    free(config.time_dependent_couplings);
    printf("Stabilizer types test passed\n");
}

// Test configuration validation
static void test_config_validation(void) {
    printf("Testing configuration validation...\n");

    // Test invalid distance
    FloquetConfig invalid_config = create_test_config();
    invalid_config.distance = 2; // Must be odd and >= 3
    bool init_result = init_floquet_lattice(&invalid_config);
    assert(!init_result);
    free(invalid_config.time_dependent_couplings);

    // Test invalid time steps
    invalid_config = create_test_config();
    invalid_config.time_steps = MAX_TIME_STEPS + 1;
    init_result = init_floquet_lattice(&invalid_config);
    assert(!init_result);
    free(invalid_config.time_dependent_couplings);

    // Test invalid period
    invalid_config = create_test_config();
    invalid_config.period = 0.0;
    init_result = init_floquet_lattice(&invalid_config);
    assert(!init_result);
    free(invalid_config.time_dependent_couplings);

    // Test valid config
    FloquetConfig valid_config = create_test_config();
    init_result = init_floquet_lattice(&valid_config);
    assert(init_result);

    cleanup_floquet_lattice();
    free(valid_config.time_dependent_couplings);
    printf("Configuration validation test passed\n");
}

// Test coordinate modulation
static void test_coordinate_modulation(void) {
    printf("Testing coordinate modulation...\n");

    FloquetConfig config = create_test_config();
    bool init_success = init_floquet_lattice(&config);
    assert(init_success);

    // Test central stabilizer coordinates at different times
    size_t central_idx = (config.width * config.height) / 2;
    double x0, y0;
    bool coord_success = get_floquet_coordinates(central_idx, 0, &x0, &y0);
    assert(coord_success);

    for (size_t t = 1; t < TEST_TIME_STEPS; t++) {
        double x, y;
        coord_success = get_floquet_coordinates(central_idx, t, &x, &y);
        assert(coord_success);

        // Verify coordinate modulation
        double time = t * config.period / config.time_steps;
        double modulation = sin(2.0 * M_PI * time / config.period);
        assert(fabs(x - (x0 + 0.1 * modulation)) < 1e-6);
        assert(fabs(y - (y0 + 0.1 * modulation)) < 1e-6);
    }

    cleanup_floquet_lattice();
    free(config.time_dependent_couplings);
    printf("Coordinate modulation test passed\n");
}

int main(void) {
    printf("Running Floquet surface code tests...\n\n");

    test_initialization();
    test_time_evolution();
    test_coupling_modulation();
    test_boundary_conditions();
    test_stabilizer_types();
    test_config_validation();
    test_coordinate_modulation();

    printf("\nAll Floquet surface code tests passed!\n");
    return 0;
}
