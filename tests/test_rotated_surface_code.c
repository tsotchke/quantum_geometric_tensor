/**
 * @file test_rotated_surface_code.c
 * @brief Test suite for rotated surface code implementation
 */

#include "quantum_geometric/physics/rotated_surface_code.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

// Test configurations
static const size_t TEST_DISTANCE = 3;
static const double TEST_ANGLE = 45.0;

// Helper function to create test configuration
static RotatedLatticeConfig create_test_config(void) {
    RotatedLatticeConfig config = {
        .distance = TEST_DISTANCE,
        .width = 0,  // Will be calculated
        .height = 0, // Will be calculated
        .angle = TEST_ANGLE,
        .use_boundary_stabilizers = true
    };
    calculate_rotated_dimensions(config.distance, &config.width, &config.height);
    return config;
}

// Test initialization
static void test_initialization(void) {
    printf("Testing rotated lattice initialization...\n");

    RotatedLatticeConfig config = create_test_config();
    bool init_success = init_rotated_lattice(&config);
    assert(init_success);

    // Verify dimensions
    assert(config.width == TEST_DISTANCE + 1);
    assert(config.height == TEST_DISTANCE + 1);

    cleanup_rotated_lattice();
    printf("Rotated lattice initialization test passed\n");
}

// Test coordinate transformation
static void test_coordinate_transformation(void) {
    printf("Testing coordinate transformation...\n");

    RotatedLatticeConfig config = create_test_config();
    bool init_success = init_rotated_lattice(&config);
    assert(init_success);

    // Test central stabilizer coordinates
    double x, y;
    size_t central_idx = (config.width * config.height) / 2;
    bool coord_success = get_rotated_coordinates(central_idx, &x, &y);
    assert(coord_success);

    // Verify rotation
    double angle_rad = TEST_ANGLE * M_PI / 180.0;
    double expected_x = cos(angle_rad) - sin(angle_rad);
    double expected_y = sin(angle_rad) + cos(angle_rad);
    
    assert(fabs(x - expected_x) < 1e-6);
    assert(fabs(y - expected_y) < 1e-6);

    cleanup_rotated_lattice();
    printf("Coordinate transformation test passed\n");
}

// Test qubit mapping
static void test_qubit_mapping(void) {
    printf("Testing qubit mapping...\n");

    RotatedLatticeConfig config = create_test_config();
    bool init_success = init_rotated_lattice(&config);
    assert(init_success);

    // Test central stabilizer qubits
    size_t qubits[4];
    size_t central_idx = (config.width * config.height) / 2;
    size_t num_qubits = get_rotated_qubits(central_idx, qubits, 4);
    
    // Verify qubit count
    assert(num_qubits > 0);
    assert(num_qubits <= 4);

    // Verify qubit indices
    for (size_t i = 0; i < num_qubits; i++) {
        assert(qubits[i] < config.width * config.height);
    }

    cleanup_rotated_lattice();
    printf("Qubit mapping test passed\n");
}

// Test boundary conditions
static void test_boundary_conditions(void) {
    printf("Testing boundary conditions...\n");

    RotatedLatticeConfig config = create_test_config();
    config.use_boundary_stabilizers = true;
    bool init_success = init_rotated_lattice(&config);
    assert(init_success);

    // Test corner stabilizer
    size_t corner_idx = 0;
    assert(is_boundary_stabilizer(corner_idx));

    // Test central stabilizer
    size_t central_idx = (config.width * config.height) / 2;
    assert(!is_boundary_stabilizer(central_idx));

    // Verify boundary qubit count
    size_t qubits[4];
    size_t num_boundary_qubits = get_rotated_qubits(corner_idx, qubits, 4);
    assert(num_boundary_qubits < 4); // Should have fewer qubits on boundary

    cleanup_rotated_lattice();
    printf("Boundary conditions test passed\n");
}

// Test stabilizer types
static void test_stabilizer_types(void) {
    printf("Testing stabilizer types...\n");

    RotatedLatticeConfig config = create_test_config();
    bool init_success = init_rotated_lattice(&config);
    assert(init_success);

    // Test alternating pattern
    for (size_t row = 0; row < config.height; row++) {
        for (size_t col = 0; col < config.width; col++) {
            size_t idx = row * config.width + col;
            if ((row + col) % 2 == 1) {  // Only check positions that should have stabilizers
                StabilizerType type = get_rotated_stabilizer_type(idx);
                assert(type == ((row + col) % 4 == 1 ? STABILIZER_X : STABILIZER_Z));
            }
        }
    }

    cleanup_rotated_lattice();
    printf("Stabilizer types test passed\n");
}

// Test neighbor relationships
static void test_neighbor_relationships(void) {
    printf("Testing neighbor relationships...\n");

    RotatedLatticeConfig config = create_test_config();
    bool init_success = init_rotated_lattice(&config);
    assert(init_success);

    // Test central stabilizer neighbors
    size_t central_idx = (config.width * config.height) / 2;
    size_t neighbors[4];
    size_t num_neighbors = get_rotated_neighbors(central_idx, neighbors, 4);

    // Verify neighbor count
    assert(num_neighbors > 0);
    assert(num_neighbors <= 4);

    // Verify neighbor indices
    for (size_t i = 0; i < num_neighbors; i++) {
        assert(neighbors[i] < config.width * config.height);
        
        // Verify reciprocal relationship
        size_t reverse_neighbors[4];
        size_t reverse_count = get_rotated_neighbors(neighbors[i], reverse_neighbors, 4);
        
        bool found = false;
        for (size_t j = 0; j < reverse_count; j++) {
            if (reverse_neighbors[j] == central_idx) {
                found = true;
                break;
            }
        }
        assert(found);
    }

    cleanup_rotated_lattice();
    printf("Neighbor relationships test passed\n");
}

// Test configuration validation
static void test_config_validation(void) {
    printf("Testing configuration validation...\n");

    // Test invalid distance
    RotatedLatticeConfig invalid_config = create_test_config();
    invalid_config.distance = 2; // Must be odd and >= 3
    bool init_result = init_rotated_lattice(&invalid_config);
    assert(!init_result);

    // Test invalid dimensions
    invalid_config = create_test_config();
    invalid_config.width = MAX_ROTATED_SIZE + 1;
    init_result = init_rotated_lattice(&invalid_config);
    assert(!init_result);

    // Test invalid angle
    invalid_config = create_test_config();
    invalid_config.angle = 400.0; // Must be < 360
    init_result = init_rotated_lattice(&invalid_config);
    assert(!init_result);

    // Test valid config
    RotatedLatticeConfig valid_config = create_test_config();
    init_result = init_rotated_lattice(&valid_config);
    assert(init_result);

    cleanup_rotated_lattice();
    printf("Configuration validation test passed\n");
}

int main(void) {
    printf("Running rotated surface code tests...\n\n");

    test_initialization();
    test_coordinate_transformation();
    test_qubit_mapping();
    test_boundary_conditions();
    test_stabilizer_types();
    test_neighbor_relationships();
    test_config_validation();

    printf("\nAll rotated surface code tests passed!\n");
    return 0;
}
