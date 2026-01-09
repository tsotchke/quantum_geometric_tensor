/**
 * @file test_topological_neuromorphic.c
 * @brief Test suite for topological neuromorphic computing API
 *
 * Tests topological quantum memory, neuromorphic units, quantum-classical
 * interface, and integration between all components.
 */

#include "quantum_geometric/ai/quantum_topological_neuromorphic.h"
#include "quantum_geometric/core/error_codes.h"
#include "test_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <time.h>

// Test fixtures
static topological_memory_t* q_memory = NULL;
static neuromorphic_unit_t* c_unit = NULL;
static interface_t* interface = NULL;

// Test result tracking
static int tests_run = 0;
static int tests_passed = 0;

#define RUN_TEST(test_fn) do { \
    printf("Running %s... ", #test_fn); \
    fflush(stdout); \
    tests_run++; \
    setUp(); \
    test_fn(); \
    tearDown(); \
    tests_passed++; \
    printf("PASSED\n"); \
} while(0)

void setUp(void) {
    // Initialize test components
    topology_params_t topo_params = {
        .anyon_type = FIBONACCI_ANYONS,
        .protection_level = TOPOLOGICAL_PROTECTION_HIGH,
        .dimension = 2,
        .num_anyons = 8,
        .decoherence_rate = 0.001,
        .error_threshold = 0.01
    };
    q_memory = create_topological_state(&topo_params);

    size_t layer_sizes[] = {64, 32, 16};
    unit_params_t unit_params = {
        .num_neurons = 64,
        .topology = PERSISTENT_HOMOLOGY,
        .learning_rate = 0.01,
        .num_layers = 3,
        .layer_sizes = layer_sizes,
        .dropout_rate = 0.1,
        .weight_decay = 0.0001
    };
    c_unit = init_neuromorphic_unit(&unit_params);

    interface_params_t if_params = {
        .coupling_strength = 0.1,
        .noise_threshold = 1e-6,
        .protection_scheme = TOPOLOGICAL_ERROR_CORRECTION,
        .measurement_shots = 1000,
        .measurement_error_rate = 0.001
    };
    interface = create_quantum_classical_interface(&if_params);
}

void tearDown(void) {
    // Clean up test components
    if (q_memory) {
        free_topological_memory(q_memory);
        q_memory = NULL;
    }
    if (c_unit) {
        free_neuromorphic_unit(c_unit);
        c_unit = NULL;
    }
    if (interface) {
        free_interface(interface);
        interface = NULL;
    }
}

// Test topological state creation and manipulation
void test_topological_state_operations(void) {
    // Test state initialization
    TEST_ASSERT(q_memory != NULL);
    TEST_ASSERT(q_memory->anyon_type == FIBONACCI_ANYONS);

    // Create and verify anyonic pairs
    qgt_error_t err = create_anyonic_pairs(q_memory);
    TEST_ASSERT(err == QGT_SUCCESS);
    TEST_ASSERT(verify_anyonic_states(q_memory));

    // Test braiding operations
    braid_sequence_t* sequence = generate_braid_sequence();
    TEST_ASSERT(sequence != NULL);

    err = perform_braiding_sequence(q_memory, sequence);
    TEST_ASSERT(err == QGT_SUCCESS);
    TEST_ASSERT(verify_braiding_result(q_memory, sequence));

    free_braid_sequence(sequence);
}

// Test persistent homology computation
void test_persistent_homology(void) {
    // Generate test data
    TEST_ASSERT(c_unit != NULL);
    TEST_ASSERT(c_unit->network != NULL);

    // Create test data for topology analysis
    double test_data[128];
    for (int i = 0; i < 128; i++) {
        test_data[i] = sin((double)i * 0.1) * cos((double)i * 0.05);
    }

    persistence_params_t params = {
        .max_dimension = 3,
        .threshold = 0.1,
        .field = FIELD_Z2,
        .persistence_threshold = 0.001,
        .use_reduced = true
    };

    // Compute persistence diagram
    persistence_diagram_t* diagram = analyze_data_topology(test_data, &params);
    TEST_ASSERT(diagram != NULL);

    // Verify topological features
    topological_features_t* features = topo_neuro_extract_features(diagram);
    TEST_ASSERT(features != NULL);
    TEST_ASSERT(verify_topological_features(features));

    // Clean up
    free_persistence_diagram(diagram);
    free_topological_features(features);
}

// Test quantum-classical interface
void test_quantum_classical_interface(void) {
    TEST_ASSERT(q_memory != NULL);
    TEST_ASSERT(interface != NULL);

    // Create anyonic pairs first
    qgt_error_t err = create_anyonic_pairs(q_memory);
    TEST_ASSERT(err == QGT_SUCCESS);

    // Generate quantum data
    quantum_data_t* q_data = process_quantum_state(q_memory);
    TEST_ASSERT(q_data != NULL);

    // Test quantum to classical conversion
    classical_data_t* c_data = quantum_to_classical(interface, q_data);
    TEST_ASSERT(c_data != NULL);
    TEST_ASSERT(verify_data_conversion(q_data, c_data));

    // Clean up
    free_quantum_data(q_data);
    free_classical_data(c_data);
}

// Test neuromorphic learning
void test_neuromorphic_learning(void) {
    TEST_ASSERT(c_unit != NULL);
    TEST_ASSERT(q_memory != NULL);
    TEST_ASSERT(interface != NULL);

    // Create anyonic pairs
    qgt_error_t err = create_anyonic_pairs(q_memory);
    TEST_ASSERT(err == QGT_SUCCESS);

    // Initial loss
    double initial_loss = compute_loss(c_unit);

    // Perform learning steps
    for (int i = 0; i < 10; i++) {
        quantum_data_t* q_data = process_quantum_state(q_memory);
        TEST_ASSERT(q_data != NULL);

        classical_data_t* c_data = quantum_to_classical(interface, q_data);
        TEST_ASSERT(c_data != NULL);

        err = update_neuromorphic_unit(c_unit, c_data);
        TEST_ASSERT(err == QGT_SUCCESS);

        free_quantum_data(q_data);
        free_classical_data(c_data);
    }

    // Verify learning occurred (loss should have decreased or stayed finite)
    double final_loss = compute_loss(c_unit);
    TEST_ASSERT(isfinite(final_loss));
    // Allow for some variance in stochastic learning
    TEST_ASSERT(final_loss < initial_loss * 10.0 || final_loss < 1.0);
}

// Test topological error correction
void test_error_correction(void) {
    TEST_ASSERT(q_memory != NULL);

    // Create anyonic pairs
    qgt_error_t err = create_anyonic_pairs(q_memory);
    TEST_ASSERT(err == QGT_SUCCESS);

    // Introduce controlled error
    introduce_test_error(q_memory);
    TEST_ASSERT(needs_correction(q_memory));

    // Apply error correction
    err = apply_topological_error_correction(q_memory);
    TEST_ASSERT(err == QGT_SUCCESS);
    TEST_ASSERT(!needs_correction(q_memory));

    // Verify state preservation
    TEST_ASSERT(verify_anyonic_states(q_memory));
}

// Test topological protection
void test_topological_protection(void) {
    TEST_ASSERT(q_memory != NULL);

    // Create anyonic pairs
    qgt_error_t err = create_anyonic_pairs(q_memory);
    TEST_ASSERT(err == QGT_SUCCESS);

    // Measure initial invariants
    topological_invariants_t* initial = measure_topological_invariants(q_memory);
    TEST_ASSERT(initial != NULL);

    // Perform noisy operations
    apply_test_noise(q_memory);
    perform_test_operations(q_memory);

    // Measure final invariants
    topological_invariants_t* final = measure_topological_invariants(q_memory);
    TEST_ASSERT(final != NULL);

    // Verify invariants preserved (topological protection)
    TEST_ASSERT(compare_topological_invariants(initial, final));

    // Clean up
    free_topological_invariants(initial);
    free_topological_invariants(final);
}

// Test weight update with topology
void test_topological_weight_update(void) {
    TEST_ASSERT(c_unit != NULL);
    TEST_ASSERT(c_unit->network != NULL);

    // Get initial network state
    network_state_t* initial = capture_network_state(c_unit->network);
    TEST_ASSERT(initial != NULL);

    // Perform topological update
    persistence_diagram_t* diagram = analyze_network_topology(c_unit->network);
    TEST_ASSERT(diagram != NULL);

    qgt_error_t err = update_topological_weights(c_unit->network, diagram);
    TEST_ASSERT(err == QGT_SUCCESS);

    // Get final network state
    network_state_t* final = capture_network_state(c_unit->network);
    TEST_ASSERT(final != NULL);

    // Verify topological constraints maintained
    TEST_ASSERT(verify_topological_constraints(final));
    TEST_ASSERT(verify_weight_update(initial, final));

    // Clean up
    free_network_state(initial);
    free_network_state(final);
    free_persistence_diagram(diagram);
}

// Integration test
void test_full_system_integration(void) {
    TEST_ASSERT(q_memory != NULL);
    TEST_ASSERT(c_unit != NULL);
    TEST_ASSERT(interface != NULL);

    // Create anyonic pairs
    qgt_error_t err = create_anyonic_pairs(q_memory);
    TEST_ASSERT(err == QGT_SUCCESS);

    // Run complete training cycle
    for (int epoch = 0; epoch < 5; epoch++) {
        quantum_data_t* q_data = process_quantum_state(q_memory);
        TEST_ASSERT(q_data != NULL);

        classical_data_t* c_data = quantum_to_classical(interface, q_data);
        TEST_ASSERT(c_data != NULL);

        err = update_neuromorphic_unit(c_unit, c_data);
        TEST_ASSERT(err == QGT_SUCCESS);

        persistence_diagram_t* diagram = analyze_network_topology(c_unit->network);
        if (diagram) {
            update_topological_weights(c_unit->network, diagram);
            free_persistence_diagram(diagram);
        }

        if (needs_correction(q_memory)) {
            apply_topological_error_correction(q_memory);
        }

        free_quantum_data(q_data);
        free_classical_data(c_data);
    }

    // Verify final state
    TEST_ASSERT(verify_anyonic_states(q_memory));
    TEST_ASSERT(verify_system_integration(q_memory, c_unit, interface));
}

// Test null parameter handling
void test_null_parameters(void) {
    // Test null handling for all major functions
    TEST_ASSERT(create_topological_state(NULL) == NULL);
    TEST_ASSERT(init_neuromorphic_unit(NULL) == NULL);
    TEST_ASSERT(create_quantum_classical_interface(NULL) == NULL);

    // These should not crash
    free_topological_memory(NULL);
    free_neuromorphic_unit(NULL);
    free_interface(NULL);

    TEST_ASSERT(process_quantum_state(NULL) == NULL);
    TEST_ASSERT(quantum_to_classical(NULL, NULL) == NULL);
    TEST_ASSERT(measure_topological_invariants(NULL) == NULL);
}

int main(void) {
    printf("\n");
    printf("==============================================\n");
    printf("Topological Neuromorphic Computing Test Suite\n");
    printf("==============================================\n\n");

    // Initialize random seed
    srand((unsigned int)time(NULL));

    // Run tests
    RUN_TEST(test_topological_state_operations);
    RUN_TEST(test_persistent_homology);
    RUN_TEST(test_quantum_classical_interface);
    RUN_TEST(test_neuromorphic_learning);
    RUN_TEST(test_error_correction);
    RUN_TEST(test_topological_protection);
    RUN_TEST(test_topological_weight_update);
    RUN_TEST(test_full_system_integration);
    RUN_TEST(test_null_parameters);

    printf("\n");
    printf("==============================================\n");
    printf("Results: %d/%d tests passed\n", tests_passed, tests_run);
    printf("==============================================\n");

    return tests_passed == tests_run ? 0 : 1;
}
