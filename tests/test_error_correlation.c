/**
 * @file test_error_correlation.c
 * @brief Test suite for error correlation analysis
 */

#include "quantum_geometric/physics/error_correlation.h"
#include "quantum_geometric/physics/error_syndrome.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

// Test configurations
static const size_t TEST_LATTICE_SIZE = 8;
static const size_t MAX_TEST_SYNDROMES = 16;
static const size_t TEST_HISTORY_LENGTH = 10;
static const double TEST_SPATIAL_THRESHOLD = 0.3;
static const double TEST_TEMPORAL_THRESHOLD = 0.2;

// Helper function to create test quantum state
static quantum_state* create_test_state(void) {
    quantum_state* state = create_quantum_state(TEST_LATTICE_SIZE);
    if (!state) {
        return NULL;
    }

    // Initialize with a simple test state
    initialize_test_state(state);
    return state;
}

// Helper function to create test syndrome graph
static MatchingGraph* create_test_graph(void) {
    MatchingGraph* graph = init_matching_graph(MAX_TEST_SYNDROMES, MAX_TEST_SYNDROMES * 2);
    if (!graph) {
        return NULL;
    }

    // Add some test vertices
    add_syndrome_vertex(graph, 1, 1, 1, 0.5, false, 0);
    add_syndrome_vertex(graph, 2, 2, 1, 0.6, false, 1);
    add_syndrome_vertex(graph, 3, 3, 1, 0.4, false, 2);
    add_syndrome_vertex(graph, 4, 4, 1, 0.7, false, 3);

    // Add some test edges
    add_syndrome_edge(graph, &graph->vertices[0], &graph->vertices[1], 1.0, false);
    add_syndrome_edge(graph, &graph->vertices[1], &graph->vertices[2], 1.2, false);
    add_syndrome_edge(graph, &graph->vertices[2], &graph->vertices[3], 0.8, false);

    return graph;
}

// Test correlation initialization
static void test_correlation_initialization(void) {
    printf("Testing correlation initialization...\n");

    CorrelationConfig config = {
        .spatial_threshold = TEST_SPATIAL_THRESHOLD,
        .temporal_threshold = TEST_TEMPORAL_THRESHOLD,
        .max_correlation_dist = TEST_LATTICE_SIZE / 2,
        .history_length = TEST_HISTORY_LENGTH,
        .enable_cross_correlation = true
    };

    bool init_success = init_error_correlation(&config);
    assert(init_success);

    cleanup_error_correlation();
    printf("Correlation initialization test passed\n");
}

// Test correlation analysis
static void test_correlation_analysis(void) {
    printf("Testing correlation analysis...\n");

    quantum_state* state = create_test_state();
    assert(state != NULL);

    MatchingGraph* graph = create_test_graph();
    assert(graph != NULL);

    CorrelationConfig config = {
        .spatial_threshold = TEST_SPATIAL_THRESHOLD,
        .temporal_threshold = TEST_TEMPORAL_THRESHOLD,
        .max_correlation_dist = TEST_LATTICE_SIZE / 2,
        .history_length = TEST_HISTORY_LENGTH,
        .enable_cross_correlation = true
    };

    bool init_success = init_error_correlation(&config);
    assert(init_success);

    // Test initial correlation analysis
    ErrorCorrelation correlation = analyze_error_correlations(graph, state);
    assert(correlation.spatial_correlation >= 0.0);
    assert(correlation.spatial_correlation <= 1.0);
    assert(correlation.temporal_correlation >= 0.0);
    assert(correlation.temporal_correlation <= 1.0);
    assert(correlation.correlation_length <= TEST_LATTICE_SIZE);
    assert(correlation.correlation_time <= TEST_HISTORY_LENGTH);

    cleanup_matching_graph(graph);
    destroy_quantum_state(state);
    cleanup_error_correlation();
    printf("Correlation analysis test passed\n");
}

// Test correlation model updates
static void test_correlation_model_updates(void) {
    printf("Testing correlation model updates...\n");

    quantum_state* state = create_test_state();
    assert(state != NULL);

    MatchingGraph* graph = create_test_graph();
    assert(graph != NULL);

    CorrelationConfig config = {
        .spatial_threshold = TEST_SPATIAL_THRESHOLD,
        .temporal_threshold = TEST_TEMPORAL_THRESHOLD,
        .max_correlation_dist = TEST_LATTICE_SIZE / 2,
        .history_length = TEST_HISTORY_LENGTH,
        .enable_cross_correlation = true
    };

    bool init_success = init_error_correlation(&config);
    assert(init_success);

    // Initial correlation
    ErrorCorrelation correlation = analyze_error_correlations(graph, state);

    // Update model multiple times
    for (size_t i = 0; i < 5; i++) {
        // Modify graph to simulate error evolution
        add_syndrome_vertex(graph, i+1, i+1, 1, 0.5 + i*0.1, false, i+4);
        
        // Update correlation model
        correlation = update_correlation_model(graph, &correlation);
        
        // Verify updated correlations
        assert(correlation.spatial_correlation >= 0.0);
        assert(correlation.spatial_correlation <= 1.0);
        assert(correlation.temporal_correlation >= 0.0);
        assert(correlation.temporal_correlation <= 1.0);
    }

    cleanup_matching_graph(graph);
    destroy_quantum_state(state);
    cleanup_error_correlation();
    printf("Correlation model updates test passed\n");
}

// Test correlation type detection
static void test_correlation_type_detection(void) {
    printf("Testing correlation type detection...\n");

    CorrelationConfig config = {
        .spatial_threshold = TEST_SPATIAL_THRESHOLD,
        .temporal_threshold = TEST_TEMPORAL_THRESHOLD,
        .max_correlation_dist = TEST_LATTICE_SIZE / 2,
        .history_length = TEST_HISTORY_LENGTH,
        .enable_cross_correlation = true
    };

    bool init_success = init_error_correlation(&config);
    assert(init_success);

    // Create test vertices
    SyndromeVertex v1 = {
        .x = 1, .y = 1, .z = 1,
        .weight = 0.5,
        .is_boundary = false,
        .timestamp = 0
    };

    SyndromeVertex v2 = {
        .x = 2, .y = 2, .z = 1,
        .weight = 0.6,
        .is_boundary = false,
        .timestamp = 1
    };

    // Test different correlation types
    CorrelationType type = detect_correlation_type(&v1, &v2);
    assert(type >= CORRELATION_NONE && type <= CORRELATION_SPATIOTEMPORAL);

    cleanup_error_correlation();
    printf("Correlation type detection test passed\n");
}

// Test spatial correlation calculations
static void test_spatial_correlation(void) {
    printf("Testing spatial correlation calculations...\n");

    CorrelationConfig config = {
        .spatial_threshold = TEST_SPATIAL_THRESHOLD,
        .temporal_threshold = TEST_TEMPORAL_THRESHOLD,
        .max_correlation_dist = TEST_LATTICE_SIZE / 2,
        .history_length = TEST_HISTORY_LENGTH,
        .enable_cross_correlation = true
    };

    bool init_success = init_error_correlation(&config);
    assert(init_success);

    // Test vertices at different distances
    SyndromeVertex v1 = {.x = 1, .y = 1, .z = 1, .weight = 0.5};
    SyndromeVertex v2 = {.x = 2, .y = 2, .z = 1, .weight = 0.6};
    SyndromeVertex v3 = {.x = 5, .y = 5, .z = 1, .weight = 0.4};

    // Close vertices should have higher correlation
    double corr_close = calculate_spatial_correlation(&v1, &v2);
    double corr_far = calculate_spatial_correlation(&v1, &v3);
    assert(corr_close > corr_far);

    cleanup_error_correlation();
    printf("Spatial correlation test passed\n");
}

// Test temporal correlation calculations
static void test_temporal_correlation(void) {
    printf("Testing temporal correlation calculations...\n");

    CorrelationConfig config = {
        .spatial_threshold = TEST_SPATIAL_THRESHOLD,
        .temporal_threshold = TEST_TEMPORAL_THRESHOLD,
        .max_correlation_dist = TEST_LATTICE_SIZE / 2,
        .history_length = TEST_HISTORY_LENGTH,
        .enable_cross_correlation = true
    };

    bool init_success = init_error_correlation(&config);
    assert(init_success);

    // Test vertices at different times
    SyndromeVertex v1 = {.x = 1, .y = 1, .z = 1, .timestamp = 0};
    SyndromeVertex v2 = {.x = 1, .y = 1, .z = 1, .timestamp = 1};
    SyndromeVertex v3 = {.x = 1, .y = 1, .z = 1, .timestamp = 5};

    // Close timestamps should have higher correlation
    double corr_close = calculate_temporal_correlation(&v1, &v2);
    double corr_far = calculate_temporal_correlation(&v1, &v3);
    assert(corr_close > corr_far);

    cleanup_error_correlation();
    printf("Temporal correlation test passed\n");
}

int main(void) {
    printf("Running error correlation tests...\n\n");

    test_correlation_initialization();
    test_correlation_analysis();
    test_correlation_model_updates();
    test_correlation_type_detection();
    test_spatial_correlation();
    test_temporal_correlation();

    printf("\nAll error correlation tests passed!\n");
    return 0;
}
