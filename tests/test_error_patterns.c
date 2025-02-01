/**
 * @file test_error_patterns.c
 * @brief Test suite for error pattern detection and analysis
 */

#include "quantum_geometric/physics/error_patterns.h"
#include "quantum_geometric/physics/error_syndrome.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

// Test configurations
static const size_t TEST_LATTICE_SIZE = 8;
static const size_t TEST_PATTERN_SIZE = 4;
static const double TEST_SIMILARITY_THRESHOLD = 0.7;
static const size_t TEST_MIN_OCCURRENCES = 2;

// Helper function to create test matching graph
static MatchingGraph* create_test_graph(void) {
    MatchingGraph* graph = init_matching_graph(TEST_PATTERN_SIZE * 2, TEST_PATTERN_SIZE * 4);
    assert(graph != NULL);

    // Add test vertices forming a line pattern
    add_syndrome_vertex(graph, 1, 1, 1, 0.8, false, 0);
    add_syndrome_vertex(graph, 2, 1, 1, 0.7, false, 0);
    add_syndrome_vertex(graph, 3, 1, 1, 0.9, false, 0);
    add_syndrome_vertex(graph, 4, 1, 1, 0.6, false, 0);

    // Add edges connecting vertices
    add_syndrome_edge(graph, &graph->vertices[0], &graph->vertices[1], 1.0, false);
    add_syndrome_edge(graph, &graph->vertices[1], &graph->vertices[2], 1.0, false);
    add_syndrome_edge(graph, &graph->vertices[2], &graph->vertices[3], 1.0, false);

    return graph;
}

// Helper function to create test correlation data
static ErrorCorrelation create_test_correlation(void) {
    ErrorCorrelation correlation = {
        .spatial_correlation = 0.8,
        .temporal_correlation = 0.6,
        .cross_correlation = 0.7,
        .correlation_length = 3,
        .correlation_time = 2
    };
    return correlation;
}

// Test pattern initialization
static void test_pattern_initialization(void) {
    printf("Testing pattern initialization...\n");

    PatternConfig config = {
        .similarity_threshold = TEST_SIMILARITY_THRESHOLD,
        .min_occurrences = TEST_MIN_OCCURRENCES,
        .max_patterns = MAX_ERROR_PATTERNS,
        .track_timing = true,
        .enable_prediction = true
    };

    bool init_success = init_error_patterns(&config);
    assert(init_success);
    assert(get_pattern_count() == 0);

    cleanup_error_patterns();
    printf("Pattern initialization test passed\n");
}

// Test pattern detection
static void test_pattern_detection(void) {
    printf("Testing pattern detection...\n");

    PatternConfig config = {
        .similarity_threshold = TEST_SIMILARITY_THRESHOLD,
        .min_occurrences = TEST_MIN_OCCURRENCES,
        .max_patterns = MAX_ERROR_PATTERNS,
        .track_timing = true,
        .enable_prediction = true
    };

    bool init_success = init_error_patterns(&config);
    assert(init_success);

    MatchingGraph* graph = create_test_graph();
    ErrorCorrelation correlation = create_test_correlation();

    // Detect patterns
    size_t patterns_found = detect_error_patterns(graph, &correlation);
    assert(patterns_found > 0);
    assert(get_pattern_count() > 0);

    // Verify pattern properties
    const ErrorPattern* pattern = get_pattern(0);
    assert(pattern != NULL);
    assert(pattern->type != PATTERN_UNKNOWN);
    assert(pattern->size > 0);
    assert(pattern->is_active);

    cleanup_matching_graph(graph);
    cleanup_error_patterns();
    printf("Pattern detection test passed\n");
}

// Test pattern classification
static void test_pattern_classification(void) {
    printf("Testing pattern classification...\n");

    // Create test vertices for different patterns
    SyndromeVertex point = {.x = 1, .y = 1, .z = 1};
    
    SyndromeVertex line[2] = {
        {.x = 1, .y = 1, .z = 1},
        {.x = 2, .y = 1, .z = 1}
    };
    
    SyndromeVertex cluster[3] = {
        {.x = 1, .y = 1, .z = 1},
        {.x = 1.5, .y = 1.5, .z = 1},
        {.x = 1.2, .y = 1.3, .z = 1}
    };

    // Test classifications
    PatternType point_type = classify_pattern_type(&point, 1);
    assert(point_type == PATTERN_POINT);

    PatternType line_type = classify_pattern_type(line, 2);
    assert(line_type == PATTERN_LINE);

    PatternType cluster_type = classify_pattern_type(cluster, 3);
    assert(cluster_type == PATTERN_CLUSTER);

    printf("Pattern classification test passed\n");
}

// Test pattern similarity calculation
static void test_pattern_similarity(void) {
    printf("Testing pattern similarity calculation...\n");

    // Create two similar patterns
    ErrorPattern pattern1 = {
        .type = PATTERN_LINE,
        .size = 2,
        .vertices = {
            {.x = 1, .y = 1, .z = 1},
            {.x = 2, .y = 1, .z = 1}
        }
    };

    ErrorPattern pattern2 = {
        .type = PATTERN_LINE,
        .size = 2,
        .vertices = {
            {.x = 1.1, .y = 1, .z = 1},
            {.x = 2.1, .y = 1, .z = 1}
        }
    };

    // Test similarity
    double similarity = calculate_pattern_similarity(&pattern1, &pattern2);
    assert(similarity > 0.5); // Should be similar

    // Test with different pattern types
    pattern2.type = PATTERN_CLUSTER;
    similarity = calculate_pattern_similarity(&pattern1, &pattern2);
    assert(similarity == 0.0); // Should be completely different

    printf("Pattern similarity test passed\n");
}

// Test pattern tracking
static void test_pattern_tracking(void) {
    printf("Testing pattern tracking...\n");

    PatternConfig config = {
        .similarity_threshold = TEST_SIMILARITY_THRESHOLD,
        .min_occurrences = TEST_MIN_OCCURRENCES,
        .max_patterns = MAX_ERROR_PATTERNS,
        .track_timing = true,
        .enable_prediction = true
    };

    bool init_success = init_error_patterns(&config);
    assert(init_success);

    MatchingGraph* graph = create_test_graph();
    ErrorCorrelation correlation = create_test_correlation();

    // Initial detection
    size_t patterns_found = detect_error_patterns(graph, &correlation);
    assert(patterns_found > 0);

    // Update database
    size_t active_patterns = update_pattern_database(graph);
    assert(active_patterns > 0);

    // Check pattern timing
    const ErrorPattern* pattern = get_pattern(0);
    assert(pattern != NULL);
    
    PatternTiming timing;
    bool got_stats = get_pattern_statistics(0, &timing);
    assert(got_stats);
    assert(timing.occurrences > 0);

    cleanup_matching_graph(graph);
    cleanup_error_patterns();
    printf("Pattern tracking test passed\n");
}

// Test pattern merging
static void test_pattern_merging(void) {
    printf("Testing pattern merging...\n");

    // Create two similar patterns
    ErrorPattern pattern1 = {
        .type = PATTERN_LINE,
        .size = 2,
        .vertices = {
            {.x = 1, .y = 1, .z = 1},
            {.x = 2, .y = 1, .z = 1}
        },
        .weight = 1.0,
        .timing = {
            .first_seen = 0,
            .last_seen = 5,
            .occurrences = 3,
            .frequency = 0.6
        }
    };

    ErrorPattern pattern2 = {
        .type = PATTERN_LINE,
        .size = 2,
        .vertices = {
            {.x = 1.1, .y = 1, .z = 1},
            {.x = 2.1, .y = 1, .z = 1}
        },
        .weight = 0.8,
        .timing = {
            .first_seen = 2,
            .last_seen = 7,
            .occurrences = 4,
            .frequency = 0.8
        }
    };

    // Test merging
    bool merged = merge_similar_patterns(&pattern1, &pattern2);
    assert(merged);
    assert(pattern1.timing.occurrences == 7); // Combined occurrences
    assert(pattern1.timing.first_seen == 0); // Earlier first seen
    assert(pattern1.timing.last_seen == 7); // Later last seen
    assert(pattern1.weight == 1.0); // Higher weight kept

    printf("Pattern merging test passed\n");
}

int main(void) {
    printf("Running error pattern tests...\n\n");

    test_pattern_initialization();
    test_pattern_detection();
    test_pattern_classification();
    test_pattern_similarity();
    test_pattern_tracking();
    test_pattern_merging();

    printf("\nAll error pattern tests passed!\n");
    return 0;
}
