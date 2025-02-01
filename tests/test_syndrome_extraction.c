/**
 * @file test_syndrome_extraction.c
 * @brief Tests for quantum error syndrome extraction
 */

#include "quantum_geometric/physics/syndrome_extraction.h"
#include "quantum_geometric/physics/z_stabilizer_operations.h"
#include "quantum_geometric/hardware/quantum_hardware_optimization.h"
#include "quantum_geometric/core/error_codes.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <time.h>

// Error handling
static qgt_error_t err;

// Test helper functions
static void test_initialization(void);
static void test_error_type_classification(void);
static void test_plaquette_vertex_operators(void);
static void test_spatial_correlations(void);
static void test_error_history_tracking(void);
static void test_edge_weights(void);
static void test_neighbor_detection(void);
static void test_boundary_matching(void);
static void test_error_prediction(void);
static void test_error_cases(void);

// Helper functions
static SyndromeConfig create_test_config(void);
static quantum_state* create_test_state(void);
static void verify_plaquette_indices(const SyndromeCache* cache, size_t width, size_t height);
static void verify_vertex_indices(const SyndromeCache* cache, size_t width, size_t height);
static void verify_spatial_correlations(const SyndromeState* state);
static void verify_error_history(const SyndromeState* state);
static void verify_edge_weights(const SyndromeState* state);
static void verify_neighbor_patterns(const SyndromeState* state);

int main(void) {
    printf("Running syndrome extraction tests...\n\n");

    test_initialization();
    test_error_type_classification();
    test_plaquette_vertex_operators();
    test_spatial_correlations();
    test_error_history_tracking();
    test_edge_weights();
    test_neighbor_detection();
    test_boundary_matching();
    test_error_prediction();
    test_error_cases();

    printf("\nAll syndrome extraction tests passed!\n");
    return 0;
}

static void test_initialization(void) {
    printf("Testing initialization...\n");

    SyndromeConfig config = create_test_config();
    SyndromeState state;
    err = init_syndrome_extraction(&state, &config);
    assert(err == QGT_SUCCESS && "Failed to initialize syndrome state");

    // Verify basic state initialization
    assert(state.graph != NULL && "Graph not allocated");
    assert(state.graph->vertices != NULL && "Vertices not allocated");
    assert(state.graph->edges != NULL && "Edges not allocated");
    assert(state.graph->correlation_matrix != NULL && "Correlation matrix not allocated");
    assert(state.graph->parallel_groups != NULL && "Parallel groups not allocated");
    assert(state.graph->pattern_weights != NULL && "Pattern weights not allocated");
    assert(state.graph->num_vertices == 0 && "Initial vertices not zero");
    assert(state.graph->num_edges == 0 && "Initial edges not zero");
    assert(state.graph->num_parallel_groups == 0 && "Initial parallel groups not zero");

    // Verify vertex initialization
    for (size_t i = 0; i < state.graph->max_vertices; i++) {
        SyndromeVertex* vertex = &state.graph->vertices[i];
        assert(vertex->error_history != NULL && "Error history not allocated");
        assert(vertex->history_size == 0 && "Initial history size not zero");
        assert(fabs(vertex->confidence - 1.0) < 1e-6 && "Initial confidence not 1.0");
        assert(fabs(vertex->correlation_weight) < 1e-6 && "Initial correlation weight not zero");
        assert(!vertex->part_of_chain && "Initial chain participation not false");
    }

    // Verify edge initialization
    for (size_t i = 0; i < state.graph->max_edges; i++) {
        SyndromeEdge* edge = &state.graph->edges[i];
        assert(fabs(edge->chain_probability) < 1e-6 && "Initial chain probability not zero");
        assert(edge->chain_length == 0 && "Initial chain length not zero");
        assert(!edge->is_matched && "Initial matched state not false");
    }

    cleanup_syndrome_extraction(&state);
    printf("Initialization test passed\n");
}

static void test_error_type_classification(void) {
    printf("Testing error type classification...\n");

    SyndromeConfig config = create_test_config();
    SyndromeState state;
    err = init_syndrome_extraction(&state, &config);
    assert(err == QGT_SUCCESS && "Failed to initialize syndrome state");

    quantum_state* qstate = create_test_state();

    // Inject X errors
    size_t x_errors[] = {0, 1};
    for (size_t i = 0; i < 2; i++) {
        qstate->amplitudes[x_errors[i] * 2] = 0.0;
        qstate->amplitudes[x_errors[i] * 2 + 1] = 1.0;
    }

    // Inject Z errors
    size_t z_errors[] = {4, 5};
    for (size_t i = 0; i < 2; i++) {
        qstate->amplitudes[z_errors[i] * 2] *= -1;
    }

    // Inject Y errors (combined X and Z)
    size_t y_errors[] = {8, 9};
    for (size_t i = 0; i < 2; i++) {
        qstate->amplitudes[y_errors[i] * 2] = 0.0;
        qstate->amplitudes[y_errors[i] * 2 + 1] = -1.0;
    }

    ErrorSyndrome syndrome;
    err = extract_error_syndrome(&state, qstate, &syndrome);
    assert(err == QGT_SUCCESS && "Failed to extract error syndrome");

    // Verify error types
    for (size_t i = 0; i < syndrome.num_errors; i++) {
        size_t loc = syndrome.error_locations[i];
        error_type_t type = syndrome.error_types[i];

        // Check X errors
        for (size_t j = 0; j < 2; j++) {
            if (loc == x_errors[j]) {
                assert(type == ERROR_X);
            }
        }

        // Check Z errors
        for (size_t j = 0; j < 2; j++) {
            if (loc == z_errors[j]) {
                assert(type == ERROR_Z);
            }
        }

        // Check Y errors
        for (size_t j = 0; j < 2; j++) {
            if (loc == y_errors[j]) {
                assert(type == ERROR_Y);
            }
        }
    }

    cleanup_syndrome_extraction(&state);
    free(qstate);
    printf("Error type classification test passed\n");
}

static void test_plaquette_vertex_operators(void) {
    printf("Testing plaquette and vertex operators...\n");

    SyndromeConfig config = create_test_config();
    SyndromeState state;
    err = init_syndrome_extraction(&state, &config);
    assert(err == QGT_SUCCESS && "Failed to initialize syndrome state");

    // Verify plaquette indices
    verify_plaquette_indices(state.cache, config.lattice_width, config.lattice_height);

    // Verify vertex indices
    verify_vertex_indices(state.cache, config.lattice_width, config.lattice_height);

    cleanup_syndrome_extraction(&state);
    printf("Plaquette and vertex operator test passed\n");
}

static void test_spatial_correlations(void) {
    printf("Testing spatial correlations...\n");

    SyndromeConfig config = create_test_config();
    SyndromeState state;
    err = init_syndrome_extraction(&state, &config);
    assert(err == QGT_SUCCESS && "Failed to initialize syndrome state");

    quantum_state* qstate = create_test_state();

    // Inject correlated errors in a pattern
    size_t error_pattern[] = {0, 1, 4, 5};  // 2x2 block
    for (size_t i = 0; i < 4; i++) {
        qstate->amplitudes[error_pattern[i] * 2] = 0.0;
        qstate->amplitudes[error_pattern[i] * 2 + 1] = 1.0;
    }

    // Extract multiple syndromes to build correlation data
    ErrorSyndrome syndrome;
    for (size_t i = 0; i < 10; i++) {
        err = extract_error_syndrome(&state, qstate, &syndrome);
        assert(err == QGT_SUCCESS && "Failed to extract error syndrome");
    }

    // Verify spatial correlations
    verify_spatial_correlations(&state);

    cleanup_syndrome_extraction(&state);
    free(qstate);
    printf("Spatial correlations test passed\n");
}

static void test_error_history_tracking(void) {
    printf("Testing error history tracking...\n");

    SyndromeConfig config = create_test_config();
    SyndromeState state;
    err = init_syndrome_extraction(&state, &config);
    assert(err == QGT_SUCCESS && "Failed to initialize syndrome state");

    quantum_state* qstate = create_test_state();

    // Inject repeating error pattern
    size_t error_pattern[] = {0, 1, 4, 5};
    for (size_t i = 0; i < 10; i++) {
        // Clear previous errors
        for (size_t j = 0; j < qstate->num_qubits; j++) {
            qstate->amplitudes[j * 2] = 1.0;
            qstate->amplitudes[j * 2 + 1] = 0.0;
        }

        // Inject new errors
        for (size_t j = 0; j < 4; j++) {
            qstate->amplitudes[error_pattern[j] * 2] = 0.0;
            qstate->amplitudes[error_pattern[j] * 2 + 1] = 1.0;
        }

        ErrorSyndrome syndrome;
        err = extract_error_syndrome(&state, qstate, &syndrome);
        assert(err == QGT_SUCCESS && "Failed to extract error syndrome");
    }

    // Verify error history
    verify_error_history(&state);

    cleanup_syndrome_extraction(&state);
    free(qstate);
    printf("Error history tracking test passed\n");
}

static void test_edge_weights(void) {
    printf("Testing edge weights...\n");

    SyndromeConfig config = create_test_config();
    SyndromeState state;
    err = init_syndrome_extraction(&state, &config);
    assert(err == QGT_SUCCESS && "Failed to initialize syndrome state");

    quantum_state* qstate = create_test_state();

    // Inject chain of errors
    size_t chain[] = {0, 1, 2, 3};  // Linear chain
    for (size_t i = 0; i < 4; i++) {
        qstate->amplitudes[chain[i] * 2] = 0.0;
        qstate->amplitudes[chain[i] * 2 + 1] = 1.0;
    }

    // Extract multiple syndromes
    ErrorSyndrome syndrome;
    for (size_t i = 0; i < 10; i++) {
        err = extract_error_syndrome(&state, qstate, &syndrome);
        assert(err == QGT_SUCCESS && "Failed to extract error syndrome");
    }

    // Verify edge weights
    verify_edge_weights(&state);

    cleanup_syndrome_extraction(&state);
    free(qstate);
    printf("Edge weights test passed\n");
}

static void test_neighbor_detection(void) {
    printf("Testing neighbor detection...\n");

    SyndromeConfig config = create_test_config();
    SyndromeState state;
    err = init_syndrome_extraction(&state, &config);
    assert(err == QGT_SUCCESS && "Failed to initialize syndrome state");

    quantum_state* qstate = create_test_state();

    // Inject errors in neighboring sites
    size_t neighbors[] = {
        0,                          // Center
        1,                          // East
        config.lattice_width,       // South
        config.lattice_width + 1    // Southeast
    };

    for (size_t i = 0; i < 4; i++) {
        qstate->amplitudes[neighbors[i] * 2] = 0.0;
        qstate->amplitudes[neighbors[i] * 2 + 1] = 1.0;
    }

    // Extract syndrome
    ErrorSyndrome syndrome;
    err = extract_error_syndrome(&state, qstate, &syndrome);
    assert(err == QGT_SUCCESS && "Failed to extract error syndrome");

    // Verify neighbor patterns
    verify_neighbor_patterns(&state);

    cleanup_syndrome_extraction(&state);
    free(qstate);
    printf("Neighbor detection test passed\n");
}

static void test_boundary_matching(void) {
    printf("Testing boundary matching...\n");

    SyndromeConfig config = create_test_config();
    config.use_boundary_matching = true;
    SyndromeState state;
    err = init_syndrome_extraction(&state, &config);
    assert(err == QGT_SUCCESS && "Failed to initialize syndrome state");

    quantum_state* qstate = create_test_state();

    // Inject errors at boundaries
    size_t boundary_errors[] = {
        0,                                          // Top left
        config.lattice_width - 1,                   // Top right
        config.lattice_width * (config.lattice_height - 1), // Bottom left
        config.lattice_width * config.lattice_height - 1    // Bottom right
    };

    for (size_t i = 0; i < 4; i++) {
        qstate->amplitudes[boundary_errors[i] * 2] = 0.0;
        qstate->amplitudes[boundary_errors[i] * 2 + 1] = 1.0;
    }

    // Extract syndrome
    ErrorSyndrome syndrome;
    err = extract_error_syndrome(&state, qstate, &syndrome);
    assert(err == QGT_SUCCESS && "Failed to extract error syndrome");

    // Verify boundary vertices and edges
    for (size_t i = 0; i < state.graph->num_vertices; i++) {
        SyndromeVertex* vertex = &state.graph->vertices[i];
        if (vertex->x == 0 || vertex->x == config.lattice_width - 1 ||
            vertex->y == 0 || vertex->y == config.lattice_height - 1) {
            assert(vertex->is_boundary);
        }
    }

    for (size_t i = 0; i < state.graph->num_edges; i++) {
        SyndromeEdge* edge = &state.graph->edges[i];
        if (edge->vertex1->is_boundary || edge->vertex2->is_boundary) {
            assert(edge->is_boundary_connection);
        }
    }

    cleanup_syndrome_extraction(&state);
    free(qstate);
    printf("Boundary matching test passed\n");
}

static void test_error_prediction(void) {
    printf("Testing error prediction...\n");

    SyndromeConfig config = create_test_config();
    SyndromeState state;
    err = init_syndrome_extraction(&state, &config);
    assert(err == QGT_SUCCESS && "Failed to initialize syndrome state");

    quantum_state* qstate = create_test_state();

    // Train on repeated error pattern
    size_t error_pattern[] = {0, 1, 4, 5};  // 2x2 block
    for (size_t i = 0; i < 20; i++) {
        // Clear previous errors
        for (size_t j = 0; j < qstate->num_qubits; j++) {
            qstate->amplitudes[j * 2] = 1.0;
            qstate->amplitudes[j * 2 + 1] = 0.0;
        }

        // Inject pattern
        for (size_t j = 0; j < 4; j++) {
            qstate->amplitudes[error_pattern[j] * 2] = 0.0;
            qstate->amplitudes[error_pattern[j] * 2 + 1] = 1.0;
        }

        ErrorSyndrome syndrome;
        err = extract_error_syndrome(&state, qstate, &syndrome);
        assert(err == QGT_SUCCESS && "Failed to extract error syndrome");
    }

    // Test prediction
    size_t predicted[4];
    size_t num_predicted;
    err = predict_next_errors(&state, predicted, 4, &num_predicted);
    assert(err == QGT_SUCCESS || err == QGT_ERROR_INSUFFICIENT_DATA && "Failed to predict errors");
    if (err == QGT_SUCCESS) {
        assert(num_predicted > 0 && "No errors predicted");
    }

    // Verify predictions match pattern
    bool found_pattern = false;
    for (size_t i = 0; i < num_predicted; i++) {
        for (size_t j = 0; j < 4; j++) {
            if (predicted[i] == error_pattern[j]) {
                found_pattern = true;
                break;
            }
        }
    }
    assert(found_pattern && "Failed to predict correct error pattern");

    cleanup_syndrome_extraction(&state);
    free(qstate);
    printf("Error prediction test passed\n");
}

static void test_error_cases(void) {
    printf("Testing error cases...\n");

    SyndromeConfig config = create_test_config();
    SyndromeState state;

    // Test null pointers
    err = init_syndrome_extraction(NULL, &config);
    assert(err == QGT_ERROR_INVALID_ARGUMENT && "Failed to catch null state");
    err = init_syndrome_extraction(&state, NULL);
    assert(err == QGT_ERROR_INVALID_ARGUMENT && "Failed to catch null config");

    // Test invalid config parameters
    SyndromeConfig invalid_config = config;
    invalid_config.lattice_width = 0;
    err = init_syndrome_extraction(&state, &invalid_config);
    assert(err == QGT_ERROR_INVALID_PARAMETER && "Failed to catch invalid width");

    invalid_config = config;
    invalid_config.confidence_threshold = 2.0;
    err = init_syndrome_extraction(&state, &invalid_config);
    assert(err == QGT_ERROR_INVALID_PARAMETER && "Failed to catch invalid confidence");

    invalid_config = config;
    invalid_config.pattern_threshold = -0.1;
    err = init_syndrome_extraction(&state, &invalid_config);
    assert(err == QGT_ERROR_INVALID_PARAMETER && "Failed to catch invalid pattern threshold");

    // Test valid initialization
    err = init_syndrome_extraction(&state, &config);
    assert(err == QGT_SUCCESS && "Failed valid initialization");

    // Test invalid syndrome extraction
    quantum_state* qstate = create_test_state();
    ErrorSyndrome syndrome;

    err = extract_error_syndrome(NULL, qstate, &syndrome);
    assert(err == QGT_ERROR_INVALID_ARGUMENT && "Failed to catch null state");
    err = extract_error_syndrome(&state, NULL, &syndrome);
    assert(err == QGT_ERROR_INVALID_ARGUMENT && "Failed to catch null quantum state");
    err = extract_error_syndrome(&state, qstate, NULL);
    assert(err == QGT_ERROR_INVALID_ARGUMENT && "Failed to catch null syndrome");

    // Test invalid error prediction
    size_t predicted[4];
    size_t num_predicted;
    err = predict_next_errors(NULL, predicted, 4, &num_predicted);
    assert(err == QGT_ERROR_INVALID_ARGUMENT && "Failed to catch null state");
    err = predict_next_errors(&state, NULL, 4, &num_predicted);
    assert(err == QGT_ERROR_INVALID_ARGUMENT && "Failed to catch null predictions array");
    err = predict_next_errors(&state, predicted, 0, &num_predicted);
    assert(err == QGT_ERROR_INVALID_PARAMETER && "Failed to catch zero max predictions");
    err = predict_next_errors(&state, predicted, 4, NULL);
    assert(err == QGT_ERROR_INVALID_ARGUMENT && "Failed to catch null num_predicted");

    cleanup_syndrome_extraction(&state);
    free(qstate);
    printf("Error cases test passed\n");
}

static SyndromeConfig create_test_config(void) {
    SyndromeConfig config = {
        .lattice_width = 4,
        .lattice_height = 4,
        .detection_threshold = 0.01,
        .weight_scale_factor = 1.0,
        .max_matching_iterations = 100,
        .use_boundary_matching = true,
        .confidence_threshold = 0.8,
        .min_measurements = 5,
        .error_rate_threshold = 0.1,
        .enable_parallel = true,
        .max_parallel_ops = 4,
        .parallel_group_size = 2,
        .history_window = HISTORY_SIZE,
        .pattern_threshold = 0.7,
        .min_pattern_occurrences = 3
    };
    return config;
}

static quantum_state* create_test_state(void) {
    quantum_state* state = malloc(sizeof(quantum_state));
    state->num_qubits = 32;  // 4x4 lattice with 2 qubits per site
    state->amplitudes = calloc(state->num_qubits * 2, sizeof(double));
    
    // Initialize to |0⟩ state
    for (size_t i = 0; i < state->num_qubits; i++) {
        state->amplitudes[i * 2] = 1.0;
    }
    
    return state;
}

static void verify_plaquette_indices(const SyndromeCache* cache,
                                   size_t width,
                                   size_t height) {
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            size_t idx = y * width + x;
            
            // Check plaquette operator indices
            assert(cache->plaquette_indices[idx * 4] == (y * width + x) * 2);
            assert(cache->plaquette_indices[idx * 4 + 1] == (y * width + x + 1) * 2);
            assert(cache->plaquette_indices[idx * 4 + 2] == ((y + 1) * width + x) * 2);
            assert(cache->plaquette_indices[idx * 4 + 3] == ((y + 1) * width + x + 1) * 2);
        }
    }
}

static void verify_vertex_indices(const SyndromeCache* cache,
                                size_t width,
                                size_t height) {
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            size_t idx = y * width + x;
            
            // Check vertex operator indices
            assert(cache->vertex_indices[idx * 4] == (y * width + x) * 2 + 1);
            assert(cache->vertex_indices[idx * 4 + 1] == (y * width + x + 1) * 2 + 1);
            assert(cache->vertex_indices[idx * 4 + 2] == ((y + 1) * width + x) * 2 + 1);
            assert(cache->vertex_indices[idx * 4 + 3] == ((y + 1) * width + x + 1) * 2 + 1);
        }
    }
}

static void verify_spatial_correlations(const SyndromeState* state) {
    size_t total_stabilizers = state->config.lattice_width *
                             state->config.lattice_height * 2;

    for (size_t i = 0; i < total_stabilizers; i++) {
        for (size_t j = i + 1; j < total_stabilizers; j++) {
            // Calculate spatial distance
            size_t x1 = i % state->config.lattice_width;
            size_t y1 = i / state->config.lattice_width;
            size_t x2 = j % state->config.lattice_width;
            size_t y2 = j / state->config.lattice_width;
            double distance = sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));

            // Check correlation decay with distance
            size_t idx = i * total_stabilizers + j;
            assert(state->cache->correlations[idx] <= exp(-distance/2.0));
        }
    }
}

static void verify_error_history(const SyndromeState* state) {
    size_t total_stabilizers = state->config.lattice_width *
                             state->config.lattice_height * 2;

    // Check error history tracking
    for (size_t i = 0; i < total_stabilizers; i++) {
        assert(state->graph->vertices[i].history_size <= HISTORY_SIZE);
        
        // Verify history values are valid
        for (size_t j = 0; j < state->graph->vertices[i].history_size; j++) {
            assert(state->graph->vertices[i].error_history[j] >= 0.0);
            assert(state->graph->vertices[i].error_history[j] <= 1.0);
        }
    }
}

static void verify_edge_weights(const SyndromeState* state) {
    // Check edge weights between adjacent vertices
    for (size_t i = 0; i < state->graph->num_edges; i++) {
        SyndromeEdge* edge = &state->graph->edges[i];
        
        // Verify weight calculation
        assert(edge->weight >= 0.0);
        assert(edge->weight <= 1.0);
        
        // Verify chain properties
        if (edge->chain_probability > 0.8) {
            assert(edge->chain_length > 0);
            assert(edge->vertex1->part_of_chain);
            assert(edge->vertex2->part_of_chain);
        }
    }
}

static void verify_neighbor_patterns(const SyndromeState* state) {
    size_t total_stabilizers = state->config.lattice_width *
                             state->config.lattice_height * 2;

    for (size_t i = 0; i < total_stabilizers; i++) {
        if (!state->cache->error_history[i]) {
            continue;
        }

        // Check neighbors
        size_t x = i % state->config.lattice_width;
        size_t y = i / state->config.lattice_width;

        size_t neighbors[] = {
            y > 0 ? i - state->config.lattice_width : (size_t)-1,
            x < state->config.lattice_width - 1 ? i + 1 : (size_t)-1,
            y < state->config.lattice_height - 1 ? i + state->config.lattice_width : (size_t)-1,
            x > 0 ? i - 1 : (size_t)-1
        };

        // Verify neighbor correlations
        for (size_t j = 0; j < 4; j++) {
            if (neighbors[j] != (size_t)-1 &&
                state->cache->error_history[neighbors[j]]) {
                size_t idx = i * total_stabilizers + neighbors[j];
                assert(state->cache->correlations[idx] > 0.0);
            }
        }
    }
}
