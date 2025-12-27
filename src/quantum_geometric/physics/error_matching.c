/**
 * @file error_matching.c
 * @brief Implementation of minimum weight perfect matching for error correction
 */

#include "quantum_geometric/physics/error_syndrome.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/core/quantum_types.h"
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>

// ============================================================================
// Helper Functions for Matching
// ============================================================================

/**
 * Check if a vertex is matched in the current matching
 */
static bool is_vertex_matched(const MatchingGraph* graph, const SyndromeVertex* vertex) {
    if (!graph || !vertex) return false;

    // Check if any matched edge contains this vertex
    for (size_t i = 0; i < graph->num_edges; i++) {
        const SyndromeEdge* edge = &graph->edges[i];
        if (edge->is_matched) {
            if (edge->vertex1 == vertex || edge->vertex2 == vertex) {
                return true;
            }
        }
    }

    return false;
}

/**
 * Calculate total weight of matched edges
 */
static double calculate_matching_weight(const MatchingGraph* graph) {
    if (!graph) return -1.0;

    double total_weight = 0.0;

    for (size_t i = 0; i < graph->num_edges; i++) {
        const SyndromeEdge* edge = &graph->edges[i];
        if (edge->is_matched) {
            total_weight += edge->weight;
        }
    }

    return total_weight;
}

/**
 * Check if an edge is part of the matching
 */
static bool is_edge_in_matching(const MatchingGraph* graph, const SyndromeEdge* edge) {
    (void)graph;  // Unused but kept for API consistency
    if (!edge) return false;
    return edge->is_matched;
}

/**
 * Apply correction chain between two vertices
 * This applies Pauli corrections along the path connecting the vertices
 */
static bool apply_correction_chain(quantum_state_t* state,
                                   const SyndromeVertex* v1,
                                   const SyndromeVertex* v2) {
    if (!state || !v1 || !v2) return false;

    // Calculate Manhattan distance path
    size_t dx = (v1->x > v2->x) ? (v1->x - v2->x) : (v2->x - v1->x);
    size_t dy = (v1->y > v2->y) ? (v1->y - v2->y) : (v2->y - v1->y);
    size_t dz = (v1->z > v2->z) ? (v1->z - v2->z) : (v2->z - v1->z);

    size_t path_length = dx + dy + dz;
    if (path_length == 0) return true;  // Same vertex, no correction needed

    // Allocate path arrays
    size_t* path_x = malloc(path_length * sizeof(size_t));
    size_t* path_y = malloc(path_length * sizeof(size_t));
    size_t* path_z = malloc(path_length * sizeof(size_t));

    if (!path_x || !path_y || !path_z) {
        free(path_x);
        free(path_y);
        free(path_z);
        return false;
    }

    // Generate the correction path
    generate_correction_path(v1, v2, path_x, path_y, path_z, path_length);

    // Apply corrections along the path
    bool success = true;
    for (size_t i = 0; i < path_length && success; i++) {
        success = apply_correction_operator(state, path_x[i], path_y[i], path_z[i]);
    }

    free(path_x);
    free(path_y);
    free(path_z);

    return success;
}

// Internal matching state
typedef struct {
    bool* matched;              // Vertex matching status
    double* dual_vars;          // Dual variables for optimization
    double* slack;              // Edge slack variables
    size_t* previous;           // Previous vertex in augmenting path
    bool* in_queue;            // Queue membership flags
    size_t* queue;             // BFS queue
    size_t queue_start;        // Queue start index
    size_t queue_end;          // Queue end index
} MatchingState;

// Initialize matching state
static MatchingState* init_matching_state(const MatchingGraph* graph) {
    MatchingState* state = (MatchingState*)malloc(sizeof(MatchingState));
    if (!state) {
        return NULL;
    }

    size_t num_vertices = graph->num_vertices;
    state->matched = (bool*)calloc(num_vertices, sizeof(bool));
    state->dual_vars = (double*)calloc(num_vertices, sizeof(double));
    state->slack = (double*)calloc(graph->num_edges, sizeof(double));
    state->previous = (size_t*)calloc(num_vertices, sizeof(size_t));
    state->in_queue = (bool*)calloc(num_vertices, sizeof(bool));
    state->queue = (size_t*)calloc(num_vertices, sizeof(size_t));

    if (!state->matched || !state->dual_vars || !state->slack ||
        !state->previous || !state->in_queue || !state->queue) {
        free(state->matched);
        free(state->dual_vars);
        free(state->slack);
        free(state->previous);
        free(state->in_queue);
        free(state->queue);
        free(state);
        return NULL;
    }

    // Initialize dual variables
    for (size_t i = 0; i < num_vertices; i++) {
        state->dual_vars[i] = 0.0;
    }

    // Initialize slack variables
    for (size_t i = 0; i < graph->num_edges; i++) {
        const SyndromeEdge* edge = &graph->edges[i];
        size_t v1_idx = edge->vertex1 - graph->vertices;
        size_t v2_idx = edge->vertex2 - graph->vertices;
        state->slack[i] = edge->weight - state->dual_vars[v1_idx] - state->dual_vars[v2_idx];
    }

    state->queue_start = 0;
    state->queue_end = 0;

    return state;
}

// Clean up matching state
static void cleanup_matching_state(MatchingState* state) {
    if (state) {
        free(state->matched);
        free(state->dual_vars);
        free(state->slack);
        free(state->previous);
        free(state->in_queue);
        free(state->queue);
        free(state);
    }
}

// Find augmenting path using BFS
static bool find_augmenting_path(const MatchingGraph* graph,
                               MatchingState* state,
                               size_t start_vertex) {
    // Reset BFS state
    memset(state->in_queue, 0, graph->num_vertices * sizeof(bool));
    memset(state->previous, 0, graph->num_vertices * sizeof(size_t));
    state->queue_start = 0;
    state->queue_end = 0;

    // Add start vertex to queue
    state->queue[state->queue_end++] = start_vertex;
    state->in_queue[start_vertex] = true;

    while (state->queue_start < state->queue_end) {
        size_t current = state->queue[state->queue_start++];

        // Check all edges from current vertex
        for (size_t i = 0; i < graph->num_edges; i++) {
            const SyndromeEdge* edge = &graph->edges[i];
            
            // Skip if edge doesn't connect to current vertex
            size_t v1_idx = edge->vertex1 - graph->vertices;
            size_t v2_idx = edge->vertex2 - graph->vertices;
            if (v1_idx != current && v2_idx != current) {
                continue;
            }

            // Get other endpoint
            size_t other = (v1_idx == current) ? v2_idx : v1_idx;

            // Skip if already visited
            if (state->in_queue[other]) {
                continue;
            }

            // Check if edge is tight (zero slack)
            if (fabs(state->slack[i]) < 1e-10) {
                state->previous[other] = current;
                state->queue[state->queue_end++] = other;
                state->in_queue[other] = true;

                // Found unmatched vertex
                if (!state->matched[other]) {
                    return true;
                }
            }
        }
    }

    return false;
}

// Update matching along augmenting path
static void augment_matching(const MatchingGraph* graph,
                           MatchingState* state,
                           size_t end_vertex) {
    size_t current = end_vertex;
    while (current != 0) {
        size_t prev = state->previous[current];
        state->matched[current] = !state->matched[current];
        state->matched[prev] = !state->matched[prev];
        current = state->previous[prev];
    }
}

// Update dual variables to make progress
static void update_dual_variables(const MatchingGraph* graph,
                                MatchingState* state) {
    double min_slack = DBL_MAX;

    // Find minimum slack on edges between in_queue and not_in_queue vertices
    for (size_t i = 0; i < graph->num_edges; i++) {
        const SyndromeEdge* edge = &graph->edges[i];
        size_t v1_idx = edge->vertex1 - graph->vertices;
        size_t v2_idx = edge->vertex2 - graph->vertices;

        if (state->in_queue[v1_idx] != state->in_queue[v2_idx]) {
            if (state->slack[i] < min_slack) {
                min_slack = state->slack[i];
            }
        }
    }

    // Update dual variables and slack
    for (size_t i = 0; i < graph->num_vertices; i++) {
        if (state->in_queue[i]) {
            state->dual_vars[i] += min_slack / 2;
        }
    }

    for (size_t i = 0; i < graph->num_edges; i++) {
        const SyndromeEdge* edge = &graph->edges[i];
        size_t v1_idx = edge->vertex1 - graph->vertices;
        size_t v2_idx = edge->vertex2 - graph->vertices;

        if (state->in_queue[v1_idx] && state->in_queue[v2_idx]) {
            state->slack[i] -= min_slack;
        }
        else if (!state->in_queue[v1_idx] && !state->in_queue[v2_idx]) {
            state->slack[i] += min_slack;
        }
    }
}

// NOTE: find_minimum_weight_matching is implemented in error_syndrome.c
// with the full Blossom algorithm for minimum weight perfect matching.
// This avoids code duplication and uses the superior implementation.

// verify_syndrome_matching() - Canonical implementation in error_syndrome.c
// (removed: canonical version has chain validation and better verification)

// apply_matching_correction() - Canonical implementation in error_syndrome.c
// (removed: duplicate with less complete chain validation)
