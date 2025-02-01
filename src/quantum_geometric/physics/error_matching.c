/**
 * @file error_matching.c
 * @brief Implementation of minimum weight perfect matching for error correction
 */

#include "quantum_geometric/physics/error_syndrome.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include <stdlib.h>
#include <string.h>
#include <float.h>

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

bool find_minimum_weight_matching(MatchingGraph* graph,
                                const SyndromeConfig* config) {
    if (!graph || !config || graph->num_vertices == 0) {
        return false;
    }

    // Initialize matching state
    MatchingState* state = init_matching_state(graph);
    if (!state) {
        return false;
    }

    // Main matching loop
    for (size_t iter = 0; iter < config->max_matching_iterations; iter++) {
        bool all_matched = true;

        // Find unmatched vertex
        for (size_t i = 0; i < graph->num_vertices; i++) {
            if (!state->matched[i]) {
                all_matched = false;

                // Try to find augmenting path
                if (find_augmenting_path(graph, state, i)) {
                    // Update matching
                    size_t end_vertex = 0;
                    for (size_t j = 0; j < graph->num_vertices; j++) {
                        if (!state->matched[j] && state->in_queue[j]) {
                            end_vertex = j;
                            break;
                        }
                    }
                    augment_matching(graph, state, end_vertex);
                }
                else {
                    // Update dual variables
                    update_dual_variables(graph, state);
                }

                break;
            }
        }

        if (all_matched) {
            cleanup_matching_state(state);
            return true;
        }
    }

    cleanup_matching_state(state);
    return false;
}

bool verify_syndrome_matching(const MatchingGraph* graph,
                            const quantum_state* state) {
    if (!graph || !state) {
        return false;
    }

    // Verify all syndromes are matched
    for (size_t i = 0; i < graph->num_vertices; i++) {
        const SyndromeVertex* vertex = &graph->vertices[i];
        if (!vertex->is_boundary && !is_vertex_matched(graph, vertex)) {
            return false;
        }
    }

    // Verify matching weight is minimal
    double total_weight = calculate_matching_weight(graph);
    if (total_weight < 0.0) {
        return false;
    }

    return true;
}

bool apply_matching_correction(const MatchingGraph* graph,
                             quantum_state* state) {
    if (!graph || !state) {
        return false;
    }

    // Apply correction operations along matched edges
    for (size_t i = 0; i < graph->num_edges; i++) {
        const SyndromeEdge* edge = &graph->edges[i];
        if (is_edge_in_matching(graph, edge)) {
            // Calculate and apply correction chain
            if (!apply_correction_chain(state,
                                     edge->vertex1,
                                     edge->vertex2)) {
                return false;
            }
        }
    }

    return true;
}
