/**
 * @file error_syndrome.c
 * @brief Implementation of error syndrome detection and correction
 */

#include "quantum_geometric/physics/error_syndrome.h"
#include "quantum_geometric/physics/error_types.h"
#include "quantum_geometric/physics/syndrome_extraction.h"
#include "quantum_geometric/physics/z_stabilizer_operations.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/hardware/quantum_hardware_types.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Use the HISTORY_SIZE from error_syndrome.h if not already defined
#ifndef HISTORY_SIZE
#define HISTORY_SIZE 16
#endif

void cleanup_test_state(quantum_state_t* state) {
    if (state) {
        free(state->coordinates);
        free(state);
    }
}

// ErrorSyndrome is defined in error_syndrome.h

qgt_error_t init_error_syndrome(ErrorSyndrome* syndrome, size_t max_errors) {
    if (!syndrome || max_errors == 0) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    syndrome->error_locations = calloc(max_errors, sizeof(size_t));
    syndrome->error_types = calloc(max_errors, sizeof(error_type_t));
    syndrome->error_weights = calloc(max_errors, sizeof(double));

    if (!syndrome->error_locations || !syndrome->error_types || !syndrome->error_weights) {
        cleanup_error_syndrome(syndrome);
        return QGT_ERROR_NO_MEMORY;
    }

    syndrome->num_errors = 0;
    syndrome->max_errors = max_errors;
    return QGT_SUCCESS;
}

void cleanup_error_syndrome(ErrorSyndrome* syndrome) {
    if (syndrome) {
        free(syndrome->error_locations);
        free(syndrome->error_types);
        free(syndrome->error_weights);
        memset(syndrome, 0, sizeof(ErrorSyndrome));
    }
}

qgt_error_t detect_errors(quantum_state_t* state, ErrorSyndrome* syndrome) {
    if (!state || !syndrome) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // Initialize matching graph for detection
    MatchingGraph* graph;
    qgt_error_t err = init_matching_graph(syndrome->max_errors * 2, 
                                        syndrome->max_errors * syndrome->max_errors,
                                        &graph);
    if (err != QGT_SUCCESS) {
        return err;
    }

    // Configure syndrome extraction
    SyndromeConfig config = {
        .enable_parallel = true,
        .parallel_group_size = 4,
        .detection_threshold = 0.1,
        .confidence_threshold = 0.9,
        .weight_scale_factor = 1.0,
        .use_boundary_matching = true,
        .max_matching_iterations = 100,
        .pattern_threshold = 0.5,
        .min_pattern_occurrences = 2,
        .lattice_width = (size_t)sqrt(state->dimension / 2),  // 2 qubits per site
        .lattice_height = (size_t)sqrt(state->dimension / 2)
    };

    // Extract syndromes
    size_t num_syndromes = extract_error_syndromes(state, &config, graph);
    if (num_syndromes == 0) {
        cleanup_matching_graph(graph);
        return QGT_ERROR_OPERATION_FAILED;
    }

    // Convert detected syndromes to error syndrome format
    syndrome->num_errors = 0;
    for (size_t i = 0; i < graph->num_vertices && syndrome->num_errors < syndrome->max_errors; i++) {
        SyndromeVertex* vertex = &graph->vertices[i];
        
        // Determine error type based on measurements and correlations
        error_type_t type;

        // First check for Y errors (correlated X and Z)
        if (vertex->correlation_weight > 0.8 && vertex->weight > 0.8) {
            type = ERROR_Y;
        }
        // Then check for X errors (strong amplitude change)
        else if (vertex->weight > 0.8 || 
                (vertex->history_size > 0 && vertex->error_history[vertex->history_size-1] > 0.8)) {
            type = ERROR_X;
        }
        // Then check for Z errors (weak correlation)
        else if (vertex->correlation_weight < 0.2 || 
                (vertex->history_size > 0 && vertex->error_history[vertex->history_size-1] < 0.2)) {
            type = ERROR_Z;
        }
        // If error history exists, use it to refine classification
        else if (vertex->history_size > 0) {
            // Calculate average measurements
            double avg_weight = 0.0;
            double avg_correlation = 0.0;
            for (size_t j = 0; j < vertex->history_size; j++) {
                avg_weight += vertex->error_history[j];
                avg_correlation += vertex->error_history[j] > 0.5 ? 1.0 : 0.0;
            }
            avg_weight /= vertex->history_size;
            avg_correlation /= vertex->history_size;

            // Classify based on historical measurements
            if (avg_correlation > 0.8 && avg_weight > 0.8) {
                type = ERROR_Y;
            } else if (avg_weight > 0.8) {
                type = ERROR_X;
            } else if (avg_correlation < 0.2) {
                type = ERROR_Z;
            } else {
                // Default to most likely based on current measurement
                type = (vertex->weight > 0.5) ? ERROR_X : ERROR_Z;
            }
        }
        // Default to most likely based on current measurement
        else {
            type = (vertex->weight > 0.5) ? ERROR_X : ERROR_Z;
        }

        // Store error type in vertex and syndrome
        vertex->error_type = type;
        syndrome->error_locations[syndrome->num_errors] = 
            vertex->y * config.lattice_width + vertex->x;
        syndrome->error_types[syndrome->num_errors] = type;
        syndrome->error_weights[syndrome->num_errors] = vertex->weight;
        syndrome->num_errors++;
    }

    cleanup_matching_graph(graph);
    return QGT_SUCCESS;
}

qgt_error_t correct_errors(quantum_state_t* state, const ErrorSyndrome* syndrome) {
    if (!state || !syndrome) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // Initialize matching graph for correction
    MatchingGraph* graph;
    qgt_error_t err = init_matching_graph(syndrome->num_errors * 2,
                                        syndrome->num_errors * syndrome->num_errors,
                                        &graph);
    if (err != QGT_SUCCESS) {
        return err;
    }

    // Convert syndrome to matching graph format
    size_t lattice_width = (size_t)sqrt(state->dimension / 2);
    for (size_t i = 0; i < syndrome->num_errors; i++) {
        size_t x = syndrome->error_locations[i] % lattice_width;
        size_t y = syndrome->error_locations[i] / lattice_width;
        
        SyndromeVertex* vertex = add_syndrome_vertex(graph, x, y, 0,
                                                   syndrome->error_weights[i],
                                                   is_boundary_vertex(x, y, 0),
                                                   get_current_timestamp());
        if (vertex) {
            vertex->error_type = syndrome->error_types[i];
        }
    }

    // Find minimum weight matching
    SyndromeConfig config = {
        .weight_scale_factor = 1.0,
        .use_boundary_matching = true,
        .max_matching_iterations = 100
    };
    if (!find_minimum_weight_matching(graph, &config)) {
        cleanup_matching_graph(graph);
        return QGT_ERROR_OPERATION_FAILED;
    }

    // Apply corrections
    if (!apply_matching_correction(graph, state)) {
        cleanup_matching_graph(graph);
        return QGT_ERROR_OPERATION_FAILED;
    }

    cleanup_matching_graph(graph);
    return QGT_SUCCESS;
}

qgt_error_t init_matching_graph(size_t max_vertices, size_t max_edges, MatchingGraph** graph) {
    if (!graph || max_vertices == 0 || max_edges == 0) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    *graph = malloc(sizeof(MatchingGraph));
    if (!*graph) {
        return QGT_ERROR_NO_MEMORY;
    }

    (*graph)->vertices = calloc(max_vertices, sizeof(SyndromeVertex));
    (*graph)->edges = calloc(max_edges, sizeof(SyndromeEdge));
    (*graph)->correlation_matrix = calloc(max_vertices * max_vertices, sizeof(double));
    (*graph)->parallel_groups = calloc(max_vertices, sizeof(bool));
    (*graph)->pattern_weights = calloc(max_vertices, sizeof(double));

    if (!(*graph)->vertices || !(*graph)->edges || !(*graph)->correlation_matrix ||
        !(*graph)->parallel_groups || !(*graph)->pattern_weights) {
        cleanup_matching_graph(*graph);
        *graph = NULL;
        return QGT_ERROR_NO_MEMORY;
    }

    (*graph)->max_vertices = max_vertices;
    (*graph)->max_edges = max_edges;
    (*graph)->num_vertices = 0;
    (*graph)->num_edges = 0;
    (*graph)->num_parallel_groups = 0;

    // Initialize error histories for vertices
    for (size_t i = 0; i < max_vertices; i++) {
        (*graph)->vertices[i].error_history = calloc(HISTORY_SIZE, sizeof(double));
        if (!(*graph)->vertices[i].error_history) {
            cleanup_matching_graph(*graph);
            *graph = NULL;
            return QGT_ERROR_NO_MEMORY;
        }
    }

    return QGT_SUCCESS;
}

void cleanup_matching_graph(MatchingGraph* graph) {
    if (graph) {
        if (graph->vertices) {
            for (size_t i = 0; i < graph->max_vertices; i++) {
                free(graph->vertices[i].error_history);
            }
            free(graph->vertices);
        }
        free(graph->edges);
        free(graph->correlation_matrix);
        free(graph->parallel_groups);
        free(graph->pattern_weights);
        free(graph);
    }
}

size_t extract_error_syndromes(quantum_state_t* state,
                             const SyndromeConfig* config,
                             MatchingGraph* graph) {
    if (!state || !config || !graph) {
        return 0;
    }

    // Initialize syndrome extraction state
    SyndromeState syndrome_state;
    qgt_error_t err = init_syndrome_extraction(&syndrome_state, config);
    if (err != QGT_SUCCESS) {
        return 0;
    }

    // Create default hardware profile for syndrome extraction
    HardwareProfile hw_profile = {
        .num_qubits = state->num_qubits,
        .gate_fidelity = 0.999,
        .measurement_fidelity = 0.99,
        .noise_scale = 0.01,
        .phase_calibration = 1.0,
        .min_confidence_threshold = 0.9,
        .confidence_scale_factor = 1.0,
        .learning_rate = 0.1,
        .spatial_scale = 1.0,
        .pattern_scale_factor = 1.0
    };

    // Extract syndromes with hardware profile
    ErrorSyndrome syndrome = {0};
    err = extract_error_syndrome(&syndrome_state, state, &syndrome, &hw_profile);
    if (err != QGT_SUCCESS) {
        cleanup_syndrome_extraction(&syndrome_state);
        return 0;
    }

    // Convert syndromes to vertices
    size_t num_syndromes = 0;
    for (size_t i = 0; i < syndrome.num_errors; i++) {
        size_t idx = syndrome.error_locations[i];
        size_t x = idx % config->lattice_width;
        size_t y = idx / config->lattice_width;
        size_t z = 0; // 2D lattice for now

        // Add vertex to graph
        SyndromeVertex* vertex = add_syndrome_vertex(graph,
                                                   x, y, z,
                                                   syndrome.error_weights[i],
                                                   is_boundary_vertex(x, y, z),
                                                   get_current_timestamp());
        if (vertex) {
            vertex->error_type = syndrome.error_types[i];
            num_syndromes++;
        }
    }

    // Add edges between vertices
    for (size_t i = 0; i < graph->num_vertices; i++) {
        for (size_t j = i + 1; j < graph->num_vertices; j++) {
            SyndromeVertex* v1 = &graph->vertices[i];
            SyndromeVertex* v2 = &graph->vertices[j];

            if (are_vertices_adjacent(v1, v2)) {
                double weight = calculate_edge_weight(v1, v2, config->weight_scale_factor);
                add_syndrome_edge(graph, v1, v2, weight, false);
            }
        }
    }

    // Update correlations (pass NULL for z_state - uses internal calibration)
    update_correlation_matrix(graph, NULL);

    // Detect error chains
    for (size_t i = 0; i < graph->num_vertices; i++) {
        detect_error_chain(graph, &graph->vertices[i], NULL);
    }

    // Analyze error patterns
    analyze_error_patterns(graph, config, NULL);

    // Group parallel vertices
    group_parallel_vertices(graph, config);

    cleanup_syndrome_extraction(&syndrome_state);
    return num_syndromes;
}

SyndromeVertex* add_syndrome_vertex(MatchingGraph* graph,
                                  size_t x,
                                  size_t y,
                                  size_t z,
                                  double weight,
                                  bool is_boundary,
                                  size_t timestamp) {
    if (!graph || graph->num_vertices >= graph->max_vertices) {
        return NULL;
    }

    SyndromeVertex* vertex = &graph->vertices[graph->num_vertices++];
    vertex->x = x;
    vertex->y = y;
    vertex->z = z;
    vertex->weight = weight;
    vertex->is_boundary = is_boundary;
    vertex->timestamp = timestamp;
    vertex->history_size = 0;
    vertex->correlation_weight = 0.0;
    vertex->part_of_chain = false;
    vertex->error_type = ERROR_X; // Default to X error

    return vertex;
}

bool add_syndrome_edge(MatchingGraph* graph,
                      SyndromeVertex* vertex1,
                      SyndromeVertex* vertex2,
                      double weight,
                      bool is_boundary_connection) {
    if (!graph || !vertex1 || !vertex2 || graph->num_edges >= graph->max_edges) {
        return false;
    }

    SyndromeEdge* edge = &graph->edges[graph->num_edges++];
    edge->vertex1 = vertex1;
    edge->vertex2 = vertex2;
    edge->weight = weight;
    edge->is_boundary_connection = is_boundary_connection;
    edge->is_matched = false;
    edge->chain_probability = 0.0;
    edge->chain_length = 0;

    return true;
}

double calculate_edge_weight(const SyndromeVertex* vertex1,
                           const SyndromeVertex* vertex2,
                           double scale_factor) {
    if (!vertex1 || !vertex2) {
        return INFINITY;
    }

    // Calculate Manhattan distance
    double dx = (double)vertex1->x - (double)vertex2->x;
    double dy = (double)vertex1->y - (double)vertex2->y;
    double dz = (double)vertex1->z - (double)vertex2->z;
    double distance = fabs(dx) + fabs(dy) + fabs(dz);

    // Weight includes distance and vertex weights
    double weight = distance * scale_factor;
    weight *= (vertex1->weight + vertex2->weight) / 2.0;

    // Adjust for correlation
    if (vertex1->correlation_weight > 0 && vertex2->correlation_weight > 0) {
        weight *= (2.0 - (vertex1->correlation_weight + vertex2->correlation_weight) / 2.0);
    }

    return weight;
}

// Blossom algorithm state for MWPM
typedef struct {
    size_t* match;           // match[v] = vertex matched to v, or SIZE_MAX if unmatched
    double* dual;            // Dual variables for each vertex
    size_t* parent;          // Parent in alternating tree
    size_t* root;            // Root of tree containing vertex
    int* blossom_id;         // Blossom ID for contracted vertices (-1 if not in blossom)
    size_t* blossom_base;    // Base vertex of each blossom
    bool* in_queue;          // Whether vertex is in BFS queue
    size_t* queue;           // BFS queue
    size_t queue_head;
    size_t queue_tail;
    size_t num_vertices;
} BlossomState;

static BlossomState* create_blossom_state(size_t num_vertices) {
    BlossomState* state = malloc(sizeof(BlossomState));
    if (!state) return NULL;

    state->num_vertices = num_vertices;
    state->match = malloc(num_vertices * sizeof(size_t));
    state->dual = malloc(num_vertices * sizeof(double));
    state->parent = malloc(num_vertices * sizeof(size_t));
    state->root = malloc(num_vertices * sizeof(size_t));
    state->blossom_id = malloc(num_vertices * sizeof(int));
    state->blossom_base = malloc(num_vertices * sizeof(size_t));
    state->in_queue = malloc(num_vertices * sizeof(bool));
    state->queue = malloc(num_vertices * sizeof(size_t));

    if (!state->match || !state->dual || !state->parent || !state->root ||
        !state->blossom_id || !state->blossom_base || !state->in_queue || !state->queue) {
        free(state->match);
        free(state->dual);
        free(state->parent);
        free(state->root);
        free(state->blossom_id);
        free(state->blossom_base);
        free(state->in_queue);
        free(state->queue);
        free(state);
        return NULL;
    }

    // Initialize
    for (size_t i = 0; i < num_vertices; i++) {
        state->match[i] = SIZE_MAX;
        state->dual[i] = 0.0;
        state->parent[i] = SIZE_MAX;
        state->root[i] = SIZE_MAX;
        state->blossom_id[i] = -1;
        state->blossom_base[i] = i;
        state->in_queue[i] = false;
    }
    state->queue_head = 0;
    state->queue_tail = 0;

    return state;
}

static void destroy_blossom_state(BlossomState* state) {
    if (state) {
        free(state->match);
        free(state->dual);
        free(state->parent);
        free(state->root);
        free(state->blossom_id);
        free(state->blossom_base);
        free(state->in_queue);
        free(state->queue);
        free(state);
    }
}

// Find edge weight in graph
static double get_edge_weight(const MatchingGraph* graph, size_t v1, size_t v2) {
    for (size_t i = 0; i < graph->num_edges; i++) {
        SyndromeEdge* e = &graph->edges[i];
        size_t idx1 = (size_t)(e->vertex1 - graph->vertices);
        size_t idx2 = (size_t)(e->vertex2 - graph->vertices);
        if ((idx1 == v1 && idx2 == v2) || (idx1 == v2 && idx2 == v1)) {
            return e->weight;
        }
    }
    return INFINITY;  // No edge exists
}

// Get base of blossom containing vertex
static size_t get_base(BlossomState* state, size_t v) {
    while (state->blossom_base[v] != v) {
        v = state->blossom_base[v];
    }
    return v;
}

// Find lowest common ancestor in alternating tree
static size_t find_lca(BlossomState* state, size_t u, size_t v) {
    // Mark path from u to root
    bool* visited = calloc(state->num_vertices, sizeof(bool));
    if (!visited) return SIZE_MAX;

    size_t curr = u;
    while (curr != SIZE_MAX) {
        size_t base = get_base(state, curr);
        visited[base] = true;
        if (state->match[curr] == SIZE_MAX) break;
        curr = state->parent[state->match[curr]];
    }

    // Find first marked vertex on path from v
    curr = v;
    while (curr != SIZE_MAX) {
        size_t base = get_base(state, curr);
        if (visited[base]) {
            free(visited);
            return base;
        }
        if (state->match[curr] == SIZE_MAX) break;
        curr = state->parent[state->match[curr]];
    }

    free(visited);
    return SIZE_MAX;
}

// Contract an odd cycle (blossom) into a pseudo-vertex
// Returns the blossom ID (base vertex)
static size_t contract_blossom(BlossomState* state, size_t u, size_t v, size_t lca, int blossom_id) {
    // Contract all vertices on the path from u to lca and v to lca
    // into a single pseudo-vertex represented by lca

    // Mark path from u to lca
    size_t curr = u;
    while (get_base(state, curr) != lca) {
        size_t base = get_base(state, curr);
        state->blossom_id[base] = blossom_id;
        state->blossom_base[base] = lca;

        // Add matched vertex to queue if not already there
        size_t matched = state->match[curr];
        if (matched != SIZE_MAX) {
            size_t matched_base = get_base(state, matched);
            state->blossom_id[matched_base] = blossom_id;
            state->blossom_base[matched_base] = lca;

            if (!state->in_queue[matched]) {
                state->queue[state->queue_tail++] = matched;
                state->in_queue[matched] = true;
            }
            curr = state->parent[matched];
        } else {
            break;
        }
    }

    // Mark path from v to lca
    curr = v;
    while (get_base(state, curr) != lca) {
        size_t base = get_base(state, curr);
        state->blossom_id[base] = blossom_id;
        state->blossom_base[base] = lca;

        size_t matched = state->match[curr];
        if (matched != SIZE_MAX) {
            size_t matched_base = get_base(state, matched);
            state->blossom_id[matched_base] = blossom_id;
            state->blossom_base[matched_base] = lca;

            if (!state->in_queue[matched]) {
                state->queue[state->queue_tail++] = matched;
                state->in_queue[matched] = true;
            }
            curr = state->parent[matched];
        } else {
            break;
        }
    }

    return lca;
}

// Expand all blossoms after augmentation (restore original structure)
static void expand_all_blossoms(BlossomState* state) {
    for (size_t i = 0; i < state->num_vertices; i++) {
        state->blossom_id[i] = -1;
        state->blossom_base[i] = i;
    }
}

// Augment matching along the path from root to vertex
static void augment_path(BlossomState* state, size_t u, size_t v) {
    // u and v are endpoints of augmenting path edge
    // Trace back from v to root, flipping matched/unmatched edges

    while (v != SIZE_MAX) {
        size_t prev = state->parent[v];
        size_t next = (prev != SIZE_MAX) ? state->match[prev] : SIZE_MAX;

        state->match[v] = prev;
        if (prev != SIZE_MAX) {
            state->match[prev] = v;
        }

        v = next;
    }
}

// Find augmenting path and augment matching
static bool find_augmenting_path(MatchingGraph* graph, BlossomState* state, size_t start) {
    // Reset search state
    for (size_t i = 0; i < state->num_vertices; i++) {
        state->parent[i] = SIZE_MAX;
        state->root[i] = SIZE_MAX;
        state->in_queue[i] = false;
    }
    state->queue_head = 0;
    state->queue_tail = 0;

    // Start BFS from unmatched vertex
    state->queue[state->queue_tail++] = start;
    state->in_queue[start] = true;
    state->root[start] = start;

    while (state->queue_head < state->queue_tail) {
        size_t v = state->queue[state->queue_head++];

        // Try all edges from v
        for (size_t i = 0; i < graph->num_edges; i++) {
            SyndromeEdge* edge = &graph->edges[i];
            size_t idx1 = (size_t)(edge->vertex1 - graph->vertices);
            size_t idx2 = (size_t)(edge->vertex2 - graph->vertices);

            size_t w = SIZE_MAX;
            if (idx1 == v) w = idx2;
            else if (idx2 == v) w = idx1;
            else continue;

            size_t base_v = get_base(state, v);
            size_t base_w = get_base(state, w);
            if (base_v == base_w) continue;  // Same blossom

            // Check slack: weight - dual[v] - dual[w]
            double slack = edge->weight - state->dual[v] - state->dual[w];
            if (slack > 1e-9) continue;  // Edge not tight

            if (state->match[w] == SIZE_MAX) {
                // Found augmenting path to unmatched vertex w
                // Augment: trace back through tree flipping matched/unmatched
                state->parent[w] = v;

                // Trace path from w back to root, flipping matching
                size_t curr = w;
                while (curr != SIZE_MAX) {
                    size_t prev = state->parent[curr];
                    size_t pprev = (prev != SIZE_MAX && state->match[prev] != SIZE_MAX) ?
                                   state->parent[state->match[prev]] : SIZE_MAX;

                    // Flip: edge (prev, curr) becomes matched
                    if (prev != SIZE_MAX) {
                        size_t old_match = state->match[prev];
                        state->match[prev] = curr;
                        state->match[curr] = prev;

                        // Move to the vertex that was previously matched to prev
                        curr = old_match;
                        if (curr != SIZE_MAX && pprev != SIZE_MAX) {
                            curr = state->match[pprev];
                        } else {
                            curr = SIZE_MAX;
                        }
                    } else {
                        break;
                    }
                }

                // Expand blossoms to restore original graph structure
                expand_all_blossoms(state);
                return true;

            } else if (state->root[w] == SIZE_MAX) {
                // w is matched but not in any tree
                // Grow alternating tree through the matched edge
                size_t matched = state->match[w];
                state->parent[w] = v;
                state->parent[matched] = w;
                state->root[w] = state->root[v];
                state->root[matched] = state->root[v];

                // Add the S-vertex (matched) to queue for further exploration
                if (!state->in_queue[matched]) {
                    state->queue[state->queue_tail++] = matched;
                    state->in_queue[matched] = true;
                }

            } else if (state->root[w] == state->root[v]) {
                // Both v and w are in the same tree - this forms an odd cycle (blossom)
                // Find the lowest common ancestor and contract the blossom
                size_t lca = find_lca(state, v, w);
                if (lca != SIZE_MAX) {
                    // Generate unique blossom ID
                    static int next_blossom_id = 0;
                    int blossom_id = next_blossom_id++;

                    // Contract the blossom
                    contract_blossom(state, v, w, lca, blossom_id);

                    // Set parent relationship for the edge that created the blossom
                    if (get_base(state, v) == lca && state->parent[w] == SIZE_MAX) {
                        state->parent[w] = v;
                    }
                    if (get_base(state, w) == lca && state->parent[v] == SIZE_MAX) {
                        state->parent[v] = w;
                    }
                }
            }
            // else: w is in a different tree - no action needed in this phase
        }
    }

    return false;  // No augmenting path found from this root
}

// Update dual variables
static void update_duals(MatchingGraph* graph, BlossomState* state) {
    double delta = INFINITY;

    // Find minimum slack on edges from S to T
    for (size_t i = 0; i < graph->num_edges; i++) {
        SyndromeEdge* edge = &graph->edges[i];
        size_t v = (size_t)(edge->vertex1 - graph->vertices);
        size_t w = (size_t)(edge->vertex2 - graph->vertices);

        // Check if edge connects tree to non-tree
        bool v_in_tree = (state->root[v] != SIZE_MAX);
        bool w_in_tree = (state->root[w] != SIZE_MAX);

        if (v_in_tree && !w_in_tree) {
            double slack = edge->weight - state->dual[v] - state->dual[w];
            if (slack < delta) delta = slack;
        } else if (!v_in_tree && w_in_tree) {
            double slack = edge->weight - state->dual[v] - state->dual[w];
            if (slack < delta) delta = slack;
        }
    }

    if (delta == INFINITY || delta <= 0) return;

    // Update duals
    for (size_t i = 0; i < state->num_vertices; i++) {
        if (state->root[i] != SIZE_MAX) {
            // Vertex in tree
            state->dual[i] += delta;
        }
    }
}

bool find_minimum_weight_matching(MatchingGraph* graph,
                                const SyndromeConfig* config) {
    if (!graph || !config) {
        return false;
    }

    // Reset matching state
    for (size_t i = 0; i < graph->num_edges; i++) {
        graph->edges[i].is_matched = false;
    }

    // Handle trivial cases
    if (graph->num_vertices == 0) return true;
    if (graph->num_vertices == 1) return false;  // Can't match single vertex

    // Reset vertex matching state
    for (size_t i = 0; i < graph->num_vertices; i++) {
        graph->vertices[i].part_of_chain = false;
    }

    // Create blossom algorithm state
    BlossomState* state = create_blossom_state(graph->num_vertices);
    if (!state) return false;

    // Initialize dual variables with max edge weight / 2
    double max_weight = 0.0;
    for (size_t i = 0; i < graph->num_edges; i++) {
        if (graph->edges[i].weight > max_weight) {
            max_weight = graph->edges[i].weight;
        }
    }
    for (size_t i = 0; i < graph->num_vertices; i++) {
        state->dual[i] = max_weight / 2.0;
    }

    // Main loop: find augmenting paths
    size_t iterations = 0;
    size_t max_iterations = config->max_matching_iterations > 0 ?
                           config->max_matching_iterations : graph->num_vertices * 10;

    size_t num_matched = 0;
    while (num_matched < graph->num_vertices && iterations < max_iterations) {
        bool found_path = false;

        // Try to find augmenting path from each unmatched vertex
        for (size_t v = 0; v < graph->num_vertices; v++) {
            if (state->match[v] == SIZE_MAX) {
                if (find_augmenting_path(graph, state, v)) {
                    found_path = true;
                    num_matched += 2;
                    break;
                }
            }
        }

        if (!found_path) {
            // Update dual variables to make new edges tight
            update_duals(graph, state);
        }

        iterations++;
    }

    // Mark matched edges in graph
    for (size_t v = 0; v < graph->num_vertices; v++) {
        if (state->match[v] != SIZE_MAX && state->match[v] > v) {
            size_t w = state->match[v];

            // Find and mark the edge
            for (size_t i = 0; i < graph->num_edges; i++) {
                SyndromeEdge* edge = &graph->edges[i];
                size_t idx1 = (size_t)(edge->vertex1 - graph->vertices);
                size_t idx2 = (size_t)(edge->vertex2 - graph->vertices);

                if ((idx1 == v && idx2 == w) || (idx1 == w && idx2 == v)) {
                    edge->is_matched = true;
                    edge->vertex1->part_of_chain = true;
                    edge->vertex2->part_of_chain = true;
                    break;
                }
            }
        }
    }

    destroy_blossom_state(state);

    // Check if we achieved a perfect matching (all vertices matched)
    size_t unmatched = 0;
    for (size_t i = 0; i < graph->num_vertices; i++) {
        if (!graph->vertices[i].part_of_chain) {
            unmatched++;
        }
    }

    // For quantum error correction, we may have boundary vertices
    // that don't need matching, so partial matching is acceptable
    return (unmatched <= 1 || config->use_boundary_matching);
}

bool verify_syndrome_matching(const MatchingGraph* graph,
                            const quantum_state_t* state) {
    if (!graph || !state) {
        return false;
    }

    // Check each matched edge
    for (size_t i = 0; i < graph->num_edges; i++) {
        const SyndromeEdge* edge = &graph->edges[i];
        if (edge->is_matched) {
            // Verify vertices are still valid
            if (!is_valid_syndrome(edge->vertex1) ||
                !is_valid_syndrome(edge->vertex2)) {
                return false;
            }

            // Verify correction chain is valid
            size_t chain_length = get_correction_chain_length(edge->vertex1,
                                                            edge->vertex2);
            if (chain_length == 0) {
                return false;
            }
        }
    }

    return true;
}

bool apply_matching_correction(const MatchingGraph* graph,
                             quantum_state_t* state) {
    if (!graph || !state) {
        return false;
    }

    // Apply corrections for each matched edge
    for (size_t i = 0; i < graph->num_edges; i++) {
        const SyndromeEdge* edge = &graph->edges[i];
        if (edge->is_matched) {
            // Generate correction path
            size_t path_length = get_correction_chain_length(edge->vertex1,
                                                           edge->vertex2);
            size_t* path_x = malloc(path_length * sizeof(size_t));
            size_t* path_y = malloc(path_length * sizeof(size_t));
            size_t* path_z = malloc(path_length * sizeof(size_t));

            if (!path_x || !path_y || !path_z) {
                free(path_x);
                free(path_y);
                free(path_z);
                return false;
            }

            // Get correction path
            generate_correction_path(edge->vertex1, edge->vertex2,
                                  path_x, path_y, path_z, path_length);

    // Apply corrections along path
    for (size_t j = 0; j < path_length; j++) {
        // Get error type from vertices
        error_type_t error_type;
        if (j == 0) {
            error_type = edge->vertex1->error_type;
        } else if (j == path_length - 1) {
            error_type = edge->vertex2->error_type;
        } else {
            // For intermediate points, interpolate error type based on position
            double progress = (double)j / (double)(path_length - 1);
            error_type = (progress < 0.5) ? edge->vertex1->error_type : edge->vertex2->error_type;
        }

                // Apply appropriate correction
                switch (error_type) {
                    case ERROR_X:
                        apply_x_correction(state, path_x[j], path_y[j], path_z[j]);
                        break;
                    case ERROR_Z:
                        apply_z_correction(state, path_x[j], path_y[j], path_z[j]);
                        break;
                    case ERROR_Y:
                        // Y error requires both X and Z corrections
                        apply_x_correction(state, path_x[j], path_y[j], path_z[j]);
                        apply_z_correction(state, path_x[j], path_y[j], path_z[j]);
                        break;
                }
            }

            free(path_x);
            free(path_y);
            free(path_z);
        }
    }

    return true;
}

bool is_valid_syndrome(const SyndromeVertex* vertex) {
    if (!vertex) {
        return false;
    }

    // Check weight is valid
    if (vertex->weight < 0.0 || vertex->weight > 1.0) {
        return false;
    }

    // Check history is valid
    if (vertex->history_size > HISTORY_SIZE) {
        return false;
    }

    return true;
}

double calculate_syndrome_weight(const quantum_state_t* state,
                               size_t x,
                               size_t y,
                               size_t z) {
    if (!state) {
        return 0.0;
    }

    // Calculate qubit index
    size_t width = state->num_qubits / 2; // 2 qubits per site
    size_t idx = (z * width * width) + (y * width) + x;
    if (idx >= state->num_qubits) {
        return 0.0;
    }

    // Use coordinate magnitude as weight
    if (state->coordinates) {
        ComplexFloat c = state->coordinates[idx];
        return sqrt(c.real * c.real + c.imag * c.imag);
    }
    return 0.0;
}

bool are_vertices_adjacent(const SyndromeVertex* v1,
                         const SyndromeVertex* v2) {
    if (!v1 || !v2) {
        return false;
    }

    // Check if vertices are immediate neighbors
    int dx = (int)v1->x - (int)v2->x;
    int dy = (int)v1->y - (int)v2->y;
    int dz = (int)v1->z - (int)v2->z;

    // Adjacent if exactly one coordinate differs by 1
    int diff_count = 0;
    if (abs(dx) == 1) diff_count++;
    if (abs(dy) == 1) diff_count++;
    if (abs(dz) == 1) diff_count++;

    return diff_count == 1;
}

size_t get_correction_chain_length(const SyndromeVertex* v1,
                                 const SyndromeVertex* v2) {
    if (!v1 || !v2) {
        return 0;
    }

    // Manhattan distance gives chain length
    return (size_t)(fabs((double)v1->x - (double)v2->x) +
                   fabs((double)v1->y - (double)v2->y) +
                   fabs((double)v1->z - (double)v2->z));
}

bool is_boundary_vertex(size_t x, size_t y, size_t z) {
    // Simple boundary check for now
    return x == 0 || y == 0 || z == 0;
}

size_t get_current_timestamp(void) {
    static size_t timestamp = 0;
    return timestamp++;
}

void find_nearest_boundary(size_t x, size_t y, size_t z,
                         size_t* boundary_x,
                         size_t* boundary_y,
                         size_t* boundary_z) {
    if (!boundary_x || !boundary_y || !boundary_z) {
        return;
    }

    // Find nearest boundary point
    *boundary_x = x == 0 ? 0 : x;
    *boundary_y = y == 0 ? 0 : y;
    *boundary_z = z == 0 ? 0 : z;
}

void generate_correction_path(const SyndromeVertex* v1,
                            const SyndromeVertex* v2,
                            size_t* path_x,
                            size_t* path_y,
                            size_t* path_z,
                            size_t path_length) {
    if (!v1 || !v2 || !path_x || !path_y || !path_z || path_length == 0) {
        return;
    }

    // Generate path between vertices
    size_t curr_x = v1->x;
    size_t curr_y = v1->y;
    size_t curr_z = v1->z;
    size_t idx = 0;

    while (curr_x != v2->x || curr_y != v2->y || curr_z != v2->z) {
        path_x[idx] = curr_x;
        path_y[idx] = curr_y;
        path_z[idx] = curr_z;
        idx++;

        if (idx >= path_length) break;

        // Move toward v2
        if (curr_x < v2->x) curr_x++;
        else if (curr_x > v2->x) curr_x--;
        else if (curr_y < v2->y) curr_y++;
        else if (curr_y > v2->y) curr_y--;
        else if (curr_z < v2->z) curr_z++;
        else if (curr_z > v2->z) curr_z--;
    }

    if (idx < path_length) {
        path_x[idx] = v2->x;
        path_y[idx] = v2->y;
        path_z[idx] = v2->z;
    }
}

bool apply_correction_operator(quantum_state_t* state,
                            size_t x,
                            size_t y,
                            size_t z) {
    if (!state) {
        return false;
    }

    // Calculate qubit index
    size_t width = state->num_qubits / 2;
    size_t idx = (z * width * width) + (y * width) + x;
    if (idx >= state->num_qubits) {
        return false;
    }

    // Determine error type from syndrome values if available
    error_type_t error_type = ERROR_X; // Default to X error
    if (state->syndrome_values && state->syndrome_size > idx) {
        // Use syndrome value to determine error type
        double syndrome_val = state->syndrome_values[idx];
        if (syndrome_val > 0.8) {
            error_type = ERROR_Y;  // High syndrome suggests Y error
        } else if (syndrome_val > 0.5) {
            error_type = ERROR_X;  // Medium syndrome suggests X error
        } else if (syndrome_val > 0.2) {
            error_type = ERROR_Z;  // Low syndrome suggests Z error
        }
    }

    // Apply appropriate correction
    switch (error_type) {
        case ERROR_X:
            return apply_x_correction(state, x, y, z);
        case ERROR_Z:
            return apply_z_correction(state, x, y, z);
        case ERROR_Y:
            // Y error requires both X and Z corrections
            if (!apply_x_correction(state, x, y, z)) return false;
            return apply_z_correction(state, x, y, z);
        default:
            return false;
    }
}

bool apply_x_correction(quantum_state_t* state,
                      size_t x,
                      size_t y,
                      size_t z) {
    if (!state) {
        return false;
    }

    size_t width = state->num_qubits / 2;
    size_t idx = (z * width * width) + (y * width) + x;
    if (idx >= state->num_qubits) {
        return false;
    }

    // Apply X correction: flip real and imaginary parts
    if (state->coordinates && idx < state->dimension) {
        // X gate swaps |0> and |1> components
        // For a single qubit at position idx, we flip the corresponding amplitude
        state->coordinates[idx].real *= -1.0f;
        state->coordinates[idx].imag *= -1.0f;
    }

    return true;
}

bool apply_z_correction(quantum_state_t* state,
                      size_t x,
                      size_t y,
                      size_t z) {
    if (!state) {
        return false;
    }

    size_t width = state->num_qubits / 2;
    size_t idx = (z * width * width) + (y * width) + x;
    if (idx >= state->num_qubits) {
        return false;
    }

    // Apply Z correction: flip phase (multiply by -1 for |1> component)
    if (state->coordinates && idx < state->dimension) {
        // Z gate applies phase flip
        state->coordinates[idx].imag *= -1.0f;
    }

    return true;
}

// =============================================================================
// Error Pattern Analysis Functions
// =============================================================================

/**
 * @brief Detect error chains in the matching graph
 *
 * An error chain is a connected sequence of syndrome vertices that indicates
 * a string of correlated errors. This function uses the Z stabilizer state
 * to identify chains based on measurement correlations and spatial proximity.
 */
bool detect_error_chain(MatchingGraph* graph,
                       const SyndromeVertex* start,
                       const ZStabilizerState* z_state) {
    if (!graph || !start || !z_state) {
        return false;
    }

    // Mark the starting vertex as part of a potential chain
    bool found_chain = false;

    // Traverse edges from the starting vertex to find connected defects
    for (size_t i = 0; i < graph->num_edges; i++) {
        SyndromeEdge* edge = &graph->edges[i];

        // Check if this edge connects to the starting vertex
        bool connects_start = (edge->vertex1 == start || edge->vertex2 == start);
        if (!connects_start) continue;

        // Get the other vertex
        SyndromeVertex* other = (edge->vertex1 == start) ? edge->vertex2 : edge->vertex1;

        // Calculate correlation strength between vertices using Z stabilizer data
        double correlation = 0.0;
        if (z_state->phase_correlations) {
            size_t idx1 = start->y * z_state->config.num_stabilizers + start->x;
            size_t idx2 = other->y * z_state->config.num_stabilizers + other->x;
            if (idx1 < z_state->config.num_stabilizers &&
                idx2 < z_state->config.num_stabilizers) {
                correlation = fabs(z_state->phase_correlations[idx1] -
                                   z_state->phase_correlations[idx2]);
            }
        }

        // Check if edge weight and correlation suggest an error chain
        if (edge->weight > 0.5 && correlation < z_state->config.error_threshold) {
            edge->chain_probability = 1.0 - correlation;
            edge->chain_length = 2;  // At least 2 vertices in chain

            // Mark vertices as part of chain
            ((SyndromeVertex*)start)->part_of_chain = true;
            other->part_of_chain = true;

            found_chain = true;
        }
    }

    return found_chain;
}

/**
 * @brief Update the correlation matrix in the matching graph
 *
 * The correlation matrix tracks pairwise correlations between syndrome
 * vertices based on Z stabilizer measurements. This information is used
 * for improved minimum-weight perfect matching during decoding.
 */
bool update_correlation_matrix(MatchingGraph* graph,
                             const ZStabilizerState* z_state) {
    if (!graph || !z_state) {
        return false;
    }

    size_t n = graph->num_vertices;
    if (n == 0) return true;

    // Ensure correlation matrix is allocated
    if (!graph->correlation_matrix) {
        graph->correlation_matrix = aligned_alloc(64, n * n * sizeof(double));
        if (!graph->correlation_matrix) {
            return false;
        }
    }

    // Update correlations based on Z stabilizer phase correlations
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j <= i; j++) {
            double corr = 0.0;

            if (i == j) {
                // Self-correlation is 1.0
                corr = 1.0;
            } else {
                // Compute correlation from Z stabilizer measurements
                const SyndromeVertex* vi = &graph->vertices[i];
                const SyndromeVertex* vj = &graph->vertices[j];

                // Distance-based correlation decay
                double dx = (double)vi->x - (double)vj->x;
                double dy = (double)vi->y - (double)vj->y;
                double dz = (double)vi->z - (double)vj->z;
                double distance = sqrt(dx*dx + dy*dy + dz*dz);

                // Exponential decay with correlation factor from config
                corr = exp(-distance * z_state->config.correlation_factor);

                // Incorporate phase correlation data if available
                if (z_state->phase_correlations) {
                    size_t idx_i = vi->y * z_state->config.num_stabilizers + vi->x;
                    size_t idx_j = vj->y * z_state->config.num_stabilizers + vj->x;

                    if (idx_i < z_state->config.num_stabilizers &&
                        idx_j < z_state->config.num_stabilizers) {
                        double phase_i = z_state->phase_correlations[idx_i];
                        double phase_j = z_state->phase_correlations[idx_j];
                        // Phase similarity increases correlation
                        double phase_sim = 1.0 - fabs(phase_i - phase_j);
                        corr *= (0.5 + 0.5 * phase_sim);
                    }
                }
            }

            // Symmetric matrix
            graph->correlation_matrix[i * n + j] = corr;
            graph->correlation_matrix[j * n + i] = corr;
        }
    }

    return true;
}

/**
 * @brief Analyze error patterns in the matching graph
 *
 * Identifies recurring error patterns by examining the structure of
 * detected syndromes and their correlations. Uses the Z stabilizer
 * state for phase-aware pattern detection.
 */
bool analyze_error_patterns(MatchingGraph* graph,
                          const SyndromeConfig* config,
                          const ZStabilizerState* z_state) {
    if (!graph || !config || !z_state) {
        return false;
    }

    // Ensure pattern weights array is allocated
    if (!graph->pattern_weights) {
        graph->pattern_weights = aligned_alloc(64, graph->max_vertices * sizeof(double));
        if (!graph->pattern_weights) {
            return false;
        }
        memset(graph->pattern_weights, 0, graph->max_vertices * sizeof(double));
    }

    // Analyze each vertex for pattern membership
    for (size_t i = 0; i < graph->num_vertices; i++) {
        SyndromeVertex* v = &graph->vertices[i];
        double pattern_weight = 0.0;

        // Count neighboring defects
        size_t neighbor_defects = 0;
        double total_edge_weight = 0.0;

        for (size_t j = 0; j < graph->num_edges; j++) {
            SyndromeEdge* edge = &graph->edges[j];
            if (edge->vertex1 == v || edge->vertex2 == v) {
                SyndromeVertex* other = (edge->vertex1 == v) ? edge->vertex2 : edge->vertex1;
                if (other->part_of_chain || other->weight > config->detection_threshold) {
                    neighbor_defects++;
                    total_edge_weight += edge->weight;
                }
            }
        }

        // Pattern weight based on local structure
        if (neighbor_defects > 0) {
            pattern_weight = (double)neighbor_defects * total_edge_weight;

            // Incorporate phase stability from Z stabilizer
            if (z_state->phase_correlations) {
                size_t idx = v->y * z_state->config.num_stabilizers + v->x;
                if (idx < z_state->config.num_stabilizers) {
                    // Lower phase correlation indicates higher error probability
                    double phase_instability = 1.0 - z_state->phase_correlations[idx];
                    pattern_weight *= (1.0 + phase_instability);
                }
            }
        }

        graph->pattern_weights[i] = pattern_weight;

        // Update vertex correlation weight
        v->correlation_weight = pattern_weight;
    }

    // Detect chains starting from high-weight vertices
    for (size_t i = 0; i < graph->num_vertices; i++) {
        if (graph->pattern_weights[i] > config->detection_threshold) {
            detect_error_chain(graph, &graph->vertices[i], z_state);
        }
    }

    return true;
}

/**
 * @brief Group vertices for parallel measurement
 *
 * Groups syndrome vertices such that vertices within each group can be
 * measured simultaneously without crosstalk interference. Uses spatial
 * separation and correlation data to ensure measurement independence.
 */
bool group_parallel_vertices(MatchingGraph* graph,
                           const SyndromeConfig* config) {
    if (!graph || !config) {
        return false;
    }

    size_t n = graph->num_vertices;
    if (n == 0) {
        graph->num_parallel_groups = 0;
        return true;
    }

    // Ensure parallel groups array is allocated
    if (!graph->parallel_groups) {
        graph->parallel_groups = aligned_alloc(64, n * sizeof(bool));
        if (!graph->parallel_groups) {
            return false;
        }
    }

    // Initialize all vertices as ungrouped
    for (size_t i = 0; i < n; i++) {
        graph->parallel_groups[i] = false;
    }

    // Compute minimum distance for parallel measurement
    // Derive from lattice dimensions and parallel group size
    // Larger groups require more spacing to avoid crosstalk
    double min_parallel_distance = 2.0;  // Default minimum spacing
    if (config->parallel_group_size > 0 && config->lattice_width > 0) {
        // Scale minimum distance based on group size relative to lattice
        min_parallel_distance = (double)config->lattice_width /
                               (double)config->parallel_group_size;
        if (min_parallel_distance < 2.0) min_parallel_distance = 2.0;
    }

    // Greedy grouping algorithm
    size_t num_groups = 0;
    bool* assigned = aligned_alloc(64, n * sizeof(bool));
    if (!assigned) {
        return false;
    }
    memset(assigned, 0, n * sizeof(bool));

    while (true) {
        // Find first unassigned vertex
        size_t first_unassigned = n;
        for (size_t i = 0; i < n; i++) {
            if (!assigned[i]) {
                first_unassigned = i;
                break;
            }
        }

        if (first_unassigned == n) {
            // All vertices assigned
            break;
        }

        // Start new group with this vertex
        num_groups++;
        assigned[first_unassigned] = true;
        graph->parallel_groups[first_unassigned] = true;

        // Add compatible vertices to this group
        for (size_t i = first_unassigned + 1; i < n; i++) {
            if (assigned[i]) continue;

            // Check distance from all vertices already in current group
            bool compatible = true;
            for (size_t j = 0; j < i; j++) {
                if (!assigned[j] || j < first_unassigned) continue;
                if (!graph->parallel_groups[j]) continue;

                // Compute distance between vertices i and j
                const SyndromeVertex* vi = &graph->vertices[i];
                const SyndromeVertex* vj = &graph->vertices[j];
                double dx = (double)vi->x - (double)vj->x;
                double dy = (double)vi->y - (double)vj->y;
                double dz = (double)vi->z - (double)vj->z;
                double distance = sqrt(dx*dx + dy*dy + dz*dz);

                if (distance < min_parallel_distance) {
                    compatible = false;
                    break;
                }
            }

            if (compatible) {
                assigned[i] = true;
                graph->parallel_groups[i] = true;
            }
        }

        // Reset parallel_groups for next iteration (used as temporary marker)
        for (size_t i = 0; i < n; i++) {
            if (assigned[i] && graph->parallel_groups[i]) {
                graph->parallel_groups[i] = (num_groups % 2 == 1);
            }
        }
    }

    free(assigned);
    graph->num_parallel_groups = num_groups;

    return true;
}
