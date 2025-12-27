/**
 * @file basic_topological_operations.c
 * @brief Implementation of basic topological operations
 *
 * Provides fundamental operations for topological quantum error correction:
 * - Pauli measurements (X, Z)
 * - Stabilizer measurements (plaquette and vertex operators)
 * - Lattice structure operations
 * - Path finding for anyon correction
 */

#include "quantum_geometric/physics/basic_topological_protection.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/core/quantum_complex.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Forward declarations for static helper functions
static double measure_average_stabilizer(quantum_state_t* state);
static double calculate_syndrome_weight(quantum_state_t* state);
static size_t get_qubit_at_position(quantum_state_t* state, Position pos);
static void apply_pauli_x_gate(quantum_state_t* state, size_t qubit);

// ============================================================================
// Pauli Measurement Operations (renamed to avoid conflicts with surface code versions)
// ============================================================================

// Renamed to avoid conflict with parallel_stabilizer.c and heavy_hex_surface_code.c
double measure_pauli_z_basic(quantum_state_t* state, size_t qubit) {
    if (!state || !state->coordinates || qubit >= state->num_qubits) {
        return 0.0;
    }

    // Compute expectation value of Z on specified qubit
    // <Z> = sum over i of |a_i|^2 * (-1)^bit_qubit(i)
    size_t dim = state->dimension;
    double expectation = 0.0;

    for (size_t i = 0; i < dim; i++) {
        double prob = complex_float_abs_squared(state->coordinates[i]);
        // Check if qubit is |1⟩ in basis state i
        int bit = (i >> qubit) & 1;
        expectation += prob * (bit ? -1.0 : 1.0);
    }

    return expectation;
}

// Renamed to avoid conflict with parallel_stabilizer.c and heavy_hex_surface_code.c
double measure_pauli_x_basic(quantum_state_t* state, size_t qubit) {
    if (!state || !state->coordinates || qubit >= state->num_qubits) {
        return 0.0;
    }

    // Compute expectation value of X on specified qubit
    // <X> = 2 * sum_{i<j where i,j differ only in qubit} Re(a_i* a_j)
    size_t dim = state->dimension;
    double expectation = 0.0;
    size_t mask = 1UL << qubit;

    for (size_t i = 0; i < dim; i++) {
        size_t j = i ^ mask;  // State with qubit flipped
        if (j > i) {
            ComplexFloat ai = state->coordinates[i];
            ComplexFloat aj = state->coordinates[j];
            ComplexFloat conj_ai = complex_float_conjugate(ai);
            ComplexFloat product = complex_float_multiply(conj_ai, aj);
            expectation += 2.0 * product.real;
        }
    }

    return expectation;
}

// ============================================================================
// Lattice Structure Operations
// ============================================================================

void get_plaquette_vertices(quantum_state_t* state, size_t plaquette_index, size_t* vertices) {
    if (!state || !vertices) return;

    // For a square lattice with (width-1) x (height-1) plaquettes
    size_t width = state->lattice_width > 0 ? state->lattice_width : 1;
    size_t plaq_width = width > 1 ? width - 1 : 1;

    size_t row = plaquette_index / plaq_width;
    size_t col = plaquette_index % plaq_width;

    // Vertices at corners of the plaquette (clockwise from top-left)
    vertices[0] = row * width + col;           // Top-left
    vertices[1] = row * width + col + 1;       // Top-right
    vertices[2] = (row + 1) * width + col + 1; // Bottom-right
    vertices[3] = (row + 1) * width + col;     // Bottom-left
}

void get_vertex_edges(quantum_state_t* state, size_t vertex_index, size_t* edges) {
    if (!state || !edges) return;

    // Initialize with invalid indices
    edges[0] = edges[1] = edges[2] = edges[3] = (size_t)-1;

    size_t width = state->lattice_width > 0 ? state->lattice_width : 1;
    size_t height = state->lattice_height > 0 ? state->lattice_height : 1;

    size_t row = vertex_index / width;
    size_t col = vertex_index % width;

    // Horizontal edges: (width-1) per row, numbered first
    // Vertical edges: width per row, numbered after horizontals
    size_t h_edges_per_row = width > 1 ? width - 1 : 0;
    size_t v_edges_per_row = width;
    size_t total_h_edges = h_edges_per_row * height;

    size_t edge_idx = 0;

    // Left horizontal edge
    if (col > 0) {
        edges[edge_idx++] = row * h_edges_per_row + (col - 1);
    }
    // Right horizontal edge
    if (col < width - 1) {
        edges[edge_idx++] = row * h_edges_per_row + col;
    }
    // Top vertical edge
    if (row > 0) {
        edges[edge_idx++] = total_h_edges + (row - 1) * v_edges_per_row + col;
    }
    // Bottom vertical edge
    if (row < height - 1) {
        edges[edge_idx++] = total_h_edges + row * v_edges_per_row + col;
    }
}

StabilizerType get_stabilizer_type(quantum_state_t* state, size_t index) {
    if (!state) return PLAQUETTE_STABILIZER;

    // First num_plaquettes stabilizers are plaquette (Z) type
    // Remaining num_vertices stabilizers are vertex (X) type
    if (index < state->num_plaquettes) {
        return PLAQUETTE_STABILIZER;
    }
    return VERTEX_STABILIZER;
}

size_t get_stabilizer_location(quantum_state_t* state, size_t index) {
    if (!state) return 0;

    if (index < state->num_plaquettes) {
        return index;  // Plaquette index directly
    }
    return index - state->num_plaquettes;  // Offset to get vertex index
}

// ============================================================================
// Stabilizer Measurement Operations
// ============================================================================

double measure_plaquette_operator(quantum_state_t* state, size_t index) {
    if (!state || index >= state->num_plaquettes) return 0.0;

    // Get the 4 vertices of this plaquette
    size_t vertices[4];
    get_plaquette_vertices(state, index, vertices);

    // Plaquette operator is product of Z on all vertices
    // In stabilizer formalism: B_p = prod_{i in p} Z_i
    double result = 1.0;
    for (int i = 0; i < 4; i++) {
        if (vertices[i] < state->num_qubits) {
            result *= measure_pauli_z(state, vertices[i]);
        }
    }

    return result;
}

double measure_vertex_operator(quantum_state_t* state, size_t index) {
    if (!state || index >= state->num_vertices) return 0.0;

    // Get the edges adjacent to this vertex
    size_t edges[4];
    get_vertex_edges(state, index, edges);

    // Vertex operator is product of X on all adjacent edges
    // In stabilizer formalism: A_v = prod_{e at v} X_e
    double result = 1.0;
    for (int i = 0; i < 4; i++) {
        if (edges[i] != (size_t)-1 && edges[i] < state->num_qubits) {
            result *= measure_pauli_x(state, edges[i]);
        }
    }

    return result;
}

double measure_stabilizer_operator(quantum_state_t* state, size_t index) {
    if (!state || index >= state->num_stabilizers) return 0.0;

    StabilizerType type = get_stabilizer_type(state, index);
    size_t location = get_stabilizer_location(state, index);

    if (type == PLAQUETTE_STABILIZER) {
        return measure_plaquette_operator(state, location);
    } else {
        return measure_vertex_operator(state, location);
    }
}

// ============================================================================
// Path Finding and Distance Operations
// ============================================================================

double calculate_anyon_distance(Position pos1, Position pos2) {
    double dx = pos1.x - pos2.x;
    double dy = pos1.y - pos2.y;
    return sqrt(dx * dx + dy * dy);
}

Path* find_shortest_path(Position start, Position end) {
    Path* path = malloc(sizeof(Path));
    if (!path) return NULL;

    // Manhattan distance path (L-shaped)
    int dx = (int)fabs(end.x - start.x);
    int dy = (int)fabs(end.y - start.y);
    path->length = (size_t)(dx + dy);
    path->total_distance = (double)(dx + dy);

    if (path->length == 0) {
        path->vertices = NULL;
        return path;
    }

    path->vertices = malloc(path->length * sizeof(Position));
    if (!path->vertices) {
        free(path);
        return NULL;
    }

    // Build path: move in x direction first, then y
    Position current = start;
    size_t idx = 0;

    // Move along x axis
    double x_step = (end.x > start.x) ? 1.0 : -1.0;
    while (fabs(current.x - end.x) > 0.5 && idx < path->length) {
        current.x += x_step;
        path->vertices[idx++] = current;
    }

    // Move along y axis
    double y_step = (end.y > start.y) ? 1.0 : -1.0;
    while (fabs(current.y - end.y) > 0.5 && idx < path->length) {
        current.y += y_step;
        path->vertices[idx++] = current;
    }

    return path;
}

void free_path(Path* path) {
    if (path) {
        free(path->vertices);
        free(path);
    }
}

// ============================================================================
// Correction Operations
// ============================================================================

static size_t get_qubit_at_position(quantum_state_t* state, Position pos) {
    if (!state) return 0;

    // Convert 2D position to qubit index
    size_t col = (size_t)pos.x;
    size_t row = (size_t)pos.y;
    size_t width = state->lattice_width > 0 ? state->lattice_width : 1;

    return row * width + col;
}

static void apply_pauli_x_gate(quantum_state_t* state, size_t qubit) {
    if (!state || !state->coordinates || qubit >= state->num_qubits) return;

    // Apply X gate: swap amplitudes where qubit differs
    size_t dim = state->dimension;
    size_t mask = 1UL << qubit;

    for (size_t i = 0; i < dim; i++) {
        size_t j = i ^ mask;
        if (j > i) {
            // Swap amplitudes
            ComplexFloat temp = state->coordinates[i];
            state->coordinates[i] = state->coordinates[j];
            state->coordinates[j] = temp;
        }
    }
}

// Renamed to avoid conflict with error_syndrome.c (this uses Position struct)
void apply_correction_operator_at_position(quantum_state_t* state, Position pos) {
    if (!state) return;

    size_t qubit = get_qubit_at_position(state, pos);
    if (qubit < state->num_qubits) {
        apply_pauli_x_gate(state, qubit);
    }
}

// ============================================================================
// Logging and Diagnostics
// ============================================================================

static double measure_average_stabilizer(quantum_state_t* state) {
    if (!state || state->num_stabilizers == 0) return 0.0;

    double sum = 0.0;
    for (size_t i = 0; i < state->num_stabilizers; i++) {
        sum += measure_stabilizer_operator(state, i);
    }

    return sum / (double)state->num_stabilizers;
}

static double calculate_syndrome_weight(quantum_state_t* state) {
    if (!state) return 0.0;

    double weight = 0.0;

    // Weight from plaquette violations
    for (size_t i = 0; i < state->num_plaquettes; i++) {
        double plaq = measure_plaquette_operator(state, i);
        weight += (1.0 - plaq) / 2.0;  // Maps +1 to 0, -1 to 1
    }

    // Weight from vertex violations
    for (size_t i = 0; i < state->num_vertices; i++) {
        double vert = measure_vertex_operator(state, i);
        weight += (1.0 - vert) / 2.0;
    }

    return weight;
}

void log_protection_failure(quantum_state_t* state) {
    if (!state) return;

    fprintf(stderr, "Topological protection failure detected:\n");
    fprintf(stderr, "  Lattice size: %zu x %zu\n",
            state->lattice_width, state->lattice_height);
    fprintf(stderr, "  Number of anyons: %zu\n", state->num_anyons);
    fprintf(stderr, "  Number of stabilizers: %zu (plaq: %zu, vert: %zu)\n",
            state->num_stabilizers, state->num_plaquettes, state->num_vertices);
    fprintf(stderr, "  Average stabilizer value: %.4f\n",
            measure_average_stabilizer(state));
    fprintf(stderr, "  Error syndrome weight: %.4f\n",
            calculate_syndrome_weight(state));
}
