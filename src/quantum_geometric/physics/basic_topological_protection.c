/**
 * @file basic_topological_protection.c
 * @brief Basic implementation of topological error protection
 *
 * Implements error detection and correction using topological stabilizer
 * measurements and anyon pairing/braiding operations.
 */

#include "quantum_geometric/physics/basic_topological_protection.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include <stdlib.h>
#include <math.h>

// Basic error thresholds
#define BASIC_ERROR_THRESHOLD 0.1
#define BASIC_STABILITY_THRESHOLD 0.9
#define MAX_CORRECTION_ATTEMPTS 3

// End marker for pair arrays
#define PAIR_END_MARKER ((size_t)-1)

// ============================================================================
// State Initialization
// ============================================================================

/**
 * Apply a single stabilizer projector (1 + S)/2 to the state.
 * This projects onto the +1 eigenspace of stabilizer S.
 * For X-type stabilizers: S = X_i1 ⊗ X_i2 ⊗ ... ⊗ X_ik
 * For Z-type stabilizers: S = Z_i1 ⊗ Z_i2 ⊗ ... ⊗ Z_ik
 */
static void apply_stabilizer_projector(quantum_state_t* state,
                                        StabilizerType type,
                                        const size_t* qubits,
                                        size_t num_qubits_in_stab) {
    if (!state || !state->coordinates || !qubits || num_qubits_in_stab == 0) return;

    size_t dim = state->dimension;

    // Allocate temporary storage for |Sψ⟩
    ComplexFloat* s_psi = calloc(dim, sizeof(ComplexFloat));
    if (!s_psi) return;

    if (type == VERTEX_STABILIZER) {
        // X-type stabilizer: X_i1 ⊗ X_i2 ⊗ ... ⊗ X_ik
        // Action: flips all qubits in the stabilizer support

        // Build the flip mask for all qubits in stabilizer
        size_t flip_mask = 0;
        for (size_t i = 0; i < num_qubits_in_stab; i++) {
            if (qubits[i] < state->num_qubits) {
                flip_mask |= (1UL << qubits[i]);
            }
        }

        // Compute S|ψ⟩ where S is the X-type stabilizer
        for (size_t i = 0; i < dim; i++) {
            size_t j = i ^ flip_mask;  // Apply X to all qubits in stabilizer
            s_psi[i] = state->coordinates[j];
        }
    } else {
        // Z-type stabilizer: Z_i1 ⊗ Z_i2 ⊗ ... ⊗ Z_ik
        // Action: multiplies by (-1)^(parity of qubits in support)

        for (size_t i = 0; i < dim; i++) {
            // Count parity of qubits in stabilizer support
            int parity = 0;
            for (size_t q = 0; q < num_qubits_in_stab; q++) {
                if (qubits[q] < state->num_qubits) {
                    parity ^= ((i >> qubits[q]) & 1);
                }
            }

            // S|i⟩ = (-1)^parity |i⟩
            if (parity) {
                s_psi[i].real = -state->coordinates[i].real;
                s_psi[i].imag = -state->coordinates[i].imag;
            } else {
                s_psi[i] = state->coordinates[i];
            }
        }
    }

    // Apply projector: |ψ'⟩ = (|ψ⟩ + S|ψ⟩) / 2
    // (Normalization happens after all projectors are applied)
    for (size_t i = 0; i < dim; i++) {
        state->coordinates[i].real = (state->coordinates[i].real + s_psi[i].real) / 2.0f;
        state->coordinates[i].imag = (state->coordinates[i].imag + s_psi[i].imag) / 2.0f;
    }

    free(s_psi);
}

void initialize_ground_state(quantum_state_t* state) {
    if (!state || !state->coordinates) return;

    // The ground state of a topological stabilizer code is the unique state
    // in the +1 eigenspace of ALL stabilizer operators:
    //
    // |ψ_0⟩ = P_G |ψ_init⟩ / ||P_G |ψ_init⟩||
    //
    // where P_G is the projector onto the ground space:
    // P_G = Π_S (1 + S)/2 for all stabilizers S
    //
    // We start with |00...0⟩ and successively project onto each stabilizer's
    // +1 eigenspace, then normalize.

    size_t dim = state->dimension;

    // Step 1: Initialize to |00...0⟩
    for (size_t i = 0; i < dim; i++) {
        state->coordinates[i].real = 0.0f;
        state->coordinates[i].imag = 0.0f;
    }
    state->coordinates[0].real = 1.0f;

    // Step 2: Apply projector for each plaquette (Z-type) stabilizer
    // Plaquette B_p = Π_{e∈p} Z_e acts on edges around plaquette p
    size_t plaq_qubits[4];
    for (size_t p = 0; p < state->num_plaquettes; p++) {
        get_plaquette_vertices(state, p, plaq_qubits);
        apply_stabilizer_projector(state, PLAQUETTE_STABILIZER, plaq_qubits, 4);
    }

    // Step 3: Apply projector for each vertex (X-type) stabilizer
    // Vertex A_v = Π_{e∈v} X_e acts on edges incident to vertex v
    size_t vert_qubits[4];
    for (size_t v = 0; v < state->num_vertices; v++) {
        get_vertex_edges(state, v, vert_qubits);
        // Count valid edges (not boundary)
        size_t num_valid = 0;
        size_t valid_qubits[4];
        for (int i = 0; i < 4; i++) {
            if (vert_qubits[i] != (size_t)-1 && vert_qubits[i] < state->num_qubits) {
                valid_qubits[num_valid++] = vert_qubits[i];
            }
        }
        if (num_valid > 0) {
            apply_stabilizer_projector(state, VERTEX_STABILIZER, valid_qubits, num_valid);
        }
    }

    // Step 4: Normalize the resulting state
    double norm_sq = 0.0;
    for (size_t i = 0; i < dim; i++) {
        norm_sq += state->coordinates[i].real * state->coordinates[i].real +
                   state->coordinates[i].imag * state->coordinates[i].imag;
    }

    if (norm_sq > 1e-15) {
        float inv_norm = 1.0f / sqrtf((float)norm_sq);
        for (size_t i = 0; i < dim; i++) {
            state->coordinates[i].real *= inv_norm;
            state->coordinates[i].imag *= inv_norm;
        }
        state->is_normalized = true;
    } else {
        // Projection failed - fall back to equal superposition of valid configurations
        // This can happen if |00...0⟩ has zero overlap with ground space
        // Try starting from |++...+⟩ instead
        float amp = 1.0f / sqrtf((float)dim);
        for (size_t i = 0; i < dim; i++) {
            state->coordinates[i].real = amp;
            state->coordinates[i].imag = 0.0f;
        }

        // Re-apply all projectors
        for (size_t p = 0; p < state->num_plaquettes; p++) {
            get_plaquette_vertices(state, p, plaq_qubits);
            apply_stabilizer_projector(state, PLAQUETTE_STABILIZER, plaq_qubits, 4);
        }
        for (size_t v = 0; v < state->num_vertices; v++) {
            get_vertex_edges(state, v, vert_qubits);
            size_t num_valid = 0;
            size_t valid_qubits[4];
            for (int i = 0; i < 4; i++) {
                if (vert_qubits[i] != (size_t)-1 && vert_qubits[i] < state->num_qubits) {
                    valid_qubits[num_valid++] = vert_qubits[i];
                }
            }
            if (num_valid > 0) {
                apply_stabilizer_projector(state, VERTEX_STABILIZER, valid_qubits, num_valid);
            }
        }

        // Normalize
        norm_sq = 0.0;
        for (size_t i = 0; i < dim; i++) {
            norm_sq += state->coordinates[i].real * state->coordinates[i].real +
                       state->coordinates[i].imag * state->coordinates[i].imag;
        }
        if (norm_sq > 1e-15) {
            float inv_norm = 1.0f / sqrtf((float)norm_sq);
            for (size_t i = 0; i < dim; i++) {
                state->coordinates[i].real *= inv_norm;
                state->coordinates[i].imag *= inv_norm;
            }
        }
        state->is_normalized = true;
    }

    // Clear any anyon states - ground state has no anyons
    state->num_anyons = 0;
}

void apply_pauli_x(quantum_state_t* state, size_t qubit) {
    if (!state || !state->coordinates || qubit >= state->num_qubits) return;

    // Pauli-X gate (bit-flip): |0⟩ ↔ |1⟩
    // In computational basis: X|ψ⟩ = Σ_i α_i |i ⊕ 2^qubit⟩
    //
    // This swaps amplitudes of basis states that differ only in the specified qubit.
    // For qubit k: amplitude at index i is swapped with amplitude at index i XOR 2^k

    size_t dim = state->dimension;
    size_t mask = 1UL << qubit;

    for (size_t i = 0; i < dim; i++) {
        size_t j = i ^ mask;
        if (j > i) {  // Only swap once per pair
            ComplexFloat temp = state->coordinates[i];
            state->coordinates[i] = state->coordinates[j];
            state->coordinates[j] = temp;
        }
    }
}

// ============================================================================
// Error Detection
// ============================================================================

ErrorCode detect_basic_errors(quantum_state_t* state) {
    if (!state) return ERROR_INVALID_STATE;

    // Measure basic error syndromes
    double syndrome_weight = 0.0;

    // Check plaquette operators (Z stabilizers)
    for (size_t i = 0; i < state->num_plaquettes; i++) {
        double plaq = measure_plaquette_operator(state, i);
        syndrome_weight += (1.0 - plaq) / 2.0;
    }

    // Check vertex operators (X stabilizers)
    for (size_t i = 0; i < state->num_vertices; i++) {
        double vert = measure_vertex_operator(state, i);
        syndrome_weight += (1.0 - vert) / 2.0;
    }

    // Basic threshold check
    if (syndrome_weight > BASIC_ERROR_THRESHOLD) {
        return ERROR_DETECTED;
    }

    return NO_ERROR;
}

// ============================================================================
// Anyon Pairing
// ============================================================================

static AnyonPair* find_nearest_pairs(quantum_state_t* state) {
    if (!state || state->num_anyons == 0) {
        // Return empty pair list with end marker
        AnyonPair* pairs = malloc(sizeof(AnyonPair));
        if (pairs) {
            pairs[0].anyon1 = PAIR_END_MARKER;
        }
        return pairs;
    }

    // Allocate maximum possible pairs + end marker
    AnyonPair* pairs = malloc((state->num_anyons / 2 + 1) * sizeof(AnyonPair));
    if (!pairs) return NULL;

    size_t num_pairs = 0;

    // Simple greedy nearest-neighbor matching
    for (size_t i = 0; i < state->num_anyons; i++) {
        if (state->anyons[i].paired) continue;

        // Find closest unpaired anyon
        double min_dist = INFINITY;
        size_t min_idx = PAIR_END_MARKER;

        for (size_t j = i + 1; j < state->num_anyons; j++) {
            if (state->anyons[j].paired) continue;

            double dist = calculate_anyon_distance(
                state->anyons[i].position,
                state->anyons[j].position
            );

            if (dist < min_dist) {
                min_dist = dist;
                min_idx = j;
            }
        }

        // Create pair if found
        if (min_idx != PAIR_END_MARKER) {
            pairs[num_pairs].anyon1 = i;
            pairs[num_pairs].anyon2 = min_idx;
            pairs[num_pairs].distance = min_dist;
            num_pairs++;

            state->anyons[i].paired = true;
            state->anyons[min_idx].paired = true;
        }
    }

    // End marker
    pairs[num_pairs].anyon1 = PAIR_END_MARKER;
    return pairs;
}

static void reset_pairing(quantum_state_t* state) {
    if (!state || !state->anyons) return;

    for (size_t i = 0; i < state->num_anyons; i++) {
        state->anyons[i].paired = false;
    }
}

// ============================================================================
// Braiding Correction
// ============================================================================

static void apply_basic_braiding(quantum_state_t* state, AnyonPair pair) {
    if (!state || !state->anyons) return;
    if (pair.anyon1 >= state->num_anyons || pair.anyon2 >= state->num_anyons) return;

    // Calculate shortest path between paired anyons
    Path* path = find_shortest_path(
        state->anyons[pair.anyon1].position,
        state->anyons[pair.anyon2].position
    );

    if (!path) return;

    // Apply correction operations along the path
    for (size_t i = 0; i < path->length; i++) {
        apply_correction_operator(state, path->vertices[i]);
    }

    free_path(path);
}

// ============================================================================
// Error Correction
// ============================================================================

void correct_basic_errors(quantum_state_t* state) {
    if (!state) return;

    int attempts = 0;
    ErrorCode error;

    do {
        // Reset pairing state
        reset_pairing(state);

        // Find anyon pairs using minimum weight matching
        AnyonPair* pairs = find_nearest_pairs(state);
        if (!pairs) return;

        // Apply braiding corrections for each pair
        for (size_t i = 0; pairs[i].anyon1 != PAIR_END_MARKER; i++) {
            apply_basic_braiding(state, pairs[i]);
        }

        free(pairs);

        // Check if errors remain
        error = detect_basic_errors(state);
        attempts++;

    } while (error == ERROR_DETECTED && attempts < MAX_CORRECTION_ATTEMPTS);
}

// ============================================================================
// State Verification
// ============================================================================

bool verify_basic_state(quantum_state_t* state) {
    if (!state || state->num_stabilizers == 0) return false;

    double stability = 0.0;

    // Check all stabilizer operators
    for (size_t i = 0; i < state->num_stabilizers; i++) {
        stability += measure_stabilizer_operator(state, i);
    }

    stability /= (double)state->num_stabilizers;

    return stability > BASIC_STABILITY_THRESHOLD;
}

// ============================================================================
// Full Protection Cycle
// ============================================================================

void protect_basic_state(quantum_state_t* state) {
    if (!state) return;

    // Detect errors
    ErrorCode error = detect_basic_errors(state);

    // Apply correction if needed
    if (error == ERROR_DETECTED) {
        correct_basic_errors(state);
    }

    // Verify correction succeeded
    if (!verify_basic_state(state)) {
        log_protection_failure(state);
    }
}
