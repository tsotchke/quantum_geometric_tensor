/**
 * @file heavy_hex_surface_code.c
 * @brief Implementation of heavy-hex surface code for quantum error correction
 */

#include "quantum_geometric/physics/heavy_hex_surface_code.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/physics/quantum_state_operations.h"
#include "quantum_geometric/physics/error_weight.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Forward declarations
static bool initialize_hex_lattice(HexLattice* lattice,
                                 const HexConfig* config);
static void cleanup_hex_lattice(HexLattice* lattice);
static bool measure_hex_stabilizers(HexLattice* lattice,
                                  quantum_state* state);
static bool apply_hex_correction(HexLattice* lattice,
                               quantum_state* state);

bool init_heavy_hex_code(HexState* state, const HexConfig* config) {
    if (!state || !config || !validate_hex_parameters(config)) {
        return false;
    }

    // Allocate and initialize hex lattice
    state->lattice = malloc(sizeof(HexLattice));
    if (!state->lattice) {
        return false;
    }

    if (!initialize_hex_lattice(state->lattice, config)) {
        free(state->lattice);
        return false;
    }

    // Initialize error weight calculation
    WeightConfig weight_config = {
        .lattice_width = config->lattice_width,
        .lattice_height = config->lattice_height,
        .lattice_depth = 1,  // 2D lattice
        .base_error_rate = config->base_error_rate,
        .probability_factor = 1.0,
        .geometric_factor = 1.0,
        .size_factor = 0.5,
        .use_geometric_scaling = true,
        .normalize_weights = true
    };

    if (!init_error_weight(&state->weights, &weight_config)) {
        cleanup_hex_lattice(state->lattice);
        free(state->lattice);
        return false;
    }

    // Copy configuration
    memcpy(&state->config, config, sizeof(HexConfig));
    state->measurement_count = 0;
    state->error_rate = 0.0;
    state->last_syndrome = NULL;

    return true;
}

void cleanup_heavy_hex_code(HexState* state) {
    if (state) {
        cleanup_hex_lattice(state->lattice);
        free(state->lattice);
        cleanup_error_weight(&state->weights);
        free(state->last_syndrome);
        memset(state, 0, sizeof(HexState));
    }
}

bool measure_hex_code(HexState* state, quantum_state* qstate) {
    if (!state || !qstate || !state->lattice) {
        return false;
    }

    // Measure stabilizers
    if (!measure_hex_stabilizers(state->lattice, qstate)) {
        return false;
    }

    // Calculate error weights
    if (!calculate_error_weights(&state->weights, qstate)) {
        return false;
    }

    // Update error rate
    WeightStatistics stats;
    if (!get_weight_statistics(&state->weights, &stats)) {
        return false;
    }
    state->error_rate = stats.max_weight;

    // Store syndrome if error rate exceeds threshold
    if (state->error_rate > state->config.error_threshold) {
        size_t syndrome_size = state->lattice->num_stabilizers * sizeof(double);
        double* new_syndrome = realloc(state->last_syndrome, syndrome_size);
        if (new_syndrome) {
            state->last_syndrome = new_syndrome;
            memcpy(state->last_syndrome,
                   state->lattice->stabilizer_values,
                   syndrome_size);
        }
    }

    // Apply error correction if needed
    if (state->config.auto_correction &&
        state->error_rate > state->config.error_threshold) {
        if (!apply_hex_correction(state->lattice, qstate)) {
            return false;
        }
    }

    state->measurement_count++;
    return true;
}

const double* get_hex_syndrome(const HexState* state, size_t* size) {
    if (!state || !size) {
        return NULL;
    }
    *size = state->lattice->num_stabilizers;
    return state->last_syndrome;
}

double get_hex_error_rate(const HexState* state) {
    return state ? state->error_rate : 0.0;
}

bool validate_hex_parameters(const HexConfig* config) {
    if (!config) {
        return false;
    }

    // Check lattice dimensions
    if (config->lattice_width < 3 || config->lattice_height < 3 ||
        config->lattice_width % 2 == 0 || config->lattice_height % 2 == 0) {
        return false;
    }

    // Check error parameters
    if (config->base_error_rate < 0.0 || config->base_error_rate > 1.0 ||
        config->error_threshold < 0.0 || config->error_threshold > 1.0) {
        return false;
    }

    return true;
}

// Helper function implementations
static bool initialize_hex_lattice(HexLattice* lattice,
                                 const HexConfig* config) {
    if (!lattice || !config) {
        return false;
    }

    // Calculate number of stabilizers in heavy-hex layout
    size_t width = config->lattice_width;
    size_t height = config->lattice_height;
    size_t num_stabilizers = ((width - 1) * (height - 1)) / 2;  // Hex layout

    // Allocate stabilizer arrays
    lattice->stabilizer_values = calloc(num_stabilizers, sizeof(double));
    lattice->stabilizer_coordinates = malloc(num_stabilizers * sizeof(HexCoordinate));
    
    if (!lattice->stabilizer_values || !lattice->stabilizer_coordinates) {
        free(lattice->stabilizer_values);
        free(lattice->stabilizer_coordinates);
        return false;
    }

    // Initialize stabilizer coordinates in heavy-hex pattern
    size_t idx = 0;
    for (size_t y = 1; y < height - 1; y += 2) {
        for (size_t x = 1; x < width - 1; x += 2) {
            lattice->stabilizer_coordinates[idx].x = x;
            lattice->stabilizer_coordinates[idx].y = y;
            lattice->stabilizer_coordinates[idx].type = 
                ((x + y) % 4 == 0) ? HEX_PLAQUETTE : HEX_VERTEX;
            idx++;
        }
    }

    lattice->num_stabilizers = num_stabilizers;
    lattice->width = width;
    lattice->height = height;

    return true;
}

static void cleanup_hex_lattice(HexLattice* lattice) {
    if (lattice) {
        free(lattice->stabilizer_values);
        free(lattice->stabilizer_coordinates);
        memset(lattice, 0, sizeof(HexLattice));
    }
}

static bool measure_hex_stabilizers(HexLattice* lattice,
                                  quantum_state* state) {
    if (!lattice || !state) {
        return false;
    }

    // Measure each stabilizer
    for (size_t i = 0; i < lattice->num_stabilizers; i++) {
        HexCoordinate* coord = &lattice->stabilizer_coordinates[i];
        
        if (coord->type == HEX_PLAQUETTE) {
            // Measure Z-type stabilizer (plaquette)
            double z1, z2, z3, z4, z5, z6;
            if (!measure_pauli_z(state, coord->x - 1, coord->y, &z1) ||
                !measure_pauli_z(state, coord->x + 1, coord->y, &z2) ||
                !measure_pauli_z(state, coord->x, coord->y - 1, &z3) ||
                !measure_pauli_z(state, coord->x, coord->y + 1, &z4) ||
                !measure_pauli_z(state, coord->x - 1, coord->y + 1, &z5) ||
                !measure_pauli_z(state, coord->x + 1, coord->y - 1, &z6)) {
                return false;
            }
            lattice->stabilizer_values[i] = z1 * z2 * z3 * z4 * z5 * z6;
        } else {
            // Measure X-type stabilizer (vertex)
            double x1, x2, x3, x4, x5, x6;
            if (!measure_pauli_x(state, coord->x - 1, coord->y, &x1) ||
                !measure_pauli_x(state, coord->x + 1, coord->y, &x2) ||
                !measure_pauli_x(state, coord->x, coord->y - 1, &x3) ||
                !measure_pauli_x(state, coord->x, coord->y + 1, &x4) ||
                !measure_pauli_x(state, coord->x - 1, coord->y - 1, &x5) ||
                !measure_pauli_x(state, coord->x + 1, coord->y + 1, &x6)) {
                return false;
            }
            lattice->stabilizer_values[i] = x1 * x2 * x3 * x4 * x5 * x6;
        }
    }

    return true;
}

static bool apply_hex_correction(HexLattice* lattice,
                               quantum_state* state) {
    if (!lattice || !state) {
        return false;
    }

    // Apply corrections based on stabilizer measurements
    for (size_t i = 0; i < lattice->num_stabilizers; i++) {
        if (fabs(lattice->stabilizer_values[i] + 1.0) < 1e-6) {
            // Negative measurement indicates error
            HexCoordinate* coord = &lattice->stabilizer_coordinates[i];
            
            if (coord->type == HEX_PLAQUETTE) {
                // Apply X corrections for Z-type stabilizer
                if (!apply_pauli_x(state, coord->x - 1, coord->y) ||
                    !apply_pauli_x(state, coord->x + 1, coord->y) ||
                    !apply_pauli_x(state, coord->x, coord->y - 1) ||
                    !apply_pauli_x(state, coord->x, coord->y + 1) ||
                    !apply_pauli_x(state, coord->x - 1, coord->y + 1) ||
                    !apply_pauli_x(state, coord->x + 1, coord->y - 1)) {
                    return false;
                }
            } else {
                // Apply Z corrections for X-type stabilizer
                if (!apply_pauli_z(state, coord->x - 1, coord->y) ||
                    !apply_pauli_z(state, coord->x + 1, coord->y) ||
                    !apply_pauli_z(state, coord->x, coord->y - 1) ||
                    !apply_pauli_z(state, coord->x, coord->y + 1) ||
                    !apply_pauli_z(state, coord->x - 1, coord->y - 1) ||
                    !apply_pauli_z(state, coord->x + 1, coord->y + 1)) {
                    return false;
                }
            }
        }
    }

    return true;
}

// ============================================================================
// Helper: Convert 2D lattice coordinates to qubit index
// ============================================================================

static inline size_t coords_to_qubit(const quantum_state* state, size_t x, size_t y) {
    if (!state) return SIZE_MAX;
    size_t width = state->lattice_width > 0 ? state->lattice_width : 16;
    return y * width + x;
}

// ============================================================================
// Public API: Pauli measurements (from error_weight.h)
// These compute expectation values <ψ|P|ψ> for Pauli operators P
// ============================================================================

bool measure_pauli_x(const quantum_state* state, size_t x, size_t y, double* result) {
    if (!state || !state->coordinates || !result) return false;

    size_t qubit_idx = coords_to_qubit(state, x, y);
    if (qubit_idx >= state->num_qubits) {
        *result = 1.0;  // Out of bounds treated as no error
        return true;
    }

    // Compute expectation value <X_q> = 2 * Σ_{i<j} Re(ψ*_i · ψ_j)
    // where j = i ⊕ (1 << q)
    double expectation = 0.0;
    size_t mask = (size_t)1 << qubit_idx;

    for (size_t i = 0; i < state->dimension; i++) {
        size_t j = i ^ mask;
        if (i < j) {  // Avoid double counting
            const ComplexFloat* amp_i = &state->coordinates[i];
            const ComplexFloat* amp_j = &state->coordinates[j];
            // X swaps |0⟩ ↔ |1⟩: <X> = 2*Re(ψ*_i · ψ_j)
            expectation += 2.0 * (amp_i->real * amp_j->real + amp_i->imag * amp_j->imag);
        }
    }

    *result = expectation;
    return true;
}

bool measure_pauli_y(const quantum_state* state, size_t x, size_t y, double* result) {
    if (!state || !state->coordinates || !result) return false;

    size_t qubit_idx = coords_to_qubit(state, x, y);
    if (qubit_idx >= state->num_qubits) {
        *result = 1.0;
        return true;
    }

    // Compute expectation value <Y_q> = 2 * Σ_{i<j} Im(ψ*_i · ψ_j) * sign
    // Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩
    double expectation = 0.0;
    size_t mask = (size_t)1 << qubit_idx;

    for (size_t i = 0; i < state->dimension; i++) {
        size_t j = i ^ mask;
        if (i < j) {
            const ComplexFloat* amp_i = &state->coordinates[i];
            const ComplexFloat* amp_j = &state->coordinates[j];
            // <Y> contribution: 2 * Im(ψ*_i · ψ_j) with appropriate sign
            // i has bit q=0, j has bit q=1
            // <i|Y|0> · <1|j> + <i|Y|1> · <0|j> = i·ψ_j + (-i)·ψ_i (from j side)
            double cross = amp_i->real * amp_j->imag - amp_i->imag * amp_j->real;
            expectation += 2.0 * cross;
        }
    }

    *result = expectation;
    return true;
}

bool measure_pauli_z(const quantum_state* state, size_t x, size_t y, double* result) {
    if (!state || !state->coordinates || !result) return false;

    size_t qubit_idx = coords_to_qubit(state, x, y);
    if (qubit_idx >= state->num_qubits) {
        *result = 1.0;
        return true;
    }

    // Compute expectation value <Z_q> = Σ_i |ψ_i|² · (-1)^{bit_q(i)}
    // Z|0⟩ = |0⟩, Z|1⟩ = -|1⟩
    double expectation = 0.0;
    size_t mask = (size_t)1 << qubit_idx;

    for (size_t i = 0; i < state->dimension; i++) {
        const ComplexFloat* amp = &state->coordinates[i];
        double prob = amp->real * amp->real + amp->imag * amp->imag;
        // +1 for |0⟩ basis, -1 for |1⟩ basis
        expectation += (i & mask) ? -prob : prob;
    }

    *result = expectation;
    return true;
}

// ============================================================================
// Public API: Pauli gate applications (from heavy_hex_surface_code.h)
// ============================================================================

bool apply_pauli_x(quantum_state* state, size_t x, size_t y) {
    if (!state || !state->coordinates) return false;

    size_t qubit_idx = coords_to_qubit(state, (int)x, (int)y);
    if (qubit_idx >= state->num_qubits) {
        return true;  // Out of bounds, no-op
    }

    // Pauli X (bit flip): swap amplitudes where qubit q differs
    // X|0⟩ = |1⟩, X|1⟩ = |0⟩
    size_t mask = (size_t)1 << qubit_idx;

    for (size_t i = 0; i < state->dimension; i++) {
        size_t j = i ^ mask;
        if (i < j) {
            // Swap amplitudes[i] and amplitudes[j]
            ComplexFloat temp = state->coordinates[i];
            state->coordinates[i] = state->coordinates[j];
            state->coordinates[j] = temp;
        }
    }

    return true;
}

bool apply_pauli_y(quantum_state* state, size_t x, size_t y) {
    if (!state || !state->coordinates) return false;

    size_t qubit_idx = coords_to_qubit(state, (int)x, (int)y);
    if (qubit_idx >= state->num_qubits) {
        return true;
    }

    // Pauli Y: Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩
    // Y = iXZ, so for each pair (i, j=i^mask):
    //   if bit_q(i) = 0: new_j = i * old_i, new_i = -i * old_j
    //   (where i here means √-1)
    size_t mask = (size_t)1 << qubit_idx;

    for (size_t i = 0; i < state->dimension; i++) {
        size_t j = i ^ mask;
        if (i < j) {
            // i has bit q = 0, j has bit q = 1
            ComplexFloat amp_i = state->coordinates[i];
            ComplexFloat amp_j = state->coordinates[j];

            // Y|0⟩ = i|1⟩: new_j = i * amp_i = (-imag, real)
            state->coordinates[j].real = -amp_i.imag;
            state->coordinates[j].imag = amp_i.real;

            // Y|1⟩ = -i|0⟩: new_i = -i * amp_j = (imag, -real)
            state->coordinates[i].real = amp_j.imag;
            state->coordinates[i].imag = -amp_j.real;
        }
    }

    return true;
}

bool apply_pauli_z(quantum_state* state, size_t x, size_t y) {
    if (!state || !state->coordinates) return false;

    size_t qubit_idx = coords_to_qubit(state, (int)x, (int)y);
    if (qubit_idx >= state->num_qubits) {
        return true;
    }

    // Pauli Z (phase flip): Z|0⟩ = |0⟩, Z|1⟩ = -|1⟩
    // Apply -1 phase to all basis states where qubit q is |1⟩
    size_t mask = (size_t)1 << qubit_idx;

    for (size_t i = 0; i < state->dimension; i++) {
        if (i & mask) {
            state->coordinates[i].real = -state->coordinates[i].real;
            state->coordinates[i].imag = -state->coordinates[i].imag;
        }
    }

    return true;
}
