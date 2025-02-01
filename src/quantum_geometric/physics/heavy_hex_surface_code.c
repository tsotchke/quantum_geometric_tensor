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
