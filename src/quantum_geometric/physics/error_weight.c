/**
 * @file error_weight.c
 * @brief Implementation of quantum error weight calculation system
 */

#include "quantum_geometric/physics/error_weight.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/physics/quantum_state_operations.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Forward declarations
static double calculate_local_weight(const quantum_state* state,
                                   size_t x,
                                   size_t y,
                                   size_t z);
static double estimate_error_probability(const quantum_state* state,
                                       const WeightConfig* config);
static bool validate_weight_parameters(const WeightConfig* config);

bool init_error_weight(WeightState* state, const WeightConfig* config) {
    if (!state || !config || !validate_weight_parameters(config)) {
        return false;
    }

    // Allocate weight map
    size_t map_size = config->lattice_width * config->lattice_height *
                      config->lattice_depth;
    state->weight_map = calloc(map_size, sizeof(double));
    if (!state->weight_map) {
        return false;
    }

    // Initialize state
    memcpy(&state->config, config, sizeof(WeightConfig));
    state->total_weight = 0.0;
    state->max_weight = 0.0;
    state->min_weight = INFINITY;
    state->measurement_count = 0;

    return true;
}

void cleanup_error_weight(WeightState* state) {
    if (state) {
        free(state->weight_map);
        memset(state, 0, sizeof(WeightState));
    }
}

bool calculate_error_weights(WeightState* state,
                           const quantum_state* qstate) {
    if (!state || !qstate || !state->weight_map) {
        return false;
    }

    size_t width = state->config.lattice_width;
    size_t height = state->config.lattice_height;
    size_t depth = state->config.lattice_depth;
    size_t map_size = width * height * depth;

    // Reset statistics
    state->total_weight = 0.0;
    state->max_weight = 0.0;
    state->min_weight = INFINITY;

    // Calculate weights for each point in the lattice
    for (size_t z = 0; z < depth; z++) {
        for (size_t y = 0; y < height; y++) {
            for (size_t x = 0; x < width; x++) {
                size_t idx = (z * height + y) * width + x;
                double weight = calculate_local_weight(qstate, x, y, z);
                
                // Apply weight scaling based on error probability
                double error_prob = estimate_error_probability(qstate,
                                                            &state->config);
                weight *= (1.0 + state->config.probability_factor * error_prob);

                // Apply geometric scaling
                if (state->config.use_geometric_scaling) {
                    // Scale based on distance from boundaries
                    double dx = fmin(x, width - 1 - x) / (double)width;
                    double dy = fmin(y, height - 1 - y) / (double)height;
                    double dz = fmin(z, depth - 1 - z) / (double)depth;
                    double boundary_factor = fmin(fmin(dx, dy), dz);
                    weight *= (1.0 + state->config.geometric_factor * boundary_factor);
                }

                // Store and update statistics
                state->weight_map[idx] = weight;
                state->total_weight += weight;
                state->max_weight = fmax(state->max_weight, weight);
                state->min_weight = fmin(state->min_weight, weight);
            }
        }
    }

    // Normalize weights if requested
    if (state->config.normalize_weights && map_size > 0) {
        double scale = 1.0 / state->total_weight;
        for (size_t i = 0; i < map_size; i++) {
            state->weight_map[i] *= scale;
        }
        state->total_weight = 1.0;
        state->max_weight *= scale;
        state->min_weight *= scale;
    }

    state->measurement_count++;
    return true;
}

double get_error_weight(const WeightState* state,
                       size_t x,
                       size_t y,
                       size_t z) {
    if (!state || !state->weight_map ||
        x >= state->config.lattice_width ||
        y >= state->config.lattice_height ||
        z >= state->config.lattice_depth) {
        return 0.0;
    }

    size_t idx = (z * state->config.lattice_height + y) *
                 state->config.lattice_width + x;
    return state->weight_map[idx];
}

bool get_weight_statistics(const WeightState* state,
                         WeightStatistics* stats) {
    if (!state || !stats) {
        return false;
    }

    stats->total_weight = state->total_weight;
    stats->max_weight = state->max_weight;
    stats->min_weight = state->min_weight;
    stats->measurement_count = state->measurement_count;

    return true;
}

const double* get_weight_map(const WeightState* state, size_t* size) {
    if (!state || !size) {
        return NULL;
    }

    *size = state->config.lattice_width *
            state->config.lattice_height *
            state->config.lattice_depth;
    return state->weight_map;
}

// Helper function implementations
static double calculate_local_weight(const quantum_state* state,
                                   size_t x,
                                   size_t y,
                                   size_t z) {
    if (!state) {
        return 0.0;
    }

    // Calculate base weight from quantum state properties
    double pauli_x, pauli_y, pauli_z;
    if (!measure_pauli_x(state, x, y, &pauli_x) ||
        !measure_pauli_y(state, x, y, &pauli_y) ||
        !measure_pauli_z(state, x, y, &pauli_z)) {
        return 0.0;
    }

    // Combine Pauli measurements into weight
    double weight = sqrt(pauli_x * pauli_x +
                        pauli_y * pauli_y +
                        pauli_z * pauli_z);

    return weight;
}

static double estimate_error_probability(const quantum_state* state,
                                       const WeightConfig* config) {
    if (!state || !config) {
        return 0.0;
    }

    // Estimate error probability using logarithmic scaling model
    // P_error = P_base × (1 + α × log₂(N)) where N is total qubits
    // This captures the scaling behavior observed in surface codes
    double base_error = config->base_error_rate;
    
    // Scale with system size
    size_t total_qubits = config->lattice_width *
                         config->lattice_height *
                         config->lattice_depth;
    double size_factor = log(total_qubits) / log(2.0);  // Log scaling
    
    return base_error * (1.0 + config->size_factor * size_factor);
}

static bool validate_weight_parameters(const WeightConfig* config) {
    if (!config) {
        return false;
    }

    // Check lattice dimensions
    if (config->lattice_width == 0 ||
        config->lattice_height == 0 ||
        config->lattice_depth == 0) {
        return false;
    }

    // Check scaling factors
    if (config->probability_factor < 0.0 ||
        config->geometric_factor < 0.0 ||
        config->size_factor < 0.0) {
        return false;
    }

    // Check error rate
    if (config->base_error_rate < 0.0 ||
        config->base_error_rate > 1.0) {
        return false;
    }

    return true;
}

// Pauli measurement implementations for error weight calculations
// Renamed to avoid conflict with heavy_hex_surface_code.c and parallel_stabilizer.c
static bool measure_pauli_x_for_weight(const quantum_state* state, size_t x, size_t y, double* result) {
    if (!state || !result) {
        return false;
    }

    // Get lattice dimensions from state
    size_t width = state->lattice_width > 0 ? state->lattice_width :
                   (size_t)sqrt((double)state->num_qubits);
    size_t idx = y * width + x;

    if (idx >= state->num_qubits || !state->coordinates) {
        *result = 0.0;
        return false;
    }

    // Calculate Pauli X expectation value
    // <X> = 2 * Re(a* × b) where state = a|0> + b|1>
    // For single qubit at position idx, extract amplitudes
    ComplexFloat c = state->coordinates[idx];

    // X measurement gives real part contribution
    *result = 2.0 * c.real;

    return true;
}

// Renamed to avoid conflict
static bool measure_pauli_y_for_weight(const quantum_state* state, size_t x, size_t y, double* result) {
    if (!state || !result) {
        return false;
    }

    // Get lattice dimensions from state
    size_t width = state->lattice_width > 0 ? state->lattice_width :
                   (size_t)sqrt((double)state->num_qubits);
    size_t idx = y * width + x;

    if (idx >= state->num_qubits || !state->coordinates) {
        *result = 0.0;
        return false;
    }

    // Calculate Pauli Y expectation value
    // <Y> = 2 * Im(a* × b) where state = a|0> + b|1>
    ComplexFloat c = state->coordinates[idx];

    // Y measurement gives imaginary part contribution
    *result = 2.0 * c.imag;

    return true;
}

// Renamed to avoid conflict
static bool measure_pauli_z_for_weight(const quantum_state* state, size_t x, size_t y, double* result) {
    if (!state || !result) {
        return false;
    }

    // Get lattice dimensions from state
    size_t width = state->lattice_width > 0 ? state->lattice_width :
                   (size_t)sqrt((double)state->num_qubits);
    size_t idx = y * width + x;

    if (idx >= state->num_qubits || !state->coordinates) {
        *result = 0.0;
        return false;
    }

    // Calculate Pauli Z expectation value
    // <Z> = |a|^2 - |b|^2 where state = a|0> + b|1>
    ComplexFloat c = state->coordinates[idx];
    double magnitude_sq = c.real * c.real + c.imag * c.imag;

    // Z measurement gives probability difference
    *result = 1.0 - 2.0 * magnitude_sq;

    return true;
}
