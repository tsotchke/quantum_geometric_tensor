/**
 * @file anyon_charge.c
 * @brief Implementation of anyon charge measurement system
 */

#include "quantum_geometric/physics/anyon_charge.h"
#include "quantum_geometric/physics/anyon_tracking.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Forward declarations
static bool initialize_charge_cache(ChargeCache* cache,
                                 const ChargeConfig* config);
static void cleanup_charge_cache(ChargeCache* cache);
static bool update_charge_measurements(ChargeState* state,
                                    const AnyonPositions* positions);
static bool detect_charge_patterns(ChargeState* state);

bool init_charge_measurement(ChargeState* state,
                           const ChargeConfig* config) {
    if (!state || !config) {
        return false;
    }

    // Initialize state
    memset(state, 0, sizeof(ChargeState));
    memcpy(&state->config, config, sizeof(ChargeConfig));

    // Initialize cache
    state->cache = malloc(sizeof(ChargeCache));
    if (!state->cache || !initialize_charge_cache(state->cache, config)) {
        cleanup_charge_measurement(state);
        return false;
    }

    // Initialize metrics
    state->total_measurements = 0;
    state->total_fusions = 0;
    state->conservation_score = 1.0;

    return true;
}

void cleanup_charge_measurement(ChargeState* state) {
    if (state) {
        if (state->cache) {
            cleanup_charge_cache(state->cache);
            free(state->cache);
        }
        memset(state, 0, sizeof(ChargeState));
    }
}

bool measure_charges(ChargeState* state,
                    const AnyonPositions* positions,
                    ChargeMeasurements* measurements) {
    if (!state || !positions || !measurements || !state->cache) {
        return false;
    }

    // Update charge measurements based on positions
    bool success = update_charge_measurements(state, positions);
    if (!success) {
        return false;
    }

    // Detect charge patterns
    success = detect_charge_patterns(state);
    if (!success) {
        return false;
    }

    // Prepare measurement output
    measurements->num_measurements = 0;
    for (size_t i = 0; i < positions->num_anyons; i++) {
        size_t idx = positions->y_coords[i] * state->config.lattice_width +
                    positions->x_coords[i];
        
        if (state->cache->charge_map[idx] != 0) {
            measurements->locations[measurements->num_measurements] = idx;
            measurements->charges[measurements->num_measurements] = 
                state->cache->charge_map[idx];
            measurements->probabilities[measurements->num_measurements] = 
                state->cache->probability_map[idx];
            measurements->num_measurements++;
        }
    }

    // Update metrics
    state->total_measurements++;
    update_charge_metrics(state);

    return true;
}

bool update_charge_metrics(ChargeState* state) {
    if (!state || !state->cache) {
        return false;
    }

    // Calculate charge conservation score
    double total_charge = 0.0;
    size_t total_sites = state->config.lattice_width *
                        state->config.lattice_height;
    
    for (size_t i = 0; i < total_sites; i++) {
        total_charge += state->cache->charge_map[i];
    }
    
    // Score based on deviation from zero total charge
    state->conservation_score = 1.0 / (1.0 + fabs(total_charge));

    return true;
}

static bool initialize_charge_cache(ChargeCache* cache,
                                 const ChargeConfig* config) {
    if (!cache || !config) {
        return false;
    }

    size_t total_sites = config->lattice_width * config->lattice_height;

    // Allocate arrays
    cache->charge_map = calloc(total_sites, sizeof(int));
    cache->probability_map = calloc(total_sites, sizeof(double));
    cache->fusion_history = calloc(total_sites * config->history_length,
                                 sizeof(int));
    cache->correlation_matrix = calloc(total_sites * total_sites,
                                    sizeof(double));

    if (!cache->charge_map || !cache->probability_map ||
        !cache->fusion_history || !cache->correlation_matrix) {
        cleanup_charge_cache(cache);
        return false;
    }

    return true;
}

static void cleanup_charge_cache(ChargeCache* cache) {
    if (cache) {
        free(cache->charge_map);
        free(cache->probability_map);
        free(cache->fusion_history);
        free(cache->correlation_matrix);
        memset(cache, 0, sizeof(ChargeCache));
    }
}

static bool update_charge_measurements(ChargeState* state,
                                    const AnyonPositions* positions) {
    if (!state || !positions || !state->cache) {
        return false;
    }

    // Clear current charge map
    size_t total_sites = state->config.lattice_width *
                        state->config.lattice_height;
    memset(state->cache->charge_map, 0, total_sites * sizeof(int));
    memset(state->cache->probability_map, 0, total_sites * sizeof(double));

    // Update charge and probability maps
    for (size_t i = 0; i < positions->num_anyons; i++) {
        size_t idx = positions->y_coords[i] * state->config.lattice_width +
                    positions->x_coords[i];
        
        if (idx < total_sites) {
            state->cache->charge_map[idx] = positions->charges[i];
            
            // Calculate measurement probability based on local charge density
            double local_density = 0.0;
            size_t neighbors = 0;
            
            // Check neighbors
            if (positions->x_coords[i] > 0) {
                size_t left = idx - 1;
                if (state->cache->charge_map[left] != 0) {
                    local_density += 1.0;
                }
                neighbors++;
            }
            if (positions->x_coords[i] < state->config.lattice_width - 1) {
                size_t right = idx + 1;
                if (state->cache->charge_map[right] != 0) {
                    local_density += 1.0;
                }
                neighbors++;
            }
            if (positions->y_coords[i] > 0) {
                size_t up = idx - state->config.lattice_width;
                if (state->cache->charge_map[up] != 0) {
                    local_density += 1.0;
                }
                neighbors++;
            }
            if (positions->y_coords[i] < state->config.lattice_height - 1) {
                size_t down = idx + state->config.lattice_width;
                if (state->cache->charge_map[down] != 0) {
                    local_density += 1.0;
                }
                neighbors++;
            }
            
            state->cache->probability_map[idx] = 
                neighbors > 0 ? local_density / neighbors : 1.0;
        }
    }

    // Update fusion history
    memmove(&state->cache->fusion_history[total_sites],
            state->cache->fusion_history,
            total_sites * (state->config.history_length - 1) * sizeof(int));
    
    memcpy(state->cache->fusion_history,
           state->cache->charge_map,
           total_sites * sizeof(int));

    return true;
}

static bool detect_charge_patterns(ChargeState* state) {
    if (!state || !state->cache) {
        return false;
    }

    size_t total_sites = state->config.lattice_width *
                        state->config.lattice_height;

    // Update correlation matrix
    for (size_t i = 0; i < total_sites; i++) {
        for (size_t j = i + 1; j < total_sites; j++) {
            // Calculate charge correlation over history
            double correlation = 0.0;
            size_t matches = 0;
            
            for (size_t k = 0; k < state->config.history_length; k++) {
                size_t hist_idx = k * total_sites;
                int charge_i = state->cache->fusion_history[hist_idx + i];
                int charge_j = state->cache->fusion_history[hist_idx + j];
                
                if (charge_i != 0 && charge_j != 0) {
                    correlation += (charge_i == charge_j) ? 1.0 : -1.0;
                    matches++;
                }
            }
            
            if (matches > 0) {
                state->cache->correlation_matrix[i * total_sites + j] =
                    correlation / matches;
            }
        }
    }

    state->total_fusions++;
    return true;
}
