/**
 * @file anyon_tracking.c
 * @brief Implementation of anyon position tracking system
 */

#include "quantum_geometric/physics/anyon_tracking.h"
#include "quantum_geometric/physics/syndrome_extraction.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Forward declarations
static bool initialize_tracking_cache(TrackingCache* cache,
                                   const TrackingConfig* config);
static void cleanup_tracking_cache(TrackingCache* cache);
static bool update_anyon_positions(TrackingState* state,
                                 const ErrorSyndrome* syndrome);
static bool detect_anyon_movement(TrackingState* state);

bool init_anyon_tracking(TrackingState* state,
                        const TrackingConfig* config) {
    if (!state || !config) {
        return false;
    }

    // Initialize state
    memset(state, 0, sizeof(TrackingState));
    memcpy(&state->config, config, sizeof(TrackingConfig));

    // Initialize cache
    state->cache = malloc(sizeof(TrackingCache));
    if (!state->cache || !initialize_tracking_cache(state->cache, config)) {
        cleanup_anyon_tracking(state);
        return false;
    }

    // Initialize metrics
    state->total_anyons = 0;
    state->total_movements = 0;
    state->stability_score = 1.0;

    return true;
}

void cleanup_anyon_tracking(TrackingState* state) {
    if (state) {
        if (state->cache) {
            cleanup_tracking_cache(state->cache);
            free(state->cache);
        }
        memset(state, 0, sizeof(TrackingState));
    }
}

bool track_anyons(TrackingState* state,
                 const ErrorSyndrome* syndrome,
                 AnyonPositions* positions) {
    if (!state || !syndrome || !positions || !state->cache) {
        return false;
    }

    // Update anyon positions based on syndrome
    bool success = update_anyon_positions(state, syndrome);
    if (!success) {
        return false;
    }

    // Detect anyon movement patterns
    success = detect_anyon_movement(state);
    if (!success) {
        return false;
    }

    // Prepare position output
    positions->num_anyons = 0;
    for (size_t i = 0; i < state->config.lattice_width *
                           state->config.lattice_height; i++) {
        if (state->cache->occupation_map[i]) {
            positions->x_coords[positions->num_anyons] = i % state->config.lattice_width;
            positions->y_coords[positions->num_anyons] = i / state->config.lattice_width;
            positions->charges[positions->num_anyons] = state->cache->charge_map[i];
            positions->num_anyons++;
        }
    }

    // Update metrics
    state->total_anyons = positions->num_anyons;
    update_tracking_metrics(state);

    return true;
}

bool update_tracking_metrics(TrackingState* state) {
    if (!state || !state->cache) {
        return false;
    }

    // Calculate stability score based on anyon movement
    double total_movement = 0.0;
    size_t total_sites = state->config.lattice_width *
                        state->config.lattice_height;
    
    for (size_t i = 0; i < total_sites; i++) {
        if (state->cache->occupation_map[i]) {
            total_movement += state->cache->movement_history[i];
        }
    }
    
    if (state->total_anyons > 0) {
        state->stability_score = 1.0 - (total_movement /
            (state->total_anyons * state->config.history_length));
    } else {
        state->stability_score = 1.0;
    }

    return true;
}

static bool initialize_tracking_cache(TrackingCache* cache,
                                   const TrackingConfig* config) {
    if (!cache || !config) {
        return false;
    }

    size_t total_sites = config->lattice_width * config->lattice_height;

    // Allocate arrays
    cache->occupation_map = calloc(total_sites, sizeof(bool));
    cache->charge_map = calloc(total_sites, sizeof(int));
    cache->movement_history = calloc(total_sites, sizeof(double));
    cache->position_history = calloc(total_sites * config->history_length,
                                   sizeof(size_t));

    if (!cache->occupation_map || !cache->charge_map ||
        !cache->movement_history || !cache->position_history) {
        cleanup_tracking_cache(cache);
        return false;
    }

    return true;
}

static void cleanup_tracking_cache(TrackingCache* cache) {
    if (cache) {
        free(cache->occupation_map);
        free(cache->charge_map);
        free(cache->movement_history);
        free(cache->position_history);
        memset(cache, 0, sizeof(TrackingCache));
    }
}

static bool update_anyon_positions(TrackingState* state,
                                 const ErrorSyndrome* syndrome) {
    if (!state || !syndrome || !state->cache) {
        return false;
    }

    // Clear current occupation map
    size_t total_sites = state->config.lattice_width *
                        state->config.lattice_height;
    memset(state->cache->occupation_map, 0, total_sites * sizeof(bool));

    // Update occupation and charge maps from syndrome
    for (size_t i = 0; i < syndrome->num_errors; i++) {
        size_t location = syndrome->error_locations[i];
        if (location < total_sites) {
            state->cache->occupation_map[location] = true;
            state->cache->charge_map[location] = 
                syndrome->error_types[i] == ERROR_X ? 1 : -1;
        }
    }

    // Update position history
    memmove(&state->cache->position_history[total_sites],
            state->cache->position_history,
            total_sites * (state->config.history_length - 1) * sizeof(size_t));
    
    for (size_t i = 0; i < total_sites; i++) {
        state->cache->position_history[i] = 
            state->cache->occupation_map[i] ? i : (size_t)-1;
    }

    return true;
}

static bool detect_anyon_movement(TrackingState* state) {
    if (!state || !state->cache) {
        return false;
    }

    size_t total_sites = state->config.lattice_width *
                        state->config.lattice_height;

    // Reset movement history
    memset(state->cache->movement_history, 0,
           total_sites * sizeof(double));

    // Analyze movement patterns
    for (size_t i = 0; i < total_sites; i++) {
        if (!state->cache->occupation_map[i]) {
            continue;
        }

        // Track position changes through history
        size_t movements = 0;
        size_t prev_pos = i;
        
        for (size_t j = 1; j < state->config.history_length; j++) {
            size_t hist_idx = j * total_sites + i;
            size_t pos = state->cache->position_history[hist_idx];
            
            if (pos != (size_t)-1 && pos != prev_pos) {
                movements++;
                prev_pos = pos;
            }
        }
        
        state->cache->movement_history[i] = 
            (double)movements / state->config.history_length;
    }

    state->total_movements++;
    return true;
}
