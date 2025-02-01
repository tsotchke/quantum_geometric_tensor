/**
 * @file anyon_fusion.c
 * @brief Implementation of anyon fusion rules system
 */

#include "quantum_geometric/physics/anyon_fusion.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Forward declarations
static bool initialize_fusion_cache(FusionCache* cache,
                                 const FusionConfig* config);
static void cleanup_fusion_cache(FusionCache* cache);
static bool update_fusion_channels(FusionState* state,
                                const ChargeMeasurements* measurements);
static bool detect_braiding_patterns(FusionState* state);

bool init_fusion_rules(FusionState* state,
                      const FusionConfig* config) {
    if (!state || !config) {
        return false;
    }

    // Initialize state
    memset(state, 0, sizeof(FusionState));
    memcpy(&state->config, config, sizeof(FusionConfig));

    // Initialize cache
    state->cache = malloc(sizeof(FusionCache));
    if (!state->cache || !initialize_fusion_cache(state->cache, config)) {
        cleanup_fusion_rules(state);
        return false;
    }

    // Initialize metrics
    state->total_fusions = 0;
    state->total_braidings = 0;
    state->consistency_score = 1.0;

    return true;
}

void cleanup_fusion_rules(FusionState* state) {
    if (state) {
        if (state->cache) {
            cleanup_fusion_cache(state->cache);
            free(state->cache);
        }
        memset(state, 0, sizeof(FusionState));
    }
}

bool determine_fusion_channels(FusionState* state,
                             const ChargeMeasurements* measurements,
                             FusionChannels* channels) {
    if (!state || !measurements || !channels || !state->cache) {
        return false;
    }

    // Update fusion channels based on measurements
    bool success = update_fusion_channels(state, measurements);
    if (!success) {
        return false;
    }

    // Detect braiding patterns if enabled
    if (state->config.track_statistics) {
        success = detect_braiding_patterns(state);
        if (!success) {
            return false;
        }
    }

    // Prepare channel output
    channels->num_channels = 0;
    size_t total_sites = state->config.lattice_width *
                        state->config.lattice_height;

    for (size_t i = 0; i < total_sites; i++) {
        if (state->cache->channel_map[i] != 0) {
            channels->locations[channels->num_channels] = i;
            
            // Find input charges from measurements
            size_t charge_count = 0;
            for (size_t j = 0; j < measurements->num_measurements; j++) {
                if (measurements->locations[j] == i ||
                    (measurements->locations[j] == i + 1 && i % state->config.lattice_width < state->config.lattice_width - 1) ||
                    (measurements->locations[j] == i + state->config.lattice_width && i / state->config.lattice_width < state->config.lattice_height - 1)) {
                    
                    if (charge_count < 2) {
                        channels->input_charges[channels->num_channels][charge_count] =
                            measurements->charges[j];
                        charge_count++;
                    }
                }
            }
            
            channels->output_charge[channels->num_channels] =
                state->cache->channel_map[i];
            channels->probabilities[channels->num_channels] =
                state->cache->probability_map[i];
            channels->num_channels++;
        }
    }

    // Update metrics
    state->total_fusions++;
    update_fusion_metrics(state);

    return true;
}

bool update_fusion_metrics(FusionState* state) {
    if (!state || !state->cache) {
        return false;
    }

    // Calculate fusion rule consistency
    double total_consistency = 0.0;
    size_t total_sites = state->config.lattice_width *
                        state->config.lattice_height;
    size_t valid_channels = 0;

    for (size_t i = 0; i < total_sites; i++) {
        if (state->cache->channel_map[i] != 0) {
            // Check fusion rule conservation
            size_t hist_idx = i;
            int expected_charge = 0;
            int actual_charge = state->cache->channel_map[i];
            
            for (size_t j = 0; j < state->config.history_length; j++) {
                expected_charge += state->cache->fusion_history[hist_idx];
                hist_idx += total_sites;
            }
            
            total_consistency += (expected_charge == actual_charge) ? 1.0 : 0.0;
            valid_channels++;
        }
    }

    state->consistency_score = valid_channels > 0 ?
        total_consistency / valid_channels : 1.0;

    return true;
}

static bool initialize_fusion_cache(FusionCache* cache,
                                 const FusionConfig* config) {
    if (!cache || !config) {
        return false;
    }

    size_t total_sites = config->lattice_width * config->lattice_height;

    // Allocate arrays
    cache->channel_map = calloc(total_sites, sizeof(int));
    cache->probability_map = calloc(total_sites, sizeof(double));
    cache->fusion_history = calloc(total_sites * config->history_length,
                                 sizeof(int));
    cache->statistics_matrix = calloc(total_sites * total_sites,
                                   sizeof(double));

    if (!cache->channel_map || !cache->probability_map ||
        !cache->fusion_history || !cache->statistics_matrix) {
        cleanup_fusion_cache(cache);
        return false;
    }

    return true;
}

static void cleanup_fusion_cache(FusionCache* cache) {
    if (cache) {
        free(cache->channel_map);
        free(cache->probability_map);
        free(cache->fusion_history);
        free(cache->statistics_matrix);
        memset(cache, 0, sizeof(FusionCache));
    }
}

static bool update_fusion_channels(FusionState* state,
                                const ChargeMeasurements* measurements) {
    if (!state || !measurements || !state->cache) {
        return false;
    }

    // Clear current channel map
    size_t total_sites = state->config.lattice_width *
                        state->config.lattice_height;
    memset(state->cache->channel_map, 0, total_sites * sizeof(int));
    memset(state->cache->probability_map, 0, total_sites * sizeof(double));

    // Identify potential fusion channels
    for (size_t i = 0; i < measurements->num_measurements; i++) {
        for (size_t j = i + 1; j < measurements->num_measurements; j++) {
            size_t loc_i = measurements->locations[i];
            size_t loc_j = measurements->locations[j];
            
            // Check if charges are adjacent
            if ((abs((int)(loc_i % state->config.lattice_width) -
                    (int)(loc_j % state->config.lattice_width)) <= 1) &&
                (abs((int)(loc_i / state->config.lattice_width) -
                    (int)(loc_j / state->config.lattice_width)) <= 1)) {
                
                // Determine fusion channel location (midpoint)
                size_t channel_loc = (loc_i + loc_j) / 2;
                
                // Apply fusion rules
                int charge_i = measurements->charges[i];
                int charge_j = measurements->charges[j];
                
                // Simple abelian fusion rule: charges add
                state->cache->channel_map[channel_loc] = charge_i + charge_j;
                
                // Calculate probability based on measurement confidences
                state->cache->probability_map[channel_loc] =
                    measurements->probabilities[i] *
                    measurements->probabilities[j];
            }
        }
    }

    // Update fusion history
    memmove(&state->cache->fusion_history[total_sites],
            state->cache->fusion_history,
            total_sites * (state->config.history_length - 1) * sizeof(int));
    
    memcpy(state->cache->fusion_history,
           state->cache->channel_map,
           total_sites * sizeof(int));

    return true;
}

static bool detect_braiding_patterns(FusionState* state) {
    if (!state || !state->cache) {
        return false;
    }

    size_t total_sites = state->config.lattice_width *
                        state->config.lattice_height;

    // Update braiding statistics matrix
    for (size_t i = 0; i < total_sites; i++) {
        for (size_t j = i + 1; j < total_sites; j++) {
            // Calculate braiding phase from history
            double phase = 0.0;
            size_t crossings = 0;
            
            for (size_t k = 1; k < state->config.history_length; k++) {
                size_t hist_idx = k * total_sites;
                int prev_i = state->cache->fusion_history[hist_idx - total_sites + i];
                int prev_j = state->cache->fusion_history[hist_idx - total_sites + j];
                int curr_i = state->cache->fusion_history[hist_idx + i];
                int curr_j = state->cache->fusion_history[hist_idx + j];
                
                if (prev_i != 0 && prev_j != 0 && curr_i != 0 && curr_j != 0) {
                    // Detect crossings through relative position changes
                    if ((prev_i < prev_j && curr_i > curr_j) ||
                        (prev_i > prev_j && curr_i < curr_j)) {
                        phase += M_PI;  // Accumulate braiding phase
                        crossings++;
                    }
                }
            }
            
            if (crossings > 0) {
                state->cache->statistics_matrix[i * total_sites + j] =
                    phase / crossings;
                state->total_braidings++;
            }
        }
    }

    return true;
}
