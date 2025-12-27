/**
 * @file anyon_tracking.h
 * @brief Anyon position tracking system for topological quantum error correction
 *
 * Provides infrastructure for tracking anyon positions and movements on a
 * lattice, detecting movement patterns, and analyzing tracking stability.
 */

#ifndef ANYON_TRACKING_H
#define ANYON_TRACKING_H

#include <stddef.h>
#include <stdbool.h>
#include "quantum_geometric/physics/anyon_charge.h"
#include "quantum_geometric/physics/syndrome_extraction.h"

#ifdef __cplusplus
extern "C" {
#endif

// Maximum tracking history length
#define MAX_TRACKING_HISTORY 256

/**
 * Configuration for anyon tracking system
 */
typedef struct TrackingConfig {
    size_t lattice_width;           // Width of the lattice
    size_t lattice_height;          // Height of the lattice
    size_t history_length;          // Number of historical positions to track
    double movement_threshold;      // Threshold for detecting movement
    double stability_threshold;     // Threshold for stability determination
    bool track_movement_patterns;   // Enable movement pattern analysis
} TrackingConfig;

/**
 * Cache for tracking computations
 */
typedef struct TrackingCache {
    bool* occupation_map;           // Current occupation at each lattice site
    int* charge_map;                // Charge at each occupied site
    double* movement_history;       // Movement frequency at each site
    size_t* position_history;       // Historical position data
    size_t cache_size;              // Size of cached data
} TrackingCache;

/**
 * State of the tracking system
 */
typedef struct TrackingState {
    TrackingConfig config;          // Configuration parameters
    TrackingCache* cache;           // Cached computation data
    size_t total_anyons;            // Total anyons currently tracked
    size_t total_movements;         // Total movement events detected
    double stability_score;         // Score indicating tracking stability [0,1]
} TrackingState;

/**
 * Initialize anyon tracking system
 * @param state Pointer to state structure to initialize
 * @param config Configuration parameters
 * @return true if initialization successful, false otherwise
 */
bool init_anyon_tracking(TrackingState* state,
                        const TrackingConfig* config);

/**
 * Clean up anyon tracking system
 * @param state State structure to clean up
 */
void cleanup_anyon_tracking(TrackingState* state);

/**
 * Track anyons based on error syndrome
 * @param state Tracking state
 * @param syndrome Current error syndrome
 * @param positions Output anyon positions
 * @return true if tracking successful, false otherwise
 */
bool track_anyons(TrackingState* state,
                 const ErrorSyndrome* syndrome,
                 AnyonPositions* positions);

/**
 * Update tracking metrics
 * @param state Tracking state
 * @return true if update successful, false otherwise
 */
bool update_tracking_metrics(TrackingState* state);

/**
 * Get current tracking stability score
 * @param state Tracking state
 * @return Stability score in range [0,1], higher is better
 */
static inline double get_stability_score(const TrackingState* state) {
    return state ? state->stability_score : 0.0;
}

/**
 * Get total number of tracked anyons
 * @param state Tracking state
 * @return Total anyon count
 */
static inline size_t get_tracked_anyon_count(const TrackingState* state) {
    return state ? state->total_anyons : 0;
}

/**
 * Check if tracking is stable within tolerance
 * @param state Tracking state
 * @param tolerance Acceptable deviation from stability
 * @return true if tracking is stable within tolerance
 */
static inline bool is_tracking_stable(const TrackingState* state, double tolerance) {
    return state && state->stability_score >= (1.0 - tolerance);
}

#ifdef __cplusplus
}
#endif

#endif // ANYON_TRACKING_H
