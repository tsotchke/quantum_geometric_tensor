/**
 * @file anyon_charge.h
 * @brief Anyon charge measurement and tracking system
 *
 * Provides infrastructure for measuring and tracking topological charges
 * of anyons on a lattice, including charge conservation verification
 * and correlation analysis.
 */

#ifndef ANYON_CHARGE_H
#define ANYON_CHARGE_H

#include <stddef.h>
#include <stdbool.h>
#include "quantum_geometric/physics/anyon_detection.h"

#ifdef __cplusplus
extern "C" {
#endif

// Maximum number of anyons for static allocation
#define MAX_CHARGE_ANYONS 1024

// Maximum measurement history length
#define MAX_CHARGE_HISTORY 128

/**
 * Configuration for charge measurement system
 */
typedef struct ChargeConfig {
    size_t lattice_width;           // Width of the lattice
    size_t lattice_height;          // Height of the lattice
    size_t history_length;          // Number of historical measurements to track
    double measurement_threshold;   // Threshold for valid charge measurement
    double correlation_threshold;   // Threshold for significant correlation
    bool track_correlations;        // Enable correlation tracking

    // Test-expected field (TDD compatibility)
    double fusion_threshold;        // Threshold for fusion detection (tests expect this)
} ChargeConfig;

/**
 * Cache for charge measurement computations
 */
typedef struct ChargeCache {
    int* charge_map;                // Current charge at each lattice site
    double* probability_map;        // Measurement probability at each site
    int* fusion_history;            // Historical charge values for pattern detection
    double* correlation_matrix;     // Charge-charge correlation matrix
    size_t cache_size;              // Size of cached data
} ChargeCache;

/**
 * State of the charge measurement system
 */
typedef struct ChargeState {
    ChargeConfig config;            // Configuration parameters
    ChargeCache* cache;             // Cached computation data
    size_t total_measurements;      // Total number of measurements performed
    size_t total_fusions;           // Total number of fusion events detected
    double conservation_score;      // Score indicating charge conservation [0,1]
} ChargeState;

/**
 * Collection of anyon positions for charge measurement
 */
#ifndef ANYON_POSITIONS_DEFINED
#define ANYON_POSITIONS_DEFINED
typedef struct AnyonPositions {
    size_t num_anyons;                      // Number of anyons
    size_t x_coords[MAX_CHARGE_ANYONS];     // X coordinates
    size_t y_coords[MAX_CHARGE_ANYONS];     // Y coordinates
    int charges[MAX_CHARGE_ANYONS];         // Charge values
} AnyonPositions;
#endif // ANYON_POSITIONS_DEFINED

/**
 * Results of charge measurements
 */
#ifndef CHARGE_MEASUREMENTS_DEFINED
#define CHARGE_MEASUREMENTS_DEFINED
typedef struct ChargeMeasurements {
    size_t num_measurements;                // Number of valid measurements
    size_t locations[MAX_CHARGE_ANYONS];    // Lattice site indices
    int charges[MAX_CHARGE_ANYONS];         // Measured charge values
    double probabilities[MAX_CHARGE_ANYONS]; // Measurement probabilities
} ChargeMeasurements;
#endif

/**
 * Initialize charge measurement system
 * @param state Pointer to state structure to initialize
 * @param config Configuration parameters
 * @return true if initialization successful, false otherwise
 */
bool init_charge_measurement(ChargeState* state,
                            const ChargeConfig* config);

/**
 * Clean up charge measurement system
 * @param state State structure to clean up
 */
void cleanup_charge_measurement(ChargeState* state);

/**
 * Measure charges at anyon positions
 * @param state Charge measurement state
 * @param positions Current anyon positions
 * @param measurements Output measurement results
 * @return true if measurement successful, false otherwise
 */
bool measure_charges(ChargeState* state,
                    const AnyonPositions* positions,
                    ChargeMeasurements* measurements);

/**
 * Update charge measurement metrics
 * @param state Charge measurement state
 * @return true if update successful, false otherwise
 */
bool update_charge_metrics(ChargeState* state);

/**
 * Get current charge conservation score
 * @param state Charge measurement state
 * @return Conservation score in range [0,1], higher is better
 */
static inline double get_conservation_score(const ChargeState* state) {
    return state ? state->conservation_score : 0.0;
}

/**
 * Get total number of measurements performed
 * @param state Charge measurement state
 * @return Total measurement count
 */
static inline size_t get_measurement_count(const ChargeState* state) {
    return state ? state->total_measurements : 0;
}

/**
 * Check if charge is conserved within tolerance
 * @param state Charge measurement state
 * @param tolerance Acceptable deviation from conservation
 * @return true if charge is conserved within tolerance
 */
static inline bool is_charge_conserved(const ChargeState* state, double tolerance) {
    return state && state->conservation_score >= (1.0 - tolerance);
}

#ifdef __cplusplus
}
#endif

#endif // ANYON_CHARGE_H
