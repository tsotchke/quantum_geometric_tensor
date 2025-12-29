/**
 * @file anyon_types.h
 * @brief Unified type definitions for anyon physics system
 *
 * This header provides unified type definitions that satisfy all test expectations.
 * Include this header FIRST before other anyon headers to ensure consistent types.
 */

#ifndef ANYON_TYPES_H
#define ANYON_TYPES_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================

#ifndef MAX_FUSION_CHANNELS
#define MAX_FUSION_CHANNELS 1024  // Tests expect 1024
#endif

#ifndef MAX_MEASUREMENTS
#define MAX_MEASUREMENTS 1024
#endif

#ifndef MAX_ANYONS
#define MAX_ANYONS 256
#endif

// ============================================================================
// Unified FusionConfig - combines fields from all modules
// ============================================================================

#ifndef FUSION_CONFIG_DEFINED
#define FUSION_CONFIG_DEFINED

/**
 * @brief Unified configuration for fusion operations
 *
 * This struct combines fields from anyon_fusion.h and anyon_detection.h
 * to satisfy all test expectations.
 */
typedef struct FusionConfig {
    // From anyon_fusion.h
    size_t lattice_width;           /**< Width of the lattice */
    size_t lattice_height;          /**< Height of the lattice */
    size_t history_length;          /**< Length of fusion history to track */
    bool track_statistics;          /**< Enable braiding statistics tracking */
    double probability_threshold;   /**< Minimum probability threshold */

    // From anyon_detection.h
    double energy_threshold;        /**< Maximum energy for fusion to occur */
    double coherence_requirement;   /**< Minimum coherence for valid fusion */
    size_t fusion_attempts;         /**< Number of fusion attempts */

    // Additional fields expected by tests
    double fusion_threshold;        /**< Fusion detection threshold (alias for tests) */
} FusionConfig;

#endif // FUSION_CONFIG_DEFINED

// ============================================================================
// Unified ChargeMeasurements
// ============================================================================

#ifndef CHARGE_MEASUREMENTS_DEFINED
#define CHARGE_MEASUREMENTS_DEFINED

/**
 * @brief Charge measurements from the lattice (static array version)
 *
 * Using static arrays for MAX_MEASUREMENTS to satisfy test expectations.
 */
typedef struct ChargeMeasurements {
    int charges[MAX_MEASUREMENTS];            /**< Measured charge values */
    size_t locations[MAX_MEASUREMENTS];       /**< Lattice locations of measurements */
    double probabilities[MAX_MEASUREMENTS];   /**< Measurement confidence values */
    size_t num_measurements;                  /**< Number of measurements */
} ChargeMeasurements;

#endif // CHARGE_MEASUREMENTS_DEFINED

// ============================================================================
// Unified FusionChannels
// ============================================================================

#ifndef FUSION_CHANNELS_DEFINED
#define FUSION_CHANNELS_DEFINED

/**
 * @brief Output fusion channels after fusion operation
 */
typedef struct FusionChannels {
    size_t num_channels;                              /**< Number of active channels */
    size_t locations[MAX_FUSION_CHANNELS];            /**< Fusion channel locations */
    int input_charges[MAX_FUSION_CHANNELS][2];        /**< Input charge pairs */
    int output_charge[MAX_FUSION_CHANNELS];           /**< Output charges */
    double probabilities[MAX_FUSION_CHANNELS];        /**< Channel probabilities */
} FusionChannels;

#endif // FUSION_CHANNELS_DEFINED

// ============================================================================
// Anyon Position Types
// ============================================================================

#ifndef ANYON_POSITIONS_DEFINED
#define ANYON_POSITIONS_DEFINED

/**
 * @brief Collection of anyon positions for tracking
 */
typedef struct AnyonPositions {
    size_t num_anyons;                    /**< Number of anyons */
    size_t x_coords[MAX_ANYONS];          /**< X coordinates */
    size_t y_coords[MAX_ANYONS];          /**< Y coordinates */
    int charges[MAX_ANYONS];              /**< Charge values */
} AnyonPositions;

#endif // ANYON_POSITIONS_DEFINED

#ifdef __cplusplus
}
#endif

#endif // ANYON_TYPES_H
