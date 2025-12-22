/**
 * @file anyon_fusion.h
 * @brief Anyon fusion rules system for topological quantum computing
 *
 * This module implements the fusion rules for anyonic excitations,
 * enabling topologically protected quantum computation through
 * braiding and fusion operations.
 */

#ifndef ANYON_FUSION_H
#define ANYON_FUSION_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Maximum number of fusion channels
#define MAX_FUSION_CHANNELS 256
#define MAX_MEASUREMENTS 1024

// ============================================================================
// Fusion Configuration
// ============================================================================

/**
 * @brief Configuration for fusion rule calculations
 */
typedef struct {
    size_t lattice_width;     /**< Width of the lattice */
    size_t lattice_height;    /**< Height of the lattice */
    size_t history_length;    /**< Length of fusion history to track */
    bool track_statistics;    /**< Enable braiding statistics tracking */
    double probability_threshold; /**< Minimum probability threshold */
} FusionConfig;

// ============================================================================
// Measurement Types
// ============================================================================

/**
 * @brief Charge measurements from the lattice
 */
typedef struct {
    int* charges;             /**< Measured charge values */
    size_t* locations;        /**< Lattice locations of measurements */
    double* probabilities;    /**< Measurement confidence values */
    size_t num_measurements;  /**< Number of measurements */
} ChargeMeasurements;

// ============================================================================
// Fusion Output Types
// ============================================================================

/**
 * @brief Output fusion channels after fusion operation
 */
typedef struct {
    size_t locations[MAX_FUSION_CHANNELS];       /**< Fusion channel locations */
    int input_charges[MAX_FUSION_CHANNELS][2];   /**< Input charge pairs */
    int output_charge[MAX_FUSION_CHANNELS];      /**< Output charges */
    double probabilities[MAX_FUSION_CHANNELS];   /**< Channel probabilities */
    size_t num_channels;                         /**< Number of active channels */
} FusionChannels;

// ============================================================================
// Internal Cache Structure
// ============================================================================

/**
 * @brief Internal cache for fusion calculations
 */
typedef struct {
    int* channel_map;         /**< Map of fusion channels on lattice */
    double* probability_map;  /**< Probability map for channels */
    int* fusion_history;      /**< History of fusion operations */
    double* statistics_matrix; /**< Braiding statistics matrix */
} FusionCache;

// ============================================================================
// Fusion State
// ============================================================================

/**
 * @brief Main state structure for fusion system
 */
typedef struct {
    FusionConfig config;      /**< Configuration parameters */
    FusionCache* cache;       /**< Internal calculation cache */

    // Metrics
    size_t total_fusions;     /**< Total fusion operations performed */
    size_t total_braidings;   /**< Total braiding operations detected */
    double consistency_score; /**< Fusion rule consistency score */
} FusionState;

// ============================================================================
// Initialization and Cleanup
// ============================================================================

/**
 * @brief Initialize fusion rules system
 *
 * @param state Fusion state to initialize
 * @param config Configuration parameters
 * @return true on success, false on failure
 */
bool init_fusion_rules(FusionState* state, const FusionConfig* config);

/**
 * @brief Clean up fusion rules system
 *
 * @param state Fusion state to clean up
 */
void cleanup_fusion_rules(FusionState* state);

// ============================================================================
// Fusion Operations
// ============================================================================

/**
 * @brief Determine fusion channels from charge measurements
 *
 * @param state Current fusion state
 * @param measurements Input charge measurements
 * @param channels Output fusion channels
 * @return true on success, false on failure
 */
bool determine_fusion_channels(FusionState* state,
                               const ChargeMeasurements* measurements,
                               FusionChannels* channels);

/**
 * @brief Update fusion metrics after operations
 *
 * @param state Fusion state to update
 * @return true on success, false on failure
 */
bool update_fusion_metrics(FusionState* state);

/**
 * @brief Apply fusion rule to pair of charges
 *
 * @param charge1 First input charge
 * @param charge2 Second input charge
 * @param result Output charge (if unique)
 * @param num_channels Number of possible output channels
 * @return true if fusion is allowed
 */
bool apply_fusion_rule(int charge1, int charge2, int* result, size_t* num_channels);

/**
 * @brief Calculate fusion coefficient (F-matrix element)
 *
 * @param a First charge
 * @param b Second charge
 * @param c Third charge
 * @param d Fourth charge
 * @return F-matrix coefficient
 */
double fusion_coefficient(int a, int b, int c, int d);

/**
 * @brief Calculate R-matrix (braiding) coefficient
 *
 * @param charge1 First charge being braided
 * @param charge2 Second charge being braided
 * @param fusion_channel Fusion outcome
 * @param real_part Output real part
 * @param imag_part Output imaginary part
 */
void braiding_coefficient(int charge1, int charge2, int fusion_channel,
                          double* real_part, double* imag_part);

// ============================================================================
// Analysis Functions
// ============================================================================

/**
 * @brief Get current fusion consistency score
 *
 * @param state Fusion state
 * @return Consistency score between 0 and 1
 */
double get_fusion_consistency(const FusionState* state);

/**
 * @brief Check if fusion outcome is valid
 *
 * @param charge1 First input charge
 * @param charge2 Second input charge
 * @param result Proposed fusion result
 * @return true if result is valid outcome
 */
bool is_valid_fusion(int charge1, int charge2, int result);

/**
 * @brief Get quantum dimension for charge
 *
 * @param charge Charge value
 * @return Quantum dimension
 */
double quantum_dimension(int charge);

#ifdef __cplusplus
}
#endif

#endif // ANYON_FUSION_H
