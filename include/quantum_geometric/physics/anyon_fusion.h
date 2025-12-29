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

// Include unified types first - these provide canonical definitions
#include "quantum_geometric/physics/anyon_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Use unified MAX_FUSION_CHANNELS from anyon_types.h (1024)
// Legacy compatibility - tests expect 1024
#ifdef MAX_FUSION_CHANNELS
#undef MAX_FUSION_CHANNELS
#endif
#define MAX_FUSION_CHANNELS 1024

#ifndef MAX_MEASUREMENTS
#define MAX_MEASUREMENTS 1024
#endif

// ============================================================================
// Fusion Configuration - Use unified FusionConfig from anyon_types.h
// ============================================================================

// FusionConfig is defined in anyon_types.h with all required fields

// ============================================================================
// Measurement Types - Use unified ChargeMeasurements from anyon_types.h
// ============================================================================

// ChargeMeasurements is defined in anyon_types.h with static arrays

// ============================================================================
// Fusion Output Types - Use unified FusionChannels from anyon_types.h
// ============================================================================

// FusionChannels is defined in anyon_types.h

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
