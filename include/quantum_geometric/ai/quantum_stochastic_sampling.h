/**
 * @file quantum_stochastic_sampling.h
 * @brief Quantum stochastic sampling operations for AI/ML applications
 *
 * This module provides efficient quantum-inspired stochastic sampling
 * operations with SIMD acceleration and GPU support for large-scale
 * machine learning applications.
 */

#ifndef QUANTUM_STOCHASTIC_SAMPLING_H
#define QUANTUM_STOCHASTIC_SAMPLING_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "quantum_geometric/core/error_codes.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Configuration Types
// ============================================================================

/**
 * @brief Configuration for sampling operations
 */
typedef struct {
    size_t num_samples;         /**< Number of samples to generate */
    size_t state_dim;           /**< Dimension of state space */
    unsigned int seed;          /**< Random seed (0 = use system time) */
    bool use_gpu;               /**< Enable GPU acceleration */
    bool normalize;             /**< Auto-normalize probabilities */
    double temperature;         /**< Sampling temperature */
} sampling_config_t;

/**
 * @brief State for sampling operations
 */
typedef struct {
    float* state_vector;        /**< Current quantum state vector */
    float* probabilities;       /**< Probability distribution */
    float* cumulative_probs;    /**< Cumulative probability distribution */
    size_t state_dim;           /**< Dimension of state */
    bool normalized;            /**< Whether probabilities are normalized */
} sampling_state_t;

/**
 * @brief Result of sampling operation
 */
typedef struct {
    size_t* samples;            /**< Sampled indices */
    float* weights;             /**< Sample weights */
    size_t num_samples;         /**< Number of samples generated */
    double sampling_time;       /**< Time taken for sampling */
} sampling_result_t;

/**
 * @brief Statistics from sampling operations
 */
typedef struct {
    size_t total_samples;       /**< Total samples generated */
    size_t total_measurements;  /**< Total measurements performed */
    double total_sampling_time; /**< Total time spent sampling */
    double avg_sampling_time;   /**< Average time per sample */
    size_t peak_memory;         /**< Peak memory usage */
} sampling_stats_t;

// ============================================================================
// Initialization and Cleanup
// ============================================================================

/**
 * @brief Initialize sampling state with configuration
 */
int qg_sampling_init(sampling_state_t* state, const sampling_config_t* config);

/**
 * @brief Clean up sampling state and release resources
 */
void qg_sampling_cleanup(sampling_state_t* state);

// ============================================================================
// State Preparation
// ============================================================================

/**
 * @brief Prepare quantum state for sampling
 */
int qg_sampling_prepare_state(sampling_state_t* state,
                              const float* initial_state,
                              size_t state_size);

/**
 * @brief Update probability distribution
 */
int qg_sampling_update_probabilities(sampling_state_t* state,
                                     const float* new_probs,
                                     size_t prob_size);

/**
 * @brief Normalize probability distribution
 */
int qg_sampling_normalize(sampling_state_t* state);

// ============================================================================
// Sampling Operations
// ============================================================================

/**
 * @brief Sample from the current probability distribution
 */
int qg_sampling_sample(sampling_state_t* state,
                       size_t num_samples,
                       sampling_result_t* result);

/**
 * @brief Perform importance sampling
 */
int qg_sampling_importance(sampling_state_t* state,
                           const float* proposal_probs,
                           size_t num_samples,
                           sampling_result_t* result);

/**
 * @brief Perform Metropolis-Hastings sampling
 */
int qg_sampling_metropolis(sampling_state_t* state,
                           const float* proposal_probs,
                           size_t num_samples,
                           double temperature,
                           sampling_result_t* result);

/**
 * @brief Sample with stratification for variance reduction
 */
int qg_sampling_stratified(sampling_state_t* state,
                           size_t num_strata,
                           size_t samples_per_stratum,
                           sampling_result_t* result);

// ============================================================================
// GPU-Accelerated Operations
// ============================================================================

/**
 * @brief GPU version of state preparation
 */
int qg_sampling_prepare_state_gpu(sampling_state_t* state,
                                  const float* initial_state,
                                  size_t state_size);

/**
 * @brief GPU version of probability update
 */
int qg_sampling_update_probabilities_gpu(sampling_state_t* state,
                                         const float* new_probs,
                                         size_t prob_size);

/**
 * @brief GPU version of sampling
 */
int qg_sampling_sample_gpu(sampling_state_t* state,
                           size_t num_samples,
                           sampling_result_t* result);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Check if GPU is available for sampling
 */
bool is_gpu_available(void);

/**
 * @brief Get sampling statistics
 */
int qg_sampling_get_stats(sampling_stats_t* stats);

/**
 * @brief Reset sampling statistics
 */
void qg_sampling_reset_stats(void);

/**
 * @brief Free sampling result
 */
void qg_sampling_free_result(sampling_result_t* result);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_STOCHASTIC_SAMPLING_H
