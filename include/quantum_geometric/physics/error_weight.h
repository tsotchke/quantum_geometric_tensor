#ifndef ERROR_WEIGHT_H
#define ERROR_WEIGHT_H

#include <stdbool.h>
#include <stddef.h>
#include "quantum_geometric/core/quantum_geometric_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration for quantum_state
struct quantum_state_t;
typedef struct quantum_state_t quantum_state;

/**
 * @brief Configuration for error weight calculations
 *
 * Controls the lattice geometry and scaling factors used when
 * computing error weights for quantum error correction.
 */
typedef struct WeightConfig {
    // Lattice dimensions
    size_t lattice_width;      /**< Width of the lattice */
    size_t lattice_height;     /**< Height of the lattice */
    size_t lattice_depth;      /**< Depth of the lattice (1 for 2D) */

    // Scaling factors
    double probability_factor; /**< Scaling factor for error probability */
    double geometric_factor;   /**< Scaling factor for geometric distance */
    double size_factor;        /**< System size scaling coefficient */

    // Error parameters
    double base_error_rate;    /**< Base error rate for the system */

    // Flags
    bool use_geometric_scaling; /**< Enable geometric distance scaling */
    bool normalize_weights;     /**< Normalize weights to sum to 1.0 */
} WeightConfig;

/**
 * @brief Statistics from weight calculations
 */
typedef struct WeightStatistics {
    double total_weight;       /**< Sum of all weights */
    double max_weight;         /**< Maximum weight value */
    double min_weight;         /**< Minimum weight value */
    size_t measurement_count;  /**< Number of measurements taken */
} WeightStatistics;

/**
 * @brief State for error weight calculations
 *
 * Maintains the weight map and configuration for ongoing
 * error weight calculations on a quantum lattice.
 */
typedef struct WeightState {
    double* weight_map;        /**< Array of weights for each lattice point */
    WeightConfig config;       /**< Configuration used for calculations */
    double total_weight;       /**< Current total weight */
    double max_weight;         /**< Current maximum weight */
    double min_weight;         /**< Current minimum weight */
    size_t measurement_count;  /**< Number of weight calculations performed */
} WeightState;

/**
 * @brief Initialize error weight calculation state
 *
 * @param state Pointer to WeightState to initialize
 * @param config Configuration for weight calculations
 * @return true on success, false on failure
 */
bool init_error_weight(WeightState* state, const WeightConfig* config);

/**
 * @brief Clean up error weight state and free resources
 *
 * @param state Pointer to WeightState to clean up
 */
void cleanup_error_weight(WeightState* state);

/**
 * @brief Calculate error weights from quantum state
 *
 * Computes the error weight for each point in the lattice based on
 * the quantum state amplitudes and configured scaling factors.
 *
 * @param state Weight state containing the lattice configuration
 * @param qstate Quantum state to analyze
 * @return true on success, false on failure
 */
bool calculate_error_weights(WeightState* state, const quantum_state* qstate);

/**
 * @brief Get error weight at a specific lattice position
 *
 * @param state Weight state containing the weight map
 * @param x X coordinate in the lattice
 * @param y Y coordinate in the lattice
 * @param z Z coordinate in the lattice
 * @return Weight value at the specified position, or 0.0 on error
 */
double get_error_weight(const WeightState* state,
                       size_t x, size_t y, size_t z);

/**
 * @brief Get statistics from weight calculations
 *
 * @param state Weight state to query
 * @param stats Output statistics structure
 * @return true on success, false on failure
 */
bool get_weight_statistics(const WeightState* state, WeightStatistics* stats);

/**
 * @brief Get pointer to the weight map array
 *
 * @param state Weight state containing the weight map
 * @param size Output parameter for array size
 * @return Pointer to weight map array, or NULL on error
 */
const double* get_weight_map(const WeightState* state, size_t* size);

/**
 * @brief Measure Pauli X expectation at a lattice position
 *
 * @param state Quantum state to measure
 * @param x X coordinate in the lattice
 * @param y Y coordinate in the lattice
 * @param result Output for measurement result
 * @return true on success, false on failure
 */
bool measure_pauli_x(const quantum_state* state, size_t x, size_t y, double* result);

/**
 * @brief Measure Pauli Y expectation at a lattice position
 *
 * @param state Quantum state to measure
 * @param x X coordinate in the lattice
 * @param y Y coordinate in the lattice
 * @param result Output for measurement result
 * @return true on success, false on failure
 */
bool measure_pauli_y(const quantum_state* state, size_t x, size_t y, double* result);

/**
 * @brief Measure Pauli Z expectation at a lattice position
 *
 * @param state Quantum state to measure
 * @param x X coordinate in the lattice
 * @param y Y coordinate in the lattice
 * @param result Output for measurement result
 * @return true on success, false on failure
 */
bool measure_pauli_z(const quantum_state* state, size_t x, size_t y, double* result);

#ifdef __cplusplus
}
#endif

#endif // ERROR_WEIGHT_H
