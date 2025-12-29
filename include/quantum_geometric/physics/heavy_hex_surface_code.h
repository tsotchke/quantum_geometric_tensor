/**
 * @file heavy_hex_surface_code.h
 * @brief Heavy-hex surface code implementation for IBM quantum hardware
 *
 * IBM's heavy-hex lattice topology provides improved error correction with
 * reduced cross-talk between qubits. This implementation follows the heavy-hex
 * geometry with degree-2 and degree-3 vertices arranged to minimize crosstalk
 * while maintaining full surface code error correction capabilities.
 *
 * The heavy-hex topology differs from standard square lattice surface codes:
 * - Hexagonal plaquettes with 6 qubits each
 * - Weight-6 Z stabilizers (plaquettes)
 * - Weight-6 X stabilizers (vertices)
 * - Optimized for superconducting qubit connectivity
 */

#ifndef HEAVY_HEX_SURFACE_CODE_H
#define HEAVY_HEX_SURFACE_CODE_H

#include <stdbool.h>
#include <stddef.h>
#include "quantum_geometric/physics/error_weight.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================

#define MAX_HEX_LATTICE_SIZE 127    // Maximum lattice dimension (IBM Eagle)
#define HEX_STABILIZER_WEIGHT 6     // Weight of hex stabilizers

// ============================================================================
// Type Definitions
// ============================================================================

/**
 * @brief Type of hex stabilizer
 */
typedef enum {
    HEX_PLAQUETTE = 0,   /**< Z-type stabilizer (plaquette) */
    HEX_VERTEX = 1       /**< X-type stabilizer (vertex) */
} HexStabilizerType;

/**
 * @brief Coordinate in heavy-hex lattice
 */
typedef struct {
    size_t x;                     /**< X coordinate */
    size_t y;                     /**< Y coordinate */
    HexStabilizerType type;       /**< Type of stabilizer at this location */
} HexCoordinate;

/**
 * @brief Heavy-hex lattice structure
 *
 * Represents the physical layout of qubits and stabilizers
 * in the heavy-hex topology.
 */
typedef struct {
    double* stabilizer_values;           /**< Current stabilizer measurement values */
    HexCoordinate* stabilizer_coordinates; /**< Coordinates of each stabilizer */
    size_t num_stabilizers;              /**< Number of stabilizers */
    size_t width;                        /**< Lattice width */
    size_t height;                       /**< Lattice height */

    // Qubit connectivity information
    size_t* qubit_indices;               /**< Physical qubit indices */
    size_t num_qubits;                   /**< Number of physical qubits */

    // Coupling map for the heavy-hex topology
    bool* coupling_map;                  /**< Adjacency matrix for qubit couplings */
    double* coupling_strengths;          /**< ZZ coupling strengths between qubits */

    // Error tracking
    double* qubit_error_rates;           /**< Per-qubit error rates */
    double* gate_error_rates;            /**< Per-gate error rates */
    double readout_error_rate;           /**< Average readout error rate */
} HexLattice;

/**
 * @brief Configuration for heavy-hex surface code
 */
typedef struct {
    // Lattice dimensions (must be odd, >= 3)
    size_t lattice_width;                /**< Width of the lattice */
    size_t lattice_height;               /**< Height of the lattice */

    // Error parameters
    double base_error_rate;              /**< Base physical error rate [0, 1] */
    double error_threshold;              /**< Threshold for error detection [0, 1] */
    double readout_error_rate;           /**< Readout error probability */
    double gate_error_rate;              /**< Two-qubit gate error rate */

    // Correction options
    bool auto_correction;                /**< Automatically apply corrections */
    bool use_soft_decoding;              /**< Use soft information in decoding */
    bool track_leakage;                  /**< Track leakage to non-computational states */

    // Timing parameters
    double measurement_time;             /**< Measurement duration in microseconds */
    double reset_time;                   /**< Reset duration in microseconds */
    double gate_time;                    /**< Two-qubit gate duration in nanoseconds */

    // Hardware-specific parameters
    double t1_time;                      /**< Qubit T1 relaxation time */
    double t2_time;                      /**< Qubit T2 dephasing time */
    double crosstalk_strength;           /**< ZZ crosstalk coefficient */
} HexConfig;

/**
 * @brief State container for heavy-hex surface code
 */
typedef struct {
    HexLattice* lattice;                 /**< Underlying hex lattice */
    WeightState weights;                 /**< Error weight calculation state */
    HexConfig config;                    /**< Configuration parameters */

    // Statistics
    size_t measurement_count;            /**< Number of measurement rounds */
    double error_rate;                   /**< Current estimated error rate */
    double logical_error_rate;           /**< Estimated logical error rate */

    // Syndrome history
    double* last_syndrome;               /**< Most recent syndrome values */
    double* syndrome_history;            /**< Historical syndrome data */
    size_t history_length;               /**< Length of syndrome history */
    size_t history_capacity;             /**< Capacity of syndrome history buffer */

    // Decoder state
    void* decoder_state;                 /**< Opaque decoder state (MWPM or Union-Find) */
} HexState;

// ============================================================================
// Initialization and Cleanup
// ============================================================================

/**
 * @brief Initialize heavy-hex surface code
 *
 * @param state Pointer to HexState to initialize
 * @param config Configuration parameters
 * @return true on success, false on failure
 */
bool init_heavy_hex_code(HexState* state, const HexConfig* config);

/**
 * @brief Clean up heavy-hex surface code and free resources
 *
 * @param state Pointer to HexState to clean up
 */
void cleanup_heavy_hex_code(HexState* state);

// ============================================================================
// Measurement and Correction
// ============================================================================

/**
 * @brief Perform stabilizer measurements on the hex code
 *
 * Measures all stabilizers in the heavy-hex lattice and updates
 * the error weights accordingly.
 *
 * @param state Heavy-hex state
 * @param qstate Quantum state to measure
 * @return true on success, false on failure
 */
bool measure_hex_code(HexState* state, quantum_state* qstate);

/**
 * @brief Get the most recent syndrome
 *
 * @param state Heavy-hex state
 * @param size Output parameter for syndrome size
 * @return Pointer to syndrome array, or NULL on error
 */
const double* get_hex_syndrome(const HexState* state, size_t* size);

/**
 * @brief Get current estimated error rate
 *
 * @param state Heavy-hex state
 * @return Estimated error rate [0, 1], or 0.0 on error
 */
double get_hex_error_rate(const HexState* state);

// ============================================================================
// Validation
// ============================================================================

/**
 * @brief Validate heavy-hex configuration parameters
 *
 * @param config Configuration to validate
 * @return true if configuration is valid
 */
bool validate_hex_parameters(const HexConfig* config);

// ============================================================================
// Pauli Operator Application (for error correction)
// ============================================================================

/**
 * @brief Apply Pauli X operator at a lattice position
 *
 * @param state Quantum state to modify
 * @param x X coordinate in the lattice
 * @param y Y coordinate in the lattice
 * @return true on success, false on failure
 */
bool hex_apply_pauli_x(quantum_state* state, size_t x, size_t y);

/**
 * @brief Apply Pauli Y operator at a lattice position
 *
 * @param state Quantum state to modify
 * @param x X coordinate in the lattice
 * @param y Y coordinate in the lattice
 * @return true on success, false on failure
 */
bool hex_apply_pauli_y(quantum_state* state, size_t x, size_t y);

/**
 * @brief Apply Pauli Z operator at a lattice position
 *
 * @param state Quantum state to modify
 * @param x X coordinate in the lattice
 * @param y Y coordinate in the lattice
 * @return true on success, false on failure
 */
bool hex_apply_pauli_z(quantum_state* state, size_t x, size_t y);

// ============================================================================
// Advanced Operations
// ============================================================================

/**
 * @brief Decode syndrome using minimum weight perfect matching
 *
 * @param state Heavy-hex state with syndrome data
 * @param corrections Output array for correction locations
 * @param max_corrections Maximum number of corrections
 * @return Number of corrections to apply
 */
size_t decode_hex_syndrome_mwpm(const HexState* state,
                                HexCoordinate* corrections,
                                size_t max_corrections);

/**
 * @brief Decode syndrome using Union-Find decoder
 *
 * Faster than MWPM but may be less optimal.
 *
 * @param state Heavy-hex state with syndrome data
 * @param corrections Output array for correction locations
 * @param max_corrections Maximum number of corrections
 * @return Number of corrections to apply
 */
size_t decode_hex_syndrome_uf(const HexState* state,
                              HexCoordinate* corrections,
                              size_t max_corrections);

/**
 * @brief Calculate logical error rate from syndrome history
 *
 * Uses statistical analysis of syndrome patterns to estimate
 * the logical error rate.
 *
 * @param state Heavy-hex state with syndrome history
 * @return Estimated logical error rate
 */
double calculate_logical_error_rate(const HexState* state);

/**
 * @brief Get coupling map for heavy-hex topology
 *
 * Returns the adjacency information for the heavy-hex lattice,
 * useful for circuit compilation.
 *
 * @param state Heavy-hex state
 * @param num_couplings Output parameter for number of couplings
 * @return Pointer to coupling map, or NULL on error
 */
const bool* get_hex_coupling_map(const HexState* state, size_t* num_couplings);

#ifdef __cplusplus
}
#endif

#endif // HEAVY_HEX_SURFACE_CODE_H
