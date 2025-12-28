/**
 * @file amplitude_amplification.h
 * @brief General Amplitude Amplification Algorithm
 *
 * Amplitude amplification generalizes Grover's algorithm to work with
 * any quantum subroutine that prepares a state with some amplitude in
 * the target subspace. It quadratically amplifies the success probability.
 *
 * Given:
 * - A quantum algorithm A that maps |0⟩ → |ψ⟩ = sin(θ)|good⟩ + cos(θ)|bad⟩
 * - An oracle O that marks |good⟩ states with a phase flip
 *
 * Amplitude amplification applies (A S₀ A† O)^k to amplify the good states,
 * where S₀ is reflection about |0⟩ and k ≈ π/(4θ) - 1/2.
 *
 * Applications:
 * - Quantum counting
 * - Fixed-point amplitude amplification
 * - Oblivious amplitude amplification
 * - Variable time amplitude amplification
 */

#ifndef AMPLITUDE_AMPLIFICATION_H
#define AMPLITUDE_AMPLIFICATION_H

#include <stddef.h>
#include <stdbool.h>
#include "quantum_geometric/core/quantum_types.h"
#include "quantum_geometric/core/quantum_state.h"
#include "quantum_geometric/core/error_codes.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Type Definitions
// ============================================================================

/**
 * @brief State preparation algorithm A: |0⟩ → |ψ⟩
 *
 * @param state Quantum state to transform
 * @param num_qubits Number of qubits
 * @param user_data User-provided context
 * @return Error code
 */
typedef qgt_error_t (*amp_prepare_func_t)(ComplexFloat* state, size_t num_qubits,
                                           void* user_data);

/**
 * @brief Oracle function O: marks good states with phase flip
 *
 * @param state Quantum state to transform
 * @param num_qubits Number of qubits
 * @param user_data User-provided context
 * @return Error code
 */
typedef qgt_error_t (*amp_oracle_func_t)(ComplexFloat* state, size_t num_qubits,
                                          void* user_data);

/**
 * @brief Amplitude amplification configuration
 */
typedef struct {
    size_t max_iterations;          // Maximum Grover iterations
    double target_probability;      // Target success probability (0.5 - 1.0)
    bool fixed_point;               // Use fixed-point amplitude amplification
    bool use_quantum_counting;      // Estimate theta before amplification
    size_t counting_precision;      // Bits of precision for quantum counting
    bool use_variable_time;         // Variable time amplitude amplification
    bool use_gpu;                   // Enable GPU acceleration
} amp_config_t;

/**
 * @brief Amplitude amplification state
 */
typedef struct amp_state {
    size_t num_qubits;              // Number of qubits
    size_t dimension;               // State vector dimension (2^n)
    ComplexFloat* amplitudes;       // State vector
    amp_prepare_func_t prepare;     // State preparation A
    amp_prepare_func_t prepare_inv; // Inverse preparation A†
    amp_oracle_func_t oracle;       // Oracle O
    void* prepare_data;             // User data for prepare
    void* oracle_data;              // User data for oracle
    amp_config_t config;            // Configuration
    double theta;                   // Estimated initial angle (if known)
    size_t iterations;              // Number of iterations applied
} amp_state_t;

/**
 * @brief Result of amplitude amplification
 */
typedef struct {
    bool success;                   // Whether target probability achieved
    double final_probability;       // Probability of measuring good state
    size_t* good_states;            // Measured good states
    size_t num_good_states;         // Number of good states found
    size_t iterations_used;         // Number of iterations applied
    double estimated_theta;         // Estimated initial angle
    double execution_time;          // Execution time in seconds
} amp_result_t;

// ============================================================================
// Core Amplitude Amplification
// ============================================================================

/**
 * @brief Create default configuration
 */
amp_config_t amp_default_config(void);

/**
 * @brief Initialize amplitude amplification state
 *
 * @param num_qubits Number of qubits
 * @param prepare State preparation function A
 * @param prepare_inv Inverse preparation A† (NULL to auto-compute)
 * @param oracle Oracle function O
 * @param prepare_data User data for prepare functions
 * @param oracle_data User data for oracle
 * @param config Configuration (NULL for defaults)
 * @return Initialized state, or NULL on error
 */
amp_state_t* amp_init(size_t num_qubits,
                       amp_prepare_func_t prepare,
                       amp_prepare_func_t prepare_inv,
                       amp_oracle_func_t oracle,
                       void* prepare_data,
                       void* oracle_data,
                       const amp_config_t* config);

/**
 * @brief Run amplitude amplification
 *
 * @param state Initialized amplification state
 * @return Result containing success probability and measurements
 */
amp_result_t* amp_run(amp_state_t* state);

/**
 * @brief Apply a single Grover iteration (A S₀ A† O)
 *
 * @param state Amplification state
 * @return Error code
 */
qgt_error_t amp_apply_iteration(amp_state_t* state);

/**
 * @brief Apply the oracle operator O
 */
qgt_error_t amp_apply_oracle(amp_state_t* state);

/**
 * @brief Apply reflection about |0⟩: S₀ = 2|0⟩⟨0| - I
 */
qgt_error_t amp_apply_zero_reflection(amp_state_t* state);

/**
 * @brief Apply reflection about |ψ⟩: S_ψ = 2|ψ⟩⟨ψ| - I = A S₀ A†
 */
qgt_error_t amp_apply_state_reflection(amp_state_t* state);

/**
 * @brief Calculate optimal number of iterations given initial angle theta
 *
 * @param theta Initial angle (sin²(θ) = probability of good state)
 * @return Optimal number of iterations
 */
size_t amp_optimal_iterations(double theta);

/**
 * @brief Calculate optimal iterations given success probability
 *
 * @param initial_probability Probability of measuring good state initially
 * @return Optimal number of iterations
 */
size_t amp_iterations_from_probability(double initial_probability);

/**
 * @brief Clean up amplification state
 */
void amp_destroy_state(amp_state_t* state);

/**
 * @brief Clean up amplification result
 */
void amp_destroy_result(amp_result_t* result);

// ============================================================================
// Quantum Counting
// ============================================================================

/**
 * @brief Estimate the number of marked items using quantum counting
 *
 * Uses phase estimation on the Grover iterate to estimate sin²(θ),
 * which gives the fraction of marked states.
 *
 * @param state Amplification state (must have oracle set)
 * @param precision_bits Number of bits of precision
 * @param estimated_count Output: estimated number of marked items
 * @return Error code
 */
qgt_error_t amp_quantum_counting(amp_state_t* state, size_t precision_bits,
                                  double* estimated_count);

/**
 * @brief Estimate the initial angle theta
 *
 * @param state Amplification state
 * @param precision_bits Number of bits of precision
 * @param theta Output: estimated theta where sin²(θ) = initial success prob
 * @return Error code
 */
qgt_error_t amp_estimate_theta(amp_state_t* state, size_t precision_bits,
                                double* theta);

// ============================================================================
// Fixed-Point Amplitude Amplification
// ============================================================================

/**
 * @brief Run fixed-point amplitude amplification
 *
 * Fixed-point amplitude amplification converges to a fixed point regardless
 * of the number of iterations (no overshooting). Uses a sequence of
 * rotations with carefully chosen angles.
 *
 * Reference: Yoder, Low, Chuang "Fixed-Point Quantum Search..." (2014)
 *
 * @param state Amplification state
 * @param target_probability Minimum target probability
 * @return Result
 */
amp_result_t* amp_run_fixed_point(amp_state_t* state, double target_probability);

// ============================================================================
// Oblivious Amplitude Amplification
// ============================================================================

/**
 * @brief Block encoding for oblivious amplitude amplification
 */
typedef struct {
    ComplexFloat* block_matrix;     // Block encoding matrix
    size_t block_size;              // Size of each block
    size_t num_blocks;              // Number of blocks
} amp_block_encoding_t;

/**
 * @brief Run oblivious amplitude amplification
 *
 * Oblivious amplitude amplification works when the initial amplitude
 * is not known a priori, using a more robust protocol.
 *
 * @param state Amplification state
 * @param block Block encoding (optional)
 * @return Result
 */
amp_result_t* amp_run_oblivious(amp_state_t* state, const amp_block_encoding_t* block);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Measure the current probability of good states
 *
 * @param state Amplification state
 * @param is_good_state Function to identify good states
 * @param user_data User data for is_good_state
 * @return Probability of measuring a good state
 */
double amp_measure_good_probability(const amp_state_t* state,
                                     bool (*is_good_state)(size_t, void*),
                                     void* user_data);

/**
 * @brief Sample from the amplified state
 *
 * @param state Amplification state
 * @param num_samples Number of samples to take
 * @param samples Output array for sampled states
 * @return Error code
 */
qgt_error_t amp_sample(const amp_state_t* state, size_t num_samples, size_t* samples);

/**
 * @brief Print amplification result
 */
void amp_print_result(const amp_result_t* result);

#ifdef __cplusplus
}
#endif

#endif // AMPLITUDE_AMPLIFICATION_H
