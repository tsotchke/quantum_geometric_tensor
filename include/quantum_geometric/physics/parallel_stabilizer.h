/**
 * @file parallel_stabilizer.h
 * @brief Parallel stabilizer measurements with hardware optimization
 *
 * Provides thread-parallel measurement of stabilizer operators with:
 * - Adaptive workload balancing based on qubit reliability
 * - Hardware-aware measurement fidelity tracking
 * - Real-time feedback for dynamic adjustment
 * - Confidence-weighted measurement aggregation
 */

#ifndef PARALLEL_STABILIZER_H
#define PARALLEL_STABILIZER_H

#include <stdbool.h>
#include <stddef.h>
#include "quantum_geometric/core/quantum_types.h"
#include "quantum_geometric/hardware/quantum_hardware_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Type Aliases
// ============================================================================

// Alias for backward compatibility
#ifndef QUANTUM_STATE_PS_DEFINED
#define QUANTUM_STATE_PS_DEFINED
typedef quantum_state_t quantum_state;
#endif

// ============================================================================
// Enumerations
// ============================================================================

// StabilizerType and Pauli types (pauli_type_t, PauliOperator, PAULI_I/X/Y/Z)
// are defined in quantum_geometric/core/quantum_types.h which is included above

// ============================================================================
// Structures
// ============================================================================

/**
 * @brief Result of a single qubit measurement
 */
typedef struct {
    double value;              /**< Measurement outcome */
    double confidence;         /**< Confidence in measurement [0,1] */
    double error_rate;         /**< Estimated error rate */
    size_t qubit_index;        /**< Index of measured qubit */
    bool valid;                /**< Whether measurement succeeded */
} MeasurementResult;

/**
 * @brief Feedback for dynamic measurement adjustment
 */
typedef struct {
    double fidelity_adjustment;  /**< Multiplicative fidelity factor */
    double gate_adjustment;      /**< Gate fidelity adjustment */
    double error_scale;          /**< Error rate scaling factor */
    double confidence_threshold; /**< Threshold for re-measurement */
    bool requires_recalibration; /**< Whether hardware needs recalibration */
} MeasurementFeedback;

// HardwareProfile is defined in quantum_hardware_types.h (included above)
// This is the unified definition for all hardware profiling needs

/**
 * @brief Thread-local data for parallel measurements
 */
typedef struct {
    // Quantum state reference
    const quantum_state* state;       /**< Quantum state being measured */
    const size_t* qubit_indices;      /**< Qubit indices for measurement */

    // Thread workload
    size_t start_index;               /**< Start of assigned stabilizer range */
    size_t end_index;                 /**< End of assigned stabilizer range */
    size_t qubits_per_stabilizer;     /**< Number of qubits per stabilizer */

    // Stabilizer configuration
    StabilizerType type;              /**< Type of stabilizer being measured */

    // Measurement results
    double* results;                  /**< Measurement results array */
    double* confidences;              /**< Confidence values array */
    double* error_rates;              /**< Error rate estimates array */

    // Hardware state
    const HardwareProfile* hw_profile;/**< Hardware profile reference */
    double measurement_fidelity;      /**< Current measurement fidelity */
    double gate_fidelity;             /**< Current gate fidelity */
    double noise_level;               /**< Current noise level */

    // Thread identification
    size_t thread_id;                 /**< Thread identifier */
    size_t total_threads;             /**< Total number of threads */
} ThreadData;

// ============================================================================
// Core Functions
// ============================================================================

/**
 * @brief Measure stabilizers in parallel across multiple threads
 *
 * Distributes stabilizer measurements across threads with adaptive
 * workload balancing based on qubit reliability.
 *
 * @param state Quantum state to measure
 * @param qubit_indices Array of qubit indices for stabilizers
 * @param num_qubits Total number of qubits
 * @param type Type of stabilizer to measure
 * @param num_threads Number of parallel threads to use
 * @param results Output array for measurement results
 * @param hw_profile Hardware profile for optimization
 * @return true on success, false on failure
 */
bool measure_stabilizers_parallel(const quantum_state* state,
                                 const size_t* qubit_indices,
                                 size_t num_qubits,
                                 StabilizerType type,
                                 size_t num_threads,
                                 double* results,
                                 const HardwareProfile* hw_profile);

/**
 * @brief Measure a single stabilizer operator (serial/non-parallel)
 *
 * Measures the expectation value of a multi-qubit Pauli stabilizer:
 * - STABILIZER_PLAQUETTE (Z-type): Z⊗Z⊗Z⊗Z
 * - STABILIZER_VERTEX (X-type): X⊗X⊗X⊗X
 *
 * @param state Quantum state to measure
 * @param qubit_indices Array of qubit indices comprising the stabilizer
 * @param num_qubits Number of qubits in the stabilizer
 * @param type Type of stabilizer (PLAQUETTE or VERTEX)
 * @param result Output: expectation value in [-1, +1]
 * @return true on success, false on failure
 */
bool measure_stabilizer(const quantum_state* state,
                       const size_t* qubit_indices,
                       size_t num_qubits,
                       StabilizerType type,
                       double* result);

// ============================================================================
// Hardware Query Functions
// ============================================================================

/**
 * @brief Get reliability factor for a qubit (stabilizer-specific)
 * @param qubit_index Qubit index
 * @return Reliability factor [0,1]
 * @note Renamed to avoid conflict with error_correlation_hardware.c
 */
double get_stabilizer_qubit_reliability(size_t qubit_index);

/**
 * @brief Get measurement fidelity for a qubit (stabilizer-specific)
 * @param qubit_index Qubit index
 * @return Measurement fidelity [0,1]
 * @note Renamed to avoid conflict with error_prediction.c
 */
double get_stabilizer_measurement_fidelity(size_t qubit_index);

/**
 * @brief Get error rate for a qubit
 * @param qubit_index Qubit index
 * @return Error rate [0,1]
 */
double get_error_rate(size_t qubit_index);

/**
 * @brief Get base confidence for a qubit
 * @param qubit_index Qubit index
 * @return Base confidence level [0,1]
 */
double get_base_confidence(size_t qubit_index);

/**
 * @brief Get stability factor for a qubit
 * @param qubit_index Qubit index
 * @return Stability factor [0,1]
 */
double get_qubit_stability(size_t qubit_index);

/**
 * @brief Get coherence factor for a qubit
 * @param qubit_index Qubit index
 * @return Coherence factor [0,1]
 */
double get_coherence_factor(size_t qubit_index);

/**
 * @brief Get measurement fidelity for a specific thread
 * @param thread_id Thread identifier
 * @return Thread-adjusted measurement fidelity
 */
double get_measurement_fidelity_for_thread(size_t thread_id);

/**
 * @brief Get gate fidelity for a specific thread
 * @param thread_id Thread identifier
 * @return Thread-adjusted gate fidelity
 */
double get_gate_fidelity_for_thread(size_t thread_id);

/**
 * @brief Get noise level for a specific thread
 * @param thread_id Thread identifier
 * @return Thread-adjusted noise level
 */
double get_noise_level_for_thread(size_t thread_id);

/**
 * @brief Get aggregate error rate from measurements
 * @param measurements Array of measurement results
 * @param count Number of measurements
 * @return Aggregate error rate
 */
double get_aggregate_error_rate(const MeasurementResult* measurements, size_t count);

/**
 * @brief Check if stabilizer feedback should be triggered
 * @param result Measurement result to check
 * @return true if feedback needed
 * @note Renamed to avoid conflict with error_prediction.c (this takes MeasurementResult)
 */
bool should_trigger_stabilizer_feedback(const MeasurementResult* result);

/**
 * @brief Calculate fidelity adjustment from measurement
 * @param result Measurement result
 * @return Fidelity adjustment factor
 */
double calculate_fidelity_adjustment(const MeasurementResult* result);

/**
 * @brief Calculate gate adjustment from measurement
 * @param result Measurement result
 * @return Gate adjustment factor
 */
double calculate_gate_adjustment(const MeasurementResult* result);

/**
 * @brief Calculate error scaling from measurement
 * @param result Measurement result
 * @return Error scaling factor
 */
double calculate_error_scale(const MeasurementResult* result);

// ============================================================================
// Pauli Measurement Functions
// ============================================================================

/**
 * @brief Measure Pauli X expectation value
 * @param state Quantum state
 * @param qubit_index Qubit to measure
 * @param result Output measurement value
 * @return true on success
 */
bool measure_pauli_x(const quantum_state* state, size_t qubit_index, double* result);

/**
 * @brief Measure Pauli Y expectation value
 * @param state Quantum state
 * @param qubit_index Qubit to measure
 * @param result Output measurement value
 * @return true on success
 */
bool measure_pauli_y(const quantum_state* state, size_t qubit_index, double* result);

/**
 * @brief Measure Pauli Z expectation value
 * @param state Quantum state
 * @param qubit_index Qubit to measure
 * @param result Output measurement value
 * @return true on success
 */
bool measure_pauli_z(const quantum_state* state, size_t qubit_index, double* result);

#ifdef __cplusplus
}
#endif

#endif // PARALLEL_STABILIZER_H
