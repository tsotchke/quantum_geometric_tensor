/**
 * @file stabilizer_error_mitigation.h
 * @brief Header for error mitigation in stabilizer measurements with hardware optimization
 */

#ifndef STABILIZER_ERROR_MITIGATION_H
#define STABILIZER_ERROR_MITIGATION_H

#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/physics/quantum_state_operations.h"
#include <stdint.h>
#include <stdbool.h>

// Forward declarations
struct quantum_state;
struct HardwareProfile;

/**
 * @brief Configuration for error mitigation
 */
typedef struct {
    size_t num_qubits;           ///< Number of qubits in the system
    size_t history_length;       ///< Length of correction history to maintain
    uint64_t calibration_interval; ///< Time between calibration matrix updates (in ns)
} MitigationConfig;

/**
 * @brief Entry in correction history with confidence tracking
 */
typedef struct {
    bool was_successful;    ///< Whether correction was successful
    double confidence;      ///< Confidence level of correction
    double error_rate;      ///< Associated error rate
    uint64_t timestamp;     ///< When correction was applied
} CorrectionHistoryEntry;

/**
 * @brief Cache for error mitigation data
 */
typedef struct {
    size_t num_qubits;                  ///< Number of qubits
    size_t history_size;                ///< Size of history buffer
    double* error_rates;                ///< Per-qubit error rates
    double* error_variances;            ///< Error rate variances
    double* calibration_matrix;         ///< Calibration matrix elements
    double* confidence_weights;         ///< Per-qubit confidence weights
    CorrectionHistoryEntry* correction_history; ///< History of corrections
} MitigationCache;

/**
 * @brief State for error mitigation
 */
typedef struct {
    MitigationConfig config;     ///< Configuration parameters
    MitigationCache* cache;      ///< Mitigation data cache
    size_t total_corrections;    ///< Total corrections applied
    double success_rate;         ///< Overall success rate
    double confidence_level;     ///< Overall confidence level
    uint64_t last_update_time;   ///< Last calibration update time
} MitigationState;

/**
 * @brief Result of a measurement with confidence tracking
 */
typedef struct {
    double value;           ///< Measured value
    double confidence;      ///< Confidence in measurement
    double error_rate;      ///< Associated error rate
    size_t qubit_index;    ///< Index of measured qubit
    bool had_error;        ///< Whether error was detected
    uint8_t prepared_state; ///< State qubit was prepared in
    uint8_t measured_value; ///< Value that was measured
} measurement_result;

/**
 * @brief Initialize error mitigation state
 * 
 * @param state State to initialize
 * @param config Configuration parameters
 * @return true if initialization successful
 */
bool init_error_mitigation(MitigationState* state,
                          const MitigationConfig* config);

/**
 * @brief Clean up error mitigation state
 * 
 * @param state State to clean up
 */
void cleanup_error_mitigation(MitigationState* state);

/**
 * @brief Apply error mitigation to measurement results
 * 
 * @param state Mitigation state
 * @param qstate Quantum state to correct
 * @param results Measurement results
 * @param num_results Number of results
 * @param hw_profile Hardware profile for optimization
 * @return true if mitigation successful
 */
bool mitigate_measurement_errors(MitigationState* state,
                               struct quantum_state* qstate,
                               const measurement_result* results,
                               size_t num_results,
                               const struct HardwareProfile* hw_profile);

/**
 * @brief Update mitigation metrics
 * 
 * @param state State to update metrics for
 * @return true if update successful
 */
bool update_mitigation_metrics(MitigationState* state);

// Hardware profile access functions
double get_qubit_reliability(size_t qubit_index);
double get_measurement_fidelity(size_t qubit_index);
double get_coherence_factor(size_t qubit_index);
double get_noise_factor(void);
double get_measurement_stability(size_t qubit_index);
double get_hardware_reliability_factor(void);
double get_measurement_fidelity_for_thread(size_t thread_id);
double get_gate_fidelity_for_thread(size_t thread_id);
double get_noise_level_for_thread(size_t thread_id);
uint64_t get_current_timestamp(void);

#endif // STABILIZER_ERROR_MITIGATION_H
