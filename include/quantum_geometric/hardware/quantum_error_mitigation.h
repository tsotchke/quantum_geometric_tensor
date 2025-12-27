/**
 * @file quantum_error_mitigation.h
 * @brief Production-grade quantum error mitigation system
 *
 * Implements multiple error mitigation strategies:
 * - Zero-noise extrapolation (ZNE)
 * - Probabilistic error cancellation (PEC)
 * - Symmetry verification
 * - Dynamic error rate adaptation
 * - Distributed error tracking
 */

#ifndef QUANTUM_ERROR_MITIGATION_H
#define QUANTUM_ERROR_MITIGATION_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <pthread.h>

#include <math.h>

#include "quantum_geometric/core/quantum_types.h"
#include "quantum_geometric/core/quantum_circuit_types.h"
#include "quantum_geometric/hardware/quantum_hardware_types.h"

// Note: We do NOT include quantum_hardware_abstraction.h here to avoid
// type conflicts with HardwareOptimizations defined there

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================

#define MAX_EXTRAPOLATION_POINTS 5
#define MAX_MITIGATION_RETRIES 3
#define CONFIDENCE_THRESHOLD 0.95
#define MIN_NOISE_SCALE 1.0
#define MAX_NOISE_SCALE 3.0
#define MAX_ERROR_HISTORY 1000

// Rigetti hardware error rates (empirical values)
#define RIGETTI_RX_ERROR 0.001
#define RIGETTI_RZ_ERROR 0.0001
#define RIGETTI_CZ_ERROR 0.01

// ============================================================================
// Type Definitions
// ============================================================================

/**
 * @brief Hardware-specific error rates
 */
typedef struct HardwareErrorRates {
    double single_qubit_error;    // Single-qubit gate error rate
    double two_qubit_error;       // Two-qubit gate error rate
    double measurement_error;     // Measurement error rate
    double t1_time;               // Relaxation time (us)
    double t2_time;               // Dephasing time (us)
    double current_fidelity;      // Current estimated fidelity
    double noise_scale;           // Noise scaling factor for mitigation
} HardwareErrorRates;

/**
 * @brief Error mitigation configuration
 */
typedef struct ErrorMitigationConfig {
    double error_threshold;           // Error threshold for acceptance
    double confidence_threshold;      // Minimum confidence level
    size_t max_retries;               // Maximum retry attempts
    bool use_distributed_tracking;    // Enable distributed error tracking
    bool dynamic_adaptation;          // Enable dynamic error rate adaptation
    size_t num_shots;                 // Number of shots per circuit
    double symmetry_threshold;        // Threshold for symmetry verification
} ErrorMitigationConfig;

/**
 * @brief Error tracking statistics
 */
typedef struct ErrorTrackingStats {
    double total_error;               // Cumulative error
    size_t error_count;               // Number of error measurements
    double error_variance;            // Error variance
    double confidence_level;          // Current confidence level
    double* error_history;            // History of error measurements
    size_t history_size;              // Current history size
    size_t history_capacity;          // Maximum history capacity
} ErrorTrackingStats;

/**
 * @brief Extrapolation data for ZNE
 */
typedef struct ExtrapolationData {
    double* noise_levels;             // Noise scale factors
    double* measurements;             // Measured values at each noise level
    double* uncertainties;            // Uncertainties at each noise level
    size_t num_points;                // Number of data points
    double confidence;                // Fit confidence
} ExtrapolationData;

/**
 * @brief Hardware context for error mitigation
 * Note: Named MitigationHardwareContext to avoid conflict with
 * HardwareOptimizations in quantum_hardware_abstraction.h
 */
typedef struct MitigationHardwareContext {
    BackendType backend_type;         // Type of quantum backend
    HardwareErrorRates error_rates;   // Hardware error rates
    bool calibration_valid;           // Whether calibration is current
    uint64_t last_calibration;        // Timestamp of last calibration
    void* backend_specific;           // Backend-specific optimization data
} MitigationHardwareContext;

/**
 * @brief Distributed error tracking configuration
 */
typedef struct DistributedConfig {
    uint32_t sync_interval;           // Sync interval in milliseconds
    uint32_t sync_timeout;            // Sync timeout in milliseconds
    bool auto_sync;                   // Enable automatic synchronization
    size_t max_retries;               // Maximum retry attempts
    double error_threshold;           // Error threshold for sync
    size_t min_responses;             // Minimum responses for consensus
} DistributedConfig;

/**
 * @brief Error statistics message for distributed tracking
 */
typedef struct ErrorStatsMessage {
    double total_error;               // Total accumulated error
    size_t error_count;               // Number of error measurements
    double error_variance;            // Error variance
    double confidence_level;          // Confidence level
    double latest_error;              // Most recent error measurement
    uint64_t timestamp;               // Message timestamp
    uint32_t node_id;                 // Originating node ID
} ErrorStatsMessage;

/**
 * @brief Rigetti backend configuration for error mitigation
 * Note: Named RigettiMitigationConfig to avoid conflict with
 * RigettiConfig in quantum_backend_types.h (API connection config)
 */
typedef struct RigettiMitigationConfig {
    double symmetry_threshold;        // Threshold for symmetry verification
    size_t num_shots;                 // Number of shots per circuit
    double* native_gate_errors;       // Native gate error rates
    size_t num_native_gates;          // Number of native gates
    bool use_parametric_compilation;  // Enable parametric compilation
    bool use_active_reset;            // Enable active qubit reset
} RigettiMitigationConfig;

/**
 * @brief Backend abstraction for error mitigation
 * Note: Named MitigationBackend to avoid conflict with other backend types
 */
typedef struct MitigationBackend {
    BackendType type;                 // Backend type
    void* handle;                     // Backend handle
    bool connected;                   // Connection status
    HardwareErrorRates* error_rates;  // Current error rates
} MitigationBackend;

/**
 * @brief Circuit execution result for error mitigation
 * Note: Named MitigationResult to avoid conflict with other result types
 */
typedef struct MitigationResult {
    double expectation_value;         // Expectation value
    double* probabilities;            // Measurement probabilities
    size_t num_measurements;          // Number of measurements
    double fidelity;                  // Estimated fidelity
    double error_rate;                // Estimated error rate
} MitigationResult;

// ============================================================================
// Initialization and Cleanup
// ============================================================================

/**
 * @brief Initialize error mitigation system
 * @param backend_type Backend type string ("ibm", "rigetti", "ionq", "dwave")
 * @param num_qubits Number of qubits in the system
 * @param distributed_mode Enable distributed error tracking
 * @return Error mitigation configuration or NULL on failure
 */
ErrorMitigationConfig* init_error_mitigation(
    const char* backend_type,
    size_t num_qubits,
    bool distributed_mode);

/**
 * @brief Clean up error mitigation resources
 * @param config Error mitigation configuration to clean up
 */
void cleanup_error_mitigation(ErrorMitigationConfig* config);

/**
 * @brief Initialize mitigation hardware context
 * @param backend_type Backend type string
 * @return Mitigation hardware context or NULL on failure
 */
MitigationHardwareContext* init_mitigation_hardware_context(const char* backend_type);

/**
 * @brief Clean up mitigation hardware context
 * @param ctx Mitigation hardware context to clean up
 */
void cleanup_mitigation_hardware_context(MitigationHardwareContext* ctx);

/**
 * @brief Initialize distributed error tracking
 * @param backend_type Backend type string
 * @param config Distributed configuration
 * @return 0 on success, non-zero on failure
 */
int init_distributed_error_tracking(const char* backend_type,
                                    const DistributedConfig* config);

/**
 * @brief Clean up distributed error tracking
 */
void cleanup_distributed_error_tracking(void);

// ============================================================================
// Extrapolation Data Management
// ============================================================================

/**
 * @brief Initialize extrapolation data structure
 * @return Extrapolation data or NULL on failure
 */
ExtrapolationData* init_extrapolation(void);

/**
 * @brief Clean up extrapolation data
 * @param data Extrapolation data to clean up
 */
void cleanup_extrapolation_data(ExtrapolationData* data);

// ============================================================================
// Error Mitigation Methods
// ============================================================================

/**
 * @brief Zero-noise extrapolation
 * @param circuit Quantum circuit to execute
 * @param backend Mitigation backend
 * @param config Mitigation configuration
 * @param uncertainty Output uncertainty estimate
 * @return Extrapolated expectation value
 */
double zero_noise_extrapolation(
    const quantum_circuit* circuit,
    const MitigationBackend* backend,
    const RigettiMitigationConfig* config,
    double* uncertainty);

/**
 * @brief Probabilistic error cancellation
 * @param circuit Quantum circuit to execute
 * @param backend Mitigation backend
 * @param config Mitigation configuration
 * @param uncertainty Output uncertainty estimate
 * @return Mitigated expectation value
 */
double probabilistic_error_cancellation(
    const quantum_circuit* circuit,
    const MitigationBackend* backend,
    const RigettiMitigationConfig* config,
    double* uncertainty);

/**
 * @brief Symmetry verification
 * @param circuit Quantum circuit to execute
 * @param backend Mitigation backend
 * @param config Mitigation configuration
 * @param uncertainty Output uncertainty estimate
 * @return Verified expectation value
 */
double symmetry_verification(
    const quantum_circuit* circuit,
    const MitigationBackend* backend,
    const RigettiMitigationConfig* config,
    double* uncertainty);

// ============================================================================
// Error Tracking and Adaptation
// ============================================================================

/**
 * @brief Adapt error rates based on tracking statistics
 * @param rates Hardware error rates to update
 * @param stats Error tracking statistics
 * @param config Error mitigation configuration
 */
void adapt_error_rates(
    HardwareErrorRates* rates,
    const ErrorTrackingStats* stats,
    const ErrorMitigationConfig* config);

/**
 * @brief Update error tracking with new measurement
 * @param stats Error tracking statistics to update
 * @param measured_error Measured error value
 * @param config Error mitigation configuration
 */
void update_error_tracking(
    ErrorTrackingStats* stats,
    double measured_error,
    const ErrorMitigationConfig* config);

/**
 * @brief Get current error statistics
 * @param backend Mitigation backend
 * @param config Error mitigation configuration
 * @return Copy of current error statistics or NULL on failure
 */
ErrorTrackingStats* get_error_stats(
    const MitigationBackend* backend,
    const ErrorMitigationConfig* config);

/**
 * @brief Broadcast error statistics to distributed nodes
 * @param stats Error statistics to broadcast
 */
void broadcast_error_stats(const ErrorTrackingStats* stats);

/**
 * @brief Broadcast message to all nodes
 * @param msg Message to broadcast
 * @param size Message size
 */
void broadcast_to_nodes(const void* msg, size_t size);

// ============================================================================
// Circuit Operations for Error Mitigation
// ============================================================================

/**
 * @brief Copy a circuit for error mitigation
 * @param circuit Circuit to copy
 * @return Copy of the circuit or NULL on failure
 */
quantum_circuit* copy_circuit_for_mitigation(const quantum_circuit* circuit);

/**
 * @brief Clean up a mitigation circuit
 * @param circuit Circuit to clean up
 */
void cleanup_mitigation_circuit(quantum_circuit* circuit);

/**
 * @brief Submit a circuit for mitigated execution
 * @param backend Mitigation backend
 * @param circuit Circuit to execute
 * @param result Output result
 * @return 0 on success, non-zero on failure
 */
int submit_mitigated_circuit(
    const MitigationBackend* backend,
    const quantum_circuit* circuit,
    MitigationResult* result);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_ERROR_MITIGATION_H
