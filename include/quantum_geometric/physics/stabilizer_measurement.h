#ifndef STABILIZER_MEASUREMENT_H
#define STABILIZER_MEASUREMENT_H

#include "quantum_geometric/physics/quantum_stabilizer.h"
#include "quantum_geometric/physics/stabilizer_types.h"
#include "quantum_geometric/hardware/quantum_ibm_backend.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include <stdbool.h>
#include <time.h>

// =============================================================================
// Single Measurement Result Structure (for low-level API)
// =============================================================================

// Measurement result structure
typedef struct {
    double expectation;           // Measured expectation value
    bool is_eigenstate;          // Whether state was eigenstate
    double eigenvalue;           // Eigenvalue if eigenstate
    double error_probability;    // Probability of measurement error
    void* auxiliary_data;        // Additional measurement data
} stabilizer_measurement_t;

// Initialize measurement structure
qgt_error_t measurement_create(stabilizer_measurement_t** measurement);

// Destroy measurement structure
void measurement_destroy(stabilizer_measurement_t* measurement);

// Perform stabilizer measurement
qgt_error_t measurement_perform(stabilizer_measurement_t* measurement,
                              const quantum_stabilizer_t* stabilizer,
                              const quantum_geometric_state_t* state);

// Check if measurement indicates error
qgt_error_t measurement_has_error(bool* has_error,
                                 const stabilizer_measurement_t* measurement,
                                 double threshold);

// Get measurement reliability
qgt_error_t measurement_reliability(double* reliability,
                                  const stabilizer_measurement_t* measurement);

// Compare two measurements
qgt_error_t measurement_compare(bool* equal,
                              const stabilizer_measurement_t* measurement1,
                              const stabilizer_measurement_t* measurement2,
                              double tolerance);

// Validate measurement result
qgt_error_t measurement_validate(const stabilizer_measurement_t* measurement);

// =============================================================================
// High-Level Stabilizer Measurement API (Lattice-Based)
// =============================================================================

/**
 * @brief Initialize stabilizer measurement state
 * @param state Stabilizer state to initialize
 * @param config Configuration parameters
 * @return true on success, false on failure
 */
bool init_stabilizer_measurement(StabilizerState* state,
                                const StabilizerConfig* config);

/**
 * @brief Initialize stabilizer measurement state with extended configuration
 *
 * Extended version that supports hardware configuration, parallel settings,
 * and reliability tracking.
 *
 * @param state Stabilizer state to initialize
 * @param config Extended configuration with hardware and performance settings
 * @return true on success, false on failure
 */
bool init_stabilizer_measurement_extended(StabilizerState* state,
                                         const StabilizerConfigExtended* config);

/**
 * @brief Clean up stabilizer measurement state
 * @param state Stabilizer state to clean up
 */
void cleanup_stabilizer_measurement(StabilizerState* state);

/**
 * @brief Perform stabilizer measurements on quantum state
 * @param state Stabilizer state
 * @param qstate Quantum state to measure
 * @return true on success, false on failure
 */
bool measure_stabilizers(StabilizerState* state, quantum_state_t* qstate);

/**
 * @brief Get stabilizer measurements of a specific type
 * @param state Stabilizer state
 * @param type Type of stabilizers (STABILIZER_PLAQUETTE or STABILIZER_VERTEX)
 * @param size Output parameter for array size
 * @return Pointer to measurements array, or NULL on error
 */
const double* get_stabilizer_measurements(const StabilizerState* state,
                                         StabilizerType type,
                                         size_t* size);

/**
 * @brief Get current error rate from stabilizer state
 * @param state Stabilizer state
 * @return Current error rate (0.0 to 1.0)
 *
 * Note: Named get_stabilizer_error_rate() to avoid conflict with
 * parallel_stabilizer's get_error_rate(size_t qubit_index).
 */
double get_stabilizer_error_rate(const StabilizerState* state);

/**
 * @brief Get the last syndrome measurement
 * @param state Stabilizer state
 * @param size Output parameter for syndrome size
 * @return Pointer to syndrome array, or NULL on error
 */
const double* get_last_syndrome(const StabilizerState* state, size_t* size);

// =============================================================================
// Pauli Measurement Helper Functions
// =============================================================================

/**
 * @brief Measure Pauli Z operator with confidence tracking
 * @param state Quantum state
 * @param x X coordinate
 * @param y Y coordinate
 * @param result Output measurement result
 * @param confidence Output confidence value
 * @return true on success
 */
bool measure_pauli_z_with_confidence(const quantum_state_t* state,
                                    size_t x, size_t y,
                                    double* result, double* confidence);

/**
 * @brief Measure Pauli X operator with confidence tracking
 * @param state Quantum state
 * @param x X coordinate
 * @param y Y coordinate
 * @param result Output measurement result
 * @param confidence Output confidence value
 * @return true on success
 */
bool measure_pauli_x_with_confidence(const quantum_state_t* state,
                                    size_t x, size_t y,
                                    double* result, double* confidence);

/**
 * @brief Get X-stabilizer correlation coefficient
 * @param state Quantum state
 * @param x X coordinate
 * @param y Y coordinate
 * @param qubit_idx Qubit index within stabilizer (0-3)
 * @return Correlation coefficient
 */
double get_x_stabilizer_correlation(const quantum_state_t* state,
                                   size_t x, size_t y, size_t qubit_idx);

/**
 * @brief Apply X-specific error mitigation sequence
 * @param state Quantum state
 * @param x X coordinate
 * @param y Y coordinate
 */
void apply_x_error_mitigation_sequence(const quantum_state_t* state,
                                      size_t x, size_t y);

/**
 * @brief Apply X measurement correction
 * @param state Quantum state
 * @param x X coordinate
 * @param y Y coordinate
 * @param result Measurement result to correct
 */
void apply_x_measurement_correction(const quantum_state_t* state,
                                   size_t x, size_t y, double* result);

// =============================================================================
// Hardware Profile Integration API
// =============================================================================

/**
 * @brief Set hardware profile for stabilizer measurements
 *
 * Configures the stabilizer measurement system to use calibration data
 * from the specified hardware profile. This enables cross-platform
 * hardware support by querying the appropriate backend for:
 * - Per-qubit readout fidelities
 * - T1/T2 coherence times
 * - Crosstalk matrix
 * - Gate fidelities
 *
 * @param state Stabilizer state to configure
 * @param profile Hardware profile with calibration data (may be NULL for defaults)
 * @return true on success, false on failure
 */
bool stabilizer_set_hardware_profile(StabilizerState* state,
                                     const HardwareProfile* profile);

/**
 * @brief Get the current hardware profile
 * @param state Stabilizer state
 * @return Pointer to hardware profile, or NULL if not set
 */
const HardwareProfile* stabilizer_get_hardware_profile(const StabilizerState* state);

/**
 * @brief Initialize stabilizer measurement with hardware profile
 *
 * Combined initialization that sets both the configuration and
 * hardware profile in one call. Preferred for production use.
 *
 * @param state Stabilizer state to initialize
 * @param config Configuration parameters
 * @param profile Hardware profile (may be NULL to use defaults)
 * @return true on success, false on failure
 */
bool init_stabilizer_measurement_with_hardware(StabilizerState* state,
                                               const StabilizerConfig* config,
                                               const HardwareProfile* profile);

/**
 * @brief Update hardware metrics from measurement results
 *
 * After measurements, this updates the hardware metrics tracking
 * in the stabilizer state based on observed performance.
 *
 * @param state Stabilizer state with recent measurements
 */
void stabilizer_update_hardware_metrics(StabilizerState* state);

/**
 * @brief Get hardware metrics from stabilizer state
 * @param state Stabilizer state
 * @return Pointer to hardware metrics, or NULL if not available
 */
const StabilizerHardwareMetrics* stabilizer_get_hardware_metrics(const StabilizerState* state);

/**
 * @brief Get resource metrics from stabilizer state
 * @param state Stabilizer state
 * @return Pointer to resource metrics, or NULL if not available
 */
const StabilizerResourceMetrics* stabilizer_get_resource_metrics(const StabilizerState* state);

/**
 * @brief Get reliability metrics from stabilizer state
 * @param state Stabilizer state
 * @return Pointer to reliability metrics, or NULL if not available
 */
const StabilizerReliabilityMetrics* stabilizer_get_reliability_metrics(const StabilizerState* state);

// =============================================================================
// Hardware-Aware Pauli Measurement Functions
// =============================================================================

/**
 * @brief Measure Pauli Z with hardware profile calibration
 *
 * Uses calibration data from the hardware profile for error mitigation:
 * - Per-qubit readout error rates from profile->measurement_fidelities
 * - T1 relaxation compensation from profile->t1_times
 * - Crosstalk correction from profile->crosstalk_matrix
 *
 * @param state Quantum state
 * @param x X coordinate
 * @param y Y coordinate
 * @param result Output measurement result
 * @param confidence Output confidence value
 * @param profile Hardware profile with calibration (NULL for defaults)
 * @return true on success
 */
bool measure_pauli_z_with_hardware(const quantum_state_t* state,
                                   size_t x, size_t y,
                                   double* result, double* confidence,
                                   const HardwareProfile* profile);

/**
 * @brief Measure Pauli X with hardware profile calibration
 *
 * Uses calibration data from the hardware profile for error mitigation:
 * - Per-qubit readout error rates from profile->measurement_fidelities
 * - T2 dephasing compensation from profile->t2_times
 * - Phase calibration from profile->phase_calibration
 * - Crosstalk correction from profile->crosstalk_matrix
 *
 * @param state Quantum state
 * @param x X coordinate
 * @param y Y coordinate
 * @param result Output measurement result
 * @param confidence Output confidence value
 * @param profile Hardware profile with calibration (NULL for defaults)
 * @return true on success
 */
bool measure_pauli_x_with_hardware(const quantum_state_t* state,
                                   size_t x, size_t y,
                                   double* result, double* confidence,
                                   const HardwareProfile* profile);

/**
 * @brief Get qubit readout errors from hardware profile
 *
 * Retrieves qubit-specific readout error rates from the hardware profile.
 * Falls back to position-dependent defaults if profile is NULL.
 *
 * @param profile Hardware profile (may be NULL)
 * @param qubit_idx Qubit index
 * @param p_0to1 Output: probability of measuring 1 given state 0
 * @param p_1to0 Output: probability of measuring 0 given state 1
 * @param lattice_width Lattice width for position-dependent defaults
 * @param lattice_height Lattice height for position-dependent defaults
 */
void get_hardware_readout_errors(const HardwareProfile* profile,
                                 size_t qubit_idx,
                                 double* p_0to1, double* p_1to0,
                                 size_t lattice_width, size_t lattice_height);

/**
 * @brief Get T1/T2 times from hardware profile
 *
 * @param profile Hardware profile (may be NULL)
 * @param qubit_idx Qubit index
 * @param t1 Output: T1 relaxation time in microseconds
 * @param t2 Output: T2 dephasing time in microseconds
 */
void get_hardware_coherence_times(const HardwareProfile* profile,
                                  size_t qubit_idx,
                                  double* t1, double* t2);

/**
 * @brief Get crosstalk coefficient between two qubits
 *
 * @param profile Hardware profile (may be NULL)
 * @param qubit_i First qubit index
 * @param qubit_j Second qubit index
 * @param num_qubits Total number of qubits
 * @return Crosstalk coefficient (0.0 if no crosstalk or profile unavailable)
 */
double get_hardware_crosstalk(const HardwareProfile* profile,
                              size_t qubit_i, size_t qubit_j,
                              size_t num_qubits);

#endif // STABILIZER_MEASUREMENT_H
