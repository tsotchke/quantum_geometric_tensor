/**
 * @file quantum_error_correction.h
 * @brief Production-grade quantum error correction system
 *
 * Implements error correction using:
 * - Hierarchical syndrome measurement
 * - Distributed error correction
 * - GPU-accelerated correction operations
 * - Decoherence handling
 */

#ifndef QUANTUM_ERROR_CORRECTION_H
#define QUANTUM_ERROR_CORRECTION_H

#include <complex.h>
#include <stdbool.h>
#include <stddef.h>

#include "quantum_geometric/core/hierarchical_matrix.h"
#include "quantum_geometric/hardware/quantum_geometric_gpu.h"
#include "quantum_geometric/config/mpi_config.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================

#define QEC_DEFAULT_THRESHOLD 1e-6
#define QEC_LEAF_SIZE 256
#define QEC_MAX_SYNDROME_SIZE 4096

// ============================================================================
// Error Correction Types
// ============================================================================

/**
 * @brief Fast approximation structure for syndrome measurement
 */
typedef struct FastApproximation {
    double complex* coefficients;
    size_t num_terms;
    double threshold;
    bool* active_terms;
} FastApproximation;

/**
 * @brief Boundary data for hierarchical error correction
 */
typedef struct BoundaryData {
    double complex* values;
    size_t size;
    double threshold;
} BoundaryData;

/**
 * @brief Decoherence correction parameters
 */
typedef struct DecoherenceParams {
    double t1;          // Relaxation time (microseconds)
    double t2;          // Dephasing time (microseconds)
    double dt;          // Time step (microseconds)
    double temperature; // Temperature (Kelvin)
} DecoherenceParams;

/**
 * @brief Error correction configuration
 */
typedef struct ErrorCorrectionConfig {
    double syndrome_threshold;      // Threshold for syndrome detection
    double correction_threshold;    // Threshold for applying corrections
    size_t max_iterations;          // Maximum correction iterations
    bool use_gpu;                   // Use GPU acceleration
    bool use_distributed;           // Use distributed correction
    DecoherenceParams decoherence;  // Decoherence parameters
} ErrorCorrectionConfig;

/**
 * @brief Error correction statistics
 */
typedef struct ErrorCorrectionStats {
    size_t syndromes_measured;      // Number of syndromes measured
    size_t corrections_applied;     // Number of corrections applied
    double total_correction_time;   // Total time spent in corrections
    double average_syndrome_value;  // Average syndrome magnitude
    double error_rate;              // Estimated error rate
} ErrorCorrectionStats;

// ============================================================================
// Core Error Correction Functions
// ============================================================================

/**
 * @brief Initialize error correction system
 */
ErrorCorrectionConfig* init_error_correction(bool use_gpu, bool use_distributed);

/**
 * @brief Cleanup error correction resources
 */
void cleanup_error_correction(ErrorCorrectionConfig* config);

/**
 * @brief Measure error syndromes
 * @param syndromes Output array for syndrome values
 * @param state Quantum state to measure
 * @param n State dimension
 */
void measure_syndromes(double complex* syndromes,
                      const double complex* state,
                      size_t n);

/**
 * @brief Apply quantum error correction
 * @param state State to correct (modified in place)
 * @param syndromes Measured syndromes
 * @param n State dimension
 */
void quantum_error_correct(double complex* state,
                          const double complex* syndromes,
                          size_t n);

/**
 * @brief Measure local error syndrome
 * @param syndrome Output syndrome value
 * @param state Local state portion
 * @param n Local state dimension
 */
void measure_error_syndrome(double complex* syndrome,
                           const double complex* state,
                           size_t n);

// ============================================================================
// Decoherence Handling
// ============================================================================

/**
 * @brief Handle decoherence effects
 * @param state State to process (modified in place)
 * @param n State dimension
 */
void handle_decoherence(double complex* state, size_t n);

/**
 * @brief Apply amplitude damping correction
 * @param state State amplitude
 * @param t1 Relaxation time
 * @param dt Time step
 * @return Corrected state amplitude
 */
double complex apply_amplitude_damping(double complex state,
                                       double t1,
                                       double dt);

/**
 * @brief Apply phase damping correction
 * @param state State amplitude
 * @param t2 Dephasing time
 * @param dt Time step
 * @return Corrected state amplitude
 */
double complex apply_phase_damping(double complex state,
                                   double t2,
                                   double dt);

// ============================================================================
// State Validation
// ============================================================================

/**
 * @brief Validate quantum state
 * @param state State to validate
 * @param n State dimension
 * @return true if state is valid
 */
bool validate_quantum_state(const double complex* state, size_t n);

// ============================================================================
// Distributed Error Correction
// ============================================================================

/**
 * @brief Distribute workload across nodes
 * @param n Total state dimension
 * @return Local workload size
 */
size_t distribute_workload(size_t n);

/**
 * @brief Get local offset for distributed work
 * @return Offset into global state
 */
size_t get_local_offset(void);

// ============================================================================
// Statistics and Monitoring
// ============================================================================

/**
 * @brief Get error correction statistics
 * @return Current statistics
 */
ErrorCorrectionStats* get_error_correction_stats(void);

/**
 * @brief Reset error correction statistics
 */
void reset_error_correction_stats(void);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_ERROR_CORRECTION_H
