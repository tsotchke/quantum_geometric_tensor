/**
 * @file quantum_phase_estimation_gpu.h
 * @brief GPU-Accelerated Quantum Phase Estimation
 *
 * Provides GPU-accelerated phase estimation including:
 * - Standard quantum phase estimation (QPE)
 * - Iterative phase estimation (IPE)
 * - Robust phase estimation
 * - Eigenvalue extraction
 * - Controlled unitary operations
 * - Circuit optimization for GPU
 *
 * Part of the QGTL Hardware Acceleration Framework.
 */

#ifndef QUANTUM_PHASE_ESTIMATION_GPU_H
#define QUANTUM_PHASE_ESTIMATION_GPU_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================

#define QPE_MAX_PRECISION_QUBITS 32
#define QPE_MAX_SYSTEM_QUBITS 64
#define QPE_MAX_NAME_LENGTH 128
#define QPE_DEFAULT_SHOTS 1000

// ============================================================================
// Enumerations
// ============================================================================

/**
 * Phase estimation algorithms
 */
typedef enum {
    QPE_ALGORITHM_STANDARD,           // Standard QPE with QFT
    QPE_ALGORITHM_ITERATIVE,          // Iterative phase estimation
    QPE_ALGORITHM_KITAEV,             // Kitaev's algorithm
    QPE_ALGORITHM_ROBUST,             // Robust phase estimation
    QPE_ALGORITHM_BAYESIAN,           // Bayesian phase estimation
    QPE_ALGORITHM_VARIATIONAL         // Variational phase estimation
} qpe_algorithm_t;

/**
 * Unitary representation
 */
typedef enum {
    QPE_UNITARY_MATRIX,               // Dense matrix
    QPE_UNITARY_SPARSE,               // Sparse matrix
    QPE_UNITARY_CIRCUIT,              // Circuit description
    QPE_UNITARY_HAMILTONIAN           // Hamiltonian (exp(-iHt))
} qpe_unitary_type_t;

/**
 * Eigenstate selection
 */
typedef enum {
    QPE_SELECT_GROUND,                // Ground state (lowest eigenvalue)
    QPE_SELECT_EXCITED,               // First excited state
    QPE_SELECT_SPECIFIED,             // User-specified eigenstate
    QPE_SELECT_ALL                    // All eigenstates
} qpe_eigenstate_select_t;

/**
 * Result quality
 */
typedef enum {
    QPE_RESULT_EXACT,                 // Exact (numerical)
    QPE_RESULT_STATISTICAL,           // Statistical estimate
    QPE_RESULT_APPROXIMATE,           // Approximate
    QPE_RESULT_FAILED                 // Failed to estimate
} qpe_result_quality_t;

// ============================================================================
// Complex Type (for C compatibility)
// ============================================================================

#ifndef COMPLEX_FLOAT_DEFINED
typedef struct {
    double real;
    double imag;
} ComplexDouble;
#endif

// ============================================================================
// Data Structures
// ============================================================================

/**
 * Unitary operator specification
 */
typedef struct {
    qpe_unitary_type_t type;
    size_t dimension;                 // 2^n for n qubits
    size_t num_qubits;
    ComplexDouble* matrix;            // Dense matrix (if applicable)
    void* sparse_data;                // Sparse representation
    void* circuit_data;               // Circuit description
    double time_parameter;            // For Hamiltonian simulation
} qpe_unitary_t;

/**
 * Phase estimation parameters
 */
typedef struct {
    qpe_algorithm_t algorithm;
    size_t precision_qubits;          // Number of precision qubits
    size_t num_shots;                 // Number of measurement shots
    double confidence_level;          // Confidence level (0-1)
    bool use_inverse_qft;             // Use inverse QFT
    bool optimize_circuit;            // Optimize for GPU
    size_t max_iterations;            // For iterative methods
    double convergence_threshold;     // For iterative methods
} qpe_params_t;

/**
 * Single phase result
 */
typedef struct {
    double phase;                     // Estimated phase [0, 2π)
    double eigenvalue_real;           // Real part of eigenvalue
    double eigenvalue_imag;           // Imag part of eigenvalue
    double probability;               // Probability of this outcome
    double uncertainty;               // Uncertainty in phase
    uint64_t measurement_count;       // Times this phase was measured
    ComplexDouble* eigenstate;        // Corresponding eigenstate (optional)
    size_t eigenstate_dim;
} qpe_phase_result_t;

/**
 * Complete QPE results
 */
typedef struct {
    qpe_result_quality_t quality;
    size_t num_phases;                // Number of phases found
    qpe_phase_result_t* phases;       // Array of phase results
    double dominant_phase;            // Most probable phase
    double phase_gap;                 // Gap between phases (if applicable)
    uint64_t total_shots;
    uint64_t successful_shots;
    double success_rate;
    uint64_t execution_time_ns;
    char algorithm_used[QPE_MAX_NAME_LENGTH];
} qpe_result_t;

/**
 * QPE circuit statistics
 */
typedef struct {
    size_t total_qubits;
    size_t precision_qubits;
    size_t system_qubits;
    size_t ancilla_qubits;
    size_t gate_count;
    size_t controlled_u_count;
    size_t circuit_depth;
    size_t two_qubit_gate_count;
    double estimated_fidelity;
} qpe_circuit_stats_t;

/**
 * Iterative estimation state
 */
typedef struct {
    size_t current_iteration;
    double current_phase_estimate;
    double current_uncertainty;
    double* phase_history;
    size_t history_length;
    bool converged;
} qpe_iteration_state_t;

/**
 * Context configuration
 */
typedef struct {
    int device_id;                    // GPU device
    size_t workspace_size;            // Workspace memory
    bool enable_profiling;            // Enable timing
    bool enable_error_mitigation;     // Error mitigation
    double noise_model_strength;      // Noise model parameter
} qpe_context_config_t;

/**
 * Opaque QPE context
 */
typedef struct qpe_context qpe_context_t;

// ============================================================================
// Initialization
// ============================================================================

/**
 * Create QPE context
 */
qpe_context_t* qpe_context_create(void);

/**
 * Create with configuration
 */
qpe_context_t* qpe_context_create_with_config(
    const qpe_context_config_t* config);

/**
 * Get default configuration
 */
qpe_context_config_t qpe_default_config(void);

/**
 * Get default parameters
 */
qpe_params_t qpe_default_params(void);

/**
 * Destroy context
 */
void qpe_context_destroy(qpe_context_t* ctx);

// ============================================================================
// Unitary Operations
// ============================================================================

/**
 * Create unitary from dense matrix
 */
qpe_unitary_t* qpe_unitary_from_matrix(
    qpe_context_t* ctx,
    const ComplexDouble* matrix,
    size_t dimension);

/**
 * Create unitary from Hamiltonian
 */
qpe_unitary_t* qpe_unitary_from_hamiltonian(
    qpe_context_t* ctx,
    const ComplexDouble* hamiltonian,
    size_t dimension,
    double time);

/**
 * Create controlled-U^(2^k)
 */
qpe_unitary_t* qpe_create_controlled_power(
    qpe_context_t* ctx,
    const qpe_unitary_t* U,
    size_t power);

/**
 * Destroy unitary
 */
void qpe_unitary_destroy(qpe_unitary_t* unitary);

/**
 * Verify unitary property
 */
bool qpe_verify_unitary(
    qpe_context_t* ctx,
    const qpe_unitary_t* U,
    double tolerance);

// ============================================================================
// Standard Phase Estimation
// ============================================================================

/**
 * Run standard quantum phase estimation
 */
qpe_result_t* qpe_estimate_phase(
    qpe_context_t* ctx,
    const qpe_unitary_t* U,
    const ComplexDouble* initial_state,
    size_t state_dim,
    const qpe_params_t* params);

/**
 * Run QPE with eigenstate selection
 */
qpe_result_t* qpe_estimate_eigenvalue(
    qpe_context_t* ctx,
    const qpe_unitary_t* U,
    qpe_eigenstate_select_t selection,
    const qpe_params_t* params);

/**
 * Extract all eigenvalues
 */
qpe_result_t* qpe_extract_all_eigenvalues(
    qpe_context_t* ctx,
    const qpe_unitary_t* U,
    const qpe_params_t* params);

// ============================================================================
// Iterative Phase Estimation
// ============================================================================

/**
 * Initialize iterative estimation
 */
qpe_iteration_state_t* qpe_iterative_init(
    qpe_context_t* ctx,
    const qpe_unitary_t* U,
    const ComplexDouble* initial_state,
    size_t state_dim);

/**
 * Perform single iteration
 */
bool qpe_iterative_step(
    qpe_context_t* ctx,
    qpe_iteration_state_t* state,
    const qpe_unitary_t* U);

/**
 * Run full iterative estimation
 */
qpe_result_t* qpe_iterative_run(
    qpe_context_t* ctx,
    const qpe_unitary_t* U,
    const ComplexDouble* initial_state,
    size_t state_dim,
    size_t max_iterations,
    double tolerance);

/**
 * Get iteration state
 */
bool qpe_iterative_get_state(
    const qpe_iteration_state_t* state,
    double* current_phase,
    double* uncertainty);

/**
 * Destroy iteration state
 */
void qpe_iterative_destroy(qpe_iteration_state_t* state);

// ============================================================================
// Robust Phase Estimation
// ============================================================================

/**
 * Run robust phase estimation
 */
qpe_result_t* qpe_robust_estimate(
    qpe_context_t* ctx,
    const qpe_unitary_t* U,
    const ComplexDouble* initial_state,
    size_t state_dim,
    double target_precision,
    double success_probability);

/**
 * Run Bayesian phase estimation
 */
qpe_result_t* qpe_bayesian_estimate(
    qpe_context_t* ctx,
    const qpe_unitary_t* U,
    const ComplexDouble* initial_state,
    size_t state_dim,
    const double* prior_distribution,
    size_t prior_bins,
    size_t num_experiments);

// ============================================================================
// Circuit Analysis
// ============================================================================

/**
 * Get circuit statistics
 */
bool qpe_get_circuit_stats(
    qpe_context_t* ctx,
    const qpe_unitary_t* U,
    const qpe_params_t* params,
    qpe_circuit_stats_t* stats);

/**
 * Estimate circuit fidelity
 */
double qpe_estimate_fidelity(
    qpe_context_t* ctx,
    const qpe_unitary_t* U,
    const qpe_params_t* params,
    double gate_error_rate);

/**
 * Optimize circuit for GPU
 */
bool qpe_optimize_circuit(
    qpe_context_t* ctx,
    const qpe_unitary_t* U,
    qpe_params_t* params);

// ============================================================================
// Results Management
// ============================================================================

/**
 * Free QPE result
 */
void qpe_result_free(qpe_result_t* result);

/**
 * Get phase from result
 */
double qpe_result_get_phase(
    const qpe_result_t* result,
    size_t index);

/**
 * Get dominant eigenvalue
 */
bool qpe_result_get_dominant_eigenvalue(
    const qpe_result_t* result,
    double* eigenvalue_real,
    double* eigenvalue_imag);

/**
 * Get eigenstate
 */
bool qpe_result_get_eigenstate(
    const qpe_result_t* result,
    size_t index,
    ComplexDouble** eigenstate,
    size_t* dimension);

/**
 * Verify result against known eigenvalue
 */
bool qpe_result_verify(
    const qpe_result_t* result,
    double known_phase,
    double tolerance,
    double* error);

// ============================================================================
// Reporting
// ============================================================================

/**
 * Generate QPE report
 */
char* qpe_generate_report(const qpe_result_t* result);

/**
 * Export result to JSON
 */
char* qpe_export_json(const qpe_result_t* result);

/**
 * Export to file
 */
bool qpe_export_to_file(
    const qpe_result_t* result,
    const char* filename);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Get algorithm name
 */
const char* qpe_algorithm_name(qpe_algorithm_t algorithm);

/**
 * Get unitary type name
 */
const char* qpe_unitary_type_name(qpe_unitary_type_t type);

/**
 * Get result quality name
 */
const char* qpe_result_quality_name(qpe_result_quality_t quality);

/**
 * Convert phase to eigenvalue
 */
void qpe_phase_to_eigenvalue(
    double phase,
    double* eigenvalue_real,
    double* eigenvalue_imag);

/**
 * Convert eigenvalue to phase
 */
double qpe_eigenvalue_to_phase(
    double eigenvalue_real,
    double eigenvalue_imag);

/**
 * Estimate required precision qubits
 */
size_t qpe_estimate_precision_qubits(
    double target_precision,
    double success_probability);

/**
 * Get last error message
 */
const char* qpe_get_last_error(qpe_context_t* ctx);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_PHASE_ESTIMATION_GPU_H
