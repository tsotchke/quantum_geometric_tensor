/**
 * @file validation_analyzer.h
 * @brief Data Validation and Integrity Analysis for Quantum Operations
 *
 * Provides comprehensive validation analysis including:
 * - Quantum state validation (normalization, unitarity)
 * - Parameter bounds checking
 * - Data integrity verification
 * - Consistency checking across operations
 * - Input/output validation pipelines
 * - Schema-based validation
 *
 * Part of the QGTL Monitoring Framework.
 */

#ifndef VALIDATION_ANALYZER_H
#define VALIDATION_ANALYZER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Include complex types from the canonical source
#include "quantum_geometric/core/quantum_complex.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================

#define VALIDATION_MAX_NAME_LENGTH 128
#define VALIDATION_MAX_RULES 256
#define VALIDATION_MAX_ERRORS 1024
#define VALIDATION_MAX_CONTEXT_LENGTH 512

// Default tolerances
#define VALIDATION_DEFAULT_NORM_TOLERANCE 1e-10
#define VALIDATION_DEFAULT_UNITARITY_TOLERANCE 1e-10
#define VALIDATION_DEFAULT_HERMITIAN_TOLERANCE 1e-10
#define VALIDATION_DEFAULT_PROBABILITY_TOLERANCE 1e-12

// ============================================================================
// Enumerations
// ============================================================================

/**
 * Validation rule types
 */
typedef enum {
    VALIDATION_RULE_NORMALIZATION,        // State normalization |ψ|² = 1
    VALIDATION_RULE_UNITARITY,            // Unitary matrix U†U = I
    VALIDATION_RULE_HERMITICITY,          // Hermitian matrix H† = H
    VALIDATION_RULE_POSITIVE_SEMIDEFINITE,// ρ ≥ 0 for density matrices
    VALIDATION_RULE_TRACE_ONE,            // Tr(ρ) = 1 for density matrices
    VALIDATION_RULE_PROBABILITY_SUM,      // Σpᵢ = 1
    VALIDATION_RULE_PROBABILITY_BOUNDS,   // 0 ≤ pᵢ ≤ 1
    VALIDATION_RULE_DIMENSION_MATCH,      // Matrix/vector dimensions
    VALIDATION_RULE_QUBIT_COUNT,          // Valid qubit count (power of 2)
    VALIDATION_RULE_PARAMETER_BOUNDS,     // Parameter within [min, max]
    VALIDATION_RULE_FINITE_VALUES,        // No NaN or Inf
    VALIDATION_RULE_NON_NEGATIVE,         // Value ≥ 0
    VALIDATION_RULE_POSITIVE,             // Value > 0
    VALIDATION_RULE_INTEGER,              // Integer value
    VALIDATION_RULE_CUSTOM,               // User-defined validation
    VALIDATION_RULE_COUNT
} validation_rule_type_t;

/**
 * Validation severity levels
 */
typedef enum {
    VALIDATION_SEVERITY_INFO,             // Informational only
    VALIDATION_SEVERITY_WARNING,          // Non-critical issue
    VALIDATION_SEVERITY_ERROR,            // Critical issue
    VALIDATION_SEVERITY_FATAL             // Unrecoverable error
} validation_severity_t;

/**
 * Validation result status
 */
typedef enum {
    VALIDATION_STATUS_PASS,               // All validations passed
    VALIDATION_STATUS_PASS_WITH_WARNINGS, // Passed with warnings
    VALIDATION_STATUS_FAIL,               // One or more failures
    VALIDATION_STATUS_SKIPPED,            // Validation skipped
    VALIDATION_STATUS_ERROR               // Validation error occurred
} validation_status_t;

/**
 * Data types for validation
 */
typedef enum {
    VALIDATION_DATA_STATE_VECTOR,         // Complex state vector
    VALIDATION_DATA_DENSITY_MATRIX,       // Density matrix
    VALIDATION_DATA_UNITARY_MATRIX,       // Unitary operator
    VALIDATION_DATA_HERMITIAN_MATRIX,     // Hermitian operator
    VALIDATION_DATA_PROBABILITY_DIST,     // Probability distribution
    VALIDATION_DATA_REAL_VECTOR,          // Real-valued vector
    VALIDATION_DATA_REAL_MATRIX,          // Real-valued matrix
    VALIDATION_DATA_PARAMETER,            // Single parameter
    VALIDATION_DATA_CIRCUIT,              // Quantum circuit
    VALIDATION_DATA_CUSTOM                // Custom data type
} validation_data_type_t;

// ============================================================================
// Data Structures
// ============================================================================

/**
 * Validation error/warning entry
 */
typedef struct {
    validation_rule_type_t rule;
    validation_severity_t severity;
    char message[256];
    char context[VALIDATION_MAX_CONTEXT_LENGTH];
    double expected_value;
    double actual_value;
    double tolerance;
    uint64_t timestamp_ns;
} validation_error_t;

/**
 * Validation rule configuration
 */
typedef struct {
    validation_rule_type_t type;
    char name[VALIDATION_MAX_NAME_LENGTH];
    bool enabled;
    validation_severity_t severity_on_fail;
    double tolerance;
    double min_bound;
    double max_bound;
    bool (*custom_validator)(const void* data, size_t size, char* error_msg);
} validation_rule_t;

/**
 * Validation result for a single check
 */
typedef struct {
    validation_rule_type_t rule;
    validation_status_t status;
    double computed_value;
    double expected_value;
    double deviation;
    char details[256];
} validation_check_result_t;

/**
 * Comprehensive validation result
 */
typedef struct {
    validation_status_t overall_status;
    size_t total_checks;
    size_t passed_checks;
    size_t warning_checks;
    size_t failed_checks;
    size_t skipped_checks;
    validation_error_t* errors;
    size_t error_count;
    uint64_t validation_time_ns;
} validation_result_t;

/**
 * Quantum state validation parameters
 */
typedef struct {
    double norm_tolerance;
    double phase_tolerance;
    bool check_sparsity;
    double sparsity_threshold;
    bool allow_subnormal;
} state_validation_params_t;

/**
 * Matrix validation parameters
 */
typedef struct {
    double unitarity_tolerance;
    double hermitian_tolerance;
    double positive_tolerance;
    double trace_tolerance;
    bool check_eigenvalues;
    bool check_condition_number;
    double max_condition_number;
} matrix_validation_params_t;

/**
 * Validation statistics
 */
typedef struct {
    uint64_t total_validations;
    uint64_t total_passes;
    uint64_t total_warnings;
    uint64_t total_failures;
    uint64_t total_errors;
    double avg_validation_time_ns;
    double max_validation_time_ns;
    uint64_t rule_pass_counts[VALIDATION_RULE_COUNT];
    uint64_t rule_fail_counts[VALIDATION_RULE_COUNT];
} validation_stats_t;

/**
 * Analyzer configuration
 */
typedef struct {
    bool strict_mode;                     // Fail on any warning
    bool collect_statistics;              // Collect validation stats
    bool cache_results;                   // Cache recent results
    size_t max_cached_results;            // Max cached results
    size_t max_errors_per_validation;     // Max errors to collect
    double default_tolerance;             // Default numerical tolerance
    bool validate_inputs_only;            // Skip output validation
    bool validate_outputs_only;           // Skip input validation
    bool log_all_validations;             // Log even successful validations
} validation_analyzer_config_t;

/**
 * Opaque analyzer handle
 */
typedef struct validation_analyzer validation_analyzer_t;

// ============================================================================
// Initialization and Configuration
// ============================================================================

/**
 * Create validation analyzer with default configuration
 */
validation_analyzer_t* validation_analyzer_create(void);

/**
 * Create validation analyzer with custom configuration
 */
validation_analyzer_t* validation_analyzer_create_with_config(
    const validation_analyzer_config_t* config);

/**
 * Get default configuration
 */
validation_analyzer_config_t validation_analyzer_default_config(void);

/**
 * Destroy validation analyzer
 */
void validation_analyzer_destroy(validation_analyzer_t* analyzer);

/**
 * Reset all statistics and cached results
 */
bool validation_analyzer_reset(validation_analyzer_t* analyzer);

// ============================================================================
// Rule Management
// ============================================================================

/**
 * Add a validation rule
 */
bool validation_add_rule(validation_analyzer_t* analyzer,
                         const validation_rule_t* rule);

/**
 * Remove a validation rule by name
 */
bool validation_remove_rule(validation_analyzer_t* analyzer,
                            const char* name);

/**
 * Enable/disable a rule
 */
bool validation_set_rule_enabled(validation_analyzer_t* analyzer,
                                  const char* name,
                                  bool enabled);

/**
 * Set tolerance for a rule
 */
bool validation_set_rule_tolerance(validation_analyzer_t* analyzer,
                                    const char* name,
                                    double tolerance);

/**
 * Get all configured rules
 */
bool validation_get_rules(validation_analyzer_t* analyzer,
                          validation_rule_t** rules,
                          size_t* count);

// ============================================================================
// Quantum State Validation
// ============================================================================

/**
 * Validate quantum state vector normalization
 */
validation_result_t validation_check_state_norm(
    validation_analyzer_t* analyzer,
    const ComplexDouble* state,
    size_t dim,
    double tolerance);

/**
 * Validate complete state vector
 */
validation_result_t validation_check_state_vector(
    validation_analyzer_t* analyzer,
    const ComplexDouble* state,
    size_t dim,
    const state_validation_params_t* params);

/**
 * Validate density matrix
 */
validation_result_t validation_check_density_matrix(
    validation_analyzer_t* analyzer,
    const ComplexDouble* rho,
    size_t dim,
    const matrix_validation_params_t* params);

/**
 * Validate unitary matrix
 */
validation_result_t validation_check_unitary(
    validation_analyzer_t* analyzer,
    const ComplexDouble* U,
    size_t dim,
    double tolerance);

/**
 * Validate Hermitian matrix
 */
validation_result_t validation_check_hermitian(
    validation_analyzer_t* analyzer,
    const ComplexDouble* H,
    size_t dim,
    double tolerance);

// ============================================================================
// Probability and Measurement Validation
// ============================================================================

/**
 * Validate probability distribution
 */
validation_result_t validation_check_probability_distribution(
    validation_analyzer_t* analyzer,
    const double* probs,
    size_t count,
    double tolerance);

/**
 * Validate measurement outcomes
 */
validation_result_t validation_check_measurement_outcomes(
    validation_analyzer_t* analyzer,
    const uint64_t* counts,
    size_t num_outcomes,
    uint64_t total_shots);

/**
 * Validate POVM elements (positive, sum to identity)
 */
validation_result_t validation_check_povm(
    validation_analyzer_t* analyzer,
    const ComplexDouble** elements,
    size_t num_elements,
    size_t dim,
    double tolerance);

// ============================================================================
// Parameter Validation
// ============================================================================

/**
 * Validate parameter within bounds
 */
validation_result_t validation_check_parameter_bounds(
    validation_analyzer_t* analyzer,
    const char* param_name,
    double value,
    double min_val,
    double max_val);

/**
 * Validate angle parameter (typically [0, 2π] or [-π, π])
 */
validation_result_t validation_check_angle(
    validation_analyzer_t* analyzer,
    double angle,
    double min_angle,
    double max_angle);

/**
 * Validate qubit index
 */
validation_result_t validation_check_qubit_index(
    validation_analyzer_t* analyzer,
    size_t qubit_index,
    size_t num_qubits);

/**
 * Validate dimension is power of 2
 */
validation_result_t validation_check_dimension_power_of_2(
    validation_analyzer_t* analyzer,
    size_t dim);

// ============================================================================
// Data Integrity Validation
// ============================================================================

/**
 * Check for NaN or Inf values in real array
 */
validation_result_t validation_check_finite_real(
    validation_analyzer_t* analyzer,
    const double* data,
    size_t count);

/**
 * Check for NaN or Inf values in complex array
 */
validation_result_t validation_check_finite_complex(
    validation_analyzer_t* analyzer,
    const ComplexDouble* data,
    size_t count);

/**
 * Validate matrix dimensions match
 */
validation_result_t validation_check_dimensions(
    validation_analyzer_t* analyzer,
    size_t rows_a, size_t cols_a,
    size_t rows_b, size_t cols_b,
    const char* operation);

/**
 * Validate data checksum (for integrity)
 */
validation_result_t validation_check_checksum(
    validation_analyzer_t* analyzer,
    const void* data,
    size_t size,
    uint64_t expected_checksum);

// ============================================================================
// Batch Validation
// ============================================================================

/**
 * Validate multiple items with same rules
 */
validation_result_t validation_batch_validate(
    validation_analyzer_t* analyzer,
    validation_data_type_t data_type,
    const void** data_items,
    const size_t* sizes,
    size_t count);

/**
 * Create validation pipeline
 */
typedef struct validation_pipeline validation_pipeline_t;

validation_pipeline_t* validation_pipeline_create(
    validation_analyzer_t* analyzer);

void validation_pipeline_destroy(validation_pipeline_t* pipeline);

bool validation_pipeline_add_stage(
    validation_pipeline_t* pipeline,
    validation_rule_type_t rule,
    double tolerance);

validation_result_t validation_pipeline_execute(
    validation_pipeline_t* pipeline,
    const void* data,
    size_t size,
    validation_data_type_t type);

// ============================================================================
// Statistics and Reporting
// ============================================================================

/**
 * Get validation statistics
 */
bool validation_get_statistics(validation_analyzer_t* analyzer,
                                validation_stats_t* stats);

/**
 * Get most common validation failures
 */
bool validation_get_common_failures(
    validation_analyzer_t* analyzer,
    validation_rule_type_t** rules,
    uint64_t** counts,
    size_t* num_rules);

/**
 * Generate validation report
 */
char* validation_generate_report(validation_analyzer_t* analyzer);

/**
 * Export validation history to JSON
 */
char* validation_export_json(validation_analyzer_t* analyzer);

/**
 * Export to file
 */
bool validation_export_to_file(validation_analyzer_t* analyzer,
                                const char* filename);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Get rule type name
 */
const char* validation_rule_name(validation_rule_type_t rule);

/**
 * Get severity name
 */
const char* validation_severity_name(validation_severity_t severity);

/**
 * Get status name
 */
const char* validation_status_name(validation_status_t status);

/**
 * Free validation result resources
 */
void validation_result_free(validation_result_t* result);

/**
 * Free allocated rules array
 */
void validation_free_rules(validation_rule_t* rules, size_t count);

/**
 * Get default state validation parameters
 */
state_validation_params_t validation_default_state_params(void);

/**
 * Get default matrix validation parameters
 */
matrix_validation_params_t validation_default_matrix_params(void);

/**
 * Get last error message
 */
const char* validation_get_last_error(validation_analyzer_t* analyzer);

#ifdef __cplusplus
}
#endif

#endif // VALIDATION_ANALYZER_H
