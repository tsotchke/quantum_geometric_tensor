/**
 * @file validation_analyzer.c
 * @brief Implementation of Data Validation and Integrity Analysis
 */

#include "quantum_geometric/core/validation_analyzer.h"
#include "quantum_geometric/core/quantum_complex.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>
#include <time.h>

#ifdef __APPLE__
#include <mach/mach_time.h>
#endif

// ============================================================================
// Internal Structures
// ============================================================================

struct validation_analyzer {
    validation_analyzer_config_t config;
    validation_rule_t* rules;
    size_t num_rules;
    size_t rules_capacity;
    validation_stats_t stats;
    pthread_mutex_t mutex;
    char last_error[512];
    bool initialized;
};

struct validation_pipeline {
    validation_analyzer_t* analyzer;
    validation_rule_type_t* stages;
    double* tolerances;
    size_t num_stages;
    size_t stages_capacity;
};

// ============================================================================
// Platform-Specific Timing
// ============================================================================

static uint64_t get_timestamp_ns(void) {
#ifdef __APPLE__
    static mach_timebase_info_data_t timebase = {0};
    if (timebase.denom == 0) {
        mach_timebase_info(&timebase);
    }
    return mach_absolute_time() * timebase.numer / timebase.denom;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

// ============================================================================
// Helper Functions
// ============================================================================

static void set_error(validation_analyzer_t* analyzer, const char* msg) {
    if (analyzer && msg) {
        snprintf(analyzer->last_error, sizeof(analyzer->last_error), "%s", msg);
    }
}

static validation_result_t create_empty_result(void) {
    validation_result_t result;
    memset(&result, 0, sizeof(result));
    result.overall_status = VALIDATION_STATUS_PASS;
    return result;
}

static void add_error_to_result(validation_result_t* result,
                                 validation_rule_type_t rule,
                                 validation_severity_t severity,
                                 const char* message,
                                 double expected,
                                 double actual,
                                 double tolerance) {
    if (!result) return;

    if (result->errors == NULL) {
        result->errors = calloc(VALIDATION_MAX_ERRORS, sizeof(validation_error_t));
        if (!result->errors) return;
    }

    if (result->error_count >= VALIDATION_MAX_ERRORS) return;

    validation_error_t* err = &result->errors[result->error_count];
    err->rule = rule;
    err->severity = severity;
    snprintf(err->message, sizeof(err->message), "%s", message ? message : "");
    err->expected_value = expected;
    err->actual_value = actual;
    err->tolerance = tolerance;
    err->timestamp_ns = get_timestamp_ns();

    result->error_count++;

    if (severity == VALIDATION_SEVERITY_ERROR ||
        severity == VALIDATION_SEVERITY_FATAL) {
        result->failed_checks++;
        result->overall_status = VALIDATION_STATUS_FAIL;
    } else if (severity == VALIDATION_SEVERITY_WARNING) {
        result->warning_checks++;
        if (result->overall_status == VALIDATION_STATUS_PASS) {
            result->overall_status = VALIDATION_STATUS_PASS_WITH_WARNINGS;
        }
    }
}

static bool is_power_of_2(size_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

// ============================================================================
// Initialization and Configuration
// ============================================================================

validation_analyzer_config_t validation_analyzer_default_config(void) {
    validation_analyzer_config_t config;
    memset(&config, 0, sizeof(config));
    config.strict_mode = false;
    config.collect_statistics = true;
    config.cache_results = false;
    config.max_cached_results = 100;
    config.max_errors_per_validation = 100;
    config.default_tolerance = 1e-10;
    config.validate_inputs_only = false;
    config.validate_outputs_only = false;
    config.log_all_validations = false;
    return config;
}

validation_analyzer_t* validation_analyzer_create(void) {
    validation_analyzer_config_t config = validation_analyzer_default_config();
    return validation_analyzer_create_with_config(&config);
}

validation_analyzer_t* validation_analyzer_create_with_config(
    const validation_analyzer_config_t* config) {
    if (!config) return NULL;

    validation_analyzer_t* analyzer = calloc(1, sizeof(validation_analyzer_t));
    if (!analyzer) return NULL;

    analyzer->config = *config;
    analyzer->rules_capacity = 32;
    analyzer->rules = calloc(analyzer->rules_capacity, sizeof(validation_rule_t));
    if (!analyzer->rules) {
        free(analyzer);
        return NULL;
    }

    pthread_mutex_init(&analyzer->mutex, NULL);
    analyzer->initialized = true;

    return analyzer;
}

void validation_analyzer_destroy(validation_analyzer_t* analyzer) {
    if (!analyzer) return;

    pthread_mutex_lock(&analyzer->mutex);

    free(analyzer->rules);
    analyzer->rules = NULL;
    analyzer->initialized = false;

    pthread_mutex_unlock(&analyzer->mutex);
    pthread_mutex_destroy(&analyzer->mutex);

    free(analyzer);
}

bool validation_analyzer_reset(validation_analyzer_t* analyzer) {
    if (!analyzer || !analyzer->initialized) return false;

    pthread_mutex_lock(&analyzer->mutex);
    memset(&analyzer->stats, 0, sizeof(analyzer->stats));
    pthread_mutex_unlock(&analyzer->mutex);

    return true;
}

// ============================================================================
// Rule Management
// ============================================================================

bool validation_add_rule(validation_analyzer_t* analyzer,
                         const validation_rule_t* rule) {
    if (!analyzer || !analyzer->initialized || !rule) return false;

    pthread_mutex_lock(&analyzer->mutex);

    if (analyzer->num_rules >= analyzer->rules_capacity) {
        size_t new_capacity = analyzer->rules_capacity * 2;
        validation_rule_t* new_rules = realloc(analyzer->rules,
            new_capacity * sizeof(validation_rule_t));
        if (!new_rules) {
            pthread_mutex_unlock(&analyzer->mutex);
            set_error(analyzer, "Failed to expand rules array");
            return false;
        }
        analyzer->rules = new_rules;
        analyzer->rules_capacity = new_capacity;
    }

    analyzer->rules[analyzer->num_rules++] = *rule;

    pthread_mutex_unlock(&analyzer->mutex);
    return true;
}

bool validation_remove_rule(validation_analyzer_t* analyzer,
                            const char* name) {
    if (!analyzer || !analyzer->initialized || !name) return false;

    pthread_mutex_lock(&analyzer->mutex);

    for (size_t i = 0; i < analyzer->num_rules; i++) {
        if (strcmp(analyzer->rules[i].name, name) == 0) {
            memmove(&analyzer->rules[i], &analyzer->rules[i + 1],
                    (analyzer->num_rules - i - 1) * sizeof(validation_rule_t));
            analyzer->num_rules--;
            pthread_mutex_unlock(&analyzer->mutex);
            return true;
        }
    }

    pthread_mutex_unlock(&analyzer->mutex);
    set_error(analyzer, "Rule not found");
    return false;
}

bool validation_set_rule_enabled(validation_analyzer_t* analyzer,
                                  const char* name,
                                  bool enabled) {
    if (!analyzer || !analyzer->initialized || !name) return false;

    pthread_mutex_lock(&analyzer->mutex);

    for (size_t i = 0; i < analyzer->num_rules; i++) {
        if (strcmp(analyzer->rules[i].name, name) == 0) {
            analyzer->rules[i].enabled = enabled;
            pthread_mutex_unlock(&analyzer->mutex);
            return true;
        }
    }

    pthread_mutex_unlock(&analyzer->mutex);
    return false;
}

bool validation_set_rule_tolerance(validation_analyzer_t* analyzer,
                                    const char* name,
                                    double tolerance) {
    if (!analyzer || !analyzer->initialized || !name) return false;

    pthread_mutex_lock(&analyzer->mutex);

    for (size_t i = 0; i < analyzer->num_rules; i++) {
        if (strcmp(analyzer->rules[i].name, name) == 0) {
            analyzer->rules[i].tolerance = tolerance;
            pthread_mutex_unlock(&analyzer->mutex);
            return true;
        }
    }

    pthread_mutex_unlock(&analyzer->mutex);
    return false;
}

bool validation_get_rules(validation_analyzer_t* analyzer,
                          validation_rule_t** rules,
                          size_t* count) {
    if (!analyzer || !analyzer->initialized || !rules || !count) return false;

    pthread_mutex_lock(&analyzer->mutex);

    *rules = calloc(analyzer->num_rules, sizeof(validation_rule_t));
    if (!*rules) {
        pthread_mutex_unlock(&analyzer->mutex);
        return false;
    }

    memcpy(*rules, analyzer->rules,
           analyzer->num_rules * sizeof(validation_rule_t));
    *count = analyzer->num_rules;

    pthread_mutex_unlock(&analyzer->mutex);
    return true;
}

// ============================================================================
// Quantum State Validation
// ============================================================================

validation_result_t validation_check_state_norm(
    validation_analyzer_t* analyzer,
    const ComplexDouble* state,
    size_t dim,
    double tolerance) {

    validation_result_t result = create_empty_result();
    uint64_t start = get_timestamp_ns();

    if (!state || dim == 0) {
        add_error_to_result(&result, VALIDATION_RULE_NORMALIZATION,
                           VALIDATION_SEVERITY_FATAL,
                           "Invalid state vector (NULL or zero dimension)",
                           1.0, 0.0, tolerance);
        result.validation_time_ns = get_timestamp_ns() - start;
        return result;
    }

    result.total_checks = 1;

    double norm_squared = 0.0;
    for (size_t i = 0; i < dim; i++) {
        norm_squared += complex_double_abs_squared(state[i]);
    }

    double norm = sqrt(norm_squared);
    double deviation = fabs(norm - 1.0);

    if (deviation > tolerance) {
        add_error_to_result(&result, VALIDATION_RULE_NORMALIZATION,
                           VALIDATION_SEVERITY_ERROR,
                           "State vector not normalized",
                           1.0, norm, tolerance);
    } else {
        result.passed_checks = 1;
    }

    result.validation_time_ns = get_timestamp_ns() - start;

    if (analyzer && analyzer->initialized) {
        pthread_mutex_lock(&analyzer->mutex);
        analyzer->stats.total_validations++;
        if (result.overall_status == VALIDATION_STATUS_PASS) {
            analyzer->stats.total_passes++;
            analyzer->stats.rule_pass_counts[VALIDATION_RULE_NORMALIZATION]++;
        } else {
            analyzer->stats.total_failures++;
            analyzer->stats.rule_fail_counts[VALIDATION_RULE_NORMALIZATION]++;
        }
        pthread_mutex_unlock(&analyzer->mutex);
    }

    return result;
}

validation_result_t validation_check_state_vector(
    validation_analyzer_t* analyzer,
    const ComplexDouble* state,
    size_t dim,
    const state_validation_params_t* params) {

    validation_result_t result = create_empty_result();
    uint64_t start = get_timestamp_ns();

    state_validation_params_t default_params = validation_default_state_params();
    if (!params) params = &default_params;

    if (!state || dim == 0) {
        add_error_to_result(&result, VALIDATION_RULE_FINITE_VALUES,
                           VALIDATION_SEVERITY_FATAL,
                           "Invalid state vector", 0, 0, 0);
        result.validation_time_ns = get_timestamp_ns() - start;
        return result;
    }

    result.total_checks = 3;

    // Check 1: Finite values
    bool has_invalid = false;
    for (size_t i = 0; i < dim; i++) {
        if (!isfinite(state[i].real) || !isfinite(state[i].imag)) {
            has_invalid = true;
            break;
        }
    }

    if (has_invalid) {
        add_error_to_result(&result, VALIDATION_RULE_FINITE_VALUES,
                           VALIDATION_SEVERITY_ERROR,
                           "State contains NaN or Inf values", 0, 0, 0);
    } else {
        result.passed_checks++;
    }

    // Check 2: Dimension is power of 2
    if (!is_power_of_2(dim)) {
        add_error_to_result(&result, VALIDATION_RULE_QUBIT_COUNT,
                           VALIDATION_SEVERITY_WARNING,
                           "Dimension is not a power of 2", 0, (double)dim, 0);
    } else {
        result.passed_checks++;
    }

    // Check 3: Normalization
    double norm_squared = 0.0;
    for (size_t i = 0; i < dim; i++) {
        norm_squared += complex_double_abs_squared(state[i]);
    }

    double norm = sqrt(norm_squared);
    if (fabs(norm - 1.0) > params->norm_tolerance) {
        add_error_to_result(&result, VALIDATION_RULE_NORMALIZATION,
                           VALIDATION_SEVERITY_ERROR,
                           "State not normalized",
                           1.0, norm, params->norm_tolerance);
    } else {
        result.passed_checks++;
    }

    result.validation_time_ns = get_timestamp_ns() - start;
    return result;
}

validation_result_t validation_check_density_matrix(
    validation_analyzer_t* analyzer,
    const ComplexDouble* rho,
    size_t dim,
    const matrix_validation_params_t* params) {

    validation_result_t result = create_empty_result();
    uint64_t start = get_timestamp_ns();

    matrix_validation_params_t default_params = validation_default_matrix_params();
    if (!params) params = &default_params;

    if (!rho || dim == 0) {
        add_error_to_result(&result, VALIDATION_RULE_FINITE_VALUES,
                           VALIDATION_SEVERITY_FATAL,
                           "Invalid density matrix", 0, 0, 0);
        result.validation_time_ns = get_timestamp_ns() - start;
        return result;
    }

    result.total_checks = 3;

    // Check 1: Hermitian (ρ = ρ†)
    bool is_hermitian = true;
    double max_deviation = 0.0;
    for (size_t i = 0; i < dim && is_hermitian; i++) {
        for (size_t j = i; j < dim; j++) {
            ComplexDouble rho_ij = rho[i * dim + j];
            ComplexDouble rho_ji = rho[j * dim + i];
            ComplexDouble rho_ji_conj = complex_double_conjugate(rho_ji);
            ComplexDouble diff = complex_double_subtract(rho_ij, rho_ji_conj);
            double dev = complex_double_abs(diff);
            if (dev > max_deviation) max_deviation = dev;
            if (dev > params->hermitian_tolerance) {
                is_hermitian = false;
                break;
            }
        }
    }

    if (!is_hermitian) {
        add_error_to_result(&result, VALIDATION_RULE_HERMITICITY,
                           VALIDATION_SEVERITY_ERROR,
                           "Density matrix is not Hermitian",
                           0.0, max_deviation, params->hermitian_tolerance);
    } else {
        result.passed_checks++;
    }

    // Check 2: Trace = 1
    double trace_real = 0.0;
    for (size_t i = 0; i < dim; i++) {
        trace_real += rho[i * dim + i].real;
    }

    double trace_deviation = fabs(trace_real - 1.0);
    if (trace_deviation > params->trace_tolerance) {
        add_error_to_result(&result, VALIDATION_RULE_TRACE_ONE,
                           VALIDATION_SEVERITY_ERROR,
                           "Trace of density matrix is not 1",
                           1.0, trace_real, params->trace_tolerance);
    } else {
        result.passed_checks++;
    }

    // Check 3: Positive semidefinite (check diagonal elements >= 0)
    bool is_positive = true;
    for (size_t i = 0; i < dim; i++) {
        double diag = rho[i * dim + i].real;
        if (diag < -params->positive_tolerance) {
            is_positive = false;
            add_error_to_result(&result, VALIDATION_RULE_POSITIVE_SEMIDEFINITE,
                               VALIDATION_SEVERITY_ERROR,
                               "Density matrix has negative diagonal element",
                               0.0, diag, params->positive_tolerance);
            break;
        }
    }

    if (is_positive) {
        result.passed_checks++;
    }

    result.validation_time_ns = get_timestamp_ns() - start;
    return result;
}

validation_result_t validation_check_unitary(
    validation_analyzer_t* analyzer,
    const ComplexDouble* U,
    size_t dim,
    double tolerance) {

    validation_result_t result = create_empty_result();
    uint64_t start = get_timestamp_ns();

    if (!U || dim == 0) {
        add_error_to_result(&result, VALIDATION_RULE_UNITARITY,
                           VALIDATION_SEVERITY_FATAL,
                           "Invalid unitary matrix", 0, 0, tolerance);
        result.validation_time_ns = get_timestamp_ns() - start;
        return result;
    }

    result.total_checks = 1;

    // Check U†U = I
    double max_deviation = 0.0;

    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            ComplexDouble sum = {0.0, 0.0};
            for (size_t k = 0; k < dim; k++) {
                // U†[i,k] * U[k,j] = conj(U[k,i]) * U[k,j]
                ComplexDouble U_ki_conj = complex_double_conjugate(U[k * dim + i]);
                ComplexDouble product = complex_double_multiply(U_ki_conj, U[k * dim + j]);
                sum = complex_double_add(sum, product);
            }

            double expected = (i == j) ? 1.0 : 0.0;
            ComplexDouble expected_c = {expected, 0.0};
            ComplexDouble diff = complex_double_subtract(sum, expected_c);
            double deviation = complex_double_abs(diff);
            if (deviation > max_deviation) {
                max_deviation = deviation;
            }
        }
    }

    if (max_deviation > tolerance) {
        add_error_to_result(&result, VALIDATION_RULE_UNITARITY,
                           VALIDATION_SEVERITY_ERROR,
                           "Matrix is not unitary (U†U ≠ I)",
                           0.0, max_deviation, tolerance);
    } else {
        result.passed_checks = 1;
    }

    result.validation_time_ns = get_timestamp_ns() - start;
    return result;
}

validation_result_t validation_check_hermitian(
    validation_analyzer_t* analyzer,
    const ComplexDouble* H,
    size_t dim,
    double tolerance) {

    validation_result_t result = create_empty_result();
    uint64_t start = get_timestamp_ns();

    if (!H || dim == 0) {
        add_error_to_result(&result, VALIDATION_RULE_HERMITICITY,
                           VALIDATION_SEVERITY_FATAL,
                           "Invalid Hermitian matrix", 0, 0, tolerance);
        result.validation_time_ns = get_timestamp_ns() - start;
        return result;
    }

    result.total_checks = 1;

    double max_deviation = 0.0;

    for (size_t i = 0; i < dim; i++) {
        for (size_t j = i; j < dim; j++) {
            ComplexDouble H_ij = H[i * dim + j];
            ComplexDouble H_ji = H[j * dim + i];
            ComplexDouble H_ji_conj = complex_double_conjugate(H_ji);
            ComplexDouble diff = complex_double_subtract(H_ij, H_ji_conj);
            double deviation = complex_double_abs(diff);
            if (deviation > max_deviation) {
                max_deviation = deviation;
            }
        }
    }

    if (max_deviation > tolerance) {
        add_error_to_result(&result, VALIDATION_RULE_HERMITICITY,
                           VALIDATION_SEVERITY_ERROR,
                           "Matrix is not Hermitian (H ≠ H†)",
                           0.0, max_deviation, tolerance);
    } else {
        result.passed_checks = 1;
    }

    result.validation_time_ns = get_timestamp_ns() - start;
    return result;
}

// ============================================================================
// Probability and Measurement Validation
// ============================================================================

validation_result_t validation_check_probability_distribution(
    validation_analyzer_t* analyzer,
    const double* probs,
    size_t count,
    double tolerance) {

    validation_result_t result = create_empty_result();
    uint64_t start = get_timestamp_ns();

    if (!probs || count == 0) {
        add_error_to_result(&result, VALIDATION_RULE_PROBABILITY_SUM,
                           VALIDATION_SEVERITY_FATAL,
                           "Invalid probability distribution", 1.0, 0.0, tolerance);
        result.validation_time_ns = get_timestamp_ns() - start;
        return result;
    }

    result.total_checks = 2;

    // Check 1: All probabilities in [0, 1]
    bool all_valid = true;
    for (size_t i = 0; i < count; i++) {
        if (probs[i] < -tolerance || probs[i] > 1.0 + tolerance) {
            all_valid = false;
            add_error_to_result(&result, VALIDATION_RULE_PROBABILITY_BOUNDS,
                               VALIDATION_SEVERITY_ERROR,
                               "Probability outside [0, 1]",
                               0.5, probs[i], tolerance);
            break;
        }
    }

    if (all_valid) {
        result.passed_checks++;
    }

    // Check 2: Sum = 1
    double sum = 0.0;
    for (size_t i = 0; i < count; i++) {
        sum += probs[i];
    }

    if (fabs(sum - 1.0) > tolerance) {
        add_error_to_result(&result, VALIDATION_RULE_PROBABILITY_SUM,
                           VALIDATION_SEVERITY_ERROR,
                           "Probabilities do not sum to 1",
                           1.0, sum, tolerance);
    } else {
        result.passed_checks++;
    }

    result.validation_time_ns = get_timestamp_ns() - start;
    return result;
}

validation_result_t validation_check_measurement_outcomes(
    validation_analyzer_t* analyzer,
    const uint64_t* counts,
    size_t num_outcomes,
    uint64_t total_shots) {

    validation_result_t result = create_empty_result();
    uint64_t start = get_timestamp_ns();

    if (!counts || num_outcomes == 0) {
        add_error_to_result(&result, VALIDATION_RULE_FINITE_VALUES,
                           VALIDATION_SEVERITY_FATAL,
                           "Invalid measurement outcomes", 0, 0, 0);
        result.validation_time_ns = get_timestamp_ns() - start;
        return result;
    }

    result.total_checks = 1;

    uint64_t sum = 0;
    for (size_t i = 0; i < num_outcomes; i++) {
        sum += counts[i];
    }

    if (sum != total_shots) {
        add_error_to_result(&result, VALIDATION_RULE_PROBABILITY_SUM,
                           VALIDATION_SEVERITY_ERROR,
                           "Measurement counts don't match total shots",
                           (double)total_shots, (double)sum, 0);
    } else {
        result.passed_checks = 1;
    }

    result.validation_time_ns = get_timestamp_ns() - start;
    return result;
}

validation_result_t validation_check_povm(
    validation_analyzer_t* analyzer,
    const ComplexDouble** elements,
    size_t num_elements,
    size_t dim,
    double tolerance) {

    validation_result_t result = create_empty_result();
    uint64_t start = get_timestamp_ns();

    if (!elements || num_elements == 0 || dim == 0) {
        add_error_to_result(&result, VALIDATION_RULE_FINITE_VALUES,
                           VALIDATION_SEVERITY_FATAL,
                           "Invalid POVM elements", 0, 0, tolerance);
        result.validation_time_ns = get_timestamp_ns() - start;
        return result;
    }

    result.total_checks = 2;

    // Check 1: Each element is positive semidefinite (check diagonal)
    bool all_positive = true;
    for (size_t e = 0; e < num_elements && all_positive; e++) {
        if (!elements[e]) continue;
        for (size_t i = 0; i < dim; i++) {
            if (elements[e][i * dim + i].real < -tolerance) {
                all_positive = false;
                add_error_to_result(&result, VALIDATION_RULE_POSITIVE_SEMIDEFINITE,
                                   VALIDATION_SEVERITY_ERROR,
                                   "POVM element has negative diagonal",
                                   0.0, elements[e][i * dim + i].real, tolerance);
                break;
            }
        }
    }

    if (all_positive) {
        result.passed_checks++;
    }

    // Check 2: Sum to identity
    double max_deviation = 0.0;
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            ComplexDouble sum = {0.0, 0.0};
            for (size_t e = 0; e < num_elements; e++) {
                if (elements[e]) {
                    sum = complex_double_add(sum, elements[e][i * dim + j]);
                }
            }
            double expected = (i == j) ? 1.0 : 0.0;
            ComplexDouble expected_c = {expected, 0.0};
            ComplexDouble diff = complex_double_subtract(sum, expected_c);
            double deviation = complex_double_abs(diff);
            if (deviation > max_deviation) {
                max_deviation = deviation;
            }
        }
    }

    if (max_deviation > tolerance) {
        add_error_to_result(&result, VALIDATION_RULE_PROBABILITY_SUM,
                           VALIDATION_SEVERITY_ERROR,
                           "POVM elements don't sum to identity",
                           0.0, max_deviation, tolerance);
    } else {
        result.passed_checks++;
    }

    result.validation_time_ns = get_timestamp_ns() - start;
    return result;
}

// ============================================================================
// Parameter Validation
// ============================================================================

validation_result_t validation_check_parameter_bounds(
    validation_analyzer_t* analyzer,
    const char* param_name,
    double value,
    double min_val,
    double max_val) {

    validation_result_t result = create_empty_result();
    result.total_checks = 1;
    uint64_t start = get_timestamp_ns();

    if (value < min_val || value > max_val) {
        char msg[256];
        snprintf(msg, sizeof(msg), "Parameter '%s' out of bounds [%.6g, %.6g]",
                param_name ? param_name : "unknown", min_val, max_val);
        add_error_to_result(&result, VALIDATION_RULE_PARAMETER_BOUNDS,
                           VALIDATION_SEVERITY_ERROR, msg,
                           (min_val + max_val) / 2.0, value, 0);
    } else {
        result.passed_checks = 1;
    }

    result.validation_time_ns = get_timestamp_ns() - start;
    return result;
}

validation_result_t validation_check_angle(
    validation_analyzer_t* analyzer,
    double angle,
    double min_angle,
    double max_angle) {

    return validation_check_parameter_bounds(analyzer, "angle",
                                              angle, min_angle, max_angle);
}

validation_result_t validation_check_qubit_index(
    validation_analyzer_t* analyzer,
    size_t qubit_index,
    size_t num_qubits) {

    validation_result_t result = create_empty_result();
    result.total_checks = 1;
    uint64_t start = get_timestamp_ns();

    if (qubit_index >= num_qubits) {
        char msg[256];
        snprintf(msg, sizeof(msg), "Qubit index %zu out of range [0, %zu)",
                qubit_index, num_qubits);
        add_error_to_result(&result, VALIDATION_RULE_PARAMETER_BOUNDS,
                           VALIDATION_SEVERITY_ERROR, msg,
                           (double)(num_qubits - 1), (double)qubit_index, 0);
    } else {
        result.passed_checks = 1;
    }

    result.validation_time_ns = get_timestamp_ns() - start;
    return result;
}

validation_result_t validation_check_dimension_power_of_2(
    validation_analyzer_t* analyzer,
    size_t dim) {

    validation_result_t result = create_empty_result();
    result.total_checks = 1;
    uint64_t start = get_timestamp_ns();

    if (!is_power_of_2(dim)) {
        char msg[256];
        snprintf(msg, sizeof(msg), "Dimension %zu is not a power of 2", dim);
        add_error_to_result(&result, VALIDATION_RULE_QUBIT_COUNT,
                           VALIDATION_SEVERITY_ERROR, msg, 0, (double)dim, 0);
    } else {
        result.passed_checks = 1;
    }

    result.validation_time_ns = get_timestamp_ns() - start;
    return result;
}

// ============================================================================
// Data Integrity Validation
// ============================================================================

validation_result_t validation_check_finite_real(
    validation_analyzer_t* analyzer,
    const double* data,
    size_t count) {

    validation_result_t result = create_empty_result();
    result.total_checks = 1;
    uint64_t start = get_timestamp_ns();

    if (!data) {
        add_error_to_result(&result, VALIDATION_RULE_FINITE_VALUES,
                           VALIDATION_SEVERITY_FATAL,
                           "NULL data pointer", 0, 0, 0);
        result.validation_time_ns = get_timestamp_ns() - start;
        return result;
    }

    for (size_t i = 0; i < count; i++) {
        if (!isfinite(data[i])) {
            char msg[256];
            snprintf(msg, sizeof(msg), "Non-finite value at index %zu", i);
            add_error_to_result(&result, VALIDATION_RULE_FINITE_VALUES,
                               VALIDATION_SEVERITY_ERROR, msg, 0, data[i], 0);
            result.validation_time_ns = get_timestamp_ns() - start;
            return result;
        }
    }

    result.passed_checks = 1;
    result.validation_time_ns = get_timestamp_ns() - start;
    return result;
}

validation_result_t validation_check_finite_complex(
    validation_analyzer_t* analyzer,
    const ComplexDouble* data,
    size_t count) {

    validation_result_t result = create_empty_result();
    result.total_checks = 1;
    uint64_t start = get_timestamp_ns();

    if (!data) {
        add_error_to_result(&result, VALIDATION_RULE_FINITE_VALUES,
                           VALIDATION_SEVERITY_FATAL,
                           "NULL data pointer", 0, 0, 0);
        result.validation_time_ns = get_timestamp_ns() - start;
        return result;
    }

    for (size_t i = 0; i < count; i++) {
        if (!isfinite(data[i].real) || !isfinite(data[i].imag)) {
            char msg[256];
            snprintf(msg, sizeof(msg), "Non-finite value at index %zu", i);
            add_error_to_result(&result, VALIDATION_RULE_FINITE_VALUES,
                               VALIDATION_SEVERITY_ERROR, msg, 0, 0, 0);
            result.validation_time_ns = get_timestamp_ns() - start;
            return result;
        }
    }

    result.passed_checks = 1;
    result.validation_time_ns = get_timestamp_ns() - start;
    return result;
}

validation_result_t validation_check_dimensions(
    validation_analyzer_t* analyzer,
    size_t rows_a, size_t cols_a,
    size_t rows_b, size_t cols_b,
    const char* operation) {

    validation_result_t result = create_empty_result();
    result.total_checks = 1;
    uint64_t start = get_timestamp_ns();

    bool match = false;

    if (operation && strcmp(operation, "multiply") == 0) {
        match = (cols_a == rows_b);
    } else if (operation && strcmp(operation, "add") == 0) {
        match = (rows_a == rows_b && cols_a == cols_b);
    } else {
        match = (rows_a == rows_b && cols_a == cols_b);
    }

    if (!match) {
        char msg[256];
        snprintf(msg, sizeof(msg),
                "Dimension mismatch for %s: (%zu×%zu) vs (%zu×%zu)",
                operation ? operation : "operation",
                rows_a, cols_a, rows_b, cols_b);
        add_error_to_result(&result, VALIDATION_RULE_DIMENSION_MATCH,
                           VALIDATION_SEVERITY_ERROR, msg, 0, 0, 0);
    } else {
        result.passed_checks = 1;
    }

    result.validation_time_ns = get_timestamp_ns() - start;
    return result;
}

validation_result_t validation_check_checksum(
    validation_analyzer_t* analyzer,
    const void* data,
    size_t size,
    uint64_t expected_checksum) {

    validation_result_t result = create_empty_result();
    result.total_checks = 1;
    uint64_t start = get_timestamp_ns();

    if (!data || size == 0) {
        add_error_to_result(&result, VALIDATION_RULE_FINITE_VALUES,
                           VALIDATION_SEVERITY_FATAL,
                           "Invalid data for checksum", 0, 0, 0);
        result.validation_time_ns = get_timestamp_ns() - start;
        return result;
    }

    // Simple FNV-1a hash
    uint64_t hash = 14695981039346656037ULL;
    const uint8_t* bytes = (const uint8_t*)data;
    for (size_t i = 0; i < size; i++) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }

    if (hash != expected_checksum) {
        add_error_to_result(&result, VALIDATION_RULE_CUSTOM,
                           VALIDATION_SEVERITY_ERROR,
                           "Checksum mismatch",
                           (double)expected_checksum, (double)hash, 0);
    } else {
        result.passed_checks = 1;
    }

    result.validation_time_ns = get_timestamp_ns() - start;
    return result;
}

// ============================================================================
// Batch Validation
// ============================================================================

validation_result_t validation_batch_validate(
    validation_analyzer_t* analyzer,
    validation_data_type_t data_type,
    const void** data_items,
    const size_t* sizes,
    size_t count) {

    validation_result_t result = create_empty_result();
    uint64_t start = get_timestamp_ns();

    if (!data_items || !sizes || count == 0) {
        add_error_to_result(&result, VALIDATION_RULE_FINITE_VALUES,
                           VALIDATION_SEVERITY_FATAL,
                           "Invalid batch data", 0, 0, 0);
        result.validation_time_ns = get_timestamp_ns() - start;
        return result;
    }

    for (size_t i = 0; i < count; i++) {
        validation_result_t item_result;

        switch (data_type) {
            case VALIDATION_DATA_STATE_VECTOR:
                item_result = validation_check_state_vector(
                    analyzer,
                    (const ComplexDouble*)data_items[i],
                    sizes[i],
                    NULL);
                break;
            case VALIDATION_DATA_PROBABILITY_DIST:
                item_result = validation_check_probability_distribution(
                    analyzer,
                    (const double*)data_items[i],
                    sizes[i],
                    VALIDATION_DEFAULT_PROBABILITY_TOLERANCE);
                break;
            default:
                item_result = validation_check_finite_complex(
                    analyzer,
                    (const ComplexDouble*)data_items[i],
                    sizes[i]);
                break;
        }

        result.total_checks += item_result.total_checks;
        result.passed_checks += item_result.passed_checks;
        result.warning_checks += item_result.warning_checks;
        result.failed_checks += item_result.failed_checks;

        if (item_result.overall_status == VALIDATION_STATUS_FAIL) {
            result.overall_status = VALIDATION_STATUS_FAIL;
        } else if (item_result.overall_status == VALIDATION_STATUS_PASS_WITH_WARNINGS &&
                   result.overall_status == VALIDATION_STATUS_PASS) {
            result.overall_status = VALIDATION_STATUS_PASS_WITH_WARNINGS;
        }

        validation_result_free(&item_result);
    }

    result.validation_time_ns = get_timestamp_ns() - start;
    return result;
}

// ============================================================================
// Pipeline
// ============================================================================

validation_pipeline_t* validation_pipeline_create(
    validation_analyzer_t* analyzer) {

    validation_pipeline_t* pipeline = calloc(1, sizeof(validation_pipeline_t));
    if (!pipeline) return NULL;

    pipeline->analyzer = analyzer;
    pipeline->stages_capacity = 16;
    pipeline->stages = calloc(pipeline->stages_capacity, sizeof(validation_rule_type_t));
    pipeline->tolerances = calloc(pipeline->stages_capacity, sizeof(double));

    if (!pipeline->stages || !pipeline->tolerances) {
        free(pipeline->stages);
        free(pipeline->tolerances);
        free(pipeline);
        return NULL;
    }

    return pipeline;
}

void validation_pipeline_destroy(validation_pipeline_t* pipeline) {
    if (!pipeline) return;
    free(pipeline->stages);
    free(pipeline->tolerances);
    free(pipeline);
}

bool validation_pipeline_add_stage(
    validation_pipeline_t* pipeline,
    validation_rule_type_t rule,
    double tolerance) {

    if (!pipeline) return false;

    if (pipeline->num_stages >= pipeline->stages_capacity) {
        size_t new_cap = pipeline->stages_capacity * 2;
        validation_rule_type_t* new_stages = realloc(pipeline->stages,
            new_cap * sizeof(validation_rule_type_t));
        double* new_tols = realloc(pipeline->tolerances, new_cap * sizeof(double));

        if (!new_stages || !new_tols) {
            free(new_stages);
            free(new_tols);
            return false;
        }

        pipeline->stages = new_stages;
        pipeline->tolerances = new_tols;
        pipeline->stages_capacity = new_cap;
    }

    pipeline->stages[pipeline->num_stages] = rule;
    pipeline->tolerances[pipeline->num_stages] = tolerance;
    pipeline->num_stages++;

    return true;
}

validation_result_t validation_pipeline_execute(
    validation_pipeline_t* pipeline,
    const void* data,
    size_t size,
    validation_data_type_t type) {

    validation_result_t result = create_empty_result();

    if (!pipeline || !data) {
        add_error_to_result(&result, VALIDATION_RULE_FINITE_VALUES,
                           VALIDATION_SEVERITY_FATAL,
                           "Invalid pipeline or data", 0, 0, 0);
        return result;
    }

    uint64_t start = get_timestamp_ns();

    for (size_t i = 0; i < pipeline->num_stages; i++) {
        validation_result_t stage_result = create_empty_result();

        switch (pipeline->stages[i]) {
            case VALIDATION_RULE_NORMALIZATION:
                if (type == VALIDATION_DATA_STATE_VECTOR) {
                    stage_result = validation_check_state_norm(
                        pipeline->analyzer,
                        (const ComplexDouble*)data,
                        size,
                        pipeline->tolerances[i]);
                }
                break;
            case VALIDATION_RULE_FINITE_VALUES:
                if (type == VALIDATION_DATA_STATE_VECTOR ||
                    type == VALIDATION_DATA_UNITARY_MATRIX) {
                    stage_result = validation_check_finite_complex(
                        pipeline->analyzer,
                        (const ComplexDouble*)data,
                        size);
                } else {
                    stage_result = validation_check_finite_real(
                        pipeline->analyzer,
                        (const double*)data,
                        size);
                }
                break;
            default:
                break;
        }

        result.total_checks += stage_result.total_checks;
        result.passed_checks += stage_result.passed_checks;
        result.failed_checks += stage_result.failed_checks;

        if (stage_result.overall_status == VALIDATION_STATUS_FAIL) {
            result.overall_status = VALIDATION_STATUS_FAIL;
        }

        validation_result_free(&stage_result);
    }

    result.validation_time_ns = get_timestamp_ns() - start;
    return result;
}

// ============================================================================
// Statistics and Reporting
// ============================================================================

bool validation_get_statistics(validation_analyzer_t* analyzer,
                                validation_stats_t* stats) {
    if (!analyzer || !analyzer->initialized || !stats) return false;

    pthread_mutex_lock(&analyzer->mutex);
    *stats = analyzer->stats;
    pthread_mutex_unlock(&analyzer->mutex);

    return true;
}

bool validation_get_common_failures(
    validation_analyzer_t* analyzer,
    validation_rule_type_t** rules,
    uint64_t** counts,
    size_t* num_rules) {

    if (!analyzer || !analyzer->initialized || !rules || !counts || !num_rules) {
        return false;
    }

    pthread_mutex_lock(&analyzer->mutex);

    size_t count = 0;
    for (int i = 0; i < VALIDATION_RULE_COUNT; i++) {
        if (analyzer->stats.rule_fail_counts[i] > 0) {
            count++;
        }
    }

    *rules = calloc(count, sizeof(validation_rule_type_t));
    *counts = calloc(count, sizeof(uint64_t));

    if (!*rules || !*counts) {
        free(*rules);
        free(*counts);
        pthread_mutex_unlock(&analyzer->mutex);
        return false;
    }

    size_t idx = 0;
    for (int i = 0; i < VALIDATION_RULE_COUNT; i++) {
        if (analyzer->stats.rule_fail_counts[i] > 0) {
            (*rules)[idx] = (validation_rule_type_t)i;
            (*counts)[idx] = analyzer->stats.rule_fail_counts[i];
            idx++;
        }
    }

    *num_rules = count;
    pthread_mutex_unlock(&analyzer->mutex);

    return true;
}

char* validation_generate_report(validation_analyzer_t* analyzer) {
    if (!analyzer || !analyzer->initialized) return NULL;

    pthread_mutex_lock(&analyzer->mutex);

    char* report = calloc(8192, sizeof(char));
    if (!report) {
        pthread_mutex_unlock(&analyzer->mutex);
        return NULL;
    }

    int offset = 0;
    offset += snprintf(report + offset, 8192 - offset,
        "=== Validation Analyzer Report ===\n\n");

    offset += snprintf(report + offset, 8192 - offset,
        "Statistics:\n");
    offset += snprintf(report + offset, 8192 - offset,
        "  Total validations: %llu\n",
        (unsigned long long)analyzer->stats.total_validations);
    offset += snprintf(report + offset, 8192 - offset,
        "  Passes: %llu\n",
        (unsigned long long)analyzer->stats.total_passes);
    offset += snprintf(report + offset, 8192 - offset,
        "  Warnings: %llu\n",
        (unsigned long long)analyzer->stats.total_warnings);
    offset += snprintf(report + offset, 8192 - offset,
        "  Failures: %llu\n",
        (unsigned long long)analyzer->stats.total_failures);

    offset += snprintf(report + offset, 8192 - offset,
        "\nRules configured: %zu\n", analyzer->num_rules);

    offset += snprintf(report + offset, 8192 - offset,
        "\nFailure counts by rule:\n");
    for (int i = 0; i < VALIDATION_RULE_COUNT; i++) {
        if (analyzer->stats.rule_fail_counts[i] > 0) {
            offset += snprintf(report + offset, 8192 - offset,
                "  %s: %llu\n",
                validation_rule_name((validation_rule_type_t)i),
                (unsigned long long)analyzer->stats.rule_fail_counts[i]);
        }
    }

    pthread_mutex_unlock(&analyzer->mutex);

    return report;
}

char* validation_export_json(validation_analyzer_t* analyzer) {
    if (!analyzer || !analyzer->initialized) return NULL;

    pthread_mutex_lock(&analyzer->mutex);

    char* json = calloc(16384, sizeof(char));
    if (!json) {
        pthread_mutex_unlock(&analyzer->mutex);
        return NULL;
    }

    int offset = 0;
    offset += snprintf(json + offset, 16384 - offset, "{\n");
    offset += snprintf(json + offset, 16384 - offset,
        "  \"total_validations\": %llu,\n",
        (unsigned long long)analyzer->stats.total_validations);
    offset += snprintf(json + offset, 16384 - offset,
        "  \"total_passes\": %llu,\n",
        (unsigned long long)analyzer->stats.total_passes);
    offset += snprintf(json + offset, 16384 - offset,
        "  \"total_failures\": %llu,\n",
        (unsigned long long)analyzer->stats.total_failures);
    offset += snprintf(json + offset, 16384 - offset,
        "  \"num_rules\": %zu\n", analyzer->num_rules);
    offset += snprintf(json + offset, 16384 - offset, "}\n");

    pthread_mutex_unlock(&analyzer->mutex);

    return json;
}

bool validation_export_to_file(validation_analyzer_t* analyzer,
                                const char* filename) {
    if (!analyzer || !filename) return false;

    char* json = validation_export_json(analyzer);
    if (!json) return false;

    FILE* f = fopen(filename, "w");
    if (!f) {
        free(json);
        return false;
    }

    fputs(json, f);
    fclose(f);
    free(json);

    return true;
}

// ============================================================================
// Utility Functions
// ============================================================================

const char* validation_rule_name(validation_rule_type_t rule) {
    static const char* names[] = {
        "NORMALIZATION",
        "UNITARITY",
        "HERMITICITY",
        "POSITIVE_SEMIDEFINITE",
        "TRACE_ONE",
        "PROBABILITY_SUM",
        "PROBABILITY_BOUNDS",
        "DIMENSION_MATCH",
        "QUBIT_COUNT",
        "PARAMETER_BOUNDS",
        "FINITE_VALUES",
        "NON_NEGATIVE",
        "POSITIVE",
        "INTEGER",
        "CUSTOM"
    };

    if (rule >= 0 && rule < VALIDATION_RULE_COUNT) {
        return names[rule];
    }
    return "UNKNOWN";
}

const char* validation_severity_name(validation_severity_t severity) {
    static const char* names[] = {
        "INFO",
        "WARNING",
        "ERROR",
        "FATAL"
    };

    if (severity >= 0 && severity <= VALIDATION_SEVERITY_FATAL) {
        return names[severity];
    }
    return "UNKNOWN";
}

const char* validation_status_name(validation_status_t status) {
    static const char* names[] = {
        "PASS",
        "PASS_WITH_WARNINGS",
        "FAIL",
        "SKIPPED",
        "ERROR"
    };

    if (status >= 0 && status <= VALIDATION_STATUS_ERROR) {
        return names[status];
    }
    return "UNKNOWN";
}

void validation_result_free(validation_result_t* result) {
    if (!result) return;
    free(result->errors);
    result->errors = NULL;
    result->error_count = 0;
}

void validation_free_rules(validation_rule_t* rules, size_t count) {
    (void)count;
    free(rules);
}

state_validation_params_t validation_default_state_params(void) {
    state_validation_params_t params;
    memset(&params, 0, sizeof(params));
    params.norm_tolerance = VALIDATION_DEFAULT_NORM_TOLERANCE;
    params.phase_tolerance = 1e-10;
    params.check_sparsity = false;
    params.sparsity_threshold = 1e-15;
    params.allow_subnormal = true;
    return params;
}

matrix_validation_params_t validation_default_matrix_params(void) {
    matrix_validation_params_t params;
    memset(&params, 0, sizeof(params));
    params.unitarity_tolerance = VALIDATION_DEFAULT_UNITARITY_TOLERANCE;
    params.hermitian_tolerance = VALIDATION_DEFAULT_HERMITIAN_TOLERANCE;
    params.positive_tolerance = 1e-10;
    params.trace_tolerance = 1e-10;
    params.check_eigenvalues = false;
    params.check_condition_number = false;
    params.max_condition_number = 1e12;
    return params;
}

const char* validation_get_last_error(validation_analyzer_t* analyzer) {
    if (!analyzer) return "NULL analyzer";
    return analyzer->last_error;
}
