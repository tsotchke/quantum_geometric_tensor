/**
 * @file uncertainty_quantification.h
 * @brief Uncertainty quantification for quantum computations
 *
 * Provides comprehensive uncertainty quantification including measurement
 * uncertainty, error bounds, confidence intervals, sensitivity analysis,
 * and propagation of uncertainties through quantum circuits and algorithms.
 */

#ifndef UNCERTAINTY_QUANTIFICATION_H
#define UNCERTAINTY_QUANTIFICATION_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct quantum_state;
struct quantum_circuit;
struct quantum_result;

// =============================================================================
// Uncertainty Types
// =============================================================================

/**
 * Types of uncertainty sources in quantum computations
 */
typedef enum {
    UNCERTAINTY_STATISTICAL,         // Shot noise, sampling variance
    UNCERTAINTY_SYSTEMATIC,          // Calibration, drift
    UNCERTAINTY_GATE_ERROR,          // Gate infidelity
    UNCERTAINTY_MEASUREMENT,         // Readout error
    UNCERTAINTY_DECOHERENCE,         // T1/T2 decay
    UNCERTAINTY_CROSSTALK,           // Qubit crosstalk
    UNCERTAINTY_TRUNCATION,          // Numerical truncation
    UNCERTAINTY_MODEL,               // Model uncertainty
    UNCERTAINTY_COMBINED             // Total combined uncertainty
} UncertaintyType;

/**
 * Uncertainty propagation methods
 */
typedef enum {
    PROPAGATION_LINEAR,              // Linear error propagation
    PROPAGATION_MONTE_CARLO,         // Monte Carlo sampling
    PROPAGATION_TAYLOR,              // Taylor series expansion
    PROPAGATION_UNSCENTED,           // Unscented transform
    PROPAGATION_POLYNOMIAL_CHAOS,    // Polynomial chaos expansion
    PROPAGATION_BAYESIAN             // Bayesian inference
} PropagationMethod;

/**
 * Confidence interval types
 */
typedef enum {
    INTERVAL_STANDARD,               // Standard deviation based
    INTERVAL_PERCENTILE,             // Percentile based
    INTERVAL_HDI,                    // Highest density interval
    INTERVAL_BOOTSTRAP,              // Bootstrap confidence interval
    INTERVAL_BAYESIAN_CREDIBLE       // Bayesian credible interval
} ConfidenceIntervalType;

/**
 * Distribution types for uncertainty modeling
 */
typedef enum {
    DIST_NORMAL,                     // Gaussian distribution
    DIST_UNIFORM,                    // Uniform distribution
    DIST_BINOMIAL,                   // Binomial (for counts)
    DIST_POISSON,                    // Poisson distribution
    DIST_BETA,                       // Beta distribution
    DIST_GAMMA,                      // Gamma distribution
    DIST_CUSTOM                      // Custom distribution
} DistributionType;

// =============================================================================
// Core Structures
// =============================================================================

/**
 * Uncertainty value with bounds
 */
typedef struct {
    double value;                    // Central value
    double uncertainty;              // Standard uncertainty
    double lower_bound;              // Lower confidence bound
    double upper_bound;              // Upper confidence bound
    double confidence_level;         // Confidence level (e.g., 0.95)
    UncertaintyType type;
} UncertainValue;

/**
 * Uncertainty distribution
 */
typedef struct {
    DistributionType type;
    double* parameters;              // Distribution parameters
    size_t num_parameters;
    double* samples;                 // Optional sample data
    size_t num_samples;
    double mean;
    double variance;
    double skewness;
    double kurtosis;
} UncertaintyDistribution;

/**
 * Covariance matrix for correlated uncertainties
 */
typedef struct {
    double* matrix;                  // Covariance matrix [n x n]
    size_t dimension;
    double* eigenvalues;             // Eigenvalues (optional)
    double* eigenvectors;            // Eigenvectors (optional)
    bool is_positive_definite;
} CovarianceMatrix;

/**
 * Sensitivity analysis result
 */
typedef struct {
    double* sensitivities;           // ∂output/∂input for each input
    size_t num_inputs;
    double* sobol_indices;           // First-order Sobol indices
    double* total_sobol_indices;     // Total Sobol indices
    double* interaction_indices;     // Second-order interactions
    char** input_names;
} SensitivityAnalysis;

/**
 * Error budget breakdown
 */
typedef struct {
    UncertainValue statistical;
    UncertainValue systematic;
    UncertainValue gate_error;
    UncertainValue measurement_error;
    UncertainValue decoherence;
    UncertainValue crosstalk;
    UncertainValue truncation;
    UncertainValue total;
    double* component_contributions; // Fractional contributions
    size_t num_components;
} ErrorBudget;

// =============================================================================
// Quantum-Specific Uncertainty Structures
// =============================================================================

/**
 * Quantum state uncertainty (fidelity bounds)
 */
typedef struct {
    struct quantum_state* nominal_state;
    double fidelity_lower_bound;
    double fidelity_upper_bound;
    double trace_distance_bound;
    double* amplitude_uncertainties;  // Per-amplitude uncertainty
    size_t num_amplitudes;
    CovarianceMatrix* correlation;   // Amplitude correlations
} QuantumStateUncertainty;

/**
 * Measurement outcome uncertainty
 */
typedef struct {
    size_t num_outcomes;
    double* probabilities;           // Measured probabilities
    double* uncertainties;           // Uncertainty per probability
    uint64_t* counts;                // Raw counts
    uint64_t total_shots;
    double chi_squared;              // Chi-squared statistic
    double p_value;                  // Goodness of fit p-value
    CovarianceMatrix* probability_covariance;
} MeasurementUncertainty;

/**
 * Expectation value uncertainty
 */
typedef struct {
    double expectation_value;
    double variance;
    double standard_error;
    double confidence_lower;
    double confidence_upper;
    double confidence_level;
    size_t num_samples;
    double effective_samples;        // Effective sample size (for correlated data)
    double autocorrelation_time;
} ExpectationUncertainty;

/**
 * Circuit execution uncertainty
 */
typedef struct {
    double* gate_fidelities;
    double* gate_uncertainties;
    size_t num_gates;
    double circuit_fidelity;
    double circuit_fidelity_uncertainty;
    double success_probability;
    double success_probability_uncertainty;
    ErrorBudget error_budget;
} CircuitUncertainty;

// =============================================================================
// Configuration Structures
// =============================================================================

/**
 * Uncertainty quantification configuration
 */
typedef struct {
    PropagationMethod propagation_method;
    ConfidenceIntervalType interval_type;
    double confidence_level;         // Default: 0.95
    size_t monte_carlo_samples;      // For MC propagation
    size_t bootstrap_samples;        // For bootstrap intervals
    bool include_correlations;
    bool compute_sensitivities;
    size_t sobol_samples;            // For Sobol analysis
    double numerical_tolerance;
} UncertaintyConfig;

/**
 * Error model for uncertainty propagation
 */
typedef struct {
    double* single_qubit_errors;     // Per-qubit error rates
    double* two_qubit_errors;        // Per-pair error rates
    double* readout_errors;          // Measurement errors per qubit
    double t1_time;                  // Relaxation time
    double t2_time;                  // Dephasing time
    double crosstalk_strength;
    size_t num_qubits;
    CovarianceMatrix* error_correlations;
} QuantumErrorModel;

// =============================================================================
// Main Context
// =============================================================================

/**
 * Uncertainty quantification context
 */
typedef struct {
    UncertaintyConfig config;
    QuantumErrorModel* error_model;
    UncertaintyDistribution** input_distributions;
    size_t num_inputs;
    void* rng_state;                 // Random number generator state
    double* workspace;               // Computation workspace
    size_t workspace_size;
} UncertaintyContext;

// =============================================================================
// Context Management
// =============================================================================

/**
 * Create uncertainty quantification context
 */
int uncertainty_context_create(UncertaintyContext** ctx,
                               UncertaintyConfig* config);

/**
 * Destroy uncertainty context
 */
void uncertainty_context_destroy(UncertaintyContext* ctx);

/**
 * Set error model for quantum uncertainty
 */
int uncertainty_set_error_model(UncertaintyContext* ctx,
                                QuantumErrorModel* model);

/**
 * Add input distribution
 */
int uncertainty_add_input(UncertaintyContext* ctx,
                          const char* name,
                          UncertaintyDistribution* dist);

// =============================================================================
// Basic Uncertainty Operations
// =============================================================================

/**
 * Create uncertain value
 */
int uncertain_value_create(UncertainValue** value,
                           double central_value,
                           double uncertainty,
                           UncertaintyType type);

/**
 * Destroy uncertain value
 */
void uncertain_value_destroy(UncertainValue* value);

/**
 * Add two uncertain values (with correlation)
 */
int uncertain_value_add(UncertainValue* a,
                        UncertainValue* b,
                        double correlation,
                        UncertainValue** result);

/**
 * Multiply two uncertain values
 */
int uncertain_value_multiply(UncertainValue* a,
                             UncertainValue* b,
                             double correlation,
                             UncertainValue** result);

/**
 * Apply function to uncertain value
 */
int uncertain_value_apply_function(UncertainValue* input,
                                   double (*func)(double),
                                   double (*derivative)(double),
                                   UncertainValue** result);

/**
 * Combine multiple uncertainty sources
 */
int uncertain_value_combine(UncertainValue** sources,
                            size_t num_sources,
                            CovarianceMatrix* correlations,
                            UncertainValue** combined);

// =============================================================================
// Distribution Operations
// =============================================================================

/**
 * Create uncertainty distribution
 */
int uncertainty_distribution_create(UncertaintyDistribution** dist,
                                    DistributionType type,
                                    double* parameters,
                                    size_t num_parameters);

/**
 * Destroy distribution
 */
void uncertainty_distribution_destroy(UncertaintyDistribution* dist);

/**
 * Sample from distribution
 */
int uncertainty_distribution_sample(UncertaintyDistribution* dist,
                                    void* rng_state,
                                    double* samples_out,
                                    size_t num_samples);

/**
 * Compute distribution statistics
 */
int uncertainty_distribution_statistics(UncertaintyDistribution* dist);

/**
 * Fit distribution to data
 */
int uncertainty_distribution_fit(UncertaintyDistribution** dist,
                                 double* data,
                                 size_t num_data,
                                 DistributionType type);

// =============================================================================
// Covariance Operations
// =============================================================================

/**
 * Create covariance matrix
 */
int covariance_matrix_create(CovarianceMatrix** cov,
                             double* matrix,
                             size_t dimension);

/**
 * Destroy covariance matrix
 */
void covariance_matrix_destroy(CovarianceMatrix* cov);

/**
 * Compute covariance from samples
 */
int covariance_matrix_from_samples(CovarianceMatrix** cov,
                                   double* samples,
                                   size_t num_samples,
                                   size_t dimension);

/**
 * Check positive definiteness
 */
bool covariance_matrix_is_valid(CovarianceMatrix* cov);

/**
 * Regularize covariance matrix
 */
int covariance_matrix_regularize(CovarianceMatrix* cov,
                                 double regularization);

/**
 * Compute correlation matrix from covariance
 */
int covariance_to_correlation(CovarianceMatrix* cov,
                              double** correlation_out);

// =============================================================================
// Quantum Uncertainty Operations
// =============================================================================

/**
 * Quantify quantum state uncertainty
 */
int quantum_state_uncertainty(UncertaintyContext* ctx,
                              struct quantum_state* state,
                              size_t num_samples,
                              QuantumStateUncertainty** result);

/**
 * Quantify measurement uncertainty from counts
 */
int measurement_uncertainty_from_counts(uint64_t* counts,
                                        size_t num_outcomes,
                                        uint64_t total_shots,
                                        double confidence_level,
                                        MeasurementUncertainty** result);

/**
 * Compute expectation value uncertainty
 */
int expectation_value_uncertainty(struct quantum_result* result,
                                  double* observable,
                                  size_t observable_size,
                                  ExpectationUncertainty** uncertainty);

/**
 * Compute circuit execution uncertainty
 */
int circuit_uncertainty(UncertaintyContext* ctx,
                        struct quantum_circuit* circuit,
                        CircuitUncertainty** result);

/**
 * Propagate uncertainty through circuit
 */
int propagate_uncertainty_through_circuit(UncertaintyContext* ctx,
                                          struct quantum_circuit* circuit,
                                          QuantumStateUncertainty* input,
                                          QuantumStateUncertainty** output);

// =============================================================================
// Confidence Intervals
// =============================================================================

/**
 * Compute confidence interval for mean
 */
int confidence_interval_mean(double* samples,
                             size_t num_samples,
                             double confidence_level,
                             ConfidenceIntervalType type,
                             double* lower_out,
                             double* upper_out);

/**
 * Compute bootstrap confidence interval
 */
int confidence_interval_bootstrap(double* samples,
                                  size_t num_samples,
                                  double (*statistic)(double*, size_t),
                                  size_t num_bootstrap,
                                  double confidence_level,
                                  double* lower_out,
                                  double* upper_out);

/**
 * Compute Bayesian credible interval
 */
int confidence_interval_bayesian(double* samples,
                                 size_t num_samples,
                                 UncertaintyDistribution* prior,
                                 double confidence_level,
                                 double* lower_out,
                                 double* upper_out);

// =============================================================================
// Sensitivity Analysis
// =============================================================================

/**
 * Perform sensitivity analysis
 */
int sensitivity_analysis_create(SensitivityAnalysis** analysis,
                                UncertaintyContext* ctx,
                                double (*model)(double*, size_t),
                                size_t num_inputs);

/**
 * Destroy sensitivity analysis
 */
void sensitivity_analysis_destroy(SensitivityAnalysis* analysis);

/**
 * Compute local sensitivities (derivatives)
 */
int sensitivity_local(SensitivityAnalysis* analysis,
                      double* nominal_inputs,
                      double perturbation);

/**
 * Compute Sobol sensitivity indices
 */
int sensitivity_sobol(SensitivityAnalysis* analysis,
                      UncertaintyDistribution** input_dists,
                      size_t num_samples);

/**
 * Compute Morris screening
 */
int sensitivity_morris(SensitivityAnalysis* analysis,
                       double* input_ranges,
                       size_t num_trajectories,
                       size_t num_levels);

// =============================================================================
// Error Budget
// =============================================================================

/**
 * Create error budget
 */
int error_budget_create(ErrorBudget** budget);

/**
 * Destroy error budget
 */
void error_budget_destroy(ErrorBudget* budget);

/**
 * Compute error budget for circuit
 */
int error_budget_compute(UncertaintyContext* ctx,
                         struct quantum_circuit* circuit,
                         ErrorBudget** budget);

/**
 * Add component to error budget
 */
int error_budget_add_component(ErrorBudget* budget,
                               UncertaintyType type,
                               UncertainValue* component);

/**
 * Combine error budget components
 */
int error_budget_combine(ErrorBudget* budget);

/**
 * Get dominant error source
 */
int error_budget_dominant_source(ErrorBudget* budget,
                                 UncertaintyType* type_out,
                                 double* contribution_out);

// =============================================================================
// Uncertainty Propagation
// =============================================================================

/**
 * Linear uncertainty propagation (first-order Taylor)
 */
int propagate_linear(double* inputs,
                     double* input_uncertainties,
                     size_t num_inputs,
                     double (*function)(double*, size_t),
                     double* jacobian,
                     CovarianceMatrix* input_cov,
                     UncertainValue** result);

/**
 * Monte Carlo uncertainty propagation
 */
int propagate_monte_carlo(UncertaintyContext* ctx,
                          UncertaintyDistribution** input_dists,
                          size_t num_inputs,
                          double (*function)(double*, size_t),
                          size_t num_samples,
                          UncertaintyDistribution** result);

/**
 * Unscented transform propagation
 */
int propagate_unscented(double* inputs,
                        CovarianceMatrix* input_cov,
                        size_t num_inputs,
                        double (*function)(double*, size_t, double*, size_t),
                        size_t output_dim,
                        double* output_mean,
                        CovarianceMatrix** output_cov);

/**
 * Polynomial chaos expansion
 */
int propagate_polynomial_chaos(UncertaintyContext* ctx,
                               UncertaintyDistribution** input_dists,
                               size_t num_inputs,
                               double (*function)(double*, size_t),
                               size_t expansion_order,
                               double** pce_coefficients,
                               size_t* num_coefficients);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Compute standard error from samples
 */
double compute_standard_error(double* samples, size_t num_samples);

/**
 * Compute effective sample size (for autocorrelated data)
 */
double compute_effective_sample_size(double* samples, size_t num_samples);

/**
 * Compute autocorrelation time
 */
double compute_autocorrelation_time(double* samples, size_t num_samples);

/**
 * Validate uncertainty value
 */
bool uncertainty_is_valid(UncertainValue* value);

/**
 * Print uncertainty value
 */
void uncertainty_print(UncertainValue* value, const char* name);

/**
 * Print error budget
 */
void error_budget_print(ErrorBudget* budget);

/**
 * Print sensitivity analysis
 */
void sensitivity_analysis_print(SensitivityAnalysis* analysis);

/**
 * Free measurement uncertainty
 */
void measurement_uncertainty_destroy(MeasurementUncertainty* meas);

/**
 * Free expectation uncertainty
 */
void expectation_uncertainty_destroy(ExpectationUncertainty* exp);

/**
 * Free circuit uncertainty
 */
void circuit_uncertainty_destroy(CircuitUncertainty* circ);

/**
 * Free quantum state uncertainty
 */
void quantum_state_uncertainty_destroy(QuantumStateUncertainty* state);

#ifdef __cplusplus
}
#endif

#endif // UNCERTAINTY_QUANTIFICATION_H
