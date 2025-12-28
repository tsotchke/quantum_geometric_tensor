/**
 * @file statistical_analyzer.h
 * @brief Statistical Analysis for Quantum Error Correction and Operations
 *
 * Provides comprehensive statistical analysis including:
 * - Error rate estimation and tracking
 * - Probability distribution fitting
 * - Hypothesis testing for quantum states
 * - Confidence interval calculations
 * - Correlation analysis
 * - Bayesian inference for quantum parameters
 *
 * Part of the QGTL Monitoring Framework.
 */

#ifndef STATISTICAL_ANALYZER_H
#define STATISTICAL_ANALYZER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================

#define STAT_MAX_SAMPLES 1000000
#define STAT_MAX_DISTRIBUTIONS 100
#define STAT_MAX_CORRELATIONS 50
#define STAT_BOOTSTRAP_ITERATIONS 1000
#define STAT_MAX_BINS 256

// ============================================================================
// Enumerations
// ============================================================================

/**
 * Distribution types for fitting
 */
typedef enum {
    STAT_DIST_NORMAL,             // Gaussian/Normal distribution
    STAT_DIST_POISSON,            // Poisson (error counts)
    STAT_DIST_BINOMIAL,           // Binomial (success/failure)
    STAT_DIST_EXPONENTIAL,        // Exponential (decay times)
    STAT_DIST_UNIFORM,            // Uniform distribution
    STAT_DIST_CHI_SQUARED,        // Chi-squared
    STAT_DIST_STUDENT_T,          // Student's t-distribution
    STAT_DIST_F_DISTRIBUTION,     // F-distribution
    STAT_DIST_BETA,               // Beta distribution
    STAT_DIST_GAMMA,              // Gamma distribution
    STAT_DIST_WEIBULL,            // Weibull (reliability)
    STAT_DIST_LOG_NORMAL,         // Log-normal
    STAT_DIST_EMPIRICAL,          // Empirical (from data)
    STAT_DIST_TYPE_COUNT
} stat_distribution_type_t;

/**
 * Hypothesis test types
 */
typedef enum {
    STAT_TEST_Z_TEST,             // Z-test (known variance)
    STAT_TEST_T_TEST,             // Student's t-test
    STAT_TEST_PAIRED_T_TEST,      // Paired t-test
    STAT_TEST_WELCH_T_TEST,       // Welch's t-test (unequal variance)
    STAT_TEST_CHI_SQUARED,        // Chi-squared test
    STAT_TEST_F_TEST,             // F-test for variance
    STAT_TEST_KOLMOGOROV_SMIRNOV, // K-S test for distribution
    STAT_TEST_SHAPIRO_WILK,       // Normality test
    STAT_TEST_ANDERSON_DARLING,   // Distribution test
    STAT_TEST_MANN_WHITNEY,       // Non-parametric
    STAT_TEST_WILCOXON_SIGNED,    // Non-parametric paired
    STAT_TEST_FISHER_EXACT,       // Exact test for 2x2
    STAT_TEST_TYPE_COUNT
} stat_test_type_t;

/**
 * Correlation types
 */
typedef enum {
    STAT_CORR_PEARSON,            // Pearson (linear)
    STAT_CORR_SPEARMAN,           // Spearman (rank)
    STAT_CORR_KENDALL,            // Kendall tau
    STAT_CORR_POINT_BISERIAL,     // Binary-continuous
    STAT_CORR_AUTOCORRELATION,    // Temporal autocorrelation
    STAT_CORR_CROSS_CORRELATION,  // Cross-correlation
    STAT_CORR_PARTIAL,            // Partial correlation
    STAT_CORR_TYPE_COUNT
} stat_correlation_type_t;

/**
 * Confidence levels
 */
typedef enum {
    STAT_CONFIDENCE_90,           // 90% confidence
    STAT_CONFIDENCE_95,           // 95% confidence (default)
    STAT_CONFIDENCE_99,           // 99% confidence
    STAT_CONFIDENCE_999           // 99.9% confidence
} stat_confidence_level_t;

// ============================================================================
// Data Structures
// ============================================================================

/**
 * Basic descriptive statistics
 */
typedef struct {
    size_t count;                 // Number of samples
    double sum;                   // Sum of values
    double mean;                  // Arithmetic mean
    double variance;              // Sample variance
    double std_dev;               // Standard deviation
    double std_error;             // Standard error of mean
    double min;                   // Minimum value
    double max;                   // Maximum value
    double range;                 // Max - min
    double median;                // Median (50th percentile)
    double q1;                    // First quartile (25th)
    double q3;                    // Third quartile (75th)
    double iqr;                   // Interquartile range
    double skewness;              // Skewness (asymmetry)
    double kurtosis;              // Kurtosis (tail weight)
    double mad;                   // Median absolute deviation
    double geometric_mean;        // Geometric mean
    double harmonic_mean;         // Harmonic mean
} stat_descriptive_t;

/**
 * Confidence interval
 */
typedef struct {
    double point_estimate;        // Point estimate
    double lower_bound;           // Lower confidence bound
    double upper_bound;           // Upper confidence bound
    double confidence_level;      // Confidence level (0-1)
    double margin_of_error;       // Half-width of interval
    stat_confidence_level_t level; // Confidence level enum
} stat_confidence_interval_t;

/**
 * Distribution parameters
 */
typedef struct {
    stat_distribution_type_t type;
    double param1;                // First parameter (mean, lambda, etc.)
    double param2;                // Second parameter (std_dev, p, etc.)
    double param3;                // Third parameter (if needed)
    double goodness_of_fit;       // Chi-squared or K-S statistic
    double p_value;               // P-value of fit
    bool is_good_fit;             // Whether fit is acceptable
} stat_distribution_t;

/**
 * Hypothesis test result
 */
typedef struct {
    stat_test_type_t test_type;
    double test_statistic;        // Z, t, F, chi-squared, etc.
    double p_value;               // P-value
    double critical_value;        // Critical value at alpha
    double alpha;                 // Significance level
    double effect_size;           // Cohen's d, eta-squared, etc.
    double power;                 // Statistical power (if calculable)
    size_t degrees_of_freedom_1;  // Primary df
    size_t degrees_of_freedom_2;  // Secondary df (for F, etc.)
    bool reject_null;             // Whether to reject H0
    char description[256];        // Human-readable result
} stat_hypothesis_result_t;

/**
 * Correlation result
 */
typedef struct {
    stat_correlation_type_t type;
    double coefficient;           // Correlation coefficient (-1 to 1)
    double p_value;               // Significance
    double confidence_lower;      // Lower CI bound
    double confidence_upper;      // Upper CI bound
    size_t sample_size;           // Number of pairs
    bool is_significant;          // At alpha = 0.05
    double r_squared;             // Coefficient of determination
} stat_correlation_result_t;

/**
 * Error rate tracking for QEC
 */
typedef struct {
    double physical_error_rate;   // Physical qubit error rate
    double logical_error_rate;    // Logical qubit error rate (after EC)
    double syndrome_detection_rate; // Syndrome detection efficiency
    double correction_success_rate; // Successful corrections / total
    double threshold_estimate;    // Estimated threshold
    double error_suppression;     // Error suppression factor
    stat_confidence_interval_t physical_ci;
    stat_confidence_interval_t logical_ci;
    size_t total_cycles;          // QEC cycles analyzed
    size_t total_errors;          // Total errors observed
    size_t corrected_errors;      // Errors successfully corrected
} stat_error_rates_t;

/**
 * Bayesian inference result
 */
typedef struct {
    double prior_mean;            // Prior mean
    double prior_variance;        // Prior variance
    double posterior_mean;        // Posterior mean
    double posterior_variance;    // Posterior variance
    double likelihood;            // Likelihood of data
    double bayes_factor;          // Bayes factor vs null
    stat_confidence_interval_t credible_interval;  // Credible interval
    size_t sample_size;           // Number of observations
} stat_bayesian_result_t;

/**
 * Time series statistics
 */
typedef struct {
    double autocorrelation[50];   // Autocorrelation coefficients
    size_t num_lags;              // Number of lags computed
    double stationarity_pvalue;   // ADF test p-value
    bool is_stationary;           // Whether series is stationary
    double trend_slope;           // Linear trend slope
    double seasonality_strength;  // Seasonality measure (0-1)
} stat_timeseries_t;

/**
 * Histogram for empirical distribution
 */
typedef struct {
    double bin_edges[STAT_MAX_BINS + 1];
    size_t bin_counts[STAT_MAX_BINS];
    size_t num_bins;
    size_t total_count;
    double bin_width;
} stat_histogram_t;

/**
 * Analyzer configuration
 */
typedef struct {
    stat_confidence_level_t default_confidence;
    double significance_alpha;    // Default alpha (0.05)
    size_t bootstrap_iterations;  // Bootstrap resampling count
    bool use_exact_tests;         // Use exact tests when possible
    bool correct_multiple_comparisons; // Bonferroni, etc.
    size_t histogram_bins;        // Default histogram bins
    double outlier_threshold;     // IQR multiplier for outliers
} stat_analyzer_config_t;

/**
 * Opaque analyzer handle
 */
typedef struct stat_analyzer stat_analyzer_t;

// ============================================================================
// Initialization and Configuration
// ============================================================================

/**
 * Create statistical analyzer with default settings
 */
stat_analyzer_t* stat_analyzer_create(void);

/**
 * Create with custom configuration
 */
stat_analyzer_t* stat_analyzer_create_with_config(
    const stat_analyzer_config_t* config);

/**
 * Get default configuration
 */
stat_analyzer_config_t stat_analyzer_default_config(void);

/**
 * Destroy analyzer
 */
void stat_analyzer_destroy(stat_analyzer_t* analyzer);

/**
 * Reset all data
 */
bool stat_analyzer_reset(stat_analyzer_t* analyzer);

// ============================================================================
// Data Management
// ============================================================================

/**
 * Create a named data series
 */
bool stat_create_series(stat_analyzer_t* analyzer,
                        const char* name,
                        size_t max_samples);

/**
 * Add sample to a series
 */
bool stat_add_sample(stat_analyzer_t* analyzer,
                     const char* series_name,
                     double value);

/**
 * Add multiple samples
 */
bool stat_add_samples(stat_analyzer_t* analyzer,
                      const char* series_name,
                      const double* values,
                      size_t count);

/**
 * Get samples from a series
 */
bool stat_get_samples(stat_analyzer_t* analyzer,
                      const char* series_name,
                      double** values,
                      size_t* count);

/**
 * Clear a series
 */
bool stat_clear_series(stat_analyzer_t* analyzer,
                       const char* series_name);

// ============================================================================
// Descriptive Statistics
// ============================================================================

/**
 * Calculate descriptive statistics for a series
 */
bool stat_calculate_descriptive(stat_analyzer_t* analyzer,
                                const char* series_name,
                                stat_descriptive_t* stats);

/**
 * Calculate descriptive statistics for raw data
 */
bool stat_descriptive_from_data(const double* data,
                                size_t count,
                                stat_descriptive_t* stats);

/**
 * Calculate specific percentile
 */
double stat_percentile(const double* sorted_data,
                       size_t count,
                       double percentile);

/**
 * Detect outliers using IQR method
 */
bool stat_detect_outliers(const double* data,
                          size_t count,
                          double iqr_multiplier,
                          size_t** outlier_indices,
                          size_t* outlier_count);

// ============================================================================
// Confidence Intervals
// ============================================================================

/**
 * Calculate confidence interval for mean
 */
bool stat_confidence_interval_mean(stat_analyzer_t* analyzer,
                                   const char* series_name,
                                   stat_confidence_level_t level,
                                   stat_confidence_interval_t* ci);

/**
 * Calculate CI for proportion
 */
bool stat_confidence_interval_proportion(size_t successes,
                                          size_t total,
                                          stat_confidence_level_t level,
                                          stat_confidence_interval_t* ci);

/**
 * Calculate CI for variance
 */
bool stat_confidence_interval_variance(stat_analyzer_t* analyzer,
                                        const char* series_name,
                                        stat_confidence_level_t level,
                                        stat_confidence_interval_t* ci);

/**
 * Calculate CI using bootstrap
 */
bool stat_confidence_interval_bootstrap(const double* data,
                                        size_t count,
                                        double (*statistic)(const double*, size_t),
                                        stat_confidence_level_t level,
                                        size_t iterations,
                                        stat_confidence_interval_t* ci);

// ============================================================================
// Distribution Fitting
// ============================================================================

/**
 * Fit distribution to data
 */
bool stat_fit_distribution(stat_analyzer_t* analyzer,
                           const char* series_name,
                           stat_distribution_type_t type,
                           stat_distribution_t* result);

/**
 * Auto-detect best distribution
 */
bool stat_auto_fit_distribution(stat_analyzer_t* analyzer,
                                const char* series_name,
                                stat_distribution_t* result);

/**
 * Calculate probability from distribution
 */
double stat_distribution_pdf(const stat_distribution_t* dist, double x);
double stat_distribution_cdf(const stat_distribution_t* dist, double x);
double stat_distribution_quantile(const stat_distribution_t* dist, double p);

/**
 * Generate random sample from distribution
 */
double stat_distribution_sample(const stat_distribution_t* dist);

// ============================================================================
// Hypothesis Testing
// ============================================================================

/**
 * One-sample t-test
 */
bool stat_t_test_one_sample(stat_analyzer_t* analyzer,
                            const char* series_name,
                            double hypothesized_mean,
                            stat_hypothesis_result_t* result);

/**
 * Two-sample t-test
 */
bool stat_t_test_two_sample(stat_analyzer_t* analyzer,
                            const char* series1_name,
                            const char* series2_name,
                            bool paired,
                            stat_hypothesis_result_t* result);

/**
 * Chi-squared goodness of fit
 */
bool stat_chi_squared_test(const size_t* observed,
                            const double* expected,
                            size_t num_categories,
                            stat_hypothesis_result_t* result);

/**
 * Normality test (Shapiro-Wilk)
 */
bool stat_normality_test(const double* data,
                         size_t count,
                         stat_hypothesis_result_t* result);

/**
 * Kolmogorov-Smirnov test
 */
bool stat_ks_test(const double* data1,
                  size_t count1,
                  const double* data2,
                  size_t count2,
                  stat_hypothesis_result_t* result);

/**
 * F-test for variance equality
 */
bool stat_f_test(stat_analyzer_t* analyzer,
                 const char* series1_name,
                 const char* series2_name,
                 stat_hypothesis_result_t* result);

// ============================================================================
// Correlation Analysis
// ============================================================================

/**
 * Calculate correlation between two series
 */
bool stat_correlation(stat_analyzer_t* analyzer,
                      const char* series1_name,
                      const char* series2_name,
                      stat_correlation_type_t type,
                      stat_correlation_result_t* result);

/**
 * Calculate correlation from raw data
 */
bool stat_correlation_from_data(const double* x,
                                const double* y,
                                size_t count,
                                stat_correlation_type_t type,
                                stat_correlation_result_t* result);

/**
 * Autocorrelation function
 */
bool stat_autocorrelation(const double* data,
                          size_t count,
                          size_t max_lag,
                          double* acf);

/**
 * Cross-correlation function
 */
bool stat_cross_correlation(const double* x,
                            const double* y,
                            size_t count,
                            size_t max_lag,
                            double* ccf);

/**
 * Correlation matrix
 */
bool stat_correlation_matrix(stat_analyzer_t* analyzer,
                             const char** series_names,
                             size_t num_series,
                             double** matrix);

// ============================================================================
// Quantum Error Correction Statistics
// ============================================================================

/**
 * Track QEC error rates
 */
bool stat_track_qec_cycle(stat_analyzer_t* analyzer,
                          size_t syndrome_count,
                          size_t correction_count,
                          bool logical_error);

/**
 * Get QEC error rate statistics
 */
bool stat_get_error_rates(stat_analyzer_t* analyzer,
                          stat_error_rates_t* rates);

/**
 * Estimate error threshold
 */
double stat_estimate_threshold(stat_analyzer_t* analyzer,
                               const double* code_distances,
                               const double* logical_rates,
                               size_t num_points);

/**
 * Calculate error suppression factor
 */
double stat_error_suppression(double physical_rate,
                              double logical_rate,
                              size_t code_distance);

// ============================================================================
// Bayesian Inference
// ============================================================================

/**
 * Bayesian update for error rate (Beta-Binomial)
 */
bool stat_bayesian_error_rate(double prior_alpha,
                              double prior_beta,
                              size_t successes,
                              size_t failures,
                              stat_bayesian_result_t* result);

/**
 * Bayesian update for mean (Normal-Normal)
 */
bool stat_bayesian_mean(double prior_mean,
                        double prior_variance,
                        const double* data,
                        size_t count,
                        double known_variance,
                        stat_bayesian_result_t* result);

/**
 * Calculate Bayes factor
 */
double stat_bayes_factor(double likelihood_h1,
                         double likelihood_h0);

// ============================================================================
// Time Series Analysis
// ============================================================================

/**
 * Analyze time series properties
 */
bool stat_analyze_timeseries(const double* data,
                             size_t count,
                             stat_timeseries_t* result);

/**
 * Augmented Dickey-Fuller test for stationarity
 */
bool stat_adf_test(const double* data,
                   size_t count,
                   stat_hypothesis_result_t* result);

/**
 * Moving average
 */
bool stat_moving_average(const double* data,
                         size_t count,
                         size_t window,
                         double* result);

/**
 * Exponential smoothing
 */
bool stat_exponential_smoothing(const double* data,
                                size_t count,
                                double alpha,
                                double* result);

// ============================================================================
// Histogram and Empirical Distribution
// ============================================================================

/**
 * Create histogram from data
 */
bool stat_create_histogram(const double* data,
                           size_t count,
                           size_t num_bins,
                           stat_histogram_t* histogram);

/**
 * Get empirical CDF
 */
bool stat_empirical_cdf(const double* data,
                        size_t count,
                        double x,
                        double* cdf_value);

/**
 * Kernel density estimation
 */
bool stat_kernel_density(const double* data,
                         size_t count,
                         double bandwidth,
                         const double* eval_points,
                         size_t num_points,
                         double* density);

// ============================================================================
// Export and Reporting
// ============================================================================

/**
 * Export analysis to JSON
 */
char* stat_export_json(stat_analyzer_t* analyzer);

/**
 * Export to file
 */
bool stat_export_to_file(stat_analyzer_t* analyzer,
                         const char* filename);

/**
 * Generate human-readable report
 */
char* stat_generate_report(stat_analyzer_t* analyzer);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Get distribution type name
 */
const char* stat_distribution_name(stat_distribution_type_t type);

/**
 * Get test type name
 */
const char* stat_test_name(stat_test_type_t type);

/**
 * Get correlation type name
 */
const char* stat_correlation_name(stat_correlation_type_t type);

/**
 * Get confidence level value
 */
double stat_confidence_value(stat_confidence_level_t level);

/**
 * Get critical value for z-distribution
 */
double stat_z_critical(double alpha, bool two_tailed);

/**
 * Get critical value for t-distribution
 */
double stat_t_critical(double alpha, size_t df, bool two_tailed);

/**
 * Get last error
 */
const char* stat_get_last_error(stat_analyzer_t* analyzer);

/**
 * Free allocated memory
 */
void stat_free_samples(double* samples);
void stat_free_indices(size_t* indices);
void stat_free_matrix(double* matrix, size_t rows);

#ifdef __cplusplus
}
#endif

#endif // STATISTICAL_ANALYZER_H
