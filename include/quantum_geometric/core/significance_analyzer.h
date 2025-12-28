/**
 * @file significance_analyzer.h
 * @brief Statistical Significance Analysis for Quantum Experiments
 *
 * Provides significance testing including:
 * - Hypothesis testing (t-test, chi-square, etc.)
 * - Effect size calculation
 * - Power analysis
 * - Multiple comparison corrections
 * - Confidence level assessment
 *
 * Part of the QGTL Monitoring Framework.
 */

#ifndef SIGNIFICANCE_ANALYZER_H
#define SIGNIFICANCE_ANALYZER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================

#define SIGNIFICANCE_MAX_SAMPLES 100000
#define SIGNIFICANCE_DEFAULT_ALPHA 0.05

// ============================================================================
// Enumerations
// ============================================================================

typedef enum {
    SIGNIFICANCE_TEST_T_TEST,
    SIGNIFICANCE_TEST_PAIRED_T,
    SIGNIFICANCE_TEST_WELCH_T,
    SIGNIFICANCE_TEST_CHI_SQUARE,
    SIGNIFICANCE_TEST_FISHER_EXACT,
    SIGNIFICANCE_TEST_MANN_WHITNEY,
    SIGNIFICANCE_TEST_WILCOXON,
    SIGNIFICANCE_TEST_ANOVA,
    SIGNIFICANCE_TEST_KRUSKAL_WALLIS,
    SIGNIFICANCE_TEST_KOLMOGOROV_SMIRNOV
} significance_test_type_t;

typedef enum {
    SIGNIFICANCE_ALTERNATIVE_TWO_SIDED,
    SIGNIFICANCE_ALTERNATIVE_GREATER,
    SIGNIFICANCE_ALTERNATIVE_LESS
} significance_alternative_t;

typedef enum {
    SIGNIFICANCE_CORRECTION_NONE,
    SIGNIFICANCE_CORRECTION_BONFERRONI,
    SIGNIFICANCE_CORRECTION_HOLM,
    SIGNIFICANCE_CORRECTION_BH_FDR,
    SIGNIFICANCE_CORRECTION_BY_FDR
} significance_correction_t;

// ============================================================================
// Data Structures
// ============================================================================

typedef struct {
    significance_test_type_t test_type;
    significance_alternative_t alternative;
    double test_statistic;
    double p_value;
    double effect_size;
    double confidence_interval_low;
    double confidence_interval_high;
    double power;
    size_t degrees_of_freedom;
    bool is_significant;
    double alpha;
} significance_result_t;

typedef struct {
    double cohens_d;
    double hedges_g;
    double glass_delta;
    double pearsons_r;
    double eta_squared;
    double omega_squared;
} effect_size_t;

typedef struct {
    double required_n;
    double achieved_power;
    double effect_size;
    double alpha;
    double beta;
} power_analysis_t;

typedef struct {
    double default_alpha;
    significance_correction_t correction;
    bool compute_effect_size;
    bool compute_power;
    size_t bootstrap_samples;
} significance_analyzer_config_t;

typedef struct significance_analyzer significance_analyzer_t;

// ============================================================================
// API Functions
// ============================================================================

significance_analyzer_t* significance_analyzer_create(void);
significance_analyzer_t* significance_analyzer_create_with_config(
    const significance_analyzer_config_t* config);
significance_analyzer_config_t significance_analyzer_default_config(void);
void significance_analyzer_destroy(significance_analyzer_t* analyzer);
bool significance_analyzer_reset(significance_analyzer_t* analyzer);

bool significance_t_test(significance_analyzer_t* analyzer,
                         const double* sample1, size_t n1,
                         const double* sample2, size_t n2,
                         significance_alternative_t alt,
                         significance_result_t* result);

bool significance_paired_t_test(significance_analyzer_t* analyzer,
                                 const double* sample1,
                                 const double* sample2,
                                 size_t n,
                                 significance_alternative_t alt,
                                 significance_result_t* result);

bool significance_one_sample_t(significance_analyzer_t* analyzer,
                                const double* sample, size_t n,
                                double mu,
                                significance_alternative_t alt,
                                significance_result_t* result);

bool significance_chi_square(significance_analyzer_t* analyzer,
                              const double* observed,
                              const double* expected,
                              size_t n,
                              significance_result_t* result);

bool significance_mann_whitney(significance_analyzer_t* analyzer,
                                const double* sample1, size_t n1,
                                const double* sample2, size_t n2,
                                significance_alternative_t alt,
                                significance_result_t* result);

bool significance_ks_test(significance_analyzer_t* analyzer,
                           const double* sample1, size_t n1,
                           const double* sample2, size_t n2,
                           significance_result_t* result);

bool significance_calculate_effect_size(significance_analyzer_t* analyzer,
                                         const double* sample1, size_t n1,
                                         const double* sample2, size_t n2,
                                         effect_size_t* effect);

bool significance_power_analysis(significance_analyzer_t* analyzer,
                                  double effect_size,
                                  double alpha,
                                  double power,
                                  power_analysis_t* result);

bool significance_apply_correction(significance_analyzer_t* analyzer,
                                    double* p_values,
                                    size_t n,
                                    significance_correction_t correction);

char* significance_generate_report(significance_analyzer_t* analyzer,
                                    const significance_result_t* result);
const char* significance_test_name(significance_test_type_t type);
const char* significance_correction_name(significance_correction_t correction);
const char* significance_get_last_error(significance_analyzer_t* analyzer);

#ifdef __cplusplus
}
#endif

#endif // SIGNIFICANCE_ANALYZER_H
