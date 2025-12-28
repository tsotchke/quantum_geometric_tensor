/**
 * @file robustness_analyzer.h
 * @brief System Robustness Analysis for Quantum Operations
 *
 * Provides robustness analysis including:
 * - Error tolerance measurement
 * - Noise resilience evaluation
 * - Fault tolerance metrics
 * - Recovery capability assessment
 * - Stability under perturbation
 *
 * Part of the QGTL Monitoring Framework.
 */

#ifndef ROBUSTNESS_ANALYZER_H
#define ROBUSTNESS_ANALYZER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================

#define ROBUSTNESS_MAX_TESTS 256
#define ROBUSTNESS_MAX_NAME_LENGTH 128

// ============================================================================
// Enumerations
// ============================================================================

typedef enum {
    ROBUSTNESS_TEST_NOISE_INJECTION,
    ROBUSTNESS_TEST_BIT_FLIP,
    ROBUSTNESS_TEST_PHASE_FLIP,
    ROBUSTNESS_TEST_DEPOLARIZATION,
    ROBUSTNESS_TEST_AMPLITUDE_DAMPING,
    ROBUSTNESS_TEST_PARAMETER_PERTURBATION,
    ROBUSTNESS_TEST_TIMING_VARIATION,
    ROBUSTNESS_TEST_RESOURCE_CONSTRAINT,
    ROBUSTNESS_TEST_CUSTOM
} robustness_test_type_t;

typedef enum {
    ROBUSTNESS_LEVEL_FRAGILE,         // Fails easily
    ROBUSTNESS_LEVEL_SENSITIVE,       // Degrades quickly
    ROBUSTNESS_LEVEL_MODERATE,        // Some tolerance
    ROBUSTNESS_LEVEL_ROBUST,          // Good tolerance
    ROBUSTNESS_LEVEL_RESILIENT        // Excellent recovery
} robustness_level_t;

// ============================================================================
// Data Structures
// ============================================================================

typedef struct {
    robustness_test_type_t type;
    char name[ROBUSTNESS_MAX_NAME_LENGTH];
    double error_magnitude;
    double success_rate;
    double recovery_time_ns;
    double fidelity_after;
    bool passed;
} robustness_test_result_t;

typedef struct {
    double overall_score;             // 0.0 to 1.0
    double noise_tolerance;
    double fault_tolerance;
    double recovery_capability;
    robustness_level_t level;
    size_t tests_passed;
    size_t tests_failed;
    double mean_fidelity_degradation;
} robustness_metrics_t;

typedef struct {
    bool enable_noise_testing;
    bool enable_fault_injection;
    double max_error_magnitude;
    size_t num_trials_per_test;
    double fidelity_threshold;
} robustness_analyzer_config_t;

typedef struct robustness_analyzer robustness_analyzer_t;

// ============================================================================
// API Functions
// ============================================================================

robustness_analyzer_t* robustness_analyzer_create(void);
robustness_analyzer_t* robustness_analyzer_create_with_config(
    const robustness_analyzer_config_t* config);
robustness_analyzer_config_t robustness_analyzer_default_config(void);
void robustness_analyzer_destroy(robustness_analyzer_t* analyzer);
bool robustness_analyzer_reset(robustness_analyzer_t* analyzer);

bool robustness_add_test(robustness_analyzer_t* analyzer,
                         robustness_test_type_t type,
                         const char* name,
                         double error_magnitude);

bool robustness_run_tests(robustness_analyzer_t* analyzer,
                          const void* system_state,
                          size_t state_size);

bool robustness_get_metrics(robustness_analyzer_t* analyzer,
                            robustness_metrics_t* metrics);

bool robustness_get_test_results(robustness_analyzer_t* analyzer,
                                  robustness_test_result_t** results,
                                  size_t* count);

char* robustness_generate_report(robustness_analyzer_t* analyzer);
char* robustness_export_json(robustness_analyzer_t* analyzer);
bool robustness_export_to_file(robustness_analyzer_t* analyzer,
                                const char* filename);

const char* robustness_test_type_name(robustness_test_type_t type);
const char* robustness_level_name(robustness_level_t level);
void robustness_free_results(robustness_test_result_t* results, size_t count);
const char* robustness_get_last_error(robustness_analyzer_t* analyzer);

#ifdef __cplusplus
}
#endif

#endif // ROBUSTNESS_ANALYZER_H
