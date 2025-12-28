/**
 * @file stability_analyzer.h
 * @brief System Stability Analysis for Quantum Operations
 *
 * Provides stability analysis including:
 * - Numerical stability monitoring
 * - Convergence tracking
 * - Oscillation detection
 * - Drift analysis
 * - Steady-state detection
 *
 * Part of the QGTL Monitoring Framework.
 */

#ifndef STABILITY_ANALYZER_H
#define STABILITY_ANALYZER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================

#define STABILITY_MAX_HISTORY 10000
#define STABILITY_DEFAULT_WINDOW 100

// ============================================================================
// Enumerations
// ============================================================================

typedef enum {
    STABILITY_STATE_UNSTABLE,
    STABILITY_STATE_OSCILLATING,
    STABILITY_STATE_CONVERGING,
    STABILITY_STATE_STABLE,
    STABILITY_STATE_STEADY_STATE
} stability_state_t;

typedef enum {
    STABILITY_METRIC_VALUE,
    STABILITY_METRIC_GRADIENT,
    STABILITY_METRIC_EIGENVALUE,
    STABILITY_METRIC_FIDELITY,
    STABILITY_METRIC_ENTROPY,
    STABILITY_METRIC_CUSTOM
} stability_metric_t;

// ============================================================================
// Data Structures
// ============================================================================

typedef struct {
    stability_state_t state;
    double stability_score;           // 0.0 to 1.0
    double convergence_rate;
    double oscillation_amplitude;
    double oscillation_frequency;
    double drift_rate;
    uint64_t time_to_stability_ns;
    bool is_numerically_stable;
} stability_assessment_t;

typedef struct {
    double current_value;
    double mean;
    double variance;
    double min;
    double max;
    double trend;
    size_t sample_count;
} stability_metric_stats_t;

typedef struct {
    size_t window_size;
    double convergence_threshold;
    double oscillation_threshold;
    double drift_threshold;
    bool enable_trend_detection;
} stability_analyzer_config_t;

typedef struct stability_analyzer stability_analyzer_t;

// ============================================================================
// API Functions
// ============================================================================

stability_analyzer_t* stability_analyzer_create(void);
stability_analyzer_t* stability_analyzer_create_with_config(
    const stability_analyzer_config_t* config);
stability_analyzer_config_t stability_analyzer_default_config(void);
void stability_analyzer_destroy(stability_analyzer_t* analyzer);
bool stability_analyzer_reset(stability_analyzer_t* analyzer);

bool stability_record_sample(stability_analyzer_t* analyzer,
                              const char* metric_name,
                              double value);

bool stability_record_vector(stability_analyzer_t* analyzer,
                              const char* metric_name,
                              const double* values,
                              size_t count);

bool stability_assess(stability_analyzer_t* analyzer,
                       const char* metric_name,
                       stability_assessment_t* assessment);

bool stability_check_convergence(stability_analyzer_t* analyzer,
                                  const char* metric_name,
                                  double threshold,
                                  bool* converged);

bool stability_detect_oscillation(stability_analyzer_t* analyzer,
                                   const char* metric_name,
                                   double* amplitude,
                                   double* frequency);

bool stability_get_metric_stats(stability_analyzer_t* analyzer,
                                 const char* metric_name,
                                 stability_metric_stats_t* stats);

char* stability_generate_report(stability_analyzer_t* analyzer);
char* stability_export_json(stability_analyzer_t* analyzer);
bool stability_export_to_file(stability_analyzer_t* analyzer,
                               const char* filename);

const char* stability_state_name(stability_state_t state);
const char* stability_metric_name(stability_metric_t metric);
const char* stability_get_last_error(stability_analyzer_t* analyzer);

#ifdef __cplusplus
}
#endif

#endif // STABILITY_ANALYZER_H
