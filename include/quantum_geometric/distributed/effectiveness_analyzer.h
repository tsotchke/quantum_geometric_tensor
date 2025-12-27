#ifndef EFFECTIVENESS_ANALYZER_H
#define EFFECTIVENESS_ANALYZER_H

/**
 * @file effectiveness_analyzer.h
 * @brief Effectiveness analysis for optimization suggestions
 *
 * Provides impact pattern detection, regression analysis, and
 * correlation analysis for tracking suggestion effectiveness.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

// Metric types for analysis
typedef enum {
    METRIC_LATENCY,
    METRIC_THROUGHPUT,
    METRIC_MEMORY,
    METRIC_CPU_USAGE,
    METRIC_GPU_USAGE,
    METRIC_NETWORK_BANDWIDTH,
    METRIC_CACHE_HIT_RATE,
    METRIC_ERROR_RATE,
    NUM_METRIC_TYPES
} MetricType;

// Effectiveness suggestion types (different from optimization_suggester types)
typedef enum {
    EFF_SUGGESTION_OPTIMIZE_MEMORY,
    EFF_SUGGESTION_REDUCE_LATENCY,
    EFF_SUGGESTION_IMPROVE_THROUGHPUT,
    EFF_SUGGESTION_BALANCE_LOAD,
    EFF_SUGGESTION_CACHE_OPTIMIZATION,
    EFF_SUGGESTION_BATCH_SIZE,
    EFF_SUGGESTION_PARALLELISM,
    EFF_SUGGESTION_COMPRESSION,
    NUM_EFF_SUGGESTION_TYPES
} EffSuggestionType;

// Optimization suggestion for effectiveness analysis
typedef struct {
    EffSuggestionType type;
    double priority;
    double expected_improvement;
    char* description;
    void* context;
} EffOptimizationSuggestion;

// Impact measurement
typedef struct {
    time_t timestamp;
    double value;
    MetricType metric;
    bool is_positive;
} ImpactMeasurement;

// Tracked suggestion with impact history
typedef struct {
    EffOptimizationSuggestion suggestion;
    ImpactMeasurement* impacts;
    size_t num_impacts;
    size_t capacity;
    double cumulative_impact;
    bool is_active;
} TrackedSuggestion;

// Pattern finding
typedef struct {
    char* description;
    double confidence;
    EffSuggestionType related_type;
} PatternFinding;

// Regression finding
typedef struct {
    MetricType metric;
    double severity;
    char* likely_cause;
    EffSuggestionType related_type;
} RegressionFinding;

// Correlation finding
typedef struct {
    EffSuggestionType type1;
    EffSuggestionType type2;
    double correlation;
    bool is_positive;
} CorrelationFinding;

// Effectiveness report
typedef struct {
    PatternFinding* patterns;
    size_t num_patterns;
    RegressionFinding* regressions;
    size_t num_regressions;
    CorrelationFinding* correlations;
    size_t num_correlations;
    double overall_effectiveness;
    char** recommendations;
    size_t num_recommendations;
} EffectivenessReport;

// Forward declarations for opaque types
typedef struct EffectivenessAnalyzerImpl EffectivenessAnalyzer;
typedef struct EffMLModelImpl EffMLModel;
typedef struct EffInsightGeneratorImpl EffInsightGenerator;
typedef struct EffImpactPatternImpl EffImpactPattern;
typedef struct EffRegressionImpl EffRegression;

// Analyzer lifecycle
EffectivenessAnalyzer* init_effectiveness_analyzer(void);
void cleanup_effectiveness_analyzer(EffectivenessAnalyzer* analyzer);

// Analysis functions
void analyze_impact_patterns(EffectivenessAnalyzer* analyzer,
                            const TrackedSuggestion* suggestion,
                            EffectivenessReport* report);
void check_for_regressions(EffectivenessAnalyzer* analyzer,
                          const TrackedSuggestion* suggestion,
                          EffectivenessReport* report);
void analyze_correlations(EffectivenessAnalyzer* analyzer,
                         const TrackedSuggestion** suggestions,
                         size_t num_suggestions,
                         EffectivenessReport* report);

// Helper functions
double* extract_impact_values(const TrackedSuggestion* suggestion);
double* extract_metric_values(const TrackedSuggestion* suggestion, MetricType metric);
EffImpactPattern* detect_impact_pattern(EffMLModel* model, double* values, size_t num_values);
void store_impact_pattern(EffectivenessAnalyzer* analyzer, EffImpactPattern* pattern);
void add_pattern_finding(EffectivenessReport* report, EffImpactPattern* pattern, EffSuggestionType type);
void analyze_impact_trend(EffectivenessAnalyzer* analyzer, double* values, size_t num_values, EffectivenessReport* report);
EffRegression* detect_regression(double* values, size_t num_values, MetricType metric);
void store_regression(EffectivenessAnalyzer* analyzer, EffRegression* regression);
void add_regression_finding(EffectivenessReport* report, EffRegression* regression, EffSuggestionType type);
void generate_regression_mitigations(EffectivenessAnalyzer* analyzer, EffRegression* regression, EffectivenessReport* report);

// ML model functions (eff_ prefix to avoid conflict with pattern_recognition_model.c)
EffMLModel* eff_init_pattern_recognition_model(void);
void eff_cleanup_pattern_recognition_model(EffMLModel* model);

// Insight generator functions (eff_ prefix to avoid conflict with insight_generator.c)
EffInsightGenerator* eff_init_insight_generator(void);
void eff_cleanup_insight_generator(EffInsightGenerator* generator);

// Report functions
EffectivenessReport* create_effectiveness_report(void);
void cleanup_effectiveness_report(EffectivenessReport* report);

#ifdef __cplusplus
}
#endif

#endif // EFFECTIVENESS_ANALYZER_H
