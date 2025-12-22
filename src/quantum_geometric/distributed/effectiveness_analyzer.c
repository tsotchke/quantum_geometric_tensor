#include "quantum_geometric/distributed/effectiveness_analyzer.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

// Analysis parameters
#define MIN_PATTERN_LENGTH 10
#define MAX_PATTERNS_INTERNAL 100
#define REGRESSION_THRESHOLD -0.05
#define CORRELATION_THRESHOLD 0.7
#define MAX_INSIGHTS 50
#define MIN_VARIANCE 1e-10

// ============================================================================
// Internal struct definitions (must come before any usage)
// ============================================================================

// Impact pattern implementation
struct EffImpactPatternImpl {
    double* values;
    size_t length;
    double mean;
    double std_dev;
    bool is_significant;
    PatternFinding finding;
};

// Regression implementation
struct EffRegressionImpl {
    time_t detection_time;
    double severity;
    char* likely_cause;
    bool is_confirmed;
    MetricType affected_metric;
};

// Effectiveness correlation (internal type)
typedef struct {
    EffSuggestionType type1;
    EffSuggestionType type2;
    double correlation;
    bool is_positive;
    double confidence;
} EffectivenessCorrelation;

// ML Model implementation
struct EffMLModelImpl {
    double* weights;
    size_t num_weights;
    double learning_rate;
    bool is_trained;
};

// Insight generator implementation
struct EffInsightGeneratorImpl {
    char** insights;
    size_t num_insights;
    size_t capacity;
};

// Effectiveness analyzer implementation
struct EffectivenessAnalyzerImpl {
    // Pattern analysis
    EffImpactPattern** patterns;
    size_t num_patterns;
    size_t pattern_capacity;

    // Regression tracking
    EffRegression* regressions;
    size_t num_regressions;
    size_t regression_capacity;

    // Correlation analysis
    EffectivenessCorrelation* correlations;
    size_t num_correlations;
    size_t correlation_capacity;

    // ML model
    EffMLModel* pattern_model;

    // Insight generation
    EffInsightGenerator* insight_generator;
};

// ============================================================================
// Forward declarations for static functions
// ============================================================================

static double compute_mean(const double* values, size_t length);
static double compute_std_dev(const double* values, size_t length, double mean);
static void cleanup_impact_pattern(EffImpactPattern* pattern);
static void cleanup_regression_internal(EffRegression* regression);
static void cleanup_ml_model(EffMLModel* model);
static void store_correlation(EffectivenessAnalyzer* analyzer, const EffectivenessCorrelation* corr);
static EffectivenessCorrelation compute_effectiveness_correlation(const TrackedSuggestion* s1, const TrackedSuggestion* s2);
static void update_correlation_model(EffMLModel* model, const EffectivenessCorrelation* correlations, size_t num);
static void generate_pattern_insights(EffInsightGenerator* gen, EffImpactPattern** patterns, size_t num, EffectivenessReport* report);
static void generate_correlation_insights(EffInsightGenerator* gen, const EffectivenessCorrelation* correlations, size_t num, EffectivenessReport* report);
static void generate_trend_insights(EffInsightGenerator* gen, const TrackedSuggestion* suggestion, EffectivenessReport* report);
static void prioritize_insights(EffectivenessReport* report);

// ============================================================================
// Public API implementations
// ============================================================================

// Initialize effectiveness analyzer
EffectivenessAnalyzer* init_effectiveness_analyzer(void) {
    EffectivenessAnalyzer* analyzer = calloc(1, sizeof(struct EffectivenessAnalyzerImpl));
    if (!analyzer) return NULL;

    // Initialize pattern storage
    analyzer->pattern_capacity = MAX_PATTERNS_INTERNAL;
    analyzer->patterns = calloc(MAX_PATTERNS_INTERNAL, sizeof(EffImpactPattern*));
    analyzer->num_patterns = 0;

    // Initialize regression tracking
    analyzer->regression_capacity = 100;
    analyzer->regressions = calloc(100, sizeof(struct EffRegressionImpl));
    analyzer->num_regressions = 0;

    // Initialize correlation analysis
    analyzer->correlation_capacity = 100;
    analyzer->correlations = calloc(100, sizeof(EffectivenessCorrelation));
    analyzer->num_correlations = 0;

    // Initialize ML model
    analyzer->pattern_model = init_pattern_recognition_model();

    // Initialize insight generator
    analyzer->insight_generator = init_insight_generator();

    return analyzer;
}

// Analyze impact patterns
void analyze_impact_patterns(
    EffectivenessAnalyzer* analyzer,
    const TrackedSuggestion* suggestion,
    EffectivenessReport* report) {

    if (!analyzer || !suggestion || !report) return;

    // Extract impact values
    double* impact_values = extract_impact_values(suggestion);
    size_t num_values = suggestion->num_impacts;

    if (num_values < MIN_PATTERN_LENGTH) {
        free(impact_values);
        return;
    }

    // Detect patterns using ML
    EffImpactPattern* pattern = detect_impact_pattern(
        analyzer->pattern_model,
        impact_values,
        num_values);

    if (pattern && pattern->is_significant) {
        // Store pattern
        store_impact_pattern(analyzer, pattern);

        // Add to report
        add_pattern_finding(report, pattern, suggestion->suggestion.type);
    }

    // Analyze trend
    analyze_impact_trend(analyzer, impact_values, num_values, report);

    free(impact_values);
}

// Check for performance regressions
void check_for_regressions(
    EffectivenessAnalyzer* analyzer,
    const TrackedSuggestion* suggestion,
    EffectivenessReport* report) {

    if (!analyzer || !suggestion || !report) return;

    // Check each metric type
    for (int i = 0; i < NUM_METRIC_TYPES; i++) {
        MetricType metric = (MetricType)i;

        // Get metric values
        double* values = extract_metric_values(suggestion, metric);
        size_t num_values = suggestion->num_impacts;

        // Detect regression
        EffRegression* regression = detect_regression(values, num_values, metric);

        if (regression) {
            // Store regression
            store_regression(analyzer, regression);

            // Add to report
            add_regression_finding(report, regression, suggestion->suggestion.type);

            // Generate mitigation suggestions
            generate_regression_mitigations(analyzer, regression, report);
        }

        free(values);
    }
}

// Analyze effectiveness correlations
void analyze_correlations(
    EffectivenessAnalyzer* analyzer,
    const TrackedSuggestion** suggestions,
    size_t num_suggestions,
    EffectivenessReport* report) {

    if (!analyzer || !suggestions) return;
    (void)report;

    // Reset correlations
    analyzer->num_correlations = 0;

    // Analyze each pair of suggestion types
    for (size_t i = 0; i < num_suggestions; i++) {
        for (size_t j = i + 1; j < num_suggestions; j++) {
            // Compute correlation
            EffectivenessCorrelation correlation =
                compute_effectiveness_correlation(suggestions[i], suggestions[j]);

            // Store if significant
            if (fabs(correlation.correlation) > CORRELATION_THRESHOLD) {
                store_correlation(analyzer, &correlation);
            }
        }
    }

    // Update ML model
    update_correlation_model(analyzer->pattern_model,
                           analyzer->correlations,
                           analyzer->num_correlations);
}

// Clean up effectiveness analyzer
void cleanup_effectiveness_analyzer(EffectivenessAnalyzer* analyzer) {
    if (!analyzer) return;

    // Clean up patterns
    for (size_t i = 0; i < analyzer->num_patterns; i++) {
        cleanup_impact_pattern(analyzer->patterns[i]);
    }
    free(analyzer->patterns);

    // Clean up regressions
    for (size_t i = 0; i < analyzer->num_regressions; i++) {
        cleanup_regression_internal(&analyzer->regressions[i]);
    }
    free(analyzer->regressions);

    // Clean up correlations
    free(analyzer->correlations);

    // Clean up ML model
    cleanup_ml_model(analyzer->pattern_model);

    // Clean up insight generator
    cleanup_insight_generator(analyzer->insight_generator);

    free(analyzer);
}

// ============================================================================
// Static helper function implementations
// ============================================================================

static double compute_mean(const double* values, size_t length) {
    if (!values || length == 0) return 0.0;

    double sum = 0.0;
    for (size_t i = 0; i < length; i++) {
        sum += values[i];
    }
    return sum / (double)length;
}

static double compute_std_dev(const double* values, size_t length, double mean) {
    if (!values || length < 2) return 0.0;

    double sum_sq = 0.0;
    for (size_t i = 0; i < length; i++) {
        double diff = values[i] - mean;
        sum_sq += diff * diff;
    }
    return sqrt(sum_sq / (double)(length - 1));
}

static void cleanup_impact_pattern(EffImpactPattern* pattern) {
    if (!pattern) return;
    free(pattern->values);
    free(pattern->finding.description);
    free(pattern);
}

static void cleanup_regression_internal(EffRegression* regression) {
    if (!regression) return;
    free(regression->likely_cause);
    regression->likely_cause = NULL;
}

static void cleanup_ml_model(EffMLModel* model) {
    if (!model) return;
    free(model->weights);
    free(model);
}

static void store_correlation(EffectivenessAnalyzer* analyzer, const EffectivenessCorrelation* corr) {
    if (!analyzer || !corr) return;
    if (analyzer->num_correlations >= analyzer->correlation_capacity) return;

    analyzer->correlations[analyzer->num_correlations++] = *corr;
}

static EffectivenessCorrelation compute_effectiveness_correlation(
    const TrackedSuggestion* s1, const TrackedSuggestion* s2) {

    EffectivenessCorrelation corr = {0};
    if (!s1 || !s2) return corr;

    corr.type1 = s1->suggestion.type;
    corr.type2 = s2->suggestion.type;

    // Compute Pearson correlation between impact values
    size_t n = (s1->num_impacts < s2->num_impacts) ? s1->num_impacts : s2->num_impacts;
    if (n < 2) {
        corr.correlation = 0.0;
        corr.confidence = 0.0;
        return corr;
    }

    double sum1 = 0.0, sum2 = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum1 += s1->impacts[i].value;
        sum2 += s2->impacts[i].value;
    }
    double mean1 = sum1 / (double)n;
    double mean2 = sum2 / (double)n;

    double cov = 0.0, var1 = 0.0, var2 = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d1 = s1->impacts[i].value - mean1;
        double d2 = s2->impacts[i].value - mean2;
        cov += d1 * d2;
        var1 += d1 * d1;
        var2 += d2 * d2;
    }

    if (var1 > MIN_VARIANCE && var2 > MIN_VARIANCE) {
        corr.correlation = cov / sqrt(var1 * var2);
    } else {
        corr.correlation = 0.0;
    }

    corr.is_positive = (corr.correlation > 0);
    corr.confidence = (double)n / 100.0;
    if (corr.confidence > 1.0) corr.confidence = 1.0;

    return corr;
}

static void update_correlation_model(EffMLModel* model, const EffectivenessCorrelation* correlations, size_t num) {
    if (!model || !correlations || num == 0) return;
    model->is_trained = true;
}

static void generate_pattern_insights(EffInsightGenerator* gen, EffImpactPattern** patterns,
                                       size_t num, EffectivenessReport* report) {
    if (!gen || !patterns || !report || num == 0) return;
}

static void generate_correlation_insights(EffInsightGenerator* gen,
                                           const EffectivenessCorrelation* correlations,
                                           size_t num, EffectivenessReport* report) {
    if (!gen || !correlations || !report || num == 0) return;
}

static void generate_trend_insights(EffInsightGenerator* gen,
                                     const TrackedSuggestion* suggestion,
                                     EffectivenessReport* report) {
    if (!gen || !suggestion || !report) return;
}

static void prioritize_insights(EffectivenessReport* report) {
    if (!report || !report->recommendations || report->num_recommendations == 0) return;
}

// ============================================================================
// Public helper function implementations
// ============================================================================

// Initialize pattern recognition model
EffMLModel* init_pattern_recognition_model(void) {
    EffMLModel* model = calloc(1, sizeof(struct EffMLModelImpl));
    if (!model) return NULL;

    model->num_weights = 64;
    model->weights = calloc(model->num_weights, sizeof(double));
    model->learning_rate = 0.01;
    model->is_trained = false;

    return model;
}

// Cleanup pattern recognition model
void cleanup_pattern_recognition_model(EffMLModel* model) {
    cleanup_ml_model(model);
}

// Initialize insight generator
EffInsightGenerator* init_insight_generator(void) {
    EffInsightGenerator* gen = calloc(1, sizeof(struct EffInsightGeneratorImpl));
    if (!gen) return NULL;

    gen->capacity = MAX_INSIGHTS;
    gen->insights = calloc(MAX_INSIGHTS, sizeof(char*));
    gen->num_insights = 0;

    return gen;
}

// Cleanup insight generator
void cleanup_insight_generator(EffInsightGenerator* generator) {
    if (!generator) return;

    for (size_t i = 0; i < generator->num_insights; i++) {
        free(generator->insights[i]);
    }
    free(generator->insights);
    free(generator);
}

// Extract impact values from tracked suggestion
double* extract_impact_values(const TrackedSuggestion* suggestion) {
    if (!suggestion || suggestion->num_impacts == 0) return NULL;

    double* values = calloc(suggestion->num_impacts, sizeof(double));
    if (!values) return NULL;

    for (size_t i = 0; i < suggestion->num_impacts; i++) {
        values[i] = suggestion->impacts[i].value;
    }

    return values;
}

// Extract metric values from tracked suggestion
double* extract_metric_values(const TrackedSuggestion* suggestion, MetricType metric) {
    if (!suggestion || suggestion->num_impacts == 0) return NULL;

    double* values = calloc(suggestion->num_impacts, sizeof(double));
    if (!values) return NULL;

    size_t count = 0;
    for (size_t i = 0; i < suggestion->num_impacts; i++) {
        if (suggestion->impacts[i].metric == metric) {
            values[count++] = suggestion->impacts[i].value;
        }
    }

    return values;
}

// Detect impact pattern using ML model
EffImpactPattern* detect_impact_pattern(EffMLModel* model, double* values, size_t num_values) {
    if (!model || !values || num_values < MIN_PATTERN_LENGTH) return NULL;

    EffImpactPattern* pattern = calloc(1, sizeof(struct EffImpactPatternImpl));
    if (!pattern) return NULL;

    pattern->length = num_values;
    pattern->values = calloc(num_values, sizeof(double));
    if (!pattern->values) {
        free(pattern);
        return NULL;
    }
    memcpy(pattern->values, values, num_values * sizeof(double));

    pattern->mean = compute_mean(values, num_values);
    pattern->std_dev = compute_std_dev(values, num_values, pattern->mean);

    // Pattern is significant if std_dev is not too small
    pattern->is_significant = (pattern->std_dev > MIN_VARIANCE);

    return pattern;
}

// Store impact pattern
void store_impact_pattern(EffectivenessAnalyzer* analyzer, EffImpactPattern* pattern) {
    if (!analyzer || !pattern) return;
    if (analyzer->num_patterns >= analyzer->pattern_capacity) {
        cleanup_impact_pattern(pattern);
        return;
    }

    analyzer->patterns[analyzer->num_patterns++] = pattern;
}

// Add pattern finding to report
void add_pattern_finding(EffectivenessReport* report, EffImpactPattern* pattern, EffSuggestionType type) {
    if (!report || !pattern) return;

    PatternFinding* new_patterns = realloc(report->patterns,
                                            (report->num_patterns + 1) * sizeof(PatternFinding));
    if (!new_patterns) return;

    report->patterns = new_patterns;

    PatternFinding* finding = &report->patterns[report->num_patterns];
    finding->confidence = pattern->is_significant ? 0.8 : 0.3;
    finding->related_type = type;
    finding->description = strdup("Pattern detected in impact values");

    report->num_patterns++;
}

// Analyze impact trend
void analyze_impact_trend(EffectivenessAnalyzer* analyzer, double* values,
                          size_t num_values, EffectivenessReport* report) {
    if (!analyzer || !values || !report || num_values < MIN_PATTERN_LENGTH) return;

    // Compute linear regression slope
    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_xx = 0.0;
    for (size_t i = 0; i < num_values; i++) {
        double x = (double)i;
        sum_x += x;
        sum_y += values[i];
        sum_xy += x * values[i];
        sum_xx += x * x;
    }

    double n = (double)num_values;
    double denom = n * sum_xx - sum_x * sum_x;
    if (fabs(denom) < MIN_VARIANCE) return;

    double slope = (n * sum_xy - sum_x * sum_y) / denom;
    (void)slope;  // Would add trend finding to report
}

// Detect regression in metric values
EffRegression* detect_regression(double* values, size_t num_values, MetricType metric) {
    if (!values || num_values < MIN_PATTERN_LENGTH) return NULL;

    // Compare recent values to earlier values
    size_t half = num_values / 2;
    double early_mean = compute_mean(values, half);
    double late_mean = compute_mean(values + half, num_values - half);

    double change = late_mean - early_mean;

    // Detect regression if significant negative change
    if (change < REGRESSION_THRESHOLD) {
        EffRegression* reg = calloc(1, sizeof(struct EffRegressionImpl));
        if (!reg) return NULL;

        reg->detection_time = time(NULL);
        reg->severity = fabs(change);
        reg->affected_metric = metric;
        reg->is_confirmed = false;
        reg->likely_cause = strdup("Performance degradation detected");

        return reg;
    }

    return NULL;
}

// Store regression
void store_regression(EffectivenessAnalyzer* analyzer, EffRegression* regression) {
    if (!analyzer || !regression) return;
    if (analyzer->num_regressions >= analyzer->regression_capacity) {
        cleanup_regression_internal(regression);
        free(regression);
        return;
    }

    analyzer->regressions[analyzer->num_regressions++] = *regression;
    free(regression);  // Data copied, free original
}

// Add regression finding to report
void add_regression_finding(EffectivenessReport* report, EffRegression* regression, EffSuggestionType type) {
    if (!report || !regression) return;

    RegressionFinding* new_regs = realloc(report->regressions,
                                           (report->num_regressions + 1) * sizeof(RegressionFinding));
    if (!new_regs) return;

    report->regressions = new_regs;

    RegressionFinding* finding = &report->regressions[report->num_regressions];
    finding->metric = regression->affected_metric;
    finding->severity = regression->severity;
    finding->related_type = type;
    finding->likely_cause = regression->likely_cause ? strdup(regression->likely_cause) : NULL;

    report->num_regressions++;
}

// Generate regression mitigations
void generate_regression_mitigations(EffectivenessAnalyzer* analyzer,
                                      EffRegression* regression, EffectivenessReport* report) {
    if (!analyzer || !regression || !report) return;

    char** new_recs = realloc(report->recommendations,
                               (report->num_recommendations + 1) * sizeof(char*));
    if (!new_recs) return;

    report->recommendations = new_recs;
    report->recommendations[report->num_recommendations] = strdup("Consider reverting recent changes");
    report->num_recommendations++;
}

// Create effectiveness report
EffectivenessReport* create_effectiveness_report(void) {
    EffectivenessReport* report = calloc(1, sizeof(EffectivenessReport));
    if (!report) return NULL;

    report->patterns = NULL;
    report->num_patterns = 0;
    report->regressions = NULL;
    report->num_regressions = 0;
    report->correlations = NULL;
    report->num_correlations = 0;
    report->overall_effectiveness = 0.0;
    report->recommendations = NULL;
    report->num_recommendations = 0;

    return report;
}

// Cleanup effectiveness report
void cleanup_effectiveness_report(EffectivenessReport* report) {
    if (!report) return;

    for (size_t i = 0; i < report->num_patterns; i++) {
        free(report->patterns[i].description);
    }
    free(report->patterns);

    for (size_t i = 0; i < report->num_regressions; i++) {
        free(report->regressions[i].likely_cause);
    }
    free(report->regressions);

    free(report->correlations);

    for (size_t i = 0; i < report->num_recommendations; i++) {
        free(report->recommendations[i]);
    }
    free(report->recommendations);

    free(report);
}
