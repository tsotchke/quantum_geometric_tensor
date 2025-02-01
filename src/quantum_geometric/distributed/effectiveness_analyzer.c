#include "quantum_geometric/distributed/effectiveness_analyzer.h"
#include "quantum_geometric/core/performance_operations.h"
#include <math.h>

// Analysis parameters
#define MIN_PATTERN_LENGTH 10
#define MAX_PATTERNS 100
#define REGRESSION_THRESHOLD -0.05
#define CORRELATION_THRESHOLD 0.7
#define MAX_INSIGHTS 50

// Impact pattern
typedef struct {
    double* values;
    size_t length;
    double mean;
    double std_dev;
    bool is_significant;
} ImpactPattern;

// Performance regression
typedef struct {
    time_t detection_time;
    double severity;
    char* likely_cause;
    bool is_confirmed;
    MetricType affected_metric;
} Regression;

// Effectiveness correlation
typedef struct {
    SuggestionType type1;
    SuggestionType type2;
    double correlation;
    bool is_positive;
    double confidence;
} EffectivenessCorrelation;

// Effectiveness analyzer
typedef struct {
    // Pattern analysis
    ImpactPattern** patterns;
    size_t num_patterns;
    
    // Regression tracking
    Regression* regressions;
    size_t num_regressions;
    
    // Correlation analysis
    EffectivenessCorrelation* correlations;
    size_t num_correlations;
    
    // ML model
    MLModel* pattern_model;
    
    // Insight generation
    InsightGenerator* insight_generator;
} EffectivenessAnalyzer;

// Initialize effectiveness analyzer
EffectivenessAnalyzer* init_effectiveness_analyzer(void) {
    EffectivenessAnalyzer* analyzer = aligned_alloc(64,
        sizeof(EffectivenessAnalyzer));
    if (!analyzer) return NULL;
    
    // Initialize pattern storage
    analyzer->patterns = aligned_alloc(64,
        MAX_PATTERNS * sizeof(ImpactPattern*));
    analyzer->num_patterns = 0;
    
    // Initialize regression tracking
    analyzer->regressions = aligned_alloc(64,
        100 * sizeof(Regression));
    analyzer->num_regressions = 0;
    
    // Initialize correlation analysis
    analyzer->correlations = aligned_alloc(64,
        100 * sizeof(EffectivenessCorrelation));
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
    
    // Extract impact values
    double* impact_values = extract_impact_values(suggestion);
    size_t num_values = suggestion->num_impacts;
    
    if (num_values < MIN_PATTERN_LENGTH) return;
    
    // Detect patterns using ML
    ImpactPattern* pattern = detect_impact_pattern(
        analyzer->pattern_model,
        impact_values,
        num_values);
    
    if (pattern && pattern->is_significant) {
        // Store pattern
        store_impact_pattern(analyzer, pattern);
        
        // Add to report
        add_pattern_finding(report, pattern,
                          suggestion->suggestion.type);
    }
    
    // Analyze trend
    analyze_impact_trend(analyzer,
                        impact_values,
                        num_values,
                        report);
    
    free(impact_values);
}

// Check for performance regressions
void check_for_regressions(
    EffectivenessAnalyzer* analyzer,
    const TrackedSuggestion* suggestion,
    EffectivenessReport* report) {
    
    // Check each metric type
    for (int i = 0; i < NUM_METRIC_TYPES; i++) {
        MetricType metric = (MetricType)i;
        
        // Get metric values
        double* values = extract_metric_values(suggestion, metric);
        size_t num_values = suggestion->num_impacts;
        
        // Detect regression
        Regression* regression = detect_regression(
            values,
            num_values,
            metric);
        
        if (regression) {
            // Store regression
            store_regression(analyzer, regression);
            
            // Add to report
            add_regression_finding(report, regression,
                                suggestion->suggestion.type);
            
            // Generate mitigation suggestions
            generate_regression_mitigations(analyzer,
                                         regression,
                                         report);
        }
        
        free(values);
    }
}

// Analyze effectiveness correlations
void analyze_correlations(
    EffectivenessAnalyzer* analyzer,
    const TrackedSuggestion** suggestions,
    size_t num_suggestions) {
    
    // Reset correlations
    analyzer->num_correlations = 0;
    
    // Analyze each pair of suggestion types
    for (size_t i = 0; i < num_suggestions; i++) {
        for (size_t j = i + 1; j < num_suggestions; j++) {
            // Compute correlation
            EffectivenessCorrelation correlation =
                compute_effectiveness_correlation(
                    suggestions[i],
                    suggestions[j]);
            
            // Store if significant
            if (fabs(correlation.correlation) >
                CORRELATION_THRESHOLD) {
                store_correlation(analyzer, &correlation);
            }
        }
    }
    
    // Update ML model
    update_correlation_model(analyzer->pattern_model,
                           analyzer->correlations,
                           analyzer->num_correlations);
}

// Generate effectiveness insights
void generate_effectiveness_insights(
    EffectivenessAnalyzer* analyzer,
    const TrackedSuggestion* suggestion,
    EffectivenessReport* report) {
    
    // Generate pattern insights
    generate_pattern_insights(analyzer->insight_generator,
                            analyzer->patterns,
                            analyzer->num_patterns,
                            report);
    
    // Generate correlation insights
    generate_correlation_insights(analyzer->insight_generator,
                                analyzer->correlations,
                                analyzer->num_correlations,
                                report);
    
    // Generate trend insights
    generate_trend_insights(analyzer->insight_generator,
                          suggestion,
                          report);
    
    // Prioritize insights
    prioritize_insights(report);
}

// Clean up
void cleanup_effectiveness_analyzer(
    EffectivenessAnalyzer* analyzer) {
    
    if (!analyzer) return;
    
    // Clean up patterns
    for (size_t i = 0; i < analyzer->num_patterns; i++) {
        cleanup_impact_pattern(analyzer->patterns[i]);
    }
    free(analyzer->patterns);
    
    // Clean up regressions
    for (size_t i = 0; i < analyzer->num_regressions; i++) {
        cleanup_regression(&analyzer->regressions[i]);
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
