#ifndef OPTIMIZATION_SUGGESTER_H
#define OPTIMIZATION_SUGGESTER_H

/**
 * @file optimization_suggester.h
 * @brief Intelligent optimization suggestion system for distributed training
 *
 * Analyzes performance bottlenecks and generates actionable optimization
 * suggestions based on historical effectiveness and ML-based impact prediction.
 */

#include <stddef.h>
#include <stdbool.h>
#include <time.h>
#include "quantum_geometric/distributed/bottleneck_detector.h"

#ifdef __cplusplus
extern "C" {
#endif

// Configuration constants
#define SUGGESTER_MAX_SUGGESTIONS 32
#define SUGGESTER_MIN_IMPACT 0.1
#define SUGGESTER_MAX_HISTORY 1000
#define SUGGESTER_TIMEOUT_SECS 3600

// Suggestion types
typedef enum {
    SUGGEST_COMPUTE_OPTIMIZATION,
    SUGGEST_MEMORY_OPTIMIZATION,
    SUGGEST_COMMUNICATION_OPTIMIZATION,
    SUGGEST_QUANTUM_OPTIMIZATION,
    SUGGEST_HARDWARE_OPTIMIZATION,
    SUGGEST_WORKLOAD_OPTIMIZATION
} SuggestionType;

// Single optimization suggestion
typedef struct {
    SuggestionType type;
    char* description;
    double impact_score;
    double confidence;
    bool is_implemented;
    time_t timestamp;
} Suggestion;

// Suggestion history entry
typedef struct {
    SuggestionType type;
    double impact_predicted;
    double impact_actual;
    time_t timestamp;
    bool was_effective;
} SuggestionHistoryEntry;

// Suggestion history
typedef struct SuggestionHistoryImpl SuggestionHistory;

// Bottleneck information for suggestion generation
typedef struct {
    BottleneckType type;
    double severity;
    double cpu_utilization;
    double gpu_utilization;
    double memory_utilization;
    double network_utilization;
    double quantum_utilization;
    bool has_memory_pressure;
    bool has_communication_overhead;
} BottleneckInfo;

// Suggester configuration
typedef struct {
    size_t max_suggestions;
    double min_impact_threshold;
    bool enable_ml_prediction;
    bool enable_history_tracking;
    double learning_rate;
} SuggesterConfig;

// Optimization suggester
typedef struct OptimizationSuggesterImpl OptimizationSuggester;

// Initialize optimization suggester
OptimizationSuggester* init_optimization_suggester(const SuggesterConfig* config);

// Generate suggestions based on bottleneck analysis
void suggester_generate(OptimizationSuggester* suggester,
                        const BottleneckInfo* bottleneck);

// Get active suggestions
const Suggestion* suggester_get_suggestions(const OptimizationSuggester* suggester,
                                            size_t* num_suggestions);

// Get top-priority suggestion
const Suggestion* suggester_get_top_suggestion(const OptimizationSuggester* suggester);

// Mark suggestion as implemented
void suggester_mark_implemented(OptimizationSuggester* suggester, size_t index);

// Track effectiveness of implemented suggestion
void suggester_track_effectiveness(OptimizationSuggester* suggester,
                                   const Suggestion* suggestion,
                                   double actual_improvement);

// Get historical effectiveness for suggestion type
double suggester_get_type_effectiveness(const OptimizationSuggester* suggester,
                                        SuggestionType type);

// Clear all suggestions
void suggester_clear(OptimizationSuggester* suggester);

// Clean up optimization suggester
void cleanup_optimization_suggester(OptimizationSuggester* suggester);

#ifdef __cplusplus
}
#endif

#endif // OPTIMIZATION_SUGGESTER_H
