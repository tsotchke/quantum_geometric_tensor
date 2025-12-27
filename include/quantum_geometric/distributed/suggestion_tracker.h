#ifndef SUGGESTION_TRACKER_H
#define SUGGESTION_TRACKER_H

/**
 * @file suggestion_tracker.h
 * @brief Suggestion tracking and effectiveness monitoring
 *
 * Tracks optimization suggestions, measures their impact over time,
 * and provides effectiveness analysis for continuous improvement.
 */

#include <stddef.h>
#include <stdbool.h>
#include <time.h>
#include "quantum_geometric/distributed/optimization_suggester.h"
#include "quantum_geometric/distributed/effectiveness_analyzer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Configuration constants
#define TRACKER_MAX_SUGGESTIONS 1000
#define TRACKER_EFFECTIVENESS_WINDOW 3600  // 1 hour
#define TRACKER_MIN_SAMPLES 10
#define TRACKER_CONFIDENCE_THRESHOLD 0.8
#define TRACKER_MAX_IMPACTS 100

// Suggestion impact measurement
typedef struct {
    double performance_improvement;
    double resource_savings;
    double stability_impact;
    time_t measurement_time;
} SuggestionImpact;

// Implementation status
typedef struct {
    bool is_implemented;
    time_t implementation_time;
    char* implementation_details;
    bool is_reverted;
    char* reversion_reason;
} SuggestionImplStatus;

// Performance metrics snapshot
typedef struct {
    double throughput;
    double latency;
    double cpu_usage;
    double memory_usage;
    double gpu_usage;
    double network_bandwidth;
    time_t timestamp;
} PerformanceMetrics;

// Tracked suggestion entry
typedef struct {
    Suggestion suggestion;
    SuggestionImplStatus status;
    SuggestionImpact* impacts;
    size_t num_impacts;
    size_t impact_capacity;
    double cumulative_effectiveness;
    double confidence_score;
} SuggestionTrackerEntry;

// Tracker configuration
typedef struct {
    size_t max_tracked;
    double effectiveness_window_sec;
    size_t min_samples_for_analysis;
    double confidence_threshold;
    bool enable_auto_reversion;
    double reversion_threshold;
} SuggestionTrackerConfig;

// Suggestion tracker (opaque)
typedef struct SuggestionTrackerImpl SuggestionTracker;

// Initialize suggestion tracker
SuggestionTracker* init_suggestion_tracker(const SuggestionTrackerConfig* config);

// Start tracking a suggestion
void tracker_track_suggestion(
    SuggestionTracker* tracker,
    const Suggestion* suggestion);

// Record suggestion implementation
void tracker_record_implementation(
    SuggestionTracker* tracker,
    const Suggestion* suggestion,
    const char* details);

// Measure current impact of a suggestion
void tracker_measure_impact(
    SuggestionTracker* tracker,
    const Suggestion* suggestion);

// Analyze effectiveness of all tracked suggestions
void tracker_analyze_effectiveness(
    SuggestionTracker* tracker,
    EffectivenessReport* report);

// Record suggestion reversion
void tracker_record_reversion(
    SuggestionTracker* tracker,
    const Suggestion* suggestion,
    const char* reason);

// Get tracked suggestion entry
const SuggestionTrackerEntry* tracker_get_entry(
    const SuggestionTracker* tracker,
    const Suggestion* suggestion);

// Get number of tracked suggestions
size_t tracker_get_count(const SuggestionTracker* tracker);

// Get suggestions by effectiveness
void tracker_get_by_effectiveness(
    const SuggestionTracker* tracker,
    SuggestionTrackerEntry** entries,
    size_t* num_entries,
    bool descending);

// Update baseline metrics
void tracker_update_baseline(SuggestionTracker* tracker);

// Check if suggestion should be reverted
bool tracker_should_revert(
    const SuggestionTracker* tracker,
    const Suggestion* suggestion);

// Clean up suggestion tracker
void cleanup_suggestion_tracker(SuggestionTracker* tracker);

#ifdef __cplusplus
}
#endif

#endif // SUGGESTION_TRACKER_H
