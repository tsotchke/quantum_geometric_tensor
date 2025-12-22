#ifndef FEATURE_HISTORY_H
#define FEATURE_HISTORY_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

// History parameters - these are also defined in .c but we need them for allocation
#define MAX_PATTERNS 100
#define MAX_ANOMALIES 100

// History configuration - required by FeatureHistoryTracker
typedef struct {
    size_t num_features;
    size_t max_history_length;
    double anomaly_threshold;
    bool enable_pattern_detection;
    bool enable_anomaly_detection;
    bool enable_importance_tracking;
} HistoryConfig;

// Forward declaration for opaque type - struct defined in .c file
typedef struct FeatureHistoryTrackerImpl FeatureHistoryTracker;

// Create and destroy
FeatureHistoryTracker* init_feature_history_tracker(const HistoryConfig* config);
void cleanup_feature_history_tracker(FeatureHistoryTracker* tracker);

// History operations
void update_history(FeatureHistoryTracker* tracker, const double* features, time_t timestamp);

// Pattern access - returns opaque pattern pointers
struct FeaturePattern;
const struct FeaturePattern** get_patterns(FeatureHistoryTracker* tracker,
                                           size_t feature_index, size_t* num_patterns);

#ifdef __cplusplus
}
#endif

#endif // FEATURE_HISTORY_H
