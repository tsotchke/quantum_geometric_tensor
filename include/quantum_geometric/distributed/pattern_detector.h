#ifndef PATTERN_DETECTOR_H
#define PATTERN_DETECTOR_H

#include <stddef.h>
#include <stdbool.h>
#include <time.h>
#include "quantum_geometric/distributed/feature_history.h"
#include "quantum_geometric/distributed/bottleneck_detector.h"

#ifdef __cplusplus
extern "C" {
#endif

// Pattern configuration
typedef struct {
    size_t max_patterns;
    double min_confidence;
    size_t min_support;
    size_t max_lag;
} PatternConfig;

// Opaque pattern detector type
typedef struct PatternDetectorImpl PatternDetector;

// Initialize pattern detector
PatternDetector* init_pattern_detector(const PatternConfig* config);

// Detect patterns in time series
void detect_patterns(PatternDetector* detector,
                    const double* values,
                    size_t length,
                    time_t* timestamps);

// Get significant patterns
const struct FeaturePattern** get_significant_patterns(PatternDetector* detector,
                                                       size_t* num_patterns);

// Clean up
void cleanup_pattern_detector(PatternDetector* detector);

#ifdef __cplusplus
}
#endif

#endif // PATTERN_DETECTOR_H
