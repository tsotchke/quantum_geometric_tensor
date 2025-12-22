#ifndef PATTERN_RECOGNITION_MODEL_H
#define PATTERN_RECOGNITION_MODEL_H

/**
 * @file pattern_recognition_model.h
 * @brief Pattern recognition model for time series and performance data
 *
 * Provides template-based pattern matching with dynamic time warping,
 * shape-based similarity, and online learning capabilities.
 */

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Configuration constants
#define PATTERN_MAX_TEMPLATES 1000
#define PATTERN_MIN_SIMILARITY 0.8
#define PATTERN_LEARNING_RATE 0.001
#define PATTERN_BATCH_SIZE 32

// Pattern types
typedef enum {
    PATTERN_TYPE_UNKNOWN = 0,
    PATTERN_TYPE_LINEAR,
    PATTERN_TYPE_EXPONENTIAL,
    PATTERN_TYPE_PERIODIC,
    PATTERN_TYPE_SPIKE,
    PATTERN_TYPE_TREND,
    PATTERN_TYPE_ANOMALY,
    PATTERN_TYPE_CORRELATION,
    PATTERN_TYPE_BOTTLENECK,
    PATTERN_TYPE_CUSTOM
} PatternType;

// Recognition configuration
typedef struct {
    size_t max_templates;
    double min_similarity;
    double learning_rate;
    size_t batch_size;
    bool enable_dtw;           // Dynamic time warping
    bool enable_shape_match;   // Shape-based matching
    bool enable_online_learn;  // Online learning
} RecognitionConfig;

// Pattern template (opaque)
typedef struct PatternTemplateImpl PatternTemplate;

// Pattern match result
typedef struct {
    PatternTemplate* pattern_template;
    double similarity;
    size_t offset;
    double confidence;
    PatternType type;
} PatternMatch;

// Pattern recognition model (opaque)
typedef struct PatternRecognitionModelImpl PatternRecognitionModel;

// Initialize pattern recognition model
PatternRecognitionModel* init_pattern_recognition_model(
    const RecognitionConfig* config);

// Add pattern template
void add_pattern_template(
    PatternRecognitionModel* model,
    const double* values,
    size_t length,
    PatternType type);

// Match patterns in input data
void match_patterns(
    PatternRecognitionModel* model,
    const double* values,
    size_t length);

// Get matched patterns
const PatternMatch* get_pattern_matches(
    const PatternRecognitionModel* model,
    size_t* num_matches);

// Get best match
const PatternMatch* get_best_match(
    const PatternRecognitionModel* model);

// Update model with feedback
void update_model(
    PatternRecognitionModel* model,
    const PatternMatch* match,
    bool is_correct);

// Get number of templates
size_t get_num_templates(const PatternRecognitionModel* model);

// Clear all matches
void clear_matches(PatternRecognitionModel* model);

// Clean up pattern recognition model
void cleanup_pattern_recognition_model(
    PatternRecognitionModel* model);

#ifdef __cplusplus
}
#endif

#endif // PATTERN_RECOGNITION_MODEL_H
