/**
 * @file pattern_recognition_model.c
 * @brief Pattern recognition model implementation
 *
 * Implements template-based pattern matching with dynamic time warping,
 * shape-based similarity, feature matching, and online learning.
 */

#include "quantum_geometric/distributed/pattern_recognition_model.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

// Internal constants
#define DTW_WINDOW_SIZE 10
#define FEATURE_DIMENSIONS 8
#define WEIGHT_DECAY 0.999
#define MIN_IMPORTANCE 0.01
#define PRUNE_THRESHOLD 0.05

// Pattern template - internal structure
struct PatternTemplateImpl {
    double* values;
    size_t length;
    PatternType type;
    double* weights;
    double importance;
    double* features;           // Extracted features for fast matching
    size_t match_count;         // Number of times matched
    size_t correct_count;       // Number of correct matches
};

// Pattern recognition model - internal structure
struct PatternRecognitionModelImpl {
    // Template storage
    PatternTemplate** templates;
    size_t num_templates;
    size_t template_capacity;

    // Pattern matching results
    PatternMatch* active_matches;
    size_t num_matches;
    size_t match_capacity;

    // Learning state
    double* learning_weights;
    size_t iteration_count;
    double* batch_gradients;
    size_t batch_count;

    // Configuration
    RecognitionConfig config;
};

// Forward declarations for static functions
static PatternTemplate* create_pattern_template(const double* values, size_t length, PatternType type);
static void cleanup_pattern_template(PatternTemplate* tmpl);
static void initialize_template_weights(PatternTemplate* tmpl);
static void extract_template_features(PatternTemplate* tmpl);
static double compute_template_importance(const PatternRecognitionModel* model, const PatternTemplate* tmpl);
static void update_learning_weights(PatternRecognitionModel* model);
static void clear_pattern_matches(PatternRecognitionModel* model);
static void store_pattern_match(PatternRecognitionModel* model, const PatternMatch* match);
static void update_match_confidences(PatternRecognitionModel* model);
static double compute_pattern_similarity(const PatternTemplate* tmpl, const double* values, size_t length, const RecognitionConfig* config);
static double compute_dtw_similarity(const double* tmpl_values, const double* values, size_t length);
static double compute_shape_similarity(const double* tmpl_values, const double* values, size_t length);
static double compute_feature_similarity(const double* tmpl_values, const double* values, size_t length);
static double combine_similarities(double dtw, double shape, double feature, const double* weights);
static void update_template_weights(PatternRecognitionModel* model, const PatternMatch* match, bool is_correct);
static void update_template_importance(PatternRecognitionModel* model, PatternTemplate* tmpl);
static void prune_templates(PatternRecognitionModel* model);
static void perform_batch_update(PatternRecognitionModel* model);
static double* compute_weight_gradient(const PatternTemplate* tmpl, const PatternMatch* match, bool is_correct);
static void normalize_weights(double* weights, size_t length);
static void extract_features(const double* values, size_t length, double* features);

// Initialize pattern recognition model
PatternRecognitionModel* init_pattern_recognition_model(const RecognitionConfig* config) {
    PatternRecognitionModel* model = calloc(1, sizeof(PatternRecognitionModel));
    if (!model) return NULL;

    // Store configuration
    if (config) {
        model->config = *config;
    } else {
        // Default configuration
        model->config.max_templates = PATTERN_MAX_TEMPLATES;
        model->config.min_similarity = PATTERN_MIN_SIMILARITY;
        model->config.learning_rate = PATTERN_LEARNING_RATE;
        model->config.batch_size = PATTERN_BATCH_SIZE;
        model->config.enable_dtw = true;
        model->config.enable_shape_match = true;
        model->config.enable_online_learn = true;
    }

    // Initialize template storage
    model->template_capacity = model->config.max_templates;
    model->templates = calloc(model->template_capacity, sizeof(PatternTemplate*));
    if (!model->templates) {
        free(model);
        return NULL;
    }
    model->num_templates = 0;

    // Initialize pattern matching results
    model->match_capacity = model->template_capacity;
    model->active_matches = calloc(model->match_capacity, sizeof(PatternMatch));
    if (!model->active_matches) {
        free(model->templates);
        free(model);
        return NULL;
    }
    model->num_matches = 0;

    // Initialize learning state
    model->learning_weights = calloc(model->template_capacity, sizeof(double));
    if (!model->learning_weights) {
        free(model->active_matches);
        free(model->templates);
        free(model);
        return NULL;
    }

    // Initialize batch gradients
    model->batch_gradients = calloc(FEATURE_DIMENSIONS, sizeof(double));
    if (!model->batch_gradients) {
        free(model->learning_weights);
        free(model->active_matches);
        free(model->templates);
        free(model);
        return NULL;
    }

    model->iteration_count = 0;
    model->batch_count = 0;

    return model;
}

// Create pattern template
static PatternTemplate* create_pattern_template(const double* values, size_t length, PatternType type) {
    if (!values || length == 0) return NULL;

    PatternTemplate* tmpl = calloc(1, sizeof(PatternTemplate));
    if (!tmpl) return NULL;

    tmpl->values = calloc(length, sizeof(double));
    if (!tmpl->values) {
        free(tmpl);
        return NULL;
    }
    memcpy(tmpl->values, values, length * sizeof(double));

    tmpl->length = length;
    tmpl->type = type;

    tmpl->weights = calloc(3, sizeof(double));  // DTW, shape, feature weights
    if (!tmpl->weights) {
        free(tmpl->values);
        free(tmpl);
        return NULL;
    }

    tmpl->features = calloc(FEATURE_DIMENSIONS, sizeof(double));
    if (!tmpl->features) {
        free(tmpl->weights);
        free(tmpl->values);
        free(tmpl);
        return NULL;
    }

    tmpl->importance = 1.0;
    tmpl->match_count = 0;
    tmpl->correct_count = 0;

    return tmpl;
}

// Cleanup pattern template
static void cleanup_pattern_template(PatternTemplate* tmpl) {
    if (!tmpl) return;
    free(tmpl->values);
    free(tmpl->weights);
    free(tmpl->features);
    free(tmpl);
}

// Initialize template weights (similarity combination weights)
static void initialize_template_weights(PatternTemplate* tmpl) {
    if (!tmpl || !tmpl->weights) return;

    // Default weights for similarity combination
    tmpl->weights[0] = 0.4;  // DTW weight
    tmpl->weights[1] = 0.35; // Shape weight
    tmpl->weights[2] = 0.25; // Feature weight
}

// Extract features from template for fast matching
static void extract_template_features(PatternTemplate* tmpl) {
    if (!tmpl || !tmpl->values || !tmpl->features) return;
    extract_features(tmpl->values, tmpl->length, tmpl->features);
}

// Extract features from time series
static void extract_features(const double* values, size_t length, double* features) {
    if (!values || !features || length == 0) return;

    // Feature 0: Mean
    double mean = 0.0;
    for (size_t i = 0; i < length; i++) {
        mean += values[i];
    }
    mean /= (double)length;
    features[0] = mean;

    // Feature 1: Standard deviation
    double variance = 0.0;
    for (size_t i = 0; i < length; i++) {
        double diff = values[i] - mean;
        variance += diff * diff;
    }
    variance /= (double)length;
    features[1] = sqrt(variance);

    // Feature 2: Min
    double min_val = values[0];
    for (size_t i = 1; i < length; i++) {
        if (values[i] < min_val) min_val = values[i];
    }
    features[2] = min_val;

    // Feature 3: Max
    double max_val = values[0];
    for (size_t i = 1; i < length; i++) {
        if (values[i] > max_val) max_val = values[i];
    }
    features[3] = max_val;

    // Feature 4: Trend (linear regression slope)
    double sum_xy = 0.0, sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0;
    for (size_t i = 0; i < length; i++) {
        double x = (double)i;
        sum_xy += x * values[i];
        sum_x += x;
        sum_y += values[i];
        sum_xx += x * x;
    }
    double n = (double)length;
    double denominator = n * sum_xx - sum_x * sum_x;
    features[4] = (denominator != 0.0) ? (n * sum_xy - sum_x * sum_y) / denominator : 0.0;

    // Feature 5: Skewness
    double skewness = 0.0;
    if (features[1] > 1e-10) {
        for (size_t i = 0; i < length; i++) {
            double diff = (values[i] - mean) / features[1];
            skewness += diff * diff * diff;
        }
        skewness /= (double)length;
    }
    features[5] = skewness;

    // Feature 6: Kurtosis
    double kurtosis = 0.0;
    if (features[1] > 1e-10) {
        for (size_t i = 0; i < length; i++) {
            double diff = (values[i] - mean) / features[1];
            kurtosis += diff * diff * diff * diff;
        }
        kurtosis = kurtosis / (double)length - 3.0;  // Excess kurtosis
    }
    features[6] = kurtosis;

    // Feature 7: Zero crossing rate
    size_t zero_crossings = 0;
    for (size_t i = 1; i < length; i++) {
        if ((values[i] >= mean && values[i-1] < mean) ||
            (values[i] < mean && values[i-1] >= mean)) {
            zero_crossings++;
        }
    }
    features[7] = (double)zero_crossings / (double)(length - 1);
}

// Compute template importance based on historical performance
static double compute_template_importance(const PatternRecognitionModel* model, const PatternTemplate* tmpl) {
    if (!tmpl) return 0.0;
    (void)model;  // May use for global statistics later

    // Base importance
    double importance = 1.0;

    // Adjust based on match history
    if (tmpl->match_count > 0) {
        double accuracy = (double)tmpl->correct_count / (double)tmpl->match_count;
        importance *= (0.5 + 0.5 * accuracy);  // Scale between 0.5 and 1.0
    }

    return importance;
}

// Update learning weights based on template performance
static void update_learning_weights(PatternRecognitionModel* model) {
    if (!model || !model->learning_weights) return;

    // Compute total importance
    double total_importance = 0.0;
    for (size_t i = 0; i < model->num_templates; i++) {
        if (model->templates[i]) {
            total_importance += model->templates[i]->importance;
        }
    }

    // Normalize learning weights
    if (total_importance > 0.0) {
        for (size_t i = 0; i < model->num_templates; i++) {
            if (model->templates[i]) {
                model->learning_weights[i] = model->templates[i]->importance / total_importance;
            }
        }
    }
}

// Add pattern template
void add_pattern_template(PatternRecognitionModel* model, const double* values, size_t length, PatternType type) {
    if (!model || !values || length == 0) return;
    if (model->num_templates >= model->template_capacity) return;

    // Create new template
    PatternTemplate* tmpl = create_pattern_template(values, length, type);
    if (!tmpl) return;

    // Initialize weights
    initialize_template_weights(tmpl);

    // Extract features
    extract_template_features(tmpl);

    // Compute importance
    tmpl->importance = compute_template_importance(model, tmpl);

    // Store template
    model->templates[model->num_templates++] = tmpl;

    // Update learning weights
    update_learning_weights(model);
}

// Clear pattern matches
static void clear_pattern_matches(PatternRecognitionModel* model) {
    if (!model) return;
    model->num_matches = 0;
}

// Store pattern match
static void store_pattern_match(PatternRecognitionModel* model, const PatternMatch* match) {
    if (!model || !match) return;
    if (model->num_matches >= model->match_capacity) return;

    model->active_matches[model->num_matches++] = *match;
}

// Update match confidences based on template importance
static void update_match_confidences(PatternRecognitionModel* model) {
    if (!model) return;

    for (size_t i = 0; i < model->num_matches; i++) {
        PatternMatch* match = &model->active_matches[i];
        if (match->pattern_template) {
            match->confidence = match->similarity * match->pattern_template->importance;
        }
    }

    // Sort matches by confidence (descending) using simple insertion sort
    for (size_t i = 1; i < model->num_matches; i++) {
        PatternMatch temp = model->active_matches[i];
        size_t j = i;
        while (j > 0 && model->active_matches[j-1].confidence < temp.confidence) {
            model->active_matches[j] = model->active_matches[j-1];
            j--;
        }
        model->active_matches[j] = temp;
    }
}

// Compute Dynamic Time Warping similarity
static double compute_dtw_similarity(const double* tmpl_values, const double* values, size_t length) {
    if (!tmpl_values || !values || length == 0) return 0.0;

    // Allocate DTW matrix
    double* dtw = calloc(length * length, sizeof(double));
    if (!dtw) return 0.0;

    // Initialize
    for (size_t i = 0; i < length * length; i++) {
        dtw[i] = DBL_MAX;
    }
    dtw[0] = fabs(tmpl_values[0] - values[0]);

    // Fill first row
    for (size_t j = 1; j < length; j++) {
        dtw[j] = dtw[j-1] + fabs(tmpl_values[0] - values[j]);
    }

    // Fill first column
    for (size_t i = 1; i < length; i++) {
        dtw[i * length] = dtw[(i-1) * length] + fabs(tmpl_values[i] - values[0]);
    }

    // Fill rest with Sakoe-Chiba band constraint
    for (size_t i = 1; i < length; i++) {
        size_t j_start = (i > DTW_WINDOW_SIZE) ? i - DTW_WINDOW_SIZE : 0;
        size_t j_end = (i + DTW_WINDOW_SIZE < length) ? i + DTW_WINDOW_SIZE : length - 1;

        for (size_t j = j_start; j <= j_end; j++) {
            if (j == 0) continue;  // Already filled

            double cost = fabs(tmpl_values[i] - values[j]);
            double min_prev = dtw[(i-1) * length + j];  // Top

            if (dtw[i * length + j - 1] < min_prev) {
                min_prev = dtw[i * length + j - 1];  // Left
            }
            if (dtw[(i-1) * length + j - 1] < min_prev) {
                min_prev = dtw[(i-1) * length + j - 1];  // Diagonal
            }

            dtw[i * length + j] = cost + min_prev;
        }
    }

    double dtw_distance = dtw[(length-1) * length + (length-1)];
    free(dtw);

    // Convert distance to similarity (0 to 1)
    double max_possible = (double)length * 2.0;  // Rough estimate
    double similarity = 1.0 - (dtw_distance / max_possible);
    return fmax(0.0, fmin(1.0, similarity));
}

// Compute shape-based similarity using normalized cross-correlation
static double compute_shape_similarity(const double* tmpl_values, const double* values, size_t length) {
    if (!tmpl_values || !values || length == 0) return 0.0;

    // Compute means
    double mean_t = 0.0, mean_v = 0.0;
    for (size_t i = 0; i < length; i++) {
        mean_t += tmpl_values[i];
        mean_v += values[i];
    }
    mean_t /= (double)length;
    mean_v /= (double)length;

    // Compute normalized cross-correlation
    double numerator = 0.0;
    double denom_t = 0.0, denom_v = 0.0;

    for (size_t i = 0; i < length; i++) {
        double dt = tmpl_values[i] - mean_t;
        double dv = values[i] - mean_v;
        numerator += dt * dv;
        denom_t += dt * dt;
        denom_v += dv * dv;
    }

    double denominator = sqrt(denom_t * denom_v);
    if (denominator < 1e-10) return 0.0;

    double correlation = numerator / denominator;

    // Map from [-1, 1] to [0, 1]
    return (correlation + 1.0) / 2.0;
}

// Compute feature-based similarity
static double compute_feature_similarity(const double* tmpl_values, const double* values, size_t length) {
    if (!tmpl_values || !values || length == 0) return 0.0;

    // Extract features from both
    double features_t[FEATURE_DIMENSIONS];
    double features_v[FEATURE_DIMENSIONS];

    extract_features(tmpl_values, length, features_t);
    extract_features(values, length, features_v);

    // Compute cosine similarity
    double dot = 0.0, norm_t = 0.0, norm_v = 0.0;
    for (int i = 0; i < FEATURE_DIMENSIONS; i++) {
        dot += features_t[i] * features_v[i];
        norm_t += features_t[i] * features_t[i];
        norm_v += features_v[i] * features_v[i];
    }

    double denominator = sqrt(norm_t) * sqrt(norm_v);
    if (denominator < 1e-10) return 0.0;

    double similarity = dot / denominator;

    // Map from [-1, 1] to [0, 1]
    return (similarity + 1.0) / 2.0;
}

// Combine different similarity measures
static double combine_similarities(double dtw, double shape, double feature, const double* weights) {
    if (!weights) {
        // Default equal weights
        return (dtw + shape + feature) / 3.0;
    }

    double total_weight = weights[0] + weights[1] + weights[2];
    if (total_weight < 1e-10) return 0.0;

    return (weights[0] * dtw + weights[1] * shape + weights[2] * feature) / total_weight;
}

// Compute pattern similarity
static double compute_pattern_similarity(const PatternTemplate* tmpl, const double* values, size_t length, const RecognitionConfig* config) {
    if (!tmpl || !tmpl->values || !values || length < tmpl->length) return 0.0;

    double max_similarity = 0.0;

    // Sliding window comparison
    for (size_t i = 0; i <= length - tmpl->length; i++) {
        const double* window = values + i;

        double dtw_sim = 0.0;
        double shape_sim = 0.0;
        double feature_sim = 0.0;

        if (!config || config->enable_dtw) {
            dtw_sim = compute_dtw_similarity(tmpl->values, window, tmpl->length);
        }

        if (!config || config->enable_shape_match) {
            shape_sim = compute_shape_similarity(tmpl->values, window, tmpl->length);
        }

        feature_sim = compute_feature_similarity(tmpl->values, window, tmpl->length);

        double similarity = combine_similarities(dtw_sim, shape_sim, feature_sim, tmpl->weights);

        if (similarity > max_similarity) {
            max_similarity = similarity;
        }
    }

    return max_similarity;
}

// Match patterns
void match_patterns(PatternRecognitionModel* model, const double* values, size_t length) {
    if (!model || !values || length == 0) return;

    // Clear previous matches
    clear_pattern_matches(model);

    // Match each template
    for (size_t i = 0; i < model->num_templates; i++) {
        PatternTemplate* tmpl = model->templates[i];
        if (!tmpl) continue;

        // Compute similarity
        double similarity = compute_pattern_similarity(tmpl, values, length, &model->config);

        if (similarity >= model->config.min_similarity) {
            // Create match
            PatternMatch match;
            match.pattern_template = tmpl;
            match.similarity = similarity;
            match.offset = 0;  // Could track actual offset
            match.confidence = similarity * tmpl->importance;
            match.type = tmpl->type;

            // Store match
            store_pattern_match(model, &match);
        }
    }

    // Update match confidences
    update_match_confidences(model);
}

// Get matched patterns
const PatternMatch* get_pattern_matches(const PatternRecognitionModel* model, size_t* num_matches) {
    if (!model) {
        if (num_matches) *num_matches = 0;
        return NULL;
    }

    if (num_matches) *num_matches = model->num_matches;
    return model->active_matches;
}

// Get best match
const PatternMatch* get_best_match(const PatternRecognitionModel* model) {
    if (!model || model->num_matches == 0) return NULL;
    return &model->active_matches[0];  // Already sorted by confidence
}

// Compute weight gradient for learning
static double* compute_weight_gradient(const PatternTemplate* tmpl, const PatternMatch* match, bool is_correct) {
    if (!tmpl) return NULL;

    double* gradient = calloc(3, sizeof(double));
    if (!gradient) return NULL;

    // Simple gradient based on match quality and correctness
    double error = is_correct ? (1.0 - match->similarity) : (-match->similarity);

    // Gradient for each weight
    gradient[0] = error * 0.4;  // DTW gradient
    gradient[1] = error * 0.35; // Shape gradient
    gradient[2] = error * 0.25; // Feature gradient

    return gradient;
}

// Normalize weights to sum to 1
static void normalize_weights(double* weights, size_t length) {
    if (!weights || length == 0) return;

    double sum = 0.0;
    for (size_t i = 0; i < length; i++) {
        sum += fabs(weights[i]);
    }

    if (sum > 1e-10) {
        for (size_t i = 0; i < length; i++) {
            weights[i] /= sum;
        }
    }
}

// Update template weights based on match result
static void update_template_weights(PatternRecognitionModel* model, const PatternMatch* match, bool is_correct) {
    if (!model || !match || !match->pattern_template) return;

    PatternTemplate* tmpl = match->pattern_template;

    // Compute gradient
    double* gradient = compute_weight_gradient(tmpl, match, is_correct);
    if (!gradient) return;

    // Apply gradient update with learning rate
    double lr = model->config.learning_rate;
    for (size_t i = 0; i < 3; i++) {
        tmpl->weights[i] += lr * gradient[i];
        // Clamp to [0, 1]
        if (tmpl->weights[i] < 0.0) tmpl->weights[i] = 0.0;
        if (tmpl->weights[i] > 1.0) tmpl->weights[i] = 1.0;
    }

    // Normalize weights
    normalize_weights(tmpl->weights, 3);

    free(gradient);
}

// Update template importance based on performance
static void update_template_importance(PatternRecognitionModel* model, PatternTemplate* tmpl) {
    if (!tmpl) return;
    (void)model;

    // Apply weight decay
    tmpl->importance *= WEIGHT_DECAY;

    // Update based on match history
    if (tmpl->match_count > 0) {
        double accuracy = (double)tmpl->correct_count / (double)tmpl->match_count;
        tmpl->importance = fmax(MIN_IMPORTANCE, tmpl->importance * (0.9 + 0.2 * accuracy));
    }
}

// Prune low-importance templates
static void prune_templates(PatternRecognitionModel* model) {
    if (!model) return;

    size_t write_idx = 0;
    for (size_t i = 0; i < model->num_templates; i++) {
        if (model->templates[i] && model->templates[i]->importance >= PRUNE_THRESHOLD) {
            if (write_idx != i) {
                model->templates[write_idx] = model->templates[i];
            }
            write_idx++;
        } else if (model->templates[i]) {
            cleanup_pattern_template(model->templates[i]);
        }
    }

    // Clear remaining slots
    for (size_t i = write_idx; i < model->num_templates; i++) {
        model->templates[i] = NULL;
    }

    model->num_templates = write_idx;
}

// Perform batch update
static void perform_batch_update(PatternRecognitionModel* model) {
    if (!model || model->batch_count == 0) return;

    // Average gradients and apply
    double scale = 1.0 / (double)model->batch_count;
    for (int i = 0; i < FEATURE_DIMENSIONS; i++) {
        model->batch_gradients[i] *= scale;
    }

    // Apply to all templates (simple averaging for now)
    for (size_t t = 0; t < model->num_templates; t++) {
        if (model->templates[t]) {
            // Update importance based on batch performance
            model->templates[t]->importance *= (1.0 + model->batch_gradients[0] * 0.1);
            model->templates[t]->importance = fmax(MIN_IMPORTANCE,
                                                    fmin(1.0, model->templates[t]->importance));
        }
    }

    // Reset batch state
    memset(model->batch_gradients, 0, FEATURE_DIMENSIONS * sizeof(double));
    model->batch_count = 0;
}

// Update model with feedback
void update_model(PatternRecognitionModel* model, const PatternMatch* match, bool is_correct) {
    if (!model || !match || !match->pattern_template) return;

    PatternTemplate* tmpl = match->pattern_template;

    // Update match statistics
    tmpl->match_count++;
    if (is_correct) {
        tmpl->correct_count++;
    }

    // Update template weights
    update_template_weights(model, match, is_correct);

    // Update template importance
    update_template_importance(model, tmpl);

    // Prune low importance templates occasionally
    if (model->iteration_count % 100 == 0) {
        prune_templates(model);
    }

    // Update learning state
    model->iteration_count++;
    model->batch_count++;

    // Accumulate batch gradient
    double grad_contribution = is_correct ? 0.1 : -0.1;
    model->batch_gradients[0] += grad_contribution;

    // Perform batch update if batch is full
    if (model->batch_count >= model->config.batch_size) {
        perform_batch_update(model);
    }

    // Update global learning weights
    update_learning_weights(model);
}

// Get number of templates
size_t get_num_templates(const PatternRecognitionModel* model) {
    return model ? model->num_templates : 0;
}

// Clear all matches
void clear_matches(PatternRecognitionModel* model) {
    if (model) {
        model->num_matches = 0;
    }
}

// Clean up pattern recognition model
void cleanup_pattern_recognition_model(PatternRecognitionModel* model) {
    if (!model) return;

    // Clean up templates
    for (size_t i = 0; i < model->num_templates; i++) {
        cleanup_pattern_template(model->templates[i]);
    }
    free(model->templates);

    // Clean up matches (no dynamic allocation per match)
    free(model->active_matches);

    // Clean up learning state
    free(model->learning_weights);
    free(model->batch_gradients);

    free(model);
}
