#ifndef PATTERN_SIMILARITY_H
#define PATTERN_SIMILARITY_H

/**
 * @file pattern_similarity.h
 * @brief Pattern similarity computation using DTW and feature-based methods
 *
 * Provides multiple similarity measures for time series patterns including
 * Dynamic Time Warping (DTW), shape-based similarity, and feature-based similarity.
 */

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Configuration constants
#define SIMILARITY_DTW_WINDOW 10
#define SIMILARITY_MIN_LENGTH 5
#define SIMILARITY_MAX_LENGTH 1024
#define SIMILARITY_NUM_FEATURES 8

// Similarity configuration
typedef struct {
    size_t max_sequence_length;
    size_t dtw_window;
    double min_similarity_threshold;
    bool enable_dtw;
    bool enable_shape;
    bool enable_feature;
    double dtw_weight;
    double shape_weight;
    double feature_weight;
} SimilarityConfig;

// Shape features
typedef struct {
    double slope;
    double curvature;
    double amplitude;
    double frequency;
    double symmetry;
    double complexity;
} ShapeFeatures;

// Similarity metrics result
typedef struct {
    double dtw_similarity;
    double shape_similarity;
    double feature_similarity;
    double combined_similarity;
} SimilarityMetrics;

// Similarity calculator (opaque)
typedef struct SimilarityCalculatorImpl SimilarityCalculator;

// Initialize similarity calculator
SimilarityCalculator* init_similarity_calculator(const SimilarityConfig* config);

// Compute DTW similarity between two sequences
double similarity_compute_dtw(
    SimilarityCalculator* calculator,
    const double* sequence1,
    const double* sequence2,
    size_t length);

// Compute shape-based similarity
double similarity_compute_shape(
    SimilarityCalculator* calculator,
    const double* sequence1,
    const double* sequence2,
    size_t length);

// Compute feature-based similarity
double similarity_compute_feature(
    SimilarityCalculator* calculator,
    const double* sequence1,
    const double* sequence2,
    size_t length);

// Compute combined similarity metrics
SimilarityMetrics similarity_compute_all(
    SimilarityCalculator* calculator,
    const double* sequence1,
    const double* sequence2,
    size_t length);

// Extract shape features from sequence
ShapeFeatures similarity_extract_shape(
    SimilarityCalculator* calculator,
    const double* sequence,
    size_t length);

// Compare two shape feature sets
double similarity_compare_shapes(
    const ShapeFeatures* features1,
    const ShapeFeatures* features2);

// Clean up similarity calculator
void cleanup_similarity_calculator(SimilarityCalculator* calculator);

#ifdef __cplusplus
}
#endif

#endif // PATTERN_SIMILARITY_H
