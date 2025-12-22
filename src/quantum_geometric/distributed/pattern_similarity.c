/**
 * @file pattern_similarity.c
 * @brief Pattern similarity computation implementation
 *
 * Implements Dynamic Time Warping (DTW), shape-based similarity,
 * and feature-based similarity for time series pattern comparison.
 */

#include "quantum_geometric/distributed/pattern_similarity.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

// DTW cell for path tracking
typedef struct {
    double cost;
    int prev_i;
    int prev_j;
} DTWCell;

// Feature vector for similarity computation
typedef struct {
    double values[SIMILARITY_NUM_FEATURES];
} FeatureVector;

// Similarity calculator - internal structure
struct SimilarityCalculatorImpl {
    // DTW matrix
    DTWCell* dtw_matrix;
    size_t matrix_size;

    // Configuration
    SimilarityConfig config;
};

// Forward declarations
static void init_dtw_matrix(SimilarityCalculator* calc, size_t length);
static void fill_dtw_matrix(SimilarityCalculator* calc, const double* seq1, const double* seq2, size_t length);
static double find_optimal_path(SimilarityCalculator* calc, size_t length);
static double normalize_dtw_distance(double distance, size_t length);
static ShapeFeatures extract_shape_features_internal(const double* sequence, size_t length);
static double compare_feature_values(double v1, double v2);
static void extract_feature_vector(const double* sequence, size_t length, FeatureVector* vec);
static double compute_vector_similarity(const FeatureVector* v1, const FeatureVector* v2);

// Initialize similarity calculator
SimilarityCalculator* init_similarity_calculator(const SimilarityConfig* config) {
    SimilarityCalculator* calc = calloc(1, sizeof(SimilarityCalculator));
    if (!calc) return NULL;

    // Store configuration
    if (config) {
        calc->config = *config;
    } else {
        // Default configuration
        calc->config.max_sequence_length = SIMILARITY_MAX_LENGTH;
        calc->config.dtw_window = SIMILARITY_DTW_WINDOW;
        calc->config.min_similarity_threshold = 0.5;
        calc->config.enable_dtw = true;
        calc->config.enable_shape = true;
        calc->config.enable_feature = true;
        calc->config.dtw_weight = 0.4;
        calc->config.shape_weight = 0.35;
        calc->config.feature_weight = 0.25;
    }

    // Allocate DTW matrix
    calc->matrix_size = calc->config.max_sequence_length;
    calc->dtw_matrix = calloc(calc->matrix_size * calc->matrix_size, sizeof(DTWCell));
    if (!calc->dtw_matrix) {
        free(calc);
        return NULL;
    }

    return calc;
}

// Initialize DTW matrix
static void init_dtw_matrix(SimilarityCalculator* calc, size_t length) {
    if (!calc || !calc->dtw_matrix) return;

    // Initialize with infinity
    for (size_t i = 0; i < length; i++) {
        for (size_t j = 0; j < length; j++) {
            size_t idx = i * length + j;
            calc->dtw_matrix[idx].cost = DBL_MAX;
            calc->dtw_matrix[idx].prev_i = -1;
            calc->dtw_matrix[idx].prev_j = -1;
        }
    }

    // Starting point
    calc->dtw_matrix[0].cost = 0.0;
}

// Fill DTW matrix with Sakoe-Chiba band constraint
static void fill_dtw_matrix(SimilarityCalculator* calc, const double* seq1, const double* seq2, size_t length) {
    if (!calc || !seq1 || !seq2) return;

    DTWCell* matrix = calc->dtw_matrix;
    size_t window = calc->config.dtw_window;

    // Fill first cell
    matrix[0].cost = fabs(seq1[0] - seq2[0]);

    // Fill first row (within window)
    for (size_t j = 1; j < length && j <= window; j++) {
        matrix[j].cost = matrix[j-1].cost + fabs(seq1[0] - seq2[j]);
        matrix[j].prev_i = 0;
        matrix[j].prev_j = (int)(j - 1);
    }

    // Fill first column (within window)
    for (size_t i = 1; i < length && i <= window; i++) {
        size_t idx = i * length;
        matrix[idx].cost = matrix[(i-1) * length].cost + fabs(seq1[i] - seq2[0]);
        matrix[idx].prev_i = (int)(i - 1);
        matrix[idx].prev_j = 0;
    }

    // Fill rest of matrix with Sakoe-Chiba band
    for (size_t i = 1; i < length; i++) {
        size_t j_start = (i > window) ? i - window : 1;
        size_t j_end = (i + window < length) ? i + window : length - 1;

        for (size_t j = j_start; j <= j_end; j++) {
            size_t idx = i * length + j;
            double cost = fabs(seq1[i] - seq2[j]);

            // Find minimum of three neighbors
            double min_cost = DBL_MAX;
            int best_i = -1, best_j = -1;

            // From (i-1, j)
            if (i > 0) {
                size_t prev_idx = (i-1) * length + j;
                if (matrix[prev_idx].cost < min_cost) {
                    min_cost = matrix[prev_idx].cost;
                    best_i = (int)(i - 1);
                    best_j = (int)j;
                }
            }

            // From (i, j-1)
            if (j > 0) {
                size_t prev_idx = i * length + (j-1);
                if (matrix[prev_idx].cost < min_cost) {
                    min_cost = matrix[prev_idx].cost;
                    best_i = (int)i;
                    best_j = (int)(j - 1);
                }
            }

            // From (i-1, j-1) - diagonal (preferred)
            if (i > 0 && j > 0) {
                size_t prev_idx = (i-1) * length + (j-1);
                if (matrix[prev_idx].cost <= min_cost) {
                    min_cost = matrix[prev_idx].cost;
                    best_i = (int)(i - 1);
                    best_j = (int)(j - 1);
                }
            }

            matrix[idx].cost = cost + min_cost;
            matrix[idx].prev_i = best_i;
            matrix[idx].prev_j = best_j;
        }
    }
}

// Find optimal DTW path and return distance
static double find_optimal_path(SimilarityCalculator* calc, size_t length) {
    if (!calc || length == 0) return DBL_MAX;
    return calc->dtw_matrix[(length-1) * length + (length-1)].cost;
}

// Normalize DTW distance to similarity [0, 1]
static double normalize_dtw_distance(double distance, size_t length) {
    if (length == 0) return 0.0;

    // Max possible distance (rough estimate)
    double max_distance = (double)length * 2.0;
    double normalized = 1.0 - (distance / max_distance);

    return fmax(0.0, fmin(1.0, normalized));
}

// Compute DTW similarity
double similarity_compute_dtw(SimilarityCalculator* calc, const double* seq1, const double* seq2, size_t length) {
    if (!calc || !seq1 || !seq2 || length < SIMILARITY_MIN_LENGTH) return 0.0;
    if (length > calc->matrix_size) return 0.0;

    // Initialize DTW matrix
    init_dtw_matrix(calc, length);

    // Fill DTW matrix
    fill_dtw_matrix(calc, seq1, seq2, length);

    // Find optimal path distance
    double distance = find_optimal_path(calc, length);

    // Convert to similarity
    return normalize_dtw_distance(distance, length);
}

// Extract shape features from sequence
static ShapeFeatures extract_shape_features_internal(const double* sequence, size_t length) {
    ShapeFeatures features = {0};
    if (!sequence || length < 2) return features;

    // Compute mean
    double mean = 0.0;
    for (size_t i = 0; i < length; i++) {
        mean += sequence[i];
    }
    mean /= (double)length;

    // Find min and max
    double min_val = sequence[0], max_val = sequence[0];
    for (size_t i = 1; i < length; i++) {
        if (sequence[i] < min_val) min_val = sequence[i];
        if (sequence[i] > max_val) max_val = sequence[i];
    }

    // Amplitude
    features.amplitude = max_val - min_val;

    // Average slope (linear regression)
    double sum_xy = 0.0, sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0;
    for (size_t i = 0; i < length; i++) {
        double x = (double)i;
        sum_xy += x * sequence[i];
        sum_x += x;
        sum_y += sequence[i];
        sum_xx += x * x;
    }
    double n = (double)length;
    double denominator = n * sum_xx - sum_x * sum_x;
    features.slope = (denominator != 0.0) ? (n * sum_xy - sum_x * sum_y) / denominator : 0.0;

    // Curvature (second derivative approximation)
    double curvature_sum = 0.0;
    for (size_t i = 1; i < length - 1; i++) {
        double second_deriv = sequence[i+1] - 2.0 * sequence[i] + sequence[i-1];
        curvature_sum += fabs(second_deriv);
    }
    features.curvature = curvature_sum / (double)(length - 2);

    // Frequency (zero crossing rate around mean)
    size_t zero_crossings = 0;
    for (size_t i = 1; i < length; i++) {
        if ((sequence[i] >= mean && sequence[i-1] < mean) ||
            (sequence[i] < mean && sequence[i-1] >= mean)) {
            zero_crossings++;
        }
    }
    features.frequency = (double)zero_crossings / (double)(length - 1);

    // Symmetry (compare first half to reversed second half)
    size_t half = length / 2;
    double symmetry_diff = 0.0;
    for (size_t i = 0; i < half; i++) {
        symmetry_diff += fabs(sequence[i] - sequence[length - 1 - i]);
    }
    features.symmetry = 1.0 - (symmetry_diff / ((double)half * features.amplitude + 1e-10));
    features.symmetry = fmax(0.0, fmin(1.0, features.symmetry));

    // Complexity (sample entropy approximation)
    double complexity = 0.0;
    for (size_t i = 1; i < length; i++) {
        double diff = fabs(sequence[i] - sequence[i-1]);
        complexity += diff;
    }
    features.complexity = complexity / ((double)(length - 1) * features.amplitude + 1e-10);

    return features;
}

// Compare two feature values with exponential decay
static double compare_feature_values(double v1, double v2) {
    double diff = fabs(v1 - v2);
    double scale = fmax(fabs(v1), fabs(v2)) + 1e-10;
    return exp(-diff / scale);
}

// Compare two shape feature sets
double similarity_compare_shapes(const ShapeFeatures* f1, const ShapeFeatures* f2) {
    if (!f1 || !f2) return 0.0;

    double similarity = 0.0;

    // Compare each feature
    similarity += compare_feature_values(f1->slope, f2->slope);
    similarity += compare_feature_values(f1->curvature, f2->curvature);
    similarity += compare_feature_values(f1->amplitude, f2->amplitude);
    similarity += compare_feature_values(f1->frequency, f2->frequency);
    similarity += f1->symmetry * f2->symmetry + (1.0 - f1->symmetry) * (1.0 - f2->symmetry);
    similarity += compare_feature_values(f1->complexity, f2->complexity);

    return similarity / 6.0;
}

// Compute shape-based similarity
double similarity_compute_shape(SimilarityCalculator* calc, const double* seq1, const double* seq2, size_t length) {
    if (!calc || !seq1 || !seq2 || length < SIMILARITY_MIN_LENGTH) return 0.0;
    (void)calc;  // Not used but may be extended

    ShapeFeatures f1 = extract_shape_features_internal(seq1, length);
    ShapeFeatures f2 = extract_shape_features_internal(seq2, length);

    return similarity_compare_shapes(&f1, &f2);
}

// Extract feature vector from sequence
static void extract_feature_vector(const double* sequence, size_t length, FeatureVector* vec) {
    if (!sequence || !vec || length == 0) return;

    // Feature 0: Mean
    double mean = 0.0;
    for (size_t i = 0; i < length; i++) {
        mean += sequence[i];
    }
    mean /= (double)length;
    vec->values[0] = mean;

    // Feature 1: Standard deviation
    double variance = 0.0;
    for (size_t i = 0; i < length; i++) {
        double diff = sequence[i] - mean;
        variance += diff * diff;
    }
    variance /= (double)length;
    vec->values[1] = sqrt(variance);

    // Feature 2: Skewness
    double skewness = 0.0;
    if (vec->values[1] > 1e-10) {
        for (size_t i = 0; i < length; i++) {
            double z = (sequence[i] - mean) / vec->values[1];
            skewness += z * z * z;
        }
        skewness /= (double)length;
    }
    vec->values[2] = skewness;

    // Feature 3: Kurtosis
    double kurtosis = 0.0;
    if (vec->values[1] > 1e-10) {
        for (size_t i = 0; i < length; i++) {
            double z = (sequence[i] - mean) / vec->values[1];
            kurtosis += z * z * z * z;
        }
        kurtosis = kurtosis / (double)length - 3.0;
    }
    vec->values[3] = kurtosis;

    // Feature 4: Min
    double min_val = sequence[0];
    for (size_t i = 1; i < length; i++) {
        if (sequence[i] < min_val) min_val = sequence[i];
    }
    vec->values[4] = min_val;

    // Feature 5: Max
    double max_val = sequence[0];
    for (size_t i = 1; i < length; i++) {
        if (sequence[i] > max_val) max_val = sequence[i];
    }
    vec->values[5] = max_val;

    // Feature 6: Range
    vec->values[6] = max_val - min_val;

    // Feature 7: Trend (slope)
    double sum_xy = 0.0, sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0;
    for (size_t i = 0; i < length; i++) {
        double x = (double)i;
        sum_xy += x * sequence[i];
        sum_x += x;
        sum_y += sequence[i];
        sum_xx += x * x;
    }
    double n = (double)length;
    double denom = n * sum_xx - sum_x * sum_x;
    vec->values[7] = (denom != 0.0) ? (n * sum_xy - sum_x * sum_y) / denom : 0.0;
}

// Compute cosine similarity between feature vectors
static double compute_vector_similarity(const FeatureVector* v1, const FeatureVector* v2) {
    if (!v1 || !v2) return 0.0;

    double dot = 0.0, norm1 = 0.0, norm2 = 0.0;

    for (int i = 0; i < SIMILARITY_NUM_FEATURES; i++) {
        dot += v1->values[i] * v2->values[i];
        norm1 += v1->values[i] * v1->values[i];
        norm2 += v2->values[i] * v2->values[i];
    }

    double denominator = sqrt(norm1) * sqrt(norm2);
    if (denominator < 1e-10) return 0.0;

    double cosine = dot / denominator;

    // Map from [-1, 1] to [0, 1]
    return (cosine + 1.0) / 2.0;
}

// Compute feature-based similarity
double similarity_compute_feature(SimilarityCalculator* calc, const double* seq1, const double* seq2, size_t length) {
    if (!calc || !seq1 || !seq2 || length < SIMILARITY_MIN_LENGTH) return 0.0;
    (void)calc;

    FeatureVector v1, v2;
    extract_feature_vector(seq1, length, &v1);
    extract_feature_vector(seq2, length, &v2);

    return compute_vector_similarity(&v1, &v2);
}

// Extract shape features (public API)
ShapeFeatures similarity_extract_shape(SimilarityCalculator* calc, const double* sequence, size_t length) {
    (void)calc;
    return extract_shape_features_internal(sequence, length);
}

// Compute all similarity metrics
SimilarityMetrics similarity_compute_all(SimilarityCalculator* calc, const double* seq1, const double* seq2, size_t length) {
    SimilarityMetrics metrics = {0};

    if (!calc || !seq1 || !seq2 || length < SIMILARITY_MIN_LENGTH) {
        return metrics;
    }

    // Compute individual similarities
    if (calc->config.enable_dtw) {
        metrics.dtw_similarity = similarity_compute_dtw(calc, seq1, seq2, length);
    }

    if (calc->config.enable_shape) {
        metrics.shape_similarity = similarity_compute_shape(calc, seq1, seq2, length);
    }

    if (calc->config.enable_feature) {
        metrics.feature_similarity = similarity_compute_feature(calc, seq1, seq2, length);
    }

    // Compute weighted combination
    double total_weight = 0.0;
    metrics.combined_similarity = 0.0;

    if (calc->config.enable_dtw) {
        metrics.combined_similarity += calc->config.dtw_weight * metrics.dtw_similarity;
        total_weight += calc->config.dtw_weight;
    }

    if (calc->config.enable_shape) {
        metrics.combined_similarity += calc->config.shape_weight * metrics.shape_similarity;
        total_weight += calc->config.shape_weight;
    }

    if (calc->config.enable_feature) {
        metrics.combined_similarity += calc->config.feature_weight * metrics.feature_similarity;
        total_weight += calc->config.feature_weight;
    }

    if (total_weight > 0.0) {
        metrics.combined_similarity /= total_weight;
    }

    return metrics;
}

// Clean up similarity calculator
void cleanup_similarity_calculator(SimilarityCalculator* calc) {
    if (!calc) return;

    free(calc->dtw_matrix);
    free(calc);
}
