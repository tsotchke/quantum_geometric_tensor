#include "quantum_geometric/distributed/pattern_similarity.h"
#include "quantum_geometric/core/performance_operations.h"
#include <math.h>

// Similarity parameters
#define DTW_WINDOW 10
#define MIN_SEQUENCE_LENGTH 5
#define MAX_WARPING_FACTOR 2.0
#define FEATURE_DIMENSIONS 8

// DTW cell
typedef struct {
    double cost;
    int prev_i;
    int prev_j;
} DTWCell;

// Shape features
typedef struct {
    double slope;
    double curvature;
    double amplitude;
    double frequency;
    double symmetry;
    double complexity;
} ShapeFeatures;

// Feature vector
typedef struct {
    double* values;
    size_t length;
    double* weights;
} FeatureVector;

// Similarity metrics
typedef struct {
    double dtw_similarity;
    double shape_similarity;
    double feature_similarity;
    double combined_similarity;
} SimilarityMetrics;

// Initialize similarity calculator
SimilarityCalculator* init_similarity_calculator(
    const SimilarityConfig* config) {
    
    SimilarityCalculator* calculator = aligned_alloc(64,
        sizeof(SimilarityCalculator));
    if (!calculator) return NULL;
    
    // Initialize DTW matrix
    calculator->dtw_matrix = aligned_alloc(64,
        MAX_SEQUENCE_LENGTH * MAX_SEQUENCE_LENGTH * sizeof(DTWCell));
    
    // Initialize feature extraction
    calculator->feature_extractor = init_feature_extractor();
    
    // Initialize shape analysis
    calculator->shape_analyzer = init_shape_analyzer();
    
    // Store configuration
    calculator->config = *config;
    
    return calculator;
}

// Compute DTW similarity
double compute_dtw_similarity(
    SimilarityCalculator* calculator,
    const double* sequence1,
    const double* sequence2,
    size_t length) {
    
    if (length < MIN_SEQUENCE_LENGTH) return 0.0;
    
    // Initialize DTW matrix
    init_dtw_matrix(calculator, length);
    
    // Fill DTW matrix
    fill_dtw_matrix(calculator,
                   sequence1,
                   sequence2,
                   length);
    
    // Find optimal path
    double similarity = find_optimal_path(calculator, length);
    
    // Normalize similarity
    similarity = normalize_dtw_similarity(similarity, length);
    
    return similarity;
}

// Compute shape similarity
double compute_shape_similarity(
    SimilarityCalculator* calculator,
    const double* sequence1,
    const double* sequence2,
    size_t length) {
    
    // Extract shape features
    ShapeFeatures features1 = extract_shape_features(
        calculator, sequence1, length);
    
    ShapeFeatures features2 = extract_shape_features(
        calculator, sequence2, length);
    
    // Compare features
    double similarity = compare_shape_features(
        &features1, &features2);
    
    return similarity;
}

// Compute feature similarity
double compute_feature_similarity(
    SimilarityCalculator* calculator,
    const double* sequence1,
    const double* sequence2,
    size_t length) {
    
    // Extract feature vectors
    FeatureVector* vector1 = extract_feature_vector(
        calculator, sequence1, length);
    
    FeatureVector* vector2 = extract_feature_vector(
        calculator, sequence2, length);
    
    // Compute similarity
    double similarity = compute_vector_similarity(
        vector1, vector2);
    
    cleanup_feature_vector(vector1);
    cleanup_feature_vector(vector2);
    
    return similarity;
}

// Fill DTW matrix
static void fill_dtw_matrix(
    SimilarityCalculator* calculator,
    const double* sequence1,
    const double* sequence2,
    size_t length) {
    
    DTWCell* matrix = calculator->dtw_matrix;
    
    // Fill first row and column
    for (size_t i = 0; i < length; i++) {
        matrix[i * length].cost = INFINITY;
        matrix[i].cost = INFINITY;
    }
    matrix[0].cost = 0;
    
    // Fill rest of matrix
    for (size_t i = 1; i < length; i++) {
        for (size_t j = 1; j < length; j++) {
            // Compute cost
            double cost = fabs(sequence1[i] - sequence2[j]);
            
            // Find minimum previous cost
            double min_prev = find_min_previous(
                matrix, i, j, length);
            
            // Update cell
            size_t idx = i * length + j;
            matrix[idx].cost = cost + min_prev;
            matrix[idx].prev_i = get_prev_i(matrix, i, j, length);
            matrix[idx].prev_j = get_prev_j(matrix, i, j, length);
        }
    }
}

// Extract shape features
static ShapeFeatures extract_shape_features(
    SimilarityCalculator* calculator,
    const double* sequence,
    size_t length) {
    
    ShapeFeatures features;
    
    // Compute slope
    features.slope = compute_average_slope(sequence, length);
    
    // Compute curvature
    features.curvature = compute_curvature(sequence, length);
    
    // Compute amplitude
    features.amplitude = compute_amplitude(sequence, length);
    
    // Compute frequency
    features.frequency = compute_frequency(sequence, length);
    
    // Compute symmetry
    features.symmetry = compute_symmetry(sequence, length);
    
    // Compute complexity
    features.complexity = compute_complexity(sequence, length);
    
    return features;
}

// Compare shape features
static double compare_shape_features(
    const ShapeFeatures* features1,
    const ShapeFeatures* features2) {
    
    double similarity = 0.0;
    
    // Compare individual features
    similarity += compare_slope(features1->slope,
                              features2->slope);
    
    similarity += compare_curvature(features1->curvature,
                                  features2->curvature);
    
    similarity += compare_amplitude(features1->amplitude,
                                  features2->amplitude);
    
    similarity += compare_frequency(features1->frequency,
                                  features2->frequency);
    
    similarity += compare_symmetry(features1->symmetry,
                                 features2->symmetry);
    
    similarity += compare_complexity(features1->complexity,
                                   features2->complexity);
    
    // Normalize similarity
    similarity /= 6.0;
    
    return similarity;
}

// Clean up
void cleanup_similarity_calculator(
    SimilarityCalculator* calculator) {
    
    if (!calculator) return;
    
    // Clean up DTW matrix
    free(calculator->dtw_matrix);
    
    // Clean up feature extraction
    cleanup_feature_extractor(calculator->feature_extractor);
    
    // Clean up shape analysis
    cleanup_shape_analyzer(calculator->shape_analyzer);
    
    free(calculator);
}
