#ifndef SHAPE_ANALYZER_H
#define SHAPE_ANALYZER_H

/**
 * @file shape_analyzer.h
 * @brief Shape analysis for time series data
 *
 * Provides geometric analysis, polynomial fitting, symmetry detection,
 * and shape characterization for distributed training patterns.
 */

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Configuration constants
#define SHAPE_MIN_POINTS 5
#define SHAPE_MAX_POLYNOMIAL_DEGREE 5
#define SHAPE_MAX_DERIVATIVES 3
#define SHAPE_MAX_SYMMETRIES 10
#define SHAPE_SYMMETRY_THRESHOLD 0.1

// Geometric features
typedef struct {
    double* derivatives;
    size_t num_derivatives;
    double* curvature;
    size_t num_curvature;
    double* inflection_points;
    size_t num_inflection_points;
} GeometricFeatures;

// Shape characteristics
typedef struct {
    double mean_curvature;
    double max_curvature;
    double total_variation;
    double smoothness;
    double periodicity;
    double self_similarity;
    double complexity;
} ShapeCharacteristics;

// Polynomial fit result
typedef struct {
    double* coefficients;
    size_t degree;
    double r_squared;
    double mse;
} PolynomialFit;

// Symmetry result
typedef struct {
    double* scores;
    size_t num_symmetries;
    bool has_reflection;
    bool has_rotation;
    double best_period;
} SymmetryResult;

// Shape configuration
typedef struct {
    size_t max_polynomial_degree;
    double symmetry_threshold;
    size_t min_points;
    bool compute_derivatives;
    bool compute_symmetry;
    bool compute_complexity;
} ShapeConfig;

// Shape analyzer (opaque)
typedef struct ShapeAnalyzerImpl ShapeAnalyzer;

// Initialize shape analyzer
ShapeAnalyzer* init_shape_analyzer(const ShapeConfig* config);

// Analyze shape of data
void shape_analyze(
    ShapeAnalyzer* analyzer,
    const double* data,
    size_t length);

// Get geometric features
const GeometricFeatures* shape_get_features(const ShapeAnalyzer* analyzer);

// Get shape characteristics
const ShapeCharacteristics* shape_get_characteristics(const ShapeAnalyzer* analyzer);

// Get polynomial fit
const PolynomialFit* shape_get_polynomial(const ShapeAnalyzer* analyzer);

// Get symmetry results
const SymmetryResult* shape_get_symmetry(const ShapeAnalyzer* analyzer);

// Compute curvature at specific point
double shape_compute_curvature(
    const ShapeAnalyzer* analyzer,
    const double* data,
    size_t length,
    size_t index);

// Fit polynomial to data
void shape_fit_polynomial(
    ShapeAnalyzer* analyzer,
    const double* data,
    size_t length,
    size_t degree);

// Check for symmetry
void shape_check_symmetry(
    ShapeAnalyzer* analyzer,
    const double* data,
    size_t length);

// Reset analyzer
void shape_reset(ShapeAnalyzer* analyzer);

// Clean up shape analyzer
void cleanup_shape_analyzer(ShapeAnalyzer* analyzer);

#ifdef __cplusplus
}
#endif

#endif // SHAPE_ANALYZER_H
