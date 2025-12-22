/**
 * @file shape_analyzer.c
 * @brief Shape analysis implementation
 *
 * Implements geometric analysis, polynomial fitting, symmetry detection,
 * and shape characterization for time series data.
 */

#include "quantum_geometric/distributed/shape_analyzer.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Shape analyzer - internal structure
struct ShapeAnalyzerImpl {
    // Geometric features
    GeometricFeatures features;

    // Shape characteristics
    ShapeCharacteristics characteristics;

    // Polynomial fitting
    PolynomialFit polynomial;

    // Symmetry analysis
    SymmetryResult symmetry;

    // Configuration
    ShapeConfig config;

    // Working buffers
    double* work_buffer;
    size_t buffer_size;
};

// Forward declarations
static void compute_derivatives(ShapeAnalyzer* analyzer, const double* data, size_t length);
static void compute_curvature_array(ShapeAnalyzer* analyzer, const double* data, size_t length);
static void find_inflection_points(ShapeAnalyzer* analyzer, const double* data, size_t length);
static void compute_characteristics(ShapeAnalyzer* analyzer, const double* data, size_t length);
static void fit_polynomial_internal(ShapeAnalyzer* analyzer, const double* data, size_t length, size_t degree);
static void check_reflection_symmetry(ShapeAnalyzer* analyzer, const double* data, size_t length);
static void check_periodicity(ShapeAnalyzer* analyzer, const double* data, size_t length);
static double compute_total_variation(const double* data, size_t length);
static double compute_smoothness(const double* data, size_t length);

// Initialize shape analyzer
ShapeAnalyzer* init_shape_analyzer(const ShapeConfig* config) {
    ShapeAnalyzer* analyzer = calloc(1, sizeof(ShapeAnalyzer));
    if (!analyzer) return NULL;

    // Store configuration
    if (config) {
        analyzer->config = *config;
    } else {
        // Default configuration
        analyzer->config.max_polynomial_degree = SHAPE_MAX_POLYNOMIAL_DEGREE;
        analyzer->config.symmetry_threshold = SHAPE_SYMMETRY_THRESHOLD;
        analyzer->config.min_points = SHAPE_MIN_POINTS;
        analyzer->config.compute_derivatives = true;
        analyzer->config.compute_symmetry = true;
        analyzer->config.compute_complexity = true;
    }

    // Initialize geometric features
    analyzer->features.derivatives = calloc(SHAPE_MAX_DERIVATIVES, sizeof(double));
    analyzer->features.curvature = calloc(1024, sizeof(double));  // Max size
    analyzer->features.inflection_points = calloc(100, sizeof(double));
    analyzer->features.num_derivatives = 0;
    analyzer->features.num_curvature = 0;
    analyzer->features.num_inflection_points = 0;

    // Initialize polynomial
    analyzer->polynomial.coefficients = calloc(SHAPE_MAX_POLYNOMIAL_DEGREE + 1, sizeof(double));
    analyzer->polynomial.degree = 0;
    analyzer->polynomial.r_squared = 0.0;
    analyzer->polynomial.mse = 0.0;

    // Initialize symmetry
    analyzer->symmetry.scores = calloc(SHAPE_MAX_SYMMETRIES, sizeof(double));
    analyzer->symmetry.num_symmetries = 0;
    analyzer->symmetry.has_reflection = false;
    analyzer->symmetry.has_rotation = false;
    analyzer->symmetry.best_period = 0.0;

    // Initialize work buffer
    analyzer->buffer_size = 4096;
    analyzer->work_buffer = calloc(analyzer->buffer_size, sizeof(double));

    // Check allocations
    if (!analyzer->features.derivatives || !analyzer->features.curvature ||
        !analyzer->features.inflection_points || !analyzer->polynomial.coefficients ||
        !analyzer->symmetry.scores || !analyzer->work_buffer) {
        cleanup_shape_analyzer(analyzer);
        return NULL;
    }

    return analyzer;
}

// Compute numerical derivatives
static void compute_derivatives(ShapeAnalyzer* analyzer, const double* data, size_t length) {
    if (!analyzer || !data || length < 3) return;

    // First derivative (central difference)
    double sum_deriv = 0.0;
    for (size_t i = 1; i < length - 1; i++) {
        double deriv = (data[i + 1] - data[i - 1]) / 2.0;
        sum_deriv += deriv;
    }
    if (analyzer->features.derivatives) {
        analyzer->features.derivatives[0] = sum_deriv / (double)(length - 2);
    }

    // Second derivative
    double sum_deriv2 = 0.0;
    for (size_t i = 1; i < length - 1; i++) {
        double deriv2 = data[i + 1] - 2.0 * data[i] + data[i - 1];
        sum_deriv2 += deriv2;
    }
    if (analyzer->features.derivatives && SHAPE_MAX_DERIVATIVES > 1) {
        analyzer->features.derivatives[1] = sum_deriv2 / (double)(length - 2);
    }

    // Third derivative
    if (length >= 5 && SHAPE_MAX_DERIVATIVES > 2) {
        double sum_deriv3 = 0.0;
        for (size_t i = 2; i < length - 2; i++) {
            double deriv3 = (data[i + 2] - 2.0 * data[i + 1] + 2.0 * data[i - 1] - data[i - 2]) / 2.0;
            sum_deriv3 += deriv3;
        }
        analyzer->features.derivatives[2] = sum_deriv3 / (double)(length - 4);
    }

    analyzer->features.num_derivatives = (length >= 5) ? 3 : 2;
}

// Compute curvature at each point
static void compute_curvature_array(ShapeAnalyzer* analyzer, const double* data, size_t length) {
    if (!analyzer || !data || length < 3) return;

    size_t n = length - 2;
    if (n > 1024) n = 1024;

    for (size_t i = 1; i < length - 1; i++) {
        if (i - 1 >= n) break;

        double dx = 1.0;  // Assuming unit spacing
        double dy = (data[i + 1] - data[i - 1]) / 2.0;
        double d2y = data[i + 1] - 2.0 * data[i] + data[i - 1];

        double denom = pow(1.0 + dy * dy, 1.5);
        if (fabs(denom) > 1e-10) {
            analyzer->features.curvature[i - 1] = fabs(d2y) / denom;
        } else {
            analyzer->features.curvature[i - 1] = 0.0;
        }
    }

    analyzer->features.num_curvature = n;
}

// Find inflection points (where second derivative changes sign)
static void find_inflection_points(ShapeAnalyzer* analyzer, const double* data, size_t length) {
    if (!analyzer || !data || length < 3) return;

    size_t count = 0;
    double prev_d2 = 0.0;

    for (size_t i = 1; i < length - 1 && count < 100; i++) {
        double d2 = data[i + 1] - 2.0 * data[i] + data[i - 1];

        if (i > 1 && prev_d2 * d2 < 0) {
            // Sign change - inflection point
            analyzer->features.inflection_points[count++] = (double)i;
        }

        prev_d2 = d2;
    }

    analyzer->features.num_inflection_points = count;
}

// Compute total variation
static double compute_total_variation(const double* data, size_t length) {
    if (!data || length < 2) return 0.0;

    double total = 0.0;
    for (size_t i = 1; i < length; i++) {
        total += fabs(data[i] - data[i - 1]);
    }
    return total;
}

// Compute smoothness (inverse of average second derivative magnitude)
static double compute_smoothness(const double* data, size_t length) {
    if (!data || length < 3) return 1.0;

    double sum = 0.0;
    for (size_t i = 1; i < length - 1; i++) {
        double d2 = fabs(data[i + 1] - 2.0 * data[i] + data[i - 1]);
        sum += d2;
    }

    double avg = sum / (double)(length - 2);
    return 1.0 / (1.0 + avg);
}

// Compute shape characteristics
static void compute_characteristics(ShapeAnalyzer* analyzer, const double* data, size_t length) {
    if (!analyzer || !data || length < 3) return;

    // Mean curvature
    double sum_curv = 0.0;
    double max_curv = 0.0;
    for (size_t i = 0; i < analyzer->features.num_curvature; i++) {
        sum_curv += analyzer->features.curvature[i];
        if (analyzer->features.curvature[i] > max_curv) {
            max_curv = analyzer->features.curvature[i];
        }
    }
    analyzer->characteristics.mean_curvature = sum_curv / (double)analyzer->features.num_curvature;
    analyzer->characteristics.max_curvature = max_curv;

    // Total variation
    analyzer->characteristics.total_variation = compute_total_variation(data, length);

    // Smoothness
    analyzer->characteristics.smoothness = compute_smoothness(data, length);

    // Self-similarity (correlation at different lags)
    if (length > 10) {
        double max_corr = 0.0;
        for (size_t lag = 1; lag <= length / 4; lag++) {
            double corr = 0.0;
            size_t n = length - lag;
            for (size_t i = 0; i < n; i++) {
                corr += data[i] * data[i + lag];
            }
            corr /= (double)n;
            if (corr > max_corr) {
                max_corr = corr;
            }
        }
        analyzer->characteristics.self_similarity = max_corr;
    }

    // Complexity (based on number of direction changes)
    if (analyzer->config.compute_complexity) {
        size_t direction_changes = 0;
        double prev_dir = 0.0;
        for (size_t i = 1; i < length; i++) {
            double dir = data[i] - data[i - 1];
            if (i > 1 && prev_dir * dir < 0) {
                direction_changes++;
            }
            prev_dir = dir;
        }
        analyzer->characteristics.complexity = (double)direction_changes / (double)(length - 2);
    }
}

// Fit polynomial using least squares
static void fit_polynomial_internal(ShapeAnalyzer* analyzer, const double* data, size_t length, size_t degree) {
    if (!analyzer || !data || length < degree + 1) return;
    if (degree > SHAPE_MAX_POLYNOMIAL_DEGREE) degree = SHAPE_MAX_POLYNOMIAL_DEGREE;

    // Simple polynomial fitting using normal equations
    // For better numerical stability, would use QR decomposition

    size_t n = degree + 1;
    double* A = calloc(n * n, sizeof(double));
    double* b = calloc(n, sizeof(double));

    if (!A || !b) {
        free(A);
        free(b);
        return;
    }

    // Build normal equations
    for (size_t i = 0; i < length; i++) {
        double x = (double)i / (double)(length - 1);  // Normalize to [0, 1]
        double y = data[i];

        for (size_t j = 0; j < n; j++) {
            double xj = pow(x, (double)j);
            b[j] += y * xj;

            for (size_t k = 0; k < n; k++) {
                A[j * n + k] += xj * pow(x, (double)k);
            }
        }
    }

    // Solve using Gaussian elimination (simple version)
    for (size_t i = 0; i < n; i++) {
        // Find pivot
        double max_val = fabs(A[i * n + i]);
        size_t max_row = i;
        for (size_t k = i + 1; k < n; k++) {
            if (fabs(A[k * n + i]) > max_val) {
                max_val = fabs(A[k * n + i]);
                max_row = k;
            }
        }

        // Swap rows
        if (max_row != i) {
            for (size_t k = 0; k < n; k++) {
                double tmp = A[i * n + k];
                A[i * n + k] = A[max_row * n + k];
                A[max_row * n + k] = tmp;
            }
            double tmp = b[i];
            b[i] = b[max_row];
            b[max_row] = tmp;
        }

        // Eliminate
        for (size_t k = i + 1; k < n; k++) {
            if (fabs(A[i * n + i]) > 1e-10) {
                double factor = A[k * n + i] / A[i * n + i];
                for (size_t j = i; j < n; j++) {
                    A[k * n + j] -= factor * A[i * n + j];
                }
                b[k] -= factor * b[i];
            }
        }
    }

    // Back substitution
    for (int i = (int)n - 1; i >= 0; i--) {
        analyzer->polynomial.coefficients[i] = b[i];
        for (size_t j = (size_t)i + 1; j < n; j++) {
            analyzer->polynomial.coefficients[i] -= A[i * n + j] * analyzer->polynomial.coefficients[j];
        }
        if (fabs(A[i * n + i]) > 1e-10) {
            analyzer->polynomial.coefficients[i] /= A[i * n + i];
        }
    }

    analyzer->polynomial.degree = degree;

    // Compute R-squared and MSE
    double ss_res = 0.0, ss_tot = 0.0;
    double mean_y = 0.0;
    for (size_t i = 0; i < length; i++) {
        mean_y += data[i];
    }
    mean_y /= (double)length;

    for (size_t i = 0; i < length; i++) {
        double x = (double)i / (double)(length - 1);
        double y_pred = 0.0;
        for (size_t j = 0; j <= degree; j++) {
            y_pred += analyzer->polynomial.coefficients[j] * pow(x, (double)j);
        }
        ss_res += (data[i] - y_pred) * (data[i] - y_pred);
        ss_tot += (data[i] - mean_y) * (data[i] - mean_y);
    }

    analyzer->polynomial.mse = ss_res / (double)length;
    analyzer->polynomial.r_squared = (ss_tot > 1e-10) ? 1.0 - (ss_res / ss_tot) : 0.0;

    free(A);
    free(b);
}

// Check for reflection symmetry
static void check_reflection_symmetry(ShapeAnalyzer* analyzer, const double* data, size_t length) {
    if (!analyzer || !data || length < 3) return;

    // Check symmetry around midpoint
    size_t half = length / 2;
    double diff_sum = 0.0;
    double max_diff = 0.0;

    for (size_t i = 0; i < half; i++) {
        double diff = fabs(data[i] - data[length - 1 - i]);
        diff_sum += diff;
        if (diff > max_diff) max_diff = diff;
    }

    double avg_diff = diff_sum / (double)half;

    // Normalize by range
    double min_val = data[0], max_val = data[0];
    for (size_t i = 1; i < length; i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }
    double range = max_val - min_val;

    if (range > 1e-10) {
        double symmetry_score = 1.0 - (avg_diff / range);
        analyzer->symmetry.has_reflection = (symmetry_score > (1.0 - analyzer->config.symmetry_threshold));
        if (analyzer->symmetry.num_symmetries < SHAPE_MAX_SYMMETRIES) {
            analyzer->symmetry.scores[analyzer->symmetry.num_symmetries++] = symmetry_score;
        }
    }
}

// Check for periodicity
static void check_periodicity(ShapeAnalyzer* analyzer, const double* data, size_t length) {
    if (!analyzer || !data || length < 10) return;

    double best_period = 0.0;
    double best_score = 0.0;

    // Try different periods
    for (size_t period = 2; period <= length / 3; period++) {
        double score = 0.0;
        size_t count = 0;

        for (size_t i = 0; i + period < length; i++) {
            double diff = fabs(data[i] - data[i + period]);
            score += 1.0 / (1.0 + diff);
            count++;
        }

        if (count > 0) {
            score /= (double)count;
            if (score > best_score) {
                best_score = score;
                best_period = (double)period;
            }
        }
    }

    analyzer->symmetry.best_period = best_period;
    analyzer->characteristics.periodicity = best_score;
    analyzer->symmetry.has_rotation = (best_score > 0.8);
}

// Main analysis function
void shape_analyze(ShapeAnalyzer* analyzer, const double* data, size_t length) {
    if (!analyzer || !data || length < analyzer->config.min_points) return;

    // Compute derivatives
    if (analyzer->config.compute_derivatives) {
        compute_derivatives(analyzer, data, length);
    }

    // Compute curvature
    compute_curvature_array(analyzer, data, length);

    // Find inflection points
    find_inflection_points(analyzer, data, length);

    // Compute characteristics
    compute_characteristics(analyzer, data, length);

    // Fit polynomial
    fit_polynomial_internal(analyzer, data, length, analyzer->config.max_polynomial_degree);

    // Check symmetry
    if (analyzer->config.compute_symmetry) {
        check_reflection_symmetry(analyzer, data, length);
        check_periodicity(analyzer, data, length);
    }
}

// Get geometric features
const GeometricFeatures* shape_get_features(const ShapeAnalyzer* analyzer) {
    return analyzer ? &analyzer->features : NULL;
}

// Get shape characteristics
const ShapeCharacteristics* shape_get_characteristics(const ShapeAnalyzer* analyzer) {
    return analyzer ? &analyzer->characteristics : NULL;
}

// Get polynomial fit
const PolynomialFit* shape_get_polynomial(const ShapeAnalyzer* analyzer) {
    return analyzer ? &analyzer->polynomial : NULL;
}

// Get symmetry results
const SymmetryResult* shape_get_symmetry(const ShapeAnalyzer* analyzer) {
    return analyzer ? &analyzer->symmetry : NULL;
}

// Compute curvature at specific point
double shape_compute_curvature(const ShapeAnalyzer* analyzer, const double* data, size_t length, size_t index) {
    if (!analyzer || !data || length < 3 || index == 0 || index >= length - 1) {
        return 0.0;
    }

    double dx = 1.0;
    double dy = (data[index + 1] - data[index - 1]) / 2.0;
    double d2y = data[index + 1] - 2.0 * data[index] + data[index - 1];

    double denom = pow(1.0 + dy * dy, 1.5);
    if (fabs(denom) > 1e-10) {
        return fabs(d2y) / denom;
    }
    return 0.0;
}

// Fit polynomial
void shape_fit_polynomial(ShapeAnalyzer* analyzer, const double* data, size_t length, size_t degree) {
    fit_polynomial_internal(analyzer, data, length, degree);
}

// Check for symmetry
void shape_check_symmetry(ShapeAnalyzer* analyzer, const double* data, size_t length) {
    if (!analyzer) return;

    analyzer->symmetry.num_symmetries = 0;
    check_reflection_symmetry(analyzer, data, length);
    check_periodicity(analyzer, data, length);
}

// Reset analyzer
void shape_reset(ShapeAnalyzer* analyzer) {
    if (!analyzer) return;

    analyzer->features.num_derivatives = 0;
    analyzer->features.num_curvature = 0;
    analyzer->features.num_inflection_points = 0;

    memset(&analyzer->characteristics, 0, sizeof(ShapeCharacteristics));

    memset(analyzer->polynomial.coefficients, 0, (SHAPE_MAX_POLYNOMIAL_DEGREE + 1) * sizeof(double));
    analyzer->polynomial.degree = 0;
    analyzer->polynomial.r_squared = 0.0;
    analyzer->polynomial.mse = 0.0;

    analyzer->symmetry.num_symmetries = 0;
    analyzer->symmetry.has_reflection = false;
    analyzer->symmetry.has_rotation = false;
    analyzer->symmetry.best_period = 0.0;
}

// Cleanup
void cleanup_shape_analyzer(ShapeAnalyzer* analyzer) {
    if (!analyzer) return;

    free(analyzer->features.derivatives);
    free(analyzer->features.curvature);
    free(analyzer->features.inflection_points);
    free(analyzer->polynomial.coefficients);
    free(analyzer->symmetry.scores);
    free(analyzer->work_buffer);

    free(analyzer);
}
