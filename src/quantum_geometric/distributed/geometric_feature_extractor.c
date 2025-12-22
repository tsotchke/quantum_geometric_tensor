/**
 * @file geometric_feature_extractor.c
 * @brief Geometric feature extraction for distributed training optimization
 */

#include "quantum_geometric/distributed/geometric_feature_extractor.h"
#include <math.h>
#include <string.h>

// Feature parameters
#define MAX_DERIVATIVES 4
#define MIN_CURVATURE 1e-6
#define MAX_CRITICAL_POINTS 100
#define SMOOTHING_WINDOW 5

// Forward declarations for static functions
static void compute_derivatives(GeometricFeatureExtractor* extractor,
                               const double* points, size_t length);
static void compute_curvature(GeometricFeatureExtractor* extractor,
                             const double* points, size_t length);
static void find_critical_points(GeometricFeatureExtractor* extractor,
                                const double* points, size_t length);
static void store_derivative(GeometricFeatureExtractor* extractor,
                            DifferentialFeature* feature);
static void compute_higher_derivatives(GeometricFeatureExtractor* extractor,
                                      const double* points, size_t length);
static void normalize_derivatives(GeometricFeatureExtractor* extractor);
static void find_local_extrema(GeometricFeatureExtractor* extractor,
                              const double* points, size_t length);
static void find_inflection_points(GeometricFeatureExtractor* extractor,
                                  const double* points, size_t length);
static void find_saddle_points(GeometricFeatureExtractor* extractor,
                              const double* points, size_t length);
static void sort_critical_points(GeometricFeatureExtractor* extractor);

// Geometric feature extractor structure
struct GeometricFeatureExtractor {
    DifferentialFeature** derivatives;
    size_t num_derivatives;
    CurvatureFeature* curvature;
    double* curvature_metrics;
    CriticalPoint** critical_points;
    size_t num_critical_points;
    GeometricConfig config;
};

// Create curvature feature
CurvatureFeature* create_curvature_feature(void) {
    CurvatureFeature* feature = malloc(sizeof(CurvatureFeature));
    if (!feature) return NULL;

    feature->values = NULL;
    feature->normals = NULL;
    feature->tangents = NULL;
    feature->length = 0;

    return feature;
}

// Cleanup curvature feature
void cleanup_curvature_feature(CurvatureFeature* feature) {
    if (!feature) return;
    free(feature->values);
    free(feature->normals);
    free(feature->tangents);
    free(feature);
}

// Cleanup differential feature
void cleanup_differential_feature(DifferentialFeature* feature) {
    if (!feature) return;
    free(feature->values);
    free(feature);
}

// Compute first derivative using central differences
DifferentialFeature* compute_first_derivative(const double* points, size_t length) {
    if (length < 3) return NULL;

    DifferentialFeature* deriv = malloc(sizeof(DifferentialFeature));
    if (!deriv) return NULL;

    deriv->values = malloc((length - 2) * sizeof(double));
    if (!deriv->values) {
        free(deriv);
        return NULL;
    }

    deriv->order = 1;
    deriv->length = length - 2;
    deriv->is_normalized = false;

    // Central difference: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
    for (size_t i = 1; i < length - 1; i++) {
        deriv->values[i - 1] = (points[i + 1] - points[i - 1]) / 2.0;
    }

    return deriv;
}

// Compute second derivative using central differences
DifferentialFeature* compute_second_derivative(const double* points, size_t length) {
    if (length < 3) return NULL;

    DifferentialFeature* deriv = malloc(sizeof(DifferentialFeature));
    if (!deriv) return NULL;

    deriv->values = malloc((length - 2) * sizeof(double));
    if (!deriv->values) {
        free(deriv);
        return NULL;
    }

    deriv->order = 2;
    deriv->length = length - 2;
    deriv->is_normalized = false;

    // Second derivative: f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h²
    for (size_t i = 1; i < length - 1; i++) {
        deriv->values[i - 1] = points[i + 1] - 2.0 * points[i] + points[i - 1];
    }

    return deriv;
}

// Compute curvature values
void compute_curvature_values(CurvatureFeature* curvature,
                             const double* points, size_t length) {
    if (!curvature || length < 3) return;

    curvature->length = length - 2;
    curvature->values = malloc(curvature->length * sizeof(double));
    if (!curvature->values) return;

    // Curvature: κ = |f''| / (1 + f'²)^(3/2)
    for (size_t i = 1; i < length - 1; i++) {
        double first_deriv = (points[i + 1] - points[i - 1]) / 2.0;
        double second_deriv = points[i + 1] - 2.0 * points[i] + points[i - 1];
        double denom = pow(1.0 + first_deriv * first_deriv, 1.5);
        curvature->values[i - 1] = fabs(second_deriv) / fmax(denom, MIN_CURVATURE);
    }
}

// Compute differential geometry (normals and tangents)
void compute_differential_geometry(CurvatureFeature* curvature,
                                  const double* points, size_t length) {
    if (!curvature || length < 3) return;

    size_t n = length - 2;
    curvature->normals = malloc(n * 2 * sizeof(double));  // 2D normals
    curvature->tangents = malloc(n * 2 * sizeof(double)); // 2D tangents

    if (!curvature->normals || !curvature->tangents) {
        free(curvature->normals);
        free(curvature->tangents);
        curvature->normals = NULL;
        curvature->tangents = NULL;
        return;
    }

    for (size_t i = 1; i < length - 1; i++) {
        // Tangent direction (normalized)
        double dx = points[i + 1] - points[i - 1];
        double dy = 1.0;  // Assuming unit spacing in y
        double mag = sqrt(dx * dx + dy * dy);

        size_t idx = (i - 1) * 2;
        curvature->tangents[idx] = dx / mag;
        curvature->tangents[idx + 1] = dy / mag;

        // Normal is perpendicular to tangent
        curvature->normals[idx] = -curvature->tangents[idx + 1];
        curvature->normals[idx + 1] = curvature->tangents[idx];
    }
}

// Compute curvature metrics
void compute_curvature_metrics(double* metrics, const CurvatureFeature* curvature) {
    if (!metrics || !curvature || !curvature->values) return;

    memset(metrics, 0, NUM_CURVATURE_METRICS * sizeof(double));

    size_t n = curvature->length;
    if (n == 0) return;

    // Metric 0: Mean curvature
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += curvature->values[i];
    }
    metrics[0] = sum / n;

    // Metric 1: Max curvature
    double max_val = curvature->values[0];
    for (size_t i = 1; i < n; i++) {
        if (curvature->values[i] > max_val) max_val = curvature->values[i];
    }
    metrics[1] = max_val;

    // Metric 2: Min curvature
    double min_val = curvature->values[0];
    for (size_t i = 1; i < n; i++) {
        if (curvature->values[i] < min_val) min_val = curvature->values[i];
    }
    metrics[2] = min_val;

    // Metric 3: Curvature variance
    double variance = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = curvature->values[i] - metrics[0];
        variance += diff * diff;
    }
    metrics[3] = variance / n;

    // Metric 4: Standard deviation
    metrics[4] = sqrt(metrics[3]);

    // Metric 5: Total curvature (integral)
    metrics[5] = sum;

    // Metric 6: Curvature energy (sum of squares)
    double energy = 0.0;
    for (size_t i = 0; i < n; i++) {
        energy += curvature->values[i] * curvature->values[i];
    }
    metrics[6] = energy;

    // Metric 7: Curvature range
    metrics[7] = max_val - min_val;
}

// Compute differential invariants
void compute_differential_invariants(GeometricFeatureExtractor* extractor,
                                    GeometricInvariants* invariants) {
    if (!extractor || !invariants) return;

    invariants->num_differential = 4;
    invariants->differential = malloc(4 * sizeof(double));
    if (!invariants->differential) return;

    // Use curvature metrics as differential invariants
    if (extractor->curvature_metrics) {
        invariants->differential[0] = extractor->curvature_metrics[0]; // Mean
        invariants->differential[1] = extractor->curvature_metrics[3]; // Variance
        invariants->differential[2] = extractor->curvature_metrics[5]; // Total
        invariants->differential[3] = extractor->curvature_metrics[6]; // Energy
    }
}

// Compute integral invariants
void compute_integral_invariants(GeometricFeatureExtractor* extractor,
                                GeometricInvariants* invariants) {
    if (!extractor || !invariants) return;

    invariants->num_integral = 2;
    invariants->integral = malloc(2 * sizeof(double));
    if (!invariants->integral) return;

    // Arc length approximation
    double arc_length = 0.0;
    if (extractor->curvature && extractor->curvature->values) {
        for (size_t i = 0; i < extractor->curvature->length; i++) {
            arc_length += 1.0;  // Unit step
        }
    }
    invariants->integral[0] = arc_length;

    // Bending energy
    invariants->integral[1] = extractor->curvature_metrics ?
                              extractor->curvature_metrics[6] : 0.0;
}

// Compute topological invariants
void compute_topological_invariants(GeometricFeatureExtractor* extractor,
                                   GeometricInvariants* invariants) {
    if (!extractor || !invariants) return;

    invariants->num_topological = 3;
    invariants->topological = malloc(3 * sizeof(double));
    if (!invariants->topological) return;

    // Number of critical points
    invariants->topological[0] = (double)extractor->num_critical_points;

    // Euler characteristic approximation
    size_t minima = 0, maxima = 0, saddles = 0;
    for (size_t i = 0; i < extractor->num_critical_points; i++) {
        switch (extractor->critical_points[i]->type) {
            case CRITICAL_MINIMUM: minima++; break;
            case CRITICAL_MAXIMUM: maxima++; break;
            case CRITICAL_SADDLE: saddles++; break;
            default: break;
        }
    }
    invariants->topological[1] = (double)(minima + maxima - saddles);

    // Sign changes in curvature (inflection count)
    size_t inflections = 0;
    for (size_t i = 0; i < extractor->num_critical_points; i++) {
        if (extractor->critical_points[i]->type == CRITICAL_INFLECTION) {
            inflections++;
        }
    }
    invariants->topological[2] = (double)inflections;
}

// Initialize geometric feature extractor
GeometricFeatureExtractor* init_geometric_feature_extractor(
    const GeometricConfig* config) {

    GeometricFeatureExtractor* extractor = malloc(sizeof(GeometricFeatureExtractor));
    if (!extractor) return NULL;

    extractor->derivatives = malloc(MAX_DERIVATIVES * sizeof(DifferentialFeature*));
    extractor->num_derivatives = 0;

    extractor->curvature = create_curvature_feature();
    extractor->curvature_metrics = malloc(NUM_CURVATURE_METRICS * sizeof(double));
    memset(extractor->curvature_metrics, 0, NUM_CURVATURE_METRICS * sizeof(double));

    extractor->critical_points = malloc(MAX_CRITICAL_POINTS * sizeof(CriticalPoint*));
    extractor->num_critical_points = 0;

    if (config) {
        extractor->config = *config;
    } else {
        extractor->config.compute_higher_derivatives = false;
        extractor->config.compute_curvature = true;
        extractor->config.find_critical_points = true;
        extractor->config.smoothing_window = SMOOTHING_WINDOW;
        extractor->config.curvature_threshold = MIN_CURVATURE;
        extractor->config.max_derivatives = MAX_DERIVATIVES;
    }

    return extractor;
}

// Extract geometric features
void extract_geometric_features(
    GeometricFeatureExtractor* extractor,
    const double* points,
    size_t length) {

    if (!extractor || !points || length < 3) return;

    compute_derivatives(extractor, points, length);
    compute_curvature(extractor, points, length);
    find_critical_points(extractor, points, length);
}

// Store derivative feature
static void store_derivative(GeometricFeatureExtractor* extractor,
                            DifferentialFeature* feature) {
    if (!extractor || !feature) return;
    if (extractor->num_derivatives >= MAX_DERIVATIVES) {
        cleanup_differential_feature(feature);
        return;
    }
    extractor->derivatives[extractor->num_derivatives++] = feature;
}

// Compute derivatives
static void compute_derivatives(
    GeometricFeatureExtractor* extractor,
    const double* points,
    size_t length) {

    DifferentialFeature* first = compute_first_derivative(points, length);
    store_derivative(extractor, first);

    DifferentialFeature* second = compute_second_derivative(points, length);
    store_derivative(extractor, second);

    if (extractor->config.compute_higher_derivatives) {
        compute_higher_derivatives(extractor, points, length);
    }

    normalize_derivatives(extractor);
}

// Compute higher order derivatives
static void compute_higher_derivatives(GeometricFeatureExtractor* extractor,
                                      const double* points, size_t length) {
    // Third and fourth derivatives using finite differences
    if (length < 5) return;

    // Third derivative
    DifferentialFeature* third = malloc(sizeof(DifferentialFeature));
    if (third) {
        third->length = length - 4;
        third->order = 3;
        third->is_normalized = false;
        third->values = malloc(third->length * sizeof(double));
        if (third->values) {
            for (size_t i = 2; i < length - 2; i++) {
                third->values[i - 2] = (points[i + 2] - 2*points[i + 1] +
                                       2*points[i - 1] - points[i - 2]) / 2.0;
            }
            store_derivative(extractor, third);
        } else {
            free(third);
        }
    }
}

// Normalize derivatives
static void normalize_derivatives(GeometricFeatureExtractor* extractor) {
    for (size_t d = 0; d < extractor->num_derivatives; d++) {
        DifferentialFeature* deriv = extractor->derivatives[d];
        if (!deriv || deriv->is_normalized) continue;

        // Find max absolute value
        double max_abs = 0.0;
        for (size_t i = 0; i < deriv->length; i++) {
            double abs_val = fabs(deriv->values[i]);
            if (abs_val > max_abs) max_abs = abs_val;
        }

        // Normalize
        if (max_abs > 1e-10) {
            for (size_t i = 0; i < deriv->length; i++) {
                deriv->values[i] /= max_abs;
            }
        }
        deriv->is_normalized = true;
    }
}

// Compute curvature
static void compute_curvature(
    GeometricFeatureExtractor* extractor,
    const double* points,
    size_t length) {

    CurvatureFeature* curvature = extractor->curvature;
    compute_curvature_values(curvature, points, length);
    compute_differential_geometry(curvature, points, length);
    compute_curvature_metrics(extractor->curvature_metrics, curvature);
}

// Add critical point
static void add_critical_point(GeometricFeatureExtractor* extractor,
                              size_t index, double value,
                              CriticalPointType type, double significance) {
    if (extractor->num_critical_points >= MAX_CRITICAL_POINTS) return;

    CriticalPoint* point = malloc(sizeof(CriticalPoint));
    if (!point) return;

    point->index = index;
    point->value = value;
    point->type = type;
    point->significance = significance;

    extractor->critical_points[extractor->num_critical_points++] = point;
}

// Find local extrema
static void find_local_extrema(GeometricFeatureExtractor* extractor,
                              const double* points, size_t length) {
    if (length < 3) return;

    for (size_t i = 1; i < length - 1; i++) {
        if (points[i] > points[i - 1] && points[i] > points[i + 1]) {
            // Local maximum
            double significance = fmin(points[i] - points[i - 1],
                                      points[i] - points[i + 1]);
            add_critical_point(extractor, i, points[i], CRITICAL_MAXIMUM, significance);
        } else if (points[i] < points[i - 1] && points[i] < points[i + 1]) {
            // Local minimum
            double significance = fmin(points[i - 1] - points[i],
                                      points[i + 1] - points[i]);
            add_critical_point(extractor, i, points[i], CRITICAL_MINIMUM, significance);
        }
    }
}

// Find inflection points
static void find_inflection_points(GeometricFeatureExtractor* extractor,
                                  const double* points, size_t length) {
    if (length < 4) return;

    // Inflection points where second derivative changes sign
    for (size_t i = 2; i < length - 1; i++) {
        double d2_prev = points[i - 1] - 2*points[i - 2] + (i > 2 ? points[i - 3] : points[i - 2]);
        double d2_curr = points[i] - 2*points[i - 1] + points[i - 2];

        if ((d2_prev > 0 && d2_curr < 0) || (d2_prev < 0 && d2_curr > 0)) {
            double significance = fabs(d2_curr - d2_prev);
            add_critical_point(extractor, i - 1, points[i - 1], CRITICAL_INFLECTION, significance);
        }
    }
}

// Find saddle points (for 1D, these are inflection points)
static void find_saddle_points(GeometricFeatureExtractor* extractor,
                              const double* points, size_t length) {
    (void)extractor;
    (void)points;
    (void)length;
    // For 1D curves, saddle points don't exist in the classical sense
    // This would be relevant for 2D surface analysis
}

// Sort critical points by significance
static void sort_critical_points(GeometricFeatureExtractor* extractor) {
    // Simple bubble sort for small arrays
    for (size_t i = 0; i < extractor->num_critical_points; i++) {
        for (size_t j = i + 1; j < extractor->num_critical_points; j++) {
            if (extractor->critical_points[j]->significance >
                extractor->critical_points[i]->significance) {
                CriticalPoint* temp = extractor->critical_points[i];
                extractor->critical_points[i] = extractor->critical_points[j];
                extractor->critical_points[j] = temp;
            }
        }
    }
}

// Find critical points
static void find_critical_points(
    GeometricFeatureExtractor* extractor,
    const double* points,
    size_t length) {

    find_local_extrema(extractor, points, length);
    find_inflection_points(extractor, points, length);
    find_saddle_points(extractor, points, length);
    sort_critical_points(extractor);
}

// Compute geometric invariants
void compute_geometric_invariants(
    GeometricFeatureExtractor* extractor,
    GeometricInvariants* invariants) {

    if (!extractor || !invariants) return;

    memset(invariants, 0, sizeof(GeometricInvariants));

    compute_differential_invariants(extractor, invariants);
    compute_integral_invariants(extractor, invariants);
    compute_topological_invariants(extractor, invariants);
}

// Get critical points
const CriticalPoint** get_critical_points(
    GeometricFeatureExtractor* extractor,
    size_t* num_points) {

    if (!extractor || !num_points) return NULL;
    *num_points = extractor->num_critical_points;
    return (const CriticalPoint**)extractor->critical_points;
}

// Get curvature metrics
const double* get_curvature_metrics(GeometricFeatureExtractor* extractor) {
    if (!extractor) return NULL;
    return extractor->curvature_metrics;
}

// Clean up
void cleanup_geometric_feature_extractor(GeometricFeatureExtractor* extractor) {
    if (!extractor) return;

    for (size_t i = 0; i < extractor->num_derivatives; i++) {
        cleanup_differential_feature(extractor->derivatives[i]);
    }
    free(extractor->derivatives);

    cleanup_curvature_feature(extractor->curvature);
    free(extractor->curvature_metrics);

    for (size_t i = 0; i < extractor->num_critical_points; i++) {
        free(extractor->critical_points[i]);
    }
    free(extractor->critical_points);

    free(extractor);
}
