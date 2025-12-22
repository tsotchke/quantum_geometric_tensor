#ifndef GEOMETRIC_FEATURE_EXTRACTOR_H
#define GEOMETRIC_FEATURE_EXTRACTOR_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

// Number of curvature metrics computed
#define NUM_CURVATURE_METRICS 8

// Critical point types
typedef enum CriticalPointType {
    CRITICAL_MINIMUM,
    CRITICAL_MAXIMUM,
    CRITICAL_INFLECTION,
    CRITICAL_SADDLE
} CriticalPointType;

// Geometric configuration
typedef struct GeometricConfig {
    bool compute_higher_derivatives;
    bool compute_curvature;
    bool find_critical_points;
    size_t smoothing_window;
    double curvature_threshold;
    size_t max_derivatives;
} GeometricConfig;

// Differential features
typedef struct DifferentialFeature {
    double* values;
    size_t order;
    size_t length;
    bool is_normalized;
} DifferentialFeature;

// Curvature features
typedef struct CurvatureFeature {
    double* values;
    double* normals;
    double* tangents;
    size_t length;
} CurvatureFeature;

// Critical point
typedef struct CriticalPoint {
    size_t index;
    double value;
    CriticalPointType type;
    double significance;
} CriticalPoint;

// Geometric invariants
typedef struct GeometricInvariants {
    double* differential;
    size_t num_differential;
    double* integral;
    size_t num_integral;
    double* topological;
    size_t num_topological;
} GeometricInvariants;

// Forward declaration for opaque type
typedef struct GeometricFeatureExtractor GeometricFeatureExtractor;

// Lifecycle functions
GeometricFeatureExtractor* init_geometric_feature_extractor(const GeometricConfig* config);
void cleanup_geometric_feature_extractor(GeometricFeatureExtractor* extractor);

// Feature extraction
void extract_geometric_features(GeometricFeatureExtractor* extractor,
                               const double* points, size_t length);

// Invariants computation
void compute_geometric_invariants(GeometricFeatureExtractor* extractor,
                                 GeometricInvariants* invariants);

// Accessors
const CriticalPoint** get_critical_points(GeometricFeatureExtractor* extractor,
                                         size_t* num_points);
const double* get_curvature_metrics(GeometricFeatureExtractor* extractor);

// Helper functions for internal use
CurvatureFeature* create_curvature_feature(void);
void cleanup_curvature_feature(CurvatureFeature* feature);
void cleanup_differential_feature(DifferentialFeature* feature);

DifferentialFeature* compute_first_derivative(const double* points, size_t length);
DifferentialFeature* compute_second_derivative(const double* points, size_t length);

void compute_curvature_values(CurvatureFeature* curvature, const double* points, size_t length);
void compute_differential_geometry(CurvatureFeature* curvature, const double* points, size_t length);
void compute_curvature_metrics(double* metrics, const CurvatureFeature* curvature);

void compute_differential_invariants(GeometricFeatureExtractor* extractor, GeometricInvariants* invariants);
void compute_integral_invariants(GeometricFeatureExtractor* extractor, GeometricInvariants* invariants);
void compute_topological_invariants(GeometricFeatureExtractor* extractor, GeometricInvariants* invariants);

#ifdef __cplusplus
}
#endif

#endif // GEOMETRIC_FEATURE_EXTRACTOR_H
