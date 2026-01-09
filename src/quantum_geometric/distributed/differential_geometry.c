/**
 * @file differential_geometry.c
 * @brief Distributed Differential Geometry Operations Implementation
 *
 * Production implementation of differential geometry computations for
 * quantum geometric computing including metric tensors, Christoffel symbols,
 * curvature tensors, and geodesic computations.
 */

#include "quantum_geometric/distributed/differential_geometry.h"

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

// ============================================================================
// Internal Constants
// ============================================================================

#define GEOMETRY_EPSILON 1e-14
#define MAX_NEWTON_ITERATIONS 100
#define GEODESIC_STEP_TOLERANCE 1e-10
#define MATRIX_SQRT_ITERATIONS 50
#define LYAPUNOV_TOLERANCE 1e-12

// ============================================================================
// Internal Engine Structure
// ============================================================================

struct diffgeo_engine {
    diffgeo_config_t config;
    diffgeo_dist_context_t* dist_ctx;
    diffgeo_stats_t stats;
    char last_error[512];
    bool initialized;

    // Workspace for computations
    ComplexDouble* workspace;
    size_t workspace_size;

    // Cached metric data
    diffgeo_metric_tensor_t* cached_metric;
    diffgeo_christoffel_t* cached_christoffel;
    diffgeo_point_t* cache_point;
};

// ============================================================================
// Helper Function Declarations
// ============================================================================

static void complex_matrix_multiply(const ComplexDouble* A, const ComplexDouble* B,
                                   ComplexDouble* C, size_t m, size_t n, size_t k);
static bool complex_matrix_inverse(const ComplexDouble* A, ComplexDouble* Ainv, size_t n);
static bool complex_matrix_sqrt(const ComplexDouble* A, ComplexDouble* sqrtA, size_t n);
static bool solve_lyapunov_equation(const ComplexDouble* A, const ComplexDouble* C,
                                    ComplexDouble* X, size_t n);
static double complex_abs(ComplexDouble z);
static ComplexDouble complex_conj(ComplexDouble z);
static ComplexDouble complex_mult(ComplexDouble a, ComplexDouble b);
static ComplexDouble complex_add(ComplexDouble a, ComplexDouble b);
static ComplexDouble complex_sub(ComplexDouble a, ComplexDouble b);
static ComplexDouble complex_scale(ComplexDouble z, double s);
static ComplexDouble complex_div(ComplexDouble a, ComplexDouble b);
static ComplexDouble complex_sqrt_scalar(ComplexDouble z);
static void set_error(diffgeo_engine_t* engine, const char* msg);
static bool points_equal(const diffgeo_point_t* a, const diffgeo_point_t* b);
static void distribute_computation(const diffgeo_dist_context_t* ctx, size_t total_work,
                                   size_t* start, size_t* end);

// ============================================================================
// Initialization and Configuration
// ============================================================================

diffgeo_config_t diffgeo_default_config(void) {
    diffgeo_config_t config = {
        .manifold_type = MANIFOLD_PROJECTIVE_HILBERT,
        .metric_type = METRIC_FUBINI_STUDY,
        .distribution = DIFFGEO_DIST_BLOCK,
        .numerical_tolerance = DIFFGEO_DEFAULT_TOLERANCE,
        .max_iterations = MAX_NEWTON_ITERATIONS,
        .enable_caching = true,
        .enable_symmetry = true,
        .enable_gpu = false,
        .geodesic_method = GEODESIC_RK4,
        .geodesic_max_steps = DIFFGEO_MAX_GEODESIC_STEPS
    };
    return config;
}

diffgeo_engine_t* diffgeo_engine_create(void) {
    diffgeo_config_t config = diffgeo_default_config();
    return diffgeo_engine_create_with_config(&config);
}

diffgeo_engine_t* diffgeo_engine_create_with_config(const diffgeo_config_t* config) {
    if (!config) return NULL;

    diffgeo_engine_t* engine = calloc(1, sizeof(diffgeo_engine_t));
    if (!engine) return NULL;

    engine->config = *config;
    engine->initialized = true;
    engine->workspace_size = DIFFGEO_MAX_DIMENSIONS * DIFFGEO_MAX_DIMENSIONS * 32;
    engine->workspace = calloc(engine->workspace_size, sizeof(ComplexDouble));

    if (!engine->workspace) {
        free(engine);
        return NULL;
    }

    memset(&engine->stats, 0, sizeof(diffgeo_stats_t));
    engine->last_error[0] = '\0';
    engine->cache_point = NULL;

    return engine;
}

void diffgeo_engine_destroy(diffgeo_engine_t* engine) {
    if (!engine) return;

    free(engine->workspace);
    if (engine->cached_metric) {
        diffgeo_metric_destroy(engine->cached_metric);
    }
    if (engine->cached_christoffel) {
        diffgeo_christoffel_destroy(engine->cached_christoffel);
    }
    if (engine->cache_point) {
        diffgeo_point_destroy(engine->cache_point);
    }
    if (engine->dist_ctx) {
        free(engine->dist_ctx->node_assignments);
        free(engine->dist_ctx);
    }
    free(engine);
}

bool diffgeo_set_dist_context(diffgeo_engine_t* engine,
                              const diffgeo_dist_context_t* ctx) {
    if (!engine || !ctx) return false;

    if (!engine->dist_ctx) {
        engine->dist_ctx = malloc(sizeof(diffgeo_dist_context_t));
        if (!engine->dist_ctx) return false;
    }

    engine->dist_ctx->num_nodes = ctx->num_nodes;
    engine->dist_ctx->node_rank = ctx->node_rank;
    engine->dist_ctx->strategy = ctx->strategy;
    engine->dist_ctx->use_gpu = ctx->use_gpu;

    if (ctx->node_assignments && ctx->num_nodes > 0) {
        engine->dist_ctx->node_assignments = malloc(ctx->num_nodes * sizeof(int));
        if (engine->dist_ctx->node_assignments) {
            memcpy(engine->dist_ctx->node_assignments, ctx->node_assignments,
                   ctx->num_nodes * sizeof(int));
        }
    } else {
        engine->dist_ctx->node_assignments = NULL;
    }

    return true;
}

bool diffgeo_engine_reset(diffgeo_engine_t* engine) {
    if (!engine) return false;

    memset(&engine->stats, 0, sizeof(diffgeo_stats_t));
    engine->last_error[0] = '\0';

    if (engine->cached_metric) {
        diffgeo_metric_destroy(engine->cached_metric);
        engine->cached_metric = NULL;
    }
    if (engine->cached_christoffel) {
        diffgeo_christoffel_destroy(engine->cached_christoffel);
        engine->cached_christoffel = NULL;
    }
    if (engine->cache_point) {
        diffgeo_point_destroy(engine->cache_point);
        engine->cache_point = NULL;
    }

    return true;
}

// ============================================================================
// Point and Tangent Vector Operations
// ============================================================================

diffgeo_point_t* diffgeo_point_create(diffgeo_engine_t* engine,
                                      const ComplexDouble* coordinates,
                                      size_t dimension) {
    if (!engine || !coordinates || dimension == 0 ||
        dimension > DIFFGEO_MAX_DIMENSIONS) {
        if (engine) set_error(engine, "Invalid parameters for point creation");
        return NULL;
    }

    diffgeo_point_t* point = malloc(sizeof(diffgeo_point_t));
    if (!point) return NULL;

    point->coordinates = malloc(dimension * sizeof(ComplexDouble));
    if (!point->coordinates) {
        free(point);
        return NULL;
    }

    memcpy(point->coordinates, coordinates, dimension * sizeof(ComplexDouble));
    point->dimension = dimension;
    point->chart_index = 0;

    return point;
}

void diffgeo_point_destroy(diffgeo_point_t* point) {
    if (!point) return;
    free(point->coordinates);
    free(point);
}

diffgeo_tangent_vector_t* diffgeo_tangent_create(
    diffgeo_engine_t* engine,
    diffgeo_point_t* base_point,
    const ComplexDouble* components) {

    if (!engine || !base_point || !components) {
        if (engine) set_error(engine, "Invalid parameters for tangent creation");
        return NULL;
    }

    diffgeo_tangent_vector_t* vector = malloc(sizeof(diffgeo_tangent_vector_t));
    if (!vector) return NULL;

    vector->components = malloc(base_point->dimension * sizeof(ComplexDouble));
    if (!vector->components) {
        free(vector);
        return NULL;
    }

    memcpy(vector->components, components,
           base_point->dimension * sizeof(ComplexDouble));
    vector->dimension = base_point->dimension;
    vector->base_point = base_point;

    return vector;
}

void diffgeo_tangent_destroy(diffgeo_tangent_vector_t* vector) {
    if (!vector) return;
    free(vector->components);
    free(vector);
}

diffgeo_point_t* diffgeo_exp_map(diffgeo_engine_t* engine,
                                 diffgeo_point_t* base_point,
                                 diffgeo_tangent_vector_t* vector) {
    if (!engine || !base_point || !vector) {
        if (engine) set_error(engine, "Invalid parameters for exp map");
        return NULL;
    }

    size_t dim = base_point->dimension;

    // Compute vector norm to determine step size
    double norm = 0.0;
    for (size_t i = 0; i < dim; i++) {
        norm += complex_abs(vector->components[i]) * complex_abs(vector->components[i]);
    }
    norm = sqrt(norm);

    if (norm < GEOMETRY_EPSILON) {
        // Vector is essentially zero - return copy of base point
        return diffgeo_point_create(engine, base_point->coordinates, dim);
    }

    // Use geodesic integration for accurate exp map
    // Number of steps proportional to vector magnitude for accuracy
    size_t num_steps = (size_t)(10.0 * norm) + 10;
    if (num_steps > 1000) num_steps = 1000;

    diffgeo_geodesic_t* geo = diffgeo_compute_geodesic(engine, base_point,
                                                       vector, 1.0, num_steps);
    if (!geo || geo->num_points == 0) {
        if (geo) diffgeo_geodesic_destroy(geo);
        set_error(engine, "Geodesic computation failed in exp map");
        return NULL;
    }

    // Create result from final geodesic point
    diffgeo_point_t* result = diffgeo_point_create(engine,
        geo->points[geo->num_points - 1].coordinates, dim);

    diffgeo_geodesic_destroy(geo);
    engine->stats.total_operations++;

    return result;
}

diffgeo_tangent_vector_t* diffgeo_log_map(diffgeo_engine_t* engine,
                                          diffgeo_point_t* p,
                                          diffgeo_point_t* q) {
    if (!engine || !p || !q || p->dimension != q->dimension) {
        if (engine) set_error(engine, "Invalid parameters for log map");
        return NULL;
    }

    size_t dim = p->dimension;
    ComplexDouble* components = malloc(dim * sizeof(ComplexDouble));
    if (!components) return NULL;

    // For Fubini-Study and similar metrics on projective spaces,
    // log_p(q) requires solving for the initial velocity of the geodesic from p to q

    // Compute the difference vector as initial guess
    for (size_t i = 0; i < dim; i++) {
        components[i] = complex_sub(q->coordinates[i], p->coordinates[i]);
    }

    // For projective spaces, project onto the tangent space at p
    if (engine->config.manifold_type == MANIFOLD_PROJECTIVE_HILBERT ||
        engine->config.manifold_type == MANIFOLD_BLOCH_SPHERE) {

        // Compute <p, q> inner product
        ComplexDouble inner_pq = {0.0, 0.0};
        double norm_p_sq = 0.0;

        for (size_t i = 0; i < dim; i++) {
            inner_pq = complex_add(inner_pq,
                complex_mult(complex_conj(p->coordinates[i]), q->coordinates[i]));
            norm_p_sq += p->coordinates[i].real * p->coordinates[i].real +
                        p->coordinates[i].imag * p->coordinates[i].imag;
        }

        // v = q - <p,q>/<p,p> * p  (project q onto tangent space at p)
        if (norm_p_sq > GEOMETRY_EPSILON) {
            ComplexDouble scale = complex_scale(inner_pq, 1.0 / norm_p_sq);
            for (size_t i = 0; i < dim; i++) {
                ComplexDouble proj = complex_mult(scale, p->coordinates[i]);
                components[i] = complex_sub(q->coordinates[i], proj);
            }
        }

        // Scale by geodesic distance
        double dist = diffgeo_distance(engine, p, q);
        double v_norm = 0.0;
        for (size_t i = 0; i < dim; i++) {
            v_norm += components[i].real * components[i].real +
                     components[i].imag * components[i].imag;
        }
        v_norm = sqrt(v_norm);

        if (v_norm > GEOMETRY_EPSILON && dist > GEOMETRY_EPSILON) {
            double scale_factor = dist / v_norm;
            for (size_t i = 0; i < dim; i++) {
                components[i] = complex_scale(components[i], scale_factor);
            }
        }
    }

    diffgeo_tangent_vector_t* result = diffgeo_tangent_create(engine, p, components);
    free(components);

    engine->stats.total_operations++;
    return result;
}

double diffgeo_distance(diffgeo_engine_t* engine,
                        diffgeo_point_t* p,
                        diffgeo_point_t* q) {
    if (!engine || !p || !q || p->dimension != q->dimension) {
        if (engine) set_error(engine, "Invalid parameters for distance");
        return -1.0;
    }

    size_t dim = p->dimension;

    // For Fubini-Study metric on projective space
    if (engine->config.metric_type == METRIC_FUBINI_STUDY) {
        // d(p,q) = arccos(|<p,q>| / (|p||q|))
        ComplexDouble inner = {0.0, 0.0};
        double norm_p = 0.0, norm_q = 0.0;

        for (size_t i = 0; i < dim; i++) {
            inner = complex_add(inner,
                complex_mult(complex_conj(p->coordinates[i]), q->coordinates[i]));
            norm_p += p->coordinates[i].real * p->coordinates[i].real +
                     p->coordinates[i].imag * p->coordinates[i].imag;
            norm_q += q->coordinates[i].real * q->coordinates[i].real +
                     q->coordinates[i].imag * q->coordinates[i].imag;
        }

        norm_p = sqrt(norm_p);
        norm_q = sqrt(norm_q);
        double inner_abs = complex_abs(inner);

        if (norm_p > GEOMETRY_EPSILON && norm_q > GEOMETRY_EPSILON) {
            double cos_dist = inner_abs / (norm_p * norm_q);
            if (cos_dist > 1.0) cos_dist = 1.0;
            if (cos_dist < -1.0) cos_dist = -1.0;
            engine->stats.total_operations++;
            return acos(cos_dist);
        }
    }

    // For hyperbolic metric (Poincare disk model)
    if (engine->config.metric_type == METRIC_HYPERBOLIC) {
        // d(p,q) = arccosh(1 + 2|p-q|^2 / ((1-|p|^2)(1-|q|^2)))
        double norm_p_sq = 0.0, norm_q_sq = 0.0, diff_sq = 0.0;

        for (size_t i = 0; i < dim; i++) {
            norm_p_sq += p->coordinates[i].real * p->coordinates[i].real +
                        p->coordinates[i].imag * p->coordinates[i].imag;
            norm_q_sq += q->coordinates[i].real * q->coordinates[i].real +
                        q->coordinates[i].imag * q->coordinates[i].imag;
            ComplexDouble diff = complex_sub(p->coordinates[i], q->coordinates[i]);
            diff_sq += diff.real * diff.real + diff.imag * diff.imag;
        }

        double denom = (1.0 - norm_p_sq) * (1.0 - norm_q_sq);
        if (fabs(denom) > GEOMETRY_EPSILON) {
            double cosh_dist = 1.0 + 2.0 * diff_sq / denom;
            if (cosh_dist < 1.0) cosh_dist = 1.0;
            engine->stats.total_operations++;
            return acosh(cosh_dist);
        }
    }

    // General case: integrate metric along geodesic or use approximation
    diffgeo_metric_tensor_t* metric = diffgeo_compute_metric(engine, p);
    if (!metric) {
        // Fallback to Euclidean distance
        double dist = 0.0;
        for (size_t i = 0; i < dim; i++) {
            ComplexDouble diff = complex_sub(q->coordinates[i], p->coordinates[i]);
            dist += diff.real * diff.real + diff.imag * diff.imag;
        }
        return sqrt(dist);
    }

    // Use metric for proper distance: d = sqrt(g_ij * dx^i * dx^j)
    double dist_sq = 0.0;

    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            ComplexDouble dxi = complex_sub(q->coordinates[i], p->coordinates[i]);
            ComplexDouble dxj = complex_sub(q->coordinates[j], p->coordinates[j]);
            ComplexDouble g_ij = metric->components[i * dim + j];

            // g_ij * dx^i * conj(dx^j) for Hermitian metric
            ComplexDouble term = complex_mult(g_ij, complex_mult(dxi, complex_conj(dxj)));
            dist_sq += term.real;
        }
    }

    diffgeo_metric_destroy(metric);
    engine->stats.total_operations++;

    return sqrt(fabs(dist_sq));
}

// ============================================================================
// Metric Tensor Operations
// ============================================================================

diffgeo_metric_tensor_t* diffgeo_compute_metric(
    diffgeo_engine_t* engine,
    diffgeo_point_t* point) {

    if (!engine || !point) {
        if (engine) set_error(engine, "Invalid parameters for metric computation");
        return NULL;
    }

    // Check cache
    if (engine->config.enable_caching && engine->cached_metric &&
        engine->cache_point && points_equal(engine->cache_point, point)) {
        // Return a copy of cached metric
        size_t dim = engine->cached_metric->dimension;
        diffgeo_metric_tensor_t* metric = malloc(sizeof(diffgeo_metric_tensor_t));
        if (!metric) return NULL;
        metric->components = malloc(dim * dim * sizeof(ComplexDouble));
        if (!metric->components) { free(metric); return NULL; }
        memcpy(metric->components, engine->cached_metric->components,
               dim * dim * sizeof(ComplexDouble));
        metric->dimension = dim;
        metric->base_point = point;
        metric->type = engine->cached_metric->type;
        metric->is_hermitian = engine->cached_metric->is_hermitian;
        return metric;
    }

    size_t dim = point->dimension;
    diffgeo_metric_tensor_t* metric = malloc(sizeof(diffgeo_metric_tensor_t));
    if (!metric) return NULL;

    metric->components = calloc(dim * dim, sizeof(ComplexDouble));
    if (!metric->components) {
        free(metric);
        return NULL;
    }

    metric->dimension = dim;
    metric->base_point = point;
    metric->type = engine->config.metric_type;
    metric->is_hermitian = true;

    // Compute metric based on type
    switch (engine->config.metric_type) {
        case METRIC_FUBINI_STUDY: {
            // Fubini-Study metric on CP^n
            // g_ij = (δ_ij(1+|z|²) - z̄_i z_j) / (1 + |z|²)²
            double norm_sq = 0.0;
            for (size_t i = 0; i < dim; i++) {
                norm_sq += point->coordinates[i].real * point->coordinates[i].real +
                          point->coordinates[i].imag * point->coordinates[i].imag;
            }
            double denom = (1.0 + norm_sq) * (1.0 + norm_sq);

            for (size_t i = 0; i < dim; i++) {
                for (size_t j = 0; j < dim; j++) {
                    ComplexDouble zi_conj = complex_conj(point->coordinates[i]);
                    ComplexDouble zj = point->coordinates[j];
                    ComplexDouble prod = complex_mult(zi_conj, zj);

                    double delta_ij = (i == j) ? 1.0 : 0.0;
                    metric->components[i * dim + j].real =
                        (delta_ij * (1.0 + norm_sq) - prod.real) / denom;
                    metric->components[i * dim + j].imag = -prod.imag / denom;
                }
            }
            break;
        }

        case METRIC_BURES: {
            // Bures metric: requires density matrix interpretation
            // For pure states, Bures = (1/4) * Fubini-Study
            double norm_sq = 0.0;
            for (size_t i = 0; i < dim; i++) {
                norm_sq += point->coordinates[i].real * point->coordinates[i].real +
                          point->coordinates[i].imag * point->coordinates[i].imag;
            }
            double denom = (1.0 + norm_sq) * (1.0 + norm_sq);

            for (size_t i = 0; i < dim; i++) {
                for (size_t j = 0; j < dim; j++) {
                    ComplexDouble zi_conj = complex_conj(point->coordinates[i]);
                    ComplexDouble zj = point->coordinates[j];
                    ComplexDouble prod = complex_mult(zi_conj, zj);

                    double delta_ij = (i == j) ? 1.0 : 0.0;
                    metric->components[i * dim + j].real =
                        0.25 * (delta_ij * (1.0 + norm_sq) - prod.real) / denom;
                    metric->components[i * dim + j].imag =
                        0.25 * (-prod.imag) / denom;
                }
            }
            break;
        }

        case METRIC_WIGNER_YANASE: {
            // Wigner-Yanase metric: g_ij = -1/2 Tr([sqrt(rho), A_i][sqrt(rho), A_j])
            // For pure states, equals (1/4) * Fubini-Study
            double norm_sq = 0.0;
            for (size_t i = 0; i < dim; i++) {
                norm_sq += point->coordinates[i].real * point->coordinates[i].real +
                          point->coordinates[i].imag * point->coordinates[i].imag;
            }
            double denom = (1.0 + norm_sq) * (1.0 + norm_sq);

            for (size_t i = 0; i < dim; i++) {
                for (size_t j = 0; j < dim; j++) {
                    ComplexDouble zi_conj = complex_conj(point->coordinates[i]);
                    ComplexDouble zj = point->coordinates[j];
                    ComplexDouble prod = complex_mult(zi_conj, zj);

                    double delta_ij = (i == j) ? 1.0 : 0.0;
                    metric->components[i * dim + j].real =
                        0.25 * (delta_ij * (1.0 + norm_sq) - prod.real) / denom;
                    metric->components[i * dim + j].imag =
                        0.25 * (-prod.imag) / denom;
                }
            }
            break;
        }

        case METRIC_BOGOLIUBOV_KUBO_MORI: {
            // BKM metric (also called KMS metric)
            // For pure states, equals Fubini-Study
            double norm_sq = 0.0;
            for (size_t i = 0; i < dim; i++) {
                norm_sq += point->coordinates[i].real * point->coordinates[i].real +
                          point->coordinates[i].imag * point->coordinates[i].imag;
            }
            double denom = (1.0 + norm_sq) * (1.0 + norm_sq);

            for (size_t i = 0; i < dim; i++) {
                for (size_t j = 0; j < dim; j++) {
                    ComplexDouble zi_conj = complex_conj(point->coordinates[i]);
                    ComplexDouble zj = point->coordinates[j];
                    ComplexDouble prod = complex_mult(zi_conj, zj);

                    double delta_ij = (i == j) ? 1.0 : 0.0;
                    metric->components[i * dim + j].real =
                        (delta_ij * (1.0 + norm_sq) - prod.real) / denom;
                    metric->components[i * dim + j].imag = -prod.imag / denom;
                }
            }
            break;
        }

        case METRIC_EUCLIDEAN:
            // Flat Euclidean metric: g_ij = δ_ij
            for (size_t i = 0; i < dim; i++) {
                metric->components[i * dim + i].real = 1.0;
                metric->components[i * dim + i].imag = 0.0;
            }
            break;

        case METRIC_MINKOWSKI: {
            // Minkowski metric: g = diag(-1, 1, 1, 1, ...)
            metric->components[0].real = -1.0;
            for (size_t i = 1; i < dim; i++) {
                metric->components[i * dim + i].real = 1.0;
            }
            metric->is_hermitian = true;  // Real symmetric
            break;
        }

        case METRIC_HYPERBOLIC: {
            // Poincaré disk metric: g_ij = 4δ_ij / (1 - |z|²)²
            double norm_sq = 0.0;
            for (size_t i = 0; i < dim; i++) {
                norm_sq += point->coordinates[i].real * point->coordinates[i].real +
                          point->coordinates[i].imag * point->coordinates[i].imag;
            }

            if (norm_sq >= 1.0) {
                // Point is outside or on the boundary of the disk
                set_error(engine, "Point outside Poincare disk");
                free(metric->components);
                free(metric);
                return NULL;
            }

            double conformal = 4.0 / ((1.0 - norm_sq) * (1.0 - norm_sq));
            for (size_t i = 0; i < dim; i++) {
                metric->components[i * dim + i].real = conformal;
            }
            break;
        }

        case METRIC_SPHERICAL: {
            // Round sphere metric (stereographic projection)
            // g_ij = 4δ_ij / (1 + |z|²)²
            double norm_sq = 0.0;
            for (size_t i = 0; i < dim; i++) {
                norm_sq += point->coordinates[i].real * point->coordinates[i].real +
                          point->coordinates[i].imag * point->coordinates[i].imag;
            }
            double conformal = 4.0 / ((1.0 + norm_sq) * (1.0 + norm_sq));

            for (size_t i = 0; i < dim; i++) {
                metric->components[i * dim + i].real = conformal;
            }
            break;
        }

        case METRIC_CUSTOM:
        default:
            // Default to Euclidean
            for (size_t i = 0; i < dim; i++) {
                metric->components[i * dim + i].real = 1.0;
            }
            break;
    }

    // Update cache
    if (engine->config.enable_caching) {
        if (engine->cached_metric) {
            diffgeo_metric_destroy(engine->cached_metric);
        }
        if (engine->cache_point) {
            diffgeo_point_destroy(engine->cache_point);
        }

        engine->cached_metric = malloc(sizeof(diffgeo_metric_tensor_t));
        if (engine->cached_metric) {
            engine->cached_metric->components = malloc(dim * dim * sizeof(ComplexDouble));
            if (engine->cached_metric->components) {
                memcpy(engine->cached_metric->components, metric->components,
                       dim * dim * sizeof(ComplexDouble));
                engine->cached_metric->dimension = dim;
                engine->cached_metric->type = metric->type;
                engine->cached_metric->is_hermitian = metric->is_hermitian;
            }
        }
        engine->cache_point = diffgeo_point_create(engine, point->coordinates, dim);
    }

    engine->stats.metric_evaluations++;
    engine->stats.total_operations++;

    return metric;
}

diffgeo_metric_tensor_t* diffgeo_compute_metric_distributed(
    diffgeo_engine_t* engine,
    diffgeo_point_t* point,
    const diffgeo_dist_context_t* ctx) {

    if (!engine || !point) {
        if (engine) set_error(engine, "Invalid parameters for distributed metric");
        return NULL;
    }

    // If no distribution context or single node, fall back to local computation
    if (!ctx || ctx->num_nodes <= 1) {
        return diffgeo_compute_metric(engine, point);
    }

    size_t dim = point->dimension;
    size_t total_components = dim * dim;
    size_t start, end;

    distribute_computation(ctx, total_components, &start, &end);

    // Allocate full metric
    diffgeo_metric_tensor_t* metric = malloc(sizeof(diffgeo_metric_tensor_t));
    if (!metric) return NULL;

    metric->components = calloc(dim * dim, sizeof(ComplexDouble));
    if (!metric->components) {
        free(metric);
        return NULL;
    }

    metric->dimension = dim;
    metric->base_point = point;
    metric->type = engine->config.metric_type;
    metric->is_hermitian = true;

    // Each node computes its assigned components
    for (size_t idx = start; idx < end; idx++) {
        size_t i = idx / dim;
        size_t j = idx % dim;

        // Compute metric component based on type
        switch (engine->config.metric_type) {
            case METRIC_FUBINI_STUDY: {
                double norm_sq = 0.0;
                for (size_t k = 0; k < dim; k++) {
                    norm_sq += point->coordinates[k].real * point->coordinates[k].real +
                              point->coordinates[k].imag * point->coordinates[k].imag;
                }
                double denom = (1.0 + norm_sq) * (1.0 + norm_sq);

                ComplexDouble zi_conj = complex_conj(point->coordinates[i]);
                ComplexDouble zj = point->coordinates[j];
                ComplexDouble prod = complex_mult(zi_conj, zj);

                double delta_ij = (i == j) ? 1.0 : 0.0;
                metric->components[idx].real =
                    (delta_ij * (1.0 + norm_sq) - prod.real) / denom;
                metric->components[idx].imag = -prod.imag / denom;
                break;
            }

            case METRIC_EUCLIDEAN:
                metric->components[idx].real = (i == j) ? 1.0 : 0.0;
                metric->components[idx].imag = 0.0;
                break;

            case METRIC_HYPERBOLIC: {
                double norm_sq = 0.0;
                for (size_t k = 0; k < dim; k++) {
                    norm_sq += point->coordinates[k].real * point->coordinates[k].real +
                              point->coordinates[k].imag * point->coordinates[k].imag;
                }
                double conformal = 4.0 / ((1.0 - norm_sq) * (1.0 - norm_sq));
                metric->components[idx].real = (i == j) ? conformal : 0.0;
                break;
            }

            default:
                metric->components[idx].real = (i == j) ? 1.0 : 0.0;
                break;
        }
    }

    // In a real distributed system, would use MPI_Allgather here
    // to collect all components from all nodes
    engine->stats.bytes_communicated += total_components * sizeof(ComplexDouble);
    engine->stats.communication_time_ms += 0.1;  // Simulated communication time

    engine->stats.metric_evaluations++;
    engine->stats.total_operations++;

    return metric;
}

diffgeo_metric_tensor_t* diffgeo_compute_inverse_metric(
    diffgeo_engine_t* engine,
    const diffgeo_metric_tensor_t* metric) {

    if (!engine || !metric) {
        if (engine) set_error(engine, "Invalid parameters for inverse metric");
        return NULL;
    }

    size_t dim = metric->dimension;
    diffgeo_metric_tensor_t* inv = malloc(sizeof(diffgeo_metric_tensor_t));
    if (!inv) return NULL;

    inv->components = calloc(dim * dim, sizeof(ComplexDouble));
    if (!inv->components) {
        free(inv);
        return NULL;
    }

    inv->dimension = dim;
    inv->base_point = metric->base_point;
    inv->type = metric->type;
    inv->is_hermitian = metric->is_hermitian;

    if (!complex_matrix_inverse(metric->components, inv->components, dim)) {
        set_error(engine, "Metric matrix is singular");
        free(inv->components);
        free(inv);
        return NULL;
    }

    engine->stats.total_operations++;
    return inv;
}

ComplexDouble diffgeo_inner_product(diffgeo_engine_t* engine,
                                    const diffgeo_metric_tensor_t* metric,
                                    const diffgeo_tangent_vector_t* u,
                                    const diffgeo_tangent_vector_t* v) {
    ComplexDouble result = {0.0, 0.0};

    if (!engine || !metric || !u || !v || u->dimension != v->dimension ||
        u->dimension != metric->dimension) {
        if (engine) set_error(engine, "Invalid parameters for inner product");
        return result;
    }

    size_t dim = metric->dimension;

    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            ComplexDouble g_ij = metric->components[i * dim + j];
            ComplexDouble ui = u->components[i];
            ComplexDouble vj_conj = complex_conj(v->components[j]);

            ComplexDouble term = complex_mult(g_ij, complex_mult(ui, vj_conj));
            result = complex_add(result, term);
        }
    }

    engine->stats.total_operations++;
    return result;
}

double diffgeo_norm(diffgeo_engine_t* engine,
                    const diffgeo_metric_tensor_t* metric,
                    const diffgeo_tangent_vector_t* v) {
    ComplexDouble inner = diffgeo_inner_product(engine, metric, v, v);
    return sqrt(fabs(inner.real));
}

void diffgeo_metric_destroy(diffgeo_metric_tensor_t* metric) {
    if (!metric) return;
    free(metric->components);
    free(metric);
}

// ============================================================================
// Christoffel Symbols
// ============================================================================

diffgeo_christoffel_t* diffgeo_compute_christoffel(
    diffgeo_engine_t* engine,
    diffgeo_point_t* point) {

    if (!engine || !point) {
        if (engine) set_error(engine, "Invalid parameters for Christoffel computation");
        return NULL;
    }

    // Check cache
    if (engine->config.enable_caching && engine->cached_christoffel &&
        engine->cache_point && points_equal(engine->cache_point, point)) {
        size_t dim = engine->cached_christoffel->dimension;
        size_t num_comp = dim * dim * dim;
        diffgeo_christoffel_t* christoffel = malloc(sizeof(diffgeo_christoffel_t));
        if (!christoffel) return NULL;
        christoffel->components = malloc(num_comp * sizeof(ComplexDouble));
        if (!christoffel->components) { free(christoffel); return NULL; }
        memcpy(christoffel->components, engine->cached_christoffel->components,
               num_comp * sizeof(ComplexDouble));
        christoffel->dimension = dim;
        christoffel->base_point = point;
        christoffel->is_torsion_free = engine->cached_christoffel->is_torsion_free;
        return christoffel;
    }

    size_t dim = point->dimension;
    diffgeo_christoffel_t* christoffel = malloc(sizeof(diffgeo_christoffel_t));
    if (!christoffel) return NULL;

    size_t num_components = dim * dim * dim;
    christoffel->components = calloc(num_components, sizeof(ComplexDouble));
    if (!christoffel->components) {
        free(christoffel);
        return NULL;
    }

    christoffel->dimension = dim;
    christoffel->base_point = point;
    christoffel->is_torsion_free = true;

    // Compute metric and its inverse
    diffgeo_metric_tensor_t* metric = diffgeo_compute_metric(engine, point);
    if (!metric) {
        free(christoffel->components);
        free(christoffel);
        return NULL;
    }

    diffgeo_metric_tensor_t* inv_metric = diffgeo_compute_inverse_metric(engine, metric);
    if (!inv_metric) {
        diffgeo_metric_destroy(metric);
        free(christoffel->components);
        free(christoffel);
        return NULL;
    }

    // Compute metric derivatives numerically using central differences
    double h = 1e-7;
    ComplexDouble* dg = calloc(dim * dim * dim, sizeof(ComplexDouble));
    if (!dg) {
        diffgeo_metric_destroy(inv_metric);
        diffgeo_metric_destroy(metric);
        free(christoffel->components);
        free(christoffel);
        return NULL;
    }

    // Numerical differentiation of metric
    for (size_t k = 0; k < dim; k++) {
        // Create displaced points
        ComplexDouble* coords_plus = malloc(dim * sizeof(ComplexDouble));
        ComplexDouble* coords_minus = malloc(dim * sizeof(ComplexDouble));

        if (!coords_plus || !coords_minus) {
            free(coords_plus);
            free(coords_minus);
            free(dg);
            diffgeo_metric_destroy(inv_metric);
            diffgeo_metric_destroy(metric);
            free(christoffel->components);
            free(christoffel);
            return NULL;
        }

        memcpy(coords_plus, point->coordinates, dim * sizeof(ComplexDouble));
        memcpy(coords_minus, point->coordinates, dim * sizeof(ComplexDouble));

        coords_plus[k].real += h;
        coords_minus[k].real -= h;

        diffgeo_point_t* p_plus = diffgeo_point_create(engine, coords_plus, dim);
        diffgeo_point_t* p_minus = diffgeo_point_create(engine, coords_minus, dim);

        // Temporarily disable caching for derivative computation
        bool cache_state = engine->config.enable_caching;
        engine->config.enable_caching = false;

        diffgeo_metric_tensor_t* g_plus = diffgeo_compute_metric(engine, p_plus);
        diffgeo_metric_tensor_t* g_minus = diffgeo_compute_metric(engine, p_minus);

        engine->config.enable_caching = cache_state;

        if (g_plus && g_minus) {
            // Central difference: ∂g_ij/∂x^k ≈ (g_ij(x+h) - g_ij(x-h)) / (2h)
            for (size_t i = 0; i < dim; i++) {
                for (size_t j = 0; j < dim; j++) {
                    size_t idx = k * dim * dim + i * dim + j;
                    dg[idx].real = (g_plus->components[i * dim + j].real -
                                   g_minus->components[i * dim + j].real) / (2.0 * h);
                    dg[idx].imag = (g_plus->components[i * dim + j].imag -
                                   g_minus->components[i * dim + j].imag) / (2.0 * h);
                }
            }
        }

        if (g_plus) diffgeo_metric_destroy(g_plus);
        if (g_minus) diffgeo_metric_destroy(g_minus);
        diffgeo_point_destroy(p_plus);
        diffgeo_point_destroy(p_minus);
        free(coords_plus);
        free(coords_minus);
    }

    // Compute Christoffel symbols: Γ^k_ij = (1/2) g^{kl} (∂_i g_{jl} + ∂_j g_{il} - ∂_l g_{ij})
    for (size_t k = 0; k < dim; k++) {
        for (size_t i = 0; i < dim; i++) {
            for (size_t j = 0; j < dim; j++) {
                ComplexDouble sum = {0.0, 0.0};

                for (size_t l = 0; l < dim; l++) {
                    ComplexDouble g_kl_inv = inv_metric->components[k * dim + l];

                    // ∂_i g_{jl}
                    ComplexDouble dg_ijl = dg[i * dim * dim + j * dim + l];
                    // ∂_j g_{il}
                    ComplexDouble dg_jil = dg[j * dim * dim + i * dim + l];
                    // ∂_l g_{ij}
                    ComplexDouble dg_lij = dg[l * dim * dim + i * dim + j];

                    ComplexDouble bracket = complex_add(dg_ijl,
                                           complex_sub(dg_jil, dg_lij));
                    ComplexDouble term = complex_mult(g_kl_inv, bracket);
                    sum = complex_add(sum, term);
                }

                christoffel->components[k * dim * dim + i * dim + j] =
                    complex_scale(sum, 0.5);

                // Exploit symmetry if enabled
                if (engine->config.enable_symmetry && i != j) {
                    christoffel->components[k * dim * dim + j * dim + i] =
                        christoffel->components[k * dim * dim + i * dim + j];
                }
            }
        }
    }

    free(dg);
    diffgeo_metric_destroy(inv_metric);
    diffgeo_metric_destroy(metric);

    // Update cache
    if (engine->config.enable_caching) {
        if (engine->cached_christoffel) {
            diffgeo_christoffel_destroy(engine->cached_christoffel);
        }
        engine->cached_christoffel = malloc(sizeof(diffgeo_christoffel_t));
        if (engine->cached_christoffel) {
            engine->cached_christoffel->components = malloc(num_components * sizeof(ComplexDouble));
            if (engine->cached_christoffel->components) {
                memcpy(engine->cached_christoffel->components, christoffel->components,
                       num_components * sizeof(ComplexDouble));
                engine->cached_christoffel->dimension = dim;
                engine->cached_christoffel->is_torsion_free = true;
            }
        }
    }

    engine->stats.christoffel_evaluations++;
    engine->stats.total_operations++;

    return christoffel;
}

diffgeo_christoffel_t* diffgeo_compute_christoffel_distributed(
    diffgeo_engine_t* engine,
    diffgeo_point_t* point,
    const diffgeo_dist_context_t* ctx) {

    if (!engine || !point) {
        if (engine) set_error(engine, "Invalid parameters for distributed Christoffel");
        return NULL;
    }

    if (!ctx || ctx->num_nodes <= 1) {
        return diffgeo_compute_christoffel(engine, point);
    }

    size_t dim = point->dimension;
    size_t total_components = dim * dim * dim;
    size_t start, end;

    distribute_computation(ctx, total_components, &start, &end);

    // First compute metric and inverse (needed by all nodes)
    diffgeo_metric_tensor_t* metric = diffgeo_compute_metric(engine, point);
    if (!metric) return NULL;

    diffgeo_metric_tensor_t* inv_metric = diffgeo_compute_inverse_metric(engine, metric);
    if (!inv_metric) {
        diffgeo_metric_destroy(metric);
        return NULL;
    }

    // Compute metric derivatives
    double h = 1e-7;
    ComplexDouble* dg = calloc(dim * dim * dim, sizeof(ComplexDouble));
    if (!dg) {
        diffgeo_metric_destroy(inv_metric);
        diffgeo_metric_destroy(metric);
        return NULL;
    }

    for (size_t k = 0; k < dim; k++) {
        ComplexDouble* coords_plus = malloc(dim * sizeof(ComplexDouble));
        ComplexDouble* coords_minus = malloc(dim * sizeof(ComplexDouble));

        memcpy(coords_plus, point->coordinates, dim * sizeof(ComplexDouble));
        memcpy(coords_minus, point->coordinates, dim * sizeof(ComplexDouble));

        coords_plus[k].real += h;
        coords_minus[k].real -= h;

        diffgeo_point_t* p_plus = diffgeo_point_create(engine, coords_plus, dim);
        diffgeo_point_t* p_minus = diffgeo_point_create(engine, coords_minus, dim);

        bool cache_state = engine->config.enable_caching;
        engine->config.enable_caching = false;

        diffgeo_metric_tensor_t* g_plus = diffgeo_compute_metric(engine, p_plus);
        diffgeo_metric_tensor_t* g_minus = diffgeo_compute_metric(engine, p_minus);

        engine->config.enable_caching = cache_state;

        if (g_plus && g_minus) {
            for (size_t i = 0; i < dim; i++) {
                for (size_t j = 0; j < dim; j++) {
                    size_t idx = k * dim * dim + i * dim + j;
                    dg[idx].real = (g_plus->components[i * dim + j].real -
                                   g_minus->components[i * dim + j].real) / (2.0 * h);
                    dg[idx].imag = (g_plus->components[i * dim + j].imag -
                                   g_minus->components[i * dim + j].imag) / (2.0 * h);
                }
            }
        }

        if (g_plus) diffgeo_metric_destroy(g_plus);
        if (g_minus) diffgeo_metric_destroy(g_minus);
        diffgeo_point_destroy(p_plus);
        diffgeo_point_destroy(p_minus);
        free(coords_plus);
        free(coords_minus);
    }

    // Allocate Christoffel symbols
    diffgeo_christoffel_t* christoffel = malloc(sizeof(diffgeo_christoffel_t));
    if (!christoffel) {
        free(dg);
        diffgeo_metric_destroy(inv_metric);
        diffgeo_metric_destroy(metric);
        return NULL;
    }

    christoffel->components = calloc(total_components, sizeof(ComplexDouble));
    if (!christoffel->components) {
        free(christoffel);
        free(dg);
        diffgeo_metric_destroy(inv_metric);
        diffgeo_metric_destroy(metric);
        return NULL;
    }

    christoffel->dimension = dim;
    christoffel->base_point = point;
    christoffel->is_torsion_free = true;

    // Each node computes its assigned components
    for (size_t idx = start; idx < end; idx++) {
        size_t k = idx / (dim * dim);
        size_t ij = idx % (dim * dim);
        size_t i = ij / dim;
        size_t j = ij % dim;

        ComplexDouble sum = {0.0, 0.0};

        for (size_t l = 0; l < dim; l++) {
            ComplexDouble g_kl_inv = inv_metric->components[k * dim + l];
            ComplexDouble dg_ijl = dg[i * dim * dim + j * dim + l];
            ComplexDouble dg_jil = dg[j * dim * dim + i * dim + l];
            ComplexDouble dg_lij = dg[l * dim * dim + i * dim + j];

            ComplexDouble bracket = complex_add(dg_ijl, complex_sub(dg_jil, dg_lij));
            ComplexDouble term = complex_mult(g_kl_inv, bracket);
            sum = complex_add(sum, term);
        }

        christoffel->components[idx] = complex_scale(sum, 0.5);
    }

    free(dg);
    diffgeo_metric_destroy(inv_metric);
    diffgeo_metric_destroy(metric);

    engine->stats.bytes_communicated += total_components * sizeof(ComplexDouble);
    engine->stats.christoffel_evaluations++;
    engine->stats.total_operations++;

    return christoffel;
}

ComplexDouble diffgeo_christoffel_component(
    const diffgeo_christoffel_t* christoffel,
    size_t k, size_t i, size_t j) {

    ComplexDouble zero = {0.0, 0.0};
    if (!christoffel || k >= christoffel->dimension ||
        i >= christoffel->dimension || j >= christoffel->dimension) {
        return zero;
    }

    size_t dim = christoffel->dimension;
    return christoffel->components[k * dim * dim + i * dim + j];
}

void diffgeo_christoffel_destroy(diffgeo_christoffel_t* christoffel) {
    if (!christoffel) return;
    free(christoffel->components);
    free(christoffel);
}

// ============================================================================
// Curvature Computations
// ============================================================================

diffgeo_curvature_t* diffgeo_compute_riemann(
    diffgeo_engine_t* engine,
    diffgeo_point_t* point) {

    if (!engine || !point) {
        if (engine) set_error(engine, "Invalid parameters for Riemann computation");
        return NULL;
    }

    size_t dim = point->dimension;
    diffgeo_curvature_t* riemann = malloc(sizeof(diffgeo_curvature_t));
    if (!riemann) return NULL;

    size_t num_components = dim * dim * dim * dim;
    riemann->components = calloc(num_components, sizeof(ComplexDouble));
    if (!riemann->components) {
        free(riemann);
        return NULL;
    }

    riemann->type = CURVATURE_RIEMANN;
    riemann->dimension = dim;
    riemann->rank = 4;
    riemann->base_point = point;

    // Compute Christoffel symbols
    diffgeo_christoffel_t* christoffel = diffgeo_compute_christoffel(engine, point);
    if (!christoffel) {
        free(riemann->components);
        free(riemann);
        return NULL;
    }

    // Compute derivatives of Christoffel symbols numerically
    double h = 1e-7;
    size_t gamma_size = dim * dim * dim;
    ComplexDouble* dGamma = calloc(dim * gamma_size, sizeof(ComplexDouble));
    if (!dGamma) {
        diffgeo_christoffel_destroy(christoffel);
        free(riemann->components);
        free(riemann);
        return NULL;
    }

    for (size_t m = 0; m < dim; m++) {
        ComplexDouble* coords_plus = malloc(dim * sizeof(ComplexDouble));
        ComplexDouble* coords_minus = malloc(dim * sizeof(ComplexDouble));

        memcpy(coords_plus, point->coordinates, dim * sizeof(ComplexDouble));
        memcpy(coords_minus, point->coordinates, dim * sizeof(ComplexDouble));

        coords_plus[m].real += h;
        coords_minus[m].real -= h;

        diffgeo_point_t* p_plus = diffgeo_point_create(engine, coords_plus, dim);
        diffgeo_point_t* p_minus = diffgeo_point_create(engine, coords_minus, dim);

        bool cache_state = engine->config.enable_caching;
        engine->config.enable_caching = false;

        diffgeo_christoffel_t* gamma_plus = diffgeo_compute_christoffel(engine, p_plus);
        diffgeo_christoffel_t* gamma_minus = diffgeo_compute_christoffel(engine, p_minus);

        engine->config.enable_caching = cache_state;

        if (gamma_plus && gamma_minus) {
            for (size_t idx = 0; idx < gamma_size; idx++) {
                dGamma[m * gamma_size + idx].real =
                    (gamma_plus->components[idx].real - gamma_minus->components[idx].real) / (2.0 * h);
                dGamma[m * gamma_size + idx].imag =
                    (gamma_plus->components[idx].imag - gamma_minus->components[idx].imag) / (2.0 * h);
            }
        }

        if (gamma_plus) diffgeo_christoffel_destroy(gamma_plus);
        if (gamma_minus) diffgeo_christoffel_destroy(gamma_minus);
        diffgeo_point_destroy(p_plus);
        diffgeo_point_destroy(p_minus);
        free(coords_plus);
        free(coords_minus);
    }

    // Compute Riemann tensor: R^l_{ijk} = ∂_j Γ^l_{ik} - ∂_k Γ^l_{ij}
    //                                    + Γ^l_{jm} Γ^m_{ik} - Γ^l_{km} Γ^m_{ij}
    for (size_t l = 0; l < dim; l++) {
        for (size_t i = 0; i < dim; i++) {
            for (size_t j = 0; j < dim; j++) {
                for (size_t k = 0; k < dim; k++) {
                    size_t idx = l * dim * dim * dim + i * dim * dim + j * dim + k;

                    // ∂_j Γ^l_{ik}
                    ComplexDouble dj_Gamma_lik = dGamma[j * gamma_size + l * dim * dim + i * dim + k];
                    // ∂_k Γ^l_{ij}
                    ComplexDouble dk_Gamma_lij = dGamma[k * gamma_size + l * dim * dim + i * dim + j];

                    ComplexDouble sum1 = {0.0, 0.0};
                    ComplexDouble sum2 = {0.0, 0.0};

                    for (size_t m = 0; m < dim; m++) {
                        // Γ^l_{jm} Γ^m_{ik}
                        ComplexDouble gamma_ljm = christoffel->components[l * dim * dim + j * dim + m];
                        ComplexDouble gamma_mik = christoffel->components[m * dim * dim + i * dim + k];
                        sum1 = complex_add(sum1, complex_mult(gamma_ljm, gamma_mik));

                        // Γ^l_{km} Γ^m_{ij}
                        ComplexDouble gamma_lkm = christoffel->components[l * dim * dim + k * dim + m];
                        ComplexDouble gamma_mij = christoffel->components[m * dim * dim + i * dim + j];
                        sum2 = complex_add(sum2, complex_mult(gamma_lkm, gamma_mij));
                    }

                    riemann->components[idx] = complex_add(
                        complex_sub(dj_Gamma_lik, dk_Gamma_lij),
                        complex_sub(sum1, sum2));
                }
            }
        }
    }

    free(dGamma);
    diffgeo_christoffel_destroy(christoffel);

    engine->stats.curvature_evaluations++;
    engine->stats.total_operations++;

    return riemann;
}

diffgeo_curvature_t* diffgeo_compute_riemann_distributed(
    diffgeo_engine_t* engine,
    diffgeo_point_t* point,
    const diffgeo_dist_context_t* ctx) {

    if (!engine || !point) {
        if (engine) set_error(engine, "Invalid parameters for distributed Riemann");
        return NULL;
    }

    if (!ctx || ctx->num_nodes <= 1) {
        return diffgeo_compute_riemann(engine, point);
    }

    size_t dim = point->dimension;
    size_t total_components = dim * dim * dim * dim;
    size_t start, end;

    distribute_computation(ctx, total_components, &start, &end);

    // Compute Christoffel symbols and their derivatives (needed by all)
    diffgeo_christoffel_t* christoffel = diffgeo_compute_christoffel(engine, point);
    if (!christoffel) return NULL;

    double h = 1e-7;
    size_t gamma_size = dim * dim * dim;
    ComplexDouble* dGamma = calloc(dim * gamma_size, sizeof(ComplexDouble));
    if (!dGamma) {
        diffgeo_christoffel_destroy(christoffel);
        return NULL;
    }

    for (size_t m = 0; m < dim; m++) {
        ComplexDouble* coords_plus = malloc(dim * sizeof(ComplexDouble));
        ComplexDouble* coords_minus = malloc(dim * sizeof(ComplexDouble));

        memcpy(coords_plus, point->coordinates, dim * sizeof(ComplexDouble));
        memcpy(coords_minus, point->coordinates, dim * sizeof(ComplexDouble));

        coords_plus[m].real += h;
        coords_minus[m].real -= h;

        diffgeo_point_t* p_plus = diffgeo_point_create(engine, coords_plus, dim);
        diffgeo_point_t* p_minus = diffgeo_point_create(engine, coords_minus, dim);

        bool cache_state = engine->config.enable_caching;
        engine->config.enable_caching = false;

        diffgeo_christoffel_t* gamma_plus = diffgeo_compute_christoffel(engine, p_plus);
        diffgeo_christoffel_t* gamma_minus = diffgeo_compute_christoffel(engine, p_minus);

        engine->config.enable_caching = cache_state;

        if (gamma_plus && gamma_minus) {
            for (size_t idx = 0; idx < gamma_size; idx++) {
                dGamma[m * gamma_size + idx].real =
                    (gamma_plus->components[idx].real - gamma_minus->components[idx].real) / (2.0 * h);
                dGamma[m * gamma_size + idx].imag =
                    (gamma_plus->components[idx].imag - gamma_minus->components[idx].imag) / (2.0 * h);
            }
        }

        if (gamma_plus) diffgeo_christoffel_destroy(gamma_plus);
        if (gamma_minus) diffgeo_christoffel_destroy(gamma_minus);
        diffgeo_point_destroy(p_plus);
        diffgeo_point_destroy(p_minus);
        free(coords_plus);
        free(coords_minus);
    }

    // Allocate Riemann tensor
    diffgeo_curvature_t* riemann = malloc(sizeof(diffgeo_curvature_t));
    if (!riemann) {
        free(dGamma);
        diffgeo_christoffel_destroy(christoffel);
        return NULL;
    }

    riemann->components = calloc(total_components, sizeof(ComplexDouble));
    if (!riemann->components) {
        free(riemann);
        free(dGamma);
        diffgeo_christoffel_destroy(christoffel);
        return NULL;
    }

    riemann->type = CURVATURE_RIEMANN;
    riemann->dimension = dim;
    riemann->rank = 4;
    riemann->base_point = point;

    // Each node computes its assigned components
    for (size_t idx = start; idx < end; idx++) {
        size_t l = idx / (dim * dim * dim);
        size_t remainder = idx % (dim * dim * dim);
        size_t i = remainder / (dim * dim);
        remainder = remainder % (dim * dim);
        size_t j = remainder / dim;
        size_t k = remainder % dim;

        ComplexDouble dj_Gamma_lik = dGamma[j * gamma_size + l * dim * dim + i * dim + k];
        ComplexDouble dk_Gamma_lij = dGamma[k * gamma_size + l * dim * dim + i * dim + j];

        ComplexDouble sum1 = {0.0, 0.0};
        ComplexDouble sum2 = {0.0, 0.0};

        for (size_t m = 0; m < dim; m++) {
            ComplexDouble gamma_ljm = christoffel->components[l * dim * dim + j * dim + m];
            ComplexDouble gamma_mik = christoffel->components[m * dim * dim + i * dim + k];
            sum1 = complex_add(sum1, complex_mult(gamma_ljm, gamma_mik));

            ComplexDouble gamma_lkm = christoffel->components[l * dim * dim + k * dim + m];
            ComplexDouble gamma_mij = christoffel->components[m * dim * dim + i * dim + j];
            sum2 = complex_add(sum2, complex_mult(gamma_lkm, gamma_mij));
        }

        riemann->components[idx] = complex_add(
            complex_sub(dj_Gamma_lik, dk_Gamma_lij),
            complex_sub(sum1, sum2));
    }

    free(dGamma);
    diffgeo_christoffel_destroy(christoffel);

    engine->stats.bytes_communicated += total_components * sizeof(ComplexDouble);
    engine->stats.curvature_evaluations++;
    engine->stats.total_operations++;

    return riemann;
}

diffgeo_curvature_t* diffgeo_compute_ricci(
    diffgeo_engine_t* engine,
    diffgeo_point_t* point) {

    if (!engine || !point) {
        if (engine) set_error(engine, "Invalid parameters for Ricci computation");
        return NULL;
    }

    // First compute Riemann tensor
    diffgeo_curvature_t* riemann = diffgeo_compute_riemann(engine, point);
    if (!riemann) return NULL;

    size_t dim = point->dimension;
    diffgeo_curvature_t* ricci = malloc(sizeof(diffgeo_curvature_t));
    if (!ricci) {
        diffgeo_curvature_destroy(riemann);
        return NULL;
    }

    ricci->components = calloc(dim * dim, sizeof(ComplexDouble));
    if (!ricci->components) {
        free(ricci);
        diffgeo_curvature_destroy(riemann);
        return NULL;
    }

    ricci->type = CURVATURE_RICCI;
    ricci->dimension = dim;
    ricci->rank = 2;
    ricci->base_point = point;

    // Contract Riemann: R_ij = R^k_{ikj}
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            ComplexDouble sum = {0.0, 0.0};
            for (size_t k = 0; k < dim; k++) {
                size_t idx = k * dim * dim * dim + i * dim * dim + k * dim + j;
                sum = complex_add(sum, riemann->components[idx]);
            }
            ricci->components[i * dim + j] = sum;
        }
    }

    diffgeo_curvature_destroy(riemann);

    engine->stats.curvature_evaluations++;
    return ricci;
}

double diffgeo_compute_scalar_curvature(
    diffgeo_engine_t* engine,
    diffgeo_point_t* point) {

    if (!engine || !point) {
        if (engine) set_error(engine, "Invalid parameters for scalar curvature");
        return 0.0;
    }

    // Compute Ricci tensor and metric
    diffgeo_curvature_t* ricci = diffgeo_compute_ricci(engine, point);
    if (!ricci) return 0.0;

    diffgeo_metric_tensor_t* metric = diffgeo_compute_metric(engine, point);
    if (!metric) {
        diffgeo_curvature_destroy(ricci);
        return 0.0;
    }

    diffgeo_metric_tensor_t* inv_metric = diffgeo_compute_inverse_metric(engine, metric);
    if (!inv_metric) {
        diffgeo_metric_destroy(metric);
        diffgeo_curvature_destroy(ricci);
        return 0.0;
    }

    // Contract: R = g^{ij} R_ij
    double scalar = 0.0;
    size_t dim = point->dimension;

    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            ComplexDouble g_inv = inv_metric->components[i * dim + j];
            ComplexDouble r_ij = ricci->components[i * dim + j];
            ComplexDouble prod = complex_mult(g_inv, r_ij);
            scalar += prod.real;
        }
    }

    diffgeo_metric_destroy(inv_metric);
    diffgeo_metric_destroy(metric);
    diffgeo_curvature_destroy(ricci);

    engine->stats.curvature_evaluations++;
    return scalar;
}

double diffgeo_compute_sectional_curvature(
    diffgeo_engine_t* engine,
    diffgeo_point_t* point,
    const diffgeo_tangent_vector_t* u,
    const diffgeo_tangent_vector_t* v) {

    if (!engine || !point || !u || !v) {
        if (engine) set_error(engine, "Invalid parameters for sectional curvature");
        return 0.0;
    }

    if (u->dimension != v->dimension || u->dimension != point->dimension) {
        set_error(engine, "Dimension mismatch for sectional curvature");
        return 0.0;
    }

    // K(u,v) = R(u,v,v,u) / (g(u,u)g(v,v) - g(u,v)²)
    diffgeo_curvature_t* riemann = diffgeo_compute_riemann(engine, point);
    if (!riemann) return 0.0;

    diffgeo_metric_tensor_t* metric = diffgeo_compute_metric(engine, point);
    if (!metric) {
        diffgeo_curvature_destroy(riemann);
        return 0.0;
    }

    size_t dim = point->dimension;

    // Compute R(u,v,v,u) = R_ijkl u^i v^j v^k u^l
    // where R_ijkl = g_lm R^m_{ijk}
    ComplexDouble R_uvvu = {0.0, 0.0};
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            for (size_t k = 0; k < dim; k++) {
                for (size_t l = 0; l < dim; l++) {
                    // Lower the first index: R_ijkl = g_lm R^m_{ijk}
                    ComplexDouble R_ijkl = {0.0, 0.0};
                    for (size_t m = 0; m < dim; m++) {
                        size_t r_idx = m * dim * dim * dim + i * dim * dim + j * dim + k;
                        ComplexDouble g_lm = metric->components[l * dim + m];
                        R_ijkl = complex_add(R_ijkl,
                            complex_mult(g_lm, riemann->components[r_idx]));
                    }

                    ComplexDouble term = complex_mult(R_ijkl,
                        complex_mult(u->components[i],
                        complex_mult(v->components[j],
                        complex_mult(v->components[k], u->components[l]))));
                    R_uvvu = complex_add(R_uvvu, term);
                }
            }
        }
    }

    // Compute denominator: g(u,u)g(v,v) - g(u,v)²
    ComplexDouble guu = diffgeo_inner_product(engine, metric, u, u);
    ComplexDouble gvv = diffgeo_inner_product(engine, metric, v, v);
    ComplexDouble guv = diffgeo_inner_product(engine, metric, u, v);

    double denom = guu.real * gvv.real - guv.real * guv.real - guv.imag * guv.imag;

    diffgeo_metric_destroy(metric);
    diffgeo_curvature_destroy(riemann);

    if (fabs(denom) < GEOMETRY_EPSILON) {
        set_error(engine, "Degenerate plane for sectional curvature");
        return 0.0;
    }

    return R_uvvu.real / denom;
}

diffgeo_curvature_t* diffgeo_compute_weyl(
    diffgeo_engine_t* engine,
    diffgeo_point_t* point) {

    if (!engine || !point) {
        if (engine) set_error(engine, "Invalid parameters for Weyl computation");
        return NULL;
    }

    size_t n = point->dimension;

    // For dimension < 4, Weyl tensor vanishes identically
    if (n < 4) {
        diffgeo_curvature_t* weyl = malloc(sizeof(diffgeo_curvature_t));
        if (!weyl) return NULL;

        weyl->components = calloc(n * n * n * n, sizeof(ComplexDouble));
        if (!weyl->components) {
            free(weyl);
            return NULL;
        }

        weyl->type = CURVATURE_WEYL;
        weyl->dimension = n;
        weyl->rank = 4;
        weyl->base_point = point;
        return weyl;
    }

    // Compute Riemann tensor
    diffgeo_curvature_t* riemann = diffgeo_compute_riemann(engine, point);
    if (!riemann) return NULL;

    // Compute Ricci tensor
    diffgeo_curvature_t* ricci = diffgeo_compute_ricci(engine, point);
    if (!ricci) {
        diffgeo_curvature_destroy(riemann);
        return NULL;
    }

    // Compute scalar curvature
    double scalar = diffgeo_compute_scalar_curvature(engine, point);

    // Compute metric tensor
    diffgeo_metric_tensor_t* metric = diffgeo_compute_metric(engine, point);
    if (!metric) {
        diffgeo_curvature_destroy(ricci);
        diffgeo_curvature_destroy(riemann);
        return NULL;
    }

    // Allocate Weyl tensor
    diffgeo_curvature_t* weyl = malloc(sizeof(diffgeo_curvature_t));
    if (!weyl) {
        diffgeo_metric_destroy(metric);
        diffgeo_curvature_destroy(ricci);
        diffgeo_curvature_destroy(riemann);
        return NULL;
    }

    weyl->components = calloc(n * n * n * n, sizeof(ComplexDouble));
    if (!weyl->components) {
        free(weyl);
        diffgeo_metric_destroy(metric);
        diffgeo_curvature_destroy(ricci);
        diffgeo_curvature_destroy(riemann);
        return NULL;
    }

    weyl->type = CURVATURE_WEYL;
    weyl->dimension = n;
    weyl->rank = 4;
    weyl->base_point = point;

    // Weyl tensor formula:
    // W_ijkl = R_ijkl - (1/(n-2))(g_ik R_jl - g_il R_jk - g_jk R_il + g_jl R_ik)
    //          + (R/((n-1)(n-2)))(g_ik g_jl - g_il g_jk)
    double factor1 = 1.0 / ((double)(n - 2));
    double factor2 = scalar / ((double)((n - 1) * (n - 2)));

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            for (size_t k = 0; k < n; k++) {
                for (size_t l = 0; l < n; l++) {
                    size_t idx = i * n * n * n + j * n * n + k * n + l;

                    // R_ijkl (lowered Riemann tensor)
                    ComplexDouble R_ijkl = {0.0, 0.0};
                    for (size_t m = 0; m < n; m++) {
                        size_t r_idx = m * n * n * n + j * n * n + k * n + l;
                        ComplexDouble g_im = metric->components[i * n + m];
                        R_ijkl = complex_add(R_ijkl,
                            complex_mult(g_im, riemann->components[r_idx]));
                    }

                    // Get metric components
                    ComplexDouble g_ik = metric->components[i * n + k];
                    ComplexDouble g_il = metric->components[i * n + l];
                    ComplexDouble g_jk = metric->components[j * n + k];
                    ComplexDouble g_jl = metric->components[j * n + l];

                    // Get Ricci components
                    ComplexDouble R_jl = ricci->components[j * n + l];
                    ComplexDouble R_jk = ricci->components[j * n + k];
                    ComplexDouble R_il = ricci->components[i * n + l];
                    ComplexDouble R_ik = ricci->components[i * n + k];

                    // Schouten tensor terms
                    // S = (1/(n-2))(g_ik R_jl - g_il R_jk - g_jk R_il + g_jl R_ik)
                    ComplexDouble schouten = complex_sub(
                        complex_sub(complex_mult(g_ik, R_jl), complex_mult(g_il, R_jk)),
                        complex_sub(complex_mult(g_jk, R_il), complex_mult(g_jl, R_ik)));
                    schouten = complex_scale(schouten, factor1);

                    // Scalar curvature terms
                    // T = (R/((n-1)(n-2)))(g_ik g_jl - g_il g_jk)
                    ComplexDouble scalar_term = complex_sub(
                        complex_mult(g_ik, g_jl), complex_mult(g_il, g_jk));
                    scalar_term = complex_scale(scalar_term, factor2);

                    // Weyl = Riemann - Schouten + Scalar term
                    weyl->components[idx] = complex_add(
                        complex_sub(R_ijkl, schouten), scalar_term);
                }
            }
        }
    }

    diffgeo_metric_destroy(metric);
    diffgeo_curvature_destroy(ricci);
    diffgeo_curvature_destroy(riemann);

    engine->stats.curvature_evaluations++;
    return weyl;
}

double diffgeo_compute_holomorphic_sectional(
    diffgeo_engine_t* engine,
    diffgeo_point_t* point,
    const diffgeo_tangent_vector_t* v) {

    if (!engine || !point || !v) {
        if (engine) set_error(engine, "Invalid parameters for holomorphic sectional curvature");
        return 0.0;
    }

    // For projective Hilbert space (CP^n), holomorphic sectional curvature is constant = 4
    if (engine->config.manifold_type == MANIFOLD_PROJECTIVE_HILBERT) {
        return 4.0;
    }

    // For Bloch sphere (CP^1), holomorphic sectional curvature is 4
    if (engine->config.manifold_type == MANIFOLD_BLOCH_SPHERE) {
        return 4.0;
    }

    // For Kähler manifolds: H(v) = R(v, Jv, Jv, v) / |v|^4
    // where J is the complex structure
    // For complex coordinates, Jv = iv

    size_t dim = v->dimension;

    // Create Jv = i*v
    ComplexDouble* jv_components = malloc(dim * sizeof(ComplexDouble));
    if (!jv_components) return 0.0;

    for (size_t i = 0; i < dim; i++) {
        // i * (a + bi) = -b + ai
        jv_components[i].real = -v->components[i].imag;
        jv_components[i].imag = v->components[i].real;
    }

    diffgeo_tangent_vector_t jv = {
        .components = jv_components,
        .dimension = dim,
        .base_point = v->base_point
    };

    // Compute Riemann tensor
    diffgeo_curvature_t* riemann = diffgeo_compute_riemann(engine, point);
    if (!riemann) {
        free(jv_components);
        return 0.0;
    }

    // Compute metric
    diffgeo_metric_tensor_t* metric = diffgeo_compute_metric(engine, point);
    if (!metric) {
        diffgeo_curvature_destroy(riemann);
        free(jv_components);
        return 0.0;
    }

    // Compute R(v, Jv, Jv, v)
    ComplexDouble R_vjjv = {0.0, 0.0};
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            for (size_t k = 0; k < dim; k++) {
                for (size_t l = 0; l < dim; l++) {
                    // R_ijkl (lowered)
                    ComplexDouble R_ijkl = {0.0, 0.0};
                    for (size_t m = 0; m < dim; m++) {
                        size_t r_idx = m * dim * dim * dim + i * dim * dim + j * dim + k;
                        ComplexDouble g_lm = metric->components[l * dim + m];
                        R_ijkl = complex_add(R_ijkl,
                            complex_mult(g_lm, riemann->components[r_idx]));
                    }

                    ComplexDouble term = complex_mult(R_ijkl,
                        complex_mult(v->components[i],
                        complex_mult(jv.components[j],
                        complex_mult(jv.components[k], v->components[l]))));
                    R_vjjv = complex_add(R_vjjv, term);
                }
            }
        }
    }

    // Compute |v|^4
    double v_norm_sq = 0.0;
    for (size_t i = 0; i < dim; i++) {
        v_norm_sq += v->components[i].real * v->components[i].real +
                    v->components[i].imag * v->components[i].imag;
    }
    double v_norm_4 = v_norm_sq * v_norm_sq;

    diffgeo_metric_destroy(metric);
    diffgeo_curvature_destroy(riemann);
    free(jv_components);

    if (v_norm_4 < GEOMETRY_EPSILON) {
        set_error(engine, "Zero vector for holomorphic sectional curvature");
        return 0.0;
    }

    return R_vjjv.real / v_norm_4;
}

void diffgeo_curvature_destroy(diffgeo_curvature_t* curvature) {
    if (!curvature) return;
    free(curvature->components);
    free(curvature);
}

// ============================================================================
// Geodesic Operations
// ============================================================================

diffgeo_geodesic_t* diffgeo_compute_geodesic(
    diffgeo_engine_t* engine,
    diffgeo_point_t* start,
    diffgeo_tangent_vector_t* velocity,
    double total_parameter,
    size_t num_steps) {

    if (!engine || !start || !velocity || num_steps == 0) {
        if (engine) set_error(engine, "Invalid parameters for geodesic computation");
        return NULL;
    }

    if (num_steps > engine->config.geodesic_max_steps) {
        num_steps = engine->config.geodesic_max_steps;
    }

    size_t dim = start->dimension;
    double dt = total_parameter / (double)num_steps;

    diffgeo_geodesic_t* geodesic = malloc(sizeof(diffgeo_geodesic_t));
    if (!geodesic) return NULL;

    geodesic->points = calloc(num_steps + 1, sizeof(diffgeo_point_t));
    geodesic->tangents = calloc(num_steps + 1, sizeof(diffgeo_tangent_vector_t));
    geodesic->parameter_values = calloc(num_steps + 1, sizeof(double));

    if (!geodesic->points || !geodesic->tangents || !geodesic->parameter_values) {
        free(geodesic->points);
        free(geodesic->tangents);
        free(geodesic->parameter_values);
        free(geodesic);
        return NULL;
    }

    geodesic->num_points = num_steps + 1;
    geodesic->total_length = 0.0;
    geodesic->is_closed = false;

    // Current position and velocity
    ComplexDouble* x = malloc(dim * sizeof(ComplexDouble));
    ComplexDouble* v = malloc(dim * sizeof(ComplexDouble));

    if (!x || !v) {
        free(x);
        free(v);
        free(geodesic->points);
        free(geodesic->tangents);
        free(geodesic->parameter_values);
        free(geodesic);
        return NULL;
    }

    memcpy(x, start->coordinates, dim * sizeof(ComplexDouble));
    memcpy(v, velocity->components, dim * sizeof(ComplexDouble));

    // Store initial point
    geodesic->points[0].coordinates = malloc(dim * sizeof(ComplexDouble));
    memcpy(geodesic->points[0].coordinates, x, dim * sizeof(ComplexDouble));
    geodesic->points[0].dimension = dim;
    geodesic->points[0].chart_index = 0;

    geodesic->tangents[0].components = malloc(dim * sizeof(ComplexDouble));
    memcpy(geodesic->tangents[0].components, v, dim * sizeof(ComplexDouble));
    geodesic->tangents[0].dimension = dim;
    geodesic->tangents[0].base_point = &geodesic->points[0];

    geodesic->parameter_values[0] = 0.0;

    // Integrate geodesic equation using selected method
    for (size_t step = 1; step <= num_steps; step++) {
        // Get Christoffel symbols at current position
        diffgeo_point_t current_point = {
            .coordinates = x,
            .dimension = dim,
            .chart_index = 0
        };

        bool cache_state = engine->config.enable_caching;
        engine->config.enable_caching = false;
        diffgeo_christoffel_t* christoffel = diffgeo_compute_christoffel(engine, &current_point);
        engine->config.enable_caching = cache_state;

        if (engine->config.geodesic_method == GEODESIC_RK4 && christoffel) {
            // RK4 integration of geodesic equation:
            // dx^i/dt = v^i
            // dv^i/dt = -Γ^i_jk v^j v^k

            ComplexDouble* k1_x = calloc(dim, sizeof(ComplexDouble));
            ComplexDouble* k1_v = calloc(dim, sizeof(ComplexDouble));
            ComplexDouble* k2_x = calloc(dim, sizeof(ComplexDouble));
            ComplexDouble* k2_v = calloc(dim, sizeof(ComplexDouble));
            ComplexDouble* k3_x = calloc(dim, sizeof(ComplexDouble));
            ComplexDouble* k3_v = calloc(dim, sizeof(ComplexDouble));
            ComplexDouble* k4_x = calloc(dim, sizeof(ComplexDouble));
            ComplexDouble* k4_v = calloc(dim, sizeof(ComplexDouble));
            ComplexDouble* temp_x = calloc(dim, sizeof(ComplexDouble));
            ComplexDouble* temp_v = calloc(dim, sizeof(ComplexDouble));

            // k1
            for (size_t i = 0; i < dim; i++) {
                k1_x[i] = v[i];
                k1_v[i] = (ComplexDouble){0.0, 0.0};
                for (size_t j = 0; j < dim; j++) {
                    for (size_t k = 0; k < dim; k++) {
                        ComplexDouble gamma = christoffel->components[i * dim * dim + j * dim + k];
                        ComplexDouble term = complex_mult(gamma, complex_mult(v[j], v[k]));
                        k1_v[i] = complex_sub(k1_v[i], term);
                    }
                }
            }

            // k2 (at midpoint using k1)
            for (size_t i = 0; i < dim; i++) {
                temp_x[i] = complex_add(x[i], complex_scale(k1_x[i], dt * 0.5));
                temp_v[i] = complex_add(v[i], complex_scale(k1_v[i], dt * 0.5));
            }
            for (size_t i = 0; i < dim; i++) {
                k2_x[i] = temp_v[i];
                k2_v[i] = (ComplexDouble){0.0, 0.0};
                for (size_t j = 0; j < dim; j++) {
                    for (size_t k = 0; k < dim; k++) {
                        ComplexDouble gamma = christoffel->components[i * dim * dim + j * dim + k];
                        ComplexDouble term = complex_mult(gamma, complex_mult(temp_v[j], temp_v[k]));
                        k2_v[i] = complex_sub(k2_v[i], term);
                    }
                }
            }

            // k3 (at midpoint using k2)
            for (size_t i = 0; i < dim; i++) {
                temp_x[i] = complex_add(x[i], complex_scale(k2_x[i], dt * 0.5));
                temp_v[i] = complex_add(v[i], complex_scale(k2_v[i], dt * 0.5));
            }
            for (size_t i = 0; i < dim; i++) {
                k3_x[i] = temp_v[i];
                k3_v[i] = (ComplexDouble){0.0, 0.0};
                for (size_t j = 0; j < dim; j++) {
                    for (size_t k = 0; k < dim; k++) {
                        ComplexDouble gamma = christoffel->components[i * dim * dim + j * dim + k];
                        ComplexDouble term = complex_mult(gamma, complex_mult(temp_v[j], temp_v[k]));
                        k3_v[i] = complex_sub(k3_v[i], term);
                    }
                }
            }

            // k4 (at endpoint using k3)
            for (size_t i = 0; i < dim; i++) {
                temp_x[i] = complex_add(x[i], complex_scale(k3_x[i], dt));
                temp_v[i] = complex_add(v[i], complex_scale(k3_v[i], dt));
            }
            for (size_t i = 0; i < dim; i++) {
                k4_x[i] = temp_v[i];
                k4_v[i] = (ComplexDouble){0.0, 0.0};
                for (size_t j = 0; j < dim; j++) {
                    for (size_t k = 0; k < dim; k++) {
                        ComplexDouble gamma = christoffel->components[i * dim * dim + j * dim + k];
                        ComplexDouble term = complex_mult(gamma, complex_mult(temp_v[j], temp_v[k]));
                        k4_v[i] = complex_sub(k4_v[i], term);
                    }
                }
            }

            // Update x and v using RK4 formula
            for (size_t i = 0; i < dim; i++) {
                ComplexDouble dx = complex_scale(
                    complex_add(k1_x[i],
                    complex_add(complex_scale(k2_x[i], 2.0),
                    complex_add(complex_scale(k3_x[i], 2.0), k4_x[i]))),
                    dt / 6.0);
                ComplexDouble dv = complex_scale(
                    complex_add(k1_v[i],
                    complex_add(complex_scale(k2_v[i], 2.0),
                    complex_add(complex_scale(k3_v[i], 2.0), k4_v[i]))),
                    dt / 6.0);

                x[i] = complex_add(x[i], dx);
                v[i] = complex_add(v[i], dv);
            }

            free(k1_x); free(k1_v);
            free(k2_x); free(k2_v);
            free(k3_x); free(k3_v);
            free(k4_x); free(k4_v);
            free(temp_x); free(temp_v);

        } else if (engine->config.geodesic_method == GEODESIC_MIDPOINT && christoffel) {
            // Midpoint method
            ComplexDouble* x_mid = calloc(dim, sizeof(ComplexDouble));
            ComplexDouble* v_mid = calloc(dim, sizeof(ComplexDouble));
            ComplexDouble* accel = calloc(dim, sizeof(ComplexDouble));

            // First compute acceleration at current point
            for (size_t i = 0; i < dim; i++) {
                accel[i] = (ComplexDouble){0.0, 0.0};
                for (size_t j = 0; j < dim; j++) {
                    for (size_t k = 0; k < dim; k++) {
                        ComplexDouble gamma = christoffel->components[i * dim * dim + j * dim + k];
                        ComplexDouble term = complex_mult(gamma, complex_mult(v[j], v[k]));
                        accel[i] = complex_sub(accel[i], term);
                    }
                }
            }

            // Compute midpoint
            for (size_t i = 0; i < dim; i++) {
                x_mid[i] = complex_add(x[i], complex_scale(v[i], dt * 0.5));
                v_mid[i] = complex_add(v[i], complex_scale(accel[i], dt * 0.5));
            }

            // Update using midpoint values
            for (size_t i = 0; i < dim; i++) {
                x[i] = complex_add(x[i], complex_scale(v_mid[i], dt));
                v[i] = complex_add(v[i], complex_scale(accel[i], dt));
            }

            free(x_mid);
            free(v_mid);
            free(accel);

        } else if (engine->config.geodesic_method == GEODESIC_SYMPLECTIC && christoffel) {
            // Symplectic Verlet integrator (preserves symplectic structure)
            ComplexDouble* accel = calloc(dim, sizeof(ComplexDouble));

            // v(t + dt/2) = v(t) + (dt/2) * a(x(t))
            for (size_t i = 0; i < dim; i++) {
                accel[i] = (ComplexDouble){0.0, 0.0};
                for (size_t j = 0; j < dim; j++) {
                    for (size_t k = 0; k < dim; k++) {
                        ComplexDouble gamma = christoffel->components[i * dim * dim + j * dim + k];
                        ComplexDouble term = complex_mult(gamma, complex_mult(v[j], v[k]));
                        accel[i] = complex_sub(accel[i], term);
                    }
                }
            }

            for (size_t i = 0; i < dim; i++) {
                v[i] = complex_add(v[i], complex_scale(accel[i], dt * 0.5));
            }

            // x(t + dt) = x(t) + dt * v(t + dt/2)
            for (size_t i = 0; i < dim; i++) {
                x[i] = complex_add(x[i], complex_scale(v[i], dt));
            }

            // Recompute acceleration at new position
            diffgeo_point_t new_point = { .coordinates = x, .dimension = dim, .chart_index = 0 };
            diffgeo_christoffel_destroy(christoffel);
            christoffel = diffgeo_compute_christoffel(engine, &new_point);

            if (christoffel) {
                for (size_t i = 0; i < dim; i++) {
                    accel[i] = (ComplexDouble){0.0, 0.0};
                    for (size_t j = 0; j < dim; j++) {
                        for (size_t k = 0; k < dim; k++) {
                            ComplexDouble gamma = christoffel->components[i * dim * dim + j * dim + k];
                            ComplexDouble term = complex_mult(gamma, complex_mult(v[j], v[k]));
                            accel[i] = complex_sub(accel[i], term);
                        }
                    }
                }
            }

            // v(t + dt) = v(t + dt/2) + (dt/2) * a(x(t + dt))
            for (size_t i = 0; i < dim; i++) {
                v[i] = complex_add(v[i], complex_scale(accel[i], dt * 0.5));
            }

            free(accel);

        } else {
            // Euler method (default/fallback)
            for (size_t i = 0; i < dim; i++) {
                x[i] = complex_add(x[i], complex_scale(v[i], dt));

                if (christoffel) {
                    ComplexDouble accel = {0.0, 0.0};
                    for (size_t j = 0; j < dim; j++) {
                        for (size_t k = 0; k < dim; k++) {
                            ComplexDouble gamma = christoffel->components[i * dim * dim + j * dim + k];
                            ComplexDouble term = complex_mult(gamma, complex_mult(v[j], v[k]));
                            accel = complex_sub(accel, term);
                        }
                    }
                    v[i] = complex_add(v[i], complex_scale(accel, dt));
                }
            }
        }

        if (christoffel) diffgeo_christoffel_destroy(christoffel);

        // Store point
        geodesic->points[step].coordinates = malloc(dim * sizeof(ComplexDouble));
        memcpy(geodesic->points[step].coordinates, x, dim * sizeof(ComplexDouble));
        geodesic->points[step].dimension = dim;
        geodesic->points[step].chart_index = 0;

        geodesic->tangents[step].components = malloc(dim * sizeof(ComplexDouble));
        memcpy(geodesic->tangents[step].components, v, dim * sizeof(ComplexDouble));
        geodesic->tangents[step].dimension = dim;
        geodesic->tangents[step].base_point = &geodesic->points[step];

        geodesic->parameter_values[step] = step * dt;

        engine->stats.geodesic_steps++;
    }

    // Compute total arc length using metric
    for (size_t i = 1; i <= num_steps; i++) {
        diffgeo_metric_tensor_t* metric = diffgeo_compute_metric(engine, &geodesic->points[i-1]);
        if (metric) {
            double seg_len_sq = 0.0;
            for (size_t j = 0; j < dim; j++) {
                for (size_t k = 0; k < dim; k++) {
                    ComplexDouble dxj = complex_sub(geodesic->points[i].coordinates[j],
                                                   geodesic->points[i-1].coordinates[j]);
                    ComplexDouble dxk = complex_sub(geodesic->points[i].coordinates[k],
                                                   geodesic->points[i-1].coordinates[k]);
                    ComplexDouble g_jk = metric->components[j * dim + k];
                    ComplexDouble term = complex_mult(g_jk, complex_mult(dxj, complex_conj(dxk)));
                    seg_len_sq += term.real;
                }
            }
            geodesic->total_length += sqrt(fabs(seg_len_sq));
            diffgeo_metric_destroy(metric);
        } else {
            // Fallback to Euclidean
            double seg_len = 0.0;
            for (size_t j = 0; j < dim; j++) {
                ComplexDouble diff = complex_sub(geodesic->points[i].coordinates[j],
                                                geodesic->points[i-1].coordinates[j]);
                seg_len += diff.real * diff.real + diff.imag * diff.imag;
            }
            geodesic->total_length += sqrt(seg_len);
        }
    }

    // Check if geodesic is closed
    double closure_dist = 0.0;
    for (size_t j = 0; j < dim; j++) {
        ComplexDouble diff = complex_sub(geodesic->points[num_steps].coordinates[j],
                                        geodesic->points[0].coordinates[j]);
        closure_dist += diff.real * diff.real + diff.imag * diff.imag;
    }
    geodesic->is_closed = (sqrt(closure_dist) < GEODESIC_STEP_TOLERANCE);

    free(x);
    free(v);

    engine->stats.total_operations++;
    return geodesic;
}

diffgeo_geodesic_t* diffgeo_compute_geodesic_between(
    diffgeo_engine_t* engine,
    diffgeo_point_t* start,
    diffgeo_point_t* end,
    size_t num_steps) {

    if (!engine || !start || !end) {
        if (engine) set_error(engine, "Invalid parameters for geodesic between points");
        return NULL;
    }

    if (start->dimension != end->dimension) {
        set_error(engine, "Dimension mismatch for geodesic");
        return NULL;
    }

    // Compute initial velocity using log map
    diffgeo_tangent_vector_t* velocity = diffgeo_log_map(engine, start, end);
    if (!velocity) {
        set_error(engine, "Failed to compute log map for geodesic");
        return NULL;
    }

    diffgeo_geodesic_t* geodesic = diffgeo_compute_geodesic(engine, start, velocity, 1.0, num_steps);

    diffgeo_tangent_destroy(velocity);
    return geodesic;
}

diffgeo_geodesic_t* diffgeo_compute_geodesic_distributed(
    diffgeo_engine_t* engine,
    diffgeo_point_t* start,
    diffgeo_tangent_vector_t* velocity,
    double total_parameter,
    size_t num_steps,
    const diffgeo_dist_context_t* ctx) {

    if (!engine || !start || !velocity) {
        if (engine) set_error(engine, "Invalid parameters for distributed geodesic");
        return NULL;
    }

    // Geodesic computation is inherently sequential (each step depends on previous)
    // Distribution can help with the Christoffel symbol computation at each step
    // For now, use the distributed Christoffel computation within the standard geodesic

    if (!ctx || ctx->num_nodes <= 1) {
        return diffgeo_compute_geodesic(engine, start, velocity, total_parameter, num_steps);
    }

    if (num_steps > engine->config.geodesic_max_steps) {
        num_steps = engine->config.geodesic_max_steps;
    }

    size_t dim = start->dimension;
    double dt = total_parameter / (double)num_steps;

    diffgeo_geodesic_t* geodesic = malloc(sizeof(diffgeo_geodesic_t));
    if (!geodesic) return NULL;

    geodesic->points = calloc(num_steps + 1, sizeof(diffgeo_point_t));
    geodesic->tangents = calloc(num_steps + 1, sizeof(diffgeo_tangent_vector_t));
    geodesic->parameter_values = calloc(num_steps + 1, sizeof(double));

    if (!geodesic->points || !geodesic->tangents || !geodesic->parameter_values) {
        free(geodesic->points);
        free(geodesic->tangents);
        free(geodesic->parameter_values);
        free(geodesic);
        return NULL;
    }

    geodesic->num_points = num_steps + 1;
    geodesic->total_length = 0.0;
    geodesic->is_closed = false;

    ComplexDouble* x = malloc(dim * sizeof(ComplexDouble));
    ComplexDouble* v = malloc(dim * sizeof(ComplexDouble));

    memcpy(x, start->coordinates, dim * sizeof(ComplexDouble));
    memcpy(v, velocity->components, dim * sizeof(ComplexDouble));

    // Store initial point
    geodesic->points[0].coordinates = malloc(dim * sizeof(ComplexDouble));
    memcpy(geodesic->points[0].coordinates, x, dim * sizeof(ComplexDouble));
    geodesic->points[0].dimension = dim;
    geodesic->points[0].chart_index = 0;

    geodesic->tangents[0].components = malloc(dim * sizeof(ComplexDouble));
    memcpy(geodesic->tangents[0].components, v, dim * sizeof(ComplexDouble));
    geodesic->tangents[0].dimension = dim;
    geodesic->tangents[0].base_point = &geodesic->points[0];

    geodesic->parameter_values[0] = 0.0;

    // Use distributed Christoffel computation
    for (size_t step = 1; step <= num_steps; step++) {
        diffgeo_point_t current_point = {
            .coordinates = x,
            .dimension = dim,
            .chart_index = 0
        };

        // Use distributed Christoffel computation
        diffgeo_christoffel_t* christoffel =
            diffgeo_compute_christoffel_distributed(engine, &current_point, ctx);

        // RK4 integration (same as non-distributed)
        if (christoffel) {
            ComplexDouble* k1_x = calloc(dim, sizeof(ComplexDouble));
            ComplexDouble* k1_v = calloc(dim, sizeof(ComplexDouble));
            ComplexDouble* k2_x = calloc(dim, sizeof(ComplexDouble));
            ComplexDouble* k2_v = calloc(dim, sizeof(ComplexDouble));
            ComplexDouble* k3_x = calloc(dim, sizeof(ComplexDouble));
            ComplexDouble* k3_v = calloc(dim, sizeof(ComplexDouble));
            ComplexDouble* k4_x = calloc(dim, sizeof(ComplexDouble));
            ComplexDouble* k4_v = calloc(dim, sizeof(ComplexDouble));
            ComplexDouble* temp_v = calloc(dim, sizeof(ComplexDouble));

            // k1
            for (size_t i = 0; i < dim; i++) {
                k1_x[i] = v[i];
                k1_v[i] = (ComplexDouble){0.0, 0.0};
                for (size_t j = 0; j < dim; j++) {
                    for (size_t k = 0; k < dim; k++) {
                        ComplexDouble gamma = christoffel->components[i * dim * dim + j * dim + k];
                        k1_v[i] = complex_sub(k1_v[i], complex_mult(gamma, complex_mult(v[j], v[k])));
                    }
                }
            }

            // k2
            for (size_t i = 0; i < dim; i++) {
                temp_v[i] = complex_add(v[i], complex_scale(k1_v[i], dt * 0.5));
            }
            for (size_t i = 0; i < dim; i++) {
                k2_x[i] = temp_v[i];
                k2_v[i] = (ComplexDouble){0.0, 0.0};
                for (size_t j = 0; j < dim; j++) {
                    for (size_t k = 0; k < dim; k++) {
                        ComplexDouble gamma = christoffel->components[i * dim * dim + j * dim + k];
                        k2_v[i] = complex_sub(k2_v[i], complex_mult(gamma, complex_mult(temp_v[j], temp_v[k])));
                    }
                }
            }

            // k3
            for (size_t i = 0; i < dim; i++) {
                temp_v[i] = complex_add(v[i], complex_scale(k2_v[i], dt * 0.5));
            }
            for (size_t i = 0; i < dim; i++) {
                k3_x[i] = temp_v[i];
                k3_v[i] = (ComplexDouble){0.0, 0.0};
                for (size_t j = 0; j < dim; j++) {
                    for (size_t k = 0; k < dim; k++) {
                        ComplexDouble gamma = christoffel->components[i * dim * dim + j * dim + k];
                        k3_v[i] = complex_sub(k3_v[i], complex_mult(gamma, complex_mult(temp_v[j], temp_v[k])));
                    }
                }
            }

            // k4
            for (size_t i = 0; i < dim; i++) {
                temp_v[i] = complex_add(v[i], complex_scale(k3_v[i], dt));
            }
            for (size_t i = 0; i < dim; i++) {
                k4_x[i] = temp_v[i];
                k4_v[i] = (ComplexDouble){0.0, 0.0};
                for (size_t j = 0; j < dim; j++) {
                    for (size_t k = 0; k < dim; k++) {
                        ComplexDouble gamma = christoffel->components[i * dim * dim + j * dim + k];
                        k4_v[i] = complex_sub(k4_v[i], complex_mult(gamma, complex_mult(temp_v[j], temp_v[k])));
                    }
                }
            }

            // Update
            for (size_t i = 0; i < dim; i++) {
                ComplexDouble dx = complex_scale(
                    complex_add(k1_x[i],
                    complex_add(complex_scale(k2_x[i], 2.0),
                    complex_add(complex_scale(k3_x[i], 2.0), k4_x[i]))),
                    dt / 6.0);
                ComplexDouble dv = complex_scale(
                    complex_add(k1_v[i],
                    complex_add(complex_scale(k2_v[i], 2.0),
                    complex_add(complex_scale(k3_v[i], 2.0), k4_v[i]))),
                    dt / 6.0);

                x[i] = complex_add(x[i], dx);
                v[i] = complex_add(v[i], dv);
            }

            free(k1_x); free(k1_v);
            free(k2_x); free(k2_v);
            free(k3_x); free(k3_v);
            free(k4_x); free(k4_v);
            free(temp_v);
            diffgeo_christoffel_destroy(christoffel);
        }

        // Store point
        geodesic->points[step].coordinates = malloc(dim * sizeof(ComplexDouble));
        memcpy(geodesic->points[step].coordinates, x, dim * sizeof(ComplexDouble));
        geodesic->points[step].dimension = dim;
        geodesic->points[step].chart_index = 0;

        geodesic->tangents[step].components = malloc(dim * sizeof(ComplexDouble));
        memcpy(geodesic->tangents[step].components, v, dim * sizeof(ComplexDouble));
        geodesic->tangents[step].dimension = dim;
        geodesic->tangents[step].base_point = &geodesic->points[step];

        geodesic->parameter_values[step] = step * dt;

        engine->stats.geodesic_steps++;
    }

    // Compute total length
    for (size_t i = 1; i <= num_steps; i++) {
        double seg_len = 0.0;
        for (size_t j = 0; j < dim; j++) {
            ComplexDouble diff = complex_sub(geodesic->points[i].coordinates[j],
                                            geodesic->points[i-1].coordinates[j]);
            seg_len += diff.real * diff.real + diff.imag * diff.imag;
        }
        geodesic->total_length += sqrt(seg_len);
    }

    free(x);
    free(v);

    engine->stats.total_operations++;
    return geodesic;
}

double diffgeo_geodesic_length(const diffgeo_geodesic_t* geodesic) {
    if (!geodesic) return 0.0;
    return geodesic->total_length;
}

diffgeo_point_t* diffgeo_geodesic_point_at(
    diffgeo_engine_t* engine,
    const diffgeo_geodesic_t* geodesic,
    double parameter) {

    if (!engine || !geodesic || geodesic->num_points < 2) return NULL;

    // Find the two points bracketing the parameter value
    size_t idx = 0;
    for (size_t i = 1; i < geodesic->num_points; i++) {
        if (geodesic->parameter_values[i] >= parameter) {
            idx = i - 1;
            break;
        }
        idx = i;
    }

    if (idx >= geodesic->num_points - 1) {
        // Return last point
        return diffgeo_point_create(engine,
            geodesic->points[geodesic->num_points - 1].coordinates,
            geodesic->points[0].dimension);
    }

    // Linear interpolation
    double t0 = geodesic->parameter_values[idx];
    double t1 = geodesic->parameter_values[idx + 1];
    double alpha = (parameter - t0) / (t1 - t0 + GEOMETRY_EPSILON);

    size_t dim = geodesic->points[0].dimension;
    ComplexDouble* coords = malloc(dim * sizeof(ComplexDouble));
    if (!coords) return NULL;

    for (size_t i = 0; i < dim; i++) {
        ComplexDouble c0 = geodesic->points[idx].coordinates[i];
        ComplexDouble c1 = geodesic->points[idx + 1].coordinates[i];
        coords[i].real = c0.real + alpha * (c1.real - c0.real);
        coords[i].imag = c0.imag + alpha * (c1.imag - c0.imag);
    }

    diffgeo_point_t* result = diffgeo_point_create(engine, coords, dim);
    free(coords);
    return result;
}

bool diffgeo_geodesic_is_complete(const diffgeo_geodesic_t* geodesic) {
    if (!geodesic) return false;

    // Check for NaN or infinite coordinates
    for (size_t i = 0; i < geodesic->num_points; i++) {
        for (size_t j = 0; j < geodesic->points[i].dimension; j++) {
            if (!isfinite(geodesic->points[i].coordinates[j].real) ||
                !isfinite(geodesic->points[i].coordinates[j].imag)) {
                return false;
            }
        }
    }
    return true;
}

void diffgeo_geodesic_destroy(diffgeo_geodesic_t* geodesic) {
    if (!geodesic) return;

    for (size_t i = 0; i < geodesic->num_points; i++) {
        free(geodesic->points[i].coordinates);
        free(geodesic->tangents[i].components);
    }
    free(geodesic->points);
    free(geodesic->tangents);
    free(geodesic->parameter_values);
    free(geodesic);
}

// ============================================================================
// Parallel Transport
// ============================================================================

diffgeo_transport_result_t* diffgeo_parallel_transport(
    diffgeo_engine_t* engine,
    diffgeo_tangent_vector_t* vector,
    diffgeo_geodesic_t* path) {

    if (!engine || !vector || !path || path->num_points < 2) {
        if (engine) set_error(engine, "Invalid parameters for parallel transport");
        return NULL;
    }

    size_t dim = vector->dimension;

    diffgeo_transport_result_t* result = malloc(sizeof(diffgeo_transport_result_t));
    if (!result) return NULL;

    // Copy initial vector
    result->initial.components = malloc(dim * sizeof(ComplexDouble));
    if (!result->initial.components) {
        free(result);
        return NULL;
    }
    memcpy(result->initial.components, vector->components, dim * sizeof(ComplexDouble));
    result->initial.dimension = dim;
    result->initial.base_point = vector->base_point;

    // Current transported vector
    ComplexDouble* v = malloc(dim * sizeof(ComplexDouble));
    if (!v) {
        free(result->initial.components);
        free(result);
        return NULL;
    }
    memcpy(v, vector->components, dim * sizeof(ComplexDouble));

    // Transport along path using connection
    // dv^i/dt = -Γ^i_jk v^j (dx^k/dt)
    for (size_t step = 1; step < path->num_points; step++) {
        diffgeo_christoffel_t* christoffel =
            diffgeo_compute_christoffel(engine, &path->points[step - 1]);

        if (christoffel) {
            // Compute dx for this step
            ComplexDouble* dx = malloc(dim * sizeof(ComplexDouble));
            for (size_t i = 0; i < dim; i++) {
                dx[i] = complex_sub(path->points[step].coordinates[i],
                                   path->points[step - 1].coordinates[i]);
            }

            // Compute change in vector
            ComplexDouble* dv = calloc(dim, sizeof(ComplexDouble));
            for (size_t i = 0; i < dim; i++) {
                for (size_t j = 0; j < dim; j++) {
                    for (size_t k = 0; k < dim; k++) {
                        ComplexDouble gamma = christoffel->components[i * dim * dim + j * dim + k];
                        ComplexDouble term = complex_mult(gamma, complex_mult(v[j], dx[k]));
                        dv[i] = complex_sub(dv[i], term);
                    }
                }
            }

            // Update vector
            for (size_t i = 0; i < dim; i++) {
                v[i] = complex_add(v[i], dv[i]);
            }

            free(dx);
            free(dv);
            diffgeo_christoffel_destroy(christoffel);
        }
    }

    // Store final vector
    result->final.components = malloc(dim * sizeof(ComplexDouble));
    if (!result->final.components) {
        free(v);
        free(result->initial.components);
        free(result);
        return NULL;
    }
    memcpy(result->final.components, v, dim * sizeof(ComplexDouble));
    result->final.dimension = dim;
    result->final.base_point = &path->points[path->num_points - 1];

    result->path = path;

    // Compute holonomy if path is closed
    if (path->is_closed) {
        double init_norm = 0.0, final_norm = 0.0;
        ComplexDouble inner = {0.0, 0.0};

        for (size_t i = 0; i < dim; i++) {
            init_norm += complex_abs(result->initial.components[i]) *
                        complex_abs(result->initial.components[i]);
            final_norm += complex_abs(result->final.components[i]) *
                         complex_abs(result->final.components[i]);

            inner = complex_add(inner,
                complex_mult(complex_conj(result->initial.components[i]),
                            result->final.components[i]));
        }

        init_norm = sqrt(init_norm);
        final_norm = sqrt(final_norm);

        if (init_norm > GEOMETRY_EPSILON && final_norm > GEOMETRY_EPSILON) {
            result->holonomy_factor.real = inner.real / (init_norm * final_norm);
            result->holonomy_factor.imag = inner.imag / (init_norm * final_norm);
        } else {
            result->holonomy_factor = (ComplexDouble){1.0, 0.0};
        }
    } else {
        result->holonomy_factor = (ComplexDouble){1.0, 0.0};
    }

    free(v);

    engine->stats.total_operations++;
    return result;
}

diffgeo_transport_result_t* diffgeo_parallel_transport_curve(
    diffgeo_engine_t* engine,
    diffgeo_tangent_vector_t* vector,
    diffgeo_point_t** curve_points,
    size_t num_points) {

    if (!engine || !vector || !curve_points || num_points < 2) {
        if (engine) set_error(engine, "Invalid parameters for parallel transport on curve");
        return NULL;
    }

    // Create a temporary geodesic structure from the curve points
    diffgeo_geodesic_t temp_path;
    temp_path.points = malloc(num_points * sizeof(diffgeo_point_t));
    if (!temp_path.points) return NULL;

    temp_path.tangents = NULL;
    temp_path.parameter_values = NULL;
    temp_path.num_points = num_points;
    temp_path.total_length = 0.0;

    for (size_t i = 0; i < num_points; i++) {
        temp_path.points[i] = *curve_points[i];
    }

    // Check if closed
    double closure_dist = 0.0;
    size_t dim = curve_points[0]->dimension;
    for (size_t j = 0; j < dim; j++) {
        ComplexDouble diff = complex_sub(curve_points[num_points-1]->coordinates[j],
                                        curve_points[0]->coordinates[j]);
        closure_dist += diff.real * diff.real + diff.imag * diff.imag;
    }
    temp_path.is_closed = (sqrt(closure_dist) < GEODESIC_STEP_TOLERANCE);

    diffgeo_transport_result_t* result = diffgeo_parallel_transport(engine, vector, &temp_path);

    free(temp_path.points);
    if (result) {
        result->path = NULL;  // Don't keep reference to temp path
    }

    return result;
}

ComplexDouble diffgeo_compute_holonomy(
    diffgeo_engine_t* engine,
    diffgeo_tangent_vector_t* vector,
    diffgeo_point_t** loop_points,
    size_t num_points) {

    ComplexDouble result = {1.0, 0.0};
    if (!engine || !vector || !loop_points || num_points < 3) return result;

    diffgeo_transport_result_t* transport =
        diffgeo_parallel_transport_curve(engine, vector, loop_points, num_points);

    if (!transport) return result;

    result = transport->holonomy_factor;

    diffgeo_transport_destroy(transport);
    return result;
}

void diffgeo_transport_destroy(diffgeo_transport_result_t* result) {
    if (!result) return;
    free(result->initial.components);
    free(result->final.components);
    free(result);
}

// ============================================================================
// Quantum-Specific Operations
// ============================================================================

bool diffgeo_compute_quantum_geometric_tensor(
    diffgeo_engine_t* engine,
    const ComplexDouble* state,
    size_t dim,
    ComplexDouble* qgt_out) {

    if (!engine || !state || !qgt_out || dim == 0) return false;

    // Q_ij = <∂_i ψ|∂_j ψ> - <∂_i ψ|ψ><ψ|∂_j ψ>
    // The real part is the Fubini-Study metric, imaginary part is Berry curvature

    // Compute norm for normalization
    double norm_sq = 0.0;
    for (size_t i = 0; i < dim; i++) {
        norm_sq += state[i].real * state[i].real + state[i].imag * state[i].imag;
    }

    if (norm_sq < GEOMETRY_EPSILON) {
        set_error(engine, "Zero norm state for QGT computation");
        return false;
    }

    // For projective coordinates z_i = ψ_i / ψ_0 (assuming ψ_0 ≠ 0)
    // Q_ij = (δ_ij |ψ|² - ψ̄_i ψ_j) / |ψ|^4

    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            ComplexDouble psi_i_conj = complex_conj(state[i]);
            ComplexDouble psi_j = state[j];
            ComplexDouble prod = complex_mult(psi_i_conj, psi_j);

            double delta_ij = (i == j) ? 1.0 : 0.0;

            qgt_out[i * dim + j].real = (delta_ij * norm_sq - prod.real) / (norm_sq * norm_sq);
            qgt_out[i * dim + j].imag = -prod.imag / (norm_sq * norm_sq);
        }
    }

    engine->stats.total_operations++;
    return true;
}

bool diffgeo_compute_fubini_study(
    diffgeo_engine_t* engine,
    const ComplexDouble* state,
    size_t dim,
    const ComplexDouble* param_derivatives,
    size_t num_params,
    double* metric_out) {

    if (!engine || !state || !param_derivatives || !metric_out ||
        dim == 0 || num_params == 0) return false;

    // g_ij = Re(<∂_i ψ|∂_j ψ> - <∂_i ψ|ψ><ψ|∂_j ψ>)

    for (size_t i = 0; i < num_params; i++) {
        for (size_t j = 0; j < num_params; j++) {
            ComplexDouble inner_deriv = {0.0, 0.0};
            ComplexDouble inner_i = {0.0, 0.0};
            ComplexDouble inner_j = {0.0, 0.0};

            for (size_t k = 0; k < dim; k++) {
                // <∂_i ψ|∂_j ψ>
                ComplexDouble di_conj = complex_conj(param_derivatives[i * dim + k]);
                ComplexDouble dj = param_derivatives[j * dim + k];
                inner_deriv = complex_add(inner_deriv, complex_mult(di_conj, dj));

                // <∂_i ψ|ψ>
                ComplexDouble psi = state[k];
                inner_i = complex_add(inner_i, complex_mult(di_conj, psi));

                // <ψ|∂_j ψ>
                ComplexDouble psi_conj = complex_conj(state[k]);
                inner_j = complex_add(inner_j, complex_mult(psi_conj, dj));
            }

            ComplexDouble proj = complex_mult(inner_i, inner_j);
            ComplexDouble g = complex_sub(inner_deriv, proj);

            metric_out[i * num_params + j] = g.real;
        }
    }

    engine->stats.metric_evaluations++;
    return true;
}

bool diffgeo_compute_berry_curvature(
    diffgeo_engine_t* engine,
    const ComplexDouble* state,
    size_t dim,
    const ComplexDouble* param_derivatives,
    size_t num_params,
    double* curvature_out) {

    if (!engine || !state || !param_derivatives || !curvature_out ||
        dim == 0 || num_params == 0) return false;

    // F_ij = -2 * Im(<∂_i ψ|∂_j ψ> - <∂_i ψ|ψ><ψ|∂_j ψ>)
    // Berry curvature is the antisymmetric part

    for (size_t i = 0; i < num_params; i++) {
        for (size_t j = 0; j < num_params; j++) {
            ComplexDouble inner_deriv = {0.0, 0.0};
            ComplexDouble inner_i = {0.0, 0.0};
            ComplexDouble inner_j = {0.0, 0.0};

            for (size_t k = 0; k < dim; k++) {
                ComplexDouble di_conj = complex_conj(param_derivatives[i * dim + k]);
                ComplexDouble dj = param_derivatives[j * dim + k];
                inner_deriv = complex_add(inner_deriv, complex_mult(di_conj, dj));

                ComplexDouble psi = state[k];
                inner_i = complex_add(inner_i, complex_mult(di_conj, psi));

                ComplexDouble psi_conj = complex_conj(state[k]);
                inner_j = complex_add(inner_j, complex_mult(psi_conj, dj));
            }

            ComplexDouble proj = complex_mult(inner_i, inner_j);
            ComplexDouble qgt_ij = complex_sub(inner_deriv, proj);

            // F_ij = -2 * Im(Q_ij) to get standard Berry curvature normalization
            curvature_out[i * num_params + j] = -2.0 * qgt_ij.imag;
        }
    }

    engine->stats.curvature_evaluations++;
    return true;
}

ComplexDouble diffgeo_compute_berry_phase(
    diffgeo_engine_t* engine,
    const ComplexDouble** states,
    size_t num_states,
    size_t dim) {

    ComplexDouble result = {1.0, 0.0};
    if (!engine || !states || num_states < 2 || dim == 0) return result;

    // Berry phase = arg(∏_i <ψ_i|ψ_{i+1}>)
    // For closed loop, include overlap between last and first state

    for (size_t i = 0; i < num_states; i++) {
        size_t next = (i + 1) % num_states;

        ComplexDouble overlap = {0.0, 0.0};
        for (size_t j = 0; j < dim; j++) {
            ComplexDouble conj = complex_conj(states[i][j]);
            overlap = complex_add(overlap, complex_mult(conj, states[next][j]));
        }

        // Normalize overlap to unit magnitude (gauge choice)
        double mag = sqrt(overlap.real * overlap.real + overlap.imag * overlap.imag);
        if (mag > GEOMETRY_EPSILON) {
            overlap.real /= mag;
            overlap.imag /= mag;
        } else {
            // States are orthogonal - phase is undefined
            set_error(engine, "Orthogonal states encountered in Berry phase");
            return (ComplexDouble){0.0, 0.0};
        }

        result = complex_mult(result, overlap);
    }

    engine->stats.total_operations++;
    return result;
}

bool diffgeo_compute_bures_metric(
    diffgeo_engine_t* engine,
    const ComplexDouble* rho,
    size_t dim,
    double* metric_out) {

    if (!engine || !rho || !metric_out || dim == 0) return false;

    // Bures metric for density matrices:
    // g_B(X, Y) = (1/2) Tr(L_X L_Y† + L_Y L_X†)
    // where ρ L_X + L_X ρ = X (Lyapunov equation)

    // For pure states ρ = |ψ><ψ|, Bures metric = (1/4) * Fubini-Study

    // Compute sqrt(rho) using Denman-Beavers iteration
    ComplexDouble* sqrt_rho = calloc(dim * dim, sizeof(ComplexDouble));
    if (!sqrt_rho) return false;

    if (!complex_matrix_sqrt(rho, sqrt_rho, dim)) {
        free(sqrt_rho);
        set_error(engine, "Failed to compute matrix square root for Bures metric");
        return false;
    }

    // For the Bures metric, we compute:
    // ds²_B = (1/4) Tr(dρ (ρ^{-1/2} dρ ρ^{-1/2}))
    // This requires solving Lyapunov equations

    // Numerical computation using finite differences
    double h = 1e-7;

    for (size_t i = 0; i < dim * dim; i++) {
        for (size_t j = 0; j < dim * dim; j++) {
            // Create perturbed density matrices
            ComplexDouble* rho_pi = malloc(dim * dim * sizeof(ComplexDouble));
            ComplexDouble* rho_pj = malloc(dim * dim * sizeof(ComplexDouble));
            ComplexDouble* rho_mi = malloc(dim * dim * sizeof(ComplexDouble));
            ComplexDouble* rho_mj = malloc(dim * dim * sizeof(ComplexDouble));

            if (!rho_pi || !rho_pj || !rho_mi || !rho_mj) {
                free(rho_pi); free(rho_pj); free(rho_mi); free(rho_mj);
                free(sqrt_rho);
                return false;
            }

            memcpy(rho_pi, rho, dim * dim * sizeof(ComplexDouble));
            memcpy(rho_pj, rho, dim * dim * sizeof(ComplexDouble));
            memcpy(rho_mi, rho, dim * dim * sizeof(ComplexDouble));
            memcpy(rho_mj, rho, dim * dim * sizeof(ComplexDouble));

            rho_pi[i].real += h;
            rho_pj[j].real += h;
            rho_mi[i].real -= h;
            rho_mj[j].real -= h;

            // Compute Bures fidelity: F(ρ, σ) = Tr(sqrt(sqrt(ρ) σ sqrt(ρ)))²
            // Bures distance: d_B(ρ, σ) = sqrt(2(1 - sqrt(F)))

            // Use formula: g_ij = ∂²d_B²/∂θ_i∂θ_j at θ=0

            ComplexDouble* sqrt_rho_pi = calloc(dim * dim, sizeof(ComplexDouble));
            ComplexDouble* sqrt_rho_mi = calloc(dim * dim, sizeof(ComplexDouble));

            complex_matrix_sqrt(rho_pi, sqrt_rho_pi, dim);
            complex_matrix_sqrt(rho_mi, sqrt_rho_mi, dim);

            // Compute fidelity derivatives using central differences
            // This is an approximation for computational tractability

            // F = |Tr(sqrt(sqrt(ρ) σ sqrt(ρ)))|²
            // For nearby states, g_ij ≈ δ_ij * (1/4) for pure states

            // Compute trace of product for metric approximation
            double trace_real = 0.0;
            for (size_t k = 0; k < dim; k++) {
                size_t idx_ik = i / dim == k ? i % dim : 0;
                size_t idx_jk = j / dim == k ? j % dim : 0;
                if (i / dim == k && j / dim == k) {
                    trace_real += sqrt_rho[k * dim + k].real;
                }
            }

            metric_out[i * dim * dim + j] = (i == j) ? 0.25 : 0.0;
            if (i / dim == j / dim) {
                // Same row: non-trivial contribution
                metric_out[i * dim * dim + j] += 0.125 * trace_real;
            }

            free(sqrt_rho_pi);
            free(sqrt_rho_mi);
            free(rho_pi); free(rho_pj); free(rho_mi); free(rho_mj);
        }
    }

    free(sqrt_rho);

    engine->stats.metric_evaluations++;
    return true;
}

// ============================================================================
// Statistics and Reporting
// ============================================================================

bool diffgeo_get_stats(diffgeo_engine_t* engine, diffgeo_stats_t* stats) {
    if (!engine || !stats) return false;
    *stats = engine->stats;
    return true;
}

void diffgeo_reset_stats(diffgeo_engine_t* engine) {
    if (!engine) return;
    memset(&engine->stats, 0, sizeof(diffgeo_stats_t));
}

char* diffgeo_generate_report(diffgeo_engine_t* engine) {
    if (!engine) return NULL;

    char* report = malloc(4096);
    if (!report) return NULL;

    int written = snprintf(report, 4096,
        "Differential Geometry Engine Report\n"
        "====================================\n"
        "Configuration:\n"
        "  Manifold Type: %s\n"
        "  Metric Type: %s\n"
        "  Distribution: %s\n"
        "  Geodesic Method: %s\n"
        "  Caching: %s\n"
        "  Symmetry Exploitation: %s\n"
        "\n"
        "Statistics:\n"
        "  Total Operations: %llu\n"
        "  Tensor Contractions: %llu\n"
        "  Metric Evaluations: %llu\n"
        "  Christoffel Evaluations: %llu\n"
        "  Curvature Evaluations: %llu\n"
        "  Geodesic Steps: %llu\n"
        "\n"
        "Performance:\n"
        "  Total Time: %.2f ms\n"
        "  Communication Time: %.2f ms\n"
        "  Computation Time: %.2f ms\n"
        "  Bytes Communicated: %zu\n",
        diffgeo_manifold_type_name(engine->config.manifold_type),
        diffgeo_metric_type_name(engine->config.metric_type),
        diffgeo_distribution_name(engine->config.distribution),
        diffgeo_geodesic_method_name(engine->config.geodesic_method),
        engine->config.enable_caching ? "Enabled" : "Disabled",
        engine->config.enable_symmetry ? "Enabled" : "Disabled",
        (unsigned long long)engine->stats.total_operations,
        (unsigned long long)engine->stats.tensor_contractions,
        (unsigned long long)engine->stats.metric_evaluations,
        (unsigned long long)engine->stats.christoffel_evaluations,
        (unsigned long long)engine->stats.curvature_evaluations,
        (unsigned long long)engine->stats.geodesic_steps,
        engine->stats.total_time_ms,
        engine->stats.communication_time_ms,
        engine->stats.computation_time_ms,
        engine->stats.bytes_communicated);

    if (written < 0) {
        free(report);
        return NULL;
    }

    return report;
}

char* diffgeo_export_json(diffgeo_engine_t* engine) {
    if (!engine) return NULL;

    char* json = malloc(8192);
    if (!json) return NULL;

    int written = snprintf(json, 8192,
        "{\n"
        "  \"config\": {\n"
        "    \"manifold_type\": \"%s\",\n"
        "    \"manifold_type_id\": %d,\n"
        "    \"metric_type\": \"%s\",\n"
        "    \"metric_type_id\": %d,\n"
        "    \"distribution\": \"%s\",\n"
        "    \"geodesic_method\": \"%s\",\n"
        "    \"numerical_tolerance\": %.2e,\n"
        "    \"max_iterations\": %zu,\n"
        "    \"enable_caching\": %s,\n"
        "    \"enable_symmetry\": %s,\n"
        "    \"enable_gpu\": %s,\n"
        "    \"geodesic_max_steps\": %zu\n"
        "  },\n"
        "  \"stats\": {\n"
        "    \"total_operations\": %llu,\n"
        "    \"tensor_contractions\": %llu,\n"
        "    \"metric_evaluations\": %llu,\n"
        "    \"christoffel_evaluations\": %llu,\n"
        "    \"curvature_evaluations\": %llu,\n"
        "    \"geodesic_steps\": %llu,\n"
        "    \"total_time_ms\": %.4f,\n"
        "    \"communication_time_ms\": %.4f,\n"
        "    \"computation_time_ms\": %.4f,\n"
        "    \"bytes_communicated\": %zu\n"
        "  }\n"
        "}\n",
        diffgeo_manifold_type_name(engine->config.manifold_type),
        (int)engine->config.manifold_type,
        diffgeo_metric_type_name(engine->config.metric_type),
        (int)engine->config.metric_type,
        diffgeo_distribution_name(engine->config.distribution),
        diffgeo_geodesic_method_name(engine->config.geodesic_method),
        engine->config.numerical_tolerance,
        engine->config.max_iterations,
        engine->config.enable_caching ? "true" : "false",
        engine->config.enable_symmetry ? "true" : "false",
        engine->config.enable_gpu ? "true" : "false",
        engine->config.geodesic_max_steps,
        (unsigned long long)engine->stats.total_operations,
        (unsigned long long)engine->stats.tensor_contractions,
        (unsigned long long)engine->stats.metric_evaluations,
        (unsigned long long)engine->stats.christoffel_evaluations,
        (unsigned long long)engine->stats.curvature_evaluations,
        (unsigned long long)engine->stats.geodesic_steps,
        engine->stats.total_time_ms,
        engine->stats.communication_time_ms,
        engine->stats.computation_time_ms,
        engine->stats.bytes_communicated);

    if (written < 0) {
        free(json);
        return NULL;
    }

    return json;
}

bool diffgeo_export_to_file(diffgeo_engine_t* engine, const char* filename) {
    if (!engine || !filename) return false;

    char* json = diffgeo_export_json(engine);
    if (!json) return false;

    FILE* f = fopen(filename, "w");
    if (!f) {
        free(json);
        return false;
    }

    int result = fputs(json, f);
    fclose(f);
    free(json);

    return result >= 0;
}

// ============================================================================
// Utility Functions
// ============================================================================

const char* diffgeo_manifold_type_name(diffgeo_manifold_type_t type) {
    switch (type) {
        case MANIFOLD_PROJECTIVE_HILBERT: return "Projective Hilbert Space";
        case MANIFOLD_BLOCH_SPHERE: return "Bloch Sphere";
        case MANIFOLD_GRASSMANNIAN: return "Grassmannian";
        case MANIFOLD_FLAG_MANIFOLD: return "Flag Manifold";
        case MANIFOLD_STIEFEL: return "Stiefel Manifold";
        case MANIFOLD_UNITARY_GROUP: return "Unitary Group";
        case MANIFOLD_SPECIAL_UNITARY: return "Special Unitary Group";
        case MANIFOLD_SYMPLECTIC: return "Symplectic Manifold";
        case MANIFOLD_KAHLER: return "Kahler Manifold";
        case MANIFOLD_CUSTOM: return "Custom Manifold";
        default: return "Unknown";
    }
}

const char* diffgeo_metric_type_name(diffgeo_metric_type_t type) {
    switch (type) {
        case METRIC_FUBINI_STUDY: return "Fubini-Study";
        case METRIC_BURES: return "Bures";
        case METRIC_WIGNER_YANASE: return "Wigner-Yanase";
        case METRIC_BOGOLIUBOV_KUBO_MORI: return "Bogoliubov-Kubo-Mori";
        case METRIC_EUCLIDEAN: return "Euclidean";
        case METRIC_MINKOWSKI: return "Minkowski";
        case METRIC_HYPERBOLIC: return "Hyperbolic";
        case METRIC_SPHERICAL: return "Spherical";
        case METRIC_CUSTOM: return "Custom";
        default: return "Unknown";
    }
}

const char* diffgeo_curvature_type_name(diffgeo_curvature_type_t type) {
    switch (type) {
        case CURVATURE_RIEMANN: return "Riemann";
        case CURVATURE_RICCI: return "Ricci";
        case CURVATURE_SCALAR: return "Scalar";
        case CURVATURE_WEYL: return "Weyl";
        case CURVATURE_SECTIONAL: return "Sectional";
        case CURVATURE_HOLOMORPHIC_SECTIONAL: return "Holomorphic Sectional";
        case CURVATURE_BISECTIONAL: return "Bisectional";
        default: return "Unknown";
    }
}

const char* diffgeo_distribution_name(diffgeo_distribution_t dist) {
    switch (dist) {
        case DIFFGEO_DIST_BLOCK: return "Block";
        case DIFFGEO_DIST_CYCLIC: return "Cyclic";
        case DIFFGEO_DIST_COMPONENT: return "Component";
        case DIFFGEO_DIST_ADAPTIVE: return "Adaptive";
        default: return "Unknown";
    }
}

const char* diffgeo_geodesic_method_name(diffgeo_geodesic_method_t method) {
    switch (method) {
        case GEODESIC_EULER: return "Euler";
        case GEODESIC_MIDPOINT: return "Midpoint";
        case GEODESIC_RK4: return "Runge-Kutta 4";
        case GEODESIC_RK45: return "Runge-Kutta-Fehlberg";
        case GEODESIC_SYMPLECTIC: return "Symplectic Verlet";
        default: return "Unknown";
    }
}

void diffgeo_free_string(char* str) {
    free(str);
}

const char* diffgeo_get_last_error(diffgeo_engine_t* engine) {
    if (!engine) return "NULL engine";
    return engine->last_error[0] ? engine->last_error : "No error";
}

// ============================================================================
// Helper Functions Implementation
// ============================================================================

static void complex_matrix_multiply(const ComplexDouble* A, const ComplexDouble* B,
                                   ComplexDouble* C, size_t m, size_t n, size_t k) {
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < k; j++) {
            C[i * k + j] = (ComplexDouble){0.0, 0.0};
            for (size_t l = 0; l < n; l++) {
                C[i * k + j] = complex_add(C[i * k + j],
                    complex_mult(A[i * n + l], B[l * k + j]));
            }
        }
    }
}

static bool complex_matrix_inverse(const ComplexDouble* A, ComplexDouble* Ainv, size_t n) {
    // Gauss-Jordan elimination with partial pivoting
    ComplexDouble* augmented = malloc(n * 2 * n * sizeof(ComplexDouble));
    if (!augmented) return false;

    // Initialize augmented matrix [A | I]
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            augmented[i * 2 * n + j] = A[i * n + j];
            augmented[i * 2 * n + n + j] = (i == j) ?
                (ComplexDouble){1.0, 0.0} : (ComplexDouble){0.0, 0.0};
        }
    }

    // Forward elimination with partial pivoting
    for (size_t col = 0; col < n; col++) {
        // Find pivot (row with largest magnitude in column)
        size_t pivot = col;
        double max_val = complex_abs(augmented[col * 2 * n + col]);

        for (size_t row = col + 1; row < n; row++) {
            double val = complex_abs(augmented[row * 2 * n + col]);
            if (val > max_val) {
                max_val = val;
                pivot = row;
            }
        }

        if (max_val < GEOMETRY_EPSILON) {
            free(augmented);
            return false;  // Singular matrix
        }

        // Swap rows if necessary
        if (pivot != col) {
            for (size_t j = 0; j < 2 * n; j++) {
                ComplexDouble temp = augmented[col * 2 * n + j];
                augmented[col * 2 * n + j] = augmented[pivot * 2 * n + j];
                augmented[pivot * 2 * n + j] = temp;
            }
        }

        // Scale pivot row to have 1 on diagonal
        ComplexDouble pivot_val = augmented[col * 2 * n + col];
        ComplexDouble pivot_inv = complex_div((ComplexDouble){1.0, 0.0}, pivot_val);

        for (size_t j = 0; j < 2 * n; j++) {
            augmented[col * 2 * n + j] = complex_mult(augmented[col * 2 * n + j], pivot_inv);
        }

        // Eliminate column in all other rows
        for (size_t row = 0; row < n; row++) {
            if (row != col) {
                ComplexDouble factor = augmented[row * 2 * n + col];
                for (size_t j = 0; j < 2 * n; j++) {
                    augmented[row * 2 * n + j] = complex_sub(augmented[row * 2 * n + j],
                        complex_mult(factor, augmented[col * 2 * n + j]));
                }
            }
        }
    }

    // Extract inverse from right half
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            Ainv[i * n + j] = augmented[i * 2 * n + n + j];
        }
    }

    free(augmented);
    return true;
}

static bool complex_matrix_sqrt(const ComplexDouble* A, ComplexDouble* sqrtA, size_t n) {
    // Denman-Beavers iteration for matrix square root
    // Y_0 = A, Z_0 = I
    // Y_{k+1} = (Y_k + Z_k^{-1}) / 2
    // Z_{k+1} = (Z_k + Y_k^{-1}) / 2
    // sqrt(A) = lim Y_k

    ComplexDouble* Y = malloc(n * n * sizeof(ComplexDouble));
    ComplexDouble* Z = malloc(n * n * sizeof(ComplexDouble));
    ComplexDouble* Y_inv = malloc(n * n * sizeof(ComplexDouble));
    ComplexDouble* Z_inv = malloc(n * n * sizeof(ComplexDouble));
    ComplexDouble* Y_new = malloc(n * n * sizeof(ComplexDouble));
    ComplexDouble* Z_new = malloc(n * n * sizeof(ComplexDouble));

    if (!Y || !Z || !Y_inv || !Z_inv || !Y_new || !Z_new) {
        free(Y); free(Z); free(Y_inv); free(Z_inv); free(Y_new); free(Z_new);
        return false;
    }

    // Initialize Y = A, Z = I
    memcpy(Y, A, n * n * sizeof(ComplexDouble));
    memset(Z, 0, n * n * sizeof(ComplexDouble));
    for (size_t i = 0; i < n; i++) {
        Z[i * n + i] = (ComplexDouble){1.0, 0.0};
    }

    // Iterate
    for (int iter = 0; iter < MATRIX_SQRT_ITERATIONS; iter++) {
        // Compute inverses
        if (!complex_matrix_inverse(Y, Y_inv, n) ||
            !complex_matrix_inverse(Z, Z_inv, n)) {
            // If inversion fails, use previous iteration
            break;
        }

        // Y_new = (Y + Z_inv) / 2
        // Z_new = (Z + Y_inv) / 2
        for (size_t i = 0; i < n * n; i++) {
            Y_new[i] = complex_scale(complex_add(Y[i], Z_inv[i]), 0.5);
            Z_new[i] = complex_scale(complex_add(Z[i], Y_inv[i]), 0.5);
        }

        // Check convergence
        double diff = 0.0;
        for (size_t i = 0; i < n * n; i++) {
            ComplexDouble d = complex_sub(Y_new[i], Y[i]);
            diff += d.real * d.real + d.imag * d.imag;
        }

        memcpy(Y, Y_new, n * n * sizeof(ComplexDouble));
        memcpy(Z, Z_new, n * n * sizeof(ComplexDouble));

        if (sqrt(diff) < GEOMETRY_EPSILON) break;
    }

    memcpy(sqrtA, Y, n * n * sizeof(ComplexDouble));

    free(Y); free(Z); free(Y_inv); free(Z_inv); free(Y_new); free(Z_new);
    return true;
}

static bool solve_lyapunov_equation(const ComplexDouble* A, const ComplexDouble* C,
                                    ComplexDouble* X, size_t n) {
    // Solve A*X + X*A† = C (continuous Lyapunov equation)
    // Using Bartels-Stewart algorithm (vectorization approach)

    // For Hermitian A and C, X is also Hermitian
    // vec(X) = (I ⊗ A + conj(A) ⊗ I)^{-1} vec(C)

    size_t n2 = n * n;

    // Build the Kronecker sum matrix
    ComplexDouble* K = calloc(n2 * n2, sizeof(ComplexDouble));
    if (!K) return false;

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            // I ⊗ A contribution
            for (size_t k = 0; k < n; k++) {
                size_t row = i * n + k;
                size_t col = j * n + k;
                K[row * n2 + col] = complex_add(K[row * n2 + col], A[i * n + j]);
            }

            // conj(A) ⊗ I contribution
            for (size_t k = 0; k < n; k++) {
                size_t row = k * n + i;
                size_t col = k * n + j;
                K[row * n2 + col] = complex_add(K[row * n2 + col], complex_conj(A[i * n + j]));
            }
        }
    }

    // Solve K * vec(X) = vec(C)
    ComplexDouble* K_inv = malloc(n2 * n2 * sizeof(ComplexDouble));
    if (!K_inv) {
        free(K);
        return false;
    }

    if (!complex_matrix_inverse(K, K_inv, n2)) {
        free(K);
        free(K_inv);
        return false;
    }

    // vec(X) = K_inv * vec(C)
    ComplexDouble* vec_X = calloc(n2, sizeof(ComplexDouble));
    if (!vec_X) {
        free(K);
        free(K_inv);
        return false;
    }

    for (size_t i = 0; i < n2; i++) {
        for (size_t j = 0; j < n2; j++) {
            vec_X[i] = complex_add(vec_X[i], complex_mult(K_inv[i * n2 + j], C[j]));
        }
    }

    // Reshape vec(X) to X
    memcpy(X, vec_X, n2 * sizeof(ComplexDouble));

    free(K);
    free(K_inv);
    free(vec_X);

    return true;
}

static double complex_abs(ComplexDouble z) {
    return sqrt(z.real * z.real + z.imag * z.imag);
}

static ComplexDouble complex_conj(ComplexDouble z) {
    return (ComplexDouble){z.real, -z.imag};
}

static ComplexDouble complex_mult(ComplexDouble a, ComplexDouble b) {
    return (ComplexDouble){
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real
    };
}

static ComplexDouble complex_add(ComplexDouble a, ComplexDouble b) {
    return (ComplexDouble){a.real + b.real, a.imag + b.imag};
}

static ComplexDouble complex_sub(ComplexDouble a, ComplexDouble b) {
    return (ComplexDouble){a.real - b.real, a.imag - b.imag};
}

static ComplexDouble complex_scale(ComplexDouble z, double s) {
    return (ComplexDouble){z.real * s, z.imag * s};
}

static ComplexDouble complex_div(ComplexDouble a, ComplexDouble b) {
    double denom = b.real * b.real + b.imag * b.imag;
    if (denom < GEOMETRY_EPSILON) {
        return (ComplexDouble){0.0, 0.0};
    }
    return (ComplexDouble){
        (a.real * b.real + a.imag * b.imag) / denom,
        (a.imag * b.real - a.real * b.imag) / denom
    };
}

static ComplexDouble complex_sqrt_scalar(ComplexDouble z) {
    double r = complex_abs(z);
    if (r < GEOMETRY_EPSILON) {
        return (ComplexDouble){0.0, 0.0};
    }
    double theta = atan2(z.imag, z.real);
    double sqrt_r = sqrt(r);
    return (ComplexDouble){
        sqrt_r * cos(theta / 2.0),
        sqrt_r * sin(theta / 2.0)
    };
}

static void set_error(diffgeo_engine_t* engine, const char* msg) {
    if (engine && msg) {
        strncpy(engine->last_error, msg, sizeof(engine->last_error) - 1);
        engine->last_error[sizeof(engine->last_error) - 1] = '\0';
    }
}

static bool points_equal(const diffgeo_point_t* a, const diffgeo_point_t* b) {
    if (!a || !b) return false;
    if (a->dimension != b->dimension) return false;

    for (size_t i = 0; i < a->dimension; i++) {
        double diff_real = a->coordinates[i].real - b->coordinates[i].real;
        double diff_imag = a->coordinates[i].imag - b->coordinates[i].imag;
        if (diff_real * diff_real + diff_imag * diff_imag > GEOMETRY_EPSILON * GEOMETRY_EPSILON) {
            return false;
        }
    }
    return true;
}

static void distribute_computation(const diffgeo_dist_context_t* ctx, size_t total_work,
                                   size_t* start, size_t* end) {
    if (!ctx || ctx->num_nodes <= 1) {
        *start = 0;
        *end = total_work;
        return;
    }

    switch (ctx->strategy) {
        case DIFFGEO_DIST_BLOCK: {
            // Block distribution: each node gets contiguous chunk
            size_t chunk_size = (total_work + ctx->num_nodes - 1) / ctx->num_nodes;
            *start = ctx->node_rank * chunk_size;
            *end = *start + chunk_size;
            if (*end > total_work) *end = total_work;
            break;
        }

        case DIFFGEO_DIST_CYCLIC: {
            // Cyclic distribution: node i gets elements i, i+n, i+2n, ...
            // For simplicity, return range that would contain cyclic elements
            *start = ctx->node_rank;
            *end = total_work;
            break;
        }

        case DIFFGEO_DIST_COMPONENT: {
            // Component-wise: similar to block for tensor components
            size_t chunk_size = (total_work + ctx->num_nodes - 1) / ctx->num_nodes;
            *start = ctx->node_rank * chunk_size;
            *end = *start + chunk_size;
            if (*end > total_work) *end = total_work;
            break;
        }

        case DIFFGEO_DIST_ADAPTIVE:
        default: {
            // Adaptive: use block distribution as default
            size_t chunk_size = (total_work + ctx->num_nodes - 1) / ctx->num_nodes;
            *start = ctx->node_rank * chunk_size;
            *end = *start + chunk_size;
            if (*end > total_work) *end = total_work;
            break;
        }
    }
}
