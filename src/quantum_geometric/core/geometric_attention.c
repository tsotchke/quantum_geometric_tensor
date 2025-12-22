/**
 * @file geometric_attention.c
 * @brief Production implementation of geometric attention mechanisms
 *
 * This module provides geometric attention computation using manifold geometry,
 * geodesic paths, and curvature-aware attention scoring.
 */

#include "quantum_geometric/core/geometric_attention.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

// Platform-specific SIMD includes
#if defined(__aarch64__) || defined(_M_ARM64)
    #if defined(__ARM_NEON) || defined(__ARM_NEON__)
        #include <arm_neon.h>
        #define USE_NEON 1
    #endif
#elif defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
    #define USE_AVX 1
#endif

// =============================================================================
// Constants
// =============================================================================

#define EPSILON 1e-10
#define DEFAULT_CACHE_CAPACITY 1024
#define MIN_CURVATURE_THRESHOLD 1e-8

// =============================================================================
// Internal Structure Definition
// =============================================================================

struct geometric_attention_t {
    attention_config_t config;
    Manifold* manifold;
    GeodesicPath** geodesics;
    size_t num_geodesics;
    GeometricCache* cache;
    attention_metrics_t metrics;
    bool is_initialized;

    // Working memory
    double* metric_buffer;
    double* christoffel_buffer;
    complex double* phase_buffer;
    size_t buffer_size;
};

// =============================================================================
// Cache Implementation
// =============================================================================

GeometricCache* create_geometric_cache(void) {
    GeometricCache* cache = calloc(1, sizeof(GeometricCache));
    if (!cache) return NULL;

    cache->cache_capacity = DEFAULT_CACHE_CAPACITY;
    cache->cached_metrics = calloc(cache->cache_capacity, sizeof(double));
    cache->cached_connections = calloc(cache->cache_capacity, sizeof(double));
    cache->cached_phases = calloc(cache->cache_capacity, sizeof(complex double));

    if (!cache->cached_metrics || !cache->cached_connections || !cache->cached_phases) {
        destroy_geometric_cache(cache);
        return NULL;
    }

    cache->cache_size = 0;
    cache->hit_count = 0;
    cache->miss_count = 0;
    cache->is_valid = true;

    return cache;
}

void destroy_geometric_cache(GeometricCache* cache) {
    if (!cache) return;
    free(cache->cached_metrics);
    free(cache->cached_connections);
    free(cache->cached_phases);
    free(cache);
}

bool check_geometric_cache(GeometricCache* cache,
                          const attention_state_t* state,
                          complex double* cached_result) {
    if (!cache || !cache->is_valid || cache->cache_size == 0) {
        if (cache) cache->miss_count++;
        return false;
    }

    // Simple hash-based cache lookup
    size_t hash = state->seq_length ^ (state->head_dim << 8) ^ (state->batch_size << 16);
    size_t idx = hash % cache->cache_capacity;

    if (idx < cache->cache_size && cached_result) {
        *cached_result = cache->cached_phases[idx];
        cache->hit_count++;
        return true;
    }

    cache->miss_count++;
    return false;
}

void update_geometric_cache(GeometricCache* cache,
                           const attention_state_t* state,
                           const complex double* result) {
    if (!cache || !result) return;

    size_t hash = state->seq_length ^ (state->head_dim << 8) ^ (state->batch_size << 16);
    size_t idx = hash % cache->cache_capacity;

    if (idx < cache->cache_capacity) {
        cache->cached_phases[idx] = *result;
        if (idx >= cache->cache_size) {
            cache->cache_size = idx + 1;
        }
    }
}

void restore_cached_patterns(geometric_attention_t* attention) {
    if (!attention || !attention->cache) return;
    // Restore cached metric and connection data if available
    if (attention->cache->is_valid && attention->cache->cache_size > 0) {
        if (attention->metric_buffer && attention->cache->cached_metrics) {
            size_t copy_size = attention->buffer_size < attention->cache->cache_size ?
                              attention->buffer_size : attention->cache->cache_size;
            memcpy(attention->metric_buffer, attention->cache->cached_metrics,
                   copy_size * sizeof(double));
        }
    }
}

// =============================================================================
// Manifold Implementation
// =============================================================================

Manifold* create_attention_manifold(size_t dimension, manifold_type_t type) {
    Manifold* manifold = calloc(1, sizeof(Manifold));
    if (!manifold) return NULL;

    manifold->type = type;
    manifold->dimension = dimension;

    // Allocate metric tensor (n x n symmetric matrix stored as upper triangular)
    size_t metric_size = dimension * dimension;
    manifold->metric = calloc(metric_size, sizeof(double));

    // Allocate Christoffel symbols (n x n x n tensor)
    size_t christoffel_size = dimension * dimension * dimension;
    manifold->christoffel = calloc(christoffel_size, sizeof(double));

    // Allocate Riemann tensor (n x n x n x n, but we store contracted form)
    size_t riemann_size = dimension * dimension;
    manifold->riemann_tensor = calloc(riemann_size, sizeof(double));

    if (!manifold->metric || !manifold->christoffel || !manifold->riemann_tensor) {
        destroy_manifold(manifold);
        return NULL;
    }

    // Initialize with flat metric (identity)
    for (size_t i = 0; i < dimension; i++) {
        manifold->metric[i * dimension + i] = 1.0;
    }

    manifold->curvature = 0.0;

    // Set type-specific curvature
    switch (type) {
        case MANIFOLD_HYPERBOLIC:
            manifold->curvature = -1.0;
            // Initialize hyperbolic metric
            for (size_t i = 0; i < dimension; i++) {
                manifold->metric[i * dimension + i] = exp(-2.0 * (double)i / dimension);
            }
            break;
        case MANIFOLD_SPHERICAL:
            manifold->curvature = 1.0;
            break;
        case MANIFOLD_EUCLIDEAN:
        case MANIFOLD_RIEMANNIAN:
        case MANIFOLD_KAHLER:
        default:
            manifold->curvature = 0.0;
            break;
    }

    return manifold;
}

void destroy_manifold(Manifold* manifold) {
    if (!manifold) return;
    free(manifold->metric);
    free(manifold->christoffel);
    free(manifold->riemann_tensor);
    free(manifold->extra_data);
    free(manifold);
}

ManifoldProperties analyze_manifold_properties(const Manifold* manifold) {
    ManifoldProperties props = {0};

    if (!manifold) return props;

    // Compute scalar curvature from Riemann tensor (trace of Ricci tensor)
    double scalar_curv = 0.0;
    if (manifold->riemann_tensor) {
        for (size_t i = 0; i < manifold->dimension; i++) {
            scalar_curv += manifold->riemann_tensor[i * manifold->dimension + i];
        }
    }

    props.scalar_curvature = scalar_curv;
    props.sectional_curvature = manifold->curvature;
    props.ricci_scalar = scalar_curv;

    // Estimate injectivity radius (for hyperbolic: pi / sqrt(-K))
    if (manifold->curvature < -EPSILON) {
        props.injectivity_radius = M_PI / sqrt(-manifold->curvature);
    } else if (manifold->curvature > EPSILON) {
        props.injectivity_radius = M_PI / sqrt(manifold->curvature);
    } else {
        props.injectivity_radius = INFINITY;
    }

    // Topology properties
    props.is_compact = (manifold->type == MANIFOLD_SPHERICAL);
    props.is_complete = true;
    props.is_negatively_curved = (manifold->curvature < -EPSILON);

    return props;
}

double compute_geodesic_distance(const Manifold* manifold,
                                const double* point1,
                                const double* point2) {
    if (!manifold || !point1 || !point2) return INFINITY;

    size_t n = manifold->dimension;
    double dist_sq = 0.0;

    // Compute Riemannian distance: sqrt(g_ij * dx^i * dx^j)
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            double dx_i = point2[i] - point1[i];
            double dx_j = point2[j] - point1[j];
            dist_sq += manifold->metric[i * n + j] * dx_i * dx_j;
        }
    }

    // For hyperbolic space, use hyperbolic distance formula
    if (manifold->type == MANIFOLD_HYPERBOLIC && manifold->curvature < -EPSILON) {
        double K = -manifold->curvature;
        // d = (1/sqrt(K)) * arccosh(1 + K * ||x-y||^2 / 2)
        double euclidean_dist_sq = 0.0;
        for (size_t i = 0; i < n; i++) {
            double dx = point2[i] - point1[i];
            euclidean_dist_sq += dx * dx;
        }
        double arg = 1.0 + K * euclidean_dist_sq / 2.0;
        if (arg >= 1.0) {
            return acosh(arg) / sqrt(K);
        }
    }

    return sqrt(fmax(dist_sq, 0.0));
}

// =============================================================================
// Optimization Functions
// =============================================================================

void optimize_hyperbolic_attention(geometric_attention_t* attention,
                                  ManifoldProperties props) {
    if (!attention || !attention->manifold) return;

    // For hyperbolic geometry, we adjust the metric to use Poincare disk model
    size_t n = attention->manifold->dimension;
    double K = fabs(props.sectional_curvature);
    if (K < MIN_CURVATURE_THRESHOLD) K = 1.0;

    // Conformal factor for Poincare disk: 4 / (1 - |x|^2)^2
    // We approximate with a simple scaling based on curvature
    double scale = 2.0 / sqrt(K);

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            attention->manifold->metric[i * n + j] *= scale;
        }
    }

    // Update Christoffel symbols for hyperbolic geometry
    // Gamma^i_jk = -K * (delta^i_j * x_k + delta^i_k * x_j)
    // Simplified: we use the negative curvature to adjust connection
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            for (size_t k = 0; k < n; k++) {
                size_t idx = i * n * n + j * n + k;
                if (i == j) {
                    attention->manifold->christoffel[idx] = -K * 0.5;
                }
            }
        }
    }
}

void optimize_riemannian_attention(geometric_attention_t* attention,
                                  ManifoldProperties props) {
    if (!attention || !attention->manifold) return;

    // For general Riemannian geometry, compute Christoffel symbols from metric
    size_t n = attention->manifold->dimension;
    double* g = attention->manifold->metric;
    double* gamma = attention->manifold->christoffel;

    // Compute inverse metric
    double* g_inv = calloc(n * n, sizeof(double));
    if (!g_inv) return;

    // Simple inverse for diagonal-dominant metric
    for (size_t i = 0; i < n; i++) {
        double diag = g[i * n + i];
        if (fabs(diag) > EPSILON) {
            g_inv[i * n + i] = 1.0 / diag;
        } else {
            g_inv[i * n + i] = 1.0;
        }
    }

    // Compute Christoffel symbols: Gamma^i_jk = (1/2) g^il (d_j g_lk + d_k g_jl - d_l g_jk)
    // Simplified for efficiency (assuming slowly varying metric)
    double h = 0.01;  // Step size for numerical derivatives

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            for (size_t k = 0; k < n; k++) {
                size_t idx = i * n * n + j * n + k;
                double sum = 0.0;

                for (size_t l = 0; l < n; l++) {
                    // Approximate derivative terms
                    double dg_jlk = (j + 1 < n) ? (g[(j+1)*n + l] - g[j*n + l]) / h : 0.0;
                    double dg_kjl = (k + 1 < n) ? (g[k*n + l] - g[(k > 0 ? k-1 : 0)*n + l]) / h : 0.0;
                    double dg_ljk = (l + 1 < n) ? (g[l*n + k] - g[(l > 0 ? l-1 : 0)*n + k]) / h : 0.0;

                    sum += 0.5 * g_inv[i * n + l] * (dg_jlk + dg_kjl - dg_ljk);
                }

                gamma[idx] = sum;
            }
        }
    }

    free(g_inv);
}

void optimize_euclidean_attention(geometric_attention_t* attention,
                                 ManifoldProperties props) {
    (void)props;  // Unused for Euclidean case

    if (!attention || !attention->manifold) return;

    // For Euclidean geometry, metric is identity and Christoffel symbols are zero
    size_t n = attention->manifold->dimension;

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            attention->manifold->metric[i * n + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Clear Christoffel symbols
    memset(attention->manifold->christoffel, 0,
           n * n * n * sizeof(double));
}

// =============================================================================
// Geodesic Path Operations
// =============================================================================

GeodesicPath* compute_geodesic(const Manifold* manifold,
                              const double* start,
                              const double* end,
                              size_t num_points) {
    if (!manifold || !start || !end || num_points < 2) return NULL;

    GeodesicPath* path = calloc(1, sizeof(GeodesicPath));
    if (!path) return NULL;

    size_t n = manifold->dimension;
    path->dimension = n;
    path->num_points = num_points;
    path->points = calloc(num_points * n, sizeof(double));
    path->tangent = calloc(num_points * n, sizeof(double));

    if (!path->points || !path->tangent) {
        destroy_geodesic_path(path);
        return NULL;
    }

    // For flat/Euclidean geometry, geodesic is a straight line
    // For curved manifolds, we use geodesic shooting with Runge-Kutta

    if (manifold->type == MANIFOLD_EUCLIDEAN || fabs(manifold->curvature) < MIN_CURVATURE_THRESHOLD) {
        // Linear interpolation
        for (size_t i = 0; i < num_points; i++) {
            double t = (double)i / (double)(num_points - 1);
            for (size_t j = 0; j < n; j++) {
                path->points[i * n + j] = (1.0 - t) * start[j] + t * end[j];
                path->tangent[i * n + j] = end[j] - start[j];
            }
        }
    } else {
        // Geodesic equation: d²x^i/dt² + Gamma^i_jk (dx^j/dt)(dx^k/dt) = 0
        // Use RK4 integration
        double dt = 1.0 / (double)(num_points - 1);
        double* x = calloc(n, sizeof(double));
        double* v = calloc(n, sizeof(double));

        if (!x || !v) {
            free(x);
            free(v);
            destroy_geodesic_path(path);
            return NULL;
        }

        // Initial conditions
        for (size_t j = 0; j < n; j++) {
            x[j] = start[j];
            v[j] = (end[j] - start[j]);  // Initial velocity guess
        }

        // Store first point
        memcpy(&path->points[0], x, n * sizeof(double));
        memcpy(&path->tangent[0], v, n * sizeof(double));

        // RK4 integration
        for (size_t i = 1; i < num_points; i++) {
            // Compute acceleration from geodesic equation
            for (size_t a = 0; a < n; a++) {
                double acc = 0.0;
                for (size_t j = 0; j < n; j++) {
                    for (size_t k = 0; k < n; k++) {
                        size_t idx = a * n * n + j * n + k;
                        acc -= manifold->christoffel[idx] * v[j] * v[k];
                    }
                }
                // Simple Euler step (could use RK4 for better accuracy)
                v[a] += acc * dt;
                x[a] += v[a] * dt;
            }

            memcpy(&path->points[i * n], x, n * sizeof(double));
            memcpy(&path->tangent[i * n], v, n * sizeof(double));
        }

        free(x);
        free(v);
    }

    // Compute path length
    path->length = 0.0;
    for (size_t i = 1; i < num_points; i++) {
        path->length += compute_geodesic_distance(manifold,
                                                  &path->points[(i-1) * n],
                                                  &path->points[i * n]);
    }

    // Compute path energy: E = (1/2) integral(g_ij v^i v^j dt)
    path->energy = 0.0;
    for (size_t i = 0; i < num_points; i++) {
        double kinetic = 0.0;
        for (size_t j = 0; j < n; j++) {
            for (size_t k = 0; k < n; k++) {
                kinetic += manifold->metric[j * n + k] *
                          path->tangent[i * n + j] *
                          path->tangent[i * n + k];
            }
        }
        path->energy += 0.5 * kinetic / num_points;
    }

    path->is_closed = false;

    return path;
}

void destroy_geodesic_path(GeodesicPath* path) {
    if (!path) return;
    free(path->points);
    free(path->tangent);
    free(path);
}

double* parallel_transport(const Manifold* manifold,
                          const GeodesicPath* path,
                          const double* vector) {
    if (!manifold || !path || !vector) return NULL;

    size_t n = path->dimension;
    double* result = calloc(n, sizeof(double));
    if (!result) return NULL;

    // Copy initial vector
    memcpy(result, vector, n * sizeof(double));

    // Parallel transport equation: dv^i/dt + Gamma^i_jk (dx^j/dt) v^k = 0
    for (size_t i = 1; i < path->num_points; i++) {
        double* v = result;
        const double* tangent = &path->tangent[(i-1) * n];

        // Compute derivative
        double dv[MAX_GEODESICS];
        memset(dv, 0, n * sizeof(double));

        for (size_t a = 0; a < n; a++) {
            for (size_t j = 0; j < n; j++) {
                for (size_t k = 0; k < n; k++) {
                    size_t idx = a * n * n + j * n + k;
                    dv[a] -= manifold->christoffel[idx] * tangent[j] * v[k];
                }
            }
        }

        // Update vector
        double dt = 1.0 / (double)(path->num_points - 1);
        for (size_t a = 0; a < n; a++) {
            v[a] += dv[a] * dt;
        }
    }

    return result;
}

// =============================================================================
// Core Attention Functions
// =============================================================================

geometric_attention_t* create_geometric_attention(const attention_config_t* config) {
    if (!config) return NULL;

    geometric_attention_t* attention = calloc(1, sizeof(geometric_attention_t));
    if (!attention) return NULL;

    attention->config = *config;

    // Create manifold based on geometry type
    manifold_type_t manifold_type = MANIFOLD_RIEMANNIAN;
    switch (config->geometry) {
        case ATTN_GEOMETRY_MANIFOLD:
            manifold_type = MANIFOLD_RIEMANNIAN;
            break;
        case ATTN_GEOMETRY_COMPLEX:
        case ATTN_GEOMETRY_FUBINI_STUDY:
            manifold_type = MANIFOLD_KAHLER;
            break;
        case ATTN_GEOMETRY_KAHLER:
            manifold_type = MANIFOLD_KAHLER;
            break;
        default:
            manifold_type = MANIFOLD_EUCLIDEAN;
    }

    attention->manifold = create_attention_manifold(config->head_dim, manifold_type);
    if (!attention->manifold) {
        free(attention);
        return NULL;
    }

    // Allocate geodesic storage
    attention->geodesics = calloc(MAX_GEODESICS, sizeof(GeodesicPath*));
    attention->num_geodesics = 0;

    // Create cache
    attention->cache = create_geometric_cache();

    // Allocate working buffers
    attention->buffer_size = config->head_dim * config->head_dim;
    attention->metric_buffer = calloc(attention->buffer_size, sizeof(double));
    attention->christoffel_buffer = calloc(attention->buffer_size * config->head_dim, sizeof(double));
    attention->phase_buffer = calloc(attention->buffer_size, sizeof(complex double));

    if (!attention->geodesics || !attention->cache ||
        !attention->metric_buffer || !attention->christoffel_buffer || !attention->phase_buffer) {
        destroy_geometric_attention(attention);
        return NULL;
    }

    // Initialize metrics
    memset(&attention->metrics, 0, sizeof(attention_metrics_t));
    attention->is_initialized = true;

    return attention;
}

void destroy_geometric_attention(geometric_attention_t* attention) {
    if (!attention) return;

    destroy_manifold(attention->manifold);

    if (attention->geodesics) {
        for (size_t i = 0; i < attention->num_geodesics; i++) {
            destroy_geodesic_path(attention->geodesics[i]);
        }
        free(attention->geodesics);
    }

    destroy_geometric_cache(attention->cache);

    free(attention->metric_buffer);
    free(attention->christoffel_buffer);
    free(attention->phase_buffer);

    free(attention);
}

bool attention_init_geometry(geometric_attention_t* attention,
                            const attn_geometric_params_t* params) {
    if (!attention || !params) return false;

    // Update manifold with parameters
    if (attention->manifold) {
        attention->manifold->curvature = params->curvature;

        // Copy metric tensor if provided
        if (attention->metric_buffer) {
            attention->metric_buffer[0] = params->metric_tensor;
        }
    }

    return true;
}

bool attention_init_state(geometric_attention_t* attention,
                         const attention_state_t* state) {
    if (!attention || !state) return false;

    // Validate state dimensions
    if (state->seq_length == 0 || state->head_dim == 0) return false;

    // Check if we need to resize manifold
    if (attention->manifold && attention->manifold->dimension != state->head_dim) {
        destroy_manifold(attention->manifold);
        attention->manifold = create_attention_manifold(state->head_dim,
                                                        attention->manifold->type);
        if (!attention->manifold) return false;
    }

    return true;
}

bool attention_validate_init(geometric_attention_t* attention) {
    if (!attention) return false;
    return attention->is_initialized &&
           attention->manifold != NULL &&
           attention->cache != NULL;
}

bool compute_attention(geometric_attention_t* attention,
                      const attention_state_t* input,
                      attention_state_t* output) {
    if (!attention || !input || !output) return false;
    if (!attention->is_initialized) return false;

    size_t seq_len = input->seq_length;
    size_t head_dim = input->head_dim;
    size_t batch_size = input->batch_size > 0 ? input->batch_size : 1;

    // Allocate output if needed
    if (!output->values) {
        output->values = calloc(seq_len * head_dim, sizeof(complex double));
        if (!output->values) return false;
    }
    output->seq_length = seq_len;
    output->head_dim = head_dim;
    output->batch_size = batch_size;

    // Compute attention scores using geometric distance
    double scale = 1.0 / sqrt((double)head_dim);

    for (size_t i = 0; i < seq_len; i++) {
        // Query point
        double* q_point = calloc(head_dim, sizeof(double));
        if (!q_point) return false;

        for (size_t d = 0; d < head_dim; d++) {
            q_point[d] = creal(input->queries[i * head_dim + d]);
        }

        // Compute attention weights using geodesic distance
        double* weights = calloc(seq_len, sizeof(double));
        double weight_sum = 0.0;

        if (!weights) {
            free(q_point);
            return false;
        }

        for (size_t j = 0; j < seq_len; j++) {
            // Key point
            double* k_point = calloc(head_dim, sizeof(double));
            if (!k_point) {
                free(q_point);
                free(weights);
                return false;
            }

            for (size_t d = 0; d < head_dim; d++) {
                k_point[d] = creal(input->keys[j * head_dim + d]);
            }

            // Compute geodesic distance as attention score
            double dist = compute_geodesic_distance(attention->manifold, q_point, k_point);

            // Convert distance to attention weight (exp(-dist^2 / 2))
            weights[j] = exp(-dist * dist * scale * 0.5);
            weight_sum += weights[j];

            free(k_point);
        }

        // Normalize weights (softmax)
        if (weight_sum > EPSILON) {
            for (size_t j = 0; j < seq_len; j++) {
                weights[j] /= weight_sum;
            }
        }

        // Apply attention weights to values
        for (size_t d = 0; d < head_dim; d++) {
            complex double sum = 0.0;
            for (size_t j = 0; j < seq_len; j++) {
                sum += weights[j] * input->values[j * head_dim + d];
            }
            output->values[i * head_dim + d] = sum;
        }

        free(q_point);
        free(weights);
    }

    // Update metrics
    attention->metrics.operation_count++;

    return true;
}

bool apply_geometric_phase(geometric_attention_t* attention,
                         phase_type_t phase_type,
                         attention_state_t* state) {
    if (!attention || !state || !state->values) return false;

    size_t n = state->seq_length * state->head_dim;

    // Compute phase factor based on type
    complex double phase_factor = 1.0;

    switch (phase_type) {
        case PHASE_BERRY:
            // Berry phase: exp(i * integral(A.dx))
            // Approximate using manifold curvature
            phase_factor = cexp(I * attention->manifold->curvature * M_PI / 4.0);
            break;
        case PHASE_GEOMETRIC:
            // Geometric phase from path integral
            phase_factor = cexp(I * M_PI / 6.0);
            break;
        case PHASE_DYNAMIC:
            // Dynamic phase from energy
            phase_factor = cexp(-I * attention->metrics.execution_time);
            break;
        case PHASE_TOPOLOGICAL:
            // Topological phase (quantized)
            phase_factor = cexp(I * M_PI);  // pi phase
            break;
        default:
            phase_factor = 1.0;
    }

    // Apply phase to state
    for (size_t i = 0; i < n; i++) {
        state->values[i] *= phase_factor;
    }

    return true;
}

bool compute_attention_weights(geometric_attention_t* attention,
                              const attention_state_t* state,
                              complex double* weights) {
    if (!attention || !state || !weights) return false;

    size_t seq_len = state->seq_length;
    size_t head_dim = state->head_dim;
    double scale = 1.0 / sqrt((double)head_dim);

    // Compute pairwise attention weights
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < seq_len; j++) {
            // Compute dot product
            complex double dot = 0.0;
            for (size_t d = 0; d < head_dim; d++) {
                dot += state->queries[i * head_dim + d] *
                       conj(state->keys[j * head_dim + d]);
            }
            weights[i * seq_len + j] = dot * scale;
        }
    }

    return true;
}

// =============================================================================
// Metric and Curvature Operations
// =============================================================================

bool attention_compute_metric(geometric_attention_t* attention,
                             const attn_geometric_params_t* params,
                             double* metric) {
    if (!attention || !metric) return false;

    if (attention->manifold && attention->manifold->metric) {
        size_t n = attention->manifold->dimension;
        memcpy(metric, attention->manifold->metric, n * n * sizeof(double));
        return true;
    }

    return false;
}

bool attention_compute_connection(geometric_attention_t* attention,
                                 const attn_geometric_params_t* params,
                                 double* connection) {
    if (!attention || !connection) return false;

    if (attention->manifold && attention->manifold->christoffel) {
        size_t n = attention->manifold->dimension;
        memcpy(connection, attention->manifold->christoffel, n * n * n * sizeof(double));
        return true;
    }

    return false;
}

bool attention_compute_curvature(geometric_attention_t* attention,
                                const attn_geometric_params_t* params,
                                double* curvature) {
    if (!attention || !curvature) return false;

    if (attention->manifold) {
        *curvature = attention->manifold->curvature;
        return true;
    }

    return false;
}

void compute_attention_curvature(geometric_attention_t* attention,
                                double* curvature_out) {
    if (!attention || !curvature_out || !attention->manifold) return;
    *curvature_out = attention->manifold->curvature;
}

void apply_geodesic_attention(geometric_attention_t* attention,
                             const attention_state_t* input,
                             attention_state_t* output) {
    if (!attention || !input || !output) return;
    compute_attention(attention, input, output);
}

// =============================================================================
// Phase Operations
// =============================================================================

bool compute_berry_phase(geometric_attention_t* attention,
                        const attention_state_t* state,
                        complex double* phase) {
    if (!attention || !state || !phase) return false;

    // Berry phase: gamma = i * integral(<psi|d|psi>)
    // Approximate using discrete sum
    complex double sum = 0.0;
    size_t n = state->seq_length * state->head_dim;

    for (size_t i = 0; i < n - 1; i++) {
        complex double overlap = conj(state->values[i]) * state->values[i + 1];
        sum += clog(overlap);
    }

    *phase = cexp(I * cimag(sum));
    return true;
}

bool compute_geometric_phase(geometric_attention_t* attention,
                           const attention_state_t* state,
                           complex double* phase) {
    if (!attention || !state || !phase) return false;

    // Geometric phase includes both Berry phase and dynamical phase
    complex double berry;
    if (!compute_berry_phase(attention, state, &berry)) return false;

    // Add contribution from curvature
    double curv = attention->manifold ? attention->manifold->curvature : 0.0;
    complex double geometric = cexp(I * curv * (double)state->seq_length);

    *phase = berry * geometric;
    return true;
}

bool apply_phase_correction(geometric_attention_t* attention,
                          attention_state_t* state) {
    if (!attention || !state) return false;

    complex double phase;
    if (!compute_geometric_phase(attention, state, &phase)) return false;

    // Remove geometric phase (correct for it)
    complex double correction = conj(phase) / cabs(phase);
    size_t n = state->seq_length * state->head_dim;

    for (size_t i = 0; i < n; i++) {
        state->values[i] *= correction;
    }

    return true;
}

// =============================================================================
// Error Detection and Correction
// =============================================================================

bool detect_errors(geometric_attention_t* attention,
                  const attention_state_t* state,
                  double* error_rates) {
    if (!attention || !state || !error_rates) return false;

    size_t n = state->seq_length * state->head_dim;

    // Compute error rate based on deviation from unitarity
    double total_norm = 0.0;
    for (size_t i = 0; i < n; i++) {
        total_norm += cabs(state->values[i]) * cabs(state->values[i]);
    }

    *error_rates = fabs(1.0 - total_norm / (double)n);

    attention->metrics.error_rate = *error_rates;
    return true;
}

bool correct_errors(geometric_attention_t* attention,
                   attention_state_t* state) {
    if (!attention || !state) return false;

    size_t n = state->seq_length * state->head_dim;

    // Normalize state to correct amplitude errors
    double norm = 0.0;
    for (size_t i = 0; i < n; i++) {
        norm += cabs(state->values[i]) * cabs(state->values[i]);
    }

    if (norm > EPSILON) {
        double scale = 1.0 / sqrt(norm / (double)n);
        for (size_t i = 0; i < n; i++) {
            state->values[i] *= scale;
        }
    }

    return true;
}

bool validate_correction(geometric_attention_t* attention,
                        const attention_state_t* state) {
    if (!attention || !state) return false;

    double error_rate;
    if (!detect_errors(attention, state, &error_rate)) return false;

    return error_rate < 0.01;  // Less than 1% error
}

// =============================================================================
// Performance Monitoring
// =============================================================================

bool attention_get_metrics(const geometric_attention_t* attention,
                          attention_metrics_t* metrics) {
    if (!attention || !metrics) return false;
    *metrics = attention->metrics;
    return true;
}

bool attention_monitor_performance(geometric_attention_t* attention,
                                  attention_metrics_t* metrics) {
    if (!attention || !metrics) return false;

    // Update cache statistics
    if (attention->cache) {
        double hit_rate = 0.0;
        size_t total = attention->cache->hit_count + attention->cache->miss_count;
        if (total > 0) {
            hit_rate = (double)attention->cache->hit_count / (double)total;
        }
        metrics->geometric_score = hit_rate;
    }

    *metrics = attention->metrics;
    return true;
}

bool attention_optimize_performance(geometric_attention_t* attention,
                                   const attention_metrics_t* metrics) {
    if (!attention || !metrics) return false;

    // Adjust based on metrics
    if (metrics->error_rate > 0.1) {
        // High error rate: increase precision
        if (attention->manifold) {
            // Refine Christoffel symbols
            ManifoldProperties props = analyze_manifold_properties(attention->manifold);
            if (props.is_negatively_curved) {
                optimize_hyperbolic_attention(attention, props);
            } else {
                optimize_riemannian_attention(attention, props);
            }
        }
    }

    return true;
}

// =============================================================================
// Utility Functions
// =============================================================================

bool export_attention_data(const geometric_attention_t* attention,
                          const char* filename) {
    if (!attention || !filename) return false;

    FILE* f = fopen(filename, "wb");
    if (!f) return false;

    // Write config
    fwrite(&attention->config, sizeof(attention_config_t), 1, f);

    // Write manifold data
    if (attention->manifold) {
        size_t n = attention->manifold->dimension;
        fwrite(&n, sizeof(size_t), 1, f);
        fwrite(&attention->manifold->type, sizeof(manifold_type_t), 1, f);
        fwrite(attention->manifold->metric, sizeof(double), n * n, f);
        fwrite(&attention->manifold->curvature, sizeof(double), 1, f);
    }

    // Write metrics
    fwrite(&attention->metrics, sizeof(attention_metrics_t), 1, f);

    fclose(f);
    return true;
}

bool import_attention_data(geometric_attention_t* attention,
                          const char* filename) {
    if (!attention || !filename) return false;

    FILE* f = fopen(filename, "rb");
    if (!f) return false;

    // Read config
    fread(&attention->config, sizeof(attention_config_t), 1, f);

    // Read manifold data
    size_t n;
    manifold_type_t type;
    if (fread(&n, sizeof(size_t), 1, f) == 1) {
        fread(&type, sizeof(manifold_type_t), 1, f);

        if (attention->manifold) {
            destroy_manifold(attention->manifold);
        }
        attention->manifold = create_attention_manifold(n, type);

        if (attention->manifold) {
            fread(attention->manifold->metric, sizeof(double), n * n, f);
            fread(&attention->manifold->curvature, sizeof(double), 1, f);
        }
    }

    // Read metrics
    fread(&attention->metrics, sizeof(attention_metrics_t), 1, f);

    fclose(f);
    return true;
}

void free_attention_state(attention_state_t* state) {
    if (!state) return;
    free(state->queries);
    free(state->keys);
    free(state->values);
    state->queries = NULL;
    state->keys = NULL;
    state->values = NULL;
}
