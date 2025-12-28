/**
 * @file differential_geometry.h
 * @brief Distributed Differential Geometry Operations for Quantum Geometric Computing
 *
 * Provides distributed differential geometry computations including:
 * - Metric tensor calculations across nodes
 * - Christoffel symbol computation
 * - Curvature tensor operations (Riemann, Ricci, scalar)
 * - Geodesic computation on quantum state manifolds
 * - Parallel transport operations
 * - Connection forms and structure equations
 *
 * Part of the QGTL Distributed Computing Framework.
 */

#ifndef DIFFERENTIAL_GEOMETRY_H
#define DIFFERENTIAL_GEOMETRY_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Forward declare complex type
#ifndef COMPLEX_FLOAT_DEFINED
typedef struct {
    double real;
    double imag;
} ComplexDouble;
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================

#define DIFFGEO_MAX_DIMENSIONS 16
#define DIFFGEO_MAX_NODES 256
#define DIFFGEO_MAX_NAME_LENGTH 128
#define DIFFGEO_DEFAULT_TOLERANCE 1e-12
#define DIFFGEO_MAX_GEODESIC_STEPS 10000

// ============================================================================
// Enumerations
// ============================================================================

/**
 * Manifold types for quantum state spaces
 */
typedef enum {
    MANIFOLD_PROJECTIVE_HILBERT,      // CP^n projective Hilbert space
    MANIFOLD_BLOCH_SPHERE,            // Single qubit Bloch sphere (S^2)
    MANIFOLD_GRASSMANNIAN,            // Grassmannian Gr(k,n)
    MANIFOLD_FLAG_MANIFOLD,           // Flag manifold
    MANIFOLD_STIEFEL,                 // Stiefel manifold V(k,n)
    MANIFOLD_UNITARY_GROUP,           // U(n) Lie group
    MANIFOLD_SPECIAL_UNITARY,         // SU(n) Lie group
    MANIFOLD_SYMPLECTIC,              // Symplectic manifold
    MANIFOLD_KAHLER,                  // Kähler manifold
    MANIFOLD_CUSTOM                   // User-defined manifold
} diffgeo_manifold_type_t;

/**
 * Metric types
 */
typedef enum {
    METRIC_FUBINI_STUDY,              // Fubini-Study metric (quantum)
    METRIC_BURES,                     // Bures metric (mixed states)
    METRIC_WIGNER_YANASE,             // Wigner-Yanase metric
    METRIC_BOGOLIUBOV_KUBO_MORI,      // BKM metric
    METRIC_EUCLIDEAN,                 // Flat Euclidean
    METRIC_MINKOWSKI,                 // Minkowski (pseudo-Riemannian)
    METRIC_HYPERBOLIC,                // Hyperbolic (negative curvature)
    METRIC_SPHERICAL,                 // Spherical (positive curvature)
    METRIC_CUSTOM                     // User-defined metric
} diffgeo_metric_type_t;

/**
 * Curvature types
 */
typedef enum {
    CURVATURE_RIEMANN,                // Full Riemann curvature tensor
    CURVATURE_RICCI,                  // Ricci curvature tensor
    CURVATURE_SCALAR,                 // Scalar curvature
    CURVATURE_WEYL,                   // Weyl conformal tensor
    CURVATURE_SECTIONAL,              // Sectional curvature
    CURVATURE_HOLOMORPHIC_SECTIONAL,  // Holomorphic sectional (Kähler)
    CURVATURE_BISECTIONAL             // Holomorphic bisectional (Kähler)
} diffgeo_curvature_type_t;

/**
 * Distribution strategy for tensor computations
 */
typedef enum {
    DIFFGEO_DIST_BLOCK,               // Block distribution
    DIFFGEO_DIST_CYCLIC,              // Cyclic distribution
    DIFFGEO_DIST_COMPONENT,           // Component-wise distribution
    DIFFGEO_DIST_ADAPTIVE             // Adaptive based on sparsity
} diffgeo_distribution_t;

/**
 * Geodesic solver methods
 */
typedef enum {
    GEODESIC_EULER,                   // Forward Euler
    GEODESIC_MIDPOINT,                // Midpoint method
    GEODESIC_RK4,                     // Classical Runge-Kutta
    GEODESIC_RK45,                    // Runge-Kutta-Fehlberg adaptive
    GEODESIC_SYMPLECTIC               // Symplectic integrator
} diffgeo_geodesic_method_t;

// ============================================================================
// Data Structures
// ============================================================================

/**
 * Point on the manifold
 */
typedef struct {
    ComplexDouble* coordinates;       // Local coordinates
    size_t dimension;                 // Manifold dimension
    int chart_index;                  // Which chart/patch
} diffgeo_point_t;

/**
 * Tangent vector at a point
 */
typedef struct {
    ComplexDouble* components;        // Vector components
    size_t dimension;                 // Vector space dimension
    diffgeo_point_t* base_point;      // Point of attachment
} diffgeo_tangent_vector_t;

/**
 * Metric tensor at a point (g_ij)
 */
typedef struct {
    ComplexDouble* components;        // Metric components (dim x dim)
    size_t dimension;                 // Manifold dimension
    diffgeo_point_t* base_point;      // Point where evaluated
    diffgeo_metric_type_t type;       // Metric type
    bool is_hermitian;                // Whether metric is Hermitian
} diffgeo_metric_tensor_t;

/**
 * Christoffel symbols (connection coefficients)
 * Γ^k_ij = (1/2) g^{kl} (∂_i g_{jl} + ∂_j g_{il} - ∂_l g_{ij})
 */
typedef struct {
    ComplexDouble* components;        // Γ^k_ij as (dim x dim x dim) array
    size_t dimension;                 // Manifold dimension
    diffgeo_point_t* base_point;      // Point where evaluated
    bool is_torsion_free;             // Whether connection is torsion-free
} diffgeo_christoffel_t;

/**
 * Curvature tensor
 */
typedef struct {
    diffgeo_curvature_type_t type;    // Type of curvature
    ComplexDouble* components;        // Tensor components
    size_t dimension;                 // Manifold dimension
    size_t rank;                      // Tensor rank (4 for Riemann, 2 for Ricci)
    diffgeo_point_t* base_point;      // Evaluation point
} diffgeo_curvature_t;

/**
 * Geodesic curve on the manifold
 */
typedef struct {
    diffgeo_point_t* points;          // Points along geodesic
    diffgeo_tangent_vector_t* tangents; // Tangent vectors along geodesic
    size_t num_points;                // Number of points
    double* parameter_values;         // Parameter values (affine parameter)
    double total_length;              // Arc length of geodesic
    bool is_closed;                   // Whether geodesic is closed
} diffgeo_geodesic_t;

/**
 * Parallel transport result
 */
typedef struct {
    diffgeo_tangent_vector_t initial; // Initial vector
    diffgeo_tangent_vector_t final;   // Final (transported) vector
    diffgeo_geodesic_t* path;         // Path of transport
    ComplexDouble holonomy_factor;    // Holonomy (for closed paths)
} diffgeo_transport_result_t;

/**
 * Distributed computation context
 */
typedef struct {
    int num_nodes;                    // Total number of nodes
    int node_rank;                    // This node's rank
    int* node_assignments;            // Which node handles which components
    diffgeo_distribution_t strategy;  // Distribution strategy
    bool use_gpu;                     // Use GPU acceleration
} diffgeo_dist_context_t;

/**
 * Computation statistics
 */
typedef struct {
    uint64_t total_operations;        // Total operations performed
    uint64_t tensor_contractions;     // Number of tensor contractions
    uint64_t metric_evaluations;      // Metric tensor evaluations
    uint64_t christoffel_evaluations; // Christoffel symbol evaluations
    uint64_t curvature_evaluations;   // Curvature tensor evaluations
    uint64_t geodesic_steps;          // Geodesic integration steps
    double total_time_ms;             // Total computation time
    double communication_time_ms;     // Time spent in communication
    double computation_time_ms;       // Time spent in computation
    size_t bytes_communicated;        // Total bytes communicated
} diffgeo_stats_t;

/**
 * Configuration options
 */
typedef struct {
    diffgeo_manifold_type_t manifold_type;
    diffgeo_metric_type_t metric_type;
    diffgeo_distribution_t distribution;
    double numerical_tolerance;
    size_t max_iterations;
    bool enable_caching;              // Cache intermediate results
    bool enable_symmetry;             // Exploit tensor symmetries
    bool enable_gpu;                  // Use GPU acceleration
    diffgeo_geodesic_method_t geodesic_method;
    size_t geodesic_max_steps;
} diffgeo_config_t;

/**
 * Opaque differential geometry engine handle
 */
typedef struct diffgeo_engine diffgeo_engine_t;

// ============================================================================
// Initialization and Configuration
// ============================================================================

/**
 * Create differential geometry engine
 */
diffgeo_engine_t* diffgeo_engine_create(void);

/**
 * Create with configuration
 */
diffgeo_engine_t* diffgeo_engine_create_with_config(
    const diffgeo_config_t* config);

/**
 * Get default configuration
 */
diffgeo_config_t diffgeo_default_config(void);

/**
 * Destroy differential geometry engine
 */
void diffgeo_engine_destroy(diffgeo_engine_t* engine);

/**
 * Set distributed context
 */
bool diffgeo_set_dist_context(diffgeo_engine_t* engine,
                               const diffgeo_dist_context_t* ctx);

/**
 * Reset engine state
 */
bool diffgeo_engine_reset(diffgeo_engine_t* engine);

// ============================================================================
// Manifold and Point Operations
// ============================================================================

/**
 * Create a point on the manifold
 */
diffgeo_point_t* diffgeo_point_create(diffgeo_engine_t* engine,
                                       const ComplexDouble* coordinates,
                                       size_t dimension);

/**
 * Destroy a point
 */
void diffgeo_point_destroy(diffgeo_point_t* point);

/**
 * Create tangent vector at a point
 */
diffgeo_tangent_vector_t* diffgeo_tangent_create(
    diffgeo_engine_t* engine,
    diffgeo_point_t* base_point,
    const ComplexDouble* components);

/**
 * Destroy tangent vector
 */
void diffgeo_tangent_destroy(diffgeo_tangent_vector_t* vector);

/**
 * Compute exponential map: exp_p(v) -> q
 */
diffgeo_point_t* diffgeo_exp_map(diffgeo_engine_t* engine,
                                  diffgeo_point_t* base_point,
                                  diffgeo_tangent_vector_t* vector);

/**
 * Compute logarithm map: log_p(q) -> v
 */
diffgeo_tangent_vector_t* diffgeo_log_map(diffgeo_engine_t* engine,
                                           diffgeo_point_t* p,
                                           diffgeo_point_t* q);

/**
 * Compute distance between points
 */
double diffgeo_distance(diffgeo_engine_t* engine,
                        diffgeo_point_t* p,
                        diffgeo_point_t* q);

// ============================================================================
// Metric Tensor Operations
// ============================================================================

/**
 * Compute metric tensor at a point
 */
diffgeo_metric_tensor_t* diffgeo_compute_metric(
    diffgeo_engine_t* engine,
    diffgeo_point_t* point);

/**
 * Compute metric tensor (distributed across nodes)
 */
diffgeo_metric_tensor_t* diffgeo_compute_metric_distributed(
    diffgeo_engine_t* engine,
    diffgeo_point_t* point,
    const diffgeo_dist_context_t* ctx);

/**
 * Compute inverse metric g^{ij}
 */
diffgeo_metric_tensor_t* diffgeo_compute_inverse_metric(
    diffgeo_engine_t* engine,
    const diffgeo_metric_tensor_t* metric);

/**
 * Compute inner product <u, v>_g
 */
ComplexDouble diffgeo_inner_product(diffgeo_engine_t* engine,
                                    const diffgeo_metric_tensor_t* metric,
                                    const diffgeo_tangent_vector_t* u,
                                    const diffgeo_tangent_vector_t* v);

/**
 * Compute norm ||v||_g
 */
double diffgeo_norm(diffgeo_engine_t* engine,
                    const diffgeo_metric_tensor_t* metric,
                    const diffgeo_tangent_vector_t* v);

/**
 * Destroy metric tensor
 */
void diffgeo_metric_destroy(diffgeo_metric_tensor_t* metric);

// ============================================================================
// Christoffel Symbols
// ============================================================================

/**
 * Compute Christoffel symbols at a point
 */
diffgeo_christoffel_t* diffgeo_compute_christoffel(
    diffgeo_engine_t* engine,
    diffgeo_point_t* point);

/**
 * Compute Christoffel symbols (distributed)
 */
diffgeo_christoffel_t* diffgeo_compute_christoffel_distributed(
    diffgeo_engine_t* engine,
    diffgeo_point_t* point,
    const diffgeo_dist_context_t* ctx);

/**
 * Get specific Christoffel symbol component Γ^k_ij
 */
ComplexDouble diffgeo_christoffel_component(
    const diffgeo_christoffel_t* christoffel,
    size_t k, size_t i, size_t j);

/**
 * Destroy Christoffel symbols
 */
void diffgeo_christoffel_destroy(diffgeo_christoffel_t* christoffel);

// ============================================================================
// Curvature Computations
// ============================================================================

/**
 * Compute Riemann curvature tensor R^l_{ijk}
 */
diffgeo_curvature_t* diffgeo_compute_riemann(
    diffgeo_engine_t* engine,
    diffgeo_point_t* point);

/**
 * Compute Riemann tensor (distributed)
 */
diffgeo_curvature_t* diffgeo_compute_riemann_distributed(
    diffgeo_engine_t* engine,
    diffgeo_point_t* point,
    const diffgeo_dist_context_t* ctx);

/**
 * Compute Ricci tensor R_ij = R^k_{ikj}
 */
diffgeo_curvature_t* diffgeo_compute_ricci(
    diffgeo_engine_t* engine,
    diffgeo_point_t* point);

/**
 * Compute scalar curvature R = g^{ij} R_ij
 */
double diffgeo_compute_scalar_curvature(
    diffgeo_engine_t* engine,
    diffgeo_point_t* point);

/**
 * Compute sectional curvature K(u,v)
 */
double diffgeo_compute_sectional_curvature(
    diffgeo_engine_t* engine,
    diffgeo_point_t* point,
    const diffgeo_tangent_vector_t* u,
    const diffgeo_tangent_vector_t* v);

/**
 * Compute Weyl conformal tensor
 */
diffgeo_curvature_t* diffgeo_compute_weyl(
    diffgeo_engine_t* engine,
    diffgeo_point_t* point);

/**
 * Compute holomorphic sectional curvature (for Kähler manifolds)
 */
double diffgeo_compute_holomorphic_sectional(
    diffgeo_engine_t* engine,
    diffgeo_point_t* point,
    const diffgeo_tangent_vector_t* v);

/**
 * Destroy curvature tensor
 */
void diffgeo_curvature_destroy(diffgeo_curvature_t* curvature);

// ============================================================================
// Geodesic Operations
// ============================================================================

/**
 * Compute geodesic from point with initial velocity
 */
diffgeo_geodesic_t* diffgeo_compute_geodesic(
    diffgeo_engine_t* engine,
    diffgeo_point_t* start,
    diffgeo_tangent_vector_t* velocity,
    double total_parameter,
    size_t num_steps);

/**
 * Compute geodesic between two points
 */
diffgeo_geodesic_t* diffgeo_compute_geodesic_between(
    diffgeo_engine_t* engine,
    diffgeo_point_t* start,
    diffgeo_point_t* end,
    size_t num_steps);

/**
 * Compute geodesic (distributed computation)
 */
diffgeo_geodesic_t* diffgeo_compute_geodesic_distributed(
    diffgeo_engine_t* engine,
    diffgeo_point_t* start,
    diffgeo_tangent_vector_t* velocity,
    double total_parameter,
    size_t num_steps,
    const diffgeo_dist_context_t* ctx);

/**
 * Compute geodesic distance (arc length)
 */
double diffgeo_geodesic_length(const diffgeo_geodesic_t* geodesic);

/**
 * Get point along geodesic at parameter value
 */
diffgeo_point_t* diffgeo_geodesic_point_at(
    diffgeo_engine_t* engine,
    const diffgeo_geodesic_t* geodesic,
    double parameter);

/**
 * Check if geodesic is complete (no singularities)
 */
bool diffgeo_geodesic_is_complete(const diffgeo_geodesic_t* geodesic);

/**
 * Destroy geodesic
 */
void diffgeo_geodesic_destroy(diffgeo_geodesic_t* geodesic);

// ============================================================================
// Parallel Transport
// ============================================================================

/**
 * Parallel transport vector along geodesic
 */
diffgeo_transport_result_t* diffgeo_parallel_transport(
    diffgeo_engine_t* engine,
    diffgeo_tangent_vector_t* vector,
    diffgeo_geodesic_t* path);

/**
 * Parallel transport along curve (not necessarily geodesic)
 */
diffgeo_transport_result_t* diffgeo_parallel_transport_curve(
    diffgeo_engine_t* engine,
    diffgeo_tangent_vector_t* vector,
    diffgeo_point_t** curve_points,
    size_t num_points);

/**
 * Compute holonomy around closed loop
 */
ComplexDouble diffgeo_compute_holonomy(
    diffgeo_engine_t* engine,
    diffgeo_tangent_vector_t* vector,
    diffgeo_point_t** loop_points,
    size_t num_points);

/**
 * Destroy transport result
 */
void diffgeo_transport_destroy(diffgeo_transport_result_t* result);

// ============================================================================
// Quantum-Specific Operations
// ============================================================================

/**
 * Compute quantum geometric tensor Q_ij = g_ij + i*F_ij
 * where g is Fubini-Study metric and F is Berry curvature
 */
bool diffgeo_compute_quantum_geometric_tensor(
    diffgeo_engine_t* engine,
    const ComplexDouble* state,
    size_t dim,
    ComplexDouble* qgt_out);

/**
 * Compute Fubini-Study metric for quantum states
 */
bool diffgeo_compute_fubini_study(
    diffgeo_engine_t* engine,
    const ComplexDouble* state,
    size_t dim,
    const ComplexDouble* param_derivatives,
    size_t num_params,
    double* metric_out);

/**
 * Compute Berry curvature F_ij
 */
bool diffgeo_compute_berry_curvature(
    diffgeo_engine_t* engine,
    const ComplexDouble* state,
    size_t dim,
    const ComplexDouble* param_derivatives,
    size_t num_params,
    double* curvature_out);

/**
 * Compute Berry phase around parameter loop
 */
ComplexDouble diffgeo_compute_berry_phase(
    diffgeo_engine_t* engine,
    const ComplexDouble** states,
    size_t num_states,
    size_t dim);

/**
 * Compute Bures metric for density matrices
 */
bool diffgeo_compute_bures_metric(
    diffgeo_engine_t* engine,
    const ComplexDouble* rho,
    size_t dim,
    double* metric_out);

// ============================================================================
// Statistics and Reporting
// ============================================================================

/**
 * Get computation statistics
 */
bool diffgeo_get_stats(diffgeo_engine_t* engine, diffgeo_stats_t* stats);

/**
 * Reset statistics
 */
void diffgeo_reset_stats(diffgeo_engine_t* engine);

/**
 * Generate report
 */
char* diffgeo_generate_report(diffgeo_engine_t* engine);

/**
 * Export to JSON
 */
char* diffgeo_export_json(diffgeo_engine_t* engine);

/**
 * Export to file
 */
bool diffgeo_export_to_file(diffgeo_engine_t* engine,
                            const char* filename);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Get manifold type name
 */
const char* diffgeo_manifold_type_name(diffgeo_manifold_type_t type);

/**
 * Get metric type name
 */
const char* diffgeo_metric_type_name(diffgeo_metric_type_t type);

/**
 * Get curvature type name
 */
const char* diffgeo_curvature_type_name(diffgeo_curvature_type_t type);

/**
 * Get distribution strategy name
 */
const char* diffgeo_distribution_name(diffgeo_distribution_t dist);

/**
 * Get geodesic method name
 */
const char* diffgeo_geodesic_method_name(diffgeo_geodesic_method_t method);

/**
 * Free allocated string
 */
void diffgeo_free_string(char* str);

/**
 * Get last error message
 */
const char* diffgeo_get_last_error(diffgeo_engine_t* engine);

#ifdef __cplusplus
}
#endif

#endif // DIFFERENTIAL_GEOMETRY_H
