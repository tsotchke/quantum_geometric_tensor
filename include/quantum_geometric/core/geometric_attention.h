#ifndef GEOMETRIC_ATTENTION_H
#define GEOMETRIC_ATTENTION_H

#include <stdbool.h>
#include <stddef.h>
#include <complex.h>

// Attention types
typedef enum {
    ATTENTION_GEOMETRIC,     // Geometric attention
    ATTENTION_QUANTUM,       // Quantum attention
    ATTENTION_HYBRID,        // Hybrid attention
    ATTENTION_ADAPTIVE      // Adaptive attention
} attention_type_t;

// Attention geometry types (module-specific to avoid conflicts)
typedef enum {
    ATTN_GEOMETRY_MANIFOLD,       // Manifold geometry
    ATTN_GEOMETRY_COMPLEX,        // Complex projective
    ATTN_GEOMETRY_KAHLER,         // Kahler manifold
    ATTN_GEOMETRY_FUBINI_STUDY   // Fubini-Study metric
} attn_geometry_type_t;

// Attention connection types (module-specific)
typedef enum {
    ATTN_CONNECTION_NATURAL,      // Natural connection
    ATTN_CONNECTION_GEOMETRIC,    // Geometric connection
    ATTN_CONNECTION_QUANTUM,      // Quantum connection
    ATTN_CONNECTION_HYBRID       // Hybrid connection
} attn_connection_type_t;

// Phase types
typedef enum {
    PHASE_BERRY,            // Berry phase
    PHASE_GEOMETRIC,        // Geometric phase
    PHASE_DYNAMIC,          // Dynamic phase
    PHASE_TOPOLOGICAL      // Topological phase
} phase_type_t;

// Attention configuration
typedef struct {
    attention_type_t type;          // Attention type
    attn_geometry_type_t geometry;  // Geometry type
    attn_connection_type_t connection; // Connection type
    size_t attention_heads;         // Number of heads
    size_t head_dim;               // Head dimension
    bool use_error_correction;     // Error correction flag
} attention_config_t;

// Geometric parameters for attention
typedef struct {
    attn_geometry_type_t type;      // Geometry type
    double metric_tensor;           // Metric tensor
    double curvature;              // Curvature
    double connection_coeff;        // Connection coefficient
    complex double* phase_factors;  // Phase factors
    size_t num_factors;            // Number of factors
} attn_geometric_params_t;

// Attention state
typedef struct {
    complex double* queries;        // Query states
    complex double* keys;           // Key states
    complex double* values;         // Value states
    size_t seq_length;             // Sequence length
    size_t batch_size;             // Batch size
    size_t head_dim;              // Head dimension
} attention_state_t;

// Attention metrics
typedef struct {
    double attention_score;         // Attention score
    double geometric_score;         // Geometric score
    double phase_coherence;         // Phase coherence
    double error_rate;             // Error rate
    size_t operation_count;        // Operation count
    double execution_time;         // Execution time
} attention_metrics_t;

// =============================================================================
// Internal Types (exposed for implementation)
// =============================================================================

// Maximum number of geodesic paths to track
#define MAX_GEODESICS 256

// Manifold type enumeration
typedef enum {
    MANIFOLD_EUCLIDEAN,          // Flat Euclidean space
    MANIFOLD_HYPERBOLIC,         // Hyperbolic space (negative curvature)
    MANIFOLD_SPHERICAL,          // Spherical manifold (positive curvature)
    MANIFOLD_RIEMANNIAN,         // General Riemannian manifold
    MANIFOLD_KAHLER              // Kahler manifold (complex)
} manifold_type_t;

// Manifold structure for geometric attention
typedef struct Manifold {
    manifold_type_t type;           // Type of manifold
    size_t dimension;               // Manifold dimension
    double* metric;                 // Metric tensor components
    double* christoffel;            // Christoffel symbols
    double curvature;               // Scalar curvature
    double* riemann_tensor;         // Riemann curvature tensor
    void* extra_data;               // Type-specific data
} Manifold;

// Geodesic path on the manifold
typedef struct GeodesicPath {
    double* points;                 // Path points (dimension * num_points)
    double* tangent;                // Tangent vectors along path
    size_t num_points;              // Number of points
    size_t dimension;               // Point dimension
    double length;                  // Geodesic length
    double energy;                  // Path energy
    bool is_closed;                 // Whether path is closed (loop)
} GeodesicPath;

// Properties of the attention manifold
typedef struct ManifoldProperties {
    double sectional_curvature;     // Sectional curvature
    double scalar_curvature;        // Scalar curvature
    double ricci_scalar;            // Ricci scalar
    double injectivity_radius;      // Injectivity radius
    bool is_compact;                // Compactness
    bool is_complete;               // Geodesic completeness
    bool is_negatively_curved;      // Negative curvature flag
} ManifoldProperties;

// Cache for geometric computations
typedef struct GeometricCache {
    double* cached_metrics;         // Cached metric tensors
    double* cached_connections;     // Cached connection coefficients
    complex double* cached_phases;  // Cached phase factors
    size_t cache_size;              // Current cache size
    size_t cache_capacity;          // Maximum cache capacity
    size_t hit_count;               // Cache hits
    size_t miss_count;              // Cache misses
    bool is_valid;                  // Cache validity flag
} GeometricCache;

// Legacy type alias for backward compatibility
typedef attention_config_t AttentionConfig;

// Opaque attention handle
typedef struct geometric_attention_t geometric_attention_t;

// Core functions
geometric_attention_t* create_geometric_attention(const attention_config_t* config);
void destroy_geometric_attention(geometric_attention_t* attention);

// Initialization functions
bool attention_init_geometry(geometric_attention_t* attention,
                            const attn_geometric_params_t* params);
bool attention_init_state(geometric_attention_t* attention,
                         const attention_state_t* state);
bool attention_validate_init(geometric_attention_t* attention);

// Attention operations
bool compute_attention(geometric_attention_t* attention,
                      const attention_state_t* input,
                      attention_state_t* output);
bool apply_geometric_phase(geometric_attention_t* attention,
                         phase_type_t phase_type,
                         attention_state_t* state);
bool compute_attention_weights(geometric_attention_t* attention,
                             const attention_state_t* state,
                             complex double* weights);

// Geometric operations
bool attention_compute_metric(geometric_attention_t* attention,
                             const attn_geometric_params_t* params,
                             double* metric);
bool attention_compute_connection(geometric_attention_t* attention,
                                 const attn_geometric_params_t* params,
                                 double* connection);
bool attention_compute_curvature(geometric_attention_t* attention,
                                const attn_geometric_params_t* params,
                                double* curvature);

// Phase operations
bool compute_berry_phase(geometric_attention_t* attention,
                        const attention_state_t* state,
                        complex double* phase);
bool compute_geometric_phase(geometric_attention_t* attention,
                           const attention_state_t* state,
                           complex double* phase);
bool apply_phase_correction(geometric_attention_t* attention,
                          attention_state_t* state);

// Error correction
bool detect_errors(geometric_attention_t* attention,
                  const attention_state_t* state,
                  double* error_rates);
bool correct_errors(geometric_attention_t* attention,
                   attention_state_t* state);
bool validate_correction(geometric_attention_t* attention,
                        const attention_state_t* state);

// Performance monitoring
bool attention_get_metrics(const geometric_attention_t* attention,
                          attention_metrics_t* metrics);
bool attention_monitor_performance(geometric_attention_t* attention,
                                  attention_metrics_t* metrics);
bool attention_optimize_performance(geometric_attention_t* attention,
                                   const attention_metrics_t* metrics);

// Utility functions
bool export_attention_data(const geometric_attention_t* attention,
                          const char* filename);
bool import_attention_data(geometric_attention_t* attention,
                          const char* filename);
void free_attention_state(attention_state_t* state);

// =============================================================================
// Internal Functions (exposed for implementation)
// =============================================================================

// Cache management
GeometricCache* create_geometric_cache(void);
void destroy_geometric_cache(GeometricCache* cache);
bool check_geometric_cache(GeometricCache* cache,
                          const attention_state_t* state,
                          complex double* cached_result);
void update_geometric_cache(GeometricCache* cache,
                           const attention_state_t* state,
                           const complex double* result);
void restore_cached_patterns(geometric_attention_t* attention);

// Manifold operations
Manifold* create_attention_manifold(size_t dimension, manifold_type_t type);
void destroy_manifold(Manifold* manifold);
ManifoldProperties analyze_manifold_properties(const Manifold* manifold);
double compute_geodesic_distance(const Manifold* manifold,
                                const double* point1,
                                const double* point2);

// Optimization for different geometries
void optimize_hyperbolic_attention(geometric_attention_t* attention,
                                  ManifoldProperties props);
void optimize_riemannian_attention(geometric_attention_t* attention,
                                  ManifoldProperties props);
void optimize_euclidean_attention(geometric_attention_t* attention,
                                 ManifoldProperties props);

// Geodesic path operations
GeodesicPath* compute_geodesic(const Manifold* manifold,
                              const double* start,
                              const double* end,
                              size_t num_points);
void destroy_geodesic_path(GeodesicPath* path);
double* parallel_transport(const Manifold* manifold,
                          const GeodesicPath* path,
                          const double* vector);

// Geometric attention internals
void compute_attention_curvature(geometric_attention_t* attention,
                                double* curvature_out);
void apply_geodesic_attention(geometric_attention_t* attention,
                             const attention_state_t* input,
                             attention_state_t* output);

#endif // GEOMETRIC_ATTENTION_H
