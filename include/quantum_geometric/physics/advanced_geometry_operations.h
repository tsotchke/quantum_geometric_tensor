/**
 * @file advanced_geometry_operations.h
 * @brief Advanced differential geometry operations for quantum systems
 *
 * Implements sophisticated geometric operations including Riemannian
 * geometry, fiber bundles, characteristic classes, and gauge theory
 * structures for quantum geometric applications.
 */

#ifndef ADVANCED_GEOMETRY_OPERATIONS_H
#define ADVANCED_GEOMETRY_OPERATIONS_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

// Portable complex type definition
typedef double _Complex qgt_complex_t;

// Forward declarations
struct geometric_tensor;
struct quantum_state;

// =============================================================================
// Geometry Type Enums
// =============================================================================

/**
 * Manifold types
 */
typedef enum {
    MANIFOLD_EUCLIDEAN,              // Flat Euclidean space
    MANIFOLD_SPHERE,                 // n-sphere S^n
    MANIFOLD_TORUS,                  // n-torus T^n
    MANIFOLD_HYPERBOLIC,             // Hyperbolic space H^n
    MANIFOLD_PROJECTIVE_COMPLEX,     // Complex projective space CP^n
    MANIFOLD_PROJECTIVE_REAL,        // Real projective space RP^n
    MANIFOLD_GRASSMANNIAN,           // Grassmannian Gr(k,n)
    MANIFOLD_FLAG,                   // Flag manifold
    MANIFOLD_STIEFEL,                // Stiefel manifold
    MANIFOLD_LIE_GROUP,              // Lie group manifold
    MANIFOLD_SYMPLECTIC,             // Symplectic manifold
    MANIFOLD_KAHLER,                 // Kahler manifold
    MANIFOLD_CALABI_YAU,             // Calabi-Yau manifold
    MANIFOLD_CUSTOM                  // User-defined
} ManifoldType;

/**
 * Fiber bundle types
 */
typedef enum {
    BUNDLE_TRIVIAL,                  // Product bundle M × F
    BUNDLE_TANGENT,                  // Tangent bundle TM
    BUNDLE_COTANGENT,                // Cotangent bundle T*M
    BUNDLE_FRAME,                    // Frame bundle
    BUNDLE_PRINCIPAL,                // Principal G-bundle
    BUNDLE_VECTOR,                   // Vector bundle
    BUNDLE_LINE,                     // Line bundle
    BUNDLE_SPINOR,                   // Spinor bundle
    BUNDLE_JET                       // Jet bundle
} FiberBundleType;

/**
 * Connection types
 */
typedef enum {
    CONNECTION_LEVI_CIVITA,          // Levi-Civita (metric compatible, torsion-free)
    CONNECTION_AFFINE,               // General affine connection
    CONNECTION_YANG_MILLS,           // Yang-Mills connection
    CONNECTION_BERRY,                // Berry connection (quantum geometric)
    CONNECTION_SPIN,                 // Spin connection
    CONNECTION_CARTAN,               // Cartan connection
    CONNECTION_EHRESMANN             // Ehresmann connection
} ConnectionType;

/**
 * Curvature types
 */
typedef enum {
    CURVATURE_RIEMANN,               // Riemann curvature tensor
    CURVATURE_RICCI,                 // Ricci curvature
    CURVATURE_SCALAR,                // Scalar curvature
    CURVATURE_WEYL,                  // Weyl conformal tensor
    CURVATURE_SECTIONAL,             // Sectional curvature
    CURVATURE_GAUSSIAN,              // Gaussian curvature (2D)
    CURVATURE_MEAN,                  // Mean curvature
    CURVATURE_YANG_MILLS             // Yang-Mills field strength
} CurvatureType;

// =============================================================================
// Manifold Structures
// =============================================================================

/**
 * Chart (local coordinate system)
 */
typedef struct {
    size_t chart_id;
    double* coord_min;               // Coordinate lower bounds
    double* coord_max;               // Coordinate upper bounds
    size_t dimension;
    void* transition_to_next;        // Transition function pointer
    size_t* overlapping_charts;
    size_t num_overlaps;
} Chart;

/**
 * Atlas (collection of charts)
 */
typedef struct {
    Chart* charts;
    size_t num_charts;
    size_t dimension;
    bool is_oriented;
    bool is_complete;
} Atlas;

/**
 * Riemannian manifold
 */
typedef struct {
    ManifoldType type;
    size_t dimension;
    Atlas* atlas;
    struct geometric_tensor* metric;          // Metric tensor g_{ij}
    struct geometric_tensor* inverse_metric;  // Inverse metric g^{ij}
    double* christoffel;                      // Christoffel symbols Γ^k_{ij}
    size_t christoffel_size;
    double determinant;                       // det(g)
    double total_volume;
    bool is_compact;
    bool is_orientable;
    bool is_simply_connected;
    double* sectional_curvature_bounds;       // [min, max]
} RiemannianManifold;

/**
 * Symplectic manifold
 */
typedef struct {
    size_t dimension;                // Must be even
    struct geometric_tensor* symplectic_form;  // ω (closed, non-degenerate 2-form)
    double* darboux_coords;          // Darboux coordinates (q_i, p_i)
    double* poisson_bracket;         // {f, g} structure
    bool is_exact;                   // ω = dα for some 1-form α
} SymplecticManifold;

/**
 * Kahler manifold
 */
typedef struct {
    size_t complex_dimension;
    struct geometric_tensor* kahler_metric;   // Hermitian metric
    struct geometric_tensor* kahler_form;     // ω = -i g_{j\bar{k}} dz^j ∧ dz^\bar{k}
    qgt_complex_t* holomorphic_coords;       // z^1, ..., z^n
    double* kahler_potential;                 // K such that g = ∂∂̄K
    double ricci_scalar;
    bool is_kahler_einstein;
} KahlerManifold;

// =============================================================================
// Fiber Bundle Structures
// =============================================================================

/**
 * Fiber
 */
typedef struct {
    size_t dimension;
    void* structure_group;           // Lie group acting on fiber
    double* typical_fiber;           // Standard fiber
    size_t fiber_size;
} Fiber;

/**
 * Principal bundle
 */
typedef struct {
    RiemannianManifold* base_space;
    Fiber* fiber;
    size_t structure_group_dim;
    qgt_complex_t* connection_form;          // Lie algebra valued 1-form
    qgt_complex_t* curvature_form;           // Field strength F = dA + A∧A
    size_t num_local_sections;
    bool is_trivial;
} PrincipalBundle;

/**
 * Vector bundle
 */
typedef struct {
    RiemannianManifold* base_space;
    size_t fiber_dimension;
    size_t rank;
    struct geometric_tensor** local_frames;
    size_t num_frames;
    qgt_complex_t* transition_functions;
    ConnectionType connection_type;
    qgt_complex_t* connection;               // Connection 1-form
    qgt_complex_t* curvature;                // Curvature 2-form
} VectorBundle;

/**
 * Line bundle
 */
typedef struct {
    RiemannianManifold* base_space;
    int first_chern_class;           // c_1 ∈ Z (for complex line bundles)
    qgt_complex_t* hermitian_metric;
    qgt_complex_t* connection;
    qgt_complex_t* curvature;
    bool is_holomorphic;
} LineBundle;

// =============================================================================
// Connection and Curvature Structures
// =============================================================================

/**
 * Affine connection
 */
typedef struct {
    ConnectionType type;
    double* christoffel_symbols;     // Γ^k_{ij}
    size_t dimension;
    double* torsion_tensor;          // T^k_{ij} = Γ^k_{ij} - Γ^k_{ji}
    bool is_metric_compatible;
    bool is_torsion_free;
} AffineConnection;

/**
 * Curvature data
 */
typedef struct {
    CurvatureType type;
    struct geometric_tensor* riemann;        // R^l_{kij}
    struct geometric_tensor* ricci;          // R_{ij} = R^k_{ikj}
    double scalar_curvature;                 // R = g^{ij} R_{ij}
    struct geometric_tensor* weyl;           // Weyl tensor (conformal)
    struct geometric_tensor* einstein;       // G_{ij} = R_{ij} - (R/2)g_{ij}
    double* sectional;                       // K(σ) for 2-planes
    size_t num_sectional;
} CurvatureData;

/**
 * Yang-Mills connection
 */
typedef struct {
    VectorBundle* bundle;
    qgt_complex_t* gauge_potential;         // A = A_μ^a T_a dx^μ
    qgt_complex_t* field_strength;          // F = dA + A∧A
    qgt_complex_t* covariant_derivative;
    size_t gauge_group_dim;
    double yang_mills_action;
    bool is_anti_self_dual;
    double instanton_number;
} YangMillsConnection;

// =============================================================================
// Characteristic Classes
// =============================================================================

/**
 * Chern classes
 */
typedef struct {
    qgt_complex_t** chern_classes;  // c_1, c_2, ..., c_n
    size_t rank;                     // Number of Chern classes
    qgt_complex_t* total_chern;     // c(E) = 1 + c_1 + c_2 + ...
    qgt_complex_t* chern_character; // ch(E) = rank + c_1 + (c_1^2-2c_2)/2 + ...
} ChernClasses;

/**
 * Pontryagin classes
 */
typedef struct {
    double** pontryagin_classes;     // p_1, p_2, ..., p_k
    size_t num_classes;
    double* total_pontryagin;
    double signature;                // Signature from Hirzebruch formula
} PontryaginClasses;

/**
 * Euler class
 */
typedef struct {
    double* euler_class;             // e(E) for oriented real vector bundle
    size_t dimension;
    double euler_characteristic;     // χ(M) = ∫_M e(TM)
} EulerClass;

/**
 * Characteristic class data
 */
typedef struct {
    ChernClasses* chern;
    PontryaginClasses* pontryagin;
    EulerClass* euler;
    double todd_genus;               // Td(M) for complex manifolds
    double a_hat_genus;              // Â(M) for spin manifolds
} CharacteristicClasses;

// =============================================================================
// Manifold Operations
// =============================================================================

/**
 * Create Riemannian manifold
 */
int riemannian_manifold_create(
    RiemannianManifold** manifold,
    ManifoldType type,
    size_t dimension
);

/**
 * Destroy Riemannian manifold
 */
void riemannian_manifold_destroy(RiemannianManifold* manifold);

/**
 * Set metric tensor
 */
int riemannian_set_metric(
    RiemannianManifold* manifold,
    struct geometric_tensor* metric
);

/**
 * Compute Christoffel symbols
 */
int riemannian_compute_christoffel(
    RiemannianManifold* manifold
);

/**
 * Compute geodesic
 */
int riemannian_geodesic(
    RiemannianManifold* manifold,
    double* start_point,
    double* initial_velocity,
    double parameter_range,
    size_t num_points,
    double** geodesic_out
);

/**
 * Compute geodesic distance
 */
int riemannian_distance(
    RiemannianManifold* manifold,
    double* point1,
    double* point2,
    double* distance_out
);

/**
 * Parallel transport along curve
 */
int riemannian_parallel_transport(
    RiemannianManifold* manifold,
    double* vector,
    double* curve,
    size_t curve_points,
    double* transported_out
);

/**
 * Compute exponential map
 */
int riemannian_exp_map(
    RiemannianManifold* manifold,
    double* base_point,
    double* tangent_vector,
    double* result_point
);

/**
 * Compute logarithm map
 */
int riemannian_log_map(
    RiemannianManifold* manifold,
    double* base_point,
    double* target_point,
    double* tangent_vector_out
);

// =============================================================================
// Curvature Operations
// =============================================================================

/**
 * Compute curvature tensors
 */
int compute_curvature(
    RiemannianManifold* manifold,
    CurvatureData** curvature_out
);

/**
 * Destroy curvature data
 */
void curvature_data_destroy(CurvatureData* curvature);

/**
 * Compute Riemann tensor
 */
int compute_riemann_tensor(
    RiemannianManifold* manifold,
    struct geometric_tensor** riemann_out
);

/**
 * Compute Ricci tensor
 */
int compute_ricci_tensor(
    RiemannianManifold* manifold,
    struct geometric_tensor** ricci_out
);

/**
 * Compute scalar curvature
 */
int compute_scalar_curvature(
    RiemannianManifold* manifold,
    double* scalar_out
);

/**
 * Compute sectional curvature
 */
int compute_sectional_curvature(
    RiemannianManifold* manifold,
    double* tangent_vector1,
    double* tangent_vector2,
    double* point,
    double* sectional_out
);

/**
 * Compute Gaussian curvature (2D)
 */
int compute_gaussian_curvature(
    RiemannianManifold* manifold,
    double* point,
    double* gaussian_out
);

// =============================================================================
// Bundle Operations
// =============================================================================

/**
 * Create principal bundle
 */
int principal_bundle_create(
    PrincipalBundle** bundle,
    RiemannianManifold* base,
    size_t structure_group_dim
);

/**
 * Destroy principal bundle
 */
void principal_bundle_destroy(PrincipalBundle* bundle);

/**
 * Create vector bundle
 */
int vector_bundle_create(
    VectorBundle** bundle,
    RiemannianManifold* base,
    size_t rank
);

/**
 * Destroy vector bundle
 */
void vector_bundle_destroy(VectorBundle* bundle);

/**
 * Set connection on bundle
 */
int bundle_set_connection(
    VectorBundle* bundle,
    qgt_complex_t* connection_form
);

/**
 * Compute curvature of bundle connection
 */
int bundle_compute_curvature(
    VectorBundle* bundle
);

/**
 * Covariant derivative of section
 */
int bundle_covariant_derivative(
    VectorBundle* bundle,
    qgt_complex_t* section,
    double* direction,
    qgt_complex_t* derivative_out
);

/**
 * Holonomy around loop
 */
int bundle_holonomy(
    VectorBundle* bundle,
    double* loop,
    size_t loop_points,
    qgt_complex_t* holonomy_out
);

// =============================================================================
// Characteristic Class Operations
// =============================================================================

/**
 * Compute Chern classes
 */
int compute_chern_classes(
    VectorBundle* bundle,
    ChernClasses** chern_out
);

/**
 * Destroy Chern classes
 */
void chern_classes_destroy(ChernClasses* chern);

/**
 * Compute Pontryagin classes
 */
int compute_pontryagin_classes(
    VectorBundle* bundle,
    PontryaginClasses** pontryagin_out
);

/**
 * Destroy Pontryagin classes
 */
void pontryagin_classes_destroy(PontryaginClasses* pontryagin);

/**
 * Compute Euler characteristic
 */
int compute_euler_characteristic(
    RiemannianManifold* manifold,
    double* euler_out
);

/**
 * Compute all characteristic classes
 */
int compute_characteristic_classes(
    VectorBundle* bundle,
    CharacteristicClasses** classes_out
);

/**
 * Destroy characteristic classes
 */
void characteristic_classes_destroy(CharacteristicClasses* classes);

// =============================================================================
// Symplectic and Kahler Operations
// =============================================================================

/**
 * Create symplectic manifold
 */
int symplectic_manifold_create(
    SymplecticManifold** manifold,
    size_t dimension
);

/**
 * Destroy symplectic manifold
 */
void symplectic_manifold_destroy(SymplecticManifold* manifold);

/**
 * Set symplectic form
 */
int symplectic_set_form(
    SymplecticManifold* manifold,
    struct geometric_tensor* omega
);

/**
 * Compute Poisson bracket
 */
int poisson_bracket(
    SymplecticManifold* manifold,
    double* function_f,
    double* function_g,
    double* bracket_out
);

/**
 * Compute Hamiltonian vector field
 */
int hamiltonian_vector_field(
    SymplecticManifold* manifold,
    double* hamiltonian,
    double* vector_field_out
);

/**
 * Create Kahler manifold
 */
int kahler_manifold_create(
    KahlerManifold** manifold,
    size_t complex_dimension
);

/**
 * Destroy Kahler manifold
 */
void kahler_manifold_destroy(KahlerManifold* manifold);

/**
 * Set Kahler potential
 */
int kahler_set_potential(
    KahlerManifold* manifold,
    double* potential
);

/**
 * Compute Kahler metric from potential
 */
int kahler_compute_metric(
    KahlerManifold* manifold
);

// =============================================================================
// Quantum Geometry Operations
// =============================================================================

/**
 * Compute quantum geometric tensor
 */
int quantum_geometric_tensor(
    struct quantum_state** parameter_states,
    size_t num_parameters,
    qgt_complex_t* qgt_out
);

/**
 * Compute Berry curvature
 */
int berry_curvature(
    struct quantum_state** parameter_states,
    size_t num_parameters,
    double* curvature_out
);

/**
 * Compute quantum metric
 */
int quantum_metric(
    struct quantum_state** parameter_states,
    size_t num_parameters,
    double* metric_out
);

/**
 * Compute Fubini-Study metric
 */
int fubini_study_metric(
    struct quantum_state* state1,
    struct quantum_state* state2,
    double* distance_out
);

/**
 * Compute Bures metric
 */
int bures_metric(
    qgt_complex_t* rho1,
    qgt_complex_t* rho2,
    size_t dimension,
    double* distance_out
);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Print manifold info
 */
void manifold_print_info(RiemannianManifold* manifold);

/**
 * Print bundle info
 */
void bundle_print_info(VectorBundle* bundle);

/**
 * Validate manifold structure
 */
bool manifold_validate(RiemannianManifold* manifold);

/**
 * Validate bundle structure
 */
bool bundle_validate(VectorBundle* bundle);

/**
 * Export manifold to mesh format
 */
int manifold_export_mesh(
    RiemannianManifold* manifold,
    const char* filename,
    const char* format
);

#ifdef __cplusplus
}
#endif

#endif // ADVANCED_GEOMETRY_OPERATIONS_H
