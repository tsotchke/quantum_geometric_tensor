/**
 * @file quantum_geometric_projections.h
 * @brief Geometric Projections for Quantum State Spaces
 *
 * Provides geometric projection operations including:
 * - Projective Hilbert space projections
 * - Subspace projections and decompositions
 * - Quantum state space embeddings
 * - Grassmannian projections
 * - Flag manifold projections
 * - Symplectic projections
 *
 * Part of the QGTL Physics Framework.
 */

#ifndef QUANTUM_GEOMETRIC_PROJECTIONS_H
#define QUANTUM_GEOMETRIC_PROJECTIONS_H

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

#define PROJECTION_MAX_DIMENSIONS 1024
#define PROJECTION_MAX_SUBSPACES 64
#define PROJECTION_MAX_NAME_LENGTH 128
#define PROJECTION_DEFAULT_TOLERANCE 1e-12

// ============================================================================
// Enumerations
// ============================================================================

/**
 * Projection types
 */
typedef enum {
    PROJECTION_ORTHOGONAL,            // Orthogonal projection
    PROJECTION_OBLIQUE,               // Oblique projection
    PROJECTION_HERMITIAN,             // Hermitian projection
    PROJECTION_SYMPLECTIC,            // Symplectic projection
    PROJECTION_STEREOGRAPHIC,         // Stereographic projection
    PROJECTION_HOPF,                  // Hopf fibration projection
    PROJECTION_PROJECTIVE,            // Projective (CP^n) projection
    PROJECTION_GRASSMANNIAN,          // Grassmannian projection
    PROJECTION_FLAG                   // Flag manifold projection
} projection_type_t;

/**
 * Subspace types
 */
typedef enum {
    SUBSPACE_LINEAR,                  // Linear subspace
    SUBSPACE_AFFINE,                  // Affine subspace
    SUBSPACE_PROJECTIVE,              // Projective subspace
    SUBSPACE_LAGRANGIAN,              // Lagrangian subspace (symplectic)
    SUBSPACE_ISOTROPIC,               // Isotropic subspace
    SUBSPACE_COISOTROPIC,             // Coisotropic subspace
    SUBSPACE_EIGENSPACE,              // Eigenspace of operator
    SUBSPACE_KERNEL,                  // Kernel/null space
    SUBSPACE_IMAGE                    // Image/range
} subspace_type_t;

/**
 * Embedding types for quantum states
 */
typedef enum {
    EMBEDDING_SEGRE,                  // Segre embedding (product states)
    EMBEDDING_VERONESE,               // Veronese embedding
    EMBEDDING_PLUCKER,                // Plücker embedding (Grassmannian)
    EMBEDDING_KODAIRA,                // Kodaira embedding
    EMBEDDING_CANONICAL,              // Canonical embedding
    EMBEDDING_ANTICANONICAL           // Anticanonical embedding
} embedding_type_t;

/**
 * Metric types for projections
 */
typedef enum {
    PROJECTION_METRIC_FUBINI_STUDY,   // Fubini-Study metric
    PROJECTION_METRIC_BURES,          // Bures metric
    PROJECTION_METRIC_TRACE,          // Trace distance metric
    PROJECTION_METRIC_HILBERT_SCHMIDT,// Hilbert-Schmidt metric
    PROJECTION_METRIC_OPERATOR        // Operator norm metric
} projection_metric_t;

/**
 * Decomposition types
 */
typedef enum {
    DECOMP_SCHMIDT,                   // Schmidt decomposition
    DECOMP_POLAR,                     // Polar decomposition
    DECOMP_CARTAN,                    // Cartan decomposition
    DECOMP_IWASAWA,                   // Iwasawa decomposition
    DECOMP_BRUHAT,                    // Bruhat decomposition
    DECOMP_KAK                        // KAK decomposition
} decomposition_type_t;

// ============================================================================
// Data Structures
// ============================================================================

/**
 * Subspace descriptor
 */
typedef struct {
    subspace_type_t type;             // Subspace type
    size_t ambient_dim;               // Ambient space dimension
    size_t subspace_dim;              // Subspace dimension
    ComplexDouble* basis;             // Orthonormal basis vectors
    char name[PROJECTION_MAX_NAME_LENGTH];  // Subspace name
    bool is_orthonormal;              // Basis is orthonormal
} projection_subspace_t;

/**
 * Projector operator (P such that P² = P)
 */
typedef struct {
    projection_type_t type;           // Projection type
    size_t dim;                       // Operator dimension
    ComplexDouble* matrix;            // Projector matrix
    size_t rank;                      // Rank of projector
    projection_subspace_t* image;     // Image subspace
    projection_subspace_t* kernel;    // Kernel subspace
    bool is_hermitian;                // P = P†
    bool is_orthogonal;               // P² = P, PP† = P†P
} projection_operator_t;

/**
 * Projected state result
 */
typedef struct {
    ComplexDouble* projected_state;   // Resulting state
    size_t dim;                       // State dimension
    double projection_norm;           // ||P|ψ⟩||
    double complement_norm;           // ||(I-P)|ψ⟩||
    ComplexDouble overlap;            // ⟨ψ|P|ψ⟩
    bool in_subspace;                 // State fully in subspace
} projection_result_t;

/**
 * Grassmannian point (k-dimensional subspace of n-dim space)
 */
typedef struct {
    size_t k;                         // Subspace dimension
    size_t n;                         // Ambient dimension
    ComplexDouble* basis;             // Representative basis (n x k)
    ComplexDouble* plucker_coords;    // Plücker coordinates
    size_t num_plucker;               // Number of Plücker coords
} grassmannian_point_t;

/**
 * Flag point (nested sequence of subspaces)
 */
typedef struct {
    size_t* dimensions;               // Subspace dimensions (increasing)
    size_t num_subspaces;             // Number of subspaces
    size_t ambient_dim;               // Ambient space dimension
    ComplexDouble** bases;            // Bases for each subspace
    bool is_complete;                 // Complete flag (all dims 1..n-1)
} flag_point_t;

/**
 * Hopf fibration data (S³ -> S²)
 */
typedef struct {
    ComplexDouble fiber_point[2];     // Point on S³ ⊂ C²
    double base_point[3];             // Point on S² (Bloch sphere)
    double phase;                     // U(1) phase on fiber
} hopf_fiber_t;

/**
 * Decomposition result
 */
typedef struct {
    decomposition_type_t type;        // Decomposition type
    ComplexDouble** factors;          // Decomposition factors
    size_t num_factors;               // Number of factors
    double* singular_values;          // Singular/Schmidt values
    size_t num_values;                // Number of values
    size_t schmidt_rank;              // Schmidt rank (for bipartite)
} decomposition_result_t;

/**
 * Projection statistics
 */
typedef struct {
    uint64_t total_projections;       // Total projections computed
    uint64_t projections_by_type[9];  // By projection type
    uint64_t subspace_ops;            // Subspace operations
    uint64_t decompositions;          // Decompositions performed
    double total_time_ms;             // Total computation time
    double avg_error;                 // Average numerical error
} projection_stats_t;

/**
 * Configuration options
 */
typedef struct {
    double numerical_tolerance;       // Numerical tolerance
    projection_metric_t metric;       // Default metric
    bool enable_caching;              // Cache projectors
    bool check_hermiticity;           // Verify Hermitian property
    bool use_sparse;                  // Use sparse representations
    bool parallel_enabled;            // Enable parallel computation
    size_t cache_size;                // Projector cache size
} projection_config_t;

/**
 * Opaque projection engine handle
 */
typedef struct projection_engine projection_engine_t;

// ============================================================================
// Initialization and Configuration
// ============================================================================

/**
 * Create projection engine
 */
projection_engine_t* projection_engine_create(void);

/**
 * Create with configuration
 */
projection_engine_t* projection_engine_create_with_config(
    const projection_config_t* config);

/**
 * Get default configuration
 */
projection_config_t projection_default_config(void);

/**
 * Destroy projection engine
 */
void projection_engine_destroy(projection_engine_t* engine);

/**
 * Reset engine state
 */
bool projection_engine_reset(projection_engine_t* engine);

// ============================================================================
// Subspace Operations
// ============================================================================

/**
 * Create subspace from basis vectors
 */
projection_subspace_t* projection_subspace_create(
    projection_engine_t* engine,
    const ComplexDouble* basis_vectors,
    size_t ambient_dim,
    size_t subspace_dim);

/**
 * Create subspace from span of vectors
 */
projection_subspace_t* projection_subspace_from_span(
    projection_engine_t* engine,
    const ComplexDouble* vectors,
    size_t ambient_dim,
    size_t num_vectors);

/**
 * Create eigenspace of operator
 */
projection_subspace_t* projection_eigenspace(
    projection_engine_t* engine,
    const ComplexDouble* op,
    size_t dim,
    ComplexDouble eigenvalue,
    double tolerance);

/**
 * Create kernel of operator
 */
projection_subspace_t* projection_kernel(
    projection_engine_t* engine,
    const ComplexDouble* op,
    size_t rows,
    size_t cols);

/**
 * Create image of operator
 */
projection_subspace_t* projection_image(
    projection_engine_t* engine,
    const ComplexDouble* op,
    size_t rows,
    size_t cols);

/**
 * Compute intersection of subspaces
 */
projection_subspace_t* projection_subspace_intersection(
    projection_engine_t* engine,
    const projection_subspace_t* s1,
    const projection_subspace_t* s2);

/**
 * Compute sum of subspaces
 */
projection_subspace_t* projection_subspace_sum(
    projection_engine_t* engine,
    const projection_subspace_t* s1,
    const projection_subspace_t* s2);

/**
 * Compute orthogonal complement
 */
projection_subspace_t* projection_orthogonal_complement(
    projection_engine_t* engine,
    const projection_subspace_t* subspace);

/**
 * Check if state is in subspace
 */
bool projection_is_in_subspace(
    projection_engine_t* engine,
    const ComplexDouble* state,
    const projection_subspace_t* subspace,
    double tolerance);

/**
 * Destroy subspace
 */
void projection_subspace_destroy(projection_subspace_t* subspace);

// ============================================================================
// Projector Operations
// ============================================================================

/**
 * Create orthogonal projector onto subspace
 */
projection_operator_t* projection_create_orthogonal(
    projection_engine_t* engine,
    const projection_subspace_t* subspace);

/**
 * Create projector from state (|ψ⟩⟨ψ|)
 */
projection_operator_t* projection_create_from_state(
    projection_engine_t* engine,
    const ComplexDouble* state,
    size_t dim);

/**
 * Create projector from matrix
 */
projection_operator_t* projection_create_from_matrix(
    projection_engine_t* engine,
    const ComplexDouble* matrix,
    size_t dim);

/**
 * Create oblique projector
 */
projection_operator_t* projection_create_oblique(
    projection_engine_t* engine,
    const projection_subspace_t* range,
    const projection_subspace_t* kernel);

/**
 * Apply projector to state
 */
projection_result_t* projection_apply(
    projection_engine_t* engine,
    const projection_operator_t* projector,
    const ComplexDouble* state);

/**
 * Apply projector to density matrix
 */
bool projection_apply_to_density(
    projection_engine_t* engine,
    const projection_operator_t* projector,
    const ComplexDouble* rho,
    ComplexDouble* result,
    size_t dim);

/**
 * Compose projectors (P₁ ∘ P₂)
 */
projection_operator_t* projection_compose(
    projection_engine_t* engine,
    const projection_operator_t* p1,
    const projection_operator_t* p2);

/**
 * Check if projectors commute
 */
bool projection_commutes(
    projection_engine_t* engine,
    const projection_operator_t* p1,
    const projection_operator_t* p2,
    double tolerance);

/**
 * Destroy projector
 */
void projection_operator_destroy(projection_operator_t* projector);

/**
 * Destroy projection result
 */
void projection_result_destroy(projection_result_t* result);

// ============================================================================
// Projective Space Operations
// ============================================================================

/**
 * Project to CP^n (complex projective space)
 */
bool projection_to_cpn(
    projection_engine_t* engine,
    const ComplexDouble* state,
    size_t dim,
    ComplexDouble* projected);

/**
 * Compute Fubini-Study distance
 */
double projection_fubini_study_distance(
    projection_engine_t* engine,
    const ComplexDouble* psi,
    const ComplexDouble* phi,
    size_t dim);

/**
 * Compute Fubini-Study metric tensor
 */
bool projection_fubini_study_metric(
    projection_engine_t* engine,
    const ComplexDouble* state,
    size_t dim,
    const ComplexDouble* tangent_vectors,
    size_t num_tangent,
    double* metric);

/**
 * Stereographic projection from S^n to R^n
 */
bool projection_stereographic(
    projection_engine_t* engine,
    const double* sphere_point,
    size_t dim,
    double* plane_point);

/**
 * Inverse stereographic projection
 */
bool projection_stereographic_inverse(
    projection_engine_t* engine,
    const double* plane_point,
    size_t dim,
    double* sphere_point);

// ============================================================================
// Grassmannian Operations
// ============================================================================

/**
 * Create Grassmannian point from basis
 */
grassmannian_point_t* projection_grassmannian_create(
    projection_engine_t* engine,
    const ComplexDouble* basis,
    size_t k,
    size_t n);

/**
 * Compute Plücker embedding
 */
bool projection_plucker_embed(
    projection_engine_t* engine,
    const grassmannian_point_t* point,
    ComplexDouble* plucker);

/**
 * Grassmannian distance (principal angles)
 */
double projection_grassmannian_distance(
    projection_engine_t* engine,
    const grassmannian_point_t* p1,
    const grassmannian_point_t* p2);

/**
 * Principal angles between subspaces
 */
bool projection_principal_angles(
    projection_engine_t* engine,
    const grassmannian_point_t* p1,
    const grassmannian_point_t* p2,
    double* angles,
    size_t* num_angles);

/**
 * Geodesic on Grassmannian
 */
grassmannian_point_t* projection_grassmannian_geodesic(
    projection_engine_t* engine,
    const grassmannian_point_t* start,
    const grassmannian_point_t* end,
    double t);

/**
 * Destroy Grassmannian point
 */
void projection_grassmannian_destroy(grassmannian_point_t* point);

// ============================================================================
// Flag Manifold Operations
// ============================================================================

/**
 * Create flag point
 */
flag_point_t* projection_flag_create(
    projection_engine_t* engine,
    const size_t* dimensions,
    size_t num_subspaces,
    size_t ambient_dim,
    const ComplexDouble* vectors);

/**
 * Create complete flag
 */
flag_point_t* projection_flag_complete(
    projection_engine_t* engine,
    const ComplexDouble* full_basis,
    size_t dim);

/**
 * Project to flag manifold
 */
flag_point_t* projection_to_flag(
    projection_engine_t* engine,
    const ComplexDouble* unitary,
    size_t dim,
    const size_t* flag_type,
    size_t num_subspaces);

/**
 * Destroy flag point
 */
void projection_flag_destroy(flag_point_t* flag);

// ============================================================================
// Hopf Fibration
// ============================================================================

/**
 * Compute Hopf projection S³ -> S² (qubit to Bloch sphere)
 */
bool projection_hopf(
    projection_engine_t* engine,
    const ComplexDouble* qubit,
    double* bloch);

/**
 * Compute Hopf fiber at Bloch point
 */
hopf_fiber_t* projection_hopf_fiber(
    projection_engine_t* engine,
    const double* bloch);

/**
 * Lift from Bloch sphere to S³ with phase
 */
bool projection_hopf_lift(
    projection_engine_t* engine,
    const double* bloch,
    double phase,
    ComplexDouble* qubit);

/**
 * Destroy Hopf fiber
 */
void projection_hopf_destroy(hopf_fiber_t* fiber);

// ============================================================================
// Decompositions
// ============================================================================

/**
 * Compute Schmidt decomposition
 */
decomposition_result_t* projection_schmidt_decompose(
    projection_engine_t* engine,
    const ComplexDouble* state,
    size_t dim_a,
    size_t dim_b);

/**
 * Compute polar decomposition A = UP
 */
decomposition_result_t* projection_polar_decompose(
    projection_engine_t* engine,
    const ComplexDouble* matrix,
    size_t rows,
    size_t cols);

/**
 * Compute Cartan decomposition (Lie group)
 */
decomposition_result_t* projection_cartan_decompose(
    projection_engine_t* engine,
    const ComplexDouble* unitary,
    size_t dim);

/**
 * Compute KAK decomposition
 */
decomposition_result_t* projection_kak_decompose(
    projection_engine_t* engine,
    const ComplexDouble* two_qubit,
    size_t dim);

/**
 * Destroy decomposition result
 */
void projection_decomposition_destroy(decomposition_result_t* result);

// ============================================================================
// Quantum-Specific Projections
// ============================================================================

/**
 * Project onto computational basis state
 */
bool projection_computational_basis(
    projection_engine_t* engine,
    const ComplexDouble* state,
    size_t dim,
    size_t basis_index,
    ComplexDouble* result);

/**
 * Project onto symmetric subspace
 */
projection_operator_t* projection_symmetric_subspace(
    projection_engine_t* engine,
    size_t num_qubits);

/**
 * Project onto antisymmetric subspace
 */
projection_operator_t* projection_antisymmetric_subspace(
    projection_engine_t* engine,
    size_t num_qubits);

/**
 * Project onto code space (QEC)
 */
projection_operator_t* projection_code_space(
    projection_engine_t* engine,
    const ComplexDouble** stabilizers,
    size_t num_stabilizers,
    size_t dim);

/**
 * Project density matrix to physical states
 */
bool projection_to_physical(
    projection_engine_t* engine,
    const ComplexDouble* rho,
    size_t dim,
    ComplexDouble* physical_rho);

// ============================================================================
// Statistics and Reporting
// ============================================================================

/**
 * Get statistics
 */
bool projection_get_stats(projection_engine_t* engine,
                          projection_stats_t* stats);

/**
 * Reset statistics
 */
void projection_reset_stats(projection_engine_t* engine);

/**
 * Generate report
 */
char* projection_generate_report(projection_engine_t* engine);

/**
 * Export to JSON
 */
char* projection_export_json(projection_engine_t* engine);

/**
 * Export to file
 */
bool projection_export_to_file(projection_engine_t* engine,
                               const char* filename);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Get projection type name
 */
const char* projection_type_name(projection_type_t type);

/**
 * Get subspace type name
 */
const char* projection_subspace_type_name(subspace_type_t type);

/**
 * Get embedding type name
 */
const char* projection_embedding_name(embedding_type_t type);

/**
 * Get metric name
 */
const char* projection_metric_name(projection_metric_t metric);

/**
 * Get decomposition type name
 */
const char* projection_decomposition_name(decomposition_type_t type);

/**
 * Free allocated string
 */
void projection_free_string(char* str);

/**
 * Get last error message
 */
const char* projection_get_last_error(projection_engine_t* engine);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_GEOMETRIC_PROJECTIONS_H
