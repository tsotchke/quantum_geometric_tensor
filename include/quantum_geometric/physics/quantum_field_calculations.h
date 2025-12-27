/**
 * @file quantum_field_calculations.h
 * @brief Quantum field theory calculations on spacetime lattice
 *
 * Implements quantum field equations with:
 * - Lorentz-covariant tensor transformations
 * - Gauge field coupling and covariant derivatives
 * - Klein-Gordon, Dirac, and Yang-Mills equations
 * - Energy and momentum calculations
 * - Field propagation and Green's functions
 *
 * The spacetime is discretized on a 4D lattice with coordinates (t, x, y, z).
 * Fields are represented as complex-valued tensors with internal indices.
 */

#ifndef QUANTUM_FIELD_CALCULATIONS_H
#define QUANTUM_FIELD_CALCULATIONS_H

#include <stddef.h>
#include <stdbool.h>
#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================

/** Number of spacetime dimensions (Minkowski space) */
#define QG_SPACETIME_DIMS 4

/** Maximum rank for field tensors */
#define QG_MAX_TENSOR_RANK 8

/** Maximum number of internal components */
#define QG_MAX_FIELD_COMPONENTS 64

/** Numerical tolerance for field calculations */
#define QG_FIELD_EPSILON 1e-14

// ============================================================================
// Type Definitions
// ============================================================================

/**
 * @brief General tensor type for field theory calculations
 *
 * Tensors on spacetime lattice with arbitrary rank and internal structure.
 * Uses complex double precision for full quantum mechanical treatment.
 */
typedef struct {
    size_t rank;                        /**< Number of dimensions/indices */
    size_t dims[QG_MAX_TENSOR_RANK];    /**< Size of each dimension */
    size_t total_size;                  /**< Total number of elements */
    complex double* data;               /**< Complex tensor data */
    bool is_allocated;                  /**< Whether data is owned */
} Tensor;

/**
 * @brief Spacetime metric tensor
 *
 * Supports both flat Minkowski and curved spacetimes.
 * Stored as 4x4 matrix in row-major order.
 */
typedef struct {
    complex double components[QG_SPACETIME_DIMS * QG_SPACETIME_DIMS];
    bool is_minkowski;                  /**< True for flat spacetime */
    double curvature_scale;             /**< Characteristic curvature scale */
} SpacetimeMetric;

/**
 * @brief Quantum field on spacetime lattice
 *
 * Represents a general quantum field with internal degrees of freedom,
 * gauge coupling, and self-interaction terms.
 */
typedef struct {
    // Field data
    Tensor* field_tensor;               /**< Field configuration φ(x) */
    Tensor* conjugate_momentum;         /**< Canonical momentum π(x) */
    Tensor* gauge_field;                /**< Gauge field A_μ(x) (optional) */

    // Spacetime structure
    complex double metric[QG_SPACETIME_DIMS * QG_SPACETIME_DIMS]; /**< Metric tensor */
    double lattice_spacing[QG_SPACETIME_DIMS];                     /**< Lattice spacings */

    // Field parameters
    double mass;                        /**< Field mass */
    double coupling;                    /**< Self-coupling constant (λφ⁴) */
    double field_strength;              /**< Gauge coupling strength (e) */
    double spin;                        /**< Field spin (0, 1/2, 1, etc.) */

    // Internal structure
    size_t num_components;              /**< Number of internal components */
    size_t gauge_group_dim;             /**< Dimension of gauge group */

    // Boundary conditions
    bool periodic_bc[QG_SPACETIME_DIMS]; /**< Periodic boundary conditions */

    // State flags
    bool is_initialized;                /**< Initialization flag */
    bool has_gauge_field;               /**< Whether gauge field is present */
} QuantumField;

/**
 * @brief Configuration for field calculations
 */
typedef struct {
    // Numerical parameters
    double time_step;                   /**< Time evolution step */
    double tolerance;                   /**< Convergence tolerance */
    size_t max_iterations;              /**< Maximum iterations */

    // Physical parameters
    double cutoff_energy;               /**< UV energy cutoff */
    double renormalization_scale;       /**< Renormalization scale μ */

    // Method selection
    bool use_symplectic;                /**< Use symplectic integrator */
    bool use_parallel;                  /**< Enable OpenMP parallelization */
    size_t num_threads;                 /**< Number of threads (0 = auto) */
} FieldConfig;

// ============================================================================
// Tensor Operations
// ============================================================================

/**
 * @brief Allocate tensor with given dimensions
 * @param tensor Output tensor
 * @param rank Number of dimensions
 * @param dims Array of dimension sizes
 * @return true on success
 */
bool tensor_allocate(Tensor* tensor, size_t rank, const size_t* dims);

/**
 * @brief Free tensor data
 * @param tensor Tensor to free
 */
void tensor_free(Tensor* tensor);

/**
 * @brief Transform field components at a spacetime point
 *
 * Applies a transformation matrix to the internal indices of the field
 * at a specific spacetime location.
 *
 * @param field Field tensor to transform
 * @param transformation Transformation matrix
 * @param t Time coordinate
 * @param x X coordinate
 * @param y Y coordinate
 * @param z Z coordinate
 */
void transform_field_components(
    Tensor* field,
    const Tensor* transformation,
    size_t t,
    size_t x,
    size_t y,
    size_t z);

/**
 * @brief Transform gauge field under gauge transformation
 *
 * Implements gauge transformation A_μ → U A_μ U† + (i/g) U ∂_μ U†
 *
 * @param gauge_field Gauge field to transform
 * @param transformation Gauge transformation matrix
 */
void transform_gauge_field(
    Tensor* gauge_field,
    const Tensor* transformation);

/**
 * @brief Transform a single generator in the gauge field
 *
 * @param gauge_field Gauge field tensor
 * @param transformation Transformation matrix
 * @param t Time coordinate
 * @param x X coordinate
 * @param y Y coordinate
 * @param z Z coordinate
 * @param g Generator index
 */
void transform_generator(
    Tensor* gauge_field,
    const Tensor* transformation,
    size_t t, size_t x, size_t y, size_t z, size_t g);

// ============================================================================
// Field Equations
// ============================================================================

/**
 * @brief Calculate kinetic terms of field equations
 *
 * Computes ∂_μ∂^μ φ (contracted with metric).
 *
 * @param field Quantum field
 * @param equations Output tensor for equation values
 */
void calculate_kinetic_terms(
    const QuantumField* field,
    Tensor* equations);

/**
 * @brief Add mass terms to field equations
 *
 * Adds m² φ contribution.
 *
 * @param field Quantum field
 * @param equations Tensor to add mass terms to
 */
void add_mass_terms(
    const QuantumField* field,
    Tensor* equations);

/**
 * @brief Add interaction terms to field equations
 *
 * Adds λ|φ|² φ contribution (quartic self-interaction).
 *
 * @param field Quantum field
 * @param equations Tensor to add interaction terms to
 */
void add_interaction_terms(
    const QuantumField* field,
    Tensor* equations);

/**
 * @brief Add gauge coupling to field equations
 *
 * Adds gauge covariant derivative terms.
 *
 * @param field Quantum field with gauge field
 * @param equations Tensor to add gauge terms to
 */
void add_gauge_coupling(
    const QuantumField* field,
    Tensor* equations);

// ============================================================================
// Derivative Calculations
// ============================================================================

/**
 * @brief Calculate partial derivatives of field
 *
 * Uses central finite differences on the lattice.
 *
 * @param field_tensor Field tensor
 * @param t Time coordinate
 * @param x X coordinate
 * @param y Y coordinate
 * @param z Z coordinate
 * @return Array of derivatives (caller must free)
 */
complex double* calculate_derivatives(
    const Tensor* field_tensor,
    size_t t, size_t x, size_t y, size_t z);

/**
 * @brief Calculate covariant derivatives including gauge field
 *
 * Implements D_μ φ = ∂_μ φ - ig A_μ φ
 *
 * @param field Quantum field with gauge field
 * @param t Time coordinate
 * @param x X coordinate
 * @param y Y coordinate
 * @param z Z coordinate
 * @return Array of covariant derivatives (caller must free)
 */
complex double* calculate_covariant_derivatives(
    const QuantumField* field,
    size_t t, size_t x, size_t y, size_t z);

// ============================================================================
// Energy and Momentum
// ============================================================================

/**
 * @brief Calculate kinetic energy of field
 *
 * Computes ∫ |π|² d³x
 *
 * @param field Quantum field
 * @return Kinetic energy
 */
double calculate_kinetic_energy(const QuantumField* field);

/**
 * @brief Calculate potential energy of field
 *
 * Computes ∫ (|∇φ|² + m²|φ|² + λ|φ|⁴) d³x
 *
 * @param field Quantum field
 * @return Potential energy
 */
double calculate_potential_energy(const QuantumField* field);

/**
 * @brief Calculate gauge field energy
 *
 * Computes ∫ F_μν F^μν d⁴x
 *
 * @param field Quantum field with gauge field
 * @return Gauge field energy
 */
double calculate_gauge_energy(const QuantumField* field);

/**
 * @brief Calculate total Hamiltonian
 *
 * @param field Quantum field
 * @return Total energy (kinetic + potential + gauge)
 */
double calculate_hamiltonian(const QuantumField* field);

/**
 * @brief Calculate momentum density
 *
 * @param field Quantum field
 * @param momentum Output tensor for momentum density
 */
void calculate_momentum_density(
    const QuantumField* field,
    Tensor* momentum);

// ============================================================================
// Field Initialization
// ============================================================================

/**
 * @brief Initialize quantum field structure
 *
 * @param field Field to initialize
 * @param lattice_dims Lattice dimensions [t, x, y, z]
 * @param num_components Number of internal components
 * @param mass Field mass
 * @param coupling Self-coupling constant
 * @return true on success
 */
bool init_quantum_field(
    QuantumField* field,
    const size_t* lattice_dims,
    size_t num_components,
    double mass,
    double coupling);

/**
 * @brief Cleanup quantum field
 *
 * @param field Field to cleanup
 */
void cleanup_quantum_field(QuantumField* field);

/**
 * @brief Initialize gauge field for quantum field
 *
 * @param field Quantum field
 * @param gauge_group_dim Dimension of gauge group
 * @param gauge_coupling Gauge coupling strength
 * @return true on success
 */
bool init_gauge_field(
    QuantumField* field,
    size_t gauge_group_dim,
    double gauge_coupling);

// ============================================================================
// Field Evolution
// ============================================================================

/**
 * @brief Evolve field by one time step
 *
 * Uses symplectic Verlet integration for Hamiltonian dynamics.
 *
 * @param field Quantum field
 * @param config Evolution configuration
 * @return true on success
 */
bool evolve_field(QuantumField* field, const FieldConfig* config);

/**
 * @brief Apply boundary conditions
 *
 * @param field Quantum field
 */
void apply_boundary_conditions(QuantumField* field);

// ============================================================================
// Green's Functions and Propagators
// ============================================================================

/**
 * @brief Calculate Feynman propagator
 *
 * @param field Quantum field
 * @param x1 First spacetime point [t1, x1, y1, z1]
 * @param x2 Second spacetime point [t2, x2, y2, z2]
 * @return Propagator value G(x1, x2)
 */
complex double calculate_feynman_propagator(
    const QuantumField* field,
    const size_t* x1,
    const size_t* x2);

/**
 * @brief Calculate retarded Green's function
 *
 * @param field Quantum field
 * @param x1 First spacetime point
 * @param x2 Second spacetime point
 * @return Retarded propagator G_R(x1, x2)
 */
complex double calculate_retarded_propagator(
    const QuantumField* field,
    const size_t* x1,
    const size_t* x2);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_FIELD_CALCULATIONS_H
