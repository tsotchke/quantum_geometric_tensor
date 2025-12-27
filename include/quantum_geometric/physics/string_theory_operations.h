/**
 * @file string_theory_operations.h
 * @brief High-performance string theory operations for quantum geometric systems
 *
 * Provides optimized implementations for:
 * - D-brane evolution using hierarchical matrix methods (O(log n))
 * - M-theory dynamics with GPU acceleration
 * - Mirror symmetry evaluation with distributed computing
 * - String moduli space computations
 *
 * Cross-platform support for Metal (macOS) and CUDA (Linux/Windows).
 */

#ifndef STRING_THEORY_OPERATIONS_H
#define STRING_THEORY_OPERATIONS_H

#include <stddef.h>
#include <stdbool.h>
#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Forward Declarations
// ============================================================================

struct HierarchicalMatrix;
struct GPUContext;

// ============================================================================
// Constants
// ============================================================================

#ifndef QG_STRING_COUPLING_DIM
#define QG_STRING_COUPLING_DIM 10     // Superstring theory dimension
#endif

#ifndef QG_MBRANE_MAX_DIM
#define QG_MBRANE_MAX_DIM 11          // M-theory dimension
#endif

#ifndef QG_CALABI_YAU_DIM
#define QG_CALABI_YAU_DIM 6           // Calabi-Yau manifold dimension
#endif

// ============================================================================
// D-Brane Configuration Types
// ============================================================================

/**
 * @brief D-brane type enumeration
 */
typedef enum {
    DBRANE_D0 = 0,     /**< D0-brane (point particle) */
    DBRANE_D1 = 1,     /**< D1-brane (string) */
    DBRANE_D2 = 2,     /**< D2-brane (membrane) */
    DBRANE_D3 = 3,     /**< D3-brane */
    DBRANE_D4 = 4,     /**< D4-brane */
    DBRANE_D5 = 5,     /**< D5-brane */
    DBRANE_D6 = 6,     /**< D6-brane */
    DBRANE_D7 = 7,     /**< D7-brane */
    DBRANE_D8 = 8,     /**< D8-brane */
    DBRANE_D9 = 9      /**< D9-brane (spacetime filling) */
} DBraneType;

/**
 * @brief D-brane boundary conditions
 */
typedef enum {
    BOUNDARY_NEUMANN = 0,    /**< Neumann boundary conditions (free endpoints) */
    BOUNDARY_DIRICHLET = 1,  /**< Dirichlet boundary conditions (fixed endpoints) */
    BOUNDARY_MIXED = 2       /**< Mixed boundary conditions */
} BraneBoundaryType;

/**
 * @brief Configuration for D-brane system
 */
typedef struct BraneConfig {
    DBraneType brane_type;              /**< Type of D-brane */
    BraneBoundaryType boundary_type;    /**< Boundary conditions */
    size_t dimension;                   /**< Worldvolume dimension */
    double string_coupling;             /**< String coupling constant g_s */
    double string_length;               /**< String length l_s */
    double tension;                     /**< Brane tension T_p */

    // Embedding coordinates
    double* position;                   /**< Position in target space */
    double* velocity;                   /**< Velocity in target space */
    size_t target_dim;                  /**< Target space dimension */

    // Gauge field on brane
    double complex* gauge_field;        /**< U(1) or non-abelian gauge field */
    size_t gauge_rank;                  /**< Rank of gauge group */

    // Evolution parameters
    double dt;                          /**< Time step for evolution */
    double tolerance;                   /**< Numerical tolerance */
    size_t max_iterations;              /**< Maximum iterations */
    bool use_hierarchical;              /**< Use hierarchical methods */
    bool use_gpu;                       /**< Use GPU acceleration */
} BraneConfig;

// ============================================================================
// M-Theory Configuration Types
// ============================================================================

/**
 * @brief M-theory object types
 */
typedef enum {
    MTHEORY_M2 = 2,      /**< M2-brane (membrane) */
    MTHEORY_M5 = 5,      /**< M5-brane */
    MTHEORY_KK = 10,     /**< Kaluza-Klein monopole */
    MTHEORY_WAVE = 11    /**< Gravitational wave solution */
} MTheoryObjectType;

/**
 * @brief Configuration for M-theory computation
 */
typedef struct MTheoryConfig {
    MTheoryObjectType object_type;      /**< Type of M-theory object */
    size_t dimension;                   /**< Number of dimensions */
    double planck_length_11d;           /**< 11D Planck length */
    double m2_tension;                  /**< M2-brane tension */
    double m5_tension;                  /**< M5-brane tension */

    // Compactification parameters
    double* compactification_radii;     /**< Radii for compact dimensions */
    size_t num_compact_dims;            /**< Number of compact dimensions */

    // Computation parameters
    double tolerance;                   /**< Numerical tolerance */
    bool include_quantum_corrections;   /**< Include quantum corrections */
    bool use_gpu;                       /**< Use GPU acceleration */
} MTheoryConfig;

// ============================================================================
// Mirror Symmetry Configuration Types
// ============================================================================

/**
 * @brief Calabi-Yau manifold types
 */
typedef enum {
    CY_QUINTIC = 0,          /**< Quintic threefold */
    CY_TORUS = 1,            /**< Torus (K3 x T^2) */
    CY_ORBIFOLD = 2,         /**< Orbifold construction */
    CY_COMPLETE_INTERSECTION = 3,  /**< Complete intersection */
    CY_HYPERSURFACE = 4      /**< General hypersurface */
} CalabiYauType;

/**
 * @brief Configuration for mirror symmetry computation
 */
typedef struct MirrorConfig {
    CalabiYauType manifold_type;        /**< Type of Calabi-Yau manifold */
    size_t hodge_h11;                   /**< Hodge number h^{1,1} */
    size_t hodge_h21;                   /**< Hodge number h^{2,1} */

    // Moduli space parameters
    double complex* kahler_moduli;      /**< Kähler moduli */
    double complex* complex_moduli;     /**< Complex structure moduli */
    size_t num_kahler;                  /**< Number of Kähler moduli */
    size_t num_complex;                 /**< Number of complex moduli */

    // Mirror map parameters
    double tolerance;                   /**< Numerical tolerance */
    size_t expansion_order;             /**< Order of series expansion */
    bool compute_instanton_corrections; /**< Include instanton corrections */
    bool use_distributed;               /**< Use distributed computing */
} MirrorConfig;

// ============================================================================
// Result Types
// ============================================================================

/**
 * @brief Result of D-brane evolution
 */
typedef struct BraneEvolutionResult {
    double complex* final_state;        /**< Final brane configuration */
    double energy;                      /**< Total energy */
    double action;                      /**< DBI action value */
    size_t iterations;                  /**< Number of iterations used */
    double error_estimate;              /**< Numerical error estimate */
    bool converged;                     /**< Whether evolution converged */
} BraneEvolutionResult;

/**
 * @brief Result of M-theory computation
 */
typedef struct MTheoryResult {
    double complex* dynamics;           /**< M-theory dynamics field */
    double effective_coupling;          /**< Effective coupling constant */
    double tension_correction;          /**< Quantum correction to tension */
    size_t num_fluxes;                  /**< Number of flux components */
    double* flux_values;                /**< Flux quantization values */
    bool is_bps;                        /**< Whether state is BPS */
} MTheoryResult;

/**
 * @brief Result of mirror symmetry computation
 */
typedef struct MirrorResult {
    double complex* mirror_map;         /**< Mirror map coefficients */
    double complex* periods;            /**< Period integrals */
    double complex* prepotential;       /**< Prepotential (type B side) */
    double complex* yukawa_couplings;   /**< Yukawa couplings */
    size_t instanton_count;             /**< Number of instanton contributions */
    double* instanton_numbers;          /**< Instanton numbers */
    bool mirror_verified;               /**< Whether mirror relation verified */
} MirrorResult;

// ============================================================================
// Core D-Brane Operations
// ============================================================================

/**
 * @brief Evolve D-brane system
 *
 * Uses hierarchical matrix methods for O(log n) complexity.
 *
 * @param branes Brane field values (modified in place)
 * @param dynamics Dynamics field (Hamiltonian)
 * @param n Field dimension
 */
void evolve_d_branes(double complex* branes,
                     const double complex* dynamics,
                     size_t n);

/**
 * @brief Evolve D-brane system with configuration
 *
 * @param branes Brane field values
 * @param config Brane configuration
 * @param n Field dimension
 * @return Evolution result
 */
BraneEvolutionResult* evolve_d_branes_with_config(double complex* branes,
                                                   const BraneConfig* config,
                                                   size_t n);

/**
 * @brief Evolve hierarchical branes using fast approximation
 *
 * @param branes Brane field values (modified in place)
 * @param config Brane configuration
 * @param n Field dimension
 */
void evolve_hierarchical_branes(double complex* branes,
                                const BraneConfig* config,
                                size_t n);

/**
 * @brief Compute DBI action for brane configuration
 *
 * @param branes Brane field values
 * @param config Brane configuration
 * @param n Field dimension
 * @return DBI action value
 */
double compute_dbi_action(const double complex* branes,
                          const BraneConfig* config,
                          size_t n);

/**
 * @brief Free brane evolution result
 *
 * @param result Result to free
 */
void free_brane_result(BraneEvolutionResult* result);

// ============================================================================
// M-Theory Operations
// ============================================================================

/**
 * @brief Compute M-theory dynamics
 *
 * Uses GPU acceleration when available.
 *
 * @param dynamics Output dynamics field
 * @param branes Input brane configuration
 * @param n Field dimension
 */
void compute_m_theory_dynamics(double complex* dynamics,
                               const double complex* branes,
                               size_t n);

/**
 * @brief Compute M-theory dynamics with configuration
 *
 * @param dynamics Output dynamics field
 * @param branes Input brane configuration
 * @param config M-theory configuration
 * @param n Field dimension
 * @return M-theory result
 */
MTheoryResult* compute_m_theory_with_config(double complex* dynamics,
                                            const double complex* branes,
                                            const MTheoryConfig* config,
                                            size_t n);

/**
 * @brief Compute M2-brane equations of motion
 *
 * @param equations Output equations
 * @param membrane M2-brane field
 * @param config Configuration
 * @param n Field dimension
 */
void compute_m2_equations(double complex* equations,
                          const double complex* membrane,
                          const MTheoryConfig* config,
                          size_t n);

/**
 * @brief Compute M5-brane equations of motion
 *
 * @param equations Output equations
 * @param fivebrane M5-brane field
 * @param config Configuration
 * @param n Field dimension
 */
void compute_m5_equations(double complex* equations,
                          const double complex* fivebrane,
                          const MTheoryConfig* config,
                          size_t n);

/**
 * @brief Free M-theory result
 *
 * @param result Result to free
 */
void free_mtheory_result(MTheoryResult* result);

// ============================================================================
// Mirror Symmetry Operations
// ============================================================================

/**
 * @brief Evaluate mirror symmetry map
 *
 * Uses distributed computing for large systems.
 *
 * @param mirror Output mirror map values
 * @param manifold Input manifold data
 * @param n Field dimension
 */
void evaluate_mirror_symmetry(double complex* mirror,
                              const double complex* manifold,
                              size_t n);

/**
 * @brief Evaluate mirror symmetry with configuration
 *
 * @param mirror Output mirror map values
 * @param manifold Input manifold data
 * @param config Mirror configuration
 * @param n Field dimension
 * @return Mirror result
 */
MirrorResult* evaluate_mirror_with_config(double complex* mirror,
                                           const double complex* manifold,
                                           const MirrorConfig* config,
                                           size_t n);

/**
 * @brief Compute period integrals on Calabi-Yau
 *
 * @param periods Output period integrals
 * @param moduli Complex structure moduli
 * @param config Mirror configuration
 * @param n Number of periods
 */
void compute_period_integrals(double complex* periods,
                              const double complex* moduli,
                              const MirrorConfig* config,
                              size_t n);

/**
 * @brief Compute instanton corrections to mirror map
 *
 * @param corrections Output instanton corrections
 * @param kahler Kähler moduli
 * @param config Mirror configuration
 * @param max_degree Maximum instanton degree
 */
void compute_instanton_corrections(double complex* corrections,
                                   const double complex* kahler,
                                   const MirrorConfig* config,
                                   size_t max_degree);

/**
 * @brief Free mirror result
 *
 * @param result Result to free
 */
void free_mirror_result(MirrorResult* result);

// ============================================================================
// Configuration Management
// ============================================================================

/**
 * @brief Create default brane configuration
 *
 * @param brane_type Type of D-brane
 * @return Default configuration (caller must free)
 */
BraneConfig* create_default_brane_config(DBraneType brane_type);

/**
 * @brief Create default M-theory configuration
 *
 * @param object_type Type of M-theory object
 * @return Default configuration (caller must free)
 */
MTheoryConfig* create_default_mtheory_config(MTheoryObjectType object_type);

/**
 * @brief Create default mirror configuration
 *
 * @param manifold_type Type of Calabi-Yau manifold
 * @return Default configuration (caller must free)
 */
MirrorConfig* create_default_mirror_config(CalabiYauType manifold_type);

/**
 * @brief Free brane configuration
 *
 * @param config Configuration to free
 */
void free_brane_config(BraneConfig* config);

/**
 * @brief Free M-theory configuration
 *
 * @param config Configuration to free
 */
void free_mtheory_config(MTheoryConfig* config);

/**
 * @brief Free mirror configuration
 *
 * @param config Configuration to free
 */
void free_mirror_config(MirrorConfig* config);

// ============================================================================
// GPU-Accelerated Operations
// ============================================================================

/**
 * @brief Compute M-theory dynamics on GPU
 *
 * @param dynamics Output dynamics field (device memory)
 * @param branes Input branes (device memory)
 * @param n Field dimension
 * @param ctx GPU context
 * @return 0 on success
 */
int compute_m_theory_dynamics_gpu(double complex* dynamics,
                                  const double complex* branes,
                                  size_t n,
                                  struct GPUContext* ctx);

/**
 * @brief Evolve D-branes on GPU
 *
 * @param branes Brane field (device memory, modified in place)
 * @param dynamics Dynamics field (device memory)
 * @param n Field dimension
 * @param ctx GPU context
 * @return 0 on success
 */
int evolve_d_branes_gpu(double complex* branes,
                        const double complex* dynamics,
                        size_t n,
                        struct GPUContext* ctx);

// ============================================================================
// Cleanup Functions
// ============================================================================

/**
 * @brief Cleanup string theory cache
 */
void cleanup_string_theory_cache(void);

/**
 * @brief Cleanup string theory buffers
 */
void cleanup_string_theory_buffers(void);

/**
 * @brief Cleanup all string theory operation resources
 */
void cleanup_string_theory_operations(void);

#ifdef __cplusplus
}
#endif

#endif // STRING_THEORY_OPERATIONS_H
