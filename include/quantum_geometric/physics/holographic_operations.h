/**
 * @file holographic_operations.h
 * @brief Holographic quantum operations for AdS/CFT correspondence
 *
 * This module implements holographic operations based on the Anti-de Sitter/
 * Conformal Field Theory (AdS/CFT) correspondence. It provides:
 *
 * - Holographic entropy computation using the Ryu-Takayanagi formula
 * - Tensor network evolution for holographic states
 * - Bulk geometry reconstruction from boundary data
 * - M-theory brane dynamics in the holographic context
 *
 * The implementation uses hierarchical matrix techniques for O(n log n)
 * complexity and supports both CPU and GPU acceleration.
 */

#ifndef HOLOGRAPHIC_OPERATIONS_H
#define HOLOGRAPHIC_OPERATIONS_H

#include <complex.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================

/** Regularization epsilon for entropy calculations */
#define HOLOGRAPHIC_ENTROPY_EPSILON 1e-15

/** Maximum bulk reconstruction depth */
#define MAX_BULK_DEPTH 1024

/** Default tolerance for hierarchical operations */
#define HOLOGRAPHIC_DEFAULT_TOLERANCE 1e-10

// ============================================================================
// Type Definitions
// ============================================================================

/**
 * @brief Configuration for holographic computations
 */
typedef struct HolographicConfig {
    size_t bulk_dimension;           /**< Dimension of bulk space */
    size_t boundary_dimension;       /**< Dimension of boundary CFT */
    double ads_radius;               /**< AdS curvature radius */
    double tolerance;                /**< Numerical tolerance */
    bool use_gpu;                    /**< Enable GPU acceleration */
    bool use_parallel;               /**< Enable parallel computation */
    size_t max_iterations;           /**< Maximum iteration count */
} HolographicConfig;

/**
 * @brief Holographic state container
 */
typedef struct HolographicState {
    double complex* boundary_state;  /**< CFT boundary state */
    double complex* bulk_field;      /**< Reconstructed bulk field */
    size_t boundary_size;            /**< Size of boundary data */
    size_t bulk_size;                /**< Size of bulk data */
    double entanglement_entropy;     /**< Computed entanglement entropy */
    HolographicConfig config;        /**< Configuration parameters */
} HolographicState;

/**
 * @brief Tensor network structure for holographic states
 */
typedef struct HolographicTensorNetwork {
    double complex* tensors;         /**< Array of tensor elements */
    size_t* bond_dimensions;         /**< Bond dimensions for each leg */
    size_t num_tensors;              /**< Number of tensors in network */
    size_t num_bonds;                /**< Number of bonds */
    double norm;                     /**< Network normalization */
} HolographicTensorNetwork;

/**
 * @brief M-theory brane configuration
 */
typedef struct BraneConfig {
    size_t dimension;                /**< Brane dimension (D-branes) */
    double tension;                  /**< Brane tension */
    double* position;                /**< Position in bulk space */
    double* orientation;             /**< Orientation angles */
    bool is_boundary;                /**< Whether brane is at boundary */
} BraneConfig;

// ============================================================================
// Core Holographic Operations
// ============================================================================

/**
 * @brief Compute holographic entanglement entropy
 *
 * Implements the Ryu-Takayanagi formula for computing entanglement
 * entropy of a boundary region from the minimal surface area in the bulk.
 * Uses hierarchical matrix techniques for O(n log n) complexity.
 *
 * @param entropy Output array for entropy values
 * @param state Boundary quantum state
 * @param n Size of the state
 */
void compute_holographic_entropy(double complex* entropy,
                                const double complex* state,
                                size_t n);

/**
 * @brief Evolve tensor network under holographic dynamics
 *
 * Performs time evolution of a holographic tensor network using
 * the bulk Hamiltonian. Supports GPU acceleration when available.
 *
 * @param network Tensor network state (modified in place)
 * @param hamiltonian Evolution Hamiltonian
 * @param n Size of the network
 */
void evolve_tensor_network(double complex* network,
                          const double complex* hamiltonian,
                          size_t n);

/**
 * @brief Reconstruct bulk geometry from boundary CFT data
 *
 * Uses the HKLL (Hamilton-Kabat-Lifschytz-Lowe) smearing prescription
 * to reconstruct bulk operators from boundary CFT data.
 *
 * @param bulk Output bulk field configuration
 * @param boundary Input boundary CFT state
 * @param n Size of the data
 */
void reconstruct_bulk_geometry(double complex* bulk,
                              const double complex* boundary,
                              size_t n);

/**
 * @brief Compute M-theory brane dynamics
 *
 * Calculates the dynamics of M-theory branes including:
 * - Dirac-Born-Infeld action
 * - Wess-Zumino terms
 * - Brane-brane interactions
 *
 * @param dynamics Output array for dynamics data
 * @param branes Input brane configuration
 * @param n Size of the configuration
 */
void compute_m_theory_dynamics(double complex* dynamics,
                              const double complex* branes,
                              size_t n);

// ============================================================================
// State Management
// ============================================================================

/**
 * @brief Initialize holographic state
 *
 * @param state State to initialize
 * @param config Configuration parameters
 * @return true on success
 */
bool init_holographic_state(HolographicState* state,
                           const HolographicConfig* config);

/**
 * @brief Clean up holographic state
 *
 * @param state State to clean up
 */
void cleanup_holographic_state(HolographicState* state);

/**
 * @brief Clean up holographic operations resources
 *
 * Releases all cached resources used by holographic operations.
 */
void cleanup_holographic_operations(void);

// ============================================================================
// Tensor Network Operations
// ============================================================================

/**
 * @brief Create holographic tensor network
 *
 * @param num_tensors Number of tensors in network
 * @param bond_dims Array of bond dimensions
 * @return Pointer to created network, or NULL on failure
 */
HolographicTensorNetwork* create_tensor_network(size_t num_tensors,
                                                const size_t* bond_dims);

/**
 * @brief Destroy holographic tensor network
 *
 * @param network Network to destroy
 */
void destroy_tensor_network(HolographicTensorNetwork* network);

/**
 * @brief Contract tensor network to scalar
 *
 * @param network Network to contract
 * @return Scalar result of contraction
 */
double complex contract_tensor_network(const HolographicTensorNetwork* network);

/**
 * @brief Apply local tensor operator
 *
 * @param network Network to modify
 * @param operator Local operator to apply
 * @param site Site index
 * @return true on success
 */
bool apply_local_operator(HolographicTensorNetwork* network,
                         const double complex* operator,
                         size_t site);

// ============================================================================
// AdS/CFT Specific Operations
// ============================================================================

/**
 * @brief Compute boundary-to-bulk propagator
 *
 * Calculates the propagator from a boundary point to a bulk point
 * in AdS space.
 *
 * @param propagator Output propagator value
 * @param boundary_point Boundary position
 * @param bulk_point Bulk position (radial coordinate)
 * @param ads_radius AdS curvature radius
 * @return Propagator value
 */
double complex compute_bulk_boundary_propagator(
    size_t boundary_point,
    double bulk_point,
    double ads_radius);

/**
 * @brief Compute Ryu-Takayanagi surface area
 *
 * Finds the minimal surface in the bulk anchored to a boundary region
 * and computes its area for the Ryu-Takayanagi entropy formula.
 *
 * @param boundary_region_start Start of boundary region
 * @param boundary_region_end End of boundary region
 * @param config Configuration parameters
 * @return Minimal surface area
 */
double compute_rt_surface_area(size_t boundary_region_start,
                              size_t boundary_region_end,
                              const HolographicConfig* config);

/**
 * @brief Compute modular Hamiltonian
 *
 * Calculates the modular Hamiltonian for a boundary subregion,
 * which generates modular flow.
 *
 * @param modular_h Output modular Hamiltonian
 * @param state Boundary state
 * @param region_start Start of region
 * @param region_end End of region
 * @param n Total system size
 */
void compute_modular_hamiltonian(double complex* modular_h,
                                const double complex* state,
                                size_t region_start,
                                size_t region_end,
                                size_t n);

// ============================================================================
// Advanced Features
// ============================================================================

/**
 * @brief Compute quantum extremal surface
 *
 * Generalizes the RT surface to include quantum corrections
 * (island formula in semi-classical gravity).
 *
 * @param state Holographic state
 * @param region Boundary region specification
 * @param region_size Size of region array
 * @return Generalized entropy (area + bulk entropy)
 */
double compute_quantum_extremal_surface(const HolographicState* state,
                                       const size_t* region,
                                       size_t region_size);

/**
 * @brief Check error correction properties
 *
 * Verifies that the holographic code satisfies approximate quantum
 * error correction for the given bulk region.
 *
 * @param state Holographic state
 * @param bulk_region Bulk region to check
 * @param bulk_size Size of bulk region
 * @return Error correction parameter (0 = perfect, 1 = no correction)
 */
double check_error_correction_properties(const HolographicState* state,
                                        const size_t* bulk_region,
                                        size_t bulk_size);

#ifdef __cplusplus
}
#endif

#endif // HOLOGRAPHIC_OPERATIONS_H
