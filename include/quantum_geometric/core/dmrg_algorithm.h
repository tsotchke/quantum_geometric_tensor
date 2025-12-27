#ifndef DMRG_ALGORITHM_H
#define DMRG_ALGORITHM_H

#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/mps_operations.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file dmrg_algorithm.h
 * @brief Density Matrix Renormalization Group (DMRG) algorithm implementation
 *
 * Implements the two-site DMRG algorithm for finding ground states of
 * 1D quantum many-body systems represented as Matrix Product States (MPS).
 */

// ============================================================================
// Data Structures
// ============================================================================

/**
 * @brief Hamiltonian represented as sum of local and nearest-neighbor terms
 *
 * H = Σ_i H_local[i] + Σ_i H_nn[i] (coupling sites i and i+1)
 */
typedef struct {
    ComplexFloat** local_terms;     ///< H_i for each site (d x d matrices)
    ComplexFloat** nn_terms;        ///< H_{i,i+1} nearest-neighbor (d^2 x d^2 matrices)
    size_t num_sites;               ///< Number of sites
    size_t local_dim;               ///< Local Hilbert space dimension (d)
    bool has_long_range;            ///< Whether long-range terms exist
    ComplexFloat** long_range_left; ///< Left operators for MPO representation
    ComplexFloat** long_range_right;///< Right operators for MPO representation
    size_t mpo_bond_dim;            ///< Bond dimension of MPO representation
} DMRGHamiltonian;

/**
 * @brief Configuration options for DMRG algorithm
 */
typedef struct {
    size_t max_sweeps;              ///< Maximum number of sweeps (default: 20)
    size_t max_bond_dim;            ///< Maximum MPS bond dimension (default: 100)
    double energy_tolerance;        ///< Convergence tolerance for energy (default: 1e-8)
    double truncation_cutoff;       ///< SVD truncation cutoff (default: 1e-10)
    bool verbose;                   ///< Print progress information
    size_t lanczos_iterations;      ///< Max Lanczos iterations (default: 100)
    double lanczos_tolerance;       ///< Lanczos convergence tolerance (default: 1e-12)
    bool use_two_site;              ///< Use two-site DMRG (default: true)
    size_t warmup_sweeps;           ///< Number of warmup sweeps with lower bond dim
    size_t warmup_bond_dim;         ///< Bond dimension during warmup
    bool compute_variance;          ///< Compute energy variance for convergence check
} DMRGConfig;

/**
 * @brief Result of DMRG optimization
 */
typedef struct {
    double* energies;               ///< Energy at each sweep
    double* truncation_errors;      ///< Max truncation error at each sweep
    double* variances;              ///< Energy variance at each sweep (if computed)
    size_t num_sweeps;              ///< Actual number of sweeps performed
    bool converged;                 ///< Whether convergence was achieved
    double final_energy;            ///< Final ground state energy
    double final_variance;          ///< Final energy variance
    size_t* bond_dimensions;        ///< Final bond dimensions
    size_t max_bond_dim_used;       ///< Maximum bond dimension used
} DMRGResult;

/**
 * @brief Environment blocks for efficient DMRG sweeping
 */
typedef struct {
    ComplexFloat** left_blocks;     ///< Left environment blocks [site]
    ComplexFloat** right_blocks;    ///< Right environment blocks [site]
    size_t* left_dims;              ///< Dimensions of left blocks
    size_t* right_dims;             ///< Dimensions of right blocks
    size_t num_sites;               ///< Number of sites
    size_t mpo_bond_dim;            ///< MPO bond dimension
} DMRGEnvironment;

// ============================================================================
// Hamiltonian Functions
// ============================================================================

/**
 * @brief Create a DMRG Hamiltonian structure
 *
 * @param num_sites Number of sites in the chain
 * @param local_dim Local Hilbert space dimension
 * @return Pointer to allocated Hamiltonian, or NULL on failure
 */
DMRGHamiltonian* dmrg_create_hamiltonian(size_t num_sites, size_t local_dim);

/**
 * @brief Destroy a DMRG Hamiltonian structure
 *
 * @param hamiltonian Hamiltonian to destroy
 */
void dmrg_destroy_hamiltonian(DMRGHamiltonian* hamiltonian);

/**
 * @brief Set a local term in the Hamiltonian
 *
 * @param hamiltonian The Hamiltonian
 * @param site Site index
 * @param term The local operator (d x d matrix, will be copied)
 * @return QGT_SUCCESS on success
 */
qgt_error_t dmrg_set_local_term(
    DMRGHamiltonian* hamiltonian,
    size_t site,
    const ComplexFloat* term);

/**
 * @brief Set a nearest-neighbor term in the Hamiltonian
 *
 * @param hamiltonian The Hamiltonian
 * @param site Left site index (couples site and site+1)
 * @param term The two-site operator (d^2 x d^2 matrix, will be copied)
 * @return QGT_SUCCESS on success
 */
qgt_error_t dmrg_set_nn_term(
    DMRGHamiltonian* hamiltonian,
    size_t site,
    const ComplexFloat* term);

/**
 * @brief Create Heisenberg XXZ Hamiltonian
 *
 * H = J Σ_i (S^x_i S^x_{i+1} + S^y_i S^y_{i+1} + Δ S^z_i S^z_{i+1}) + h Σ_i S^z_i
 *
 * @param num_sites Number of sites
 * @param J Exchange coupling
 * @param delta Anisotropy parameter
 * @param h Magnetic field strength
 * @return Pointer to allocated Hamiltonian
 */
DMRGHamiltonian* dmrg_create_heisenberg_xxz(
    size_t num_sites,
    double J,
    double delta,
    double h);

/**
 * @brief Create transverse-field Ising Hamiltonian
 *
 * H = -J Σ_i S^z_i S^z_{i+1} - h Σ_i S^x_i
 *
 * @param num_sites Number of sites
 * @param J Coupling strength
 * @param h Transverse field strength
 * @return Pointer to allocated Hamiltonian
 */
DMRGHamiltonian* dmrg_create_tfim(
    size_t num_sites,
    double J,
    double h);

// ============================================================================
// Configuration Functions
// ============================================================================

/**
 * @brief Get default DMRG configuration
 *
 * @return Default configuration with reasonable parameters
 */
DMRGConfig dmrg_get_default_config(void);

// ============================================================================
// Result Functions
// ============================================================================

/**
 * @brief Create a DMRG result structure
 *
 * @param max_sweeps Maximum number of sweeps to allocate for
 * @param num_sites Number of sites (for bond dimensions)
 * @return Pointer to allocated result, or NULL on failure
 */
DMRGResult* dmrg_create_result(size_t max_sweeps, size_t num_sites);

/**
 * @brief Destroy a DMRG result structure
 *
 * @param result Result to destroy
 */
void dmrg_destroy_result(DMRGResult* result);

// ============================================================================
// Main DMRG Algorithm
// ============================================================================

/**
 * @brief Find ground state using DMRG algorithm
 *
 * This is the main entry point for DMRG. It performs sweeps through the
 * chain, optimizing the MPS to minimize the energy.
 *
 * @param ground_state Output MPS (must be pre-allocated)
 * @param hamiltonian The Hamiltonian to minimize
 * @param config Configuration options
 * @param result Output result structure (optional, can be NULL)
 * @return QGT_SUCCESS on success
 */
qgt_error_t dmrg_ground_state(
    MatrixProductState* ground_state,
    const DMRGHamiltonian* hamiltonian,
    const DMRGConfig* config,
    DMRGResult* result);

/**
 * @brief Perform a single DMRG sweep (left-to-right or right-to-left)
 *
 * @param mps The MPS to optimize
 * @param hamiltonian The Hamiltonian
 * @param env Environment blocks
 * @param config Configuration
 * @param direction true for left-to-right, false for right-to-left
 * @param energy Output: energy after sweep
 * @param max_trunc_error Output: maximum truncation error in sweep
 * @return QGT_SUCCESS on success
 */
qgt_error_t dmrg_sweep(
    MatrixProductState* mps,
    const DMRGHamiltonian* hamiltonian,
    DMRGEnvironment* env,
    const DMRGConfig* config,
    bool direction,
    double* energy,
    double* max_trunc_error);

// ============================================================================
// Environment Functions
// ============================================================================

/**
 * @brief Create DMRG environment structure
 *
 * @param num_sites Number of sites
 * @param mpo_bond_dim MPO bond dimension
 * @return Pointer to allocated environment, or NULL on failure
 */
DMRGEnvironment* dmrg_create_environment(size_t num_sites, size_t mpo_bond_dim);

/**
 * @brief Destroy DMRG environment structure
 *
 * @param env Environment to destroy
 */
void dmrg_destroy_environment(DMRGEnvironment* env);

/**
 * @brief Initialize environment blocks from MPS
 *
 * @param env Environment to initialize
 * @param mps The MPS
 * @param hamiltonian The Hamiltonian
 * @return QGT_SUCCESS on success
 */
qgt_error_t dmrg_init_environment(
    DMRGEnvironment* env,
    const MatrixProductState* mps,
    const DMRGHamiltonian* hamiltonian);

/**
 * @brief Update left environment block after optimizing a site
 *
 * @param env Environment
 * @param mps The MPS
 * @param hamiltonian The Hamiltonian
 * @param site Site that was just optimized
 * @return QGT_SUCCESS on success
 */
qgt_error_t dmrg_update_left_block(
    DMRGEnvironment* env,
    const MatrixProductState* mps,
    const DMRGHamiltonian* hamiltonian,
    size_t site);

/**
 * @brief Update right environment block after optimizing a site
 *
 * @param env Environment
 * @param mps The MPS
 * @param hamiltonian The Hamiltonian
 * @param site Site that was just optimized
 * @return QGT_SUCCESS on success
 */
qgt_error_t dmrg_update_right_block(
    DMRGEnvironment* env,
    const MatrixProductState* mps,
    const DMRGHamiltonian* hamiltonian,
    size_t site);

// ============================================================================
// Optimization Functions
// ============================================================================

/**
 * @brief Build effective Hamiltonian for two-site optimization
 *
 * @param h_eff Output: effective Hamiltonian matrix
 * @param h_eff_dim Output: dimension of effective Hamiltonian
 * @param mps The MPS
 * @param hamiltonian The Hamiltonian
 * @param env Environment blocks
 * @param site Left site index for two-site optimization
 * @return QGT_SUCCESS on success
 */
qgt_error_t dmrg_build_effective_hamiltonian(
    ComplexFloat** h_eff,
    size_t* h_eff_dim,
    const MatrixProductState* mps,
    const DMRGHamiltonian* hamiltonian,
    const DMRGEnvironment* env,
    size_t site);

/**
 * @brief Optimize two-site tensor using Lanczos
 *
 * @param two_site Output: optimized two-site tensor
 * @param dim Dimension of two-site tensor
 * @param h_eff Effective Hamiltonian
 * @param h_eff_dim Dimension of effective Hamiltonian
 * @param energy Output: energy eigenvalue
 * @param config Configuration
 * @return QGT_SUCCESS on success
 */
qgt_error_t dmrg_optimize_two_site(
    ComplexFloat* two_site,
    size_t dim,
    const ComplexFloat* h_eff,
    size_t h_eff_dim,
    double* energy,
    const DMRGConfig* config);

/**
 * @brief Split two-site tensor into two MPS tensors using SVD
 *
 * @param two_site Input two-site tensor
 * @param left_dim Left bond dimension
 * @param d Physical dimension
 * @param right_dim Right bond dimension
 * @param A_left Output: left tensor
 * @param A_right Output: right tensor
 * @param new_bond Output: new bond dimension after truncation
 * @param max_bond Maximum bond dimension
 * @param cutoff Truncation cutoff
 * @param trunc_error Output: truncation error
 * @param direction true = move orthogonality center right, false = left
 * @return QGT_SUCCESS on success
 */
qgt_error_t dmrg_split_two_site(
    const ComplexFloat* two_site,
    size_t left_dim,
    size_t d,
    size_t right_dim,
    ComplexFloat** A_left,
    ComplexFloat** A_right,
    size_t* new_bond,
    size_t max_bond,
    double cutoff,
    double* trunc_error,
    bool direction);

// ============================================================================
// Lanczos Eigensolver
// ============================================================================

/**
 * @brief Find ground state using Lanczos algorithm
 *
 * Implements the Lanczos algorithm for finding the lowest eigenvalue
 * and eigenvector of a Hermitian matrix.
 *
 * @param matrix Input Hermitian matrix (can be NULL for matrix-free)
 * @param dim Matrix dimension
 * @param eigenvector Output eigenvector
 * @param eigenvalue Output eigenvalue
 * @param max_iterations Maximum Lanczos iterations
 * @param tolerance Convergence tolerance
 * @param matvec Optional matrix-vector product function (used if matrix is NULL)
 * @param matvec_data Optional data for matvec function
 * @return QGT_SUCCESS on success
 */
qgt_error_t lanczos_ground_state(
    const ComplexFloat* matrix,
    size_t dim,
    ComplexFloat* eigenvector,
    double* eigenvalue,
    size_t max_iterations,
    double tolerance,
    void (*matvec)(const ComplexFloat* v, ComplexFloat* Av, size_t dim, void* data),
    void* matvec_data);

// ============================================================================
// Observables and Measurements
// ============================================================================

/**
 * @brief Compute energy expectation value <H>
 *
 * @param mps The MPS state
 * @param hamiltonian The Hamiltonian
 * @param energy Output: energy
 * @return QGT_SUCCESS on success
 */
qgt_error_t dmrg_compute_energy(
    const MatrixProductState* mps,
    const DMRGHamiltonian* hamiltonian,
    double* energy);

/**
 * @brief Compute energy variance <H^2> - <H>^2
 *
 * @param mps The MPS state
 * @param hamiltonian The Hamiltonian
 * @param variance Output: variance
 * @return QGT_SUCCESS on success
 */
qgt_error_t dmrg_compute_variance(
    const MatrixProductState* mps,
    const DMRGHamiltonian* hamiltonian,
    double* variance);

#ifdef __cplusplus
}
#endif

#endif // DMRG_ALGORITHM_H
