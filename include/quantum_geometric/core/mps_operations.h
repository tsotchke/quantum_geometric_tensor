#ifndef MPS_OPERATIONS_H
#define MPS_OPERATIONS_H

#include "quantum_geometric/core/quantum_complex.h"
#include "quantum_geometric/core/error_codes.h"
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Matrix Product State (MPS) Data Structures
// =============================================================================

/**
 * @brief Canonical form of the MPS
 */
typedef enum {
    MPS_CANONICAL_NONE,     // No canonical form
    MPS_CANONICAL_LEFT,     // Left-canonical: A†A = I
    MPS_CANONICAL_RIGHT,    // Right-canonical: AA† = I
    MPS_CANONICAL_MIXED     // Mixed canonical with orthogonality center
} mps_canonical_form_t;

/**
 * @brief Matrix Product State representation
 *
 * An MPS represents a quantum state |ψ⟩ as a product of tensors:
 * |ψ⟩ = Σ_{s1,...,sN} A[1]^{s1} A[2]^{s2} ... A[N]^{sN} |s1,...,sN⟩
 *
 * Each tensor A[i] has shape (χ_{i-1}, d, χ_i) where:
 * - χ_i is the bond dimension at cut i
 * - d is the physical dimension (local Hilbert space)
 */
typedef struct {
    ComplexFloat** tensors;         // tensors[site] points to tensor data
    size_t* tensor_sizes;           // Size of each tensor (left_bond * phys * right_bond)
    size_t* left_bond_dims;         // left_bond_dims[site] = χ at left of site
    size_t* right_bond_dims;        // right_bond_dims[site] = χ at right of site
    size_t num_sites;               // Number of sites N
    size_t physical_dim;            // Local Hilbert space dimension d
    size_t max_bond_dim;            // Maximum allowed bond dimension
    mps_canonical_form_t form;      // Current canonical form
    size_t orthogonality_center;    // Site of orthogonality center (for mixed canonical)
    bool is_normalized;             // Whether the state is normalized
} MatrixProductState;

/**
 * @brief Configuration for MPS operations
 */
typedef struct {
    size_t max_bond_dim;            // Maximum bond dimension for truncation
    double truncation_cutoff;       // Cutoff for small singular values
    double normalization_tolerance; // Tolerance for normalization checks
    bool normalize_after_truncation; // Auto-normalize after truncation
} mps_config_t;

// =============================================================================
// Creation and Destruction
// =============================================================================

/**
 * @brief Create a new MPS with given parameters
 *
 * @param num_sites Number of physical sites
 * @param physical_dim Local Hilbert space dimension
 * @param max_bond_dim Maximum allowed bond dimension
 * @return Pointer to created MPS, or NULL on failure
 */
MatrixProductState* mps_create(size_t num_sites, size_t physical_dim, size_t max_bond_dim);

/**
 * @brief Destroy an MPS and free all memory
 *
 * @param mps The MPS to destroy
 */
void mps_destroy(MatrixProductState* mps);

/**
 * @brief Create a deep copy of an MPS
 *
 * @param src The MPS to clone
 * @return Pointer to cloned MPS, or NULL on failure
 */
MatrixProductState* mps_clone(const MatrixProductState* src);

// =============================================================================
// Initialization
// =============================================================================

/**
 * @brief Initialize MPS from a full state vector (requires SVD decomposition)
 *
 * Decomposes a 2^N dimensional state vector into MPS form using SVD.
 * The resulting MPS will be in left-canonical form.
 *
 * @param mps The MPS to initialize
 * @param state Full state vector of dimension d^N
 * @param max_bond_dim Maximum bond dimension to use
 * @return QGT_SUCCESS on success
 */
qgt_error_t mps_from_state_vector(MatrixProductState* mps,
                                  const ComplexFloat* state,
                                  size_t max_bond_dim);

/**
 * @brief Convert MPS back to a full state vector
 *
 * Warning: This is exponential in the number of sites!
 *
 * @param mps The MPS to convert
 * @param state Output state vector (must be pre-allocated with d^N elements)
 * @return QGT_SUCCESS on success
 */
qgt_error_t mps_to_state_vector(const MatrixProductState* mps,
                                ComplexFloat* state);

/**
 * @brief Initialize MPS to a random state
 *
 * @param mps The MPS to initialize
 * @param bond_dim Bond dimension to use
 * @return QGT_SUCCESS on success
 */
qgt_error_t mps_initialize_random(MatrixProductState* mps, size_t bond_dim);

/**
 * @brief Initialize MPS to a product state
 *
 * Creates |s1⟩ ⊗ |s2⟩ ⊗ ... ⊗ |sN⟩
 *
 * @param mps The MPS to initialize
 * @param local_states Array of local state indices (0 to d-1)
 * @return QGT_SUCCESS on success
 */
qgt_error_t mps_initialize_product_state(MatrixProductState* mps,
                                         const int* local_states);

/**
 * @brief Initialize MPS to all zeros state |00...0⟩
 *
 * @param mps The MPS to initialize
 * @return QGT_SUCCESS on success
 */
qgt_error_t mps_initialize_zero_state(MatrixProductState* mps);

// =============================================================================
// Canonicalization (CRITICAL for DMRG)
// =============================================================================

/**
 * @brief Left-canonicalize the entire MPS
 *
 * After this, all tensors satisfy A†A = I (isometric from left).
 * The norm is absorbed into the rightmost tensor.
 *
 * @param mps The MPS to canonicalize
 * @return QGT_SUCCESS on success
 */
qgt_error_t mps_left_canonicalize(MatrixProductState* mps);

/**
 * @brief Right-canonicalize the entire MPS
 *
 * After this, all tensors satisfy AA† = I (isometric from right).
 * The norm is absorbed into the leftmost tensor.
 *
 * @param mps The MPS to canonicalize
 * @return QGT_SUCCESS on success
 */
qgt_error_t mps_right_canonicalize(MatrixProductState* mps);

/**
 * @brief Put MPS in mixed canonical form with given center
 *
 * Sites 0 to center-1 are left-canonical.
 * Sites center+1 to N-1 are right-canonical.
 * Site center is the orthogonality center.
 *
 * @param mps The MPS to canonicalize
 * @param center Site index for orthogonality center
 * @return QGT_SUCCESS on success
 */
qgt_error_t mps_mixed_canonicalize(MatrixProductState* mps, size_t center);

/**
 * @brief Move the orthogonality center to a new site
 *
 * Requires MPS to already be in mixed canonical form.
 *
 * @param mps The MPS
 * @param new_center New site for orthogonality center
 * @return QGT_SUCCESS on success
 */
qgt_error_t mps_move_orthogonality_center(MatrixProductState* mps, size_t new_center);

// =============================================================================
// SVD and Truncation
// =============================================================================

/**
 * @brief Truncate MPS bond dimensions using SVD
 *
 * @param mps The MPS to truncate
 * @param max_bond_dim Maximum bond dimension after truncation
 * @param cutoff Cutoff for small singular values (relative to largest)
 * @param truncation_error Output: total truncation error (sum of discarded weights)
 * @return QGT_SUCCESS on success
 */
qgt_error_t mps_truncate(MatrixProductState* mps,
                         size_t max_bond_dim,
                         double cutoff,
                         double* truncation_error);

/**
 * @brief Compress MPS to target fidelity
 *
 * Adaptively chooses bond dimensions to achieve target fidelity.
 *
 * @param mps The MPS to compress
 * @param target_fidelity Target fidelity (1 - truncation_error²)
 * @return QGT_SUCCESS on success
 */
qgt_error_t mps_compress(MatrixProductState* mps, double target_fidelity);

// =============================================================================
// Physical Observables
// =============================================================================

/**
 * @brief Compute expectation value of a local operator
 *
 * Computes ⟨ψ|O_i|ψ⟩ where O_i acts on site i.
 *
 * @param mps The MPS state
 * @param local_op The local operator (d×d matrix)
 * @param site Site index
 * @param result Output: expectation value
 * @return QGT_SUCCESS on success
 */
qgt_error_t mps_expectation_local(const MatrixProductState* mps,
                                  const ComplexFloat* local_op,
                                  size_t site,
                                  ComplexFloat* result);

/**
 * @brief Compute two-point correlation function
 *
 * Computes ⟨ψ|O_A O_B|ψ⟩ where O_A acts on site_A and O_B on site_B.
 *
 * @param mps The MPS state
 * @param op_A Operator at site A
 * @param site_A Site index for operator A
 * @param op_B Operator at site B
 * @param site_B Site index for operator B
 * @param result Output: correlation value
 * @return QGT_SUCCESS on success
 */
qgt_error_t mps_correlation_function(const MatrixProductState* mps,
                                     const ComplexFloat* op_A,
                                     size_t site_A,
                                     const ComplexFloat* op_B,
                                     size_t site_B,
                                     ComplexFloat* result);

/**
 * @brief Compute entanglement entropy at a cut
 *
 * Computes S = -Σ λ_i² log(λ_i²) where λ_i are singular values at the cut.
 * Requires MPS to be in canonical form with orthogonality center at or adjacent to cut.
 *
 * @param mps The MPS state
 * @param cut Cut position (between site 'cut' and 'cut+1')
 * @param entropy Output: entanglement entropy
 * @return QGT_SUCCESS on success
 */
qgt_error_t mps_entanglement_entropy(const MatrixProductState* mps,
                                     size_t cut,
                                     double* entropy);

/**
 * @brief Get the singular value spectrum at a cut
 *
 * @param mps The MPS state
 * @param cut Cut position
 * @param singular_values Output array (must be pre-allocated)
 * @param num_values Output: number of singular values
 * @return QGT_SUCCESS on success
 */
qgt_error_t mps_get_singular_values(const MatrixProductState* mps,
                                    size_t cut,
                                    double* singular_values,
                                    size_t* num_values);

// =============================================================================
// MPS Arithmetic
// =============================================================================

/**
 * @brief Compute inner product of two MPS: ⟨bra|ket⟩
 *
 * @param bra The bra state
 * @param ket The ket state
 * @param result Output: inner product
 * @return QGT_SUCCESS on success
 */
qgt_error_t mps_inner_product(const MatrixProductState* bra,
                              const MatrixProductState* ket,
                              ComplexFloat* result);

/**
 * @brief Compute norm of MPS: √⟨ψ|ψ⟩
 *
 * @param mps The MPS state
 * @param norm Output: norm
 * @return QGT_SUCCESS on success
 */
qgt_error_t mps_norm(const MatrixProductState* mps, double* norm);

/**
 * @brief Normalize MPS to have unit norm
 *
 * @param mps The MPS to normalize
 * @return QGT_SUCCESS on success
 */
qgt_error_t mps_normalize(MatrixProductState* mps);

/**
 * @brief Add two MPS: |result⟩ = |a⟩ + |b⟩
 *
 * The bond dimension of result is sum of bond dimensions of a and b.
 *
 * @param result Output MPS (will be created)
 * @param a First MPS
 * @param b Second MPS
 * @return QGT_SUCCESS on success
 */
qgt_error_t mps_add(MatrixProductState** result,
                    const MatrixProductState* a,
                    const MatrixProductState* b);

/**
 * @brief Scale MPS by a complex scalar
 *
 * @param mps The MPS to scale
 * @param scalar The scaling factor
 * @return QGT_SUCCESS on success
 */
qgt_error_t mps_scale(MatrixProductState* mps, ComplexFloat scalar);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Get the total number of parameters in the MPS
 *
 * @param mps The MPS
 * @return Total number of complex parameters
 */
size_t mps_num_parameters(const MatrixProductState* mps);

/**
 * @brief Get the maximum bond dimension currently in use
 *
 * @param mps The MPS
 * @return Maximum bond dimension
 */
size_t mps_get_max_bond_dim(const MatrixProductState* mps);

/**
 * @brief Check if MPS is in valid canonical form
 *
 * @param mps The MPS to check
 * @param tolerance Tolerance for isometry checks
 * @return true if canonical form is valid
 */
bool mps_verify_canonical_form(const MatrixProductState* mps, double tolerance);

/**
 * @brief Print MPS information for debugging
 *
 * @param mps The MPS to print
 */
void mps_print_info(const MatrixProductState* mps);

#ifdef __cplusplus
}
#endif

#endif // MPS_OPERATIONS_H
