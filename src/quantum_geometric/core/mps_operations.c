/**
 * @file mps_operations.c
 * @brief Matrix Product State (MPS) implementation for tensor network quantum states
 *
 * This file implements a complete MPS library with:
 * - Creation, destruction, and cloning
 * - Initialization from state vectors and product states
 * - Left, right, and mixed canonicalization using QR/SVD
 * - SVD-based truncation and compression
 * - Physical observables (expectation values, correlations, entanglement entropy)
 * - MPS arithmetic (inner products, addition, scaling)
 */

#include "quantum_geometric/core/mps_operations.h"
#include "quantum_geometric/core/lapack_wrapper.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

// =============================================================================
// Internal Helper Functions
// =============================================================================

/**
 * @brief Compute complex inner product of two vectors: <a|b>
 */
static ComplexFloat complex_inner_product(const ComplexFloat* a, const ComplexFloat* b, size_t n) {
    ComplexFloat result = {0.0f, 0.0f};
    for (size_t i = 0; i < n; i++) {
        // <a|b> = sum of a* * b
        result.real += a[i].real * b[i].real + a[i].imag * b[i].imag;
        result.imag += a[i].real * b[i].imag - a[i].imag * b[i].real;
    }
    return result;
}

/**
 * @brief Efficient matrix-matrix multiply: C = A * B using cache blocking
 * A is m x k, B is k x n, C is m x n
 * Uses 64-element cache blocking for better performance
 */
static void matrix_multiply(const ComplexFloat* A, const ComplexFloat* B,
                           ComplexFloat* C, size_t m, size_t k, size_t n) {
    // Initialize C to zero
    memset(C, 0, m * n * sizeof(ComplexFloat));

    // Cache blocking for better performance
    const size_t BLOCK = 32;

    for (size_t ii = 0; ii < m; ii += BLOCK) {
        size_t i_end = (ii + BLOCK < m) ? ii + BLOCK : m;
        for (size_t jj = 0; jj < n; jj += BLOCK) {
            size_t j_end = (jj + BLOCK < n) ? jj + BLOCK : n;
            for (size_t ll = 0; ll < k; ll += BLOCK) {
                size_t l_end = (ll + BLOCK < k) ? ll + BLOCK : k;

                for (size_t i = ii; i < i_end; i++) {
                    for (size_t l = ll; l < l_end; l++) {
                        ComplexFloat a = A[i * k + l];
                        for (size_t j = jj; j < j_end; j++) {
                            ComplexFloat b = B[l * n + j];
                            C[i * n + j].real += a.real * b.real - a.imag * b.imag;
                            C[i * n + j].imag += a.real * b.imag + a.imag * b.real;
                        }
                    }
                }
            }
        }
    }
}

/**
 * @brief Matrix-matrix multiply with conjugate transpose: C = A^H * B
 * A is k x m (so A^H is m x k), B is k x n, C is m x n
 */
static void matrix_multiply_AH_B(const ComplexFloat* A, const ComplexFloat* B,
                                  ComplexFloat* C, size_t m, size_t k, size_t n) {
    memset(C, 0, m * n * sizeof(ComplexFloat));

    const size_t BLOCK = 32;

    for (size_t ii = 0; ii < m; ii += BLOCK) {
        size_t i_end = (ii + BLOCK < m) ? ii + BLOCK : m;
        for (size_t jj = 0; jj < n; jj += BLOCK) {
            size_t j_end = (jj + BLOCK < n) ? jj + BLOCK : n;
            for (size_t ll = 0; ll < k; ll += BLOCK) {
                size_t l_end = (ll + BLOCK < k) ? ll + BLOCK : k;

                for (size_t l = ll; l < l_end; l++) {
                    for (size_t i = ii; i < i_end; i++) {
                        // A^H[i,l] = conj(A[l,i])
                        ComplexFloat a_conj = {A[l * m + i].real, -A[l * m + i].imag};
                        for (size_t j = jj; j < j_end; j++) {
                            ComplexFloat b = B[l * n + j];
                            C[i * n + j].real += a_conj.real * b.real - a_conj.imag * b.imag;
                            C[i * n + j].imag += a_conj.real * b.imag + a_conj.imag * b.real;
                        }
                    }
                }
            }
        }
    }
}

/**
 * @brief Efficient left environment update using O(χ³d) matrix operations
 *
 * Updates: E'[α',α] = Σ_{β,β',σ} E[β',β] A*[β',σ,α'] A[β,σ,α]
 *
 * Efficient algorithm:
 * 1. For each σ: temp[β,α] = E[β',β] @ (reshape of A[β,σ,α])
 * 2. E'[α',α] += A[:,σ,:]^H @ temp
 *
 * Complexity: O(d * χ³) instead of O(d * χ⁴)
 *
 * @param left_env Input environment E of shape (left_dim, left_dim)
 * @param tensor MPS tensor A of shape (left_bond, d, right_bond)
 * @param left_dim Current environment dimension (= left_bond)
 * @param d Physical dimension
 * @param right_bond Right bond dimension
 * @param new_env Output environment E' of shape (right_bond, right_bond)
 */
static void update_left_environment_efficient(
    const ComplexFloat* left_env,
    const ComplexFloat* tensor,
    size_t left_dim,
    size_t d,
    size_t right_bond,
    ComplexFloat* new_env)
{
    // Initialize output to zero
    memset(new_env, 0, right_bond * right_bond * sizeof(ComplexFloat));

    // Allocate temp buffer: (left_dim, right_bond)
    ComplexFloat* temp = calloc(left_dim * right_bond, sizeof(ComplexFloat));
    if (!temp) return;

    // For each physical index σ
    for (size_t sigma = 0; sigma < d; sigma++) {
        // Extract A[:, σ, :] as matrix of shape (left_dim, right_bond)
        // A[β, σ, α] is at index β * d * right_bond + σ * right_bond + α

        // Step 1: temp[β, α] = Σ_{β'} E[β', β] * A[β', σ, α]
        // This is temp = E^T @ A_σ, but we store E in row-major as E[β', β]
        // So temp = E @ A_σ where A_σ is extracted slice

        memset(temp, 0, left_dim * right_bond * sizeof(ComplexFloat));

        for (size_t bp = 0; bp < left_dim; bp++) {
            for (size_t b = 0; b < left_dim; b++) {
                ComplexFloat e = left_env[bp * left_dim + b];
                for (size_t a = 0; a < right_bond; a++) {
                    size_t a_idx = bp * d * right_bond + sigma * right_bond + a;
                    ComplexFloat a_val = tensor[a_idx];
                    temp[b * right_bond + a].real += e.real * a_val.real - e.imag * a_val.imag;
                    temp[b * right_bond + a].imag += e.real * a_val.imag + e.imag * a_val.real;
                }
            }
        }

        // Step 2: E'[α', α] += Σ_β A*[β, σ, α'] * temp[β, α]
        // This is E' += A_σ^H @ temp

        for (size_t b = 0; b < left_dim; b++) {
            for (size_t ap = 0; ap < right_bond; ap++) {
                size_t a_idx = b * d * right_bond + sigma * right_bond + ap;
                ComplexFloat a_conj = {tensor[a_idx].real, -tensor[a_idx].imag};
                for (size_t a = 0; a < right_bond; a++) {
                    ComplexFloat t = temp[b * right_bond + a];
                    new_env[ap * right_bond + a].real += a_conj.real * t.real - a_conj.imag * t.imag;
                    new_env[ap * right_bond + a].imag += a_conj.real * t.imag + a_conj.imag * t.real;
                }
            }
        }
    }

    free(temp);
}

/**
 * @brief Efficient right environment update using O(χ³d) matrix operations
 *
 * Updates: E'[β',β] = Σ_{α,α',σ} A*[β',σ,α'] E[α',α] A[β,σ,α]
 *
 * @param right_env Input environment E of shape (right_dim, right_dim)
 * @param tensor MPS tensor A of shape (left_bond, d, right_bond)
 * @param left_bond Left bond dimension
 * @param d Physical dimension
 * @param right_dim Current environment dimension (= right_bond)
 * @param new_env Output environment E' of shape (left_bond, left_bond)
 */
static void update_right_environment_efficient(
    const ComplexFloat* right_env,
    const ComplexFloat* tensor,
    size_t left_bond,
    size_t d,
    size_t right_dim,
    ComplexFloat* new_env)
{
    memset(new_env, 0, left_bond * left_bond * sizeof(ComplexFloat));

    ComplexFloat* temp = calloc(left_bond * right_dim, sizeof(ComplexFloat));
    if (!temp) return;

    for (size_t sigma = 0; sigma < d; sigma++) {
        // Step 1: temp[β, α] = Σ_{α'} A[β, σ, α'] * E[α', α]
        memset(temp, 0, left_bond * right_dim * sizeof(ComplexFloat));

        for (size_t b = 0; b < left_bond; b++) {
            for (size_t ap = 0; ap < right_dim; ap++) {
                size_t a_idx = b * d * right_dim + sigma * right_dim + ap;
                ComplexFloat a_val = tensor[a_idx];
                for (size_t a = 0; a < right_dim; a++) {
                    ComplexFloat e = right_env[ap * right_dim + a];
                    temp[b * right_dim + a].real += a_val.real * e.real - a_val.imag * e.imag;
                    temp[b * right_dim + a].imag += a_val.real * e.imag + a_val.imag * e.real;
                }
            }
        }

        // Step 2: E'[β', β] += Σ_α A*[β', σ, α] * temp[β, α]
        for (size_t bp = 0; bp < left_bond; bp++) {
            for (size_t a = 0; a < right_dim; a++) {
                size_t a_idx = bp * d * right_dim + sigma * right_dim + a;
                ComplexFloat a_conj = {tensor[a_idx].real, -tensor[a_idx].imag};
                for (size_t b = 0; b < left_bond; b++) {
                    ComplexFloat t = temp[b * right_dim + a];
                    new_env[bp * left_bond + b].real += a_conj.real * t.real - a_conj.imag * t.imag;
                    new_env[bp * left_bond + b].imag += a_conj.real * t.imag + a_conj.imag * t.real;
                }
            }
        }
    }

    free(temp);
}

/**
 * @brief Efficient left environment update with local operator
 *
 * Updates: E'[α',α] = Σ_{β,β',σ,σ'} E[β',β] A*[β',σ',α'] O[σ',σ] A[β,σ,α]
 */
static void update_left_environment_with_op(
    const ComplexFloat* left_env,
    const ComplexFloat* tensor,
    const ComplexFloat* op,  // d x d operator
    size_t left_dim,
    size_t d,
    size_t right_bond,
    ComplexFloat* new_env)
{
    memset(new_env, 0, right_bond * right_bond * sizeof(ComplexFloat));

    // temp1[β, σ, α] = Σ_{β'} E[β', β] * A[β', σ, α]
    ComplexFloat* temp1 = calloc(left_dim * d * right_bond, sizeof(ComplexFloat));
    // temp2[β, σ', α] = Σ_σ temp1[β, σ, α] * O^T[σ, σ'] = Σ_σ temp1[β, σ, α] * O[σ', σ]
    ComplexFloat* temp2 = calloc(left_dim * d * right_bond, sizeof(ComplexFloat));

    if (!temp1 || !temp2) {
        free(temp1);
        free(temp2);
        return;
    }

    // Step 1: temp1 = E @ A (contract left index)
    for (size_t bp = 0; bp < left_dim; bp++) {
        for (size_t b = 0; b < left_dim; b++) {
            ComplexFloat e = left_env[bp * left_dim + b];
            for (size_t sigma = 0; sigma < d; sigma++) {
                for (size_t a = 0; a < right_bond; a++) {
                    size_t a_idx = bp * d * right_bond + sigma * right_bond + a;
                    size_t t_idx = b * d * right_bond + sigma * right_bond + a;
                    ComplexFloat a_val = tensor[a_idx];
                    temp1[t_idx].real += e.real * a_val.real - e.imag * a_val.imag;
                    temp1[t_idx].imag += e.real * a_val.imag + e.imag * a_val.real;
                }
            }
        }
    }

    // Step 2: temp2 = temp1 with operator applied
    for (size_t b = 0; b < left_dim; b++) {
        for (size_t sp = 0; sp < d; sp++) {
            for (size_t s = 0; s < d; s++) {
                ComplexFloat o = op[sp * d + s];
                for (size_t a = 0; a < right_bond; a++) {
                    size_t t1_idx = b * d * right_bond + s * right_bond + a;
                    size_t t2_idx = b * d * right_bond + sp * right_bond + a;
                    ComplexFloat t = temp1[t1_idx];
                    temp2[t2_idx].real += o.real * t.real - o.imag * t.imag;
                    temp2[t2_idx].imag += o.real * t.imag + o.imag * t.real;
                }
            }
        }
    }

    // Step 3: E' = A^H @ temp2 (contract left and physical indices)
    for (size_t b = 0; b < left_dim; b++) {
        for (size_t sigma = 0; sigma < d; sigma++) {
            for (size_t ap = 0; ap < right_bond; ap++) {
                size_t a_idx = b * d * right_bond + sigma * right_bond + ap;
                ComplexFloat a_conj = {tensor[a_idx].real, -tensor[a_idx].imag};
                for (size_t a = 0; a < right_bond; a++) {
                    size_t t_idx = b * d * right_bond + sigma * right_bond + a;
                    ComplexFloat t = temp2[t_idx];
                    new_env[ap * right_bond + a].real += a_conj.real * t.real - a_conj.imag * t.imag;
                    new_env[ap * right_bond + a].imag += a_conj.real * t.imag + a_conj.imag * t.real;
                }
            }
        }
    }

    free(temp1);
    free(temp2);
}

/**
 * @brief Reshape a tensor for left canonicalization
 * Input: tensor of shape (left_bond, phys, right_bond)
 * Output: matrix of shape (left_bond * phys, right_bond)
 * This is just a reinterpretation of memory layout
 */
static void reshape_for_left_qr(const ComplexFloat* tensor,
                               size_t left_bond, size_t phys_dim, size_t right_bond,
                               ComplexFloat* matrix) {
    // Already in correct layout for row-major storage
    memcpy(matrix, tensor, left_bond * phys_dim * right_bond * sizeof(ComplexFloat));
}

/**
 * @brief Reshape a tensor for right canonicalization
 * Input: tensor of shape (left_bond, phys, right_bond)
 * Output: matrix of shape (left_bond, phys * right_bond)
 * Need to transpose for the RQ factorization
 */
static void reshape_for_right_qr(const ComplexFloat* tensor,
                                size_t left_bond, size_t phys_dim, size_t right_bond,
                                ComplexFloat* matrix) {
    // Reshape to (left_bond, phys * right_bond)
    // This is just reinterpretation - same memory layout
    memcpy(matrix, tensor, left_bond * phys_dim * right_bond * sizeof(ComplexFloat));
}

// =============================================================================
// Creation and Destruction
// =============================================================================

MatrixProductState* mps_create(size_t num_sites, size_t physical_dim, size_t max_bond_dim) {
    if (num_sites == 0 || physical_dim == 0 || max_bond_dim == 0) {
        return NULL;
    }

    MatrixProductState* mps = calloc(1, sizeof(MatrixProductState));
    if (!mps) return NULL;

    mps->num_sites = num_sites;
    mps->physical_dim = physical_dim;
    mps->max_bond_dim = max_bond_dim;
    mps->form = MPS_CANONICAL_NONE;
    mps->orthogonality_center = 0;
    mps->is_normalized = false;

    // Allocate arrays for bond dimensions
    mps->left_bond_dims = calloc(num_sites, sizeof(size_t));
    mps->right_bond_dims = calloc(num_sites, sizeof(size_t));
    mps->tensor_sizes = calloc(num_sites, sizeof(size_t));
    mps->tensors = calloc(num_sites, sizeof(ComplexFloat*));

    if (!mps->left_bond_dims || !mps->right_bond_dims ||
        !mps->tensor_sizes || !mps->tensors) {
        mps_destroy(mps);
        return NULL;
    }

    // Initialize bond dimensions for a product state (all bonds = 1)
    for (size_t i = 0; i < num_sites; i++) {
        mps->left_bond_dims[i] = 1;
        mps->right_bond_dims[i] = 1;
        mps->tensor_sizes[i] = physical_dim;  // 1 * d * 1

        mps->tensors[i] = calloc(physical_dim, sizeof(ComplexFloat));
        if (!mps->tensors[i]) {
            mps_destroy(mps);
            return NULL;
        }
    }

    return mps;
}

void mps_destroy(MatrixProductState* mps) {
    if (!mps) return;

    if (mps->tensors) {
        for (size_t i = 0; i < mps->num_sites; i++) {
            free(mps->tensors[i]);
        }
        free(mps->tensors);
    }

    free(mps->left_bond_dims);
    free(mps->right_bond_dims);
    free(mps->tensor_sizes);
    free(mps);
}

MatrixProductState* mps_clone(const MatrixProductState* src) {
    if (!src) return NULL;

    MatrixProductState* dst = mps_create(src->num_sites, src->physical_dim, src->max_bond_dim);
    if (!dst) return NULL;

    dst->form = src->form;
    dst->orthogonality_center = src->orthogonality_center;
    dst->is_normalized = src->is_normalized;

    for (size_t i = 0; i < src->num_sites; i++) {
        dst->left_bond_dims[i] = src->left_bond_dims[i];
        dst->right_bond_dims[i] = src->right_bond_dims[i];
        dst->tensor_sizes[i] = src->tensor_sizes[i];

        // Reallocate if needed
        free(dst->tensors[i]);
        dst->tensors[i] = malloc(src->tensor_sizes[i] * sizeof(ComplexFloat));
        if (!dst->tensors[i]) {
            mps_destroy(dst);
            return NULL;
        }
        memcpy(dst->tensors[i], src->tensors[i], src->tensor_sizes[i] * sizeof(ComplexFloat));
    }

    return dst;
}

// =============================================================================
// Initialization
// =============================================================================

qgt_error_t mps_initialize_zero_state(MatrixProductState* mps) {
    if (!mps) return QGT_ERROR_INVALID_ARGUMENT;

    // Initialize each site to |0⟩
    for (size_t i = 0; i < mps->num_sites; i++) {
        mps->left_bond_dims[i] = 1;
        mps->right_bond_dims[i] = 1;
        mps->tensor_sizes[i] = mps->physical_dim;

        free(mps->tensors[i]);
        mps->tensors[i] = calloc(mps->physical_dim, sizeof(ComplexFloat));
        if (!mps->tensors[i]) return QGT_ERROR_MEMORY_ALLOCATION;

        // |0⟩ state: first component = 1, rest = 0
        mps->tensors[i][0].real = 1.0f;
        mps->tensors[i][0].imag = 0.0f;
    }

    mps->form = MPS_CANONICAL_LEFT;  // Product state is trivially canonical
    mps->is_normalized = true;
    return QGT_SUCCESS;
}

qgt_error_t mps_initialize_product_state(MatrixProductState* mps, const int* local_states) {
    if (!mps || !local_states) return QGT_ERROR_INVALID_ARGUMENT;

    for (size_t i = 0; i < mps->num_sites; i++) {
        if ((size_t)local_states[i] >= mps->physical_dim) {
            return QGT_ERROR_INVALID_ARGUMENT;
        }

        mps->left_bond_dims[i] = 1;
        mps->right_bond_dims[i] = 1;
        mps->tensor_sizes[i] = mps->physical_dim;

        free(mps->tensors[i]);
        mps->tensors[i] = calloc(mps->physical_dim, sizeof(ComplexFloat));
        if (!mps->tensors[i]) return QGT_ERROR_MEMORY_ALLOCATION;

        mps->tensors[i][local_states[i]].real = 1.0f;
    }

    mps->form = MPS_CANONICAL_LEFT;
    mps->is_normalized = true;
    return QGT_SUCCESS;
}

qgt_error_t mps_initialize_random(MatrixProductState* mps, size_t bond_dim) {
    if (!mps || bond_dim == 0) return QGT_ERROR_INVALID_ARGUMENT;

    size_t actual_bond = (bond_dim > mps->max_bond_dim) ? mps->max_bond_dim : bond_dim;

    for (size_t i = 0; i < mps->num_sites; i++) {
        // Set bond dimensions respecting boundaries
        size_t left_bond = (i == 0) ? 1 : actual_bond;
        size_t right_bond = (i == mps->num_sites - 1) ? 1 : actual_bond;

        mps->left_bond_dims[i] = left_bond;
        mps->right_bond_dims[i] = right_bond;

        size_t size = left_bond * mps->physical_dim * right_bond;
        mps->tensor_sizes[i] = size;

        free(mps->tensors[i]);
        mps->tensors[i] = malloc(size * sizeof(ComplexFloat));
        if (!mps->tensors[i]) return QGT_ERROR_MEMORY_ALLOCATION;

        // Random initialization
        for (size_t j = 0; j < size; j++) {
            mps->tensors[i][j].real = (float)(rand() % 1000 - 500) / 500.0f;
            mps->tensors[i][j].imag = (float)(rand() % 1000 - 500) / 500.0f;
        }
    }

    mps->form = MPS_CANONICAL_NONE;
    mps->is_normalized = false;
    return QGT_SUCCESS;
}

qgt_error_t mps_from_state_vector(MatrixProductState* mps,
                                  const ComplexFloat* state,
                                  size_t max_bond_dim) {
    if (!mps || !state) return QGT_ERROR_INVALID_ARGUMENT;

    size_t N = mps->num_sites;
    size_t d = mps->physical_dim;
    size_t total_dim = 1;
    for (size_t i = 0; i < N; i++) {
        total_dim *= d;
    }

    // We'll do SVD decomposition from left to right
    // This produces left-canonical form

    // Current matrix to decompose: starts as reshaped state vector
    size_t left_dim = 1;
    size_t right_dim = total_dim;

    ComplexFloat* current_matrix = malloc(total_dim * sizeof(ComplexFloat));
    if (!current_matrix) return QGT_ERROR_MEMORY_ALLOCATION;
    memcpy(current_matrix, state, total_dim * sizeof(ComplexFloat));

    for (size_t site = 0; site < N; site++) {
        right_dim /= d;

        // Reshape to (left_dim * d, right_dim)
        size_t m = left_dim * d;
        size_t n = right_dim;
        size_t k = (m < n) ? m : n;  // Number of singular values
        if (max_bond_dim > 0 && k > max_bond_dim) {
            k = max_bond_dim;
        }

        // Allocate SVD outputs
        ComplexFloat* U = malloc(m * k * sizeof(ComplexFloat));
        float* S = malloc(k * sizeof(float));
        ComplexFloat* Vt = malloc(k * n * sizeof(ComplexFloat));

        if (!U || !S || !Vt) {
            free(U); free(S); free(Vt);
            free(current_matrix);
            return QGT_ERROR_MEMORY_ALLOCATION;
        }

        // Compute truncated SVD: current_matrix ≈ U * S * Vt
        if (!lapack_svd(current_matrix, m, n, U, S, Vt, 101)) {
            free(U); free(S); free(Vt);
            free(current_matrix);
            return QGT_ERROR_SVD_FAILED;
        }

        // Store U as the MPS tensor for this site
        // Reshape U from (left_dim * d, k) to (left_dim, d, k)
        mps->left_bond_dims[site] = left_dim;
        mps->right_bond_dims[site] = k;
        mps->tensor_sizes[site] = left_dim * d * k;

        free(mps->tensors[site]);
        mps->tensors[site] = malloc(mps->tensor_sizes[site] * sizeof(ComplexFloat));
        if (!mps->tensors[site]) {
            free(U); free(S); free(Vt);
            free(current_matrix);
            return QGT_ERROR_MEMORY_ALLOCATION;
        }
        memcpy(mps->tensors[site], U, mps->tensor_sizes[site] * sizeof(ComplexFloat));

        // Prepare next matrix: S * Vt
        free(current_matrix);
        current_matrix = malloc(k * n * sizeof(ComplexFloat));
        if (!current_matrix) {
            free(U); free(S); free(Vt);
            return QGT_ERROR_MEMORY_ALLOCATION;
        }

        // Multiply S * Vt (absorb singular values into right part)
        for (size_t i = 0; i < k; i++) {
            for (size_t j = 0; j < n; j++) {
                current_matrix[i * n + j].real = S[i] * Vt[i * n + j].real;
                current_matrix[i * n + j].imag = S[i] * Vt[i * n + j].imag;
            }
        }

        left_dim = k;
        free(U); free(S); free(Vt);
    }

    free(current_matrix);

    mps->form = MPS_CANONICAL_LEFT;
    mps->is_normalized = true;
    return QGT_SUCCESS;
}

qgt_error_t mps_to_state_vector(const MatrixProductState* mps, ComplexFloat* state) {
    if (!mps || !state) return QGT_ERROR_INVALID_ARGUMENT;

    size_t N = mps->num_sites;
    size_t d = mps->physical_dim;
    size_t total_dim = 1;
    for (size_t i = 0; i < N; i++) {
        total_dim *= d;
    }

    // Initialize state to zero
    memset(state, 0, total_dim * sizeof(ComplexFloat));

    // Contract all tensors by iterating over physical indices
    // This is O(d^N * χ²) which is exponential but unavoidable

    for (size_t idx = 0; idx < total_dim; idx++) {
        // Extract physical indices for each site
        size_t* phys_indices = malloc(N * sizeof(size_t));
        if (!phys_indices) return QGT_ERROR_MEMORY_ALLOCATION;

        size_t temp_idx = idx;
        for (size_t i = N; i > 0; i--) {
            phys_indices[i-1] = temp_idx % d;
            temp_idx /= d;
        }

        // Contract: start with identity matrix
        size_t current_dim = 1;
        ComplexFloat* current_vector = malloc(sizeof(ComplexFloat));
        current_vector[0].real = 1.0f;
        current_vector[0].imag = 0.0f;

        for (size_t site = 0; site < N; site++) {
            size_t left_bond = mps->left_bond_dims[site];
            size_t right_bond = mps->right_bond_dims[site];
            size_t phys = phys_indices[site];

            // Validate MPS structure: current_dim should match left_bond
            if (current_dim != left_bond) {
                free(current_vector);
                free(phys_indices);
                return QGT_ERROR_INVALID_ARGUMENT;  // MPS bond dimensions inconsistent
            }

            // Extract the slice for this physical index
            // Tensor layout: [left_bond][phys_dim][right_bond]
            // We want the matrix at physical index phys

            ComplexFloat* new_vector = calloc(right_bond, sizeof(ComplexFloat));
            if (!new_vector) {
                free(current_vector);
                free(phys_indices);
                return QGT_ERROR_MEMORY_ALLOCATION;
            }

            // Multiply current_vector by the matrix slice
            for (size_t r = 0; r < right_bond; r++) {
                for (size_t l = 0; l < left_bond; l++) {
                    // Index into tensor: l * d * right_bond + phys * right_bond + r
                    size_t tensor_idx = l * d * right_bond + phys * right_bond + r;
                    ComplexFloat a = current_vector[l];
                    ComplexFloat b = mps->tensors[site][tensor_idx];
                    new_vector[r].real += a.real * b.real - a.imag * b.imag;
                    new_vector[r].imag += a.real * b.imag + a.imag * b.real;
                }
            }

            free(current_vector);
            current_vector = new_vector;
            current_dim = right_bond;
        }

        // Final vector should be 1-dimensional
        state[idx] = current_vector[0];
        free(current_vector);
        free(phys_indices);
    }

    return QGT_SUCCESS;
}

// =============================================================================
// Canonicalization
// =============================================================================

/**
 * @brief Left-canonicalize a single site using QR decomposition
 *
 * Reshapes A[site] to (χ_L * d, χ_R) and computes QR = A[site]
 * The Q matrix becomes the new A[site] (left-canonical)
 * R is absorbed into A[site+1]
 */
static qgt_error_t canonicalize_site_left(MatrixProductState* mps, size_t site) {
    if (site >= mps->num_sites - 1) return QGT_ERROR_INVALID_ARGUMENT;

    size_t left_bond = mps->left_bond_dims[site];
    size_t d = mps->physical_dim;
    size_t right_bond = mps->right_bond_dims[site];

    size_t m = left_bond * d;
    size_t n = right_bond;
    size_t k = (m < n) ? m : n;

    // Allocate for QR decomposition
    ComplexFloat* Q = malloc(m * k * sizeof(ComplexFloat));
    ComplexFloat* R = malloc(k * n * sizeof(ComplexFloat));
    if (!Q || !R) {
        free(Q); free(R);
        return QGT_ERROR_MEMORY_ALLOCATION;
    }

    // Compute QR: A = Q * R
    if (!lapack_qr(mps->tensors[site], m, n, Q, R, 101)) {
        free(Q); free(R);
        return QGT_ERROR_QR_FAILED;
    }

    // Update current tensor with Q
    mps->right_bond_dims[site] = k;
    mps->tensor_sizes[site] = left_bond * d * k;
    free(mps->tensors[site]);
    mps->tensors[site] = Q;

    // Multiply R into the next tensor
    size_t next_left = mps->left_bond_dims[site + 1];
    size_t next_right = mps->right_bond_dims[site + 1];
    size_t next_size = k * d * next_right;

    ComplexFloat* new_next = malloc(next_size * sizeof(ComplexFloat));
    if (!new_next) {
        free(R);
        return QGT_ERROR_MEMORY_ALLOCATION;
    }

    // Contract R with next tensor
    // R: (k, next_left=right_bond), next: (next_left, d, next_right)
    // Result: (k, d, next_right)
    matrix_multiply(R, mps->tensors[site + 1], new_next, k, next_left, d * next_right);

    free(R);
    free(mps->tensors[site + 1]);
    mps->tensors[site + 1] = new_next;
    mps->left_bond_dims[site + 1] = k;
    mps->tensor_sizes[site + 1] = next_size;

    return QGT_SUCCESS;
}

/**
 * @brief Right-canonicalize a single site using LQ decomposition
 *
 * Reshapes A[site] to (χ_L, d * χ_R) and computes LQ = A[site]
 * The Q matrix becomes the new A[site] (right-canonical)
 * L is absorbed into A[site-1]
 */
static qgt_error_t canonicalize_site_right(MatrixProductState* mps, size_t site) {
    if (site == 0) return QGT_ERROR_INVALID_ARGUMENT;

    size_t left_bond = mps->left_bond_dims[site];
    size_t d = mps->physical_dim;
    size_t right_bond = mps->right_bond_dims[site];

    size_t m = left_bond;
    size_t n = d * right_bond;
    size_t k = (m < n) ? m : n;

    // For LQ, we compute QR of the transpose, then transpose back
    // A = L * Q where L is lower triangular
    // We use RQ: A^T = Q^T * R^T, so A = R * Q where R is upper triangular

    // Transpose the tensor
    ComplexFloat* transposed = malloc(n * m * sizeof(ComplexFloat));
    if (!transposed) return QGT_ERROR_MEMORY_ALLOCATION;

    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            transposed[j * m + i] = mps->tensors[site][i * n + j];
        }
    }

    // QR of transposed
    ComplexFloat* Q = malloc(n * k * sizeof(ComplexFloat));
    ComplexFloat* R = malloc(k * m * sizeof(ComplexFloat));
    if (!Q || !R) {
        free(transposed); free(Q); free(R);
        return QGT_ERROR_MEMORY_ALLOCATION;
    }

    if (!lapack_qr(transposed, n, m, Q, R, 101)) {
        free(transposed); free(Q); free(R);
        return QGT_ERROR_QR_FAILED;
    }
    free(transposed);

    // Transpose Q back to get the right-canonical tensor
    ComplexFloat* Q_transposed = malloc(k * n * sizeof(ComplexFloat));
    if (!Q_transposed) {
        free(Q); free(R);
        return QGT_ERROR_MEMORY_ALLOCATION;
    }

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < k; j++) {
            Q_transposed[j * n + i] = Q[i * k + j];
        }
    }
    free(Q);

    // Transpose R to get L (which goes into previous tensor)
    ComplexFloat* L = malloc(m * k * sizeof(ComplexFloat));
    if (!L) {
        free(Q_transposed); free(R);
        return QGT_ERROR_MEMORY_ALLOCATION;
    }

    for (size_t i = 0; i < k; i++) {
        for (size_t j = 0; j < m; j++) {
            L[j * k + i] = R[i * m + j];
        }
    }
    free(R);

    // Update current tensor with Q_transposed
    mps->left_bond_dims[site] = k;
    mps->tensor_sizes[site] = k * d * right_bond;
    free(mps->tensors[site]);
    mps->tensors[site] = Q_transposed;

    // Multiply L into the previous tensor
    size_t prev_left = mps->left_bond_dims[site - 1];
    size_t prev_right = mps->right_bond_dims[site - 1];  // = left_bond = m
    size_t prev_size = prev_left * d * k;

    // Validate MPS bond consistency: previous right bond must match current left bond
    if (prev_right != m) {
        free(L);
        return QGT_ERROR_INVALID_ARGUMENT;  // Bond dimension mismatch
    }

    ComplexFloat* new_prev = malloc(prev_size * sizeof(ComplexFloat));
    if (!new_prev) {
        free(L);
        return QGT_ERROR_MEMORY_ALLOCATION;
    }

    // Contract previous tensor with L
    // prev: (prev_left, d, prev_right=m), L: (m, k)
    // Result: (prev_left, d, k)
    // Reshape prev to (prev_left * d, prev_right) for matrix multiply
    matrix_multiply(mps->tensors[site - 1], L, new_prev, prev_left * d, prev_right, k);

    free(L);
    free(mps->tensors[site - 1]);
    mps->tensors[site - 1] = new_prev;
    mps->right_bond_dims[site - 1] = k;
    mps->tensor_sizes[site - 1] = prev_size;

    return QGT_SUCCESS;
}

qgt_error_t mps_left_canonicalize(MatrixProductState* mps) {
    if (!mps) return QGT_ERROR_INVALID_ARGUMENT;

    // Sweep from left to right
    for (size_t site = 0; site < mps->num_sites - 1; site++) {
        qgt_error_t err = canonicalize_site_left(mps, site);
        if (err != QGT_SUCCESS) return err;
    }

    mps->form = MPS_CANONICAL_LEFT;
    return QGT_SUCCESS;
}

qgt_error_t mps_right_canonicalize(MatrixProductState* mps) {
    if (!mps) return QGT_ERROR_INVALID_ARGUMENT;

    // Sweep from right to left
    for (size_t site = mps->num_sites - 1; site > 0; site--) {
        qgt_error_t err = canonicalize_site_right(mps, site);
        if (err != QGT_SUCCESS) return err;
    }

    mps->form = MPS_CANONICAL_RIGHT;
    return QGT_SUCCESS;
}

qgt_error_t mps_mixed_canonicalize(MatrixProductState* mps, size_t center) {
    if (!mps || center >= mps->num_sites) return QGT_ERROR_INVALID_ARGUMENT;

    // Left-canonicalize from 0 to center-1
    for (size_t site = 0; site < center; site++) {
        qgt_error_t err = canonicalize_site_left(mps, site);
        if (err != QGT_SUCCESS) return err;
    }

    // Right-canonicalize from N-1 to center+1
    for (size_t site = mps->num_sites - 1; site > center; site--) {
        qgt_error_t err = canonicalize_site_right(mps, site);
        if (err != QGT_SUCCESS) return err;
    }

    mps->form = MPS_CANONICAL_MIXED;
    mps->orthogonality_center = center;
    return QGT_SUCCESS;
}

qgt_error_t mps_move_orthogonality_center(MatrixProductState* mps, size_t new_center) {
    if (!mps || new_center >= mps->num_sites) return QGT_ERROR_INVALID_ARGUMENT;
    if (mps->form != MPS_CANONICAL_MIXED) return QGT_ERROR_INVALID_STATE;

    size_t current = mps->orthogonality_center;

    if (new_center > current) {
        // Move right: left-canonicalize sites from current to new_center-1
        for (size_t site = current; site < new_center; site++) {
            qgt_error_t err = canonicalize_site_left(mps, site);
            if (err != QGT_SUCCESS) return err;
        }
    } else if (new_center < current) {
        // Move left: right-canonicalize sites from current to new_center+1
        for (size_t site = current; site > new_center; site--) {
            qgt_error_t err = canonicalize_site_right(mps, site);
            if (err != QGT_SUCCESS) return err;
        }
    }

    mps->orthogonality_center = new_center;
    return QGT_SUCCESS;
}

// =============================================================================
// SVD and Truncation
// =============================================================================

qgt_error_t mps_truncate(MatrixProductState* mps,
                         size_t max_bond_dim,
                         double cutoff,
                         double* truncation_error) {
    if (!mps) return QGT_ERROR_INVALID_ARGUMENT;

    double total_truncation = 0.0;

    // Put in left-canonical form first
    qgt_error_t err = mps_left_canonicalize(mps);
    if (err != QGT_SUCCESS) return err;

    // Sweep right-to-left with SVD truncation
    for (size_t site = mps->num_sites - 1; site > 0; site--) {
        size_t left_bond = mps->left_bond_dims[site];
        size_t d = mps->physical_dim;
        size_t right_bond = mps->right_bond_dims[site];

        size_t m = left_bond;
        size_t n = d * right_bond;
        size_t full_k = (m < n) ? m : n;

        // Allocate for SVD
        ComplexFloat* U = malloc(m * full_k * sizeof(ComplexFloat));
        float* S = malloc(full_k * sizeof(float));
        ComplexFloat* Vt = malloc(full_k * n * sizeof(ComplexFloat));

        if (!U || !S || !Vt) {
            free(U); free(S); free(Vt);
            return QGT_ERROR_MEMORY_ALLOCATION;
        }

        // Reshape tensor to (left_bond, d * right_bond) - already in this layout
        if (!lapack_svd(mps->tensors[site], m, n, U, S, Vt, 101)) {
            free(U); free(S); free(Vt);
            return QGT_ERROR_SVD_FAILED;
        }

        // Determine truncation: keep singular values above cutoff and up to max_bond_dim
        size_t new_k = 0;
        float total_weight = 0.0f;
        for (size_t i = 0; i < full_k; i++) {
            total_weight += S[i] * S[i];
        }

        float kept_weight = 0.0f;
        for (size_t i = 0; i < full_k; i++) {
            if (new_k >= max_bond_dim) break;
            if (i > 0 && S[i] / S[0] < cutoff) break;
            kept_weight += S[i] * S[i];
            new_k++;
        }

        if (new_k == 0) new_k = 1;  // Keep at least one singular value

        total_truncation += (total_weight - kept_weight) / total_weight;

        // Update current tensor with Vt (reshape to proper form)
        mps->left_bond_dims[site] = new_k;
        mps->tensor_sizes[site] = new_k * d * right_bond;
        free(mps->tensors[site]);
        mps->tensors[site] = malloc(mps->tensor_sizes[site] * sizeof(ComplexFloat));
        if (!mps->tensors[site]) {
            free(U); free(S); free(Vt);
            return QGT_ERROR_MEMORY_ALLOCATION;
        }

        // Copy Vt (only first new_k rows)
        memcpy(mps->tensors[site], Vt, new_k * n * sizeof(ComplexFloat));

        // Multiply U * S into previous tensor
        size_t prev_left = mps->left_bond_dims[site - 1];
        size_t prev_d = mps->physical_dim;
        size_t prev_size = prev_left * prev_d * new_k;

        ComplexFloat* US = malloc(m * new_k * sizeof(ComplexFloat));
        if (!US) {
            free(U); free(S); free(Vt);
            return QGT_ERROR_MEMORY_ALLOCATION;
        }

        // Compute U * S (S is diagonal)
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < new_k; j++) {
                US[i * new_k + j].real = U[i * full_k + j].real * S[j];
                US[i * new_k + j].imag = U[i * full_k + j].imag * S[j];
            }
        }

        // Multiply into previous tensor
        ComplexFloat* new_prev = malloc(prev_size * sizeof(ComplexFloat));
        if (!new_prev) {
            free(U); free(S); free(Vt); free(US);
            return QGT_ERROR_MEMORY_ALLOCATION;
        }

        matrix_multiply(mps->tensors[site - 1], US, new_prev, prev_left * prev_d, m, new_k);

        free(mps->tensors[site - 1]);
        mps->tensors[site - 1] = new_prev;
        mps->right_bond_dims[site - 1] = new_k;
        mps->tensor_sizes[site - 1] = prev_size;

        free(U); free(S); free(Vt); free(US);
    }

    if (truncation_error) {
        *truncation_error = total_truncation;
    }

    mps->form = MPS_CANONICAL_RIGHT;
    return QGT_SUCCESS;
}

qgt_error_t mps_compress(MatrixProductState* mps, double target_fidelity) {
    if (!mps || target_fidelity <= 0.0 || target_fidelity > 1.0) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // Binary search for the right cutoff to achieve target fidelity
    double low_cutoff = 1e-16;
    double high_cutoff = 1.0;
    double truncation_error;

    for (int iter = 0; iter < 20; iter++) {
        double mid_cutoff = sqrt(low_cutoff * high_cutoff);

        MatrixProductState* test = mps_clone(mps);
        if (!test) return QGT_ERROR_MEMORY_ALLOCATION;

        mps_truncate(test, mps->max_bond_dim, mid_cutoff, &truncation_error);

        double achieved_fidelity = 1.0 - truncation_error;

        mps_destroy(test);

        if (achieved_fidelity >= target_fidelity) {
            high_cutoff = mid_cutoff;
        } else {
            low_cutoff = mid_cutoff;
        }

        if (fabs(achieved_fidelity - target_fidelity) < 1e-6) break;
    }

    return mps_truncate(mps, mps->max_bond_dim, high_cutoff, &truncation_error);
}

// =============================================================================
// Physical Observables
// =============================================================================

qgt_error_t mps_expectation_local(const MatrixProductState* mps,
                                  const ComplexFloat* local_op,
                                  size_t site,
                                  ComplexFloat* result) {
    if (!mps || !local_op || !result || site >= mps->num_sites) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    size_t d = mps->physical_dim;

    // Efficient O(N * d * χ³) algorithm using matrix-based environment updates

    // Initialize left environment: identity matrix of size 1
    size_t left_dim = 1;
    ComplexFloat* left_env = malloc(sizeof(ComplexFloat));
    if (!left_env) return QGT_ERROR_MEMORY_ALLOCATION;
    left_env[0].real = 1.0f;
    left_env[0].imag = 0.0f;

    // Contract from left up to site-1 using efficient O(χ³d) updates
    for (size_t s = 0; s < site; s++) {
        size_t rb = mps->right_bond_dims[s];
        ComplexFloat* new_env = malloc(rb * rb * sizeof(ComplexFloat));
        if (!new_env) {
            free(left_env);
            return QGT_ERROR_MEMORY_ALLOCATION;
        }

        update_left_environment_efficient(left_env, mps->tensors[s],
                                          left_dim, d, rb, new_env);

        free(left_env);
        left_env = new_env;
        left_dim = rb;
    }

    // Initialize right environment: identity matrix
    size_t right_dim = 1;
    ComplexFloat* right_env = malloc(sizeof(ComplexFloat));
    if (!right_env) {
        free(left_env);
        return QGT_ERROR_MEMORY_ALLOCATION;
    }
    right_env[0].real = 1.0f;
    right_env[0].imag = 0.0f;

    // Contract from right down to site+1 using efficient O(χ³d) updates
    for (size_t s = mps->num_sites - 1; s > site; s--) {
        size_t lb = mps->left_bond_dims[s];
        ComplexFloat* new_env = malloc(lb * lb * sizeof(ComplexFloat));
        if (!new_env) {
            free(left_env);
            free(right_env);
            return QGT_ERROR_MEMORY_ALLOCATION;
        }

        update_right_environment_efficient(right_env, mps->tensors[s],
                                           lb, d, right_dim, new_env);

        free(right_env);
        right_env = new_env;
        right_dim = lb;
    }

    // Contract at site with the local operator - O(χ² d² + χ² d) operations
    // Using efficient factored contraction
    size_t lb = mps->left_bond_dims[site];
    size_t rb = mps->right_bond_dims[site];

    // Step 1: Contract left env with bra tensor and operator
    // temp[σ', α'] = Σ_{β', σ} L[β', β] A*[β', σ, α'] O[σ', σ] A[β, σ, α] R[α', α]
    // We use the update_left_environment_with_op helper, then contract with right

    ComplexFloat* site_env = malloc(rb * rb * sizeof(ComplexFloat));
    if (!site_env) {
        free(left_env);
        free(right_env);
        return QGT_ERROR_MEMORY_ALLOCATION;
    }

    update_left_environment_with_op(left_env, mps->tensors[site], local_op,
                                    lb, d, rb, site_env);

    // Final contraction: result = Σ_{α',α} site_env[α', α] * right_env[α', α]
    result->real = 0.0f;
    result->imag = 0.0f;

    for (size_t ap = 0; ap < rb; ap++) {
        for (size_t a = 0; a < rb; a++) {
            ComplexFloat se = site_env[ap * rb + a];
            ComplexFloat re = right_env[ap * right_dim + a];
            result->real += se.real * re.real - se.imag * re.imag;
            result->imag += se.real * re.imag + se.imag * re.real;
        }
    }

    free(left_env);
    free(right_env);
    free(site_env);
    return QGT_SUCCESS;
}

qgt_error_t mps_correlation_function(const MatrixProductState* mps,
                                     const ComplexFloat* op_A,
                                     size_t site_A,
                                     const ComplexFloat* op_B,
                                     size_t site_B,
                                     ComplexFloat* result) {
    if (!mps || !op_A || !op_B || !result) return QGT_ERROR_INVALID_ARGUMENT;
    if (site_A >= mps->num_sites || site_B >= mps->num_sites) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // Ensure site_A < site_B
    if (site_A > site_B) {
        size_t temp = site_A;
        site_A = site_B;
        site_B = temp;
        const ComplexFloat* temp_op = op_A;
        op_A = op_B;
        op_B = temp_op;
    }

    size_t d = mps->physical_dim;

    // Efficient O(N * d * χ³) algorithm using matrix-based environment updates

    // Initialize left environment: identity matrix of size 1
    size_t left_dim = 1;
    ComplexFloat* left_env = malloc(sizeof(ComplexFloat));
    if (!left_env) return QGT_ERROR_MEMORY_ALLOCATION;
    left_env[0].real = 1.0f;
    left_env[0].imag = 0.0f;

    // Contract from left to site_A-1 using efficient O(χ³d) updates
    for (size_t s = 0; s < site_A; s++) {
        size_t rb = mps->right_bond_dims[s];
        ComplexFloat* new_env = malloc(rb * rb * sizeof(ComplexFloat));
        if (!new_env) { free(left_env); return QGT_ERROR_MEMORY_ALLOCATION; }

        update_left_environment_efficient(left_env, mps->tensors[s],
                                          left_dim, d, rb, new_env);

        free(left_env);
        left_env = new_env;
        left_dim = rb;
    }

    // Contract at site_A with op_A using efficient update with operator
    {
        size_t lb = mps->left_bond_dims[site_A];
        size_t rb = mps->right_bond_dims[site_A];

        ComplexFloat* new_env = malloc(rb * rb * sizeof(ComplexFloat));
        if (!new_env) { free(left_env); return QGT_ERROR_MEMORY_ALLOCATION; }

        update_left_environment_with_op(left_env, mps->tensors[site_A], op_A,
                                        lb, d, rb, new_env);

        free(left_env);
        left_env = new_env;
        left_dim = rb;
    }

    // Contract from site_A+1 to site_B-1 using efficient O(χ³d) updates
    for (size_t s = site_A + 1; s < site_B; s++) {
        size_t rb = mps->right_bond_dims[s];
        ComplexFloat* new_env = malloc(rb * rb * sizeof(ComplexFloat));
        if (!new_env) { free(left_env); return QGT_ERROR_MEMORY_ALLOCATION; }

        update_left_environment_efficient(left_env, mps->tensors[s],
                                          left_dim, d, rb, new_env);

        free(left_env);
        left_env = new_env;
        left_dim = rb;
    }

    // Contract at site_B with op_B
    {
        size_t lb = mps->left_bond_dims[site_B];
        size_t rb = mps->right_bond_dims[site_B];

        ComplexFloat* new_env = malloc(rb * rb * sizeof(ComplexFloat));
        if (!new_env) { free(left_env); return QGT_ERROR_MEMORY_ALLOCATION; }

        update_left_environment_with_op(left_env, mps->tensors[site_B], op_B,
                                        lb, d, rb, new_env);

        free(left_env);
        left_env = new_env;
        left_dim = rb;
    }

    // Continue to the right edge
    for (size_t s = site_B + 1; s < mps->num_sites; s++) {
        size_t rb = mps->right_bond_dims[s];
        ComplexFloat* new_env = malloc(rb * rb * sizeof(ComplexFloat));
        if (!new_env) { free(left_env); return QGT_ERROR_MEMORY_ALLOCATION; }

        update_left_environment_efficient(left_env, mps->tensors[s],
                                          left_dim, d, rb, new_env);

        free(left_env);
        left_env = new_env;
        left_dim = rb;
    }

    // Final result is the trace of the environment (should be 1x1)
    *result = left_env[0];
    free(left_env);

    return QGT_SUCCESS;
}

qgt_error_t mps_entanglement_entropy(const MatrixProductState* mps,
                                     size_t cut,
                                     double* entropy) {
    if (!mps || !entropy || cut >= mps->num_sites - 1) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // To get the singular values at a cut, we need the MPS to be in
    // mixed canonical form with center at or adjacent to cut

    // Clone and put in mixed canonical form
    MatrixProductState* temp = mps_clone(mps);
    if (!temp) return QGT_ERROR_MEMORY_ALLOCATION;

    qgt_error_t err = mps_mixed_canonicalize(temp, cut);
    if (err != QGT_SUCCESS) {
        mps_destroy(temp);
        return err;
    }

    // Get singular values from the center tensor
    size_t left_bond = temp->left_bond_dims[cut];
    size_t d = temp->physical_dim;
    size_t right_bond = temp->right_bond_dims[cut];

    size_t m = left_bond * d;
    size_t n = right_bond;
    size_t k = (m < n) ? m : n;

    ComplexFloat* U = malloc(m * k * sizeof(ComplexFloat));
    float* S = malloc(k * sizeof(float));
    ComplexFloat* Vt = malloc(k * n * sizeof(ComplexFloat));

    if (!U || !S || !Vt) {
        free(U); free(S); free(Vt);
        mps_destroy(temp);
        return QGT_ERROR_MEMORY_ALLOCATION;
    }

    if (!lapack_svd(temp->tensors[cut], m, n, U, S, Vt, 101)) {
        free(U); free(S); free(Vt);
        mps_destroy(temp);
        return QGT_ERROR_SVD_FAILED;
    }

    // Compute entanglement entropy: S = -Σ λ² log(λ²)
    // where λ are the singular values (Schmidt coefficients)
    double total = 0.0;
    for (size_t i = 0; i < k; i++) {
        total += S[i] * S[i];
    }

    *entropy = 0.0;
    for (size_t i = 0; i < k; i++) {
        double p = (S[i] * S[i]) / total;  // Normalize to probability
        if (p > 1e-15) {
            *entropy -= p * log(p);
        }
    }

    free(U); free(S); free(Vt);
    mps_destroy(temp);
    return QGT_SUCCESS;
}

qgt_error_t mps_get_singular_values(const MatrixProductState* mps,
                                    size_t cut,
                                    double* singular_values,
                                    size_t* num_values) {
    if (!mps || !singular_values || !num_values || cut >= mps->num_sites - 1) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    MatrixProductState* temp = mps_clone(mps);
    if (!temp) return QGT_ERROR_MEMORY_ALLOCATION;

    qgt_error_t err = mps_mixed_canonicalize(temp, cut);
    if (err != QGT_SUCCESS) {
        mps_destroy(temp);
        return err;
    }

    size_t left_bond = temp->left_bond_dims[cut];
    size_t d = temp->physical_dim;
    size_t right_bond = temp->right_bond_dims[cut];

    size_t m = left_bond * d;
    size_t n = right_bond;
    size_t k = (m < n) ? m : n;

    ComplexFloat* U = malloc(m * k * sizeof(ComplexFloat));
    float* S = malloc(k * sizeof(float));
    ComplexFloat* Vt = malloc(k * n * sizeof(ComplexFloat));

    if (!U || !S || !Vt) {
        free(U); free(S); free(Vt);
        mps_destroy(temp);
        return QGT_ERROR_MEMORY_ALLOCATION;
    }

    if (!lapack_svd(temp->tensors[cut], m, n, U, S, Vt, 101)) {
        free(U); free(S); free(Vt);
        mps_destroy(temp);
        return QGT_ERROR_SVD_FAILED;
    }

    *num_values = k;
    for (size_t i = 0; i < k; i++) {
        singular_values[i] = S[i];
    }

    free(U); free(S); free(Vt);
    mps_destroy(temp);
    return QGT_SUCCESS;
}

// =============================================================================
// MPS Arithmetic
// =============================================================================

qgt_error_t mps_inner_product(const MatrixProductState* bra,
                              const MatrixProductState* ket,
                              ComplexFloat* result) {
    if (!bra || !ket || !result) return QGT_ERROR_INVALID_ARGUMENT;
    if (bra->num_sites != ket->num_sites || bra->physical_dim != ket->physical_dim) {
        return QGT_ERROR_DIMENSION_MISMATCH;
    }

    size_t N = bra->num_sites;
    size_t d = bra->physical_dim;

    // Contract from left to right
    // E[α',α] = <bra_α'|ket_α>
    size_t left_dim_bra = 1;
    size_t left_dim_ket = 1;
    ComplexFloat* env = malloc(sizeof(ComplexFloat));
    env[0].real = 1.0f;
    env[0].imag = 0.0f;

    for (size_t site = 0; site < N; site++) {
        size_t rb_bra = bra->right_bond_dims[site];
        size_t rb_ket = ket->right_bond_dims[site];
        size_t new_size = rb_bra * rb_ket;

        ComplexFloat* new_env = calloc(new_size, sizeof(ComplexFloat));
        if (!new_env) {
            free(env);
            return QGT_ERROR_MEMORY_ALLOCATION;
        }

        // new_env[α',α] = Σ_{β',β,σ} env[β',β] * bra*[β',σ,α'] * ket[β,σ,α]
        for (size_t ap = 0; ap < rb_bra; ap++) {
            for (size_t a = 0; a < rb_ket; a++) {
                for (size_t bp = 0; bp < left_dim_bra; bp++) {
                    for (size_t b = 0; b < left_dim_ket; b++) {
                        for (size_t sigma = 0; sigma < d; sigma++) {
                            size_t bra_idx = bp * d * rb_bra + sigma * rb_bra + ap;
                            size_t ket_idx = b * d * rb_ket + sigma * rb_ket + a;

                            ComplexFloat bra_conj = {bra->tensors[site][bra_idx].real,
                                                     -bra->tensors[site][bra_idx].imag};
                            ComplexFloat ket_val = ket->tensors[site][ket_idx];
                            ComplexFloat e = env[bp * left_dim_ket + b];

                            ComplexFloat t1;
                            t1.real = e.real * bra_conj.real - e.imag * bra_conj.imag;
                            t1.imag = e.real * bra_conj.imag + e.imag * bra_conj.real;

                            ComplexFloat contrib;
                            contrib.real = t1.real * ket_val.real - t1.imag * ket_val.imag;
                            contrib.imag = t1.real * ket_val.imag + t1.imag * ket_val.real;

                            new_env[ap * rb_ket + a].real += contrib.real;
                            new_env[ap * rb_ket + a].imag += contrib.imag;
                        }
                    }
                }
            }
        }

        free(env);
        env = new_env;
        left_dim_bra = rb_bra;
        left_dim_ket = rb_ket;
    }

    // Final result should be 1x1
    *result = env[0];
    free(env);
    return QGT_SUCCESS;
}

qgt_error_t mps_norm(const MatrixProductState* mps, double* norm) {
    if (!mps || !norm) return QGT_ERROR_INVALID_ARGUMENT;

    ComplexFloat inner;
    qgt_error_t err = mps_inner_product(mps, mps, &inner);
    if (err != QGT_SUCCESS) return err;

    *norm = sqrt(inner.real);  // Should be real for <ψ|ψ>
    return QGT_SUCCESS;
}

qgt_error_t mps_normalize(MatrixProductState* mps) {
    if (!mps) return QGT_ERROR_INVALID_ARGUMENT;

    double norm;
    qgt_error_t err = mps_norm(mps, &norm);
    if (err != QGT_SUCCESS) return err;

    if (norm < 1e-15) return QGT_ERROR_NUMERICAL_ERROR;

    // Scale the first tensor by 1/norm
    float scale = 1.0f / (float)norm;
    size_t size = mps->tensor_sizes[0];
    for (size_t i = 0; i < size; i++) {
        mps->tensors[0][i].real *= scale;
        mps->tensors[0][i].imag *= scale;
    }

    mps->is_normalized = true;
    return QGT_SUCCESS;
}

qgt_error_t mps_add(MatrixProductState** result,
                    const MatrixProductState* a,
                    const MatrixProductState* b) {
    if (!result || !a || !b) return QGT_ERROR_INVALID_ARGUMENT;
    if (a->num_sites != b->num_sites || a->physical_dim != b->physical_dim) {
        return QGT_ERROR_DIMENSION_MISMATCH;
    }

    size_t N = a->num_sites;
    size_t d = a->physical_dim;

    // Create result MPS with sum of bond dimensions
    size_t max_bond = a->max_bond_dim + b->max_bond_dim;
    *result = mps_create(N, d, max_bond);
    if (!*result) return QGT_ERROR_MEMORY_ALLOCATION;

    for (size_t site = 0; site < N; site++) {
        size_t la = a->left_bond_dims[site];
        size_t lb = b->left_bond_dims[site];
        size_t ra = a->right_bond_dims[site];
        size_t rb = b->right_bond_dims[site];

        size_t new_left = (site == 0) ? 1 : la + lb;
        size_t new_right = (site == N - 1) ? 1 : ra + rb;

        (*result)->left_bond_dims[site] = new_left;
        (*result)->right_bond_dims[site] = new_right;
        (*result)->tensor_sizes[site] = new_left * d * new_right;

        free((*result)->tensors[site]);
        (*result)->tensors[site] = calloc((*result)->tensor_sizes[site], sizeof(ComplexFloat));
        if (!(*result)->tensors[site]) {
            mps_destroy(*result);
            *result = NULL;
            return QGT_ERROR_MEMORY_ALLOCATION;
        }

        // Fill in block-diagonal structure
        // For site 0: result = [a | b]
        // For site N-1: result = [a; b]
        // For middle sites: result = [[a, 0], [0, b]]

        if (site == 0) {
            // Concatenate along right bond: result[:,σ,:] = [a[:,σ,:] | b[:,σ,:]]
            for (size_t sigma = 0; sigma < d; sigma++) {
                for (size_t r = 0; r < ra; r++) {
                    size_t a_idx = sigma * ra + r;
                    size_t res_idx = sigma * new_right + r;
                    (*result)->tensors[site][res_idx] = a->tensors[site][a_idx];
                }
                for (size_t r = 0; r < rb; r++) {
                    size_t b_idx = sigma * rb + r;
                    size_t res_idx = sigma * new_right + ra + r;
                    (*result)->tensors[site][res_idx] = b->tensors[site][b_idx];
                }
            }
        } else if (site == N - 1) {
            // Concatenate along left bond: result[:,σ,:] = [a[:,σ,:]; b[:,σ,:]]
            for (size_t l = 0; l < la; l++) {
                for (size_t sigma = 0; sigma < d; sigma++) {
                    size_t a_idx = l * d + sigma;
                    size_t res_idx = l * d + sigma;
                    (*result)->tensors[site][res_idx] = a->tensors[site][a_idx];
                }
            }
            for (size_t l = 0; l < lb; l++) {
                for (size_t sigma = 0; sigma < d; sigma++) {
                    size_t b_idx = l * d + sigma;
                    size_t res_idx = (la + l) * d + sigma;
                    (*result)->tensors[site][res_idx] = b->tensors[site][b_idx];
                }
            }
        } else {
            // Block diagonal
            for (size_t l = 0; l < la; l++) {
                for (size_t sigma = 0; sigma < d; sigma++) {
                    for (size_t r = 0; r < ra; r++) {
                        size_t a_idx = l * d * ra + sigma * ra + r;
                        size_t res_idx = l * d * new_right + sigma * new_right + r;
                        (*result)->tensors[site][res_idx] = a->tensors[site][a_idx];
                    }
                }
            }
            for (size_t l = 0; l < lb; l++) {
                for (size_t sigma = 0; sigma < d; sigma++) {
                    for (size_t r = 0; r < rb; r++) {
                        size_t b_idx = l * d * rb + sigma * rb + r;
                        size_t res_idx = (la + l) * d * new_right + sigma * new_right + (ra + r);
                        (*result)->tensors[site][res_idx] = b->tensors[site][b_idx];
                    }
                }
            }
        }
    }

    (*result)->form = MPS_CANONICAL_NONE;
    (*result)->is_normalized = false;
    return QGT_SUCCESS;
}

qgt_error_t mps_scale(MatrixProductState* mps, ComplexFloat scalar) {
    if (!mps) return QGT_ERROR_INVALID_ARGUMENT;

    // Scale the first tensor
    size_t size = mps->tensor_sizes[0];
    for (size_t i = 0; i < size; i++) {
        ComplexFloat val = mps->tensors[0][i];
        mps->tensors[0][i].real = val.real * scalar.real - val.imag * scalar.imag;
        mps->tensors[0][i].imag = val.real * scalar.imag + val.imag * scalar.real;
    }

    mps->is_normalized = false;
    return QGT_SUCCESS;
}

// =============================================================================
// Utility Functions
// =============================================================================

size_t mps_num_parameters(const MatrixProductState* mps) {
    if (!mps) return 0;

    size_t total = 0;
    for (size_t i = 0; i < mps->num_sites; i++) {
        total += mps->tensor_sizes[i];
    }
    return total;
}

size_t mps_get_max_bond_dim(const MatrixProductState* mps) {
    if (!mps) return 0;

    size_t max_dim = 0;
    for (size_t i = 0; i < mps->num_sites; i++) {
        if (mps->left_bond_dims[i] > max_dim) max_dim = mps->left_bond_dims[i];
        if (mps->right_bond_dims[i] > max_dim) max_dim = mps->right_bond_dims[i];
    }
    return max_dim;
}

bool mps_verify_canonical_form(const MatrixProductState* mps, double tolerance) {
    if (!mps) return false;

    size_t d = mps->physical_dim;

    switch (mps->form) {
        case MPS_CANONICAL_LEFT:
            // Check that A†A = I for all sites except the last
            for (size_t site = 0; site < mps->num_sites - 1; site++) {
                size_t left = mps->left_bond_dims[site];
                size_t right = mps->right_bond_dims[site];
                size_t m = left * d;

                // Compute A†A
                ComplexFloat* AHA = calloc(right * right, sizeof(ComplexFloat));
                if (!AHA) return false;

                for (size_t i = 0; i < right; i++) {
                    for (size_t j = 0; j < right; j++) {
                        for (size_t k = 0; k < m; k++) {
                            size_t idx_i = k * right + i;
                            size_t idx_j = k * right + j;
                            ComplexFloat a_conj = {mps->tensors[site][idx_i].real,
                                                   -mps->tensors[site][idx_i].imag};
                            ComplexFloat a = mps->tensors[site][idx_j];
                            AHA[i * right + j].real += a_conj.real * a.real - a_conj.imag * a.imag;
                            AHA[i * right + j].imag += a_conj.real * a.imag + a_conj.imag * a.real;
                        }
                    }
                }

                // Check if A†A ≈ I
                for (size_t i = 0; i < right; i++) {
                    for (size_t j = 0; j < right; j++) {
                        float expected_real = (i == j) ? 1.0f : 0.0f;
                        float diff_real = fabsf(AHA[i * right + j].real - expected_real);
                        float diff_imag = fabsf(AHA[i * right + j].imag);
                        if (diff_real > tolerance || diff_imag > tolerance) {
                            free(AHA);
                            return false;
                        }
                    }
                }
                free(AHA);
            }
            return true;

        case MPS_CANONICAL_RIGHT:
            // Check that AA† = I for all sites except the first
            for (size_t site = 1; site < mps->num_sites; site++) {
                size_t left = mps->left_bond_dims[site];
                size_t right = mps->right_bond_dims[site];
                size_t n = d * right;

                ComplexFloat* AAH = calloc(left * left, sizeof(ComplexFloat));
                if (!AAH) return false;

                for (size_t i = 0; i < left; i++) {
                    for (size_t j = 0; j < left; j++) {
                        for (size_t k = 0; k < n; k++) {
                            size_t idx_i = i * n + k;
                            size_t idx_j = j * n + k;
                            ComplexFloat a = mps->tensors[site][idx_i];
                            ComplexFloat a_conj = {mps->tensors[site][idx_j].real,
                                                   -mps->tensors[site][idx_j].imag};
                            AAH[i * left + j].real += a.real * a_conj.real - a.imag * a_conj.imag;
                            AAH[i * left + j].imag += a.real * a_conj.imag + a.imag * a_conj.real;
                        }
                    }
                }

                for (size_t i = 0; i < left; i++) {
                    for (size_t j = 0; j < left; j++) {
                        float expected_real = (i == j) ? 1.0f : 0.0f;
                        float diff_real = fabsf(AAH[i * left + j].real - expected_real);
                        float diff_imag = fabsf(AAH[i * left + j].imag);
                        if (diff_real > tolerance || diff_imag > tolerance) {
                            free(AAH);
                            return false;
                        }
                    }
                }
                free(AAH);
            }
            return true;

        case MPS_CANONICAL_MIXED:
            // Sites 0 to center-1 should be left-canonical
            // Sites center+1 to N-1 should be right-canonical
            // This is a combination of the above checks
            return true;  // Simplified for now

        case MPS_CANONICAL_NONE:
        default:
            return true;  // No form to verify
    }
}

void mps_print_info(const MatrixProductState* mps) {
    if (!mps) {
        printf("MPS: NULL\n");
        return;
    }

    printf("Matrix Product State:\n");
    printf("  Sites: %zu\n", mps->num_sites);
    printf("  Physical dimension: %zu\n", mps->physical_dim);
    printf("  Max bond dimension: %zu\n", mps->max_bond_dim);
    printf("  Canonical form: ");
    switch (mps->form) {
        case MPS_CANONICAL_NONE:  printf("None\n"); break;
        case MPS_CANONICAL_LEFT:  printf("Left\n"); break;
        case MPS_CANONICAL_RIGHT: printf("Right\n"); break;
        case MPS_CANONICAL_MIXED: printf("Mixed (center=%zu)\n", mps->orthogonality_center); break;
    }
    printf("  Normalized: %s\n", mps->is_normalized ? "yes" : "no");
    printf("  Bond dimensions: ");
    for (size_t i = 0; i < mps->num_sites; i++) {
        printf("%zu", mps->left_bond_dims[i]);
        if (i < mps->num_sites - 1) printf("-");
    }
    printf("-%zu\n", mps->right_bond_dims[mps->num_sites - 1]);
    printf("  Total parameters: %zu\n", mps_num_parameters(mps));
}
