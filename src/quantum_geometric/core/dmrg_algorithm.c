/**
 * @file dmrg_algorithm.c
 * @brief Density Matrix Renormalization Group (DMRG) algorithm implementation
 *
 * Implements the two-site DMRG algorithm for finding ground states of
 * 1D quantum many-body systems represented as Matrix Product States (MPS).
 */

#include "quantum_geometric/core/dmrg_algorithm.h"
#include "quantum_geometric/core/lapack_wrapper.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

// ============================================================================
// Internal Helper Functions
// ============================================================================

/**
 * @brief Complex number operations
 */
static inline ComplexFloat cf_add(ComplexFloat a, ComplexFloat b) {
    return (ComplexFloat){a.real + b.real, a.imag + b.imag};
}

static inline ComplexFloat cf_sub(ComplexFloat a, ComplexFloat b) {
    return (ComplexFloat){a.real - b.real, a.imag - b.imag};
}

static inline ComplexFloat cf_mul(ComplexFloat a, ComplexFloat b) {
    return (ComplexFloat){
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real
    };
}

static inline ComplexFloat cf_conj(ComplexFloat a) {
    return (ComplexFloat){a.real, -a.imag};
}

static inline ComplexFloat cf_scale(ComplexFloat a, double s) {
    return (ComplexFloat){a.real * s, a.imag * s};
}

static inline double cf_norm_sq(ComplexFloat a) {
    return a.real * a.real + a.imag * a.imag;
}

static inline double cf_norm(ComplexFloat a) {
    return sqrt(cf_norm_sq(a));
}

/**
 * @brief Compute inner product of two complex vectors
 */
static ComplexFloat vector_inner_product(
    const ComplexFloat* a,
    const ComplexFloat* b,
    size_t n)
{
    ComplexFloat result = {0.0f, 0.0f};
    for (size_t i = 0; i < n; i++) {
        result = cf_add(result, cf_mul(cf_conj(a[i]), b[i]));
    }
    return result;
}

/**
 * @brief Compute norm of a complex vector
 */
static double vector_norm(const ComplexFloat* v, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += cf_norm_sq(v[i]);
    }
    return sqrt(sum);
}

/**
 * @brief Normalize a complex vector in-place
 */
static void vector_normalize(ComplexFloat* v, size_t n) {
    double norm = vector_norm(v, n);
    if (norm > 1e-15) {
        double inv_norm = 1.0 / norm;
        for (size_t i = 0; i < n; i++) {
            v[i] = cf_scale(v[i], inv_norm);
        }
    }
}

/**
 * @brief Matrix-vector product: y = A * x
 */
static void matrix_vector_multiply(
    const ComplexFloat* A,
    const ComplexFloat* x,
    ComplexFloat* y,
    size_t m,
    size_t n)
{
    memset(y, 0, m * sizeof(ComplexFloat));
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            y[i] = cf_add(y[i], cf_mul(A[i * n + j], x[j]));
        }
    }
}

/**
 * @brief Compute eigenvalues of a real symmetric tridiagonal matrix
 *
 * Uses the QL algorithm with implicit shifts
 */
static bool tridiagonal_eigenvalues(
    double* diag,      // Main diagonal (modified in place)
    double* offdiag,   // Off-diagonal (modified in place)
    size_t n,
    double* eigenvecs) // n x n matrix of eigenvectors (column-major)
{
    if (n == 0) return false;
    if (n == 1) {
        if (eigenvecs) eigenvecs[0] = 1.0;
        return true;
    }

    // Initialize eigenvectors to identity
    if (eigenvecs) {
        memset(eigenvecs, 0, n * n * sizeof(double));
        for (size_t i = 0; i < n; i++) {
            eigenvecs[i * n + i] = 1.0;
        }
    }

    const int max_iterations = 30;
    const double eps = 1e-12;

    for (size_t l = 0; l < n; l++) {
        int iter = 0;
        size_t m;

        do {
            // Find small off-diagonal element
            for (m = l; m < n - 1; m++) {
                double dd = fabs(diag[m]) + fabs(diag[m + 1]);
                if (fabs(offdiag[m]) <= eps * dd) break;
            }

            if (m != l) {
                if (iter++ >= max_iterations) {
                    return false;  // Failed to converge
                }

                // Form shift
                double g = (diag[l + 1] - diag[l]) / (2.0 * offdiag[l]);
                double r = sqrt(g * g + 1.0);
                g = diag[m] - diag[l] + offdiag[l] / (g + (g >= 0 ? r : -r));

                double s = 1.0, c = 1.0, p = 0.0;

                for (size_t i = m; i > l; i--) {
                    double f = s * offdiag[i - 1];
                    double b = c * offdiag[i - 1];

                    if (fabs(f) >= fabs(g)) {
                        c = g / f;
                        r = sqrt(c * c + 1.0);
                        offdiag[i] = f * r;
                        s = 1.0 / r;
                        c *= s;
                    } else {
                        s = f / g;
                        r = sqrt(s * s + 1.0);
                        offdiag[i] = g * r;
                        c = 1.0 / r;
                        s *= c;
                    }

                    g = diag[i] - p;
                    r = (diag[i - 1] - g) * s + 2.0 * c * b;
                    p = s * r;
                    diag[i] = g + p;
                    g = c * r - b;

                    // Accumulate eigenvectors
                    if (eigenvecs) {
                        for (size_t k = 0; k < n; k++) {
                            f = eigenvecs[i * n + k];
                            eigenvecs[i * n + k] = s * eigenvecs[(i - 1) * n + k] + c * f;
                            eigenvecs[(i - 1) * n + k] = c * eigenvecs[(i - 1) * n + k] - s * f;
                        }
                    }
                }

                diag[l] -= p;
                offdiag[l] = g;
                offdiag[m] = 0.0;
            }
        } while (m != l);
    }

    return true;
}

// ============================================================================
// Hamiltonian Functions
// ============================================================================

DMRGHamiltonian* dmrg_create_hamiltonian(size_t num_sites, size_t local_dim) {
    if (num_sites == 0 || local_dim == 0) return NULL;

    DMRGHamiltonian* h = calloc(1, sizeof(DMRGHamiltonian));
    if (!h) return NULL;

    h->num_sites = num_sites;
    h->local_dim = local_dim;
    h->has_long_range = false;
    h->mpo_bond_dim = 3;  // Default for nearest-neighbor

    // Allocate local terms
    h->local_terms = calloc(num_sites, sizeof(ComplexFloat*));
    if (!h->local_terms) {
        free(h);
        return NULL;
    }

    // Allocate nearest-neighbor terms (num_sites - 1 bonds)
    h->nn_terms = calloc(num_sites - 1, sizeof(ComplexFloat*));
    if (!h->nn_terms) {
        free(h->local_terms);
        free(h);
        return NULL;
    }

    // Allocate individual term matrices
    size_t d2 = local_dim * local_dim;
    size_t d4 = d2 * d2;

    for (size_t i = 0; i < num_sites; i++) {
        h->local_terms[i] = calloc(d2, sizeof(ComplexFloat));
        if (!h->local_terms[i]) {
            dmrg_destroy_hamiltonian(h);
            return NULL;
        }
    }

    for (size_t i = 0; i < num_sites - 1; i++) {
        h->nn_terms[i] = calloc(d4, sizeof(ComplexFloat));
        if (!h->nn_terms[i]) {
            dmrg_destroy_hamiltonian(h);
            return NULL;
        }
    }

    return h;
}

void dmrg_destroy_hamiltonian(DMRGHamiltonian* hamiltonian) {
    if (!hamiltonian) return;

    if (hamiltonian->local_terms) {
        for (size_t i = 0; i < hamiltonian->num_sites; i++) {
            free(hamiltonian->local_terms[i]);
        }
        free(hamiltonian->local_terms);
    }

    if (hamiltonian->nn_terms) {
        for (size_t i = 0; i < hamiltonian->num_sites - 1; i++) {
            free(hamiltonian->nn_terms[i]);
        }
        free(hamiltonian->nn_terms);
    }

    if (hamiltonian->long_range_left) {
        for (size_t i = 0; i < hamiltonian->num_sites; i++) {
            free(hamiltonian->long_range_left[i]);
        }
        free(hamiltonian->long_range_left);
    }

    if (hamiltonian->long_range_right) {
        for (size_t i = 0; i < hamiltonian->num_sites; i++) {
            free(hamiltonian->long_range_right[i]);
        }
        free(hamiltonian->long_range_right);
    }

    free(hamiltonian);
}

qgt_error_t dmrg_set_local_term(
    DMRGHamiltonian* hamiltonian,
    size_t site,
    const ComplexFloat* term)
{
    if (!hamiltonian || !term) return QGT_ERROR_INVALID_ARGUMENT;
    if (site >= hamiltonian->num_sites) return QGT_ERROR_INVALID_ARGUMENT;

    size_t d2 = hamiltonian->local_dim * hamiltonian->local_dim;
    memcpy(hamiltonian->local_terms[site], term, d2 * sizeof(ComplexFloat));

    return QGT_SUCCESS;
}

qgt_error_t dmrg_set_nn_term(
    DMRGHamiltonian* hamiltonian,
    size_t site,
    const ComplexFloat* term)
{
    if (!hamiltonian || !term) return QGT_ERROR_INVALID_ARGUMENT;
    if (site >= hamiltonian->num_sites - 1) return QGT_ERROR_INVALID_ARGUMENT;

    size_t d2 = hamiltonian->local_dim * hamiltonian->local_dim;
    size_t d4 = d2 * d2;
    memcpy(hamiltonian->nn_terms[site], term, d4 * sizeof(ComplexFloat));

    return QGT_SUCCESS;
}

DMRGHamiltonian* dmrg_create_heisenberg_xxz(
    size_t num_sites,
    double J,
    double delta,
    double h)
{
    DMRGHamiltonian* ham = dmrg_create_hamiltonian(num_sites, 2);
    if (!ham) return NULL;

    // Pauli matrices for spin-1/2
    // Sx = (1/2) * [[0, 1], [1, 0]]
    // Sy = (1/2) * [[0, -i], [i, 0]]
    // Sz = (1/2) * [[1, 0], [0, -1]]

    // Local terms: h * Sz
    for (size_t i = 0; i < num_sites; i++) {
        ham->local_terms[i][0] = (ComplexFloat){0.5f * (float)h, 0.0f};   // |0><0|
        ham->local_terms[i][3] = (ComplexFloat){-0.5f * (float)h, 0.0f};  // |1><1|
    }

    // Nearest-neighbor terms: J * (Sx Sx + Sy Sy + delta * Sz Sz)
    // = J/4 * (sigma_x sigma_x + sigma_y sigma_y + delta * sigma_z sigma_z)
    // = J/4 * (2 * (|01><10| + |10><01|) + delta * (|00><00| + |11><11| - |01><01| - |10><10|))
    // Rewritten as J/2 * (S+ S- + S- S+) + J*delta * Sz Sz
    // = J/2 * (|01><10| + |10><01|) + J*delta/4 * (|00><00| - |01><01| - |10><10| + |11><11|)

    for (size_t i = 0; i < num_sites - 1; i++) {
        // Two-site basis: |00>, |01>, |10>, |11>
        // Index: i1 * 2 + i2 for first site, j1 * 2 + j2 for second site
        // Matrix element [i1*2+i2, j1*2+j2] = row, col

        // Sz Sz term: delta * J/4
        float jd4 = (float)(J * delta / 4.0);
        ham->nn_terms[i][0 * 4 + 0] = (ComplexFloat){jd4, 0.0f};   // |00><00|
        ham->nn_terms[i][1 * 4 + 1] = (ComplexFloat){-jd4, 0.0f};  // |01><01|
        ham->nn_terms[i][2 * 4 + 2] = (ComplexFloat){-jd4, 0.0f};  // |10><10|
        ham->nn_terms[i][3 * 4 + 3] = (ComplexFloat){jd4, 0.0f};   // |11><11|

        // S+ S- + S- S+ = 2*(Sx Sx + Sy Sy) term: J/2
        float j2 = (float)(J / 2.0);
        ham->nn_terms[i][1 * 4 + 2] = (ComplexFloat){j2, 0.0f};  // |01><10|
        ham->nn_terms[i][2 * 4 + 1] = (ComplexFloat){j2, 0.0f};  // |10><01|
    }

    return ham;
}

DMRGHamiltonian* dmrg_create_tfim(
    size_t num_sites,
    double J,
    double h)
{
    DMRGHamiltonian* ham = dmrg_create_hamiltonian(num_sites, 2);
    if (!ham) return NULL;

    // Local terms: -h * Sx = -h/2 * sigma_x
    for (size_t i = 0; i < num_sites; i++) {
        float hx = (float)(-h / 2.0);
        ham->local_terms[i][0 * 2 + 1] = (ComplexFloat){hx, 0.0f};  // |0><1|
        ham->local_terms[i][1 * 2 + 0] = (ComplexFloat){hx, 0.0f};  // |1><0|
    }

    // Nearest-neighbor terms: -J * Sz Sz = -J/4 * sigma_z sigma_z
    for (size_t i = 0; i < num_sites - 1; i++) {
        float jz = (float)(-J / 4.0);
        ham->nn_terms[i][0 * 4 + 0] = (ComplexFloat){jz, 0.0f};   // |00><00|
        ham->nn_terms[i][1 * 4 + 1] = (ComplexFloat){-jz, 0.0f};  // |01><01|
        ham->nn_terms[i][2 * 4 + 2] = (ComplexFloat){-jz, 0.0f};  // |10><10|
        ham->nn_terms[i][3 * 4 + 3] = (ComplexFloat){jz, 0.0f};   // |11><11|
    }

    return ham;
}

// ============================================================================
// Configuration Functions
// ============================================================================

DMRGConfig dmrg_get_default_config(void) {
    DMRGConfig config = {
        .max_sweeps = 20,
        .max_bond_dim = 100,
        .energy_tolerance = 1e-8,
        .truncation_cutoff = 1e-10,
        .verbose = false,
        .lanczos_iterations = 100,
        .lanczos_tolerance = 1e-12,
        .use_two_site = true,
        .warmup_sweeps = 4,
        .warmup_bond_dim = 20,
        .compute_variance = false
    };
    return config;
}

// ============================================================================
// Result Functions
// ============================================================================

DMRGResult* dmrg_create_result(size_t max_sweeps, size_t num_sites) {
    DMRGResult* result = calloc(1, sizeof(DMRGResult));
    if (!result) return NULL;

    result->energies = calloc(max_sweeps, sizeof(double));
    result->truncation_errors = calloc(max_sweeps, sizeof(double));
    result->variances = calloc(max_sweeps, sizeof(double));
    result->bond_dimensions = calloc(num_sites, sizeof(size_t));

    if (!result->energies || !result->truncation_errors ||
        !result->variances || !result->bond_dimensions) {
        dmrg_destroy_result(result);
        return NULL;
    }

    return result;
}

void dmrg_destroy_result(DMRGResult* result) {
    if (!result) return;
    free(result->energies);
    free(result->truncation_errors);
    free(result->variances);
    free(result->bond_dimensions);
    free(result);
}

// ============================================================================
// Environment Functions
// ============================================================================

DMRGEnvironment* dmrg_create_environment(size_t num_sites, size_t mpo_bond_dim) {
    DMRGEnvironment* env = calloc(1, sizeof(DMRGEnvironment));
    if (!env) return NULL;

    env->num_sites = num_sites;
    env->mpo_bond_dim = mpo_bond_dim;

    env->left_blocks = calloc(num_sites, sizeof(ComplexFloat*));
    env->right_blocks = calloc(num_sites, sizeof(ComplexFloat*));
    env->left_dims = calloc(num_sites, sizeof(size_t));
    env->right_dims = calloc(num_sites, sizeof(size_t));

    if (!env->left_blocks || !env->right_blocks ||
        !env->left_dims || !env->right_dims) {
        dmrg_destroy_environment(env);
        return NULL;
    }

    return env;
}

void dmrg_destroy_environment(DMRGEnvironment* env) {
    if (!env) return;

    if (env->left_blocks) {
        for (size_t i = 0; i < env->num_sites; i++) {
            free(env->left_blocks[i]);
        }
        free(env->left_blocks);
    }

    if (env->right_blocks) {
        for (size_t i = 0; i < env->num_sites; i++) {
            free(env->right_blocks[i]);
        }
        free(env->right_blocks);
    }

    free(env->left_dims);
    free(env->right_dims);
    free(env);
}

qgt_error_t dmrg_init_environment(
    DMRGEnvironment* env,
    const MatrixProductState* mps,
    const DMRGHamiltonian* hamiltonian)
{
    if (!env || !mps || !hamiltonian) return QGT_ERROR_INVALID_ARGUMENT;

    size_t n = mps->num_sites;
    size_t w = env->mpo_bond_dim;  // MPO bond dimension

    // Initialize left boundary (site 0)
    // For nearest-neighbor Hamiltonian, MPO bond dimension is 3:
    // W[0] = [I, S_L, h_L] (left boundary)
    // W[i] = [[I, 0, 0], [S_R, 0, 0], [h_i, S_L, I]] (bulk)
    // W[n-1] = [h_R, S_R, I]^T (right boundary)

    // Left block at site -1 (before chain): just identity on MPO index
    env->left_dims[0] = mps->right_bond_dims[0] > 0 ? mps->right_bond_dims[0] : 1;
    size_t left_mps_dim = env->left_dims[0];

    // Allocate left boundary block
    // Dimension: (left_mps_dim * w) x (left_mps_dim)
    // But we store it as flat: mps_dim^2 * w
    size_t block_size = left_mps_dim * left_mps_dim * w;
    env->left_blocks[0] = calloc(block_size, sizeof(ComplexFloat));
    if (!env->left_blocks[0]) return QGT_ERROR_NO_MEMORY;

    // Initialize: only the w=0 component has identity
    for (size_t i = 0; i < left_mps_dim; i++) {
        // Index: w * mps_dim^2 + i * mps_dim + i
        env->left_blocks[0][0 * left_mps_dim * left_mps_dim + i * left_mps_dim + i] =
            (ComplexFloat){1.0f, 0.0f};
    }

    // Initialize right boundary (site n)
    // Right block at site n: identity on last MPO index
    size_t right_mps_dim = mps->right_bond_dims[n - 1] > 0 ? mps->right_bond_dims[n - 1] : 1;
    env->right_dims[n - 1] = right_mps_dim;

    block_size = right_mps_dim * right_mps_dim * w;
    env->right_blocks[n - 1] = calloc(block_size, sizeof(ComplexFloat));
    if (!env->right_blocks[n - 1]) return QGT_ERROR_NO_MEMORY;

    // Initialize: only the w=w-1 component has identity
    for (size_t i = 0; i < right_mps_dim; i++) {
        env->right_blocks[n - 1][(w - 1) * right_mps_dim * right_mps_dim + i * right_mps_dim + i] =
            (ComplexFloat){1.0f, 0.0f};
    }

    // Build right blocks from right to left
    for (size_t site = n - 1; site > 0; site--) {
        qgt_error_t err = dmrg_update_right_block(env, mps, hamiltonian, site);
        if (err != QGT_SUCCESS) return err;
    }

    return QGT_SUCCESS;
}

qgt_error_t dmrg_update_left_block(
    DMRGEnvironment* env,
    const MatrixProductState* mps,
    const DMRGHamiltonian* hamiltonian,
    size_t site)
{
    if (!env || !mps || !hamiltonian) return QGT_ERROR_INVALID_ARGUMENT;
    if (site >= mps->num_sites) return QGT_ERROR_INVALID_ARGUMENT;

    size_t d = mps->physical_dim;
    size_t w = env->mpo_bond_dim;

    // Get dimensions
    size_t left_dim = (site == 0) ? 1 : mps->right_bond_dims[site - 1];
    size_t right_dim = mps->right_bond_dims[site];

    // Current MPS tensor: A[left_dim, d, right_dim]
    const ComplexFloat* A = mps->tensors[site];

    // Previous left block: L[w, left_dim, left_dim]
    const ComplexFloat* L_prev = env->left_blocks[site];

    // New left block: L_new[w, right_dim, right_dim]
    size_t new_block_size = w * right_dim * right_dim;
    ComplexFloat* L_new = calloc(new_block_size, sizeof(ComplexFloat));
    if (!L_new) return QGT_ERROR_NO_MEMORY;

    // Contract: L_new[w', a', a] = Σ_{b,b',σ} L_prev[w, b', b] * A*[b', σ, a'] * MPO[w, w', σ', σ] * A[b, σ', a]

    // For nearest-neighbor Hamiltonian, MPO structure:
    // w=0: Identity (copy from prev)
    // w=1: Apply S_L operator
    // w=2: Apply H_local and finish S_L S_R

    const ComplexFloat* H_local = hamiltonian->local_terms[site];

    // w'=0 (Identity on output): L_new[0] = A^H @ L_prev[0] @ A
    for (size_t ap = 0; ap < right_dim; ap++) {
        for (size_t a = 0; a < right_dim; a++) {
            ComplexFloat sum = {0.0f, 0.0f};
            for (size_t sigma = 0; sigma < d; sigma++) {
                for (size_t bp = 0; bp < left_dim; bp++) {
                    for (size_t b = 0; b < left_dim; b++) {
                        // A*[bp, sigma, ap]
                        ComplexFloat A_conj = cf_conj(A[bp * d * right_dim + sigma * right_dim + ap]);
                        // L_prev[0, bp, b]
                        ComplexFloat L_val = L_prev[0 * left_dim * left_dim + bp * left_dim + b];
                        // A[b, sigma, a]
                        ComplexFloat A_val = A[b * d * right_dim + sigma * right_dim + a];

                        sum = cf_add(sum, cf_mul(cf_mul(A_conj, L_val), A_val));
                    }
                }
            }
            L_new[0 * right_dim * right_dim + ap * right_dim + a] = sum;
        }
    }

    // w'=2 (Hamiltonian output): L_new[2] = A^H @ L_prev[2] @ A + A^H @ H_local @ A (from L_prev[0])
    // Plus contribution from nearest-neighbor term via L_prev[1]
    for (size_t ap = 0; ap < right_dim; ap++) {
        for (size_t a = 0; a < right_dim; a++) {
            ComplexFloat sum = {0.0f, 0.0f};

            // Identity continuation from L_prev[2]
            for (size_t sigma = 0; sigma < d; sigma++) {
                for (size_t bp = 0; bp < left_dim; bp++) {
                    for (size_t b = 0; b < left_dim; b++) {
                        ComplexFloat A_conj = cf_conj(A[bp * d * right_dim + sigma * right_dim + ap]);
                        ComplexFloat L_val = L_prev[2 * left_dim * left_dim + bp * left_dim + b];
                        ComplexFloat A_val = A[b * d * right_dim + sigma * right_dim + a];

                        sum = cf_add(sum, cf_mul(cf_mul(A_conj, L_val), A_val));
                    }
                }
            }

            // Local term from L_prev[0]
            for (size_t sigma = 0; sigma < d; sigma++) {
                for (size_t sigmap = 0; sigmap < d; sigmap++) {
                    ComplexFloat h_val = H_local[sigmap * d + sigma];
                    if (cf_norm_sq(h_val) < 1e-20f) continue;

                    for (size_t bp = 0; bp < left_dim; bp++) {
                        for (size_t b = 0; b < left_dim; b++) {
                            ComplexFloat A_conj = cf_conj(A[bp * d * right_dim + sigmap * right_dim + ap]);
                            ComplexFloat L_val = L_prev[0 * left_dim * left_dim + bp * left_dim + b];
                            ComplexFloat A_val = A[b * d * right_dim + sigma * right_dim + a];

                            sum = cf_add(sum, cf_mul(cf_mul(cf_mul(A_conj, h_val), L_val), A_val));
                        }
                    }
                }
            }

            L_new[2 * right_dim * right_dim + ap * right_dim + a] = sum;
        }
    }

    // Store new left block
    free(env->left_blocks[site + 1]);
    env->left_blocks[site + 1] = L_new;
    env->left_dims[site + 1] = right_dim;

    return QGT_SUCCESS;
}

qgt_error_t dmrg_update_right_block(
    DMRGEnvironment* env,
    const MatrixProductState* mps,
    const DMRGHamiltonian* hamiltonian,
    size_t site)
{
    if (!env || !mps || !hamiltonian) return QGT_ERROR_INVALID_ARGUMENT;
    if (site == 0 || site >= mps->num_sites) return QGT_ERROR_INVALID_ARGUMENT;

    size_t d = mps->physical_dim;
    size_t w = env->mpo_bond_dim;

    // Get dimensions
    size_t left_dim = mps->right_bond_dims[site - 1];
    size_t right_dim = (site == mps->num_sites - 1) ? 1 : mps->right_bond_dims[site];

    // Current MPS tensor: A[left_dim, d, right_dim]
    const ComplexFloat* A = mps->tensors[site];

    // Previous right block: R[w, right_dim, right_dim]
    const ComplexFloat* R_prev = env->right_blocks[site];

    // New right block: R_new[w, left_dim, left_dim]
    size_t new_block_size = w * left_dim * left_dim;
    ComplexFloat* R_new = calloc(new_block_size, sizeof(ComplexFloat));
    if (!R_new) return QGT_ERROR_NO_MEMORY;

    const ComplexFloat* H_local = hamiltonian->local_terms[site];

    // w=2 (Identity on input): R_new[2] = A @ R_prev[2] @ A^H
    for (size_t bp = 0; bp < left_dim; bp++) {
        for (size_t b = 0; b < left_dim; b++) {
            ComplexFloat sum = {0.0f, 0.0f};
            for (size_t sigma = 0; sigma < d; sigma++) {
                for (size_t ap = 0; ap < right_dim; ap++) {
                    for (size_t a = 0; a < right_dim; a++) {
                        // A[bp, sigma, ap]
                        ComplexFloat A_bra = A[bp * d * right_dim + sigma * right_dim + ap];
                        // R_prev[2, ap, a]
                        ComplexFloat R_val = R_prev[2 * right_dim * right_dim + ap * right_dim + a];
                        // A*[b, sigma, a]
                        ComplexFloat A_ket = cf_conj(A[b * d * right_dim + sigma * right_dim + a]);

                        sum = cf_add(sum, cf_mul(cf_mul(A_bra, R_val), A_ket));
                    }
                }
            }
            R_new[2 * left_dim * left_dim + bp * left_dim + b] = sum;
        }
    }

    // w=0 (Hamiltonian on input): includes local term and continuation from R_prev[0]
    for (size_t bp = 0; bp < left_dim; bp++) {
        for (size_t b = 0; b < left_dim; b++) {
            ComplexFloat sum = {0.0f, 0.0f};

            // Identity continuation from R_prev[0]
            for (size_t sigma = 0; sigma < d; sigma++) {
                for (size_t ap = 0; ap < right_dim; ap++) {
                    for (size_t a = 0; a < right_dim; a++) {
                        ComplexFloat A_bra = A[bp * d * right_dim + sigma * right_dim + ap];
                        ComplexFloat R_val = R_prev[0 * right_dim * right_dim + ap * right_dim + a];
                        ComplexFloat A_ket = cf_conj(A[b * d * right_dim + sigma * right_dim + a]);

                        sum = cf_add(sum, cf_mul(cf_mul(A_bra, R_val), A_ket));
                    }
                }
            }

            // Local term contribution from R_prev[2]
            for (size_t sigma = 0; sigma < d; sigma++) {
                for (size_t sigmap = 0; sigmap < d; sigmap++) {
                    ComplexFloat h_val = H_local[sigmap * d + sigma];
                    if (cf_norm_sq(h_val) < 1e-20f) continue;

                    for (size_t ap = 0; ap < right_dim; ap++) {
                        for (size_t a = 0; a < right_dim; a++) {
                            ComplexFloat A_bra = A[bp * d * right_dim + sigmap * right_dim + ap];
                            ComplexFloat R_val = R_prev[2 * right_dim * right_dim + ap * right_dim + a];
                            ComplexFloat A_ket = cf_conj(A[b * d * right_dim + sigma * right_dim + a]);

                            sum = cf_add(sum, cf_mul(cf_mul(cf_mul(A_bra, h_val), R_val), A_ket));
                        }
                    }
                }
            }

            R_new[0 * left_dim * left_dim + bp * left_dim + b] = sum;
        }
    }

    // Store new right block
    free(env->right_blocks[site - 1]);
    env->right_blocks[site - 1] = R_new;
    env->right_dims[site - 1] = left_dim;

    return QGT_SUCCESS;
}

// ============================================================================
// Lanczos Eigensolver
// ============================================================================

qgt_error_t lanczos_ground_state(
    const ComplexFloat* matrix,
    size_t dim,
    ComplexFloat* eigenvector,
    double* eigenvalue,
    size_t max_iterations,
    double tolerance,
    void (*matvec)(const ComplexFloat* v, ComplexFloat* Av, size_t dim, void* data),
    void* matvec_data)
{
    if (!eigenvector || !eigenvalue) return QGT_ERROR_INVALID_ARGUMENT;
    if (dim == 0) return QGT_ERROR_INVALID_ARGUMENT;

    // Limit Lanczos iterations to dimension
    if (max_iterations > dim) max_iterations = dim;
    if (max_iterations < 2) max_iterations = 2;

    // Allocate Lanczos vectors
    ComplexFloat** lanczos_vecs = malloc(max_iterations * sizeof(ComplexFloat*));
    if (!lanczos_vecs) return QGT_ERROR_NO_MEMORY;

    for (size_t i = 0; i < max_iterations; i++) {
        lanczos_vecs[i] = calloc(dim, sizeof(ComplexFloat));
        if (!lanczos_vecs[i]) {
            for (size_t j = 0; j < i; j++) free(lanczos_vecs[j]);
            free(lanczos_vecs);
            return QGT_ERROR_NO_MEMORY;
        }
    }

    // Allocate tridiagonal matrix elements
    double* alpha = calloc(max_iterations, sizeof(double));  // Diagonal
    double* beta = calloc(max_iterations, sizeof(double));   // Off-diagonal
    ComplexFloat* w = calloc(dim, sizeof(ComplexFloat));     // Work vector

    if (!alpha || !beta || !w) {
        free(alpha); free(beta); free(w);
        for (size_t i = 0; i < max_iterations; i++) free(lanczos_vecs[i]);
        free(lanczos_vecs);
        return QGT_ERROR_NO_MEMORY;
    }

    // Initialize with random vector
    for (size_t i = 0; i < dim; i++) {
        lanczos_vecs[0][i] = (ComplexFloat){
            (float)(rand() / (double)RAND_MAX - 0.5),
            (float)(rand() / (double)RAND_MAX - 0.5)
        };
    }
    vector_normalize(lanczos_vecs[0], dim);

    double prev_eigenvalue = 1e10;
    size_t num_iters = 0;

    for (size_t k = 0; k < max_iterations; k++) {
        num_iters = k + 1;

        // w = A * v_k
        if (matvec) {
            matvec(lanczos_vecs[k], w, dim, matvec_data);
        } else if (matrix) {
            matrix_vector_multiply(matrix, lanczos_vecs[k], w, dim, dim);
        } else {
            // Error: no way to compute matrix-vector product
            free(alpha); free(beta); free(w);
            for (size_t i = 0; i < max_iterations; i++) free(lanczos_vecs[i]);
            free(lanczos_vecs);
            return QGT_ERROR_INVALID_ARGUMENT;
        }

        // alpha_k = <v_k, w>
        ComplexFloat alpha_c = vector_inner_product(lanczos_vecs[k], w, dim);
        alpha[k] = alpha_c.real;  // Should be real for Hermitian matrix

        // w = w - alpha_k * v_k - beta_{k-1} * v_{k-1}
        for (size_t i = 0; i < dim; i++) {
            w[i] = cf_sub(w[i], cf_scale(lanczos_vecs[k][i], alpha[k]));
            if (k > 0) {
                w[i] = cf_sub(w[i], cf_scale(lanczos_vecs[k - 1][i], beta[k - 1]));
            }
        }

        // Reorthogonalization (full)
        for (size_t j = 0; j <= k; j++) {
            ComplexFloat overlap = vector_inner_product(lanczos_vecs[j], w, dim);
            for (size_t i = 0; i < dim; i++) {
                w[i] = cf_sub(w[i], cf_mul(overlap, lanczos_vecs[j][i]));
            }
        }

        // beta_k = ||w||
        beta[k] = vector_norm(w, dim);

        // Check for convergence
        if (k >= 2) {
            // Solve tridiagonal eigenvalue problem
            double* diag = malloc((k + 1) * sizeof(double));
            double* offdiag = malloc((k + 1) * sizeof(double));
            double* eigvecs = malloc((k + 1) * (k + 1) * sizeof(double));

            memcpy(diag, alpha, (k + 1) * sizeof(double));
            memcpy(offdiag, beta, k * sizeof(double));
            offdiag[k] = 0.0;

            if (tridiagonal_eigenvalues(diag, offdiag, k + 1, eigvecs)) {
                // Find minimum eigenvalue
                double min_eval = diag[0];
                size_t min_idx = 0;
                for (size_t i = 1; i <= k; i++) {
                    if (diag[i] < min_eval) {
                        min_eval = diag[i];
                        min_idx = i;
                    }
                }

                // Check convergence
                if (fabs(min_eval - prev_eigenvalue) < tolerance) {
                    *eigenvalue = min_eval;

                    // Construct eigenvector
                    memset(eigenvector, 0, dim * sizeof(ComplexFloat));
                    for (size_t j = 0; j <= k; j++) {
                        double coeff = eigvecs[min_idx * (k + 1) + j];
                        for (size_t i = 0; i < dim; i++) {
                            eigenvector[i] = cf_add(eigenvector[i],
                                cf_scale(lanczos_vecs[j][i], coeff));
                        }
                    }
                    vector_normalize(eigenvector, dim);

                    free(diag); free(offdiag); free(eigvecs);
                    goto cleanup_success;
                }

                prev_eigenvalue = min_eval;
            }

            free(diag); free(offdiag); free(eigvecs);
        }

        // Check for breakdown
        if (beta[k] < 1e-14) {
            // Krylov subspace exhausted
            break;
        }

        // Normalize to get next Lanczos vector
        if (k + 1 < max_iterations) {
            double inv_beta = 1.0 / beta[k];
            for (size_t i = 0; i < dim; i++) {
                lanczos_vecs[k + 1][i] = cf_scale(w[i], inv_beta);
            }
        }
    }

    // Final eigenvalue computation
    {
        double* diag = malloc(num_iters * sizeof(double));
        double* offdiag = malloc(num_iters * sizeof(double));
        double* eigvecs = malloc(num_iters * num_iters * sizeof(double));

        memcpy(diag, alpha, num_iters * sizeof(double));
        if (num_iters > 1) {
            memcpy(offdiag, beta, (num_iters - 1) * sizeof(double));
        }
        offdiag[num_iters - 1] = 0.0;

        if (tridiagonal_eigenvalues(diag, offdiag, num_iters, eigvecs)) {
            double min_eval = diag[0];
            size_t min_idx = 0;
            for (size_t i = 1; i < num_iters; i++) {
                if (diag[i] < min_eval) {
                    min_eval = diag[i];
                    min_idx = i;
                }
            }

            *eigenvalue = min_eval;

            memset(eigenvector, 0, dim * sizeof(ComplexFloat));
            for (size_t j = 0; j < num_iters; j++) {
                double coeff = eigvecs[min_idx * num_iters + j];
                for (size_t i = 0; i < dim; i++) {
                    eigenvector[i] = cf_add(eigenvector[i],
                        cf_scale(lanczos_vecs[j][i], coeff));
                }
            }
            vector_normalize(eigenvector, dim);
        }

        free(diag); free(offdiag); free(eigvecs);
    }

cleanup_success:
    free(alpha);
    free(beta);
    free(w);
    for (size_t i = 0; i < max_iterations; i++) free(lanczos_vecs[i]);
    free(lanczos_vecs);

    return QGT_SUCCESS;
}

// ============================================================================
// Two-Site Optimization
// ============================================================================

/**
 * @brief Data for effective Hamiltonian matrix-vector product
 */
typedef struct {
    const ComplexFloat* L;      // Left environment
    const ComplexFloat* R;      // Right environment
    const ComplexFloat* H_nn;   // Nearest-neighbor term
    const ComplexFloat* H_L;    // Local term left
    const ComplexFloat* H_R;    // Local term right
    size_t left_dim;
    size_t d;
    size_t right_dim;
    size_t w;
} EffectiveHData;

/**
 * @brief Apply effective Hamiltonian to two-site tensor
 */
static void effective_h_matvec(
    const ComplexFloat* psi,
    ComplexFloat* h_psi,
    size_t dim,
    void* data)
{
    EffectiveHData* hd = (EffectiveHData*)data;

    size_t left_dim = hd->left_dim;
    size_t d = hd->d;
    size_t right_dim = hd->right_dim;
    size_t d2 = d * d;

    // psi has shape: [left_dim, d, d, right_dim]
    // Flatten index: b * d * d * right_dim + s1 * d * right_dim + s2 * right_dim + a

    memset(h_psi, 0, dim * sizeof(ComplexFloat));

    // Apply nearest-neighbor Hamiltonian term
    if (hd->H_nn) {
        for (size_t b = 0; b < left_dim; b++) {
            for (size_t a = 0; a < right_dim; a++) {
                // H_nn acts on (s1, s2) indices
                for (size_t s1p = 0; s1p < d; s1p++) {
                    for (size_t s2p = 0; s2p < d; s2p++) {
                        ComplexFloat sum = {0.0f, 0.0f};
                        for (size_t s1 = 0; s1 < d; s1++) {
                            for (size_t s2 = 0; s2 < d; s2++) {
                                // H_nn[(s1p, s2p), (s1, s2)]
                                size_t bra = s1p * d + s2p;
                                size_t ket = s1 * d + s2;
                                ComplexFloat h_val = hd->H_nn[bra * d2 + ket];

                                size_t psi_idx = b * d2 * right_dim + s1 * d * right_dim + s2 * right_dim + a;
                                sum = cf_add(sum, cf_mul(h_val, psi[psi_idx]));
                            }
                        }
                        size_t out_idx = b * d2 * right_dim + s1p * d * right_dim + s2p * right_dim + a;
                        h_psi[out_idx] = cf_add(h_psi[out_idx], sum);
                    }
                }
            }
        }
    }

    // Apply local term on left site
    if (hd->H_L) {
        for (size_t b = 0; b < left_dim; b++) {
            for (size_t s2 = 0; s2 < d; s2++) {
                for (size_t a = 0; a < right_dim; a++) {
                    for (size_t s1p = 0; s1p < d; s1p++) {
                        ComplexFloat sum = {0.0f, 0.0f};
                        for (size_t s1 = 0; s1 < d; s1++) {
                            ComplexFloat h_val = hd->H_L[s1p * d + s1];
                            size_t psi_idx = b * d2 * right_dim + s1 * d * right_dim + s2 * right_dim + a;
                            sum = cf_add(sum, cf_mul(h_val, psi[psi_idx]));
                        }
                        size_t out_idx = b * d2 * right_dim + s1p * d * right_dim + s2 * right_dim + a;
                        h_psi[out_idx] = cf_add(h_psi[out_idx], sum);
                    }
                }
            }
        }
    }

    // Apply local term on right site
    if (hd->H_R) {
        for (size_t b = 0; b < left_dim; b++) {
            for (size_t s1 = 0; s1 < d; s1++) {
                for (size_t a = 0; a < right_dim; a++) {
                    for (size_t s2p = 0; s2p < d; s2p++) {
                        ComplexFloat sum = {0.0f, 0.0f};
                        for (size_t s2 = 0; s2 < d; s2++) {
                            ComplexFloat h_val = hd->H_R[s2p * d + s2];
                            size_t psi_idx = b * d2 * right_dim + s1 * d * right_dim + s2 * right_dim + a;
                            sum = cf_add(sum, cf_mul(h_val, psi[psi_idx]));
                        }
                        size_t out_idx = b * d2 * right_dim + s1 * d * right_dim + s2p * right_dim + a;
                        h_psi[out_idx] = cf_add(h_psi[out_idx], sum);
                    }
                }
            }
        }
    }

    // Apply left environment (on left MPS index)
    if (hd->L) {
        for (size_t w = 0; w < hd->w; w++) {
            for (size_t bp = 0; bp < left_dim; bp++) {
                for (size_t s1 = 0; s1 < d; s1++) {
                    for (size_t s2 = 0; s2 < d; s2++) {
                        for (size_t a = 0; a < right_dim; a++) {
                            ComplexFloat sum = {0.0f, 0.0f};
                            for (size_t b = 0; b < left_dim; b++) {
                                // L[w, bp, b]
                                ComplexFloat l_val = hd->L[w * left_dim * left_dim + bp * left_dim + b];
                                size_t psi_idx = b * d2 * right_dim + s1 * d * right_dim + s2 * right_dim + a;
                                sum = cf_add(sum, cf_mul(l_val, psi[psi_idx]));
                            }
                            size_t out_idx = bp * d2 * right_dim + s1 * d * right_dim + s2 * right_dim + a;
                            // Only add if w corresponds to Hamiltonian action
                            if (w == 2) {
                                h_psi[out_idx] = cf_add(h_psi[out_idx], sum);
                            }
                        }
                    }
                }
            }
        }
    }

    // Apply right environment (on right MPS index)
    if (hd->R) {
        for (size_t w = 0; w < hd->w; w++) {
            for (size_t b = 0; b < left_dim; b++) {
                for (size_t s1 = 0; s1 < d; s1++) {
                    for (size_t s2 = 0; s2 < d; s2++) {
                        for (size_t ap = 0; ap < right_dim; ap++) {
                            ComplexFloat sum = {0.0f, 0.0f};
                            for (size_t a = 0; a < right_dim; a++) {
                                // R[w, ap, a]
                                ComplexFloat r_val = hd->R[w * right_dim * right_dim + ap * right_dim + a];
                                size_t psi_idx = b * d2 * right_dim + s1 * d * right_dim + s2 * right_dim + a;
                                sum = cf_add(sum, cf_mul(r_val, psi[psi_idx]));
                            }
                            size_t out_idx = b * d2 * right_dim + s1 * d * right_dim + s2 * right_dim + ap;
                            // Only add if w corresponds to Hamiltonian action
                            if (w == 0) {
                                h_psi[out_idx] = cf_add(h_psi[out_idx], sum);
                            }
                        }
                    }
                }
            }
        }
    }
}

qgt_error_t dmrg_optimize_two_site(
    ComplexFloat* two_site,
    size_t dim,
    const ComplexFloat* h_eff,
    size_t h_eff_dim,
    double* energy,
    const DMRGConfig* config)
{
    // Use Lanczos to find ground state of effective Hamiltonian
    return lanczos_ground_state(
        h_eff,
        dim,
        two_site,
        energy,
        config->lanczos_iterations,
        config->lanczos_tolerance,
        NULL,
        NULL);
}

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
    bool direction)
{
    // Reshape two_site from [left_dim, d, d, right_dim] to matrix
    // For left-to-right: [left_dim * d, d * right_dim]
    // For right-to-left: [left_dim * d, d * right_dim] then transpose result

    size_t m = left_dim * d;
    size_t n = d * right_dim;
    size_t min_mn = (m < n) ? m : n;

    // Allocate SVD arrays
    float* S = malloc(min_mn * sizeof(float));
    ComplexFloat* U = malloc(m * min_mn * sizeof(ComplexFloat));
    ComplexFloat* Vt = malloc(min_mn * n * sizeof(ComplexFloat));
    ComplexFloat* matrix = malloc(m * n * sizeof(ComplexFloat));

    if (!S || !U || !Vt || !matrix) {
        free(S); free(U); free(Vt); free(matrix);
        return QGT_ERROR_NO_MEMORY;
    }

    // Copy and reshape two_site tensor
    memcpy(matrix, two_site, m * n * sizeof(ComplexFloat));

    // Perform SVD: two_site = U * S * Vt
    bool svd_success = lapack_svd(matrix, m, n, U, S, Vt, LAPACK_ROW_MAJOR);
    if (!svd_success) {
        free(S); free(U); free(Vt); free(matrix);
        return QGT_ERROR_INTERNAL;
    }

    // Determine bond dimension after truncation
    double total_weight = 0.0;
    for (size_t i = 0; i < min_mn; i++) {
        total_weight += S[i] * S[i];
    }

    double cumulative = 0.0;
    size_t bond = 0;
    for (size_t i = 0; i < min_mn && bond < max_bond; i++) {
        cumulative += S[i] * S[i];
        bond++;
        if ((total_weight - cumulative) / total_weight < cutoff * cutoff) {
            break;
        }
    }

    if (bond == 0) bond = 1;  // At least one singular value

    // Compute truncation error
    double kept_weight = 0.0;
    for (size_t i = 0; i < bond; i++) {
        kept_weight += S[i] * S[i];
    }
    *trunc_error = sqrt(fmax(0.0, total_weight - kept_weight)) / sqrt(total_weight);
    *new_bond = bond;

    // Allocate output tensors
    *A_left = malloc(left_dim * d * bond * sizeof(ComplexFloat));
    *A_right = malloc(bond * d * right_dim * sizeof(ComplexFloat));

    if (!*A_left || !*A_right) {
        free(*A_left); free(*A_right);
        free(S); free(U); free(Vt); free(matrix);
        return QGT_ERROR_NO_MEMORY;
    }

    if (direction) {
        // Moving right: A_left = U[:, :bond], A_right = S[:bond] * Vt[:bond, :]

        // Copy U to A_left: reshape from [m, bond] to [left_dim, d, bond]
        for (size_t b = 0; b < left_dim; b++) {
            for (size_t s = 0; s < d; s++) {
                for (size_t a = 0; a < bond; a++) {
                    size_t u_idx = (b * d + s) * min_mn + a;
                    size_t out_idx = b * d * bond + s * bond + a;
                    (*A_left)[out_idx] = U[u_idx];
                }
            }
        }

        // Compute S * Vt and reshape to A_right: [bond, d, right_dim]
        for (size_t a = 0; a < bond; a++) {
            for (size_t s = 0; s < d; s++) {
                for (size_t b = 0; b < right_dim; b++) {
                    size_t vt_idx = a * n + s * right_dim + b;
                    size_t out_idx = a * d * right_dim + s * right_dim + b;
                    (*A_right)[out_idx] = cf_scale(Vt[vt_idx], S[a]);
                }
            }
        }
    } else {
        // Moving left: A_left = U[:, :bond] * S[:bond], A_right = Vt[:bond, :]

        // Compute U * S and reshape to A_left: [left_dim, d, bond]
        for (size_t b = 0; b < left_dim; b++) {
            for (size_t s = 0; s < d; s++) {
                for (size_t a = 0; a < bond; a++) {
                    size_t u_idx = (b * d + s) * min_mn + a;
                    size_t out_idx = b * d * bond + s * bond + a;
                    (*A_left)[out_idx] = cf_scale(U[u_idx], S[a]);
                }
            }
        }

        // Copy Vt to A_right: reshape from [bond, n] to [bond, d, right_dim]
        for (size_t a = 0; a < bond; a++) {
            for (size_t s = 0; s < d; s++) {
                for (size_t b = 0; b < right_dim; b++) {
                    size_t vt_idx = a * n + s * right_dim + b;
                    size_t out_idx = a * d * right_dim + s * right_dim + b;
                    (*A_right)[out_idx] = Vt[vt_idx];
                }
            }
        }
    }

    free(S);
    free(U);
    free(Vt);
    free(matrix);

    return QGT_SUCCESS;
}

// ============================================================================
// DMRG Sweep
// ============================================================================

qgt_error_t dmrg_sweep(
    MatrixProductState* mps,
    const DMRGHamiltonian* hamiltonian,
    DMRGEnvironment* env,
    const DMRGConfig* config,
    bool direction,
    double* energy,
    double* max_trunc_error)
{
    if (!mps || !hamiltonian || !env || !config) return QGT_ERROR_INVALID_ARGUMENT;

    size_t n = mps->num_sites;
    size_t d = mps->physical_dim;

    *max_trunc_error = 0.0;
    *energy = 0.0;

    if (direction) {
        // Left-to-right sweep
        for (size_t site = 0; site < n - 1; site++) {
            size_t left_dim = (site == 0) ? 1 : mps->right_bond_dims[site - 1];
            size_t right_dim = (site + 1 == n - 1) ? 1 : mps->right_bond_dims[site + 1];
            size_t mid_dim = mps->right_bond_dims[site];

            // Form two-site tensor
            size_t two_site_dim = left_dim * d * d * right_dim;
            ComplexFloat* two_site = calloc(two_site_dim, sizeof(ComplexFloat));
            if (!two_site) return QGT_ERROR_NO_MEMORY;

            // Contract: Θ[b, s1, s2, a] = A1[b, s1, m] * A2[m, s2, a]
            const ComplexFloat* A1 = mps->tensors[site];
            const ComplexFloat* A2 = mps->tensors[site + 1];

            for (size_t b = 0; b < left_dim; b++) {
                for (size_t s1 = 0; s1 < d; s1++) {
                    for (size_t s2 = 0; s2 < d; s2++) {
                        for (size_t a = 0; a < right_dim; a++) {
                            ComplexFloat sum = {0.0f, 0.0f};
                            for (size_t m = 0; m < mid_dim; m++) {
                                // A1[b, s1, m]
                                ComplexFloat a1 = A1[b * d * mid_dim + s1 * mid_dim + m];
                                // A2[m, s2, a]
                                ComplexFloat a2 = A2[m * d * right_dim + s2 * right_dim + a];
                                sum = cf_add(sum, cf_mul(a1, a2));
                            }
                            two_site[b * d * d * right_dim + s1 * d * right_dim + s2 * right_dim + a] = sum;
                        }
                    }
                }
            }

            // Set up effective Hamiltonian data
            EffectiveHData hdata = {
                .L = env->left_blocks[site],
                .R = env->right_blocks[site + 1],
                .H_nn = hamiltonian->nn_terms[site],
                .H_L = hamiltonian->local_terms[site],
                .H_R = hamiltonian->local_terms[site + 1],
                .left_dim = left_dim,
                .d = d,
                .right_dim = right_dim,
                .w = env->mpo_bond_dim
            };

            // Optimize two-site tensor using Lanczos
            double site_energy;
            qgt_error_t err = lanczos_ground_state(
                NULL,
                two_site_dim,
                two_site,
                &site_energy,
                config->lanczos_iterations,
                config->lanczos_tolerance,
                effective_h_matvec,
                &hdata);

            if (err != QGT_SUCCESS) {
                free(two_site);
                return err;
            }

            *energy = site_energy;

            // Split two-site tensor
            ComplexFloat* A_left = NULL;
            ComplexFloat* A_right = NULL;
            size_t new_bond;
            double trunc_error;

            err = dmrg_split_two_site(
                two_site,
                left_dim,
                d,
                right_dim,
                &A_left,
                &A_right,
                &new_bond,
                config->max_bond_dim,
                config->truncation_cutoff,
                &trunc_error,
                true);  // Moving right

            free(two_site);

            if (err != QGT_SUCCESS) {
                free(A_left); free(A_right);
                return err;
            }

            if (trunc_error > *max_trunc_error) {
                *max_trunc_error = trunc_error;
            }

            // Update MPS tensors
            free(mps->tensors[site]);
            free(mps->tensors[site + 1]);
            mps->tensors[site] = A_left;
            mps->tensors[site + 1] = A_right;
            mps->right_bond_dims[site] = new_bond;

            // Update left environment
            dmrg_update_left_block(env, mps, hamiltonian, site);
        }
    } else {
        // Right-to-left sweep
        for (size_t site = n - 2; site < n - 1; site--) {  // Careful with size_t underflow
            size_t left_dim = (site == 0) ? 1 : mps->right_bond_dims[site - 1];
            size_t right_dim = (site + 1 == n - 1) ? 1 : mps->right_bond_dims[site + 1];
            size_t mid_dim = mps->right_bond_dims[site];

            // Form two-site tensor
            size_t two_site_dim = left_dim * d * d * right_dim;
            ComplexFloat* two_site = calloc(two_site_dim, sizeof(ComplexFloat));
            if (!two_site) return QGT_ERROR_NO_MEMORY;

            const ComplexFloat* A1 = mps->tensors[site];
            const ComplexFloat* A2 = mps->tensors[site + 1];

            for (size_t b = 0; b < left_dim; b++) {
                for (size_t s1 = 0; s1 < d; s1++) {
                    for (size_t s2 = 0; s2 < d; s2++) {
                        for (size_t a = 0; a < right_dim; a++) {
                            ComplexFloat sum = {0.0f, 0.0f};
                            for (size_t m = 0; m < mid_dim; m++) {
                                ComplexFloat a1 = A1[b * d * mid_dim + s1 * mid_dim + m];
                                ComplexFloat a2 = A2[m * d * right_dim + s2 * right_dim + a];
                                sum = cf_add(sum, cf_mul(a1, a2));
                            }
                            two_site[b * d * d * right_dim + s1 * d * right_dim + s2 * right_dim + a] = sum;
                        }
                    }
                }
            }

            EffectiveHData hdata = {
                .L = env->left_blocks[site],
                .R = env->right_blocks[site + 1],
                .H_nn = hamiltonian->nn_terms[site],
                .H_L = hamiltonian->local_terms[site],
                .H_R = hamiltonian->local_terms[site + 1],
                .left_dim = left_dim,
                .d = d,
                .right_dim = right_dim,
                .w = env->mpo_bond_dim
            };

            double site_energy;
            qgt_error_t err = lanczos_ground_state(
                NULL,
                two_site_dim,
                two_site,
                &site_energy,
                config->lanczos_iterations,
                config->lanczos_tolerance,
                effective_h_matvec,
                &hdata);

            if (err != QGT_SUCCESS) {
                free(two_site);
                return err;
            }

            *energy = site_energy;

            ComplexFloat* A_left = NULL;
            ComplexFloat* A_right = NULL;
            size_t new_bond;
            double trunc_error;

            err = dmrg_split_two_site(
                two_site,
                left_dim,
                d,
                right_dim,
                &A_left,
                &A_right,
                &new_bond,
                config->max_bond_dim,
                config->truncation_cutoff,
                &trunc_error,
                false);  // Moving left

            free(two_site);

            if (err != QGT_SUCCESS) {
                free(A_left); free(A_right);
                return err;
            }

            if (trunc_error > *max_trunc_error) {
                *max_trunc_error = trunc_error;
            }

            free(mps->tensors[site]);
            free(mps->tensors[site + 1]);
            mps->tensors[site] = A_left;
            mps->tensors[site + 1] = A_right;
            mps->right_bond_dims[site] = new_bond;

            // Update right environment
            dmrg_update_right_block(env, mps, hamiltonian, site + 1);

            if (site == 0) break;  // Prevent underflow
        }
    }

    return QGT_SUCCESS;
}

// ============================================================================
// Main DMRG Algorithm
// ============================================================================

qgt_error_t dmrg_ground_state(
    MatrixProductState* ground_state,
    const DMRGHamiltonian* hamiltonian,
    const DMRGConfig* config,
    DMRGResult* result)
{
    if (!ground_state || !hamiltonian || !config) return QGT_ERROR_INVALID_ARGUMENT;
    if (ground_state->num_sites != hamiltonian->num_sites) return QGT_ERROR_INVALID_ARGUMENT;

    size_t n = ground_state->num_sites;

    // Use provided config or default
    DMRGConfig cfg = config ? *config : dmrg_get_default_config();

    // Create environment
    DMRGEnvironment* env = dmrg_create_environment(n, 3);  // MPO bond dim = 3 for NN
    if (!env) return QGT_ERROR_NO_MEMORY;

    // Initialize environment from MPS
    qgt_error_t err = dmrg_init_environment(env, ground_state, hamiltonian);
    if (err != QGT_SUCCESS) {
        dmrg_destroy_environment(env);
        return err;
    }

    double prev_energy = 1e10;
    bool converged = false;
    size_t sweep = 0;

    // Main DMRG loop
    for (sweep = 0; sweep < cfg.max_sweeps && !converged; sweep++) {
        double energy = 0.0;
        double max_trunc = 0.0;

        // Determine bond dimension for this sweep
        size_t current_max_bond = cfg.max_bond_dim;
        if (sweep < cfg.warmup_sweeps && cfg.warmup_bond_dim > 0) {
            current_max_bond = cfg.warmup_bond_dim +
                (cfg.max_bond_dim - cfg.warmup_bond_dim) * sweep / cfg.warmup_sweeps;
        }

        // Create temporary config with current bond dim
        DMRGConfig sweep_cfg = cfg;
        sweep_cfg.max_bond_dim = current_max_bond;

        // Right-to-left sweep
        double sweep_energy;
        double sweep_trunc;
        err = dmrg_sweep(ground_state, hamiltonian, env, &sweep_cfg, false, &sweep_energy, &sweep_trunc);
        if (err != QGT_SUCCESS) {
            dmrg_destroy_environment(env);
            return err;
        }
        energy = sweep_energy;
        if (sweep_trunc > max_trunc) max_trunc = sweep_trunc;

        // Left-to-right sweep
        err = dmrg_sweep(ground_state, hamiltonian, env, &sweep_cfg, true, &sweep_energy, &sweep_trunc);
        if (err != QGT_SUCCESS) {
            dmrg_destroy_environment(env);
            return err;
        }
        energy = sweep_energy;
        if (sweep_trunc > max_trunc) max_trunc = sweep_trunc;

        // Store results
        if (result) {
            result->energies[sweep] = energy;
            result->truncation_errors[sweep] = max_trunc;
        }

        // Check convergence
        if (fabs(energy - prev_energy) < cfg.energy_tolerance) {
            converged = true;
        }
        prev_energy = energy;

        if (cfg.verbose) {
            printf("DMRG sweep %zu: E = %.10f, max_bond = %zu, trunc = %.2e\n",
                   sweep, energy, current_max_bond, max_trunc);
        }
    }

    // Finalize result
    if (result) {
        result->num_sweeps = sweep;
        result->converged = converged;
        result->final_energy = prev_energy;
        result->max_bond_dim_used = cfg.max_bond_dim;

        // Copy bond dimensions
        for (size_t i = 0; i < n - 1; i++) {
            result->bond_dimensions[i] = ground_state->right_bond_dims[i];
        }
    }

    dmrg_destroy_environment(env);

    return QGT_SUCCESS;
}

// ============================================================================
// Observable Computation
// ============================================================================

/**
 * @brief Compute two-site expectation value <O_{ij}>
 *
 * Computes the expectation value of a two-site operator acting on sites i and i+1.
 * Uses efficient contraction via transfer matrices.
 */
static qgt_error_t compute_two_site_expectation(
    const MatrixProductState* mps,
    const ComplexFloat* op,    // d^2 x d^2 operator
    size_t site,               // Left site index
    ComplexFloat* result)
{
    if (!mps || !op || !result) return QGT_ERROR_INVALID_ARGUMENT;
    if (site + 1 >= mps->num_sites) return QGT_ERROR_INVALID_ARGUMENT;

    size_t n = mps->num_sites;
    size_t d = mps->physical_dim;
    size_t d2 = d * d;

    // Get tensor dimensions
    size_t left_dim = (site == 0) ? 1 : mps->right_bond_dims[site - 1];
    size_t mid_dim = mps->right_bond_dims[site];
    size_t right_dim = (site + 1 == n - 1) ? 1 : mps->right_bond_dims[site + 1];

    // Get MPS tensors for the two sites
    const ComplexFloat* A_left = mps->tensors[site];
    const ComplexFloat* A_right = mps->tensors[site + 1];

    // Build left environment (contract all sites to the left)
    ComplexFloat* left_env = NULL;
    size_t left_env_dim = left_dim;

    if (site > 0) {
        // Start with identity
        left_env = calloc(left_dim * left_dim, sizeof(ComplexFloat));
        if (!left_env) return QGT_ERROR_NO_MEMORY;
        for (size_t i = 0; i < left_dim; i++) {
            left_env[i * left_dim + i] = (ComplexFloat){1.0f, 0.0f};
        }

        // Contract sites 0 to site-1
        for (size_t s = 0; s < site; s++) {
            size_t l_dim = (s == 0) ? 1 : mps->right_bond_dims[s - 1];
            size_t r_dim = mps->right_bond_dims[s];
            const ComplexFloat* A = mps->tensors[s];

            ComplexFloat* new_env = calloc(r_dim * r_dim, sizeof(ComplexFloat));
            if (!new_env) {
                free(left_env);
                return QGT_ERROR_NO_MEMORY;
            }

            // new_env[a', a] = Σ_{b', b, σ} left_env[b', b] * A*[b', σ, a'] * A[b, σ, a]
            for (size_t ap = 0; ap < r_dim; ap++) {
                for (size_t a = 0; a < r_dim; a++) {
                    ComplexFloat sum = {0.0f, 0.0f};
                    for (size_t sigma = 0; sigma < d; sigma++) {
                        for (size_t bp = 0; bp < l_dim; bp++) {
                            for (size_t b = 0; b < l_dim; b++) {
                                ComplexFloat e_val = (s == 0) ?
                                    ((bp == b) ? (ComplexFloat){1.0f, 0.0f} : (ComplexFloat){0.0f, 0.0f}) :
                                    left_env[bp * l_dim + b];

                                ComplexFloat a_conj = cf_conj(A[bp * d * r_dim + sigma * r_dim + ap]);
                                ComplexFloat a_val = A[b * d * r_dim + sigma * r_dim + a];

                                sum = cf_add(sum, cf_mul(cf_mul(e_val, a_conj), a_val));
                            }
                        }
                    }
                    new_env[ap * r_dim + a] = sum;
                }
            }

            free(left_env);
            left_env = new_env;
            left_env_dim = r_dim;
        }
    }

    // Build right environment (contract all sites to the right)
    ComplexFloat* right_env = NULL;
    size_t right_env_dim = right_dim;

    if (site + 2 < n) {
        // Start with identity
        right_env = calloc(right_dim * right_dim, sizeof(ComplexFloat));
        if (!right_env) {
            free(left_env);
            return QGT_ERROR_NO_MEMORY;
        }
        for (size_t i = 0; i < right_dim; i++) {
            right_env[i * right_dim + i] = (ComplexFloat){1.0f, 0.0f};
        }

        // Contract sites n-1 down to site+2
        for (size_t s = n - 1; s > site + 1; s--) {
            size_t l_dim = mps->right_bond_dims[s - 1];
            size_t r_dim = (s == n - 1) ? 1 : mps->right_bond_dims[s];
            const ComplexFloat* A = mps->tensors[s];

            ComplexFloat* new_env = calloc(l_dim * l_dim, sizeof(ComplexFloat));
            if (!new_env) {
                free(left_env); free(right_env);
                return QGT_ERROR_NO_MEMORY;
            }

            // new_env[b', b] = Σ_{a', a, σ} A[b', σ, a'] * right_env[a', a] * A*[b, σ, a]
            for (size_t bp = 0; bp < l_dim; bp++) {
                for (size_t b = 0; b < l_dim; b++) {
                    ComplexFloat sum = {0.0f, 0.0f};
                    for (size_t sigma = 0; sigma < d; sigma++) {
                        for (size_t ap = 0; ap < r_dim; ap++) {
                            for (size_t a = 0; a < r_dim; a++) {
                                ComplexFloat r_val = (s == n - 1) ?
                                    ((ap == a) ? (ComplexFloat){1.0f, 0.0f} : (ComplexFloat){0.0f, 0.0f}) :
                                    right_env[ap * r_dim + a];

                                ComplexFloat a_bra = A[bp * d * r_dim + sigma * r_dim + ap];
                                ComplexFloat a_ket = cf_conj(A[b * d * r_dim + sigma * r_dim + a]);

                                sum = cf_add(sum, cf_mul(cf_mul(a_bra, r_val), a_ket));
                            }
                        }
                    }
                    new_env[bp * l_dim + b] = sum;
                }
            }

            free(right_env);
            right_env = new_env;
            right_env_dim = l_dim;
        }
    }

    // Now contract the two-site block with the operator
    // <O> = Σ left_env[b', b] * A1*[b', σ1', m'] * A2*[m', σ2', a'] *
    //         O[(σ1', σ2'), (σ1, σ2)] *
    //         A1[b, σ1, m] * A2[m, σ2, a] * right_env[a', a]

    ComplexFloat total = {0.0f, 0.0f};

    for (size_t bp = 0; bp < left_dim; bp++) {
        for (size_t b = 0; b < left_dim; b++) {
            ComplexFloat l_val = (left_env == NULL) ?
                ((bp == b) ? (ComplexFloat){1.0f, 0.0f} : (ComplexFloat){0.0f, 0.0f}) :
                left_env[bp * left_dim + b];

            if (cf_norm_sq(l_val) < 1e-20f) continue;

            for (size_t ap = 0; ap < right_dim; ap++) {
                for (size_t a = 0; a < right_dim; a++) {
                    ComplexFloat r_val = (right_env == NULL) ?
                        ((ap == a) ? (ComplexFloat){1.0f, 0.0f} : (ComplexFloat){0.0f, 0.0f}) :
                        right_env[ap * right_dim + a];

                    if (cf_norm_sq(r_val) < 1e-20f) continue;

                    for (size_t s1p = 0; s1p < d; s1p++) {
                        for (size_t s2p = 0; s2p < d; s2p++) {
                            for (size_t s1 = 0; s1 < d; s1++) {
                                for (size_t s2 = 0; s2 < d; s2++) {
                                    // O[(s1', s2'), (s1, s2)]
                                    size_t bra_idx = s1p * d + s2p;
                                    size_t ket_idx = s1 * d + s2;
                                    ComplexFloat o_val = op[bra_idx * d2 + ket_idx];

                                    if (cf_norm_sq(o_val) < 1e-20f) continue;

                                    // Sum over middle bond
                                    for (size_t mp = 0; mp < mid_dim; mp++) {
                                        for (size_t m = 0; m < mid_dim; m++) {
                                            // A1*[bp, s1p, mp] * A2*[mp, s2p, ap]
                                            ComplexFloat a1_bra = cf_conj(A_left[bp * d * mid_dim + s1p * mid_dim + mp]);
                                            ComplexFloat a2_bra = cf_conj(A_right[mp * d * right_dim + s2p * right_dim + ap]);

                                            // A1[b, s1, m] * A2[m, s2, a]
                                            ComplexFloat a1_ket = A_left[b * d * mid_dim + s1 * mid_dim + m];
                                            ComplexFloat a2_ket = A_right[m * d * right_dim + s2 * right_dim + a];

                                            // Combine all factors
                                            ComplexFloat contrib = cf_mul(l_val, a1_bra);
                                            contrib = cf_mul(contrib, a2_bra);
                                            contrib = cf_mul(contrib, o_val);
                                            contrib = cf_mul(contrib, a1_ket);
                                            contrib = cf_mul(contrib, a2_ket);
                                            contrib = cf_mul(contrib, r_val);

                                            total = cf_add(total, contrib);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    free(left_env);
    free(right_env);

    *result = total;
    return QGT_SUCCESS;
}

qgt_error_t dmrg_compute_energy(
    const MatrixProductState* mps,
    const DMRGHamiltonian* hamiltonian,
    double* energy)
{
    if (!mps || !hamiltonian || !energy) return QGT_ERROR_INVALID_ARGUMENT;

    *energy = 0.0;

    size_t n = mps->num_sites;

    // Local terms: Σ_i <H_local_i>
    for (size_t site = 0; site < n; site++) {
        ComplexFloat local_exp;
        qgt_error_t err = mps_expectation_local(mps, hamiltonian->local_terms[site], site, &local_exp);
        if (err != QGT_SUCCESS) return err;
        *energy += local_exp.real;
    }

    // Nearest-neighbor terms: Σ_i <H_nn_i>
    for (size_t site = 0; site < n - 1; site++) {
        ComplexFloat nn_exp;
        qgt_error_t err = compute_two_site_expectation(mps, hamiltonian->nn_terms[site], site, &nn_exp);
        if (err != QGT_SUCCESS) return err;
        *energy += nn_exp.real;
    }

    return QGT_SUCCESS;
}

/**
 * @brief Decompose a two-site operator into sum of tensor products using SVD
 *
 * Given H_nn as d² x d² matrix, reshapes to (d x d) matrix and performs SVD
 * to get H_nn = Σ_k s_k * S_L[k] ⊗ S_R[k]
 *
 * @param nn_term The nearest-neighbor term (d² x d² matrix)
 * @param d Physical dimension
 * @param S_L Output: array of left operators (d x d each), caller must free
 * @param S_R Output: array of right operators (d x d each), caller must free
 * @param num_terms Output: number of non-zero singular values (coupling terms)
 * @param cutoff SVD cutoff for singular values
 * @return QGT_SUCCESS on success
 */
static qgt_error_t decompose_nn_term(
    const ComplexFloat* nn_term,
    size_t d,
    ComplexFloat** S_L,
    ComplexFloat** S_R,
    size_t* num_terms,
    double cutoff)
{
    size_t d2 = d * d;

    // Reshape nn_term from (d² x d²) to (d² x d²) for SVD
    // Interpret as matrix M[σ1*d+σ1', σ2*d+σ2'] = H_nn[(σ1,σ2), (σ1',σ2')]
    // SVD: M = U * S * V^H
    // Then S_L[k][σ1,σ1'] = U[σ1*d+σ1', k] * sqrt(s[k])
    //      S_R[k][σ2,σ2'] = V^H[k, σ2*d+σ2'] * sqrt(s[k])

    ComplexFloat* U = malloc(d2 * d2 * sizeof(ComplexFloat));
    ComplexFloat* VT = malloc(d2 * d2 * sizeof(ComplexFloat));
    float* S_vals = malloc(d2 * sizeof(float));

    if (!U || !VT || !S_vals) {
        free(U); free(VT); free(S_vals);
        return QGT_ERROR_NO_MEMORY;
    }

    // Perform SVD on the nn_term matrix
    if (!lapack_svd(nn_term, d2, d2, U, S_vals, VT, LAPACK_ROW_MAJOR)) {
        free(U); free(VT); free(S_vals);
        return QGT_ERROR_SVD_FAILED;
    }

    // Count significant singular values
    size_t count = 0;
    for (size_t k = 0; k < d2; k++) {
        if (S_vals[k] > cutoff) count++;
    }

    if (count == 0) {
        // No coupling terms (nn_term is effectively zero)
        *S_L = NULL;
        *S_R = NULL;
        *num_terms = 0;
        free(U); free(VT); free(S_vals);
        return QGT_SUCCESS;
    }

    // Allocate output arrays
    *S_L = malloc(count * d2 * sizeof(ComplexFloat));
    *S_R = malloc(count * d2 * sizeof(ComplexFloat));

    if (!*S_L || !*S_R) {
        free(*S_L); free(*S_R);
        free(U); free(VT); free(S_vals);
        return QGT_ERROR_NO_MEMORY;
    }

    // Extract coupling operators
    size_t term_idx = 0;
    for (size_t k = 0; k < d2 && term_idx < count; k++) {
        if (S_vals[k] <= cutoff) continue;

        float sqrt_s = sqrtf(S_vals[k]);

        // S_L[term][σ1, σ1'] = U[σ1*d+σ1', k] * sqrt(s[k])
        // S_R[term][σ2, σ2'] = V^H[k, σ2*d+σ2'] * sqrt(s[k])
        for (size_t i = 0; i < d2; i++) {
            // U is stored column-major from LAPACK, so U[i,k] = U[k*d2 + i]
            ComplexFloat u_val = U[k * d2 + i];
            (*S_L)[term_idx * d2 + i] = cf_scale(u_val, sqrt_s);

            // VT is V^H, stored row-major, so VT[k,i] = VT[k*d2 + i]
            ComplexFloat vt_val = VT[k * d2 + i];
            (*S_R)[term_idx * d2 + i] = cf_scale(vt_val, sqrt_s);
        }
        term_idx++;
    }

    *num_terms = count;

    free(U); free(VT); free(S_vals);
    return QGT_SUCCESS;
}

/**
 * @brief Apply MPO to MPS: |ψ'> = H |ψ>
 *
 * Creates a new MPS representing the application of the Hamiltonian MPO to the input MPS.
 * Uses SVD decomposition of nearest-neighbor terms to properly construct the MPO.
 * The result has increased bond dimension that may be compressed afterward.
 *
 * MPO structure for site i (with coupling term decomposition):
 *   W[0,0] = I               (identity pass-through)
 *   W[0,1..r] = S_L[1..r]    (left coupling operators from SVD)
 *   W[0,r+1] = H_local       (on-site Hamiltonian)
 *   W[1..r,r+1] = S_R[1..r]  (right coupling operators from SVD)
 *   W[r+1,r+1] = I           (identity pass-through)
 *
 * where r is the number of coupling terms from SVD decomposition.
 */
static qgt_error_t apply_mpo_to_mps(
    MatrixProductState** result,
    const MatrixProductState* mps,
    const DMRGHamiltonian* hamiltonian)
{
    if (!result || !mps || !hamiltonian) return QGT_ERROR_INVALID_ARGUMENT;

    size_t n = mps->num_sites;
    size_t d = mps->physical_dim;
    size_t d2 = d * d;
    const double svd_cutoff = 1e-12;

    // First, decompose all nearest-neighbor terms to determine MPO bond dimension
    ComplexFloat** all_S_L = calloc(n - 1, sizeof(ComplexFloat*));
    ComplexFloat** all_S_R = calloc(n - 1, sizeof(ComplexFloat*));
    size_t* num_coupling_terms = calloc(n - 1, sizeof(size_t));

    if (!all_S_L || !all_S_R || !num_coupling_terms) {
        free(all_S_L); free(all_S_R); free(num_coupling_terms);
        return QGT_ERROR_NO_MEMORY;
    }

    size_t max_coupling = 0;
    for (size_t bond = 0; bond < n - 1; bond++) {
        qgt_error_t err = decompose_nn_term(
            hamiltonian->nn_terms[bond], d,
            &all_S_L[bond], &all_S_R[bond], &num_coupling_terms[bond],
            svd_cutoff);
        if (err != QGT_SUCCESS) {
            for (size_t j = 0; j < bond; j++) {
                free(all_S_L[j]); free(all_S_R[j]);
            }
            free(all_S_L); free(all_S_R); free(num_coupling_terms);
            return err;
        }
        if (num_coupling_terms[bond] > max_coupling) {
            max_coupling = num_coupling_terms[bond];
        }
    }

    // MPO bond dimension: w = 2 + max_coupling (identity in, coupling terms, identity out)
    size_t w = 2 + max_coupling;
    if (w < 3) w = 3;  // Minimum for local + identity terms

    // Create result MPS with inflated bond dimension
    *result = mps_create(n, d, mps->max_bond_dim * w);
    if (!*result) {
        for (size_t j = 0; j < n - 1; j++) {
            free(all_S_L[j]); free(all_S_R[j]);
        }
        free(all_S_L); free(all_S_R); free(num_coupling_terms);
        return QGT_ERROR_NO_MEMORY;
    }

    // For each site, contract MPS tensor with MPO tensor
    for (size_t site = 0; site < n; site++) {
        size_t left_dim = (site == 0) ? 1 : mps->right_bond_dims[site - 1];
        size_t right_dim = (site == n - 1) ? 1 : mps->right_bond_dims[site];

        // Output dimensions: left MPS bond * w, right MPS bond * w
        // Boundary sites collapse the MPO indices
        size_t out_left = (site == 0) ? 1 : left_dim * w;
        size_t out_right = (site == n - 1) ? 1 : right_dim * w;

        const ComplexFloat* A = mps->tensors[site];
        const ComplexFloat* H_local = hamiltonian->local_terms[site];

        // Get coupling operators for bonds adjacent to this site
        ComplexFloat* S_L_left = (site > 0) ? all_S_L[site - 1] : NULL;
        ComplexFloat* S_R_left = (site > 0) ? all_S_R[site - 1] : NULL;
        size_t n_left = (site > 0) ? num_coupling_terms[site - 1] : 0;

        ComplexFloat* S_L_right = (site < n - 1) ? all_S_L[site] : NULL;
        size_t n_right = (site < n - 1) ? num_coupling_terms[site] : 0;

        // Allocate output tensor
        ComplexFloat* A_new = calloc(out_left * d * out_right, sizeof(ComplexFloat));
        if (!A_new) {
            mps_destroy(*result); *result = NULL;
            for (size_t j = 0; j < n - 1; j++) {
                free(all_S_L[j]); free(all_S_R[j]);
            }
            free(all_S_L); free(all_S_R); free(num_coupling_terms);
            return QGT_ERROR_NO_MEMORY;
        }

        // Build MPO tensor and contract with MPS tensor
        // MPO indices: w_L (left MPO bond), w_R (right MPO bond)
        // MPO structure:
        //   Row 0: [I, S_L[1], ..., S_L[r], H_local]
        //   Row 1..r: [0, 0, ..., 0, S_R[1..r]]
        //   Row w-1: [0, 0, ..., 0, I]

        if (site == 0) {
            // Left boundary: MPO starts from row 0 only
            // A'[σ', a*w_R] = Σ_σ A[σ, a] * W[0, w_R, σ', σ]
            for (size_t sigma_p = 0; sigma_p < d; sigma_p++) {
                for (size_t a = 0; a < right_dim; a++) {
                    for (size_t w_r = 0; w_r < w; w_r++) {
                        ComplexFloat sum = {0.0f, 0.0f};

                        for (size_t sigma = 0; sigma < d; sigma++) {
                            ComplexFloat a_val = A[sigma * right_dim + a];
                            ComplexFloat w_val = {0.0f, 0.0f};

                            if (w_r == 0 && sigma_p == sigma) {
                                // W[0,0] = I
                                w_val = (ComplexFloat){1.0f, 0.0f};
                            } else if (w_r >= 1 && w_r <= n_right && S_L_right) {
                                // W[0, 1..r] = S_L[1..r]
                                size_t term_idx = w_r - 1;
                                w_val = S_L_right[term_idx * d2 + sigma_p * d + sigma];
                            } else if (w_r == w - 1) {
                                // W[0, w-1] = H_local
                                w_val = H_local[sigma_p * d + sigma];
                            }

                            sum = cf_add(sum, cf_mul(a_val, w_val));
                        }

                        size_t out_idx = sigma_p * out_right + a * w + w_r;
                        A_new[out_idx] = sum;
                    }
                }
            }
        } else if (site == n - 1) {
            // Right boundary: MPO ends at column w-1 only
            // A'[b*w_L, σ'] = Σ_σ A[b, σ] * W[w_L, w-1, σ', σ]
            for (size_t b = 0; b < left_dim; b++) {
                for (size_t w_l = 0; w_l < w; w_l++) {
                    for (size_t sigma_p = 0; sigma_p < d; sigma_p++) {
                        ComplexFloat sum = {0.0f, 0.0f};

                        for (size_t sigma = 0; sigma < d; sigma++) {
                            ComplexFloat a_val = A[b * d + sigma];
                            ComplexFloat w_val = {0.0f, 0.0f};

                            if (w_l == 0) {
                                // W[0, w-1] = H_local
                                w_val = H_local[sigma_p * d + sigma];
                            } else if (w_l >= 1 && w_l <= n_left && S_R_left) {
                                // W[1..r, w-1] = S_R[1..r]
                                size_t term_idx = w_l - 1;
                                w_val = S_R_left[term_idx * d2 + sigma_p * d + sigma];
                            } else if (w_l == w - 1 && sigma_p == sigma) {
                                // W[w-1, w-1] = I
                                w_val = (ComplexFloat){1.0f, 0.0f};
                            }

                            sum = cf_add(sum, cf_mul(a_val, w_val));
                        }

                        size_t out_idx = (b * w + w_l) * d + sigma_p;
                        A_new[out_idx] = sum;
                    }
                }
            }
        } else {
            // Bulk site: full MPO structure
            for (size_t b = 0; b < left_dim; b++) {
                for (size_t w_l = 0; w_l < w; w_l++) {
                    for (size_t sigma_p = 0; sigma_p < d; sigma_p++) {
                        for (size_t a = 0; a < right_dim; a++) {
                            for (size_t w_r = 0; w_r < w; w_r++) {
                                ComplexFloat sum = {0.0f, 0.0f};

                                for (size_t sigma = 0; sigma < d; sigma++) {
                                    ComplexFloat a_val = A[b * d * right_dim + sigma * right_dim + a];
                                    ComplexFloat w_val = {0.0f, 0.0f};

                                    // Identity flow: W[0,0] = I
                                    if (w_l == 0 && w_r == 0 && sigma_p == sigma) {
                                        w_val = (ComplexFloat){1.0f, 0.0f};
                                    }
                                    // Identity flow: W[w-1, w-1] = I
                                    else if (w_l == w - 1 && w_r == w - 1 && sigma_p == sigma) {
                                        w_val = (ComplexFloat){1.0f, 0.0f};
                                    }
                                    // Local term: W[0, w-1] = H_local
                                    else if (w_l == 0 && w_r == w - 1) {
                                        w_val = H_local[sigma_p * d + sigma];
                                    }
                                    // Left coupling: W[0, 1..r] = S_L (for right bond)
                                    else if (w_l == 0 && w_r >= 1 && w_r <= n_right && S_L_right) {
                                        size_t term_idx = w_r - 1;
                                        w_val = S_L_right[term_idx * d2 + sigma_p * d + sigma];
                                    }
                                    // Right coupling: W[1..r, w-1] = S_R (for left bond)
                                    else if (w_l >= 1 && w_l <= n_left && w_r == w - 1 && S_R_left) {
                                        size_t term_idx = w_l - 1;
                                        w_val = S_R_left[term_idx * d2 + sigma_p * d + sigma];
                                    }
                                    // Coupling propagation: W[k, k] = I for intermediate terms
                                    else if (w_l == w_r && w_l >= 1 && w_l < w - 1 && sigma_p == sigma) {
                                        w_val = (ComplexFloat){1.0f, 0.0f};
                                    }

                                    sum = cf_add(sum, cf_mul(a_val, w_val));
                                }

                                size_t out_idx = (b * w + w_l) * d * out_right +
                                                sigma_p * out_right +
                                                a * w + w_r;
                                A_new[out_idx] = sum;
                            }
                        }
                    }
                }
            }
        }

        (*result)->tensors[site] = A_new;
        if (site < n - 1) {
            (*result)->right_bond_dims[site] = out_right;
        }
    }

    // Cleanup
    for (size_t j = 0; j < n - 1; j++) {
        free(all_S_L[j]); free(all_S_R[j]);
    }
    free(all_S_L); free(all_S_R); free(num_coupling_terms);

    return QGT_SUCCESS;
}

qgt_error_t dmrg_compute_variance(
    const MatrixProductState* mps,
    const DMRGHamiltonian* hamiltonian,
    double* variance)
{
    if (!mps || !hamiltonian || !variance) return QGT_ERROR_INVALID_ARGUMENT;

    // Variance = <H²> - <H>²
    // where <H²> = <ψ|H²|ψ> = <Hψ|Hψ>
    //
    // Algorithm:
    // 1. Compute <H> using energy calculation
    // 2. Apply MPO to get |Hψ⟩ = H|ψ⟩
    // 3. Compute <Hψ|Hψ⟩ = <H²> using MPS inner product
    // 4. Variance = <H²> - <H>²

    // Step 1: Compute <H>
    double energy;
    qgt_error_t err = dmrg_compute_energy(mps, hamiltonian, &energy);
    if (err != QGT_SUCCESS) return err;

    // Step 2: Apply MPO to get H|ψ⟩
    MatrixProductState* h_psi = NULL;
    err = apply_mpo_to_mps(&h_psi, mps, hamiltonian);
    if (err != QGT_SUCCESS) return err;

    // Step 3: Compute <Hψ|Hψ⟩ = <ψ|H²|ψ⟩
    // Note: The result MPS from apply_mpo_to_mps has inflated bond dimension
    // For numerical stability, we should compress it first, but for variance
    // calculation we can compute the inner product directly

    ComplexFloat h2_inner;
    err = mps_inner_product(h_psi, h_psi, &h2_inner);
    mps_destroy(h_psi);

    if (err != QGT_SUCCESS) return err;

    // <H²> should be real and positive for a Hermitian Hamiltonian
    double h2_expectation = h2_inner.real;

    // Step 4: Variance = <H²> - <H>²
    *variance = h2_expectation - energy * energy;

    // Ensure non-negative (numerical errors can cause slight negative values)
    if (*variance < 0.0) *variance = 0.0;

    return QGT_SUCCESS;
}
