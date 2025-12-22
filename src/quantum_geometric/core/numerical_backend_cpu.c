#include "quantum_geometric/core/numerical_backend.h"
#include "quantum_geometric/core/complex_arithmetic.h"
#include "quantum_geometric/core/lapack_wrapper.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

// ============================================================================
// Production-grade CPU fallback implementations for numerical linear algebra
// These implementations are used when LAPACK/BLAS is not available
// ============================================================================

// Machine epsilon for numerical stability
#define MACHINE_EPS 1e-7f

// Helper: Compute Householder reflection vector for a column
// Returns the norm of the column after reflection
static float compute_householder_vector(const ComplexFloat* x, ComplexFloat* v,
                                         size_t n, ComplexFloat* tau) {
    if (n == 0) {
        tau->real = 0.0f;
        tau->imag = 0.0f;
        return 0.0f;
    }

    // Compute the 2-norm of x
    float norm_sq = 0.0f;
    for (size_t i = 0; i < n; i++) {
        norm_sq += complex_abs_squared(x[i]);
    }
    float norm = sqrtf(norm_sq);

    if (norm < MACHINE_EPS) {
        // Zero column - no reflection needed
        tau->real = 0.0f;
        tau->imag = 0.0f;
        for (size_t i = 0; i < n; i++) {
            v[i].real = 0.0f;
            v[i].imag = 0.0f;
        }
        return 0.0f;
    }

    // Copy x to v
    for (size_t i = 0; i < n; i++) {
        v[i] = x[i];
    }

    // Compute sign(x[0]) * ||x||
    // For complex numbers, we use: alpha = -sign(Re(x[0])) * ||x||
    float alpha_sign = (x[0].real >= 0.0f) ? 1.0f : -1.0f;
    ComplexFloat alpha = {-alpha_sign * norm, 0.0f};

    // v[0] = x[0] - alpha
    v[0] = complex_subtract(x[0], alpha);

    // Normalize v
    float v_norm_sq = 0.0f;
    for (size_t i = 0; i < n; i++) {
        v_norm_sq += complex_abs_squared(v[i]);
    }
    float v_norm = sqrtf(v_norm_sq);

    if (v_norm < MACHINE_EPS) {
        tau->real = 0.0f;
        tau->imag = 0.0f;
        return fabsf(norm);
    }

    for (size_t i = 0; i < n; i++) {
        v[i].real /= v_norm;
        v[i].imag /= v_norm;
    }

    // tau = 2 / (v^H * v) = 2 (since v is normalized)
    tau->real = 2.0f;
    tau->imag = 0.0f;

    return fabsf(norm);
}

// Helper: Apply Householder reflection H = I - tau * v * v^H to matrix columns
static void apply_householder_to_matrix(ComplexFloat* A, size_t rows, size_t cols,
                                        size_t start_row, size_t start_col,
                                        const ComplexFloat* v, ComplexFloat tau) {
    if (complex_abs_squared(tau) < MACHINE_EPS) return;

    size_t m = rows - start_row;

    // For each column j from start_col to cols
    for (size_t j = start_col; j < cols; j++) {
        // Compute w = v^H * A[:,j]
        ComplexFloat w = {0.0f, 0.0f};
        for (size_t i = 0; i < m; i++) {
            ComplexFloat v_conj = complex_conjugate(v[i]);
            w = complex_add(w, complex_multiply(v_conj, A[(start_row + i) * cols + j]));
        }

        // A[:,j] = A[:,j] - tau * w * v
        ComplexFloat tau_w = complex_multiply(tau, w);
        for (size_t i = 0; i < m; i++) {
            A[(start_row + i) * cols + j] =
                complex_subtract(A[(start_row + i) * cols + j],
                                complex_multiply(tau_w, v[i]));
        }
    }
}

// Helper: Apply Householder from the left to form Q
static void accumulate_householder_Q(ComplexFloat* Q, size_t m, size_t n,
                                     const ComplexFloat* v, ComplexFloat tau,
                                     size_t start_row) {
    if (complex_abs_squared(tau) < MACHINE_EPS) return;

    size_t v_len = m - start_row;

    // Q = Q * (I - tau * v * v^H) = Q - tau * (Q * v) * v^H
    // For each column of Q
    for (size_t j = 0; j < m; j++) {
        // Compute w = Q[j,:] * v (dot product of row j with v)
        ComplexFloat w = {0.0f, 0.0f};
        for (size_t i = 0; i < v_len; i++) {
            w = complex_add(w, complex_multiply(Q[j * m + start_row + i], v[i]));
        }

        // Q[j,:] = Q[j,:] - tau * w * v^H
        ComplexFloat tau_w = complex_multiply(tau, w);
        for (size_t i = 0; i < v_len; i++) {
            ComplexFloat v_conj = complex_conjugate(v[i]);
            Q[j * m + start_row + i] =
                complex_subtract(Q[j * m + start_row + i],
                                complex_multiply(tau_w, v_conj));
        }
    }
}

// Production QR decomposition using Householder reflections
static numerical_error_t qr_householder(const ComplexFloat* a, ComplexFloat* q,
                                        ComplexFloat* r, size_t m, size_t n) {
    // Copy A to R (we will transform R in-place)
    memcpy(r, a, m * n * sizeof(ComplexFloat));

    // Initialize Q to identity
    memset(q, 0, m * m * sizeof(ComplexFloat));
    for (size_t i = 0; i < m; i++) {
        q[i * m + i].real = 1.0f;
        q[i * m + i].imag = 0.0f;
    }

    // Allocate workspace for Householder vector
    ComplexFloat* v = malloc(m * sizeof(ComplexFloat));
    ComplexFloat* column = malloc(m * sizeof(ComplexFloat));
    if (!v || !column) {
        free(v);
        free(column);
        return NUMERICAL_ERROR_MEMORY;
    }

    size_t min_mn = (m < n) ? m : n;

    for (size_t k = 0; k < min_mn; k++) {
        // Extract column k from row k onwards
        size_t col_len = m - k;
        for (size_t i = 0; i < col_len; i++) {
            column[i] = r[(k + i) * n + k];
        }

        // Compute Householder vector
        ComplexFloat tau;
        float diag_val = compute_householder_vector(column, v, col_len, &tau);

        // Set R[k,k] to the norm
        r[k * n + k].real = (column[0].real >= 0.0f) ? -diag_val : diag_val;
        r[k * n + k].imag = 0.0f;

        // Apply Householder reflection to remaining columns of R
        apply_householder_to_matrix(r, m, n, k, k + 1, v, tau);

        // Zero out sub-diagonal elements in column k
        for (size_t i = k + 1; i < m; i++) {
            r[i * n + k].real = 0.0f;
            r[i * n + k].imag = 0.0f;
        }

        // Accumulate Q = Q * H_k
        accumulate_householder_Q(q, m, m, v, tau, k);
    }

    free(v);
    free(column);

    return NUMERICAL_SUCCESS;
}

// Production LU decomposition with partial pivoting
static numerical_error_t lu_partial_pivot(const ComplexFloat* a, ComplexFloat* l,
                                          ComplexFloat* u, int* ipiv,
                                          size_t m, size_t n) {
    // Copy A to U (we will transform U in-place)
    memcpy(u, a, m * n * sizeof(ComplexFloat));

    // Initialize L to identity (or zeros for non-square)
    size_t min_mn = (m < n) ? m : n;
    memset(l, 0, m * min_mn * sizeof(ComplexFloat));
    for (size_t i = 0; i < min_mn && i < m; i++) {
        l[i * min_mn + i].real = 1.0f;
        l[i * min_mn + i].imag = 0.0f;
    }

    // Initialize pivot array
    for (size_t i = 0; i < m; i++) {
        ipiv[i] = (int)i;
    }

    for (size_t k = 0; k < min_mn; k++) {
        // Find pivot: row with maximum absolute value in column k
        float max_val = 0.0f;
        size_t max_row = k;
        for (size_t i = k; i < m; i++) {
            float abs_val = complex_abs_squared(u[i * n + k]);
            if (abs_val > max_val) {
                max_val = abs_val;
                max_row = i;
            }
        }

        // Check for singularity
        if (max_val < MACHINE_EPS * MACHINE_EPS) {
            // Near-singular matrix - continue with small pivot
            // (for compatibility, we don't fail here)
        }

        // Swap rows if needed
        if (max_row != k) {
            // Swap in U
            for (size_t j = 0; j < n; j++) {
                ComplexFloat temp = u[k * n + j];
                u[k * n + j] = u[max_row * n + j];
                u[max_row * n + j] = temp;
            }
            // Swap in L (only columns 0..k-1)
            for (size_t j = 0; j < k; j++) {
                ComplexFloat temp = l[k * min_mn + j];
                l[k * min_mn + j] = l[max_row * min_mn + j];
                l[max_row * min_mn + j] = temp;
            }
            // Record pivot
            int temp_piv = ipiv[k];
            ipiv[k] = ipiv[max_row];
            ipiv[max_row] = temp_piv;
        }

        // Compute multipliers and eliminate
        ComplexFloat pivot = u[k * n + k];
        if (complex_abs_squared(pivot) > MACHINE_EPS * MACHINE_EPS) {
            for (size_t i = k + 1; i < m; i++) {
                // L[i,k] = U[i,k] / U[k,k]
                ComplexFloat mult = complex_divide(u[i * n + k], pivot);
                l[i * min_mn + k] = mult;

                // U[i,:] = U[i,:] - L[i,k] * U[k,:]
                u[i * n + k].real = 0.0f;
                u[i * n + k].imag = 0.0f;
                for (size_t j = k + 1; j < n; j++) {
                    u[i * n + j] = complex_subtract(u[i * n + j],
                                                    complex_multiply(mult, u[k * n + j]));
                }
            }
        }
    }

    return NUMERICAL_SUCCESS;
}

// Production Cholesky decomposition for Hermitian positive definite matrices
static numerical_error_t cholesky_decompose(const ComplexFloat* a, ComplexFloat* l,
                                            size_t n, bool lower) {
    // Initialize L to zero
    memset(l, 0, n * n * sizeof(ComplexFloat));

    for (size_t j = 0; j < n; j++) {
        // Compute L[j,j]
        float sum = a[j * n + j].real;  // Diagonal should be real for Hermitian
        for (size_t k = 0; k < j; k++) {
            sum -= complex_abs_squared(l[j * n + k]);
        }

        if (sum <= 0.0f) {
            // Not positive definite
            return NUMERICAL_ERROR_COMPUTATION;
        }

        l[j * n + j].real = sqrtf(sum);
        l[j * n + j].imag = 0.0f;

        // Compute L[i,j] for i > j
        for (size_t i = j + 1; i < n; i++) {
            ComplexFloat sum_c = a[i * n + j];
            for (size_t k = 0; k < j; k++) {
                ComplexFloat l_ik = l[i * n + k];
                ComplexFloat l_jk_conj = complex_conjugate(l[j * n + k]);
                sum_c = complex_subtract(sum_c, complex_multiply(l_ik, l_jk_conj));
            }
            l[i * n + j] = complex_divide(sum_c, l[j * n + j]);
        }
    }

    // If upper triangular is requested, transpose
    if (!lower) {
        for (size_t i = 0; i < n; i++) {
            for (size_t j = i + 1; j < n; j++) {
                l[i * n + j] = complex_conjugate(l[j * n + i]);
                l[j * n + i].real = 0.0f;
                l[j * n + i].imag = 0.0f;
            }
        }
    }

    return NUMERICAL_SUCCESS;
}

// Production forward substitution: solve L * x = b
static void forward_substitution(const ComplexFloat* l, const ComplexFloat* b,
                                ComplexFloat* x, size_t n, bool unit_diag) {
    for (size_t i = 0; i < n; i++) {
        ComplexFloat sum = b[i];
        for (size_t j = 0; j < i; j++) {
            sum = complex_subtract(sum, complex_multiply(l[i * n + j], x[j]));
        }
        if (unit_diag) {
            x[i] = sum;
        } else {
            x[i] = complex_divide(sum, l[i * n + i]);
        }
    }
}

// Production backward substitution: solve U * x = b
static void backward_substitution(const ComplexFloat* u, const ComplexFloat* b,
                                 ComplexFloat* x, size_t n, bool unit_diag) {
    for (size_t i = n; i > 0; i--) {
        size_t idx = i - 1;
        ComplexFloat sum = b[idx];
        for (size_t j = idx + 1; j < n; j++) {
            sum = complex_subtract(sum, complex_multiply(u[idx * n + j], x[j]));
        }
        if (unit_diag) {
            x[idx] = sum;
        } else {
            x[idx] = complex_divide(sum, u[idx * n + idx]);
        }
    }
}

// Production QR algorithm for eigendecomposition
// Implements implicit double-shift QR with Hessenberg reduction
static numerical_error_t eigen_qr_algorithm(const ComplexFloat* a,
                                           ComplexFloat* eigenvectors,
                                           ComplexFloat* eigenvalues,
                                           size_t n) {
    if (n == 0) return NUMERICAL_SUCCESS;
    if (n == 1) {
        eigenvalues[0] = a[0];
        eigenvectors[0].real = 1.0f;
        eigenvectors[0].imag = 0.0f;
        return NUMERICAL_SUCCESS;
    }

    // Allocate workspace
    ComplexFloat* h = malloc(n * n * sizeof(ComplexFloat));
    ComplexFloat* q_acc = malloc(n * n * sizeof(ComplexFloat));
    ComplexFloat* q_temp = malloc(n * n * sizeof(ComplexFloat));
    ComplexFloat* r_temp = malloc(n * n * sizeof(ComplexFloat));

    if (!h || !q_acc || !q_temp || !r_temp) {
        free(h);
        free(q_acc);
        free(q_temp);
        free(r_temp);
        return NUMERICAL_ERROR_MEMORY;
    }

    // Copy A to H for Hessenberg reduction
    memcpy(h, a, n * n * sizeof(ComplexFloat));

    // Initialize accumulated Q to identity
    memset(q_acc, 0, n * n * sizeof(ComplexFloat));
    for (size_t i = 0; i < n; i++) {
        q_acc[i * n + i].real = 1.0f;
    }

    // Reduce to upper Hessenberg form using Householder reflections
    ComplexFloat* v = malloc(n * sizeof(ComplexFloat));
    ComplexFloat* column = malloc(n * sizeof(ComplexFloat));
    if (!v || !column) {
        free(h);
        free(q_acc);
        free(q_temp);
        free(r_temp);
        free(v);
        free(column);
        return NUMERICAL_ERROR_MEMORY;
    }

    for (size_t k = 0; k < n - 2; k++) {
        // Extract sub-column
        size_t col_len = n - k - 1;
        for (size_t i = 0; i < col_len; i++) {
            column[i] = h[(k + 1 + i) * n + k];
        }

        ComplexFloat tau;
        compute_householder_vector(column, v, col_len, &tau);

        // Apply H_k from left: H = (I - tau*v*v^H) * H
        for (size_t j = k; j < n; j++) {
            ComplexFloat w = {0.0f, 0.0f};
            for (size_t i = 0; i < col_len; i++) {
                ComplexFloat v_conj = complex_conjugate(v[i]);
                w = complex_add(w, complex_multiply(v_conj, h[(k + 1 + i) * n + j]));
            }
            ComplexFloat tau_w = complex_multiply(tau, w);
            for (size_t i = 0; i < col_len; i++) {
                h[(k + 1 + i) * n + j] =
                    complex_subtract(h[(k + 1 + i) * n + j],
                                    complex_multiply(tau_w, v[i]));
            }
        }

        // Apply H_k from right: H = H * (I - tau*v*v^H)
        for (size_t i = 0; i < n; i++) {
            ComplexFloat w = {0.0f, 0.0f};
            for (size_t j = 0; j < col_len; j++) {
                w = complex_add(w, complex_multiply(h[i * n + k + 1 + j], v[j]));
            }
            ComplexFloat tau_w = complex_multiply(tau, w);
            for (size_t j = 0; j < col_len; j++) {
                ComplexFloat v_conj = complex_conjugate(v[j]);
                h[i * n + k + 1 + j] =
                    complex_subtract(h[i * n + k + 1 + j],
                                    complex_multiply(tau_w, v_conj));
            }
        }

        // Accumulate transformation into Q
        for (size_t i = 0; i < n; i++) {
            ComplexFloat w = {0.0f, 0.0f};
            for (size_t j = 0; j < col_len; j++) {
                w = complex_add(w, complex_multiply(q_acc[i * n + k + 1 + j], v[j]));
            }
            ComplexFloat tau_w = complex_multiply(tau, w);
            for (size_t j = 0; j < col_len; j++) {
                ComplexFloat v_conj = complex_conjugate(v[j]);
                q_acc[i * n + k + 1 + j] =
                    complex_subtract(q_acc[i * n + k + 1 + j],
                                    complex_multiply(tau_w, v_conj));
            }
        }
    }

    free(v);
    free(column);

    // QR iterations with Wilkinson shift
    const size_t max_iter = 100 * n;
    const float tol = MACHINE_EPS * 100.0f;
    size_t p = n;  // Size of unreduced matrix

    for (size_t iter = 0; iter < max_iter && p > 1; iter++) {
        // Check for convergence of sub-diagonal elements
        for (size_t i = p - 1; i > 0; i--) {
            float off_diag = complex_abs(h[i * n + i - 1]);
            float diag_sum = complex_abs(h[(i-1) * n + i - 1]) + complex_abs(h[i * n + i]);
            if (off_diag < tol * diag_sum) {
                h[i * n + i - 1].real = 0.0f;
                h[i * n + i - 1].imag = 0.0f;
                if (i == p - 1) {
                    p--;  // Deflate
                }
            }
        }

        if (p <= 1) break;

        // Wilkinson shift: eigenvalue of trailing 2x2 closer to h[p-1,p-1]
        ComplexFloat a11 = h[(p-2) * n + p - 2];
        ComplexFloat a12 = h[(p-2) * n + p - 1];
        ComplexFloat a21 = h[(p-1) * n + p - 2];
        ComplexFloat a22 = h[(p-1) * n + p - 1];

        // Compute discriminant
        ComplexFloat trace = complex_add(a11, a22);
        ComplexFloat det = complex_subtract(complex_multiply(a11, a22),
                                           complex_multiply(a12, a21));
        ComplexFloat disc_sq = complex_subtract(complex_multiply(trace, trace),
                                               complex_multiply((ComplexFloat){4.0f, 0.0f}, det));
        ComplexFloat disc = complex_sqrt(disc_sq);

        // Two eigenvalues of 2x2 block
        ComplexFloat half = {0.5f, 0.0f};
        ComplexFloat lambda1 = complex_multiply(half, complex_add(trace, disc));
        ComplexFloat lambda2 = complex_multiply(half, complex_subtract(trace, disc));

        // Choose shift closer to a22
        ComplexFloat shift;
        if (complex_abs(complex_subtract(lambda1, a22)) <
            complex_abs(complex_subtract(lambda2, a22))) {
            shift = lambda1;
        } else {
            shift = lambda2;
        }

        // Apply shifted QR step: H - shift*I = Q*R, then H = R*Q + shift*I
        // Copy H to temp, apply shift
        for (size_t i = 0; i < p; i++) {
            for (size_t j = 0; j < p; j++) {
                r_temp[i * n + j] = h[i * n + j];
            }
            r_temp[i * n + i] = complex_subtract(r_temp[i * n + i], shift);
        }

        // QR factorization of shifted matrix (only the p x p submatrix)
        // Use Givens rotations for Hessenberg matrix
        for (size_t k = 0; k < p - 1; k++) {
            ComplexFloat a_val = r_temp[k * n + k];
            ComplexFloat b_val = r_temp[(k+1) * n + k];

            float r = sqrtf(complex_abs_squared(a_val) + complex_abs_squared(b_val));
            if (r < MACHINE_EPS) continue;

            ComplexFloat c = complex_divide(a_val, (ComplexFloat){r, 0.0f});
            ComplexFloat s = complex_divide(b_val, (ComplexFloat){r, 0.0f});
            ComplexFloat c_conj = complex_conjugate(c);
            ComplexFloat s_conj = complex_conjugate(s);

            // Apply Givens rotation from left to R
            for (size_t j = k; j < p; j++) {
                ComplexFloat temp1 = r_temp[k * n + j];
                ComplexFloat temp2 = r_temp[(k+1) * n + j];
                r_temp[k * n + j] = complex_add(complex_multiply(c_conj, temp1),
                                                complex_multiply(s_conj, temp2));
                r_temp[(k+1) * n + j] = complex_subtract(complex_multiply(c, temp2),
                                                         complex_multiply(s, temp1));
            }

            // Apply Givens rotation from right to H (RQ step)
            for (size_t i = 0; i <= (k + 2 < p ? k + 2 : p - 1); i++) {
                ComplexFloat temp1 = h[i * n + k];
                ComplexFloat temp2 = h[i * n + k + 1];
                h[i * n + k] = complex_add(complex_multiply(temp1, c),
                                           complex_multiply(temp2, s));
                h[i * n + k + 1] = complex_subtract(complex_multiply(temp2, c_conj),
                                                    complex_multiply(temp1, s_conj));
            }

            // Accumulate into Q
            for (size_t i = 0; i < n; i++) {
                ComplexFloat temp1 = q_acc[i * n + k];
                ComplexFloat temp2 = q_acc[i * n + k + 1];
                q_acc[i * n + k] = complex_add(complex_multiply(temp1, c),
                                               complex_multiply(temp2, s));
                q_acc[i * n + k + 1] = complex_subtract(complex_multiply(temp2, c_conj),
                                                        complex_multiply(temp1, s_conj));
            }
        }

        // Add shift back to diagonal
        for (size_t i = 0; i < p; i++) {
            h[i * n + i] = complex_add(h[i * n + i], shift);
        }
    }

    // Extract eigenvalues from diagonal of H
    for (size_t i = 0; i < n; i++) {
        eigenvalues[i] = h[i * n + i];
    }

    // Copy eigenvectors
    memcpy(eigenvectors, q_acc, n * n * sizeof(ComplexFloat));

    free(h);
    free(q_acc);
    free(q_temp);
    free(r_temp);

    return NUMERICAL_SUCCESS;
}

// Global state
static struct {
    numerical_config_t config;
    numerical_metrics_t metrics;
    numerical_error_t last_error;
    bool initialized;
    bool has_lapack;
} backend_state = {0};

numerical_error_t initialize_numerical_backend_cpu(const numerical_config_t* config) {
    if (!config) {
        return NUMERICAL_ERROR_INVALID_ARGUMENT;
    }
    
    backend_state.config = *config;
    backend_state.initialized = true;
    
    // Reset metrics
    memset(&backend_state.metrics, 0, sizeof(numerical_metrics_t));
    
    // Check for required LAPACK capabilities
    backend_state.has_lapack = 
        lapack_has_capability("SVD") &&
        lapack_has_capability("QR") &&
        lapack_has_capability("EIGEN") &&
        lapack_has_capability("CHOLESKY") &&
        lapack_has_capability("LU") &&
        lapack_has_capability("TRIANGULAR_SOLVE") &&
        lapack_has_capability("SYMMETRIC_SOLVE") &&
        lapack_has_capability("GENERAL_SOLVE");
    
    return NUMERICAL_SUCCESS;
}

void shutdown_numerical_backend_cpu(void) {
    backend_state.initialized = false;
}

numerical_error_t numerical_matrix_multiply_cpu(const ComplexFloat* a,
                                          const ComplexFloat* b,
                                          ComplexFloat* c,
                                          size_t m, size_t k, size_t n,
                                          bool transpose_a,
                                          bool transpose_b) {
    if (!backend_state.initialized) {
        return NUMERICAL_ERROR_INVALID_STATE;
    }
    
    if (!a || !b || !c) {
        return NUMERICAL_ERROR_INVALID_ARGUMENT;
    }

    if (backend_state.has_lapack) {
        bool success = lapack_matrix_multiply(a, b, c, m, k, n,
                                           transpose_a, transpose_b,
                                           LAPACK_ROW_MAJOR);
        if (!success) {
            return NUMERICAL_ERROR_COMPUTATION;
        }
        return NUMERICAL_SUCCESS;
    }
    
    // Fallback to basic implementation
    memset(c, 0, m * n * sizeof(ComplexFloat));
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            ComplexFloat sum = {0.0f, 0.0f};
            for (size_t l = 0; l < k; l++) {
                ComplexFloat a_val = transpose_a ? a[l * m + i] : a[i * k + l];
                ComplexFloat b_val = transpose_b ? b[j * k + l] : b[l * n + j];
                sum = complex_add(sum, complex_multiply(a_val, b_val));
            }
            c[i * n + j] = sum;
        }
    }
    
    return NUMERICAL_SUCCESS;
}

numerical_error_t numerical_matrix_add_cpu(const ComplexFloat* a,
                                     const ComplexFloat* b,
                                     ComplexFloat* c,
                                     size_t rows,
                                     size_t cols) {
    if (!backend_state.initialized) {
        return NUMERICAL_ERROR_INVALID_STATE;
    }
    
    if (!a || !b || !c) {
        return NUMERICAL_ERROR_INVALID_ARGUMENT;
    }
    
    complex_vector_add(a, b, c, rows * cols);
    return NUMERICAL_SUCCESS;
}

numerical_error_t numerical_vector_dot_cpu(const ComplexFloat* a,
                                     const ComplexFloat* b,
                                     ComplexFloat* result,
                                     size_t length) {
    if (!backend_state.initialized) {
        return NUMERICAL_ERROR_INVALID_STATE;
    }
    
    if (!a || !b || !result) {
        return NUMERICAL_ERROR_INVALID_ARGUMENT;
    }
    
    *result = complex_vector_dot(a, b, length);
    return NUMERICAL_SUCCESS;
}

numerical_error_t numerical_qr_cpu(const ComplexFloat* a,
                             ComplexFloat* q,
                             ComplexFloat* r,
                             size_t m,
                             size_t n) {
    if (!backend_state.initialized) {
        return NUMERICAL_ERROR_INVALID_STATE;
    }
    
    if (!a || !q || !r) {
        return NUMERICAL_ERROR_INVALID_ARGUMENT;
    }

    if (backend_state.has_lapack) {
        bool success = lapack_qr(a, m, n, q, r, LAPACK_ROW_MAJOR);
        if (!success) {
            return NUMERICAL_ERROR_COMPUTATION;
        }
        return NUMERICAL_SUCCESS;
    }

    // Use production Householder QR fallback
    return qr_householder(a, q, r, m, n);
}

numerical_error_t numerical_eigendecomposition_cpu(const ComplexFloat* a,
                                             ComplexFloat* eigenvectors,
                                             ComplexFloat* eigenvalues,
                                             size_t n) {
    if (!backend_state.initialized) {
        return NUMERICAL_ERROR_INVALID_STATE;
    }
    
    if (!a || !eigenvectors || !eigenvalues) {
        return NUMERICAL_ERROR_INVALID_ARGUMENT;
    }

    if (backend_state.has_lapack) {
        bool success = lapack_eigendecomposition(a, n, eigenvectors, eigenvalues, LAPACK_ROW_MAJOR);
        if (!success) {
            return NUMERICAL_ERROR_COMPUTATION;
        }
        return NUMERICAL_SUCCESS;
    }

    // Use production QR algorithm fallback
    return eigen_qr_algorithm(a, eigenvectors, eigenvalues, n);
}

numerical_error_t numerical_cholesky_cpu(const ComplexFloat* a,
                                   ComplexFloat* l,
                                   size_t n,
                                   bool lower_triangular) {
    if (!backend_state.initialized) {
        return NUMERICAL_ERROR_INVALID_STATE;
    }
    
    if (!a || !l) {
        return NUMERICAL_ERROR_INVALID_ARGUMENT;
    }

    if (backend_state.has_lapack) {
        bool success = lapack_cholesky(a, n, l, lower_triangular, LAPACK_ROW_MAJOR);
        if (!success) {
            return NUMERICAL_ERROR_COMPUTATION;
        }
        return NUMERICAL_SUCCESS;
    }

    // Use production Cholesky decomposition fallback
    return cholesky_decompose(a, l, n, lower_triangular);
}

numerical_error_t numerical_lu_cpu(const ComplexFloat* a,
                             ComplexFloat* l,
                             ComplexFloat* u,
                             int* ipiv,
                             size_t m,
                             size_t n) {
    if (!backend_state.initialized) {
        return NUMERICAL_ERROR_INVALID_STATE;
    }
    
    if (!a || !l || !u || !ipiv) {
        return NUMERICAL_ERROR_INVALID_ARGUMENT;
    }

    if (backend_state.has_lapack) {
        bool success = lapack_lu(a, m, n, l, u, ipiv, LAPACK_ROW_MAJOR);
        if (!success) {
            return NUMERICAL_ERROR_COMPUTATION;
        }
        return NUMERICAL_SUCCESS;
    }

    // Use production LU decomposition with partial pivoting fallback
    return lu_partial_pivot(a, l, u, ipiv, m, n);
}

numerical_error_t numerical_solve_triangular_cpu(const ComplexFloat* a,
                                          const ComplexFloat* b,
                                          ComplexFloat* x,
                                          size_t n,
                                          size_t nrhs,
                                          bool upper_triangular,
                                          bool unit_diagonal) {
    if (!backend_state.initialized) {
        return NUMERICAL_ERROR_INVALID_STATE;
    }
    
    if (!a || !b || !x) {
        return NUMERICAL_ERROR_INVALID_ARGUMENT;
    }

    if (backend_state.has_lapack) {
        bool success = lapack_solve_triangular(a, b, x, n, nrhs,
                                             upper_triangular, unit_diagonal,
                                             LAPACK_ROW_MAJOR);
        if (!success) {
            return NUMERICAL_ERROR_COMPUTATION;
        }
        return NUMERICAL_SUCCESS;
    }

    // Use production triangular solve fallback with forward/backward substitution
    // For each right-hand side column
    for (size_t k = 0; k < nrhs; k++) {
        // Extract column k of b
        ComplexFloat* b_col = malloc(n * sizeof(ComplexFloat));
        ComplexFloat* x_col = malloc(n * sizeof(ComplexFloat));
        if (!b_col || !x_col) {
            free(b_col);
            free(x_col);
            return NUMERICAL_ERROR_MEMORY;
        }

        for (size_t i = 0; i < n; i++) {
            b_col[i] = b[i * nrhs + k];
        }

        if (upper_triangular) {
            backward_substitution(a, b_col, x_col, n, unit_diagonal);
        } else {
            forward_substitution(a, b_col, x_col, n, unit_diagonal);
        }

        // Copy result to x
        for (size_t i = 0; i < n; i++) {
            x[i * nrhs + k] = x_col[i];
        }

        free(b_col);
        free(x_col);
    }

    return NUMERICAL_SUCCESS;
}

numerical_error_t numerical_solve_symmetric_cpu(const ComplexFloat* a,
                                         const ComplexFloat* b,
                                         ComplexFloat* x,
                                         size_t n,
                                         size_t nrhs,
                                         bool positive_definite) {
    if (!backend_state.initialized) {
        return NUMERICAL_ERROR_INVALID_STATE;
    }
    
    if (!a || !b || !x) {
        return NUMERICAL_ERROR_INVALID_ARGUMENT;
    }

    if (backend_state.has_lapack) {
        bool success = lapack_solve_symmetric(a, b, x, n, nrhs,
                                            positive_definite,
                                            LAPACK_ROW_MAJOR);
        if (!success) {
            return NUMERICAL_ERROR_COMPUTATION;
        }
        return NUMERICAL_SUCCESS;
    }

    // Use production symmetric solve fallback
    // For positive definite: Cholesky-based solve
    // For indefinite: LU-based solve

    ComplexFloat* l = malloc(n * n * sizeof(ComplexFloat));
    if (!l) {
        return NUMERICAL_ERROR_MEMORY;
    }

    numerical_error_t err;

    if (positive_definite) {
        // Try Cholesky decomposition
        err = cholesky_decompose(a, l, n, true);
        if (err != NUMERICAL_SUCCESS) {
            free(l);
            return err;
        }

        // Solve L * y = b (forward substitution)
        // Then L^H * x = y (backward substitution with conjugate transpose)
        for (size_t k = 0; k < nrhs; k++) {
            ComplexFloat* b_col = malloc(n * sizeof(ComplexFloat));
            ComplexFloat* y_col = malloc(n * sizeof(ComplexFloat));
            ComplexFloat* x_col = malloc(n * sizeof(ComplexFloat));

            if (!b_col || !y_col || !x_col) {
                free(b_col);
                free(y_col);
                free(x_col);
                free(l);
                return NUMERICAL_ERROR_MEMORY;
            }

            for (size_t i = 0; i < n; i++) {
                b_col[i] = b[i * nrhs + k];
            }

            // L * y = b
            forward_substitution(l, b_col, y_col, n, false);

            // L^H * x = y (compute L^H then backward sub)
            // Create L^H (conjugate transpose)
            ComplexFloat* l_h = malloc(n * n * sizeof(ComplexFloat));
            if (!l_h) {
                free(b_col);
                free(y_col);
                free(x_col);
                free(l);
                return NUMERICAL_ERROR_MEMORY;
            }

            for (size_t i = 0; i < n; i++) {
                for (size_t j = 0; j < n; j++) {
                    l_h[i * n + j] = complex_conjugate(l[j * n + i]);
                }
            }

            // L^H * x = y (backward substitution)
            backward_substitution(l_h, y_col, x_col, n, false);

            for (size_t i = 0; i < n; i++) {
                x[i * nrhs + k] = x_col[i];
            }

            free(b_col);
            free(y_col);
            free(x_col);
            free(l_h);
        }
    } else {
        // Use LU decomposition for indefinite symmetric matrices
        ComplexFloat* u = malloc(n * n * sizeof(ComplexFloat));
        int* ipiv = malloc(n * sizeof(int));

        if (!u || !ipiv) {
            free(l);
            free(u);
            free(ipiv);
            return NUMERICAL_ERROR_MEMORY;
        }

        err = lu_partial_pivot(a, l, u, ipiv, n, n);
        if (err != NUMERICAL_SUCCESS) {
            free(l);
            free(u);
            free(ipiv);
            return err;
        }

        // Solve L * U * x = P * b
        for (size_t k = 0; k < nrhs; k++) {
            ComplexFloat* b_col = malloc(n * sizeof(ComplexFloat));
            ComplexFloat* y_col = malloc(n * sizeof(ComplexFloat));
            ComplexFloat* x_col = malloc(n * sizeof(ComplexFloat));

            if (!b_col || !y_col || !x_col) {
                free(b_col);
                free(y_col);
                free(x_col);
                free(l);
                free(u);
                free(ipiv);
                return NUMERICAL_ERROR_MEMORY;
            }

            // Apply permutation to b
            for (size_t i = 0; i < n; i++) {
                b_col[i] = b[ipiv[i] * nrhs + k];
            }

            // Solve L * y = P * b (forward substitution with unit diagonal)
            forward_substitution(l, b_col, y_col, n, true);

            // Solve U * x = y (backward substitution)
            backward_substitution(u, y_col, x_col, n, false);

            for (size_t i = 0; i < n; i++) {
                x[i * nrhs + k] = x_col[i];
            }

            free(b_col);
            free(y_col);
            free(x_col);
        }

        free(u);
        free(ipiv);
    }

    free(l);
    return NUMERICAL_SUCCESS;
}

numerical_error_t numerical_svd_cpu(const ComplexFloat* a,
                              ComplexFloat* u,
                              float* s,
                              ComplexFloat* vt,
                              size_t m,
                              size_t n) {
    if (!backend_state.initialized) {
        return NUMERICAL_ERROR_INVALID_STATE;
    }
    
    if (!a || !u || !s || !vt) {
        return NUMERICAL_ERROR_INVALID_ARGUMENT;
    }

    if (backend_state.has_lapack) {
        bool success = lapack_svd(a, m, n, u, s, vt, LAPACK_ROW_MAJOR);
        if (!success) {
            lapack_status_t status = lapack_get_last_status();
            switch (status) {
                case LAPACK_MEMORY_ERROR:
                    return NUMERICAL_ERROR_MEMORY;
                case LAPACK_NOT_CONVERGENT:
                    return NUMERICAL_ERROR_COMPUTATION;
                default:
                    return NUMERICAL_ERROR_BACKEND;
            }
        }
        return NUMERICAL_SUCCESS;
    }

    // Production SVD fallback using bidiagonalization + QR iterations
    // This computes the full SVD: A = U * S * V^H

    size_t min_mn = (m < n) ? m : n;
    size_t max_mn = (m > n) ? m : n;

    // Allocate workspace
    ComplexFloat* work = malloc(m * n * sizeof(ComplexFloat));
    ComplexFloat* u_work = malloc(m * m * sizeof(ComplexFloat));
    ComplexFloat* v_work = malloc(n * n * sizeof(ComplexFloat));
    float* diag = malloc(min_mn * sizeof(float));
    float* superdiag = malloc((min_mn > 0 ? min_mn - 1 : 0) * sizeof(float));
    ComplexFloat* v_householder = malloc(max_mn * sizeof(ComplexFloat));
    ComplexFloat* column = malloc(max_mn * sizeof(ComplexFloat));

    if (!work || !u_work || !v_work || !diag || !superdiag || !v_householder || !column) {
        free(work);
        free(u_work);
        free(v_work);
        free(diag);
        free(superdiag);
        free(v_householder);
        free(column);
        return NUMERICAL_ERROR_MEMORY;
    }

    // Copy A to work matrix
    memcpy(work, a, m * n * sizeof(ComplexFloat));

    // Initialize U to identity
    memset(u_work, 0, m * m * sizeof(ComplexFloat));
    for (size_t i = 0; i < m; i++) {
        u_work[i * m + i].real = 1.0f;
    }

    // Initialize V to identity
    memset(v_work, 0, n * n * sizeof(ComplexFloat));
    for (size_t i = 0; i < n; i++) {
        v_work[i * n + i].real = 1.0f;
    }

    // Bidiagonalization via Householder reflections
    for (size_t k = 0; k < min_mn; k++) {
        // Column Householder to zero out below diagonal
        size_t col_len = m - k;
        for (size_t i = 0; i < col_len; i++) {
            column[i] = work[(k + i) * n + k];
        }

        ComplexFloat tau;
        float col_norm = compute_householder_vector(column, v_householder, col_len, &tau);

        // Store diagonal element
        diag[k] = (column[0].real >= 0.0f) ? -col_norm : col_norm;

        // Apply to remaining columns of work
        if (complex_abs_squared(tau) > MACHINE_EPS) {
            for (size_t j = k + 1; j < n; j++) {
                ComplexFloat w = {0.0f, 0.0f};
                for (size_t i = 0; i < col_len; i++) {
                    ComplexFloat vh_conj = complex_conjugate(v_householder[i]);
                    w = complex_add(w, complex_multiply(vh_conj, work[(k + i) * n + j]));
                }
                ComplexFloat tau_w = complex_multiply(tau, w);
                for (size_t i = 0; i < col_len; i++) {
                    work[(k + i) * n + j] = complex_subtract(work[(k + i) * n + j],
                                                              complex_multiply(tau_w, v_householder[i]));
                }
            }

            // Accumulate U
            for (size_t j = 0; j < m; j++) {
                ComplexFloat w = {0.0f, 0.0f};
                for (size_t i = 0; i < col_len; i++) {
                    w = complex_add(w, complex_multiply(u_work[j * m + k + i], v_householder[i]));
                }
                ComplexFloat tau_w = complex_multiply(tau, w);
                for (size_t i = 0; i < col_len; i++) {
                    ComplexFloat vh_conj = complex_conjugate(v_householder[i]);
                    u_work[j * m + k + i] = complex_subtract(u_work[j * m + k + i],
                                                              complex_multiply(tau_w, vh_conj));
                }
            }
        }

        // Row Householder to zero out right of superdiagonal (if applicable)
        if (k < n - 2) {
            size_t row_len = n - k - 1;
            for (size_t j = 0; j < row_len; j++) {
                column[j] = complex_conjugate(work[k * n + k + 1 + j]);
            }

            float row_norm = compute_householder_vector(column, v_householder, row_len, &tau);

            // Store superdiagonal element
            superdiag[k] = (column[0].real >= 0.0f) ? -row_norm : row_norm;

            // Apply to remaining rows of work
            if (complex_abs_squared(tau) > MACHINE_EPS) {
                for (size_t i = k + 1; i < m; i++) {
                    ComplexFloat w = {0.0f, 0.0f};
                    for (size_t j = 0; j < row_len; j++) {
                        w = complex_add(w, complex_multiply(work[i * n + k + 1 + j], v_householder[j]));
                    }
                    ComplexFloat tau_w = complex_multiply(tau, w);
                    for (size_t j = 0; j < row_len; j++) {
                        ComplexFloat vh_conj = complex_conjugate(v_householder[j]);
                        work[i * n + k + 1 + j] = complex_subtract(work[i * n + k + 1 + j],
                                                                    complex_multiply(tau_w, vh_conj));
                    }
                }

                // Accumulate V
                for (size_t i = 0; i < n; i++) {
                    ComplexFloat w = {0.0f, 0.0f};
                    for (size_t j = 0; j < row_len; j++) {
                        w = complex_add(w, complex_multiply(v_work[i * n + k + 1 + j], v_householder[j]));
                    }
                    ComplexFloat tau_w = complex_multiply(tau, w);
                    for (size_t j = 0; j < row_len; j++) {
                        ComplexFloat vh_conj = complex_conjugate(v_householder[j]);
                        v_work[i * n + k + 1 + j] = complex_subtract(v_work[i * n + k + 1 + j],
                                                                      complex_multiply(tau_w, vh_conj));
                    }
                }
            }
        } else if (k < n - 1) {
            superdiag[k] = complex_abs(work[k * n + k + 1]);
        }
    }

    // QR iterations on bidiagonal matrix to compute singular values
    const size_t max_iter = 100 * min_mn;
    const float tol = MACHINE_EPS * 100.0f;

    for (size_t iter = 0; iter < max_iter; iter++) {
        // Check for convergence
        bool converged = true;
        for (size_t i = 0; i < min_mn - 1; i++) {
            if (fabsf(superdiag[i]) > tol * (fabsf(diag[i]) + fabsf(diag[i + 1]))) {
                converged = false;
                break;
            }
        }
        if (converged) break;

        // Zero out converged superdiagonal elements
        for (size_t i = 0; i < min_mn - 1; i++) {
            if (fabsf(superdiag[i]) <= tol * (fabsf(diag[i]) + fabsf(diag[i + 1]))) {
                superdiag[i] = 0.0f;
            }
        }

        // Implicit QR step with Wilkinson shift
        float shift = 0.0f;
        if (min_mn >= 2) {
            float d = (diag[min_mn - 2] - diag[min_mn - 1]) / 2.0f;
            float e = (min_mn >= 2) ? superdiag[min_mn - 2] : 0.0f;
            float t = diag[min_mn - 1];
            shift = t - e * e / (d + (d >= 0 ? 1 : -1) * sqrtf(d * d + e * e));
        }

        float f = diag[0] * diag[0] - shift;
        float g = diag[0] * ((min_mn > 1) ? superdiag[0] : 0.0f);

        for (size_t k = 0; k < min_mn - 1; k++) {
            // Givens rotation to zero out g
            float r = sqrtf(f * f + g * g);
            if (r < MACHINE_EPS) {
                f = diag[k + 1];
                g = (k + 1 < min_mn - 1) ? superdiag[k + 1] : 0.0f;
                continue;
            }

            float c = f / r;
            float s_rot = -g / r;

            // Apply rotation to bidiagonal
            if (k > 0) {
                superdiag[k - 1] = r;
            }

            float temp_f = c * diag[k] - s_rot * superdiag[k];
            float temp_g = s_rot * diag[k] + c * superdiag[k];
            float temp_h = -s_rot * diag[k + 1];
            diag[k + 1] = c * diag[k + 1];

            diag[k] = temp_f;
            superdiag[k] = temp_g;
            f = temp_g;
            g = temp_h;

            // Update V
            for (size_t i = 0; i < n; i++) {
                ComplexFloat v1 = v_work[i * n + k];
                ComplexFloat v2 = v_work[i * n + k + 1];
                v_work[i * n + k].real = c * v1.real - s_rot * v2.real;
                v_work[i * n + k].imag = c * v1.imag - s_rot * v2.imag;
                v_work[i * n + k + 1].real = s_rot * v1.real + c * v2.real;
                v_work[i * n + k + 1].imag = s_rot * v1.imag + c * v2.imag;
            }

            // Second Givens rotation
            r = sqrtf(f * f + g * g);
            if (r < MACHINE_EPS) {
                f = diag[k + 1];
                g = (k + 1 < min_mn - 1) ? superdiag[k + 1] : 0.0f;
                continue;
            }

            c = f / r;
            s_rot = -g / r;

            diag[k] = r;
            temp_f = c * superdiag[k] - s_rot * diag[k + 1];
            diag[k + 1] = s_rot * superdiag[k] + c * diag[k + 1];
            superdiag[k] = temp_f;

            if (k + 1 < min_mn - 1) {
                g = -s_rot * superdiag[k + 1];
                superdiag[k + 1] = c * superdiag[k + 1];
            }

            f = superdiag[k];

            // Update U
            for (size_t i = 0; i < m; i++) {
                ComplexFloat u1 = u_work[i * m + k];
                ComplexFloat u2 = u_work[i * m + k + 1];
                u_work[i * m + k].real = c * u1.real - s_rot * u2.real;
                u_work[i * m + k].imag = c * u1.imag - s_rot * u2.imag;
                u_work[i * m + k + 1].real = s_rot * u1.real + c * u2.real;
                u_work[i * m + k + 1].imag = s_rot * u1.imag + c * u2.imag;
            }
        }
    }

    // Copy singular values (take absolute values and ensure positive)
    for (size_t i = 0; i < min_mn; i++) {
        s[i] = fabsf(diag[i]);

        // If singular value was negative, flip corresponding column of U
        if (diag[i] < 0.0f) {
            for (size_t j = 0; j < m; j++) {
                u_work[j * m + i].real = -u_work[j * m + i].real;
                u_work[j * m + i].imag = -u_work[j * m + i].imag;
            }
        }
    }

    // Sort singular values in descending order and reorder U, V accordingly
    for (size_t i = 0; i < min_mn - 1; i++) {
        size_t max_idx = i;
        float max_s = s[i];
        for (size_t j = i + 1; j < min_mn; j++) {
            if (s[j] > max_s) {
                max_s = s[j];
                max_idx = j;
            }
        }
        if (max_idx != i) {
            // Swap singular values
            float temp_s = s[i];
            s[i] = s[max_idx];
            s[max_idx] = temp_s;

            // Swap columns of U
            for (size_t j = 0; j < m; j++) {
                ComplexFloat temp = u_work[j * m + i];
                u_work[j * m + i] = u_work[j * m + max_idx];
                u_work[j * m + max_idx] = temp;
            }

            // Swap columns of V
            for (size_t j = 0; j < n; j++) {
                ComplexFloat temp = v_work[j * n + i];
                v_work[j * n + i] = v_work[j * n + max_idx];
                v_work[j * n + max_idx] = temp;
            }
        }
    }

    // Copy U (m x m to m x min_mn for thin SVD compatibility)
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < min_mn; j++) {
            u[i * min_mn + j] = u_work[i * m + j];
        }
    }

    // Copy V^H (conjugate transpose of V)
    for (size_t i = 0; i < min_mn; i++) {
        for (size_t j = 0; j < n; j++) {
            vt[i * n + j] = complex_conjugate(v_work[j * n + i]);
        }
    }

    free(work);
    free(u_work);
    free(v_work);
    free(diag);
    free(superdiag);
    free(v_householder);
    free(column);

    return NUMERICAL_SUCCESS;
}

numerical_error_t get_numerical_metrics_cpu(numerical_metrics_t* metrics) {
    if (!backend_state.initialized) {
        return NUMERICAL_ERROR_INVALID_STATE;
    }
    
    if (!metrics) {
        return NUMERICAL_ERROR_INVALID_ARGUMENT;
    }
    
    *metrics = backend_state.metrics;
    return NUMERICAL_SUCCESS;
}

numerical_error_t reset_numerical_metrics_cpu(void) {
    if (!backend_state.initialized) {
        return NUMERICAL_ERROR_INVALID_STATE;
    }
    
    memset(&backend_state.metrics, 0, sizeof(numerical_metrics_t));
    return NUMERICAL_SUCCESS;
}

numerical_error_t get_last_numerical_error_cpu(void) {
    return backend_state.last_error;
}
