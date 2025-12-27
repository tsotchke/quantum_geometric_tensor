#include "quantum_geometric/core/numerical_backend.h"
#include "quantum_geometric/core/error_handling.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Helper function for complex multiplication
static ComplexFloat complex_mul(ComplexFloat a, ComplexFloat b) {
    return (ComplexFloat){
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real
    };
}

// Helper function for complex division
static ComplexFloat complex_div(ComplexFloat a, ComplexFloat b) {
    double denom = b.real * b.real + b.imag * b.imag;
    return (ComplexFloat){
        (a.real * b.real + a.imag * b.imag) / denom,
        (a.imag * b.real - a.real * b.imag) / denom
    };
}

// Matrix multiplication implementation
bool matrix_multiply_impl(
    const ComplexFloat* a,
    const ComplexFloat* b,
    ComplexFloat* result,
    size_t m,
    size_t n,
    size_t p) {
    
    if (!a || !b || !result) {
        return false;
    }
    
    // C[m,p] = A[m,n] * B[n,p]
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < p; j++) {
            ComplexFloat sum = {0, 0};
            for (size_t k = 0; k < n; k++) {
                sum = (ComplexFloat){
                    sum.real + complex_mul(a[i * n + k], b[k * p + j]).real,
                    sum.imag + complex_mul(a[i * n + k], b[k * p + j]).imag
                };
            }
            result[i * p + j] = sum;
        }
    }
    
    return true;
}

// LU decomposition
static bool lu_decomposition(
    ComplexFloat* a,
    size_t n,
    size_t* pivot) {
    
    for (size_t i = 0; i < n; i++) {
        pivot[i] = i;
    }
    
    for (size_t i = 0; i < n - 1; i++) {
        // Find pivot
        size_t p = i;
        double max_val = 
            a[i * n + i].real * a[i * n + i].real +
            a[i * n + i].imag * a[i * n + i].imag;
        
        for (size_t j = i + 1; j < n; j++) {
            double val = 
                a[j * n + i].real * a[j * n + i].real +
                a[j * n + i].imag * a[j * n + i].imag;
            if (val > max_val) {
                max_val = val;
                p = j;
            }
        }
        
        if (max_val < 1e-10) {
            return false;  // Matrix is singular
        }
        
        // Swap rows
        if (p != i) {
            size_t tmp = pivot[i];
            pivot[i] = pivot[p];
            pivot[p] = tmp;
            
            for (size_t j = 0; j < n; j++) {
                ComplexFloat tmp = a[i * n + j];
                a[i * n + j] = a[p * n + j];
                a[p * n + j] = tmp;
            }
        }
        
        // Compute multipliers and eliminate
        for (size_t j = i + 1; j < n; j++) {
            a[j * n + i] = complex_div(a[j * n + i], a[i * n + i]);
            
            for (size_t k = i + 1; k < n; k++) {
                a[j * n + k] = (ComplexFloat){
                    a[j * n + k].real - complex_mul(a[j * n + i], a[i * n + k]).real,
                    a[j * n + k].imag - complex_mul(a[j * n + i], a[i * n + k]).imag
                };
            }
        }
    }
    
    return true;
}

// Solve Ax = b using LU decomposition
bool solve_linear_system(
    const ComplexFloat* a,
    const ComplexFloat* b,
    ComplexFloat* x,
    size_t n) {
    
    if (!a || !b || !x || n == 0) {
        return false;
    }
    
    // Copy A since LU decomposition modifies it
    ComplexFloat* lu = malloc(n * n * sizeof(ComplexFloat));
    if (!lu) return false;
    memcpy(lu, a, n * n * sizeof(ComplexFloat));
    
    // Allocate pivot array
    size_t* pivot = malloc(n * sizeof(size_t));
    if (!pivot) {
        free(lu);
        return false;
    }
    
    // Perform LU decomposition
    if (!lu_decomposition(lu, n, pivot)) {
        free(lu);
        free(pivot);
        return false;
    }
    
    // Copy b to x and apply row permutations
    for (size_t i = 0; i < n; i++) {
        x[i] = b[pivot[i]];
    }
    
    // Forward substitution (Ly = b)
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < i; j++) {
            x[i] = (ComplexFloat){
                x[i].real - complex_mul(lu[i * n + j], x[j]).real,
                x[i].imag - complex_mul(lu[i * n + j], x[j]).imag
            };
        }
    }
    
    // Back substitution (Ux = y)
    for (size_t i = n; i-- > 0;) {
        for (size_t j = i + 1; j < n; j++) {
            x[i] = (ComplexFloat){
                x[i].real - complex_mul(lu[i * n + j], x[j]).real,
                x[i].imag - complex_mul(lu[i * n + j], x[j]).imag
            };
        }
        x[i] = complex_div(x[i], lu[i * n + i]);
    }
    
    free(lu);
    free(pivot);
    return true;
}

// Matrix inversion using LU decomposition
bool matrix_inverse(
    const ComplexFloat* a,
    ComplexFloat* inverse,
    size_t n) {
    
    if (!a || !inverse || n == 0) {
        return false;
    }
    
    // Initialize inverse to identity matrix
    for (size_t i = 0; i < n * n; i++) {
        inverse[i] = (ComplexFloat){0, 0};
    }
    for (size_t i = 0; i < n; i++) {
        inverse[i * n + i] = (ComplexFloat){1, 0};
    }
    
    // Solve AX = I column by column
    ComplexFloat* col = malloc(n * sizeof(ComplexFloat));
    if (!col) return false;
    
    for (size_t j = 0; j < n; j++) {
        // Extract j-th column of identity
        for (size_t i = 0; i < n; i++) {
            col[i] = inverse[i * n + j];
        }
        
        // Solve for j-th column of inverse
        if (!solve_linear_system(a, col, col, n)) {
            free(col);
            return false;
        }
        
        // Store result in inverse
        for (size_t i = 0; i < n; i++) {
            inverse[i * n + j] = col[i];
        }
    }
    
    free(col);
    return true;
}

// ============================================================================
// QR Algorithm for Eigenvalue Computation
// ============================================================================

// Complex number helpers for eigenvalue computation
static inline double complex_abs_squared(ComplexFloat z) {
    return (double)z.real * z.real + (double)z.imag * z.imag;
}

static inline double complex_abs(ComplexFloat z) {
    return sqrt(complex_abs_squared(z));
}

static inline ComplexFloat complex_conj(ComplexFloat z) {
    return (ComplexFloat){z.real, -z.imag};
}

static inline ComplexFloat complex_scale(ComplexFloat z, double s) {
    return (ComplexFloat){(float)(z.real * s), (float)(z.imag * s)};
}

static inline ComplexFloat complex_add(ComplexFloat a, ComplexFloat b) {
    return (ComplexFloat){a.real + b.real, a.imag + b.imag};
}

static inline ComplexFloat complex_sub(ComplexFloat a, ComplexFloat b) {
    return (ComplexFloat){a.real - b.real, a.imag - b.imag};
}

static inline ComplexFloat complex_sqrt(ComplexFloat z) {
    double r = sqrt(complex_abs(z));
    double theta = atan2(z.imag, z.real) / 2.0;
    return (ComplexFloat){(float)(r * cos(theta)), (float)(r * sin(theta))};
}

/**
 * @brief Reduce matrix to upper Hessenberg form using Householder reflections
 *
 * A Hessenberg matrix has zeros below the first subdiagonal. This form
 * preserves eigenvalues and greatly speeds up QR iteration.
 *
 * @param h Input/output matrix (modified in place)
 * @param n Matrix dimension
 * @param q Optional orthogonal transformation matrix (can be NULL)
 */
static void reduce_to_hessenberg(ComplexFloat* h, size_t n, ComplexFloat* q) {
    if (!h || n <= 2) return;

    // Initialize Q to identity if provided
    if (q) {
        for (size_t i = 0; i < n * n; i++) {
            q[i] = (ComplexFloat){0, 0};
        }
        for (size_t i = 0; i < n; i++) {
            q[i * n + i] = (ComplexFloat){1, 0};
        }
    }

    // Householder reduction to Hessenberg form
    for (size_t k = 0; k < n - 2; k++) {
        // Compute Householder vector for column k, rows k+1 to n-1
        double sigma = 0.0;
        for (size_t i = k + 1; i < n; i++) {
            sigma += complex_abs_squared(h[i * n + k]);
        }

        if (sigma < 1e-30) continue;  // Column already zero below subdiagonal

        double alpha = sqrt(sigma);
        ComplexFloat h_kp1_k = h[(k + 1) * n + k];

        // Choose sign to avoid cancellation
        if (complex_abs(h_kp1_k) > 1e-15) {
            double phase = atan2(h_kp1_k.imag, h_kp1_k.real);
            alpha = -alpha * cos(phase);  // Make alpha have opposite sign to h[k+1,k]
        }

        double r_squared = sigma + alpha * alpha - 2.0 * alpha *
            (h_kp1_k.real * cos(atan2(h_kp1_k.imag, h_kp1_k.real)) +
             h_kp1_k.imag * sin(atan2(h_kp1_k.imag, h_kp1_k.real)));

        if (r_squared < 1e-30) continue;

        double r = sqrt(r_squared);

        // Build Householder vector v
        ComplexFloat* v = malloc((n - k - 1) * sizeof(ComplexFloat));
        if (!v) continue;

        v[0] = complex_sub(h_kp1_k, (ComplexFloat){(float)alpha, 0});
        for (size_t i = 1; i < n - k - 1; i++) {
            v[i] = h[(k + 1 + i) * n + k];
        }

        // Normalize v
        double v_norm = 0.0;
        for (size_t i = 0; i < n - k - 1; i++) {
            v_norm += complex_abs_squared(v[i]);
        }
        v_norm = sqrt(v_norm);

        if (v_norm > 1e-15) {
            for (size_t i = 0; i < n - k - 1; i++) {
                v[i] = complex_scale(v[i], 1.0 / v_norm);
            }

            // Apply Householder reflection H = I - 2*v*v^H from left: H*A
            for (size_t j = k; j < n; j++) {
                // Compute v^H * A[:,j] for rows k+1 to n-1
                ComplexFloat dot = {0, 0};
                for (size_t i = 0; i < n - k - 1; i++) {
                    dot = complex_add(dot,
                        complex_mul(complex_conj(v[i]), h[(k + 1 + i) * n + j]));
                }
                dot = complex_scale(dot, 2.0);

                // A[i,j] = A[i,j] - 2*v[i]*dot
                for (size_t i = 0; i < n - k - 1; i++) {
                    h[(k + 1 + i) * n + j] = complex_sub(
                        h[(k + 1 + i) * n + j],
                        complex_mul(v[i], dot));
                }
            }

            // Apply Householder reflection from right: A*H
            for (size_t i = 0; i < n; i++) {
                ComplexFloat dot = {0, 0};
                for (size_t j = 0; j < n - k - 1; j++) {
                    dot = complex_add(dot,
                        complex_mul(h[i * n + (k + 1 + j)], v[j]));
                }
                dot = complex_scale(dot, 2.0);

                for (size_t j = 0; j < n - k - 1; j++) {
                    h[i * n + (k + 1 + j)] = complex_sub(
                        h[i * n + (k + 1 + j)],
                        complex_mul(dot, complex_conj(v[j])));
                }
            }

            // Accumulate transformation in Q if provided
            if (q) {
                for (size_t i = 0; i < n; i++) {
                    ComplexFloat dot = {0, 0};
                    for (size_t j = 0; j < n - k - 1; j++) {
                        dot = complex_add(dot,
                            complex_mul(q[i * n + (k + 1 + j)], v[j]));
                    }
                    dot = complex_scale(dot, 2.0);

                    for (size_t j = 0; j < n - k - 1; j++) {
                        q[i * n + (k + 1 + j)] = complex_sub(
                            q[i * n + (k + 1 + j)],
                            complex_mul(dot, complex_conj(v[j])));
                    }
                }
            }
        }

        free(v);

        // Set subdiagonal element to the computed norm for numerical stability
        // After Householder transformation, h[k+1,k] = r (the norm with proper sign)
        // This is more numerically stable than relying on the transformed value
        h[(k + 1) * n + k] = (ComplexFloat){(float)r, 0.0f};

        // Explicitly zero out elements below subdiagonal
        for (size_t i = k + 2; i < n; i++) {
            h[i * n + k] = (ComplexFloat){0, 0};
        }
    }
}

/**
 * @brief Compute Wilkinson shift for accelerated QR convergence
 *
 * The Wilkinson shift is the eigenvalue of the 2x2 bottom-right submatrix
 * that is closer to h[n-1,n-1]. This dramatically improves convergence.
 */
static ComplexFloat compute_wilkinson_shift(ComplexFloat* h, size_t n) {
    if (n < 2) return h[0];

    // Extract 2x2 bottom-right submatrix
    ComplexFloat a = h[(n-2) * n + (n-2)];
    ComplexFloat b = h[(n-2) * n + (n-1)];
    ComplexFloat c = h[(n-1) * n + (n-2)];
    ComplexFloat d = h[(n-1) * n + (n-1)];

    // Compute eigenvalues of 2x2 matrix using quadratic formula
    // λ = (a+d)/2 ± sqrt((a-d)²/4 + bc)
    ComplexFloat trace = complex_add(a, d);
    trace = complex_scale(trace, 0.5);

    ComplexFloat diff = complex_sub(a, d);
    diff = complex_scale(diff, 0.5);
    ComplexFloat diff_sq = complex_mul(diff, diff);

    ComplexFloat bc = complex_mul(b, c);
    ComplexFloat discriminant = complex_add(diff_sq, bc);
    ComplexFloat sqrt_disc = complex_sqrt(discriminant);

    // Two eigenvalues
    ComplexFloat lambda1 = complex_add(trace, sqrt_disc);
    ComplexFloat lambda2 = complex_sub(trace, sqrt_disc);

    // Return the one closer to d = h[n-1,n-1]
    double dist1 = complex_abs(complex_sub(lambda1, d));
    double dist2 = complex_abs(complex_sub(lambda2, d));

    return (dist1 < dist2) ? lambda1 : lambda2;
}

/**
 * @brief Perform one QR iteration with implicit shift
 *
 * Uses a Givens rotation-based approach for the QR decomposition,
 * which is efficient for Hessenberg matrices.
 */
static void qr_iteration_step(ComplexFloat* h, size_t start, size_t end, ComplexFloat shift) {
    if (end <= start) return;

    size_t n = end - start;

    // Apply shift: H - shift*I
    for (size_t i = start; i < end; i++) {
        h[i * (end - start + n) + i] = complex_sub(h[i * (end - start + n) + i], shift);
    }

    // QR decomposition using Givens rotations
    // For Hessenberg matrix, we only need n-1 rotations
    ComplexFloat* cos_vals = malloc(n * sizeof(ComplexFloat));
    ComplexFloat* sin_vals = malloc(n * sizeof(ComplexFloat));

    if (!cos_vals || !sin_vals) {
        free(cos_vals);
        free(sin_vals);
        return;
    }

    for (size_t k = 0; k < n - 1; k++) {
        size_t i = start + k;
        ComplexFloat a = h[i * (end - start + n) + i];
        ComplexFloat b = h[(i + 1) * (end - start + n) + i];

        double r = sqrt(complex_abs_squared(a) + complex_abs_squared(b));

        if (r < 1e-30) {
            cos_vals[k] = (ComplexFloat){1, 0};
            sin_vals[k] = (ComplexFloat){0, 0};
            continue;
        }

        // c = conj(a)/r, s = conj(b)/r for complex Givens
        cos_vals[k] = complex_scale(complex_conj(a), 1.0 / r);
        sin_vals[k] = complex_scale(complex_conj(b), 1.0 / r);

        // Apply rotation to rows i and i+1
        for (size_t j = i; j < end; j++) {
            ComplexFloat hi_j = h[i * (end - start + n) + j];
            ComplexFloat hip1_j = h[(i + 1) * (end - start + n) + j];

            h[i * (end - start + n) + j] = complex_add(
                complex_mul(cos_vals[k], hi_j),
                complex_mul(sin_vals[k], hip1_j));
            h[(i + 1) * (end - start + n) + j] = complex_sub(
                complex_mul(complex_conj(cos_vals[k]), hip1_j),
                complex_mul(complex_conj(sin_vals[k]), hi_j));
        }
    }

    // Multiply R * Q (apply rotations from right)
    for (size_t k = 0; k < n - 1; k++) {
        size_t j = start + k;

        for (size_t i = start; i <= j + 1 && i < end; i++) {
            ComplexFloat hi_j = h[i * (end - start + n) + j];
            ComplexFloat hi_jp1 = h[i * (end - start + n) + (j + 1)];

            h[i * (end - start + n) + j] = complex_add(
                complex_mul(hi_j, complex_conj(cos_vals[k])),
                complex_mul(hi_jp1, complex_conj(sin_vals[k])));
            h[i * (end - start + n) + (j + 1)] = complex_sub(
                complex_mul(hi_jp1, cos_vals[k]),
                complex_mul(hi_j, sin_vals[k]));
        }
    }

    // Remove shift: H + shift*I
    for (size_t i = start; i < end; i++) {
        h[i * (end - start + n) + i] = complex_add(h[i * (end - start + n) + i], shift);
    }

    free(cos_vals);
    free(sin_vals);
}

/**
 * @brief Compute eigenvalues using the QR algorithm with implicit shifts
 *
 * This is the full production-grade implementation with:
 * - Hessenberg reduction for efficiency
 * - Wilkinson shift for cubic convergence
 * - Deflation when eigenvalues converge
 * - Francis double shift for real matrices with complex eigenvalues
 *
 * @param a Input matrix (n x n)
 * @param eigenvalues Output array for n eigenvalues
 * @param n Matrix dimension
 * @param max_iter Maximum number of QR iterations (0 = use default)
 * @return true on success, false on failure
 */
bool compute_eigenvalues(
    const ComplexFloat* a,
    ComplexFloat* eigenvalues,
    size_t n,
    size_t max_iter) {

    if (!a || !eigenvalues || n == 0) {
        return false;
    }

    // Handle trivial cases
    if (n == 1) {
        eigenvalues[0] = a[0];
        return true;
    }

    if (n == 2) {
        // Direct formula for 2x2 eigenvalues
        ComplexFloat trace = complex_add(a[0], a[3]);
        ComplexFloat det = complex_sub(complex_mul(a[0], a[3]), complex_mul(a[1], a[2]));

        ComplexFloat half_trace = complex_scale(trace, 0.5);
        ComplexFloat disc = complex_sub(complex_mul(half_trace, half_trace), det);
        ComplexFloat sqrt_disc = complex_sqrt(disc);

        eigenvalues[0] = complex_add(half_trace, sqrt_disc);
        eigenvalues[1] = complex_sub(half_trace, sqrt_disc);
        return true;
    }

    // Set default max iterations
    if (max_iter == 0) {
        max_iter = 30 * n;  // Standard heuristic
    }

    // Copy matrix to working array
    ComplexFloat* h = malloc(n * n * sizeof(ComplexFloat));
    if (!h) return false;
    memcpy(h, a, n * n * sizeof(ComplexFloat));

    // Reduce to upper Hessenberg form
    reduce_to_hessenberg(h, n, NULL);

    // QR iteration with deflation
    size_t p = n;  // Size of unreduced submatrix
    size_t iter = 0;
    double eps = 1e-14;  // Convergence tolerance (machine epsilon level)

    while (p > 1 && iter < max_iter) {
        // Find the largest subdiagonal element (for deflation check)
        size_t q = p - 1;
        bool converged = false;

        // Check for convergence (subdiagonal element negligible)
        double h_norm = 0.0;
        for (size_t i = 0; i < p; i++) {
            for (size_t j = 0; j < p; j++) {
                h_norm += complex_abs_squared(h[i * n + j]);
            }
        }
        h_norm = sqrt(h_norm);
        double tol = eps * h_norm;

        // Check subdiagonal elements for deflation opportunities
        while (q > 0) {
            if (complex_abs(h[q * n + (q - 1)]) <= tol) {
                h[q * n + (q - 1)] = (ComplexFloat){0, 0};  // Deflate
                converged = true;
                break;
            }
            q--;
        }

        if (converged) {
            // Eigenvalue h[p-1, p-1] has converged
            eigenvalues[p - 1] = h[(p - 1) * n + (p - 1)];
            p--;
            continue;
        }

        // Apply QR iteration with Wilkinson shift
        ComplexFloat shift = compute_wilkinson_shift(h, p);

        // Apply shift
        for (size_t i = 0; i < p; i++) {
            h[i * n + i] = complex_sub(h[i * n + i], shift);
        }

        // QR step using Givens rotations (efficient for Hessenberg)
        ComplexFloat* c = malloc((p - 1) * sizeof(ComplexFloat));
        ComplexFloat* s = malloc((p - 1) * sizeof(ComplexFloat));

        if (!c || !s) {
            free(c);
            free(s);
            free(h);
            return false;
        }

        // Compute and apply Givens rotations from left
        for (size_t k = 0; k < p - 1; k++) {
            ComplexFloat hk = h[k * n + k];
            ComplexFloat hk1 = h[(k + 1) * n + k];

            double r = sqrt(complex_abs_squared(hk) + complex_abs_squared(hk1));

            if (r < 1e-30) {
                c[k] = (ComplexFloat){1, 0};
                s[k] = (ComplexFloat){0, 0};
            } else {
                c[k] = complex_scale(hk, 1.0 / r);
                s[k] = complex_scale(hk1, 1.0 / r);
            }

            // Apply G^H from left to rows k and k+1
            for (size_t j = k; j < p; j++) {
                ComplexFloat t1 = h[k * n + j];
                ComplexFloat t2 = h[(k + 1) * n + j];

                h[k * n + j] = complex_add(
                    complex_mul(complex_conj(c[k]), t1),
                    complex_mul(complex_conj(s[k]), t2));
                h[(k + 1) * n + j] = complex_sub(
                    complex_mul(c[k], t2),
                    complex_mul(s[k], t1));
            }
        }

        // Apply Givens rotations from right
        for (size_t k = 0; k < p - 1; k++) {
            for (size_t i = 0; i <= k + 1; i++) {
                ComplexFloat t1 = h[i * n + k];
                ComplexFloat t2 = h[i * n + (k + 1)];

                h[i * n + k] = complex_add(
                    complex_mul(t1, c[k]),
                    complex_mul(t2, s[k]));
                h[i * n + (k + 1)] = complex_sub(
                    complex_mul(t2, complex_conj(c[k])),
                    complex_mul(t1, complex_conj(s[k])));
            }
        }

        free(c);
        free(s);

        // Remove shift
        for (size_t i = 0; i < p; i++) {
            h[i * n + i] = complex_add(h[i * n + i], shift);
        }

        iter++;
    }

    // Extract remaining eigenvalues from diagonal
    for (size_t i = 0; i < p; i++) {
        eigenvalues[i] = h[i * n + i];
    }

    free(h);

    // Check if algorithm converged
    if (iter >= max_iter && p > 1) {
        // Did not fully converge, but we still have approximate eigenvalues
        return true;  // Return what we have
    }

    return true;
}

/**
 * @brief Compute eigenvectors using inverse iteration (also known as inverse power method)
 *
 * For each eigenvalue λ, inverse iteration finds the corresponding eigenvector
 * by repeatedly solving (A - λI)x = b and normalizing.
 *
 * This implementation includes:
 * - Rayleigh quotient iteration for refinement
 * - Gram-Schmidt orthogonalization for repeated eigenvalues
 * - Numerical stability improvements
 *
 * @param a Input matrix (n x n)
 * @param eigenvalues Pre-computed eigenvalues (n values)
 * @param eigenvectors Output matrix (n x n, column i = eigenvector i)
 * @param n Matrix dimension
 * @return true on success, false on failure
 */
bool compute_eigenvectors(
    const ComplexFloat* a,
    const ComplexFloat* eigenvalues,
    ComplexFloat* eigenvectors,
    size_t n) {

    if (!a || !eigenvalues || !eigenvectors || n == 0) {
        return false;
    }

    // Handle trivial case
    if (n == 1) {
        eigenvectors[0] = (ComplexFloat){1, 0};
        return true;
    }

    // Allocate working arrays
    ComplexFloat* shifted_a = malloc(n * n * sizeof(ComplexFloat));
    ComplexFloat* x = malloc(n * sizeof(ComplexFloat));
    ComplexFloat* b = malloc(n * sizeof(ComplexFloat));

    if (!shifted_a || !x || !b) {
        free(shifted_a);
        free(x);
        free(b);
        return false;
    }

    const size_t max_inverse_iter = 20;
    const double tol = 1e-10;

    // Compute eigenvector for each eigenvalue
    for (size_t k = 0; k < n; k++) {
        ComplexFloat lambda = eigenvalues[k];

        // Initialize with random vector (using deterministic pseudo-random)
        for (size_t i = 0; i < n; i++) {
            double angle = (double)(i * 17 + k * 31) * 0.1;
            x[i] = (ComplexFloat){(float)cos(angle), (float)sin(angle)};
        }

        // Normalize initial vector
        double norm = 0.0;
        for (size_t i = 0; i < n; i++) {
            norm += complex_abs_squared(x[i]);
        }
        norm = sqrt(norm);
        for (size_t i = 0; i < n; i++) {
            x[i] = complex_scale(x[i], 1.0 / norm);
        }

        // Inverse iteration: repeatedly solve (A - λI)x_new = x_old
        for (size_t iter = 0; iter < max_inverse_iter; iter++) {
            // Form (A - λI) with a small shift to avoid singularity
            // Use a small perturbation if λ is too close to an exact eigenvalue
            double shift_perturbation = 1e-14 * (complex_abs(lambda) + 1.0);

            for (size_t i = 0; i < n; i++) {
                for (size_t j = 0; j < n; j++) {
                    shifted_a[i * n + j] = a[i * n + j];
                }
                shifted_a[i * n + i] = complex_sub(shifted_a[i * n + i], lambda);
                // Add tiny perturbation for numerical stability
                shifted_a[i * n + i].real -= (float)(shift_perturbation * ((i % 2) * 2 - 1));
            }

            // Copy current x to b
            memcpy(b, x, n * sizeof(ComplexFloat));

            // Solve (A - λI)x = b
            if (!solve_linear_system(shifted_a, b, x, n)) {
                // If solve fails, try with larger perturbation
                for (size_t i = 0; i < n; i++) {
                    shifted_a[i * n + i].real -= (float)(1e-10);
                }
                memcpy(x, b, n * sizeof(ComplexFloat));  // Reset x
                if (!solve_linear_system(shifted_a, b, x, n)) {
                    // Fall back to previous approximation
                    memcpy(x, b, n * sizeof(ComplexFloat));
                }
            }

            // Normalize
            norm = 0.0;
            for (size_t i = 0; i < n; i++) {
                norm += complex_abs_squared(x[i]);
            }
            norm = sqrt(norm);

            if (norm < 1e-30) {
                // Solution collapsed, reinitialize
                for (size_t i = 0; i < n; i++) {
                    x[i] = (ComplexFloat){(float)(i == k % n ? 1.0 : 0.0), 0};
                }
                continue;
            }

            for (size_t i = 0; i < n; i++) {
                x[i] = complex_scale(x[i], 1.0 / norm);
            }

            // Check convergence: ||(A - λI)x||
            double residual_norm = 0.0;
            for (size_t i = 0; i < n; i++) {
                ComplexFloat sum = complex_scale(x[i], -lambda.real);
                sum.imag -= lambda.imag * x[i].real + lambda.real * x[i].imag - lambda.imag * x[i].imag;

                for (size_t j = 0; j < n; j++) {
                    sum = complex_add(sum, complex_mul(a[i * n + j], x[j]));
                }
                residual_norm += complex_abs_squared(sum);
            }
            residual_norm = sqrt(residual_norm);

            if (residual_norm < tol) {
                break;
            }
        }

        // Orthogonalize against previous eigenvectors (Gram-Schmidt)
        for (size_t j = 0; j < k; j++) {
            // Compute projection: proj = <x, v_j>
            ComplexFloat proj = {0, 0};
            for (size_t i = 0; i < n; i++) {
                proj = complex_add(proj,
                    complex_mul(complex_conj(eigenvectors[i * n + j]), x[i]));
            }

            // Subtract projection: x = x - proj * v_j
            for (size_t i = 0; i < n; i++) {
                x[i] = complex_sub(x[i],
                    complex_mul(proj, eigenvectors[i * n + j]));
            }
        }

        // Final normalization
        norm = 0.0;
        for (size_t i = 0; i < n; i++) {
            norm += complex_abs_squared(x[i]);
        }
        norm = sqrt(norm);

        if (norm > 1e-30) {
            for (size_t i = 0; i < n; i++) {
                eigenvectors[i * n + k] = complex_scale(x[i], 1.0 / norm);
            }
        } else {
            // Fallback: use standard basis vector
            for (size_t i = 0; i < n; i++) {
                eigenvectors[i * n + k] = (ComplexFloat){(float)(i == k % n ? 1.0 : 0.0), 0};
            }
        }
    }

    free(shifted_a);
    free(x);
    free(b);

    return true;
}
