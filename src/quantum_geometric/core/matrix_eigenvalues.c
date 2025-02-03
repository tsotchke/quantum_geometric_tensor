#include "quantum_geometric/core/matrix_eigenvalues.h"
#include "quantum_geometric/core/matrix_qr.h"
#include "quantum_geometric/core/matrix_operations.h"
#include "quantum_geometric/core/quantum_complex.h"
#include "quantum_geometric/core/error_handling.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

// Global QR configuration
static qr_config_t g_qr_config = {
    .convergence_threshold = 1e-10,
    .max_iterations = 100,
    .compute_eigenvectors = true,
    .use_shifts = true,
    .balance_matrix = true,
    .custom_config = NULL
};

bool set_qr_config(const qr_config_t* config) {
    if (!config) return false;
    memcpy(&g_qr_config, config, sizeof(qr_config_t));
    return true;
}

bool get_qr_config(qr_config_t* config) {
    if (!config) return false;
    memcpy(config, &g_qr_config, sizeof(qr_config_t));
    return true;
}

// Helper function to check if subdiagonal element is negligible
static bool is_negligible(
    const ComplexFloat* a,
    size_t n,
    size_t i) {
    
    double scale = complex_float_abs(a[i * n + i]) + 
                   complex_float_abs(a[(i + 1) * n + i + 1]);
    
    if (scale == 0.0) scale = 1.0;
    
    return complex_float_abs(a[(i + 1) * n + i]) <= 
           g_qr_config.convergence_threshold * scale;
}

// Helper function to select eigenvalue closer to a22
static ComplexFloat select_eigenvalue(ComplexFloat e1, ComplexFloat e2, ComplexFloat a22) {
    double dist1 = (double)complex_float_abs_squared(complex_float_subtract(e1, a22));
    double dist2 = (double)complex_float_abs_squared(complex_float_subtract(e2, a22));
    if (dist1 <= dist2) {
        return e1;
    } else {
        return e2;
    }
}

// Helper function to compute Wilkinson shift
static ComplexFloat compute_wilkinson_shift(
    const ComplexFloat* a,
    size_t n,
    size_t m) {
    
    ComplexFloat a11 = a[(m - 1) * n + (m - 1)];
    ComplexFloat a12 = a[(m - 1) * n + m];
    ComplexFloat a21 = a[m * n + (m - 1)];
    ComplexFloat a22 = a[m * n + m];
    
    // Compute eigenvalues of 2x2 block
    ComplexFloat sum = complex_float_add(a11, a22);
    ComplexFloat d = complex_float_multiply_real(sum, 0.5f);
    
    ComplexFloat prod1 = complex_float_multiply(a11, a22);
    ComplexFloat prod2 = complex_float_multiply(a12, a21);
    ComplexFloat t = complex_float_subtract(prod1, prod2);
    
    ComplexFloat s = complex_float_sqrt(complex_float_subtract(
        complex_float_multiply(d, d),
        t
    ));
    
    // Return eigenvalue closer to a22
    ComplexFloat e1 = complex_float_add(d, s);
    ComplexFloat e2 = complex_float_subtract(d, s);
    
    return select_eigenvalue(e1, e2, a22);
}

// QR iteration with shifts
static bool qr_iteration(
    ComplexFloat* h,
    ComplexFloat* s,
    size_t n,
    size_t start,
    size_t end) {
    
    if (end - start <= 1) return true;
    
    // Compute shift
    ComplexFloat shift;
    bool use_shifts = g_qr_config.use_shifts;
    if (use_shifts) {
        shift = compute_wilkinson_shift(h, n, end - 1);
    } else {
        shift = complex_float_create(0.0f, 0.0f);
    }
    
    // Apply shift
    for (size_t i = start; i < end; i++) {
        h[i * n + i] = complex_float_subtract(h[i * n + i], shift);
    }
    
    // Allocate workspace
    ComplexFloat* q = malloc(n * n * sizeof(ComplexFloat));
    ComplexFloat* r = malloc(n * n * sizeof(ComplexFloat));
    if (!q || !r) {
        free(q);
        free(r);
        return false;
    }
    
    // Compute QR decomposition
    if (!compute_qr_decomposition(h + start * n + start,
                                q, r,
                                end - start,
                                end - start)) {
        free(q);
        free(r);
        return false;
    }
    
    // Form RQ
    if (!matrix_multiply(r, q, h + start * n + start,
                        end - start,
                        end - start,
                        end - start)) {
        free(q);
        free(r);
        return false;
    }
    
    // Unapply shift
    for (size_t i = start; i < end; i++) {
        h[i * n + i] = complex_float_add(h[i * n + i], shift);
    }
    
    // Update similarity transformation
    if (s) {
        if (!matrix_multiply(s + start * n,
                           q,
                           s + start * n,
                           n,
                           end - start,
                           end - start)) {
            free(q);
            free(r);
            return false;
        }
    }
    
    free(q);
    free(r);
    return true;
}

// Find eigenvalues using QR algorithm
bool find_eigenvalues(
    ComplexFloat* a,
    ComplexFloat* eigenvalues,
    ComplexFloat* eigenvectors,
    size_t n) {
    
    if (!a || !eigenvalues || n == 0) {
        return false;
    }
    
    // Allocate workspace
    ComplexFloat* h = malloc(n * n * sizeof(ComplexFloat));
    ComplexFloat* s = NULL;
    if (g_qr_config.compute_eigenvectors) {
        s = malloc(n * n * sizeof(ComplexFloat));
        if (!s) {
            free(h);
            return false;
        }
    }
    if (!h) {
        free(s);
        return false;
    }
    
    // Copy input matrix
    memcpy(h, a, n * n * sizeof(ComplexFloat));
    
    // Initialize similarity transformation
    if (s) {
        for (size_t i = 0; i < n * n; i++) {
            s[i] = complex_float_create(0.0f, 0.0f);
        }
        for (size_t i = 0; i < n; i++) {
            s[i * n + i] = complex_float_create(1.0f, 0.0f);
        }
    }
    
    // Reduce to Hessenberg form
    if (!compute_hessenberg_form(h, s, n)) {
        free(h);
        free(s);
        return false;
    }
    
    // Main QR iteration loop
    size_t total_iter = 0;
    size_t m = n;
    
    while (m > 1 && total_iter < g_qr_config.max_iterations) {
        // Look for single small subdiagonal element
        size_t l;
        for (l = m - 1; l > 0; l--) {
            if (is_negligible(h, n, l - 1)) {
                break;
            }
        }
        
        if (l == m - 1) {
            // One eigenvalue found
            eigenvalues[m - 1] = complex_float_create(
                h[(m - 1) * n + (m - 1)].real,
                h[(m - 1) * n + (m - 1)].imag
            );
            m--;
        } else {
            // QR iteration on active submatrix
            if (!qr_iteration(h, s, n, l, m)) {
                free(h);
                free(s);
                return false;
            }
        }
        
        total_iter++;
    }
    
    // Get last eigenvalue
    if (m == 1) {
        eigenvalues[0] = complex_float_create(h[0].real, h[0].imag);
    }
    
    // Copy eigenvectors if requested
    if (eigenvectors && s) {
        memcpy(eigenvectors, s, n * n * sizeof(ComplexFloat));
    }
    
    free(h);
    free(s);
    
    return total_iter < g_qr_config.max_iterations;
}
