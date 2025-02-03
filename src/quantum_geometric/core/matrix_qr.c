#include "quantum_geometric/core/matrix_operations.h"
#include "quantum_geometric/core/error_handling.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Helper function for complex conjugate
static ComplexFloat complex_conj(ComplexFloat a) {
    return (ComplexFloat){a.real, -a.imag};
}

// Helper function for complex absolute value squared
static double complex_abs_sq(ComplexFloat a) {
    return a.real * a.real + a.imag * a.imag;
}

// Helper function for complex dot product
static ComplexFloat complex_dot(
    const ComplexFloat* a,
    const ComplexFloat* b,
    size_t n) {
    
    ComplexFloat sum = {0, 0};
    for (size_t i = 0; i < n; i++) {
        ComplexFloat prod = {
            a[i].real * b[i].real + a[i].imag * b[i].imag,
            a[i].real * b[i].imag - a[i].imag * b[i].real
        };
        sum.real += prod.real;
        sum.imag += prod.imag;
    }
    return sum;
}

// Compute Householder reflection vector
static bool compute_householder_vector(
    const ComplexFloat* x,
    ComplexFloat* v,
    size_t n) {
    
    if (!x || !v || n == 0) {
        return false;
    }
    
    // Compute norm of x
    double norm = 0;
    for (size_t i = 0; i < n; i++) {
        norm += complex_abs_sq(x[i]);
    }
    norm = sqrt(norm);
    
    if (norm < 1e-10) {
        return false;
    }
    
    // v = x + sign(x[0])|x|e1
    double sign = (x[0].real > 0) ? 1.0 : -1.0;
    memcpy(v, x, n * sizeof(ComplexFloat));
    v[0].real += sign * norm;
    
    // Normalize v
    norm = 0;
    for (size_t i = 0; i < n; i++) {
        norm += complex_abs_sq(v[i]);
    }
    norm = sqrt(norm);
    
    if (norm < 1e-10) {
        return false;
    }
    
    for (size_t i = 0; i < n; i++) {
        v[i].real /= norm;
        v[i].imag /= norm;
    }
    
    return true;
}

// Apply Householder reflection to matrix
static bool apply_householder(
    ComplexFloat* a,
    const ComplexFloat* v,
    size_t m,
    size_t n,
    bool left) {
    
    if (!a || !v || m == 0 || n == 0) {
        return false;
    }
    
    ComplexFloat* temp = malloc(left ? n : m * sizeof(ComplexFloat));
    if (!temp) return false;
    
    if (left) {
        // Left multiplication: H*A
        for (size_t j = 0; j < n; j++) {
            // Compute v'*A(:,j)
            ComplexFloat dot = {0, 0};
            for (size_t i = 0; i < m; i++) {
                dot.real += v[i].real * a[i * n + j].real + 
                           v[i].imag * a[i * n + j].imag;
                dot.imag += v[i].real * a[i * n + j].imag - 
                           v[i].imag * a[i * n + j].real;
            }
            
            // Update column: A(:,j) = A(:,j) - 2*v*(v'*A(:,j))
            for (size_t i = 0; i < m; i++) {
                ComplexFloat prod = {
                    v[i].real * dot.real - v[i].imag * dot.imag,
                    v[i].real * dot.imag + v[i].imag * dot.real
                };
                a[i * n + j].real -= 2 * prod.real;
                a[i * n + j].imag -= 2 * prod.imag;
            }
        }
    } else {
        // Right multiplication: A*H
        for (size_t i = 0; i < m; i++) {
            // Compute A(i,:)*v
            ComplexFloat dot = {0, 0};
            for (size_t j = 0; j < n; j++) {
                dot.real += a[i * n + j].real * v[j].real + 
                           a[i * n + j].imag * v[j].imag;
                dot.imag += a[i * n + j].real * v[j].imag - 
                           a[i * n + j].imag * v[j].real;
            }
            
            // Update row: A(i,:) = A(i,:) - 2*(A(i,:)*v)*v'
            for (size_t j = 0; j < n; j++) {
                ComplexFloat prod = {
                    dot.real * v[j].real - dot.imag * v[j].imag,
                    dot.real * v[j].imag + dot.imag * v[j].real
                };
                a[i * n + j].real -= 2 * prod.real;
                a[i * n + j].imag -= 2 * prod.imag;
            }
        }
    }
    
    free(temp);
    return true;
}

// Compute QR decomposition using Householder reflections
bool compute_qr_decomposition(
    ComplexFloat* a,
    ComplexFloat* q,
    ComplexFloat* r,
    size_t m,
    size_t n) {
    
    if (!a || !q || !r || m == 0 || n == 0 || m < n) {
        return false;
    }
    
    // Initialize Q to identity
    for (size_t i = 0; i < m * m; i++) {
        q[i] = (ComplexFloat){0, 0};
    }
    for (size_t i = 0; i < m; i++) {
        q[i * m + i] = (ComplexFloat){1, 0};
    }
    
    // Copy A to R
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            r[i * n + j] = a[i * n + j];
        }
    }
    
    // Allocate workspace for Householder vector
    ComplexFloat* v = malloc(m * sizeof(ComplexFloat));
    if (!v) return false;
    
    // Compute QR decomposition
    for (size_t j = 0; j < n; j++) {
        size_t h = m - j;  // Size of Householder vector
        
        // Get column to eliminate
        for (size_t i = 0; i < h; i++) {
            v[i] = r[(i + j) * n + j];
        }
        
        // Compute Householder vector
        if (!compute_householder_vector(v, v, h)) {
            free(v);
            return false;
        }
        
        // Apply to R
        if (!apply_householder(r + j * n + j, v, h, n - j, true)) {
            free(v);
            return false;
        }
        
        // Accumulate Q
        if (!apply_householder(q + j * m, v, h, m, false)) {
            free(v);
            return false;
        }
    }
    
    free(v);
    return true;
}

// Compute Hessenberg form using Householder reflections
bool compute_hessenberg_form(
    ComplexFloat* a,
    ComplexFloat* q,
    size_t n) {
    
    if (!a || !q || n == 0) {
        return false;
    }
    
    // Initialize Q to identity
    for (size_t i = 0; i < n * n; i++) {
        q[i] = (ComplexFloat){0, 0};
    }
    for (size_t i = 0; i < n; i++) {
        q[i * n + i] = (ComplexFloat){1, 0};
    }
    
    // Allocate workspace
    ComplexFloat* v = malloc(n * sizeof(ComplexFloat));
    if (!v) return false;
    
    // Reduce to Hessenberg form
    for (size_t j = 0; j < n - 2; j++) {
        size_t h = n - j - 1;  // Size of Householder vector
        
        // Get column to eliminate
        for (size_t i = 0; i < h; i++) {
            v[i] = a[(i + j + 1) * n + j];
        }
        
        // Compute Householder vector
        if (!compute_householder_vector(v, v, h)) {
            free(v);
            return false;
        }
        
        // Apply to A from both sides
        if (!apply_householder(a + (j + 1) * n + j, v, h, n - j, true)) {
            free(v);
            return false;
        }
        if (!apply_householder(a + j * n + j + 1, v, h, n - j - 1, false)) {
            free(v);
            return false;
        }
        
        // Accumulate Q
        if (!apply_householder(q + (j + 1) * n, v, h, n, false)) {
            free(v);
            return false;
        }
    }
    
    free(v);
    return true;
}
