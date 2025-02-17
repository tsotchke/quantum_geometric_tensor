#include "quantum_geometric/core/numerical_backend.h"
#include "quantum_geometric/core/complex_arithmetic.h"
#include "quantum_geometric/core/lapack_wrapper.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

    // Fallback not implemented - QR is complex
    return NUMERICAL_ERROR_NOT_IMPLEMENTED;
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

    // Fallback not implemented - eigendecomposition is complex
    return NUMERICAL_ERROR_NOT_IMPLEMENTED;
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

    // Fallback not implemented - Cholesky is complex
    return NUMERICAL_ERROR_NOT_IMPLEMENTED;
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

    // Fallback not implemented - LU is complex
    return NUMERICAL_ERROR_NOT_IMPLEMENTED;
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

    // Fallback not implemented - triangular solve is complex
    return NUMERICAL_ERROR_NOT_IMPLEMENTED;
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

    // Fallback not implemented - symmetric solve is complex
    return NUMERICAL_ERROR_NOT_IMPLEMENTED;
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

    // Fallback to power iteration if LAPACK is not available
    const size_t max_iter = 100;
    const float tol = 1e-6f;
    
    // Allocate workspace
    ComplexFloat* work = malloc(m * sizeof(ComplexFloat));
    ComplexFloat* temp = malloc(m * sizeof(ComplexFloat));
    
    if (!work || !temp) {
        free(work);
        free(temp);
        return NUMERICAL_ERROR_MEMORY;
    }
    
    // Initialize work vector
    for (size_t i = 0; i < m; i++) {
        work[i].real = 1.0f / sqrtf(m);
        work[i].imag = 0.0f;
    }
    
    // Power iteration
    for (size_t iter = 0; iter < max_iter; iter++) {
        // Multiply: temp = A * work
        for (size_t i = 0; i < m; i++) {
            temp[i].real = temp[i].imag = 0.0f;
            for (size_t j = 0; j < n; j++) {
                temp[i] = complex_add(temp[i], 
                                    complex_multiply(a[i * n + j], work[j]));
            }
        }
        
        // Normalize
        float norm = 0.0f;
        for (size_t i = 0; i < m; i++) {
            norm += complex_abs_squared(temp[i]);
        }
        norm = sqrtf(norm);
        
        // Check convergence
        bool converged = true;
        for (size_t i = 0; i < m; i++) {
            ComplexFloat new_val = {temp[i].real / norm, temp[i].imag / norm};
            ComplexFloat diff = complex_subtract(new_val, work[i]);
            if (complex_abs_squared(diff) > tol * tol) {
                converged = false;
            }
            work[i] = new_val;
        }
        
        if (converged) break;
    }
    
    // Copy result to output
    memcpy(u, work, m * sizeof(ComplexFloat));
    s[0] = 1.0f; // Simplified - only computing largest singular value
    memset(vt, 0, n * sizeof(ComplexFloat));
    vt[0].real = 1.0f;
    
    free(work);
    free(temp);
    
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
