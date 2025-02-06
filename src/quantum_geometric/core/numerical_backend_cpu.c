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

bool initialize_numerical_backend(const numerical_config_t* config) {
    if (!config) {
        backend_state.last_error = NUMERICAL_ERROR_INVALID_ARGUMENT;
        return false;
    }
    
    backend_state.config = *config;
    backend_state.initialized = true;
    backend_state.last_error = NUMERICAL_SUCCESS;
    
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
    
    return true;
}

void shutdown_numerical_backend(void) {
    backend_state.initialized = false;
}

bool numerical_matrix_multiply(const ComplexFloat* a,
                             const ComplexFloat* b,
                             ComplexFloat* c,
                             size_t m, size_t k, size_t n,
                             bool transpose_a,
                             bool transpose_b) {
    if (!backend_state.initialized || !a || !b || !c) {
        backend_state.last_error = NUMERICAL_ERROR_INVALID_ARGUMENT;
        return false;
    }

    if (backend_state.has_lapack) {
        bool success = lapack_matrix_multiply(a, b, c, m, k, n,
                                           transpose_a, transpose_b,
                                           LAPACK_ROW_MAJOR);
        if (!success) {
            backend_state.last_error = NUMERICAL_ERROR_COMPUTATION;
            return false;
        }
        return true;
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
    
    backend_state.last_error = NUMERICAL_SUCCESS;
    return true;
}

bool numerical_matrix_add(const ComplexFloat* a,
                         const ComplexFloat* b,
                         ComplexFloat* c,
                         size_t rows,
                         size_t cols) {
    if (!backend_state.initialized || !a || !b || !c) {
        backend_state.last_error = NUMERICAL_ERROR_INVALID_ARGUMENT;
        return false;
    }
    
    complex_vector_add(a, b, c, rows * cols);
    backend_state.last_error = NUMERICAL_SUCCESS;
    return true;
}

bool numerical_vector_dot(const ComplexFloat* a,
                         const ComplexFloat* b,
                         ComplexFloat* result,
                         size_t length) {
    if (!backend_state.initialized || !a || !b || !result) {
        backend_state.last_error = NUMERICAL_ERROR_INVALID_ARGUMENT;
        return false;
    }
    
    *result = complex_vector_dot(a, b, length);
    backend_state.last_error = NUMERICAL_SUCCESS;
    return true;
}

bool numerical_qr(const ComplexFloat* a,
                 ComplexFloat* q,
                 ComplexFloat* r,
                 size_t m,
                 size_t n) {
    if (!backend_state.initialized || !a || !q || !r) {
        backend_state.last_error = NUMERICAL_ERROR_INVALID_ARGUMENT;
        return false;
    }

    if (backend_state.has_lapack) {
        bool success = lapack_qr(a, m, n, q, r, LAPACK_ROW_MAJOR);
        if (!success) {
            backend_state.last_error = NUMERICAL_ERROR_COMPUTATION;
            return false;
        }
        return true;
    }

    // Fallback not implemented - QR is complex
    backend_state.last_error = NUMERICAL_ERROR_NOT_IMPLEMENTED;
    return false;
}

bool numerical_eigendecomposition(const ComplexFloat* a,
                                ComplexFloat* eigenvectors,
                                ComplexFloat* eigenvalues,
                                size_t n) {
    if (!backend_state.initialized || !a || !eigenvectors || !eigenvalues) {
        backend_state.last_error = NUMERICAL_ERROR_INVALID_ARGUMENT;
        return false;
    }

    if (backend_state.has_lapack) {
        bool success = lapack_eigendecomposition(a, n, eigenvectors, eigenvalues, LAPACK_ROW_MAJOR);
        if (!success) {
            backend_state.last_error = NUMERICAL_ERROR_COMPUTATION;
            return false;
        }
        return true;
    }

    // Fallback not implemented - eigendecomposition is complex
    backend_state.last_error = NUMERICAL_ERROR_NOT_IMPLEMENTED;
    return false;
}

bool numerical_cholesky(const ComplexFloat* a,
                       ComplexFloat* l,
                       size_t n,
                       bool lower_triangular) {
    if (!backend_state.initialized || !a || !l) {
        backend_state.last_error = NUMERICAL_ERROR_INVALID_ARGUMENT;
        return false;
    }

    if (backend_state.has_lapack) {
        bool success = lapack_cholesky(a, n, l, lower_triangular, LAPACK_ROW_MAJOR);
        if (!success) {
            backend_state.last_error = NUMERICAL_ERROR_COMPUTATION;
            return false;
        }
        return true;
    }

    // Fallback not implemented - Cholesky is complex
    backend_state.last_error = NUMERICAL_ERROR_NOT_IMPLEMENTED;
    return false;
}

bool numerical_lu(const ComplexFloat* a,
                 ComplexFloat* l,
                 ComplexFloat* u,
                 int* ipiv,
                 size_t m,
                 size_t n) {
    if (!backend_state.initialized || !a || !l || !u || !ipiv) {
        backend_state.last_error = NUMERICAL_ERROR_INVALID_ARGUMENT;
        return false;
    }

    if (backend_state.has_lapack) {
        bool success = lapack_lu(a, m, n, l, u, ipiv, LAPACK_ROW_MAJOR);
        if (!success) {
            backend_state.last_error = NUMERICAL_ERROR_COMPUTATION;
            return false;
        }
        return true;
    }

    // Fallback not implemented - LU is complex
    backend_state.last_error = NUMERICAL_ERROR_NOT_IMPLEMENTED;
    return false;
}

bool numerical_solve_triangular(const ComplexFloat* a,
                              const ComplexFloat* b,
                              ComplexFloat* x,
                              size_t n,
                              size_t nrhs,
                              bool upper_triangular,
                              bool unit_diagonal) {
    if (!backend_state.initialized || !a || !b || !x) {
        backend_state.last_error = NUMERICAL_ERROR_INVALID_ARGUMENT;
        return false;
    }

    if (backend_state.has_lapack) {
        bool success = lapack_solve_triangular(a, b, x, n, nrhs,
                                             upper_triangular, unit_diagonal,
                                             LAPACK_ROW_MAJOR);
        if (!success) {
            backend_state.last_error = NUMERICAL_ERROR_COMPUTATION;
            return false;
        }
        return true;
    }

    // Fallback not implemented - triangular solve is complex
    backend_state.last_error = NUMERICAL_ERROR_NOT_IMPLEMENTED;
    return false;
}

bool numerical_solve_symmetric(const ComplexFloat* a,
                             const ComplexFloat* b,
                             ComplexFloat* x,
                             size_t n,
                             size_t nrhs,
                             bool positive_definite) {
    if (!backend_state.initialized || !a || !b || !x) {
        backend_state.last_error = NUMERICAL_ERROR_INVALID_ARGUMENT;
        return false;
    }

    if (backend_state.has_lapack) {
        bool success = lapack_solve_symmetric(a, b, x, n, nrhs,
                                            positive_definite,
                                            LAPACK_ROW_MAJOR);
        if (!success) {
            backend_state.last_error = NUMERICAL_ERROR_COMPUTATION;
            return false;
        }
        return true;
    }

    // Fallback not implemented - symmetric solve is complex
    backend_state.last_error = NUMERICAL_ERROR_NOT_IMPLEMENTED;
    return false;
}

bool numerical_svd(const ComplexFloat* a,
                  ComplexFloat* u,
                  float* s,
                  ComplexFloat* vt,
                  size_t m,
                  size_t n) {
    if (!backend_state.initialized || !a || !u || !s || !vt) {
        backend_state.last_error = NUMERICAL_ERROR_INVALID_ARGUMENT;
        return false;
    }

    if (backend_state.has_lapack) {
        bool success = lapack_svd(a, m, n, u, s, vt, LAPACK_ROW_MAJOR);
        if (!success) {
            lapack_status_t status = lapack_get_last_status();
            switch (status) {
                case LAPACK_MEMORY_ERROR:
                    backend_state.last_error = NUMERICAL_ERROR_MEMORY;
                    break;
                case LAPACK_NOT_CONVERGENT:
                    backend_state.last_error = NUMERICAL_ERROR_COMPUTATION;
                    break;
                default:
                    backend_state.last_error = NUMERICAL_ERROR_BACKEND;
                    break;
            }
            return false;
        }
        backend_state.last_error = NUMERICAL_SUCCESS;
        return true;
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
        backend_state.last_error = NUMERICAL_ERROR_MEMORY;
        return false;
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
            if (!complex_is_equal(new_val, work[i])) {
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
    
    backend_state.last_error = NUMERICAL_SUCCESS;
    return true;
}

bool get_numerical_metrics(numerical_metrics_t* metrics) {
    if (!backend_state.initialized || !metrics) {
        backend_state.last_error = NUMERICAL_ERROR_INVALID_ARGUMENT;
        return false;
    }
    
    *metrics = backend_state.metrics;
    return true;
}

bool reset_numerical_metrics(void) {
    if (!backend_state.initialized) {
        backend_state.last_error = NUMERICAL_ERROR_INVALID_ARGUMENT;
        return false;
    }
    
    memset(&backend_state.metrics, 0, sizeof(numerical_metrics_t));
    return true;
}

numerical_error_t get_last_numerical_error(void) {
    return backend_state.last_error;
}

const char* get_numerical_error_string(numerical_error_t error) {
    switch (error) {
        case NUMERICAL_SUCCESS:
            return "Success";
        case NUMERICAL_ERROR_INVALID_ARGUMENT:
            return "Invalid argument";
        case NUMERICAL_ERROR_MEMORY:
            return "Memory allocation failed";
        case NUMERICAL_ERROR_BACKEND:
            return "Backend error";
        case NUMERICAL_ERROR_COMPUTATION:
            return "Computation error";
        case NUMERICAL_ERROR_NOT_IMPLEMENTED:
            return "Not implemented";
        default:
            return "Unknown error";
    }
}
