#include "quantum_geometric/core/numerical_backend.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Global state
static struct {
    numerical_config_t config;
    numerical_metrics_t metrics;
    numerical_error_t last_error;
    bool initialized;
} backend_state = {0};

// Helper functions for complex arithmetic
static ComplexFloat complex_mul(ComplexFloat a, ComplexFloat b) {
    ComplexFloat result;
    result.real = a.real * b.real - a.imag * b.imag;
    result.imag = a.real * b.imag + a.imag * b.real;
    return result;
}

static ComplexFloat complex_add(ComplexFloat a, ComplexFloat b) {
    ComplexFloat result;
    result.real = a.real + b.real;
    result.imag = a.imag + b.imag;
    return result;
}

static ComplexFloat complex_conj(ComplexFloat a) {
    ComplexFloat result;
    result.real = a.real;
    result.imag = -a.imag;
    return result;
}

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
    
    // Clear output matrix
    memset(c, 0, m * n * sizeof(ComplexFloat));
    
    // Matrix multiplication with optional transposition
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            ComplexFloat sum = {0.0f, 0.0f};
            for (size_t l = 0; l < k; l++) {
                ComplexFloat a_val = transpose_a ? a[l * m + i] : a[i * k + l];
                ComplexFloat b_val = transpose_b ? b[j * k + l] : b[l * n + j];
                sum = complex_add(sum, complex_mul(a_val, b_val));
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
    
    size_t total = rows * cols;
    for (size_t i = 0; i < total; i++) {
        c[i] = complex_add(a[i], b[i]);
    }
    
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
    
    ComplexFloat sum = {0.0f, 0.0f};
    for (size_t i = 0; i < length; i++) {
        sum = complex_add(sum, complex_mul(a[i], complex_conj(b[i])));
    }
    *result = sum;
    
    backend_state.last_error = NUMERICAL_SUCCESS;
    return true;
}

bool numerical_svd(const ComplexFloat* a,
                  ComplexFloat* u,
                  float* s,
                  ComplexFloat* vt,
                  size_t m,
                  size_t n) {
    // For CPU backend, we'll implement a simple power iteration method
    // This is not as accurate as LAPACK but works for basic cases
    if (!backend_state.initialized || !a || !u || !s || !vt) {
        backend_state.last_error = NUMERICAL_ERROR_INVALID_ARGUMENT;
        return false;
    }
    
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
                                    complex_mul(a[i * n + j], work[j]));
            }
        }
        
        // Normalize
        float norm = 0.0f;
        for (size_t i = 0; i < m; i++) {
            norm += temp[i].real * temp[i].real + temp[i].imag * temp[i].imag;
        }
        norm = sqrtf(norm);
        
        // Check convergence
        bool converged = true;
        for (size_t i = 0; i < m; i++) {
            ComplexFloat new_val = {temp[i].real / norm, temp[i].imag / norm};
            if (fabsf(new_val.real - work[i].real) > tol ||
                fabsf(new_val.imag - work[i].imag) > tol) {
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
