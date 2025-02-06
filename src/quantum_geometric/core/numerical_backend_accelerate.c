#include "quantum_geometric/core/numerical_backend.h"
#include "quantum_geometric/core/complex_arithmetic.h"
#include "quantum_geometric/core/lapack_wrapper.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __APPLE__

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
    
    // Check for LAPACK availability
    backend_state.has_lapack = lapack_has_capability("svd");
    
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
    
    // Convert to DSPComplex format
    DSPComplex* dsp_a = malloc(m * k * sizeof(DSPComplex));
    DSPComplex* dsp_b = malloc(k * n * sizeof(DSPComplex));
    DSPComplex* dsp_c = malloc(m * n * sizeof(DSPComplex));
    
    if (!dsp_a || !dsp_b || !dsp_c) {
        free(dsp_a);
        free(dsp_b);
        free(dsp_c);
        backend_state.last_error = NUMERICAL_ERROR_MEMORY;
        return false;
    }
    
    // Convert input matrices using our conversion utilities
    for (size_t i = 0; i < m * k; i++) {
        dsp_a[i] = to_dsp_complex(a[i]);
    }
    for (size_t i = 0; i < k * n; i++) {
        dsp_b[i] = to_dsp_complex(b[i]);
    }
    
    // Perform matrix multiplication using vDSP
    vDSP_zmmul(dsp_a, 1, dsp_b, 1, dsp_c, 1, m, n, k);
    
    // Convert result back using our conversion utilities
    for (size_t i = 0; i < m * n; i++) {
        c[i] = from_dsp_complex(dsp_c[i]);
    }
    
    free(dsp_a);
    free(dsp_b);
    free(dsp_c);
    
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
    DSPComplex* dsp_a = malloc(total * sizeof(DSPComplex));
    DSPComplex* dsp_b = malloc(total * sizeof(DSPComplex));
    
    if (!dsp_a || !dsp_b) {
        free(dsp_a);
        free(dsp_b);
        backend_state.last_error = NUMERICAL_ERROR_MEMORY;
        return false;
    }
    
    // Convert inputs using our conversion utilities
    for (size_t i = 0; i < total; i++) {
        dsp_a[i] = to_dsp_complex(a[i]);
        dsp_b[i] = to_dsp_complex(b[i]);
    }
    
    // Perform addition using vDSP
    vDSP_zvadd(dsp_a, 1, dsp_b, 1, (DSPComplex*)c, 1, total);
    
    free(dsp_a);
    free(dsp_b);
    
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
    
    DSPComplex* dsp_a = malloc(length * sizeof(DSPComplex));
    DSPComplex* dsp_b = malloc(length * sizeof(DSPComplex));
    
    if (!dsp_a || !dsp_b) {
        free(dsp_a);
        free(dsp_b);
        backend_state.last_error = NUMERICAL_ERROR_MEMORY;
        return false;
    }
    
    // Convert inputs using our conversion utilities
    for (size_t i = 0; i < length; i++) {
        dsp_a[i] = to_dsp_complex(a[i]);
        dsp_b[i] = to_dsp_complex(b[i]);
    }
    
    // Compute dot product using vDSP
    DSPComplex dot;
    vDSP_zdotpr(dsp_a, 1, dsp_b, 1, &dot, length);
    *result = from_dsp_complex(dot);
    
    free(dsp_a);
    free(dsp_b);
    
    backend_state.last_error = NUMERICAL_SUCCESS;
    return true;
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

    // If LAPACK is not available, fall back to CPU implementation
    return numerical_svd_cpu(a, u, s, vt, m, n);
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

#endif // __APPLE__
