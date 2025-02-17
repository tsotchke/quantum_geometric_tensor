#include "quantum_geometric/core/numerical_backend.h"
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

// Static backend instance
static struct {
    numerical_backend_t active_backend;
    bool initialized;
    void* backend_handle;
} backend_selector = {
    .active_backend = NUMERICAL_BACKEND_CPU,
    .initialized = false,
    .backend_handle = NULL
};

// Forward declarations for backend-specific functions
#ifdef __APPLE__
extern bool initialize_numerical_backend_accelerate(const numerical_config_t* config);
extern void shutdown_numerical_backend_accelerate(void);
extern numerical_error_t get_last_numerical_error_accelerate(void);
#endif

extern bool initialize_numerical_backend_cpu(const numerical_config_t* config);
extern void shutdown_numerical_backend_cpu(void);
extern numerical_error_t get_last_numerical_error_cpu(void);

// Error string lookup table
static const char* error_strings[] = {
    [NUMERICAL_SUCCESS] = "Success",
    [NUMERICAL_ERROR_INVALID_ARGUMENT] = "Invalid argument",
    [NUMERICAL_ERROR_MEMORY] = "Memory error",
    [NUMERICAL_ERROR_BACKEND] = "Backend error",
    [NUMERICAL_ERROR_COMPUTATION] = "Computation error",
    [NUMERICAL_ERROR_NOT_IMPLEMENTED] = "Not implemented",
    [NUMERICAL_ERROR_INVALID_STATE] = "Invalid state"
};

const char* get_numerical_error_string(numerical_error_t error) {
    if (error < 0 || error >= sizeof(error_strings)/sizeof(error_strings[0])) {
        return "Unknown error";
    }
    return error_strings[error];
}

numerical_error_t select_optimal_backend(numerical_config_t* config) {
    if (!config) {
        return NUMERICAL_ERROR_INVALID_ARGUMENT;
    }
    
    // Check available backends and their capabilities
    bool has_accelerate = false;
    bool has_metal = false;
    
#ifdef __APPLE__
    has_accelerate = true;
    #if defined(TARGET_OS_MAC) && TARGET_OS_MAC
    has_metal = true;
    #endif
#endif

    // Select best available backend based on capabilities and preferences
    if (config->type == NUMERICAL_BACKEND_CPU) {
        backend_selector.active_backend = NUMERICAL_BACKEND_CPU;
    }
#ifdef __APPLE__
    else if (has_metal && config->type == NUMERICAL_BACKEND_METAL) {
        backend_selector.active_backend = NUMERICAL_BACKEND_METAL;
    }
    else if (has_accelerate && (config->type == NUMERICAL_BACKEND_ACCELERATE || 
             config->type == NUMERICAL_BACKEND_CPU)) {
        backend_selector.active_backend = NUMERICAL_BACKEND_ACCELERATE;
    }
#endif
#ifdef __CUDACC__
    else if (config->type == NUMERICAL_BACKEND_CUDA) {
        backend_selector.active_backend = NUMERICAL_BACKEND_CUDA;
    }
#endif
    else {
        // Default to CPU if requested backend is not available
        backend_selector.active_backend = NUMERICAL_BACKEND_CPU;
    }
    
    return NUMERICAL_SUCCESS;
}

numerical_error_t initialize_numerical_backend(const numerical_config_t* config) {
    if (!config) {
        return NUMERICAL_ERROR_INVALID_ARGUMENT;
    }
    
    // Clean up any existing backend
    if (backend_selector.initialized) {
        shutdown_numerical_backend();
    }
    
    // Select appropriate backend
    numerical_config_t local_config = *config;
    numerical_error_t error = select_optimal_backend(&local_config);
    if (error != NUMERICAL_SUCCESS) {
        return error;
    }
    
    // Initialize selected backend
    bool success = false;
    switch (backend_selector.active_backend) {
        case NUMERICAL_BACKEND_CPU:
            success = initialize_numerical_backend_cpu(&local_config);
            break;
            
#ifdef __APPLE__
        case NUMERICAL_BACKEND_ACCELERATE:
            success = initialize_numerical_backend_accelerate(&local_config);
            break;
#endif
            
        default:
            return NUMERICAL_ERROR_NOT_IMPLEMENTED;
    }
    
    if (!success) {
        return NUMERICAL_ERROR_BACKEND;
    }
    
    backend_selector.initialized = true;
    return NUMERICAL_SUCCESS;
}

void shutdown_numerical_backend(void) {
    if (!backend_selector.initialized) return;
    
    switch (backend_selector.active_backend) {
        case NUMERICAL_BACKEND_CPU:
            shutdown_numerical_backend_cpu();
            break;
            
#ifdef __APPLE__
        case NUMERICAL_BACKEND_ACCELERATE:
            shutdown_numerical_backend_accelerate();
            break;
#endif
            
        default:
            break;
    }
    
    backend_selector.initialized = false;
    backend_selector.backend_handle = NULL;
}

numerical_error_t numerical_matrix_multiply(const ComplexFloat* a,
                                          const ComplexFloat* b,
                                          ComplexFloat* c,
                                          size_t m, size_t k, size_t n,
                                          bool transpose_a,
                                          bool transpose_b) {
    if (!backend_selector.initialized) {
        return NUMERICAL_ERROR_INVALID_STATE;
    }
    
    if (!a || !b || !c) {
        return NUMERICAL_ERROR_INVALID_ARGUMENT;
    }
    
    switch (backend_selector.active_backend) {
        case NUMERICAL_BACKEND_CPU:
            return numerical_matrix_multiply_cpu(a, b, c, m, k, n, transpose_a, transpose_b);
            
#ifdef __APPLE__
        case NUMERICAL_BACKEND_ACCELERATE:
            return numerical_matrix_multiply_accelerate(a, b, c, m, k, n, transpose_a, transpose_b);
#endif
            
        default:
            return NUMERICAL_ERROR_NOT_IMPLEMENTED;
    }
}

numerical_error_t numerical_matrix_add(const ComplexFloat* a,
                                     const ComplexFloat* b,
                                     ComplexFloat* c,
                                     size_t rows,
                                     size_t cols) {
    if (!backend_selector.initialized) {
        return NUMERICAL_ERROR_INVALID_STATE;
    }
    
    if (!a || !b || !c) {
        return NUMERICAL_ERROR_INVALID_ARGUMENT;
    }
    
    switch (backend_selector.active_backend) {
        case NUMERICAL_BACKEND_CPU:
            return numerical_matrix_add_cpu(a, b, c, rows, cols);
            
#ifdef __APPLE__
        case NUMERICAL_BACKEND_ACCELERATE:
            return numerical_matrix_add_accelerate(a, b, c, rows, cols);
#endif
            
        default:
            return NUMERICAL_ERROR_NOT_IMPLEMENTED;
    }
}

numerical_error_t numerical_vector_dot(const ComplexFloat* a,
                                     const ComplexFloat* b,
                                     ComplexFloat* result,
                                     size_t length) {
    if (!backend_selector.initialized) {
        return NUMERICAL_ERROR_INVALID_STATE;
    }
    
    if (!a || !b || !result) {
        return NUMERICAL_ERROR_INVALID_ARGUMENT;
    }
    
    switch (backend_selector.active_backend) {
        case NUMERICAL_BACKEND_CPU:
            return numerical_vector_dot_cpu(a, b, result, length);
            
#ifdef __APPLE__
        case NUMERICAL_BACKEND_ACCELERATE:
            return numerical_vector_dot_accelerate(a, b, result, length);
#endif
            
        default:
            return NUMERICAL_ERROR_NOT_IMPLEMENTED;
    }
}

numerical_error_t numerical_svd(const ComplexFloat* a,
                              ComplexFloat* u,
                              float* s,
                              ComplexFloat* vt,
                              size_t m,
                              size_t n) {
    if (!backend_selector.initialized) {
        return NUMERICAL_ERROR_INVALID_STATE;
    }
    
    if (!a || !u || !s || !vt) {
        return NUMERICAL_ERROR_INVALID_ARGUMENT;
    }
    
    switch (backend_selector.active_backend) {
        case NUMERICAL_BACKEND_CPU:
            return numerical_svd_cpu(a, u, s, vt, m, n);
            
#ifdef __APPLE__
        case NUMERICAL_BACKEND_ACCELERATE:
            return numerical_svd_accelerate(a, u, s, vt, m, n);
#endif
            
        default:
            return NUMERICAL_ERROR_NOT_IMPLEMENTED;
    }
}

numerical_error_t get_numerical_metrics(numerical_metrics_t* metrics) {
    if (!backend_selector.initialized) {
        return NUMERICAL_ERROR_INVALID_STATE;
    }
    
    if (!metrics) {
        return NUMERICAL_ERROR_INVALID_ARGUMENT;
    }
    
    switch (backend_selector.active_backend) {
        case NUMERICAL_BACKEND_CPU:
            return get_numerical_metrics_cpu(metrics);
            
#ifdef __APPLE__
        case NUMERICAL_BACKEND_ACCELERATE:
            return get_numerical_metrics_accelerate(metrics);
#endif
            
        default:
            return NUMERICAL_ERROR_NOT_IMPLEMENTED;
    }
}

numerical_error_t reset_numerical_metrics(void) {
    if (!backend_selector.initialized) {
        return NUMERICAL_ERROR_INVALID_STATE;
    }
    
    switch (backend_selector.active_backend) {
        case NUMERICAL_BACKEND_CPU:
            return reset_numerical_metrics_cpu();
            
#ifdef __APPLE__
        case NUMERICAL_BACKEND_ACCELERATE:
            return reset_numerical_metrics_accelerate();
#endif
            
        default:
            return NUMERICAL_ERROR_NOT_IMPLEMENTED;
    }
}

numerical_error_t get_last_numerical_error(void) {
    if (!backend_selector.initialized) return NUMERICAL_ERROR_INVALID_STATE;
    
    switch (backend_selector.active_backend) {
        case NUMERICAL_BACKEND_CPU:
            return get_last_numerical_error_cpu();
            
#ifdef __APPLE__
        case NUMERICAL_BACKEND_ACCELERATE:
            return get_last_numerical_error_accelerate();
#endif
            
        default:
            return NUMERICAL_ERROR_INVALID_STATE;
    }
}
