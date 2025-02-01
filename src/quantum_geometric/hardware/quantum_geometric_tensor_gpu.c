#include "quantum_geometric/hardware/quantum_geometric_tensor_gpu.h"
#include "quantum_geometric/core/memory_pool.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Default configuration values
static const QGTConfig default_config = {
    .precision = 1e-10,
    .use_quantum_estimation = true,
    .use_quantum_memory = true,
    .error_correction = 2,  // Adaptive error correction
    .optimization_level = 3  // Aggressive optimization
};

// Error handling
const char* qgt_error_string(qgt_error_t error) {
    switch (error) {
        case QGT_SUCCESS:
            return "Success";
        case QGT_ERROR_INVALID_PARAMETER:
            return "Invalid parameter";
        case QGT_ERROR_MEMORY_ALLOCATION:
            return "Memory allocation failed";
        case QGT_ERROR_DIMENSION_MISMATCH:
            return "Dimension mismatch";
        case QGT_ERROR_INVALID_STATE:
            return "Invalid state";
        case QGT_ERROR_VALIDATION_FAILED:
            return "Validation failed";
        case QGT_ERROR_HARDWARE_FAILURE:
            return "Hardware failure";
        case QGT_ERROR_NOT_IMPLEMENTED:
            return "Not implemented";
        case QGT_ERROR_OVERFLOW:
            return "Overflow";
        case QGT_ERROR_UNDERFLOW:
            return "Underflow";
        case QGT_ERROR_DIVISION_BY_ZERO:
            return "Division by zero";
        case QGT_ERROR_INVALID_OPERATION:
            return "Invalid operation";
        case QGT_ERROR_SYSTEM_FAILURE:
            return "System failure";
        case QGT_ERROR_TIMEOUT:
            return "Timeout";
        case QGT_ERROR_RESOURCE_EXHAUSTED:
            return "Resource exhausted";
        case QGT_ERROR_INTERNAL:
            return "Internal error";
        default:
            return "Unknown error";
    }
}

// Utility functions
bool validate_quantum_state(const ComplexFloat* state,
                          size_t dim,
                          float tolerance) {
    if (!state || dim == 0) return false;
    
    // Check normalization
    float norm = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        norm += state[i].real * state[i].real +
                state[i].imag * state[i].imag;
    }
    
    if (fabs(norm - 1.0) > tolerance) return false;
    
    return true;
}

bool check_gpu_compatibility(GPUContext* ctx,
                           size_t required_memory,
                           const QGTConfig* config) {
    if (!ctx || !config) return false;
    
    // Check if GPU is available
    if (!ctx->is_available) return false;
    
    // Check memory requirements
    size_t available_memory = 0;
    #ifdef __APPLE__
    available_memory = ctx->metal.device_memory;
    #else
    cudaMemGetInfo(&available_memory, NULL);
    #endif
    
    if (required_memory > available_memory) return false;
    
    return true;
}

QGTConfig qgt_default_config(void) {
    return default_config;
}

// Implementation of quantum geometric tensor operations
qgt_error_t compute_quantum_metric_gpu(GPUContext* ctx,
                                  const ComplexFloat* state,
                                  ComplexFloat* metric,
                                  size_t rows,
                                  size_t cols,
                                  const QGTConfig* config) {
    // Validate inputs
    if (!ctx || !state || !metric || !config) {
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    if (!validate_quantum_state(state, rows * cols, config->precision)) {
        return QGT_ERROR_INVALID_STATE;
    }
    
    // Check GPU compatibility
    size_t required_memory = rows * cols * sizeof(ComplexFloat) * 3;
    if (!check_gpu_compatibility(ctx, required_memory, config)) {
        return QGT_ERROR_HARDWARE_FAILURE;
    }
    
    // Allocate GPU memory
    void* d_state = NULL;
    void* d_metric = NULL;
    qgt_error_t err;

    // Allocate GPU memory
    err = ctx->malloc(&d_state, rows * cols * sizeof(ComplexFloat));
    if (err != QGT_SUCCESS) {
        return QGT_ERROR_MEMORY_ALLOCATION;
    }

    err = ctx->malloc(&d_metric, rows * cols * sizeof(ComplexFloat));
    if (err != QGT_SUCCESS) {
        ctx->free(d_state);
        return QGT_ERROR_MEMORY_ALLOCATION;
    }
    
    // Copy input to GPU
    err = ctx->memcpy_to_device(d_state, state,
                               rows * cols * sizeof(ComplexFloat));
    if (err != QGT_SUCCESS) {
        ctx->free(d_state);
        ctx->free(d_metric);
        return QGT_ERROR_HARDWARE_FAILURE;
    }
    
    // Configure grid and block dimensions
    size_t block_size = ctx->get_optimal_block_size();
    size_t grid_size = (rows * cols + block_size - 1) / block_size;
    
    // Execute kernel
    #ifdef __APPLE__
    err = ctx->metal.execute_metric(d_state, d_metric, rows, cols);
    #else
    err = ctx->cuda.execute_metric(d_state, d_metric, rows, cols);
    #endif
    
    if (err != QGT_SUCCESS) {
        ctx->free(d_state);
        ctx->free(d_metric);
        return QGT_ERROR_HARDWARE_FAILURE;
    }
    
    // Copy result back to host
    err = ctx->memcpy_from_device(metric, d_metric,
                                 rows * cols * sizeof(ComplexFloat));
    if (err != QGT_SUCCESS) {
        ctx->free(d_state);
        ctx->free(d_metric);
        return QGT_ERROR_HARDWARE_FAILURE;
    }
    
    // Cleanup
    ctx->free(d_state);
    ctx->free(d_metric);
    
    return QGT_SUCCESS;
}

qgt_error_t compute_quantum_connection_gpu(GPUContext* ctx,
                                      const ComplexFloat* state,
                                      ComplexFloat* connection,
                                      size_t rows,
                                      size_t cols,
                                      const QGTConfig* config) {
    // Validate inputs
    if (!ctx || !state || !connection || !config) {
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    if (!validate_quantum_state(state, rows * cols, config->precision)) {
        return QGT_ERROR_INVALID_STATE;
    }
    
    // Check GPU compatibility
    size_t required_memory = rows * cols * sizeof(ComplexFloat) * 2;
    if (!check_gpu_compatibility(ctx, required_memory, config)) {
        return QGT_ERROR_HARDWARE_FAILURE;
    }
    
    // Allocate GPU memory
    void* d_state = NULL;
    void* d_connection = NULL;
    qgt_error_t err;

    // Allocate GPU memory
    err = ctx->malloc(&d_state, rows * cols * sizeof(ComplexFloat));
    if (err != QGT_SUCCESS) {
        return QGT_ERROR_MEMORY_ALLOCATION;
    }

    err = ctx->malloc(&d_connection, rows * cols * sizeof(ComplexFloat));
    if (err != QGT_SUCCESS) {
        ctx->free(d_state);
        return QGT_ERROR_MEMORY_ALLOCATION;
    }
    
    // Copy input to GPU
    err = ctx->memcpy_to_device(d_state, state,
                               rows * cols * sizeof(ComplexFloat));
    if (err != QGT_SUCCESS) {
        ctx->free(d_state);
        ctx->free(d_connection);
        return QGT_ERROR_HARDWARE_FAILURE;
    }
    
    // Configure grid and block dimensions
    size_t block_size = ctx->get_optimal_block_size();
    size_t grid_size = (rows * cols + block_size - 1) / block_size;
    
    // Execute kernel
    #ifdef __APPLE__
    err = ctx->metal.execute_connection(d_state, d_connection, rows, cols);
    #else
    err = ctx->cuda.execute_connection(d_state, d_connection, rows, cols);
    #endif
    
    if (err != QGT_SUCCESS) {
        ctx->free(d_state);
        ctx->free(d_connection);
        return QGT_ERROR_HARDWARE_FAILURE;
    }
    
    // Copy result back to host
    err = ctx->memcpy_from_device(connection, d_connection,
                                 rows * cols * sizeof(ComplexFloat));
    if (err != QGT_SUCCESS) {
        ctx->free(d_state);
        ctx->free(d_connection);
        return QGT_ERROR_HARDWARE_FAILURE;
    }
    
    // Cleanup
    ctx->free(d_state);
    ctx->free(d_connection);
    
    return QGT_SUCCESS;
}

qgt_error_t compute_quantum_curvature_gpu(GPUContext* ctx,
                                     const ComplexFloat* state,
                                     ComplexFloat* curvature,
                                     size_t rows,
                                     size_t cols,
                                     const QGTConfig* config) {
    // Validate inputs
    if (!ctx || !state || !curvature || !config) {
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    if (!validate_quantum_state(state, rows * cols, config->precision)) {
        return QGT_ERROR_INVALID_STATE;
    }
    
    // Check GPU compatibility
    size_t required_memory = rows * cols * sizeof(ComplexFloat) * 2;
    if (!check_gpu_compatibility(ctx, required_memory, config)) {
        return QGT_ERROR_HARDWARE_FAILURE;
    }
    
    // Allocate GPU memory
    void* d_state = NULL;
    void* d_curvature = NULL;
    qgt_error_t err;

    // Allocate GPU memory
    err = ctx->malloc(&d_state, rows * cols * sizeof(ComplexFloat));
    if (err != QGT_SUCCESS) {
        return QGT_ERROR_MEMORY_ALLOCATION;
    }

    err = ctx->malloc(&d_curvature, rows * cols * sizeof(ComplexFloat));
    if (err != QGT_SUCCESS) {
        ctx->free(d_state);
        return QGT_ERROR_MEMORY_ALLOCATION;
    }
    
    // Copy input to GPU
    err = ctx->memcpy_to_device(d_state, state,
                               rows * cols * sizeof(ComplexFloat));
    if (err != QGT_SUCCESS) {
        ctx->free(d_state);
        ctx->free(d_curvature);
        return QGT_ERROR_HARDWARE_FAILURE;
    }
    
    // Configure grid and block dimensions
    size_t block_size = ctx->get_optimal_block_size();
    size_t grid_size = (rows * cols + block_size - 1) / block_size;
    
    // Execute kernel
    #ifdef __APPLE__
    err = ctx->metal.execute_curvature(d_state, d_curvature, rows, cols);
    #else
    err = ctx->cuda.execute_curvature(d_state, d_curvature, rows, cols);
    #endif
    
    if (err != QGT_SUCCESS) {
        ctx->free(d_state);
        ctx->free(d_curvature);
        return QGT_ERROR_HARDWARE_FAILURE;
    }
    
    // Copy result back to host
    err = ctx->memcpy_from_device(curvature, d_curvature,
                                 rows * cols * sizeof(ComplexFloat));
    if (err != QGT_SUCCESS) {
        ctx->free(d_state);
        ctx->free(d_curvature);
        return QGT_ERROR_HARDWARE_FAILURE;
    }
    
    // Cleanup
    ctx->free(d_state);
    ctx->free(d_curvature);
    
    return QGT_SUCCESS;
}
