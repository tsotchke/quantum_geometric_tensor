#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_geometric_constants.h"
#include "quantum_geometric/hardware/quantum_hardware_types.h"
#include "quantum_geometric/core/quantum_geometric_operations.h"
#include "quantum_geometric/hardware/quantum_geometric_gpu.h"
#include <string.h>
#include <stdlib.h>

#ifdef ENABLE_METAL
#include "quantum_geometric/hardware/metal/quantum_geometric_metal.h"
#endif

#ifdef ENABLE_CUDA
#include "quantum_geometric/hardware/quantum_geometric_cuda.h"
#endif

// Internal context structure
struct GPUContext {
    GPUBackendType backend_type;
    void* backend_context;  // Metal or CUDA context
    GPUDeviceInfo device_info;
    char last_error[256];
    GPUPerformanceMetrics metrics;
};

// Initialize GPU system
int gpu_initialize(void) {
#ifdef __APPLE__
    #ifdef __arm64__
    // On Apple Silicon, prioritize Metal
    #ifdef ENABLE_METAL
        return metal_initialize();
    #endif
    #endif
#endif

    // Fall back to CUDA if available
#ifdef ENABLE_CUDA
    return cuda_initialize();
#endif

    return QGT_ERROR_HARDWARE_FAILURE;
}

// Cleanup GPU system
void gpu_cleanup(void) {
#ifdef ENABLE_METAL
    metal_cleanup();
#endif

#ifdef ENABLE_CUDA
    cuda_cleanup();
#endif
}

// Get available GPU devices
int gpu_get_devices(GPUDeviceInfo* devices, int max_devices) {
    int num_devices = 0;

#ifdef __APPLE__
    #ifdef __arm64__
    // On Apple Silicon, prioritize Metal devices
    #ifdef ENABLE_METAL
        num_devices = metal_get_devices(devices, max_devices);
        if (num_devices > 0) {
            for (int i = 0; i < num_devices; i++) {
                devices[i].backend_type = GPU_BACKEND_METAL;
                devices[i].supports_amx = true;  // M1/M2 supports AMX
            }
            return num_devices;
        }
    #endif
    #endif
#endif

    // Fall back to CUDA if available
#ifdef ENABLE_CUDA
    num_devices = cuda_get_devices(devices, max_devices);
    if (num_devices > 0) {
        for (int i = 0; i < num_devices; i++) {
            devices[i].backend_type = GPU_BACKEND_CUDA;
        }
        return num_devices;
    }
#endif

    return 0;
}

// Create GPU context
GPUContext* gpu_create_context(int device_index) {
    GPUContext* context = (GPUContext*)malloc(sizeof(GPUContext));
    if (!context) return NULL;

    memset(context, 0, sizeof(GPUContext));

#ifdef __APPLE__
    #ifdef __arm64__
    // On Apple Silicon, prioritize Metal
    #ifdef ENABLE_METAL
        context->backend_context = metal_create_context(device_index);
        if (context->backend_context) {
            context->backend_type = GPU_BACKEND_METAL;
            metal_get_device_info(context->backend_context, &context->device_info);
            context->device_info.backend_type = GPU_BACKEND_METAL;
            context->device_info.supports_amx = true;
            return context;
        }
    #endif
    #endif
#endif

    // Fall back to CUDA if available
#ifdef ENABLE_CUDA
    context->backend_context = cuda_create_context(device_index);
    if (context->backend_context) {
        context->backend_type = GPU_BACKEND_CUDA;
        cuda_get_device_info(context->backend_context, &context->device_info);
        context->device_info.backend_type = GPU_BACKEND_CUDA;
        return context;
    }
#endif

    free(context);
    return NULL;
}

// Destroy GPU context
void gpu_destroy_context(GPUContext* context) {
    if (!context) return;

    switch (context->backend_type) {
#ifdef ENABLE_METAL
        case GPU_BACKEND_METAL:
            metal_destroy_context(context->backend_context);
            break;
#endif

#ifdef ENABLE_CUDA
        case GPU_BACKEND_CUDA:
            cuda_destroy_context(context->backend_context);
            break;
#endif

        default:
            break;
    }

    free(context);
}

// Memory management
void* gpu_allocate(GPUContext* context, size_t size) {
    if (!context) return NULL;

    switch (context->backend_type) {
#ifdef ENABLE_METAL
        case GPU_BACKEND_METAL:
            return metal_allocate(context->backend_context, size);
#endif

#ifdef ENABLE_CUDA
        case GPU_BACKEND_CUDA:
            return cuda_allocate(context->backend_context, size);
#endif

        default:
            return NULL;
    }
}

void gpu_free(GPUContext* context, void* ptr) {
    if (!context || !ptr) return;

    switch (context->backend_type) {
#ifdef ENABLE_METAL
        case GPU_BACKEND_METAL:
            metal_free(context->backend_context, ptr);
            break;
#endif

#ifdef ENABLE_CUDA
        case GPU_BACKEND_CUDA:
            cuda_free(context->backend_context, ptr);
            break;
#endif

        default:
            break;
    }
}

int gpu_memcpy_to_device(GPUContext* context, void* dst, const void* src, size_t size) {
    if (!context) return QGT_ERROR_INVALID_STATE;
    if (!dst || !src) return QGT_ERROR_INVALID_PARAMETER;
    if (size == 0) return QGT_ERROR_INVALID_PARAMETER;

    switch (context->backend_type) {
#ifdef ENABLE_METAL
        case GPU_BACKEND_METAL:
            return metal_memcpy_to_device(context->backend_context, dst, src, size);
#endif

#ifdef ENABLE_CUDA
        case GPU_BACKEND_CUDA:
            return cuda_memcpy_to_device(context->backend_context, dst, src, size);
#endif

        default:
            return QGT_ERROR_HARDWARE_FAILURE;
    }
}

int gpu_memcpy_from_device(GPUContext* context, void* dst, const void* src, size_t size) {
    if (!context) return QGT_ERROR_INVALID_STATE;
    if (!dst || !src) return QGT_ERROR_INVALID_PARAMETER;
    if (size == 0) return QGT_ERROR_INVALID_PARAMETER;

    switch (context->backend_type) {
#ifdef ENABLE_METAL
        case GPU_BACKEND_METAL:
            return metal_memcpy_from_device(context->backend_context, dst, src, size);
#endif

#ifdef ENABLE_CUDA
        case GPU_BACKEND_CUDA:
            return cuda_memcpy_from_device(context->backend_context, dst, src, size);
#endif

        default:
            return QGT_ERROR_HARDWARE_FAILURE;
    }
}

// Quantum operations
int gpu_quantum_tensor_multiply(
    GPUContext* context,
    const ComplexFloat* a,
    const ComplexFloat* b,
    ComplexFloat* c,
    int m, int n, int k
) {
    if (!context) return QGT_ERROR_INVALID_STATE;

    switch (context->backend_type) {
#ifdef ENABLE_METAL
        case GPU_BACKEND_METAL:
            return metal_quantum_tensor_multiply(context->backend_context, a, b, c, m, n, k);
#endif

#ifdef ENABLE_CUDA
        case GPU_BACKEND_CUDA:
            return cuda_quantum_tensor_multiply(context->backend_context, a, b, c, m, n, k);
#endif

        default:
            return QGT_ERROR_HARDWARE_FAILURE;
    }
}

int gpu_quantum_geometric_transform(
    GPUContext* context,
    const ComplexFloat* input,
    ComplexFloat* output,
    const QuantumGeometricParams* params,
    size_t size
) {
    if (!context) return QGT_ERROR_INVALID_STATE;

    switch (context->backend_type) {
#ifdef ENABLE_METAL
        case GPU_BACKEND_METAL:
            return metal_quantum_geometric_transform(context->backend_context, input, output, params, size);
#endif

#ifdef ENABLE_CUDA
        case GPU_BACKEND_CUDA:
            return cuda_quantum_geometric_transform(context->backend_context, input, output, params, size);
#endif

        default:
            return QGT_ERROR_HARDWARE_FAILURE;
    }
}

int gpu_quantum_attention(
    GPUContext* context,
    const ComplexFloat* queries,
    const ComplexFloat* keys,
    const ComplexFloat* values,
    ComplexFloat* output,
    int batch_size,
    int num_heads,
    int seq_length,
    int head_dim
) {
    if (!context) return QGT_ERROR_INVALID_STATE;
    if (!queries || !keys || !values || !output) return QGT_ERROR_INVALID_PARAMETER;
    if (batch_size <= 0 || num_heads <= 0 || seq_length <= 0 || head_dim <= 0) 
        return QGT_ERROR_INVALID_PARAMETER;

    switch (context->backend_type) {
#ifdef ENABLE_METAL
        case GPU_BACKEND_METAL:
            return metal_quantum_attention(context->backend_context, queries, keys, values, 
                                        output, batch_size, num_heads, seq_length, head_dim);
#endif

#ifdef ENABLE_CUDA
        case GPU_BACKEND_CUDA:
            return cuda_quantum_attention(context->backend_context, queries, keys, values,
                                       output, batch_size, num_heads, seq_length, head_dim);
#endif

        default:
            return QGT_ERROR_HARDWARE_FAILURE;
    }
}

int gpu_batch_quantum_operations(
    GPUContext* context,
    const ComplexFloat* states,
    ComplexFloat* results,
    const QuantumOperation* operations,
    int num_states,
    int num_operations
) {
    if (!context) return QGT_ERROR_INVALID_STATE;
    if (!states || !results || !operations) return QGT_ERROR_INVALID_PARAMETER;
    if (num_states <= 0 || num_operations <= 0) return QGT_ERROR_INVALID_PARAMETER;

    switch (context->backend_type) {
#ifdef ENABLE_METAL
        case GPU_BACKEND_METAL:
            return metal_batch_quantum_operations(context->backend_context, states, results,
                                               operations, num_states, num_operations);
#endif

#ifdef ENABLE_CUDA
        case GPU_BACKEND_CUDA:
            return cuda_batch_quantum_operations(context->backend_context, states, results,
                                              operations, num_states, num_operations);
#endif

        default:
            return QGT_ERROR_HARDWARE_FAILURE;
    }
}

// Metal-specific optimizations
#ifdef ENABLE_METAL
int gpu_enable_amx(GPUContext* context) {
    if (!context || context->backend_type != GPU_BACKEND_METAL)
        return QGT_ERROR_INVALID_STATE;
    
    return metal_enable_amx(context->backend_context);
}

int gpu_set_metal_compute_units(GPUContext* context, int num_units) {
    if (!context || context->backend_type != GPU_BACKEND_METAL)
        return QGT_ERROR_INVALID_STATE;
    
    return metal_set_compute_units(context->backend_context, num_units);
}

int gpu_optimize_for_m1(GPUContext* context) {
    if (!context || context->backend_type != GPU_BACKEND_METAL)
        return QGT_ERROR_INVALID_STATE;
    
    return metal_optimize_for_m1(context->backend_context);
}
#endif

// Optional CUDA-specific operations
#ifdef ENABLE_CUDA
int gpu_enable_tensor_cores(GPUContext* context) {
    if (!context || context->backend_type != GPU_BACKEND_CUDA)
        return QGT_ERROR_INVALID_STATE;
    
    return cuda_enable_tensor_cores(context->backend_context);
}

int gpu_set_cuda_stream(GPUContext* context, void* stream) {
    if (!context || context->backend_type != GPU_BACKEND_CUDA)
        return QGT_ERROR_INVALID_STATE;
    
    return cuda_set_stream(context->backend_context, stream);
}
#endif

// Error handling
const char* gpu_get_last_error(void) {
    static char error_buffer[256];
    
#ifdef ENABLE_METAL
    const char* metal_error = metal_get_last_error();
    if (metal_error) {
        strncpy(error_buffer, metal_error, sizeof(error_buffer) - 1);
        return error_buffer;
    }
#endif

#ifdef ENABLE_CUDA
    const char* cuda_error = cuda_get_last_error();
    if (cuda_error) {
        strncpy(error_buffer, cuda_error, sizeof(error_buffer) - 1);
        return error_buffer;
    }
#endif

    return "No error";
}

void gpu_clear_error(void) {
#ifdef ENABLE_METAL
    metal_clear_error();
#endif

#ifdef ENABLE_CUDA
    cuda_clear_error();
#endif
}

// Performance monitoring
int gpu_get_performance_metrics(GPUContext* context, GPUPerformanceMetrics* metrics) {
    if (!context || !metrics) return QGT_ERROR_INVALID_STATE;

    switch (context->backend_type) {
#ifdef ENABLE_METAL
        case GPU_BACKEND_METAL:
            return metal_get_performance_metrics(context->backend_context, metrics);
#endif

#ifdef ENABLE_CUDA
        case GPU_BACKEND_CUDA:
            return cuda_get_performance_metrics(context->backend_context, metrics);
#endif

        default:
            return QGT_ERROR_HARDWARE_FAILURE;
    }
}
