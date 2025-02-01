#include "quantum_geometric/physics/quantum_field_operations.h"
#include <stdbool.h>

// GPU backend types
typedef enum {
    GPU_BACKEND_NONE,
    GPU_BACKEND_CUDA,
    GPU_BACKEND_METAL
} GpuBackendType;

// GPU backend state
static struct {
    GpuBackendType type;
    bool initialized;
    void* context;
} gpu_backend = {
    .type = GPU_BACKEND_NONE,
    .initialized = false,
    .context = NULL
};

// External CUDA functions
extern int apply_rotation_cuda(
    QuantumField* field,
    size_t qubit,
    double theta,
    double phi);

extern double calculate_field_energy_cuda(
    const QuantumField* field);

extern int calculate_field_equations_cuda(
    const QuantumField* field,
    Tensor* equations);

// External Metal functions
extern int apply_rotation_metal(
    QuantumField* field,
    size_t qubit,
    double theta,
    double phi);

extern double calculate_field_energy_metal(
    const QuantumField* field);

extern int calculate_field_equations_metal(
    const QuantumField* field,
    Tensor* equations);

// Initialize GPU backend
static bool init_gpu_backend() {
    if (gpu_backend.initialized) {
        return true;
    }
    
    // Try CUDA first
    #ifdef HAVE_CUDA
    cudaError_t error = cudaGetDevice(NULL);
    if (error == cudaSuccess) {
        gpu_backend.type = GPU_BACKEND_CUDA;
        gpu_backend.initialized = true;
        return true;
    }
    #endif
    
    // Try Metal next
    #ifdef HAVE_METAL
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device) {
        gpu_backend.type = GPU_BACKEND_METAL;
        gpu_backend.context = (__bridge_retained void*)device;
        gpu_backend.initialized = true;
        return true;
    }
    #endif
    
    // No GPU available
    gpu_backend.type = GPU_BACKEND_NONE;
    gpu_backend.initialized = true;
    return false;
}

// Apply rotation using GPU
int apply_rotation_gpu(
    QuantumField* field,
    size_t qubit,
    double theta,
    double phi) {
    
    if (!init_gpu_backend()) {
        return -1;
    }
    
    switch (gpu_backend.type) {
        case GPU_BACKEND_CUDA:
            return apply_rotation_cuda(field, qubit, theta, phi);
            
        case GPU_BACKEND_METAL:
            return apply_rotation_metal(field, qubit, theta, phi);
            
        default:
            return -1;
    }
}

// Calculate field energy using GPU
double calculate_field_energy_gpu(const QuantumField* field) {
    if (!init_gpu_backend()) {
        return 0.0;
    }
    
    switch (gpu_backend.type) {
        case GPU_BACKEND_CUDA:
            return calculate_field_energy_cuda(field);
            
        case GPU_BACKEND_METAL:
            return calculate_field_energy_metal(field);
            
        default:
            return 0.0;
    }
}

// Calculate field equations using GPU
int calculate_field_equations_gpu(
    const QuantumField* field,
    Tensor* equations) {
    
    if (!init_gpu_backend()) {
        return -1;
    }
    
    switch (gpu_backend.type) {
        case GPU_BACKEND_CUDA:
            return calculate_field_equations_cuda(field, equations);
            
        case GPU_BACKEND_METAL:
            return calculate_field_equations_metal(field, equations);
            
        default:
            return -1;
    }
}

// Get GPU backend type
GpuBackendType get_gpu_backend_type() {
    if (!init_gpu_backend()) {
        return GPU_BACKEND_NONE;
    }
    return gpu_backend.type;
}

// Check if GPU acceleration is available
bool has_gpu_acceleration() {
    return init_gpu_backend() && gpu_backend.type != GPU_BACKEND_NONE;
}

// Get GPU device name
const char* get_gpu_device_name() {
    if (!init_gpu_backend()) {
        return "No GPU";
    }
    
    switch (gpu_backend.type) {
        case GPU_BACKEND_CUDA: {
            #ifdef HAVE_CUDA
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            return prop.name;
            #else
            return "CUDA Device";
            #endif
        }
        
        case GPU_BACKEND_METAL: {
            #ifdef HAVE_METAL
            id<MTLDevice> device = (__bridge id<MTLDevice>)gpu_backend.context;
            return [[device name] UTF8String];
            #else
            return "Metal Device";
            #endif
        }
        
        default:
            return "No GPU";
    }
}

// Clean up GPU backend
void cleanup_gpu_backend() {
    if (!gpu_backend.initialized) {
        return;
    }
    
    switch (gpu_backend.type) {
        case GPU_BACKEND_CUDA:
            #ifdef HAVE_CUDA
            cudaDeviceReset();
            #endif
            break;
            
        case GPU_BACKEND_METAL:
            #ifdef HAVE_METAL
            if (gpu_backend.context) {
                CFRelease(gpu_backend.context);
            }
            #endif
            break;
            
        default:
            break;
    }
    
    gpu_backend.type = GPU_BACKEND_NONE;
    gpu_backend.initialized = false;
    gpu_backend.context = NULL;
}
