#include "quantum_geometric/hardware/quantum_field_gpu.h"
#include <string.h>

// Last error state
static struct {
    int code;
    char message[256];
} last_error = {
    .code = GPU_SUCCESS,
    .message = ""
};

// Error strings
static const char* error_strings[] = {
    [GPU_SUCCESS] = "Success",
    [GPU_ERROR_NO_DEVICE] = "No GPU device available",
    [GPU_ERROR_INIT_FAILED] = "GPU initialization failed",
    [GPU_ERROR_INVALID_ARG] = "Invalid argument",
    [GPU_ERROR_OUT_OF_MEM] = "Out of GPU memory",
    [GPU_ERROR_LAUNCH_FAILED] = "Kernel launch failed"
};

// Set last error
static void set_last_error(int code, const char* msg) {
    last_error.code = code;
    strncpy(last_error.message, msg, sizeof(last_error.message) - 1);
    last_error.message[sizeof(last_error.message) - 1] = '\0';
}

// Error handling functions
const char* gpu_error_string(int error) {
    if (error >= 0 || error < -5) {
        return "Unknown error";
    }
    return error_strings[-error];
}

int get_last_gpu_error(void) {
    return last_error.code;
}

void clear_gpu_error(void) {
    last_error.code = GPU_SUCCESS;
    last_error.message[0] = '\0';
}

// Performance monitoring functions
size_t get_gpu_memory_usage(void) {
    GpuBackendType backend = get_gpu_backend_type();
    
    switch (backend) {
        case GPU_BACKEND_CUDA: {
            #ifdef HAVE_CUDA
            size_t free, total;
            if (cudaMemGetInfo(&free, &total) == cudaSuccess) {
                return total - free;
            }
            #endif
            break;
        }
        
        case GPU_BACKEND_METAL: {
            #ifdef HAVE_METAL
            id<MTLDevice> device = (__bridge id<MTLDevice>)gpu_backend.context;
            if (device) {
                return device.currentAllocatedSize;
            }
            #endif
            break;
        }
        
        default:
            break;
    }
    
    return 0;
}

int get_gpu_utilization(void) {
    GpuBackendType backend = get_gpu_backend_type();
    
    switch (backend) {
        case GPU_BACKEND_CUDA: {
            #ifdef HAVE_CUDA
            nvmlDevice_t device;
            if (nvmlDeviceGetHandleByIndex(0, &device) == NVML_SUCCESS) {
                nvmlUtilization_t utilization;
                if (nvmlDeviceGetUtilizationRates(device, &utilization) == NVML_SUCCESS) {
                    return utilization.gpu;
                }
            }
            #endif
            break;
        }
        
        case GPU_BACKEND_METAL: {
            // Metal doesn't provide direct GPU utilization info
            // Could estimate based on command buffer completion times
            return -1;
        }
        
        default:
            break;
    }
    
    return -1;
}

int get_gpu_temperature(void) {
    GpuBackendType backend = get_gpu_backend_type();
    
    switch (backend) {
        case GPU_BACKEND_CUDA: {
            #ifdef HAVE_CUDA
            nvmlDevice_t device;
            if (nvmlDeviceGetHandleByIndex(0, &device) == NVML_SUCCESS) {
                unsigned int temp;
                if (nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp) == NVML_SUCCESS) {
                    return (int)temp;
                }
            }
            #endif
            break;
        }
        
        case GPU_BACKEND_METAL: {
            // Metal doesn't provide temperature info
            return -1;
        }
        
        default:
            break;
    }
    
    return -1;
}

float get_gpu_power_usage(void) {
    GpuBackendType backend = get_gpu_backend_type();
    
    switch (backend) {
        case GPU_BACKEND_CUDA: {
            #ifdef HAVE_CUDA
            nvmlDevice_t device;
            if (nvmlDeviceGetHandleByIndex(0, &device) == NVML_SUCCESS) {
                unsigned int power;
                if (nvmlDeviceGetPowerUsage(device, &power) == NVML_SUCCESS) {
                    return power / 1000.0f; // Convert from milliwatts to watts
                }
            }
            #endif
            break;
        }
        
        case GPU_BACKEND_METAL: {
            // Metal doesn't provide power usage info
            return -1.0f;
        }
        
        default:
            break;
    }
    
    return -1.0f;
}

// Internal error reporting functions
void report_cuda_error(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        char msg[256];
        snprintf(msg, sizeof(msg), "%s failed: %s", operation, cudaGetErrorString(error));
        set_last_error(GPU_ERROR_LAUNCH_FAILED, msg);
    }
}

void report_metal_error(NSError* error, const char* operation) {
    if (error) {
        char msg[256];
        snprintf(msg, sizeof(msg), "%s failed: %s", operation, [[error localizedDescription] UTF8String]);
        set_last_error(GPU_ERROR_LAUNCH_FAILED, msg);
    }
}
