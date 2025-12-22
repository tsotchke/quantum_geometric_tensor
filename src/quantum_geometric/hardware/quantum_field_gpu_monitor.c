/**
 * @file quantum_field_gpu_monitor.c
 * @brief GPU monitoring and error handling implementation
 *
 * Provides error tracking, performance monitoring, and diagnostic functions
 * for GPU-accelerated quantum field computations.
 */

#include "quantum_geometric/hardware/quantum_field_gpu.h"
#include <string.h>
#include <stdio.h>

// ============================================================================
// Global GPU State Tracking
// ============================================================================

// Global GPU context reference (weak link to avoid circular dependencies)
static GPUContext* g_active_gpu_context = NULL;
static GpuBackendType g_active_backend = GPU_BACKEND_NONE;

// Last error state
static struct {
    int code;
    char message[256];
} last_error = {
    .code = GPU_SUCCESS,
    .message = ""
};

// Error strings (indexed by absolute value of error code)
static const char* error_strings[] = {
    "Success",                    // 0: GPU_SUCCESS
    "No GPU device available",    // 1: GPU_ERROR_NO_DEVICE
    "GPU initialization failed",  // 2: GPU_ERROR_INIT_FAILED
    "Invalid argument",           // 3: GPU_ERROR_INVALID_ARG
    "Out of GPU memory",          // 4: GPU_ERROR_OUT_OF_MEM
    "Kernel launch failed"        // 5: GPU_ERROR_LAUNCH_FAILED
};

#define NUM_ERROR_STRINGS (sizeof(error_strings) / sizeof(error_strings[0]))

// ============================================================================
// Internal Helpers
// ============================================================================

static void set_last_error(int code, const char* msg) {
    last_error.code = code;
    if (msg) {
        strncpy(last_error.message, msg, sizeof(last_error.message) - 1);
        last_error.message[sizeof(last_error.message) - 1] = '\0';
    } else {
        last_error.message[0] = '\0';
    }
}

// ============================================================================
// GPU State Management
// ============================================================================

GpuBackendType get_gpu_backend_type(void) {
    // If we have an active context, use its backend type
    if (g_active_gpu_context && g_active_gpu_context->is_initialized) {
        return (GpuBackendType)g_active_gpu_context->backend_type;
    }

    // Return cached backend type
    return g_active_backend;
}

bool is_gpu_available(void) {
    return g_active_gpu_context != NULL &&
           g_active_gpu_context->is_initialized &&
           g_active_backend != GPU_BACKEND_NONE;
}

/**
 * @brief Set the active GPU context for monitoring
 *
 * Called by GPU initialization code to register the active context.
 *
 * @param context The GPU context to monitor
 */
void set_gpu_monitor_context(GPUContext* context) {
    g_active_gpu_context = context;
    if (context && context->is_initialized) {
        g_active_backend = (GpuBackendType)context->backend_type;
    } else {
        g_active_backend = GPU_BACKEND_NONE;
    }
}

// ============================================================================
// Error Handling Functions
// ============================================================================

const char* gpu_error_string(int error) {
    // Convert error code to index (error codes are negative, index by absolute value)
    int index;
    if (error == GPU_SUCCESS) {
        index = 0;
    } else if (error < 0 && (size_t)(-error) < NUM_ERROR_STRINGS) {
        index = -error;
    } else {
        return "Unknown error";
    }
    return error_strings[index];
}

int get_last_gpu_error(void) {
    return last_error.code;
}

void clear_gpu_error(void) {
    last_error.code = GPU_SUCCESS;
    last_error.message[0] = '\0';
}

const char* get_last_gpu_error_message(void) {
    return last_error.message;
}

// ============================================================================
// Performance Monitoring Functions
// ============================================================================

size_t get_gpu_memory_usage(void) {
    GpuBackendType backend = get_gpu_backend_type();

    switch (backend) {
        case GPU_BACKEND_CUDA: {
#ifdef HAVE_CUDA
            size_t free_mem, total_mem;
            if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
                return total_mem - free_mem;
            }
#endif
            break;
        }

        case GPU_BACKEND_METAL: {
#if defined(HAVE_METAL) && defined(__OBJC__)
            if (g_active_gpu_context && g_active_gpu_context->device_handle) {
                id<MTLDevice> device = (__bridge id<MTLDevice>)g_active_gpu_context->device_handle;
                if (device) {
                    return device.currentAllocatedSize;
                }
            }
#endif
            // For non-Objective-C builds, return tracked allocation
            if (g_active_gpu_context) {
                return g_active_gpu_context->allocated_memory;
            }
            break;
        }

        case GPU_BACKEND_NONE:
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
                    return (int)utilization.gpu;
                }
            }
#endif
            break;
        }

        case GPU_BACKEND_METAL:
            // Metal doesn't provide direct GPU utilization info
            // Could estimate based on command buffer completion times
            return -1;

        case GPU_BACKEND_NONE:
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

        case GPU_BACKEND_METAL:
            // Metal doesn't provide temperature info
            return -1;

        case GPU_BACKEND_NONE:
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

        case GPU_BACKEND_METAL:
            // Metal doesn't provide power usage info
            return -1.0f;

        case GPU_BACKEND_NONE:
        default:
            break;
    }

    return -1.0f;
}

// ============================================================================
// Extended Performance Metrics
// ============================================================================

/**
 * @brief Get comprehensive GPU statistics
 *
 * @param memory_used Output: current memory usage in bytes
 * @param memory_total Output: total GPU memory in bytes
 * @param utilization Output: GPU utilization percentage (0-100)
 * @param temperature Output: GPU temperature in Celsius
 * @param power_watts Output: Power consumption in watts
 * @return true if at least some metrics were retrieved
 */
bool get_gpu_stats(size_t* memory_used, size_t* memory_total,
                   int* utilization, int* temperature, float* power_watts) {
    bool got_any = false;
    GpuBackendType backend = get_gpu_backend_type();

    // Initialize outputs
    if (memory_used) *memory_used = 0;
    if (memory_total) *memory_total = 0;
    if (utilization) *utilization = -1;
    if (temperature) *temperature = -1;
    if (power_watts) *power_watts = -1.0f;

    if (backend == GPU_BACKEND_NONE) {
        return false;
    }

#ifdef HAVE_CUDA
    if (backend == GPU_BACKEND_CUDA) {
        // Memory info
        size_t free_mem, total_mem;
        if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
            if (memory_used) *memory_used = total_mem - free_mem;
            if (memory_total) *memory_total = total_mem;
            got_any = true;
        }

        // NVML metrics
        nvmlDevice_t device;
        if (nvmlDeviceGetHandleByIndex(0, &device) == NVML_SUCCESS) {
            nvmlUtilization_t util;
            if (utilization && nvmlDeviceGetUtilizationRates(device, &util) == NVML_SUCCESS) {
                *utilization = (int)util.gpu;
                got_any = true;
            }

            unsigned int temp;
            if (temperature && nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp) == NVML_SUCCESS) {
                *temperature = (int)temp;
                got_any = true;
            }

            unsigned int power;
            if (power_watts && nvmlDeviceGetPowerUsage(device, &power) == NVML_SUCCESS) {
                *power_watts = power / 1000.0f;
                got_any = true;
            }
        }
    }
#endif

    if (backend == GPU_BACKEND_METAL && g_active_gpu_context) {
        if (memory_used) {
            *memory_used = g_active_gpu_context->allocated_memory;
            got_any = true;
        }
        if (memory_total) {
            *memory_total = g_active_gpu_context->max_memory;
            got_any = true;
        }
    }

    return got_any;
}

// ============================================================================
// Internal Error Reporting (for GPU backend implementations)
// ============================================================================

#ifdef HAVE_CUDA
void report_cuda_error(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        char msg[256];
        snprintf(msg, sizeof(msg), "%s failed: %s",
                 operation ? operation : "CUDA operation",
                 cudaGetErrorString(error));
        set_last_error(GPU_ERROR_LAUNCH_FAILED, msg);
    }
}
#endif

#if defined(HAVE_METAL) && defined(__OBJC__)
void report_metal_error(NSError* error, const char* operation) {
    if (error) {
        char msg[256];
        snprintf(msg, sizeof(msg), "%s failed: %s",
                 operation ? operation : "Metal operation",
                 [[error localizedDescription] UTF8String]);
        set_last_error(GPU_ERROR_LAUNCH_FAILED, msg);
    }
}
#endif

// ============================================================================
// Generic Error Reporting (for non-CUDA/Metal code)
// ============================================================================

void report_gpu_error(int error_code, const char* operation, const char* details) {
    char msg[256];
    if (details && details[0]) {
        snprintf(msg, sizeof(msg), "%s failed: %s",
                 operation ? operation : "GPU operation", details);
    } else {
        snprintf(msg, sizeof(msg), "%s failed",
                 operation ? operation : "GPU operation");
    }
    set_last_error(error_code, msg);
}
