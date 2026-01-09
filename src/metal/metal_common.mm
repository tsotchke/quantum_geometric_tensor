#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "quantum_geometric/hardware/metal/metal_common.h"
#include <pthread.h>
#include <string>
#include <unordered_map>
#include <vector>

// ===========================================================================
// Internal State
// ===========================================================================

namespace {

// Global Metal state
struct MetalCommonState {
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLLibrary> defaultLibrary;
    std::vector<id<MTLLibrary>> registeredLibraries;
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipelineCache;
    pthread_mutex_t mutex;
    bool initialized;
};

MetalCommonState g_state = {
    .device = nil,
    .commandQueue = nil,
    .defaultLibrary = nil,
    .registeredLibraries = {},
    .pipelineCache = {},
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .initialized = false
};

// Thread group size for compute operations
constexpr uint32_t THREAD_GROUP_SIZE = 256;

// Embedded shader source for common kernels
static const char* g_commonShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

// Stochastic sampling kernel
kernel void stochastic_sampling(
    device const float2* input_states [[buffer(0)]],
    device const float* probabilities [[buffer(1)]],
    device float2* output_states [[buffer(2)]],
    constant uint& num_states [[buffer(3)]],
    constant uint& num_samples [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_samples) return;

    // Use probability to select state
    float rand_val = fract(sin(float(gid) * 12.9898 + 78.233) * 43758.5453);
    float cumulative = 0.0f;
    uint selected_idx = 0;

    for (uint i = 0; i < num_states; i++) {
        cumulative += probabilities[i];
        if (rand_val <= cumulative) {
            selected_idx = i;
            break;
        }
    }

    output_states[gid] = input_states[selected_idx];
}

// Stochastic gradient kernel
kernel void stochastic_gradient(
    device const float2* states [[buffer(0)]],
    device const float2* gradients [[buffer(1)]],
    device float2* output [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant float& learning_rate [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= batch_size) return;

    float2 state = states[gid];
    float2 grad = gradients[gid];

    // Apply gradient update with learning rate
    output[gid] = float2(
        state.x - learning_rate * grad.x,
        state.y - learning_rate * grad.y
    );
}

// Importance sampling kernel
kernel void importance_sampling(
    device const float2* states [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device uint* indices [[buffer(2)]],
    device float* resampled_weights [[buffer(3)]],
    constant uint& num_particles [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_particles) return;

    // Systematic resampling
    float total_weight = 0.0f;
    for (uint i = 0; i < num_particles; i++) {
        total_weight += weights[i];
    }

    float step = total_weight / float(num_particles);
    float rand_start = fract(sin(float(gid) * 12.9898) * 43758.5453) * step;
    float target = rand_start + float(gid) * step;

    float cumulative = 0.0f;
    uint selected_idx = 0;

    for (uint i = 0; i < num_particles; i++) {
        cumulative += weights[i];
        if (target <= cumulative) {
            selected_idx = i;
            break;
        }
    }

    indices[gid] = selected_idx;
    resampled_weights[gid] = 1.0f / float(num_particles);
}
)";

// Find kernel in registered libraries
id<MTLFunction> findKernelFunction(const std::string& name) {
    // Try default library first
    if (g_state.defaultLibrary) {
        NSString* funcName = [NSString stringWithUTF8String:name.c_str()];
        id<MTLFunction> func = [g_state.defaultLibrary newFunctionWithName:funcName];
        if (func) return func;
    }

    // Try registered libraries
    for (id<MTLLibrary> lib : g_state.registeredLibraries) {
        NSString* funcName = [NSString stringWithUTF8String:name.c_str()];
        id<MTLFunction> func = [lib newFunctionWithName:funcName];
        if (func) return func;
    }

    return nil;
}

} // anonymous namespace

// ===========================================================================
// Public API Implementation
// ===========================================================================

extern "C" {

metal_error_t metal_common_initialize(void) {
    pthread_mutex_lock(&g_state.mutex);

    if (g_state.initialized) {
        pthread_mutex_unlock(&g_state.mutex);
        return METAL_SUCCESS;
    }

    @autoreleasepool {
        // Get default device
        g_state.device = MTLCreateSystemDefaultDevice();
        if (!g_state.device) {
            pthread_mutex_unlock(&g_state.mutex);
            return METAL_ERROR_DEVICE_NOT_FOUND;
        }

        // Create command queue
        g_state.commandQueue = [g_state.device newCommandQueue];
        if (!g_state.commandQueue) {
            g_state.device = nil;
            pthread_mutex_unlock(&g_state.mutex);
            return METAL_ERROR_INTERNAL;
        }

        // Try to load default library
        NSError* error = nil;
        g_state.defaultLibrary = [g_state.device newDefaultLibrary];
        // Default library may not exist, that's OK

        // Compile embedded common shaders
        NSString* source = [NSString stringWithUTF8String:g_commonShaderSource];
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.fastMathEnabled = YES;

        id<MTLLibrary> commonLib = [g_state.device newLibraryWithSource:source
                                                               options:options
                                                                 error:&error];
        if (commonLib) {
            g_state.registeredLibraries.push_back(commonLib);
        }

        g_state.initialized = true;
    }

    pthread_mutex_unlock(&g_state.mutex);
    return METAL_SUCCESS;
}

void metal_common_cleanup(void) {
    pthread_mutex_lock(&g_state.mutex);

    if (!g_state.initialized) {
        pthread_mutex_unlock(&g_state.mutex);
        return;
    }

    @autoreleasepool {
        g_state.pipelineCache.clear();
        g_state.registeredLibraries.clear();
        g_state.defaultLibrary = nil;
        g_state.commandQueue = nil;
        g_state.device = nil;
        g_state.initialized = false;
    }

    pthread_mutex_unlock(&g_state.mutex);
}

bool metal_common_is_available(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        return device != nil;
    }
}

metal_error_t metal_create_compute_pipeline(const char* kernel_name, void** pipeline) {
    if (!kernel_name || !pipeline) {
        return METAL_ERROR_INVALID_PARAMS;
    }

    pthread_mutex_lock(&g_state.mutex);

    if (!g_state.initialized) {
        // Auto-initialize if needed
        metal_error_t err = metal_common_initialize();
        if (err != METAL_SUCCESS) {
            pthread_mutex_unlock(&g_state.mutex);
            return err;
        }
    }

    @autoreleasepool {
        std::string name(kernel_name);

        // Check cache
        auto it = g_state.pipelineCache.find(name);
        if (it != g_state.pipelineCache.end()) {
            *pipeline = (__bridge_retained void*)it->second;
            pthread_mutex_unlock(&g_state.mutex);
            return METAL_SUCCESS;
        }

        // Find kernel function
        id<MTLFunction> function = findKernelFunction(name);
        if (!function) {
            pthread_mutex_unlock(&g_state.mutex);
            return METAL_ERROR_SHADER_NOT_FOUND;
        }

        // Create pipeline
        NSError* error = nil;
        id<MTLComputePipelineState> pipelineState =
            [g_state.device newComputePipelineStateWithFunction:function error:&error];

        if (!pipelineState) {
            pthread_mutex_unlock(&g_state.mutex);
            return METAL_ERROR_PIPELINE_FAILED;
        }

        // Cache pipeline
        g_state.pipelineCache[name] = pipelineState;

        *pipeline = (__bridge_retained void*)pipelineState;
        pthread_mutex_unlock(&g_state.mutex);
        return METAL_SUCCESS;
    }
}

void metal_destroy_compute_pipeline(void* pipeline) {
    if (!pipeline) return;

    @autoreleasepool {
        // Release the retained reference
        id<MTLComputePipelineState> pipelineState =
            (__bridge_transfer id<MTLComputePipelineState>)pipeline;
        (void)pipelineState; // Let ARC release it
    }
}

metal_error_t metal_execute_command(
    void* pipeline,
    void** buffers,
    uint32_t num_buffers,
    const void* params,
    size_t params_size,
    uint32_t thread_groups_x,
    uint32_t thread_groups_y,
    uint32_t thread_groups_z
) {
    if (!pipeline) {
        return METAL_ERROR_INVALID_PARAMS;
    }

    pthread_mutex_lock(&g_state.mutex);

    if (!g_state.initialized) {
        pthread_mutex_unlock(&g_state.mutex);
        return METAL_ERROR_NOT_INITIALIZED;
    }

    @autoreleasepool {
        id<MTLComputePipelineState> pipelineState =
            (__bridge id<MTLComputePipelineState>)pipeline;

        // Create command buffer
        id<MTLCommandBuffer> commandBuffer = [g_state.commandQueue commandBuffer];
        if (!commandBuffer) {
            pthread_mutex_unlock(&g_state.mutex);
            return METAL_ERROR_COMMAND_FAILED;
        }

        // Create compute encoder
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (!encoder) {
            pthread_mutex_unlock(&g_state.mutex);
            return METAL_ERROR_COMMAND_FAILED;
        }

        [encoder setComputePipelineState:pipelineState];

        // Set buffers
        for (uint32_t i = 0; i < num_buffers; i++) {
            if (buffers[i]) {
                id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)buffers[i];
                [encoder setBuffer:buffer offset:0 atIndex:i];
            }
        }

        // Set parameters if provided
        if (params && params_size > 0) {
            [encoder setBytes:params length:params_size atIndex:num_buffers];

            // Also set individual parameter components for kernels expecting them
            if (params_size >= sizeof(uint32_t)) {
                [encoder setBytes:params length:sizeof(uint32_t) atIndex:num_buffers];
            }
            if (params_size >= 2 * sizeof(uint32_t)) {
                [encoder setBytes:((const char*)params + sizeof(uint32_t))
                           length:sizeof(uint32_t)
                          atIndex:num_buffers + 1];
            }
        }

        // Calculate thread configuration
        NSUInteger threadGroupSize = MIN(THREAD_GROUP_SIZE, pipelineState.maxTotalThreadsPerThreadgroup);
        MTLSize threadsPerGroup = MTLSizeMake(threadGroupSize, 1, 1);
        MTLSize threadGroups = MTLSizeMake(thread_groups_x, thread_groups_y, thread_groups_z);

        [encoder dispatchThreadgroups:threadGroups threadsPerThreadgroup:threadsPerGroup];
        [encoder endEncoding];

        // Execute and wait
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        pthread_mutex_unlock(&g_state.mutex);

        if (commandBuffer.status == MTLCommandBufferStatusError) {
            return METAL_ERROR_COMMAND_FAILED;
        }

        return METAL_SUCCESS;
    }
}

metal_error_t metal_create_buffer(size_t size, void** buffer) {
    if (!buffer || size == 0) {
        return METAL_ERROR_INVALID_PARAMS;
    }

    pthread_mutex_lock(&g_state.mutex);

    if (!g_state.initialized) {
        metal_error_t err = metal_common_initialize();
        if (err != METAL_SUCCESS) {
            pthread_mutex_unlock(&g_state.mutex);
            return err;
        }
    }

    @autoreleasepool {
        id<MTLBuffer> mtlBuffer = [g_state.device newBufferWithLength:size
                                                             options:MTLResourceStorageModeShared];
        if (!mtlBuffer) {
            pthread_mutex_unlock(&g_state.mutex);
            return METAL_ERROR_BUFFER_FAILED;
        }

        *buffer = (__bridge_retained void*)mtlBuffer;
        pthread_mutex_unlock(&g_state.mutex);
        return METAL_SUCCESS;
    }
}

metal_error_t metal_create_buffer_with_data(const void* data, size_t size, void** buffer) {
    if (!data || !buffer || size == 0) {
        return METAL_ERROR_INVALID_PARAMS;
    }

    pthread_mutex_lock(&g_state.mutex);

    if (!g_state.initialized) {
        metal_error_t err = metal_common_initialize();
        if (err != METAL_SUCCESS) {
            pthread_mutex_unlock(&g_state.mutex);
            return err;
        }
    }

    @autoreleasepool {
        id<MTLBuffer> mtlBuffer = [g_state.device newBufferWithBytes:data
                                                             length:size
                                                            options:MTLResourceStorageModeShared];
        if (!mtlBuffer) {
            pthread_mutex_unlock(&g_state.mutex);
            return METAL_ERROR_BUFFER_FAILED;
        }

        *buffer = (__bridge_retained void*)mtlBuffer;
        pthread_mutex_unlock(&g_state.mutex);
        return METAL_SUCCESS;
    }
}

metal_error_t metal_copy_to_buffer(void* buffer, const void* data, size_t size) {
    if (!buffer || !data || size == 0) {
        return METAL_ERROR_INVALID_PARAMS;
    }

    @autoreleasepool {
        id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)buffer;
        if (mtlBuffer.length < size) {
            return METAL_ERROR_INVALID_PARAMS;
        }

        memcpy(mtlBuffer.contents, data, size);
        return METAL_SUCCESS;
    }
}

metal_error_t metal_copy_from_buffer(void* data, void* buffer, size_t size) {
    if (!buffer || !data || size == 0) {
        return METAL_ERROR_INVALID_PARAMS;
    }

    @autoreleasepool {
        id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)buffer;
        if (mtlBuffer.length < size) {
            return METAL_ERROR_INVALID_PARAMS;
        }

        memcpy(data, mtlBuffer.contents, size);
        return METAL_SUCCESS;
    }
}

void metal_destroy_buffer(void* buffer) {
    if (!buffer) return;

    @autoreleasepool {
        id<MTLBuffer> mtlBuffer = (__bridge_transfer id<MTLBuffer>)buffer;
        (void)mtlBuffer; // Let ARC release it
    }
}

void* metal_get_buffer_contents(void* buffer) {
    if (!buffer) return NULL;

    @autoreleasepool {
        id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)buffer;
        return mtlBuffer.contents;
    }
}

metal_error_t metal_register_shader_source(const char* shader_source) {
    if (!shader_source) {
        return METAL_ERROR_INVALID_PARAMS;
    }

    pthread_mutex_lock(&g_state.mutex);

    if (!g_state.initialized) {
        metal_error_t err = metal_common_initialize();
        if (err != METAL_SUCCESS) {
            pthread_mutex_unlock(&g_state.mutex);
            return err;
        }
    }

    @autoreleasepool {
        NSString* source = [NSString stringWithUTF8String:shader_source];
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.fastMathEnabled = YES;

        NSError* error = nil;
        id<MTLLibrary> library = [g_state.device newLibraryWithSource:source
                                                             options:options
                                                               error:&error];

        if (!library) {
            pthread_mutex_unlock(&g_state.mutex);
            return METAL_ERROR_PIPELINE_FAILED;
        }

        g_state.registeredLibraries.push_back(library);
        pthread_mutex_unlock(&g_state.mutex);
        return METAL_SUCCESS;
    }
}

metal_error_t metal_register_library_path(const char* library_path) {
    if (!library_path) {
        return METAL_ERROR_INVALID_PARAMS;
    }

    pthread_mutex_lock(&g_state.mutex);

    if (!g_state.initialized) {
        metal_error_t err = metal_common_initialize();
        if (err != METAL_SUCCESS) {
            pthread_mutex_unlock(&g_state.mutex);
            return err;
        }
    }

    @autoreleasepool {
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:library_path]];
        NSError* error = nil;
        id<MTLLibrary> library = [g_state.device newLibraryWithURL:url error:&error];

        if (!library) {
            pthread_mutex_unlock(&g_state.mutex);
            return METAL_ERROR_SHADER_NOT_FOUND;
        }

        g_state.registeredLibraries.push_back(library);
        pthread_mutex_unlock(&g_state.mutex);
        return METAL_SUCCESS;
    }
}

} // extern "C"
