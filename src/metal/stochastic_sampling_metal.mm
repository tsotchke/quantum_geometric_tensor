#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <pthread.h>
#include <stdint.h>

// ===========================================================================
// Error Types
// ===========================================================================

typedef int32_t metal_error_t;

#define METAL_SUCCESS                   0
#define METAL_ERROR_INVALID_PARAMS      -1
#define METAL_ERROR_DEVICE_NOT_FOUND    -2
#define METAL_ERROR_OUT_OF_MEMORY       -3
#define METAL_ERROR_SHADER_FAILED       -4
#define METAL_ERROR_PIPELINE_FAILED     -5
#define METAL_ERROR_COMMAND_FAILED      -6
#define METAL_ERROR_NOT_INITIALIZED     -7

// ===========================================================================
// Internal State
// ===========================================================================

namespace {

struct StochasticSamplingState {
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLLibrary> library;
    id<MTLComputePipelineState> stochasticSamplingPipeline;
    id<MTLComputePipelineState> stochasticGradientPipeline;
    id<MTLComputePipelineState> importanceSamplingPipeline;
    id<MTLComputePipelineState> systematicResamplingPipeline;
    pthread_mutex_t mutex;
    bool initialized;
};

StochasticSamplingState g_state = {
    .device = nil,
    .commandQueue = nil,
    .library = nil,
    .stochasticSamplingPipeline = nil,
    .stochasticGradientPipeline = nil,
    .importanceSamplingPipeline = nil,
    .systematicResamplingPipeline = nil,
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .initialized = false
};

// Metal shader source for stochastic sampling operations
static const char* g_shaderSource = R"(
#include <metal_stdlib>
using namespace metal;

// High-quality hash function for random number generation
inline float hash_to_float(uint seed) {
    seed = (seed ^ 61u) ^ (seed >> 16u);
    seed *= 9u;
    seed = seed ^ (seed >> 4u);
    seed *= 0x27d4eb2du;
    seed = seed ^ (seed >> 15u);
    return float(seed) / float(0xFFFFFFFFu);
}

// Generate uniform random number in [0, 1)
inline float random_uniform(uint gid, uint round, uint seed) {
    uint combined = gid * 1664525u + round * 1013904223u + seed;
    return hash_to_float(combined);
}

// Stochastic sampling kernel - samples states based on probability distribution
kernel void stochastic_sampling(
    device const float2* input_states [[buffer(0)]],
    device const float* probabilities [[buffer(1)]],
    device float2* output_states [[buffer(2)]],
    constant uint& num_states [[buffer(3)]],
    constant uint& num_samples [[buffer(4)]],
    constant uint& random_seed [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_samples) return;

    // Generate random value for this sample
    float rand_val = random_uniform(gid, 0, random_seed);

    // Binary search for the selected state using cumulative distribution
    float cumulative = 0.0f;
    uint selected_idx = 0;

    for (uint i = 0; i < num_states; i++) {
        cumulative += probabilities[i];
        if (rand_val <= cumulative) {
            selected_idx = i;
            break;
        }
    }

    // Ensure we don't overflow
    if (selected_idx >= num_states) {
        selected_idx = num_states - 1;
    }

    output_states[gid] = input_states[selected_idx];
}

// Stochastic gradient descent kernel with momentum
kernel void stochastic_gradient(
    device const float2* states [[buffer(0)]],
    device const float2* gradients [[buffer(1)]],
    device float2* output [[buffer(2)]],
    device float2* momentum [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    constant float& learning_rate [[buffer(5)]],
    constant float& momentum_factor [[buffer(6)]],
    constant float& weight_decay [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= batch_size) return;

    float2 state = states[gid];
    float2 grad = gradients[gid];
    float2 mom = momentum[gid];

    // Apply weight decay (L2 regularization)
    grad.x += weight_decay * state.x;
    grad.y += weight_decay * state.y;

    // Update momentum with exponential moving average
    mom.x = momentum_factor * mom.x + (1.0f - momentum_factor) * grad.x;
    mom.y = momentum_factor * mom.y + (1.0f - momentum_factor) * grad.y;

    // Apply gradient update with momentum
    output[gid] = float2(
        state.x - learning_rate * mom.x,
        state.y - learning_rate * mom.y
    );

    // Store updated momentum
    momentum[gid] = mom;
}

// Importance sampling with effective sample size calculation
kernel void importance_sampling(
    device const float2* states [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device uint* indices [[buffer(2)]],
    device float* resampled_weights [[buffer(3)]],
    device float* ess_output [[buffer(4)]],
    constant uint& num_particles [[buffer(5)]],
    constant uint& random_seed [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_particles) return;

    // Compute normalized weights and cumulative sum (done in parallel approximation)
    // Each thread handles one particle

    // First pass: compute local contribution to total weight
    threadgroup float shared_weights[256];
    threadgroup float shared_weights_sq[256];

    uint local_id = gid % 256;
    uint group_id = gid / 256;

    float w = weights[gid];
    shared_weights[local_id] = w;
    shared_weights_sq[local_id] = w * w;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction for sum
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (local_id < stride && (local_id + stride) < 256) {
            shared_weights[local_id] += shared_weights[local_id + stride];
            shared_weights_sq[local_id] += shared_weights_sq[local_id + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float total_weight = shared_weights[0];
    float sum_weights_sq = shared_weights_sq[0];

    // Compute effective sample size: ESS = (sum w)^2 / sum(w^2)
    if (gid == 0 && total_weight > 0.0f) {
        float ess = (total_weight * total_weight) / sum_weights_sq;
        ess_output[0] = ess;
    }

    // Systematic resampling
    float step = total_weight / float(num_particles);
    float rand_start = random_uniform(gid, 0, random_seed) * step;
    float target = rand_start + float(gid) * step;

    // Find selected index via cumulative distribution
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

// Systematic resampling kernel for particle filters
kernel void systematic_resampling(
    device const float2* particles [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const float* cumulative_weights [[buffer(2)]],
    device float2* resampled_particles [[buffer(3)]],
    constant uint& num_particles [[buffer(4)]],
    constant float& total_weight [[buffer(5)]],
    constant float& u0 [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_particles) return;

    // Compute target cumulative weight for this particle
    float step = total_weight / float(num_particles);
    float target = u0 + float(gid) * step;

    // Binary search for the particle index
    uint left = 0;
    uint right = num_particles;

    while (left < right) {
        uint mid = (left + right) / 2;
        if (cumulative_weights[mid] < target) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    uint selected_idx = min(left, num_particles - 1);
    resampled_particles[gid] = particles[selected_idx];
}
)";

metal_error_t initialize_stochastic_sampling() {
    if (g_state.initialized) return METAL_SUCCESS;

    pthread_mutex_lock(&g_state.mutex);

    if (g_state.initialized) {
        pthread_mutex_unlock(&g_state.mutex);
        return METAL_SUCCESS;
    }

    @autoreleasepool {
        // Get default Metal device
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
            return METAL_ERROR_COMMAND_FAILED;
        }

        // Compile shader library
        NSError* error = nil;
        NSString* source = [NSString stringWithUTF8String:g_shaderSource];
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.fastMathEnabled = YES;

        g_state.library = [g_state.device newLibraryWithSource:source
                                                       options:options
                                                         error:&error];
        if (!g_state.library) {
            NSLog(@"Failed to compile stochastic sampling shaders: %@", error);
            g_state.commandQueue = nil;
            g_state.device = nil;
            pthread_mutex_unlock(&g_state.mutex);
            return METAL_ERROR_SHADER_FAILED;
        }

        // Create compute pipelines for each kernel
        id<MTLFunction> stochasticSamplingFunc = [g_state.library newFunctionWithName:@"stochastic_sampling"];
        id<MTLFunction> stochasticGradientFunc = [g_state.library newFunctionWithName:@"stochastic_gradient"];
        id<MTLFunction> importanceSamplingFunc = [g_state.library newFunctionWithName:@"importance_sampling"];
        id<MTLFunction> systematicResamplingFunc = [g_state.library newFunctionWithName:@"systematic_resampling"];

        if (!stochasticSamplingFunc || !stochasticGradientFunc ||
            !importanceSamplingFunc || !systematicResamplingFunc) {
            NSLog(@"Failed to load kernel functions");
            g_state.library = nil;
            g_state.commandQueue = nil;
            g_state.device = nil;
            pthread_mutex_unlock(&g_state.mutex);
            return METAL_ERROR_SHADER_FAILED;
        }

        g_state.stochasticSamplingPipeline = [g_state.device newComputePipelineStateWithFunction:stochasticSamplingFunc error:&error];
        g_state.stochasticGradientPipeline = [g_state.device newComputePipelineStateWithFunction:stochasticGradientFunc error:&error];
        g_state.importanceSamplingPipeline = [g_state.device newComputePipelineStateWithFunction:importanceSamplingFunc error:&error];
        g_state.systematicResamplingPipeline = [g_state.device newComputePipelineStateWithFunction:systematicResamplingFunc error:&error];

        if (!g_state.stochasticSamplingPipeline || !g_state.stochasticGradientPipeline ||
            !g_state.importanceSamplingPipeline || !g_state.systematicResamplingPipeline) {
            NSLog(@"Failed to create compute pipelines: %@", error);
            g_state.library = nil;
            g_state.commandQueue = nil;
            g_state.device = nil;
            pthread_mutex_unlock(&g_state.mutex);
            return METAL_ERROR_PIPELINE_FAILED;
        }

        g_state.initialized = true;
    }

    pthread_mutex_unlock(&g_state.mutex);
    return METAL_SUCCESS;
}

} // anonymous namespace

// ===========================================================================
// External Interface
// ===========================================================================

extern "C" {

metal_error_t compute_stochastic_sampling(
    void* input_states_buffer,
    void* probabilities_buffer,
    void* output_states_buffer,
    uint32_t num_states,
    uint32_t num_samples
) {
    if (!input_states_buffer || !probabilities_buffer || !output_states_buffer) {
        return METAL_ERROR_INVALID_PARAMS;
    }

    if (num_states == 0 || num_samples == 0) {
        return METAL_ERROR_INVALID_PARAMS;
    }

    metal_error_t init_result = initialize_stochastic_sampling();
    if (init_result != METAL_SUCCESS) {
        return init_result;
    }

    @autoreleasepool {
        // Create Metal buffers
        id<MTLBuffer> inputBuffer = [g_state.device newBufferWithBytes:input_states_buffer
                                                               length:num_states * sizeof(float) * 2
                                                              options:MTLResourceStorageModeShared];
        id<MTLBuffer> probBuffer = [g_state.device newBufferWithBytes:probabilities_buffer
                                                              length:num_states * sizeof(float)
                                                             options:MTLResourceStorageModeShared];
        id<MTLBuffer> outputBuffer = [g_state.device newBufferWithLength:num_samples * sizeof(float) * 2
                                                                options:MTLResourceStorageModeShared];

        if (!inputBuffer || !probBuffer || !outputBuffer) {
            return METAL_ERROR_OUT_OF_MEMORY;
        }

        // Generate random seed
        uint32_t random_seed = arc4random();

        // Create command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [g_state.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        if (!commandBuffer || !encoder) {
            return METAL_ERROR_COMMAND_FAILED;
        }

        [encoder setComputePipelineState:g_state.stochasticSamplingPipeline];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:probBuffer offset:0 atIndex:1];
        [encoder setBuffer:outputBuffer offset:0 atIndex:2];
        [encoder setBytes:&num_states length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&num_samples length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&random_seed length:sizeof(uint32_t) atIndex:5];

        // Calculate thread configuration
        NSUInteger threadGroupSize = MIN(256, g_state.stochasticSamplingPipeline.maxTotalThreadsPerThreadgroup);
        MTLSize threadsPerGroup = MTLSizeMake(threadGroupSize, 1, 1);
        MTLSize numThreadGroups = MTLSizeMake((num_samples + threadGroupSize - 1) / threadGroupSize, 1, 1);

        [encoder dispatchThreadgroups:numThreadGroups threadsPerThreadgroup:threadsPerGroup];
        [encoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        if (commandBuffer.status == MTLCommandBufferStatusError) {
            NSLog(@"Stochastic sampling command failed: %@", commandBuffer.error);
            return METAL_ERROR_COMMAND_FAILED;
        }

        // Copy results back
        memcpy(output_states_buffer, outputBuffer.contents, num_samples * sizeof(float) * 2);

        return METAL_SUCCESS;
    }
}

metal_error_t compute_stochastic_gradient(
    void* states_buffer,
    void* gradients_buffer,
    void* output_buffer,
    uint32_t batch_size,
    float learning_rate
) {
    if (!states_buffer || !gradients_buffer || !output_buffer) {
        return METAL_ERROR_INVALID_PARAMS;
    }

    if (batch_size == 0) {
        return METAL_ERROR_INVALID_PARAMS;
    }

    metal_error_t init_result = initialize_stochastic_sampling();
    if (init_result != METAL_SUCCESS) {
        return init_result;
    }

    @autoreleasepool {
        size_t buffer_size = batch_size * sizeof(float) * 2;

        // Create Metal buffers
        id<MTLBuffer> statesBuffer = [g_state.device newBufferWithBytes:states_buffer
                                                                length:buffer_size
                                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> gradientsBuffer = [g_state.device newBufferWithBytes:gradients_buffer
                                                                   length:buffer_size
                                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> outputBuffer = [g_state.device newBufferWithLength:buffer_size
                                                                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> momentumBuffer = [g_state.device newBufferWithLength:buffer_size
                                                                   options:MTLResourceStorageModeShared];

        if (!statesBuffer || !gradientsBuffer || !outputBuffer || !momentumBuffer) {
            return METAL_ERROR_OUT_OF_MEMORY;
        }

        // Initialize momentum to zero
        memset(momentumBuffer.contents, 0, buffer_size);

        // Default hyperparameters
        float momentum_factor = 0.9f;
        float weight_decay = 0.0001f;

        // Create command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [g_state.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        if (!commandBuffer || !encoder) {
            return METAL_ERROR_COMMAND_FAILED;
        }

        [encoder setComputePipelineState:g_state.stochasticGradientPipeline];
        [encoder setBuffer:statesBuffer offset:0 atIndex:0];
        [encoder setBuffer:gradientsBuffer offset:0 atIndex:1];
        [encoder setBuffer:outputBuffer offset:0 atIndex:2];
        [encoder setBuffer:momentumBuffer offset:0 atIndex:3];
        [encoder setBytes:&batch_size length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&learning_rate length:sizeof(float) atIndex:5];
        [encoder setBytes:&momentum_factor length:sizeof(float) atIndex:6];
        [encoder setBytes:&weight_decay length:sizeof(float) atIndex:7];

        // Calculate thread configuration
        NSUInteger threadGroupSize = MIN(256, g_state.stochasticGradientPipeline.maxTotalThreadsPerThreadgroup);
        MTLSize threadsPerGroup = MTLSizeMake(threadGroupSize, 1, 1);
        MTLSize numThreadGroups = MTLSizeMake((batch_size + threadGroupSize - 1) / threadGroupSize, 1, 1);

        [encoder dispatchThreadgroups:numThreadGroups threadsPerThreadgroup:threadsPerGroup];
        [encoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        if (commandBuffer.status == MTLCommandBufferStatusError) {
            NSLog(@"Stochastic gradient command failed: %@", commandBuffer.error);
            return METAL_ERROR_COMMAND_FAILED;
        }

        // Copy results back
        memcpy(output_buffer, outputBuffer.contents, buffer_size);

        return METAL_SUCCESS;
    }
}

metal_error_t compute_importance_sampling(
    void* states_buffer,
    void* weights_buffer,
    void* indices_buffer,
    void* resampled_weights_buffer,
    uint32_t num_particles
) {
    if (!states_buffer || !weights_buffer || !indices_buffer || !resampled_weights_buffer) {
        return METAL_ERROR_INVALID_PARAMS;
    }

    if (num_particles == 0) {
        return METAL_ERROR_INVALID_PARAMS;
    }

    metal_error_t init_result = initialize_stochastic_sampling();
    if (init_result != METAL_SUCCESS) {
        return init_result;
    }

    @autoreleasepool {
        // Create Metal buffers
        id<MTLBuffer> statesBuffer = [g_state.device newBufferWithBytes:states_buffer
                                                                length:num_particles * sizeof(float) * 2
                                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> weightsBuffer = [g_state.device newBufferWithBytes:weights_buffer
                                                                 length:num_particles * sizeof(float)
                                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> indicesBuffer = [g_state.device newBufferWithLength:num_particles * sizeof(uint32_t)
                                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> resampledWeightsBuffer = [g_state.device newBufferWithLength:num_particles * sizeof(float)
                                                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> essBuffer = [g_state.device newBufferWithLength:sizeof(float)
                                                              options:MTLResourceStorageModeShared];

        if (!statesBuffer || !weightsBuffer || !indicesBuffer ||
            !resampledWeightsBuffer || !essBuffer) {
            return METAL_ERROR_OUT_OF_MEMORY;
        }

        // Generate random seed
        uint32_t random_seed = arc4random();

        // Create command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [g_state.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        if (!commandBuffer || !encoder) {
            return METAL_ERROR_COMMAND_FAILED;
        }

        [encoder setComputePipelineState:g_state.importanceSamplingPipeline];
        [encoder setBuffer:statesBuffer offset:0 atIndex:0];
        [encoder setBuffer:weightsBuffer offset:0 atIndex:1];
        [encoder setBuffer:indicesBuffer offset:0 atIndex:2];
        [encoder setBuffer:resampledWeightsBuffer offset:0 atIndex:3];
        [encoder setBuffer:essBuffer offset:0 atIndex:4];
        [encoder setBytes:&num_particles length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&random_seed length:sizeof(uint32_t) atIndex:6];

        // Calculate thread configuration
        NSUInteger threadGroupSize = MIN(256, g_state.importanceSamplingPipeline.maxTotalThreadsPerThreadgroup);
        MTLSize threadsPerGroup = MTLSizeMake(threadGroupSize, 1, 1);
        MTLSize numThreadGroups = MTLSizeMake((num_particles + threadGroupSize - 1) / threadGroupSize, 1, 1);

        [encoder dispatchThreadgroups:numThreadGroups threadsPerThreadgroup:threadsPerGroup];
        [encoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        if (commandBuffer.status == MTLCommandBufferStatusError) {
            NSLog(@"Importance sampling command failed: %@", commandBuffer.error);
            return METAL_ERROR_COMMAND_FAILED;
        }

        // Copy results back
        memcpy(indices_buffer, indicesBuffer.contents, num_particles * sizeof(uint32_t));
        memcpy(resampled_weights_buffer, resampledWeightsBuffer.contents, num_particles * sizeof(float));

        return METAL_SUCCESS;
    }
}

// Additional utility function for systematic resampling with precomputed cumulative weights
metal_error_t compute_systematic_resampling(
    void* particles_buffer,
    void* weights_buffer,
    void* cumulative_weights_buffer,
    void* resampled_particles_buffer,
    uint32_t num_particles,
    float total_weight
) {
    if (!particles_buffer || !weights_buffer || !cumulative_weights_buffer || !resampled_particles_buffer) {
        return METAL_ERROR_INVALID_PARAMS;
    }

    if (num_particles == 0 || total_weight <= 0.0f) {
        return METAL_ERROR_INVALID_PARAMS;
    }

    metal_error_t init_result = initialize_stochastic_sampling();
    if (init_result != METAL_SUCCESS) {
        return init_result;
    }

    @autoreleasepool {
        // Create Metal buffers
        id<MTLBuffer> particlesBuffer = [g_state.device newBufferWithBytes:particles_buffer
                                                                   length:num_particles * sizeof(float) * 2
                                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> weightsBuffer = [g_state.device newBufferWithBytes:weights_buffer
                                                                 length:num_particles * sizeof(float)
                                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> cumulativeBuffer = [g_state.device newBufferWithBytes:cumulative_weights_buffer
                                                                    length:num_particles * sizeof(float)
                                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> resampledBuffer = [g_state.device newBufferWithLength:num_particles * sizeof(float) * 2
                                                                    options:MTLResourceStorageModeShared];

        if (!particlesBuffer || !weightsBuffer || !cumulativeBuffer || !resampledBuffer) {
            return METAL_ERROR_OUT_OF_MEMORY;
        }

        // Generate uniform random start point in [0, step)
        float step = total_weight / (float)num_particles;
        float u0 = ((float)arc4random() / (float)UINT32_MAX) * step;

        // Create command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [g_state.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        if (!commandBuffer || !encoder) {
            return METAL_ERROR_COMMAND_FAILED;
        }

        [encoder setComputePipelineState:g_state.systematicResamplingPipeline];
        [encoder setBuffer:particlesBuffer offset:0 atIndex:0];
        [encoder setBuffer:weightsBuffer offset:0 atIndex:1];
        [encoder setBuffer:cumulativeBuffer offset:0 atIndex:2];
        [encoder setBuffer:resampledBuffer offset:0 atIndex:3];
        [encoder setBytes:&num_particles length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&total_weight length:sizeof(float) atIndex:5];
        [encoder setBytes:&u0 length:sizeof(float) atIndex:6];

        // Calculate thread configuration
        NSUInteger threadGroupSize = MIN(256, g_state.systematicResamplingPipeline.maxTotalThreadsPerThreadgroup);
        MTLSize threadsPerGroup = MTLSizeMake(threadGroupSize, 1, 1);
        MTLSize numThreadGroups = MTLSizeMake((num_particles + threadGroupSize - 1) / threadGroupSize, 1, 1);

        [encoder dispatchThreadgroups:numThreadGroups threadsPerThreadgroup:threadsPerGroup];
        [encoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        if (commandBuffer.status == MTLCommandBufferStatusError) {
            NSLog(@"Systematic resampling command failed: %@", commandBuffer.error);
            return METAL_ERROR_COMMAND_FAILED;
        }

        // Copy results back
        memcpy(resampled_particles_buffer, resampledBuffer.contents, num_particles * sizeof(float) * 2);

        return METAL_SUCCESS;
    }
}

// Cleanup function
void cleanup_stochastic_sampling(void) {
    pthread_mutex_lock(&g_state.mutex);

    if (!g_state.initialized) {
        pthread_mutex_unlock(&g_state.mutex);
        return;
    }

    @autoreleasepool {
        g_state.systematicResamplingPipeline = nil;
        g_state.importanceSamplingPipeline = nil;
        g_state.stochasticGradientPipeline = nil;
        g_state.stochasticSamplingPipeline = nil;
        g_state.library = nil;
        g_state.commandQueue = nil;
        g_state.device = nil;
        g_state.initialized = false;
    }

    pthread_mutex_unlock(&g_state.mutex);
}

} // extern "C"
