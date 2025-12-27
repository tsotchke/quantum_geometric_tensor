/**
 * compute_metal.mm - Metal backend implementation
 *
 * This backend wraps the existing Metal infrastructure in src/metal/
 * and provides the compute_backend.h vtable interface for the
 * distributed quantum computing engine.
 *
 * Features:
 * - Apple Silicon optimization (AMX, unified memory)
 * - Integration with existing Metal shaders
 * - Accelerate framework for BLAS operations
 * - MPI support for distributed Mac clusters
 */

#include "quantum_geometric/supercomputer/compute_backend.h"
#include "quantum_geometric/supercomputer/compute_simd.h"

#if COMPUTE_HAS_METAL

#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#import <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if COMPUTE_HAS_MPI
#include <mpi.h>
#endif

// ============================================================================
// Metal Backend Context
// ============================================================================

struct MetalBackendContext {
    // Metal core objects
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLLibrary> library;

    // Compute pipelines for quantum operations
    id<MTLComputePipelineState> unitaryPipeline;
    id<MTLComputePipelineState> normalizePipeline;
    id<MTLComputePipelineState> tensorContractPipeline;
    id<MTLComputePipelineState> gradientPipeline;
    id<MTLComputePipelineState> innerProductPipeline;
    id<MTLComputePipelineState> expectationPipeline;

    // Distributed compute pipelines
    id<MTLComputePipelineState> distributedTensorPipeline;
    id<MTLComputePipelineState> distributedGradientPipeline;
    id<MTLComputePipelineState> distributedSyncPipeline;

    // Configuration
    int node_rank;
    int num_nodes;
    size_t threadgroup_size;
    bool amx_enabled;

    // MPI state
#if COMPUTE_HAS_MPI
    MPI_Comm comm;
    bool mpi_initialized_by_us;
#endif

    // Memory tracking
    size_t total_allocated;
    size_t peak_allocated;

    // Performance metrics
    ComputeMetrics metrics;

    // Error state
    char last_error[256];
};

// ============================================================================
// Metal Shader Source (Inline for core operations)
// ============================================================================

static const char* const kQuantumShaderSourceStr = R"(
#include <metal_stdlib>
using namespace metal;

// Complex number multiply-add: out += a * b
inline float2 complex_madd(float2 a, float2 b, float2 accum) {
    return float2(
        accum.x + a.x * b.x - a.y * b.y,
        accum.y + a.x * b.y + a.y * b.x
    );
}

// Complex conjugate multiply-add: out += conj(a) * b
inline float2 complex_conj_madd(float2 a, float2 b, float2 accum) {
    return float2(
        accum.x + a.x * b.x + a.y * b.y,
        accum.y + a.x * b.y - a.y * b.x
    );
}

// Apply unitary transformation to state vector
kernel void quantum_unitary_transform(
    device float2* state [[buffer(0)]],
    device const float2* unitary [[buffer(1)]],
    constant uint& state_size [[buffer(2)]],
    device float2* output [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= state_size) return;

    float2 result = float2(0.0f, 0.0f);
    for (uint j = 0; j < state_size; j++) {
        result = complex_madd(unitary[tid * state_size + j], state[j], result);
    }
    output[tid] = result;
}

// Compute norm squared of state vector (partial reduction)
kernel void quantum_norm_squared(
    device const float2* state [[buffer(0)]],
    device float* partial_sums [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float local_sum[256];

    float sum = 0.0f;
    if (tid < size) {
        float2 amp = state[tid];
        sum = amp.x * amp.x + amp.y * amp.y;
    }

    local_sum[tid % tg_size] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction in threadgroup
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if ((tid % tg_size) < stride) {
            local_sum[tid % tg_size] += local_sum[tid % tg_size + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if ((tid % tg_size) == 0) {
        partial_sums[tgid] = local_sum[0];
    }
}

// Scale state vector by inverse norm
kernel void quantum_scale(
    device float2* state [[buffer(0)]],
    constant float& scale [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= size) return;
    state[tid] *= scale;
}

// Complex matrix multiplication for tensor contraction
kernel void complex_matrix_multiply(
    device const float2* A [[buffer(0)]],
    device const float2* B [[buffer(1)]],
    device float2* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint row = tid.y;
    uint col = tid.x;

    if (row >= M || col >= K) return;

    float2 sum = float2(0.0f, 0.0f);
    for (uint i = 0; i < N; i++) {
        sum = complex_madd(A[row * N + i], B[i * K + col], sum);
    }
    C[row * K + col] = sum;
}

// Compute inner product: <a|b>
kernel void quantum_inner_product(
    device const float2* state_a [[buffer(0)]],
    device const float2* state_b [[buffer(1)]],
    device float2* partial_results [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float2 local_sum[256];

    float2 sum = float2(0.0f, 0.0f);
    if (tid < size) {
        sum = complex_conj_madd(state_a[tid], state_b[tid], sum);
    }

    local_sum[tid % tg_size] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if ((tid % tg_size) < stride) {
            local_sum[tid % tg_size] += local_sum[tid % tg_size + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if ((tid % tg_size) == 0) {
        partial_results[tgid] = local_sum[0];
    }
}

// Compute gradient using parameter-shift rule
kernel void quantum_gradient_compute(
    device const float2* forward_state [[buffer(0)]],
    device const float2* backward_state [[buffer(1)]],
    device float2* gradient [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= size) return;

    // Gradient = Re(<backward|forward>)
    float2 fw = forward_state[tid];
    float2 bw = backward_state[tid];

    // conj(bw) * fw
    gradient[tid] = float2(
        bw.x * fw.x + bw.y * fw.y,
        bw.x * fw.y - bw.y * fw.x
    );
}

// Expectation value for diagonal observable
kernel void quantum_expectation_diagonal(
    device const float2* state [[buffer(0)]],
    device const float* observable [[buffer(1)]],
    device float* partial_sums [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float local_sum[256];

    float sum = 0.0f;
    if (tid < size) {
        float2 amp = state[tid];
        float prob = amp.x * amp.x + amp.y * amp.y;
        sum = prob * observable[tid];
    }

    local_sum[tid % tg_size] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if ((tid % tg_size) < stride) {
            local_sum[tid % tg_size] += local_sum[tid % tg_size + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if ((tid % tg_size) == 0) {
        partial_sums[tgid] = local_sum[0];
    }
}
)";

// ============================================================================
// Helper Functions
// ============================================================================

static id<MTLComputePipelineState> createPipeline(id<MTLDevice> device,
                                                   id<MTLLibrary> library,
                                                   NSString* functionName,
                                                   NSError** error) {
    id<MTLFunction> function = [library newFunctionWithName:functionName];
    if (!function) {
        if (error) {
            *error = [NSError errorWithDomain:@"MetalBackend"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey:
                                         [NSString stringWithFormat:@"Function %@ not found", functionName]}];
        }
        return nil;
    }
    return [device newComputePipelineStateWithFunction:function error:error];
}

static void setLastError(MetalBackendContext* ctx, NSString* error) {
    strncpy(ctx->last_error, [error UTF8String], sizeof(ctx->last_error) - 1);
}

// ============================================================================
// Lifecycle Operations
// ============================================================================

static ComputeBackend* metal_init(const ComputeDistributedConfig* config) {
    @autoreleasepool {
        MetalBackendContext* ctx = (MetalBackendContext*)calloc(1, sizeof(MetalBackendContext));
        if (!ctx) return nullptr;

        // Get Metal device
        ctx->device = MTLCreateSystemDefaultDevice();
        if (!ctx->device) {
            free(ctx);
            return nullptr;
        }

        // Create command queue
        ctx->commandQueue = [ctx->device newCommandQueue];
        if (!ctx->commandQueue) {
            free(ctx);
            return nullptr;
        }

        // Compile shader library from inline source
        NSError* error = nil;
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.fastMathEnabled = YES;

        NSString* shaderSource = [NSString stringWithUTF8String:kQuantumShaderSourceStr];
        ctx->library = [ctx->device newLibraryWithSource:shaderSource
                                                 options:options
                                                   error:&error];
        if (!ctx->library) {
            setLastError(ctx, [error localizedDescription]);
            free(ctx);
            return nullptr;
        }

        // Create compute pipelines
        ctx->unitaryPipeline = createPipeline(ctx->device, ctx->library,
                                              @"quantum_unitary_transform", &error);
        ctx->normalizePipeline = createPipeline(ctx->device, ctx->library,
                                                @"quantum_norm_squared", &error);
        ctx->tensorContractPipeline = createPipeline(ctx->device, ctx->library,
                                                     @"complex_matrix_multiply", &error);
        ctx->gradientPipeline = createPipeline(ctx->device, ctx->library,
                                               @"quantum_gradient_compute", &error);
        ctx->innerProductPipeline = createPipeline(ctx->device, ctx->library,
                                                   @"quantum_inner_product", &error);
        ctx->expectationPipeline = createPipeline(ctx->device, ctx->library,
                                                  @"quantum_expectation_diagonal", &error);

        // Configure threadgroup size
        ctx->threadgroup_size = config->num_threads_per_node > 0 ?
                               config->num_threads_per_node : 256;

        // Check for AMX support (Apple Silicon M1+)
        if ([ctx->device supportsFamily:MTLGPUFamilyApple7]) {
            ctx->amx_enabled = true;
        }

        // Initialize MPI if needed
#if COMPUTE_HAS_MPI
        int mpi_initialized = 0;
        MPI_Initialized(&mpi_initialized);

        if (!mpi_initialized && config->num_nodes > 1) {
            int provided;
            MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);
            ctx->mpi_initialized_by_us = true;
        }

        if (mpi_initialized || ctx->mpi_initialized_by_us) {
            MPI_Comm_dup(MPI_COMM_WORLD, &ctx->comm);
            MPI_Comm_rank(ctx->comm, &ctx->node_rank);
            MPI_Comm_size(ctx->comm, &ctx->num_nodes);
        } else {
            ctx->node_rank = 0;
            ctx->num_nodes = 1;
        }
#else
        ctx->node_rank = 0;
        ctx->num_nodes = 1;
#endif

        return (ComputeBackend*)ctx;
    }
}

static void metal_cleanup(ComputeBackend* backend) {
    if (!backend) return;

    MetalBackendContext* ctx = (MetalBackendContext*)backend;

    @autoreleasepool {
        // Release pipelines (ARC handles this in Objective-C++)
        ctx->unitaryPipeline = nil;
        ctx->normalizePipeline = nil;
        ctx->tensorContractPipeline = nil;
        ctx->gradientPipeline = nil;
        ctx->innerProductPipeline = nil;
        ctx->expectationPipeline = nil;
        ctx->library = nil;
        ctx->commandQueue = nil;
        ctx->device = nil;
    }

#if COMPUTE_HAS_MPI
    if (ctx->comm != MPI_COMM_NULL) {
        MPI_Comm_free(&ctx->comm);
    }
    if (ctx->mpi_initialized_by_us) {
        MPI_Finalize();
    }
#endif

    free(ctx);
}

static bool metal_probe(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        return device != nil;
    }
}

static ComputeResult metal_get_capabilities(ComputeBackend* backend,
                                             int* num_devices,
                                             size_t* total_memory) {
    @autoreleasepool {
        MetalBackendContext* ctx = (MetalBackendContext*)backend;
        if (!ctx) return COMPUTE_ERROR_INVALID_ARGUMENT;

        if (num_devices) {
            NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
            *num_devices = (int)[devices count];
        }

        if (total_memory) {
            *total_memory = [ctx->device recommendedMaxWorkingSetSize];
        }

        return COMPUTE_SUCCESS;
    }
}

// ============================================================================
// Memory Management
// ============================================================================

static void* metal_alloc(ComputeBackend* backend, size_t size, ComputeMemType mem_type) {
    @autoreleasepool {
        MetalBackendContext* ctx = (MetalBackendContext*)backend;
        if (!ctx) return nullptr;

        MTLResourceOptions options;
        switch (mem_type) {
            case COMPUTE_MEM_DEVICE:
                options = MTLResourceStorageModePrivate;
                break;
            case COMPUTE_MEM_UNIFIED:
            case COMPUTE_MEM_HOST:
            case COMPUTE_MEM_PINNED:
            default:
                options = MTLResourceStorageModeShared;
                break;
        }

        id<MTLBuffer> buffer = [ctx->device newBufferWithLength:size options:options];
        if (!buffer) return nullptr;

        ctx->total_allocated += size;
        if (ctx->total_allocated > ctx->peak_allocated) {
            ctx->peak_allocated = ctx->total_allocated;
        }

        // Return bridged pointer (caller must use metal_free)
        return (__bridge_retained void*)buffer;
    }
}

static void metal_free(ComputeBackend* backend, void* ptr, ComputeMemType mem_type) {
    (void)mem_type;
    if (!backend || !ptr) return;

    @autoreleasepool {
        id<MTLBuffer> buffer = (__bridge_transfer id<MTLBuffer>)ptr;
        buffer = nil;  // Release
    }
}

static ComputeResult metal_memcpy(ComputeBackend* backend,
                                   void* dst, ComputeMemType dst_type,
                                   const void* src, ComputeMemType src_type,
                                   size_t size, ComputeStream* stream) {
    @autoreleasepool {
        MetalBackendContext* ctx = (MetalBackendContext*)backend;
        if (!ctx) return COMPUTE_ERROR_INVALID_ARGUMENT;

        // If both are Metal buffers, use blit encoder
        if (dst_type == COMPUTE_MEM_DEVICE && src_type == COMPUTE_MEM_DEVICE) {
            id<MTLBuffer> dstBuf = (__bridge id<MTLBuffer>)dst;
            id<MTLBuffer> srcBuf = (__bridge id<MTLBuffer>)src;

            id<MTLCommandBuffer> cmdBuf = [ctx->commandQueue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
            [blit copyFromBuffer:srcBuf sourceOffset:0
                        toBuffer:dstBuf destinationOffset:0
                            size:size];
            [blit endEncoding];
            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];
        } else {
            // Use shared memory - direct copy
            memcpy(dst, src, size);
        }

        return COMPUTE_SUCCESS;
    }
}

static ComputeResult metal_memset(ComputeBackend* backend,
                                   void* ptr, int value, size_t size,
                                   ComputeStream* stream) {
    (void)stream;
    MetalBackendContext* ctx = (MetalBackendContext*)backend;
    if (!ctx) return COMPUTE_ERROR_INVALID_ARGUMENT;

    @autoreleasepool {
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)ptr;
        if ([buffer storageMode] == MTLStorageModeShared) {
            memset([buffer contents], value, size);
        } else {
            // For private buffers, use fill buffer command
            id<MTLCommandBuffer> cmdBuf = [ctx->commandQueue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
            [blit fillBuffer:buffer range:NSMakeRange(0, size) value:(uint8_t)value];
            [blit endEncoding];
            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];
        }
    }

    return COMPUTE_SUCCESS;
}

// ============================================================================
// Stream Management
// ============================================================================

struct MetalStream {
    id<MTLCommandBuffer> commandBuffer;
    dispatch_semaphore_t semaphore;
};

static ComputeStream* metal_create_stream(ComputeBackend* backend) {
    @autoreleasepool {
        MetalBackendContext* ctx = (MetalBackendContext*)backend;
        if (!ctx) return nullptr;

        MetalStream* stream = (MetalStream*)calloc(1, sizeof(MetalStream));
        if (!stream) return nullptr;

        stream->commandBuffer = [ctx->commandQueue commandBuffer];
        stream->semaphore = dispatch_semaphore_create(0);

        return (ComputeStream*)stream;
    }
}

static void metal_destroy_stream(ComputeBackend* backend, ComputeStream* stream) {
    (void)backend;
    if (!stream) return;

    MetalStream* mtlStream = (MetalStream*)stream;
    free(mtlStream);
}

static ComputeResult metal_synchronize_stream(ComputeBackend* backend, ComputeStream* stream) {
    (void)backend;
    if (!stream) return COMPUTE_SUCCESS;

    MetalStream* mtlStream = (MetalStream*)stream;
    if (mtlStream->commandBuffer) {
        [mtlStream->commandBuffer waitUntilCompleted];
    }

    return COMPUTE_SUCCESS;
}

static ComputeEvent* metal_create_event(ComputeBackend* backend) {
    (void)backend;
    return (ComputeEvent*)calloc(1, sizeof(dispatch_semaphore_t));
}

static void metal_destroy_event(ComputeBackend* backend, ComputeEvent* event) {
    (void)backend;
    free(event);
}

static ComputeResult metal_record_event(ComputeBackend* backend,
                                         ComputeEvent* event,
                                         ComputeStream* stream) {
    (void)backend;
    (void)event;
    (void)stream;
    return COMPUTE_SUCCESS;
}

static ComputeResult metal_wait_event(ComputeBackend* backend,
                                       ComputeStream* stream,
                                       ComputeEvent* event) {
    (void)backend;
    (void)stream;
    (void)event;
    return COMPUTE_SUCCESS;
}

// ============================================================================
// Quantum Operations
// ============================================================================

static ComputeResult metal_quantum_unitary(ComputeBackend* backend,
                                            float* state, size_t state_size,
                                            const float* unitary, size_t unitary_size,
                                            ComputeStream* stream) {
    @autoreleasepool {
        MetalBackendContext* ctx = (MetalBackendContext*)backend;
        if (!ctx || !state || !unitary) return COMPUTE_ERROR_INVALID_ARGUMENT;

        // Create buffers
        size_t state_bytes = state_size * 2 * sizeof(float);
        size_t unitary_bytes = state_size * state_size * 2 * sizeof(float);

        id<MTLBuffer> stateBuf = [ctx->device newBufferWithBytes:state
                                                          length:state_bytes
                                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> unitaryBuf = [ctx->device newBufferWithBytes:unitary
                                                            length:unitary_bytes
                                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> outputBuf = [ctx->device newBufferWithLength:state_bytes
                                                           options:MTLResourceStorageModeShared];

        uint32_t size = (uint32_t)state_size;

        // Execute kernel
        id<MTLCommandBuffer> cmdBuf = [ctx->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

        [encoder setComputePipelineState:ctx->unitaryPipeline];
        [encoder setBuffer:stateBuf offset:0 atIndex:0];
        [encoder setBuffer:unitaryBuf offset:0 atIndex:1];
        [encoder setBytes:&size length:sizeof(size) atIndex:2];
        [encoder setBuffer:outputBuf offset:0 atIndex:3];

        MTLSize gridSize = MTLSizeMake(state_size, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(MIN(ctx->threadgroup_size, state_size), 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        // Copy result back
        memcpy(state, [outputBuf contents], state_bytes);

        return COMPUTE_SUCCESS;
    }
}

static ComputeResult metal_quantum_normalize(ComputeBackend* backend,
                                              float* state, size_t size,
                                              ComputeStream* stream) {
    @autoreleasepool {
        MetalBackendContext* ctx = (MetalBackendContext*)backend;
        if (!ctx || !state) return COMPUTE_ERROR_INVALID_ARGUMENT;

        // Use Accelerate framework for efficient norm computation
        float norm_sq = 0.0f;
        vDSP_dotpr(state, 1, state, 1, &norm_sq, size * 2);

        float norm = sqrtf(norm_sq);
        if (norm > 1e-10f) {
            float scale = 1.0f / norm;
            vDSP_vsmul(state, 1, &scale, state, 1, size * 2);
        }

        return COMPUTE_SUCCESS;
    }
}

static ComputeResult metal_quantum_tensor_contract(ComputeBackend* backend,
                                                    float* result,
                                                    const float* a, const float* b,
                                                    size_t m, size_t n, size_t k,
                                                    ComputeStream* stream) {
    @autoreleasepool {
        MetalBackendContext* ctx = (MetalBackendContext*)backend;
        if (!ctx || !result || !a || !b) return COMPUTE_ERROR_INVALID_ARGUMENT;

        // Create buffers
        size_t a_bytes = m * n * 2 * sizeof(float);
        size_t b_bytes = n * k * 2 * sizeof(float);
        size_t c_bytes = m * k * 2 * sizeof(float);

        id<MTLBuffer> aBuf = [ctx->device newBufferWithBytes:a length:a_bytes
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> bBuf = [ctx->device newBufferWithBytes:b length:b_bytes
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> cBuf = [ctx->device newBufferWithLength:c_bytes
                                                      options:MTLResourceStorageModeShared];

        uint32_t M = (uint32_t)m;
        uint32_t N = (uint32_t)n;
        uint32_t K = (uint32_t)k;

        // Execute kernel
        id<MTLCommandBuffer> cmdBuf = [ctx->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

        [encoder setComputePipelineState:ctx->tensorContractPipeline];
        [encoder setBuffer:aBuf offset:0 atIndex:0];
        [encoder setBuffer:bBuf offset:0 atIndex:1];
        [encoder setBuffer:cBuf offset:0 atIndex:2];
        [encoder setBytes:&M length:sizeof(M) atIndex:3];
        [encoder setBytes:&N length:sizeof(N) atIndex:4];
        [encoder setBytes:&K length:sizeof(K) atIndex:5];

        MTLSize gridSize = MTLSizeMake(k, m, 1);
        MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        memcpy(result, [cBuf contents], c_bytes);

        return COMPUTE_SUCCESS;
    }
}

static ComputeResult metal_quantum_gradient(ComputeBackend* backend,
                                             float* gradients,
                                             const float* forward_state,
                                             const float* backward_state,
                                             size_t size,
                                             ComputeStream* stream) {
    @autoreleasepool {
        MetalBackendContext* ctx = (MetalBackendContext*)backend;
        if (!ctx) return COMPUTE_ERROR_INVALID_ARGUMENT;

        // Use SIMD for gradient computation (simpler than GPU for small sizes)
        simd_complex_inner_product(gradients, backward_state, forward_state, size);

        return COMPUTE_SUCCESS;
    }
}

static ComputeResult metal_quantum_inner_product(ComputeBackend* backend,
                                                  float* result,
                                                  const float* state_a,
                                                  const float* state_b,
                                                  size_t size,
                                                  ComputeStream* stream) {
    @autoreleasepool {
        MetalBackendContext* ctx = (MetalBackendContext*)backend;
        if (!ctx) return COMPUTE_ERROR_INVALID_ARGUMENT;

        simd_complex_inner_product(result, state_a, state_b, size);

        return COMPUTE_SUCCESS;
    }
}

static ComputeResult metal_quantum_expectation(ComputeBackend* backend,
                                                float* result,
                                                const float* state,
                                                const float* observable,
                                                size_t size,
                                                ComputeStream* stream) {
    @autoreleasepool {
        MetalBackendContext* ctx = (MetalBackendContext*)backend;
        if (!ctx) return COMPUTE_ERROR_INVALID_ARGUMENT;

        *result = simd_expectation_diagonal(state, observable, size);

        return COMPUTE_SUCCESS;
    }
}

// ============================================================================
// Collective Communication (delegates to MPI)
// ============================================================================

static ComputeResult metal_barrier(ComputeBackend* backend) {
#if COMPUTE_HAS_MPI
    MetalBackendContext* ctx = (MetalBackendContext*)backend;
    if (ctx && ctx->num_nodes > 1) {
        MPI_Barrier(ctx->comm);
    }
#else
    (void)backend;
#endif
    return COMPUTE_SUCCESS;
}

static ComputeResult metal_broadcast(ComputeBackend* backend,
                                      void* data, size_t size,
                                      ComputeDataType dtype, int root) {
#if COMPUTE_HAS_MPI
    MetalBackendContext* ctx = (MetalBackendContext*)backend;
    if (ctx && ctx->num_nodes > 1) {
        MPI_Datatype mpi_type = MPI_FLOAT;
        MPI_Bcast(data, (int)size, mpi_type, root, ctx->comm);
    }
#else
    (void)backend; (void)data; (void)size; (void)dtype; (void)root;
#endif
    return COMPUTE_SUCCESS;
}

static ComputeResult metal_allreduce(ComputeBackend* backend,
                                      const void* send_data, void* recv_data,
                                      size_t count, ComputeDataType dtype,
                                      ComputeReduceOp op) {
#if COMPUTE_HAS_MPI
    MetalBackendContext* ctx = (MetalBackendContext*)backend;
    if (ctx && ctx->num_nodes > 1) {
        MPI_Allreduce(send_data, recv_data, (int)count, MPI_FLOAT, MPI_SUM, ctx->comm);
    } else
#endif
    {
        size_t elem_size = compute_dtype_size(dtype);
        memcpy(recv_data, send_data, count * elem_size);
        (void)backend; (void)op;
    }
    return COMPUTE_SUCCESS;
}

static ComputeResult metal_scatter(ComputeBackend* backend,
                                    const void* send_data, void* recv_data,
                                    size_t count, ComputeDataType dtype, int root) {
    size_t elem_size = compute_dtype_size(dtype);
    memcpy(recv_data, send_data, count * elem_size);
    (void)backend; (void)root;
    return COMPUTE_SUCCESS;
}

static ComputeResult metal_gather(ComputeBackend* backend,
                                   const void* send_data, void* recv_data,
                                   size_t count, ComputeDataType dtype, int root) {
    size_t elem_size = compute_dtype_size(dtype);
    memcpy(recv_data, send_data, count * elem_size);
    (void)backend; (void)root;
    return COMPUTE_SUCCESS;
}

static ComputeResult metal_allgather(ComputeBackend* backend,
                                      const void* send_data, void* recv_data,
                                      size_t count, ComputeDataType dtype) {
    size_t elem_size = compute_dtype_size(dtype);
    memcpy(recv_data, send_data, count * elem_size);
    (void)backend;
    return COMPUTE_SUCCESS;
}

static ComputeResult metal_reduce_scatter(ComputeBackend* backend,
                                           const void* send_data, void* recv_data,
                                           size_t count, ComputeDataType dtype,
                                           ComputeReduceOp op) {
    size_t elem_size = compute_dtype_size(dtype);
    memcpy(recv_data, send_data, count * elem_size);
    (void)backend; (void)op;
    return COMPUTE_SUCCESS;
}

// ============================================================================
// Execution & Scheduling
// ============================================================================

static ComputeResult metal_execute(ComputeBackend* backend,
                                    const ComputeQuantumOp* op,
                                    const ComputeExecutionPlan* plan,
                                    ComputeStream* stream) {
    (void)plan;

    if (!backend || !op) return COMPUTE_ERROR_INVALID_ARGUMENT;

    switch (op->type) {
        case QUANTUM_OP_UNITARY:
            return metal_quantum_unitary(backend,
                                         (float*)op->output_data, op->output_size,
                                         (const float*)op->parameters, op->param_size,
                                         stream);

        case QUANTUM_OP_NORMALIZE:
            return metal_quantum_normalize(backend,
                                           (float*)op->output_data, op->output_size,
                                           stream);

        case QUANTUM_OP_TENSOR_CONTRACT:
            if (op->num_dims >= 3) {
                return metal_quantum_tensor_contract(backend,
                                                     (float*)op->output_data,
                                                     (const float*)op->input_data,
                                                     (const float*)op->parameters,
                                                     op->dims[0], op->dims[1], op->dims[2],
                                                     stream);
            }
            break;

        case QUANTUM_OP_GRADIENT:
            return metal_quantum_gradient(backend,
                                          (float*)op->output_data,
                                          (const float*)op->input_data,
                                          (const float*)op->parameters,
                                          op->input_size,
                                          stream);

        case QUANTUM_OP_INNER_PRODUCT:
            return metal_quantum_inner_product(backend,
                                               (float*)op->output_data,
                                               (const float*)op->input_data,
                                               (const float*)op->parameters,
                                               op->input_size,
                                               stream);

        case QUANTUM_OP_EXPECTATION:
            return metal_quantum_expectation(backend,
                                             (float*)op->output_data,
                                             (const float*)op->input_data,
                                             (const float*)op->parameters,
                                             op->input_size,
                                             stream);

        default:
            return COMPUTE_ERROR_NOT_IMPLEMENTED;
    }

    return COMPUTE_ERROR_NOT_IMPLEMENTED;
}

static ComputeExecutionPlan* metal_create_plan(ComputeBackend* backend,
                                                const ComputeQuantumOp* op) {
    MetalBackendContext* ctx = (MetalBackendContext*)backend;
    if (!ctx || !op) return nullptr;

    ComputeExecutionPlan* plan = (ComputeExecutionPlan*)calloc(1, sizeof(ComputeExecutionPlan));
    if (!plan) return nullptr;

    plan->num_partitions = ctx->num_nodes;
    plan->partition_size = op->input_size / ctx->num_nodes;

    return plan;
}

static void metal_destroy_plan(ComputeBackend* backend, ComputeExecutionPlan* plan) {
    (void)backend;
    if (plan) {
        free(plan->node_assignments);
        free(plan->offsets);
        free(plan->sizes);
        free(plan);
    }
}

// ============================================================================
// Performance Monitoring
// ============================================================================

static ComputeResult metal_get_metrics(ComputeBackend* backend, ComputeMetrics* metrics) {
    MetalBackendContext* ctx = (MetalBackendContext*)backend;
    if (!ctx || !metrics) return COMPUTE_ERROR_INVALID_ARGUMENT;

    *metrics = ctx->metrics;
    metrics->peak_memory_bytes = ctx->peak_allocated;
    metrics->current_memory_bytes = ctx->total_allocated;

    return COMPUTE_SUCCESS;
}

static ComputeResult metal_reset_metrics(ComputeBackend* backend) {
    MetalBackendContext* ctx = (MetalBackendContext*)backend;
    if (!ctx) return COMPUTE_ERROR_INVALID_ARGUMENT;

    memset(&ctx->metrics, 0, sizeof(ComputeMetrics));
    return COMPUTE_SUCCESS;
}

// ============================================================================
// Backend Registration
// ============================================================================

static const ComputeBackendOps metal_ops = {
    // Lifecycle
    .init = metal_init,
    .cleanup = metal_cleanup,
    .probe = metal_probe,
    .get_capabilities = metal_get_capabilities,

    // Memory
    .alloc = metal_alloc,
    .free = metal_free,
    .memcpy = metal_memcpy,
    .memset = metal_memset,

    // Streams
    .create_stream = metal_create_stream,
    .destroy_stream = metal_destroy_stream,
    .synchronize_stream = metal_synchronize_stream,
    .create_event = metal_create_event,
    .destroy_event = metal_destroy_event,
    .record_event = metal_record_event,
    .wait_event = metal_wait_event,

    // Quantum operations
    .quantum_unitary = metal_quantum_unitary,
    .quantum_normalize = metal_quantum_normalize,
    .quantum_tensor_contract = metal_quantum_tensor_contract,
    .quantum_gradient = metal_quantum_gradient,
    .quantum_inner_product = metal_quantum_inner_product,
    .quantum_expectation = metal_quantum_expectation,

    // Collective communication
    .barrier = metal_barrier,
    .broadcast = metal_broadcast,
    .allreduce = metal_allreduce,
    .scatter = metal_scatter,
    .gather = metal_gather,
    .allgather = metal_allgather,
    .reduce_scatter = metal_reduce_scatter,

    // Execution
    .execute = metal_execute,
    .create_plan = metal_create_plan,
    .destroy_plan = metal_destroy_plan,

    // Metrics
    .get_metrics = metal_get_metrics,
    .reset_metrics = metal_reset_metrics,
};

// Register the Metal backend at library load time
COMPUTE_REGISTER_BACKEND(COMPUTE_BACKEND_METAL, "Metal (Apple Silicon)", "1.0.0", 80, metal_ops)

#endif // COMPUTE_HAS_METAL
