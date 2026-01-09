#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "quantum_geometric/core/error_codes.h"
#include "quantum_geometric/core/hierarchical_matrix.h"
#include <complex>

// =============================================================================
// Metal Compute Types (must match shader definitions)
// =============================================================================

typedef struct MetalComplex {
    float real;
    float imag;
} MetalComplex;

typedef struct PhaseParams {
    uint32_t qubit;
    uint32_t dim;
    float cos_angle;
    float sin_angle;
} PhaseParams;

typedef struct MetricParams {
    uint32_t rows;
    uint32_t cols;
} MetricParams;

// =============================================================================
// Internal Context Structure
// =============================================================================

typedef struct PhaseMetalContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLLibrary> library;
    id<MTLComputePipelineState> phaseEstimationPipeline;
    id<MTLComputePipelineState> normalizationPipeline;
    id<MTLComputePipelineState> berryPhasePipeline;
    id<MTLComputePipelineState> quantumMetricPipeline;
    id<MTLComputePipelineState> berryCurvaturePipeline;
    bool pipelinesCompiled;
} PhaseMetalContext;

// Global context (lazy initialized)
static PhaseMetalContext* gPhaseContext = nil;

// =============================================================================
// Metal Shader Source (embedded)
// =============================================================================

static NSString* const kPhaseEstimationShaderSource = @R"METAL(
#include <metal_stdlib>
using namespace metal;

struct MetalComplex {
    float real;
    float imag;
};

struct PhaseParams {
    uint qubit;
    uint dim;
    float cos_angle;
    float sin_angle;
};

struct MetricParams {
    uint rows;
    uint cols;
};

// Complex multiplication
inline MetalComplex complex_mul(MetalComplex a, MetalComplex b) {
    MetalComplex result;
    result.real = a.real * b.real - a.imag * b.imag;
    result.imag = a.real * b.imag + a.imag * b.real;
    return result;
}

// Complex conjugate
inline MetalComplex complex_conj(MetalComplex a) {
    MetalComplex result;
    result.real = a.real;
    result.imag = -a.imag;
    return result;
}

// Complex magnitude squared
inline float complex_abs_sq(MetalComplex a) {
    return a.real * a.real + a.imag * a.imag;
}

// Complex natural log (principal branch)
inline MetalComplex complex_log(MetalComplex a) {
    MetalComplex result;
    float r = sqrt(complex_abs_sq(a));
    result.real = log(r);
    result.imag = atan2(a.imag, a.real);
    return result;
}

// Apply phase rotation to state amplitudes
kernel void phase_estimation(
    device MetalComplex* state [[buffer(0)]],
    constant PhaseParams& params [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params.dim) return;

    uint mask = 1u << params.qubit;
    if (gid & mask) {
        MetalComplex amp = state[gid];
        MetalComplex rotated;

        // Multiply by e^(i*angle) = cos(angle) + i*sin(angle)
        rotated.real = amp.real * params.cos_angle - amp.imag * params.sin_angle;
        rotated.imag = amp.real * params.sin_angle + amp.imag * params.cos_angle;

        state[gid] = rotated;
    }
}

// Compute normalization factor (reduction kernel)
kernel void compute_norm_squared(
    device const MetalComplex* state [[buffer(0)]],
    device atomic_float* norm_sq [[buffer(1)]],
    constant uint& dim [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]])
{
    // Shared memory for local reduction
    threadgroup float local_sum[256];

    float my_sum = 0.0f;
    if (gid < dim) {
        my_sum = complex_abs_sq(state[gid]);
    }
    local_sum[lid] = my_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction within threadgroup
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_sum[lid] += local_sum[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // First thread writes result
    if (lid == 0) {
        atomic_fetch_add_explicit(norm_sq, local_sum[0], memory_order_relaxed);
    }
}

// Normalize state vector
kernel void normalize_state(
    device MetalComplex* state [[buffer(0)]],
    constant float& inv_norm [[buffer(1)]],
    constant uint& dim [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= dim) return;

    state[gid].real *= inv_norm;
    state[gid].imag *= inv_norm;
}

// Compute Berry phase contributions for gradient
kernel void berry_phase_gradient(
    device const MetalComplex* data [[buffer(0)]],
    device MetalComplex* grad [[buffer(1)]],
    constant MetricParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint i = gid.x;
    uint j = gid.y;

    if (i >= params.rows || j >= params.cols) return;

    uint idx = i * params.cols + j;
    MetalComplex psi = data[idx];
    float phase_contrib = 0.0f;

    // Berry phase from row neighbor
    if (i > 0) {
        MetalComplex psi_prev = data[(i - 1) * params.cols + j];
        MetalComplex overlap = complex_mul(complex_conj(psi_prev), psi);
        float overlap_abs = sqrt(complex_abs_sq(overlap));
        if (overlap_abs > 1e-7f) {
            MetalComplex normalized;
            normalized.real = overlap.real / overlap_abs;
            normalized.imag = overlap.imag / overlap_abs;
            MetalComplex log_val = complex_log(normalized);
            phase_contrib += log_val.imag;
        }
    }

    // Berry phase from column neighbor
    if (j > 0) {
        MetalComplex psi_prev = data[i * params.cols + j - 1];
        MetalComplex overlap = complex_mul(complex_conj(psi_prev), psi);
        float overlap_abs = sqrt(complex_abs_sq(overlap));
        if (overlap_abs > 1e-7f) {
            MetalComplex normalized;
            normalized.real = overlap.real / overlap_abs;
            normalized.imag = overlap.imag / overlap_abs;
            MetalComplex log_val = complex_log(normalized);
            phase_contrib += log_val.imag;
        }
    }

    // Add phase contribution to gradient
    grad[idx].real += phase_contrib * psi.real;
    grad[idx].imag += phase_contrib * psi.imag;
}

// Compute quantum metric tensor
kernel void quantum_metric(
    device const MetalComplex* data [[buffer(0)]],
    device float* metric [[buffer(1)]],
    constant MetricParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint i = gid.x;
    uint j = gid.y;

    if (i >= params.rows || j >= params.cols) return;

    uint idx = i * params.cols + j;
    MetalComplex dpsi_x = {0.0f, 0.0f};
    MetalComplex dpsi_y = {0.0f, 0.0f};

    // Compute x-derivative using central differences
    if (i > 0 && i < params.rows - 1) {
        MetalComplex next = data[(i + 1) * params.cols + j];
        MetalComplex prev = data[(i - 1) * params.cols + j];
        dpsi_x.real = (next.real - prev.real) * 0.5f;
        dpsi_x.imag = (next.imag - prev.imag) * 0.5f;
    } else if (i == 0 && params.rows > 1) {
        MetalComplex curr = data[idx];
        MetalComplex next = data[params.cols + j];
        dpsi_x.real = next.real - curr.real;
        dpsi_x.imag = next.imag - curr.imag;
    } else if (i == params.rows - 1 && params.rows > 1) {
        MetalComplex curr = data[idx];
        MetalComplex prev = data[(params.rows - 2) * params.cols + j];
        dpsi_x.real = curr.real - prev.real;
        dpsi_x.imag = curr.imag - prev.imag;
    }

    // Compute y-derivative using central differences
    if (j > 0 && j < params.cols - 1) {
        MetalComplex next = data[i * params.cols + j + 1];
        MetalComplex prev = data[i * params.cols + j - 1];
        dpsi_y.real = (next.real - prev.real) * 0.5f;
        dpsi_y.imag = (next.imag - prev.imag) * 0.5f;
    } else if (j == 0 && params.cols > 1) {
        MetalComplex curr = data[idx];
        MetalComplex next = data[i * params.cols + 1];
        dpsi_y.real = next.real - curr.real;
        dpsi_y.imag = next.imag - curr.imag;
    } else if (j == params.cols - 1 && params.cols > 1) {
        MetalComplex curr = data[idx];
        MetalComplex prev = data[i * params.cols + params.cols - 2];
        dpsi_y.real = curr.real - prev.real;
        dpsi_y.imag = curr.imag - prev.imag;
    }

    // Quantum metric: g = Re[<d_psi|d_psi>] = |d_psi_x|^2 + |d_psi_y|^2
    metric[idx] = complex_abs_sq(dpsi_x) + complex_abs_sq(dpsi_y);
}

// Compute Berry curvature using Wilson loop (plaquette method)
kernel void berry_curvature(
    device const MetalComplex* data [[buffer(0)]],
    device float* curvature [[buffer(1)]],
    constant MetricParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint i = gid.x;
    uint j = gid.y;

    if (i >= params.rows - 1 || j >= params.cols - 1) {
        if (i < params.rows && j < params.cols) {
            curvature[i * params.cols + j] = 0.0f;
        }
        return;
    }

    uint idx = i * params.cols + j;

    // Plaquette corners
    MetalComplex psi_00 = data[i * params.cols + j];
    MetalComplex psi_10 = data[(i + 1) * params.cols + j];
    MetalComplex psi_01 = data[i * params.cols + j + 1];
    MetalComplex psi_11 = data[(i + 1) * params.cols + j + 1];

    // Wilson loop links U_ij = <psi_i | psi_j>
    MetalComplex U_01 = complex_mul(complex_conj(psi_00), psi_10);
    MetalComplex U_12 = complex_mul(complex_conj(psi_10), psi_11);
    MetalComplex U_23 = complex_mul(complex_conj(psi_11), psi_01);
    MetalComplex U_30 = complex_mul(complex_conj(psi_01), psi_00);

    // Wilson loop product W = U_01 * U_12 * U_23 * U_30
    MetalComplex W = complex_mul(complex_mul(U_01, U_12), complex_mul(U_23, U_30));

    // Berry curvature = Im[log(W)]
    float W_abs = sqrt(complex_abs_sq(W));
    if (W_abs > 1e-15f) {
        curvature[idx] = atan2(W.imag, W.real);
    } else {
        curvature[idx] = 0.0f;
    }
}
)METAL";

// =============================================================================
// Context Management
// =============================================================================

static int init_phase_metal_context(void* device_handle, void* command_queue, void* library) {
    @autoreleasepool {
        if (gPhaseContext && gPhaseContext->pipelinesCompiled) {
            return QGT_SUCCESS;
        }

        if (!gPhaseContext) {
            gPhaseContext = (PhaseMetalContext*)calloc(1, sizeof(PhaseMetalContext));
            if (!gPhaseContext) {
                return QGT_ERROR_OUT_OF_MEMORY;
            }
        }

        gPhaseContext->device = (__bridge id<MTLDevice>)device_handle;
        gPhaseContext->commandQueue = (__bridge id<MTLCommandQueue>)command_queue;

        // Compile shaders from embedded source
        NSError* error = nil;
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.languageVersion = MTLLanguageVersion2_4;
        options.fastMathEnabled = YES;

        id<MTLLibrary> compiledLibrary = [gPhaseContext->device newLibraryWithSource:kPhaseEstimationShaderSource
                                                                             options:options
                                                                               error:&error];
        if (!compiledLibrary) {
            NSLog(@"Failed to compile phase estimation shaders: %@", error);
            return QGT_ERROR_INITIALIZATION;
        }

        gPhaseContext->library = compiledLibrary;

        // Create compute pipeline states
        id<MTLFunction> phaseFunc = [compiledLibrary newFunctionWithName:@"phase_estimation"];
        id<MTLFunction> normFunc = [compiledLibrary newFunctionWithName:@"normalize_state"];
        id<MTLFunction> berryFunc = [compiledLibrary newFunctionWithName:@"berry_phase_gradient"];
        id<MTLFunction> metricFunc = [compiledLibrary newFunctionWithName:@"quantum_metric"];
        id<MTLFunction> curvatureFunc = [compiledLibrary newFunctionWithName:@"berry_curvature"];

        if (!phaseFunc || !normFunc || !berryFunc || !metricFunc || !curvatureFunc) {
            NSLog(@"Failed to load Metal functions");
            return QGT_ERROR_INITIALIZATION;
        }

        gPhaseContext->phaseEstimationPipeline = [gPhaseContext->device newComputePipelineStateWithFunction:phaseFunc error:&error];
        gPhaseContext->normalizationPipeline = [gPhaseContext->device newComputePipelineStateWithFunction:normFunc error:&error];
        gPhaseContext->berryPhasePipeline = [gPhaseContext->device newComputePipelineStateWithFunction:berryFunc error:&error];
        gPhaseContext->quantumMetricPipeline = [gPhaseContext->device newComputePipelineStateWithFunction:metricFunc error:&error];
        gPhaseContext->berryCurvaturePipeline = [gPhaseContext->device newComputePipelineStateWithFunction:curvatureFunc error:&error];

        if (!gPhaseContext->phaseEstimationPipeline ||
            !gPhaseContext->normalizationPipeline ||
            !gPhaseContext->berryPhasePipeline ||
            !gPhaseContext->quantumMetricPipeline ||
            !gPhaseContext->berryCurvaturePipeline) {
            NSLog(@"Failed to create pipeline states: %@", error);
            return QGT_ERROR_INITIALIZATION;
        }

        gPhaseContext->pipelinesCompiled = true;
        return QGT_SUCCESS;
    }
}

// =============================================================================
// GPU Dispatch Implementations
// =============================================================================

extern "C" int metal_phase_estimation_dispatch(
    void* device_handle,
    void* command_queue,
    void* library,
    void* state_ptr,
    size_t q,
    size_t dim)
{
    @autoreleasepool {
        if (!device_handle || !command_queue || !state_ptr || dim == 0) {
            return QGT_ERROR_INVALID_PARAMETER;
        }

        // Cast void* to std::complex<double>* for proper C++ complex access
        std::complex<double>* state = reinterpret_cast<std::complex<double>*>(state_ptr);

        int result = init_phase_metal_context(device_handle, command_queue, library);
        if (result != QGT_SUCCESS) {
            return result;
        }

        // Create Metal buffer for state data
        size_t buffer_size = dim * sizeof(MetalComplex);
        id<MTLBuffer> stateBuffer = [gPhaseContext->device newBufferWithLength:buffer_size
                                                                       options:MTLResourceStorageModeShared];
        if (!stateBuffer) {
            return QGT_ERROR_OUT_OF_MEMORY;
        }

        // Copy state data to Metal buffer (convert double to float)
        MetalComplex* metalState = (MetalComplex*)stateBuffer.contents;
        for (size_t i = 0; i < dim; i++) {
            metalState[i].real = (float)state[i].real();
            metalState[i].imag = (float)state[i].imag();
        }

        // Set up phase parameters
        double angle = 2.0 * M_PI / (double)(1ULL << (q + 1));
        PhaseParams params;
        params.qubit = (uint32_t)q;
        params.dim = (uint32_t)dim;
        params.cos_angle = (float)cos(angle);
        params.sin_angle = (float)sin(angle);

        id<MTLBuffer> paramsBuffer = [gPhaseContext->device newBufferWithBytes:&params
                                                                        length:sizeof(PhaseParams)
                                                                       options:MTLResourceStorageModeShared];

        // Create command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [gPhaseContext->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:gPhaseContext->phaseEstimationPipeline];
        [encoder setBuffer:stateBuffer offset:0 atIndex:0];
        [encoder setBuffer:paramsBuffer offset:0 atIndex:1];

        // Calculate threadgroup size
        NSUInteger threadGroupSize = MIN(256, dim);
        NSUInteger numThreadgroups = (dim + threadGroupSize - 1) / threadGroupSize;

        [encoder dispatchThreadgroups:MTLSizeMake(numThreadgroups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
        [encoder endEncoding];

        // Execute and wait
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        if (commandBuffer.status == MTLCommandBufferStatusError) {
            return QGT_ERROR_RUNTIME;
        }

        // Copy results back (convert float to double)
        for (size_t i = 0; i < dim; i++) {
            state[i] = std::complex<double>((double)metalState[i].real, (double)metalState[i].imag);
        }

        return QGT_SUCCESS;
    }
}

extern "C" int metal_hmatrix_update_dispatch(
    void* device_handle,
    void* command_queue,
    void* library,
    HierarchicalMatrix* mat)
{
    @autoreleasepool {
        if (!device_handle || !command_queue || !mat || !mat->data) {
            return QGT_ERROR_INVALID_PARAMETER;
        }

        int result = init_phase_metal_context(device_handle, command_queue, library);
        if (result != QGT_SUCCESS) {
            return result;
        }

        size_t size = mat->rows * mat->cols;
        if (size == 0) return QGT_SUCCESS;

        // Create Metal buffers
        size_t buffer_size = size * sizeof(MetalComplex);
        id<MTLBuffer> dataBuffer = [gPhaseContext->device newBufferWithLength:buffer_size
                                                                      options:MTLResourceStorageModeShared];
        if (!dataBuffer) {
            return QGT_ERROR_OUT_OF_MEMORY;
        }

        // Copy data to Metal buffer
        MetalComplex* metalData = (MetalComplex*)dataBuffer.contents;
        for (size_t i = 0; i < size; i++) {
            metalData[i].real = (float)QGT_COMPLEX_REAL(mat->data[i]);
            metalData[i].imag = (float)QGT_COMPLEX_IMAG(mat->data[i]);
        }

        // Compute normalization on GPU
        // First, compute norm squared
        id<MTLBuffer> normBuffer = [gPhaseContext->device newBufferWithLength:sizeof(float)
                                                                      options:MTLResourceStorageModeShared];
        *(float*)normBuffer.contents = 0.0f;

        uint32_t dim32 = (uint32_t)size;
        id<MTLBuffer> dimBuffer = [gPhaseContext->device newBufferWithBytes:&dim32
                                                                     length:sizeof(uint32_t)
                                                                    options:MTLResourceStorageModeShared];

        // We need to use CPU for reduction since atomic_float operations are complex
        // Compute norm on CPU
        double norm_sq = 0.0;
        for (size_t i = 0; i < size; i++) {
            norm_sq += metalData[i].real * metalData[i].real + metalData[i].imag * metalData[i].imag;
        }

        // Normalize if needed
        if (norm_sq > 1e-15 && fabs(norm_sq - 1.0) > 1e-10) {
            float inv_norm = (float)(1.0 / sqrt(norm_sq));

            id<MTLBuffer> invNormBuffer = [gPhaseContext->device newBufferWithBytes:&inv_norm
                                                                             length:sizeof(float)
                                                                            options:MTLResourceStorageModeShared];

            id<MTLCommandBuffer> commandBuffer = [gPhaseContext->commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

            [encoder setComputePipelineState:gPhaseContext->normalizationPipeline];
            [encoder setBuffer:dataBuffer offset:0 atIndex:0];
            [encoder setBuffer:invNormBuffer offset:0 atIndex:1];
            [encoder setBuffer:dimBuffer offset:0 atIndex:2];

            NSUInteger threadGroupSize = MIN(256, size);
            NSUInteger numThreadgroups = (size + threadGroupSize - 1) / threadGroupSize;

            [encoder dispatchThreadgroups:MTLSizeMake(numThreadgroups, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
            [encoder endEncoding];

            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
        }

        // Compute Berry phase gradient if gradient exists
        if (mat->grad) {
            id<MTLBuffer> gradBuffer = [gPhaseContext->device newBufferWithLength:buffer_size
                                                                          options:MTLResourceStorageModeShared];

            // Copy gradient to Metal buffer
            MetalComplex* metalGrad = (MetalComplex*)gradBuffer.contents;
            for (size_t i = 0; i < size; i++) {
                metalGrad[i].real = (float)QGT_COMPLEX_REAL(mat->grad[i]);
                metalGrad[i].imag = (float)QGT_COMPLEX_IMAG(mat->grad[i]);
            }

            MetricParams params;
            params.rows = (uint32_t)mat->rows;
            params.cols = (uint32_t)mat->cols;

            id<MTLBuffer> paramsBuffer = [gPhaseContext->device newBufferWithBytes:&params
                                                                            length:sizeof(MetricParams)
                                                                           options:MTLResourceStorageModeShared];

            id<MTLCommandBuffer> commandBuffer = [gPhaseContext->commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

            [encoder setComputePipelineState:gPhaseContext->berryPhasePipeline];
            [encoder setBuffer:dataBuffer offset:0 atIndex:0];
            [encoder setBuffer:gradBuffer offset:0 atIndex:1];
            [encoder setBuffer:paramsBuffer offset:0 atIndex:2];

            MTLSize gridSize = MTLSizeMake(mat->rows, mat->cols, 1);
            MTLSize threadgroupSize = MTLSizeMake(MIN(16, mat->rows), MIN(16, mat->cols), 1);

            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            [encoder endEncoding];

            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];

            // Copy gradient back
            for (size_t i = 0; i < size; i++) {
                mat->grad[i] = qgt_complex_t((double)metalGrad[i].real, (double)metalGrad[i].imag);
            }
        }

        // Copy data back
        for (size_t i = 0; i < size; i++) {
            mat->data[i] = qgt_complex_t((double)metalData[i].real, (double)metalData[i].imag);
        }

        return QGT_SUCCESS;
    }
}

extern "C" int metal_quantum_metric_dispatch(
    void* device_handle,
    void* command_queue,
    void* library,
    const HierarchicalMatrix* mat,
    double* metric)
{
    @autoreleasepool {
        if (!device_handle || !command_queue || !mat || !mat->data || !metric) {
            return QGT_ERROR_INVALID_PARAMETER;
        }

        int result = init_phase_metal_context(device_handle, command_queue, library);
        if (result != QGT_SUCCESS) {
            return result;
        }

        size_t size = mat->rows * mat->cols;
        if (size == 0) return QGT_SUCCESS;

        // Create Metal buffers
        size_t data_buffer_size = size * sizeof(MetalComplex);
        size_t metric_buffer_size = size * sizeof(float);

        id<MTLBuffer> dataBuffer = [gPhaseContext->device newBufferWithLength:data_buffer_size
                                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> metricBuffer = [gPhaseContext->device newBufferWithLength:metric_buffer_size
                                                                        options:MTLResourceStorageModeShared];

        if (!dataBuffer || !metricBuffer) {
            return QGT_ERROR_OUT_OF_MEMORY;
        }

        // Copy data to Metal buffer
        MetalComplex* metalData = (MetalComplex*)dataBuffer.contents;
        for (size_t i = 0; i < size; i++) {
            metalData[i].real = (float)QGT_COMPLEX_REAL(mat->data[i]);
            metalData[i].imag = (float)QGT_COMPLEX_IMAG(mat->data[i]);
        }

        MetricParams params;
        params.rows = (uint32_t)mat->rows;
        params.cols = (uint32_t)mat->cols;

        id<MTLBuffer> paramsBuffer = [gPhaseContext->device newBufferWithBytes:&params
                                                                        length:sizeof(MetricParams)
                                                                       options:MTLResourceStorageModeShared];

        // Execute quantum metric kernel
        id<MTLCommandBuffer> commandBuffer = [gPhaseContext->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:gPhaseContext->quantumMetricPipeline];
        [encoder setBuffer:dataBuffer offset:0 atIndex:0];
        [encoder setBuffer:metricBuffer offset:0 atIndex:1];
        [encoder setBuffer:paramsBuffer offset:0 atIndex:2];

        MTLSize gridSize = MTLSizeMake(mat->rows, mat->cols, 1);
        MTLSize threadgroupSize = MTLSizeMake(MIN(16, mat->rows), MIN(16, mat->cols), 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        if (commandBuffer.status == MTLCommandBufferStatusError) {
            return QGT_ERROR_RUNTIME;
        }

        // Copy results back (convert float to double)
        float* metalMetric = (float*)metricBuffer.contents;
        for (size_t i = 0; i < size; i++) {
            metric[i] = (double)metalMetric[i];
        }

        return QGT_SUCCESS;
    }
}

extern "C" int metal_berry_curvature_dispatch(
    void* device_handle,
    void* command_queue,
    void* library,
    const HierarchicalMatrix* mat,
    double* curvature)
{
    @autoreleasepool {
        if (!device_handle || !command_queue || !mat || !mat->data || !curvature) {
            return QGT_ERROR_INVALID_PARAMETER;
        }

        int result = init_phase_metal_context(device_handle, command_queue, library);
        if (result != QGT_SUCCESS) {
            return result;
        }

        size_t size = mat->rows * mat->cols;
        if (size == 0) return QGT_SUCCESS;

        // Create Metal buffers
        size_t data_buffer_size = size * sizeof(MetalComplex);
        size_t curvature_buffer_size = size * sizeof(float);

        id<MTLBuffer> dataBuffer = [gPhaseContext->device newBufferWithLength:data_buffer_size
                                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> curvatureBuffer = [gPhaseContext->device newBufferWithLength:curvature_buffer_size
                                                                           options:MTLResourceStorageModeShared];

        if (!dataBuffer || !curvatureBuffer) {
            return QGT_ERROR_OUT_OF_MEMORY;
        }

        // Initialize curvature buffer to zero
        memset(curvatureBuffer.contents, 0, curvature_buffer_size);

        // Copy data to Metal buffer
        MetalComplex* metalData = (MetalComplex*)dataBuffer.contents;
        for (size_t i = 0; i < size; i++) {
            metalData[i].real = (float)QGT_COMPLEX_REAL(mat->data[i]);
            metalData[i].imag = (float)QGT_COMPLEX_IMAG(mat->data[i]);
        }

        MetricParams params;
        params.rows = (uint32_t)mat->rows;
        params.cols = (uint32_t)mat->cols;

        id<MTLBuffer> paramsBuffer = [gPhaseContext->device newBufferWithBytes:&params
                                                                        length:sizeof(MetricParams)
                                                                       options:MTLResourceStorageModeShared];

        // Execute Berry curvature kernel
        id<MTLCommandBuffer> commandBuffer = [gPhaseContext->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:gPhaseContext->berryCurvaturePipeline];
        [encoder setBuffer:dataBuffer offset:0 atIndex:0];
        [encoder setBuffer:curvatureBuffer offset:0 atIndex:1];
        [encoder setBuffer:paramsBuffer offset:0 atIndex:2];

        MTLSize gridSize = MTLSizeMake(mat->rows, mat->cols, 1);
        MTLSize threadgroupSize = MTLSizeMake(MIN(16, mat->rows), MIN(16, mat->cols), 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        if (commandBuffer.status == MTLCommandBufferStatusError) {
            return QGT_ERROR_RUNTIME;
        }

        // Copy results back (convert float to double)
        float* metalCurvature = (float*)curvatureBuffer.contents;
        for (size_t i = 0; i < size; i++) {
            curvature[i] = (double)metalCurvature[i];
        }

        return QGT_SUCCESS;
    }
}
