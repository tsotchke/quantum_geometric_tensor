#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "quantum_geometric/hardware/metal/quantum_geometric_syndrome.h"
#include "quantum_geometric/core/error_codes.h"
#include <mach/mach_time.h>
#include <string.h>
#include <stdlib.h>

// =============================================================================
// Internal Context Structure
// =============================================================================

typedef struct SyndromeMetalContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLLibrary> library;
    id<MTLComputePipelineState> extractPipeline;
    id<MTLComputePipelineState> decodePipeline;
    id<MTLComputePipelineState> patternPipeline;
    id<MTLComputePipelineState> graphPipeline;
    SyndromeConfig config;
    bool pipelinesCompiled;

    // Statistics tracking
    size_t total_extracted;
    size_t total_decoded;
    double total_chain_length;
    size_t num_results;

    // Timing
    mach_timebase_info_data_t timebase;
} SyndromeMetalContext;

// =============================================================================
// Metal Shader Source (embedded)
// =============================================================================

static NSString* const kSyndromeShaderSource = @R"METAL(
#include <metal_stdlib>
using namespace metal;

struct SyndromeCoord {
    uint x;
    uint y;
    uint z;
};

struct SyndromeVertex {
    SyndromeCoord position;
    float weight;
    bool is_boundary;
    uint timestamp;
    float confidence;
    float correlation_weight;
    bool part_of_chain;
    uint matched_to;
};

struct SyndromeConfig {
    float detection_threshold;
    float confidence_threshold;
    float weight_scale_factor;
    float pattern_threshold;
    uint parallel_group_size;
    uint min_pattern_occurrences;
    bool enable_parallel;
    bool use_boundary_matching;
    uint max_iterations;
    uint code_distance;
};

// Extract syndromes from stabilizer measurements
kernel void extract_syndromes(
    device const float2* measurements [[buffer(0)]],
    device SyndromeVertex* vertices [[buffer(1)]],
    constant SyndromeConfig& config [[buffer(2)]],
    constant uint& num_measurements [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= num_measurements) return;

    float2 measurement = measurements[gid];
    float magnitude = length(measurement);

    // Initialize vertex
    SyndromeVertex vertex;
    vertex.position.x = gid % config.code_distance;
    vertex.position.y = (gid / config.code_distance) % config.code_distance;
    vertex.position.z = gid / (config.code_distance * config.code_distance);
    vertex.timestamp = vertex.position.z;
    vertex.is_boundary = (vertex.position.x == 0 || vertex.position.y == 0 ||
                          vertex.position.x == config.code_distance - 1 ||
                          vertex.position.y == config.code_distance - 1);
    vertex.matched_to = 0xFFFFFFFF;
    vertex.part_of_chain = false;

    // Detect syndrome based on measurement
    if (magnitude > config.detection_threshold) {
        vertex.weight = magnitude * config.weight_scale_factor;
        vertex.confidence = clamp(magnitude / (magnitude + 0.1f), 0.0f, 1.0f);
        vertex.correlation_weight = vertex.weight;
    } else {
        vertex.weight = 0.0f;
        vertex.confidence = 0.0f;
        vertex.correlation_weight = 0.0f;
    }

    vertices[gid] = vertex;
}

// Build syndrome graph edges
kernel void build_graph_edges(
    device const SyndromeVertex* vertices [[buffer(0)]],
    device float* edge_weights [[buffer(1)]],
    constant uint& num_vertices [[buffer(2)]],
    constant SyndromeConfig& config [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint i = gid.x;
    uint j = gid.y;

    if (i >= num_vertices || j >= num_vertices || i >= j) {
        if (i < num_vertices && j < num_vertices) {
            edge_weights[i * num_vertices + j] = INFINITY;
        }
        return;
    }

    SyndromeVertex v1 = vertices[i];
    SyndromeVertex v2 = vertices[j];

    // Only connect vertices with non-zero weight
    if (v1.weight < config.detection_threshold && v2.weight < config.detection_threshold) {
        edge_weights[i * num_vertices + j] = INFINITY;
        edge_weights[j * num_vertices + i] = INFINITY;
        return;
    }

    // Compute Manhattan distance
    int dx = abs((int)v1.position.x - (int)v2.position.x);
    int dy = abs((int)v1.position.y - (int)v2.position.y);
    int dz = abs((int)v1.position.z - (int)v2.position.z);
    int distance = dx + dy + dz;

    // Edge weight based on distance and vertex weights
    float weight = (float)distance * config.weight_scale_factor;
    if (v1.weight > 0 && v2.weight > 0) {
        weight *= 1.0f / (v1.confidence * v2.confidence + 0.01f);
    }

    // Boundary edges have lower weight to prefer boundary matching
    if (v1.is_boundary || v2.is_boundary) {
        if (config.use_boundary_matching) {
            weight *= 0.5f;
        }
    }

    edge_weights[i * num_vertices + j] = weight;
    edge_weights[j * num_vertices + i] = weight;
}

// Simple greedy matching for syndromes
kernel void greedy_match(
    device SyndromeVertex* vertices [[buffer(0)]],
    device const float* edge_weights [[buffer(1)]],
    constant uint& num_vertices [[buffer(2)]],
    constant SyndromeConfig& config [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= num_vertices) return;

    SyndromeVertex vertex = vertices[gid];
    if (vertex.weight < config.detection_threshold || vertex.matched_to != 0xFFFFFFFF) {
        return;
    }

    // Find best match
    float best_weight = INFINITY;
    uint best_match = 0xFFFFFFFF;

    for (uint j = 0; j < num_vertices; j++) {
        if (j == gid) continue;

        SyndromeVertex other = vertices[j];
        if (other.weight < config.detection_threshold || other.matched_to != 0xFFFFFFFF) {
            continue;
        }

        float weight = edge_weights[gid * num_vertices + j];
        if (weight < best_weight) {
            best_weight = weight;
            best_match = j;
        }
    }

    // Match to boundary if no pair found and boundary matching enabled
    if (best_match == 0xFFFFFFFF && config.use_boundary_matching && vertex.is_boundary) {
        vertex.matched_to = gid;  // Self-match indicates boundary
        vertex.part_of_chain = true;
    } else if (best_match != 0xFFFFFFFF) {
        // Only match if we're the lower index to avoid race conditions
        if (gid < best_match) {
            vertex.matched_to = best_match;
            vertex.part_of_chain = true;
        }
    }

    vertices[gid] = vertex;
}

// Detect patterns in syndrome data
kernel void detect_patterns(
    device const SyndromeVertex* vertices [[buffer(0)]],
    device uint* pattern_counts [[buffer(1)]],
    constant uint& num_vertices [[buffer(2)]],
    constant SyndromeConfig& config [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= num_vertices) return;

    SyndromeVertex vertex = vertices[gid];
    if (vertex.weight < config.detection_threshold) return;

    // Simple pattern: count syndromes in local neighborhood
    uint pattern_id = 0;
    for (uint i = 0; i < num_vertices; i++) {
        SyndromeVertex other = vertices[i];
        if (other.weight < config.detection_threshold) continue;

        int dx = abs((int)vertex.position.x - (int)other.position.x);
        int dy = abs((int)vertex.position.y - (int)other.position.y);

        if (dx <= 1 && dy <= 1) {
            pattern_id |= (1u << (dx * 3 + dy));
        }
    }

    // Atomic increment pattern count
    atomic_fetch_add_explicit((device atomic_uint*)&pattern_counts[pattern_id % 256], 1u, memory_order_relaxed);
}
)METAL";

// =============================================================================
// Timing Utilities
// =============================================================================

static double get_elapsed_seconds(uint64_t start, uint64_t end, mach_timebase_info_data_t* timebase) {
    uint64_t elapsed = end - start;
    return (double)elapsed * timebase->numer / timebase->denom / 1e9;
}

// =============================================================================
// Context Management
// =============================================================================

extern "C" void* syndrome_create_context(const SyndromeConfig* config) {
    @autoreleasepool {
        if (!config) return NULL;

        SyndromeMetalContext* ctx = (SyndromeMetalContext*)calloc(1, sizeof(SyndromeMetalContext));
        if (!ctx) return NULL;

        // Get Metal device
        ctx->device = MTLCreateSystemDefaultDevice();
        if (!ctx->device) {
            free(ctx);
            return NULL;
        }

        // Create command queue
        ctx->commandQueue = [ctx->device newCommandQueue];
        if (!ctx->commandQueue) {
            free(ctx);
            return NULL;
        }

        // Compile shaders
        NSError* error = nil;
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.languageVersion = MTLLanguageVersion2_4;
        options.fastMathEnabled = YES;

        ctx->library = [ctx->device newLibraryWithSource:kSyndromeShaderSource
                                                 options:options
                                                   error:&error];
        if (!ctx->library) {
            NSLog(@"Failed to compile syndrome shaders: %@", error);
            free(ctx);
            return NULL;
        }

        // Create pipeline states
        id<MTLFunction> extractFunc = [ctx->library newFunctionWithName:@"extract_syndromes"];
        id<MTLFunction> graphFunc = [ctx->library newFunctionWithName:@"build_graph_edges"];
        id<MTLFunction> matchFunc = [ctx->library newFunctionWithName:@"greedy_match"];
        id<MTLFunction> patternFunc = [ctx->library newFunctionWithName:@"detect_patterns"];

        if (!extractFunc || !graphFunc || !matchFunc || !patternFunc) {
            NSLog(@"Failed to load syndrome Metal functions");
            free(ctx);
            return NULL;
        }

        ctx->extractPipeline = [ctx->device newComputePipelineStateWithFunction:extractFunc error:&error];
        ctx->graphPipeline = [ctx->device newComputePipelineStateWithFunction:graphFunc error:&error];
        ctx->decodePipeline = [ctx->device newComputePipelineStateWithFunction:matchFunc error:&error];
        ctx->patternPipeline = [ctx->device newComputePipelineStateWithFunction:patternFunc error:&error];

        if (!ctx->extractPipeline || !ctx->graphPipeline || !ctx->decodePipeline || !ctx->patternPipeline) {
            NSLog(@"Failed to create syndrome pipeline states: %@", error);
            free(ctx);
            return NULL;
        }

        ctx->config = *config;
        ctx->pipelinesCompiled = true;
        mach_timebase_info(&ctx->timebase);

        return ctx;
    }
}

extern "C" void syndrome_destroy_context(void* context) {
    if (!context) return;

    @autoreleasepool {
        SyndromeMetalContext* ctx = (SyndromeMetalContext*)context;

        // ARC handles Metal object cleanup
        ctx->extractPipeline = nil;
        ctx->graphPipeline = nil;
        ctx->decodePipeline = nil;
        ctx->patternPipeline = nil;
        ctx->library = nil;
        ctx->commandQueue = nil;
        ctx->device = nil;

        free(ctx);
    }
}

// =============================================================================
// Syndrome Extraction
// =============================================================================

extern "C" int syndrome_extract(void* context,
                                const float2* measurements,
                                size_t num_measurements,
                                SyndromeResult* result) {
    @autoreleasepool {
        if (!context || !measurements || !result || num_measurements == 0) {
            return QGT_ERROR_INVALID_PARAMETER;
        }

        SyndromeMetalContext* ctx = (SyndromeMetalContext*)context;
        if (!ctx->pipelinesCompiled) {
            return QGT_ERROR_NOT_INITIALIZED;
        }

        uint64_t start_time = mach_absolute_time();

        // Allocate result arrays
        result->vertices = (SyndromeVertex*)calloc(num_measurements, sizeof(SyndromeVertex));
        if (!result->vertices) {
            return QGT_ERROR_OUT_OF_MEMORY;
        }

        // Create Metal buffers
        size_t measurement_size = num_measurements * sizeof(float) * 2;  // float2
        size_t vertex_size = num_measurements * sizeof(SyndromeVertex);

        id<MTLBuffer> measurementBuffer = [ctx->device newBufferWithBytes:measurements
                                                                   length:measurement_size
                                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> vertexBuffer = [ctx->device newBufferWithLength:vertex_size
                                                              options:MTLResourceStorageModeShared];
        id<MTLBuffer> configBuffer = [ctx->device newBufferWithBytes:&ctx->config
                                                              length:sizeof(SyndromeConfig)
                                                             options:MTLResourceStorageModeShared];
        uint32_t num_meas32 = (uint32_t)num_measurements;
        id<MTLBuffer> countBuffer = [ctx->device newBufferWithBytes:&num_meas32
                                                             length:sizeof(uint32_t)
                                                            options:MTLResourceStorageModeShared];

        if (!measurementBuffer || !vertexBuffer || !configBuffer || !countBuffer) {
            free(result->vertices);
            return QGT_ERROR_OUT_OF_MEMORY;
        }

        // Create command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [ctx->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:ctx->extractPipeline];
        [encoder setBuffer:measurementBuffer offset:0 atIndex:0];
        [encoder setBuffer:vertexBuffer offset:0 atIndex:1];
        [encoder setBuffer:configBuffer offset:0 atIndex:2];
        [encoder setBuffer:countBuffer offset:0 atIndex:3];

        NSUInteger threadGroupSize = MIN(256, num_measurements);
        NSUInteger numThreadgroups = (num_measurements + threadGroupSize - 1) / threadGroupSize;

        [encoder dispatchThreadgroups:MTLSizeMake(numThreadgroups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
        [encoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        if (commandBuffer.status == MTLCommandBufferStatusError) {
            free(result->vertices);
            return QGT_ERROR_RUNTIME;
        }

        // Copy results
        memcpy(result->vertices, vertexBuffer.contents, vertex_size);
        result->num_vertices = num_measurements;

        // Count active syndromes and update statistics
        size_t active_count = 0;
        for (size_t i = 0; i < num_measurements; i++) {
            if (result->vertices[i].weight > ctx->config.detection_threshold) {
                active_count++;
            }
        }

        ctx->total_extracted += active_count;

        uint64_t end_time = mach_absolute_time();
        result->extraction_time = get_elapsed_seconds(start_time, end_time, &ctx->timebase);
        result->decoding_time = 0.0;
        result->edges = NULL;
        result->num_edges = 0;
        result->error_chain = NULL;
        result->chain_length = 0;
        result->logical_error_rate = 0.0f;

        return QGT_SUCCESS;
    }
}

extern "C" int syndrome_extract_rounds(void* context,
                                       const float2** measurements,
                                       size_t num_rounds,
                                       size_t measurements_per_round,
                                       SyndromeResult* result) {
    if (!context || !measurements || !result || num_rounds == 0 || measurements_per_round == 0) {
        return QGT_ERROR_INVALID_PARAMETER;
    }

    // Flatten measurements into single array
    size_t total_measurements = num_rounds * measurements_per_round;
    float2* flat_measurements = (float2*)malloc(total_measurements * sizeof(float2));
    if (!flat_measurements) {
        return QGT_ERROR_OUT_OF_MEMORY;
    }

    for (size_t r = 0; r < num_rounds; r++) {
        memcpy(&flat_measurements[r * measurements_per_round],
               measurements[r],
               measurements_per_round * sizeof(float2));
    }

    int status = syndrome_extract(context, flat_measurements, total_measurements, result);
    free(flat_measurements);

    return status;
}

// =============================================================================
// Syndrome Decoding
// =============================================================================

extern "C" int syndrome_decode_mwpm(void* context, SyndromeResult* result) {
    @autoreleasepool {
        if (!context || !result || !result->vertices) {
            return QGT_ERROR_INVALID_PARAMETER;
        }

        SyndromeMetalContext* ctx = (SyndromeMetalContext*)context;

        uint64_t start_time = mach_absolute_time();

        size_t num_vertices = result->num_vertices;
        size_t edge_matrix_size = num_vertices * num_vertices * sizeof(float);

        // Create buffers
        id<MTLBuffer> vertexBuffer = [ctx->device newBufferWithBytes:result->vertices
                                                              length:num_vertices * sizeof(SyndromeVertex)
                                                             options:MTLResourceStorageModeShared];
        id<MTLBuffer> edgeBuffer = [ctx->device newBufferWithLength:edge_matrix_size
                                                            options:MTLResourceStorageModeShared];
        uint32_t num_verts32 = (uint32_t)num_vertices;
        id<MTLBuffer> countBuffer = [ctx->device newBufferWithBytes:&num_verts32
                                                             length:sizeof(uint32_t)
                                                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> configBuffer = [ctx->device newBufferWithBytes:&ctx->config
                                                              length:sizeof(SyndromeConfig)
                                                             options:MTLResourceStorageModeShared];

        if (!vertexBuffer || !edgeBuffer || !countBuffer || !configBuffer) {
            return QGT_ERROR_OUT_OF_MEMORY;
        }

        // Build graph edges
        id<MTLCommandBuffer> commandBuffer = [ctx->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:ctx->graphPipeline];
        [encoder setBuffer:vertexBuffer offset:0 atIndex:0];
        [encoder setBuffer:edgeBuffer offset:0 atIndex:1];
        [encoder setBuffer:countBuffer offset:0 atIndex:2];
        [encoder setBuffer:configBuffer offset:0 atIndex:3];

        MTLSize gridSize = MTLSizeMake(num_vertices, num_vertices, 1);
        MTLSize threadgroupSize = MTLSizeMake(MIN(16, num_vertices), MIN(16, num_vertices), 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Run greedy matching (iteratively to handle conflicts)
        for (uint32_t iter = 0; iter < ctx->config.max_iterations; iter++) {
            commandBuffer = [ctx->commandQueue commandBuffer];
            encoder = [commandBuffer computeCommandEncoder];

            [encoder setComputePipelineState:ctx->decodePipeline];
            [encoder setBuffer:vertexBuffer offset:0 atIndex:0];
            [encoder setBuffer:edgeBuffer offset:0 atIndex:1];
            [encoder setBuffer:countBuffer offset:0 atIndex:2];
            [encoder setBuffer:configBuffer offset:0 atIndex:3];

            NSUInteger threadGroupSize = MIN(256, num_vertices);
            NSUInteger numThreadgroups = (num_vertices + threadGroupSize - 1) / threadGroupSize;

            [encoder dispatchThreadgroups:MTLSizeMake(numThreadgroups, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
            [encoder endEncoding];

            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
        }

        // Copy results back
        memcpy(result->vertices, vertexBuffer.contents, num_vertices * sizeof(SyndromeVertex));

        // Build error chain from matched vertices
        size_t chain_capacity = num_vertices;
        result->error_chain = (uint32_t*)malloc(chain_capacity * sizeof(uint32_t));
        result->chain_length = 0;

        if (result->error_chain) {
            for (size_t i = 0; i < num_vertices; i++) {
                if (result->vertices[i].part_of_chain && result->chain_length < chain_capacity) {
                    result->error_chain[result->chain_length++] = (uint32_t)i;
                }
            }
        }

        // Update statistics
        ctx->total_decoded += result->chain_length;
        ctx->total_chain_length += result->chain_length;
        ctx->num_results++;

        uint64_t end_time = mach_absolute_time();
        result->decoding_time = get_elapsed_seconds(start_time, end_time, &ctx->timebase);

        return QGT_SUCCESS;
    }
}

extern "C" int syndrome_decode_union_find(void* context, SyndromeResult* result) {
    // Use MWPM as fallback - Union-Find can be added as optimization
    return syndrome_decode_mwpm(context, result);
}

extern "C" int syndrome_decode_belief_propagation(void* context,
                                                  SyndromeResult* result,
                                                  uint32_t max_iterations) {
    if (!context || !result) {
        return QGT_ERROR_INVALID_PARAMETER;
    }

    // Override max iterations
    SyndromeMetalContext* ctx = (SyndromeMetalContext*)context;
    uint32_t saved_iterations = ctx->config.max_iterations;
    ctx->config.max_iterations = max_iterations;

    int status = syndrome_decode_mwpm(context, result);

    ctx->config.max_iterations = saved_iterations;
    return status;
}

// =============================================================================
// Pattern Detection
// =============================================================================

extern "C" size_t syndrome_detect_patterns(void* context,
                                           const SyndromeResult* result,
                                           SyndromePattern* patterns,
                                           size_t max_patterns) {
    @autoreleasepool {
        if (!context || !result || !patterns || max_patterns == 0) {
            return 0;
        }

        SyndromeMetalContext* ctx = (SyndromeMetalContext*)context;
        size_t num_vertices = result->num_vertices;

        // Create buffers
        id<MTLBuffer> vertexBuffer = [ctx->device newBufferWithBytes:result->vertices
                                                              length:num_vertices * sizeof(SyndromeVertex)
                                                             options:MTLResourceStorageModeShared];

        const size_t PATTERN_BUCKETS = 256;
        id<MTLBuffer> patternBuffer = [ctx->device newBufferWithLength:PATTERN_BUCKETS * sizeof(uint32_t)
                                                               options:MTLResourceStorageModeShared];
        memset(patternBuffer.contents, 0, PATTERN_BUCKETS * sizeof(uint32_t));

        uint32_t num_verts32 = (uint32_t)num_vertices;
        id<MTLBuffer> countBuffer = [ctx->device newBufferWithBytes:&num_verts32
                                                             length:sizeof(uint32_t)
                                                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> configBuffer = [ctx->device newBufferWithBytes:&ctx->config
                                                              length:sizeof(SyndromeConfig)
                                                             options:MTLResourceStorageModeShared];

        if (!vertexBuffer || !patternBuffer || !countBuffer || !configBuffer) {
            return 0;
        }

        // Run pattern detection
        id<MTLCommandBuffer> commandBuffer = [ctx->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:ctx->patternPipeline];
        [encoder setBuffer:vertexBuffer offset:0 atIndex:0];
        [encoder setBuffer:patternBuffer offset:0 atIndex:1];
        [encoder setBuffer:countBuffer offset:0 atIndex:2];
        [encoder setBuffer:configBuffer offset:0 atIndex:3];

        NSUInteger threadGroupSize = MIN(256, num_vertices);
        NSUInteger numThreadgroups = (num_vertices + threadGroupSize - 1) / threadGroupSize;

        [encoder dispatchThreadgroups:MTLSizeMake(numThreadgroups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
        [encoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Extract top patterns
        uint32_t* counts = (uint32_t*)patternBuffer.contents;
        size_t num_patterns = 0;

        for (size_t i = 0; i < PATTERN_BUCKETS && num_patterns < max_patterns; i++) {
            if (counts[i] >= ctx->config.min_pattern_occurrences) {
                patterns[num_patterns].vertex_indices = NULL;
                patterns[num_patterns].pattern_size = 0;
                patterns[num_patterns].occurrence_count = counts[i];
                patterns[num_patterns].occurrence_probability = (float)counts[i] / (float)num_vertices;
                patterns[num_patterns].is_logical_error = (counts[i] > num_vertices / 4);
                num_patterns++;
            }
        }

        return num_patterns;
    }
}

extern "C" int syndrome_apply_patterns(void* context,
                                       SyndromeResult* result,
                                       const SyndromePattern* patterns,
                                       size_t num_patterns) {
    if (!context || !result || !patterns) {
        return QGT_ERROR_INVALID_PARAMETER;
    }

    // Apply pattern-based corrections to vertex weights
    for (size_t p = 0; p < num_patterns; p++) {
        if (patterns[p].is_logical_error) {
            result->logical_error_rate += patterns[p].occurrence_probability;
        }
    }

    return QGT_SUCCESS;
}

// =============================================================================
// Graph Construction
// =============================================================================

extern "C" int syndrome_build_graph(void* context,
                                    const SyndromeVertex* vertices,
                                    size_t num_vertices,
                                    const SyndromeConfig* config,
                                    SyndromeEdge** edges,
                                    size_t* num_edges) {
    if (!context || !vertices || !config || !edges || !num_edges) {
        return QGT_ERROR_INVALID_PARAMETER;
    }

    // Count active vertices
    size_t active_count = 0;
    for (size_t i = 0; i < num_vertices; i++) {
        if (vertices[i].weight > config->detection_threshold) {
            active_count++;
        }
    }

    if (active_count == 0) {
        *edges = NULL;
        *num_edges = 0;
        return QGT_SUCCESS;
    }

    // Allocate edges (worst case: fully connected)
    size_t max_edges = active_count * (active_count - 1) / 2;
    *edges = (SyndromeEdge*)malloc(max_edges * sizeof(SyndromeEdge));
    if (!*edges) {
        return QGT_ERROR_OUT_OF_MEMORY;
    }

    // Build edges between active vertices
    size_t edge_count = 0;
    for (size_t i = 0; i < num_vertices && edge_count < max_edges; i++) {
        if (vertices[i].weight < config->detection_threshold) continue;

        for (size_t j = i + 1; j < num_vertices && edge_count < max_edges; j++) {
            if (vertices[j].weight < config->detection_threshold) continue;

            // Compute distance-based weight
            int dx = abs((int)vertices[i].position.x - (int)vertices[j].position.x);
            int dy = abs((int)vertices[i].position.y - (int)vertices[j].position.y);
            int dz = abs((int)vertices[i].position.z - (int)vertices[j].position.z);

            SyndromeEdge* edge = &(*edges)[edge_count];
            edge->vertex1 = (uint32_t)i;
            edge->vertex2 = (uint32_t)j;
            edge->weight = (float)(dx + dy + dz) * config->weight_scale_factor;
            edge->is_boundary_edge = vertices[i].is_boundary || vertices[j].is_boundary;
            edge->correlation = vertices[i].confidence * vertices[j].confidence;

            edge_count++;
        }
    }

    *num_edges = edge_count;
    return QGT_SUCCESS;
}

extern "C" int syndrome_compute_weights(void* context,
                                        SyndromeEdge* edges,
                                        size_t num_edges,
                                        const float* error_rates,
                                        size_t num_qubits) {
    if (!context || !edges || !error_rates) {
        return QGT_ERROR_INVALID_PARAMETER;
    }

    // Update edge weights based on error rates
    for (size_t i = 0; i < num_edges; i++) {
        uint32_t q1 = edges[i].vertex1 % num_qubits;
        uint32_t q2 = edges[i].vertex2 % num_qubits;

        float combined_rate = (error_rates[q1] + error_rates[q2]) / 2.0f;
        edges[i].weight *= (1.0f - combined_rate);
    }

    return QGT_SUCCESS;
}

// =============================================================================
// Utility Functions
// =============================================================================

extern "C" void syndrome_free_result(SyndromeResult* result) {
    if (!result) return;

    if (result->vertices) {
        free(result->vertices);
        result->vertices = NULL;
    }
    if (result->edges) {
        free(result->edges);
        result->edges = NULL;
    }
    if (result->error_chain) {
        free(result->error_chain);
        result->error_chain = NULL;
    }

    result->num_vertices = 0;
    result->num_edges = 0;
    result->chain_length = 0;
}

extern "C" void syndrome_free_pattern(SyndromePattern* pattern) {
    if (!pattern) return;

    if (pattern->vertex_indices) {
        free(pattern->vertex_indices);
        pattern->vertex_indices = NULL;
    }
    pattern->pattern_size = 0;
}

extern "C" float syndrome_estimate_logical_error_rate(void* context,
                                                      const SyndromeResult* results,
                                                      size_t num_results) {
    if (!context || !results || num_results == 0) {
        return 0.0f;
    }

    float total_rate = 0.0f;
    for (size_t i = 0; i < num_results; i++) {
        total_rate += results[i].logical_error_rate;
    }

    return total_rate / (float)num_results;
}

extern "C" int syndrome_get_statistics(void* context,
                                       size_t* total_extracted,
                                       size_t* total_decoded,
                                       float* avg_chain_length) {
    if (!context) {
        return QGT_ERROR_INVALID_PARAMETER;
    }

    SyndromeMetalContext* ctx = (SyndromeMetalContext*)context;

    if (total_extracted) *total_extracted = ctx->total_extracted;
    if (total_decoded) *total_decoded = ctx->total_decoded;
    if (avg_chain_length) {
        *avg_chain_length = ctx->num_results > 0 ?
            (float)ctx->total_chain_length / (float)ctx->num_results : 0.0f;
    }

    return QGT_SUCCESS;
}
