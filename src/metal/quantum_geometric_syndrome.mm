#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "quantum_geometric/hardware/metal/quantum_geometric_syndrome.h"
#include "quantum_geometric/core/performance_monitor.h"
#include "quantum_geometric/hardware/quantum_hardware_optimization.h"

// Metal state
static id<MTLDevice> device = nil;
static id<MTLCommandQueue> commandQueue = nil;
static id<MTLLibrary> library = nil;
static id<MTLComputePipelineState> extractPipeline = nil;
static id<MTLComputePipelineState> correlatePipeline = nil;

// Performance metrics
static double extraction_time = 0.0;
static double correlation_time = 0.0;
static size_t syndrome_count = 0;
static float resource_usage = 0.0;

bool init_metal_syndrome_resources(void) {
    @autoreleasepool {
        // Get default Metal device
        device = MTLCreateSystemDefaultDevice();
        if (!device) return false;
        
        // Create command queue
        commandQueue = [device newCommandQueue];
        if (!commandQueue) return false;
        
        // Load Metal library
        NSError* error = nil;
        NSString* metalLibPath = @"quantum_geometric_syndrome.metallib";
        NSString* resourcePath = [[NSBundle mainBundle] resourcePath];
        NSString* fullPath = [resourcePath stringByAppendingPathComponent:metalLibPath];
        
        library = [device newLibraryWithFile:fullPath error:&error];
        if (!library) {
            NSLog(@"Failed to load Metal library from %@: %@", fullPath, error);
            return false;
        }
        
        // Create compute pipelines
        id<MTLFunction> extractFunc = [library newFunctionWithName:@"extract_syndromes"];
        id<MTLFunction> correlateFunc = [library newFunctionWithName:@"compute_syndrome_correlations"];
        
        if (!extractFunc || !correlateFunc) {
            NSLog(@"Failed to load Metal functions");
            return false;
        }
        
        extractPipeline = [device newComputePipelineStateWithFunction:extractFunc error:&error];
        correlatePipeline = [device newComputePipelineStateWithFunction:correlateFunc error:&error];
        
        if (!extractPipeline || !correlatePipeline) {
            NSLog(@"Failed to create compute pipelines: %@", error);
            return false;
        }
        
        return true;
    }
}

void cleanup_metal_syndrome_resources(void) {
    @autoreleasepool {
        [extractPipeline release];
        [correlatePipeline release];
        [library release];
        [commandQueue release];
        [device release];
        
        extractPipeline = nil;
        correlatePipeline = nil;
        library = nil;
        commandQueue = nil;
        device = nil;
    }
}

size_t extract_syndromes_metal(quantum_state* state,
                             const SyndromeConfig* config,
                             MatchingGraph* graph) {
    @autoreleasepool {
        if (!state || !config || !graph) return 0;
        
        // Start performance monitoring
        uint64_t start_time = get_hardware_timestamp();
        
        // Create buffers
        id<MTLBuffer> stateBuffer = [device newBufferWithBytes:state->amplitudes
                                                      length:state->size * sizeof(float2)
                                                     options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> indicesBuffer = [device newBufferWithBytes:state->indices
                                                        length:state->size * sizeof(uint32_t)
                                                       options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> configBuffer = [device newBufferWithBytes:config
                                                       length:sizeof(SyndromeConfig)
                                                      options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> vertexBuffer = [device newBufferWithBytes:graph->vertices
                                                       length:graph->max_vertices * sizeof(SyndromeVertex)
                                                      options:MTLResourceStorageModeShared];
        
        if (!stateBuffer || !indicesBuffer || !configBuffer || !vertexBuffer) {
            return 0;
        }
        
        // Create command buffer
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        // Configure pipeline
        [encoder setComputePipelineState:extractPipeline];
        [encoder setBuffer:stateBuffer offset:0 atIndex:0];
        [encoder setBuffer:indicesBuffer offset:0 atIndex:1];
        [encoder setBuffer:configBuffer offset:0 atIndex:2];
        [encoder setBuffer:vertexBuffer offset:0 atIndex:3];
        
        // Calculate grid and threadgroup sizes
        NSUInteger maxThreads = extractPipeline.maxTotalThreadsPerThreadgroup;
        NSUInteger threadExecutionWidth = extractPipeline.threadExecutionWidth;
        
        MTLSize threadgroupSize = MTLSizeMake(threadExecutionWidth, 1, 1);
        MTLSize gridSize = MTLSizeMake(graph->max_vertices, 1, 1);
        
        // Dispatch
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        
        // Execute and wait
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Copy results back
        memcpy(graph->vertices, [vertexBuffer contents], 
               graph->max_vertices * sizeof(SyndromeVertex));
        
        // Count active syndromes
        size_t active_count = 0;
        for (size_t i = 0; i < graph->max_vertices; i++) {
            if (graph->vertices[i].weight > config->detection_threshold) {
                active_count++;
            }
        }
        
        // Update performance metrics
        extraction_time = get_hardware_elapsed_time(start_time);
        syndrome_count = active_count;
        resource_usage = get_hardware_resource_usage();
        
        // Cleanup
        [stateBuffer release];
        [indicesBuffer release];
        [configBuffer release];
        [vertexBuffer release];
        
        return active_count;
    }
}

bool compute_syndrome_correlations_metal(MatchingGraph* graph,
                                       const SyndromeConfig* config) {
    @autoreleasepool {
        if (!graph || !config) return false;
        
        // Start performance monitoring
        uint64_t start_time = get_hardware_timestamp();
        
        // Create buffers
        id<MTLBuffer> vertexBuffer = [device newBufferWithBytes:graph->vertices
                                                       length:graph->max_vertices * sizeof(SyndromeVertex)
                                                      options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> configBuffer = [device newBufferWithBytes:config
                                                       length:sizeof(SyndromeConfig)
                                                      options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> correlationBuffer = [device newBufferWithBytes:graph->correlation_matrix
                                                            length:graph->max_vertices * graph->max_vertices * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
        
        if (!vertexBuffer || !configBuffer || !correlationBuffer) {
            return false;
        }
        
        // Create command buffer
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        // Configure pipeline
        [encoder setComputePipelineState:correlatePipeline];
        [encoder setBuffer:vertexBuffer offset:0 atIndex:0];
        [encoder setBuffer:configBuffer offset:0 atIndex:1];
        [encoder setBuffer:correlationBuffer offset:0 atIndex:2];
        
        // Calculate grid and threadgroup sizes
        NSUInteger maxThreads = correlatePipeline.maxTotalThreadsPerThreadgroup;
        NSUInteger threadExecutionWidth = correlatePipeline.threadExecutionWidth;
        
        MTLSize threadgroupSize = MTLSizeMake(threadExecutionWidth, threadExecutionWidth, 1);
        MTLSize gridSize = MTLSizeMake(graph->max_vertices, graph->max_vertices, 1);
        
        // Dispatch
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        
        // Execute and wait
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Copy results back
        memcpy(graph->correlation_matrix, [correlationBuffer contents],
               graph->max_vertices * graph->max_vertices * sizeof(float));
        
        // Update performance metrics
        correlation_time = get_hardware_elapsed_time(start_time);
        resource_usage = get_hardware_resource_usage();
        
        // Cleanup
        [vertexBuffer release];
        [configBuffer release];
        [correlationBuffer release];
        
        return true;
    }
}

bool verify_metal_syndrome_results(const MatchingGraph* graph,
                                 const quantum_state* state) {
    if (!graph || !state) return false;
    
    // Verify active syndromes have valid weights
    for (size_t i = 0; i < graph->num_vertices; i++) {
        if (graph->vertices[i].weight > 0.0f) {
            if (graph->vertices[i].confidence <= 0.0f ||
                graph->vertices[i].confidence > 1.0f) {
                return false;
            }
        }
    }
    
    // Verify correlation matrix is symmetric and normalized
    for (size_t i = 0; i < graph->num_vertices; i++) {
        for (size_t j = 0; j < graph->num_vertices; j++) {
            float corr_ij = graph->correlation_matrix[i * graph->num_vertices + j];
            float corr_ji = graph->correlation_matrix[j * graph->num_vertices + i];
            
            if (fabsf(corr_ij - corr_ji) > 1e-6f ||
                corr_ij < -1.0f || corr_ij > 1.0f) {
                return false;
            }
        }
    }
    
    return true;
}

// Performance monitoring functions
double get_metal_syndrome_extraction_time(void) {
    return extraction_time;
}

double get_metal_correlation_time(void) {
    return correlation_time;
}

size_t get_metal_syndrome_count(void) {
    return syndrome_count;
}

float get_metal_resource_usage(void) {
    return resource_usage;
}
