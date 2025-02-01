#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "quantum_geometric/core/differential_transformer.h"
#include <stdio.h>

// Metal device and command queue
static id<MTLDevice> device = nil;
static id<MTLCommandQueue> commandQueue = nil;
static id<MTLLibrary> library = nil;

// Compiled shader functions
static id<MTLFunction> derivativesFunction = nil;
static id<MTLFunction> attentionScoresFunction = nil;
static id<MTLFunction> softmaxFunction = nil;
static id<MTLFunction> attentionOutputFunction = nil;

// Pipeline states
static id<MTLComputePipelineState> derivativesPipeline = nil;
static id<MTLComputePipelineState> attentionScoresPipeline = nil;
static id<MTLComputePipelineState> softmaxPipeline = nil;
static id<MTLComputePipelineState> attentionOutputPipeline = nil;

// Initialize Metal resources
extern "C" bool metal_init_differential() {
    @autoreleasepool {
        // Get default Metal device
        device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "Failed to create Metal device\n");
            return false;
        }
        
        // Create command queue
        commandQueue = [device newCommandQueue];
        if (!commandQueue) {
            fprintf(stderr, "Failed to create command queue\n");
            return false;
        }
        
        // Load Metal library
        NSString* metalLibPath = @"differential_transformer_metal.metallib";
        NSString* resourcePath = [[NSBundle mainBundle] resourcePath];
        NSString* fullPath = [resourcePath stringByAppendingPathComponent:metalLibPath];
        
        NSError* error = nil;
        library = [device newLibraryWithFile:fullPath error:&error];
        if (!library) {
            fprintf(stderr, "Failed to load Metal library from %s: %s\n", 
                    [fullPath UTF8String], 
                    [error.localizedDescription UTF8String]);
            return false;
        }
        
        // Get shader functions
        derivativesFunction = [library newFunctionWithName:@"compute_token_derivatives"];
        attentionScoresFunction = [library newFunctionWithName:@"differential_attention_scores"];
        softmaxFunction = [library newFunctionWithName:@"differential_softmax"];
        attentionOutputFunction = [library newFunctionWithName:@"differential_attention_output"];
        
        if (!derivativesFunction || !attentionScoresFunction || 
            !softmaxFunction || !attentionOutputFunction) {
            fprintf(stderr, "Failed to load Metal functions\n");
            return false;
        }
        
        // Create pipeline states
        derivativesPipeline = [device newComputePipelineStateWithFunction:derivativesFunction error:&error];
        attentionScoresPipeline = [device newComputePipelineStateWithFunction:attentionScoresFunction error:&error];
        softmaxPipeline = [device newComputePipelineStateWithFunction:softmaxFunction error:&error];
        attentionOutputPipeline = [device newComputePipelineStateWithFunction:attentionOutputFunction error:&error];
        
        if (!derivativesPipeline || !attentionScoresPipeline || 
            !softmaxPipeline || !attentionOutputPipeline) {
            fprintf(stderr, "Failed to create pipeline states: %s\n", 
                    [error.localizedDescription UTF8String]);
            return false;
        }
        
        return true;
    }
}

// Clean up Metal resources
extern "C" void metal_cleanup_differential() {
    @autoreleasepool {
        [derivativesPipeline release];
        [attentionScoresPipeline release];
        [softmaxPipeline release];
        [attentionOutputPipeline release];
        
        [derivativesFunction release];
        [attentionScoresFunction release];
        [softmaxFunction release];
        [attentionOutputFunction release];
        
        [library release];
        [commandQueue release];
        [device release];
    }
}

// Helper function to create Metal buffer
static id<MTLBuffer> create_metal_buffer(const void* data, size_t size) {
    id<MTLBuffer> buffer = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
    if (data) {
        memcpy(buffer.contents, data, size);
    }
    return buffer;
}

// GPU-accelerated differential transformer forward pass
extern "C" void metal_diff_transformer_forward(
    DiffTransformerState* state,
    const double* input,
    double* output
) {
    @autoreleasepool {
        size_t seq_len = state->seq_length;
        size_t hidden_dim = state->hidden_dim;
        size_t total_size = seq_len * hidden_dim;
        
        // Create command buffer
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        
        // Create parameter buffer
        typedef struct {
            uint32_t seq_length;
            uint32_t hidden_dim;
            uint32_t num_heads;
            float learning_rate;
        } DiffParams;
        
        DiffParams params = {
            (uint32_t)seq_len,
            (uint32_t)hidden_dim,
            (uint32_t)state->num_heads,
            (float)state->learning_rate
        };
        
        id<MTLBuffer> paramsBuffer = create_metal_buffer(&params, sizeof(DiffParams));
        
        // Create data buffers
        id<MTLBuffer> inputBuffer = create_metal_buffer(input, total_size * sizeof(double));
        id<MTLBuffer> outputBuffer = create_metal_buffer(NULL, total_size * sizeof(double));
        id<MTLBuffer> derivBuffer = create_metal_buffer(NULL, total_size * sizeof(double));
        
        // Compute token derivatives
        {
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:derivativesPipeline];
            [encoder setBuffer:inputBuffer offset:0 atIndex:0];
            [encoder setBuffer:derivBuffer offset:0 atIndex:1];
            [encoder setBuffer:paramsBuffer offset:0 atIndex:2];
            
            MTLSize gridSize = MTLSizeMake(total_size, 1, 1);
            MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
            [encoder endEncoding];
        }
        
        // Process attention layers
        size_t head_dim = hidden_dim / state->num_heads;
        
        // Compute attention scores
        {
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:attentionScoresPipeline];
            
            id<MTLBuffer> scoresBuffer = create_metal_buffer(NULL, seq_len * seq_len * sizeof(double));
            id<MTLBuffer> scoresDerivsBuffer = create_metal_buffer(NULL, seq_len * seq_len * sizeof(double));
            
            [encoder setBuffer:inputBuffer offset:0 atIndex:0];
            [encoder setBuffer:inputBuffer offset:0 atIndex:1];
            [encoder setBuffer:derivBuffer offset:0 atIndex:2];
            [encoder setBuffer:derivBuffer offset:0 atIndex:3];
            [encoder setBuffer:scoresBuffer offset:0 atIndex:4];
            [encoder setBuffer:scoresDerivsBuffer offset:0 atIndex:5];
            [encoder setBuffer:paramsBuffer offset:0 atIndex:6];
            
            MTLSize gridSize = MTLSizeMake(seq_len, seq_len, 1);
            MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
            [encoder endEncoding];
            
            // Apply softmax
            encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:softmaxPipeline];
            [encoder setBuffer:scoresBuffer offset:0 atIndex:0];
            [encoder setBuffer:scoresDerivsBuffer offset:0 atIndex:1];
            [encoder setBuffer:paramsBuffer offset:0 atIndex:2];
            
            gridSize = MTLSizeMake(seq_len, 1, 1);
            threadGroupSize = MTLSizeMake(256, 1, 1);
            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
            [encoder endEncoding];
            
            // Compute attention output
            encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:attentionOutputPipeline];
            [encoder setBuffer:scoresBuffer offset:0 atIndex:0];
            [encoder setBuffer:inputBuffer offset:0 atIndex:1];
            [encoder setBuffer:scoresDerivsBuffer offset:0 atIndex:2];
            [encoder setBuffer:outputBuffer offset:0 atIndex:3];
            [encoder setBuffer:derivBuffer offset:0 atIndex:4];
            [encoder setBuffer:paramsBuffer offset:0 atIndex:5];
            
            gridSize = MTLSizeMake(hidden_dim, seq_len, 1);
            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
            [encoder endEncoding];
            
            [scoresBuffer release];
            [scoresDerivsBuffer release];
        }
        
        // Commit and wait
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Copy results back
        memcpy(output, outputBuffer.contents, total_size * sizeof(double));
        memcpy(state->derivatives, derivBuffer.contents, total_size * sizeof(double));
        
        // Cleanup
        [inputBuffer release];
        [outputBuffer release];
        [derivBuffer release];
        [paramsBuffer release];
    }
}
