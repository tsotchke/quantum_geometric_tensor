#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <complex>
#include <vector>

struct ComplexFloat {
    float real;
    float imag;
};

@interface MetalDevice : NSObject
@property (strong, nonatomic) id<MTLDevice> device;
@property (strong, nonatomic) id<MTLCommandQueue> commandQueue;
@property (strong, nonatomic) id<MTLLibrary> library;
@property (strong, nonatomic) id<MTLComputePipelineState> encodePipeline;
@property (strong, nonatomic) id<MTLComputePipelineState> metricPipeline;
@property (strong, nonatomic) id<MTLComputePipelineState> transformPipeline;
@property (strong, nonatomic) id<MTLComputePipelineState> measurePipeline;

- (instancetype)init;
- (void)setupPipeline;
@end

@implementation MetalDevice

- (instancetype)init {
    self = [super init];
    if (self) {
        _device = MTLCreateSystemDefaultDevice();
        _commandQueue = [_device newCommandQueue];
        [self setupPipeline];
    }
    return self;
}

- (void)setupPipeline {
    NSError *error = nil;
    
    // Load Metal library
    NSString* metalLibPath = @"quantum_geometric_tensor.metallib";
    NSString* resourcePath = [[NSBundle mainBundle] resourcePath];
    NSString* fullPath = [resourcePath stringByAppendingPathComponent:metalLibPath];
    
    _library = [_device newLibraryWithFile:fullPath error:&error];
    if (!_library) {
        NSLog(@"Failed to load Metal library from %@: %@", fullPath, error);
        return;
    }
    
    // Get kernel functions
    id<MTLFunction> encodeFunction = [_library newFunctionWithName:@"encode_mnist_geometric"];
    id<MTLFunction> metricFunction = [_library newFunctionWithName:@"compute_geometric_metric"];
    id<MTLFunction> transformFunction = [_library newFunctionWithName:@"apply_geometric_transform"];
    id<MTLFunction> measureFunction = [_library newFunctionWithName:@"measure_geometric_state"];
    
    if (!encodeFunction || !metricFunction || !transformFunction || !measureFunction) {
        NSLog(@"Failed to load Metal functions");
        return;
    }
    
    // Create pipeline states
    _encodePipeline = [_device newComputePipelineStateWithFunction:encodeFunction error:&error];
    if (!_encodePipeline) {
        NSLog(@"Failed to create encode pipeline: %@", error);
        return;
    }
    
    _metricPipeline = [_device newComputePipelineStateWithFunction:metricFunction error:&error];
    if (!_metricPipeline) {
        NSLog(@"Failed to create metric pipeline: %@", error);
        return;
    }
    
    _transformPipeline = [_device newComputePipelineStateWithFunction:transformFunction error:&error];
    if (!_transformPipeline) {
        NSLog(@"Failed to create transform pipeline: %@", error);
        return;
    }
    
    _measurePipeline = [_device newComputePipelineStateWithFunction:measureFunction error:&error];
    if (!_measurePipeline) {
        NSLog(@"Failed to create measure pipeline: %@", error);
        return;
    }
}

@end

extern "C" {

bool process_mnist_geometric(
    const float* input_data,
    float* output_probabilities,
    const float* parameters,
    size_t batch_size,
    size_t input_dim,
    size_t num_classes)
{
    @autoreleasepool {
        MetalDevice *metalDevice = [[MetalDevice alloc] init];
        if (!metalDevice.device) {
            return false;
        }
        
        // Create Metal buffers
        id<MTLBuffer> inputBuffer = [metalDevice.device newBufferWithBytes:input_data
                                                                  length:batch_size * input_dim * sizeof(float)
                                                                 options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> stateBuffer = [metalDevice.device newBufferWithLength:batch_size * input_dim * sizeof(ComplexFloat)
                                                                  options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> phasesBuffer = [metalDevice.device newBufferWithBytes:parameters
                                                                   length:input_dim * sizeof(float)
                                                                  options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> metricBuffer = [metalDevice.device newBufferWithLength:batch_size * input_dim * input_dim * sizeof(float)
                                                                   options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> probBuffer = [metalDevice.device newBufferWithBytes:output_probabilities
                                                                 length:batch_size * num_classes * sizeof(float)
                                                                options:MTLResourceStorageModeShared];
        
        // Create command buffer
        id<MTLCommandBuffer> commandBuffer = [metalDevice.commandQueue commandBuffer];
        
        // Encode MNIST data
        {
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:metalDevice.encodePipeline];
            [encoder setBuffer:inputBuffer offset:0 atIndex:0];
            [encoder setBuffer:stateBuffer offset:0 atIndex:1];
            [encoder setBuffer:phasesBuffer offset:0 atIndex:2];
            
            uint32_t params[] = {(uint32_t)batch_size, (uint32_t)input_dim};
            [encoder setBytes:params length:sizeof(params) atIndex:3];
            
            MTLSize gridSize = MTLSizeMake(batch_size, input_dim, 1);
            MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
            [encoder endEncoding];
        }
        
        // Compute geometric metric
        {
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:metalDevice.metricPipeline];
            [encoder setBuffer:stateBuffer offset:0 atIndex:0];
            [encoder setBuffer:metricBuffer offset:0 atIndex:1];
            
            uint32_t params[] = {(uint32_t)batch_size, (uint32_t)input_dim};
            [encoder setBytes:params length:sizeof(params) atIndex:2];
            
            MTLSize gridSize = MTLSizeMake(batch_size, input_dim * input_dim, 1);
            MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
            [encoder endEncoding];
        }
        
        // Apply geometric transformation
        {
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:metalDevice.transformPipeline];
            [encoder setBuffer:stateBuffer offset:0 atIndex:0];
            [encoder setBuffer:phasesBuffer offset:0 atIndex:1];
            
            uint32_t params[] = {(uint32_t)batch_size, (uint32_t)input_dim};
            [encoder setBytes:params length:sizeof(params) atIndex:2];
            
            MTLSize gridSize = MTLSizeMake(batch_size, input_dim, 1);
            MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
            [encoder endEncoding];
        }
        
        // Measure quantum state
        {
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:metalDevice.measurePipeline];
            [encoder setBuffer:stateBuffer offset:0 atIndex:0];
            [encoder setBuffer:probBuffer offset:0 atIndex:1];
            
            uint32_t params[] = {(uint32_t)batch_size, (uint32_t)num_classes, (uint32_t)input_dim};
            [encoder setBytes:params length:sizeof(params) atIndex:2];
            
            MTLSize gridSize = MTLSizeMake(batch_size, num_classes, 1);
            MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
            [encoder endEncoding];
        }
        
        // Execute and wait
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Copy results back
        memcpy(output_probabilities, probBuffer.contents, batch_size * num_classes * sizeof(float));
        
        return true;
    }
}

}
