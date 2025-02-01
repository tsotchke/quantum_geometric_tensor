#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "quantum_geometric/physics/quantum_field_operations.h"

// Metal context
static id<MTLDevice> device = nil;
static id<MTLCommandQueue> commandQueue = nil;
static id<MTLLibrary> library = nil;
static id<MTLComputePipelineState> rotationPipeline = nil;
static id<MTLComputePipelineState> energyPipeline = nil;
static id<MTLComputePipelineState> equationsPipeline = nil;

// Initialize Metal
static bool initMetal() {
    if (device) return true;
    
    // Get default device
    device = MTLCreateSystemDefaultDevice();
    if (!device) return false;
    
    // Create command queue
    commandQueue = [device newCommandQueue];
    if (!commandQueue) return false;
    
    // Load Metal library
    NSError* error = nil;
    NSString* metalLibPath = @"quantum_field_metal.metallib";
    NSString* resourcePath = [[NSBundle mainBundle] resourcePath];
    NSString* fullPath = [resourcePath stringByAppendingPathComponent:metalLibPath];
    
    library = [device newLibraryWithFile:fullPath error:&error];
    if (!library) {
        NSLog(@"Failed to load Metal library from %@: %@", fullPath, error);
        return false;
    }
    
    // Create compute pipelines
    id<MTLFunction> rotationFunc = [library newFunctionWithName:@"apply_rotation_kernel"];
    id<MTLFunction> energyFunc = [library newFunctionWithName:@"calculate_field_energy_kernel"];
    id<MTLFunction> equationsFunc = [library newFunctionWithName:@"calculate_field_equations_kernel"];
    
    rotationPipeline = [device newComputePipelineStateWithFunction:rotationFunc error:&error];
    energyPipeline = [device newComputePipelineStateWithFunction:energyFunc error:&error];
    equationsPipeline = [device newComputePipelineStateWithFunction:equationsFunc error:&error];
    
    if (!rotationPipeline || !energyPipeline || !equationsPipeline) {
        NSLog(@"Failed to create compute pipelines: %@", error);
        return false;
    }
    
    return true;
}

// Metal implementation of field operations
extern "C" {

int apply_rotation_metal(
    QuantumField* field,
    size_t qubit,
    double theta,
    double phi) {
    
    if (!initMetal()) return -1;
    
    // Create command buffer
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    // Set pipeline
    [encoder setComputePipelineState:rotationPipeline];
    
    // Create buffers
    size_t field_size = field->field_tensor->size * sizeof(complex double);
    id<MTLBuffer> fieldBuffer = [device newBufferWithBytes:field->field_tensor->data
                                                  length:field_size
                                                 options:MTLResourceStorageModeShared];
    
    // Create rotation matrix
    complex double rotation[4] = {
        cos(theta/2),
        -sin(theta/2) * cexp(-I*phi),
        sin(theta/2) * cexp(I*phi),
        cos(theta/2)
    };
    
    id<MTLBuffer> rotationBuffer = [device newBufferWithBytes:rotation
                                                     length:4 * sizeof(complex double)
                                                    options:MTLResourceStorageModeShared];
    
    // Set parameters
    [encoder setBuffer:fieldBuffer offset:0 atIndex:0];
    [encoder setBuffer:rotationBuffer offset:0 atIndex:1];
    uint32_t params[] = {
        (uint32_t)field->field_tensor->size,
        (uint32_t)field->field_tensor->dims[4],
        (uint32_t)qubit
    };
    [encoder setBytes:params length:sizeof(params) atIndex:2];
    
    // Dispatch
    MTLSize gridSize = MTLSizeMake(field->field_tensor->size, 1, 1);
    MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    
    // Finish
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Copy result back
    memcpy(field->field_tensor->data, [fieldBuffer contents], field_size);
    
    return 0;
}

double calculate_field_energy_metal(const QuantumField* field) {
    if (!initMetal()) return 0.0;
    
    // Create command buffer
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    // Set pipeline
    [encoder setComputePipelineState:energyPipeline];
    
    // Create buffers
    size_t field_size = field->field_tensor->size * sizeof(complex double);
    id<MTLBuffer> fieldBuffer = [device newBufferWithBytes:field->field_tensor->data
                                                  length:field_size
                                                 options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> momentumBuffer = [device newBufferWithBytes:field->conjugate_momentum->data
                                                     length:field_size
                                                    options:MTLResourceStorageModeShared];
    
    float energy = 0.0f;
    id<MTLBuffer> energyBuffer = [device newBufferWithBytes:&energy
                                                   length:sizeof(float)
                                                  options:MTLResourceStorageModeShared];
    
    // Set parameters
    [encoder setBuffer:fieldBuffer offset:0 atIndex:0];
    [encoder setBuffer:momentumBuffer offset:0 atIndex:1];
    [encoder setBuffer:energyBuffer offset:0 atIndex:2];
    uint32_t params[] = {
        (uint32_t)field->field_tensor->size,
        (uint32_t)field->field_tensor->dims[4]
    };
    [encoder setBytes:params length:sizeof(params) atIndex:3];
    
    // Dispatch
    MTLSize gridSize = MTLSizeMake(field->field_tensor->size, 1, 1);
    MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    
    // Finish
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Get result
    float* result = (float*)[energyBuffer contents];
    return (double)*result;
}

int calculate_field_equations_metal(
    const QuantumField* field,
    Tensor* equations) {
    
    if (!initMetal()) return -1;
    
    // Create command buffer
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    // Set pipeline
    [encoder setComputePipelineState:equationsPipeline];
    
    // Create buffers
    size_t field_size = field->field_tensor->size * sizeof(complex double);
    id<MTLBuffer> fieldBuffer = [device newBufferWithBytes:field->field_tensor->data
                                                  length:field_size
                                                 options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> equationsBuffer = [device newBufferWithBytes:equations->data
                                                      length:field_size
                                                     options:MTLResourceStorageModeShared];
    
    // Set parameters
    [encoder setBuffer:fieldBuffer offset:0 atIndex:0];
    [encoder setBuffer:equationsBuffer offset:0 atIndex:1];
    uint32_t params[] = {
        (uint32_t)field->field_tensor->size,
        (uint32_t)field->field_tensor->dims[4]
    };
    [encoder setBytes:params length:sizeof(params) atIndex:2];
    float mass = field->mass;
    float coupling = field->coupling;
    [encoder setBytes:&mass length:sizeof(float) atIndex:3];
    [encoder setBytes:&coupling length:sizeof(float) atIndex:4];
    
    // Dispatch
    MTLSize gridSize = MTLSizeMake(field->field_tensor->size, 1, 1);
    MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    
    // Finish
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Copy result back
    memcpy(equations->data, [equationsBuffer contents], field_size);
    
    return 0;
}

} // extern "C"
