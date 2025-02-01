#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "quantum_geometric/hardware/metal/mnist_metal.h"
#include "quantum_geometric/core/error_codes.h"

// Metal device and command queue
static id<MTLDevice> device = nil;
static id<MTLCommandQueue> commandQueue = nil;

// Metal compute pipeline states
static id<MTLComputePipelineState> encodeQuantumStatePipeline = nil;
static id<MTLComputePipelineState> applyGeometricTransformPipeline = nil;
static id<MTLComputePipelineState> measureQuantumStatePipeline = nil;

// Metal buffers
static id<MTLBuffer> quantumStateBuffer = nil;
static id<MTLBuffer> metricTensorBuffer = nil;
static id<MTLBuffer> parametersBuffer = nil;
static id<MTLBuffer> anglesBuffer = nil;
static id<MTLBuffer> phasesBuffer = nil;  // Buffer for geometric phases

// Initialize Metal device and command queue
int init_metal_device(void) {
    @autoreleasepool {
        // Get default Metal device
        device = MTLCreateSystemDefaultDevice();
        if (!device) {
            return QGT_ERROR_GPU_UNAVAILABLE;
        }
        
        // Create command queue
        commandQueue = [device newCommandQueue];
        if (!commandQueue) {
            return QGT_ERROR_INITIALIZATION;
        }
        
        // Initialize Metal buffers with default sizes
        int result = create_metal_buffers(32); // Default batch size
        if (result != QGT_SUCCESS) {
            return result;
        }
        
        // Compile Metal shaders
        result = compile_metal_shaders();
        if (result != QGT_SUCCESS) {
            return result;
        }
        
        return QGT_SUCCESS;
    }
}

// Compile Metal shader library
int compile_metal_shaders(void) {
    @autoreleasepool {
        NSError* error = nil;
        
        // Use path in current working directory
        const char* path = "src/metal/mnist_metal.metallib";
        NSString* metalLibPath = [NSString stringWithCString:path encoding:NSUTF8StringEncoding];
        NSString* source = [NSString stringWithContentsOfFile:metalLibPath 
                                                   encoding:NSUTF8StringEncoding 
                                                      error:&error];
        if (!source) {
            NSLog(@"Failed to read Metal shader source from %@: %@", metalLibPath, error);
            return QGT_ERROR_INITIALIZATION;
        }
        
        // Compile shader with optimizations for M1/M2
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.languageVersion = MTLLanguageVersion2_4; // Latest version for M1/M2
        options.optimizationLevel = MTLCompileOptimizationLevelDefault;
        options.fastMathEnabled = YES; // Enable fast math for M1/M2
        options.preprocessorMacros = @{
            @"THREADS_PER_GROUP": @256, // Optimal threadgroup size
            @"BATCH_SIZE": @32,         // Optimal batch size
            @"USE_SIMD": @YES           // Enable SIMD operations
        };
        
        id<MTLLibrary> library = [device newLibraryWithSource:source
                                                    options:options
                                                      error:&error];
        if (!library) {
            NSLog(@"Failed to compile Metal library: %@", error);
            return QGT_ERROR_INITIALIZATION;
        }
        
        // Create compute pipeline states
        id<MTLFunction> encodeFunction = [library newFunctionWithName:@"encode_quantum_state"];
        id<MTLFunction> transformFunction = [library newFunctionWithName:@"apply_geometric_transform"];
        id<MTLFunction> measureFunction = [library newFunctionWithName:@"measure_quantum_state"];
        
        if (!encodeFunction || !transformFunction || !measureFunction) {
            NSLog(@"Failed to get Metal functions");
            return QGT_ERROR_INITIALIZATION;
        }
        
        encodeQuantumStatePipeline = [device newComputePipelineStateWithFunction:encodeFunction error:&error];
        applyGeometricTransformPipeline = [device newComputePipelineStateWithFunction:transformFunction error:&error];
        measureQuantumStatePipeline = [device newComputePipelineStateWithFunction:measureFunction error:&error];
        
        if (!encodeQuantumStatePipeline || !applyGeometricTransformPipeline || !measureQuantumStatePipeline) {
            NSLog(@"Failed to create pipeline states: %@", error);
            return QGT_ERROR_INITIALIZATION;
        }
        
        return QGT_SUCCESS;
    }
}

// Create Metal buffers
int create_metal_buffers(size_t max_batch_size) {
    // Calculate buffer sizes with proper quantum state dimensions
    const size_t quantum_state_size = max_batch_size * (1 << MAX_QUBITS) * sizeof(float) * 2; // Complex numbers
    const size_t metric_tensor_size = max_batch_size * MAX_QUBITS * MAX_QUBITS * sizeof(float);
    const size_t parameters_size = MAX_QUBITS * sizeof(float);
    const size_t angles_size = max_batch_size * MNIST_IMAGE_SIZE * sizeof(float);
    const size_t phases_size = (1 << MAX_QUBITS) * sizeof(float); // One phase per quantum state
    
    // Create buffers with optimal storage modes for M1/M2
    MTLResourceOptions quantumStateOptions = MTLResourceStorageModeShared | MTLResourceHazardTrackingModeTracked;
    MTLResourceOptions metricTensorOptions = MTLResourceStorageModeShared | MTLResourceHazardTrackingModeTracked;
    MTLResourceOptions parameterOptions = MTLResourceStorageModeShared;
    MTLResourceOptions angleOptions = MTLResourceStorageModeShared;
    MTLResourceOptions phaseOptions = MTLResourceStorageModeShared;
    
    // Align buffer sizes to optimal M1/M2 page size
    const size_t pageSize = device.maxBufferLength;
    size_t aligned_quantum_state_size = ((quantum_state_size + pageSize - 1) / pageSize) * pageSize;
    size_t aligned_metric_tensor_size = ((metric_tensor_size + pageSize - 1) / pageSize) * pageSize;
    
    quantumStateBuffer = [device newBufferWithLength:aligned_quantum_state_size options:quantumStateOptions];
    metricTensorBuffer = [device newBufferWithLength:aligned_metric_tensor_size options:metricTensorOptions];
    parametersBuffer = [device newBufferWithLength:parameters_size options:parameterOptions];
    anglesBuffer = [device newBufferWithLength:angles_size options:angleOptions];
    phasesBuffer = [device newBufferWithLength:phases_size options:phaseOptions];
    
    if (!quantumStateBuffer || !metricTensorBuffer || !parametersBuffer || !anglesBuffer || !phasesBuffer) {
        return QGT_ERROR_OUT_OF_MEMORY;
    }
    
    // Initialize phases with enhanced geometric distribution
    float* phases = (float*)phasesBuffer.contents;
    for (int i = 0; i < (1 << MAX_QUBITS); i++) {
        // Calculate geometric coordinates
        int x = i % 28;
        int y = (i / 28) % 28;
        float dx = (x - 14.0f) / 14.0f;
        float dy = (y - 14.0f) / 14.0f;
        
        // Enhanced geometric phase pattern
        float r = sqrtf(dx * dx + dy * dy);
        float theta = atan2f(dy, dx);
        float phase = theta + 2.0f * M_PI * (
            r +  // Radial component
            0.5f * sinf(4.0f * theta) * (1.0f - r) + // Angular modulation
            0.25f * expf(-2.0f * r * r) // Gaussian envelope
        );
        phases[i] = phase;
    }
    
    return QGT_SUCCESS;
}

// Release Metal buffers
void release_metal_buffers(void) {
    quantumStateBuffer = nil;
    metricTensorBuffer = nil;
    parametersBuffer = nil;
    anglesBuffer = nil;
    phasesBuffer = nil;
}

// Clean up Metal resources
void cleanup_metal_device(void) {
    release_metal_buffers();
    encodeQuantumStatePipeline = nil;
    applyGeometricTransformPipeline = nil;
    measureQuantumStatePipeline = nil;
    commandQueue = nil;
    device = nil;
}

// Encode quantum state using Metal with geometric encoding
int encode_quantum_state_metal(
    QuantumGeometricState* state,
    const float* data,
    size_t batch_size,
    float input_dim) {
    
    if (!state || !data || batch_size == 0) {
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    // Copy input data to Metal buffer
    memcpy(anglesBuffer.contents, data, batch_size * input_dim * sizeof(float));
    
    // Create command buffer
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    
    // Set compute pipeline
    [computeEncoder setComputePipelineState:encodeQuantumStatePipeline];
    
    // Set buffers
    [computeEncoder setBuffer:anglesBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:quantumStateBuffer offset:0 atIndex:1];
    [computeEncoder setBuffer:phasesBuffer offset:0 atIndex:2];
    
    // Calculate grid and threadgroup sizes optimized for M1/M2
    NSUInteger gridSize = batch_size * MNIST_IMAGE_SIZE;
    NSUInteger threadGroupSize = MIN(256, gridSize); // Optimal for M1/M2, capped by grid size
    
    // Ensure grid size is multiple of threadgroup size
    NSUInteger adjustedGridSize = ((gridSize + threadGroupSize - 1) / threadGroupSize) * threadGroupSize;
    
    MTLSize threadsPerGrid = MTLSizeMake(adjustedGridSize, 1, 1);
    MTLSize threadsPerThreadgroup = MTLSizeMake(threadGroupSize, 1, 1);
    
    // Dispatch compute kernel
    [computeEncoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    [computeEncoder endEncoding];
    
    // Execute command buffer
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Copy results back to host
    memcpy(state->amplitudes, quantumStateBuffer.contents, 
           batch_size * state->state_size * sizeof(complex double));
    
    return QGT_SUCCESS;
}

// Apply geometric transformations using Metal
int apply_geometric_transform_metal(
    QuantumGeometricState* state,
    const float* parameters,
    size_t batch_size,
    float latent_dim) {
    
    if (!state || !parameters || batch_size == 0) {
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    // Copy parameters to Metal buffer
    memcpy(parametersBuffer.contents, parameters, latent_dim * sizeof(float));
    
    // Create command buffer
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    
    // Set compute pipeline
    [computeEncoder setComputePipelineState:applyGeometricTransformPipeline];
    
    // Set buffers
    [computeEncoder setBuffer:quantumStateBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:parametersBuffer offset:0 atIndex:1];
    [computeEncoder setBuffer:metricTensorBuffer offset:0 atIndex:2];
    [computeEncoder setBuffer:phasesBuffer offset:0 atIndex:3];
    
    // Calculate grid and threadgroup sizes optimized for M1/M2
    NSUInteger gridSize = batch_size * MAX_QUBITS * MAX_QUBITS;
    NSUInteger threadGroupSize = MIN(256, gridSize); // Optimal for M1/M2, capped by grid size
    
    // Ensure grid size is multiple of threadgroup size
    NSUInteger adjustedGridSize = ((gridSize + threadGroupSize - 1) / threadGroupSize) * threadGroupSize;
    
    MTLSize threadsPerGrid = MTLSizeMake(adjustedGridSize, 1, 1);
    MTLSize threadsPerThreadgroup = MTLSizeMake(threadGroupSize, 1, 1);
    
    // Dispatch compute kernel
    [computeEncoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    [computeEncoder endEncoding];
    
    // Execute command buffer
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Copy results back to host
    memcpy(state->metric_tensor, metricTensorBuffer.contents,
           batch_size * MAX_QUBITS * MAX_QUBITS * sizeof(double));
    
    return QGT_SUCCESS;
}

// Measure quantum state using Metal
int measure_quantum_state_metal(
    QuantumGeometricState* state,
    float* probabilities,
    size_t batch_size,
    float num_classes) {
    
    if (!state || !probabilities || batch_size == 0) {
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    // Create command buffer
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    
    // Set compute pipeline
    [computeEncoder setComputePipelineState:measureQuantumStatePipeline];
    
    // Set buffers
    [computeEncoder setBuffer:quantumStateBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:metricTensorBuffer offset:0 atIndex:1]; // Reuse for probabilities
    
    // Calculate grid and threadgroup sizes optimized for M1/M2
    NSUInteger gridSize = batch_size * (int)num_classes;
    NSUInteger threadGroupSize = MIN(256, gridSize); // Optimal for M1/M2, capped by grid size
    
    // Ensure grid size is multiple of threadgroup size
    NSUInteger adjustedGridSize = ((gridSize + threadGroupSize - 1) / threadGroupSize) * threadGroupSize;
    
    MTLSize threadsPerGrid = MTLSizeMake(adjustedGridSize, 1, 1);
    MTLSize threadsPerThreadgroup = MTLSizeMake(threadGroupSize, 1, 1);
    
    // Dispatch compute kernel
    [computeEncoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    [computeEncoder endEncoding];
    
    // Execute command buffer
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Copy results back to host
    memcpy(probabilities, metricTensorBuffer.contents,
           batch_size * (int)num_classes * sizeof(float));
    
    return QGT_SUCCESS;
}
