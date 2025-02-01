#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "quantum_geometric/hardware/metal/tensor_operations_metal.h"

// Metal device and command queue
static id<MTLDevice> device = nil;
static id<MTLCommandQueue> commandQueue = nil;
static id<MTLLibrary> library = nil;
static id<MTLFunction> matrixMultiplyOptimizedFunction = nil;
static id<MTLComputePipelineState> matrixMultiplyOptimizedPipeline = nil;

// Check memory alignment
static bool is_aligned(const void* ptr, size_t alignment) {
    return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
}

// Initialize Metal state
static metal_tensor_error_t initialize_metal() {
    if (device) return METAL_TENSOR_SUCCESS;  // Already initialized
    
    // Get default Metal device
    device = MTLCreateSystemDefaultDevice();
    if (!device) return METAL_TENSOR_ERROR_NO_DEVICE;
    
    // Create command queue
    commandQueue = [device newCommandQueue];
    if (!commandQueue) return METAL_TENSOR_ERROR_INIT_FAILED;
    
    // Compile Metal library from source
    NSString* shaderSource = [NSString stringWithFormat:@"#include <metal_stdlib>\n"
                            "using namespace metal;\n"
                            "\n"
                            "struct ComplexFloat {\n"
                            "    float real;\n"
                            "    float imag;\n"
                            "    bool valid;\n"
                            "};\n"
                            "\n"
                            "kernel void matrix_multiply_optimized(\n"
                            "    device const ComplexFloat* A [[buffer(0)]],\n"
                            "    device const ComplexFloat* B [[buffer(1)]],\n"
                            "    device ComplexFloat* C [[buffer(2)]],\n"
                            "    constant uint& M [[buffer(3)]],\n"
                            "    constant uint& N [[buffer(4)]],\n"
                            "    constant uint& K [[buffer(5)]],\n"
                            "    uint2 gid [[thread_position_in_grid]]) {\n"
                            "    if (gid.x >= N || gid.y >= M) return;\n"
                            "    float sum_real = 0.0f;\n"
                            "    float sum_imag = 0.0f;\n"
                            "    bool valid = true;\n"
                            "    \n"
                            "    for (uint k = 0; k < K; k++) {\n"
                            "        ComplexFloat a = A[gid.y * K + k];\n"
                            "        ComplexFloat b = B[k * N + gid.x];\n"
                            "        if (!a.valid || !b.valid) {\n"
                            "            valid = false;\n"
                            "            break;\n"
                            "        }\n"
                            "        sum_real += a.real * b.real - a.imag * b.imag;\n"
                            "        sum_imag += a.real * b.imag + a.imag * b.real;\n"
                            "    }\n"
                            "    \n"
                            "    ComplexFloat result;\n"
                            "    result.real = sum_real;\n"
                            "    result.imag = sum_imag;\n"
                            "    result.valid = valid;\n"
                            "    C[gid.y * N + gid.x] = result;\n"
                            "}\n";
    
    NSError* error = nil;
    library = [device newLibraryWithSource:shaderSource
                                 options:nil
                                   error:&error];
    if (!library) {
        NSLog(@"Failed to compile Metal library: %@", error);
        return METAL_TENSOR_ERROR_INIT_FAILED;
    }
    
    // Get kernel function
    matrixMultiplyOptimizedFunction = [library newFunctionWithName:@"matrix_multiply_optimized"];
    if (!matrixMultiplyOptimizedFunction) {
        NSLog(@"Failed to load Metal function");
        return METAL_TENSOR_ERROR_INIT_FAILED;
    }
    
    // Create compute pipeline
    matrixMultiplyOptimizedPipeline = [device newComputePipelineStateWithFunction:matrixMultiplyOptimizedFunction error:&error];
    if (!matrixMultiplyOptimizedPipeline) {
        NSLog(@"Failed to create compute pipeline: %@", error);
        return METAL_TENSOR_ERROR_INIT_FAILED;
    }
    
    return METAL_TENSOR_SUCCESS;
}

// Check if Metal is available
bool metal_is_available(void) {
    if (device) return true;
    return MTLCreateSystemDefaultDevice() != nil;
}

// Complex number struct matching Metal shader
typedef struct {
    float real;
    float imag;
    bool valid;
} ComplexFloat;

// Convert float arrays to complex numbers
static void convert_to_complex(ComplexFloat* dst, const float* src_real, const float* src_imag, size_t count) {
    for (size_t i = 0; i < count; i++) {
        dst[i].real = src_real[i];
        dst[i].imag = src_imag ? src_imag[i] : 0.0f;
        dst[i].valid = true;
    }
}

// Convert complex numbers back to float arrays
static void convert_from_complex(float* dst_real, float* dst_imag, const ComplexFloat* src, size_t count) {
    for (size_t i = 0; i < count; i++) {
        if (src[i].valid) {
            dst_real[i] = src[i].real;
            if (dst_imag) dst_imag[i] = src[i].imag;
        } else {
            dst_real[i] = 0.0f;
            if (dst_imag) dst_imag[i] = 0.0f;
        }
    }
}

// Optimized Metal matrix multiplication implementation
metal_tensor_error_t metal_matrix_multiply(float* C_real, float* C_imag, 
                                         const float* A_real, const float* A_imag,
                                         const float* B_real, const float* B_imag,
                                         uint M, uint N, uint K) {
    // Validate inputs
    if (!C_real || !A_real || !B_real) {
        return METAL_TENSOR_ERROR_INVALID_ARGS;
    }
    
    if (M == 0 || N == 0 || K == 0) {
        return METAL_TENSOR_ERROR_INVALID_ARGS;
    }
    
    // Check alignment for optimal performance
    const NSUInteger alignment = 256;
    if (!is_aligned(C_real, alignment) || !is_aligned(A_real, alignment) || !is_aligned(B_real, alignment)) {
        NSLog(@"Warning: Input matrices not aligned to %lu bytes, performance may be degraded", alignment);
    }
    
    // Initialize Metal
    metal_tensor_error_t init_result = initialize_metal();
    if (init_result != METAL_TENSOR_SUCCESS) {
        return init_result;
    }
    
    // Allocate memory for complex format
    const size_t A_elements = M * K;
    const size_t B_elements = K * N;
    const size_t C_elements = M * N;
    
    ComplexFloat* A_complex = (ComplexFloat*)malloc(A_elements * sizeof(ComplexFloat));
    ComplexFloat* B_complex = (ComplexFloat*)malloc(B_elements * sizeof(ComplexFloat));
    ComplexFloat* C_complex = (ComplexFloat*)malloc(C_elements * sizeof(ComplexFloat));
    
    if (!A_complex || !B_complex || !C_complex) {
        free(A_complex);
        free(B_complex);
        free(C_complex);
        return METAL_TENSOR_ERROR_OUT_OF_MEMORY;
    }
    
    // Convert input to complex format
    convert_to_complex(A_complex, A_real, A_imag, A_elements);
    convert_to_complex(B_complex, B_real, B_imag, B_elements);
    
    // Create aligned Metal buffers with error handling
    const NSUInteger buffer_A_size = ((A_elements * sizeof(ComplexFloat) + alignment - 1) / alignment) * alignment;
    const NSUInteger buffer_B_size = ((B_elements * sizeof(ComplexFloat) + alignment - 1) / alignment) * alignment;
    const NSUInteger buffer_C_size = ((C_elements * sizeof(ComplexFloat) + alignment - 1) / alignment) * alignment;
    
    NSError* error = nil;
    id<MTLBuffer> bufferA = [device newBufferWithLength:buffer_A_size options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferB = [device newBufferWithLength:buffer_B_size options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferC = [device newBufferWithLength:buffer_C_size options:MTLResourceStorageModeShared];
    
    if (!bufferA || !bufferB || !bufferC) {
        free(A_complex);
        free(B_complex);
        free(C_complex);
        return METAL_TENSOR_ERROR_OUT_OF_MEMORY;
    }
    
    // Copy data to Metal buffers and execute
    @try {
        memcpy([bufferA contents], A_complex, A_elements * sizeof(ComplexFloat));
        memcpy([bufferB contents], B_complex, B_elements * sizeof(ComplexFloat));
        
        // Create command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        
        if (!commandBuffer || !computeEncoder) {
            free(A_complex);
            free(B_complex);
            free(C_complex);
            return METAL_TENSOR_ERROR_EXECUTION_FAILED;
        }
        
        // Set compute pipeline and arguments
        [computeEncoder setComputePipelineState:matrixMultiplyOptimizedPipeline];
        [computeEncoder setBuffer:bufferA offset:0 atIndex:0];
        [computeEncoder setBuffer:bufferB offset:0 atIndex:1];
        [computeEncoder setBuffer:bufferC offset:0 atIndex:2];
        [computeEncoder setBytes:&M length:sizeof(uint) atIndex:3];
        [computeEncoder setBytes:&N length:sizeof(uint) atIndex:4];
        [computeEncoder setBytes:&K length:sizeof(uint) atIndex:5];
        
        // Calculate optimized grid and threadgroup sizes
        const uint TILE_SIZE = 32; // Must match shader constant
        const uint num_tiles_M = (M + TILE_SIZE - 1) / TILE_SIZE;
        const uint num_tiles_N = (N + TILE_SIZE - 1) / TILE_SIZE;
        
        MTLSize gridSize = MTLSizeMake(num_tiles_N * TILE_SIZE, num_tiles_M * TILE_SIZE, 1);
        MTLSize threadgroupSize = MTLSizeMake(TILE_SIZE, TILE_SIZE, 1);
        
        // Dispatch threads
        [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [computeEncoder endEncoding];
        
        // Execute and wait for completion
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        if (commandBuffer.error) {
            NSLog(@"Metal execution failed: %@", commandBuffer.error);
            free(A_complex);
            free(B_complex);
            free(C_complex);
            return METAL_TENSOR_ERROR_EXECUTION_FAILED;
        }
        
        // Copy result back and convert to float arrays
        memcpy(C_complex, [bufferC contents], C_elements * sizeof(ComplexFloat));
        convert_from_complex(C_real, C_imag, C_complex, C_elements);
        
        // Cleanup
        free(A_complex);
        free(B_complex);
        free(C_complex);
        
        return METAL_TENSOR_SUCCESS;
    }
    @catch (NSException* exception) {
        NSLog(@"Metal execution exception: %@", exception);
        free(A_complex);
        free(B_complex);
        free(C_complex);
        return METAL_TENSOR_ERROR_EXECUTION_FAILED;
    }
}
