/**
 * @file quantum_field_metal.mm
 * @brief Production Metal GPU backend for quantum field operations
 *
 * Full implementation of GPU-accelerated quantum field operations
 * using Apple's Metal framework. Converts between C99 complex double
 * and Metal's float2 representation.
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "quantum_geometric/physics/quantum_field_calculations.h"
#include "quantum_geometric/hardware/quantum_field_gpu.h"
#include <pthread.h>
#include <complex.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ============================================================================
// Metal Complex Type (matches Metal shader Complex struct)
// ============================================================================

typedef struct {
    float real;
    float imag;
} MetalComplex;

// ============================================================================
// Metal Backend State
// ============================================================================

static id<MTLDevice> device = nil;
static id<MTLCommandQueue> commandQueue = nil;
static id<MTLLibrary> library = nil;
static id<MTLComputePipelineState> rotationPipeline = nil;
static id<MTLComputePipelineState> energyPipeline = nil;
static id<MTLComputePipelineState> equationsPipeline = nil;
static pthread_mutex_t metalLock = PTHREAD_MUTEX_INITIALIZER;
static bool metalInitialized = false;

// ============================================================================
// Embedded Metal Shader Source (fallback when .metallib not found)
// ============================================================================

static const char* const embeddedShaderSourceC = R"(
#include <metal_stdlib>
using namespace metal;

constant float MAX_MAGNITUDE = 1e3f;
constant float MIN_MAGNITUDE = 1e-6f;
constant float ERROR_THRESHOLD = 1e-6f;

struct Complex {
    float2 val;
    Complex() : val(float2(0.0f)) {}
    Complex(float2 v) : val(v) {}
    Complex(float r, float i) : val(float2(r, i)) {}

    bool is_valid() const {
        float2 abs_val = abs(val);
        return !any(isnan(val)) && !any(isinf(val)) &&
               all((abs_val == float2(0.0f)) ||
                   (abs_val >= float2(MIN_MAGNITUDE) && abs_val <= float2(MAX_MAGNITUDE)));
    }

    float magnitude() const { return length(val); }

    void normalize(float max_allowed = MAX_MAGNITUDE) {
        float mag = magnitude();
        if (mag > max_allowed) val *= (max_allowed / mag);
        else if (mag > 0.0f && mag < MIN_MAGNITUDE) val *= (MIN_MAGNITUDE / mag);
    }

    friend Complex multiply(thread const Complex& a, thread const Complex& b) {
        if (!a.is_valid() || !b.is_valid()) return Complex();
        Complex a_norm = a; Complex b_norm = b;
        a_norm.normalize(); b_norm.normalize();
        Complex result(float2(a_norm.val.x * b_norm.val.x - a_norm.val.y * b_norm.val.y,
                              a_norm.val.x * b_norm.val.y + a_norm.val.y * b_norm.val.x));
        result.normalize();
        return result;
    }

    friend Complex multiply(device const Complex* a, thread const Complex& b) {
        Complex a_thread((*a).val);
        return multiply(a_thread, b);
    }

    Complex operator+(thread const Complex& rhs) const {
        if (!rhs.is_valid()) return *this;
        Complex result = *this;
        result.val += rhs.val;
        result.normalize();
        return result;
    }
};

inline bool validate_field_params(uint idx, uint field_size, uint num_components, uint qubit = 0) {
    return idx < field_size && qubit < 32 && num_components > 0 && num_components <= 256;
}

kernel void apply_rotation_kernel(
    device Complex* field [[buffer(0)]],
    device const Complex* rotation [[buffer(1)]],
    constant uint& field_size [[buffer(2)]],
    constant uint& num_components [[buffer(3)]],
    constant uint& qubit [[buffer(4)]],
    uint idx [[thread_position_in_grid]])
{
    if (!validate_field_params(idx, field_size, num_components, qubit)) return;
    bool valid = true;
    for (uint i = 0; i < 4 && valid; i++) valid = Complex(rotation[i].val).is_valid();
    if (!valid) return;

    uint mask = 1u << qubit;
    if (idx & mask) {
        Complex psi_0(field[idx ^ mask].val);
        Complex psi_1(field[idx].val);
        if (!psi_0.is_valid() || !psi_1.is_valid()) {
            field[idx ^ mask].val = float2(0.0f);
            field[idx].val = float2(0.0f);
            return;
        }
        Complex new_psi_0 = multiply(&rotation[0], psi_0) + multiply(&rotation[1], psi_1);
        Complex new_psi_1 = multiply(&rotation[2], psi_0) + multiply(&rotation[3], psi_1);
        new_psi_0.normalize(); new_psi_1.normalize();
        field[idx ^ mask].val = new_psi_0.val;
        field[idx].val = new_psi_1.val;
    }
}

kernel void calculate_field_energy_kernel(
    device const Complex* field [[buffer(0)]],
    device const Complex* momentum [[buffer(1)]],
    device atomic_float* energy [[buffer(2)]],
    constant uint& field_size [[buffer(3)]],
    constant uint& num_components [[buffer(4)]],
    uint idx [[thread_position_in_grid]])
{
    if (!validate_field_params(idx, field_size, num_components)) return;
    float local_energy = 0.0f, max_term = 0.0f;
    for (uint i = 0; i < num_components; i++) {
        Complex pi(momentum[idx * num_components + i].val);
        Complex phi(field[idx * num_components + i].val);
        if (!pi.is_valid() || !phi.is_valid()) continue;
        max_term = max(max_term, max(pi.magnitude(), phi.magnitude()));
    }
    if (max_term > 0.0f && max_term < MAX_MAGNITUDE) {
        float scale = min(1.0f, MAX_MAGNITUDE / max_term);
        float phi_sq = 0.0f;
        for (uint i = 0; i < num_components; i++) {
            Complex pi(momentum[idx * num_components + i].val);
            Complex phi(field[idx * num_components + i].val);
            if (pi.is_valid()) local_energy += dot(pi.val * scale, pi.val * scale);
            if (phi.is_valid()) phi_sq += dot(phi.val * scale, phi.val * scale);
        }
        local_energy = (local_energy + 0.5f * phi_sq) / (scale * scale);
    }
    if (local_energy > MIN_MAGNITUDE && local_energy < MAX_MAGNITUDE)
        atomic_fetch_add_explicit(energy, local_energy, memory_order_relaxed);
}

kernel void calculate_field_equations_kernel(
    device const Complex* field [[buffer(0)]],
    device Complex* equations [[buffer(1)]],
    constant uint& field_size [[buffer(2)]],
    constant uint& num_components [[buffer(3)]],
    constant float& mass [[buffer(4)]],
    constant float& coupling [[buffer(5)]],
    uint idx [[thread_position_in_grid]])
{
    if (!validate_field_params(idx, field_size, num_components)) return;
    if (mass < ERROR_THRESHOLD || coupling < ERROR_THRESHOLD ||
        !isfinite(mass) || !isfinite(coupling)) return;

    float phi_sq = 0.0f, max_mag = 0.0f;
    for (uint i = 0; i < num_components; i++) {
        Complex phi(field[idx * num_components + i].val);
        if (!phi.is_valid()) continue;
        max_mag = max(max_mag, phi.magnitude());
        phi_sq += dot(phi.val, phi.val);
    }
    float scale = max_mag > MAX_MAGNITUDE ? MAX_MAGNITUDE / max_mag : 1.0f;
    phi_sq *= (scale * scale);

    for (uint i = 0; i < num_components; i++) {
        Complex phi(field[idx * num_components + i].val);
        if (!phi.is_valid()) { equations[idx * num_components + i].val = float2(0.0f); continue; }
        float2 sp = phi.val * scale;
        Complex eq(mass * mass * sp);
        if (phi_sq > MIN_MAGNITUDE && phi_sq < MAX_MAGNITUDE)
            eq = eq + Complex(coupling * phi_sq * sp);
        eq.val /= scale;
        eq.normalize();
        equations[idx * num_components + i].val = eq.val;
    }
}
)";

static NSString* getEmbeddedShaderSource() {
    return [NSString stringWithUTF8String:embeddedShaderSourceC];
}

// ============================================================================
// Helper: Convert QuantumField data to/from Metal buffers
// ============================================================================

static id<MTLBuffer> createFieldBuffer(id<MTLDevice> dev, const Tensor* tensor) {
    if (!tensor || !tensor->data || tensor->total_size == 0) return nil;

    size_t bufferSize = tensor->total_size * sizeof(MetalComplex);
    MetalComplex* metalData = (MetalComplex*)malloc(bufferSize);
    if (!metalData) return nil;

    for (size_t i = 0; i < tensor->total_size; i++) {
        // Use __real__ and __imag__ extensions (works in clang C++ mode)
        metalData[i].real = (float)__real__(tensor->data[i]);
        metalData[i].imag = (float)__imag__(tensor->data[i]);
    }

    id<MTLBuffer> buffer = [dev newBufferWithBytes:metalData
                                            length:bufferSize
                                           options:MTLResourceStorageModeShared];
    free(metalData);
    return buffer;
}

static void copyBufferToTensor(id<MTLBuffer> buffer, Tensor* tensor) {
    if (!buffer || !tensor || !tensor->data) return;
    MetalComplex* data = (MetalComplex*)buffer.contents;
    for (size_t i = 0; i < tensor->total_size; i++) {
        // Construct complex value using __real__ and __imag__ extensions
        __real__(tensor->data[i]) = (double)data[i].real;
        __imag__(tensor->data[i]) = (double)data[i].imag;
    }
}

// ============================================================================
// Metal Backend Initialization
// ============================================================================

static bool initMetal() {
    pthread_mutex_lock(&metalLock);

    if (metalInitialized) {
        pthread_mutex_unlock(&metalLock);
        return device != nil;
    }

    @autoreleasepool {
        // Get default device
        device = MTLCreateSystemDefaultDevice();
        if (!device) {
            NSLog(@"Metal: Failed to create system default device");
            metalInitialized = true;
            pthread_mutex_unlock(&metalLock);
            return false;
        }

        // Create command queue
        commandQueue = [device newCommandQueue];
        if (!commandQueue) {
            NSLog(@"Metal: Failed to create command queue");
            device = nil;
            metalInitialized = true;
            pthread_mutex_unlock(&metalLock);
            return false;
        }

        // Try to load Metal library from various paths
        NSError* error = nil;
        library = [device newDefaultLibrary];

        if (!library) {
            NSArray<NSString*>* paths = @[
                @"quantum_field_metal.metallib",
                @"./lib/quantum_field_metal.metallib",
                @"../lib/quantum_field_metal.metallib"
            ];
            for (NSString* path in paths) {
                library = [device newLibraryWithFile:path error:&error];
                if (library) break;
            }
        }

        // Compile from embedded source as fallback
        if (!library) {
            MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
            opts.mathMode = MTLMathModeFast;
            library = [device newLibraryWithSource:getEmbeddedShaderSource() options:opts error:&error];
            if (!library) {
                NSLog(@"Metal: Failed to compile shaders: %@", error.localizedDescription);
                commandQueue = nil;
                device = nil;
                metalInitialized = true;
                pthread_mutex_unlock(&metalLock);
                return false;
            }
            NSLog(@"Metal: Compiled shaders from embedded source");
        }

        // Create compute pipelines
        id<MTLFunction> rotFunc = [library newFunctionWithName:@"apply_rotation_kernel"];
        id<MTLFunction> engFunc = [library newFunctionWithName:@"calculate_field_energy_kernel"];
        id<MTLFunction> eqFunc = [library newFunctionWithName:@"calculate_field_equations_kernel"];

        if (!rotFunc || !engFunc || !eqFunc) {
            NSLog(@"Metal: Failed to load kernel functions");
            library = nil; commandQueue = nil; device = nil;
            metalInitialized = true;
            pthread_mutex_unlock(&metalLock);
            return false;
        }

        rotationPipeline = [device newComputePipelineStateWithFunction:rotFunc error:&error];
        energyPipeline = [device newComputePipelineStateWithFunction:engFunc error:&error];
        equationsPipeline = [device newComputePipelineStateWithFunction:eqFunc error:&error];

        if (!rotationPipeline || !energyPipeline || !equationsPipeline) {
            NSLog(@"Metal: Failed to create pipelines: %@", error.localizedDescription);
            library = nil; commandQueue = nil; device = nil;
            metalInitialized = true;
            pthread_mutex_unlock(&metalLock);
            return false;
        }

        metalInitialized = true;
        NSLog(@"Metal: Backend initialized on device: %@", device.name);
    }

    pthread_mutex_unlock(&metalLock);
    return true;
}

// Metal implementation of field operations
extern "C" {

int apply_rotation_metal(
    QuantumField* field,
    size_t qubit,
    double theta,
    double phi) {

    if (!field || !field->field_tensor || !field->field_tensor->data) {
        return -1;
    }

    if (!initMetal()) return -1;

    @autoreleasepool {
        // Get dimensions
        uint32_t field_size = (uint32_t)field->field_tensor->total_size;
        uint32_t num_components = (uint32_t)field->num_components;
        uint32_t qubit_idx = (uint32_t)qubit;

        // Create rotation matrix (RZ(phi) * RY(theta) decomposition)
        MetalComplex rotation[4];
        float cos_t = cosf((float)theta / 2.0f);
        float sin_t = sinf((float)theta / 2.0f);
        float cos_p = cosf((float)phi);
        float sin_p = sinf((float)phi);

        rotation[0].real = cos_t;
        rotation[0].imag = 0.0f;
        rotation[1].real = -sin_t * cos_p;
        rotation[1].imag = -sin_t * sin_p;
        rotation[2].real = sin_t * cos_p;
        rotation[2].imag = sin_t * sin_p;
        rotation[3].real = cos_t;
        rotation[3].imag = 0.0f;

        // Create Metal buffers with proper data conversion
        id<MTLBuffer> fieldBuffer = createFieldBuffer(device, field->field_tensor);
        if (!fieldBuffer) return -1;

        id<MTLBuffer> rotationBuffer = [device newBufferWithBytes:rotation
                                                          length:4 * sizeof(MetalComplex)
                                                         options:MTLResourceStorageModeShared];

        id<MTLBuffer> fieldSizeBuffer = [device newBufferWithBytes:&field_size
                                                            length:sizeof(uint32_t)
                                                           options:MTLResourceStorageModeShared];

        id<MTLBuffer> numComponentsBuffer = [device newBufferWithBytes:&num_components
                                                                length:sizeof(uint32_t)
                                                               options:MTLResourceStorageModeShared];

        id<MTLBuffer> qubitBuffer = [device newBufferWithBytes:&qubit_idx
                                                        length:sizeof(uint32_t)
                                                       options:MTLResourceStorageModeShared];

        // Create command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        // Set pipeline and buffers
        [encoder setComputePipelineState:rotationPipeline];
        [encoder setBuffer:fieldBuffer offset:0 atIndex:0];
        [encoder setBuffer:rotationBuffer offset:0 atIndex:1];
        [encoder setBuffer:fieldSizeBuffer offset:0 atIndex:2];
        [encoder setBuffer:numComponentsBuffer offset:0 atIndex:3];
        [encoder setBuffer:qubitBuffer offset:0 atIndex:4];

        // Calculate thread configuration
        NSUInteger threadGroupSize = MIN(256, (NSUInteger)rotationPipeline.maxTotalThreadsPerThreadgroup);
        MTLSize threadsPerGroup = MTLSizeMake(threadGroupSize, 1, 1);
        MTLSize numThreadgroups = MTLSizeMake((field_size + threadGroupSize - 1) / threadGroupSize, 1, 1);

        // Dispatch
        [encoder dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:threadsPerGroup];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Check for errors
        if (commandBuffer.status == MTLCommandBufferStatusError) {
            NSLog(@"Metal: apply_rotation failed: %@", commandBuffer.error.localizedDescription);
            return -1;
        }

        // Copy results back (convert from float to double complex)
        copyBufferToTensor(fieldBuffer, field->field_tensor);
        return 0;
    }
}

double calculate_field_energy_metal(const QuantumField* field) {
    if (!field || !field->field_tensor || !field->field_tensor->data) {
        return 0.0;
    }

    if (!initMetal()) return 0.0;

    @autoreleasepool {
        uint32_t field_size = (uint32_t)field->field_tensor->total_size;
        uint32_t num_components = (uint32_t)field->num_components;

        // Create field buffer
        id<MTLBuffer> fieldBuffer = createFieldBuffer(device, field->field_tensor);
        if (!fieldBuffer) return 0.0;

        // Create momentum buffer (zeros if not available)
        id<MTLBuffer> momentumBuffer;
        if (field->conjugate_momentum && field->conjugate_momentum->data) {
            momentumBuffer = createFieldBuffer(device, field->conjugate_momentum);
        } else {
            size_t bufSize = field_size * sizeof(MetalComplex);
            MetalComplex* zeros = (MetalComplex*)calloc(field_size, sizeof(MetalComplex));
            momentumBuffer = [device newBufferWithBytes:zeros
                                                 length:bufSize
                                                options:MTLResourceStorageModeShared];
            free(zeros);
        }

        // Energy accumulator
        float zero = 0.0f;
        id<MTLBuffer> energyBuffer = [device newBufferWithBytes:&zero
                                                         length:sizeof(float)
                                                        options:MTLResourceStorageModeShared];

        id<MTLBuffer> fieldSizeBuffer = [device newBufferWithBytes:&field_size
                                                            length:sizeof(uint32_t)
                                                           options:MTLResourceStorageModeShared];

        id<MTLBuffer> numComponentsBuffer = [device newBufferWithBytes:&num_components
                                                                length:sizeof(uint32_t)
                                                               options:MTLResourceStorageModeShared];

        // Create command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:energyPipeline];
        [encoder setBuffer:fieldBuffer offset:0 atIndex:0];
        [encoder setBuffer:momentumBuffer offset:0 atIndex:1];
        [encoder setBuffer:energyBuffer offset:0 atIndex:2];
        [encoder setBuffer:fieldSizeBuffer offset:0 atIndex:3];
        [encoder setBuffer:numComponentsBuffer offset:0 atIndex:4];

        NSUInteger threadGroupSize = MIN(256, (NSUInteger)energyPipeline.maxTotalThreadsPerThreadgroup);
        MTLSize threadsPerGroup = MTLSizeMake(threadGroupSize, 1, 1);
        MTLSize numThreadgroups = MTLSizeMake((field_size + threadGroupSize - 1) / threadGroupSize, 1, 1);

        [encoder dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:threadsPerGroup];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        if (commandBuffer.status == MTLCommandBufferStatusError) {
            NSLog(@"Metal: calculate_energy failed: %@", commandBuffer.error.localizedDescription);
            return 0.0;
        }

        float* result = (float*)energyBuffer.contents;
        return (double)*result;
    }
}

int calculate_field_equations_metal(
    const QuantumField* field,
    Tensor* equations) {

    if (!field || !field->field_tensor || !field->field_tensor->data ||
        !equations || !equations->data) {
        return -1;
    }

    if (!initMetal()) return -1;

    @autoreleasepool {
        uint32_t field_size = (uint32_t)field->field_tensor->total_size;
        uint32_t num_components = (uint32_t)field->num_components;
        float mass = (float)field->mass;
        float coupling = (float)field->coupling;

        // Create buffers
        id<MTLBuffer> fieldBuffer = createFieldBuffer(device, field->field_tensor);
        if (!fieldBuffer) return -1;

        size_t eqBufferSize = equations->total_size * sizeof(MetalComplex);
        id<MTLBuffer> equationsBuffer = [device newBufferWithLength:eqBufferSize
                                                            options:MTLResourceStorageModeShared];

        id<MTLBuffer> fieldSizeBuffer = [device newBufferWithBytes:&field_size
                                                            length:sizeof(uint32_t)
                                                           options:MTLResourceStorageModeShared];

        id<MTLBuffer> numComponentsBuffer = [device newBufferWithBytes:&num_components
                                                                length:sizeof(uint32_t)
                                                               options:MTLResourceStorageModeShared];

        id<MTLBuffer> massBuffer = [device newBufferWithBytes:&mass
                                                       length:sizeof(float)
                                                      options:MTLResourceStorageModeShared];

        id<MTLBuffer> couplingBuffer = [device newBufferWithBytes:&coupling
                                                           length:sizeof(float)
                                                          options:MTLResourceStorageModeShared];

        // Command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:equationsPipeline];
        [encoder setBuffer:fieldBuffer offset:0 atIndex:0];
        [encoder setBuffer:equationsBuffer offset:0 atIndex:1];
        [encoder setBuffer:fieldSizeBuffer offset:0 atIndex:2];
        [encoder setBuffer:numComponentsBuffer offset:0 atIndex:3];
        [encoder setBuffer:massBuffer offset:0 atIndex:4];
        [encoder setBuffer:couplingBuffer offset:0 atIndex:5];

        NSUInteger threadGroupSize = MIN(256, (NSUInteger)equationsPipeline.maxTotalThreadsPerThreadgroup);
        MTLSize threadsPerGroup = MTLSizeMake(threadGroupSize, 1, 1);
        MTLSize numThreadgroups = MTLSizeMake((field_size + threadGroupSize - 1) / threadGroupSize, 1, 1);

        [encoder dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:threadsPerGroup];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        if (commandBuffer.status == MTLCommandBufferStatusError) {
            NSLog(@"Metal: calculate_equations failed: %@", commandBuffer.error.localizedDescription);
            return -1;
        }

        // Copy results back
        copyBufferToTensor(equationsBuffer, equations);
        return 0;
    }
}

// Cleanup function
void cleanup_metal_field_backend(void) {
    pthread_mutex_lock(&metalLock);

    if (metalInitialized) {
        equationsPipeline = nil;
        energyPipeline = nil;
        rotationPipeline = nil;
        library = nil;
        commandQueue = nil;
        device = nil;
        metalInitialized = false;
    }

    pthread_mutex_unlock(&metalLock);
}

// ============================================================================
// Functions called from quantum_field_gpu.c for backend management
// ============================================================================

bool init_metal_backend(void** context) {
    if (!initMetal()) {
        return false;
    }
    if (context) {
        *context = (__bridge_retained void*)device;
    }
    return true;
}

void cleanup_metal_backend(void* context) {
    (void)context;  // Context managed by cleanup_metal_field_backend
    cleanup_metal_field_backend();
}

static char deviceNameBuffer[256] = {0};

const char* get_metal_device_name(void* context) {
    (void)context;
    if (!initMetal() || !device) {
        return "No Metal Device";
    }
    @autoreleasepool {
        NSString* name = device.name;
        if (name) {
            strncpy(deviceNameBuffer, name.UTF8String, sizeof(deviceNameBuffer) - 1);
            deviceNameBuffer[sizeof(deviceNameBuffer) - 1] = '\0';
            return deviceNameBuffer;
        }
    }
    return "Unknown Metal Device";
}

} // extern "C"
