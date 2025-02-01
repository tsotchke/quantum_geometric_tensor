#ifndef QUANTUM_GEOMETRIC_GPU_H
#define QUANTUM_GEOMETRIC_GPU_H

#include "quantum_geometric/core/quantum_complex.h"
#include "quantum_geometric/core/quantum_geometric_constants.h"
#include "quantum_geometric/core/quantum_geometric_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Quantum operation structure
typedef struct {
    size_t num_qubits;
    ComplexFloat* matrix;
    size_t* target_qubits;
    size_t* control_qubits;
    size_t num_controls;
    bool is_controlled;
} QuantumOperation;

// Parameters for quantum geometric transformations
typedef struct {
    geometric_transform_type_t transform_type;
    size_t dimension;
    ComplexFloat* parameters;
    void* auxiliary_data;
} QuantumGeometricParams;

// GPU Backend types
typedef enum {
    GPU_BACKEND_NONE = 0,
    GPU_BACKEND_METAL,
    GPU_BACKEND_CUDA,
} GPUBackendType;

// GPU Device Info
typedef struct GPUDeviceInfo {
    char name[256];
    size_t total_memory;
    size_t available_memory;
    int compute_units;
    GPUBackendType backend_type;
    bool supports_unified_memory;
    bool supports_tensor_cores;
    bool supports_amx;  // Apple Matrix coprocessor
} GPUDeviceInfo;

// GPU Context
typedef struct GPUContext GPUContext;

// Initialize GPU system
int gpu_initialize(void);

// Cleanup GPU system
void gpu_cleanup(void);

// Get available GPU devices
int gpu_get_devices(GPUDeviceInfo* devices, int max_devices);

// Create GPU context
GPUContext* gpu_create_context(int device_index);

// Destroy GPU context
void gpu_destroy_context(GPUContext* context);

// Memory management
void* gpu_allocate(GPUContext* context, size_t size);
void gpu_free(GPUContext* context, void* ptr);
int gpu_memcpy_to_device(GPUContext* context, void* dst, const void* src, size_t size);
int gpu_memcpy_from_device(GPUContext* context, void* dst, const void* src, size_t size);

// Quantum operations
int gpu_quantum_tensor_multiply(
    GPUContext* context,
    const ComplexFloat* a,
    const ComplexFloat* b,
    ComplexFloat* c,
    int m, int n, int k
);

int gpu_quantum_geometric_transform(
    GPUContext* context,
    const ComplexFloat* input,
    ComplexFloat* output,
    const QuantumGeometricParams* params,
    size_t size
);

int gpu_quantum_attention(
    GPUContext* context,
    const ComplexFloat* queries,
    const ComplexFloat* keys,
    const ComplexFloat* values,
    ComplexFloat* output,
    int batch_size,
    int num_heads,
    int seq_length,
    int head_dim
);

// Batch operations
int gpu_batch_quantum_operations(
    GPUContext* context,
    const ComplexFloat* states,
    ComplexFloat* results,
    const QuantumOperation* operations,
    int num_states,
    int num_operations
);

// Error handling
const char* gpu_get_last_error(void);
void gpu_clear_error(void);

// Performance monitoring
typedef struct GPUPerformanceMetrics {
    double compute_time;
    double memory_transfer_time;
    size_t memory_used;
    int num_operations;
} GPUPerformanceMetrics;

int gpu_get_performance_metrics(GPUContext* context, GPUPerformanceMetrics* metrics);

#ifdef ENABLE_METAL
// Metal-specific optimizations for Apple Silicon
int gpu_enable_amx(GPUContext* context);
int gpu_set_metal_compute_units(GPUContext* context, int num_units);
int gpu_optimize_for_m1(GPUContext* context);
#endif

#ifdef ENABLE_CUDA
// Optional CUDA-specific operations
int gpu_enable_tensor_cores(GPUContext* context);
int gpu_set_cuda_stream(GPUContext* context, void* stream);
#endif

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_GEOMETRIC_GPU_H
