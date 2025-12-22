#ifndef QUANTUM_GEOMETRIC_GPU_H
#define QUANTUM_GEOMETRIC_GPU_H

#include "quantum_geometric/core/quantum_complex.h"
#include "quantum_geometric/core/quantum_geometric_constants.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct MemoryPool;

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

// GPU Context - full definition for direct access
typedef struct GPUContext {
    int device_index;
    GPUBackendType backend_type;
    void* device_handle;          // Metal device or CUDA device
    void* command_queue;          // Metal command queue or CUDA stream
    void* library;                // Metal library or CUDA module
    size_t allocated_memory;
    size_t max_memory;
    bool is_initialized;
} GPUContext;

// Multi-GPU Context for distributed attention
typedef struct MultiGPUContext {
    GPUContext** contexts;
    int num_contexts;
    int primary_device;
    bool synchronized;
} MultiGPUContext;

// Multi-GPU functions
MultiGPUContext* multi_gpu_create_context(int* device_indices, int num_devices);
void multi_gpu_destroy_context(MultiGPUContext* ctx);
int multi_gpu_synchronize(MultiGPUContext* ctx);

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

// Forward declaration for HierarchicalMatrix
struct HierarchicalMatrix;

// ============================================================================
// Hierarchical Matrix GPU Operations
// ============================================================================

// Convert data to hierarchical matrix on GPU
struct HierarchicalMatrix* convert_to_hierarchical_gpu(
    const ComplexFloat* data,
    size_t rows,
    size_t cols,
    double tolerance,
    GPUContext* ctx
);

// Create empty hierarchical matrix on GPU
struct HierarchicalMatrix* create_hierarchical_matrix_gpu(
    size_t rows,
    size_t cols,
    double tolerance,
    GPUContext* ctx
);

// Destroy hierarchical matrix on GPU
void destroy_hierarchical_matrix_gpu(struct HierarchicalMatrix* matrix, GPUContext* ctx);

// Hierarchical matrix multiplication on GPU
int hierarchical_multiply_gpu(
    struct HierarchicalMatrix* result,
    const struct HierarchicalMatrix* a,
    const struct HierarchicalMatrix* b,
    GPUContext* ctx
);

// Convert from hierarchical with dropout (for attention)
void convert_from_hierarchical_with_dropout_gpu(
    ComplexFloat* output,
    const struct HierarchicalMatrix* matrix,
    size_t output_size,
    double dropout_rate,
    GPUContext* ctx
);

// ============================================================================
// Async Memory Operations
// ============================================================================

// Async memory copy to device
void gpu_memcpy_to_device_async(void* dst, const void* src, size_t size, void* stream);

// Async memory copy to host
void gpu_memcpy_to_host_async(void* dst, const void* src, size_t size, void* stream);

// Stream synchronization
void gpu_stream_synchronize(void* stream);

// ============================================================================
// Memory Pool GPU Operations
// ============================================================================

// Allocate from GPU memory pool
void* gpu_alloc_from_pool(struct MemoryPool* pool, size_t size);

// Free to GPU memory pool
void gpu_free_to_pool(struct MemoryPool* pool, void* ptr);

// ============================================================================
// Multi-GPU Context Operations
// ============================================================================

// Initialize multi-GPU context (returns NULL if no GPU available)
MultiGPUContext* init_multi_gpu_context(void);

// Synchronize all GPUs in context
void sync_multi_gpu_context(MultiGPUContext* ctx);

// Cleanup multi-GPU context
void cleanup_multi_gpu_context(MultiGPUContext* ctx);

// Get specific GPU context from multi-GPU context
GPUContext* get_gpu_context(MultiGPUContext* ctx, int device_id);

// ============================================================================
// Attention-Specific GPU Operations
// ============================================================================

// Save attention checkpoint for gradient computation
void save_attention_checkpoint(
    void* query, void* key, void* value,
    size_t size, size_t batch_idx, size_t head_idx,
    GPUContext* ctx
);

// Cleanup attention cache
void cleanup_attention_cache(void);

// Cleanup attention buffers
void cleanup_attention_buffers(void);

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
