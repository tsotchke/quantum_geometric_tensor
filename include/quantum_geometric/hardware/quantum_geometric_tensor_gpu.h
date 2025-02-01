#ifndef QUANTUM_GEOMETRIC_TENSOR_GPU_H
#define QUANTUM_GEOMETRIC_TENSOR_GPU_H

#include "quantum_geometric/core/quantum_geometric_gpu.h"
#include "quantum_geometric/core/quantum_complex.h"
#include "quantum_geometric/core/quantum_geometric_types.h"

// Configuration for quantum geometric tensor operations
typedef struct {
    double precision;              // Numerical precision threshold
    bool use_quantum_estimation;   // Use quantum estimation techniques
    bool use_quantum_memory;       // Use quantum memory optimization
    int error_correction;          // Error correction level (0-3)
    int optimization_level;        // Optimization level (0-3)
} QGTConfig;

// Metal-specific context
typedef struct {
    void* device;                 // Metal device
    void* command_queue;          // Metal command queue
    void* library;                // Metal shader library
    size_t device_memory;         // Available device memory
    qgt_error_t (*execute_metric)(void* state, void* metric, size_t rows, size_t cols);
    qgt_error_t (*execute_connection)(void* state, void* connection, size_t rows, size_t cols);
    qgt_error_t (*execute_curvature)(void* state, void* curvature, size_t rows, size_t cols);
} MetalContext;

// CUDA-specific context
typedef struct {
    void* stream;                 // CUDA stream
    void* module;                 // CUDA module
    qgt_error_t (*execute_metric)(void* state, void* metric, size_t rows, size_t cols);
    qgt_error_t (*execute_connection)(void* state, void* connection, size_t rows, size_t cols);
    qgt_error_t (*execute_curvature)(void* state, void* curvature, size_t rows, size_t cols);
} CUDAContext;

// GPU context for quantum geometric tensor operations
typedef struct {
    bool is_available;            // Whether GPU is available
    qgt_error_t (*malloc)(void** ptr, size_t size);
    qgt_error_t (*free)(void* ptr);
    qgt_error_t (*memcpy_to_device)(void* dst, const void* src, size_t size);
    qgt_error_t (*memcpy_from_device)(void* dst, const void* src, size_t size);
    size_t (*get_optimal_block_size)(void);
    union {
        MetalContext metal;       // Metal-specific context
        CUDAContext cuda;         // CUDA-specific context
    };
} GPUContext;

// Function declarations

// Get default configuration
QGTConfig qgt_default_config(void);

// Get error string
const char* qgt_error_string(qgt_error_t error);

// Compute quantum metric tensor on GPU
qgt_error_t compute_quantum_metric_gpu(GPUContext* ctx,
                                  const ComplexFloat* state,
                                  ComplexFloat* metric,
                                  size_t rows,
                                  size_t cols,
                                  const QGTConfig* config);

// Compute quantum connection on GPU
qgt_error_t compute_quantum_connection_gpu(GPUContext* ctx,
                                      const ComplexFloat* state,
                                      ComplexFloat* connection,
                                      size_t rows,
                                      size_t cols,
                                      const QGTConfig* config);

// Compute quantum curvature on GPU
qgt_error_t compute_quantum_curvature_gpu(GPUContext* ctx,
                                     const ComplexFloat* state,
                                     ComplexFloat* curvature,
                                     size_t rows,
                                     size_t cols,
                                     const QGTConfig* config);

#endif // QUANTUM_GEOMETRIC_TENSOR_GPU_H
