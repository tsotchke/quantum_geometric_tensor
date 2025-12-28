/**
 * @file quantum_geometric_tensor_ops.h
 * @brief GPU-Accelerated Tensor Operations for Quantum Geometric Computing
 *
 * Provides hardware-accelerated tensor operations including:
 * - GPU tensor creation and manipulation
 * - Tensor contraction on GPU
 * - Matrix operations (multiply, transpose, conjugate)
 * - Batch operations for quantum circuits
 * - Memory-efficient sparse tensor support
 * - Multi-GPU distribution
 *
 * Part of the QGTL Hardware Acceleration Framework.
 */

#ifndef QUANTUM_GEOMETRIC_TENSOR_OPS_H
#define QUANTUM_GEOMETRIC_TENSOR_OPS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================

#define TENSOR_OPS_MAX_DIMENSIONS 16
#define TENSOR_OPS_MAX_NAME_LENGTH 128
#define TENSOR_OPS_MAX_GPUS 8
#define TENSOR_OPS_ALIGNMENT 64

// ============================================================================
// Enumerations
// ============================================================================

/**
 * Tensor data types
 */
typedef enum {
    TENSOR_TYPE_FLOAT32,              // Single precision real
    TENSOR_TYPE_FLOAT64,              // Double precision real
    TENSOR_TYPE_COMPLEX64,            // Single precision complex
    TENSOR_TYPE_COMPLEX128,           // Double precision complex
    TENSOR_TYPE_INT32,                // 32-bit integer
    TENSOR_TYPE_INT64                 // 64-bit integer
} tensor_data_type_t;

/**
 * Tensor storage format
 */
typedef enum {
    TENSOR_STORAGE_DENSE,             // Dense storage
    TENSOR_STORAGE_SPARSE_COO,        // Coordinate sparse
    TENSOR_STORAGE_SPARSE_CSR,        // Compressed sparse row
    TENSOR_STORAGE_SPARSE_CSC,        // Compressed sparse column
    TENSOR_STORAGE_BLOCK_SPARSE       // Block sparse
} tensor_storage_t;

/**
 * Memory location
 */
typedef enum {
    TENSOR_MEMORY_HOST,               // CPU memory
    TENSOR_MEMORY_DEVICE,             // GPU memory
    TENSOR_MEMORY_UNIFIED,            // Unified memory
    TENSOR_MEMORY_PINNED              // Pinned host memory
} tensor_memory_t;

/**
 * Contraction algorithm
 */
typedef enum {
    CONTRACTION_AUTO,                 // Automatic selection
    CONTRACTION_DIRECT,               // Direct contraction
    CONTRACTION_BATCH,                // Batched matrix multiply
    CONTRACTION_SLICE,                // Sliced contraction
    CONTRACTION_TREE                  // Tree contraction
} contraction_algorithm_t;

/**
 * Transpose mode
 */
typedef enum {
    TRANSPOSE_NONE,                   // No transpose
    TRANSPOSE_NORMAL,                 // Transpose
    TRANSPOSE_CONJUGATE,              // Conjugate (no transpose)
    TRANSPOSE_HERMITIAN               // Hermitian (conjugate transpose)
} transpose_mode_t;

// ============================================================================
// Data Structures
// ============================================================================

/**
 * Tensor descriptor
 */
typedef struct {
    tensor_data_type_t dtype;
    tensor_storage_t storage;
    tensor_memory_t memory;
    size_t ndim;
    size_t shape[TENSOR_OPS_MAX_DIMENSIONS];
    size_t strides[TENSOR_OPS_MAX_DIMENSIONS];
    size_t total_elements;
    size_t memory_bytes;
    int device_id;                    // GPU device ID (-1 for host)
    bool is_contiguous;
    char name[TENSOR_OPS_MAX_NAME_LENGTH];
} tensor_descriptor_t;

/**
 * Opaque tensor handle
 */
typedef struct gpu_tensor gpu_tensor_t;

/**
 * Contraction specification
 */
typedef struct {
    size_t tensor_a_indices[TENSOR_OPS_MAX_DIMENSIONS];
    size_t tensor_b_indices[TENSOR_OPS_MAX_DIMENSIONS];
    size_t output_indices[TENSOR_OPS_MAX_DIMENSIONS];
    size_t num_contracted;            // Number of contracted indices
    contraction_algorithm_t algorithm;
    bool optimize_memory;
} contraction_spec_t;

/**
 * Batch operation descriptor
 */
typedef struct {
    size_t batch_size;
    size_t* batch_offsets;
    bool parallel_execution;
    int stream_id;
} batch_op_desc_t;

/**
 * Memory pool for tensor operations
 */
typedef struct tensor_memory_pool tensor_memory_pool_t;

/**
 * Operation context for managing GPU state
 */
typedef struct tensor_ops_context tensor_ops_context_t;

/**
 * Operation statistics
 */
typedef struct {
    uint64_t total_operations;
    uint64_t total_flops;
    uint64_t total_bytes_transferred;
    double total_time_ms;
    double avg_throughput_gflops;
    double memory_efficiency;
} tensor_ops_stats_t;

/**
 * Context configuration
 */
typedef struct {
    int device_id;                    // Primary GPU device
    size_t memory_pool_size;          // Memory pool size in bytes
    bool enable_async;                // Enable async operations
    bool enable_profiling;            // Enable operation profiling
    size_t workspace_size;            // Workspace size for algorithms
    int num_streams;                  // Number of CUDA streams
} tensor_ops_config_t;

// ============================================================================
// Initialization and Context
// ============================================================================

/**
 * Create tensor operations context
 */
tensor_ops_context_t* tensor_ops_context_create(void);

/**
 * Create context with configuration
 */
tensor_ops_context_t* tensor_ops_context_create_with_config(
    const tensor_ops_config_t* config);

/**
 * Get default configuration
 */
tensor_ops_config_t tensor_ops_default_config(void);

/**
 * Destroy context
 */
void tensor_ops_context_destroy(tensor_ops_context_t* ctx);

/**
 * Synchronize all operations
 */
bool tensor_ops_synchronize(tensor_ops_context_t* ctx);

/**
 * Get operation statistics
 */
bool tensor_ops_get_stats(tensor_ops_context_t* ctx, tensor_ops_stats_t* stats);

/**
 * Reset statistics
 */
void tensor_ops_reset_stats(tensor_ops_context_t* ctx);

// ============================================================================
// Tensor Creation and Memory
// ============================================================================

/**
 * Create tensor with specified shape
 */
gpu_tensor_t* tensor_create(
    tensor_ops_context_t* ctx,
    tensor_data_type_t dtype,
    size_t ndim,
    const size_t* shape,
    tensor_memory_t memory);

/**
 * Create tensor from host data
 */
gpu_tensor_t* tensor_create_from_data(
    tensor_ops_context_t* ctx,
    tensor_data_type_t dtype,
    size_t ndim,
    const size_t* shape,
    const void* data,
    tensor_memory_t memory);

/**
 * Create sparse tensor
 */
gpu_tensor_t* tensor_create_sparse(
    tensor_ops_context_t* ctx,
    tensor_data_type_t dtype,
    size_t ndim,
    const size_t* shape,
    tensor_storage_t storage,
    size_t nnz);

/**
 * Create identity matrix tensor
 */
gpu_tensor_t* tensor_create_identity(
    tensor_ops_context_t* ctx,
    tensor_data_type_t dtype,
    size_t dim,
    tensor_memory_t memory);

/**
 * Create zero tensor
 */
gpu_tensor_t* tensor_create_zeros(
    tensor_ops_context_t* ctx,
    tensor_data_type_t dtype,
    size_t ndim,
    const size_t* shape,
    tensor_memory_t memory);

/**
 * Clone tensor
 */
gpu_tensor_t* tensor_clone(
    tensor_ops_context_t* ctx,
    const gpu_tensor_t* src);

/**
 * Destroy tensor
 */
void tensor_destroy(gpu_tensor_t* tensor);

/**
 * Get tensor descriptor
 */
bool tensor_get_descriptor(
    const gpu_tensor_t* tensor,
    tensor_descriptor_t* desc);

/**
 * Copy tensor data to host
 */
bool tensor_copy_to_host(
    tensor_ops_context_t* ctx,
    const gpu_tensor_t* tensor,
    void* host_data);

/**
 * Copy tensor data from host
 */
bool tensor_copy_from_host(
    tensor_ops_context_t* ctx,
    gpu_tensor_t* tensor,
    const void* host_data);

/**
 * Transfer tensor to device
 */
bool tensor_to_device(
    tensor_ops_context_t* ctx,
    gpu_tensor_t* tensor,
    int device_id);

/**
 * Transfer tensor to host
 */
bool tensor_to_host(
    tensor_ops_context_t* ctx,
    gpu_tensor_t* tensor);

// ============================================================================
// Basic Operations
// ============================================================================

/**
 * Tensor addition: C = alpha*A + beta*B
 */
bool tensor_add(
    tensor_ops_context_t* ctx,
    double alpha, const gpu_tensor_t* A,
    double beta, const gpu_tensor_t* B,
    gpu_tensor_t* C);

/**
 * Element-wise multiplication
 */
bool tensor_elementwise_multiply(
    tensor_ops_context_t* ctx,
    const gpu_tensor_t* A,
    const gpu_tensor_t* B,
    gpu_tensor_t* C);

/**
 * Scale tensor: A = alpha * A
 */
bool tensor_scale(
    tensor_ops_context_t* ctx,
    gpu_tensor_t* tensor,
    double alpha);

/**
 * Conjugate tensor elements
 */
bool tensor_conjugate(
    tensor_ops_context_t* ctx,
    gpu_tensor_t* tensor);

/**
 * Transpose tensor
 */
gpu_tensor_t* tensor_transpose(
    tensor_ops_context_t* ctx,
    const gpu_tensor_t* tensor,
    const size_t* axes);

/**
 * Reshape tensor (view)
 */
gpu_tensor_t* tensor_reshape(
    tensor_ops_context_t* ctx,
    gpu_tensor_t* tensor,
    size_t new_ndim,
    const size_t* new_shape);

// ============================================================================
// Matrix Operations
// ============================================================================

/**
 * Matrix multiplication: C = alpha * op(A) * op(B) + beta * C
 */
bool tensor_matmul(
    tensor_ops_context_t* ctx,
    transpose_mode_t trans_a,
    transpose_mode_t trans_b,
    double alpha,
    const gpu_tensor_t* A,
    const gpu_tensor_t* B,
    double beta,
    gpu_tensor_t* C);

/**
 * Batched matrix multiplication
 */
bool tensor_batched_matmul(
    tensor_ops_context_t* ctx,
    transpose_mode_t trans_a,
    transpose_mode_t trans_b,
    double alpha,
    const gpu_tensor_t* A,
    const gpu_tensor_t* B,
    double beta,
    gpu_tensor_t* C,
    size_t batch_count);

/**
 * Matrix-vector product
 */
bool tensor_matvec(
    tensor_ops_context_t* ctx,
    transpose_mode_t trans,
    double alpha,
    const gpu_tensor_t* A,
    const gpu_tensor_t* x,
    double beta,
    gpu_tensor_t* y);

/**
 * Outer product
 */
gpu_tensor_t* tensor_outer(
    tensor_ops_context_t* ctx,
    const gpu_tensor_t* a,
    const gpu_tensor_t* b);

/**
 * Kronecker product
 */
gpu_tensor_t* tensor_kron(
    tensor_ops_context_t* ctx,
    const gpu_tensor_t* A,
    const gpu_tensor_t* B);

// ============================================================================
// Tensor Contraction
// ============================================================================

/**
 * General tensor contraction
 */
gpu_tensor_t* tensor_contract(
    tensor_ops_context_t* ctx,
    const gpu_tensor_t* A,
    const gpu_tensor_t* B,
    const contraction_spec_t* spec);

/**
 * Einstein summation
 */
gpu_tensor_t* tensor_einsum(
    tensor_ops_context_t* ctx,
    const char* subscripts,
    const gpu_tensor_t** tensors,
    size_t num_tensors);

/**
 * Trace of tensor
 */
bool tensor_trace(
    tensor_ops_context_t* ctx,
    const gpu_tensor_t* tensor,
    void* result);

/**
 * Partial trace
 */
gpu_tensor_t* tensor_partial_trace(
    tensor_ops_context_t* ctx,
    const gpu_tensor_t* tensor,
    const size_t* trace_indices,
    size_t num_trace);

// ============================================================================
// Quantum-Specific Operations
// ============================================================================

/**
 * Apply quantum gate to state tensor
 */
bool tensor_apply_gate(
    tensor_ops_context_t* ctx,
    gpu_tensor_t* state,
    const gpu_tensor_t* gate,
    const size_t* qubit_indices,
    size_t num_qubits);

/**
 * Batched gate application
 */
bool tensor_apply_gates_batched(
    tensor_ops_context_t* ctx,
    gpu_tensor_t* state,
    const gpu_tensor_t** gates,
    const size_t** qubit_indices,
    const size_t* num_qubits_per_gate,
    size_t num_gates);

/**
 * Compute expectation value <psi|O|psi>
 */
bool tensor_expectation(
    tensor_ops_context_t* ctx,
    const gpu_tensor_t* state,
    const gpu_tensor_t* observable,
    void* result);

/**
 * Compute state overlap <psi|phi>
 */
bool tensor_overlap(
    tensor_ops_context_t* ctx,
    const gpu_tensor_t* psi,
    const gpu_tensor_t* phi,
    void* result);

/**
 * Normalize quantum state
 */
bool tensor_normalize_state(
    tensor_ops_context_t* ctx,
    gpu_tensor_t* state);

// ============================================================================
// Decompositions
// ============================================================================

/**
 * Singular value decomposition
 */
bool tensor_svd(
    tensor_ops_context_t* ctx,
    const gpu_tensor_t* A,
    gpu_tensor_t** U,
    gpu_tensor_t** S,
    gpu_tensor_t** V);

/**
 * QR decomposition
 */
bool tensor_qr(
    tensor_ops_context_t* ctx,
    const gpu_tensor_t* A,
    gpu_tensor_t** Q,
    gpu_tensor_t** R);

/**
 * Eigenvalue decomposition
 */
bool tensor_eig(
    tensor_ops_context_t* ctx,
    const gpu_tensor_t* A,
    gpu_tensor_t** eigenvalues,
    gpu_tensor_t** eigenvectors);

// ============================================================================
// Memory Pool Operations
// ============================================================================

/**
 * Create memory pool
 */
tensor_memory_pool_t* tensor_pool_create(
    tensor_ops_context_t* ctx,
    size_t initial_size);

/**
 * Destroy memory pool
 */
void tensor_pool_destroy(tensor_memory_pool_t* pool);

/**
 * Allocate from pool
 */
void* tensor_pool_alloc(
    tensor_memory_pool_t* pool,
    size_t size);

/**
 * Free to pool
 */
void tensor_pool_free(
    tensor_memory_pool_t* pool,
    void* ptr);

/**
 * Get pool statistics
 */
bool tensor_pool_get_stats(
    tensor_memory_pool_t* pool,
    size_t* allocated,
    size_t* available,
    size_t* peak);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Get data type name
 */
const char* tensor_dtype_name(tensor_data_type_t dtype);

/**
 * Get storage format name
 */
const char* tensor_storage_name(tensor_storage_t storage);

/**
 * Get memory location name
 */
const char* tensor_memory_name(tensor_memory_t memory);

/**
 * Get element size for data type
 */
size_t tensor_dtype_size(tensor_data_type_t dtype);

/**
 * Get last error message
 */
const char* tensor_ops_get_last_error(tensor_ops_context_t* ctx);

/**
 * Query available GPU memory
 */
bool tensor_ops_query_memory(
    tensor_ops_context_t* ctx,
    size_t* free_bytes,
    size_t* total_bytes);

/**
 * Get number of available GPUs
 */
int tensor_ops_get_num_gpus(void);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_GEOMETRIC_TENSOR_OPS_H
