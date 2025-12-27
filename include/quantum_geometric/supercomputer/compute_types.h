#ifndef COMPUTE_TYPES_H
#define COMPUTE_TYPES_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Platform Detection
// ============================================================================

// CUDA detection
#if defined(__NVCC__) || defined(USE_CUDA) || defined(QGT_HAS_CUDA)
#define COMPUTE_HAS_CUDA 1
#else
#define COMPUTE_HAS_CUDA 0
#endif

// NCCL detection
#if defined(USE_NCCL) || defined(QGT_HAS_NCCL)
#define COMPUTE_HAS_NCCL 1
#else
#define COMPUTE_HAS_NCCL 0
#endif

// MPI detection
#if defined(USE_MPI) || defined(QGT_HAS_MPI)
#define COMPUTE_HAS_MPI 1
#else
#define COMPUTE_HAS_MPI 0
#endif

// Metal detection (Apple platforms)
#if defined(__APPLE__) && (defined(USE_METAL) || defined(QGT_HAS_METAL) || !COMPUTE_HAS_CUDA)
#define COMPUTE_HAS_METAL 1
#else
#define COMPUTE_HAS_METAL 0
#endif

// OpenCL detection
#if defined(USE_OPENCL) || defined(QGT_HAS_OPENCL)
#define COMPUTE_HAS_OPENCL 1
#else
#define COMPUTE_HAS_OPENCL 0
#endif

// SIMD detection
#if defined(__x86_64__) || defined(_M_X64)
#define COMPUTE_ARCH_X86_64 1
#define COMPUTE_HAS_AVX2 1
#else
#define COMPUTE_ARCH_X86_64 0
#define COMPUTE_HAS_AVX2 0
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#define COMPUTE_ARCH_ARM64 1
#define COMPUTE_HAS_NEON 1
#else
#define COMPUTE_ARCH_ARM64 0
#define COMPUTE_HAS_NEON 0
#endif

// ============================================================================
// Backend Types
// ============================================================================

typedef enum {
    COMPUTE_BACKEND_AUTO = -1,   // Auto-select best available backend
    COMPUTE_BACKEND_CPU = 0,     // CPU with OpenMP + SIMD
    COMPUTE_BACKEND_CUDA,        // NVIDIA CUDA + NCCL
    COMPUTE_BACKEND_METAL,       // Apple Metal + Accelerate
    COMPUTE_BACKEND_OPENCL,      // Cross-platform OpenCL
    COMPUTE_BACKEND_COUNT
} ComputeBackendType;

// ============================================================================
// Memory Types
// ============================================================================

typedef enum {
    COMPUTE_MEM_HOST = 0,        // Host (CPU) memory
    COMPUTE_MEM_DEVICE,          // Device (GPU) memory
    COMPUTE_MEM_UNIFIED,         // Unified/managed memory
    COMPUTE_MEM_PINNED           // Pinned host memory for fast transfers
} ComputeMemType;

// ============================================================================
// Data Types for Communication
// ============================================================================

typedef enum {
    COMPUTE_DTYPE_FLOAT32 = 0,
    COMPUTE_DTYPE_FLOAT64,
    COMPUTE_DTYPE_COMPLEX64,     // Complex float (2 x float32)
    COMPUTE_DTYPE_COMPLEX128,    // Complex double (2 x float64)
    COMPUTE_DTYPE_INT32,
    COMPUTE_DTYPE_INT64,
    COMPUTE_DTYPE_UINT8
} ComputeDataType;

// ============================================================================
// Reduction Operations
// ============================================================================

typedef enum {
    COMPUTE_REDUCE_SUM = 0,
    COMPUTE_REDUCE_PROD,
    COMPUTE_REDUCE_MIN,
    COMPUTE_REDUCE_MAX,
    COMPUTE_REDUCE_AVG
} ComputeReduceOp;

// ============================================================================
// Result Codes
// ============================================================================

typedef enum {
    COMPUTE_SUCCESS = 0,
    COMPUTE_ERROR_INVALID_ARGUMENT,
    COMPUTE_ERROR_OUT_OF_MEMORY,
    COMPUTE_ERROR_DEVICE_NOT_FOUND,
    COMPUTE_ERROR_BACKEND_NOT_AVAILABLE,
    COMPUTE_ERROR_COMMUNICATION_FAILED,
    COMPUTE_ERROR_SYNCHRONIZATION_FAILED,
    COMPUTE_ERROR_KERNEL_FAILED,
    COMPUTE_ERROR_NOT_IMPLEMENTED,
    COMPUTE_ERROR_INTERNAL
} ComputeResult;

// ============================================================================
// Quantum Operation Types
// ============================================================================

typedef enum {
    QUANTUM_OP_UNITARY = 0,      // Unitary transformation
    QUANTUM_OP_MEASUREMENT,       // Quantum measurement
    QUANTUM_OP_TENSOR_CONTRACT,   // Tensor network contraction
    QUANTUM_OP_GRADIENT,          // Quantum gradient computation
    QUANTUM_OP_NORMALIZE,         // State normalization
    QUANTUM_OP_INNER_PRODUCT,     // Inner product of states
    QUANTUM_OP_EXPECTATION,       // Expectation value computation
    QUANTUM_OP_DENSITY_MATRIX     // Density matrix operations
} QuantumOpType;

// ============================================================================
// Network Topology Types
// ============================================================================

typedef enum {
    COMPUTE_TOPO_SINGLE = 0,     // Single node
    COMPUTE_TOPO_RING,           // Ring topology
    COMPUTE_TOPO_TREE,           // Tree/hierarchical topology
    COMPUTE_TOPO_MESH,           // 2D mesh topology
    COMPUTE_TOPO_TORUS,          // 3D torus topology
    COMPUTE_TOPO_FULLY_CONNECTED, // Fully connected (all-to-all)
    COMPUTE_TOPO_HYBRID          // Hybrid topology
} ComputeTopologyType;

// ============================================================================
// Configuration Structures
// ============================================================================

// Monitor configuration
typedef struct {
    bool enable_timing;
    bool enable_memory_tracking;
    bool enable_bandwidth_tracking;
    bool enable_power_monitoring;
    size_t sample_interval_ms;
    size_t history_size;
} ComputeMonitorConfig;

// Node configuration
typedef struct {
    int node_id;
    int num_devices;             // GPUs or accelerators
    int num_cores;               // CPU cores
    size_t memory_per_device;    // Per-device memory
    size_t host_memory;          // Host memory
    double network_bandwidth;    // GB/s
    double network_latency;      // microseconds
} ComputeNodeConfig;

// Backend-specific configuration (opaque)
typedef struct ComputeBackendConfig ComputeBackendConfig;

// Main distributed configuration
typedef struct {
    // Cluster configuration
    int num_nodes;
    int devices_per_node;
    ComputeTopologyType topology;

    // Per-process rank information (for MPI-like environments)
    int node_rank;               // Rank of this node in the cluster
    int local_rank;              // Local rank within this node
    int local_size;              // Number of processes on this node

    // Buffer sizes
    size_t device_buffer_size;
    size_t host_buffer_size;
    size_t comm_buffer_size;

    // Communication options
    bool use_nccl;               // Use NCCL for GPU communication
    bool use_rdma;               // Use RDMA for low-latency networking
    bool use_compression;        // Compress data for communication

    // Backend selection
    ComputeBackendType preferred_backend;
    bool allow_fallback;         // Fall back to lower-priority backend if preferred unavailable

    // Performance tuning
    int num_streams;             // Concurrent execution streams
    int num_threads_per_node;    // OpenMP threads per node
    bool enable_async;           // Enable asynchronous execution

    // Monitoring
    ComputeMonitorConfig* monitor_config;

    // Backend-specific configuration
    ComputeBackendConfig* backend_config;
} ComputeDistributedConfig;

// ============================================================================
// Operation Structures
// ============================================================================

// Quantum operation descriptor
typedef struct {
    QuantumOpType type;

    // Input data
    void* input_data;
    size_t input_size;
    ComputeDataType input_dtype;

    // Output data
    void* output_data;
    size_t output_size;
    ComputeDataType output_dtype;

    // Operation parameters
    void* parameters;
    size_t param_size;

    // Dimensions for tensor operations
    size_t* dims;
    size_t num_dims;

    // For unitary operations
    size_t num_qubits;
    size_t* target_qubits;
    size_t num_targets;

    // For gradient operations
    float* parameter_gradients;
    size_t num_parameters;
} ComputeQuantumOp;

// Execution plan
typedef struct {
    // Work distribution
    int* node_assignments;       // Which node handles which partition
    size_t num_partitions;
    size_t partition_size;

    // Memory layout
    size_t* offsets;             // Data offsets for each partition
    size_t* sizes;               // Data sizes for each partition

    // Communication pattern
    int* send_targets;           // Nodes to send to
    int* recv_sources;           // Nodes to receive from
    size_t num_comm_ops;

    // Workspace
    void* workspace;
    size_t workspace_size;

    // Scheduling hints
    int priority;
    bool requires_sync;
} ComputeExecutionPlan;

// ============================================================================
// Handle Types (Opaque)
// ============================================================================

// Forward declarations for opaque handles
typedef struct ComputeBackend ComputeBackend;
typedef struct ComputeStream ComputeStream;
typedef struct ComputeEvent ComputeEvent;
typedef struct ComputeEngine ComputeEngine;
typedef struct ComputeBuffer ComputeBuffer;
typedef struct ComputeKernel ComputeKernel;

// ============================================================================
// Callback Types
// ============================================================================

// Progress callback for long operations
typedef void (*ComputeProgressCallback)(size_t completed, size_t total, void* user_data);

// Error callback for asynchronous errors
typedef void (*ComputeErrorCallback)(ComputeResult error, const char* message, void* user_data);

// Completion callback for async operations
typedef void (*ComputeCompletionCallback)(ComputeResult result, void* output, void* user_data);

// ============================================================================
// Performance Metrics
// ============================================================================

typedef struct {
    // Timing
    double total_time_ms;
    double compute_time_ms;
    double communication_time_ms;
    double synchronization_time_ms;
    double execution_time;           // Total execution time (seconds)

    // Throughput
    double operations_per_second;
    double flops;
    double bandwidth_gbps;

    // Memory
    size_t peak_memory_bytes;
    size_t current_memory_bytes;
    size_t memory_used;              // Current memory usage (bytes)
    double memory_efficiency;

    // Communication
    size_t bytes_sent;
    size_t bytes_received;
    size_t num_messages;
    double avg_latency_us;

    // Quantum-specific
    double gate_fidelity;
    double state_fidelity;
    double error_rate;
} ComputeMetrics;

// ============================================================================
// Utility Macros
// ============================================================================

// Get size of data type in bytes
static inline size_t compute_dtype_size(ComputeDataType dtype) {
    switch (dtype) {
        case COMPUTE_DTYPE_FLOAT32:   return 4;
        case COMPUTE_DTYPE_FLOAT64:   return 8;
        case COMPUTE_DTYPE_COMPLEX64: return 8;
        case COMPUTE_DTYPE_COMPLEX128: return 16;
        case COMPUTE_DTYPE_INT32:     return 4;
        case COMPUTE_DTYPE_INT64:     return 8;
        case COMPUTE_DTYPE_UINT8:     return 1;
        default:                      return 0;
    }
}

// Get string name for backend type
static inline const char* compute_backend_name(ComputeBackendType type) {
    switch (type) {
        case COMPUTE_BACKEND_CPU:    return "CPU";
        case COMPUTE_BACKEND_CUDA:   return "CUDA";
        case COMPUTE_BACKEND_METAL:  return "Metal";
        case COMPUTE_BACKEND_OPENCL: return "OpenCL";
        default:                     return "Unknown";
    }
}

// Get string name for result code
static inline const char* compute_result_string(ComputeResult result) {
    switch (result) {
        case COMPUTE_SUCCESS:                    return "Success";
        case COMPUTE_ERROR_INVALID_ARGUMENT:     return "Invalid argument";
        case COMPUTE_ERROR_OUT_OF_MEMORY:        return "Out of memory";
        case COMPUTE_ERROR_DEVICE_NOT_FOUND:     return "Device not found";
        case COMPUTE_ERROR_BACKEND_NOT_AVAILABLE: return "Backend not available";
        case COMPUTE_ERROR_COMMUNICATION_FAILED: return "Communication failed";
        case COMPUTE_ERROR_SYNCHRONIZATION_FAILED: return "Synchronization failed";
        case COMPUTE_ERROR_KERNEL_FAILED:        return "Kernel execution failed";
        case COMPUTE_ERROR_NOT_IMPLEMENTED:      return "Not implemented";
        case COMPUTE_ERROR_INTERNAL:             return "Internal error";
        default:                                 return "Unknown error";
    }
}

#ifdef __cplusplus
}
#endif

#endif // COMPUTE_TYPES_H
