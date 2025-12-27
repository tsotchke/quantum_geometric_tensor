#ifndef QUANTUM_DISTRIBUTED_ENGINE_H
#define QUANTUM_DISTRIBUTED_ENGINE_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Backend Detection Macros
// ============================================================================

// CUDA/NCCL detection
#if defined(__NVCC__) || defined(USE_CUDA)
#define HAS_CUDA 1
#else
#define HAS_CUDA 0
#endif

#if defined(USE_NCCL)
#define HAS_NCCL 1
#else
#define HAS_NCCL 0
#endif

// MPI detection - only define if not already set via command line
#ifndef HAS_MPI
#if defined(USE_MPI)
#define HAS_MPI 1
#else
#define HAS_MPI 0
#endif
#endif

// Metal detection (Apple platforms)
#if defined(__APPLE__) && (defined(USE_METAL) || !HAS_CUDA)
#define HAS_METAL 1
#else
#define HAS_METAL 0
#endif

// OpenCL detection
#if defined(USE_OPENCL)
#define HAS_OPENCL 1
#else
#define HAS_OPENCL 0
#endif

// ============================================================================
// Backend Type Definitions
// ============================================================================

// Distributed backend type
typedef enum {
    DISTRIBUTED_BACKEND_CPU = 0,     // CPU-only with MPI
    DISTRIBUTED_BACKEND_CUDA,        // CUDA + NCCL
    DISTRIBUTED_BACKEND_METAL,       // Metal for Apple
    DISTRIBUTED_BACKEND_OPENCL       // OpenCL for cross-platform GPU
} DistributedBackendType;

// ============================================================================
// Forward Declarations for External Types
// ============================================================================

#if !HAS_CUDA
typedef int cudaStream_t;
typedef int cudaError_t;
#define cudaSuccess 0
#endif

#if !HAS_NCCL
typedef int ncclComm_t;
typedef int ncclUniqueId;
typedef int ncclResult_t;
typedef int ncclDataType_t;
#define ncclFloat32 0
#define ncclSuccess 0
#endif

#if !HAS_MPI
typedef int MPI_Comm;
typedef int MPI_Request;
typedef int MPI_Status;
typedef int MPI_Info;
#define MPI_COMM_WORLD 0
#define MPI_COMM_TYPE_SHARED 0
#define MPI_INFO_NULL 0
#endif

// ============================================================================
// Configuration Structures
// ============================================================================

// Performance monitor forward declaration
typedef struct PerformanceMonitor PerformanceMonitor;

// Monitor configuration
typedef struct {
    bool enable_timing;
    bool enable_memory_tracking;
    bool enable_bandwidth_tracking;
    size_t sample_interval_ms;
} MonitorConfig;

// Distributed configuration
typedef struct {
    int num_nodes;
    int gpus_per_node;
    size_t gpu_buffer_size;
    size_t host_buffer_size;
    bool use_nccl;
    bool use_rdma;
    DistributedBackendType preferred_backend;  // Preferred backend type
    MonitorConfig* monitor_config;
} DistributedConfig;

// Quantum operation type
typedef struct {
    void* data;
    size_t data_size;
    int operation_type;
    void* parameters;
    size_t param_size;
} QuantumOperation;

// Execution plan
typedef struct {
    int* node_assignments;
    size_t num_work_items;
    size_t work_item_size;
    void* workspace;
} ExecutionPlan;

// ============================================================================
// Opaque Handle Types
// ============================================================================

// Node context (internal)
typedef struct NodeContext NodeContext;

// Distributed context (opaque handle)
typedef struct DistributedContext DistributedContext;

// ============================================================================
// Core API Functions
// ============================================================================

/**
 * @brief Initialize the distributed engine
 *
 * Automatically selects the best available backend:
 * - CUDA + NCCL if available
 * - Metal on Apple platforms
 * - OpenCL for cross-platform GPU
 * - CPU with MPI/OpenMP as fallback
 *
 * @param config Configuration for distributed execution
 * @return Distributed context handle, or NULL on failure
 */
DistributedContext* init_distributed_engine(const DistributedConfig* config);

/**
 * @brief Execute a distributed quantum operation
 *
 * Distributes the operation across available nodes/devices using
 * the appropriate backend.
 *
 * @param ctx Distributed context
 * @param op Quantum operation to execute
 * @return 0 on success, negative error code on failure
 */
int execute_distributed_operation(DistributedContext* ctx, const QuantumOperation* op);

/**
 * @brief Clean up and release distributed engine resources
 *
 * @param ctx Distributed context to clean up
 */
void cleanup_distributed_engine(DistributedContext* ctx);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Calculate data size for a quantum operation
 */
size_t calculate_data_size(const QuantumOperation* op);

/**
 * @brief Create an execution plan for a quantum operation
 */
ExecutionPlan* create_execution_plan(DistributedContext* ctx, const QuantumOperation* op);

/**
 * @brief Clean up an execution plan
 */
void cleanup_execution(ExecutionPlan* plan);

/**
 * @brief Update performance metrics after operation
 */
void update_performance_metrics(PerformanceMonitor* monitor,
                                const QuantumOperation* op,
                                const ExecutionPlan* plan);

/**
 * @brief Get the active backend type
 */
DistributedBackendType get_distributed_backend_type(const DistributedContext* ctx);

/**
 * @brief Check if a backend is available on this system
 */
bool is_backend_available(DistributedBackendType backend);

// ============================================================================
// Backend-Specific Functions (Internal Use)
// ============================================================================

#if HAS_CUDA
/**
 * @brief Launch GPU kernel for quantum operation
 */
void launch_gpu_kernel(void* buffer, const QuantumOperation* op,
                       const ExecutionPlan* plan, cudaStream_t stream);
#endif

/**
 * @brief Initialize performance monitor for distributed operations
 */
PerformanceMonitor* init_performance_monitor_distributed(MonitorConfig* config);

/**
 * @brief Cleanup performance monitor for distributed operations
 */
void cleanup_performance_monitor_distributed(PerformanceMonitor* monitor);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_DISTRIBUTED_ENGINE_H
