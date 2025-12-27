#ifndef COMPUTE_BACKEND_H
#define COMPUTE_BACKEND_H

#include "compute_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Backend Operations VTable
// ============================================================================

/**
 * Backend operations vtable - polymorphic interface for all compute backends.
 * Each backend (CUDA, Metal, OpenCL, CPU) implements this interface.
 */
typedef struct ComputeBackendOps {
    // ========================================================================
    // Lifecycle Operations
    // ========================================================================

    /**
     * Initialize the backend with given configuration.
     * @param config Distributed configuration
     * @return Initialized backend handle, or NULL on failure
     */
    ComputeBackend* (*init)(const ComputeDistributedConfig* config);

    /**
     * Clean up and release all backend resources.
     * @param backend Backend to clean up
     */
    void (*cleanup)(ComputeBackend* backend);

    /**
     * Check if this backend is available on the current system.
     * @return true if backend can be used, false otherwise
     */
    bool (*probe)(void);

    /**
     * Get backend capabilities and properties.
     * @param backend Backend handle
     * @param num_devices Output: number of available devices
     * @param total_memory Output: total device memory in bytes
     * @return COMPUTE_SUCCESS or error code
     */
    ComputeResult (*get_capabilities)(ComputeBackend* backend,
                                       int* num_devices,
                                       size_t* total_memory);

    // ========================================================================
    // Memory Management
    // ========================================================================

    /**
     * Allocate memory of specified type.
     * @param backend Backend handle
     * @param size Size in bytes to allocate
     * @param mem_type Type of memory (host, device, unified, pinned)
     * @return Pointer to allocated memory, or NULL on failure
     */
    void* (*alloc)(ComputeBackend* backend, size_t size, ComputeMemType mem_type);

    /**
     * Free previously allocated memory.
     * @param backend Backend handle
     * @param ptr Pointer to memory to free
     * @param mem_type Type of memory
     */
    void (*free)(ComputeBackend* backend, void* ptr, ComputeMemType mem_type);

    /**
     * Copy memory between locations.
     * @param backend Backend handle
     * @param dst Destination pointer
     * @param dst_type Destination memory type
     * @param src Source pointer
     * @param src_type Source memory type
     * @param size Size in bytes to copy
     * @param stream Stream for async copy (NULL for sync)
     * @return COMPUTE_SUCCESS or error code
     */
    ComputeResult (*memcpy)(ComputeBackend* backend,
                            void* dst, ComputeMemType dst_type,
                            const void* src, ComputeMemType src_type,
                            size_t size, ComputeStream* stream);

    /**
     * Set memory to a value.
     * @param backend Backend handle
     * @param ptr Pointer to memory
     * @param value Value to set (as int, will be converted appropriately)
     * @param size Size in bytes
     * @param stream Stream for async operation (NULL for sync)
     * @return COMPUTE_SUCCESS or error code
     */
    ComputeResult (*memset)(ComputeBackend* backend,
                            void* ptr, int value, size_t size,
                            ComputeStream* stream);

    // ========================================================================
    // Stream Management
    // ========================================================================

    /**
     * Create an execution stream for async operations.
     * @param backend Backend handle
     * @return New stream handle, or NULL on failure
     */
    ComputeStream* (*create_stream)(ComputeBackend* backend);

    /**
     * Destroy an execution stream.
     * @param backend Backend handle
     * @param stream Stream to destroy
     */
    void (*destroy_stream)(ComputeBackend* backend, ComputeStream* stream);

    /**
     * Synchronize a stream (wait for all operations to complete).
     * @param backend Backend handle
     * @param stream Stream to synchronize (NULL for all streams)
     * @return COMPUTE_SUCCESS or error code
     */
    ComputeResult (*synchronize_stream)(ComputeBackend* backend, ComputeStream* stream);

    /**
     * Create an event for timing/synchronization.
     * @param backend Backend handle
     * @return New event handle, or NULL on failure
     */
    ComputeEvent* (*create_event)(ComputeBackend* backend);

    /**
     * Destroy an event.
     * @param backend Backend handle
     * @param event Event to destroy
     */
    void (*destroy_event)(ComputeBackend* backend, ComputeEvent* event);

    /**
     * Record an event in a stream.
     * @param backend Backend handle
     * @param event Event to record
     * @param stream Stream to record in
     * @return COMPUTE_SUCCESS or error code
     */
    ComputeResult (*record_event)(ComputeBackend* backend,
                                   ComputeEvent* event,
                                   ComputeStream* stream);

    /**
     * Wait for an event in a stream.
     * @param backend Backend handle
     * @param stream Stream to wait in
     * @param event Event to wait for
     * @return COMPUTE_SUCCESS or error code
     */
    ComputeResult (*wait_event)(ComputeBackend* backend,
                                 ComputeStream* stream,
                                 ComputeEvent* event);

    // ========================================================================
    // Quantum Operations
    // ========================================================================

    /**
     * Apply a unitary transformation to quantum state.
     * @param backend Backend handle
     * @param state State vector (complex, interleaved real/imag)
     * @param state_size Size of state vector (number of complex elements)
     * @param unitary Unitary matrix (row-major, complex)
     * @param unitary_size Size of unitary matrix dimension
     * @param stream Stream for async execution
     * @return COMPUTE_SUCCESS or error code
     */
    ComputeResult (*quantum_unitary)(ComputeBackend* backend,
                                      float* state, size_t state_size,
                                      const float* unitary, size_t unitary_size,
                                      ComputeStream* stream);

    /**
     * Normalize a quantum state vector.
     * @param backend Backend handle
     * @param state State vector (complex, interleaved)
     * @param size Number of complex elements
     * @param stream Stream for async execution
     * @return COMPUTE_SUCCESS or error code
     */
    ComputeResult (*quantum_normalize)(ComputeBackend* backend,
                                        float* state, size_t size,
                                        ComputeStream* stream);

    /**
     * Contract two tensors in a tensor network.
     * @param backend Backend handle
     * @param result Output tensor
     * @param a First input tensor
     * @param b Second input tensor
     * @param m First dimension of a
     * @param n Contraction dimension
     * @param k Second dimension of b
     * @param stream Stream for async execution
     * @return COMPUTE_SUCCESS or error code
     */
    ComputeResult (*quantum_tensor_contract)(ComputeBackend* backend,
                                              float* result,
                                              const float* a, const float* b,
                                              size_t m, size_t n, size_t k,
                                              ComputeStream* stream);

    /**
     * Compute quantum gradients (parameter-shift rule).
     * @param backend Backend handle
     * @param gradients Output gradient vector
     * @param forward_state Forward pass state
     * @param backward_state Backward pass state
     * @param size Number of elements
     * @param stream Stream for async execution
     * @return COMPUTE_SUCCESS or error code
     */
    ComputeResult (*quantum_gradient)(ComputeBackend* backend,
                                       float* gradients,
                                       const float* forward_state,
                                       const float* backward_state,
                                       size_t size,
                                       ComputeStream* stream);

    /**
     * Compute inner product of two quantum states.
     * @param backend Backend handle
     * @param result Output: inner product (complex, 2 floats)
     * @param state_a First state vector
     * @param state_b Second state vector
     * @param size Number of complex elements
     * @param stream Stream for async execution
     * @return COMPUTE_SUCCESS or error code
     */
    ComputeResult (*quantum_inner_product)(ComputeBackend* backend,
                                            float* result,
                                            const float* state_a,
                                            const float* state_b,
                                            size_t size,
                                            ComputeStream* stream);

    /**
     * Compute expectation value of an observable.
     * @param backend Backend handle
     * @param result Output: expectation value (real)
     * @param state State vector
     * @param observable Observable matrix (Hermitian)
     * @param size State vector size
     * @param stream Stream for async execution
     * @return COMPUTE_SUCCESS or error code
     */
    ComputeResult (*quantum_expectation)(ComputeBackend* backend,
                                          float* result,
                                          const float* state,
                                          const float* observable,
                                          size_t size,
                                          ComputeStream* stream);

    // ========================================================================
    // Collective Communication
    // ========================================================================

    /**
     * Barrier synchronization across all nodes.
     * @param backend Backend handle
     * @return COMPUTE_SUCCESS or error code
     */
    ComputeResult (*barrier)(ComputeBackend* backend);

    /**
     * Broadcast data from root to all nodes.
     * @param backend Backend handle
     * @param data Data buffer (in/out)
     * @param size Size in bytes
     * @param dtype Data type
     * @param root Root node rank
     * @return COMPUTE_SUCCESS or error code
     */
    ComputeResult (*broadcast)(ComputeBackend* backend,
                                void* data, size_t size,
                                ComputeDataType dtype, int root);

    /**
     * All-reduce operation (sum/min/max across all nodes).
     * @param backend Backend handle
     * @param send_data Input data
     * @param recv_data Output data (can be same as send_data)
     * @param count Number of elements
     * @param dtype Data type
     * @param op Reduction operation
     * @return COMPUTE_SUCCESS or error code
     */
    ComputeResult (*allreduce)(ComputeBackend* backend,
                                const void* send_data, void* recv_data,
                                size_t count, ComputeDataType dtype,
                                ComputeReduceOp op);

    /**
     * Scatter data from root to all nodes.
     * @param backend Backend handle
     * @param send_data Data to send (only valid on root)
     * @param recv_data Receive buffer
     * @param count Elements per node
     * @param dtype Data type
     * @param root Root node rank
     * @return COMPUTE_SUCCESS or error code
     */
    ComputeResult (*scatter)(ComputeBackend* backend,
                              const void* send_data, void* recv_data,
                              size_t count, ComputeDataType dtype, int root);

    /**
     * Gather data from all nodes to root.
     * @param backend Backend handle
     * @param send_data Data to send
     * @param recv_data Receive buffer (only valid on root)
     * @param count Elements per node
     * @param dtype Data type
     * @param root Root node rank
     * @return COMPUTE_SUCCESS or error code
     */
    ComputeResult (*gather)(ComputeBackend* backend,
                             const void* send_data, void* recv_data,
                             size_t count, ComputeDataType dtype, int root);

    /**
     * All-gather: gather data from all nodes to all nodes.
     * @param backend Backend handle
     * @param send_data Data to send
     * @param recv_data Receive buffer (count * num_nodes elements)
     * @param count Elements per node
     * @param dtype Data type
     * @return COMPUTE_SUCCESS or error code
     */
    ComputeResult (*allgather)(ComputeBackend* backend,
                                const void* send_data, void* recv_data,
                                size_t count, ComputeDataType dtype);

    /**
     * Reduce-scatter: reduce and scatter result.
     * @param backend Backend handle
     * @param send_data Data to send
     * @param recv_data Receive buffer
     * @param count Elements per node after scatter
     * @param dtype Data type
     * @param op Reduction operation
     * @return COMPUTE_SUCCESS or error code
     */
    ComputeResult (*reduce_scatter)(ComputeBackend* backend,
                                     const void* send_data, void* recv_data,
                                     size_t count, ComputeDataType dtype,
                                     ComputeReduceOp op);

    // ========================================================================
    // Execution & Scheduling
    // ========================================================================

    /**
     * Execute a quantum operation according to plan.
     * @param backend Backend handle
     * @param op Quantum operation descriptor
     * @param plan Execution plan
     * @param stream Stream for async execution
     * @return COMPUTE_SUCCESS or error code
     */
    ComputeResult (*execute)(ComputeBackend* backend,
                              const ComputeQuantumOp* op,
                              const ComputeExecutionPlan* plan,
                              ComputeStream* stream);

    /**
     * Create an execution plan for an operation.
     * @param backend Backend handle
     * @param op Quantum operation descriptor
     * @return Execution plan, or NULL on failure
     */
    ComputeExecutionPlan* (*create_plan)(ComputeBackend* backend,
                                          const ComputeQuantumOp* op);

    /**
     * Destroy an execution plan.
     * @param backend Backend handle
     * @param plan Plan to destroy
     */
    void (*destroy_plan)(ComputeBackend* backend, ComputeExecutionPlan* plan);

    // ========================================================================
    // Performance Monitoring
    // ========================================================================

    /**
     * Get current performance metrics.
     * @param backend Backend handle
     * @param metrics Output metrics structure
     * @return COMPUTE_SUCCESS or error code
     */
    ComputeResult (*get_metrics)(ComputeBackend* backend, ComputeMetrics* metrics);

    /**
     * Reset performance counters.
     * @param backend Backend handle
     * @return COMPUTE_SUCCESS or error code
     */
    ComputeResult (*reset_metrics)(ComputeBackend* backend);

} ComputeBackendOps;

// ============================================================================
// Backend Information
// ============================================================================

/**
 * Backend registration information.
 */
typedef struct {
    ComputeBackendType type;     // Backend type identifier
    const char* name;            // Human-readable name
    const char* version;         // Backend version string
    int priority;                // Selection priority (higher = preferred)
    const ComputeBackendOps* ops; // Operations vtable
} ComputeBackendInfo;

// ============================================================================
// Backend Registry Functions
// ============================================================================

/**
 * Register a backend implementation.
 * Called by each backend's constructor function.
 * @param info Backend information and operations
 * @return COMPUTE_SUCCESS or error code
 */
ComputeResult compute_register_backend(const ComputeBackendInfo* info);

/**
 * Get number of registered backends.
 * @return Number of backends
 */
int compute_get_backend_count(void);

/**
 * Get backend info by index.
 * @param index Backend index (0 to count-1)
 * @return Backend info, or NULL if index invalid
 */
const ComputeBackendInfo* compute_get_backend_info(int index);

/**
 * Get backend info by type.
 * @param type Backend type
 * @return Backend info, or NULL if not found
 */
const ComputeBackendInfo* compute_get_backend_by_type(ComputeBackendType type);

/**
 * Select the best available backend.
 * @param preferred Preferred backend type, or -1 for auto-select
 * @param allow_fallback Allow falling back to lower-priority backend
 * @return Best available backend info, or NULL if none available
 */
const ComputeBackendInfo* compute_select_backend(ComputeBackendType preferred,
                                                   bool allow_fallback);

/**
 * Check if a specific backend is available.
 * @param type Backend type to check
 * @return true if available, false otherwise
 */
bool compute_backend_available(ComputeBackendType type);

// ============================================================================
// Engine Functions (High-Level API)
// ============================================================================

/**
 * Initialize the compute engine with automatic backend selection.
 * @param config Distributed configuration
 * @return Engine handle, or NULL on failure
 */
ComputeEngine* compute_engine_init(const ComputeDistributedConfig* config);

/**
 * Clean up the compute engine.
 * @param engine Engine to clean up
 */
void compute_engine_cleanup(ComputeEngine* engine);

/**
 * Get the active backend type.
 * @param engine Engine handle
 * @return Active backend type
 */
ComputeBackendType compute_engine_get_backend_type(const ComputeEngine* engine);

/**
 * Get the backend operations for the engine.
 * @param engine Engine handle
 * @return Backend operations vtable
 */
const ComputeBackendOps* compute_engine_get_ops(const ComputeEngine* engine);

/**
 * Get the underlying backend handle.
 * @param engine Engine handle
 * @return Backend handle
 */
ComputeBackend* compute_engine_get_backend(const ComputeEngine* engine);

// ============================================================================
// Convenience Macros for Backend Implementation
// ============================================================================

/**
 * Macro for declaring backend registration constructor.
 * Use this at the end of each backend implementation file.
 */
#define COMPUTE_REGISTER_BACKEND(backend_type, backend_name, backend_version, backend_priority, backend_ops) \
    __attribute__((constructor)) \
    static void register_##backend_type##_backend(void) { \
        static const ComputeBackendInfo info = { \
            .type = backend_type, \
            .name = backend_name, \
            .version = backend_version, \
            .priority = backend_priority, \
            .ops = &backend_ops \
        }; \
        compute_register_backend(&info); \
    }

#ifdef __cplusplus
}
#endif

#endif // COMPUTE_BACKEND_H
