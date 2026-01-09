#ifndef QUANTUM_GEOMETRIC_GPU_H
#define QUANTUM_GEOMETRIC_GPU_H

#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/memory_pool.h"
#include "quantum_geometric/core/error_codes.h"
#include <stddef.h>
#include <stdbool.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================

#define MAX_QUEUED_KERNELS 256
#define WARP_SIZE 32
#define MEMORY_POOL_BLOCK_SIZE (1024 * 1024)  // 1 MB blocks
#define MAX_GPU_NAME_LENGTH 256

// ============================================================================
// GPU Error Codes (for backward compatibility)
// ============================================================================

typedef enum {
    QG_GPU_SUCCESS = 0,
    QG_GPU_ERROR_NO_DEVICE = -1,
    QG_GPU_ERROR_NOT_INITIALIZED = -2,
    QG_GPU_ERROR_OUT_OF_MEMORY = -3,
    QG_GPU_ERROR_INVALID_DEVICE = -4,
    QG_GPU_ERROR_INVALID_VALUE = -5,
    QG_GPU_ERROR_LAUNCH_FAILED = -6,
    QG_GPU_ERROR_SYNC_FAILED = -7,
    QG_GPU_ERROR_INTERNAL = -8
} gpu_error_t;

// Additional GPU error codes (qgt_error_t values)
#define QGT_ERROR_GPU_NOT_AVAILABLE 300
#define QGT_ERROR_GPU_OUT_OF_MEMORY 301
#define QGT_ERROR_GPU_INVALID_VALUE 302
#define QGT_ERROR_GPU_LAUNCH_FAILED 303
#define QGT_ERROR_GPU_SYNC_FAILED 304
#define QGT_ERROR_GPU_INTERNAL 305

// ============================================================================
// Type Definitions
// ============================================================================

/**
 * @brief Grid/block dimensions (compatible with CUDA dim3)
 */
typedef struct {
    unsigned int x;
    unsigned int y;
    unsigned int z;
} dim3;

/**
 * @brief GPU buffer for memory management
 */
typedef struct {
    void* device_ptr;       ///< Pointer to device memory
    size_t size;            ///< Size of allocation in bytes
    bool is_pinned;         ///< Whether memory is pinned (page-locked)
} gpu_buffer_t;

/**
 * @brief GPU device state (internal)
 */
typedef struct {
    int device_id;
    char name[MAX_GPU_NAME_LENGTH];
    size_t total_memory;
    size_t free_memory;
    size_t used_memory;
    int compute_capability;
    int num_multiprocessors;
    int max_threads_per_block;
    int warp_size;
    int max_block_dimensions[3];
    int max_grid_dimensions[3];
    bool unified_memory;
    bool concurrent_kernels;
    void* device_handle;
} gpu_device_state_t;

/**
 * @brief GPU memory pool
 */
typedef struct {
    void* base_ptr;
    size_t total_size;
    size_t used_size;
    size_t block_size;
    bool uses_unified_memory;
    pthread_mutex_t lock;
    void* free_list;
} gpu_memory_pool_t;

/**
 * @brief Kernel launch information
 */
typedef struct {
    const char* name;
    void* args;
    size_t args_size;
    dim3 grid;
    dim3 block;
    size_t shared_memory;
    int stream_id;
    bool completed;
} kernel_launch_info_t;

/**
 * @brief GPU device info (public API)
 */
typedef struct {
    int device_id;
    char name[MAX_GPU_NAME_LENGTH];
    size_t total_memory;
    size_t free_memory;
    int compute_capability_major;
    int compute_capability_minor;
    int max_threads_per_block;
    int max_block_dimensions[3];
    int max_grid_dimensions[3];
    int compute_units;
    int backend_type;
    bool supports_unified_memory;
} gpu_device_info_t;

// ============================================================================
// GPU Memory Management (low-level)
// ============================================================================

qgt_error_t gpu_malloc(void** ptr, size_t size);
qgt_error_t qgt_gpu_free_buffer(void* ptr);  // Context-free version
qgt_error_t gpu_memcpy_host_to_device(void* dst, const void* src, size_t size);
qgt_error_t gpu_memcpy_device_to_host(void* dst, const void* src, size_t size);

// GPU memory pool integration
void gpu_free_pooled(MemoryPool* pool, void* ptr);

// ============================================================================
// GPU System Functions
// ============================================================================

/**
 * @brief Initialize GPU system
 * @return QG_GPU_SUCCESS on success, error code otherwise
 */
int qg_gpu_init(void);

/**
 * @brief Cleanup and shutdown GPU system
 */
void qg_gpu_cleanup(void);

/**
 * @brief Shutdown GPU system (alias for qg_gpu_cleanup)
 */
void qg_gpu_shutdown(void);

/**
 * @brief Get number of available GPU devices
 * @param count Output parameter for device count
 * @return QG_GPU_SUCCESS on success, error code otherwise
 */
int qg_gpu_get_device_count(int* count);

/**
 * @brief Get device information
 * @param device_id Device index
 * @param info Output device info
 * @return QG_GPU_SUCCESS on success
 */
int qg_gpu_get_device_info(int device_id, gpu_device_info_t* info);

/**
 * @brief Set the current GPU device
 * @param device_id Device index to set as current
 * @return QG_GPU_SUCCESS on success
 */
int qg_gpu_set_device(int device_id);

/**
 * @brief Get last error code
 * @return Last error that occurred
 */
gpu_error_t qg_gpu_get_last_error(void);

/**
 * @brief Get error string for error code
 * @param error Error code
 * @return Human-readable error string
 */
const char* qg_gpu_get_error_string(gpu_error_t error);

// ============================================================================
// GPU Buffer Management
// ============================================================================

/**
 * @brief Allocate GPU buffer
 * @param buffer Output buffer structure
 * @param size Size in bytes to allocate
 * @return QG_GPU_SUCCESS on success
 */
int qg_gpu_allocate(gpu_buffer_t* buffer, size_t size);

/**
 * @brief Allocate pinned (page-locked) GPU buffer
 * @param buffer Output buffer structure
 * @param size Size in bytes to allocate
 * @return QG_GPU_SUCCESS on success
 */
int qg_gpu_allocate_pinned(gpu_buffer_t* buffer, size_t size);

/**
 * @brief Free GPU buffer
 * @param buffer Buffer to free
 * @return QG_GPU_SUCCESS on success
 */
int qg_gpu_free(gpu_buffer_t* buffer);

/**
 * @brief Copy data from host to device buffer
 * @param dst Destination GPU buffer
 * @param src Source host pointer
 * @param size Size in bytes to copy
 * @return QG_GPU_SUCCESS on success
 */
int qg_gpu_memcpy_to_device(gpu_buffer_t* dst, const void* src, size_t size);

/**
 * @brief Copy data from device buffer to host
 * @param dst Destination host pointer
 * @param src Source GPU buffer
 * @param size Size in bytes to copy
 * @return QG_GPU_SUCCESS on success
 */
int qg_gpu_memcpy_to_host(void* dst, const gpu_buffer_t* src, size_t size);

// ============================================================================
// GPU Stream Management
// ============================================================================

/**
 * @brief Create a GPU stream
 * @param stream_id Output stream identifier
 * @return QG_GPU_SUCCESS on success
 */
int qg_gpu_create_stream(int* stream_id);

/**
 * @brief Destroy a GPU stream
 * @param stream_id Stream to destroy
 * @return QG_GPU_SUCCESS on success
 */
int qg_gpu_destroy_stream(int stream_id);

/**
 * @brief Synchronize a specific stream
 * @param stream_id Stream to synchronize
 * @return QG_GPU_SUCCESS on success
 */
int qg_gpu_synchronize_stream(int stream_id);

/**
 * @brief Synchronize all GPU operations
 * @return QG_GPU_SUCCESS on success
 */
int qg_gpu_synchronize(void);

// ============================================================================
// GPU Kernel Execution
// ============================================================================

/**
 * @brief Launch a GPU kernel
 * @param name Kernel name
 * @param args Kernel arguments
 * @param args_size Size of arguments
 * @param grid Grid dimensions
 * @param block Block dimensions
 * @param shared_memory Shared memory size
 * @param stream_id Stream ID
 * @return QG_GPU_SUCCESS on success
 */
int qg_gpu_launch_kernel(const char* name, const void* args, size_t args_size,
                         dim3 grid, dim3 block, size_t shared_memory, int stream_id);

// ============================================================================
// GPU Memory Pool Functions
// ============================================================================

/**
 * @brief Allocate from GPU memory pool
 * @param pool Memory pool
 * @param size Allocation size
 * @return Allocated pointer or NULL
 */
void* gpu_alloc_from_pool(gpu_memory_pool_t* pool, size_t size);

/**
 * @brief Free to GPU memory pool
 * @param pool Memory pool
 * @param ptr Pointer to free
 */
void gpu_free_to_pool(gpu_memory_pool_t* pool, void* ptr);

// ============================================================================
// Metal Backend Functions (conditionally available)
// ============================================================================

#ifdef QGT_ENABLE_METAL
void* qgt_metal_allocate(size_t size);
void* qgt_metal_allocate_pinned(size_t size);
void* qgt_metal_allocate_pool(size_t size);
void qgt_metal_free(void* ptr);
void qgt_metal_free_pinned(void* ptr);
void qgt_metal_free_pool(void* ptr);
void qgt_metal_memcpy(void* dst, const void* src, size_t size);
int qgt_metal_upload(const void* src, void* dst, size_t size);
int qgt_metal_download(const void* src, void* dst, size_t size);
int qgt_metal_set_device(int device_id);
void qgt_metal_cleanup(void);

// Metal command queue management
void* qgt_metal_create_command_queue(void);
int qgt_metal_register_command_queue(void* queue);
void* qgt_metal_get_command_queue(int stream_id);
void qgt_metal_unregister_command_queue(int stream_id);
void qgt_metal_launch_kernel(const char* name, const void* args, size_t args_size,
                            dim3 grid, dim3 block, size_t shared_memory, int stream_id);
#endif

// ============================================================================
// CUDA Backend Functions (conditionally available)
// ============================================================================

#ifdef QGT_ENABLE_CUDA
int qgt_cuda_register_stream(void* stream);
void* qgt_cuda_get_stream(int stream_id);
void qgt_cuda_unregister_stream(int stream_id);
void qgt_cuda_launch_kernel(const char* name, const void* args, size_t args_size,
                           dim3 grid, dim3 block, size_t shared_memory, int stream_id);
#endif

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_GEOMETRIC_GPU_H
