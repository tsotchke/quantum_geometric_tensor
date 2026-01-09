#ifndef METAL_COMMON_H
#define METAL_COMMON_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ===========================================================================
// Metal Common Error Types
// ===========================================================================

typedef int32_t metal_error_t;

#define METAL_SUCCESS                   0
#define METAL_ERROR_INVALID_PARAMS      -1
#define METAL_ERROR_DEVICE_NOT_FOUND    -2
#define METAL_ERROR_OUT_OF_MEMORY       -3
#define METAL_ERROR_SHADER_NOT_FOUND    -4
#define METAL_ERROR_PIPELINE_FAILED     -5
#define METAL_ERROR_COMMAND_FAILED      -6
#define METAL_ERROR_NOT_INITIALIZED     -7
#define METAL_ERROR_BUFFER_FAILED       -8
#define METAL_ERROR_INTERNAL            -9

// ===========================================================================
// Metal Common Pipeline Management
// ===========================================================================

/**
 * Initialize the common Metal backend
 * Must be called before any other metal_* functions
 * @return METAL_SUCCESS on success, error code on failure
 */
metal_error_t metal_common_initialize(void);

/**
 * Cleanup the common Metal backend
 * Call when done with Metal operations
 */
void metal_common_cleanup(void);

/**
 * Check if Metal is available on this system
 * @return true if Metal is available
 */
bool metal_common_is_available(void);

/**
 * Create a compute pipeline for the given kernel
 * @param kernel_name Name of the kernel function
 * @param pipeline Output pipeline handle
 * @return METAL_SUCCESS on success, error code on failure
 */
metal_error_t metal_create_compute_pipeline(const char* kernel_name, void** pipeline);

/**
 * Destroy a compute pipeline
 * @param pipeline Pipeline handle to destroy
 */
void metal_destroy_compute_pipeline(void* pipeline);

/**
 * Execute a compute command
 * @param pipeline Pipeline handle
 * @param buffers Array of buffer pointers (already Metal buffers)
 * @param num_buffers Number of buffers
 * @param params Pointer to parameter data
 * @param params_size Size of parameter data in bytes
 * @param thread_groups_x Number of thread groups in X dimension
 * @param thread_groups_y Number of thread groups in Y dimension
 * @param thread_groups_z Number of thread groups in Z dimension
 * @return METAL_SUCCESS on success, error code on failure
 */
metal_error_t metal_execute_command(
    void* pipeline,
    void** buffers,
    uint32_t num_buffers,
    const void* params,
    size_t params_size,
    uint32_t thread_groups_x,
    uint32_t thread_groups_y,
    uint32_t thread_groups_z
);

// ===========================================================================
// Metal Common Buffer Management
// ===========================================================================

/**
 * Create a Metal buffer
 * @param size Size in bytes
 * @param buffer Output buffer handle
 * @return METAL_SUCCESS on success, error code on failure
 */
metal_error_t metal_create_buffer(size_t size, void** buffer);

/**
 * Create a Metal buffer with initial data
 * @param data Source data to copy
 * @param size Size in bytes
 * @param buffer Output buffer handle
 * @return METAL_SUCCESS on success, error code on failure
 */
metal_error_t metal_create_buffer_with_data(const void* data, size_t size, void** buffer);

/**
 * Copy data from host to Metal buffer
 * @param buffer Metal buffer handle
 * @param data Source data
 * @param size Size in bytes
 * @return METAL_SUCCESS on success, error code on failure
 */
metal_error_t metal_copy_to_buffer(void* buffer, const void* data, size_t size);

/**
 * Copy data from Metal buffer to host
 * @param data Destination buffer
 * @param buffer Metal buffer handle
 * @param size Size in bytes
 * @return METAL_SUCCESS on success, error code on failure
 */
metal_error_t metal_copy_from_buffer(void* data, void* buffer, size_t size);

/**
 * Destroy a Metal buffer
 * @param buffer Buffer handle to destroy
 */
void metal_destroy_buffer(void* buffer);

/**
 * Get raw pointer to buffer contents (for shared/managed buffers)
 * @param buffer Metal buffer handle
 * @return Pointer to buffer contents, or NULL on failure
 */
void* metal_get_buffer_contents(void* buffer);

// ===========================================================================
// Metal Common Shader Registration
// ===========================================================================

/**
 * Register Metal shader source code
 * This allows dynamic shader compilation for kernels not in the default library
 * @param shader_source Metal shading language source code
 * @return METAL_SUCCESS on success, error code on failure
 */
metal_error_t metal_register_shader_source(const char* shader_source);

/**
 * Register a precompiled Metal library from a file path
 * @param library_path Path to .metallib file
 * @return METAL_SUCCESS on success, error code on failure
 */
metal_error_t metal_register_library_path(const char* library_path);

#ifdef __cplusplus
}
#endif

#endif // METAL_COMMON_H
