/**
 * @file quantum_geometric_tensor_error.h
 * @brief Error Tracking and Reporting for GPU Tensor Operations
 *
 * Provides comprehensive error handling including:
 * - GPU/CUDA error translation
 * - Error code definitions
 * - Error callback registration
 * - Error stack traces
 * - Recovery suggestions
 * - Error logging
 *
 * Part of the QGTL Hardware Acceleration Framework.
 */

#ifndef QUANTUM_GEOMETRIC_TENSOR_ERROR_H
#define QUANTUM_GEOMETRIC_TENSOR_ERROR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================

#define TENSOR_ERROR_MAX_MESSAGE 512
#define TENSOR_ERROR_MAX_STACK 32
#define TENSOR_ERROR_MAX_CALLBACKS 16
#define TENSOR_ERROR_LOG_SIZE 1024

// ============================================================================
// Error Codes
// ============================================================================

/**
 * Error codes for tensor operations
 */
typedef enum {
    // Success
    TENSOR_ERROR_SUCCESS = 0,

    // General errors (1-99)
    TENSOR_ERROR_UNKNOWN = 1,
    TENSOR_ERROR_INVALID_ARGUMENT = 2,
    TENSOR_ERROR_NULL_POINTER = 3,
    TENSOR_ERROR_NOT_INITIALIZED = 4,
    TENSOR_ERROR_ALREADY_INITIALIZED = 5,
    TENSOR_ERROR_NOT_SUPPORTED = 6,
    TENSOR_ERROR_NOT_IMPLEMENTED = 7,
    TENSOR_ERROR_INTERNAL = 8,

    // Memory errors (100-199)
    TENSOR_ERROR_OUT_OF_MEMORY = 100,
    TENSOR_ERROR_HOST_MEMORY = 101,
    TENSOR_ERROR_DEVICE_MEMORY = 102,
    TENSOR_ERROR_ALLOCATION_FAILED = 103,
    TENSOR_ERROR_DEALLOCATION_FAILED = 104,
    TENSOR_ERROR_MEMORY_POOL_EXHAUSTED = 105,
    TENSOR_ERROR_INVALID_POINTER = 106,
    TENSOR_ERROR_MEMORY_LEAK = 107,

    // Dimension/shape errors (200-299)
    TENSOR_ERROR_DIMENSION_MISMATCH = 200,
    TENSOR_ERROR_SHAPE_MISMATCH = 201,
    TENSOR_ERROR_INDEX_OUT_OF_BOUNDS = 202,
    TENSOR_ERROR_INVALID_SHAPE = 203,
    TENSOR_ERROR_TOO_MANY_DIMENSIONS = 204,
    TENSOR_ERROR_INVALID_STRIDE = 205,
    TENSOR_ERROR_NOT_CONTIGUOUS = 206,

    // Data type errors (300-399)
    TENSOR_ERROR_TYPE_MISMATCH = 300,
    TENSOR_ERROR_INVALID_TYPE = 301,
    TENSOR_ERROR_TYPE_CONVERSION = 302,
    TENSOR_ERROR_PRECISION_LOSS = 303,

    // GPU/CUDA errors (400-499)
    TENSOR_ERROR_GPU_NOT_AVAILABLE = 400,
    TENSOR_ERROR_GPU_DEVICE_ERROR = 401,
    TENSOR_ERROR_GPU_KERNEL_FAILED = 402,
    TENSOR_ERROR_GPU_LAUNCH_FAILED = 403,
    TENSOR_ERROR_GPU_SYNC_FAILED = 404,
    TENSOR_ERROR_GPU_INVALID_DEVICE = 405,
    TENSOR_ERROR_GPU_STREAM_ERROR = 406,
    TENSOR_ERROR_GPU_DRIVER = 407,
    TENSOR_ERROR_GPU_CONTEXT = 408,
    TENSOR_ERROR_CUBLAS = 409,
    TENSOR_ERROR_CUSOLVER = 410,
    TENSOR_ERROR_CUFFT = 411,
    TENSOR_ERROR_CUDNN = 412,

    // Numerical errors (500-599)
    TENSOR_ERROR_NUMERICAL_INSTABILITY = 500,
    TENSOR_ERROR_SINGULAR_MATRIX = 501,
    TENSOR_ERROR_ILL_CONDITIONED = 502,
    TENSOR_ERROR_NAN_DETECTED = 503,
    TENSOR_ERROR_INF_DETECTED = 504,
    TENSOR_ERROR_UNDERFLOW = 505,
    TENSOR_ERROR_OVERFLOW = 506,
    TENSOR_ERROR_CONVERGENCE_FAILED = 507,

    // Operation errors (600-699)
    TENSOR_ERROR_INVALID_OPERATION = 600,
    TENSOR_ERROR_OPERATION_FAILED = 601,
    TENSOR_ERROR_CONTRACTION_FAILED = 602,
    TENSOR_ERROR_DECOMPOSITION_FAILED = 603,
    TENSOR_ERROR_ASYNC_PENDING = 604,
    TENSOR_ERROR_TIMEOUT = 605,
    TENSOR_ERROR_CANCELLED = 606,

    // I/O errors (700-799)
    TENSOR_ERROR_FILE_NOT_FOUND = 700,
    TENSOR_ERROR_FILE_READ = 701,
    TENSOR_ERROR_FILE_WRITE = 702,
    TENSOR_ERROR_INVALID_FORMAT = 703,
    TENSOR_ERROR_SERIALIZATION = 704

} tensor_error_code_t;

/**
 * Error severity levels
 */
typedef enum {
    TENSOR_ERROR_SEVERITY_INFO,       // Informational
    TENSOR_ERROR_SEVERITY_WARNING,    // Warning, operation continued
    TENSOR_ERROR_SEVERITY_ERROR,      // Error, operation failed
    TENSOR_ERROR_SEVERITY_FATAL       // Fatal, unrecoverable
} tensor_error_severity_t;

/**
 * Error categories
 */
typedef enum {
    TENSOR_ERROR_CAT_GENERAL,
    TENSOR_ERROR_CAT_MEMORY,
    TENSOR_ERROR_CAT_DIMENSION,
    TENSOR_ERROR_CAT_TYPE,
    TENSOR_ERROR_CAT_GPU,
    TENSOR_ERROR_CAT_NUMERICAL,
    TENSOR_ERROR_CAT_OPERATION,
    TENSOR_ERROR_CAT_IO
} tensor_error_category_t;

// ============================================================================
// Data Structures
// ============================================================================

/**
 * Error context information
 */
typedef struct {
    tensor_error_code_t code;
    tensor_error_severity_t severity;
    tensor_error_category_t category;
    char message[TENSOR_ERROR_MAX_MESSAGE];
    const char* file;
    int line;
    const char* function;
    uint64_t timestamp_ns;
    int gpu_device;                   // GPU device if applicable
    int cuda_error;                   // Original CUDA error code
    char cuda_message[256];           // Original CUDA error message
} tensor_error_info_t;

/**
 * Error stack entry
 */
typedef struct {
    const char* file;
    int line;
    const char* function;
    char context[256];
} tensor_error_stack_entry_t;

/**
 * Error stack for tracing
 */
typedef struct {
    tensor_error_stack_entry_t entries[TENSOR_ERROR_MAX_STACK];
    size_t depth;
} tensor_error_stack_t;

/**
 * Error callback function type
 */
typedef void (*tensor_error_callback_t)(
    const tensor_error_info_t* error,
    void* user_data);

/**
 * Error statistics
 */
typedef struct {
    uint64_t total_errors;
    uint64_t errors_by_severity[4];   // Indexed by severity
    uint64_t errors_by_category[8];   // Indexed by category
    uint64_t most_common_code;
    uint64_t most_common_count;
    uint64_t last_error_timestamp_ns;
} tensor_error_stats_t;

/**
 * Error log entry
 */
typedef struct {
    tensor_error_info_t error;
    tensor_error_stack_t stack;
    bool was_handled;
} tensor_error_log_entry_t;

/**
 * Opaque error handler
 */
typedef struct tensor_error_handler tensor_error_handler_t;

// ============================================================================
// Initialization
// ============================================================================

/**
 * Create error handler
 */
tensor_error_handler_t* tensor_error_handler_create(void);

/**
 * Destroy error handler
 */
void tensor_error_handler_destroy(tensor_error_handler_t* handler);

/**
 * Get global error handler
 */
tensor_error_handler_t* tensor_error_get_global_handler(void);

/**
 * Set global error handler
 */
void tensor_error_set_global_handler(tensor_error_handler_t* handler);

// ============================================================================
// Error Reporting
// ============================================================================

/**
 * Set last error
 */
void tensor_error_set(
    tensor_error_handler_t* handler,
    tensor_error_code_t code,
    const char* message,
    const char* file,
    int line,
    const char* function);

/**
 * Set last error with formatting
 */
void tensor_error_setf(
    tensor_error_handler_t* handler,
    tensor_error_code_t code,
    const char* file,
    int line,
    const char* function,
    const char* format,
    ...);

/**
 * Get last error
 */
tensor_error_code_t tensor_error_get_last(
    tensor_error_handler_t* handler,
    tensor_error_info_t* info);

/**
 * Clear last error
 */
void tensor_error_clear(tensor_error_handler_t* handler);

/**
 * Check if error occurred
 */
bool tensor_error_occurred(tensor_error_handler_t* handler);

// Convenience macros
#define TENSOR_SET_ERROR(handler, code, msg) \
    tensor_error_set(handler, code, msg, __FILE__, __LINE__, __func__)

#define TENSOR_SET_ERRORF(handler, code, ...) \
    tensor_error_setf(handler, code, __FILE__, __LINE__, __func__, __VA_ARGS__)

#define TENSOR_CHECK(handler, condition, code, msg) \
    do { \
        if (!(condition)) { \
            TENSOR_SET_ERROR(handler, code, msg); \
            return false; \
        } \
    } while(0)

#define TENSOR_CHECK_NULL(handler, ptr) \
    TENSOR_CHECK(handler, (ptr) != NULL, TENSOR_ERROR_NULL_POINTER, "Null pointer: " #ptr)

// ============================================================================
// Error Stack
// ============================================================================

/**
 * Push error context onto stack
 */
void tensor_error_push_context(
    tensor_error_handler_t* handler,
    const char* file,
    int line,
    const char* function,
    const char* context);

/**
 * Pop error context from stack
 */
void tensor_error_pop_context(tensor_error_handler_t* handler);

/**
 * Get current error stack
 */
bool tensor_error_get_stack(
    tensor_error_handler_t* handler,
    tensor_error_stack_t* stack);

/**
 * Clear error stack
 */
void tensor_error_clear_stack(tensor_error_handler_t* handler);

// Convenience macros for scoped context
#define TENSOR_ERROR_CONTEXT(handler, ctx) \
    tensor_error_push_context(handler, __FILE__, __LINE__, __func__, ctx)

#define TENSOR_ERROR_POP(handler) \
    tensor_error_pop_context(handler)

// ============================================================================
// Callbacks
// ============================================================================

/**
 * Register error callback
 */
bool tensor_error_register_callback(
    tensor_error_handler_t* handler,
    tensor_error_callback_t callback,
    void* user_data,
    tensor_error_severity_t min_severity);

/**
 * Unregister error callback
 */
bool tensor_error_unregister_callback(
    tensor_error_handler_t* handler,
    tensor_error_callback_t callback);

/**
 * Clear all callbacks
 */
void tensor_error_clear_callbacks(tensor_error_handler_t* handler);

// ============================================================================
// GPU Error Integration
// ============================================================================

/**
 * Translate CUDA error to tensor error
 */
tensor_error_code_t tensor_error_from_cuda(int cuda_error);

/**
 * Translate cuBLAS error to tensor error
 */
tensor_error_code_t tensor_error_from_cublas(int cublas_status);

/**
 * Translate cuSOLVER error to tensor error
 */
tensor_error_code_t tensor_error_from_cusolver(int cusolver_status);

/**
 * Check CUDA error and set if failed
 */
bool tensor_error_check_cuda(
    tensor_error_handler_t* handler,
    int cuda_error,
    const char* file,
    int line,
    const char* function);

#define TENSOR_CHECK_CUDA(handler, cuda_call) \
    tensor_error_check_cuda(handler, cuda_call, __FILE__, __LINE__, __func__)

// ============================================================================
// Error Logging
// ============================================================================

/**
 * Enable error logging
 */
void tensor_error_enable_logging(
    tensor_error_handler_t* handler,
    bool enable);

/**
 * Get error log
 */
bool tensor_error_get_log(
    tensor_error_handler_t* handler,
    tensor_error_log_entry_t** entries,
    size_t* count);

/**
 * Clear error log
 */
void tensor_error_clear_log(tensor_error_handler_t* handler);

/**
 * Export error log to file
 */
bool tensor_error_export_log(
    tensor_error_handler_t* handler,
    const char* filename);

// ============================================================================
// Statistics
// ============================================================================

/**
 * Get error statistics
 */
bool tensor_error_get_stats(
    tensor_error_handler_t* handler,
    tensor_error_stats_t* stats);

/**
 * Reset error statistics
 */
void tensor_error_reset_stats(tensor_error_handler_t* handler);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Get error code name
 */
const char* tensor_error_code_name(tensor_error_code_t code);

/**
 * Get error code description
 */
const char* tensor_error_code_description(tensor_error_code_t code);

/**
 * Get severity name
 */
const char* tensor_error_severity_name(tensor_error_severity_t severity);

/**
 * Get category name
 */
const char* tensor_error_category_name(tensor_error_category_t category);

/**
 * Get category for error code
 */
tensor_error_category_t tensor_error_get_category(tensor_error_code_t code);

/**
 * Get default severity for error code
 */
tensor_error_severity_t tensor_error_get_severity(tensor_error_code_t code);

/**
 * Get recovery suggestion for error
 */
const char* tensor_error_get_suggestion(tensor_error_code_t code);

/**
 * Check if error is recoverable
 */
bool tensor_error_is_recoverable(tensor_error_code_t code);

/**
 * Format error info as string
 */
char* tensor_error_format(const tensor_error_info_t* info);

/**
 * Format error stack as string
 */
char* tensor_error_format_stack(const tensor_error_stack_t* stack);

/**
 * Free formatted string
 */
void tensor_error_free_string(char* str);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_GEOMETRIC_TENSOR_ERROR_H
