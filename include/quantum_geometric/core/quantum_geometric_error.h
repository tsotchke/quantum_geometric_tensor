#ifndef QUANTUM_GEOMETRIC_ERROR_H
#define QUANTUM_GEOMETRIC_ERROR_H

#include "quantum_geometric/core/quantum_geometric_types.h"
#include <stdbool.h>

// Error context structure
typedef struct {
    qgt_error_t code;
    char message[QGT_MAX_ERROR_MESSAGE_LENGTH];
    const char* file;
    int line;
    const char* function;
} error_context_t;

// Error handler function type
typedef void (*error_handler_t)(const error_context_t* error);

// Set error details
void geometric_set_error(qgt_error_t code,
                        const char* file,
                        int line,
                        const char* function,
                        const char* format,
                        ...);

// Get current error context
const error_context_t* geometric_get_error(void);

// Clear current error
void geometric_clear_error(void);

// Format error message
void geometric_format_error(char* buffer,
                          size_t size,
                          const error_context_t* error);

// Check error condition
bool geometric_has_error(void);

// Get error code
qgt_error_t geometric_get_error_code(void);

// Get error message
const char* geometric_get_error_message(void);

// Set error handler
void geometric_set_error_handler(error_handler_t handler);

// Handle error
void geometric_handle_error(const error_context_t* error);

// Convenience macros for error handling
#define QGT_SET_ERROR(code, ...) \
    geometric_set_error(code, __FILE__, __LINE__, __func__, __VA_ARGS__)

// Error checking macros are defined in quantum_geometric_types.h

#define QGT_CHECK_INITIALIZATION(obj) \
    do { \
        if (!(obj)->is_initialized) { \
            QGT_SET_ERROR(QGT_ERROR_NOT_INITIALIZED, "Object not initialized"); \
            return QGT_ERROR_NOT_INITIALIZED; \
        } \
    } while (0)

#define QGT_CHECK_ERROR(err) \
    do { \
        qgt_error_t _err = (err); \
        if (_err != QGT_SUCCESS) { \
            return _err; \
        } \
    } while (0)

#endif // QUANTUM_GEOMETRIC_ERROR_H
