#include "quantum_geometric/core/quantum_geometric_error.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_geometric_constants.h"
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

// Thread-local error context
static __thread error_context_t current_error = {
    .code = QGT_SUCCESS,
    .message = {0},
    .file = NULL,
    .line = 0,
    .function = NULL
};

// Set error details
void geometric_set_error(qgt_error_t code,
                        const char* file,
                        int line,
                        const char* function,
                        const char* format,
                        ...) {
    current_error.code = code;
    current_error.file = file;
    current_error.line = line;
    current_error.function = function;
    
    va_list args;
    va_start(args, format);
    vsnprintf(current_error.message,
             QGT_MAX_ERROR_MESSAGE_LENGTH,
             format,
             args);
    va_end(args);
}

// Get current error context
const error_context_t* geometric_get_error(void) {
    return &current_error;
}

// Clear current error
void geometric_clear_error(void) {
    current_error.code = QGT_SUCCESS;
    current_error.message[0] = '\0';
    current_error.file = NULL;
    current_error.line = 0;
    current_error.function = NULL;
}

// Format error message
void geometric_format_error(char* buffer,
                          size_t size,
                          const error_context_t* error) {
    if (!buffer || !error || size == 0) {
        return;
    }
    
    const char* error_string = "Unknown error";
    switch (error->code) {
        case QGT_SUCCESS:
            error_string = "Success";
            break;
        case QGT_ERROR_MEMORY_ALLOCATION:
            error_string = "Memory allocation failed";
            break;
        case QGT_ERROR_INVALID_PARAMETER:
            error_string = "Invalid parameter";
            break;
        case QGT_ERROR_NOT_INITIALIZED:
            error_string = "Not initialized";
            break;
        case QGT_ERROR_ALREADY_INITIALIZED:
            error_string = "Already initialized";
            break;
        case QGT_ERROR_NO_DEVICE:
            error_string = "No device available";
            break;
        case QGT_ERROR_NOT_IMPLEMENTED:
            error_string = "Not implemented";
            break;
        case QGT_ERROR_VALIDATION_FAILED:
            error_string = "Validation failed";
            break;
    }
    
    if (error->file && error->function) {
        snprintf(buffer, size,
                "%s: %s [%s:%d in %s()]",
                error_string,
                error->message,
                error->file,
                error->line,
                error->function);
    } else {
        snprintf(buffer, size,
                "%s: %s",
                error_string,
                error->message);
    }
}

// Check error condition
bool geometric_has_error(void) {
    return current_error.code != QGT_SUCCESS;
}

// Get error code
qgt_error_t geometric_get_error_code(void) {
    return current_error.code;
}

// Get error message
const char* geometric_get_error_message(void) {
    return current_error.message;
}

// Push error handler
static error_handler_t error_handler = NULL;

void geometric_set_error_handler(error_handler_t handler) {
    error_handler = handler;
}

// Handle error
void geometric_handle_error(const error_context_t* error) {
    if (error_handler) {
        error_handler(error);
    }
}
