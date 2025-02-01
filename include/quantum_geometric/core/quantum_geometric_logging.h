#ifndef QUANTUM_GEOMETRIC_LOGGING_H
#define QUANTUM_GEOMETRIC_LOGGING_H

#include "quantum_geometric/core/quantum_geometric_types.h"
#include <stdio.h>

// Maximum length of log messages
#define QGT_MAX_LOG_MESSAGE_LENGTH 4096

// Logging levels
typedef enum {
    LOG_LEVEL_ERROR = 0,
    LOG_LEVEL_WARNING = 1,
    LOG_LEVEL_INFO = 2,
    LOG_LEVEL_DEBUG = 3,
    LOG_LEVEL_TRACE = 4
} log_level_t;

// Logging flags
typedef enum {
    LOG_FLAG_NONE = 0,
    LOG_FLAG_TIMESTAMP = 1 << 0,
    LOG_FLAG_LEVEL = 1 << 1
} log_flags_t;

// Logging context structure
typedef struct {
    FILE* file;
    log_level_t level;
    log_flags_t flags;
} logging_context_t;

// Initialize logging system
qgt_error_t geometric_init_logging(const char* log_file);

// Cleanup logging system
void geometric_cleanup_logging(void);

// Set logging level
void geometric_set_log_level(log_level_t level);

// Set logging flags
void geometric_set_log_flags(log_flags_t flags);

// Log messages at different levels
void geometric_log_error(const char* format, ...);
void geometric_log_warning(const char* format, ...);
void geometric_log_info(const char* format, ...);
void geometric_log_debug(const char* format, ...);
void geometric_log_trace(const char* format, ...);

// Convenience macros for logging
#define QGT_LOG_ERROR(...) \
    geometric_log_error(__VA_ARGS__)

#define QGT_LOG_WARNING(...) \
    geometric_log_warning(__VA_ARGS__)

#define QGT_LOG_INFO(...) \
    geometric_log_info(__VA_ARGS__)

#define QGT_LOG_DEBUG(...) \
    geometric_log_debug(__VA_ARGS__)

#define QGT_LOG_TRACE(...) \
    geometric_log_trace(__VA_ARGS__)

// Function entry/exit logging macros
#define QGT_FUNCTION_ENTRY() \
    QGT_LOG_TRACE("Entering %s", __func__)

#define QGT_FUNCTION_EXIT() \
    QGT_LOG_TRACE("Exiting %s", __func__)

#endif // QUANTUM_GEOMETRIC_LOGGING_H
