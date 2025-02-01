#include "quantum_geometric/core/quantum_geometric_logging.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include <stdarg.h>
#include <time.h>
#include <string.h>
#include <pthread.h>
#include <stdatomic.h>

// Global logging context with atomic flags
static struct {
    FILE* file;
    atomic_int level;
    atomic_int flags;
} g_logging_context = {
    .file = NULL,
    .level = LOG_LEVEL_INFO,
    .flags = LOG_FLAG_TIMESTAMP | LOG_FLAG_LEVEL
};

// Mutex for thread safety
static pthread_mutex_t g_logging_mutex = PTHREAD_MUTEX_INITIALIZER;

// Level strings
static const char* level_strings[] = {
    "ERROR",
    "WARNING",
    "INFO",
    "DEBUG",
    "TRACE"
};

// Initialize logging system
qgt_error_t geometric_init_logging(const char* log_file) {
    pthread_mutex_lock(&g_logging_mutex);
    
    FILE* old_file = g_logging_context.file;
    if (old_file && old_file != stderr) {
        fclose(old_file);
    }
    
    if (log_file) {
        g_logging_context.file = fopen(log_file, "a");
        if (!g_logging_context.file) {
            pthread_mutex_unlock(&g_logging_mutex);
            return QGT_ERROR_SYSTEM_FAILURE;
        }
    } else {
        g_logging_context.file = stderr;
    }
    
    pthread_mutex_unlock(&g_logging_mutex);
    return QGT_SUCCESS;
}

// Cleanup logging system
void geometric_cleanup_logging(void) {
    pthread_mutex_lock(&g_logging_mutex);
    
    if (g_logging_context.file && g_logging_context.file != stderr) {
        fclose(g_logging_context.file);
        g_logging_context.file = NULL;
    }
    
    pthread_mutex_unlock(&g_logging_mutex);
}

// Set logging level
void geometric_set_log_level(log_level_t level) {
    pthread_mutex_lock(&g_logging_mutex);
    atomic_store_explicit(&g_logging_context.level, level, memory_order_release);
    pthread_mutex_unlock(&g_logging_mutex);
}

// Set logging flags
void geometric_set_log_flags(log_flags_t flags) {
    pthread_mutex_lock(&g_logging_mutex);
    atomic_store_explicit(&g_logging_context.flags, flags, memory_order_release);
    pthread_mutex_unlock(&g_logging_mutex);
}

// Internal logging function
static void log_message(log_level_t level, const char* format, va_list args) {
    log_level_t current_level = atomic_load_explicit(&g_logging_context.level, memory_order_acquire);
    if (!g_logging_context.file || level > current_level) {
        return;
    }
    
    pthread_mutex_lock(&g_logging_mutex);
    
    char message[QGT_MAX_LOG_MESSAGE_LENGTH];
    char final_message[QGT_MAX_LOG_MESSAGE_LENGTH];
    size_t offset = 0;
    
    // Add timestamp if enabled
    log_flags_t current_flags = atomic_load_explicit(&g_logging_context.flags, memory_order_acquire);
    if (current_flags & LOG_FLAG_TIMESTAMP) {
        time_t now;
        struct tm* timeinfo;
        char timestamp[32];
        
        time(&now);
        timeinfo = localtime(&now);
        strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", timeinfo);
        
        offset += snprintf(final_message + offset,
                         sizeof(final_message) - offset,
                         "[%s] ", timestamp);
    }
    
    // Add log level if enabled
    if (current_flags & LOG_FLAG_LEVEL) {
        offset += snprintf(final_message + offset,
                         sizeof(final_message) - offset,
                         "[%s] ", level_strings[level]);
    }
    
    // Format message
    vsnprintf(message, sizeof(message), format, args);
    
    // Add message to final output
    snprintf(final_message + offset,
             sizeof(final_message) - offset,
             "%s\n", message);
    
    // Write to log file
    fputs(final_message, g_logging_context.file);
    fflush(g_logging_context.file);
    
    pthread_mutex_unlock(&g_logging_mutex);
}

// Log error messages
void geometric_log_error(const char* format, ...) {
    va_list args;
    va_start(args, format);
    log_message(LOG_LEVEL_ERROR, format, args);
    va_end(args);
}

// Log warning messages
void geometric_log_warning(const char* format, ...) {
    va_list args;
    va_start(args, format);
    log_message(LOG_LEVEL_WARNING, format, args);
    va_end(args);
}

// Log info messages
void geometric_log_info(const char* format, ...) {
    va_list args;
    va_start(args, format);
    log_message(LOG_LEVEL_INFO, format, args);
    va_end(args);
}

// Log debug messages
void geometric_log_debug(const char* format, ...) {
    va_list args;
    va_start(args, format);
    log_message(LOG_LEVEL_DEBUG, format, args);
    va_end(args);
}

// Log trace messages
void geometric_log_trace(const char* format, ...) {
    va_list args;
    va_start(args, format);
    log_message(LOG_LEVEL_TRACE, format, args);
    va_end(args);
}
