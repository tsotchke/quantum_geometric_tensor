#ifndef QUANTUM_GEOMETRIC_CONFIG_H
#define QUANTUM_GEOMETRIC_CONFIG_H

#include "quantum_geometric/core/quantum_geometric_types.h"
#include <stdbool.h>
#include <stddef.h>

// Configuration keys
typedef enum {
    CONFIG_MAX_THREADS,
    CONFIG_MEMORY_LIMIT,
    CONFIG_GPU_ENABLED,
    CONFIG_OPTIMIZATION_LEVEL
} config_key_t;

// Optimization levels
typedef enum {
    OPTIMIZATION_LEVEL_NONE = 0,
    OPTIMIZATION_LEVEL_NORMAL = 1,
    OPTIMIZATION_LEVEL_AGGRESSIVE = 2
} optimization_level_t;

// Configuration context structure
typedef struct {
    int max_threads;
    size_t memory_limit;
    bool gpu_enabled;
    optimization_level_t optimization_level;
} config_context_t;

// Initialize configuration system
qgt_error_t geometric_init_config(const char* config_file);

// Cleanup configuration system
void geometric_cleanup_config(void);

// Get configuration value
qgt_error_t geometric_get_config(config_key_t key, void* value);

// Set configuration value
qgt_error_t geometric_set_config(config_key_t key, const void* value);

// Save configuration to file
qgt_error_t geometric_save_config(const char* config_file);

// Convenience macros for configuration
#define QGT_CONFIG_GET_INT(key, value) \
    geometric_get_config(key, &value)

#define QGT_CONFIG_GET_SIZE(key, value) \
    geometric_get_config(key, &value)

#define QGT_CONFIG_GET_BOOL(key, value) \
    geometric_get_config(key, &value)

#define QGT_CONFIG_GET_ENUM(key, value) \
    geometric_get_config(key, &value)

#define QGT_CONFIG_SET_INT(key, value) \
    do { \
        int _value = value; \
        geometric_set_config(key, &_value); \
    } while (0)

#define QGT_CONFIG_SET_SIZE(key, value) \
    do { \
        size_t _value = value; \
        geometric_set_config(key, &_value); \
    } while (0)

#define QGT_CONFIG_SET_BOOL(key, value) \
    do { \
        bool _value = value; \
        geometric_set_config(key, &_value); \
    } while (0)

#define QGT_CONFIG_SET_ENUM(key, value) \
    do { \
        optimization_level_t _value = value; \
        geometric_set_config(key, &_value); \
    } while (0)

#endif // QUANTUM_GEOMETRIC_CONFIG_H
