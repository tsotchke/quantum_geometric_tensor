#include "quantum_geometric/core/quantum_geometric_config.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_geometric_constants.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Global configuration context
static config_context_t* global_config_context = NULL;

// Initialize configuration system
qgt_error_t geometric_init_config(const char* config_file) {
    if (global_config_context) {
        return QGT_ERROR_ALREADY_INITIALIZED;
    }
    
    global_config_context = (config_context_t*)calloc(1, sizeof(config_context_t));
    if (!global_config_context) {
        return QGT_ERROR_ALLOCATION_FAILED;
    }
    
    // Set default values
    global_config_context->max_threads = 4;
    global_config_context->memory_limit = 1024 * 1024 * 1024; // 1GB
    global_config_context->gpu_enabled = false;
    global_config_context->optimization_level = OPTIMIZATION_LEVEL_NORMAL;
    
    // Load configuration file if provided
    if (config_file) {
        FILE* file = fopen(config_file, "r");
        if (!file) {
            free(global_config_context);
            global_config_context = NULL;
            return QGT_ERROR_IO_ERROR;
        }
        
        char line[256];
        while (fgets(line, sizeof(line), file)) {
            char key[128], value[128];
            if (sscanf(line, "%127[^=]=%127s", key, value) == 2) {
                if (strcmp(key, "max_threads") == 0) {
                    global_config_context->max_threads = atoi(value);
                } else if (strcmp(key, "memory_limit") == 0) {
                    global_config_context->memory_limit = strtoull(value, NULL, 10);
                } else if (strcmp(key, "gpu_enabled") == 0) {
                    global_config_context->gpu_enabled = strcmp(value, "true") == 0;
                } else if (strcmp(key, "optimization_level") == 0) {
                    if (strcmp(value, "none") == 0) {
                        global_config_context->optimization_level = OPTIMIZATION_LEVEL_NONE;
                    } else if (strcmp(value, "aggressive") == 0) {
                        global_config_context->optimization_level = OPTIMIZATION_LEVEL_AGGRESSIVE;
                    }
                }
            }
        }
        
        fclose(file);
    }
    
    return QGT_SUCCESS;
}

// Cleanup configuration system
void geometric_cleanup_config(void) {
    if (global_config_context) {
        free(global_config_context);
        global_config_context = NULL;
    }
}

// Get configuration value
qgt_error_t geometric_get_config(config_key_t key, void* value) {
    if (!global_config_context || !value) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    switch (key) {
        case CONFIG_MAX_THREADS:
            *(int*)value = global_config_context->max_threads;
            break;
            
        case CONFIG_MEMORY_LIMIT:
            *(size_t*)value = global_config_context->memory_limit;
            break;
            
        case CONFIG_GPU_ENABLED:
            *(bool*)value = global_config_context->gpu_enabled;
            break;
            
        case CONFIG_OPTIMIZATION_LEVEL:
            *(optimization_level_t*)value = global_config_context->optimization_level;
            break;
            
        default:
            return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    return QGT_SUCCESS;
}

// Set configuration value
qgt_error_t geometric_set_config(config_key_t key, const void* value) {
    if (!global_config_context || !value) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    switch (key) {
        case CONFIG_MAX_THREADS:
            global_config_context->max_threads = *(const int*)value;
            break;
            
        case CONFIG_MEMORY_LIMIT:
            global_config_context->memory_limit = *(const size_t*)value;
            break;
            
        case CONFIG_GPU_ENABLED:
            global_config_context->gpu_enabled = *(const bool*)value;
            break;
            
        case CONFIG_OPTIMIZATION_LEVEL:
            global_config_context->optimization_level = *(const optimization_level_t*)value;
            break;
            
        default:
            return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    return QGT_SUCCESS;
}

// Save configuration to file
qgt_error_t geometric_save_config(const char* config_file) {
    if (!global_config_context || !config_file) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    FILE* file = fopen(config_file, "w");
    if (!file) {
        return QGT_ERROR_IO_ERROR;
    }
    
    fprintf(file, "max_threads=%d\n", global_config_context->max_threads);
    fprintf(file, "memory_limit=%zu\n", global_config_context->memory_limit);
    fprintf(file, "gpu_enabled=%s\n", global_config_context->gpu_enabled ? "true" : "false");
    
    const char* opt_level;
    switch (global_config_context->optimization_level) {
        case OPTIMIZATION_LEVEL_NONE:
            opt_level = "none";
            break;
        case OPTIMIZATION_LEVEL_AGGRESSIVE:
            opt_level = "aggressive";
            break;
        default:
            opt_level = "normal";
            break;
    }
    fprintf(file, "optimization_level=%s\n", opt_level);
    
    fclose(file);
    return QGT_SUCCESS;
}
