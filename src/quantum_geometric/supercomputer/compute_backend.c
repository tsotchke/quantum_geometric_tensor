/**
 * compute_backend.c - Backend registry and engine implementation
 *
 * This file manages backend registration and provides the high-level
 * compute engine API that automatically selects the best available backend.
 */

#include "quantum_geometric/supercomputer/compute_backend.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ============================================================================
// Backend Registry
// ============================================================================

#define MAX_BACKENDS 16

typedef struct {
    ComputeBackendInfo backends[MAX_BACKENDS];
    int count;
    bool initialized;
} BackendRegistry;

static BackendRegistry g_registry = { .count = 0, .initialized = false };

// ============================================================================
// Engine Structure
// ============================================================================

struct ComputeEngine {
    ComputeBackend* backend;
    const ComputeBackendInfo* backend_info;
    ComputeDistributedConfig config;
};

// ============================================================================
// Registry Functions
// ============================================================================

ComputeResult compute_register_backend(const ComputeBackendInfo* info) {
    if (!info || !info->ops) {
        return COMPUTE_ERROR_INVALID_ARGUMENT;
    }

    if (g_registry.count >= MAX_BACKENDS) {
        return COMPUTE_ERROR_INTERNAL;
    }

    // Check for duplicates
    for (int i = 0; i < g_registry.count; i++) {
        if (g_registry.backends[i].type == info->type) {
            // Update existing registration
            g_registry.backends[i] = *info;
            return COMPUTE_SUCCESS;
        }
    }

    // Add new registration
    g_registry.backends[g_registry.count++] = *info;
    g_registry.initialized = true;

    return COMPUTE_SUCCESS;
}

int compute_get_backend_count(void) {
    return g_registry.count;
}

const ComputeBackendInfo* compute_get_backend_info(int index) {
    if (index < 0 || index >= g_registry.count) {
        return NULL;
    }
    return &g_registry.backends[index];
}

const ComputeBackendInfo* compute_get_backend_by_type(ComputeBackendType type) {
    for (int i = 0; i < g_registry.count; i++) {
        if (g_registry.backends[i].type == type) {
            return &g_registry.backends[i];
        }
    }
    return NULL;
}

const ComputeBackendInfo* compute_select_backend(ComputeBackendType preferred,
                                                   bool allow_fallback) {
    const ComputeBackendInfo* best = NULL;
    int best_priority = -1;

    // If preferred is specified, check if it's available
    if (preferred >= 0 && preferred < COMPUTE_BACKEND_COUNT) {
        const ComputeBackendInfo* pref = compute_get_backend_by_type(preferred);
        if (pref && pref->ops->probe && pref->ops->probe()) {
            return pref;
        }
        if (!allow_fallback) {
            return NULL;
        }
    }

    // Find highest priority available backend
    for (int i = 0; i < g_registry.count; i++) {
        const ComputeBackendInfo* info = &g_registry.backends[i];
        if (info->ops->probe && info->ops->probe()) {
            if (info->priority > best_priority) {
                best = info;
                best_priority = info->priority;
            }
        }
    }

    return best;
}

bool compute_backend_available(ComputeBackendType type) {
    const ComputeBackendInfo* info = compute_get_backend_by_type(type);
    if (!info || !info->ops->probe) {
        return false;
    }
    return info->ops->probe();
}

// ============================================================================
// Engine Functions
// ============================================================================

ComputeEngine* compute_engine_init(const ComputeDistributedConfig* config) {
    if (!config) {
        return NULL;
    }

    // Select backend
    const ComputeBackendInfo* backend_info = compute_select_backend(
        config->preferred_backend,
        config->allow_fallback
    );

    if (!backend_info) {
        fprintf(stderr, "compute_engine_init: No available backend\n");
        return NULL;
    }

    // Allocate engine
    ComputeEngine* engine = calloc(1, sizeof(ComputeEngine));
    if (!engine) {
        return NULL;
    }

    // Store configuration
    engine->config = *config;
    engine->backend_info = backend_info;

    // Initialize backend
    if (backend_info->ops->init) {
        engine->backend = backend_info->ops->init(config);
        if (!engine->backend) {
            fprintf(stderr, "compute_engine_init: Failed to initialize %s backend\n",
                    backend_info->name);
            free(engine);
            return NULL;
        }
    }

    return engine;
}

void compute_engine_cleanup(ComputeEngine* engine) {
    if (!engine) return;

    if (engine->backend && engine->backend_info->ops->cleanup) {
        engine->backend_info->ops->cleanup(engine->backend);
    }

    free(engine);
}

ComputeBackendType compute_engine_get_backend_type(const ComputeEngine* engine) {
    if (!engine || !engine->backend_info) {
        return COMPUTE_BACKEND_CPU;
    }
    return engine->backend_info->type;
}

const ComputeBackendOps* compute_engine_get_ops(const ComputeEngine* engine) {
    if (!engine || !engine->backend_info) {
        return NULL;
    }
    return engine->backend_info->ops;
}

ComputeBackend* compute_engine_get_backend(const ComputeEngine* engine) {
    if (!engine) {
        return NULL;
    }
    return engine->backend;
}

// ============================================================================
// Convenience Functions
// ============================================================================

const char* compute_engine_get_backend_name(const ComputeEngine* engine) {
    if (!engine || !engine->backend_info) {
        return "Unknown";
    }
    return engine->backend_info->name;
}

void compute_print_available_backends(void) {
    printf("Available compute backends:\n");
    for (int i = 0; i < g_registry.count; i++) {
        const ComputeBackendInfo* info = &g_registry.backends[i];
        bool available = info->ops->probe ? info->ops->probe() : false;
        printf("  %s (v%s) - Priority: %d, Available: %s\n",
               info->name, info->version, info->priority,
               available ? "Yes" : "No");
    }
}

// ============================================================================
// Default Configuration
// ============================================================================

void compute_config_init_default(ComputeDistributedConfig* config) {
    if (!config) return;

    memset(config, 0, sizeof(ComputeDistributedConfig));

    config->num_nodes = 1;
    config->devices_per_node = 1;
    config->topology = COMPUTE_TOPO_SINGLE;

    config->device_buffer_size = 256 * 1024 * 1024;  // 256 MB
    config->host_buffer_size = 64 * 1024 * 1024;     // 64 MB
    config->comm_buffer_size = 16 * 1024 * 1024;     // 16 MB

    config->use_nccl = false;
    config->use_rdma = false;
    config->use_compression = false;

    config->preferred_backend = COMPUTE_BACKEND_CPU;
    config->allow_fallback = true;

    config->num_streams = 4;
    config->num_threads_per_node = 0;  // Auto-detect
    config->enable_async = true;

    config->monitor_config = NULL;
    config->backend_config = NULL;
}

// ============================================================================
// Static Backend Registration
// ============================================================================
//
// Note: Each backend uses COMPUTE_REGISTER_BACKEND macro which creates
// a constructor function that automatically registers the backend at
// library load time. The order of registration doesn't matter since
// we select by priority during compute_select_backend().
//
// If constructor functions aren't available (some embedded systems),
// you can manually register backends by calling:
//
//   extern void register_cpu_backend(void);
//   extern void register_metal_backend(void);
//   extern void register_opencl_backend(void);
//
//   register_cpu_backend();
//   register_metal_backend();
//   register_opencl_backend();
//
// ============================================================================
