#include "quantum_geometric/core/error_handling.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/memory_pool.h"
#include <stdlib.h>
#include <string.h>

// Internal helper functions
static void cleanup_protection_system(quantum_protection_t* protection) {
    if (!protection) return;
    if (protection->error_correction) free(protection->error_correction);
    if (protection->syndrome_extraction) free(protection->syndrome_extraction);
    if (protection->decoder) free(protection->decoder);
    free(protection);
}

static qgt_error_t validate_system_state(const quantum_system_t* system) {
    if (!system) return QGT_ERROR_INVALID_PARAMETER;
    if (system->num_qubits == 0) return QGT_ERROR_INVALID_DIMENSION;
    if (system->flags & ~(QUANTUM_SYSTEM_FLAG_OPTIMIZED | 
                         QUANTUM_SYSTEM_FLAG_PROTECTED |
                         QUANTUM_SYSTEM_FLAG_DISTRIBUTED |
                         QUANTUM_SYSTEM_FLAG_CACHED |
                         QUANTUM_SYSTEM_FLAG_MONITORED)) {
        return QGT_ERROR_INVALID_PARAMETER;
    }
    return QGT_SUCCESS;
}

quantum_system_t* quantum_system_create(size_t num_qubits, int flags) {
    // Parameter validation
    if (num_qubits == 0 || num_qubits > QGT_MAX_DIMENSIONS) {
        return NULL;
    }
    
    // Allocate system
    quantum_system_t* system = (quantum_system_t*)malloc(sizeof(quantum_system_t));
    if (!system) {
        return NULL;
    }
    
    // Initialize with default values
    system->num_qubits = num_qubits;
    system->num_classical_bits = 0;
    system->flags = flags;
    system->device_type = 0;
    system->device_data = NULL;
    system->state = NULL;
    system->operations = NULL;
    system->hardware = NULL;
    
    // Validate initial state
    if (validate_system_state(system) != QGT_SUCCESS) {
        free(system);
        return NULL;
    }
    
    return system;
}

void quantum_system_destroy(quantum_system_t* system) {
    if (!system) {
        return;
    }
    
    // Free device data if it exists
    if (system->device_data) {
        if (system->flags & QUANTUM_SYSTEM_FLAG_PROTECTED) {
            cleanup_protection_system((quantum_protection_t*)system->device_data);
        } else {
            free(system->device_data);
        }
        system->device_data = NULL;
    }
    
    // Free other resources
    if (system->state) free(system->state);
    if (system->operations) free(system->operations);
    if (system->hardware) free(system->hardware);
    
    free(system);
}

qgt_error_t quantum_protect_config(void* config, quantum_system_t* system) {
    // Parameter validation
    if (!config || !system) {
        return QGT_ERROR_INVALID_PARAMETER;
    }

    qgt_error_t validation_result = validate_system_state(system);
    if (validation_result != QGT_SUCCESS) {
        return validation_result;
    }

    geometric_encoding_config_t* encoding_config = (geometric_encoding_config_t*)config;
    
    // Validate configuration
    if (encoding_config->error_rate < 0.0f || encoding_config->error_rate > 1.0f) {
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    // Apply error protection based on configuration
    if (encoding_config->flags & QG_FLAG_ERROR_CORRECT) {
        // Initialize error correction
        quantum_protection_t* protection = (quantum_protection_t*)malloc(sizeof(quantum_protection_t));
        if (!protection) {
            return QGT_ERROR_ALLOCATION_FAILED;
        }
        
        protection->num_qubits = system->num_qubits;
        protection->code_distance = 3; // Default code distance
        protection->error_correction = NULL;
        protection->syndrome_extraction = NULL;
        protection->decoder = NULL;
        
        // Clean up existing protection if any
        if (system->device_data && (system->flags & QUANTUM_SYSTEM_FLAG_PROTECTED)) {
            cleanup_protection_system((quantum_protection_t*)system->device_data);
        }
        
        // Store protection configuration
        system->device_data = protection;
        system->flags |= QUANTUM_SYSTEM_FLAG_PROTECTED;
    }
    
    return QGT_SUCCESS;
}

qgt_error_t quantum_protect_distribution(void* distribution, quantum_system_t* system) {
    // Parameter validation
    if (!distribution || !system) {
        return QGT_ERROR_INVALID_PARAMETER;
    }

    qgt_error_t validation_result = validate_system_state(system);
    if (validation_result != QGT_SUCCESS) {
        return validation_result;
    }

    memory_distribution_t* mem_dist = (memory_distribution_t*)distribution;
    
    // Validate distribution configuration
    if (!mem_dist->node_sizes || !mem_dist->node_ptrs || mem_dist->num_nodes == 0) {
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    // Apply memory protection and distribution
    if (system->flags & QUANTUM_SYSTEM_FLAG_DISTRIBUTED) {
        // Initialize distributed memory protection
        distributed_memory_config_t* dist_config = 
            (distributed_memory_config_t*)malloc(sizeof(distributed_memory_config_t));
        if (!dist_config) {
            return QGT_ERROR_ALLOCATION_FAILED;
        }
        
        // Calculate memory distribution
        dist_config->total_memory = 0;
        for (size_t i = 0; i < mem_dist->num_nodes; i++) {
            if (mem_dist->node_sizes[i] == 0) {
                free(dist_config);
                return QGT_ERROR_INVALID_PARAMETER;
            }
            dist_config->total_memory += mem_dist->node_sizes[i];
        }
        
        dist_config->local_memory = mem_dist->node_sizes[0];
        dist_config->shared_memory = dist_config->total_memory - dist_config->local_memory;
        dist_config->use_numa = true;
        dist_config->numa_node = 0;
        
        // Clean up existing configuration if any
        if (system->device_data) {
            free(system->device_data);
        }
        
        // Store distribution configuration
        system->device_data = dist_config;
    }
    
    return QGT_SUCCESS;
}
