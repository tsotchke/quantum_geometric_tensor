#include "quantum_geometric/core/quantum_system.h"
#include <stdlib.h>
#include <string.h>

quantum_system_t* quantum_system_create(size_t num_qubits, int flags) {
    quantum_system_t* system = malloc(sizeof(quantum_system_t));
    if (!system) return NULL;
    
    system->num_qubits = num_qubits;
    system->num_classical_bits = 0;
    system->flags = flags;
    system->device_type = QUANTUM_USE_CPU;  // Default to CPU
    system->device_data = NULL;  // No device data by default
    system->state = NULL;
    system->operations = NULL;
    system->hardware = NULL;
    
    return system;
}

void quantum_system_destroy(quantum_system_t* system) {
    if (!system) return;
    
    // Free all allocated resources in reverse order of allocation
    if (system->hardware) {
        free(system->hardware);
        system->hardware = NULL;
    }
    
    if (system->operations) {
        free(system->operations);
        system->operations = NULL;
    }
    
    if (system->state) {
        free(system->state);
        system->state = NULL;
    }
    
    if (system->device_data) {
        free(system->device_data);
        system->device_data = NULL;
    }
    
    free(system);
}

int quantum_system_set_device(quantum_system_t* system, int device_type) {
    if (!system) return QGT_ERROR_INVALID_PARAMETER;
    
    // Validate device type
    switch (device_type) {
        case QUANTUM_USE_CPU:
        case QUANTUM_USE_GPU:
        case QUANTUM_USE_METAL:
        case QUANTUM_USE_CUDA:
        case QUANTUM_USE_OPENCL:
            system->device_type = device_type;
            return QGT_SUCCESS;
        default:
            return QGT_ERROR_INVALID_PARAMETER;
    }
}

int quantum_system_get_device(const quantum_system_t* system) {
    if (!system) return QGT_ERROR_INVALID_PARAMETER;
    return system->device_type;
}

size_t quantum_system_get_dimension(const quantum_system_t* system) {
    if (!system) return 0;
    return 1ULL << system->num_qubits;  // 2^num_qubits
}
