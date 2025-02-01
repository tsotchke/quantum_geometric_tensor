#ifndef QUANTUM_GEOMETRIC_HARDWARE_H
#define QUANTUM_GEOMETRIC_HARDWARE_H

#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/hardware/quantum_hardware_types.h"

// Hardware configuration structure
typedef struct {
    int gpu_device_id;
    const char* qpu_connection_string;
} hardware_config_t;

// Hardware status structure
typedef struct {
    bool is_initialized;
    bool is_available;
    HardwareBackendType type;
    size_t dimension;
    size_t memory_used;
} hardware_status_t;

// Create hardware context
qgt_error_t geometric_create_hardware(quantum_geometric_hardware_t** hardware,
                                    HardwareBackendType type,
                                    size_t dimension);

// Destroy hardware context
void geometric_destroy_hardware(quantum_geometric_hardware_t* hardware);

// Initialize hardware
qgt_error_t geometric_initialize_hardware(quantum_geometric_hardware_t* hardware,
                                        const hardware_config_t* config);

// Execute hardware operation
qgt_error_t geometric_execute_hardware(quantum_geometric_hardware_t* hardware,
                                     const ComplexFloat* input,
                                     ComplexFloat* output,
                                     size_t size);

// Get hardware status
qgt_error_t geometric_get_hardware_status(const quantum_geometric_hardware_t* hardware,
                                        hardware_status_t* status);

#endif // QUANTUM_GEOMETRIC_HARDWARE_H
