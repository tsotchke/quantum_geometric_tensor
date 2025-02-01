#ifndef QUANTUM_SYSTEM_H
#define QUANTUM_SYSTEM_H

#include "quantum_geometric/core/quantum_types.h"

// System flags
#define QUANTUM_USE_CPU (1 << 0)
#define QUANTUM_USE_GPU (1 << 1)
#define QUANTUM_USE_METAL (1 << 2)
#define QUANTUM_USE_CUDA (1 << 3)
#define QUANTUM_USE_OPENCL (1 << 4)
#define QUANTUM_USE_ESTIMATION (1 << 5)

// Error correction flags
#define QUANTUM_ERROR_NONE (0)
#define QUANTUM_ERROR_BASIC (1 << 0)
#define QUANTUM_ERROR_ADAPTIVE (1 << 1)
#define QUANTUM_ERROR_SURFACE_CODE (1 << 2)

// System functions
quantum_system_t* quantum_system_create(size_t num_qubits, int flags);
void quantum_system_destroy(quantum_system_t* system);
int quantum_system_set_device(quantum_system_t* system, int device_type);
int quantum_system_get_device(const quantum_system_t* system);
size_t quantum_system_get_dimension(const quantum_system_t* system);

#endif // QUANTUM_SYSTEM_H
