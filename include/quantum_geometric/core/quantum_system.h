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
#define QUANTUM_USE_TOPOLOGY (1 << 6)

// Error correction flags (with guards to prevent redefinition)
#ifndef QUANTUM_ERROR_NONE
#define QUANTUM_ERROR_NONE (0)
#endif
#ifndef QUANTUM_ERROR_BASIC
#define QUANTUM_ERROR_BASIC (1 << 0)
#endif
#ifndef QUANTUM_ERROR_ADAPTIVE
#define QUANTUM_ERROR_ADAPTIVE (1 << 1)
#endif
#ifndef QUANTUM_ERROR_SURFACE_CODE
#define QUANTUM_ERROR_SURFACE_CODE (1 << 2)
#endif

// System functions
quantum_system_t* quantum_system_create(size_t num_qubits, int flags);
void quantum_system_destroy(quantum_system_t* system);
int quantum_system_set_device(quantum_system_t* system, int device_type);
int quantum_system_get_device(const quantum_system_t* system);
size_t quantum_system_get_dimension(const quantum_system_t* system);

// Quantum cache configuration structure
typedef struct {
    double precision;              // Numerical precision threshold
    double success_probability;    // Target success probability for operations
    bool use_quantum_memory;       // Enable quantum-optimized memory access
    uint32_t error_correction;     // Error correction mode flags
    uint32_t optimization_level;   // Optimization aggressiveness
    uint32_t cache_type;           // Cache optimization strategy
} quantum_system_config_t;

// Configure quantum system with cache parameters
int quantum_system_configure(quantum_system_t* system, const quantum_system_config_t* config);

// Get current system precision
double quantum_system_get_precision(const quantum_system_t* system);

// Set error correction mode
int quantum_system_set_error_correction(quantum_system_t* system, uint32_t mode);

#endif // QUANTUM_SYSTEM_H
