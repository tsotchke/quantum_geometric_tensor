#ifndef CACHE_MANAGER_H
#define CACHE_MANAGER_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include "quantum_geometric/core/quantum_complex.h"
#include "quantum_geometric/hardware/quantum_hardware_types.h"
#include "quantum_geometric/core/quantum_geometric_types.h"

// Cache optimization flags
#define QUANTUM_OPTIMIZE_AGGRESSIVE  (1 << 0)
#define QUANTUM_USE_CACHE           (1 << 1)
#define QUANTUM_USE_MEMORY          (1 << 2)
#define QUANTUM_ERROR_ADAPTIVE      (1 << 3)
#define QUANTUM_OPT_AGGRESSIVE      (1 << 4)
#define QUANTUM_MEMORY_OPTIMAL      (1 << 5)

// Forward declarations
typedef struct quantum_circuit_t quantum_circuit_t;
typedef struct quantum_system_t quantum_system_t;

// Quantum protection type
typedef struct {
    ComplexFloat* error_syndrome;
    size_t syndrome_size;
    float error_threshold;
    bool use_quantum_correction;
} quantum_protection_t;

// Cache entry type
typedef struct cache_entry {
    void* data;
    size_t size;
    uint64_t tag;
    bool valid;
    bool dirty;
    uint64_t access_count;
    quantum_protection_t protection;
    struct cache_entry* next;
} cache_entry_t;

// Quantum cache type
typedef struct {
    HardwareBackendType hardware_type;
    quantum_geometric_hardware_t* hardware;
    ComplexFloat* quantum_state;
    size_t state_size;
    quantum_protection_t protection;
} quantum_cache_t;

// Cache configuration
typedef struct {
    size_t total_size;                    // Total cache size in bytes
    size_t line_size;                     // Cache line size in bytes
    bool use_quantum_optimization;         // Whether to use quantum optimization
    float error_threshold;                // Error threshold for quantum operations
    HardwareBackendType hw_type;    // Hardware type to use
    const char* qpu_connection_string;    // Connection string for quantum hardware (optional)
} cache_config_t;

// Error codes
#define CACHE_SUCCESS 0
#define CACHE_ERROR_INVALID_CONFIG -1
#define CACHE_ERROR_INIT_FAILED -2
#define CACHE_ERROR_NOT_INITIALIZED -3
#define CACHE_ERROR_INVALID_POINTER -4
#define CACHE_ERROR_OUT_OF_MEMORY -5

// Core functions
int qg_cache_init(const cache_config_t* config);
void* qg_cache_allocate(size_t size);
int qg_cache_free(void* ptr);
int qg_cache_optimize(void);
void qg_cache_cleanup(void);
float qg_cache_get_efficiency(void);

// Quantum system functions
quantum_system_t* quantum_system_create(size_t dimension, uint32_t flags);
void quantum_system_destroy(quantum_system_t* system);

// Quantum circuit functions
void quantum_circuit_destroy(quantum_circuit_t* circuit);
quantum_circuit_t* quantum_create_cache_circuit(size_t num_qubits, uint32_t flags);

// Cache configuration type
typedef struct {
    double precision;
    double success_probability;
    bool use_quantum_memory;
    uint32_t error_correction;
    uint32_t optimization_level;
    uint32_t cache_type;
} quantum_cache_config_t;

// Cache management functions
void quantum_cache_destroy(quantum_cache_t cache);

// Cache entry management
cache_entry_t* quantum_remove_cache_entry(
    cache_entry_t* entry,
    void* ptr,
    quantum_cache_t* cache,
    quantum_system_t* system,
    quantum_circuit_t* circuit,
    cache_config_t* config
);

void quantum_free_cache_entry(
    cache_entry_t* entry,
    quantum_cache_t* cache,
    quantum_system_t* system
);

#endif // CACHE_MANAGER_H
