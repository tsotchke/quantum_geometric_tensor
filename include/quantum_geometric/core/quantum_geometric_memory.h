#ifndef QUANTUM_GEOMETRIC_MEMORY_H
#define QUANTUM_GEOMETRIC_MEMORY_H

#include "quantum_geometric/core/quantum_geometric_types.h"
#include <stddef.h>

#include "quantum_geometric/core/memory_pool.h"

// Initialize memory system with optimized configuration
qgt_error_t geometric_init_memory(void);

// Cleanup memory system
void geometric_cleanup_memory(void);

// Get memory pool instance
MemoryPool* geometric_get_memory_pool(void);

// Allocate aligned memory
void* geometric_allocate(size_t size);

// Free memory
void geometric_free(void* ptr);

// Get memory statistics
void geometric_get_memory_stats(size_t* total, size_t* peak, size_t* count);

// Reset memory system
qgt_error_t geometric_reset_memory(void);

// Convenience macros for memory allocation
#define QGT_ALLOC(size) geometric_allocate(size)
#define QGT_FREE(ptr) geometric_free(ptr)

#endif // QUANTUM_GEOMETRIC_MEMORY_H
