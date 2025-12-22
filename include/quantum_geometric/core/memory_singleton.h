#ifndef MEMORY_SINGLETON_H
#define MEMORY_SINGLETON_H

#include "quantum_geometric/core/advanced_memory_system.h"
#include <stdbool.h>

// Get the global memory system
advanced_memory_system_t* get_global_memory_system(void);

// Register a memory system as the global singleton
void register_memory_system(advanced_memory_system_t* system);

// Unregister a memory system
void unregister_memory_system(advanced_memory_system_t* system);

// Memory block tracking functions
bool track_memory_block(void* ptr);
bool untrack_memory_block(void* ptr);
bool is_memory_block_tracked(void* ptr);

// Safe memory allocation/deallocation wrappers
void* safe_memory_allocate(advanced_memory_system_t* system, size_t size, size_t alignment);
void safe_memory_free(advanced_memory_system_t* system, void* ptr);

#endif // MEMORY_SINGLETON_H
