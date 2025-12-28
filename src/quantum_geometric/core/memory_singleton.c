#include "quantum_geometric/core/advanced_memory_system.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_types.h"
#include "quantum_geometric/core/quantum_geometric_logging.h"
#include "quantum_geometric/core/error_codes.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>
#include <stdatomic.h>

// Global singleton memory system with thread-safe access
static advanced_memory_system_t* g_memory_system = NULL;
static _Atomic int g_ref_count = 0;
static pthread_mutex_t g_singleton_mutex = PTHREAD_MUTEX_INITIALIZER;

// Memory block tracking
#define MAX_TRACKED_BLOCKS 1024
static void* g_tracked_blocks[MAX_TRACKED_BLOCKS];
static size_t g_num_tracked_blocks = 0;
static pthread_mutex_t g_tracking_mutex = PTHREAD_MUTEX_INITIALIZER;

// Get the global memory system
advanced_memory_system_t* get_global_memory_system(void) {
    return g_memory_system;
}

// Register a memory system as the global singleton (thread-safe)
void register_memory_system(advanced_memory_system_t* system) {
    if (!system) return;

    pthread_mutex_lock(&g_singleton_mutex);

    if (g_memory_system == NULL) {
        g_memory_system = system;
    }

    int new_count = atomic_fetch_add(&g_ref_count, 1) + 1;

    pthread_mutex_unlock(&g_singleton_mutex);

    geometric_log_debug("Registered memory system (ref_count=%d)", new_count);
}

// Unregister a memory system (thread-safe)
void unregister_memory_system(advanced_memory_system_t* system) {
    if (!system) return;

    pthread_mutex_lock(&g_singleton_mutex);

    if (g_memory_system != system) {
        pthread_mutex_unlock(&g_singleton_mutex);
        return;
    }

    int new_count = atomic_fetch_sub(&g_ref_count, 1) - 1;

    geometric_log_debug("Unregistered memory system (ref_count=%d)", new_count);

    if (new_count <= 0) {
        g_memory_system = NULL;
        atomic_store(&g_ref_count, 0);

        // Clean up any remaining tracked blocks under tracking mutex
        pthread_mutex_lock(&g_tracking_mutex);
        for (size_t i = 0; i < g_num_tracked_blocks; i++) {
            if (g_tracked_blocks[i]) {
                geometric_log_warning("Freeing leaked block at %p", g_tracked_blocks[i]);
                free(g_tracked_blocks[i]);
                g_tracked_blocks[i] = NULL;
            }
        }
        g_num_tracked_blocks = 0;
        pthread_mutex_unlock(&g_tracking_mutex);
    }

    pthread_mutex_unlock(&g_singleton_mutex);
}

// Track an allocated memory block (thread-safe)
bool track_memory_block(void* ptr) {
    if (!ptr) return false;

    pthread_mutex_lock(&g_tracking_mutex);

    // Check if already tracked
    for (size_t i = 0; i < g_num_tracked_blocks; i++) {
        if (g_tracked_blocks[i] == ptr) {
            pthread_mutex_unlock(&g_tracking_mutex);
            return true; // Already tracked
        }
    }

    // Check if we have room to track more blocks
    if (g_num_tracked_blocks >= MAX_TRACKED_BLOCKS) {
        pthread_mutex_unlock(&g_tracking_mutex);
        geometric_log_error("Maximum tracked blocks reached (%d)", MAX_TRACKED_BLOCKS);
        return false;
    }

    // Add to tracking array
    g_tracked_blocks[g_num_tracked_blocks++] = ptr;

    pthread_mutex_unlock(&g_tracking_mutex);
    return true;
}

// Untrack a memory block (thread-safe)
bool untrack_memory_block(void* ptr) {
    if (!ptr) return false;

    pthread_mutex_lock(&g_tracking_mutex);

    for (size_t i = 0; i < g_num_tracked_blocks; i++) {
        if (g_tracked_blocks[i] == ptr) {
            // Remove by shifting remaining elements
            if (i < g_num_tracked_blocks - 1) {
                memmove(&g_tracked_blocks[i],
                       &g_tracked_blocks[i + 1],
                       (g_num_tracked_blocks - i - 1) * sizeof(void*));
            }
            g_num_tracked_blocks--;
            pthread_mutex_unlock(&g_tracking_mutex);
            return true;
        }
    }

    pthread_mutex_unlock(&g_tracking_mutex);
    return false;
}

// Check if a memory block is tracked (thread-safe)
bool is_memory_block_tracked(void* ptr) {
    if (!ptr) return false;

    pthread_mutex_lock(&g_tracking_mutex);

    for (size_t i = 0; i < g_num_tracked_blocks; i++) {
        if (g_tracked_blocks[i] == ptr) {
            pthread_mutex_unlock(&g_tracking_mutex);
            return true;
        }
    }

    pthread_mutex_unlock(&g_tracking_mutex);
    return false;
}

// Safe memory allocation wrapper
void* safe_memory_allocate(advanced_memory_system_t* system, size_t size, size_t alignment) {
    if (!system || size == 0) return NULL;
    
    // Use the original memory_allocate function
    void* ptr = memory_allocate(system, size, alignment);
    
    // Track the allocation
    if (ptr) {
        track_memory_block(ptr);
    }
    
    return ptr;
}

// Safe memory free wrapper
void safe_memory_free(advanced_memory_system_t* system, void* ptr) {
    if (!system || !ptr) return;
    
    // Check if this block is tracked
    if (!is_memory_block_tracked(ptr)) {
        geometric_log_warning("Attempting to free untracked block at %p", ptr);
        return;  // Don't free memory we don't own
    }
    
    // Untrack the block
    untrack_memory_block(ptr);
    
    // Use the original memory_free function
    memory_free(system, ptr);
}
