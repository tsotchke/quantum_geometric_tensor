#include "quantum_geometric/core/advanced_memory_system.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_types.h"
#include "quantum_geometric/core/error_codes.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Global singleton memory system
static advanced_memory_system_t* g_memory_system = NULL;
static int g_ref_count = 0;

// Memory block tracking
#define MAX_TRACKED_BLOCKS 1024
static void* g_tracked_blocks[MAX_TRACKED_BLOCKS];
static size_t g_num_tracked_blocks = 0;

// Get the global memory system
advanced_memory_system_t* get_global_memory_system(void) {
    return g_memory_system;
}

// Register a memory system as the global singleton
void register_memory_system(advanced_memory_system_t* system) {
    if (!system) return;
    
    if (g_memory_system == NULL) {
        g_memory_system = system;
    }
    
    g_ref_count++;
    printf("DEBUG: Registered memory system (ref_count=%d)\n", g_ref_count);
}

// Unregister a memory system
void unregister_memory_system(advanced_memory_system_t* system) {
    if (!system || g_memory_system != system) return;
    
    g_ref_count--;
    printf("DEBUG: Unregistered memory system (ref_count=%d)\n", g_ref_count);
    
    if (g_ref_count <= 0) {
        g_memory_system = NULL;
        g_ref_count = 0;
        
        // Clean up any remaining tracked blocks
        for (size_t i = 0; i < g_num_tracked_blocks; i++) {
            if (g_tracked_blocks[i]) {
                printf("DEBUG: Warning: Freeing leaked block at %p\n", g_tracked_blocks[i]);
                free(g_tracked_blocks[i]);
                g_tracked_blocks[i] = NULL;
            }
        }
        g_num_tracked_blocks = 0;
    }
}

// Track an allocated memory block
bool track_memory_block(void* ptr) {
    if (!ptr) return false;
    
    // Check if already tracked
    for (size_t i = 0; i < g_num_tracked_blocks; i++) {
        if (g_tracked_blocks[i] == ptr) {
            return true; // Already tracked
        }
    }
    
    // Check if we have room to track more blocks
    if (g_num_tracked_blocks >= MAX_TRACKED_BLOCKS) {
        printf("DEBUG: Error: Maximum tracked blocks reached\n");
        return false;
    }
    
    // Add to tracking array
    g_tracked_blocks[g_num_tracked_blocks++] = ptr;
    return true;
}

// Untrack a memory block
bool untrack_memory_block(void* ptr) {
    if (!ptr) return false;
    
    for (size_t i = 0; i < g_num_tracked_blocks; i++) {
        if (g_tracked_blocks[i] == ptr) {
            // Remove by shifting remaining elements
            if (i < g_num_tracked_blocks - 1) {
                memmove(&g_tracked_blocks[i],
                       &g_tracked_blocks[i + 1],
                       (g_num_tracked_blocks - i - 1) * sizeof(void*));
            }
            g_num_tracked_blocks--;
            return true;
        }
    }
    return false;
}

// Check if a memory block is tracked
bool is_memory_block_tracked(void* ptr) {
    if (!ptr) return false;
    
    for (size_t i = 0; i < g_num_tracked_blocks; i++) {
        if (g_tracked_blocks[i] == ptr) {
            return true;
        }
    }
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
        printf("DEBUG: Warning: Attempting to free untracked block at %p\n", ptr);
        return;  // Don't free memory we don't own
    }
    
    // Untrack the block
    untrack_memory_block(ptr);
    
    // Use the original memory_free function
    memory_free(system, ptr);
}
