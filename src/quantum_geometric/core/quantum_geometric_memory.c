#include "quantum_geometric/core/quantum_geometric_memory.h"
#include "quantum_geometric/core/quantum_geometric_logging.h"
#include <stdlib.h>
#include <string.h>

// Global memory pool instance
static MemoryPool* global_memory_pool = NULL;

// Initialize memory system with optimized configuration
qgt_error_t geometric_init_memory(void) {
    if (global_memory_pool) {
        return QGT_ERROR_ALREADY_INITIALIZED;
    }
    
    // Configure memory pool with conservative settings
    PoolConfig config = {
        .min_block_size = QG_MIN_BLOCK_SIZE,
        .alignment = QG_POOL_ALIGNMENT,
        .num_size_classes = QG_NUM_SIZE_CLASSES,
        .growth_factor = 2.0f,
        .prefetch_distance = 1,  // Conservative prefetching
        .use_huge_pages = false, // Don't use huge pages
        .cache_local_free_lists = true,
        .max_blocks_per_class = 1024, // Reduced block limit
        .thread_cache_size = 64,   // Smaller thread cache
        .enable_stats = true
    };

    // Try to initialize with optimized settings
    global_memory_pool = init_memory_pool(&config);
    if (!global_memory_pool) {
        // Fall back to minimal configuration
        config.min_block_size = QG_MIN_BLOCK_SIZE;  // Keep minimum block size
        config.num_size_classes = 32;  // Reduced but still sufficient
        config.max_blocks_per_class = 2048;  // Moderate block limit
        config.thread_cache_size = 128;  // Moderate thread cache
        config.prefetch_distance = 2;  // Moderate prefetching
        config.cache_local_free_lists = true;  // Keep thread caching
        
        global_memory_pool = init_memory_pool(&config);
        if (!global_memory_pool) {
            geometric_log_error("Failed to initialize memory pool with fallback configuration");
            return QGT_ERROR_ALLOCATION_FAILED;
        }
        geometric_log_info("Memory pool initialized with fallback configuration");
    } else {
        geometric_log_info("Memory pool initialized with optimized configuration");
    }
    
    geometric_log_info("Memory pool initialized successfully");
    return QGT_SUCCESS;
}

// Cleanup memory system
void geometric_cleanup_memory(void) {
    if (global_memory_pool) {
        cleanup_memory_pool(global_memory_pool);
        global_memory_pool = NULL;
    }
}

// Get memory pool instance
MemoryPool* geometric_get_memory_pool(void) {
    return global_memory_pool;
}

// Allocate aligned memory
void* geometric_allocate(size_t size) {
    if (!global_memory_pool) {
        geometric_log_error("Memory pool not initialized");
        return NULL;
    }
    
    void* ptr = pool_malloc(global_memory_pool, size);
    if (!ptr) {
        geometric_log_error("Failed to allocate %zu bytes", size);
    }
    return ptr;
}

// Free memory
void geometric_free(void* ptr) {
    if (!global_memory_pool || !ptr) {
        return;
    }
    pool_free(global_memory_pool, ptr);
}

// Get memory statistics
void geometric_get_memory_stats(size_t* total, size_t* peak, size_t* count) {
    if (!global_memory_pool || !total || !peak || !count) {
        return;
    }
    
    *total = get_total_allocated(global_memory_pool);
    *peak = get_peak_allocated(global_memory_pool);
    *count = get_num_allocations(global_memory_pool);
}

// Reset memory system
qgt_error_t geometric_reset_memory(void) {
    if (!global_memory_pool) {
        return QGT_ERROR_NOT_INITIALIZED;
    }
    
    cleanup_memory_pool(global_memory_pool);
    return geometric_init_memory();
}
