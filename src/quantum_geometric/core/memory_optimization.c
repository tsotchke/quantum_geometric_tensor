/**
 * @file memory_optimization.c
 * @brief Platform-independent memory optimization wrapper
 */

#include "quantum_geometric/core/memory_optimization.h"
#include "quantum_geometric/core/memory_optimization_impl.h"
#include "quantum_geometric/core/quantum_geometric_logging.h"
#include <stdlib.h>
#include <string.h>

#ifdef __linux__
#include <numa.h>
#else
// NUMA stub functions for non-Linux systems
static inline int numa_available(void) { return -1; }
static inline int numa_num_configured_nodes(void) { return 1; }
static inline void numa_set_preferred(int node) {}
static inline void numa_set_localalloc(void) {}
static inline void* numa_alloc_onnode(size_t size, int node) { return malloc(size); }
static inline void numa_free(void* ptr, size_t size) { free(ptr); }
#endif

// Global memory manager instance
static MemoryManager* g_memory_manager = NULL;

// Initialize memory optimization system
qgt_error_t init_memory_optimization(const memory_optimization_config_t* config) {
    if (!config) {
        log_error("Invalid memory optimization configuration");
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    if (g_memory_manager) {
        log_warning("Memory optimization system already initialized");
        return QGT_ERROR_ALREADY_INITIALIZED;
    }

    g_memory_manager = init_memory_manager();
    if (!g_memory_manager) {
        log_error("Failed to initialize memory manager");
        return QGT_ERROR_INITIALIZATION_FAILED;
    }

    return QGT_SUCCESS;
}

// Cleanup memory optimization system
void cleanup_memory_optimization(void) {
    if (g_memory_manager) {
        cleanup_memory_manager(g_memory_manager);
        g_memory_manager = NULL;
    }
}

// Register memory region for optimization
qgt_error_t register_memory_region(memory_region_t* region, void* base, size_t size) {
    if (!g_memory_manager) {
        log_error("Memory optimization system not initialized");
        return QGT_ERROR_NOT_INITIALIZED;
    }

    if (!region || !base || size == 0) {
        log_error("Invalid arguments for register_memory_region");
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    region->base_address = base;
    region->size = size;
    region->access_pattern = ACCESS_PATTERN_SEQUENTIAL; // Default pattern
    region->strategy = STRATEGY_POOL_ALLOCATION;        // Default strategy
    region->is_optimized = false;

    // Initialize stats
    memset(&region->stats, 0, sizeof(memory_stats_t));
    region->stats.access_pattern = ACCESS_PATTERN_SEQUENTIAL;

    return QGT_SUCCESS;
}

// Analyze memory access patterns
qgt_error_t analyze_memory_pattern(memory_region_t* region) {
    if (!g_memory_manager || !region) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // Get current stats from platform-specific implementation
    region->stats = get_memory_stats(g_memory_manager);

    // Analyze access pattern based on stats
    if (region->stats.cache_misses < region->stats.total_allocations / 10) {
        region->access_pattern = ACCESS_PATTERN_SEQUENTIAL;
    } else if (region->stats.page_faults > region->stats.total_allocations / 5) {
        region->access_pattern = ACCESS_PATTERN_RANDOM;
    } else {
        region->access_pattern = ACCESS_PATTERN_STRIDED;
    }

    return QGT_SUCCESS;
}

// Apply optimization strategy
qgt_error_t optimize_memory_region(memory_region_t* region, memory_strategy_t strategy) {
    if (!g_memory_manager || !region) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    region->strategy = strategy;

    // Apply strategy using platform-specific allocator
    void* new_base = optimized_malloc(g_memory_manager, region->size);
    if (!new_base) {
        region->is_optimized = false;
        return QGT_ERROR_ALLOCATION_FAILED;
    }

    // Copy existing data to optimized location
    memcpy(new_base, region->base_address, region->size);
    optimized_free(g_memory_manager, region->base_address);
    region->base_address = new_base;
    region->is_optimized = true;

    return QGT_SUCCESS;
}

// Update memory statistics
qgt_error_t update_memory_stats(memory_region_t* region) {
    if (!g_memory_manager || !region) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    region->stats = get_memory_stats(g_memory_manager);
    return QGT_SUCCESS;
}

// Get memory statistics
const memory_stats_t* get_memory_stats(const memory_region_t* region) {
    if (!region) {
        return NULL;
    }
    return &region->stats;
}

// Memory prefetching functions
qgt_error_t prefetch_memory(const memory_region_t* region, size_t offset, size_t size) {
    if (!g_memory_manager || !region || offset + size > region->size) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    void* prefetch_addr = (char*)region->base_address + offset;
    if (!prefetch_memory_range(prefetch_addr, size)) {
        return QGT_ERROR_OPERATION_FAILED;
    }

    return QGT_SUCCESS;
}

qgt_error_t configure_prefetch(size_t distance, size_t stride) {
    if (!g_memory_manager) {
        return QGT_ERROR_NOT_INITIALIZED;
    }
    return QGT_SUCCESS; // Platform-specific implementation may override
}

// Memory pool integration
qgt_error_t optimize_pool_allocation(MemoryPool* pool) {
    if (!g_memory_manager || !pool) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    return optimize_pool(pool) ? QGT_SUCCESS : QGT_ERROR_OPERATION_FAILED;
}

qgt_error_t optimize_pool_fragmentation(MemoryPool* pool) {
    if (!g_memory_manager || !pool) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    return defragment_pool(pool) ? QGT_SUCCESS : QGT_ERROR_OPERATION_FAILED;
}

// Memory compression functions
qgt_error_t compress_memory_region(memory_region_t* region) {
    if (!g_memory_manager || !region) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    size_t compressed_size;
    if (!compress_memory(region->base_address, region->size, &compressed_size)) {
        return QGT_ERROR_OPERATION_FAILED;
    }

    return QGT_SUCCESS;
}

qgt_error_t decompress_memory_region(memory_region_t* region) {
    if (!g_memory_manager || !region) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    size_t output_size;
    if (!decompress_memory(region->base_address, region->size, 
                          region->base_address, &output_size)) {
        return QGT_ERROR_OPERATION_FAILED;
    }

    return QGT_SUCCESS;
}

// Memory access optimization
qgt_error_t optimize_access_pattern(memory_region_t* region) {
    if (!g_memory_manager || !region) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    void* aligned_addr = region->base_address;
    if (!align_memory_to_cache(&aligned_addr)) {
        return QGT_ERROR_OPERATION_FAILED;
    }

    region->base_address = aligned_addr;
    return QGT_SUCCESS;
}

qgt_error_t reorder_memory_layout(memory_region_t* region) {
    if (!g_memory_manager || !region) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // Platform-specific implementation may provide optimized layout
    return QGT_SUCCESS;
}

// Memory defragmentation
qgt_error_t defragment_memory(memory_region_t* region) {
    if (!g_memory_manager || !region) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // Use platform-specific pool defragmentation
    MemoryPool pool = {0};
    pool.base_address = region->base_address;
    pool.total_size = region->size;

    return defragment_pool(&pool) ? QGT_SUCCESS : QGT_ERROR_OPERATION_FAILED;
}

qgt_error_t compact_memory(memory_region_t* region) {
    if (!g_memory_manager || !region) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // Platform-specific implementation may provide memory compaction
    return QGT_SUCCESS;
}

// Memory monitoring functions
qgt_error_t start_memory_monitoring(memory_region_t* region) {
    if (!g_memory_manager || !region) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // Platform-specific implementation may start monitoring
    return QGT_SUCCESS;
}

qgt_error_t stop_memory_monitoring(memory_region_t* region) {
    if (!g_memory_manager || !region) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // Platform-specific implementation may stop monitoring
    return QGT_SUCCESS;
}

qgt_error_t get_monitoring_results(const memory_region_t* region, memory_stats_t* stats) {
    if (!g_memory_manager || !region || !stats) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    *stats = get_memory_stats(g_memory_manager);
    return QGT_SUCCESS;
}

// Memory optimization suggestions
qgt_error_t get_optimization_suggestions(const memory_region_t* region,
                                       optimization_suggestion_t* suggestions,
                                       size_t* count) {
    if (!g_memory_manager || !region || !suggestions || !count) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    memory_stats_t stats = get_memory_stats(g_memory_manager);
    *count = 0;

    // Suggest based on fragmentation
    if (stats.fragmentation_ratio > 0.3) {
        suggestions[*count].recommended_strategy = STRATEGY_POOL_ALLOCATION;
        suggestions[*count].expected_improvement = 0.4;
        snprintf(suggestions[*count].description, 256,
                "High fragmentation detected (%.2f). Pool allocation recommended.",
                stats.fragmentation_ratio);
        (*count)++;
    }

    // Suggest based on cache misses
    if (stats.cache_misses > stats.total_allocations / 10) {
        suggestions[*count].recommended_strategy = STRATEGY_CACHE_ALIGNED;
        suggestions[*count].expected_improvement = 0.3;
        snprintf(suggestions[*count].description, 256,
                "High cache miss rate detected. Cache alignment recommended.");
        (*count)++;
    }

    // Suggest NUMA optimization if supported
    if (has_numa_support() && get_numa_node_count() > 1) {
        suggestions[*count].recommended_strategy = STRATEGY_NUMA_AWARE;
        suggestions[*count].expected_improvement = 0.2;
        snprintf(suggestions[*count].description, 256,
                "Multiple NUMA nodes detected. NUMA-aware allocation recommended.");
        (*count)++;
    }

    return QGT_SUCCESS;
}

// Memory validation functions
qgt_error_t validate_memory_optimization(const memory_region_t* region) {
    if (!g_memory_manager || !region) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    return region->is_optimized ? QGT_SUCCESS : QGT_ERROR_NOT_OPTIMIZED;
}

qgt_error_t verify_memory_integrity(const memory_region_t* region) {
    if (!g_memory_manager || !region || !region->base_address) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    return verify_memory_protection(region->base_address, region->size) ?
           QGT_SUCCESS : QGT_ERROR_MEMORY_CORRUPTION;
}
