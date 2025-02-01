#ifndef QUANTUM_GEOMETRIC_MEMORY_OPTIMIZATION_H
#define QUANTUM_GEOMETRIC_MEMORY_OPTIMIZATION_H

#include <stddef.h>
#include <stdbool.h>
#include "quantum_geometric/core/memory_pool.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/system_memory.h"

#ifdef __cplusplus
extern "C" {
#endif

// Memory access patterns
typedef enum {
    ACCESS_PATTERN_SEQUENTIAL,
    ACCESS_PATTERN_RANDOM,
    ACCESS_PATTERN_STRIDED,
    ACCESS_PATTERN_BLOCKED
} memory_access_pattern_t;

// Memory optimization strategy
typedef enum {
    STRATEGY_POOL_ALLOCATION,
    STRATEGY_PREFETCH,
    STRATEGY_CACHE_ALIGNED,
    STRATEGY_NUMA_AWARE,
    STRATEGY_COMPRESSION
} memory_strategy_t;

// Memory region statistics
typedef struct {
    size_t total_allocations;
    size_t total_deallocations;
    size_t peak_memory;
    size_t current_memory;
    size_t cache_misses;
    size_t page_faults;
    double fragmentation_ratio;
    memory_access_pattern_t access_pattern;
} memory_stats_t;

//
// Platform-specific memory manager interface
//

// Forward declaration of opaque memory manager type
typedef struct MemoryManager MemoryManager;

/**
 * @brief Initialize the platform-specific memory manager
 * @return Pointer to initialized memory manager or NULL on failure
 */
MemoryManager* init_memory_manager(void);

/**
 * @brief Allocate memory with platform-specific optimizations
 * @param manager Memory manager instance
 * @param size Size in bytes to allocate
 * @return Pointer to allocated memory or NULL on failure
 */
void* optimized_malloc(MemoryManager* manager, size_t size);

/**
 * @brief Free memory allocated by optimized_malloc
 * @param manager Memory manager instance
 * @param ptr Pointer to memory to free
 */
void optimized_free(MemoryManager* manager, void* ptr);

/**
 * @brief Get memory statistics
 * @param manager Memory manager instance
 * @return Memory statistics structure
 */
const memory_stats_t* get_memory_stats(const MemoryManager* manager);

/**
 * @brief Clean up memory manager and free all resources
 * @param manager Memory manager instance
 */
void cleanup_memory_manager(MemoryManager* manager);

//
// Legacy memory optimization system
//

// Memory optimization configuration
typedef struct {
    bool enable_prefetch;
    bool enable_compression;
    bool enable_numa_awareness;
    size_t prefetch_distance;
    size_t cache_line_size;
    size_t page_size;
    size_t numa_node_count;
} memory_optimization_config_t;

// Memory region descriptor
typedef struct {
    void* base_address;
    size_t size;
    memory_access_pattern_t access_pattern;
    memory_strategy_t strategy;
    memory_stats_t stats;
    bool is_optimized;
} memory_region_t;

// Initialize memory optimization system
qgt_error_t init_memory_optimization(const memory_optimization_config_t* config);

// Cleanup memory optimization system
void cleanup_memory_optimization(void);

// Register memory region for optimization
qgt_error_t register_memory_region(memory_region_t* region, void* base, size_t size);

// Analyze memory access patterns
qgt_error_t analyze_memory_pattern(memory_region_t* region);

// Apply optimization strategy
qgt_error_t optimize_memory_region(memory_region_t* region, memory_strategy_t strategy);

// Update memory statistics
qgt_error_t update_memory_stats(memory_region_t* region);

// Get memory statistics for a region
const memory_stats_t* get_region_stats(const memory_region_t* region);

// Memory prefetching functions
qgt_error_t prefetch_memory(const memory_region_t* region, size_t offset, size_t size);
qgt_error_t configure_prefetch(size_t distance, size_t stride);

// NUMA optimization functions
qgt_error_t bind_to_numa_node(const memory_region_t* region, int node);
qgt_error_t get_optimal_numa_node(const memory_region_t* region);

// Cache optimization functions
qgt_error_t align_to_cache_line(void** ptr);
qgt_error_t optimize_cache_layout(memory_region_t* region);

// Memory pool integration
qgt_error_t optimize_pool_allocation(MemoryPool* pool);
qgt_error_t optimize_pool_fragmentation(MemoryPool* pool);

// Memory compression functions
qgt_error_t compress_memory_region(memory_region_t* region);
qgt_error_t decompress_memory_region(memory_region_t* region);

// Memory access optimization
qgt_error_t optimize_access_pattern(memory_region_t* region);
qgt_error_t reorder_memory_layout(memory_region_t* region);

// Memory defragmentation
qgt_error_t defragment_memory(memory_region_t* region);
qgt_error_t compact_memory(memory_region_t* region);

// Memory monitoring functions
qgt_error_t start_memory_monitoring(memory_region_t* region);
qgt_error_t stop_memory_monitoring(memory_region_t* region);
qgt_error_t get_monitoring_results(const memory_region_t* region, memory_stats_t* stats);

// Memory optimization suggestions
typedef struct {
    memory_strategy_t recommended_strategy;
    double expected_improvement;
    char description[256];
} optimization_suggestion_t;

qgt_error_t get_optimization_suggestions(const memory_region_t* region, 
                                       optimization_suggestion_t* suggestions,
                                       size_t* count);

// Memory validation functions
qgt_error_t validate_memory_optimization(const memory_region_t* region);
qgt_error_t verify_memory_integrity(const memory_region_t* region);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_GEOMETRIC_MEMORY_OPTIMIZATION_H
