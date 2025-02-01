/**
 * @file memory_optimization_impl.h
 * @brief Platform-specific memory optimization interface
 */

#ifndef QUANTUM_GEOMETRIC_MEMORY_OPTIMIZATION_IMPL_H
#define QUANTUM_GEOMETRIC_MEMORY_OPTIMIZATION_IMPL_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include "quantum_geometric/hardware/hardware_capabilities.h"
#include "quantum_geometric/hardware/quantum_hardware_types.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/memory_optimization.h"
#include "quantum_geometric/core/memory_pool.h"
#include "quantum_geometric/core/performance_monitor.h"

// Forward declarations
struct MemoryBlock;
struct MemoryPool;
struct MemoryManager;

// Use existing types from memory_pool.h
typedef struct MemoryManager MemoryManager;

// Core memory management interface
MemoryManager* init_memory_manager(void);
void cleanup_memory_manager(MemoryManager* manager);
void* optimized_malloc(MemoryManager* manager, size_t size);
void optimized_free(MemoryManager* manager, void* ptr);
const memory_stats_t* get_memory_stats(const MemoryManager* manager);

// Platform-specific memory optimization interface
bool has_numa_support(void);
int get_numa_node_count(void);
bool bind_memory_to_node(void* ptr, size_t size, int node);
bool get_memory_node(const void* ptr, int* node);

// Platform-specific cache optimization interface
size_t get_cache_line_size(CacheLevel level);
bool align_memory_to_cache(void** ptr);
bool prefetch_memory_range(const void* ptr, size_t size);

// Platform-specific huge page support
bool has_huge_page_support(void);
size_t get_huge_page_size(void);
void* allocate_huge_pages(size_t size);
void free_huge_pages(void* ptr, size_t size);

// Performance counter types
typedef enum {
    COUNTER_PAGE_FAULTS,
    COUNTER_CACHE_MISSES,
    COUNTER_TLB_MISSES,
    COUNTER_CPU_CYCLES,
    COUNTER_BRANCH_MISSES,
    COUNTER_CACHE_REFS
} performance_counter_t;

// Platform-specific performance monitoring
uint64_t get_performance_counter(performance_counter_t counter);

// Platform-specific memory pool operations
bool init_memory_pool_impl(MemoryPool* pool, const PoolConfig* config);
void cleanup_memory_pool(MemoryPool* pool);
bool optimize_pool(MemoryPool* pool);
bool defragment_pool(MemoryPool* pool);

// Platform-specific memory protection
bool protect_memory_range(void* ptr, size_t size, bool readonly);
bool verify_memory_protection(const void* ptr, size_t size);

// Platform-specific memory compression
bool compress_memory(void* ptr, size_t size, size_t* compressed_size);
bool decompress_memory(void* compressed_ptr, size_t compressed_size, 
                      void* output_ptr, size_t* output_size);

// Performance monitoring (must be after performance_monitor.h include)
void update_memory_performance_metrics(PerformanceMetrics* metrics);

#endif // QUANTUM_GEOMETRIC_MEMORY_OPTIMIZATION_IMPL_H
