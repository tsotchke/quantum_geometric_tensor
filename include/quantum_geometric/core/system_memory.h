/**
 * @file system_memory.h
 * @brief System-specific memory management includes and definitions
 */

#ifndef QUANTUM_GEOMETRIC_SYSTEM_MEMORY_H
#define QUANTUM_GEOMETRIC_SYSTEM_MEMORY_H

#include <stddef.h>

#ifdef __linux__
#include <numa.h>
#define HAVE_NUMA 1
#else
#define HAVE_NUMA 0
#endif

#include <sys/mman.h>

// Define container_of macro for both platforms
#ifndef container_of
#define container_of(ptr, type, member) \
    ((type *)((char *)(ptr) - offsetof(type, member)))
#endif

#ifdef __APPLE__
// macOS doesn't have MAP_HUGETLB or MAP_ALIGNED_SUPER
#define MAP_HUGETLB 0
#define MAP_ALIGNED_SUPER MAP_PRIVATE

// macOS doesn't have MADV_HUGEPAGE
#ifndef MADV_HUGEPAGE
#define MADV_HUGEPAGE 0
#endif
#endif

// Common memory-related constants
#ifndef PAGE_SIZE
#define PAGE_SIZE 4096
#endif
#define CACHE_LINE_SIZE 64
#define HUGE_PAGE_SIZE (2 * 1024 * 1024) // 2MB huge pages

// Memory alignment requirements
#define MEMORY_ALIGNMENT 64
#define MEMORY_GUARD_SIZE 64

// Memory pool types
typedef enum {
    POOL_SMALL,    // < 4KB
    POOL_MEDIUM,   // 4KB - 1MB
    POOL_LARGE,    // 1MB - 64MB
    POOL_HUGE      // > 64MB
} PoolType;

// Memory statistics structure
typedef struct {
    size_t total_allocated;
    size_t peak_allocated;
    size_t page_faults;
    size_t cache_misses;
    size_t tlb_misses;
    double efficiency;
    struct {
        size_t allocations;
        size_t deallocations;
        size_t fragmentation;
        size_t peak_usage;
        double utilization;
    } pool_metrics[16]; // MAX_MEMORY_POOLS = 16
} MemoryStats;

#endif // QUANTUM_GEOMETRIC_SYSTEM_MEMORY_H
