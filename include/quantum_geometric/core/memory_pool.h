#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include <pthread.h>
#include <stdatomic.h>

// Constants
#define QG_POOL_ALIGNMENT 64   // Cache line alignment
#define QG_NUM_SIZE_CLASSES 64  // More size classes for better granularity
#define QG_MIN_BLOCK_SIZE 128  // Larger minimum block for quantum tensors
#define QG_MAX_BLOCK_SIZE (4 * 1024 * 1024) // 4MB max block
#define QG_MAX_THREAD_CACHE 1024 // Max thread cache entries

#include "quantum_geometric/core/quantum_geometric_types.h"

// Block header (cache aligned)
typedef struct Block {
    size_t size;           // Block size
    uint32_t magic;        // Magic number for validation
    bool is_free;          // Free flag
    uint16_t size_class;   // Size class index
    struct Block* next;    // Next block
    struct Block* prev;    // Previous block
    char padding[40];      // Padding for 64-byte alignment
    void* data;           // Actual data follows
} __attribute__((aligned(64))) Block;

// Thread cache entry
typedef struct ThreadCacheEntry {
    void* ptr;           // Block pointer
    size_t size_class;   // Size class index
    struct ThreadCacheEntry* next;
} ThreadCacheEntry;

// Thread local cache
typedef struct ThreadCache {
    ThreadCacheEntry* entries[QG_NUM_SIZE_CLASSES];
    size_t count[QG_NUM_SIZE_CLASSES];
} ThreadCache;

// Size class
typedef struct SizeClass {
    size_t block_size;      // Size of blocks in this class
    Block* free_list;       // List of free blocks
    pthread_mutex_t mutex;  // Per-class mutex
    size_t num_blocks;      // Current number of blocks
    size_t max_blocks;      // Maximum blocks allowed
    size_t hits;           // Cache hit counter
    size_t misses;         // Cache miss counter
} SizeClass;

// Memory pool
typedef struct MemoryPool {
    // Base memory
    void* base_address;     // Base address of pool memory
    size_t total_size;      // Total size of pool
    
    // Size classes
    SizeClass* size_classes;
    size_t num_classes;
    
    // Large allocations
    Block* large_blocks;
    
    // Statistics (atomic)
    _Atomic(size_t) total_allocated;
    _Atomic(size_t) peak_allocated;
    _Atomic(size_t) num_allocations;
    
    // Thread safety
    pthread_mutex_t mutex;
    
    // Configuration
    struct PoolConfig config;
} MemoryPool;

// Memory pool functions
MemoryPool* init_memory_pool(const struct PoolConfig* config);
void* pool_malloc(MemoryPool* pool, size_t size);
void pool_free(MemoryPool* pool, void* ptr);
void cleanup_memory_pool(MemoryPool* pool);

// Statistics functions
size_t get_total_allocated(const MemoryPool* pool);
size_t get_peak_allocated(const MemoryPool* pool);
size_t get_num_allocations(const MemoryPool* pool);

#endif // MEMORY_POOL_H
