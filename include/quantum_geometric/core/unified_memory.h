#ifndef UNIFIED_MEMORY_H
#define UNIFIED_MEMORY_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Memory statistics structure
typedef struct {
    size_t total_allocated;      // Current total allocated memory
    size_t peak_allocated;       // Peak allocated memory
    size_t num_allocations;      // Total number of allocations
    size_t num_frees;           // Total number of frees
    size_t largest_allocation;   // Size of largest single allocation
    size_t fragmentation;       // Current fragmentation percentage
    size_t cache_hits;          // Number of cache hits
    size_t cache_misses;        // Number of cache misses
} memory_stats_t;

// Memory allocation flags
typedef enum {
    MEM_FLAG_NONE = 0,
    MEM_FLAG_GEOMETRIC = 1 << 0,    // Use geometric layout
    MEM_FLAG_HIERARCHICAL = 1 << 1,  // Use hierarchical compression
    MEM_FLAG_STREAMING = 1 << 2,     // Enable streaming access
    MEM_FLAG_PINNED = 1 << 3,        // Pin memory (for GPU/quantum)
    MEM_FLAG_ZERO_COPY = 1 << 4,     // Enable zero-copy access
    MEM_FLAG_PREFETCH = 1 << 5,      // Enable prefetching
    MEM_FLAG_CACHE = 1 << 6,         // Enable caching
    MEM_FLAG_TRACK = 1 << 7          // Enable tracking
} memory_flags_t;

// Memory allocation properties
typedef struct {
    size_t size;               // Allocation size
    size_t alignment;          // Required alignment
    memory_flags_t flags;      // Allocation flags
    double tolerance;          // Compression tolerance
    size_t chunk_size;         // Streaming chunk size
    size_t* dimensions;        // Geometric dimensions
    size_t num_dimensions;     // Number of dimensions
} memory_properties_t;

// Forward declarations
typedef struct unified_memory_interface unified_memory_interface_t;
typedef struct memory_allocator memory_allocator_t;

// Memory allocator interface
struct memory_allocator {
    // Core allocation functions
    void* (*allocate)(memory_allocator_t* allocator, const memory_properties_t* props);
    void (*free)(memory_allocator_t* allocator, void* ptr);
    void* (*realloc)(memory_allocator_t* allocator, void* ptr, size_t size);
    
    // Memory tracking
    bool (*track)(memory_allocator_t* allocator, void* ptr, size_t size);
    bool (*untrack)(memory_allocator_t* allocator, void* ptr);
    
    // Statistics
    void (*get_stats)(memory_allocator_t* allocator, memory_stats_t* stats);
    void (*reset_stats)(memory_allocator_t* allocator);
    
    // Implementation-specific data
    void* impl;
};

// Unified memory interface
struct unified_memory_interface {
    // Memory allocator
    memory_allocator_t* allocator;
    
    // Helper functions for common allocation patterns
    void* (*allocate_tensor)(unified_memory_interface_t* memory, 
                            size_t* dimensions, 
                            size_t num_dimensions);
    
    void* (*allocate_hierarchical)(unified_memory_interface_t* memory,
                                  size_t size,
                                  double tolerance);
    
    void* (*allocate_streaming)(unified_memory_interface_t* memory,
                               size_t total_size,
                               size_t chunk_size);
    
    // Memory management
    bool (*can_allocate)(unified_memory_interface_t* memory, size_t size);
    void (*defragment)(unified_memory_interface_t* memory);
    void (*trim)(unified_memory_interface_t* memory);
    
    // Implementation-specific data
    void* impl;
};

// Global interface functions
unified_memory_interface_t* get_global_memory_interface(void);
void register_memory_interface(unified_memory_interface_t* interface);
void unregister_memory_interface(unified_memory_interface_t* interface);

// Helper functions
void* unified_malloc(size_t size);
void unified_free(void* ptr);
void* unified_realloc(void* ptr, size_t size);

// Advanced allocation helpers
void* unified_allocate_tensor(size_t* dimensions, size_t num_dimensions);
void* unified_allocate_hierarchical(size_t size, double tolerance);
void* unified_allocate_streaming(size_t total_size, size_t chunk_size);

// Statistics functions
void get_memory_statistics(memory_stats_t* stats);
void reset_memory_statistics(void);

#ifdef __cplusplus
}
#endif

#endif // UNIFIED_MEMORY_H
