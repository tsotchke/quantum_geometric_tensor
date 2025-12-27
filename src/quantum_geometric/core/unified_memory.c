#include "quantum_geometric/core/unified_memory.h"
#include "quantum_geometric/core/advanced_memory_system.h"
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdio.h>

// Platform-specific includes for memory trimming
#if defined(__linux__)
#include <malloc.h>
#elif defined(__APPLE__)
#include <sys/mman.h>
#endif

// Internal logging macro (silent by default, enable for debugging)
#ifdef UNIFIED_MEMORY_DEBUG
#define unified_log(fmt, ...) fprintf(stderr, "[unified_memory] " fmt "\n", ##__VA_ARGS__)
#else
#define unified_log(fmt, ...) ((void)0)
#endif

// Replace geometric_log_error with our internal logging
#define geometric_log_error unified_log

// Memory pool constants (matching memory_pool.h for compatibility)
#ifndef QG_POOL_ALIGNMENT
#define QG_POOL_ALIGNMENT 64
#endif
#ifndef QG_MIN_BLOCK_SIZE
#define QG_MIN_BLOCK_SIZE 128
#endif

// Global interface instance
static unified_memory_interface_t* g_memory_interface = NULL;
static pthread_mutex_t g_interface_mutex = PTHREAD_MUTEX_INITIALIZER;
static int g_ref_count = 0;

// Implementation data structure
typedef struct {
    memory_allocator_t allocator;
    void* pool;  // Changed to void* to match advanced_memory_system.h
    advanced_memory_system_t* advanced;
    memory_stats_t stats;
    pthread_mutex_t mutex;
} unified_memory_impl_t;

// Forward declarations
static void* unified_allocate_impl(memory_allocator_t* allocator, const memory_properties_t* props);
static void unified_free_impl(memory_allocator_t* allocator, void* ptr);
static void* unified_realloc_impl(memory_allocator_t* allocator, void* ptr, size_t size);
static bool unified_track(memory_allocator_t* allocator, void* ptr, size_t size);
static bool unified_untrack(memory_allocator_t* allocator, void* ptr);
static void unified_get_stats(memory_allocator_t* allocator, memory_stats_t* stats);
static void unified_reset_stats(memory_allocator_t* allocator);

// Helper functions for tensor allocation
static void* unified_allocate_tensor_impl(unified_memory_interface_t* memory,
                                        size_t* dimensions,
                                        size_t num_dimensions);

static void* unified_allocate_hierarchical_impl(unified_memory_interface_t* memory,
                                              size_t size,
                                              double tolerance);

static void* unified_allocate_streaming_impl(unified_memory_interface_t* memory,
                                           size_t total_size,
                                           size_t chunk_size);

// Memory management helper functions
static bool unified_can_allocate_impl(unified_memory_interface_t* memory, size_t size);
static void unified_defragment_impl(unified_memory_interface_t* memory);
static void unified_trim_impl(unified_memory_interface_t* memory);

// Create unified memory implementation
static unified_memory_impl_t* create_unified_memory_impl(void) {
    unified_memory_impl_t* impl = malloc(sizeof(unified_memory_impl_t));
    if (!impl) {
        geometric_log_error("Failed to allocate unified memory implementation");
        return NULL;
    }
    
    // Initialize mutex with error handling
    int mutex_result = pthread_mutex_init(&impl->mutex, NULL);
    if (mutex_result != 0) {
        geometric_log_error("Failed to initialize mutex: error code %d", mutex_result);
        free(impl);
        return NULL;
    }
    
    // Initialize advanced memory system first
    memory_system_config_t sys_config = {
        .type = MEM_SYSTEM_QUANTUM,
        .strategy = ALLOC_STRATEGY_BUDDY,
        .optimization = MEM_OPT_ADVANCED,
        .alignment = QG_POOL_ALIGNMENT,
        .enable_monitoring = true,
        .enable_defragmentation = true,
        .total_size = 1024 * 1024 * 1024,  // 1GB total size
        .block_size = QG_MIN_BLOCK_SIZE
    };
    
    impl->advanced = create_memory_system(&sys_config);
    if (!impl->advanced) {
        geometric_log_error("Failed to create advanced memory system");
        pthread_mutex_destroy(&impl->mutex);
        free(impl);
        return NULL;
    }

    // Initialize memory pool
    pool_config_t pool_config = {
        .pool_size = 1024 * 1024 * 1024,  // 1GB pool
        .block_size = QG_MIN_BLOCK_SIZE,
        .max_blocks = 1024 * 1024,
        .fixed_size = false,
        .thread_safe = true,
        .enable_growth = true,
        .enable_stats = true
    };
    
    impl->pool = ams_create_memory_pool(impl->advanced, &pool_config);
    if (!impl->pool) {
        geometric_log_error("Failed to create memory pool");
        destroy_memory_system(impl->advanced);
        pthread_mutex_destroy(&impl->mutex);
        free(impl);
        return NULL;
    }
    
    // Initialize allocator interface
    impl->allocator.allocate = unified_allocate_impl;
    impl->allocator.free = unified_free_impl;
    impl->allocator.realloc = unified_realloc_impl;
    impl->allocator.track = unified_track;
    impl->allocator.untrack = unified_untrack;
    impl->allocator.get_stats = unified_get_stats;
    impl->allocator.reset_stats = unified_reset_stats;
    impl->allocator.impl = impl;
    
    // Initialize statistics
    memset(&impl->stats, 0, sizeof(memory_stats_t));
    
    return impl;
}

// Destroy unified memory implementation
static void destroy_unified_memory_impl(unified_memory_impl_t* impl) {
    if (!impl) {
        geometric_log_error("Attempted to destroy NULL memory implementation");
        return;
    }
    
    if (pthread_mutex_lock(&impl->mutex) != 0) {
        geometric_log_error("Failed to lock mutex in destroy_impl");
        return;
    }
    
    if (impl->pool) {
        ams_destroy_memory_pool(impl->advanced, impl->pool);
    }
    
    if (impl->advanced) {
        destroy_memory_system(impl->advanced);
    }
    
    if (pthread_mutex_unlock(&impl->mutex) != 0) {
        geometric_log_error("Failed to unlock mutex in destroy_impl");
    }
    pthread_mutex_destroy(&impl->mutex);
    
    free(impl);
}

// Create unified memory interface
static unified_memory_interface_t* create_unified_memory_interface(void) {
    unified_memory_interface_t* interface = malloc(sizeof(unified_memory_interface_t));
    if (!interface) {
        geometric_log_error("Failed to allocate unified memory interface");
        return NULL;
    }
    
    // Create implementation
    unified_memory_impl_t* impl = create_unified_memory_impl();
    if (!impl) {
        geometric_log_error("Failed to create unified memory implementation");
        free(interface);
        return NULL;
    }
    
    // Initialize interface
    interface->allocator = &impl->allocator;
    interface->allocate_tensor = unified_allocate_tensor_impl;
    interface->allocate_hierarchical = unified_allocate_hierarchical_impl;
    interface->allocate_streaming = unified_allocate_streaming_impl;
    interface->can_allocate = unified_can_allocate_impl;
    interface->defragment = unified_defragment_impl;
    interface->trim = unified_trim_impl;
    interface->impl = impl;
    
    return interface;
}

// Destroy unified memory interface
static void destroy_unified_memory_interface(unified_memory_interface_t* interface) {
    if (!interface) return;
    
    if (interface->impl) {
        destroy_unified_memory_impl(interface->impl);
    }
    
    free(interface);
}

// Global interface functions
unified_memory_interface_t* get_global_memory_interface(void) {
    if (pthread_mutex_lock(&g_interface_mutex) != 0) {
        geometric_log_error("Failed to lock global interface mutex");
        return NULL;
    }
    
    if (!g_memory_interface) {
        g_memory_interface = create_unified_memory_interface();
        if (!g_memory_interface) {
            geometric_log_error("Failed to create global memory interface");
            if (pthread_mutex_unlock(&g_interface_mutex) != 0) {
        geometric_log_error("Failed to unlock global interface mutex");
    }
            return NULL;
        }
        g_ref_count = 1;
    } else {
        g_ref_count++;  // Increment reference count for existing interface
    }
    
    pthread_mutex_unlock(&g_interface_mutex);
    return g_memory_interface;
}

void register_memory_interface(unified_memory_interface_t* interface) {
    if (!interface) {
        geometric_log_error("Attempted to register NULL memory interface");
        return;
    }
    
    if (pthread_mutex_lock(&g_interface_mutex) != 0) {
        geometric_log_error("Failed to lock global interface mutex in register");
        return;
    }
    
    if (g_memory_interface == interface) {
        g_ref_count++;
        geometric_log_error("Memory interface reference count increased to %d", g_ref_count);
    } else {
        geometric_log_error("Attempted to register different memory interface instance");
    }
    
    if (pthread_mutex_unlock(&g_interface_mutex) != 0) {
        geometric_log_error("Failed to unlock global interface mutex in register");
    }
}

void unregister_memory_interface(unified_memory_interface_t* interface) {
    if (!interface) {
        geometric_log_error("Attempted to unregister NULL memory interface");
        return;
    }
    
    if (pthread_mutex_lock(&g_interface_mutex) != 0) {
        geometric_log_error("Failed to lock global interface mutex in unregister");
        return;
    }
    
    if (g_memory_interface == interface) {
        g_ref_count--;
        geometric_log_error("Memory interface reference count decreased to %d", g_ref_count);
        
        if (g_ref_count <= 0) {
            geometric_log_error("Destroying global memory interface");
            destroy_unified_memory_interface(g_memory_interface);
            g_memory_interface = NULL;
            g_ref_count = 0;
        }
    } else {
        geometric_log_error("Attempted to unregister different memory interface instance");
    }
    
    if (pthread_mutex_unlock(&g_interface_mutex) != 0) {
        geometric_log_error("Failed to unlock global interface mutex in unregister");
    }
}

// Core allocation functions
static void* unified_allocate_impl(memory_allocator_t* allocator, const memory_properties_t* props) {
    if (!allocator || !props) return NULL;
    
    unified_memory_impl_t* impl = allocator->impl;
    void* ptr = NULL;
    
    if (pthread_mutex_lock(&impl->mutex) != 0) {
        geometric_log_error("Failed to lock mutex in allocate_impl");
        return NULL;
    }
    
    // Use advanced memory system for geometric/hierarchical allocations
    if (props->flags & (MEM_FLAG_GEOMETRIC | MEM_FLAG_HIERARCHICAL)) {
        if (props->flags & MEM_FLAG_GEOMETRIC) {
            // Create geometric layout
            size_t total_size = props->size;
            for (size_t i = 0; i < props->num_dimensions; i++) {
                total_size *= props->dimensions[i];
            }
            ptr = memory_allocate(impl->advanced, total_size, props->alignment);
            if (!ptr) {
                geometric_log_error("Failed to allocate geometric memory of size %zu", total_size);
            }
        } else {
            // Use hierarchical allocation
            ptr = memory_allocate(impl->advanced, props->size, props->alignment);
            if (!ptr) {
                geometric_log_error("Failed to allocate hierarchical memory of size %zu", props->size);
            }
        }
    } else {
        // Use memory pool for standard allocations
        ptr = ams_pool_allocate(impl->advanced, impl->pool, props->size);
        if (!ptr) {
            geometric_log_error("Failed to allocate memory from pool of size %zu", props->size);
        }
    }
    
    if (ptr && props->flags & MEM_FLAG_TRACK) {
        unified_track(allocator, ptr, props->size);
    }
    
    if (pthread_mutex_unlock(&impl->mutex) != 0) {
        geometric_log_error("Failed to unlock mutex in allocate_impl");
    }
    return ptr;
}

static void unified_free_impl(memory_allocator_t* allocator, void* ptr) {
    if (!allocator || !ptr) return;
    
    unified_memory_impl_t* impl = allocator->impl;
    
    if (pthread_mutex_lock(&impl->mutex) != 0) {
        geometric_log_error("Failed to lock mutex in free_impl");
        return;
    }
    
    // Try pool free first
    ams_pool_free(impl->advanced, impl->pool, ptr);
    
    // If not in pool, try advanced memory system
    memory_free(impl->advanced, ptr);
    
    unified_untrack(allocator, ptr);
    
    if (pthread_mutex_unlock(&impl->mutex) != 0) {
        geometric_log_error("Failed to unlock mutex in free_impl");
    }
}

static void* unified_realloc_impl(memory_allocator_t* allocator, void* ptr, size_t size) {
    if (!allocator) return NULL;
    if (!ptr) return unified_allocate_impl(allocator, &(memory_properties_t){.size = size});
    if (size == 0) {
        unified_free_impl(allocator, ptr);
        return NULL;
    }
    
    unified_memory_impl_t* impl = allocator->impl;
    void* new_ptr = NULL;
    size_t old_size = 0;
    
    if (pthread_mutex_lock(&impl->mutex) != 0) {
        geometric_log_error("Failed to lock mutex in realloc_impl");
        return NULL;
    }
    
    // Get current allocation size
    old_size = get_allocation_size(impl->advanced, ptr);
    if (old_size == 0) {
        geometric_log_error("Failed to get allocation size for realloc");
        pthread_mutex_unlock(&impl->mutex);
        return NULL;
    }
    
    // Allocate new memory with same properties
    memory_properties_t props = {
        .size = size,
        .alignment = sizeof(void*),
        .flags = MEM_FLAG_TRACK
    };
    
    new_ptr = unified_allocate_impl(allocator, &props);
    if (!new_ptr) {
        pthread_mutex_unlock(&impl->mutex);
        return NULL;
    }
    
    // Copy data and handle old allocation
    memcpy(new_ptr, ptr, old_size < size ? old_size : size);
    unified_free_impl(allocator, ptr);
    
    if (pthread_mutex_unlock(&impl->mutex) != 0) {
        geometric_log_error("Failed to unlock mutex in realloc_impl");
    }
    return new_ptr;
}

// Memory tracking functions
static bool unified_track(memory_allocator_t* allocator, void* ptr, size_t size) {
    if (!allocator || !ptr) {
        geometric_log_error("Invalid parameters for memory tracking");
        return false;
    }
    
    unified_memory_impl_t* impl = allocator->impl;
    if (!impl) {
        geometric_log_error("Invalid memory implementation for tracking");
        return false;
    }
    
    // Note: No need for mutex here since this is called within functions that already hold the mutex
    impl->stats.total_allocated += size;
    if (impl->stats.total_allocated > impl->stats.peak_allocated) {
        impl->stats.peak_allocated = impl->stats.total_allocated;
    }
    impl->stats.num_allocations++;
    if (size > impl->stats.largest_allocation) {
        impl->stats.largest_allocation = size;
    }
    
    return true;
}

static bool unified_untrack(memory_allocator_t* allocator, void* ptr) {
    if (!allocator || !ptr) {
        geometric_log_error("Invalid parameters for memory untracking");
        return false;
    }
    
    unified_memory_impl_t* impl = allocator->impl;
    if (!impl) {
        geometric_log_error("Invalid memory implementation for untracking");
        return false;
    }
    
    // Note: No need for mutex here since this is called within functions that already hold the mutex
    impl->stats.num_frees++;
    
    return true;
}

// Statistics functions
static void unified_get_stats(memory_allocator_t* allocator, memory_stats_t* stats) {
    if (!allocator || !stats) {
        geometric_log_error("Invalid parameters for get_stats");
        return;
    }
    
    unified_memory_impl_t* impl = allocator->impl;
    if (!impl) {
        geometric_log_error("Invalid memory implementation for get_stats");
        return;
    }
    
    if (pthread_mutex_lock(&impl->mutex) != 0) {
        geometric_log_error("Failed to lock mutex in get_stats");
        return;
    }
    
    memcpy(stats, &impl->stats, sizeof(memory_stats_t));
    
    if (pthread_mutex_unlock(&impl->mutex) != 0) {
        geometric_log_error("Failed to unlock mutex in get_stats");
    }
}

static void unified_reset_stats(memory_allocator_t* allocator) {
    if (!allocator) {
        geometric_log_error("Invalid allocator for reset_stats");
        return;
    }
    
    unified_memory_impl_t* impl = allocator->impl;
    if (!impl) {
        geometric_log_error("Invalid memory implementation for reset_stats");
        return;
    }
    
    if (pthread_mutex_lock(&impl->mutex) != 0) {
        geometric_log_error("Failed to lock mutex in reset_stats");
        return;
    }
    
    memset(&impl->stats, 0, sizeof(memory_stats_t));
    
    if (pthread_mutex_unlock(&impl->mutex) != 0) {
        geometric_log_error("Failed to unlock mutex in reset_stats");
    }
}

// Helper functions implementation
static void* unified_allocate_tensor_impl(unified_memory_interface_t* memory,
                                        size_t* dimensions,
                                        size_t num_dimensions) {
    if (!memory || !dimensions || num_dimensions == 0) return NULL;
    
    memory_properties_t props = {
        .flags = MEM_FLAG_GEOMETRIC | MEM_FLAG_TRACK,
        .dimensions = dimensions,
        .num_dimensions = num_dimensions,
        .alignment = 64
    };
    
    // Calculate total size
    size_t total_size = 1;
    for (size_t i = 0; i < num_dimensions; i++) {
        total_size *= dimensions[i];
    }
    props.size = total_size;
    
    void* ptr = memory->allocator->allocate(memory->allocator, &props);
    if (!ptr) {
        geometric_log_error("Failed to allocate tensor memory of size %zu with %zu dimensions", total_size, num_dimensions);
    }
    return ptr;
}

static void* unified_allocate_hierarchical_impl(unified_memory_interface_t* memory,
                                              size_t size,
                                              double tolerance) {
    if (!memory || size == 0) return NULL;
    
    memory_properties_t props = {
        .size = size,
        .alignment = 64,
        .flags = MEM_FLAG_HIERARCHICAL | MEM_FLAG_TRACK,
        .tolerance = tolerance
    };
    
    void* ptr = memory->allocator->allocate(memory->allocator, &props);
    if (!ptr) {
        geometric_log_error("Failed to allocate hierarchical memory of size %zu with tolerance %f", size, tolerance);
    }
    return ptr;
}

static void* unified_allocate_streaming_impl(unified_memory_interface_t* memory,
                                           size_t total_size,
                                           size_t chunk_size) {
    if (!memory || total_size == 0 || chunk_size == 0) return NULL;

    memory_properties_t props = {
        .size = total_size,
        .alignment = 64,
        .flags = MEM_FLAG_STREAMING | MEM_FLAG_TRACK,
        .chunk_size = chunk_size
    };

    void* ptr = memory->allocator->allocate(memory->allocator, &props);
    if (!ptr) {
        geometric_log_error("Failed to allocate streaming memory of size %zu with chunk size %zu", total_size, chunk_size);
    }
    return ptr;
}

// ============================================================================
// Memory Management Functions Implementation
// ============================================================================

// Check if memory of a given size can be allocated
static bool unified_can_allocate_impl(unified_memory_interface_t* memory, size_t size) {
    if (!memory || size == 0) {
        return false;
    }

    unified_memory_impl_t* impl = (unified_memory_impl_t*)memory->impl;
    if (!impl || !impl->advanced) {
        geometric_log_error("Invalid memory implementation for can_allocate");
        return false;
    }

    if (pthread_mutex_lock(&impl->mutex) != 0) {
        geometric_log_error("Failed to lock mutex in can_allocate");
        return false;
    }

    // Get current memory metrics
    memory_metrics_t metrics;
    bool result = false;

    if (get_memory_metrics(impl->advanced, &metrics)) {
        // Calculate available memory considering fragmentation
        // Use the configured total size (1GB) minus current usage
        size_t total_size = 1024 * 1024 * 1024;  // 1GB - same as config
        size_t available = total_size - metrics.current_usage;

        // Account for fragmentation - if fragmentation is high, we may not
        // be able to allocate a contiguous block even if total free is enough
        double fragmentation = metrics.fragmentation;

        // Fragmentation penalty: if fragmentation is 50%, effective available is halved
        size_t effective_available = (size_t)(available * (1.0 - fragmentation));

        // Add some margin for allocation overhead (alignment, headers, etc.)
        size_t required_with_overhead = size + (size / 10) + 64;

        result = (effective_available >= required_with_overhead);

        if (!result) {
            geometric_log_error("Cannot allocate %zu bytes: available=%zu, effective=%zu, fragmentation=%.2f",
                              size, available, effective_available, fragmentation);
        }
    } else {
        geometric_log_error("Failed to get memory metrics for can_allocate");
    }

    if (pthread_mutex_unlock(&impl->mutex) != 0) {
        geometric_log_error("Failed to unlock mutex in can_allocate");
    }

    return result;
}

// Defragment memory to reduce fragmentation
static void unified_defragment_impl(unified_memory_interface_t* memory) {
    if (!memory) {
        geometric_log_error("Invalid memory interface for defragment");
        return;
    }

    unified_memory_impl_t* impl = (unified_memory_impl_t*)memory->impl;
    if (!impl || !impl->advanced) {
        geometric_log_error("Invalid memory implementation for defragment");
        return;
    }

    if (pthread_mutex_lock(&impl->mutex) != 0) {
        geometric_log_error("Failed to lock mutex in defragment");
        return;
    }

    // Check if defragmentation is needed
    double fragmentation_level = 0.0;
    if (!is_defragmentation_needed(impl->advanced, &fragmentation_level)) {
        geometric_log_error("Defragmentation check: level=%.2f, not needed", fragmentation_level);
        pthread_mutex_unlock(&impl->mutex);
        return;
    }

    geometric_log_error("Starting defragmentation: fragmentation level=%.2f", fragmentation_level);

    // Configure defragmentation
    defrag_config_t defrag_config = {
        .threshold = 0.3,           // Defragment when fragmentation > 30%
        .max_moves = 1000,          // Maximum number of block moves
        .compact_pools = true,      // Compact memory pools
        .preserve_order = false,    // Allow reordering for better compaction
        .incremental = true,        // Use incremental defragmentation
        .batch_size = 100           // Process 100 blocks per batch
    };

    // Start defragmentation
    if (!start_defragmentation(impl->advanced, &defrag_config)) {
        geometric_log_error("Failed to start defragmentation");
        pthread_mutex_unlock(&impl->mutex);
        return;
    }

    // Defragmentation runs asynchronously or synchronously depending on implementation
    // The stop_defragmentation can be called to wait for completion or cancel

    // Check fragmentation level after
    double new_fragmentation = 0.0;
    if (analyze_fragmentation(impl->advanced, &new_fragmentation)) {
        double reduction = fragmentation_level - new_fragmentation;
        geometric_log_error("Defragmentation complete: new level=%.2f, reduced by=%.2f",
                          new_fragmentation, reduction);

        // Update statistics
        impl->stats.fragmentation = (size_t)(new_fragmentation * 100);
    }

    if (pthread_mutex_unlock(&impl->mutex) != 0) {
        geometric_log_error("Failed to unlock mutex in defragment");
    }
}

// Trim unused memory back to the operating system
static void unified_trim_impl(unified_memory_interface_t* memory) {
    if (!memory) {
        geometric_log_error("Invalid memory interface for trim");
        return;
    }

    unified_memory_impl_t* impl = (unified_memory_impl_t*)memory->impl;
    if (!impl || !impl->advanced) {
        geometric_log_error("Invalid memory implementation for trim");
        return;
    }

    if (pthread_mutex_lock(&impl->mutex) != 0) {
        geometric_log_error("Failed to lock mutex in trim");
        return;
    }

    // Get current memory metrics before trim
    memory_metrics_t metrics_before;
    if (!get_memory_metrics(impl->advanced, &metrics_before)) {
        geometric_log_error("Failed to get memory metrics before trim");
        pthread_mutex_unlock(&impl->mutex);
        return;
    }

    size_t unused_memory = metrics_before.total_allocated - metrics_before.current_usage;
    geometric_log_error("Trim: current_usage=%zu, total_allocated=%zu, unused=%zu",
                      metrics_before.current_usage, metrics_before.total_allocated, unused_memory);

    // Calculate trim threshold - only trim if we have significant unused memory
    // Trim if unused memory is more than 10% of total allocated or more than 10MB
    size_t trim_threshold_percentage = metrics_before.total_allocated / 10;
    size_t trim_threshold_absolute = 10 * 1024 * 1024;  // 10 MB
    size_t trim_threshold = (trim_threshold_percentage > trim_threshold_absolute) ?
                           trim_threshold_percentage : trim_threshold_absolute;

    if (unused_memory < trim_threshold) {
        geometric_log_error("Trim not needed: unused=%zu < threshold=%zu", unused_memory, trim_threshold);
        pthread_mutex_unlock(&impl->mutex);
        return;
    }

    // Optimize memory usage to release unused memory
    // This triggers internal pool compaction and unused block release
    if (!optimize_memory_usage(impl->advanced, MEM_OPT_ADVANCED)) {
        geometric_log_error("Memory optimization failed during trim");
    }

    // On POSIX systems, we can use malloc_trim to return memory to OS
    // This is platform-specific
#if defined(__linux__)
    // malloc_trim(0) returns as much memory as possible to the OS
    int trimmed = malloc_trim(0);
    geometric_log_error("malloc_trim returned: %d", trimmed);
#elif defined(__APPLE__)
    // On macOS, the advanced memory system should handle this internally
    // through memory optimization. We log the operation for visibility.
    geometric_log_error("Trim on macOS: delegated to memory system optimization");
#else
    geometric_log_error("Trim: platform-specific trimming not available");
#endif

    // Get metrics after trim
    memory_metrics_t metrics_after;
    if (get_memory_metrics(impl->advanced, &metrics_after)) {
        size_t freed = metrics_before.total_allocated - metrics_after.total_allocated;
        geometric_log_error("Trim complete: freed=%zu bytes", freed);
    }

    if (pthread_mutex_unlock(&impl->mutex) != 0) {
        geometric_log_error("Failed to unlock mutex in trim");
    }
}

// Global helper functions
void* unified_malloc(size_t size) {
    unified_memory_interface_t* memory = get_global_memory_interface();
    if (!memory) {
        geometric_log_error("Failed to get global memory interface for malloc");
        return NULL;
    }
    
    memory_properties_t props = {
        .size = size,
        .alignment = sizeof(void*),
        .flags = MEM_FLAG_TRACK
    };
    
    void* ptr = memory->allocator->allocate(memory->allocator, &props);
    if (!ptr) {
        geometric_log_error("Failed to allocate memory of size %zu", size);
    }
    return ptr;
}

void unified_free(void* ptr) {
    if (!ptr) return;
    
    unified_memory_interface_t* memory = get_global_memory_interface();
    if (!memory) {
        geometric_log_error("Failed to get global memory interface for free");
        return;
    }
    
    memory->allocator->free(memory->allocator, ptr);
}

void* unified_realloc(void* ptr, size_t size) {
    unified_memory_interface_t* memory = get_global_memory_interface();
    if (!memory) return NULL;
    
    if (!ptr) {
        return unified_malloc(size);
    }
    
    if (size == 0) {
        unified_free(ptr);
        return NULL;
    }
    
    unified_memory_impl_t* impl = (unified_memory_impl_t*)memory->impl;
    void* new_ptr = NULL;
    size_t old_size = 0;
    
    if (pthread_mutex_lock(&impl->mutex) != 0) {
        geometric_log_error("Failed to lock mutex in global realloc");
        return NULL;
    }
    
    // Get current allocation size
    old_size = get_allocation_size(impl->advanced, ptr);
    if (old_size == 0) {
        geometric_log_error("Failed to get allocation size for realloc");
        pthread_mutex_unlock(&impl->mutex);
        return NULL;
    }
    
    // Allocate new memory with same properties
    memory_properties_t props = {
        .size = size,
        .alignment = sizeof(void*),
        .flags = MEM_FLAG_TRACK
    };
    
    new_ptr = memory->allocator->allocate(memory->allocator, &props);
    if (!new_ptr) {
        geometric_log_error("Failed to allocate memory for realloc of size %zu", size);
        pthread_mutex_unlock(&impl->mutex);
        return NULL;
    }
    
    // Copy data and handle old allocation
    memcpy(new_ptr, ptr, old_size < size ? old_size : size);
    memory->allocator->free(memory->allocator, ptr);
    
    if (pthread_mutex_unlock(&impl->mutex) != 0) {
        geometric_log_error("Failed to unlock mutex in global realloc");
    }
    return new_ptr;
}

void* unified_allocate_tensor(size_t* dimensions, size_t num_dimensions) {
    if (!dimensions || num_dimensions == 0) {
        geometric_log_error("Invalid tensor dimensions");
        return NULL;
    }
    
    unified_memory_interface_t* memory = get_global_memory_interface();
    if (!memory) {
        geometric_log_error("Failed to get global memory interface for tensor allocation");
        return NULL;
    }
    
    return memory->allocate_tensor(memory, dimensions, num_dimensions);
}

void* unified_allocate_hierarchical(size_t size, double tolerance) {
    if (size == 0) {
        geometric_log_error("Invalid hierarchical allocation size");
        return NULL;
    }
    
    unified_memory_interface_t* memory = get_global_memory_interface();
    if (!memory) {
        geometric_log_error("Failed to get global memory interface for hierarchical allocation");
        return NULL;
    }
    
    return memory->allocate_hierarchical(memory, size, tolerance);
}

void* unified_allocate_streaming(size_t total_size, size_t chunk_size) {
    if (total_size == 0 || chunk_size == 0) {
        geometric_log_error("Invalid streaming allocation parameters: total_size=%zu, chunk_size=%zu", total_size, chunk_size);
        return NULL;
    }
    
    unified_memory_interface_t* memory = get_global_memory_interface();
    if (!memory) {
        geometric_log_error("Failed to get global memory interface for streaming allocation");
        return NULL;
    }
    
    return memory->allocate_streaming(memory, total_size, chunk_size);
}

void get_memory_statistics(memory_stats_t* stats) {
    if (!stats) {
        geometric_log_error("Invalid NULL stats pointer");
        return;
    }
    
    unified_memory_interface_t* memory = get_global_memory_interface();
    if (!memory) {
        geometric_log_error("Failed to get global memory interface for statistics");
        return;
    }
    
    memory->allocator->get_stats(memory->allocator, stats);
}

void reset_memory_statistics(void) {
    unified_memory_interface_t* memory = get_global_memory_interface();
    if (!memory) {
        geometric_log_error("Failed to get global memory interface for reset statistics");
        return;
    }
    
    memory->allocator->reset_stats(memory->allocator);
}
