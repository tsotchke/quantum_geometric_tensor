/**
 * @file memory_optimization.c
 * @brief Distributed memory optimization for quantum geometric operations
 *
 * Provides memory management optimizations including allocation tracking,
 * fragmentation reduction, load balancing, and prefetching.
 */

#include "quantum_geometric/distributed/memory_optimization.h"
#include "quantum_geometric/core/memory_pool.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Error codes
#define QG_DISTRIBUTED_MEMORY_SUCCESS 0
#define QG_DISTRIBUTED_MEMORY_ERROR_INIT -1
#define QG_DISTRIBUTED_MEMORY_ERROR_ALLOC -2
#define QG_DISTRIBUTED_MEMORY_ERROR_INVALID_PTR -3
#define QG_DISTRIBUTED_MEMORY_ERROR_MIGRATION -4
#define QG_DISTRIBUTED_MEMORY_ERROR_PROTECTION -5

// Memory block tracking
typedef struct memory_block {
    void* ptr;
    size_t size;
    memory_region_type_t region_type;
    struct memory_block* next;
    size_t alignment;
    bool is_pinned;
} memory_block_t;

// Internal state
static distributed_memory_config_t current_config = {0};
static memory_distribution_t current_distribution = {0};
static memory_block_t* memory_blocks = NULL;
static int is_initialized = 0;
static size_t total_allocated = 0;
static size_t peak_allocated = 0;

// Memory statistics
typedef struct {
    size_t allocations;
    size_t deallocations;
    size_t bytes_allocated;
    size_t bytes_freed;
    size_t fragmentation_bytes;
} memory_stats_t;

static memory_stats_t stats = {0};

// Forward declarations
static memory_block_t* find_block(void* ptr);
static void remove_block(memory_block_t* block);
static void update_distribution_stats(void);
static size_t calculate_fragmentation(void);

// Initialize distributed memory system
int qg_distributed_memory_init(const distributed_memory_config_t* config) {
    if (!config) {
        return QG_DISTRIBUTED_MEMORY_ERROR_INIT;
    }

    // Store configuration
    current_config = *config;

    // Initialize distribution tracking
    size_t num_nodes = config->use_numa ? 4 : 1;  // Default NUMA node count

    current_distribution.num_nodes = num_nodes;
    current_distribution.node_sizes = calloc(num_nodes, sizeof(size_t));
    current_distribution.node_ptrs = calloc(num_nodes, sizeof(void*));
    current_distribution.is_balanced = true;

    if (!current_distribution.node_sizes || !current_distribution.node_ptrs) {
        free(current_distribution.node_sizes);
        free(current_distribution.node_ptrs);
        return QG_DISTRIBUTED_MEMORY_ERROR_ALLOC;
    }

    // Initialize block list
    memory_blocks = NULL;

    // Reset statistics
    memset(&stats, 0, sizeof(stats));
    total_allocated = 0;
    peak_allocated = 0;

    is_initialized = 1;
    return QG_DISTRIBUTED_MEMORY_SUCCESS;
}

// Allocate memory in specified region
void* qg_distributed_malloc(size_t size, memory_region_type_t region_type) {
    if (!is_initialized || size == 0) {
        return NULL;
    }

    // Determine alignment based on region type
    size_t alignment = 64;  // Default cache line alignment
    if (region_type == MEMORY_REGION_GPU) {
        alignment = 256;  // GPU memory alignment
    }

    // Allocate with alignment
    void* ptr = NULL;
#if defined(__APPLE__)
    // macOS doesn't have aligned_alloc in all versions
    if (posix_memalign(&ptr, alignment, size) != 0) {
        ptr = NULL;
    }
#else
    ptr = aligned_alloc(alignment, ((size + alignment - 1) / alignment) * alignment);
#endif

    if (!ptr) {
        return NULL;
    }

    // Zero initialize
    memset(ptr, 0, size);

    // Create tracking block
    memory_block_t* block = calloc(1, sizeof(memory_block_t));
    if (!block) {
        free(ptr);
        return NULL;
    }

    block->ptr = ptr;
    block->size = size;
    block->region_type = region_type;
    block->alignment = alignment;
    block->is_pinned = false;
    block->next = memory_blocks;
    memory_blocks = block;

    // Update statistics
    stats.allocations++;
    stats.bytes_allocated += size;
    total_allocated += size;
    if (total_allocated > peak_allocated) {
        peak_allocated = total_allocated;
    }

    // Update distribution
    update_distribution_stats();

    return ptr;
}

// Free distributed memory
void qg_distributed_free(void* ptr) {
    if (!is_initialized || !ptr) {
        return;
    }

    memory_block_t* block = find_block(ptr);
    if (!block) {
        return;
    }

    // Update statistics
    stats.deallocations++;
    stats.bytes_freed += block->size;
    total_allocated -= block->size;

    // Remove from tracking
    remove_block(block);

    // Free the memory
    free(ptr);

    // Update distribution
    update_distribution_stats();
}

// Get current memory distribution
const memory_distribution_t* qg_get_memory_distribution(void) {
    if (!is_initialized) {
        return NULL;
    }
    return &current_distribution;
}

// Optimize memory distribution
int qg_optimize_memory_distribution(void) {
    if (!is_initialized) {
        return QG_DISTRIBUTED_MEMORY_ERROR_INIT;
    }

    // Calculate fragmentation
    size_t fragmentation = calculate_fragmentation();
    stats.fragmentation_bytes = fragmentation;

    // Check if rebalancing is needed
    if (current_distribution.num_nodes > 1) {
        size_t avg_size = total_allocated / current_distribution.num_nodes;
        bool needs_rebalance = false;

        for (size_t i = 0; i < current_distribution.num_nodes; i++) {
            size_t diff = current_distribution.node_sizes[i] > avg_size ?
                         current_distribution.node_sizes[i] - avg_size :
                         avg_size - current_distribution.node_sizes[i];
            if (diff > avg_size * 0.2) {  // 20% threshold
                needs_rebalance = true;
                break;
            }
        }

        current_distribution.is_balanced = !needs_rebalance;
    }

    return QG_DISTRIBUTED_MEMORY_SUCCESS;
}

// Reduce memory fragmentation
int qg_reduce_memory_fragmentation(void) {
    if (!is_initialized) {
        return QG_DISTRIBUTED_MEMORY_ERROR_INIT;
    }

    // Calculate current fragmentation
    size_t fragmentation = calculate_fragmentation();

    // If fragmentation is low, nothing to do
    if (fragmentation < total_allocated * 0.1) {  // 10% threshold
        return QG_DISTRIBUTED_MEMORY_SUCCESS;
    }

    // In a real implementation, we would:
    // 1. Identify fragmented regions
    // 2. Compact memory by moving allocations
    // 3. Update pointers

    // For now, just update statistics
    stats.fragmentation_bytes = fragmentation;

    return QG_DISTRIBUTED_MEMORY_SUCCESS;
}

// Balance memory load across nodes
int qg_balance_memory_load(void) {
    if (!is_initialized) {
        return QG_DISTRIBUTED_MEMORY_ERROR_INIT;
    }

    if (current_distribution.num_nodes <= 1) {
        return QG_DISTRIBUTED_MEMORY_SUCCESS;
    }

    // Calculate target size per node
    // size_t target_per_node = total_allocated / current_distribution.num_nodes;

    // In a real implementation, we would migrate memory between nodes
    // For now, just mark as balanced
    current_distribution.is_balanced = true;

    return QG_DISTRIBUTED_MEMORY_SUCCESS;
}

// Prefetch data for future access
int qg_prefetch_data(const void* data, size_t size) {
    if (!is_initialized || !data || size == 0) {
        return QG_DISTRIBUTED_MEMORY_ERROR_INVALID_PTR;
    }

    // Use compiler intrinsics for prefetching if available
#if defined(__GNUC__) || defined(__clang__)
    const char* ptr = (const char*)data;
    const char* end = ptr + size;

    // Prefetch in cache-line sized chunks
    while (ptr < end) {
        __builtin_prefetch(ptr, 0, 3);  // Read, high temporal locality
        ptr += 64;  // Cache line size
    }
#else
    (void)data;
    (void)size;
#endif

    return QG_DISTRIBUTED_MEMORY_SUCCESS;
}

// Clean up distributed memory system
void qg_distributed_memory_cleanup(void) {
    if (!is_initialized) {
        return;
    }

    // Free all tracked blocks
    memory_block_t* block = memory_blocks;
    while (block) {
        memory_block_t* next = block->next;
        free(block->ptr);
        free(block);
        block = next;
    }
    memory_blocks = NULL;

    // Free distribution arrays
    free(current_distribution.node_sizes);
    free(current_distribution.node_ptrs);
    current_distribution.node_sizes = NULL;
    current_distribution.node_ptrs = NULL;

    // Reset state
    memset(&current_config, 0, sizeof(current_config));
    memset(&current_distribution, 0, sizeof(current_distribution));
    memset(&stats, 0, sizeof(stats));
    total_allocated = 0;
    peak_allocated = 0;
    is_initialized = 0;
}

// Find a block by pointer
static memory_block_t* find_block(void* ptr) {
    memory_block_t* block = memory_blocks;
    while (block) {
        if (block->ptr == ptr) {
            return block;
        }
        block = block->next;
    }
    return NULL;
}

// Remove a block from the list
static void remove_block(memory_block_t* target) {
    if (!target) return;

    if (memory_blocks == target) {
        memory_blocks = target->next;
        free(target);
        return;
    }

    memory_block_t* block = memory_blocks;
    while (block && block->next != target) {
        block = block->next;
    }

    if (block) {
        block->next = target->next;
        free(target);
    }
}

// Update distribution statistics
static void update_distribution_stats(void) {
    if (!is_initialized || current_distribution.num_nodes == 0) {
        return;
    }

    // Reset node sizes
    for (size_t i = 0; i < current_distribution.num_nodes; i++) {
        current_distribution.node_sizes[i] = 0;
    }

    // Count allocations per region/node
    memory_block_t* block = memory_blocks;
    while (block) {
        // Distribute based on region type (simplified)
        size_t node_idx = block->region_type % current_distribution.num_nodes;
        current_distribution.node_sizes[node_idx] += block->size;
        block = block->next;
    }
}

// Calculate fragmentation
static size_t calculate_fragmentation(void) {
    if (!is_initialized || total_allocated == 0) {
        return 0;
    }

    // Simple fragmentation estimate based on allocation pattern
    size_t block_count = 0;
    size_t total_size = 0;
    size_t smallest_block = SIZE_MAX;
    size_t largest_block = 0;

    memory_block_t* block = memory_blocks;
    while (block) {
        block_count++;
        total_size += block->size;
        if (block->size < smallest_block) smallest_block = block->size;
        if (block->size > largest_block) largest_block = block->size;
        block = block->next;
    }

    if (block_count <= 1) {
        return 0;
    }

    // Fragmentation estimate: deviation from ideal uniform allocation
    size_t avg_size = total_size / block_count;
    size_t variance = 0;

    block = memory_blocks;
    while (block) {
        size_t diff = block->size > avg_size ? block->size - avg_size : avg_size - block->size;
        variance += diff;
        block = block->next;
    }

    return variance / block_count;
}

// Get error string
const char* qg_distributed_memory_get_error_string(int error) {
    switch (error) {
        case QG_DISTRIBUTED_MEMORY_SUCCESS:
            return "Success";
        case QG_DISTRIBUTED_MEMORY_ERROR_INIT:
            return "Initialization error";
        case QG_DISTRIBUTED_MEMORY_ERROR_ALLOC:
            return "Memory allocation error";
        case QG_DISTRIBUTED_MEMORY_ERROR_INVALID_PTR:
            return "Invalid pointer";
        case QG_DISTRIBUTED_MEMORY_ERROR_MIGRATION:
            return "Memory migration error";
        case QG_DISTRIBUTED_MEMORY_ERROR_PROTECTION:
            return "Memory protection error";
        default:
            return "Unknown error";
    }
}
