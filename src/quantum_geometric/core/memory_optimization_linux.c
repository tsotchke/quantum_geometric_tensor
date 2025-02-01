/**
 * @file memory_optimization_linux.c
 * @brief Linux-specific memory optimization implementation with NUMA support
 */

#include "quantum_geometric/core/memory_optimization_impl.h"
#include "quantum_geometric/core/performance_monitor.h"
#include "quantum_geometric/core/quantum_geometric_logging.h"
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <sys/mman.h>

#ifdef __linux__
#include <numa.h>
#endif

// Memory optimization parameters
#define CACHE_LINE_SIZE 64
#define HUGE_PAGE_SIZE (2 * 1024 * 1024) // 2MB huge pages
#define MAX_MEMORY_POOLS 16
#define ALIGNMENT_SIZE 64
#define MEMORY_GUARD_SIZE 64 // Guard pages for bounds checking

// Memory pool types
typedef enum {
    POOL_SMALL,    // < 4KB
    POOL_MEDIUM,   // 4KB - 1MB
    POOL_LARGE,    // 1MB - 64MB
    POOL_HUGE      // > 64MB
} PoolType;

// Memory block header with guard pages
typedef struct MemoryBlock {
    size_t size;
    bool is_free;
    struct MemoryBlock* next;
    struct MemoryBlock* prev;
    uint32_t guard_pattern;  // For bounds checking
    char padding[40];        // Maintain alignment
    void* guard_page_start; // Start guard page
    void* guard_page_end;   // End guard page
    char data[] __attribute__((aligned(ALIGNMENT_SIZE)));
} MemoryBlock;

// Memory pool with monitoring
typedef struct {
    void* base_address;
    size_t total_size;
    size_t used_size;
    size_t block_size;
    MemoryBlock* free_list;
    pthread_mutex_t mutex;
    int numa_node;
    struct {
        size_t allocations;
        size_t deallocations;
        size_t fragmentation;
        size_t peak_usage;
        double utilization;
    } metrics;
} MemoryPool;

// Memory manager with enhanced monitoring
struct MemoryManager {
    MemoryPool pools[MAX_MEMORY_POOLS];
    size_t num_pools;
    bool numa_enabled;
    size_t total_memory;
    size_t peak_memory;
    struct {
        size_t page_faults;
        size_t cache_misses;
        size_t tlb_misses;
        double efficiency;
    } performance;
};

// Forward declarations
static void init_guard_pages(MemoryBlock* block);
static bool verify_guard_pages(const MemoryBlock* block);
static void update_performance_metrics(MemoryManager* manager);

// Platform-specific interface implementation
bool has_numa_support(void) {
    return numa_available() >= 0;
}

int get_numa_node_count(void) {
    if (!has_numa_support()) return 1;
    return numa_num_configured_nodes();
}

bool bind_memory_to_node(void* ptr, size_t size, int node) {
    if (!has_numa_support()) return false;
    void* new_ptr = numa_alloc_onnode(size, node);
    if (!new_ptr) return false;
    memcpy(new_ptr, ptr, size);
    numa_free(ptr, size);
    return true;
}

bool get_memory_node(const void* ptr, int* node) {
    if (!has_numa_support() || !ptr || !node) return false;
    *node = numa_preferred();
    return true;
}

size_t get_cache_line_size(void) {
    return CACHE_LINE_SIZE;
}

bool align_memory_to_cache(void** ptr) {
    if (!ptr || !*ptr) return false;
    void* aligned = (void*)(((uintptr_t)*ptr + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1));
    *ptr = aligned;
    return true;
}

bool prefetch_memory_range(const void* ptr, size_t size) {
    if (!ptr) return false;
    __builtin_prefetch(ptr, 0, 3);
    return true;
}

bool has_huge_page_support(void) {
    return access("/sys/kernel/mm/hugepages", F_OK) == 0;
}

size_t get_huge_page_size(void) {
    return HUGE_PAGE_SIZE;
}

void* allocate_huge_pages(size_t size) {
    if (!has_huge_page_support()) return NULL;
    return mmap(NULL, size, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
}

void free_huge_pages(void* ptr, size_t size) {
    if (ptr) munmap(ptr, size);
}

uint64_t get_page_faults(void) {
    return get_performance_counter(COUNTER_PAGE_FAULTS);
}

uint64_t get_cache_misses(void) {
    return get_performance_counter(COUNTER_CACHE_MISSES);
}

uint64_t get_tlb_misses(void) {
    return get_performance_counter(COUNTER_TLB_MISSES);
}

// Update performance metrics
static void update_performance_metrics(MemoryManager* manager) {
    // Get hardware performance counters
    uint64_t page_faults = get_page_faults();
    uint64_t cache_misses = get_cache_misses();
    uint64_t tlb_misses = get_tlb_misses();
    
    manager->performance.page_faults = page_faults;
    manager->performance.cache_misses = cache_misses;
    manager->performance.tlb_misses = tlb_misses;
    
    // Calculate efficiency metric
    double efficiency = 1.0;
    if (manager->total_memory > 0) {
        efficiency *= (1.0 - (double)cache_misses / manager->total_memory);
        efficiency *= (1.0 - (double)tlb_misses / manager->total_memory);
    }
    manager->performance.efficiency = efficiency;
    
    // Log significant changes
    if (efficiency < 0.5) {
        log_warning("Low memory efficiency detected: %.2f", efficiency);
    }
}

// Initialize guard pages for bounds checking
static void init_guard_pages(MemoryBlock* block) {
    // Create guard pages with PROT_NONE
    block->guard_page_start = mmap(NULL, MEMORY_GUARD_SIZE,
        PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    block->guard_page_end = mmap(NULL, MEMORY_GUARD_SIZE,
        PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    
    // Set guard pattern
    block->guard_pattern = 0xDEADBEEF;
}

// Verify guard pages and pattern
static bool verify_guard_pages(const MemoryBlock* block) {
    if (block->guard_pattern != 0xDEADBEEF) {
        log_error("Memory corruption detected: invalid guard pattern");
        return false;
    }
    
    // Attempt to access guard pages should trigger SIGSEGV
    if (msync(block->guard_page_start, MEMORY_GUARD_SIZE, MS_ASYNC) == 0 ||
        msync(block->guard_page_end, MEMORY_GUARD_SIZE, MS_ASYNC) == 0) {
        log_error("Memory corruption detected: guard page violation");
        return false;
    }
    
    return true;
}

// Initialize memory pool with monitoring
static bool init_memory_pool(MemoryPool* pool,
                           size_t size,
                           int numa_node) {
    pool->total_size = size;
    pool->used_size = 0;
    pool->block_size = size;
    pool->numa_node = numa_node;
    
    // Initialize metrics
    pool->metrics.allocations = 0;
    pool->metrics.deallocations = 0;
    pool->metrics.fragmentation = 0;
    pool->metrics.peak_usage = 0;
    pool->metrics.utilization = 0.0;
    
    // Allocate memory with huge pages if size is large enough
    if (size >= HUGE_PAGE_SIZE) {
        if (numa_available() >= 0) {
            pool->base_address = numa_alloc_onnode(size, numa_node);
            madvise(pool->base_address, size, MADV_HUGEPAGE);
        } else {
            pool->base_address = mmap(NULL, size,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                -1, 0);
        }
    } else {
        if (numa_available() >= 0) {
            pool->base_address = numa_alloc_onnode(size, numa_node);
        } else {
            pool->base_address = aligned_alloc(ALIGNMENT_SIZE, size);
        }
    }
    
    if (!pool->base_address) {
        log_error("Failed to allocate memory pool");
        return false;
    }
    
    // Initialize mutex
    if (pthread_mutex_init(&pool->mutex, NULL) != 0) {
        if (numa_available() >= 0) {
            numa_free(pool->base_address, size);
        } else {
            if (size >= HUGE_PAGE_SIZE) {
                munmap(pool->base_address, size);
            } else {
                free(pool->base_address);
            }
        }
        log_error("Failed to initialize mutex");
        return false;
    }
    
    // Initialize free list with guard pages
    pool->free_list = (MemoryBlock*)pool->base_address;
    pool->free_list->size = size - sizeof(MemoryBlock);
    pool->free_list->is_free = true;
    pool->free_list->next = NULL;
    pool->free_list->prev = NULL;
    init_guard_pages(pool->free_list);
    
    return true;
}

// Initialize memory manager
MemoryManager* init_memory_manager(void) {
    MemoryManager* manager = aligned_alloc(ALIGNMENT_SIZE,
        sizeof(MemoryManager));
    if (!manager) {
        log_error("Failed to allocate memory manager");
        return NULL;
    }
    
    memset(manager, 0, sizeof(MemoryManager));
    
    // Check NUMA support
    manager->numa_enabled = numa_available() >= 0;
    
    // Initialize performance metrics
    manager->performance.page_faults = 0;
    manager->performance.cache_misses = 0;
    manager->performance.tlb_misses = 0;
    manager->performance.efficiency = 1.0;
    
    // Initialize memory pools for different sizes
    size_t pool_sizes[] = {
        4 * 1024,        // 4KB
        1 * 1024 * 1024, // 1MB
        64 * 1024 * 1024 // 64MB
    };
    
    for (size_t i = 0; i < sizeof(pool_sizes) / sizeof(size_t); i++) {
        if (!init_memory_pool(&manager->pools[i],
                            pool_sizes[i],
                            i % numa_num_configured_nodes())) {
            cleanup_memory_manager(manager);
            return NULL;
        }
        manager->num_pools++;
    }
    
    log_info("Memory manager initialized successfully");
    return manager;
}

// Allocate memory with monitoring
void* optimized_malloc(MemoryManager* manager, size_t size) {
    if (!manager || size == 0) {
        log_error("Invalid malloc parameters");
        return NULL;
    }
    
    // Add header size and align
    size_t total_size = size + sizeof(MemoryBlock);
    total_size = (total_size + ALIGNMENT_SIZE - 1) &
                 ~(ALIGNMENT_SIZE - 1);
    
    // Select appropriate pool
    MemoryPool* pool = NULL;
    for (size_t i = 0; i < manager->num_pools; i++) {
        if (total_size <= manager->pools[i].block_size) {
            pool = &manager->pools[i];
            break;
        }
    }
    
    if (!pool) {
        // Fallback to huge pages for large allocations
        MemoryBlock* block = mmap(NULL, total_size,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
            -1, 0);
            
        if (block == MAP_FAILED) {
            // Fallback to regular allocation
            block = aligned_alloc(ALIGNMENT_SIZE, total_size);
            if (!block) {
                log_error("Failed to allocate memory");
                return NULL;
            }
        }
        
        block->size = size;
        block->is_free = false;
        block->next = NULL;
        block->prev = NULL;
        init_guard_pages(block);
        
        manager->total_memory += total_size;
        manager->peak_memory = manager->total_memory > manager->peak_memory ?
                             manager->total_memory : manager->peak_memory;
        
        // Update performance metrics
        update_performance_metrics(manager);
        
        return block->data;
    }
    
    // Lock pool
    pthread_mutex_lock(&pool->mutex);
    
    // Find free block
    MemoryBlock* block = pool->free_list;
    while (block) {
        if (!verify_guard_pages(block)) {
            log_error("Memory corruption detected during allocation");
            pthread_mutex_unlock(&pool->mutex);
            return NULL;
        }
        
        if (block->is_free && block->size >= size) {
            // Split block if large enough
            if (block->size >= total_size + sizeof(MemoryBlock) +
                ALIGNMENT_SIZE) {
                MemoryBlock* new_block = (MemoryBlock*)(
                    (char*)block + total_size);
                
                new_block->size = block->size - total_size;
                new_block->is_free = true;
                new_block->next = block->next;
                new_block->prev = block;
                init_guard_pages(new_block);
                
                if (block->next) {
                    block->next->prev = new_block;
                }
                
                block->next = new_block;
                block->size = size;
            }
            
            block->is_free = false;
            pool->used_size += block->size;
            pool->metrics.allocations++;
            pool->metrics.peak_usage = pool->used_size > pool->metrics.peak_usage ?
                                     pool->used_size : pool->metrics.peak_usage;
            pool->metrics.utilization = (double)pool->used_size /
                                      pool->total_size;
            
            manager->total_memory += block->size;
            manager->peak_memory = manager->total_memory > manager->peak_memory ?
                                 manager->total_memory : manager->peak_memory;
            
            // Update performance metrics
            update_performance_metrics(manager);
            
            pthread_mutex_unlock(&pool->mutex);
            return block->data;
        }
        block = block->next;
    }
    
    pthread_mutex_unlock(&pool->mutex);
    log_error("No suitable memory block found");
    return NULL;
}

// Free memory with validation
void optimized_free(MemoryManager* manager, void* ptr) {
    if (!manager || !ptr) {
        log_error("Invalid free parameters");
        return;
    }
    
    // Get block header
    MemoryBlock* block = (MemoryBlock*)((char*)ptr - offsetof(MemoryBlock, data));
    
    if (!verify_guard_pages(block)) {
        log_error("Memory corruption detected during free");
        return;
    }
    
    // Find containing pool
    MemoryPool* pool = NULL;
    for (size_t i = 0; i < manager->num_pools; i++) {
        if (ptr >= manager->pools[i].base_address &&
            ptr < (void*)((char*)manager->pools[i].base_address +
                         manager->pools[i].total_size)) {
            pool = &manager->pools[i];
            break;
        }
    }
    
    if (!pool) {
        // Direct allocation
        manager->total_memory -= block->size;
        munmap(block->guard_page_start, MEMORY_GUARD_SIZE);
        munmap(block->guard_page_end, MEMORY_GUARD_SIZE);
        
        if (block->size >= HUGE_PAGE_SIZE) {
            munmap(block, block->size + sizeof(MemoryBlock));
        } else {
            free(block);
        }
        
        // Update performance metrics
        update_performance_metrics(manager);
        return;
    }
    
    // Lock pool
    pthread_mutex_lock(&pool->mutex);
    
    // Mark block as free
    block->is_free = true;
    pool->used_size -= block->size;
    pool->metrics.deallocations++;
    pool->metrics.utilization = (double)pool->used_size /
                               pool->total_size;
    manager->total_memory -= block->size;
    
    // Coalesce with neighbors if possible
    if (block->prev && block->prev->is_free) {
        block->prev->size += sizeof(MemoryBlock) + block->size;
        block->prev->next = block->next;
        if (block->next) {
            block->next->prev = block->prev;
        }
        munmap(block->guard_page_start, MEMORY_GUARD_SIZE);
        munmap(block->guard_page_end, MEMORY_GUARD_SIZE);
        block = block->prev;
    }
    
    if (block->next && block->next->is_free) {
        block->size += sizeof(MemoryBlock) + block->next->size;
        munmap(block->next->guard_page_start, MEMORY_GUARD_SIZE);
        munmap(block->next->guard_page_end, MEMORY_GUARD_SIZE);
        block->next = block->next->next;
        if (block->next) {
            block->next->prev = block;
        }
    }
    
    // Update fragmentation metric
    size_t total_free = 0;
    size_t free_blocks = 0;
    MemoryBlock* curr = pool->free_list;
    while (curr) {
        if (curr->is_free) {
            total_free += curr->size;
            free_blocks++;
        }
        curr = curr->next;
    }
    pool->metrics.fragmentation = free_blocks > 1 ?
        (1.0 - (double)total_free / (pool->total_size - pool->used_size)) : 0;
    
    // Update performance metrics
    update_performance_metrics(manager);
    
    pthread_mutex_unlock(&pool->mutex);
}

// Get memory statistics
memory_stats_t get_memory_stats(const MemoryManager* manager) {
    memory_stats_t stats = {0};
    
    if (!manager) {
        log_error("Invalid manager for stats");
        return stats;
    }
    
    stats.total_allocations = 0;
    stats.total_deallocations = 0;
    stats.peak_memory = manager->peak_memory;
    stats.current_memory = manager->total_memory;
    stats.cache_misses = manager->performance.cache_misses;
    stats.page_faults = manager->performance.page_faults;
    stats.fragmentation_ratio = 0.0;
    stats.access_pattern = ACCESS_PATTERN_SEQUENTIAL;
    
    // Aggregate stats from all pools
    for (size_t i = 0; i < manager->num_pools; i++) {
        const MemoryPool* pool = &manager->pools[i];
        stats.total_allocations += pool->metrics.allocations;
        stats.total_deallocations += pool->metrics.deallocations;
        stats.fragmentation_ratio += pool->metrics.fragmentation;
    }
    
    if (manager->num_pools > 0) {
        stats.fragmentation_ratio /= manager->num_pools;
    }
    
    return stats;
}

// Clean up memory manager
void cleanup_memory_manager(MemoryManager* manager) {
    if (!manager) return;
    
    for (size_t i = 0; i < manager->num_pools; i++) {
        MemoryPool* pool = &manager->pools[i];
        
        // Free all guard pages
        MemoryBlock* block = pool->free_list;
        while (block) {
            munmap(block->guard_page_start, MEMORY_GUARD_SIZE);
            munmap(block->guard_page_end, MEMORY_GUARD_SIZE);
            block = block->next;
        }
        
        pthread_mutex_destroy(&pool->mutex);
        
        if (manager->numa_enabled) {
            numa_free(pool->base_address, pool->total_size);
        } else {
            if (pool->total_size >= HUGE_PAGE_SIZE) {
                munmap(pool->base_address, pool->total_size);
            } else {
                free(pool->base_address);
            }
        }
    }
    
    free(manager);
    log_info("Memory manager cleaned up successfully");
}
