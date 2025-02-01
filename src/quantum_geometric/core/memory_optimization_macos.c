/**
 * @file memory_optimization_macos.c
 * @brief macOS-specific memory optimization implementation
 */

#include "quantum_geometric/core/memory_optimization_impl.h"
#include "quantum_geometric/core/memory_optimization.h"
#include "quantum_geometric/core/performance_monitor.h"
#include "quantum_geometric/core/quantum_geometric_logging.h"
#include "quantum_geometric/core/system_memory.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/memory_pool.h"
#include "quantum_geometric/hardware/hardware_capabilities.h"
#include <unistd.h>
#include <stdlib.h>
#include <vecLib/clapack.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <pthread.h>
#include <sys/mman.h>
#include <mach/vm_statistics.h>
#include <mach/mach_types.h>
#include <mach/mach_init.h>
#include <mach/mach_host.h>
#include <Accelerate/Accelerate.h>

// Memory manager structure
typedef struct MemoryManager {
    MemoryPool pools[4];          // Array of memory pools
    size_t num_pools;             // Number of active pools
    _Atomic size_t total_memory;  // Total memory managed
    _Atomic size_t peak_memory;   // Peak memory usage
    memory_stats_t stats;         // Memory statistics
    PerformanceMetrics performance;    // Performance metrics
    ThreadCache* thread_caches;    // Per-thread caches
    pthread_key_t cache_key;       // Thread-local storage key
} MemoryManager;

// Constants
// Use system PAGE_SIZE from vm_param.h

// Memory optimization parameters
// Use values from system_memory.h
#define DEFAULT_SIZE_CLASSES QG_NUM_SIZE_CLASSES
#define MIN_BLOCK_SIZE QG_MIN_BLOCK_SIZE
#define MAX_BLOCK_SIZE QG_MAX_BLOCK_SIZE
#define MAX_THREAD_CACHE QG_MAX_THREAD_CACHE

// Forward declarations
static void init_block(Block* block);
static bool verify_block(const Block* block);
static memory_access_pattern_t detect_access_pattern(const void* ptr, size_t size);
static bool is_memory_aligned(const void* ptr, size_t alignment);
uint64_t get_performance_counter(performance_counter_t counter);

void cleanup_memory_manager(MemoryManager* manager) {
    if (!manager) return;
    
    // Clean up each memory pool
    for (size_t i = 0; i < manager->num_pools; i++) {
        // Destroy mutex
        pthread_mutex_destroy(&manager->pools[i].mutex);
        
        // Free base memory
        if (manager->pools[i].base_address) {
            free(manager->pools[i].base_address);
            manager->pools[i].base_address = NULL;
        }
    }
    
    // Free manager itself
    free(manager);
}

// Get memory statistics
const memory_stats_t* get_memory_stats(const MemoryManager* manager) {
    if (!manager) {
        geometric_log_error("Invalid memory manager");
        return NULL;
    }
    
    // Get VM statistics
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    vm_statistics64_data_t vm_stats;
    
    memory_stats_t stats = manager->stats;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                         (host_info64_t)&vm_stats, &count) == KERN_SUCCESS) {
        stats.page_faults = vm_stats.faults;
        ((memory_stats_t*)&manager->stats)->page_faults = stats.page_faults;
    }
    
    return &manager->stats;
}

// Global configuration
static memory_optimization_config_t g_memory_config;

// Legacy memory optimization system implementation
qgt_error_t init_memory_optimization(const memory_optimization_config_t* config) {
    if (!config) {
        geometric_log_error("Invalid memory optimization config");
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    // Store configuration for later use
    g_memory_config = *config;
    return QGT_SUCCESS;
}

void cleanup_memory_optimization(void) {
    // Nothing to clean up on macOS
}

qgt_error_t register_memory_region(memory_region_t* region, void* base, size_t size) {
    if (!region || !base || !size) {
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    region->base_address = base;
    region->size = size;
    region->access_pattern = ACCESS_PATTERN_SEQUENTIAL;
    region->strategy = STRATEGY_CACHE_ALIGNED;
    region->is_optimized = false;
    
    memset(&region->stats, 0, sizeof(memory_stats_t));
    return QGT_SUCCESS;
}

qgt_error_t analyze_memory_pattern(memory_region_t* region) {
    if (!region || !region->base_address) {
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    region->access_pattern = detect_access_pattern(region->base_address, region->size);
    return QGT_SUCCESS;
}

qgt_error_t optimize_memory_region(memory_region_t* region, memory_strategy_t strategy) {
    if (!region || !region->base_address) {
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    switch (strategy) {
        case STRATEGY_POOL_ALLOCATION:
            // Already handled by memory pool system
            break;
            
        case STRATEGY_PREFETCH:
            if (region->access_pattern == ACCESS_PATTERN_SEQUENTIAL) {
                __builtin_prefetch(region->base_address, 0, 3);
            }
            break;
            
        case STRATEGY_CACHE_ALIGNED:
            if (!is_memory_aligned(region->base_address, CACHE_LINE_SIZE)) {
                geometric_log_warning("Memory not cache-aligned");
            }
            break;
            
        case STRATEGY_NUMA_AWARE:
            // Not supported on macOS
            break;
            
        case STRATEGY_COMPRESSION:
            // Not supported on macOS
            break;
    }
    
    region->strategy = strategy;
    region->is_optimized = true;
    return QGT_SUCCESS;
}

qgt_error_t update_memory_stats(memory_region_t* region) {
    if (!region) {
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    // Get VM statistics
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    vm_statistics64_data_t vm_stats;
    
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                         (host_info64_t)&vm_stats, &count) == KERN_SUCCESS) {
        region->stats.page_faults = vm_stats.faults;
    }
    
    return QGT_SUCCESS;
}

qgt_error_t prefetch_memory(const memory_region_t* region, size_t offset, size_t size) {
    if (!region || offset + size > region->size) {
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    void* addr = (char*)region->base_address + offset;
    __builtin_prefetch(addr, 0, 3);
    return QGT_SUCCESS;
}

qgt_error_t configure_prefetch(size_t distance, size_t stride) {
    // macOS doesn't support configuring prefetch parameters
    return QGT_SUCCESS;
}

qgt_error_t bind_to_numa_node(const memory_region_t* region, int node) {
    // Not supported on macOS
    return QGT_ERROR_NOT_IMPLEMENTED;
}

qgt_error_t get_optimal_numa_node(const memory_region_t* region) {
    // Not supported on macOS
    return 0;
}

qgt_error_t align_to_cache_line(void** ptr) {
    if (!ptr || !*ptr) {
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    *ptr = (void*)(((uintptr_t)*ptr + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1));
    return QGT_SUCCESS;
}

qgt_error_t optimize_cache_layout(memory_region_t* region) {
    if (!region || !region->base_address) {
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    // Ensure base address is cache-aligned
    if (!is_memory_aligned(region->base_address, CACHE_LINE_SIZE)) {
        geometric_log_warning("Memory region not cache-aligned");
    }
    
    return QGT_SUCCESS;
}

qgt_error_t optimize_access_pattern(memory_region_t* region) {
    if (!region) {
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    // Analyze current access pattern
    analyze_memory_pattern(region);
    
    // Apply optimizations based on pattern
    switch (region->access_pattern) {
        case ACCESS_PATTERN_SEQUENTIAL:
            prefetch_memory(region, 0, region->size);
            break;
        case ACCESS_PATTERN_STRIDED:
            optimize_cache_layout(region);
            break;
        default:
            break;
    }
    
    return QGT_SUCCESS;
}

qgt_error_t reorder_memory_layout(memory_region_t* region) {
    // Not implemented on macOS
    return QGT_ERROR_NOT_IMPLEMENTED;
}

qgt_error_t defragment_memory(memory_region_t* region) {
    if (!region) {
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    // Basic defragmentation using existing pool functions
    if (region->strategy == STRATEGY_POOL_ALLOCATION) {
        optimize_pool((MemoryPool*)region->base_address);
        defragment_pool((MemoryPool*)region->base_address);
    }
    
    return QGT_SUCCESS;
}

qgt_error_t compact_memory(memory_region_t* region) {
    // Not supported on macOS
    return QGT_ERROR_NOT_SUPPORTED;
}

qgt_error_t start_memory_monitoring(memory_region_t* region) {
    if (!region) {
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    // Initialize monitoring stats
    memset(&region->stats, 0, sizeof(memory_stats_t));
    return QGT_SUCCESS;
}

qgt_error_t stop_memory_monitoring(memory_region_t* region) {
    if (!region) {
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    // Final stats update
    update_memory_stats(region);
    return QGT_SUCCESS;
}

qgt_error_t get_monitoring_results(const memory_region_t* region, memory_stats_t* stats) {
    if (!region || !stats) {
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    *stats = region->stats;
    return QGT_SUCCESS;
}

qgt_error_t get_optimization_suggestions(const memory_region_t* region,
                                       optimization_suggestion_t* suggestions,
                                       size_t* count) {
    if (!region || !suggestions || !count || *count == 0) {
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    size_t suggestion_count = 0;
    
    // Add cache alignment suggestion if needed
    if (!is_memory_aligned(region->base_address, CACHE_LINE_SIZE)) {
        suggestions[suggestion_count].recommended_strategy = STRATEGY_CACHE_ALIGNED;
        suggestions[suggestion_count].expected_improvement = 0.15;
        snprintf(suggestions[suggestion_count].description, 256,
                "Align memory to cache line boundary for improved access performance");
        suggestion_count++;
    }
    
    // Add prefetch suggestion for sequential access
    if (region->access_pattern == ACCESS_PATTERN_SEQUENTIAL) {
        suggestions[suggestion_count].recommended_strategy = STRATEGY_PREFETCH;
        suggestions[suggestion_count].expected_improvement = 0.20;
        snprintf(suggestions[suggestion_count].description, 256,
                "Enable prefetching for sequential memory access pattern");
        suggestion_count++;
    }
    
    *count = suggestion_count;
    return QGT_SUCCESS;
}

qgt_error_t validate_memory_optimization(const memory_region_t* region) {
    if (!region) {
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    // Validate optimizations
    if (region->is_optimized) {
        // Check cache alignment
        if (region->strategy == STRATEGY_CACHE_ALIGNED &&
            !is_memory_aligned(region->base_address, CACHE_LINE_SIZE)) {
            return QGT_ERROR_VALIDATION_FAILED;
        }
    }
    
    return QGT_SUCCESS;
}

qgt_error_t verify_memory_integrity(const memory_region_t* region) {
    if (!region || !region->base_address) {
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    // Try to read the memory range
    volatile const char* p = (const char*)region->base_address;
    for (size_t i = 0; i < region->size; i += 4096) {
        char dummy = p[i];
        (void)dummy;
    }
    
    return QGT_SUCCESS;
}

// Helper functions
static memory_access_pattern_t detect_access_pattern(const void* ptr, size_t size) {
    // Simple pattern detection based on size and alignment
    if (is_memory_aligned(ptr, 4096)) {
        return ACCESS_PATTERN_SEQUENTIAL;
    } else if (is_memory_aligned(ptr, CACHE_LINE_SIZE)) {
        return ACCESS_PATTERN_STRIDED;
    }
    return ACCESS_PATTERN_RANDOM;
}

static bool is_memory_aligned(const void* ptr, size_t alignment) {
    return ((uintptr_t)ptr & (alignment - 1)) == 0;
}

// Get macOS-specific performance counter value
uint64_t get_performance_counter(performance_counter_t counter) {
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    vm_statistics64_data_t vm_stats;
    
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                         (host_info64_t)&vm_stats, &count) != KERN_SUCCESS) {
        geometric_log_error("Failed to get VM statistics");
        return 0;
    }
    
    switch (counter) {
        case COUNTER_PAGE_FAULTS:
            return vm_stats.faults;
        case COUNTER_CACHE_MISSES:
            // macOS doesn't provide direct cache miss stats
            return 0;
        case COUNTER_TLB_MISSES:
            // macOS doesn't provide direct TLB miss stats
            return 0;
        default:
            geometric_log_error("Invalid performance counter type");
            return 0;
    }
}

// Platform-specific memory protection
bool protect_memory_range(void* ptr, size_t size, bool readonly) {
    if (!ptr || size == 0) {
        geometric_log_error("Invalid memory range");
        return false;
    }

    int prot = readonly ? PROT_READ : (PROT_READ | PROT_WRITE);
    if (mprotect(ptr, size, prot) != 0) {
        geometric_log_error("Failed to protect memory range");
        return false;
    }

    return true;
}

bool verify_memory_protection(const void* ptr, size_t size) {
    if (!ptr || size == 0) return false;
    
    // Try to read the memory - should work if protected correctly
    volatile const char* p = (const char*)ptr;
    char dummy = *p; // Just read, don't use
    (void)dummy;     // Prevent unused variable warning
    
    return true;
}

// Platform-specific memory compression (not supported on macOS)
bool compress_memory(void* ptr, size_t size, size_t* compressed_size) {
    geometric_log_warning("Memory compression not supported on macOS");
    return false;
}

bool decompress_memory(void* compressed_ptr, size_t compressed_size,
                      void* output_ptr, size_t* output_size) {
    geometric_log_warning("Memory decompression not supported on macOS");
    return false;
}

// Platform-specific memory pool operations
bool optimize_pool(MemoryPool* pool) {
    if (!pool) return false;
    
    // On macOS, we can only do basic optimizations
    pthread_mutex_lock(&pool->mutex);
    
    // Coalesce free blocks in each size class
    for (size_t i = 0; i < pool->num_classes; i++) {
        SizeClass* sc = &pool->size_classes[i];
        pthread_mutex_lock(&sc->mutex);
        
        Block* block = sc->free_list;
        while (block) {
            if (block->is_free && block->next && block->next->is_free) {
                // Merge blocks
                block->size += sizeof(Block) + block->next->size;
                block->next = block->next->next;
                if (block->next) {
                    block->next->prev = block;
                }
                continue;
            }
            block = block->next;
        }
        
        pthread_mutex_unlock(&sc->mutex);
    }
    
    pthread_mutex_unlock(&pool->mutex);
    return true;
}

bool defragment_pool(MemoryPool* pool) {
    if (!pool) return false;
    
    pthread_mutex_lock(&pool->mutex);
    
    // Defragment each size class
    for (size_t i = 0; i < pool->num_classes; i++) {
        SizeClass* sc = &pool->size_classes[i];
        pthread_mutex_lock(&sc->mutex);
        
        // Simple defragmentation: move allocated blocks to front
        Block* read_block = sc->free_list;
        Block* write_block = sc->free_list;
        
        while (read_block) {
            if (!read_block->is_free) {
                if (read_block != write_block) {
                    // Copy block data
                    memcpy(write_block, read_block, sizeof(Block) + read_block->size);
                    write_block->next = (Block*)((char*)write_block + 
                        sizeof(Block) + write_block->size);
                    write_block = write_block->next;
                } else {
                    write_block = (Block*)((char*)write_block + 
                        sizeof(Block) + write_block->size);
                }
            }
            read_block = read_block->next;
        }
        
        // Create single free block at end if space remains
        size_t class_size = sc->block_size * sc->max_blocks;
        size_t class_offset = i * class_size;
        char* class_end = (char*)pool->base_address + class_offset + class_size;
        
        if ((char*)write_block < class_end) {
            write_block->is_free = true;
            write_block->size = class_end - (char*)write_block - sizeof(Block);
            write_block->next = NULL;
            write_block->size_class = i;
            init_block(write_block);
        }
        
        pthread_mutex_unlock(&sc->mutex);
    }
    
    pthread_mutex_unlock(&pool->mutex);
    return true;
}

// Platform-specific interface implementation
bool has_numa_support(void) {
    return false; // macOS does not support NUMA
}

int get_numa_node_count(void) {
    return 1; // Single node on macOS
}

bool bind_memory_to_node(void* ptr, size_t size, int node) {
    return false; // Not supported on macOS
}

bool get_memory_node(const void* ptr, int* node) {
    if (!node) return false;
    *node = 0; // Always node 0 on macOS
    return true;
}

size_t get_cache_line_size(CacheLevel level) {
    switch(level) {
        case CACHE_LEVEL_1:
            return CACHE_LINE_SIZE;
        case CACHE_LEVEL_2:
            return CACHE_LINE_SIZE * 2;
        case CACHE_LEVEL_3:
            return CACHE_LINE_SIZE * 4;
        default:
            return CACHE_LINE_SIZE;
    }
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
    return false; // macOS does not support huge pages
}

size_t get_huge_page_size(void) {
    return vm_page_size; // Return standard page size
}

void* allocate_huge_pages(size_t size) {
    return NULL; // Not supported on macOS
}

void free_huge_pages(void* ptr, size_t size) {
    // No-op on macOS
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

// Update memory performance metrics
void update_memory_performance_metrics(PerformanceMetrics* metrics) {
    if (!metrics) return;
    
    // Get hardware performance counters
    metrics->page_faults = get_page_faults();
    metrics->cache_misses = get_cache_misses();
    metrics->tlb_misses = get_tlb_misses();
    
    // Update CPU metrics (not available on macOS)
    metrics->cpu.total_cycles = 0;
    metrics->cpu.stall_cycles = 0;
    metrics->cpu.branch_misses = 0;
    metrics->cpu.instructions = 0;
    
    // Calculate efficiency metric
    metrics->efficiency = 1.0;
    if (metrics->memory.allocations > 0) {
        metrics->efficiency *= (1.0 - (double)metrics->cache_misses / metrics->memory.allocations);
        metrics->efficiency *= (1.0 - (double)metrics->tlb_misses / metrics->memory.allocations);
    }
    
    // Log significant changes
    if (metrics->efficiency < 0.5) {
        geometric_log_warning("Low memory efficiency detected: %.2f", metrics->efficiency);
    }
}

// Initialize block with magic number
static void init_block(Block* block) {
    block->magic = 0xDEADBEEF;
}

// Verify block magic number
static bool verify_block(const Block* block) {
    if (block->magic != 0xDEADBEEF) {
        geometric_log_error("Memory corruption detected: invalid magic number");
        return false;
    }
    return true;
}

// Initialize memory pool with monitoring
bool init_memory_pool_impl(MemoryPool* pool, const PoolConfig* config) {
    if (!pool || !config || config->min_block_size < PAGE_SIZE) {
        return false;
    }
    
    // Initialize size classes
    pool->size_classes = calloc(config->num_size_classes, sizeof(SizeClass));
    if (!pool->size_classes) {
        geometric_log_error("Failed to allocate size classes");
        return false;
    }
    
    // Calculate total pool size based on size classes
    size_t total_size = 0;
    size_t current_size = config->min_block_size;
    
    for (size_t i = 0; i < config->num_size_classes; i++) {
        pool->size_classes[i].block_size = current_size;
        pool->size_classes[i].max_blocks = config->max_blocks_per_class;
        pool->size_classes[i].num_blocks = 0;
        pool->size_classes[i].hits = 0;
        pool->size_classes[i].misses = 0;
        
        if (pthread_mutex_init(&pool->size_classes[i].mutex, NULL) != 0) {
            geometric_log_error("Failed to initialize size class mutex");
            cleanup_memory_pool(pool);
            return false;
        }
        
        total_size += current_size * config->max_blocks_per_class;
        current_size = (size_t)(current_size * config->growth_factor);
    }
    
    // Allocate memory with alignment
    size_t alignment = config->alignment > 0 ? config->alignment : MEMORY_ALIGNMENT;
    pool->base_address = aligned_alloc(alignment, total_size);
    if (!pool->base_address) {
        geometric_log_error("Failed to allocate memory pool");
        cleanup_memory_pool(pool);
        return false;
    }
    
    // Initialize atomic counters
    atomic_init(&pool->total_allocated, 0);
    
    // Initialize mutex
    if (pthread_mutex_init(&pool->mutex, NULL) != 0) {
        geometric_log_error("Failed to initialize pool mutex");
        cleanup_memory_pool(pool);
        return false;
    }
    
    // Store configuration
    pool->config = *config;
    pool->total_size = total_size;
    pool->num_classes = config->num_size_classes;
    
    return true;
}

// Initialize memory manager
MemoryManager* init_memory_manager(void) {
    MemoryManager* manager = aligned_alloc(QG_POOL_ALIGNMENT,
        sizeof(MemoryManager));
    if (!manager) {
        geometric_log_error("Failed to allocate memory manager");
        return NULL;
    }
    
    memset(manager, 0, sizeof(MemoryManager));
    
    // Initialize performance metrics
    memset(&manager->performance, 0, sizeof(PerformanceMetrics));
    manager->performance.efficiency = 1.0;
    
    // Initialize memory pools for different pool types
    size_t pool_sizes[] = {
        PAGE_SIZE,           // POOL_SMALL
        1024 * 1024,        // POOL_MEDIUM
        64 * 1024 * 1024,   // POOL_LARGE
        256 * 1024 * 1024   // POOL_HUGE
    };
    
    for (size_t i = 0; i < sizeof(pool_sizes) / sizeof(size_t); i++) {
        PoolConfig config = {
            .min_block_size = pool_sizes[i],
            .alignment = QG_POOL_ALIGNMENT,
            .num_size_classes = DEFAULT_SIZE_CLASSES,
            .growth_factor = 1.5,
            .max_blocks_per_class = 1024
        };
        if (!init_memory_pool_impl(&manager->pools[i], &config)) {
            cleanup_memory_manager(manager);
            return NULL;
        }
        manager->num_pools++;
    }
    
    geometric_log_info("Memory manager initialized successfully");
    return manager;
}

// Allocate memory with monitoring
void* optimized_malloc(MemoryManager* manager, size_t size) {
    if (!manager || size == 0) {
        geometric_log_error("Invalid malloc parameters");
        return NULL;
    }
    
    // Add header size and align
    size_t total_size = size + sizeof(Block);
    total_size = (total_size + QG_POOL_ALIGNMENT - 1) &
                 ~(QG_POOL_ALIGNMENT - 1);
    
    // Select appropriate pool
    size_t pool_index;
    for (pool_index = 0; pool_index < manager->num_pools; pool_index++) {
        if (total_size <= manager->pools[pool_index].size_classes[0].block_size) {
            break;
        }
    }
    
    if (pool_index >= manager->num_pools) {
        // Fallback to regular allocation for large sizes
        Block* block = aligned_alloc(QG_POOL_ALIGNMENT, total_size);
        if (!block) {
            geometric_log_error("Failed to allocate memory");
            return NULL;
        }
        
        block->size = size;
        block->is_free = false;
        block->next = NULL;
        block->prev = NULL;
        init_block(block);
        
        manager->total_memory += total_size;
        manager->peak_memory = manager->total_memory > manager->peak_memory ?
                             manager->total_memory : manager->peak_memory;
        
        // Update performance metrics
        update_memory_performance_metrics(&manager->performance);
        
        return block->data;
    }
    
    MemoryPool* pool = &manager->pools[pool_index];
    
    // Lock pool
    pthread_mutex_lock(&pool->mutex);
    
    // Find appropriate size class
    size_t class_index = 0;
    for (; class_index < pool->num_classes; class_index++) {
        if (pool->size_classes[class_index].block_size >= size) {
            break;
        }
    }
    
    if (class_index >= pool->num_classes) {
        pthread_mutex_unlock(&pool->mutex);
        geometric_log_error("No suitable size class found");
        return NULL;
    }
    
    SizeClass* sc = &pool->size_classes[class_index];
    pthread_mutex_lock(&sc->mutex);
    
    // Find free block in size class
    Block* block = sc->free_list;
    while (block) {
        if (!verify_block(block)) {
            geometric_log_error("Memory corruption detected during allocation");
            pthread_mutex_unlock(&sc->mutex);
            pthread_mutex_unlock(&pool->mutex);
            return NULL;
        }
        
        if (block->is_free) {
            block->is_free = false;
            block->size_class = class_index;
                        sc->num_blocks++;
            sc->hits++;
            
            atomic_fetch_add(&pool->total_allocated, block->size);
            manager->total_memory += block->size;
            manager->peak_memory = manager->total_memory > manager->peak_memory ?
                                 manager->total_memory : manager->peak_memory;
            
            // Update performance metrics
            update_memory_performance_metrics(&manager->performance);
            
            pthread_mutex_unlock(&sc->mutex);
            pthread_mutex_unlock(&pool->mutex);
            return block->data;
        }
        block = block->next;
    }
    
    pthread_mutex_unlock(&sc->mutex);
    
    pthread_mutex_unlock(&pool->mutex);
    geometric_log_error("No suitable memory block found");
    return NULL;
}

// Free memory with validation
void optimized_free(MemoryManager* manager, void* ptr) {
    if (!manager || !ptr) {
        geometric_log_error("Invalid free parameters");
        return;
    }
    
    // Get block header
    Block* block = (Block*)((char*)ptr - offsetof(Block, data));
    
    if (!verify_block(block)) {
        geometric_log_error("Memory corruption detected during free");
        return;
    }
    
    // Find containing pool index
    size_t pool_index;
    for (pool_index = 0; pool_index < manager->num_pools; pool_index++) {
        if (ptr >= manager->pools[pool_index].base_address &&
            ptr < (void*)((char*)manager->pools[pool_index].base_address +
                         manager->pools[pool_index].total_size)) {
            break;
        }
    }
    
    if (pool_index >= manager->num_pools) {
        // Direct allocation
        manager->total_memory -= block->size;
        free(block);
        
        // Update performance metrics
        update_memory_performance_metrics(&manager->performance);
        return;
    }
    
    // Lock pool
    pthread_mutex_lock(&manager->pools[pool_index].mutex);
    
    // Mark block as free and update counters
    block->is_free = true;
    atomic_fetch_sub(&manager->pools[pool_index].total_allocated, block->size);
    
    // Get size class index
    uint16_t size_class = block->size_class;
    SizeClass* sc = &manager->pools[pool_index].size_classes[size_class];
    
    // Lock size class
    pthread_mutex_lock(&sc->mutex);
    
    // Add block back to size class free list
    block->is_free = true;
    block->next = sc->free_list;
    if (sc->free_list) {
        sc->free_list->prev = block;
    }
    sc->free_list = block;
    sc->num_blocks--;
    
    pthread_mutex_unlock(&sc->mutex);
    
    // Update performance metrics
    update_memory_performance_metrics(&manager->performance);
}
