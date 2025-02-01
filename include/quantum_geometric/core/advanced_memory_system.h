#ifndef ADVANCED_MEMORY_SYSTEM_H
#define ADVANCED_MEMORY_SYSTEM_H

#include <stdbool.h>
#include <stddef.h>

// Memory system types
typedef enum {
    MEM_SYSTEM_STANDARD,     // Standard memory system
    MEM_SYSTEM_POOLED,      // Pooled memory system
    MEM_SYSTEM_QUANTUM,     // Quantum memory system
    MEM_SYSTEM_HYBRID      // Hybrid memory system
} memory_system_type_t;

// Memory allocation strategies
typedef enum {
    ALLOC_STRATEGY_BEST_FIT,    // Best fit allocation
    ALLOC_STRATEGY_FIRST_FIT,   // First fit allocation
    ALLOC_STRATEGY_BUDDY,       // Buddy system allocation
    ALLOC_STRATEGY_SLAB        // Slab allocation
} allocation_strategy_t;

// Memory optimization levels
typedef enum {
    MEM_OPT_NONE,              // No optimization
    MEM_OPT_BASIC,             // Basic optimization
    MEM_OPT_ADVANCED,          // Advanced optimization
    MEM_OPT_AGGRESSIVE        // Aggressive optimization
} optimization_level_t;

// Memory system configuration
typedef struct {
    memory_system_type_t type;          // Memory system type
    allocation_strategy_t strategy;      // Allocation strategy
    optimization_level_t optimization;   // Optimization level
    size_t total_size;                  // Total memory size
    size_t block_size;                  // Memory block size
    size_t alignment;                   // Memory alignment
    bool enable_monitoring;             // Enable monitoring
    bool enable_defragmentation;        // Enable defragmentation
} memory_system_config_t;

// Memory pool configuration
typedef struct {
    size_t pool_size;                   // Pool size
    size_t block_size;                  // Block size
    size_t max_blocks;                  // Maximum blocks
    bool fixed_size;                    // Fixed size blocks
    bool thread_safe;                   // Thread safety
    bool enable_growth;                 // Enable pool growth
} pool_config_t;

// Memory metrics
typedef struct {
    size_t total_allocated;             // Total allocated memory
    size_t total_freed;                 // Total freed memory
    size_t peak_usage;                  // Peak memory usage
    size_t current_usage;               // Current memory usage
    double fragmentation;               // Fragmentation ratio
    size_t allocation_count;            // Number of allocations
} memory_metrics_t;

// Memory block information
typedef struct {
    void* address;                      // Block address
    size_t size;                        // Block size
    bool is_allocated;                  // Allocation status
    size_t alignment;                   // Block alignment
    size_t padding;                     // Alignment padding
    void* pool;                         // Parent pool
} block_info_t;

// Memory defragmentation configuration
typedef struct {
    double threshold;                   // Fragmentation threshold
    size_t max_moves;                   // Maximum block moves
    bool compact_pools;                 // Pool compaction
    bool preserve_order;                // Preserve block order
    bool incremental;                   // Incremental defrag
    size_t batch_size;                  // Defrag batch size
} defrag_config_t;

// Opaque memory system handle
typedef struct advanced_memory_system_t advanced_memory_system_t;

// Core functions
advanced_memory_system_t* create_memory_system(const memory_system_config_t* config);
void destroy_memory_system(advanced_memory_system_t* system);

// Memory allocation functions
void* memory_allocate(advanced_memory_system_t* system,
                     size_t size,
                     size_t alignment);
void memory_free(advanced_memory_system_t* system,
                void* ptr);
void* memory_reallocate(advanced_memory_system_t* system,
                       void* ptr,
                       size_t new_size);

// Memory pool functions
void* create_memory_pool(advanced_memory_system_t* system,
                        const pool_config_t* config);
void destroy_memory_pool(advanced_memory_system_t* system,
                        void* pool);
void* pool_allocate(advanced_memory_system_t* system,
                   void* pool,
                   size_t size);
void pool_free(advanced_memory_system_t* system,
               void* pool,
               void* ptr);

// Memory optimization functions
bool optimize_memory_usage(advanced_memory_system_t* system,
                         optimization_level_t level);
bool optimize_allocation_strategy(advanced_memory_system_t* system,
                                allocation_strategy_t strategy);
bool optimize_pool_configuration(advanced_memory_system_t* system,
                               void* pool,
                               const pool_config_t* config);

// Defragmentation functions
bool start_defragmentation(advanced_memory_system_t* system,
                          const defrag_config_t* config);
bool stop_defragmentation(advanced_memory_system_t* system);
bool is_defragmentation_needed(const advanced_memory_system_t* system,
                             double* fragmentation_level);

// Memory analysis functions
bool get_block_info(const advanced_memory_system_t* system,
                   void* ptr,
                   block_info_t* info);
bool get_memory_metrics(const advanced_memory_system_t* system,
                       memory_metrics_t* metrics);
bool analyze_fragmentation(const advanced_memory_system_t* system,
                          double* fragmentation_level);

// Memory monitoring functions
bool start_memory_monitoring(advanced_memory_system_t* system);
bool stop_memory_monitoring(advanced_memory_system_t* system);
bool reset_memory_metrics(advanced_memory_system_t* system);

// Utility functions
bool validate_memory_block(const advanced_memory_system_t* system,
                         void* ptr);
bool check_memory_corruption(const advanced_memory_system_t* system,
                           void* ptr);
size_t get_allocation_size(const advanced_memory_system_t* system,
                          void* ptr);

#endif // ADVANCED_MEMORY_SYSTEM_H
