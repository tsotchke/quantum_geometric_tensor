#include "quantum_geometric/core/advanced_memory_system.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_types.h"
#include "quantum_geometric/core/error_codes.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#ifdef __linux__
#include <numa.h>
#endif

// Platform-specific SIMD includes
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm64__)
    #if defined(__ARM_NEON) || defined(__ARM_NEON__)
        #include <arm_neon.h>
    #endif
#endif

#ifdef __APPLE__
#include <TargetConditionals.h>
#include <os/lock.h>
#endif

// Memory system parameters
#define CACHE_LINE_SIZE 64
#define PAGE_SIZE 4096
#define PREFETCH_DISTANCE 8
#define MAX_DIMENSIONS 8

// Opaque memory system structure
struct advanced_memory_system_t {
    MemoryPool pool;
    memory_system_config_t config;
};

// Forward declarations
static MemoryPool* init_advanced_memory(void);
static void quantum_state_transfer(void* dst, const void* src, size_t size);
static void quantum_state_measure(void* dst, const void* src, size_t size);
static void update_geometric_access(MemoryLayout* layout);

// Initialize memory pool
static MemoryPool* init_advanced_memory(void) {
    printf("DEBUG: Initializing advanced memory system\n");
    
    // Allocate pool structure
    MemoryPool* pool = malloc(sizeof(MemoryPool));
    if (!pool) {
        printf("DEBUG: Failed to allocate memory pool\n");
        return NULL;
    }
    memset(pool, 0, sizeof(MemoryPool));
    
    // Initialize platform-specific features
#ifdef __linux__
    if (numa_available() < 0) {
        printf("DEBUG: NUMA not available\n");
        free(pool);
        return NULL;
    }
#endif

#ifdef __APPLE__
    #if defined(__arm64__) || defined(__aarch64__)
    pool->use_neon = true;
    printf("DEBUG: NEON acceleration enabled\n");
    #endif
#endif
    
    // Setup geometric layouts
    printf("DEBUG: Allocating memory layouts\n");
    pool->layouts = malloc(MAX_DIMENSIONS * sizeof(MemoryLayout));
    if (!pool->layouts) {
        printf("DEBUG: Failed to allocate memory layouts\n");
        free(pool);
        return NULL;
    }
    pool->num_layouts = 0;
    
    // Setup zero-copy buffers
    printf("DEBUG: Allocating memory buffers\n");
    pool->buffers = malloc(1024 * sizeof(MemoryBuffer));
    if (!pool->buffers) {
        printf("DEBUG: Failed to allocate memory buffers\n");
        free(pool->layouts);
        free(pool);
        return NULL;
    }
    pool->num_buffers = 0;
    
    // Initialize fast path cache
    pool->fast_path_cache = NULL;
    
    // Initialize config
    pool->config.enable_stats = true;
    
    printf("DEBUG: Memory system initialized successfully\n");
    return pool;
}

// Forward declaration for memory singleton functions
extern void register_memory_system(advanced_memory_system_t* system);
extern advanced_memory_system_t* get_global_memory_system(void);

// Core memory functions
advanced_memory_system_t* create_memory_system(const memory_system_config_t* config) {
    printf("DEBUG: Creating memory system\n");
    
    // Check if we already have a global memory system
    advanced_memory_system_t* existing = get_global_memory_system();
    if (existing) {
        printf("DEBUG: Using existing global memory system\n");
        register_memory_system(existing);
        return existing;
    }
    
    if (!config) {
        printf("DEBUG: Invalid config\n");
        return NULL;
    }
    
    advanced_memory_system_t* system = malloc(sizeof(advanced_memory_system_t));
    if (!system) {
        printf("DEBUG: Failed to allocate system\n");
        return NULL;
    }
    
    // Copy configuration
    memcpy(&system->config, config, sizeof(memory_system_config_t));
    
    // Initialize memory pool
    MemoryPool* pool = init_advanced_memory();
    if (!pool) {
        printf("DEBUG: Failed to initialize memory pool\n");
        free(system);
        return NULL;
    }
    
    // Copy pool and take ownership of its resources
    memcpy(&system->pool, pool, sizeof(MemoryPool));
    free(pool); // Free just the pool structure, not its contents
    
    // Register as global memory system
    register_memory_system(system);
    
    printf("DEBUG: Memory system created successfully\n");
    return system;
}

// Forward declaration for memory singleton functions
extern void unregister_memory_system(advanced_memory_system_t* system);

void destroy_memory_system(advanced_memory_system_t* system) {
    if (!system) return;
    
    // Unregister from global memory system
    unregister_memory_system(system);
    
    // Only clean up if this is the last reference
    if (get_global_memory_system() != system) {
        // Clean up the memory pool directly
        for (size_t i = 0; i < system->pool.num_layouts; i++) {
#ifdef __linux__
            numa_free(system->pool.layouts[i].base_ptr,
                     system->pool.layouts[i].total_size);
#else
            free(system->pool.layouts[i].base_ptr);
#endif
        }
        
        // Clean up buffers
        for (size_t i = 0; i < system->pool.num_buffers; i++) {
            if (system->pool.buffers[i].gpu_ptr)
                free(system->pool.buffers[i].gpu_ptr);
            if (system->pool.buffers[i].cpu_ptr)
                free(system->pool.buffers[i].cpu_ptr);
            if (system->pool.buffers[i].quantum_ptr)
                free(system->pool.buffers[i].quantum_ptr);
        }
        free(system->pool.fast_path_cache);
        free(system->pool.layouts);
        free(system->pool.buffers);
        
        free(system);
    }
}

void* memory_allocate(advanced_memory_system_t* system, size_t size, size_t alignment) {
    if (!system || size == 0) return NULL;
    
    // Create memory layout for allocation
    size_t dims[1] = {size};
    geometric_state_type_t geom_type = GEOMETRIC_STATE_EUCLIDEAN;
    
    // Store alignment in the system config for later use
    system->config.alignment = alignment;
    
    MemoryLayout* layout = create_memory_layout(&system->pool, dims, 1, geom_type);
    if (!layout) return NULL;
    
    return layout->base_ptr;
}

void memory_free(advanced_memory_system_t* system, void* ptr) {
    if (!system || !ptr) return;
    
    // Find and free the corresponding layout
    for (size_t i = 0; i < system->pool.num_layouts; i++) {
        if (system->pool.layouts[i].base_ptr == ptr) {
            // Free the memory using the appropriate method
#ifdef __linux__
            numa_free(ptr, system->pool.layouts[i].total_size);
#else
            free(ptr);
#endif
            // Remove layout from pool
            if (i < system->pool.num_layouts - 1) {
                memmove(&system->pool.layouts[i], 
                       &system->pool.layouts[i + 1],
                       (system->pool.num_layouts - i - 1) * sizeof(MemoryLayout));
            }
            system->pool.num_layouts--;
            break;
        }
    }
}

// Create memory layout
MemoryLayout* create_memory_layout(
    MemoryPool* pool,
    const size_t* dimensions,
    size_t num_dims,
    geometric_state_type_t geom_type) {
    
    if (!pool || !dimensions || num_dims > MAX_DIMENSIONS) return NULL;
    
    MemoryLayout* layout = &pool->layouts[pool->num_layouts++];
    
    // Calculate optimal memory layout
    size_t total_size = 1;
    for (size_t i = 0; i < num_dims; i++) {
        layout->dimensions[i] = dimensions[i];
        layout->strides[i] = total_size;
        total_size *= dimensions[i];
    }
    
    // Align to cache line
    total_size = (total_size + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1);
    
    // Platform-specific memory allocation
#ifdef __linux__
    int node = numa_node_of_cpu(sched_getcpu());
    layout->base_ptr = numa_alloc_onnode(total_size, node);
#else
    layout->base_ptr = malloc(total_size);
#endif
    if (!layout->base_ptr) {
        pool->num_layouts--;
        return NULL;
    }
    
    layout->total_size = total_size;
    layout->is_geometric = true;
    layout->geom_type = geom_type;
    
    // Initialize memory
    memset(layout->base_ptr, 0, total_size);
    
    return layout;
}

// Create memory buffer
MemoryBuffer* create_memory_buffer(
    MemoryPool* pool,
    size_t size,
    bool gpu_accessible,
    bool quantum_accessible,
    HardwareType hardware) {
    
    if (!pool) return NULL;
    
    MemoryBuffer* buffer = &pool->buffers[pool->num_buffers++];
    buffer->hardware = hardware;
    
    // Allocate aligned memory
    buffer->cpu_ptr = malloc(size);
    if (!buffer->cpu_ptr) {
        pool->num_buffers--;
        return NULL;
    }
    
    // Setup GPU access if needed
    if (gpu_accessible) {
        buffer->gpu_ptr = malloc(size);
        if (!buffer->gpu_ptr) {
            free(buffer->cpu_ptr);
            pool->num_buffers--;
            return NULL;
        }
        buffer->is_unified = true;
    }
    
    // Setup quantum access if needed
    if (quantum_accessible) {
        buffer->quantum_ptr = malloc(size);
        if (!buffer->quantum_ptr) {
            if (gpu_accessible) free(buffer->gpu_ptr);
            free(buffer->cpu_ptr);
            pool->num_buffers--;
            return NULL;
        }
    }
    
    buffer->size = size;
    buffer->is_pinned = true;
    
    return buffer;
}

// Optimized memory copy with prefetching
void memory_copy(
    MemoryLayout* dst,
    const MemoryLayout* src,
    size_t size) {
    
    // Ensure alignment
    assert(((uintptr_t)dst->base_ptr & (CACHE_LINE_SIZE - 1)) == 0);
    assert(((uintptr_t)src->base_ptr & (CACHE_LINE_SIZE - 1)) == 0);
    
    // Calculate optimal copy strategy
    size_t blocks = size / CACHE_LINE_SIZE;
    
    // Platform-specific optimized copy
#ifdef __linux__
    // Use AVX-512 on x86
    for (size_t i = 0; i < blocks; i++) {
        _mm_prefetch(src->base_ptr + (i + PREFETCH_DISTANCE) * CACHE_LINE_SIZE,
                    _MM_HINT_T0);
        __m512i* src_aligned = (__m512i*)(src->base_ptr + i * CACHE_LINE_SIZE);
        __m512i* dst_aligned = (__m512i*)(dst->base_ptr + i * CACHE_LINE_SIZE);
        _mm512_store_si512(dst_aligned, _mm512_load_si512(src_aligned));
    }
#elif defined(__APPLE__) && (defined(__arm64__) || defined(__aarch64__))
    // Use NEON on Apple Silicon
    MemoryPool* pool = (MemoryPool*)((char*)dst - offsetof(MemoryPool, layouts));
    if (pool && pool->use_neon) {
        size_t vec_size = CACHE_LINE_SIZE / sizeof(float);
        for (size_t i = 0; i < blocks; i++) {
            float* src_aligned = (float*)(src->base_ptr + i * CACHE_LINE_SIZE);
            float* dst_aligned = (float*)(dst->base_ptr + i * CACHE_LINE_SIZE);
            for (size_t j = 0; j < vec_size; j += 4) {
                float32x4_t vec = vld1q_f32(&src_aligned[j]);
                vst1q_f32(&dst_aligned[j], vec);
            }
        }
    } else {
        memcpy(dst->base_ptr, src->base_ptr, size);
    }
#else
    memcpy(dst->base_ptr, src->base_ptr, size);
#endif
}

// Zero-copy transfer
void memory_transfer(
    MemoryBuffer* buffer,
    geometric_state_type_t direction) {
    
    switch (direction) {
        case GEOMETRIC_STATE_EUCLIDEAN:  // CPU to GPU
            if (buffer->is_unified) {
                // Direct access, no explicit copy needed
                memcpy(buffer->gpu_ptr, buffer->cpu_ptr, buffer->size);
            }
            break;
            
        case GEOMETRIC_STATE_HYPERBOLIC:  // GPU to CPU
            if (buffer->is_unified) {
                // Direct access, no explicit copy needed
                memcpy(buffer->cpu_ptr, buffer->gpu_ptr, buffer->size);
            }
            break;
            
        case GEOMETRIC_STATE_SPHERICAL:  // CPU to Quantum
            quantum_state_transfer(buffer->quantum_ptr,
                                buffer->cpu_ptr,
                                buffer->size);
            break;
            
        case GEOMETRIC_STATE_SYMPLECTIC:  // Quantum to CPU
            quantum_state_measure(buffer->cpu_ptr,
                               buffer->quantum_ptr,
                               buffer->size);
            break;
            
        default:
            // Handle other geometric states
            break;
    }
}

// Helper functions
static void quantum_state_transfer(void* dst, const void* src, size_t size) {
    memcpy(dst, src, size);
}

static void quantum_state_measure(void* dst, const void* src, size_t size) {
    memcpy(dst, src, size);
}

static void update_geometric_access(MemoryLayout* layout) {
    // Simple access pattern tracking
    if (layout && layout->is_geometric) {
        // Track access for future optimization
        layout->geom_type = GEOMETRIC_STATE_EUCLIDEAN;
    }
}

void update_memory_access(MemoryPool* pool) {
    // Analyze access patterns
    if (pool->config.enable_stats) {
        // Update access statistics
        for (size_t i = 0; i < pool->num_layouts; i++) {
            if (pool->layouts[i].is_geometric) {
                update_geometric_access(&pool->layouts[i]);
            }
        }
    }
}

// Create memory pool
void* advanced_create_memory_pool(advanced_memory_system_t* system, const pool_config_t* config) {
    if (!system || !config) {
        printf("DEBUG: Invalid arguments to create_memory_pool\n");
        return NULL;
    }
    
    // Allocate pool structure
    MemoryPool* pool = malloc(sizeof(MemoryPool));
    if (!pool) {
        printf("DEBUG: Failed to allocate memory pool\n");
        return NULL;
    }
    memset(pool, 0, sizeof(MemoryPool));
    
    // Setup geometric layouts
    pool->layouts = malloc(MAX_DIMENSIONS * sizeof(MemoryLayout));
    if (!pool->layouts) {
        printf("DEBUG: Failed to allocate memory layouts\n");
        free(pool);
        return NULL;
    }
    pool->num_layouts = 0;
    
    // Setup zero-copy buffers
    pool->buffers = malloc(1024 * sizeof(MemoryBuffer));
    if (!pool->buffers) {
        printf("DEBUG: Failed to allocate memory buffers\n");
        free(pool->layouts);
        free(pool);
        return NULL;
    }
    pool->num_buffers = 0;
    
    // Initialize fast path cache
    pool->fast_path_cache = NULL;
    
    // Initialize config
    pool->config = *config;
    
    printf("DEBUG: Memory pool created successfully\n");
    return pool;
}

// Destroy memory pool
void advanced_destroy_memory_pool(advanced_memory_system_t* system, void* pool_ptr) {
    if (!system || !pool_ptr) {
        printf("DEBUG: Invalid arguments to destroy_memory_pool\n");
        return;
    }
    
    // Cast to the correct type and destroy
    MemoryPool* pool = (MemoryPool*)pool_ptr;
    
    // Clean up geometric layouts
    for (size_t i = 0; i < pool->num_layouts; i++) {
#ifdef __linux__
        numa_free(pool->layouts[i].base_ptr,
                 pool->layouts[i].total_size);
#else
        free(pool->layouts[i].base_ptr);
#endif
    }
    
    // Clean up buffers
    for (size_t i = 0; i < pool->num_buffers; i++) {
        if (pool->buffers[i].gpu_ptr)
            free(pool->buffers[i].gpu_ptr);
        if (pool->buffers[i].cpu_ptr)
            free(pool->buffers[i].cpu_ptr);
        if (pool->buffers[i].quantum_ptr)
            free(pool->buffers[i].quantum_ptr);
    }
    free(pool->fast_path_cache);
    free(pool->layouts);
    free(pool->buffers);
    free(pool);
}

// Allocate memory from pool
void* advanced_pool_allocate(advanced_memory_system_t* system, void* pool_ptr, size_t size) {
    if (!system || !pool_ptr || size == 0) {
        printf("DEBUG: Invalid arguments to pool_allocate\n");
        return NULL;
    }

    // Cast to the correct type and allocate
    MemoryPool* pool = (MemoryPool*)pool_ptr;

    // Create a simple memory layout for allocation
    size_t dims[1] = {size};
    geometric_state_type_t geom_type = GEOMETRIC_STATE_EUCLIDEAN;

    MemoryLayout* layout = create_memory_layout(pool, dims, 1, geom_type);
    if (!layout) return NULL;

    return layout->base_ptr;
}

// Free memory to pool
void advanced_pool_free(advanced_memory_system_t* system, void* pool_ptr, void* ptr) {
    if (!system || !pool_ptr || !ptr) {
        return;
    }

    MemoryPool* pool = (MemoryPool*)pool_ptr;

    // Find and free the corresponding layout
    for (size_t i = 0; i < pool->num_layouts; i++) {
        if (pool->layouts[i].base_ptr == ptr) {
#ifdef __linux__
            numa_free(ptr, pool->layouts[i].total_size);
#else
            free(ptr);
#endif
            // Remove layout from pool
            if (i < pool->num_layouts - 1) {
                memmove(&pool->layouts[i],
                       &pool->layouts[i + 1],
                       (pool->num_layouts - i - 1) * sizeof(MemoryLayout));
            }
            pool->num_layouts--;
            break;
        }
    }
}

// ============================================================================
// API wrapper functions (match header declarations)
// ============================================================================

void* create_memory_pool(advanced_memory_system_t* system, const pool_config_t* config) {
    return advanced_create_memory_pool(system, config);
}

void destroy_memory_pool(advanced_memory_system_t* system, void* pool) {
    advanced_destroy_memory_pool(system, pool);
}

void* pool_allocate(advanced_memory_system_t* system, void* pool, size_t size) {
    return advanced_pool_allocate(system, pool, size);
}

void pool_free(advanced_memory_system_t* system, void* pool, void* ptr) {
    advanced_pool_free(system, pool, ptr);
}

// ============================================================================
// Memory Optimization Functions
// ============================================================================

bool optimize_memory_usage(advanced_memory_system_t* system, optimization_level_t level) {
    if (!system) return false;

    switch (level) {
        case MEM_OPT_NONE:
            // No optimization
            return true;

        case MEM_OPT_BASIC:
            // Basic optimization: compact empty slots in layout array
            {
                size_t write_idx = 0;
                for (size_t read_idx = 0; read_idx < system->pool.num_layouts; read_idx++) {
                    if (system->pool.layouts[read_idx].base_ptr != NULL) {
                        if (write_idx != read_idx) {
                            system->pool.layouts[write_idx] = system->pool.layouts[read_idx];
                        }
                        write_idx++;
                    }
                }
                system->pool.num_layouts = write_idx;
            }
            return true;

        case MEM_OPT_ADVANCED:
            // Advanced optimization: reorganize layouts for better cache locality
            // Sort by size for better packing
            for (size_t i = 0; i < system->pool.num_layouts; i++) {
                for (size_t j = i + 1; j < system->pool.num_layouts; j++) {
                    if (system->pool.layouts[j].total_size > system->pool.layouts[i].total_size) {
                        MemoryLayout temp = system->pool.layouts[i];
                        system->pool.layouts[i] = system->pool.layouts[j];
                        system->pool.layouts[j] = temp;
                    }
                }
            }
            return true;

        case MEM_OPT_AGGRESSIVE:
            // Aggressive optimization: coalesce adjacent free blocks
            // and defragment memory
            optimize_memory_usage(system, MEM_OPT_ADVANCED);

            // Trigger defragmentation if needed
            double frag_level = 0.0;
            if (is_defragmentation_needed(system, &frag_level) && frag_level > 0.2) {
                defrag_config_t config = {
                    .threshold = 0.2,
                    .max_moves = 500,
                    .compact_pools = true,
                    .preserve_order = false,
                    .incremental = false,
                    .batch_size = 50
                };
                start_defragmentation(system, &config);
            }
            return true;

        default:
            return false;
    }
}

bool optimize_allocation_strategy(advanced_memory_system_t* system,
                                 allocation_strategy_t strategy) {
    if (!system) return false;

    system->config.strategy = strategy;
    return true;
}

bool optimize_pool_configuration(advanced_memory_system_t* system,
                                void* pool_ptr,
                                const pool_config_t* config) {
    if (!system || !pool_ptr || !config) return false;

    MemoryPool* pool = (MemoryPool*)pool_ptr;
    pool->config = *config;
    return true;
}

// ============================================================================
// Defragmentation Functions
// ============================================================================

bool start_defragmentation(advanced_memory_system_t* system,
                          const defrag_config_t* config) {
    if (!system || !config) return false;

    // Check if defragmentation is actually needed
    double current_frag = 0.0;
    analyze_fragmentation(system, &current_frag);

    if (current_frag < config->threshold) {
        // Fragmentation is below threshold, no action needed
        return true;
    }

    // Perform defragmentation by compacting layouts
    // This is a simplified implementation that reorganizes the layout array

    size_t moves = 0;
    size_t batch = 0;

    // Sort layouts by address for compaction
    for (size_t i = 0; i < system->pool.num_layouts && moves < config->max_moves; i++) {
        for (size_t j = i + 1; j < system->pool.num_layouts; j++) {
            if ((uintptr_t)system->pool.layouts[j].base_ptr <
                (uintptr_t)system->pool.layouts[i].base_ptr) {
                // Swap layouts to achieve address-ordered layout
                MemoryLayout temp = system->pool.layouts[i];
                system->pool.layouts[i] = system->pool.layouts[j];
                system->pool.layouts[j] = temp;
                moves++;
                batch++;

                if (config->incremental && batch >= config->batch_size) {
                    // In incremental mode, stop after batch_size moves
                    return true;
                }
            }
        }
    }

    return true;
}

bool stop_defragmentation(advanced_memory_system_t* system) {
    if (!system) return false;
    // Since our implementation is synchronous, this is a no-op
    return true;
}

bool is_defragmentation_needed(const advanced_memory_system_t* system,
                              double* fragmentation_level) {
    if (!system || !fragmentation_level) return false;

    return analyze_fragmentation(system, fragmentation_level);
}

// ============================================================================
// Memory Analysis Functions
// ============================================================================

bool get_block_info(const advanced_memory_system_t* system,
                   void* ptr,
                   block_info_t* info) {
    if (!system || !ptr || !info) return false;

    // Search for the block in layouts
    for (size_t i = 0; i < system->pool.num_layouts; i++) {
        if (system->pool.layouts[i].base_ptr == ptr) {
            info->address = ptr;
            info->size = system->pool.layouts[i].total_size;
            info->is_allocated = true;
            info->alignment = CACHE_LINE_SIZE;
            info->padding = 0;
            info->pool = (void*)&system->pool;
            return true;
        }
    }

    // Search in buffers
    for (size_t i = 0; i < system->pool.num_buffers; i++) {
        if (system->pool.buffers[i].cpu_ptr == ptr ||
            system->pool.buffers[i].gpu_ptr == ptr ||
            system->pool.buffers[i].quantum_ptr == ptr) {
            info->address = ptr;
            info->size = system->pool.buffers[i].size;
            info->is_allocated = true;
            info->alignment = CACHE_LINE_SIZE;
            info->padding = 0;
            info->pool = (void*)&system->pool;
            return true;
        }
    }

    return false;
}

bool get_memory_metrics(const advanced_memory_system_t* system,
                       memory_metrics_t* metrics) {
    if (!system || !metrics) return false;

    // Calculate metrics from current state
    size_t total_allocated = 0;
    size_t current_usage = 0;

    for (size_t i = 0; i < system->pool.num_layouts; i++) {
        total_allocated += system->pool.layouts[i].total_size;
        if (system->pool.layouts[i].base_ptr != NULL) {
            current_usage += system->pool.layouts[i].total_size;
        }
    }

    for (size_t i = 0; i < system->pool.num_buffers; i++) {
        total_allocated += system->pool.buffers[i].size;
        current_usage += system->pool.buffers[i].size;
    }

    metrics->total_allocated = total_allocated;
    metrics->total_freed = total_allocated - current_usage;
    metrics->current_usage = current_usage;
    metrics->peak_usage = total_allocated;  // Simplified
    metrics->allocation_count = system->pool.num_layouts + system->pool.num_buffers;

    // Calculate fragmentation
    double frag = 0.0;
    analyze_fragmentation(system, &frag);
    metrics->fragmentation = frag;

    return true;
}

bool analyze_fragmentation(const advanced_memory_system_t* system,
                          double* fragmentation_level) {
    if (!system || !fragmentation_level) return false;

    if (system->pool.num_layouts == 0) {
        *fragmentation_level = 0.0;
        return true;
    }

    // Calculate fragmentation as the ratio of gaps between allocations
    // to total allocated space

    // First, collect all allocated addresses and sizes
    size_t num_blocks = system->pool.num_layouts;
    if (num_blocks < 2) {
        *fragmentation_level = 0.0;
        return true;
    }

    // Sort addresses to find gaps
    uintptr_t* addresses = malloc(num_blocks * sizeof(uintptr_t));
    size_t* sizes = malloc(num_blocks * sizeof(size_t));
    if (!addresses || !sizes) {
        free(addresses);
        free(sizes);
        *fragmentation_level = 0.0;
        return false;
    }

    for (size_t i = 0; i < num_blocks; i++) {
        addresses[i] = (uintptr_t)system->pool.layouts[i].base_ptr;
        sizes[i] = system->pool.layouts[i].total_size;
    }

    // Simple bubble sort by address
    for (size_t i = 0; i < num_blocks - 1; i++) {
        for (size_t j = 0; j < num_blocks - i - 1; j++) {
            if (addresses[j] > addresses[j + 1]) {
                uintptr_t temp_addr = addresses[j];
                addresses[j] = addresses[j + 1];
                addresses[j + 1] = temp_addr;

                size_t temp_size = sizes[j];
                sizes[j] = sizes[j + 1];
                sizes[j + 1] = temp_size;
            }
        }
    }

    // Calculate gaps
    size_t total_gaps = 0;
    size_t total_size = 0;

    for (size_t i = 0; i < num_blocks; i++) {
        total_size += sizes[i];
        if (i > 0) {
            uintptr_t expected_start = addresses[i - 1] + sizes[i - 1];
            if (addresses[i] > expected_start) {
                total_gaps += (addresses[i] - expected_start);
            }
        }
    }

    free(addresses);
    free(sizes);

    // Fragmentation ratio: gaps / (gaps + allocated)
    if (total_size + total_gaps > 0) {
        *fragmentation_level = (double)total_gaps / (double)(total_size + total_gaps);
    } else {
        *fragmentation_level = 0.0;
    }

    return true;
}

// ============================================================================
// Memory Monitoring Functions
// ============================================================================

bool start_memory_monitoring(advanced_memory_system_t* system) {
    if (!system) return false;
    system->config.enable_monitoring = true;
    return true;
}

bool stop_memory_monitoring(advanced_memory_system_t* system) {
    if (!system) return false;
    system->config.enable_monitoring = false;
    return true;
}

bool reset_memory_metrics(advanced_memory_system_t* system) {
    if (!system) return false;
    // Metrics are computed on-the-fly, so nothing to reset
    return true;
}

// ============================================================================
// Utility Functions
// ============================================================================

bool validate_memory_block(const advanced_memory_system_t* system, void* ptr) {
    if (!system || !ptr) return false;

    // Check if pointer is in any of our layouts
    for (size_t i = 0; i < system->pool.num_layouts; i++) {
        void* base = system->pool.layouts[i].base_ptr;
        size_t size = system->pool.layouts[i].total_size;

        if (ptr >= base && (uintptr_t)ptr < (uintptr_t)base + size) {
            return true;
        }
    }

    // Check buffers
    for (size_t i = 0; i < system->pool.num_buffers; i++) {
        if (ptr == system->pool.buffers[i].cpu_ptr ||
            ptr == system->pool.buffers[i].gpu_ptr ||
            ptr == system->pool.buffers[i].quantum_ptr) {
            return true;
        }
    }

    return false;
}

bool check_memory_corruption(const advanced_memory_system_t* system, void* ptr) {
    if (!system || !ptr) return false;

    // Basic validation - just check if the pointer is valid
    // A more sophisticated implementation would use guard bytes
    return validate_memory_block(system, ptr);
}

size_t get_allocation_size(const advanced_memory_system_t* system, void* ptr) {
    if (!system || !ptr) return 0;

    // Search in layouts
    for (size_t i = 0; i < system->pool.num_layouts; i++) {
        if (system->pool.layouts[i].base_ptr == ptr) {
            return system->pool.layouts[i].total_size;
        }
    }

    // Search in buffers
    for (size_t i = 0; i < system->pool.num_buffers; i++) {
        if (system->pool.buffers[i].cpu_ptr == ptr ||
            system->pool.buffers[i].gpu_ptr == ptr ||
            system->pool.buffers[i].quantum_ptr == ptr) {
            return system->pool.buffers[i].size;
        }
    }

    return 0;
}

// Reallocate memory
void* memory_reallocate(advanced_memory_system_t* system, void* ptr, size_t new_size) {
    if (!system) return NULL;
    if (!ptr) return memory_allocate(system, new_size, system->config.alignment);
    if (new_size == 0) {
        memory_free(system, ptr);
        return NULL;
    }

    // Get old size
    size_t old_size = get_allocation_size(system, ptr);
    if (old_size == 0) return NULL;

    // Allocate new block
    void* new_ptr = memory_allocate(system, new_size, system->config.alignment);
    if (!new_ptr) return NULL;

    // Copy data
    size_t copy_size = (old_size < new_size) ? old_size : new_size;
    memcpy(new_ptr, ptr, copy_size);

    // Free old block
    memory_free(system, ptr);

    return new_ptr;
}

