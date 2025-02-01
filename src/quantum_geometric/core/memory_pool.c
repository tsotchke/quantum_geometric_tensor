#include "quantum_geometric/core/memory_pool.h"
#include "quantum_geometric/core/quantum_geometric_logging.h"
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>

// Magic number for validation
#define BLOCK_MAGIC 0xDEADBEEF

// Thread-local storage for thread cache
static __thread ThreadCache thread_cache = {0};
static __thread bool thread_cache_initialized = false;

// Size class lookup table
static _Atomic(size_t) size_class_table[QG_NUM_SIZE_CLASSES];
static atomic_bool size_classes_initialized = false;

// Initialize thread cache
static void init_thread_cache(void) {
    if (!thread_cache_initialized) {
        memset(&thread_cache, 0, sizeof(ThreadCache));
        thread_cache_initialized = true;
    }
}

// Initialize size class table with quantum-optimized progression
static void init_size_classes(void) {
    bool expected = false;
    if (!atomic_compare_exchange_strong(&size_classes_initialized, &expected, true)) {
        return; // Already initialized
    }

    size_t size = QG_MIN_BLOCK_SIZE;
    for (int i = 0; i < QG_NUM_SIZE_CLASSES; i++) {
        // Ensure quantum tensor alignment
        size = (size + QG_POOL_ALIGNMENT - 1) & ~(QG_POOL_ALIGNMENT - 1);
        atomic_store_explicit(&size_class_table[i], size, memory_order_release);
        
        // Quantum tensor optimized progression
        if (size < 1024) {
            size *= 2;  // Double sizes up to 1KB for small tensors
        } else if (size < 16384) {
            size *= 2; // Double sizes up to 16KB for medium tensors
        } else if (size < 262144) {
            size *= 2; // Double sizes up to 256KB for large tensors
        } else if (size < 4194304) {
            size *= 2; // Double sizes up to 4MB for huge tensors
        } else {
            size *= 2; // Keep doubling for massive tensors
        }
    }
}

// Size class lookup table with power-of-2 optimization
static const uint8_t size_class_lookup[256] = {0};  // Will be initialized
static atomic_bool lookup_table_initialized = false;

// Initialize lookup table for faster size class determination
static void init_lookup_table(void) {
    bool expected = false;
    if (!atomic_compare_exchange_strong(&lookup_table_initialized, &expected, true)) {
        return;
    }
    
    size_t current_class = 0;
    size_t current_size = atomic_load_explicit(&size_class_table[0], memory_order_acquire);
    
    for (int i = 0; i < 256; i++) {
        while (current_class < QG_NUM_SIZE_CLASSES - 1 && 
               i * 32 >= atomic_load_explicit(&size_class_table[current_class], memory_order_acquire)) {
            current_class++;
        }
        ((uint8_t*)size_class_lookup)[i] = current_class;
    }
    
    atomic_thread_fence(memory_order_release);
}

// Optimized size class lookup using power-of-2 table
static uint16_t get_size_class(size_t size) {
    if (!atomic_load_explicit(&lookup_table_initialized, memory_order_acquire)) {
        init_lookup_table();
    }
    
    if (size <= 8192) {  // Handle common case quickly
        return size_class_lookup[(size + 31) >> 5];
    }
    
    // Fall back to binary search for large sizes
    uint16_t left = size_class_lookup[255];
    uint16_t right = QG_NUM_SIZE_CLASSES - 1;
    
    while (left < right) {
        uint16_t mid = left + ((right - left) >> 1);
        if (atomic_load_explicit(&size_class_table[mid], memory_order_acquire) < size) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    
    return left;
}

// Initialize memory pool
MemoryPool* init_memory_pool(const PoolConfig* config) {
    if (!config) {
        geometric_log_error("Null config passed to init_memory_pool");
        return NULL;
    }

    // Initialize size classes if needed
    init_size_classes();

    // Allocate and align pool structure
    MemoryPool* pool = NULL;
    if (posix_memalign((void**)&pool, QG_POOL_ALIGNMENT, sizeof(MemoryPool)) != 0) {
        geometric_log_error("Failed to allocate aligned memory for pool");
        return NULL;
    }
    memset(pool, 0, sizeof(MemoryPool));

    // Allocate and align size class array
    void* size_classes = NULL;
    if (posix_memalign(&size_classes, QG_POOL_ALIGNMENT, config->num_size_classes * sizeof(SizeClass)) != 0) {
        geometric_log_error("Failed to allocate size classes");
        free(pool);
        return NULL;
    }
    pool->size_classes = size_classes;
    memset(pool->size_classes, 0, config->num_size_classes * sizeof(SizeClass));
    
    // Initialize mutex attributes
    pthread_mutexattr_t attr;
    if (pthread_mutexattr_init(&attr) != 0) {
        geometric_log_error("Failed to initialize mutex attributes");
        free(size_classes);
        free(pool);
        return NULL;
    }
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);

    // Initialize size class mutexes
    for (int i = 0; i < config->num_size_classes; i++) {
        if (pthread_mutex_init(&pool->size_classes[i].mutex, &attr) != 0) {
            geometric_log_error("Failed to initialize mutex for size class %d", i);
            // Clean up previously initialized mutexes
            for (int j = 0; j < i; j++) {
                pthread_mutex_destroy(&pool->size_classes[j].mutex);
            }
            pthread_mutexattr_destroy(&attr);
            free(size_classes);
            free(pool);
            return NULL;
        }
        pool->size_classes[i].block_size = atomic_load_explicit(&size_class_table[i], memory_order_acquire);
        pool->size_classes[i].max_blocks = config->max_blocks_per_class;
    }

    // Initialize pool mutex
    if (pthread_mutex_init(&pool->mutex, &attr) != 0) {
        geometric_log_error("Failed to initialize pool mutex");
        for (int i = 0; i < config->num_size_classes; i++) {
            pthread_mutex_destroy(&pool->size_classes[i].mutex);
        }
        pthread_mutexattr_destroy(&attr);
        free(size_classes);
        free(pool);
        return NULL;
    }

    pthread_mutexattr_destroy(&attr);
    
    pool->config = *config;
    pool->num_classes = QG_NUM_SIZE_CLASSES;
    
    // Initialize atomic fields
    atomic_init(&pool->total_allocated, 0);
    atomic_init(&pool->peak_allocated, 0);
    atomic_init(&pool->num_allocations, 0);
    
    geometric_log_info("Memory pool initialized with %d size classes", QG_NUM_SIZE_CLASSES);
    return pool;
}

// Allocate memory from thread cache
static void* allocate_from_cache(MemoryPool* pool, size_t size) {
    if (!pool->config.thread_cache_size) {
        return NULL;
    }

    init_thread_cache();
    uint16_t size_class = get_size_class(size);
    
    ThreadCacheEntry* entry = thread_cache.entries[size_class];
    if (entry) {
        void* ptr = entry->ptr;
        thread_cache.entries[size_class] = entry->next;
        thread_cache.count[size_class]--;
        free(entry);
        return ptr;
    }
    
    return NULL;
}

// Allocate memory from size class
static void* allocate_from_class(MemoryPool* pool, uint16_t size_class) {
    SizeClass* sc = &pool->size_classes[size_class];
    
    pthread_mutex_lock(&sc->mutex);
    
    Block* block = sc->free_list;
    if (block) {
        sc->free_list = block->next;
        sc->num_blocks--;
        sc->hits++;
        pthread_mutex_unlock(&sc->mutex);
        
        block->is_free = false;
        return block->data;
    }
    
    // Allocate new block with alignment
    size_t total_size = sizeof(Block) + sc->block_size;
    total_size = (total_size + QG_POOL_ALIGNMENT - 1) & ~(QG_POOL_ALIGNMENT - 1);
    
    void* block_ptr = NULL;
    if (posix_memalign(&block_ptr, QG_POOL_ALIGNMENT, total_size) != 0) {
        pthread_mutex_unlock(&sc->mutex);
        return NULL;
    }
    block = block_ptr;
    
    // Use SIMD to zero-initialize the block
    #ifdef __AVX512F__
    char* ptr = (char*)block;
    size_t simd_size = total_size & ~63;
    __m512i zero = _mm512_setzero_si512();
    
    for (size_t i = 0; i < simd_size; i += 64) {
        _mm512_store_si512((__m512i*)(ptr + i), zero);
    }
    
    // Handle remaining bytes
    for (size_t i = simd_size; i < total_size; i++) {
        ptr[i] = 0;
    }
    #elif defined(__ARM_NEON)
    char* ptr = (char*)block;
    size_t simd_size = total_size & ~15;
    uint8x16_t zero = vdupq_n_u8(0);
    
    for (size_t i = 0; i < simd_size; i += 16) {
        vst1q_u8((uint8_t*)(ptr + i), zero);
    }
    
    // Handle remaining bytes
    for (size_t i = simd_size; i < total_size; i++) {
        ptr[i] = 0;
    }
    #else
    memset(block, 0, total_size);
    #endif
    
    block->size = sc->block_size;
    block->magic = BLOCK_MAGIC;
    block->is_free = false;
    block->size_class = size_class;
    block->data = (void*)((char*)block + sizeof(Block));
    block->data = (void*)(((uintptr_t)block->data + QG_POOL_ALIGNMENT - 1) & ~(QG_POOL_ALIGNMENT - 1));
    
    // Prefetch next likely allocation
    __builtin_prefetch((char*)block + total_size, 1, 3);
    
    sc->num_blocks++;
    sc->misses++;
    
    pthread_mutex_unlock(&sc->mutex);
    return block->data;
}

// Allocate large block
static void* allocate_large_block(MemoryPool* pool, size_t size) {
    // Ensure size is properly aligned
    size = (size + QG_POOL_ALIGNMENT - 1) & ~(QG_POOL_ALIGNMENT - 1);
    size_t total_size = sizeof(Block) + size;
    total_size = (total_size + QG_POOL_ALIGNMENT - 1) & ~(QG_POOL_ALIGNMENT - 1);
    
    void* block_ptr = NULL;
    if (posix_memalign(&block_ptr, QG_POOL_ALIGNMENT, total_size) != 0) {
        return NULL;
    }
    Block* block = block_ptr;
    
    block->size = size;
    block->magic = BLOCK_MAGIC;
    block->is_free = false;
    block->size_class = QG_NUM_SIZE_CLASSES;
    block->data = (void*)((char*)block + sizeof(Block));
    
    // Ensure data pointer is properly aligned
    block->data = (void*)(((uintptr_t)block->data + QG_POOL_ALIGNMENT - 1) & ~(QG_POOL_ALIGNMENT - 1));
    
    pthread_mutex_lock(&pool->mutex);
    block->next = pool->large_blocks;
    if (pool->large_blocks) {
        pool->large_blocks->prev = block;
    }
    pool->large_blocks = block;
    pthread_mutex_unlock(&pool->mutex);
    
    return block->data;
}

// Update peak memory with proper memory ordering
static void update_peak_memory(MemoryPool* pool, size_t current) {
    size_t peak;
    do {
        peak = atomic_load_explicit(&pool->peak_allocated, memory_order_acquire);
        if (current <= peak) {
            break;
        }
    } while (!atomic_compare_exchange_strong_explicit(
        &pool->peak_allocated,
        &peak,
        current,
        memory_order_release,
        memory_order_relaxed));
}

// Allocate memory
void* pool_malloc(MemoryPool* pool, size_t size) {
    if (!pool || !size) {
        return NULL;
    }
    
    // Try thread cache first
    void* ptr = allocate_from_cache(pool, size);
    if (ptr) {
        return ptr;
    }
    
    // Check if size fits in size classes
    if (size <= atomic_load_explicit(&size_class_table[QG_NUM_SIZE_CLASSES-1], memory_order_acquire)) {
        uint16_t size_class = get_size_class(size);
        ptr = allocate_from_class(pool, size_class);
    } else {
        ptr = allocate_large_block(pool, size);
    }
    
    if (ptr && pool->config.enable_stats) {
        size_t current = atomic_fetch_add_explicit(&pool->total_allocated, size, memory_order_relaxed) + size;
        atomic_fetch_add_explicit(&pool->num_allocations, 1, memory_order_relaxed);
        update_peak_memory(pool, current);
    }
    
    return ptr;
}

// Get block from pointer with enhanced validation
static Block* get_block(void* ptr) {
    if (!ptr) {
        geometric_log_error("Null pointer passed to get_block");
        return NULL;
    }
    
    // Check basic pointer validity
    if ((uintptr_t)ptr < QG_POOL_ALIGNMENT) {
        geometric_log_error("Invalid pointer value");
        return NULL;
    }
    
    // Check alignment
    if ((uintptr_t)ptr % QG_POOL_ALIGNMENT != 0) {
        geometric_log_error("Invalid pointer alignment");
        return NULL;
    }
    
    // Get block pointer with alignment check
    Block* block = (Block*)((uintptr_t)((char*)ptr - sizeof(Block)) & ~(QG_POOL_ALIGNMENT - 1));
    
    // Validate block pointer
    if ((uintptr_t)block % QG_POOL_ALIGNMENT != 0) {
        geometric_log_error("Invalid block alignment");
        return NULL;
    }
    
    // Validate magic number
    if (block->magic != BLOCK_MAGIC) {
        geometric_log_error("Invalid block magic number");
        return NULL;
    }
    
    // Validate size class
    if (block->size_class > QG_NUM_SIZE_CLASSES) {
        geometric_log_error("Invalid size class");
        return NULL;
    }
    
    // Validate data pointer alignment
    if ((uintptr_t)block->data % QG_POOL_ALIGNMENT != 0) {
        geometric_log_error("Invalid data pointer alignment");
        return NULL;
    }
    
    // Validate data pointer
    if (block->data != ptr) {
        geometric_log_error("Invalid block data pointer");
        return NULL;
    }
    
    // Validate block size
    if (block->size_class < QG_NUM_SIZE_CLASSES) {
        size_t expected_size = atomic_load_explicit(&size_class_table[block->size_class], memory_order_acquire);
        if (block->size != expected_size) {
            geometric_log_error("Invalid block size");
            return NULL;
        }
    }
    
    return block;
}

// Add to thread cache with error handling
static bool add_to_cache(uint16_t size_class, void* ptr) {
    init_thread_cache();
    
    if (thread_cache.count[size_class] >= QG_MAX_THREAD_CACHE) {
        return false;
    }
    
    void* entry_ptr = NULL;
    if (posix_memalign(&entry_ptr, QG_POOL_ALIGNMENT, sizeof(ThreadCacheEntry)) != 0) {
        geometric_log_error("Failed to allocate thread cache entry");
        return false;
    }
    ThreadCacheEntry* entry = entry_ptr;
    
    entry->ptr = ptr;
    entry->size_class = size_class;
    entry->next = thread_cache.entries[size_class];
    thread_cache.entries[size_class] = entry;
    thread_cache.count[size_class]++;
    return true;
}

// Free memory
void pool_free(MemoryPool* pool, void* ptr) {
    if (!pool || !ptr) {
        return;
    }
    
    Block* block = get_block(ptr);
    if (!block) {
        geometric_log_error("Invalid pointer passed to pool_free");
        return;
    }
    
    if (block->is_free) {
        geometric_log_error("Double free detected");
        return;
    }
    
    // Update stats
    if (pool->config.enable_stats) {
        atomic_fetch_sub_explicit(&pool->total_allocated, block->size, memory_order_relaxed);
    }
    
    // Handle large blocks
    if (block->size_class == QG_NUM_SIZE_CLASSES) {
        pthread_mutex_lock(&pool->mutex);
        if (block->prev) {
            block->prev->next = block->next;
        } else {
            pool->large_blocks = block->next;
        }
        if (block->next) {
            block->next->prev = block->prev;
        }
        pthread_mutex_unlock(&pool->mutex);
        free(block);
        return;
    }
    
    // Try adding to thread cache if enabled
    if (pool->config.thread_cache_size > 0 && add_to_cache(block->size_class, ptr)) {
        return;
    }
    
    // Return to size class if thread cache is full or disabled
    SizeClass* sc = &pool->size_classes[block->size_class];
    
    pthread_mutex_lock(&sc->mutex);
    block->is_free = true;
    block->next = sc->free_list;
    sc->free_list = block;
    sc->num_blocks++;
    pthread_mutex_unlock(&sc->mutex);
}

// Cleanup memory pool
void cleanup_memory_pool(MemoryPool* pool) {
    if (!pool) return;
    
    // Free all blocks in size classes
    for (int i = 0; i < QG_NUM_SIZE_CLASSES; i++) {
        SizeClass* sc = &pool->size_classes[i];
        pthread_mutex_lock(&sc->mutex);
        Block* block = sc->free_list;
        while (block) {
            Block* next = block->next;
            free(block);
            block = next;
        }
        pthread_mutex_unlock(&sc->mutex);
        pthread_mutex_destroy(&sc->mutex);
    }
    
    // Free large blocks
    pthread_mutex_lock(&pool->mutex);
    Block* block = pool->large_blocks;
    while (block) {
        Block* next = block->next;
        free(block);
        block = next;
    }
    pthread_mutex_unlock(&pool->mutex);
    
    // Cleanup thread caches
    if (thread_cache_initialized) {
        for (int i = 0; i < QG_NUM_SIZE_CLASSES; i++) {
            ThreadCacheEntry* entry = thread_cache.entries[i];
            while (entry) {
                ThreadCacheEntry* next = entry->next;
                free(entry);
                entry = next;
            }
            thread_cache.entries[i] = NULL;
            thread_cache.count[i] = 0;
        }
        thread_cache_initialized = false;
    }
    
    pthread_mutex_destroy(&pool->mutex);
    free(pool->size_classes);
    free(pool);
}

// Get statistics with proper memory ordering
size_t get_total_allocated(const MemoryPool* pool) {
    return pool ? atomic_load_explicit(&pool->total_allocated, memory_order_acquire) : 0;
}

size_t get_peak_allocated(const MemoryPool* pool) {
    return pool ? atomic_load_explicit(&pool->peak_allocated, memory_order_acquire) : 0;
}

size_t get_num_allocations(const MemoryPool* pool) {
    return pool ? atomic_load_explicit(&pool->num_allocations, memory_order_acquire) : 0;
}
