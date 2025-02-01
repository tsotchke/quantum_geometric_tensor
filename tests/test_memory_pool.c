#include "quantum_geometric/core/memory_pool.h"
#include "quantum_geometric/core/quantum_geometric_logging.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>

// Test configuration
static PoolConfig test_config = {
    .max_blocks_per_class = 1024,
    .thread_cache_size = 256,
    .enable_stats = true,
    .prefetch_distance = 1
};

// Initialize logging with error handling
static void init_test_logging(void) {
    if (geometric_init_logging(NULL) != 0) {
        fprintf(stderr, "Failed to initialize logging\n");
        exit(1);
    }
    geometric_set_log_level(LOG_LEVEL_DEBUG);
    geometric_set_log_flags(LOG_FLAG_LEVEL);
}

// Test basic allocation and deallocation
static void test_basic_allocation(void) {
    printf("Running basic allocation test...\n");
    
    MemoryPool* pool = init_memory_pool(&test_config);
    assert(pool != NULL);
    
    // Allocate and verify
    void* ptr = pool_malloc(pool, 64);
    assert(ptr != NULL);
    assert(get_total_allocated(pool) == 64);
    assert(get_num_allocations(pool) == 1);
    
    // Write pattern and verify
    memset(ptr, 0xAA, 64);
    
    // Free and verify stats
    pool_free(pool, ptr);
    assert(get_total_allocated(pool) == 0);
    
    cleanup_memory_pool(pool);
    printf("Basic allocation test passed\n");
}

// Test multiple size classes
static void test_size_classes(void) {
    printf("Running size classes test...\n");
    
    MemoryPool* pool = init_memory_pool(&test_config);
    assert(pool != NULL);
    
    // Test different sizes
    void* ptrs[5];
    size_t sizes[] = {32, 64, 128, 256, 512};
    size_t total = 0;
    
    for (int i = 0; i < 5; i++) {
        ptrs[i] = pool_malloc(pool, sizes[i]);
        assert(ptrs[i] != NULL);
        total += sizes[i];
        assert(get_total_allocated(pool) == total);
        memset(ptrs[i], 0xBB, sizes[i]);
    }
    
    // Free in reverse order
    for (int i = 4; i >= 0; i--) {
        pool_free(pool, ptrs[i]);
        total -= sizes[i];
        assert(get_total_allocated(pool) == total);
    }
    
    cleanup_memory_pool(pool);
    printf("Size classes test passed\n");
}

// Test thread cache
static void test_thread_cache(void) {
    printf("Running thread cache test...\n");
    
    MemoryPool* pool = init_memory_pool(&test_config);
    assert(pool != NULL);
    
    // Allocate and free repeatedly to exercise thread cache
    void* ptrs[100];
    for (int i = 0; i < 100; i++) {
        ptrs[i] = pool_malloc(pool, 64);
        assert(ptrs[i] != NULL);
        memset(ptrs[i], 0xCC, 64);
    }
    
    // Free all allocations
    for (int i = 0; i < 100; i++) {
        pool_free(pool, ptrs[i]);
    }
    
    // Verify stats
    assert(get_total_allocated(pool) == 0);
    
    cleanup_memory_pool(pool);
    printf("Thread cache test passed\n");
}

// Test large allocations
static void test_large_allocations(void) {
    printf("Running large allocations test...\n");
    
    MemoryPool* pool = init_memory_pool(&test_config);
    assert(pool != NULL);
    
    // Test allocation larger than max size class
    size_t large_size = 5 * 1024 * 1024; // 5MB
    void* ptr = pool_malloc(pool, large_size);
    assert(ptr != NULL);
    assert(get_total_allocated(pool) == large_size);
    
    memset(ptr, 0xDD, large_size);
    pool_free(pool, ptr);
    
    assert(get_total_allocated(pool) == 0);
    cleanup_memory_pool(pool);
    printf("Large allocations test passed\n");
}

// Test error handling
static void test_error_handling(void) {
    printf("Running error handling test...\n");
    
    MemoryPool* pool = init_memory_pool(&test_config);
    assert(pool != NULL);
    
    // Test null pointer free
    pool_free(pool, NULL);
    
    // Test zero size allocation
    void* ptr = pool_malloc(pool, 0);
    assert(ptr == NULL);
    
    // Test invalid pointer free
    char invalid[64];
    pool_free(pool, invalid);
    
    cleanup_memory_pool(pool);
    printf("Error handling test passed\n");
}

// Test peak memory tracking
static void test_peak_memory(void) {
    printf("Running peak memory test...\n");
    
    MemoryPool* pool = init_memory_pool(&test_config);
    assert(pool != NULL);
    
    void* ptrs[3];
    size_t sizes[] = {1024, 2048, 4096};
    size_t total = 0;
    
    // Allocate incrementally
    for (int i = 0; i < 3; i++) {
        ptrs[i] = pool_malloc(pool, sizes[i]);
        assert(ptrs[i] != NULL);
        total += sizes[i];
        assert(get_peak_allocated(pool) == total);
    }
    
    // Free in order
    for (int i = 0; i < 3; i++) {
        pool_free(pool, ptrs[i]);
    }
    
    // Peak should remain at highest point
    assert(get_peak_allocated(pool) == total);
    assert(get_total_allocated(pool) == 0);
    
    cleanup_memory_pool(pool);
    printf("Peak memory test passed\n");
}

// Test concurrent allocations
static void* concurrent_allocation_thread(void* arg) {
    MemoryPool* pool = (MemoryPool*)arg;
    void* ptrs[100];
    
    // Perform allocations
    for (int i = 0; i < 100; i++) {
        ptrs[i] = pool_malloc(pool, 128);
        assert(ptrs[i] != NULL);
        memset(ptrs[i], 0xEE, 128);
    }
    
    // Free allocations
    for (int i = 0; i < 100; i++) {
        pool_free(pool, ptrs[i]);
    }
    
    return NULL;
}

static void test_concurrent_allocations(void) {
    printf("Running concurrent allocations test...\n");
    
    MemoryPool* pool = init_memory_pool(&test_config);
    assert(pool != NULL);
    
    pthread_t threads[4];
    
    // Create threads
    for (int i = 0; i < 4; i++) {
        int ret = pthread_create(&threads[i], NULL, concurrent_allocation_thread, pool);
        assert(ret == 0);
    }
    
    // Wait for threads
    for (int i = 0; i < 4; i++) {
        pthread_join(threads[i], NULL);
    }
    
    assert(get_total_allocated(pool) == 0);
    cleanup_memory_pool(pool);
    printf("Concurrent allocations test passed\n");
}

int main() {
    init_test_logging();
    printf("Running memory pool tests...\n");
    
    test_basic_allocation();
    test_size_classes();
    test_thread_cache();
    test_large_allocations();
    test_error_handling();
    test_peak_memory();
    test_concurrent_allocations();
    
    printf("All memory pool tests passed!\n");
    return 0;
}
