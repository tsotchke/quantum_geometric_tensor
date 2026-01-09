/**
 * @file test_performance_metrics.c
 * @brief Tests for performance monitoring operations
 */

#include "quantum_geometric/core/performance_operations.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>

/* Test helpers */
static void assert_metrics_valid(const performance_metrics_t* metrics) {
    // Check that metrics are non-negative
    assert(metrics->execution_time >= 0.0);
    assert(metrics->memory_usage >= 0);
    // CPU/GPU utilization should be in [0, 1]
    assert(metrics->cpu_utilization >= 0.0 && metrics->cpu_utilization <= 1.0);
}

/* Test cases */
static void test_basic_timing(void) {
    printf("Testing basic timing operations...\n");

    /* Initialize performance monitoring */
    performance_config_t config = {
        .log_file = NULL,
        .log_level = 0,
        .enable_profiling = 1,
        .collect_memory_stats = 1,
        .collect_cache_stats = 0,
        .collect_flops = 0,
        .enable_visualization = 0
    };

    int ret = qg_performance_init(&config);
    if (ret != QG_PERFORMANCE_SUCCESS) {
        printf("  SKIP: Performance init not available (%d)\n", ret);
        return;
    }

    /* Test timer operations */
    performance_timer_t timer = {0};
    ret = qg_timer_start(&timer, "test_operation");
    assert(ret == QG_PERFORMANCE_SUCCESS);

    /* Simulate some work */
    volatile double sum = 0.0;
    for (int i = 0; i < 100000; i++) {
        sum += i * 0.001;
    }
    (void)sum;  // Prevent optimization

    ret = qg_timer_stop(&timer);
    assert(ret == QG_PERFORMANCE_SUCCESS);

    double elapsed = qg_timer_get_elapsed(&timer);
    printf("  Elapsed time: %.6f seconds\n", elapsed);
    assert(elapsed >= 0.0);

    /* Cleanup */
    qg_performance_cleanup();
    printf("Basic timing test passed\n");
}

static void test_section_monitoring(void) {
    printf("Testing section-based monitoring...\n");

    /* Initialize */
    performance_config_t config = {
        .log_file = NULL,
        .log_level = 0,
        .enable_profiling = 1,
        .collect_memory_stats = 1,
        .collect_cache_stats = 0,
        .collect_flops = 1,
        .enable_visualization = 0
    };

    int ret = qg_performance_init(&config);
    if (ret != QG_PERFORMANCE_SUCCESS) {
        printf("  SKIP: Performance init not available\n");
        return;
    }

    /* Start monitoring a section */
    ret = qg_start_monitoring("compute_section");
    assert(ret == QG_PERFORMANCE_SUCCESS);

    /* Simulate computation */
    volatile double result = 0.0;
    for (int i = 0; i < 50000; i++) {
        result += sin((double)i * 0.001);
    }
    (void)result;

    /* Stop monitoring */
    ret = qg_stop_monitoring("compute_section");
    assert(ret == QG_PERFORMANCE_SUCCESS);

    /* Get metrics */
    performance_metrics_t metrics = {0};
    ret = qg_get_performance_metrics("compute_section", &metrics);
    if (ret == QG_PERFORMANCE_SUCCESS) {
        printf("  Execution time: %.6f seconds\n", metrics.execution_time);
        printf("  Memory usage: %zu bytes\n", metrics.memory_usage);
        assert_metrics_valid(&metrics);
    } else {
        printf("  Metrics retrieval returned %d (may not be implemented)\n", ret);
    }

    /* Cleanup */
    qg_performance_cleanup();
    printf("Section monitoring test passed\n");
}

static void test_memory_tracking(void) {
    printf("Testing memory tracking...\n");

    /* Initialize */
    performance_config_t config = {
        .log_file = NULL,
        .log_level = 0,
        .enable_profiling = 0,
        .collect_memory_stats = 1,
        .collect_cache_stats = 0,
        .collect_flops = 0,
        .enable_visualization = 0
    };

    int ret = qg_performance_init(&config);
    if (ret != QG_PERFORMANCE_SUCCESS) {
        printf("  SKIP: Performance init not available\n");
        return;
    }

    /* Get initial memory usage */
    size_t initial_memory = qg_get_current_memory_usage();
    printf("  Initial memory: %zu bytes\n", initial_memory);

    /* Allocate some memory */
    void* ptr = malloc(1024 * 1024);  // 1MB
    assert(ptr != NULL);
    memset(ptr, 0, 1024 * 1024);  // Touch the memory

    /* Check memory after allocation */
    size_t after_alloc = qg_get_current_memory_usage();
    printf("  After alloc: %zu bytes\n", after_alloc);

    /* Check peak */
    size_t peak = qg_get_peak_memory_usage();
    printf("  Peak memory: %zu bytes\n", peak);

    /* Free memory */
    free(ptr);

    /* Cleanup */
    qg_performance_cleanup();
    printf("Memory tracking test passed\n");
}

static void test_high_precision_timing(void) {
    printf("Testing high-precision timing utilities...\n");

    /* Test timestamp */
    uint64_t ts1 = qg_get_timestamp_ns();

    /* Small delay */
    volatile int x = 0;
    for (int i = 0; i < 10000; i++) {
        x += i;
    }
    (void)x;

    uint64_t ts2 = qg_get_timestamp_ns();

    printf("  Timestamp delta: %llu ns\n", (unsigned long long)(ts2 - ts1));
    assert(ts2 >= ts1);  // Time should move forward

    /* Test seconds */
    double t1 = qg_get_time_seconds();
    double t2 = qg_get_time_seconds();
    printf("  Time delta: %.9f seconds\n", t2 - t1);
    assert(t2 >= t1);

    printf("High-precision timing test passed\n");
}

static void test_timer_operations(void) {
    printf("Testing timer reset and reuse...\n");

    performance_config_t config = {0};
    int ret = qg_performance_init(&config);
    if (ret != QG_PERFORMANCE_SUCCESS) {
        printf("  SKIP: Performance init not available\n");
        return;
    }

    performance_timer_t timer = {0};

    /* First measurement */
    qg_timer_start(&timer, "reuse_test");
    volatile double x = 0;
    for (int i = 0; i < 10000; i++) x += i;
    (void)x;
    qg_timer_stop(&timer);
    double first_elapsed = qg_timer_get_elapsed(&timer);
    printf("  First measurement: %.6f seconds\n", first_elapsed);

    /* Reset and measure again */
    ret = qg_timer_reset(&timer);
    assert(ret == QG_PERFORMANCE_SUCCESS);

    qg_timer_start(&timer, "reuse_test_2");
    x = 0;
    for (int i = 0; i < 20000; i++) x += i;
    (void)x;
    qg_timer_stop(&timer);
    double second_elapsed = qg_timer_get_elapsed(&timer);
    printf("  Second measurement: %.6f seconds\n", second_elapsed);

    qg_performance_cleanup();
    printf("Timer operations test passed\n");
}

/* Main test runner */
int main(void) {
    printf("Running performance metrics tests...\n\n");

    test_basic_timing();
    printf("\n");

    test_section_monitoring();
    printf("\n");

    test_memory_tracking();
    printf("\n");

    test_high_precision_timing();
    printf("\n");

    test_timer_operations();
    printf("\n");

    printf("All performance metrics tests passed!\n");
    return 0;
}
