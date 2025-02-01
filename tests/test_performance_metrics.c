#include "core/performance_operations.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

/* Test helpers */
static void assert_performance_stats(
    const performance_stats_t* stats,
    size_t expected_total,
    size_t expected_peak,
    size_t expected_current,
    size_t expected_count
) {
    assert(stats->total_allocated == expected_total);
    assert(stats->peak_allocated == expected_peak);
    assert(stats->current_allocated == expected_current);
    assert(stats->allocation_count == expected_count);
}

/* Test cases */
static void test_basic_monitoring(void) {
    printf("Testing basic performance monitoring...\n");
    
    /* Start monitoring */
    qgt_error_t err = start_performance_monitoring();
    assert(err == QGT_SUCCESS);
    
    /* Simulate some allocations */
    update_performance_metrics(1000, 100);
    update_performance_metrics(2000, 200);
    update_performance_metrics(-1000, 0);
    
    /* Check stats */
    performance_stats_t stats;
    err = get_performance_stats(&stats);
    assert(err == QGT_SUCCESS);
    
    assert_performance_stats(&stats, 3000, 3000, 2000, 2);
    
    /* Stop monitoring */
    stop_performance_monitoring();
    printf("Basic monitoring test passed\n");
}

static void test_gpu_timing(void) {
    printf("Testing GPU timing...\n");
    
    /* Start monitoring */
    qgt_error_t err = start_performance_monitoring();
    assert(err == QGT_SUCCESS);
    
    /* Record some GPU operations */
    record_gpu_time(10.5);
    record_gpu_time(20.3);
    
    /* Check stats */
    performance_stats_t stats;
    err = get_performance_stats(&stats);
    assert(err == QGT_SUCCESS);
    
    /* Compare with small tolerance for floating point */
    assert(stats.gpu_time > 0.030 && stats.gpu_time < 0.031);
    
    /* Stop monitoring */
    stop_performance_monitoring();
    printf("GPU timing test passed\n");
}

static void test_operation_counting(void) {
    printf("Testing operation counting...\n");
    
    /* Start monitoring */
    qgt_error_t err = start_performance_monitoring();
    assert(err == QGT_SUCCESS);
    
    /* Record some operations */
    update_performance_metrics(0, 100);
    update_performance_metrics(0, 250);
    
    /* Check stats */
    performance_stats_t stats;
    err = get_performance_stats(&stats);
    assert(err == QGT_SUCCESS);
    
    assert(stats.total_operations == 350);
    
    /* Stop monitoring */
    stop_performance_monitoring();
    printf("Operation counting test passed\n");
}

static void test_error_handling(void) {
    printf("Testing error handling...\n");
    
    /* Try to get stats before starting */
    performance_stats_t stats;
    qgt_error_t err = get_performance_stats(&stats);
    assert(err == QGT_ERROR_NOT_INITIALIZED);
    
    /* Try to start twice */
    err = start_performance_monitoring();
    assert(err == QGT_SUCCESS);
    
    err = start_performance_monitoring();
    assert(err == QGT_ERROR_ALREADY_INITIALIZED);
    
    /* Try to get stats with NULL pointer */
    err = get_performance_stats(NULL);
    assert(err == QGT_ERROR_INVALID_ARGUMENT);
    
    /* Stop monitoring */
    stop_performance_monitoring();
    printf("Error handling test passed\n");
}

static void test_concurrent_updates(void) {
    printf("Testing concurrent updates...\n");
    
    /* Start monitoring */
    qgt_error_t err = start_performance_monitoring();
    assert(err == QGT_SUCCESS);
    
    /* Simulate concurrent allocations */
    update_performance_metrics(1000, 100);
    update_performance_metrics(2000, 200);
    update_performance_metrics(1500, 150);
    update_performance_metrics(-500, 0);
    
    /* Check stats */
    performance_stats_t stats;
    err = get_performance_stats(&stats);
    assert(err == QGT_SUCCESS);
    
    assert_performance_stats(&stats, 4500, 4500, 4000, 3);
    
    /* Stop monitoring */
    stop_performance_monitoring();
    printf("Concurrent updates test passed\n");
}

/* Main test runner */
int main(void) {
    printf("Running performance metrics tests...\n\n");
    
    test_basic_monitoring();
    test_gpu_timing();
    test_operation_counting();
    test_error_handling();
    test_concurrent_updates();
    
    printf("\nAll performance metrics tests passed!\n");
    return 0;
}
