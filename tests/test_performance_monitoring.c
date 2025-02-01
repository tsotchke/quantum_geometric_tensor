/**
 * @file test_performance_monitoring.c
 * @brief Tests for performance monitoring system
 */

#include "quantum_geometric/core/performance_monitor.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <unistd.h>

// Test helper functions
static void test_initialization(void);
static void test_operation_timing(void);
static void test_resource_monitoring(void);
static void test_success_metrics(void);
static void test_recovery_metrics(void);
static void test_performance_thresholds(void);
static void test_error_cases(void);

// Mock operations for testing
static void perform_mock_operation(const char* name, useconds_t duration);
static void allocate_mock_memory(size_t size_mb);
static void simulate_cpu_load(void);

int main(void) {
    printf("Running performance monitoring tests...\n");

    // Run all tests
    test_initialization();
    test_operation_timing();
    test_resource_monitoring();
    test_success_metrics();
    test_recovery_metrics();
    test_performance_thresholds();
    test_error_cases();

    printf("All performance monitoring tests passed!\n");
    return 0;
}

static void test_initialization(void) {
    printf("Testing initialization...\n");

    // Test initialization
    bool success = init_performance_monitoring();
    assert(success);

    // Test double initialization
    success = init_performance_monitoring();
    assert(success);  // Should be idempotent

    // Test cleanup
    cleanup_performance_monitoring();

    // Test re-initialization after cleanup
    success = init_performance_monitoring();
    assert(success);

    cleanup_performance_monitoring();
    printf("Initialization tests passed\n");
}

static void test_operation_timing(void) {
    printf("Testing operation timing...\n");

    bool success = init_performance_monitoring();
    assert(success);

    // Test error detection timing
    perform_mock_operation("error_detection", 5000);  // 5ms
    
    // Test correction cycle timing
    perform_mock_operation("correction_cycle", 25000);  // 25ms
    
    // Test state verification timing
    perform_mock_operation("state_verification", 50000);  // 50ms

    // Get metrics and verify
    PerformanceMetrics metrics = get_performance_metrics();
    assert(metrics.avg_latency > 0);

    cleanup_performance_monitoring();
    printf("Operation timing tests passed\n");
}

static void test_resource_monitoring(void) {
    printf("Testing resource monitoring...\n");

    bool success = init_performance_monitoring();
    assert(success);

    // Test memory monitoring
    allocate_mock_memory(100);  // 100MB
    update_resource_usage();
    
    // Test CPU monitoring
    simulate_cpu_load();
    update_resource_usage();

    // Get metrics and verify
    PerformanceMetrics metrics = get_performance_metrics();
    assert(metrics.peak_memory_usage > 0);
    assert(metrics.avg_cpu_utilization > 0);

    cleanup_performance_monitoring();
    printf("Resource monitoring tests passed\n");
}

static void test_success_metrics(void) {
    printf("Testing success metrics...\n");

    bool success = init_performance_monitoring();
    assert(success);

    // Record successful operations
    for (int i = 0; i < 90; i++) {
        record_operation_result(true, false);
    }

    // Record failed operations
    for (int i = 0; i < 10; i++) {
        record_operation_result(false, false);
    }

    // Record false positives
    for (int i = 0; i < 5; i++) {
        record_operation_result(true, true);
    }

    // Get metrics and verify
    PerformanceMetrics metrics = get_performance_metrics();
    assert(metrics.success_rate == 90.0);
    assert(metrics.false_positive_rate == 5.0);

    cleanup_performance_monitoring();
    printf("Success metrics tests passed\n");
}

static void test_recovery_metrics(void) {
    printf("Testing recovery metrics...\n");

    bool success = init_performance_monitoring();
    assert(success);

    // Record successful recoveries
    for (int i = 0; i < 95; i++) {
        record_recovery_result(true);
    }

    // Record failed recoveries
    for (int i = 0; i < 5; i++) {
        record_recovery_result(false);
    }

    // Get metrics and verify
    PerformanceMetrics metrics = get_performance_metrics();
    assert(metrics.recovery_success_rate == 95.0);

    cleanup_performance_monitoring();
    printf("Recovery metrics tests passed\n");
}

static void test_performance_thresholds(void) {
    printf("Testing performance thresholds...\n");

    bool success = init_performance_monitoring();
    assert(success);

    // Test latency threshold
    perform_mock_operation("error_detection", 15000);  // 15ms > 10ms threshold
    
    // Test memory threshold
    allocate_mock_memory(500);  // 500MB to exceed threshold
    update_resource_usage();
    
    // Test CPU threshold
    simulate_cpu_load();  // Should exceed 80% threshold
    update_resource_usage();

    // Test success rate threshold
    for (int i = 0; i < 98; i++) {
        record_operation_result(false, false);  // Below 99% threshold
    }

    // Test recovery rate threshold
    for (int i = 0; i < 98; i++) {
        record_recovery_result(false);  // Below 99% threshold
    }

    cleanup_performance_monitoring();
    printf("Performance threshold tests passed\n");
}

static void test_error_cases(void) {
    printf("Testing error cases...\n");

    // Test operations before initialization
    start_operation_timing("test");
    end_operation_timing("test");
    update_resource_usage();
    record_operation_result(true, false);
    record_recovery_result(true);

    PerformanceMetrics metrics = get_performance_metrics();
    assert(metrics.avg_latency == 0);
    assert(metrics.peak_memory_usage == 0);
    assert(metrics.success_rate == 0);
    assert(metrics.recovery_success_rate == 0);

    // Test cleanup without initialization
    cleanup_performance_monitoring();

    printf("Error case tests passed\n");
}

// Mock implementations

static void perform_mock_operation(const char* name, useconds_t duration) {
    start_operation_timing(name);
    usleep(duration);
    end_operation_timing(name);
}

static void allocate_mock_memory(size_t size_mb) {
    void* ptr = malloc(size_mb * 1024 * 1024);
    assert(ptr != NULL);
    // Touch pages to ensure allocation
    memset(ptr, 0, size_mb * 1024 * 1024);
    // Don't free to test peak memory usage
}

static void simulate_cpu_load(void) {
    // Perform CPU-intensive operation
    for (int i = 0; i < 1000000; i++) {
        double x = rand() / (double)RAND_MAX;
        x = sqrt(x);  // Force FPU usage
    }
}
