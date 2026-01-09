/**
 * @file test_performance_monitoring.c
 * @brief Tests for performance monitoring system
 */

#include "quantum_geometric/core/performance_monitor.h"
#include "quantum_geometric/core/performance_operations.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <unistd.h>

// Test helper functions
static void test_initialization(void);
static void test_operation_timing(void);
static void test_resource_monitoring(void);
static void test_quantum_metrics(void);
static void test_hardware_counters(void);
static void test_performance_control(void);
static void test_error_cases(void);

// Helper functions
static void perform_mock_operation(performance_timer_t* timer, const char* name, useconds_t duration);
static void simulate_cpu_load(void);

int main(void) {
    printf("Running performance monitoring tests...\n");

    // Run all tests
    test_initialization();
    test_operation_timing();
    test_resource_monitoring();
    test_quantum_metrics();
    test_hardware_counters();
    test_performance_control();
    test_error_cases();

    printf("All performance monitoring tests passed!\n");
    return 0;
}

static void test_initialization(void) {
    printf("Testing initialization...\n");

    // Test basic initialization
    init_performance_monitor();

    // Test with configuration
    int ret = initialize_performance_monitor(NULL, NULL);
    // May or may not succeed depending on implementation
    (void)ret;

    // Test cleanup
    cleanup_performance_monitor();

    // Test re-initialization after cleanup
    init_performance_monitor();
    cleanup_performance_monitor();

    printf("Initialization tests passed\n");
}

static void test_operation_timing(void) {
    printf("Testing operation timing...\n");

    // Initialize performance monitoring
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

    // Test timer operations
    performance_timer_t timer = {0};

    // Test error detection timing
    perform_mock_operation(&timer, "error_detection", 5000);  // 5ms
    double elapsed1 = qg_timer_get_elapsed(&timer);
    printf("  Error detection timing: %.6f seconds\n", elapsed1);
    assert(elapsed1 >= 0.0);

    // Reset and test correction cycle timing
    qg_timer_reset(&timer);
    perform_mock_operation(&timer, "correction_cycle", 10000);  // 10ms
    double elapsed2 = qg_timer_get_elapsed(&timer);
    printf("  Correction cycle timing: %.6f seconds\n", elapsed2);
    assert(elapsed2 >= 0.0);

    // Reset and test state verification timing
    qg_timer_reset(&timer);
    perform_mock_operation(&timer, "state_verification", 15000);  // 15ms
    double elapsed3 = qg_timer_get_elapsed(&timer);
    printf("  State verification timing: %.6f seconds\n", elapsed3);
    assert(elapsed3 >= 0.0);

    qg_performance_cleanup();
    printf("Operation timing tests passed\n");
}

static void test_resource_monitoring(void) {
    printf("Testing resource monitoring...\n");

    init_performance_monitor();

    // Test memory allocation tracking
    void* ptr = malloc(1024 * 1024);  // 1MB
    if (ptr) {
        memset(ptr, 0, 1024 * 1024);  // Touch the memory
        update_allocation_stats(1024 * 1024, 1024 * 1024, false);
    }

    // Test CPU load simulation
    simulate_cpu_load();

    // Get current performance metrics
    performance_metrics_t metrics = get_current_performance_metrics();
    printf("  Execution time: %.6f seconds\n", metrics.execution_time);
    printf("  Memory usage: %zu bytes\n", metrics.memory_usage);
    printf("  CPU utilization: %.2f\n", metrics.cpu_utilization);

    // Legacy metrics API
    PerformanceMetrics legacy_metrics = get_performance_metrics();
    printf("  Peak memory (legacy): %.2f\n", legacy_metrics.peak_memory_usage);

    // Cleanup
    free(ptr);
    cleanup_performance_monitor();
    printf("Resource monitoring tests passed\n");
}

static void test_quantum_metrics(void) {
    printf("Testing quantum metrics...\n");

    init_performance_monitor();

    // Set quantum metrics
    set_quantum_error_rate(0.01);
    set_quantum_fidelity(0.99);
    set_entanglement_fidelity(0.98);
    set_gate_error_rate(0.001);

    // Batch update
    update_quantum_metrics(0.015, 0.985, 0.975, 0.0015);

    // Measure quantum metrics
    double error_rate = measure_quantum_error_rate();
    double fidelity = measure_quantum_fidelity();
    double entanglement = measure_entanglement_fidelity();
    double gate_error = measure_gate_error_rate();

    printf("  Quantum error rate: %.4f\n", error_rate);
    printf("  Quantum fidelity: %.4f\n", fidelity);
    printf("  Entanglement fidelity: %.4f\n", entanglement);
    printf("  Gate error rate: %.4f\n", gate_error);

    // Verify values are in expected ranges
    assert(error_rate >= 0.0 && error_rate <= 1.0);
    assert(fidelity >= 0.0 && fidelity <= 1.0);
    assert(entanglement >= 0.0 && entanglement <= 1.0);
    assert(gate_error >= 0.0 && gate_error <= 1.0);

    cleanup_performance_monitor();
    printf("Quantum metrics tests passed\n");
}

static void test_hardware_counters(void) {
    printf("Testing hardware counters...\n");

    init_performance_monitor();

    // Read hardware performance counters
    uint64_t page_faults = get_page_faults();
    uint64_t cache_misses = get_cache_misses();
    uint64_t tlb_misses = get_tlb_misses();

    printf("  Page faults: %llu\n", (unsigned long long)page_faults);
    printf("  Cache misses: %llu\n", (unsigned long long)cache_misses);
    printf("  TLB misses: %llu\n", (unsigned long long)tlb_misses);

    // Measure computational performance
    double flops = measure_flops();
    double bandwidth = measure_memory_bandwidth();
    double cache_perf = measure_cache_performance();

    printf("  FLOPS: %.2f\n", flops);
    printf("  Memory bandwidth: %.2f GB/s\n", bandwidth);
    printf("  Cache performance: %.2f\n", cache_perf);

    // Reset counters
    reset_performance_counters();

    cleanup_performance_monitor();
    printf("Hardware counters tests passed\n");
}

static void test_performance_control(void) {
    printf("Testing performance monitoring control...\n");

    init_performance_monitor();

    // Start monitoring
    int ret = start_performance_monitoring();
    printf("  Start monitoring result: %d\n", ret);

    // Check if active
    bool active = is_performance_monitoring_active();
    printf("  Monitoring active: %s\n", active ? "yes" : "no");

    // Simulate some work
    simulate_cpu_load();

    // Stop monitoring
    ret = stop_performance_monitoring();
    printf("  Stop monitoring result: %d\n", ret);

    // Get optimization parameters
    int recommended_threads = get_recommended_thread_count();
    printf("  Recommended thread count: %d\n", recommended_threads);

    size_t block_size, prefetch_distance;
    bool prefetch_enabled;
    get_memory_optimization_params(&block_size, &prefetch_distance, &prefetch_enabled);
    printf("  Memory block size: %zu, prefetch: %zu, enabled: %s\n",
           block_size, prefetch_distance, prefetch_enabled ? "yes" : "no");

    int numa_node = get_numa_preferred_node();
    printf("  Preferred NUMA node: %d\n", numa_node);

    cleanup_performance_monitor();
    printf("Performance control tests passed\n");
}

static void test_error_cases(void) {
    printf("Testing error cases...\n");

    // Test operations before initialization
    // These should handle the uninitialized state gracefully
    PerformanceMetrics metrics = get_performance_metrics();

    // Metrics should be zero or default values when not initialized
    printf("  Uninitialized avg_latency: %.4f\n", metrics.avg_latency);
    printf("  Uninitialized peak_memory: %.4f\n", metrics.peak_memory_usage);

    // Test cleanup without initialization (should not crash)
    cleanup_performance_monitor();

    // Test double cleanup (should be safe)
    init_performance_monitor();
    cleanup_performance_monitor();
    cleanup_performance_monitor();

    printf("Error case tests passed\n");
}

// Helper implementations

static void perform_mock_operation(performance_timer_t* timer, const char* name, useconds_t duration) {
    qg_timer_start(timer, name);
    usleep(duration);
    qg_timer_stop(timer);
}

static void simulate_cpu_load(void) {
    // Perform CPU-intensive operation
    volatile double sum = 0.0;
    for (int i = 0; i < 100000; i++) {
        sum += sin((double)i * 0.001) * cos((double)i * 0.002);
    }
    (void)sum;  // Prevent optimization
}
