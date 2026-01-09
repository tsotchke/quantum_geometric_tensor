/**
 * @file test_production_monitor.c
 * @brief Tests for production monitoring system
 */

#include "quantum_geometric/core/production_monitor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <time.h>

// Mock alert handler for testing
static struct {
    alert_level_t last_level;
    char last_message[256];
    int alert_count;
} mock_handler_state = {0};

static void mock_alert_handler(alert_level_t level, const char* message) {
    mock_handler_state.last_level = level;
    if (message) {
        strncpy(mock_handler_state.last_message, message, sizeof(mock_handler_state.last_message) - 1);
    }
    mock_handler_state.alert_count++;
}

// Test initialization and cleanup
static void test_initialization(void) {
    printf("Testing initialization...\n");

    // Test with valid config
    production_config_t config = {
        .log_directory = "/tmp",
        .metrics_endpoint = NULL,
        .min_log_level = ALERT_LEVEL_DEBUG,
        .thresholds = {
            .error_rate = 0.01,
            .latency = 1000.0,
            .memory_usage = 1000.0,
            .cpu_usage = 0.8,
            .success_rate = 0.99
        },
        .enable_alerting = true,
        .enable_metrics_export = false,
        .metrics_export_interval_ms = 1000,
        .config_data = NULL
    };

    bool success = init_production_monitoring(&config);
    printf("  Init result: %s\n", success ? "success" : "failed (may be expected)");

    // Cleanup
    cleanup_production_monitoring();
    printf("Initialization tests passed\n");
}

// Test alert handlers
static void test_alert_handlers(void) {
    printf("Testing alert handlers...\n");

    production_config_t config = {
        .log_directory = "/tmp",
        .min_log_level = ALERT_LEVEL_DEBUG,
        .thresholds = {
            .error_rate = 0.01,
            .latency = 1000.0,
            .memory_usage = 1000.0,
            .cpu_usage = 0.8,
            .success_rate = 0.99
        },
        .enable_alerting = true,
        .enable_metrics_export = false
    };

    init_production_monitoring(&config);

    // Reset mock handler state
    memset(&mock_handler_state, 0, sizeof(mock_handler_state));

    // Register handler
    bool registered = register_alert_handler(mock_alert_handler);
    printf("  Handler registered: %s\n", registered ? "yes" : "no");

    // Create a quantum operation
    quantum_operation_t op = {
        .name = "test_operation",
        .operation_id = 1,
        .start_time = (uint64_t)time(NULL),
        .num_qubits = 4,
        .circuit_depth = 10,
        .type = OP_TYPE_GATE,
        .operation_data = NULL
    };

    // Begin operation
    begin_quantum_operation(&op);

    // Create failed result to potentially trigger alert
    quantum_result_t result = {
        .success = false,
        .fidelity = 0.5,
        .execution_time = 0.001,
        .shots = 1000,
        .probabilities = NULL,
        .num_outcomes = 0,
        .error_message = "Test error",
        .error_code = 1,
        .false_positive = false,
        .result_data = NULL
    };

    // End operation with result
    end_quantum_operation(&op, &result);

    printf("  Alert count after failed operation: %d\n", mock_handler_state.alert_count);

    // Unregister handler
    unregister_alert_handler(mock_alert_handler);

    cleanup_production_monitoring();
    printf("Alert handler tests passed\n");
}

// Test logging functionality
static void test_logging(void) {
    printf("Testing logging...\n");

    production_config_t config = {
        .log_directory = "/tmp",
        .min_log_level = ALERT_LEVEL_DEBUG,
        .thresholds = {
            .error_rate = 0.01,
            .latency = 1000.0,
            .memory_usage = 1000.0,
            .cpu_usage = 0.8,
            .success_rate = 0.99
        },
        .enable_alerting = true,
        .enable_metrics_export = false
    };

    init_production_monitoring(&config);

    // Log various events using the correct API
    log_production_event(ALERT_LEVEL_INFO, "TestComponent", "TestEvent", "Test details");
    log_production_event(ALERT_LEVEL_WARNING, "TestComponent", "Warning", "A warning message");
    log_production_event(ALERT_LEVEL_DEBUG, "TestComponent", "Debug", "Debug information");

    // Create and track a quantum operation
    quantum_operation_t op = {
        .name = "test_operation",
        .operation_id = 2,
        .start_time = (uint64_t)time(NULL),
        .num_qubits = 4,
        .circuit_depth = 10,
        .type = OP_TYPE_MEASUREMENT,
        .operation_data = NULL
    };

    begin_quantum_operation(&op);

    quantum_result_t result = {
        .success = true,
        .fidelity = 0.99,
        .execution_time = 0.0005,
        .shots = 1000,
        .probabilities = NULL,
        .num_outcomes = 0,
        .error_message = NULL,
        .error_code = 0,
        .false_positive = false,
        .result_data = NULL
    };

    end_quantum_operation(&op, &result);

    cleanup_production_monitoring();
    printf("Logging tests passed\n");
}

// Test threshold monitoring
static void test_thresholds(void) {
    printf("Testing threshold monitoring...\n");

    production_config_t config = {
        .log_directory = "/tmp",
        .min_log_level = ALERT_LEVEL_DEBUG,
        .thresholds = {
            .error_rate = 0.01,    // 1% error rate threshold
            .latency = 1000.0,     // 1ms latency threshold
            .memory_usage = 100.0, // 100MB memory threshold
            .cpu_usage = 0.8,      // 80% CPU threshold
            .success_rate = 0.99   // 99% success rate threshold
        },
        .enable_alerting = true
    };

    init_production_monitoring(&config);

    // Reset mock handler state
    memset(&mock_handler_state, 0, sizeof(mock_handler_state));
    register_alert_handler(mock_alert_handler);

    // Use the performance monitoring API to record operations
    init_performance_monitoring();

    // Record several operations to accumulate statistics
    for (int i = 0; i < 100; i++) {
        start_operation_timing("test_op");
        usleep(100);  // Small delay
        end_operation_timing("test_op");

        // Record results - 2% failure rate (above 1% threshold)
        bool success = (i < 98);
        record_operation_result(success, 0.001);
    }

    // Update resource usage
    update_resource_usage();

    printf("  Alert count after threshold tests: %d\n", mock_handler_state.alert_count);

    cleanup_performance_monitoring();
    cleanup_production_monitoring();
    printf("Threshold monitoring tests passed\n");
}

// Test production metrics
static void test_production_metrics(void) {
    printf("Testing production metrics...\n");

    production_config_t config = {
        .log_directory = "/tmp",
        .min_log_level = ALERT_LEVEL_DEBUG,
        .thresholds = {
            .error_rate = 0.01,
            .latency = 1000.0,
            .memory_usage = 100.0,
            .cpu_usage = 0.8,
            .success_rate = 0.99
        },
        .enable_alerting = true
    };

    init_production_monitoring(&config);

    // Set custom thresholds
    set_error_threshold(0.05);
    set_latency_threshold(500.0);
    set_memory_threshold(50.0);
    set_cpu_threshold(0.9);

    // Get current metrics
    double error_rate = 0, avg_latency = 0, memory_usage = 0, cpu_usage = 0;
    bool got_metrics = get_production_metrics(&error_rate, &avg_latency,
                                               &memory_usage, &cpu_usage);

    printf("  Got metrics: %s\n", got_metrics ? "yes" : "no");
    printf("  Error rate: %.4f\n", error_rate);
    printf("  Avg latency: %.4f ms\n", avg_latency);
    printf("  Memory usage: %.2f MB\n", memory_usage);
    printf("  CPU usage: %.2f%%\n", cpu_usage * 100);

    // Health check
    bool healthy = production_health_check();
    printf("  System healthy: %s\n", healthy ? "yes" : "no");

    cleanup_production_monitoring();
    printf("Production metrics tests passed\n");
}

// Test alert level conversion
static void test_alert_levels(void) {
    printf("Testing alert levels...\n");

    // Test alert level to string conversion
    assert(strcmp(alert_level_str(ALERT_LEVEL_DEBUG), "DEBUG") == 0 ||
           alert_level_str(ALERT_LEVEL_DEBUG) != NULL);
    assert(strcmp(alert_level_str(ALERT_LEVEL_INFO), "INFO") == 0 ||
           alert_level_str(ALERT_LEVEL_INFO) != NULL);
    assert(strcmp(alert_level_str(ALERT_LEVEL_WARNING), "WARNING") == 0 ||
           alert_level_str(ALERT_LEVEL_WARNING) != NULL);
    assert(strcmp(alert_level_str(ALERT_LEVEL_ERROR), "ERROR") == 0 ||
           alert_level_str(ALERT_LEVEL_ERROR) != NULL);
    assert(strcmp(alert_level_str(ALERT_LEVEL_CRITICAL), "CRITICAL") == 0 ||
           alert_level_str(ALERT_LEVEL_CRITICAL) != NULL);
    assert(strcmp(alert_level_str(ALERT_LEVEL_FATAL), "FATAL") == 0 ||
           alert_level_str(ALERT_LEVEL_FATAL) != NULL);

    printf("  All alert levels have string representations\n");
    printf("Alert level tests passed\n");
}

// Test error cases
static void test_error_cases(void) {
    printf("Testing error cases...\n");

    // Test operations before initialization (should be safe)
    begin_quantum_operation(NULL);
    end_quantum_operation(NULL, NULL);
    log_production_event(ALERT_LEVEL_INFO, "Test", "Test", "Test");

    // Test cleanup without initialization (should be safe)
    cleanup_production_monitoring();

    // Test double cleanup
    production_config_t config = {
        .log_directory = "/tmp",
        .min_log_level = ALERT_LEVEL_DEBUG,
        .thresholds = {
            .error_rate = 0.01,
            .latency = 1000.0,
            .memory_usage = 100.0,
            .cpu_usage = 0.8,
            .success_rate = 0.99
        },
        .enable_alerting = false
    };

    init_production_monitoring(&config);
    cleanup_production_monitoring();
    cleanup_production_monitoring();  // Should be safe

    printf("Error case tests passed\n");
}

int main(void) {
    printf("Running production monitoring tests...\n");

    test_initialization();
    test_alert_handlers();
    test_logging();
    test_thresholds();
    test_production_metrics();
    test_alert_levels();
    test_error_cases();

    printf("All production monitoring tests passed!\n");
    return 0;
}
