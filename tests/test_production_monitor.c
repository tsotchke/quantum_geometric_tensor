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

// Mock alert handler for testing
static struct {
    alert_level_t last_level;
    char last_message[256];
    int alert_count;
} mock_handler_state = {0};

static void mock_alert_handler(alert_level_t level, const char* message) {
    mock_handler_state.last_level = level;
    strncpy(mock_handler_state.last_message, message, sizeof(mock_handler_state.last_message) - 1);
    mock_handler_state.alert_count++;
}

// Test initialization and cleanup
static void test_initialization(void) {
    printf("Testing initialization...\n");
    
    // Test with NULL config
    assert(!init_production_monitoring(NULL));
    
    // Test with valid config
    production_config_t config = {
        .thresholds = {
            .error_rate = 0.01,
            .latency = 1000.0,
            .memory_usage = 1000.0,
            .cpu_usage = 0.8,
            .success_rate = 0.99
        },
        .logging = {
            .log_dir = "/tmp",
            .console_output = true,
            .file_output = true,
            .syslog_output = false
        },
        .alerts = {
            .email_alerts = false,
            .slack_alerts = false,
            .pagerduty_alerts = false
        }
    };
    
    assert(init_production_monitoring(&config));
    
    // Test double initialization
    assert(init_production_monitoring(&config));  // Should succeed but be idempotent
    
    cleanup_production_monitoring();
    printf("Initialization tests passed\n");
}

// Test alert handlers
static void test_alert_handlers(void) {
    printf("Testing alert handlers...\n");
    
    production_config_t config = {
        .thresholds = {
            .error_rate = 0.01,
            .latency = 1000.0,
            .memory_usage = 1000.0,
            .cpu_usage = 0.8,
            .success_rate = 0.99
        }
    };
    
    assert(init_production_monitoring(&config));
    
    // Reset mock handler state
    memset(&mock_handler_state, 0, sizeof(mock_handler_state));
    
    // Register handler
    assert(register_alert_handler(mock_alert_handler));
    
    // Trigger alerts through operations
    quantum_operation_t op = {
        .name = "test_operation",
        .type = 1,
        .context = NULL
    };
    
    quantum_result_t result = {
        .success = false,
        .false_positive = false,
        .error_code = 1,
        .error_msg = "Test error"
    };
    
    // Record failed operation to trigger alert
    record_quantum_operation(&op);
    record_quantum_result(&op, &result);
    
    // Verify alert was triggered
    assert(mock_handler_state.alert_count > 0);
    assert(mock_handler_state.last_level == ALERT_LEVEL_ERROR);
    
    cleanup_production_monitoring();
    printf("Alert handler tests passed\n");
}

// Test logging functionality
static void test_logging(void) {
    printf("Testing logging...\n");
    
    production_config_t config = {
        .thresholds = {
            .error_rate = 0.01,
            .latency = 1000.0,
            .memory_usage = 1000.0,
            .cpu_usage = 0.8,
            .success_rate = 0.99
        },
        .logging = {
            .log_dir = "/tmp",
            .console_output = true,
            .file_output = true,
            .syslog_output = false
        }
    };
    
    assert(init_production_monitoring(&config));
    
    // Log various events
    log_quantum_event("TestComponent", "TestEvent", "Test details");
    
    quantum_operation_t op = {
        .name = "test_operation",
        .type = 1,
        .context = NULL
    };
    
    record_quantum_operation(&op);
    
    quantum_result_t result = {
        .success = true,
        .false_positive = false,
        .error_code = 0,
        .error_msg = NULL
    };
    
    record_quantum_result(&op, &result);
    
    cleanup_production_monitoring();
    printf("Logging tests passed\n");
}

// Test threshold monitoring
static void test_thresholds(void) {
    printf("Testing threshold monitoring...\n");
    
    production_config_t config = {
        .thresholds = {
            .error_rate = 0.01,    // 1% error rate threshold
            .latency = 1000.0,     // 1ms latency threshold
            .memory_usage = 100.0,  // 100MB memory threshold
            .cpu_usage = 0.8,      // 80% CPU threshold
            .success_rate = 0.99   // 99% success rate threshold
        }
    };
    
    assert(init_production_monitoring(&config));
    
    // Reset mock handler state
    memset(&mock_handler_state, 0, sizeof(mock_handler_state));
    assert(register_alert_handler(mock_alert_handler));
    
    // Test error rate threshold
    for (int i = 0; i < 100; i++) {
        quantum_operation_t op = {
            .name = "test_operation",
            .type = 1,
            .context = NULL
        };
        
        quantum_result_t result = {
            .success = (i < 98),  // 2% error rate
            .false_positive = false,
            .error_code = i < 98 ? 0 : 1,
            .error_msg = i < 98 ? NULL : "Test error"
        };
        
        record_quantum_operation(&op);
        record_quantum_result(&op, &result);
    }
    
    // Verify error rate threshold alert was triggered
    assert(mock_handler_state.alert_count > 0);
    
    cleanup_production_monitoring();
    printf("Threshold monitoring tests passed\n");
}

// Test resource monitoring
static void test_resource_monitoring(void) {
    printf("Testing resource monitoring...\n");
    
    production_config_t config = {
        .thresholds = {
            .error_rate = 0.01,
            .latency = 1000.0,
            .memory_usage = 100.0,  // 100MB threshold
            .cpu_usage = 0.8,
            .success_rate = 0.99
        }
    };
    
    assert(init_production_monitoring(&config));
    
    // Reset mock handler state
    memset(&mock_handler_state, 0, sizeof(mock_handler_state));
    assert(register_alert_handler(mock_alert_handler));
    
    // Allocate memory to trigger threshold
    void* data = malloc(200 * 1024 * 1024);  // 200MB
    assert(data != NULL);
    
    // Force memory usage update
    quantum_operation_t op = {
        .name = "memory_test",
        .type = 1,
        .context = NULL
    };
    
    record_quantum_operation(&op);
    
    // Verify memory threshold alert was triggered
    assert(mock_handler_state.alert_count > 0);
    
    free(data);
    cleanup_production_monitoring();
    printf("Resource monitoring tests passed\n");
}

int main(void) {
    printf("Running production monitoring tests...\n");
    
    test_initialization();
    test_alert_handlers();
    test_logging();
    test_thresholds();
    test_resource_monitoring();
    
    printf("All production monitoring tests passed!\n");
    return 0;
}
