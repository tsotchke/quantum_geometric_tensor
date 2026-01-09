/**
 * @file production_monitor.c
 * @brief Production monitoring system for quantum geometric operations
 */

#include "quantum_geometric/core/production_monitor.h"
#include "quantum_geometric/core/performance_monitor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <sys/time.h>

// Monitor state
static struct {
    bool initialized;
    FILE* log_file;
    pthread_mutex_t log_mutex;
    alert_callback alert_handlers[MAX_ALERT_HANDLERS];
    size_t num_alert_handlers;
    pthread_t monitor_thread;
    bool monitor_running;
    threshold_config_t thresholds;
} monitor_state = {0};

// Forward declarations
static void* monitor_thread_func(void* arg);
static void check_thresholds(void);
static void log_metrics(const char* component, const char* event, const char* details);
static void trigger_alerts(alert_level_t level, const char* message);

bool init_production_monitoring(const production_config_t* config) {
    if (monitor_state.initialized) {
        return true;  // Already initialized
    }

    // Initialize performance monitoring
    if (!init_performance_monitoring()) {
        return false;
    }

    // Open log file
    char filename[256];
    time_t now = time(NULL);
    strftime(filename, sizeof(filename), "quantum_geometric_%Y%m%d_%H%M%S.log", localtime(&now));
    
    monitor_state.log_file = fopen(filename, "a");
    if (!monitor_state.log_file) {
        return false;
    }

    // Initialize mutex
    if (pthread_mutex_init(&monitor_state.log_mutex, NULL) != 0) {
        fclose(monitor_state.log_file);
        return false;
    }

    // Set thresholds
    monitor_state.thresholds = config->thresholds;

    // Start monitoring thread
    monitor_state.monitor_running = true;
    if (pthread_create(&monitor_state.monitor_thread, NULL, monitor_thread_func, NULL) != 0) {
        pthread_mutex_destroy(&monitor_state.log_mutex);
        fclose(monitor_state.log_file);
        return false;
    }

    monitor_state.initialized = true;
    log_metrics("Monitor", "Initialization", "Production monitoring system initialized");
    return true;
}

void cleanup_production_monitoring(void) {
    if (!monitor_state.initialized) {
        return;
    }

    // Stop monitoring thread
    monitor_state.monitor_running = false;
    pthread_join(monitor_state.monitor_thread, NULL);

    // Cleanup resources
    pthread_mutex_destroy(&monitor_state.log_mutex);
    if (monitor_state.log_file) {
        fclose(monitor_state.log_file);
    }

    cleanup_performance_monitoring();
    monitor_state.initialized = false;
}

bool register_alert_handler(alert_callback handler) {
    if (!monitor_state.initialized || 
        monitor_state.num_alert_handlers >= MAX_ALERT_HANDLERS) {
        return false;
    }

    monitor_state.alert_handlers[monitor_state.num_alert_handlers++] = handler;
    return true;
}

void log_quantum_event(const char* component, const char* event, const char* details) {
    if (!monitor_state.initialized) {
        return;
    }
    log_metrics(component, event, details);
}

void record_quantum_operation(const quantum_operation_t* operation) {
    if (!monitor_state.initialized) {
        return;
    }

    // Record timing
    start_operation_timing(operation->name);
    
    // Log operation start
    char details[256];
    snprintf(details, sizeof(details), "Operation started: %s (type=%d)", 
             operation->name, operation->type);
    log_metrics("Quantum", "Operation", details);
    
    // Update resource usage
    update_resource_usage();
    
    // Check thresholds
    check_thresholds();
}

void record_quantum_result(const quantum_operation_t* operation, 
                         const quantum_result_t* result) {
    if (!monitor_state.initialized) {
        return;
    }

    // End timing
    end_operation_timing(operation->name);
    
    // Record result
    record_operation_result(result->success, result->false_positive);
    
    // Log completion
    char details[256];
    snprintf(details, sizeof(details), 
             "Operation completed: %s (success=%d, error_code=%d)", 
             operation->name, result->success, result->error_code);
    log_metrics("Quantum", "Result", details);
    
    // Check thresholds
    check_thresholds();
}

static void* monitor_thread_func(void* arg) {
    while (monitor_state.monitor_running) {
        // Get current metrics
        PerformanceMetrics metrics = get_performance_metrics();
        
        // Check thresholds
        if (metrics.error_rate > monitor_state.thresholds.error_rate) {
            trigger_alerts(ALERT_LEVEL_ERROR, "Error rate threshold exceeded");
        }
        
        if (metrics.avg_latency > monitor_state.thresholds.latency) {
            trigger_alerts(ALERT_LEVEL_WARNING, "Latency threshold exceeded");
        }
        
        if (metrics.peak_memory_usage > monitor_state.thresholds.memory_usage) {
            trigger_alerts(ALERT_LEVEL_WARNING, "Memory usage threshold exceeded");
        }
        
        if (metrics.avg_cpu_utilization > monitor_state.thresholds.cpu_usage) {
            trigger_alerts(ALERT_LEVEL_WARNING, "CPU usage threshold exceeded");
        }
        
        if (metrics.success_rate < monitor_state.thresholds.success_rate) {
            trigger_alerts(ALERT_LEVEL_ERROR, "Success rate below threshold");
        }
        
        // Sleep for monitoring interval
        struct timespec sleep_time = {.tv_sec = 0, .tv_nsec = 100000000}; // 100ms
        nanosleep(&sleep_time, NULL);
    }
    
    return NULL;
}

static void check_thresholds(void) {
    PerformanceMetrics metrics = get_performance_metrics();
    
    // Check each threshold and trigger alerts if exceeded
    if (metrics.error_rate > monitor_state.thresholds.error_rate) {
        trigger_alerts(ALERT_LEVEL_ERROR, "Error rate threshold exceeded");
    }
    
    if (metrics.avg_latency > monitor_state.thresholds.latency) {
        trigger_alerts(ALERT_LEVEL_WARNING, "Latency threshold exceeded");
    }
    
    if (metrics.peak_memory_usage > monitor_state.thresholds.memory_usage) {
        trigger_alerts(ALERT_LEVEL_WARNING, "Memory usage threshold exceeded");
    }
    
    if (metrics.avg_cpu_utilization > monitor_state.thresholds.cpu_usage) {
        trigger_alerts(ALERT_LEVEL_WARNING, "CPU usage threshold exceeded");
    }
    
    if (metrics.success_rate < monitor_state.thresholds.success_rate) {
        trigger_alerts(ALERT_LEVEL_ERROR, "Success rate below threshold");
    }
}

static void log_metrics(const char* component, const char* event, const char* details) {
    if (!monitor_state.log_file) {
        return;
    }

    // Get current timestamp
    struct timeval tv;
    gettimeofday(&tv, NULL);
    
    char timestamp[32];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", localtime(&tv.tv_sec));
    
    // Format log entry
    char log_entry[1024];
    snprintf(log_entry, sizeof(log_entry),
             "[%s.%06ld] [%s] [%s] %s\n",
             timestamp, (long)tv.tv_usec, component, event, details);
    
    // Write to log file
    pthread_mutex_lock(&monitor_state.log_mutex);
    fputs(log_entry, monitor_state.log_file);
    fflush(monitor_state.log_file);
    pthread_mutex_unlock(&monitor_state.log_mutex);
}

static void trigger_alerts(alert_level_t level, const char* message) {
    // Log alert
    log_metrics("Alert", alert_level_str(level), message);
    
    // Trigger all registered handlers
    for (size_t i = 0; i < monitor_state.num_alert_handlers; i++) {
        if (monitor_state.alert_handlers[i]) {
            monitor_state.alert_handlers[i](level, message);
        }
    }
}

const char* alert_level_str(alert_level_t level) {
    switch (level) {
        case ALERT_LEVEL_DEBUG:    return "DEBUG";
        case ALERT_LEVEL_INFO:     return "INFO";
        case ALERT_LEVEL_WARNING:  return "WARNING";
        case ALERT_LEVEL_ERROR:    return "ERROR";
        case ALERT_LEVEL_CRITICAL: return "CRITICAL";
        case ALERT_LEVEL_FATAL:    return "FATAL";
        default:                   return "UNKNOWN";
    }
}

// Alias for cleanup
void shutdown_production_monitoring(void) {
    cleanup_production_monitoring();
}

// Unregister an alert handler
bool unregister_alert_handler(alert_callback callback) {
    if (!monitor_state.initialized || !callback) {
        return false;
    }

    for (size_t i = 0; i < monitor_state.num_alert_handlers; i++) {
        if (monitor_state.alert_handlers[i] == callback) {
            // Shift remaining handlers
            for (size_t j = i; j < monitor_state.num_alert_handlers - 1; j++) {
                monitor_state.alert_handlers[j] = monitor_state.alert_handlers[j + 1];
            }
            monitor_state.num_alert_handlers--;
            return true;
        }
    }

    return false;
}

// Start monitoring a quantum operation (alias for record_quantum_operation)
void begin_quantum_operation(const quantum_operation_t* operation) {
    if (!monitor_state.initialized || !operation) {
        return;
    }

    start_operation_timing(operation->name);

    char details[256];
    snprintf(details, sizeof(details), "Operation started: %s (type=%d, qubits=%zu)",
             operation->name, operation->type, operation->num_qubits);
    log_metrics("Quantum", "Begin", details);

    update_resource_usage();
}

// End monitoring a quantum operation (alias for record_quantum_result)
void end_quantum_operation(const quantum_operation_t* operation, const quantum_result_t* result) {
    if (!monitor_state.initialized) {
        return;
    }

    if (operation) {
        end_operation_timing(operation->name);
    }

    if (result) {
        record_operation_result(result->success, result->execution_time);

        char details[256];
        snprintf(details, sizeof(details),
                 "Operation completed: %s (success=%d, fidelity=%.4f, error_code=%d)",
                 operation ? operation->name : "unknown",
                 result->success, result->fidelity, result->error_code);
        log_metrics("Quantum", "End", details);

        if (!result->success) {
            trigger_alerts(ALERT_LEVEL_ERROR, result->error_message ? result->error_message : "Operation failed");
        }
    }

    check_thresholds();
}

// Log production events with severity level
void log_production_event(alert_level_t level, const char* component,
                         const char* event, const char* details) {
    if (!monitor_state.initialized) {
        return;
    }

    log_metrics(component, event, details);

    // If level is warning or above, also trigger alerts
    if (level >= ALERT_LEVEL_WARNING) {
        char message[512];
        snprintf(message, sizeof(message), "[%s] %s: %s", component, event, details);
        trigger_alerts(level, message);
    }
}

// Get current production metrics
bool get_production_metrics(double* error_rate, double* avg_latency,
                           double* memory_usage, double* cpu_usage) {
    if (!monitor_state.initialized) {
        if (error_rate) *error_rate = 0;
        if (avg_latency) *avg_latency = 0;
        if (memory_usage) *memory_usage = 0;
        if (cpu_usage) *cpu_usage = 0;
        return false;
    }

    PerformanceMetrics metrics = get_performance_metrics();

    if (error_rate) *error_rate = metrics.error_rate;
    if (avg_latency) *avg_latency = metrics.avg_latency;
    if (memory_usage) *memory_usage = metrics.peak_memory_usage;
    if (cpu_usage) *cpu_usage = metrics.avg_cpu_utilization;

    return true;
}

// Set threshold values
void set_error_threshold(double threshold) {
    monitor_state.thresholds.error_rate = threshold;
}

void set_latency_threshold(double threshold_ms) {
    monitor_state.thresholds.latency = threshold_ms;
}

void set_memory_threshold(double threshold_percent) {
    monitor_state.thresholds.memory_usage = threshold_percent;
}

void set_cpu_threshold(double threshold_percent) {
    monitor_state.thresholds.cpu_usage = threshold_percent;
}

// Health check
bool production_health_check(void) {
    if (!monitor_state.initialized) {
        return false;
    }

    PerformanceMetrics metrics = get_performance_metrics();

    // Check all thresholds
    if (metrics.error_rate > monitor_state.thresholds.error_rate) {
        return false;
    }
    if (metrics.avg_latency > monitor_state.thresholds.latency) {
        return false;
    }
    if (metrics.peak_memory_usage > monitor_state.thresholds.memory_usage) {
        return false;
    }
    if (metrics.avg_cpu_utilization > monitor_state.thresholds.cpu_usage) {
        return false;
    }
    if (metrics.success_rate < monitor_state.thresholds.success_rate) {
        return false;
    }

    return true;
}
