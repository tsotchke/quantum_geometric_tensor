/**
 * @file performance_monitoring.c
 * @brief Performance monitoring system for quantum operations
 */

#include "quantum_geometric/core/performance_monitor.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>

// Performance thresholds from requirements
#define MAX_ERROR_DETECTION_LATENCY 10    // microseconds
#define MAX_CORRECTION_CYCLE_TIME 50      // microseconds
#define MAX_STATE_VERIFICATION_TIME 100   // microseconds
#define MAX_MEMORY_OVERHEAD 20            // percent
#define MAX_CPU_UTILIZATION 80           // percent
#define MAX_GPU_UTILIZATION 90           // percent
#define MIN_SUCCESS_RATE 99              // percent
#define MAX_FALSE_POSITIVE_RATE 1        // percent
#define MIN_RECOVERY_SUCCESS_RATE 99     // percent

// Internal monitoring state
typedef struct {
    // Timing metrics
    struct timespec* operation_start_times;
    struct timespec* operation_end_times;
    double* operation_latencies;
    size_t num_operations;
    
    // Resource usage
    struct rusage initial_usage;
    struct rusage current_usage;
    double peak_memory_usage;
    double avg_cpu_utilization;
    double avg_gpu_utilization;
    
    // Success metrics
    size_t total_operations;
    size_t successful_operations;
    size_t false_positives;
    size_t recovery_attempts;
    size_t successful_recoveries;
    
    // State
    bool initialized;
    bool monitoring_active;
} MonitorState;

static MonitorState monitor_state = {0};

bool init_performance_monitoring(void) {
    if (monitor_state.initialized) {
        return true;
    }

    // Initialize state
    memset(&monitor_state, 0, sizeof(MonitorState));
    
    // Allocate timing arrays
    monitor_state.operation_start_times = calloc(1000, sizeof(struct timespec));
    monitor_state.operation_end_times = calloc(1000, sizeof(struct timespec));
    monitor_state.operation_latencies = calloc(1000, sizeof(double));
    
    if (!monitor_state.operation_start_times ||
        !monitor_state.operation_end_times ||
        !monitor_state.operation_latencies) {
        cleanup_performance_monitoring();
        return false;
    }

    // Get initial resource usage
    if (getrusage(RUSAGE_SELF, &monitor_state.initial_usage) != 0) {
        cleanup_performance_monitoring();
        return false;
    }

    monitor_state.initialized = true;
    monitor_state.monitoring_active = true;
    return true;
}

void cleanup_performance_monitoring(void) {
    free(monitor_state.operation_start_times);
    free(monitor_state.operation_end_times);
    free(monitor_state.operation_latencies);
    memset(&monitor_state, 0, sizeof(MonitorState));
}

void start_operation_timing(const char* operation_name) {
    if (!monitor_state.initialized || !monitor_state.monitoring_active) {
        return;
    }

    size_t idx = monitor_state.num_operations++;
    clock_gettime(CLOCK_MONOTONIC, &monitor_state.operation_start_times[idx]);
}

void end_operation_timing(const char* operation_name) {
    if (!monitor_state.initialized || !monitor_state.monitoring_active ||
        monitor_state.num_operations == 0) {
        return;
    }

    size_t idx = monitor_state.num_operations - 1;
    clock_gettime(CLOCK_MONOTONIC, &monitor_state.operation_end_times[idx]);

    // Calculate latency in microseconds
    double latency = (monitor_state.operation_end_times[idx].tv_sec -
                     monitor_state.operation_start_times[idx].tv_sec) * 1e6 +
                    (monitor_state.operation_end_times[idx].tv_nsec -
                     monitor_state.operation_start_times[idx].tv_nsec) / 1e3;
    
    monitor_state.operation_latencies[idx] = latency;

    // Check against thresholds
    if (strcmp(operation_name, "error_detection") == 0) {
        if (latency > MAX_ERROR_DETECTION_LATENCY) {
            log_performance_warning("Error detection latency exceeded threshold",
                                 latency, MAX_ERROR_DETECTION_LATENCY);
        }
    } else if (strcmp(operation_name, "correction_cycle") == 0) {
        if (latency > MAX_CORRECTION_CYCLE_TIME) {
            log_performance_warning("Correction cycle time exceeded threshold",
                                 latency, MAX_CORRECTION_CYCLE_TIME);
        }
    } else if (strcmp(operation_name, "state_verification") == 0) {
        if (latency > MAX_STATE_VERIFICATION_TIME) {
            log_performance_warning("State verification time exceeded threshold",
                                 latency, MAX_STATE_VERIFICATION_TIME);
        }
    }
}

void update_resource_usage(void) {
    if (!monitor_state.initialized || !monitor_state.monitoring_active) {
        return;
    }

    // Get current resource usage
    if (getrusage(RUSAGE_SELF, &monitor_state.current_usage) != 0) {
        return;
    }

    // Update memory usage (in MB)
    double current_memory = monitor_state.current_usage.ru_maxrss / 1024.0;
    if (current_memory > monitor_state.peak_memory_usage) {
        monitor_state.peak_memory_usage = current_memory;
    }

    // Calculate memory overhead percentage
    double initial_memory = monitor_state.initial_usage.ru_maxrss / 1024.0;
    double memory_overhead = ((current_memory - initial_memory) / initial_memory) * 100;

    if (memory_overhead > MAX_MEMORY_OVERHEAD) {
        log_performance_warning("Memory overhead exceeded threshold",
                             memory_overhead, MAX_MEMORY_OVERHEAD);
    }

    // Update CPU utilization
    double cpu_time = (monitor_state.current_usage.ru_utime.tv_sec +
                      monitor_state.current_usage.ru_utime.tv_usec / 1e6) -
                     (monitor_state.initial_usage.ru_utime.tv_sec +
                      monitor_state.initial_usage.ru_utime.tv_usec / 1e6);
    
    struct timespec current_time;
    clock_gettime(CLOCK_MONOTONIC, &current_time);
    double elapsed_time = current_time.tv_sec +
                         current_time.tv_nsec / 1e9;

    double cpu_utilization = (cpu_time / elapsed_time) * 100;
    monitor_state.avg_cpu_utilization = 
        (monitor_state.avg_cpu_utilization + cpu_utilization) / 2.0;

    if (cpu_utilization > MAX_CPU_UTILIZATION) {
        log_performance_warning("CPU utilization exceeded threshold",
                             cpu_utilization, MAX_CPU_UTILIZATION);
    }

    // Update GPU utilization if available
    double gpu_utilization = get_gpu_utilization();
    if (gpu_utilization >= 0) {
        monitor_state.avg_gpu_utilization =
            (monitor_state.avg_gpu_utilization + gpu_utilization) / 2.0;

        if (gpu_utilization > MAX_GPU_UTILIZATION) {
            log_performance_warning("GPU utilization exceeded threshold",
                                 gpu_utilization, MAX_GPU_UTILIZATION);
        }
    }
}

void record_operation_result(bool success, bool false_positive) {
    if (!monitor_state.initialized || !monitor_state.monitoring_active) {
        return;
    }

    monitor_state.total_operations++;
    if (success) {
        monitor_state.successful_operations++;
    }
    if (false_positive) {
        monitor_state.false_positives++;
    }

    // Check success rate
    double success_rate = (monitor_state.successful_operations * 100.0) /
                         monitor_state.total_operations;
    if (success_rate < MIN_SUCCESS_RATE) {
        log_performance_warning("Success rate below threshold",
                             success_rate, MIN_SUCCESS_RATE);
    }

    // Check false positive rate
    double false_positive_rate = (monitor_state.false_positives * 100.0) /
                                monitor_state.total_operations;
    if (false_positive_rate > MAX_FALSE_POSITIVE_RATE) {
        log_performance_warning("False positive rate exceeded threshold",
                             false_positive_rate, MAX_FALSE_POSITIVE_RATE);
    }
}

void record_recovery_result(bool success) {
    if (!monitor_state.initialized || !monitor_state.monitoring_active) {
        return;
    }

    monitor_state.recovery_attempts++;
    if (success) {
        monitor_state.successful_recoveries++;
    }

    // Check recovery success rate
    double recovery_rate = (monitor_state.successful_recoveries * 100.0) /
                          monitor_state.recovery_attempts;
    if (recovery_rate < MIN_RECOVERY_SUCCESS_RATE) {
        log_performance_warning("Recovery success rate below threshold",
                             recovery_rate, MIN_RECOVERY_SUCCESS_RATE);
    }
}

PerformanceMetrics get_performance_metrics(void) {
    PerformanceMetrics metrics = {0};
    
    if (!monitor_state.initialized) {
        return metrics;
    }

    // Calculate average latencies
    for (size_t i = 0; i < monitor_state.num_operations; i++) {
        metrics.avg_latency += monitor_state.operation_latencies[i];
    }
    if (monitor_state.num_operations > 0) {
        metrics.avg_latency /= monitor_state.num_operations;
    }

    // Set resource usage
    metrics.peak_memory_usage = monitor_state.peak_memory_usage;
    metrics.avg_cpu_utilization = monitor_state.avg_cpu_utilization;
    metrics.avg_gpu_utilization = monitor_state.avg_gpu_utilization;

    // Calculate success rates
    if (monitor_state.total_operations > 0) {
        metrics.success_rate = (monitor_state.successful_operations * 100.0) /
                             monitor_state.total_operations;
        metrics.false_positive_rate = (monitor_state.false_positives * 100.0) /
                                    monitor_state.total_operations;
    }

    if (monitor_state.recovery_attempts > 0) {
        metrics.recovery_success_rate = 
            (monitor_state.successful_recoveries * 100.0) /
            monitor_state.recovery_attempts;
    }

    return metrics;
}

static void log_performance_warning(const char* message,
                                  double actual_value,
                                  double threshold) {
    // Log warning with details
    printf("Performance Warning: %s\n", message);
    printf("  Actual: %.2f\n", actual_value);
    printf("  Threshold: %.2f\n", threshold);
}

static double get_gpu_utilization(void) {
    // Get GPU utilization through hardware abstraction layer
    // Returns -1 if GPU monitoring not available
    return get_gpu_usage();
}
