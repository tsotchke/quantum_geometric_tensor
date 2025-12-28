#include "quantum_geometric/hybrid/performance_monitoring.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>
#include <unistd.h>
#include <errno.h>

// Platform-specific includes for real monitoring
#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <mach/processor_info.h>
#include <mach/task.h>
#include <sys/sysctl.h>
#include <net/if.h>
#include <ifaddrs.h>
#include <net/if_dl.h>
// IOKit for GPU monitoring
#include <IOKit/IOKitLib.h>
#include <CoreFoundation/CoreFoundation.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#include <sys/statvfs.h>
#endif

// max macro if not defined
#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif

// ============================================================================
// GPU Monitoring State (persistent across calls)
// ============================================================================
typedef struct {
    bool initialized;
    double last_sample_time;
    double last_utilization;
    // For delta calculations
    uint64_t last_gpu_active_ns;
    uint64_t last_gpu_sample_ns;
#ifdef __APPLE__
    // Metal GPU tracking via IOKit
    void* io_gpu_device;
#endif
} GPUMonitorState;

static GPUMonitorState g_gpu_state = {0};

// ============================================================================
// Network Monitoring State
// ============================================================================
typedef struct {
    bool initialized;
    double last_sample_time;
    uint64_t last_bytes_sent;
    uint64_t last_bytes_recv;
    double current_send_rate;  // bytes/sec
    double current_recv_rate;  // bytes/sec
} NetworkMonitorState;

static NetworkMonitorState g_network_state = {0};

// ============================================================================
// Quantum Hardware Monitoring State
// ============================================================================
typedef struct {
    bool initialized;
    size_t active_qubits;
    size_t max_qubits;
    size_t circuit_depth;
    size_t gates_executed;
    double coherence_time_used;
    double max_coherence_time;  // T2 time in microseconds
    double error_rate_accumulator;
    size_t error_samples;
    double fidelity_accumulator;
    size_t fidelity_samples;
    // Power estimation
    double cryostat_power_watts;
    double control_power_watts;
    double last_measurement_time;
} QuantumHardwareState;

static QuantumHardwareState g_quantum_state = {0};

// Forward declarations for static helper functions
static void update_resource_metrics(PerformanceMonitor* monitor);
static void log_performance_metrics(PerformanceMonitor* monitor);
static double get_elapsed_time(const PerformanceMonitor* monitor);
static double get_gpu_utilization(void);
static double get_network_bandwidth(double* send_rate, double* recv_rate);
static double estimate_quantum_resources(const PerformanceMonitor* monitor);
static double estimate_quantum_energy(const PerformanceMonitor* monitor);
static double estimate_classical_energy(const PerformanceMonitor* monitor);
static void init_gpu_monitoring(void);
static void init_network_monitoring(void);
static void init_quantum_monitoring(void);

// MonitoringMetrics and PerformanceMonitor are defined in the header

// ============================================================================
// Initialization Functions
// ============================================================================

static void init_gpu_monitoring(void) {
    if (g_gpu_state.initialized) return;

    g_gpu_state.initialized = true;
    g_gpu_state.last_utilization = 0.0;

    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    g_gpu_state.last_sample_time = now.tv_sec + now.tv_nsec * 1e-9;
    g_gpu_state.last_gpu_active_ns = 0;
    g_gpu_state.last_gpu_sample_ns = (uint64_t)(now.tv_sec * 1000000000ULL + now.tv_nsec);

#ifdef __APPLE__
    // On macOS, GPU stats are available via IOKit or Metal
    // We'll use activity sampling approach
    g_gpu_state.io_gpu_device = NULL;
#endif
}

static void init_network_monitoring(void) {
    if (g_network_state.initialized) return;

    g_network_state.initialized = true;
    g_network_state.last_bytes_sent = 0;
    g_network_state.last_bytes_recv = 0;
    g_network_state.current_send_rate = 0.0;
    g_network_state.current_recv_rate = 0.0;

    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    g_network_state.last_sample_time = now.tv_sec + now.tv_nsec * 1e-9;

    // Get initial byte counts
#ifdef __APPLE__
    struct ifaddrs* ifap = NULL;
    if (getifaddrs(&ifap) == 0) {
        for (struct ifaddrs* ifa = ifap; ifa != NULL; ifa = ifa->ifa_next) {
            if (ifa->ifa_addr == NULL) continue;
            if (ifa->ifa_addr->sa_family != AF_LINK) continue;
            if (ifa->ifa_data == NULL) continue;

            // Skip loopback
            if (ifa->ifa_flags & IFF_LOOPBACK) continue;

            struct if_data* if_data = (struct if_data*)ifa->ifa_data;
            g_network_state.last_bytes_sent += if_data->ifi_obytes;
            g_network_state.last_bytes_recv += if_data->ifi_ibytes;
        }
        freeifaddrs(ifap);
    }
#elif defined(__linux__)
    FILE* fp = fopen("/proc/net/dev", "r");
    if (fp) {
        char line[512];
        // Skip header lines
        if (fgets(line, sizeof(line), fp) && fgets(line, sizeof(line), fp)) {
            while (fgets(line, sizeof(line), fp)) {
                char iface[32];
                unsigned long long rbytes, tbytes;
                unsigned long long dummy;
                if (sscanf(line, "%31[^:]: %llu %llu %llu %llu %llu %llu %llu %llu %llu",
                           iface, &rbytes, &dummy, &dummy, &dummy, &dummy, &dummy, &dummy, &dummy, &tbytes) >= 10) {
                    // Skip loopback
                    if (strncmp(iface, "lo", 2) != 0) {
                        g_network_state.last_bytes_recv += rbytes;
                        g_network_state.last_bytes_sent += tbytes;
                    }
                }
            }
        }
        fclose(fp);
    }
#endif
}

static void init_quantum_monitoring(void) {
    if (g_quantum_state.initialized) return;

    g_quantum_state.initialized = true;
    g_quantum_state.active_qubits = 0;
    g_quantum_state.max_qubits = 127;  // Default IBM Eagle processor
    g_quantum_state.circuit_depth = 0;
    g_quantum_state.gates_executed = 0;
    g_quantum_state.coherence_time_used = 0.0;
    g_quantum_state.max_coherence_time = 100.0;  // 100 microseconds T2 typical
    g_quantum_state.error_rate_accumulator = 0.0;
    g_quantum_state.error_samples = 0;
    g_quantum_state.fidelity_accumulator = 0.0;
    g_quantum_state.fidelity_samples = 0;

    // Typical power consumption for superconducting quantum computer
    g_quantum_state.cryostat_power_watts = 15000.0;  // 15kW for dilution refrigerator
    g_quantum_state.control_power_watts = 5000.0;    // 5kW for control electronics

    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    g_quantum_state.last_measurement_time = now.tv_sec + now.tv_nsec * 1e-9;
}

// Initialize hybrid performance monitoring (renamed to avoid conflict with core/performance_monitor.c)
PerformanceMonitor* init_hybrid_performance_monitor(void) {
    PerformanceMonitor* monitor = malloc(sizeof(PerformanceMonitor));
    if (!monitor) return NULL;

    // Initialize current metrics
    memset(&monitor->current, 0, sizeof(MonitoringMetrics));
    clock_gettime(CLOCK_MONOTONIC, &monitor->current.start_time);
    monitor->current.last_update = monitor->current.start_time;

    // Initialize history
    monitor->history_capacity = 1000;
    monitor->history = malloc(
        monitor->history_capacity * sizeof(MonitoringMetrics));

    if (!monitor->history) {
        free(monitor);
        return NULL;
    }

    monitor->history_size = 0;
    monitor->monitoring_enabled = true;

    // Initialize subsystem monitors
    init_gpu_monitoring();
    init_network_monitoring();
    init_quantum_monitoring();

    // Open log file
    monitor->log_file = fopen("performance_log.txt", "w");
    if (!monitor->log_file) {
        free(monitor->history);
        free(monitor);
        return NULL;
    }

    return monitor;
}

// Start monitoring operation
void start_operation(PerformanceMonitor* monitor,
                    MonitoringOperationType type) {
    if (!monitor || !monitor->monitoring_enabled) return;

    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);

    // Update operation counters
    switch (type) {
        case MONITOR_OP_QUANTUM:
            monitor->current.num_quantum_operations++;
            break;

        case MONITOR_OP_CLASSICAL:
            monitor->current.num_classical_operations++;
            break;

        case MONITOR_OP_COMMUNICATION:
            monitor->current.num_communications++;
            break;
    }

    monitor->current.last_update = now;
}

// End monitoring operation
void end_operation(PerformanceMonitor* monitor,
                  MonitoringOperationType type) {
    if (!monitor || !monitor->monitoring_enabled) return;

    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);

    // Calculate elapsed time
    double elapsed = (now.tv_sec - monitor->current.last_update.tv_sec) +
                    (now.tv_nsec - monitor->current.last_update.tv_nsec) * 1e-9;

    // Update timing metrics
    switch (type) {
        case MONITOR_OP_QUANTUM:
            monitor->current.quantum_execution_time += elapsed;
            break;

        case MONITOR_OP_CLASSICAL:
            monitor->current.classical_execution_time += elapsed;
            break;

        case MONITOR_OP_COMMUNICATION:
            monitor->current.communication_overhead += elapsed;
            break;
    }

    // Update resource usage
    update_resource_metrics(monitor);

    // Log metrics
    log_performance_metrics(monitor);
}

// Update resource metrics
static void update_resource_metrics(PerformanceMonitor* monitor) {
    // Get CPU usage
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    
    double user_time = usage.ru_utime.tv_sec +
                      usage.ru_utime.tv_usec * 1e-6;
    double sys_time = usage.ru_stime.tv_sec +
                     usage.ru_stime.tv_usec * 1e-6;
    
    monitor->current.cpu_utilization =
        (user_time + sys_time) / get_elapsed_time(monitor);
    
    // Get memory usage
    monitor->current.memory_usage =
        (double)usage.ru_maxrss / 1024.0;  // Convert to MB
    
    // Get GPU utilization if available
    if (monitor->current.gpu_utilization >= 0) {
        monitor->current.gpu_utilization =
            get_gpu_utilization();
    }
    
    // Estimate quantum resource usage
    monitor->current.quantum_resource_usage =
        estimate_quantum_resources(monitor);
    
    // Update energy consumption
    monitor->current.quantum_energy_consumption +=
        estimate_quantum_energy(monitor);
    monitor->current.classical_energy_consumption +=
        estimate_classical_energy(monitor);
}

// Log performance metrics
static void log_performance_metrics(PerformanceMonitor* monitor) {
    if (!monitor->log_file) return;
    
    // Add to history
    if (monitor->history_size < monitor->history_capacity) {
        monitor->history[monitor->history_size++] = monitor->current;
    }
    
    // Write to log file
    fprintf(monitor->log_file,
            "Time: %.3f s\n"
            "Quantum Execution: %.3f s\n"
            "Classical Execution: %.3f s\n"
            "Communication: %.3f s\n"
            "CPU Usage: %.1f%%\n"
            "GPU Usage: %.1f%%\n"
            "Memory: %.1f MB\n"
            "Quantum Resources: %.1f%%\n"
            "Error Rate: %.2e\n"
            "Fidelity: %.3f\n"
            "Energy: %.1f J\n\n",
            get_elapsed_time(monitor),
            monitor->current.quantum_execution_time,
            monitor->current.classical_execution_time,
            monitor->current.communication_overhead,
            monitor->current.cpu_utilization * 100.0,
            monitor->current.gpu_utilization * 100.0,
            monitor->current.memory_usage,
            monitor->current.quantum_resource_usage * 100.0,
            monitor->current.quantum_error_rate,
            monitor->current.total_fidelity,
            monitor->current.quantum_energy_consumption +
            monitor->current.classical_energy_consumption);
    
    fflush(monitor->log_file);
}

// Get performance summary
PerformanceSummary get_performance_summary(
    const PerformanceMonitor* monitor) {
    
    PerformanceSummary summary = {0};
    
    if (!monitor || monitor->history_size == 0) {
        return summary;
    }
    
    // Calculate averages
    for (size_t i = 0; i < monitor->history_size; i++) {
        summary.avg_quantum_time +=
            monitor->history[i].quantum_execution_time;
        summary.avg_classical_time +=
            monitor->history[i].classical_execution_time;
        summary.avg_communication_time +=
            monitor->history[i].communication_overhead;
        summary.avg_error_rate +=
            monitor->history[i].quantum_error_rate;
        summary.avg_fidelity +=
            monitor->history[i].total_fidelity;
        summary.total_energy +=
            monitor->history[i].quantum_energy_consumption +
            monitor->history[i].classical_energy_consumption;
    }
    
    double n = (double)monitor->history_size;
    summary.avg_quantum_time /= n;
    summary.avg_classical_time /= n;
    summary.avg_communication_time /= n;
    summary.avg_error_rate /= n;
    summary.avg_fidelity /= n;
    
    // Find peak resource usage
    for (size_t i = 0; i < monitor->history_size; i++) {
        summary.peak_memory = max(summary.peak_memory,
            monitor->history[i].memory_usage);
        summary.peak_quantum_resources = max(
            summary.peak_quantum_resources,
            monitor->history[i].quantum_resource_usage);
    }
    
    return summary;
}

// Helper functions

static double get_elapsed_time(const PerformanceMonitor* monitor) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    
    return (now.tv_sec - monitor->current.start_time.tv_sec) +
           (now.tv_nsec - monitor->current.start_time.tv_nsec) * 1e-9;
}

// ============================================================================
// GPU Utilization Monitoring
// ============================================================================

#ifdef __APPLE__
// IOKit-based GPU performance counter reading for macOS
static double get_gpu_utilization_iokit(void) {
    double utilization = 0.0;

    // Find GPU accelerator service
    io_iterator_t iterator;
    CFMutableDictionaryRef matching = IOServiceMatching("IOAccelerator");
    if (!matching) {
        return 0.0;
    }

    // Use kIOMainPortDefault (macOS 12+) or fallback to kIOMasterPortDefault
#if defined(__MAC_12_0) && __MAC_OS_X_VERSION_MIN_REQUIRED >= __MAC_12_0
    kern_return_t kr = IOServiceGetMatchingServices(kIOMainPortDefault, matching, &iterator);
#else
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wdeprecated-declarations"
    kern_return_t kr = IOServiceGetMatchingServices(kIOMasterPortDefault, matching, &iterator);
    #pragma clang diagnostic pop
#endif
    if (kr != KERN_SUCCESS) {
        return 0.0;
    }

    io_service_t service;
    while ((service = IOIteratorNext(iterator)) != IO_OBJECT_NULL) {
        CFMutableDictionaryRef properties = NULL;
        kr = IORegistryEntryCreateCFProperties(service, &properties, kCFAllocatorDefault, 0);

        if (kr == KERN_SUCCESS && properties) {
            // Look for PerformanceStatistics dictionary
            CFDictionaryRef perf_stats = CFDictionaryGetValue(properties, CFSTR("PerformanceStatistics"));
            if (perf_stats && CFGetTypeID(perf_stats) == CFDictionaryGetTypeID()) {
                // Try to get GPU utilization - key varies by GPU vendor

                // Apple Silicon GPUs: "Device Utilization %"
                CFNumberRef util_num = CFDictionaryGetValue(perf_stats, CFSTR("Device Utilization %"));
                if (util_num && CFGetTypeID(util_num) == CFNumberGetTypeID()) {
                    int64_t util_val = 0;
                    if (CFNumberGetValue(util_num, kCFNumberSInt64Type, &util_val)) {
                        utilization = util_val / 100.0;
                    }
                }

                // AMD GPUs: "GPU Activity(%)"
                if (utilization == 0.0) {
                    util_num = CFDictionaryGetValue(perf_stats, CFSTR("GPU Activity(%)"));
                    if (util_num && CFGetTypeID(util_num) == CFNumberGetTypeID()) {
                        int64_t util_val = 0;
                        if (CFNumberGetValue(util_num, kCFNumberSInt64Type, &util_val)) {
                            utilization = util_val / 100.0;
                        }
                    }
                }

                // Intel GPUs: "GPU Core Utilization"
                if (utilization == 0.0) {
                    util_num = CFDictionaryGetValue(perf_stats, CFSTR("GPU Core Utilization"));
                    if (util_num && CFGetTypeID(util_num) == CFNumberGetTypeID()) {
                        int64_t util_val = 0;
                        if (CFNumberGetValue(util_num, kCFNumberSInt64Type, &util_val)) {
                            utilization = util_val / 100.0;
                        }
                    }
                }

                // Alternative: use "In use by this process" flag
                if (utilization == 0.0) {
                    CFBooleanRef in_use = CFDictionaryGetValue(perf_stats, CFSTR("In use by this process"));
                    if (in_use && CFGetTypeID(in_use) == CFBooleanGetTypeID()) {
                        if (CFBooleanGetValue(in_use)) {
                            // GPU is in use, estimate based on command buffer activity
                            CFNumberRef cmd_buffers = CFDictionaryGetValue(perf_stats, CFSTR("CommandBufferCount"));
                            if (cmd_buffers && CFGetTypeID(cmd_buffers) == CFNumberGetTypeID()) {
                                int64_t count = 0;
                                CFNumberGetValue(cmd_buffers, kCFNumberSInt64Type, &count);
                                // Rough estimate: more command buffers = higher utilization
                                utilization = count > 0 ? (count > 100 ? 1.0 : count / 100.0) : 0.1;
                            } else {
                                utilization = 0.5;  // GPU active but no detailed metrics
                            }
                        }
                    }
                }
            }

            CFRelease(properties);
        }

        IOObjectRelease(service);

        // If we found utilization, use the first GPU
        if (utilization > 0.0) {
            break;
        }
    }

    IOObjectRelease(iterator);
    return utilization;
}
#endif

static double get_gpu_utilization(void) {
    if (!g_gpu_state.initialized) {
        init_gpu_monitoring();
    }

    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    double current_time = now.tv_sec + now.tv_nsec * 1e-9;
    double dt = current_time - g_gpu_state.last_sample_time;

    // Avoid sampling too frequently (minimum 100ms between samples)
    if (dt < 0.1) {
        return g_gpu_state.last_utilization;
    }

#ifdef __APPLE__
    // Use IOKit to get real GPU utilization
    g_gpu_state.last_utilization = get_gpu_utilization_iokit();

#elif defined(__linux__)
    // Linux: Try NVIDIA GPU via nvidia-smi or /sys/class/drm
    // First try nvidia-smi (works for NVIDIA GPUs)
    FILE* fp = popen("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null", "r");
    if (fp) {
        int util = 0;
        if (fscanf(fp, "%d", &util) == 1) {
            g_gpu_state.last_utilization = util / 100.0;
            pclose(fp);
            g_gpu_state.last_sample_time = current_time;
            return g_gpu_state.last_utilization;
        }
        pclose(fp);
    }

    // Try AMD GPU via /sys/class/drm/card0/device/gpu_busy_percent
    fp = fopen("/sys/class/drm/card0/device/gpu_busy_percent", "r");
    if (fp) {
        int busy = 0;
        if (fscanf(fp, "%d", &busy) == 1) {
            g_gpu_state.last_utilization = busy / 100.0;
        }
        fclose(fp);
    } else {
        // Fallback: check for intel GPU via i915 driver
        fp = fopen("/sys/kernel/debug/dri/0/i915_gem_objects", "r");
        if (fp) {
            // Parse active objects as a proxy for GPU activity
            char line[256];
            while (fgets(line, sizeof(line), fp)) {
                if (strstr(line, "active")) {
                    int active_objs = 0;
                    if (sscanf(line, "%d", &active_objs) == 1) {
                        g_gpu_state.last_utilization = active_objs > 0 ? 0.5 : 0.0;
                        break;
                    }
                }
            }
            fclose(fp);
        }
    }
#endif

    g_gpu_state.last_sample_time = current_time;
    return g_gpu_state.last_utilization;
}

// ============================================================================
// Network Bandwidth Monitoring
// ============================================================================
static double get_network_bandwidth(double* send_rate, double* recv_rate) {
    if (!g_network_state.initialized) {
        init_network_monitoring();
    }

    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    double current_time = now.tv_sec + now.tv_nsec * 1e-9;
    double dt = current_time - g_network_state.last_sample_time;

    // Minimum sample interval of 100ms
    if (dt < 0.1) {
        if (send_rate) *send_rate = g_network_state.current_send_rate;
        if (recv_rate) *recv_rate = g_network_state.current_recv_rate;
        return g_network_state.current_send_rate + g_network_state.current_recv_rate;
    }

    uint64_t current_bytes_sent = 0;
    uint64_t current_bytes_recv = 0;

#ifdef __APPLE__
    struct ifaddrs* ifap = NULL;
    if (getifaddrs(&ifap) == 0) {
        for (struct ifaddrs* ifa = ifap; ifa != NULL; ifa = ifa->ifa_next) {
            if (ifa->ifa_addr == NULL) continue;
            if (ifa->ifa_addr->sa_family != AF_LINK) continue;
            if (ifa->ifa_data == NULL) continue;
            if (ifa->ifa_flags & IFF_LOOPBACK) continue;

            struct if_data* if_data = (struct if_data*)ifa->ifa_data;
            current_bytes_sent += if_data->ifi_obytes;
            current_bytes_recv += if_data->ifi_ibytes;
        }
        freeifaddrs(ifap);
    }
#elif defined(__linux__)
    FILE* fp = fopen("/proc/net/dev", "r");
    if (fp) {
        char line[512];
        if (fgets(line, sizeof(line), fp) && fgets(line, sizeof(line), fp)) {
            while (fgets(line, sizeof(line), fp)) {
                char iface[32];
                unsigned long long rbytes, tbytes;
                unsigned long long dummy;
                if (sscanf(line, "%31[^:]: %llu %llu %llu %llu %llu %llu %llu %llu %llu",
                           iface, &rbytes, &dummy, &dummy, &dummy, &dummy, &dummy, &dummy, &dummy, &tbytes) >= 10) {
                    if (strncmp(iface, "lo", 2) != 0) {
                        current_bytes_recv += rbytes;
                        current_bytes_sent += tbytes;
                    }
                }
            }
        }
        fclose(fp);
    }
#endif

    // Calculate rates
    if (dt > 0.0) {
        g_network_state.current_send_rate = (double)(current_bytes_sent - g_network_state.last_bytes_sent) / dt;
        g_network_state.current_recv_rate = (double)(current_bytes_recv - g_network_state.last_bytes_recv) / dt;
    }

    g_network_state.last_bytes_sent = current_bytes_sent;
    g_network_state.last_bytes_recv = current_bytes_recv;
    g_network_state.last_sample_time = current_time;

    if (send_rate) *send_rate = g_network_state.current_send_rate;
    if (recv_rate) *recv_rate = g_network_state.current_recv_rate;

    return g_network_state.current_send_rate + g_network_state.current_recv_rate;
}

// ============================================================================
// Quantum Resource Estimation
// ============================================================================
static double estimate_quantum_resources(const PerformanceMonitor* monitor) {
    if (!g_quantum_state.initialized) {
        init_quantum_monitoring();
    }

    // Calculate resource utilization based on multiple factors:
    // 1. Qubit utilization: active_qubits / max_qubits
    // 2. Coherence time utilization: time_used / max_coherence_time
    // 3. Circuit depth factor: deeper circuits use more resources
    // 4. Operation count relative to maximum

    double qubit_util = (g_quantum_state.max_qubits > 0) ?
        (double)g_quantum_state.active_qubits / g_quantum_state.max_qubits : 0.0;

    double coherence_util = (g_quantum_state.max_coherence_time > 0) ?
        g_quantum_state.coherence_time_used / g_quantum_state.max_coherence_time : 0.0;

    double depth_factor = (g_quantum_state.circuit_depth > 0) ?
        1.0 - exp(-0.01 * g_quantum_state.circuit_depth) : 0.0;

    double op_util = (double)monitor->current.num_quantum_operations / (double)MAX_QUANTUM_OPERATIONS;

    // Combined utilization (weighted average)
    double utilization = 0.3 * qubit_util +
                         0.3 * coherence_util +
                         0.2 * depth_factor +
                         0.2 * op_util;

    // Clamp to [0, 1]
    if (utilization > 1.0) utilization = 1.0;
    if (utilization < 0.0) utilization = 0.0;

    return utilization;
}

// ============================================================================
// Energy Estimation Models
// ============================================================================
static double estimate_quantum_energy(const PerformanceMonitor* monitor) {
    if (!g_quantum_state.initialized) {
        init_quantum_monitoring();
    }

    // Quantum computer energy model:
    // 1. Baseline: Dilution refrigerator (constant power draw when running)
    // 2. Control electronics: Proportional to gate operations
    // 3. Measurement: Additional power per measurement

    double execution_time = monitor->current.quantum_execution_time;
    if (execution_time <= 0.0) return 0.0;

    // Cryostat baseline power (always on during operation)
    double cryostat_energy = g_quantum_state.cryostat_power_watts * execution_time;

    // Control electronics power (scales with operation count)
    // Typical: ~10W per active control line, ~100 control lines for 127 qubit system
    double control_power = g_quantum_state.control_power_watts;
    double active_fraction = (g_quantum_state.max_qubits > 0) ?
        (double)g_quantum_state.active_qubits / g_quantum_state.max_qubits : 0.5;
    double control_energy = control_power * active_fraction * execution_time;

    // Gate-level energy (microwave pulses, ~1 nJ per gate typical)
    double gate_energy = g_quantum_state.gates_executed * 1e-9;  // 1 nJ per gate

    // Total quantum energy
    double total_energy = cryostat_energy + control_energy + gate_energy;

    // Apply PUE (Power Usage Effectiveness) factor for datacenter overhead
    const double PUE = 1.5;  // Typical for well-cooled facility
    total_energy *= PUE;

    return total_energy;
}

static double estimate_classical_energy(const PerformanceMonitor* monitor) {
    // Classical computing energy model:
    // 1. CPU: TDP * utilization * time
    // 2. GPU: TDP * utilization * time
    // 3. Memory: ~3W per 8GB DIMM
    // 4. Storage I/O: ~10W for SSD activity

    double cpu_energy = monitor->current.cpu_utilization *
                        CPU_POWER_CONSUMPTION *
                        monitor->current.classical_execution_time;

    double gpu_energy = monitor->current.gpu_utilization *
                        GPU_POWER_CONSUMPTION *
                        monitor->current.classical_execution_time;

    // Memory power estimate (assume 32GB = 4 DIMMs)
    const double MEMORY_POWER = 12.0;  // Watts for 32GB
    double memory_energy = MEMORY_POWER * monitor->current.classical_execution_time;

    // Storage I/O power (estimate based on operation count)
    const double STORAGE_POWER = 5.0;  // Watts for NVMe SSD activity
    double storage_energy = STORAGE_POWER * monitor->current.classical_execution_time;

    double total_energy = cpu_energy + gpu_energy + memory_energy + storage_energy;

    // Apply PUE
    const double PUE = 1.3;  // Better than quantum due to less cooling
    total_energy *= PUE;

    return total_energy;
}

// ============================================================================
// Quantum Hardware State Update Functions (for external callers)
// ============================================================================

void perf_update_quantum_state(size_t active_qubits, size_t circuit_depth,
                               size_t gates_executed, double coherence_time_used) {
    if (!g_quantum_state.initialized) {
        init_quantum_monitoring();
    }

    g_quantum_state.active_qubits = active_qubits;
    g_quantum_state.circuit_depth = circuit_depth;
    g_quantum_state.gates_executed = gates_executed;
    g_quantum_state.coherence_time_used = coherence_time_used;
}

void perf_set_quantum_hardware(size_t max_qubits, double max_coherence_time,
                               double cryostat_power, double control_power) {
    if (!g_quantum_state.initialized) {
        init_quantum_monitoring();
    }

    g_quantum_state.max_qubits = max_qubits;
    g_quantum_state.max_coherence_time = max_coherence_time;
    g_quantum_state.cryostat_power_watts = cryostat_power;
    g_quantum_state.control_power_watts = control_power;
}

void perf_record_error_rate(double error_rate) {
    if (!g_quantum_state.initialized) {
        init_quantum_monitoring();
    }

    g_quantum_state.error_rate_accumulator += error_rate;
    g_quantum_state.error_samples++;
}

void perf_record_fidelity(double fidelity) {
    if (!g_quantum_state.initialized) {
        init_quantum_monitoring();
    }

    g_quantum_state.fidelity_accumulator += fidelity;
    g_quantum_state.fidelity_samples++;
}

double perf_get_average_error_rate(void) {
    if (!g_quantum_state.initialized || g_quantum_state.error_samples == 0) {
        return 0.0;
    }
    return g_quantum_state.error_rate_accumulator / g_quantum_state.error_samples;
}

double perf_get_average_fidelity(void) {
    if (!g_quantum_state.initialized || g_quantum_state.fidelity_samples == 0) {
        return 1.0;
    }
    return g_quantum_state.fidelity_accumulator / g_quantum_state.fidelity_samples;
}

double perf_get_network_bandwidth_mbps(void) {
    double send_rate, recv_rate;
    get_network_bandwidth(&send_rate, &recv_rate);
    // Convert bytes/sec to Mbps
    return (send_rate + recv_rate) * 8.0 / (1024.0 * 1024.0);
}

// Renamed to avoid conflict with core/performance_monitor.c (this takes a monitor pointer)
void cleanup_hybrid_performance_monitor(PerformanceMonitor* monitor) {
    if (!monitor) return;

    if (monitor->log_file) {
        fclose(monitor->log_file);
    }

    free(monitor->history);
    free(monitor);
}
