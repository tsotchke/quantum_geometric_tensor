/**
 * @file performance_monitor.c
 * @brief Distributed system performance monitoring implementation
 *
 * Implements real-time monitoring of CPU, GPU, memory, network,
 * and quantum device metrics with threading support and alerting.
 */

#include "quantum_geometric/distributed/performance_monitor.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/resource.h>

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <sys/sysctl.h>
#include <sys/socket.h>
#include <net/if.h>
#include <net/route.h>
#include <net/if_var.h>
#include <IOKit/IOKitLib.h>
#include <IOKit/storage/IOBlockStorageDriver.h>
#include <CoreFoundation/CoreFoundation.h>
#else
#include <sys/statvfs.h>
#endif

// GPU monitoring state
static struct {
    double last_sample_time;
    uint64_t last_bytes_rx;
    uint64_t last_bytes_tx;
    uint64_t last_io_reads;
    uint64_t last_io_writes;
} g_perf_state = {0};

// Forward declarations for metric collection
static double get_gpu_usage(void);
static double get_network_bandwidth(void);
static double get_disk_io(void);
static double get_quantum_usage(int device_id);
static void get_device_metrics_impl(int device_id, DeviceMetrics* metrics);

// Internal constants
#define UPDATE_INTERVAL_US (PERF_MON_SAMPLING_INTERVAL_MS * 1000)

// Performance monitor - internal structure
struct PerformanceMonitorImpl {
    // System monitoring
    SystemMetrics* metrics_history;
    size_t history_capacity;
    size_t history_count;
    size_t history_index;
    SystemMetrics current_metrics;

    // Device monitoring
    DeviceMetrics* device_metrics;
    int num_devices;

    // Bottleneck detection
    bool has_bottleneck;
    double bottleneck_score;

    // Alert system
    AlertConfig alert_config;
    AlertCallback alert_callback;
    void* callback_user_data;
    size_t alert_count;

    // Threading
    bool is_running;
    bool enable_threading;
    pthread_t monitor_thread;
    pthread_mutex_t metrics_mutex;

    // Configuration
    MonitorConfig config;
};

// Forward declarations
static void collect_system_metrics(PerformanceMonitor* monitor);
static void collect_device_metrics(PerformanceMonitor* monitor);
static void detect_bottlenecks(PerformanceMonitor* monitor);
static void check_alerts(PerformanceMonitor* monitor);
static void store_metrics(PerformanceMonitor* monitor);
static void* monitoring_thread_func(void* arg);
static double get_cpu_usage(void);
static double get_memory_usage(void);
static DeviceState analyze_device_state(const DeviceMetrics* metrics);
static void trigger_alert(PerformanceMonitor* monitor, AlertType type, int device_id, double value, double threshold, const char* message);

// Initialize performance monitor
PerformanceMonitor* init_performance_monitor(const MonitorConfig* config) {
    PerformanceMonitor* monitor = calloc(1, sizeof(PerformanceMonitor));
    if (!monitor) return NULL;

    // Store configuration
    if (config) {
        monitor->config = *config;
        monitor->alert_config = config->alert_config;
        monitor->alert_callback = config->alert_callback;
        monitor->callback_user_data = config->callback_user_data;
        monitor->enable_threading = config->enable_threading;
        monitor->num_devices = config->num_devices;
    } else {
        // Default configuration
        monitor->config.num_devices = 1;
        monitor->config.history_length = PERF_MON_HISTORY_LENGTH;
        monitor->config.sampling_interval_ms = PERF_MON_SAMPLING_INTERVAL_MS;
        monitor->config.alert_config.cpu_threshold = PERF_MON_ALERT_THRESHOLD;
        monitor->config.alert_config.memory_threshold = PERF_MON_ALERT_THRESHOLD;
        monitor->config.alert_config.gpu_threshold = PERF_MON_ALERT_THRESHOLD;
        monitor->config.alert_config.temp_threshold = 85.0;
        monitor->config.alert_config.error_rate_threshold = 0.1;
        monitor->config.alert_config.enable_alerts = true;
        monitor->config.enable_threading = false;
        monitor->alert_config = monitor->config.alert_config;
        monitor->num_devices = 1;
        monitor->enable_threading = false;
    }

    // Initialize metrics history
    monitor->history_capacity = monitor->config.history_length;
    if (monitor->history_capacity == 0) {
        monitor->history_capacity = PERF_MON_HISTORY_LENGTH;
    }
    monitor->metrics_history = calloc(monitor->history_capacity, sizeof(SystemMetrics));
    if (!monitor->metrics_history) {
        free(monitor);
        return NULL;
    }
    monitor->history_count = 0;
    monitor->history_index = 0;

    // Initialize device metrics
    if (monitor->num_devices > 0) {
        monitor->device_metrics = calloc((size_t)monitor->num_devices, sizeof(DeviceMetrics));
        if (!monitor->device_metrics) {
            free(monitor->metrics_history);
            free(monitor);
            return NULL;
        }

        for (int i = 0; i < monitor->num_devices; i++) {
            monitor->device_metrics[i].device_id = i;
            monitor->device_metrics[i].state = DEVICE_STATE_IDLE;
        }
    }

    // Initialize state
    monitor->has_bottleneck = false;
    monitor->bottleneck_score = 0.0;
    monitor->is_running = false;
    monitor->alert_count = 0;

    // Initialize mutex
    pthread_mutex_init(&monitor->metrics_mutex, NULL);

    return monitor;
}

// Get CPU usage (platform-specific)
static double get_cpu_usage(void) {
#ifdef __APPLE__
    host_cpu_load_info_data_t cpu_info;
    mach_msg_type_number_t count = HOST_CPU_LOAD_INFO_COUNT;

    if (host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO,
                        (host_info_t)&cpu_info, &count) == KERN_SUCCESS) {
        natural_t total = 0;
        for (int i = 0; i < CPU_STATE_MAX; i++) {
            total += cpu_info.cpu_ticks[i];
        }
        if (total > 0) {
            natural_t idle = cpu_info.cpu_ticks[CPU_STATE_IDLE];
            return 1.0 - ((double)idle / (double)total);
        }
    }
    return 0.0;
#else
    // Linux: read /proc/stat
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        double user_time = (double)usage.ru_utime.tv_sec + (double)usage.ru_utime.tv_usec / 1e6;
        double sys_time = (double)usage.ru_stime.tv_sec + (double)usage.ru_stime.tv_usec / 1e6;
        // Rough estimate - actual CPU usage requires comparing with previous sample
        return (user_time + sys_time) / 100.0;  // Normalized
    }
    return 0.0;
#endif
}

// Get memory usage (platform-specific)
static double get_memory_usage(void) {
#ifdef __APPLE__
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        // maxrss is in bytes on macOS
        double used_mb = (double)usage.ru_maxrss / (1024.0 * 1024.0);
        // Assume 16GB total for normalization
        return used_mb / (16.0 * 1024.0);
    }
    return 0.0;
#else
    // Linux: maxrss is in KB
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        double used_mb = (double)usage.ru_maxrss / 1024.0;
        return used_mb / (16.0 * 1024.0);
    }
    return 0.0;
#endif
}

// Get GPU usage - platform specific implementation
static double get_gpu_usage(void) {
#ifdef __APPLE__
    // macOS: Query GPU via IOKit
    io_iterator_t iterator;
    io_object_t device;
    double gpu_usage = 0.0;

    CFMutableDictionaryRef match = IOServiceMatching("IOAccelerator");
    if (IOServiceGetMatchingServices(kIOMainPortDefault, match, &iterator) == KERN_SUCCESS) {
        while ((device = IOIteratorNext(iterator))) {
            CFMutableDictionaryRef properties = NULL;
            if (IORegistryEntryCreateCFProperties(device, &properties,
                                                  kCFAllocatorDefault, 0) == KERN_SUCCESS) {
                // Look for performance statistics
                CFTypeRef perf_stats = CFDictionaryGetValue(properties,
                    CFSTR("PerformanceStatistics"));
                if (perf_stats && CFGetTypeID(perf_stats) == CFDictionaryGetTypeID()) {
                    CFDictionaryRef stats = (CFDictionaryRef)perf_stats;

                    // Get device utilization
                    CFTypeRef util = CFDictionaryGetValue(stats,
                        CFSTR("Device Utilization %"));
                    if (util && CFGetTypeID(util) == CFNumberGetTypeID()) {
                        int64_t value;
                        if (CFNumberGetValue((CFNumberRef)util, kCFNumberSInt64Type, &value)) {
                            gpu_usage = (double)value / 100.0;
                        }
                    }

                    // Alternative: GPU Activity
                    if (gpu_usage == 0.0) {
                        CFTypeRef activity = CFDictionaryGetValue(stats,
                            CFSTR("GPU Activity(%)"));
                        if (activity && CFGetTypeID(activity) == CFNumberGetTypeID()) {
                            int64_t value;
                            if (CFNumberGetValue((CFNumberRef)activity, kCFNumberSInt64Type, &value)) {
                                gpu_usage = (double)value / 100.0;
                            }
                        }
                    }
                }
                CFRelease(properties);
            }
            IOObjectRelease(device);
            if (gpu_usage > 0.0) break;  // Found valid GPU
        }
        IOObjectRelease(iterator);
    }
    return gpu_usage;
#else
    // Linux: Query nvidia-smi or read from sysfs
    FILE* fp = popen("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null", "r");
    if (fp) {
        char buffer[64];
        if (fgets(buffer, sizeof(buffer), fp)) {
            double usage = atof(buffer) / 100.0;
            pclose(fp);
            return usage;
        }
        pclose(fp);
    }

    // Fallback: Try AMD ROCm
    fp = popen("rocm-smi --showuse 2>/dev/null | grep GPU | awk '{print $3}'", "r");
    if (fp) {
        char buffer[64];
        if (fgets(buffer, sizeof(buffer), fp)) {
            double usage = atof(buffer) / 100.0;
            pclose(fp);
            return usage;
        }
        pclose(fp);
    }

    return 0.0;
#endif
}

// Get network bandwidth usage
static double get_network_bandwidth(void) {
#ifdef __APPLE__
    // macOS: Use sysctl for network interface stats
    int mib[6] = {CTL_NET, PF_ROUTE, 0, 0, NET_RT_IFLIST2, 0};
    size_t len;

    if (sysctl(mib, 6, NULL, &len, NULL, 0) < 0) return 0.0;

    char* buf = malloc(len);
    if (!buf) return 0.0;

    if (sysctl(mib, 6, buf, &len, NULL, 0) < 0) {
        free(buf);
        return 0.0;
    }

    uint64_t total_bytes_rx = 0, total_bytes_tx = 0;
    char* end = buf + len;
    char* ptr = buf;

    while (ptr < end) {
        struct if_msghdr* ifm = (struct if_msghdr*)ptr;
        if (ifm->ifm_type == RTM_IFINFO2) {
            struct if_msghdr2* ifm2 = (struct if_msghdr2*)ifm;
            total_bytes_rx += ifm2->ifm_data.ifi_ibytes;
            total_bytes_tx += ifm2->ifm_data.ifi_obytes;
        }
        ptr += ifm->ifm_msglen;
    }
    free(buf);

    // Calculate bandwidth (delta from last sample)
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    double current_time = now.tv_sec + now.tv_nsec / 1e9;

    double bandwidth = 0.0;
    if (g_perf_state.last_sample_time > 0.0) {
        double dt = current_time - g_perf_state.last_sample_time;
        if (dt > 0.0) {
            uint64_t delta_rx = total_bytes_rx - g_perf_state.last_bytes_rx;
            uint64_t delta_tx = total_bytes_tx - g_perf_state.last_bytes_tx;
            // Bandwidth in MB/s
            bandwidth = (double)(delta_rx + delta_tx) / (dt * 1024.0 * 1024.0);
        }
    }

    g_perf_state.last_sample_time = current_time;
    g_perf_state.last_bytes_rx = total_bytes_rx;
    g_perf_state.last_bytes_tx = total_bytes_tx;

    // Normalize to fraction of assumed 1 Gbps capacity
    return bandwidth / 125.0;  // 1 Gbps = 125 MB/s
#else
    // Linux: Read /proc/net/dev
    FILE* fp = fopen("/proc/net/dev", "r");
    if (!fp) return 0.0;

    char line[512];
    uint64_t total_bytes_rx = 0, total_bytes_tx = 0;

    // Skip header lines
    fgets(line, sizeof(line), fp);
    fgets(line, sizeof(line), fp);

    while (fgets(line, sizeof(line), fp)) {
        char iface[32];
        uint64_t rx, tx;
        // Parse: iface: rx_bytes ... tx_bytes
        if (sscanf(line, "%31[^:]: %lu %*u %*u %*u %*u %*u %*u %*u %lu",
                   iface, &rx, &tx) >= 3) {
            // Skip loopback
            if (strncmp(iface, "lo", 2) != 0) {
                total_bytes_rx += rx;
                total_bytes_tx += tx;
            }
        }
    }
    fclose(fp);

    // Calculate bandwidth delta
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    double current_time = now.tv_sec + now.tv_nsec / 1e9;

    double bandwidth = 0.0;
    if (g_perf_state.last_sample_time > 0.0) {
        double dt = current_time - g_perf_state.last_sample_time;
        if (dt > 0.0) {
            uint64_t delta_rx = total_bytes_rx - g_perf_state.last_bytes_rx;
            uint64_t delta_tx = total_bytes_tx - g_perf_state.last_bytes_tx;
            bandwidth = (double)(delta_rx + delta_tx) / (dt * 1024.0 * 1024.0);
        }
    }

    g_perf_state.last_sample_time = current_time;
    g_perf_state.last_bytes_rx = total_bytes_rx;
    g_perf_state.last_bytes_tx = total_bytes_tx;

    return bandwidth / 125.0;
#endif
}

// Get disk I/O usage
static double get_disk_io(void) {
#ifdef __APPLE__
    // macOS: Use iostat-like metrics via sysctl
    io_iterator_t iterator;
    io_object_t disk;
    uint64_t total_reads = 0, total_writes = 0;

    CFMutableDictionaryRef match = IOServiceMatching("IOBlockStorageDriver");
    if (IOServiceGetMatchingServices(kIOMainPortDefault, match, &iterator) == KERN_SUCCESS) {
        while ((disk = IOIteratorNext(iterator))) {
            CFMutableDictionaryRef properties = NULL;
            if (IORegistryEntryCreateCFProperties(disk, &properties,
                                                  kCFAllocatorDefault, 0) == KERN_SUCCESS) {
                CFTypeRef stats = CFDictionaryGetValue(properties, CFSTR("Statistics"));
                if (stats && CFGetTypeID(stats) == CFDictionaryGetTypeID()) {
                    CFDictionaryRef statsDict = (CFDictionaryRef)stats;

                    CFTypeRef reads = CFDictionaryGetValue(statsDict,
                        CFSTR("Bytes (Read)"));
                    CFTypeRef writes = CFDictionaryGetValue(statsDict,
                        CFSTR("Bytes (Write)"));

                    if (reads && CFGetTypeID(reads) == CFNumberGetTypeID()) {
                        int64_t value;
                        CFNumberGetValue((CFNumberRef)reads, kCFNumberSInt64Type, &value);
                        total_reads += value;
                    }
                    if (writes && CFGetTypeID(writes) == CFNumberGetTypeID()) {
                        int64_t value;
                        CFNumberGetValue((CFNumberRef)writes, kCFNumberSInt64Type, &value);
                        total_writes += value;
                    }
                }
                CFRelease(properties);
            }
            IOObjectRelease(disk);
        }
        IOObjectRelease(iterator);
    }

    // Calculate I/O rate
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    double current_time = now.tv_sec + now.tv_nsec / 1e9;

    double io_rate = 0.0;
    if (g_perf_state.last_io_reads > 0 || g_perf_state.last_io_writes > 0) {
        double dt = current_time - g_perf_state.last_sample_time;
        if (dt > 0.0) {
            uint64_t delta_reads = total_reads - g_perf_state.last_io_reads;
            uint64_t delta_writes = total_writes - g_perf_state.last_io_writes;
            // I/O in MB/s
            io_rate = (double)(delta_reads + delta_writes) / (dt * 1024.0 * 1024.0);
        }
    }

    g_perf_state.last_io_reads = total_reads;
    g_perf_state.last_io_writes = total_writes;

    // Normalize to fraction of assumed 500 MB/s capacity (SSD)
    return io_rate / 500.0;
#else
    // Linux: Read /proc/diskstats
    FILE* fp = fopen("/proc/diskstats", "r");
    if (!fp) return 0.0;

    char line[256];
    uint64_t total_reads = 0, total_writes = 0;

    while (fgets(line, sizeof(line), fp)) {
        unsigned int major, minor;
        char device[32];
        uint64_t rd_sectors, wr_sectors;

        // Parse diskstats format
        if (sscanf(line, "%u %u %31s %*u %*u %lu %*u %*u %*u %lu",
                   &major, &minor, device, &rd_sectors, &wr_sectors) >= 5) {
            // Only count main disks (sdX, nvmeXnY, not partitions)
            if ((strncmp(device, "sd", 2) == 0 && strlen(device) == 3) ||
                strncmp(device, "nvme", 4) == 0) {
                total_reads += rd_sectors * 512;   // 512 bytes per sector
                total_writes += wr_sectors * 512;
            }
        }
    }
    fclose(fp);

    // Calculate I/O rate
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    double current_time = now.tv_sec + now.tv_nsec / 1e9;

    double io_rate = 0.0;
    if (g_perf_state.last_io_reads > 0 || g_perf_state.last_io_writes > 0) {
        double dt = current_time - g_perf_state.last_sample_time;
        if (dt > 0.0) {
            uint64_t delta_reads = total_reads - g_perf_state.last_io_reads;
            uint64_t delta_writes = total_writes - g_perf_state.last_io_writes;
            io_rate = (double)(delta_reads + delta_writes) / (dt * 1024.0 * 1024.0);
        }
    }

    g_perf_state.last_io_reads = total_reads;
    g_perf_state.last_io_writes = total_writes;

    return io_rate / 500.0;
#endif
}

// Get quantum device usage (simulated or from actual hardware interface)
static double get_quantum_usage(int device_id) {
    // Check for IBM Quantum or similar backend connection
    // In production this would query actual quantum hardware/simulator load
    (void)device_id;

    // Try reading from quantum device status file if available
    char status_path[256];
    snprintf(status_path, sizeof(status_path),
             "/tmp/quantum_device_%d_status", device_id);

    FILE* fp = fopen(status_path, "r");
    if (fp) {
        double usage;
        if (fscanf(fp, "%lf", &usage) == 1) {
            fclose(fp);
            return usage;
        }
        fclose(fp);
    }

    // Check environment for quantum simulator load
    char* sim_load = getenv("QUANTUM_SIMULATOR_LOAD");
    if (sim_load) {
        return atof(sim_load);
    }

    // No quantum device status available via file or environment
    // Return 0.0 indicating idle/unavailable quantum backend
    return 0.0;
}

// Get detailed device metrics implementation
static void get_device_metrics_impl(int device_id, DeviceMetrics* metrics) {
    if (!metrics) return;

    metrics->device_id = device_id;

#ifdef __APPLE__
    // Query GPU via IOKit for temperature, power, memory
    io_iterator_t iterator;
    io_object_t device;

    CFMutableDictionaryRef match = IOServiceMatching("IOAccelerator");
    if (IOServiceGetMatchingServices(kIOMainPortDefault, match, &iterator) == KERN_SUCCESS) {
        int current_device = 0;
        while ((device = IOIteratorNext(iterator))) {
            if (current_device == device_id) {
                CFMutableDictionaryRef properties = NULL;
                if (IORegistryEntryCreateCFProperties(device, &properties,
                                                      kCFAllocatorDefault, 0) == KERN_SUCCESS) {
                    CFTypeRef perf_stats = CFDictionaryGetValue(properties,
                        CFSTR("PerformanceStatistics"));
                    if (perf_stats && CFGetTypeID(perf_stats) == CFDictionaryGetTypeID()) {
                        CFDictionaryRef stats = (CFDictionaryRef)perf_stats;

                        // Utilization
                        CFTypeRef util = CFDictionaryGetValue(stats,
                            CFSTR("Device Utilization %"));
                        if (util && CFGetTypeID(util) == CFNumberGetTypeID()) {
                            int64_t value;
                            if (CFNumberGetValue((CFNumberRef)util, kCFNumberSInt64Type, &value)) {
                                metrics->utilization = (double)value / 100.0;
                            }
                        }

                        // VRAM used
                        CFTypeRef vram = CFDictionaryGetValue(stats,
                            CFSTR("VRAM Used Bytes"));
                        if (vram && CFGetTypeID(vram) == CFNumberGetTypeID()) {
                            int64_t value;
                            if (CFNumberGetValue((CFNumberRef)vram, kCFNumberSInt64Type, &value)) {
                                metrics->memory_used = (double)value / (1024.0 * 1024.0);  // MB
                            }
                        }
                    }
                    CFRelease(properties);
                }
                IOObjectRelease(device);
                break;
            }
            IOObjectRelease(device);
            current_device++;
        }
        IOObjectRelease(iterator);
    }

    // Temperature estimation based on GPU utilization
    // Formula: T = T_idle + (T_max - T_idle) * utilization
    // Typical GPU: ~45°C idle, ~75°C at full load
    metrics->temperature = 45.0 + metrics->utilization * 30.0;
    // Power estimation: P ≈ P_max * utilization (assuming ~100W TDP)
    metrics->power_usage = metrics->utilization * 100.0;
#else
    // Linux: Query nvidia-smi
    char cmd[256];
    snprintf(cmd, sizeof(cmd),
             "nvidia-smi -i %d --query-gpu=utilization.gpu,temperature.gpu,power.draw,memory.used "
             "--format=csv,noheader,nounits 2>/dev/null", device_id);

    FILE* fp = popen(cmd, "r");
    if (fp) {
        float util, temp, power, mem;
        if (fscanf(fp, "%f, %f, %f, %f", &util, &temp, &power, &mem) == 4) {
            metrics->utilization = util / 100.0;
            metrics->temperature = temp;
            metrics->power_usage = power;
            metrics->memory_used = mem;
        }
        pclose(fp);
    }
#endif

    // Query quantum error rate if this is a quantum device
    metrics->error_rate = get_quantum_usage(device_id) > 0 ? 0.001 : 0.0;
}

// Analyze device state based on metrics
static DeviceState analyze_device_state(const DeviceMetrics* metrics) {
    if (!metrics) return DEVICE_STATE_UNKNOWN;

    if (metrics->error_rate > 0.5) {
        return DEVICE_STATE_ERROR;
    }

    if (metrics->utilization < 0.1) {
        return DEVICE_STATE_IDLE;
    } else if (metrics->utilization < 0.5) {
        return DEVICE_STATE_ACTIVE;
    } else if (metrics->utilization < 0.85) {
        return DEVICE_STATE_BUSY;
    } else {
        return DEVICE_STATE_OVERLOADED;
    }
}

// Collect system metrics
static void collect_system_metrics(PerformanceMonitor* monitor) {
    if (!monitor) return;

    SystemMetrics metrics = {0};

    // Get current time
    clock_gettime(CLOCK_REALTIME, &metrics.timestamp);

    // CPU usage
    metrics.cpu_usage = get_cpu_usage();

    // Memory usage
    metrics.memory_usage = get_memory_usage();

    // GPU usage
    metrics.gpu_usage = get_gpu_usage();

    // Quantum usage (aggregate across devices)
    metrics.quantum_usage = 0.0;
    for (int i = 0; i < monitor->num_devices; i++) {
        metrics.quantum_usage += get_quantum_usage(i);
    }
    if (monitor->num_devices > 0) {
        metrics.quantum_usage /= monitor->num_devices;
    }

    // Network bandwidth
    metrics.network_bandwidth = get_network_bandwidth();

    // Disk I/O
    metrics.disk_io = get_disk_io();

    // Update current metrics
    pthread_mutex_lock(&monitor->metrics_mutex);
    monitor->current_metrics = metrics;
    pthread_mutex_unlock(&monitor->metrics_mutex);
}

// Collect device metrics
static void collect_device_metrics(PerformanceMonitor* monitor) {
    if (!monitor || !monitor->device_metrics) return;

    for (int i = 0; i < monitor->num_devices; i++) {
        DeviceMetrics* metrics = &monitor->device_metrics[i];

        // Get real device metrics
        get_device_metrics_impl(i, metrics);

        // Update device state based on collected metrics
        metrics->state = analyze_device_state(metrics);
    }
}

// Detect performance bottlenecks
static void detect_bottlenecks(PerformanceMonitor* monitor) {
    if (!monitor) return;

    double score = 0.0;
    int bottleneck_count = 0;

    // Check CPU bottleneck
    if (monitor->current_metrics.cpu_usage > 0.9) {
        score += monitor->current_metrics.cpu_usage;
        bottleneck_count++;
    }

    // Check memory bottleneck
    if (monitor->current_metrics.memory_usage > 0.9) {
        score += monitor->current_metrics.memory_usage;
        bottleneck_count++;
    }

    // Check GPU bottleneck
    if (monitor->current_metrics.gpu_usage > 0.9) {
        score += monitor->current_metrics.gpu_usage;
        bottleneck_count++;
    }

    // Check device bottlenecks
    for (int i = 0; i < monitor->num_devices; i++) {
        if (monitor->device_metrics[i].state == DEVICE_STATE_OVERLOADED) {
            score += monitor->device_metrics[i].utilization;
            bottleneck_count++;
        }
    }

    monitor->has_bottleneck = (bottleneck_count > 0);
    monitor->bottleneck_score = (bottleneck_count > 0) ? score / bottleneck_count : 0.0;
}

// Trigger an alert
static void trigger_alert(PerformanceMonitor* monitor, AlertType type, int device_id, double value, double threshold, const char* message) {
    if (!monitor || !monitor->alert_callback) return;
    if (!monitor->alert_config.enable_alerts) return;

    AlertInfo alert;
    alert.type = type;
    alert.device_id = device_id;
    alert.value = value;
    alert.threshold = threshold;
    clock_gettime(CLOCK_REALTIME, &alert.timestamp);
    strncpy(alert.message, message ? message : "", sizeof(alert.message) - 1);
    alert.message[sizeof(alert.message) - 1] = '\0';

    monitor->alert_count++;
    monitor->alert_callback(&alert, monitor->callback_user_data);
}

// Check for performance alerts
static void check_alerts(PerformanceMonitor* monitor) {
    if (!monitor || !monitor->alert_callback) return;

    // Check CPU
    if (monitor->current_metrics.cpu_usage > monitor->alert_config.cpu_threshold) {
        trigger_alert(monitor, ALERT_TYPE_CPU, -1,
                      monitor->current_metrics.cpu_usage,
                      monitor->alert_config.cpu_threshold,
                      "High CPU usage detected");
    }

    // Check memory
    if (monitor->current_metrics.memory_usage > monitor->alert_config.memory_threshold) {
        trigger_alert(monitor, ALERT_TYPE_MEMORY, -1,
                      monitor->current_metrics.memory_usage,
                      monitor->alert_config.memory_threshold,
                      "High memory usage detected");
    }

    // Check devices
    for (int i = 0; i < monitor->num_devices; i++) {
        DeviceMetrics* dm = &monitor->device_metrics[i];

        // Temperature check
        if (dm->temperature > monitor->alert_config.temp_threshold) {
            trigger_alert(monitor, ALERT_TYPE_TEMPERATURE, i,
                          dm->temperature,
                          monitor->alert_config.temp_threshold,
                          "Device temperature threshold exceeded");
        }

        // Error rate check
        if (dm->error_rate > monitor->alert_config.error_rate_threshold) {
            trigger_alert(monitor, ALERT_TYPE_QUANTUM_ERROR, i,
                          dm->error_rate,
                          monitor->alert_config.error_rate_threshold,
                          "Device error rate threshold exceeded");
        }
    }

    // Bottleneck alert
    if (monitor->has_bottleneck) {
        trigger_alert(monitor, ALERT_TYPE_BOTTLENECK, -1,
                      monitor->bottleneck_score, 0.9,
                      "Performance bottleneck detected");
    }
}

// Store metrics in history
static void store_metrics(PerformanceMonitor* monitor) {
    if (!monitor || !monitor->metrics_history) return;

    pthread_mutex_lock(&monitor->metrics_mutex);

    // Store in circular buffer
    monitor->metrics_history[monitor->history_index] = monitor->current_metrics;
    monitor->history_index = (monitor->history_index + 1) % monitor->history_capacity;

    if (monitor->history_count < monitor->history_capacity) {
        monitor->history_count++;
    }

    pthread_mutex_unlock(&monitor->metrics_mutex);
}

// Monitoring thread function
static void* monitoring_thread_func(void* arg) {
    PerformanceMonitor* monitor = (PerformanceMonitor*)arg;

    while (monitor->is_running) {
        // Collect metrics
        collect_system_metrics(monitor);
        collect_device_metrics(monitor);

        // Detect bottlenecks
        detect_bottlenecks(monitor);

        // Check alerts
        check_alerts(monitor);

        // Store in history
        store_metrics(monitor);

        // Wait for next interval
        usleep(UPDATE_INTERVAL_US);
    }

    return NULL;
}

// Start monitoring
void perf_monitor_start(PerformanceMonitor* monitor) {
    if (!monitor || monitor->is_running) return;

    monitor->is_running = true;

    if (monitor->enable_threading) {
        pthread_create(&monitor->monitor_thread, NULL, monitoring_thread_func, monitor);
    }
}

// Stop monitoring
void perf_monitor_stop(PerformanceMonitor* monitor) {
    if (!monitor || !monitor->is_running) return;

    monitor->is_running = false;

    if (monitor->enable_threading) {
        pthread_join(monitor->monitor_thread, NULL);
    }
}

// Manual update
void perf_monitor_update(PerformanceMonitor* monitor) {
    if (!monitor) return;

    collect_system_metrics(monitor);
    collect_device_metrics(monitor);
    detect_bottlenecks(monitor);
    check_alerts(monitor);
    store_metrics(monitor);
}

// Get current system metrics
void perf_monitor_get_system_metrics(PerformanceMonitor* monitor, SystemMetrics* metrics) {
    if (!monitor || !metrics) return;

    pthread_mutex_lock(&monitor->metrics_mutex);
    *metrics = monitor->current_metrics;
    pthread_mutex_unlock(&monitor->metrics_mutex);
}

// Get device metrics
void perf_monitor_get_device_metrics(PerformanceMonitor* monitor, int device_id, DeviceMetrics* metrics) {
    if (!monitor || !metrics) return;
    if (device_id < 0 || device_id >= monitor->num_devices) return;

    *metrics = monitor->device_metrics[device_id];
}

// Get metrics history
const SystemMetrics* perf_monitor_get_history(const PerformanceMonitor* monitor, size_t* count) {
    if (!monitor) {
        if (count) *count = 0;
        return NULL;
    }

    if (count) *count = monitor->history_count;
    return monitor->metrics_history;
}

// Check if bottleneck detected
bool perf_monitor_has_bottleneck(const PerformanceMonitor* monitor) {
    return monitor ? monitor->has_bottleneck : false;
}

// Get alert count
size_t perf_monitor_get_alert_count(const PerformanceMonitor* monitor) {
    return monitor ? monitor->alert_count : 0;
}

// Set alert callback
void perf_monitor_set_callback(PerformanceMonitor* monitor, AlertCallback callback, void* user_data) {
    if (!monitor) return;
    monitor->alert_callback = callback;
    monitor->callback_user_data = user_data;
}

// Reset statistics
void perf_monitor_reset(PerformanceMonitor* monitor) {
    if (!monitor) return;

    pthread_mutex_lock(&monitor->metrics_mutex);

    memset(&monitor->current_metrics, 0, sizeof(SystemMetrics));
    memset(monitor->metrics_history, 0, monitor->history_capacity * sizeof(SystemMetrics));
    monitor->history_count = 0;
    monitor->history_index = 0;
    monitor->alert_count = 0;
    monitor->has_bottleneck = false;
    monitor->bottleneck_score = 0.0;

    for (int i = 0; i < monitor->num_devices; i++) {
        memset(&monitor->device_metrics[i], 0, sizeof(DeviceMetrics));
        monitor->device_metrics[i].device_id = i;
        monitor->device_metrics[i].state = DEVICE_STATE_IDLE;
    }

    pthread_mutex_unlock(&monitor->metrics_mutex);
}

// Clean up performance monitor
void cleanup_performance_monitor(PerformanceMonitor* monitor) {
    if (!monitor) return;

    // Stop monitoring if running
    if (monitor->is_running) {
        perf_monitor_stop(monitor);
    }

    // Destroy mutex
    pthread_mutex_destroy(&monitor->metrics_mutex);

    // Free memory
    free(monitor->metrics_history);
    free(monitor->device_metrics);
    free(monitor);
}
