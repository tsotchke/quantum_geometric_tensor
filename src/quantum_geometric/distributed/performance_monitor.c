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
#include <TargetConditionals.h>
#include <AvailabilityMacros.h>
#include <IOKit/IOKitLib.h>
#include <CoreFoundation/CoreFoundation.h>
#include <sys/socket.h>
#include <net/if.h>
#include <net/if_dl.h>
#include <ifaddrs.h>
#include <sys/sysctl.h>
#endif

#ifdef __linux__
#include <sys/sysinfo.h>
#include <sys/socket.h>
#include <net/if.h>
#include <ifaddrs.h>
#endif

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

// Initialize distributed performance monitor
PerformanceMonitor* init_distributed_performance_monitor(const MonitorConfig* config) {
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

// ============================================================================
// Real GPU Monitoring via IOKit (macOS) / sysfs (Linux)
// ============================================================================

#ifdef __APPLE__
static double get_gpu_usage_iokit(void) {
    io_iterator_t iterator;
    CFMutableDictionaryRef matching = IOServiceMatching("IOAccelerator");
    if (!matching) return 0.0;

    // Handle macOS 12+ deprecation of kIOMasterPortDefault
#if defined(__MAC_12_0) && __MAC_OS_X_VERSION_MIN_REQUIRED >= __MAC_12_0
    kern_return_t kr = IOServiceGetMatchingServices(kIOMainPortDefault, matching, &iterator);
#else
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wdeprecated-declarations"
    kern_return_t kr = IOServiceGetMatchingServices(kIOMasterPortDefault, matching, &iterator);
    #pragma clang diagnostic pop
#endif

    if (kr != KERN_SUCCESS) return 0.0;

    io_object_t accelerator;
    double max_utilization = 0.0;

    while ((accelerator = IOIteratorNext(iterator)) != 0) {
        CFMutableDictionaryRef props = NULL;
        if (IORegistryEntryCreateCFProperties(accelerator, &props, kCFAllocatorDefault, 0) == KERN_SUCCESS && props) {
            // Get PerformanceStatistics dictionary
            CFDictionaryRef perf_stats = CFDictionaryGetValue(props, CFSTR("PerformanceStatistics"));
            if (perf_stats && CFGetTypeID(perf_stats) == CFDictionaryGetTypeID()) {
                // Try different keys used by different GPU vendors
                const CFStringRef utilization_keys[] = {
                    CFSTR("Device Utilization %"),      // Apple Silicon
                    CFSTR("GPU Activity(%)"),           // AMD
                    CFSTR("GPU Core Utilization"),      // Intel
                    CFSTR("hardwareWaitTime"),          // Alternative metric
                };

                for (size_t i = 0; i < sizeof(utilization_keys)/sizeof(utilization_keys[0]); i++) {
                    CFNumberRef util_val = CFDictionaryGetValue(perf_stats, utilization_keys[i]);
                    if (util_val && CFGetTypeID(util_val) == CFNumberGetTypeID()) {
                        double utilization = 0.0;
                        if (CFNumberGetValue(util_val, kCFNumberDoubleType, &utilization)) {
                            // Normalize to 0-1 range
                            if (utilization > 1.0) {
                                utilization /= 100.0;
                            }
                            if (utilization > max_utilization) {
                                max_utilization = utilization;
                            }
                            break;
                        }
                    }
                }
            }
            CFRelease(props);
        }
        IOObjectRelease(accelerator);
    }
    IOObjectRelease(iterator);

    return max_utilization;
}
#endif

#ifdef __linux__
static double get_gpu_usage_linux(void) {
    // NVIDIA GPU via nvidia-smi
    FILE* fp = popen("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null", "r");
    if (fp) {
        double util = 0.0;
        if (fscanf(fp, "%lf", &util) == 1) {
            pclose(fp);
            return util / 100.0;
        }
        pclose(fp);
    }

    // AMD GPU via sysfs
    FILE* amd = fopen("/sys/class/drm/card0/device/gpu_busy_percent", "r");
    if (amd) {
        int percent = 0;
        if (fscanf(amd, "%d", &percent) == 1) {
            fclose(amd);
            return (double)percent / 100.0;
        }
        fclose(amd);
    }

    return 0.0;
}
#endif

static double get_gpu_usage(void) {
#ifdef __APPLE__
    return get_gpu_usage_iokit();
#elif defined(__linux__)
    return get_gpu_usage_linux();
#else
    return 0.0;
#endif
}

// ============================================================================
// Network Bandwidth Monitoring
// ============================================================================

typedef struct {
    uint64_t bytes_in;
    uint64_t bytes_out;
    struct timespec timestamp;
} NetworkSnapshot;

static NetworkSnapshot last_network_snapshot = {0, 0, {0, 0}};

static double get_network_bandwidth(void) {
    NetworkSnapshot current = {0, 0, {0, 0}};
    clock_gettime(CLOCK_MONOTONIC, &current.timestamp);

#if defined(__APPLE__) || defined(__linux__)
    struct ifaddrs* ifap = NULL;
    if (getifaddrs(&ifap) == 0) {
        for (struct ifaddrs* ifa = ifap; ifa != NULL; ifa = ifa->ifa_next) {
            if (ifa->ifa_data == NULL) continue;
            if ((ifa->ifa_flags & IFF_LOOPBACK) != 0) continue;
            if ((ifa->ifa_flags & IFF_UP) == 0) continue;

#ifdef __APPLE__
            struct if_data* ifd = (struct if_data*)ifa->ifa_data;
            current.bytes_in += ifd->ifi_ibytes;
            current.bytes_out += ifd->ifi_obytes;
#endif
        }
        freeifaddrs(ifap);
    }
#endif

#ifdef __linux__
    // Linux: read from /proc/net/dev
    FILE* fp = fopen("/proc/net/dev", "r");
    if (fp) {
        char line[512];
        // Skip header lines
        if (fgets(line, sizeof(line), fp) && fgets(line, sizeof(line), fp)) {
            while (fgets(line, sizeof(line), fp)) {
                char iface[32];
                unsigned long long rx_bytes, tx_bytes;
                unsigned long long dummy;
                if (sscanf(line, " %31[^:]: %llu %llu %llu %llu %llu %llu %llu %llu %llu",
                          iface, &rx_bytes, &dummy, &dummy, &dummy, &dummy, &dummy, &dummy, &dummy, &tx_bytes) >= 10) {
                    // Skip loopback
                    if (strncmp(iface, "lo", 2) != 0) {
                        current.bytes_in += rx_bytes;
                        current.bytes_out += tx_bytes;
                    }
                }
            }
        }
        fclose(fp);
    }
#endif

    double bandwidth_mbps = 0.0;

    if (last_network_snapshot.timestamp.tv_sec != 0) {
        double elapsed = (double)(current.timestamp.tv_sec - last_network_snapshot.timestamp.tv_sec) +
                        (double)(current.timestamp.tv_nsec - last_network_snapshot.timestamp.tv_nsec) / 1e9;

        if (elapsed > 0.01) {
            uint64_t bytes_delta = (current.bytes_in - last_network_snapshot.bytes_in) +
                                   (current.bytes_out - last_network_snapshot.bytes_out);
            bandwidth_mbps = ((double)bytes_delta * 8.0) / (elapsed * 1e6);
        }
    }

    last_network_snapshot = current;
    return bandwidth_mbps;
}

// ============================================================================
// Disk I/O Monitoring
// ============================================================================

typedef struct {
    uint64_t read_bytes;
    uint64_t write_bytes;
    struct timespec timestamp;
} DiskSnapshot;

static DiskSnapshot last_disk_snapshot = {0, 0, {0, 0}};

static double get_disk_io(void) {
    DiskSnapshot current = {0, 0, {0, 0}};
    clock_gettime(CLOCK_MONOTONIC, &current.timestamp);

#ifdef __linux__
    // Read from /proc/diskstats
    FILE* fp = fopen("/proc/diskstats", "r");
    if (fp) {
        char line[512];
        while (fgets(line, sizeof(line), fp)) {
            unsigned int major, minor;
            char device[64];
            unsigned long long rd_ios, rd_merges, rd_sectors, rd_ticks;
            unsigned long long wr_ios, wr_merges, wr_sectors, wr_ticks;

            if (sscanf(line, "%u %u %63s %llu %llu %llu %llu %llu %llu %llu %llu",
                      &major, &minor, device,
                      &rd_ios, &rd_merges, &rd_sectors, &rd_ticks,
                      &wr_ios, &wr_merges, &wr_sectors, &wr_ticks) >= 11) {
                // Only count whole disk devices (e.g., sda, nvme0n1)
                if ((strncmp(device, "sd", 2) == 0 && strlen(device) == 3) ||
                    strncmp(device, "nvme", 4) == 0) {
                    current.read_bytes += rd_sectors * 512;
                    current.write_bytes += wr_sectors * 512;
                }
            }
        }
        fclose(fp);
    }
#endif

#ifdef __APPLE__
    // macOS: use IOKit for disk statistics
    io_iterator_t iterator;
    CFMutableDictionaryRef matching = IOServiceMatching("IOBlockStorageDriver");
    if (matching) {
#if defined(__MAC_12_0) && __MAC_OS_X_VERSION_MIN_REQUIRED >= __MAC_12_0
        kern_return_t kr = IOServiceGetMatchingServices(kIOMainPortDefault, matching, &iterator);
#else
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wdeprecated-declarations"
        kern_return_t kr = IOServiceGetMatchingServices(kIOMasterPortDefault, matching, &iterator);
        #pragma clang diagnostic pop
#endif
        if (kr == KERN_SUCCESS) {
            io_object_t disk;
            while ((disk = IOIteratorNext(iterator)) != 0) {
                CFMutableDictionaryRef props = NULL;
                if (IORegistryEntryCreateCFProperties(disk, &props, kCFAllocatorDefault, 0) == KERN_SUCCESS && props) {
                    CFDictionaryRef stats = CFDictionaryGetValue(props, CFSTR("Statistics"));
                    if (stats && CFGetTypeID(stats) == CFDictionaryGetTypeID()) {
                        CFNumberRef read_bytes_ref = CFDictionaryGetValue(stats, CFSTR("Bytes (Read)"));
                        CFNumberRef write_bytes_ref = CFDictionaryGetValue(stats, CFSTR("Bytes (Write)"));

                        if (read_bytes_ref) {
                            long long val = 0;
                            CFNumberGetValue(read_bytes_ref, kCFNumberLongLongType, &val);
                            current.read_bytes += (uint64_t)val;
                        }
                        if (write_bytes_ref) {
                            long long val = 0;
                            CFNumberGetValue(write_bytes_ref, kCFNumberLongLongType, &val);
                            current.write_bytes += (uint64_t)val;
                        }
                    }
                    CFRelease(props);
                }
                IOObjectRelease(disk);
            }
            IOObjectRelease(iterator);
        }
    }
#endif

    double io_mbps = 0.0;

    if (last_disk_snapshot.timestamp.tv_sec != 0) {
        double elapsed = (double)(current.timestamp.tv_sec - last_disk_snapshot.timestamp.tv_sec) +
                        (double)(current.timestamp.tv_nsec - last_disk_snapshot.timestamp.tv_nsec) / 1e9;

        if (elapsed > 0.01) {
            uint64_t bytes_delta = (current.read_bytes - last_disk_snapshot.read_bytes) +
                                   (current.write_bytes - last_disk_snapshot.write_bytes);
            io_mbps = (double)bytes_delta / (elapsed * 1e6);
        }
    }

    last_disk_snapshot = current;
    return io_mbps;
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

    // GPU usage (real implementation via IOKit/sysfs)
    metrics.gpu_usage = get_gpu_usage();

    // Quantum usage (tracked separately via quantum hardware state)
    // This is populated by quantum execution callbacks
    metrics.quantum_usage = 0.0;

    // Network bandwidth (real implementation via getifaddrs/proc)
    metrics.network_bandwidth = get_network_bandwidth();

    // Disk I/O (real implementation via IOKit/proc)
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

        // Simulated metrics (would need actual device queries)
        metrics->utilization = 0.0;
        metrics->temperature = 40.0;
        metrics->power_usage = 0.0;
        metrics->memory_used = 0.0;
        metrics->error_rate = 0.0;

        // Update device state
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

// Clean up distributed performance monitor
void cleanup_distributed_performance_monitor(PerformanceMonitor* monitor) {
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
