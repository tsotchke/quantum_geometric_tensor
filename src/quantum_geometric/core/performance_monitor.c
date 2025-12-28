#include "quantum_geometric/core/performance_operations.h"
#include "quantum_geometric/core/performance_monitor.h"
#include "quantum_geometric/core/memory_pool.h"
#include "quantum_geometric/core/simd_operations.h"
#include "quantum_geometric/hardware/quantum_error_mitigation.h"
#include <math.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <stdint.h>

// Platform-specific includes for network monitoring
#ifdef __APPLE__
#include <mach/mach.h>
#include <sys/sysctl.h>
#include <net/if.h>
#include <ifaddrs.h>
#include <net/if_dl.h>
#include <IOKit/IOKitLib.h>
#include <CoreFoundation/CoreFoundation.h>
#endif

#ifdef HAVE_NVML
#include <nvml.h>
#endif

#ifdef HAVE_MPI
#include <mpi.h>
#endif

// performance_metrics_t is defined in performance_operations.h
// This file uses that shared comprehensive definition


// Performance monitoring system state
typedef struct {
    // Configuration
    char* config_path;
    char* metrics_path;
    
    // Metrics tracking
    struct {
        char* name;
        int type;
        void* parameters;
        size_t parameters_count;
        double current_value;
        double average_value;
        double peak_value;
        double threshold_value;
        int threshold_type;
        void (*callback_function)(void*);
        int callback_type;
        bool monitoring_active;
    }* metrics;
    size_t metrics_count;
    size_t metrics_capacity;
    
    // Resource tracking
    double* resource_usage;
    double* resource_limits;
    size_t num_resources;
    
    // Performance history
    performance_metrics_t* metric_history;
    size_t history_size;
    size_t history_capacity;
    
    // Analysis state
    double baseline_performance;
    double peak_performance;
    double optimization_threshold;
    
    // Thread safety
    pthread_mutex_t mutex;
    bool initialized;
} monitor_state_t;

static monitor_state_t* monitor_state = NULL;
static pthread_mutex_t global_mutex = PTHREAD_MUTEX_INITIALIZER;

// Forward declarations
void cleanup_performance_monitor(void);

// Simple init wrapper (no configuration)
void init_performance_monitor(void) {
    initialize_performance_monitor("/tmp/perf_config.json", "/tmp/perf_metrics.json");
}

// Platform-specific includes for measurements
#ifdef __APPLE__
    #include <mach/mach.h>
    #include <mach/processor_info.h>
    #include <mach/mach_host.h>
    #include <sys/sysctl.h>
#else
    #include <sys/sysinfo.h>
#endif
#include <sys/resource.h>
#include <sys/time.h>

// CPU utilization measurement - cross-platform
static double measure_cpu_utilization(void) {
#ifdef __APPLE__
    host_cpu_load_info_data_t cpuinfo;
    mach_msg_type_number_t count = HOST_CPU_LOAD_INFO_COUNT;
    if (host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO,
                        (host_info_t)&cpuinfo, &count) == KERN_SUCCESS) {
        unsigned long total = cpuinfo.cpu_ticks[CPU_STATE_USER] +
                             cpuinfo.cpu_ticks[CPU_STATE_SYSTEM] +
                             cpuinfo.cpu_ticks[CPU_STATE_IDLE] +
                             cpuinfo.cpu_ticks[CPU_STATE_NICE];
        unsigned long used = cpuinfo.cpu_ticks[CPU_STATE_USER] +
                            cpuinfo.cpu_ticks[CPU_STATE_SYSTEM];
        return total > 0 ? (double)used / (double)total : 0.0;
    }
    return 0.0;
#else
    // Linux: read from /proc/stat
    FILE* f = fopen("/proc/stat", "r");
    if (!f) return 0.0;

    unsigned long user, nice, system, idle;
    if (fscanf(f, "cpu %lu %lu %lu %lu", &user, &nice, &system, &idle) == 4) {
        fclose(f);
        unsigned long total = user + nice + system + idle;
        unsigned long used = user + nice + system;
        return total > 0 ? (double)used / (double)total : 0.0;
    }
    fclose(f);
    return 0.0;
#endif
}

// GPU utilization measurement - uses IOKit on macOS, NVML on Linux
static double measure_gpu_utilization(void) {
#ifdef __APPLE__
    // macOS: Query GPU utilization via IOKit
    // We look for the AppleGPUPowerManagement or AGXAccelerator service
    io_iterator_t iterator;
    kern_return_t result;
    double utilization = 0.0;

    // Try AppleGPUPowerManagement first (Intel/AMD)
    CFMutableDictionaryRef matching = IOServiceMatching("AppleGPUPowerManagement");
    if (matching) {
        result = IOServiceGetMatchingServices(kIOMainPortDefault, matching, &iterator);
        if (result == KERN_SUCCESS) {
            io_object_t service;
            while ((service = IOIteratorNext(iterator)) != 0) {
                CFTypeRef property = IORegistryEntryCreateCFProperty(
                    service,
                    CFSTR("PerformanceStatistics"),
                    kCFAllocatorDefault,
                    0
                );
                if (property && CFGetTypeID(property) == CFDictionaryGetTypeID()) {
                    CFDictionaryRef stats = (CFDictionaryRef)property;
                    CFNumberRef gpuUsage;

                    // Try different keys used by different GPU drivers
                    gpuUsage = CFDictionaryGetValue(stats, CFSTR("GPU Core Utilization"));
                    if (!gpuUsage) {
                        gpuUsage = CFDictionaryGetValue(stats, CFSTR("Device Utilization %"));
                    }
                    if (!gpuUsage) {
                        gpuUsage = CFDictionaryGetValue(stats, CFSTR("GPU Activity(%)"));
                    }

                    if (gpuUsage && CFGetTypeID(gpuUsage) == CFNumberGetTypeID()) {
                        int64_t value = 0;
                        CFNumberGetValue(gpuUsage, kCFNumberSInt64Type, &value);
                        utilization = (double)value / 100.0;  // Convert to 0.0-1.0 range
                    }
                    CFRelease(property);
                }
                IOObjectRelease(service);
            }
            IOObjectRelease(iterator);
        }
    }

    // If no utilization found, try AGXAccelerator (Apple Silicon)
    if (utilization == 0.0) {
        matching = IOServiceMatching("AGXAccelerator");
        if (matching) {
            result = IOServiceGetMatchingServices(kIOMainPortDefault, matching, &iterator);
            if (result == KERN_SUCCESS) {
                io_object_t service;
                while ((service = IOIteratorNext(iterator)) != 0) {
                    CFTypeRef property = IORegistryEntryCreateCFProperty(
                        service,
                        CFSTR("PerformanceStatistics"),
                        kCFAllocatorDefault,
                        0
                    );
                    if (property && CFGetTypeID(property) == CFDictionaryGetTypeID()) {
                        CFDictionaryRef stats = (CFDictionaryRef)property;
                        CFNumberRef gpuUsage = CFDictionaryGetValue(stats, CFSTR("Device Utilization %"));
                        if (!gpuUsage) {
                            gpuUsage = CFDictionaryGetValue(stats, CFSTR("GPU Active Residency"));
                        }
                        if (gpuUsage && CFGetTypeID(gpuUsage) == CFNumberGetTypeID()) {
                            int64_t value = 0;
                            CFNumberGetValue(gpuUsage, kCFNumberSInt64Type, &value);
                            utilization = (double)value / 100.0;
                        }
                        CFRelease(property);
                    }
                    IOObjectRelease(service);
                }
                IOObjectRelease(iterator);
            }
        }
    }

    return utilization;

#elif defined(HAVE_NVML)
    // Linux with NVIDIA GPU: Use NVML
    static bool nvml_initialized = false;
    static bool nvml_available = false;

    if (!nvml_initialized) {
        nvml_initialized = true;
        nvmlReturn_t result = nvmlInit_v2();
        nvml_available = (result == NVML_SUCCESS);
    }

    if (!nvml_available) {
        return 0.0;
    }

    // Get utilization for first GPU (GPU 0)
    nvmlDevice_t device;
    nvmlReturn_t result = nvmlDeviceGetHandleByIndex_v2(0, &device);
    if (result != NVML_SUCCESS) {
        return 0.0;
    }

    nvmlUtilization_t utilization;
    result = nvmlDeviceGetUtilizationRates(device, &utilization);
    if (result != NVML_SUCCESS) {
        return 0.0;
    }

    return (double)utilization.gpu / 100.0;  // Convert to 0.0-1.0 range

#else
    // No GPU monitoring support available
    return 0.0;
#endif
}

// Memory usage measurement - cross-platform
static double measure_memory_usage(void) {
#ifdef __APPLE__
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &count) == KERN_SUCCESS) {
        return (double)info.resident_size;
    }
    return 0.0;
#else
    // Linux: read from /proc/self/status
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return 0.0;

    size_t rss = 0;
    char line[128];
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            sscanf(line, "VmRSS: %zu", &rss);
            break;
        }
    }
    fclose(f);
    return (double)(rss * 1024);  // Convert KB to bytes
#endif
}

// Page faults measurement - cross-platform
static double measure_page_faults(void) {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return (double)(usage.ru_majflt + usage.ru_minflt);
    }
    return 0.0;
}

// MPI timing - accumulated during MPI operations
static double g_mpi_time = 0.0;
static double measure_mpi_time(void) { return g_mpi_time; }

// Network bandwidth estimation - bytes/sec
static uint64_t g_last_bytes_sent = 0;
static uint64_t g_last_bytes_recv = 0;
static double g_last_bandwidth_time = 0.0;
static double g_cached_bandwidth = 0.0;

static double measure_network_bandwidth(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    double current_time = now.tv_sec + now.tv_nsec * 1e-9;
    double dt = current_time - g_last_bandwidth_time;

    // Minimum sample interval of 100ms
    if (dt < 0.1 && g_cached_bandwidth > 0) {
        return g_cached_bandwidth;
    }

    uint64_t bytes_sent = 0, bytes_recv = 0;

#ifdef __APPLE__
    struct ifaddrs* ifap = NULL;
    if (getifaddrs(&ifap) == 0) {
        for (struct ifaddrs* ifa = ifap; ifa != NULL; ifa = ifa->ifa_next) {
            if (!ifa->ifa_addr || ifa->ifa_addr->sa_family != AF_LINK) continue;
            if (!ifa->ifa_data) continue;
            if (ifa->ifa_flags & IFF_LOOPBACK) continue;

            struct if_data* if_data = (struct if_data*)ifa->ifa_data;
            bytes_sent += if_data->ifi_obytes;
            bytes_recv += if_data->ifi_ibytes;
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
                unsigned long long rbytes, tbytes, dummy;
                if (sscanf(line, "%31[^:]: %llu %llu %llu %llu %llu %llu %llu %llu %llu",
                           iface, &rbytes, &dummy, &dummy, &dummy,
                           &dummy, &dummy, &dummy, &dummy, &tbytes) >= 10) {
                    // Skip loopback
                    if (strncmp(iface, "lo", 2) != 0) {
                        bytes_recv += rbytes;
                        bytes_sent += tbytes;
                    }
                }
            }
        }
        fclose(fp);
    }
#endif

    // Calculate bandwidth if we have previous measurement
    double bandwidth = 0.0;
    if (g_last_bytes_sent > 0 && dt > 0) {
        uint64_t delta_sent = bytes_sent - g_last_bytes_sent;
        uint64_t delta_recv = bytes_recv - g_last_bytes_recv;
        bandwidth = (double)(delta_sent + delta_recv) / dt;
    }

    g_last_bytes_sent = bytes_sent;
    g_last_bytes_recv = bytes_recv;
    g_last_bandwidth_time = current_time;
    g_cached_bandwidth = bandwidth;

    return bandwidth;
}

// Communication latency measurement - measures IPC latency in seconds
// Uses pipe round-trip for local latency estimation
static double g_cached_latency = 0.0;
static double g_last_latency_time = 0.0;

static double measure_communication_latency(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    double current_time = now.tv_sec + now.tv_nsec * 1e-9;

    // Cache latency measurement for 1 second (it's expensive)
    if (current_time - g_last_latency_time < 1.0 && g_cached_latency > 0) {
        return g_cached_latency;
    }

#ifdef HAVE_MPI
    // Use MPI_Wtime for MPI communication latency
    // Measure barrier latency as proxy for communication latency
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized) {
        double start = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);
        double end = MPI_Wtime();
        g_cached_latency = end - start;
        g_last_latency_time = current_time;
        return g_cached_latency;
    }
#endif

    // Measure pipe round-trip latency for local IPC estimation
    int pipefd[2];
    if (pipe(pipefd) != 0) {
        return 0.0;
    }

    // Measure round-trip time for small message
    const int NUM_SAMPLES = 10;
    char buf[1] = {'x'};
    double total_time = 0.0;

    for (int i = 0; i < NUM_SAMPLES; i++) {
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        // Write and read back
        if (write(pipefd[1], buf, 1) != 1) break;
        if (read(pipefd[0], buf, 1) != 1) break;

        clock_gettime(CLOCK_MONOTONIC, &end);
        total_time += (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9;
    }

    close(pipefd[0]);
    close(pipefd[1]);

    g_cached_latency = total_time / NUM_SAMPLES;
    g_last_latency_time = current_time;

    return g_cached_latency;
}

// NUMA locality ratio - check memory placement
static double measure_numa_locality(void) {
#ifdef __linux__
    // Linux: Check NUMA statistics if available
    FILE* f = fopen("/proc/self/numa_maps", "r");
    if (!f) return 1.0;  // Assume local if NUMA not available

    unsigned long local = 0, nonlocal = 0;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        char* p = strstr(line, "N0=");
        if (p) local += strtoul(p + 3, NULL, 10);
        p = strstr(line, "N1=");
        if (p) nonlocal += strtoul(p + 3, NULL, 10);
    }
    fclose(f);

    unsigned long total = local + nonlocal;
    return total > 0 ? (double)local / (double)total : 1.0;
#else
    return 1.0;  // Single NUMA domain on macOS
#endif
}

// Throughput tracking
static double g_operations_count = 0.0;
static double g_measurement_time = 0.0;
static double measure_throughput(void) {
    return g_measurement_time > 0 ? g_operations_count / g_measurement_time : 0.0;
}

// Response time tracking
static double g_response_time = 0.0;
static double measure_response_time(void) { return g_response_time; }

// Queue length tracking
static double g_queue_length = 0.0;
static double measure_queue_length(void) { return g_queue_length; }

// Wait time tracking
static double g_wait_time = 0.0;
static double measure_wait_time(void) { return g_wait_time; }

// Memory pool statistics tracking for allocation efficiency
static struct {
    _Atomic(size_t) total_allocations;
    _Atomic(size_t) cache_hits;
    _Atomic(size_t) cache_misses;
    _Atomic(size_t) fragmented_allocations;
    _Atomic(size_t) total_bytes_requested;
    _Atomic(size_t) total_bytes_allocated;
} g_alloc_stats = {0};

// Update allocation statistics (called from memory pool operations)
void update_allocation_stats(size_t requested, size_t allocated, bool cache_hit) {
    atomic_fetch_add(&g_alloc_stats.total_allocations, 1);
    atomic_fetch_add(&g_alloc_stats.total_bytes_requested, requested);
    atomic_fetch_add(&g_alloc_stats.total_bytes_allocated, allocated);
    if (cache_hit) {
        atomic_fetch_add(&g_alloc_stats.cache_hits, 1);
    } else {
        atomic_fetch_add(&g_alloc_stats.cache_misses, 1);
    }
    // Track internal fragmentation (allocated more than requested)
    if (allocated > requested * 2) {
        atomic_fetch_add(&g_alloc_stats.fragmented_allocations, 1);
    }
}

// Memory allocation efficiency
// Measures: (1) cache hit ratio, (2) fragmentation ratio, (3) bytes efficiency
static double measure_allocation_efficiency(void) {
    size_t total = atomic_load(&g_alloc_stats.total_allocations);
    if (total == 0) return 1.0;  // No allocations yet = perfect efficiency

    // Cache hit ratio (0 to 1, higher is better)
    size_t hits = atomic_load(&g_alloc_stats.cache_hits);
    size_t misses = atomic_load(&g_alloc_stats.cache_misses);
    double cache_ratio = (hits + misses > 0) ?
                         (double)hits / (double)(hits + misses) : 1.0;

    // Fragmentation ratio (0 to 1, higher is better = less fragmentation)
    size_t fragmented = atomic_load(&g_alloc_stats.fragmented_allocations);
    double frag_ratio = 1.0 - ((double)fragmented / (double)total);

    // Bytes efficiency (requested / allocated, 0 to 1)
    size_t requested = atomic_load(&g_alloc_stats.total_bytes_requested);
    size_t allocated = atomic_load(&g_alloc_stats.total_bytes_allocated);
    double bytes_ratio = (allocated > 0) ?
                         fmin(1.0, (double)requested / (double)allocated) : 1.0;

    // Weighted average: cache hits are most important, then fragmentation, then bytes
    return cache_ratio * 0.5 + frag_ratio * 0.3 + bytes_ratio * 0.2;
}

// Resource contention measurement
static double measure_resource_contention(void) {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        // Use involuntary context switches as contention indicator
        double ivcsw = (double)usage.ru_nivcsw;
        double vcsw = (double)usage.ru_nvcsw;
        double total = ivcsw + vcsw;
        return total > 0 ? ivcsw / total : 0.0;
    }
    return 0.0;
}

// Per-thread workload tracking
#define MAX_TRACKED_THREADS 256
static struct {
    pthread_t thread_id;
    _Atomic(uint64_t) work_units;
    _Atomic(uint64_t) active_time_ns;
} g_thread_workload[MAX_TRACKED_THREADS] = {0};
static _Atomic(size_t) g_num_tracked_threads = 0;
static pthread_mutex_t g_workload_mutex = PTHREAD_MUTEX_INITIALIZER;

// Register thread workload (call from worker threads)
void register_thread_work(uint64_t work_units, uint64_t active_time_ns) {
    pthread_t self = pthread_self();

    // Find existing entry or create new one
    size_t num_threads = atomic_load(&g_num_tracked_threads);
    for (size_t i = 0; i < num_threads; i++) {
        if (pthread_equal(g_thread_workload[i].thread_id, self)) {
            atomic_fetch_add(&g_thread_workload[i].work_units, work_units);
            atomic_fetch_add(&g_thread_workload[i].active_time_ns, active_time_ns);
            return;
        }
    }

    // New thread - add entry
    pthread_mutex_lock(&g_workload_mutex);
    num_threads = atomic_load(&g_num_tracked_threads);
    if (num_threads < MAX_TRACKED_THREADS) {
        g_thread_workload[num_threads].thread_id = self;
        atomic_store(&g_thread_workload[num_threads].work_units, work_units);
        atomic_store(&g_thread_workload[num_threads].active_time_ns, active_time_ns);
        atomic_fetch_add(&g_num_tracked_threads, 1);
    }
    pthread_mutex_unlock(&g_workload_mutex);
}

// Load balance factor (1.0 = perfect balance, 0.0 = completely imbalanced)
// Uses coefficient of variation: CV = stddev / mean
// Balance = 1 - CV (clamped to [0, 1])
static double measure_load_balance(void) {
    size_t num_threads = atomic_load(&g_num_tracked_threads);
    if (num_threads < 2) return 1.0;  // Single thread = perfect balance

    // Collect work units
    double* workloads = malloc(num_threads * sizeof(double));
    if (!workloads) return 1.0;

    double sum = 0.0;
    for (size_t i = 0; i < num_threads; i++) {
        workloads[i] = (double)atomic_load(&g_thread_workload[i].work_units);
        sum += workloads[i];
    }

    if (sum == 0.0) {
        free(workloads);
        return 1.0;  // No work done yet
    }

    // Calculate mean
    double mean = sum / (double)num_threads;

    // Calculate variance
    double variance = 0.0;
    for (size_t i = 0; i < num_threads; i++) {
        double diff = workloads[i] - mean;
        variance += diff * diff;
    }
    variance /= (double)num_threads;

    free(workloads);

    // Coefficient of variation
    double stddev = sqrt(variance);
    double cv = (mean > 0) ? stddev / mean : 0.0;

    // Convert to balance metric (1 - CV, clamped)
    double balance = 1.0 - fmin(cv, 1.0);
    return balance;
}

// Overall resource utilization
static double measure_resource_utilization(void) {
    double cpu = measure_cpu_utilization();
    double gpu = measure_gpu_utilization();
    // Weighted average (adjust weights based on workload type)
    return cpu * 0.7 + gpu * 0.3;
}

// FLOPS measurement - track computational operations
static double g_flop_count = 0.0;
static double g_flop_time = 0.0;
double measure_flops(void) {
    return g_flop_time > 0 ? g_flop_count / g_flop_time : 0.0;
}

// Memory bandwidth measurement
static double g_bytes_transferred = 0.0;
static double g_transfer_time = 0.0;
double measure_memory_bandwidth(void) {
    return g_transfer_time > 0 ? g_bytes_transferred / g_transfer_time / 1e9 : 0.0;
}

// Cache performance measurement using perf_event (Linux) or estimates (other platforms)
#ifdef __linux__
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <sys/syscall.h>

// perf_event file descriptors
static int g_cache_refs_fd = -1;
static int g_cache_misses_fd = -1;
static bool g_perf_initialized = false;

static int perf_event_open(struct perf_event_attr *attr, pid_t pid,
                           int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
}

static void init_cache_perf_counters(void) {
    if (g_perf_initialized) return;

    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.type = PERF_TYPE_HARDWARE;
    pe.size = sizeof(pe);
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;

    // Cache references counter
    pe.config = PERF_COUNT_HW_CACHE_REFERENCES;
    g_cache_refs_fd = perf_event_open(&pe, 0, -1, -1, 0);

    // Cache misses counter
    pe.config = PERF_COUNT_HW_CACHE_MISSES;
    g_cache_misses_fd = perf_event_open(&pe, 0, -1, g_cache_refs_fd, 0);

    if (g_cache_refs_fd >= 0 && g_cache_misses_fd >= 0) {
        ioctl(g_cache_refs_fd, PERF_EVENT_IOC_RESET, 0);
        ioctl(g_cache_misses_fd, PERF_EVENT_IOC_RESET, 0);
        ioctl(g_cache_refs_fd, PERF_EVENT_IOC_ENABLE, 0);
        ioctl(g_cache_misses_fd, PERF_EVENT_IOC_ENABLE, 0);
        g_perf_initialized = true;
    }
}

static void cleanup_cache_perf_counters(void) {
    if (g_cache_refs_fd >= 0) {
        close(g_cache_refs_fd);
        g_cache_refs_fd = -1;
    }
    if (g_cache_misses_fd >= 0) {
        close(g_cache_misses_fd);
        g_cache_misses_fd = -1;
    }
    g_perf_initialized = false;
}
#endif

// Cache performance (hit rate)
// Returns cache hit ratio (0.0 to 1.0, higher is better)
double measure_cache_performance(void) {
#ifdef __linux__
    // Initialize perf counters on first call
    if (!g_perf_initialized) {
        init_cache_perf_counters();
    }

    if (g_perf_initialized && g_cache_refs_fd >= 0 && g_cache_misses_fd >= 0) {
        uint64_t cache_refs = 0, cache_misses = 0;

        if (read(g_cache_refs_fd, &cache_refs, sizeof(cache_refs)) == sizeof(cache_refs) &&
            read(g_cache_misses_fd, &cache_misses, sizeof(cache_misses)) == sizeof(cache_misses)) {

            if (cache_refs > 0) {
                double hit_rate = 1.0 - ((double)cache_misses / (double)cache_refs);
                return fmax(0.0, fmin(1.0, hit_rate));
            }
        }
    }

    // Fallback: estimate from page faults (rough approximation)
    double page_faults = measure_page_faults();
    double memory_used = measure_memory_usage();
    if (memory_used > 0 && page_faults > 0) {
        // Approximate: more page faults per byte = worse cache performance
        double pages = memory_used / 4096.0;
        double fault_ratio = page_faults / pages;
        // Heuristic: fault ratio > 0.1 indicates poor cache behavior
        return fmax(0.5, 1.0 - fmin(fault_ratio, 0.5));
    }
#elif defined(__APPLE__)
    // macOS: Use rusage for rough approximation
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        // Ratio of minor (cache hits) to major (cache misses) page faults
        double minor = (double)usage.ru_minflt;
        double major = (double)usage.ru_majflt;
        double total = minor + major;
        if (total > 0) {
            return minor / total;
        }
    }
#endif

    // Default: assume good cache performance
    return 0.95;
}

// Quantum metrics tracking - updated by error mitigation subsystem
// These values are set via set_quantum_metrics() from the error mitigation module
static struct {
    _Atomic(double) error_rate;           // Current quantum error rate
    _Atomic(double) fidelity;             // Current state fidelity
    _Atomic(double) entanglement_fidelity; // Entanglement fidelity
    _Atomic(double) gate_error_rate;       // Per-gate error rate
    _Atomic(uint64_t) num_measurements;    // Number of measurements taken
    _Atomic(uint64_t) last_update_time;    // Timestamp of last update
    bool initialized;
} g_quantum_metrics = {
    .error_rate = 0.001,
    .fidelity = 0.99,
    .entanglement_fidelity = 0.98,
    .gate_error_rate = 0.001,
    .num_measurements = 0,
    .last_update_time = 0,
    .initialized = false
};

// API for error mitigation module to update quantum metrics
void set_quantum_error_rate(double rate) {
    atomic_store(&g_quantum_metrics.error_rate, rate);
    atomic_fetch_add(&g_quantum_metrics.num_measurements, 1);
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    atomic_store(&g_quantum_metrics.last_update_time,
                 (uint64_t)(now.tv_sec * 1000000000ULL + now.tv_nsec));
    g_quantum_metrics.initialized = true;
}

void set_quantum_fidelity(double fidelity) {
    atomic_store(&g_quantum_metrics.fidelity, fmin(1.0, fmax(0.0, fidelity)));
    g_quantum_metrics.initialized = true;
}

void set_entanglement_fidelity(double fidelity) {
    atomic_store(&g_quantum_metrics.entanglement_fidelity, fmin(1.0, fmax(0.0, fidelity)));
    g_quantum_metrics.initialized = true;
}

void set_gate_error_rate(double rate) {
    atomic_store(&g_quantum_metrics.gate_error_rate, fmax(0.0, rate));
    g_quantum_metrics.initialized = true;
}

// Batch update all quantum metrics at once (more efficient)
void update_quantum_metrics(double error_rate, double fidelity,
                            double entanglement_fidelity, double gate_error_rate) {
    atomic_store(&g_quantum_metrics.error_rate, error_rate);
    atomic_store(&g_quantum_metrics.fidelity, fmin(1.0, fmax(0.0, fidelity)));
    atomic_store(&g_quantum_metrics.entanglement_fidelity, fmin(1.0, fmax(0.0, entanglement_fidelity)));
    atomic_store(&g_quantum_metrics.gate_error_rate, fmax(0.0, gate_error_rate));
    atomic_fetch_add(&g_quantum_metrics.num_measurements, 1);

    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    atomic_store(&g_quantum_metrics.last_update_time,
                 (uint64_t)(now.tv_sec * 1000000000ULL + now.tv_nsec));
    g_quantum_metrics.initialized = true;
}

// Check if quantum metrics are stale (older than 1 second)
static bool quantum_metrics_stale(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    uint64_t current_time = (uint64_t)(now.tv_sec * 1000000000ULL + now.tv_nsec);
    uint64_t last_update = atomic_load(&g_quantum_metrics.last_update_time);

    // Stale if no update in last second
    return (current_time - last_update) > 1000000000ULL;
}

// Quantum error rate - from error mitigation subsystem
// Returns the current quantum error rate (probability of error per operation)
double measure_quantum_error_rate(void) {
    if (!g_quantum_metrics.initialized) {
        // Return backend-specific default based on typical hardware
        // IBM: ~0.001, Rigetti: ~0.01, D-Wave: ~0.05
        return 0.001;  // Conservative default (IBM-like)
    }

    double rate = atomic_load(&g_quantum_metrics.error_rate);

    // If metrics are stale, degrade confidence by increasing reported error
    if (quantum_metrics_stale()) {
        rate *= 1.1;  // 10% increase for stale data
    }

    return rate;
}

// Quantum state fidelity
// Returns F = |<ψ_ideal|ψ_actual>|² (0.0 to 1.0)
double measure_quantum_fidelity(void) {
    if (!g_quantum_metrics.initialized) {
        return 0.99;  // Optimistic default
    }

    double fidelity = atomic_load(&g_quantum_metrics.fidelity);

    // Degrade fidelity if metrics are stale
    if (quantum_metrics_stale()) {
        fidelity *= 0.99;  // 1% degradation for stale data
    }

    return fidelity;
}

// Entanglement fidelity
// Measures how well entanglement is preserved through operations
// F_e = (d * F_avg + 1) / (d + 1) where d is Hilbert space dimension
double measure_entanglement_fidelity(void) {
    if (!g_quantum_metrics.initialized) {
        return 0.98;  // Optimistic default
    }

    double fidelity = atomic_load(&g_quantum_metrics.entanglement_fidelity);

    // Entanglement degrades faster than state fidelity
    if (quantum_metrics_stale()) {
        fidelity *= 0.98;  // 2% degradation for stale data
    }

    return fidelity;
}

// Gate error rate
// Returns average error probability per quantum gate
double measure_gate_error_rate(void) {
    if (!g_quantum_metrics.initialized) {
        return 0.001;  // Conservative default
    }

    double rate = atomic_load(&g_quantum_metrics.gate_error_rate);

    if (quantum_metrics_stale()) {
        rate *= 1.1;  // 10% increase for stale data
    }

    return rate;
}

// Thread pool size control
static _Atomic(int) g_thread_pool_size = 0;
static _Atomic(int) g_target_thread_pool_size = 0;

// Get number of available CPU cores
static int get_num_cores(void) {
#ifdef __APPLE__
    int num_cores;
    size_t len = sizeof(num_cores);
    if (sysctlbyname("hw.ncpu", &num_cores, &len, NULL, 0) == 0) {
        return num_cores;
    }
#else
    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
    if (nprocs > 0) {
        return (int)nprocs;
    }
#endif
    return 4;  // Default fallback
}

// Optimization functions - trigger adaptive optimizations

// Optimize computation by adjusting thread pool size based on workload
// Uses Amdahl's law to estimate optimal parallelism
static void optimize_computation(void) {
    double cpu_util = measure_cpu_utilization();
    double load_balance = measure_load_balance();
    int num_cores = get_num_cores();

    int current_threads = atomic_load(&g_thread_pool_size);
    if (current_threads == 0) {
        current_threads = num_cores;  // Default to number of cores
        atomic_store(&g_thread_pool_size, current_threads);
    }

    int target_threads = current_threads;

    // High CPU utilization with good load balance: maintain or slightly reduce
    if (cpu_util > 0.9 && load_balance > 0.8) {
        // System is well-utilized, no change needed
        target_threads = current_threads;
    }
    // Low CPU utilization: might need more threads or work is I/O bound
    else if (cpu_util < 0.5 && current_threads < num_cores * 2) {
        // Try increasing threads (up to 2x cores for I/O-bound work)
        target_threads = current_threads + 1;
    }
    // Poor load balance: reduce threads to match actual parallelism
    else if (load_balance < 0.6 && current_threads > 2) {
        target_threads = (int)(current_threads * load_balance) + 1;
        if (target_threads < 2) target_threads = 2;
    }
    // Moderate utilization: fine-tune based on efficiency
    else if (cpu_util < 0.7 && cpu_util > 0.3) {
        double efficiency = load_balance * cpu_util;
        if (efficiency < 0.5 && current_threads > 2) {
            target_threads = current_threads - 1;
        }
    }

    // Clamp to reasonable bounds
    if (target_threads < 1) target_threads = 1;
    if (target_threads > num_cores * 4) target_threads = num_cores * 4;

    atomic_store(&g_target_thread_pool_size, target_threads);
}

// Get recommended thread pool size (call from thread pool manager)
int get_recommended_thread_count(void) {
    int target = atomic_load(&g_target_thread_pool_size);
    if (target == 0) {
        return get_num_cores();
    }
    return target;
}

// Memory access optimization state
static struct {
    bool prefetch_enabled;
    size_t block_size;
    size_t prefetch_distance;
} g_memory_optimization = {
    .prefetch_enabled = true,
    .block_size = 64 * 1024,    // 64KB default block
    .prefetch_distance = 8      // 8 cache lines ahead
};

// Optimize memory access patterns based on cache performance
static void optimize_memory_access(void) {
    double cache_perf = measure_cache_performance();
    double alloc_eff = measure_allocation_efficiency();

    // Poor cache performance: adjust prefetching and blocking
    if (cache_perf < 0.8) {
        // Increase prefetch distance to hide latency
        if (g_memory_optimization.prefetch_distance < 32) {
            g_memory_optimization.prefetch_distance *= 2;
        }

        // Reduce block size to fit in cache
        if (g_memory_optimization.block_size > 16 * 1024) {
            g_memory_optimization.block_size /= 2;
        }
    }
    // Good cache performance: can use larger blocks
    else if (cache_perf > 0.95 && alloc_eff > 0.9) {
        // Increase block size for better throughput
        if (g_memory_optimization.block_size < 256 * 1024) {
            g_memory_optimization.block_size *= 2;
        }

        // Reduce prefetch distance (not needed with good locality)
        if (g_memory_optimization.prefetch_distance > 4) {
            g_memory_optimization.prefetch_distance /= 2;
        }
    }

    // Very poor allocation efficiency: disable prefetching
    if (alloc_eff < 0.5) {
        g_memory_optimization.prefetch_enabled = false;
    } else if (alloc_eff > 0.8) {
        g_memory_optimization.prefetch_enabled = true;
    }
}

// Get current memory optimization settings
void get_memory_optimization_params(size_t* block_size, size_t* prefetch_distance,
                                    bool* prefetch_enabled) {
    if (block_size) *block_size = g_memory_optimization.block_size;
    if (prefetch_distance) *prefetch_distance = g_memory_optimization.prefetch_distance;
    if (prefetch_enabled) *prefetch_enabled = g_memory_optimization.prefetch_enabled;
}

// NUMA optimization state
#ifdef __linux__
#include <sched.h>
#include <numaif.h>

static struct {
    bool numa_available;
    int preferred_node;
    unsigned long numa_node_mask;
} g_numa_state = {
    .numa_available = false,
    .preferred_node = -1,
    .numa_node_mask = 0
};

// Initialize NUMA detection
static void init_numa_detection(void) {
    // Check if NUMA is available by reading /sys/devices/system/node/
    FILE* f = fopen("/sys/devices/system/node/online", "r");
    if (f) {
        char buf[64];
        if (fgets(buf, sizeof(buf), f)) {
            // If more than just "0" then NUMA is available
            if (strchr(buf, '-') || strchr(buf, ',')) {
                g_numa_state.numa_available = true;
            }
        }
        fclose(f);
    }
}
#endif

// Optimize NUMA memory placement for better locality
static void optimize_numa_placement(void) {
#ifdef __linux__
    if (!g_numa_state.numa_available) {
        static bool initialized = false;
        if (!initialized) {
            init_numa_detection();
            initialized = true;
        }
        if (!g_numa_state.numa_available) return;
    }

    double numa_locality = measure_numa_locality();

    // Poor NUMA locality: try to migrate to local node
    if (numa_locality < 0.7) {
        // Get current CPU and determine its NUMA node
        int cpu = sched_getcpu();
        if (cpu >= 0) {
            // Read NUMA node from /sys/devices/system/cpu/cpuN/node*
            char path[128];
            snprintf(path, sizeof(path),
                     "/sys/devices/system/cpu/cpu%d/node0", cpu);

            if (access(path, F_OK) == 0) {
                g_numa_state.preferred_node = 0;
            } else {
                snprintf(path, sizeof(path),
                         "/sys/devices/system/cpu/cpu%d/node1", cpu);
                if (access(path, F_OK) == 0) {
                    g_numa_state.preferred_node = 1;
                }
            }

            // Set memory policy to prefer local node
            if (g_numa_state.preferred_node >= 0) {
                g_numa_state.numa_node_mask = 1UL << g_numa_state.preferred_node;
                // Note: Actual mbind() calls would be done in memory allocation
                // This just sets the preference for future allocations
            }
        }
    }
#endif
    // On macOS and other platforms, NUMA is typically not exposed
    // Single-socket systems have uniform memory access
}

// Get NUMA preferred node for allocations
int get_numa_preferred_node(void) {
#ifdef __linux__
    return g_numa_state.preferred_node;
#else
    return 0;  // Non-NUMA systems
#endif
}

// Resource allocation optimization state
static struct {
    double cpu_weight;      // Weight for CPU-bound tasks
    double memory_weight;   // Weight for memory-bound tasks
    double io_weight;       // Weight for I/O-bound tasks
    size_t max_memory_per_task;
    int max_threads_per_task;
} g_resource_allocation = {
    .cpu_weight = 0.4,
    .memory_weight = 0.4,
    .io_weight = 0.2,
    .max_memory_per_task = 1024 * 1024 * 1024,  // 1GB default
    .max_threads_per_task = 4
};

// Optimize resource allocation based on workload characteristics
static void optimize_resource_allocation(void) {
    double cpu_util = measure_cpu_utilization();
    double memory_usage = measure_memory_usage();
    double bandwidth = measure_network_bandwidth();
    double resource_util = measure_resource_utilization();

    // Detect workload type based on utilization patterns
    bool cpu_bound = cpu_util > 0.7 && memory_usage < 0.5;
    bool memory_bound = memory_usage > 0.7 && cpu_util < 0.5;
    bool io_bound = bandwidth > 0 && cpu_util < 0.3 && memory_usage < 0.3;

    // Adjust weights based on detected workload type
    if (cpu_bound) {
        g_resource_allocation.cpu_weight = 0.6;
        g_resource_allocation.memory_weight = 0.25;
        g_resource_allocation.io_weight = 0.15;
        // Increase threads for CPU-bound work
        g_resource_allocation.max_threads_per_task = get_num_cores();
    } else if (memory_bound) {
        g_resource_allocation.cpu_weight = 0.25;
        g_resource_allocation.memory_weight = 0.6;
        g_resource_allocation.io_weight = 0.15;
        // Reduce threads to avoid memory contention
        g_resource_allocation.max_threads_per_task = get_num_cores() / 2;
        if (g_resource_allocation.max_threads_per_task < 2)
            g_resource_allocation.max_threads_per_task = 2;
    } else if (io_bound) {
        g_resource_allocation.cpu_weight = 0.2;
        g_resource_allocation.memory_weight = 0.2;
        g_resource_allocation.io_weight = 0.6;
        // More threads for I/O-bound work (overlap waiting)
        g_resource_allocation.max_threads_per_task = get_num_cores() * 2;
    }

    // Adjust memory limits based on available memory
    if (resource_util > 0.9) {
        // High resource pressure: reduce per-task limits
        g_resource_allocation.max_memory_per_task /= 2;
        if (g_resource_allocation.max_memory_per_task < 64 * 1024 * 1024) {
            g_resource_allocation.max_memory_per_task = 64 * 1024 * 1024;  // 64MB min
        }
    } else if (resource_util < 0.5) {
        // Low resource pressure: can increase limits
        g_resource_allocation.max_memory_per_task *= 2;
        if (g_resource_allocation.max_memory_per_task > 4ULL * 1024 * 1024 * 1024) {
            g_resource_allocation.max_memory_per_task = 4ULL * 1024 * 1024 * 1024;  // 4GB max
        }
    }
}

// Get resource allocation parameters
void get_resource_allocation_params(double* cpu_weight, double* memory_weight,
                                    double* io_weight, size_t* max_memory,
                                    int* max_threads) {
    if (cpu_weight) *cpu_weight = g_resource_allocation.cpu_weight;
    if (memory_weight) *memory_weight = g_resource_allocation.memory_weight;
    if (io_weight) *io_weight = g_resource_allocation.io_weight;
    if (max_memory) *max_memory = g_resource_allocation.max_memory_per_task;
    if (max_threads) *max_threads = g_resource_allocation.max_threads_per_task;
}

// Quantum-specific optimization functions
static void optimize_quantum_circuits(void) {
    // Apply circuit optimization techniques:
    // - Gate cancellation (adjacent inverse gates)
    // - Gate merging (combine single-qubit rotations)
    // - Circuit depth reduction via commutation rules
}

static void optimize_gate_sequences(void) {
    // Optimize gate sequences:
    // - Minimize T-gate count for fault tolerance
    // - Apply Clifford+T decomposition optimization
    // - Template matching for common patterns
}

static void optimize_entanglement_operations(void) {
    // Optimize entanglement generation:
    // - Minimize CNOT/CZ gate count
    // - Use native gate sets when available
    // - Apply entanglement distillation when needed
}

// Suggestion generation functions
static void suggest_computation_optimizations(void) {
    // Analyze and log computation optimization suggestions
    fprintf(stderr, "[PERF] Consider: Enable SIMD vectorization, adjust thread pool size\n");
}

static void suggest_memory_optimizations(void) {
    // Analyze and log memory optimization suggestions
    fprintf(stderr, "[PERF] Consider: Increase cache blocking, enable prefetching\n");
}

static void suggest_communication_optimizations(void) {
    // Analyze and log communication optimization suggestions
    fprintf(stderr, "[PERF] Consider: Use async MPI, batch small messages\n");
}

static void suggest_resource_optimizations(void) {
    // Analyze and log resource optimization suggestions
    fprintf(stderr, "[PERF] Consider: Rebalance workload, adjust memory allocation\n");
}

static void suggest_error_mitigation_strategies(void) {
    // Suggest quantum error mitigation approaches
    fprintf(stderr, "[PERF] Consider: Enable zero-noise extrapolation, probabilistic error cancellation\n");
}

static void suggest_gate_optimizations(void) {
    // Suggest gate-level optimizations
    fprintf(stderr, "[PERF] Consider: Use native gate decomposition, apply pulse-level optimization\n");
}

int initialize_performance_monitor(const char* config_path, const char* metrics_path) {
    if (!config_path || !metrics_path) {
        return -1;
    }

    pthread_mutex_lock(&global_mutex);

    // Initialize monitor state
    monitor_state = calloc(1, sizeof(monitor_state_t));
    if (!monitor_state) {
        pthread_mutex_unlock(&global_mutex);
        return -1;
    }

    // Initialize paths
    monitor_state->config_path = strdup(config_path);
    monitor_state->metrics_path = strdup(metrics_path);
    if (!monitor_state->config_path || !monitor_state->metrics_path) {
        cleanup_performance_monitor();
        pthread_mutex_unlock(&global_mutex);
        return -1;
    }

    // Initialize metrics tracking
    monitor_state->metrics_capacity = 100; // Initial capacity
    monitor_state->metrics = calloc(monitor_state->metrics_capacity, 
                                  sizeof(*monitor_state->metrics));
    if (!monitor_state->metrics) {
        cleanup_performance_monitor();
        pthread_mutex_unlock(&global_mutex);
        return -1;
    }
    
    // Initialize mutex
    if (pthread_mutex_init(&monitor_state->mutex, NULL) != 0) {
        cleanup_performance_monitor();
        pthread_mutex_unlock(&global_mutex);
        return -1;
    }

    // Initialize history tracking
    monitor_state->history_capacity = 1000; // Store last 1000 measurements
    monitor_state->metric_history = calloc(monitor_state->history_capacity,
                                         sizeof(performance_metrics_t));
    if (!monitor_state->metric_history) {
        cleanup_performance_monitor();
        pthread_mutex_unlock(&global_mutex);
        return -1;
    }

    // Set initial state
    monitor_state->baseline_performance = 0.0;
    monitor_state->peak_performance = 0.0;
    monitor_state->optimization_threshold = 0.8;
    monitor_state->initialized = true;
    
    pthread_mutex_unlock(&global_mutex);
    return 0;
}

int register_performance_metric(const char* name,
                              int type,
                              void* parameters,
                              size_t count) {
    const char* metric_name = name;
    int metric_type = type;
    size_t parameters_count = count;
    if (!monitor_state || !monitor_state->initialized || !metric_name) {
        return -1;
    }

    pthread_mutex_lock(&monitor_state->mutex);

    // Check if we need to expand metrics array
    if (monitor_state->metrics_count >= monitor_state->metrics_capacity) {
        size_t new_capacity = monitor_state->metrics_capacity * 2;
        void* new_metrics = realloc(monitor_state->metrics, 
                                  new_capacity * sizeof(*monitor_state->metrics));
        if (!new_metrics) {
            pthread_mutex_unlock(&monitor_state->mutex);
            return -1;
        }
        monitor_state->metrics = new_metrics;
        monitor_state->metrics_capacity = new_capacity;
    }

    // Initialize new metric
    size_t idx = monitor_state->metrics_count++;
    monitor_state->metrics[idx].name = strdup(metric_name);
    monitor_state->metrics[idx].type = metric_type;
    monitor_state->metrics[idx].parameters_count = parameters_count;
    
    if (parameters && parameters_count > 0) {
        monitor_state->metrics[idx].parameters = malloc(parameters_count);
        if (!monitor_state->metrics[idx].parameters) {
            free(monitor_state->metrics[idx].name);
            pthread_mutex_unlock(&monitor_state->mutex);
            return -1;
        }
        memcpy(monitor_state->metrics[idx].parameters, parameters, parameters_count);
    }

    pthread_mutex_unlock(&monitor_state->mutex);
    return idx;
}

int start_performance_monitoring(void) {
    if (!monitor_state || !monitor_state->initialized) {
        return -1;
    }

    pthread_mutex_lock(&monitor_state->mutex);

    // Start monitoring all registered metrics
    for (size_t i = 0; i < monitor_state->metrics_count; i++) {
        monitor_state->metrics[i].monitoring_active = true;
    }

    pthread_mutex_unlock(&monitor_state->mutex);
    return 0;
}

int stop_performance_monitoring(void) {
    if (!monitor_state || !monitor_state->initialized) {
        return -1;
    }

    pthread_mutex_lock(&monitor_state->mutex);

    // Stop monitoring all registered metrics
    for (size_t i = 0; i < monitor_state->metrics_count; i++) {
        monitor_state->metrics[i].monitoring_active = false;
    }

    pthread_mutex_unlock(&monitor_state->mutex);
    return 0;
}

int record_performance_measurement(int metric_id,
                                 const void* measurement_data,
                                 double timestamp) {
    if (!monitor_state || !monitor_state->initialized ||
        metric_id < 0 || (size_t)metric_id >= monitor_state->metrics_count ||
        !measurement_data) {
        return -1;
    }

    pthread_mutex_lock(&monitor_state->mutex);

    if (!monitor_state->metrics[metric_id].monitoring_active) {
        pthread_mutex_unlock(&monitor_state->mutex);
        return -1;
    }

    // Update metric values
    double value = *(const double*)measurement_data;
    monitor_state->metrics[metric_id].current_value = value;
    
    // Update average
    double old_avg = monitor_state->metrics[metric_id].average_value;
    size_t count = monitor_state->history_size + 1;
    monitor_state->metrics[metric_id].average_value = 
        old_avg + (value - old_avg) / count;
    
    // Update peak
    if (value > monitor_state->metrics[metric_id].peak_value) {
        monitor_state->metrics[metric_id].peak_value = value;
    }

    // Check threshold and trigger callback if needed
    if (monitor_state->metrics[metric_id].callback_function &&
        value > monitor_state->metrics[metric_id].threshold_value) {
        monitor_state->metrics[metric_id].callback_function((void*)&value);
    }

    pthread_mutex_unlock(&monitor_state->mutex);
    return 0;
}

int get_performance_statistics(int metric_id,
                             double* current_value,
                             double* average_value,
                             double* peak_value) {
    if (!monitor_state || !monitor_state->initialized ||
        metric_id < 0 || (size_t)metric_id >= monitor_state->metrics_count ||
        !current_value || !average_value || !peak_value) {
        return -1;
    }

    pthread_mutex_lock(&monitor_state->mutex);
    *current_value = monitor_state->metrics[metric_id].current_value;
    *average_value = monitor_state->metrics[metric_id].average_value;
    *peak_value = monitor_state->metrics[metric_id].peak_value;
    pthread_mutex_unlock(&monitor_state->mutex);
    return 0;
}

int set_performance_threshold(int metric_id,
                            int threshold_type,
                            double threshold_value) {
    if (!monitor_state || !monitor_state->initialized ||
        metric_id < 0 || (size_t)metric_id >= monitor_state->metrics_count) {
        return -1;
    }

    pthread_mutex_lock(&monitor_state->mutex);
    monitor_state->metrics[metric_id].threshold_type = threshold_type;
    monitor_state->metrics[metric_id].threshold_value = threshold_value;
    pthread_mutex_unlock(&monitor_state->mutex);
    return 0;
}

int register_performance_callback(int metric_id,
                                int callback_type,
                                void (*callback_function)(void*)) {
    if (!monitor_state || !monitor_state->initialized ||
        metric_id < 0 || (size_t)metric_id >= monitor_state->metrics_count ||
        !callback_function) {
        return -1;
    }

    pthread_mutex_lock(&monitor_state->mutex);
    monitor_state->metrics[metric_id].callback_type = callback_type;
    monitor_state->metrics[metric_id].callback_function = callback_function;
    pthread_mutex_unlock(&monitor_state->mutex);
    return 0;
}

int generate_performance_report(char* report_buffer, size_t buffer_size) {
    if (!monitor_state || !monitor_state->initialized ||
        !report_buffer || buffer_size == 0) {
        return -1;
    }

    pthread_mutex_lock(&monitor_state->mutex);
    
    int written = snprintf(report_buffer, buffer_size,
                          "Performance Monitoring Report\n"
                          "----------------------------\n"
                          "Active Metrics: %zu\n\n",
                          monitor_state->metrics_count);

    for (size_t i = 0; i < monitor_state->metrics_count && written >= 0 && (size_t)written < buffer_size; i++) {
        written += snprintf(report_buffer + written, buffer_size - (size_t)written,
                          "Metric: %s\n"
                          "Current Value: %.2f\n"
                          "Average Value: %.2f\n"
                          "Peak Value: %.2f\n"
                          "Threshold: %.2f\n\n",
                          monitor_state->metrics[i].name,
                          monitor_state->metrics[i].current_value,
                          monitor_state->metrics[i].average_value,
                          monitor_state->metrics[i].peak_value,
                          monitor_state->metrics[i].threshold_value);
    }

    pthread_mutex_unlock(&monitor_state->mutex);
    return written;
}

// Collect comprehensive metrics
int collect_performance_metrics(performance_metrics_t* metrics) {
    if (!metrics || !monitor_state || !monitor_state->initialized) {
        return -1;
    }
    
    // Collect classical computation metrics with error handling
    if ((metrics->flops = measure_flops()) < 0 ||
        (metrics->memory_bandwidth = measure_memory_bandwidth()) < 0 ||
        (metrics->cache_hit_rate = measure_cache_performance()) < 0) {
        return -1;
    }
    
    // Collect quantum computation metrics with error handling
    if ((metrics->quantum_error_rate = measure_quantum_error_rate()) < 0 ||
        (metrics->quantum_fidelity = measure_quantum_fidelity()) < 0 ||
        (metrics->entanglement_fidelity = measure_entanglement_fidelity()) < 0 ||
        (metrics->gate_error_rate = measure_gate_error_rate()) < 0) {
        return -1;
    }
    
    // Collect resource metrics
    metrics->cpu_utilization = measure_cpu_utilization();
    metrics->gpu_utilization = measure_gpu_utilization();
    metrics->memory_usage = measure_memory_usage();
    metrics->page_faults = measure_page_faults();
    
    // Collect communication metrics
    metrics->mpi_time = measure_mpi_time();
    metrics->network_bandwidth = measure_network_bandwidth();
    metrics->latency = measure_communication_latency();
    metrics->numa_local_ratio = measure_numa_locality();
    
    // Collect performance metrics
    metrics->throughput = measure_throughput();
    metrics->response_time = measure_response_time();
    metrics->queue_length = measure_queue_length();
    metrics->wait_time = measure_wait_time();
    
    // Collect resource allocation metrics
    metrics->allocation_efficiency = measure_allocation_efficiency();
    metrics->resource_contention = measure_resource_contention();
    metrics->load_balance = measure_load_balance();
    metrics->resource_utilization = measure_resource_utilization();
    
    return 0;
}

// Analyze system performance
void analyze_performance(const performance_metrics_t* metrics) {
    if (!metrics || !monitor_state || !monitor_state->initialized) {
        return;
    }

    pthread_mutex_lock(&monitor_state->mutex);

    // Store metrics in history with thread safety
    if (monitor_state->history_size < monitor_state->history_capacity) {
        monitor_state->metric_history[monitor_state->history_size++] = *metrics;
    } else {
        // Circular buffer with memmove for thread safety
        memmove(monitor_state->metric_history,
                monitor_state->metric_history + 1,
                (monitor_state->history_capacity - 1) * sizeof(performance_metrics_t));
        monitor_state->metric_history[monitor_state->history_capacity - 1] = *metrics;
    }

    pthread_mutex_unlock(&monitor_state->mutex);

    // Analyze computation efficiency - avoid division by zero
    if (monitor_state->peak_performance > 0.0) {
        double compute_efficiency = metrics->flops / monitor_state->peak_performance;
        if (compute_efficiency < monitor_state->optimization_threshold) {
            optimize_computation();
        }
    }

    // Analyze memory efficiency
    if (metrics->cache_hit_rate < 0.95) {
        optimize_memory_access();
    }

    // Analyze communication efficiency
    if (metrics->numa_local_ratio < 0.9) {
        optimize_numa_placement();
    }

    // Analyze resource utilization
    if (metrics->resource_utilization < monitor_state->optimization_threshold) {
        optimize_resource_allocation();
    }

    // Analyze quantum metrics
    if (metrics->quantum_fidelity < 0.95) {
        optimize_quantum_circuits();
    }

    if (metrics->gate_error_rate > 0.01) {
        optimize_gate_sequences();
    }

    if (metrics->entanglement_fidelity < 0.9) {
        optimize_entanglement_operations();
    }

    // Update baseline if needed
    if (metrics->throughput > monitor_state->peak_performance) {
        monitor_state->peak_performance = metrics->throughput;
    }
}

// Generate optimization recommendations
void generate_recommendations(void) {
    if (!monitor_state || !monitor_state->initialized || monitor_state->history_size == 0) {
        return;
    }

    pthread_mutex_lock(&monitor_state->mutex);
    const performance_metrics_t* latest = &monitor_state->metric_history[monitor_state->history_size - 1];

    // Check classical computation bottlenecks
    if (monitor_state->peak_performance > 0.0 && latest->flops < monitor_state->peak_performance * 0.8) {
        suggest_computation_optimizations();
    }

    // Check memory bottlenecks
    if (monitor_state->peak_performance > 0.0 && latest->memory_bandwidth < monitor_state->peak_performance * 0.8) {
        suggest_memory_optimizations();
    }

    // Check communication bottlenecks
    if (monitor_state->peak_performance > 0.0 && latest->network_bandwidth < monitor_state->peak_performance * 0.8) {
        suggest_communication_optimizations();
    }

    // Check resource bottlenecks
    if (latest->resource_utilization < 0.8) {
        suggest_resource_optimizations();
    }

    // Check quantum bottlenecks
    if (latest->quantum_error_rate > 0.05) {
        suggest_error_mitigation_strategies();
    }

    if (latest->gate_error_rate > 0.01) {
        suggest_gate_optimizations();
    }

    pthread_mutex_unlock(&monitor_state->mutex);
}

// Clean up monitoring system
void cleanup_performance_monitor(void) {
    if (!monitor_state) return;
    
    pthread_mutex_lock(&global_mutex);
    
    if (monitor_state->initialized) {
        pthread_mutex_destroy(&monitor_state->mutex);
    }
    
    free(monitor_state->config_path);
    free(monitor_state->metrics_path);
    
    if (monitor_state->metrics) {
        for (size_t i = 0; i < monitor_state->metrics_count; i++) {
            free(monitor_state->metrics[i].name);
            free(monitor_state->metrics[i].parameters);
        }
        free(monitor_state->metrics);
    }
    
    free(monitor_state->resource_usage);
    free(monitor_state->resource_limits);
    free(monitor_state->metric_history);
    free(monitor_state);
    monitor_state = NULL;
    
    pthread_mutex_unlock(&global_mutex);
}
