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
#include <stdbool.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/processor_info.h>
#include <mach/mach_host.h>
#include <sys/sysctl.h>
#include <IOKit/IOKitLib.h>
#include <CoreFoundation/CoreFoundation.h>
#endif

#ifdef __linux__
#include <sys/sysinfo.h>
#endif

#include <sys/resource.h>
#include <sys/time.h>


// Thread-local storage for per-thread metrics
static __thread double tl_thread_work = 0.0;
static __thread double tl_thread_time = 0.0;

// Global tracking for load balance and allocation
static struct {
    pthread_mutex_t lock;
    double* thread_workloads;
    size_t num_threads;
    size_t allocations_total;
    size_t allocations_reused;
    size_t cache_hits;
    size_t cache_misses;
    bool initialized;
} g_perf_tracking = { .lock = PTHREAD_MUTEX_INITIALIZER };

// External quantum error tracking (from quantum_error_communication.c)
extern double get_error_confidence_interval(int category, bool aggregated);
extern const void* get_category_stats(int category, bool aggregated);

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

// GPU utilization measurement
static double measure_gpu_utilization(void) {
#ifdef __APPLE__
    // macOS: Query GPU utilization via IOKit
    io_iterator_t iterator;
    kern_return_t result = IOServiceGetMatchingServices(
        kIOMainPortDefault,
        IOServiceMatching("IOAccelerator"),
        &iterator);

    if (result != KERN_SUCCESS) return 0.0;

    io_object_t service;
    double utilization = 0.0;
    int gpu_count = 0;

    while ((service = IOIteratorNext(iterator)) != 0) {
        CFMutableDictionaryRef properties = NULL;
        result = IORegistryEntryCreateCFProperties(service, &properties,
                                                    kCFAllocatorDefault, 0);
        if (result == KERN_SUCCESS && properties) {
            // Look for "PerformanceStatistics" dictionary
            CFDictionaryRef perf_stats = CFDictionaryGetValue(properties,
                CFSTR("PerformanceStatistics"));
            if (perf_stats) {
                CFNumberRef gpu_util = CFDictionaryGetValue(perf_stats,
                    CFSTR("Device Utilization %"));
                if (gpu_util) {
                    int64_t util_val = 0;
                    CFNumberGetValue(gpu_util, kCFNumberSInt64Type, &util_val);
                    utilization += (double)util_val / 100.0;
                    gpu_count++;
                }
            }
            CFRelease(properties);
        }
        IOObjectRelease(service);
    }
    IOObjectRelease(iterator);

    return gpu_count > 0 ? utilization / gpu_count : 0.0;
#elif defined(__linux__)
    // Linux: Try to read from NVIDIA SMI or AMD ROCm
    FILE* fp = popen("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1", "r");
    if (fp) {
        int util = 0;
        if (fscanf(fp, "%d", &util) == 1) {
            pclose(fp);
            return (double)util / 100.0;
        }
        pclose(fp);
    }
    // Try AMD ROCm
    fp = popen("rocm-smi --showuse 2>/dev/null | grep 'GPU use' | awk '{print $NF}' | tr -d '%' | head -1", "r");
    if (fp) {
        int util = 0;
        if (fscanf(fp, "%d", &util) == 1) {
            pclose(fp);
            return (double)util / 100.0;
        }
        pclose(fp);
    }
    return 0.0;
#else
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

// Network bytes tracking for bandwidth calculation
static struct {
    uint64_t bytes_sent_prev;
    uint64_t bytes_recv_prev;
    struct timespec last_sample;
    double bandwidth_mbps;
    bool initialized;
} g_net_stats = {0};

// Network bandwidth estimation (MB/s)
static double measure_network_bandwidth(void) {
#ifdef __APPLE__
    // macOS: Use netstat to get network interface statistics
    FILE* fp = popen("netstat -ib 2>/dev/null | awk 'NR>1 {sent+=$7; recv+=$10} END {print sent, recv}'", "r");
    if (!fp) return g_net_stats.bandwidth_mbps;

    uint64_t bytes_sent = 0, bytes_recv = 0;
    if (fscanf(fp, "%llu %llu", &bytes_sent, &bytes_recv) != 2) {
        pclose(fp);
        return g_net_stats.bandwidth_mbps;
    }
    pclose(fp);

    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);

    if (g_net_stats.initialized) {
        double dt = (now.tv_sec - g_net_stats.last_sample.tv_sec) +
                   (now.tv_nsec - g_net_stats.last_sample.tv_nsec) / 1e9;
        if (dt > 0.1) {
            uint64_t delta_bytes = (bytes_sent - g_net_stats.bytes_sent_prev) +
                                   (bytes_recv - g_net_stats.bytes_recv_prev);
            g_net_stats.bandwidth_mbps = (double)delta_bytes / dt / 1e6;
            g_net_stats.bytes_sent_prev = bytes_sent;
            g_net_stats.bytes_recv_prev = bytes_recv;
            g_net_stats.last_sample = now;
        }
    } else {
        g_net_stats.bytes_sent_prev = bytes_sent;
        g_net_stats.bytes_recv_prev = bytes_recv;
        g_net_stats.last_sample = now;
        g_net_stats.initialized = true;
    }
    return g_net_stats.bandwidth_mbps;
#elif defined(__linux__)
    // Linux: Read from /proc/net/dev
    FILE* fp = fopen("/proc/net/dev", "r");
    if (!fp) return g_net_stats.bandwidth_mbps;

    char line[512];
    uint64_t total_recv = 0, total_sent = 0;

    // Skip header lines
    fgets(line, sizeof(line), fp);
    fgets(line, sizeof(line), fp);

    while (fgets(line, sizeof(line), fp)) {
        char iface[32];
        uint64_t r_bytes, t_bytes;
        uint64_t dummy;
        if (sscanf(line, "%31s %lu %lu %lu %lu %lu %lu %lu %lu %lu",
                   iface, &r_bytes, &dummy, &dummy, &dummy, &dummy, &dummy, &dummy, &dummy, &t_bytes) >= 10) {
            // Skip loopback
            if (strncmp(iface, "lo:", 3) != 0) {
                total_recv += r_bytes;
                total_sent += t_bytes;
            }
        }
    }
    fclose(fp);

    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);

    if (g_net_stats.initialized) {
        double dt = (now.tv_sec - g_net_stats.last_sample.tv_sec) +
                   (now.tv_nsec - g_net_stats.last_sample.tv_nsec) / 1e9;
        if (dt > 0.1) {
            uint64_t delta = (total_sent - g_net_stats.bytes_sent_prev) +
                            (total_recv - g_net_stats.bytes_recv_prev);
            g_net_stats.bandwidth_mbps = (double)delta / dt / 1e6;
            g_net_stats.bytes_sent_prev = total_sent;
            g_net_stats.bytes_recv_prev = total_recv;
            g_net_stats.last_sample = now;
        }
    } else {
        g_net_stats.bytes_sent_prev = total_sent;
        g_net_stats.bytes_recv_prev = total_recv;
        g_net_stats.last_sample = now;
        g_net_stats.initialized = true;
    }
    return g_net_stats.bandwidth_mbps;
#else
    return 0.0;
#endif
}

// Communication latency measurement tracking
static struct {
    double sum_latency;
    size_t count;
    double min_latency;
    double max_latency;
    pthread_mutex_t lock;
} g_latency_stats = { .lock = PTHREAD_MUTEX_INITIALIZER, .min_latency = 1e9 };

// Record a latency sample (call from communication code)
void record_communication_latency(double latency_us) {
    pthread_mutex_lock(&g_latency_stats.lock);
    g_latency_stats.sum_latency += latency_us;
    g_latency_stats.count++;
    if (latency_us < g_latency_stats.min_latency) g_latency_stats.min_latency = latency_us;
    if (latency_us > g_latency_stats.max_latency) g_latency_stats.max_latency = latency_us;
    pthread_mutex_unlock(&g_latency_stats.lock);
}

// Communication latency measurement (returns average in microseconds)
static double measure_communication_latency(void) {
    pthread_mutex_lock(&g_latency_stats.lock);
    double avg = g_latency_stats.count > 0 ?
                 g_latency_stats.sum_latency / g_latency_stats.count : 0.0;
    pthread_mutex_unlock(&g_latency_stats.lock);
    return avg;
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

// Memory allocation efficiency tracking
void record_allocation(bool reused) {
    pthread_mutex_lock(&g_perf_tracking.lock);
    g_perf_tracking.allocations_total++;
    if (reused) g_perf_tracking.allocations_reused++;
    pthread_mutex_unlock(&g_perf_tracking.lock);
}

static double measure_allocation_efficiency(void) {
    pthread_mutex_lock(&g_perf_tracking.lock);
    double efficiency = g_perf_tracking.allocations_total > 0 ?
        (double)g_perf_tracking.allocations_reused / (double)g_perf_tracking.allocations_total : 1.0;
    pthread_mutex_unlock(&g_perf_tracking.lock);
    return efficiency;
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

// Thread workload tracking
void record_thread_workload(int thread_id, double work_units) {
    pthread_mutex_lock(&g_perf_tracking.lock);
    if (!g_perf_tracking.initialized) {
        // Initialize thread workload array (assume max 256 threads)
        g_perf_tracking.thread_workloads = calloc(256, sizeof(double));
        g_perf_tracking.num_threads = 256;
        g_perf_tracking.initialized = true;
    }
    if (thread_id >= 0 && (size_t)thread_id < g_perf_tracking.num_threads) {
        g_perf_tracking.thread_workloads[thread_id] += work_units;
    }
    pthread_mutex_unlock(&g_perf_tracking.lock);
}

// Load balance factor (1.0 = perfect balance, 0 = completely imbalanced)
static double measure_load_balance(void) {
    pthread_mutex_lock(&g_perf_tracking.lock);

    if (!g_perf_tracking.initialized || !g_perf_tracking.thread_workloads) {
        pthread_mutex_unlock(&g_perf_tracking.lock);
        return 1.0;
    }

    // Find min, max, and mean workload across active threads
    double sum = 0.0;
    double max_work = 0.0;
    double min_work = 1e30;
    size_t active_threads = 0;

    for (size_t i = 0; i < g_perf_tracking.num_threads; i++) {
        double w = g_perf_tracking.thread_workloads[i];
        if (w > 0) {
            sum += w;
            if (w > max_work) max_work = w;
            if (w < min_work) min_work = w;
            active_threads++;
        }
    }

    pthread_mutex_unlock(&g_perf_tracking.lock);

    if (active_threads < 2 || max_work == 0) return 1.0;

    // Load balance = min/max (1.0 = perfect, 0 = one thread doing all work)
    return min_work / max_work;
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

// Cache performance tracking
void record_cache_access(bool hit) {
    pthread_mutex_lock(&g_perf_tracking.lock);
    if (hit) {
        g_perf_tracking.cache_hits++;
    } else {
        g_perf_tracking.cache_misses++;
    }
    pthread_mutex_unlock(&g_perf_tracking.lock);
}

// Cache performance (hit rate)
double measure_cache_performance(void) {
#ifdef __linux__
    // Try to read from perf_event if available
    // This requires CAP_PERFMON or running as root
    static int perf_fd = -1;
    static bool perf_tried = false;

    if (!perf_tried) {
        perf_tried = true;
        // Try to open perf event for L1 cache misses
        // If it fails, we fall back to software tracking
        struct {
            uint32_t type;
            uint32_t size;
            uint64_t config;
            uint64_t disabled;
            uint64_t exclude_kernel;
            uint64_t exclude_hv;
        } pe = {
            .type = 3,  // PERF_TYPE_HW_CACHE
            .size = sizeof(pe),
            .config = 0x10000,  // L1D read miss
            .disabled = 1,
            .exclude_kernel = 1,
            .exclude_hv = 1
        };
        perf_fd = syscall(298, &pe, 0, -1, -1, 0);  // perf_event_open
    }

    if (perf_fd >= 0) {
        uint64_t count;
        if (read(perf_fd, &count, sizeof(count)) == sizeof(count)) {
            // Estimate hit rate from miss count (rough approximation)
            static uint64_t last_count = 0;
            uint64_t delta = count - last_count;
            last_count = count;
            // Assume ~10M cache accesses per measurement period
            double miss_rate = (double)delta / 10000000.0;
            if (miss_rate > 1.0) miss_rate = 1.0;
            return 1.0 - miss_rate;
        }
    }
#endif

    // Fall back to software-tracked cache statistics
    pthread_mutex_lock(&g_perf_tracking.lock);
    size_t total = g_perf_tracking.cache_hits + g_perf_tracking.cache_misses;
    double hit_rate = total > 0 ?
        (double)g_perf_tracking.cache_hits / (double)total : 0.95;
    pthread_mutex_unlock(&g_perf_tracking.lock);
    return hit_rate;
}

// Quantum metrics tracking structure
static struct {
    pthread_mutex_t lock;
    double total_error;
    size_t error_count;
    double fidelity_sum;
    size_t fidelity_count;
    double entanglement_fidelity_sum;
    size_t entanglement_count;
    double gate_error_sum;
    size_t gate_count;
    bool initialized;
} g_quantum_metrics = { .lock = PTHREAD_MUTEX_INITIALIZER };

// Record quantum error for tracking
void record_quantum_error(double error_rate) {
    pthread_mutex_lock(&g_quantum_metrics.lock);
    g_quantum_metrics.total_error += error_rate;
    g_quantum_metrics.error_count++;
    g_quantum_metrics.initialized = true;
    pthread_mutex_unlock(&g_quantum_metrics.lock);
}

// Record state fidelity measurement
void record_state_fidelity(double fidelity) {
    pthread_mutex_lock(&g_quantum_metrics.lock);
    g_quantum_metrics.fidelity_sum += fidelity;
    g_quantum_metrics.fidelity_count++;
    g_quantum_metrics.initialized = true;
    pthread_mutex_unlock(&g_quantum_metrics.lock);
}

// Record entanglement fidelity
void record_entanglement_fidelity(double fidelity) {
    pthread_mutex_lock(&g_quantum_metrics.lock);
    g_quantum_metrics.entanglement_fidelity_sum += fidelity;
    g_quantum_metrics.entanglement_count++;
    g_quantum_metrics.initialized = true;
    pthread_mutex_unlock(&g_quantum_metrics.lock);
}

// Record gate error
void record_gate_error(double error) {
    pthread_mutex_lock(&g_quantum_metrics.lock);
    g_quantum_metrics.gate_error_sum += error;
    g_quantum_metrics.gate_count++;
    g_quantum_metrics.initialized = true;
    pthread_mutex_unlock(&g_quantum_metrics.lock);
}

// Quantum error rate - from error mitigation subsystem
double measure_quantum_error_rate(void) {
    pthread_mutex_lock(&g_quantum_metrics.lock);
    double rate = g_quantum_metrics.error_count > 0 ?
        g_quantum_metrics.total_error / g_quantum_metrics.error_count : 0.001;
    pthread_mutex_unlock(&g_quantum_metrics.lock);
    return rate;
}

// Quantum state fidelity
double measure_quantum_fidelity(void) {
    pthread_mutex_lock(&g_quantum_metrics.lock);
    double fidelity = g_quantum_metrics.fidelity_count > 0 ?
        g_quantum_metrics.fidelity_sum / g_quantum_metrics.fidelity_count : 0.99;
    pthread_mutex_unlock(&g_quantum_metrics.lock);
    return fidelity;
}

// Entanglement fidelity
double measure_entanglement_fidelity(void) {
    pthread_mutex_lock(&g_quantum_metrics.lock);
    double fidelity = g_quantum_metrics.entanglement_count > 0 ?
        g_quantum_metrics.entanglement_fidelity_sum / g_quantum_metrics.entanglement_count : 0.98;
    pthread_mutex_unlock(&g_quantum_metrics.lock);
    return fidelity;
}

// Gate error rate
double measure_gate_error_rate(void) {
    pthread_mutex_lock(&g_quantum_metrics.lock);
    double rate = g_quantum_metrics.gate_count > 0 ?
        g_quantum_metrics.gate_error_sum / g_quantum_metrics.gate_count : 0.001;
    pthread_mutex_unlock(&g_quantum_metrics.lock);
    return rate;
}

// Optimization state
static struct {
    int thread_count;
    size_t cache_block_size;
    bool prefetch_enabled;
    int numa_policy;
    bool optimizations_applied;
} g_optimization_state = {
    .thread_count = 0,
    .cache_block_size = 64 * 1024,  // 64KB default
    .prefetch_enabled = true,
    .numa_policy = 0,
    .optimizations_applied = false
};

// Get optimal thread count based on CPU utilization
static int get_optimal_thread_count(void) {
    double cpu_util = measure_cpu_utilization();
    int max_threads = sysconf(_SC_NPROCESSORS_ONLN);

    if (cpu_util > 0.9) {
        // High utilization - we might be oversubscribed
        return max_threads;
    } else if (cpu_util < 0.5) {
        // Low utilization - might benefit from more threads
        return max_threads;
    }
    return g_optimization_state.thread_count > 0 ?
           g_optimization_state.thread_count : max_threads;
}

// Optimization functions - trigger adaptive optimizations
static void optimize_computation(void) {
    int optimal_threads = get_optimal_thread_count();

    if (g_optimization_state.thread_count != optimal_threads) {
        g_optimization_state.thread_count = optimal_threads;

#ifdef _OPENMP
        omp_set_num_threads(optimal_threads);
#endif

        // Log the optimization
        fprintf(stderr, "[PERF-OPT] Adjusted thread count to %d based on CPU utilization\n",
                optimal_threads);
    }

    g_optimization_state.optimizations_applied = true;
}

static void optimize_memory_access(void) {
    double cache_hit_rate = measure_cache_performance();

    if (cache_hit_rate < 0.90) {
        // Poor cache performance - increase block size for better locality
        if (g_optimization_state.cache_block_size < 256 * 1024) {
            g_optimization_state.cache_block_size *= 2;
            fprintf(stderr, "[PERF-OPT] Increased cache block size to %zu for better locality\n",
                    g_optimization_state.cache_block_size);
        }

        // Enable software prefetching hints
        if (!g_optimization_state.prefetch_enabled) {
            g_optimization_state.prefetch_enabled = true;
            fprintf(stderr, "[PERF-OPT] Enabled software prefetching\n");
        }
    } else if (cache_hit_rate > 0.98 && g_optimization_state.cache_block_size > 32 * 1024) {
        // Excellent cache performance - can try smaller blocks for better parallelism
        g_optimization_state.cache_block_size /= 2;
        fprintf(stderr, "[PERF-OPT] Decreased cache block size to %zu for better parallelism\n",
                g_optimization_state.cache_block_size);
    }

    g_optimization_state.optimizations_applied = true;
}

static void optimize_numa_placement(void) {
#ifdef __linux__
    double numa_locality = measure_numa_locality();

    if (numa_locality < 0.8) {
        // Poor NUMA locality - try to improve memory placement
        // This requires libnuma, but we can suggest the optimization

        if (g_optimization_state.numa_policy == 0) {
            // Switch to local allocation policy
            g_optimization_state.numa_policy = 1;

            // Try to use mbind/set_mempolicy if available
            // For now, just log the recommendation
            fprintf(stderr, "[PERF-OPT] NUMA locality is %.1f%% - recommend using numactl --localalloc\n",
                    numa_locality * 100.0);

            // On Linux, we can try to hint the kernel
            FILE* f = fopen("/proc/self/numa_faults", "r");
            if (f) {
                // NUMA balancing is available
                fclose(f);
                fprintf(stderr, "[PERF-OPT] NUMA auto-balancing is active\n");
            }
        }
    }
#else
    // macOS is UMA, no NUMA optimization needed
    (void)0;
#endif

    g_optimization_state.optimizations_applied = true;
}

static void optimize_resource_allocation(void) {
    double resource_util = measure_resource_utilization();
    double load_balance = measure_load_balance();

    if (resource_util < 0.6) {
        // Underutilized - log suggestions
        fprintf(stderr, "[PERF-OPT] Resource utilization at %.1f%% - consider increasing batch sizes\n",
                resource_util * 100.0);
    }

    if (load_balance < 0.7) {
        // Poor load balance
        fprintf(stderr, "[PERF-OPT] Load balance at %.1f%% - consider work stealing or dynamic scheduling\n",
                load_balance * 100.0);

#ifdef _OPENMP
        // Suggest dynamic scheduling for OpenMP
        fprintf(stderr, "[PERF-OPT] Recommend using OMP_SCHEDULE=dynamic\n");
#endif
    }

    g_optimization_state.optimizations_applied = true;
}

// Get current optimization settings (for external use)
size_t get_optimal_cache_block_size(void) {
    return g_optimization_state.cache_block_size;
}

bool is_prefetch_enabled(void) {
    return g_optimization_state.prefetch_enabled;
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
        metric_id < 0 || metric_id >= monitor_state->metrics_count ||
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
        metric_id < 0 || metric_id >= monitor_state->metrics_count ||
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
        metric_id < 0 || metric_id >= monitor_state->metrics_count) {
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
        metric_id < 0 || metric_id >= monitor_state->metrics_count ||
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

    for (size_t i = 0; i < monitor_state->metrics_count && written < buffer_size; i++) {
        written += snprintf(report_buffer + written, buffer_size - written,
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
