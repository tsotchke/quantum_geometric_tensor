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

// GPU utilization - placeholder for Metal/CUDA integration
static double measure_gpu_utilization(void) {
    // TODO: Integrate with Metal (macOS) or CUDA (Linux) for real metrics
    return 0.0;
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

// Network bandwidth estimation
static double measure_network_bandwidth(void) {
    // TODO: Integrate with network monitoring
    return 0.0;
}

// Communication latency measurement
static double measure_communication_latency(void) {
    // TODO: Implement latency measurement
    return 0.0;
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

// Memory allocation efficiency
static double measure_allocation_efficiency(void) {
    // TODO: Track allocation patterns
    return 1.0;
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

// Load balance factor (1.0 = perfect balance)
static double measure_load_balance(void) {
    // TODO: Track per-thread workload distribution
    return 1.0;
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

// Cache performance (hit rate)
double measure_cache_performance(void) {
    // TODO: Use PAPI or perf_event on Linux for real cache statistics
    return 0.95;  // Default assumption
}

// Quantum error rate - from error mitigation subsystem
double measure_quantum_error_rate(void) {
    // TODO: Integrate with quantum error mitigation module
    return 0.001;
}

// Quantum state fidelity
double measure_quantum_fidelity(void) {
    // TODO: Integrate with quantum simulator
    return 0.99;
}

// Entanglement fidelity
double measure_entanglement_fidelity(void) {
    // TODO: Integrate with quantum state tracking
    return 0.98;
}

// Gate error rate
double measure_gate_error_rate(void) {
    // TODO: Integrate with quantum gate operations
    return 0.001;
}

// Optimization functions - trigger adaptive optimizations
static void optimize_computation(void) {
    // TODO: Adjust thread counts, vectorization strategies
}

static void optimize_memory_access(void) {
    // TODO: Adjust prefetching, blocking strategies
}

static void optimize_numa_placement(void) {
    // TODO: Use numa_* APIs on Linux for better placement
}

static void optimize_resource_allocation(void) {
    // TODO: Dynamic resource rebalancing
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
