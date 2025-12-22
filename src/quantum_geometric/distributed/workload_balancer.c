#include "quantum_geometric/distributed/workload_balancer.h"
#include "quantum_geometric/distributed/communication_optimizer.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef NO_HWLOC
#include <hwloc.h>
#endif

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <mach/mach.h>
#endif

// ============================================================================
// Internal Constants
// ============================================================================

#define MAX_DEVICES 64

// ============================================================================
// Internal Type Definitions
// ============================================================================

// Device capabilities (internal representation)
typedef struct {
    int type;                    // Device type (uses BalancerDeviceType values)
    size_t memory;               // Total memory in bytes
    double compute_power;        // Relative compute power (1.0 = baseline)
    double bandwidth;            // Memory bandwidth in GB/s
    bool is_available;           // Device is available for work
    DeviceStats stats;           // Performance statistics
} DeviceCapabilities;

// Workload chunk (internal representation)
typedef struct {
    void* data;
    size_t size;
    int owner;
    int priority;
    WorkloadType type;
} WorkloadChunk;

// Load balancer internal structure
struct LoadBalancer {
#ifndef NO_HWLOC
    hwloc_topology_t topology;
#endif
    DeviceCapabilities* devices;
    int num_devices;

    // Workload management
    WorkloadChunk** chunks;
    size_t num_chunks;
    size_t chunk_capacity;
    double* load_factors;

    // Performance monitoring
    PerformanceMonitor* monitor;
    bool needs_rebalancing;

    // Communication
    CommunicationOptimizer* comm_optimizer;
};

// ============================================================================
// Forward Declarations (internal functions)
// ============================================================================

static DeviceCapabilities* detect_devices_internal(int* num_devices);
static int select_optimal_device(LoadBalancer* balancer, const WorkloadChunk* chunk);
static double compute_device_score(const DeviceCapabilities* device,
                                   const WorkloadChunk* chunk, double load_factor);
static void assign_chunk_to_device(LoadBalancer* balancer, WorkloadChunk* chunk, int device);
static void update_load_factors_internal(LoadBalancer* balancer, int device,
                                         const WorkloadChunk* chunk);

// ============================================================================
// Helper Macros
// ============================================================================

#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif

// ============================================================================
// Performance Monitor Implementation
// ============================================================================

PerformanceMonitor* init_performance_monitor(size_t num_devices) {
    PerformanceMonitor* monitor = calloc(1, sizeof(PerformanceMonitor));
    if (!monitor) return NULL;

    monitor->device_stats = calloc(num_devices, sizeof(DeviceStats));
    if (!monitor->device_stats) {
        free(monitor);
        return NULL;
    }

    monitor->num_devices = num_devices;
    monitor->total_utilization = 0.0;
    monitor->efficiency = 1.0;
    monitor->is_monitoring = true;

    return monitor;
}

void cleanup_performance_monitor(PerformanceMonitor* monitor) {
    if (!monitor) return;
    free(monitor->device_stats);
    free(monitor);
}

// ============================================================================
// Device Detection Implementation
// ============================================================================

void detect_cpu_capabilities(void* devices_ptr, int* count) {
    DeviceCapabilities* devices = (DeviceCapabilities*)devices_ptr;
    if (!devices || !count) return;

    int idx = *count;
    if (idx >= MAX_DEVICES) return;

    devices[idx].type = DEVICE_CPU;
    devices[idx].is_available = true;
    devices[idx].compute_power = 1.0;

#ifdef __APPLE__
    // Get CPU info on macOS
    size_t mem_size = sizeof(size_t);
    size_t memory = 0;
    sysctlbyname("hw.memsize", &memory, &mem_size, NULL, 0);
    devices[idx].memory = memory;

    // Estimate bandwidth (rough approximation)
    devices[idx].bandwidth = 50.0;  // ~50 GB/s for modern DDR4/DDR5
#else
    // Linux fallback
    devices[idx].memory = 8UL * 1024 * 1024 * 1024;  // Default 8GB
    devices[idx].bandwidth = 40.0;
#endif

    // Initialize stats
    memset(&devices[idx].stats, 0, sizeof(DeviceStats));
    devices[idx].stats.memory_total = (double)devices[idx].memory;

    (*count)++;
}

void detect_gpu_capabilities(void* devices_ptr, int* count) {
    // GPU detection is platform-specific and requires CUDA/Metal/OpenCL
    // For now, check environment for GPU presence indicators
    DeviceCapabilities* devices = (DeviceCapabilities*)devices_ptr;
    if (!devices || !count) return;

#ifdef QGTL_HAS_CUDA
    // Would enumerate CUDA devices here
    int idx = *count;
    if (idx < MAX_DEVICES) {
        devices[idx].type = DEVICE_GPU;
        devices[idx].is_available = true;
        devices[idx].memory = 8UL * 1024 * 1024 * 1024;  // 8GB default
        devices[idx].compute_power = 10.0;  // GPUs are ~10x faster for parallel work
        devices[idx].bandwidth = 500.0;  // ~500 GB/s for modern GPUs
        memset(&devices[idx].stats, 0, sizeof(DeviceStats));
        (*count)++;
    }
#endif

#ifdef QGTL_HAS_METAL
    // Metal is available on this system (detected at cmake time)
    int idx = *count;
    if (idx < MAX_DEVICES) {
        devices[idx].type = DEVICE_GPU;
        devices[idx].is_available = true;
        devices[idx].memory = 16UL * 1024 * 1024 * 1024;  // Unified memory
        devices[idx].compute_power = 8.0;
        devices[idx].bandwidth = 400.0;
        memset(&devices[idx].stats, 0, sizeof(DeviceStats));
        (*count)++;
    }
#endif
}

void detect_quantum_capabilities(void* devices_ptr, int* count) {
    // Quantum device detection - check for backend connections
    // This would interface with quantum hardware backends when available
    (void)devices_ptr;
    (void)count;
    // Quantum devices are added dynamically when backends connect
}

void analyze_device_performance(void* device_ptr) {
    DeviceCapabilities* device = (DeviceCapabilities*)device_ptr;
    if (!device) return;

    // Initialize performance metrics
    device->stats.utilization = 0.0;
    device->stats.memory_used = 0.0;
    device->stats.memory_total = (double)device->memory;
    device->stats.compute_time_ms = 0.0;
    device->stats.queue_depth = 0.0;
    device->stats.throughput = 0.0;
    device->stats.tasks_completed = 0;
    device->stats.errors = 0;
}

void cleanup_device(void* device_ptr) {
    // Device cleanup - release any device-specific resources
    (void)device_ptr;
    // No dynamic allocations per device currently
}

// ============================================================================
// Device Statistics Implementation
// ============================================================================

void update_device_stats(DeviceStats* stats) {
    if (!stats) return;

    // Update utilization based on recent activity
    // In production, this would query actual device metrics
    if (stats->tasks_completed > 0) {
        stats->throughput = (double)stats->tasks_completed /
                           (stats->compute_time_ms / 1000.0 + 0.001);
    }
}

bool detect_performance_degradation(const DeviceStats* stats) {
    if (!stats) return false;

    // Detect potential issues
    if (stats->utilization > 0.95) return true;  // Overloaded
    if (stats->errors > 10) return true;          // Too many errors
    if (stats->queue_depth > 100) return true;    // Queue backup

    return false;
}

// ============================================================================
// Chunk Operations Implementation
// ============================================================================

size_t calculate_chunk_count(const LoadBalancer* balancer, size_t size, WorkloadType type) {
    if (!balancer || size == 0) return 0;

    size_t min_chunk = MIN_CHUNK_SIZE;
    size_t num_devices = (size_t)balancer->num_devices;

    // Adjust based on workload type
    switch (type) {
        case WORKLOAD_QUANTUM:
            // Quantum workloads benefit from fewer, larger chunks
            min_chunk *= 4;
            break;
        case WORKLOAD_GPU:
            // GPU workloads prefer medium chunks
            min_chunk *= 2;
            break;
        default:
            break;
    }

    size_t max_chunks = size / min_chunk;
    if (max_chunks == 0) max_chunks = 1;

    // Target: 2-4 chunks per device for good load balancing
    size_t target_chunks = num_devices * 3;

    return min(max_chunks, max(target_chunks, num_devices));
}

void* create_chunk(void* data, size_t size, WorkloadType type) {
    WorkloadChunk* chunk = calloc(1, sizeof(WorkloadChunk));
    if (!chunk) return NULL;

    chunk->data = data;  // Reference, not copy
    chunk->size = size;
    chunk->owner = -1;   // Unassigned
    chunk->priority = 0;
    chunk->type = type;

    return chunk;
}

void cleanup_chunk(void* chunk_ptr) {
    WorkloadChunk* chunk = (WorkloadChunk*)chunk_ptr;
    if (!chunk) return;
    // Note: chunk->data is not owned, don't free it
    free(chunk);
}

// ============================================================================
// Load Balancing Implementation
// ============================================================================

bool should_rebalance(const LoadBalancer* balancer) {
    if (!balancer) return false;
    if (balancer->needs_rebalancing) return true;
    if (balancer->num_devices <= 1) return false;

    // Check load imbalance
    double max_load = 0.0;
    double min_load = 1.0;

    for (int i = 0; i < balancer->num_devices; i++) {
        double load = balancer->load_factors[i];
        max_load = max(max_load, load);
        min_load = min(min_load, load);
    }

    return (max_load - min_load) > REBALANCE_THRESHOLD;
}

void migrate_chunks_from_device(LoadBalancer* balancer, int device_idx) {
    if (!balancer || device_idx < 0 || device_idx >= balancer->num_devices) return;

    // Find underloaded device
    int target = -1;
    double min_load = balancer->load_factors[device_idx];

    for (int i = 0; i < balancer->num_devices; i++) {
        if (i != device_idx && balancer->load_factors[i] < min_load) {
            min_load = balancer->load_factors[i];
            target = i;
        }
    }

    if (target < 0) return;  // No suitable target

    // Migrate chunks (simplified - in production would do actual data movement)
    for (size_t i = 0; i < balancer->num_chunks; i++) {
        WorkloadChunk* chunk = balancer->chunks[i];
        if (chunk && chunk->owner == device_idx) {
            chunk->owner = target;
            balancer->load_factors[device_idx] -= 0.1;
            balancer->load_factors[target] += 0.1;
            break;  // Migrate one chunk at a time
        }
    }
}

// ============================================================================
// Internal Device Detection
// ============================================================================

static DeviceCapabilities* detect_devices_internal(int* num_devices) {
    *num_devices = 0;

    DeviceCapabilities* devices = calloc(MAX_DEVICES, sizeof(DeviceCapabilities));
    if (!devices) return NULL;

    detect_cpu_capabilities(devices, num_devices);
    detect_gpu_capabilities(devices, num_devices);
    detect_quantum_capabilities(devices, num_devices);

    for (int i = 0; i < *num_devices; i++) {
        analyze_device_performance(&devices[i]);
    }

    return devices;
}

// ============================================================================
// Device Selection
// ============================================================================

static int select_optimal_device(LoadBalancer* balancer, const WorkloadChunk* chunk) {
    if (!balancer || !chunk) return 0;

    double best_score = -1.0;
    int best_device = 0;

    for (int i = 0; i < balancer->num_devices; i++) {
        if (!balancer->devices[i].is_available) continue;

        double score = compute_device_score(&balancer->devices[i], chunk,
                                            balancer->load_factors[i]);
        if (score > best_score) {
            best_score = score;
            best_device = i;
        }
    }

    return best_device;
}

static double compute_device_score(const DeviceCapabilities* device,
                                   const WorkloadChunk* chunk,
                                   double load_factor) {
    if (!device || !chunk) return 0.0;

    double score = device->compute_power;
    score *= (1.0 - load_factor);

    // Match workload type to device type
    switch (chunk->type) {
        case WORKLOAD_QUANTUM:
            if (device->type == DEVICE_QUANTUM) score *= 2.0;
            break;
        case WORKLOAD_GPU:
            if (device->type == DEVICE_GPU) score *= 1.5;
            break;
        case WORKLOAD_CPU:
            if (device->type == DEVICE_CPU) score *= 1.2;
            break;
        default:
            break;
    }

    // Memory availability check
    if (device->memory > 0) {
        double mem_ratio = (double)chunk->size / (double)device->memory;
        if (mem_ratio > 0.8) {
            score *= (1.0 - mem_ratio);
        }
    }

    return score;
}

static void assign_chunk_to_device(LoadBalancer* balancer, WorkloadChunk* chunk, int device) {
    if (!balancer || !chunk) return;

    chunk->owner = device;

    // Add to chunk list
    if (balancer->num_chunks < balancer->chunk_capacity) {
        balancer->chunks[balancer->num_chunks++] = chunk;
    }
}

static void update_load_factors_internal(LoadBalancer* balancer, int device,
                                        const WorkloadChunk* chunk) {
    if (!balancer || !chunk || device < 0 || device >= balancer->num_devices) return;

    // Update load factor based on chunk size
    double load_increase = (double)chunk->size /
                          (double)balancer->devices[device].memory;
    balancer->load_factors[device] += load_increase;

    // Clamp to [0, 1]
    if (balancer->load_factors[device] > 1.0) {
        balancer->load_factors[device] = 1.0;
    }
}

// ============================================================================
// Public API Implementation
// ============================================================================

LoadBalancer* init_load_balancer(const BalancerConfig* config) {
    LoadBalancer* balancer = calloc(1, sizeof(LoadBalancer));
    if (!balancer) return NULL;

#ifndef NO_HWLOC
    if (hwloc_topology_init(&balancer->topology) < 0) {
        free(balancer);
        return NULL;
    }

    if (hwloc_topology_load(balancer->topology) < 0) {
        hwloc_topology_destroy(balancer->topology);
        free(balancer);
        return NULL;
    }
#endif

    balancer->devices = detect_devices_internal(&balancer->num_devices);
    if (!balancer->devices) {
#ifndef NO_HWLOC
        hwloc_topology_destroy(balancer->topology);
#endif
        free(balancer);
        return NULL;
    }

    balancer->monitor = init_performance_monitor((size_t)balancer->num_devices);

    balancer->chunk_capacity = MAX_CHUNKS;
    balancer->chunks = calloc(MAX_CHUNKS, sizeof(WorkloadChunk*));
    balancer->load_factors = calloc((size_t)balancer->num_devices, sizeof(double));
    balancer->num_chunks = 0;
    balancer->needs_rebalancing = false;

    if (config) {
        balancer->comm_optimizer = init_communication_optimizer(&config->comm_config);
    }

    return balancer;
}

void distribute_workload(LoadBalancer* balancer, void* data, size_t size,
                        WorkloadType type) {
    if (!balancer || !data || size == 0) return;

    if (should_rebalance(balancer)) {
        rebalance_workload(balancer);
    }

    size_t num_chunks = calculate_chunk_count(balancer, size, type);
    size_t chunk_size = size / num_chunks;

    for (size_t i = 0; i < num_chunks; i++) {
        void* chunk_data = (char*)data + (i * chunk_size);
        WorkloadChunk* chunk = create_chunk(chunk_data, chunk_size, type);
        if (!chunk) continue;

        int device = select_optimal_device(balancer, chunk);
        assign_chunk_to_device(balancer, chunk, device);
        update_load_factors_internal(balancer, device, chunk);
    }
}

void rebalance_workload(LoadBalancer* balancer) {
    if (!balancer) return;

    double max_load = 0.0;
    double min_load = 1.0;

    for (int i = 0; i < balancer->num_devices; i++) {
        double load = balancer->load_factors[i];
        max_load = max(max_load, load);
        min_load = min(min_load, load);
    }

    if (max_load - min_load > REBALANCE_THRESHOLD) {
        for (int src = 0; src < balancer->num_devices; src++) {
            if (balancer->load_factors[src] > max_load - REBALANCE_THRESHOLD) {
                migrate_chunks_from_device(balancer, src);
            }
        }
    }

    balancer->needs_rebalancing = false;
}

void monitor_performance(LoadBalancer* balancer) {
    if (!balancer) return;

    for (int i = 0; i < balancer->num_devices; i++) {
        update_device_stats(&balancer->devices[i].stats);
    }

    for (int i = 0; i < balancer->num_devices; i++) {
        if (detect_performance_degradation(&balancer->devices[i].stats)) {
            balancer->needs_rebalancing = true;
            break;
        }
    }
}

void cleanup_load_balancer(LoadBalancer* balancer) {
    if (!balancer) return;

#ifndef NO_HWLOC
    hwloc_topology_destroy(balancer->topology);
#endif

    if (balancer->devices) {
        for (int i = 0; i < balancer->num_devices; i++) {
            cleanup_device(&balancer->devices[i]);
        }
        free(balancer->devices);
    }

    if (balancer->chunks) {
        for (size_t i = 0; i < balancer->num_chunks; i++) {
            cleanup_chunk(balancer->chunks[i]);
        }
        free(balancer->chunks);
    }

    cleanup_performance_monitor(balancer->monitor);

    if (balancer->comm_optimizer) {
        cleanup_communication_optimizer(balancer->comm_optimizer);
    }

    free(balancer->load_factors);
    free(balancer);
}
