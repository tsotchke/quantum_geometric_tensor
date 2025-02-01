#include "quantum_geometric/distributed/workload_balancer.h"
#include "quantum_geometric/distributed/communication_optimizer.h"
#include "quantum_geometric/core/performance_operations.h"

#ifndef NO_HWLOC
#include <hwloc.h>
#endif

// Load balancing parameters
#define MAX_DEVICES 64
#define REBALANCE_THRESHOLD 0.2
#define MONITORING_INTERVAL 100
#define MIN_CHUNK_SIZE 1024

// Device capabilities
typedef struct {
    DeviceType type;
    size_t memory;
    double compute_power;
    double bandwidth;
    bool is_available;
    DeviceStats stats;
} DeviceCapabilities;

// Workload chunk
typedef struct {
    void* data;
    size_t size;
    int owner;
    int priority;
    WorkloadType type;
} WorkloadChunk;

// Load balancer
typedef struct {
#ifndef NO_HWLOC
    // Hardware topology
    hwloc_topology_t topology;
#endif
    DeviceCapabilities* devices;
    int num_devices;
    
    // Workload management
    WorkloadChunk** chunks;
    size_t num_chunks;
    double* load_factors;
    
    // Performance monitoring
    PerformanceMonitor* monitor;
    bool needs_rebalancing;
    
    // Communication
    CommunicationOptimizer* comm_optimizer;
} LoadBalancer;

// Initialize load balancer
LoadBalancer* init_load_balancer(
    const BalancerConfig* config) {
    
    LoadBalancer* balancer = aligned_alloc(64,
        sizeof(LoadBalancer));
    if (!balancer) return NULL;
    
#ifndef NO_HWLOC
    // Initialize topology
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
    
    // Detect devices
    balancer->devices = detect_devices(&balancer->num_devices);
    if (!balancer->devices) {
#ifndef NO_HWLOC
        hwloc_topology_destroy(balancer->topology);
#endif
        free(balancer);
        return NULL;
    }
    
    // Initialize performance monitoring
    balancer->monitor = init_performance_monitor(
        balancer->num_devices);
    
    // Initialize workload management
    balancer->chunks = aligned_alloc(64,
        MAX_CHUNKS * sizeof(WorkloadChunk*));
    balancer->load_factors = aligned_alloc(64,
        balancer->num_devices * sizeof(double));
    
    // Initialize communication
    balancer->comm_optimizer = init_communication_optimizer(
        &config->comm_config);
    
    return balancer;
}

// Detect and analyze available devices
static DeviceCapabilities* detect_devices(int* num_devices) {
    *num_devices = 0;
    
    // Allocate device array
    DeviceCapabilities* devices = aligned_alloc(64,
        MAX_DEVICES * sizeof(DeviceCapabilities));
    if (!devices) return NULL;
    
    // Detect CPUs
    detect_cpu_capabilities(devices, num_devices);
    
    // Detect GPUs
    detect_gpu_capabilities(devices, num_devices);
    
    // Detect quantum devices
    detect_quantum_capabilities(devices, num_devices);
    
    // Analyze device characteristics
    for (int i = 0; i < *num_devices; i++) {
        analyze_device_performance(&devices[i]);
    }
    
    return devices;
}

// Distribute workload
void distribute_workload(
    LoadBalancer* balancer,
    void* data,
    size_t size,
    WorkloadType type) {
    
    // Check if rebalancing needed
    if (should_rebalance(balancer)) {
        rebalance_workload(balancer);
    }
    
    // Create chunks
    size_t num_chunks = calculate_chunk_count(
        balancer, size, type);
    
    // Distribute chunks
    for (size_t i = 0; i < num_chunks; i++) {
        WorkloadChunk* chunk = create_chunk(data, size / num_chunks,
                                          type);
        
        // Assign to device
        int device = select_optimal_device(balancer, chunk);
        assign_chunk_to_device(balancer, chunk, device);
        
        // Update load factors
        update_load_factors(balancer, device, chunk);
    }
}

// Select optimal device for chunk
static int select_optimal_device(
    LoadBalancer* balancer,
    const WorkloadChunk* chunk) {
    
    double best_score = -1;
    int best_device = -1;
    
    // Score each device
    for (int i = 0; i < balancer->num_devices; i++) {
        if (!balancer->devices[i].is_available) continue;
        
        double score = compute_device_score(
            &balancer->devices[i],
            chunk,
            balancer->load_factors[i]);
        
        if (score > best_score) {
            best_score = score;
            best_device = i;
        }
    }
    
    return best_device;
}

// Compute device score for chunk
static double compute_device_score(
    const DeviceCapabilities* device,
    const WorkloadChunk* chunk,
    double load_factor) {
    
    // Base score on device capabilities
    double score = device->compute_power;
    
    // Adjust for current load
    score *= (1.0 - load_factor);
    
    // Adjust for workload type
    switch (chunk->type) {
        case QUANTUM_WORKLOAD:
            if (device->type == DEVICE_QUANTUM) {
                score *= 2.0;
            }
            break;
            
        case GPU_WORKLOAD:
            if (device->type == DEVICE_GPU) {
                score *= 1.5;
            }
            break;
            
        case CPU_WORKLOAD:
            if (device->type == DEVICE_CPU) {
                score *= 1.2;
            }
            break;
    }
    
    // Adjust for memory availability
    double mem_ratio = (double)chunk->size / device->memory;
    if (mem_ratio > 0.8) {
        score *= (1.0 - mem_ratio);
    }
    
    return score;
}

// Rebalance workload if needed
static void rebalance_workload(LoadBalancer* balancer) {
    // Compute load imbalance
    double max_load = 0;
    double min_load = 1.0;
    
    for (int i = 0; i < balancer->num_devices; i++) {
        double load = balancer->load_factors[i];
        max_load = max(max_load, load);
        min_load = min(min_load, load);
    }
    
    // Check if rebalancing needed
    if (max_load - min_load > REBALANCE_THRESHOLD) {
        // Find overloaded and underloaded devices
        for (int src = 0; src < balancer->num_devices; src++) {
            if (balancer->load_factors[src] > max_load - REBALANCE_THRESHOLD) {
                // Move chunks from overloaded device
                migrate_chunks_from_device(balancer, src);
            }
        }
    }
}

// Monitor device performance
void monitor_performance(LoadBalancer* balancer) {
    // Update device statistics
    for (int i = 0; i < balancer->num_devices; i++) {
        update_device_stats(&balancer->devices[i].stats);
    }
    
    // Check for performance degradation
    for (int i = 0; i < balancer->num_devices; i++) {
        if (detect_performance_degradation(
                &balancer->devices[i].stats)) {
            // Mark for rebalancing
            balancer->needs_rebalancing = true;
            break;
        }
    }
}

// Clean up
void cleanup_load_balancer(LoadBalancer* balancer) {
    if (!balancer) return;
    
#ifndef NO_HWLOC
    // Clean up topology
    hwloc_topology_destroy(balancer->topology);
#endif
    
    // Clean up devices
    for (int i = 0; i < balancer->num_devices; i++) {
        cleanup_device(&balancer->devices[i]);
    }
    free(balancer->devices);
    
    // Clean up chunks
    for (size_t i = 0; i < balancer->num_chunks; i++) {
        cleanup_chunk(balancer->chunks[i]);
    }
    free(balancer->chunks);
    
    // Clean up monitoring
    cleanup_performance_monitor(balancer->monitor);
    
    // Clean up communication
    cleanup_communication_optimizer(balancer->comm_optimizer);
    
    free(balancer->load_factors);
    free(balancer);
}
