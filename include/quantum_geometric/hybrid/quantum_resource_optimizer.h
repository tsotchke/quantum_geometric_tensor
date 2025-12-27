#ifndef QUANTUM_RESOURCE_OPTIMIZER_H
#define QUANTUM_RESOURCE_OPTIMIZER_H

#include <stddef.h>
#include <stdbool.h>
#include "quantum_geometric/hardware/quantum_hardware_abstraction.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct MemoryManager MemoryManager;
typedef struct NetworkManager NetworkManager;
typedef struct ComputeManager ComputeManager;
typedef struct ResourceMonitor ResourceMonitor;
typedef struct UsageTracker UsageTracker;

// System configuration for resource optimizer
typedef struct {
    size_t total_memory;      // Total available memory in bytes
    size_t cache_size;        // Cache size in bytes
    double bandwidth;         // Network bandwidth in MB/s
    double latency;           // Network latency in ms
    int num_cores;            // Number of CPU cores
    int num_gpus;             // Number of GPUs
} SystemConfig;

// Resource optimizer structure
typedef struct ResourceOptimizer {
    MemoryManager* memory_mgr;
    NetworkManager* network_mgr;
    ComputeManager* compute_mgr;
    ResourceMonitor* monitor;
    UsageTracker* tracker;
} ResourceOptimizer;

// Resource allocation result
typedef struct {
    // Memory allocation
    void* workspace;
    void* buffer;
    size_t allocated_size;

    // Network configuration
    double allocated_bandwidth;
    int network_priority;
    size_t packet_size;

    // Compute allocation
    int num_cores;
    int num_gpus;
    int thread_priority;
} ResourceAllocation;

// Core functions
ResourceOptimizer* init_resource_optimizer(const SystemConfig* config);
void cleanup_resource_optimizer(ResourceOptimizer* opt);

int optimize_resources(ResourceOptimizer* opt,
                       const QuantumOperation* op,
                       ResourceAllocation* alloc);

void release_resources(ResourceOptimizer* opt, ResourceAllocation* alloc);

// Memory manager functions
MemoryManager* init_memory_manager(size_t total_memory, size_t cache_size);
void cleanup_memory_manager(MemoryManager* mgr);
void* allocate_aligned(MemoryManager* mgr, size_t size, size_t alignment);
void free_aligned(MemoryManager* mgr, void* ptr);

// Network manager functions
NetworkManager* init_network_manager(double bandwidth, double latency);
void cleanup_network_manager(NetworkManager* mgr);
double allocate_bandwidth(NetworkManager* mgr, double requested);
void release_bandwidth(NetworkManager* mgr, double amount);

// Compute manager functions
ComputeManager* init_compute_manager(int num_cores, int num_gpus);
void cleanup_compute_manager(ComputeManager* mgr);
int allocate_cores(ComputeManager* mgr, int requested, double intensity);
int allocate_gpus(ComputeManager* mgr, int requested);
void release_cores(ComputeManager* mgr, int count);
void release_gpus(ComputeManager* mgr, int count);

// Resource monitoring functions
ResourceMonitor* init_resource_monitor(void);
void cleanup_resource_monitor(ResourceMonitor* monitor);
UsageTracker* init_usage_tracker(void);
void cleanup_usage_tracker(UsageTracker* tracker);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_RESOURCE_OPTIMIZER_H
