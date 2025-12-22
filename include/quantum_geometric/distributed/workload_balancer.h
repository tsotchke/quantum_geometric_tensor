#ifndef WORKLOAD_BALANCER_H
#define WORKLOAD_BALANCER_H

#include <stddef.h>
#include <stdbool.h>
#include "quantum_geometric/distributed/training_orchestrator.h"
#include "quantum_geometric/distributed/communication_optimizer.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Workload Balancer Constants
// ============================================================================

#define MAX_CHUNKS 4096
#define MAX_BALANCER_DEVICES 64
#define REBALANCE_THRESHOLD 0.2
#define MONITORING_INTERVAL_MS 100
#define MIN_CHUNK_SIZE 1024

// ============================================================================
// Type Definitions
// ============================================================================

// Workload type enumeration
typedef enum {
    WORKLOAD_CPU,
    WORKLOAD_GPU,
    WORKLOAD_QUANTUM,
    WORKLOAD_HYBRID,
    // Compatibility aliases
    CPU_WORKLOAD = WORKLOAD_CPU,
    GPU_WORKLOAD = WORKLOAD_GPU,
    QUANTUM_WORKLOAD = WORKLOAD_QUANTUM
} WorkloadType;

// Use DeviceType from training_orchestrator.h
// Additional device type aliases for workload balancer compatibility
#define DEVICE_GPU DEVICE_GPU_CUDA
#define DEVICE_QUANTUM DEVICE_QPU

// Device statistics for performance tracking
typedef struct {
    double utilization;             // CPU/GPU utilization percentage (0.0-1.0)
    double memory_used;             // Memory used in bytes
    double memory_total;            // Total memory in bytes
    double compute_time_ms;         // Recent computation time in ms
    double queue_depth;             // Average queue depth
    double throughput;              // Operations per second
    size_t tasks_completed;         // Total tasks completed
    size_t errors;                  // Error count
} DeviceStats;

// Performance monitor structure
typedef struct PerformanceMonitor {
    DeviceStats* device_stats;
    size_t num_devices;
    double total_utilization;
    double efficiency;
    bool is_monitoring;
} PerformanceMonitor;

// Load balancer configuration
typedef struct {
    size_t num_workers;              // Number of worker threads
    size_t chunk_size;               // Default chunk size
    double load_threshold;           // Load imbalance threshold for rebalancing
    bool dynamic_scaling;            // Enable dynamic workload scaling
    bool affinity_enabled;           // Enable CPU affinity
    CommConfig comm_config;          // Communication configuration
} BalancerConfig;

// Forward declaration of load balancer (opaque type)
typedef struct LoadBalancer LoadBalancer;

// ============================================================================
// Load Balancer API
// ============================================================================

// Initialization and cleanup
LoadBalancer* init_load_balancer(const BalancerConfig* config);
void cleanup_load_balancer(LoadBalancer* balancer);

// Workload distribution
void distribute_workload(LoadBalancer* balancer, void* data, size_t size, WorkloadType type);

// Performance monitoring
void monitor_performance(LoadBalancer* balancer);
PerformanceMonitor* init_performance_monitor(size_t num_devices);
void cleanup_performance_monitor(PerformanceMonitor* monitor);

// Device detection helpers
void detect_cpu_capabilities(void* devices, int* count);
void detect_gpu_capabilities(void* devices, int* count);
void detect_quantum_capabilities(void* devices, int* count);
void analyze_device_performance(void* device);
void cleanup_device(void* device);

// Device statistics
void update_device_stats(DeviceStats* stats);
bool detect_performance_degradation(const DeviceStats* stats);

// Chunk operations
size_t calculate_chunk_count(const LoadBalancer* balancer, size_t size, WorkloadType type);
void* create_chunk(void* data, size_t size, WorkloadType type);
void cleanup_chunk(void* chunk);

// Load balancing
bool should_rebalance(const LoadBalancer* balancer);
void rebalance_workload(LoadBalancer* balancer);
void migrate_chunks_from_device(LoadBalancer* balancer, int device_idx);

#ifdef __cplusplus
}
#endif

#endif // WORKLOAD_BALANCER_H
