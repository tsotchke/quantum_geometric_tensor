/**
 * @file quantum_resource_optimizer.c
 * @brief Resource optimization for hybrid quantum-classical operations
 *
 * This module manages memory, network, and compute resources to optimize
 * the execution of quantum operations across hybrid systems.
 */

#include "quantum_geometric/hybrid/quantum_resource_optimizer.h"
#include "quantum_geometric/core/quantum_geometric_operations.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// Forward declarations
void cleanup_quantum_resource_optimizer(ResourceOptimizer* opt);

// Resource parameters
#define MAX_MEMORY_POOL (1ULL << 30)  // 1GB
#define MAX_NETWORK_BANDWIDTH 10000.0  // MB/s
#define MAX_COMPUTE_UNITS 32
#define CACHE_LINE 64
#define DEFAULT_ALIGNMENT 64

// Optimization thresholds
#define MIN_BATCH_SIZE 16
#define MAX_BATCH_SIZE 1024
#define MEMORY_THRESHOLD 0.8
#define NETWORK_THRESHOLD 0.7
#define COMPUTE_THRESHOLD 0.9

// ============================================================================
// Internal structures
// ============================================================================

struct MemoryManager {
    size_t total_memory;
    size_t available_memory;
    size_t cache_size;
    size_t peak_usage;
    size_t allocation_count;
    double fragmentation;
};

struct NetworkManager {
    double total_bandwidth;
    double available_bandwidth;
    double base_latency;
    double current_latency;
    size_t active_connections;
    double congestion_factor;
};

struct ComputeManager {
    int total_cores;
    int available_cores;
    int total_gpus;
    int available_gpus;
    double cpu_utilization;
    double gpu_utilization;
};

struct ResourceMonitor {
    double last_update_time;
    double memory_trend;
    double network_trend;
    double compute_trend;
    size_t sample_count;
};

struct UsageTracker {
    double* memory_history;
    double* network_history;
    double* compute_history;
    size_t history_size;
    size_t current_index;
};

// Internal resource status
typedef struct {
    size_t total_memory;
    size_t available_memory;
    size_t cache_size;
    double fragmentation;
    double bandwidth;
    double latency;
    size_t packet_size;
    double congestion;
    int total_cores;
    int available_cores;
    double cpu_utilization;
    double gpu_utilization;
} ResourceStatus;

// Internal resource requirements
typedef struct {
    size_t workspace_size;
    size_t buffer_size;
    size_t alignment;
    double bandwidth_needed;
    double max_latency;
    size_t data_size;
    int min_cores;
    int min_gpus;
    double compute_intensity;
} ResourceRequirements;

// ============================================================================
// Forward declarations
// ============================================================================

static ResourceStatus* get_resource_status(ResourceOptimizer* opt);
static ResourceRequirements* analyze_requirements(const QuantumOperation* op);
static bool check_availability(const ResourceStatus* status, const ResourceRequirements* requirements);
static ResourceAllocation* create_allocation_plan(ResourceOptimizer* opt,
                                                   const ResourceStatus* status,
                                                   const ResourceRequirements* requirements);
static bool allocate_memory_resources(ResourceAllocation* plan,
                                       const ResourceStatus* status,
                                       const ResourceRequirements* requirements,
                                       MemoryManager* mgr);
static bool configure_network_resources(ResourceAllocation* plan,
                                         const ResourceStatus* status,
                                         const ResourceRequirements* requirements,
                                         NetworkManager* mgr);
static bool allocate_compute_resources(ResourceAllocation* plan,
                                        const ResourceStatus* status,
                                        const ResourceRequirements* requirements,
                                        ComputeManager* mgr);
static bool validate_allocation(const ResourceAllocation* plan, const ResourceStatus* status);
static void update_resource_status(ResourceOptimizer* opt, const ResourceAllocation* plan);
static void free_memory_resources(ResourceAllocation* plan, MemoryManager* mgr);
static int calculate_priority(const ResourceRequirements* requirements);
static size_t optimize_packet_size(size_t data_size, double latency);
static int calculate_thread_priority(const ResourceRequirements* requirements);
static void cleanup_optimization(ResourceStatus* status, ResourceRequirements* requirements, ResourceAllocation* plan);

// ============================================================================
// Memory manager implementation (renamed to avoid conflict with platform-specific versions)
// ============================================================================

// Renamed to avoid conflict with memory_optimization_macos.c / memory_optimization_linux.c
MemoryManager* init_resource_memory_manager(size_t total_memory, size_t cache_size) {
    MemoryManager* mgr = calloc(1, sizeof(MemoryManager));
    if (!mgr) return NULL;

    mgr->total_memory = total_memory > 0 ? total_memory : MAX_MEMORY_POOL;
    mgr->available_memory = mgr->total_memory;
    mgr->cache_size = cache_size > 0 ? cache_size : (32 * 1024 * 1024); // 32MB default
    mgr->peak_usage = 0;
    mgr->allocation_count = 0;
    mgr->fragmentation = 0.0;

    return mgr;
}

// Renamed to avoid conflict with platform-specific versions
void cleanup_resource_memory_manager(MemoryManager* mgr) {
    free(mgr);
}

void* allocate_aligned(MemoryManager* mgr, size_t size, size_t alignment) {
    if (!mgr || size == 0) return NULL;
    if (alignment == 0) alignment = DEFAULT_ALIGNMENT;

    // Ensure alignment is power of 2
    alignment = (alignment < CACHE_LINE) ? CACHE_LINE : alignment;

    // Check availability
    size_t actual_size = size + alignment - 1;
    if (actual_size > mgr->available_memory) return NULL;

    void* ptr = NULL;
#if defined(_POSIX_VERSION) && _POSIX_VERSION >= 200112L
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
#else
    ptr = malloc(size + alignment);
    if (ptr) {
        uintptr_t addr = (uintptr_t)ptr;
        uintptr_t aligned = (addr + alignment - 1) & ~(alignment - 1);
        ptr = (void*)aligned;
    }
#endif

    if (ptr) {
        mgr->available_memory -= size;
        mgr->allocation_count++;
        if (mgr->total_memory - mgr->available_memory > mgr->peak_usage) {
            mgr->peak_usage = mgr->total_memory - mgr->available_memory;
        }
    }

    return ptr;
}

void free_aligned(MemoryManager* mgr, void* ptr) {
    if (!mgr || !ptr) return;
    free(ptr);
    mgr->allocation_count--;
}

// ============================================================================
// Network manager implementation
// ============================================================================

// Renamed to avoid conflict with quantum_hardware_scheduler.c (similar implementation, different context)
NetworkManager* init_quantum_network_manager(double bandwidth, double latency) {
    NetworkManager* mgr = calloc(1, sizeof(NetworkManager));
    if (!mgr) return NULL;

    mgr->total_bandwidth = bandwidth > 0 ? bandwidth : MAX_NETWORK_BANDWIDTH;
    mgr->available_bandwidth = mgr->total_bandwidth;
    mgr->base_latency = latency > 0 ? latency : 0.1;  // 0.1ms default
    mgr->current_latency = mgr->base_latency;
    mgr->active_connections = 0;
    mgr->congestion_factor = 1.0;

    return mgr;
}

void cleanup_network_manager(NetworkManager* mgr) {
    free(mgr);
}

double allocate_bandwidth(NetworkManager* mgr, double requested) {
    if (!mgr || requested <= 0) return 0.0;

    double allocated = (requested <= mgr->available_bandwidth)
                       ? requested
                       : mgr->available_bandwidth;

    if (allocated > 0) {
        mgr->available_bandwidth -= allocated;
        mgr->active_connections++;
        // Update congestion and latency
        double utilization = 1.0 - (mgr->available_bandwidth / mgr->total_bandwidth);
        mgr->congestion_factor = 1.0 + (utilization * utilization);
        mgr->current_latency = mgr->base_latency * mgr->congestion_factor;
    }

    return allocated;
}

void release_bandwidth(NetworkManager* mgr, double amount) {
    if (!mgr || amount <= 0) return;

    mgr->available_bandwidth += amount;
    if (mgr->available_bandwidth > mgr->total_bandwidth) {
        mgr->available_bandwidth = mgr->total_bandwidth;
    }
    if (mgr->active_connections > 0) {
        mgr->active_connections--;
    }
    // Update congestion
    double utilization = 1.0 - (mgr->available_bandwidth / mgr->total_bandwidth);
    mgr->congestion_factor = 1.0 + (utilization * utilization);
    mgr->current_latency = mgr->base_latency * mgr->congestion_factor;
}

// ============================================================================
// Compute manager implementation
// ============================================================================

ComputeManager* init_compute_manager(int num_cores, int num_gpus) {
    ComputeManager* mgr = calloc(1, sizeof(ComputeManager));
    if (!mgr) return NULL;

#ifdef _OPENMP
    int system_cores = omp_get_num_procs();
#else
    int system_cores = 4;  // Default fallback
#endif

    mgr->total_cores = (num_cores > 0) ? num_cores : system_cores;
    mgr->available_cores = mgr->total_cores;
    mgr->total_gpus = (num_gpus >= 0) ? num_gpus : 0;
    mgr->available_gpus = mgr->total_gpus;
    mgr->cpu_utilization = 0.0;
    mgr->gpu_utilization = 0.0;

    return mgr;
}

void cleanup_compute_manager(ComputeManager* mgr) {
    free(mgr);
}

int allocate_cores(ComputeManager* mgr, int requested, double intensity) {
    if (!mgr || requested <= 0) return 0;

    // Adjust request based on intensity
    int adjusted = (int)(requested * (1.0 + intensity * 0.5));
    int allocated = (adjusted <= mgr->available_cores) ? adjusted : mgr->available_cores;

    if (allocated > 0) {
        mgr->available_cores -= allocated;
        mgr->cpu_utilization = 1.0 - ((double)mgr->available_cores / mgr->total_cores);
    }

    return allocated;
}

int allocate_gpus(ComputeManager* mgr, int requested) {
    if (!mgr || requested <= 0) return 0;

    int allocated = (requested <= mgr->available_gpus) ? requested : mgr->available_gpus;

    if (allocated > 0) {
        mgr->available_gpus -= allocated;
        if (mgr->total_gpus > 0) {
            mgr->gpu_utilization = 1.0 - ((double)mgr->available_gpus / mgr->total_gpus);
        }
    }

    return allocated;
}

void release_cores(ComputeManager* mgr, int count) {
    if (!mgr || count <= 0) return;

    mgr->available_cores += count;
    if (mgr->available_cores > mgr->total_cores) {
        mgr->available_cores = mgr->total_cores;
    }
    mgr->cpu_utilization = 1.0 - ((double)mgr->available_cores / mgr->total_cores);
}

void release_gpus(ComputeManager* mgr, int count) {
    if (!mgr || count <= 0) return;

    mgr->available_gpus += count;
    if (mgr->available_gpus > mgr->total_gpus) {
        mgr->available_gpus = mgr->total_gpus;
    }
    if (mgr->total_gpus > 0) {
        mgr->gpu_utilization = 1.0 - ((double)mgr->available_gpus / mgr->total_gpus);
    }
}

// ============================================================================
// Resource monitoring implementation
// ============================================================================

ResourceMonitor* init_resource_monitor(void) {
    ResourceMonitor* monitor = calloc(1, sizeof(ResourceMonitor));
    if (!monitor) return NULL;

    monitor->last_update_time = 0.0;
    monitor->memory_trend = 0.0;
    monitor->network_trend = 0.0;
    monitor->compute_trend = 0.0;
    monitor->sample_count = 0;

    return monitor;
}

void cleanup_resource_monitor(ResourceMonitor* monitor) {
    free(monitor);
}

#define USAGE_HISTORY_SIZE 100

UsageTracker* init_usage_tracker(void) {
    UsageTracker* tracker = calloc(1, sizeof(UsageTracker));
    if (!tracker) return NULL;

    tracker->history_size = USAGE_HISTORY_SIZE;
    tracker->memory_history = calloc(tracker->history_size, sizeof(double));
    tracker->network_history = calloc(tracker->history_size, sizeof(double));
    tracker->compute_history = calloc(tracker->history_size, sizeof(double));
    tracker->current_index = 0;

    if (!tracker->memory_history || !tracker->network_history || !tracker->compute_history) {
        free(tracker->memory_history);
        free(tracker->network_history);
        free(tracker->compute_history);
        free(tracker);
        return NULL;
    }

    return tracker;
}

void cleanup_usage_tracker(UsageTracker* tracker) {
    if (!tracker) return;
    free(tracker->memory_history);
    free(tracker->network_history);
    free(tracker->compute_history);
    free(tracker);
}

// ============================================================================
// Resource optimizer implementation
// ============================================================================

// Renamed to avoid conflict with resource_optimization.c (this takes SystemConfig, that takes OptimizationStrategy)
ResourceOptimizer* init_quantum_resource_optimizer(const SystemConfig* config) {
    if (!config) return NULL;

    ResourceOptimizer* opt = calloc(1, sizeof(ResourceOptimizer));
    if (!opt) return NULL;

    opt->memory_mgr = init_memory_manager(config->total_memory, config->cache_size);
    opt->network_mgr = init_quantum_network_manager(config->bandwidth, config->latency);
    opt->compute_mgr = init_compute_manager(config->num_cores, config->num_gpus);
    opt->monitor = init_resource_monitor();
    opt->tracker = init_usage_tracker();

    if (!opt->memory_mgr || !opt->network_mgr || !opt->compute_mgr ||
        !opt->monitor || !opt->tracker) {
        cleanup_quantum_resource_optimizer(opt);
        return NULL;
    }

    return opt;
}

// Renamed to avoid conflict with resource_optimization.c
void cleanup_quantum_resource_optimizer(ResourceOptimizer* opt) {
    if (!opt) return;

    cleanup_memory_manager(opt->memory_mgr);
    cleanup_network_manager(opt->network_mgr);
    cleanup_compute_manager(opt->compute_mgr);
    cleanup_resource_monitor(opt->monitor);
    cleanup_usage_tracker(opt->tracker);
    free(opt);
}

int optimize_resources(ResourceOptimizer* opt,
                       const QuantumOperation* op,
                       ResourceAllocation* alloc) {
    if (!opt || !op || !alloc) return -1;

    memset(alloc, 0, sizeof(ResourceAllocation));

    // Get current resource status
    ResourceStatus* status = get_resource_status(opt);
    if (!status) return -1;

    // Analyze requirements
    ResourceRequirements* requirements = analyze_requirements(op);
    if (!requirements) {
        free(status);
        return -1;
    }

    // Check resource availability
    if (!check_availability(status, requirements)) {
        cleanup_optimization(status, requirements, NULL);
        return -1;
    }

    // Create allocation plan
    ResourceAllocation* plan = create_allocation_plan(opt, status, requirements);
    if (!plan) {
        cleanup_optimization(status, requirements, NULL);
        return -1;
    }

    // Validate allocation
    if (!validate_allocation(plan, status)) {
        cleanup_optimization(status, requirements, plan);
        return -1;
    }

    // Update resource status
    update_resource_status(opt, plan);

    // Copy allocation to output
    memcpy(alloc, plan, sizeof(ResourceAllocation));

    cleanup_optimization(status, requirements, plan);
    return 0;
}

// Renamed to avoid conflict with core/quantum_scheduler.c (this takes ResourceOptimizer*, that takes quantum_task_t*)
void release_quantum_resources(ResourceOptimizer* opt, ResourceAllocation* alloc) {
    if (!opt || !alloc) return;

    // Release memory
    if (alloc->workspace) {
        free_aligned(opt->memory_mgr, alloc->workspace);
        alloc->workspace = NULL;
    }
    if (alloc->buffer) {
        free_aligned(opt->memory_mgr, alloc->buffer);
        alloc->buffer = NULL;
    }

    // Release bandwidth
    release_bandwidth(opt->network_mgr, alloc->allocated_bandwidth);

    // Release compute resources
    release_cores(opt->compute_mgr, alloc->num_cores);
    release_gpus(opt->compute_mgr, alloc->num_gpus);

    memset(alloc, 0, sizeof(ResourceAllocation));
}

// ============================================================================
// Static helper implementations
// ============================================================================

static ResourceStatus* get_resource_status(ResourceOptimizer* opt) {
    ResourceStatus* status = calloc(1, sizeof(ResourceStatus));
    if (!status) return NULL;

    status->total_memory = opt->memory_mgr->total_memory;
    status->available_memory = opt->memory_mgr->available_memory;
    status->cache_size = opt->memory_mgr->cache_size;
    status->fragmentation = opt->memory_mgr->fragmentation;

    status->bandwidth = opt->network_mgr->available_bandwidth;
    status->latency = opt->network_mgr->current_latency;
    status->congestion = opt->network_mgr->congestion_factor;

    status->total_cores = opt->compute_mgr->total_cores;
    status->available_cores = opt->compute_mgr->available_cores;
    status->cpu_utilization = opt->compute_mgr->cpu_utilization;
    status->gpu_utilization = opt->compute_mgr->gpu_utilization;

    return status;
}

// Helper to estimate qubit count from an operation
static size_t estimate_operation_qubits(const QuantumOperation* op) {
    if (!op) return 1;

    switch (op->type) {
        case OPERATION_GATE:
            {
                uint32_t max_qubit = op->op.gate.qubit;
                if (op->op.gate.control_qubit > max_qubit) max_qubit = op->op.gate.control_qubit;
                if (op->op.gate.target_qubit > max_qubit) max_qubit = op->op.gate.target_qubit;
                return (size_t)(max_qubit + 1);
            }
        case OPERATION_MEASURE:
            return (size_t)(op->op.measure.qubit + 1);
        case OPERATION_RESET:
            return (size_t)(op->op.reset.qubit + 1);
        case OPERATION_BARRIER:
            return op->op.barrier.num_qubits > 0 ? op->op.barrier.num_qubits : 1;
        case OPERATION_ANNEAL:
            return 10;
        default:
            return 1;
    }
}

static ResourceRequirements* analyze_requirements(const QuantumOperation* op) {
    ResourceRequirements* req = calloc(1, sizeof(ResourceRequirements));
    if (!req) return NULL;

    // Estimate qubit count from operation
    size_t num_qubits = estimate_operation_qubits(op);

    // Estimate memory requirements using floating point for large qubit counts
    double state_size = pow(2.0, (double)num_qubits) * sizeof(double) * 2;

    req->workspace_size = (size_t)(state_size * 2);  // For intermediate calculations
    req->buffer_size = (size_t)state_size;
    req->alignment = CACHE_LINE;

    // Estimate network requirements
    req->bandwidth_needed = state_size / (1024.0 * 1024.0);  // MB
    req->max_latency = 10.0;  // ms
    req->data_size = (size_t)state_size;

    // Estimate compute requirements
    req->min_cores = (num_qubits <= 10) ? 1 : (int)(1 << (num_qubits > 20 ? 10 : (num_qubits - 10)));
    req->min_cores = (req->min_cores > MAX_COMPUTE_UNITS) ? MAX_COMPUTE_UNITS : req->min_cores;
    req->min_gpus = (num_qubits > 16) ? 1 : 0;
    req->compute_intensity = (double)num_qubits / 20.0;

    return req;
}

static bool check_availability(const ResourceStatus* status, const ResourceRequirements* requirements) {
    // Check memory
    size_t total_memory_needed = requirements->workspace_size + requirements->buffer_size;
    if (total_memory_needed > status->available_memory) return false;

    // Check network
    if (requirements->bandwidth_needed > status->bandwidth) return false;

    // Check compute
    if (requirements->min_cores > status->available_cores) return false;

    return true;
}

static ResourceAllocation* create_allocation_plan(ResourceOptimizer* opt,
                                                   const ResourceStatus* status,
                                                   const ResourceRequirements* requirements) {
    ResourceAllocation* plan = calloc(1, sizeof(ResourceAllocation));
    if (!plan) return NULL;

    if (!allocate_memory_resources(plan, status, requirements, opt->memory_mgr)) {
        free(plan);
        return NULL;
    }

    if (!configure_network_resources(plan, status, requirements, opt->network_mgr)) {
        free_memory_resources(plan, opt->memory_mgr);
        free(plan);
        return NULL;
    }

    if (!allocate_compute_resources(plan, status, requirements, opt->compute_mgr)) {
        free_memory_resources(plan, opt->memory_mgr);
        free(plan);
        return NULL;
    }

    return plan;
}

static bool allocate_memory_resources(ResourceAllocation* plan,
                                       const ResourceStatus* status,
                                       const ResourceRequirements* requirements,
                                       MemoryManager* mgr) {
    if (requirements->workspace_size > status->available_memory) return false;

    plan->workspace = allocate_aligned(mgr, requirements->workspace_size, requirements->alignment);
    if (!plan->workspace) return false;

    plan->buffer = allocate_aligned(mgr, requirements->buffer_size, CACHE_LINE);
    if (!plan->buffer) {
        free_aligned(mgr, plan->workspace);
        return false;
    }

    plan->allocated_size = requirements->workspace_size + requirements->buffer_size;
    return true;
}

static bool configure_network_resources(ResourceAllocation* plan,
                                         const ResourceStatus* status,
                                         const ResourceRequirements* requirements,
                                         NetworkManager* mgr) {
    if (requirements->bandwidth_needed > status->bandwidth) return false;

    plan->allocated_bandwidth = allocate_bandwidth(mgr, requirements->bandwidth_needed);
    if (plan->allocated_bandwidth < requirements->bandwidth_needed) return false;

    plan->network_priority = calculate_priority(requirements);
    plan->packet_size = optimize_packet_size(requirements->data_size, status->latency);

    return true;
}

static bool allocate_compute_resources(ResourceAllocation* plan,
                                        const ResourceStatus* status,
                                        const ResourceRequirements* requirements,
                                        ComputeManager* mgr) {
    if (requirements->min_cores > status->available_cores) return false;

    plan->num_cores = allocate_cores(mgr, requirements->min_cores, requirements->compute_intensity);
    if (plan->num_cores < requirements->min_cores) return false;

    if (requirements->min_gpus > 0) {
        plan->num_gpus = allocate_gpus(mgr, requirements->min_gpus);
        if (plan->num_gpus < requirements->min_gpus) return false;
    }

    plan->thread_priority = calculate_thread_priority(requirements);
    return true;
}

static bool validate_allocation(const ResourceAllocation* plan, const ResourceStatus* status) {
    if (plan->allocated_size > status->available_memory) return false;
    if (plan->allocated_bandwidth > status->bandwidth) return false;
    if (plan->num_cores > status->available_cores) return false;
    return true;
}

static void update_resource_status(ResourceOptimizer* opt, const ResourceAllocation* plan) {
    // Update usage tracker
    if (opt->tracker) {
        size_t idx = opt->tracker->current_index;
        double mem_usage = (double)plan->allocated_size / opt->memory_mgr->total_memory;
        double net_usage = plan->allocated_bandwidth / opt->network_mgr->total_bandwidth;
        double cpu_usage = (double)plan->num_cores / opt->compute_mgr->total_cores;

        opt->tracker->memory_history[idx] = mem_usage;
        opt->tracker->network_history[idx] = net_usage;
        opt->tracker->compute_history[idx] = cpu_usage;
        opt->tracker->current_index = (idx + 1) % opt->tracker->history_size;
    }
}

static void free_memory_resources(ResourceAllocation* plan, MemoryManager* mgr) {
    if (plan->workspace) {
        free_aligned(mgr, plan->workspace);
        plan->workspace = NULL;
    }
    if (plan->buffer) {
        free_aligned(mgr, plan->buffer);
        plan->buffer = NULL;
    }
}

static int calculate_priority(const ResourceRequirements* requirements) {
    // Higher intensity = higher priority
    if (requirements->compute_intensity > 0.8) return 0;  // Highest
    if (requirements->compute_intensity > 0.5) return 5;
    return 10;  // Normal
}

static size_t optimize_packet_size(size_t data_size, double latency) {
    // Balance between throughput and latency
    size_t base_packet = 1500;  // Standard MTU
    if (latency < 1.0 && data_size > 65536) {
        return 65536;  // Jumbo frames for low latency, large data
    }
    if (data_size < base_packet) return data_size;
    return base_packet;
}

static int calculate_thread_priority(const ResourceRequirements* requirements) {
    if (requirements->compute_intensity > 0.8) return -10;  // Higher priority (lower nice value)
    if (requirements->compute_intensity > 0.5) return 0;
    return 10;  // Lower priority
}

static void cleanup_optimization(ResourceStatus* status, ResourceRequirements* requirements, ResourceAllocation* plan) {
    free(status);
    free(requirements);
    free(plan);
}
