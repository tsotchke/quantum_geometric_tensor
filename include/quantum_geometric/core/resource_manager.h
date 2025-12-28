/**
 * @file resource_manager.h
 * @brief Quantum computing resource management
 *
 * Manages quantum and classical computing resources including
 * qubit allocation, memory management, and hardware scheduling.
 */

#ifndef RESOURCE_MANAGER_H
#define RESOURCE_MANAGER_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct quantum_circuit;

// Resource types
typedef enum {
    RESOURCE_TYPE_QUBIT,
    RESOURCE_TYPE_CLASSICAL_REGISTER,
    RESOURCE_TYPE_MEMORY,
    RESOURCE_TYPE_GPU_MEMORY,
    RESOURCE_TYPE_NETWORK_BANDWIDTH,
    RESOURCE_TYPE_BACKEND_TIME,
    RESOURCE_TYPE_ANCILLA_QUBIT
} ResourceType;

// Resource state
typedef enum {
    RESOURCE_STATE_AVAILABLE,
    RESOURCE_STATE_ALLOCATED,
    RESOURCE_STATE_IN_USE,
    RESOURCE_STATE_RESERVED,
    RESOURCE_STATE_FAULTY,
    RESOURCE_STATE_OFFLINE,
    RESOURCE_STATE_CALIBRATING
} ResourceState;

// Allocation priority
typedef enum {
    ALLOC_PRIORITY_LOW = 0,
    ALLOC_PRIORITY_NORMAL = 1,
    ALLOC_PRIORITY_HIGH = 2,
    ALLOC_PRIORITY_CRITICAL = 3
} AllocationPriority;

// Allocation policy
typedef enum {
    ALLOC_POLICY_FIRST_FIT,
    ALLOC_POLICY_BEST_FIT,
    ALLOC_POLICY_WORST_FIT,
    ALLOC_POLICY_FIDELITY_AWARE,
    ALLOC_POLICY_TOPOLOGY_AWARE
} AllocationPolicy;

// Qubit resource
typedef struct QubitResource {
    size_t qubit_id;
    ResourceState state;
    double t1_time;               // Relaxation time (microseconds)
    double t2_time;               // Dephasing time (microseconds)
    double gate_fidelity;         // Single-qubit gate fidelity
    double readout_fidelity;      // Measurement fidelity
    size_t* connected_qubits;     // Coupling map (adjacent qubits)
    size_t num_connections;
    char* owner_task_id;
    uint64_t allocated_at;
    uint64_t last_calibrated;
} QubitResource;

// Memory resource
typedef struct MemoryResource {
    void* base_address;
    size_t size_bytes;
    size_t used_bytes;
    ResourceState state;
    size_t alignment;
    char* owner_task_id;
    bool is_gpu_memory;
    int device_id;
} MemoryResource;

// Backend resource
typedef struct BackendResource {
    char* backend_name;
    char* backend_type;           // "ibm", "rigetti", "dwave", "simulator"
    size_t num_qubits;
    size_t max_shots;
    size_t queue_depth;
    double average_wait_time;     // Seconds
    bool is_available;
    double* coupling_map;
    size_t coupling_map_size;
} BackendResource;

// Resource constraints for allocation requests
typedef struct ResourceConstraints {
    double min_fidelity;
    double min_t1;
    double min_t2;
    size_t* required_connectivity;  // Required adjacent qubits
    size_t num_connectivity_reqs;
    bool require_contiguous;
    size_t* preferred_qubits;
    size_t num_preferred;
} ResourceConstraints;

// Resource allocation request
typedef struct ResourceRequest {
    ResourceType type;
    size_t count;                 // Number of resources needed
    ResourceConstraints constraints;
    AllocationPriority priority;
    uint64_t timeout_ms;
    bool blocking;
    char* requester_id;
} ResourceRequest;

// Allocation handle (opaque)
typedef uint64_t AllocationHandle;
typedef uint64_t ReservationHandle;

// Resource allocation result
typedef struct ResourceAllocation {
    ResourceType type;
    void** resources;             // Array of allocated resources
    size_t count;
    AllocationHandle handle;
    uint64_t allocated_at;
    uint64_t expires_at;
    char* owner_id;
} ResourceAllocation;

// Resource metrics
typedef struct ResourceMetrics {
    size_t total_qubits;
    size_t available_qubits;
    size_t faulty_qubits;
    size_t reserved_qubits;
    double average_fidelity;
    double average_t1;
    double average_t2;
    size_t total_memory_bytes;
    size_t used_memory_bytes;
    size_t active_allocations;
    size_t pending_requests;
    double utilization_percent;
} ResourceMetrics;

// Coupling map for qubit topology
typedef struct CouplingMap {
    size_t num_qubits;
    bool** adjacency;             // adjacency[i][j] = true if qubits i,j are coupled
    double** coupling_strength;   // Two-qubit gate fidelity for each pair
    double** gate_times;          // Two-qubit gate time for each pair
} CouplingMap;

// Resource pool configuration
typedef struct ResourcePoolConfig {
    size_t max_qubits;
    size_t max_memory_bytes;
    size_t max_gpu_memory_bytes;
    AllocationPolicy allocation_policy;
    bool enable_fidelity_tracking;
    bool enable_automatic_calibration;
    uint64_t calibration_interval_ms;
    bool enable_preemption;
    size_t max_pending_requests;
} ResourcePoolConfig;

// Resource pool (main manager structure)
typedef struct ResourcePool {
    QubitResource* qubits;
    size_t num_qubits;
    MemoryResource* memory_pools;
    size_t num_memory_pools;
    BackendResource* backends;
    size_t num_backends;
    CouplingMap* topology;
    ResourcePoolConfig config;
    pthread_mutex_t lock;
    pthread_cond_t resource_available;
    ResourceAllocation** active_allocations;
    size_t num_allocations;
    size_t allocations_capacity;
    ResourceMetrics metrics;
} ResourcePool;

// =============================================================================
// Resource Pool Management
// =============================================================================

/**
 * Create a new resource pool
 */
int resource_manager_create(ResourcePool** pool, ResourcePoolConfig* config);

/**
 * Destroy a resource pool and free all resources
 */
void resource_manager_destroy(ResourcePool* pool);

/**
 * Initialize resource pool from backend properties
 */
int resource_manager_init_from_backend(ResourcePool* pool,
                                        const char* backend_name,
                                        void* backend_handle);

// =============================================================================
// Qubit Resource Management
// =============================================================================

/**
 * Add qubits to the resource pool
 */
int resource_manager_add_qubits(ResourcePool* pool,
                                 QubitResource* qubits,
                                 size_t num_qubits);

/**
 * Update qubit calibration data
 */
int resource_manager_update_calibration(ResourcePool* pool,
                                          size_t qubit_id,
                                          double t1, double t2,
                                          double gate_fidelity,
                                          double readout_fidelity);

/**
 * Set qubit coupling map (topology)
 */
int resource_manager_set_topology(ResourcePool* pool, CouplingMap* topology);

/**
 * Get qubit topology
 */
int resource_manager_get_qubit_topology(ResourcePool* pool, CouplingMap** topology_out);

/**
 * Mark a qubit as faulty
 */
int resource_manager_mark_faulty(ResourcePool* pool, size_t qubit_id);

/**
 * Mark a qubit as recovered
 */
int resource_manager_mark_recovered(ResourcePool* pool, size_t qubit_id);

// =============================================================================
// Resource Allocation
// =============================================================================

/**
 * Allocate resources based on request
 */
int resource_manager_allocate(ResourcePool* pool,
                               ResourceRequest* request,
                               ResourceAllocation** allocation_out);

/**
 * Release allocated resources
 */
int resource_manager_release(ResourcePool* pool, AllocationHandle handle);

/**
 * Reserve resources for future use
 */
int resource_manager_reserve(ResourcePool* pool,
                              ResourceRequest* request,
                              ReservationHandle* handle_out);

/**
 * Cancel a reservation
 */
int resource_manager_cancel_reservation(ResourcePool* pool,
                                          ReservationHandle handle);

/**
 * Claim a reservation (convert to allocation)
 */
int resource_manager_claim_reservation(ResourcePool* pool,
                                         ReservationHandle reservation,
                                         ResourceAllocation** allocation_out);

// =============================================================================
// Qubit Selection Algorithms
// =============================================================================

/**
 * Find best qubits for a circuit (fidelity-aware)
 */
int resource_manager_find_best_qubits(ResourcePool* pool,
                                        size_t num_qubits,
                                        ResourceConstraints* constraints,
                                        size_t** selected_qubits,
                                        size_t* num_selected);

/**
 * Find contiguous qubit region matching topology requirements
 */
int resource_manager_find_contiguous_region(ResourcePool* pool,
                                              size_t num_qubits,
                                              size_t** region_qubits);

/**
 * Map logical qubits to physical qubits
 */
int resource_manager_map_qubits(ResourcePool* pool,
                                 struct quantum_circuit* circuit,
                                 size_t** mapping);

// =============================================================================
// Resource Monitoring
// =============================================================================

/**
 * Get current resource availability
 */
int resource_manager_get_availability(ResourcePool* pool,
                                        ResourceType type,
                                        size_t* available_out,
                                        size_t* total_out);

/**
 * Get comprehensive resource metrics
 */
int resource_manager_get_metrics(ResourcePool* pool, ResourceMetrics* metrics_out);

/**
 * Get qubit by ID
 */
QubitResource* resource_manager_get_qubit(ResourcePool* pool, size_t qubit_id);

/**
 * Get all available qubits
 */
int resource_manager_get_available_qubits(ResourcePool* pool,
                                            size_t** qubit_ids,
                                            size_t* num_qubits);

// =============================================================================
// Backend Management
// =============================================================================

/**
 * Register a quantum backend
 */
int resource_manager_register_backend(ResourcePool* pool,
                                        BackendResource* backend);

/**
 * Unregister a backend
 */
int resource_manager_unregister_backend(ResourcePool* pool,
                                          const char* backend_name);

/**
 * Get available backends
 */
int resource_manager_get_backends(ResourcePool* pool,
                                    BackendResource*** backends,
                                    size_t* num_backends);

/**
 * Select best backend for a circuit
 */
int resource_manager_select_backend(ResourcePool* pool,
                                      struct quantum_circuit* circuit,
                                      BackendResource** best_backend);

// =============================================================================
// Memory Management
// =============================================================================

/**
 * Allocate memory from the resource pool
 */
void* resource_manager_alloc_memory(ResourcePool* pool,
                                     size_t size,
                                     size_t alignment);

/**
 * Free memory back to the resource pool
 */
void resource_manager_free_memory(ResourcePool* pool, void* ptr);

/**
 * Allocate GPU memory
 */
void* resource_manager_alloc_gpu_memory(ResourcePool* pool,
                                          size_t size,
                                          int device_id);

/**
 * Free GPU memory
 */
void resource_manager_free_gpu_memory(ResourcePool* pool, void* ptr);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Create a coupling map from adjacency list
 */
CouplingMap* coupling_map_create(size_t num_qubits);

/**
 * Add an edge to the coupling map
 */
int coupling_map_add_edge(CouplingMap* map,
                           size_t qubit1,
                           size_t qubit2,
                           double strength);

/**
 * Check if two qubits are connected
 */
bool coupling_map_are_connected(CouplingMap* map, size_t qubit1, size_t qubit2);

/**
 * Destroy a coupling map
 */
void coupling_map_destroy(CouplingMap* map);

/**
 * Get shortest path between two qubits
 */
int coupling_map_shortest_path(CouplingMap* map,
                                size_t source,
                                size_t target,
                                size_t** path,
                                size_t* path_length);

/**
 * Free a resource allocation structure
 */
void resource_allocation_free(ResourceAllocation* allocation);

/**
 * Print resource pool status (debug)
 */
void resource_manager_print_status(ResourcePool* pool);

#ifdef __cplusplus
}
#endif

#endif // RESOURCE_MANAGER_H
