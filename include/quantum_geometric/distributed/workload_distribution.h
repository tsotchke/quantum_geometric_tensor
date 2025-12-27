#ifndef WORKLOAD_DISTRIBUTION_H
#define WORKLOAD_DISTRIBUTION_H

#include <stddef.h>
#include <pthread.h>
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/config/mpi_config.h"

// Work item types
typedef enum {
    WORK_COMPUTE,
    WORK_COMMUNICATION, 
    WORK_IO,
    WORK_CUSTOM
} WorkItemType;

// Work item
typedef struct {
    void* data;
    size_t size;
    WorkItemType type;
    int priority;
    double cost_estimate;
} WorkItem;

// Work queue
typedef struct {
    WorkItem* items;
    size_t capacity;
    size_t size;
    size_t head;
    size_t tail;
    pthread_mutex_t mutex;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
} WorkQueue;

// Work metrics for performance tracking
typedef struct {
    size_t compute_time;
    size_t io_time;
    size_t network_time;
    size_t custom_time;
    size_t total_time;
    double average_cost;
} WorkMetrics;

// Workload manager
typedef struct {
    WorkQueue* local_queue;
    int rank;
    int world_size;
#ifdef USE_MPI
    qg_mpi_comm_t comm;
#endif
    bool is_coordinator;
    pthread_t balance_thread;
    bool running;
    double* node_loads;
    size_t total_items_processed;
    WorkMetrics metrics;
} WorkloadManager;

// Public API - Workload Manager
WorkloadManager* init_workload_manager(void);
void cleanup_workload_manager(WorkloadManager* manager);
int submit_work(WorkloadManager* manager, const void* data, size_t size, WorkItemType type, int priority);
int get_work(WorkloadManager* manager, void** data, size_t* size, WorkItemType* type);

// ============================================================================
// Production Distributed Computing API
// ============================================================================

/**
 * @brief Distribution strategy for workload partitioning
 */
typedef enum {
    DIST_STRATEGY_BLOCK = 0,       /**< Block distribution (contiguous chunks) */
    DIST_STRATEGY_CYCLIC,          /**< Cyclic distribution (round-robin) */
    DIST_STRATEGY_BLOCK_CYCLIC,    /**< Block-cyclic (blocks in round-robin) */
    DIST_STRATEGY_ADAPTIVE         /**< Adaptive based on node performance */
} DistributionStrategy;

/**
 * @brief Node load information for adaptive distribution
 */
typedef struct {
    int rank;                      /**< Node rank */
    double compute_capacity;       /**< Relative compute capacity (1.0 = baseline) */
    double current_load;           /**< Current load factor [0, 1] */
    double memory_available;       /**< Available memory in bytes */
    double network_bandwidth;      /**< Network bandwidth to coordinator (bytes/sec) */
    double latency;                /**< Network latency to coordinator (seconds) */
} NodeLoadInfo;

/**
 * @brief Distribution context for a workload partition
 */
typedef struct {
    size_t total_size;             /**< Total workload size */
    size_t local_size;             /**< Local partition size */
    size_t local_offset;           /**< Offset into global array */
    size_t element_size;           /**< Size of each element in bytes */
    DistributionStrategy strategy; /**< Distribution strategy used */

    // MPI information
    int rank;                      /**< This node's rank */
    int world_size;                /**< Total number of nodes */

    // Partition boundaries for all nodes (for irregular distributions)
    size_t* node_offsets;          /**< Starting offset for each node */
    size_t* node_sizes;            /**< Size for each node */

    // Performance tracking
    double distribution_time;      /**< Time taken to compute distribution */
    double imbalance_factor;       /**< Load imbalance factor (1.0 = perfect) */
} DistributionContext;

/**
 * @brief Synchronization options for result aggregation
 */
typedef struct {
    bool use_nonblocking;          /**< Use non-blocking communication */
    bool verify_integrity;         /**< Verify data integrity after sync */
    size_t chunk_size;             /**< Chunk size for large transfers (0 = auto) */
    int max_retries;               /**< Maximum retry attempts on failure */
    double timeout_seconds;        /**< Timeout for synchronization (0 = infinite) */
} SyncOptions;

/**
 * @brief Initialize distribution context for a workload
 *
 * Creates a distribution context that partitions a workload of size n
 * across available nodes using the specified strategy.
 *
 * @param n Total workload size (number of elements)
 * @param element_size Size of each element in bytes
 * @param strategy Distribution strategy to use
 * @return Distribution context (caller must free with cleanup_distribution)
 */
DistributionContext* init_distribution(size_t n, size_t element_size, DistributionStrategy strategy);

/**
 * @brief Initialize adaptive distribution based on node loads
 *
 * Creates an adaptive distribution that balances work based on
 * current node capacities and loads.
 *
 * @param n Total workload size
 * @param element_size Size of each element in bytes
 * @param node_loads Array of node load information (NULL for auto-detect)
 * @param num_nodes Number of nodes (0 for auto-detect)
 * @return Distribution context
 */
DistributionContext* init_adaptive_distribution(size_t n, size_t element_size,
                                                const NodeLoadInfo* node_loads,
                                                int num_nodes);

/**
 * @brief Cleanup distribution context
 *
 * @param ctx Context to cleanup
 */
void cleanup_distribution(DistributionContext* ctx);

/**
 * @brief Distribute workload and return local size
 *
 * Convenience function that creates a distribution and returns
 * the local partition size for this node.
 *
 * @param n Total workload size
 * @return Local partition size
 */
size_t distribute_workload(size_t n);

/**
 * @brief Get local offset for current distribution
 *
 * Returns the offset into the global array for this node's partition.
 * Must be called after distribute_workload.
 *
 * @return Local offset
 */
size_t get_local_offset(void);

/**
 * @brief Get current local workload size
 *
 * Returns the size of the current local partition.
 * Must be called after distribute_workload.
 *
 * @return Local size
 */
size_t get_local_workload_size(void);

/**
 * @brief Synchronize results across all nodes
 *
 * Gathers results from all nodes into a complete array.
 * Uses MPI_Allgatherv for variable-sized partitions.
 *
 * @param results Local results array (output is full gathered array)
 * @param n Total size of results array
 */
void synchronize_results(void* results, size_t n);

/**
 * @brief Synchronize results with options
 *
 * Extended version with configurable options.
 *
 * @param results Local results array
 * @param n Total size of results array
 * @param element_size Size of each element in bytes
 * @param options Synchronization options
 * @return 0 on success, error code on failure
 */
int synchronize_results_ex(void* results, size_t n, size_t element_size,
                          const SyncOptions* options);

/**
 * @brief Synchronize double complex field results
 *
 * Specialized version for complex-valued quantum field data
 * with optimized packing for complex numbers.
 *
 * @param results Complex array to synchronize
 * @param n Total number of elements
 */
void synchronize_complex_results(double _Complex* results, size_t n);

/**
 * @brief Scatter data from coordinator to all nodes
 *
 * Distributes data from the coordinator node to all nodes
 * according to the current distribution.
 *
 * @param sendbuf Send buffer (only used on coordinator)
 * @param recvbuf Receive buffer for local portion
 * @param n Total size
 * @param element_size Size of each element
 * @return 0 on success
 */
int scatter_data(const void* sendbuf, void* recvbuf, size_t n, size_t element_size);

/**
 * @brief Gather data from all nodes to coordinator
 *
 * Collects data from all nodes to the coordinator.
 *
 * @param sendbuf Local data to send
 * @param recvbuf Receive buffer (only used on coordinator)
 * @param n Total size
 * @param element_size Size of each element
 * @return 0 on success
 */
int gather_data(const void* sendbuf, void* recvbuf, size_t n, size_t element_size);

/**
 * @brief Get distributed rank (0 if not using MPI)
 *
 * @return This node's rank
 */
int get_distributed_rank(void);

/**
 * @brief Get distributed size (1 if not using MPI)
 *
 * @return Total number of nodes
 */
int get_distributed_size(void);

/**
 * @brief Check if this is the coordinator node
 *
 * @return true if rank == 0
 */
bool is_coordinator(void);

/**
 * @brief Barrier synchronization across all nodes
 */
void distributed_barrier(void);

/**
 * @brief Reduce operation across all nodes
 *
 * @param sendbuf Local data
 * @param recvbuf Result (on all nodes for allreduce)
 * @param count Number of elements
 * @param element_size Size of each element
 * @param op Reduction operation (0=sum, 1=max, 2=min, 3=prod)
 * @return 0 on success
 */
int distributed_reduce(const void* sendbuf, void* recvbuf, size_t count,
                       size_t element_size, int op);

/**
 * @brief Get node load information
 *
 * Collects current load information for this node.
 *
 * @param info Output load information
 * @return 0 on success
 */
int get_node_load_info(NodeLoadInfo* info);

/**
 * @brief Collect load information from all nodes
 *
 * Gathers load information from all nodes to the coordinator.
 *
 * @param infos Array of load info (size = world_size, coordinator only)
 * @return 0 on success
 */
int collect_all_node_loads(NodeLoadInfo* infos);

/**
 * @brief Rebalance workload based on current loads
 *
 * Redistributes data based on updated load information.
 * May involve data migration between nodes.
 *
 * @param ctx Distribution context to rebalance
 * @param data Data array to redistribute
 * @param element_size Size of each element
 * @return 0 on success
 */
int rebalance_workload(DistributionContext* ctx, void* data, size_t element_size);

#endif // WORKLOAD_DISTRIBUTION_H
