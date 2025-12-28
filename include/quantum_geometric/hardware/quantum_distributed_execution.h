/**
 * @file quantum_distributed_execution.h
 * @brief Distributed Execution for Quantum Computations
 *
 * Provides distributed execution capabilities including:
 * - Multi-node quantum simulation
 * - State distribution and gathering
 * - Distributed tensor networks
 * - Load balancing across nodes
 * - Fault-tolerant execution
 * - Communication optimization
 *
 * Part of the QGTL Hardware Acceleration Framework.
 */

#ifndef QUANTUM_DISTRIBUTED_EXECUTION_H
#define QUANTUM_DISTRIBUTED_EXECUTION_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================

#define DIST_MAX_NODES 1024
#define DIST_MAX_NAME_LENGTH 256
#define DIST_MAX_TASKS 65536
#define DIST_CHUNK_SIZE_DEFAULT (1ULL << 20)  // 1 MB

// ============================================================================
// Enumerations
// ============================================================================

/**
 * Distribution strategies
 */
typedef enum {
    DIST_STRATEGY_STATE_VECTOR,       // Distribute state vector
    DIST_STRATEGY_AMPLITUDE,          // Distribute by amplitude
    DIST_STRATEGY_QUBIT,              // Distribute by qubit
    DIST_STRATEGY_GATE,               // Distribute by gate
    DIST_STRATEGY_TENSOR,             // Distribute tensor network
    DIST_STRATEGY_HYBRID,             // Hybrid distribution
    DIST_STRATEGY_AUTO                // Automatic selection
} dist_strategy_t;

/**
 * Node roles
 */
typedef enum {
    DIST_ROLE_COORDINATOR,            // Master/coordinator node
    DIST_ROLE_WORKER,                 // Worker node
    DIST_ROLE_HYBRID                  // Both coordinator and worker
} dist_role_t;

/**
 * Task states
 */
typedef enum {
    DIST_TASK_PENDING,                // Not yet started
    DIST_TASK_QUEUED,                 // In queue
    DIST_TASK_RUNNING,                // Currently executing
    DIST_TASK_COMPLETED,              // Successfully completed
    DIST_TASK_FAILED,                 // Failed
    DIST_TASK_CANCELLED               // Cancelled
} dist_task_state_t;

/**
 * Communication patterns
 */
typedef enum {
    DIST_COMM_POINT_TO_POINT,         // Direct P2P
    DIST_COMM_BROADCAST,              // One to all
    DIST_COMM_GATHER,                 // All to one
    DIST_COMM_SCATTER,                // One to all (different data)
    DIST_COMM_ALL_TO_ALL,             // All to all
    DIST_COMM_REDUCE,                 // All to one with operation
    DIST_COMM_ALL_REDUCE              // Reduce then broadcast
} dist_comm_pattern_t;

/**
 * Fault tolerance modes
 */
typedef enum {
    DIST_FAULT_NONE,                  // No fault tolerance
    DIST_FAULT_CHECKPOINT,            // Periodic checkpointing
    DIST_FAULT_REPLICATE,             // Task replication
    DIST_FAULT_MIGRATE                // Task migration
} dist_fault_mode_t;

/**
 * Load balancing strategies
 */
typedef enum {
    DIST_BALANCE_STATIC,              // Static assignment
    DIST_BALANCE_DYNAMIC,             // Dynamic work stealing
    DIST_BALANCE_ADAPTIVE,            // Adaptive load balancing
    DIST_BALANCE_ROUND_ROBIN          // Round robin
} dist_balance_t;

// ============================================================================
// Data Structures
// ============================================================================

/**
 * Node information
 */
typedef struct {
    int node_id;
    char hostname[DIST_MAX_NAME_LENGTH];
    dist_role_t role;
    int num_gpus;
    size_t gpu_memory_total;
    size_t gpu_memory_available;
    size_t cpu_memory_total;
    size_t cpu_memory_available;
    int cpu_cores;
    double network_bandwidth_gbps;
    bool is_available;
    double load_factor;               // 0.0 to 1.0
} dist_node_info_t;

/**
 * Task specification
 */
typedef struct {
    uint64_t task_id;
    char name[DIST_MAX_NAME_LENGTH];
    void* input_data;
    size_t input_size;
    void* output_data;
    size_t output_size;
    int target_node;                  // -1 for any
    uint32_t priority;
    uint64_t deadline_ns;             // 0 for no deadline
    bool requires_gpu;
    size_t gpu_memory_required;
    size_t cpu_memory_required;
} dist_task_spec_t;

/**
 * Task result
 */
typedef struct {
    uint64_t task_id;
    dist_task_state_t state;
    int executed_on_node;
    uint64_t start_time_ns;
    uint64_t end_time_ns;
    void* result_data;
    size_t result_size;
    int error_code;
    char error_message[DIST_MAX_NAME_LENGTH];
} dist_task_result_t;

/**
 * State partition
 */
typedef struct {
    int node_id;
    size_t start_index;
    size_t count;
    size_t bytes;
    void* data;
    bool is_local;
} dist_partition_t;

/**
 * Communication statistics
 */
typedef struct {
    uint64_t total_messages;
    uint64_t total_bytes_sent;
    uint64_t total_bytes_received;
    double avg_latency_ns;
    double max_latency_ns;
    double bandwidth_achieved_gbps;
    uint64_t collisions;
    uint64_t retries;
} dist_comm_stats_t;

/**
 * Execution statistics
 */
typedef struct {
    uint64_t total_tasks;
    uint64_t completed_tasks;
    uint64_t failed_tasks;
    uint64_t cancelled_tasks;
    double avg_task_time_ns;
    double total_compute_time_ns;
    double total_comm_time_ns;
    double load_imbalance;            // Std dev of node loads
    double parallel_efficiency;       // Speedup / num_nodes
    dist_comm_stats_t comm_stats;
} dist_exec_stats_t;

/**
 * Checkpoint data
 */
typedef struct {
    uint64_t checkpoint_id;
    uint64_t timestamp_ns;
    size_t state_size;
    void* state_data;
    uint64_t completed_tasks;
    char description[DIST_MAX_NAME_LENGTH];
} dist_checkpoint_t;

/**
 * Configuration
 */
typedef struct {
    dist_strategy_t strategy;
    dist_fault_mode_t fault_mode;
    dist_balance_t balance_strategy;
    size_t chunk_size;
    uint64_t checkpoint_interval_ns;
    int max_retries;
    uint64_t timeout_ns;
    bool enable_compression;
    int compression_level;
    bool enable_profiling;
    size_t max_concurrent_tasks;
} dist_exec_config_t;

/**
 * Opaque executor handle
 */
typedef struct dist_executor dist_executor_t;

// ============================================================================
// Initialization
// ============================================================================

/**
 * Initialize distributed execution (must call before any other functions)
 */
bool dist_init(int* argc, char*** argv);

/**
 * Finalize distributed execution
 */
bool dist_finalize(void);

/**
 * Create executor
 */
dist_executor_t* dist_executor_create(void);

/**
 * Create with configuration
 */
dist_executor_t* dist_executor_create_with_config(
    const dist_exec_config_t* config);

/**
 * Get default configuration
 */
dist_exec_config_t dist_default_config(void);

/**
 * Destroy executor
 */
void dist_executor_destroy(dist_executor_t* executor);

// ============================================================================
// Node Management
// ============================================================================

/**
 * Get local node ID
 */
int dist_get_local_node_id(void);

/**
 * Get total number of nodes
 */
int dist_get_num_nodes(void);

/**
 * Check if this is coordinator
 */
bool dist_is_coordinator(void);

/**
 * Get node information
 */
bool dist_get_node_info(
    dist_executor_t* executor,
    int node_id,
    dist_node_info_t* info);

/**
 * Get all node information
 */
bool dist_get_all_nodes(
    dist_executor_t* executor,
    dist_node_info_t** nodes,
    int* count);

/**
 * Update node availability
 */
bool dist_set_node_available(
    dist_executor_t* executor,
    int node_id,
    bool available);

// ============================================================================
// State Distribution
// ============================================================================

/**
 * Distribute state vector across nodes
 */
bool dist_distribute_state(
    dist_executor_t* executor,
    const void* state,
    size_t state_size,
    size_t num_elements,
    dist_strategy_t strategy,
    dist_partition_t** partitions,
    int* num_partitions);

/**
 * Gather distributed state
 */
bool dist_gather_state(
    dist_executor_t* executor,
    const dist_partition_t* partitions,
    int num_partitions,
    void* output,
    size_t output_size);

/**
 * Redistribute state
 */
bool dist_redistribute(
    dist_executor_t* executor,
    dist_partition_t* partitions,
    int num_partitions,
    dist_strategy_t new_strategy);

/**
 * Exchange boundaries between partitions
 */
bool dist_exchange_boundaries(
    dist_executor_t* executor,
    dist_partition_t* partitions,
    int num_partitions,
    size_t boundary_size);

// ============================================================================
// Task Execution
// ============================================================================

/**
 * Submit single task
 */
uint64_t dist_submit_task(
    dist_executor_t* executor,
    const dist_task_spec_t* task);

/**
 * Submit batch of tasks
 */
bool dist_submit_tasks(
    dist_executor_t* executor,
    const dist_task_spec_t* tasks,
    size_t num_tasks,
    uint64_t* task_ids);

/**
 * Wait for task completion
 */
bool dist_wait_task(
    dist_executor_t* executor,
    uint64_t task_id,
    dist_task_result_t* result);

/**
 * Wait for multiple tasks
 */
bool dist_wait_tasks(
    dist_executor_t* executor,
    const uint64_t* task_ids,
    size_t num_tasks,
    dist_task_result_t** results);

/**
 * Wait for all tasks
 */
bool dist_wait_all(dist_executor_t* executor);

/**
 * Cancel task
 */
bool dist_cancel_task(
    dist_executor_t* executor,
    uint64_t task_id);

/**
 * Get task status
 */
dist_task_state_t dist_get_task_state(
    dist_executor_t* executor,
    uint64_t task_id);

// ============================================================================
// Communication
// ============================================================================

/**
 * Send data to node
 */
bool dist_send(
    dist_executor_t* executor,
    int dest_node,
    const void* data,
    size_t size,
    int tag);

/**
 * Receive data from node
 */
bool dist_recv(
    dist_executor_t* executor,
    int src_node,
    void* data,
    size_t size,
    int tag);

/**
 * Broadcast data to all nodes
 */
bool dist_broadcast(
    dist_executor_t* executor,
    void* data,
    size_t size,
    int root_node);

/**
 * Gather data from all nodes
 */
bool dist_gather(
    dist_executor_t* executor,
    const void* send_data,
    size_t send_size,
    void* recv_data,
    size_t recv_size,
    int root_node);

/**
 * Scatter data to all nodes
 */
bool dist_scatter(
    dist_executor_t* executor,
    const void* send_data,
    size_t send_size,
    void* recv_data,
    size_t recv_size,
    int root_node);

/**
 * All-reduce operation
 */
bool dist_all_reduce(
    dist_executor_t* executor,
    const void* send_data,
    void* recv_data,
    size_t count,
    size_t element_size,
    void (*reduce_op)(const void*, const void*, void*, size_t));

/**
 * Barrier synchronization
 */
bool dist_barrier(dist_executor_t* executor);

// ============================================================================
// Fault Tolerance
// ============================================================================

/**
 * Create checkpoint
 */
bool dist_create_checkpoint(
    dist_executor_t* executor,
    const void* state,
    size_t state_size,
    const char* description,
    uint64_t* checkpoint_id);

/**
 * Restore from checkpoint
 */
bool dist_restore_checkpoint(
    dist_executor_t* executor,
    uint64_t checkpoint_id,
    void* state,
    size_t state_size);

/**
 * List checkpoints
 */
bool dist_list_checkpoints(
    dist_executor_t* executor,
    dist_checkpoint_t** checkpoints,
    size_t* count);

/**
 * Delete checkpoint
 */
bool dist_delete_checkpoint(
    dist_executor_t* executor,
    uint64_t checkpoint_id);

/**
 * Handle node failure
 */
bool dist_handle_failure(
    dist_executor_t* executor,
    int failed_node);

// ============================================================================
// Load Balancing
// ============================================================================

/**
 * Rebalance workload
 */
bool dist_rebalance(dist_executor_t* executor);

/**
 * Get load statistics
 */
bool dist_get_load_stats(
    dist_executor_t* executor,
    double* node_loads,
    int* num_nodes);

/**
 * Set node weight
 */
bool dist_set_node_weight(
    dist_executor_t* executor,
    int node_id,
    double weight);

// ============================================================================
// Statistics and Monitoring
// ============================================================================

/**
 * Get execution statistics
 */
bool dist_get_stats(
    dist_executor_t* executor,
    dist_exec_stats_t* stats);

/**
 * Reset statistics
 */
void dist_reset_stats(dist_executor_t* executor);

/**
 * Get communication statistics
 */
bool dist_get_comm_stats(
    dist_executor_t* executor,
    dist_comm_stats_t* stats);

// ============================================================================
// Reporting
// ============================================================================

/**
 * Generate execution report
 */
char* dist_generate_report(dist_executor_t* executor);

/**
 * Export to JSON
 */
char* dist_export_json(dist_executor_t* executor);

/**
 * Export to file
 */
bool dist_export_to_file(
    dist_executor_t* executor,
    const char* filename);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Get strategy name
 */
const char* dist_strategy_name(dist_strategy_t strategy);

/**
 * Get role name
 */
const char* dist_role_name(dist_role_t role);

/**
 * Get task state name
 */
const char* dist_task_state_name(dist_task_state_t state);

/**
 * Get communication pattern name
 */
const char* dist_comm_pattern_name(dist_comm_pattern_t pattern);

/**
 * Get fault mode name
 */
const char* dist_fault_mode_name(dist_fault_mode_t mode);

/**
 * Get balance strategy name
 */
const char* dist_balance_name(dist_balance_t balance);

/**
 * Free node info array
 */
void dist_free_nodes(dist_node_info_t* nodes);

/**
 * Free partitions
 */
void dist_free_partitions(dist_partition_t* partitions, int count);

/**
 * Free task results
 */
void dist_free_results(dist_task_result_t* results, size_t count);

/**
 * Free checkpoints
 */
void dist_free_checkpoints(dist_checkpoint_t* checkpoints, size_t count);

/**
 * Get last error message
 */
const char* dist_get_last_error(dist_executor_t* executor);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_DISTRIBUTED_EXECUTION_H
