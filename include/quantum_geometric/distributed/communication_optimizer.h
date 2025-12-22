#ifndef COMMUNICATION_OPTIMIZER_H
#define COMMUNICATION_OPTIMIZER_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct CommunicationOptimizer CommunicationOptimizer;
typedef struct TopologyInfo TopologyInfo;
typedef struct CommScheduler CommScheduler;

// Configuration structure
typedef struct CommConfig {
    size_t buffer_size;
    size_t min_message_size;
    int max_concurrent;
    bool enable_compression;
    bool enable_topology_aware;
    bool use_pinned_memory;
    bool numa_aware;
    bool topology_aware;
    int numa_policy;
} CommConfig;

// Collective operation type
typedef enum CollectiveOperation {
    COLLECTIVE_ALLREDUCE,
    COLLECTIVE_BROADCAST,
    COLLECTIVE_ALLGATHER,
    COLLECTIVE_REDUCE_SCATTER
} CollectiveOperation;

// Collective algorithm selection
typedef enum CollectiveAlgorithm {
    BINOMIAL_TREE,
    RECURSIVE_DOUBLING,
    RING,
    HYBRID
} CollectiveAlgorithm;

// Gradient synchronization algorithm
typedef enum SyncAlgorithm {
    SYNC_ALLREDUCE,
    HIERARCHICAL,
    BUTTERFLY,
    PIPELINED
} SyncAlgorithm;

// Gradient buffer structure
typedef struct GradientBuffer {
    void* data;
    size_t size;
    size_t count;
    int dtype;
    bool is_compressed;
} GradientBuffer;

// Initialize communication optimizer
CommunicationOptimizer* init_communication_optimizer(const CommConfig* config);

// Cleanup
void cleanup_communication_optimizer(CommunicationOptimizer* optimizer);

// Algorithm selection
CollectiveAlgorithm select_collective_algorithm(
    CommunicationOptimizer* optimizer,
    CollectiveOperation op,
    size_t message_size,
    int datatype);

// Point-to-point communication
int optimized_send_recv(
    CommunicationOptimizer* optimizer,
    void* send_data,
    size_t send_size,
    int dest,
    void* recv_data,
    size_t recv_size,
    int source);

// Gradient synchronization
int sync_gradients(
    CommunicationOptimizer* optimizer,
    GradientBuffer* gradients,
    size_t num_buffers);

// NUMA-aware operations
int numa_aware_allreduce(
    CommunicationOptimizer* optimizer,
    void* data,
    size_t count,
    int datatype);

// Topology initialization (internal)
TopologyInfo* init_topology_info(void);
void cleanup_topology_info(TopologyInfo* info);

#ifdef __cplusplus
}
#endif

#endif // COMMUNICATION_OPTIMIZER_H
