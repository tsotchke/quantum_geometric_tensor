#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#include "quantum_geometric/distributed/communication_optimizer.h"
#include "quantum_geometric/distributed/gradient_optimizer.h"

// Define NO_MPI and NO_HWLOC by default since these are optional
#ifndef HAS_MPI
#define NO_MPI
#endif
#ifndef HAS_HWLOC
#define NO_HWLOC
#endif

#ifndef NO_MPI
#include <mpi.h>
#endif
#ifndef NO_HWLOC
#include <hwloc.h>
#else
// Stub for hwloc topology type
typedef void* hwloc_topology_t;
#endif

#ifdef NO_MPI
typedef int MPI_Request;
typedef int MPI_Comm;
typedef int MPI_Datatype;
#define MPI_COMM_WORLD 0
#define MPI_COMM_NULL 0
#define MPI_BYTE 0
#endif

// Communication parameters
#define MAX_BUFFER_SIZE (1 << 30)  // 1GB
#define MIN_MESSAGE_SIZE (1 << 16)  // 64KB
#define MAX_CONCURRENT_COMMS 16
#define TOPOLOGY_LEVELS 4

// Communication buffer
typedef struct CommBuffer {
    void* data;
    size_t size;
    bool is_pinned;
    int device_id;
} CommBuffer;

// Topology information struct (complete definition)
struct TopologyInfo {
    hwloc_topology_t topology;
    int num_nodes;
    int* node_distances;
    int* numa_mapping;
    bool is_initialized;
};

// Communication scheduler struct (complete definition)
struct CommScheduler {
    MPI_Request* active_requests;
    size_t num_requests;
    double* bandwidth_matrix;
    double* latency_matrix;
    bool* is_busy;
};

// Communication optimizer struct (complete definition)
struct CommunicationOptimizer {
    TopologyInfo* topology;
    CommBuffer* send_buffer;
    CommBuffer* recv_buffer;
    CommScheduler* scheduler;
    MPI_Comm node_comm;
    MPI_Comm inter_node_comm;
    int world_rank;
    int world_size;
    int node_rank;
    int node_size;
};

// Forward declarations for internal functions
static CommBuffer* create_comm_buffer(size_t size, bool use_pinned);
static CommScheduler* create_comm_scheduler(int max_concurrent);
static void setup_communicators(CommunicationOptimizer* optimizer, const CommConfig* config);
static void cleanup_comm_buffer(CommBuffer* buffer);
static void cleanup_comm_scheduler(CommScheduler* scheduler);

// Initialize communication optimizer
CommunicationOptimizer* init_communication_optimizer(
    const CommConfig* config) {

    CommunicationOptimizer* optimizer = aligned_alloc(64,
        sizeof(CommunicationOptimizer));
    if (!optimizer) return NULL;

    memset(optimizer, 0, sizeof(CommunicationOptimizer));

    // Initialize topology
    optimizer->topology = init_topology_info();

    // Create communication buffers
    optimizer->send_buffer = create_comm_buffer(
        config->buffer_size, config->use_pinned_memory);
    optimizer->recv_buffer = create_comm_buffer(
        config->buffer_size, config->use_pinned_memory);

    // Initialize scheduler
    optimizer->scheduler = create_comm_scheduler(
        config->max_concurrent);

    // Setup MPI communicators
    setup_communicators(optimizer, config);

#ifndef NO_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &optimizer->world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &optimizer->world_size);
#else
    optimizer->world_rank = 0;
    optimizer->world_size = 1;
#endif

    return optimizer;
}

// Initialize topology information
TopologyInfo* init_topology_info(void) {
    TopologyInfo* info = aligned_alloc(64, sizeof(TopologyInfo));
    if (!info) return NULL;

    memset(info, 0, sizeof(TopologyInfo));

#ifndef NO_HWLOC
    // Initialize hwloc topology
    if (hwloc_topology_init(&info->topology) < 0) {
        free(info);
        return NULL;
    }

    if (hwloc_topology_load(info->topology) < 0) {
        hwloc_topology_destroy(info->topology);
        free(info);
        return NULL;
    }

    // Get node information
    info->num_nodes = hwloc_get_nbobjs_by_type(info->topology,
                                              HWLOC_OBJ_NUMANODE);
    if (info->num_nodes == 0) info->num_nodes = 1;

    // Allocate distance matrices
    info->node_distances = aligned_alloc(64,
        info->num_nodes * info->num_nodes * sizeof(int));
    info->numa_mapping = aligned_alloc(64,
        info->num_nodes * sizeof(int));

    if (info->node_distances && info->numa_mapping) {
        // Initialize with uniform distances
        for (int i = 0; i < info->num_nodes * info->num_nodes; i++) {
            info->node_distances[i] = (i % (info->num_nodes + 1) == 0) ? 0 : 1;
        }
        for (int i = 0; i < info->num_nodes; i++) {
            info->numa_mapping[i] = i;
        }
    }
#else
    // Stub implementation without hwloc
    info->topology = NULL;
    info->num_nodes = 1;
    info->node_distances = aligned_alloc(64, sizeof(int));
    info->numa_mapping = aligned_alloc(64, sizeof(int));
    if (info->node_distances) info->node_distances[0] = 0;
    if (info->numa_mapping) info->numa_mapping[0] = 0;
#endif

    info->is_initialized = true;
    return info;
}

// Setup optimized communicators
static void setup_communicators(
    CommunicationOptimizer* optimizer,
    const CommConfig* config) {
    (void)config;  // May be unused in stub

#ifndef NO_MPI
    // Create node-local communicator
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &optimizer->node_comm);
    MPI_Comm_rank(optimizer->node_comm, &optimizer->node_rank);
    MPI_Comm_size(optimizer->node_comm, &optimizer->node_size);
#else
    optimizer->node_comm = 0;
    optimizer->inter_node_comm = 0;
    optimizer->node_rank = 0;
    optimizer->node_size = 1;
#endif
}

// Algorithm selection
CollectiveAlgorithm select_collective_algorithm(
    CommunicationOptimizer* optimizer,
    CollectiveOperation op,
    size_t message_size,
    int datatype) {
    (void)optimizer;
    (void)datatype;

    // Simple heuristic based on message size and operation
    if (message_size < MIN_MESSAGE_SIZE) {
        return BINOMIAL_TREE;
    } else if (op == COLLECTIVE_ALLREDUCE && message_size > MAX_BUFFER_SIZE / 4) {
        return RING;
    } else if (message_size > MAX_BUFFER_SIZE / 2) {
        return RECURSIVE_DOUBLING;
    }
    return HYBRID;
}

// Point-to-point communication
int optimized_send_recv(
    CommunicationOptimizer* optimizer,
    void* send_data,
    size_t send_size,
    int dest,
    void* recv_data,
    size_t recv_size,
    int source) {
#ifndef NO_MPI
    MPI_Status status;
    return MPI_Sendrecv(send_data, send_size, MPI_BYTE, dest, 0,
                        recv_data, recv_size, MPI_BYTE, source, 0,
                        MPI_COMM_WORLD, &status);
#else
    (void)optimizer;
    (void)dest;
    (void)source;
    // Stub: just copy send to recv if same process
    if (send_data && recv_data && send_size <= recv_size) {
        memcpy(recv_data, send_data, send_size);
    }
    return 0;
#endif
}

// Gradient synchronization
int sync_gradients(
    CommunicationOptimizer* optimizer,
    GradientBuffer* gradients,
    size_t num_buffers) {
#ifndef NO_MPI
    for (size_t i = 0; i < num_buffers; i++) {
        MPI_Allreduce(MPI_IN_PLACE, gradients[i].data,
                      gradients[i].count, MPI_FLOAT,
                      MPI_SUM, MPI_COMM_WORLD);
    }
    return 0;
#else
    (void)optimizer;
    (void)gradients;
    (void)num_buffers;
    return 0;  // Single process, no sync needed
#endif
}

// NUMA-aware allreduce
int numa_aware_allreduce(
    CommunicationOptimizer* optimizer,
    void* data,
    size_t count,
    int datatype) {
#ifndef NO_MPI
    (void)datatype;
    // First reduce within NUMA node, then across nodes
    MPI_Allreduce(MPI_IN_PLACE, data, count, MPI_FLOAT, MPI_SUM,
                  optimizer->node_comm);
    // Cross-node reduction (simplified)
    MPI_Allreduce(MPI_IN_PLACE, data, count, MPI_FLOAT, MPI_SUM,
                  MPI_COMM_WORLD);
    return 0;
#else
    (void)optimizer;
    (void)data;
    (void)count;
    (void)datatype;
    return 0;
#endif
}

// Helper: create communication buffer
static CommBuffer* create_comm_buffer(size_t size, bool use_pinned) {
    CommBuffer* buffer = aligned_alloc(64, sizeof(CommBuffer));
    if (!buffer) return NULL;

    buffer->size = size;
    buffer->is_pinned = use_pinned;
    buffer->device_id = -1;

    if (use_pinned) {
        // Try pinned allocation
        buffer->data = aligned_alloc(4096, size);
    } else {
        buffer->data = aligned_alloc(64, size);
    }

    if (!buffer->data) {
        free(buffer);
        return NULL;
    }
    return buffer;
}

// Helper: create scheduler
static CommScheduler* create_comm_scheduler(int max_concurrent) {
    CommScheduler* scheduler = aligned_alloc(64, sizeof(CommScheduler));
    if (!scheduler) return NULL;

    scheduler->active_requests = aligned_alloc(64, max_concurrent * sizeof(MPI_Request));
    scheduler->num_requests = 0;
    scheduler->bandwidth_matrix = NULL;
    scheduler->latency_matrix = NULL;
    scheduler->is_busy = calloc(max_concurrent, sizeof(bool));

    return scheduler;
}

// Helper: cleanup buffer
static void cleanup_comm_buffer(CommBuffer* buffer) {
    if (buffer) {
        free(buffer->data);
        free(buffer);
    }
}

// Helper: cleanup scheduler
static void cleanup_comm_scheduler(CommScheduler* scheduler) {
    if (scheduler) {
        free(scheduler->active_requests);
        free(scheduler->bandwidth_matrix);
        free(scheduler->latency_matrix);
        free(scheduler->is_busy);
        free(scheduler);
    }
}

// Cleanup topology info
void cleanup_topology_info(TopologyInfo* info) {
    if (info) {
#ifndef NO_HWLOC
        if (info->topology) {
            hwloc_topology_destroy(info->topology);
        }
#endif
        free(info->node_distances);
        free(info->numa_mapping);
        free(info);
    }
}

// Clean up optimizer
void cleanup_communication_optimizer(
    CommunicationOptimizer* optimizer) {

    if (!optimizer) return;

    // Clean up topology
    cleanup_topology_info(optimizer->topology);

    // Clean up buffers
    cleanup_comm_buffer(optimizer->send_buffer);
    cleanup_comm_buffer(optimizer->recv_buffer);

    // Clean up scheduler
    cleanup_comm_scheduler(optimizer->scheduler);

#ifndef NO_MPI
    // Clean up communicators
    if (optimizer->node_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&optimizer->node_comm);
    }
#endif

    free(optimizer);
}
