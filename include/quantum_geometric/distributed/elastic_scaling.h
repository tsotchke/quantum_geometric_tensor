#ifndef ELASTIC_SCALING_H
#define ELASTIC_SCALING_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

// MPI forward declarations for non-MPI builds
#ifndef HAS_MPI
#ifndef NO_MPI
#define NO_MPI
#endif
#endif

#ifdef NO_MPI
typedef int MPI_Comm;
#define MPI_COMM_NULL 0
#define MPI_COMM_WORLD 0
#else
#include <mpi.h>
#endif

// Elastic scaling parameters
#define ELASTIC_MIN_WORKERS 1
#define ELASTIC_MAX_WORKERS 128
#define ELASTIC_SCALING_INTERVAL 30  // seconds
#define ELASTIC_LOAD_THRESHOLD 0.8
#define ELASTIC_SCALE_FACTOR 1.5

// Node status structure
typedef struct NodeStatus {
    int rank;                    // MPI rank of the node
    double cpu_util;             // CPU utilization (0-1)
    double gpu_util;             // GPU utilization (0-1)
    double memory_util;          // Memory utilization (0-1)
    double network_util;         // Network utilization (0-1)
    bool is_active;              // Whether node is currently active
    time_t last_heartbeat;       // Last heartbeat timestamp
} NodeStatus;

// Elastic manager structure
typedef struct ElasticManager {
    NodeStatus* nodes;           // Array of node status
    size_t num_nodes;            // Total number of nodes
    size_t active_nodes;         // Currently active nodes
    pthread_t monitor_thread;    // Resource monitoring thread
    pthread_mutex_t mutex;       // Thread synchronization
    bool running;                // Whether manager is running
    MPI_Comm scale_comm;         // Communicator for scaling operations
    int world_rank;              // This node's rank
    int world_size;              // Total world size
} ElasticManager;

// Public API - Create and destroy
ElasticManager* init_elastic_scaling(void);
void cleanup_elastic_scaling(ElasticManager* manager);

// Helper macros for min/max if not defined
#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifdef __cplusplus
}
#endif

#endif // ELASTIC_SCALING_H
