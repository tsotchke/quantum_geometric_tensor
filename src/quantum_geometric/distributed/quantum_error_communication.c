/**
 * @file quantum_error_communication.c
 * @brief Distributed Error Communication for Quantum Computing Systems
 *
 * Complete implementation of error propagation and communication across
 * distributed quantum computing nodes with MPI support.
 */

#include "quantum_geometric/distributed/quantum_error_communication.h"

#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <pthread.h>
#include <stdatomic.h>
#include <math.h>

// MPI detection and stubs
#ifndef HAS_MPI
#ifndef NO_MPI
#define NO_MPI
#endif
#endif

#ifndef NO_MPI
#include <mpi.h>
#else
// MPI type stubs for non-MPI builds
typedef int MPI_Comm;
typedef int MPI_Status;
typedef int MPI_Datatype;
typedef int MPI_Request;
#define MPI_COMM_WORLD 0
#define MPI_BYTE 0
#define MPI_DOUBLE 0
#define MPI_INT 0
#define MPI_UINT64_T 0
#define MPI_SUCCESS 0
#define MPI_ANY_SOURCE -1
#define MPI_STATUS_IGNORE NULL
#define MPI_REQUEST_NULL 0

static inline int MPI_Comm_rank(MPI_Comm comm, int* rank) { (void)comm; *rank = 0; return 0; }
static inline int MPI_Comm_size(MPI_Comm comm, int* size) { (void)comm; *size = 1; return 0; }
static inline int MPI_Barrier(MPI_Comm comm) { (void)comm; return 0; }
static inline int MPI_Bcast(void* buf, int count, MPI_Datatype dt, int root, MPI_Comm comm) {
    (void)buf; (void)count; (void)dt; (void)root; (void)comm; return 0;
}
static inline int MPI_Send(const void* buf, int count, MPI_Datatype dt, int dest, int tag, MPI_Comm comm) {
    (void)buf; (void)count; (void)dt; (void)dest; (void)tag; (void)comm; return 0;
}
static inline int MPI_Recv(void* buf, int count, MPI_Datatype dt, int src, int tag, MPI_Comm comm, MPI_Status* st) {
    (void)buf; (void)count; (void)dt; (void)src; (void)tag; (void)comm; (void)st; return 0;
}
static inline int MPI_Isend(const void* buf, int count, MPI_Datatype dt, int dest, int tag, MPI_Comm comm, MPI_Request* req) {
    (void)buf; (void)count; (void)dt; (void)dest; (void)tag; (void)comm; *req = 0; return 0;
}
static inline int MPI_Irecv(void* buf, int count, MPI_Datatype dt, int src, int tag, MPI_Comm comm, MPI_Request* req) {
    (void)buf; (void)count; (void)dt; (void)src; (void)tag; (void)comm; *req = 0; return 0;
}
static inline int MPI_Wait(MPI_Request* req, MPI_Status* st) { (void)req; (void)st; return 0; }
static inline int MPI_Test(MPI_Request* req, int* flag, MPI_Status* st) { (void)req; *flag = 1; (void)st; return 0; }
static inline int MPI_Gather(const void* sb, int sc, MPI_Datatype st, void* rb, int rc, MPI_Datatype rt, int root, MPI_Comm comm) {
    (void)sb; (void)sc; (void)st; (void)rb; (void)rc; (void)rt; (void)root; (void)comm; return 0;
}
static inline int MPI_Allgather(const void* sb, int sc, MPI_Datatype st, void* rb, int rc, MPI_Datatype rt, MPI_Comm comm) {
    (void)sb; (void)sc; (void)st; (void)rb; (void)rc; (void)rt; (void)comm; return 0;
}
static inline int MPI_Allreduce(const void* sb, void* rb, int c, MPI_Datatype dt, int op, MPI_Comm comm) {
    (void)sb; (void)rb; (void)c; (void)dt; (void)op; (void)comm; return 0;
}
static inline int MPI_Iprobe(int src, int tag, MPI_Comm comm, int* flag, MPI_Status* st) {
    (void)src; (void)tag; (void)comm; *flag = 0; (void)st; return 0;
}
#define MPI_SUM 0
#define MPI_MAX 1
#define MPI_MIN 2
#endif

// ============================================================================
// Internal Constants
// ============================================================================

#define MAX_CALLBACKS 16
#define MAX_PENDING_EVENTS 256
#define DEFAULT_HISTORY_SIZE 1000
#define STATS_WINDOW_SIZE 100

// MPI Tags
#define TAG_SYNDROME (QEC_COMM_TAG_BASE + 0)
#define TAG_ERROR_EVENT (QEC_COMM_TAG_BASE + 1)
#define TAG_RECOVERY (QEC_COMM_TAG_BASE + 2)
#define TAG_RECOVERY_STATUS (QEC_COMM_TAG_BASE + 3)
#define TAG_NODE_INFO (QEC_COMM_TAG_BASE + 4)

// ============================================================================
// Internal Structures
// ============================================================================

// Callback registration
typedef struct {
    qec_error_callback_t callback;
    void* user_data;
    qec_severity_t min_severity;
    bool active;
} error_callback_entry_t;

typedef struct {
    qec_syndrome_callback_t callback;
    void* user_data;
    bool active;
} syndrome_callback_entry_t;

typedef struct {
    qec_recovery_callback_t callback;
    void* user_data;
    bool active;
} recovery_callback_entry_t;

// Recovery tracking
typedef struct {
    uint64_t event_id;
    bool* node_complete;
    size_t num_nodes;
    bool all_complete;
} recovery_tracker_t;

// Manager implementation
struct qec_comm_manager {
    // Configuration
    qec_comm_config_t config;

    // MPI state
    MPI_Comm comm;
    int rank;
    int world_size;
    bool initialized;

    // Node information
    qec_node_info_t local_node;
    qec_node_info_t* all_nodes;
    size_t num_nodes;

    // Callbacks
    error_callback_entry_t error_callbacks[MAX_CALLBACKS];
    size_t num_error_callbacks;
    syndrome_callback_entry_t syndrome_callbacks[MAX_CALLBACKS];
    size_t num_syndrome_callbacks;
    recovery_callback_entry_t recovery_callbacks[MAX_CALLBACKS];
    size_t num_recovery_callbacks;

    // Event queues
    qec_error_event_t* pending_errors;
    size_t num_pending_errors;
    size_t pending_errors_capacity;

    // Error history
    qec_error_event_t* error_history;
    size_t history_size;
    size_t history_capacity;
    bool logging_enabled;

    // Statistics
    qec_comm_stats_t stats;
    double* latency_samples;
    size_t latency_count;

    // Recovery tracking
    recovery_tracker_t* recovery_trackers;
    size_t num_trackers;
    size_t trackers_capacity;

    // Event ID generation
    atomic_uint_least64_t next_event_id;

    // Synchronization
    pthread_mutex_t mutex;
    pthread_mutex_t stats_mutex;

    // Error message
    char last_error[512];
};

// ============================================================================
// Utility Functions
// ============================================================================

static uint64_t get_timestamp_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static void set_error(qec_comm_manager_t* manager, const char* msg) {
    if (manager && msg) {
        strncpy(manager->last_error, msg, sizeof(manager->last_error) - 1);
        manager->last_error[sizeof(manager->last_error) - 1] = '\0';
    }
}

// ============================================================================
// Name Lookup Functions
// ============================================================================

const char* qec_error_type_name(qec_error_type_t type) {
    switch (type) {
        case QEC_ERROR_BIT_FLIP: return "Bit Flip (X)";
        case QEC_ERROR_PHASE_FLIP: return "Phase Flip (Z)";
        case QEC_ERROR_BIT_PHASE_FLIP: return "Bit-Phase Flip (Y)";
        case QEC_ERROR_DEPOLARIZING: return "Depolarizing";
        case QEC_ERROR_AMPLITUDE_DAMPING: return "Amplitude Damping";
        case QEC_ERROR_PHASE_DAMPING: return "Phase Damping";
        case QEC_ERROR_MEASUREMENT: return "Measurement";
        case QEC_ERROR_LEAKAGE: return "Leakage";
        case QEC_ERROR_CROSSTALK: return "Crosstalk";
        case QEC_ERROR_COSMIC_RAY: return "Cosmic Ray";
        case QEC_ERROR_CLASSICAL: return "Classical";
        case QEC_ERROR_COMMUNICATION: return "Communication";
        default: return "Unknown";
    }
}

const char* qec_severity_name(qec_severity_t severity) {
    switch (severity) {
        case QEC_SEVERITY_DEBUG: return "DEBUG";
        case QEC_SEVERITY_INFO: return "INFO";
        case QEC_SEVERITY_WARNING: return "WARNING";
        case QEC_SEVERITY_ERROR: return "ERROR";
        case QEC_SEVERITY_CRITICAL: return "CRITICAL";
        case QEC_SEVERITY_FATAL: return "FATAL";
        default: return "UNKNOWN";
    }
}

const char* qec_comm_pattern_name(qec_comm_pattern_t pattern) {
    switch (pattern) {
        case QEC_COMM_BROADCAST: return "Broadcast";
        case QEC_COMM_GATHER: return "Gather";
        case QEC_COMM_SCATTER: return "Scatter";
        case QEC_COMM_ALLGATHER: return "AllGather";
        case QEC_COMM_ALLREDUCE: return "AllReduce";
        case QEC_COMM_NEIGHBOR: return "Neighbor";
        case QEC_COMM_RING: return "Ring";
        case QEC_COMM_TREE: return "Tree";
        case QEC_COMM_POINT_TO_POINT: return "Point-to-Point";
        default: return "Unknown";
    }
}

const char* qec_aggregate_method_name(qec_aggregate_method_t method) {
    switch (method) {
        case QEC_AGGREGATE_NONE: return "None";
        case QEC_AGGREGATE_OR: return "OR";
        case QEC_AGGREGATE_AND: return "AND";
        case QEC_AGGREGATE_XOR: return "XOR";
        case QEC_AGGREGATE_MAJORITY: return "Majority";
        case QEC_AGGREGATE_WEIGHTED: return "Weighted";
        case QEC_AGGREGATE_CUSTOM: return "Custom";
        default: return "Unknown";
    }
}

const char* qec_recovery_mode_name(qec_recovery_mode_t mode) {
    switch (mode) {
        case QEC_RECOVERY_LOCAL: return "Local";
        case QEC_RECOVERY_COORDINATED: return "Coordinated";
        case QEC_RECOVERY_HIERARCHICAL: return "Hierarchical";
        case QEC_RECOVERY_CONSENSUS: return "Consensus";
        default: return "Unknown";
    }
}

const char* qec_delivery_name(qec_delivery_t delivery) {
    switch (delivery) {
        case QEC_DELIVERY_BEST_EFFORT: return "Best Effort";
        case QEC_DELIVERY_AT_LEAST_ONCE: return "At Least Once";
        case QEC_DELIVERY_EXACTLY_ONCE: return "Exactly Once";
        case QEC_DELIVERY_ORDERED: return "Ordered";
        default: return "Unknown";
    }
}

// ============================================================================
// Initialization and Configuration
// ============================================================================

qec_comm_config_t qec_comm_default_config(void) {
    qec_comm_config_t config = {
        .default_pattern = QEC_COMM_BROADCAST,
        .syndrome_agg = QEC_AGGREGATE_MAJORITY,
        .recovery_mode = QEC_RECOVERY_COORDINATED,
        .delivery_guarantee = QEC_DELIVERY_AT_LEAST_ONCE,
        .enable_compression = false,
        .enable_batching = true,
        .batch_size = 16,
        .batch_timeout_ms = 10.0,
        .enable_encryption = false,
        .max_retries = 3,
        .retry_timeout_ms = 100.0,
        .history_size = DEFAULT_HISTORY_SIZE
    };
    return config;
}

qec_comm_manager_t* qec_comm_create(void) {
    return qec_comm_create_with_config(NULL);
}

qec_comm_manager_t* qec_comm_create_with_config(const qec_comm_config_t* config) {
    qec_comm_manager_t* manager = calloc(1, sizeof(qec_comm_manager_t));
    if (!manager) return NULL;

    // Set configuration
    if (config) {
        manager->config = *config;
    } else {
        manager->config = qec_comm_default_config();
    }

    // Initialize mutexes
    pthread_mutex_init(&manager->mutex, NULL);
    pthread_mutex_init(&manager->stats_mutex, NULL);

    // Allocate pending events queue
    manager->pending_errors_capacity = MAX_PENDING_EVENTS;
    manager->pending_errors = calloc(manager->pending_errors_capacity,
                                     sizeof(qec_error_event_t));

    // Allocate error history
    manager->history_capacity = manager->config.history_size;
    manager->error_history = calloc(manager->history_capacity,
                                    sizeof(qec_error_event_t));

    // Allocate latency samples
    manager->latency_samples = calloc(STATS_WINDOW_SIZE, sizeof(double));

    // Allocate recovery trackers
    manager->trackers_capacity = 64;
    manager->recovery_trackers = calloc(manager->trackers_capacity,
                                        sizeof(recovery_tracker_t));

    // Initialize event ID
    atomic_init(&manager->next_event_id, 1);

    manager->logging_enabled = true;
    manager->comm = MPI_COMM_WORLD;

    return manager;
}

void qec_comm_destroy(qec_comm_manager_t* manager) {
    if (!manager) return;

    if (manager->initialized) {
        qec_comm_finalize(manager);
    }

    // Free node information
    if (manager->all_nodes) {
        for (size_t i = 0; i < manager->num_nodes; i++) {
            free(manager->all_nodes[i].neighbors);
        }
        free(manager->all_nodes);
    }
    free(manager->local_node.neighbors);

    // Free pending errors
    if (manager->pending_errors) {
        for (size_t i = 0; i < manager->num_pending_errors; i++) {
            free(manager->pending_errors[i].affected_nodes);
            free(manager->pending_errors[i].qubit_indices);
        }
        free(manager->pending_errors);
    }

    // Free error history
    if (manager->error_history) {
        for (size_t i = 0; i < manager->history_size; i++) {
            free(manager->error_history[i].affected_nodes);
            free(manager->error_history[i].qubit_indices);
        }
        free(manager->error_history);
    }

    // Free latency samples
    free(manager->latency_samples);

    // Free recovery trackers
    if (manager->recovery_trackers) {
        for (size_t i = 0; i < manager->num_trackers; i++) {
            free(manager->recovery_trackers[i].node_complete);
        }
        free(manager->recovery_trackers);
    }

    pthread_mutex_destroy(&manager->mutex);
    pthread_mutex_destroy(&manager->stats_mutex);

    free(manager);
}

bool qec_comm_init(qec_comm_manager_t* manager) {
    if (!manager) return false;

    pthread_mutex_lock(&manager->mutex);

    // Get MPI rank and size
    int result = MPI_Comm_rank(manager->comm, &manager->rank);
    if (result != MPI_SUCCESS) {
        set_error(manager, "Failed to get MPI rank");
        pthread_mutex_unlock(&manager->mutex);
        return false;
    }

    result = MPI_Comm_size(manager->comm, &manager->world_size);
    if (result != MPI_SUCCESS) {
        set_error(manager, "Failed to get MPI world size");
        pthread_mutex_unlock(&manager->mutex);
        return false;
    }

    // Initialize local node info
    manager->local_node.node_id = manager->rank;
    manager->local_node.rank = manager->rank;
    snprintf(manager->local_node.name, QEC_COMM_MAX_NAME_LENGTH,
             "node_%d", manager->rank);
    manager->local_node.is_active = true;

    // Allocate node array
    manager->all_nodes = calloc(manager->world_size, sizeof(qec_node_info_t));
    if (!manager->all_nodes) {
        set_error(manager, "Failed to allocate node array");
        pthread_mutex_unlock(&manager->mutex);
        return false;
    }
    manager->num_nodes = manager->world_size;

    // Initialize all nodes with default values
    for (int i = 0; i < manager->world_size; i++) {
        manager->all_nodes[i].node_id = i;
        manager->all_nodes[i].rank = i;
        snprintf(manager->all_nodes[i].name, QEC_COMM_MAX_NAME_LENGTH,
                 "node_%d", i);
        manager->all_nodes[i].is_active = true;
    }

    manager->initialized = true;

    pthread_mutex_unlock(&manager->mutex);
    return true;
}

bool qec_comm_finalize(qec_comm_manager_t* manager) {
    if (!manager || !manager->initialized) return false;

    pthread_mutex_lock(&manager->mutex);

    // Barrier to ensure all pending messages are processed
    MPI_Barrier(manager->comm);

    manager->initialized = false;

    pthread_mutex_unlock(&manager->mutex);
    return true;
}

bool qec_comm_reset(qec_comm_manager_t* manager) {
    if (!manager) return false;

    pthread_mutex_lock(&manager->mutex);

    // Clear pending errors
    manager->num_pending_errors = 0;

    // Clear history
    manager->history_size = 0;

    // Reset statistics
    memset(&manager->stats, 0, sizeof(qec_comm_stats_t));
    manager->latency_count = 0;

    // Clear callbacks
    manager->num_error_callbacks = 0;
    manager->num_syndrome_callbacks = 0;
    manager->num_recovery_callbacks = 0;

    // Reset event ID
    atomic_store(&manager->next_event_id, 1);

    pthread_mutex_unlock(&manager->mutex);
    return true;
}

// ============================================================================
// Node Registration
// ============================================================================

bool qec_comm_register_node(qec_comm_manager_t* manager,
                            const qec_node_info_t* info) {
    if (!manager || !info || !manager->initialized) return false;

    pthread_mutex_lock(&manager->mutex);

    // Update local node info
    memcpy(&manager->local_node, info, sizeof(qec_node_info_t));

    // Copy neighbors if provided
    if (info->neighbors && info->num_neighbors > 0) {
        manager->local_node.neighbors = malloc(info->num_neighbors * sizeof(int));
        if (manager->local_node.neighbors) {
            memcpy(manager->local_node.neighbors, info->neighbors,
                   info->num_neighbors * sizeof(int));
            manager->local_node.num_neighbors = info->num_neighbors;
        }
    }

    // Update in all_nodes array
    if (manager->rank < (int)manager->num_nodes) {
        memcpy(&manager->all_nodes[manager->rank], &manager->local_node,
               sizeof(qec_node_info_t));
    }

    // Broadcast node info to all nodes
    MPI_Bcast(&manager->local_node, sizeof(qec_node_info_t), MPI_BYTE,
              manager->rank, manager->comm);

    pthread_mutex_unlock(&manager->mutex);
    return true;
}

bool qec_comm_get_node_info(qec_comm_manager_t* manager,
                            int node_id,
                            qec_node_info_t* info) {
    if (!manager || !info || node_id < 0) return false;

    pthread_mutex_lock(&manager->mutex);

    if ((size_t)node_id >= manager->num_nodes) {
        set_error(manager, "Invalid node ID");
        pthread_mutex_unlock(&manager->mutex);
        return false;
    }

    memcpy(info, &manager->all_nodes[node_id], sizeof(qec_node_info_t));

    pthread_mutex_unlock(&manager->mutex);
    return true;
}

bool qec_comm_get_all_nodes(qec_comm_manager_t* manager,
                            qec_node_info_t** nodes,
                            size_t* count) {
    if (!manager || !nodes || !count) return false;

    pthread_mutex_lock(&manager->mutex);

    *count = manager->num_nodes;
    *nodes = malloc(manager->num_nodes * sizeof(qec_node_info_t));
    if (!*nodes) {
        set_error(manager, "Failed to allocate nodes array");
        pthread_mutex_unlock(&manager->mutex);
        return false;
    }

    memcpy(*nodes, manager->all_nodes, manager->num_nodes * sizeof(qec_node_info_t));

    pthread_mutex_unlock(&manager->mutex);
    return true;
}

bool qec_comm_set_neighbors(qec_comm_manager_t* manager,
                            const int* neighbors,
                            size_t count) {
    if (!manager || !neighbors || count == 0) return false;

    pthread_mutex_lock(&manager->mutex);

    free(manager->local_node.neighbors);
    manager->local_node.neighbors = malloc(count * sizeof(int));
    if (!manager->local_node.neighbors) {
        set_error(manager, "Failed to allocate neighbors array");
        pthread_mutex_unlock(&manager->mutex);
        return false;
    }

    memcpy(manager->local_node.neighbors, neighbors, count * sizeof(int));
    manager->local_node.num_neighbors = count;

    pthread_mutex_unlock(&manager->mutex);
    return true;
}

// ============================================================================
// Syndrome Communication
// ============================================================================

bool qec_comm_broadcast_syndrome(qec_comm_manager_t* manager,
                                 const qec_syndrome_t* syndrome) {
    if (!manager || !syndrome || !manager->initialized) return false;

    pthread_mutex_lock(&manager->mutex);

    uint64_t start_time = get_timestamp_ns();

    // Broadcast syndrome to all nodes
    qec_syndrome_t local_syndrome = *syndrome;
    local_syndrome.source_node = manager->rank;
    local_syndrome.timestamp_ns = start_time;

    int result = MPI_Bcast(&local_syndrome, sizeof(qec_syndrome_t), MPI_BYTE,
                           manager->rank, manager->comm);

    uint64_t end_time = get_timestamp_ns();
    double latency = (double)(end_time - start_time);

    // Update statistics
    pthread_mutex_lock(&manager->stats_mutex);
    manager->stats.syndromes_processed++;
    manager->stats.messages_sent += (manager->world_size - 1);
    manager->stats.bytes_communicated += sizeof(qec_syndrome_t) * (manager->world_size - 1);

    if (manager->latency_count < STATS_WINDOW_SIZE) {
        manager->latency_samples[manager->latency_count++] = latency;
    }
    pthread_mutex_unlock(&manager->stats_mutex);

    // Invoke syndrome callbacks
    for (size_t i = 0; i < manager->num_syndrome_callbacks; i++) {
        if (manager->syndrome_callbacks[i].active &&
            manager->syndrome_callbacks[i].callback) {
            manager->syndrome_callbacks[i].callback(
                &local_syndrome,
                manager->syndrome_callbacks[i].user_data);
        }
    }

    pthread_mutex_unlock(&manager->mutex);
    return result == MPI_SUCCESS;
}

bool qec_comm_send_syndrome(qec_comm_manager_t* manager,
                            const qec_syndrome_t* syndrome,
                            int target_node) {
    if (!manager || !syndrome || !manager->initialized) return false;
    if (target_node < 0 || target_node >= manager->world_size) return false;

    pthread_mutex_lock(&manager->mutex);

    qec_syndrome_t local_syndrome = *syndrome;
    local_syndrome.source_node = manager->rank;
    local_syndrome.timestamp_ns = get_timestamp_ns();

    int result = MPI_Send(&local_syndrome, sizeof(qec_syndrome_t), MPI_BYTE,
                          target_node, TAG_SYNDROME, manager->comm);

    // Update statistics
    pthread_mutex_lock(&manager->stats_mutex);
    manager->stats.messages_sent++;
    manager->stats.bytes_communicated += sizeof(qec_syndrome_t);
    pthread_mutex_unlock(&manager->stats_mutex);

    pthread_mutex_unlock(&manager->mutex);
    return result == MPI_SUCCESS;
}

bool qec_comm_receive_syndrome(qec_comm_manager_t* manager,
                               qec_syndrome_t* syndrome,
                               int source_node) {
    if (!manager || !syndrome || !manager->initialized) return false;

    pthread_mutex_lock(&manager->mutex);

    int result = MPI_Recv(syndrome, sizeof(qec_syndrome_t), MPI_BYTE,
                          source_node, TAG_SYNDROME, manager->comm,
                          MPI_STATUS_IGNORE);

    if (result == MPI_SUCCESS) {
        pthread_mutex_lock(&manager->stats_mutex);
        manager->stats.messages_received++;
        manager->stats.syndromes_processed++;
        pthread_mutex_unlock(&manager->stats_mutex);

        // Invoke syndrome callbacks
        for (size_t i = 0; i < manager->num_syndrome_callbacks; i++) {
            if (manager->syndrome_callbacks[i].active &&
                manager->syndrome_callbacks[i].callback) {
                manager->syndrome_callbacks[i].callback(
                    syndrome,
                    manager->syndrome_callbacks[i].user_data);
            }
        }
    }

    pthread_mutex_unlock(&manager->mutex);
    return result == MPI_SUCCESS;
}

bool qec_comm_receive_syndrome_async(qec_comm_manager_t* manager,
                                     qec_syndrome_t* syndrome,
                                     int source_node,
                                     int* request) {
    if (!manager || !syndrome || !request || !manager->initialized) return false;

    pthread_mutex_lock(&manager->mutex);

    MPI_Request mpi_request;
    int result = MPI_Irecv(syndrome, sizeof(qec_syndrome_t), MPI_BYTE,
                           source_node, TAG_SYNDROME, manager->comm,
                           &mpi_request);

    *request = (int)mpi_request;

    pthread_mutex_unlock(&manager->mutex);
    return result == MPI_SUCCESS;
}

bool qec_comm_gather_syndromes(qec_comm_manager_t* manager,
                               const qec_syndrome_t* local,
                               qec_syndrome_t** all_syndromes,
                               size_t* count) {
    if (!manager || !local || !all_syndromes || !count || !manager->initialized) {
        return false;
    }

    pthread_mutex_lock(&manager->mutex);

    qec_syndrome_t local_syndrome = *local;
    local_syndrome.source_node = manager->rank;
    local_syndrome.timestamp_ns = get_timestamp_ns();

    // Only root gets all syndromes
    qec_syndrome_t* recv_buf = NULL;
    if (manager->rank == 0) {
        recv_buf = malloc(manager->world_size * sizeof(qec_syndrome_t));
        if (!recv_buf) {
            set_error(manager, "Failed to allocate gather buffer");
            pthread_mutex_unlock(&manager->mutex);
            return false;
        }
    }

    int result = MPI_Gather(&local_syndrome, sizeof(qec_syndrome_t), MPI_BYTE,
                            recv_buf, sizeof(qec_syndrome_t), MPI_BYTE,
                            0, manager->comm);

    if (result == MPI_SUCCESS && manager->rank == 0) {
        *all_syndromes = recv_buf;
        *count = manager->world_size;

        pthread_mutex_lock(&manager->stats_mutex);
        manager->stats.syndromes_processed += manager->world_size;
        manager->stats.messages_received += (manager->world_size - 1);
        pthread_mutex_unlock(&manager->stats_mutex);
    } else {
        *all_syndromes = NULL;
        *count = 0;
        free(recv_buf);
    }

    pthread_mutex_unlock(&manager->mutex);
    return result == MPI_SUCCESS;
}

bool qec_comm_allgather_syndromes(qec_comm_manager_t* manager,
                                  const qec_syndrome_t* local,
                                  qec_syndrome_t** all_syndromes,
                                  size_t* count) {
    if (!manager || !local || !all_syndromes || !count || !manager->initialized) {
        return false;
    }

    pthread_mutex_lock(&manager->mutex);

    qec_syndrome_t local_syndrome = *local;
    local_syndrome.source_node = manager->rank;
    local_syndrome.timestamp_ns = get_timestamp_ns();

    // All nodes get all syndromes
    qec_syndrome_t* recv_buf = malloc(manager->world_size * sizeof(qec_syndrome_t));
    if (!recv_buf) {
        set_error(manager, "Failed to allocate allgather buffer");
        pthread_mutex_unlock(&manager->mutex);
        return false;
    }

    int result = MPI_Allgather(&local_syndrome, sizeof(qec_syndrome_t), MPI_BYTE,
                               recv_buf, sizeof(qec_syndrome_t), MPI_BYTE,
                               manager->comm);

    if (result == MPI_SUCCESS) {
        *all_syndromes = recv_buf;
        *count = manager->world_size;

        pthread_mutex_lock(&manager->stats_mutex);
        manager->stats.syndromes_processed += manager->world_size;
        pthread_mutex_unlock(&manager->stats_mutex);
    } else {
        *all_syndromes = NULL;
        *count = 0;
        free(recv_buf);
    }

    pthread_mutex_unlock(&manager->mutex);
    return result == MPI_SUCCESS;
}

bool qec_comm_aggregate_syndromes(qec_comm_manager_t* manager,
                                  const qec_syndrome_t* syndromes,
                                  size_t count,
                                  qec_syndrome_t* result) {
    if (!manager || !syndromes || count == 0 || !result) return false;

    // Start with first syndrome
    *result = syndromes[0];

    switch (manager->config.syndrome_agg) {
        case QEC_AGGREGATE_OR:
            for (size_t i = 1; i < count; i++) {
                result->syndrome_bits |= syndromes[i].syndrome_bits;
            }
            break;

        case QEC_AGGREGATE_AND:
            for (size_t i = 1; i < count; i++) {
                result->syndrome_bits &= syndromes[i].syndrome_bits;
            }
            break;

        case QEC_AGGREGATE_XOR:
            for (size_t i = 1; i < count; i++) {
                result->syndrome_bits ^= syndromes[i].syndrome_bits;
            }
            break;

        case QEC_AGGREGATE_MAJORITY: {
            // Count votes for each bit position
            for (size_t bit = 0; bit < result->num_bits && bit < 64; bit++) {
                size_t ones = 0;
                for (size_t i = 0; i < count; i++) {
                    if (syndromes[i].syndrome_bits & (1ULL << bit)) {
                        ones++;
                    }
                }
                if (ones > count / 2) {
                    result->syndrome_bits |= (1ULL << bit);
                } else {
                    result->syndrome_bits &= ~(1ULL << bit);
                }
            }
            break;
        }

        case QEC_AGGREGATE_WEIGHTED: {
            // Weight by confidence
            for (size_t bit = 0; bit < result->num_bits && bit < 64; bit++) {
                double weighted_sum = 0.0;
                double total_weight = 0.0;
                for (size_t i = 0; i < count; i++) {
                    double weight = syndromes[i].confidence;
                    if (syndromes[i].syndrome_bits & (1ULL << bit)) {
                        weighted_sum += weight;
                    }
                    total_weight += weight;
                }
                if (total_weight > 0 && weighted_sum / total_weight > 0.5) {
                    result->syndrome_bits |= (1ULL << bit);
                } else {
                    result->syndrome_bits &= ~(1ULL << bit);
                }
            }
            break;
        }

        case QEC_AGGREGATE_NONE:
        case QEC_AGGREGATE_CUSTOM:
        default:
            // Just use first syndrome
            break;
    }

    // Average confidence
    result->confidence = 0.0;
    for (size_t i = 0; i < count; i++) {
        result->confidence += syndromes[i].confidence;
    }
    result->confidence /= count;

    result->timestamp_ns = get_timestamp_ns();
    result->source_node = -1;  // Aggregated from multiple sources

    return true;
}

// ============================================================================
// Error Event Communication
// ============================================================================

static void add_to_history(qec_comm_manager_t* manager,
                           const qec_error_event_t* event) {
    if (!manager->logging_enabled) return;

    if (manager->history_size >= manager->history_capacity) {
        // Circular buffer - overwrite oldest
        memmove(manager->error_history, manager->error_history + 1,
                (manager->history_capacity - 1) * sizeof(qec_error_event_t));
        manager->history_size = manager->history_capacity - 1;
    }

    // Deep copy event
    qec_error_event_t* dest = &manager->error_history[manager->history_size];
    memcpy(dest, event, sizeof(qec_error_event_t));

    // Copy arrays
    if (event->affected_nodes && event->num_affected > 0) {
        dest->affected_nodes = malloc(event->num_affected * sizeof(int));
        if (dest->affected_nodes) {
            memcpy(dest->affected_nodes, event->affected_nodes,
                   event->num_affected * sizeof(int));
        }
    }

    if (event->qubit_indices && event->num_qubits > 0) {
        dest->qubit_indices = malloc(event->num_qubits * sizeof(size_t));
        if (dest->qubit_indices) {
            memcpy(dest->qubit_indices, event->qubit_indices,
                   event->num_qubits * sizeof(size_t));
        }
    }

    manager->history_size++;

    // Update statistics
    pthread_mutex_lock(&manager->stats_mutex);
    manager->stats.total_errors++;
    if (event->type < 13) {
        manager->stats.errors_by_type[event->type]++;
    }
    if (event->severity < 6) {
        manager->stats.errors_by_severity[event->severity]++;
    }
    pthread_mutex_unlock(&manager->stats_mutex);
}

bool qec_comm_report_error(qec_comm_manager_t* manager,
                           const qec_error_event_t* event) {
    if (!manager || !event || !manager->initialized) return false;

    pthread_mutex_lock(&manager->mutex);

    // Assign event ID
    qec_error_event_t local_event = *event;
    local_event.event_id = atomic_fetch_add(&manager->next_event_id, 1);
    local_event.source_node = manager->rank;
    local_event.timestamp_ns = get_timestamp_ns();

    // Add to history
    add_to_history(manager, &local_event);

    // Invoke callbacks
    for (size_t i = 0; i < manager->num_error_callbacks; i++) {
        if (manager->error_callbacks[i].active &&
            manager->error_callbacks[i].callback &&
            event->severity >= manager->error_callbacks[i].min_severity) {
            manager->error_callbacks[i].callback(
                &local_event,
                manager->error_callbacks[i].user_data);
        }
    }

    pthread_mutex_unlock(&manager->mutex);
    return true;
}

bool qec_comm_broadcast_error(qec_comm_manager_t* manager,
                              const qec_error_event_t* event) {
    if (!manager || !event || !manager->initialized) return false;

    pthread_mutex_lock(&manager->mutex);

    qec_error_event_t local_event = *event;
    local_event.event_id = atomic_fetch_add(&manager->next_event_id, 1);
    local_event.source_node = manager->rank;
    local_event.timestamp_ns = get_timestamp_ns();

    // Serialize event (without pointer data for broadcast)
    int result = MPI_Bcast(&local_event, sizeof(qec_error_event_t), MPI_BYTE,
                           manager->rank, manager->comm);

    if (result == MPI_SUCCESS) {
        add_to_history(manager, &local_event);

        pthread_mutex_lock(&manager->stats_mutex);
        manager->stats.messages_sent += (manager->world_size - 1);
        manager->stats.bytes_communicated +=
            sizeof(qec_error_event_t) * (manager->world_size - 1);
        pthread_mutex_unlock(&manager->stats_mutex);
    }

    pthread_mutex_unlock(&manager->mutex);
    return result == MPI_SUCCESS;
}

bool qec_comm_send_error(qec_comm_manager_t* manager,
                         const qec_error_event_t* event,
                         const int* target_nodes,
                         size_t num_targets) {
    if (!manager || !event || !target_nodes || num_targets == 0) return false;
    if (!manager->initialized) return false;

    pthread_mutex_lock(&manager->mutex);

    qec_error_event_t local_event = *event;
    local_event.event_id = atomic_fetch_add(&manager->next_event_id, 1);
    local_event.source_node = manager->rank;
    local_event.timestamp_ns = get_timestamp_ns();

    bool all_success = true;
    for (size_t i = 0; i < num_targets; i++) {
        if (target_nodes[i] >= 0 && target_nodes[i] < manager->world_size) {
            int result = MPI_Send(&local_event, sizeof(qec_error_event_t),
                                  MPI_BYTE, target_nodes[i], TAG_ERROR_EVENT,
                                  manager->comm);
            if (result != MPI_SUCCESS) {
                all_success = false;
            } else {
                pthread_mutex_lock(&manager->stats_mutex);
                manager->stats.messages_sent++;
                manager->stats.bytes_communicated += sizeof(qec_error_event_t);
                pthread_mutex_unlock(&manager->stats_mutex);
            }
        }
    }

    if (all_success) {
        add_to_history(manager, &local_event);
    }

    pthread_mutex_unlock(&manager->mutex);
    return all_success;
}

bool qec_comm_receive_error(qec_comm_manager_t* manager,
                            qec_error_event_t* event) {
    if (!manager || !event || !manager->initialized) return false;

    pthread_mutex_lock(&manager->mutex);

    int result = MPI_Recv(event, sizeof(qec_error_event_t), MPI_BYTE,
                          MPI_ANY_SOURCE, TAG_ERROR_EVENT, manager->comm,
                          MPI_STATUS_IGNORE);

    if (result == MPI_SUCCESS) {
        add_to_history(manager, event);

        pthread_mutex_lock(&manager->stats_mutex);
        manager->stats.messages_received++;
        pthread_mutex_unlock(&manager->stats_mutex);

        // Invoke callbacks
        for (size_t i = 0; i < manager->num_error_callbacks; i++) {
            if (manager->error_callbacks[i].active &&
                manager->error_callbacks[i].callback &&
                event->severity >= manager->error_callbacks[i].min_severity) {
                manager->error_callbacks[i].callback(
                    event,
                    manager->error_callbacks[i].user_data);
            }
        }
    }

    pthread_mutex_unlock(&manager->mutex);
    return result == MPI_SUCCESS;
}

bool qec_comm_receive_error_async(qec_comm_manager_t* manager,
                                  qec_error_event_t* event,
                                  int* request) {
    if (!manager || !event || !request || !manager->initialized) return false;

    pthread_mutex_lock(&manager->mutex);

    MPI_Request mpi_request;
    int result = MPI_Irecv(event, sizeof(qec_error_event_t), MPI_BYTE,
                           MPI_ANY_SOURCE, TAG_ERROR_EVENT, manager->comm,
                           &mpi_request);

    *request = (int)mpi_request;

    pthread_mutex_unlock(&manager->mutex);
    return result == MPI_SUCCESS;
}

bool qec_comm_poll_errors(qec_comm_manager_t* manager, bool* has_pending) {
    if (!manager || !has_pending || !manager->initialized) return false;

    pthread_mutex_lock(&manager->mutex);

    int flag = 0;
    MPI_Iprobe(MPI_ANY_SOURCE, TAG_ERROR_EVENT, manager->comm,
               &flag, MPI_STATUS_IGNORE);

    *has_pending = (flag != 0);

    pthread_mutex_unlock(&manager->mutex);
    return true;
}

bool qec_comm_get_pending_errors(qec_comm_manager_t* manager,
                                 qec_error_event_t** events,
                                 size_t* count) {
    if (!manager || !events || !count || !manager->initialized) return false;

    pthread_mutex_lock(&manager->mutex);

    // Count pending
    size_t pending_count = 0;
    int flag;
    do {
        MPI_Iprobe(MPI_ANY_SOURCE, TAG_ERROR_EVENT, manager->comm,
                   &flag, MPI_STATUS_IGNORE);
        if (flag) pending_count++;
    } while (flag && pending_count < MAX_PENDING_EVENTS);

    if (pending_count == 0) {
        *events = NULL;
        *count = 0;
        pthread_mutex_unlock(&manager->mutex);
        return true;
    }

    // Allocate and receive
    *events = malloc(pending_count * sizeof(qec_error_event_t));
    if (!*events) {
        set_error(manager, "Failed to allocate events array");
        pthread_mutex_unlock(&manager->mutex);
        return false;
    }

    size_t received = 0;
    for (size_t i = 0; i < pending_count; i++) {
        MPI_Iprobe(MPI_ANY_SOURCE, TAG_ERROR_EVENT, manager->comm,
                   &flag, MPI_STATUS_IGNORE);
        if (flag) {
            int result = MPI_Recv(&(*events)[received], sizeof(qec_error_event_t),
                                  MPI_BYTE, MPI_ANY_SOURCE, TAG_ERROR_EVENT,
                                  manager->comm, MPI_STATUS_IGNORE);
            if (result == MPI_SUCCESS) {
                received++;
            }
        }
    }

    *count = received;

    pthread_mutex_unlock(&manager->mutex);
    return true;
}

// ============================================================================
// Recovery Coordination
// ============================================================================

bool qec_comm_coordinate_recovery(qec_comm_manager_t* manager,
                                  const qec_recovery_action_t* action) {
    if (!manager || !action || !manager->initialized) return false;

    pthread_mutex_lock(&manager->mutex);

    // Create recovery tracker
    if (manager->num_trackers >= manager->trackers_capacity) {
        // Grow capacity
        size_t new_cap = manager->trackers_capacity * 2;
        recovery_tracker_t* new_trackers = realloc(
            manager->recovery_trackers, new_cap * sizeof(recovery_tracker_t));
        if (!new_trackers) {
            set_error(manager, "Failed to expand recovery trackers");
            pthread_mutex_unlock(&manager->mutex);
            return false;
        }
        manager->recovery_trackers = new_trackers;
        manager->trackers_capacity = new_cap;
    }

    recovery_tracker_t* tracker = &manager->recovery_trackers[manager->num_trackers];
    tracker->event_id = action->event_id;
    tracker->num_nodes = manager->world_size;
    tracker->node_complete = calloc(manager->world_size, sizeof(bool));
    tracker->all_complete = false;
    manager->num_trackers++;

    // Invoke recovery callbacks
    for (size_t i = 0; i < manager->num_recovery_callbacks; i++) {
        if (manager->recovery_callbacks[i].active &&
            manager->recovery_callbacks[i].callback) {
            manager->recovery_callbacks[i].callback(
                action,
                manager->recovery_callbacks[i].user_data);
        }
    }

    pthread_mutex_lock(&manager->stats_mutex);
    manager->stats.recoveries_attempted++;
    pthread_mutex_unlock(&manager->stats_mutex);

    pthread_mutex_unlock(&manager->mutex);
    return true;
}

bool qec_comm_broadcast_recovery(qec_comm_manager_t* manager,
                                 const qec_recovery_action_t* action) {
    if (!manager || !action || !manager->initialized) return false;

    pthread_mutex_lock(&manager->mutex);

    int result = MPI_Bcast((void*)action, sizeof(qec_recovery_action_t), MPI_BYTE,
                           manager->rank, manager->comm);

    if (result == MPI_SUCCESS) {
        // Invoke local callbacks
        for (size_t i = 0; i < manager->num_recovery_callbacks; i++) {
            if (manager->recovery_callbacks[i].active &&
                manager->recovery_callbacks[i].callback) {
                manager->recovery_callbacks[i].callback(
                    action,
                    manager->recovery_callbacks[i].user_data);
            }
        }
    }

    pthread_mutex_unlock(&manager->mutex);
    return result == MPI_SUCCESS;
}

bool qec_comm_recovery_consensus(qec_comm_manager_t* manager,
                                 const qec_recovery_action_t* proposed,
                                 qec_recovery_action_t* agreed) {
    if (!manager || !proposed || !agreed || !manager->initialized) return false;

    pthread_mutex_lock(&manager->mutex);

    // Simple consensus: all-reduce on priority, highest priority wins
    int local_priority = proposed->priority;
    int max_priority;

    int result = MPI_Allreduce(&local_priority, &max_priority, 1,
                               MPI_INT, MPI_MAX, manager->comm);

    if (result != MPI_SUCCESS) {
        pthread_mutex_unlock(&manager->mutex);
        return false;
    }

    // Node with highest priority broadcasts its action
    if (local_priority == max_priority) {
        *agreed = *proposed;
    }

    result = MPI_Bcast(agreed, sizeof(qec_recovery_action_t), MPI_BYTE,
                       0, manager->comm);  // Simplified: use rank 0 for tie-break

    pthread_mutex_unlock(&manager->mutex);
    return result == MPI_SUCCESS;
}

bool qec_comm_recovery_barrier(qec_comm_manager_t* manager) {
    if (!manager || !manager->initialized) return false;

    int result = MPI_Barrier(manager->comm);
    return result == MPI_SUCCESS;
}

bool qec_comm_report_recovery_complete(qec_comm_manager_t* manager,
                                       uint64_t event_id,
                                       bool success) {
    if (!manager || !manager->initialized) return false;

    pthread_mutex_lock(&manager->mutex);

    // Find tracker
    for (size_t i = 0; i < manager->num_trackers; i++) {
        if (manager->recovery_trackers[i].event_id == event_id) {
            manager->recovery_trackers[i].node_complete[manager->rank] = true;

            if (success) {
                pthread_mutex_lock(&manager->stats_mutex);
                manager->stats.recoveries_successful++;
                pthread_mutex_unlock(&manager->stats_mutex);
            }
            break;
        }
    }

    // Broadcast completion status
    uint64_t status[2] = {event_id, success ? 1ULL : 0ULL};
    int result = MPI_Bcast(status, 2, MPI_UINT64_T, manager->rank, manager->comm);

    pthread_mutex_unlock(&manager->mutex);
    return result == MPI_SUCCESS;
}

bool qec_comm_check_recovery_status(qec_comm_manager_t* manager,
                                    uint64_t event_id,
                                    bool* all_complete) {
    if (!manager || !all_complete || !manager->initialized) return false;

    pthread_mutex_lock(&manager->mutex);

    *all_complete = false;

    for (size_t i = 0; i < manager->num_trackers; i++) {
        if (manager->recovery_trackers[i].event_id == event_id) {
            // Check all nodes
            bool complete = true;
            for (size_t j = 0; j < manager->recovery_trackers[i].num_nodes; j++) {
                if (!manager->recovery_trackers[i].node_complete[j]) {
                    complete = false;
                    break;
                }
            }
            manager->recovery_trackers[i].all_complete = complete;
            *all_complete = complete;
            break;
        }
    }

    pthread_mutex_unlock(&manager->mutex);
    return true;
}

// ============================================================================
// Callbacks
// ============================================================================

bool qec_comm_register_error_callback(qec_comm_manager_t* manager,
                                      qec_error_callback_t callback,
                                      void* user_data,
                                      qec_severity_t min_severity) {
    if (!manager || !callback) return false;
    if (manager->num_error_callbacks >= MAX_CALLBACKS) return false;

    pthread_mutex_lock(&manager->mutex);

    manager->error_callbacks[manager->num_error_callbacks].callback = callback;
    manager->error_callbacks[manager->num_error_callbacks].user_data = user_data;
    manager->error_callbacks[manager->num_error_callbacks].min_severity = min_severity;
    manager->error_callbacks[manager->num_error_callbacks].active = true;
    manager->num_error_callbacks++;

    pthread_mutex_unlock(&manager->mutex);
    return true;
}

bool qec_comm_register_syndrome_callback(qec_comm_manager_t* manager,
                                         qec_syndrome_callback_t callback,
                                         void* user_data) {
    if (!manager || !callback) return false;
    if (manager->num_syndrome_callbacks >= MAX_CALLBACKS) return false;

    pthread_mutex_lock(&manager->mutex);

    manager->syndrome_callbacks[manager->num_syndrome_callbacks].callback = callback;
    manager->syndrome_callbacks[manager->num_syndrome_callbacks].user_data = user_data;
    manager->syndrome_callbacks[manager->num_syndrome_callbacks].active = true;
    manager->num_syndrome_callbacks++;

    pthread_mutex_unlock(&manager->mutex);
    return true;
}

bool qec_comm_register_recovery_callback(qec_comm_manager_t* manager,
                                         qec_recovery_callback_t callback,
                                         void* user_data) {
    if (!manager || !callback) return false;
    if (manager->num_recovery_callbacks >= MAX_CALLBACKS) return false;

    pthread_mutex_lock(&manager->mutex);

    manager->recovery_callbacks[manager->num_recovery_callbacks].callback = callback;
    manager->recovery_callbacks[manager->num_recovery_callbacks].user_data = user_data;
    manager->recovery_callbacks[manager->num_recovery_callbacks].active = true;
    manager->num_recovery_callbacks++;

    pthread_mutex_unlock(&manager->mutex);
    return true;
}

void qec_comm_clear_callbacks(qec_comm_manager_t* manager) {
    if (!manager) return;

    pthread_mutex_lock(&manager->mutex);

    manager->num_error_callbacks = 0;
    manager->num_syndrome_callbacks = 0;
    manager->num_recovery_callbacks = 0;

    pthread_mutex_unlock(&manager->mutex);
}

// ============================================================================
// Error History and Logging
// ============================================================================

bool qec_comm_enable_logging(qec_comm_manager_t* manager, bool enable) {
    if (!manager) return false;

    pthread_mutex_lock(&manager->mutex);
    manager->logging_enabled = enable;
    pthread_mutex_unlock(&manager->mutex);
    return true;
}

bool qec_comm_get_error_history(qec_comm_manager_t* manager,
                                qec_error_event_t** events,
                                size_t* count) {
    if (!manager || !events || !count) return false;

    pthread_mutex_lock(&manager->mutex);

    *count = manager->history_size;
    if (*count == 0) {
        *events = NULL;
        pthread_mutex_unlock(&manager->mutex);
        return true;
    }

    *events = malloc(*count * sizeof(qec_error_event_t));
    if (!*events) {
        set_error(manager, "Failed to allocate history array");
        pthread_mutex_unlock(&manager->mutex);
        return false;
    }

    memcpy(*events, manager->error_history, *count * sizeof(qec_error_event_t));

    pthread_mutex_unlock(&manager->mutex);
    return true;
}

bool qec_comm_get_errors_by_type(qec_comm_manager_t* manager,
                                 qec_error_type_t type,
                                 qec_error_event_t** events,
                                 size_t* count) {
    if (!manager || !events || !count) return false;

    pthread_mutex_lock(&manager->mutex);

    // Count matching
    size_t matching = 0;
    for (size_t i = 0; i < manager->history_size; i++) {
        if (manager->error_history[i].type == type) {
            matching++;
        }
    }

    *count = matching;
    if (matching == 0) {
        *events = NULL;
        pthread_mutex_unlock(&manager->mutex);
        return true;
    }

    *events = malloc(matching * sizeof(qec_error_event_t));
    if (!*events) {
        set_error(manager, "Failed to allocate filtered array");
        pthread_mutex_unlock(&manager->mutex);
        return false;
    }

    size_t idx = 0;
    for (size_t i = 0; i < manager->history_size; i++) {
        if (manager->error_history[i].type == type) {
            (*events)[idx++] = manager->error_history[i];
        }
    }

    pthread_mutex_unlock(&manager->mutex);
    return true;
}

bool qec_comm_get_errors_by_severity(qec_comm_manager_t* manager,
                                     qec_severity_t severity,
                                     qec_error_event_t** events,
                                     size_t* count) {
    if (!manager || !events || !count) return false;

    pthread_mutex_lock(&manager->mutex);

    size_t matching = 0;
    for (size_t i = 0; i < manager->history_size; i++) {
        if (manager->error_history[i].severity == severity) {
            matching++;
        }
    }

    *count = matching;
    if (matching == 0) {
        *events = NULL;
        pthread_mutex_unlock(&manager->mutex);
        return true;
    }

    *events = malloc(matching * sizeof(qec_error_event_t));
    if (!*events) {
        set_error(manager, "Failed to allocate filtered array");
        pthread_mutex_unlock(&manager->mutex);
        return false;
    }

    size_t idx = 0;
    for (size_t i = 0; i < manager->history_size; i++) {
        if (manager->error_history[i].severity == severity) {
            (*events)[idx++] = manager->error_history[i];
        }
    }

    pthread_mutex_unlock(&manager->mutex);
    return true;
}

bool qec_comm_get_errors_in_range(qec_comm_manager_t* manager,
                                  uint64_t start_ns,
                                  uint64_t end_ns,
                                  qec_error_event_t** events,
                                  size_t* count) {
    if (!manager || !events || !count) return false;

    pthread_mutex_lock(&manager->mutex);

    size_t matching = 0;
    for (size_t i = 0; i < manager->history_size; i++) {
        uint64_t ts = manager->error_history[i].timestamp_ns;
        if (ts >= start_ns && ts <= end_ns) {
            matching++;
        }
    }

    *count = matching;
    if (matching == 0) {
        *events = NULL;
        pthread_mutex_unlock(&manager->mutex);
        return true;
    }

    *events = malloc(matching * sizeof(qec_error_event_t));
    if (!*events) {
        set_error(manager, "Failed to allocate filtered array");
        pthread_mutex_unlock(&manager->mutex);
        return false;
    }

    size_t idx = 0;
    for (size_t i = 0; i < manager->history_size; i++) {
        uint64_t ts = manager->error_history[i].timestamp_ns;
        if (ts >= start_ns && ts <= end_ns) {
            (*events)[idx++] = manager->error_history[i];
        }
    }

    pthread_mutex_unlock(&manager->mutex);
    return true;
}

void qec_comm_clear_history(qec_comm_manager_t* manager) {
    if (!manager) return;

    pthread_mutex_lock(&manager->mutex);

    for (size_t i = 0; i < manager->history_size; i++) {
        free(manager->error_history[i].affected_nodes);
        free(manager->error_history[i].qubit_indices);
    }
    manager->history_size = 0;

    pthread_mutex_unlock(&manager->mutex);
}

bool qec_comm_export_log(qec_comm_manager_t* manager, const char* filename) {
    if (!manager || !filename) return false;

    pthread_mutex_lock(&manager->mutex);

    FILE* f = fopen(filename, "w");
    if (!f) {
        set_error(manager, "Failed to open log file");
        pthread_mutex_unlock(&manager->mutex);
        return false;
    }

    fprintf(f, "# Quantum Error Communication Log\n");
    fprintf(f, "# Node: %d, Total errors: %zu\n", manager->rank, manager->history_size);
    fprintf(f, "#\n");
    fprintf(f, "# event_id, timestamp_ns, type, severity, source_node, message\n");

    for (size_t i = 0; i < manager->history_size; i++) {
        qec_error_event_t* e = &manager->error_history[i];
        fprintf(f, "%lu, %lu, %s, %s, %d, \"%s\"\n",
                (unsigned long)e->event_id,
                (unsigned long)e->timestamp_ns,
                qec_error_type_name(e->type),
                qec_severity_name(e->severity),
                e->source_node,
                e->message);
    }

    fclose(f);

    pthread_mutex_unlock(&manager->mutex);
    return true;
}

// ============================================================================
// Statistics
// ============================================================================

bool qec_comm_get_stats(qec_comm_manager_t* manager, qec_comm_stats_t* stats) {
    if (!manager || !stats) return false;

    pthread_mutex_lock(&manager->stats_mutex);
    *stats = manager->stats;

    // Calculate latency statistics
    if (manager->latency_count > 0) {
        double sum = 0.0;
        for (size_t i = 0; i < manager->latency_count; i++) {
            sum += manager->latency_samples[i];
        }
        stats->avg_detection_latency_ns = sum / manager->latency_count;
    }

    pthread_mutex_unlock(&manager->stats_mutex);
    return true;
}

void qec_comm_reset_stats(qec_comm_manager_t* manager) {
    if (!manager) return;

    pthread_mutex_lock(&manager->stats_mutex);
    memset(&manager->stats, 0, sizeof(qec_comm_stats_t));
    manager->latency_count = 0;
    pthread_mutex_unlock(&manager->stats_mutex);
}

double qec_comm_get_qubit_error_rate(qec_comm_manager_t* manager,
                                     size_t qubit_index) {
    if (!manager) return 0.0;

    pthread_mutex_lock(&manager->mutex);

    size_t errors = 0;
    for (size_t i = 0; i < manager->history_size; i++) {
        qec_error_event_t* e = &manager->error_history[i];
        for (size_t j = 0; j < e->num_qubits; j++) {
            if (e->qubit_indices && e->qubit_indices[j] == qubit_index) {
                errors++;
                break;
            }
        }
    }

    double rate = (manager->history_size > 0) ?
                  (double)errors / (double)manager->history_size : 0.0;

    pthread_mutex_unlock(&manager->mutex);
    return rate;
}

double qec_comm_get_node_error_rate(qec_comm_manager_t* manager, int node_id) {
    if (!manager) return 0.0;

    pthread_mutex_lock(&manager->mutex);

    size_t errors = 0;
    for (size_t i = 0; i < manager->history_size; i++) {
        if (manager->error_history[i].source_node == node_id) {
            errors++;
        }
    }

    double rate = (manager->history_size > 0) ?
                  (double)errors / (double)manager->history_size : 0.0;

    pthread_mutex_unlock(&manager->mutex);
    return rate;
}

bool qec_comm_get_latency_stats(qec_comm_manager_t* manager,
                                double* min_ns,
                                double* max_ns,
                                double* avg_ns) {
    if (!manager || !min_ns || !max_ns || !avg_ns) return false;

    pthread_mutex_lock(&manager->stats_mutex);

    if (manager->latency_count == 0) {
        *min_ns = 0.0;
        *max_ns = 0.0;
        *avg_ns = 0.0;
        pthread_mutex_unlock(&manager->stats_mutex);
        return true;
    }

    double min_val = manager->latency_samples[0];
    double max_val = manager->latency_samples[0];
    double sum = 0.0;

    for (size_t i = 0; i < manager->latency_count; i++) {
        double v = manager->latency_samples[i];
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
        sum += v;
    }

    *min_ns = min_val;
    *max_ns = max_val;
    *avg_ns = sum / manager->latency_count;

    pthread_mutex_unlock(&manager->stats_mutex);
    return true;
}

// ============================================================================
// Reporting
// ============================================================================

char* qec_comm_generate_report(qec_comm_manager_t* manager) {
    if (!manager) return NULL;

    pthread_mutex_lock(&manager->mutex);

    qec_comm_stats_t stats;
    qec_comm_get_stats(manager, &stats);

    // Estimate buffer size
    size_t buf_size = 4096;
    char* report = malloc(buf_size);
    if (!report) {
        pthread_mutex_unlock(&manager->mutex);
        return NULL;
    }

    int written = snprintf(report, buf_size,
        "=== Quantum Error Communication Report ===\n"
        "Node: %d of %d\n"
        "Status: %s\n\n"
        "Error Statistics:\n"
        "  Total errors: %lu\n"
        "  Messages sent: %lu\n"
        "  Messages received: %lu\n"
        "  Syndromes processed: %lu\n"
        "  Bytes communicated: %lu\n\n"
        "Recovery Statistics:\n"
        "  Attempts: %lu\n"
        "  Successful: %lu\n"
        "  Success rate: %.1f%%\n\n"
        "Latency:\n"
        "  Average detection: %.2f ns\n"
        "  Average recovery: %.2f ns\n\n"
        "Error rates by qubit: %.4f\n",
        manager->rank,
        manager->world_size,
        manager->initialized ? "Active" : "Inactive",
        (unsigned long)stats.total_errors,
        (unsigned long)stats.messages_sent,
        (unsigned long)stats.messages_received,
        (unsigned long)stats.syndromes_processed,
        (unsigned long)stats.bytes_communicated,
        (unsigned long)stats.recoveries_attempted,
        (unsigned long)stats.recoveries_successful,
        stats.recoveries_attempted > 0 ?
            100.0 * stats.recoveries_successful / stats.recoveries_attempted : 0.0,
        stats.avg_detection_latency_ns,
        stats.avg_recovery_latency_ns,
        stats.error_rate_per_qubit);

    if (written < 0 || (size_t)written >= buf_size) {
        free(report);
        pthread_mutex_unlock(&manager->mutex);
        return NULL;
    }

    pthread_mutex_unlock(&manager->mutex);
    return report;
}

char* qec_comm_export_json(qec_comm_manager_t* manager) {
    if (!manager) return NULL;

    pthread_mutex_lock(&manager->mutex);

    qec_comm_stats_t stats;
    qec_comm_get_stats(manager, &stats);

    size_t buf_size = 8192;
    char* json = malloc(buf_size);
    if (!json) {
        pthread_mutex_unlock(&manager->mutex);
        return NULL;
    }

    int written = snprintf(json, buf_size,
        "{\n"
        "  \"node_id\": %d,\n"
        "  \"world_size\": %d,\n"
        "  \"initialized\": %s,\n"
        "  \"statistics\": {\n"
        "    \"total_errors\": %lu,\n"
        "    \"messages_sent\": %lu,\n"
        "    \"messages_received\": %lu,\n"
        "    \"syndromes_processed\": %lu,\n"
        "    \"bytes_communicated\": %lu,\n"
        "    \"recoveries_attempted\": %lu,\n"
        "    \"recoveries_successful\": %lu,\n"
        "    \"avg_detection_latency_ns\": %.2f,\n"
        "    \"avg_recovery_latency_ns\": %.2f\n"
        "  },\n"
        "  \"history_size\": %zu\n"
        "}\n",
        manager->rank,
        manager->world_size,
        manager->initialized ? "true" : "false",
        (unsigned long)stats.total_errors,
        (unsigned long)stats.messages_sent,
        (unsigned long)stats.messages_received,
        (unsigned long)stats.syndromes_processed,
        (unsigned long)stats.bytes_communicated,
        (unsigned long)stats.recoveries_attempted,
        (unsigned long)stats.recoveries_successful,
        stats.avg_detection_latency_ns,
        stats.avg_recovery_latency_ns,
        manager->history_size);

    if (written < 0 || (size_t)written >= buf_size) {
        free(json);
        pthread_mutex_unlock(&manager->mutex);
        return NULL;
    }

    pthread_mutex_unlock(&manager->mutex);
    return json;
}

bool qec_comm_export_to_file(qec_comm_manager_t* manager, const char* filename) {
    char* json = qec_comm_export_json(manager);
    if (!json) return false;

    FILE* f = fopen(filename, "w");
    if (!f) {
        free(json);
        return false;
    }

    fputs(json, f);
    fclose(f);
    free(json);
    return true;
}

// ============================================================================
// Memory Management
// ============================================================================

void qec_comm_free_events(qec_error_event_t* events, size_t count) {
    if (!events) return;
    for (size_t i = 0; i < count; i++) {
        free(events[i].affected_nodes);
        free(events[i].qubit_indices);
    }
    free(events);
}

void qec_comm_free_syndromes(qec_syndrome_t* syndromes, size_t count) {
    (void)count;
    free(syndromes);
}

void qec_comm_free_nodes(qec_node_info_t* nodes, size_t count) {
    if (!nodes) return;
    for (size_t i = 0; i < count; i++) {
        free(nodes[i].neighbors);
    }
    free(nodes);
}

void qec_comm_free_string(char* str) {
    free(str);
}

const char* qec_comm_get_last_error(qec_comm_manager_t* manager) {
    if (!manager) return "Invalid manager";
    return manager->last_error;
}
