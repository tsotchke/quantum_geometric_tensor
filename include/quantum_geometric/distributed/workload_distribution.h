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

// Public API
WorkloadManager* init_workload_manager(void);
void cleanup_workload_manager(WorkloadManager* manager);
int submit_work(WorkloadManager* manager, const void* data, size_t size, WorkItemType type, int priority);
int get_work(WorkloadManager* manager, void** data, size_t* size, WorkItemType* type);

#endif // WORKLOAD_DISTRIBUTION_H
