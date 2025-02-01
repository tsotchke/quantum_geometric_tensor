#include <stdlib.h>
#include <stdbool.h>
#include <stddef.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>
#include "quantum_geometric/distributed/workload_distribution.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/config/mpi_config.h"

// Forward declarations of internal functions
size_t distribute_workload(size_t total_size);
size_t get_local_offset(void);
static double estimate_compute_cost(size_t size);
static double estimate_communication_cost(size_t size);
static double estimate_io_cost(size_t size);
static void* balance_workload(void* arg);
static double calculate_local_load(WorkloadManager* manager);
static void update_metrics(WorkloadManager* manager, const WorkItem* item);
static double estimate_work_cost(const void* data, size_t size, WorkItemType type);
static void cleanup_work_queue(WorkQueue* queue);
#ifdef USE_MPI
static int try_steal_work(WorkloadManager* manager);
static void handle_work_request(WorkloadManager* manager, int source);
#endif

// Forward declarations of internal functions
static WorkQueue* init_work_queue(size_t capacity);
static int enqueue_work(WorkQueue* queue, WorkItem* item);
static WorkItem* dequeue_work(WorkQueue* queue);

// Cost estimation functions
// Global variable to store total size for workload distribution
static size_t g_total_size = 0;

// Get local workload size
size_t distribute_workload(size_t total_size) {
    WorkloadManager* manager = init_workload_manager();
    if (!manager) return total_size;  // Fallback to processing everything locally
    
    // Store total size for get_local_offset
    g_total_size = total_size;
    
    size_t chunk_size = total_size / manager->world_size;
    size_t remainder = total_size % manager->world_size;
    
    // Distribute remainder across first few ranks
    if (manager->rank < remainder) {
        chunk_size++;
    }
    
    cleanup_workload_manager(manager);
    return chunk_size;
}

// Get offset for local workload
size_t get_local_offset(void) {
    WorkloadManager* manager = init_workload_manager();
    if (!manager) return 0;
    
    int world_size = manager->world_size;
    int rank = manager->rank;
    
    // Calculate base chunk size
    size_t chunk_size = g_total_size / world_size;
    size_t remainder = g_total_size % world_size;
    
    // Calculate offset based on rank and chunk distribution
    size_t offset = rank * chunk_size;
    
    // Add extra offset for ranks that got an extra item from remainder
    if (rank < remainder) {
        offset += rank;
    } else {
        offset += remainder;
    }
    
    cleanup_workload_manager(manager);
    return offset;
}

static double estimate_compute_cost(size_t size) {
    // Basic compute cost model based on data size
    // Assumes O(n) complexity for basic operations
    return (double)size * 0.001; // 1ms per KB as base cost
}

static double estimate_communication_cost(size_t size) {
    // Network transfer cost model
    // Assumes basic network overhead plus size-dependent transfer time
    const double base_latency = 0.1; // Base network latency in ms
    const double bandwidth_factor = 0.005; // ms per KB transfer rate
    return base_latency + (double)size * bandwidth_factor;
}

static double estimate_io_cost(size_t size) {
    // I/O operation cost model
    // Assumes basic I/O overhead plus size-dependent read/write time
    const double base_io_cost = 0.5; // Base I/O operation cost in ms
    const double io_factor = 0.002; // ms per KB I/O rate
    return base_io_cost + (double)size * io_factor;
}

// Configuration parameters
#define MIN_CHUNK_SIZE 1024
#define MAX_CHUNK_SIZE (16 * 1024 * 1024)
#define LOAD_BALANCE_INTERVAL 100
#define COMM_TAG_DATA 1
#define COMM_TAG_REQUEST 2
#define MAX_QUEUE_SIZE 1024
#define STEAL_THRESHOLD 0.2

// Initialize workload manager
WorkloadManager* init_workload_manager(void) {
    WorkloadManager* manager = malloc(sizeof(WorkloadManager));
    if (!manager) return NULL;
    
#ifdef USE_MPI
    // Get MPI info
    qg_mpi_comm_rank(QG_MPI_COMM_WORLD, &manager->rank);
    qg_mpi_comm_size(QG_MPI_COMM_WORLD, &manager->world_size);
    manager->comm = QG_MPI_COMM_WORLD;
#else
    manager->rank = 0;
    manager->world_size = 1;
#endif
    
    manager->is_coordinator = (manager->rank == 0);
    manager->running = true;
    
    // Initialize work queue
    manager->local_queue = init_work_queue(MAX_QUEUE_SIZE);
    if (!manager->local_queue) {
        free(manager);
        return NULL;
    }
    
    // Allocate load tracking array
    manager->node_loads = calloc(manager->world_size, sizeof(double));
    if (!manager->node_loads) {
        cleanup_work_queue(manager->local_queue);
        free(manager);
        return NULL;
    }
    
    // Initialize metrics
    memset(&manager->metrics, 0, sizeof(WorkMetrics));
    manager->total_items_processed = 0;
    
    // Start load balancing thread
    if (pthread_create(&manager->balance_thread,
                      NULL,
                      balance_workload,
                      manager) != 0) {
        cleanup_work_queue(manager->local_queue);
        free(manager->node_loads);
        free(manager);
        return NULL;
    }
    
    return manager;
}

// Initialize work queue
static WorkQueue* init_work_queue(size_t capacity) {
    WorkQueue* queue = malloc(sizeof(WorkQueue));
    if (!queue) return NULL;
    
    queue->items = malloc(capacity * sizeof(WorkItem));
    if (!queue->items) {
        free(queue);
        return NULL;
    }
    
    queue->capacity = capacity;
    queue->size = 0;
    queue->head = 0;
    queue->tail = 0;
    
    if (pthread_mutex_init(&queue->mutex, NULL) != 0 ||
        pthread_cond_init(&queue->not_empty, NULL) != 0 ||
        pthread_cond_init(&queue->not_full, NULL) != 0) {
        free(queue->items);
        free(queue);
        return NULL;
    }
    
    return queue;
}

// Submit work item
int submit_work(WorkloadManager* manager,
               const void* data,
               size_t size,
               WorkItemType type,
               int priority) {
    if (!manager || !data) return -1;
    
    WorkItem item = {
        .data = malloc(size),
        .size = size,
        .type = type,
        .priority = priority,
        .cost_estimate = estimate_work_cost(data, size, type)
    };
    
    if (!item.data) return -1;
    memcpy(item.data, data, size);
    
    // Add to local queue
    pthread_mutex_lock(&manager->local_queue->mutex);
    
    while (manager->local_queue->size >= manager->local_queue->capacity) {
        pthread_cond_wait(&manager->local_queue->not_full,
                         &manager->local_queue->mutex);
    }
    
    // Insert based on priority
    size_t insert_pos = manager->local_queue->tail;
    for (size_t i = 0; i < manager->local_queue->size; i++) {
        size_t idx = (manager->local_queue->head + i) %
                    manager->local_queue->capacity;
        if (item.priority > manager->local_queue->items[idx].priority) {
            insert_pos = idx;
            break;
        }
    }
    
    // Shift items if needed
    if (insert_pos != manager->local_queue->tail) {
        memmove(&manager->local_queue->items[insert_pos + 1],
                &manager->local_queue->items[insert_pos],
                (manager->local_queue->size - insert_pos) *
                sizeof(WorkItem));
    }
    
    manager->local_queue->items[insert_pos] = item;
    manager->local_queue->size++;
    manager->local_queue->tail = (manager->local_queue->tail + 1) %
                                manager->local_queue->capacity;
    
    pthread_cond_signal(&manager->local_queue->not_empty);
    pthread_mutex_unlock(&manager->local_queue->mutex);
    
    return 0;
}

// Get work item
int get_work(WorkloadManager* manager,
            void** data,
            size_t* size,
            WorkItemType* type) {
    if (!manager || !data || !size || !type) return -1;
    
    pthread_mutex_lock(&manager->local_queue->mutex);
    
    while (manager->local_queue->size == 0) {
#ifdef USE_MPI
        // Try work stealing
        if (try_steal_work(manager) == 0) {
            continue;
        }
#endif
        pthread_cond_wait(&manager->local_queue->not_empty,
                         &manager->local_queue->mutex);
    }
    
    // Get highest priority item
    WorkItem item = manager->local_queue->items[manager->local_queue->head];
    manager->local_queue->head = (manager->local_queue->head + 1) %
                                manager->local_queue->capacity;
    manager->local_queue->size--;
    
    pthread_cond_signal(&manager->local_queue->not_full);
    pthread_mutex_unlock(&manager->local_queue->mutex);
    
    *data = item.data;
    *size = item.size;
    *type = item.type;
    
    // Update metrics
    update_metrics(manager, &item);
    
    return 0;
}

#ifdef USE_MPI
// Try to steal work from other nodes
static int try_steal_work(WorkloadManager* manager) {
    // Find most loaded node
    int target = -1;
    double max_load = 0.0;
    
    for (int i = 0; i < manager->world_size; i++) {
        if (i != manager->rank && manager->node_loads[i] > max_load) {
            max_load = manager->node_loads[i];
            target = i;
        }
    }
    
    if (target == -1 || max_load < STEAL_THRESHOLD) {
        return -1;
    }
    
    // Request work
    qg_mpi_send(NULL, 0, QG_MPI_BYTE, target, COMM_TAG_REQUEST,
             manager->comm);
    
    // Receive response
    qg_mpi_status_t status;
    int count;
    qg_mpi_probe(target, COMM_TAG_DATA, manager->comm, &status);
    qg_mpi_get_count(&status, QG_MPI_BYTE, &count);
    
    if (count == 0) return -1;  // No work available
    
    // Receive work item
    WorkItem item;
    item.data = malloc(count);
    if (!item.data) return -1;
    
    qg_mpi_recv(item.data, count, QG_MPI_BYTE, target,
             COMM_TAG_DATA, manager->comm, &status);
    
    // Add to local queue
    pthread_mutex_lock(&manager->local_queue->mutex);
    
    manager->local_queue->items[manager->local_queue->tail] = item;
    manager->local_queue->size++;
    manager->local_queue->tail = (manager->local_queue->tail + 1) %
                                manager->local_queue->capacity;
    
    pthread_mutex_unlock(&manager->local_queue->mutex);
    
    return 0;
}
#endif

// Balance workload thread
static void* balance_workload(void* arg) {
    WorkloadManager* manager = (WorkloadManager*)arg;
    
    while (manager->running) {
#ifdef USE_MPI
        // Gather load information
        double local_load = calculate_local_load(manager);
        
        qg_mpi_allgather(&local_load, 1, QG_MPI_DOUBLE,
                     manager->node_loads, 1, QG_MPI_DOUBLE,
                     manager->comm);
        
        // Check for work requests
        qg_mpi_status_t status;
        int flag;
        
        qg_mpi_iprobe(QG_MPI_ANY_SOURCE, COMM_TAG_REQUEST,
                  manager->comm, &flag, &status);
        
        if (flag) {
            handle_work_request(manager, status.MPI_SOURCE);
        }
#endif
        // Sleep before next balance
        usleep(LOAD_BALANCE_INTERVAL * 1000);
    }
    
    return NULL;
}

// Calculate local load
static double calculate_local_load(WorkloadManager* manager) {
    pthread_mutex_lock(&manager->local_queue->mutex);
    double load = (double)manager->local_queue->size /
                 manager->local_queue->capacity;
    pthread_mutex_unlock(&manager->local_queue->mutex);
    
    return load;
}

#ifdef USE_MPI
// Handle work request
static void handle_work_request(WorkloadManager* manager,
                              int source) {
    // Receive request
    qg_mpi_recv(NULL, 0, QG_MPI_BYTE, source, COMM_TAG_REQUEST,
             manager->comm, QG_MPI_STATUS_IGNORE);
    
    pthread_mutex_lock(&manager->local_queue->mutex);
    
    if (manager->local_queue->size == 0) {
        // No work available
        qg_mpi_send(NULL, 0, QG_MPI_BYTE, source, COMM_TAG_DATA,
                manager->comm);
    } else {
        // Share half of remaining work
        size_t share = manager->local_queue->size / 2;
        WorkItem* items = malloc(share * sizeof(WorkItem));
        
        for (size_t i = 0; i < share; i++) {
            items[i] = manager->local_queue->items[
                manager->local_queue->head + i];
        }
        
        // Update local queue
        memmove(&manager->local_queue->items[manager->local_queue->head],
                &manager->local_queue->items[manager->local_queue->head + share],
                (manager->local_queue->size - share) * sizeof(WorkItem));
        
        manager->local_queue->size -= share;
        
        // Send work
        qg_mpi_send(items, share * sizeof(WorkItem), QG_MPI_BYTE,
                source, COMM_TAG_DATA, manager->comm);
        
        free(items);
    }
    
    pthread_mutex_unlock(&manager->local_queue->mutex);
}
#endif

// Update metrics
static void update_metrics(WorkloadManager* manager,
                         const WorkItem* item) {
    manager->total_items_processed++;
    
    // Update type-specific metrics
    switch (item->type) {
        case WORK_COMPUTE:
            manager->metrics.compute_time += item->cost_estimate;
            break;
        case WORK_COMMUNICATION:
            manager->metrics.network_time += item->cost_estimate;
            break;
        case WORK_IO:
            manager->metrics.io_time += item->cost_estimate;
            break;
        case WORK_CUSTOM:
            // Custom work items are tracked separately
            manager->metrics.custom_time += item->cost_estimate;
            break;
    }
    
    // Update general metrics
    manager->metrics.total_time += item->cost_estimate;
    manager->metrics.average_cost = manager->metrics.total_time /
                                  manager->total_items_processed;
}

// Estimate work cost
static double estimate_work_cost(const void* data,
                               size_t size,
                               WorkItemType type) {
    // Cost model depends on work type
    switch (type) {
        case WORK_COMPUTE:
            return estimate_compute_cost(size);
        case WORK_COMMUNICATION:
            return estimate_communication_cost(size);
        case WORK_IO:
            return estimate_io_cost(size);
        case WORK_CUSTOM:
            // Custom work items use a default cost estimate
            return 1.0;
        default:
            return 1.0;
    }
}

// Clean up
void cleanup_workload_manager(WorkloadManager* manager) {
    if (!manager) return;
    
    manager->running = false;
    pthread_join(manager->balance_thread, NULL);
    
    cleanup_work_queue(manager->local_queue);
    free(manager->node_loads);
    free(manager);
}

// Queue operations
static int enqueue_work(WorkQueue* queue, WorkItem* item) {
    if (!queue || !item || queue->size >= queue->capacity) {
        return -1;
    }
    
    queue->items[queue->tail] = *item;
    queue->tail = (queue->tail + 1) % queue->capacity;
    queue->size++;
    return 0;
}

static WorkItem* dequeue_work(WorkQueue* queue) {
    if (!queue || queue->size == 0) {
        return NULL;
    }
    
    WorkItem* item = &queue->items[queue->head];
    queue->head = (queue->head + 1) % queue->capacity;
    queue->size--;
    return item;
}

static void cleanup_work_queue(WorkQueue* queue) {
    if (!queue) return;
    
    for (size_t i = 0; i < queue->size; i++) {
        free(queue->items[(queue->head + i) % queue->capacity].data);
    }
    
    pthread_mutex_destroy(&queue->mutex);
    pthread_cond_destroy(&queue->not_empty);
    pthread_cond_destroy(&queue->not_full);
    
    free(queue->items);
    free(queue);
}
