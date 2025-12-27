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
    if ((size_t)manager->rank < remainder) {
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
    if ((size_t)rank < remainder) {
        offset += (size_t)rank;
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

// ============================================================================
// Production Distribution API Implementation
// ============================================================================

// Thread-local distribution context
static __thread DistributionContext* tl_current_dist = NULL;
static pthread_mutex_t g_dist_mutex = PTHREAD_MUTEX_INITIALIZER;

// Get current local workload size (after distribute_workload called)
size_t get_local_workload_size(void) {
    pthread_mutex_lock(&g_dist_mutex);
    size_t result = tl_current_dist ? tl_current_dist->local_size : g_total_size;
    pthread_mutex_unlock(&g_dist_mutex);
    return result;
}

// Initialize distribution context
DistributionContext* init_distribution(size_t n, size_t element_size, DistributionStrategy strategy) {
    DistributionContext* ctx = malloc(sizeof(DistributionContext));
    if (!ctx) return NULL;

    memset(ctx, 0, sizeof(DistributionContext));
    ctx->total_size = n;
    ctx->element_size = element_size;
    ctx->strategy = strategy;

#ifdef USE_MPI
    qg_mpi_comm_rank(QG_MPI_COMM_WORLD, &ctx->rank);
    qg_mpi_comm_size(QG_MPI_COMM_WORLD, &ctx->world_size);
#else
    ctx->rank = 0;
    ctx->world_size = 1;
#endif

    // Allocate partition arrays
    ctx->node_offsets = calloc(ctx->world_size, sizeof(size_t));
    ctx->node_sizes = calloc(ctx->world_size, sizeof(size_t));
    if (!ctx->node_offsets || !ctx->node_sizes) {
        free(ctx->node_offsets);
        free(ctx->node_sizes);
        free(ctx);
        return NULL;
    }

    struct timeval start, end;
    gettimeofday(&start, NULL);

    // Compute distribution based on strategy
    switch (strategy) {
        case DIST_STRATEGY_BLOCK: {
            // Block distribution: equal contiguous chunks
            size_t base_chunk = n / ctx->world_size;
            size_t remainder = n % ctx->world_size;
            size_t offset = 0;

            for (int i = 0; i < ctx->world_size; i++) {
                ctx->node_offsets[i] = offset;
                ctx->node_sizes[i] = base_chunk + (i < (int)remainder ? 1 : 0);
                offset += ctx->node_sizes[i];
            }
            break;
        }

        case DIST_STRATEGY_CYCLIC: {
            // Cyclic distribution: round-robin element assignment
            // Each node gets every world_size-th element
            // max_per_node is the maximum elements any single node will handle
            size_t max_per_node = (n + ctx->world_size - 1) / ctx->world_size;
            for (int i = 0; i < ctx->world_size; i++) {
                ctx->node_offsets[i] = i;  // Starting element
                // Each node's actual count may be less than max if near end
                size_t this_count = (n > (size_t)i) ?
                    ((n - i - 1) / ctx->world_size + 1) : 0;
                ctx->node_sizes[i] = (this_count <= max_per_node) ? this_count : max_per_node;
            }
            break;
        }

        case DIST_STRATEGY_BLOCK_CYCLIC: {
            // Block-cyclic: blocks of BLOCK_SIZE distributed round-robin
            const size_t BLOCK_SIZE = 64;
            size_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            size_t blocks_per_node = num_blocks / ctx->world_size;
            size_t extra_blocks = num_blocks % ctx->world_size;

            size_t offset = 0;
            for (int i = 0; i < ctx->world_size; i++) {
                size_t node_blocks = blocks_per_node + (i < (int)extra_blocks ? 1 : 0);
                ctx->node_offsets[i] = offset;
                ctx->node_sizes[i] = node_blocks * BLOCK_SIZE;
                if (i == ctx->world_size - 1) {
                    // Last node may have partial final block
                    ctx->node_sizes[i] = n - offset;
                }
                offset += ctx->node_sizes[i];
            }
            break;
        }

        case DIST_STRATEGY_ADAPTIVE: {
            // Adaptive distribution - requires load information
            // Fall back to block for now, use init_adaptive_distribution for full support
            size_t base_chunk = n / ctx->world_size;
            size_t remainder = n % ctx->world_size;
            size_t offset = 0;

            for (int i = 0; i < ctx->world_size; i++) {
                ctx->node_offsets[i] = offset;
                ctx->node_sizes[i] = base_chunk + (i < (int)remainder ? 1 : 0);
                offset += ctx->node_sizes[i];
            }
            break;
        }
    }

    // Set local values
    ctx->local_offset = ctx->node_offsets[ctx->rank];
    ctx->local_size = ctx->node_sizes[ctx->rank];

    gettimeofday(&end, NULL);
    ctx->distribution_time = (end.tv_sec - start.tv_sec) * 1000.0 +
                            (end.tv_usec - start.tv_usec) / 1000.0;

    // Calculate imbalance factor
    double max_size = 0, min_size = n;
    for (int i = 0; i < ctx->world_size; i++) {
        if (ctx->node_sizes[i] > max_size) max_size = ctx->node_sizes[i];
        if (ctx->node_sizes[i] < min_size) min_size = ctx->node_sizes[i];
    }
    ctx->imbalance_factor = (min_size > 0) ? (max_size / min_size) : 1.0;

    return ctx;
}

// Adaptive distribution based on node loads
DistributionContext* init_adaptive_distribution(size_t n, size_t element_size,
                                                const NodeLoadInfo* node_loads,
                                                int num_nodes) {
    DistributionContext* ctx = malloc(sizeof(DistributionContext));
    if (!ctx) return NULL;

    memset(ctx, 0, sizeof(DistributionContext));
    ctx->total_size = n;
    ctx->element_size = element_size;
    ctx->strategy = DIST_STRATEGY_ADAPTIVE;

#ifdef USE_MPI
    qg_mpi_comm_rank(QG_MPI_COMM_WORLD, &ctx->rank);
    qg_mpi_comm_size(QG_MPI_COMM_WORLD, &ctx->world_size);
#else
    ctx->rank = 0;
    ctx->world_size = 1;
#endif

    if (num_nodes > 0 && num_nodes != ctx->world_size) {
        ctx->world_size = num_nodes;
    }

    // Allocate partition arrays
    ctx->node_offsets = calloc(ctx->world_size, sizeof(size_t));
    ctx->node_sizes = calloc(ctx->world_size, sizeof(size_t));
    if (!ctx->node_offsets || !ctx->node_sizes) {
        free(ctx->node_offsets);
        free(ctx->node_sizes);
        free(ctx);
        return NULL;
    }

    struct timeval start, end;
    gettimeofday(&start, NULL);

    // Compute adaptive distribution
    double total_capacity = 0.0;
    double* capacities = malloc(ctx->world_size * sizeof(double));
    if (!capacities) {
        cleanup_distribution(ctx);
        return NULL;
    }

    // Calculate effective capacity for each node
    for (int i = 0; i < ctx->world_size; i++) {
        if (node_loads && i < num_nodes) {
            // Factor in compute capacity and current load
            capacities[i] = node_loads[i].compute_capacity *
                           (1.0 - 0.5 * node_loads[i].current_load);
            if (capacities[i] < 0.1) capacities[i] = 0.1;  // Minimum capacity
        } else {
            capacities[i] = 1.0;  // Default capacity
        }
        total_capacity += capacities[i];
    }

    // Distribute work proportionally to capacity
    size_t assigned = 0;
    for (int i = 0; i < ctx->world_size; i++) {
        ctx->node_offsets[i] = assigned;
        if (i == ctx->world_size - 1) {
            ctx->node_sizes[i] = n - assigned;
        } else {
            ctx->node_sizes[i] = (size_t)((double)n * capacities[i] / total_capacity);
        }
        assigned += ctx->node_sizes[i];
    }

    free(capacities);

    // Set local values
    ctx->local_offset = ctx->node_offsets[ctx->rank];
    ctx->local_size = ctx->node_sizes[ctx->rank];

    gettimeofday(&end, NULL);
    ctx->distribution_time = (end.tv_sec - start.tv_sec) * 1000.0 +
                            (end.tv_usec - start.tv_usec) / 1000.0;

    // Calculate imbalance factor (capacity-weighted)
    double weighted_max = 0, weighted_min = n;
    for (int i = 0; i < ctx->world_size; i++) {
        double weighted = ctx->node_sizes[i] / (node_loads ? node_loads[i].compute_capacity : 1.0);
        if (weighted > weighted_max) weighted_max = weighted;
        if (weighted < weighted_min) weighted_min = weighted;
    }
    ctx->imbalance_factor = (weighted_min > 0) ? (weighted_max / weighted_min) : 1.0;

    return ctx;
}

// Cleanup distribution context
void cleanup_distribution(DistributionContext* ctx) {
    if (!ctx) return;
    free(ctx->node_offsets);
    free(ctx->node_sizes);
    free(ctx);
}

// Synchronize results across all nodes
void synchronize_results(void* results, size_t n) {
#ifdef USE_MPI
    if (!results) return;

    // Get current distribution info
    int rank, world_size;
    qg_mpi_comm_rank(QG_MPI_COMM_WORLD, &rank);
    qg_mpi_comm_size(QG_MPI_COMM_WORLD, &world_size);

    if (world_size <= 1) return;

    // Calculate local portion info
    size_t base_chunk = n / world_size;
    size_t remainder = n % world_size;

    int* recvcounts = malloc(world_size * sizeof(int));
    int* displs = malloc(world_size * sizeof(int));
    if (!recvcounts || !displs) {
        free(recvcounts);
        free(displs);
        return;
    }

    size_t offset = 0;
    for (int i = 0; i < world_size; i++) {
        size_t count = base_chunk + (i < (int)remainder ? 1 : 0);
        recvcounts[i] = (int)count;
        displs[i] = (int)offset;
        offset += count;
    }

    // Use MPI_Allgatherv for variable-sized partitions
    size_t local_count = recvcounts[rank];
    void* sendbuf = (char*)results + displs[rank];

    qg_mpi_allgatherv(sendbuf, local_count, QG_MPI_BYTE,
                      results, recvcounts, displs, QG_MPI_BYTE,
                      QG_MPI_COMM_WORLD);

    free(recvcounts);
    free(displs);
#else
    // No MPI - results are already local
    (void)results;
    (void)n;
#endif
}

// Synchronize complex results
void synchronize_complex_results(double _Complex* results, size_t n) {
    synchronize_results(results, n * sizeof(double _Complex));
}

// Extended synchronize with options
int synchronize_results_ex(void* results, size_t n, size_t element_size,
                          const SyncOptions* options) {
    if (!results) return -1;

#ifdef USE_MPI
    int rank, world_size;
    qg_mpi_comm_rank(QG_MPI_COMM_WORLD, &rank);
    qg_mpi_comm_size(QG_MPI_COMM_WORLD, &world_size);

    if (world_size <= 1) return 0;

    size_t total_bytes = n * element_size;
    size_t chunk_size = options && options->chunk_size > 0 ?
                        options->chunk_size : total_bytes;

    // For large transfers, use chunked approach
    if (total_bytes > chunk_size) {
        size_t offset = 0;
        while (offset < total_bytes) {
            size_t this_chunk = (offset + chunk_size > total_bytes) ?
                               (total_bytes - offset) : chunk_size;

            qg_mpi_allgather((char*)results + offset, this_chunk, QG_MPI_BYTE,
                            (char*)results + offset, this_chunk, QG_MPI_BYTE,
                            QG_MPI_COMM_WORLD);
            offset += this_chunk;
        }
    } else {
        qg_mpi_allgather(results, total_bytes, QG_MPI_BYTE,
                        results, total_bytes, QG_MPI_BYTE,
                        QG_MPI_COMM_WORLD);
    }

    return 0;
#else
    (void)options;
    (void)element_size;
    return 0;
#endif
}

// Scatter data from coordinator
int scatter_data(const void* sendbuf, void* recvbuf, size_t n, size_t element_size) {
#ifdef USE_MPI
    int rank, world_size;
    qg_mpi_comm_rank(QG_MPI_COMM_WORLD, &rank);
    qg_mpi_comm_size(QG_MPI_COMM_WORLD, &world_size);

    size_t chunk_size = n / world_size;
    qg_mpi_scatter(sendbuf, chunk_size * element_size, QG_MPI_BYTE,
                  recvbuf, chunk_size * element_size, QG_MPI_BYTE,
                  0, QG_MPI_COMM_WORLD);
    return 0;
#else
    memcpy(recvbuf, sendbuf, n * element_size);
    return 0;
#endif
}

// Gather data to coordinator
int gather_data(const void* sendbuf, void* recvbuf, size_t n, size_t element_size) {
#ifdef USE_MPI
    int rank, world_size;
    qg_mpi_comm_rank(QG_MPI_COMM_WORLD, &rank);
    qg_mpi_comm_size(QG_MPI_COMM_WORLD, &world_size);

    size_t chunk_size = n / world_size;
    qg_mpi_gather(sendbuf, chunk_size * element_size, QG_MPI_BYTE,
                 recvbuf, chunk_size * element_size, QG_MPI_BYTE,
                 0, QG_MPI_COMM_WORLD);
    return 0;
#else
    memcpy(recvbuf, sendbuf, n * element_size);
    return 0;
#endif
}

// Get distributed rank
int get_distributed_rank(void) {
#ifdef USE_MPI
    int rank;
    qg_mpi_comm_rank(QG_MPI_COMM_WORLD, &rank);
    return rank;
#else
    return 0;
#endif
}

// Get distributed size
int get_distributed_size(void) {
#ifdef USE_MPI
    int size;
    qg_mpi_comm_size(QG_MPI_COMM_WORLD, &size);
    return size;
#else
    return 1;
#endif
}

// Check if coordinator
bool is_coordinator(void) {
    return get_distributed_rank() == 0;
}

// Barrier synchronization
void distributed_barrier(void) {
#ifdef USE_MPI
    qg_mpi_barrier(QG_MPI_COMM_WORLD);
#endif
}

// Reduce operation
int distributed_reduce(const void* sendbuf, void* recvbuf, size_t count,
                       size_t element_size, int op) {
#ifdef USE_MPI
    qg_mpi_op_t mpi_op;
    switch (op) {
        case 0: mpi_op = QG_MPI_SUM; break;
        case 1: mpi_op = QG_MPI_MAX; break;
        case 2: mpi_op = QG_MPI_MIN; break;
        case 3: mpi_op = QG_MPI_PROD; break;
        default: return -1;
    }

    qg_mpi_datatype_t dtype;
    if (element_size == sizeof(double)) {
        dtype = QG_MPI_DOUBLE;
    } else if (element_size == sizeof(float)) {
        dtype = QG_MPI_FLOAT;
    } else if (element_size == sizeof(int)) {
        dtype = QG_MPI_INT;
    } else {
        return -1;  // Unsupported type
    }

    qg_mpi_allreduce(sendbuf, recvbuf, count, dtype, mpi_op, QG_MPI_COMM_WORLD);
    return 0;
#else
    memcpy(recvbuf, sendbuf, count * element_size);
    return 0;
#endif
}

// Get node load information
int get_node_load_info(NodeLoadInfo* info) {
    if (!info) return -1;

    info->rank = get_distributed_rank();
    info->compute_capacity = 1.0;  // Baseline

    // Get system load average
    double loadavg[3];
    if (getloadavg(loadavg, 3) >= 1) {
        info->current_load = loadavg[0] / (double)sysconf(_SC_NPROCESSORS_ONLN);
        if (info->current_load > 1.0) info->current_load = 1.0;
    } else {
        info->current_load = 0.5;  // Default
    }

    // Get available memory (platform-specific)
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    info->memory_available = (double)pages * page_size;

    // Network info defaults (would need actual measurement)
    info->network_bandwidth = 1e9;   // 1 GB/s default
    info->latency = 0.001;           // 1ms default

    return 0;
}

// Collect all node loads
int collect_all_node_loads(NodeLoadInfo* infos) {
#ifdef USE_MPI
    int world_size = get_distributed_size();

    NodeLoadInfo local_info;
    get_node_load_info(&local_info);

    qg_mpi_allgather(&local_info, sizeof(NodeLoadInfo), QG_MPI_BYTE,
                    infos, sizeof(NodeLoadInfo), QG_MPI_BYTE,
                    QG_MPI_COMM_WORLD);
    return 0;
#else
    return get_node_load_info(&infos[0]);
#endif
}

// Renamed to avoid conflict with workload_balancer.c (this is for distributed MPI contexts)
int rebalance_distributed_workload(DistributionContext* ctx, void* data, size_t element_size) {
    if (!ctx || !data) return -1;

#ifdef USE_MPI
    // Collect current load information
    NodeLoadInfo* loads = malloc(ctx->world_size * sizeof(NodeLoadInfo));
    if (!loads) return -1;

    collect_all_node_loads(loads);

    // Create new distribution based on loads
    DistributionContext* new_ctx = init_adaptive_distribution(
        ctx->total_size, element_size, loads, ctx->world_size);

    if (!new_ctx) {
        free(loads);
        return -1;
    }

    // Check if redistribution is needed (>20% imbalance change)
    if (fabs(new_ctx->imbalance_factor - ctx->imbalance_factor) < 0.2) {
        cleanup_distribution(new_ctx);
        free(loads);
        return 0;  // No significant change needed
    }

    // Perform data redistribution using MPI_Alltoallv
    // This is complex - simplified version here

    // Copy new distribution to current context
    ctx->local_offset = new_ctx->local_offset;
    ctx->local_size = new_ctx->local_size;
    memcpy(ctx->node_offsets, new_ctx->node_offsets, ctx->world_size * sizeof(size_t));
    memcpy(ctx->node_sizes, new_ctx->node_sizes, ctx->world_size * sizeof(size_t));
    ctx->imbalance_factor = new_ctx->imbalance_factor;

    cleanup_distribution(new_ctx);
    free(loads);
    return 0;
#else
    (void)element_size;
    return 0;
#endif
}
