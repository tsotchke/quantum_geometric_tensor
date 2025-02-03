#include "quantum_geometric/core/quantum_scheduler.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Static helper function declarations
static bool allocate_task_resources(quantum_task_t* task);
static void release_task_resources(quantum_task_t* task);
static bool validate_task_dependencies(quantum_task_t* task);
static double get_current_time(void);

// Global scheduler state
static struct {
    bool initialized;
    scheduler_config_t config;
    quantum_task_t** task_queue;
    size_t queue_size;
    size_t queue_capacity;
    resource_pool_t** resource_pools;
    size_t num_pools;
    task_callback_t* callbacks;
    scheduler_error_t last_error;
    bool is_running;
    double start_time;
} scheduler = {0};

#define INITIAL_QUEUE_CAPACITY 64
#define MAX_RESOURCE_POOLS 16
#define MAX_CALLBACKS 32

bool initialize_quantum_scheduler(scheduler_config_t* config) {
    if (!config) {
        scheduler.last_error = SCHEDULER_ERROR_INVALID_TASK;
        return false;
    }
    
    // Initialize task queue
    scheduler.task_queue = malloc(INITIAL_QUEUE_CAPACITY * sizeof(quantum_task_t*));
    if (!scheduler.task_queue) {
        scheduler.last_error = SCHEDULER_ERROR_SYSTEM;
        return false;
    }
    
    // Initialize resource pools array
    scheduler.resource_pools = malloc(MAX_RESOURCE_POOLS * sizeof(resource_pool_t*));
    if (!scheduler.resource_pools) {
        free(scheduler.task_queue);
        scheduler.last_error = SCHEDULER_ERROR_SYSTEM;
        return false;
    }
    
    // Initialize callbacks array
    scheduler.callbacks = malloc(MAX_CALLBACKS * sizeof(task_callback_t));
    if (!scheduler.callbacks) {
        free(scheduler.task_queue);
        free(scheduler.resource_pools);
        scheduler.last_error = SCHEDULER_ERROR_SYSTEM;
        return false;
    }
    
    scheduler.config = *config;
    scheduler.queue_size = 0;
    scheduler.queue_capacity = INITIAL_QUEUE_CAPACITY;
    scheduler.num_pools = 0;
    scheduler.initialized = true;
    scheduler.is_running = false;
    scheduler.last_error = SCHEDULER_SUCCESS;
    
    return true;
}

void shutdown_quantum_scheduler(void) {
    if (!scheduler.initialized) return;
    
    // Stop scheduler if running
    if (scheduler.is_running) {
        stop_scheduler();
    }
    
    // Free all tasks in queue
    for (size_t i = 0; i < scheduler.queue_size; i++) {
        quantum_task_t* task = scheduler.task_queue[i];
        if (task) {
            free(task->requirements);
            free(task->dependencies);
            free(task);
        }
    }
    
    // Free resource pools
    for (size_t i = 0; i < scheduler.num_pools; i++) {
        if (scheduler.resource_pools[i]) {
            free(scheduler.resource_pools[i]->allocation_map);
            free(scheduler.resource_pools[i]);
        }
    }
    
    free(scheduler.task_queue);
    free(scheduler.resource_pools);
    free(scheduler.callbacks);
    
    scheduler.initialized = false;
}

quantum_task_t* create_task(computation_node_t* node,
                           execution_priority_t priority) {
    if (!scheduler.initialized || !node) {
        scheduler.last_error = SCHEDULER_ERROR_INVALID_TASK;
        return NULL;
    }
    
    quantum_task_t* task = malloc(sizeof(quantum_task_t));
    if (!task) {
        scheduler.last_error = SCHEDULER_ERROR_SYSTEM;
        return NULL;
    }
    
    task->node = node;
    task->priority = priority;
    task->requirements = NULL;
    task->num_requirements = 0;
    task->dependencies = NULL;
    task->num_dependencies = 0;
    task->status = STATUS_PENDING;
    task->result = NULL;
    task->start_time = 0;
    task->end_time = 0;
    
    return task;
}

bool submit_task(quantum_task_t* task) {
    if (!scheduler.initialized || !task) {
        scheduler.last_error = SCHEDULER_ERROR_INVALID_TASK;
        return false;
    }
    
    // Check queue capacity
    if (scheduler.queue_size >= scheduler.queue_capacity) {
        if (scheduler.queue_size >= scheduler.config.max_queue_size) {
            scheduler.last_error = SCHEDULER_ERROR_QUEUE_FULL;
            return false;
        }
        
        // Resize queue
        size_t new_capacity = scheduler.queue_capacity * 2;
        quantum_task_t** new_queue = realloc(scheduler.task_queue,
                                           new_capacity * sizeof(quantum_task_t*));
        if (!new_queue) {
            scheduler.last_error = SCHEDULER_ERROR_SYSTEM;
            return false;
        }
        
        scheduler.task_queue = new_queue;
        scheduler.queue_capacity = new_capacity;
    }
    
    // Validate task dependencies
    if (!validate_task_dependencies(task)) {
        scheduler.last_error = SCHEDULER_ERROR_INVALID_DEPENDENCY;
        return false;
    }
    
    // Add task to queue
    scheduler.task_queue[scheduler.queue_size++] = task;
    
    // Sort queue by priority if needed
    if (scheduler.config.enable_preemption) {
        reorder_queue(scheduler.config.min_priority);
    }
    
    return true;
}

bool start_scheduler(void) {
    if (!scheduler.initialized) {
        scheduler.last_error = SCHEDULER_ERROR_SYSTEM;
        return false;
    }
    
    if (scheduler.is_running) return true;
    
    scheduler.is_running = true;
    scheduler.start_time = get_current_time();
    
    return true;
}

bool stop_scheduler(void) {
    if (!scheduler.initialized || !scheduler.is_running) return false;
    
    scheduler.is_running = false;
    return true;
}

bool register_resource_pool(resource_pool_t* pool) {
    if (!scheduler.initialized || !pool) {
        scheduler.last_error = SCHEDULER_ERROR_INVALID_TASK;
        return false;
    }
    
    if (scheduler.num_pools >= MAX_RESOURCE_POOLS) {
        scheduler.last_error = SCHEDULER_ERROR_RESOURCE_UNAVAILABLE;
        return false;
    }
    
    // Initialize allocation map
    pool->allocation_map = calloc(pool->total, sizeof(bool));
    if (!pool->allocation_map) {
        scheduler.last_error = SCHEDULER_ERROR_SYSTEM;
        return false;
    }
    
    pool->available = pool->total;
    pool->utilization = 0.0;
    
    scheduler.resource_pools[scheduler.num_pools++] = pool;
    return true;
}

bool allocate_resources(quantum_task_t* task) {
    if (!scheduler.initialized || !task) {
        scheduler.last_error = SCHEDULER_ERROR_INVALID_TASK;
        return false;
    }
    
    for (size_t i = 0; i < task->num_requirements; i++) {
        resource_requirement_t* req = &task->requirements[i];
        bool allocated = false;
        
        // Find appropriate resource pool
        for (size_t j = 0; j < scheduler.num_pools; j++) {
            resource_pool_t* pool = scheduler.resource_pools[j];
            if (pool->type == req->type && pool->available >= req->quantity) {
                // Find contiguous block if exclusive access required
                if (req->exclusive) {
                    size_t start = 0;
                    size_t count = 0;
                    
                    for (size_t k = 0; k < pool->total; k++) {
                        if (!pool->allocation_map[k]) {
                            if (count == 0) start = k;
                            count++;
                            if (count == req->quantity) {
                                // Allocate block
                                for (size_t l = start; l < start + count; l++) {
                                    pool->allocation_map[l] = true;
                                }
                                pool->available -= req->quantity;
                                allocated = true;
                                break;
                            }
                        } else {
                            count = 0;
                        }
                    }
                } else {
                    // Allocate any available resources
                    size_t allocated_count = 0;
                    for (size_t k = 0; k < pool->total && allocated_count < req->quantity; k++) {
                        if (!pool->allocation_map[k]) {
                            pool->allocation_map[k] = true;
                            allocated_count++;
                        }
                    }
                    if (allocated_count == req->quantity) {
                        pool->available -= req->quantity;
                        allocated = true;
                    }
                }
                
                if (allocated) {
                    pool->utilization = (double)(pool->total - pool->available) / pool->total;
                    break;
                }
            }
        }
        
        if (!allocated) {
            // Release any allocated resources
            release_resources(task);
            scheduler.last_error = SCHEDULER_ERROR_RESOURCE_UNAVAILABLE;
            return false;
        }
    }
    
    return true;
}

bool release_resources(quantum_task_t* task) {
    if (!scheduler.initialized || !task) {
        scheduler.last_error = SCHEDULER_ERROR_INVALID_TASK;
        return false;
    }
    
    for (size_t i = 0; i < task->num_requirements; i++) {
        resource_requirement_t* req = &task->requirements[i];
        
        // Find resource pool
        for (size_t j = 0; j < scheduler.num_pools; j++) {
            resource_pool_t* pool = scheduler.resource_pools[j];
            if (pool->type == req->type) {
                size_t released = 0;
                
                // Release allocated resources
                for (size_t k = 0; k < pool->total && released < req->quantity; k++) {
                    if (pool->allocation_map[k]) {
                        pool->allocation_map[k] = false;
                        released++;
                    }
                }
                
                pool->available += released;
                pool->utilization = (double)(pool->total - pool->available) / pool->total;
                break;
            }
        }
    }
    
    return true;
}

bool get_scheduler_metrics(scheduler_metrics_t* metrics) {
    if (!scheduler.initialized || !metrics) return false;
    
    metrics->total_tasks = 0;
    metrics->completed_tasks = 0;
    metrics->failed_tasks = 0;
    metrics->avg_waiting_time = 0;
    metrics->avg_execution_time = 0;
    metrics->resource_utilization = 0;
    metrics->preemptions = 0;
    metrics->scheduling_overhead = 0;
    
    // Calculate metrics
    double total_wait_time = 0;
    double total_exec_time = 0;
    double total_utilization = 0;
    
    for (size_t i = 0; i < scheduler.queue_size; i++) {
        quantum_task_t* task = scheduler.task_queue[i];
        metrics->total_tasks++;
        
        switch (task->status) {
            case STATUS_COMPLETED:
                metrics->completed_tasks++;
                total_exec_time += task->end_time - task->start_time;
                break;
            case STATUS_FAILED:
                metrics->failed_tasks++;
                break;
            default:
                break;
        }
        
        if (task->start_time > 0) {
            total_wait_time += task->start_time - scheduler.start_time;
        }
    }
    
    // Calculate resource utilization
    for (size_t i = 0; i < scheduler.num_pools; i++) {
        total_utilization += scheduler.resource_pools[i]->utilization;
    }
    
    if (metrics->completed_tasks > 0) {
        metrics->avg_execution_time = total_exec_time / metrics->completed_tasks;
    }
    
    if (metrics->total_tasks > 0) {
        metrics->avg_waiting_time = total_wait_time / metrics->total_tasks;
    }
    
    if (scheduler.num_pools > 0) {
        metrics->resource_utilization = total_utilization / scheduler.num_pools;
    }
    
    return true;
}

// Static helper functions
static bool validate_task_dependencies(quantum_task_t* task) {
    if (!task) return false;
    
    // Check for circular dependencies
    for (size_t i = 0; i < task->num_dependencies; i++) {
        quantum_task_t* dep = task->dependencies[i];
        if (!dep) return false;
        
        // Check if dependency is already completed
        if (dep->status == STATUS_COMPLETED) continue;
        
        // Check if dependency depends on this task
        for (size_t j = 0; j < dep->num_dependencies; j++) {
            if (dep->dependencies[j] == task) return false;
        }
    }
    
    return true;
}

static double get_current_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
