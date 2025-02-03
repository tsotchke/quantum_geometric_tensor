#ifndef QUANTUM_SCHEDULER_H
#define QUANTUM_SCHEDULER_H

#include "computational_graph.h"
#include "operation_fusion.h"
#include <stdbool.h>
#include <stddef.h>

// Execution priority levels
typedef enum {
    PRIORITY_LOW,
    PRIORITY_MEDIUM,
    PRIORITY_HIGH,
    PRIORITY_CRITICAL
} execution_priority_t;

// Resource types
typedef enum {
    RESOURCE_CPU,
    RESOURCE_GPU,
    RESOURCE_QPU,
    RESOURCE_MEMORY,
    RESOURCE_BANDWIDTH
} resource_type_t;

// Execution status
typedef enum {
    STATUS_PENDING,
    STATUS_RUNNING,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_CANCELLED
} execution_status_t;

// Resource requirements
typedef struct {
    resource_type_t type;        // Resource type
    size_t quantity;            // Required quantity
    double duration;            // Expected duration
    bool exclusive;             // Requires exclusive access
} resource_requirement_t;

// Task definition
typedef struct quantum_task_t {
    computation_node_t* node;    // Associated computation node
    execution_priority_t priority; // Execution priority
    resource_requirement_t* requirements; // Resource requirements
    size_t num_requirements;    // Number of requirements
    struct quantum_task_t** dependencies; // Task dependencies
    size_t num_dependencies;    // Number of dependencies
    execution_status_t status;  // Current status
    void* result;              // Task result
    double start_time;         // Start timestamp
    double end_time;           // End timestamp
} quantum_task_t;

// Scheduler configuration
typedef struct {
    size_t max_concurrent_tasks;  // Maximum concurrent tasks
    size_t max_queue_size;       // Maximum queue size
    bool enable_preemption;      // Enable task preemption
    bool enable_load_balancing;  // Enable load balancing
    double scheduling_interval;   // Scheduling interval
    execution_priority_t min_priority; // Minimum priority to schedule
} scheduler_config_t;

// Resource pool
typedef struct {
    resource_type_t type;        // Resource type
    size_t total;               // Total quantity
    size_t available;           // Available quantity
    double utilization;         // Current utilization
    bool* allocation_map;       // Resource allocation map
} resource_pool_t;

// Scheduler metrics
typedef struct {
    size_t total_tasks;         // Total tasks
    size_t completed_tasks;     // Completed tasks
    size_t failed_tasks;        // Failed tasks
    double avg_waiting_time;    // Average waiting time
    double avg_execution_time;  // Average execution time
    double resource_utilization; // Resource utilization
    size_t preemptions;        // Number of preemptions
    double scheduling_overhead; // Scheduling overhead
} scheduler_metrics_t;

// Core functions
bool initialize_quantum_scheduler(scheduler_config_t* config);
void shutdown_quantum_scheduler(void);

// Task management
quantum_task_t* create_task(computation_node_t* node,
                           execution_priority_t priority);
bool submit_task(quantum_task_t* task);
bool cancel_task(quantum_task_t* task);
execution_status_t get_task_status(quantum_task_t* task);

// Resource management
bool register_resource_pool(resource_pool_t* pool);
bool allocate_resources(quantum_task_t* task);
bool release_resources(quantum_task_t* task);
double get_resource_utilization(resource_type_t type);

// Scheduling operations
bool start_scheduler(void);
bool stop_scheduler(void);
bool pause_scheduler(void);
bool resume_scheduler(void);

// Queue management
bool reorder_queue(execution_priority_t min_priority);
bool clear_queue(void);
size_t get_queue_size(void);

// Dependency management
bool add_task_dependency(quantum_task_t* task,
                        quantum_task_t* dependency);
bool remove_task_dependency(quantum_task_t* task,
                          quantum_task_t* dependency);
bool validate_dependencies(quantum_task_t* task);

// Performance optimization
bool optimize_schedule(void);
bool balance_load(void);
bool predict_resource_usage(resource_type_t type,
                          double* prediction);

// Event handling
typedef void (*task_callback_t)(quantum_task_t* task);

bool register_task_callback(execution_status_t status,
                          task_callback_t callback);
bool unregister_task_callback(execution_status_t status);

// Monitoring and metrics
bool get_scheduler_metrics(scheduler_metrics_t* metrics);
bool export_metrics(const char* filename);
void print_scheduler_status(void);

// Error handling
typedef enum {
    SCHEDULER_SUCCESS,
    SCHEDULER_ERROR_INVALID_TASK,
    SCHEDULER_ERROR_RESOURCE_UNAVAILABLE,
    SCHEDULER_ERROR_QUEUE_FULL,
    SCHEDULER_ERROR_INVALID_DEPENDENCY,
    SCHEDULER_ERROR_SYSTEM
} scheduler_error_t;

scheduler_error_t get_last_error(void);
const char* get_error_string(scheduler_error_t error);

// Advanced features
bool enable_quantum_priority(void);
bool enable_adaptive_scheduling(void);
bool set_quantum_resource_policy(const char* policy);

#endif // QUANTUM_SCHEDULER_H
