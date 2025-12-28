/**
 * @file task_analyzer.h
 * @brief Quantum task scheduling and analysis system
 *
 * Provides analysis of quantum computational tasks including
 * resource requirements, execution time estimation, and
 * optimal scheduling strategies.
 */

#ifndef TASK_ANALYZER_H
#define TASK_ANALYZER_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct quantum_circuit;
struct ResourcePool;

// Task types
typedef enum {
    TASK_TYPE_CIRCUIT_EXECUTION,
    TASK_TYPE_VARIATIONAL_OPTIMIZATION,
    TASK_TYPE_STATE_PREPARATION,
    TASK_TYPE_MEASUREMENT,
    TASK_TYPE_ERROR_CORRECTION,
    TASK_TYPE_TENSOR_CONTRACTION,
    TASK_TYPE_CLASSICAL_PROCESSING,
    TASK_TYPE_HYBRID_COMPUTATION,
    TASK_TYPE_CALIBRATION,
    TASK_TYPE_BENCHMARK
} TaskType;

// Task priority levels
typedef enum {
    TASK_PRIORITY_LOW = 0,
    TASK_PRIORITY_NORMAL = 1,
    TASK_PRIORITY_HIGH = 2,
    TASK_PRIORITY_CRITICAL = 3,
    TASK_PRIORITY_REALTIME = 4
} TaskPriority;

// Task state
typedef enum {
    TASK_STATE_PENDING,
    TASK_STATE_QUEUED,
    TASK_STATE_SCHEDULED,
    TASK_STATE_RUNNING,
    TASK_STATE_PAUSED,
    TASK_STATE_COMPLETED,
    TASK_STATE_FAILED,
    TASK_STATE_CANCELLED,
    TASK_STATE_TIMEOUT
} TaskState;

// Dependency types
typedef enum {
    DEPENDENCY_DATA,
    DEPENDENCY_CONTROL,
    DEPENDENCY_TEMPORAL,
    DEPENDENCY_RESOURCE
} DependencyType;

// Backend type for execution
typedef enum {
    BACKEND_SIMULATOR,
    BACKEND_IBM_QUANTUM,
    BACKEND_RIGETTI,
    BACKEND_DWAVE,
    BACKEND_IONQ,
    BACKEND_LOCAL_GPU,
    BACKEND_DISTRIBUTED
} BackendType;

// Resource requirements for a task
typedef struct TaskResourceRequirements {
    size_t num_qubits;
    size_t circuit_depth;
    size_t num_gates;
    size_t num_two_qubit_gates;
    size_t num_measurements;
    size_t classical_memory_bytes;
    size_t gpu_memory_bytes;
    double estimated_runtime_ms;
    double estimated_fidelity;
    bool requires_error_correction;
    bool requires_gpu;
    bool requires_distributed;
    size_t num_shots;
    size_t* required_connectivity;
    size_t connectivity_size;
} TaskResourceRequirements;

// Task dependency specification
typedef struct TaskDependency {
    char* dependency_task_id;
    DependencyType type;
    void* data_ref;
} TaskDependency;

// Task dependencies container
typedef struct TaskDependencies {
    TaskDependency* deps;
    size_t num_dependencies;
    size_t capacity;
} TaskDependencies;

// Callback types
typedef void (*TaskCallback)(void* task, void* user_data);
typedef void (*TaskProgressCallback)(void* task, double progress, void* user_data);

// Task definition
typedef struct QuantumTask {
    char* task_id;
    char* task_name;
    TaskType type;
    TaskPriority priority;
    TaskState state;
    TaskResourceRequirements resources;
    TaskDependencies dependencies;
    void* task_data;
    size_t task_data_size;
    struct quantum_circuit* circuit;
    TaskCallback on_complete;
    TaskCallback on_error;
    TaskProgressCallback on_progress;
    void* callback_data;
    uint64_t created_at;
    uint64_t scheduled_at;
    uint64_t started_at;
    uint64_t completed_at;
    uint64_t deadline;
    BackendType assigned_backend;
    char* assigned_backend_name;
    size_t* qubit_mapping;
    size_t mapping_size;
    void* result_data;
    size_t result_size;
    char* error_message;
    int error_code;
    double actual_fidelity;
} QuantumTask;

// Scheduling policies
typedef enum {
    SCHEDULE_FIFO,
    SCHEDULE_PRIORITY,
    SCHEDULE_SHORTEST_JOB_FIRST,
    SCHEDULE_RESOURCE_AWARE,
    SCHEDULE_DEADLINE_DRIVEN,
    SCHEDULE_FIDELITY_OPTIMIZED,
    SCHEDULE_COHERENCE_AWARE
} SchedulingPolicy;

// Analyzer metrics
typedef struct AnalyzerMetrics {
    size_t total_tasks_submitted;
    size_t total_tasks_completed;
    size_t total_tasks_failed;
    size_t current_queue_depth;
    double average_wait_time_ms;
    double average_execution_time_ms;
    double average_fidelity;
    double throughput_tasks_per_sec;
    size_t peak_queue_depth;
    uint64_t total_runtime_ms;
} AnalyzerMetrics;

// Analyzer configuration
typedef struct TaskAnalyzerConfig {
    SchedulingPolicy policy;
    size_t max_queue_size;
    size_t max_concurrent_tasks;
    uint64_t default_timeout_ms;
    bool enable_preemption;
    bool enable_task_fusion;
    bool enable_speculation;
    double fidelity_threshold;
} TaskAnalyzerConfig;

// Task analyzer context
typedef struct TaskAnalyzerContext {
    QuantumTask** task_queue;
    size_t queue_size;
    size_t queue_capacity;
    QuantumTask** running_tasks;
    size_t num_running;
    size_t max_concurrent;
    TaskAnalyzerConfig config;
    struct ResourcePool* resources;
    AnalyzerMetrics metrics;
    pthread_mutex_t lock;
    pthread_cond_t task_available;
    pthread_cond_t task_completed;
    bool shutdown;
    pthread_t scheduler_thread;
} TaskAnalyzerContext;

// Task Analyzer Lifecycle
int task_analyzer_create(TaskAnalyzerContext** ctx, TaskAnalyzerConfig* config,
                          struct ResourcePool* resources);
void task_analyzer_destroy(TaskAnalyzerContext* ctx);
int task_analyzer_start(TaskAnalyzerContext* ctx);
int task_analyzer_stop(TaskAnalyzerContext* ctx);

// Task Creation and Submission
QuantumTask* task_create(const char* name, TaskType type);
void task_destroy(QuantumTask* task);
int task_set_circuit(QuantumTask* task, struct quantum_circuit* circuit);
int task_add_dependency(QuantumTask* task, const char* dep_task_id, DependencyType type);
int task_set_priority(QuantumTask* task, TaskPriority priority);
int task_set_deadline(QuantumTask* task, uint64_t deadline_ms);
int task_set_callbacks(QuantumTask* task, TaskCallback on_complete, TaskCallback on_error,
                        TaskProgressCallback on_progress, void* user_data);
int task_analyzer_submit(TaskAnalyzerContext* ctx, QuantumTask* task);

// Task Analysis
int task_analyzer_analyze_circuit(TaskAnalyzerContext* ctx, struct quantum_circuit* circuit,
                                   TaskResourceRequirements* requirements_out);
int task_analyzer_estimate_runtime(TaskAnalyzerContext* ctx, QuantumTask* task,
                                    BackendType backend, double* runtime_ms_out);
int task_analyzer_estimate_fidelity(TaskAnalyzerContext* ctx, QuantumTask* task,
                                     BackendType backend, double* fidelity_out);
bool task_analyzer_can_schedule(TaskAnalyzerContext* ctx, QuantumTask* task);
bool task_analyzer_dependencies_met(TaskAnalyzerContext* ctx, QuantumTask* task);

// Task Scheduling
int task_analyzer_get_next(TaskAnalyzerContext* ctx, QuantumTask** task_out);
int task_analyzer_schedule(TaskAnalyzerContext* ctx, QuantumTask* task);
int task_analyzer_optimize_schedule(TaskAnalyzerContext* ctx, QuantumTask** tasks,
                                     size_t num_tasks, QuantumTask*** optimized_order_out,
                                     size_t* optimized_count);
int task_analyzer_fuse_tasks(TaskAnalyzerContext* ctx, QuantumTask** tasks,
                              size_t num_tasks, QuantumTask** fused_task_out);

// Task State Management
int task_analyzer_update_state(TaskAnalyzerContext* ctx, const char* task_id, TaskState new_state);
QuantumTask* task_analyzer_get_task(TaskAnalyzerContext* ctx, const char* task_id);
TaskState task_analyzer_get_state(TaskAnalyzerContext* ctx, const char* task_id);
int task_analyzer_cancel(TaskAnalyzerContext* ctx, const char* task_id);
int task_analyzer_pause(TaskAnalyzerContext* ctx, const char* task_id);
int task_analyzer_resume(TaskAnalyzerContext* ctx, const char* task_id);
int task_analyzer_wait(TaskAnalyzerContext* ctx, const char* task_id, uint64_t timeout_ms);
int task_analyzer_wait_all(TaskAnalyzerContext* ctx, const char** task_ids,
                            size_t num_tasks, uint64_t timeout_ms);

// Task Results
int task_analyzer_get_result(TaskAnalyzerContext* ctx, const char* task_id,
                              void** result_out, size_t* result_size_out);
const char* task_analyzer_get_error(TaskAnalyzerContext* ctx, const char* task_id);

// Metrics and Monitoring
int task_analyzer_get_metrics(TaskAnalyzerContext* ctx, AnalyzerMetrics* metrics_out);
void task_analyzer_reset_metrics(TaskAnalyzerContext* ctx);
int task_analyzer_get_queue_status(TaskAnalyzerContext* ctx, size_t* pending_out,
                                    size_t* running_out, size_t* completed_out);
void task_analyzer_print_status(TaskAnalyzerContext* ctx);

// Utility Functions
char* task_generate_id(void);
const char* task_type_name(TaskType type);
const char* task_state_name(TaskState state);
const char* scheduling_policy_name(SchedulingPolicy policy);
QuantumTask* task_clone(const QuantumTask* task);
int task_serialize(const QuantumTask* task, void** buffer, size_t* size);
int task_deserialize(const void* buffer, size_t size, QuantumTask** task_out);

#ifdef __cplusplus
}
#endif

#endif // TASK_ANALYZER_H
