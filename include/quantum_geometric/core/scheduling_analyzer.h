/**
 * @file scheduling_analyzer.h
 * @brief Scheduling Optimization Analysis for Quantum Operations
 *
 * Provides scheduling analysis and optimization including:
 * - Operation dependency analysis
 * - Critical path identification
 * - Resource-aware scheduling
 * - Parallel execution optimization
 * - Schedule quality metrics
 * - Deadlock detection
 *
 * Part of the QGTL Monitoring Framework.
 */

#ifndef SCHEDULING_ANALYZER_H
#define SCHEDULING_ANALYZER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================

#define SCHEDULING_MAX_NAME_LENGTH 128
#define SCHEDULING_MAX_OPERATIONS 4096
#define SCHEDULING_MAX_RESOURCES 256
#define SCHEDULING_MAX_DEPENDENCIES 16384

// ============================================================================
// Enumerations
// ============================================================================

/**
 * Scheduling strategies
 */
typedef enum {
    SCHEDULING_STRATEGY_FIFO,         // First-in-first-out
    SCHEDULING_STRATEGY_LIFO,         // Last-in-first-out
    SCHEDULING_STRATEGY_PRIORITY,     // Priority-based
    SCHEDULING_STRATEGY_DEADLINE,     // Deadline-driven
    SCHEDULING_STRATEGY_CRITICAL_PATH,// Critical path method
    SCHEDULING_STRATEGY_RESOURCE_LEVEL,// Resource leveling
    SCHEDULING_STRATEGY_GENETIC,      // Genetic algorithm
    SCHEDULING_STRATEGY_CUSTOM        // Custom strategy
} scheduling_strategy_t;

/**
 * Operation states
 */
typedef enum {
    SCHEDULING_OP_PENDING,            // Waiting to be scheduled
    SCHEDULING_OP_READY,              // Dependencies met
    SCHEDULING_OP_RUNNING,            // Currently executing
    SCHEDULING_OP_COMPLETED,          // Finished execution
    SCHEDULING_OP_BLOCKED,            // Blocked on dependency
    SCHEDULING_OP_FAILED,             // Execution failed
    SCHEDULING_OP_CANCELLED           // Cancelled
} scheduling_op_state_t;

/**
 * Resource types
 */
typedef enum {
    SCHEDULING_RESOURCE_CPU,
    SCHEDULING_RESOURCE_GPU,
    SCHEDULING_RESOURCE_MEMORY,
    SCHEDULING_RESOURCE_QUBIT,
    SCHEDULING_RESOURCE_NETWORK,
    SCHEDULING_RESOURCE_CUSTOM
} scheduling_resource_type_t;

/**
 * Dependency types
 */
typedef enum {
    SCHEDULING_DEP_FINISH_TO_START,   // A finishes before B starts
    SCHEDULING_DEP_START_TO_START,    // A and B start together
    SCHEDULING_DEP_FINISH_TO_FINISH,  // A and B finish together
    SCHEDULING_DEP_START_TO_FINISH,   // A starts before B finishes
    SCHEDULING_DEP_DATA,              // Data dependency
    SCHEDULING_DEP_RESOURCE           // Resource dependency
} scheduling_dependency_type_t;

/**
 * Schedule quality levels
 */
typedef enum {
    SCHEDULING_QUALITY_OPTIMAL,       // Known optimal
    SCHEDULING_QUALITY_NEAR_OPTIMAL,  // Within 5% of optimal
    SCHEDULING_QUALITY_GOOD,          // Within 15% of optimal
    SCHEDULING_QUALITY_ACCEPTABLE,    // Within 30% of optimal
    SCHEDULING_QUALITY_POOR,          // > 30% from optimal
    SCHEDULING_QUALITY_UNKNOWN        // Unknown quality
} scheduling_quality_t;

// ============================================================================
// Data Structures
// ============================================================================

/**
 * Operation descriptor
 */
typedef struct {
    uint32_t op_id;
    char name[SCHEDULING_MAX_NAME_LENGTH];
    scheduling_op_state_t state;
    uint32_t priority;                // Higher = more important
    uint64_t estimated_duration_ns;
    uint64_t actual_duration_ns;
    uint64_t deadline_ns;             // 0 if no deadline
    uint64_t scheduled_start_ns;
    uint64_t actual_start_ns;
    uint64_t completion_time_ns;
    double resource_requirements[SCHEDULING_MAX_RESOURCES];
    size_t num_resource_reqs;
} scheduling_operation_t;

/**
 * Dependency descriptor
 */
typedef struct {
    uint32_t from_op_id;
    uint32_t to_op_id;
    scheduling_dependency_type_t type;
    uint64_t lag_ns;                  // Minimum lag time
} scheduling_dependency_t;

/**
 * Resource descriptor
 */
typedef struct {
    uint32_t resource_id;
    char name[SCHEDULING_MAX_NAME_LENGTH];
    scheduling_resource_type_t type;
    double total_capacity;
    double available_capacity;
    double peak_utilization;
    uint64_t busy_time_ns;
} scheduling_resource_t;

/**
 * Schedule entry
 */
typedef struct {
    uint32_t op_id;
    uint32_t resource_id;
    uint64_t start_ns;
    uint64_t end_ns;
    double resource_allocated;
} scheduling_entry_t;

/**
 * Critical path information
 */
typedef struct {
    uint32_t* operations;             // Operation IDs on critical path
    size_t num_operations;
    uint64_t total_duration_ns;
    double slack_ns;                  // Total slack in schedule
} scheduling_critical_path_t;

/**
 * Schedule metrics
 */
typedef struct {
    uint64_t makespan_ns;             // Total schedule duration
    double average_utilization;
    double peak_utilization;
    uint64_t total_idle_time_ns;
    uint64_t total_wait_time_ns;
    size_t num_parallel_ops;
    double parallelism_degree;
    size_t missed_deadlines;
    scheduling_quality_t quality;
    double efficiency_score;          // 0.0 to 1.0
} scheduling_metrics_t;

/**
 * Bottleneck information
 */
typedef struct {
    uint32_t resource_id;
    char description[256];
    double severity;                  // 0.0 to 1.0
    uint64_t delay_caused_ns;
    char recommendation[256];
} scheduling_bottleneck_t;

/**
 * Deadlock detection result
 */
typedef struct {
    bool deadlock_detected;
    uint32_t* involved_operations;    // Operations in deadlock
    size_t num_operations;
    char description[256];
    char resolution[256];
} scheduling_deadlock_t;

/**
 * Optimization suggestion
 */
typedef struct {
    char description[256];
    double improvement_estimate;      // Estimated improvement %
    uint32_t* affected_operations;
    size_t num_affected;
    scheduling_strategy_t suggested_strategy;
} scheduling_suggestion_t;

/**
 * Complete schedule
 */
typedef struct {
    scheduling_entry_t* entries;
    size_t num_entries;
    scheduling_metrics_t metrics;
    scheduling_critical_path_t critical_path;
    scheduling_strategy_t strategy_used;
    uint64_t computation_time_ns;
} scheduling_schedule_t;

/**
 * Analyzer configuration
 */
typedef struct {
    scheduling_strategy_t default_strategy;
    bool detect_deadlocks;
    bool compute_critical_path;
    bool track_resource_utilization;
    bool generate_suggestions;
    uint64_t scheduling_timeout_ns;
    size_t max_iterations;            // For iterative algorithms
    double convergence_threshold;
} scheduling_analyzer_config_t;

/**
 * Opaque analyzer handle
 */
typedef struct scheduling_analyzer scheduling_analyzer_t;

// ============================================================================
// Initialization and Configuration
// ============================================================================

/**
 * Create scheduling analyzer with default configuration
 */
scheduling_analyzer_t* scheduling_analyzer_create(void);

/**
 * Create with custom configuration
 */
scheduling_analyzer_t* scheduling_analyzer_create_with_config(
    const scheduling_analyzer_config_t* config);

/**
 * Get default configuration
 */
scheduling_analyzer_config_t scheduling_analyzer_default_config(void);

/**
 * Destroy scheduling analyzer
 */
void scheduling_analyzer_destroy(scheduling_analyzer_t* analyzer);

/**
 * Reset analyzer state
 */
bool scheduling_analyzer_reset(scheduling_analyzer_t* analyzer);

// ============================================================================
// Operation Management
// ============================================================================

/**
 * Add operation to scheduler
 */
uint32_t scheduling_add_operation(
    scheduling_analyzer_t* analyzer,
    const char* name,
    uint64_t estimated_duration_ns,
    uint32_t priority);

/**
 * Set operation deadline
 */
bool scheduling_set_deadline(
    scheduling_analyzer_t* analyzer,
    uint32_t op_id,
    uint64_t deadline_ns);

/**
 * Add resource requirement to operation
 */
bool scheduling_add_resource_requirement(
    scheduling_analyzer_t* analyzer,
    uint32_t op_id,
    uint32_t resource_id,
    double amount);

/**
 * Update operation state
 */
bool scheduling_update_state(
    scheduling_analyzer_t* analyzer,
    uint32_t op_id,
    scheduling_op_state_t state);

/**
 * Get operation info
 */
bool scheduling_get_operation(
    scheduling_analyzer_t* analyzer,
    uint32_t op_id,
    scheduling_operation_t* op);

/**
 * Remove operation
 */
bool scheduling_remove_operation(
    scheduling_analyzer_t* analyzer,
    uint32_t op_id);

// ============================================================================
// Dependency Management
// ============================================================================

/**
 * Add dependency between operations
 */
bool scheduling_add_dependency(
    scheduling_analyzer_t* analyzer,
    uint32_t from_op_id,
    uint32_t to_op_id,
    scheduling_dependency_type_t type);

/**
 * Add dependency with lag
 */
bool scheduling_add_dependency_with_lag(
    scheduling_analyzer_t* analyzer,
    uint32_t from_op_id,
    uint32_t to_op_id,
    scheduling_dependency_type_t type,
    uint64_t lag_ns);

/**
 * Remove dependency
 */
bool scheduling_remove_dependency(
    scheduling_analyzer_t* analyzer,
    uint32_t from_op_id,
    uint32_t to_op_id);

/**
 * Get dependencies for operation
 */
bool scheduling_get_dependencies(
    scheduling_analyzer_t* analyzer,
    uint32_t op_id,
    scheduling_dependency_t** deps,
    size_t* count);

// ============================================================================
// Resource Management
// ============================================================================

/**
 * Register resource
 */
uint32_t scheduling_register_resource(
    scheduling_analyzer_t* analyzer,
    const char* name,
    scheduling_resource_type_t type,
    double capacity);

/**
 * Update resource capacity
 */
bool scheduling_update_resource_capacity(
    scheduling_analyzer_t* analyzer,
    uint32_t resource_id,
    double capacity);

/**
 * Get resource info
 */
bool scheduling_get_resource(
    scheduling_analyzer_t* analyzer,
    uint32_t resource_id,
    scheduling_resource_t* resource);

// ============================================================================
// Schedule Generation
// ============================================================================

/**
 * Generate schedule using default strategy
 */
scheduling_schedule_t* scheduling_generate(
    scheduling_analyzer_t* analyzer);

/**
 * Generate schedule using specific strategy
 */
scheduling_schedule_t* scheduling_generate_with_strategy(
    scheduling_analyzer_t* analyzer,
    scheduling_strategy_t strategy);

/**
 * Get ready operations (can be executed now)
 */
bool scheduling_get_ready_operations(
    scheduling_analyzer_t* analyzer,
    uint32_t** op_ids,
    size_t* count);

/**
 * Get next operation to execute
 */
bool scheduling_get_next_operation(
    scheduling_analyzer_t* analyzer,
    uint32_t* op_id);

// ============================================================================
// Analysis
// ============================================================================

/**
 * Compute critical path
 */
bool scheduling_compute_critical_path(
    scheduling_analyzer_t* analyzer,
    scheduling_critical_path_t* path);

/**
 * Get schedule metrics
 */
bool scheduling_get_metrics(
    scheduling_analyzer_t* analyzer,
    const scheduling_schedule_t* schedule,
    scheduling_metrics_t* metrics);

/**
 * Detect bottlenecks
 */
bool scheduling_detect_bottlenecks(
    scheduling_analyzer_t* analyzer,
    const scheduling_schedule_t* schedule,
    scheduling_bottleneck_t** bottlenecks,
    size_t* count);

/**
 * Detect deadlocks
 */
bool scheduling_detect_deadlocks(
    scheduling_analyzer_t* analyzer,
    scheduling_deadlock_t* result);

/**
 * Get optimization suggestions
 */
bool scheduling_get_suggestions(
    scheduling_analyzer_t* analyzer,
    const scheduling_schedule_t* schedule,
    scheduling_suggestion_t** suggestions,
    size_t* count);

/**
 * Compare two schedules
 */
double scheduling_compare_schedules(
    const scheduling_schedule_t* schedule1,
    const scheduling_schedule_t* schedule2);

// ============================================================================
// Validation
// ============================================================================

/**
 * Validate schedule feasibility
 */
bool scheduling_validate_schedule(
    scheduling_analyzer_t* analyzer,
    const scheduling_schedule_t* schedule,
    char** validation_errors,
    size_t* error_count);

/**
 * Check for circular dependencies
 */
bool scheduling_check_circular_deps(
    scheduling_analyzer_t* analyzer,
    uint32_t** cycle_ops,
    size_t* cycle_length);

// ============================================================================
// Reporting
// ============================================================================

/**
 * Generate schedule report
 */
char* scheduling_generate_report(
    scheduling_analyzer_t* analyzer,
    const scheduling_schedule_t* schedule);

/**
 * Export schedule to JSON
 */
char* scheduling_export_json(
    scheduling_analyzer_t* analyzer,
    const scheduling_schedule_t* schedule);

/**
 * Export to Gantt chart data
 */
char* scheduling_export_gantt(
    scheduling_analyzer_t* analyzer,
    const scheduling_schedule_t* schedule);

/**
 * Export to file
 */
bool scheduling_export_to_file(
    scheduling_analyzer_t* analyzer,
    const scheduling_schedule_t* schedule,
    const char* filename);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Get strategy name
 */
const char* scheduling_strategy_name(scheduling_strategy_t strategy);

/**
 * Get operation state name
 */
const char* scheduling_op_state_name(scheduling_op_state_t state);

/**
 * Get resource type name
 */
const char* scheduling_resource_type_name(scheduling_resource_type_t type);

/**
 * Get dependency type name
 */
const char* scheduling_dependency_type_name(scheduling_dependency_type_t type);

/**
 * Get quality level name
 */
const char* scheduling_quality_name(scheduling_quality_t quality);

/**
 * Free schedule resources
 */
void scheduling_free_schedule(scheduling_schedule_t* schedule);

/**
 * Free bottleneck array
 */
void scheduling_free_bottlenecks(scheduling_bottleneck_t* bottlenecks, size_t count);

/**
 * Free suggestions array
 */
void scheduling_free_suggestions(scheduling_suggestion_t* suggestions, size_t count);

/**
 * Free dependencies array
 */
void scheduling_free_dependencies(scheduling_dependency_t* deps, size_t count);

/**
 * Free critical path resources
 */
void scheduling_free_critical_path(scheduling_critical_path_t* path);

/**
 * Get last error message
 */
const char* scheduling_get_last_error(scheduling_analyzer_t* analyzer);

#ifdef __cplusplus
}
#endif

#endif // SCHEDULING_ANALYZER_H
