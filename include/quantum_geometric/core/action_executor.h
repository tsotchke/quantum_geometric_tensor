#ifndef ACTION_EXECUTOR_H
#define ACTION_EXECUTOR_H

#include <stdbool.h>
#include <stddef.h>
#include "quantum_geometric/core/quantum_types.h"

// Action types
typedef enum {
    ACTION_QUANTUM_GATE,        // Quantum gate operation
    ACTION_MEASUREMENT,         // Quantum measurement
    ACTION_STATE_PREP,         // State preparation
    ACTION_ERROR_CORRECTION,    // Error correction
    ACTION_GEOMETRIC,           // Geometric operation
    ACTION_CALIBRATION         // Hardware calibration
} action_type_t;

// Execution modes
typedef enum {
    EXEC_MODE_SYNC,            // Synchronous execution
    EXEC_MODE_ASYNC,           // Asynchronous execution
    EXEC_MODE_BATCHED,         // Batched execution
    EXEC_MODE_PIPELINED       // Pipelined execution
} execution_mode_t;

// Hardware targets
typedef enum {
    TARGET_IBM,                // IBM quantum hardware
    TARGET_RIGETTI,           // Rigetti quantum hardware
    TARGET_DWAVE,             // D-Wave quantum hardware
    TARGET_SIMULATOR          // Quantum simulator
} hardware_target_t;

// Error protection levels
typedef enum {
    PROTECTION_NONE,           // No error protection
    PROTECTION_MINIMAL,        // Minimal error protection
    PROTECTION_STANDARD,       // Standard error protection
    PROTECTION_MAXIMAL        // Maximum error protection
} protection_level_t;

// Execution configuration
typedef struct {
    execution_mode_t mode;     // Execution mode
    hardware_target_t target;  // Hardware target
    protection_level_t protection; // Error protection level
    bool validate_results;     // Enable result validation
    bool collect_metrics;      // Enable metrics collection
    bool optimize_execution;   // Enable execution optimization
} executor_config_t;

// Action parameters
typedef struct {
    action_type_t type;        // Action type
    void* data;               // Action-specific data
    size_t data_size;         // Size of action data
    bool requires_calibration; // Whether calibration is needed
    bool is_geometric;        // Whether action is geometric
    bool is_error_protected;  // Whether action has error protection
} action_params_t;

// Execution results
typedef struct {
    bool success;              // Execution success flag
    double fidelity;          // Result fidelity
    double error_rate;        // Error rate
    double execution_time;    // Execution time
    size_t resource_usage;    // Resource usage
    char* error_message;      // Error message if any
} execution_result_t;

// Performance metrics
typedef struct {
    double gate_fidelity;     // Gate fidelity
    double state_fidelity;    // State fidelity
    double error_rate;        // Error rate
    double coherence_time;    // Coherence time
    size_t gate_count;        // Gate count
    size_t circuit_depth;     // Circuit depth
} performance_metrics_t;

// Opaque executor handle
typedef struct action_executor_t action_executor_t;

// Core functions
action_executor_t* create_action_executor(const executor_config_t* config);
void destroy_action_executor(action_executor_t* executor);

// Execution functions
execution_result_t execute_action(action_executor_t* executor,
                                const action_params_t* params);
execution_result_t execute_action_batch(action_executor_t* executor,
                                      const action_params_t* params,
                                      size_t num_actions);
bool cancel_execution(action_executor_t* executor);

// Asynchronous execution
bool schedule_action(action_executor_t* executor,
                    const action_params_t* params,
                    void (*callback)(execution_result_t));
bool wait_for_completion(action_executor_t* executor,
                        double timeout_seconds);

// Hardware control
bool calibrate_hardware(action_executor_t* executor);
bool validate_hardware_state(action_executor_t* executor);
bool reset_hardware_state(action_executor_t* executor);

// Error protection
bool enable_error_protection(action_executor_t* executor,
                           protection_level_t level);
bool validate_error_bounds(action_executor_t* executor,
                         const execution_result_t* result);
bool apply_error_correction(action_executor_t* executor,
                          execution_result_t* result);

// Performance monitoring
bool get_performance_metrics(const action_executor_t* executor,
                           performance_metrics_t* metrics);
bool reset_performance_metrics(action_executor_t* executor);

// Optimization functions
bool optimize_execution_plan(action_executor_t* executor,
                           const action_params_t* params);
bool suggest_optimization_strategy(const action_executor_t* executor,
                                 const action_params_t* params,
                                 executor_config_t* suggested_config);

#endif // ACTION_EXECUTOR_H
