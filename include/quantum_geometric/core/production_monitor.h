/**
 * @file production_monitor.h
 * @brief Production monitoring system for quantum geometric operations
 *
 * Provides real-time monitoring, alerting, and logging for production
 * deployments of quantum geometric computations on HPC systems.
 */

#ifndef PRODUCTION_MONITOR_H
#define PRODUCTION_MONITOR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Maximum number of alert handlers
#define MAX_ALERT_HANDLERS 16

// Alert severity levels
typedef enum {
    ALERT_LEVEL_DEBUG = 0,
    ALERT_LEVEL_INFO = 1,
    ALERT_LEVEL_WARNING = 2,
    ALERT_LEVEL_ERROR = 3,
    ALERT_LEVEL_CRITICAL = 4,
    ALERT_LEVEL_FATAL = 5
} alert_level_t;

// Alert callback function type (single argument version for compatibility)
typedef void (*alert_callback)(alert_level_t level, const char* message);

// Operation types
typedef enum {
    OP_TYPE_GATE = 0,
    OP_TYPE_MEASUREMENT,
    OP_TYPE_STATE_PREP,
    OP_TYPE_CIRCUIT,
    OP_TYPE_VQE,
    OP_TYPE_QAOA,
    OP_TYPE_OTHER
} operation_type_t;

// Quantum operation type
typedef struct {
    const char* name;
    uint32_t operation_id;
    uint64_t start_time;
    size_t num_qubits;
    size_t circuit_depth;
    operation_type_t type;
    void* operation_data;
} quantum_operation_t;

// Quantum result type
typedef struct {
    bool success;
    double fidelity;
    double execution_time;
    size_t shots;
    double* probabilities;
    size_t num_outcomes;
    char* error_message;
    int error_code;
    bool false_positive;
    void* result_data;
} quantum_result_t;

// Threshold configuration
typedef struct {
    double error_rate;
    double latency;
    double memory_usage;
    double cpu_usage;
    double success_rate;
} threshold_config_t;

// Production configuration
typedef struct {
    const char* log_directory;
    const char* metrics_endpoint;
    alert_level_t min_log_level;
    threshold_config_t thresholds;
    bool enable_alerting;
    bool enable_metrics_export;
    size_t metrics_export_interval_ms;
    void* config_data;
} production_config_t;

// Initialize production monitoring
bool init_production_monitoring(const production_config_t* config);

// Shutdown production monitoring
void cleanup_production_monitoring(void);
void shutdown_production_monitoring(void);

// Register an alert handler (simple version)
bool register_alert_handler(alert_callback handler);

// Unregister an alert handler
bool unregister_alert_handler(alert_callback callback);

// Start monitoring a quantum operation
void begin_quantum_operation(const quantum_operation_t* operation);

// End monitoring a quantum operation
void end_quantum_operation(const quantum_operation_t* operation, const quantum_result_t* result);

// Log production events
void log_production_event(alert_level_t level, const char* component,
                         const char* event, const char* details);

// Get current production metrics
bool get_production_metrics(double* error_rate, double* avg_latency,
                           double* memory_usage, double* cpu_usage);

// Set threshold values
void set_error_threshold(double threshold);
void set_latency_threshold(double threshold_ms);
void set_memory_threshold(double threshold_percent);
void set_cpu_threshold(double threshold_percent);

// Health check
bool production_health_check(void);

// Alert level to string
const char* alert_level_str(alert_level_t level);

// Performance monitoring integration
bool init_performance_monitoring(void);
void cleanup_performance_monitoring(void);
void start_operation_timing(const char* operation_name);
void end_operation_timing(const char* operation_name);
void update_resource_usage(void);
void record_operation_result(bool success, double latency);

#ifdef __cplusplus
}
#endif

#endif // PRODUCTION_MONITOR_H
