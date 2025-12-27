#ifndef QUANTUM_HARDWARE_SCHEDULER_H
#define QUANTUM_HARDWARE_SCHEDULER_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include "quantum_geometric/hardware/quantum_hardware_abstraction.h"
#include "quantum_geometric/hybrid/performance_monitoring.h"

#ifdef __cplusplus
extern "C" {
#endif

// System status for hardware scheduler
typedef struct SystemStatus {
    bool ibm_available;
    bool rigetti_available;
    bool dwave_available;
    double ibm_utilization;
    double rigetti_utilization;
    double dwave_utilization;
    size_t ibm_max_qubits;
    double ibm_gate_error_rate;
    double ibm_readout_error_rate;
    uint32_t ibm_optimization_level;
    double bandwidth;
    double latency;
    size_t buffer_usage;
    size_t available_memory;
    int available_threads;
    double cpu_utilization;
} SystemStatus;
typedef struct OperationQueue OperationQueue;
typedef struct ExecutionLog ExecutionLog;
typedef struct ResourceManager ResourceManager;
typedef struct NetworkManager NetworkManager;

// Hardware scheduler configuration
typedef struct {
    size_t max_memory;
    int max_threads;
    double bandwidth;
    double latency;
    bool enable_monitoring;
} HardwareSchedulerConfig;

// Execution schedule (output of scheduling)
typedef struct {
    double start_time;
    double end_time;
    double sync_points[5];  // MAX_SYNC_POINTS

    // Backend-specific resource allocations
    size_t ibm_qubits;        // IBM Quantum qubits allocated
    size_t rigetti_qubits;    // Rigetti qubits allocated
    size_t dwave_vars;        // DWave variables allocated

    // IBM-specific optimization data
    double ibm_error_budget;          // Accumulated error allowance
    size_t ibm_circuit_depth;         // Circuit depth for optimization
    uint32_t ibm_optimization_level;  // Transpilation level (0-3)
    double ibm_estimated_fidelity;    // Expected measurement fidelity
    bool ibm_feedback_enabled;        // Fast feedback active

    // Shared resource requirements
    size_t workspace_size;
    size_t data_transfer;
    double bandwidth_needed;
} ExecutionSchedule;

// Hardware scheduler
typedef struct HardwareScheduler {
    SystemStatus* status;
    PerformanceMonitor* monitor;
    ResourceManager* resource_mgr;
    NetworkManager* network_mgr;
    OperationQueue* operation_queue;
    ExecutionLog* execution_log;
} HardwareScheduler;

// Initialize hardware scheduler
HardwareScheduler* init_hardware_scheduler(const HardwareSchedulerConfig* config);

// Schedule quantum operation
int schedule_quantum_operation(HardwareScheduler* scheduler,
                               const QuantumOperation* op,
                               ExecutionSchedule* schedule);

// Cleanup hardware scheduler
void cleanup_hardware_scheduler(HardwareScheduler* scheduler);

// Helper function declarations (internal use)
SystemStatus* init_system_status(void);
SystemStatus* get_system_status(HardwareScheduler* scheduler);
void update_system_status(HardwareScheduler* scheduler, const ExecutionSchedule* schedule);

ResourceManager* init_resource_manager(size_t max_memory, int max_threads);
NetworkManager* init_network_manager(double bandwidth, double latency);
OperationQueue* init_operation_queue(size_t max_ops);
ExecutionLog* init_execution_log(void);

struct OperationRequirements* analyze_requirements(const QuantumOperation* op);
bool check_resources(const SystemStatus* status,
                     const struct OperationRequirements* requirements);

double get_next_start_time(HardwareScheduler* scheduler);
double calculate_data_transfer(const QuantumOperation* op);
double calculate_bandwidth_needed(size_t data_transfer, double duration);
double calculate_sync_time(const ExecutionSchedule* schedule, const QuantumOperation* op);
double calculate_end_time(const ExecutionSchedule* schedule);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_HARDWARE_SCHEDULER_H
