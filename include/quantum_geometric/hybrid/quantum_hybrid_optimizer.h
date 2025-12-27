#ifndef QUANTUM_HYBRID_OPTIMIZER_H
#define QUANTUM_HYBRID_OPTIMIZER_H

#include <stddef.h>
#include <stdbool.h>
#include "quantum_geometric/hardware/quantum_hardware_abstraction.h"
#include "quantum_geometric/hybrid/quantum_hardware_scheduler.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations for component types
struct CircuitComponent;
typedef struct CircuitComponent CircuitComponent;

struct AnnealingComponent;
typedef struct AnnealingComponent AnnealingComponent;

// Extended hardware config for hybrid operations
typedef struct {
    size_t rigetti_qubits;
    size_t dwave_vars;
    double rigetti_fidelity;
    double dwave_efficiency;
    double max_runtime;
    void* backend_config;
} HybridHardwareConfig;

// Optimized operation result
typedef struct {
    CircuitComponent* rigetti_circuit;
    AnnealingComponent* dwave_problem;
    ExecutionSchedule* execution_schedule;
    double expected_fidelity;
    double expected_runtime;
    bool success;
} OptimizedOperation;

// Analysis functions
double analyze_rigetti_fidelity(const HardwareConfig* hw);
double analyze_dwave_efficiency(const HardwareConfig* hw);
size_t analyze_circuit_depth(const QuantumOperation* op);
size_t analyze_problem_size(const QuantumOperation* op);
double analyze_hybrid_overlap(const QuantumOperation* op);
size_t calculate_memory_requirements(const QuantumOperation* op);
size_t estimate_network_bandwidth(const QuantumOperation* op);
double estimate_runtime(const QuantumOperation* op, const HardwareConfig* hw);

// Allocation optimization
size_t optimize_rigetti_allocation(const void* profile);
size_t optimize_dwave_allocation(const void* profile);
size_t optimize_workspace_size(const void* profile);

// Target calculation
double calculate_target_fidelity(const void* profile);
double calculate_target_runtime(const void* profile);
double calculate_error_budget(const void* profile);

// Component extraction
CircuitComponent* extract_circuit_component(const QuantumOperation* op);
AnnealingComponent* extract_annealing_component(const QuantumOperation* op);

// Optimization functions
void optimize_gate_sequence(CircuitComponent* circuit, size_t num_qubits);
void optimize_qubit_mapping(CircuitComponent* circuit, size_t num_qubits);
void add_error_mitigation(CircuitComponent* circuit, double target_fidelity);

void optimize_variable_embedding(AnnealingComponent* problem, size_t num_vars);
void optimize_annealing_schedule(AnnealingComponent* problem, double target_runtime);
void add_error_correction(AnnealingComponent* problem, double error_budget);

ExecutionSchedule* create_execution_schedule(const void* plan);
void optimize_resource_allocation(ExecutionSchedule* schedule, const void* plan);
void optimize_data_transfer(ExecutionSchedule* schedule, const void* plan);
void add_synchronization_points(ExecutionSchedule* schedule, const void* plan);

// Validation functions (hybrid_ prefix to avoid conflict with backend validators)
bool hybrid_validate_circuit_component(const CircuitComponent* circuit, const void* plan);
bool hybrid_validate_annealing_component(const AnnealingComponent* problem, const void* plan);
bool hybrid_validate_execution_schedule(const ExecutionSchedule* schedule, const void* plan);

// Cleanup functions
void cleanup_circuit_component(CircuitComponent* circuit);
void cleanup_annealing_component(AnnealingComponent* problem);
void cleanup_execution_schedule(ExecutionSchedule* schedule);

// Main optimization function
int optimize_quantum_operation(const QuantumOperation* op,
                               const HardwareConfig* hw,
                               OptimizedOperation* result);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_HYBRID_OPTIMIZER_H
