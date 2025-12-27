#ifndef QUANTUM_CLASSICAL_ORCHESTRATOR_H
#define QUANTUM_CLASSICAL_ORCHESTRATOR_H

#include <stddef.h>
#include <stdbool.h>
#include "quantum_geometric/core/quantum_types.h"
#include "quantum_geometric/hardware/quantum_hardware_types.h"
#include "quantum_geometric/hardware/quantum_hardware_abstraction.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct QuantumHardware;
typedef struct QuantumHardware QuantumHardware;
struct OptimizationContext;
typedef struct OptimizationContext OptimizationContext;

// Algorithm types for hybrid execution
typedef enum {
    ALGORITHM_VQE,
    ALGORITHM_QAOA,
    ALGORITHM_QML,
    ALGORITHM_QUANTUM_CHEMISTRY,
    ALGORITHM_OPTIMIZATION,
    ALGORITHM_SIMULATION,
    ALGORITHM_GENERIC
} AlgorithmType;

// Classical resource configuration
typedef struct {
    size_t num_cores;
    size_t memory_size;
    bool has_gpu;
    size_t gpu_memory;
    double energy_budget;
    size_t num_threads;
} ClassicalResources;

// Classical computation configuration
typedef struct {
    size_t num_threads;
    bool use_gpu;
    size_t memory_limit;
} ClassicalConfig;

// Workload scheduler
typedef struct WorkloadScheduler {
    size_t max_quantum_jobs;
    size_t max_classical_jobs;
    size_t pending_jobs;
    void* scheduler_data;
} WorkloadScheduler;

// Quantum task definition
typedef struct {
    QuantumCircuit* circuit;
    AlgorithmType algorithm_type;
    size_t num_shots;
    double* parameters;
    size_t num_parameters;
    void* task_data;
} QuantumTask;

// Classical task definition
typedef struct {
    void* input_data;
    size_t input_size;
    AlgorithmType algorithm_type;
    void* task_data;
} ClassicalTask;

// Quantum execution result
typedef struct {
    double* expectation_values;
    size_t num_values;
    double* probabilities;
    size_t num_probabilities;
    double fidelity;
    double execution_time;
} QuantumResult;

// Classical execution result
typedef struct {
    double* output_data;
    size_t output_size;
    double execution_time;
    double error_metric;
} ClassicalResult;

// Hybrid execution result
typedef struct {
    QuantumResult quantum_result;
    ClassicalResult classical_result;
    AlgorithmType algorithm_type;
    double total_energy;
    double total_time;
    double combined_fidelity;
} HybridResult;

// Hybrid orchestrator (opaque, defined in .c)
typedef struct HybridOrchestrator HybridOrchestrator;

// Note: OptimizationObjective is defined in classical_optimization_engine.h

// Core orchestrator functions
HybridOrchestrator* init_hybrid_orchestrator(QuantumHardware* quantum_hw,
                                             const ClassicalResources* classical);
void cleanup_hybrid_orchestrator(HybridOrchestrator* orchestrator);

// Task execution
int execute_hybrid_task(HybridOrchestrator* orchestrator,
                       const QuantumTask* task,
                       HybridResult* result);

// Workload scheduler functions
WorkloadScheduler* init_workload_scheduler(void);
void cleanup_workload_scheduler(WorkloadScheduler* scheduler);

// Circuit optimization for hardware (hw_circuit_ prefix for hardware circuit operations)
QuantumCircuit* hw_circuit_optimize_for_hardware(const QuantumCircuit* circuit,
                                                  const HardwareCapabilities* capabilities);
void hw_circuit_cleanup(QuantumCircuit* circuit);

// Hardware submission
int submit_quantum_circuit(QuantumHardware* hardware,
                          const QuantumCircuit* circuit,
                          QuantumResult* result);

// Classical computation
int classical_computation(const QuantumTask* task,
                         const ClassicalConfig* config,
                         ClassicalResult* result);

// Resource calculations
double calculate_quantum_utilization(const QuantumHardware* hardware);
double calculate_classical_utilization(const ClassicalResources* resources);
double calculate_communication_overhead(const QuantumTask* task);
double calculate_energy_consumption(const HybridOrchestrator* orchestrator);

// Resource adjustment
void increase_classical_allocation(HybridOrchestrator* orchestrator);
void increase_quantum_allocation(HybridOrchestrator* orchestrator);
void optimize_communication_patterns(HybridOrchestrator* orchestrator);
void optimize_energy_usage(HybridOrchestrator* orchestrator);

// Task splitting for hybrid execution
void split_vqe_task(const QuantumTask* task,
                   QuantumTask* quantum_part,
                   ClassicalTask* classical_part);
void split_qaoa_task(const QuantumTask* task,
                    QuantumTask* quantum_part,
                    ClassicalTask* classical_part);
void split_qml_task(const QuantumTask* task,
                   QuantumTask* quantum_part,
                   ClassicalTask* classical_part);
void split_generic_task(const QuantumTask* task,
                       QuantumTask* quantum_part,
                       ClassicalTask* classical_part);

// Result combination
void combine_vqe_results(const QuantumResult* quantum_result,
                        const ClassicalResult* classical_result,
                        HybridResult* final_result);
void combine_qaoa_results(const QuantumResult* quantum_result,
                         const ClassicalResult* classical_result,
                         HybridResult* final_result);
void combine_qml_results(const QuantumResult* quantum_result,
                        const ClassicalResult* classical_result,
                        HybridResult* final_result);
void combine_generic_results(const QuantumResult* quantum_result,
                            const ClassicalResult* classical_result,
                            HybridResult* final_result);

// Optimization parameter tuning (uses OptimizationContext from classical_optimization_engine.h)
int orchestrator_optimize_parameters(OptimizationContext* optimizer,
                                     double (*objective_fn)(const double*, double*, void*),
                                     void* objective_data,
                                     HybridOrchestrator* orchestrator);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_CLASSICAL_ORCHESTRATOR_H
