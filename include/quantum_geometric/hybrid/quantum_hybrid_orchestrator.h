#ifndef QUANTUM_HYBRID_ORCHESTRATOR_H
#define QUANTUM_HYBRID_ORCHESTRATOR_H

/**
 * @file quantum_hybrid_orchestrator.h
 * @brief High-level orchestration for hybrid quantum systems
 *
 * This module provides a unified interface for coordinating operations across
 * multiple quantum backends (Rigetti, DWave) with classical processing.
 * It USES the existing backend implementations - does not redefine them.
 */

#include <stddef.h>
#include <stdbool.h>
#include "quantum_geometric/hardware/quantum_hardware_abstraction.h"
#include "quantum_geometric/hardware/quantum_rigetti_backend.h"
#include "quantum_geometric/hardware/quantum_dwave_backend.h"
#include "quantum_geometric/hybrid/performance_monitoring.h"
#include "quantum_geometric/hybrid/quantum_hybrid_optimizer.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Forward declarations for orchestrator-specific opaque types
// ============================================================================

typedef struct HybridPerformanceStats HybridPerformanceStats;
typedef struct HybridErrorTracker HybridErrorTracker;
typedef struct HybridWorkloadBalancer HybridWorkloadBalancer;
typedef struct HybridCircuitOptimizer HybridCircuitOptimizer;
typedef struct HybridAnnealingOptimizer HybridAnnealingOptimizer;
typedef struct HybridQUBO HybridQUBO;

// ============================================================================
// Result types for hybrid backend operations
// ============================================================================

// Result from hybrid backend execution (Rigetti+DWave coordination)
// Named HybridBackendResult to distinguish from HybridBackendResult in quantum_classical_orchestrator.h
typedef struct {
    double* measurements;
    size_t num_measurements;
    double* probabilities;
    size_t num_probabilities;
    double fidelity;
    double execution_time;
    bool success;
    char* error_message;
} HybridBackendResult;

// Execution strategy for hybrid operations
typedef struct {
    bool use_rigetti;
    bool use_dwave;
    bool use_hybrid;
    bool parallel_execution;
} HybridExecutionStrategy;

// Profile of a quantum operation for planning
typedef struct {
    size_t circuit_depth;
    size_t num_gates;
    int* gate_types;
    size_t num_gate_types;
    size_t optimization_size;
    size_t constraint_count;
    int objective_type;
    bool requires_both;
    double estimated_runtime;
} HybridOperationProfile;

// ============================================================================
// Hybrid quantum system - wraps existing backends
// ============================================================================

typedef struct HybridQuantumSystem {
    // Backend configurations (uses existing types from backend headers)
    struct RigettiConfig* rigetti;
    struct DWaveConfig* dwave;

    // System parameters
    size_t num_qubits;
    size_t num_vars;
    double target_precision;

    // Orchestrator-specific components
    HybridPerformanceStats* stats;
    HybridErrorTracker* error_tracker;
    HybridWorkloadBalancer* workload_mgr;
    HybridCircuitOptimizer* circuit_opt;
    HybridAnnealingOptimizer* anneal_opt;

    // State
    bool initialized;
} HybridQuantumSystem;

// ============================================================================
// System lifecycle
// ============================================================================

// Create hybrid system from backend configs (preferred method)
HybridQuantumSystem* hybrid_system_init(
    const struct RigettiBackendConfig* rigetti_config,
    const DWaveBackendConfig* dwave_config);

// Create hybrid system from URLs (convenience wrapper)
HybridQuantumSystem* hybrid_system_init_from_urls(
    const char* rigetti_url,
    const char* dwave_url,
    size_t num_qubits,
    size_t num_vars);

// Cleanup hybrid system
void hybrid_system_cleanup(HybridQuantumSystem* sys);

// ============================================================================
// Operation execution
// ============================================================================

// Analyze operation to determine execution strategy
HybridOperationProfile* hybrid_analyze_operation(const QuantumOperation* op);
void hybrid_profile_cleanup(HybridOperationProfile* profile);

// Execute operation with automatic backend selection
int hybrid_execute(HybridQuantumSystem* sys,
                   const QuantumOperation* op,
                   HybridBackendResult* result);

// Execute with explicit strategy
int hybrid_execute_with_strategy(HybridQuantumSystem* sys,
                                 const QuantumOperation* op,
                                 const HybridExecutionStrategy* strategy,
                                 HybridBackendResult* result);

// ============================================================================
// Backend-specific execution (wraps existing functions)
// ============================================================================

// Execute circuit on Rigetti backend
int hybrid_rigetti_execute(HybridQuantumSystem* sys,
                           struct QuantumCircuit* circuit,
                           HybridBackendResult* result);

// Execute annealing on DWave backend
int hybrid_dwave_execute(HybridQuantumSystem* sys,
                         DWaveProblem* problem,
                         HybridBackendResult* result);

// ============================================================================
// Orchestrator-specific optimization
// ============================================================================

HybridCircuitOptimizer* hybrid_circuit_optimizer_init(size_t num_qubits);
void hybrid_circuit_optimizer_cleanup(HybridCircuitOptimizer* opt);
void hybrid_circuit_optimize(HybridCircuitOptimizer* opt, struct QuantumCircuit* circuit);

HybridAnnealingOptimizer* hybrid_annealing_optimizer_init(size_t num_vars);
void hybrid_annealing_optimizer_cleanup(HybridAnnealingOptimizer* opt);
void hybrid_annealing_optimize(HybridAnnealingOptimizer* opt, DWaveProblem* problem);

// ============================================================================
// Resource management
// ============================================================================

HybridWorkloadBalancer* hybrid_workload_balancer_init(size_t batch_size);
void hybrid_workload_balancer_cleanup(HybridWorkloadBalancer* balancer);

HybridPerformanceStats* hybrid_stats_init(void);
void hybrid_stats_update(HybridPerformanceStats* stats, const HybridOperationProfile* profile, int status);
void hybrid_stats_cleanup(HybridPerformanceStats* stats);

HybridErrorTracker* hybrid_error_tracker_init(void);
void hybrid_error_tracker_cleanup(HybridErrorTracker* tracker);
double hybrid_error_get_rate(const HybridErrorTracker* tracker);

// ============================================================================
// Conversion utilities
// ============================================================================

// Convert quantum operation to circuit for Rigetti
struct QuantumCircuit* hybrid_op_to_circuit(const QuantumOperation* op);

// Convert quantum operation to QUBO for DWave
HybridQUBO* hybrid_op_to_qubo(const QuantumOperation* op);
void hybrid_qubo_cleanup(HybridQUBO* qubo);

// ============================================================================
// Result processing
// ============================================================================

void hybrid_result_from_rigetti(HybridBackendResult* result, const struct RigettiConfig* backend);
void hybrid_result_from_dwave(HybridBackendResult* result, const struct DWaveConfig* backend);
void hybrid_result_combine(HybridBackendResult* result,
                           const HybridBackendResult* rigetti_result,
                           const HybridBackendResult* dwave_result,
                           HybridErrorTracker* tracker);
void hybrid_result_cleanup(HybridBackendResult* result);

// ============================================================================
// Analysis utilities (prefixed to avoid conflicts)
// ============================================================================

size_t hybrid_compute_depth(const QuantumOperation* op);
size_t hybrid_count_gates(const QuantumOperation* op);
size_t hybrid_compute_problem_size(const QuantumOperation* op);
bool hybrid_requires_both_backends(const QuantumOperation* op);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_HYBRID_ORCHESTRATOR_H
