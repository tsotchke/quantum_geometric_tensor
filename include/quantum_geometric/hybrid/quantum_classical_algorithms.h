#ifndef QUANTUM_CLASSICAL_ALGORITHMS_H
#define QUANTUM_CLASSICAL_ALGORITHMS_H

#include <stdbool.h>
#include "quantum_geometric/core/quantum_types.h"
#include "quantum_geometric/core/quantum_circuit_types.h"
#include "quantum_geometric/core/quantum_geometric_types.h"

// Forward declarations
// Forward declarations
typedef struct HybridOrchestrator HybridOrchestrator;
typedef struct OptimizationContext OptimizationContext;
typedef struct HamiltonianOperator HamiltonianOperator;
typedef struct quantum_gate_t quantum_gate_t;

// VQE context
typedef struct VQEContext {
    struct quantum_circuit* ansatz;
    HamiltonianOperator* hamiltonian;
    OptimizationContext* optimizer;
    double* parameters;
    size_t num_parameters;
    double current_energy;
} VQEContext;

// QAOA context
typedef struct QAOAContext {
    struct quantum_circuit* mixer;
    struct quantum_circuit* problem;
    OptimizationContext* optimizer;
    double* parameters;
    size_t num_parameters;
    size_t depth;
    double current_cost;
} QAOAContext;

// VQE result structure
typedef struct {
    double energy;
    double* parameters;
    size_t num_parameters;
} VQEResult;

// QAOA result structure
typedef struct {
    double cost;
    double* parameters;
    size_t num_parameters;
} QAOAResult;

// VQE initialization and execution
VQEContext* init_vqe(const struct quantum_circuit* ansatz,
                    const HamiltonianOperator* hamiltonian);
int run_vqe(VQEContext* ctx,
            HybridOrchestrator* orchestrator,
            VQEResult* result);
void cleanup_vqe(VQEContext* ctx);

// QAOA initialization and execution
QAOAContext* init_qaoa(const struct quantum_circuit* problem,
                      size_t depth);
int run_qaoa(QAOAContext* ctx,
             HybridOrchestrator* orchestrator,
             QAOAResult* result);
void cleanup_qaoa(QAOAContext* ctx);

#endif // QUANTUM_CLASSICAL_ALGORITHMS_H
