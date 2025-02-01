#ifndef QUANTUM_STATE_OPERATIONS_H
#define QUANTUM_STATE_OPERATIONS_H

#include "quantum_geometric/core/quantum_types.h"
#include "quantum_geometric/core/quantum_complex.h"
#include "quantum_geometric/core/error_codes.h"
#include "quantum_geometric/core/quantum_state.h"
#include "quantum_geometric/core/quantum_geometric_constants.h"
#include "quantum_geometric/core/memory_pool.h"
#include <stdbool.h>
#include <complex.h>
#include <pthread.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// X-stabilizer state structure
typedef struct {
    double* correlations;
    double* confidences;
    size_t history_size;
    double error_rate;
} XStabilizerState;

// Configuration structures
typedef struct {
    bool enable_x_optimization;
    size_t repetition_count;
    double error_threshold;
    double confidence_threshold;
    bool use_dynamic_decoupling;
    bool track_correlations;
} XStabilizerConfig;

typedef struct {
    size_t num_qubits;
    double decoherence_rate;
    bool track_phase;
    bool enable_error_correction;
    XStabilizerConfig x_stabilizer_config;
} StateConfig;

// Operation results
typedef struct {
    size_t measurement_count;
    double average_fidelity;
} XStabilizerResults;

typedef struct {
    XStabilizerResults x_stabilizer_results;
} StateOperationResult;

// State operations handle
typedef struct QuantumStateOps QuantumStateOps;

// State operations initialization
QuantumStateOps* init_quantum_state_ops(const StateConfig* config);
void cleanup_quantum_state_ops(QuantumStateOps* ops);

// State operation execution
void perform_state_operation(QuantumStateOps* ops, quantum_state_t** states, size_t count);
StateOperationResult* get_state_result(QuantumStateOps* ops);
void cleanup_state_result(StateOperationResult* result);

// Error mitigation and measurement
void apply_x_error_mitigation_sequence(const quantum_state_t* state,
                                     size_t x,
                                     size_t y);

void apply_x_measurement_correction(const quantum_state_t* state,
                                  size_t x,
                                  size_t y,
                                  double* result);

double get_x_stabilizer_correlation(const quantum_state_t* state,
                                  size_t x,
                                  size_t y,
                                  size_t qubit_idx);

// Helper functions
void apply_hadamard_gate(const quantum_state_t* state, size_t x, size_t y);
void apply_rotation_x(const quantum_state_t* state, size_t x, size_t y, double angle);
void quantum_wait(const quantum_state_t* state, double duration);
void apply_composite_x_pulse(const quantum_state_t* state, size_t x, size_t y);
double get_readout_error_rate(size_t x, size_t y);
double get_gate_error_rate(size_t x, size_t y);

// Hierarchical matrix operations
void update_hmatrix_quantum_state(HierarchicalMatrix* mat);
void cleanup_hmatrix_quantum_state(HierarchicalMatrix* mat);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_STATE_OPERATIONS_H
