#ifndef QUANTUM_STATE_OPERATIONS_H
#define QUANTUM_STATE_OPERATIONS_H

#include "quantum_geometric/core/quantum_types.h"
#include "quantum_geometric/core/quantum_complex.h"
#include "quantum_geometric/core/error_codes.h"
#include "quantum_geometric/core/quantum_state.h"
#include "quantum_geometric/core/quantum_geometric_constants.h"
#include "quantum_geometric/core/memory_pool.h"
#include "quantum_geometric/core/hierarchical_matrix.h"
#include "quantum_geometric/physics/stabilizer_types.h"
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

// Helper functions for quantum state operations (local to this module)
void apply_hadamard_gate(const quantum_state_t* state, size_t x, size_t y);
void apply_rotation_x(const quantum_state_t* state, size_t x, size_t y, double angle);
void quantum_wait(const quantum_state_t* state, double duration);
void apply_composite_x_pulse(const quantum_state_t* state, size_t x, size_t y);
double get_readout_error_rate(size_t x, size_t y);
double get_gate_error_rate(size_t x, size_t y);

// =============================================================================
// Stabilizer measurement API - see stabilizer_measurement.h for full API
// =============================================================================
// Note: The following functions are declared in stabilizer_measurement.h:
// - init_stabilizer_measurement, cleanup_stabilizer_measurement
// - measure_stabilizers, get_stabilizer_measurements
// - measure_pauli_z_with_confidence, measure_pauli_x_with_confidence
// - apply_x_error_mitigation_sequence, apply_x_measurement_correction
// - get_x_stabilizer_correlation, get_error_rate
// Include stabilizer_measurement.h for the full stabilizer measurement API.

// Hierarchical matrix operations
void update_hmatrix_quantum_state(HierarchicalMatrix* mat);
void cleanup_hmatrix_quantum_state(HierarchicalMatrix* mat);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_STATE_OPERATIONS_H
