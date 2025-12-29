/**
 * @file qgt_context.h
 * @brief QGT context and distributed operations API
 *
 * High-level context-based API for quantum geometric operations.
 * Provides unified interface for state management, error correction,
 * and distributed computing.
 */

#ifndef QGT_CONTEXT_H
#define QGT_CONTEXT_H

#include <stddef.h>
#include <stdbool.h>
#include "quantum_geometric/core/quantum_state.h"
#include "quantum_geometric/core/error_codes.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Context Types
// ============================================================================

/**
 * @brief Main QGT context for quantum geometric operations
 */
typedef struct qgt_context qgt_context_t;

/**
 * @brief Distributed computing context for multi-node operations
 */
typedef struct qgt_distributed_context qgt_distributed_context_t;

// ============================================================================
// Context Management
// ============================================================================

/**
 * @brief Create a QGT context
 * @param ctx Output context pointer
 * @return QGT_SUCCESS on success
 */
qgt_error_t qgt_create_context(qgt_context_t** ctx);

/**
 * @brief Destroy a QGT context
 * @param ctx Context to destroy
 */
void qgt_destroy_context(qgt_context_t* ctx);

// ============================================================================
// Distributed Computing
// ============================================================================

/**
 * @brief Create distributed context
 * @param ctx Parent context
 * @param num_nodes Number of distributed nodes
 * @param dist_ctx Output distributed context
 * @return QGT_SUCCESS on success
 */
qgt_error_t qgt_create_distributed_context(qgt_context_t* ctx,
                                           size_t num_nodes,
                                           qgt_distributed_context_t** dist_ctx);

/**
 * @brief Destroy distributed context
 * @param dist_ctx Distributed context to destroy
 */
void qgt_destroy_distributed_context(qgt_distributed_context_t* dist_ctx);

/**
 * @brief Distribute state across nodes
 */
qgt_error_t qgt_distribute_state(qgt_distributed_context_t* dist_ctx, quantum_state_t* state);

/**
 * @brief Gather distributed state
 */
qgt_error_t qgt_gather_state(qgt_distributed_context_t* dist_ctx, quantum_state_t* state);

/**
 * @brief Simulate node failure for testing
 */
qgt_error_t qgt_simulate_node_failure(qgt_distributed_context_t* dist_ctx, size_t node_id);

/**
 * @brief Distributed geometric rotation
 */
qgt_error_t qgt_distributed_geometric_rotate(qgt_distributed_context_t* dist_ctx,
                                              quantum_state_t* state,
                                              double angle,
                                              const double* axis);

/**
 * @brief Distributed geometric parallel transport
 */
qgt_error_t qgt_distributed_geometric_parallel_transport(qgt_distributed_context_t* dist_ctx,
                                                          quantum_state_t* state,
                                                          const double* path,
                                                          size_t num_points);

// ============================================================================
// Error Correction
// ============================================================================

/**
 * @brief Apply error channel to state
 */
qgt_error_t qgt_apply_error_channel(qgt_context_t* ctx, quantum_state_t* state, double error_rate);

/**
 * @brief Apply error correction to state
 */
qgt_error_t qgt_apply_error_correction(qgt_context_t* ctx, quantum_state_t* state);

/**
 * @brief Apply error correction cycle
 */
qgt_error_t qgt_apply_error_correction_cycle(qgt_context_t* ctx, quantum_state_t* state);

// ============================================================================
// Fidelity Measurement
// ============================================================================

/**
 * @brief Measure fidelity between states
 */
qgt_error_t qgt_geometric_measure_fidelity(qgt_context_t* ctx,
                                            quantum_state_t* state1,
                                            quantum_state_t* state2,
                                            double* fidelity);

// ============================================================================
// Logical State Operations
// ============================================================================

/**
 * @brief Create logical state
 */
qgt_error_t qgt_create_logical_state(qgt_context_t* ctx, size_t num_qubits, quantum_state_t** state);

/**
 * @brief Measure logical fidelity
 */
qgt_error_t qgt_measure_logical_fidelity(qgt_context_t* ctx, quantum_state_t* state, double* fidelity);

// ============================================================================
// Noise Model
// ============================================================================

/**
 * @brief Enable noise model
 */
qgt_error_t qgt_enable_noise_model(qgt_context_t* ctx, double noise_strength);

// ============================================================================
// Geometric Operations
// ============================================================================

/**
 * @brief Apply geometric rotation
 */
qgt_error_t qgt_geometric_rotate(qgt_context_t* ctx, quantum_state_t* state,
                                  double angle, const double* axis);

/**
 * @brief Compute geometric metric tensor
 */
qgt_error_t qgt_geometric_compute_metric(qgt_context_t* ctx, quantum_state_t* state, double* metric);

#ifdef __cplusplus
}
#endif

#endif // QGT_CONTEXT_H
