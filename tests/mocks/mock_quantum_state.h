/**
 * @file mock_quantum_state.h
 * @brief Mock helper declarations for quantum state operations in tests
 *
 * This header provides test helper functions and type aliases for creating
 * and managing quantum states in tests. The types are aliased to the actual
 * library types for compatibility.
 */

#ifndef MOCK_QUANTUM_STATE_H
#define MOCK_QUANTUM_STATE_H

#include <stddef.h>
#include <stdbool.h>

// Include actual library types
#include "quantum_geometric/core/quantum_types.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_geometric_error.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Test Constants
// ============================================================================

#ifndef TEST_NUM_QUBITS
#define TEST_NUM_QUBITS 4
#endif

#ifndef TEST_DIMENSION
#define TEST_DIMENSION 16  // 2^TEST_NUM_QUBITS
#endif

#ifndef QGT_TEST_CIRCLE_PATH_POINTS
#define QGT_TEST_CIRCLE_PATH_POINTS 100
#endif

// ============================================================================
// Type Aliases for Test Compatibility
// ============================================================================

// Context type alias - tests expect qgt_context_t
// We'll use the quantum_geometric_state_t as the context for now
#ifndef QGT_CONTEXT_TYPE_DEFINED
#define QGT_CONTEXT_TYPE_DEFINED
typedef struct qgt_context {
    void* internal_state;
    size_t max_qubits;
    bool initialized;
} qgt_context_t;
#endif

// State type alias - tests expect qgt_state_t
#ifndef QGT_STATE_TYPE_DEFINED
#define QGT_STATE_TYPE_DEFINED
typedef quantum_state_t qgt_state_t;
#endif

// Error type alias
typedef qgt_error_t qgt_error_t;

// Distributed context type alias
typedef struct qgt_distributed_context {
    size_t num_nodes;
    size_t rank;
    bool initialized;
    void* internal_state;
} qgt_distributed_context_t;

// ============================================================================
// Entangled State Types
// ============================================================================

typedef enum {
    ENTANGLED_STATE_BELL = 0,     // Bell state (2 qubits)
    ENTANGLED_STATE_GHZ = 1,      // GHZ state (n qubits)
    ENTANGLED_STATE_W = 2,        // W state (n qubits)
    ENTANGLED_STATE_CLUSTER = 3   // Cluster state (n qubits)
} entangled_state_type_t;

// ============================================================================
// Mock Function Declarations
// ============================================================================

/**
 * Create a quantum state for testing
 * @param ctx Quantum geometric context
 * @param num_qubits Number of qubits in the state
 * @param state Output state pointer
 * @return QGT_SUCCESS on success, error code otherwise
 */
qgt_error_t mock_create_state(qgt_context_t* ctx,
                              size_t num_qubits,
                              qgt_state_t** state);

/**
 * Create a quantum state (production API)
 * This is the expected test API that wraps mock_create_state
 */
qgt_error_t qgt_create_state(qgt_context_t* ctx,
                             size_t num_qubits,
                             qgt_state_t** state);

/**
 * Create an entangled state for testing
 * @param ctx Quantum geometric context
 * @param type Type of entanglement (0=Bell, 1=GHZ, etc.)
 * @param num_qubits Number of qubits in the state
 * @param state Output state pointer
 * @return QGT_SUCCESS on success, error code otherwise
 */
qgt_error_t mock_create_entangled_state(qgt_context_t* ctx,
                                        int type,
                                        size_t num_qubits,
                                        qgt_state_t** state);

/**
 * Destroy a mock state
 * @param ctx Quantum geometric context
 * @param state State to destroy
 */
void mock_destroy_state(qgt_context_t* ctx, qgt_state_t* state);

/**
 * Create a random quantum state for testing
 * @param ctx Quantum geometric context
 * @param num_qubits Number of qubits
 * @param state Output state pointer
 * @return QGT_SUCCESS on success, error code otherwise
 */
qgt_error_t mock_create_random_state(qgt_context_t* ctx,
                                     size_t num_qubits,
                                     qgt_state_t** state);

/**
 * Create a product state for testing
 * @param ctx Quantum geometric context
 * @param num_qubits Number of qubits
 * @param state Output state pointer
 * @return QGT_SUCCESS on success, error code otherwise
 */
qgt_error_t mock_create_product_state(qgt_context_t* ctx,
                                      size_t num_qubits,
                                      qgt_state_t** state);

// ============================================================================
// Context Management Functions
// ============================================================================

/**
 * Create a QGT context for testing
 * @param ctx Output context pointer
 * @return QGT_SUCCESS on success
 */
qgt_error_t qgt_create_context(qgt_context_t** ctx);

/**
 * Destroy a QGT context
 * @param ctx Context to destroy
 */
void qgt_destroy_context(qgt_context_t* ctx);

/**
 * Destroy a QGT state
 * @param ctx Context
 * @param state State to destroy
 */
void qgt_destroy_state(qgt_context_t* ctx, qgt_state_t* state);

// ============================================================================
// Distributed Computing Functions (TDD - required by tests)
// ============================================================================

/**
 * Create distributed context
 * @param ctx Parent context
 * @param num_nodes Number of distributed nodes
 * @param dist_ctx Output distributed context
 * @return QGT_SUCCESS on success
 */
qgt_error_t qgt_create_distributed_context(qgt_context_t* ctx,
                                           size_t num_nodes,
                                           qgt_distributed_context_t** dist_ctx);

/**
 * Destroy distributed context
 * @param dist_ctx Distributed context to destroy
 */
void qgt_destroy_distributed_context(qgt_distributed_context_t* dist_ctx);

/**
 * Distribute state across nodes
 * @param dist_ctx Distributed context
 * @param state State to distribute
 * @return QGT_SUCCESS on success
 */
qgt_error_t qgt_distribute_state(qgt_distributed_context_t* dist_ctx, qgt_state_t* state);

/**
 * Gather distributed state
 * @param dist_ctx Distributed context
 * @param state State to gather into
 * @return QGT_SUCCESS on success
 */
qgt_error_t qgt_gather_state(qgt_distributed_context_t* dist_ctx, qgt_state_t* state);

/**
 * Distributed geometric rotation
 * @param dist_ctx Distributed context
 * @param state State to rotate
 * @param angle Rotation angle
 * @param axis Rotation axis
 * @return QGT_SUCCESS on success
 */
qgt_error_t qgt_distributed_geometric_rotate(qgt_distributed_context_t* dist_ctx,
                                              qgt_state_t* state,
                                              double angle,
                                              const double* axis);

/**
 * Distributed geometric parallel transport
 */
qgt_error_t qgt_distributed_geometric_parallel_transport(qgt_distributed_context_t* dist_ctx,
                                                          qgt_state_t* state,
                                                          const double* path,
                                                          size_t num_points);

/**
 * Simulate node failure for testing
 * @param dist_ctx Distributed context
 * @param node_id Node to fail
 * @return QGT_SUCCESS on success
 */
qgt_error_t qgt_simulate_node_failure(qgt_distributed_context_t* dist_ctx, size_t node_id);

// ============================================================================
// Additional Functions Expected by Tests
// ============================================================================

/**
 * Apply error channel to state
 */
qgt_error_t qgt_apply_error_channel(qgt_context_t* ctx, qgt_state_t* state, double error_rate);

/**
 * Apply error correction to state
 */
qgt_error_t qgt_apply_error_correction(qgt_context_t* ctx, qgt_state_t* state);

/**
 * Measure fidelity between states
 */
qgt_error_t qgt_geometric_measure_fidelity(qgt_context_t* ctx,
                                            qgt_state_t* state1,
                                            qgt_state_t* state2,
                                            double* fidelity);

/**
 * Create logical state
 */
qgt_error_t qgt_create_logical_state(qgt_context_t* ctx, size_t num_qubits, qgt_state_t** state);

/**
 * Enable noise model
 */
qgt_error_t qgt_enable_noise_model(qgt_context_t* ctx, double noise_strength);

/**
 * Apply geometric rotation
 */
qgt_error_t qgt_geometric_rotate(qgt_context_t* ctx, qgt_state_t* state, double angle, const double* axis);

/**
 * Apply error correction cycle
 */
qgt_error_t qgt_apply_error_correction_cycle(qgt_context_t* ctx, qgt_state_t* state);

/**
 * Measure logical fidelity
 */
qgt_error_t qgt_measure_logical_fidelity(qgt_context_t* ctx, qgt_state_t* state, double* fidelity);

/**
 * Compute geometric metric tensor
 */
qgt_error_t qgt_geometric_compute_metric(qgt_context_t* ctx, qgt_state_t* state, double* metric);

#ifdef __cplusplus
}
#endif

#endif // MOCK_QUANTUM_STATE_H
