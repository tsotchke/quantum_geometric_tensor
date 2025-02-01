#ifndef QGT_TEST_HELPERS_H
#define QGT_TEST_HELPERS_H

#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/core/quantum_geometric_operations.h"
#include <complex.h>
#include <stdbool.h>

/**
 * @brief Initialize state to |0⟩
 * @param ctx Context for quantum operations
 * @param state State to initialize
 */
void init_zero_state(qgt_context_t* ctx, qgt_state_t* state);

/**
 * @brief Initialize state with random amplitudes
 * @param ctx Context for quantum operations
 * @param state State to initialize
 */
void init_random_state(qgt_context_t* ctx, qgt_state_t* state);

/**
 * @brief Initialize state to specific basis state
 * @param ctx Context for quantum operations
 * @param state State to initialize
 * @param basis Index of basis state
 */
void init_basis_state(qgt_context_t* ctx, qgt_state_t* state, size_t basis);

/**
 * @brief Initialize state to Bell state (|00⟩ + |11⟩)/√2
 * @param ctx Context for quantum operations
 * @param state State to initialize
 */
void init_bell_state(qgt_context_t* ctx, qgt_state_t* state);

/**
 * @brief Initialize state to GHZ state (|00...0⟩ + |11...1⟩)/√2
 * @param ctx Context for quantum operations
 * @param state State to initialize
 */
void init_ghz_state(qgt_context_t* ctx, qgt_state_t* state);

/**
 * @brief Initialize state to W state (|100...0⟩ + |010...0⟩ + ... + |000...1⟩)/√n
 * @param ctx Context for quantum operations
 * @param state State to initialize
 */
void init_w_state(qgt_context_t* ctx, qgt_state_t* state);

/**
 * @brief Verify state is properly normalized
 * @param state State to verify
 * @param tolerance Maximum allowed deviation from unit norm
 * @return true if state is normalized within tolerance
 */
bool verify_state_normalization(const qgt_state_t* state, double tolerance);

/**
 * @brief Verify two states are orthogonal
 * @param state1 First state
 * @param state2 Second state
 * @param tolerance Maximum allowed inner product magnitude
 * @return true if states are orthogonal within tolerance
 */
bool verify_state_orthogonality(const qgt_state_t* state1,
                              const qgt_state_t* state2,
                              double tolerance);

/**
 * @brief Verify geometric phase acquired during parallel transport
 * @param ctx Context for quantum operations
 * @param state Initial state
 * @param path Path points as array of 3D coordinates
 * @param num_points Number of points in path
 * @param tolerance Maximum allowed deviation from pure phase
 * @return true if geometric phase is valid
 */
bool verify_geometric_phase(qgt_context_t* ctx,
                          const qgt_state_t* state,
                          const double* path,
                          size_t num_points,
                          double tolerance);

/**
 * @brief Verify metric tensor is positive definite
 * @param metric Metric tensor as flattened array
 * @param dim Dimension of metric tensor
 * @param tolerance Tolerance for positive definiteness check
 * @return true if metric is positive definite
 */
bool verify_metric_positivity(const double* metric,
                            size_t dim,
                            double tolerance);

/**
 * @brief Verify connection coefficients are compatible with metric
 * @param connection Connection coefficients as flattened array
 * @param metric Metric tensor as flattened array
 * @param dim Dimension of tensors
 * @param tolerance Maximum allowed incompatibility
 * @return true if connection is compatible with metric
 */
bool verify_connection_compatibility(const double complex* connection,
                                  const double* metric,
                                  size_t dim,
                                  double tolerance);

#endif /* QGT_TEST_HELPERS_H */
