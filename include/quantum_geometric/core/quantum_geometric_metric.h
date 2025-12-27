#ifndef QUANTUM_GEOMETRIC_METRIC_H
#define QUANTUM_GEOMETRIC_METRIC_H

#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_state.h"
#include "quantum_geometric/core/quantum_geometric_tensor.h"

// Forward declarations
struct quantum_geometric_tensor_network;
typedef struct quantum_geometric_tensor_network quantum_geometric_tensor_network_t;

// Create geometric metric
qgt_error_t geometric_create_metric(quantum_geometric_metric_t** metric,
                                  geometric_metric_type_t type,
                                  size_t dimension);

// Destroy geometric metric
void geometric_destroy_metric(quantum_geometric_metric_t* metric);

// Compute geometric metric from state (legacy API - for Euclidean/Minkowski metrics)
// Note: For Fubini-Study metric with parameterized circuits, use
// geometric_compute_fubini_study_metric() instead.
qgt_error_t geometric_compute_metric(quantum_geometric_metric_t* metric,
                                   const quantum_state_t* state);

// Clone geometric metric
qgt_error_t geometric_clone_metric(quantum_geometric_metric_t** dest,
                                 const quantum_geometric_metric_t* src);

// Transform geometric metric
qgt_error_t geometric_transform_metric(quantum_geometric_metric_t* result,
                                     const quantum_geometric_metric_t* metric,
                                     const quantum_geometric_tensor_t* transform);

/**
 * @brief Compute the Fubini-Study metric for a parameterized quantum circuit.
 *
 * The Fubini-Study metric is defined as:
 *   g_μν = Re[<∂_μψ|∂_νψ> - <∂_μψ|ψ><ψ|∂_νψ>]
 *
 * This is the correct quantum natural gradient metric for variational circuits.
 *
 * @param metric Output metric tensor (must be pre-allocated with dimension = num_params)
 * @param qgtn The quantum geometric tensor network containing the circuit
 * @param num_params Number of variational parameters
 * @return QGT_SUCCESS on success, error code otherwise
 */
qgt_error_t geometric_compute_fubini_study_metric(
    quantum_geometric_metric_t* metric,
    const quantum_geometric_tensor_network_t* qgtn,
    size_t num_params);

/**
 * @brief Compute a single element of the Fubini-Study metric tensor.
 *
 * Computes g_μν = Re[<∂_μψ|∂_νψ> - <∂_μψ|ψ><ψ|∂_νψ>] for specific indices.
 *
 * @param qgtn The quantum geometric tensor network
 * @param param_mu First parameter index
 * @param param_nu Second parameter index
 * @param result Output: the metric element g_μν
 * @return QGT_SUCCESS on success, error code otherwise
 */
qgt_error_t geometric_compute_fubini_study_element(
    const quantum_geometric_tensor_network_t* qgtn,
    size_t param_mu,
    size_t param_nu,
    float* result);

#endif // QUANTUM_GEOMETRIC_METRIC_H
