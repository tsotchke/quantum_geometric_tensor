#ifndef QUANTUM_GEOMETRIC_CURVATURE_H
#define QUANTUM_GEOMETRIC_CURVATURE_H

#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_geometric_connection.h"
#include "quantum_geometric/core/quantum_geometric_tensor.h"

// Forward declarations
struct quantum_geometric_tensor_network;
typedef struct quantum_geometric_tensor_network quantum_geometric_tensor_network_t;

// Note: quantum_geometric_metric_t is defined in quantum_geometric_types.h

// Create geometric curvature
qgt_error_t geometric_create_curvature(quantum_geometric_curvature_t** curvature,
                                     geometric_curvature_type_t type,
                                     size_t dimension);

// Destroy geometric curvature
void geometric_destroy_curvature(quantum_geometric_curvature_t* curvature);

// Clone geometric curvature
qgt_error_t geometric_clone_curvature(quantum_geometric_curvature_t** dest,
                                    const quantum_geometric_curvature_t* src);

// Compute geometric curvature from connection (Riemann tensor)
qgt_error_t geometric_compute_curvature(quantum_geometric_curvature_t* curvature,
                                      const quantum_geometric_connection_t* connection);

// Transform geometric curvature
qgt_error_t geometric_transform_curvature(quantum_geometric_curvature_t* result,
                                        const quantum_geometric_curvature_t* curvature,
                                        const quantum_geometric_tensor_t* transform);

/**
 * @brief Compute the Berry curvature for a parameterized quantum circuit.
 *
 * The Berry curvature is defined as:
 *   Ω_μν = Im[<∂_μψ|∂_νψ> - <∂_μψ|ψ><ψ|∂_νψ>]
 *
 * This is the antisymmetric part of the Quantum Geometric Tensor.
 * Note: Ω_μν = -Ω_νμ (antisymmetric)
 *
 * @param curvature Output curvature tensor (must be pre-allocated with type GEOMETRIC_CURVATURE_BERRY)
 * @param qgtn The quantum geometric tensor network containing the circuit
 * @param num_params Number of variational parameters
 * @return QGT_SUCCESS on success, error code otherwise
 */
qgt_error_t geometric_compute_berry_curvature(
    quantum_geometric_curvature_t* curvature,
    const quantum_geometric_tensor_network_t* qgtn,
    size_t num_params);

/**
 * @brief Compute a single element of the Berry curvature tensor.
 *
 * Computes Ω_μν = Im[<∂_μψ|∂_νψ> - <∂_μψ|ψ><ψ|∂_νψ>] for specific indices.
 *
 * @param qgtn The quantum geometric tensor network
 * @param param_mu First parameter index
 * @param param_nu Second parameter index
 * @param result Output: the curvature element Ω_μν
 * @return QGT_SUCCESS on success, error code otherwise
 */
qgt_error_t geometric_compute_berry_curvature_element(
    const quantum_geometric_tensor_network_t* qgtn,
    size_t param_mu,
    size_t param_nu,
    float* result);

/**
 * @brief Compose the full Quantum Geometric Tensor from metric and Berry curvature.
 *
 * The QGT is defined as: Q_μν = g_μν + i*Ω_μν
 * where g_μν is the Fubini-Study metric and Ω_μν is the Berry curvature.
 *
 * @param qgt Output: the full QGT tensor (dimension x dimension complex matrix)
 * @param metric The Fubini-Study metric tensor (must be symmetric)
 * @param curvature The Berry curvature tensor (must be antisymmetric)
 * @param dimension The dimension of the parameter space
 * @return QGT_SUCCESS on success, error code otherwise
 */
qgt_error_t geometric_compose_qgt(
    ComplexFloat* qgt,
    const quantum_geometric_metric_t* metric,
    const quantum_geometric_curvature_t* curvature,
    size_t dimension);

/**
 * @brief Compute the full QGT directly from a parameterized circuit.
 *
 * This computes both the metric (real part) and Berry curvature (imaginary part)
 * in a single pass.
 *
 * @param qgt Output: the full QGT tensor (num_params x num_params complex matrix)
 * @param qgtn The quantum geometric tensor network containing the circuit
 * @param num_params Number of variational parameters
 * @return QGT_SUCCESS on success, error code otherwise
 */
qgt_error_t geometric_compute_full_qgt(
    ComplexFloat* qgt,
    const quantum_geometric_tensor_network_t* qgtn,
    size_t num_params);

#endif // QUANTUM_GEOMETRIC_CURVATURE_H
