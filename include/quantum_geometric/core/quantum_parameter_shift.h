#ifndef QUANTUM_PARAMETER_SHIFT_H
#define QUANTUM_PARAMETER_SHIFT_H

#include "quantum_geometric/core/quantum_geometric_tensor_network.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Compute gradient using parameter shift rule
 * 
 * Uses the parameter shift rule:
 * df/dθ = [f(θ + r) - f(θ - r)]/(2r)
 * where r is the shift amount
 * 
 * @param qgtn The quantum geometric tensor network
 * @param param_idx Index of the parameter to compute gradient for
 * @param shift_amount Amount to shift the parameter
 * @param gradient Output pointer to gradient array (caller must free)
 * @param dimension Output dimension of gradient array
 * @return true if successful, false otherwise
 */
bool compute_parameter_shift_gradient(
    const quantum_geometric_tensor_network_t* qgtn,
    size_t param_idx,
    double shift_amount,
    ComplexFloat** gradient,
    size_t* dimension);

/**
 * @brief Compute higher order gradient using multiple shifts
 * 
 * Uses multiple parameter shifts and Richardson extrapolation
 * to compute a higher order gradient approximation
 * 
 * @param qgtn The quantum geometric tensor network
 * @param param_idx Index of the parameter to compute gradient for
 * @param shift_amounts Array of shift amounts
 * @param num_shifts Number of shifts to use
 * @param gradient Output pointer to gradient array (caller must free)
 * @param dimension Output dimension of gradient array
 * @return true if successful, false otherwise
 */
bool compute_higher_order_gradient(
    const quantum_geometric_tensor_network_t* qgtn,
    size_t param_idx,
    const double* shift_amounts,
    size_t num_shifts,
    ComplexFloat** gradient,
    size_t* dimension);

/**
 * @brief Compute centered finite difference gradient
 * 
 * Uses centered finite difference formula:
 * df/dθ = [f(θ + h) - f(θ - h)]/(2h)
 * where h is the step size
 * 
 * @param qgtn The quantum geometric tensor network
 * @param param_idx Index of the parameter to compute gradient for
 * @param step_size Step size for finite difference
 * @param gradient Output pointer to gradient array (caller must free)
 * @param dimension Output dimension of gradient array
 * @return true if successful, false otherwise
 */
bool compute_centered_difference_gradient(
    const quantum_geometric_tensor_network_t* qgtn,
    size_t param_idx,
    double step_size,
    ComplexFloat** gradient,
    size_t* dimension);

/**
 * @brief Compute gradient with error estimation
 * 
 * Computes gradient using two different step sizes and
 * estimates the error from their difference
 * 
 * @param qgtn The quantum geometric tensor network
 * @param param_idx Index of the parameter to compute gradient for
 * @param gradient Output pointer to gradient array (caller must free)
 * @param error_estimate Output error estimate
 * @param dimension Output dimension of gradient array
 * @return true if successful, false otherwise
 */
bool compute_gradient_with_error(
    const quantum_geometric_tensor_network_t* qgtn,
    size_t param_idx,
    ComplexFloat** gradient,
    double* error_estimate,
    size_t* dimension);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_PARAMETER_SHIFT_H
