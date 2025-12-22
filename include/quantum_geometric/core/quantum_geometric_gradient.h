#ifndef QUANTUM_GEOMETRIC_GRADIENT_H
#define QUANTUM_GEOMETRIC_GRADIENT_H

#include "quantum_geometric/core/quantum_geometric_tensor_network.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Compute gradient of quantum state with respect to a parameter
 * 
 * Uses the parameter shift rule to compute the gradient:
 * df/dθ = [f(θ + π/2) - f(θ - π/2)]/2
 * 
 * @param qgtn The quantum geometric tensor network
 * @param param_idx Index of the parameter to compute gradient for
 * @param gradient Output pointer to gradient array (caller must free)
 * @param dimension Output dimension of gradient array
 * @return true if successful, false otherwise
 */
bool compute_quantum_gradient(
    const quantum_geometric_tensor_network_t* qgtn,
    size_t param_idx,
    ComplexFloat** gradient,
    size_t* dimension);

/**
 * @brief Compute natural gradient using quantum geometric tensor
 * 
 * The natural gradient is computed using the quantum geometric tensor
 * as a metric: g_nat = G^{-1} g where G is the metric tensor
 * 
 * @param qgtn The quantum geometric tensor network
 * @param param_idx Index of the parameter to compute gradient for
 * @param natural_gradient Output pointer to natural gradient array (caller must free)
 * @param dimension Output dimension of gradient array
 * @return true if successful, false otherwise
 */
bool compute_quantum_natural_gradient(
    const quantum_geometric_tensor_network_t* qgtn,
    size_t param_idx,
    ComplexFloat** natural_gradient,
    size_t* dimension);

/**
 * @brief Compute gradient of expectation value
 * 
 * Computes gradient of expectation value using:
 * d<O>/dθ = <dψ/dθ|O|ψ> + <ψ|O|dψ/dθ>
 * 
 * @param qgtn The quantum geometric tensor network
 * @param op The quantum operator to compute expectation for
 * @param param_idx Index of the parameter to compute gradient for
 * @param gradient Output gradient value
 * @return true if successful, false otherwise
 */
bool compute_expectation_gradient(
    const quantum_geometric_tensor_network_t* qgtn,
    const quantum_geometric_operator_t* op,
    size_t param_idx,
    double* gradient);

// Advanced gradient computation options
typedef struct {
    double shift_amount;           // Amount to shift parameters (default: π/2)
    bool use_centered_difference;  // Use centered difference formula
    bool use_higher_order;         // Use higher order approximation
    size_t num_points;            // Number of points for higher order
    bool accumulate_gradients;     // Accumulate gradients over parameters
    void* custom_options;          // Additional custom options
} gradient_options_t;

/**
 * @brief Set options for gradient computation
 * 
 * @param options The options to set
 * @return true if successful, false otherwise
 */
bool set_gradient_options(const gradient_options_t* options);

/**
 * @brief Get current gradient computation options
 * 
 * @param options Output options structure
 * @return true if successful, false otherwise
 */
bool get_gradient_options(gradient_options_t* options);

/**
 * @brief Compute higher order gradient using multiple shifts
 *
 * @param qgtn The quantum geometric tensor network
 * @param param_idx Index of the parameter
 * @param shift_amounts Array of shift amounts
 * @param num_shifts Number of shifts
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
    size_t* dimension
);

/**
 * @brief Count the number of trainable parameters in the quantum circuit
 *
 * @param qgtn The quantum geometric tensor network
 * @return Number of trainable parameters, or 0 if none found
 */
size_t count_trainable_parameters(const quantum_geometric_tensor_network_t* qgtn);

/**
 * @brief Compute loss gradient using parameter shift rule for a single parameter
 *
 * Computes the gradient of the MSE loss with respect to a circuit parameter:
 * dL/dθ = [L(θ + π/2) - L(θ - π/2)] / 2
 *
 * @param qgtn The quantum geometric tensor network (will be modified temporarily)
 * @param param_idx Index of the parameter to compute gradient for
 * @param labels Target labels array
 * @param num_samples Number of samples
 * @param output_dim Dimension of output
 * @param gradient Output gradient value
 * @return true if successful, false otherwise
 */
bool compute_loss_gradient_single_param(
    quantum_geometric_tensor_network_t* qgtn,
    size_t param_idx,
    const ComplexFloat* labels,
    size_t num_samples,
    size_t output_dim,
    double* gradient);

/**
 * @brief Compute loss gradients for ALL trainable parameters
 *
 * Uses the parameter shift rule to compute the gradient of the loss function
 * with respect to each trainable parameter in the quantum circuit.
 *
 * @param qgtn The quantum geometric tensor network (will be modified temporarily)
 * @param labels Target labels array
 * @param num_samples Number of samples
 * @param output_dim Dimension of output
 * @param gradients Output array of gradients (caller must free)
 * @param num_gradients Output number of gradients (equals number of trainable params)
 * @return true if successful, false otherwise
 */
bool compute_all_loss_gradients(
    quantum_geometric_tensor_network_t* qgtn,
    const ComplexFloat* labels,
    size_t num_samples,
    size_t output_dim,
    double** gradients,
    size_t* num_gradients);

/**
 * @brief Run forward pass and compute loss
 *
 * Helper function that runs the forward pass through the quantum circuit
 * and computes the MSE loss against the target labels.
 *
 * @param qgtn The quantum geometric tensor network
 * @param labels Target labels array
 * @param num_samples Number of samples
 * @param output_dim Dimension of output
 * @param loss Output loss value
 * @return true if successful, false otherwise
 */
bool compute_forward_loss(
    quantum_geometric_tensor_network_t* qgtn,
    const ComplexFloat* labels,
    size_t num_samples,
    size_t output_dim,
    double* loss);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_GEOMETRIC_GRADIENT_H
