/**
 * @file quantum_diffusion.h
 * @brief Quantum diffusion processes with physics-informed neural networks
 *
 * This module implements quantum diffusion processes for modeling stochastic
 * quantum systems. It combines quantum mechanics with diffusion theory and
 * uses physics-informed neural networks (PINNs) for drift estimation.
 *
 * Key features:
 * - Quantum state diffusion with geometric phase
 * - Physics-informed neural network drift estimation
 * - PDE residual computation for validation
 * - Stochastic quantum evolution
 */

#ifndef QUANTUM_DIFFUSION_H
#define QUANTUM_DIFFUSION_H

#include <stdbool.h>
#include <stddef.h>
#include "quantum_geometric/core/quantum_types.h"
#include "quantum_geometric/core/error_codes.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Enumerations
// ============================================================================

/**
 * @brief PINN activation function types
 */
typedef enum {
    PINN_ACTIVATION_TANH = 0,    /**< Hyperbolic tangent */
    PINN_ACTIVATION_RELU = 1,    /**< Rectified linear unit */
    PINN_ACTIVATION_SIGMOID = 2, /**< Sigmoid function */
    PINN_ACTIVATION_SWISH = 3,   /**< Swish activation (x * sigmoid(x)) */
    PINN_ACTIVATION_GELU = 4     /**< Gaussian error linear unit */
} PINNActivationType;

/**
 * @brief Diffusion process types
 */
typedef enum {
    DIFFUSION_BROWNIAN = 0,      /**< Standard Brownian motion */
    DIFFUSION_GEOMETRIC = 1,     /**< Geometric Brownian motion */
    DIFFUSION_ORNSTEIN = 2,      /**< Ornstein-Uhlenbeck process */
    DIFFUSION_JUMP = 3           /**< Jump-diffusion process */
} DiffusionProcessType;

/**
 * @brief Diffusion status codes
 */
typedef enum {
    DIFFUSION_SUCCESS = 0,
    DIFFUSION_ERROR_NULL_PTR = -1,
    DIFFUSION_ERROR_INVALID_CONFIG = -2,
    DIFFUSION_ERROR_NOT_INITIALIZED = -3,
    DIFFUSION_ERROR_CONVERGENCE = -4,
    DIFFUSION_ERROR_MEMORY = -5
} DiffusionStatus;

// ============================================================================
// Configuration Structures
// ============================================================================

/**
 * @brief PINN (Physics-Informed Neural Network) configuration
 */
typedef struct pinn_config {
    size_t input_dim;                 /**< Input dimension (state dimension) */
    size_t hidden_dim;                /**< Hidden layer dimension */
    size_t num_layers;                /**< Number of hidden layers */
    size_t output_dim;                /**< Output dimension (drift dimension) */
    PINNActivationType activation;    /**< Activation function type */
    float learning_rate;              /**< Learning rate for training */
    float tolerance;                  /**< Convergence tolerance */
    float* params;                    /**< Network parameters (weights/biases) */
    size_t num_params;                /**< Total number of parameters */
    bool initialized;                 /**< Initialization flag */
} pinn_config;

/**
 * @brief Quantum diffusion configuration
 */
typedef struct {
    size_t num_qubits;                /**< Number of qubits in the system */
    DiffusionProcessType process_type; /**< Type of diffusion process */
    float sigma;                      /**< Volatility/diffusion coefficient */
    float dt;                         /**< Time step for evolution */
    bool use_geometric_phase;         /**< Include geometric phase effects */
    bool use_error_mitigation;        /**< Apply error mitigation */
    pinn_config* pinn;                /**< PINN configuration for drift estimation */
} quantum_diffusion_config_t;

/**
 * @brief Diffusion state for tracking evolution
 */
typedef struct {
    quantum_state_t* state;           /**< Current quantum state */
    float time;                       /**< Current time */
    float* drift_history;             /**< History of drift values */
    float* phase_history;             /**< History of geometric phases */
    size_t history_length;            /**< Length of history arrays */
    size_t max_history;               /**< Maximum history capacity */
} diffusion_state_t;

// ============================================================================
// PINN Functions
// ============================================================================

/**
 * @brief Initialize PINN network
 *
 * Allocates and initializes the neural network parameters using Xavier
 * initialization for weights and zero initialization for biases.
 *
 * @param config PINN configuration (modified with allocated params)
 * @return true on success, false on failure
 */
bool pinn_initialize(pinn_config* config);

/**
 * @brief Clean up PINN resources
 *
 * Frees all allocated memory for network parameters.
 *
 * @param config PINN configuration to clean up
 */
void pinn_cleanup(pinn_config* config);

/**
 * @brief Estimate drift using PINN
 *
 * Computes the drift vector for the quantum state at a given time using
 * the physics-informed neural network.
 *
 * @param state Current quantum state
 * @param t Current time
 * @param config PINN configuration
 * @return Allocated drift vector (caller must free), NULL on error
 */
float* estimate_drift(quantum_state_t* state, float t, pinn_config* config);

/**
 * @brief Forward pass through PINN
 *
 * Performs a forward pass through the neural network to compute output.
 *
 * @param input Input vector
 * @param input_size Input dimension
 * @param output Output vector (pre-allocated)
 * @param output_size Output dimension
 * @param config PINN configuration
 * @return DIFFUSION_SUCCESS on success
 */
DiffusionStatus pinn_forward(const float* input, size_t input_size,
                             float* output, size_t output_size,
                             const pinn_config* config);

/**
 * @brief Train PINN on physics residuals
 *
 * Trains the network to minimize PDE residuals using gradient descent.
 *
 * @param config PINN configuration
 * @param states Array of training states
 * @param times Array of corresponding times
 * @param num_samples Number of training samples
 * @param num_epochs Number of training epochs
 * @return DIFFUSION_SUCCESS on success
 */
DiffusionStatus pinn_train(pinn_config* config,
                           quantum_state_t** states,
                           const float* times,
                           size_t num_samples,
                           size_t num_epochs);

// ============================================================================
// PDE Residual Functions
// ============================================================================

/**
 * @brief Compute PDE residual
 *
 * Computes the residual of the quantum diffusion PDE for a given state.
 * The PDE is: dρ/dt = L[ρ] + D[ρ]
 * where L is the Lindblad superoperator and D is the diffusion term.
 *
 * @param state Current quantum state
 * @param t Current time
 * @param sigma Pointer to sigma parameters (volatility array)
 * @return PDE residual value
 */
float compute_pde_residual(quantum_state_t* state, float t, float* sigma);

/**
 * @brief Compute Lindblad dissipator term
 *
 * @param state Current quantum state
 * @return Dissipator contribution to residual
 */
float compute_lindblad_term(quantum_state_t* state);

/**
 * @brief Compute diffusion term
 *
 * @param state Current quantum state
 * @param sigma Diffusion coefficient
 * @return Diffusion contribution to residual
 */
float compute_diffusion_term(quantum_state_t* state, float sigma);

// ============================================================================
// Geometric Phase Functions
// ============================================================================

/**
 * @brief Compute geometric phase for diffusion
 *
 * Computes the Berry phase accumulated during quantum diffusion.
 * This is distinct from the attention-based geometric phase.
 *
 * @param state Current quantum state
 * @return Geometric phase in radians
 */
float compute_diffusion_geometric_phase(quantum_state_t* state);

/**
 * @brief Compute instantaneous geometric phase rate
 *
 * @param state Current quantum state
 * @param drift Current drift vector
 * @return Rate of geometric phase accumulation
 */
float compute_phase_rate(quantum_state_t* state, const float* drift);

// ============================================================================
// State Evolution Functions
// ============================================================================

/**
 * @brief Update quantum state with drift-diffusion
 *
 * Evolves the quantum state according to the stochastic differential equation:
 * d|ψ⟩ = drift*dt + sigma*dW + i*phase*|ψ⟩*dt
 *
 * @param state Quantum state to update (modified in place)
 * @param drift Drift vector
 * @param sigma Diffusion coefficient
 * @param phase Geometric phase
 * @param dt Time step
 * @return QGT_SUCCESS on success
 */
qgt_error_t quantum_diffusion_state_update(quantum_state_t* state,
                                           float* drift,
                                           float sigma,
                                           float phase,
                                           float dt);

/**
 * @brief Perform single diffusion step
 *
 * Combines drift estimation, phase computation, and state update.
 *
 * @param state Quantum state (modified)
 * @param t Current time
 * @param dt Time step
 * @param config Diffusion configuration
 * @return QGT_SUCCESS on success
 */
qgt_error_t quantum_diffusion_step(quantum_state_t* state,
                                   float t,
                                   float dt,
                                   const quantum_diffusion_config_t* config);

/**
 * @brief Evolve state over multiple time steps
 *
 * @param state Quantum state (modified)
 * @param t_start Start time
 * @param t_end End time
 * @param num_steps Number of steps
 * @param config Diffusion configuration
 * @return QGT_SUCCESS on success
 */
qgt_error_t quantum_diffusion_evolve(quantum_state_t* state,
                                     float t_start,
                                     float t_end,
                                     size_t num_steps,
                                     const quantum_diffusion_config_t* config);

// ============================================================================
// State Management Functions
// ============================================================================

/**
 * @brief Create quantum state for diffusion
 *
 * @param num_qubits Number of qubits
 * @return Allocated quantum state, NULL on error
 */
quantum_state_t* quantum_diffusion_create_state(size_t num_qubits);

/**
 * @brief Destroy quantum state
 *
 * @param state State to destroy
 */
void quantum_diffusion_destroy_state(quantum_state_t* state);

/**
 * @brief Set state amplitudes
 *
 * @param state Quantum state
 * @param amps Complex amplitudes array
 * @param num_amps Number of amplitudes
 */
void quantum_state_set_amplitudes(quantum_state_t* state,
                                  ComplexFloat* amps,
                                  size_t num_amps);

/**
 * @brief Get single amplitude
 *
 * @param state Quantum state
 * @param index Amplitude index
 * @return Complex amplitude value
 */
ComplexFloat quantum_state_get_amplitude(quantum_state_t* state, size_t index);

/**
 * @brief Normalize quantum state
 *
 * @param state State to normalize
 */
void quantum_diffusion_normalize(quantum_state_t* state);

// ============================================================================
// Diffusion State Tracking
// ============================================================================

/**
 * @brief Create diffusion state tracker
 *
 * @param state Initial quantum state
 * @param max_history Maximum history to track
 * @return Allocated diffusion state, NULL on error
 */
diffusion_state_t* diffusion_state_create(quantum_state_t* state,
                                          size_t max_history);

/**
 * @brief Destroy diffusion state tracker
 *
 * @param dstate Diffusion state to destroy
 */
void diffusion_state_destroy(diffusion_state_t* dstate);

/**
 * @brief Record current state to history
 *
 * @param dstate Diffusion state tracker
 * @param drift Current drift value (magnitude)
 * @param phase Current geometric phase
 */
void diffusion_state_record(diffusion_state_t* dstate,
                            float drift,
                            float phase);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Create default PINN configuration
 *
 * @param state_dim Dimension of quantum state
 * @return Default PINN configuration
 */
pinn_config pinn_create_default_config(size_t state_dim);

/**
 * @brief Create default diffusion configuration
 *
 * @param num_qubits Number of qubits
 * @return Default diffusion configuration
 */
quantum_diffusion_config_t quantum_diffusion_create_default_config(size_t num_qubits);

/**
 * @brief Validate diffusion configuration
 *
 * @param config Configuration to validate
 * @return true if valid, false otherwise
 */
bool quantum_diffusion_validate_config(const quantum_diffusion_config_t* config);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_DIFFUSION_H
