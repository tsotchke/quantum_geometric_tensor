#ifndef QUANTUM_FUNCTIONS_H
#define QUANTUM_FUNCTIONS_H

#include <stdbool.h>
#include "quantum_geometric/hybrid/classical_optimization_engine.h"
#include "quantum_geometric/core/hierarchical_matrix.h"
#include "quantum_geometric/core/tensor_network_operations.h"
#include "quantum_geometric/learning/learning_task.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize weights for a quantum network
 * 
 * @param matrix The hierarchical matrix to initialize
 * @param rows Number of rows in the matrix
 * @param cols Number of columns in the matrix
 * @return true if successful, false otherwise
 */
bool quantum_initialize_weights(HierarchicalMatrix* matrix, size_t rows, size_t cols);

/**
 * @brief Create a tensor network for quantum operations
 * 
 * @param features Input features
 * @param batch_size Batch size
 * @param input_dim Input dimension
 * @return tensor_network_t* Created tensor network, or NULL on failure
 */
tensor_network_t* quantum_create_tensor_network(
    const double* features,
    size_t batch_size,
    size_t input_dim
);

/**
 * @brief Get layer weights from model state
 * 
 * @param model_state Model state as hierarchical matrix
 * @param layer Layer index
 * @return HierarchicalMatrix* Layer weights, or NULL on failure
 */
HierarchicalMatrix* quantum_get_layer_weights(
    const HierarchicalMatrix* model_state,
    size_t layer
);

/**
 * @brief Multiply tensor network by weights
 * 
 * @param network Tensor network
 * @param weights Weight matrix
 * @return true if successful, false otherwise
 */
bool quantum_tensor_network_multiply(
    tensor_network_t* network,
    const HierarchicalMatrix* weights
);

/**
 * @brief Apply activation function to tensor network
 * 
 * @param network Tensor network
 * @param activation_type Type of activation ("relu", "tanh", etc)
 * @return true if successful, false otherwise
 */
bool quantum_apply_activation(
    tensor_network_t* network,
    const char* activation_type
);

/**
 * @brief Extract output from tensor network
 * 
 * @param network Tensor network
 * @param output Output buffer
 * @return true if successful, false otherwise
 */
bool quantum_extract_output(
    const tensor_network_t* network,
    double* output
);

/**
 * @brief Create gradient network for backpropagation
 * 
 * @param features Input features
 * @param labels True labels
 * @param predictions Model predictions
 * @param batch_size Batch size
 * @param input_dim Input dimension
 * @param output_dim Output dimension
 * @return tensor_network_t* Created gradient network, or NULL on failure
 */
tensor_network_t* quantum_create_gradient_network(
    const double* features,
    const double* labels,
    const double* predictions,
    size_t batch_size,
    size_t input_dim,
    size_t output_dim
);

/**
 * @brief Free tensor network resources
 * 
 * @param network Tensor network to free
 */
void quantum_free_tensor_network(tensor_network_t* network);

/**
 * @brief Calculate gradients for network weights
 * 
 * @param network Gradient network
 * @param weights Weight matrix
 * @return true if successful, false otherwise
 */
bool quantum_calculate_gradients(
    tensor_network_t* network,
    HierarchicalMatrix* weights
);

// Optimizer state structure for managing momentum and adaptive learning rates
typedef struct {
    optimizer_type_t type;  // Type of optimizer being used
    double beta1;          // Exponential decay rate for first moment (Adam)
    double beta2;          // Exponential decay rate for second moment (Adam)
    double epsilon;        // Small constant to prevent division by zero
    size_t t;             // Time step counter
    ComplexFloat* m;       // First moment vector (Adam)
    ComplexFloat* v;       // Second moment vector (Adam)
    size_t size;          // Size of moment vectors
} quantum_optimizer_state_t;

/**
 * @brief Create and initialize an optimizer state
 * 
 * @param type Type of optimizer (OPTIMIZER_QUANTUM_GRADIENT_DESCENT or OPTIMIZER_QUANTUM_ADAM)
 * @param param_size Number of parameters in the model
 * @return quantum_optimizer_state_t* Initialized optimizer state, or NULL on failure
 */
quantum_optimizer_state_t* quantum_create_optimizer_state(
    optimizer_type_t type,
    size_t param_size
);

/**
 * @brief Free resources used by optimizer state
 * 
 * @param state Optimizer state to free
 */
void quantum_free_optimizer_state(quantum_optimizer_state_t* state);

/**
 * @brief Update weights using specified optimizer
 * 
 * @param weights Weight matrix to update
 * @param network Gradient network containing gradients
 * @param learning_rate Base learning rate for updates
 * @param optimizer_state Optimizer state containing momentum/adaptive terms
 * @return true if successful, false otherwise
 */
bool quantum_update_weights(
    HierarchicalMatrix* weights,
    tensor_network_t* network,
    double learning_rate,
    quantum_optimizer_state_t* optimizer_state,
    const ComplexFloat* weight_gradients
);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_FUNCTIONS_H
