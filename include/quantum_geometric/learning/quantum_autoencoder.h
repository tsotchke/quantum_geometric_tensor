/**
 * @file quantum_autoencoder.h
 * @brief Quantum autoencoder module for the Quantum Geometric Tensor Library
 *
 * Provides quantum variational autoencoders for dimensionality reduction
 * and quantum state compression using variational quantum circuits.
 */

#ifndef QUANTUM_AUTOENCODER_H
#define QUANTUM_AUTOENCODER_H

#include <stddef.h>
#include <stdbool.h>
#include "quantum_geometric/core/quantum_types.h"
#include "quantum_geometric/core/error_codes.h"
#include "quantum_geometric/learning/quantum_clustering.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct distributed_manager_t;

// =============================================================================
// Enumerations
// =============================================================================

/**
 * @brief Encoder architecture types
 */
typedef enum {
    ENCODER_VARIATIONAL = 0,   ///< Variational quantum encoder
    ENCODER_AMPLITUDE = 1,     ///< Amplitude encoding encoder
    ENCODER_ANGLE = 2,         ///< Angle encoding encoder
    ENCODER_HARDWARE = 3       ///< Hardware-efficient encoder
} EncoderType;

/**
 * @brief Decoder architecture types
 */
typedef enum {
    DECODER_QUANTUM = 0,      ///< Full quantum decoder
    DECODER_CLASSICAL = 1,    ///< Classical decoder
    DECODER_HYBRID = 2        ///< Hybrid quantum-classical decoder
} DecoderType;

/**
 * @brief Quantum activation function types
 */
typedef enum {
    ACTIVATION_QUANTUM_RELU = 0,     ///< Quantum ReLU activation
    ACTIVATION_QUANTUM_SIGMOID = 1,  ///< Quantum sigmoid activation
    ACTIVATION_QUANTUM_TANH = 2,     ///< Quantum tanh activation
    ACTIVATION_QUANTUM_ELU = 3       ///< Quantum ELU activation
} QuantumActivationType;

/**
 * @brief Regularization types
 */
typedef enum {
    REG_QUANTUM_ENTROPY = 0,  ///< Quantum entropy regularization
    REG_L1 = 1,               ///< L1 regularization
    REG_L2 = 2,               ///< L2 regularization
    REG_COMBINED = 3          ///< Combined L1+L2 regularization
} RegularizationType;

/**
 * @brief Quantum state types for synthetic data generation
 */
typedef enum {
    STATE_TYPE_PURE = 0,       ///< Pure quantum states
    STATE_TYPE_MIXED = 1,      ///< Mixed quantum states
    STATE_TYPE_ENTANGLED = 2   ///< Entangled states
} StateType;

/**
 * @brief Training status codes
 */
typedef enum {
    TRAINING_SUCCESS = 0,        ///< Training completed successfully
    TRAINING_ERROR = -1,         ///< General training error
    TRAINING_NOT_CONVERGED = -2, ///< Training did not converge
    TRAINING_EARLY_STOPPED = 1   ///< Training stopped early (good)
} TrainingStatus;

// =============================================================================
// Configuration Structures
// =============================================================================

/**
 * @brief Architecture configuration
 */
typedef struct {
    EncoderType encoder_type;           ///< Type of encoder
    DecoderType decoder_type;           ///< Type of decoder
    QuantumActivationType activation;   ///< Activation function
} architecture_config_t;

/**
 * @brief Regularization configuration
 */
typedef struct {
    RegularizationType type;  ///< Regularization type
    double strength;          ///< Regularization strength
} regularization_config_t;

/**
 * @brief Optimization configuration for autoencoder
 */
typedef struct {
    double learning_rate;              ///< Learning rate
    bool geometric_enhancement;        ///< Enable geometric enhancement
    regularization_config_t regularization;  ///< Regularization config
} autoencoder_optimization_config_t;

/**
 * @brief Main configuration for quantum autoencoder
 */
typedef struct {
    size_t input_dim;                           ///< Input dimension (num qubits)
    size_t latent_dim;                          ///< Latent dimension (compressed)
    size_t quantum_depth;                       ///< Circuit depth
    architecture_config_t architecture;         ///< Architecture config
    autoencoder_optimization_config_t optimization;  ///< Optimization config
} quantum_autoencoder_config_t;

/**
 * @brief Early stopping configuration
 */
typedef struct {
    bool enabled;        ///< Enable early stopping
    size_t patience;     ///< Number of epochs to wait
    double min_delta;    ///< Minimum improvement threshold
} early_stopping_config_t;

/**
 * @brief Training optimization configuration
 */
typedef struct {
    bool geometric_enhancement;      ///< Enable geometric enhancement
    bool error_mitigation;           ///< Enable error mitigation
    early_stopping_config_t early_stopping;  ///< Early stopping config
} training_optimization_config_t;

/**
 * @brief Training configuration
 */
typedef struct {
    size_t num_epochs;                         ///< Number of training epochs
    size_t batch_size;                         ///< Batch size
    double learning_rate;                      ///< Learning rate
    training_optimization_config_t optimization;  ///< Optimization config
} training_config_t;

// =============================================================================
// Result Structures
// =============================================================================

/**
 * @brief Training result
 */
typedef struct {
    TrainingStatus status;      ///< Training status
    size_t epochs_completed;    ///< Number of epochs completed
    double final_loss;          ///< Final training loss
    double best_loss;           ///< Best validation loss
    double* loss_history;       ///< Loss history per epoch
    size_t history_length;      ///< Length of loss history
} training_result_t;

/**
 * @brief Autoencoder evaluation result
 */
typedef struct {
    double reconstruction_error;  ///< Average reconstruction error
    double avg_state_fidelity;    ///< Average state fidelity
    double latent_entropy;        ///< Entropy in latent space
    double compression_ratio;     ///< Effective compression ratio
} autoencoder_evaluation_result_t;

// =============================================================================
// Opaque Model Structure
// =============================================================================

/**
 * @brief Quantum autoencoder model structure
 */
typedef struct quantum_autoencoder_t {
    size_t input_dim;                    ///< Input dimension
    size_t latent_dim;                   ///< Latent dimension
    size_t quantum_depth;                ///< Circuit depth
    quantum_autoencoder_config_t config; ///< Configuration
    double* encoder_params;              ///< Encoder parameters
    size_t num_encoder_params;           ///< Number of encoder parameters
    double* decoder_params;              ///< Decoder parameters
    size_t num_decoder_params;           ///< Number of decoder parameters
    bool is_trained;                     ///< Whether model is trained
    void* internal_data;                 ///< Internal implementation data
} quantum_autoencoder_t;

// =============================================================================
// Core API Functions
// =============================================================================

/**
 * @brief Create a quantum autoencoder model
 * @param config Configuration for the autoencoder
 * @return Pointer to created model, or NULL on failure
 */
quantum_autoencoder_t* quantum_autoencoder_create(const quantum_autoencoder_config_t* config);

/**
 * @brief Destroy a quantum autoencoder model
 * @param model Model to destroy
 */
void quantum_autoencoder_destroy(quantum_autoencoder_t* model);

// =============================================================================
// Encoding/Decoding Functions
// =============================================================================

/**
 * @brief Encode a quantum state to latent space
 * @param model Autoencoder model
 * @param state Input quantum state
 * @return Encoded (compressed) quantum state
 */
quantum_state_t* quantum_encode_state(quantum_autoencoder_t* model,
                                      quantum_state_t* state);

/**
 * @brief Decode a latent state back to input space
 * @param model Autoencoder model
 * @param state Latent quantum state
 * @return Decoded quantum state
 */
quantum_state_t* quantum_decode_state(quantum_autoencoder_t* model,
                                      quantum_state_t* state);

// =============================================================================
// Training Functions
// =============================================================================

/**
 * @brief Train autoencoder with distributed computing
 * @param model Autoencoder model
 * @param data Training quantum dataset
 * @param manager Distributed training manager
 * @param config Training configuration
 * @param context Optional context
 * @return Training result
 */
training_result_t quantum_train_autoencoder_distributed(
    quantum_autoencoder_t* model,
    quantum_dataset_t* data,
    struct distributed_manager_t* manager,
    const training_config_t* config,
    void* context);

/**
 * @brief Evaluate autoencoder performance
 * @param model Trained autoencoder
 * @param data Test dataset
 * @return Evaluation results
 */
autoencoder_evaluation_result_t quantum_evaluate_autoencoder(
    quantum_autoencoder_t* model,
    quantum_dataset_t* data);

// =============================================================================
// Data Generation Functions
// =============================================================================

/**
 * @brief Generate synthetic quantum states for training
 * @param num_samples Number of states to generate
 * @param dim Number of qubits
 * @param type Type of states to generate
 * @return Generated quantum dataset
 */
quantum_dataset_t* quantum_generate_synthetic_states(size_t num_samples,
                                                     size_t dim,
                                                     StateType type);

// =============================================================================
// State Utility Functions
// =============================================================================

/**
 * @brief Create a Bell state (maximally entangled 2-qubit state)
 * @return Bell state |00> + |11>
 */
quantum_state_t* quantum_create_bell_state(void);

/**
 * @brief Calculate fidelity between two quantum states
 * @param a First state
 * @param b Second state
 * @return Fidelity value in [0, 1]
 */
float quantum_autoencoder_state_fidelity(quantum_state_t* a, quantum_state_t* b);

/**
 * @brief Destroy a quantum state
 * @param state State to destroy
 */
void quantum_destroy_state(quantum_state_t* state);

// =============================================================================
// Model Persistence
// =============================================================================

/**
 * @brief Save autoencoder model to file
 * @param model Model to save
 * @param path File path
 * @return 0 on success, non-zero on failure
 */
int quantum_save_autoencoder_model(quantum_autoencoder_t* model, const char* path);

/**
 * @brief Load autoencoder model from file
 * @param path File path
 * @return Loaded model, or NULL on failure
 */
quantum_autoencoder_t* quantum_load_autoencoder_model(const char* path);

/**
 * @brief Check if two autoencoder models are equal
 * @param a First model
 * @param b Second model
 * @return true if models are equal
 */
bool autoencoder_models_equal(quantum_autoencoder_t* a, quantum_autoencoder_t* b);

// =============================================================================
// Test Helpers
// =============================================================================

/**
 * @brief Create a test autoencoder for unit tests
 * @param input_dim Input dimension
 * @param latent_dim Latent dimension
 * @param quantum_depth Circuit depth
 * @return Test autoencoder model
 */
quantum_autoencoder_t* create_test_autoencoder(size_t input_dim,
                                               size_t latent_dim,
                                               size_t quantum_depth);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_AUTOENCODER_H
