#ifndef QUANTUM_LLM_CORE_H
#define QUANTUM_LLM_CORE_H

#include <stdint.h>
#include <stdbool.h>
#include "quantum_geometric/core/quantum_geometric_core.h"

// Status codes
typedef enum {
    QUANTUM_STATUS_SUCCESS,
    QUANTUM_STATUS_ERROR,
    QUANTUM_STATUS_INVALID_ARGUMENT,
    QUANTUM_STATUS_OUT_OF_MEMORY,
    QUANTUM_STATUS_HARDWARE_ERROR,
    QUANTUM_STATUS_COHERENCE_LOST
} quantum_status_t;

// Model configuration
typedef struct {
    uint64_t total_parameters;    // Total number of model parameters
    uint32_t model_layers;        // Number of model layers
    uint32_t embedding_dimension; // Embedding dimension
    uint32_t attention_dimension; // Attention head dimension
} ModelConfig;

// Encoding configuration
typedef struct {
    uint32_t geometric_dimension;     // Geometric encoding dimension
    float compression_ratio;          // Target compression ratio
    uint32_t encoding_qubits;         // Number of encoding qubits
    bool use_topological_protection;  // Enable topological protection
    uint32_t code_distance;           // Error correction code distance
    float error_threshold;            // Error correction threshold
} EncodingConfig;

// Distributed configuration
typedef struct {
    uint32_t quantum_nodes;          // Number of quantum nodes
    uint32_t qubits_per_node;        // Qubits per node
    float coherence_time;            // Coherence time in microseconds
    bool use_error_correction;       // Enable error correction
    uint32_t syndrome_qubits;        // Number of syndrome qubits
    float correction_threshold;       // Error correction threshold
} DistributedConfig;

// Tensor configuration
typedef struct {
    uint32_t tensor_dimension;       // Tensor network dimension
    uint32_t attention_heads;        // Number of attention heads
    float gate_fidelity;            // Target gate fidelity
    bool use_quantum_memory;        // Enable quantum memory
    bool parallel_execution;        // Enable parallel execution
    float operation_throughput;     // Target operations per second
} TensorConfig;

// Combined LLM configuration
typedef struct {
    ModelConfig model_config;
    EncodingConfig encoding_config;
    DistributedConfig distributed_config;
    TensorConfig tensor_config;
} QuantumLLMConfig;

// Opaque type declarations
typedef struct quantum_llm_state_t quantum_llm_state_t;
typedef struct quantum_geometric_state_t quantum_geometric_state_t;
typedef struct quantum_state_t quantum_state_t;
typedef struct quantum_attention_t quantum_attention_t;
typedef struct training_data_t training_data_t;

// Initialization and cleanup
quantum_status_t initialize_quantum_llm(const QuantumLLMConfig* config,
                                      quantum_llm_state_t** state);
void cleanup_quantum_llm(quantum_llm_state_t* state);

// Quantum geometric encoding
quantum_status_t encode_quantum_parameters(quantum_llm_state_t* state,
                                         const float* parameters,
                                         uint64_t param_count,
                                         quantum_geometric_state_t* quantum_state);
float calculate_compression_ratio(const void* distributed_system);
float measure_encoding_fidelity(const quantum_geometric_state_t* state);

// Distributed quantum processing
quantum_status_t prepare_test_quantum_state(quantum_state_t* state);
quantum_status_t distribute_quantum_input(quantum_state_t* state,
                                        void* distributed_system);
float measure_node_synchronization(uint32_t node,
                                 const void* distributed_system);
quantum_status_t execute_parallel_quantum_operations(quantum_state_t* state);
float measure_operation_throughput(const void* distributed_system);

// Quantum attention operations
quantum_status_t initialize_quantum_attention(quantum_attention_t* attention);
quantum_status_t execute_quantum_attention(uint32_t layer,
                                         quantum_attention_t* attention,
                                         const quantum_state_t* parameters,
                                         const quantum_state_t* input,
                                         quantum_state_t* output);
float measure_attention_quality(const quantum_attention_t* attention);
float measure_gate_fidelity(const quantum_state_t* state);

// Training operations
quantum_status_t prepare_test_gradients(quantum_state_t* gradients);
quantum_status_t quantum_backward_pass(quantum_llm_state_t* state,
                                     const quantum_state_t* gradients,
                                     quantum_state_t* parameter_gradients);
float measure_gradient_norm(const quantum_state_t* gradients);
quantum_status_t update_quantum_parameters(quantum_llm_state_t* state,
                                         const quantum_state_t* gradients);
float measure_parameter_update_fidelity(const quantum_llm_state_t* state);

// Error correction and stability
void prepare_noisy_quantum_state(quantum_state_t* state, float noise_level);
quantum_status_t apply_error_correction(void* distributed_system,
                                      quantum_state_t* state);
float measure_quantum_error_rate(const quantum_state_t* state);
float measure_quantum_stability(const quantum_state_t* state);

// Training data management
quantum_status_t prepare_test_training_data(training_data_t* data);
void prepare_quantum_input(quantum_state_t* state,
                         const training_data_t* data,
                         uint32_t index);
void cleanup_training_data(training_data_t* data);

// Forward and backward passes
quantum_status_t quantum_forward_pass(quantum_llm_state_t* state,
                                    const quantum_state_t* input,
                                    quantum_state_t* output);
quantum_status_t compute_quantum_loss(quantum_state_t* gradients,
                                    const quantum_state_t* output,
                                    const training_data_t* data,
                                    uint32_t index);
float measure_quantum_loss(const quantum_state_t* gradients);

// State management
void cleanup_quantum_state(quantum_state_t* state);

#endif // QUANTUM_LLM_CORE_H
