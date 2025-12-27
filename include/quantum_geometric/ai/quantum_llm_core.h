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
    QUANTUM_STATUS_COHERENCE_LOST,
    QUANTUM_STATUS_INSUFFICIENT_RESOURCES,
    QUANTUM_STATUS_INSUFFICIENT_MEMORY,
    QUANTUM_STATUS_INVALID_CONFIGURATION
} quantum_status_t;

// Model configuration
typedef struct {
    uint64_t total_parameters;    // Total number of model parameters
    uint32_t model_layers;        // Number of model layers
    uint32_t embedding_dimension; // Embedding dimension
    uint32_t attention_dimension; // Attention head dimension
    float learning_rate;          // Learning rate for parameter updates
} ModelConfig;

// Encoding configuration
typedef struct {
    uint32_t geometric_dimension;     // Geometric encoding dimension
    float compression_ratio;          // Current compression ratio
    float target_compression_ratio;   // Target compression ratio
    uint32_t encoding_qubits;         // Number of encoding qubits
    bool use_topological_protection;  // Enable topological protection
    bool use_holographic_encoding;    // Enable holographic encoding
    uint32_t holographic_dimension;   // Holographic encoding dimension
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
    uint32_t topology;               // Network topology type
    float learning_rate;             // Learning rate for optimization
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

// Type alias for backward compatibility
typedef QuantumLLMConfig quantum_llm_config_t;
// Note: distributed_config_t is defined in distributed_training_manager.h

// Forward declarations for internal types
struct quantum_distributed_system;
struct MemoryPool;
struct quantum_geometric_projector;
struct tensor_network;

typedef struct quantum_distributed_system quantum_distributed_system_t;
typedef struct MemoryPool memory_pool_t;
typedef struct quantum_geometric_projector quantum_geometric_projector_t;

// Opaque/external type declarations
typedef struct quantum_geometric_state_t quantum_geometric_state_t;
typedef struct quantum_state_t quantum_state_t;
typedef struct quantum_attention_t quantum_attention_t;

// Training data structure
typedef struct training_data_t {
    void* data_buffer;
    uint64_t buffer_size;
    uint32_t batch_size;
    uint32_t sequence_length;
    uint32_t feature_dimension;
    uint32_t num_samples;
    float* labels;
} training_data_t;

// Metrics structure for performance tracking
typedef struct quantum_llm_metrics_t {
    float loss;
    float accuracy;
    float throughput;
    float latency;
    uint64_t total_iterations;
    uint64_t total_parameters_updated;
    float encoding_fidelity;
    float compression_ratio;
    float operation_throughput;
    float communication_overhead;
    float error_rate;
    float memory_efficiency;
} quantum_llm_metrics_t;

// Full definition of quantum_llm_state_t
typedef struct quantum_llm_state_t {
    quantum_llm_config_t config;
    quantum_distributed_system_t* distributed_system;
    quantum_geometric_state_t** parameter_states;
    uint32_t num_parameter_states;
    float current_loss;
    memory_pool_t* memory_pool;
    struct tensor_network* attention_network;
    quantum_geometric_projector_t* projector;
} quantum_llm_state_t;

// ============================================================================
// Subsystem Initialization Functions
// ============================================================================

// Distributed system functions
quantum_status_t initialize_quantum_distributed_system(
    uint32_t num_nodes,
    uint32_t qubits_per_node,
    uint32_t topology,
    quantum_distributed_system_t** system);
void cleanup_quantum_distributed_system(quantum_distributed_system_t* system);

// Memory pool functions
quantum_status_t initialize_memory_pool(uint64_t size, memory_pool_t** pool);
void cleanup_memory_pool(memory_pool_t* pool);

// Tensor network functions
quantum_status_t initialize_tensor_network(
    uint32_t dimension,
    uint32_t num_heads,
    struct tensor_network** network);

// Projector functions
quantum_status_t initialize_quantum_geometric_projector(
    uint32_t dimension,
    uint32_t num_qubits,
    quantum_geometric_projector_t** projector);
void cleanup_quantum_geometric_projector(quantum_geometric_projector_t* projector);

// Tensor network cleanup
void cleanup_tensor_network(struct tensor_network* network);

// ============================================================================
// Main API Functions
// ============================================================================

// Initialization and cleanup
quantum_status_t initialize_quantum_llm(const QuantumLLMConfig* config,
                                      quantum_llm_state_t** state);
void cleanup_quantum_llm(quantum_llm_state_t* state);

// Quantum geometric encoding
quantum_status_t encode_quantum_parameters(quantum_llm_state_t* state,
                                         const float* parameters,
                                         uint64_t param_count,
                                         quantum_geometric_state_t* quantum_state);
float calculate_compression_ratio(const quantum_distributed_system_t* distributed_system);
float measure_encoding_fidelity(const quantum_geometric_state_t* state);

// Distributed quantum processing
quantum_status_t prepare_test_quantum_state(quantum_state_t* state);
quantum_status_t distribute_quantum_input(quantum_state_t* state,
                                        void* distributed_system);
float measure_node_synchronization(uint32_t node,
                                 const quantum_distributed_system_t* distributed_system);
quantum_status_t execute_parallel_quantum_operations(quantum_state_t* state);
float measure_operation_throughput(const quantum_distributed_system_t* distributed_system);

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
                                     void* aux_data);
float measure_gradient_norm(const quantum_state_t* gradients);
quantum_status_t update_quantum_parameters(quantum_llm_state_t* state,
                                         const quantum_state_t* gradients);
float measure_parameter_update_fidelity(const quantum_llm_state_t* state);

// Error correction and stability
quantum_status_t prepare_noisy_quantum_state(quantum_state_t* state, float noise_level);
quantum_status_t apply_error_correction(quantum_distributed_system_t* system,
                                      quantum_state_t* state);
float llm_measure_quantum_error_rate(const quantum_state_t* state);  // llm_ prefix to avoid conflict
float measure_quantum_stability(const quantum_state_t* state);

// Training data management
quantum_status_t prepare_test_training_data(training_data_t* data);
quantum_status_t prepare_quantum_input(quantum_state_t* state,
                                      const training_data_t* data,
                                      uint32_t batch_index);
void cleanup_training_data(training_data_t* data);

// Forward and backward passes
quantum_status_t quantum_forward_pass(quantum_llm_state_t* state,
                                    const quantum_state_t* input,
                                    quantum_state_t* output);
quantum_status_t compute_quantum_loss(const quantum_state_t* output_state,
                                    const quantum_state_t* target_state,
                                    quantum_state_t* gradients,
                                    float* loss);
float measure_quantum_loss(const quantum_state_t* gradients);

// State management
void cleanup_llm_quantum_state(quantum_state_t* state);

// ============================================================================
// Internal Helper Functions (implementation-specific)
// ============================================================================

// Quantum geometric encoding/decoding
quantum_status_t encode_quantum_geometric_state(
    quantum_geometric_projector_t* projector,
    const float* parameters,
    uint64_t param_count,
    float compression_ratio,
    quantum_geometric_state_t* state);

quantum_status_t decode_quantum_geometric_state(
    const quantum_geometric_state_t* state,
    float* parameters,
    uint64_t param_count);

// Attention operations
quantum_status_t apply_quantum_attention(
    struct tensor_network* network,
    const quantum_state_t* input,
    quantum_geometric_state_t** parameter_states,
    uint32_t num_layers,
    quantum_state_t* output);

// Loss and gradient operations
quantum_status_t compute_quantum_geometric_loss(
    const quantum_state_t* output,
    const quantum_state_t* target,
    quantum_state_t* gradients,
    float* loss);

quantum_status_t propagate_quantum_gradients(
    struct tensor_network* network,
    const quantum_state_t* gradients,
    quantum_geometric_state_t** parameter_states,
    uint32_t num_states,
    void* aux_data);

quantum_status_t update_quantum_geometric_parameters(
    quantum_geometric_projector_t* projector,
    quantum_geometric_state_t** parameter_states,
    uint32_t num_states,
    const quantum_state_t* gradients,
    float learning_rate);

// Input preparation
quantum_status_t prepare_quantum_geometric_input(
    void* data_buffer,
    uint32_t feature_dim,
    uint32_t batch_index,
    quantum_state_t* state);

// Test utilities
quantum_status_t initialize_test_quantum_state(quantum_state_t* state);
quantum_status_t add_quantum_noise(quantum_state_t* state, float noise_level);
void cleanup_quantum_geometric_state(quantum_state_t* state);

// Training data utilities
quantum_status_t initialize_test_training_data(training_data_t* data);
quantum_status_t load_quantum_training_batch(uint32_t batch_index, training_data_t* data);

// Error correction
quantum_status_t apply_quantum_error_correction(
    quantum_distributed_system_t* system,
    quantum_state_t* state);

// Metrics collection
quantum_status_t collect_quantum_llm_metrics(
    const quantum_llm_state_t* state,
    quantum_llm_metrics_t* metrics);

// Memory efficiency measurement
float measure_memory_efficiency(const memory_pool_t* pool);
float calculate_memory_efficiency(const memory_pool_t* pool);

// Communication overhead
float measure_communication_overhead(const quantum_distributed_system_t* system);

// Calculation helpers (internal)
float calculate_quantum_fidelity(const quantum_geometric_state_t* state);
float calculate_quantum_error_rate(const quantum_state_t* state);
float calculate_quantum_stability(const quantum_state_t* state);
float calculate_gate_fidelity(const quantum_state_t* state);
float calculate_attention_quality(const quantum_attention_t* attention);
float calculate_node_synchronization(uint32_t node_index, const quantum_distributed_system_t* system);
float calculate_operation_throughput(const quantum_distributed_system_t* system);
float get_compression_ratio(const quantum_distributed_system_t* system);

// Checkpoint operations
quantum_status_t save_quantum_checkpoint(const quantum_llm_state_t* state, const char* filename);
quantum_status_t load_quantum_checkpoint(quantum_llm_state_t* state, const char* filename);
quantum_status_t save_quantum_state_checkpoint(const quantum_llm_state_t* state, const char* filename);
quantum_status_t load_quantum_state_checkpoint(quantum_llm_state_t* state, const char* filename);

#endif // QUANTUM_LLM_CORE_H
