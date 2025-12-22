/**
 * @file quantum_attention.h
 * @brief Quantum attention mechanism for transformer architectures
 *
 * Provides quantum-enhanced attention computation using quantum circuits
 * and sparse attention patterns for O(log n) complexity.
 */

#ifndef QUANTUM_ATTENTION_H
#define QUANTUM_ATTENTION_H

#include <stdbool.h>
#include <stddef.h>
#include <complex.h>
#include "quantum_geometric/core/quantum_complex.h"
#include "quantum_geometric/core/quantum_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct quantum_circuit_t;
struct quantum_system_t;
struct quantum_register_t;
struct QuantumWorkspace;
struct SparsityPattern;

// Quantum attention configuration
typedef struct quantum_attention_config {
    size_t num_heads;                // Number of attention heads
    size_t head_dim;                 // Dimension per head
    size_t hidden_dim;               // Total hidden dimension
    bool use_quantum;                // Enable quantum enhancement
    bool use_sparse;                 // Enable sparse attention
    bool use_causal_mask;            // Apply causal masking
    double dropout_rate;             // Dropout probability
    double temperature;              // Softmax temperature
    size_t max_sparse_patterns;      // Maximum sparsity patterns
} quantum_attention_config_t;

// Quantum attention head state
typedef struct quantum_attention_head {
    ComplexFloat* query_weights;     // Query projection weights
    ComplexFloat* key_weights;       // Key projection weights
    ComplexFloat* value_weights;     // Value projection weights
    ComplexFloat* output_weights;    // Output projection weights
    size_t head_dim;                 // Head dimension
    size_t hidden_dim;               // Hidden dimension
    double* cached_attention;        // Cached attention scores
    bool cache_valid;                // Cache validity flag
} quantum_attention_head_t;

// Quantum attention mechanism
typedef struct quantum_attention {
    quantum_attention_head_t** heads;       // Attention heads
    size_t num_heads;                        // Number of heads
    size_t head_dim;                         // Dimension per head
    size_t hidden_dim;                       // Total hidden dimension
    ComplexFloat* output_projection;         // Final output projection
    ComplexFloat* layer_norm_gamma;          // Layer norm scale
    ComplexFloat* layer_norm_beta;           // Layer norm bias
    quantum_attention_config_t config;       // Configuration

    // Quantum circuit components
    struct quantum_circuit_t* attention_circuit; // Pre-built attention circuit
    size_t* sparse_indices;                  // Sparse attention indices
    size_t num_sparse_indices;               // Number of sparse indices

    // Performance tracking
    double last_attention_time;              // Last operation time
    size_t total_operations;                 // Total operations count
    double average_sparsity;                 // Average sparsity ratio
} quantum_attention_t;

// Quantum estimation configuration for attention
typedef struct quantum_estimation_config {
    double precision;                // Estimation precision
    double success_probability;      // Target success probability
    bool use_quantum_memory;         // Use quantum memory
    int error_correction;            // Error correction level
    int optimization_level;          // Optimization level
} quantum_estimation_config_t;

// ============================================================================
// Core Functions
// ============================================================================

/**
 * @brief Initialize quantum attention mechanism
 * @param num_heads Number of attention heads
 * @param head_dim Dimension per head
 * @param config Attention configuration
 * @return Pointer to initialized attention or NULL on failure
 */
quantum_attention_t* init_quantum_attention(
    size_t num_heads,
    size_t head_dim,
    quantum_attention_config_t config);

/**
 * @brief Initialize quantum attention with full configuration
 * @param config Complete configuration structure
 * @return Pointer to initialized attention or NULL on failure
 */
quantum_attention_t* create_quantum_attention(
    const quantum_attention_config_t* config);

/**
 * @brief Destroy quantum attention and free resources
 * @param attention Attention to destroy
 */
void cleanup_quantum_attention(quantum_attention_t* attention);

/**
 * @brief Destroy quantum attention (alias)
 * @param attention Attention to destroy
 */
void destroy_quantum_attention(quantum_attention_t* attention);

// ============================================================================
// Forward Pass Functions
// ============================================================================

/**
 * @brief Compute quantum attention forward pass
 * @param attention Quantum attention mechanism
 * @param query Query tensor
 * @param key Key tensor
 * @param value Value tensor
 * @param output Output tensor
 * @param seq_length Sequence length
 * @param batch_size Batch size
 * @return true on success, false on failure
 */
bool quantum_attention_forward(
    quantum_attention_t* attention,
    const ComplexFloat* query,
    const ComplexFloat* key,
    const ComplexFloat* value,
    ComplexFloat* output,
    size_t seq_length,
    size_t batch_size);

/**
 * @brief Compute sparse quantum attention
 * @param attention Quantum attention mechanism
 * @param reg_input Input quantum register
 * @param reg_output Output quantum register
 * @param patterns Sparsity patterns
 * @param num_patterns Number of patterns
 * @param circuit Quantum circuit
 * @param system Quantum system
 * @param config Estimation configuration
 * @return true on success
 */
bool compute_quantum_attention_sparse(
    quantum_attention_t* attention,
    struct quantum_register_t* reg_input,
    struct quantum_register_t* reg_output,
    const struct SparsityPattern* patterns,
    size_t num_patterns,
    struct quantum_circuit_t* circuit,
    struct quantum_system_t* system,
    const quantum_estimation_config_t* config);

/**
 * @brief Compute quantum attention on register
 * @param attention Quantum attention mechanism
 * @param reg Input quantum register
 * @param circuit Quantum circuit
 * @param workspace Quantum workspace
 * @return Output quantum register
 */
struct quantum_register_t* quantum_compute_attention(
    quantum_attention_t* attention,
    struct quantum_register_t* reg,
    struct quantum_circuit_t* circuit,
    struct QuantumWorkspace* workspace);

// ============================================================================
// Attention Weight Functions
// ============================================================================

/**
 * @brief Compute attention weights
 * @param attention Quantum attention mechanism
 * @param query Query tensor
 * @param key Key tensor
 * @param weights Output weights
 * @param seq_length Sequence length
 * @return true on success
 */
bool compute_attention_weights_quantum(
    quantum_attention_t* attention,
    const ComplexFloat* query,
    const ComplexFloat* key,
    ComplexFloat* weights,
    size_t seq_length);

/**
 * @brief Apply softmax to attention weights
 * @param weights Attention weights to transform
 * @param seq_length Sequence length
 * @param temperature Softmax temperature
 */
void apply_attention_softmax(
    ComplexFloat* weights,
    size_t seq_length,
    double temperature);

/**
 * @brief Apply causal mask to attention weights
 * @param weights Attention weights
 * @param seq_length Sequence length
 */
void apply_causal_mask(
    ComplexFloat* weights,
    size_t seq_length);

// ============================================================================
// Quantum Enhancement Functions
// ============================================================================

/**
 * @brief Apply quantum amplitude encoding to attention
 * @param attention Quantum attention mechanism
 * @param input Input data
 * @param size Input size
 * @param circuit Output circuit
 * @return true on success
 */
bool quantum_encode_attention(
    quantum_attention_t* attention,
    const ComplexFloat* input,
    size_t size,
    struct quantum_circuit_t* circuit);

/**
 * @brief Apply quantum phase estimation for attention
 * @param attention Quantum attention mechanism
 * @param circuit Quantum circuit
 * @param system Quantum system
 * @param precision Estimation precision
 * @return Estimated phase value
 */
double quantum_attention_phase_estimation(
    quantum_attention_t* attention,
    struct quantum_circuit_t* circuit,
    struct quantum_system_t* system,
    double precision);

// ============================================================================
// Sparse Attention Functions
// ============================================================================

/**
 * @brief Build sparse attention pattern
 * @param attention Quantum attention mechanism
 * @param seq_length Sequence length
 * @param sparsity_ratio Target sparsity ratio
 * @return true on success
 */
bool build_sparse_pattern(
    quantum_attention_t* attention,
    size_t seq_length,
    double sparsity_ratio);

/**
 * @brief Get sparse attention indices
 * @param attention Quantum attention mechanism
 * @param indices Output indices array
 * @param num_indices Output number of indices
 * @return true on success
 */
bool get_sparse_indices(
    const quantum_attention_t* attention,
    size_t** indices,
    size_t* num_indices);

// ============================================================================
// Multi-Head Functions
// ============================================================================

/**
 * @brief Split input into multiple heads
 * @param input Input tensor
 * @param outputs Per-head output tensors
 * @param seq_length Sequence length
 * @param hidden_dim Hidden dimension
 * @param num_heads Number of heads
 */
void split_heads(
    const ComplexFloat* input,
    ComplexFloat** outputs,
    size_t seq_length,
    size_t hidden_dim,
    size_t num_heads);

/**
 * @brief Merge heads back into single tensor
 * @param inputs Per-head input tensors
 * @param output Merged output tensor
 * @param seq_length Sequence length
 * @param head_dim Head dimension
 * @param num_heads Number of heads
 */
void merge_heads(
    ComplexFloat** inputs,
    ComplexFloat* output,
    size_t seq_length,
    size_t head_dim,
    size_t num_heads);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Get attention statistics
 * @param attention Quantum attention mechanism
 * @param avg_attention_score Output average attention score
 * @param sparsity Output sparsity ratio
 * @param operations Output operation count
 */
void get_attention_stats(
    const quantum_attention_t* attention,
    double* avg_attention_score,
    double* sparsity,
    size_t* operations);

/**
 * @brief Reset attention caches
 * @param attention Quantum attention mechanism
 */
void reset_attention_cache(quantum_attention_t* attention);

/**
 * @brief Validate attention configuration
 * @param config Configuration to validate
 * @return true if valid
 */
bool validate_attention_config(const quantum_attention_config_t* config);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_ATTENTION_H
