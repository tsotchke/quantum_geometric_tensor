/**
 * @file quantum_geometric_attention.h
 * @brief Quantum geometric attention mechanisms for quantum machine learning
 *
 * Implements attention mechanisms that leverage quantum geometric structures
 * including quantum self-attention, cross-attention, and geometric attention
 * using quantum circuits and tensor networks.
 */

#ifndef QUANTUM_GEOMETRIC_ATTENTION_H
#define QUANTUM_GEOMETRIC_ATTENTION_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct quantum_circuit;
struct quantum_state;
struct geometric_tensor;

// =============================================================================
// Attention Types and Configurations
// =============================================================================

/**
 * Types of quantum attention mechanisms
 */
typedef enum {
    ATTENTION_QUANTUM_SELF,          // Quantum self-attention
    ATTENTION_QUANTUM_CROSS,         // Quantum cross-attention
    ATTENTION_GEOMETRIC,             // Geometric attention using manifold structure
    ATTENTION_SPARSE_QUANTUM,        // Sparse quantum attention
    ATTENTION_LINEAR_QUANTUM,        // Linear complexity quantum attention
    ATTENTION_MULTI_HEAD_QUANTUM,    // Multi-head quantum attention
    ATTENTION_RELATIVE_QUANTUM,      // Relative position quantum attention
    ATTENTION_GRAPH_QUANTUM          // Graph-based quantum attention
} QuantumAttentionType;

/**
 * Attention score computation methods
 */
typedef enum {
    SCORE_DOT_PRODUCT,               // Standard dot product attention
    SCORE_SCALED_DOT_PRODUCT,        // Scaled dot product (/ sqrt(d_k))
    SCORE_ADDITIVE,                  // Additive attention (Bahdanau)
    SCORE_QUANTUM_FIDELITY,          // Quantum state fidelity
    SCORE_QUANTUM_KERNEL,            // Quantum kernel-based
    SCORE_GEODESIC,                  // Geodesic distance on manifold
    SCORE_RIEMANN_METRIC             // Riemannian metric-based
} AttentionScoreMethod;

/**
 * Normalization methods for attention weights
 */
typedef enum {
    NORMALIZE_SOFTMAX,               // Standard softmax
    NORMALIZE_SPARSEMAX,             // Sparsemax (sparse attention)
    NORMALIZE_QUANTUM_SOFTMAX,       // Quantum-enhanced softmax
    NORMALIZE_ENTMAX,                // Entmax (α-entmax)
    NORMALIZE_L2                     // L2 normalization
} AttentionNormalization;

/**
 * Positional encoding types
 */
typedef enum {
    POSITION_NONE,                   // No positional encoding
    POSITION_SINUSOIDAL,             // Sinusoidal (Transformer)
    POSITION_LEARNED,                // Learned embeddings
    POSITION_ROTARY,                 // Rotary position embedding (RoPE)
    POSITION_ALIBI,                  // ALiBi linear bias
    POSITION_QUANTUM_PHASE,          // Quantum phase encoding
    POSITION_GEOMETRIC               // Geometric (manifold-based)
} PositionalEncodingType;

// =============================================================================
// Core Structures
// =============================================================================

/**
 * Query-Key-Value projections
 */
typedef struct {
    double* query_weights;           // [hidden_dim, head_dim]
    double* key_weights;             // [hidden_dim, head_dim]
    double* value_weights;           // [hidden_dim, head_dim]
    double* output_weights;          // [head_dim, hidden_dim]
    double* query_bias;              // Optional bias [head_dim]
    double* key_bias;                // Optional bias [head_dim]
    double* value_bias;              // Optional bias [head_dim]
    double* output_bias;             // Optional bias [hidden_dim]
    size_t hidden_dim;
    size_t head_dim;
    bool use_bias;
} QKVProjection;

/**
 * Attention head configuration
 */
typedef struct {
    size_t head_dim;                 // Dimension per head
    AttentionScoreMethod score_method;
    AttentionNormalization normalization;
    double dropout_rate;
    double temperature;              // Softmax temperature
    bool use_causal_mask;            // Causal (autoregressive) masking
    bool use_relative_position;
    size_t max_relative_distance;
} AttentionHeadConfig;

/**
 * Multi-head attention configuration
 */
typedef struct {
    QuantumAttentionType type;
    size_t num_heads;
    size_t hidden_dim;
    size_t head_dim;                 // hidden_dim / num_heads typically
    AttentionHeadConfig head_config;
    PositionalEncodingType position_encoding;
    size_t max_sequence_length;
    bool use_layer_norm;
    bool use_pre_norm;               // Pre-normalization vs post-norm
    double attention_dropout;
    double residual_dropout;
} MultiHeadAttentionConfig;

/**
 * Quantum circuit configuration for attention
 */
typedef struct {
    size_t num_qubits;
    size_t num_layers;               // Variational layers
    size_t num_parameters;
    bool use_entanglement;
    bool use_data_reuploading;
    char* ansatz_type;               // "hardware_efficient", "QAOA", etc.
} QuantumCircuitConfig;

// =============================================================================
// Attention Layer Structures
// =============================================================================

/**
 * Single attention head
 */
typedef struct {
    AttentionHeadConfig config;
    QKVProjection projection;
    double* attention_scores;        // Cached scores [seq_len, seq_len]
    double* attention_weights;       // Normalized weights
    size_t current_seq_len;
    struct quantum_circuit* quantum_circuit;  // Optional quantum circuit
} AttentionHead;

/**
 * Multi-head attention layer
 */
typedef struct {
    MultiHeadAttentionConfig config;
    AttentionHead* heads;            // Array of attention heads
    double* output_projection;       // Final projection [num_heads * head_dim, hidden_dim]
    double* output_bias;
    double* layer_norm_gamma;
    double* layer_norm_beta;
    double* positional_encoding;     // Positional encoding table
    size_t num_parameters;
    bool is_initialized;
} MultiHeadAttention;

/**
 * Quantum geometric attention layer
 */
typedef struct {
    MultiHeadAttention* classical_attention;
    struct quantum_circuit* query_circuit;
    struct quantum_circuit* key_circuit;
    struct quantum_circuit* value_circuit;
    QuantumCircuitConfig circuit_config;
    double* quantum_parameters;
    size_t num_quantum_params;
    struct geometric_tensor* manifold;  // Geometric structure
    bool use_quantum_kernel;
    bool use_geometric_distance;
} QuantumGeometricAttention;

/**
 * Attention mask
 */
typedef struct {
    double* mask;                    // [seq_len, seq_len] or [batch, seq_len, seq_len]
    size_t* dimensions;
    size_t num_dimensions;
    bool is_causal;
    bool is_padding_mask;
} AttentionMask;

// =============================================================================
// Attention Output
// =============================================================================

/**
 * Attention computation result
 */
typedef struct {
    double* output;                  // [batch, seq_len, hidden_dim]
    double* attention_weights;       // [batch, num_heads, seq_len, seq_len]
    size_t batch_size;
    size_t seq_len;
    size_t hidden_dim;
    size_t num_heads;
    double compute_time_ms;
} AttentionOutput;

/**
 * Attention gradients for backpropagation
 */
typedef struct {
    double* d_query_weights;
    double* d_key_weights;
    double* d_value_weights;
    double* d_output_weights;
    double* d_query_bias;
    double* d_key_bias;
    double* d_value_bias;
    double* d_output_bias;
    double* d_layer_norm_gamma;
    double* d_layer_norm_beta;
    double* d_quantum_params;        // Quantum parameter gradients
    size_t num_parameters;
} AttentionGradients;

// =============================================================================
// Classical Attention Operations
// =============================================================================

/**
 * Create multi-head attention layer
 */
int attention_create(MultiHeadAttention** attention,
                     MultiHeadAttentionConfig* config);

/**
 * Destroy attention layer
 */
void attention_destroy(MultiHeadAttention* attention);

/**
 * Initialize attention weights
 */
int attention_init_weights(MultiHeadAttention* attention,
                           const char* init_method);  // "xavier", "he", "orthogonal"

/**
 * Forward pass through attention layer
 */
int attention_forward(MultiHeadAttention* attention,
                      double* query,           // [batch, seq_q, hidden_dim]
                      double* key,             // [batch, seq_k, hidden_dim]
                      double* value,           // [batch, seq_k, hidden_dim]
                      AttentionMask* mask,     // Optional attention mask
                      size_t batch_size,
                      size_t seq_len_q,
                      size_t seq_len_k,
                      AttentionOutput** output);

/**
 * Self-attention forward (query = key = value)
 */
int attention_self_forward(MultiHeadAttention* attention,
                           double* input,
                           AttentionMask* mask,
                           size_t batch_size,
                           size_t seq_len,
                           AttentionOutput** output);

/**
 * Backward pass through attention layer
 */
int attention_backward(MultiHeadAttention* attention,
                       double* d_output,        // Gradient from next layer
                       AttentionOutput* forward_output,
                       AttentionGradients** gradients);

/**
 * Update attention weights with gradients
 */
int attention_update_weights(MultiHeadAttention* attention,
                             AttentionGradients* gradients,
                             double learning_rate);

// =============================================================================
// Quantum Geometric Attention Operations
// =============================================================================

/**
 * Create quantum geometric attention layer
 */
int quantum_attention_create(QuantumGeometricAttention** attention,
                             MultiHeadAttentionConfig* classical_config,
                             QuantumCircuitConfig* quantum_config);

/**
 * Destroy quantum geometric attention
 */
void quantum_attention_destroy(QuantumGeometricAttention* attention);

/**
 * Initialize quantum circuits for attention
 */
int quantum_attention_init_circuits(QuantumGeometricAttention* attention);

/**
 * Forward pass through quantum attention
 */
int quantum_attention_forward(QuantumGeometricAttention* attention,
                              double* input,
                              AttentionMask* mask,
                              size_t batch_size,
                              size_t seq_len,
                              AttentionOutput** output);

/**
 * Compute quantum kernel attention scores
 */
int quantum_attention_kernel_scores(QuantumGeometricAttention* attention,
                                    double* queries,
                                    double* keys,
                                    size_t num_queries,
                                    size_t num_keys,
                                    double** scores_out);

/**
 * Compute attention using quantum fidelity
 */
int quantum_attention_fidelity_scores(QuantumGeometricAttention* attention,
                                      struct quantum_state** query_states,
                                      struct quantum_state** key_states,
                                      size_t num_queries,
                                      size_t num_keys,
                                      double** scores_out);

/**
 * Backward pass with parameter-shift rule for quantum gradients
 */
int quantum_attention_backward(QuantumGeometricAttention* attention,
                               double* d_output,
                               AttentionOutput* forward_output,
                               AttentionGradients** gradients);

// =============================================================================
// Geometric Attention Operations
// =============================================================================

/**
 * Compute geodesic attention scores on manifold
 */
int geometric_attention_geodesic(QuantumGeometricAttention* attention,
                                 double* queries,
                                 double* keys,
                                 size_t num_queries,
                                 size_t num_keys,
                                 double** scores_out);

/**
 * Compute attention using Riemannian metric
 */
int geometric_attention_riemannian(QuantumGeometricAttention* attention,
                                   double* queries,
                                   double* keys,
                                   double* metric_tensor,
                                   size_t num_queries,
                                   size_t num_keys,
                                   double** scores_out);

/**
 * Set geometric manifold for attention
 */
int geometric_attention_set_manifold(QuantumGeometricAttention* attention,
                                     struct geometric_tensor* manifold);

// =============================================================================
// Attention Mask Operations
// =============================================================================

/**
 * Create attention mask
 */
int attention_mask_create(AttentionMask** mask,
                          size_t* dimensions,
                          size_t num_dimensions);

/**
 * Destroy attention mask
 */
void attention_mask_destroy(AttentionMask* mask);

/**
 * Create causal (autoregressive) mask
 */
int attention_mask_causal(AttentionMask** mask,
                          size_t seq_len);

/**
 * Create padding mask from sequence lengths
 */
int attention_mask_padding(AttentionMask** mask,
                           size_t* sequence_lengths,
                           size_t batch_size,
                           size_t max_seq_len);

/**
 * Combine multiple masks
 */
int attention_mask_combine(AttentionMask* mask1,
                           AttentionMask* mask2,
                           AttentionMask** combined);

// =============================================================================
// Positional Encoding Operations
// =============================================================================

/**
 * Generate sinusoidal positional encoding
 */
int positional_encoding_sinusoidal(double** encoding_out,
                                   size_t max_seq_len,
                                   size_t hidden_dim);

/**
 * Generate rotary position embedding (RoPE)
 */
int positional_encoding_rotary(double** encoding_out,
                               size_t max_seq_len,
                               size_t head_dim);

/**
 * Apply rotary position embedding to queries and keys
 */
int positional_encoding_apply_rotary(double* queries,
                                     double* keys,
                                     double* encoding,
                                     size_t seq_len,
                                     size_t head_dim);

/**
 * Generate quantum phase positional encoding
 */
int positional_encoding_quantum_phase(double** encoding_out,
                                      size_t max_seq_len,
                                      size_t num_qubits);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Compute scaled dot-product attention
 */
int scaled_dot_product_attention(double* query,
                                 double* key,
                                 double* value,
                                 AttentionMask* mask,
                                 size_t batch_size,
                                 size_t num_heads,
                                 size_t seq_len_q,
                                 size_t seq_len_k,
                                 size_t head_dim,
                                 double** output);

/**
 * Apply softmax normalization
 */
int attention_softmax(double* scores,
                      size_t batch_size,
                      size_t num_heads,
                      size_t seq_len_q,
                      size_t seq_len_k,
                      double temperature);

/**
 * Apply dropout to attention weights
 */
int attention_dropout(double* weights,
                      size_t size,
                      double dropout_rate,
                      bool training);

/**
 * Free attention output
 */
void attention_output_free(AttentionOutput* output);

/**
 * Free attention gradients
 */
void attention_gradients_free(AttentionGradients* gradients);

/**
 * Get attention layer parameter count
 */
size_t attention_parameter_count(MultiHeadAttention* attention);

/**
 * Get quantum attention parameter count
 */
size_t quantum_attention_parameter_count(QuantumGeometricAttention* attention);

/**
 * Print attention configuration
 */
void attention_print_config(MultiHeadAttention* attention);

/**
 * Print quantum attention configuration
 */
void quantum_attention_print_config(QuantumGeometricAttention* attention);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_GEOMETRIC_ATTENTION_H
