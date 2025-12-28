/**
 * @file quantum_geometric_transformer.h
 * @brief Quantum-enhanced transformer architecture
 *
 * Implements quantum geometric transformer models combining classical
 * transformer architecture with quantum circuits and geometric structures
 * for enhanced representation learning.
 */

#ifndef QUANTUM_GEOMETRIC_TRANSFORMER_H
#define QUANTUM_GEOMETRIC_TRANSFORMER_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct quantum_circuit;
struct quantum_state;
struct geometric_tensor;
struct MultiHeadAttention;
struct QuantumGeometricAttention;

// =============================================================================
// Transformer Configuration Types
// =============================================================================

/**
 * Transformer model types
 */
typedef enum {
    TRANSFORMER_ENCODER_ONLY,        // BERT-style encoder
    TRANSFORMER_DECODER_ONLY,        // GPT-style decoder
    TRANSFORMER_ENCODER_DECODER,     // T5-style full transformer
    TRANSFORMER_QUANTUM_HYBRID,      // Hybrid quantum-classical
    TRANSFORMER_FULLY_QUANTUM        // Fully quantum transformer
} TransformerType;

/**
 * Feed-forward network types
 */
typedef enum {
    FFN_STANDARD,                    // Linear -> GELU -> Linear
    FFN_GATED,                       // Gated linear unit (GLU)
    FFN_SWIGLU,                      // SwiGLU activation
    FFN_GEGLU,                       // GeGLU activation
    FFN_QUANTUM                      // Quantum feed-forward
} FFNType;

/**
 * Activation functions
 */
typedef enum {
    ACTIVATION_RELU,
    ACTIVATION_GELU,
    ACTIVATION_SILU,                 // Swish
    ACTIVATION_TANH,
    ACTIVATION_SOFTMAX,
    ACTIVATION_QUANTUM               // Quantum activation
} ActivationType;

/**
 * Normalization types
 */
typedef enum {
    NORM_LAYER,                      // Layer normalization
    NORM_RMS,                        // RMS normalization
    NORM_BATCH,                      // Batch normalization
    NORM_GROUP,                      // Group normalization
    NORM_NONE                        // No normalization
} NormalizationType;

/**
 * Weight initialization methods
 */
typedef enum {
    INIT_XAVIER,                     // Xavier/Glorot
    INIT_HE,                         // He/Kaiming
    INIT_ORTHOGONAL,                 // Orthogonal
    INIT_NORMAL,                     // Normal distribution
    INIT_TRUNCATED_NORMAL,           // Truncated normal
    INIT_QUANTUM                     // Quantum random
} InitializationType;

// =============================================================================
// Core Configuration Structures
// =============================================================================

/**
 * Transformer layer configuration
 */
typedef struct {
    size_t hidden_dim;               // Hidden dimension
    size_t num_attention_heads;      // Number of attention heads
    size_t head_dim;                 // Dimension per head
    size_t ffn_dim;                  // Feed-forward intermediate dim
    FFNType ffn_type;
    ActivationType activation;
    NormalizationType normalization;
    double attention_dropout;
    double ffn_dropout;
    double residual_dropout;
    bool use_pre_norm;               // Pre-LN vs Post-LN
    bool use_bias;
    bool use_rotary_embedding;
    bool use_flash_attention;
} TransformerLayerConfig;

/**
 * Quantum layer configuration
 */
typedef struct {
    size_t num_qubits;
    size_t num_layers;               // Variational layers
    size_t num_parameters;
    bool use_entanglement;
    bool use_data_reuploading;
    char* ansatz_type;
    double noise_strength;           // For training robustness
} QuantumLayerConfig;

/**
 * Full transformer model configuration
 */
typedef struct {
    TransformerType type;
    size_t vocab_size;               // Vocabulary size
    size_t max_sequence_length;
    size_t num_encoder_layers;
    size_t num_decoder_layers;
    size_t hidden_dim;
    size_t num_attention_heads;
    size_t ffn_dim;
    TransformerLayerConfig layer_config;
    QuantumLayerConfig quantum_config;
    InitializationType weight_init;
    double weight_decay;
    bool tie_embeddings;             // Tie input/output embeddings
    bool use_quantum_attention;
    bool use_quantum_ffn;
    bool use_geometric_embeddings;
} TransformerConfig;

// =============================================================================
// Layer Structures
// =============================================================================

/**
 * Embedding layer
 */
typedef struct {
    double* token_embeddings;        // [vocab_size, hidden_dim]
    double* position_embeddings;     // [max_seq_len, hidden_dim]
    double* segment_embeddings;      // [num_segments, hidden_dim] (optional)
    size_t vocab_size;
    size_t max_seq_len;
    size_t hidden_dim;
    size_t num_segments;
    bool use_learned_positions;
    double* layer_norm_gamma;
    double* layer_norm_beta;
    double dropout_rate;
} EmbeddingLayer;

/**
 * Feed-forward network layer
 */
typedef struct {
    FFNType type;
    size_t input_dim;
    size_t intermediate_dim;
    size_t output_dim;
    double* weights_up;              // [input_dim, intermediate_dim]
    double* weights_down;            // [intermediate_dim, output_dim]
    double* weights_gate;            // For gated FFN [input_dim, intermediate_dim]
    double* bias_up;
    double* bias_down;
    double* bias_gate;
    ActivationType activation;
    double dropout_rate;
    struct quantum_circuit* quantum_circuit;  // For quantum FFN
} FeedForwardLayer;

// Forward declarations for tensor networks and attention types
// These match the actual typedef names used in the codebase
typedef struct quantum_attention quantum_attention_t;
typedef struct geometric_attention_t geometric_attention_t;
typedef struct tensor_network_t tensor_network_t;

/**
 * Layer configuration for quantum geometric transformer
 */
typedef struct {
    size_t hidden_dim;
    size_t num_heads;
    size_t head_dim;
    bool use_quantum;
    bool use_geometric;
    bool use_tensor_opt;
} LayerConfig;

/**
 * Transformer layer (encoder or decoder)
 */
typedef struct {
    // Configuration - both types supported
    TransformerLayerConfig config;
    LayerConfig layer_cfg;           // Simplified config for quantum ops

    // Self-attention (standard transformer)
    struct MultiHeadAttention* self_attention;
    struct QuantumGeometricAttention* quantum_self_attention;

    // Cross-attention (decoder only)
    struct MultiHeadAttention* cross_attention;
    struct QuantumGeometricAttention* quantum_cross_attention;

    // Quantum/Geometric attention components
    quantum_attention_t* quantum_attention;
    geometric_attention_t* geometric_attention;

    // Tensor networks for projections
    tensor_network_t* query_network;
    tensor_network_t* key_network;
    tensor_network_t* value_network;
    tensor_network_t* output_network;

    // Feed-forward (high-level)
    FeedForwardLayer* ffn;

    // Feed forward tensor networks
    tensor_network_t* ff_network1;
    tensor_network_t* ff_network2;

    // Feed forward weight matrices (direct access)
    double* ff1_weights;             // [hidden_dim x ff_dim]
    double* ff1_bias;
    double* ff2_weights;             // [ff_dim x hidden_dim]
    double* ff2_bias;
    size_t ff_dim;                   // Feed forward intermediate dimension

    // Layer normalization (standard)
    double* ln1_gamma;
    double* ln1_beta;
    double* ln2_gamma;
    double* ln2_beta;
    double* ln3_gamma;               // For cross-attention
    double* ln3_beta;

    // Layer normalization (direct weights for quantum ops)
    double* layer_norm_weights;
    double* layer_norm_bias;

    // Cached values for backprop
    double* attention_cache;
    double* ffn_cache;

    // State flags
    size_t hidden_dim;               // Convenience copy
    size_t layer_index;
    bool is_decoder_layer;
    bool weights_initialized;
    bool is_optimized;
} TransformerLayer;

/**
 * Transformer encoder
 */
typedef struct {
    TransformerLayer** layers;
    size_t num_layers;
    EmbeddingLayer* embeddings;
    double* final_ln_gamma;
    double* final_ln_beta;
    size_t hidden_dim;
    size_t max_seq_len;
} TransformerEncoder;

/**
 * Transformer decoder
 */
typedef struct {
    TransformerLayer** layers;
    size_t num_layers;
    EmbeddingLayer* embeddings;
    double* final_ln_gamma;
    double* final_ln_beta;
    double* lm_head;                 // [hidden_dim, vocab_size]
    size_t hidden_dim;
    size_t vocab_size;
    size_t max_seq_len;
    bool tie_embeddings;
} TransformerDecoder;

/**
 * Full transformer model
 */
typedef struct {
    TransformerConfig config;
    TransformerEncoder* encoder;
    TransformerDecoder* decoder;
    size_t total_parameters;
    bool is_training;

    // Quantum components
    struct quantum_circuit** quantum_circuits;
    size_t num_quantum_circuits;
    double* quantum_parameters;
    size_t num_quantum_params;

    // Geometric structure
    struct geometric_tensor* manifold;
} TransformerModel;

// =============================================================================
// Transformer Input/Output
// =============================================================================

/**
 * Transformer input batch
 */
typedef struct {
    size_t* input_ids;               // [batch_size, seq_len]
    double* attention_mask;          // [batch_size, seq_len]
    size_t* segment_ids;             // [batch_size, seq_len] (optional)
    size_t* position_ids;            // [batch_size, seq_len] (optional)
    size_t batch_size;
    size_t seq_len;
} TransformerInput;

/**
 * Transformer output
 */
typedef struct {
    double* hidden_states;           // [batch_size, seq_len, hidden_dim]
    double* logits;                  // [batch_size, seq_len, vocab_size] (for LM)
    double* attention_weights;       // [batch_size, num_layers, num_heads, seq_len, seq_len]
    double* pooled_output;           // [batch_size, hidden_dim] (for classification)
    size_t batch_size;
    size_t seq_len;
    size_t hidden_dim;
    size_t vocab_size;
    size_t num_layers;
    size_t num_heads;
} TransformerOutput;

/**
 * Transformer gradients
 */
typedef struct {
    double** layer_gradients;        // Gradients for each layer
    double* embedding_gradients;
    double* lm_head_gradients;
    double* quantum_gradients;
    size_t num_layers;
    size_t total_gradients;
} TransformerGradients;

// =============================================================================
// Model Creation and Destruction
// =============================================================================

/**
 * Create transformer model
 */
int transformer_create(TransformerModel** model, TransformerConfig* config);

/**
 * Destroy transformer model
 */
void transformer_destroy(TransformerModel* model);

/**
 * Initialize model weights
 */
int transformer_init_weights(TransformerModel* model, InitializationType init_type);

/**
 * Load model from file
 */
int transformer_load(TransformerModel** model, const char* path);

/**
 * Save model to file
 */
int transformer_save(TransformerModel* model, const char* path);

/**
 * Clone model
 */
int transformer_clone(TransformerModel* src, TransformerModel** dst);

// =============================================================================
// Forward Pass
// =============================================================================

/**
 * Full forward pass
 */
int transformer_forward(TransformerModel* model,
                        TransformerInput* input,
                        TransformerOutput** output);

/**
 * Encoder forward pass
 */
int transformer_encode(TransformerModel* model,
                       TransformerInput* input,
                       double** encoder_output,
                       size_t* output_seq_len);

/**
 * Decoder forward pass
 */
int transformer_decode(TransformerModel* model,
                       TransformerInput* input,
                       double* encoder_output,
                       size_t encoder_seq_len,
                       TransformerOutput** output);

/**
 * Single layer forward
 */
int transformer_layer_forward(TransformerLayer* layer,
                              double* input,
                              double* encoder_output,
                              double* attention_mask,
                              size_t batch_size,
                              size_t seq_len,
                              double** output);

// =============================================================================
// Backward Pass and Training
// =============================================================================

/**
 * Backward pass
 */
int transformer_backward(TransformerModel* model,
                         TransformerOutput* output,
                         double* d_loss,
                         TransformerGradients** gradients);

/**
 * Update model parameters
 */
int transformer_update(TransformerModel* model,
                       TransformerGradients* gradients,
                       double learning_rate);

/**
 * Set training mode
 */
void transformer_train_mode(TransformerModel* model, bool training);

/**
 * Compute loss
 */
int transformer_compute_loss(TransformerModel* model,
                             TransformerOutput* output,
                             size_t* target_ids,
                             size_t batch_size,
                             size_t seq_len,
                             double* loss_out);

// =============================================================================
// Generation
// =============================================================================

/**
 * Generation configuration
 */
typedef struct {
    size_t max_length;
    size_t min_length;
    double temperature;
    double top_p;                    // Nucleus sampling
    size_t top_k;                    // Top-k sampling
    double repetition_penalty;
    size_t num_beams;                // Beam search (1 = greedy)
    bool do_sample;
    bool early_stopping;
    size_t* eos_token_ids;
    size_t num_eos_tokens;
    size_t pad_token_id;
} GenerationConfig;

/**
 * Generate tokens autoregressively
 */
int transformer_generate(TransformerModel* model,
                         size_t* input_ids,
                         size_t input_len,
                         GenerationConfig* config,
                         size_t** output_ids,
                         size_t* output_len);

/**
 * Beam search generation
 */
int transformer_beam_search(TransformerModel* model,
                            size_t* input_ids,
                            size_t input_len,
                            GenerationConfig* config,
                            size_t** output_ids,
                            size_t* output_len);

// =============================================================================
// Quantum Operations
// =============================================================================

/**
 * Apply quantum layer
 */
int transformer_quantum_layer(TransformerModel* model,
                              double* input,
                              size_t batch_size,
                              size_t seq_len,
                              double** output);

/**
 * Compute quantum gradients
 */
int transformer_quantum_gradients(TransformerModel* model,
                                  double* d_output,
                                  double** d_quantum_params);

/**
 * Update quantum parameters
 */
int transformer_update_quantum(TransformerModel* model,
                               double* gradients,
                               double learning_rate);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Get model parameter count
 */
size_t transformer_parameter_count(TransformerModel* model);

/**
 * Get model memory usage
 */
size_t transformer_memory_usage(TransformerModel* model);

/**
 * Print model summary
 */
void transformer_print_summary(TransformerModel* model);

/**
 * Validate model configuration
 */
bool transformer_validate_config(TransformerConfig* config);

/**
 * Free transformer input
 */
void transformer_input_free(TransformerInput* input);

/**
 * Free transformer output
 */
void transformer_output_free(TransformerOutput* output);

/**
 * Free transformer gradients
 */
void transformer_gradients_free(TransformerGradients* gradients);

/**
 * Apply layer normalization
 */
int layer_normalize(double* input, double* gamma, double* beta,
                    size_t batch_size, size_t seq_len, size_t hidden_dim,
                    double eps, double** output);

/**
 * Apply RMS normalization
 */
int rms_normalize(double* input, double* gamma,
                  size_t batch_size, size_t seq_len, size_t hidden_dim,
                  double eps, double** output);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_GEOMETRIC_TRANSFORMER_H
