#ifndef DIFFERENTIAL_TRANSFORMER_H
#define DIFFERENTIAL_TRANSFORMER_H

#include <stdbool.h>
#include <stddef.h>
#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

// Transformer types
typedef enum {
    TRANSFORMER_CLASSICAL,    // Classical transformer
    TRANSFORMER_QUANTUM,      // Quantum transformer
    TRANSFORMER_HYBRID,       // Hybrid transformer
    TRANSFORMER_GEOMETRIC    // Geometric transformer
} transformer_type_t;

// Attention mechanisms
typedef enum {
    ATTENTION_STANDARD,      // Standard attention
    ATTENTION_QUANTUM,       // Quantum attention
    ATTENTION_GEOMETRIC,     // Geometric attention
    ATTENTION_ADAPTIVE      // Adaptive attention
} attention_type_t;

// Optimization modes
typedef enum {
    OPT_MODE_SGD,           // Stochastic gradient descent
    OPT_MODE_ADAM,          // Adam optimizer
    OPT_MODE_QUANTUM,       // Quantum optimizer
    OPT_MODE_HYBRID        // Hybrid optimizer
} optimization_mode_t;

// Layer normalization types
typedef enum {
    NORM_STANDARD,          // Standard layer normalization
    NORM_QUANTUM,           // Quantum normalization
    NORM_GEOMETRIC,         // Geometric normalization
    NORM_ADAPTIVE         // Adaptive normalization
} normalization_type_t;

// Transformer configuration
typedef struct {
    transformer_type_t type;         // Transformer type
    attention_type_t attention;      // Attention type
    optimization_mode_t opt_mode;    // Optimization mode
    normalization_type_t norm;       // Normalization type
    size_t num_layers;              // Number of layers
    size_t hidden_dim;              // Hidden dimension
    size_t num_heads;               // Number of attention heads
    double dropout_rate;            // Dropout rate
    bool use_bias;                  // Use bias terms
    bool use_residual;              // Use residual connections
} transformer_config_t;

// Model state
typedef struct {
    size_t seq_length;              // Sequence length
    size_t batch_size;              // Batch size
    size_t vocab_size;              // Vocabulary size
    size_t max_position;            // Maximum position
    double learning_rate;           // Learning rate
    double* weights;                // Model weights
    double* gradients;              // Weight gradients
    void* state_data;              // Additional state data
} model_state_t;

// Layer state
typedef struct {
    size_t layer_index;             // Layer index
    double* attention_weights;       // Attention weights
    double* ffn_weights;            // Feed-forward weights
    double* layer_norm_params;       // Layer normalization parameters
    double* attention_cache;         // Attention cache
    void* layer_data;              // Additional layer data
} layer_state_t;

// Training metrics
typedef struct {
    double loss;                    // Training loss
    double accuracy;                // Model accuracy
    double gradient_norm;           // Gradient norm
    double learning_rate;           // Current learning rate
    size_t iteration;              // Training iteration
    void* metric_data;             // Additional metrics
} training_metrics_t;

// Opaque transformer handle
typedef struct differential_transformer_t differential_transformer_t;

// Core functions
differential_transformer_t* create_transformer(const transformer_config_t* config);
void destroy_transformer(differential_transformer_t* transformer);

// Initialization functions
bool init_model_state(differential_transformer_t* transformer,
                     model_state_t* state);
bool init_layer_state(differential_transformer_t* transformer,
                     layer_state_t* state,
                     size_t layer_index);
bool validate_initialization(differential_transformer_t* transformer);

// Forward pass functions
bool forward_pass(differential_transformer_t* transformer,
                 const double* input,
                 double* output,
                 model_state_t* state);
bool attention_forward(differential_transformer_t* transformer,
                      const double* queries,
                      const double* keys,
                      const double* values,
                      double* output,
                      layer_state_t* state);
bool layer_norm_forward(differential_transformer_t* transformer,
                       const double* input,
                       double* output,
                       layer_state_t* state);

// Backward pass functions
bool backward_pass(differential_transformer_t* transformer,
                  const double* gradient,
                  double* input_gradient,
                  model_state_t* state);
bool attention_backward(differential_transformer_t* transformer,
                       const double* output_gradient,
                       double* query_gradient,
                       double* key_gradient,
                       double* value_gradient,
                       layer_state_t* state);
bool layer_norm_backward(differential_transformer_t* transformer,
                        const double* output_gradient,
                        double* input_gradient,
                        layer_state_t* state);

// Optimization functions
bool optimize_parameters(differential_transformer_t* transformer,
                       model_state_t* state,
                       const training_metrics_t* metrics);
bool update_learning_rate(differential_transformer_t* transformer,
                         model_state_t* state,
                         const training_metrics_t* metrics);
bool clip_gradients(differential_transformer_t* transformer,
                   model_state_t* state,
                   double max_norm);

// Quantum-specific functions
bool quantum_attention(differential_transformer_t* transformer,
                      const complex double* quantum_state,
                      complex double* output,
                      layer_state_t* state);
bool quantum_layer_norm(differential_transformer_t* transformer,
                       const complex double* input,
                       complex double* output,
                       layer_state_t* state);
bool quantum_optimization(differential_transformer_t* transformer,
                        model_state_t* state,
                        const training_metrics_t* metrics);

// Metal/GPU acceleration functions
bool metal_init_differential(void);
void metal_cleanup_differential(void);
bool metal_forward_pass(differential_transformer_t* transformer,
                       const double* input,
                       double* output,
                       model_state_t* state);
bool metal_backward_pass(differential_transformer_t* transformer,
                        const double* gradient,
                        double* input_gradient,
                        model_state_t* state);

// Utility functions
bool export_model(const differential_transformer_t* transformer,
                 const char* filename);
bool import_model(differential_transformer_t* transformer,
                 const char* filename);
void free_model_state(model_state_t* state);

#ifdef __cplusplus
}
#endif

#endif // DIFFERENTIAL_TRANSFORMER_H
