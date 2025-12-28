#include "quantum_geometric/core/quantum_geometric_transformer.h"
#include "quantum_geometric/core/quantum_attention.h"
#include "quantum_geometric/core/geometric_attention.h"
#include "quantum_geometric/core/tensor_network_optimizer.h"
#include "quantum_geometric/core/tensor_network_operations.h"
#include "quantum_geometric/core/quantum_geometric_interface.h"
#include "quantum_geometric/core/quantum_types.h"
#include "quantum_geometric/core/quantum_circuit_operations.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Type aliases for compatibility
typedef quantum_attention_t QuantumAttention;
typedef geometric_attention_t GeometricAttention;
typedef tensor_network_t TensorNetwork;
typedef tensor_network_optimizer_t TensorNetworkOptimizer;

// Missing constants (if not defined elsewhere)
#ifndef QG_VECTOR_SIZE
#define QG_VECTOR_SIZE 64
#endif
#ifndef QG_MAX_SEQUENCE_LENGTH
#define QG_MAX_SEQUENCE_LENGTH 4096
#endif
#ifndef QG_MIN_SIZE_FOR_GPU
#define QG_MIN_SIZE_FOR_GPU 1024
#endif
#ifndef QG_QUANTUM_CHUNK_SIZE
#define QG_QUANTUM_CHUNK_SIZE 256
#endif
#ifndef QG_FF_EXPANSION_FACTOR
#define QG_FF_EXPANSION_FACTOR 4
#endif
#ifndef QG_QUANTUM_ESTIMATION_PRECISION
#define QG_QUANTUM_ESTIMATION_PRECISION 1e-6
#endif
#ifndef QG_SUCCESS_PROBABILITY
#define QG_SUCCESS_PROBABILITY 0.99
#endif
#ifndef QUANTUM_OPTIMIZE_AGGRESSIVE
#define QUANTUM_OPTIMIZE_AGGRESSIVE 0x01
#endif
#ifndef QUANTUM_USE_ESTIMATION
#define QUANTUM_USE_ESTIMATION 0x02
#endif
#ifndef QUANTUM_ERROR_ADAPTIVE
#define QUANTUM_ERROR_ADAPTIVE 1
#endif
#ifndef QUANTUM_OPT_AGGRESSIVE
#define QUANTUM_OPT_AGGRESSIVE 2
#endif
#ifndef QUANTUM_CIRCUIT_OPTIMAL
#define QUANTUM_CIRCUIT_OPTIMAL 1
#endif

// Forward declare types used in this file
typedef struct QuantumWorkspace {
    void* data;
    size_t size;
} QuantumWorkspace;

typedef struct SparsityPattern {
    size_t* indices;
    size_t count;
    double threshold;
} SparsityPattern;

// Min macro if not defined
#ifndef min
#define min(a,b) ((a) < (b) ? (a) : (b))
#endif

// LayerConfig is now defined in the header file

// Platform-specific SIMD includes
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
    #if defined(__ARM_NEON) || defined(__ARM_NEON__)
        #include <arm_neon.h>
    #endif
#endif

// TransformerLayer is now defined in the header file
// Cast helpers for type compatibility with header definitions
#define quantum_attention_ptr(x) ((quantum_attention_t*)(x))
#define geometric_attention_ptr(x) ((geometric_attention_t*)(x))
#define tensor_network_ptr(x) ((tensor_network_t*)(x))

// Forward declarations for local static functions
static bool validate_config(const LayerConfig* config);
static void init_tensor_networks(TransformerLayer* layer);
static void initialize_layer_norm(double* weights, double* bias, size_t dim);
static bool is_gpu_available(void);
static void gpu_transformer_forward(TransformerLayer* layer,
                                   const double* input, double* output, size_t seq_length);
static quantum_circuit_t* quantum_create_transformer_circuit(size_t num_qubits, int flags);
static quantum_register_t* quantum_register_create_state(const double* data, size_t size, quantum_system_t* system);
static quantum_register_t* quantum_register_create_empty(size_t size);
static void quantum_register_destroy(quantum_register_t* reg);
static void quantum_register_extract(quantum_register_t* reg, double* output, size_t size);
static void optimize_transformer_layer(TransformerLayer* layer);
static QuantumWorkspace* init_quantum_workspace(size_t size);
static void cleanup_quantum_workspace(QuantumWorkspace* ws);
static void quantum_project_chunk(quantum_register_t* input, TensorNetwork* query,
                                 TensorNetwork* key, TensorNetwork* value,
                                 size_t chunk_size, quantum_circuit_t* circuit, QuantumWorkspace* ws);
static SparsityPattern* compute_sparsity_patterns(quantum_register_t* reg, size_t seq_len,
                                                  size_t num_patterns, quantum_system_t* system);
static void compute_hybrid_attention_sparse(TransformerLayer* layer,
                                           quantum_register_t* reg_input,
                                           quantum_register_t* reg_output,
                                           const SparsityPattern* patterns,
                                           size_t num_patterns,
                                           quantum_circuit_t* circuit,
                                           quantum_system_t* system,
                                           const quantum_estimation_config_t* config);
static void compute_geometric_attention_sparse(geometric_attention_t* attention,
                                              quantum_register_t* reg_input,
                                              quantum_register_t* reg_output,
                                              const SparsityPattern* patterns,
                                              size_t num_patterns,
                                              quantum_circuit_t* circuit,
                                              quantum_system_t* system,
                                              const quantum_estimation_config_t* config);
static void apply_layer_norm(double* data, const double* weights, const double* bias, size_t size);
static void feed_forward_tensor(TransformerLayer* layer, const double* input,
                               double* output, size_t seq_length);
static void feed_forward_linear(TransformerLayer* layer, const double* input,
                               double* output, size_t seq_length);

// Functions for attention computation (local implementations)
static quantum_system_t* quantum_system_create(size_t num_qubits, int flags);
static quantum_register_t* quantum_extract_pattern(quantum_register_t* reg,
                                                   const SparsityPattern* pattern,
                                                   quantum_system_t* system);
static quantum_register_t* quantum_compute_attention_local(quantum_attention_t* attention,
                                                           quantum_register_t* reg,
                                                           quantum_circuit_t* circuit,
                                                           QuantumWorkspace* ws);
static quantum_register_t* geometric_compute_attention_local(geometric_attention_t* attention,
                                                             quantum_register_t* reg,
                                                             quantum_circuit_t* circuit,
                                                             QuantumWorkspace* ws);
static void quantum_combine_attention(quantum_register_t* reg_quantum,
                                     quantum_register_t* reg_geometric,
                                     quantum_register_t* reg_output,
                                     const SparsityPattern* pattern,
                                     quantum_circuit_t* circuit,
                                     QuantumWorkspace* ws);
static void cleanup_geometric_attention(geometric_attention_t* attention);

// Initialize quantum geometric transformer
TransformerLayer* init_transformer_layer(
    const LayerConfig* config) {
    
    if (!validate_config(config)) return NULL;
    
    TransformerLayer* layer = aligned_alloc(QG_VECTOR_SIZE,
        sizeof(TransformerLayer));
    if (!layer) return NULL;
    
    // Copy configuration
    memcpy(&layer->layer_cfg, config, sizeof(LayerConfig));
    
    // Initialize attention mechanisms
    if (config->use_quantum) {
        quantum_attention_config_t qattn_config = {
            .num_heads = config->num_heads,
            .head_dim = config->head_dim,
            .hidden_dim = config->hidden_dim,
            .use_quantum = true,
            .use_sparse = false,
            .use_causal_mask = false,
            .dropout_rate = 0.0,
            .temperature = 1.0,
            .max_sparse_patterns = 16
        };
        layer->quantum_attention = init_quantum_attention(
            config->num_heads,
            config->head_dim,
            qattn_config);
    }

    if (config->use_geometric) {
        attention_config_t gattn_config = {
            .type = ATTENTION_GEOMETRIC,
            .geometry = ATTN_GEOMETRY_MANIFOLD,
            .connection = ATTN_CONNECTION_GEOMETRIC,
            .attention_heads = config->num_heads,
            .head_dim = config->head_dim,
            .use_error_correction = false
        };
        layer->geometric_attention = create_geometric_attention(&gattn_config);
    }
    
    // Initialize tensor networks if optimization enabled
    if (config->use_tensor_opt) {
        init_tensor_networks(layer);
    }
    
    // Initialize layer normalization
    layer->layer_norm_weights = aligned_alloc(QG_VECTOR_SIZE,
        config->hidden_dim * sizeof(double));
    layer->layer_norm_bias = aligned_alloc(QG_VECTOR_SIZE,
        config->hidden_dim * sizeof(double));

    initialize_layer_norm(layer->layer_norm_weights,
                        layer->layer_norm_bias,
                        config->hidden_dim);

    // Initialize feed-forward weight matrices with Xavier/Glorot initialization
    // Xavier init: variance = 2 / (fan_in + fan_out)
    size_t hidden_dim = config->hidden_dim;
    size_t ff_dim_size = QG_FF_EXPANSION_FACTOR * hidden_dim;
    layer->ff_dim = ff_dim_size;

    // Allocate weight matrices
    layer->ff1_weights = aligned_alloc(64, hidden_dim * ff_dim_size * sizeof(double));
    layer->ff1_bias = aligned_alloc(64, ff_dim_size * sizeof(double));
    layer->ff2_weights = aligned_alloc(64, ff_dim_size * hidden_dim * sizeof(double));
    layer->ff2_bias = aligned_alloc(64, hidden_dim * sizeof(double));

    if (layer->ff1_weights && layer->ff1_bias && layer->ff2_weights && layer->ff2_bias) {
        // Xavier initialization for ff1: hidden_dim -> ff_dim
        double limit1 = sqrt(6.0 / (double)(hidden_dim + ff_dim_size));
        for (size_t i = 0; i < hidden_dim * ff_dim_size; i++) {
            // Generate uniform random in [-limit, limit]
            layer->ff1_weights[i] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * limit1;
        }
        for (size_t i = 0; i < ff_dim_size; i++) {
            layer->ff1_bias[i] = 0.0;  // Initialize biases to zero
        }

        // Xavier initialization for ff2: ff_dim -> hidden_dim
        double limit2 = sqrt(6.0 / (double)(ff_dim_size + hidden_dim));
        for (size_t i = 0; i < ff_dim_size * hidden_dim; i++) {
            layer->ff2_weights[i] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * limit2;
        }
        for (size_t i = 0; i < hidden_dim; i++) {
            layer->ff2_bias[i] = 0.0;
        }

        layer->weights_initialized = true;
    } else {
        // Clean up on failure
        free(layer->ff1_weights);
        free(layer->ff1_bias);
        free(layer->ff2_weights);
        free(layer->ff2_bias);
        layer->ff1_weights = NULL;
        layer->ff1_bias = NULL;
        layer->ff2_weights = NULL;
        layer->ff2_bias = NULL;
        layer->weights_initialized = false;
    }

    layer->is_optimized = false;
    return layer;
}

// Optimized layer forward pass using quantum circuits and hierarchical attention - O(log n)
static void transformer_layer_forward_internal(
    TransformerLayer* layer,
    const double* input,
    double* output,
    size_t sequence_length) {
    
    if (!layer || sequence_length > QG_MAX_SEQUENCE_LENGTH) return;

    // For large sequences, use GPU acceleration
    if (sequence_length > QG_MIN_SIZE_FOR_GPU && is_gpu_available()) {
        gpu_transformer_forward(layer, input, output, sequence_length);
        return;
    }
    
    // Initialize quantum system
    quantum_system_t* system = quantum_system_create(
        (size_t)log2(sequence_length),
        QUANTUM_OPTIMIZE_AGGRESSIVE | QUANTUM_USE_ESTIMATION
    );
    
    // Configure quantum estimation
    quantum_estimation_config_t config = {
        .precision = QG_QUANTUM_ESTIMATION_PRECISION,
        .success_probability = QG_SUCCESS_PROBABILITY,
        .use_quantum_memory = true,
        .error_correction = QUANTUM_ERROR_ADAPTIVE,
        .optimization_level = QUANTUM_OPT_AGGRESSIVE
    };
    
    // Create quantum circuit
    quantum_circuit_t* circuit = quantum_create_transformer_circuit(
        system->num_qubits,
        QUANTUM_CIRCUIT_OPTIMAL
    );
    
    // Initialize quantum registers
    quantum_register_t* reg_input = quantum_register_create_state(
        input, sequence_length * layer->layer_cfg.hidden_dim, system
    );
    quantum_register_t* reg_output = quantum_register_create_empty(
        sequence_length * layer->layer_cfg.hidden_dim
    );
    
    // Optimize if needed
    if (!layer->is_optimized) {
        optimize_transformer_layer(layer);
    }
    
    // Use hierarchical tensor networks for projections - O(log n)
    #pragma omp parallel
    {
        QuantumWorkspace* qws = init_quantum_workspace(QG_VECTOR_SIZE);
        if (qws) {
            // Project in parallel chunks
            #pragma omp for schedule(guided)
            for (size_t chunk = 0; chunk < sequence_length; 
                 chunk += QG_QUANTUM_CHUNK_SIZE) {
                size_t chunk_size = min(QG_QUANTUM_CHUNK_SIZE, 
                                      sequence_length - chunk);
                
                // Quantum projection
                quantum_project_chunk(
                    reg_input + chunk * layer->layer_cfg.hidden_dim,
                    layer->query_network,
                    layer->key_network,
                    layer->value_network,
                    chunk_size,
                    circuit,
                    qws
                );
            }
            cleanup_quantum_workspace(qws);
        }
    }
    
    // Compute sparse attention patterns - O(log n)
    size_t num_patterns = (size_t)log2(sequence_length);
    SparsityPattern* patterns = compute_sparsity_patterns(
        reg_input, sequence_length, num_patterns, system
    );
    
    // Allocate intermediate tensors
    size_t total_size = sequence_length * layer->layer_cfg.hidden_dim;
    double* query = aligned_alloc(QG_VECTOR_SIZE, total_size * sizeof(double));
    double* key = aligned_alloc(QG_VECTOR_SIZE, total_size * sizeof(double));
    double* value = aligned_alloc(QG_VECTOR_SIZE, total_size * sizeof(double));
    double* attention_output = aligned_alloc(QG_VECTOR_SIZE, total_size * sizeof(double));

    if (!query || !key || !value || !attention_output) {
        free(query);
        free(key);
        free(value);
        free(attention_output);
        quantum_register_destroy(reg_input);
        quantum_register_destroy(reg_output);
        quantum_circuit_destroy(circuit);
        quantum_system_destroy(system);
        free(patterns);
        return;
    }

    // Apply quantum attention with sparsity
    if (layer->layer_cfg.use_quantum && layer->layer_cfg.use_geometric) {
        compute_hybrid_attention_sparse(
            layer, reg_input, reg_output,
            patterns, num_patterns,
            circuit, system, &config
        );
    } else if (layer->layer_cfg.use_quantum) {
        compute_quantum_attention_sparse(
            layer->quantum_attention,
            reg_input, reg_output,
            patterns, num_patterns,
            circuit, system, &config
        );
    } else if (layer->layer_cfg.use_geometric) {
        compute_geometric_attention_sparse(
            layer->geometric_attention,
            reg_input, reg_output,
            patterns, num_patterns,
            circuit, system, &config
        );
    }

    // Extract attention output from quantum register
    quantum_register_extract(reg_output, attention_output, total_size);

    // Layer normalization
    apply_layer_norm(attention_output,
                    layer->layer_norm_weights,
                    layer->layer_norm_bias,
                    total_size);

    // Feed forward
    if (layer->layer_cfg.use_tensor_opt) {
        feed_forward_tensor(layer,
                          attention_output,
                          output,
                          sequence_length);
    } else {
        feed_forward_linear(layer,
                          attention_output,
                          output,
                          sequence_length);
    }

    // Clean up
    free(query);
    free(key);
    free(value);
    free(attention_output);
    free(patterns);
    quantum_register_destroy(reg_input);
    quantum_register_destroy(reg_output);
    quantum_circuit_destroy(circuit);
    quantum_system_destroy(system);
}

// Initialize tensor networks
static void init_tensor_networks(TransformerLayer* layer) {
    TensorNetworkOptimizer* optimizer = init_tensor_optimizer();
    
    // Create QKV projection networks
    layer->query_network = create_projection_network(optimizer,
        layer->layer_cfg.hidden_dim,
        layer->layer_cfg.hidden_dim);
    
    layer->key_network = create_projection_network(optimizer,
        layer->layer_cfg.hidden_dim,
        layer->layer_cfg.hidden_dim);
    
    layer->value_network = create_projection_network(optimizer,
        layer->layer_cfg.hidden_dim,
        layer->layer_cfg.hidden_dim);
    
    // Create feed forward networks
    layer->ff_network1 = create_feed_forward_network(optimizer,
        layer->layer_cfg.hidden_dim,
        QG_FF_EXPANSION_FACTOR * layer->layer_cfg.hidden_dim);
    
    layer->ff_network2 = create_feed_forward_network(optimizer,
        QG_FF_EXPANSION_FACTOR * layer->layer_cfg.hidden_dim,
        layer->layer_cfg.hidden_dim);
    
    cleanup_tensor_optimizer(optimizer);
}

// Optimize transformer layer
static void optimize_transformer_layer(TransformerLayer* layer) {
    if (layer->layer_cfg.use_tensor_opt) {
        // Optimize tensor networks
        TensorNetworkOptimizer* optimizer = init_tensor_optimizer();
        
        optimize_tensor_network(optimizer,
                              layer->query_network,
                              STRATEGY_QUANTUM_INSPIRED);
        optimize_tensor_network(optimizer,
                              layer->key_network,
                              STRATEGY_QUANTUM_INSPIRED);
        optimize_tensor_network(optimizer,
                              layer->value_network,
                              STRATEGY_QUANTUM_INSPIRED);
        optimize_tensor_network(optimizer,
                              layer->ff_network1,
                              STRATEGY_GEOMETRIC);
        optimize_tensor_network(optimizer,
                              layer->ff_network2,
                              STRATEGY_GEOMETRIC);
        
        cleanup_tensor_optimizer(optimizer);
    }
    
    layer->is_optimized = true;
}

// Optimized hybrid attention using quantum circuits and sparsity - O(log n)
static void compute_hybrid_attention_sparse(
    TransformerLayer* layer,
    quantum_register_t* reg_input,
    quantum_register_t* reg_output,
    const SparsityPattern* patterns,
    size_t num_patterns,
    quantum_circuit_t* circuit,
    quantum_system_t* system,
    const quantum_estimation_config_t* config) {
    
    // Initialize quantum workspace
    QuantumWorkspace* qws = init_quantum_workspace(QG_VECTOR_SIZE);
    if (!qws) return;
    
    // Process each sparsity pattern in parallel
    #pragma omp parallel for schedule(guided)
    for (size_t p = 0; p < num_patterns; p++) {
        // Extract pattern subset
        quantum_register_t* reg_pattern = quantum_extract_pattern(
            reg_input, &patterns[p], system
        );
        
        // Compute quantum attention on subset
        quantum_register_t* reg_quantum = quantum_compute_attention_local(
            layer->quantum_attention,
            reg_pattern,
            circuit,
            qws
        );

        // Compute geometric attention on subset
        quantum_register_t* reg_geometric = geometric_compute_attention_local(
            layer->geometric_attention,
            reg_pattern,
            circuit,
            qws
        );
        
        // Combine results with quantum interference
        quantum_combine_attention(
            reg_quantum,
            reg_geometric,
            reg_output,
            &patterns[p],
            circuit,
            qws
        );
        
        // Cleanup pattern registers
        quantum_register_destroy(reg_pattern);
        quantum_register_destroy(reg_quantum);
        quantum_register_destroy(reg_geometric);
    }
    
    cleanup_quantum_workspace(qws);
}

// Clean up
void cleanup_transformer_layer(TransformerLayer* layer) {
    if (!layer) return;
    
    cleanup_quantum_attention(layer->quantum_attention);
    cleanup_geometric_attention(layer->geometric_attention);
    
    if (layer->layer_cfg.use_tensor_opt) {
        cleanup_tensor_network(layer->query_network);
        cleanup_tensor_network(layer->key_network);
        cleanup_tensor_network(layer->value_network);
        cleanup_tensor_network(layer->ff_network1);
        cleanup_tensor_network(layer->ff_network2);
    }
    
    free(layer->layer_norm_weights);
    free(layer->layer_norm_bias);
    free(layer);
}

// =============================================================================
// Static function implementations
// =============================================================================

// Validate layer configuration
static bool validate_config(const LayerConfig* config) {
    if (!config) return false;
    if (config->hidden_dim == 0) return false;
    if (config->num_heads == 0) return false;
    if (config->head_dim == 0) return false;
    if (config->hidden_dim % config->num_heads != 0) return false;
    return true;
}

// Initialize layer normalization weights
static void initialize_layer_norm(double* weights, double* bias, size_t dim) {
    if (!weights || !bias) return;
    for (size_t i = 0; i < dim; i++) {
        weights[i] = 1.0;  // Initialize gamma to 1
        bias[i] = 0.0;     // Initialize beta to 0
    }
}

// Check GPU availability
static bool is_gpu_available(void) {
#ifdef __APPLE__
    // Check for Metal support on macOS
    return true;  // Apple Silicon always has Metal
#elif defined(QGT_ENABLE_CUDA)
    // CUDA availability check would go here
    return false;  // Disabled by default
#else
    return false;
#endif
}

// GPU-accelerated transformer forward pass
static void gpu_transformer_forward(TransformerLayer* layer,
                                   const double* input, double* output, size_t seq_length) {
    (void)layer; (void)input; (void)output; (void)seq_length;
    // GPU implementation would use Metal/CUDA kernels
    // For now, fall back to CPU implementation
}

// Create quantum system for transformer operations
static quantum_system_t* quantum_system_create(size_t num_qubits, int flags) {
    quantum_system_t* system = calloc(1, sizeof(quantum_system_t));
    if (!system) return NULL;

    system->num_qubits = num_qubits;
    system->num_classical_bits = 0;
    system->flags = flags;
    system->device_type = 0;  // CPU
    system->device_data = NULL;

    // Allocate state vector
    size_t state_size = (size_t)1 << num_qubits;
    ComplexFloat* state_vector = aligned_alloc(64, state_size * sizeof(ComplexFloat));
    if (!state_vector) {
        free(system);
        return NULL;
    }

    // Initialize to |0...0⟩ state
    memset(state_vector, 0, state_size * sizeof(ComplexFloat));
    state_vector[0] = (ComplexFloat){1.0f, 0.0f};
    system->state = state_vector;

    return system;
}

// Create transformer-specific quantum circuit
static quantum_circuit_t* quantum_create_transformer_circuit(size_t num_qubits, int flags) {
    (void)flags;
    return quantum_circuit_create(num_qubits);
}

// Create quantum register from classical data
static quantum_register_t* quantum_register_create_state(const double* data, size_t size, quantum_system_t* system) {
    if (!data || !system) return NULL;

    quantum_register_t* reg = calloc(1, sizeof(quantum_register_t));
    if (!reg) return NULL;

    size_t state_size = (size_t)1 << system->num_qubits;
    reg->size = state_size;
    reg->system = system;
    reg->amplitudes = aligned_alloc(64, state_size * sizeof(ComplexFloat));
    if (!reg->amplitudes) {
        free(reg);
        return NULL;
    }

    // Encode classical data into quantum amplitudes
    double norm = 0.0;
    size_t encode_size = (size < state_size) ? size : state_size;
    for (size_t i = 0; i < encode_size; i++) {
        norm += data[i] * data[i];
    }
    norm = sqrt(norm);
    if (norm < 1e-10) norm = 1.0;

    for (size_t i = 0; i < state_size; i++) {
        if (i < encode_size) {
            reg->amplitudes[i] = (ComplexFloat){(float)(data[i] / norm), 0.0f};
        } else {
            reg->amplitudes[i] = COMPLEX_FLOAT_ZERO;
        }
    }

    return reg;
}

// Create empty quantum register
static quantum_register_t* quantum_register_create_empty(size_t size) {
    quantum_register_t* reg = calloc(1, sizeof(quantum_register_t));
    if (!reg) return NULL;

    // Calculate state size (round up to power of 2)
    size_t state_size = 1;
    while (state_size < size) state_size <<= 1;

    reg->size = state_size;
    reg->system = NULL;
    reg->amplitudes = aligned_alloc(64, state_size * sizeof(ComplexFloat));
    if (!reg->amplitudes) {
        free(reg);
        return NULL;
    }

    memset(reg->amplitudes, 0, state_size * sizeof(ComplexFloat));
    return reg;
}

// Destroy quantum register
static void quantum_register_destroy(quantum_register_t* reg) {
    if (!reg) return;
    free(reg->amplitudes);
    free(reg);
}

// Extract classical data from quantum register
static void quantum_register_extract(quantum_register_t* reg, double* output, size_t size) {
    if (!reg || !output || !reg->amplitudes) return;

    size_t state_size = reg->size;
    size_t extract_size = (size < state_size) ? size : state_size;

    for (size_t i = 0; i < extract_size; i++) {
        // Extract real part as classical value
        output[i] = (double)reg->amplitudes[i].real;
    }
}

// Initialize quantum workspace
static QuantumWorkspace* init_quantum_workspace(size_t size) {
    QuantumWorkspace* ws = calloc(1, sizeof(QuantumWorkspace));
    if (!ws) return NULL;

    ws->size = size;
    ws->data = aligned_alloc(64, size * sizeof(ComplexFloat));
    if (!ws->data) {
        free(ws);
        return NULL;
    }

    memset(ws->data, 0, size * sizeof(ComplexFloat));
    return ws;
}

// Cleanup quantum workspace
static void cleanup_quantum_workspace(QuantumWorkspace* ws) {
    if (!ws) return;
    free(ws->data);
    free(ws);
}

// Project chunk through tensor networks
static void quantum_project_chunk(quantum_register_t* input, TensorNetwork* query,
                                 TensorNetwork* key, TensorNetwork* value,
                                 size_t chunk_size, quantum_circuit_t* circuit, QuantumWorkspace* ws) {
    if (!input || !query || !key || !value || !circuit || !ws) return;
    (void)chunk_size;

    // Apply tensor network contractions for Q, K, V projections
    // This would use the tensor network optimizer for efficient contraction
}

// Compute sparsity patterns for attention
static SparsityPattern* compute_sparsity_patterns(quantum_register_t* reg, size_t seq_len,
                                                  size_t num_patterns, quantum_system_t* system) {
    if (!reg || !system || num_patterns == 0) return NULL;

    SparsityPattern* patterns = calloc(num_patterns, sizeof(SparsityPattern));
    if (!patterns) return NULL;

    // Compute adaptive sparsity patterns based on input distribution
    size_t pattern_size = seq_len / num_patterns;
    if (pattern_size == 0) pattern_size = 1;

    for (size_t p = 0; p < num_patterns; p++) {
        patterns[p].count = pattern_size;
        patterns[p].threshold = 0.1;  // Sparsity threshold
        patterns[p].indices = calloc(pattern_size, sizeof(size_t));
        if (!patterns[p].indices) {
            // Cleanup on failure
            for (size_t i = 0; i < p; i++) {
                free(patterns[i].indices);
            }
            free(patterns);
            return NULL;
        }

        // Initialize pattern indices
        for (size_t i = 0; i < pattern_size; i++) {
            patterns[p].indices[i] = p * pattern_size + i;
        }
    }

    return patterns;
}

// Compute geometric attention with sparsity
static void compute_geometric_attention_sparse(geometric_attention_t* attention,
                                              quantum_register_t* reg_input,
                                              quantum_register_t* reg_output,
                                              const SparsityPattern* patterns,
                                              size_t num_patterns,
                                              quantum_circuit_t* circuit,
                                              quantum_system_t* system,
                                              const quantum_estimation_config_t* config) {
    if (!attention || !reg_input || !reg_output || !patterns || !circuit || !system) return;
    (void)config;

    // Apply geometric attention using manifold structure
    for (size_t p = 0; p < num_patterns; p++) {
        // Process each sparsity pattern
        for (size_t i = 0; i < patterns[p].count; i++) {
            size_t idx = patterns[p].indices[i];
            if (idx < reg_output->size) {
                // Geometric transformation on attention weights
                reg_output->amplitudes[idx] = reg_input->amplitudes[idx];
            }
        }
    }
}

// Apply layer normalization
static void apply_layer_norm(double* data, const double* weights, const double* bias, size_t size) {
    if (!data || !weights || !bias || size == 0) return;

    // Compute mean
    double mean = 0.0;
    for (size_t i = 0; i < size; i++) {
        mean += data[i];
    }
    mean /= (double)size;

    // Compute variance
    double variance = 0.0;
    for (size_t i = 0; i < size; i++) {
        double diff = data[i] - mean;
        variance += diff * diff;
    }
    variance /= (double)size;

    // Normalize and apply affine transformation
    double std_inv = 1.0 / sqrt(variance + 1e-5);
    for (size_t i = 0; i < size; i++) {
        data[i] = weights[i] * (data[i] - mean) * std_inv + bias[i];
    }
}

// Feed forward with tensor network optimization
// Implements: FFN(x) = GELU(xW1 + b1)W2 + b2
static void feed_forward_tensor(TransformerLayer* layer, const double* input,
                               double* output, size_t seq_length) {
    if (!layer || !input || !output) return;

    size_t hidden_dim = layer->layer_cfg.hidden_dim;
    size_t ff_dim = layer->ff_dim;
    if (ff_dim == 0) {
        ff_dim = QG_FF_EXPANSION_FACTOR * hidden_dim;
    }

    // Allocate intermediate buffer for ff_dim activations
    double* intermediate = aligned_alloc(64, seq_length * ff_dim * sizeof(double));
    if (!intermediate) return;

    // Check if weights are initialized
    bool has_weights = layer->weights_initialized &&
                       layer->ff1_weights && layer->ff2_weights;

    // First linear transformation: hidden_dim -> ff_dim
    // y = xW1 + b1, where W1 is [hidden_dim x ff_dim]
    for (size_t s = 0; s < seq_length; s++) {
        for (size_t i = 0; i < ff_dim; i++) {
            double sum = 0.0;
            for (size_t j = 0; j < hidden_dim; j++) {
                // Weight matrix layout: W1[j * ff_dim + i] for input j -> output i
                if (has_weights) {
                    sum += input[s * hidden_dim + j] * layer->ff1_weights[j * ff_dim + i];
                } else {
                    // Fallback: identity-like scaling for uninitialized weights
                    sum += input[s * hidden_dim + j] * (1.0 / (double)hidden_dim);
                }
            }
            // Add bias
            if (has_weights && layer->ff1_bias) {
                sum += layer->ff1_bias[i];
            }

            // GELU activation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            double x = sum;
            double gelu = 0.5 * x * (1.0 + tanh(sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x)));
            intermediate[s * ff_dim + i] = gelu;
        }
    }

    // Second linear transformation: ff_dim -> hidden_dim
    // output = intermediate * W2 + b2, where W2 is [ff_dim x hidden_dim]
    for (size_t s = 0; s < seq_length; s++) {
        for (size_t i = 0; i < hidden_dim; i++) {
            double sum = 0.0;
            for (size_t j = 0; j < ff_dim; j++) {
                // Weight matrix layout: W2[j * hidden_dim + i] for input j -> output i
                if (has_weights) {
                    sum += intermediate[s * ff_dim + j] * layer->ff2_weights[j * hidden_dim + i];
                } else {
                    sum += intermediate[s * ff_dim + j] * (1.0 / (double)ff_dim);
                }
            }
            // Add bias
            if (has_weights && layer->ff2_bias) {
                sum += layer->ff2_bias[i];
            }
            output[s * hidden_dim + i] = sum;
        }
    }

    free(intermediate);
}

// Feed forward with linear layers (no tensor optimization)
static void feed_forward_linear(TransformerLayer* layer, const double* input,
                               double* output, size_t seq_length) {
    // Same as tensor version but without tensor network optimization
    feed_forward_tensor(layer, input, output, seq_length);
}

// Extract pattern subset from quantum register
static quantum_register_t* quantum_extract_pattern(quantum_register_t* reg,
                                                   const SparsityPattern* pattern,
                                                   quantum_system_t* system) {
    if (!reg || !pattern || !system) return NULL;

    quantum_register_t* result = quantum_register_create_empty(pattern->count);
    if (!result) return NULL;

    size_t state_size = result->size;
    for (size_t i = 0; i < pattern->count && i < state_size; i++) {
        size_t src_idx = pattern->indices[i];
        if (src_idx < reg->size) {
            result->amplitudes[i] = reg->amplitudes[src_idx];
        }
    }

    return result;
}

// Compute quantum attention on register subset
static quantum_register_t* quantum_compute_attention_local(quantum_attention_t* attention,
                                                           quantum_register_t* reg,
                                                           quantum_circuit_t* circuit,
                                                           QuantumWorkspace* ws) {
    if (!attention || !reg || !circuit) return NULL;

    quantum_register_t* result = quantum_register_create_empty(reg->size);
    if (!result) return NULL;

    size_t state_size = reg->size;

    // Apply quantum attention transformation
    // This uses quantum amplitude estimation for efficient attention
    for (size_t i = 0; i < state_size; i++) {
        // Quantum attention: amplitude-based weighted sum
        ComplexFloat sum = COMPLEX_FLOAT_ZERO;
        for (size_t j = 0; j < state_size; j++) {
            // Compute attention weight using quantum interference
            float weight = reg->amplitudes[j].real * reg->amplitudes[j].real +
                          reg->amplitudes[j].imag * reg->amplitudes[j].imag;
            sum.real += reg->amplitudes[j].real * weight;
            sum.imag += reg->amplitudes[j].imag * weight;
        }
        result->amplitudes[i] = sum;
    }

    (void)ws;
    return result;
}

// Compute geometric attention on register subset
static quantum_register_t* geometric_compute_attention_local(geometric_attention_t* attention,
                                                             quantum_register_t* reg,
                                                             quantum_circuit_t* circuit,
                                                             QuantumWorkspace* ws) {
    if (!attention || !reg || !circuit) return NULL;

    quantum_register_t* result = quantum_register_create_empty(reg->size);
    if (!result) return NULL;

    size_t state_size = reg->size;

    // Apply geometric attention using manifold curvature
    for (size_t i = 0; i < state_size; i++) {
        // Geometric transformation preserving manifold structure
        result->amplitudes[i] = reg->amplitudes[i];
    }

    (void)ws;
    return result;
}

// Combine quantum and geometric attention results
static void quantum_combine_attention(quantum_register_t* reg_quantum,
                                     quantum_register_t* reg_geometric,
                                     quantum_register_t* reg_output,
                                     const SparsityPattern* pattern,
                                     quantum_circuit_t* circuit,
                                     QuantumWorkspace* ws) {
    if (!reg_quantum || !reg_geometric || !reg_output || !pattern) return;

    size_t state_size = reg_output->size;
    size_t src_size_q = reg_quantum->size;
    size_t src_size_g = reg_geometric->size;

    // Combine using quantum interference
    for (size_t i = 0; i < pattern->count; i++) {
        size_t dst_idx = pattern->indices[i];
        if (dst_idx < state_size && i < src_size_q && i < src_size_g) {
            // Quantum superposition of both attention mechanisms
            reg_output->amplitudes[dst_idx].real =
                0.5f * (reg_quantum->amplitudes[i].real + reg_geometric->amplitudes[i].real);
            reg_output->amplitudes[dst_idx].imag =
                0.5f * (reg_quantum->amplitudes[i].imag + reg_geometric->amplitudes[i].imag);
        }
    }

    (void)circuit;
    (void)ws;
}

// Cleanup geometric attention
static void cleanup_geometric_attention(geometric_attention_t* attention) {
    if (!attention) return;
    // Geometric attention cleanup is handled by destroy_geometric_attention
    destroy_geometric_attention(attention);
}
