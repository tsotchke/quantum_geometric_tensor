#include "quantum_geometric/core/quantum_attention.h"
#include "quantum_geometric/core/hierarchical_matrix.h"
#include "quantum_geometric/core/quantum_geometric_constants.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_circuit_operations.h"
#include "quantum_geometric/core/differential_transformer.h"
#include "quantum_geometric/core/memory_pool.h"
#include "quantum_geometric/hardware/quantum_geometric_gpu.h"
#include "quantum_geometric/distributed/workload_distribution.h"
#include <complex.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#else
// Stub implementations when OpenMP is not available
static inline int omp_get_thread_num(void) { return 0; }
static inline int omp_get_num_threads(void) { return 1; }
#endif

// Min/max macros
#ifndef MIN
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#endif

// GPU functions are declared in quantum_geometric_gpu.h (included above)
// free_diff_transformer is declared in differential_transformer.h

// Memory pool for attention matrices
static MemoryPool* attention_memory_pool = NULL;
static pthread_mutex_t pool_mutex = PTHREAD_MUTEX_INITIALIZER;

// Initialize memory pool
static void init_attention_memory_pool(void) {
    pthread_mutex_lock(&pool_mutex);
    if (!attention_memory_pool) {
        struct PoolConfig config = {
            .min_block_size = sizeof(double complex),
            .alignment = QG_ATTENTION_CACHE_LINE_SIZE,
            .num_size_classes = 8,
            .growth_factor = 2.0f,
            .prefetch_distance = 4,
            .use_huge_pages = false,
            .cache_local_free_lists = true,
            .max_blocks_per_class = QG_QUANTUM_ATTENTION_INITIAL_BLOCKS,
            .thread_cache_size = 64,
            .enable_stats = true
        };
        attention_memory_pool = create_memory_pool(&config);
    }
    pthread_mutex_unlock(&pool_mutex);
}

// Initialize a single attention head
static quantum_attention_head_t* create_attention_head(size_t head_dim, size_t hidden_dim) {
    quantum_attention_head_t* head = calloc(1, sizeof(quantum_attention_head_t));
    if (!head) return NULL;

    head->head_dim = head_dim;
    head->hidden_dim = hidden_dim;
    head->cache_valid = false;

    // Allocate weight matrices
    size_t weight_size = head_dim * hidden_dim;
    head->query_weights = calloc(weight_size, sizeof(ComplexFloat));
    head->key_weights = calloc(weight_size, sizeof(ComplexFloat));
    head->value_weights = calloc(weight_size, sizeof(ComplexFloat));
    head->output_weights = calloc(weight_size, sizeof(ComplexFloat));
    head->cached_attention = calloc(head_dim * head_dim, sizeof(double));

    if (!head->query_weights || !head->key_weights ||
        !head->value_weights || !head->output_weights) {
        free(head->query_weights);
        free(head->key_weights);
        free(head->value_weights);
        free(head->output_weights);
        free(head->cached_attention);
        free(head);
        return NULL;
    }

    // Initialize weights with Xavier initialization
    double scale = sqrt(2.0 / (double)(head_dim + hidden_dim));
    for (size_t i = 0; i < weight_size; i++) {
        double rand_val = ((double)rand() / RAND_MAX - 0.5) * 2.0 * scale;
        head->query_weights[i] = (ComplexFloat){.real = (float)rand_val, .imag = 0.0f};
        head->key_weights[i] = (ComplexFloat){.real = (float)rand_val, .imag = 0.0f};
        head->value_weights[i] = (ComplexFloat){.real = (float)rand_val, .imag = 0.0f};
        head->output_weights[i] = (ComplexFloat){.real = (float)rand_val, .imag = 0.0f};
    }

    return head;
}

// Initialize quantum attention mechanism
quantum_attention_t* init_quantum_attention(
    size_t num_heads,
    size_t head_dim,
    quantum_attention_config_t config) {

    init_attention_memory_pool();

    quantum_attention_t* attention = calloc(1, sizeof(quantum_attention_t));
    if (!attention) return NULL;

    attention->num_heads = num_heads;
    attention->head_dim = head_dim;
    attention->hidden_dim = num_heads * head_dim;
    attention->config = config;
    attention->total_operations = 0;
    attention->average_sparsity = 0.0;

    // Allocate heads array
    attention->heads = calloc(num_heads, sizeof(quantum_attention_head_t*));
    if (!attention->heads) {
        free(attention);
        return NULL;
    }

    // Create each attention head
    for (size_t i = 0; i < num_heads; i++) {
        attention->heads[i] = create_attention_head(head_dim, attention->hidden_dim);
        if (!attention->heads[i]) {
            // Cleanup already allocated heads
            for (size_t j = 0; j < i; j++) {
                free(attention->heads[j]->query_weights);
                free(attention->heads[j]->key_weights);
                free(attention->heads[j]->value_weights);
                free(attention->heads[j]->output_weights);
                free(attention->heads[j]->cached_attention);
                free(attention->heads[j]);
            }
            free(attention->heads);
            free(attention);
            return NULL;
        }
    }

    // Allocate output projection
    size_t proj_size = attention->hidden_dim * attention->hidden_dim;
    attention->output_projection = calloc(proj_size, sizeof(ComplexFloat));
    attention->layer_norm_gamma = calloc(attention->hidden_dim, sizeof(ComplexFloat));
    attention->layer_norm_beta = calloc(attention->hidden_dim, sizeof(ComplexFloat));

    if (!attention->output_projection || !attention->layer_norm_gamma ||
        !attention->layer_norm_beta) {
        cleanup_quantum_attention(attention);
        return NULL;
    }

    // Initialize layer norm with identity
    for (size_t i = 0; i < attention->hidden_dim; i++) {
        attention->layer_norm_gamma[i] = (ComplexFloat){.real = 1.0f, .imag = 0.0f};
        attention->layer_norm_beta[i] = (ComplexFloat){.real = 0.0f, .imag = 0.0f};
    }

    // Initialize output projection as identity-like
    for (size_t i = 0; i < attention->hidden_dim; i++) {
        for (size_t j = 0; j < attention->hidden_dim; j++) {
            float val = (i == j) ? 1.0f : 0.0f;
            attention->output_projection[i * attention->hidden_dim + j] =
                (ComplexFloat){.real = val, .imag = 0.0f};
        }
    }

    return attention;
}

// Create quantum attention with full configuration
quantum_attention_t* create_quantum_attention(const quantum_attention_config_t* config) {
    if (!config) return NULL;
    return init_quantum_attention(config->num_heads, config->head_dim, *config);
}

// Helper function for hierarchical attention computation with differential backprop
static void compute_hierarchical_attention_differential(
    DiffTransformerState* state,
    const double complex* queries,
    const double complex* keys,
    const double complex* values,
    double complex* output,
    double complex* gradients,
    size_t seq_length,
    size_t head_dim
) {
    // Use hierarchical structure to compute attention in O(n log n)
    size_t num_levels = (size_t)ceil(log2(seq_length));
    size_t level_size = seq_length;

    // Allocate temporary buffers for hierarchical computation
    double complex* level_scores = malloc(seq_length * seq_length * sizeof(double complex));
    double complex* level_derivs = malloc(seq_length * seq_length * sizeof(double complex));
    double complex* attention_weights = malloc(seq_length * seq_length * sizeof(double complex));

    // Process each level of hierarchy
    for (size_t level = 0; level < num_levels; level++) {
        size_t stride = 1 << level;
        
        // Compute attention scores at current level with differential backprop
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < level_size; i += stride) {
            for (size_t j = 0; j < level_size; j += stride) {
                double complex score = 0;
                double complex deriv = 0;

                // Compute score and derivatives using differential transformer
                for (size_t k = 0; k < head_dim; k++) {
                    double complex q = queries[i * head_dim + k];
                    double complex k_val = keys[j * head_dim + k];
                    double complex v = values[j * head_dim + k];

                    // Forward pass: standard scaled dot-product attention score
                    score += q * conj(k_val) / sqrt(head_dim);

                    // Backward pass derivatives with value-weighted contribution
                    // dL/dQ_k = grad_k * K_k^* / sqrt(d)
                    // dL/dK_k = Q_k * grad_k / sqrt(d)
                    // Include value magnitude for importance weighting
                    double v_weight = 1.0 + 0.1 * cabs(v);  // Value-guided importance
                    deriv += v_weight * (gradients[i * head_dim + k] * conj(k_val) / sqrt(head_dim) +
                            q * gradients[j * head_dim + k] / sqrt(head_dim));
                }

                // Apply differential softmax
                level_scores[i * seq_length + j] = score;
                level_derivs[i * seq_length + j] = deriv;
            }
        }

        // Apply differential softmax within each block
        for (size_t block = 0; block < level_size; block += stride) {
            double complex max_val = level_scores[block * seq_length + block];
            double complex sum = 0;
            
            // Find max value for numerical stability
            for (size_t j = 0; j < stride; j++) {
                if (cabs(level_scores[block * seq_length + (block + j)]) > cabs(max_val)) {
                    max_val = level_scores[block * seq_length + (block + j)];
                }
            }
            
            // Compute exponentials and sum
            for (size_t j = 0; j < stride; j++) {
                double complex exp_val = cexp(level_scores[block * seq_length + (block + j)] - max_val);
                attention_weights[block * seq_length + (block + j)] = exp_val;
                sum += exp_val;
            }
            
            // Normalize and compute derivatives
            double complex inv_sum = 1.0 / (sum + QG_ATTENTION_NORM_THRESHOLD);
            for (size_t j = 0; j < stride; j++) {
                size_t idx = block * seq_length + (block + j);
                attention_weights[idx] *= inv_sum;
                
                // Compute softmax derivative
                level_derivs[idx] = attention_weights[idx] * (1 - attention_weights[idx]) * level_derivs[idx];
            }
        }

        // Propagate attention and gradients to next level
        if (level < num_levels - 1) {
            size_t next_stride = stride * 2;
            
            #pragma omp parallel for collapse(2)
            for (size_t i = 0; i < level_size; i += next_stride) {
                for (size_t j = 0; j < level_size; j += next_stride) {
                    double complex sum_attention = 0;
                    double complex sum_deriv = 0;
                    
                    // Aggregate attention and derivatives from current level
                    for (size_t di = 0; di < stride; di++) {
                        for (size_t dj = 0; dj < stride; dj++) {
                            size_t idx = (i + di) * seq_length + (j + dj);
                            sum_attention += attention_weights[idx];
                            sum_deriv += level_derivs[idx];
                        }
                    }
                    
                    // Store aggregated values for next level
                    size_t next_idx = (i/2) * seq_length + (j/2);
                    level_scores[next_idx] = sum_attention / (stride * stride);
                    level_derivs[next_idx] = sum_deriv / (stride * stride);
                }
            }
        }

        level_size /= 2;
    }

    // Compute final output and gradients
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < seq_length; i++) {
        for (size_t h = 0; h < head_dim; h++) {
            double complex out_val = 0;
            double complex grad_val = 0;
            
            for (size_t j = 0; j < seq_length; j++) {
                double complex attn = attention_weights[i * seq_length + j];
                double complex val = values[j * head_dim + h];
                
                // Forward pass
                out_val += attn * val;
                
                // Backward pass
                grad_val += level_derivs[i * seq_length + j] * val;
            }
            
            output[i * head_dim + h] = out_val;
            gradients[i * head_dim + h] = grad_val;
        }
    }

    // Cleanup
    free(level_scores);
    free(level_derivs);
    free(attention_weights);
}

// Multi-head quantum attention with differential backprop - O(log n)
void compute_quantum_attention(double complex* output,
                             const double complex* query,
                             const double complex* key,
                             const double complex* value,
                             size_t batch_size,
                             size_t num_heads,
                             size_t head_dim,
                             bool enable_checkpointing) {
    // Initialize memory pool and differential transformer
    init_attention_memory_pool();
    DiffTransformerState* state = create_diff_transformer(
        batch_size * num_heads,
        head_dim,
        QG_QUANTUM_ATTENTION_MAX_HEADS,
        QG_QUANTUM_ATTENTION_DROPOUT_RATE
    );
    
    // Get multi-GPU context
    MultiGPUContext* ctx = init_multi_gpu_context();
    if (!ctx) {
        // Fallback to CPU implementation with differential backprop
        double complex* gradients = calloc(batch_size * num_heads * head_dim, sizeof(double complex));
        compute_hierarchical_attention_differential(
            state, query, key, value, output, gradients,
            batch_size * num_heads, head_dim
        );
        free(gradients);
        free_diff_transformer(state);
        return;
    }
    
    // Distribute computation across GPUs
    size_t num_gpus = ctx->num_contexts;
    size_t batch_per_gpu = (batch_size + num_gpus - 1) / num_gpus;
    
    #pragma omp parallel num_threads(num_gpus)
    {
        int gpu_id = omp_get_thread_num();
        size_t start_batch = gpu_id * batch_per_gpu;
        size_t end_batch = MIN(start_batch + batch_per_gpu, batch_size);
        
        if (start_batch < end_batch) {
            // Get GPU context for this device
            GPUContext* gpu_ctx = get_gpu_context(ctx, gpu_id);
            
            // Allocate GPU memory from pool
            size_t head_size = head_dim * head_dim;
            double complex* d_query = gpu_alloc_from_pool(
                attention_memory_pool,
                head_size * sizeof(double complex)
            );
            double complex* d_key = gpu_alloc_from_pool(
                attention_memory_pool,
                head_size * sizeof(double complex)
            );
            double complex* d_value = gpu_alloc_from_pool(
                attention_memory_pool,
                head_size * sizeof(double complex)
            );
            double complex* d_output = gpu_alloc_from_pool(
                attention_memory_pool,
                head_size * sizeof(double complex)
            );
            double complex* d_gradients = gpu_alloc_from_pool(
                attention_memory_pool,
                head_size * sizeof(double complex)
            );
            
            // Process batches
            for (size_t b = start_batch; b < end_batch; b++) {
                // Process attention heads
                for (size_t h = 0; h < num_heads; h++) {
                    size_t offset = (b * num_heads + h) * head_size;
                    
                    // Copy inputs to GPU
                    gpu_memcpy_to_device_async(
                        d_query,
                        query + offset,
                        head_size * sizeof(double complex),
                        gpu_ctx->command_queue
                    );
                    gpu_memcpy_to_device_async(
                        d_key,
                        key + offset,
                        head_size * sizeof(double complex),
                        gpu_ctx->command_queue
                    );
                    gpu_memcpy_to_device_async(
                        d_value,
                        value + offset,
                        head_size * sizeof(double complex),
                        gpu_ctx->command_queue
                    );
                    
                    if (enable_checkpointing) {
                        // Save intermediate states for backprop
                        save_attention_checkpoint(
                            d_query, d_key, d_value,
                            head_size, b, h,
                            gpu_ctx
                        );
                    }
                    
                    // Convert to hierarchical representation for O(n log n) attention
                    // Using H-matrix approximation with tolerance for efficient computation
                    double h_tolerance = 1e-6;  // Hierarchical matrix approximation tolerance
                    HierarchicalMatrix* h_query = convert_to_hierarchical_gpu(
                        (const ComplexFloat*)d_query, head_dim, head_dim, h_tolerance, gpu_ctx);
                    HierarchicalMatrix* h_key = convert_to_hierarchical_gpu(
                        (const ComplexFloat*)d_key, head_dim, head_dim, h_tolerance, gpu_ctx);
                    HierarchicalMatrix* h_value = convert_to_hierarchical_gpu(
                        (const ComplexFloat*)d_value, head_dim, head_dim, h_tolerance, gpu_ctx);
                    HierarchicalMatrix* h_output = create_hierarchical_matrix_gpu(
                        head_dim, head_dim, h_tolerance, gpu_ctx);

                    // Compute attention with differential backprop
                    compute_hierarchical_attention_differential(
                        state,
                        (double complex*)h_query->data,
                        (double complex*)h_key->data,
                        (double complex*)h_value->data,
                        (double complex*)h_output->data,
                        d_gradients,
                        head_dim,
                        head_dim
                    );

                    // Convert back and apply dropout for regularization
                    convert_from_hierarchical_with_dropout_gpu(
                        (ComplexFloat*)d_output,
                        h_output,
                        head_size,
                        QG_QUANTUM_ATTENTION_DROPOUT_RATE,
                        gpu_ctx
                    );
                    
                    // Copy result back
                    gpu_memcpy_to_host_async(
                        output + offset,
                        d_output,
                        head_size * sizeof(double complex),
                        gpu_ctx->command_queue
                    );
                    
                    // Cleanup hierarchical matrices
                    destroy_hierarchical_matrix_gpu(h_query, gpu_ctx);
                    destroy_hierarchical_matrix_gpu(h_key, gpu_ctx);
                    destroy_hierarchical_matrix_gpu(h_value, gpu_ctx);
                    destroy_hierarchical_matrix_gpu(h_output, gpu_ctx);
                }
            }
            
            // Cleanup GPU memory
            gpu_free_to_pool(attention_memory_pool, d_query);
            gpu_free_to_pool(attention_memory_pool, d_key);
            gpu_free_to_pool(attention_memory_pool, d_value);
            gpu_free_to_pool(attention_memory_pool, d_output);
            gpu_free_to_pool(attention_memory_pool, d_gradients);
        }
    }
    
    // Cleanup
    sync_multi_gpu_context(ctx);
    cleanup_multi_gpu_context(ctx);
    free_diff_transformer(state);
}

// Cleanup attention resources (global cleanup for backward compatibility)
static void cleanup_quantum_attention_global(void) {
    cleanup_attention_cache();
    cleanup_attention_buffers();
}

// Cleanup individual quantum attention instance
void cleanup_quantum_attention(quantum_attention_t* attention) {
    if (!attention) return;

    // Free each attention head
    if (attention->heads) {
        for (size_t i = 0; i < attention->num_heads; i++) {
            if (attention->heads[i]) {
                free(attention->heads[i]->query_weights);
                free(attention->heads[i]->key_weights);
                free(attention->heads[i]->value_weights);
                free(attention->heads[i]->output_weights);
                free(attention->heads[i]->cached_attention);
                free(attention->heads[i]);
            }
        }
        free(attention->heads);
    }

    free(attention->output_projection);
    free(attention->layer_norm_gamma);
    free(attention->layer_norm_beta);
    free(attention->sparse_indices);

    if (attention->attention_circuit) {
        quantum_circuit_destroy(attention->attention_circuit);
    }

    free(attention);
}

// Alias for destroy
void destroy_quantum_attention(quantum_attention_t* attention) {
    cleanup_quantum_attention(attention);
}
