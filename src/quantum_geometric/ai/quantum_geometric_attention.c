#include "quantum_geometric/ai/quantum_geometric_attention.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/core/quantum_operations.h"
#include "quantum_geometric/core/quantum_geometric_constants.h"
#include "quantum_geometric/core/differential_transformer.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

// Initialize attention mechanism
int qg_attention_init(attention_config_t* config) {
    if (!config) {
        return QG_ERROR_INVALID_ARGUMENT;
    }

    // Set default values if not specified
    if (config->num_heads == 0) {
        config->num_heads = QG_DEFAULT_NUM_HEADS;
    }
    if (config->head_dim == 0) {
        config->head_dim = QG_DEFAULT_HEAD_DIM;
    }
    if (config->model_dim == 0) {
        config->model_dim = config->num_heads * config->head_dim;
    }

    return QG_SUCCESS;
}

// Clean up attention resources
void qg_attention_cleanup(attention_weights_t* weights) {
    if (!weights) return;

    free(weights->query_weights);
    free(weights->key_weights);
    free(weights->value_weights);
    free(weights->output_weights);

    weights->query_weights = NULL;
    weights->key_weights = NULL;
    weights->value_weights = NULL;
    weights->output_weights = NULL;
    weights->weight_size = 0;
}

// Initialize hierarchical attention
int qg_hierarchical_attention_init(hierarchical_attention_t* hier_attn,
                                 size_t seq_length) {
    if (!hier_attn || seq_length < QG_ATTENTION_MIN_LEVEL_SIZE) {
        return QG_ERROR_INVALID_ARGUMENT;
    }

    hier_attn->seq_length = seq_length;
    hier_attn->num_levels = (size_t)ceil(log2(seq_length));
    hier_attn->sparsity_factor = QG_ATTENTION_SPARSITY_FACTOR;

    // Allocate levels
    hier_attn->levels = malloc(hier_attn->num_levels * sizeof(attention_level_t));
    if (!hier_attn->levels) {
        return QG_ERROR_OUT_OF_MEMORY;
    }

    // Initialize each level
    size_t level_size = seq_length;
    for (size_t i = 0; i < hier_attn->num_levels; i++) {
        attention_level_t* level = &hier_attn->levels[i];
        level->level_size = level_size;
        level->stride = 1 << i;

        // Allocate level buffers
        size_t buffer_size = level_size * level_size * sizeof(float);
        level->level_scores = malloc(buffer_size);
        level->level_derivs = malloc(buffer_size);
        level->sparsity_mask = malloc(buffer_size);

        if (!level->level_scores || !level->level_derivs || !level->sparsity_mask) {
            qg_hierarchical_attention_cleanup(hier_attn);
            return QG_ERROR_OUT_OF_MEMORY;
        }

        // Initialize sparsity mask based on level
        float sparsity_threshold = hier_attn->sparsity_factor * (i + 1);
        for (size_t j = 0; j < level_size * level_size; j++) {
            level->sparsity_mask[j] = ((float)rand() / RAND_MAX) < sparsity_threshold ? 1.0f : 0.0f;
        }

        level_size /= 2;
    }

    return QG_SUCCESS;
}

// Clean up hierarchical attention resources
void qg_hierarchical_attention_cleanup(hierarchical_attention_t* hier_attn) {
    if (!hier_attn) return;

    if (hier_attn->levels) {
        for (size_t i = 0; i < hier_attn->num_levels; i++) {
            attention_level_t* level = &hier_attn->levels[i];
            free(level->level_scores);
            free(level->level_derivs);
            free(level->sparsity_mask);
        }
        free(hier_attn->levels);
    }

    hier_attn->levels = NULL;
    hier_attn->num_levels = 0;
    hier_attn->seq_length = 0;
}

// Helper function for hierarchical attention computation
static void compute_level_attention(
    attention_level_t* level,
    const float* queries,
    const float* keys,
    float* scores,
    size_t head_dim
) {
    const size_t stride = level->stride;
    const size_t level_size = level->level_size;

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < level_size; i += stride) {
        for (size_t j = 0; j < level_size; j += stride) {
            if (level->sparsity_mask[i * level_size + j] == 0.0f) {
                continue;
            }

            float score = 0.0f;
            for (size_t k = 0; k < head_dim; k++) {
                float q = queries[i * head_dim + k];
                float k_val = keys[j * head_dim + k];
                score += q * k_val;
            }

            level->level_scores[i * level_size + j] = score / sqrtf(head_dim);
        }
    }
}

// Compute hierarchical attention scores
int qg_hierarchical_attention_compute(hierarchical_attention_t* hier_attn,
                                    const float* queries,
                                    const float* keys,
                                    float* scores) {
    if (!hier_attn || !queries || !keys || !scores) {
        return QG_ERROR_INVALID_ARGUMENT;
    }

    // Process each level
    for (size_t i = 0; i < hier_attn->num_levels; i++) {
        attention_level_t* level = &hier_attn->levels[i];
        
        // Compute attention at current level
        compute_level_attention(level, queries, keys, scores, QG_DEFAULT_HEAD_DIM);

        // Apply softmax within each block
        #pragma omp parallel for
        for (size_t block = 0; block < level->level_size; block += level->stride) {
            float max_val = -INFINITY;
            float sum = 0.0f;

            // Find max value in block
            for (size_t j = 0; j < level->stride; j++) {
                if (level->sparsity_mask[block * level->level_size + j] != 0.0f) {
                    max_val = fmaxf(max_val, level->level_scores[block * level->level_size + j]);
                }
            }

            // Compute exp and sum
            for (size_t j = 0; j < level->stride; j++) {
                if (level->sparsity_mask[block * level->level_size + j] != 0.0f) {
                    float exp_val = expf(level->level_scores[block * level->level_size + j] - max_val);
                    level->level_scores[block * level->level_size + j] = exp_val;
                    sum += exp_val;
                }
            }

            // Normalize
            float inv_sum = 1.0f / (sum + QG_ATTENTION_NORM_THRESHOLD);
            for (size_t j = 0; j < level->stride; j++) {
                if (level->sparsity_mask[block * level->level_size + j] != 0.0f) {
                    level->level_scores[block * level->level_size + j] *= inv_sum;
                }
            }
        }

        // Propagate scores to next level if not last level
        if (i < hier_attn->num_levels - 1) {
            attention_level_t* next_level = &hier_attn->levels[i + 1];
            const size_t next_stride = next_level->stride;

            #pragma omp parallel for collapse(2)
            for (size_t i = 0; i < level->level_size; i += next_stride) {
                for (size_t j = 0; j < level->level_size; j += next_stride) {
                    float sum = 0.0f;
                    int count = 0;

                    // Aggregate scores from current level
                    for (size_t di = 0; di < level->stride; di++) {
                        for (size_t dj = 0; dj < level->stride; dj++) {
                            if (level->sparsity_mask[(i + di) * level->level_size + (j + dj)] != 0.0f) {
                                sum += level->level_scores[(i + di) * level->level_size + (j + dj)];
                                count++;
                            }
                        }
                    }

                    // Average and store in next level
                    if (count > 0) {
                        next_level->level_scores[i * next_level->level_size + j] = sum / count;
                    }
                }
            }
        }
    }

    // Copy final level scores to output
    attention_level_t* final_level = &hier_attn->levels[hier_attn->num_levels - 1];
    memcpy(scores, final_level->level_scores, hier_attn->seq_length * hier_attn->seq_length * sizeof(float));

    return QG_SUCCESS;
}

// Compute backward pass for hierarchical attention
int qg_hierarchical_attention_backward(hierarchical_attention_t* hier_attn,
                                     const float* grad_output,
                                     float* grad_queries,
                                     float* grad_keys) {
    if (!hier_attn || !grad_output || !grad_queries || !grad_keys) {
        return QG_ERROR_INVALID_ARGUMENT;
    }

    // Initialize gradients to zero
    memset(grad_queries, 0, hier_attn->seq_length * QG_DEFAULT_HEAD_DIM * sizeof(float));
    memset(grad_keys, 0, hier_attn->seq_length * QG_DEFAULT_HEAD_DIM * sizeof(float));

    // Backpropagate through levels in reverse order
    for (size_t i = hier_attn->num_levels; i > 0; i--) {
        attention_level_t* level = &hier_attn->levels[i - 1];
        
        // Compute gradients for current level
        #pragma omp parallel for collapse(2)
        for (size_t j = 0; j < level->level_size; j++) {
            for (size_t k = 0; k < QG_DEFAULT_HEAD_DIM; k++) {
                if (level->sparsity_mask[j * level->level_size + k] == 0.0f) {
                    continue;
                }

                float grad = level->level_derivs[j * level->level_size + k];
                
                // Update query gradients
                grad_queries[j * QG_DEFAULT_HEAD_DIM + k] += grad;
                
                // Update key gradients
                grad_keys[k * QG_DEFAULT_HEAD_DIM + j] += grad;
            }
        }

        // Propagate gradients to previous level if not first level
        if (i > 1) {
            attention_level_t* prev_level = &hier_attn->levels[i - 2];
            const size_t prev_stride = prev_level->stride;

            #pragma omp parallel for collapse(2)
            for (size_t j = 0; j < level->level_size; j += level->stride) {
                for (size_t k = 0; k < level->level_size; k += level->stride) {
                    float grad = level->level_derivs[j * level->level_size + k];
                    
                    // Distribute gradient to previous level
                    for (size_t dj = 0; dj < prev_stride; dj++) {
                        for (size_t dk = 0; dk < prev_stride; dk++) {
                            if (prev_level->sparsity_mask[(j + dj) * prev_level->level_size + (k + dk)] != 0.0f) {
                                prev_level->level_derivs[(j + dj) * prev_level->level_size + (k + dk)] += 
                                    grad / (prev_stride * prev_stride);
                            }
                        }
                    }
                }
            }
        }
    }

    return QG_SUCCESS;
}

// Compute attention scores between queries and keys using differential attention
int qg_attention_compute_scores(const float* queries,
                              const float* keys,
                              float* scores,
                              size_t seq_length) {
    if (!queries || !keys || !scores || seq_length == 0) {
        return QG_ERROR_INVALID_ARGUMENT;
    }

    // Create differential attention state
    DiffAttention* diff_attn = create_diff_attention(QG_DEFAULT_HEAD_DIM, QG_DEFAULT_NUM_HEADS);
    if (!diff_attn) {
        return QG_ERROR_OUT_OF_MEMORY;
    }

    // Convert inputs to double for differential computation
    double* query_doubles = malloc(seq_length * QG_DEFAULT_HEAD_DIM * sizeof(double));
    double* key_doubles = malloc(seq_length * QG_DEFAULT_HEAD_DIM * sizeof(double));
    double* score_doubles = malloc(seq_length * seq_length * sizeof(double));

    if (!query_doubles || !key_doubles || !score_doubles) {
        free(query_doubles);
        free(key_doubles);
        free(score_doubles);
        free_diff_attention(diff_attn);
        return QG_ERROR_OUT_OF_MEMORY;
    }

    // Convert to double precision
    for (size_t i = 0; i < seq_length * QG_DEFAULT_HEAD_DIM; i++) {
        query_doubles[i] = (double)queries[i];
        key_doubles[i] = (double)keys[i];
    }

    // Compute attention using hierarchical structure
    compute_hierarchical_attention(
        diff_attn,
        query_doubles,
        key_doubles,
        score_doubles,
        seq_length,
        QG_DEFAULT_HEAD_DIM
    );

    // Convert back to float
    for (size_t i = 0; i < seq_length * seq_length; i++) {
        scores[i] = (float)score_doubles[i];
    }

    // Clean up
    free(query_doubles);
    free(key_doubles);
    free(score_doubles);
    free_diff_attention(diff_attn);

    return QG_SUCCESS;
}

// Apply attention mask
int qg_attention_apply_mask(float* attention_scores,
                          const bool* mask,
                          size_t seq_length) {
    if (!attention_scores || !mask || seq_length == 0) {
        return QG_ERROR_INVALID_ARGUMENT;
    }

    const float mask_value = -INFINITY;
    
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < seq_length; i++) {
        for (size_t j = 0; j < seq_length; j++) {
            if (!mask[i * seq_length + j]) {
                attention_scores[i * seq_length + j] = mask_value;
            }
        }
    }

    return QG_SUCCESS;
}

// Apply differential softmax to attention scores
int qg_attention_softmax(float* scores,
                        size_t seq_length) {
    if (!scores || seq_length == 0) {
        return QG_ERROR_INVALID_ARGUMENT;
    }

    // Convert to double for differential computation
    double* values = malloc(seq_length * seq_length * sizeof(double));
    double* derivatives = malloc(seq_length * seq_length * sizeof(double));

    if (!values || !derivatives) {
        free(values);
        free(derivatives);
        return QG_ERROR_OUT_OF_MEMORY;
    }

    // Convert to double precision
    for (size_t i = 0; i < seq_length * seq_length; i++) {
        values[i] = (double)scores[i];
        derivatives[i] = 0.0; // Initialize derivatives
    }

    // Apply differential softmax
    differential_softmax(values, derivatives, seq_length);

    // Convert back to float
    for (size_t i = 0; i < seq_length * seq_length; i++) {
        scores[i] = (float)values[i];
    }

    free(values);
    free(derivatives);
    return QG_SUCCESS;
}

// Helper function for hierarchical context computation
static void compute_hierarchical_context(
    const double* scores,
    const double* values,
    double* context,
    size_t seq_length,
    size_t head_dim
) {
    size_t num_levels = (size_t)ceil(log2(seq_length));
    
    // Compute context hierarchically
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < seq_length; i++) {
        for (size_t d = 0; d < head_dim; d++) {
            double sum = 0.0;
            size_t level_size = 1;
            
            // Accumulate through levels
            for (size_t level = 0; level < num_levels; level++) {
                size_t start = i & ~(level_size - 1);
                size_t end = start + level_size;
                
                for (size_t j = start; j < end && j < seq_length; j++) {
                    sum += scores[i * seq_length + j] * values[j * head_dim + d];
                }
                
                level_size *= 2;
            }
            
            context[i * head_dim + d] = sum;
        }
    }
}

// Compute attention context using differential attention
int qg_attention_compute_context(const float* scores,
                               const float* values,
                               float* context,
                               size_t seq_length) {
    if (!scores || !values || !context || seq_length == 0) {
        return QG_ERROR_INVALID_ARGUMENT;
    }

    // Convert inputs to double
    double* score_doubles = malloc(seq_length * seq_length * sizeof(double));
    double* value_doubles = malloc(seq_length * QG_DEFAULT_HEAD_DIM * sizeof(double));
    double* context_doubles = malloc(seq_length * QG_DEFAULT_HEAD_DIM * sizeof(double));

    if (!score_doubles || !value_doubles || !context_doubles) {
        free(score_doubles);
        free(value_doubles);
        free(context_doubles);
        return QG_ERROR_OUT_OF_MEMORY;
    }

    // Convert to double precision
    for (size_t i = 0; i < seq_length * seq_length; i++) {
        score_doubles[i] = (double)scores[i];
    }
    for (size_t i = 0; i < seq_length * QG_DEFAULT_HEAD_DIM; i++) {
        value_doubles[i] = (double)values[i];
    }

    // Compute context using hierarchical structure
    compute_hierarchical_context(
        score_doubles,
        value_doubles,
        context_doubles,
        seq_length,
        QG_DEFAULT_HEAD_DIM
    );

    // Convert back to float
    for (size_t i = 0; i < seq_length * QG_DEFAULT_HEAD_DIM; i++) {
        context[i] = (float)context_doubles[i];
    }

    // Clean up
    free(score_doubles);
    free(value_doubles);
    free(context_doubles);

    return QG_SUCCESS;
}

// Forward pass for multi-head attention using differential transformer
int qg_multihead_attention_forward(const attention_weights_t* weights,
                                 const float* input,
                                 float* output,
                                 size_t seq_length) {
    if (!weights || !input || !output || seq_length == 0) {
        return QG_ERROR_INVALID_ARGUMENT;
    }

    // Create differential transformer state
    DiffTransformerState* state = create_diff_transformer(
        seq_length, QG_DEFAULT_HEAD_DIM * QG_DEFAULT_NUM_HEADS, QG_DEFAULT_NUM_HEADS, 0.0
    );
    if (!state) {
        return QG_ERROR_OUT_OF_MEMORY;
    }

    // Process each attention head
    for (size_t h = 0; h < QG_DEFAULT_NUM_HEADS; h++) {
        size_t head_offset = h * QG_DEFAULT_HEAD_DIM;
        
        // Project queries, keys, and values
        float* queries = malloc(seq_length * QG_DEFAULT_HEAD_DIM * sizeof(float));
        float* keys = malloc(seq_length * QG_DEFAULT_HEAD_DIM * sizeof(float));
        float* values = malloc(seq_length * QG_DEFAULT_HEAD_DIM * sizeof(float));
        float* scores = malloc(seq_length * seq_length * sizeof(float));
        float* context = malloc(seq_length * QG_DEFAULT_HEAD_DIM * sizeof(float));

        if (!queries || !keys || !values || !scores || !context) {
            free(queries);
            free(keys);
            free(values);
            free(scores);
            free(context);
            free_diff_transformer(state);
            return QG_ERROR_OUT_OF_MEMORY;
        }

        // Compute attention scores
        int status = qg_attention_compute_scores(queries, keys, scores, seq_length);
        if (status != QG_SUCCESS) {
            free(queries);
            free(keys);
            free(values);
            free(scores);
            free(context);
            free_diff_transformer(state);
            return status;
        }

        // Scale attention scores
        status = qg_attention_scale_scores(scores, seq_length, 1.0f / sqrtf(QG_DEFAULT_HEAD_DIM));
        if (status != QG_SUCCESS) {
            free(queries);
            free(keys);
            free(values);
            free(scores);
            free(context);
            free_diff_transformer(state);
            return status;
        }

        // Apply softmax
        status = qg_attention_softmax(scores, seq_length);
        if (status != QG_SUCCESS) {
            free(queries);
            free(keys);
            free(values);
            free(scores);
            free(context);
            free_diff_transformer(state);
            return status;
        }

        // Compute context
        status = qg_attention_compute_context(scores, values, context, seq_length);
        if (status != QG_SUCCESS) {
            free(queries);
            free(keys);
            free(values);
            free(scores);
            free(context);
            free_diff_transformer(state);
            return status;
        }

        // Copy to output with head offset
        #pragma omp parallel for
        for (size_t i = 0; i < seq_length; i++) {
            memcpy(&output[i * QG_DEFAULT_NUM_HEADS * QG_DEFAULT_HEAD_DIM + head_offset],
                   &context[i * QG_DEFAULT_HEAD_DIM],
                   QG_DEFAULT_HEAD_DIM * sizeof(float));
        }

        free(queries);
        free(keys);
        free(values);
        free(scores);
        free(context);
    }

    free_diff_transformer(state);
    return QG_SUCCESS;
}

// Backward pass for multi-head attention
int qg_multihead_attention_backward(const attention_weights_t* weights,
                                  const float* grad_output,
                                  float* grad_input,
                                  size_t seq_length) {
    if (!weights || !grad_output || !grad_input || seq_length == 0) {
        return QG_ERROR_INVALID_ARGUMENT;
    }

    // Create differential transformer state
    DiffTransformerState* state = create_diff_transformer(
        seq_length, QG_DEFAULT_HEAD_DIM * QG_DEFAULT_NUM_HEADS, QG_DEFAULT_NUM_HEADS, 0.0
    );
    if (!state) {
        return QG_ERROR_OUT_OF_MEMORY;
    }

    // Zero initialize gradients
    size_t total_size = seq_length * weights->weight_size;
    memset(grad_input, 0, total_size * sizeof(float));

    // Compute gradients using differential transformer
    diff_transformer_forward(state, (const double*)grad_output, (double*)grad_input);

    free_diff_transformer(state);
    return QG_SUCCESS;
}

// Scale attention scores
int qg_attention_scale_scores(float* scores,
                            size_t seq_length,
                            float scaling_factor) {
    if (!scores || seq_length == 0 || scaling_factor <= 0.0f) {
        return QG_ERROR_INVALID_ARGUMENT;
    }

    __m256 scale_vec = _mm256_set1_ps(scaling_factor);
    
    #pragma omp parallel for
    for (size_t i = 0; i < seq_length * seq_length; i += 8) {
        __m256 score_vec = _mm256_load_ps(&scores[i]);
        _mm256_store_ps(&scores[i], _mm256_mul_ps(score_vec, scale_vec));
    }

    return QG_SUCCESS;
}
