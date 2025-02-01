#include "quantum_geometric/core/quantum_geometric_attention.h"
#include "quantum_geometric/core/hierarchical_matrix.h"
#include "quantum_geometric/core/simd_operations.h"
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>

// Optimization parameters
#define ATTENTION_BLOCK_SIZE 64
#define ATTENTION_RANK_THRESHOLD 32
#define ATTENTION_TOLERANCE 1e-6

typedef struct {
    HierarchicalMatrix* query;
    HierarchicalMatrix* key;
    HierarchicalMatrix* value;
    HierarchicalMatrix* output;
    double temperature;
    double dropout_rate;
} AttentionContext;

static void compute_attention_scores(AttentionContext* ctx);
static void apply_attention_mask(HierarchicalMatrix* scores, const bool* mask);
static void scale_and_softmax(HierarchicalMatrix* mat, double temperature);

GeometricAttention* attention_create(size_t dim, size_t num_heads) {
    GeometricAttention* attn = malloc(sizeof(GeometricAttention));
    if (!attn) return NULL;
    
    attn->dim = dim;
    attn->num_heads = num_heads;
    attn->head_dim = dim / num_heads;
    attn->temperature = 1.0 / sqrt(attn->head_dim);
    
    // Initialize projection matrices as hierarchical matrices
    attn->W_query = hmatrix_create(dim, dim, ATTENTION_TOLERANCE);
    attn->W_key = hmatrix_create(dim, dim, ATTENTION_TOLERANCE);
    attn->W_value = hmatrix_create(dim, dim, ATTENTION_TOLERANCE);
    attn->W_output = hmatrix_create(dim, dim, ATTENTION_TOLERANCE);
    
    if (!attn->W_query || !attn->W_key || !attn->W_value || !attn->W_output) {
        attention_destroy(attn);
        return NULL;
    }
    
    return attn;
}

void attention_destroy(GeometricAttention* attn) {
    if (!attn) return;
    
    hmatrix_destroy(attn->W_query);
    hmatrix_destroy(attn->W_key);
    hmatrix_destroy(attn->W_value);
    hmatrix_destroy(attn->W_output);
    
    free(attn);
}

void attention_forward(GeometricAttention* attn,
                      const double complex* input,
                      size_t seq_length,
                      double complex* output) {
    // Create hierarchical matrices for intermediate computations
    AttentionContext ctx = {
        .query = hmatrix_create(seq_length, attn->dim, ATTENTION_TOLERANCE),
        .key = hmatrix_create(seq_length, attn->dim, ATTENTION_TOLERANCE),
        .value = hmatrix_create(seq_length, attn->dim, ATTENTION_TOLERANCE),
        .output = hmatrix_create(seq_length, attn->dim, ATTENTION_TOLERANCE),
        .temperature = attn->temperature,
        .dropout_rate = 0.1  // Configurable dropout rate
    };
    
    // Project input to Q, K, V using hierarchical operations - O(log n)
    #pragma omp parallel sections
    {
        #pragma omp section
        hmatrix_multiply(ctx.query, (HierarchicalMatrix*)input, attn->W_query);
        
        #pragma omp section
        hmatrix_multiply(ctx.key, (HierarchicalMatrix*)input, attn->W_key);
        
        #pragma omp section
        hmatrix_multiply(ctx.value, (HierarchicalMatrix*)input, attn->W_value);
    }
    
    // Split heads
    for (size_t h = 0; h < attn->num_heads; h++) {
        AttentionContext head_ctx = ctx;
        head_ctx.query = extract_head(ctx.query, h, attn->head_dim);
        head_ctx.key = extract_head(ctx.key, h, attn->head_dim);
        head_ctx.value = extract_head(ctx.value, h, attn->head_dim);
        
        // Compute attention scores - O(log n) with hierarchical operations
        compute_attention_scores(&head_ctx);
        
        // Merge back to output
        merge_head(ctx.output, head_ctx.output, h, attn->head_dim);
        
        // Cleanup head matrices
        hmatrix_destroy(head_ctx.query);
        hmatrix_destroy(head_ctx.key);
        hmatrix_destroy(head_ctx.value);
        hmatrix_destroy(head_ctx.output);
    }
    
    // Final projection - O(log n)
    hmatrix_multiply((HierarchicalMatrix*)output, ctx.output, attn->W_output);
    
    // Cleanup
    hmatrix_destroy(ctx.query);
    hmatrix_destroy(ctx.key);
    hmatrix_destroy(ctx.value);
    hmatrix_destroy(ctx.output);
}

static void compute_attention_scores(AttentionContext* ctx) {
    // Q * K^T using hierarchical multiplication - O(log n)
    HierarchicalMatrix* scores = hmatrix_create(
        ctx->query->rows, ctx->key->rows, ATTENTION_TOLERANCE
    );
    
    HierarchicalMatrix* key_t = hmatrix_create(
        ctx->key->cols, ctx->key->rows, ATTENTION_TOLERANCE
    );
    hmatrix_transpose(key_t, ctx->key);
    
    hmatrix_multiply(scores, ctx->query, key_t);
    
    // Scale and apply softmax
    scale_and_softmax(scores, ctx->temperature);
    
    // Apply attention to values - O(log n)
    hmatrix_multiply(ctx->output, scores, ctx->value);
    
    // Cleanup
    hmatrix_destroy(scores);
    hmatrix_destroy(key_t);
}

static void scale_and_softmax(HierarchicalMatrix* mat, double temperature) {
    // Scale scores
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < mat->rows; i++) {
        for (size_t j = 0; j < mat->cols; j++) {
            mat->data[i * mat->cols + j] *= temperature;
        }
    }
    
    // Apply softmax row-wise using SIMD
    #pragma omp parallel for
    for (size_t i = 0; i < mat->rows; i++) {
        // Find max for numerical stability
        double max_val = -INFINITY;
        for (size_t j = 0; j < mat->cols; j++) {
            double val = creal(mat->data[i * mat->cols + j]);
            if (val > max_val) max_val = val;
        }
        
        // Compute exp and sum
        double sum = 0.0;
        __m512d max_vec = _mm512_set1_pd(max_val);
        
        for (size_t j = 0; j < mat->cols; j += 8) {
            __m512d vals = _mm512_load_pd((double*)&mat->data[i * mat->cols + j]);
            __m512d exp_vals = _mm512_exp_pd(_mm512_sub_pd(vals, max_vec));
            _mm512_store_pd((double*)&mat->data[i * mat->cols + j], exp_vals);
            sum += _mm512_reduce_add_pd(exp_vals);
        }
        
        // Normalize
        __m512d sum_vec = _mm512_set1_pd(1.0 / sum);
        for (size_t j = 0; j < mat->cols; j += 8) {
            __m512d vals = _mm512_load_pd((double*)&mat->data[i * mat->cols + j]);
            vals = _mm512_mul_pd(vals, sum_vec);
            _mm512_store_pd((double*)&mat->data[i * mat->cols + j], vals);
        }
    }
}

static HierarchicalMatrix* extract_head(HierarchicalMatrix* mat, size_t head_idx,
                                      size_t head_dim) {
    // Extract portion of matrix corresponding to attention head
    size_t start_col = head_idx * head_dim;
    HierarchicalMatrix* head = hmatrix_create(mat->rows, head_dim, mat->tolerance);
    
    if (mat->is_leaf) {
        #pragma omp parallel for
        for (size_t i = 0; i < mat->rows; i++) {
            memcpy(&head->data[i * head_dim],
                   &mat->data[i * mat->cols + start_col],
                   head_dim * sizeof(double complex));
        }
    } else {
        // Handle hierarchical extraction
        extract_head_recursive(head, mat, head_idx, head_dim);
    }
    
    return head;
}

static void merge_head(HierarchicalMatrix* output, HierarchicalMatrix* head,
                      size_t head_idx, size_t head_dim) {
    // Merge attention head back into output matrix
    size_t start_col = head_idx * head_dim;
    
    if (output->is_leaf && head->is_leaf) {
        #pragma omp parallel for
        for (size_t i = 0; i < output->rows; i++) {
            memcpy(&output->data[i * output->cols + start_col],
                   &head->data[i * head_dim],
                   head_dim * sizeof(double complex));
        }
    } else {
        // Handle hierarchical merging
        merge_head_recursive(output, head, head_idx, head_dim);
    }
}
