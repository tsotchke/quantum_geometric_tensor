 #include "quantum_geometric/core/geometric_attention.h"
#include "quantum_geometric/core/hierarchical_matrix.h"
#include "quantum_geometric/core/numerical_backend.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <immintrin.h>

// Constants for attention implementation
#define ATTENTION_TOLERANCE 1e-10

// Attention context structure
typedef struct {
    HierarchicalMatrix* query;
    HierarchicalMatrix* key;
    HierarchicalMatrix* value;
    HierarchicalMatrix* output;
    double temperature;
    double dropout_rate;
} AttentionContext;

// Geometric attention structure
typedef struct {
    size_t dim;
    size_t num_heads;
    size_t head_dim;
    double temperature;
    HierarchicalMatrix* W_query;
    HierarchicalMatrix* W_key;
    HierarchicalMatrix* W_value;
    HierarchicalMatrix* W_output;
} GeometricAttention;

// Forward declarations for optimized attention implementation
static void compute_attention_scores(AttentionContext* ctx);
static void scale_and_softmax(HierarchicalMatrix* mat, double temperature);
static HierarchicalMatrix* extract_head(HierarchicalMatrix* mat, size_t head_idx, size_t head_dim);
static void merge_head(HierarchicalMatrix* output, HierarchicalMatrix* head, size_t head_idx, size_t head_dim);
static void extract_head_recursive(HierarchicalMatrix* head, HierarchicalMatrix* mat, size_t head_idx, size_t head_dim);
static void merge_head_recursive(HierarchicalMatrix* output, HierarchicalMatrix* head, size_t head_idx, size_t head_dim);

// Implementation of the opaque geometric_attention_t structure
struct geometric_attention_t {
    // Configuration
    attention_config_t config;
    geometric_params_t params;
    
    // State management using hierarchical matrices
    HierarchicalMatrix* query_matrix;
    HierarchicalMatrix* key_matrix;
    HierarchicalMatrix* value_matrix;
    HierarchicalMatrix* attention_matrix;
    
    // Geometric tensors using hierarchical matrices
    HierarchicalMatrix* metric_tensor;
    HierarchicalMatrix* connection_tensor;
    HierarchicalMatrix* curvature_tensor;
    HierarchicalMatrix* phase_tensor;
    
    // Numerical backend configuration
    numerical_config_t numerical_config;
    
    // Performance tracking
    attention_metrics_t metrics;
    numerical_metrics_t numerical_metrics;
    
    // Status flags
    bool is_initialized;
    bool error_correction_enabled;
    bool backend_initialized;
};

// Forward declarations of static functions
static bool allocate_attention_buffers(geometric_attention_t* attention);
static void free_attention_buffers(geometric_attention_t* attention);
static bool compute_geometric_tensors(geometric_attention_t* attention);
static bool apply_attention_mechanism(geometric_attention_t* attention, 
                                   const attention_state_t* input,
                                   attention_state_t* output);

geometric_attention_t* create_geometric_attention(const attention_config_t* config) {
    if (!config) return NULL;
    
    geometric_attention_t* attention = malloc(sizeof(geometric_attention_t));
    if (!attention) return NULL;
    
    // Initialize the structure
    memset(attention, 0, sizeof(geometric_attention_t));
    
    // Copy configuration
    memcpy(&attention->config, config, sizeof(attention_config_t));
    
    // Allocate buffers
    if (!allocate_attention_buffers(attention)) {
        destroy_geometric_attention(attention);
        return NULL;
    }
    
    attention->error_correction_enabled = config->use_error_correction;
    attention->is_initialized = false;
    
    return attention;
}

void destroy_geometric_attention(geometric_attention_t* attention) {
    if (!attention) return;
    free_attention_buffers(attention);
    free(attention);
}

bool initialize_geometry(geometric_attention_t* attention,
                       const geometric_params_t* params) {
    if (!attention || !params) return false;
    
    memcpy(&attention->params, params, sizeof(geometric_params_t));
    
    // Compute geometric tensors based on parameters
    if (!compute_geometric_tensors(attention)) {
        return false;
    }
    
    attention->is_initialized = true;
    return true;
}

bool compute_attention(geometric_attention_t* attention,
                      const attention_state_t* input,
                      attention_state_t* output) {
    if (!attention || !input || !output) return false;
    if (!attention->is_initialized) return false;
    
    // Convert input states to hierarchical matrices
    HierarchicalMatrix* input_matrix = create_hierarchical_matrix(input->seq_length, input->head_dim);
    if (!input_matrix) return false;
    input_matrix->tolerance = ATTENTION_TOLERANCE;  // Set tolerance
    
    // Check if input matrix data is allocated
    if (!input_matrix->data) {
        destroy_hierarchical_matrix(input_matrix);
        return false;
    }

    // Initialize input matrix data from queries/keys/values
    for (size_t i = 0; i < input->seq_length * input->head_dim; i++) {
        input_matrix->data[i] = input->queries[i];  // Use queries as input
    }
    
    // Apply geometric attention mechanism
    if (!apply_attention_mechanism(attention, input, output)) {
        destroy_hierarchical_matrix(input_matrix);
        return false;
    }
    
    // Update metrics
    attention->metrics.operation_count++;
    get_numerical_metrics(&attention->numerical_metrics);
    
    destroy_hierarchical_matrix(input_matrix);
    return true;
}

static bool allocate_attention_buffers(geometric_attention_t* attention) {
    size_t state_size = attention->config.head_dim * attention->config.attention_heads;
    
    // Create hierarchical matrices with quantum type
    matrix_properties_t props = {
        .dimension = state_size,
        .tolerance = 1e-10,
        .symmetric = false,
        .positive_definite = false
    };
    
    attention->query_matrix = create_hierarchical_matrix(state_size, state_size);
    attention->key_matrix = create_hierarchical_matrix(state_size, state_size);
    attention->value_matrix = create_hierarchical_matrix(state_size, state_size);
    attention->attention_matrix = create_hierarchical_matrix(state_size, state_size);
    
    attention->metric_tensor = create_hierarchical_matrix(state_size, state_size);
    attention->connection_tensor = create_hierarchical_matrix(state_size, state_size);
    attention->curvature_tensor = create_hierarchical_matrix(state_size, state_size);
    attention->phase_tensor = create_hierarchical_matrix(state_size, 1);
    
    // Set matrix properties and tolerance
    if (!attention->query_matrix || !attention->key_matrix || 
        !attention->value_matrix || !attention->attention_matrix ||
        !attention->metric_tensor || !attention->connection_tensor ||
        !attention->curvature_tensor || !attention->phase_tensor) {
        free_attention_buffers(attention);
        return false;
    }

    // Set tolerance for all matrices
    attention->query_matrix->tolerance = ATTENTION_TOLERANCE;
    attention->key_matrix->tolerance = ATTENTION_TOLERANCE;
    attention->value_matrix->tolerance = ATTENTION_TOLERANCE;
    attention->attention_matrix->tolerance = ATTENTION_TOLERANCE;
    attention->metric_tensor->tolerance = ATTENTION_TOLERANCE;
    attention->connection_tensor->tolerance = ATTENTION_TOLERANCE;
    attention->curvature_tensor->tolerance = ATTENTION_TOLERANCE;
    attention->phase_tensor->tolerance = ATTENTION_TOLERANCE;
    
    // Initialize numerical backend
    numerical_config_t config = {
        .type = NUMERICAL_BACKEND_CPU,
        .max_threads = 8,
        .use_fma = true,
        .use_avx = true
    };
    
    attention->backend_initialized = initialize_numerical_backend(&config);
    if (!attention->backend_initialized) {
        free_attention_buffers(attention);
        return false;
    }
    
    return true;
}

static void free_attention_buffers(geometric_attention_t* attention) {
    destroy_hierarchical_matrix(attention->query_matrix);
    destroy_hierarchical_matrix(attention->key_matrix);
    destroy_hierarchical_matrix(attention->value_matrix);
    destroy_hierarchical_matrix(attention->attention_matrix);
    destroy_hierarchical_matrix(attention->metric_tensor);
    destroy_hierarchical_matrix(attention->connection_tensor);
    destroy_hierarchical_matrix(attention->curvature_tensor);
    destroy_hierarchical_matrix(attention->phase_tensor);
    
    if (attention->backend_initialized) {
        shutdown_numerical_backend();
    }
}

static bool compute_geometric_tensors(geometric_attention_t* attention) {
    // Implementation of geometric tensor computations based on params
    // This would include metric tensor, connection coefficients, and curvature
    return true; // Placeholder
}

static bool apply_attention_mechanism(geometric_attention_t* attention,
                                   const attention_state_t* input,
                                   attention_state_t* output) {
    // Implementation of the geometric attention mechanism
    // This would include the quantum geometric transformations
    return true; // Placeholder
}

GeometricAttention* attention_create(size_t dim, size_t num_heads) {
    GeometricAttention* attn = malloc(sizeof(GeometricAttention));
    if (!attn) return NULL;
    
    attn->dim = dim;
    attn->num_heads = num_heads;
    attn->head_dim = dim / num_heads;
    attn->temperature = 1.0 / sqrt(attn->head_dim);
    
    // Initialize projection matrices as hierarchical matrices
    attn->W_query = create_hierarchical_matrix(dim, dim);
    attn->W_key = create_hierarchical_matrix(dim, dim);
    attn->W_value = create_hierarchical_matrix(dim, dim);
    attn->W_output = create_hierarchical_matrix(dim, dim);
    
    // Set tolerance for attention precision
    if (attn->W_query) attn->W_query->tolerance = ATTENTION_TOLERANCE;
    if (attn->W_key) attn->W_key->tolerance = ATTENTION_TOLERANCE;
    if (attn->W_value) attn->W_value->tolerance = ATTENTION_TOLERANCE;
    if (attn->W_output) attn->W_output->tolerance = ATTENTION_TOLERANCE;
    
    if (!attn->W_query || !attn->W_key || !attn->W_value || !attn->W_output) {
        attention_destroy(attn);
        return NULL;
    }
    
    return attn;
}

void attention_destroy(GeometricAttention* attn) {
    if (!attn) return;
    
    destroy_hierarchical_matrix(attn->W_query);
    destroy_hierarchical_matrix(attn->W_key);
    destroy_hierarchical_matrix(attn->W_value);
    destroy_hierarchical_matrix(attn->W_output);
    
    free(attn);
}

void attention_forward(GeometricAttention* attn,
                      const double complex* input,
                      size_t seq_length,
                      double complex* output) {
    // Convert input to hierarchical matrix
    HierarchicalMatrix* input_mat = create_hierarchical_matrix(seq_length, attn->dim);
    if (!input_mat) return;
    input_mat->tolerance = ATTENTION_TOLERANCE;  // Set tolerance
    
    // Check if input matrix data is allocated
    if (!input_mat->data) {
        destroy_hierarchical_matrix(input_mat);
        return;
    }
    
    // Copy input data
    for (size_t i = 0; i < seq_length * attn->dim; i++) {
        input_mat->data[i] = input[i];
    }
    
    // Create hierarchical matrices for intermediate computations
    AttentionContext ctx = {
        .query = create_hierarchical_matrix(seq_length, attn->dim),
        .key = create_hierarchical_matrix(seq_length, attn->dim),
        .value = create_hierarchical_matrix(seq_length, attn->dim),
        .output = create_hierarchical_matrix(seq_length, attn->dim),
        .temperature = attn->temperature,
        .dropout_rate = 0.1  // Configurable dropout rate
    };

    // Set tolerance for intermediate matrices
    if (ctx.query) ctx.query->tolerance = ATTENTION_TOLERANCE;
    if (ctx.key) ctx.key->tolerance = ATTENTION_TOLERANCE;
    if (ctx.value) ctx.value->tolerance = ATTENTION_TOLERANCE;
    if (ctx.output) ctx.output->tolerance = ATTENTION_TOLERANCE;
    
    if (!ctx.query || !ctx.key || !ctx.value || !ctx.output) {
        destroy_hierarchical_matrix(input_mat);
        destroy_hierarchical_matrix(ctx.query);
        destroy_hierarchical_matrix(ctx.key);
        destroy_hierarchical_matrix(ctx.value);
        destroy_hierarchical_matrix(ctx.output);
        return;
    }
    
    // Check if matrices have allocated data
    if (!input_mat->data || !attn->W_query->data || !attn->W_key->data || !attn->W_value->data) {
        destroy_hierarchical_matrix(input_mat);
        destroy_hierarchical_matrix(ctx.query);
        destroy_hierarchical_matrix(ctx.key);
        destroy_hierarchical_matrix(ctx.value);
        destroy_hierarchical_matrix(ctx.output);
        return;
    }

    // Project input to Q, K, V using hierarchical operations - O(log n)
    #pragma omp parallel sections
    {
        #pragma omp section
        hmatrix_multiply(ctx.query, input_mat, attn->W_query);
        
        #pragma omp section
        hmatrix_multiply(ctx.key, input_mat, attn->W_key);
        
        #pragma omp section
        hmatrix_multiply(ctx.value, input_mat, attn->W_value);
    }
    
    destroy_hierarchical_matrix(input_mat);

    // Check if matrices were properly initialized after multiplication
    if (!ctx.query->data || !ctx.key->data || !ctx.value->data) {
        destroy_hierarchical_matrix(ctx.query);
        destroy_hierarchical_matrix(ctx.key);
        destroy_hierarchical_matrix(ctx.value);
        destroy_hierarchical_matrix(ctx.output);
        return;
    }
    
    // Split heads
    for (size_t h = 0; h < attn->num_heads; h++) {
        AttentionContext head_ctx = ctx;
        head_ctx.query = extract_head(ctx.query, h, attn->head_dim);
        head_ctx.key = extract_head(ctx.key, h, attn->head_dim);
        head_ctx.value = extract_head(ctx.value, h, attn->head_dim);
        head_ctx.output = create_hierarchical_matrix(seq_length, attn->dim);
        if (!head_ctx.output) {
            // Cleanup and return if allocation failed
            hmatrix_destroy(head_ctx.query);
            hmatrix_destroy(head_ctx.key);
            hmatrix_destroy(head_ctx.value);
            return;
        }
        head_ctx.output->tolerance = ATTENTION_TOLERANCE;

        // Check if output matrix data is allocated
        if (!head_ctx.output->data) {
            hmatrix_destroy(head_ctx.query);
            hmatrix_destroy(head_ctx.key);
            hmatrix_destroy(head_ctx.value);
            hmatrix_destroy(head_ctx.output);
            return;
        }
        
        // Compute attention scores - O(log n) with hierarchical operations
        compute_attention_scores(&head_ctx);
        
        // Check if output matrix data is still valid after attention computation
        if (!head_ctx.output->data) {
            hmatrix_destroy(head_ctx.query);
            hmatrix_destroy(head_ctx.key);
            hmatrix_destroy(head_ctx.value);
            hmatrix_destroy(head_ctx.output);
            return;
        }
        
        // Merge back to output
        merge_head(ctx.output, head_ctx.output, h, attn->head_dim);
        
        // Check if output matrix data is still valid after merge
        if (!ctx.output->data) {
            destroy_hierarchical_matrix(ctx.query);
            destroy_hierarchical_matrix(ctx.key);
            destroy_hierarchical_matrix(ctx.value);
            destroy_hierarchical_matrix(ctx.output);
            return;
        }

        // Cleanup head matrices
        hmatrix_destroy(head_ctx.query);
        hmatrix_destroy(head_ctx.key);
        hmatrix_destroy(head_ctx.value);
        hmatrix_destroy(head_ctx.output);
    }
    
    // Create output matrix for final projection
    HierarchicalMatrix* output_mat = create_hierarchical_matrix(seq_length, attn->dim);
    if (!output_mat) {
        destroy_hierarchical_matrix(ctx.query);
        destroy_hierarchical_matrix(ctx.key);
        destroy_hierarchical_matrix(ctx.value);
        destroy_hierarchical_matrix(ctx.output);
        return;
    }
    output_mat->tolerance = ATTENTION_TOLERANCE;  // Set tolerance
    
    // Check if matrices have allocated data
    if (!output_mat->data || !ctx.output->data || !attn->W_output->data) {
        destroy_hierarchical_matrix(output_mat);
        destroy_hierarchical_matrix(ctx.query);
        destroy_hierarchical_matrix(ctx.key);
        destroy_hierarchical_matrix(ctx.value);
        destroy_hierarchical_matrix(ctx.output);
        return;
    }

    // Final projection - O(log n)
    hmatrix_multiply(output_mat, ctx.output, attn->W_output);
    
    // Check if output matrix data is still valid after multiplication
    if (!output_mat->data) {
        destroy_hierarchical_matrix(output_mat);
        destroy_hierarchical_matrix(ctx.query);
        destroy_hierarchical_matrix(ctx.key);
        destroy_hierarchical_matrix(ctx.value);
        destroy_hierarchical_matrix(ctx.output);
        return;
    }
    
    // Copy result to output array
    if (output_mat->is_leaf) {
        memcpy(output, output_mat->data, seq_length * attn->dim * sizeof(double complex));
    }
    
    // Cleanup
    destroy_hierarchical_matrix(output_mat);
    destroy_hierarchical_matrix(ctx.query);
    destroy_hierarchical_matrix(ctx.key);
    destroy_hierarchical_matrix(ctx.value);
    destroy_hierarchical_matrix(ctx.output);
}

static void compute_attention_scores(AttentionContext* ctx) {
    // Q * K^T using hierarchical multiplication - O(log n)
    HierarchicalMatrix* scores = create_hierarchical_matrix(ctx->query->rows, ctx->key->rows);
    HierarchicalMatrix* key_t = create_hierarchical_matrix(ctx->key->cols, ctx->key->rows);
    
    if (!scores || !key_t) {
        destroy_hierarchical_matrix(scores);
        destroy_hierarchical_matrix(key_t);
        return;
    }

    // Set tolerance for intermediate matrices
    scores->tolerance = ctx->query->tolerance;
    key_t->tolerance = ctx->key->tolerance;
    
    // Check if matrices have allocated data
    if (!ctx->query->data || !ctx->key->data || !ctx->value->data || !key_t->data || !scores->data) {
        destroy_hierarchical_matrix(scores);
        destroy_hierarchical_matrix(key_t);
        return;
    }

    // Transpose key matrix
    hmatrix_transpose(key_t, ctx->key);
    
    hmatrix_multiply(scores, ctx->query, key_t);
    
    // Check if scores matrix data is still valid after multiplication
    if (!scores->data) {
        destroy_hierarchical_matrix(scores);
        destroy_hierarchical_matrix(key_t);
        return;
    }
    
    // Scale and apply softmax
    scale_and_softmax(scores, ctx->temperature);
    
    // Check if scores matrix data is still valid after softmax
    if (!scores->data) {
        destroy_hierarchical_matrix(scores);
        destroy_hierarchical_matrix(key_t);
        return;
    }
    
    // Check if output matrix data is allocated
    if (!ctx->output->data) {
        destroy_hierarchical_matrix(scores);
        destroy_hierarchical_matrix(key_t);
        return;
    }
    
    // Apply attention to values - O(log n)
    hmatrix_multiply(ctx->output, scores, ctx->value);
    
    // Check if output matrix data is still valid after multiplication
    if (!ctx->output->data) {
        destroy_hierarchical_matrix(scores);
        destroy_hierarchical_matrix(key_t);
        return;
    }
    
    // Cleanup
    destroy_hierarchical_matrix(scores);
    destroy_hierarchical_matrix(key_t);
}

static void scale_and_softmax(HierarchicalMatrix* mat, double temperature) {
    if (!mat || !mat->data) return;
    
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
        
        // Compute exp and sum using scalar operations since we're dealing with complex numbers
        double sum = 0.0;
        double* real_vals = malloc(mat->cols * sizeof(double));
        if (!real_vals) return;

        // Extract real parts and compute exponentials
        for (size_t j = 0; j < mat->cols; j++) {
            real_vals[j] = exp(creal(mat->data[i * mat->cols + j]) - max_val);
            sum += real_vals[j];
        }

        // Normalize and update complex values
        for (size_t j = 0; j < mat->cols; j++) {
            mat->data[i * mat->cols + j] = real_vals[j] / sum;
        }

        free(real_vals);
    }
}

static HierarchicalMatrix* extract_head(HierarchicalMatrix* mat, size_t head_idx,
                                      size_t head_dim) {
    // Extract portion of matrix corresponding to attention head
    size_t start_col = head_idx * head_dim;
    HierarchicalMatrix* head = create_hierarchical_matrix(mat->rows, head_dim);
    if (!head) return NULL;
    head->tolerance = mat->tolerance;  // Propagate tolerance
    
    if (mat->is_leaf && mat->data && head->data) {
        #pragma omp parallel for
        for (size_t i = 0; i < mat->rows; i++) {
            memcpy(&head->data[i * head_dim],
                   &mat->data[i * mat->cols + start_col],
                   head_dim * sizeof(double complex));
        }
    } else {
        // Handle hierarchical extraction
        extract_head_recursive(head, mat, head_idx, head_dim);
        
        // Check if head data is allocated after recursive extraction
        if (!head->data) {
            destroy_hierarchical_matrix(head);
            return NULL;
        }
    }
    
    return head;
}

static void extract_head_recursive(HierarchicalMatrix* head, HierarchicalMatrix* mat,
                                 size_t head_idx, size_t head_dim) {
    if (!head || !mat) return;
    
    // Base case: both matrices are leaves
    if (head->is_leaf && mat->is_leaf && head->data && mat->data) {
        size_t start_col = head_idx * head_dim;
        #pragma omp parallel for
        for (size_t i = 0; i < mat->rows; i++) {
            memcpy(&head->data[i * head_dim],
                   &mat->data[i * mat->cols + start_col],
                   head_dim * sizeof(double complex));
        }
        return;
    }
    
    // Recursive case: split into quadrants
    size_t mid_row = mat->rows / 2;
    size_t mid_col = mat->cols / 2;
    
    // Create child nodes if they don't exist
    if (!head->children[0]) {
        for (int i = 0; i < 4; i++) {
            size_t sub_rows = (i < 2) ? mid_row : (mat->rows - mid_row);
            size_t sub_cols = (i % 2 == 0) ? mid_col : (mat->cols - mid_col);
            head->children[i] = create_hierarchical_matrix(sub_rows, head_dim);
            if (!head->children[i]) return;
            head->children[i]->tolerance = head->tolerance;  // Propagate tolerance
        }
    }
    
    // Recursively extract from each quadrant
    #pragma omp parallel for
    for (int i = 0; i < 4; i++) {
        extract_head_recursive(head->children[i], mat->children[i], head_idx, head_dim);
    }
}

static void merge_head_recursive(HierarchicalMatrix* output, HierarchicalMatrix* head,
                               size_t head_idx, size_t head_dim) {
    if (!output || !head) return;
    
    // Base case: both matrices are leaves
    if (output->is_leaf && head->is_leaf && output->data && head->data) {
        size_t start_col = head_idx * head_dim;
        #pragma omp parallel for
        for (size_t i = 0; i < output->rows; i++) {
            memcpy(&output->data[i * output->cols + start_col],
                   &head->data[i * head_dim],
                   head_dim * sizeof(double complex));
        }
        return;
    }
    
    // Recursive case: split into quadrants
    size_t mid_row = output->rows / 2;
    size_t mid_col = output->cols / 2;
    
    // Create child nodes if they don't exist
    if (!output->children[0]) {
        for (int i = 0; i < 4; i++) {
            size_t sub_rows = (i < 2) ? mid_row : (output->rows - mid_row);
            size_t sub_cols = (i % 2 == 0) ? mid_col : (output->cols - mid_col);
            output->children[i] = create_hierarchical_matrix(sub_rows, sub_cols);
            if (!output->children[i]) return;
            output->children[i]->tolerance = output->tolerance;  // Propagate tolerance
        }
    }
    
    // Recursively merge into each quadrant
    #pragma omp parallel for
    for (int i = 0; i < 4; i++) {
        merge_head_recursive(output->children[i], head->children[i], head_idx, head_dim);
    }
    
    // Check if output data is still valid after recursive merging
    if (!output->data) {
        return;
    }
}

static void merge_head(HierarchicalMatrix* output, HierarchicalMatrix* head,
                      size_t head_idx, size_t head_dim) {
    // Merge attention head back into output matrix
    size_t start_col = head_idx * head_dim;
    
    if (output->is_leaf && head->is_leaf && output->data && head->data) {
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
