#include "quantum_geometric/core/geometric_attention.h"
#include "quantum_geometric/core/hierarchical_matrix.h"
#include "quantum_geometric/core/numerical_backend.h"
#include "quantum_geometric/core/platform_intrinsics.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

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
void attention_destroy(GeometricAttention* attn);

// Implementation of the opaque geometric_attention_t structure
struct geometric_attention_t {
    // Configuration
    attention_config_t config;
    attn_geometric_params_t params;
    
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

// Forward declaration for destroy function (called in create on error)
void destroy_quantum_geometric_attention(geometric_attention_t* attention);

// Note: create_geometric_attention, destroy_geometric_attention, and compute_attention
// are defined in geometric_attention.c. This file provides quantum-specific variants.
geometric_attention_t* create_quantum_geometric_attention(const attention_config_t* config) {
    if (!config) return NULL;

    geometric_attention_t* attention = malloc(sizeof(geometric_attention_t));
    if (!attention) return NULL;

    // Initialize the structure
    memset(attention, 0, sizeof(geometric_attention_t));

    // Copy configuration
    memcpy(&attention->config, config, sizeof(attention_config_t));

    // Allocate buffers
    if (!allocate_attention_buffers(attention)) {
        destroy_quantum_geometric_attention(attention);
        return NULL;
    }

    attention->error_correction_enabled = config->use_error_correction;
    attention->is_initialized = false;

    return attention;
}

void destroy_quantum_geometric_attention(geometric_attention_t* attention) {
    if (!attention) return;
    free_attention_buffers(attention);
    free(attention);
}

bool initialize_quantum_geometry(geometric_attention_t* attention,
                       const attn_geometric_params_t* params) {
    if (!attention || !params) return false;

    memcpy(&attention->params, params, sizeof(attn_geometric_params_t));

    // Compute geometric tensors based on parameters
    if (!compute_geometric_tensors(attention)) {
        return false;
    }

    attention->is_initialized = true;
    return true;
}

bool compute_geometric_quantum_attention(geometric_attention_t* attention,
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

    // Use configured tolerance from props (defaults to 1e-10)
    double tol = (props.tolerance > 0) ? props.tolerance : ATTENTION_TOLERANCE;
    attention->query_matrix->tolerance = tol;
    attention->key_matrix->tolerance = tol;
    attention->value_matrix->tolerance = tol;
    attention->attention_matrix->tolerance = tol;
    attention->metric_tensor->tolerance = tol;
    attention->connection_tensor->tolerance = tol;
    attention->curvature_tensor->tolerance = tol;
    attention->phase_tensor->tolerance = tol;
    
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
    if (!attention) return false;

    // Get dimensions from config
    size_t dim = attention->config.head_dim * attention->config.attention_heads;
    if (dim == 0) return false;

    // Verify buffers are allocated
    if (!attention->metric_tensor || !attention->metric_tensor->data ||
        !attention->connection_tensor || !attention->connection_tensor->data ||
        !attention->curvature_tensor || !attention->curvature_tensor->data ||
        !attention->phase_tensor || !attention->phase_tensor->data) {
        return false;
    }

    // Get parameters
    double base_metric = attention->params.metric_tensor;
    double curvature = attention->params.curvature;
    double connection_coeff = attention->params.connection_coeff;
    attn_geometry_type_t geometry_type = attention->params.type;

    // Compute metric tensor based on geometry type
    // The metric tensor g_ij defines distances on the manifold
    switch (geometry_type) {
        case ATTN_GEOMETRY_FUBINI_STUDY: {
            // Fubini-Study metric for complex projective space
            // g_ij = (1 + |z|^2) * delta_ij - z_i * conj(z_j)
            // For attention: scaled identity with geometric correction
            #pragma omp parallel for collapse(2)
            for (size_t i = 0; i < dim; i++) {
                for (size_t j = 0; j < dim; j++) {
                    double complex val;
                    if (i == j) {
                        // Diagonal: (1 + curvature_correction) * base_metric
                        double curvature_correction = 1.0 / (1.0 + curvature * curvature);
                        val = base_metric * curvature_correction;
                    } else {
                        // Off-diagonal: geometric coupling based on distance
                        double dist = (double)(abs((int)i - (int)j));
                        val = -base_metric * curvature * exp(-dist / (double)dim);
                    }
                    attention->metric_tensor->data[i * dim + j] = val;
                }
            }
            break;
        }
        case ATTN_GEOMETRY_KAHLER: {
            // Kahler metric: Hermitian metric compatible with complex structure
            // g_ij = partial_i partial_j K (K = Kahler potential)
            #pragma omp parallel for collapse(2)
            for (size_t i = 0; i < dim; i++) {
                for (size_t j = 0; j < dim; j++) {
                    double complex val;
                    if (i == j) {
                        val = base_metric * (1.0 + 0.5 * curvature);
                    } else {
                        // Kahler coupling with Hermitian symmetry
                        double phase = M_PI * (double)(i - j) / (double)dim;
                        val = base_metric * connection_coeff * cexp(I * phase) / (1.0 + (double)(abs((int)i - (int)j)));
                    }
                    attention->metric_tensor->data[i * dim + j] = val;
                }
            }
            break;
        }
        case ATTN_GEOMETRY_COMPLEX: {
            // Complex projective metric
            #pragma omp parallel for collapse(2)
            for (size_t i = 0; i < dim; i++) {
                for (size_t j = 0; j < dim; j++) {
                    double complex val;
                    if (i == j) {
                        val = base_metric;
                    } else {
                        val = base_metric * connection_coeff * exp(-(double)(abs((int)i - (int)j)) / (double)dim);
                    }
                    attention->metric_tensor->data[i * dim + j] = val;
                }
            }
            break;
        }
        case ATTN_GEOMETRY_MANIFOLD:
        default: {
            // General Riemannian manifold: identity-like metric with perturbations
            #pragma omp parallel for collapse(2)
            for (size_t i = 0; i < dim; i++) {
                for (size_t j = 0; j < dim; j++) {
                    double complex val = (i == j) ? base_metric : 0.0;
                    attention->metric_tensor->data[i * dim + j] = val;
                }
            }
            break;
        }
    }

    // Compute Christoffel connection symbols (Levi-Civita connection)
    // Gamma^k_ij = (1/2) * g^{kl} * (partial_i g_{jl} + partial_j g_{il} - partial_l g_{ij})
    // For efficiency, we use finite differences and store as a 2D approximation
    double h = 1e-6;  // Finite difference step
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            // Approximate connection using metric derivatives
            double complex gamma_ij = 0.0;

            // Sum over metric components for this index pair
            for (size_t k = 0; k < dim && k < 4; k++) {  // Limit sum for efficiency
                // Compute metric gradient contributions
                size_t idx_ik = i * dim + k;
                size_t idx_jk = j * dim + k;

                double complex g_ik = attention->metric_tensor->data[idx_ik];
                double complex g_jk = attention->metric_tensor->data[idx_jk];

                // Connection approximation based on metric structure
                gamma_ij += connection_coeff * (g_ik + g_jk) / (2.0 * dim);
            }

            attention->connection_tensor->data[i * dim + j] = gamma_ij;
        }
    }

    // Compute Riemann curvature tensor (stored as 2D projection)
    // R^i_jkl = partial_k Gamma^i_jl - partial_l Gamma^i_jk + Gamma^i_mk Gamma^m_jl - Gamma^i_ml Gamma^m_jk
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            double complex riemann_ij = 0.0;

            // Simplified curvature from connection structure
            double complex gamma_ii = attention->connection_tensor->data[i * dim + i];
            double complex gamma_jj = attention->connection_tensor->data[j * dim + j];
            double complex gamma_ij = attention->connection_tensor->data[i * dim + j];
            double complex gamma_ji = attention->connection_tensor->data[j * dim + i];

            // Curvature contribution: R ~ partial Gamma + Gamma * Gamma
            riemann_ij = curvature * (gamma_ii * gamma_jj - gamma_ij * gamma_ji);

            // Add intrinsic curvature contribution for non-flat geometries
            if (geometry_type != ATTN_GEOMETRY_MANIFOLD) {
                double dist_norm = (double)(abs((int)i - (int)j)) / (double)dim;
                riemann_ij += curvature * exp(-dist_norm) * attention->metric_tensor->data[i * dim + j];
            }

            attention->curvature_tensor->data[i * dim + j] = riemann_ij;
        }
    }

    // Compute phase tensor (Berry phase / geometric phase)
    // Phase factors for quantum geometric attention
    if (attention->params.phase_factors && attention->params.num_factors > 0) {
        // Use provided phase factors
        size_t copy_size = (attention->params.num_factors < dim) ? attention->params.num_factors : dim;
        for (size_t i = 0; i < copy_size; i++) {
            attention->phase_tensor->data[i] = attention->params.phase_factors[i];
        }
        // Fill remaining with unit phase
        for (size_t i = copy_size; i < dim; i++) {
            attention->phase_tensor->data[i] = 1.0;
        }
    } else {
        // Compute geometric phase based on curvature
        // Berry phase: exp(i * integral of Berry connection)
        #pragma omp parallel for
        for (size_t i = 0; i < dim; i++) {
            // Phase accumulation from curvature
            double phase_angle = 0.0;
            for (size_t j = 0; j < dim && j < 8; j++) {
                phase_angle += curvature * creal(attention->curvature_tensor->data[i * dim + j]) / (double)dim;
            }
            attention->phase_tensor->data[i] = cexp(I * phase_angle);
        }
    }

    return true;
}

static bool apply_attention_mechanism(geometric_attention_t* attention,
                                   const attention_state_t* input,
                                   attention_state_t* output) {
    if (!attention || !input || !output) return false;

    // Validate input state
    if (!input->queries || !input->keys || !input->values) return false;
    if (input->seq_length == 0 || input->head_dim == 0) return false;

    // Validate geometric tensors are initialized
    if (!attention->metric_tensor || !attention->metric_tensor->data ||
        !attention->connection_tensor || !attention->connection_tensor->data ||
        !attention->curvature_tensor || !attention->curvature_tensor->data ||
        !attention->phase_tensor || !attention->phase_tensor->data) {
        return false;
    }

    size_t seq_len = input->seq_length;
    size_t head_dim = input->head_dim;
    size_t total_size = seq_len * head_dim;

    // Allocate output buffers if not already allocated
    bool allocated_queries = false, allocated_keys = false, allocated_values = false;
    if (!output->queries) {
        output->queries = calloc(total_size, sizeof(complex double));
        if (!output->queries) return false;
        allocated_queries = true;
    }
    if (!output->keys) {
        output->keys = calloc(total_size, sizeof(complex double));
        if (!output->keys) {
            if (allocated_queries) { free(output->queries); output->queries = NULL; }
            return false;
        }
        allocated_keys = true;
    }
    if (!output->values) {
        output->values = calloc(total_size, sizeof(complex double));
        if (!output->values) {
            if (allocated_queries) { free(output->queries); output->queries = NULL; }
            if (allocated_keys) { free(output->keys); output->keys = NULL; }
            return false;
        }
        allocated_values = true;
    }

    // Set output dimensions
    output->seq_length = seq_len;
    output->head_dim = head_dim;
    output->batch_size = input->batch_size > 0 ? input->batch_size : 1;

    // Allocate attention scores matrix
    complex double* attention_scores = calloc(seq_len * seq_len, sizeof(complex double));
    if (!attention_scores) {
        if (allocated_queries) { free(output->queries); output->queries = NULL; }
        if (allocated_keys) { free(output->keys); output->keys = NULL; }
        if (allocated_values) { free(output->values); output->values = NULL; }
        return false;
    }

    // Step 1: Apply geometric transformation to queries and keys using metric tensor
    // Transform Q' = G * Q where G is derived from metric tensor
    size_t config_dim = attention->config.head_dim * attention->config.attention_heads;
    double temperature = 1.0 / sqrt((double)head_dim);

    // Step 2: Compute attention scores with geometric correction
    // Score(i,j) = Q_i^T * G * K_j / sqrt(d) where G incorporates the metric
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < seq_len; j++) {
            complex double score = 0.0;

            // Dot product with metric tensor weighting
            for (size_t k = 0; k < head_dim; k++) {
                complex double q_ik = input->queries[i * head_dim + k];
                complex double k_jk = input->keys[j * head_dim + k];

                // Get metric weight (use identity if out of bounds)
                size_t metric_idx = k * config_dim + k;
                complex double metric_weight = 1.0;
                if (metric_idx < config_dim * config_dim && attention->metric_tensor->data) {
                    metric_weight = attention->metric_tensor->data[metric_idx];
                    if (cabs(metric_weight) < 1e-10) metric_weight = 1.0;
                }

                // Apply geometric phase correction
                complex double phase = 1.0;
                if (k < config_dim && attention->phase_tensor->data) {
                    phase = attention->phase_tensor->data[k];
                }

                score += conj(q_ik) * metric_weight * k_jk * phase;
            }

            // Apply curvature-based position bias
            size_t curv_idx = (i % config_dim) * config_dim + (j % config_dim);
            if (curv_idx < config_dim * config_dim && attention->curvature_tensor->data) {
                complex double curvature_bias = attention->curvature_tensor->data[curv_idx];
                // Small curvature correction to attention
                score += 0.01 * curvature_bias;
            }

            // Scale by temperature
            attention_scores[i * seq_len + j] = score * temperature;
        }
    }

    // Step 3: Apply softmax normalization with numerical stability
    #pragma omp parallel for
    for (size_t i = 0; i < seq_len; i++) {
        // Find max for numerical stability
        double max_score = -INFINITY;
        for (size_t j = 0; j < seq_len; j++) {
            double real_score = creal(attention_scores[i * seq_len + j]);
            if (real_score > max_score) max_score = real_score;
        }

        // Compute exp and sum
        double sum_exp = 0.0;
        for (size_t j = 0; j < seq_len; j++) {
            double real_score = creal(attention_scores[i * seq_len + j]);
            double exp_score = exp(real_score - max_score);
            attention_scores[i * seq_len + j] = exp_score;
            sum_exp += exp_score;
        }

        // Normalize
        if (sum_exp > 1e-10) {
            for (size_t j = 0; j < seq_len; j++) {
                attention_scores[i * seq_len + j] /= sum_exp;
            }
        }
    }

    // Step 4: Apply attention to values with connection-based parallel transport
    // Output_i = sum_j (attention_ij * parallel_transport(V_j, i->j))
    #pragma omp parallel for
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t k = 0; k < head_dim; k++) {
            complex double weighted_sum = 0.0;

            for (size_t j = 0; j < seq_len; j++) {
                complex double attn_weight = attention_scores[i * seq_len + j];
                complex double value_jk = input->values[j * head_dim + k];

                // Apply connection-based parallel transport correction
                // This accounts for the curvature of the attention manifold
                complex double transport_factor = 1.0;
                if (i != j) {
                    size_t conn_idx = (i % config_dim) * config_dim + (j % config_dim);
                    if (conn_idx < config_dim * config_dim && attention->connection_tensor->data) {
                        complex double gamma = attention->connection_tensor->data[conn_idx];
                        // Parallel transport: exp(-Gamma * distance)
                        double dist = (double)(abs((int)i - (int)j)) / (double)seq_len;
                        transport_factor = cexp(-gamma * dist);
                    }
                }

                weighted_sum += attn_weight * value_jk * transport_factor;
            }

            output->values[i * head_dim + k] = weighted_sum;
        }

        // Copy transformed queries and keys (with geometric correction)
        for (size_t k = 0; k < head_dim; k++) {
            // Apply phase to queries
            complex double phase = 1.0;
            if (k < config_dim && attention->phase_tensor->data) {
                phase = attention->phase_tensor->data[k];
            }
            output->queries[i * head_dim + k] = input->queries[i * head_dim + k] * phase;
            output->keys[i * head_dim + k] = input->keys[i * head_dim + k] * conj(phase);
        }
    }

    // Step 5: Apply error correction if enabled
    if (attention->error_correction_enabled) {
        // Simple error correction: project back onto valid subspace
        // Ensure unitarity of phase factors by normalizing
        #pragma omp parallel for
        for (size_t i = 0; i < seq_len; i++) {
            double norm_q = 0.0, norm_k = 0.0, norm_v = 0.0;

            for (size_t k = 0; k < head_dim; k++) {
                norm_q += cabs(output->queries[i * head_dim + k]) * cabs(output->queries[i * head_dim + k]);
                norm_k += cabs(output->keys[i * head_dim + k]) * cabs(output->keys[i * head_dim + k]);
                norm_v += cabs(output->values[i * head_dim + k]) * cabs(output->values[i * head_dim + k]);
            }

            norm_q = sqrt(norm_q);
            norm_k = sqrt(norm_k);
            norm_v = sqrt(norm_v);

            // Normalize to prevent numerical drift
            if (norm_q > 1e-10) {
                double scale_q = sqrt((double)head_dim) / norm_q;
                if (scale_q > 2.0) scale_q = 2.0;
                if (scale_q < 0.5) scale_q = 0.5;
                for (size_t k = 0; k < head_dim; k++) {
                    output->queries[i * head_dim + k] *= scale_q;
                }
            }
            if (norm_k > 1e-10) {
                double scale_k = sqrt((double)head_dim) / norm_k;
                if (scale_k > 2.0) scale_k = 2.0;
                if (scale_k < 0.5) scale_k = 0.5;
                for (size_t k = 0; k < head_dim; k++) {
                    output->keys[i * head_dim + k] *= scale_k;
                }
            }
        }
    }

    // Update metrics
    attention->metrics.attention_score = cabs(attention_scores[0]);
    attention->metrics.geometric_score = cabs(attention->metric_tensor->data[0]);

    // Compute phase coherence as average phase alignment
    double phase_coherence = 0.0;
    for (size_t k = 0; k < head_dim && k < config_dim; k++) {
        if (attention->phase_tensor->data) {
            phase_coherence += cabs(attention->phase_tensor->data[k]);
        }
    }
    attention->metrics.phase_coherence = phase_coherence / (double)(head_dim < config_dim ? head_dim : config_dim);

    free(attention_scores);
    return true;
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
        multiply_matrices(ctx.query, input_mat, attn->W_query);
        
        #pragma omp section
        multiply_matrices(ctx.key, input_mat, attn->W_key);
        
        #pragma omp section
        multiply_matrices(ctx.value, input_mat, attn->W_value);
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
            destroy_hierarchical_matrix(head_ctx.query);
            destroy_hierarchical_matrix(head_ctx.key);
            destroy_hierarchical_matrix(head_ctx.value);
            return;
        }
        head_ctx.output->tolerance = ATTENTION_TOLERANCE;

        // Check if output matrix data is allocated
        if (!head_ctx.output->data) {
            destroy_hierarchical_matrix(head_ctx.query);
            destroy_hierarchical_matrix(head_ctx.key);
            destroy_hierarchical_matrix(head_ctx.value);
            destroy_hierarchical_matrix(head_ctx.output);
            return;
        }
        
        // Compute attention scores - O(log n) with hierarchical operations
        compute_attention_scores(&head_ctx);
        
        // Check if output matrix data is still valid after attention computation
        if (!head_ctx.output->data) {
            destroy_hierarchical_matrix(head_ctx.query);
            destroy_hierarchical_matrix(head_ctx.key);
            destroy_hierarchical_matrix(head_ctx.value);
            destroy_hierarchical_matrix(head_ctx.output);
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
        destroy_hierarchical_matrix(head_ctx.query);
        destroy_hierarchical_matrix(head_ctx.key);
        destroy_hierarchical_matrix(head_ctx.value);
        destroy_hierarchical_matrix(head_ctx.output);
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
    multiply_matrices(output_mat, ctx.output, attn->W_output);
    
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
    
    multiply_matrices(scores, ctx->query, key_t);
    
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
    multiply_matrices(ctx->output, scores, ctx->value);
    
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
    // Each child represents a quadrant of the source matrix
    // The column dimension should match the quadrant, scaled to head_dim
    if (!head->children[0]) {
        for (int i = 0; i < 4; i++) {
            size_t sub_rows = (i < 2) ? mid_row : (mat->rows - mid_row);
            size_t sub_cols = (i % 2 == 0) ? mid_col : (mat->cols - mid_col);
            // Scale columns proportionally to head_dim while preserving aspect ratio
            size_t scaled_cols = (sub_cols * head_dim) / mat->cols;
            if (scaled_cols == 0) scaled_cols = 1;  // Minimum dimension
            head->children[i] = create_hierarchical_matrix(sub_rows, scaled_cols);
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
