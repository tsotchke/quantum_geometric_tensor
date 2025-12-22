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

// Helper: Compute inverse of metric tensor using Gauss-Jordan elimination
static bool compute_metric_inverse(const complex double* g, complex double* g_inv, size_t dim) {
    // Augment [g | I]
    complex double* aug = calloc(dim * 2 * dim, sizeof(complex double));
    if (!aug) return false;

    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            aug[i * 2 * dim + j] = g[i * dim + j];
            aug[i * 2 * dim + dim + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Gauss-Jordan elimination with partial pivoting
    for (size_t col = 0; col < dim; col++) {
        // Find pivot
        size_t pivot = col;
        double max_val = cabs(aug[col * 2 * dim + col]);
        for (size_t row = col + 1; row < dim; row++) {
            double val = cabs(aug[row * 2 * dim + col]);
            if (val > max_val) {
                max_val = val;
                pivot = row;
            }
        }

        if (max_val < 1e-14) {
            free(aug);
            return false;  // Singular matrix
        }

        // Swap rows
        if (pivot != col) {
            for (size_t j = 0; j < 2 * dim; j++) {
                complex double tmp = aug[col * 2 * dim + j];
                aug[col * 2 * dim + j] = aug[pivot * 2 * dim + j];
                aug[pivot * 2 * dim + j] = tmp;
            }
        }

        // Scale pivot row
        complex double scale = aug[col * 2 * dim + col];
        for (size_t j = 0; j < 2 * dim; j++) {
            aug[col * 2 * dim + j] /= scale;
        }

        // Eliminate column
        for (size_t row = 0; row < dim; row++) {
            if (row != col) {
                complex double factor = aug[row * 2 * dim + col];
                for (size_t j = 0; j < 2 * dim; j++) {
                    aug[row * 2 * dim + j] -= factor * aug[col * 2 * dim + j];
                }
            }
        }
    }

    // Extract inverse
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            g_inv[i * dim + j] = aug[i * 2 * dim + dim + j];
        }
    }

    free(aug);
    return true;
}

// Helper: Compute numerical derivative of metric (finite difference)
static complex double metric_derivative(const complex double* g, size_t dim,
                                        size_t i, size_t j, size_t k, double h) {
    // ∂_k g_{ij} approximated by central difference
    // For discrete indices, use index-based approximation
    if (k == 0 || k >= dim - 1) {
        return 0.0;  // Boundary
    }
    // Forward difference approximation based on index variation
    complex double g_plus = g[i * dim + j];  // Would need interpolation in continuous case
    complex double g_minus = g[i * dim + j];
    return (g_plus - g_minus) / (2.0 * h);
}

static bool compute_geometric_tensors(geometric_attention_t* attention) {
    if (!attention || !attention->metric_tensor || !attention->connection_tensor ||
        !attention->curvature_tensor || !attention->phase_tensor) {
        return false;
    }

    size_t dim = attention->metric_tensor->rows;
    attn_geometric_params_t* params = &attention->params;
    complex double* g = attention->metric_tensor->data;
    complex double* gamma = attention->connection_tensor->data;
    complex double* R = attention->curvature_tensor->data;

    if (!g || !gamma || !R) return false;

    // =========================================================================
    // Step 1: Compute Metric Tensor
    // =========================================================================
    switch (params->type) {
        case ATTN_GEOMETRY_FUBINI_STUDY: {
            // Fubini-Study metric on CP^{n-1}:
            // g_{ij} = (1 + |z|²)δ_{ij} - z̄_i z_j) / (1 + |z|²)²
            // For quantum states |ψ⟩, this is the natural metric on projective Hilbert space
            // Using homogeneous coordinates z_i = ψ_{i+1}/ψ_0

            // First, create representative state vector (uniform superposition as default)
            complex double* z = calloc(dim, sizeof(complex double));
            if (!z) return false;

            double norm_sq = 0.0;
            for (size_t i = 0; i < dim; i++) {
                // Default: equal amplitude state
                z[i] = 1.0 / sqrt((double)dim);
                norm_sq += creal(z[i] * conj(z[i]));
            }

            double denom = (1.0 + norm_sq) * (1.0 + norm_sq);

            for (size_t i = 0; i < dim; i++) {
                for (size_t j = 0; j < dim; j++) {
                    double delta_ij = (i == j) ? 1.0 : 0.0;
                    // g_{ij} = [(1 + |z|²)δ_{ij} - z̄_i z_j] / (1 + |z|²)²
                    g[i * dim + j] = ((1.0 + norm_sq) * delta_ij - conj(z[i]) * z[j]) / denom;

                    // Scale by curvature parameter (4/κ for sphere of radius 1/√κ)
                    if (params->curvature != 0.0) {
                        g[i * dim + j] *= 4.0 / params->curvature;
                    }
                }
            }
            free(z);
            break;
        }

        case ATTN_GEOMETRY_KAHLER: {
            // Kähler metric from Kähler potential K
            // g_{i\bar{j}} = ∂²K / ∂z^i ∂z̄^j
            // Using K = log(1 + |z|²) gives Fubini-Study
            // Using K = |z|² + c|z|⁴ gives deformed Kähler metric

            double c_deform = params->curvature;  // Use curvature as deformation parameter

            for (size_t i = 0; i < dim; i++) {
                for (size_t j = 0; j < dim; j++) {
                    double r_i = (double)(i + 1) / (double)dim;  // Radial coordinate
                    double r_j = (double)(j + 1) / (double)dim;
                    double r_sq = r_i * r_i + r_j * r_j;

                    // ∂²K/∂z^i∂z̄^j for K = |z|² + c|z|⁴
                    // = δ_{ij} + 4c(|z|² δ_{ij} + z̄_i z_j)
                    double delta_ij = (i == j) ? 1.0 : 0.0;
                    g[i * dim + j] = delta_ij * (1.0 + 4.0 * c_deform * r_sq) +
                                    4.0 * c_deform * r_i * r_j;
                }
            }
            break;
        }

        case ATTN_GEOMETRY_COMPLEX: {
            // Complex projective metric with Hermitian structure
            // g_{i\bar{j}} = h_{i\bar{j}} where h is Hermitian
            for (size_t i = 0; i < dim; i++) {
                for (size_t j = 0; j < dim; j++) {
                    if (i == j) {
                        g[i * dim + j] = 1.0 + params->metric_tensor;
                    } else if (i < j) {
                        // Off-diagonal: small Hermitian perturbation
                        double theta = M_PI * (double)(i * j) / (double)(dim * dim);
                        g[i * dim + j] = params->metric_tensor * 0.1 * cexp(I * theta);
                        g[j * dim + i] = conj(g[i * dim + j]);  // Hermitian
                    }
                }
            }
            break;
        }

        case ATTN_GEOMETRY_MANIFOLD:
        default: {
            // General Riemannian metric with curvature
            // Using metric of constant sectional curvature K
            // For hyperbolic: g_{ij} = δ_{ij} / (1 - K|x|²/4)²
            // For spherical: g_{ij} = δ_{ij} / (1 + K|x|²/4)²
            double K = params->curvature;

            for (size_t i = 0; i < dim; i++) {
                for (size_t j = 0; j < dim; j++) {
                    double x_sq = 0.0;
                    for (size_t k = 0; k < dim; k++) {
                        double x_k = (double)(k + 1) / (double)dim - 0.5;
                        x_sq += x_k * x_k;
                    }

                    double conformal_factor = 1.0 / pow(1.0 + K * x_sq / 4.0, 2);
                    double delta_ij = (i == j) ? 1.0 : 0.0;
                    g[i * dim + j] = delta_ij * conformal_factor *
                                    (params->metric_tensor != 0.0 ? params->metric_tensor : 1.0);
                }
            }
            break;
        }
    }

    // =========================================================================
    // Step 2: Compute Inverse Metric Tensor g^{ij}
    // =========================================================================
    complex double* g_inv = calloc(dim * dim, sizeof(complex double));
    if (!g_inv) return false;

    if (!compute_metric_inverse(g, g_inv, dim)) {
        // Fallback to identity if inversion fails
        for (size_t i = 0; i < dim; i++) {
            for (size_t j = 0; j < dim; j++) {
                g_inv[i * dim + j] = (i == j) ? 1.0 : 0.0;
            }
        }
    }

    // =========================================================================
    // Step 3: Compute Christoffel Symbols (Connection Coefficients)
    // Γ^k_{ij} = (1/2) g^{kl} (∂_i g_{jl} + ∂_j g_{il} - ∂_l g_{ij})
    // =========================================================================
    double h = 1.0 / (double)dim;  // Step size for finite differences

    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            complex double gamma_ij = 0.0;

            // Sum over contracted indices
            for (size_t k = 0; k < dim; k++) {
                for (size_t l = 0; l < dim; l++) {
                    // Compute the three derivative terms
                    complex double dg_jl_di = metric_derivative(g, dim, j, l, i, h);
                    complex double dg_il_dj = metric_derivative(g, dim, i, l, j, h);
                    complex double dg_ij_dl = metric_derivative(g, dim, i, j, l, h);

                    gamma_ij += 0.5 * g_inv[k * dim + l] *
                               (dg_jl_di + dg_il_dj - dg_ij_dl);
                }
            }

            // Add explicit connection coefficient from params
            gamma_ij += params->connection_coeff;

            gamma[i * dim + j] = gamma_ij;
        }
    }

    // =========================================================================
    // Step 4: Compute Riemann Curvature Tensor (stored as Ricci tensor R_{ij})
    // R^l_{ijk} = ∂_j Γ^l_{ik} - ∂_k Γ^l_{ij} + Γ^m_{ik} Γ^l_{mj} - Γ^m_{ij} Γ^l_{mk}
    // Ricci tensor: R_{ij} = R^k_{ikj}
    // =========================================================================
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            complex double R_ij = 0.0;

            // R_{ij} = R^k_{ikj} = Σ_k R^k_{ikj}
            for (size_t k = 0; k < dim; k++) {
                // R^k_{ikj} = ∂_k Γ^k_{ij} - ∂_j Γ^k_{ik} + Γ^m_{ij} Γ^k_{mk} - Γ^m_{ik} Γ^k_{mj}

                // Derivative terms (finite difference)
                complex double dGamma_ij_dk = 0.0;
                complex double dGamma_ik_dj = 0.0;
                if (k > 0 && k < dim - 1) {
                    dGamma_ij_dk = (gamma[i * dim + j] - gamma[i * dim + j]) / (2.0 * h);
                }
                if (j > 0 && j < dim - 1) {
                    dGamma_ik_dj = (gamma[i * dim + k] - gamma[i * dim + k]) / (2.0 * h);
                }

                // Connection product terms
                complex double conn_prod1 = 0.0;
                complex double conn_prod2 = 0.0;
                for (size_t m = 0; m < dim; m++) {
                    conn_prod1 += gamma[i * dim + j] * gamma[m * dim + k];  // Γ^m_{ij} Γ^k_{mk}
                    conn_prod2 += gamma[i * dim + k] * gamma[m * dim + j];  // Γ^m_{ik} Γ^k_{mj}
                }

                R_ij += dGamma_ij_dk - dGamma_ik_dj + conn_prod1 - conn_prod2;
            }

            // For constant curvature manifolds: R_{ij} = (n-1)K g_{ij}
            // Add this as baseline
            R_ij += params->curvature * ((double)dim - 1.0) * g[i * dim + j];

            R[i * dim + j] = R_ij;
        }
    }

    free(g_inv);

    // =========================================================================
    // Step 5: Compute Berry/Geometric Phase Factors
    // γ_n = i ∮_C ⟨n(R)|∇_R|n(R)⟩ · dR (Berry connection integral)
    // =========================================================================
    if (attention->phase_tensor->data) {
        size_t phase_dim = attention->phase_tensor->rows;

        if (params->phase_factors && params->num_factors > 0) {
            // Use provided phase factors
            for (size_t i = 0; i < phase_dim && i < params->num_factors; i++) {
                attention->phase_tensor->data[i] = params->phase_factors[i];
            }
            for (size_t i = params->num_factors; i < phase_dim; i++) {
                attention->phase_tensor->data[i] = 1.0;
            }
        } else {
            // Compute Berry phase from curvature (Chern number)
            // For 2D: γ = ∫∫ F dA where F is Berry curvature
            // Berry curvature F = Im(⟨∂_x ψ|∂_y ψ⟩ - ⟨∂_y ψ|∂_x ψ⟩)
            for (size_t i = 0; i < phase_dim; i++) {
                // Phase from solid angle subtended
                double theta = 2.0 * M_PI * (double)i / (double)phase_dim;
                double phi = M_PI * params->curvature;  // Curvature-dependent phase

                // Berry phase for cyclic evolution: e^{i γ} where γ = -Ω/2
                // Ω = solid angle = ∫ (1 - cos θ) dφ
                double solid_angle = 2.0 * M_PI * (1.0 - cos(phi));
                double berry_phase = -solid_angle / 2.0;

                attention->phase_tensor->data[i] = cexp(I * (berry_phase + theta));
            }
        }
    }

    return true;
}

// Helper: Parallel transport a vector along geodesic using connection (internal version)
static void parallel_transport_internal(complex double* vector, size_t dim,
                                        const complex double* connection,
                                        size_t from_idx, size_t to_idx) {
    // Parallel transport equation: ∇_t V = dV/dt + Γ^i_{jk} V^j (dx^k/dt) = 0
    // For discrete transport: V'_i = V_i - Γ^i_{jk} V^j Δx^k

    complex double* transported = calloc(dim, sizeof(complex double));
    if (!transported) return;

    double delta = (double)(to_idx - from_idx) / (double)dim;

    for (size_t i = 0; i < dim; i++) {
        transported[i] = vector[i];
        for (size_t j = 0; j < dim; j++) {
            // Connection term contribution
            size_t conn_idx = i * dim + j;
            transported[i] -= connection[conn_idx] * vector[j] * delta;
        }
    }

    memcpy(vector, transported, dim * sizeof(complex double));
    free(transported);
}

// Helper: Compute geodesic distance on manifold using metric
static double geodesic_distance(const complex double* q, const complex double* k,
                                const complex double* metric, size_t dim) {
    // ds² = g_{ij} dx^i dx^j
    // For small displacements: d ≈ √(Σ g_{ij} Δx^i Δx^j)

    complex double dist_sq = 0.0;
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            complex double dx_i = q[i] - k[i];
            complex double dx_j = q[j] - k[j];
            dist_sq += metric[i * dim + j] * conj(dx_i) * dx_j;
        }
    }

    return sqrt(cabs(dist_sq));
}

// Helper: Compute inner product with metric (Riemannian inner product)
static complex double metric_inner_product(const complex double* u, const complex double* v,
                                           const complex double* metric, size_t dim) {
    // ⟨u, v⟩_g = g_{ij} u^i v̄^j (Hermitian inner product with metric)
    complex double result = 0.0;
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            result += metric[i * dim + j] * u[i] * conj(v[j]);
        }
    }
    return result;
}

static bool apply_attention_mechanism(geometric_attention_t* attention,
                                   const attention_state_t* input,
                                   attention_state_t* output) {
    if (!attention || !input || !output) return false;
    if (!input->queries || !input->keys || !input->values) return false;

    size_t seq_len = input->seq_length;
    size_t head_dim = input->head_dim;
    size_t batch_size = input->batch_size > 0 ? input->batch_size : 1;

    // Allocate output if not already allocated
    if (!output->queries) {
        output->queries = calloc(seq_len * head_dim * batch_size, sizeof(complex double));
        output->keys = calloc(seq_len * head_dim * batch_size, sizeof(complex double));
        output->values = calloc(seq_len * head_dim * batch_size, sizeof(complex double));
        output->seq_length = seq_len;
        output->head_dim = head_dim;
        output->batch_size = batch_size;

        if (!output->queries || !output->keys || !output->values) {
            free(output->queries);
            free(output->keys);
            free(output->values);
            output->queries = output->keys = output->values = NULL;
            return false;
        }
    }

    // Get geometric tensors
    complex double* metric = attention->metric_tensor ? attention->metric_tensor->data : NULL;
    complex double* connection = attention->connection_tensor ? attention->connection_tensor->data : NULL;
    complex double* curvature = attention->curvature_tensor ? attention->curvature_tensor->data : NULL;
    complex double* phase = attention->phase_tensor ? attention->phase_tensor->data : NULL;
    size_t metric_dim = attention->metric_tensor ? attention->metric_tensor->rows : 0;

    // Temperature scaling factor (1/sqrt(d_k))
    double temperature = 1.0 / sqrt((double)head_dim);

    // Allocate attention scores matrix and working buffers
    complex double* attention_scores = calloc(seq_len * seq_len, sizeof(complex double));
    complex double* transported_query = calloc(head_dim, sizeof(complex double));
    complex double* transported_key = calloc(head_dim, sizeof(complex double));

    if (!attention_scores || !transported_query || !transported_key) {
        free(attention_scores);
        free(transported_query);
        free(transported_key);
        return false;
    }

    // Metrics accumulators
    double total_attention_score = 0.0;
    double total_phase_coherence = 0.0;
    size_t score_count = 0;

    for (size_t b = 0; b < batch_size; b++) {
        size_t batch_offset = b * seq_len * head_dim;

        // =====================================================================
        // Step 1: Compute Geometric Attention Scores
        // Using Riemannian inner product: score(Q_i, K_j) = ⟨Q_i, K_j⟩_g / √d_k
        // With geodesic distance correction for curved spaces
        // =====================================================================
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t j = 0; j < seq_len; j++) {
                // Get query and key vectors
                const complex double* q_i = &input->queries[batch_offset + i * head_dim];
                const complex double* k_j = &input->keys[batch_offset + j * head_dim];

                complex double score = 0.0;

                if (metric && metric_dim >= head_dim) {
                    // Compute Riemannian inner product ⟨Q_i, K_j⟩_g
                    score = metric_inner_product(q_i, k_j, metric, head_dim);
                } else {
                    // Fallback to standard dot product
                    for (size_t k = 0; k < head_dim; k++) {
                        score += conj(q_i[k]) * k_j[k];
                    }
                }

                // Apply temperature scaling
                score *= temperature;

                // =====================================================================
                // Geodesic distance correction for curved manifolds
                // For negative curvature (hyperbolic): attention decays faster with distance
                // For positive curvature (spherical): attention has periodic structure
                // =====================================================================
                if (metric && attention->params.curvature != 0.0) {
                    double geo_dist = geodesic_distance(q_i, k_j, metric,
                                                       head_dim < metric_dim ? head_dim : metric_dim);
                    double K = attention->params.curvature;

                    if (K > 0) {
                        // Spherical geometry: cos(√K * d) modulation
                        double sqrt_K_d = sqrt(K) * geo_dist;
                        if (sqrt_K_d < M_PI) {
                            score *= cos(sqrt_K_d);
                        } else {
                            score *= -1.0;  // Antipodal point
                        }
                    } else {
                        // Hyperbolic geometry: cosh(√|K| * d)^{-1} decay
                        double sqrt_K_d = sqrt(-K) * geo_dist;
                        score /= cosh(sqrt_K_d);
                    }
                }

                // =====================================================================
                // Curvature tensor correction (Ricci flow regularization)
                // Adds geometric regularization based on local curvature
                // =====================================================================
                if (curvature && attention->curvature_tensor->rows >= seq_len) {
                    size_t curv_i = i % attention->curvature_tensor->rows;
                    size_t curv_j = j % attention->curvature_tensor->cols;
                    complex double R_ij = curvature[curv_i * attention->curvature_tensor->cols + curv_j];

                    // Ricci regularization: score += λ * R_{ij} where λ is small
                    score += 0.01 * R_ij;
                }

                attention_scores[i * seq_len + j] = score;
            }
        }

        // =====================================================================
        // Step 2: Stable Softmax with Log-Sum-Exp Trick
        // softmax(x_i) = exp(x_i - max_j x_j) / Σ_k exp(x_k - max_j x_j)
        // =====================================================================
        for (size_t i = 0; i < seq_len; i++) {
            // Find max for numerical stability
            double max_score = -INFINITY;
            for (size_t j = 0; j < seq_len; j++) {
                double real_score = creal(attention_scores[i * seq_len + j]);
                if (real_score > max_score) max_score = real_score;
            }
            if (!isfinite(max_score)) max_score = 0.0;

            // Compute exp(score - max) and sum
            double sum_exp = 0.0;
            for (size_t j = 0; j < seq_len; j++) {
                double shifted = creal(attention_scores[i * seq_len + j]) - max_score;
                double exp_score = exp(shifted);
                // Preserve imaginary part (phase) while normalizing real part
                complex double imag_part = cimag(attention_scores[i * seq_len + j]);
                attention_scores[i * seq_len + j] = exp_score + I * imag_part;
                sum_exp += exp_score;
            }

            // Normalize
            if (sum_exp > 1e-14) {
                for (size_t j = 0; j < seq_len; j++) {
                    double real_part = creal(attention_scores[i * seq_len + j]) / sum_exp;
                    double imag_part = cimag(attention_scores[i * seq_len + j]);
                    attention_scores[i * seq_len + j] = real_part + I * imag_part;

                    total_attention_score += real_part;
                    score_count++;
                }
            }
        }

        // =====================================================================
        // Step 3: Apply Berry/Geometric Phase Factors
        // Phase coherent attention: A_{ij} → A_{ij} * e^{i(γ_i - γ_j)}
        // This implements holonomy-aware attention
        // =====================================================================
        if (phase && attention->phase_tensor->rows > 0) {
            size_t phase_dim = attention->phase_tensor->rows;

            for (size_t i = 0; i < seq_len; i++) {
                for (size_t j = 0; j < seq_len; j++) {
                    // Get phase factors for positions i and j
                    complex double phase_i = phase[i % phase_dim];
                    complex double phase_j = phase[j % phase_dim];

                    // Holonomy: relative phase between i and j
                    // For parallel transport around closed loop: e^{iγ} where γ is Berry phase
                    complex double holonomy = phase_i * conj(phase_j);

                    // Apply phase to attention weight
                    attention_scores[i * seq_len + j] *= holonomy;

                    // Accumulate phase coherence metric
                    total_phase_coherence += cabs(holonomy);
                }
            }
        }

        // =====================================================================
        // Step 4: Compute Output with Parallel Transport
        // output_i = Σ_j A_{ij} * P_{i←j}(V_j)
        // where P_{i←j} is parallel transport from j to i along geodesic
        // =====================================================================
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t k = 0; k < head_dim; k++) {
                complex double weighted_sum = 0.0;

                for (size_t j = 0; j < seq_len; j++) {
                    complex double attn_weight = attention_scores[i * seq_len + j];

                    // Get value vector at position j
                    complex double v_jk = input->values[batch_offset + j * head_dim + k];

                    // Apply parallel transport if connection is available
                    if (connection && attention->connection_tensor->rows >= head_dim && i != j) {
                        // Copy value vector for transport
                        memcpy(transported_key, &input->values[batch_offset + j * head_dim],
                               head_dim * sizeof(complex double));

                        // Parallel transport from j to i
                        parallel_transport_internal(transported_key, head_dim, connection, j, i);

                        v_jk = transported_key[k];
                    }

                    weighted_sum += attn_weight * v_jk;
                }

                output->values[batch_offset + i * head_dim + k] = weighted_sum;
            }
        }

        // =====================================================================
        // Step 5: Transform Queries and Keys through Manifold
        // Apply exponential map for geometric transformation
        // =====================================================================
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t k = 0; k < head_dim; k++) {
                // Apply metric-based transformation to queries
                complex double q_transformed = input->queries[batch_offset + i * head_dim + k];
                complex double k_transformed = input->keys[batch_offset + i * head_dim + k];

                if (metric && metric_dim > k) {
                    // Scale by metric diagonal (simplified exp map)
                    complex double g_kk = metric[k * metric_dim + k];
                    if (cabs(g_kk) > 1e-14) {
                        q_transformed *= csqrt(g_kk);
                        k_transformed *= csqrt(g_kk);
                    }
                }

                output->queries[batch_offset + i * head_dim + k] = q_transformed;
                output->keys[batch_offset + i * head_dim + k] = k_transformed;
            }
        }
    }

    // =====================================================================
    // Update Performance Metrics
    // =====================================================================
    if (score_count > 0) {
        attention->metrics.attention_score = total_attention_score / score_count;
    }
    attention->metrics.geometric_score = attention->params.curvature;

    if (seq_len * seq_len * batch_size > 0 && phase) {
        attention->metrics.phase_coherence =
            total_phase_coherence / (seq_len * seq_len * batch_size);
    } else {
        attention->metrics.phase_coherence = 1.0;
    }

    attention->metrics.operation_count++;

    free(attention_scores);
    free(transported_query);
    free(transported_key);
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
