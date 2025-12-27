#ifndef QUANTUM_GEOMETRIC_ATTENTION_H
#define QUANTUM_GEOMETRIC_ATTENTION_H

#include <stdbool.h>
#include <stddef.h>
#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/core/error_codes.h"

// Error code aliases for backward compatibility
#ifndef QG_SUCCESS
#define QG_SUCCESS QGT_SUCCESS
#endif
#ifndef QG_ERROR_INVALID_ARGUMENT
#define QG_ERROR_INVALID_ARGUMENT QGT_ERROR_INVALID_ARGUMENT
#endif
#ifndef QG_ERROR_OUT_OF_MEMORY
#define QG_ERROR_OUT_OF_MEMORY QGT_ERROR_MEMORY_ALLOCATION
#endif

// Manifold types for attention geometry
typedef enum {
    MANIFOLD_COMPLEX_PROJECTIVE,
    MANIFOLD_HYPERBOLIC,
    MANIFOLD_SPHERICAL,
    MANIFOLD_EUCLIDEAN
} attention_manifold_t;

// Metric types for attention
typedef enum {
    METRIC_FUBINI_STUDY,
    METRIC_POINCARE,
    METRIC_EUCLIDEAN,
    METRIC_ADAPTIVE
} attention_metric_t;

// Connection types for parallel transport
typedef enum {
    CONNECTION_NATURAL,
    CONNECTION_RIEMANNIAN,
    CONNECTION_CHERN,
    CONNECTION_ADAPTIVE
} attention_connection_t;

// Optimization type enum
typedef enum {
    OPTIMIZATION_GEOMETRIC,
    OPTIMIZATION_NATURAL_GRADIENT,
    OPTIMIZATION_QUANTUM,
    OPTIMIZATION_HYBRID
} attention_optimization_type_t;

// Complexity types
typedef enum {
    COMPLEXITY_LINEAR,
    COMPLEXITY_LOG_LINEAR,
    COMPLEXITY_QUADRATIC,
    COMPLEXITY_ADAPTIVE
} attention_complexity_t;

// Hardware backend types
typedef enum {
    BACKEND_QUANTUM,
    BACKEND_CLASSICAL,
    BACKEND_HYBRID,
    BACKEND_AUTO
} attention_backend_t;

// Default configuration values
#define QG_DEFAULT_NUM_HEADS 8
#define QG_DEFAULT_HEAD_DIM 64
#define QG_DEFAULT_MODEL_DIM 512
#define QG_ATTENTION_MIN_LEVEL_SIZE 16
#define QG_ATTENTION_SPARSITY_FACTOR 0.1

// Geometric configuration
typedef struct {
    attention_manifold_t manifold;    // Geometric manifold type
    attention_metric_t metric;        // Metric tensor type
    attention_connection_t connection; // Connection type
} attention_geometry_t;

// Optimization configuration
typedef struct {
    attention_optimization_type_t type; // Optimization method
    attention_complexity_t complexity;  // Computational complexity
    bool error_protection;              // Enable error protection
} attention_optimization_config_t;

// Hardware configuration
typedef struct {
    attention_backend_t backend;      // Hardware backend
    void* topology;                   // Hardware topology (backend-specific)
} attention_hardware_t;

// Attention configuration
typedef struct {
    attention_geometry_t geometry;            // Geometric configuration
    attention_optimization_config_t optimization; // Optimization configuration
    attention_hardware_t hardware;            // Hardware configuration
    size_t num_heads;                         // Number of attention heads
    size_t head_dim;                          // Dimension per head
    size_t model_dim;                         // Model dimension
} attention_config_t;

// Attention weights structure
typedef struct {
    float* query_weights;             // Query projection weights
    float* key_weights;               // Key projection weights
    float* value_weights;             // Value projection weights
    float* output_weights;            // Output projection weights
    size_t weight_size;               // Size of each weight matrix
} attention_weights_t;

// Attention level for hierarchical attention
typedef struct {
    size_t level_size;                // Size at this level
    size_t stride;                    // Stride for this level
    float* level_scores;              // Attention scores at this level
    float* level_derivs;              // Derivatives at this level
    float* sparsity_mask;             // Sparsity mask for efficient attention
} attention_level_t;

// Hierarchical attention structure
typedef struct {
    size_t seq_length;                // Sequence length
    size_t num_levels;                // Number of hierarchical levels
    double sparsity_factor;           // Sparsity factor for efficiency
    attention_level_t* levels;        // Array of attention levels
} hierarchical_attention_t;

// Attention statistics
typedef struct {
    bool track_complexity;            // Track computational complexity
    bool monitor_errors;              // Monitor error rates
} attention_stats_t;

// Attention results
typedef struct {
    double complexity_order;          // Empirical complexity order
    double error_rate;                // Error rate
    double memory_usage;              // Memory usage (0-1)
} attention_result_t;

// Opaque types - forward declarations must match actual struct names
// quantum_attention is defined in quantum_attention.h with struct quantum_attention
typedef struct quantum_attention quantum_attention_t;
typedef struct quantum_state_t quantum_state_t;
typedef struct quantum_tensor_t quantum_tensor_t;

// Core attention functions
quantum_attention_t* quantum_attention_create(const attention_config_t* config);
void quantum_attention_free(quantum_attention_t* attention);

// Attention operations
attention_result_t quantum_attention_apply(quantum_attention_t* attention,
                                         const quantum_tensor_t* queries,
                                         const quantum_tensor_t* keys,
                                         const quantum_tensor_t* values,
                                         const attention_stats_t* stats);

// Geometric operations
double compute_berry_curvature(const quantum_attention_t* attention,
                             const quantum_state_t* state);
double compute_geometric_phase(const quantum_attention_t* attention,
                             const quantum_state_t* state_i,
                             const quantum_state_t* state_j);
double compute_attention_metric(const quantum_attention_t* attention,
                              const quantum_state_t* state);

// Error mitigation
bool verify_error_bounds(const quantum_attention_t* attention,
                        const quantum_state_t* state,
                        double tolerance);
bool apply_error_correction(quantum_attention_t* attention,
                          quantum_state_t* state);
double estimate_error_rate(const quantum_attention_t* attention);

// Performance monitoring
typedef struct {
    double attention_time;            // Time for attention computation
    double memory_usage;              // Memory usage in bytes
    double gpu_utilization;           // GPU utilization percentage
    double error_rate;                // Error rate
    double geometric_fidelity;        // Geometric fidelity measure
} attention_metrics_t;

bool get_attention_metrics(const quantum_attention_t* attention,
                         attention_metrics_t* metrics);
bool reset_attention_metrics(quantum_attention_t* attention);

// Attention initialization and cleanup
int qg_attention_init(attention_config_t* config);
void qg_attention_cleanup(attention_weights_t* weights);

// Hierarchical attention operations
int qg_hierarchical_attention_init(hierarchical_attention_t* hier_attn,
                                   size_t seq_length);
void qg_hierarchical_attention_cleanup(hierarchical_attention_t* hier_attn);

#endif // QUANTUM_GEOMETRIC_ATTENTION_H
