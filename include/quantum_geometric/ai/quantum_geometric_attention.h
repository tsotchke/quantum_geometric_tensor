#ifndef QUANTUM_GEOMETRIC_ATTENTION_H
#define QUANTUM_GEOMETRIC_ATTENTION_H

#include <stdbool.h>
#include <stddef.h>
#include "quantum_geometric/core/quantum_geometric_core.h"

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

// Optimization types
typedef enum {
    OPTIMIZATION_GEOMETRIC,
    OPTIMIZATION_NATURAL_GRADIENT,
    OPTIMIZATION_QUANTUM,
    OPTIMIZATION_HYBRID
} attention_optimization_t;

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

// Geometric configuration
typedef struct {
    attention_manifold_t manifold;    // Geometric manifold type
    attention_metric_t metric;        // Metric tensor type
    attention_connection_t connection; // Connection type
} attention_geometry_t;

// Optimization configuration
typedef struct {
    attention_optimization_t type;     // Optimization method
    attention_complexity_t complexity; // Computational complexity
    bool error_protection;            // Enable error protection
} attention_optimization_t;

// Hardware configuration
typedef struct {
    attention_backend_t backend;      // Hardware backend
    void* topology;                   // Hardware topology (backend-specific)
} attention_hardware_t;

// Attention configuration
typedef struct {
    attention_geometry_t geometry;     // Geometric configuration
    attention_optimization_t optimization; // Optimization configuration
    attention_hardware_t hardware;     // Hardware configuration
} attention_config_t;

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

// Opaque types
typedef struct quantum_attention_t quantum_attention_t;
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

#endif // QUANTUM_GEOMETRIC_ATTENTION_H
