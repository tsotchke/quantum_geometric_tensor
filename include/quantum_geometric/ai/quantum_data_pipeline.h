#ifndef QUANTUM_DATA_PIPELINE_H
#define QUANTUM_DATA_PIPELINE_H

#include <stdbool.h>
#include <stddef.h>
#include "quantum_geometric/core/quantum_geometric_core.h"

// Sampler types
typedef enum {
    SAMPLER_DIFFUSION_PINN,
    SAMPLER_FLOW_PINN,
    SAMPLER_SCORE_PINN
} sampler_type_t;

// Drift types
typedef enum {
    DRIFT_QUANTUM_FORCE,
    DRIFT_GEOMETRIC,
    DRIFT_ADAPTIVE
} drift_type_t;

// Noise types
typedef enum {
    NOISE_FIXED,
    NOISE_ADAPTIVE,
    NOISE_SCHEDULED
} noise_type_t;

// Manifold types
typedef enum {
    MANIFOLD_KAHLER,
    MANIFOLD_COMPLEX_PROJECTIVE,
    MANIFOLD_UNITARY
} manifold_type_t;

// Metric types
typedef enum {
    METRIC_QUANTUM_FISHER,
    METRIC_FUBINI_STUDY,
    METRIC_DISCRETE_RICCI
} metric_type_t;

// Connection types
typedef enum {
    CONNECTION_NATURAL,
    CONNECTION_LEVI_CIVITA,
    CONNECTION_KAHLER
} connection_type_t;

// Physics configuration
typedef struct {
    sampler_type_t type;        // Type of physics-informed sampler
    drift_type_t drift;         // Type of drift term
    noise_type_t noise;         // Type of noise schedule
} physics_config_t;

// Geometry configuration
typedef struct {
    manifold_type_t manifold;   // Type of manifold
    metric_type_t metric;       // Type of metric
    connection_type_t connection; // Type of connection
} geometry_config_t;

// Optimization configuration
typedef struct {
    bool error_bounds;          // Enable error bounds
    bool validation;            // Enable validation
    bool monitoring;            // Enable monitoring
} optimization_config_t;

// Sampler configuration
typedef struct {
    physics_config_t physics;    // Physics configuration
    geometry_config_t geometry;  // Geometry configuration
    optimization_config_t optimization; // Optimization configuration
} sampler_config_t;

// Sampling statistics
typedef struct {
    bool track_accuracy;        // Track sampling accuracy
    bool monitor_physics;       // Monitor physics constraints
} sampling_stats_t;

// Sampling results
typedef struct {
    double pinn_residual;       // PINN residual error
    double physics_violation;   // Physics constraint violation
    double sample_quality;      // Sample quality metric
} sampling_result_t;

// Opaque types
typedef struct quantum_sampler_t quantum_sampler_t;
typedef struct quantum_state_t quantum_state_t;

// Core sampling functions
quantum_sampler_t* quantum_sampler_create(const sampler_config_t* config);
void quantum_sampler_free(quantum_sampler_t* sampler);

// Sampling operations
sampling_result_t quantum_sample(quantum_sampler_t* sampler,
                               const quantum_state_t* initial_state,
                               const sampling_stats_t* stats);

// Physics-informed operations
double compute_pinn_residual(const quantum_sampler_t* sampler,
                           const quantum_state_t* state);
double verify_physics_constraints(const quantum_sampler_t* sampler,
                                const quantum_state_t* state);
double evaluate_sample_quality(const quantum_sampler_t* sampler,
                             const quantum_state_t* state);

// Geometric operations
double compute_geometric_drift(const quantum_sampler_t* sampler,
                             const quantum_state_t* state);
double compute_noise_schedule(const quantum_sampler_t* sampler,
                            double time);
double compute_metric_tensor(const quantum_sampler_t* sampler,
                           const quantum_state_t* state);

// Performance monitoring
typedef struct {
    double sampling_time;       // Time taken for sampling
    double memory_usage;        // Memory usage in bytes
    double gpu_utilization;     // GPU utilization percentage
    size_t num_samples;        // Number of samples generated
    double acceptance_rate;     // MCMC acceptance rate
} performance_metrics_t;

bool get_performance_metrics(const quantum_sampler_t* sampler,
                           performance_metrics_t* metrics);
bool reset_performance_metrics(quantum_sampler_t* sampler);

#endif // QUANTUM_DATA_PIPELINE_H
