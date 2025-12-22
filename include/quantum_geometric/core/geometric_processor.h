#ifndef GEOMETRIC_PROCESSOR_H
#define GEOMETRIC_PROCESSOR_H

#include <stdbool.h>
#include <stddef.h>
#include <complex.h>

// Processor types
typedef enum {
    PROC_GEOMETRIC,         // Geometric processor
    PROC_QUANTUM,           // Quantum processor
    PROC_HYBRID,            // Hybrid processor
    PROC_CLASSICAL        // Classical processor
} processor_type_t;

// Operation modes
typedef enum {
    MODE_SEQUENTIAL,        // Sequential processing
    MODE_PARALLEL,          // Parallel processing
    MODE_DISTRIBUTED,       // Distributed processing
    MODE_ADAPTIVE         // Adaptive processing
} operation_mode_t;

// Processor geometry types (module-specific to avoid conflicts)
typedef enum {
    PROC_GEOM_EUCLIDEAN,        // Euclidean geometry
    PROC_GEOM_RIEMANNIAN,       // Riemannian geometry
    PROC_GEOM_SYMPLECTIC,       // Symplectic geometry
    PROC_GEOM_KAHLER           // Kahler geometry
} proc_geometry_type_t;

// Transform types
typedef enum {
    TRANSFORM_LINEAR,       // Linear transform
    TRANSFORM_NONLINEAR,   // Nonlinear transform
    TRANSFORM_QUANTUM,      // Quantum transform
    TRANSFORM_HYBRID      // Hybrid transform
} transform_type_t;

// Processor configuration
typedef struct {
    processor_type_t type;         // Processor type
    operation_mode_t mode;         // Operation mode
    proc_geometry_type_t geometry; // Geometry type
    size_t num_dimensions;         // Number of dimensions
    bool enable_optimization;      // Enable optimization
    bool use_quantum_acceleration; // Use quantum acceleration
} processor_config_t;

// Geometric state for processor
typedef struct {
    proc_geometry_type_t type;     // Geometry type
    double* metric_tensor;         // Metric tensor
    double* connection_coeffs;     // Connection coefficients
    double* curvature_tensor;      // Curvature tensor
    size_t dimensions;            // State dimensions
    void* state_data;            // Additional data
} proc_geometric_state_t;

// Transform parameters
typedef struct {
    transform_type_t type;         // Transform type
    _Complex double* matrix;        // Transform matrix
    double* parameters;            // Transform parameters
    size_t param_count;           // Parameter count
    bool is_unitary;              // Unitary flag
    void* transform_data;         // Additional data
} transform_params_t;

// Processing metrics
typedef struct {
    double accuracy;               // Processing accuracy
    double efficiency;             // Processing efficiency
    double stability;              // Numerical stability
    double quantum_advantage;      // Quantum advantage
    size_t operation_count;        // Operation count
    double execution_time;         // Execution time
} processing_metrics_t;

// Opaque processor handle
typedef struct geometric_processor_t geometric_processor_t;

// Core functions
geometric_processor_t* create_geometric_processor(const processor_config_t* config);
void destroy_geometric_processor(geometric_processor_t* processor);

// Initialization functions
bool processor_init_geometry(geometric_processor_t* processor,
                            const proc_geometric_state_t* state);
bool processor_init_transform(geometric_processor_t* processor,
                             const transform_params_t* params);
bool processor_validate_init(geometric_processor_t* processor);

// Processing operations
bool processor_process_state(geometric_processor_t* processor,
                            const proc_geometric_state_t* input,
                            proc_geometric_state_t* output);
bool processor_apply_transform(geometric_processor_t* processor,
                              const transform_params_t* transform,
                              proc_geometric_state_t* state);
bool processor_compute_invariants(geometric_processor_t* processor,
                                 const proc_geometric_state_t* state,
                                 double* invariants);

// Geometric operations
bool processor_compute_metric(geometric_processor_t* processor,
                             const proc_geometric_state_t* state,
                             double* metric);
bool processor_compute_connection(geometric_processor_t* processor,
                                 const proc_geometric_state_t* state,
                                 double* connection);
bool processor_compute_curvature(geometric_processor_t* processor,
                                const proc_geometric_state_t* state,
                                double* curvature);

// Transform operations
bool generate_transform(geometric_processor_t* processor,
                       transform_type_t type,
                       transform_params_t* params);
bool optimize_transform(geometric_processor_t* processor,
                       transform_params_t* params);
bool validate_transform(geometric_processor_t* processor,
                       const transform_params_t* params);

// Quantum operations
bool processor_process_quantum_state(geometric_processor_t* processor,
                                    const proc_geometric_state_t* state,
                                    proc_geometric_state_t* output);
bool processor_apply_quantum_transform(geometric_processor_t* processor,
                                      const transform_params_t* transform,
                                      proc_geometric_state_t* state);
bool processor_validate_quantum_state(geometric_processor_t* processor,
                                     const proc_geometric_state_t* state);

// Performance monitoring
bool processor_get_metrics(const geometric_processor_t* processor,
                          processing_metrics_t* metrics);
bool processor_monitor_performance(geometric_processor_t* processor,
                                  processing_metrics_t* metrics);
bool processor_optimize_performance(geometric_processor_t* processor,
                                   const processing_metrics_t* metrics);

// Utility functions
bool processor_export_data(const geometric_processor_t* processor,
                          const char* filename);
bool processor_import_data(geometric_processor_t* processor,
                          const char* filename);
void processor_free_state(proc_geometric_state_t* state);

#endif // GEOMETRIC_PROCESSOR_H
