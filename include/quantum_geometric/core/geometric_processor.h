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

// Geometry types
typedef enum {
    GEOM_EUCLIDEAN,        // Euclidean geometry
    GEOM_RIEMANNIAN,       // Riemannian geometry
    GEOM_SYMPLECTIC,       // Symplectic geometry
    GEOM_KAHLER           // Kähler geometry
} geometry_type_t;

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
    geometry_type_t geometry;      // Geometry type
    size_t num_dimensions;         // Number of dimensions
    bool enable_optimization;      // Enable optimization
    bool use_quantum_acceleration; // Use quantum acceleration
} processor_config_t;

// Geometric state
typedef struct {
    geometry_type_t type;          // Geometry type
    double* metric_tensor;         // Metric tensor
    double* connection_coeffs;     // Connection coefficients
    double* curvature_tensor;      // Curvature tensor
    size_t dimensions;            // State dimensions
    void* state_data;            // Additional data
} geometric_state_t;

// Transform parameters
typedef struct {
    transform_type_t type;         // Transform type
    complex double* matrix;        // Transform matrix
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
bool initialize_geometry(geometric_processor_t* processor,
                        const geometric_state_t* state);
bool initialize_transform(geometric_processor_t* processor,
                         const transform_params_t* params);
bool validate_initialization(geometric_processor_t* processor);

// Processing operations
bool process_geometric_state(geometric_processor_t* processor,
                           const geometric_state_t* input,
                           geometric_state_t* output);
bool apply_transform(geometric_processor_t* processor,
                    const transform_params_t* transform,
                    geometric_state_t* state);
bool compute_geometric_invariants(geometric_processor_t* processor,
                                const geometric_state_t* state,
                                double* invariants);

// Geometric operations
bool compute_metric(geometric_processor_t* processor,
                   const geometric_state_t* state,
                   double* metric);
bool compute_connection(geometric_processor_t* processor,
                       const geometric_state_t* state,
                       double* connection);
bool compute_curvature(geometric_processor_t* processor,
                      const geometric_state_t* state,
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
bool process_quantum_state(geometric_processor_t* processor,
                          const geometric_state_t* state,
                          geometric_state_t* output);
bool apply_quantum_transform(geometric_processor_t* processor,
                           const transform_params_t* transform,
                           geometric_state_t* state);
bool validate_quantum_state(geometric_processor_t* processor,
                          const geometric_state_t* state);

// Performance monitoring
bool get_processing_metrics(const geometric_processor_t* processor,
                           processing_metrics_t* metrics);
bool monitor_performance(geometric_processor_t* processor,
                        processing_metrics_t* metrics);
bool optimize_performance(geometric_processor_t* processor,
                         const processing_metrics_t* metrics);

// Utility functions
bool export_processor_data(const geometric_processor_t* processor,
                          const char* filename);
bool import_processor_data(geometric_processor_t* processor,
                          const char* filename);
void free_geometric_state(geometric_state_t* state);

#endif // GEOMETRIC_PROCESSOR_H
