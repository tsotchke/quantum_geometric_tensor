#ifndef GEOMETRIC_ATTENTION_H
#define GEOMETRIC_ATTENTION_H

#include <stdbool.h>
#include <stddef.h>
#include <complex.h>

// Attention types
typedef enum {
    ATTENTION_GEOMETRIC,     // Geometric attention
    ATTENTION_QUANTUM,       // Quantum attention
    ATTENTION_HYBRID,        // Hybrid attention
    ATTENTION_ADAPTIVE      // Adaptive attention
} attention_type_t;

// Geometry types
typedef enum {
    GEOMETRY_MANIFOLD,       // Manifold geometry
    GEOMETRY_COMPLEX,        // Complex projective
    GEOMETRY_KAHLER,         // Kähler manifold
    GEOMETRY_FUBINI_STUDY   // Fubini-Study metric
} geometry_type_t;

// Connection types
typedef enum {
    CONNECTION_NATURAL,      // Natural connection
    CONNECTION_GEOMETRIC,    // Geometric connection
    CONNECTION_QUANTUM,      // Quantum connection
    CONNECTION_HYBRID       // Hybrid connection
} connection_type_t;

// Phase types
typedef enum {
    PHASE_BERRY,            // Berry phase
    PHASE_GEOMETRIC,        // Geometric phase
    PHASE_DYNAMIC,          // Dynamic phase
    PHASE_TOPOLOGICAL      // Topological phase
} phase_type_t;

// Attention configuration
typedef struct {
    attention_type_t type;          // Attention type
    geometry_type_t geometry;       // Geometry type
    connection_type_t connection;   // Connection type
    size_t attention_heads;         // Number of heads
    size_t head_dim;               // Head dimension
    bool use_error_correction;     // Error correction flag
} attention_config_t;

// Geometric parameters
typedef struct {
    geometry_type_t type;           // Geometry type
    double metric_tensor;           // Metric tensor
    double curvature;              // Curvature
    double connection_coeff;        // Connection coefficient
    complex double* phase_factors;  // Phase factors
    size_t num_factors;            // Number of factors
} geometric_params_t;

// Attention state
typedef struct {
    complex double* queries;        // Query states
    complex double* keys;           // Key states
    complex double* values;         // Value states
    size_t seq_length;             // Sequence length
    size_t batch_size;             // Batch size
    size_t head_dim;              // Head dimension
} attention_state_t;

// Attention metrics
typedef struct {
    double attention_score;         // Attention score
    double geometric_score;         // Geometric score
    double phase_coherence;         // Phase coherence
    double error_rate;             // Error rate
    size_t operation_count;        // Operation count
    double execution_time;         // Execution time
} attention_metrics_t;

// Opaque attention handle
typedef struct geometric_attention_t geometric_attention_t;

// Core functions
geometric_attention_t* create_geometric_attention(const attention_config_t* config);
void destroy_geometric_attention(geometric_attention_t* attention);

// Initialization functions
bool initialize_geometry(geometric_attention_t* attention,
                        const geometric_params_t* params);
bool initialize_attention_state(geometric_attention_t* attention,
                              const attention_state_t* state);
bool validate_initialization(geometric_attention_t* attention);

// Attention operations
bool compute_attention(geometric_attention_t* attention,
                      const attention_state_t* input,
                      attention_state_t* output);
bool apply_geometric_phase(geometric_attention_t* attention,
                         phase_type_t phase_type,
                         attention_state_t* state);
bool compute_attention_weights(geometric_attention_t* attention,
                             const attention_state_t* state,
                             complex double* weights);

// Geometric operations
bool compute_metric_tensor(geometric_attention_t* attention,
                          const geometric_params_t* params,
                          double* metric);
bool compute_connection(geometric_attention_t* attention,
                       const geometric_params_t* params,
                       double* connection);
bool compute_curvature(geometric_attention_t* attention,
                      const geometric_params_t* params,
                      double* curvature);

// Phase operations
bool compute_berry_phase(geometric_attention_t* attention,
                        const attention_state_t* state,
                        complex double* phase);
bool compute_geometric_phase(geometric_attention_t* attention,
                           const attention_state_t* state,
                           complex double* phase);
bool apply_phase_correction(geometric_attention_t* attention,
                          attention_state_t* state);

// Error correction
bool detect_errors(geometric_attention_t* attention,
                  const attention_state_t* state,
                  double* error_rates);
bool correct_errors(geometric_attention_t* attention,
                   attention_state_t* state);
bool validate_correction(geometric_attention_t* attention,
                        const attention_state_t* state);

// Performance monitoring
bool get_attention_metrics(const geometric_attention_t* attention,
                          attention_metrics_t* metrics);
bool monitor_performance(geometric_attention_t* attention,
                        attention_metrics_t* metrics);
bool optimize_performance(geometric_attention_t* attention,
                         const attention_metrics_t* metrics);

// Utility functions
bool export_attention_data(const geometric_attention_t* attention,
                          const char* filename);
bool import_attention_data(geometric_attention_t* attention,
                          const char* filename);
void free_attention_state(attention_state_t* state);

#endif // GEOMETRIC_ATTENTION_H
