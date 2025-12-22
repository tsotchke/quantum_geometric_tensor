#ifndef GRADIENT_OPTIMIZER_H
#define GRADIENT_OPTIMIZER_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations for opaque types
typedef struct GradientOptimizer GradientOptimizer;
typedef struct OptimizerState OptimizerState;
typedef struct QuantumGradientState QuantumGradientState;
typedef struct GeometricProcessor GeometricProcessor;
typedef struct Manifold Manifold;

// Use forward declarations to avoid conflicts with other headers
struct QuantumCircuit;
struct QuantumState;

// Compression types
#ifndef COMPRESSION_TYPE_DEFINED
#define COMPRESSION_TYPE_DEFINED
typedef enum CompressionType {
    COMPRESSION_NONE,
    COMPRESSION_QUANTIZATION,
    COMPRESSION_SPARSIFICATION,
    COMPRESSION_TOPK,
    COMPRESSION_RANDOM_K,
    COMPRESSION_QUANTUM
} CompressionType;
#endif

// Optimization algorithms
typedef enum OptimizationAlgorithm {
    ADAM,
    QUANTUM_ADAM,
    GEOMETRIC_ADAM,
    HYBRID_ADAM,
    SGD,
    LBFGS,
    NATURAL_GRADIENT
} OptimizationAlgorithm;

// Optimization configuration
typedef struct OptimizationConfig {
    OptimizationAlgorithm algorithm;
    double learning_rate;
    double beta1;                    // Adam momentum parameter
    double beta2;                    // Adam velocity parameter
    double epsilon;                  // Numerical stability
    double weight_decay;             // L2 regularization
    size_t model_size;               // Number of parameters
    bool use_quantum;                // Enable quantum gradient processing
    bool use_geometric;              // Enable geometric gradient processing
    bool noise_reduction;            // Enable quantum noise reduction
    CompressionType compression;     // Gradient compression type
    double compression_ratio;        // Target compression ratio
} OptimizationConfig;

// Gradient buffer structure for optimizer (internal use)
#ifndef GRAD_OPT_GRADIENT_BUFFER_DEFINED
#define GRAD_OPT_GRADIENT_BUFFER_DEFINED
typedef struct GradOptGradientBuffer {
    double* data;
    size_t size;
    bool is_compressed;
    CompressionType compression;
} GradOptGradientBuffer;
#endif

// Gradient optimizer structure
struct GradientOptimizer {
    OptimizerState* state;
    QuantumGradientState* quantum_state;
    GeometricProcessor* geometric_processor;
    GradOptGradientBuffer* gradient_buffer;
    GradOptGradientBuffer* compressed_buffer;
};

// Type alias for distributed_training_manager compatibility
typedef GradientOptimizer gradient_optimizer_t;

// Create and destroy
GradientOptimizer* init_gradient_optimizer(const OptimizationConfig* config);
void cleanup_gradient_optimizer(GradientOptimizer* optimizer);

// Gradient processing
void process_gradients(GradientOptimizer* optimizer, double* gradients, size_t size);
void preprocess_gradients(GradientOptimizer* optimizer, double* gradients, size_t size);

// Compression and decompression
void compress_gradients(GradientOptimizer* optimizer, const double* gradients, size_t size);
void decompress_gradients(GradientOptimizer* optimizer, const GradOptGradientBuffer* compressed,
                         double* gradients, size_t size);

// Internal helper functions (implemented in .c file)
OptimizerState* create_optimizer_state(const OptimizationConfig* config);
void cleanup_optimizer_state(OptimizerState* state);
void update_optimizer_state(OptimizerState* state, const double* gradients, size_t size);

GradOptGradientBuffer* create_gradient_buffer(size_t size);
void cleanup_gradient_buffer(GradOptGradientBuffer* buffer);

QuantumGradientState* init_quantum_gradient_state(size_t model_size);
void cleanup_quantum_gradient_state(QuantumGradientState* state);
void prepare_gradient_state(QuantumGradientState* state, const double* gradients, size_t size);
void execute_gradient_circuit(struct QuantumCircuit* circuit, struct QuantumState* state, size_t batch_size);
void measure_quantum_gradients(QuantumGradientState* state, double* gradients, size_t size);
void reduce_quantum_noise(double* gradients, size_t size);

GeometricProcessor* init_geometric_processor(void);
void cleanup_geometric_processor(GeometricProcessor* processor);
Manifold* compute_gradient_manifold(GeometricProcessor* processor, const double* gradients, size_t size);
void optimize_geometric_gradients(Manifold* manifold, double* gradients, size_t size);
void update_geometric_metrics(double* metrics, const Manifold* manifold);
void cleanup_manifold(Manifold* manifold);

// Optimization algorithms
void apply_adam_optimization(OptimizerState* state, double* gradients, size_t size);
void apply_quantum_adam(OptimizerState* state, double* gradients, size_t size);
void apply_geometric_adam(OptimizerState* state, double* gradients, size_t size);
void apply_hybrid_adam(OptimizerState* state, double* gradients, size_t size);

// Compression functions
void compress_quantum_gradients(QuantumGradientState* state, const double* gradients,
                               GradOptGradientBuffer* buffer, size_t size);
void decompress_quantum_gradients(QuantumGradientState* state, const GradOptGradientBuffer* compressed,
                                 double* gradients, size_t size);
void compress_geometric_gradients(GeometricProcessor* processor, const double* gradients,
                                 GradOptGradientBuffer* buffer, size_t size);
void decompress_geometric_gradients(GeometricProcessor* processor, const GradOptGradientBuffer* compressed,
                                   double* gradients, size_t size);
void compress_standard_gradients(const double* gradients, GradOptGradientBuffer* buffer, size_t size);
void decompress_standard_gradients(const GradOptGradientBuffer* compressed, double* gradients, size_t size);

// Wrapper functions for distributed_training_manager compatibility
gradient_optimizer_t* gradient_optimizer_create(void);
void gradient_optimizer_destroy(gradient_optimizer_t* optimizer);

#ifdef __cplusplus
}
#endif

#endif // GRADIENT_OPTIMIZER_H
