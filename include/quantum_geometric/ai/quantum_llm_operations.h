#ifndef QUANTUM_LLM_OPERATIONS_H
#define QUANTUM_LLM_OPERATIONS_H

#include <stdbool.h>
#include <stddef.h>
#include "quantum_geometric/core/quantum_geometric_core.h"

// Architecture types
typedef enum {
    ARCHITECTURE_DIFFERENTIAL,
    ARCHITECTURE_GEOMETRIC,
    ARCHITECTURE_HYBRID,
    ARCHITECTURE_ADAPTIVE
} architecture_type_t;

// Attention types
typedef enum {
    ATTENTION_GEOMETRIC,
    ATTENTION_QUANTUM,
    ATTENTION_HYBRID,
    ATTENTION_ADAPTIVE
} attention_type_t;

// Optimization methods
typedef enum {
    OPTIMIZATION_NATURAL_GRADIENT,
    OPTIMIZATION_QUANTUM_NATURAL,
    OPTIMIZATION_GEOMETRIC,
    OPTIMIZATION_HYBRID
} optimization_method_t;

// Protection types
typedef enum {
    PROTECTION_GEOMETRIC,
    PROTECTION_TOPOLOGICAL,
    PROTECTION_QUANTUM,
    PROTECTION_ADAPTIVE
} protection_type_t;

// Scheduling types
typedef enum {
    SCHEDULING_STATIC,
    SCHEDULING_ADAPTIVE,
    SCHEDULING_GEOMETRIC,
    SCHEDULING_QUANTUM
} scheduling_type_t;

// Precision types
typedef enum {
    PRECISION_FP32,
    PRECISION_FP16,
    PRECISION_BF16,
    PRECISION_MIXED_BF16
} precision_type_t;

// Pipeline types
typedef enum {
    PIPELINE_STANDARD,
    PIPELINE_GEOMETRIC,
    PIPELINE_QUANTUM,
    PIPELINE_HYBRID
} pipeline_type_t;

// Accelerator types
typedef enum {
    ACCELERATOR_AUTO,
    ACCELERATOR_GPU,
    ACCELERATOR_QPU,
    ACCELERATOR_HYBRID
} accelerator_type_t;

// Memory types
typedef enum {
    MEMORY_STANDARD,
    MEMORY_UNIFIED,
    MEMORY_QUANTUM,
    MEMORY_HYBRID
} memory_type_t;

// Compute types
typedef enum {
    COMPUTE_STANDARD,
    COMPUTE_TENSOR,
    COMPUTE_QUANTUM,
    COMPUTE_HYBRID
} compute_type_t;

// Architecture configuration
typedef struct {
    architecture_type_t type;     // Architecture type
    attention_type_t attention;   // Attention mechanism
    uint32_t layers;             // Number of layers
    uint32_t dim;                // Model dimension
    uint32_t vocab;              // Vocabulary size
} llm_architecture_t;

// Geometry configuration
typedef struct {
    manifold_type_t manifold;    // Geometric manifold
    metric_type_t metric;        // Metric tensor type
    connection_type_t connection; // Connection type
} llm_geometry_t;

// Optimization configuration
typedef struct {
    optimization_method_t method; // Optimization method
    protection_type_t protection; // Protection mechanism
    scheduling_type_t scheduling; // Learning rate scheduling
    precision_type_t precision;   // Computation precision
    pipeline_type_t pipeline;     // Pipeline configuration
} llm_optimization_t;

// Hardware configuration
typedef struct {
    accelerator_type_t accelerator; // Hardware accelerator
    memory_type_t memory;           // Memory configuration
    compute_type_t compute;         // Compute configuration
} llm_hardware_t;

// Training configuration
typedef struct {
    uint32_t batch_size;           // Training batch size
    bool gradient_checkpointing;    // Enable gradient checkpointing
    bool zero_redundancy;          // Enable ZeRO optimization
} llm_training_t;

// Combined LLM configuration
typedef struct {
    llm_architecture_t architecture; // Architecture configuration
    llm_geometry_t geometry;         // Geometry configuration
    llm_optimization_t optimization; // Optimization configuration
    llm_hardware_t hardware;         // Hardware configuration
    llm_training_t training;         // Training configuration
} llm_config_t;

// Statistics configuration
typedef struct {
    bool track_attention;           // Track attention metrics
    bool monitor_geometry;          // Monitor geometric properties
} llm_stats_t;

// Training configuration
typedef struct {
    uint32_t epochs;               // Number of epochs
    float learning_rate;           // Learning rate
    uint32_t warmup_steps;         // Learning rate warmup steps
    uint32_t max_steps;            // Maximum training steps
    uint32_t save_steps;           // Checkpoint save frequency
    uint32_t eval_steps;           // Evaluation frequency
    struct {
        bool wandb;                // Enable Weights & Biases logging
        bool tensorboard;          // Enable TensorBoard logging
        bool console;              // Enable console logging
    } logging;
} training_config_t;

// Results and metrics
typedef struct {
    float complexity;              // Attention complexity
    float protection;              // Geometric protection level
    float time_ms;                 // Processing time in milliseconds
} llm_result_t;

typedef struct {
    float loss;                    // Training loss
    float perplexity;              // Model perplexity
    float learning_rate;           // Current learning rate
    float throughput;              // Tokens per second
    float gpu_memory;              // GPU memory usage
    float step_time;               // Time per step
} training_result_t;

// Opaque type declarations
typedef struct quantum_llm_t quantum_llm_t;
typedef struct dataset_t dataset_t;

// Core LLM operations
quantum_llm_t* quantum_llm_create(const llm_config_t* config);
void quantum_llm_free(quantum_llm_t* model);

// Processing operations
llm_result_t quantum_llm_process(quantum_llm_t* model,
                                const char* input_text,
                                const llm_stats_t* stats);

// Training operations
training_result_t quantum_llm_train(quantum_llm_t* model,
                                   const dataset_t* dataset,
                                   const training_config_t* config);

// Performance monitoring
bool get_llm_metrics(const quantum_llm_t* model,
                    llm_result_t* result);
bool get_training_metrics(const quantum_llm_t* model,
                         training_result_t* result);

#endif // QUANTUM_LLM_OPERATIONS_H
