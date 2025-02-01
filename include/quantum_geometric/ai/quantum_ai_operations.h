#ifndef QUANTUM_AI_OPERATIONS_H
#define QUANTUM_AI_OPERATIONS_H

#include <stdbool.h>
#include <stddef.h>
#include "quantum_geometric/core/quantum_geometric_core.h"

// Memory types
typedef enum {
    QGT_MEM_STANDARD,
    QGT_MEM_HUGE_PAGES,
    QGT_MEM_PINNED,
    QGT_MEM_UNIFIED
} qgt_memory_type_t;

// Forward operation types
typedef enum {
    QGT_FORWARD_STANDARD,
    QGT_FORWARD_DISTRIBUTED,
    QGT_FORWARD_CHECKPOINTED
} qgt_forward_type_t;

// Backward operation types
typedef enum {
    QGT_BACKWARD_STANDARD,
    QGT_BACKWARD_DISTRIBUTED,
    QGT_BACKWARD_ACCUMULATED
} qgt_backward_type_t;

// Update types
typedef enum {
    QGT_UPDATE_STANDARD,
    QGT_UPDATE_PRESERVE_GEOMETRY,
    QGT_UPDATE_PRESERVE_TOPOLOGY
} qgt_update_type_t;

// Loss types
typedef enum {
    QGT_LOSS_EUCLIDEAN,
    QGT_LOSS_HYPERBOLIC,
    QGT_LOSS_SPHERICAL
} qgt_loss_type_t;

// Optimizer types
typedef enum {
    OPTIMIZER_SGD,
    OPTIMIZER_ADAM,
    OPTIMIZER_GEOMETRIC,
    OPTIMIZER_QUANTUM
} optimizer_type_t;

// Model configuration
typedef struct {
    size_t hidden_dim;          // Hidden dimension
    size_t num_layers;          // Number of layers
    size_t num_heads;           // Number of attention heads
    size_t vocab_size;          // Vocabulary size
    size_t seq_length;          // Sequence length
    size_t bond_dim;            // Bond dimension for tensor networks
} ModelConfig;

// Physical constraints
typedef struct {
    double energy_threshold;     // Energy threshold
    double symmetry_tolerance;   // Symmetry preservation tolerance
    double conservation_tolerance; // Conservation law tolerance
    double gauge_tolerance;      // Gauge invariance tolerance
    double locality_tolerance;   // Locality constraint tolerance
    double renormalization_scale; // Renormalization scale
    double causality_tolerance;  // Causality preservation tolerance
} PhysicalConstraints;

// Distributed configuration
typedef struct {
    size_t world_size;          // Number of processes
    size_t pipeline_stages;     // Number of pipeline stages
    size_t tensor_parallel;     // Tensor parallelism degree
    bool activation_checkpointing; // Use activation checkpointing
    size_t zero_optimization_stage; // ZeRO optimization stage
    bool mixed_precision;       // Use mixed precision training
} DistributedConfig;

// Training configuration
typedef struct {
    size_t batch_size;          // Batch size
    double learning_rate;       // Learning rate
    size_t warmup_steps;        // Learning rate warmup steps
    size_t total_steps;         // Total training steps
    double weight_decay;        // Weight decay factor
    double gradient_clipping;   // Gradient clipping threshold
} TrainingConfig;

// Opaque type declarations
typedef struct TreeTensorNetwork TreeTensorNetwork;
typedef struct GeometricOptimizer GeometricOptimizer;
typedef struct quantum_geometric_tensor quantum_geometric_tensor;

// Core tensor operations
quantum_geometric_tensor* create_quantum_tensor(size_t rows, size_t cols, qgt_memory_type_t mem_type);
void free_quantum_tensor(quantum_geometric_tensor* tensor);
void initialize_geometric_embeddings(quantum_geometric_tensor* tensor, qgt_loss_type_t type);
void initialize_random_state(quantum_geometric_tensor* tensor, unsigned int seed);

// Network operations
TreeTensorNetwork* create_transformer_layer(const ModelConfig* config);
void physicsml_ttn_destroy(TreeTensorNetwork* network);
TreeTensorNetwork* forward_geometric_network(TreeTensorNetwork** layers, size_t num_layers,
                                          quantum_geometric_tensor* input, qgt_forward_type_t type);
TreeTensorNetwork* forward_uncompressed_network(TreeTensorNetwork** layers, size_t num_layers,
                                              quantum_geometric_tensor* input);
void backward_geometric_network(TreeTensorNetwork* network, double loss,
                              GeometricOptimizer* optimizer, qgt_backward_type_t type);

// Geometric operations
double calculate_geometric_curvature(const TreeTensorNetwork* network);
quantum_geometric_tensor* extract_geometric_properties(const TreeTensorNetwork* network);
double compare_metric_tensors(const quantum_geometric_tensor* a, const quantum_geometric_tensor* b);
double calculate_geometric_loss(const TreeTensorNetwork* output, const quantum_geometric_tensor* target,
                              qgt_loss_type_t type);

// Physical constraint operations
qgt_error_t apply_physical_constraints(quantum_geometric_tensor* state, const PhysicalConstraints* constraints);
double calculate_total_energy(const quantum_geometric_tensor* state);
bool verify_symmetry_constraints(const quantum_geometric_tensor* state, double tolerance);
bool verify_causality_constraints(const quantum_geometric_tensor* state, double tolerance);

// Network analysis
size_t count_network_parameters(const TreeTensorNetwork* network);
double compare_tensor_outputs(const TreeTensorNetwork* a, const quantum_geometric_tensor* b, double tolerance);

// Distributed operations
void initialize_distributed_training(const DistributedConfig* config);
bool verify_distributed_consistency(const TreeTensorNetwork* network, double tolerance);

// Optimization
GeometricOptimizer* create_geometric_optimizer(optimizer_type_t type, const TrainingConfig* config,
                                             qgt_update_type_t update_type);
void free_geometric_optimizer(GeometricOptimizer* optimizer);
void update_geometric_parameters(TreeTensorNetwork** layers, size_t num_layers,
                               GeometricOptimizer* optimizer, qgt_update_type_t type);

#endif // QUANTUM_AI_OPERATIONS_H
