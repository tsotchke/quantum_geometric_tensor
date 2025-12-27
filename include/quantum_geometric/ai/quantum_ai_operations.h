#ifndef QUANTUM_AI_OPERATIONS_H
#define QUANTUM_AI_OPERATIONS_H

#include <stdbool.h>
#include <stddef.h>
#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/physics/advanced_geometry_types.h"

// Memory types are defined in quantum_geometric_types.h (qgt_memory_type_t)
// PhysicalConstraints is defined in quantum_geometric_types.h
// qgt_advanced_geometry_t is defined in advanced_geometry_types.h for AI/ML operations
// that require spin system and advanced geometric structures

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

// Embedding types for geometric embeddings
typedef enum {
    QGT_EMBED_EUCLIDEAN,     // Euclidean embeddings
    QGT_EMBED_HYPERBOLIC,    // Hyperbolic (Poincaré ball) embeddings
    QGT_EMBED_SPHERICAL      // Spherical embeddings
} qgt_embedding_type_t;

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

// Opaque type declarations - use consistent struct names from core headers
// TreeTensorNetwork is defined in tree_tensor_network.h as struct tree_tensor_network
#ifndef TREE_TENSOR_NETWORK_TYPEDEF_DEFINED
#define TREE_TENSOR_NETWORK_TYPEDEF_DEFINED
struct tree_tensor_network;
typedef struct tree_tensor_network TreeTensorNetwork;
#endif

typedef struct GeometricOptimizer GeometricOptimizer;

// AI operations use qgt_advanced_geometry_t for access to spin systems and geometry structures
// These are essential for Kähler/Calabi-Yau/G2 geometric ML operations

// Performance metrics for AI operations
typedef struct {
    double conversion_time;          // Time for tensor conversion
    double network_creation_time;    // Time for network creation
    double constraint_time;          // Time for constraint application
    size_t memory_usage;            // Memory usage in bytes
    double gpu_utilization;          // GPU utilization percentage
    size_t num_operations;          // Number of operations performed
} qgt_ai_performance_metrics_t;

// Core tensor operations - use advanced geometry type for spin/geometry access
qgt_advanced_geometry_t* qgt_ai_create_tensor(size_t dimension, size_t num_spins, qgt_memory_type_t mem_type);
void qgt_ai_free_tensor(qgt_advanced_geometry_t* tensor);
void qgt_ai_initialize_geometric_embeddings(qgt_advanced_geometry_t* tensor, qgt_embedding_type_t type);
void qgt_ai_initialize_random_state(qgt_advanced_geometry_t* tensor, unsigned int seed);

// Network operations
TreeTensorNetwork* qgt_ai_create_transformer_layer(const ModelConfig* config);
void qgt_ai_destroy_ttn(TreeTensorNetwork* network);
TreeTensorNetwork* qgt_ai_forward_geometric_network(TreeTensorNetwork** layers, size_t num_layers,
                                                    qgt_advanced_geometry_t* input, qgt_forward_type_t type);
TreeTensorNetwork* qgt_ai_forward_uncompressed_network(TreeTensorNetwork** layers, size_t num_layers,
                                                       qgt_advanced_geometry_t* input);
void qgt_ai_backward_geometric_network(TreeTensorNetwork* network, double loss,
                                       GeometricOptimizer* optimizer, qgt_backward_type_t type);

// Geometric operations
double qgt_ai_calculate_geometric_curvature(const TreeTensorNetwork* network);
qgt_advanced_geometry_t* qgt_ai_extract_geometric_properties(const TreeTensorNetwork* network);
double qgt_ai_compare_metric_tensors(const qgt_advanced_geometry_t* a, const qgt_advanced_geometry_t* b);
double qgt_ai_calculate_geometric_loss(const TreeTensorNetwork* output, const qgt_advanced_geometry_t* target,
                                       qgt_loss_type_t type);

// Physical constraint operations
qgt_error_t qgt_ai_apply_physical_constraints(qgt_advanced_geometry_t* state, const PhysicalConstraints* constraints);
double qgt_ai_calculate_total_energy(const qgt_advanced_geometry_t* state);
bool qgt_ai_verify_symmetry_constraints(const qgt_advanced_geometry_t* state, double tolerance);
bool qgt_ai_verify_causality_constraints(const qgt_advanced_geometry_t* state, double tolerance);

// Network analysis
size_t qgt_ai_count_network_parameters(const TreeTensorNetwork* network);
double qgt_ai_compare_tensor_outputs(const TreeTensorNetwork* a, const qgt_advanced_geometry_t* b, double tolerance);

// Distributed operations
void qgt_ai_initialize_distributed_training(const DistributedConfig* config);
bool qgt_ai_verify_distributed_consistency(const TreeTensorNetwork* network, double tolerance);

// Optimization
GeometricOptimizer* qgt_ai_create_geometric_optimizer(optimizer_type_t type, const TrainingConfig* config,
                                                      qgt_update_type_t update_type);
void qgt_ai_free_geometric_optimizer(GeometricOptimizer* optimizer);
void qgt_ai_update_geometric_parameters(TreeTensorNetwork** layers, size_t num_layers,
                                        GeometricOptimizer* optimizer, qgt_update_type_t type);

// Performance metrics
bool qgt_ai_get_performance_metrics(qgt_ai_performance_metrics_t* metrics);
bool qgt_ai_reset_performance_metrics(void);

// Constraint verification utilities
bool qgt_ai_verify_energy_constraint(const qgt_advanced_geometry_t* tensor, double threshold, double* energy);
bool qgt_ai_verify_gauge_constraint(const qgt_advanced_geometry_t* tensor, double tolerance);
bool qgt_ai_verify_locality_constraint(const qgt_advanced_geometry_t* tensor, double tolerance);
bool qgt_ai_verify_conservation_constraint(const qgt_advanced_geometry_t* tensor, double tolerance);

#endif // QUANTUM_AI_OPERATIONS_H
