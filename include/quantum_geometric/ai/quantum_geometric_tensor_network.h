#ifndef QUANTUM_GEOMETRIC_TENSOR_NETWORK_H
#define QUANTUM_GEOMETRIC_TENSOR_NETWORK_H

#include <stdbool.h>
#include <stddef.h>
#include <complex.h>
#include "quantum_geometric/core/quantum_geometric_core.h"

// Memory types
typedef enum {
    QGT_MEM_STANDARD,
    QGT_MEM_HUGE_PAGES,
    QGT_MEM_PINNED,
    QGT_MEM_UNIFIED
} qgt_memory_type_t;

// Return codes
#define QGT_SUCCESS 0
#define PHYSICSML_SUCCESS 0

// Quantum geometric tensor structure
typedef struct {
    size_t dimension;                // Tensor dimension
    size_t num_spins;               // Number of spins
    struct {
        complex double* spin_states; // Spin states
        double* metric_tensor;       // Metric tensor
    } spin_system;
    struct {
        double* metric_tensor;       // Geometric metric tensor
    } geometry;
} quantum_geometric_tensor;

// Physical constraints
typedef struct {
    double energy_threshold;         // Energy threshold
    double fidelity_threshold;       // Fidelity threshold
    double symmetry_tolerance;       // Symmetry preservation tolerance
    double conservation_tolerance;   // Conservation law tolerance
    double gauge_tolerance;          // Gauge invariance tolerance
    double locality_tolerance;       // Locality constraint tolerance
    double renormalization_scale;    // Renormalization scale
    double causality_tolerance;      // Causality preservation tolerance
} PhysicalConstraints;

// PhysicsML tensor (opaque type)
typedef struct PhysicsMLTensor PhysicsMLTensor;

// Tree tensor network (opaque type)
typedef struct TreeTensorNetwork TreeTensorNetwork;

// Core tensor operations
quantum_geometric_tensor* create_quantum_tensor(size_t dimension,
                                              size_t num_spins,
                                              qgt_memory_type_t mem_type);
void free_quantum_tensor(quantum_geometric_tensor* tensor);

// Tensor conversion functions
PhysicsMLTensor* qgt_to_physicsml_tensor(const quantum_geometric_tensor* qgt);
quantum_geometric_tensor* physicsml_to_qgt_tensor(const PhysicsMLTensor* pml);
bool verify_tensor_consistency(const quantum_geometric_tensor* qgt,
                             const PhysicsMLTensor* pml,
                             double tolerance);

// Tensor network operations
TreeTensorNetwork* create_geometric_network(const quantum_geometric_tensor* qgt,
                                          size_t bond_dimension);
void physicsml_ttn_destroy(TreeTensorNetwork* network);
quantum_geometric_tensor* extract_geometric_properties(const TreeTensorNetwork* network);
PhysicsMLTensor* physicsml_contract_network(const TreeTensorNetwork* network);

// Physical constraint operations
int apply_physical_constraints(quantum_geometric_tensor* qgt,
                             const PhysicalConstraints* constraints);
int apply_geometric_constraints(TreeTensorNetwork* network,
                              const quantum_geometric_tensor* qgt);

// Performance monitoring
typedef struct {
    double conversion_time;          // Time for tensor conversion
    double network_creation_time;    // Time for network creation
    double constraint_time;          // Time for constraint application
    size_t memory_usage;            // Memory usage in bytes
    double gpu_utilization;          // GPU utilization percentage
    size_t num_operations;          // Number of operations performed
} performance_metrics_t;

bool get_performance_metrics(performance_metrics_t* metrics);
bool reset_performance_metrics(void);

// Utility functions
bool verify_energy_constraint(const quantum_geometric_tensor* qgt,
                            double threshold,
                            double* energy);
bool verify_symmetry_constraint(const quantum_geometric_tensor* qgt,
                              double tolerance);
bool verify_conservation_constraint(const quantum_geometric_tensor* qgt,
                                  double tolerance);
bool verify_gauge_constraint(const quantum_geometric_tensor* qgt,
                           double tolerance);
bool verify_locality_constraint(const quantum_geometric_tensor* qgt,
                              double tolerance);
bool verify_causality_constraint(const quantum_geometric_tensor* qgt,
                               double tolerance);

#endif // QUANTUM_GEOMETRIC_TENSOR_NETWORK_H
