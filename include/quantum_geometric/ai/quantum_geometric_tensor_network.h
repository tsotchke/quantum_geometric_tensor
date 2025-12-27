#ifndef QUANTUM_GEOMETRIC_TENSOR_NETWORK_AI_H
#define QUANTUM_GEOMETRIC_TENSOR_NETWORK_AI_H

#include <stdbool.h>
#include <stddef.h>
#include <complex.h>
#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/physics/advanced_geometry_types.h"
#include "quantum_geometric/ai/quantum_ai_operations.h"

// AI operations use qgt_advanced_geometry_t for access to:
// - spin_system (spin states, operators, spin-foam metric)
// - geometry (Kähler metric, Ricci tensor, Calabi-Yau form, etc.)
// These are essential for physics-informed ML operations

// Return codes (may already be defined)
#ifndef QGT_SUCCESS
#define QGT_SUCCESS 0
#endif
#ifndef PHYSICSML_SUCCESS
#define PHYSICSML_SUCCESS 0
#endif

// PhysicsML tensor (opaque type for external ML framework integration)
typedef struct PhysicsMLTensor PhysicsMLTensor;

// TreeTensorNetwork is defined in tree_tensor_network.h
// Forward declare here for AI operations
#ifndef TREE_TENSOR_NETWORK_TYPEDEF_DEFINED
#define TREE_TENSOR_NETWORK_TYPEDEF_DEFINED
struct tree_tensor_network;
typedef struct tree_tensor_network TreeTensorNetwork;
#endif

// Core tensor operations - use advanced geometry type for spin/geometry access
qgt_advanced_geometry_t* qgt_ai_create_tensor(size_t dimension,
                                               size_t num_spins,
                                               qgt_memory_type_t mem_type);
void qgt_ai_free_tensor(qgt_advanced_geometry_t* tensor);

// Tensor conversion functions
PhysicsMLTensor* qgt_to_physicsml_tensor(const qgt_advanced_geometry_t* qgt);
qgt_advanced_geometry_t* physicsml_to_qgt_tensor(const PhysicsMLTensor* pml);
bool verify_tensor_consistency(const qgt_advanced_geometry_t* qgt,
                               const PhysicsMLTensor* pml,
                               double tolerance);

// Tensor network operations
TreeTensorNetwork* qgt_ai_create_geometric_network(const qgt_advanced_geometry_t* qgt,
                                                   size_t bond_dimension);
void qgt_ai_destroy_network(TreeTensorNetwork* network);
qgt_advanced_geometry_t* qgt_ai_extract_geometric_properties(const TreeTensorNetwork* network);
PhysicsMLTensor* physicsml_contract_network(const TreeTensorNetwork* network);

// Physical constraint operations
qgt_error_t qgt_ai_apply_physical_constraints(qgt_advanced_geometry_t* state,
                                              const PhysicalConstraints* constraints);
int qgt_ai_apply_geometric_constraints(TreeTensorNetwork* network,
                                       const qgt_advanced_geometry_t* qgt);

// Performance monitoring - qgt_ai_performance_metrics_t defined in quantum_ai_operations.h
// Forward declare the functions here
bool qgt_ai_get_performance_metrics(qgt_ai_performance_metrics_t* metrics);
bool qgt_ai_reset_performance_metrics(void);

// Utility functions for constraint verification
bool qgt_ai_verify_energy_constraint(const qgt_advanced_geometry_t* tensor,
                                     double threshold,
                                     double* energy);
bool qgt_ai_verify_symmetry_constraint(const qgt_advanced_geometry_t* tensor,
                                       double tolerance);
bool qgt_ai_verify_conservation_constraint(const qgt_advanced_geometry_t* tensor,
                                           double tolerance);
bool qgt_ai_verify_gauge_constraint(const qgt_advanced_geometry_t* tensor,
                                    double tolerance);
bool qgt_ai_verify_locality_constraint(const qgt_advanced_geometry_t* tensor,
                                       double tolerance);
bool qgt_ai_verify_causality_constraint(const qgt_advanced_geometry_t* tensor,
                                        double tolerance);

#endif // QUANTUM_GEOMETRIC_TENSOR_NETWORK_AI_H
