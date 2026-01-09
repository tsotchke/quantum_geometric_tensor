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

// =============================================================================
// Backward Compatibility Type Aliases
// =============================================================================
// Legacy code uses 'quantum_geometric_tensor' with spin_system and geometry
// These are now provided by qgt_advanced_geometry_t which has these fields:
//   - spin_system: qgt_spin_system_t with spin_states, spin_operators, etc.
//   - geometry: qgt_geometry_t with metric_tensor, kahler_metric, etc.
#ifndef QUANTUM_GEOMETRIC_TENSOR_LEGACY_ALIAS
#define QUANTUM_GEOMETRIC_TENSOR_LEGACY_ALIAS
typedef qgt_advanced_geometry_t quantum_geometric_tensor;
#endif

// =============================================================================
// PhysicsML Integration Types
// =============================================================================

// PhysicsML tensor structure for ML framework integration
#ifndef PHYSICSML_TENSOR_STRUCT_DEFINED
#define PHYSICSML_TENSOR_STRUCT_DEFINED
typedef struct PhysicsMLTensor {
    ComplexDouble* data;          // Tensor data in ML framework format
    size_t* shape;               // Tensor shape array
    size_t ndim;                 // Number of dimensions
    size_t size;                 // Total number of elements
    bool is_contiguous;          // Whether data is contiguous in memory
    void* ml_handle;             // Handle to external ML framework tensor
    char* device;                // Device location ("cpu", "gpu", etc.)
} PhysicsMLTensor;
#endif

// TreeTensorNetwork is defined in tree_tensor_network.h
// Forward declare here for AI operations
#ifndef TREE_TENSOR_NETWORK_TYPEDEF_DEFINED
#define TREE_TENSOR_NETWORK_TYPEDEF_DEFINED
struct tree_tensor_network;
typedef struct tree_tensor_network TreeTensorNetwork;
#endif

// =============================================================================
// Core Tensor Creation/Destruction
// =============================================================================

// Create quantum geometric tensor with advanced geometry support
qgt_advanced_geometry_t* qgt_ai_create_tensor(size_t dimension,
                                               size_t num_spins,
                                               qgt_memory_type_t mem_type);
void qgt_ai_free_tensor(qgt_advanced_geometry_t* tensor);

// Legacy function aliases for backward compatibility
#define create_quantum_tensor qgt_ai_create_tensor
#define free_quantum_tensor qgt_ai_free_tensor

// =============================================================================
// Tensor Conversion Functions
// =============================================================================

// Convert QGT tensor to PhysicsML tensor format
PhysicsMLTensor* qgt_to_physicsml_tensor(const qgt_advanced_geometry_t* qgt);

// Convert PhysicsML tensor back to QGT format
qgt_advanced_geometry_t* physicsml_to_qgt_tensor(const PhysicsMLTensor* pml);

// Verify consistency between QGT and PhysicsML tensor representations
bool verify_tensor_consistency(const qgt_advanced_geometry_t* qgt,
                               const PhysicsMLTensor* pml,
                               double tolerance);

// Destroy PhysicsML tensor and free resources
void physicsml_tensor_destroy(PhysicsMLTensor* tensor);

// =============================================================================
// Tensor Network Operations
// =============================================================================

// Create geometric tensor network from quantum geometric tensor
TreeTensorNetwork* qgt_ai_create_geometric_network(const qgt_advanced_geometry_t* qgt,
                                                   size_t bond_dimension);

// Destroy tensor network and free resources
void qgt_ai_destroy_network(TreeTensorNetwork* network);

// Extract geometric properties from tensor network
qgt_advanced_geometry_t* qgt_ai_extract_geometric_properties(const TreeTensorNetwork* network);

// Contract tensor network to single PhysicsML tensor
PhysicsMLTensor* physicsml_contract_network(const TreeTensorNetwork* network);

// Destroy tree tensor network
void physicsml_ttn_destroy(TreeTensorNetwork* network);

// Legacy function aliases for backward compatibility
#define create_geometric_network qgt_ai_create_geometric_network
#define extract_geometric_properties qgt_ai_extract_geometric_properties

// =============================================================================
// Physical Constraint Operations
// =============================================================================

// Apply physical constraints to quantum geometric tensor
qgt_error_t qgt_ai_apply_physical_constraints(qgt_advanced_geometry_t* state,
                                              const PhysicalConstraints* constraints);

// Apply geometric constraints from tensor to network
int qgt_ai_apply_geometric_constraints(TreeTensorNetwork* network,
                                       const qgt_advanced_geometry_t* qgt);

// Legacy function aliases for backward compatibility
#define apply_physical_constraints qgt_ai_apply_physical_constraints
#define apply_geometric_constraints qgt_ai_apply_geometric_constraints

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
