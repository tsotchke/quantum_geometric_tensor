/**
 * @file tensor_network_operations.h
 * @brief Advanced tensor network operations with physical constraints
 * 
 * This header defines the interface for tensor network operations that incorporate
 * physical constraints from quantum field theory, holography, and topology.
 * The operations are optimized for performance and maintain physical consistency.
 */

#ifndef PHYSICSML_TENSOR_NETWORK_OPERATIONS_H
#define PHYSICSML_TENSOR_NETWORK_OPERATIONS_H

#include <physicsml/core/tensor.h>
#include <physicsml/core/error.h>
#include <complex.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct TreeNode TreeNode;
typedef struct TreeTensorNetwork TreeTensorNetwork;
typedef struct MinimalSurface MinimalSurface;

/**
 * @brief Physical constraints for tensor network optimization
 */
typedef struct {
    double energy_threshold;         // Energy convergence threshold
    double fidelity_threshold;       // Fidelity convergence threshold
    double symmetry_tolerance;       // Tolerance for symmetry violations
    double conservation_tolerance;   // Tolerance for conservation law violations
    double gauge_tolerance;         // Gauge symmetry violation tolerance
    double locality_tolerance;      // Locality constraint tolerance
    double renormalization_scale;   // RG flow scale parameter
    double causality_tolerance;     // Causality violation tolerance
} PhysicalConstraints;

/**
 * @brief Holographic constraints for tensor network optimization
 */
typedef struct {
    double boundary_error_threshold;     // Boundary reconstruction error threshold
    double bulk_energy_threshold;        // Bulk energy convergence threshold
    double entanglement_entropy_threshold; // Entanglement entropy threshold
    double area_law_tolerance;           // Area law violation tolerance
    double rt_formula_tolerance;         // Ryu-Takayanagi formula tolerance
    double bulk_locality_tolerance;      // Bulk locality constraint tolerance
    double radial_cutoff;               // Holographic radial cutoff
    double uv_cutoff;                   // UV regulator scale
} HolographicConstraints;

/**
 * @brief Topological constraints for tensor network optimization
 */
typedef struct {
    double winding_number_tolerance;     // Winding number quantization tolerance
    double braiding_phase_tolerance;     // Braiding phase quantization tolerance
    double anyonic_fusion_tolerance;     // Anyonic fusion rule tolerance
    double topological_order_tolerance;  // Topological order preservation tolerance
    double wilson_loop_tolerance;       // Wilson loop expectation tolerance
    double edge_mode_tolerance;         // Edge mode localization tolerance
    double monopole_charge_tolerance;   // Magnetic monopole charge tolerance
    double instanton_number_tolerance;  // Instanton number quantization tolerance
} TopologicalConstraints;

// Tree tensor network operations
TreeTensorNetwork* physicsml_ttn_create(size_t num_levels, size_t bond_dim, 
                                      size_t physical_dim, PhysicsMLDType dtype);
void physicsml_ttn_destroy(TreeTensorNetwork* ttn);
PhysicsMLTensor* physicsml_ttn_clone(const TreeTensorNetwork* ttn);
PhysicsMLError physicsml_ttn_compress(TreeTensorNetwork* ttn, double truncation_error);
PhysicsMLError physicsml_ttn_coarse_grain(TreeTensorNetwork* ttn, size_t num_levels);

// Network optimization with physical constraints
PhysicsMLError physicsml_optimize_physical_network(TreeTensorNetwork* ttn,
    const PhysicsMLTensor* target, const PhysicsMLTensor* hamiltonian,
    const PhysicsMLTensor* observables[], size_t num_observables,
    double learning_rate, size_t max_iterations);

PhysicsMLError physicsml_optimize_holographic_network(TreeTensorNetwork* ttn,
    const PhysicsMLTensor* boundary_data, const PhysicsMLTensor* bulk_hamiltonian,
    double learning_rate, size_t max_iterations);

PhysicsMLError physicsml_optimize_topological_network(TreeTensorNetwork* ttn,
    const PhysicsMLTensor* target, const PhysicsMLTensor* hamiltonian,
    double learning_rate, size_t max_iterations);

// Physical observables and measurements
double physicsml_compute_expectation_value(TreeTensorNetwork* ttn,
                                         const PhysicsMLTensor* operator);
double physicsml_compute_entanglement_entropy(TreeTensorNetwork* ttn);
double physicsml_compute_boundary_error(TreeTensorNetwork* ttn,
                                      const PhysicsMLTensor* boundary_data);
double physicsml_compute_topological_error(TreeTensorNetwork* ttn,
                                         const PhysicsMLTensor* target);
double physicsml_compute_wilson_loop(TreeTensorNetwork* ttn,
                                   const size_t* loop_path,
                                   size_t path_length);
double physicsml_compute_monopole_charge(TreeTensorNetwork* ttn,
                                       const size_t* surface_points,
                                       size_t num_points);
double physicsml_compute_instanton_number(TreeTensorNetwork* ttn);

// Network contraction and manipulation
PhysicsMLTensor* physicsml_contract_network(TreeTensorNetwork* ttn);
PhysicsMLTensor* physicsml_contract_boundary(TreeTensorNetwork* ttn);
PhysicsMLTensor* physicsml_contract_topological(TreeTensorNetwork* ttn);
PhysicsMLTensor* physicsml_contract_bulk_region(TreeTensorNetwork* ttn,
                                              const size_t* region,
                                              size_t region_size);
PhysicsMLTensor* physicsml_contract_causal_cone(TreeTensorNetwork* ttn,
                                              const size_t* points,
                                              size_t num_points);

// Parameter extraction and updates
size_t physicsml_count_network_parameters(const TreeTensorNetwork* ttn);
PhysicsMLError physicsml_extract_network_parameters(TreeTensorNetwork* ttn,
    PhysicsMLTensor** parameters, size_t start_index);
PhysicsMLError physicsml_update_network_parameters(TreeTensorNetwork* ttn,
    PhysicsMLTensor** parameters, size_t start_index);

// Physical constraint verification
bool physicsml_verify_winding_numbers(TreeTensorNetwork* ttn, double tolerance);
bool physicsml_verify_braiding_phases(TreeTensorNetwork* ttn, double tolerance);
bool physicsml_verify_fusion_rules(TreeTensorNetwork* ttn, double tolerance);
bool physicsml_verify_topological_order(TreeTensorNetwork* ttn, double tolerance);
bool physicsml_verify_area_law(TreeTensorNetwork* ttn, double tolerance);
bool physicsml_verify_gauge_invariance(TreeTensorNetwork* ttn, double tolerance);
bool physicsml_verify_locality(TreeTensorNetwork* ttn, double tolerance);
bool physicsml_verify_causality(TreeTensorNetwork* ttn, double tolerance);
bool physicsml_verify_unitarity(TreeTensorNetwork* ttn, double tolerance);

// Advanced operations
MinimalSurface* physicsml_compute_minimal_surfaces(const TreeTensorNetwork* ttn);
void physicsml_free_minimal_surfaces(MinimalSurface* surfaces);
double* physicsml_compute_entanglement_entropies(const TreeTensorNetwork* ttn);
size_t physicsml_count_bipartitions(const TreeTensorNetwork* ttn);
double physicsml_compute_boundary_area(const TreeTensorNetwork* ttn, size_t partition);
double physicsml_compute_bulk_curvature(const TreeTensorNetwork* ttn);
double physicsml_compute_beta_function(const TreeTensorNetwork* ttn);
double physicsml_compute_anomalous_dimension(const TreeTensorNetwork* ttn);

// Utility functions
PhysicsMLTensor* physicsml_compute_reduced_density_matrix(const PhysicsMLTensor* state);
double physicsml_compute_von_neumann_entropy(const PhysicsMLTensor* rho);
complex double physicsml_project_to_winding(complex double value, double tolerance);
complex double physicsml_project_to_braiding_phase(complex double value, double tolerance);
complex double physicsml_project_to_fusion_channel(complex double value, double tolerance);
complex double physicsml_project_to_topological_sector(complex double value, double tolerance);
complex double physicsml_project_to_gauge_orbit(complex double value, double tolerance);
complex double physicsml_project_to_physical_state(complex double value, double tolerance);

#ifdef __cplusplus
}
#endif

#endif // PHYSICSML_TENSOR_NETWORK_OPERATIONS_H
