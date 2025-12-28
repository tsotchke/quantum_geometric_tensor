/**
 * @file quantum_geometric_hamiltonian.h
 * @brief Quantum Hamiltonian construction with geometric structure
 *
 * Provides facilities for constructing, manipulating, and analyzing
 * quantum Hamiltonians with underlying geometric structure, including
 * molecular Hamiltonians, lattice models, and topological systems.
 */

#ifndef QUANTUM_GEOMETRIC_HAMILTONIAN_H
#define QUANTUM_GEOMETRIC_HAMILTONIAN_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

// Portable complex type definition
typedef double _Complex qgt_complex_t;

// Forward declarations
struct quantum_state;
struct geometric_tensor;

// =============================================================================
// Hamiltonian Types and Enums
// =============================================================================

/**
 * Geometric Hamiltonian types
 */
typedef enum {
    GEOM_HAMILTONIAN_GENERIC,        // Generic geometric Hamiltonian
    GEOM_HAMILTONIAN_MOLECULAR,      // Second-quantized molecular
    GEOM_HAMILTONIAN_LATTICE,        // Lattice model with geometry
    GEOM_HAMILTONIAN_TOPOLOGICAL,    // Topological Hamiltonian
    GEOM_HAMILTONIAN_SPIN_ORBIT,     // Spin-orbit coupled
    GEOM_HAMILTONIAN_GAUGE_FIELD,    // Gauge field theory
    GEOM_HAMILTONIAN_CURVED_SPACE,   // Hamiltonian on curved manifold
    GEOM_HAMILTONIAN_BERRY_PHASE     // Berry phase Hamiltonian
} GeometricHamiltonianType;

/**
 * Fermion mapping types
 */
typedef enum {
    FERMION_MAP_JORDAN_WIGNER,       // Jordan-Wigner transformation
    FERMION_MAP_BRAVYI_KITAEV,       // Bravyi-Kitaev transformation
    FERMION_MAP_PARITY,              // Parity mapping
    FERMION_MAP_TERNARY_TREE,        // Ternary tree mapping
    FERMION_MAP_SUPERFAST            // Superfast encoding
} FermionMappingType;

/**
 * Integral types for molecular Hamiltonians
 */
typedef enum {
    INTEGRAL_ONE_ELECTRON,           // One-electron integrals h_pq
    INTEGRAL_TWO_ELECTRON,           // Two-electron integrals g_pqrs
    INTEGRAL_NUCLEAR,                // Nuclear repulsion energy
    INTEGRAL_SPIN_ORBIT              // Spin-orbit coupling integrals
} IntegralType;

/**
 * Symmetry types
 */
typedef enum {
    SYMMETRY_NONE,                   // No symmetry
    SYMMETRY_PARTICLE_NUMBER,        // Particle number conservation
    SYMMETRY_SPIN_Z,                 // Sz conservation
    SYMMETRY_POINT_GROUP,            // Point group symmetry
    SYMMETRY_TRANSLATION,            // Translational symmetry
    SYMMETRY_TIME_REVERSAL           // Time-reversal symmetry
} SymmetryType;

// =============================================================================
// Molecular Hamiltonian Structures
// =============================================================================

/**
 * One-electron integrals
 */
typedef struct {
    qgt_complex_t* values;          // h_pq values
    size_t num_orbitals;
    size_t num_spin_orbitals;
    bool is_spin_restricted;
} OneElectronIntegrals;

/**
 * Two-electron integrals (chemist notation: (pq|rs))
 */
typedef struct {
    qgt_complex_t* values;          // g_pqrs values
    size_t num_orbitals;
    size_t num_spin_orbitals;
    size_t* nonzero_indices;         // For sparse storage
    size_t num_nonzero;
    bool use_8fold_symmetry;
    bool is_sparse;
} TwoElectronIntegrals;

/**
 * Active space configuration
 */
typedef struct {
    size_t num_active_electrons;
    size_t num_active_orbitals;
    size_t* active_orbital_indices;
    size_t num_frozen_core;
    size_t num_frozen_virtual;
    double frozen_core_energy;
} ActiveSpaceConfig;

/**
 * Molecular Hamiltonian
 */
typedef struct {
    OneElectronIntegrals* one_electron;
    TwoElectronIntegrals* two_electron;
    double nuclear_repulsion;
    ActiveSpaceConfig* active_space;
    size_t num_electrons;
    size_t num_orbitals;
    size_t num_spin_orbitals;
    FermionMappingType mapping;
    SymmetryType* symmetries;
    size_t num_symmetries;
    double* orbital_energies;
    char* basis_name;
} MolecularHamiltonian;

// =============================================================================
// Geometric Hamiltonian Structures
// =============================================================================

/**
 * Geometric term in Hamiltonian
 */
typedef struct {
    qgt_complex_t coefficient;
    size_t* qubit_indices;
    char* pauli_string;              // e.g., "XYZII"
    size_t num_qubits;
    double* geometric_weight;        // Weight from geometric structure
} GeometricTerm;

/**
 * Geometric Hamiltonian
 */
typedef struct {
    GeometricHamiltonianType type;
    GeometricTerm* terms;
    size_t num_terms;
    size_t num_qubits;

    // Geometric structure
    struct geometric_tensor* metric;       // Metric tensor
    struct geometric_tensor* connection;   // Connection coefficients
    struct geometric_tensor* curvature;    // Curvature tensor

    // Spectral properties
    double* eigenvalues;
    size_t num_computed_eigenvalues;
    double ground_state_energy;
    double energy_gap;

    // Symmetry information
    SymmetryType* symmetries;
    size_t num_symmetries;
    int* quantum_numbers;

    // Sparsity information
    size_t num_nonzero_terms;
    double sparsity_ratio;
} GeometricHamiltonian;

/**
 * Berry phase Hamiltonian parameters
 */
typedef struct {
    double* parameter_point;         // Point in parameter space
    size_t num_parameters;
    qgt_complex_t* berry_connection; // Berry connection components
    double* berry_curvature;         // Berry curvature tensor
    size_t band_index;               // Band for Berry phase
} BerryPhaseParams;

/**
 * Gauge field Hamiltonian parameters
 */
typedef struct {
    size_t gauge_group_dim;          // Dimension of gauge group
    qgt_complex_t* gauge_field;     // Gauge field configuration
    double coupling_constant;        // Gauge coupling
    bool include_matter;             // Include matter fields
    size_t num_colors;               // For QCD-like theories
} GaugeFieldParams;

// =============================================================================
// Construction Functions
// =============================================================================

/**
 * Create molecular Hamiltonian from integrals
 */
int molecular_hamiltonian_create(
    MolecularHamiltonian** H,
    OneElectronIntegrals* one_e,
    TwoElectronIntegrals* two_e,
    double nuclear_repulsion,
    size_t num_electrons
);

/**
 * Destroy molecular Hamiltonian
 */
void molecular_hamiltonian_destroy(MolecularHamiltonian* H);

/**
 * Create geometric Hamiltonian
 */
int geometric_hamiltonian_create(
    GeometricHamiltonian** H,
    GeometricHamiltonianType type,
    size_t num_qubits
);

/**
 * Destroy geometric Hamiltonian
 */
void geometric_hamiltonian_destroy(GeometricHamiltonian* H);

/**
 * Add term to geometric Hamiltonian
 */
int geometric_hamiltonian_add_term(
    GeometricHamiltonian* H,
    qgt_complex_t coefficient,
    const char* pauli_string,
    size_t* qubit_indices,
    size_t num_qubits
);

// =============================================================================
// Fermion-to-Qubit Mapping
// =============================================================================

/**
 * Convert molecular Hamiltonian to qubit Hamiltonian
 */
int molecular_to_qubit_hamiltonian(
    MolecularHamiltonian* molecular,
    FermionMappingType mapping,
    GeometricHamiltonian** qubit_H
);

/**
 * Apply Jordan-Wigner transformation
 */
int jordan_wigner_transform(
    MolecularHamiltonian* molecular,
    GeometricHamiltonian** qubit_H
);

/**
 * Apply Bravyi-Kitaev transformation
 */
int bravyi_kitaev_transform(
    MolecularHamiltonian* molecular,
    GeometricHamiltonian** qubit_H
);

/**
 * Apply parity mapping
 */
int parity_mapping_transform(
    MolecularHamiltonian* molecular,
    GeometricHamiltonian** qubit_H
);

// =============================================================================
// Hamiltonian Operations
// =============================================================================

/**
 * Compute expectation value
 */
int geometric_hamiltonian_expectation(
    GeometricHamiltonian* H,
    struct quantum_state* state,
    qgt_complex_t* expectation_out
);

/**
 * Compute variance
 */
int geometric_hamiltonian_variance(
    GeometricHamiltonian* H,
    struct quantum_state* state,
    double* variance_out
);

/**
 * Apply Hamiltonian to state
 */
int geometric_hamiltonian_apply(
    GeometricHamiltonian* H,
    struct quantum_state* in,
    struct quantum_state** out
);

/**
 * Compute ground state via exact diagonalization
 */
int geometric_hamiltonian_ground_state(
    GeometricHamiltonian* H,
    struct quantum_state** ground_state,
    double* energy_out
);

/**
 * Compute low-lying spectrum
 */
int geometric_hamiltonian_spectrum(
    GeometricHamiltonian* H,
    size_t num_states,
    double** energies_out,
    struct quantum_state*** states_out
);

// =============================================================================
// Geometric Operations
// =============================================================================

/**
 * Compute Berry phase for parameter loop
 */
int compute_berry_phase(
    GeometricHamiltonian* H,
    double** parameter_path,
    size_t num_points,
    size_t num_parameters,
    size_t band_index,
    double* berry_phase_out
);

/**
 * Compute Berry curvature at parameter point
 */
int compute_berry_curvature(
    GeometricHamiltonian* H,
    double* parameter_point,
    size_t num_parameters,
    size_t band_index,
    double* curvature_out
);

/**
 * Compute quantum metric tensor
 */
int compute_quantum_metric(
    GeometricHamiltonian* H,
    double* parameter_point,
    size_t num_parameters,
    double** metric_out
);

/**
 * Compute quantum geometric tensor
 */
int compute_quantum_geometric_tensor(
    GeometricHamiltonian* H,
    double* parameter_point,
    size_t num_parameters,
    qgt_complex_t** qgt_out
);

/**
 * Add geometric structure to Hamiltonian
 */
int geometric_hamiltonian_set_metric(
    GeometricHamiltonian* H,
    struct geometric_tensor* metric
);

/**
 * Compute Hamiltonian on curved manifold
 */
int hamiltonian_on_manifold(
    GeometricHamiltonian* flat_H,
    struct geometric_tensor* metric,
    GeometricHamiltonian** curved_H
);

// =============================================================================
// Symmetry Operations
// =============================================================================

/**
 * Add symmetry to Hamiltonian
 */
int geometric_hamiltonian_add_symmetry(
    GeometricHamiltonian* H,
    SymmetryType symmetry,
    int quantum_number
);

/**
 * Reduce Hamiltonian using symmetries
 */
int geometric_hamiltonian_symmetry_reduce(
    GeometricHamiltonian* H,
    GeometricHamiltonian** reduced_H
);

/**
 * Project onto symmetry sector
 */
int geometric_hamiltonian_project_sector(
    GeometricHamiltonian* H,
    int* quantum_numbers,
    GeometricHamiltonian** projected_H
);

/**
 * Check if state satisfies symmetry
 */
int check_symmetry_eigenvalue(
    GeometricHamiltonian* H,
    struct quantum_state* state,
    SymmetryType symmetry,
    int* eigenvalue_out
);

// =============================================================================
// Optimization and Grouping
// =============================================================================

/**
 * Group commuting terms for simultaneous measurement
 */
int geometric_hamiltonian_group_commuting(
    GeometricHamiltonian* H,
    GeometricTerm*** groups,
    size_t** group_sizes,
    size_t* num_groups
);

/**
 * Sort terms by magnitude
 */
int geometric_hamiltonian_sort_terms(
    GeometricHamiltonian* H,
    bool descending
);

/**
 * Truncate small terms
 */
int geometric_hamiltonian_truncate(
    GeometricHamiltonian* H,
    double threshold,
    size_t* num_removed
);

/**
 * Combine duplicate terms
 */
int geometric_hamiltonian_simplify(
    GeometricHamiltonian* H
);

// =============================================================================
// Analysis Functions
// =============================================================================

/**
 * Compute Hamiltonian norm
 */
int geometric_hamiltonian_norm(
    GeometricHamiltonian* H,
    double* norm_out
);

/**
 * Estimate spectral gap
 */
int geometric_hamiltonian_estimate_gap(
    GeometricHamiltonian* H,
    double* gap_out
);

/**
 * Compute locality (max qubit weight)
 */
int geometric_hamiltonian_locality(
    GeometricHamiltonian* H,
    size_t* locality_out
);

/**
 * Compute number of measurements needed
 */
int geometric_hamiltonian_measurement_cost(
    GeometricHamiltonian* H,
    double precision,
    size_t* num_measurements_out
);

// =============================================================================
// I/O Functions
// =============================================================================

/**
 * Print Hamiltonian terms
 */
void geometric_hamiltonian_print(GeometricHamiltonian* H);

/**
 * Export to OpenFermion format
 */
int geometric_hamiltonian_to_openfermion(
    GeometricHamiltonian* H,
    char** json_out
);

/**
 * Import from OpenFermion format
 */
int geometric_hamiltonian_from_openfermion(
    const char* json,
    GeometricHamiltonian** H
);

/**
 * Export integrals to FCIDUMP format
 */
int molecular_hamiltonian_to_fcidump(
    MolecularHamiltonian* H,
    const char* filename
);

/**
 * Import from FCIDUMP format
 */
int molecular_hamiltonian_from_fcidump(
    const char* filename,
    MolecularHamiltonian** H
);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_GEOMETRIC_HAMILTONIAN_H
