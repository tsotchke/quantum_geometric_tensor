/**
 * @file quantum_physics_operations.h
 * @brief Quantum physics simulation operations
 *
 * Implements core quantum physics operations including time evolution,
 * measurement, expectation values, correlation functions, and various
 * physical observables for quantum many-body systems.
 */

#ifndef QUANTUM_PHYSICS_OPERATIONS_H
#define QUANTUM_PHYSICS_OPERATIONS_H

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
struct quantum_circuit;
struct geometric_tensor;

// =============================================================================
// Physics Simulation Types
// =============================================================================

/**
 * Time evolution methods
 */
typedef enum {
    EVOLUTION_EXACT,                 // Exact diagonalization
    EVOLUTION_TROTTER,               // Trotter-Suzuki decomposition
    EVOLUTION_KRYLOV,                // Krylov subspace methods
    EVOLUTION_CHEBYSHEV,             // Chebyshev polynomial expansion
    EVOLUTION_RUNGE_KUTTA,           // Classical Runge-Kutta
    EVOLUTION_LANCZOS,               // Lanczos propagation
    EVOLUTION_TDVP,                  // Time-dependent variational principle
    EVOLUTION_TEBD,                  // Time-evolving block decimation
    EVOLUTION_MPO                    // Matrix product operator
} TimeEvolutionMethod;

/**
 * Hamiltonian types
 */
typedef enum {
    HAMILTONIAN_GENERIC,             // Generic sparse Hamiltonian
    HAMILTONIAN_HEISENBERG,          // Heisenberg spin model
    HAMILTONIAN_ISING,               // Ising model
    HAMILTONIAN_HUBBARD,             // Hubbard model
    HAMILTONIAN_BCS,                 // BCS superconductor
    HAMILTONIAN_XXZ,                 // XXZ spin chain
    HAMILTONIAN_KITAEV,              // Kitaev model
    HAMILTONIAN_TORIC_CODE,          // Toric code
    HAMILTONIAN_MOLECULAR,           // Molecular Hamiltonian
    HAMILTONIAN_CUSTOM               // User-defined
} HamiltonianType;

/**
 * Observable types
 */
typedef enum {
    OBSERVABLE_ENERGY,               // Total energy
    OBSERVABLE_MAGNETIZATION,        // Magnetization
    OBSERVABLE_SPIN,                 // Spin components
    OBSERVABLE_DENSITY,              // Particle density
    OBSERVABLE_CURRENT,              // Current operator
    OBSERVABLE_CORRELATION,          // Two-point correlation
    OBSERVABLE_ENTANGLEMENT,         // Entanglement measures
    OBSERVABLE_FIDELITY,             // State fidelity
    OBSERVABLE_CUSTOM                // Custom observable
} ObservableType;

/**
 * Boundary conditions
 */
typedef enum {
    BOUNDARY_OPEN,                   // Open boundaries
    BOUNDARY_PERIODIC,               // Periodic boundaries
    BOUNDARY_ANTIPERIODIC,           // Anti-periodic
    BOUNDARY_TWISTED,                // Twisted boundaries
    BOUNDARY_REFLECTING              // Reflecting boundaries
} BoundaryCondition;

/**
 * Spin types
 */
typedef enum {
    SPIN_HALF,                       // S = 1/2
    SPIN_ONE,                        // S = 1
    SPIN_THREE_HALF,                 // S = 3/2
    SPIN_TWO,                        // S = 2
    SPIN_CUSTOM                      // Custom spin
} SpinType;

// =============================================================================
// Hamiltonian Structures
// =============================================================================

/**
 * Hamiltonian term (sum of tensor products)
 */
typedef struct {
    qgt_complex_t coefficient;
    size_t* sites;                   // Sites this term acts on
    char** operators;                // Operator names ("X", "Y", "Z", "I", etc.)
    size_t num_sites;
} HamiltonianTerm;

/**
 * Generic Hamiltonian
 */
typedef struct {
    HamiltonianType type;
    HamiltonianTerm* terms;
    size_t num_terms;
    size_t num_sites;
    size_t local_dim;                // Local Hilbert space dimension
    BoundaryCondition boundary;
    qgt_complex_t* matrix;          // Dense matrix (optional)
    size_t matrix_dim;
    bool is_hermitian;
    bool is_sparse;

    // Sparse representation
    qgt_complex_t* sparse_data;
    size_t* sparse_row_ptr;
    size_t* sparse_col_idx;
    size_t sparse_nnz;
} Hamiltonian;

/**
 * Heisenberg model parameters
 */
typedef struct {
    double Jx;                       // XX coupling
    double Jy;                       // YY coupling
    double Jz;                       // ZZ coupling
    double h;                        // External field strength
    double* h_field;                 // Site-dependent field (optional)
    double* J_couplings;             // Bond-dependent couplings (optional)
    SpinType spin;
    BoundaryCondition boundary;
} HeisenbergParams;

/**
 * Hubbard model parameters
 */
typedef struct {
    double t;                        // Hopping amplitude
    double U;                        // On-site interaction
    double mu;                       // Chemical potential
    double* t_matrix;                // Hopping matrix (optional)
    size_t num_sites;
    size_t num_electrons;
    bool spin_polarized;
    BoundaryCondition boundary;
} HubbardParams;

/**
 * Ising model parameters
 */
typedef struct {
    double J;                        // Nearest-neighbor coupling
    double h;                        // Transverse field
    double* J_couplings;             // Bond-dependent J (optional)
    double* h_field;                 // Site-dependent h (optional)
    BoundaryCondition boundary;
    bool transverse;                 // Transverse field Ising model
} IsingParams;

// =============================================================================
// Observable Structures
// =============================================================================

/**
 * Generic observable
 */
typedef struct {
    ObservableType type;
    char* name;
    HamiltonianTerm* operator_terms;
    size_t num_terms;
    qgt_complex_t* matrix;          // Dense matrix (optional)
    size_t matrix_dim;
    bool is_hermitian;
} Observable;

/**
 * Correlation function result
 */
typedef struct {
    qgt_complex_t* values;          // Correlation values
    size_t* sites_i;                 // First site indices
    size_t* sites_j;                 // Second site indices
    size_t num_pairs;
    double* distances;               // Spatial distances (optional)
    bool is_connected;               // Connected correlator
} CorrelationResult;

/**
 * Expectation value result
 */
typedef struct {
    qgt_complex_t value;
    double variance;
    double standard_error;
    size_t num_samples;              // For stochastic methods
    bool is_exact;
} ExpectationResult;

// =============================================================================
// Time Evolution Structures
// =============================================================================

/**
 * Time evolution configuration
 */
typedef struct {
    TimeEvolutionMethod method;
    double dt;                       // Time step
    double total_time;
    size_t num_steps;
    size_t trotter_order;            // For Trotter methods
    size_t krylov_dim;               // For Krylov methods
    size_t chebyshev_order;          // For Chebyshev methods
    double tolerance;
    bool store_intermediate;         // Store states at each step
    bool compute_observables;
} TimeEvolutionConfig;

/**
 * Time evolution result
 */
typedef struct {
    struct quantum_state** states;   // States at each time step
    double* times;
    size_t num_steps;
    qgt_complex_t** observable_values;  // Observables over time
    size_t num_observables;
    double total_time;
    double computation_time_ms;
} TimeEvolutionResult;

/**
 * Trotter decomposition layer
 */
typedef struct {
    struct quantum_circuit** gates;  // Gates for this layer
    size_t num_gates;
    double* times;                   // Time for each gate
} TrotterLayer;

// =============================================================================
// Hamiltonian Operations
// =============================================================================

/**
 * Create Hamiltonian
 */
int hamiltonian_create(Hamiltonian** H, size_t num_sites, size_t local_dim);

/**
 * Destroy Hamiltonian
 */
void hamiltonian_destroy(Hamiltonian* H);

/**
 * Add term to Hamiltonian
 */
int hamiltonian_add_term(Hamiltonian* H,
                         qgt_complex_t coeff,
                         size_t* sites,
                         char** operators,
                         size_t num_sites);

/**
 * Create Heisenberg Hamiltonian
 */
int hamiltonian_heisenberg(Hamiltonian** H,
                           size_t num_sites,
                           HeisenbergParams* params);

/**
 * Create Hubbard Hamiltonian
 */
int hamiltonian_hubbard(Hamiltonian** H,
                        HubbardParams* params);

/**
 * Create Ising Hamiltonian
 */
int hamiltonian_ising(Hamiltonian** H,
                      size_t num_sites,
                      IsingParams* params);

/**
 * Build sparse matrix representation
 */
int hamiltonian_build_sparse(Hamiltonian* H);

/**
 * Build dense matrix representation
 */
int hamiltonian_build_dense(Hamiltonian* H);

/**
 * Apply Hamiltonian to state: |out> = H|in>
 */
int hamiltonian_apply(Hamiltonian* H,
                      struct quantum_state* in,
                      struct quantum_state** out);

/**
 * Compute ground state energy (exact diagonalization)
 */
int hamiltonian_ground_state(Hamiltonian* H,
                             struct quantum_state** ground_state,
                             double* energy);

/**
 * Compute spectrum (eigenvalues and eigenvectors)
 */
int hamiltonian_spectrum(Hamiltonian* H,
                         size_t num_states,
                         double** energies,
                         struct quantum_state*** eigenstates);

// =============================================================================
// Time Evolution Operations
// =============================================================================

/**
 * Time evolve quantum state
 */
int time_evolve(struct quantum_state* initial,
                Hamiltonian* H,
                TimeEvolutionConfig* config,
                TimeEvolutionResult** result);

/**
 * Build Trotter decomposition
 */
int trotter_decomposition(Hamiltonian* H,
                          double dt,
                          size_t order,
                          TrotterLayer** layers,
                          size_t* num_layers);

/**
 * Apply Trotter step
 */
int trotter_step(struct quantum_state* state,
                 TrotterLayer* layers,
                 size_t num_layers);

/**
 * Krylov subspace evolution
 */
int krylov_evolve(struct quantum_state* state,
                  Hamiltonian* H,
                  double dt,
                  size_t krylov_dim,
                  double tolerance);

/**
 * Chebyshev polynomial evolution
 */
int chebyshev_evolve(struct quantum_state* state,
                     Hamiltonian* H,
                     double dt,
                     size_t order);

/**
 * Free time evolution result
 */
void time_evolution_result_free(TimeEvolutionResult* result);

// =============================================================================
// Observable Operations
// =============================================================================

/**
 * Create observable
 */
int observable_create(Observable** obs, const char* name);

/**
 * Destroy observable
 */
void observable_destroy(Observable* obs);

/**
 * Create standard observable
 */
int observable_create_standard(Observable** obs,
                               ObservableType type,
                               size_t* sites,
                               size_t num_sites);

/**
 * Compute expectation value
 */
int expectation_value(struct quantum_state* state,
                      Observable* obs,
                      ExpectationResult** result);

/**
 * Compute variance
 */
int observable_variance(struct quantum_state* state,
                        Observable* obs,
                        double* variance);

/**
 * Compute correlation function
 */
int correlation_function(struct quantum_state* state,
                         Observable* obs_A,
                         Observable* obs_B,
                         CorrelationResult** result);

/**
 * Compute connected correlation
 */
int connected_correlation(struct quantum_state* state,
                          Observable* obs_A,
                          Observable* obs_B,
                          CorrelationResult** result);

/**
 * Free correlation result
 */
void correlation_result_free(CorrelationResult* result);

/**
 * Free expectation result
 */
void expectation_result_free(ExpectationResult* result);

// =============================================================================
// Spin Operations
// =============================================================================

/**
 * Create spin operator
 */
int spin_operator(Observable** obs,
                  char component,          // 'X', 'Y', 'Z', '+', '-'
                  size_t site,
                  SpinType spin);

/**
 * Compute total spin
 */
int total_spin(struct quantum_state* state,
               size_t num_sites,
               SpinType spin,
               double* S_total,
               double* Sz_total);

/**
 * Compute spin-spin correlation
 */
int spin_correlation(struct quantum_state* state,
                     size_t site_i,
                     size_t site_j,
                     SpinType spin,
                     qgt_complex_t* Sxx,
                     qgt_complex_t* Syy,
                     qgt_complex_t* Szz);

/**
 * Compute structure factor S(q)
 */
int structure_factor(struct quantum_state* state,
                     size_t num_sites,
                     double* q_points,
                     size_t num_q,
                     qgt_complex_t** S_q);

// =============================================================================
// Entanglement Operations
// =============================================================================

/**
 * Compute entanglement entropy
 */
int entanglement_entropy(struct quantum_state* state,
                         size_t* subsystem_sites,
                         size_t num_subsystem,
                         double* entropy);

/**
 * Compute Renyi entropy
 */
int renyi_entropy(struct quantum_state* state,
                  size_t* subsystem_sites,
                  size_t num_subsystem,
                  double alpha,
                  double* entropy);

/**
 * Compute entanglement spectrum
 */
int entanglement_spectrum(struct quantum_state* state,
                          size_t* subsystem_sites,
                          size_t num_subsystem,
                          double** spectrum,
                          size_t* spectrum_size);

/**
 * Compute mutual information
 */
int mutual_information(struct quantum_state* state,
                       size_t* subsystem_A,
                       size_t num_A,
                       size_t* subsystem_B,
                       size_t num_B,
                       double* I_AB);

/**
 * Compute reduced density matrix
 */
int reduced_density_matrix(struct quantum_state* state,
                           size_t* subsystem_sites,
                           size_t num_subsystem,
                           qgt_complex_t** rho,
                           size_t* rho_dim);

// =============================================================================
// Fidelity and Distance Measures
// =============================================================================

/**
 * Compute state fidelity
 */
int state_fidelity(struct quantum_state* state1,
                   struct quantum_state* state2,
                   double* fidelity);

/**
 * Compute trace distance
 */
int trace_distance(struct quantum_state* state1,
                   struct quantum_state* state2,
                   double* distance);

/**
 * Compute Bures distance
 */
int bures_distance(struct quantum_state* state1,
                   struct quantum_state* state2,
                   double* distance);

/**
 * Compute Loschmidt echo
 */
int loschmidt_echo(struct quantum_state* initial,
                   Hamiltonian* H1,
                   Hamiltonian* H2,
                   double* times,
                   size_t num_times,
                   double** echo);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Create Pauli operator
 */
int pauli_operator(Observable** obs,
                   char pauli,              // 'I', 'X', 'Y', 'Z'
                   size_t site);

/**
 * Create ladder operators
 */
int ladder_operator(Observable** obs,
                    bool raising,            // true = raising, false = lowering
                    size_t site,
                    size_t local_dim);

/**
 * Create number operator
 */
int number_operator(Observable** obs,
                    size_t site,
                    size_t local_dim);

/**
 * Print Hamiltonian
 */
void hamiltonian_print(Hamiltonian* H);

/**
 * Print observable
 */
void observable_print(Observable* obs);

/**
 * Validate Hamiltonian
 */
bool hamiltonian_validate(Hamiltonian* H);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_PHYSICS_OPERATIONS_H
