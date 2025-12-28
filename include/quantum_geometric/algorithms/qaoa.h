/**
 * @file qaoa.h
 * @brief Quantum Approximate Optimization Algorithm (QAOA) implementation
 *
 * QAOA is a hybrid quantum-classical algorithm for solving combinatorial
 * optimization problems. This implementation supports:
 * - MaxCut problems on arbitrary graphs
 * - General QUBO (Quadratic Unconstrained Binary Optimization)
 * - Custom cost Hamiltonians
 * - Multiple mixer Hamiltonians (X-mixer, XY-mixer, Grover-mixer)
 */

#ifndef QAOA_H
#define QAOA_H

#include <stddef.h>
#include <stdbool.h>
#include "quantum_geometric/core/quantum_types.h"
#include "quantum_geometric/core/quantum_state.h"
#include "quantum_geometric/core/quantum_state_types.h"
#include "quantum_geometric/core/error_codes.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Problem Definition Types
// ============================================================================

/**
 * @brief Edge in an optimization problem graph
 */
typedef struct {
    size_t i;           // First vertex
    size_t j;           // Second vertex
    double weight;      // Edge weight
} qaoa_edge_t;

/**
 * @brief Graph representation for QAOA problems
 */
typedef struct {
    size_t num_vertices;        // Number of vertices (qubits)
    size_t num_edges;           // Number of edges
    qaoa_edge_t* edges;         // Array of edges
    double* vertex_weights;     // Optional vertex weights (linear terms)
} qaoa_graph_t;

/**
 * @brief Types of QAOA problems
 */
typedef enum {
    QAOA_PROBLEM_MAXCUT,        // Maximum cut problem
    QAOA_PROBLEM_QUBO,          // Quadratic Unconstrained Binary Optimization
    QAOA_PROBLEM_MAX_INDEPENDENT_SET,
    QAOA_PROBLEM_VERTEX_COVER,
    QAOA_PROBLEM_GRAPH_COLORING,
    QAOA_PROBLEM_TSP,           // Traveling Salesman (needs penalty encoding)
    QAOA_PROBLEM_CUSTOM         // User-defined Hamiltonian
} qaoa_problem_type_t;

/**
 * @brief Types of mixer Hamiltonians
 */
typedef enum {
    QAOA_MIXER_X,               // Standard X-mixer (sum of Pauli X)
    QAOA_MIXER_XY,              // XY-mixer for constrained problems
    QAOA_MIXER_GROVER,          // Grover-style mixer
    QAOA_MIXER_CUSTOM           // User-defined mixer
} qaoa_mixer_type_t;

/**
 * @brief Classical optimizer types for QAOA
 */
typedef enum {
    QAOA_OPTIMIZER_COBYLA,      // Constrained optimization
    QAOA_OPTIMIZER_NELDER_MEAD, // Simplex method
    QAOA_OPTIMIZER_POWELL,      // Powell's method
    QAOA_OPTIMIZER_BFGS,        // Quasi-Newton
    QAOA_OPTIMIZER_SPSA,        // Stochastic gradient
    QAOA_OPTIMIZER_ADAM,        // Adaptive moment estimation
    QAOA_OPTIMIZER_GRADIENT_DESCENT
} qaoa_optimizer_type_t;

// ============================================================================
// QAOA Configuration and State
// ============================================================================

/**
 * @brief QAOA algorithm configuration
 */
typedef struct {
    size_t p;                       // Number of QAOA layers
    qaoa_problem_type_t problem_type;
    qaoa_mixer_type_t mixer_type;
    qaoa_optimizer_type_t optimizer_type;

    // Initial parameters (optional - NULL for random init)
    double* initial_gamma;          // Cost layer parameters
    double* initial_beta;           // Mixer layer parameters

    // Optimizer settings
    size_t max_iterations;
    double tolerance;
    double learning_rate;           // For gradient-based optimizers

    // Shots and sampling
    size_t num_shots;               // Number of measurement shots
    bool use_expectation;           // Use exact expectation vs sampling

    // Hardware settings
    bool use_gpu;
    void* backend;                  // Hardware backend (NULL for simulator)
} qaoa_config_t;

/**
 * @brief QAOA algorithm state
 */
typedef struct qaoa_state {
    // Problem definition
    qaoa_graph_t* graph;
    quantum_operator_t* cost_hamiltonian;
    quantum_operator_t* mixer_hamiltonian;

    // Current parameters
    double* gamma;                  // Cost layer angles
    double* beta;                   // Mixer layer angles
    size_t p;                       // Number of layers
    size_t num_qubits;

    // Quantum state (uses QuantumState for amplitudes, not geometric coordinates)
    QuantumState* qstate;

    // Optimization state
    double current_cost;
    double best_cost;
    double* best_gamma;
    double* best_beta;
    size_t iteration;

    // Configuration
    qaoa_config_t config;

    // Results
    int* best_solution;             // Best bitstring found
    double* solution_probabilities; // Probability distribution
} qaoa_state_t;

/**
 * @brief Result of QAOA optimization
 */
typedef struct {
    double optimal_cost;            // Best cost function value found
    int* optimal_solution;          // Optimal bitstring (size = num_qubits)
    double* optimal_gamma;          // Optimal gamma parameters
    double* optimal_beta;           // Optimal beta parameters
    size_t num_iterations;          // Number of optimizer iterations
    double* cost_history;           // Cost function history
    size_t history_length;
    double execution_time;          // Total execution time
    double approximation_ratio;     // Ratio to optimal (if known)
} qaoa_result_t;

// ============================================================================
// Graph Construction
// ============================================================================

/**
 * @brief Create an empty graph
 */
qaoa_graph_t* qaoa_create_graph(size_t num_vertices);

/**
 * @brief Add an edge to the graph
 */
qgt_error_t qaoa_add_edge(qaoa_graph_t* graph, size_t i, size_t j, double weight);

/**
 * @brief Set vertex weight (linear term)
 */
qgt_error_t qaoa_set_vertex_weight(qaoa_graph_t* graph, size_t vertex, double weight);

/**
 * @brief Create a random graph for testing
 */
qaoa_graph_t* qaoa_create_random_graph(size_t num_vertices, double edge_probability,
                                        double min_weight, double max_weight);

/**
 * @brief Create a graph from adjacency matrix
 */
qaoa_graph_t* qaoa_create_from_adjacency(const double* adjacency, size_t n);

/**
 * @brief Destroy a graph and free resources
 */
void qaoa_destroy_graph(qaoa_graph_t* graph);

// ============================================================================
// QAOA Algorithm Functions
// ============================================================================

/**
 * @brief Initialize QAOA with given problem and configuration
 */
qaoa_state_t* qaoa_init(const qaoa_graph_t* graph, const qaoa_config_t* config);

/**
 * @brief Construct cost Hamiltonian from graph
 * For MaxCut: H_C = sum_{(i,j) in E} w_ij * (1 - Z_i Z_j) / 2
 */
qgt_error_t qaoa_construct_cost_hamiltonian(qaoa_state_t* state);

/**
 * @brief Construct mixer Hamiltonian
 * For X-mixer: H_M = sum_i X_i
 */
qgt_error_t qaoa_construct_mixer_hamiltonian(qaoa_state_t* state);

/**
 * @brief Prepare initial state (uniform superposition |+>^n)
 */
qgt_error_t qaoa_prepare_initial_state(qaoa_state_t* state);

/**
 * @brief Apply one QAOA layer (cost evolution + mixer evolution)
 */
qgt_error_t qaoa_apply_layer(qaoa_state_t* state, size_t layer_idx);

/**
 * @brief Apply full QAOA circuit with given parameters
 */
qgt_error_t qaoa_apply_circuit(qaoa_state_t* state,
                               const double* gamma,
                               const double* beta);

/**
 * @brief Compute cost function expectation value
 */
qgt_error_t qaoa_compute_expectation(qaoa_state_t* state, double* expectation);

/**
 * @brief Compute gradient of cost function with respect to parameters
 */
qgt_error_t qaoa_compute_gradient(qaoa_state_t* state,
                                  double* gamma_grad,
                                  double* beta_grad);

/**
 * @brief Run full QAOA optimization
 */
qaoa_result_t* qaoa_optimize(qaoa_state_t* state);

/**
 * @brief Sample solutions from the final QAOA state
 */
qgt_error_t qaoa_sample(qaoa_state_t* state, int** samples, size_t num_samples);

/**
 * @brief Evaluate cost function for a given bitstring
 */
double qaoa_evaluate_solution(const qaoa_graph_t* graph, const int* solution);

/**
 * @brief Clean up QAOA state
 */
void qaoa_destroy(qaoa_state_t* state);

/**
 * @brief Clean up QAOA result
 */
void qaoa_destroy_result(qaoa_result_t* result);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Create default QAOA configuration
 */
qaoa_config_t qaoa_default_config(size_t p);

/**
 * @brief Estimate optimal p value for a given problem size
 */
size_t qaoa_estimate_optimal_p(size_t num_vertices, size_t num_edges);

/**
 * @brief Compute approximation ratio (if optimal is known)
 */
double qaoa_approximation_ratio(const qaoa_graph_t* graph,
                                const int* solution,
                                double optimal_cost);

/**
 * @brief Print QAOA state summary
 */
void qaoa_print_state(const qaoa_state_t* state);

/**
 * @brief Print QAOA result summary
 */
void qaoa_print_result(const qaoa_result_t* result);

#ifdef __cplusplus
}
#endif

#endif // QAOA_H
