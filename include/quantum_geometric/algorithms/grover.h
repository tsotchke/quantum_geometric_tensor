/**
 * @file grover.h
 * @brief Grover's Quantum Search Algorithm implementation
 *
 * Grover's algorithm provides quadratic speedup for unstructured search.
 * Given an oracle that marks target states, it finds a marked state
 * in O(sqrt(N)) queries instead of O(N) classically.
 */

#ifndef GROVER_H
#define GROVER_H

#include <stddef.h>
#include <stdbool.h>
#include "quantum_geometric/core/quantum_types.h"
#include "quantum_geometric/core/quantum_state_types.h"
#include "quantum_geometric/core/error_codes.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Oracle Types
// ============================================================================

/**
 * @brief Oracle function type
 * Returns true if the given basis state index is a target (marked) state
 */
typedef bool (*grover_oracle_func_t)(size_t state_index, void* user_data);

/**
 * @brief Oracle types
 */
typedef enum {
    GROVER_ORACLE_FUNCTION,     // User-provided function
    GROVER_ORACLE_BITSTRING,    // Single target bitstring
    GROVER_ORACLE_SET,          // Set of target states
    GROVER_ORACLE_SAT           // SAT formula (CNF)
} grover_oracle_type_t;

/**
 * @brief Oracle specification
 */
typedef struct {
    grover_oracle_type_t type;

    // For FUNCTION type
    grover_oracle_func_t func;
    void* user_data;

    // For BITSTRING type
    size_t target_state;

    // For SET type
    size_t* target_states;
    size_t num_targets;

    // For SAT type (CNF formula)
    int** clauses;          // Each clause is array of literals (positive/negative var indices)
    size_t* clause_sizes;   // Size of each clause
    size_t num_clauses;
    size_t num_variables;
} grover_oracle_t;

// ============================================================================
// Grover Configuration and State
// ============================================================================

/**
 * @brief Grover algorithm configuration
 */
typedef struct {
    size_t num_qubits;          // Search space size = 2^num_qubits
    size_t num_iterations;      // Number of Grover iterations (0 = auto-compute optimal)
    size_t num_targets;         // Number of target states (for iteration count, 0 = unknown)
    bool use_exact_count;       // Use exact iteration count vs estimated
    bool use_gpu;               // Use GPU acceleration
    void* backend;              // Hardware backend (NULL for simulator)
} grover_config_t;

/**
 * @brief Grover algorithm state
 */
typedef struct grover_state {
    // Configuration
    grover_config_t config;
    size_t num_qubits;

    // Oracle
    grover_oracle_t* oracle;

    // Quantum state
    QuantumState* qstate;

    // Execution state
    size_t current_iteration;
    size_t optimal_iterations;

    // Results
    size_t* measured_states;    // History of measured states
    size_t num_measurements;
    double success_probability; // Estimated success probability
} grover_state_t;

/**
 * @brief Result of Grover search
 */
typedef struct {
    size_t found_state;         // The state found by measurement
    bool is_target;             // Whether found_state is actually a target
    size_t num_iterations;      // Number of iterations performed
    double final_probability;   // Probability of measuring target state
    double execution_time;      // Execution time in seconds
    size_t num_oracle_calls;    // Number of oracle queries
} grover_result_t;

// ============================================================================
// Oracle Construction
// ============================================================================

/**
 * @brief Create oracle from function
 */
grover_oracle_t* grover_create_function_oracle(grover_oracle_func_t func, void* user_data);

/**
 * @brief Create oracle for single target state
 */
grover_oracle_t* grover_create_bitstring_oracle(size_t target_state);

/**
 * @brief Create oracle for multiple target states
 */
grover_oracle_t* grover_create_set_oracle(const size_t* targets, size_t num_targets);

/**
 * @brief Create oracle from SAT formula (CNF)
 * @param clauses Array of clauses, each clause is array of literals
 * @param clause_sizes Size of each clause
 * @param num_clauses Number of clauses
 * @param num_variables Number of variables
 */
grover_oracle_t* grover_create_sat_oracle(int** clauses, size_t* clause_sizes,
                                           size_t num_clauses, size_t num_variables);

/**
 * @brief Destroy oracle
 */
void grover_destroy_oracle(grover_oracle_t* oracle);

/**
 * @brief Check if a state is marked by the oracle
 */
bool grover_oracle_check(const grover_oracle_t* oracle, size_t state);

// ============================================================================
// Grover Algorithm Functions
// ============================================================================

/**
 * @brief Create default configuration
 */
grover_config_t grover_default_config(size_t num_qubits);

/**
 * @brief Initialize Grover search
 */
grover_state_t* grover_init(const grover_oracle_t* oracle, const grover_config_t* config);

/**
 * @brief Prepare initial superposition state |s> = H^n|0>
 */
qgt_error_t grover_prepare_superposition(grover_state_t* state);

/**
 * @brief Apply oracle (marks target states with phase flip)
 */
qgt_error_t grover_apply_oracle(grover_state_t* state);

/**
 * @brief Apply diffusion operator (2|s><s| - I)
 */
qgt_error_t grover_apply_diffusion(grover_state_t* state);

/**
 * @brief Apply one Grover iteration (oracle + diffusion)
 */
qgt_error_t grover_apply_iteration(grover_state_t* state);

/**
 * @brief Compute optimal number of iterations
 * @param N Search space size
 * @param M Number of target states
 */
size_t grover_optimal_iterations(size_t N, size_t M);

/**
 * @brief Run full Grover search
 */
grover_result_t* grover_search(grover_state_t* state);

/**
 * @brief Measure the quantum state
 */
qgt_error_t grover_measure(grover_state_t* state, size_t* result);

/**
 * @brief Get probability of measuring target state
 */
qgt_error_t grover_success_probability(grover_state_t* state, double* prob);

/**
 * @brief Clean up Grover state
 */
void grover_destroy(grover_state_t* state);

/**
 * @brief Clean up Grover result
 */
void grover_destroy_result(grover_result_t* result);

// ============================================================================
// Amplitude Amplification (Generalization)
// ============================================================================

/**
 * @brief Amplitude amplification with custom initial state preparation
 * @param prepare_func Function to prepare initial state
 * @param oracle Oracle marking good states
 * @param num_iterations Number of iterations
 */
grover_result_t* amplitude_amplification(
    qgt_error_t (*prepare_func)(QuantumState*),
    const grover_oracle_t* oracle,
    size_t num_qubits,
    size_t num_iterations
);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Print Grover state summary
 */
void grover_print_state(const grover_state_t* state);

/**
 * @brief Print Grover result summary
 */
void grover_print_result(const grover_result_t* result);

#ifdef __cplusplus
}
#endif

#endif // GROVER_H
