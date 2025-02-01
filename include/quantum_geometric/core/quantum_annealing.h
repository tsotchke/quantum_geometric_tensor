/**
 * @file quantum_annealing.h
 * @brief Quantum annealing operations and types for quantum geometric learning
 */

#ifndef QUANTUM_GEOMETRIC_QUANTUM_ANNEALING_H
#define QUANTUM_GEOMETRIC_QUANTUM_ANNEALING_H

#include "quantum_geometric/core/error_codes.h"
#include "quantum_geometric/core/quantum_types.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Configuration for quantum annealing process
 */
typedef struct {
    double initial_temperature;     // Starting temperature for annealing
    double final_temperature;       // Target temperature to reach
    double cooling_rate;           // Rate at which temperature decreases
    size_t num_sweeps;            // Number of Monte Carlo sweeps per temperature
    double acceptance_threshold;   // Threshold for accepting worse solutions
    bool use_quantum_fluctuations; // Whether to include quantum fluctuations
    double tunneling_strength;     // Strength of quantum tunneling
    size_t max_iterations;        // Maximum number of iterations
} quantum_annealing_config_t;

/**
 * @brief State of the quantum annealing process
 */
typedef struct {
    double current_temperature;    // Current temperature in annealing schedule
    double current_energy;        // Current energy of the system
    double best_energy;          // Best energy found so far
    size_t iteration_count;      // Number of iterations performed
    bool converged;              // Whether annealing has converged
    void* state_data;           // Pointer to implementation-specific state data
    size_t state_size;          // Size of state data in bytes
} quantum_annealing_state_t;

/**
 * @brief Initialize quantum annealing with given configuration
 * @param config Annealing configuration parameters
 * @param state Pointer to state structure to initialize
 * @return Error code indicating success/failure
 */
qgt_error_t quantum_annealing_init(const quantum_annealing_config_t* config,
                                  quantum_annealing_state_t** state);

/**
 * @brief Perform one step of quantum annealing
 * @param state Current annealing state
 * @return Error code indicating success/failure
 */
qgt_error_t quantum_annealing_step(quantum_annealing_state_t* state);

/**
 * @brief Check if annealing has converged
 * @param state Current annealing state
 * @param converged Pointer to store convergence result
 * @return Error code indicating success/failure
 */
qgt_error_t quantum_annealing_check_convergence(const quantum_annealing_state_t* state,
                                               bool* converged);

/**
 * @brief Get best solution found through annealing
 * @param state Current annealing state
 * @param solution Pointer to store solution data
 * @param solution_size Size of solution data in bytes
 * @return Error code indicating success/failure
 */
qgt_error_t quantum_annealing_get_solution(const quantum_annealing_state_t* state,
                                          void* solution,
                                          size_t solution_size);

/**
 * @brief Clean up quantum annealing state
 * @param state State to clean up
 */
void quantum_annealing_cleanup(quantum_annealing_state_t* state);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_GEOMETRIC_QUANTUM_ANNEALING_H
