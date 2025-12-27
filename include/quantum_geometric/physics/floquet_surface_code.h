/**
 * @file floquet_surface_code.h
 * @brief Floquet surface code for time-dependent quantum error correction
 *
 * Implements Floquet codes which use periodic time-dependent measurements
 * to achieve topological protection. This enables more efficient error
 * correction using time as an additional dimension.
 */

#ifndef FLOQUET_SURFACE_CODE_H
#define FLOQUET_SURFACE_CODE_H

#include <stdbool.h>
#include <stddef.h>
#include "quantum_geometric/physics/stabilizer_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================

#define MAX_FLOQUET_SIZE 100   // Maximum lattice dimension
#define MAX_TIME_STEPS 1000    // Maximum number of time steps in period

// ============================================================================
// Configuration Structure
// ============================================================================

/**
 * Configuration for Floquet surface code
 */
typedef struct FloquetConfig {
    // Lattice parameters
    size_t distance;               // Code distance (must be odd >= 3)
    size_t width;                  // Lattice width
    size_t height;                 // Lattice height

    // Time parameters
    size_t time_steps;             // Number of time steps per period
    double period;                 // Period duration

    // Coupling parameters
    double coupling_strength;      // Base coupling strength [0, 1]
    double* time_dependent_couplings; // Array of coupling modulations per time step

    // Options
    bool use_boundary_stabilizers; // Enable boundary stabilizers
} FloquetConfig;

// ============================================================================
// Initialization and Cleanup
// ============================================================================

/**
 * Initialize Floquet lattice with given configuration
 * @param config Configuration parameters
 * @return true on success, false on failure
 */
bool init_floquet_lattice(const FloquetConfig* config);

/**
 * Clean up Floquet lattice resources
 */
void cleanup_floquet_lattice(void);

// ============================================================================
// Configuration Validation
// ============================================================================

/**
 * Validate Floquet configuration parameters
 * @param config Configuration to validate
 * @return true if configuration is valid
 */
bool validate_floquet_config(const FloquetConfig* config);

/**
 * Calculate lattice dimensions for given code distance
 * @param distance Code distance
 * @param width Output width
 * @param height Output height
 */
void calculate_floquet_dimensions(size_t distance,
                                 size_t* width,
                                 size_t* height);

// ============================================================================
// Stabilizer Information
// ============================================================================

/**
 * Get coordinates of a stabilizer at a given time step
 * @param stabilizer_idx Stabilizer index
 * @param time_step Time step within period
 * @param x Output X coordinate
 * @param y Output Y coordinate
 * @return true on success
 */
bool get_floquet_coordinates(size_t stabilizer_idx,
                            size_t time_step,
                            double* x,
                            double* y);

/**
 * Get qubits involved in a stabilizer measurement
 * @param stabilizer_idx Stabilizer index
 * @param time_step Time step within period
 * @param qubits Output array for qubit indices
 * @param max_qubits Maximum number of qubits to return
 * @return Number of qubits returned
 */
size_t get_floquet_qubits(size_t stabilizer_idx,
                         size_t time_step,
                         size_t* qubits,
                         size_t max_qubits);

/**
 * Check if stabilizer is on the boundary
 * @param stabilizer_idx Stabilizer index
 * @param time_step Time step within period
 * @return true if boundary stabilizer
 */
bool is_floquet_boundary_stabilizer(size_t stabilizer_idx,
                                   size_t time_step);

/**
 * Get the type of a stabilizer (X or Z) at given time
 * @param stabilizer_idx Stabilizer index
 * @param time_step Time step within period
 * @return Stabilizer type
 */
StabilizerType get_floquet_stabilizer_type(size_t stabilizer_idx,
                                          size_t time_step);

/**
 * Get neighboring stabilizers
 * @param stabilizer_idx Stabilizer index
 * @param time_step Time step within period
 * @param neighbors Output array for neighbor indices
 * @param max_neighbors Maximum neighbors to return
 * @return Number of neighbors found
 */
size_t get_floquet_neighbors(size_t stabilizer_idx,
                            size_t time_step,
                            size_t* neighbors,
                            size_t max_neighbors);

// ============================================================================
// Time-Dependent Couplings
// ============================================================================

/**
 * Get coupling strength between two qubits at a time step
 * @param qubit1_idx First qubit index
 * @param qubit2_idx Second qubit index
 * @param time_step Time step within period
 * @return Coupling strength
 */
double get_floquet_coupling_strength(size_t qubit1_idx,
                                    size_t qubit2_idx,
                                    size_t time_step);

/**
 * Get time evolution operator between two time steps
 * @param time_step1 First time step
 * @param time_step2 Second time step
 * @param operator Output array for operator matrix
 * @param max_size Maximum size of output array
 * @return true on success
 */
bool get_floquet_evolution_operator(size_t time_step1,
                                   size_t time_step2,
                                   double* operator,
                                   size_t max_size);

// ============================================================================
// Internal Setup Functions (declared for consistency checking)
// ============================================================================

/**
 * Setup stabilizer configuration
 */
void setup_floquet_stabilizers(void);

/**
 * Setup boundary stabilizers
 */
void setup_floquet_boundaries(void);

/**
 * Calculate stabilizer weights
 */
void calculate_floquet_weights(void);

/**
 * Update coupling matrices for a time step
 * @param time_step Time step to update
 */
void update_floquet_couplings(size_t time_step);

/**
 * Check consistency of Floquet lattice setup
 * @return true if consistent
 */
bool check_floquet_consistency(void);

#ifdef __cplusplus
}
#endif

#endif // FLOQUET_SURFACE_CODE_H
