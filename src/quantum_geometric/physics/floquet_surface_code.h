/**
 * @file floquet_surface_code.h
 * @brief Floquet surface code implementation for time-dependent quantum error correction
 */

#ifndef QUANTUM_GEOMETRIC_FLOQUET_SURFACE_CODE_H
#define QUANTUM_GEOMETRIC_FLOQUET_SURFACE_CODE_H

#include "quantum_geometric/physics/surface_code.h"
#include <stdbool.h>

// Floquet lattice parameters
#define MAX_FLOQUET_SIZE (MAX_SURFACE_SIZE / 2)
#define MAX_TIME_STEPS 100

// Floquet lattice configuration
typedef struct {
    size_t distance;                         // Code distance
    size_t width;                           // Lattice width
    size_t height;                          // Lattice height
    size_t time_steps;                      // Number of time steps
    double period;                          // Driving period
    bool use_boundary_stabilizers;          // Use boundary stabilizers
    double coupling_strength;               // Base qubit coupling strength
    double* time_dependent_couplings;       // Time-dependent coupling strengths
} FloquetConfig;

/**
 * @brief Initialize Floquet surface code lattice
 * @param config Lattice configuration
 * @return Success status
 */
bool init_floquet_lattice(const FloquetConfig* config);

/**
 * @brief Clean up Floquet lattice
 */
void cleanup_floquet_lattice(void);

/**
 * @brief Get stabilizer coordinates in Floquet lattice at specific time
 * @param stabilizer_idx Stabilizer index
 * @param time_step Time step index
 * @param x Output x coordinate
 * @param y Output y coordinate
 * @return Success status
 */
bool get_floquet_coordinates(size_t stabilizer_idx,
                           size_t time_step,
                           double* x,
                           double* y);

/**
 * @brief Get qubit indices for Floquet stabilizer at specific time
 * @param stabilizer_idx Stabilizer index
 * @param time_step Time step index
 * @param qubits Output qubit array
 * @param max_qubits Maximum number of qubits
 * @return Number of qubits
 */
size_t get_floquet_qubits(size_t stabilizer_idx,
                         size_t time_step,
                         size_t* qubits,
                         size_t max_qubits);

/**
 * @brief Check if stabilizer is on boundary at specific time
 * @param stabilizer_idx Stabilizer index
 * @param time_step Time step index
 * @return Boundary status
 */
bool is_floquet_boundary_stabilizer(size_t stabilizer_idx,
                                  size_t time_step);

/**
 * @brief Get stabilizer type in Floquet lattice at specific time
 * @param stabilizer_idx Stabilizer index
 * @param time_step Time step index
 * @return Stabilizer type
 */
StabilizerType get_floquet_stabilizer_type(size_t stabilizer_idx,
                                         size_t time_step);

/**
 * @brief Get neighboring stabilizers in Floquet lattice at specific time
 * @param stabilizer_idx Stabilizer index
 * @param time_step Time step index
 * @param neighbors Output neighbor array
 * @param max_neighbors Maximum number of neighbors
 * @return Number of neighbors
 */
size_t get_floquet_neighbors(size_t stabilizer_idx,
                           size_t time_step,
                           size_t* neighbors,
                           size_t max_neighbors);

/**
 * @brief Calculate Floquet lattice dimensions
 * @param distance Code distance
 * @param width Output width
 * @param height Output height
 */
void calculate_floquet_dimensions(size_t distance,
                                size_t* width,
                                size_t* height);

/**
 * @brief Validate Floquet lattice configuration
 * @param config Configuration to validate
 * @return Validation status
 */
bool validate_floquet_config(const FloquetConfig* config);

/**
 * @brief Get coupling strength between qubits at specific time
 * @param qubit1_idx First qubit index
 * @param qubit2_idx Second qubit index
 * @param time_step Time step index
 * @return Coupling strength
 */
double get_floquet_coupling_strength(size_t qubit1_idx,
                                   size_t qubit2_idx,
                                   size_t time_step);

/**
 * @brief Get time evolution operator between time steps
 * @param time_step1 First time step
 * @param time_step2 Second time step
 * @param operator Output evolution operator matrix
 * @param max_size Maximum matrix size
 * @return Success status
 */
bool get_floquet_evolution_operator(size_t time_step1,
                                  size_t time_step2,
                                  double* operator,
                                  size_t max_size);

// Helper functions
void setup_floquet_stabilizers(void);
void setup_floquet_boundaries(void);
void calculate_floquet_weights(void);
bool check_floquet_consistency(void);
void update_floquet_couplings(size_t time_step);

#endif // QUANTUM_GEOMETRIC_FLOQUET_SURFACE_CODE_H
