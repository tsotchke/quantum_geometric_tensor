/**
 * @file rotated_surface_code.h
 * @brief Rotated surface code implementation
 */

#ifndef QUANTUM_GEOMETRIC_ROTATED_SURFACE_CODE_H
#define QUANTUM_GEOMETRIC_ROTATED_SURFACE_CODE_H

#include "quantum_geometric/physics/surface_code.h"
#include <stdbool.h>

// Rotated lattice parameters
#define MAX_ROTATED_SIZE (MAX_SURFACE_SIZE / 2)

// Rotated lattice configuration
typedef struct {
    size_t distance;                         // Code distance
    size_t width;                           // Lattice width
    size_t height;                          // Lattice height
    double angle;                           // Rotation angle
    bool use_boundary_stabilizers;          // Use boundary stabilizers
} RotatedLatticeConfig;

/**
 * @brief Initialize rotated surface code lattice
 * @param config Lattice configuration
 * @return Success status
 */
bool init_rotated_lattice(const RotatedLatticeConfig* config);

/**
 * @brief Clean up rotated lattice
 */
void cleanup_rotated_lattice(void);

/**
 * @brief Get stabilizer coordinates in rotated lattice
 * @param stabilizer_idx Stabilizer index
 * @param x Output x coordinate
 * @param y Output y coordinate
 * @return Success status
 */
bool get_rotated_coordinates(size_t stabilizer_idx,
                           double* x,
                           double* y);

/**
 * @brief Get qubit indices for rotated stabilizer
 * @param stabilizer_idx Stabilizer index
 * @param qubits Output qubit array
 * @param max_qubits Maximum number of qubits
 * @return Number of qubits
 */
size_t get_rotated_qubits(size_t stabilizer_idx,
                         size_t* qubits,
                         size_t max_qubits);

/**
 * @brief Check if stabilizer is on boundary
 * @param stabilizer_idx Stabilizer index
 * @return Boundary status
 */
bool is_boundary_stabilizer(size_t stabilizer_idx);

/**
 * @brief Get stabilizer type in rotated lattice
 * @param stabilizer_idx Stabilizer index
 * @return Stabilizer type
 */
StabilizerType get_rotated_stabilizer_type(size_t stabilizer_idx);

/**
 * @brief Get neighboring stabilizers in rotated lattice
 * @param stabilizer_idx Stabilizer index
 * @param neighbors Output neighbor array
 * @param max_neighbors Maximum number of neighbors
 * @return Number of neighbors
 */
size_t get_rotated_neighbors(size_t stabilizer_idx,
                           size_t* neighbors,
                           size_t max_neighbors);

/**
 * @brief Calculate rotated lattice dimensions
 * @param distance Code distance
 * @param width Output width
 * @param height Output height
 */
void calculate_rotated_dimensions(size_t distance,
                                size_t* width,
                                size_t* height);

/**
 * @brief Validate rotated lattice configuration
 * @param config Configuration to validate
 * @return Validation status
 */
bool validate_rotated_config(const RotatedLatticeConfig* config);

// Helper functions
void setup_rotated_stabilizers(void);
void setup_rotated_boundaries(void);
void calculate_rotated_weights(void);
bool check_rotated_consistency(void);

#endif // QUANTUM_GEOMETRIC_ROTATED_SURFACE_CODE_H
