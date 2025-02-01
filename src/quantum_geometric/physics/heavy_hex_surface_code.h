/**
 * @file heavy_hex_surface_code.h
 * @brief Heavy hexagonal surface code implementation optimized for IBM quantum hardware
 */

#ifndef QUANTUM_GEOMETRIC_HEAVY_HEX_SURFACE_CODE_H
#define QUANTUM_GEOMETRIC_HEAVY_HEX_SURFACE_CODE_H

#include "quantum_geometric/physics/surface_code.h"
#include <stdbool.h>

// Heavy hex lattice parameters
#define MAX_HEX_SIZE (MAX_SURFACE_SIZE / 2)

// Heavy hex lattice configuration
typedef struct {
    size_t distance;                         // Code distance
    size_t width;                           // Lattice width
    size_t height;                          // Lattice height
    bool use_boundary_stabilizers;          // Use boundary stabilizers
    double coupling_strength;               // Qubit coupling strength
} HeavyHexConfig;

/**
 * @brief Initialize heavy hex surface code lattice
 * @param config Lattice configuration
 * @return Success status
 */
bool init_heavy_hex_lattice(const HeavyHexConfig* config);

/**
 * @brief Clean up heavy hex lattice
 */
void cleanup_heavy_hex_lattice(void);

/**
 * @brief Get stabilizer coordinates in heavy hex lattice
 * @param stabilizer_idx Stabilizer index
 * @param x Output x coordinate
 * @param y Output y coordinate
 * @return Success status
 */
bool get_hex_coordinates(size_t stabilizer_idx,
                        double* x,
                        double* y);

/**
 * @brief Get qubit indices for heavy hex stabilizer
 * @param stabilizer_idx Stabilizer index
 * @param qubits Output qubit array
 * @param max_qubits Maximum number of qubits
 * @return Number of qubits
 */
size_t get_hex_qubits(size_t stabilizer_idx,
                      size_t* qubits,
                      size_t max_qubits);

/**
 * @brief Check if stabilizer is on boundary
 * @param stabilizer_idx Stabilizer index
 * @return Boundary status
 */
bool is_hex_boundary_stabilizer(size_t stabilizer_idx);

/**
 * @brief Get stabilizer type in heavy hex lattice
 * @param stabilizer_idx Stabilizer index
 * @return Stabilizer type
 */
StabilizerType get_hex_stabilizer_type(size_t stabilizer_idx);

/**
 * @brief Get neighboring stabilizers in heavy hex lattice
 * @param stabilizer_idx Stabilizer index
 * @param neighbors Output neighbor array
 * @param max_neighbors Maximum number of neighbors
 * @return Number of neighbors
 */
size_t get_hex_neighbors(size_t stabilizer_idx,
                        size_t* neighbors,
                        size_t max_neighbors);

/**
 * @brief Calculate heavy hex lattice dimensions
 * @param distance Code distance
 * @param width Output width
 * @param height Output height
 */
void calculate_hex_dimensions(size_t distance,
                            size_t* width,
                            size_t* height);

/**
 * @brief Validate heavy hex lattice configuration
 * @param config Configuration to validate
 * @return Validation status
 */
bool validate_hex_config(const HeavyHexConfig* config);

/**
 * @brief Get coupling strength between qubits
 * @param qubit1_idx First qubit index
 * @param qubit2_idx Second qubit index
 * @return Coupling strength
 */
double get_hex_coupling_strength(size_t qubit1_idx,
                               size_t qubit2_idx);

// Helper functions
void setup_hex_stabilizers(void);
void setup_hex_boundaries(void);
void calculate_hex_weights(void);
bool check_hex_consistency(void);

#endif // QUANTUM_GEOMETRIC_HEAVY_HEX_SURFACE_CODE_H
