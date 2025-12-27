/**
 * @file basic_topological_protection.h
 * @brief Basic topological error protection operations
 *
 * Provides fundamental operations for topological quantum error correction
 * including syndrome measurement, anyon detection, and basic correction.
 */

#ifndef BASIC_TOPOLOGICAL_PROTECTION_H
#define BASIC_TOPOLOGICAL_PROTECTION_H

#include <stddef.h>
#include <stdbool.h>
#include "quantum_geometric/core/quantum_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Backwards compatibility aliases
typedef TopologicalErrorCode ErrorCode;
#define NO_ERROR TOPO_NO_ERROR
#define ERROR_DETECTED TOPO_ERROR_DETECTED
#define ERROR_INVALID_STATE TOPO_ERROR_INVALID_STATE

// AnyonPair alias for basic topological operations
typedef TopologicalAnyonPair AnyonPair;

// ============================================================================
// Stabilizer Measurement Operations
// ============================================================================

/**
 * Measure a plaquette (Z-type) stabilizer operator
 * @param state Quantum state to measure
 * @param index Plaquette index
 * @return Measurement result (+1 or -1), 0.0 on error
 */
double measure_plaquette_operator(quantum_state_t* state, size_t index);

/**
 * Measure a vertex (X-type) stabilizer operator
 * @param state Quantum state to measure
 * @param index Vertex index
 * @return Measurement result (+1 or -1), 0.0 on error
 */
double measure_vertex_operator(quantum_state_t* state, size_t index);

/**
 * Measure a stabilizer operator by index
 * @param state Quantum state to measure
 * @param index Stabilizer index
 * @return Measurement result (+1 or -1), 0.0 on error
 */
double measure_stabilizer_operator(quantum_state_t* state, size_t index);

// ============================================================================
// Pauli Measurement Operations
// ============================================================================

/**
 * Measure Pauli-Z operator on a qubit
 * @param state Quantum state
 * @param qubit Qubit index
 * @return Measurement result
 */
double measure_pauli_z(quantum_state_t* state, size_t qubit);

/**
 * Measure Pauli-X operator on a qubit
 * @param state Quantum state
 * @param qubit Qubit index
 * @return Measurement result
 */
double measure_pauli_x(quantum_state_t* state, size_t qubit);

// ============================================================================
// Lattice Structure Operations
// ============================================================================

/**
 * Get vertices of a plaquette
 * @param state Quantum state with lattice structure
 * @param plaquette_index Plaquette index
 * @param vertices Output array of 4 vertex indices
 */
void get_plaquette_vertices(quantum_state_t* state, size_t plaquette_index, size_t* vertices);

/**
 * Get edges adjacent to a vertex
 * @param state Quantum state with lattice structure
 * @param vertex_index Vertex index
 * @param edges Output array of edge indices
 */
void get_vertex_edges(quantum_state_t* state, size_t vertex_index, size_t* edges);

/**
 * Get stabilizer type by index
 * @param state Quantum state
 * @param index Stabilizer index
 * @return Stabilizer type
 */
StabilizerType get_stabilizer_type(quantum_state_t* state, size_t index);

/**
 * Get stabilizer location by index
 * @param state Quantum state
 * @param index Stabilizer index
 * @return Location index (plaquette or vertex index depending on type)
 */
size_t get_stabilizer_location(quantum_state_t* state, size_t index);

// ============================================================================
// Error Detection and Correction
// ============================================================================

/**
 * Detect basic errors using syndrome measurements
 * @param state Quantum state to check
 * @return Error code indicating detection result
 */
ErrorCode detect_basic_errors(quantum_state_t* state);

/**
 * Apply basic error correction
 * @param state Quantum state to correct
 */
void correct_basic_errors(quantum_state_t* state);

/**
 * Verify quantum state after correction
 * @param state Quantum state to verify
 * @return true if state is valid
 */
bool verify_basic_state(quantum_state_t* state);

/**
 * Full topological protection cycle: detect, correct, verify
 * @param state Quantum state to protect
 */
void protect_basic_state(quantum_state_t* state);

// ============================================================================
// Path Finding and Anyon Operations
// ============================================================================

/**
 * Calculate distance between two positions
 * @param pos1 First position
 * @param pos2 Second position
 * @return Euclidean distance
 */
double calculate_anyon_distance(Position pos1, Position pos2);

/**
 * Find shortest path between two positions (Manhattan distance)
 * @param start Starting position
 * @param end Ending position
 * @return Allocated path structure (caller must free)
 */
Path* find_shortest_path(Position start, Position end);

/**
 * Free a path structure
 * @param path Path to free
 */
void free_path(Path* path);

/**
 * Apply correction operator at a position
 * @param state Quantum state
 * @param position Position to apply correction
 */
void apply_correction_operator(quantum_state_t* state, Position position);

/**
 * Log a protection failure for debugging
 * @param state Quantum state that failed protection
 */
void log_protection_failure(quantum_state_t* state);

#ifdef __cplusplus
}
#endif

#endif // BASIC_TOPOLOGICAL_PROTECTION_H
