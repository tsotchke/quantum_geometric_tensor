/**
 * @file stabilizer_types.h
 * @brief Type definitions for quantum stabilizer measurement system
 */

#ifndef QUANTUM_GEOMETRIC_STABILIZER_TYPES_H
#define QUANTUM_GEOMETRIC_STABILIZER_TYPES_H

#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/error_codes.h"
#include "quantum_geometric/hardware/quantum_hardware_types.h"
#include <stdbool.h>
#include <stddef.h>

// Forward declarations
typedef struct ProtectionSystem ProtectionSystem;
typedef struct MatchingGraph MatchingGraph;
typedef struct AnyonSet AnyonSet;
typedef struct CorrectionPattern CorrectionPattern;

// Stabilizer array structure
typedef struct {
    size_t size;              // Number of stabilizers in array
    double* measurements;     // Array of measurement results
} StabilizerArray;

// Stabilizer type enumeration
typedef enum {
    STABILIZER_PLAQUETTE,    // Plaquette stabilizer
    STABILIZER_VERTEX        // Vertex stabilizer
} StabilizerType;

// Stabilizer configuration
typedef struct {
    size_t lattice_width;            // Width of stabilizer lattice
    size_t lattice_height;           // Height of stabilizer lattice
    double error_threshold;          // Error detection threshold
    bool auto_correction;            // Enable automatic error correction
    bool enable_parallel;            // Enable parallel measurements
    size_t max_parallel_ops;        // Maximum parallel operations
    double correlation_threshold;    // Correlation threshold for parallel ops
    size_t repetition_count;        // Number of measurement repetitions
    size_t min_valid_measurements;  // Minimum valid measurements required
    double min_confidence;          // Minimum confidence threshold
    double measurement_error_rate;  // Expected measurement error rate
    double confidence_threshold;    // Confidence threshold for measurements
    bool periodic_boundaries;       // Use periodic boundary conditions
    bool handle_boundaries;        // Special handling for boundaries
} StabilizerConfig;

// Stabilizer state structure
typedef struct {
    StabilizerArray* plaquette_stabilizers;  // Plaquette stabilizer array
    StabilizerArray* vertex_stabilizers;     // Vertex stabilizer array
    StabilizerConfig config;                // Configuration parameters
    size_t measurement_count;               // Number of measurements performed
    double error_rate;                      // Current error rate
    double* last_syndrome;                  // Last syndrome measurement
    double* measurement_confidence;         // Confidence values for measurements
    size_t* repetition_results;            // Results from repeated measurements
    double* error_correlations;            // Error correlation tracking
    bool* measured_in_parallel;            // Parallel measurement tracking
    size_t current_parallel_group;         // Current parallel measurement group
    size_t history_capacity;               // Capacity of measurement history
    size_t history_size;                   // Current size of measurement history
    double** measurement_history;          // History of measurements
} StabilizerState;

// Protection system configuration
typedef struct {
    double detection_threshold;      // Error detection threshold
    double confidence_threshold;     // Confidence threshold
    double weight_scale_factor;      // Weight scaling factor
    bool use_boundary_matching;      // Enable boundary matching
    bool enable_parallel;            // Enable parallel operations
    size_t parallel_group_size;      // Size of parallel groups
    size_t min_pattern_occurrences; // Minimum pattern occurrences
    double pattern_threshold;       // Pattern recognition threshold
    size_t max_matching_iterations; // Maximum matching iterations
} SyndromeConfig;

#endif // QUANTUM_GEOMETRIC_STABILIZER_TYPES_H
