#ifndef SYNDROME_EXTRACTION_H
#define SYNDROME_EXTRACTION_H

#include "quantum_geometric/physics/z_stabilizer_operations.h"
#include "quantum_geometric/physics/error_syndrome.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_types.h"
#include <stdbool.h>
#include <time.h>

// ZStabilizerConfig, ZHardwareConfig, ZStabilizerState are defined in z_stabilizer_operations.h
// quantum_state_t is defined in quantum_types.h
// SyndromeVertex, SyndromeEdge, MatchingGraph, ErrorSyndrome, SyndromeConfig
// are defined in error_syndrome.h

// Alias for backward compatibility with code expecting 'quantum_state'
#ifndef QUANTUM_STATE_ALIAS_DEFINED
#define QUANTUM_STATE_ALIAS_DEFINED
typedef quantum_state_t quantum_state;
#endif

// Measurement result with confidence tracking
typedef struct {
    size_t qubit_index;      // Index of measured qubit
    double measured_value;   // Measurement value
    bool had_error;          // Whether error was detected
    double error_prob;       // Error probability
    double confidence;       // Measurement confidence
    double hardware_factor;  // Hardware reliability factor
} measurement_result;

// Syndrome cache for optimization
typedef struct {
    double* error_rates;       // Error rates for each qubit
    bool* error_history;       // History of errors
    double* correlations;      // Error correlations
    size_t* plaquette_indices; // Indices for plaquette operators
    size_t* vertex_indices;    // Indices for vertex operators
    double* hardware_weights;  // Hardware-specific weights
    double* confidence_history; // History of confidence values

    // Enhanced fields (TDD - required by tests)
    ZStabilizerState* z_state;         // Z-stabilizer state for optimization
    double* temporal_correlations;     // Temporal correlation data
    double* spatial_correlations;      // Spatial correlation matrix
    double* phase_correlations;        // Phase correlation data
} SyndromeCache;

// Syndrome extraction state
typedef struct {
    SyndromeConfig config;       // Configuration
    SyndromeCache* cache;        // Measurement cache
    MatchingGraph* graph;        // Matching graph
    size_t total_syndromes;      // Total syndromes extracted
    double error_rate;           // Current error rate
    double confidence_level;     // Confidence in measurements
    double detection_threshold;  // Current detection threshold
    double confidence_threshold; // Current confidence threshold
    double avg_extraction_time;  // Average extraction time
    double max_extraction_time;  // Maximum extraction time
    time_t last_update_time;     // Time of last update

    // Enhanced state fields (TDD - required by tests)
    double phase_stability;      // Phase stability metric [0,1]
    double hardware_efficiency;  // Hardware efficiency metric [0,1]
    double temporal_stability;   // Temporal stability metric [0,1]
    double spatial_coherence;    // Spatial coherence metric [0,1]
    double cache_hit_rate;       // Cache hit rate [0,1]
    double simd_utilization;     // SIMD utilization [0,1]
    double gpu_utilization;      // GPU utilization [0,1]
    double memory_bandwidth_utilization; // Memory bandwidth utilization [0,1]
    double parallel_efficiency;  // Parallel processing efficiency [0,1]
    size_t num_parallel_groups;  // Number of parallel groups
    size_t parallel_group_size;  // Size of each parallel group
    bool parallel_enabled;       // Whether parallel processing is enabled
} SyndromeState;

// Initialize syndrome extraction
qgt_error_t init_syndrome_extraction(SyndromeState* state,
                                    const SyndromeConfig* config);

// Clean up syndrome extraction
void cleanup_syndrome_extraction(SyndromeState* state);

// Forward declare HardwareProfile if not included
#ifndef HARDWARE_PROFILE_DEFINED
struct HardwareProfile;
#endif

// Extract error syndrome from quantum state
qgt_error_t extract_error_syndrome(SyndromeState* state,
                                  const quantum_state* qstate,
                                  ErrorSyndrome* syndrome,
                                  const struct HardwareProfile* hw_profile);

// Update syndrome metrics
qgt_error_t update_syndrome_metrics(SyndromeState* state);

// Set measurement confidence for a qubit
qgt_error_t set_measurement_confidence(quantum_state* state,
                                      size_t x,
                                      size_t y,
                                      size_t z,
                                      double confidence);

// Predict next error locations
qgt_error_t predict_next_errors(const SyndromeState* state,
                               size_t* predicted_locations,
                               size_t max_predictions,
                               size_t* num_predicted);

// ============================================================================
// Enhanced Syndrome Extraction (TDD - required by tests)
// ============================================================================

/**
 * Initialize enhanced syndrome extraction with Z-stabilizer optimizations
 * @param state Pointer to state structure to initialize
 * @param config Configuration parameters
 * @return true if initialization successful, false otherwise
 */
bool init_syndrome_extraction_enhanced(SyndromeState* state,
                                       const SyndromeConfig* config);

/**
 * Extract error syndrome with enhanced Z-stabilizer optimizations
 * @param state Syndrome extraction state
 * @param qstate Quantum state to analyze
 * @param syndrome Output error syndrome
 * @return true if extraction successful, false otherwise
 */
bool extract_error_syndrome_enhanced(SyndromeState* state,
                                     const quantum_state* qstate,
                                     ErrorSyndrome* syndrome);

#endif // SYNDROME_EXTRACTION_H
