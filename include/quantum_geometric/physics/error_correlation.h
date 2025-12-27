/**
 * @file error_correlation.h
 * @brief Error correlation analysis for quantum error correction
 *
 * Provides types and functions for analyzing spatial and temporal
 * correlations in error syndromes, enabling more sophisticated
 * error correction strategies.
 */

#ifndef ERROR_CORRELATION_H
#define ERROR_CORRELATION_H

#include <stddef.h>
#include <stdbool.h>
#include "quantum_geometric/physics/error_syndrome.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Correlation Types
// ============================================================================

/**
 * Types of error correlations
 */
typedef enum CorrelationType {
    CORRELATION_NONE = 0,           // No correlation detected
    CORRELATION_SPATIAL,            // Spatial correlation only
    CORRELATION_TEMPORAL,           // Temporal correlation only
    CORRELATION_SPATIOTEMPORAL      // Both spatial and temporal
} CorrelationType;

// ============================================================================
// Configuration Structure
// ============================================================================

/**
 * Configuration for error correlation analysis
 */
typedef struct CorrelationConfig {
    size_t history_length;          // Number of syndrome rounds to track
    double spatial_threshold;       // Threshold for spatial correlation
    double temporal_threshold;      // Threshold for temporal correlation
    double max_correlation_dist;    // Maximum distance for correlation
    bool enable_cross_correlation;  // Enable cross-correlation analysis
    double correlation_decay_rate;  // Decay rate for historical correlations
    size_t min_samples;             // Minimum samples for reliable estimate
    double confidence_threshold;    // Confidence threshold for correlation
} CorrelationConfig;

// ============================================================================
// Error Correlation Results
// ============================================================================

/**
 * Error correlation analysis results
 */
typedef struct ErrorCorrelation {
    double spatial_correlation;     // Overall spatial correlation strength
    double temporal_correlation;    // Overall temporal correlation strength
    double cross_correlation;       // Cross-correlation (spatial-temporal)
    size_t correlation_length;      // Characteristic correlation length
    size_t correlation_time;        // Characteristic correlation time
    double confidence;              // Confidence in correlation estimate
    CorrelationType dominant_type;  // Dominant correlation type
    size_t sample_count;            // Number of samples analyzed

    // Spatial offsets for prediction
    int spatial_offset_x;           // X offset for predicted error location
    int spatial_offset_y;           // Y offset for predicted error location
    int spatial_offset_z;           // Z offset for predicted error location

    // Additional correlation metrics
    double drift_rate;              // Rate of correlation drift over time
    double stability_metric;        // Stability of correlation [0, 1]
} ErrorCorrelation;

// ============================================================================
// Initialization and Cleanup
// ============================================================================

/**
 * Initialize error correlation tracking
 * @param config Configuration parameters
 * @return true on success, false on failure
 */
bool init_error_correlation(const CorrelationConfig* config);

/**
 * Clean up error correlation resources
 */
void cleanup_error_correlation(void);

// ============================================================================
// Correlation Analysis Functions
// ============================================================================

/**
 * Analyze error correlations in a matching graph
 * @param graph Matching graph with syndrome vertices
 * @param state Quantum state (optional, for additional context)
 * @return Correlation analysis results
 */
ErrorCorrelation analyze_error_correlations(const MatchingGraph* graph,
                                           const quantum_state_t* state);

/**
 * Update correlation model with new data
 * @param graph Current matching graph
 * @param prev_correlation Previous correlation estimate
 * @return Updated correlation estimate
 */
ErrorCorrelation update_correlation_model(const MatchingGraph* graph,
                                         const ErrorCorrelation* prev_correlation);

/**
 * Detect the type of correlation between two syndrome vertices
 * @param vertex1 First syndrome vertex
 * @param vertex2 Second syndrome vertex
 * @return Type of correlation detected
 */
CorrelationType detect_correlation_type(const SyndromeVertex* vertex1,
                                       const SyndromeVertex* vertex2);

// ============================================================================
// Correlation Calculation Functions
// ============================================================================

/**
 * Calculate spatial correlation between two vertices
 * @param vertex1 First syndrome vertex
 * @param vertex2 Second syndrome vertex
 * @return Spatial correlation strength [0, 1]
 */
double calculate_spatial_correlation(const SyndromeVertex* vertex1,
                                    const SyndromeVertex* vertex2);

/**
 * Calculate temporal correlation between two vertices
 * @param vertex1 First syndrome vertex
 * @param vertex2 Second syndrome vertex
 * @return Temporal correlation strength [0, 1]
 */
double calculate_temporal_correlation(const SyndromeVertex* vertex1,
                                     const SyndromeVertex* vertex2);

/**
 * Calculate overall correlation strength between two vertices
 * @param v1 First syndrome vertex
 * @param v2 Second syndrome vertex
 * @return Combined correlation strength
 */
double calculate_correlation_strength(const SyndromeVertex* v1,
                                     const SyndromeVertex* v2);

// ============================================================================
// Correlation Testing Functions
// ============================================================================

/**
 * Test if two vertices are spatially correlated
 * @param v1 First syndrome vertex
 * @param v2 Second syndrome vertex
 * @param threshold Correlation threshold
 * @return true if spatially correlated above threshold
 */
bool is_spatially_correlated(const SyndromeVertex* v1,
                            const SyndromeVertex* v2,
                            double threshold);

/**
 * Test if two vertices are temporally correlated
 * @param v1 First syndrome vertex
 * @param v2 Second syndrome vertex
 * @param threshold Correlation threshold
 * @return true if temporally correlated above threshold
 */
bool is_temporally_correlated(const SyndromeVertex* v1,
                             const SyndromeVertex* v2,
                             double threshold);

// ============================================================================
// Scale Estimation Functions
// ============================================================================

/**
 * Estimate characteristic correlation length
 * @param graph Matching graph to analyze
 * @return Estimated correlation length in lattice units
 */
size_t estimate_correlation_length(const MatchingGraph* graph);

/**
 * Estimate characteristic correlation time
 * @param graph Matching graph to analyze
 * @return Estimated correlation time in syndrome rounds
 */
size_t estimate_correlation_time(const MatchingGraph* graph);

// ============================================================================
// Hardware-Aware Correlation Functions
// ============================================================================

/**
 * Get hardware reliability factor
 * @return Reliability factor [0, 1]
 */
double get_hardware_reliability_factor(void);

/**
 * Get overall noise factor
 * @return Noise factor [0, 1]
 */
double get_noise_factor(void);

/**
 * Get reliability factor for a specific qubit
 * @param qubit_index Qubit index
 * @return Qubit reliability [0, 1]
 */
double get_qubit_reliability(size_t qubit_index);

/**
 * Get average gate fidelity
 * @return Gate fidelity [0, 1]
 */
double get_gate_fidelity(void);

/**
 * Get crosstalk factor between two qubits
 * @param qubit1 First qubit index
 * @param qubit2 Second qubit index
 * @return Crosstalk factor [0, 1]
 */
double get_crosstalk_factor(size_t qubit1, size_t qubit2);

/**
 * Get temporal noise factor
 * @return Temporal noise factor [0, 1]
 */
double get_temporal_noise_factor(void);

/**
 * Get coherence time factor
 * @return Coherence time factor [0, 1]
 */
double get_coherence_time_factor(void);

/**
 * Get stability factor for a vertex
 * @param vertex_index Vertex index
 * @return Stability factor [0, 1]
 */
double get_vertex_stability(size_t vertex_index);

/**
 * Get measurement stability for a qubit
 * @param qubit_index Qubit index
 * @return Measurement stability [0, 1]
 */
double get_measurement_stability(size_t qubit_index);

/**
 * Get feedback factor between vertices
 * @param vertex1 First vertex index
 * @param vertex2 Second vertex index
 * @return Feedback factor
 */
double get_feedback_factor(size_t vertex1, size_t vertex2);

/**
 * Check if correlation feedback should be triggered
 * @param correlation Current correlation value
 * @return true if feedback should be triggered
 */
bool should_trigger_correlation_feedback(double correlation);

/**
 * Trigger correlation-based feedback
 * @param vertex1 First vertex index
 * @param vertex2 Second vertex index
 * @param correlation Correlation value
 */
void trigger_correlation_feedback(size_t vertex1, size_t vertex2, double correlation);

/**
 * Get correction chain length between vertices
 * @param v1 First syndrome vertex
 * @param v2 Second syndrome vertex
 * @return Chain length in lattice steps
 */
size_t get_correction_chain_length(const SyndromeVertex* v1,
                                  const SyndromeVertex* v2);

#ifdef __cplusplus
}
#endif

#endif // ERROR_CORRELATION_H
