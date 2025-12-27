/**
 * @file error_prediction.h
 * @brief Error prediction and forecasting types
 *
 * Provides types and functions for predicting future errors based on
 * pattern analysis and correlation data.
 */

#ifndef ERROR_PREDICTION_H
#define ERROR_PREDICTION_H

#include <stddef.h>
#include <stdbool.h>
#include "quantum_geometric/physics/error_patterns.h"
#include "quantum_geometric/physics/error_correlation.h"
#include "quantum_geometric/core/quantum_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Error Prediction Structure
// ============================================================================

/**
 * Error prediction result
 */
typedef struct ErrorPrediction {
    size_t x;                   // Predicted x coordinate
    size_t y;                   // Predicted y coordinate
    size_t z;                   // Predicted z coordinate / time layer
    PatternType type;           // Predicted error type
    double confidence;          // Prediction confidence [0, 1]
    size_t pattern_id;          // Associated pattern ID
    size_t correlation_id;      // Associated correlation ID
    size_t timestamp;           // Prediction timestamp
    double expected_weight;     // Expected error weight
    bool verified;              // Whether prediction was verified
} ErrorPrediction;

// ============================================================================
// Prediction Configuration
// ============================================================================

/**
 * Configuration for error prediction
 */
typedef struct PredictionConfig {
    size_t max_predictions;         // Maximum predictions to generate
    size_t history_length;          // Length of prediction history
    double confidence_threshold;    // Minimum confidence for predictions
    double temporal_weight;         // Weight for temporal correlations
    double spatial_weight;          // Weight for spatial correlations
    double min_success_rate;        // Minimum acceptable success rate
    double min_confidence;          // Minimum confidence to report
    bool enable_adaptive;           // Enable adaptive threshold adjustment
    size_t lookahead_steps;         // Number of timesteps to predict ahead
    double prediction_lookahead;    // Timing multiplier for prediction (adapts to detection latency)
} PredictionConfig;

// ============================================================================
// Hardware State for Prediction
// ============================================================================

/**
 * Hardware state snapshot for prediction history
 */
typedef struct PredictionHardwareState {
    double measurement_fidelity;    // Measurement fidelity at prediction time
    double gate_fidelity;           // Gate fidelity at prediction time
    double noise_level;             // Noise level at prediction time
    double coherence_factor;        // Coherence factor at prediction time
} PredictionHardwareState;

// ============================================================================
// Error Statistics for Prediction
// ============================================================================

/**
 * Error statistics tracked with predictions
 */
typedef struct PredictionErrorStats {
    double detection_latency;       // Time between prediction and detection
    double correction_overhead;     // Overhead for correction
    double prediction_accuracy;     // Accuracy of this prediction
    size_t correction_attempts;     // Number of correction attempts
} PredictionErrorStats;

// ============================================================================
// Prediction History Structure
// ============================================================================

/**
 * Historical prediction record
 */
typedef struct PredictionHistory {
    ErrorPrediction prediction;         // The prediction that was made
    bool was_correct;                   // Whether prediction was correct
    size_t timestamp;                   // When the prediction was made
    double confidence_delta;            // Change in confidence after verification
    PredictionHardwareState hardware_state; // Hardware state at prediction time
    PredictionErrorStats error_stats;   // Error statistics
} PredictionHistory;

// ============================================================================
// Prediction State
// ============================================================================

/**
 * State for prediction system
 */
typedef struct PredictionState {
    ErrorPrediction* predictions;       // Array of current predictions
    PredictionHistory* history;         // Historical prediction records
    size_t num_predictions;             // Current number of predictions
    size_t max_predictions;             // Maximum predictions capacity
    size_t history_length;              // History array length
    size_t current_history_index;       // Current position in history ring buffer
    double success_rate;                // Overall prediction success rate
    PredictionConfig config;            // Configuration parameters
} PredictionState;

// ============================================================================
// Initialization and Cleanup
// ============================================================================

/**
 * Initialize error prediction system
 * @param state State to initialize
 * @param config Configuration parameters
 * @return true on success, false on failure
 */
bool init_error_prediction(PredictionState* state, const PredictionConfig* config);

/**
 * Clean up error prediction resources
 * @param state State to clean up
 */
void cleanup_error_prediction(PredictionState* state);

// ============================================================================
// Prediction Functions
// ============================================================================

/**
 * Generate error predictions from patterns and correlations
 * @param state Prediction state
 * @param patterns Array of error patterns
 * @param num_patterns Number of patterns
 * @param correlations Array of correlations
 * @param num_correlations Number of correlations
 * @return Number of predictions generated
 */
size_t predict_errors(PredictionState* state,
                     const ErrorPattern* patterns,
                     size_t num_patterns,
                     const ErrorCorrelation* correlations,
                     size_t num_correlations);

/**
 * Verify predictions against current state
 * @param state Prediction state
 * @param current_state Current quantum state
 * @return true on success
 */
bool verify_predictions(PredictionState* state,
                       const quantum_state_t* current_state);

/**
 * Update prediction model based on history
 * @param state Prediction state
 * @return true on success
 */
bool update_prediction_model(PredictionState* state);

// ============================================================================
// Query Functions
// ============================================================================

/**
 * Get a specific prediction
 * @param state Prediction state
 * @param index Prediction index
 * @return Prediction pointer or NULL
 */
const ErrorPrediction* get_prediction(const PredictionState* state,
                                     size_t index);

/**
 * Get current prediction success rate
 * @param state Prediction state
 * @return Success rate [0, 1]
 */
double get_prediction_success_rate(const PredictionState* state);

// ============================================================================
// Hardware-Aware Helper Functions (External)
// ============================================================================

/**
 * Get spatial reliability factor for predictions
 * @return Spatial reliability [0, 1]
 */
double get_spatial_reliability_factor(void);

/**
 * Get temporal reliability factor for predictions
 * @return Temporal reliability [0, 1]
 */
double get_temporal_reliability_factor(void);

/**
 * Get cross-correlation factor
 * @return Cross-correlation factor [0, 1]
 */
double get_cross_correlation_factor(void);

/**
 * Get error decay rate
 * @return Decay rate
 */
double get_error_decay_rate(void);

/**
 * Get frequency stability factor
 * @return Stability factor [0, 1]
 */
double get_frequency_stability_factor(void);

/**
 * Get measurement fidelity
 * @return Measurement fidelity [0, 1]
 */
double get_measurement_fidelity(void);

/**
 * Get hardware error threshold
 * @return Error threshold
 */
double get_hardware_error_threshold(void);

/**
 * Calculate error weight at a position
 * @param state Quantum state
 * @param x X coordinate
 * @param y Y coordinate
 * @param z Z coordinate
 * @return Error weight
 */
double calculate_error_weight(const quantum_state_t* state,
                             size_t x, size_t y, size_t z);

/**
 * Get error weights of neighboring sites
 * @param state Quantum state
 * @param x X coordinate
 * @param y Y coordinate
 * @param z Z coordinate
 * @param weights Output array of 6 neighbor weights
 */
void get_neighbor_error_weights(const quantum_state_t* state,
                               size_t x, size_t y, size_t z,
                               double* weights);

/**
 * Get spread coefficient for a direction
 * @param direction Direction index (0-5)
 * @return Spread coefficient
 */
double get_spread_coefficient(int direction);

/**
 * Get spread factor
 * @return Spread factor
 */
double get_spread_factor(void);

/**
 * Get measurement correction factor
 * @return Correction factor
 */
double get_measurement_correction_factor(void);

/**
 * Get confidence scaling factor
 * @return Scaling factor
 */
double get_confidence_scaling(void);

/**
 * Get current noise level
 * @return Noise level [0, 1]
 */
double get_noise_level(void);

/**
 * Measure correction overhead
 * @return Correction overhead value
 */
double measure_correction_overhead(void);

/**
 * Calculate prediction accuracy
 * @param prediction Prediction to analyze
 * @return Accuracy value
 */
double calculate_prediction_accuracy(const ErrorPrediction* prediction);

/**
 * Check if feedback should be triggered based on history
 * @param history Prediction history entry
 * @return true if feedback should be triggered
 */
bool should_trigger_feedback(const PredictionHistory* history);

/**
 * Trigger prediction feedback mechanism
 * @param history Prediction history entry
 */
void trigger_prediction_feedback(const PredictionHistory* history);

#ifdef __cplusplus
}
#endif

#endif // ERROR_PREDICTION_H
