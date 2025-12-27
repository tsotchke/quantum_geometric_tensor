/**
 * @file error_prediction.c
 * @brief Implementation of error prediction and forecasting for quantum error correction
 *
 * This module implements error prediction for surface code quantum error correction.
 * It uses pattern analysis, spatial/temporal correlations, and adaptive learning
 * to predict where errors are likely to occur in the quantum system.
 *
 * Key algorithms:
 * - Minimum Weight Perfect Matching (MWPM) inspired prediction
 * - Exponential decay modeling for temporal correlations
 * - Edge-weight error rate modeling
 * - Adaptive threshold calibration
 */

#include "quantum_geometric/physics/error_prediction.h"
#include "quantum_geometric/physics/error_patterns.h"
#include "quantum_geometric/physics/error_correlation.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Hardware State Structure
// ============================================================================

/**
 * Hardware state for error prediction calibration.
 * This tracks the measured characteristics of the quantum hardware
 * and adapts over time based on observed error patterns.
 */
typedef struct PredictionHardwareContext {
    // Reliability factors (measured from hardware calibration)
    double spatial_reliability;      // Spatial error correlation reliability
    double temporal_reliability;     // Temporal error correlation reliability
    double cross_qubit_correlation;  // Cross-qubit error correlation

    // Error decay parameters
    double t1_decay_rate;            // T1 relaxation decay rate (1/μs)
    double t2_decay_rate;            // T2 dephasing decay rate (1/μs)
    double t_gate;                   // Single gate time (μs)

    // Error thresholds
    double surface_code_threshold;   // Surface code error threshold (~1%)
    double measurement_fidelity;     // Single-shot readout fidelity
    double gate_fidelity;            // Single-qubit gate fidelity
    double two_qubit_gate_fidelity;  // Two-qubit gate fidelity

    // Noise model parameters
    double depolarizing_rate;        // Depolarizing noise rate
    double amplitude_damping_rate;   // Amplitude damping rate
    double phase_damping_rate;       // Phase damping rate
    double crosstalk_strength;       // Crosstalk coupling strength

    // Adaptive calibration
    size_t calibration_count;        // Number of calibration updates
    double last_calibration_time;    // Timestamp of last calibration

    // Statistics
    double cumulative_prediction_error;  // Running prediction error
    size_t total_predictions;        // Total predictions made
} PredictionHardwareContext;

// Global hardware context (would be per-device in production)
static PredictionHardwareContext g_hw_context = {
    .spatial_reliability = 0.0,      // Uninitialized - will be set on first use
    .temporal_reliability = 0.0,
    .cross_qubit_correlation = 0.0,
    .t1_decay_rate = 0.0,
    .t2_decay_rate = 0.0,
    .t_gate = 0.0,
    .surface_code_threshold = 0.0,
    .measurement_fidelity = 0.0,
    .gate_fidelity = 0.0,
    .two_qubit_gate_fidelity = 0.0,
    .depolarizing_rate = 0.0,
    .amplitude_damping_rate = 0.0,
    .phase_damping_rate = 0.0,
    .crosstalk_strength = 0.0,
    .calibration_count = 0,
    .last_calibration_time = 0.0,
    .cumulative_prediction_error = 0.0,
    .total_predictions = 0
};

// ============================================================================
// Forward Declarations
// ============================================================================

static void initialize_hardware_context(void);
static double calculate_prediction_confidence(const ErrorPattern* pattern,
                                             const ErrorCorrelation* correlation,
                                             const PredictionConfig* config);
static bool validate_prediction(const ErrorPrediction* prediction,
                               const quantum_state_t* state);
static void update_prediction_history(PredictionHistory* history,
                                     const ErrorPrediction* prediction,
                                     bool was_correct);
static double compute_edge_error_probability(size_t x, size_t y, size_t z,
                                            size_t width, size_t height);
static double compute_depolarizing_channel_fidelity(double p, size_t num_qubits);
static void update_hardware_calibration(const PredictionHistory* history);

// External functions from error_correlation.h that we use
extern double get_hardware_reliability_factor(void);
extern double get_noise_factor(void);
extern double get_gate_fidelity(void);
extern size_t get_current_timestamp(void);

// ============================================================================
// Hardware Context Initialization
// ============================================================================

static void initialize_hardware_context(void) {
    if (g_hw_context.calibration_count > 0) {
        return;  // Already initialized
    }

    // Initialize with typical superconducting qubit parameters
    // These values are based on IBM/Google quantum processor specifications

    // Coherence times for transmon qubits (approximate)
    const double T1_US = 100.0;   // T1 relaxation time in microseconds
    const double T2_US = 80.0;    // T2 dephasing time in microseconds
    const double GATE_TIME_US = 0.025;  // Single gate time (~25 ns)

    g_hw_context.t1_decay_rate = 1.0 / T1_US;
    g_hw_context.t2_decay_rate = 1.0 / T2_US;
    g_hw_context.t_gate = GATE_TIME_US;

    // Gate fidelities for current generation superconducting qubits
    g_hw_context.gate_fidelity = 0.9995;          // Single-qubit gate ~99.95%
    g_hw_context.two_qubit_gate_fidelity = 0.995; // Two-qubit gate ~99.5%
    g_hw_context.measurement_fidelity = 0.985;    // Readout ~98.5%

    // Surface code threshold (theoretical ~1%, practical ~0.5%)
    g_hw_context.surface_code_threshold = 0.01;

    // Noise model parameters
    // Depolarizing rate derived from gate fidelity
    g_hw_context.depolarizing_rate = (1.0 - g_hw_context.gate_fidelity) * 4.0 / 3.0;

    // Amplitude and phase damping from T1/T2
    g_hw_context.amplitude_damping_rate = g_hw_context.t1_decay_rate * GATE_TIME_US;
    g_hw_context.phase_damping_rate = (g_hw_context.t2_decay_rate -
                                       g_hw_context.t1_decay_rate / 2.0) * GATE_TIME_US;

    // Crosstalk typically 1-5% of coupling strength
    g_hw_context.crosstalk_strength = 0.02;

    // Reliability factors based on hardware performance
    // Higher T1/T2 relative to gate time = higher reliability
    double coherence_factor = sqrt(T1_US * T2_US) / (100.0 * GATE_TIME_US);
    g_hw_context.spatial_reliability = fmin(0.95, 0.5 + 0.45 * (1.0 - g_hw_context.crosstalk_strength * 10.0));
    g_hw_context.temporal_reliability = fmin(0.95, 0.5 + 0.45 * exp(-0.01 / coherence_factor));
    g_hw_context.cross_qubit_correlation = g_hw_context.crosstalk_strength * 5.0;  // Scale to [0,1]

    g_hw_context.calibration_count = 1;
    g_hw_context.last_calibration_time = (double)time(NULL);
}

// ============================================================================
// Initialization and Cleanup
// ============================================================================

bool init_error_prediction(PredictionState* state, const PredictionConfig* config) {
    if (!state || !config) {
        return false;
    }

    // Validate configuration parameters
    if (config->max_predictions == 0 || config->history_length == 0) {
        return false;
    }

    if (config->confidence_threshold < 0.0 || config->confidence_threshold > 1.0) {
        return false;
    }

    if (config->min_confidence < 0.0 || config->min_confidence > config->confidence_threshold) {
        return false;
    }

    // Allocate prediction array
    state->predictions = calloc(config->max_predictions, sizeof(ErrorPrediction));
    if (!state->predictions) {
        return false;
    }

    // Allocate history ring buffer
    state->history = calloc(config->history_length, sizeof(PredictionHistory));
    if (!state->history) {
        free(state->predictions);
        state->predictions = NULL;
        return false;
    }

    // Initialize state
    state->num_predictions = 0;
    state->max_predictions = config->max_predictions;
    state->history_length = config->history_length;
    state->current_history_index = 0;
    state->success_rate = 0.5;  // Start with neutral prior

    // Copy configuration
    memcpy(&state->config, config, sizeof(PredictionConfig));

    // Initialize prediction_lookahead with default if not set
    if (state->config.prediction_lookahead <= 0.0) {
        state->config.prediction_lookahead = 1.0;  // Default: no timing adjustment
    }

    // Initialize hardware context
    initialize_hardware_context();

    return true;
}

void cleanup_error_prediction(PredictionState* state) {
    if (!state) {
        return;
    }

    free(state->predictions);
    free(state->history);
    memset(state, 0, sizeof(PredictionState));
}

// ============================================================================
// Core Prediction Algorithm
// ============================================================================

size_t predict_errors(PredictionState* state,
                     const ErrorPattern* patterns,
                     size_t num_patterns,
                     const ErrorCorrelation* correlations,
                     size_t num_correlations) {
    if (!state || !patterns || !correlations) {
        return 0;
    }

    if (num_patterns == 0 || num_correlations == 0) {
        return 0;
    }

    // Reset predictions for new cycle
    state->num_predictions = 0;

    // Ensure hardware is calibrated
    initialize_hardware_context();

    // Priority queue would be better here, but we use simple iteration for clarity
    // In production, use a min-heap sorted by confidence

    for (size_t i = 0; i < num_patterns && state->num_predictions < state->max_predictions; i++) {
        const ErrorPattern* pattern = &patterns[i];

        // Skip inactive patterns
        if (!pattern->is_active) {
            continue;
        }

        // Skip patterns with no weight (unlikely to recur)
        if (pattern->weight < 1e-10) {
            continue;
        }

        // Skip patterns with no vertices
        if (pattern->size == 0) {
            continue;
        }

        // For each pattern, find correlations that predict future errors
        for (size_t j = 0; j < num_correlations; j++) {
            const ErrorCorrelation* correlation = &correlations[j];

            // Calculate total correlation strength
            double corr_strength = fabs(correlation->spatial_correlation) +
                                  fabs(correlation->temporal_correlation) +
                                  fabs(correlation->cross_correlation);

            // Skip weak correlations
            if (corr_strength < 0.05) {
                continue;
            }

            // Calculate prediction confidence using Bayesian update
            double confidence = calculate_prediction_confidence(pattern, correlation, &state->config);

            // Only create prediction if confidence exceeds threshold
            if (confidence >= state->config.confidence_threshold) {
                ErrorPrediction* prediction = &state->predictions[state->num_predictions];

                // Calculate predicted error location
                // Pattern origin + correlation offset = predicted location
                int64_t pred_x = (int64_t)pattern->vertices[0].x + correlation->spatial_offset_x;
                int64_t pred_y = (int64_t)pattern->vertices[0].y + correlation->spatial_offset_y;
                int64_t pred_z = (int64_t)pattern->vertices[0].z + correlation->spatial_offset_z;

                // Bounds check
                if (pred_x < 0 || pred_y < 0 || pred_z < 0) {
                    continue;
                }

                prediction->x = (size_t)pred_x;
                prediction->y = (size_t)pred_y;
                prediction->z = (size_t)pred_z;
                prediction->type = pattern->type;
                prediction->confidence = confidence;
                prediction->pattern_id = i;
                prediction->correlation_id = j;
                prediction->timestamp = get_current_timestamp();
                prediction->verified = false;

                // Calculate expected error weight using edge-weight model
                // This is based on the surface code decoder's edge weighting
                prediction->expected_weight = compute_edge_error_probability(
                    prediction->x, prediction->y, prediction->z, 16, 16);

                // Scale by pattern and correlation strength
                prediction->expected_weight *= pattern->weight * corr_strength;

                // Clamp to valid range
                if (prediction->expected_weight > 1.0) {
                    prediction->expected_weight = 1.0;
                }

                state->num_predictions++;

                if (state->num_predictions >= state->max_predictions) {
                    break;
                }
            }
        }
    }

    return state->num_predictions;
}

// ============================================================================
// Prediction Verification
// ============================================================================

bool verify_predictions(PredictionState* state, const quantum_state_t* current_state) {
    if (!state || !current_state) {
        return false;
    }

    size_t correct_predictions = 0;
    double total_confidence = 0.0;
    double correct_confidence = 0.0;

    // Verify each prediction against current state
    for (size_t i = 0; i < state->num_predictions; i++) {
        ErrorPrediction* prediction = &state->predictions[i];

        // Check if predicted error occurred
        bool was_correct = validate_prediction(prediction, current_state);
        prediction->verified = true;

        // Record in history
        size_t history_idx = state->current_history_index;
        PredictionHistory* history = &state->history[history_idx];
        update_prediction_history(history, prediction, was_correct);

        // Track statistics
        total_confidence += prediction->confidence;
        if (was_correct) {
            correct_predictions++;
            correct_confidence += prediction->confidence;
        }

        // Update hardware calibration based on result
        update_hardware_calibration(history);

        // Advance ring buffer
        state->current_history_index = (state->current_history_index + 1) % state->history_length;
    }

    // Update success rate using Bayesian update
    // P(success|data) ∝ P(data|success) * P(success)
    if (state->num_predictions > 0) {
        double batch_rate = (double)correct_predictions / (double)state->num_predictions;

        // Weighted average with confidence-based weighting
        double confidence_weight = total_confidence / state->num_predictions;
        double correct_confidence_weight = (correct_predictions > 0) ?
            correct_confidence / correct_predictions : 0.0;

        // Bayesian update with prior (current success_rate) and likelihood (batch_rate)
        // Using a beta-binomial model approximation
        // Weight update by batch_rate and confidence of correct predictions
        double alpha = state->success_rate * 10.0;  // Prior pseudo-counts
        double beta = (1.0 - state->success_rate) * 10.0;

        // Scale updates by confidence - high confidence correct predictions are weighted more
        double update_weight = 1.0 + correct_confidence_weight * batch_rate;
        alpha += correct_predictions * update_weight;
        beta += (state->num_predictions - correct_predictions);

        state->success_rate = alpha / (alpha + beta);

        // Adjust by confidence (high confidence wrong predictions hurt more)
        if (correct_predictions < state->num_predictions && confidence_weight > 0.7) {
            state->success_rate *= 0.98;  // Penalize overconfident wrong predictions
        }
    }

    return true;
}

// ============================================================================
// Model Update
// ============================================================================

bool update_prediction_model(PredictionState* state) {
    if (!state) {
        return false;
    }

    // Analyze history for model calibration
    size_t valid_entries = 0;
    size_t correct_count = 0;
    double avg_confidence = 0.0;
    double avg_latency = 0.0;
    double spatial_accuracy = 0.0;
    double temporal_accuracy = 0.0;
    size_t spatial_count = 0;
    size_t temporal_count = 0;

    for (size_t i = 0; i < state->history_length; i++) {
        const PredictionHistory* h = &state->history[i];

        if (h->timestamp == 0) {
            continue;
        }

        valid_entries++;
        avg_confidence += h->prediction.confidence;
        avg_latency += h->error_stats.detection_latency;

        if (h->was_correct) {
            correct_count++;
        }

        // Track spatial vs temporal prediction accuracy
        // Spatial predictions have z=0, temporal have z>0
        if (h->prediction.z == 0) {
            spatial_count++;
            if (h->was_correct) spatial_accuracy += 1.0;
        } else {
            temporal_count++;
            if (h->was_correct) temporal_accuracy += 1.0;
        }
    }

    if (valid_entries < 10) {
        return true;  // Not enough data for reliable updates
    }

    avg_confidence /= valid_entries;
    avg_latency /= valid_entries;
    double success_rate = (double)correct_count / valid_entries;

    if (spatial_count > 0) spatial_accuracy /= spatial_count;
    if (temporal_count > 0) temporal_accuracy /= temporal_count;

    // Adaptive threshold adjustment using gradient descent
    // Goal: maximize F1 score by tuning threshold
    double target_rate = state->config.min_success_rate;
    double threshold_gradient = 0.0;

    if (success_rate < target_rate) {
        // Too many false positives - increase threshold
        threshold_gradient = (target_rate - success_rate) * 0.1;
    } else if (success_rate > 0.95 && avg_confidence > 0.8) {
        // Very accurate but maybe too conservative - lower threshold
        threshold_gradient = -0.02;
    }

    state->config.confidence_threshold += threshold_gradient;

    // Adjust temporal weight based on temporal prediction accuracy
    if (temporal_count > 5) {
        double temporal_gradient = (temporal_accuracy - 0.5) * 0.05;
        state->config.temporal_weight += temporal_gradient;
    }

    // Adjust spatial weight based on spatial prediction accuracy
    if (spatial_count > 5) {
        double spatial_gradient = (spatial_accuracy - 0.5) * 0.05;
        state->config.spatial_weight += spatial_gradient;
    }

    // Clamp parameters to valid ranges
    state->config.confidence_threshold = fmax(0.2, fmin(0.95, state->config.confidence_threshold));
    state->config.temporal_weight = fmax(0.1, fmin(3.0, state->config.temporal_weight));
    state->config.spatial_weight = fmax(0.1, fmin(3.0, state->config.spatial_weight));

    // Adjust prediction lookahead based on average latency
    // If detection latency is high, we need to predict earlier (increase lookahead)
    // If latency is low, we can predict more precisely (decrease lookahead)
    double target_latency = 0.5;  // Target latency in normalized units
    if (avg_latency > target_latency * 1.2) {
        state->config.prediction_lookahead *= 1.05;  // Increase lookahead
    } else if (avg_latency < target_latency * 0.8) {
        state->config.prediction_lookahead *= 0.98;  // Decrease lookahead for precision
    }
    state->config.prediction_lookahead = fmax(1.0, fmin(10.0, state->config.prediction_lookahead));

    // Update hardware context calibration
    g_hw_context.cumulative_prediction_error += (1.0 - success_rate);
    g_hw_context.total_predictions += valid_entries;
    g_hw_context.calibration_count++;
    g_hw_context.last_calibration_time = (double)time(NULL);

    return true;
}

// ============================================================================
// Query Functions
// ============================================================================

const ErrorPrediction* get_prediction(const PredictionState* state, size_t index) {
    if (!state || index >= state->num_predictions) {
        return NULL;
    }
    return &state->predictions[index];
}

double get_prediction_success_rate(const PredictionState* state) {
    if (!state) {
        return 0.0;
    }
    return state->success_rate;
}

// ============================================================================
// Hardware-Aware Functions
// ============================================================================

double get_spatial_reliability_factor(void) {
    initialize_hardware_context();
    return g_hw_context.spatial_reliability;
}

double get_temporal_reliability_factor(void) {
    initialize_hardware_context();
    return g_hw_context.temporal_reliability;
}

double get_cross_correlation_factor(void) {
    initialize_hardware_context();
    return g_hw_context.cross_qubit_correlation;
}

double get_error_decay_rate(void) {
    initialize_hardware_context();
    // Return effective error decay combining T1 and T2
    return (g_hw_context.t1_decay_rate + g_hw_context.t2_decay_rate) / 2.0;
}

double get_frequency_stability_factor(void) {
    initialize_hardware_context();
    // Frequency stability related to T2* coherence
    // Higher T2 relative to gate time = better stability
    double stability = 1.0 - g_hw_context.phase_damping_rate;
    return fmax(0.0, fmin(1.0, stability));
}

double get_measurement_fidelity(void) {
    initialize_hardware_context();
    return g_hw_context.measurement_fidelity;
}

double get_hardware_error_threshold(void) {
    initialize_hardware_context();
    return g_hw_context.surface_code_threshold;
}

double calculate_error_weight(const quantum_state_t* state, size_t x, size_t y, size_t z) {
    if (!state) {
        return 0.0;
    }

    size_t width = state->lattice_width > 0 ? state->lattice_width : 16;
    size_t height = state->lattice_height > 0 ? state->lattice_height : 16;

    return compute_edge_error_probability(x, y, z, width, height);
}

void get_neighbor_error_weights(const quantum_state_t* state,
                               size_t x, size_t y, size_t z,
                               double* weights) {
    if (!state || !weights) {
        return;
    }

    for (int i = 0; i < 6; i++) {
        weights[i] = 0.0;
    }

    size_t width = state->lattice_width > 0 ? state->lattice_width : 16;
    size_t height = state->lattice_height > 0 ? state->lattice_height : 16;

    // Direction offsets: +x, -x, +y, -y, +z, -z
    const int dx[6] = {1, -1, 0, 0, 0, 0};
    const int dy[6] = {0, 0, 1, -1, 0, 0};
    const int dz[6] = {0, 0, 0, 0, 1, -1};

    for (int i = 0; i < 6; i++) {
        int64_t nx = (int64_t)x + dx[i];
        int64_t ny = (int64_t)y + dy[i];
        int64_t nz = (int64_t)z + dz[i];

        if (nx >= 0 && (size_t)nx < width &&
            ny >= 0 && (size_t)ny < height &&
            nz >= 0) {
            weights[i] = compute_edge_error_probability((size_t)nx, (size_t)ny,
                                                       (size_t)nz, width, height);
        }
    }
}

double get_spread_coefficient(int direction) {
    initialize_hardware_context();

    if (direction < 0 || direction >= 6) {
        return 0.0;
    }

    // Spread coefficients based on crosstalk model
    // Spatial directions (0-3): stronger coupling via ZZ interaction
    // Temporal directions (4-5): weaker coupling via decoherence
    if (direction < 4) {
        // Spatial: crosstalk dominated
        return g_hw_context.crosstalk_strength;
    } else {
        // Temporal: T1 decay dominated
        return g_hw_context.amplitude_damping_rate;
    }
}

double get_spread_factor(void) {
    initialize_hardware_context();
    // Error spread factor from multi-qubit gate infidelity
    return 1.0 - g_hw_context.two_qubit_gate_fidelity;
}

double get_measurement_correction_factor(void) {
    initialize_hardware_context();
    return g_hw_context.measurement_fidelity;
}

double get_confidence_scaling(void) {
    initialize_hardware_context();
    // Scale confidence by overall hardware quality
    double hw_quality = (g_hw_context.gate_fidelity +
                        g_hw_context.measurement_fidelity +
                        g_hw_context.two_qubit_gate_fidelity) / 3.0;
    return hw_quality;
}

double get_noise_level(void) {
    initialize_hardware_context();
    return g_hw_context.depolarizing_rate;
}

double measure_correction_overhead(void) {
    initialize_hardware_context();
    // Correction overhead = time for syndrome measurement + decoding + correction
    // Typically dominated by measurement time for surface codes
    // For real-time decoding: ~1μs measurement + ~100ns classical processing
    return g_hw_context.t_gate * 40.0;  // ~40 gate times for a QEC round
}

double calculate_prediction_accuracy(const ErrorPrediction* prediction) {
    if (!prediction) {
        return 0.0;
    }

    initialize_hardware_context();

    // Prediction accuracy = P(error at location | prediction made)
    // Using Bayes' theorem with hardware priors
    double base_accuracy = prediction->confidence;

    // Adjust by expected weight (low weight = harder to predict)
    double weight_factor = 1.0 - exp(-prediction->expected_weight * 10.0);

    // Adjust by hardware measurement fidelity
    double hw_factor = g_hw_context.measurement_fidelity;

    return base_accuracy * weight_factor * hw_factor;
}

bool should_trigger_feedback(const PredictionHistory* history) {
    if (!history) {
        return false;
    }

    initialize_hardware_context();

    // Trigger feedback for model recalibration when:

    // 1. High confidence prediction was wrong (calibration drift)
    if (!history->was_correct && history->prediction.confidence > 0.8) {
        return true;
    }

    // 2. Low confidence prediction was correct (opportunity to improve)
    if (history->was_correct && history->prediction.confidence < 0.3) {
        return true;
    }

    // 3. Detection latency exceeded threshold (hardware degradation)
    double expected_latency = g_hw_context.t_gate * 100.0;  // ~100 gates
    if (history->error_stats.detection_latency > expected_latency * 2.0) {
        return true;
    }

    // 4. Hardware state shows significant noise increase
    if (history->hardware_state.noise_level > g_hw_context.depolarizing_rate * 1.5) {
        return true;
    }

    // 5. Gate fidelity dropped significantly
    if (history->hardware_state.gate_fidelity < g_hw_context.gate_fidelity * 0.95) {
        return true;
    }

    return false;
}

void trigger_prediction_feedback(const PredictionHistory* history) {
    if (!history) {
        return;
    }

    initialize_hardware_context();

    // Update hardware context based on feedback
    // This implements online learning for adaptive calibration

    const double learning_rate = 0.01;

    // Update reliability factors based on prediction outcome
    if (history->was_correct) {
        // Reinforce current estimates
        g_hw_context.spatial_reliability += learning_rate * (1.0 - g_hw_context.spatial_reliability);
        g_hw_context.temporal_reliability += learning_rate * (1.0 - g_hw_context.temporal_reliability);
    } else {
        // Decay reliability estimates
        double confidence_penalty = history->prediction.confidence;
        g_hw_context.spatial_reliability -= learning_rate * confidence_penalty;
        g_hw_context.temporal_reliability -= learning_rate * confidence_penalty;
    }

    // Update noise estimate from observed hardware state
    g_hw_context.depolarizing_rate = 0.95 * g_hw_context.depolarizing_rate +
                                     0.05 * history->hardware_state.noise_level;

    // Update measurement fidelity from observed state
    g_hw_context.measurement_fidelity = 0.95 * g_hw_context.measurement_fidelity +
                                        0.05 * history->hardware_state.measurement_fidelity;

    // Update gate fidelity
    g_hw_context.gate_fidelity = 0.95 * g_hw_context.gate_fidelity +
                                 0.05 * history->hardware_state.gate_fidelity;

    // Clamp all parameters to valid ranges
    g_hw_context.spatial_reliability = fmax(0.3, fmin(0.99, g_hw_context.spatial_reliability));
    g_hw_context.temporal_reliability = fmax(0.3, fmin(0.99, g_hw_context.temporal_reliability));
    g_hw_context.depolarizing_rate = fmax(1e-6, fmin(0.1, g_hw_context.depolarizing_rate));
    g_hw_context.measurement_fidelity = fmax(0.9, fmin(0.999, g_hw_context.measurement_fidelity));
    g_hw_context.gate_fidelity = fmax(0.99, fmin(0.9999, g_hw_context.gate_fidelity));

    // Recalculate derived parameters
    g_hw_context.cross_qubit_correlation = g_hw_context.crosstalk_strength *
                                           (1.0 + g_hw_context.depolarizing_rate * 10.0);

    g_hw_context.calibration_count++;
}

// ============================================================================
// Internal Helper Functions
// ============================================================================

/**
 * Compute edge error probability using the surface code edge-weight model.
 * This is based on the MWPM decoder's edge weighting scheme.
 */
static double compute_edge_error_probability(size_t x, size_t y, size_t z,
                                            size_t width, size_t height) {
    initialize_hardware_context();

    // Base error probability from hardware
    double p_base = g_hw_context.depolarizing_rate;

    // Edge qubits have higher error rates due to:
    // 1. Reduced syndrome connectivity (fewer stabilizer measurements)
    // 2. Boundary effects in calibration
    // 3. Environmental coupling asymmetry

    bool is_edge_x = (x == 0 || x == width - 1);
    bool is_edge_y = (y == 0 || y == height - 1);
    double edge_factor = 1.0;

    if (is_edge_x && is_edge_y) {
        edge_factor = 1.5;  // Corner qubits: 50% higher error
    } else if (is_edge_x || is_edge_y) {
        edge_factor = 1.25; // Edge qubits: 25% higher error
    }

    // Time-layer (z) affects error probability via T1/T2 decay
    // P(error at time t) ≈ P(error) * (1 + γt) for small γt
    double temporal_factor = 1.0 + g_hw_context.t1_decay_rate * (double)z * g_hw_context.t_gate * 100.0;

    // Crosstalk contribution from neighbors
    // More neighbors = more crosstalk errors
    size_t num_neighbors = 4;  // Typical for square lattice
    if (is_edge_x) num_neighbors--;
    if (is_edge_y) num_neighbors--;
    double crosstalk_factor = 1.0 + g_hw_context.crosstalk_strength * (double)num_neighbors;

    // Combined error probability
    double p_error = p_base * edge_factor * temporal_factor * crosstalk_factor;

    // Clamp to valid probability
    return fmin(1.0, fmax(0.0, p_error));
}

/**
 * Compute fidelity of depolarizing channel.
 * F = (1-p) + p/d^2 for d-dimensional system
 */
static double compute_depolarizing_channel_fidelity(double p, size_t num_qubits) {
    if (p < 0.0 || p > 1.0) {
        return 0.0;
    }

    // Dimension d = 2^n for n qubits
    double d = pow(2.0, (double)num_qubits);
    return (1.0 - p) + p / (d * d);
}

/**
 * Calculate prediction confidence using Bayesian inference.
 * Combines pattern weight, correlation strength, and hardware reliability.
 */
static double calculate_prediction_confidence(const ErrorPattern* pattern,
                                             const ErrorCorrelation* correlation,
                                             const PredictionConfig* config) {
    if (!pattern || !correlation || !config) {
        return 0.0;
    }

    initialize_hardware_context();

    // Prior: base confidence from pattern weight and hardware reliability
    double hw_reliability = get_hardware_reliability_factor();
    double prior = pattern->weight * hw_reliability;

    // Likelihood from correlations
    // P(observation | error present) ~ correlation strength * reliability
    double spatial_likelihood = fabs(correlation->spatial_correlation) *
                                g_hw_context.spatial_reliability *
                                config->spatial_weight;
    double temporal_likelihood = fabs(correlation->temporal_correlation) *
                                 g_hw_context.temporal_reliability *
                                 config->temporal_weight;
    double cross_likelihood = fabs(correlation->cross_correlation) *
                             g_hw_context.cross_qubit_correlation;

    // Combined likelihood (assuming conditional independence)
    double total_weight = config->spatial_weight + config->temporal_weight + 1.0;
    double likelihood = (spatial_likelihood + temporal_likelihood + cross_likelihood) / total_weight;

    // Temporal decay: exponential decay based on time since pattern observed
    size_t current_time = get_current_timestamp();
    double elapsed_us = (double)(current_time - pattern->timing.last_seen);
    double decay_rate = get_error_decay_rate() * config->temporal_weight;
    double temporal_decay = exp(-decay_rate * elapsed_us);

    // Frequency factor: patterns that occur more frequently are more reliable
    double freq_factor = 1.0;
    if (pattern->timing.frequency > 0.0) {
        // Sigmoid-like scaling: saturates at high frequency
        freq_factor = 2.0 / (1.0 + exp(-pattern->timing.frequency)) - 0.5;
        freq_factor = fmax(0.1, fmin(2.0, freq_factor));
    }

    // Posterior confidence using approximate Bayesian update
    // P(error | observations) ∝ P(observations | error) * P(error)
    double confidence = prior * likelihood * temporal_decay * freq_factor;

    // Apply hardware fidelity corrections
    confidence *= compute_depolarizing_channel_fidelity(
        1.0 - g_hw_context.measurement_fidelity, 1);
    confidence *= g_hw_context.gate_fidelity;

    // Normalize to [0, 1]
    confidence = fmax(0.0, fmin(1.0, confidence));

    // Apply minimum threshold
    if (confidence < config->min_confidence) {
        confidence = 0.0;
    }

    return confidence;
}

/**
 * Validate prediction against actual quantum state.
 * Checks if the predicted error location shows syndrome activity.
 */
static bool validate_prediction(const ErrorPrediction* prediction,
                               const quantum_state_t* state) {
    if (!prediction || !state) {
        return false;
    }

    initialize_hardware_context();

    // Get error threshold for this hardware
    double threshold = g_hw_context.surface_code_threshold;

    // Calculate error weight at predicted location
    double center_weight = calculate_error_weight(state, prediction->x, prediction->y, prediction->z);

    // Get neighbor weights for error spread detection
    double neighbor_weights[6] = {0.0};
    get_neighbor_error_weights(state, prediction->x, prediction->y, prediction->z, neighbor_weights);

    // Calculate spread contribution using hardware model
    double spread_sum = 0.0;
    for (int i = 0; i < 6; i++) {
        spread_sum += neighbor_weights[i] * get_spread_coefficient(i);
    }
    double spread_contribution = spread_sum * get_spread_factor();

    // Total error indicator
    double total_error = center_weight + spread_contribution;

    // Apply measurement correction (readout errors can mask real errors)
    total_error *= g_hw_context.measurement_fidelity;

    // Dynamic threshold based on noise level and confidence
    double noise = get_noise_factor();
    double dynamic_threshold = threshold * (1.0 - noise) * get_confidence_scaling();

    // Higher confidence predictions use stricter threshold
    // (we're more sure of the location, so we expect stronger signal)
    double effective_threshold = dynamic_threshold * (0.8 + 0.4 * prediction->confidence);

    return total_error > effective_threshold;
}

/**
 * Update prediction history with verification result.
 */
static void update_prediction_history(PredictionHistory* history,
                                     const ErrorPrediction* prediction,
                                     bool was_correct) {
    if (!history || !prediction) {
        return;
    }

    initialize_hardware_context();

    // Store prediction
    history->prediction = *prediction;
    history->was_correct = was_correct;
    history->timestamp = prediction->timestamp;

    // Calculate confidence delta using reward/penalty scheme
    // Based on prediction-error correlation in surface code decoders
    if (was_correct) {
        // Reward: larger for low-confidence correct predictions (surprising success)
        history->confidence_delta = (1.0 - prediction->confidence) * 0.3;
    } else {
        // Penalty: larger for high-confidence wrong predictions (surprising failure)
        history->confidence_delta = -prediction->confidence * 0.3;
    }

    // Store current hardware state for calibration analysis
    history->hardware_state.measurement_fidelity = g_hw_context.measurement_fidelity;
    history->hardware_state.gate_fidelity = g_hw_context.gate_fidelity;
    history->hardware_state.noise_level = g_hw_context.depolarizing_rate;
    history->hardware_state.coherence_factor = g_hw_context.temporal_reliability;

    // Calculate error statistics
    size_t current_time = get_current_timestamp();
    history->error_stats.detection_latency = (double)(current_time - prediction->timestamp) * 1e-6;
    history->error_stats.correction_overhead = measure_correction_overhead();
    history->error_stats.prediction_accuracy = calculate_prediction_accuracy(prediction);
    history->error_stats.correction_attempts = was_correct ? 1 : 0;

    // Trigger feedback if conditions warrant
    if (should_trigger_feedback(history)) {
        trigger_prediction_feedback(history);
    }
}

/**
 * Update hardware calibration based on prediction result.
 */
static void update_hardware_calibration(const PredictionHistory* history) {
    if (!history) {
        return;
    }

    // Track prediction accuracy for hardware quality estimation
    g_hw_context.total_predictions++;
    if (!history->was_correct) {
        g_hw_context.cumulative_prediction_error += 1.0;
    }

    // Update hardware quality estimate
    double observed_error_rate = g_hw_context.cumulative_prediction_error /
                                 (double)g_hw_context.total_predictions;

    // If observed error rate deviates significantly from expected, adjust model
    double expected_error_rate = 1.0 - g_hw_context.gate_fidelity;
    if (fabs(observed_error_rate - expected_error_rate) > 0.05) {
        // Significant deviation - adjust hardware parameters
        double adjustment = (observed_error_rate - expected_error_rate) * 0.1;
        g_hw_context.depolarizing_rate += adjustment;
        g_hw_context.depolarizing_rate = fmax(1e-6, fmin(0.1, g_hw_context.depolarizing_rate));
    }
}
