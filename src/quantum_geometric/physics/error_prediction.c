/**
 * @file error_prediction.c
 * @brief Implementation of error prediction and forecasting
 */

#include "quantum_geometric/physics/error_prediction.h"
#include "quantum_geometric/physics/error_patterns.h"
#include "quantum_geometric/physics/error_correlation.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Forward declarations of helper functions
static double calculate_prediction_confidence(const ErrorPattern* pattern,
                                           const ErrorCorrelation* correlation,
                                           const PredictionConfig* config);
static bool validate_prediction(const ErrorPrediction* prediction,
                              const quantum_state* state);
static void update_prediction_history(PredictionHistory* history,
                                    const ErrorPrediction* prediction,
                                    bool was_correct);

bool init_error_prediction(PredictionState* state, const PredictionConfig* config) {
    if (!state || !config) {
        return false;
    }

    state->predictions = calloc(config->max_predictions, sizeof(ErrorPrediction));
    state->history = calloc(config->history_length, sizeof(PredictionHistory));
    
    if (!state->predictions || !state->history) {
        free(state->predictions);
        free(state->history);
        return false;
    }

    state->num_predictions = 0;
    state->max_predictions = config->max_predictions;
    state->history_length = config->history_length;
    state->current_history_index = 0;
    memcpy(&state->config, config, sizeof(PredictionConfig));

    return true;
}

void cleanup_error_prediction(PredictionState* state) {
    if (state) {
        free(state->predictions);
        free(state->history);
        memset(state, 0, sizeof(PredictionState));
    }
}

size_t predict_errors(PredictionState* state,
                     const ErrorPattern* patterns,
                     size_t num_patterns,
                     const ErrorCorrelation* correlations,
                     size_t num_correlations) {
    if (!state || !patterns || !correlations) {
        return 0;
    }

    // Reset predictions
    state->num_predictions = 0;

    // Generate predictions from patterns and correlations
    for (size_t i = 0; i < num_patterns && 
         state->num_predictions < state->max_predictions; i++) {
        const ErrorPattern* pattern = &patterns[i];

        // Skip inactive patterns
        if (!pattern->is_active) {
            continue;
        }

        // Find relevant correlations
        for (size_t j = 0; j < num_correlations; j++) {
            const ErrorCorrelation* correlation = &correlations[j];

            // Calculate prediction confidence
            double confidence = calculate_prediction_confidence(pattern,
                                                             correlation,
                                                             &state->config);

            // Add prediction if confidence is high enough
            if (confidence > state->config.confidence_threshold) {
                ErrorPrediction* prediction = &state->predictions[state->num_predictions];
                
                // Set prediction location based on pattern and correlation
                prediction->x = pattern->vertices[0].x + correlation->spatial_offset_x;
                prediction->y = pattern->vertices[0].y + correlation->spatial_offset_y;
                prediction->z = pattern->vertices[0].z + correlation->spatial_offset_z;
                
                prediction->type = pattern->type;
                prediction->confidence = confidence;
                prediction->pattern_id = i;
                prediction->correlation_id = j;
                prediction->timestamp = get_current_timestamp();

                state->num_predictions++;
            }
        }
    }

    return state->num_predictions;
}

bool verify_predictions(PredictionState* state,
                       const quantum_state* current_state) {
    if (!state || !current_state) {
        return false;
    }

    size_t correct_predictions = 0;

    // Verify each prediction
    for (size_t i = 0; i < state->num_predictions; i++) {
        ErrorPrediction* prediction = &state->predictions[i];
        bool was_correct = validate_prediction(prediction, current_state);

        // Update prediction history
        PredictionHistory* history = &state->history[state->current_history_index];
        update_prediction_history(history, prediction, was_correct);

        if (was_correct) {
            correct_predictions++;
        }

        // Move to next history slot
        state->current_history_index = (state->current_history_index + 1) %
                                     state->history_length;
    }

    // Calculate and update success rate
    if (state->num_predictions > 0) {
        state->success_rate = (double)correct_predictions / state->num_predictions;
    }

    return true;
}

bool update_prediction_model(PredictionState* state) {
    if (!state) {
        return false;
    }

    // Analyze prediction history
    size_t total_predictions = 0;
    size_t correct_predictions = 0;
    double avg_confidence = 0.0;

    for (size_t i = 0; i < state->history_length; i++) {
        const PredictionHistory* history = &state->history[i];
        if (history->timestamp > 0) {  // Valid history entry
            total_predictions++;
            if (history->was_correct) {
                correct_predictions++;
            }
            avg_confidence += history->prediction.confidence;
        }
    }

    if (total_predictions > 0) {
        // Update model parameters based on history
        double success_rate = (double)correct_predictions / total_predictions;
        avg_confidence /= total_predictions;

        // Adjust confidence threshold based on success rate
        if (success_rate < state->config.min_success_rate) {
            state->config.confidence_threshold *= 1.1;  // Increase threshold
        } else if (success_rate > 0.95) {  // Very high success rate
            state->config.confidence_threshold *= 0.95;  // Decrease threshold
        }

        // Adjust temporal weight based on prediction accuracy
        if (success_rate > 0.8) {
            state->config.temporal_weight *= 1.05;  // Increase temporal importance
        } else {
            state->config.temporal_weight *= 0.95;  // Decrease temporal importance
        }

        // Keep parameters within valid ranges
        if (state->config.confidence_threshold > 0.95) {
            state->config.confidence_threshold = 0.95;
        }
        if (state->config.confidence_threshold < 0.5) {
            state->config.confidence_threshold = 0.5;
        }
        if (state->config.temporal_weight > 2.0) {
            state->config.temporal_weight = 2.0;
        }
        if (state->config.temporal_weight < 0.5) {
            state->config.temporal_weight = 0.5;
        }
    }

    return true;
}

const ErrorPrediction* get_prediction(const PredictionState* state,
                                    size_t index) {
    if (!state || index >= state->num_predictions) {
        return NULL;
    }
    return &state->predictions[index];
}

double get_prediction_success_rate(const PredictionState* state) {
    return state ? state->success_rate : 0.0;
}

// Helper function implementations
static double calculate_prediction_confidence(const ErrorPattern* pattern,
                                           const ErrorCorrelation* correlation,
                                           const PredictionConfig* config) {
    if (!pattern || !correlation || !config) {
        return 0.0;
    }

    // Base confidence from pattern weight with hardware-specific adjustment
    double hw_factor = get_hardware_reliability_factor();
    double confidence = pattern->weight * hw_factor;

    // Enhanced correlation strength calculation
    double spatial_conf = correlation->spatial_correlation * 
                         get_spatial_reliability_factor();
    double temporal_conf = correlation->temporal_correlation *
                          get_temporal_reliability_factor();
    double cross_conf = correlation->cross_correlation *
                       get_cross_correlation_factor();
                       
    confidence *= (spatial_conf + temporal_conf + cross_conf) / 3.0;

    // Apply adaptive temporal weighting
    double elapsed_time = get_current_timestamp() - pattern->timing.last_seen;
    double decay_rate = get_error_decay_rate();
    double time_factor = exp(-config->temporal_weight * elapsed_time * decay_rate);
    confidence *= time_factor;

    // Adjust based on pattern frequency and stability
    double freq_factor = pattern->timing.frequency * 
                        get_frequency_stability_factor();
    confidence *= freq_factor;

    // Apply hardware-specific corrections
    confidence *= get_measurement_fidelity();
    confidence *= get_gate_fidelity();

    // Normalize to [0,1] range with minimum confidence threshold
    if (confidence > 1.0) {
        confidence = 1.0;
    } else if (confidence < config->min_confidence) {
        confidence = 0.0;
    }

    return confidence;
}

static bool validate_prediction(const ErrorPrediction* prediction,
                              const quantum_state* state) {
    if (!prediction || !state) {
        return false;
    }

    // Get hardware-specific error threshold
    double error_threshold = get_hardware_error_threshold();
    
    // Calculate error weight with spatial context
    double center_weight = calculate_error_weight(state,
                                                prediction->x,
                                                prediction->y,
                                                prediction->z);
                                                
    // Check neighboring sites for error spread
    double neighbor_weights[6] = {0.0};
    get_neighbor_error_weights(state, prediction->x, prediction->y, prediction->z,
                             neighbor_weights);
                             
    // Calculate spread factor
    double spread_factor = 0.0;
    for (int i = 0; i < 6; i++) {
        spread_factor += neighbor_weights[i] * get_spread_coefficient(i);
    }
    
    // Combine central weight and spread
    double total_weight = center_weight + spread_factor * get_spread_factor();
    
    // Apply hardware-specific corrections
    total_weight *= get_measurement_correction_factor();
    
    // Compare with dynamic threshold
    double dynamic_threshold = error_threshold * 
                             (1.0 - get_noise_factor()) *
                             get_confidence_scaling();
                             
    return total_weight > dynamic_threshold;
}

static void update_prediction_history(PredictionHistory* history,
                                    const ErrorPrediction* prediction,
                                    bool was_correct) {
    if (!history || !prediction) {
        return;
    }

    // Store prediction details
    history->prediction = *prediction;
    history->was_correct = was_correct;
    history->timestamp = prediction->timestamp;
    
    // Calculate and store confidence delta
    history->confidence_delta = was_correct ? 
        (1.0 - prediction->confidence) :  // Increase if correct
        (-prediction->confidence);        // Decrease if wrong
        
    // Store hardware state
    history->hardware_state.measurement_fidelity = get_measurement_fidelity();
    history->hardware_state.gate_fidelity = get_gate_fidelity();
    history->hardware_state.noise_level = get_noise_level();
    
    // Update error statistics
    history->error_stats.detection_latency = 
        get_current_timestamp() - prediction->timestamp;
    history->error_stats.correction_overhead = 
        measure_correction_overhead();
    history->error_stats.prediction_accuracy = 
        calculate_prediction_accuracy(prediction);
        
    // Trigger fast feedback if needed
    if (should_trigger_feedback(history)) {
        trigger_prediction_feedback(history);
    }
}

static size_t get_current_timestamp(void) {
    static size_t timestamp = 0;
    return timestamp++;
}
