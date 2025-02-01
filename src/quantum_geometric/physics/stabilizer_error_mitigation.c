/**
 * @file stabilizer_error_mitigation.c
 * @brief Implementation of error mitigation for stabilizer measurements with hardware optimization
 */

#include "quantum_geometric/physics/stabilizer_error_mitigation.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/physics/quantum_state_operations.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Forward declarations
static bool initialize_mitigation_cache(MitigationCache* cache,
                                     const MitigationConfig* config);
static void cleanup_mitigation_cache(MitigationCache* cache);
static bool apply_readout_correction(quantum_state* state,
                                   const MitigationCache* cache,
                                   const HardwareProfile* hw_profile);
static bool update_error_model(MitigationCache* cache,
                             const measurement_result* results,
                             size_t num_results,
                             const HardwareProfile* hw_profile);
static bool update_calibration_matrix(MitigationCache* cache,
                                    const measurement_result* results,
                                    size_t num_results,
                                    const HardwareProfile* hw_profile);
static double calculate_confidence_weight(const measurement_result* result,
                                       const HardwareProfile* hw_profile);

bool init_error_mitigation(MitigationState* state,
                          const MitigationConfig* config) {
    if (!state || !config) {
        return false;
    }

    // Initialize state
    memset(state, 0, sizeof(MitigationState));
    memcpy(&state->config, config, sizeof(MitigationConfig));

    // Initialize cache
    state->cache = malloc(sizeof(MitigationCache));
    if (!state->cache || !initialize_mitigation_cache(state->cache, config)) {
        cleanup_error_mitigation(state);
        return false;
    }

    // Initialize metrics
    state->total_corrections = 0;
    state->success_rate = 1.0;
    state->confidence_level = 1.0;
    state->last_update_time = 0;

    return true;
}

void cleanup_error_mitigation(MitigationState* state) {
    if (state) {
        if (state->cache) {
            cleanup_mitigation_cache(state->cache);
            free(state->cache);
        }
        memset(state, 0, sizeof(MitigationState));
    }
}

bool mitigate_measurement_errors(MitigationState* state,
                               quantum_state* qstate,
                               const measurement_result* results,
                               size_t num_results,
                               const HardwareProfile* hw_profile) {
    if (!state || !qstate || !results || num_results == 0 || !hw_profile) {
        return false;
    }

    // Check if calibration matrix needs update
    uint64_t current_time = get_current_timestamp();
    bool needs_update = (current_time - state->last_update_time) > 
                       state->config.calibration_interval;

    // Update error model with new measurements
    if (needs_update) {
        if (!update_error_model(state->cache, results, num_results, hw_profile)) {
            return false;
        }
        
        // Update calibration matrix
        if (!update_calibration_matrix(state->cache, results, num_results, hw_profile)) {
            return false;
        }
        
        state->last_update_time = current_time;
    }

    // Apply readout error correction
    if (!apply_readout_correction(qstate, state->cache, hw_profile)) {
        return false;
    }

    // Update metrics
    state->total_corrections++;
    update_mitigation_metrics(state);

    return true;
}

bool update_mitigation_metrics(MitigationState* state) {
    if (!state || !state->cache) {
        return false;
    }

    // Calculate success rate from recent corrections with confidence weighting
    double weighted_success = 0.0;
    double total_weight = 0.0;
    
    for (size_t i = 0; i < state->cache->history_size; i++) {
        if (state->cache->correction_history[i].was_successful) {
            weighted_success += state->cache->correction_history[i].confidence;
        }
        total_weight += state->cache->correction_history[i].confidence;
    }
    
    state->success_rate = total_weight > 0.0 ? 
                         weighted_success / total_weight : 0.0;

    // Calculate confidence level using error variances and hardware factors
    double total_variance = 0.0;
    double hw_factor = get_hardware_reliability_factor();
    
    for (size_t i = 0; i < state->cache->num_qubits; i++) {
        double qubit_reliability = get_qubit_reliability(i);
        total_variance += state->cache->error_variances[i] * 
                         (1.0 - qubit_reliability);
    }
    
    state->confidence_level = (1.0 - (total_variance / state->cache->num_qubits)) * 
                            hw_factor;

    return true;
}

static bool initialize_mitigation_cache(MitigationCache* cache,
                                      const MitigationConfig* config) {
    if (!cache || !config) {
        return false;
    }

    // Allocate arrays
    cache->num_qubits = config->num_qubits;
    cache->history_size = config->history_length;
    
    cache->error_rates = calloc(config->num_qubits, sizeof(double));
    cache->error_variances = calloc(config->num_qubits, sizeof(double));
    cache->correction_history = calloc(config->history_length, 
                                     sizeof(CorrectionHistoryEntry));
    cache->calibration_matrix = calloc(config->num_qubits * 4, sizeof(double));
    cache->confidence_weights = calloc(config->num_qubits, sizeof(double));

    if (!cache->error_rates || !cache->error_variances ||
        !cache->correction_history || !cache->calibration_matrix ||
        !cache->confidence_weights) {
        cleanup_mitigation_cache(cache);
        return false;
    }

    // Initialize calibration matrix to identity with confidence
    for (size_t i = 0; i < config->num_qubits; i++) {
        // P(0|0) and P(1|1)
        cache->calibration_matrix[i * 4] = 1.0;
        cache->calibration_matrix[i * 4 + 3] = 1.0;
        
        // P(0|1) and P(1|0)
        cache->calibration_matrix[i * 4 + 1] = 0.0;
        cache->calibration_matrix[i * 4 + 2] = 0.0;
        
        // Initial confidence
        cache->confidence_weights[i] = 1.0;
    }

    return true;
}

static void cleanup_mitigation_cache(MitigationCache* cache) {
    if (cache) {
        free(cache->error_rates);
        free(cache->error_variances);
        free(cache->correction_history);
        free(cache->calibration_matrix);
        free(cache->confidence_weights);
        memset(cache, 0, sizeof(MitigationCache));
    }
}

static bool apply_readout_correction(quantum_state* state,
                                   const MitigationCache* cache,
                                   const HardwareProfile* hw_profile) {
    if (!state || !cache || !hw_profile) {
        return false;
    }

    // Apply calibration matrix to measurement results with hardware factors
    for (size_t i = 0; i < cache->num_qubits; i++) {
        // Get hardware-specific factors
        double qubit_reliability = get_qubit_reliability(i);
        double measurement_fidelity = get_measurement_fidelity(i);
        double noise_factor = get_noise_factor();
        
        // Skip if reliability is too low
        if (qubit_reliability < hw_profile->min_reliability_threshold) {
            continue;
        }

        // Get calibration matrix elements
        double p00 = cache->calibration_matrix[i * 4];     // P(0|0)
        double p01 = cache->calibration_matrix[i * 4 + 1]; // P(0|1)
        double p10 = cache->calibration_matrix[i * 4 + 2]; // P(1|0)
        double p11 = cache->calibration_matrix[i * 4 + 3]; // P(1|1)
        
        // Apply hardware corrections
        p00 *= qubit_reliability * measurement_fidelity * (1.0 - noise_factor);
        p11 *= qubit_reliability * measurement_fidelity * (1.0 - noise_factor);
        p01 *= (1.0 - qubit_reliability * measurement_fidelity);
        p10 *= (1.0 - qubit_reliability * measurement_fidelity);

        // Get current amplitudes
        double p0 = state->amplitudes[i * 2];
        double p1 = state->amplitudes[i * 2 + 1];
        
        // Apply correction with confidence weighting
        double confidence = cache->confidence_weights[i];
        state->amplitudes[i * 2] = confidence * (p0 * p00 + p1 * p01) + 
                                  (1.0 - confidence) * p0;
        state->amplitudes[i * 2 + 1] = confidence * (p0 * p10 + p1 * p11) + 
                                      (1.0 - confidence) * p1;
    }

    return true;
}

static bool update_error_model(MitigationCache* cache,
                             const measurement_result* results,
                             size_t num_results,
                             const HardwareProfile* hw_profile) {
    if (!cache || !results || num_results == 0 || !hw_profile) {
        return false;
    }

    // Update error rates and variances with hardware factors
    for (size_t i = 0; i < cache->num_qubits; i++) {
        size_t error_count = 0;
        double weighted_error_sum = 0.0;
        double total_weight = 0.0;

        // Calculate weighted error rate
        for (size_t j = 0; j < num_results; j++) {
            if (results[j].qubit_index == i) {
                double confidence = calculate_confidence_weight(&results[j], 
                                                             hw_profile);
                if (results[j].had_error) {
                    weighted_error_sum += confidence;
                    error_count++;
                }
                total_weight += confidence;
            }
        }

        // Update error rate
        double old_rate = cache->error_rates[i];
        double new_rate = total_weight > 0.0 ? 
                         weighted_error_sum / total_weight : 0.0;
        
        // Apply exponential moving average
        double alpha = hw_profile->error_rate_learning_rate;
        cache->error_rates[i] = alpha * new_rate + (1.0 - alpha) * old_rate;

        // Update error variance
        double diff = new_rate - old_rate;
        cache->error_variances[i] = sqrt(diff * diff);
        
        // Update confidence weight
        cache->confidence_weights[i] = calculate_confidence_weight(&results[0], 
                                                                hw_profile);
    }

    return true;
}

static bool update_calibration_matrix(MitigationCache* cache,
                                    const measurement_result* results,
                                    size_t num_results,
                                    const HardwareProfile* hw_profile) {
    if (!cache || !results || num_results == 0 || !hw_profile) {
        return false;
    }

    // Update calibration matrix elements for each qubit
    for (size_t i = 0; i < cache->num_qubits; i++) {
        // Count state preparation and measurement results
        size_t count_00 = 0, count_01 = 0, count_10 = 0, count_11 = 0;
        double weight_00 = 0.0, weight_01 = 0.0, weight_10 = 0.0, weight_11 = 0.0;
        
        for (size_t j = 0; j < num_results; j++) {
            if (results[j].qubit_index == i) {
                double confidence = calculate_confidence_weight(&results[j], 
                                                             hw_profile);
                
                // Classify measurement result
                if (results[j].prepared_state == 0) {
                    if (results[j].measured_value == 0) {
                        count_00++;
                        weight_00 += confidence;
                    } else {
                        count_01++;
                        weight_01 += confidence;
                    }
                } else {
                    if (results[j].measured_value == 0) {
                        count_10++;
                        weight_10 += confidence;
                    } else {
                        count_11++;
                        weight_11 += confidence;
                    }
                }
            }
        }
        
        // Calculate new calibration matrix elements
        double total_0 = count_00 + count_01;
        double total_1 = count_10 + count_11;
        
        if (total_0 > 0) {
            cache->calibration_matrix[i * 4] = weight_00 / total_0;     // P(0|0)
            cache->calibration_matrix[i * 4 + 1] = weight_01 / total_0; // P(0|1)
        }
        
        if (total_1 > 0) {
            cache->calibration_matrix[i * 4 + 2] = weight_10 / total_1; // P(1|0)
            cache->calibration_matrix[i * 4 + 3] = weight_11 / total_1; // P(1|1)
        }
    }

    return true;
}

static double calculate_confidence_weight(const measurement_result* result,
                                       const HardwareProfile* hw_profile) {
    if (!result || !hw_profile) {
        return 0.0;
    }

    // Get hardware-specific factors
    double qubit_reliability = get_qubit_reliability(result->qubit_index);
    double measurement_fidelity = get_measurement_fidelity(result->qubit_index);
    double coherence_factor = get_coherence_factor(result->qubit_index);
    double noise_level = get_noise_factor();
    
    // Calculate base confidence
    double confidence = qubit_reliability * measurement_fidelity * 
                       coherence_factor * (1.0 - noise_level);
    
    // Adjust based on measurement stability
    confidence *= get_measurement_stability(result->qubit_index);
    
    // Apply hardware-specific scaling
    confidence *= hw_profile->confidence_scale_factor;
    
    // Clamp to valid range
    if (confidence > 1.0) confidence = 1.0;
    if (confidence < 0.0) confidence = 0.0;
    
    return confidence;
}
