/**
 * @file z_stabilizer_operations.c
 * @brief Implementation of Z-stabilizer measurement and optimization operations
 */

#include "quantum_geometric/physics/z_stabilizer_operations.h"
#include "quantum_geometric/core/quantum_complex.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Internal helper functions
static bool allocate_z_stabilizer_resources(ZStabilizerState* state, size_t num_qubits);
static void cleanup_z_stabilizer_resources(ZStabilizerState* state);
static bool validate_z_stabilizer_config(const ZStabilizerConfig* config);
static bool validate_hardware_config(const ZHardwareConfig* hardware);
static double calculate_phase_correlation(const ZStabilizerState* state, size_t idx1, size_t idx2);
static void apply_phase_echo_sequence(ZStabilizerState* state, size_t x, size_t y);
static bool update_phase_tracking(ZStabilizerState* state, size_t x, size_t y, double measurement);

ZStabilizerState* init_z_stabilizer_measurement(
    const ZStabilizerConfig* config,
    const ZHardwareConfig* hardware
) {
    if (!validate_z_stabilizer_config(config) || !validate_hardware_config(hardware)) {
        return NULL;
    }

    ZStabilizerState* state = malloc(sizeof(ZStabilizerState));
    if (!state) {
        return NULL;
    }

    // Copy configuration
    memcpy(&state->config, config, sizeof(ZStabilizerConfig));

    // Calculate total number of qubits
    size_t num_qubits = config->repetition_count * config->repetition_count;

    // Allocate resources
    if (!allocate_z_stabilizer_resources(state, num_qubits)) {
        free(state);
        return NULL;
    }

    // Initialize arrays
    for (size_t i = 0; i < num_qubits; i++) {
        state->phase_correlations[i] = 1.0;  // Start with perfect correlation
        state->measurement_confidences[i] = 1.0;  // Start with perfect confidence
        state->stabilizer_values[i] = 1.0;  // Start in +1 eigenstate
    }

    // Initialize metrics
    state->history_size = 0;
    state->phase_error_rate = 0.0;

    // Apply hardware-specific optimizations if enabled
    if (config->enable_z_optimization) {
        apply_hardware_z_optimizations(state, hardware);
    }

    return state;
}

qgt_error_t z_measurement_create(z_measurement_t** measurement) {
    if (!measurement) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    *measurement = calloc(1, sizeof(z_measurement_t));
    if (!*measurement) {
        return QGT_ERROR_NO_MEMORY;
    }

    (*measurement)->value = 1;  // Initialize to +1 eigenstate
    (*measurement)->confidence = 1.0;
    (*measurement)->error_rate = 0.0;
    (*measurement)->needs_correction = false;
    (*measurement)->auxiliary_data = NULL;

    return QGT_SUCCESS;
}

void z_measurement_destroy(z_measurement_t* measurement) {
    if (!measurement) {
        return;
    }
    free(measurement->auxiliary_data);
    free(measurement);
}

qgt_error_t z_stabilizer_measure(z_measurement_t* measurement,
                                const quantum_stabilizer_t* stabilizer,
                                const quantum_geometric_state_t* state) {
    if (!measurement || !stabilizer || !state) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // Perform Z-basis measurement
    double result = 1.0;
    double confidence = 1.0;
    
    // Apply stabilizer operator
    for (size_t i = 0; i < stabilizer->num_terms; i++) {
        const quantum_pauli_t* pauli = &stabilizer->terms[i];
        if (pauli->type != PAULI_Z) {
            continue;
        }
        
        // Get qubit state
        size_t idx = pauli->qubit_index;
        if (idx >= state->num_qubits) {
            return QGT_ERROR_INVALID_STATE;
        }
        
        // Multiply measurement result by qubit Z expectation value
        result *= state->coordinates[idx].real;
        confidence *= (1.0 - state->error_rates[idx]);
    }

    // Update measurement result
    measurement->value = (result > 0) ? 1 : -1;
    measurement->confidence = confidence;
    measurement->error_rate = 1.0 - confidence;
    measurement->needs_correction = (measurement->value == -1);

    return QGT_SUCCESS;
}

qgt_error_t z_measurement_has_error(bool* has_error,
                                   const z_measurement_t* measurement,
                                   double threshold) {
    if (!has_error || !measurement) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    if (threshold < 0.0 || threshold > 1.0) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    *has_error = (measurement->error_rate > threshold);
    return QGT_SUCCESS;
}

qgt_error_t z_measurement_reliability(double* reliability,
                                    const z_measurement_t* measurement) {
    if (!reliability || !measurement) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    *reliability = measurement->confidence;
    return QGT_SUCCESS;
}

qgt_error_t z_measurement_compare(bool* equal,
                                const z_measurement_t* measurement1,
                                const z_measurement_t* measurement2,
                                double tolerance) {
    if (!equal || !measurement1 || !measurement2) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    if (tolerance < 0.0) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    *equal = (fabs(measurement1->value - measurement2->value) < tolerance);
    return QGT_SUCCESS;
}

qgt_error_t z_measurement_validate(const z_measurement_t* measurement) {
    if (!measurement) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    if (measurement->value != 1 && measurement->value != -1) {
        return QGT_ERROR_INVALID_STATE;
    }

    if (measurement->confidence < 0.0 || measurement->confidence > 1.0) {
        return QGT_ERROR_INVALID_STATE;
    }

    if (measurement->error_rate < 0.0 || measurement->error_rate > 1.0) {
        return QGT_ERROR_INVALID_STATE;
    }

    return QGT_SUCCESS;
}

qgt_error_t z_stabilizer_correct(quantum_geometric_state_t* state,
                                const z_measurement_t* measurement,
                                const quantum_stabilizer_t* stabilizer) {
    if (!state || !measurement || !stabilizer) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    if (!measurement->needs_correction) {
        return QGT_SUCCESS;
    }

    // Apply correction operations
    for (size_t i = 0; i < stabilizer->num_terms; i++) {
        const quantum_pauli_t* pauli = &stabilizer->terms[i];
        if (pauli->type != PAULI_Z) {
            continue;
        }
        
        size_t idx = pauli->qubit_index;
        if (idx >= state->num_qubits) {
            return QGT_ERROR_INVALID_STATE;
        }
        
        // Apply Z correction
        state->coordinates[idx].real *= -1.0;
    }

    return QGT_SUCCESS;
}

qgt_error_t z_stabilizer_correlation(double* correlation,
                                   const quantum_stabilizer_t* stabilizer1,
                                   const quantum_stabilizer_t* stabilizer2,
                                   const quantum_geometric_state_t* state) {
    if (!correlation || !stabilizer1 || !stabilizer2 || !state) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // Calculate correlation between stabilizer measurements
    double corr = 0.0;
    size_t overlap_count = 0;

    for (size_t i = 0; i < stabilizer1->num_terms; i++) {
        for (size_t j = 0; j < stabilizer2->num_terms; j++) {
            const quantum_pauli_t* p1 = &stabilizer1->terms[i];
            const quantum_pauli_t* p2 = &stabilizer2->terms[j];
            
            if (p1->qubit_index == p2->qubit_index && 
                p1->type == PAULI_Z && p2->type == PAULI_Z) {
                corr += state->coordinates[p1->qubit_index].real;
                overlap_count++;
            }
        }
    }

    *correlation = overlap_count > 0 ? corr / overlap_count : 0.0;
    return QGT_SUCCESS;
}

qgt_error_t z_stabilizer_commute(bool* commute,
                                const quantum_stabilizer_t* stabilizer1,
                                const quantum_stabilizer_t* stabilizer2) {
    if (!commute || !stabilizer1 || !stabilizer2) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // Z stabilizers always commute with each other
    *commute = true;
    return QGT_SUCCESS;
}

qgt_error_t z_stabilizer_weight(size_t* weight,
                               const quantum_stabilizer_t* stabilizer) {
    if (!weight || !stabilizer) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // Count number of Z terms
    size_t z_count = 0;
    for (size_t i = 0; i < stabilizer->num_terms; i++) {
        if (stabilizer->terms[i].type == PAULI_Z) {
            z_count++;
        }
    }

    *weight = z_count;
    return QGT_SUCCESS;
}

qgt_error_t z_stabilizer_validate(const quantum_stabilizer_t* stabilizer) {
    if (!stabilizer) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    if (stabilizer->num_terms == 0) {
        return QGT_ERROR_INVALID_STATE;
    }

    // Verify all terms are valid
    for (size_t i = 0; i < stabilizer->num_terms; i++) {
        const quantum_pauli_t* pauli = &stabilizer->terms[i];
        
        if (pauli->type != PAULI_Z) {
            return QGT_ERROR_INVALID_STATE;
        }
    }

    return QGT_SUCCESS;
}

bool apply_z_error_mitigation_sequence(
    ZStabilizerState* state,
    size_t x,
    size_t y
) {
    if (!state) return false;

    // Apply phase echo sequence for phase error mitigation
    apply_phase_echo_sequence(state, x, y);

    // Update phase tracking
    size_t idx = y * state->config.repetition_count + x;
    if (idx >= state->config.repetition_count * state->config.repetition_count) {
        return false;
    }

    // Apply dynamic phase correction if enabled
    if (state->config.dynamic_phase_correction) {
        double current_phase = state->phase_correlations[idx];
        double correction = -atan2(sin(current_phase), cos(current_phase));
        state->phase_correlations[idx] *= exp(COMPLEX_FLOAT_I.imag * correction);
    }

    return true;
}

double get_z_stabilizer_correlation(
    const ZStabilizerState* state,
    size_t x1,
    size_t y1,
    size_t x2,
    size_t y2
) {
    if (!state) return 0.0;

    size_t idx1 = y1 * state->config.repetition_count + x1;
    size_t idx2 = y2 * state->config.repetition_count + x2;
    
    return calculate_phase_correlation(state, idx1, idx2);
}

void apply_z_measurement_correction(
    ZStabilizerState* state,
    size_t x,
    size_t y,
    double* result
) {
    if (!state || !result) return;

    size_t idx = y * state->config.repetition_count + x;
    if (idx >= state->config.repetition_count * state->config.repetition_count) {
        return;
    }

    // Apply confidence-weighted correction
    double confidence = state->measurement_confidences[idx];
    if (confidence < state->config.confidence_threshold) {
        *result = 0.0;  // Zero out low-confidence measurements
        return;
    }

    // Apply phase correction based on correlation history
    double phase_factor = state->phase_correlations[idx];
    *result *= phase_factor * confidence;
}

bool measure_z_stabilizers_parallel(
    ZStabilizerState* state,
    const size_t* qubit_coords,
    size_t num_qubits,
    double* results
) {
    if (!state || !qubit_coords || !results || num_qubits == 0) {
        return false;
    }

    // Perform parallel measurements while avoiding crosstalk
    for (size_t i = 0; i < num_qubits; i++) {
        size_t x = qubit_coords[2 * i];
        size_t y = qubit_coords[2 * i + 1];

        // Apply error mitigation before measurement
        if (!apply_z_error_mitigation_sequence(state, x, y)) {
            return false;
        }

        // Simulate measurement (in real hardware this would be a Z-basis measurement)
        results[i] = state->stabilizer_values[y * state->config.repetition_count + x];

        // Apply measurement correction
        apply_z_measurement_correction(state, x, y, &results[i]);

        // Update phase tracking
        update_phase_tracking(state, x, y, results[i]);
    }

    return true;
}

bool update_z_measurement_history(
    ZStabilizerState* state,
    size_t x,
    size_t y,
    double result
) {
    if (!state) return false;

    size_t idx = y * state->config.repetition_count + x;
    if (idx >= state->config.repetition_count * state->config.repetition_count) {
        return false;
    }

    // Update history if capacity allows
    if (state->history_size < state->config.history_capacity) {
        state->measurement_history[state->history_size++] = result;
    } else {
        // Shift history and add new result at end
        memmove(state->measurement_history, 
                state->measurement_history + 1,
                (state->config.history_capacity - 1) * sizeof(double));
        state->measurement_history[state->config.history_capacity - 1] = result;
    }

    return true;
}

ZStabilizerResults get_z_stabilizer_results(const ZStabilizerState* state) {
    ZStabilizerResults results = {0};
    if (!state) return results;

    // Calculate average fidelity
    double total_fidelity = 0.0;
    for (size_t i = 0; i < state->config.repetition_count * state->config.repetition_count; i++) {
        total_fidelity += state->measurement_confidences[i];
    }
    results.average_fidelity = total_fidelity / 
        (state->config.repetition_count * state->config.repetition_count);

    // Calculate phase stability
    double total_phase_stability = 0.0;
    for (size_t i = 0; i < state->config.repetition_count * state->config.repetition_count; i++) {
        total_phase_stability += fabs(state->phase_correlations[i]);
    }
    results.phase_stability = total_phase_stability /
        (state->config.repetition_count * state->config.repetition_count);

    // Calculate correlation strength
    results.correlation_strength = 0.0;
    size_t correlation_count = 0;
    for (size_t i = 0; i < state->config.repetition_count; i++) {
        for (size_t j = i + 1; j < state->config.repetition_count; j++) {
            results.correlation_strength += 
                calculate_phase_correlation(state, i, j);
            correlation_count++;
        }
    }
    if (correlation_count > 0) {
        results.correlation_strength /= correlation_count;
    }

    results.measurement_count = state->history_size;
    results.error_suppression_factor = 1.0 - state->phase_error_rate;

    return results;
}

double get_z_error_rate(const ZStabilizerState* state) {
    return state ? state->phase_error_rate : 1.0;
}

bool optimize_z_measurement_sequence(
    ZStabilizerState* state,
    const ZHardwareConfig* hardware
) {
    if (!state || !hardware) return false;

    // Optimize echo sequence length based on decoherence time
    size_t optimal_length = (size_t)(hardware->phase_calibration * 
                                   hardware->measurement_fidelity * 10.0);
    optimal_length = optimal_length < 2 ? 2 : optimal_length;
    optimal_length = optimal_length > 20 ? 20 : optimal_length;

    state->config.echo_sequence_length = optimal_length;

    // Update phase calibration based on measurement history
    if (state->history_size > 0) {
        double avg_phase_error = 0.0;
        for (size_t i = 0; i < state->history_size; i++) {
            avg_phase_error += fabs(1.0 - fabs(state->measurement_history[i]));
        }
        avg_phase_error /= state->history_size;
        
        // Update phase calibration factor
        state->config.phase_calibration *= (1.0 - avg_phase_error);
    }

    return true;
}

bool apply_hardware_z_optimizations(
    ZStabilizerState* state,
    const ZHardwareConfig* hardware
) {
    if (!state || !hardware) return false;

    // Apply hardware-specific optimizations
    if (hardware->dynamic_phase_correction) {
        // Implement dynamic phase tracking
        for (size_t i = 0; i < state->config.repetition_count * state->config.repetition_count; i++) {
            double current_phase = state->phase_correlations[i];
            state->phase_correlations[i] *= exp(COMPLEX_FLOAT_I.imag * hardware->phase_calibration);
        }
    }

    // Optimize measurement fidelity based on hardware capabilities
    if (hardware->z_gate_fidelity > 0.9) {
        state->config.confidence_threshold *= hardware->z_gate_fidelity;
    }

    return true;
}

// Internal helper function implementations
static bool allocate_z_stabilizer_resources(ZStabilizerState* state, size_t num_qubits) {
    state->phase_correlations = calloc(num_qubits, sizeof(double));
    state->measurement_confidences = calloc(num_qubits, sizeof(double));
    state->measurement_history = calloc(state->config.history_capacity, sizeof(double));
    state->stabilizer_values = calloc(num_qubits, sizeof(double));

    if (!state->phase_correlations || !state->measurement_confidences ||
        !state->measurement_history || !state->stabilizer_values) {
        cleanup_z_stabilizer_resources(state);
        return false;
    }

    return true;
}

static void cleanup_z_stabilizer_resources(ZStabilizerState* state) {
    if (!state) return;
    free(state->phase_correlations);
    free(state->measurement_confidences);
    free(state->measurement_history);
    free(state->stabilizer_values);
}

static bool validate_z_stabilizer_config(const ZStabilizerConfig* config) {
    if (!config) return false;
    if (config->repetition_count == 0) return false;
    if (config->error_threshold < 0.0 || config->error_threshold > 1.0) return false;
    if (config->confidence_threshold < 0.0 || config->confidence_threshold > 1.0) return false;
    if (config->history_capacity == 0) return false;
    if (config->phase_calibration <= 0.0) return false;
    if (config->echo_sequence_length == 0) return false;
    // Note: enable_z_optimization is a boolean flag and doesn't need validation
    return true;
}

void cleanup_z_stabilizer_measurement(ZStabilizerState* state) {
    if (!state) return;
    cleanup_z_stabilizer_resources(state);
    free(state);
}

static bool validate_hardware_config(const ZHardwareConfig* hardware) {
    if (!hardware) return false;
    if (hardware->phase_calibration <= 0.0) return false;
    if (hardware->z_gate_fidelity <= 0.0 || hardware->z_gate_fidelity > 1.0) return false;
    if (hardware->measurement_fidelity <= 0.0 || hardware->measurement_fidelity > 1.0) return false;
    if (hardware->echo_sequence_length == 0) return false;
    return true;
}

static double calculate_phase_correlation(const ZStabilizerState* state, size_t idx1, size_t idx2) {
    if (!state) return 0.0;
    if (idx1 >= state->config.repetition_count * state->config.repetition_count ||
        idx2 >= state->config.repetition_count * state->config.repetition_count) {
        return 0.0;
    }

    // Calculate correlation between phase values
    double phase1 = state->phase_correlations[idx1];
    double phase2 = state->phase_correlations[idx2];
    
    return cos(phase1 - phase2);
}

static void apply_phase_echo_sequence(ZStabilizerState* state, size_t x, size_t y) {
    if (!state) return;

    size_t idx = y * state->config.repetition_count + x;
    if (idx >= state->config.repetition_count * state->config.repetition_count) {
        return;
    }

    // Apply echo sequence to mitigate phase errors
    for (size_t i = 0; i < state->config.echo_sequence_length; i++) {
        // Simulate Z-gate application
        state->phase_correlations[idx] *= -1.0;
        
        // Add small random phase error
        double phase_error = (rand() / (double)RAND_MAX - 0.5) * 0.1;
        state->phase_correlations[idx] *= exp(COMPLEX_FLOAT_I.imag * phase_error);
    }
}

static bool update_phase_tracking(ZStabilizerState* state, size_t x, size_t y, double measurement) {
    if (!state) return false;

    size_t idx = y * state->config.repetition_count + x;
    if (idx >= state->config.repetition_count * state->config.repetition_count) {
        return false;
    }

    // Update phase correlation based on measurement result
    double expected_phase = state->phase_correlations[idx];
    double measured_phase = acos(measurement);
    double phase_difference = fabs(expected_phase - measured_phase);
    
    // Update confidence based on phase difference
    state->measurement_confidences[idx] *= exp(-phase_difference);
    
    // Update error rate estimate
    state->phase_error_rate = (state->phase_error_rate * 0.9 + phase_difference * 0.1);

    return true;
}
