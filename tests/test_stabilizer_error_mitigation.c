/**
 * @file test_stabilizer_error_mitigation.c
 * @brief Tests for stabilizer error mitigation system with hardware optimization
 */

#include "quantum_geometric/physics/stabilizer_error_mitigation.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/hardware/quantum_hardware_optimization.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

// Test helper functions
static quantum_state* create_test_state(size_t num_qubits) {
    quantum_state* state = malloc(sizeof(quantum_state));
    state->num_qubits = num_qubits;
    state->amplitudes = calloc(num_qubits * 2, sizeof(double));
    // Initialize to |0⟩ state
    for (size_t i = 0; i < num_qubits; i++) {
        state->amplitudes[i * 2] = 1.0;
    }
    return state;
}

static void cleanup_test_state(quantum_state* state) {
    if (state) {
        free(state->amplitudes);
        free(state);
    }
}

static measurement_result* create_test_results(size_t num_results, 
                                            size_t num_qubits,
                                            double error_rate,
                                            const HardwareProfile* hw_profile) {
    measurement_result* results = malloc(num_results * sizeof(measurement_result));
    for (size_t i = 0; i < num_results; i++) {
        results[i].qubit_index = i % num_qubits;
        results[i].measured_value = (i % 2) ? 1.0 : 0.0;
        results[i].had_error = ((double)rand() / RAND_MAX) < error_rate;
        results[i].error_prob = error_rate;
        results[i].confidence = calculate_measurement_confidence(hw_profile, i);
        results[i].hardware_factor = get_hardware_reliability_factor(hw_profile, i);
        results[i].prepared_state = 0;
        results[i].measured_value = results[i].had_error ? 1 : 0;
    }
    return results;
}

static HardwareProfile* create_test_profile() {
    HardwareProfile* profile = malloc(sizeof(HardwareProfile));
    profile->min_confidence_threshold = 0.8;
    profile->learning_rate = 0.1;
    profile->spatial_scale = 2.0;
    profile->pattern_scale_factor = 1.5;
    profile->noise_scale = 0.05;
    profile->phase_calibration = 0.99;
    profile->gate_fidelity = 0.995;
    profile->measurement_fidelity = 0.99;
    return profile;
}

// Test cases
static void test_initialization() {
    printf("Testing initialization...\n");

    MitigationConfig config = {
        .num_qubits = 4,
        .history_length = 100,
        .calibration_interval = 1000000,  // 1ms
        .error_threshold = 0.01,
        .confidence_threshold = 0.9,
        .pattern_threshold = 0.1,
        .min_pattern_occurrences = 3,
        .max_parallel_ops = 8,
        .parallel_group_size = 4,
        .history_window = 100,
        .weight_scale_factor = 1.0,
        .detection_threshold = 0.05
    };

    MitigationState state;
    bool success = init_error_mitigation(&state, &config);
    assert(success && "Failed to initialize mitigation state");
    assert(state.cache != NULL && "Cache not allocated");
    assert(state.cache->error_rates != NULL && "Error rates not allocated");
    assert(state.total_syndromes == 0 && "Initial syndromes not zero");
    assert(fabs(state.confidence_level - 1.0) < 1e-6 && "Initial confidence not 1.0");

    cleanup_error_mitigation(&state);
    printf("Initialization test passed\n");
}

static void test_hardware_optimized_mitigation() {
    printf("Testing hardware-optimized error mitigation...\n");

    // Setup
    size_t num_qubits = 4;
    size_t num_results = 20;
    double error_rate = 0.1;

    MitigationConfig config = {
        .num_qubits = num_qubits,
        .history_length = 100,
        .calibration_interval = 1000000,
        .error_threshold = 0.05,
        .confidence_threshold = 0.9,
        .pattern_threshold = 0.1,
        .min_pattern_occurrences = 3,
        .max_parallel_ops = 8,
        .parallel_group_size = 4,
        .history_window = 100,
        .weight_scale_factor = 1.0,
        .detection_threshold = 0.05
    };

    MitigationState state;
    bool success = init_error_mitigation(&state, &config);
    assert(success && "Failed to initialize mitigation state");

    HardwareProfile* hw_profile = create_test_profile();
    quantum_state* qstate = create_test_state(num_qubits);
    measurement_result* results = create_test_results(num_results, num_qubits, 
                                                    error_rate, hw_profile);

    // Test mitigation with hardware profile
    success = mitigate_measurement_errors(&state, qstate, results, num_results, hw_profile);
    assert(success && "Hardware-optimized error mitigation failed");
    assert(state.total_syndromes == 1 && "Syndrome count not updated");
    assert(state.confidence_level > 0.0 && state.confidence_level <= 1.0 && 
           "Invalid confidence level");

    // Verify hardware factors are applied
    for (size_t i = 0; i < num_qubits; i++) {
        assert(state.graph->hardware_factors[i] > 0.0 && 
               state.graph->hardware_factors[i] <= 1.0 &&
               "Invalid hardware factor");
        assert(state.graph->confidence_weights[i] > 0.0 &&
               state.graph->confidence_weights[i] <= 1.0 &&
               "Invalid confidence weight");
    }

    // Cleanup
    cleanup_error_mitigation(&state);
    cleanup_test_state(qstate);
    free(results);
    free(hw_profile);
    printf("Hardware-optimized error mitigation test passed\n");
}

static void test_fast_feedback() {
    printf("Testing fast feedback system...\n");

    size_t num_qubits = 4;
    size_t num_results = 10;
    double error_rate = 0.15;

    MitigationConfig config = {
        .num_qubits = num_qubits,
        .history_length = 100,
        .calibration_interval = 1000000,
        .error_threshold = 0.05,
        .confidence_threshold = 0.9,
        .pattern_threshold = 0.1,
        .min_pattern_occurrences = 3,
        .max_parallel_ops = 8,
        .parallel_group_size = 4,
        .history_window = 100,
        .weight_scale_factor = 1.0,
        .detection_threshold = 0.05
    };

    MitigationState state;
    bool success = init_error_mitigation(&state, &config);
    assert(success && "Failed to initialize mitigation state");

    HardwareProfile* hw_profile = create_test_profile();
    quantum_state* qstate = create_test_state(num_qubits);
    
    // Perform multiple measurements to test feedback
    uint64_t last_update = 0;
    for (size_t i = 0; i < 3; i++) {
        measurement_result* results = create_test_results(num_results, num_qubits,
                                                        error_rate, hw_profile);
        
        success = mitigate_measurement_errors(&state, qstate, results, num_results, hw_profile);
        assert(success && "Fast feedback mitigation failed");
        
        // Verify calibration updates
        assert(state.last_update_time >= last_update && 
               "Calibration time not updated");
        last_update = state.last_update_time;
        
        free(results);
    }

    cleanup_error_mitigation(&state);
    cleanup_test_state(qstate);
    free(hw_profile);
    printf("Fast feedback test passed\n");
}

static void test_pattern_detection() {
    printf("Testing hardware-aware pattern detection...\n");

    size_t num_qubits = 6;  // Larger lattice for better pattern detection
    size_t num_results = 30;
    double error_rate = 0.2;

    MitigationConfig config = {
        .num_qubits = num_qubits,
        .history_length = 100,
        .calibration_interval = 1000000,
        .error_threshold = 0.05,
        .confidence_threshold = 0.9,
        .pattern_threshold = 0.1,
        .min_pattern_occurrences = 3,
        .max_parallel_ops = 8,
        .parallel_group_size = 4,
        .history_window = 100,
        .weight_scale_factor = 1.0,
        .detection_threshold = 0.05
    };

    MitigationState state;
    bool success = init_error_mitigation(&state, &config);
    assert(success && "Failed to initialize mitigation state");

    HardwareProfile* hw_profile = create_test_profile();
    quantum_state* qstate = create_test_state(num_qubits);
    
    // Create results with correlated errors
    measurement_result* results = create_test_results(num_results, num_qubits,
                                                    error_rate, hw_profile);
    // Force some adjacent errors
    results[0].had_error = true;
    results[1].had_error = true;
    results[2].had_error = false;
    results[3].had_error = true;
    results[4].had_error = true;

    success = mitigate_measurement_errors(&state, qstate, results, num_results, hw_profile);
    assert(success && "Pattern detection failed");

    // Verify pattern weights
    bool found_pattern = false;
    for (size_t i = 0; i < num_qubits; i++) {
        if (state.graph->pattern_weights[i] > config.pattern_threshold) {
            found_pattern = true;
            break;
        }
    }
    assert(found_pattern && "No error patterns detected");

    cleanup_error_mitigation(&state);
    cleanup_test_state(qstate);
    free(results);
    free(hw_profile);
    printf("Pattern detection test passed\n");
}

static void test_confidence_tracking() {
    printf("Testing confidence tracking...\n");

    size_t num_qubits = 4;
    size_t num_results = 20;
    double error_rate = 0.1;

    MitigationConfig config = {
        .num_qubits = num_qubits,
        .history_length = 100,
        .calibration_interval = 1000000,
        .error_threshold = 0.05,
        .confidence_threshold = 0.9,
        .pattern_threshold = 0.1,
        .min_pattern_occurrences = 3,
        .max_parallel_ops = 8,
        .parallel_group_size = 4,
        .history_window = 100,
        .weight_scale_factor = 1.0,
        .detection_threshold = 0.05
    };

    MitigationState state;
    bool success = init_error_mitigation(&state, &config);
    assert(success && "Failed to initialize mitigation state");

    HardwareProfile* hw_profile = create_test_profile();
    quantum_state* qstate = create_test_state(num_qubits);
    measurement_result* results = create_test_results(num_results, num_qubits,
                                                    error_rate, hw_profile);

    // Test confidence tracking
    success = mitigate_measurement_errors(&state, qstate, results, num_results, hw_profile);
    assert(success && "Confidence tracking failed");

    // Verify confidence history
    for (size_t i = 0; i < num_qubits; i++) {
        assert(state.graph->vertices[i].confidence_history != NULL &&
               "Confidence history not allocated");
        assert(state.graph->confidence_weights[i] > 0.0 &&
               state.graph->confidence_weights[i] <= 1.0 &&
               "Invalid confidence weight");
    }

    cleanup_error_mitigation(&state);
    cleanup_test_state(qstate);
    free(results);
    free(hw_profile);
    printf("Confidence tracking test passed\n");
}

int main() {
    printf("Running stabilizer error mitigation tests...\n\n");

    test_initialization();
    test_hardware_optimized_mitigation();
    test_fast_feedback();
    test_pattern_detection();
    test_confidence_tracking();

    printf("\nAll tests passed!\n");
    return 0;
}
