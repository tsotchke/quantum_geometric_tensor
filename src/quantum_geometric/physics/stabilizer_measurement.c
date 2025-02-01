/**
 * @file stabilizer_measurement.c
 * @brief Implementation of quantum stabilizer measurement system
 */

#include "quantum_geometric/physics/stabilizer_measurement.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/physics/quantum_state_operations.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Forward declarations
static bool initialize_stabilizer_array(StabilizerArray* array,
                                      const StabilizerConfig* config);
static void cleanup_stabilizer_array(StabilizerArray* array);
static bool measure_plaquette_operator(const quantum_state* state,
                                     size_t x,
                                     size_t y,
                                     double* result,
                                     const StabilizerConfig* config);
static bool measure_vertex_operator(const quantum_state* state,
                                  size_t x,
                                  size_t y,
                                  double* result,
                                  const StabilizerConfig* config);
static bool apply_error_correction(quantum_state* state,
                                 const StabilizerArray* array);

// Helper functions for parallel measurement and error tracking
static void update_error_correlations(StabilizerState* state);
static bool can_measure_in_parallel(const StabilizerState* state,
                                  size_t idx1,
                                  size_t idx2);
static void track_measurement_pattern(StabilizerState* state,
                                    size_t stabilizer_idx,
                                    double measurement);

// X-stabilizer specific helper functions
static size_t determine_optimal_repetitions(size_t base_count, double error_rate) {
    // Dynamically adjust repetition count based on error rate
    double scale_factor = 1.0 + (error_rate * 2.0);
    return (size_t)(base_count * scale_factor);
}

static double get_x_stabilizer_correlation(const quantum_state* state,
                                         size_t x,
                                         size_t y,
                                         size_t qubit_idx) {
    // Get historical correlation data for specific qubit in X-stabilizer
    double correlation = 0.0;
    size_t measurements = get_qubit_measurement_history(state, x, y, qubit_idx);
    if (measurements > 0) {
        correlation = get_x_basis_error_rate(state, x, y, qubit_idx) *
                     get_spatial_correlation(state, x, y);
    }
    return correlation;
}

static void apply_x_error_mitigation_sequence(const quantum_state* state,
                                            size_t x,
                                            size_t y) {
    // Apply X-specific dynamical decoupling sequence
    apply_hadamard_frame(state, x, y);
    apply_echo_sequence(state, x, y);
    apply_composite_pulse(state, x, y);
}

static bool measure_pauli_x_with_enhanced_confidence(const quantum_state* state,
                                                   size_t x,
                                                   size_t y,
                                                   double* result,
                                                   double* confidence,
                                                   double correlation) {
    // Enhanced X measurement with correlation-aware confidence
    if (!measure_pauli_x_with_confidence(state, x, y, result, confidence)) {
        return false;
    }
    
    // Apply correlation-based confidence adjustment
    *confidence *= (1.0 - correlation);
    
    // Apply additional X-specific error mitigation
    apply_x_measurement_correction(state, x, y, result);
    return true;
}

static void update_x_stabilizer_correlations(double* correlations,
                                           double x1,
                                           double x2,
                                           double x3,
                                           double x4,
                                           double c1,
                                           double c2,
                                           double c3,
                                           double c4) {
    // Update correlation model based on measurement results
    const double decay_rate = 0.95;
    const double update_weight = 0.1;
    
    correlations[0] = decay_rate * correlations[0] + 
                      update_weight * fabs(x1 * x2) * (c1 + c2) / 2.0;
    correlations[1] = decay_rate * correlations[1] + 
                      update_weight * fabs(x2 * x3) * (c2 + c3) / 2.0;
    correlations[2] = decay_rate * correlations[2] + 
                      update_weight * fabs(x3 * x4) * (c3 + c4) / 2.0;
    correlations[3] = decay_rate * correlations[3] + 
                      update_weight * fabs(x4 * x1) * (c4 + c1) / 2.0;
}

static void sort_measurements_by_confidence(double* measurements,
                                          double* confidences,
                                          size_t count) {
    // Sort measurements by confidence using insertion sort
    for (size_t i = 1; i < count; i++) {
        double temp_m = measurements[i];
        double temp_c = confidences[i];
        size_t j = i;
        while (j > 0 && confidences[j-1] < temp_c) {
            measurements[j] = measurements[j-1];
            confidences[j] = confidences[j-1];
            j--;
        }
        measurements[j] = temp_m;
        confidences[j] = temp_c;
    }
}

static size_t calculate_optimal_window(const double* confidences,
                                     const double* correlations,
                                     size_t valid_count,
                                     const StabilizerConfig* config) {
    // Calculate optimal measurement window based on confidences and correlations
    double total_correlation = 0.0;
    for (size_t i = 0; i < 4; i++) {
        total_correlation += correlations[i];
    }
    
    double avg_correlation = total_correlation / 4.0;
    double window_scale = 1.0 - (avg_correlation * config->measurement_error_rate);
    
    size_t base_window = (size_t)(valid_count * config->confidence_threshold);
    return (size_t)(base_window * window_scale);
}

static double calculate_x_confidence_factor(const double* correlations) {
    // Calculate confidence adjustment factor for X-stabilizers
    double total_correlation = 0.0;
    for (size_t i = 0; i < 4; i++) {
        total_correlation += correlations[i];
    }
    return 1.0 - (total_correlation / 8.0); // Scale factor between 0.5 and 1.0
}

static double get_correlation_adjustment(const double* correlations) {
    // Calculate threshold adjustment based on correlations
    double max_correlation = 0.0;
    for (size_t i = 0; i < 4; i++) {
        if (correlations[i] > max_correlation) {
            max_correlation = correlations[i];
        }
    }
    return max_correlation * 0.5; // Up to 50% threshold adjustment
}

bool init_stabilizer_measurement(StabilizerState* state,
                               const StabilizerConfig* config) {
    if (!state || !config) {
        return false;
    }

    // Allocate and initialize stabilizer arrays
    state->plaquette_stabilizers = malloc(sizeof(StabilizerArray));
    state->vertex_stabilizers = malloc(sizeof(StabilizerArray));

    if (!state->plaquette_stabilizers || !state->vertex_stabilizers) {
        free(state->plaquette_stabilizers);
        free(state->vertex_stabilizers);
        return false;
    }

    if (!initialize_stabilizer_array(state->plaquette_stabilizers, config) ||
        !initialize_stabilizer_array(state->vertex_stabilizers, config)) {
        cleanup_stabilizer_array(state->plaquette_stabilizers);
        cleanup_stabilizer_array(state->vertex_stabilizers);
        free(state->plaquette_stabilizers);
        free(state->vertex_stabilizers);
        return false;
    }

    memcpy(&state->config, config, sizeof(StabilizerConfig));
    state->measurement_count = 0;
    state->error_rate = 0.0;
    state->last_syndrome = NULL;

    // Initialize error detection enhancements
    size_t total_stabilizers = state->plaquette_stabilizers->size + 
                             state->vertex_stabilizers->size;
    
    state->measurement_confidence = calloc(total_stabilizers, sizeof(double));
    state->repetition_results = calloc(total_stabilizers * config->repetition_count, 
                                     sizeof(size_t));
    state->error_correlations = calloc(total_stabilizers, sizeof(double));

    // Initialize parallel measurement tracking
    state->measured_in_parallel = calloc(total_stabilizers, sizeof(bool));
    state->current_parallel_group = 0;

    // Initialize error pattern recognition
    state->history_capacity = 1000; // Start with space for 1000 measurements
    state->history_size = 0;
    state->measurement_history = malloc(state->history_capacity * sizeof(double*));
    
    if (!state->measurement_confidence || !state->repetition_results ||
        !state->error_correlations || !state->measured_in_parallel ||
        !state->measurement_history) {
        cleanup_stabilizer_measurement(state);
        return false;
    }

    // Allocate initial history entries
    for (size_t i = 0; i < state->history_capacity; i++) {
        state->measurement_history[i] = malloc(total_stabilizers * sizeof(double));
        if (!state->measurement_history[i]) {
            cleanup_stabilizer_measurement(state);
            return false;
        }
    }

    return true;
}

void cleanup_stabilizer_measurement(StabilizerState* state) {
    if (state) {
        cleanup_stabilizer_array(state->plaquette_stabilizers);
        cleanup_stabilizer_array(state->vertex_stabilizers);
        free(state->plaquette_stabilizers);
        free(state->vertex_stabilizers);
        free(state->last_syndrome);
        
        // Clean up error detection enhancements
        free(state->measurement_confidence);
        free(state->repetition_results);
        free(state->error_correlations);
        
        // Clean up parallel measurement tracking
        free(state->measured_in_parallel);
        
        // Clean up error pattern recognition
        if (state->measurement_history) {
            for (size_t i = 0; i < state->history_capacity; i++) {
                free(state->measurement_history[i]);
            }
            free(state->measurement_history);
        }
        
        memset(state, 0, sizeof(StabilizerState));
    }
}

bool measure_stabilizers(StabilizerState* state,
                        quantum_state* qstate) {
    if (!state || !qstate) {
        return false;
    }

    // Reset stabilizer measurements and tracking arrays
    size_t total_stabilizers = state->plaquette_stabilizers->size +
                             state->vertex_stabilizers->size;
    memset(state->plaquette_stabilizers->measurements, 0,
           state->plaquette_stabilizers->size * sizeof(double));
    memset(state->vertex_stabilizers->measurements, 0,
           state->vertex_stabilizers->size * sizeof(double));
    memset(state->measured_in_parallel, 0,
           total_stabilizers * sizeof(bool));
    memset(state->measurement_confidence, 0,
           total_stabilizers * sizeof(double));

    size_t width = state->config.lattice_width;
    size_t height = state->config.lattice_height;
    bool success = true;

    // Initialize parallel measurement groups
    if (state->config.enable_parallel) {
        // Reset parallel group counter
        state->current_parallel_group = 0;

        // Calculate optimal group sizes based on error correlations
        size_t max_group_size = state->config.max_parallel_ops;
        if (state->measurement_count > 0) {
            // Adjust group size based on observed error correlations
            double avg_correlation = 0.0;
            size_t correlation_count = 0;
            
            for (size_t i = 0; i < total_stabilizers; i++) {
                if (state->error_correlations[i] > 0.0) {
                    avg_correlation += state->error_correlations[i];
                    correlation_count++;
                }
            }
            
            if (correlation_count > 0) {
                avg_correlation /= correlation_count;
                // Reduce group size if high correlations detected
                max_group_size = (size_t)(state->config.max_parallel_ops * 
                                        (1.0 - avg_correlation));
                if (max_group_size < 1) max_group_size = 1;
            }
        }

        // Measure plaquette operators in optimized parallel groups
        for (size_t x = 0; x < width - 1; x += 2) {
            for (size_t y = 0; y < height - 1; y += 2) {
                size_t group_size = 0;
                size_t indices[4]; // Store indices for parallel measurement
                double correlations[4] = {0.0}; // Store correlation scores

                // Collect potential plaquettes and their correlations
                for (size_t dx = 0; dx < 2 && x + dx < width - 1; dx++) {
                    for (size_t dy = 0; dy < 2 && y + dy < height - 1; dy++) {
                        size_t idx = (y + dy) * (width - 1) + (x + dx);
                        
                        // Check correlation with existing group members
                        double max_correlation = 0.0;
                        for (size_t i = 0; i < group_size; i++) {
                            if (state->error_correlations[indices[i]] > max_correlation) {
                                max_correlation = state->error_correlations[indices[i]];
                            }
                        }
                        
                        // Add to group if correlation is below threshold
                        if (max_correlation < state->config.correlation_threshold) {
                            indices[group_size] = idx;
                            correlations[group_size] = max_correlation;
                            group_size++;
                        }
                    }
                }

                // Sort by correlation score
                for (size_t i = 0; i < group_size; i++) {
                    for (size_t j = i + 1; j < group_size; j++) {
                        if (correlations[j] < correlations[i]) {
                            // Swap indices and correlations
                            size_t temp_idx = indices[i];
                            indices[i] = indices[j];
                            indices[j] = temp_idx;
                            
                            double temp_corr = correlations[i];
                            correlations[i] = correlations[j];
                            correlations[j] = temp_corr;
                        }
                    }
                }

                // Measure optimal group
                size_t optimal_size = (group_size < max_group_size) ? 
                                    group_size : max_group_size;
                
                if (optimal_size > 0) {
                    for (size_t i = 0; i < optimal_size; i++) {
                        size_t idx = indices[i];
                        size_t px = idx % (width - 1);
                        size_t py = idx / (width - 1);
                        double result;
                        if (!measure_plaquette_operator(qstate, px, py, &result, &state->config)) {
                            success = false;
                            break;
                        }
                        state->plaquette_stabilizers->measurements[idx] = result;
                        state->measured_in_parallel[idx] = true;
                        
                        // Update confidence based on correlation
                        double confidence = 1.0 - state->config.measurement_error_rate;
                        if (correlations[i] > 0.0) {
                            confidence *= (1.0 - correlations[i]);
                        }
                        state->measurement_confidence[idx] = confidence;
                    }
                    if (!success) break;
                    state->current_parallel_group++;
                }
            }
            if (!success) break;
        }

        // Similar optimized parallel grouping for vertex operators
        if (success) {
            for (size_t x = 1; x < width; x += 2) {
                for (size_t y = 1; y < height; y += 2) {
                    size_t group_size = 0;
                    size_t indices[4];
                    double correlations[4] = {0.0};

                    // Collect potential vertices and their correlations
                    for (size_t dx = 0; dx < 2 && x + dx < width; dx++) {
                        for (size_t dy = 0; dy < 2 && y + dy < height; dy++) {
                            size_t idx = state->plaquette_stabilizers->size +
                                       ((y + dy - 1) * (width - 1) + (x + dx - 1));
                            
                            // Check correlation with existing group members
                            double max_correlation = 0.0;
                            for (size_t i = 0; i < group_size; i++) {
                                if (state->error_correlations[indices[i]] > max_correlation) {
                                    max_correlation = state->error_correlations[indices[i]];
                                }
                            }
                            
                            // Add to group if correlation is below threshold
                            if (max_correlation < state->config.correlation_threshold) {
                                indices[group_size] = idx;
                                correlations[group_size] = max_correlation;
                                group_size++;
                            }
                        }
                    }

                    // Sort by correlation score
                    for (size_t i = 0; i < group_size; i++) {
                        for (size_t j = i + 1; j < group_size; j++) {
                            if (correlations[j] < correlations[i]) {
                                // Swap indices and correlations
                                size_t temp_idx = indices[i];
                                indices[i] = indices[j];
                                indices[j] = temp_idx;
                                
                                double temp_corr = correlations[i];
                                correlations[i] = correlations[j];
                                correlations[j] = temp_corr;
                            }
                        }
                    }

                    // Measure optimal group
                    size_t optimal_size = (group_size < max_group_size) ? 
                                        group_size : max_group_size;
                    
                    if (optimal_size > 0) {
                        for (size_t i = 0; i < optimal_size; i++) {
                            size_t idx = indices[i] - state->plaquette_stabilizers->size;
                            size_t vx = (idx % (width - 1)) + 1;
                            size_t vy = (idx / (width - 1)) + 1;
                            double result;
                            if (!measure_vertex_operator(qstate, vx, vy, &result, &state->config)) {
                                success = false;
                                break;
                            }
                            state->vertex_stabilizers->measurements[idx] = result;
                            state->measured_in_parallel[indices[i]] = true;
                            
                            // Update confidence based on correlation
                            double confidence = 1.0 - state->config.measurement_error_rate;
                            if (correlations[i] > 0.0) {
                                confidence *= (1.0 - correlations[i]);
                            }
                            state->measurement_confidence[indices[i]] = confidence;
                        }
                        if (!success) break;
                        state->current_parallel_group++;
                    }
                }
                if (!success) break;
            }
        }
    } else {
        // Sequential measurements when parallel is disabled
        for (size_t x = 0; x < width - 1 && success; x++) {
            for (size_t y = 0; y < height - 1 && success; y++) {
                double result;
                if (!measure_plaquette_operator(qstate, x, y, &result, &state->config)) {
                    success = false;
                    break;
                }
                size_t idx = y * (width - 1) + x;
                state->plaquette_stabilizers->measurements[idx] = result;
                state->measurement_confidence[idx] = 1.0 - state->config.measurement_error_rate;
            }
        }

        if (success) {
            for (size_t x = 1; x < width && success; x++) {
                for (size_t y = 1; y < height && success; y++) {
                    double result;
                    if (!measure_vertex_operator(qstate, x, y, &result, &state->config)) {
                        success = false;
                        break;
                    }
                    size_t idx = (y - 1) * (width - 1) + (x - 1);
                    state->vertex_stabilizers->measurements[idx] = result;
                    state->measurement_confidence[idx + state->plaquette_stabilizers->size] = 
                        1.0 - state->config.measurement_error_rate;
                }
            }
        }
    }

    if (success) {
        state->measurement_count++;
        
        // Initialize hardware backend
        IBMBackendState* ibm_state = init_ibm_backend_state();
        if (!ibm_state) {
            return false;
        }

        // Configure IBM backend
        IBMBackendConfig ibm_config = {
            .backend_name = "ibmq_manhattan",
            .optimization_level = 3,
            .error_mitigation = true,
            .fast_feedback = true,
            .dynamic_decoupling = true,
            .measurement_error_mitigation = true,
            .readout_error_mitigation = true,
            .noise_extrapolation = true
        };

        if (!init_ibm_backend(ibm_state, &ibm_config)) {
            cleanup_ibm_backend(ibm_state);
            return false;
        }

        // Initialize protection system
        ProtectionSystem* protection = init_protection_system(qstate, &ibm_state->config);
        if (!protection) {
            cleanup_ibm_backend(ibm_state);
            return false;
        }

        // Initialize error syndrome tracking
        MatchingGraph* graph = init_matching_graph(total_stabilizers, total_stabilizers * 2);
        if (!graph) {
            cleanup_protection_system(protection);
            cleanup_ibm_backend(ibm_state);
            return false;
        }

        // Configure syndrome detection with hardware-aware and protection parameters
        SyndromeConfig syndrome_config = {
            .detection_threshold = state->config.error_threshold * 
                                 (1.0 - ibm_state->error_rates[0]) *
                                 (1.0 - protection->error_tracker->total_weight),
            .confidence_threshold = state->config.confidence_threshold *
                                  (1.0 - ibm_state->readout_errors[0]) *
                                  protection->verifier->threshold_stability,
            .weight_scale_factor = 1.0,
            .use_boundary_matching = true,
            .enable_parallel = state->config.enable_parallel,
            .parallel_group_size = min(state->config.max_parallel_ops,
                                     ibm_state->num_qubits / 2),
            .min_pattern_occurrences = 3,
            .pattern_threshold = 0.7,
            .max_matching_iterations = 1000
        };

        // Run protection cycle
        if (should_run_fast_cycle(protection)) {
            // Fast cycle: Error detection only
            detect_topological_errors(qstate, &ibm_state->config);
        }

        if (should_run_medium_cycle(protection)) {
            // Medium cycle: Error correction
            detect_topological_errors(qstate, &ibm_state->config);
            correct_topological_errors(qstate, &ibm_state->config);
        }

        if (should_run_slow_cycle(protection)) {
            // Slow cycle: Full verification
            if (!verify_topological_state(qstate, &ibm_state->config)) {
                // State verification failed, perform recovery
                log_correction_failure(qstate, NULL);
                // Attempt recovery through stronger correction
                AnyonSet* anyons = detect_mitigated_anyons(qstate, NULL);
                if (anyons) {
                    CorrectionPattern* pattern = optimize_correction_pattern(anyons, NULL);
                    if (pattern) {
                        apply_mitigated_correction(qstate, pattern, NULL);
                        free_correction_pattern(pattern);
                    }
                    free_anyon_set(anyons);
                }
            }
        }

        // Create optimized quantum circuit
        quantum_circuit* circuit = create_stabilizer_circuit(state, qstate);
        if (!circuit) {
            cleanup_ibm_backend(ibm_state);
            cleanup_matching_graph(graph);
            return false;
        }

        // Optimize circuit for hardware
        if (!optimize_circuit(ibm_state, circuit)) {
            cleanup_quantum_circuit(circuit);
            cleanup_ibm_backend(ibm_state);
            cleanup_matching_graph(graph);
            return false;
        }

        // Execute optimized circuit
        quantum_result* result = create_quantum_result();
        if (!result) {
            cleanup_quantum_circuit(circuit);
            cleanup_ibm_backend(ibm_state);
            cleanup_matching_graph(graph);
            return false;
        }

        if (!execute_circuit(ibm_state, circuit, result)) {
            cleanup_quantum_result(result);
            cleanup_quantum_circuit(circuit);
            cleanup_ibm_backend(ibm_state);
            cleanup_matching_graph(graph);
            return false;
        }

        // Extract error syndromes from hardware results
        size_t num_syndromes = extract_error_syndromes(result, &syndrome_config, graph);
        
        // Update error rate and track patterns with hardware-aware confidence
        size_t error_count = 0;
        for (size_t i = 0; i < total_stabilizers; i++) {
            // Apply hardware-specific error mitigation
            double raw_measurement = (i < state->plaquette_stabilizers->size) ?
                state->plaquette_stabilizers->measurements[i] :
                state->vertex_stabilizers->measurements[i - state->plaquette_stabilizers->size];
            
            double mitigated_measurement = apply_error_mitigation(ibm_state,
                                                                raw_measurement,
                                                                i);
            double measurement = (i < state->plaquette_stabilizers->size) ?
                state->plaquette_stabilizers->measurements[i] :
                state->vertex_stabilizers->measurements[i - state->plaquette_stabilizers->size];
            
            if (fabs(measurement + 1.0) < 1e-6) {
                error_count++;
            }

            // Update error pattern history with hardware-aware correlation
            if (state->history_size == state->history_capacity) {
                // Shift history to make room
                for (size_t j = 1; j < state->history_capacity; j++) {
                    memcpy(state->measurement_history[j-1],
                          state->measurement_history[j],
                          total_stabilizers * sizeof(double));
                }
                state->history_size--;
            }
            state->measurement_history[state->history_size][i] = mitigated_measurement;

            // Update correlation tracking with hardware noise model
            if (i < graph->num_vertices) {
                track_measurement_pattern(state, i, mitigated_measurement);
                
                // Update confidence based on hardware and syndrome analysis
                double hw_confidence = 1.0 - ibm_state->error_rates[i % ibm_state->num_qubits];
                if (graph->vertices[i].confidence > 0.0) {
                    state->measurement_confidence[i] *= graph->vertices[i].confidence * hw_confidence;
                }
            }
        }
        state->history_size++;
        state->error_rate = (double)error_count / total_stabilizers;

        // Find minimum weight perfect matching if syndromes detected
        if (num_syndromes > 0) {
            if (find_minimum_weight_matching(graph, &syndrome_config)) {
                // Store syndrome data
                size_t syndrome_size = total_stabilizers * sizeof(double);
                double* new_syndrome = realloc(state->last_syndrome, syndrome_size);
                if (new_syndrome) {
                    state->last_syndrome = new_syndrome;
                    memcpy(state->last_syndrome,
                          state->plaquette_stabilizers->measurements,
                          state->plaquette_stabilizers->size * sizeof(double));
                    memcpy(state->last_syndrome + state->plaquette_stabilizers->size,
                          state->vertex_stabilizers->measurements,
                          state->vertex_stabilizers->size * sizeof(double));
                }

                // Apply error correction if needed
                if (state->config.auto_correction &&
                    verify_syndrome_matching(graph, qstate)) {
                    success = apply_matching_correction(graph, qstate);
                }
            }
        }

        // Update error correlations for future measurements
        update_error_correlations(state);

        // Clean up
        cleanup_quantum_result(result);
        cleanup_quantum_circuit(circuit);
        cleanup_protection_system(protection);
        cleanup_ibm_backend(ibm_state);
        cleanup_matching_graph(graph);

        // Wait for next protection cycle
        wait_protection_interval(protection);
    }

    return success;
}

const double* get_stabilizer_measurements(const StabilizerState* state,
                                        StabilizerType type,
                                        size_t* size) {
    if (!state || !size) {
        return NULL;
    }

    switch (type) {
        case STABILIZER_PLAQUETTE:
            *size = state->plaquette_stabilizers->size;
            return state->plaquette_stabilizers->measurements;
        case STABILIZER_VERTEX:
            *size = state->vertex_stabilizers->size;
            return state->vertex_stabilizers->measurements;
        default:
            return NULL;
    }
}

double get_error_rate(const StabilizerState* state) {
    return state ? state->error_rate : 0.0;
}

const double* get_last_syndrome(const StabilizerState* state, size_t* size) {
    if (!state || !size) {
        return NULL;
    }
    *size = state->plaquette_stabilizers->size + state->vertex_stabilizers->size;
    return state->last_syndrome;
}

// Helper function implementations
static bool initialize_stabilizer_array(StabilizerArray* array,
                                      const StabilizerConfig* config) {
    if (!array || !config) {
        return false;
    }

    // Calculate array size based on lattice dimensions
    array->size = (config->lattice_width - 1) * (config->lattice_height - 1);
    array->measurements = calloc(array->size, sizeof(double));
    
    return array->measurements != NULL;
}

static void cleanup_stabilizer_array(StabilizerArray* array) {
    if (array) {
        free(array->measurements);
        memset(array, 0, sizeof(StabilizerArray));
    }
}

static bool measure_plaquette_operator(const quantum_state* state,
                                     size_t x,
                                     size_t y,
                                     double* result,
                                     const StabilizerConfig* config) {
    if (!state || !result || !config) {
        return false;
    }

    // Handle periodic boundary conditions if enabled
    size_t x1 = x, x2 = x + 1;
    size_t y1 = y, y2 = y + 1;
    
    if (config->periodic_boundaries) {
        x2 = x2 % config->lattice_width;
        y2 = y2 % config->lattice_height;
    } else if (config->handle_boundaries) {
        // Skip measurement if any qubit is outside lattice
        if (x2 >= config->lattice_width || y2 >= config->lattice_height) {
            *result = 1.0; // Identity for boundary
            return true;
        }
    }

    // Initialize error mitigation
    double* measurements = calloc(config->repetition_count, sizeof(double));
    double* confidences = calloc(config->repetition_count, sizeof(double));
    if (!measurements || !confidences) {
        free(measurements);
        free(confidences);
        return false;
    }

    // Perform repeated measurements with error mitigation
    size_t valid_measurements = 0;
    for (size_t i = 0; i < config->repetition_count; i++) {
        // Measure Z operators on all qubits around the plaquette
        double z1, z2, z3, z4;
        double c1, c2, c3, c4;
        
        if (!measure_pauli_z_with_confidence(state, x1, y1, &z1, &c1) ||
            !measure_pauli_z_with_confidence(state, x2, y1, &z2, &c2) ||
            !measure_pauli_z_with_confidence(state, x1, y2, &z3, &c3) ||
            !measure_pauli_z_with_confidence(state, x2, y2, &z4, &c4)) {
            continue;
        }

        // Calculate measurement and confidence
        measurements[valid_measurements] = z1 * z2 * z3 * z4;
        confidences[valid_measurements] = c1 * c2 * c3 * c4;
        valid_measurements++;
    }

    // Require minimum number of valid measurements
    if (valid_measurements < config->min_valid_measurements) {
        free(measurements);
        free(confidences);
        return false;
    }

    // Apply error mitigation
    double final_result = 0.0;
    double total_confidence = 0.0;
    
    // Sort measurements by confidence
    for (size_t i = 0; i < valid_measurements; i++) {
        for (size_t j = i + 1; j < valid_measurements; j++) {
            if (confidences[j] > confidences[i]) {
                // Swap measurements and confidences
                double temp_m = measurements[i];
                measurements[i] = measurements[j];
                measurements[j] = temp_m;
                
                double temp_c = confidences[i];
                confidences[i] = confidences[j];
                confidences[j] = temp_c;
            }
        }
    }

    // Calculate weighted average of highest confidence measurements
    size_t top_k = (size_t)(valid_measurements * config->confidence_threshold);
    if (top_k == 0) top_k = 1;
    
    for (size_t i = 0; i < top_k; i++) {
        final_result += measurements[i] * confidences[i];
        total_confidence += confidences[i];
    }

    *result = final_result / total_confidence;

    // Apply dynamic threshold adjustment
    if (total_confidence < config->min_confidence) {
        *result = 0.0; // Uncertain measurement
    } else if (fabs(*result) < config->error_threshold) {
        *result = (*result > 0) ? 1.0 : -1.0; // Snap to nearest eigenvalue
    }

    free(measurements);
    free(confidences);
    return true;
}

static bool measure_vertex_operator(const quantum_state* state,
                                  size_t x,
                                  size_t y,
                                  double* result,
                                  const StabilizerConfig* config) {
    if (!state || !result || !config) {
        return false;
    }

    // Handle periodic boundary conditions if enabled
    size_t x1 = (x > 0) ? x - 1 : (config->periodic_boundaries ? config->lattice_width - 1 : 0);
    size_t y1 = (y > 0) ? y - 1 : (config->periodic_boundaries ? config->lattice_height - 1 : 0);
    
    if (!config->periodic_boundaries && config->handle_boundaries) {
        // Skip measurement if any qubit is outside lattice
        if (x == 0 || y == 0) {
            *result = 1.0; // Identity for boundary
            return true;
        }
    }

    // Enhanced error mitigation for X-stabilizers
    size_t optimal_reps = determine_optimal_repetitions(config->repetition_count, 
                                                      config->measurement_error_rate);
    double* measurements = calloc(optimal_reps, sizeof(double));
    double* confidences = calloc(optimal_reps, sizeof(double));
    double* correlations = calloc(4, sizeof(double)); // For each qubit
    
    if (!measurements || !confidences || !correlations) {
        free(measurements);
        free(confidences);
        free(correlations);
        return false;
    }

    // Initialize correlation tracking
    for (size_t i = 0; i < 4; i++) {
        correlations[i] = get_x_stabilizer_correlation(state, x, y, i);
    }

    // Perform optimized X-basis measurements
    size_t valid_measurements = 0;
    for (size_t i = 0; i < optimal_reps; i++) {
        // Apply X-specific error mitigation sequences
        apply_x_error_mitigation_sequence(state, x, y);
        
        // Measure X operators with dynamic decoupling
        double x1_val, x2_val, x3_val, x4_val;
        double c1, c2, c3, c4;
        
        if (!measure_pauli_x_with_enhanced_confidence(state, x1, y1, &x1_val, &c1, correlations[0]) ||
            !measure_pauli_x_with_enhanced_confidence(state, x, y1, &x2_val, &c2, correlations[1]) ||
            !measure_pauli_x_with_enhanced_confidence(state, x1, y, &x3_val, &c3, correlations[2]) ||
            !measure_pauli_x_with_enhanced_confidence(state, x, y, &x4_val, &c4, correlations[3])) {
            continue;
        }

        // Calculate measurement with correlation-weighted confidence
        measurements[valid_measurements] = x1_val * x2_val * x3_val * x4_val;
        double combined_confidence = c1 * c2 * c3 * c4;
        
        // Apply correlation-based confidence adjustment
        for (size_t j = 0; j < 4; j++) {
            combined_confidence *= (1.0 - correlations[j] * config->measurement_error_rate);
        }
        
        confidences[valid_measurements] = combined_confidence;
        valid_measurements++;
        
        // Update correlations based on measurement results
        update_x_stabilizer_correlations(correlations, 
                                       x1_val, x2_val, x3_val, x4_val,
                                       c1, c2, c3, c4);
    }

    // Enhanced error mitigation and result processing
    if (valid_measurements < config->min_valid_measurements) {
        free(measurements);
        free(confidences);
        free(correlations);
        return false;
    }

    // Apply advanced error mitigation
    double final_result = 0.0;
    double total_confidence = 0.0;
    
    // Sort measurements by confidence and apply correlation weighting
    sort_measurements_by_confidence(measurements, confidences, valid_measurements);
    
    // Calculate optimal measurement window based on correlations
    size_t window_size = calculate_optimal_window(confidences, correlations, 
                                                valid_measurements, config);
    
    // Apply windowed averaging with correlation weights
    for (size_t i = 0; i < window_size; i++) {
        double correlation_weight = 1.0;
        for (size_t j = 0; j < 4; j++) {
            correlation_weight *= (1.0 - correlations[j]);
        }
        
        final_result += measurements[i] * confidences[i] * correlation_weight;
        total_confidence += confidences[i] * correlation_weight;
    }

    *result = final_result / total_confidence;

    // Apply enhanced threshold adjustment for X-stabilizers
    if (total_confidence < config->min_confidence * 
        calculate_x_confidence_factor(correlations)) {
        *result = 0.0; // Uncertain measurement
    } else {
        // Apply correlation-aware thresholding
        double adjusted_threshold = config->error_threshold * 
                                  (1.0 + get_correlation_adjustment(correlations));
        if (fabs(*result) < adjusted_threshold) {
            *result = (*result > 0) ? 1.0 : -1.0;
        }
    }

    free(measurements);
    free(confidences);
    free(correlations);
    return true;
}

// Helper functions for parallel measurement and error tracking
static void update_error_correlations(StabilizerState* state) {
    if (!state) {
        return;
    }

    size_t total_stabilizers = state->plaquette_stabilizers->size +
                             state->vertex_stabilizers->size;

    // Calculate spatial correlations between errors
    for (size_t i = 0; i < total_stabilizers; i++) {
        double correlation = 0.0;
        size_t correlation_count = 0;

        // Get current stabilizer's measurement
        double measurement_i = (i < state->plaquette_stabilizers->size) ?
            state->plaquette_stabilizers->measurements[i] :
            state->vertex_stabilizers->measurements[i - state->plaquette_stabilizers->size];

        // Compare with neighboring stabilizers
        for (size_t j = 0; j < total_stabilizers; j++) {
            if (i == j) continue;

            double measurement_j = (j < state->plaquette_stabilizers->size) ?
                state->plaquette_stabilizers->measurements[j] :
                state->vertex_stabilizers->measurements[j - state->plaquette_stabilizers->size];

            // Check if these stabilizers are neighbors
            if (can_measure_in_parallel(state, i, j)) {
                correlation += measurement_i * measurement_j;
                correlation_count++;
            }
        }

        // Update correlation value
        state->error_correlations[i] = correlation_count > 0 ?
            correlation / correlation_count : 0.0;
    }
}

static bool can_measure_in_parallel(const StabilizerState* state,
                                  size_t idx1,
                                  size_t idx2) {
    if (!state) {
        return false;
    }

    size_t width = state->config.lattice_width;
    bool is_plaquette1 = idx1 < state->plaquette_stabilizers->size;
    bool is_plaquette2 = idx2 < state->plaquette_stabilizers->size;

    // Convert indices to grid coordinates
    size_t x1, y1, x2, y2;
    if (is_plaquette1) {
        x1 = idx1 % (width - 1);
        y1 = idx1 / (width - 1);
    } else {
        idx1 -= state->plaquette_stabilizers->size;
        x1 = (idx1 % (width - 1)) + 1;
        y1 = (idx1 / (width - 1)) + 1;
    }

    if (is_plaquette2) {
        x2 = idx2 % (width - 1);
        y2 = idx2 / (width - 1);
    } else {
        idx2 -= state->plaquette_stabilizers->size;
        x2 = (idx2 % (width - 1)) + 1;
        y2 = (idx2 / (width - 1)) + 1;
    }

    // Check if stabilizers share any qubits
    if (is_plaquette1 == is_plaquette2) {
        // Same type stabilizers
        return abs((int)x1 - (int)x2) > 1 || abs((int)y1 - (int)y2) > 1;
    } else {
        // Different type stabilizers
        return abs((int)x1 - (int)x2) >= 1 && abs((int)y1 - (int)y2) >= 1;
    }
}

static void track_measurement_pattern(StabilizerState* state,
                                    size_t stabilizer_idx,
                                    double measurement) {
    if (!state) {
        return;
    }

    // Update measurement history
    if (state->history_size > 0) {
        // Check for repeating patterns
        size_t pattern_length = 0;
        for (size_t len = 1; len <= state->history_size / 2; len++) {
            bool is_pattern = true;
            for (size_t i = 0; i < len && i < state->history_size - len; i++) {
                if (fabs(state->measurement_history[state->history_size - 1 - i][stabilizer_idx] -
                        state->measurement_history[state->history_size - 1 - len - i][stabilizer_idx]) > 1e-6) {
                    is_pattern = false;
                    break;
                }
            }
            if (is_pattern) {
                pattern_length = len;
                break;
            }
        }

        // Update confidence based on pattern recognition
        if (pattern_length > 0) {
            double predicted = state->measurement_history[state->history_size - pattern_length][stabilizer_idx];
            if (fabs(predicted - measurement) < 1e-6) {
                state->measurement_confidence[stabilizer_idx] *= 1.1; // Increase confidence
            } else {
                state->measurement_confidence[stabilizer_idx] *= 0.9; // Decrease confidence
            }
        }
    }
}

static bool apply_error_correction(quantum_state* state,
                                 const StabilizerArray* array) {
    if (!state || !array) {
        return false;
    }

    // Apply correction operations based on stabilizer measurements
    bool success = true;
    for (size_t i = 0; i < array->size; i++) {
        if (fabs(array->measurements[i] + 1.0) < 1e-6) {
            // Negative measurement indicates error
            // Apply appropriate correction based on stabilizer type
            size_t x = i % (state->width - 1);
            size_t y = i / (state->width - 1);
            
            // For plaquette stabilizers, apply X corrections
            success = apply_pauli_x(state, x, y) &&
                     apply_pauli_x(state, x + 1, y) &&
                     apply_pauli_x(state, x, y + 1) &&
                     apply_pauli_x(state, x + 1, y + 1);
            
            if (!success) break;
        }
    }

    return success;

