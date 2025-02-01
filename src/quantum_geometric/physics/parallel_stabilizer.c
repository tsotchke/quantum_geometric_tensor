/**
 * @file parallel_stabilizer.c
 * @brief Implementation of parallel stabilizer measurements with hardware optimization
 */

#include "quantum_geometric/physics/parallel_stabilizer.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/physics/quantum_state_operations.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

// Thread synchronization primitives
static pthread_mutex_t feedback_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t feedback_cond = PTHREAD_COND_INITIALIZER;
static volatile bool feedback_ready = false;
static MeasurementFeedback global_feedback = {0};

// Forward declarations
static void* measurement_thread(void* arg);
static bool initialize_thread_data(ThreadData* data,
                                const quantum_state* state,
                                const size_t* qubit_indices,
                                size_t num_qubits,
                                StabilizerType type,
                                size_t thread_id,
                                size_t total_threads,
                                const HardwareProfile* hw_profile);
static void cleanup_thread_data(ThreadData* data);
static void update_thread_workload(ThreadData* data, 
                                 const MeasurementFeedback* feedback);
static double calculate_measurement_confidence(const ThreadData* data,
                                            size_t qubit_index);
static void trigger_fast_feedback(const ThreadData* data,
                                const MeasurementResult* result);

bool measure_stabilizers_parallel(const quantum_state* state,
                                const size_t* qubit_indices,
                                size_t num_qubits,
                                StabilizerType type,
                                size_t num_threads,
                                double* results,
                                const HardwareProfile* hw_profile) {
    if (!state || !qubit_indices || !results || num_qubits == 0 || num_threads == 0) {
        return false;
    }

    // Adjust thread count if needed
    num_threads = (num_threads > num_qubits) ? num_qubits : num_threads;

    // Allocate thread resources
    pthread_t* threads = malloc(num_threads * sizeof(pthread_t));
    ThreadData* thread_data = malloc(num_threads * sizeof(ThreadData));
    if (!threads || !thread_data) {
        free(threads);
        free(thread_data);
        return false;
    }

    // Reset feedback state
    pthread_mutex_lock(&feedback_mutex);
    feedback_ready = false;
    memset(&global_feedback, 0, sizeof(MeasurementFeedback));
    pthread_mutex_unlock(&feedback_mutex);

    // Initialize thread data with hardware profile
    for (size_t i = 0; i < num_threads; i++) {
        if (!initialize_thread_data(&thread_data[i], state, qubit_indices,
                                  num_qubits, type, i, num_threads,
                                  hw_profile)) {
            // Cleanup previously created threads
            for (size_t j = 0; j < i; j++) {
                pthread_cancel(threads[j]);
                pthread_join(threads[j], NULL);
                cleanup_thread_data(&thread_data[j]);
            }
            free(threads);
            free(thread_data);
            return false;
        }

        if (pthread_create(&threads[i], NULL, measurement_thread, &thread_data[i])) {
            // Cleanup on thread creation failure
            for (size_t j = 0; j < i; j++) {
                pthread_cancel(threads[j]);
                pthread_join(threads[j], NULL);
                cleanup_thread_data(&thread_data[j]);
            }
            cleanup_thread_data(&thread_data[i]);
            free(threads);
            free(thread_data);
            return false;
        }
    }

    // Wait for all threads to complete
    for (size_t i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
        
        // Copy results and confidence values
        size_t start_idx = thread_data[i].start_index;
        size_t end_idx = thread_data[i].end_index;
        for (size_t j = start_idx; j < end_idx; j++) {
            results[j] = thread_data[i].results[j - start_idx];
            
            // Apply confidence weighting
            results[j] *= thread_data[i].confidences[j - start_idx];
        }
        
        cleanup_thread_data(&thread_data[i]);
    }

    free(threads);
    free(thread_data);
    return true;
}

static void* measurement_thread(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    
    // Allocate operator array
    PauliOperator* operators = malloc(data->qubits_per_stabilizer * sizeof(PauliOperator));
    if (!operators) {
        return NULL;
    }

    // Configure operators based on stabilizer type and hardware profile
    for (size_t i = 0; i < data->qubits_per_stabilizer; i++) {
        switch (data->type) {
            case STABILIZER_PLAQUETTE:
                operators[i] = PAULI_Z;
                break;
            case STABILIZER_VERTEX:
                operators[i] = PAULI_X;
                break;
            default:
                free(operators);
                return NULL;
        }
    }

    // Initialize measurement tracking
    MeasurementResult* measurements = calloc(data->qubits_per_stabilizer, 
                                           sizeof(MeasurementResult));
    if (!measurements) {
        free(operators);
        return NULL;
    }

    // Perform measurements for assigned range
    for (size_t i = data->start_index; i < data->end_index; i++) {
        size_t base_idx = i * data->qubits_per_stabilizer;
        
        // Extract qubit indices for this stabilizer
        size_t* stabilizer_qubits = malloc(data->qubits_per_stabilizer * sizeof(size_t));
        if (!stabilizer_qubits) {
            free(measurements);
            free(operators);
            return NULL;
        }
        
        for (size_t j = 0; j < data->qubits_per_stabilizer; j++) {
            stabilizer_qubits[j] = data->qubit_indices[base_idx + j];
        }

        // Measure stabilizer with confidence tracking
        double result = 1.0;
        double confidence = 1.0;
        
        for (size_t j = 0; j < data->qubits_per_stabilizer; j++) {
            measurements[j].value = 0.0;
            measurements[j].confidence = 0.0;
            measurements[j].error_rate = 0.0;
            bool success = false;

            // Apply hardware-specific corrections
            double hw_correction = data->measurement_fidelity * 
                                 (1.0 - data->noise_level) *
                                 data->gate_fidelity;

            switch (operators[j]) {
                case PAULI_X:
                    success = measure_pauli_x(data->state, stabilizer_qubits[j], 
                                            &measurements[j].value);
                    break;
                case PAULI_Y:
                    success = measure_pauli_y(data->state, stabilizer_qubits[j], 
                                            &measurements[j].value);
                    break;
                case PAULI_Z:
                    success = measure_pauli_z(data->state, stabilizer_qubits[j], 
                                            &measurements[j].value);
                    break;
                default:
                    success = false;
            }

            if (!success) {
                free(stabilizer_qubits);
                free(measurements);
                free(operators);
                return NULL;
            }

            // Calculate measurement confidence
            measurements[j].confidence = calculate_measurement_confidence(data, 
                                                                       stabilizer_qubits[j]);
            measurements[j].error_rate = get_error_rate(stabilizer_qubits[j]);
            
            // Apply hardware correction
            measurements[j].value *= hw_correction;
            
            result *= measurements[j].value;
            confidence *= measurements[j].confidence;
            
            // Trigger fast feedback if needed
            if (should_trigger_feedback(&measurements[j])) {
                trigger_fast_feedback(data, &measurements[j]);
            }
        }

        // Store results with confidence
        data->results[i - data->start_index] = result;
        data->confidences[i - data->start_index] = confidence;
        data->error_rates[i - data->start_index] = get_aggregate_error_rate(measurements,
                                                                          data->qubits_per_stabilizer);

        // Check for feedback and update workload if needed
        pthread_mutex_lock(&feedback_mutex);
        if (feedback_ready) {
            update_thread_workload(data, &global_feedback);
            feedback_ready = false;
        }
        pthread_mutex_unlock(&feedback_mutex);

        free(stabilizer_qubits);
    }

    free(measurements);
    free(operators);
    return NULL;
}

static bool initialize_thread_data(ThreadData* data,
                                const quantum_state* state,
                                const size_t* qubit_indices,
                                size_t num_qubits,
                                StabilizerType type,
                                size_t thread_id,
                                size_t total_threads,
                                const HardwareProfile* hw_profile) {
    if (!data || !state || !qubit_indices || !hw_profile) {
        return false;
    }

    // Get hardware-specific qubit reliability factors
    double* qubit_weights = malloc(num_qubits * sizeof(double));
    if (!qubit_weights) {
        return false;
    }
    
    for (size_t i = 0; i < num_qubits; i++) {
        qubit_weights[i] = get_qubit_reliability(qubit_indices[i]) * 
                          get_measurement_fidelity(qubit_indices[i]);
    }
    
    // Calculate workload distribution based on qubit weights
    size_t qubits_per_stabilizer = 4;  // Assuming 4 qubits per stabilizer
    size_t num_stabilizers = num_qubits / qubits_per_stabilizer;
    
    // Calculate total weight for each stabilizer
    double* stabilizer_weights = malloc(num_stabilizers * sizeof(double));
    if (!stabilizer_weights) {
        free(qubit_weights);
        return false;
    }
    
    for (size_t i = 0; i < num_stabilizers; i++) {
        stabilizer_weights[i] = 0.0;
        for (size_t j = 0; j < qubits_per_stabilizer; j++) {
            size_t qubit_idx = i * qubits_per_stabilizer + j;
            stabilizer_weights[i] += qubit_weights[qubit_idx];
        }
    }
    
    // Distribute stabilizers to balance total weight per thread
    double total_weight = 0.0;
    for (size_t i = 0; i < num_stabilizers; i++) {
        total_weight += stabilizer_weights[i];
    }
    
    double target_weight = total_weight / total_threads;
    double current_weight = 0.0;
    size_t current_stabilizer = 0;
    
    // Find start index for this thread
    for (size_t i = 0; i < thread_id; i++) {
        while (current_weight < (i + 1) * target_weight && 
               current_stabilizer < num_stabilizers) {
            current_weight += stabilizer_weights[current_stabilizer++];
        }
    }
    data->start_index = current_stabilizer;
    
    // Find end index for this thread
    while (current_weight < (thread_id + 1) * target_weight && 
           current_stabilizer < num_stabilizers) {
        current_weight += stabilizer_weights[current_stabilizer++];
    }
    data->end_index = current_stabilizer;
    
    free(stabilizer_weights);
    free(qubit_weights);

    // Initialize measurement data
    data->state = state;
    data->qubit_indices = qubit_indices;
    data->type = type;
    data->qubits_per_stabilizer = qubits_per_stabilizer;
    data->hw_profile = hw_profile;
    
    // Allocate measurement tracking arrays
    size_t num_measurements = data->end_index - data->start_index;
    data->results = calloc(num_measurements, sizeof(double));
    data->confidences = calloc(num_measurements, sizeof(double));
    data->error_rates = calloc(num_measurements, sizeof(double));
    
    if (!data->results || !data->confidences || !data->error_rates) {
        free(data->results);
        free(data->confidences);
        free(data->error_rates);
        return false;
    }
    
    // Initialize thread-specific hardware state
    data->measurement_fidelity = get_measurement_fidelity_for_thread(thread_id);
    data->gate_fidelity = get_gate_fidelity_for_thread(thread_id);
    data->noise_level = get_noise_level_for_thread(thread_id);

    return true;
}

static void cleanup_thread_data(ThreadData* data) {
    if (data) {
        free(data->results);
        free(data->confidences);
        free(data->error_rates);
        memset(data, 0, sizeof(ThreadData));
    }
}

static void update_thread_workload(ThreadData* data,
                                 const MeasurementFeedback* feedback) {
    if (!data || !feedback) {
        return;
    }

    // Adjust measurement parameters based on feedback
    data->measurement_fidelity *= feedback->fidelity_adjustment;
    data->gate_fidelity *= feedback->gate_adjustment;
    
    // Update error thresholds
    for (size_t i = 0; i < (data->end_index - data->start_index); i++) {
        data->error_rates[i] *= feedback->error_scale;
    }
}

static double calculate_measurement_confidence(const ThreadData* data,
                                            size_t qubit_index) {
    if (!data) {
        return 0.0;
    }

    // Get hardware-specific factors
    double base_confidence = get_base_confidence(qubit_index);
    double stability = get_qubit_stability(qubit_index);
    double coherence = get_coherence_factor(qubit_index);
    
    // Calculate confidence with hardware weighting
    double confidence = base_confidence * stability * coherence;
    
    // Apply measurement fidelity adjustment
    confidence *= data->measurement_fidelity;
    
    // Scale by noise level
    confidence *= (1.0 - data->noise_level);
    
    return confidence;
}

static void trigger_fast_feedback(const ThreadData* data,
                                const MeasurementResult* result) {
    if (!data || !result) {
        return;
    }

    pthread_mutex_lock(&feedback_mutex);
    
    // Update global feedback based on measurement result
    global_feedback.fidelity_adjustment = calculate_fidelity_adjustment(result);
    global_feedback.gate_adjustment = calculate_gate_adjustment(result);
    global_feedback.error_scale = calculate_error_scale(result);
    
    feedback_ready = true;
    pthread_cond_broadcast(&feedback_cond);
    
    pthread_mutex_unlock(&feedback_mutex);
}
