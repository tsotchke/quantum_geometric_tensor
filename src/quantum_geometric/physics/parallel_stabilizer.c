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
#include <complex.h>
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
            if (should_trigger_stabilizer_feedback(&measurements[j])) {
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
        qubit_weights[i] = get_stabilizer_qubit_reliability(qubit_indices[i]) *
                          get_stabilizer_measurement_fidelity(qubit_indices[i]);
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

// ============================================================================
// Hardware Query Functions Implementation
// ============================================================================

// Physical qubit characterization based on superconducting qubit models
// Reference values from IBM/Google/Rigetti calibration data (2024)
typedef struct {
    double t1_us;              // T1 relaxation time in microseconds
    double t2_us;              // T2 dephasing time in microseconds
    double readout_fidelity;   // Single-shot readout fidelity
    double gate_fidelity_1q;   // Single-qubit gate fidelity
    double gate_fidelity_2q;   // Two-qubit gate fidelity
    double frequency_ghz;      // Qubit frequency in GHz
    double anharmonicity_mhz;  // Anharmonicity in MHz
    double residual_zz_khz;    // Residual ZZ coupling in kHz
} QubitCalibration;

// Default calibration (representative of current superconducting qubits)
static const QubitCalibration default_calibration = {
    .t1_us = 100.0,            // 100 μs T1 (typical for transmon)
    .t2_us = 80.0,             // 80 μs T2 (often limited by T1)
    .readout_fidelity = 0.985, // 98.5% readout fidelity
    .gate_fidelity_1q = 0.9995,// 99.95% single-qubit gate fidelity
    .gate_fidelity_2q = 0.995, // 99.5% two-qubit gate fidelity
    .frequency_ghz = 5.0,      // 5 GHz qubit frequency
    .anharmonicity_mhz = -300, // -300 MHz anharmonicity
    .residual_zz_khz = 50.0    // 50 kHz residual ZZ
};

// Measurement timing parameters
static const double readout_time_us = 1.0;     // 1 μs readout
static const double gate_time_1q_ns = 25.0;    // 25 ns single-qubit gate
static const double gate_time_2q_ns = 200.0;   // 200 ns two-qubit gate

// Physical model for qubit-dependent variations
// Models spatial inhomogeneity across the chip and frequency crowding effects

static double compute_t1_variation(size_t qubit_index) {
    // T1 varies across chip due to material defects, TLS, and fabrication
    // Model: Gaussian variation with ~10% standard deviation
    // Also includes position-dependent loss mechanisms
    double chip_position = (double)(qubit_index % 16) / 15.0;  // Assume 4x4 grid
    double edge_effect = 1.0 - 0.1 * fabs(chip_position - 0.5) * 2.0;  // Edge qubits slightly worse

    // Pseudo-random but deterministic variation based on qubit index
    double phase = (double)qubit_index * 2.71828;  // Use e as irrational multiplier
    double variation = 1.0 + 0.15 * sin(phase) * cos(phase * 1.41421);  // ~15% spread

    return default_calibration.t1_us * edge_effect * variation;
}

static double compute_t2_variation(size_t qubit_index) {
    // T2 is bounded by T1 and affected by dephasing from flux noise
    double t1 = compute_t1_variation(qubit_index);
    double t2_max = 2.0 * t1;  // T2 <= 2*T1 (fundamental limit)

    // Additional dephasing from 1/f flux noise, TLS, and crosstalk
    double phase = (double)qubit_index * 3.14159;
    double dephasing_factor = 0.7 + 0.2 * cos(phase * 0.618);  // Golden ratio phase

    double t2 = t1 * dephasing_factor;
    return t2 < t2_max ? t2 : t2_max;
}

static double compute_readout_fidelity_variation(size_t qubit_index) {
    // Readout fidelity depends on:
    // 1. Readout resonator coupling (κ)
    // 2. Purcell decay through readout
    // 3. State preparation errors
    // 4. Discrimination threshold optimization

    double base = default_calibration.readout_fidelity;

    // Position-dependent coupling strength variation
    double coupling_variation = 1.0 + 0.02 * sin((double)qubit_index * 0.5);

    // Frequency-dependent Purcell effect (qubits near resonator are worse)
    double frequency_offset = 0.1 * sin((double)qubit_index * 0.3);
    double purcell_factor = 1.0 - 0.01 * exp(-frequency_offset * frequency_offset);

    // Thermal population correction (ground state preparation)
    double thermal_correction = 0.995;  // ~0.5% thermal excitation at 20mK

    double fidelity = base * coupling_variation * purcell_factor * thermal_correction;
    return fidelity > 0.9 ? fidelity : 0.9;  // Floor at 90%
}

static double compute_gate_error_variation(size_t qubit_index, bool is_two_qubit) {
    // Gate errors from:
    // 1. Coherence-limited (T1, T2)
    // 2. Control pulse errors (amplitude, frequency, timing)
    // 3. Leakage to non-computational states
    // 4. Crosstalk from neighboring qubits

    double t1 = compute_t1_variation(qubit_index);
    double t2 = compute_t2_variation(qubit_index);

    double gate_time_us = is_two_qubit ?
        gate_time_2q_ns / 1000.0 : gate_time_1q_ns / 1000.0;

    // Coherence-limited error: 1 - exp(-t_gate/T_coherence)
    double coherence_error = 1.0 - exp(-gate_time_us / t2);

    // Control error (irreducible hardware limit)
    double control_error = is_two_qubit ? 0.002 : 0.0002;

    // Leakage error (scales with gate time and anharmonicity)
    double leakage_error = is_two_qubit ? 0.001 : 0.0001;

    // Crosstalk error (position-dependent)
    double crosstalk_error = 0.0005 * (1.0 + 0.5 * sin((double)qubit_index * 0.7));

    double total_error = coherence_error + control_error + leakage_error + crosstalk_error;

    return total_error < 0.1 ? total_error : 0.1;  // Cap at 10%
}

// Renamed to avoid conflict with error_correlation_hardware.c (this is stabilizer-specific)
double get_stabilizer_qubit_reliability(size_t qubit_index) {
    // Reliability metric combines multiple physical factors:
    // 1. Coherence times (T1, T2)
    // 2. Gate fidelities
    // 3. Readout fidelity
    // 4. Historical stability (drift)

    double t1 = compute_t1_variation(qubit_index);
    double t2 = compute_t2_variation(qubit_index);

    // Coherence factor: normalized by typical gate sequence length (~100 gates)
    double typical_circuit_time_us = 100 * gate_time_1q_ns / 1000.0;
    double coherence_reliability = exp(-typical_circuit_time_us / t2);

    // Gate fidelity contribution
    double gate_error = compute_gate_error_variation(qubit_index, false);
    double gate_reliability = 1.0 - gate_error;

    // Readout contribution
    double readout = compute_readout_fidelity_variation(qubit_index);

    // Stability factor (models drift between calibrations)
    // Assumes ~1% drift over 24 hours, linear decay
    double hours_since_calibration = 2.0;  // Typical
    double stability = 1.0 - 0.01 * hours_since_calibration / 24.0;

    // Combined reliability
    double reliability = coherence_reliability * gate_reliability * readout * stability;

    return reliability > 0.5 ? reliability : 0.5;  // Floor at 50%
}

// Renamed to avoid conflict with error_prediction.c (this is stabilizer-specific)
double get_stabilizer_measurement_fidelity(size_t qubit_index) {
    return compute_readout_fidelity_variation(qubit_index);
}

double get_error_rate(size_t qubit_index) {
    // Total error rate per operation
    double gate_error = compute_gate_error_variation(qubit_index, false);
    double readout_error = 1.0 - compute_readout_fidelity_variation(qubit_index);

    // Combine assuming independent errors
    return gate_error + readout_error - gate_error * readout_error;
}

double get_base_confidence(size_t qubit_index) {
    // Confidence in measurement based on signal-to-noise ratio
    double readout = compute_readout_fidelity_variation(qubit_index);
    double t1 = compute_t1_variation(qubit_index);

    // T1 decay during readout reduces confidence
    double decay_during_readout = exp(-readout_time_us / t1);

    // State discrimination confidence (depends on readout SNR)
    double snr_factor = 0.95 + 0.05 * sin((double)qubit_index * 0.4);

    return readout * decay_during_readout * snr_factor;
}

double get_qubit_stability(size_t qubit_index) {
    // Stability measures how consistent qubit properties are over time
    // Affected by: TLS fluctuators, cosmic rays, temperature drift

    double t1 = compute_t1_variation(qubit_index);
    double base_stability = 0.98;  // 98% baseline

    // T1 fluctuations (correlated with TLS activity)
    double t1_stability = 1.0 - 0.02 * exp(-t1 / 50.0);  // Better T1 = more stable

    // Position-dependent cosmic ray susceptibility
    double position = (double)(qubit_index % 16) / 15.0;
    double cosmic_factor = 1.0 - 0.005 * (0.5 - fabs(position - 0.5));

    return base_stability * t1_stability * cosmic_factor;
}

double get_coherence_factor(size_t qubit_index) {
    // Coherence factor for error correction purposes
    double t1 = compute_t1_variation(qubit_index);
    double t2 = compute_t2_variation(qubit_index);

    // Effective coherence for typical syndrome extraction circuit
    double syndrome_time_us = 4 * gate_time_2q_ns / 1000.0;  // ~4 CNOT gates

    double t1_factor = exp(-syndrome_time_us / t1);
    double t2_factor = exp(-syndrome_time_us / t2);

    return sqrt(t1_factor * t2_factor);  // Geometric mean
}

double get_measurement_fidelity_for_thread(size_t thread_id) {
    // Thread contention can affect timing precision
    // Models jitter in measurement timing from CPU scheduling

    double base = default_calibration.readout_fidelity;

    // Timing jitter increases with thread count
    double jitter_factor = 1.0 - 0.001 * (double)thread_id;

    // Memory bandwidth contention
    double bandwidth_factor = 1.0 - 0.0005 * (double)thread_id;

    return base * jitter_factor * bandwidth_factor;
}

double get_gate_fidelity_for_thread(size_t thread_id) {
    // Gate fidelity is hardware-intrinsic, less affected by threading
    // Small effect from control signal synchronization

    double base = default_calibration.gate_fidelity_1q;

    // Synchronization overhead
    double sync_factor = 1.0 - 0.0002 * (double)thread_id;

    return base * sync_factor;
}

double get_noise_level_for_thread(size_t thread_id) {
    // Effective noise level considering processing overhead

    double base_noise = 1.0 - default_calibration.readout_fidelity;

    // Additional effective noise from timing errors
    double timing_noise = 0.0001 * (double)thread_id;

    // Quantization noise from parallel data handling
    double quant_noise = 0.00005 * (double)thread_id;

    return base_noise + timing_noise + quant_noise;
}

double get_aggregate_error_rate(const MeasurementResult* measurements, size_t count) {
    if (!measurements || count == 0) {
        return 0.0;
    }

    double total_error = 0.0;
    double total_weight = 0.0;

    for (size_t i = 0; i < count; i++) {
        double weight = measurements[i].confidence;
        total_error += measurements[i].error_rate * weight;
        total_weight += weight;
    }

    if (total_weight > 0.0) {
        return total_error / total_weight;
    }
    return 0.0;
}

// Renamed to avoid conflict with error_prediction.c (this takes MeasurementResult, that takes PredictionHistory)
bool should_trigger_stabilizer_feedback(const MeasurementResult* result) {
    if (!result) {
        return false;
    }

    // Trigger feedback if:
    // 1. Error rate exceeds threshold
    // 2. Confidence drops below threshold
    // 3. Measurement value is unexpected

    const double error_threshold = 0.01;
    const double confidence_threshold = 0.9;

    if (result->error_rate > error_threshold) {
        return true;
    }

    if (result->confidence < confidence_threshold) {
        return true;
    }

    // Check for unexpected measurement (should be close to +/-1 for Pauli)
    if (fabs(fabs(result->value) - 1.0) > 0.1) {
        return true;
    }

    return false;
}

double calculate_fidelity_adjustment(const MeasurementResult* result) {
    if (!result) {
        return 1.0;
    }

    // Reduce fidelity estimate if measurement is noisy
    double adjustment = 1.0;

    if (result->error_rate > 0.005) {
        adjustment *= (1.0 - result->error_rate);
    }

    if (result->confidence < 0.95) {
        adjustment *= result->confidence;
    }

    return adjustment;
}

double calculate_gate_adjustment(const MeasurementResult* result) {
    if (!result) {
        return 1.0;
    }

    // Gate errors typically scale with measurement errors
    double adjustment = 1.0 - 0.5 * result->error_rate;
    return adjustment > 0.8 ? adjustment : 0.8;
}

double calculate_error_scale(const MeasurementResult* result) {
    if (!result) {
        return 1.0;
    }

    // Scale error estimates based on observed confidence
    if (result->confidence < 0.9) {
        return 1.5;  // Increase error estimate
    } else if (result->confidence > 0.99) {
        return 0.8;  // Decrease error estimate
    }

    return 1.0;
}

// ============================================================================
// Pauli Measurement Functions (renamed to avoid conflicts with heavy_hex_surface_code.c)
// ============================================================================

// Renamed to avoid conflict with heavy_hex_surface_code.c
bool measure_pauli_x_parallel(const quantum_state* state, size_t qubit_index, double* result) {
    if (!state || !result) {
        return false;
    }

    // Measure <X> = <psi|X|psi>
    // For a single qubit at qubit_index in the computational basis:
    // X|0> = |1>, X|1> = |0>
    // <X> = 2 * Re(c_0* c_1) where c_0, c_1 are amplitudes

    // Access the quantum state coordinates (amplitudes)
    if (!state->coordinates || qubit_index >= state->num_qubits) {
        return false;
    }

    size_t dim = state->dimension;
    size_t mask = 1UL << qubit_index;

    double expectation = 0.0;

    // Sum over all basis states
    for (size_t i = 0; i < dim; i++) {
        size_t j = i ^ mask;  // Flip the qubit_index bit
        if (i < j) {  // Only count each pair once
            // <X> contribution from |i><j| + |j><i|
            // coordinates is ComplexFloat, which has .real and .imag fields
            double real_part = state->coordinates[i].real * state->coordinates[j].real +
                              state->coordinates[i].imag * state->coordinates[j].imag;
            expectation += 2.0 * real_part;
        }
    }

    *result = expectation;
    return true;
}

// Renamed to avoid conflict with heavy_hex_surface_code.c
bool measure_pauli_y_parallel(const quantum_state* state, size_t qubit_index, double* result) {
    if (!state || !result) {
        return false;
    }

    // Measure <Y> = <psi|Y|psi>
    // Y|0> = i|1>, Y|1> = -i|0>
    // <Y> = 2 * Im(c_1* c_0) = -2 * Im(c_0* c_1)

    if (!state->coordinates || qubit_index >= state->num_qubits) {
        return false;
    }

    size_t dim = state->dimension;
    size_t mask = 1UL << qubit_index;

    double expectation = 0.0;

    for (size_t i = 0; i < dim; i++) {
        size_t j = i ^ mask;
        if (i < j) {
            // For Y: contribution is imaginary part with sign depending on which bit is set
            double imag_part = state->coordinates[i].real * state->coordinates[j].imag -
                              state->coordinates[i].imag * state->coordinates[j].real;

            // Sign depends on whether qubit_index bit is 0 or 1 in i
            if (i & mask) {
                expectation -= 2.0 * imag_part;
            } else {
                expectation += 2.0 * imag_part;
            }
        }
    }

    *result = expectation;
    return true;
}

// Renamed to avoid conflict with heavy_hex_surface_code.c
bool measure_pauli_z_parallel(const quantum_state* state, size_t qubit_index, double* result) {
    if (!state || !result) {
        return false;
    }

    // Measure <Z> = <psi|Z|psi>
    // Z|0> = |0>, Z|1> = -|1>
    // <Z> = sum_i |c_i|^2 * (-1)^{bit qubit_index of i}

    if (!state->coordinates || qubit_index >= state->num_qubits) {
        return false;
    }

    size_t dim = state->dimension;
    size_t mask = 1UL << qubit_index;

    double expectation = 0.0;

    for (size_t i = 0; i < dim; i++) {
        double prob = state->coordinates[i].real * state->coordinates[i].real +
                     state->coordinates[i].imag * state->coordinates[i].imag;

        // +1 if bit is 0, -1 if bit is 1
        if (i & mask) {
            expectation -= prob;
        } else {
            expectation += prob;
        }
    }

    *result = expectation;
    return true;
}
