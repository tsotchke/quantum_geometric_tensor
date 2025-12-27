/**
 * @file error_correlation_hardware.c
 * @brief Hardware-aware correlation helper functions
 *
 * Provides hardware-specific factors for error correlation analysis.
 * These functions integrate with the hardware abstraction layer to
 * obtain real-time hardware characteristics and calibration data.
 */

#include "quantum_geometric/physics/error_correlation.h"
#include "quantum_geometric/hardware/quantum_hardware_abstraction.h"
#include "quantum_geometric/hardware/quantum_hardware_types.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

// ============================================================================
// Hardware Correlation State
// ============================================================================

/**
 * Hardware correlation state structure
 * Maintains cached hardware information for correlation analysis
 */
typedef struct HardwareCorrelationState {
    bool initialized;

    // Hardware reference
    void* hardware_handle;           // Handle to hardware abstraction layer
    HardwareBackendType backend_type;

    // Cached capabilities
    size_t num_qubits;
    double coherence_time_us;        // T1 coherence time in microseconds
    double dephasing_time_us;        // T2 dephasing time in microseconds
    double gate_time_us;             // Single gate time in microseconds
    double readout_time_us;          // Measurement time in microseconds
    double readout_error;            // Readout error rate

    // Per-qubit calibration data
    double* qubit_t1_times;          // T1 times per qubit
    double* qubit_t2_times;          // T2 times per qubit
    double* qubit_readout_errors;    // Readout errors per qubit
    double* qubit_gate_errors;       // Gate errors per qubit

    // Connectivity and crosstalk
    double* connectivity_matrix;     // Qubit connectivity [i*num_qubits + j]
    double* crosstalk_matrix;        // Crosstalk coefficients [i*num_qubits + j]

    // Temporal tracking
    double last_calibration_time;    // Last calibration timestamp
    double calibration_drift_rate;   // Rate of parameter drift

    // Feedback state
    double feedback_threshold;       // Threshold for triggering feedback
    size_t feedback_count;           // Number of feedback events triggered
    double* feedback_history;        // Recent correlation values for feedback
    size_t feedback_history_size;
    size_t feedback_history_capacity;

    // Thread safety
    pthread_mutex_t state_mutex;
} HardwareCorrelationState;

// Global hardware correlation state
static HardwareCorrelationState g_hw_corr_state = {
    .initialized = false,
    .hardware_handle = NULL,
    .num_qubits = 0
};

// ============================================================================
// Forward Declarations
// ============================================================================

void cleanup_correlation_hardware(void);

// ============================================================================
// Internal Helper Functions
// ============================================================================

/**
 * Compute exponential decay factor
 */
static inline double exponential_decay(double time, double characteristic_time) {
    if (characteristic_time <= 0.0) return 0.0;
    return exp(-time / characteristic_time);
}

/**
 * Compute Gaussian decay factor (for dephasing)
 */
static inline double gaussian_decay(double time, double characteristic_time) {
    if (characteristic_time <= 0.0) return 0.0;
    double ratio = time / characteristic_time;
    return exp(-0.5 * ratio * ratio);
}

/**
 * Update calibration data from hardware
 */
static void refresh_calibration_data(HardwareCorrelationState* state) {
    if (!state || !state->hardware_handle) return;

    HardwareCapabilities* caps = get_hardware_capabilities(state->hardware_handle);
    if (!caps) return;

    // Update global parameters
    state->coherence_time_us = caps->coherence_time;
    state->gate_time_us = caps->gate_time;
    state->readout_time_us = caps->readout_time;

    // Update connectivity if available
    if (caps->connectivity && caps->connectivity_size > 0) {
        size_t conn_size = state->num_qubits * state->num_qubits;
        if (state->connectivity_matrix) {
            size_t copy_size = (conn_size < caps->connectivity_size) ?
                              conn_size : caps->connectivity_size;
            memcpy(state->connectivity_matrix, caps->connectivity,
                   copy_size * sizeof(double));
        }
    }

    cleanup_capabilities(caps);
}

// ============================================================================
// Initialization and Cleanup
// ============================================================================

/**
 * Initialize hardware correlation state
 */
bool init_correlation_hardware(void* hardware_handle, size_t num_qubits) {
    if (g_hw_corr_state.initialized) {
        return true;  // Already initialized
    }

    pthread_mutex_init(&g_hw_corr_state.state_mutex, NULL);
    pthread_mutex_lock(&g_hw_corr_state.state_mutex);

    g_hw_corr_state.hardware_handle = hardware_handle;
    g_hw_corr_state.num_qubits = num_qubits;

    // Allocate per-qubit arrays
    if (num_qubits > 0) {
        g_hw_corr_state.qubit_t1_times = calloc(num_qubits, sizeof(double));
        g_hw_corr_state.qubit_t2_times = calloc(num_qubits, sizeof(double));
        g_hw_corr_state.qubit_readout_errors = calloc(num_qubits, sizeof(double));
        g_hw_corr_state.qubit_gate_errors = calloc(num_qubits, sizeof(double));
        g_hw_corr_state.connectivity_matrix = calloc(num_qubits * num_qubits, sizeof(double));
        g_hw_corr_state.crosstalk_matrix = calloc(num_qubits * num_qubits, sizeof(double));

        if (!g_hw_corr_state.qubit_t1_times || !g_hw_corr_state.qubit_t2_times ||
            !g_hw_corr_state.qubit_readout_errors || !g_hw_corr_state.qubit_gate_errors ||
            !g_hw_corr_state.connectivity_matrix || !g_hw_corr_state.crosstalk_matrix) {
            pthread_mutex_unlock(&g_hw_corr_state.state_mutex);
            cleanup_correlation_hardware();
            return false;
        }

        // Initialize with typical values
        for (size_t i = 0; i < num_qubits; i++) {
            g_hw_corr_state.qubit_t1_times[i] = 100.0;     // 100 μs typical T1
            g_hw_corr_state.qubit_t2_times[i] = 50.0;      // 50 μs typical T2
            g_hw_corr_state.qubit_readout_errors[i] = 0.01; // 1% readout error
            g_hw_corr_state.qubit_gate_errors[i] = 0.001;   // 0.1% gate error
        }

        // Initialize crosstalk model (nearest-neighbor decay)
        for (size_t i = 0; i < num_qubits; i++) {
            for (size_t j = 0; j < num_qubits; j++) {
                if (i == j) {
                    g_hw_corr_state.crosstalk_matrix[i * num_qubits + j] = 0.0;
                } else {
                    // Crosstalk decays with distance
                    double dist = (double)abs((int)i - (int)j);
                    g_hw_corr_state.crosstalk_matrix[i * num_qubits + j] =
                        0.01 * exp(-dist / 3.0);  // 1% base, decays over 3 qubits
                }
            }
        }
    }

    // Initialize feedback tracking
    g_hw_corr_state.feedback_threshold = 0.7;
    g_hw_corr_state.feedback_count = 0;
    g_hw_corr_state.feedback_history_capacity = 100;
    g_hw_corr_state.feedback_history_size = 0;
    g_hw_corr_state.feedback_history = calloc(g_hw_corr_state.feedback_history_capacity,
                                               sizeof(double));

    // Set default timing parameters
    g_hw_corr_state.coherence_time_us = 100.0;
    g_hw_corr_state.dephasing_time_us = 50.0;
    g_hw_corr_state.gate_time_us = 0.1;
    g_hw_corr_state.readout_time_us = 1.0;
    g_hw_corr_state.readout_error = 0.01;
    g_hw_corr_state.calibration_drift_rate = 0.001;  // 0.1% drift per second

    // Refresh from actual hardware if available
    if (hardware_handle) {
        refresh_calibration_data(&g_hw_corr_state);
    }

    g_hw_corr_state.initialized = true;
    pthread_mutex_unlock(&g_hw_corr_state.state_mutex);

    return true;
}

/**
 * Cleanup hardware correlation state
 */
void cleanup_correlation_hardware(void) {
    if (!g_hw_corr_state.initialized) return;

    pthread_mutex_lock(&g_hw_corr_state.state_mutex);

    free(g_hw_corr_state.qubit_t1_times);
    free(g_hw_corr_state.qubit_t2_times);
    free(g_hw_corr_state.qubit_readout_errors);
    free(g_hw_corr_state.qubit_gate_errors);
    free(g_hw_corr_state.connectivity_matrix);
    free(g_hw_corr_state.crosstalk_matrix);
    free(g_hw_corr_state.feedback_history);

    g_hw_corr_state.qubit_t1_times = NULL;
    g_hw_corr_state.qubit_t2_times = NULL;
    g_hw_corr_state.qubit_readout_errors = NULL;
    g_hw_corr_state.qubit_gate_errors = NULL;
    g_hw_corr_state.connectivity_matrix = NULL;
    g_hw_corr_state.crosstalk_matrix = NULL;
    g_hw_corr_state.feedback_history = NULL;

    g_hw_corr_state.num_qubits = 0;
    g_hw_corr_state.initialized = false;

    pthread_mutex_unlock(&g_hw_corr_state.state_mutex);
    pthread_mutex_destroy(&g_hw_corr_state.state_mutex);
}

/**
 * Update calibration data for a specific qubit
 */
void update_qubit_calibration(size_t qubit_index,
                             double t1_time,
                             double t2_time,
                             double readout_error,
                             double gate_error) {
    if (!g_hw_corr_state.initialized) return;
    if (qubit_index >= g_hw_corr_state.num_qubits) return;

    pthread_mutex_lock(&g_hw_corr_state.state_mutex);

    if (t1_time > 0) g_hw_corr_state.qubit_t1_times[qubit_index] = t1_time;
    if (t2_time > 0) g_hw_corr_state.qubit_t2_times[qubit_index] = t2_time;
    if (readout_error >= 0) g_hw_corr_state.qubit_readout_errors[qubit_index] = readout_error;
    if (gate_error >= 0) g_hw_corr_state.qubit_gate_errors[qubit_index] = gate_error;

    pthread_mutex_unlock(&g_hw_corr_state.state_mutex);
}

// ============================================================================
// Hardware Reliability Functions
// ============================================================================

double get_hardware_reliability_factor(void) {
    if (!g_hw_corr_state.initialized) {
        // Return reasonable default when not initialized
        return 0.99;
    }

    pthread_mutex_lock(&g_hw_corr_state.state_mutex);

    // Compute overall reliability from multiple factors:
    // 1. Average qubit coherence relative to gate time
    // 2. Average gate fidelity
    // 3. Average readout fidelity

    double avg_coherence_factor = 0.0;
    double avg_gate_fidelity = 0.0;
    double avg_readout_fidelity = 0.0;

    if (g_hw_corr_state.num_qubits > 0) {
        for (size_t i = 0; i < g_hw_corr_state.num_qubits; i++) {
            // Coherence factor: how many gates can we run within T2
            double gates_in_t2 = g_hw_corr_state.qubit_t2_times[i] /
                                g_hw_corr_state.gate_time_us;
            avg_coherence_factor += (gates_in_t2 > 1000) ? 1.0 :
                                   (gates_in_t2 / 1000.0);

            // Gate fidelity
            avg_gate_fidelity += 1.0 - g_hw_corr_state.qubit_gate_errors[i];

            // Readout fidelity
            avg_readout_fidelity += 1.0 - g_hw_corr_state.qubit_readout_errors[i];
        }

        avg_coherence_factor /= g_hw_corr_state.num_qubits;
        avg_gate_fidelity /= g_hw_corr_state.num_qubits;
        avg_readout_fidelity /= g_hw_corr_state.num_qubits;
    } else {
        avg_coherence_factor = 0.99;
        avg_gate_fidelity = 0.999;
        avg_readout_fidelity = 0.99;
    }

    // Combined reliability (geometric mean for multiplicative errors)
    double reliability = pow(avg_coherence_factor * avg_gate_fidelity *
                            avg_readout_fidelity, 1.0/3.0);

    pthread_mutex_unlock(&g_hw_corr_state.state_mutex);

    return fmin(1.0, fmax(0.0, reliability));
}

double get_noise_factor(void) {
    if (!g_hw_corr_state.initialized) {
        return 0.01;  // 1% default noise
    }

    pthread_mutex_lock(&g_hw_corr_state.state_mutex);

    // Noise factor combines multiple error sources:
    // 1. Gate errors
    // 2. Decoherence during operations
    // 3. Readout errors
    // 4. Crosstalk

    double total_noise = 0.0;

    if (g_hw_corr_state.num_qubits > 0) {
        // Average gate error contribution
        double avg_gate_error = 0.0;
        for (size_t i = 0; i < g_hw_corr_state.num_qubits; i++) {
            avg_gate_error += g_hw_corr_state.qubit_gate_errors[i];
        }
        avg_gate_error /= g_hw_corr_state.num_qubits;

        // Decoherence contribution (assuming typical circuit depth of 100 gates)
        double circuit_time = 100.0 * g_hw_corr_state.gate_time_us;
        double decoherence_prob = 1.0 - exponential_decay(circuit_time,
                                                          g_hw_corr_state.coherence_time_us);

        // Average readout error
        double avg_readout_error = 0.0;
        for (size_t i = 0; i < g_hw_corr_state.num_qubits; i++) {
            avg_readout_error += g_hw_corr_state.qubit_readout_errors[i];
        }
        avg_readout_error /= g_hw_corr_state.num_qubits;

        // Average crosstalk
        double avg_crosstalk = 0.0;
        size_t crosstalk_count = 0;
        for (size_t i = 0; i < g_hw_corr_state.num_qubits; i++) {
            for (size_t j = i + 1; j < g_hw_corr_state.num_qubits; j++) {
                avg_crosstalk += g_hw_corr_state.crosstalk_matrix[i * g_hw_corr_state.num_qubits + j];
                crosstalk_count++;
            }
        }
        if (crosstalk_count > 0) avg_crosstalk /= crosstalk_count;

        // Combine noise sources (not simply additive due to correlations)
        total_noise = 1.0 - (1.0 - avg_gate_error) *
                           (1.0 - decoherence_prob) *
                           (1.0 - avg_readout_error) *
                           (1.0 - avg_crosstalk);
    }

    pthread_mutex_unlock(&g_hw_corr_state.state_mutex);

    return fmin(1.0, fmax(0.0, total_noise));
}

double get_qubit_reliability(size_t qubit_index) {
    if (!g_hw_corr_state.initialized) {
        return 0.99;
    }

    pthread_mutex_lock(&g_hw_corr_state.state_mutex);

    double reliability = 0.99;

    if (qubit_index < g_hw_corr_state.num_qubits) {
        // Qubit reliability based on its specific characteristics
        double t1 = g_hw_corr_state.qubit_t1_times[qubit_index];
        double t2 = g_hw_corr_state.qubit_t2_times[qubit_index];
        double gate_err = g_hw_corr_state.qubit_gate_errors[qubit_index];
        double readout_err = g_hw_corr_state.qubit_readout_errors[qubit_index];

        // Coherence quality (T2/T1 ratio, ideally close to 2)
        double coherence_quality = (t1 > 0) ? fmin(t2 / t1, 2.0) / 2.0 : 0.5;

        // Gate quality
        double gate_quality = 1.0 - gate_err;

        // Readout quality
        double readout_quality = 1.0 - readout_err;

        reliability = pow(coherence_quality * gate_quality * readout_quality, 1.0/3.0);
    }

    pthread_mutex_unlock(&g_hw_corr_state.state_mutex);

    return fmin(1.0, fmax(0.0, reliability));
}

double get_gate_fidelity(void) {
    if (!g_hw_corr_state.initialized) {
        return 0.999;
    }

    pthread_mutex_lock(&g_hw_corr_state.state_mutex);

    double avg_fidelity = 0.999;

    if (g_hw_corr_state.num_qubits > 0 && g_hw_corr_state.qubit_gate_errors) {
        avg_fidelity = 0.0;
        for (size_t i = 0; i < g_hw_corr_state.num_qubits; i++) {
            avg_fidelity += 1.0 - g_hw_corr_state.qubit_gate_errors[i];
        }
        avg_fidelity /= g_hw_corr_state.num_qubits;
    }

    pthread_mutex_unlock(&g_hw_corr_state.state_mutex);

    return fmin(1.0, fmax(0.0, avg_fidelity));
}

double get_crosstalk_factor(size_t qubit1, size_t qubit2) {
    if (!g_hw_corr_state.initialized) {
        // Default: exponential decay with distance
        double distance = (double)abs((int)qubit1 - (int)qubit2);
        return 0.01 * exp(-distance / 3.0);
    }

    pthread_mutex_lock(&g_hw_corr_state.state_mutex);

    double crosstalk = 0.0;

    if (qubit1 < g_hw_corr_state.num_qubits &&
        qubit2 < g_hw_corr_state.num_qubits &&
        g_hw_corr_state.crosstalk_matrix) {
        crosstalk = g_hw_corr_state.crosstalk_matrix[
            qubit1 * g_hw_corr_state.num_qubits + qubit2];
    } else {
        // Fallback to distance-based model
        double distance = (double)abs((int)qubit1 - (int)qubit2);
        crosstalk = 0.01 * exp(-distance / 3.0);
    }

    pthread_mutex_unlock(&g_hw_corr_state.state_mutex);

    return fmin(1.0, fmax(0.0, crosstalk));
}

// ============================================================================
// Temporal Factors
// ============================================================================

double get_temporal_noise_factor(void) {
    if (!g_hw_corr_state.initialized) {
        return 0.02;  // Slightly higher than static noise
    }

    pthread_mutex_lock(&g_hw_corr_state.state_mutex);

    // Temporal noise includes drift and fluctuations
    double base_noise = get_noise_factor();

    // Add contribution from parameter drift since last calibration
    double drift_contribution = g_hw_corr_state.calibration_drift_rate;

    // Temporal noise is typically 20-50% higher than static noise
    double temporal_noise = base_noise * 1.3 + drift_contribution;

    pthread_mutex_unlock(&g_hw_corr_state.state_mutex);

    return fmin(1.0, fmax(0.0, temporal_noise));
}

double get_coherence_time_factor(void) {
    if (!g_hw_corr_state.initialized) {
        return 0.95;
    }

    pthread_mutex_lock(&g_hw_corr_state.state_mutex);

    // Coherence factor based on T1 and T2 relative to typical operation times
    // A good coherence factor means we can perform many operations within coherence time

    double typical_circuit_time = 100.0 * g_hw_corr_state.gate_time_us +
                                  g_hw_corr_state.readout_time_us;

    // T1 decay factor
    double t1_factor = exponential_decay(typical_circuit_time,
                                         g_hw_corr_state.coherence_time_us);

    // T2 decay factor (Gaussian for dephasing)
    double t2_factor = gaussian_decay(typical_circuit_time,
                                      g_hw_corr_state.dephasing_time_us);

    // Combined coherence factor
    double coherence_factor = sqrt(t1_factor * t2_factor);

    pthread_mutex_unlock(&g_hw_corr_state.state_mutex);

    return fmin(1.0, fmax(0.0, coherence_factor));
}

double get_vertex_stability(size_t vertex_index) {
    if (!g_hw_corr_state.initialized) {
        return 0.98;
    }

    pthread_mutex_lock(&g_hw_corr_state.state_mutex);

    // Vertex stability depends on associated qubits and their neighborhood
    // Vertices at boundaries or near defects have lower stability

    double stability = 0.98;

    if (g_hw_corr_state.num_qubits > 0) {
        // Map vertex to qubit index (assumes 1:1 or known mapping)
        size_t qubit_idx = vertex_index % g_hw_corr_state.num_qubits;

        // Base stability from qubit T2 time
        double t2 = g_hw_corr_state.qubit_t2_times[qubit_idx];
        double t2_factor = (t2 > 100.0) ? 1.0 : t2 / 100.0;

        // Reduce stability for boundary vertices (first and last)
        double boundary_factor = 1.0;
        if (qubit_idx == 0 || qubit_idx == g_hw_corr_state.num_qubits - 1) {
            boundary_factor = 0.95;  // 5% reduction at boundaries
        }

        // Account for crosstalk from neighbors
        double crosstalk_factor = 1.0;
        if (qubit_idx > 0) {
            crosstalk_factor -= g_hw_corr_state.crosstalk_matrix[
                (qubit_idx - 1) * g_hw_corr_state.num_qubits + qubit_idx] * 0.5;
        }
        if (qubit_idx < g_hw_corr_state.num_qubits - 1) {
            crosstalk_factor -= g_hw_corr_state.crosstalk_matrix[
                (qubit_idx + 1) * g_hw_corr_state.num_qubits + qubit_idx] * 0.5;
        }

        stability = 0.98 * t2_factor * boundary_factor * fmax(0.8, crosstalk_factor);
    }

    pthread_mutex_unlock(&g_hw_corr_state.state_mutex);

    return fmin(1.0, fmax(0.0, stability));
}

double get_measurement_stability(size_t qubit_index) {
    if (!g_hw_corr_state.initialized) {
        return 0.98;
    }

    pthread_mutex_lock(&g_hw_corr_state.state_mutex);

    double stability = 0.98;

    if (qubit_index < g_hw_corr_state.num_qubits) {
        // Measurement stability based on readout characteristics
        double readout_err = g_hw_corr_state.qubit_readout_errors[qubit_index];

        // Base stability from readout fidelity
        stability = 1.0 - readout_err;

        // Account for T1 decay during readout
        double readout_time = g_hw_corr_state.readout_time_us;
        double t1 = g_hw_corr_state.qubit_t1_times[qubit_index];
        double t1_decay = exponential_decay(readout_time, t1);

        stability *= t1_decay;
    }

    pthread_mutex_unlock(&g_hw_corr_state.state_mutex);

    return fmin(1.0, fmax(0.0, stability));
}

// ============================================================================
// Feedback Functions
// ============================================================================

double get_feedback_factor(size_t vertex1, size_t vertex2) {
    if (!g_hw_corr_state.initialized) {
        double distance = (double)abs((int)vertex1 - (int)vertex2);
        return exp(-distance / 10.0);
    }

    pthread_mutex_lock(&g_hw_corr_state.state_mutex);

    // Feedback factor based on:
    // 1. Physical distance between vertices
    // 2. Connectivity in the qubit graph
    // 3. Historical correlation strength

    double distance = (double)abs((int)vertex1 - (int)vertex2);
    double distance_factor = exp(-distance / 10.0);

    // Check if vertices are directly connected
    double connectivity_factor = 1.0;
    if (g_hw_corr_state.connectivity_matrix &&
        vertex1 < g_hw_corr_state.num_qubits &&
        vertex2 < g_hw_corr_state.num_qubits) {
        double conn = g_hw_corr_state.connectivity_matrix[
            vertex1 * g_hw_corr_state.num_qubits + vertex2];
        connectivity_factor = (conn > 0) ? 1.5 : 1.0;  // Boost for connected qubits
    }

    // Historical factor from recent correlations
    double historical_factor = 1.0;
    if (g_hw_corr_state.feedback_history_size > 0) {
        double recent_avg = 0.0;
        size_t count = fmin(10, g_hw_corr_state.feedback_history_size);
        for (size_t i = g_hw_corr_state.feedback_history_size - count;
             i < g_hw_corr_state.feedback_history_size; i++) {
            recent_avg += g_hw_corr_state.feedback_history[i];
        }
        recent_avg /= count;
        historical_factor = 1.0 + recent_avg * 0.5;  // Boost based on recent correlations
    }

    double factor = distance_factor * connectivity_factor * historical_factor;

    pthread_mutex_unlock(&g_hw_corr_state.state_mutex);

    return fmin(2.0, fmax(0.0, factor));  // Cap at 2.0
}

bool should_trigger_correlation_feedback(double correlation) {
    if (!g_hw_corr_state.initialized) {
        return correlation > 0.7;  // Default threshold
    }

    pthread_mutex_lock(&g_hw_corr_state.state_mutex);

    bool trigger = correlation > g_hw_corr_state.feedback_threshold;

    // Adaptive threshold: if we're triggering too often, raise the threshold
    if (g_hw_corr_state.feedback_count > 100) {
        double trigger_rate = (double)g_hw_corr_state.feedback_count /
                             (double)(g_hw_corr_state.feedback_history_size + 1);
        if (trigger_rate > 0.3) {
            // Too many triggers, be more selective
            trigger = correlation > g_hw_corr_state.feedback_threshold * 1.2;
        }
    }

    pthread_mutex_unlock(&g_hw_corr_state.state_mutex);

    return trigger;
}

void trigger_correlation_feedback(size_t vertex1, size_t vertex2, double correlation) {
    if (!g_hw_corr_state.initialized) return;

    pthread_mutex_lock(&g_hw_corr_state.state_mutex);

    // Record feedback event
    g_hw_corr_state.feedback_count++;

    // Store correlation in history (circular buffer)
    if (g_hw_corr_state.feedback_history) {
        if (g_hw_corr_state.feedback_history_size < g_hw_corr_state.feedback_history_capacity) {
            g_hw_corr_state.feedback_history[g_hw_corr_state.feedback_history_size++] = correlation;
        } else {
            // Shift history and add new value
            memmove(g_hw_corr_state.feedback_history,
                   g_hw_corr_state.feedback_history + 1,
                   (g_hw_corr_state.feedback_history_capacity - 1) * sizeof(double));
            g_hw_corr_state.feedback_history[g_hw_corr_state.feedback_history_capacity - 1] = correlation;
        }
    }

    // Update crosstalk estimate based on correlation
    if (g_hw_corr_state.crosstalk_matrix &&
        vertex1 < g_hw_corr_state.num_qubits &&
        vertex2 < g_hw_corr_state.num_qubits) {
        // Exponential moving average update
        double alpha = 0.1;  // Learning rate
        size_t idx = vertex1 * g_hw_corr_state.num_qubits + vertex2;
        g_hw_corr_state.crosstalk_matrix[idx] =
            (1 - alpha) * g_hw_corr_state.crosstalk_matrix[idx] +
            alpha * correlation * 0.1;  // Scale correlation to crosstalk estimate

        // Symmetric update
        idx = vertex2 * g_hw_corr_state.num_qubits + vertex1;
        g_hw_corr_state.crosstalk_matrix[idx] =
            (1 - alpha) * g_hw_corr_state.crosstalk_matrix[idx] +
            alpha * correlation * 0.1;
    }

    // Adaptive threshold adjustment
    if (g_hw_corr_state.feedback_history_size > 50) {
        double avg_correlation = 0.0;
        for (size_t i = 0; i < g_hw_corr_state.feedback_history_size; i++) {
            avg_correlation += g_hw_corr_state.feedback_history[i];
        }
        avg_correlation /= g_hw_corr_state.feedback_history_size;

        // Adjust threshold to maintain reasonable trigger rate
        g_hw_corr_state.feedback_threshold = 0.5 * g_hw_corr_state.feedback_threshold +
                                            0.5 * (avg_correlation + 0.1);
        g_hw_corr_state.feedback_threshold = fmin(0.9, fmax(0.5,
                                                  g_hw_corr_state.feedback_threshold));
    }

    pthread_mutex_unlock(&g_hw_corr_state.state_mutex);
}

// ============================================================================
// Correction Chain Functions
// ============================================================================

// get_correction_chain_length() - Canonical implementation in error_syndrome.c
// (removed: dead code, canonical version has better path optimization)

// ============================================================================
// Calibration Update Interface
// ============================================================================

/**
 * Set crosstalk coefficient between two qubits
 */
void set_crosstalk_coefficient(size_t qubit1, size_t qubit2, double coefficient) {
    if (!g_hw_corr_state.initialized) return;
    if (qubit1 >= g_hw_corr_state.num_qubits || qubit2 >= g_hw_corr_state.num_qubits) return;

    pthread_mutex_lock(&g_hw_corr_state.state_mutex);

    if (g_hw_corr_state.crosstalk_matrix) {
        g_hw_corr_state.crosstalk_matrix[qubit1 * g_hw_corr_state.num_qubits + qubit2] = coefficient;
        g_hw_corr_state.crosstalk_matrix[qubit2 * g_hw_corr_state.num_qubits + qubit1] = coefficient;
    }

    pthread_mutex_unlock(&g_hw_corr_state.state_mutex);
}

/**
 * Set connectivity between two qubits
 */
void set_qubit_connectivity(size_t qubit1, size_t qubit2, double strength) {
    if (!g_hw_corr_state.initialized) return;
    if (qubit1 >= g_hw_corr_state.num_qubits || qubit2 >= g_hw_corr_state.num_qubits) return;

    pthread_mutex_lock(&g_hw_corr_state.state_mutex);

    if (g_hw_corr_state.connectivity_matrix) {
        g_hw_corr_state.connectivity_matrix[qubit1 * g_hw_corr_state.num_qubits + qubit2] = strength;
        g_hw_corr_state.connectivity_matrix[qubit2 * g_hw_corr_state.num_qubits + qubit1] = strength;
    }

    pthread_mutex_unlock(&g_hw_corr_state.state_mutex);
}

/**
 * Get current hardware correlation statistics
 */
void get_correlation_statistics(size_t* feedback_count, double* avg_correlation,
                               double* current_threshold) {
    if (!g_hw_corr_state.initialized) {
        if (feedback_count) *feedback_count = 0;
        if (avg_correlation) *avg_correlation = 0.0;
        if (current_threshold) *current_threshold = 0.7;
        return;
    }

    pthread_mutex_lock(&g_hw_corr_state.state_mutex);

    if (feedback_count) {
        *feedback_count = g_hw_corr_state.feedback_count;
    }

    if (avg_correlation && g_hw_corr_state.feedback_history_size > 0) {
        double sum = 0.0;
        for (size_t i = 0; i < g_hw_corr_state.feedback_history_size; i++) {
            sum += g_hw_corr_state.feedback_history[i];
        }
        *avg_correlation = sum / g_hw_corr_state.feedback_history_size;
    } else if (avg_correlation) {
        *avg_correlation = 0.0;
    }

    if (current_threshold) {
        *current_threshold = g_hw_corr_state.feedback_threshold;
    }

    pthread_mutex_unlock(&g_hw_corr_state.state_mutex);
}
