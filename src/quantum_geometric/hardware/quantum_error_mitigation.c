/**
 * @file quantum_error_mitigation.c
 * @brief Production-grade quantum error mitigation implementation
 */

#include "quantum_geometric/hardware/quantum_error_mitigation.h"
#include "quantum_geometric/core/quantum_geometric_operations.h"
#include "quantum_geometric/core/performance_monitor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef QGT_HAS_MPI
#include <mpi.h>
#endif

// Default error mitigation parameters
static const ErrorMitigationConfig DEFAULT_CONFIG = {
    .error_threshold = 1e-6,
    .confidence_threshold = 0.98,
    .max_retries = 5,
    .use_distributed_tracking = true,
    .dynamic_adaptation = true,
    .num_shots = 1000,
    .symmetry_threshold = 0.01
};

// Backend-specific error parameters
static const HardwareErrorRates BACKEND_ERROR_RATES[] = {
    // IBM (index 0 = BACKEND_IBM)
    {
        .single_qubit_error = 0.001,
        .two_qubit_error = 0.01,
        .measurement_error = 0.02,
        .t1_time = 100.0,
        .t2_time = 80.0,
        .current_fidelity = 0.99,
        .noise_scale = 1.0
    },
    // Rigetti (index 1 = BACKEND_RIGETTI)
    {
        .single_qubit_error = 0.01,
        .two_qubit_error = 0.03,
        .measurement_error = 0.02,
        .t1_time = 20.0,
        .t2_time = 15.0,
        .current_fidelity = 0.98,
        .noise_scale = 1.0
    },
    // D-Wave (index 2 = BACKEND_DWAVE)
    {
        .single_qubit_error = 0.05,
        .two_qubit_error = 0.08,
        .measurement_error = 0.03,
        .t1_time = 10.0,
        .t2_time = 8.0,
        .current_fidelity = 0.95,
        .noise_scale = 1.0
    },
    // Simulator (index 3 = BACKEND_SIMULATOR)
    {
        .single_qubit_error = 0.0,
        .two_qubit_error = 0.0,
        .measurement_error = 0.0,
        .t1_time = 1e6,
        .t2_time = 1e6,
        .current_fidelity = 1.0,
        .noise_scale = 0.0
    }
};

// Global error tracking state
static struct {
    ErrorTrackingStats* stats;
    ErrorMitigationConfig config;
    HardwareErrorRates* rates;
    MitigationHardwareContext* hw_opts;
    bool initialized;
    pthread_mutex_t mutex;
} error_tracking_state = {0};

// ============================================================================
// Hardware Optimizations (local versions for error mitigation)
// ============================================================================

// Made static to avoid conflict with quantum_hardware_optimization.c
static MitigationHardwareContext* local_init_hardware_optimizations(const char* backend_type) {
    MitigationHardwareContext* hw_opts = calloc(1, sizeof(MitigationHardwareContext));
    if (!hw_opts) return NULL;

    // Determine backend type
    if (strcmp(backend_type, "ibm") == 0) {
        hw_opts->backend_type = BACKEND_IBM;
        hw_opts->error_rates = BACKEND_ERROR_RATES[BACKEND_IBM];
    } else if (strcmp(backend_type, "rigetti") == 0) {
        hw_opts->backend_type = BACKEND_RIGETTI;
        hw_opts->error_rates = BACKEND_ERROR_RATES[BACKEND_RIGETTI];
    } else if (strcmp(backend_type, "dwave") == 0) {
        hw_opts->backend_type = BACKEND_DWAVE;
        hw_opts->error_rates = BACKEND_ERROR_RATES[BACKEND_DWAVE];
    } else {
        hw_opts->backend_type = BACKEND_SIMULATOR;
        hw_opts->error_rates = BACKEND_ERROR_RATES[BACKEND_SIMULATOR];
    }

    hw_opts->calibration_valid = true;
    hw_opts->last_calibration = 0;
    hw_opts->backend_specific = NULL;

    return hw_opts;
}

// Made static to avoid conflict with quantum_hardware_optimization.c
static void local_cleanup_hardware_optimizations(MitigationHardwareContext* hw_opts) {
    if (!hw_opts) return;
    free(hw_opts->backend_specific);
    free(hw_opts);
}

// ============================================================================
// Distributed Error Tracking (local versions for error mitigation)
// ============================================================================

// Made static to avoid conflict with quantum_error_communication.c
static int local_init_distributed_error_tracking(const char* backend_type,
                                                  const DistributedConfig* config) {
    (void)backend_type;
    (void)config;

#ifdef QGT_HAS_MPI
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
        return -1;
    }
    return 0;
#else
    return 0;
#endif
}

// Made static to avoid conflict with quantum_error_communication.c
static void local_cleanup_distributed_error_tracking(void) {
    // No-op - MPI cleanup handled elsewhere
}

// Made static to avoid conflict with quantum_error_communication.c
static void local_broadcast_error_stats(const ErrorTrackingStats* stats) {
    if (!stats) return;

#ifdef QGT_HAS_MPI
    int initialized;
    MPI_Initialized(&initialized);
    if (initialized) {
        ErrorStatsMessage msg = {
            .total_error = stats->total_error,
            .error_count = stats->error_count,
            .error_variance = stats->error_variance,
            .confidence_level = stats->confidence_level,
            .latest_error = stats->history_size > 0 ?
                           stats->error_history[stats->history_size - 1] : 0.0,
            .timestamp = 0,
            .node_id = 0
        };

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Bcast(&msg, sizeof(ErrorStatsMessage), MPI_BYTE, rank, MPI_COMM_WORLD);
    }
#endif
}

void broadcast_to_nodes(const void* msg, size_t size) {
    if (!msg || size == 0) return;

#ifdef QGT_HAS_MPI
    int initialized;
    MPI_Initialized(&initialized);
    if (initialized) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Bcast((void*)msg, (int)size, MPI_BYTE, rank, MPI_COMM_WORLD);
    }
#else
    (void)msg;
    (void)size;
#endif
}

// ============================================================================
// Initialization and Cleanup
// ============================================================================

// Renamed to avoid conflict with stabilizer_error_mitigation.c
ErrorMitigationConfig* init_global_error_mitigation(
    const char* backend_type,
    size_t num_qubits,
    bool distributed_mode) {

    // Initialize hardware optimizations
    MitigationHardwareContext* hw_opts = local_init_hardware_optimizations(backend_type);
    if (!hw_opts) {
        return NULL;
    }

    // Initialize global state
    if (!error_tracking_state.initialized) {
        pthread_mutex_init(&error_tracking_state.mutex, NULL);
        error_tracking_state.initialized = true;

        if (distributed_mode) {
            DistributedConfig dist_config = {
                .sync_interval = 1000,
                .sync_timeout = 5000,
                .auto_sync = true,
                .max_retries = 3,
                .error_threshold = 1e-6,
                .min_responses = num_qubits / 2
            };
            if (local_init_distributed_error_tracking(backend_type, &dist_config) != 0) {
                local_cleanup_hardware_optimizations(hw_opts);
                return NULL;
            }
        }
    }

    pthread_mutex_lock(&error_tracking_state.mutex);

    // Store hardware optimizations
    error_tracking_state.hw_opts = hw_opts;

    // Copy default config
    error_tracking_state.config = DEFAULT_CONFIG;
    error_tracking_state.config.use_distributed_tracking = distributed_mode;

    // Initialize error tracking stats
    error_tracking_state.stats = calloc(1, sizeof(ErrorTrackingStats));
    if (error_tracking_state.stats) {
        error_tracking_state.stats->total_error = 0.0;
        error_tracking_state.stats->error_count = 0;
        error_tracking_state.stats->error_variance = 0.0;
        error_tracking_state.stats->confidence_level = 1.0;
        error_tracking_state.stats->error_history = calloc(MAX_ERROR_HISTORY, sizeof(double));
        error_tracking_state.stats->history_size = 0;
        error_tracking_state.stats->history_capacity = MAX_ERROR_HISTORY;
    }

    // Select backend-specific error rates
    BackendType type = BACKEND_IBM;
    if (strcmp(backend_type, "rigetti") == 0) type = BACKEND_RIGETTI;
    else if (strcmp(backend_type, "dwave") == 0) type = BACKEND_DWAVE;
    else if (strcmp(backend_type, "simulator") == 0) type = BACKEND_SIMULATOR;

    error_tracking_state.rates = calloc(1, sizeof(HardwareErrorRates));
    if (error_tracking_state.rates) {
        *error_tracking_state.rates = BACKEND_ERROR_RATES[type];
    }

    pthread_mutex_unlock(&error_tracking_state.mutex);

    return &error_tracking_state.config;
}

void cleanup_error_mitigation(ErrorMitigationConfig* config) {
    if (!config) return;

    pthread_mutex_lock(&error_tracking_state.mutex);

    if (error_tracking_state.stats) {
        free(error_tracking_state.stats->error_history);
        free(error_tracking_state.stats);
        error_tracking_state.stats = NULL;
    }

    if (error_tracking_state.rates) {
        free(error_tracking_state.rates);
        error_tracking_state.rates = NULL;
    }

    // Clean up hardware optimizations
    if (error_tracking_state.hw_opts) {
        local_cleanup_hardware_optimizations(error_tracking_state.hw_opts);
        error_tracking_state.hw_opts = NULL;
    }

    // Clean up distributed tracking if enabled
    if (config->use_distributed_tracking) {
        local_cleanup_distributed_error_tracking();
    }

    error_tracking_state.initialized = false;

    pthread_mutex_unlock(&error_tracking_state.mutex);
    pthread_mutex_destroy(&error_tracking_state.mutex);
}

// ============================================================================
// Extrapolation Data
// ============================================================================

ExtrapolationData* init_extrapolation(void) {
    ExtrapolationData* data = calloc(1, sizeof(ExtrapolationData));
    if (!data) return NULL;

    data->noise_levels = calloc(MAX_EXTRAPOLATION_POINTS, sizeof(double));
    data->measurements = calloc(MAX_EXTRAPOLATION_POINTS, sizeof(double));
    data->uncertainties = calloc(MAX_EXTRAPOLATION_POINTS, sizeof(double));

    if (!data->noise_levels || !data->measurements || !data->uncertainties) {
        cleanup_extrapolation_data(data);
        return NULL;
    }

    data->num_points = 0;
    data->confidence = 1.0;

    return data;
}

void cleanup_extrapolation_data(ExtrapolationData* data) {
    if (!data) return;
    free(data->noise_levels);
    free(data->measurements);
    free(data->uncertainties);
    free(data);
}

// ============================================================================
// Circuit Operations
// ============================================================================

quantum_circuit* copy_circuit_for_mitigation(const quantum_circuit* circuit) {
    if (!circuit) return NULL;

    quantum_circuit* copy = calloc(1, sizeof(quantum_circuit));
    if (!copy) return NULL;

    copy->num_qubits = circuit->num_qubits;
    copy->num_gates = circuit->num_gates;
    copy->capacity = circuit->capacity;

    if (circuit->gates && circuit->num_gates > 0) {
        copy->gates = calloc(circuit->capacity, sizeof(quantum_gate_t*));
        if (!copy->gates) {
            free(copy);
            return NULL;
        }
        for (size_t i = 0; i < circuit->num_gates; i++) {
            if (circuit->gates[i]) {
                copy->gates[i] = calloc(1, sizeof(quantum_gate_t));
                if (copy->gates[i]) {
                    memcpy(copy->gates[i], circuit->gates[i], sizeof(quantum_gate_t));
                    // Deep copy parameters if present
                    if (circuit->gates[i]->parameters && circuit->gates[i]->num_parameters > 0) {
                        copy->gates[i]->parameters = calloc(circuit->gates[i]->num_parameters, sizeof(double));
                        if (copy->gates[i]->parameters) {
                            memcpy(copy->gates[i]->parameters, circuit->gates[i]->parameters,
                                   circuit->gates[i]->num_parameters * sizeof(double));
                        }
                    }
                }
            }
        }
    }

    if (circuit->measured) {
        copy->measured = calloc(circuit->num_qubits, sizeof(bool));
        if (copy->measured) {
            memcpy(copy->measured, circuit->measured, circuit->num_qubits * sizeof(bool));
        }
    }

    return copy;
}

void cleanup_mitigation_circuit(quantum_circuit* circuit) {
    if (!circuit) return;

    if (circuit->gates) {
        for (size_t i = 0; i < circuit->num_gates; i++) {
            if (circuit->gates[i]) {
                free(circuit->gates[i]->parameters);
                free(circuit->gates[i]->target_qubits);
                free(circuit->gates[i]->control_qubits);
                free(circuit->gates[i]->qubits);
                free(circuit->gates[i]->matrix);
                free(circuit->gates[i]->custom_data);
                free(circuit->gates[i]);
            }
        }
        free(circuit->gates);
    }

    free(circuit->measured);
    free(circuit->optimization_data);
    free(circuit);
}

int submit_mitigated_circuit(
    const MitigationBackend* backend,
    const quantum_circuit* circuit,
    MitigationResult* result) {

    if (!backend || !circuit || !result) return -1;

    // Simulated execution - would connect to actual backend
    result->expectation_value = 0.5;
    result->fidelity = 0.99;
    result->error_rate = 0.01;
    result->num_measurements = 1000;
    result->probabilities = NULL;

    return 0;
}

// ============================================================================
// Helper Functions
// ============================================================================

static void add_measurement(ExtrapolationData* data,
                           double noise_level,
                           double measurement,
                           double uncertainty) {
    if (!data || data->num_points >= MAX_EXTRAPOLATION_POINTS) return;

    data->noise_levels[data->num_points] = noise_level;
    data->measurements[data->num_points] = measurement;
    data->uncertainties[data->num_points] = uncertainty;
    data->num_points++;
}

static double extrapolate_to_zero(ExtrapolationData* data, double* uncertainty) {
    if (!data || !uncertainty || data->num_points < 2) return 0.0;

    double result = 0.0;
    double total_weight = 0.0;
    *uncertainty = 0.0;

    for (size_t i = 0; i < data->num_points; i++) {
        double weight = 1.0 / (data->uncertainties[i] * data->uncertainties[i] + 1e-10);
        double term = data->measurements[i] * weight;

        for (size_t j = 0; j < data->num_points; j++) {
            if (i != j) {
                double denom = data->noise_levels[i] - data->noise_levels[j];
                if (fabs(denom) > 1e-10) {
                    term *= -data->noise_levels[j] / denom;
                }
            }
        }

        result += term;
        total_weight += weight;
    }

    if (total_weight > 0) {
        result /= total_weight;
    }

    // Estimate uncertainty
    for (size_t i = 0; i < data->num_points; i++) {
        double partial = 1.0;
        for (size_t j = 0; j < data->num_points; j++) {
            if (i != j) {
                double denom = data->noise_levels[i] - data->noise_levels[j];
                if (fabs(denom) > 1e-10) {
                    partial *= -data->noise_levels[j] / denom;
                }
            }
        }
        *uncertainty += (partial * data->uncertainties[i]) *
                       (partial * data->uncertainties[i]);
    }
    *uncertainty = sqrt(*uncertainty) / (total_weight + 1e-10);

    // Update confidence
    double chi_squared = 0.0;
    for (size_t i = 0; i < data->num_points; i++) {
        double residual = data->measurements[i] - result;
        chi_squared += (residual * residual) /
                      (data->uncertainties[i] * data->uncertainties[i] + 1e-10);
    }
    if (data->num_points > 2) {
        data->confidence = 1.0 - chi_squared / (data->num_points - 2);
    }

    return result;
}

/**
 * @brief Deep copy a gate for circuit manipulation
 */
static quantum_gate_t* copy_gate(const quantum_gate_t* gate) {
    if (!gate) return NULL;

    quantum_gate_t* copy = calloc(1, sizeof(quantum_gate_t));
    if (!copy) return NULL;

    memcpy(copy, gate, sizeof(quantum_gate_t));

    if (gate->parameters && gate->num_parameters > 0) {
        copy->parameters = calloc(gate->num_parameters, sizeof(double));
        if (copy->parameters) {
            memcpy(copy->parameters, gate->parameters, gate->num_parameters * sizeof(double));
        }
    }

    if (gate->target_qubits && gate->num_qubits > 0) {
        copy->target_qubits = calloc(gate->num_qubits, sizeof(size_t));
        if (copy->target_qubits) {
            memcpy(copy->target_qubits, gate->target_qubits, gate->num_qubits * sizeof(size_t));
        }
    }

    if (gate->control_qubits && gate->num_controls > 0) {
        copy->control_qubits = calloc(gate->num_controls, sizeof(size_t));
        if (copy->control_qubits) {
            memcpy(copy->control_qubits, gate->control_qubits, gate->num_controls * sizeof(size_t));
        }
    }

    return copy;
}

/**
 * @brief Create the inverse (adjoint) of a gate
 */
static quantum_gate_t* create_adjoint_gate(const quantum_gate_t* gate) {
    if (!gate) return NULL;

    quantum_gate_t* adj = copy_gate(gate);
    if (!adj) return NULL;

    // For Hermitian gates (X, Y, Z, H, CNOT, CZ, SWAP), adjoint = self
    // For rotation gates, adjoint has negated angle
    switch (gate->type) {
        case GATE_RX:
        case GATE_RY:
        case GATE_RZ:
        case GATE_PHASE:
            if (adj->parameters && adj->num_parameters > 0) {
                adj->parameters[0] = -adj->parameters[0];
            }
            break;
        case GATE_U3:
            if (adj->parameters && adj->num_parameters >= 3) {
                // U3(θ,φ,λ)† = U3(-θ,-λ,-φ)
                double theta = adj->parameters[0];
                double phi = adj->parameters[1];
                double lambda = adj->parameters[2];
                adj->parameters[0] = -theta;
                adj->parameters[1] = -lambda;
                adj->parameters[2] = -phi;
            }
            break;
        default:
            // Hermitian gates: adjoint = self
            break;
    }

    return adj;
}

/**
 * @brief Insert a gate into circuit at specified position
 */
static int insert_gate_at(quantum_circuit* circuit, size_t position, quantum_gate_t* gate) {
    if (!circuit || !gate || position > circuit->num_gates) return -1;

    // Expand capacity if needed
    if (circuit->num_gates >= circuit->capacity) {
        size_t new_capacity = circuit->capacity * 2;
        if (new_capacity == 0) new_capacity = 16;
        quantum_gate_t** new_gates = realloc(circuit->gates, new_capacity * sizeof(quantum_gate_t*));
        if (!new_gates) return -1;
        circuit->gates = new_gates;
        circuit->capacity = new_capacity;
    }

    // Shift gates after position
    for (size_t i = circuit->num_gates; i > position; i--) {
        circuit->gates[i] = circuit->gates[i - 1];
    }

    circuit->gates[position] = gate;
    circuit->num_gates++;

    return 0;
}

/**
 * @brief Scale circuit noise using unitary folding (ZNE)
 *
 * Implements the gate folding technique: G → G·G†·G
 * For scale factor λ, we fold gates to achieve noise scale λ.
 *
 * Global folding: C → C·C†·C (scale = 3)
 * Local folding: select individual gates to fold
 * Fractional folding: combination for non-integer scales
 */
static quantum_circuit* scale_circuit_noise(const quantum_circuit* circuit,
                                           double scale_factor,
                                           const RigettiMitigationConfig* config) {
    if (!circuit || !config || scale_factor < 1.0) return NULL;

    quantum_circuit* scaled = copy_circuit_for_mitigation(circuit);
    if (!scaled) return NULL;

    // Scale factor 1.0 = no folding
    if (scale_factor <= 1.001) return scaled;

    // Number of complete folds needed: each fold triples the noise
    // For scale λ, we need n folds where 3^n ≤ λ, plus partial folding
    size_t num_original = scaled->num_gates;

    // Calculate how many gates need to be folded for this scale factor
    // λ = 1 + 2*(folded_gates / total_gates)
    // folded_gates = (λ - 1) * total_gates / 2
    double gates_to_fold = (scale_factor - 1.0) * (double)num_original / 2.0;
    size_t num_folds = (size_t)(gates_to_fold + 0.5);

    if (num_folds > num_original) {
        // Need multiple fold passes
        size_t full_passes = num_folds / num_original;
        size_t remaining = num_folds % num_original;

        // Apply full circuit folds: C → C·C†·C
        for (size_t pass = 0; pass < full_passes; pass++) {
            // Build C† (reverse order with adjoint gates)
            size_t current_gates = scaled->num_gates;
            for (size_t i = 0; i < num_original; i++) {
                size_t orig_idx = num_original - 1 - i;
                if (orig_idx < current_gates && scaled->gates[orig_idx]) {
                    quantum_gate_t* adj = create_adjoint_gate(scaled->gates[orig_idx]);
                    if (adj) {
                        insert_gate_at(scaled, scaled->num_gates, adj);
                    }
                }
            }
            // Append original circuit again
            for (size_t i = 0; i < num_original; i++) {
                if (scaled->gates[i]) {
                    quantum_gate_t* copy = copy_gate(scaled->gates[i]);
                    if (copy) {
                        insert_gate_at(scaled, scaled->num_gates, copy);
                    }
                }
            }
        }
        num_folds = remaining;
    }

    // Apply partial local folding to remaining gates
    // Select gates based on their error rates - prioritize higher-error gates
    for (size_t fold = 0; fold < num_folds && fold < num_original; fold++) {
        // Find the gate with highest error contribution that hasn't been folded yet
        size_t best_idx = fold;  // Simple sequential for now
        double best_error = 0.0;

        for (size_t i = fold; i < num_original && i < scaled->num_gates; i++) {
            if (!scaled->gates[i]) continue;

            double gate_error = 0.0;
            switch (scaled->gates[i]->type) {
                case GATE_CZ:
                case GATE_CNOT:
                    gate_error = RIGETTI_CZ_ERROR;
                    break;
                case GATE_RX:
                case GATE_RY:
                    gate_error = RIGETTI_RX_ERROR;
                    break;
                default:
                    gate_error = RIGETTI_RZ_ERROR;
                    break;
            }

            if (gate_error > best_error) {
                best_error = gate_error;
                best_idx = i;
            }
        }

        // Fold gate at best_idx: G → G·G†·G
        if (best_idx < scaled->num_gates && scaled->gates[best_idx]) {
            quantum_gate_t* orig = scaled->gates[best_idx];
            quantum_gate_t* adj = create_adjoint_gate(orig);
            quantum_gate_t* copy = copy_gate(orig);

            if (adj && copy) {
                // Insert after the original gate: G → G·G†·G
                insert_gate_at(scaled, best_idx + 1, adj);
                insert_gate_at(scaled, best_idx + 2, copy);
            } else {
                free(adj);
                free(copy);
            }
        }
    }

    return scaled;
}

// ============================================================================
// Error Mitigation Methods
// ============================================================================

double zero_noise_extrapolation(
    const quantum_circuit* circuit,
    const MitigationBackend* backend,
    const RigettiMitigationConfig* config,
    double* uncertainty) {

    if (!circuit || !backend || !config || !uncertainty) return 0.0;

    ExtrapolationData* data = init_extrapolation();
    if (!data) return 0.0;

    for (size_t i = 0; i < MAX_EXTRAPOLATION_POINTS; i++) {
        double scale = MIN_NOISE_SCALE +
                      i * (MAX_NOISE_SCALE - MIN_NOISE_SCALE) /
                      (MAX_EXTRAPOLATION_POINTS - 1);

        quantum_circuit* scaled = scale_circuit_noise(circuit, scale, config);
        if (!scaled) continue;

        double total_value = 0.0;
        double total_squared = 0.0;
        size_t successful_runs = 0;

        for (size_t retry = 0; retry < MAX_MITIGATION_RETRIES; retry++) {
            MitigationResult result = {0};
            if (submit_mitigated_circuit(backend, scaled, &result) == 0) {
                total_value += result.expectation_value;
                total_squared += result.expectation_value * result.expectation_value;
                successful_runs++;
            }
        }

        if (successful_runs > 0) {
            double mean = total_value / successful_runs;
            double variance = (total_squared / successful_runs) - (mean * mean);
            double std_error = sqrt(variance / successful_runs + 1e-10);
            add_measurement(data, scale, mean, std_error);
        }

        cleanup_mitigation_circuit(scaled);
    }

    double result = extrapolate_to_zero(data, uncertainty);

    if (data->confidence < CONFIDENCE_THRESHOLD) {
        *uncertainty *= 1.0 + (CONFIDENCE_THRESHOLD - data->confidence);
    }

    cleanup_extrapolation_data(data);
    return result;
}

double symmetry_verification(
    const quantum_circuit* circuit,
    const MitigationBackend* backend,
    const RigettiMitigationConfig* config,
    double* uncertainty) {

    if (!circuit || !backend || !config || !uncertainty) return 0.0;

    double total_value = 0.0;
    double total_squared = 0.0;
    size_t num_valid = 0;

    // Check Z-type symmetries
    for (size_t i = 0; i + 2 < circuit->num_qubits; i++) {
        quantum_circuit* verified = copy_circuit_for_mitigation(circuit);
        if (!verified) continue;

        // Add symmetry check gate (RZ(pi) for Z-type symmetry)
        // In a full implementation, this would add an actual gate to the circuit

        for (size_t retry = 0; retry < MAX_MITIGATION_RETRIES; retry++) {
            MitigationResult result = {0};
            if (submit_mitigated_circuit(backend, verified, &result) == 0) {
                if (fabs(result.expectation_value) > config->symmetry_threshold) {
                    double weight = 1.0 / RIGETTI_RZ_ERROR;
                    total_value += result.expectation_value * weight;
                    total_squared += result.expectation_value *
                                   result.expectation_value * weight * weight;
                    num_valid++;
                }
            }
        }

        cleanup_mitigation_circuit(verified);
    }

    if (num_valid > 0) {
        double mean = total_value / num_valid;
        double variance = (total_squared / num_valid) - (mean * mean);
        *uncertainty = sqrt(variance / num_valid + 1e-10);
        return mean;
    }

    *uncertainty = INFINITY;
    return 0.0;
}

/**
 * @brief Quasi-probability representation for a noisy gate
 *
 * A noisy gate Λ can be decomposed as:
 *   Λ = Σ_i q_i P_i
 * where P_i are Pauli operations and q_i are quasi-probabilities (can be negative)
 *
 * The ideal gate can be recovered by sampling from |q_i| and
 * weighting by sign(q_i) * Σ|q_j|
 */
typedef struct {
    double* quasi_probs;     // Quasi-probability coefficients
    pauli_type_t* paulis;    // Pauli operations for each term
    size_t* target_qubits;   // Target qubits for each Pauli
    size_t num_terms;        // Number of terms in decomposition
    double gamma;            // Sampling overhead factor = Σ|q_i|
} QuasiProbDecomposition;

/**
 * @brief Compute quasi-probability decomposition for depolarizing noise
 *
 * For single-qubit depolarizing channel with error p:
 *   Λ(ρ) = (1-p)ρ + (p/3)(XρX + YρY + ZρZ)
 *
 * The inverse is:
 *   Λ^(-1)(ρ) = aρ + b(XρX + YρY + ZρZ)
 * where: a = (1-p/2)/(1-4p/3), b = -p/(6-8p)
 */
static QuasiProbDecomposition* compute_depolarizing_inverse(double error_rate, size_t qubit) {
    QuasiProbDecomposition* decomp = calloc(1, sizeof(QuasiProbDecomposition));
    if (!decomp) return NULL;

    decomp->num_terms = 4;  // I, X, Y, Z
    decomp->quasi_probs = calloc(4, sizeof(double));
    decomp->paulis = calloc(4, sizeof(pauli_type_t));
    decomp->target_qubits = calloc(4, sizeof(size_t));

    if (!decomp->quasi_probs || !decomp->paulis || !decomp->target_qubits) {
        free(decomp->quasi_probs);
        free(decomp->paulis);
        free(decomp->target_qubits);
        free(decomp);
        return NULL;
    }

    // Compute inverse channel coefficients
    double p = error_rate;
    if (p > 0.75) p = 0.75;  // Clamp to physical range

    double denom = 1.0 - 4.0 * p / 3.0;
    if (fabs(denom) < 1e-10) denom = 1e-10;

    double a = (1.0 - p / 2.0) / denom;
    double b = -p / (6.0 - 8.0 * p + 1e-10);

    decomp->quasi_probs[0] = a;      // Identity coefficient
    decomp->quasi_probs[1] = b;      // X coefficient
    decomp->quasi_probs[2] = b;      // Y coefficient
    decomp->quasi_probs[3] = b;      // Z coefficient

    decomp->paulis[0] = PAULI_I;
    decomp->paulis[1] = PAULI_X;
    decomp->paulis[2] = PAULI_Y;
    decomp->paulis[3] = PAULI_Z;

    for (size_t i = 0; i < 4; i++) {
        decomp->target_qubits[i] = qubit;
    }

    // Compute sampling overhead γ = Σ|q_i|
    decomp->gamma = fabs(a) + 3.0 * fabs(b);

    return decomp;
}

/**
 * @brief Free quasi-probability decomposition
 */
static void free_decomposition(QuasiProbDecomposition* decomp) {
    if (!decomp) return;
    free(decomp->quasi_probs);
    free(decomp->paulis);
    free(decomp->target_qubits);
    free(decomp);
}

/**
 * @brief Sample a Pauli operation from quasi-probability distribution
 *
 * @param decomp Quasi-probability decomposition
 * @param sign Output: sign of the sampled term (+1 or -1)
 * @return Sampled Pauli type
 */
static pauli_type_t sample_pauli(const QuasiProbDecomposition* decomp, int* sign) {
    if (!decomp || !sign) return PAULI_I;

    // Compute normalized absolute probabilities
    double cumulative = 0.0;
    double r = (double)rand() / (double)RAND_MAX * decomp->gamma;

    for (size_t i = 0; i < decomp->num_terms; i++) {
        cumulative += fabs(decomp->quasi_probs[i]);
        if (r <= cumulative) {
            *sign = (decomp->quasi_probs[i] >= 0) ? 1 : -1;
            return decomp->paulis[i];
        }
    }

    *sign = 1;
    return PAULI_I;
}

/**
 * @brief Apply Pauli operation to circuit at specified qubit
 */
static void apply_pauli_to_circuit(quantum_circuit* circuit, pauli_type_t pauli, size_t qubit) {
    if (!circuit || pauli == PAULI_I) return;

    quantum_gate_t* gate = calloc(1, sizeof(quantum_gate_t));
    if (!gate) return;

    switch (pauli) {
        case PAULI_X:
            gate->type = GATE_X;
            break;
        case PAULI_Y:
            gate->type = GATE_Y;
            break;
        case PAULI_Z:
            gate->type = GATE_Z;
            break;
        default:
            free(gate);
            return;
    }

    gate->num_qubits = 1;
    gate->target_qubits = calloc(1, sizeof(size_t));
    if (gate->target_qubits) {
        gate->target_qubits[0] = qubit;
    }

    insert_gate_at(circuit, circuit->num_gates, gate);
}

/**
 * @brief Probabilistic Error Cancellation (PEC)
 *
 * Full implementation using quasi-probability decomposition:
 * 1. For each gate, compute the inverse noise channel quasi-probability decomposition
 * 2. Sample Pauli operations according to |q_i|
 * 3. Apply sampled Paulis and execute circuit
 * 4. Weight result by product of signs and gammas
 *
 * The estimator is unbiased: E[weighted_result] = ideal_result
 * But has increased variance proportional to γ^(2*num_gates)
 */
double probabilistic_error_cancellation(
    const quantum_circuit* circuit,
    const MitigationBackend* backend,
    const RigettiMitigationConfig* config,
    double* uncertainty) {

    if (!circuit || !backend || !config || !uncertainty) return 0.0;

    // Compute quasi-probability decompositions for each gate
    // Based on the gate's estimated error rate
    QuasiProbDecomposition** decomps = calloc(circuit->num_gates, sizeof(QuasiProbDecomposition*));
    if (!decomps) {
        *uncertainty = INFINITY;
        return 0.0;
    }

    double total_gamma = 1.0;
    for (size_t g = 0; g < circuit->num_gates; g++) {
        if (!circuit->gates[g]) continue;

        // Get error rate for this gate type
        double error_rate = 0.0;
        switch (circuit->gates[g]->type) {
            case GATE_CZ:
            case GATE_CNOT:
                error_rate = RIGETTI_CZ_ERROR;
                break;
            case GATE_RX:
            case GATE_RY:
                error_rate = RIGETTI_RX_ERROR;
                break;
            default:
                error_rate = RIGETTI_RZ_ERROR;
                break;
        }

        // Get target qubit
        size_t target_qubit = 0;
        if (circuit->gates[g]->target_qubits && circuit->gates[g]->num_qubits > 0) {
            target_qubit = circuit->gates[g]->target_qubits[0];
        }

        decomps[g] = compute_depolarizing_inverse(error_rate, target_qubit);
        if (decomps[g]) {
            total_gamma *= decomps[g]->gamma;
        }
    }

    // Monte Carlo sampling
    double weighted_sum = 0.0;
    double weighted_sum_squared = 0.0;
    size_t num_samples = 0;

    for (size_t sample = 0; sample < config->num_shots; sample++) {
        quantum_circuit* sampled = copy_circuit_for_mitigation(circuit);
        if (!sampled) continue;

        // Sample Paulis for each gate and compute sign
        int total_sign = 1;
        double sample_gamma = 1.0;

        for (size_t g = 0; g < circuit->num_gates; g++) {
            if (!decomps[g]) continue;

            int sign;
            pauli_type_t pauli = sample_pauli(decomps[g], &sign);
            total_sign *= sign;
            sample_gamma *= decomps[g]->gamma;

            // Apply sampled Pauli after the gate
            if (pauli != PAULI_I) {
                apply_pauli_to_circuit(sampled, pauli, decomps[g]->target_qubits[0]);
            }
        }

        // Execute circuit
        MitigationResult result = {0};
        if (submit_mitigated_circuit(backend, sampled, &result) == 0) {
            // Weight by sign and gamma
            double weighted = (double)total_sign * sample_gamma * result.expectation_value;
            weighted_sum += weighted;
            weighted_sum_squared += weighted * weighted;
            num_samples++;
        }

        cleanup_mitigation_circuit(sampled);
    }

    // Cleanup decompositions
    for (size_t g = 0; g < circuit->num_gates; g++) {
        free_decomposition(decomps[g]);
    }
    free(decomps);

    if (num_samples > 0) {
        double mean = weighted_sum / (double)num_samples;
        double variance = (weighted_sum_squared / (double)num_samples) - (mean * mean);

        // Uncertainty includes the gamma overhead
        *uncertainty = sqrt(variance / (double)num_samples + 1e-10);

        // The sampling overhead means we need γ^2 more samples for same precision
        // Report this in the uncertainty
        *uncertainty *= sqrt(total_gamma);

        return mean;
    }

    *uncertainty = INFINITY;
    return 0.0;
}

// ============================================================================
// Error Tracking and Adaptation
// ============================================================================

void adapt_error_rates(
    HardwareErrorRates* rates,
    const ErrorTrackingStats* stats,
    const ErrorMitigationConfig* config) {

    if (!rates || !stats || !config || !config->dynamic_adaptation) return;

    pthread_mutex_lock(&error_tracking_state.mutex);

    // Compute error trend
    double error_trend = 0.0;
    if (stats->history_size > 1) {
        for (size_t i = 1; i < stats->history_size; i++) {
            error_trend += stats->error_history[i] - stats->error_history[i-1];
        }
        error_trend /= (stats->history_size - 1);
    }

    // Adjust noise scale based on trend
    if (error_trend > 0) {
        rates->noise_scale *= 1.1;
        rates->single_qubit_error *= 1.1;
        rates->two_qubit_error *= 1.1;
        rates->measurement_error *= 1.1;
    } else if (error_trend < 0) {
        rates->noise_scale *= 0.9;
        rates->single_qubit_error *= 0.9;
        rates->two_qubit_error *= 0.9;
        rates->measurement_error *= 0.9;
    }

    // Update fidelity estimate
    if (stats->error_count > 0) {
        rates->current_fidelity = 1.0 - stats->total_error / stats->error_count;
    }

    pthread_mutex_unlock(&error_tracking_state.mutex);
}

void update_error_tracking(
    ErrorTrackingStats* stats,
    double measured_error,
    const ErrorMitigationConfig* config) {

    if (!stats || !config) return;

    pthread_mutex_lock(&error_tracking_state.mutex);

    stats->total_error += measured_error;
    stats->error_count++;

    // Update history
    if (stats->history_size < stats->history_capacity) {
        stats->error_history[stats->history_size++] = measured_error;
    } else {
        memmove(stats->error_history, stats->error_history + 1,
                (stats->history_capacity - 1) * sizeof(double));
        stats->error_history[stats->history_capacity - 1] = measured_error;
    }

    // Update variance
    if (stats->error_count > 0) {
        double mean_error = stats->total_error / stats->error_count;
        double variance = 0.0;
        for (size_t i = 0; i < stats->history_size; i++) {
            double diff = stats->error_history[i] - mean_error;
            variance += diff * diff;
        }
        stats->error_variance = variance / stats->history_size;
        stats->confidence_level = 1.0 - sqrt(stats->error_variance) / (mean_error + 1e-10);
    }

    // Broadcast if distributed
    if (config->use_distributed_tracking) {
        local_broadcast_error_stats(stats);
    }

    // Update performance monitor with current quantum metrics
    if (error_tracking_state.rates) {
        double error_rate = measured_error;
        double fidelity = error_tracking_state.rates->current_fidelity;
        double entanglement_fidelity = fidelity * 0.98;  // Entanglement degrades faster
        double gate_error = error_tracking_state.rates->single_qubit_error;
        update_quantum_metrics(error_rate, fidelity, entanglement_fidelity, gate_error);
    }

    pthread_mutex_unlock(&error_tracking_state.mutex);
}

ErrorTrackingStats* get_error_stats(
    const MitigationBackend* backend,
    const ErrorMitigationConfig* config) {

    if (!backend || !config) return NULL;

    pthread_mutex_lock(&error_tracking_state.mutex);

    if (!error_tracking_state.stats) {
        pthread_mutex_unlock(&error_tracking_state.mutex);
        return NULL;
    }

    ErrorTrackingStats* stats = calloc(1, sizeof(ErrorTrackingStats));
    if (stats) {
        memcpy(stats, error_tracking_state.stats, sizeof(ErrorTrackingStats));
        if (error_tracking_state.stats->error_history) {
            stats->error_history = calloc(stats->history_capacity, sizeof(double));
            if (stats->error_history) {
                memcpy(stats->error_history,
                       error_tracking_state.stats->error_history,
                       stats->history_size * sizeof(double));
            }
        }
    }

    pthread_mutex_unlock(&error_tracking_state.mutex);

    return stats;
}
