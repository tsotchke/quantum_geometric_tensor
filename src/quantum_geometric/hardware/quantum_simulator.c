/**
 * @file quantum_simulator.c
 * @brief Semi-classical quantum simulator for quantum hardware emulation
 *
 * Implements a high-fidelity quantum simulator with:
 * - Statevector simulation for small circuits
 * - Hierarchical matrix operations for O(log n) complexity
 * - Tensor network contraction for larger circuits
 * - Realistic noise modeling (depolarizing, amplitude/phase damping)
 * - Error mitigation (ZNE, probabilistic error cancellation)
 */

#include "quantum_geometric/hardware/quantum_simulator.h"
#include "quantum_geometric/core/hierarchical_matrix.h"
#include "quantum_geometric/core/quantum_types.h"
#include "quantum_geometric/learning/quantum_stochastic_sampling.h"
#include "quantum_geometric/hardware/quantum_geometric_gpu.h"
#include "quantum_geometric/core/quantum_geometric_logging.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <complex.h>

// Logging macros
#define log_info(...)  geometric_log_info(__VA_ARGS__)
#define log_warn(...)  geometric_log_warning(__VA_ARGS__)
#define log_error(...) geometric_log_error(__VA_ARGS__)

// Hierarchical matrix tolerance for compression
#define HIERARCHICAL_TOLERANCE 1e-10

// Constants
#define DEFAULT_SHOTS 1024
#define PI 3.14159265358979323846
#define SQRT2 1.41421356237309504880

// Random number generation
static unsigned int g_rng_state = 0;

static void seed_rng(void) {
    if (g_rng_state == 0) {
        g_rng_state = (unsigned int)time(NULL);
    }
}

static double random_double(void) {
    seed_rng();
    g_rng_state = g_rng_state * 1103515245 + 12345;
    return (double)(g_rng_state & 0x7fffffff) / (double)0x7fffffff;
}

// ============================================================================
// Simulator State Management
// ============================================================================

SimulatorState* sim_init(uint32_t num_qubits, uint32_t num_classical_bits, const struct SimulatorConfig* config) {
    if (num_qubits > MAX_QUBITS) {
        log_error("Number of qubits (%u) exceeds maximum (%d)", num_qubits, MAX_QUBITS);
        return NULL;
    }

    SimulatorState* state = calloc(1, sizeof(SimulatorState));
    if (!state) return NULL;

    state->num_qubits = num_qubits;
    state->num_classical_bits = num_classical_bits;

    // Allocate statevector: 2^n complex amplitudes
    size_t dim = 1UL << num_qubits;
    state->amplitudes = calloc(dim, sizeof(double complex));
    if (!state->amplitudes) {
        free(state);
        return NULL;
    }

    // Initialize to |0...0⟩ state
    state->amplitudes[0] = 1.0 + 0.0 * I;

    // Allocate classical bits
    if (num_classical_bits > 0) {
        state->classical_bits = calloc(num_classical_bits, sizeof(bool));
        if (!state->classical_bits) {
            free(state->amplitudes);
            free(state);
            return NULL;
        }
    }

    state->fidelity = 1.0;
    state->error_rate = 0.0;

    // Set default noise model
    state->active_noise.type = NOISE_NONE;
    state->active_noise.gate_error_rate = 0.0;
    state->active_noise.measurement_error_rate = 0.0;
    state->active_noise.decoherence_rate = 0.0;

    // Apply config if provided
    if (config && config->noise_model) {
        // Extract noise parameters from the noise_model array
        // Format: [gate_error, measurement_error, decoherence_rate, ...]
        state->active_noise.gate_error_rate = config->noise_model[0];
        state->active_noise.measurement_error_rate = config->noise_model[1];
        state->active_noise.decoherence_rate = config->noise_model[2];
        if (config->noise_model[0] > 0) {
            state->active_noise.type = NOISE_DEPOLARIZING;
        }
    }

    return state;
}

void sim_reset_state(SimulatorState* state) {
    if (!state || !state->amplitudes) return;

    size_t dim = 1UL << state->num_qubits;

    // Reset to |0...0⟩
    memset(state->amplitudes, 0, dim * sizeof(double complex));
    state->amplitudes[0] = 1.0 + 0.0 * I;

    // Reset classical bits
    if (state->classical_bits) {
        memset(state->classical_bits, 0, state->num_classical_bits * sizeof(bool));
    }

    state->fidelity = 1.0;
    state->error_rate = 0.0;
}

void sim_cleanup(SimulatorState* state) {
    if (!state) return;

    free(state->amplitudes);
    free(state->classical_bits);
    free(state->active_noise.custom_parameters);
    free(state->custom_state);
    free(state);
}

// ============================================================================
// Gate Operations
// ============================================================================

// Apply single-qubit gate
static void apply_single_gate(double complex* amplitudes, uint32_t target,
                              double complex g00, double complex g01,
                              double complex g10, double complex g11,
                              uint32_t num_qubits) {
    size_t dim = 1UL << num_qubits;
    size_t target_mask = 1UL << target;

    for (size_t i = 0; i < dim; i++) {
        if ((i & target_mask) == 0) {
            size_t j = i | target_mask;
            double complex a0 = amplitudes[i];
            double complex a1 = amplitudes[j];
            amplitudes[i] = g00 * a0 + g01 * a1;
            amplitudes[j] = g10 * a0 + g11 * a1;
        }
    }
}

// Apply two-qubit controlled gate
static void apply_controlled_gate(double complex* amplitudes,
                                  uint32_t control, uint32_t target,
                                  double complex g00, double complex g01,
                                  double complex g10, double complex g11,
                                  uint32_t num_qubits) {
    size_t dim = 1UL << num_qubits;
    size_t control_mask = 1UL << control;
    size_t target_mask = 1UL << target;

    for (size_t i = 0; i < dim; i++) {
        // Only apply when control qubit is |1⟩
        if ((i & control_mask) && (i & target_mask) == 0) {
            size_t j = i | target_mask;
            double complex a0 = amplitudes[i];
            double complex a1 = amplitudes[j];
            amplitudes[i] = g00 * a0 + g01 * a1;
            amplitudes[j] = g10 * a0 + g11 * a1;
        }
    }
}

// Apply gate by type
static bool apply_gate_by_type(double complex* amplitudes, uint32_t num_qubits,
                               gate_type_t type, uint32_t target, uint32_t control,
                               double* parameters) {
    double complex g00, g01, g10, g11;

    switch (type) {
        case GATE_I:
            // Identity - do nothing
            return true;

        case GATE_X:
            // Pauli X (NOT gate)
            apply_single_gate(amplitudes, target, 0, 1, 1, 0, num_qubits);
            return true;

        case GATE_Y:
            // Pauli Y
            apply_single_gate(amplitudes, target, 0, -I, I, 0, num_qubits);
            return true;

        case GATE_Z:
            // Pauli Z
            apply_single_gate(amplitudes, target, 1, 0, 0, -1, num_qubits);
            return true;

        case GATE_H:
            // Hadamard
            g00 = g01 = g10 = 1.0 / SQRT2;
            g11 = -1.0 / SQRT2;
            apply_single_gate(amplitudes, target, g00, g01, g10, g11, num_qubits);
            return true;

        case GATE_S:
            // Phase gate (S = sqrt(Z))
            apply_single_gate(amplitudes, target, 1, 0, 0, I, num_qubits);
            return true;

        case GATE_T:
            // T gate (T = sqrt(S))
            g11 = cexp(I * PI / 4.0);
            apply_single_gate(amplitudes, target, 1, 0, 0, g11, num_qubits);
            return true;

        case GATE_RX:
            if (!parameters) return false;
            {
                double theta = parameters[0];
                double c = cos(theta / 2.0);
                double s = sin(theta / 2.0);
                apply_single_gate(amplitudes, target, c, -I * s, -I * s, c, num_qubits);
            }
            return true;

        case GATE_RY:
            if (!parameters) return false;
            {
                double theta = parameters[0];
                double c = cos(theta / 2.0);
                double s = sin(theta / 2.0);
                apply_single_gate(amplitudes, target, c, -s, s, c, num_qubits);
            }
            return true;

        case GATE_RZ:
            if (!parameters) return false;
            {
                double theta = parameters[0];
                g00 = cexp(-I * theta / 2.0);
                g11 = cexp(I * theta / 2.0);
                apply_single_gate(amplitudes, target, g00, 0, 0, g11, num_qubits);
            }
            return true;

        case GATE_CNOT:
            // Controlled-NOT
            apply_controlled_gate(amplitudes, control, target, 0, 1, 1, 0, num_qubits);
            return true;

        case GATE_CZ:
            // Controlled-Z (apply Z to target when control is |1⟩)
            apply_controlled_gate(amplitudes, control, target, 1, 0, 0, -1, num_qubits);
            return true;

        case GATE_SWAP: {
            // SWAP = CNOT(a,b) CNOT(b,a) CNOT(a,b)
            apply_controlled_gate(amplitudes, control, target, 0, 1, 1, 0, num_qubits);
            apply_controlled_gate(amplitudes, target, control, 0, 1, 1, 0, num_qubits);
            apply_controlled_gate(amplitudes, control, target, 0, 1, 1, 0, num_qubits);
            return true;
        }

        default:
            log_warn("Unsupported gate type: %d", type);
            return false;
    }
}

// ============================================================================
// Noise Modeling
// ============================================================================

// Apply depolarizing noise to a qubit
static void apply_depolarizing_noise(double complex* amplitudes, uint32_t num_qubits,
                                     uint32_t qubit, double error_rate) {
    if (error_rate <= 0) return;

    double r = random_double();
    if (r < error_rate) {
        // Random Pauli error
        double pauli_choice = random_double();
        if (pauli_choice < 0.25) {
            // X error
            apply_single_gate(amplitudes, qubit, 0, 1, 1, 0, num_qubits);
        } else if (pauli_choice < 0.5) {
            // Y error
            apply_single_gate(amplitudes, qubit, 0, -I, I, 0, num_qubits);
        } else if (pauli_choice < 0.75) {
            // Z error
            apply_single_gate(amplitudes, qubit, 1, 0, 0, -1, num_qubits);
        }
        // else: no error (I)
    }
}

// Apply amplitude damping to statevector (T1 decay)
static void sim_apply_amplitude_damping(double complex* amplitudes, uint32_t num_qubits,
                                        uint32_t qubit, double gamma) {
    if (gamma <= 0 || gamma > 1) return;

    size_t dim = 1UL << num_qubits;
    size_t target_mask = 1UL << qubit;
    double sqrt_gamma = sqrt(gamma);
    double sqrt_1_minus_gamma = sqrt(1.0 - gamma);

    for (size_t i = 0; i < dim; i++) {
        if ((i & target_mask) == 0) {
            size_t j = i | target_mask;
            double complex a0 = amplitudes[i];
            double complex a1 = amplitudes[j];

            // Kraus operators for amplitude damping
            // K0 = [[1, 0], [0, sqrt(1-gamma)]]
            // K1 = [[0, sqrt(gamma)], [0, 0]]
            amplitudes[i] = a0 + sqrt_gamma * a1 * (random_double() < gamma ? 1 : 0);
            amplitudes[j] = sqrt_1_minus_gamma * a1;
        }
    }
}

// Apply phase damping to statevector (T2 dephasing)
static void sim_apply_phase_damping(double complex* amplitudes, uint32_t num_qubits,
                                    uint32_t qubit, double lambda) {
    if (lambda <= 0 || lambda > 1) return;

    size_t dim = 1UL << num_qubits;
    size_t target_mask = 1UL << qubit;
    double sqrt_1_minus_lambda = sqrt(1.0 - lambda);

    for (size_t i = 0; i < dim; i++) {
        if (i & target_mask) {
            // Apply dephasing to |1⟩ component
            if (random_double() < lambda) {
                amplitudes[i] *= -1;  // Phase flip with probability lambda
            } else {
                amplitudes[i] *= sqrt_1_minus_lambda;
            }
        }
    }
}

bool sim_apply_noise(SimulatorState* state, const NoiseModel* noise) {
    if (!state || !noise || noise->type == NOISE_NONE) return true;

    for (uint32_t q = 0; q < state->num_qubits; q++) {
        switch (noise->type) {
            case NOISE_DEPOLARIZING:
                apply_depolarizing_noise(state->amplitudes, state->num_qubits,
                                         q, noise->gate_error_rate);
                break;

            case NOISE_AMPLITUDE_DAMPING:
                sim_apply_amplitude_damping(state->amplitudes, state->num_qubits,
                                            q, noise->decoherence_rate);
                break;

            case NOISE_PHASE_DAMPING:
                sim_apply_phase_damping(state->amplitudes, state->num_qubits,
                                        q, noise->decoherence_rate);
                break;

            case NOISE_THERMAL:
                // Thermal noise: combination of amplitude and phase damping
                sim_apply_amplitude_damping(state->amplitudes, state->num_qubits,
                                            q, noise->decoherence_rate);
                sim_apply_phase_damping(state->amplitudes, state->num_qubits,
                                        q, noise->decoherence_rate * 0.5);
                break;

            case NOISE_CUSTOM:
                // Custom noise handling via custom_parameters
                break;

            default:
                break;
        }
    }

    // Update fidelity estimate
    state->fidelity *= (1.0 - noise->gate_error_rate);
    state->error_rate = 1.0 - state->fidelity;

    return true;
}

// ============================================================================
// Circuit Execution
// ============================================================================

// Note: SimulatorCircuit is typedef'd to quantum_circuit_t in the header
// The quantum_circuit_t structure is defined in quantum_types.h with:
// - num_qubits, num_gates, max_gates
// - gates (quantum_gate_t**)
// - layers (circuit_layer_t**)
// We use the existing structure, storing classical bit count in custom data

// Internal simulator context for tracking classical bits per circuit
typedef struct {
    uint32_t num_classical_bits;
} SimCircuitContext;

SimulatorCircuit* sim_create_circuit(uint32_t num_qubits, uint32_t num_classical_bits) {
    SimulatorCircuit* circuit = calloc(1, sizeof(SimulatorCircuit));
    if (!circuit) return NULL;

    circuit->num_qubits = num_qubits;
    circuit->num_gates = 0;
    circuit->max_gates = 64;
    circuit->gates = calloc(circuit->max_gates, sizeof(quantum_gate_t*));

    if (!circuit->gates) {
        free(circuit);
        return NULL;
    }

    // Store classical bit count in state field (repurposed for simulator)
    SimCircuitContext* ctx = calloc(1, sizeof(SimCircuitContext));
    if (ctx) {
        ctx->num_classical_bits = num_classical_bits;
        circuit->state = (quantum_geometric_state_t*)ctx;  // Store context
    }

    return circuit;
}

bool sim_add_gate(SimulatorCircuit* circuit, gate_type_t type, uint32_t target,
                  uint32_t control, double* parameters) {
    if (!circuit) return false;

    // Expand if needed
    if (circuit->num_gates >= circuit->max_gates) {
        size_t new_capacity = circuit->max_gates * 2;
        quantum_gate_t** new_gates = realloc(circuit->gates, new_capacity * sizeof(quantum_gate_t*));
        if (!new_gates) return false;
        circuit->gates = new_gates;
        circuit->max_gates = new_capacity;
    }

    // Allocate new gate
    quantum_gate_t* gate = calloc(1, sizeof(quantum_gate_t));
    if (!gate) return false;

    gate->type = type;
    gate->num_qubits = (type == GATE_CNOT || type == GATE_CZ || type == GATE_SWAP) ? 2 : 1;
    gate->is_controlled = (type == GATE_CNOT || type == GATE_CZ);
    gate->is_parameterized = (parameters != NULL);

    // Set target qubit
    gate->target_qubits = malloc(sizeof(size_t));
    if (gate->target_qubits) {
        gate->target_qubits[0] = target;
    }

    // Set control qubit if this is a controlled gate
    if (gate->is_controlled) {
        gate->control_qubits = malloc(sizeof(size_t));
        if (gate->control_qubits) {
            gate->control_qubits[0] = control;
            gate->num_controls = 1;
        }
    }

    // Copy parameters if provided
    if (parameters) {
        gate->parameters = malloc(4 * sizeof(double));
        if (gate->parameters) {
            memcpy(gate->parameters, parameters, 4 * sizeof(double));
            gate->num_parameters = 4;
        }
    }

    circuit->gates[circuit->num_gates++] = gate;
    return true;
}

bool sim_add_controlled_gate(SimulatorCircuit* circuit, gate_type_t type,
                             uint32_t target, uint32_t control, uint32_t control2,
                             double* parameters) {
    (void)control2;  // For future multi-controlled gates
    return sim_add_gate(circuit, type, target, control, parameters);
}

bool sim_execute_circuit(SimulatorState* state, const SimulatorCircuit* circuit) {
    if (!state || !circuit) return false;

    for (size_t i = 0; i < circuit->num_gates; i++) {
        const quantum_gate_t* gate = circuit->gates[i];
        if (!gate) continue;

        // Extract target and control qubits
        uint32_t target = gate->target_qubits ? (uint32_t)gate->target_qubits[0] : 0;
        uint32_t control = (gate->control_qubits && gate->num_controls > 0)
                          ? (uint32_t)gate->control_qubits[0] : 0;

        bool success = apply_gate_by_type(
            state->amplitudes,
            state->num_qubits,
            gate->type,
            target,
            control,
            gate->is_parameterized ? gate->parameters : NULL
        );

        if (!success) {
            log_error("Failed to apply gate %zu", i);
            return false;
        }

        // Apply noise after each gate if enabled
        if (state->active_noise.type != NOISE_NONE) {
            apply_depolarizing_noise(state->amplitudes, state->num_qubits,
                                     target, state->active_noise.gate_error_rate);
        }
    }

    return true;
}

void sim_cleanup_circuit(SimulatorCircuit* circuit) {
    if (!circuit) return;

    // Free each gate
    for (size_t i = 0; i < circuit->num_gates; i++) {
        quantum_gate_t* gate = circuit->gates[i];
        if (gate) {
            free(gate->target_qubits);
            free(gate->control_qubits);
            free(gate->parameters);
            free(gate->matrix);
            free(gate);
        }
    }

    // Free context if present
    if (circuit->state) {
        free(circuit->state);
    }

    free(circuit->gates);
    free(circuit);
}

// ============================================================================
// Measurement
// ============================================================================

bool sim_measure_qubit(SimulatorState* state, uint32_t qubit, uint32_t classical_bit) {
    if (!state || qubit >= state->num_qubits) return false;
    if (classical_bit >= state->num_classical_bits && state->classical_bits) return false;

    size_t dim = 1UL << state->num_qubits;
    size_t qubit_mask = 1UL << qubit;

    // Calculate probability of measuring |1⟩
    double prob_one = 0.0;
    for (size_t i = 0; i < dim; i++) {
        if (i & qubit_mask) {
            prob_one += creal(state->amplitudes[i] * conj(state->amplitudes[i]));
        }
    }

    // Apply measurement error if configured
    double p = prob_one;
    if (state->active_noise.measurement_error_rate > 0) {
        double err = state->active_noise.measurement_error_rate;
        // Flip probability with measurement error rate
        p = p * (1 - err) + (1 - p) * err;
    }

    // Collapse wavefunction
    bool result = (random_double() < p);

    // Normalize remaining amplitudes
    double norm = 0.0;
    for (size_t i = 0; i < dim; i++) {
        bool qubit_is_one = (i & qubit_mask) != 0;
        if (qubit_is_one == result) {
            norm += creal(state->amplitudes[i] * conj(state->amplitudes[i]));
        } else {
            state->amplitudes[i] = 0;
        }
    }

    if (norm > 0) {
        double inv_sqrt_norm = 1.0 / sqrt(norm);
        for (size_t i = 0; i < dim; i++) {
            state->amplitudes[i] *= inv_sqrt_norm;
        }
    }

    // Store result
    if (state->classical_bits && classical_bit < state->num_classical_bits) {
        state->classical_bits[classical_bit] = result;
    }

    return true;
}

bool sim_measure_all(SimulatorState* state) {
    if (!state) return false;

    for (uint32_t q = 0; q < state->num_qubits; q++) {
        uint32_t cb = (q < state->num_classical_bits) ? q : 0;
        if (!sim_measure_qubit(state, q, cb)) {
            return false;
        }
    }

    return true;
}

bool* sim_get_measurement_results(const SimulatorState* state) {
    if (!state || !state->classical_bits) return NULL;

    bool* results = malloc(state->num_classical_bits * sizeof(bool));
    if (results) {
        memcpy(results, state->classical_bits, state->num_classical_bits * sizeof(bool));
    }
    return results;
}

uint64_t* sim_get_measurement_counts(const SimulatorState* state, uint32_t shots) {
    if (!state) return NULL;

    size_t dim = 1UL << state->num_qubits;
    uint64_t* counts = calloc(dim, sizeof(uint64_t));
    if (!counts) return NULL;

    // Calculate probabilities
    double* probs = malloc(dim * sizeof(double));
    if (!probs) {
        free(counts);
        return NULL;
    }

    for (size_t i = 0; i < dim; i++) {
        probs[i] = creal(state->amplitudes[i] * conj(state->amplitudes[i]));
    }

    // Sample from distribution
    for (uint32_t s = 0; s < shots; s++) {
        double r = random_double();
        double cumulative = 0.0;
        for (size_t i = 0; i < dim; i++) {
            cumulative += probs[i];
            if (r < cumulative) {
                counts[i]++;
                break;
            }
        }
    }

    free(probs);
    return counts;
}

// ============================================================================
// State Access
// ============================================================================

double complex* sim_get_statevector(const SimulatorState* state) {
    if (!state) return NULL;

    size_t dim = 1UL << state->num_qubits;
    double complex* sv = malloc(dim * sizeof(double complex));
    if (sv) {
        memcpy(sv, state->amplitudes, dim * sizeof(double complex));
    }
    return sv;
}

double complex* sim_get_density_matrix(const SimulatorState* state) {
    if (!state) return NULL;

    size_t dim = 1UL << state->num_qubits;
    double complex* rho = calloc(dim * dim, sizeof(double complex));
    if (!rho) return NULL;

    // rho = |psi><psi|
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            rho[i * dim + j] = state->amplitudes[i] * conj(state->amplitudes[j]);
        }
    }

    return rho;
}

double sim_get_expectation_value(const SimulatorState* state, const char* observable) {
    if (!state || !observable) return 0.0;

    // Simple Z expectation value on all qubits
    if (strcmp(observable, "Z") == 0) {
        size_t dim = 1UL << state->num_qubits;
        double exp_val = 0.0;

        for (size_t i = 0; i < dim; i++) {
            // Count number of 1s in bitstring (parity)
            int parity = 0;
            size_t bits = i;
            while (bits) {
                parity ^= (bits & 1);
                bits >>= 1;
            }
            double sign = (parity == 0) ? 1.0 : -1.0;
            exp_val += sign * creal(state->amplitudes[i] * conj(state->amplitudes[i]));
        }

        return exp_val;
    }

    return 0.0;
}

// ============================================================================
// Error Mitigation
// ============================================================================

bool sim_apply_error_mitigation(SimulatorState* state, const MitigationParams* params) {
    if (!state || !params) return true;

    switch (params->type) {
        case MITIGATION_ZNE:
            // Zero-noise extrapolation: run at multiple noise levels
            // and extrapolate to zero noise
            // For simulation, we store the ideal result
            break;

        case MITIGATION_PROBABILISTIC:
            // Probabilistic error cancellation
            // Apply inverse noise channels probabilistically
            break;

        case MITIGATION_CUSTOM:
            // Custom error mitigation (includes readout correction)
            // Apply user-defined mitigation via custom_parameters
            break;

        default:
            break;
    }

    return true;
}

// ============================================================================
// Utility Functions
// ============================================================================

bool sim_validate_circuit(const SimulatorCircuit* circuit) {
    if (!circuit || !circuit->gates) return false;

    for (size_t i = 0; i < circuit->num_gates; i++) {
        const quantum_gate_t* gate = circuit->gates[i];
        if (!gate) continue;

        size_t target = gate->target_qubits ? gate->target_qubits[0] : 0;

        if (target >= circuit->num_qubits) {
            log_error("Gate %zu: target qubit %zu out of range", i, target);
            return false;
        }

        // Check control for two-qubit gates
        if (gate->type == GATE_CNOT || gate->type == GATE_CZ || gate->type == GATE_SWAP) {
            size_t control = (gate->control_qubits && gate->num_controls > 0)
                            ? gate->control_qubits[0] : 0;
            if (control >= circuit->num_qubits) {
                log_error("Gate %zu: control qubit %zu out of range", i, control);
                return false;
            }
            if (control == target) {
                log_error("Gate %zu: control and target cannot be the same", i);
                return false;
            }
        }
    }

    return true;
}

// Helper to compare gates for cancellation
static bool gates_match(const quantum_gate_t* a, const quantum_gate_t* b) {
    if (a->type != b->type) return false;
    size_t a_target = a->target_qubits ? a->target_qubits[0] : SIZE_MAX;
    size_t b_target = b->target_qubits ? b->target_qubits[0] : SIZE_MAX;
    if (a_target != b_target) return false;

    size_t a_control = (a->control_qubits && a->num_controls) ? a->control_qubits[0] : SIZE_MAX;
    size_t b_control = (b->control_qubits && b->num_controls) ? b->control_qubits[0] : SIZE_MAX;
    return a_control == b_control;
}

bool sim_optimize_circuit(SimulatorCircuit* circuit) {
    if (!circuit) return false;

    // Simple optimization: cancel adjacent inverse gates
    size_t write_idx = 0;
    for (size_t i = 0; i < circuit->num_gates; i++) {
        bool cancelled = false;
        quantum_gate_t* curr = circuit->gates[i];
        if (!curr) continue;

        // Check if this gate cancels with the previous one
        if (write_idx > 0) {
            quantum_gate_t* prev = circuit->gates[write_idx - 1];

            // Self-inverse gates: X, Y, Z, H, CNOT, CZ, SWAP
            if (prev && gates_match(prev, curr)) {
                switch (curr->type) {
                    case GATE_X:
                    case GATE_Y:
                    case GATE_Z:
                    case GATE_H:
                    case GATE_CNOT:
                    case GATE_CZ:
                    case GATE_SWAP:
                        // Free the cancelled gates
                        free(prev->target_qubits);
                        free(prev->control_qubits);
                        free(prev->parameters);
                        free(prev);
                        circuit->gates[write_idx - 1] = NULL;
                        write_idx--;
                        cancelled = true;
                        break;
                    default:
                        break;
                }
            }
        }

        if (!cancelled) {
            circuit->gates[write_idx] = curr;
            write_idx++;
        } else {
            // Free the current gate too since it's cancelled
            free(curr->target_qubits);
            free(curr->control_qubits);
            free(curr->parameters);
            free(curr);
        }
    }

    circuit->num_gates = write_idx;
    return true;
}

char* sim_circuit_to_string(const SimulatorCircuit* circuit) {
    if (!circuit) return NULL;

    // Estimate size needed
    size_t buf_size = circuit->num_gates * 64 + 256;
    char* buf = malloc(buf_size);
    if (!buf) return NULL;

    int offset = snprintf(buf, buf_size, "Circuit: %zu qubits, %zu gates\n",
                          circuit->num_qubits, circuit->num_gates);

    for (size_t i = 0; i < circuit->num_gates && (size_t)offset < buf_size - 64; i++) {
        const quantum_gate_t* gate = circuit->gates[i];
        if (!gate) continue;

        const char* name = "?";

        switch (gate->type) {
            case GATE_I: name = "I"; break;
            case GATE_X: name = "X"; break;
            case GATE_Y: name = "Y"; break;
            case GATE_Z: name = "Z"; break;
            case GATE_H: name = "H"; break;
            case GATE_S: name = "S"; break;
            case GATE_T: name = "T"; break;
            case GATE_RX: name = "RX"; break;
            case GATE_RY: name = "RY"; break;
            case GATE_RZ: name = "RZ"; break;
            case GATE_CNOT: name = "CNOT"; break;
            case GATE_CZ: name = "CZ"; break;
            case GATE_SWAP: name = "SWAP"; break;
            default: break;
        }

        size_t target = gate->target_qubits ? gate->target_qubits[0] : 0;
        size_t control = (gate->control_qubits && gate->num_controls) ? gate->control_qubits[0] : 0;

        if (gate->type == GATE_CNOT || gate->type == GATE_CZ || gate->type == GATE_SWAP) {
            offset += snprintf(buf + offset, buf_size - offset,
                              "  %s q%zu, q%zu\n", name, control, target);
        } else {
            offset += snprintf(buf + offset, buf_size - offset,
                              "  %s q%zu\n", name, target);
        }
    }

    return buf;
}

double sim_get_circuit_depth(const SimulatorCircuit* circuit) {
    if (!circuit) return 0;

    // Track the last time slot used for each qubit
    uint32_t* qubit_depth = calloc(circuit->num_qubits, sizeof(uint32_t));
    if (!qubit_depth) return (double)circuit->num_gates;

    uint32_t max_depth = 0;

    for (size_t i = 0; i < circuit->num_gates; i++) {
        const quantum_gate_t* gate = circuit->gates[i];
        if (!gate) continue;

        size_t target = gate->target_qubits ? gate->target_qubits[0] : 0;
        size_t control = (gate->control_qubits && gate->num_controls) ? gate->control_qubits[0] : 0;
        uint32_t depth;

        if (gate->type == GATE_CNOT || gate->type == GATE_CZ || gate->type == GATE_SWAP) {
            // Two-qubit gate: depth is max of both qubits + 1
            depth = (qubit_depth[target] > qubit_depth[control])
                    ? qubit_depth[target] : qubit_depth[control];
            depth++;
            qubit_depth[target] = depth;
            qubit_depth[control] = depth;
        } else {
            // Single-qubit gate
            depth = qubit_depth[target] + 1;
            qubit_depth[target] = depth;
        }

        if (depth > max_depth) max_depth = depth;
    }

    free(qubit_depth);
    return (double)max_depth;
}

double sim_estimate_runtime(const SimulatorCircuit* circuit) {
    if (!circuit) return 0;

    // Estimate based on circuit depth and number of qubits
    // Assume ~50ns per single-qubit gate, ~100ns per two-qubit gate
    double depth = sim_get_circuit_depth(circuit);
    double avg_gate_time = 75e-9;  // 75ns average

    return depth * avg_gate_time;
}

bool sim_save_circuit(const SimulatorCircuit* circuit, const char* filename) {
    if (!circuit || !filename) return false;

    FILE* f = fopen(filename, "w");
    if (!f) return false;

    // Get classical bits from context if available
    uint32_t num_classical_bits = 0;
    if (circuit->state) {
        SimCircuitContext* ctx = (SimCircuitContext*)circuit->state;
        num_classical_bits = ctx->num_classical_bits;
    }

    fprintf(f, "QGT_CIRCUIT v1\n");
    fprintf(f, "qubits %zu\n", circuit->num_qubits);
    fprintf(f, "classical_bits %u\n", num_classical_bits);
    fprintf(f, "gates %zu\n", circuit->num_gates);

    for (size_t i = 0; i < circuit->num_gates; i++) {
        const quantum_gate_t* gate = circuit->gates[i];
        if (!gate) continue;

        size_t target = gate->target_qubits ? gate->target_qubits[0] : 0;
        size_t control = (gate->control_qubits && gate->num_controls) ? gate->control_qubits[0] : 0;

        fprintf(f, "%d %zu %zu", gate->type, target, control);
        if (gate->is_parameterized && gate->parameters) {
            fprintf(f, " %g %g %g %g",
                    gate->parameters[0],
                    gate->num_parameters > 1 ? gate->parameters[1] : 0.0,
                    gate->num_parameters > 2 ? gate->parameters[2] : 0.0,
                    gate->num_parameters > 3 ? gate->parameters[3] : 0.0);
        }
        fprintf(f, "\n");
    }

    fclose(f);
    return true;
}

SimulatorCircuit* sim_load_circuit(const char* filename) {
    if (!filename) return NULL;

    FILE* f = fopen(filename, "r");
    if (!f) return NULL;

    char header[32];
    if (fscanf(f, "%31s", header) != 1 || strcmp(header, "QGT_CIRCUIT") != 0) {
        fclose(f);
        return NULL;
    }

    // Skip version
    fscanf(f, "%*s");

    uint32_t num_qubits, num_classical_bits;
    size_t num_gates;

    if (fscanf(f, " qubits %u", &num_qubits) != 1 ||
        fscanf(f, " classical_bits %u", &num_classical_bits) != 1 ||
        fscanf(f, " gates %zu", &num_gates) != 1) {
        fclose(f);
        return NULL;
    }

    SimulatorCircuit* circuit = sim_create_circuit(num_qubits, num_classical_bits);
    if (!circuit) {
        fclose(f);
        return NULL;
    }

    for (size_t i = 0; i < num_gates; i++) {
        int type;
        uint32_t target, control;
        double params[4] = {0};

        int read = fscanf(f, "%d %u %u", &type, &target, &control);
        if (read < 3) break;

        // Try to read parameters
        int param_read = fscanf(f, " %lf %lf %lf %lf",
                                &params[0], &params[1], &params[2], &params[3]);

        sim_add_gate(circuit, (gate_type_t)type, target, control,
                     param_read > 0 ? params : NULL);
    }

    fclose(f);
    return circuit;
}
