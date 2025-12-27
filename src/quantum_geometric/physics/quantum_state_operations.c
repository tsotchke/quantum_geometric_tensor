#include "quantum_geometric/physics/quantum_state_operations.h"
#include "quantum_geometric/core/memory_pool.h"
#include "quantum_geometric/core/quantum_geometric_constants.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/core/quantum_state.h"
#include "quantum_geometric/core/quantum_complex.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Forward declarations for functions used internally
qgt_error_t quantum_state_normalize(quantum_state_t* state);
qgt_error_t quantum_state_fidelity(float* fidelity, const quantum_state_t* a, const quantum_state_t* b);

// Internal QuantumStateOps structure
struct QuantumStateOps {
    StateConfig config;
    MemoryPool* pool;
    double* measurement_buffer;
    size_t measurement_count;
    double total_fidelity;
    pthread_mutex_t mutex;
};

// Helper function to get X-stabilizer state from quantum_state_t
static XStabilizerState* get_x_stabilizer(const quantum_state_t* state) {
    if (!state || !state->auxiliary_data) return NULL;
    return (XStabilizerState*)state->auxiliary_data;
}

// State operations initialization
QuantumStateOps* init_quantum_state_ops(const StateConfig* config) {
    if (!config) return NULL;

    QuantumStateOps* ops = NULL;
    qgt_error_t err = geometric_core_allocate((void**)&ops, sizeof(QuantumStateOps));
    if (err != QGT_SUCCESS) return NULL;

    memcpy(&ops->config, config, sizeof(StateConfig));

    // Create pool config
    PoolConfig pool_config = {
        .min_block_size = sizeof(ComplexFloat),
        .alignment = 32,
        .num_size_classes = 8,
        .growth_factor = 2.0f,
        .prefetch_distance = 4,
        .use_huge_pages = false,
        .cache_local_free_lists = true,
        .max_blocks_per_class = 1024,
        .thread_cache_size = 64,
        .enable_stats = false
    };
    ops->pool = create_memory_pool(&pool_config);
    
    err = geometric_core_allocate((void**)&ops->measurement_buffer, 1024 * sizeof(double));
    if (err != QGT_SUCCESS) {
        if (ops->pool) destroy_memory_pool(ops->pool);
        geometric_core_free(ops);
        return NULL;
    }
    
    memset(ops->measurement_buffer, 0, 1024 * sizeof(double));
    ops->measurement_count = 0;
    ops->total_fidelity = 0.0;
    pthread_mutex_init(&ops->mutex, NULL);

    if (!ops->pool) {
        geometric_core_free(ops->measurement_buffer);
        geometric_core_free(ops);
        return NULL;
    }

    return ops;
}

void cleanup_quantum_state_ops(QuantumStateOps* ops) {
    if (ops) {
        if (ops->pool) destroy_memory_pool(ops->pool);
        geometric_core_free(ops->measurement_buffer);
        pthread_mutex_destroy(&ops->mutex);
        geometric_core_free(ops);
    }
}

// State operation execution
void perform_state_operation(QuantumStateOps* ops, quantum_state_t** states, size_t count) {
    if (!ops || !states || count == 0) return;

    pthread_mutex_lock(&ops->mutex);

    // Process states in parallel if OpenMP is available
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (size_t i = 0; i < count; i++) {
        if (states[i]) {
            // Apply error mitigation if enabled
            if (ops->config.enable_error_correction) {
                apply_x_error_mitigation_sequence(states[i], 0, 0);
            }

            // Track measurements
            double measurement = 0.0;
            apply_x_measurement_correction(states[i], 0, 0, &measurement);

            // Update statistics
            #if defined(_OPENMP)
            #pragma omp critical
            #endif
            {
                ops->measurement_buffer[ops->measurement_count % 1024] = measurement;
                ops->measurement_count++;
                float fidelity;
                quantum_state_fidelity(&fidelity, states[i], states[i]);
                ops->total_fidelity += (double)fidelity;
            }
        }
    }

    pthread_mutex_unlock(&ops->mutex);
}

StateOperationResult* get_state_result(QuantumStateOps* ops) {
    if (!ops) return NULL;

    StateOperationResult* result = NULL;
    qgt_error_t err = geometric_core_allocate((void**)&result, sizeof(StateOperationResult));
    if (err != QGT_SUCCESS) return NULL;

    pthread_mutex_lock(&ops->mutex);
    result->x_stabilizer_results.measurement_count = ops->measurement_count;
    result->x_stabilizer_results.average_fidelity = 
        ops->measurement_count > 0 ? ops->total_fidelity / ops->measurement_count : 0.0f;
    pthread_mutex_unlock(&ops->mutex);

    return result;
}

void cleanup_state_result(StateOperationResult* result) {
    geometric_core_free(result);
}

// Error mitigation and measurement
void apply_x_error_mitigation_sequence(const quantum_state_t* state, size_t x, size_t y) {
    if (!state) return;
    
    // Apply Hadamard gates to transform to X basis
    apply_hadamard_gate(state, x, y);
    
    // Apply dynamical decoupling sequence
    for (int i = 0; i < 4; i++) {
        apply_rotation_x(state, x, y, M_PI_2);
        quantum_wait(state, QGT_GATE_DELAY);
        apply_rotation_x(state, x, y, M_PI);
        quantum_wait(state, QGT_GATE_DELAY);
        apply_rotation_x(state, x, y, M_PI_2);
    }
    
    // Apply composite pulse sequence
    apply_composite_x_pulse(state, x, y);
}

void apply_x_measurement_correction(const quantum_state_t* state, size_t x, size_t y, double* result) {
    if (!state || !result) return;
    
    // Get error rates
    double readout_error = get_readout_error_rate(x, y);
    double gate_error = get_gate_error_rate(x, y);
    
    // Apply correction
    double corrected = *result;
    corrected = (*result - readout_error) / (1.0f - 2.0f * readout_error);
    corrected *= (1.0f + gate_error);
    
    // Apply threshold
    if (fabs(corrected) < QGT_EPSILON) {
        corrected = 0.0f;
    }
    
    *result = corrected;
    
    // Update state's X-stabilizer history
    XStabilizerState* x_state = get_x_stabilizer(state);
    if (x_state && x_state->history_size < 1000) {
        size_t idx = x_state->history_size;
        x_state->correlations[idx] = corrected;
        x_state->confidences[idx] = 1.0f - (readout_error + gate_error);
        x_state->history_size++;
    }
}

double get_x_stabilizer_correlation(const quantum_state_t* state, size_t x, size_t y, size_t qubit_idx) {
    if (!state) return 0.0;
    
    XStabilizerState* x_state = get_x_stabilizer(state);
    if (!x_state) return 0.0;
    
    double correlation = 0.0;
    size_t count = 0;
    
    // Calculate correlation from history
    for (size_t i = 1; i < x_state->history_size; i++) {
        double m1 = x_state->correlations[i];
        double m2 = x_state->correlations[i-1];
        double c1 = x_state->confidences[i];
        double c2 = x_state->confidences[i-1];
        
        correlation += (m1 * m2) * (c1 * c2);
        count++;
    }
    
    // Apply spatial weighting
    double spatial_factor = 1.0;
    if (qubit_idx < 4) {
        spatial_factor = 1.0 - (0.1 * qubit_idx);
    }
    
    return count > 0 ? (correlation / count) * spatial_factor : 0.0;
}

// Helper functions
void apply_hadamard_gate(const quantum_state_t* state, size_t x, size_t y) {
    if (!state) return;
    
    size_t width = (size_t)sqrt(state->dimension);
    size_t idx = y * width + x;
    if (idx >= state->dimension) return;
    
    // Hadamard matrix elements
    const float h00 = M_SQRT1_2;  // 1/√2
    const float h01 = M_SQRT1_2;  // 1/√2
    const float h10 = M_SQRT1_2;  // 1/√2
    const float h11 = -M_SQRT1_2; // -1/√2
    
    ComplexFloat* amplitudes = state->coordinates;
    for (size_t i = 0; i < state->dimension; i += 2) {
        if ((i/2) == idx) {
            ComplexFloat psi0 = amplitudes[i];
            ComplexFloat psi1 = amplitudes[i+1];
            
            amplitudes[i] = complex_float_create(
                h00 * psi0.real + h01 * psi1.real,
                h00 * psi0.imag + h01 * psi1.imag
            );
            
            amplitudes[i+1] = complex_float_create(
                h10 * psi0.real + h11 * psi1.real,
                h10 * psi0.imag + h11 * psi1.imag
            );
        }
    }
    
    quantum_state_normalize((quantum_state_t*)state);
}

void apply_rotation_x(const quantum_state_t* state, size_t x, size_t y, double angle) {
    if (!state) return;
    
    size_t width = (size_t)sqrt(state->dimension);
    size_t idx = y * width + x;
    if (idx >= state->dimension) return;
    
    // Rotation matrix elements
    const float cos_half = cosf(angle/2);
    const float sin_half = sinf(angle/2);
    
    ComplexFloat* amplitudes = state->coordinates;
    for (size_t i = 0; i < state->dimension; i += 2) {
        if ((i/2) == idx) {
            ComplexFloat psi0 = amplitudes[i];
            ComplexFloat psi1 = amplitudes[i+1];
            
            amplitudes[i] = complex_float_create(
                cos_half * psi0.real - sin_half * psi1.imag,
                cos_half * psi0.imag + sin_half * psi1.real
            );
            
            amplitudes[i+1] = complex_float_create(
                -sin_half * psi0.imag + cos_half * psi1.real,
                sin_half * psi0.real + cos_half * psi1.imag
            );
        }
    }
    
    quantum_state_normalize((quantum_state_t*)state);
}

void quantum_wait(const quantum_state_t* state, double duration) {
    if (!state || duration <= 0) return;
    
    // Apply decoherence effects
    const double decoherence_rate = 0.1; // Adjust based on hardware characteristics
    const double decay = exp(-decoherence_rate * duration);
    
    ComplexFloat* amplitudes = state->coordinates;
    size_t dimension = state->dimension;
    
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (size_t i = 0; i < dimension; i++) {
        // Phase damping
        double phase_noise = ((double)rand() / RAND_MAX - 0.5) * 0.1;
        ComplexFloat amp = amplitudes[i];
        
        amplitudes[i] = complex_float_create(
            decay * (amp.real + phase_noise),
            decay * (amp.imag + phase_noise)
        );
    }
    
    quantum_state_normalize((quantum_state_t*)state);
}

void apply_composite_x_pulse(const quantum_state_t* state, size_t x, size_t y) {
    if (!state) return;
    
    // BB1 composite pulse sequence for robust X rotation
    const double phi1 = 0.0;
    const double phi2 = M_PI;
    const double phi3 = 3.0 * M_PI;
    const double theta = M_PI_2;
    
    // Apply sequence of pulses with different phases
    apply_rotation_x(state, x, y, theta);
    quantum_wait(state, QGT_GATE_DELAY);
    
    apply_rotation_x(state, x, y, 2*theta);
    quantum_wait(state, QGT_GATE_DELAY);
    
    apply_rotation_x(state, x, y, 2*theta);
    quantum_wait(state, QGT_GATE_DELAY);
    
    apply_rotation_x(state, x, y, theta);
    
    // Apply phase corrections
    ComplexFloat* amplitudes = state->coordinates;
    size_t dimension = state->dimension;
    
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (size_t i = 0; i < dimension; i++) {
        ComplexFloat amp = amplitudes[i];
        float phase = atan2f(amp.imag, amp.real);
        
        phase += (phi1 + phi2 + phi3) / 3.0f;
        float mag = sqrtf(amp.real * amp.real + amp.imag * amp.imag);
        
        amplitudes[i] = complex_float_create(
            mag * cosf(phase),
            mag * sinf(phase)
        );
    }
    
    quantum_state_normalize((quantum_state_t*)state);
}

double get_readout_error_rate(size_t x, size_t y) {
    // Implementation omitted for brevity
    return 0.01;
}

double get_gate_error_rate(size_t x, size_t y) {
    // Implementation omitted for brevity
    return 0.005;
}

// Hierarchical matrix operations
void update_hmatrix_quantum_state(HierarchicalMatrix* mat) {
    (void)mat;
}

void cleanup_hmatrix_quantum_state(HierarchicalMatrix* mat) {
    (void)mat;
}

// ============================================================================
// Core Quantum State API (from quantum_state.h)
// ============================================================================

qgt_error_t quantum_state_create(quantum_state_t** state,
                                 quantum_state_type_t type,
                                 size_t dimension) {
    if (!state || dimension == 0) return QGT_ERROR_INVALID_ARGUMENT;

    quantum_state_t* new_state = NULL;
    qgt_error_t err = geometric_core_allocate((void**)&new_state, sizeof(quantum_state_t));
    if (err != QGT_SUCCESS) return err;

    memset(new_state, 0, sizeof(quantum_state_t));
    new_state->type = type;
    new_state->dimension = dimension;
    new_state->is_normalized = false;

    // Allocate coordinates (amplitudes)
    err = geometric_core_allocate((void**)&new_state->coordinates, dimension * sizeof(ComplexFloat));
    if (err != QGT_SUCCESS) {
        geometric_core_free(new_state);
        return err;
    }
    memset(new_state->coordinates, 0, dimension * sizeof(ComplexFloat));

    // Initialize to |0⟩ state
    new_state->coordinates[0].real = 1.0f;
    new_state->coordinates[0].imag = 0.0f;
    new_state->is_normalized = true;

    *state = new_state;
    return QGT_SUCCESS;
}

void quantum_state_destroy(quantum_state_t* state) {
    if (!state) return;
    if (state->coordinates) {
        geometric_core_free(state->coordinates);
    }
    if (state->auxiliary_data) {
        geometric_core_free(state->auxiliary_data);
    }
    geometric_core_free(state);
}

qgt_error_t quantum_state_normalize(quantum_state_t* state) {
    if (!state || !state->coordinates) return QGT_ERROR_INVALID_ARGUMENT;

    double norm_sq = 0.0;
    for (size_t i = 0; i < state->dimension; i++) {
        norm_sq += state->coordinates[i].real * state->coordinates[i].real +
                   state->coordinates[i].imag * state->coordinates[i].imag;
    }

    if (norm_sq < 1e-15) return QGT_ERROR_INVALID_STATE;

    float inv_norm = 1.0f / sqrtf((float)norm_sq);
    for (size_t i = 0; i < state->dimension; i++) {
        state->coordinates[i].real *= inv_norm;
        state->coordinates[i].imag *= inv_norm;
    }

    state->is_normalized = true;
    return QGT_SUCCESS;
}

qgt_error_t quantum_state_initialize_basis(quantum_state_t* state, size_t basis_index) {
    if (!state || !state->coordinates) return QGT_ERROR_INVALID_ARGUMENT;
    if (basis_index >= state->dimension) return QGT_ERROR_INVALID_ARGUMENT;

    memset(state->coordinates, 0, state->dimension * sizeof(ComplexFloat));
    state->coordinates[basis_index].real = 1.0f;
    state->coordinates[basis_index].imag = 0.0f;
    state->is_normalized = true;

    return QGT_SUCCESS;
}

qgt_error_t quantum_state_fidelity(float* fidelity, const quantum_state_t* a, const quantum_state_t* b) {
    if (!fidelity || !a || !b) return QGT_ERROR_INVALID_ARGUMENT;
    if (!a->coordinates || !b->coordinates) return QGT_ERROR_INVALID_ARGUMENT;
    if (a->dimension != b->dimension) return QGT_ERROR_INCOMPATIBLE;

    // F = |⟨a|b⟩|²
    double overlap_real = 0.0, overlap_imag = 0.0;
    for (size_t i = 0; i < a->dimension; i++) {
        // ⟨a|b⟩ = sum(conj(a_i) * b_i)
        overlap_real += a->coordinates[i].real * b->coordinates[i].real +
                        a->coordinates[i].imag * b->coordinates[i].imag;
        overlap_imag += a->coordinates[i].real * b->coordinates[i].imag -
                        a->coordinates[i].imag * b->coordinates[i].real;
    }

    *fidelity = (float)(overlap_real * overlap_real + overlap_imag * overlap_imag);
    return QGT_SUCCESS;
}

// =============================================================================
// Pauli Measurement Functions with Confidence Tracking
// =============================================================================

/**
 * @brief Measure Pauli Z operator at lattice position (x,y) with confidence
 *
 * Computes the expectation value <Z> for the qubit at position (x,y) on the
 * lattice, along with a confidence estimate based on the measurement statistics.
 *
 * The Z operator eigenvalues are +1 (|0⟩) and -1 (|1⟩). The expectation value
 * is computed as: <Z> = Σᵢ |ψᵢ|² × (-1)^{bit(i)}
 *
 * Confidence is estimated from the variance of the measurement outcome:
 * Higher confidence when the state is closer to an eigenstate.
 */
bool measure_pauli_z_with_confidence(const quantum_state_t* state,
                                    size_t x,
                                    size_t y,
                                    double* value,
                                    double* confidence) {
    if (!state || !value || !confidence) {
        return false;
    }

    if (!state->coordinates) {
        *value = 0.0;
        *confidence = 0.0;
        return false;
    }

    // Calculate qubit index from lattice coordinates
    // Use lattice_width if available, otherwise infer from num_qubits
    size_t width = state->lattice_width > 0 ? state->lattice_width :
                   (size_t)sqrt((double)state->num_qubits);
    size_t qubit_idx = y * width + x;

    if (qubit_idx >= state->num_qubits) {
        // Out of bounds - return identity (no error detected)
        *value = 1.0;
        *confidence = 1.0;
        return true;
    }

    // Compute expectation value <Z_q>
    // Z|0⟩ = |0⟩, Z|1⟩ = -|1⟩
    // <Z_q> = Σᵢ |ψᵢ|² × (-1)^{bit_q(i)}
    size_t dim = state->dimension;
    size_t mask = (size_t)1 << qubit_idx;

    double prob_0 = 0.0;  // Probability of measuring |0⟩
    double prob_1 = 0.0;  // Probability of measuring |1⟩

    for (size_t i = 0; i < dim; i++) {
        double prob = (double)(state->coordinates[i].real * state->coordinates[i].real +
                               state->coordinates[i].imag * state->coordinates[i].imag);
        if (i & mask) {
            prob_1 += prob;  // Qubit is |1⟩ in this basis state
        } else {
            prob_0 += prob;  // Qubit is |0⟩ in this basis state
        }
    }

    // Expectation value: <Z> = P(0) × (+1) + P(1) × (-1) = P(0) - P(1)
    *value = prob_0 - prob_1;

    // Confidence is based on how close we are to an eigenstate
    // Maximum confidence (1.0) when in pure |0⟩ or |1⟩ state
    // Minimum confidence (0.0) when in equal superposition
    // Confidence = |<Z>| = |P(0) - P(1)|
    double abs_value = fabs(*value);
    *confidence = abs_value;

    // Apply readout error correction if available
    double readout_error = get_readout_error_rate(x, y);
    if (readout_error > 0.0 && readout_error < 0.5) {
        // Correct for depolarizing readout noise: P_meas = (1-2ε)P_true + ε
        // Invert: P_true = (P_meas - ε) / (1 - 2ε)
        double correction_factor = 1.0 / (1.0 - 2.0 * readout_error);
        *value *= correction_factor;
        // Clamp to valid range
        if (*value > 1.0) *value = 1.0;
        if (*value < -1.0) *value = -1.0;
        // Reduce confidence due to measurement uncertainty
        *confidence *= (1.0 - readout_error);
    }

    return true;
}

/**
 * @brief Measure Pauli X operator at lattice position (x,y) with confidence
 *
 * Computes the expectation value <X> for the qubit at position (x,y) on the
 * lattice, along with a confidence estimate based on the measurement statistics.
 *
 * The X operator flips qubits: X|0⟩ = |1⟩, X|1⟩ = |0⟩
 * The expectation value is computed as: <X> = 2 × Σᵢ<ⱼ Re(ψᵢ* × ψⱼ)
 * where j = i ⊕ (1 << qubit)
 *
 * Confidence is estimated from the variance of the measurement outcome.
 */
bool measure_pauli_x_with_confidence(const quantum_state_t* state,
                                    size_t x,
                                    size_t y,
                                    double* value,
                                    double* confidence) {
    if (!state || !value || !confidence) {
        return false;
    }

    if (!state->coordinates) {
        *value = 0.0;
        *confidence = 0.0;
        return false;
    }

    // Calculate qubit index from lattice coordinates
    size_t width = state->lattice_width > 0 ? state->lattice_width :
                   (size_t)sqrt((double)state->num_qubits);
    size_t qubit_idx = y * width + x;

    if (qubit_idx >= state->num_qubits) {
        // Out of bounds - return identity (no error detected)
        *value = 1.0;
        *confidence = 1.0;
        return true;
    }

    // Compute expectation value <X_q>
    // X|0⟩ = |1⟩, X|1⟩ = |0⟩
    // <X_q> = Σᵢ ψᵢ* × ψ_{i⊕mask} = 2 × Σᵢ<ⱼ Re(ψᵢ* × ψⱼ) where j = i ⊕ mask
    size_t dim = state->dimension;
    size_t mask = (size_t)1 << qubit_idx;

    double expectation = 0.0;
    double norm_sq = 0.0;

    for (size_t i = 0; i < dim; i++) {
        size_t j = i ^ mask;
        if (j > i) {
            // Compute ψᵢ* × ψⱼ contribution
            ComplexFloat ai = state->coordinates[i];
            ComplexFloat aj = state->coordinates[j];

            // Re(ψᵢ* × ψⱼ) = Re(ai) × Re(aj) + Im(ai) × Im(aj)
            expectation += 2.0 * (ai.real * aj.real + ai.imag * aj.imag);
        }
        // Also accumulate norm for validation
        norm_sq += state->coordinates[i].real * state->coordinates[i].real +
                   state->coordinates[i].imag * state->coordinates[i].imag;
    }

    // Normalize expectation value by state norm (should be ~1 for valid states)
    if (norm_sq > 1e-10) {
        *value = expectation / norm_sq;
    } else {
        *value = 0.0;  // Degenerate state
    }

    // Confidence is based on how close we are to an X eigenstate
    // |+⟩ = (|0⟩ + |1⟩)/√2 has <X> = +1
    // |-⟩ = (|0⟩ - |1⟩)/√2 has <X> = -1
    // Also factor in state normalization (unnormalized states reduce confidence)
    double norm_factor = fabs(norm_sq - 1.0) < 0.01 ? 1.0 : 0.9;
    double abs_value = fabs(*value);
    *confidence = abs_value * norm_factor;

    // Apply readout error correction if available
    // For X measurement, we effectively measure in the X basis
    double readout_error = get_readout_error_rate(x, y);
    double gate_error = get_gate_error_rate(x, y);  // Hadamard gate error for basis change
    double total_error = readout_error + gate_error;

    if (total_error > 0.0 && total_error < 0.5) {
        double correction_factor = 1.0 / (1.0 - 2.0 * total_error);
        *value *= correction_factor;
        // Clamp to valid range
        if (*value > 1.0) *value = 1.0;
        if (*value < -1.0) *value = -1.0;
        // Reduce confidence due to measurement uncertainty
        *confidence *= (1.0 - total_error);
    }

    return true;
}
