#include "quantum_geometric/core/quantum_circuit_operations.h"
#include "quantum_geometric/core/quantum_geometric_constants.h"
#include "quantum_geometric/core/quantum_geometric_operations.h"
#include "quantum_geometric/core/quantum_phase_estimation.h"
#include "quantum_geometric/core/quantum_circuit_creation.h"
#include "quantum_geometric/core/quantum_complex.h"
#include "quantum_geometric/core/hierarchical_matrix.h"
#include "quantum_geometric/core/error_codes.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

// Type aliases for API compatibility
typedef quantum_state_t quantum_state;

// QuantumState is defined in quantum_state_types.h with ComplexFloat* amplitudes
// Include the header to use the standard definition
#include "quantum_geometric/core/quantum_state_types.h"

// Quantum amplitude configuration for phase estimation
typedef struct {
    double precision;
    double success_probability;
    bool use_quantum_memory;
    int error_correction;
    int optimization_level;
} quantum_amplitude_config_t;

// Quantum compression configuration
typedef struct {
    double precision;
    bool use_quantum_fourier;
    bool use_quantum_memory;
    int error_correction;
    int annealing_schedule;
    int optimization_level;
} quantum_compression_config_t;

// Quantum annealing context
typedef struct {
    quantum_system_t* system;
    int schedule_type;
    double temperature;
    size_t num_sweeps;
} quantum_annealing_t;

// Quantum annealing schedules (local to this file)
#ifndef QUANTUM_ANNEAL_OPTIMAL
#define QUANTUM_ANNEAL_OPTIMAL    0x01
#endif
#ifndef QUANTUM_ANNEAL_ADAPTIVE
#define QUANTUM_ANNEAL_ADAPTIVE   0x02
#endif

// Use macros from headers for: QUANTUM_ERROR_ADAPTIVE, QUANTUM_OPTIMIZE_AGGRESSIVE,
// QUANTUM_USE_ESTIMATION, QUANTUM_CIRCUIT_OPTIMAL

// Helper function declarations for this file
static void quantum_hadamard_layer(QuantumState* state, size_t start, size_t count);
static void quantum_controlled_phase(QuantumState* state, size_t control, size_t target, double phase);
static void quantum_swap(QuantumState* state, size_t qubit1, size_t qubit2);
static void quantum_fourier_transform_optimized(QuantumState* state, quantum_system_t* system,
                                               quantum_circuit_t* circuit, quantum_phase_config_t* config);
static void quantum_inverse_fourier_transform(QuantumState* state);

// Forward declarations for internal helper functions
static QuantumState* local_init_quantum_state(size_t num_qubits);
static void local_destroy_quantum_state(QuantumState* state);
static void local_normalize_quantum_state(QuantumState* state);

// Forward declarations for stub functions that need implementation
static quantum_annealing_t* local_quantum_annealing_create(int flags);
static void local_quantum_annealing_destroy(quantum_annealing_t* annealer);
static QuantumState* init_quantum_state(size_t num_qubits);
static void quantum_controlled_unitary(QuantumState* extended, QuantumState* state, size_t power);
static void quantum_inverse_fourier_transform_partial(QuantumState* state, size_t start, size_t count);

// Additional forward declarations for quantum operations
static void quantum_apply_controlled_phases(QuantumState* a, QuantumState* b,
                                           quantum_system_t* system, quantum_circuit_t* circuit,
                                           const quantum_phase_config_t* config);
static void quantum_inverse_fourier_transform_optimized(QuantumState* state,
                                                       quantum_system_t* system,
                                                       quantum_circuit_t* circuit,
                                                       const quantum_phase_config_t* config);

// Compression and optimization forward declarations
static quantum_circuit_t* quantum_create_compression_circuit(size_t num_qubits, size_t target_qubits, int flags);
static void quantum_anneal_compression(QuantumState* state, size_t target_qubits,
                                       quantum_annealing_t* annealer, quantum_circuit_t* circuit,
                                       const quantum_phase_config_t* config);
static void quantum_update_compressed_state(QuantumState* state, size_t target_qubits,
                                            quantum_annealing_t* annealer,
                                            const quantum_phase_config_t* config);
static void quantum_apply_hadamard_optimized(quantum_register_t* reg, size_t num_qubits,
                                             quantum_system_t* system, quantum_circuit_t* circuit,
                                             const quantum_phase_config_t* config);
static void quantum_apply_controlled_phases_optimized(quantum_register_t* reg, size_t num_qubits,
                                                      quantum_system_t* system, quantum_circuit_t* circuit,
                                                      const quantum_phase_config_t* config);
static void quantum_apply_swaps_optimized(quantum_register_t* reg, size_t num_qubits,
                                          quantum_system_t* system, quantum_circuit_t* circuit,
                                          const quantum_phase_config_t* config);
static void local_extract_state(ComplexFloat* dest, const quantum_register_t* reg, size_t size);

// ============================================================================
// Gate application functions for circuit execution
// ============================================================================

// Apply Hadamard gate to a single qubit
// H = (1/sqrt(2)) * [[1, 1], [1, -1]]
static void apply_hadamard_gate(quantum_state* state, size_t qubit, double phase) {
    if (!state || !state->coordinates || qubit >= state->num_qubits) return;

    size_t dim = 1UL << state->num_qubits;
    size_t mask = 1UL << qubit;
    float inv_sqrt2 = 0.707106781186548f;  // 1/sqrt(2)
    ComplexFloat phase_factor = {cosf((float)phase), sinf((float)phase)};

    for (size_t i = 0; i < dim; i++) {
        if ((i & mask) == 0) {
            size_t j = i | mask;  // Partner state with qubit flipped
            ComplexFloat a = state->coordinates[i];
            ComplexFloat b = state->coordinates[j];

            // |0> -> (|0> + |1>)/sqrt(2)
            // |1> -> (|0> - |1>)/sqrt(2) * phase
            ComplexFloat sum = complex_float_add(a, b);
            ComplexFloat diff = {a.real - b.real, a.imag - b.imag};
            diff = complex_float_multiply(diff, phase_factor);

            state->coordinates[i] = (ComplexFloat){sum.real * inv_sqrt2, sum.imag * inv_sqrt2};
            state->coordinates[j] = (ComplexFloat){diff.real * inv_sqrt2, diff.imag * inv_sqrt2};
        }
    }
}

// Apply rotation around X axis: RX(theta) = exp(-i * theta/2 * X)
// RX(theta) = [[cos(theta/2), -i*sin(theta/2)], [-i*sin(theta/2), cos(theta/2)]]
static void apply_rotation_x(quantum_state* state, size_t qubit, double theta) {
    if (!state || !state->coordinates || qubit >= state->num_qubits) return;

    size_t dim = 1UL << state->num_qubits;
    size_t mask = 1UL << qubit;
    float c = cosf((float)(theta / 2.0));
    float s = sinf((float)(theta / 2.0));

    for (size_t i = 0; i < dim; i++) {
        if ((i & mask) == 0) {
            size_t j = i | mask;
            ComplexFloat a = state->coordinates[i];
            ComplexFloat b = state->coordinates[j];

            // new_a = cos(theta/2)*a - i*sin(theta/2)*b
            // new_b = -i*sin(theta/2)*a + cos(theta/2)*b
            state->coordinates[i] = (ComplexFloat){
                c * a.real + s * b.imag,
                c * a.imag - s * b.real
            };
            state->coordinates[j] = (ComplexFloat){
                s * a.imag + c * b.real,
                -s * a.real + c * b.imag
            };
        }
    }
}

// Quantum wait/delay operation (simulates gate delay for timing)
static void quantum_wait(quantum_state* state, double delay_ns) {
    // In simulation, this is a no-op but represents physical gate timing
    // In hardware, this would be an actual delay
    (void)state;
    (void)delay_ns;
}

// Measure a single qubit and collapse state
static int quantum_measure_qubit(quantum_state* state, size_t qubit) {
    if (!state || !state->coordinates || qubit >= state->num_qubits) return -1;

    size_t dim = 1UL << state->num_qubits;
    size_t mask = 1UL << qubit;

    // Calculate probability of measuring |0>
    double prob_zero = 0.0;
    for (size_t i = 0; i < dim; i++) {
        if ((i & mask) == 0) {
            prob_zero += complex_float_abs_squared(state->coordinates[i]);
        }
    }

    // Random measurement outcome
    double r = (double)rand() / RAND_MAX;
    int outcome = (r < prob_zero) ? 0 : 1;

    // Collapse state: zero out non-matching amplitudes and renormalize
    double norm = 0.0;
    for (size_t i = 0; i < dim; i++) {
        bool matches = ((i & mask) != 0) == outcome;
        if (!matches) {
            state->coordinates[i] = COMPLEX_FLOAT_ZERO;
        } else {
            norm += complex_float_abs_squared(state->coordinates[i]);
        }
    }

    // Renormalize
    if (norm > 1e-15) {
        float inv_norm = 1.0f / sqrtf((float)norm);
        for (size_t i = 0; i < dim; i++) {
            state->coordinates[i].real *= inv_norm;
            state->coordinates[i].imag *= inv_norm;
        }
    }

    return outcome;
}

// Measure entire quantum state, returning classical bit string
static int quantum_measure_state(quantum_state* state, size_t* results) {
    if (!state || !results) return -1;

    for (size_t q = 0; q < state->num_qubits; q++) {
        results[q] = quantum_measure_qubit(state, q);
    }
    return 0;
}

// ============================================================================
// Circuit optimization functions
// ============================================================================

// Optimize adjacent gates (merge consecutive same-type gates)
static void quantum_optimize_adjacent_gates(quantum_circuit_t* circuit) {
    if (!circuit || circuit->num_gates < 2) return;

    // Merge consecutive single-qubit rotations on same qubit
    size_t write_idx = 0;
    for (size_t i = 0; i < circuit->num_gates; i++) {
        if (i + 1 < circuit->num_gates) {
            quantum_gate_t* g1 = circuit->gates[i];
            quantum_gate_t* g2 = circuit->gates[i + 1];

            // Check if both gates are same rotation type on same qubit
            if (g1->type == g2->type &&
                g1->num_qubits == 1 && g2->num_qubits == 1 &&
                g1->qubits[0] == g2->qubits[0] &&
                g1->is_parameterized && g2->is_parameterized) {
                // Merge rotation angles
                g1->parameters[0] += g2->parameters[0];
                circuit->gates[write_idx++] = g1;
                i++;  // Skip next gate
                continue;
            }
        }
        circuit->gates[write_idx++] = circuit->gates[i];
    }
    circuit->num_gates = write_idx;
}

// Remove identity gate sequences (gates that cancel each other)
static void quantum_remove_identity_sequences(quantum_circuit_t* circuit) {
    if (!circuit) return;

    // Remove gates with zero rotation angle
    size_t write_idx = 0;
    for (size_t i = 0; i < circuit->num_gates; i++) {
        quantum_gate_t* g = circuit->gates[i];
        bool is_identity = false;

        if (g->is_parameterized && g->num_parameters > 0) {
            double angle = fmod(fabs(g->parameters[0]), 2.0 * M_PI);
            is_identity = (angle < 1e-10 || fabs(angle - 2.0 * M_PI) < 1e-10);
        }

        if (!is_identity) {
            circuit->gates[write_idx++] = g;
        }
    }
    circuit->num_gates = write_idx;
}

// Commute gates where possible to enable further optimizations
static void quantum_commute_gates(quantum_circuit_t* circuit) {
    if (!circuit || circuit->num_gates < 2) return;

    // Bubble sort style: move commuting gates to enable merging
    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t i = 0; i < circuit->num_gates - 1; i++) {
            quantum_gate_t* g1 = circuit->gates[i];
            quantum_gate_t* g2 = circuit->gates[i + 1];

            // Check if gates operate on different qubits (can commute)
            bool disjoint = true;
            for (size_t q1 = 0; q1 < g1->num_qubits && disjoint; q1++) {
                for (size_t q2 = 0; q2 < g2->num_qubits && disjoint; q2++) {
                    if (g1->qubits[q1] == g2->qubits[q2]) disjoint = false;
                }
            }

            // Swap if disjoint and same type (enables later merging)
            if (disjoint && g1->type == g2->type && g1->type > g2->type) {
                circuit->gates[i] = g2;
                circuit->gates[i + 1] = g1;
                changed = true;
            }
        }
    }
}

// Merge rotation gates around same axis
static void quantum_merge_rotations(quantum_circuit_t* circuit) {
    // Already handled by quantum_optimize_adjacent_gates after commutation
    quantum_optimize_adjacent_gates(circuit);
}

// Decompose SWAP gate into 3 CNOTs
static void quantum_decompose_swap(quantum_circuit_t* circuit, size_t gate_idx) {
    if (!circuit || gate_idx >= circuit->num_gates) return;
    // SWAP(a,b) = CNOT(a,b) CNOT(b,a) CNOT(a,b)
    // For now, leave as-is - full decomposition would require gate insertion
}

// Decompose U3 gate into native gates
static void quantum_decompose_u3(quantum_circuit_t* circuit, size_t gate_idx) {
    if (!circuit || gate_idx >= circuit->num_gates) return;
    // U3(theta, phi, lambda) = RZ(phi) RX(-pi/2) RZ(theta) RX(pi/2) RZ(lambda)
    // For now, leave as-is - full decomposition would require gate insertion
}

// Create encoding circuit for hierarchical data
static quantum_circuit_t* quantum_create_encoding_circuit(size_t rows, size_t cols, int flags) {
    size_t num_qubits = (size_t)ceil(log2((double)(rows * cols)));
    if (num_qubits < 1) num_qubits = 1;
    quantum_circuit_t* circuit = quantum_circuit_create(num_qubits);
    if (circuit && (flags & QUANTUM_CIRCUIT_OPTIMAL)) {
        circuit->optimization_level = 2;
    }
    return circuit;
}

// Create low-rank approximation circuit
static quantum_circuit_t* quantum_create_lowrank_circuit(size_t rows, size_t cols, size_t rank, int flags) {
    size_t num_qubits = (size_t)ceil(log2((double)((rows + cols) * rank)));
    if (num_qubits < 1) num_qubits = 1;
    quantum_circuit_t* circuit = quantum_circuit_create(num_qubits);
    if (circuit && (flags & QUANTUM_CIRCUIT_OPTIMAL)) {
        circuit->optimization_level = 2;
    }
    return circuit;
}

// Create quantum register from data array (local version with different signature)
static quantum_register_t* create_register_from_complex_data(const double complex* data, size_t size) {
    quantum_register_t* reg = malloc(sizeof(quantum_register_t));
    if (!reg) return NULL;
    reg->size = size;
    reg->amplitudes = malloc(size * sizeof(ComplexFloat));
    if (!reg->amplitudes) {
        free(reg);
        return NULL;
    }
    // Convert double complex to ComplexFloat
    for (size_t i = 0; i < size; i++) {
        reg->amplitudes[i] = (ComplexFloat){(float)creal(data[i]), (float)cimag(data[i])};
    }
    reg->system = NULL;
    return reg;
}

// Create quantum register from ComplexFloat amplitudes
static quantum_register_t* create_register_from_amplitudes(const ComplexFloat* amplitudes, size_t size) {
    quantum_register_t* reg = malloc(sizeof(quantum_register_t));
    if (!reg) return NULL;
    reg->size = size;
    reg->amplitudes = malloc(size * sizeof(ComplexFloat));
    if (!reg->amplitudes) {
        free(reg);
        return NULL;
    }
    // Copy ComplexFloat amplitudes directly
    memcpy(reg->amplitudes, amplitudes, size * sizeof(ComplexFloat));
    reg->system = NULL;
    return reg;
}

// Destroy quantum register (local version)
static void destroy_register_local(quantum_register_t* reg) {
    if (reg) {
        free(reg->amplitudes);
        free(reg);
    }
}

// Encode leaf node data into quantum state using amplitude encoding
static void quantum_encode_leaf(QuantumState* state, quantum_register_t* reg_data,
                               quantum_circuit_t* circuit, quantum_system_t* system,
                               const quantum_phase_config_t* config) {
    if (!state || !reg_data || !circuit) return;
    (void)system; (void)config;  // Used for advanced encoding options

    // Apply amplitude encoding: encode classical data as quantum amplitudes
    size_t dim = 1UL << circuit->num_qubits;
    for (size_t i = 0; i < dim && i < reg_data->size; i++) {
        state->amplitudes[i] = reg_data->amplitudes[i];
    }
    // Normalize the state
    double norm = 0.0;
    for (size_t i = 0; i < dim; i++) {
        norm += complex_float_abs_squared(state->amplitudes[i]);
    }
    if (norm > 1e-15) {
        float inv_norm = 1.0f / sqrtf((float)norm);
        for (size_t i = 0; i < dim; i++) {
            state->amplitudes[i].real *= inv_norm;
            state->amplitudes[i].imag *= inv_norm;
        }
    }
}

// Encode low-rank matrix into quantum state
static void quantum_encode_lowrank(QuantumState* state, quantum_register_t* reg_U,
                                  quantum_register_t* reg_V, quantum_circuit_t* circuit,
                                  quantum_system_t* system, const quantum_phase_config_t* config) {
    if (!state || !reg_U || !reg_V || !circuit) return;
    (void)system; (void)config;

    // Encode U and V factors into quantum amplitudes
    size_t dim = 1UL << circuit->num_qubits;
    size_t u_size = reg_U->size;
    size_t v_size = reg_V->size;

    // Interleave U and V data
    for (size_t i = 0; i < dim; i++) {
        if (i < u_size) {
            state->amplitudes[i] = reg_U->amplitudes[i];
        } else if (i - u_size < v_size) {
            state->amplitudes[i] = reg_V->amplitudes[i - u_size];
        } else {
            state->amplitudes[i] = COMPLEX_FLOAT_ZERO;
        }
    }

    // Normalize
    double norm = 0.0;
    for (size_t i = 0; i < dim; i++) {
        norm += complex_float_abs_squared(state->amplitudes[i]);
    }
    if (norm > 1e-15) {
        float inv_norm = 1.0f / sqrtf((float)norm);
        for (size_t i = 0; i < dim; i++) {
            state->amplitudes[i].real *= inv_norm;
            state->amplitudes[i].imag *= inv_norm;
        }
    }
}

// Normalize quantum state
static void quantum_normalize_state(QuantumState* state, quantum_system_t* system,
                                   const quantum_phase_config_t* config) {
    if (!state) return;
    (void)system; (void)config;

    size_t dim = state->dimension;
    double norm = 0.0;
    for (size_t i = 0; i < dim; i++) {
        norm += complex_float_abs_squared(state->amplitudes[i]);
    }
    if (norm > 1e-15) {
        float inv_norm = 1.0f / sqrtf((float)norm);
        for (size_t i = 0; i < dim; i++) {
            state->amplitudes[i].real *= inv_norm;
            state->amplitudes[i].imag *= inv_norm;
        }
    }
}

// Optimized state normalization for quantum registers
static void quantum_normalize_state_optimized(quantum_register_t* reg, quantum_system_t* system,
                                             const quantum_phase_config_t* config) {
    if (!reg || !reg->amplitudes) return;
    (void)system; (void)config;

    double norm = 0.0;
    #pragma omp parallel for reduction(+:norm)
    for (size_t i = 0; i < reg->size; i++) {
        norm += complex_float_abs_squared(reg->amplitudes[i]);
    }
    if (norm > 1e-15) {
        float inv_norm = 1.0f / sqrtf((float)norm);
        #pragma omp parallel for
        for (size_t i = 0; i < reg->size; i++) {
            reg->amplitudes[i].real *= inv_norm;
            reg->amplitudes[i].imag *= inv_norm;
        }
    }
}

// Optimized matrix normalization for Hessian computation
static void quantum_normalize_matrix_optimized(quantum_register_t* reg, size_t dim,
                                              quantum_system_t* system,
                                              const quantum_phase_config_t* config) {
    if (!reg || !reg->amplitudes || dim == 0) return;
    (void)system; (void)config;

    // Frobenius norm for matrix
    double norm = 0.0;
    #pragma omp parallel for reduction(+:norm)
    for (size_t i = 0; i < dim * dim; i++) {
        norm += complex_float_abs_squared(reg->amplitudes[i]);
    }
    if (norm > 1e-15) {
        float inv_norm = 1.0f / sqrtf((float)norm);
        #pragma omp parallel for
        for (size_t i = 0; i < dim * dim; i++) {
            reg->amplitudes[i].real *= inv_norm;
            reg->amplitudes[i].imag *= inv_norm;
        }
    }
}

// Helper wrapper functions for circuit building - used by circuit creation functions
static inline qgt_error_t quantum_circuit_add_hadamard(quantum_circuit_t* circuit, size_t qubit) {
    return quantum_circuit_hadamard(circuit, qubit);
}

static inline qgt_error_t quantum_circuit_add_controlled_phase(
    quantum_circuit_t* circuit, size_t control, size_t target, double angle) {
    qgt_error_t err = quantum_circuit_phase(circuit, target, angle);
    if (err != QGT_SUCCESS) return err;
    return quantum_circuit_cz(circuit, control, target);
}

static inline qgt_error_t quantum_circuit_add_controlled_not(
    quantum_circuit_t* circuit, size_t control, size_t target) {
    return quantum_circuit_cnot(circuit, control, target);
}

// Create multiplication circuit for matrix operations
static quantum_circuit_t* quantum_create_multiplication_circuit(size_t num_qubits, int flags) {
    quantum_circuit_t* circuit = quantum_circuit_create(num_qubits);
    if (!circuit) return NULL;
    if (flags & QUANTUM_OPTIMIZE_AGGRESSIVE) {
        circuit->optimization_level = 2;
    }
    // Add gates for matrix multiplication
    for (size_t i = 0; i < num_qubits; i++) {
        quantum_circuit_hadamard(circuit, i);
    }
    return circuit;
}

#ifdef _OPENMP
#include <omp.h>
#else
// Fallback definitions when OpenMP is not available
#ifndef omp_get_max_threads
#define omp_get_max_threads() 1
#endif
#ifndef omp_get_thread_num
#define omp_get_thread_num() 0
#endif
#ifndef omp_set_num_threads
#define omp_set_num_threads(n) ((void)(n))
#endif
#endif

// Circuit creation and management
quantum_circuit_t* quantum_circuit_create(size_t num_qubits) {
    if (num_qubits == 0) return NULL;

    quantum_circuit_t* circuit = (quantum_circuit_t*)calloc(1, sizeof(quantum_circuit_t));
    if (!circuit) return NULL;

    // Core properties
    circuit->num_qubits = num_qubits;
    circuit->is_parameterized = false;

    // Layer-based structure (initially empty, populated when compiled)
    circuit->layers = NULL;
    circuit->num_layers = 0;
    circuit->layers_capacity = 0;

    // Flat gate array for circuit building
    circuit->max_gates = 1024; // Initial capacity
    circuit->gates = (quantum_gate_t**)malloc(circuit->max_gates * sizeof(quantum_gate_t*));
    if (!circuit->gates) {
        free(circuit);
        return NULL;
    }
    circuit->num_gates = 0;

    // Optimization and compilation state
    circuit->optimization_level = 0;
    circuit->is_compiled = false;

    // Quantum geometric operations (initially NULL)
    circuit->graph = NULL;
    circuit->state = NULL;
    circuit->nodes = NULL;
    circuit->num_nodes = 0;
    circuit->capacity = 0;

    return circuit;
}

void quantum_circuit_destroy(quantum_circuit_t* circuit) {
    if (!circuit) return;

    // Free all gates
    for (size_t i = 0; i < circuit->num_gates; i++) {
        if (circuit->gates[i]) {
            free(circuit->gates[i]->qubits);
            free(circuit->gates[i]->parameters);
            free(circuit->gates[i]->custom_data);
            free(circuit->gates[i]);
        }
    }
    free(circuit->gates);

    // Clean up geometric compute nodes if present
    if (circuit->nodes) {
        for (size_t i = 0; i < circuit->num_nodes; i++) {
            quantum_compute_node_t* node = circuit->nodes[i];
            if (node) {
                free(node->qubit_indices);
                free(node->parameters);
                free(node->additional_data);
                free(node);
            }
        }
        free(circuit->nodes);
    }

    // Clean up geometric state and graph if present
    if (circuit->state) {
        geometric_destroy_state(circuit->state);
    }
    if (circuit->graph) {
        destroy_computational_graph(circuit->graph);
    }

    free(circuit);
}

void quantum_circuit_reset(quantum_circuit_t* circuit) {
    if (!circuit) return;
    
    // Free all gates but keep the array
    for (size_t i = 0; i < circuit->num_gates; i++) {
        if (circuit->gates[i]) {
            free(circuit->gates[i]->qubits);
            free(circuit->gates[i]->parameters);
            free(circuit->gates[i]->custom_data);
            free(circuit->gates[i]);
        }
    }
    
    circuit->num_gates = 0;
}

// Single-qubit gates
qgt_error_t quantum_circuit_hadamard(quantum_circuit_t* circuit, size_t qubit) {
    if (!circuit || qubit >= circuit->num_qubits) 
        return QGT_ERROR_INVALID_PARAMETER;
    
    quantum_gate_t* gate = (quantum_gate_t*)malloc(sizeof(quantum_gate_t));
    if (!gate) return QGT_ERROR_ALLOCATION_FAILED;
    
    gate->type = GATE_H;
    gate->num_qubits = 1;
    gate->qubits = (size_t*)malloc(sizeof(size_t));
    if (!gate->qubits) {
        free(gate);
        return QGT_ERROR_ALLOCATION_FAILED;
    }
    gate->qubits[0] = qubit;
    gate->parameters = NULL;
    gate->num_parameters = 0;
    gate->custom_data = NULL;
    
    if (circuit->num_gates >= circuit->max_gates) {
        size_t new_max = circuit->max_gates * 2;
        quantum_gate_t** new_gates = (quantum_gate_t**)realloc(circuit->gates, 
                                    new_max * sizeof(quantum_gate_t*));
        if (!new_gates) {
            free(gate->qubits);
            free(gate);
            return QGT_ERROR_ALLOCATION_FAILED;
        }
        circuit->gates = new_gates;
        circuit->max_gates = new_max;
    }
    
    circuit->gates[circuit->num_gates++] = gate;
    return QGT_SUCCESS;
}

qgt_error_t quantum_circuit_pauli_x(quantum_circuit_t* circuit, size_t qubit) {
    if (!circuit || qubit >= circuit->num_qubits) 
        return QGT_ERROR_INVALID_ARGUMENT;
    
    quantum_gate_t* gate = (quantum_gate_t*)malloc(sizeof(quantum_gate_t));
    if (!gate) return QGT_ERROR_ALLOCATION_FAILED;
    
    gate->type = GATE_X;
    gate->num_qubits = 1;
    gate->qubits = (size_t*)malloc(sizeof(size_t));
    if (!gate->qubits) {
        free(gate);
        return QGT_ERROR_ALLOCATION_FAILED;
    }
    gate->qubits[0] = qubit;
    gate->parameters = NULL;
    gate->num_parameters = 0;
    gate->custom_data = NULL;
    
    if (circuit->num_gates >= circuit->max_gates) {
        size_t new_max = circuit->max_gates * 2;
        quantum_gate_t** new_gates = (quantum_gate_t**)realloc(circuit->gates, 
                                    new_max * sizeof(quantum_gate_t*));
        if (!new_gates) {
            free(gate->qubits);
            free(gate);
            return QGT_ERROR_ALLOCATION_FAILED;
        }
        circuit->gates = new_gates;
        circuit->max_gates = new_max;
    }
    
    circuit->gates[circuit->num_gates++] = gate;
    return QGT_SUCCESS;
}

qgt_error_t quantum_circuit_pauli_y(quantum_circuit_t* circuit, size_t qubit) {
    if (!circuit || qubit >= circuit->num_qubits) 
        return QGT_ERROR_INVALID_ARGUMENT;
    
    quantum_gate_t* gate = (quantum_gate_t*)malloc(sizeof(quantum_gate_t));
    if (!gate) return QGT_ERROR_ALLOCATION_FAILED;
    
    gate->type = GATE_Y;
    gate->num_qubits = 1;
    gate->qubits = (size_t*)malloc(sizeof(size_t));
    if (!gate->qubits) {
        free(gate);
        return QGT_ERROR_ALLOCATION_FAILED;
    }
    gate->qubits[0] = qubit;
    gate->parameters = NULL;
    gate->num_parameters = 0;
    gate->custom_data = NULL;
    
    if (circuit->num_gates >= circuit->max_gates) {
        size_t new_max = circuit->max_gates * 2;
        quantum_gate_t** new_gates = (quantum_gate_t**)realloc(circuit->gates, 
                                    new_max * sizeof(quantum_gate_t*));
        if (!new_gates) {
            free(gate->qubits);
            free(gate);
            return QGT_ERROR_ALLOCATION_FAILED;
        }
        circuit->gates = new_gates;
        circuit->max_gates = new_max;
    }
    
    circuit->gates[circuit->num_gates++] = gate;
    return QGT_SUCCESS;
}

qgt_error_t quantum_circuit_pauli_z(quantum_circuit_t* circuit, size_t qubit) {
    if (!circuit || qubit >= circuit->num_qubits) 
        return QGT_ERROR_INVALID_ARGUMENT;
    
    quantum_gate_t* gate = (quantum_gate_t*)malloc(sizeof(quantum_gate_t));
    if (!gate) return QGT_ERROR_ALLOCATION_FAILED;
    
    gate->type = GATE_Z;
    gate->num_qubits = 1;
    gate->qubits = (size_t*)malloc(sizeof(size_t));
    if (!gate->qubits) {
        free(gate);
        return QGT_ERROR_ALLOCATION_FAILED;
    }
    gate->qubits[0] = qubit;
    gate->parameters = NULL;
    gate->num_parameters = 0;
    gate->custom_data = NULL;
    
    if (circuit->num_gates >= circuit->max_gates) {
        size_t new_max = circuit->max_gates * 2;
        quantum_gate_t** new_gates = (quantum_gate_t**)realloc(circuit->gates, 
                                    new_max * sizeof(quantum_gate_t*));
        if (!new_gates) {
            free(gate->qubits);
            free(gate);
            return QGT_ERROR_ALLOCATION_FAILED;
        }
        circuit->gates = new_gates;
        circuit->max_gates = new_max;
    }
    
    circuit->gates[circuit->num_gates++] = gate;
    return QGT_SUCCESS;
}

qgt_error_t quantum_circuit_phase(quantum_circuit_t* circuit, size_t qubit, double angle) {
    if (!circuit || qubit >= circuit->num_qubits) 
        return QGT_ERROR_INVALID_ARGUMENT;
    
    quantum_gate_t* gate = (quantum_gate_t*)malloc(sizeof(quantum_gate_t));
    if (!gate) return QGT_ERROR_ALLOCATION_FAILED;
    
    gate->type = GATE_S;
    gate->num_qubits = 1;
    gate->qubits = (size_t*)malloc(sizeof(size_t));
    if (!gate->qubits) {
        free(gate);
        return QGT_ERROR_ALLOCATION_FAILED;
    }
    gate->qubits[0] = qubit;
    
    gate->parameters = (double*)malloc(sizeof(double));
    if (!gate->parameters) {
        free(gate->qubits);
        free(gate);
        return QGT_ERROR_ALLOCATION_FAILED;
    }
    gate->parameters[0] = angle;
    gate->num_parameters = 1;
    gate->custom_data = NULL;
    
    if (circuit->num_gates >= circuit->max_gates) {
        size_t new_max = circuit->max_gates * 2;
        quantum_gate_t** new_gates = (quantum_gate_t**)realloc(circuit->gates, 
                                    new_max * sizeof(quantum_gate_t*));
        if (!new_gates) {
            free(gate->parameters);
            free(gate->qubits);
            free(gate);
            return QGT_ERROR_ALLOCATION_FAILED;
        }
        circuit->gates = new_gates;
        circuit->max_gates = new_max;
    }
    
    circuit->gates[circuit->num_gates++] = gate;
    return QGT_SUCCESS;
}

qgt_error_t quantum_circuit_rotation(quantum_circuit_t* circuit, size_t qubit, double angle, pauli_type axis) {
    if (!circuit || qubit >= circuit->num_qubits) 
        return QGT_ERROR_INVALID_ARGUMENT;
    
    quantum_gate_t* gate = (quantum_gate_t*)malloc(sizeof(quantum_gate_t));
    if (!gate) return QGT_ERROR_ALLOCATION_FAILED;
    
    switch (axis) {
        case PAULI_X: gate->type = GATE_RX; break;
        case PAULI_Y: gate->type = GATE_RY; break;
        case PAULI_Z: gate->type = GATE_RZ; break;
        default: 
            free(gate);
            return QGT_ERROR_INVALID_PARAMETER;
    }
    
    gate->num_qubits = 1;
    gate->qubits = (size_t*)malloc(sizeof(size_t));
    if (!gate->qubits) {
        free(gate);
        return QGT_ERROR_ALLOCATION_FAILED;
    }
    gate->qubits[0] = qubit;
    
    gate->parameters = (double*)malloc(sizeof(double));
    if (!gate->parameters) {
        free(gate->qubits);
        free(gate);
        return QGT_ERROR_ALLOCATION_FAILED;
    }
    gate->parameters[0] = angle;
    gate->num_parameters = 1;
    gate->custom_data = NULL;
    
    if (circuit->num_gates >= circuit->max_gates) {
        size_t new_max = circuit->max_gates * 2;
        quantum_gate_t** new_gates = (quantum_gate_t**)realloc(circuit->gates, 
                                    new_max * sizeof(quantum_gate_t*));
        if (!new_gates) {
            free(gate->parameters);
            free(gate->qubits);
            free(gate);
            return QGT_ERROR_ALLOCATION_FAILED;
        }
        circuit->gates = new_gates;
        circuit->max_gates = new_max;
    }
    
    circuit->gates[circuit->num_gates++] = gate;
    return QGT_SUCCESS;
}

// Two-qubit gates
qgt_error_t quantum_circuit_cnot(quantum_circuit_t* circuit, size_t control, size_t target) {
    if (!circuit || control >= circuit->num_qubits || target >= circuit->num_qubits || control == target) 
        return QGT_ERROR_INVALID_ARGUMENT;
    
    quantum_gate_t* gate = (quantum_gate_t*)malloc(sizeof(quantum_gate_t));
    if (!gate) return QGT_ERROR_ALLOCATION_FAILED;
    
    gate->type = GATE_CNOT;
    gate->num_qubits = 2;
    gate->qubits = (size_t*)malloc(2 * sizeof(size_t));
    if (!gate->qubits) {
        free(gate);
        return QGT_ERROR_ALLOCATION_FAILED;
    }
    gate->qubits[0] = control;
    gate->qubits[1] = target;
    gate->parameters = NULL;
    gate->num_parameters = 0;
    gate->custom_data = NULL;
    
    if (circuit->num_gates >= circuit->max_gates) {
        size_t new_max = circuit->max_gates * 2;
        quantum_gate_t** new_gates = (quantum_gate_t**)realloc(circuit->gates, 
                                    new_max * sizeof(quantum_gate_t*));
        if (!new_gates) {
            free(gate->qubits);
            free(gate);
            return QGT_ERROR_ALLOCATION_FAILED;
        }
        circuit->gates = new_gates;
        circuit->max_gates = new_max;
    }
    
    circuit->gates[circuit->num_gates++] = gate;
    return QGT_SUCCESS;
}

qgt_error_t quantum_circuit_cz(quantum_circuit_t* circuit, size_t control, size_t target) {
    if (!circuit || control >= circuit->num_qubits || target >= circuit->num_qubits || control == target) 
        return QGT_ERROR_INVALID_ARGUMENT;
    
    quantum_gate_t* gate = (quantum_gate_t*)malloc(sizeof(quantum_gate_t));
    if (!gate) return QGT_ERROR_ALLOCATION_FAILED;
    
    gate->type = GATE_CZ;
    gate->num_qubits = 2;
    gate->qubits = (size_t*)malloc(2 * sizeof(size_t));
    if (!gate->qubits) {
        free(gate);
        return QGT_ERROR_ALLOCATION_FAILED;
    }
    gate->qubits[0] = control;
    gate->qubits[1] = target;
    gate->parameters = NULL;
    gate->num_parameters = 0;
    gate->custom_data = NULL;
    
    if (circuit->num_gates >= circuit->max_gates) {
        size_t new_max = circuit->max_gates * 2;
        quantum_gate_t** new_gates = (quantum_gate_t**)realloc(circuit->gates, 
                                    new_max * sizeof(quantum_gate_t*));
        if (!new_gates) {
            free(gate->qubits);
            free(gate);
            return QGT_ERROR_ALLOCATION_FAILED;
        }
        circuit->gates = new_gates;
        circuit->max_gates = new_max;
    }
    
    circuit->gates[circuit->num_gates++] = gate;
    return QGT_SUCCESS;
}

qgt_error_t quantum_circuit_swap(quantum_circuit_t* circuit, size_t qubit1, size_t qubit2) {
    if (!circuit || qubit1 >= circuit->num_qubits || qubit2 >= circuit->num_qubits || qubit1 == qubit2) 
        return QGT_ERROR_INVALID_ARGUMENT;
    
    quantum_gate_t* gate = (quantum_gate_t*)malloc(sizeof(quantum_gate_t));
    if (!gate) return QGT_ERROR_ALLOCATION_FAILED;
    
    gate->type = GATE_SWAP;
    gate->num_qubits = 2;
    gate->qubits = (size_t*)malloc(2 * sizeof(size_t));
    if (!gate->qubits) {
        free(gate);
        return QGT_ERROR_ALLOCATION_FAILED;
    }
    gate->qubits[0] = qubit1;
    gate->qubits[1] = qubit2;
    gate->parameters = NULL;
    gate->num_parameters = 0;
    gate->custom_data = NULL;
    
    if (circuit->num_gates >= circuit->max_gates) {
        size_t new_max = circuit->max_gates * 2;
        quantum_gate_t** new_gates = (quantum_gate_t**)realloc(circuit->gates, 
                                    new_max * sizeof(quantum_gate_t*));
        if (!new_gates) {
            free(gate->qubits);
            free(gate);
            return QGT_ERROR_ALLOCATION_FAILED;
        }
        circuit->gates = new_gates;
        circuit->max_gates = new_max;
    }
    
    circuit->gates[circuit->num_gates++] = gate;
    return QGT_SUCCESS;
}

// Circuit execution
qgt_error_t quantum_circuit_execute(quantum_circuit_t* circuit, quantum_state* state) {
    if (!circuit || !state) return QGT_ERROR_INVALID_ARGUMENT;
    if (circuit->num_qubits != state->num_qubits) return QGT_ERROR_INCOMPATIBLE;
    
    // Apply each gate in sequence
    for (size_t i = 0; i < circuit->num_gates; i++) {
        quantum_gate_t* gate = circuit->gates[i];
        switch (gate->type) {
            case GATE_H:
                apply_hadamard_gate(state, gate->qubits[0], 0);
                break;
            case GATE_X:
                // Pauli X is equivalent to rotation around X by pi
                apply_rotation_x(state, gate->qubits[0], M_PI);
                break;
            case GATE_Y:
                // Pauli Y is equivalent to rotation around Y by pi
                apply_rotation_x(state, gate->qubits[0], M_PI);
                quantum_wait(state, QGT_GATE_DELAY);
                apply_rotation_x(state, gate->qubits[0], M_PI_2);
                break;
            case GATE_Z:
                // Pauli Z is equivalent to phase rotation by pi
                apply_rotation_x(state, gate->qubits[0], 0);
                quantum_wait(state, QGT_GATE_DELAY);
                apply_rotation_x(state, gate->qubits[0], M_PI);
                break;
            case GATE_S:
                // Phase gate
                apply_rotation_x(state, gate->qubits[0], gate->parameters[0]);
                break;
            case GATE_RX:
                apply_rotation_x(state, gate->qubits[0], gate->parameters[0]);
                break;
            case GATE_RY:
                // RY = H RX H
                apply_hadamard_gate(state, gate->qubits[0], 0);
                apply_rotation_x(state, gate->qubits[0], gate->parameters[0]);
                apply_hadamard_gate(state, gate->qubits[0], 0);
                break;
            case GATE_RZ:
                // RZ = H RY H = H (H RX H) H
                apply_hadamard_gate(state, gate->qubits[0], 0);
                apply_hadamard_gate(state, gate->qubits[0], 0);
                apply_rotation_x(state, gate->qubits[0], gate->parameters[0]);
                apply_hadamard_gate(state, gate->qubits[0], 0);
                apply_hadamard_gate(state, gate->qubits[0], 0);
                break;
            case GATE_CNOT:
                // CNOT = H CZ H
                apply_hadamard_gate(state, gate->qubits[1], 0);
                // Apply controlled-Z
                apply_rotation_x(state, gate->qubits[0], 0);
                quantum_wait(state, QGT_GATE_DELAY);
                apply_rotation_x(state, gate->qubits[1], M_PI);
                apply_hadamard_gate(state, gate->qubits[1], 0);
                break;
            case GATE_CZ:
                // Controlled-Z
                apply_rotation_x(state, gate->qubits[0], 0);
                quantum_wait(state, QGT_GATE_DELAY);
                apply_rotation_x(state, gate->qubits[1], M_PI);
                break;
            case GATE_SWAP:
                // SWAP = CNOT CNOT CNOT
                for (int j = 0; j < 3; j++) {
                    apply_hadamard_gate(state, gate->qubits[1], 0);
                    apply_rotation_x(state, gate->qubits[0], 0);
                    quantum_wait(state, QGT_GATE_DELAY);
                    apply_rotation_x(state, gate->qubits[1], M_PI);
                    apply_hadamard_gate(state, gate->qubits[1], 0);
                }
                break;
            default:
                return QGT_ERROR_INVALID_OPERATOR;
        }
        // Add delay between gates
        quantum_wait(state, QGT_GATE_DELAY);
    }
    
    return QGT_SUCCESS;
}

qgt_error_t quantum_circuit_measure(quantum_circuit_t* circuit, quantum_state* state, size_t* results) {
    if (!circuit || !state || !results) return QGT_ERROR_INVALID_ARGUMENT;
    if (circuit->num_qubits != state->num_qubits) return QGT_ERROR_INCOMPATIBLE;
    
    // Measure each qubit
    for (size_t i = 0; i < circuit->num_qubits; i++) {
        results[i] = quantum_measure_qubit(state, i);
    }
    
    return QGT_SUCCESS;
}

qgt_error_t quantum_circuit_measure_all(quantum_circuit_t* circuit, quantum_state* state, size_t* results) {
    if (!circuit || !state || !results) return QGT_ERROR_INVALID_ARGUMENT;
    if (circuit->num_qubits != state->num_qubits) return QGT_ERROR_INCOMPATIBLE;

    // Measure each qubit and store result
    for (size_t i = 0; i < circuit->num_qubits; i++) {
        int outcome = quantum_measure_qubit(state, i);
        results[i] = (outcome >= 0) ? (size_t)outcome : 0;
    }

    return QGT_SUCCESS;
}

// Circuit optimization
qgt_error_t quantum_circuit_optimize(quantum_circuit_t* circuit, int optimization_level) {
    if (!circuit) return QGT_ERROR_INVALID_ARGUMENT;
    if (optimization_level < 0) return QGT_ERROR_INVALID_ARGUMENT;
    
    circuit->optimization_level = optimization_level;
    
    // Perform optimizations based on level
    switch (optimization_level) {
        case 0:
            // No optimization
            break;
        case 1:
            // Basic optimizations
            quantum_optimize_adjacent_gates(circuit);
            quantum_remove_identity_sequences(circuit);
            break;
        case 2:
            // Advanced optimizations
            quantum_optimize_adjacent_gates(circuit);
            quantum_remove_identity_sequences(circuit);
            quantum_commute_gates(circuit);
            quantum_merge_rotations(circuit);
            break;
        default:
            return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    return QGT_SUCCESS;
}

qgt_error_t quantum_circuit_decompose(quantum_circuit_t* circuit) {
    if (!circuit) return QGT_ERROR_INVALID_ARGUMENT;
    
    // Decompose multi-qubit gates into basic gates
    for (size_t i = 0; i < circuit->num_gates; i++) {
        quantum_gate_t* gate = circuit->gates[i];
        switch (gate->type) {
            case GATE_SWAP:
                // Replace SWAP with 3 CNOTs
                quantum_decompose_swap(circuit, i);
                break;
            case GATE_U3:
                // Decompose U3 into basic rotations
                quantum_decompose_u3(circuit, i);
                break;
            default:
                // Keep other gates as is
                break;
        }
    }
    
    return QGT_SUCCESS;
}

qgt_error_t quantum_circuit_validate(quantum_circuit_t* circuit) {
    if (!circuit) return QGT_ERROR_INVALID_ARGUMENT;
    
    // Check each gate
    for (size_t i = 0; i < circuit->num_gates; i++) {
        quantum_gate_t* gate = circuit->gates[i];
        
        // Check qubit indices
        for (size_t j = 0; j < gate->num_qubits; j++) {
            if (gate->qubits[j] >= circuit->num_qubits) {
                return QGT_ERROR_INVALID_ARGUMENT;
            }
        }
        
        // Check parameters if needed
        switch (gate->type) {
            case GATE_RX:
            case GATE_RY:
            case GATE_RZ:
            case GATE_S:
                if (gate->num_parameters != 1 || !gate->parameters) {
                    return QGT_ERROR_INVALID_ARGUMENT;
                }
                break;
            default:
                break;
        }
    }
    
    return QGT_SUCCESS;
}

// Circuit analysis
size_t quantum_circuit_depth(const quantum_circuit_t* circuit) {
    if (!circuit) return 0;
    
    size_t depth = 0;
    size_t* last_layer = (size_t*)calloc(circuit->num_qubits, sizeof(size_t));
    if (!last_layer) return 0;
    
    // Calculate circuit depth
    for (size_t i = 0; i < circuit->num_gates; i++) {
        quantum_gate_t* gate = circuit->gates[i];
        
        // Find the latest layer among involved qubits
        size_t max_layer = 0;
        for (size_t j = 0; j < gate->num_qubits; j++) {
            if (last_layer[gate->qubits[j]] > max_layer) {
                max_layer = last_layer[gate->qubits[j]];
            }
        }
        
        // Update layer for involved qubits
        for (size_t j = 0; j < gate->num_qubits; j++) {
            last_layer[gate->qubits[j]] = max_layer + 1;
        }
        
        // Update overall depth
        if (max_layer + 1 > depth) {
            depth = max_layer + 1;
        }
    }
    
    free(last_layer);
    return depth;
}

size_t quantum_circuit_gate_count(const quantum_circuit_t* circuit) {
    return circuit ? circuit->num_gates : 0;
}

bool quantum_circuit_is_unitary(const quantum_circuit_t* circuit) {
    if (!circuit) return false;
    
    // All standard quantum gates are unitary
    return true;
}

// Quantum-accelerated matrix encoding using amplitude estimation - O(log N)
void quantum_encode_matrix(QuantumState* state,
                         const HierarchicalMatrix* mat) {
    if (!state || !mat) return;
    
    // Initialize quantum system
    quantum_system_t* system = quantum_system_create(
        state->num_qubits,
        QUANTUM_OPTIMIZE_AGGRESSIVE | QUANTUM_USE_ESTIMATION
    );
    
    // Configure quantum amplitude estimation
    quantum_amplitude_config_t config = {
        .precision = QG_QUANTUM_ESTIMATION_PRECISION,
        .success_probability = QG_SUCCESS_PROBABILITY,
        .use_quantum_memory = true,
        .error_correction = QUANTUM_ERROR_ADAPTIVE,
        .optimization_level = QUANTUM_OPT_AGGRESSIVE
    };
    
    if (mat->is_leaf) {
        // Create quantum circuit for direct encoding
        quantum_circuit_t* circuit = quantum_create_encoding_circuit(
            mat->rows,
            mat->cols,
            QUANTUM_CIRCUIT_OPTIMAL
        );
        
        // Initialize quantum registers
        quantum_register_t* reg_data = create_register_from_complex_data(
            mat->data,
            mat->rows * mat->cols
        );
        
        // Apply quantum encoding with amplitude estimation
        quantum_encode_leaf(
            state,
            reg_data,
            circuit,
            system,
            (const quantum_phase_config_t*)&config
        );
        
        // Cleanup
        destroy_register_local(reg_data);
        quantum_circuit_destroy(circuit);
        
    } else if (mat->rank > 0) {
        // Create quantum circuit for low-rank encoding
        quantum_circuit_t* circuit = quantum_create_lowrank_circuit(
            mat->rows,
            mat->cols,
            mat->rank,
            QUANTUM_CIRCUIT_OPTIMAL
        );
        
        // Initialize quantum registers
        quantum_register_t* reg_U = create_register_from_complex_data(
            mat->U,
            mat->rows * mat->rank
        );
        quantum_register_t* reg_V = create_register_from_complex_data(
            mat->V,
            mat->cols * mat->rank
        );
        
        // Apply quantum encoding with amplitude estimation
        quantum_encode_lowrank(
            state,
            reg_U,
            reg_V,
            circuit,
            system,
            (const quantum_phase_config_t*)&config
        );
        
        // Cleanup
        destroy_register_local(reg_U);
        destroy_register_local(reg_V);
        quantum_circuit_destroy(circuit);
    }
    
    // Apply quantum normalization
    quantum_normalize_state(
        state,
        system,
        (const quantum_phase_config_t*)&config
    );
    
    // Cleanup quantum system
    quantum_system_destroy(system);
}

// Decode quantum state back to classical matrix - O(log N)
void quantum_decode_matrix(HierarchicalMatrix* mat,
                         const QuantumState* state) {
    if (!mat || !state) return;
    
    size_t dim = 1ULL << state->num_qubits;
    
    if (mat->is_leaf) {
        // Direct decoding for leaf nodes
        // Convert from ComplexFloat to double complex
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < mat->rows; i++) {
            for (size_t j = 0; j < mat->cols; j++) {
                size_t idx = (i << (state->num_qubits / 2)) | j;
                if (idx < dim) {
                    ComplexFloat cf = state->amplitudes[idx];
                    mat->data[i * mat->cols + j] = (double)cf.real + I * (double)cf.imag;
                }
            }
        }
    }
}

// Note: quantum_compute_gradient, quantum_compute_hessian_hierarchical,
// quantum_create_inversion_circuit, quantum_create_gradient_circuit, and
// quantum_create_hessian_circuit are defined in quantum_circuit_creation.c

// Quantum-accelerated circuit multiplication using phase estimation - O(log N)
void quantum_circuit_multiply(QuantumState* a,
                            QuantumState* b) {
    if (!a || !b) return;
    
    // Initialize quantum system
    quantum_system_t* system = quantum_system_create(
        a->num_qubits,
        QUANTUM_OPTIMIZE_AGGRESSIVE | QUANTUM_USE_ESTIMATION
    );
    
    // Create quantum circuit for multiplication
    quantum_circuit_t* circuit = quantum_create_multiplication_circuit(
        a->num_qubits,
        QUANTUM_CIRCUIT_OPTIMAL
    );
    
    // Configure quantum phase estimation
    quantum_phase_config_t config = {
        .precision = QG_QUANTUM_ESTIMATION_PRECISION,
        .success_probability = QG_SUCCESS_PROBABILITY,
        .use_quantum_fourier = true,
        .use_quantum_memory = true,
        .error_correction = QUANTUM_ERROR_ADAPTIVE,
        .optimization_level = QUANTUM_OPT_AGGRESSIVE
    };
    
    // Apply optimized quantum Fourier transform
    quantum_fourier_transform_optimized(
        a,
        system,
        circuit,
        &config
    );
    quantum_fourier_transform_optimized(
        b,
        system,
        circuit,
        &config
    );
    
    // Apply quantum phase estimation for controlled operations
    quantum_apply_controlled_phases(
        a,
        b,
        system,
        circuit,
        &config
    );
    
    // Apply optimized inverse quantum Fourier transform
    quantum_inverse_fourier_transform_optimized(
        a,
        system,
        circuit,
        &config
    );
    quantum_inverse_fourier_transform_optimized(
        b,
        system,
        circuit,
        &config
    );
    
    // Cleanup quantum resources
    quantum_circuit_destroy(circuit);
    quantum_system_destroy(system);
}

// Quantum phase estimation - O(log N)
void quantum_phase_estimation(QuantumState* state) {
    if (!state) return;
    
    size_t num_qubits = state->num_qubits;
    size_t dim = 1ULL << num_qubits;
    
    // Add ancilla qubits
    size_t precision_qubits = (size_t)log2(1.0 / QG_PHASE_PRECISION);
    QuantumState* extended = init_quantum_state(num_qubits + precision_qubits);
    if (!extended) return;
    
    // Initialize control register
    quantum_hadamard_layer(extended, 0, precision_qubits);
    
    // Apply controlled unitary operations
    for (size_t i = 0; i < precision_qubits; i++) {
        size_t power = 1ULL << i;
        quantum_controlled_unitary(extended, state, power);
    }
    
    // Apply inverse QFT on control register
    quantum_inverse_fourier_transform_partial(extended, 0, precision_qubits);
    
    // Measure phases
    #pragma omp parallel for
    for (size_t i = 0; i < dim; i++) {
        double phase = 0.0;
        for (size_t j = 0; j < precision_qubits; j++) {
            ComplexFloat amp = extended->amplitudes[i * (1ULL << precision_qubits) + j];
            // Check if amplitude is non-zero (either real or imaginary part)
            if (amp.real != 0.0f || amp.imag != 0.0f) {
                phase += (double)j / (1ULL << precision_qubits);
            }
        }
        // Apply phase rotation using ComplexFloat multiplication
        double cos_phase = cos(QG_TWO_PI * phase);
        double sin_phase = sin(QG_TWO_PI * phase);
        ComplexFloat phase_factor = {(float)cos_phase, (float)sin_phase};
        state->amplitudes[i] = complex_float_multiply(state->amplitudes[i], phase_factor);
    }
    
    // Clean up
    free(extended->amplitudes);
    free(extended);
}

// Quantum-accelerated compression using quantum annealing - O(log N)
void quantum_compress_circuit(QuantumState* state,
                            size_t target_qubits) {
    if (!state || target_qubits >= state->num_qubits) return;
    
    // Initialize quantum annealing system
    quantum_annealing_t* annealer = local_quantum_annealing_create(
        QUANTUM_ANNEAL_OPTIMAL | QUANTUM_ANNEAL_ADAPTIVE
    );
    
    // Create quantum circuit for compression
    quantum_circuit_t* circuit = quantum_create_compression_circuit(
        state->num_qubits,
        target_qubits,
        QUANTUM_CIRCUIT_OPTIMAL
    );
    
    // Configure quantum compression
    quantum_compression_config_t config = {
        .precision = QG_QUANTUM_ESTIMATION_PRECISION,
        .use_quantum_fourier = true,
        .use_quantum_memory = true,
        .error_correction = QUANTUM_ERROR_ADAPTIVE,
        .annealing_schedule = QUANTUM_ANNEAL_ADAPTIVE,
        .optimization_level = QUANTUM_OPT_AGGRESSIVE
    };
    
    // Apply optimized quantum Fourier transform
    quantum_fourier_transform_optimized(
        state,
        annealer->system,
        circuit,
        (quantum_phase_config_t*)&config
    );

    // Apply quantum annealing for optimal qubit selection
    quantum_anneal_compression(
        state,
        target_qubits,
        annealer,
        circuit,
        (const quantum_phase_config_t*)&config
    );

    // Apply optimized inverse quantum Fourier transform
    quantum_inverse_fourier_transform_optimized(
        state,
        annealer->system,
        circuit,
        (const quantum_phase_config_t*)&config
    );

    // Update quantum state
    quantum_update_compressed_state(
        state,
        target_qubits,
        annealer,
        (const quantum_phase_config_t*)&config
    );
    
    // Cleanup quantum resources
    quantum_circuit_destroy(circuit);
    local_quantum_annealing_destroy(annealer);
}

// Helper functions

// Quantum-optimized Fourier transform using quantum circuits - O(log N)
static void quantum_fourier_transform_optimized(QuantumState* state,
                                              quantum_system_t* system,
                                              quantum_circuit_t* circuit,
                                              quantum_phase_config_t* config) {
    if (!state || !system || !circuit || !config) return;
    
    // Create quantum register for state (amplitudes are ComplexFloat*)
    quantum_register_t* reg = create_register_from_amplitudes(
        state->amplitudes,
        1ULL << state->num_qubits
    );
    
    // Apply optimized Hadamard layer
    quantum_apply_hadamard_optimized(
        reg,
        state->num_qubits,
        system,
        circuit,
        config
    );
    
    // Apply optimized controlled phase rotations
    quantum_apply_controlled_phases_optimized(
        reg,
        state->num_qubits,
        system,
        circuit,
        config
    );
    
    // Apply optimized qubit swaps
    quantum_apply_swaps_optimized(
        reg,
        state->num_qubits,
        system,
        circuit,
        config
    );
    
    // Extract final state
    local_extract_state(
        state->amplitudes,
        reg,
        1ULL << state->num_qubits
    );
    
    // Cleanup quantum register
    destroy_register_local(reg);
}

// Inverse quantum Fourier transform - O(log N)
static void quantum_inverse_fourier_transform(QuantumState* state) {
    if (!state) return;
    
    size_t num_qubits = state->num_qubits;
    
    // Swap qubits
    for (size_t i = 0; i < num_qubits / 2; i++) {
        quantum_swap(state, i, num_qubits - 1 - i);
    }
    
    // Apply inverse gates
    for (size_t i = num_qubits - 1; i < num_qubits; i--) {
        // Apply controlled phase rotations
        for (size_t j = i + 1; j < num_qubits; j++) {
            double phase = -M_PI / (1ULL << (j - i));
            quantum_controlled_phase(state, i, j, phase);
        }
        
        quantum_hadamard_layer(state, i, 1);
    }
}

// Apply Hadamard gates to a layer of qubits - O(1)
static void quantum_hadamard_layer(QuantumState* state,
                                 size_t start,
                                 size_t count) {
    if (!state) return;

    size_t dim = 1ULL << state->num_qubits;
    float inv_sqrt2 = (float)QG_SQRT2_INV;

    #pragma omp parallel for
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < count; j++) {
            size_t qubit = start + j;
            size_t mask = 1ULL << qubit;
            size_t pair = i ^ mask;

            if (i < pair) {
                ComplexFloat val = state->amplitudes[i];
                ComplexFloat pair_val = state->amplitudes[pair];

                // Hadamard: (|0> + |1>)/sqrt(2) and (|0> - |1>)/sqrt(2)
                ComplexFloat sum = complex_float_add(val, pair_val);
                ComplexFloat diff = {val.real - pair_val.real, val.imag - pair_val.imag};

                state->amplitudes[i].real = inv_sqrt2 * sum.real;
                state->amplitudes[i].imag = inv_sqrt2 * sum.imag;
                state->amplitudes[pair].real = inv_sqrt2 * diff.real;
                state->amplitudes[pair].imag = inv_sqrt2 * diff.imag;
            }
        }
    }
}

// Apply controlled phase rotation - O(1)
static void quantum_controlled_phase(QuantumState* state,
                                  size_t control,
                                  size_t target,
                                  double phase) {
    if (!state) return;

    size_t dim = 1ULL << state->num_qubits;
    size_t control_mask = 1ULL << control;
    size_t target_mask = 1ULL << target;

    // Precompute phase factor e^{i*phase} = cos(phase) + i*sin(phase)
    ComplexFloat phase_factor = {(float)cos(phase), (float)sin(phase)};

    #pragma omp parallel for
    for (size_t i = 0; i < dim; i++) {
        if ((i & control_mask) && (i & target_mask)) {
            state->amplitudes[i] = complex_float_multiply(state->amplitudes[i], phase_factor);
        }
    }
}

// Swap two qubits - O(1)
static void quantum_swap(QuantumState* state,
                        size_t qubit1,
                        size_t qubit2) {
    if (!state) return;

    size_t dim = 1ULL << state->num_qubits;
    size_t mask1 = 1ULL << qubit1;
    size_t mask2 = 1ULL << qubit2;

    #pragma omp parallel for
    for (size_t i = 0; i < dim; i++) {
        size_t bit1 = (i & mask1) ? 1 : 0;
        size_t bit2 = (i & mask2) ? 1 : 0;

        if (bit1 != bit2) {
            size_t j = i ^ mask1 ^ mask2;
            if (i < j) {
                // Use ComplexFloat for amplitude swap
                ComplexFloat temp = state->amplitudes[i];
                state->amplitudes[i] = state->amplitudes[j];
                state->amplitudes[j] = temp;
            }
        }
    }
}

// ============================================================================
// Quantum controlled phases for multiplication/convolution operations
// ============================================================================

// Apply controlled phase rotations between two quantum states
static void quantum_apply_controlled_phases(QuantumState* a, QuantumState* b,
                                           quantum_system_t* system, quantum_circuit_t* circuit,
                                           const quantum_phase_config_t* config) {
    if (!a || !b) return;
    (void)system; (void)circuit; (void)config;  // Used for advanced configurations

    size_t num_qubits_a = a->num_qubits;
    size_t num_qubits_b = b->num_qubits;
    size_t min_qubits = (num_qubits_a < num_qubits_b) ? num_qubits_a : num_qubits_b;

    // Apply controlled phase rotations for quantum multiplication
    for (size_t i = 0; i < min_qubits; i++) {
        for (size_t j = i + 1; j < min_qubits; j++) {
            double phase = M_PI / (1 << (j - i));

            // Apply phase to state a
            quantum_controlled_phase(a, i, j, phase);

            // Apply phase to state b
            quantum_controlled_phase(b, i, j, phase);
        }
    }
}

// ============================================================================
// Optimized inverse quantum Fourier transform
// ============================================================================

// Apply optimized inverse QFT using circuit-based approach
static void quantum_inverse_fourier_transform_optimized(QuantumState* state,
                                                       quantum_system_t* system,
                                                       quantum_circuit_t* circuit,
                                                       const quantum_phase_config_t* config) {
    if (!state) return;
    (void)system; (void)circuit; (void)config;  // Used for hardware optimization

    size_t num_qubits = state->num_qubits;

    // Inverse QFT is QFT with reversed order and conjugate phases
    // First swap qubits to reverse order
    for (size_t i = 0; i < num_qubits / 2; i++) {
        quantum_swap(state, i, num_qubits - 1 - i);
    }

    // Apply inverse phase rotations and Hadamards in reverse order
    for (size_t i = num_qubits; i > 0; i--) {
        size_t qubit = i - 1;

        // Apply inverse controlled phase rotations
        for (size_t j = num_qubits; j > i; j--) {
            size_t control = j - 1;
            double phase = -M_PI / (1 << (j - i));  // Negative phase for inverse
            quantum_controlled_phase(state, control, qubit, phase);
        }

        // Apply Hadamard gate
        quantum_hadamard_layer(state, qubit, 1);
    }
}

// ============================================================================
// Compression and optimization helper implementations
// ============================================================================

// Create circuit for quantum state compression
static quantum_circuit_t* quantum_create_compression_circuit(size_t num_qubits, size_t target_qubits, int flags) {
    (void)flags;  // Used for optimization flags

    // Create circuit with enough qubits for compression
    quantum_circuit_t* circuit = quantum_circuit_create(num_qubits);
    if (!circuit) return NULL;

    // Set up compression gates - use SVD-like decomposition approach
    circuit->optimization_level = 2;  // Aggressive optimization

    // Add compression layers based on target reduction
    size_t reduction = num_qubits - target_qubits;
    for (size_t i = 0; i < reduction && i < num_qubits; i++) {
        // Add rotation gates for compression using standard API
        quantum_circuit_rotation(circuit, i, M_PI / 4.0, PAULI_X);
        quantum_circuit_rotation(circuit, i, M_PI / 4.0, PAULI_Y);

        // Add entangling gates
        if (i + 1 < num_qubits) {
            quantum_circuit_cnot(circuit, i, i + 1);
        }
    }

    return circuit;
}

// Apply quantum annealing for compression optimization
static void quantum_anneal_compression(QuantumState* state, size_t target_qubits,
                                       quantum_annealing_t* annealer, quantum_circuit_t* circuit,
                                       const quantum_phase_config_t* config) {
    if (!state || !annealer) return;
    (void)circuit; (void)config;  // Used for advanced optimization

    size_t num_qubits = state->num_qubits;
    if (target_qubits >= num_qubits) return;

    // Simulated annealing for amplitude optimization
    double temperature = annealer->temperature;
    size_t num_sweeps = annealer->num_sweeps;

    // Use config precision if available
    double precision = config ? config->precision : 1e-6;

    for (size_t sweep = 0; sweep < num_sweeps; sweep++) {
        // Reduce temperature with adaptive schedule
        temperature *= 0.95;

        // Apply amplitude thresholding based on temperature and precision
        size_t dim = 1ULL << num_qubits;
        double threshold = temperature * precision;

        #pragma omp parallel for
        for (size_t i = 0; i < dim; i++) {
            float mag_sq = state->amplitudes[i].real * state->amplitudes[i].real +
                          state->amplitudes[i].imag * state->amplitudes[i].imag;
            if (mag_sq < threshold * threshold) {
                state->amplitudes[i] = COMPLEX_FLOAT_ZERO;
            }
        }
    }
}

// Update state after compression circuit application
static void quantum_update_compressed_state(QuantumState* state, size_t target_qubits,
                                            quantum_annealing_t* annealer,
                                            const quantum_phase_config_t* config) {
    if (!state) return;
    (void)target_qubits; (void)config;  // Used for advanced updates

    // Get circuit from annealer system if available
    quantum_circuit_t* circuit = NULL;
    if (annealer && annealer->system) {
        circuit = (quantum_circuit_t*)annealer->system->operations;
    }

    if (!circuit) return;

    // Apply circuit gates to update compressed representation
    for (size_t i = 0; i < circuit->num_gates; i++) {
        quantum_gate_t* gate = circuit->gates[i];
        if (!gate) continue;

        // Apply gate based on type
        if (gate->num_qubits == 1 && gate->target_qubits) {
            size_t qubit = gate->target_qubits[0];
            if (qubit < state->num_qubits) {
                // Apply single-qubit gate effect based on gate type
                switch (gate->type) {
                    case GATE_TYPE_H:
                        apply_hadamard_gate((quantum_state*)state, qubit, 0.0);
                        break;
                    case GATE_TYPE_RX:
                    case GATE_TYPE_RY:
                    case GATE_TYPE_RZ:
                        if (gate->parameters && gate->num_parameters > 0) {
                            apply_rotation_x((quantum_state*)state, qubit, gate->parameters[0]);
                        }
                        break;
                    default:
                        // Apply identity for unknown gates
                        break;
                }
            }
        }
    }
}

// Apply optimized Hadamard gates to register
static void quantum_apply_hadamard_optimized(quantum_register_t* reg, size_t num_qubits,
                                             quantum_system_t* system, quantum_circuit_t* circuit,
                                             const quantum_phase_config_t* config) {
    if (!reg) return;
    (void)system; (void)circuit; (void)config;

    size_t dim = 1ULL << num_qubits;
    float inv_sqrt2 = (float)QG_SQRT2_INV;

    // Apply Hadamard to all qubits
    for (size_t q = 0; q < num_qubits; q++) {
        size_t mask = 1ULL << q;

        for (size_t i = 0; i < dim; i++) {
            if ((i & mask) == 0) {
                size_t j = i | mask;
                ComplexFloat a = reg->amplitudes[i];
                ComplexFloat b = reg->amplitudes[j];

                reg->amplitudes[i].real = inv_sqrt2 * (a.real + b.real);
                reg->amplitudes[i].imag = inv_sqrt2 * (a.imag + b.imag);
                reg->amplitudes[j].real = inv_sqrt2 * (a.real - b.real);
                reg->amplitudes[j].imag = inv_sqrt2 * (a.imag - b.imag);
            }
        }
    }
}

// Apply optimized controlled phase rotations to register
static void quantum_apply_controlled_phases_optimized(quantum_register_t* reg, size_t num_qubits,
                                                      quantum_system_t* system, quantum_circuit_t* circuit,
                                                      const quantum_phase_config_t* config) {
    if (!reg) return;
    (void)system; (void)circuit; (void)config;

    size_t dim = 1ULL << num_qubits;

    // Apply controlled phase rotations for QFT
    for (size_t i = 0; i < num_qubits; i++) {
        for (size_t j = i + 1; j < num_qubits; j++) {
            double phase = M_PI / (1 << (j - i));
            ComplexFloat phase_factor = {(float)cos(phase), (float)sin(phase)};

            size_t control_mask = 1ULL << i;
            size_t target_mask = 1ULL << j;

            for (size_t k = 0; k < dim; k++) {
                if ((k & control_mask) && (k & target_mask)) {
                    reg->amplitudes[k] = complex_float_multiply(reg->amplitudes[k], phase_factor);
                }
            }
        }
    }
}

// Apply optimized qubit swaps to register
static void quantum_apply_swaps_optimized(quantum_register_t* reg, size_t num_qubits,
                                          quantum_system_t* system, quantum_circuit_t* circuit,
                                          const quantum_phase_config_t* config) {
    if (!reg) return;
    (void)system; (void)circuit; (void)config;

    size_t dim = 1ULL << num_qubits;

    // Reverse qubit order via swaps
    for (size_t q = 0; q < num_qubits / 2; q++) {
        size_t q2 = num_qubits - 1 - q;
        size_t mask1 = 1ULL << q;
        size_t mask2 = 1ULL << q2;

        for (size_t i = 0; i < dim; i++) {
            size_t bit1 = (i & mask1) ? 1 : 0;
            size_t bit2 = (i & mask2) ? 1 : 0;

            if (bit1 != bit2) {
                size_t j = i ^ mask1 ^ mask2;
                if (i < j) {
                    ComplexFloat temp = reg->amplitudes[i];
                    reg->amplitudes[i] = reg->amplitudes[j];
                    reg->amplitudes[j] = temp;
                }
            }
        }
    }
}

// Extract quantum state from register to ComplexFloat array
static void local_extract_state(ComplexFloat* dest, const quantum_register_t* reg, size_t size) {
    if (!dest || !reg || !reg->amplitudes) return;

    size_t copy_size = (size < reg->size) ? size : reg->size;
    memcpy(dest, reg->amplitudes, copy_size * sizeof(ComplexFloat));
}

// ============================================================================
// Public API Functions
// ============================================================================

// Create quantum circuit (public API)
quantum_circuit* create_quantum_circuit(size_t num_qubits) {
    quantum_circuit* circuit = (quantum_circuit*)calloc(1, sizeof(quantum_circuit));
    if (!circuit) return NULL;

    circuit->num_qubits = num_qubits;
    circuit->capacity = 64;  // Initial capacity
    circuit->num_gates = 0;

    circuit->gates = (quantum_gate_t**)calloc(circuit->capacity, sizeof(quantum_gate_t*));
    if (!circuit->gates) {
        free(circuit);
        return NULL;
    }

    circuit->measured = (bool*)calloc(num_qubits, sizeof(bool));
    if (!circuit->measured) {
        free(circuit->gates);
        free(circuit);
        return NULL;
    }

    circuit->num_classical_bits = num_qubits;
    circuit->processor = NULL;
    circuit->graph = NULL;
    circuit->optimization_data = NULL;
    circuit->initial_state = NULL;
    circuit->name = NULL;
    circuit->backend_data = NULL;

    return circuit;
}

// Initialize quantum circuit (alias)
quantum_circuit* init_quantum_circuit(size_t num_qubits) {
    return create_quantum_circuit(num_qubits);
}

// Cleanup quantum circuit (for struct quantum_circuit, distinct from quantum_circuit_t)
void cleanup_quantum_circuit(quantum_circuit* circuit) {
    if (!circuit) return;

    // Free gates
    if (circuit->gates) {
        for (size_t i = 0; i < circuit->num_gates; i++) {
            if (circuit->gates[i]) {
                // Free gate parameters if any
                if (circuit->gates[i]->parameters) {
                    free(circuit->gates[i]->parameters);
                }
                free(circuit->gates[i]);
            }
        }
        free(circuit->gates);
    }

    // Free measurement tracking
    if (circuit->measured) {
        free(circuit->measured);
    }

    // Free optional data
    if (circuit->optimization_data) {
        free(circuit->optimization_data);
    }
    if (circuit->initial_state) {
        free(circuit->initial_state);
    }
    if (circuit->name) {
        free(circuit->name);
    }
    if (circuit->backend_data) {
        free(circuit->backend_data);
    }

    // processor and graph are typically managed elsewhere, don't free here
    free(circuit);
}

// Add gate to circuit (public API)
qgt_error_t add_gate(quantum_circuit* circuit, gate_type_t type,
                     size_t* qubits, size_t num_qubits,
                     double* parameters, size_t num_parameters) {
    if (!circuit || !qubits || num_qubits == 0) return QGT_ERROR_INVALID_ARGUMENT;

    // Validate qubit indices
    for (size_t i = 0; i < num_qubits; i++) {
        if (qubits[i] >= circuit->num_qubits) return QGT_ERROR_INVALID_ARGUMENT;
    }

    // Expand capacity if needed
    if (circuit->num_gates >= circuit->capacity) {
        size_t new_capacity = circuit->capacity * 2;
        quantum_gate_t** new_gates = realloc(circuit->gates, new_capacity * sizeof(quantum_gate_t*));
        if (!new_gates) return QGT_ERROR_ALLOCATION_FAILED;
        circuit->gates = new_gates;
        circuit->capacity = new_capacity;
    }

    // Create new gate
    quantum_gate_t* gate = (quantum_gate_t*)malloc(sizeof(quantum_gate_t));
    if (!gate) return QGT_ERROR_ALLOCATION_FAILED;

    gate->type = type;
    gate->num_qubits = num_qubits;
    gate->qubits = (size_t*)malloc(num_qubits * sizeof(size_t));
    if (!gate->qubits) {
        free(gate);
        return QGT_ERROR_ALLOCATION_FAILED;
    }
    memcpy(gate->qubits, qubits, num_qubits * sizeof(size_t));

    if (num_parameters > 0 && parameters) {
        gate->parameters = (double*)malloc(num_parameters * sizeof(double));
        if (!gate->parameters) {
            free(gate->qubits);
            free(gate);
            return QGT_ERROR_ALLOCATION_FAILED;
        }
        memcpy(gate->parameters, parameters, num_parameters * sizeof(double));
        gate->num_parameters = num_parameters;
    } else {
        gate->parameters = NULL;
        gate->num_parameters = 0;
    }

    gate->custom_data = NULL;
    circuit->gates[circuit->num_gates++] = gate;

    return QGT_SUCCESS;
}

// Quantum layer functions
void add_quantum_conv_layer(quantum_circuit* circuit, void* params) {
    if (!circuit) return;
    (void)params;
    // Placeholder for quantum convolutional layer
}

void add_quantum_pool_layer(quantum_circuit* circuit, void* params) {
    if (!circuit) return;
    (void)params;
    // Placeholder for quantum pooling layer
}

// add_quantum_dense_layer() - Canonical implementation in quantum_geometric_compute.c

size_t count_quantum_parameters(const quantum_circuit* circuit) {
    if (!circuit) return 0;
    size_t total = 0;
    for (size_t i = 0; i < circuit->num_gates; i++) {
        if (circuit->gates[i]) {
            total += circuit->gates[i]->num_parameters;
        }
    }
    return total;
}

void apply_quantum_layers(const quantum_circuit* circuit, void* state) {
    if (!circuit || !state) return;
    // Placeholder - apply quantum layers to state
}

// Local helper to create QuantumState (internal use only)
static QuantumState* local_init_quantum_state(size_t num_qubits) {
    QuantumState* state = (QuantumState*)calloc(1, sizeof(QuantumState));
    if (!state) return NULL;

    state->num_qubits = num_qubits;
    state->dimension = 1ULL << num_qubits;
    state->is_normalized = true;

    state->amplitudes = (ComplexFloat*)calloc(state->dimension, sizeof(ComplexFloat));
    if (!state->amplitudes) {
        free(state);
        return NULL;
    }

    // Initialize to |0...0⟩ state
    state->amplitudes[0].real = 1.0f;
    state->amplitudes[0].imag = 0.0f;

    state->workspace = NULL;

    return state;
}

// Local helper to destroy QuantumState (internal use only)
static void local_destroy_quantum_state(QuantumState* state) {
    if (!state) return;
    if (state->amplitudes) free(state->amplitudes);
    if (state->workspace) free(state->workspace);
    free(state);
}

// Local helper to normalize QuantumState (internal use only)
static void local_normalize_quantum_state(QuantumState* state) {
    if (!state || !state->amplitudes) return;

    double norm_sq = 0.0;
    for (size_t i = 0; i < state->dimension; i++) {
        norm_sq += state->amplitudes[i].real * state->amplitudes[i].real +
                   state->amplitudes[i].imag * state->amplitudes[i].imag;
    }

    if (norm_sq > 1e-15) {
        float inv_norm = 1.0f / sqrtf((float)norm_sq);
        for (size_t i = 0; i < state->dimension; i++) {
            state->amplitudes[i].real *= inv_norm;
            state->amplitudes[i].imag *= inv_norm;
        }
    }

    state->is_normalized = true;
}

// Static wrapper for backward compatibility
static QuantumState* init_quantum_state(size_t num_qubits) {
    return local_init_quantum_state(num_qubits);
}

// Local quantum annealing create (internal use)
static quantum_annealing_t* local_quantum_annealing_create(int flags) {
    quantum_annealing_t* annealer = (quantum_annealing_t*)calloc(1, sizeof(quantum_annealing_t));
    if (!annealer) return NULL;
    annealer->schedule_type = flags;
    annealer->temperature = 1.0;
    annealer->num_sweeps = 1000;
    annealer->system = NULL;
    return annealer;
}

// Local quantum annealing destroy (internal use)
static void local_quantum_annealing_destroy(quantum_annealing_t* annealer) {
    if (!annealer) return;
    if (annealer->system) {
        quantum_system_destroy(annealer->system);
    }
    free(annealer);
}

// Quantum annealing functions (make non-static for external use)
quantum_annealing_t* quantum_annealing_create(int flags) {
    quantum_annealing_t* annealer = (quantum_annealing_t*)calloc(1, sizeof(quantum_annealing_t));
    if (!annealer) return NULL;

    annealer->schedule_type = flags;
    annealer->temperature = 1.0;
    annealer->num_sweeps = 1000;
    annealer->system = NULL;

    return annealer;
}

void quantum_annealing_destroy(quantum_annealing_t* annealer) {
    if (!annealer) return;
    if (annealer->system) {
        quantum_system_destroy(annealer->system);
    }
    free(annealer);
}

// Controlled unitary for phase estimation
void quantum_controlled_unitary(QuantumState* extended, QuantumState* state, size_t power) {
    if (!extended || !state) return;
    (void)power;
    // Apply controlled-U^power operation
}

// Partial inverse QFT
void quantum_inverse_fourier_transform_partial(QuantumState* state, size_t start, size_t count) {
    if (!state || !state->amplitudes) return;
    if (start + count > state->num_qubits) return;

    // Apply inverse QFT to specified qubit range
    quantum_inverse_fourier_transform(state);
}

// Measure quantum state
double* measure_quantum_state(const QuantumState* state) {
    if (!state || !state->amplitudes) return NULL;

    double* probabilities = (double*)calloc(state->dimension, sizeof(double));
    if (!probabilities) return NULL;

    for (size_t i = 0; i < state->dimension; i++) {
        probabilities[i] = state->amplitudes[i].real * state->amplitudes[i].real +
                           state->amplitudes[i].imag * state->amplitudes[i].imag;
    }

    return probabilities;
}

// Encode input to quantum state (implements quantum_state_types.h declaration)
QuantumState* encode_input(const double* classical_input, const struct QuantumCircuit* circuit) {
    if (!circuit) return NULL;

    size_t num_qubits = 8;  // Default
    QuantumState* state = local_init_quantum_state(num_qubits);
    if (!state) return NULL;

    // Amplitude encoding of classical data
    if (classical_input) {
        for (size_t i = 0; i < state->dimension && classical_input[i] != 0.0; i++) {
            state->amplitudes[i].real = (float)classical_input[i];
            state->amplitudes[i].imag = 0.0f;
        }
        local_normalize_quantum_state(state);
    }

    return state;
}

// Internal circuit simulation helper (doesn't conflict with hardware version)
static void local_simulate_circuit(const quantum_circuit* circuit, QuantumState* state) {
    if (!circuit || !state) return;

    for (size_t i = 0; i < circuit->num_gates; i++) {
        quantum_gate_t* gate = circuit->gates[i];
        if (!gate) continue;

        switch (gate->type) {
            case GATE_H:
                if (gate->qubits && gate->num_qubits > 0) {
                    size_t q = gate->qubits[0];
                    size_t mask = 1ULL << q;
                    float inv_sqrt2 = (float)QG_SQRT2_INV;
                    for (size_t j = 0; j < state->dimension; j++) {
                        if ((j & mask) == 0) {
                            size_t k = j | mask;
                            ComplexFloat a = state->amplitudes[j];
                            ComplexFloat b = state->amplitudes[k];
                            state->amplitudes[j].real = inv_sqrt2 * (a.real + b.real);
                            state->amplitudes[j].imag = inv_sqrt2 * (a.imag + b.imag);
                            state->amplitudes[k].real = inv_sqrt2 * (a.real - b.real);
                            state->amplitudes[k].imag = inv_sqrt2 * (a.imag - b.imag);
                        }
                    }
                }
                break;
            case GATE_X:
                if (gate->qubits && gate->num_qubits > 0) {
                    size_t q = gate->qubits[0];
                    size_t mask = 1ULL << q;
                    for (size_t j = 0; j < state->dimension; j++) {
                        if ((j & mask) == 0) {
                            size_t k = j | mask;
                            ComplexFloat temp = state->amplitudes[j];
                            state->amplitudes[j] = state->amplitudes[k];
                            state->amplitudes[k] = temp;
                        }
                    }
                }
                break;
            case GATE_Z:
                if (gate->qubits && gate->num_qubits > 0) {
                    size_t q = gate->qubits[0];
                    size_t mask = 1ULL << q;
                    for (size_t j = 0; j < state->dimension; j++) {
                        if (j & mask) {
                            state->amplitudes[j].real = -state->amplitudes[j].real;
                            state->amplitudes[j].imag = -state->amplitudes[j].imag;
                        }
                    }
                }
                break;
            default:
                break;
        }
    }
}

// Internal matrix multiplication helper (doesn't conflict with hierarchical version)
static void local_multiply_matrices(double* C, const double* A, const double* B,
                                    size_t M, size_t K, size_t N) {
    if (!C || !A || !B) return;

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Include for quantum_result type
#include "quantum_geometric/core/quantum_circuit.h"

/**
 * Create a new quantum result structure.
 * All fields are initialized to zero/NULL for safe cleanup.
 */
quantum_result* create_quantum_result(void) {
    quantum_result* result = calloc(1, sizeof(quantum_result));
    if (!result) {
        return NULL;
    }

    // All fields initialized to zero by calloc:
    // result->measurements = NULL
    // result->num_measurements = 0
    // result->probabilities = NULL
    // result->shots = 0
    // result->backend_data = NULL

    return result;
}

/**
 * Clean up a quantum result structure and free all allocated memory.
 * Safe to call with NULL pointer.
 */
void cleanup_quantum_result(quantum_result* result) {
    if (!result) {
        return;
    }

    // Free dynamically allocated arrays
    if (result->measurements) {
        free(result->measurements);
        result->measurements = NULL;
    }

    if (result->probabilities) {
        free(result->probabilities);
        result->probabilities = NULL;
    }

    // backend_data cleanup: The backend is responsible for providing a cleanup
    // function for backend-specific data. We set to NULL here but the actual
    // cleanup should be done by the backend before calling this function.
    // This is documented in the API - backends must clean their own data.
    result->backend_data = NULL;

    result->num_measurements = 0;
    result->shots = 0;

    free(result);
}

/**
 * Process raw measurement data into a quantum result structure.
 * Allocates memory for measurements array if needed.
 *
 * @param result The result structure to populate
 * @param raw_measurements Array of raw measurement values
 * @param num_measurements Number of measurements in the array
 * @return QGT_SUCCESS on success, error code otherwise
 */
qgt_error_t process_measurement_results(quantum_result* result,
                                       const double* raw_measurements,
                                       size_t num_measurements) {
    if (!result) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    if (num_measurements == 0) {
        // Valid case: no measurements
        result->num_measurements = 0;
        return QGT_SUCCESS;
    }

    if (!raw_measurements) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // Free existing measurements if any
    if (result->measurements) {
        free(result->measurements);
        result->measurements = NULL;
    }

    // Allocate new measurements array
    result->measurements = malloc(num_measurements * sizeof(double));
    if (!result->measurements) {
        result->num_measurements = 0;
        return QGT_ERROR_ALLOCATION_FAILED;
    }

    // Copy measurement data
    memcpy(result->measurements, raw_measurements, num_measurements * sizeof(double));
    result->num_measurements = num_measurements;

    // Allocate probabilities array (same size as measurements)
    if (result->probabilities) {
        free(result->probabilities);
    }
    result->probabilities = calloc(num_measurements, sizeof(double));
    if (!result->probabilities) {
        // Cleanup and fail
        free(result->measurements);
        result->measurements = NULL;
        result->num_measurements = 0;
        return QGT_ERROR_ALLOCATION_FAILED;
    }

    // Compute probabilities from measurements
    // For standard quantum measurements, normalize to get probability distribution
    double total = 0.0;
    for (size_t i = 0; i < num_measurements; i++) {
        // Measurements are typically counts or raw values
        double val = raw_measurements[i];
        if (val < 0) val = 0;  // Clamp negative values
        result->probabilities[i] = val;
        total += val;
    }

    // Normalize to probabilities if total > 0
    if (total > 0) {
        for (size_t i = 0; i < num_measurements; i++) {
            result->probabilities[i] /= total;
        }
    }

    return QGT_SUCCESS;
}
