/**
 * @file quantum_simulator_cpu.c
 * @brief CPU-optimized quantum simulator implementation
 *
 * This file provides CPU-only semi-classical simulation using:
 * - Tensor network operations from tensor_network_operations.h
 * - Hierarchical matrices from hierarchical_matrix.h
 * - Error syndrome handling from error_syndrome.h
 */

#include "quantum_geometric/hardware/quantum_simulator.h"
#include "quantum_geometric/hardware/quantum_hardware_abstraction.h"
#include "quantum_geometric/core/hierarchical_matrix.h"
#include "quantum_geometric/core/tensor_network_operations.h"
#include "quantum_geometric/core/quantum_types.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/error_codes.h"
#include "quantum_geometric/core/quantum_geometric_operations.h"
#include "quantum_geometric/core/quantum_rng.h"
#include "quantum_geometric/physics/error_syndrome.h"
#include <complex.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

// Global QRNG context
static qrng_ctx* g_qrng_ctx = NULL;

// Initialize QRNG if needed
static void init_qrng_if_needed(void) {
    if (!g_qrng_ctx) {
        qrng_init(&g_qrng_ctx, NULL, 0);
    }
}

// ============================================================================
// CPU Simulator-specific Gate Types
// ============================================================================

// Gate types for CPU simulator (distinct from HAL gate types)
typedef enum {
    CPU_GATE_SINGLE = 0,      // Single-qubit gate
    CPU_GATE_TWO = 1,         // Two-qubit gate
    CPU_GATE_MEASURE = 2,     // Measurement operation
    CPU_GATE_ERROR_DETECT = 3,  // Error detection
    CPU_GATE_ERROR_CORRECT = 4  // Error correction
} CPUGateType;

// Gate structure for CPU simulator
typedef struct {
    CPUGateType type;          // Gate type
    size_t target;             // Target qubit
    size_t control;            // Control qubit (for two-qubit gates)
    double complex* matrix;    // Gate matrix (2x2 or 4x4)
    double error_threshold;    // Error threshold for correction gates
} CPUGate;

// Circuit structure for CPU simulator
typedef struct {
    CPUGate* gates;            // Array of gates
    size_t n_gates;            // Number of gates
    size_t max_gates;          // Maximum gates capacity
    bool use_error_correction; // Enable error correction
    bool use_tensor_networks;  // Enable tensor network optimization
    size_t cache_line_size;    // Cache line size for optimization
} CPUCircuit;

// ============================================================================
// Forward Declarations
// ============================================================================

static void apply_single_qubit_gate_cpu(double complex* state,
                                       const double complex* gate_matrix,
                                       size_t target_qubit,
                                       size_t n_qubits);

static void apply_two_qubit_gate_cpu(double complex* state,
                                    const double complex* gate_matrix,
                                    size_t control_qubit,
                                    size_t target_qubit,
                                    size_t n_qubits);

static void apply_two_qubit_gate_basic(double complex* state,
                                      const double complex* gate_matrix,
                                      size_t control_qubit,
                                      size_t target_qubit,
                                      size_t n_qubits);

static int measure_qubit_basic(double complex* state,
                              size_t target_qubit,
                              size_t n_qubits);

// ============================================================================
// State Initialization
// ============================================================================

// Initialize simulator state to |0⟩
void init_simulator_state(double complex* state, size_t n) {
    memset(state, 0, n * sizeof(double complex));
    state[0] = 1.0 + 0.0*I;
}

// ============================================================================
// Single Qubit Gate Implementation
// ============================================================================

static void apply_single_qubit_gate_cpu(double complex* state,
                                       const double complex* gate_matrix,
                                       size_t target_qubit,
                                       size_t n_qubits) {
    size_t n = 1ULL << n_qubits;
    size_t mask = 1ULL << target_qubit;
    size_t block_size = 64 / sizeof(double complex); // Cache line size

    #pragma omp parallel
    {
        // Thread-local buffer for cache efficiency
        double complex local_buffer[64];

        #pragma omp for schedule(static)
        for (size_t block = 0; block < n; block += block_size) {
            size_t block_end = block + block_size < n ? block + block_size : n;

            // Load block into local buffer
            for (size_t i = block; i < block_end; i++) {
                if ((i & mask) == 0) {
                    size_t i1 = i;
                    size_t i2 = i | mask;
                    local_buffer[i - block] = state[i1];
                    local_buffer[i - block + 1] = state[i2];
                }
            }

            // Process block with SIMD
            #pragma omp simd
            for (size_t i = block; i < block_end; i++) {
                if ((i & mask) == 0) {
                    size_t i1 = i;
                    size_t i2 = i | mask;
                    size_t buf_idx = i - block;

                    double complex v1 = local_buffer[buf_idx];
                    double complex v2 = local_buffer[buf_idx + 1];

                    state[i1] = gate_matrix[0] * v1 + gate_matrix[1] * v2;
                    state[i2] = gate_matrix[2] * v1 + gate_matrix[3] * v2;
                }
            }
        }
    }
}

// ============================================================================
// Two Qubit Gate Implementation with Tensor Networks
// ============================================================================

static void apply_two_qubit_gate_cpu(double complex* state,
                                    const double complex* gate_matrix,
                                    size_t control_qubit,
                                    size_t target_qubit,
                                    size_t n_qubits) {
    size_t n = 1ULL << n_qubits;
    size_t control_mask = 1ULL << control_qubit;
    size_t target_mask = 1ULL << target_qubit;

    // Create hierarchical matrix representation for the gate
    // Gate matrix is 4x4, so dimension is 4
    HierarchicalMatrix* h_matrix = create_hierarchical_matrix(4, 1e-10);
    if (h_matrix && h_matrix->data) {
        // Copy gate matrix data into hierarchical matrix
        for (size_t i = 0; i < 16; i++) {
            h_matrix->data[i] = gate_matrix[i];
        }
    }

    #pragma omp parallel
    {
        // Create tensor network for this thread
        tensor_network_t* network = create_tensor_network();
        double complex local_result[4];

        #pragma omp for schedule(guided)
        for (size_t i = 0; i < n; i++) {
            // Only apply when control qubit is 1 and target qubit is 0
            if ((i & control_mask) && (i & target_mask) == 0) {
                size_t i00 = i & ~control_mask & ~target_mask;
                size_t i01 = i00 | target_mask;
                size_t i10 = i00 | control_mask;
                size_t i11 = i00 | control_mask | target_mask;

                // Get the 4 basis state amplitudes
                double complex v[4] = {
                    state[i00], state[i01], state[i10], state[i11]
                };

                // Use tensor network for contraction if network created
                if (network) {
                    // Add state tensor node
                    size_t dims_state[1] = {4};
                    ComplexFloat* state_data = malloc(4 * sizeof(ComplexFloat));
                    if (state_data) {
                        for (int j = 0; j < 4; j++) {
                            state_data[j].real = (float)creal(v[j]);
                            state_data[j].imag = (float)cimag(v[j]);
                        }

                        size_t node_id;
                        add_tensor_node(network, state_data, dims_state, 1, &node_id);
                        free(state_data);
                    }

                    // Contract with gate matrix (h_matrix)
                    if (h_matrix && h_matrix->data) {
                        // Manual matrix-vector multiplication for 4x4 gate
                        for (int row = 0; row < 4; row++) {
                            local_result[row] = 0;
                            for (int col = 0; col < 4; col++) {
                                local_result[row] += h_matrix->data[row * 4 + col] * v[col];
                            }
                        }
                    } else {
                        // Fallback: direct matrix multiplication
                        for (int row = 0; row < 4; row++) {
                            local_result[row] = 0;
                            for (int col = 0; col < 4; col++) {
                                local_result[row] += gate_matrix[row * 4 + col] * v[col];
                            }
                        }
                    }
                } else {
                    // Fallback: direct matrix multiplication without tensor network
                    for (int row = 0; row < 4; row++) {
                        local_result[row] = 0;
                        for (int col = 0; col < 4; col++) {
                            local_result[row] += gate_matrix[row * 4 + col] * v[col];
                        }
                    }
                }

                // Write results back
                state[i00] = local_result[0];
                state[i01] = local_result[1];
                state[i10] = local_result[2];
                state[i11] = local_result[3];
            }
        }

        if (network) {
            destroy_tensor_network(network);
        }
    }

    if (h_matrix) {
        destroy_hierarchical_matrix(h_matrix);
    }
}

// Basic two-qubit gate implementation without tensor networks
static void apply_two_qubit_gate_basic(double complex* state,
                                      const double complex* gate_matrix,
                                      size_t control_qubit,
                                      size_t target_qubit,
                                      size_t n_qubits) {
    size_t n = 1ULL << n_qubits;
    size_t control_mask = 1ULL << control_qubit;
    size_t target_mask = 1ULL << target_qubit;

    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        if ((i & control_mask) && (i & target_mask) == 0) {
            size_t i1 = i;
            size_t i2 = i | target_mask;
            double complex v1 = state[i1];
            double complex v2 = state[i2];

            state[i1] = gate_matrix[0] * v1 + gate_matrix[1] * v2;
            state[i2] = gate_matrix[2] * v1 + gate_matrix[3] * v2;
        }
    }
}

// ============================================================================
// Measurement Implementation
// ============================================================================

// Create quantum_state_t wrapper for error syndrome API
static quantum_state_t* create_state_wrapper(double complex* amplitudes, size_t n_qubits) {
    quantum_state_t* state = calloc(1, sizeof(quantum_state_t));
    if (!state) return NULL;

    state->num_qubits = n_qubits;
    size_t dim = 1ULL << n_qubits;
    state->dimension = dim;
    state->coordinates = malloc(dim * sizeof(ComplexFloat));
    if (!state->coordinates) {
        free(state);
        return NULL;
    }

    // Copy amplitudes
    for (size_t i = 0; i < dim; i++) {
        state->coordinates[i].real = (float)creal(amplitudes[i]);
        state->coordinates[i].imag = (float)cimag(amplitudes[i]);
    }

    return state;
}

// Copy state wrapper back to double complex array
static void copy_state_wrapper_back(const quantum_state_t* state, double complex* amplitudes, size_t n_qubits) {
    size_t dim = 1ULL << n_qubits;
    for (size_t i = 0; i < dim; i++) {
        amplitudes[i] = state->coordinates[i].real + state->coordinates[i].imag * I;
    }
}

// Free state wrapper
static void free_state_wrapper(quantum_state_t* state) {
    if (state) {
        if (state->coordinates) {
            free(state->coordinates);
        }
        free(state);
    }
}

// Measure qubit with error correction using existing API
int measure_qubit_cpu(double complex* state,
                     size_t target_qubit,
                     size_t n_qubits) {
    size_t n = 1ULL << n_qubits;
    size_t mask = 1ULL << target_qubit;
    double prob_0 = 0.0;

    // Calculate probability with error correction
    #pragma omp parallel reduction(+:prob_0)
    {
        double local_prob = 0.0;

        #pragma omp for simd
        for (size_t i = 0; i < n; i++) {
            if ((i & mask) == 0) {
                double complex amp = state[i];
                local_prob += creal(amp * conj(amp));
            }
        }

        prob_0 += local_prob;
    }

    // Apply error correction using proper API
    quantum_state_t* q_state = create_state_wrapper(state, n_qubits);
    if (q_state) {
        ErrorSyndrome syndrome;
        if (init_error_syndrome(&syndrome, n_qubits * 2) == QGT_SUCCESS) {
            if (detect_errors(q_state, &syndrome) == QGT_SUCCESS) {
                correct_errors(q_state, &syndrome);
                // Copy corrected state back
                copy_state_wrapper_back(q_state, state, n_qubits);
            }
            cleanup_error_syndrome(&syndrome);
        }
        free_state_wrapper(q_state);
    }

    // Random measurement with quantum RNG
    init_qrng_if_needed();
    double r = qrng_double(g_qrng_ctx);
    int outcome = (r > prob_0) ? 1 : 0;

    // Collapse state with noise reduction
    double norm = 0.0;

    #pragma omp parallel reduction(+:norm)
    {
        double local_norm = 0.0;

        #pragma omp for simd
        for (size_t i = 0; i < n; i++) {
            if (((i & mask) == 0 && outcome == 0) ||
                ((i & mask) != 0 && outcome == 1)) {
                double complex amp = state[i];
                local_norm += creal(amp * conj(amp));
            } else {
                state[i] = 0;
            }
        }

        norm += local_norm;
    }

    // Normalize with stability check
    norm = sqrt(norm);
    if (norm > 1e-10) {
        #pragma omp parallel for simd
        for (size_t i = 0; i < n; i++) {
            state[i] /= norm;
        }
    }

    return outcome;
}

// Basic measurement without error correction
static int measure_qubit_basic(double complex* state,
                              size_t target_qubit,
                              size_t n_qubits) {
    size_t n = 1ULL << n_qubits;
    size_t mask = 1ULL << target_qubit;
    double prob_0 = 0.0;

    #pragma omp parallel reduction(+:prob_0)
    {
        double local_prob = 0.0;

        #pragma omp for simd
        for (size_t i = 0; i < n; i++) {
            if ((i & mask) == 0) {
                double complex amp = state[i];
                local_prob += creal(amp * conj(amp));
            }
        }

        prob_0 += local_prob;
    }

    init_qrng_if_needed();
    double r = qrng_double(g_qrng_ctx);
    int outcome = (r > prob_0) ? 1 : 0;

    double norm = 0.0;

    #pragma omp parallel reduction(+:norm)
    {
        double local_norm = 0.0;

        #pragma omp for simd
        for (size_t i = 0; i < n; i++) {
            if (((i & mask) == 0 && outcome == 0) ||
                ((i & mask) != 0 && outcome == 1)) {
                double complex amp = state[i];
                local_norm += creal(amp * conj(amp));
            } else {
                state[i] = 0;
            }
        }

        norm += local_norm;
    }

    norm = sqrt(norm);
    if (norm > 1e-10) {
        #pragma omp parallel for simd
        for (size_t i = 0; i < n; i++) {
            state[i] /= norm;
        }
    }

    return outcome;
}

// ============================================================================
// Circuit Simulation
// ============================================================================

// Simulate CPU circuit
void simulate_cpu_circuit(double complex* state,
                         const CPUCircuit* circuit,
                         size_t n_qubits) {
    if (!state || !circuit) {
        return;
    }

    // Track error rates for monitoring
    double total_error = 0.0;
    size_t error_gates = 0;

    for (size_t i = 0; i < circuit->n_gates; i++) {
        const CPUGate* gate = &circuit->gates[i];

        switch (gate->type) {
            case CPU_GATE_SINGLE:
                apply_single_qubit_gate_cpu(state, gate->matrix,
                                           gate->target, n_qubits);
                break;

            case CPU_GATE_TWO:
                if (circuit->use_tensor_networks) {
                    apply_two_qubit_gate_cpu(state, gate->matrix,
                                            gate->control, gate->target,
                                            n_qubits);
                } else {
                    apply_two_qubit_gate_basic(state, gate->matrix,
                                              gate->control, gate->target,
                                              n_qubits);
                }
                break;

            case CPU_GATE_MEASURE:
                if (circuit->use_error_correction) {
                    measure_qubit_cpu(state, gate->target, n_qubits);
                } else {
                    measure_qubit_basic(state, gate->target, n_qubits);
                }
                break;

            case CPU_GATE_ERROR_DETECT:
                if (circuit->use_error_correction) {
                    quantum_state_t* q_state = create_state_wrapper(state, n_qubits);
                    if (q_state) {
                        ErrorSyndrome syndrome;
                        if (init_error_syndrome(&syndrome, n_qubits * 2) == QGT_SUCCESS) {
                            if (detect_errors(q_state, &syndrome) == QGT_SUCCESS) {
                                if (gate->target < syndrome.num_errors) {
                                    total_error += syndrome.error_weights[gate->target];
                                }
                                error_gates++;
                            }
                            cleanup_error_syndrome(&syndrome);
                        }
                        free_state_wrapper(q_state);
                    }
                }
                break;

            case CPU_GATE_ERROR_CORRECT:
                if (circuit->use_error_correction && error_gates > 0 &&
                    total_error / error_gates > gate->error_threshold) {
                    quantum_state_t* q_state = create_state_wrapper(state, n_qubits);
                    if (q_state) {
                        ErrorSyndrome syndrome;
                        if (init_error_syndrome(&syndrome, n_qubits * 2) == QGT_SUCCESS) {
                            if (detect_errors(q_state, &syndrome) == QGT_SUCCESS) {
                                correct_errors(q_state, &syndrome);
                                copy_state_wrapper_back(q_state, state, n_qubits);
                            }
                            cleanup_error_syndrome(&syndrome);
                        }
                        free_state_wrapper(q_state);
                        total_error = 0.0;
                        error_gates = 0;
                    }
                }
                break;
        }
    }
}

// ============================================================================
// CPU Circuit Management
// ============================================================================

// Initialize CPU circuit
CPUCircuit* init_cpu_circuit(size_t max_gates) {
    CPUCircuit* circuit = malloc(sizeof(CPUCircuit));
    if (!circuit) {
        return NULL;
    }

    circuit->gates = malloc(max_gates * sizeof(CPUGate));
    if (!circuit->gates) {
        free(circuit);
        return NULL;
    }

    circuit->n_gates = 0;
    circuit->max_gates = max_gates;
    circuit->use_error_correction = true;  // Enable by default
    circuit->use_tensor_networks = true;   // Enable by default
    circuit->cache_line_size = 64;         // Default cache line size
    return circuit;
}

// Configure circuit optimization parameters
void configure_cpu_circuit_optimization(CPUCircuit* circuit,
                                        bool use_error_correction,
                                        bool use_tensor_networks,
                                        size_t cache_line_size) {
    if (!circuit) {
        return;
    }

    circuit->use_error_correction = use_error_correction;
    circuit->use_tensor_networks = use_tensor_networks;
    circuit->cache_line_size = cache_line_size;
}

// Add gate to CPU circuit
void add_gate_to_cpu_circuit(CPUCircuit* circuit, const CPUGate* gate) {
    if (!circuit || !gate || circuit->n_gates >= circuit->max_gates) {
        return;
    }

    // Copy gate with validation
    CPUGate* new_gate = &circuit->gates[circuit->n_gates];
    memcpy(new_gate, gate, sizeof(CPUGate));

    // Set default error threshold if not specified
    if (gate->type == CPU_GATE_ERROR_DETECT || gate->type == CPU_GATE_ERROR_CORRECT) {
        if (new_gate->error_threshold <= 0.0) {
            new_gate->error_threshold = 0.01; // 1% default threshold
        }
    }

    circuit->n_gates++;
}

// Get CPU circuit error statistics
void get_cpu_circuit_error_statistics(const CPUCircuit* circuit,
                                      double* avg_error_rate,
                                      double* max_error_rate) {
    if (!circuit || !avg_error_rate || !max_error_rate) {
        return;
    }

    double total_error = 0.0;
    *max_error_rate = 0.0;
    size_t error_gates = 0;

    for (size_t i = 0; i < circuit->n_gates; i++) {
        if (circuit->gates[i].type == CPU_GATE_ERROR_DETECT ||
            circuit->gates[i].type == CPU_GATE_ERROR_CORRECT) {
            double error = circuit->gates[i].error_threshold;
            total_error += error;
            if (error > *max_error_rate) {
                *max_error_rate = error;
            }
            error_gates++;
        }
    }

    *avg_error_rate = error_gates > 0 ? total_error / error_gates : 0.0;
}

// Cleanup CPU circuit
void cleanup_cpu_circuit(CPUCircuit* circuit) {
    if (!circuit) {
        return;
    }

    // Free gate matrices
    for (size_t i = 0; i < circuit->n_gates; i++) {
        if (circuit->gates[i].matrix) {
            free(circuit->gates[i].matrix);
        }
    }

    if (circuit->gates) {
        free(circuit->gates);
    }
    free(circuit);
}

// ============================================================================
// Standard Gate Matrices
// ============================================================================

// Generate standard gate matrix from gate type
static double complex* generate_gate_matrix(gate_type_t type, double parameter) {
    double complex* matrix = NULL;

    // Determine if single qubit (2x2) or two qubit (4x4)
    bool is_two_qubit = (type == GATE_CNOT || type == GATE_CZ ||
                         type == GATE_SWAP || type == GATE_ISWAP ||
                         type == GATE_CRZ || type == GATE_CRX || type == GATE_CRY);

    size_t dim = is_two_qubit ? 16 : 4;
    matrix = calloc(dim, sizeof(double complex));
    if (!matrix) return NULL;

    switch (type) {
        case GATE_X:  // Pauli X
            matrix[1] = 1.0;
            matrix[2] = 1.0;
            break;
        case GATE_Y:  // Pauli Y
            matrix[1] = -I;
            matrix[2] = I;
            break;
        case GATE_Z:  // Pauli Z
            matrix[0] = 1.0;
            matrix[3] = -1.0;
            break;
        case GATE_H:  // Hadamard
            matrix[0] = 1.0 / sqrt(2.0);
            matrix[1] = 1.0 / sqrt(2.0);
            matrix[2] = 1.0 / sqrt(2.0);
            matrix[3] = -1.0 / sqrt(2.0);
            break;
        case GATE_S:  // S gate
            matrix[0] = 1.0;
            matrix[3] = I;
            break;
        case GATE_T:  // T gate
            matrix[0] = 1.0;
            matrix[3] = cexp(I * M_PI / 4.0);
            break;
        case GATE_RX:  // Rotation X
            matrix[0] = cos(parameter / 2.0);
            matrix[1] = -I * sin(parameter / 2.0);
            matrix[2] = -I * sin(parameter / 2.0);
            matrix[3] = cos(parameter / 2.0);
            break;
        case GATE_RY:  // Rotation Y
            matrix[0] = cos(parameter / 2.0);
            matrix[1] = -sin(parameter / 2.0);
            matrix[2] = sin(parameter / 2.0);
            matrix[3] = cos(parameter / 2.0);
            break;
        case GATE_RZ:  // Rotation Z
            matrix[0] = cexp(-I * parameter / 2.0);
            matrix[3] = cexp(I * parameter / 2.0);
            break;
        case GATE_CNOT:  // CNOT gate (4x4)
            matrix[0] = 1.0;  // |00⟩ -> |00⟩
            matrix[5] = 1.0;  // |01⟩ -> |01⟩
            matrix[11] = 1.0; // |10⟩ -> |11⟩
            matrix[14] = 1.0; // |11⟩ -> |10⟩
            break;
        case GATE_CZ:  // CZ gate (4x4)
            matrix[0] = 1.0;
            matrix[5] = 1.0;
            matrix[10] = 1.0;
            matrix[15] = -1.0;
            break;
        case GATE_SWAP:  // SWAP gate (4x4)
            matrix[0] = 1.0;
            matrix[6] = 1.0;
            matrix[9] = 1.0;
            matrix[15] = 1.0;
            break;
        case GATE_I:  // Identity
        default:
            matrix[0] = 1.0;
            if (is_two_qubit) {
                matrix[5] = 1.0;
                matrix[10] = 1.0;
                matrix[15] = 1.0;
            } else {
                matrix[3] = 1.0;
            }
            break;
    }

    return matrix;
}

// Check if gate type is a two-qubit gate
static bool is_two_qubit_gate(gate_type_t type) {
    return (type == GATE_CNOT || type == GATE_CZ ||
            type == GATE_SWAP || type == GATE_ISWAP ||
            type == GATE_CRZ || type == GATE_CRX || type == GATE_CRY);
}

// ============================================================================
// HAL Circuit Integration
// ============================================================================

// Simulate HAL QuantumCircuit on CPU
// This converts from HAL QuantumCircuit to internal format and simulates
void simulate_circuit_cpu(double complex* state,
                         const QuantumCircuit* circuit,
                         size_t n_qubits) {
    if (!state || !circuit) {
        return;
    }

    // Create internal CPU circuit from HAL circuit
    CPUCircuit* cpu_circuit = init_cpu_circuit(circuit->num_gates + 1);
    if (!cpu_circuit) {
        return;
    }

    // Default settings for HAL circuit execution
    cpu_circuit->use_error_correction = true;
    cpu_circuit->use_tensor_networks = true;
    cpu_circuit->cache_line_size = 64;

    // Convert HAL gates to CPU gates
    for (size_t i = 0; i < circuit->num_gates; i++) {
        const HardwareGate* hal_gate = &circuit->gates[i];
        CPUGate cpu_gate = {0};

        // Determine gate type and generate matrix from gate type
        if (is_two_qubit_gate(hal_gate->type)) {
            cpu_gate.type = CPU_GATE_TWO;
            cpu_gate.control = hal_gate->control;
            cpu_gate.target = hal_gate->target;
        } else if (hal_gate->type == GATE_MEASURE) {
            cpu_gate.type = CPU_GATE_MEASURE;
            cpu_gate.target = hal_gate->target;
        } else {
            cpu_gate.type = CPU_GATE_SINGLE;
            cpu_gate.target = hal_gate->target;
        }

        // Generate gate matrix from type
        cpu_gate.matrix = generate_gate_matrix(hal_gate->type, hal_gate->parameter);

        add_gate_to_cpu_circuit(cpu_circuit, &cpu_gate);
    }

    // Execute simulation
    simulate_cpu_circuit(state, cpu_circuit, n_qubits);

    // Cleanup
    cleanup_cpu_circuit(cpu_circuit);
}

// ============================================================================
// Global Cleanup
// ============================================================================

// Cleanup QRNG context
void cleanup_cpu_simulator(void) {
    if (g_qrng_ctx) {
        qrng_free(g_qrng_ctx);
        g_qrng_ctx = NULL;
    }
}
