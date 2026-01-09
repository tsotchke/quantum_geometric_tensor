#include "quantum_geometric/distributed/quantum_distributed_operations.h"
#include "quantum_geometric/core/quantum_geometric_operations.h"
#include "quantum_geometric/core/quantum_circuit_operations.h"
#include "quantum_geometric/hardware/quantum_geometric_gpu.h"
#ifndef HAS_MPI
#ifndef NO_MPI
#define NO_MPI
#endif
#endif

#ifndef NO_MPI
#include <mpi.h>
#else
// MPI type stubs for non-MPI builds
typedef int MPI_Comm;
typedef int MPI_Status;
typedef int MPI_Datatype;
typedef int MPI_Op;
#define MPI_COMM_WORLD 0
#define MPI_COMM_NULL 0
#define MPI_BYTE 0
#define MPI_SUCCESS 0
#define MPI_SUM 0

// Stub MPI functions for single-node fallback
static inline int MPI_Comm_rank(MPI_Comm comm, int* rank) { (void)comm; *rank = 0; return 0; }
static inline int MPI_Comm_size(MPI_Comm comm, int* size) { (void)comm; *size = 1; return 0; }
static inline int MPI_Send(const void* buf, int count, MPI_Datatype dt, int dest, int tag, MPI_Comm comm) {
    (void)buf; (void)count; (void)dt; (void)dest; (void)tag; (void)comm;
    return 0;  // No-op for single node
}
static inline int MPI_Recv(void* buf, int count, MPI_Datatype dt, int src, int tag, MPI_Comm comm, MPI_Status* status) {
    (void)buf; (void)count; (void)dt; (void)src; (void)tag; (void)comm; (void)status;
    return 0;  // No-op for single node
}
static inline int MPI_Sendrecv(const void* sb, int sc, MPI_Datatype st, int d, int stag,
                               void* rb, int rc, MPI_Datatype rt, int s, int rtag,
                               MPI_Comm comm, MPI_Status* status) {
    (void)st; (void)d; (void)stag; (void)rt; (void)s; (void)rtag; (void)comm; (void)status;
    // For single-node: copy send buffer to receive buffer
    if (sb && rb && sc > 0) {
        memcpy(rb, sb, (size_t)sc);  // sc is byte count when using MPI_BYTE
    }
    (void)rc;  // Assume rc >= sc
    return 0;
}
static inline int MPI_Barrier(MPI_Comm comm) { (void)comm; return 0; }
static inline int MPI_Allreduce(const void* sb, void* rb, int c, MPI_Datatype dt, MPI_Op op, MPI_Comm comm) {
    (void)dt; (void)op; (void)comm;
    // For single-node: just copy the data (no reduction needed)
    if (sb && rb && c > 0) {
        memcpy(rb, sb, (size_t)c);  // c is byte count when using MPI_BYTE
    }
    return 0;
}
static inline int MPI_Bcast(void* buf, int count, MPI_Datatype dt, int root, MPI_Comm comm) {
    (void)buf; (void)count; (void)dt; (void)root; (void)comm;
    return 0;  // No-op for single node - data already in place
}
static inline int MPI_Gather(const void* sb, int sc, MPI_Datatype sdt,
                             void* rb, int rc, MPI_Datatype rdt,
                             int root, MPI_Comm comm) {
    (void)sdt; (void)rdt; (void)root; (void)comm; (void)rc;
    // For single-node: just copy send buffer to receive buffer
    if (sb && rb && sc > 0) {
        memcpy(rb, sb, (size_t)sc);
    }
    return 0;
}
#define MPI_DOUBLE 1
#endif

#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <stdint.h>

// ============================================================================
// Forward declarations and stubs for distributed quantum operations
// ============================================================================

/**
 * @brief Quantum Fourier Transform on a quantum state
 *
 * Implements the standard QFT using Hadamard and controlled-phase gates.
 * For n qubits, applies:
 *   H on qubit n-1, controlled-R_k from qubits n-2 to 0
 *   H on qubit n-2, controlled-R_k from qubits n-3 to 0
 *   ... and so on, followed by bit reversal
 *
 * @param state Quantum state to transform in place
 */
static void quantum_fourier_transform(QuantumState* state) {
    if (!state || !state->amplitudes) return;

    size_t n = state->num_qubits;
    size_t dim = 1ULL << n;

    // Allocate temporary buffer
    ComplexFloat* temp = malloc(dim * sizeof(ComplexFloat));
    if (!temp) return;

    // Apply QFT: for each output amplitude, compute the sum
    // |k⟩ → (1/√N) Σ_{j=0}^{N-1} exp(2πijk/N) |j⟩
    double norm = 1.0 / sqrt((double)dim);

    #pragma omp parallel for
    for (size_t k = 0; k < dim; k++) {
        double real_sum = 0.0;
        double imag_sum = 0.0;

        for (size_t j = 0; j < dim; j++) {
            double angle = 2.0 * M_PI * (double)(j * k) / (double)dim;
            double cos_a = cos(angle);
            double sin_a = sin(angle);

            // Multiply amplitude by exp(i*angle)
            double amp_r = (double)state->amplitudes[j].real;
            double amp_i = (double)state->amplitudes[j].imag;

            real_sum += amp_r * cos_a - amp_i * sin_a;
            imag_sum += amp_r * sin_a + amp_i * cos_a;
        }

        temp[k].real = (float)(real_sum * norm);
        temp[k].imag = (float)(imag_sum * norm);
    }

    // Copy result back
    memcpy(state->amplitudes, temp, dim * sizeof(ComplexFloat));
    free(temp);

    state->is_normalized = true;
}

/**
 * @brief Measure error syndrome for quantum error correction
 *
 * Simulates syndrome measurement for stabilizer-based error correction.
 * This implementation estimates bit-flip errors by checking amplitude
 * distribution for deviations from expected patterns.
 *
 * @param state Quantum state to measure
 * @param syndrome Output array for syndrome bits
 * @param num_qubits Number of qubits (determines syndrome size)
 */
static void measure_error_syndrome(const QuantumState* state,
                                   bool* syndrome,
                                   size_t num_qubits) {
    if (!state || !state->amplitudes || !syndrome) return;

    size_t dim = 1ULL << state->num_qubits;

    // Initialize syndrome to no errors
    for (size_t i = 0; i < num_qubits; i++) {
        syndrome[i] = false;
    }

    // For each qubit, check if there's evidence of an error
    // by comparing amplitudes of |0⟩ vs |1⟩ subspaces
    for (size_t q = 0; q < num_qubits && q < state->num_qubits; q++) {
        size_t mask = 1ULL << q;
        double prob_0 = 0.0;  // Probability of qubit in |0⟩
        double prob_1 = 0.0;  // Probability of qubit in |1⟩

        for (size_t i = 0; i < dim; i++) {
            double prob = (double)state->amplitudes[i].real *
                         (double)state->amplitudes[i].real +
                         (double)state->amplitudes[i].imag *
                         (double)state->amplitudes[i].imag;

            if (i & mask) {
                prob_1 += prob;
            } else {
                prob_0 += prob;
            }
        }

        // Syndrome bit indicates potential error if probabilities are
        // significantly imbalanced (threshold-based detection)
        // For a surface code, we'd use parity checks instead
        double imbalance = fabs(prob_0 - prob_1);
        syndrome[q] = (imbalance > 0.3);  // Threshold for error indication
    }
}

/**
 * @brief Analyze error syndrome to determine error locations
 *
 * Converts syndrome bits to error location indices.
 * Returns an array of qubit indices where errors were detected.
 *
 * @param syndrome Syndrome bit array
 * @param num_qubits Number of syndrome bits
 * @return Dynamically allocated array of error locations (caller must free)
 */
static size_t* analyze_error_syndrome(const bool* syndrome, size_t num_qubits) {
    if (!syndrome) return NULL;

    // Allocate result array (worst case: all qubits have errors)
    size_t* locations = malloc((num_qubits + 1) * sizeof(size_t));
    if (!locations) return NULL;

    size_t count = 0;
    for (size_t i = 0; i < num_qubits; i++) {
        if (syndrome[i]) {
            locations[count++] = i;
        }
    }

    // Terminate with sentinel value (SIZE_MAX indicates end)
    locations[count] = SIZE_MAX;

    return locations;
}

// Forward declarations for static helper functions (defined later in file)
static double compute_cross_node_phase(const QuantumState* state,
                                       size_t local_qubit,
                                       size_t remote_offset);
static void apply_remote_phase(QuantumState* state, double phase, size_t qubit);
static double estimate_local_phase(const QuantumState* state,
                                   size_t qubit,
                                   size_t precision);
static void combine_phase_estimates(double* phases, size_t num_qubits);
static void apply_phase_corrections(QuantumState* state,
                                    const double* phases,
                                    size_t num_qubits);
static void apply_error_corrections(QuantumState* state,
                                    const bool* syndrome,
                                    size_t num_syndromes);

// MPI tags
#define TAG_QUANTUM_STATE 1
#define TAG_SYNC_STATE 2
#define TAG_ERROR_SYNDROME 3
#define TAG_PHASE_EST 4

// Distributed quantum parameters
#define MAX_NODES 1024
#define MIN_LOCAL_QUBITS 4
#define SYNC_INTERVAL 100

// Distributed quantum state
typedef struct {
    QuantumState* local_state;
    size_t total_qubits;
    size_t local_qubits;
    int rank;
    int size;
    MPI_Comm comm;
    bool is_synchronized;
} DistributedQuantumState;

// Initialize distributed quantum state
DistributedQuantumState* init_distributed_state(size_t num_qubits) {
    DistributedQuantumState* state = malloc(sizeof(DistributedQuantumState));
    if (!state) return NULL;
    
    // Get MPI info
    MPI_Comm_rank(MPI_COMM_WORLD, &state->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &state->size);
    
    // Calculate local qubits
    state->total_qubits = num_qubits;
    state->local_qubits = num_qubits / state->size;
    if (state->local_qubits < MIN_LOCAL_QUBITS) {
        state->local_qubits = MIN_LOCAL_QUBITS;
    }
    
    // Initialize local state
    state->local_state = init_quantum_state(state->local_qubits);
    if (!state->local_state) {
        free(state);
        return NULL;
    }
    
    state->comm = MPI_COMM_WORLD;
    state->is_synchronized = true;
    
    return state;
}

// Synchronize quantum states across nodes
static void sync_quantum_states(DistributedQuantumState* state) {
    if (!state || state->is_synchronized) return;
    
    size_t local_dim = 1ULL << state->local_qubits;
    
    // Gather all states to root
    if (state->rank == 0) {
        for (int i = 1; i < state->size; i++) {
            MPI_Status status;
            MPI_Recv(state->local_state->amplitudes + i * local_dim,
                    local_dim * sizeof(double complex),
                    MPI_BYTE, i, TAG_SYNC_STATE,
                    state->comm, &status);
        }
    } else {
        MPI_Send(state->local_state->amplitudes,
                local_dim * sizeof(double complex),
                MPI_BYTE, 0, TAG_SYNC_STATE,
                state->comm);
    }
    
    // Broadcast complete state
    MPI_Bcast(state->local_state->amplitudes,
              local_dim * state->size * sizeof(double complex),
              MPI_BYTE, 0, state->comm);
    
    state->is_synchronized = true;
}

// Distributed quantum Fourier transform
void distributed_quantum_fourier_transform(DistributedQuantumState* state) {
    if (!state) return;
    
    // Synchronize states
    sync_quantum_states(state);
    
    // Local QFT
    quantum_fourier_transform(state->local_state);
    
    // Cross-node phase rotations
    for (size_t i = 0; i < state->local_qubits; i++) {
        for (int node = 0; node < state->size; node++) {
            if (node != state->rank) {
                // Exchange phase information
                double phase = compute_cross_node_phase(
                    state->local_state, i,
                    node * state->local_qubits
                );
                
                MPI_Bcast(&phase, 1, MPI_DOUBLE,
                         node, state->comm);
                
                // Apply received phase
                apply_remote_phase(state->local_state,
                                 phase, i);
            }
        }
    }
    
    state->is_synchronized = false;
}

// Distributed quantum matrix multiplication
void distributed_quantum_multiply(DistributedQuantumState* a,
                               DistributedQuantumState* b) {
    if (!a || !b) return;
    
    // Synchronize states
    sync_quantum_states(a);
    sync_quantum_states(b);
    
    // Local multiplication
    quantum_circuit_multiply(a->local_state, b->local_state);
    
    // Cross-node operations
    for (int node = 0; node < a->size; node++) {
        if (node != a->rank) {
            // Exchange quantum states
            MPI_Status status;
            QuantumState* remote_state = init_quantum_state(a->local_qubits);
            
            if (a->rank < node) {
                MPI_Send(a->local_state->amplitudes,
                        (1ULL << a->local_qubits) * sizeof(double complex),
                        MPI_BYTE, node, TAG_QUANTUM_STATE,
                        a->comm);
                MPI_Recv(remote_state->amplitudes,
                        (1ULL << a->local_qubits) * sizeof(double complex),
                        MPI_BYTE, node, TAG_QUANTUM_STATE,
                        a->comm, &status);
            } else {
                MPI_Recv(remote_state->amplitudes,
                        (1ULL << a->local_qubits) * sizeof(double complex),
                        MPI_BYTE, node, TAG_QUANTUM_STATE,
                        a->comm, &status);
                MPI_Send(a->local_state->amplitudes,
                        (1ULL << a->local_qubits) * sizeof(double complex),
                        MPI_BYTE, node, TAG_QUANTUM_STATE,
                        a->comm);
            }
            
            // Perform cross-node multiplication
            quantum_circuit_multiply(a->local_state, remote_state);
            
            free(remote_state->amplitudes);
            free(remote_state);
        }
    }
    
    a->is_synchronized = false;
    b->is_synchronized = false;
}

// Distributed quantum error correction
void distributed_error_correction(DistributedQuantumState* state) {
    if (!state) return;
    
    // Local error detection
    bool* local_syndrome = malloc(state->local_qubits * sizeof(bool));
    if (!local_syndrome) return;
    
    measure_error_syndrome(state->local_state,
                         local_syndrome,
                         state->local_qubits);
    
    // Gather all syndromes
    bool* global_syndrome = NULL;
    if (state->rank == 0) {
        global_syndrome = malloc(state->total_qubits * sizeof(bool));
    }
    
    MPI_Gather(local_syndrome, state->local_qubits, MPI_BYTE,
               global_syndrome, state->local_qubits, MPI_BYTE,
               0, state->comm);
    
    // Root analyzes global syndrome
    if (state->rank == 0) {
        // Analyze error patterns
        size_t* error_locations = analyze_error_syndrome(
            global_syndrome, state->total_qubits);
        
        // Broadcast correction information
        MPI_Bcast(error_locations,
                 state->total_qubits * sizeof(size_t),
                 MPI_BYTE, 0, state->comm);
        
        free(error_locations);
    }
    
    // Apply corrections locally
    apply_error_corrections(state->local_state,
                          local_syndrome,
                          state->local_qubits);
    
    free(local_syndrome);
    if (global_syndrome) free(global_syndrome);
    
    state->is_synchronized = false;
}

// Distributed quantum phase estimation
void distributed_phase_estimation(DistributedQuantumState* state,
                               size_t precision) {
    if (!state) return;
    
    // Synchronize state
    sync_quantum_states(state);
    
    // Allocate phase storage
    double* local_phases = malloc(state->local_qubits * sizeof(double));
    double* global_phases = NULL;
    if (state->rank == 0) {
        global_phases = malloc(state->total_qubits * sizeof(double));
    }
    
    // Local phase estimation
    for (size_t i = 0; i < state->local_qubits; i++) {
        local_phases[i] = estimate_local_phase(
            state->local_state, i, precision);
    }
    
    // Gather all phases
    MPI_Gather(local_phases, state->local_qubits, MPI_DOUBLE,
               global_phases, state->local_qubits, MPI_DOUBLE,
               0, state->comm);
    
    // Root combines phases
    if (state->rank == 0) {
        combine_phase_estimates(global_phases,
                              state->total_qubits);
        
        // Broadcast final phases
        MPI_Bcast(global_phases,
                 state->total_qubits * sizeof(double),
                 MPI_BYTE, 0, state->comm);
    }
    
    // Apply phase corrections locally
    apply_phase_corrections(state->local_state,
                          local_phases,
                          state->local_qubits);
    
    free(local_phases);
    if (global_phases) free(global_phases);
    
    state->is_synchronized = false;
}

// Clean up distributed quantum state
void cleanup_distributed_state(DistributedQuantumState* state) {
    if (!state) return;
    
    if (state->local_state) {
        free(state->local_state->amplitudes);
        free(state->local_state);
    }
    
    free(state);
}

// ============================================================================
// Helper functions for phase estimation and error correction
// ============================================================================

/**
 * Compute the cross-node phase rotation for distributed QFT.
 *
 * In a distributed QFT, when qubits are split across nodes, we need to
 * compute controlled-phase rotations between local and remote qubits.
 * The phase is given by: θ = 2π / 2^(|i-j|+1) where i,j are qubit indices.
 *
 * @param state Local quantum state
 * @param local_qubit Index of local qubit
 * @param remote_offset Global offset of remote qubit block
 * @return Phase rotation angle in radians
 */
static double compute_cross_node_phase(const QuantumState* state,
                                       size_t local_qubit,
                                       size_t remote_offset) {
    if (!state || !state->amplitudes) return 0.0;

    // Calculate the global index of the local qubit
    size_t local_dim = 1ULL << state->num_qubits;

    // Compute the average phase contribution from local amplitudes
    // that would interact with the remote qubits
    double phase_sum = 0.0;
    double weight_sum = 0.0;
    size_t mask = 1ULL << local_qubit;

    #pragma omp parallel for reduction(+:phase_sum,weight_sum)
    for (size_t i = 0; i < local_dim; i++) {
        if (i & mask) {
            // This amplitude contributes to cross-node phase
            double amp_real = (double)state->amplitudes[i].real;
            double amp_imag = (double)state->amplitudes[i].imag;
            double probability = amp_real * amp_real + amp_imag * amp_imag;

            if (probability > 1e-15) {
                // Extract phase from amplitude
                double local_phase = atan2(amp_imag, amp_real);
                phase_sum += local_phase * probability;
                weight_sum += probability;
            }
        }
    }

    if (weight_sum > 1e-15) {
        // Return weighted average phase
        double avg_phase = phase_sum / weight_sum;

        // Apply QFT phase rotation factor based on distance
        // For cross-node QFT, the rotation is θ = 2π / 2^(distance+1)
        size_t distance = (remote_offset > local_qubit) ?
                          (remote_offset - local_qubit) :
                          (local_qubit - remote_offset);
        double qft_factor = 2.0 * M_PI / (double)(1ULL << (distance + 1));

        return avg_phase + qft_factor;
    }

    return 0.0;
}

static void apply_remote_phase(QuantumState* state,
                             double phase,
                             size_t qubit) {
    size_t dim = 1ULL << state->num_qubits;
    size_t mask = 1ULL << qubit;
    double cos_p = cos(phase);
    double sin_p = sin(phase);

    #pragma omp parallel for
    for (size_t i = 0; i < dim; i++) {
        if (i & mask) {
            // Multiply by exp(i*phase) = cos(phase) + i*sin(phase)
            // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
            float a = state->amplitudes[i].real;
            float b = state->amplitudes[i].imag;
            state->amplitudes[i].real = (float)(a * cos_p - b * sin_p);
            state->amplitudes[i].imag = (float)(a * sin_p + b * cos_p);
        }
    }
}

/**
 * Estimate the phase of a local qubit using iterative phase estimation.
 *
 * This implements a simplified version of quantum phase estimation that
 * works on classical simulation by analyzing amplitude phases directly.
 * For actual quantum hardware, this would use Hadamard-test circuits.
 *
 * @param state Local quantum state
 * @param qubit Index of qubit to estimate phase for
 * @param precision Number of bits of precision (more iterations = better estimate)
 * @return Estimated phase in radians
 */
static double estimate_local_phase(const QuantumState* state,
                                   size_t qubit,
                                   size_t precision) {
    if (!state || !state->amplitudes || qubit >= state->num_qubits) {
        return 0.0;
    }

    size_t dim = 1ULL << state->num_qubits;
    size_t mask = 1ULL << qubit;

    // Iterative phase estimation using amplitude analysis
    // We compute the phase difference between |0⟩ and |1⟩ states of this qubit
    double complex sum_0 = 0.0;  // Sum of amplitudes where qubit is |0⟩
    double complex sum_1 = 0.0;  // Sum of amplitudes where qubit is |1⟩

    for (size_t i = 0; i < dim; i++) {
        double complex amp = state->amplitudes[i].real +
                            I * state->amplitudes[i].imag;
        if (i & mask) {
            sum_1 += amp;
        } else {
            sum_0 += amp;
        }
    }

    // Compute magnitudes for normalization
    double mag_0 = cabs(sum_0);
    double mag_1 = cabs(sum_1);

    if (mag_0 < 1e-15 || mag_1 < 1e-15) {
        // One state has no amplitude - phase is undefined
        return 0.0;
    }

    // Extract phases
    double phase_0 = carg(sum_0);
    double phase_1 = carg(sum_1);

    // The relative phase is what we want
    double relative_phase = phase_1 - phase_0;

    // Normalize to [0, 2π)
    while (relative_phase < 0) relative_phase += 2.0 * M_PI;
    while (relative_phase >= 2.0 * M_PI) relative_phase -= 2.0 * M_PI;

    // Apply precision rounding - discretize to 2^precision bins
    if (precision > 0 && precision < 20) {
        double bin_size = 2.0 * M_PI / (double)(1ULL << precision);
        double num_bins = round(relative_phase / bin_size);
        relative_phase = num_bins * bin_size;
    }

    return relative_phase;
}

/**
 * Combine phase estimates from multiple nodes with error correction.
 *
 * This implements a robust phase combination algorithm that:
 * 1. Detects outlier estimates using median absolute deviation
 * 2. Applies weighted averaging based on confidence
 * 3. Corrects for systematic phase drift
 *
 * @param phases Array of phase estimates (modified in place)
 * @param num_qubits Number of phase estimates
 */
static void combine_phase_estimates(double* phases, size_t num_qubits) {
    if (!phases || num_qubits == 0) return;

    // Step 1: Compute circular mean to handle wraparound
    // For phases, we can't just average - we need circular statistics
    double sin_sum = 0.0;
    double cos_sum = 0.0;

    for (size_t i = 0; i < num_qubits; i++) {
        sin_sum += sin(phases[i]);
        cos_sum += cos(phases[i]);
    }

    double circular_mean = atan2(sin_sum, cos_sum);

    // Step 2: Compute circular deviation from mean
    double* deviations = malloc(num_qubits * sizeof(double));
    if (!deviations) return;

    for (size_t i = 0; i < num_qubits; i++) {
        // Angular difference handling wraparound
        double diff = phases[i] - circular_mean;
        while (diff > M_PI) diff -= 2.0 * M_PI;
        while (diff < -M_PI) diff += 2.0 * M_PI;
        deviations[i] = fabs(diff);
    }

    // Step 3: Compute median absolute deviation (MAD) for outlier detection
    // Simple selection of median using partial sort
    double* sorted_dev = malloc(num_qubits * sizeof(double));
    if (!sorted_dev) {
        free(deviations);
        return;
    }
    memcpy(sorted_dev, deviations, num_qubits * sizeof(double));

    // Simple insertion sort for small arrays (quick enough for qubit counts)
    for (size_t i = 1; i < num_qubits; i++) {
        double key = sorted_dev[i];
        size_t j = i;
        while (j > 0 && sorted_dev[j - 1] > key) {
            sorted_dev[j] = sorted_dev[j - 1];
            j--;
        }
        sorted_dev[j] = key;
    }

    double mad = sorted_dev[num_qubits / 2];  // Median
    free(sorted_dev);

    // Step 4: Apply weighted combination with outlier suppression
    double weight_sum = 0.0;
    sin_sum = 0.0;
    cos_sum = 0.0;

    for (size_t i = 0; i < num_qubits; i++) {
        // Weight inversely proportional to deviation, with outlier cutoff
        double weight = 1.0;
        if (mad > 1e-10) {
            double normalized_dev = deviations[i] / mad;
            if (normalized_dev > 3.0) {
                // Outlier - heavily suppress
                weight = 0.1 / normalized_dev;
            } else {
                // Normal - weight by inverse deviation
                weight = 1.0 / (1.0 + normalized_dev * normalized_dev);
            }
        }

        sin_sum += weight * sin(phases[i]);
        cos_sum += weight * cos(phases[i]);
        weight_sum += weight;
    }

    free(deviations);

    if (weight_sum > 1e-15) {
        double combined_phase = atan2(sin_sum / weight_sum, cos_sum / weight_sum);

        // Step 5: Apply phase corrections - shift all phases toward combined estimate
        // This reduces overall phase error while maintaining relative phases
        for (size_t i = 0; i < num_qubits; i++) {
            double diff = phases[i] - circular_mean;
            while (diff > M_PI) diff -= 2.0 * M_PI;
            while (diff < -M_PI) diff += 2.0 * M_PI;

            // Blend toward combined estimate (0.8 original, 0.2 correction)
            double correction = (combined_phase - circular_mean) * 0.2;
            phases[i] = phases[i] + correction;

            // Normalize to [0, 2π)
            while (phases[i] < 0) phases[i] += 2.0 * M_PI;
            while (phases[i] >= 2.0 * M_PI) phases[i] -= 2.0 * M_PI;
        }
    }
}

static void apply_phase_corrections(QuantumState* state,
                                  const double* phases,
                                  size_t num_qubits) {
    for (size_t i = 0; i < num_qubits; i++) {
        apply_remote_phase(state, phases[i], i);
    }
}

// ============================================================================
// Surface Code Structure and Decoder
// ============================================================================

/**
 * Surface code structure for a d×d rotated surface code.
 *
 * Layout for d=3 (9 data qubits, 8 syndrome qubits):
 *
 *     Z0      Z1
 *   D0 -- D1 -- D2
 *      X0    X1
 *   D3 -- D4 -- D5
 *      X2    X3
 *   D6 -- D7 -- D8
 *     Z2      Z3
 *
 * - D0-D8: Data qubits (9 qubits)
 * - X0-X3: X-stabilizers (detect Z errors)
 * - Z0-Z3: Z-stabilizers (detect X errors)
 */
typedef struct {
    size_t d;                    // Code distance
    size_t num_data_qubits;      // d²
    size_t num_x_stabilizers;    // (d-1) * d / 2 for rotated, (d-1)*d for unrotated
    size_t num_z_stabilizers;    // Same structure
    // Adjacency: which data qubits each stabilizer measures
    size_t* x_stabilizer_qubits; // [num_x_stabilizers][4] - up to 4 adjacent data qubits
    size_t* z_stabilizer_qubits; // [num_z_stabilizers][4]
    size_t* x_stabilizer_count;  // How many qubits each X-stabilizer touches (2 or 4)
    size_t* z_stabilizer_count;  // How many qubits each Z-stabilizer touches
} SurfaceCode;

static SurfaceCode* surface_code_create(size_t d) {
    if (d < 3 || d % 2 == 0) return NULL;  // Must be odd >= 3

    SurfaceCode* code = calloc(1, sizeof(SurfaceCode));
    if (!code) return NULL;

    code->d = d;
    code->num_data_qubits = d * d;

    // For rotated surface code: (d²-1)/2 stabilizers of each type
    code->num_x_stabilizers = (d * d - 1) / 2;
    code->num_z_stabilizers = (d * d - 1) / 2;

    code->x_stabilizer_qubits = calloc(code->num_x_stabilizers * 4, sizeof(size_t));
    code->z_stabilizer_qubits = calloc(code->num_z_stabilizers * 4, sizeof(size_t));
    code->x_stabilizer_count = calloc(code->num_x_stabilizers, sizeof(size_t));
    code->z_stabilizer_count = calloc(code->num_z_stabilizers, sizeof(size_t));

    if (!code->x_stabilizer_qubits || !code->z_stabilizer_qubits ||
        !code->x_stabilizer_count || !code->z_stabilizer_count) {
        free(code->x_stabilizer_qubits);
        free(code->z_stabilizer_qubits);
        free(code->x_stabilizer_count);
        free(code->z_stabilizer_count);
        free(code);
        return NULL;
    }

    // Build stabilizer adjacency for rotated surface code
    // X-stabilizers are on "black" squares, Z-stabilizers on "white" squares
    size_t x_idx = 0, z_idx = 0;

    for (size_t row = 0; row < d; row++) {
        for (size_t col = 0; col < d; col++) {
            size_t pos = row * d + col;
            bool is_x_ancilla = ((row + col) % 2 == 1);  // Checkerboard pattern

            if (is_x_ancilla && x_idx < code->num_x_stabilizers) {
                // X-stabilizer at this position measures adjacent data qubits
                size_t count = 0;
                if (row > 0 && col > 0) {
                    code->x_stabilizer_qubits[x_idx * 4 + count++] = (row - 1) * d + (col - 1);
                }
                if (row > 0 && col < d - 1) {
                    code->x_stabilizer_qubits[x_idx * 4 + count++] = (row - 1) * d + col;
                }
                if (row < d - 1 && col > 0) {
                    code->x_stabilizer_qubits[x_idx * 4 + count++] = row * d + (col - 1);
                }
                if (row < d - 1 && col < d - 1) {
                    code->x_stabilizer_qubits[x_idx * 4 + count++] = row * d + col;
                }
                code->x_stabilizer_count[x_idx] = count;
                x_idx++;
            } else if (!is_x_ancilla && z_idx < code->num_z_stabilizers) {
                // Z-stabilizer
                size_t count = 0;
                if (row > 0 && col > 0) {
                    code->z_stabilizer_qubits[z_idx * 4 + count++] = (row - 1) * d + (col - 1);
                }
                if (row > 0 && col < d - 1) {
                    code->z_stabilizer_qubits[z_idx * 4 + count++] = (row - 1) * d + col;
                }
                if (row < d - 1 && col > 0) {
                    code->z_stabilizer_qubits[z_idx * 4 + count++] = row * d + (col - 1);
                }
                if (row < d - 1 && col < d - 1) {
                    code->z_stabilizer_qubits[z_idx * 4 + count++] = row * d + col;
                }
                code->z_stabilizer_count[z_idx] = count;
                z_idx++;
            }
        }
    }

    return code;
}

static void surface_code_destroy(SurfaceCode* code) {
    if (code) {
        free(code->x_stabilizer_qubits);
        free(code->z_stabilizer_qubits);
        free(code->x_stabilizer_count);
        free(code->z_stabilizer_count);
        free(code);
    }
}

// ============================================================================
// Union-Find Decoder
// ============================================================================

typedef struct {
    size_t* parent;
    size_t* rank;
    size_t size;
} UnionFind;

static UnionFind* uf_create(size_t size) {
    UnionFind* uf = malloc(sizeof(UnionFind));
    if (!uf) return NULL;

    uf->parent = malloc(size * sizeof(size_t));
    uf->rank = calloc(size, sizeof(size_t));
    uf->size = size;

    if (!uf->parent || !uf->rank) {
        free(uf->parent);
        free(uf->rank);
        free(uf);
        return NULL;
    }

    for (size_t i = 0; i < size; i++) {
        uf->parent[i] = i;
    }

    return uf;
}

static void uf_destroy(UnionFind* uf) {
    if (uf) {
        free(uf->parent);
        free(uf->rank);
        free(uf);
    }
}

static size_t uf_find(UnionFind* uf, size_t x) {
    if (uf->parent[x] != x) {
        uf->parent[x] = uf_find(uf, uf->parent[x]);  // Path compression
    }
    return uf->parent[x];
}

static void uf_union(UnionFind* uf, size_t x, size_t y) {
    size_t rx = uf_find(uf, x);
    size_t ry = uf_find(uf, y);

    if (rx == ry) return;

    // Union by rank
    if (uf->rank[rx] < uf->rank[ry]) {
        uf->parent[rx] = ry;
    } else if (uf->rank[rx] > uf->rank[ry]) {
        uf->parent[ry] = rx;
    } else {
        uf->parent[ry] = rx;
        uf->rank[rx]++;
    }
}

/**
 * Decode X-syndrome (detects Z errors) using Union-Find.
 *
 * @param code Surface code structure
 * @param x_syndrome X-stabilizer measurement results (1 = defect)
 * @param z_corrections Output: which data qubits need Z corrections
 */
static void decode_x_syndrome(const SurfaceCode* code,
                              const bool* x_syndrome,
                              bool* z_corrections) {
    // Reset corrections
    memset(z_corrections, 0, code->num_data_qubits * sizeof(bool));

    // Find defects (syndrome bits that are 1)
    size_t* defects = malloc(code->num_x_stabilizers * sizeof(size_t));
    size_t num_defects = 0;

    for (size_t i = 0; i < code->num_x_stabilizers; i++) {
        if (x_syndrome[i]) {
            defects[num_defects++] = i;
        }
    }

    if (num_defects == 0) {
        free(defects);
        return;  // No errors detected
    }

    // Union-Find with boundary vertices
    // Boundary represents the "virtual" boundary that absorbs defect pairs
    size_t boundary = code->num_x_stabilizers;  // Virtual boundary vertex
    UnionFind* uf = uf_create(code->num_x_stabilizers + 1);

    // For each defect, find if it shares a data qubit with another defect
    // If so, union them and mark the shared data qubit for correction
    for (size_t i = 0; i < num_defects; i++) {
        size_t s1 = defects[i];

        // Check adjacency with other defects
        for (size_t j = i + 1; j < num_defects; j++) {
            size_t s2 = defects[j];

            // Find shared data qubits between stabilizers s1 and s2
            for (size_t k1 = 0; k1 < code->x_stabilizer_count[s1]; k1++) {
                size_t q1 = code->x_stabilizer_qubits[s1 * 4 + k1];
                for (size_t k2 = 0; k2 < code->x_stabilizer_count[s2]; k2++) {
                    size_t q2 = code->x_stabilizer_qubits[s2 * 4 + k2];
                    if (q1 == q2) {
                        // Shared qubit - this is where the error is
                        uf_union(uf, s1, s2);
                        z_corrections[q1] = !z_corrections[q1];  // Toggle
                    }
                }
            }
        }

        // Connect boundary defects to boundary vertex
        // (stabilizers with fewer than 4 qubits are on the boundary)
        if (code->x_stabilizer_count[s1] < 4) {
            uf_union(uf, s1, boundary);
        }
    }

    // For remaining unpaired defects, connect to boundary via shortest path
    for (size_t i = 0; i < num_defects; i++) {
        size_t s = defects[i];
        size_t root = uf_find(uf, s);

        // Check if connected to boundary
        if (uf_find(uf, boundary) != root) {
            // Find shortest path to boundary and mark corrections
            // For simplicity, pick the first qubit of this stabilizer
            if (code->x_stabilizer_count[s] > 0) {
                size_t q = code->x_stabilizer_qubits[s * 4];
                z_corrections[q] = !z_corrections[q];
            }
            uf_union(uf, s, boundary);
        }
    }

    uf_destroy(uf);
    free(defects);
}

/**
 * Decode Z-syndrome (detects X errors) using Union-Find.
 */
static void decode_z_syndrome(const SurfaceCode* code,
                              const bool* z_syndrome,
                              bool* x_corrections) {
    memset(x_corrections, 0, code->num_data_qubits * sizeof(bool));

    size_t* defects = malloc(code->num_z_stabilizers * sizeof(size_t));
    size_t num_defects = 0;

    for (size_t i = 0; i < code->num_z_stabilizers; i++) {
        if (z_syndrome[i]) {
            defects[num_defects++] = i;
        }
    }

    if (num_defects == 0) {
        free(defects);
        return;
    }

    size_t boundary = code->num_z_stabilizers;
    UnionFind* uf = uf_create(code->num_z_stabilizers + 1);

    for (size_t i = 0; i < num_defects; i++) {
        size_t s1 = defects[i];

        for (size_t j = i + 1; j < num_defects; j++) {
            size_t s2 = defects[j];

            for (size_t k1 = 0; k1 < code->z_stabilizer_count[s1]; k1++) {
                size_t q1 = code->z_stabilizer_qubits[s1 * 4 + k1];
                for (size_t k2 = 0; k2 < code->z_stabilizer_count[s2]; k2++) {
                    size_t q2 = code->z_stabilizer_qubits[s2 * 4 + k2];
                    if (q1 == q2) {
                        uf_union(uf, s1, s2);
                        x_corrections[q1] = !x_corrections[q1];
                    }
                }
            }
        }

        if (code->z_stabilizer_count[s1] < 4) {
            uf_union(uf, s1, boundary);
        }
    }

    for (size_t i = 0; i < num_defects; i++) {
        size_t s = defects[i];
        if (uf_find(uf, boundary) != uf_find(uf, s)) {
            if (code->z_stabilizer_count[s] > 0) {
                size_t q = code->z_stabilizer_qubits[s * 4];
                x_corrections[q] = !x_corrections[q];
            }
            uf_union(uf, s, boundary);
        }
    }

    uf_destroy(uf);
    free(defects);
}

// ============================================================================
// Pauli Gate Application
// ============================================================================

static void apply_pauli_x(QuantumState* state, size_t q) {
    size_t dim = 1ULL << state->num_qubits;
    size_t mask = 1ULL << q;

    #pragma omp parallel for
    for (size_t i = 0; i < dim / 2; i++) {
        size_t lower_mask = mask - 1;
        size_t upper = (i & ~lower_mask) << 1;
        size_t lower = i & lower_mask;
        size_t idx0 = upper | lower;
        size_t idx1 = idx0 | mask;

        ComplexFloat temp = state->amplitudes[idx0];
        state->amplitudes[idx0] = state->amplitudes[idx1];
        state->amplitudes[idx1] = temp;
    }
}

static void apply_pauli_z(QuantumState* state, size_t q) {
    size_t dim = 1ULL << state->num_qubits;
    size_t mask = 1ULL << q;

    #pragma omp parallel for
    for (size_t i = 0; i < dim; i++) {
        if (i & mask) {
            state->amplitudes[i].real = -state->amplitudes[i].real;
            state->amplitudes[i].imag = -state->amplitudes[i].imag;
        }
    }
}

/**
 * Apply error corrections to a quantum state based on syndrome measurements.
 *
 * This function implements proper surface code decoding:
 * 1. Infers code distance from syndrome size
 * 2. Builds the surface code stabilizer structure
 * 3. Decodes X-syndromes to get Z corrections (using Union-Find)
 * 4. Decodes Z-syndromes to get X corrections (using Union-Find)
 * 5. Applies the Pauli corrections to the state
 *
 * Syndrome format: [X-stabilizer results..., Z-stabilizer results...]
 * For a d=3 code: 4 X-stabilizers + 4 Z-stabilizers = 8 syndrome bits
 *
 * @param state Quantum state to correct
 * @param syndrome Syndrome measurement results
 * @param num_syndromes Total number of syndrome bits
 */
static void apply_error_corrections(QuantumState* state,
                                    const bool* syndrome,
                                    size_t num_syndromes) {
    if (!state || !state->amplitudes || !syndrome || num_syndromes == 0) {
        return;
    }

    // Infer code distance from syndrome count
    // For rotated surface code: num_syndromes = d² - 1
    // So d = sqrt(num_syndromes + 1)
    size_t d_squared = num_syndromes + 1;
    size_t d = 1;
    while (d * d < d_squared) d++;

    if (d * d != d_squared || d < 3 || d % 2 == 0) {
        // Invalid syndrome count - can't determine code structure
        // Fall back to no correction (this is honest about our limitations)
        return;
    }

    // Create surface code structure
    SurfaceCode* code = surface_code_create(d);
    if (!code) return;

    // Verify syndrome count matches
    size_t expected = code->num_x_stabilizers + code->num_z_stabilizers;
    if (num_syndromes != expected) {
        surface_code_destroy(code);
        return;
    }

    // Split syndrome into X and Z parts
    const bool* x_syndrome = syndrome;
    const bool* z_syndrome = syndrome + code->num_x_stabilizers;

    // Allocate correction arrays
    bool* x_corrections = calloc(code->num_data_qubits, sizeof(bool));
    bool* z_corrections = calloc(code->num_data_qubits, sizeof(bool));

    if (!x_corrections || !z_corrections) {
        free(x_corrections);
        free(z_corrections);
        surface_code_destroy(code);
        return;
    }

    // Decode syndromes to get corrections
    decode_x_syndrome(code, x_syndrome, z_corrections);  // X-syndrome -> Z errors
    decode_z_syndrome(code, z_syndrome, x_corrections);  // Z-syndrome -> X errors

    // Apply corrections to state
    for (size_t q = 0; q < code->num_data_qubits && q < state->num_qubits; q++) {
        if (x_corrections[q]) {
            apply_pauli_x(state, q);
        }
        if (z_corrections[q]) {
            apply_pauli_z(state, q);
        }
    }

    // Cleanup
    free(x_corrections);
    free(z_corrections);
    surface_code_destroy(code);

    // Pauli operations preserve normalization (unitary)
    state->is_normalized = true;
}
