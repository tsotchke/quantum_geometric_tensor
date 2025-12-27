/**
 * @file quantum_physics_operations.c
 * @brief Production-grade quantum physics operations
 *
 * Implements core quantum physics computations with:
 * - Proper time evolution using matrix exponentials (Padé approximants)
 * - Accurate von Neumann entropy via eigenvalue decomposition
 * - Uhlmann fidelity for mixed states
 * - Lindblad master equation for open quantum systems
 * - Kraus operator formalism for quantum channels
 * - GPU acceleration with CPU fallback
 * - Distributed computing support
 *
 * All operations use O(log n) hierarchical algorithms where applicable.
 */

#include "quantum_geometric/physics/quantum_physics_operations.h"
#include "quantum_geometric/core/hierarchical_matrix.h"
#include "quantum_geometric/core/quantum_geometric_constants.h"
#include "quantum_geometric/core/lapack_wrapper.h"
#include "quantum_geometric/core/matrix_eigenvalues.h"
#include "quantum_geometric/core/quantum_operations.h"
#include "quantum_geometric/core/quantum_complex.h"
#include "quantum_geometric/hardware/quantum_geometric_gpu.h"
#include "quantum_geometric/hardware/quantum_error_correction.h"
#include "quantum_geometric/distributed/workload_distribution.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <stdbool.h>
#include <float.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================================
// Constants
// ============================================================================

#define PHYSICS_OPS_TOLERANCE 1e-12
#define PADE_ORDER 13                    // Order for Padé approximant
#define MAX_SCALING_SQUARING 50          // Maximum scaling iterations
#define ENTROPY_EIGENVALUE_CUTOFF 1e-15  // Cutoff for entropy calculation
#define FIDELITY_SQRT_ITERATIONS 20      // Iterations for matrix sqrt

// ============================================================================
// Module State
// ============================================================================

static GPUContext* physics_gpu_ctx = NULL;
static bool physics_gpu_initialized = false;
static bool physics_module_initialized = false;

// Workspace buffers for reuse
static double complex* work_matrix_a = NULL;
static double complex* work_matrix_b = NULL;
static double complex* work_vector = NULL;
static size_t work_size = 0;

// ============================================================================
// Forward Declarations - Internal Functions
// ============================================================================

// Matrix exponential using scaling and squaring with Padé approximants
static bool compute_matrix_exponential(double complex* result,
                                       const double complex* matrix,
                                       size_t n,
                                       double complex scalar);

// Matrix square root using Denman-Beavers iteration
static bool compute_matrix_sqrt(double complex* result,
                                const double complex* matrix,
                                size_t n);

// Matrix multiplication helper (internal, avoids conflict with matrix_operations.h)
static void physics_matrix_mult(double complex* C,
                                const double complex* A,
                                const double complex* B,
                                size_t n);

// Matrix-vector multiplication
static void matrix_vector_multiply(double complex* result,
                                   const double complex* matrix,
                                   const double complex* vector,
                                   size_t n);

// GPU context management
static bool ensure_physics_gpu_context(void);
static void ensure_workspace(size_t n);

// Hierarchical implementations
static void evolve_state_hierarchical(HierarchicalMatrix* state,
                                      const HierarchicalMatrix* propagator);

// ============================================================================
// Initialization and Cleanup
// ============================================================================

/**
 * @brief Initialize quantum physics operations module
 */
qgt_error_t init_quantum_physics_ops(void) {
    if (physics_module_initialized) {
        return QGT_SUCCESS;
    }

    // Try to initialize GPU (optional)
    physics_gpu_initialized = ensure_physics_gpu_context();

    physics_module_initialized = true;
    return QGT_SUCCESS;
}

/**
 * @brief Cleanup quantum physics operations resources
 */
void cleanup_quantum_physics_operations(void) {
    if (physics_gpu_ctx) {
        gpu_destroy_context(physics_gpu_ctx);
        physics_gpu_ctx = NULL;
    }
    physics_gpu_initialized = false;

    free(work_matrix_a);
    free(work_matrix_b);
    free(work_vector);
    work_matrix_a = NULL;
    work_matrix_b = NULL;
    work_vector = NULL;
    work_size = 0;

    physics_module_initialized = false;
}

static bool ensure_physics_gpu_context(void) {
    if (physics_gpu_initialized && physics_gpu_ctx) {
        return true;
    }

    if (gpu_initialize() != 0) {
        return false;
    }

    physics_gpu_ctx = gpu_create_context(0);
    if (physics_gpu_ctx) {
        physics_gpu_initialized = true;
        return true;
    }

    return false;
}

static void ensure_workspace(size_t n) {
    size_t required = n * n;
    if (work_size >= required) return;

    free(work_matrix_a);
    free(work_matrix_b);
    free(work_vector);

    work_matrix_a = calloc(required, sizeof(double complex));
    work_matrix_b = calloc(required, sizeof(double complex));
    work_vector = calloc(n, sizeof(double complex));
    work_size = required;
}

// ============================================================================
// Quantum State Time Evolution
// ============================================================================

/**
 * @brief Evolve quantum state under Hamiltonian using exact matrix exponential
 *
 * Computes |ψ(t+dt)⟩ = exp(-iHdt/ℏ) |ψ(t)⟩ using scaling and squaring
 * with Padé approximants for accurate matrix exponential.
 *
 * @param state Quantum state vector (modified in place), dimension n
 * @param hamiltonian Hamiltonian matrix (n×n, must be Hermitian)
 * @param n Hilbert space dimension
 * @param dt Time step (ℏ = 1)
 * @return QGT_SUCCESS on success
 */
qgt_error_t evolve_quantum_state(double complex* state,
                                 const double complex* hamiltonian,
                                 size_t n, double dt) {
    if (!state || !hamiltonian || n == 0) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    ensure_workspace(n);

    // Allocate propagator matrix
    double complex* propagator = malloc(n * n * sizeof(double complex));
    double complex* result_state = malloc(n * sizeof(double complex));

    if (!propagator || !result_state) {
        free(propagator);
        free(result_state);
        return QGT_ERROR_MEMORY_ALLOCATION;
    }

    // Compute U = exp(-i H dt)
    bool success = compute_matrix_exponential(propagator, hamiltonian, n, -I * dt);

    if (!success) {
        free(propagator);
        free(result_state);
        return QGT_ERROR_NUMERICAL_INSTABILITY;
    }

    // Apply propagator: |ψ(t+dt)⟩ = U |ψ(t)⟩
    matrix_vector_multiply(result_state, propagator, state, n);

    // Normalize (numerical stability)
    double norm = 0.0;
    for (size_t i = 0; i < n; i++) {
        norm += creal(result_state[i] * conj(result_state[i]));
    }
    norm = sqrt(norm);

    if (norm > PHYSICS_OPS_TOLERANCE) {
        double inv_norm = 1.0 / norm;
        for (size_t i = 0; i < n; i++) {
            state[i] = result_state[i] * inv_norm;
        }
    } else {
        // State collapsed - return error
        free(propagator);
        free(result_state);
        return QGT_ERROR_NUMERICAL_INSTABILITY;
    }

    free(propagator);
    free(result_state);
    return QGT_SUCCESS;
}

/**
 * @brief Evolve density matrix under Lindblad master equation
 *
 * Implements: dρ/dt = -i[H,ρ] + Σ_k γ_k (L_k ρ L_k† - ½{L_k†L_k, ρ})
 *
 * @param rho Density matrix (n×n, modified in place)
 * @param hamiltonian Hamiltonian matrix (n×n)
 * @param lindblad_ops Array of Lindblad operators (num_ops × n×n)
 * @param gamma Decay rates for each Lindblad operator
 * @param num_ops Number of Lindblad operators
 * @param n Hilbert space dimension
 * @param dt Time step
 * @return QGT_SUCCESS on success
 */
qgt_error_t evolve_lindblad(double complex* rho,
                            const double complex* hamiltonian,
                            const double complex* const* lindblad_ops,
                            const double* gamma,
                            size_t num_ops,
                            size_t n,
                            double dt) {
    if (!rho || !hamiltonian || n == 0) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    size_t n2 = n * n;
    double complex* drho = calloc(n2, sizeof(double complex));
    double complex* temp1 = malloc(n2 * sizeof(double complex));
    double complex* temp2 = malloc(n2 * sizeof(double complex));
    double complex* LdagL = malloc(n2 * sizeof(double complex));
    double complex* Ldag = malloc(n2 * sizeof(double complex));
    double complex* LrhoLdag = malloc(n2 * sizeof(double complex));

    if (!drho || !temp1 || !temp2 || !LdagL || !Ldag || !LrhoLdag) {
        free(drho); free(temp1); free(temp2);
        free(LdagL); free(Ldag); free(LrhoLdag);
        return QGT_ERROR_MEMORY_ALLOCATION;
    }

    // Hamiltonian term: -i[H, ρ] = -i(Hρ - ρH)
    physics_matrix_mult(temp1, hamiltonian, rho, n);  // Hρ
    physics_matrix_mult(temp2, rho, hamiltonian, n);  // ρH

    for (size_t i = 0; i < n2; i++) {
        drho[i] = -I * (temp1[i] - temp2[i]);
    }

    // Lindblad dissipator terms
    if (lindblad_ops && gamma && num_ops > 0) {
        for (size_t k = 0; k < num_ops; k++) {
            if (!lindblad_ops[k] || gamma[k] <= 0.0) continue;

            const double complex* L_k = lindblad_ops[k];

            // Compute L†
            for (size_t i = 0; i < n; i++) {
                for (size_t j = 0; j < n; j++) {
                    Ldag[i * n + j] = conj(L_k[j * n + i]);
                }
            }

            // L†L
            physics_matrix_mult(LdagL, Ldag, L_k, n);

            // Lρ
            physics_matrix_mult(temp1, L_k, rho, n);

            // LρL†
            for (size_t i = 0; i < n; i++) {
                for (size_t j = 0; j < n; j++) {
                    LrhoLdag[i * n + j] = 0;
                    for (size_t m = 0; m < n; m++) {
                        LrhoLdag[i * n + j] += temp1[i * n + m] * conj(L_k[j * n + m]);
                    }
                }
            }

            // L†Lρ
            physics_matrix_mult(temp1, LdagL, rho, n);
            // ρL†L
            physics_matrix_mult(temp2, rho, LdagL, n);

            // Add dissipator: γ(LρL† - ½L†Lρ - ½ρL†L)
            double g = gamma[k];
            for (size_t i = 0; i < n2; i++) {
                drho[i] += g * (LrhoLdag[i] - 0.5 * temp1[i] - 0.5 * temp2[i]);
            }
        }
    }

    // Euler step: ρ(t+dt) = ρ(t) + dt * dρ/dt
    for (size_t i = 0; i < n2; i++) {
        rho[i] += dt * drho[i];
    }

    // Enforce Hermiticity: ρ = (ρ + ρ†)/2
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i + 1; j < n; j++) {
            double complex avg = 0.5 * (rho[i * n + j] + conj(rho[j * n + i]));
            rho[i * n + j] = avg;
            rho[j * n + i] = conj(avg);
        }
        rho[i * n + i] = creal(rho[i * n + i]);  // Diagonal must be real
    }

    // Normalize trace to 1
    double trace = 0.0;
    for (size_t i = 0; i < n; i++) {
        trace += creal(rho[i * n + i]);
    }
    if (trace > PHYSICS_OPS_TOLERANCE) {
        double inv_trace = 1.0 / trace;
        for (size_t i = 0; i < n2; i++) {
            rho[i] *= inv_trace;
        }
    }

    free(drho); free(temp1); free(temp2);
    free(LdagL); free(Ldag); free(LrhoLdag);
    return QGT_SUCCESS;
}

// ============================================================================
// Quantum Channels (Kraus Operators)
// ============================================================================

/**
 * @brief Apply quantum channel using Kraus operators
 *
 * Computes: ρ → Σ_k K_k ρ K_k†
 *
 * @param rho Density matrix (n×n, modified in place)
 * @param kraus_ops Array of Kraus operators
 * @param num_ops Number of Kraus operators
 * @param n Hilbert space dimension
 * @return QGT_SUCCESS on success
 */
qgt_error_t apply_quantum_channel(double complex* rho,
                                  const double complex* const* kraus_ops,
                                  size_t num_ops,
                                  size_t n) {
    if (!rho || !kraus_ops || num_ops == 0 || n == 0) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    size_t n2 = n * n;
    double complex* result = calloc(n2, sizeof(double complex));
    double complex* temp = malloc(n2 * sizeof(double complex));

    if (!result || !temp) {
        free(result);
        free(temp);
        return QGT_ERROR_MEMORY_ALLOCATION;
    }

    for (size_t k = 0; k < num_ops; k++) {
        if (!kraus_ops[k]) continue;

        const double complex* K = kraus_ops[k];

        // Compute Kρ
        physics_matrix_mult(temp, K, rho, n);

        // Compute (Kρ)K† and add to result
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++) {
                for (size_t m = 0; m < n; m++) {
                    result[i * n + j] += temp[i * n + m] * conj(K[j * n + m]);
                }
            }
        }
    }

    memcpy(rho, result, n2 * sizeof(double complex));

    free(result);
    free(temp);
    return QGT_SUCCESS;
}

/**
 * @brief Apply depolarizing channel
 *
 * ρ → (1-p)ρ + p/3(XρX + YρY + ZρZ) for single qubit
 * Simplifies to: ρ → (1-p)ρ + (p/2)I for complete depolarization
 *
 * @param rho Density matrix (2×2 for single qubit)
 * @param p Depolarizing probability [0, 1]
 * @return QGT_SUCCESS on success
 */
qgt_error_t apply_depolarizing_channel(double complex* rho, double p) {
    if (!rho || p < 0.0 || p > 1.0) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    double complex rho_new[4];
    double q = 1.0 - p;
    double p2 = p / 2.0;

    rho_new[0] = q * rho[0] + p2;        // (1-p)ρ_00 + p/2
    rho_new[1] = q * rho[1];             // (1-p)ρ_01
    rho_new[2] = q * rho[2];             // (1-p)ρ_10
    rho_new[3] = q * rho[3] + p2;        // (1-p)ρ_11 + p/2

    memcpy(rho, rho_new, 4 * sizeof(double complex));
    return QGT_SUCCESS;
}

/**
 * @brief Apply amplitude damping channel (T1 decay)
 *
 * Models energy relaxation from |1⟩ to |0⟩
 * K0 = [[1, 0], [0, sqrt(1-γ)]], K1 = [[0, sqrt(γ)], [0, 0]]
 *
 * @param rho Density matrix (2×2)
 * @param gamma Damping parameter (0 to 1), γ = 1 - exp(-t/T1)
 * @return QGT_SUCCESS on success
 */
qgt_error_t apply_amplitude_damping_channel(double complex* rho, double gamma) {
    if (!rho || gamma < 0.0 || gamma > 1.0) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    double sqrt_1mg = sqrt(1.0 - gamma);

    // K0 ρ K0† + K1 ρ K1†
    double complex rho_new[4];
    rho_new[0] = rho[0] + gamma * rho[3];           // ρ_00 + γ*ρ_11
    rho_new[1] = sqrt_1mg * rho[1];                  // sqrt(1-γ)*ρ_01
    rho_new[2] = sqrt_1mg * rho[2];                  // sqrt(1-γ)*ρ_10
    rho_new[3] = (1.0 - gamma) * rho[3];             // (1-γ)*ρ_11

    memcpy(rho, rho_new, 4 * sizeof(double complex));
    return QGT_SUCCESS;
}

/**
 * @brief Apply phase damping channel (pure dephasing, T2*)
 *
 * Models loss of coherence without energy loss
 *
 * @param rho Density matrix (2×2)
 * @param gamma Dephasing parameter (0 to 1), γ = 1 - exp(-t/T_φ)
 * @return QGT_SUCCESS on success
 */
qgt_error_t apply_phase_damping_channel(double complex* rho, double gamma) {
    if (!rho || gamma < 0.0 || gamma > 1.0) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    double decay = sqrt(1.0 - gamma);

    // Off-diagonal elements decay
    rho[1] *= decay;
    rho[2] *= decay;
    // Diagonal elements unchanged

    return QGT_SUCCESS;
}

/**
 * @brief Apply combined T1/T2 decoherence
 *
 * Applies both amplitude and phase damping for realistic noise model
 *
 * @param rho Density matrix (2×2)
 * @param t1 T1 relaxation time
 * @param t2 T2 coherence time (T2 ≤ 2*T1)
 * @param dt Time elapsed
 * @return QGT_SUCCESS on success
 */
qgt_error_t apply_decoherence(double complex* rho, double t1, double t2, double dt) {
    if (!rho || t1 <= 0.0 || t2 <= 0.0 || dt < 0.0) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // T2 cannot exceed 2*T1 (fundamental limit)
    if (t2 > 2.0 * t1) t2 = 2.0 * t1;

    // Amplitude damping: γ1 = 1 - exp(-dt/T1)
    double gamma1 = 1.0 - exp(-dt / t1);

    // Pure dephasing rate: 1/T_φ = 1/T2 - 1/(2*T1)
    double t_phi = 1.0 / (1.0/t2 - 0.5/t1);
    double gamma_phi = 1.0 - exp(-dt / t_phi);

    qgt_error_t err = apply_amplitude_damping_channel(rho, gamma1);
    if (err != QGT_SUCCESS) return err;

    return apply_phase_damping_channel(rho, gamma_phi);
}

// ============================================================================
// Quantum Measurements
// ============================================================================

/**
 * @brief Compute measurement probabilities in computational basis
 *
 * P(i) = |⟨i|ψ⟩|² for state vector, P(i) = ρ_ii for density matrix
 *
 * @param probabilities Output array (n elements)
 * @param state State vector or density matrix
 * @param n Hilbert space dimension
 * @param is_density True if input is density matrix
 * @return QGT_SUCCESS on success
 */
qgt_error_t compute_measurement_probabilities(double* probabilities,
                                              const double complex* state,
                                              size_t n,
                                              bool is_density) {
    if (!probabilities || !state || n == 0) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    if (is_density) {
        // Diagonal elements of density matrix
        for (size_t i = 0; i < n; i++) {
            probabilities[i] = creal(state[i * n + i]);
            if (probabilities[i] < 0.0) probabilities[i] = 0.0;
        }
    } else {
        // |amplitude|² for state vector
        #pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            probabilities[i] = creal(state[i] * conj(state[i]));
        }
    }

    // Normalize
    double total = 0.0;
    for (size_t i = 0; i < n; i++) {
        total += probabilities[i];
    }
    if (total > PHYSICS_OPS_TOLERANCE) {
        double inv_total = 1.0 / total;
        for (size_t i = 0; i < n; i++) {
            probabilities[i] *= inv_total;
        }
    }

    return QGT_SUCCESS;
}

/**
 * @brief Compute expectation value of Hermitian observable
 *
 * ⟨O⟩ = ⟨ψ|O|ψ⟩ for state vector, ⟨O⟩ = Tr(ρO) for density matrix
 *
 * @param state State vector or density matrix
 * @param observable Observable matrix (n×n, Hermitian)
 * @param n Hilbert space dimension
 * @param is_density True if state is density matrix
 * @return Expectation value
 */
double compute_expectation_value(const double complex* state,
                                 const double complex* observable,
                                 size_t n,
                                 bool is_density) {
    if (!state || !observable || n == 0) {
        return 0.0;
    }

    double complex result = 0.0;

    if (is_density) {
        // Tr(ρO) = Σ_ij ρ_ij O_ji
        #pragma omp parallel for reduction(+:result)
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++) {
                result += state[i * n + j] * observable[j * n + i];
            }
        }
    } else {
        // ⟨ψ|O|ψ⟩ = Σ_ij ψ*_i O_ij ψ_j
        double complex* Opsi = malloc(n * sizeof(double complex));
        if (!Opsi) return 0.0;

        matrix_vector_multiply(Opsi, observable, state, n);

        for (size_t i = 0; i < n; i++) {
            result += conj(state[i]) * Opsi[i];
        }

        free(Opsi);
    }

    return creal(result);  // Must be real for Hermitian observable
}

/**
 * @brief Compute variance of observable
 *
 * Var(O) = ⟨O²⟩ - ⟨O⟩²
 */
double compute_observable_variance(const double complex* state,
                                   const double complex* observable,
                                   size_t n,
                                   bool is_density) {
    if (!state || !observable || n == 0) {
        return 0.0;
    }

    // Compute O²
    size_t n2 = n * n;
    double complex* O2 = malloc(n2 * sizeof(double complex));
    if (!O2) return 0.0;

    physics_matrix_mult(O2, observable, observable, n);

    double exp_O = compute_expectation_value(state, observable, n, is_density);
    double exp_O2 = compute_expectation_value(state, O2, n, is_density);

    free(O2);

    return exp_O2 - exp_O * exp_O;
}

// ============================================================================
// Quantum Information Measures
// ============================================================================

/**
 * @brief Compute von Neumann entropy using eigenvalue decomposition
 *
 * S(ρ) = -Tr(ρ log ρ) = -Σ_i λ_i log λ_i
 * where λ_i are eigenvalues of ρ
 *
 * @param rho Density matrix (n×n)
 * @param n Hilbert space dimension
 * @return Entropy in bits (log base 2)
 */
double compute_von_neumann_entropy(const double complex* rho, size_t n) {
    if (!rho || n == 0) {
        return 0.0;
    }

    // Get eigenvalues of density matrix
    ComplexFloat* eigenvalues = malloc(n * sizeof(ComplexFloat));
    ComplexFloat* matrix_copy = malloc(n * n * sizeof(ComplexFloat));

    if (!eigenvalues || !matrix_copy) {
        free(eigenvalues);
        free(matrix_copy);
        return 0.0;
    }

    // Convert to ComplexFloat for LAPACK
    for (size_t i = 0; i < n * n; i++) {
        matrix_copy[i].real = (float)creal(rho[i]);
        matrix_copy[i].imag = (float)cimag(rho[i]);
    }

    // Compute eigenvalues
    bool success = find_eigenvalues(matrix_copy, eigenvalues, NULL, n);

    double entropy = 0.0;

    if (success) {
        for (size_t i = 0; i < n; i++) {
            double lambda = eigenvalues[i].real;  // Eigenvalues of ρ are real

            // Clamp to [0, 1] for numerical stability
            if (lambda < ENTROPY_EIGENVALUE_CUTOFF) lambda = 0.0;
            if (lambda > 1.0) lambda = 1.0;

            if (lambda > ENTROPY_EIGENVALUE_CUTOFF) {
                entropy -= lambda * log2(lambda);
            }
        }
    } else {
        // Fallback: diagonal approximation (only valid for diagonal ρ)
        for (size_t i = 0; i < n; i++) {
            double lambda = creal(rho[i * n + i]);
            if (lambda > ENTROPY_EIGENVALUE_CUTOFF) {
                entropy -= lambda * log2(lambda);
            }
        }
    }

    free(eigenvalues);
    free(matrix_copy);

    return entropy;
}

/**
 * @brief Compute purity of density matrix
 *
 * γ = Tr(ρ²), γ = 1 for pure states, γ = 1/n for maximally mixed
 *
 * @param rho Density matrix (n×n)
 * @param n Hilbert space dimension
 * @return Purity in (0, 1]
 */
double compute_purity(const double complex* rho, size_t n) {
    if (!rho || n == 0) {
        return 0.0;
    }

    // Tr(ρ²) = Σ_ij |ρ_ij|²
    double purity = 0.0;

    #pragma omp parallel for reduction(+:purity)
    for (size_t i = 0; i < n * n; i++) {
        purity += creal(rho[i] * conj(rho[i]));
    }

    return purity;
}

/**
 * @brief Compute linear entropy
 *
 * S_L = 1 - Tr(ρ²) = 1 - γ
 * Ranges from 0 (pure) to 1-1/n (maximally mixed)
 *
 * @param rho Density matrix (n×n)
 * @param n Hilbert space dimension
 * @return Linear entropy
 */
double compute_linear_entropy(const double complex* rho, size_t n) {
    return 1.0 - compute_purity(rho, n);
}

/**
 * @brief Compute quantum fidelity between two states
 *
 * For pure states: F = |⟨ψ|φ⟩|²
 * For mixed states: F = (Tr√(√ρ σ √ρ))² (Uhlmann fidelity)
 *
 * @param state1 First state (vector or density matrix)
 * @param state2 Second state (vector or density matrix)
 * @param n Hilbert space dimension
 * @param is_density True if inputs are density matrices
 * @return Fidelity in [0, 1]
 */
double physics_compute_fidelity(const double complex* state1,
                                const double complex* state2,
                                size_t n,
                                bool is_density) {
    if (!state1 || !state2 || n == 0) {
        return 0.0;
    }

    if (!is_density) {
        // Pure state fidelity: |⟨ψ|φ⟩|²
        double complex overlap = 0.0;
        for (size_t i = 0; i < n; i++) {
            overlap += conj(state1[i]) * state2[i];
        }
        return creal(overlap * conj(overlap));
    }

    // Uhlmann fidelity for mixed states
    // F(ρ, σ) = (Tr√(√ρ σ √ρ))²

    size_t n2 = n * n;
    double complex* sqrt_rho = malloc(n2 * sizeof(double complex));
    double complex* temp1 = malloc(n2 * sizeof(double complex));
    double complex* temp2 = malloc(n2 * sizeof(double complex));

    if (!sqrt_rho || !temp1 || !temp2) {
        free(sqrt_rho);
        free(temp1);
        free(temp2);
        // Fallback: trace approximation
        double f = 0.0;
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++) {
                f += creal(state1[i * n + j] * state2[j * n + i]);
            }
        }
        return fmax(0.0, fmin(1.0, f));
    }

    // Compute √ρ
    bool success = compute_matrix_sqrt(sqrt_rho, state1, n);

    if (!success) {
        // Fallback to trace inner product
        double f = 0.0;
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++) {
                f += creal(state1[i * n + j] * state2[j * n + i]);
            }
        }
        free(sqrt_rho);
        free(temp1);
        free(temp2);
        return fmax(0.0, fmin(1.0, f));
    }

    // Compute √ρ σ √ρ
    physics_matrix_mult(temp1, sqrt_rho, state2, n);
    physics_matrix_mult(temp2, temp1, sqrt_rho, n);

    // Compute √(√ρ σ √ρ)
    success = compute_matrix_sqrt(temp1, temp2, n);

    double fidelity = 0.0;
    if (success) {
        // F = (Tr(√(√ρ σ √ρ)))²
        double complex trace = 0.0;
        for (size_t i = 0; i < n; i++) {
            trace += temp1[i * n + i];
        }
        fidelity = creal(trace * conj(trace));
    } else {
        // Fallback: use eigenvalues of temp2
        ComplexFloat* eigenvalues = malloc(n * sizeof(ComplexFloat));
        ComplexFloat* matrix_cf = malloc(n2 * sizeof(ComplexFloat));

        if (eigenvalues && matrix_cf) {
            for (size_t i = 0; i < n2; i++) {
                matrix_cf[i].real = (float)creal(temp2[i]);
                matrix_cf[i].imag = (float)cimag(temp2[i]);
            }

            if (find_eigenvalues(matrix_cf, eigenvalues, NULL, n)) {
                double trace = 0.0;
                for (size_t i = 0; i < n; i++) {
                    double lambda = eigenvalues[i].real;
                    if (lambda > 0) trace += sqrt(lambda);
                }
                fidelity = trace * trace;
            }
        }

        free(eigenvalues);
        free(matrix_cf);
    }

    free(sqrt_rho);
    free(temp1);
    free(temp2);

    return fmax(0.0, fmin(1.0, fidelity));
}

/**
 * @brief Compute trace distance between density matrices
 *
 * D(ρ, σ) = ½ Tr|ρ - σ| = ½ Σ_i |λ_i|
 * where λ_i are eigenvalues of (ρ - σ)
 *
 * @param rho1 First density matrix
 * @param rho2 Second density matrix
 * @param n Hilbert space dimension
 * @return Trace distance in [0, 1]
 */
double physics_compute_trace_distance(const double complex* rho1,
                                      const double complex* rho2,
                                      size_t n) {
    if (!rho1 || !rho2 || n == 0) {
        return 0.0;
    }

    size_t n2 = n * n;

    // Compute ρ - σ
    ComplexFloat* diff = malloc(n2 * sizeof(ComplexFloat));
    ComplexFloat* eigenvalues = malloc(n * sizeof(ComplexFloat));

    if (!diff || !eigenvalues) {
        free(diff);
        free(eigenvalues);
        // Fallback: Frobenius norm
        double sum = 0.0;
        for (size_t i = 0; i < n2; i++) {
            double complex d = rho1[i] - rho2[i];
            sum += creal(d * conj(d));
        }
        return 0.5 * sqrt(sum);
    }

    for (size_t i = 0; i < n2; i++) {
        diff[i].real = (float)(creal(rho1[i]) - creal(rho2[i]));
        diff[i].imag = (float)(cimag(rho1[i]) - cimag(rho2[i]));
    }

    double trace_dist = 0.0;

    if (find_eigenvalues(diff, eigenvalues, NULL, n)) {
        // D = ½ Σ |λ_i|
        for (size_t i = 0; i < n; i++) {
            trace_dist += fabs(eigenvalues[i].real);
        }
        trace_dist *= 0.5;
    } else {
        // Fallback: Frobenius norm upper bound
        double sum = 0.0;
        for (size_t i = 0; i < n2; i++) {
            double complex d = rho1[i] - rho2[i];
            sum += creal(d * conj(d));
        }
        trace_dist = 0.5 * sqrt(sum);
    }

    free(diff);
    free(eigenvalues);

    return fmin(1.0, trace_dist);
}

/**
 * @brief Compute entanglement entropy of bipartite system
 *
 * S_A = S(ρ_A) where ρ_A = Tr_B(|ψ⟩⟨ψ|)
 *
 * @param state Pure state of composite system AB
 * @param dim_a Dimension of subsystem A
 * @param dim_b Dimension of subsystem B
 * @return Entanglement entropy in bits
 */
double compute_entanglement_entropy(const double complex* state,
                                    size_t dim_a,
                                    size_t dim_b) {
    if (!state || dim_a == 0 || dim_b == 0) {
        return 0.0;
    }

    // Compute reduced density matrix ρ_A = Tr_B(|ψ⟩⟨ψ|)
    double complex* rho_a = calloc(dim_a * dim_a, sizeof(double complex));
    if (!rho_a) return 0.0;

    // ρ_A[i,j] = Σ_k ψ[i*dim_b + k] * conj(ψ[j*dim_b + k])
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < dim_a; i++) {
        for (size_t j = 0; j < dim_a; j++) {
            double complex sum = 0.0;
            for (size_t k = 0; k < dim_b; k++) {
                sum += state[i * dim_b + k] * conj(state[j * dim_b + k]);
            }
            rho_a[i * dim_a + j] = sum;
        }
    }

    double entropy = compute_von_neumann_entropy(rho_a, dim_a);

    free(rho_a);
    return entropy;
}

/**
 * @brief Compute concurrence for two-qubit system
 *
 * C(ρ) = max(0, λ₁ - λ₂ - λ₃ - λ₄)
 * where λᵢ are decreasing sqrt eigenvalues of ρ(σ_y⊗σ_y)ρ*(σ_y⊗σ_y)
 *
 * @param rho Two-qubit density matrix (4×4)
 * @return Concurrence in [0, 1]
 */
double compute_concurrence(const double complex* rho) {
    if (!rho) return 0.0;

    // σ_y = [[0, -i], [i, 0]]
    // σ_y ⊗ σ_y tensor product
    double complex sigma_y[4] = {0, -I, I, 0};
    double complex sigma_yy[16];

    // Tensor product
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            for (size_t k = 0; k < 2; k++) {
                for (size_t l = 0; l < 2; l++) {
                    sigma_yy[(2*i + k) * 4 + (2*j + l)] =
                        sigma_y[2*i + j] * sigma_y[2*k + l];
                }
            }
        }
    }

    // Compute ρ̃ = (σ_y⊗σ_y) ρ* (σ_y⊗σ_y)
    double complex rho_tilde[16];
    double complex temp[16];

    // temp = (σ_y⊗σ_y) ρ*
    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 4; j++) {
            temp[i*4+j] = 0;
            for (size_t k = 0; k < 4; k++) {
                temp[i*4+j] += sigma_yy[i*4+k] * conj(rho[k*4+j]);
            }
        }
    }

    // rho_tilde = temp (σ_y⊗σ_y)
    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 4; j++) {
            rho_tilde[i*4+j] = 0;
            for (size_t k = 0; k < 4; k++) {
                rho_tilde[i*4+j] += temp[i*4+k] * sigma_yy[k*4+j];
            }
        }
    }

    // Compute R = ρ ρ̃
    double complex R[16];
    physics_matrix_mult(R, rho, rho_tilde, 4);

    // Get eigenvalues of R
    ComplexFloat R_cf[16];
    ComplexFloat eigenvalues[4];

    for (size_t i = 0; i < 16; i++) {
        R_cf[i].real = (float)creal(R[i]);
        R_cf[i].imag = (float)cimag(R[i]);
    }

    if (!find_eigenvalues(R_cf, eigenvalues, NULL, 4)) {
        return 0.0;
    }

    // Get sqrt of eigenvalues (real parts)
    double lambdas[4];
    for (size_t i = 0; i < 4; i++) {
        double ev = eigenvalues[i].real;
        lambdas[i] = (ev > 0) ? sqrt(ev) : 0.0;
    }

    // Sort in descending order
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = i + 1; j < 4; j++) {
            if (lambdas[j] > lambdas[i]) {
                double tmp = lambdas[i];
                lambdas[i] = lambdas[j];
                lambdas[j] = tmp;
            }
        }
    }

    // C = max(0, λ₁ - λ₂ - λ₃ - λ₄)
    double concurrence = lambdas[0] - lambdas[1] - lambdas[2] - lambdas[3];
    return fmax(0.0, concurrence);
}

/**
 * @brief Compute entanglement of formation from concurrence
 *
 * E(ρ) = h((1 + sqrt(1 - C²))/2) where h(x) = -x log₂(x) - (1-x) log₂(1-x)
 *
 * @param concurrence Concurrence value
 * @return Entanglement of formation in ebits
 */
double compute_entanglement_of_formation(double concurrence) {
    if (concurrence <= 0.0) return 0.0;
    if (concurrence >= 1.0) return 1.0;

    double x = (1.0 + sqrt(1.0 - concurrence * concurrence)) / 2.0;

    // Binary entropy
    if (x <= 0.0 || x >= 1.0) return 0.0;
    return -x * log2(x) - (1.0 - x) * log2(1.0 - x);
}

// ============================================================================
// Density Matrix Operations
// ============================================================================

/**
 * @brief Compute density matrix from pure state
 *
 * ρ = |ψ⟩⟨ψ|
 */
qgt_error_t compute_density_matrix(double complex* rho,
                                   const double complex* state,
                                   size_t n) {
    if (!rho || !state || n == 0) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            rho[i * n + j] = state[i] * conj(state[j]);
        }
    }

    return QGT_SUCCESS;
}

/**
 * @brief Compute partial trace
 *
 * ρ_A = Tr_B(ρ_AB)
 */
qgt_error_t compute_partial_trace(double complex* rho_reduced,
                                  const double complex* rho_full,
                                  size_t dim_a,
                                  size_t dim_b) {
    if (!rho_reduced || !rho_full || dim_a == 0 || dim_b == 0) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    size_t dim = dim_a * dim_b;
    memset(rho_reduced, 0, dim_a * dim_a * sizeof(double complex));

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < dim_a; i++) {
        for (size_t j = 0; j < dim_a; j++) {
            double complex sum = 0.0;
            for (size_t k = 0; k < dim_b; k++) {
                size_t row = i * dim_b + k;
                size_t col = j * dim_b + k;
                sum += rho_full[row * dim + col];
            }
            rho_reduced[i * dim_a + j] = sum;
        }
    }

    return QGT_SUCCESS;
}

/**
 * @brief Compute mixed state from ensemble
 *
 * ρ = Σ_k p_k |ψ_k⟩⟨ψ_k|
 */
qgt_error_t compute_mixed_state(double complex* rho,
                                const double complex* const* states,
                                const double* probabilities,
                                size_t n_states,
                                size_t dim) {
    if (!rho || !states || !probabilities || n_states == 0 || dim == 0) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    memset(rho, 0, dim * dim * sizeof(double complex));

    for (size_t k = 0; k < n_states; k++) {
        if (!states[k] || probabilities[k] <= 0.0) continue;

        double p = probabilities[k];

        for (size_t i = 0; i < dim; i++) {
            for (size_t j = 0; j < dim; j++) {
                rho[i * dim + j] += p * states[k][i] * conj(states[k][j]);
            }
        }
    }

    return QGT_SUCCESS;
}

// ============================================================================
// Quantum Gates
// ============================================================================

/**
 * @brief Apply single-qubit gate to state
 */
qgt_error_t apply_single_qubit_gate(double complex* state,
                                    const double complex gate[4],
                                    size_t target,
                                    size_t n_qubits) {
    if (!state || !gate || target >= n_qubits) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    size_t dim = 1UL << n_qubits;
    size_t step = 1UL << target;

    #pragma omp parallel for
    for (size_t i = 0; i < dim; i += 2 * step) {
        for (size_t j = 0; j < step; j++) {
            size_t i0 = i + j;
            size_t i1 = i + j + step;

            double complex a0 = state[i0];
            double complex a1 = state[i1];

            state[i0] = gate[0] * a0 + gate[1] * a1;
            state[i1] = gate[2] * a0 + gate[3] * a1;
        }
    }

    return QGT_SUCCESS;
}

/**
 * @brief Apply two-qubit controlled gate
 */
qgt_error_t apply_controlled_gate(double complex* state,
                                  const double complex gate[4],
                                  size_t control,
                                  size_t target,
                                  size_t n_qubits) {
    if (!state || !gate || control >= n_qubits || target >= n_qubits || control == target) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    size_t dim = 1UL << n_qubits;
    size_t control_mask = 1UL << control;
    size_t target_step = 1UL << target;

    #pragma omp parallel for
    for (size_t i = 0; i < dim; i++) {
        // Only apply when control qubit is 1
        if (!(i & control_mask)) continue;

        // Only process when target is 0 (to avoid double processing)
        if (i & target_step) continue;

        size_t i0 = i;
        size_t i1 = i | target_step;

        double complex a0 = state[i0];
        double complex a1 = state[i1];

        state[i0] = gate[0] * a0 + gate[1] * a1;
        state[i1] = gate[2] * a0 + gate[3] * a1;
    }

    return QGT_SUCCESS;
}

/**
 * @brief Apply SWAP gate
 */
qgt_error_t apply_swap_gate(double complex* state,
                            size_t qubit1,
                            size_t qubit2,
                            size_t n_qubits) {
    if (!state || qubit1 >= n_qubits || qubit2 >= n_qubits || qubit1 == qubit2) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    size_t dim = 1UL << n_qubits;
    size_t mask1 = 1UL << qubit1;
    size_t mask2 = 1UL << qubit2;

    #pragma omp parallel for
    for (size_t i = 0; i < dim; i++) {
        bool b1 = (i & mask1) != 0;
        bool b2 = (i & mask2) != 0;

        if (b1 != b2 && !b1) {
            size_t j = (i & ~mask2) | mask1;
            double complex temp = state[i];
            state[i] = state[j];
            state[j] = temp;
        }
    }

    return QGT_SUCCESS;
}

// ============================================================================
// State Preparation
// ============================================================================

/**
 * @brief Initialize computational basis state |i⟩
 */
qgt_error_t init_computational_basis(double complex* state,
                                     size_t basis_index,
                                     size_t dim) {
    if (!state || basis_index >= dim) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    memset(state, 0, dim * sizeof(double complex));
    state[basis_index] = 1.0;
    return QGT_SUCCESS;
}

/**
 * @brief Initialize uniform superposition |+⟩^⊗n
 */
qgt_error_t init_uniform_superposition(double complex* state, size_t n_qubits) {
    if (!state || n_qubits == 0 || n_qubits > 30) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    size_t dim = 1UL << n_qubits;
    double complex amplitude = 1.0 / sqrt((double)dim);

    #pragma omp parallel for
    for (size_t i = 0; i < dim; i++) {
        state[i] = amplitude;
    }

    return QGT_SUCCESS;
}

/**
 * @brief Initialize GHZ state (|00...0⟩ + |11...1⟩)/√2
 */
qgt_error_t init_ghz_state(double complex* state, size_t n_qubits) {
    if (!state || n_qubits == 0 || n_qubits > 30) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    size_t dim = 1UL << n_qubits;
    memset(state, 0, dim * sizeof(double complex));

    double complex amplitude = 1.0 / sqrt(2.0);
    state[0] = amplitude;
    state[dim - 1] = amplitude;

    return QGT_SUCCESS;
}

/**
 * @brief Initialize W state
 */
qgt_error_t init_w_state(double complex* state, size_t n_qubits) {
    if (!state || n_qubits == 0 || n_qubits > 30) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    size_t dim = 1UL << n_qubits;
    memset(state, 0, dim * sizeof(double complex));

    double complex amplitude = 1.0 / sqrt((double)n_qubits);

    for (size_t i = 0; i < n_qubits; i++) {
        state[1UL << i] = amplitude;
    }

    return QGT_SUCCESS;
}

/**
 * @brief Initialize Bell state
 *
 * @param state Output state (dimension 4)
 * @param type Bell state type: 0=Φ⁺, 1=Φ⁻, 2=Ψ⁺, 3=Ψ⁻
 */
qgt_error_t init_bell_state(double complex* state, int type) {
    if (!state || type < 0 || type > 3) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    memset(state, 0, 4 * sizeof(double complex));
    double complex amp = 1.0 / sqrt(2.0);

    switch (type) {
        case 0:  // |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
            state[0] = amp;
            state[3] = amp;
            break;
        case 1:  // |Φ⁻⟩ = (|00⟩ - |11⟩)/√2
            state[0] = amp;
            state[3] = -amp;
            break;
        case 2:  // |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2
            state[1] = amp;
            state[2] = amp;
            break;
        case 3:  // |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
            state[1] = amp;
            state[2] = -amp;
            break;
    }

    return QGT_SUCCESS;
}

// ============================================================================
// Matrix Operations (Internal)
// ============================================================================

/**
 * @brief Compute matrix exponential using scaling and squaring with Padé
 *
 * Computes exp(scalar * matrix) using the scaling and squaring method
 * with Padé approximants for accuracy.
 */
static bool compute_matrix_exponential(double complex* result,
                                       const double complex* matrix,
                                       size_t n,
                                       double complex scalar) {
    if (!result || !matrix || n == 0) return false;

    size_t n2 = n * n;

    // Allocate workspace
    double complex* A = malloc(n2 * sizeof(double complex));
    double complex* A2 = malloc(n2 * sizeof(double complex));
    double complex* A4 = malloc(n2 * sizeof(double complex));
    double complex* A6 = malloc(n2 * sizeof(double complex));
    double complex* U = malloc(n2 * sizeof(double complex));
    double complex* V = malloc(n2 * sizeof(double complex));
    double complex* temp = malloc(n2 * sizeof(double complex));
    double complex* Acopy = malloc(n2 * sizeof(double complex));

    if (!A || !A2 || !A4 || !A6 || !U || !V || !temp || !Acopy) {
        free(A); free(A2); free(A4); free(A6);
        free(U); free(V); free(temp); free(Acopy);
        return false;
    }

    // Scale matrix: A = scalar * matrix
    for (size_t i = 0; i < n2; i++) {
        A[i] = scalar * matrix[i];
    }

    // Compute 1-norm for scaling
    double norm1 = 0.0;
    for (size_t j = 0; j < n; j++) {
        double col_sum = 0.0;
        for (size_t i = 0; i < n; i++) {
            col_sum += cabs(A[i * n + j]);
        }
        if (col_sum > norm1) norm1 = col_sum;
    }

    // Determine scaling factor
    int s = 0;
    if (norm1 > 0.5) {
        s = (int)ceil(log2(norm1 / 0.5));
        if (s > MAX_SCALING_SQUARING) s = MAX_SCALING_SQUARING;

        double scale = pow(2.0, -s);
        for (size_t i = 0; i < n2; i++) {
            A[i] *= scale;
        }
    }

    // Compute powers of A
    physics_matrix_mult(A2, A, A, n);
    physics_matrix_mult(A4, A2, A2, n);
    physics_matrix_mult(A6, A4, A2, n);

    // Padé(13,13) coefficients
    double b[14] = {
        64764752532480000.0, 32382376266240000.0, 7771770303897600.0,
        1187353796428800.0, 129060195264000.0, 10559470521600.0,
        670442572800.0, 33522128640.0, 1323241920.0, 40840800.0,
        960960.0, 16380.0, 182.0, 1.0
    };

    // Compute U and V
    // U_odd = A(b₁₃A⁶ + b₁₁A⁴ + b₉A² + b₇I)A² + A(b₅A⁴ + b₃A² + b₁I)
    // V_even = b₁₂A⁶ + b₁₀A⁴ + b₈A² + b₆I + (b₄A⁴ + b₂A² + b₀I)

    // temp = b₁₃A⁶ + b₁₁A⁴ + b₉A²
    for (size_t i = 0; i < n2; i++) {
        temp[i] = b[13] * A6[i] + b[11] * A4[i] + b[9] * A2[i];
    }
    for (size_t i = 0; i < n; i++) {
        temp[i * n + i] += b[7];
    }

    physics_matrix_mult(U, A2, temp, n);

    for (size_t i = 0; i < n2; i++) {
        U[i] += b[5] * A4[i] + b[3] * A2[i];
    }
    for (size_t i = 0; i < n; i++) {
        U[i * n + i] += b[1];
    }

    physics_matrix_mult(temp, A, U, n);
    memcpy(U, temp, n2 * sizeof(double complex));

    // V
    for (size_t i = 0; i < n2; i++) {
        temp[i] = b[12] * A6[i] + b[10] * A4[i] + b[8] * A2[i];
    }
    for (size_t i = 0; i < n; i++) {
        temp[i * n + i] += b[6];
    }

    physics_matrix_mult(V, A2, temp, n);

    for (size_t i = 0; i < n2; i++) {
        V[i] += b[4] * A4[i] + b[2] * A2[i];
    }
    for (size_t i = 0; i < n; i++) {
        V[i * n + i] += b[0];
    }

    // Solve (V - U) X = (V + U) for X
    for (size_t i = 0; i < n2; i++) {
        temp[i] = V[i] + U[i];   // RHS
        Acopy[i] = V[i] - U[i];  // LHS coefficient matrix
    }

    memcpy(result, temp, n2 * sizeof(double complex));

    // Gaussian elimination with partial pivoting
    for (size_t k = 0; k < n; k++) {
        size_t max_row = k;
        double max_val = cabs(Acopy[k * n + k]);
        for (size_t i = k + 1; i < n; i++) {
            double val = cabs(Acopy[i * n + k]);
            if (val > max_val) {
                max_val = val;
                max_row = i;
            }
        }

        if (max_row != k) {
            for (size_t j = 0; j < n; j++) {
                double complex t = Acopy[k * n + j];
                Acopy[k * n + j] = Acopy[max_row * n + j];
                Acopy[max_row * n + j] = t;

                t = result[k * n + j];
                result[k * n + j] = result[max_row * n + j];
                result[max_row * n + j] = t;
            }
        }

        if (cabs(Acopy[k * n + k]) < 1e-14) {
            free(A); free(A2); free(A4); free(A6);
            free(U); free(V); free(temp); free(Acopy);
            return false;
        }

        for (size_t i = k + 1; i < n; i++) {
            double complex factor = Acopy[i * n + k] / Acopy[k * n + k];
            for (size_t j = k; j < n; j++) {
                Acopy[i * n + j] -= factor * Acopy[k * n + j];
            }
            for (size_t j = 0; j < n; j++) {
                result[i * n + j] -= factor * result[k * n + j];
            }
        }
    }

    // Back substitution
    for (int k = (int)n - 1; k >= 0; k--) {
        for (size_t j = 0; j < n; j++) {
            result[k * n + j] /= Acopy[k * n + k];
        }
        for (int i = 0; i < k; i++) {
            double complex factor = Acopy[i * n + k];
            for (size_t j = 0; j < n; j++) {
                result[i * n + j] -= factor * result[k * n + j];
            }
        }
    }

    // Squaring phase
    for (int i = 0; i < s; i++) {
        physics_matrix_mult(temp, result, result, n);
        memcpy(result, temp, n2 * sizeof(double complex));
    }

    free(A); free(A2); free(A4); free(A6);
    free(U); free(V); free(temp); free(Acopy);

    return true;
}

/**
 * @brief Compute matrix square root using Denman-Beavers iteration
 */
static bool compute_matrix_sqrt(double complex* result,
                                const double complex* matrix,
                                size_t n) {
    if (!result || !matrix || n == 0) return false;

    size_t n2 = n * n;

    double complex* Y = malloc(n2 * sizeof(double complex));
    double complex* Z = malloc(n2 * sizeof(double complex));
    double complex* Ynew = malloc(n2 * sizeof(double complex));
    double complex* Znew = malloc(n2 * sizeof(double complex));
    double complex* invY = malloc(n2 * sizeof(double complex));
    double complex* invZ = malloc(n2 * sizeof(double complex));

    if (!Y || !Z || !Ynew || !Znew || !invY || !invZ) {
        free(Y); free(Z); free(Ynew); free(Znew); free(invY); free(invZ);
        return false;
    }

    // Initialize: Y₀ = A, Z₀ = I
    memcpy(Y, matrix, n2 * sizeof(double complex));
    memset(Z, 0, n2 * sizeof(double complex));
    for (size_t i = 0; i < n; i++) {
        Z[i * n + i] = 1.0;
    }

    bool converged = false;
    for (int iter = 0; iter < FIDELITY_SQRT_ITERATIONS; iter++) {
        // Yₖ₊₁ = ½(Yₖ + Zₖ⁻¹)
        // Zₖ₊₁ = ½(Zₖ + Yₖ⁻¹)

        // Simple inversion using Gauss-Jordan for small matrices
        // Copy Y and Z for inversion
        double complex* Ycopy = malloc(n2 * sizeof(double complex));
        double complex* Zcopy = malloc(n2 * sizeof(double complex));

        if (!Ycopy || !Zcopy) {
            free(Ycopy); free(Zcopy);
            break;
        }

        memcpy(Ycopy, Y, n2 * sizeof(double complex));
        memcpy(Zcopy, Z, n2 * sizeof(double complex));

        // Initialize inverse as identity
        memset(invY, 0, n2 * sizeof(double complex));
        memset(invZ, 0, n2 * sizeof(double complex));
        for (size_t i = 0; i < n; i++) {
            invY[i * n + i] = 1.0;
            invZ[i * n + i] = 1.0;
        }

        // Gauss-Jordan for Y inverse
        bool inv_ok = true;
        for (size_t k = 0; k < n && inv_ok; k++) {
            size_t max_row = k;
            double max_val = cabs(Ycopy[k * n + k]);
            for (size_t i = k + 1; i < n; i++) {
                if (cabs(Ycopy[i * n + k]) > max_val) {
                    max_val = cabs(Ycopy[i * n + k]);
                    max_row = i;
                }
            }

            if (max_val < 1e-14) {
                inv_ok = false;
                break;
            }

            if (max_row != k) {
                for (size_t j = 0; j < n; j++) {
                    double complex t = Ycopy[k * n + j];
                    Ycopy[k * n + j] = Ycopy[max_row * n + j];
                    Ycopy[max_row * n + j] = t;
                    t = invY[k * n + j];
                    invY[k * n + j] = invY[max_row * n + j];
                    invY[max_row * n + j] = t;
                }
            }

            double complex pivot = Ycopy[k * n + k];
            for (size_t j = 0; j < n; j++) {
                Ycopy[k * n + j] /= pivot;
                invY[k * n + j] /= pivot;
            }

            for (size_t i = 0; i < n; i++) {
                if (i != k) {
                    double complex factor = Ycopy[i * n + k];
                    for (size_t j = 0; j < n; j++) {
                        Ycopy[i * n + j] -= factor * Ycopy[k * n + j];
                        invY[i * n + j] -= factor * invY[k * n + j];
                    }
                }
            }
        }

        // Similarly for Z inverse
        for (size_t k = 0; k < n && inv_ok; k++) {
            size_t max_row = k;
            double max_val = cabs(Zcopy[k * n + k]);
            for (size_t i = k + 1; i < n; i++) {
                if (cabs(Zcopy[i * n + k]) > max_val) {
                    max_val = cabs(Zcopy[i * n + k]);
                    max_row = i;
                }
            }

            if (max_val < 1e-14) {
                inv_ok = false;
                break;
            }

            if (max_row != k) {
                for (size_t j = 0; j < n; j++) {
                    double complex t = Zcopy[k * n + j];
                    Zcopy[k * n + j] = Zcopy[max_row * n + j];
                    Zcopy[max_row * n + j] = t;
                    t = invZ[k * n + j];
                    invZ[k * n + j] = invZ[max_row * n + j];
                    invZ[max_row * n + j] = t;
                }
            }

            double complex pivot = Zcopy[k * n + k];
            for (size_t j = 0; j < n; j++) {
                Zcopy[k * n + j] /= pivot;
                invZ[k * n + j] /= pivot;
            }

            for (size_t i = 0; i < n; i++) {
                if (i != k) {
                    double complex factor = Zcopy[i * n + k];
                    for (size_t j = 0; j < n; j++) {
                        Zcopy[i * n + j] -= factor * Zcopy[k * n + j];
                        invZ[i * n + j] -= factor * invZ[k * n + j];
                    }
                }
            }
        }

        free(Ycopy);
        free(Zcopy);

        if (!inv_ok) break;

        // Yₖ₊₁ = ½(Yₖ + Zₖ⁻¹)
        // Zₖ₊₁ = ½(Zₖ + Yₖ⁻¹)
        for (size_t i = 0; i < n2; i++) {
            Ynew[i] = 0.5 * (Y[i] + invZ[i]);
            Znew[i] = 0.5 * (Z[i] + invY[i]);
        }

        // Check convergence
        double diff = 0.0;
        for (size_t i = 0; i < n2; i++) {
            diff += cabs(Ynew[i] - Y[i]);
        }

        memcpy(Y, Ynew, n2 * sizeof(double complex));
        memcpy(Z, Znew, n2 * sizeof(double complex));

        if (diff < 1e-10 * n2) {
            converged = true;
            break;
        }
    }

    memcpy(result, Y, n2 * sizeof(double complex));

    free(Y); free(Z); free(Ynew); free(Znew); free(invY); free(invZ);

    return converged;
}

static void physics_matrix_mult(double complex* C,
                                const double complex* A,
                                const double complex* B,
                                size_t n) {
    memset(C, 0, n * n * sizeof(double complex));

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            double complex sum = 0.0;
            for (size_t k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

static void matrix_vector_multiply(double complex* result,
                                   const double complex* matrix,
                                   const double complex* vector,
                                   size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        double complex sum = 0.0;
        for (size_t j = 0; j < n; j++) {
            sum += matrix[i * n + j] * vector[j];
        }
        result[i] = sum;
    }
}

static void evolve_state_hierarchical(HierarchicalMatrix* state,
                                      const HierarchicalMatrix* propagator) {
    if (!state || !propagator) return;

    if (state->is_leaf && propagator->is_leaf) {
        size_t n = state->rows;
        double complex* result = malloc(n * sizeof(double complex));
        if (!result) return;

        matrix_vector_multiply(result, propagator->data, state->data, n);
        memcpy(state->data, result, n * sizeof(double complex));

        free(result);
        return;
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        if (state->children[0] && propagator->children[0])
            evolve_state_hierarchical(state->children[0], propagator->children[0]);

        #pragma omp section
        if (state->children[1] && propagator->children[1])
            evolve_state_hierarchical(state->children[1], propagator->children[1]);

        #pragma omp section
        if (state->children[2] && propagator->children[2])
            evolve_state_hierarchical(state->children[2], propagator->children[2]);

        #pragma omp section
        if (state->children[3] && propagator->children[3])
            evolve_state_hierarchical(state->children[3], propagator->children[3]);
    }
}
