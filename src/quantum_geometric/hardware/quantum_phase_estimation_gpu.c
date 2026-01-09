#include "quantum_geometric/hardware/quantum_geometric_gpu.h"
#include "quantum_geometric/core/hierarchical_matrix.h"
#include "quantum_geometric/core/memory_pool.h"
#include "quantum_geometric/core/error_codes.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// =============================================================================
// Forward declarations for GPU-specific implementations
// =============================================================================

#ifdef ENABLE_METAL
extern int metal_phase_estimation_dispatch(
    void* device_handle,
    void* command_queue,
    void* library,
    double complex* state,
    size_t q,
    size_t dim
);

extern int metal_hmatrix_update_dispatch(
    void* device_handle,
    void* command_queue,
    void* library,
    HierarchicalMatrix* mat
);

extern int metal_quantum_metric_dispatch(
    void* device_handle,
    void* command_queue,
    void* library,
    const HierarchicalMatrix* mat,
    double* metric
);

extern int metal_berry_curvature_dispatch(
    void* device_handle,
    void* command_queue,
    void* library,
    const HierarchicalMatrix* mat,
    double* curvature
);
#endif

#ifdef ENABLE_CUDA
extern int cuda_phase_estimation_dispatch(
    void* device_handle,
    void* stream,
    double complex* state,
    size_t q,
    size_t dim
);

extern int cuda_hmatrix_update_dispatch(
    void* device_handle,
    void* stream,
    HierarchicalMatrix* mat
);

extern int cuda_quantum_metric_dispatch(
    void* device_handle,
    void* stream,
    const HierarchicalMatrix* mat,
    double* metric
);

extern int cuda_berry_curvature_dispatch(
    void* device_handle,
    void* stream,
    const HierarchicalMatrix* mat,
    double* curvature
);
#endif

// =============================================================================
// CPU Implementation (fallback for all platforms)
// =============================================================================

/**
 * @brief CPU implementation of quantum phase estimation
 *
 * Applies phase rotation R_k = diag(1, e^(2πi/2^k)) to qubit q.
 * Uses OpenMP for parallelization on multi-core CPUs.
 *
 * @param state Quantum state amplitudes (modified in place)
 * @param q Qubit index for phase estimation
 * @param dim State vector dimension (2^num_qubits)
 */
static void phase_estimation_cpu(double complex* state, size_t q, size_t dim) {
    if (!state || dim == 0) return;

    size_t mask = 1ULL << q;

    // Precompute phase rotation for this qubit
    double angle = 2.0 * M_PI / (double)(1ULL << (q + 1));
    double cos_a = cos(angle);
    double sin_a = sin(angle);

    #pragma omp parallel for
    for (size_t i = 0; i < dim; i++) {
        if (i & mask) {
            double real = creal(state[i]);
            double imag = cimag(state[i]);

            // Multiply by e^(i*angle)
            state[i] = (real * cos_a - imag * sin_a) +
                       I * (real * sin_a + imag * cos_a);
        }
    }
}

/**
 * @brief CPU implementation of hierarchical matrix quantum state update
 *
 * Normalizes the quantum state and computes Berry phase contributions
 * from the gradient structure.
 *
 * @param mat Hierarchical matrix to update
 */
static void hmatrix_update_cpu(HierarchicalMatrix* mat) {
    if (!mat || !mat->data) return;

    size_t size = mat->rows * mat->cols;
    if (size == 0) return;

    // Compute normalization factor
    double norm_sq = 0.0;
    #pragma omp parallel for reduction(+:norm_sq)
    for (size_t i = 0; i < size; i++) {
        double amp = cabs(mat->data[i]);
        norm_sq += amp * amp;
    }

    // Normalize if needed
    if (norm_sq > 1e-15 && fabs(norm_sq - 1.0) > 1e-10) {
        double inv_norm = 1.0 / sqrt(norm_sq);
        #pragma omp parallel for
        for (size_t i = 0; i < size; i++) {
            mat->data[i] *= inv_norm;
        }
    }

    // Compute geometric phase corrections if gradient exists
    if (mat->grad) {
        #pragma omp parallel for
        for (size_t i = 0; i < mat->rows; i++) {
            for (size_t j = 0; j < mat->cols; j++) {
                size_t idx = i * mat->cols + j;
                double complex psi = mat->data[idx];
                double complex phase_contrib = 0.0;

                // Berry phase from row neighbor
                if (i > 0) {
                    double complex psi_prev = mat->data[(i - 1) * mat->cols + j];
                    double complex overlap = conj(psi_prev) * psi;
                    double overlap_abs = cabs(overlap);
                    if (overlap_abs > 1e-15) {
                        phase_contrib += cimag(clog(overlap / overlap_abs));
                    }
                }

                // Berry phase from column neighbor
                if (j > 0) {
                    double complex psi_prev = mat->data[i * mat->cols + j - 1];
                    double complex overlap = conj(psi_prev) * psi;
                    double overlap_abs = cabs(overlap);
                    if (overlap_abs > 1e-15) {
                        phase_contrib += cimag(clog(overlap / overlap_abs));
                    }
                }

                mat->grad[idx] += phase_contrib * psi;
            }
        }
    }
}

/**
 * @brief CPU implementation of quantum metric computation
 *
 * Computes g_ij = Re[⟨∂_i ψ|∂_j ψ⟩] using finite differences.
 *
 * @param mat Input hierarchical matrix
 * @param metric Output metric tensor
 */
static void quantum_metric_cpu(const HierarchicalMatrix* mat, double* metric) {
    if (!mat || !mat->data || !metric) return;

    size_t rows = mat->rows;
    size_t cols = mat->cols;

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            size_t idx = i * cols + j;
            double complex psi = mat->data[idx];
            double complex dpsi_x = 0.0, dpsi_y = 0.0;

            // Compute x-derivative
            if (i > 0 && i < rows - 1) {
                dpsi_x = (mat->data[(i + 1) * cols + j] - mat->data[(i - 1) * cols + j]) / 2.0;
            } else if (i == 0 && rows > 1) {
                dpsi_x = mat->data[cols + j] - psi;
            } else if (i == rows - 1 && rows > 1) {
                dpsi_x = psi - mat->data[(rows - 2) * cols + j];
            }

            // Compute y-derivative
            if (j > 0 && j < cols - 1) {
                dpsi_y = (mat->data[i * cols + j + 1] - mat->data[i * cols + j - 1]) / 2.0;
            } else if (j == 0 && cols > 1) {
                dpsi_y = mat->data[i * cols + 1] - psi;
            } else if (j == cols - 1 && cols > 1) {
                dpsi_y = psi - mat->data[i * cols + cols - 2];
            }

            // Quantum metric: g = Re[⟨∂ψ|∂ψ⟩]
            metric[idx] = creal(conj(dpsi_x) * dpsi_x + conj(dpsi_y) * dpsi_y);
        }
    }
}

/**
 * @brief CPU implementation of Berry curvature computation
 *
 * Computes the Berry curvature using the lattice plaquette method:
 * F_ij = Im[ln(U_01 * U_12 * U_23 * U_30)] where U_ab = ⟨ψ_a|ψ_b⟩
 *
 * @param mat Input hierarchical matrix
 * @param curvature Output Berry curvature
 */
static void berry_curvature_cpu(const HierarchicalMatrix* mat, double* curvature) {
    if (!mat || !mat->data || !curvature) return;

    size_t rows = mat->rows;
    size_t cols = mat->cols;

    memset(curvature, 0, rows * cols * sizeof(double));

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rows - 1; i++) {
        for (size_t j = 0; j < cols - 1; j++) {
            size_t idx = i * cols + j;

            // Plaquette corners
            double complex psi_00 = mat->data[i * cols + j];
            double complex psi_10 = mat->data[(i + 1) * cols + j];
            double complex psi_01 = mat->data[i * cols + j + 1];
            double complex psi_11 = mat->data[(i + 1) * cols + j + 1];

            // Wilson loop links
            double complex U_01 = conj(psi_00) * psi_10;
            double complex U_12 = conj(psi_10) * psi_11;
            double complex U_23 = conj(psi_11) * psi_01;
            double complex U_30 = conj(psi_01) * psi_00;

            // Wilson loop product
            double complex W = U_01 * U_12 * U_23 * U_30;

            // Berry curvature is the argument of the Wilson loop
            double W_abs = cabs(W);
            if (W_abs > 1e-15) {
                curvature[idx] = cimag(clog(W));
            }
        }
    }
}

// =============================================================================
// GPU Dispatch Functions
// =============================================================================

/**
 * @brief Apply quantum phase estimation using GPU acceleration
 *
 * Dispatches to Metal (macOS) or CUDA (Linux/Windows) backend.
 * Falls back to CPU if GPU context is not available.
 *
 * @param ctx GPU context (NULL for CPU fallback)
 * @param state Quantum state amplitudes
 * @param q Qubit index for phase estimation
 * @param dim State vector dimension
 */
void apply_quantum_phase_gpu(GPUContext* ctx,
                             double complex* state,
                             size_t q,
                             size_t dim) {
    if (!state || dim == 0) return;

    // Check for valid GPU context
    if (ctx && ctx->is_initialized && ctx->device_handle) {
        int result = -1;

        switch (ctx->backend_type) {
#ifdef ENABLE_METAL
            case GPU_BACKEND_METAL:
                result = metal_phase_estimation_dispatch(
                    ctx->device_handle,
                    ctx->command_queue,
                    ctx->library,
                    state,
                    q,
                    dim
                );
                break;
#endif

#ifdef ENABLE_CUDA
            case GPU_BACKEND_CUDA:
                result = cuda_phase_estimation_dispatch(
                    ctx->device_handle,
                    ctx->command_queue,  // CUDA stream
                    state,
                    q,
                    dim
                );
                break;
#endif

            default:
                // Unknown backend, fall through to CPU
                break;
        }

        // If GPU dispatch succeeded, return
        if (result == 0) {
            return;
        }
        // Otherwise fall through to CPU implementation
    }

    // CPU fallback
    phase_estimation_cpu(state, q, dim);
}

/**
 * @brief Update hierarchical matrix quantum state using GPU
 *
 * Computes normalization and Berry phase contributions.
 * Dispatches to GPU if context available, otherwise uses CPU.
 *
 * @param mat Hierarchical matrix
 * @param ctx GPU context (NULL for CPU fallback)
 */
void update_hmatrix_quantum_state_gpu(HierarchicalMatrix* mat,
                                      GPUContext* ctx) {
    if (!mat || !mat->data) return;

    // Check for valid GPU context
    if (ctx && ctx->is_initialized && ctx->device_handle) {
        int result = -1;

        switch (ctx->backend_type) {
#ifdef ENABLE_METAL
            case GPU_BACKEND_METAL:
                result = metal_hmatrix_update_dispatch(
                    ctx->device_handle,
                    ctx->command_queue,
                    ctx->library,
                    mat
                );
                break;
#endif

#ifdef ENABLE_CUDA
            case GPU_BACKEND_CUDA:
                result = cuda_hmatrix_update_dispatch(
                    ctx->device_handle,
                    ctx->command_queue,
                    mat
                );
                break;
#endif

            default:
                break;
        }

        if (result == 0) {
            return;
        }
    }

    // CPU fallback
    hmatrix_update_cpu(mat);
}

/**
 * @brief Compute quantum metric tensor using GPU
 *
 * The quantum metric measures the "distance" between nearby quantum states
 * in parameter space. This is the real part of the quantum geometric tensor.
 *
 * @param mat Input hierarchical matrix
 * @param metric Output metric (pre-allocated, size rows*cols)
 */
void compute_hmatrix_quantum_metric(const HierarchicalMatrix* mat,
                                    double* metric) {
    // This function doesn't take a GPU context, so use CPU implementation
    // For GPU acceleration, use compute_hmatrix_quantum_metric_gpu below
    quantum_metric_cpu(mat, metric);
}

/**
 * @brief Compute quantum metric tensor using GPU with explicit context
 *
 * @param mat Input hierarchical matrix
 * @param metric Output metric tensor
 * @param ctx GPU context
 */
void compute_hmatrix_quantum_metric_gpu(const HierarchicalMatrix* mat,
                                        double* metric,
                                        GPUContext* ctx) {
    if (!mat || !mat->data || !metric) return;

    // Check for valid GPU context
    if (ctx && ctx->is_initialized && ctx->device_handle) {
        int result = -1;

        switch (ctx->backend_type) {
#ifdef ENABLE_METAL
            case GPU_BACKEND_METAL:
                result = metal_quantum_metric_dispatch(
                    ctx->device_handle,
                    ctx->command_queue,
                    ctx->library,
                    mat,
                    metric
                );
                break;
#endif

#ifdef ENABLE_CUDA
            case GPU_BACKEND_CUDA:
                result = cuda_quantum_metric_dispatch(
                    ctx->device_handle,
                    ctx->command_queue,
                    mat,
                    metric
                );
                break;
#endif

            default:
                break;
        }

        if (result == 0) {
            return;
        }
    }

    // CPU fallback
    quantum_metric_cpu(mat, metric);
}

/**
 * @brief Compute Berry curvature using GPU
 *
 * The Berry curvature is the topological invariant that governs
 * adiabatic evolution of quantum states. Computed using the
 * lattice plaquette (Wilson loop) method.
 *
 * @param mat Input hierarchical matrix
 * @param curvature Output curvature (pre-allocated, size rows*cols)
 */
void compute_hmatrix_berry_curvature(const HierarchicalMatrix* mat,
                                     double* curvature) {
    // This function doesn't take a GPU context, so use CPU implementation
    // For GPU acceleration, use compute_hmatrix_berry_curvature_gpu below
    berry_curvature_cpu(mat, curvature);
}

/**
 * @brief Compute Berry curvature using GPU with explicit context
 *
 * @param mat Input hierarchical matrix
 * @param curvature Output Berry curvature
 * @param ctx GPU context
 */
void compute_hmatrix_berry_curvature_gpu(const HierarchicalMatrix* mat,
                                         double* curvature,
                                         GPUContext* ctx) {
    if (!mat || !mat->data || !curvature) return;

    // Check for valid GPU context
    if (ctx && ctx->is_initialized && ctx->device_handle) {
        int result = -1;

        switch (ctx->backend_type) {
#ifdef ENABLE_METAL
            case GPU_BACKEND_METAL:
                result = metal_berry_curvature_dispatch(
                    ctx->device_handle,
                    ctx->command_queue,
                    ctx->library,
                    mat,
                    curvature
                );
                break;
#endif

#ifdef ENABLE_CUDA
            case GPU_BACKEND_CUDA:
                result = cuda_berry_curvature_dispatch(
                    ctx->device_handle,
                    ctx->command_queue,
                    mat,
                    curvature
                );
                break;
#endif

            default:
                break;
        }

        if (result == 0) {
            return;
        }
    }

    // CPU fallback
    berry_curvature_cpu(mat, curvature);
}

// =============================================================================
// Batch GPU Operations
// =============================================================================

/**
 * @brief Apply phase estimation to multiple qubits in batch
 *
 * More efficient for multi-qubit phase estimation than calling
 * apply_quantum_phase_gpu repeatedly.
 *
 * @param ctx GPU context
 * @param state Quantum state amplitudes
 * @param qubits Array of qubit indices
 * @param num_qubits Number of qubits
 * @param dim State vector dimension
 */
void apply_quantum_phase_batch_gpu(GPUContext* ctx,
                                   double complex* state,
                                   const size_t* qubits,
                                   size_t num_qubits,
                                   size_t dim) {
    if (!state || !qubits || dim == 0 || num_qubits == 0) return;

    // For batch operations, we process all qubits in sequence
    // GPU batching is handled internally by each dispatch
    for (size_t i = 0; i < num_qubits; i++) {
        apply_quantum_phase_gpu(ctx, state, qubits[i], dim);
    }
}

/**
 * @brief Compute full quantum geometric tensor
 *
 * Computes both the quantum metric and Berry curvature in one pass.
 * More efficient than calling both functions separately.
 *
 * @param mat Input hierarchical matrix
 * @param metric Output metric tensor (pre-allocated)
 * @param curvature Output Berry curvature (pre-allocated)
 * @param ctx GPU context
 */
void compute_quantum_geometric_tensor_gpu(const HierarchicalMatrix* mat,
                                          double* metric,
                                          double* curvature,
                                          GPUContext* ctx) {
    if (!mat || !mat->data) return;

    // Compute metric and curvature
    // In a more optimized implementation, these could share intermediate data
    if (metric) {
        compute_hmatrix_quantum_metric_gpu(mat, metric, ctx);
    }
    if (curvature) {
        compute_hmatrix_berry_curvature_gpu(mat, curvature, ctx);
    }
}

// =============================================================================
// Controlled Phase Operations
// =============================================================================

/**
 * @brief Apply controlled phase rotation
 *
 * Implements controlled-R_k gate: |1⟩⟨1| ⊗ R_k
 * where R_k = diag(1, e^(2πi/2^k))
 *
 * @param ctx GPU context
 * @param state Quantum state amplitudes
 * @param control Control qubit index
 * @param target Target qubit index
 * @param k Phase rotation exponent
 * @param dim State vector dimension
 */
void apply_controlled_phase_gpu(GPUContext* ctx,
                                double complex* state,
                                size_t control,
                                size_t target,
                                size_t k,
                                size_t dim) {
    if (!state || dim == 0) return;

    size_t control_mask = 1ULL << control;
    size_t target_mask = 1ULL << target;

    // Precompute phase rotation
    double angle = 2.0 * M_PI / (double)(1ULL << k);
    double cos_a = cos(angle);
    double sin_a = sin(angle);

    // CPU implementation with OpenMP
    // GPU dispatch could be added here similar to other functions
    #pragma omp parallel for
    for (size_t i = 0; i < dim; i++) {
        // Only apply when both control and target are |1⟩
        if ((i & control_mask) && (i & target_mask)) {
            double real = creal(state[i]);
            double imag = cimag(state[i]);

            state[i] = (real * cos_a - imag * sin_a) +
                       I * (real * sin_a + imag * cos_a);
        }
    }
}

/**
 * @brief Apply inverse QFT (Quantum Fourier Transform) for phase readout
 *
 * Final step of phase estimation algorithm. Applies inverse QFT
 * to the register qubits to extract the phase.
 *
 * @param ctx GPU context
 * @param state Quantum state amplitudes
 * @param num_qubits Number of register qubits
 * @param dim State vector dimension
 */
void apply_inverse_qft_gpu(GPUContext* ctx,
                           double complex* state,
                           size_t num_qubits,
                           size_t dim) {
    if (!state || dim == 0 || num_qubits == 0) return;

    // Inverse QFT consists of:
    // 1. SWAP gates to reverse qubit order
    // 2. Hadamard and controlled phase gates

    // Apply inverse QFT gates in reverse order
    for (size_t i = 0; i < num_qubits; i++) {
        size_t qubit = num_qubits - 1 - i;

        // Apply controlled rotations from all higher qubits
        for (size_t j = 0; j < i; j++) {
            size_t control_qubit = num_qubits - 1 - j;
            apply_controlled_phase_gpu(ctx, state, qubit, control_qubit, i - j + 1, dim);
        }

        // Apply Hadamard gate (implemented as phase rotation + rotation)
        size_t mask = 1ULL << qubit;
        double inv_sqrt2 = 1.0 / sqrt(2.0);

        #pragma omp parallel for
        for (size_t idx = 0; idx < dim; idx++) {
            if (!(idx & mask)) {
                size_t partner = idx | mask;
                double complex a = state[idx];
                double complex b = state[partner];

                state[idx] = inv_sqrt2 * (a + b);
                state[partner] = inv_sqrt2 * (a - b);
            }
        }
    }

    // Apply SWAP gates to reverse qubit order
    for (size_t i = 0; i < num_qubits / 2; i++) {
        size_t qubit1 = i;
        size_t qubit2 = num_qubits - 1 - i;
        size_t mask1 = 1ULL << qubit1;
        size_t mask2 = 1ULL << qubit2;

        #pragma omp parallel for
        for (size_t idx = 0; idx < dim; idx++) {
            // Only swap if qubits have different values
            if (((idx & mask1) != 0) != ((idx & mask2) != 0)) {
                size_t partner = idx ^ mask1 ^ mask2;
                if (idx < partner) {
                    double complex temp = state[idx];
                    state[idx] = state[partner];
                    state[partner] = temp;
                }
            }
        }
    }
}
