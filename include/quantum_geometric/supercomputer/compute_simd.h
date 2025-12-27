#ifndef COMPUTE_SIMD_H
#define COMPUTE_SIMD_H

#include "compute_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// SIMD Abstraction Layer
// ============================================================================
//
// This module provides cross-platform SIMD operations for quantum computing.
// Implementations are provided for:
// - x86_64: AVX2 + FMA
// - ARM64: NEON
// - Fallback: Scalar operations
//
// All complex numbers are stored as interleaved real/imaginary pairs.
// ============================================================================

// ============================================================================
// Vector Operations
// ============================================================================

/**
 * Compute the L2 norm of a vector.
 * @param data Input vector (real values)
 * @param n Number of elements
 * @return L2 norm
 */
float simd_vector_norm(const float* data, size_t n);

/**
 * Compute the L2 norm of a complex vector.
 * @param data Input vector (interleaved real/imag)
 * @param n Number of complex elements
 * @return L2 norm (real value)
 */
float simd_complex_norm_float(const float* data, size_t n);

/**
 * Scale a vector by a scalar.
 * @param data Input/output vector
 * @param n Number of elements
 * @param scale Scale factor
 */
void simd_vector_scale(float* data, size_t n, float scale);

/**
 * Scale a complex vector by a real scalar.
 * @param data Input/output vector (interleaved real/imag)
 * @param n Number of complex elements
 * @param scale Scale factor (real)
 */
void simd_complex_scale_float(float* data, size_t n, float scale);

/**
 * Add two vectors: out = a + b
 * @param out Output vector
 * @param a First input vector
 * @param b Second input vector
 * @param n Number of elements
 */
void simd_vector_add(float* out, const float* a, const float* b, size_t n);

/**
 * Subtract two vectors: out = a - b
 * @param out Output vector
 * @param a First input vector
 * @param b Second input vector
 * @param n Number of elements
 */
void simd_vector_sub(float* out, const float* a, const float* b, size_t n);

/**
 * Multiply two vectors element-wise: out = a * b
 * @param out Output vector
 * @param a First input vector
 * @param b Second input vector
 * @param n Number of elements
 */
void simd_vector_mul(float* out, const float* a, const float* b, size_t n);

/**
 * Dot product of two vectors.
 * @param a First input vector
 * @param b Second input vector
 * @param n Number of elements
 * @return Dot product
 */
float simd_vector_dot(const float* a, const float* b, size_t n);

// ============================================================================
// Complex Number Operations
// ============================================================================

/**
 * Complex multiply-accumulate: out += a * b
 * @param out Output vector (interleaved real/imag)
 * @param a First input vector (interleaved real/imag)
 * @param b Second input vector (interleaved real/imag)
 * @param n Number of complex elements
 */
void simd_complex_macc(float* out, const float* a, const float* b, size_t n);

/**
 * Complex conjugate multiply-accumulate: out += conj(a) * b
 * @param out Output vector (interleaved real/imag)
 * @param a First input vector (will be conjugated)
 * @param b Second input vector
 * @param n Number of complex elements
 */
void simd_complex_conj_macc(float* out, const float* a, const float* b, size_t n);

/**
 * Complex inner product: <a|b> = sum(conj(a) * b)
 * @param result Output: complex result (2 floats: real, imag)
 * @param a First input vector (will be conjugated)
 * @param b Second input vector
 * @param n Number of complex elements
 */
void simd_complex_inner_product(float* result, const float* a, const float* b, size_t n);

/**
 * Complex multiply: out = a * b
 * @param out Output vector (interleaved real/imag)
 * @param a First input vector
 * @param b Second input vector
 * @param n Number of complex elements
 */
void simd_complex_mul(float* out, const float* a, const float* b, size_t n);

/**
 * Apply complex phase: out = a * exp(i*theta)
 * @param out Output vector (interleaved real/imag)
 * @param a Input vector
 * @param theta Phase angle in radians
 * @param n Number of complex elements
 */
void simd_complex_phase(float* out, const float* a, float theta, size_t n);

// ============================================================================
// Matrix Operations
// ============================================================================

/**
 * Matrix-vector multiply: out = A * x
 * @param out Output vector (m elements)
 * @param A Input matrix (m x n, row-major)
 * @param x Input vector (n elements)
 * @param m Number of rows
 * @param n Number of columns
 */
void simd_matrix_vector_mul(float* out, const float* A, const float* x,
                            size_t m, size_t n);

/**
 * Complex matrix-vector multiply: out = A * x
 * @param out Output vector (m complex elements)
 * @param A Input matrix (m x n complex, row-major, interleaved)
 * @param x Input vector (n complex elements)
 * @param m Number of rows
 * @param n Number of columns
 */
void simd_complex_matrix_vector_mul(float* out, const float* A, const float* x,
                                    size_t m, size_t n);

/**
 * Matrix multiply: C = A * B
 * @param C Output matrix (m x k)
 * @param A First input matrix (m x n)
 * @param B Second input matrix (n x k)
 * @param m Number of rows in A/C
 * @param n Number of columns in A, rows in B
 * @param k Number of columns in B/C
 */
void simd_matrix_multiply(float* C, const float* A, const float* B,
                          size_t m, size_t n, size_t k);

/**
 * Complex matrix multiply: C = A * B
 * @param C Output matrix (m x k complex)
 * @param A First input matrix (m x n complex)
 * @param B Second input matrix (n x k complex)
 * @param m Number of rows in A/C
 * @param n Number of columns in A, rows in B
 * @param k Number of columns in B/C
 */
void simd_complex_matrix_multiply(float* C, const float* A, const float* B,
                                  size_t m, size_t n, size_t k);

// ============================================================================
// Quantum-Specific Operations
// ============================================================================

/**
 * Apply 2x2 gate to two amplitudes at specified indices.
 * @param state State vector (interleaved complex)
 * @param gate 2x2 gate matrix (4 complex elements, row-major)
 * @param idx0 First amplitude index
 * @param idx1 Second amplitude index
 */
void simd_apply_2x2_gate(float* state, const float* gate,
                         size_t idx0, size_t idx1);

/**
 * Apply single-qubit gate to entire state.
 * @param state State vector (interleaved complex)
 * @param gate 2x2 gate matrix
 * @param target Target qubit index
 * @param num_qubits Total number of qubits
 */
void simd_apply_single_qubit_gate(float* state, const float* gate,
                                  size_t target, size_t num_qubits);

/**
 * Apply controlled gate (CNOT-like).
 * @param state State vector (interleaved complex)
 * @param gate 2x2 gate matrix to apply when control is |1>
 * @param control Control qubit index
 * @param target Target qubit index
 * @param num_qubits Total number of qubits
 */
void simd_apply_controlled_gate(float* state, const float* gate,
                                size_t control, size_t target,
                                size_t num_qubits);

/**
 * Compute probability distribution from state vector.
 * @param probs Output probabilities (real array)
 * @param state Input state vector (interleaved complex)
 * @param n Number of amplitudes (2^num_qubits)
 */
void simd_compute_probabilities(float* probs, const float* state, size_t n);

/**
 * Compute expectation value: <psi|O|psi> for diagonal observable.
 * @param state State vector (interleaved complex)
 * @param observable Diagonal of observable (real array)
 * @param n Number of amplitudes
 * @return Expectation value (real)
 */
float simd_expectation_diagonal(const float* state, const float* observable, size_t n);

// ============================================================================
// Reduction Operations
// ============================================================================

/**
 * Sum all elements of a vector.
 * @param data Input vector
 * @param n Number of elements
 * @return Sum
 */
float simd_reduce_sum(const float* data, size_t n);

/**
 * Find maximum element.
 * @param data Input vector
 * @param n Number of elements
 * @return Maximum value
 */
float simd_reduce_max(const float* data, size_t n);

/**
 * Find minimum element.
 * @param data Input vector
 * @param n Number of elements
 * @return Minimum value
 */
float simd_reduce_min(const float* data, size_t n);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Check if SIMD operations are available.
 * @return Bitmask of available SIMD instruction sets
 */
int simd_get_capabilities(void);

/**
 * Get SIMD implementation name.
 * @return String describing the active SIMD implementation
 */
const char* simd_get_implementation_name(void);

/**
 * Align pointer to SIMD boundary.
 * @param ptr Pointer to align
 * @param alignment Alignment boundary (must be power of 2)
 * @return Aligned pointer
 */
static inline void* simd_align_ptr(void* ptr, size_t alignment) {
    return (void*)(((size_t)ptr + alignment - 1) & ~(alignment - 1));
}

/**
 * Check if pointer is aligned for SIMD.
 * @param ptr Pointer to check
 * @param alignment Alignment to check (must be power of 2)
 * @return true if aligned
 */
static inline bool simd_is_aligned(const void* ptr, size_t alignment) {
    return ((size_t)ptr & (alignment - 1)) == 0;
}

// Standard alignment for SIMD operations
#define SIMD_ALIGNMENT 32

// SIMD capability flags
#define SIMD_CAP_SSE2   (1 << 0)
#define SIMD_CAP_AVX    (1 << 1)
#define SIMD_CAP_AVX2   (1 << 2)
#define SIMD_CAP_FMA    (1 << 3)
#define SIMD_CAP_AVX512 (1 << 4)
#define SIMD_CAP_NEON   (1 << 5)

#ifdef __cplusplus
}
#endif

#endif // COMPUTE_SIMD_H
