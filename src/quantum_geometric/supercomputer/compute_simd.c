/**
 * compute_simd.c - Cross-platform SIMD dispatch
 *
 * This file dispatches to platform-specific SIMD implementations:
 * - AVX2+FMA on x86_64
 * - NEON on ARM64 (Apple Silicon)
 * - Scalar fallback on other platforms
 */

#include "quantum_geometric/supercomputer/compute_simd.h"
#include <math.h>
#include <string.h>

// ============================================================================
// Platform Detection and Implementation Selection
// ============================================================================

#if COMPUTE_ARCH_ARM64 && COMPUTE_HAS_NEON
    // Use ARM NEON - implementation in compute_simd_neon.c
    #define SIMD_IMPL_NAME "ARM NEON"
    #define USE_NEON_IMPL 1
#elif COMPUTE_ARCH_X86_64 && COMPUTE_HAS_AVX2
    // Use AVX2 - implementation in compute_simd_avx2.c
    #define SIMD_IMPL_NAME "x86 AVX2+FMA"
    #define USE_AVX2_IMPL 1
#else
    // Scalar fallback
    #define SIMD_IMPL_NAME "Scalar"
    #define USE_SCALAR_IMPL 1
#endif

// ============================================================================
// Scalar Fallback Implementations
// ============================================================================

#ifdef USE_SCALAR_IMPL

float simd_vector_norm(const float* data, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += data[i] * data[i];
    }
    return sqrtf(sum);
}

float simd_complex_norm_float(const float* data, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float re = data[2*i];
        float im = data[2*i + 1];
        sum += re * re + im * im;
    }
    return sqrtf(sum);
}

void simd_vector_scale(float* data, size_t n, float scale) {
    for (size_t i = 0; i < n; i++) {
        data[i] *= scale;
    }
}

void simd_complex_scale_float(float* data, size_t n, float scale) {
    for (size_t i = 0; i < 2 * n; i++) {
        data[i] *= scale;
    }
}

void simd_vector_add(float* out, const float* a, const float* b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

void simd_vector_sub(float* out, const float* a, const float* b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        out[i] = a[i] - b[i];
    }
}

void simd_vector_mul(float* out, const float* a, const float* b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        out[i] = a[i] * b[i];
    }
}

float simd_vector_dot(const float* a, const float* b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

void simd_complex_macc(float* out, const float* a, const float* b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float a_re = a[2*i];
        float a_im = a[2*i + 1];
        float b_re = b[2*i];
        float b_im = b[2*i + 1];
        // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        out[2*i]     += a_re * b_re - a_im * b_im;
        out[2*i + 1] += a_re * b_im + a_im * b_re;
    }
}

void simd_complex_conj_macc(float* out, const float* a, const float* b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float a_re = a[2*i];
        float a_im = -a[2*i + 1];  // Conjugate
        float b_re = b[2*i];
        float b_im = b[2*i + 1];
        out[2*i]     += a_re * b_re - a_im * b_im;
        out[2*i + 1] += a_re * b_im + a_im * b_re;
    }
}

void simd_complex_inner_product(float* result, const float* a, const float* b, size_t n) {
    result[0] = 0.0f;
    result[1] = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float a_re = a[2*i];
        float a_im = -a[2*i + 1];  // Conjugate of a
        float b_re = b[2*i];
        float b_im = b[2*i + 1];
        result[0] += a_re * b_re - a_im * b_im;
        result[1] += a_re * b_im + a_im * b_re;
    }
}

void simd_complex_mul(float* out, const float* a, const float* b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float a_re = a[2*i];
        float a_im = a[2*i + 1];
        float b_re = b[2*i];
        float b_im = b[2*i + 1];
        out[2*i]     = a_re * b_re - a_im * b_im;
        out[2*i + 1] = a_re * b_im + a_im * b_re;
    }
}

void simd_complex_phase(float* out, const float* a, float theta, size_t n) {
    float cos_t = cosf(theta);
    float sin_t = sinf(theta);
    for (size_t i = 0; i < n; i++) {
        float a_re = a[2*i];
        float a_im = a[2*i + 1];
        out[2*i]     = a_re * cos_t - a_im * sin_t;
        out[2*i + 1] = a_re * sin_t + a_im * cos_t;
    }
}

void simd_matrix_vector_mul(float* out, const float* A, const float* x,
                            size_t m, size_t n) {
    for (size_t i = 0; i < m; i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < n; j++) {
            sum += A[i * n + j] * x[j];
        }
        out[i] = sum;
    }
}

void simd_complex_matrix_vector_mul(float* out, const float* A, const float* x,
                                    size_t m, size_t n) {
    for (size_t i = 0; i < m; i++) {
        float sum_re = 0.0f;
        float sum_im = 0.0f;
        for (size_t j = 0; j < n; j++) {
            float a_re = A[2 * (i * n + j)];
            float a_im = A[2 * (i * n + j) + 1];
            float x_re = x[2 * j];
            float x_im = x[2 * j + 1];
            sum_re += a_re * x_re - a_im * x_im;
            sum_im += a_re * x_im + a_im * x_re;
        }
        out[2 * i] = sum_re;
        out[2 * i + 1] = sum_im;
    }
}

void simd_matrix_multiply(float* C, const float* A, const float* B,
                          size_t m, size_t n, size_t k) {
    memset(C, 0, m * k * sizeof(float));
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < k; j++) {
            float sum = 0.0f;
            for (size_t l = 0; l < n; l++) {
                sum += A[i * n + l] * B[l * k + j];
            }
            C[i * k + j] = sum;
        }
    }
}

void simd_complex_matrix_multiply(float* C, const float* A, const float* B,
                                  size_t m, size_t n, size_t k) {
    memset(C, 0, 2 * m * k * sizeof(float));
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < k; j++) {
            float sum_re = 0.0f;
            float sum_im = 0.0f;
            for (size_t l = 0; l < n; l++) {
                float a_re = A[2 * (i * n + l)];
                float a_im = A[2 * (i * n + l) + 1];
                float b_re = B[2 * (l * k + j)];
                float b_im = B[2 * (l * k + j) + 1];
                sum_re += a_re * b_re - a_im * b_im;
                sum_im += a_re * b_im + a_im * b_re;
            }
            C[2 * (i * k + j)] = sum_re;
            C[2 * (i * k + j) + 1] = sum_im;
        }
    }
}

void simd_apply_2x2_gate(float* state, const float* gate, size_t idx0, size_t idx1) {
    // Load amplitudes
    float a0_re = state[2 * idx0];
    float a0_im = state[2 * idx0 + 1];
    float a1_re = state[2 * idx1];
    float a1_im = state[2 * idx1 + 1];

    // Gate elements
    float g00_re = gate[0], g00_im = gate[1];
    float g01_re = gate[2], g01_im = gate[3];
    float g10_re = gate[4], g10_im = gate[5];
    float g11_re = gate[6], g11_im = gate[7];

    // Apply gate: |out> = G|in>
    // out0 = g00 * a0 + g01 * a1
    float out0_re = (g00_re * a0_re - g00_im * a0_im) + (g01_re * a1_re - g01_im * a1_im);
    float out0_im = (g00_re * a0_im + g00_im * a0_re) + (g01_re * a1_im + g01_im * a1_re);

    // out1 = g10 * a0 + g11 * a1
    float out1_re = (g10_re * a0_re - g10_im * a0_im) + (g11_re * a1_re - g11_im * a1_im);
    float out1_im = (g10_re * a0_im + g10_im * a0_re) + (g11_re * a1_im + g11_im * a1_re);

    // Store results
    state[2 * idx0] = out0_re;
    state[2 * idx0 + 1] = out0_im;
    state[2 * idx1] = out1_re;
    state[2 * idx1 + 1] = out1_im;
}

void simd_apply_single_qubit_gate(float* state, const float* gate,
                                  size_t target, size_t num_qubits) {
    size_t n = 1ULL << num_qubits;
    size_t target_mask = 1ULL << target;

    for (size_t i = 0; i < n; i++) {
        if ((i & target_mask) == 0) {
            size_t idx0 = i;
            size_t idx1 = i | target_mask;
            simd_apply_2x2_gate(state, gate, idx0, idx1);
        }
    }
}

void simd_apply_controlled_gate(float* state, const float* gate,
                                size_t control, size_t target,
                                size_t num_qubits) {
    size_t n = 1ULL << num_qubits;
    size_t control_mask = 1ULL << control;
    size_t target_mask = 1ULL << target;

    for (size_t i = 0; i < n; i++) {
        // Only apply when control qubit is |1> and target is |0>
        if ((i & control_mask) && (i & target_mask) == 0) {
            size_t idx0 = i;
            size_t idx1 = i | target_mask;
            simd_apply_2x2_gate(state, gate, idx0, idx1);
        }
    }
}

void simd_compute_probabilities(float* probs, const float* state, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float re = state[2*i];
        float im = state[2*i + 1];
        probs[i] = re * re + im * im;
    }
}

float simd_expectation_diagonal(const float* state, const float* observable, size_t n) {
    float result = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float re = state[2*i];
        float im = state[2*i + 1];
        float prob = re * re + im * im;
        result += prob * observable[i];
    }
    return result;
}

float simd_reduce_sum(const float* data, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum;
}

float simd_reduce_max(const float* data, size_t n) {
    if (n == 0) return 0.0f;
    float max_val = data[0];
    for (size_t i = 1; i < n; i++) {
        if (data[i] > max_val) max_val = data[i];
    }
    return max_val;
}

float simd_reduce_min(const float* data, size_t n) {
    if (n == 0) return 0.0f;
    float min_val = data[0];
    for (size_t i = 1; i < n; i++) {
        if (data[i] < min_val) min_val = data[i];
    }
    return min_val;
}

int simd_get_capabilities(void) {
    return 0;  // No SIMD in scalar mode
}

#endif // USE_SCALAR_IMPL

// ============================================================================
// Shared Utility Functions
// ============================================================================

const char* simd_get_implementation_name(void) {
    return SIMD_IMPL_NAME;
}
