/**
 * compute_simd_neon.c - ARM NEON SIMD implementations
 *
 * Optimized SIMD operations for ARM64 processors (Apple Silicon, etc.)
 * Uses NEON intrinsics for vectorized quantum computing operations.
 */

#include "quantum_geometric/supercomputer/compute_simd.h"

#if COMPUTE_ARCH_ARM64 && COMPUTE_HAS_NEON

#include <arm_neon.h>
#include <math.h>
#include <string.h>

// ============================================================================
// Vector Operations
// ============================================================================

float simd_vector_norm(const float* data, size_t n) {
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    size_t i = 0;

    // Process 4 elements at a time
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        sum_vec = vmlaq_f32(sum_vec, v, v);  // sum += v * v
    }

    // Horizontal sum
    float sum = vaddvq_f32(sum_vec);

    // Handle remaining elements
    for (; i < n; i++) {
        sum += data[i] * data[i];
    }

    return sqrtf(sum);
}

float simd_complex_norm_float(const float* data, size_t n) {
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    size_t i = 0;

    // Process 2 complex numbers (4 floats) at a time
    for (; i + 2 <= n; i += 2) {
        float32x4_t v = vld1q_f32(data + 2*i);  // [re0, im0, re1, im1]
        sum_vec = vmlaq_f32(sum_vec, v, v);     // sum += v * v
    }

    // Horizontal sum (all components contribute to norm squared)
    float sum = vaddvq_f32(sum_vec);

    // Handle remaining complex number
    for (; i < n; i++) {
        float re = data[2*i];
        float im = data[2*i + 1];
        sum += re * re + im * im;
    }

    return sqrtf(sum);
}

void simd_vector_scale(float* data, size_t n, float scale) {
    float32x4_t scale_vec = vdupq_n_f32(scale);
    size_t i = 0;

    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        v = vmulq_f32(v, scale_vec);
        vst1q_f32(data + i, v);
    }

    for (; i < n; i++) {
        data[i] *= scale;
    }
}

void simd_complex_scale_float(float* data, size_t n, float scale) {
    // Complex scale with real scalar is same as scaling 2n floats
    simd_vector_scale(data, 2 * n, scale);
}

void simd_vector_add(float* out, const float* a, const float* b, size_t n) {
    size_t i = 0;

    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vc = vaddq_f32(va, vb);
        vst1q_f32(out + i, vc);
    }

    for (; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

void simd_vector_sub(float* out, const float* a, const float* b, size_t n) {
    size_t i = 0;

    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vc = vsubq_f32(va, vb);
        vst1q_f32(out + i, vc);
    }

    for (; i < n; i++) {
        out[i] = a[i] - b[i];
    }
}

void simd_vector_mul(float* out, const float* a, const float* b, size_t n) {
    size_t i = 0;

    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vc = vmulq_f32(va, vb);
        vst1q_f32(out + i, vc);
    }

    for (; i < n; i++) {
        out[i] = a[i] * b[i];
    }
}

float simd_vector_dot(const float* a, const float* b, size_t n) {
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    size_t i = 0;

    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        sum_vec = vmlaq_f32(sum_vec, va, vb);
    }

    float sum = vaddvq_f32(sum_vec);

    for (; i < n; i++) {
        sum += a[i] * b[i];
    }

    return sum;
}

// ============================================================================
// Complex Number Operations
// ============================================================================

void simd_complex_macc(float* out, const float* a, const float* b, size_t n) {
    size_t i = 0;

    // Process 2 complex numbers at a time using NEON
    for (; i + 2 <= n; i += 2) {
        // Load: [a0_re, a0_im, a1_re, a1_im]
        float32x4_t va = vld1q_f32(a + 2*i);
        float32x4_t vb = vld1q_f32(b + 2*i);
        float32x4_t vo = vld1q_f32(out + 2*i);

        // Separate real and imaginary parts
        // a_re = [a0_re, a0_re, a1_re, a1_re]
        // a_im = [a0_im, a0_im, a1_im, a1_im]
        float32x4_t a_re = vtrn1q_f32(va, va);  // Duplicate reals
        float32x4_t a_im = vtrn2q_f32(va, va);  // Duplicate imags

        // b shuffled for complex multiply
        // b_ri = [b0_re, b0_im, b1_re, b1_im] (original)
        // b_ir = [b0_im, b0_re, b1_im, b1_re] (swapped)
        float32x4_t b_ri = vb;
        float32x4_t b_ir = vrev64q_f32(vb);

        // Complex multiply: (a_re + i*a_im) * (b_re + i*b_im)
        // = (a_re*b_re - a_im*b_im) + i*(a_re*b_im + a_im*b_re)
        // Using: [a_re*b_re, a_re*b_im] and [a_im*b_im, a_im*b_re]
        float32x4_t prod1 = vmulq_f32(a_re, b_ri);  // [a_re*b_re, a_re*b_im, ...]
        float32x4_t prod2 = vmulq_f32(a_im, b_ir);  // [a_im*b_im, a_im*b_re, ...]

        // Negate odd elements of prod2 for subtraction of imaginary*imaginary
        // result_re = a_re*b_re - a_im*b_im (subtract)
        // result_im = a_re*b_im + a_im*b_re (add)
        float32x4_t neg_mask = {1.0f, -1.0f, 1.0f, -1.0f};
        prod2 = vmulq_f32(prod2, neg_mask);

        // Final result
        float32x4_t result = vaddq_f32(prod1, prod2);
        vo = vaddq_f32(vo, result);

        vst1q_f32(out + 2*i, vo);
    }

    // Scalar fallback for remaining
    for (; i < n; i++) {
        float a_re = a[2*i];
        float a_im = a[2*i + 1];
        float b_re = b[2*i];
        float b_im = b[2*i + 1];
        out[2*i]     += a_re * b_re - a_im * b_im;
        out[2*i + 1] += a_re * b_im + a_im * b_re;
    }
}

void simd_complex_conj_macc(float* out, const float* a, const float* b, size_t n) {
    size_t i = 0;

    for (; i + 2 <= n; i += 2) {
        float32x4_t va = vld1q_f32(a + 2*i);
        float32x4_t vb = vld1q_f32(b + 2*i);
        float32x4_t vo = vld1q_f32(out + 2*i);

        // Conjugate a: negate imaginary parts
        float32x4_t conj_mask = {1.0f, -1.0f, 1.0f, -1.0f};
        va = vmulq_f32(va, conj_mask);

        float32x4_t a_re = vtrn1q_f32(va, va);
        float32x4_t a_im = vtrn2q_f32(va, va);
        float32x4_t b_ri = vb;
        float32x4_t b_ir = vrev64q_f32(vb);

        float32x4_t prod1 = vmulq_f32(a_re, b_ri);
        float32x4_t prod2 = vmulq_f32(a_im, b_ir);

        float32x4_t neg_mask = {1.0f, -1.0f, 1.0f, -1.0f};
        prod2 = vmulq_f32(prod2, neg_mask);

        float32x4_t result = vaddq_f32(prod1, prod2);
        vo = vaddq_f32(vo, result);

        vst1q_f32(out + 2*i, vo);
    }

    for (; i < n; i++) {
        float a_re = a[2*i];
        float a_im = -a[2*i + 1];  // Conjugate
        float b_re = b[2*i];
        float b_im = b[2*i + 1];
        out[2*i]     += a_re * b_re - a_im * b_im;
        out[2*i + 1] += a_re * b_im + a_im * b_re;
    }
}

void simd_complex_inner_product(float* result, const float* a, const float* b, size_t n) {
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    size_t i = 0;

    for (; i + 2 <= n; i += 2) {
        float32x4_t va = vld1q_f32(a + 2*i);
        float32x4_t vb = vld1q_f32(b + 2*i);

        // Conjugate a
        float32x4_t conj_mask = {1.0f, -1.0f, 1.0f, -1.0f};
        va = vmulq_f32(va, conj_mask);

        float32x4_t a_re = vtrn1q_f32(va, va);
        float32x4_t a_im = vtrn2q_f32(va, va);
        float32x4_t b_ri = vb;
        float32x4_t b_ir = vrev64q_f32(vb);

        float32x4_t prod1 = vmulq_f32(a_re, b_ri);
        float32x4_t prod2 = vmulq_f32(a_im, b_ir);

        float32x4_t neg_mask = {1.0f, -1.0f, 1.0f, -1.0f};
        prod2 = vmulq_f32(prod2, neg_mask);

        float32x4_t prod = vaddq_f32(prod1, prod2);
        sum_vec = vaddq_f32(sum_vec, prod);
    }

    // Horizontal sum - sum pairs of complex numbers
    float temp[4];
    vst1q_f32(temp, sum_vec);
    result[0] = temp[0] + temp[2];  // Real parts
    result[1] = temp[1] + temp[3];  // Imaginary parts

    // Handle remaining
    for (; i < n; i++) {
        float a_re = a[2*i];
        float a_im = -a[2*i + 1];  // Conjugate
        float b_re = b[2*i];
        float b_im = b[2*i + 1];
        result[0] += a_re * b_re - a_im * b_im;
        result[1] += a_re * b_im + a_im * b_re;
    }
}

void simd_complex_mul(float* out, const float* a, const float* b, size_t n) {
    size_t i = 0;

    for (; i + 2 <= n; i += 2) {
        float32x4_t va = vld1q_f32(a + 2*i);
        float32x4_t vb = vld1q_f32(b + 2*i);

        float32x4_t a_re = vtrn1q_f32(va, va);
        float32x4_t a_im = vtrn2q_f32(va, va);
        float32x4_t b_ri = vb;
        float32x4_t b_ir = vrev64q_f32(vb);

        float32x4_t prod1 = vmulq_f32(a_re, b_ri);
        float32x4_t prod2 = vmulq_f32(a_im, b_ir);

        float32x4_t neg_mask = {1.0f, -1.0f, 1.0f, -1.0f};
        prod2 = vmulq_f32(prod2, neg_mask);

        float32x4_t result = vaddq_f32(prod1, prod2);
        vst1q_f32(out + 2*i, result);
    }

    for (; i < n; i++) {
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

    // Phase rotation: (a + bi) * (cos + i*sin) = (a*cos - b*sin) + i*(a*sin + b*cos)
    float32x4_t cos_vec = vdupq_n_f32(cos_t);
    float32x4_t sin_vec = vdupq_n_f32(sin_t);

    size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float32x4_t va = vld1q_f32(a + 2*i);

        // Extract real and imaginary
        float32x2_t lo = vget_low_f32(va);
        float32x2_t hi = vget_high_f32(va);
        float32x4_t a_re = vcombine_f32(vdup_lane_f32(lo, 0), vdup_lane_f32(hi, 0));
        float32x4_t a_im = vcombine_f32(vdup_lane_f32(lo, 1), vdup_lane_f32(hi, 1));

        // Compute rotation
        float32x4_t out_re = vsubq_f32(vmulq_f32(a_re, cos_vec), vmulq_f32(a_im, sin_vec));
        float32x4_t out_im = vaddq_f32(vmulq_f32(a_re, sin_vec), vmulq_f32(a_im, cos_vec));

        // Interleave back
        float32x4x2_t interleaved = vzipq_f32(out_re, out_im);
        float temp[4];
        vst1q_f32(temp, interleaved.val[0]);
        out[2*i]     = temp[0];
        out[2*i + 1] = temp[1];
        out[2*i + 2] = temp[2];
        out[2*i + 3] = temp[3];
    }

    for (; i < n; i++) {
        float a_re = a[2*i];
        float a_im = a[2*i + 1];
        out[2*i]     = a_re * cos_t - a_im * sin_t;
        out[2*i + 1] = a_re * sin_t + a_im * cos_t;
    }
}

// ============================================================================
// Matrix Operations
// ============================================================================

void simd_matrix_vector_mul(float* out, const float* A, const float* x,
                            size_t m, size_t n) {
    for (size_t i = 0; i < m; i++) {
        float32x4_t sum_vec = vdupq_n_f32(0.0f);
        size_t j = 0;

        for (; j + 4 <= n; j += 4) {
            float32x4_t a_vec = vld1q_f32(A + i * n + j);
            float32x4_t x_vec = vld1q_f32(x + j);
            sum_vec = vmlaq_f32(sum_vec, a_vec, x_vec);
        }

        float sum = vaddvq_f32(sum_vec);

        for (; j < n; j++) {
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

    // Simple implementation - for production, use Accelerate framework
    for (size_t i = 0; i < m; i++) {
        for (size_t l = 0; l < n; l++) {
            float a_val = A[i * n + l];
            float32x4_t a_vec = vdupq_n_f32(a_val);
            size_t j = 0;

            for (; j + 4 <= k; j += 4) {
                float32x4_t b_vec = vld1q_f32(B + l * k + j);
                float32x4_t c_vec = vld1q_f32(C + i * k + j);
                c_vec = vmlaq_f32(c_vec, a_vec, b_vec);
                vst1q_f32(C + i * k + j, c_vec);
            }

            for (; j < k; j++) {
                C[i * k + j] += a_val * B[l * k + j];
            }
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

// ============================================================================
// Quantum-Specific Operations
// ============================================================================

void simd_apply_2x2_gate(float* state, const float* gate, size_t idx0, size_t idx1) {
    // Load amplitudes as float32x2 pairs (complex numbers)
    float32x2_t a0 = vld1_f32(state + 2 * idx0);  // [re0, im0]
    float32x2_t a1 = vld1_f32(state + 2 * idx1);  // [re1, im1]

    // Load gate elements
    float32x2_t g00 = vld1_f32(gate + 0);  // [g00_re, g00_im]
    float32x2_t g01 = vld1_f32(gate + 2);  // [g01_re, g01_im]
    float32x2_t g10 = vld1_f32(gate + 4);  // [g10_re, g10_im]
    float32x2_t g11 = vld1_f32(gate + 6);  // [g11_re, g11_im]

    // Complex multiply helper: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    #define COMPLEX_MUL(result, x, y) do { \
        float x_re = vget_lane_f32(x, 0); \
        float x_im = vget_lane_f32(x, 1); \
        float y_re = vget_lane_f32(y, 0); \
        float y_im = vget_lane_f32(y, 1); \
        result = (float32x2_t){x_re * y_re - x_im * y_im, x_re * y_im + x_im * y_re}; \
    } while(0)

    // out0 = g00 * a0 + g01 * a1
    float32x2_t t1, t2;
    COMPLEX_MUL(t1, g00, a0);
    COMPLEX_MUL(t2, g01, a1);
    float32x2_t out0 = vadd_f32(t1, t2);

    // out1 = g10 * a0 + g11 * a1
    COMPLEX_MUL(t1, g10, a0);
    COMPLEX_MUL(t2, g11, a1);
    float32x2_t out1 = vadd_f32(t1, t2);

    #undef COMPLEX_MUL

    // Store results
    vst1_f32(state + 2 * idx0, out0);
    vst1_f32(state + 2 * idx1, out1);
}

void simd_apply_single_qubit_gate(float* state, const float* gate,
                                  size_t target, size_t num_qubits) {
    size_t n = 1ULL << num_qubits;
    size_t target_mask = 1ULL << target;

    #pragma omp parallel for if(n > 1024)
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

    #pragma omp parallel for if(n > 1024)
    for (size_t i = 0; i < n; i++) {
        if ((i & control_mask) && (i & target_mask) == 0) {
            size_t idx0 = i;
            size_t idx1 = i | target_mask;
            simd_apply_2x2_gate(state, gate, idx0, idx1);
        }
    }
}

void simd_compute_probabilities(float* probs, const float* state, size_t n) {
    size_t i = 0;

    for (; i + 2 <= n; i += 2) {
        float32x4_t v = vld1q_f32(state + 2*i);  // [re0, im0, re1, im1]
        float32x4_t v_sq = vmulq_f32(v, v);       // [re0^2, im0^2, re1^2, im1^2]

        // Sum pairs to get probabilities
        float temp[4];
        vst1q_f32(temp, v_sq);
        probs[i]     = temp[0] + temp[1];
        probs[i + 1] = temp[2] + temp[3];
    }

    for (; i < n; i++) {
        float re = state[2*i];
        float im = state[2*i + 1];
        probs[i] = re * re + im * im;
    }
}

float simd_expectation_diagonal(const float* state, const float* observable, size_t n) {
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    size_t i = 0;

    for (; i + 2 <= n; i += 2) {
        float32x4_t s = vld1q_f32(state + 2*i);
        float32x2_t o = vld1_f32(observable + i);

        // Compute |amplitude|^2
        float32x4_t s_sq = vmulq_f32(s, s);
        float temp[4];
        vst1q_f32(temp, s_sq);
        float prob0 = temp[0] + temp[1];
        float prob1 = temp[2] + temp[3];

        // Multiply by observable
        sum_vec = vsetq_lane_f32(vgetq_lane_f32(sum_vec, 0) + prob0 * vget_lane_f32(o, 0), sum_vec, 0);
        sum_vec = vsetq_lane_f32(vgetq_lane_f32(sum_vec, 1) + prob1 * vget_lane_f32(o, 1), sum_vec, 1);
    }

    float sum = vaddvq_f32(sum_vec);

    for (; i < n; i++) {
        float re = state[2*i];
        float im = state[2*i + 1];
        float prob = re * re + im * im;
        sum += prob * observable[i];
    }

    return sum;
}

// ============================================================================
// Reduction Operations
// ============================================================================

float simd_reduce_sum(const float* data, size_t n) {
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    size_t i = 0;

    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        sum_vec = vaddq_f32(sum_vec, v);
    }

    float sum = vaddvq_f32(sum_vec);

    for (; i < n; i++) {
        sum += data[i];
    }

    return sum;
}

float simd_reduce_max(const float* data, size_t n) {
    if (n == 0) return 0.0f;

    float32x4_t max_vec = vdupq_n_f32(data[0]);
    size_t i = 0;

    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        max_vec = vmaxq_f32(max_vec, v);
    }

    float max_val = vmaxvq_f32(max_vec);

    for (; i < n; i++) {
        if (data[i] > max_val) max_val = data[i];
    }

    return max_val;
}

float simd_reduce_min(const float* data, size_t n) {
    if (n == 0) return 0.0f;

    float32x4_t min_vec = vdupq_n_f32(data[0]);
    size_t i = 0;

    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        min_vec = vminq_f32(min_vec, v);
    }

    float min_val = vminvq_f32(min_vec);

    for (; i < n; i++) {
        if (data[i] < min_val) min_val = data[i];
    }

    return min_val;
}

// ============================================================================
// Capability Detection
// ============================================================================

int simd_get_capabilities(void) {
    return SIMD_CAP_NEON;
}

#endif // COMPUTE_ARCH_ARM64 && COMPUTE_HAS_NEON
