/**
 * @file advanced_geometry_operations.c
 * @brief Implementation of advanced geometric operations with cross-platform SIMD support
 *
 * Implements Kähler flow evolution, G2 structure computation, and twistor space analysis
 * with optimized SIMD paths for x86 (AVX) and ARM (NEON) architectures.
 */

#include "quantum_geometric/physics/advanced_geometry_types.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/core/platform_intrinsics.h"
#include <math.h>
#include <string.h>

// Platform-specific SIMD includes
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #define QGT_USE_AVX 1
    #include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
    #define QGT_USE_NEON 1
    #if defined(__ARM_NEON) || defined(__ARM_NEON__)
        #include <arm_neon.h>
    #endif
#endif

/*
 * =============================================================================
 * SIMD Helper Functions - Platform-specific implementations
 * =============================================================================
 */

#if QGT_USE_AVX
/* SIMD helper for Kähler operations - AVX version (operates on __m256d registers) */
static inline void qgt_kahler_multiply_avx(__m256d* result_real, __m256d* result_imag,
                                           const __m256d* omega_real, const __m256d* omega_imag,
                                           const __m256d* metric_real, const __m256d* metric_imag) {
    /* Complex multiplication with SIMD: (a+bi)(c+di) = (ac-bd) + (ad+bc)i */
    *result_real = _mm256_sub_pd(
        _mm256_mul_pd(*omega_real, *metric_real),
        _mm256_mul_pd(*omega_imag, *metric_imag)
    );
    *result_imag = _mm256_add_pd(
        _mm256_mul_pd(*omega_real, *metric_imag),
        _mm256_mul_pd(*omega_imag, *metric_real)
    );
}
#endif

#if QGT_USE_NEON
/* SIMD helper for Kähler operations - NEON version (operates on float64x2_t registers) */
static inline void qgt_kahler_multiply_neon(float64x2_t* result_real, float64x2_t* result_imag,
                                            const float64x2_t* omega_real, const float64x2_t* omega_imag,
                                            const float64x2_t* metric_real, const float64x2_t* metric_imag) {
    /* Complex multiplication with SIMD: (a+bi)(c+di) = (ac-bd) + (ad+bc)i */
    *result_real = vsubq_f64(
        vmulq_f64(*omega_real, *metric_real),
        vmulq_f64(*omega_imag, *metric_imag)
    );
    *result_imag = vaddq_f64(
        vmulq_f64(*omega_real, *metric_imag),
        vmulq_f64(*omega_imag, *metric_real)
    );
}
#endif

/* Scalar helper for Kähler operations - portable fallback */
static inline void qgt_kahler_multiply_scalar(double* result_real, double* result_imag,
                                              const double* omega_real, const double* omega_imag,
                                              const double* metric_real, const double* metric_imag,
                                              size_t count) {
    for (size_t i = 0; i < count; i++) {
        result_real[i] = omega_real[i] * metric_real[i] - omega_imag[i] * metric_imag[i];
        result_imag[i] = omega_real[i] * metric_imag[i] + omega_imag[i] * metric_real[i];
    }
}

/*
 * =============================================================================
 * Kähler Flow Evolution
 * =============================================================================
 */

#if QGT_USE_AVX
/* AVX implementation of Kähler flow evolution */
static qgt_error_t evolve_kahler_flow_avx(qgt_advanced_geometry_t* tensor, uint32_t flags) {
    const size_t dim = tensor->dimension;

    #pragma omp parallel for collapse(2) if(flags & QGT_OP_PARALLEL)
    for (size_t i = 0; i < dim; i += 4) {
        for (size_t j = 0; j < dim; j += 4) {
            size_t idx = i * dim + j;

            /* Load Kähler form - real and imaginary parts from ComplexDouble array */
            __m256d kahler_real = _mm256_set_pd(
                tensor->geometry.kahler_metric[idx + 3].real,
                tensor->geometry.kahler_metric[idx + 2].real,
                tensor->geometry.kahler_metric[idx + 1].real,
                tensor->geometry.kahler_metric[idx + 0].real
            );
            __m256d kahler_imag = _mm256_set_pd(
                tensor->geometry.kahler_metric[idx + 3].imag,
                tensor->geometry.kahler_metric[idx + 2].imag,
                tensor->geometry.kahler_metric[idx + 1].imag,
                tensor->geometry.kahler_metric[idx + 0].imag
            );

            /* Compute Ricci flow */
            __m256d ricci = _mm256_loadu_pd(&tensor->geometry.ricci_tensor[idx]);

            /* Evolve metric through holomorphic terms */
            for (size_t k = 0; k < dim; k += 4) {
                /* Load holomorphic terms from Calabi-Yau form */
                __m256d holo_real = _mm256_set_pd(
                    tensor->geometry.calabi_yau[k + 3].real,
                    tensor->geometry.calabi_yau[k + 2].real,
                    tensor->geometry.calabi_yau[k + 1].real,
                    tensor->geometry.calabi_yau[k + 0].real
                );
                __m256d holo_imag = _mm256_set_pd(
                    tensor->geometry.calabi_yau[k + 3].imag,
                    tensor->geometry.calabi_yau[k + 2].imag,
                    tensor->geometry.calabi_yau[k + 1].imag,
                    tensor->geometry.calabi_yau[k + 0].imag
                );

                /* Compute flow terms */
                __m256d flow_real, flow_imag;
                qgt_kahler_multiply_avx(&flow_real, &flow_imag,
                                       &kahler_real, &kahler_imag,
                                       &holo_real, &holo_imag);

                /* Update metric with Ricci flow contribution */
                kahler_real = _mm256_add_pd(kahler_real,
                    _mm256_mul_pd(flow_real, ricci));
                kahler_imag = _mm256_add_pd(kahler_imag,
                    _mm256_mul_pd(flow_imag, ricci));
            }

            /* Store evolved metric back to ComplexDouble array */
            double real_arr[4], imag_arr[4];
            _mm256_storeu_pd(real_arr, kahler_real);
            _mm256_storeu_pd(imag_arr, kahler_imag);
            for (size_t n = 0; n < 4 && (idx + n) < dim * dim; n++) {
                tensor->geometry.kahler_metric[idx + n].real = real_arr[n];
                tensor->geometry.kahler_metric[idx + n].imag = imag_arr[n];
            }
        }
    }

    return QGT_SUCCESS;
}
#endif

#if QGT_USE_NEON
/* NEON implementation of Kähler flow evolution */
static qgt_error_t evolve_kahler_flow_neon(qgt_advanced_geometry_t* tensor, uint32_t flags) {
    const size_t dim = tensor->dimension;

    #pragma omp parallel for collapse(2) if(flags & QGT_OP_PARALLEL)
    for (size_t i = 0; i < dim; i += 2) {
        for (size_t j = 0; j < dim; j += 2) {
            size_t idx = i * dim + j;

            /* Load Kähler form - real and imaginary parts from ComplexDouble array */
            double real_tmp[2] = {
                tensor->geometry.kahler_metric[idx + 0].real,
                tensor->geometry.kahler_metric[idx + 1].real
            };
            double imag_tmp[2] = {
                tensor->geometry.kahler_metric[idx + 0].imag,
                tensor->geometry.kahler_metric[idx + 1].imag
            };
            float64x2_t kahler_real = vld1q_f64(real_tmp);
            float64x2_t kahler_imag = vld1q_f64(imag_tmp);

            /* Compute Ricci flow */
            float64x2_t ricci = vld1q_f64(&tensor->geometry.ricci_tensor[idx]);

            /* Evolve metric through holomorphic terms */
            for (size_t k = 0; k < dim; k += 2) {
                /* Load holomorphic terms from Calabi-Yau form */
                double holo_real_tmp[2] = {
                    tensor->geometry.calabi_yau[k + 0].real,
                    tensor->geometry.calabi_yau[k + 1].real
                };
                double holo_imag_tmp[2] = {
                    tensor->geometry.calabi_yau[k + 0].imag,
                    tensor->geometry.calabi_yau[k + 1].imag
                };
                float64x2_t holo_real = vld1q_f64(holo_real_tmp);
                float64x2_t holo_imag = vld1q_f64(holo_imag_tmp);

                /* Compute flow terms */
                float64x2_t flow_real, flow_imag;
                qgt_kahler_multiply_neon(&flow_real, &flow_imag,
                                        &kahler_real, &kahler_imag,
                                        &holo_real, &holo_imag);

                /* Update metric with Ricci flow contribution */
                kahler_real = vaddq_f64(kahler_real, vmulq_f64(flow_real, ricci));
                kahler_imag = vaddq_f64(kahler_imag, vmulq_f64(flow_imag, ricci));
            }

            /* Store evolved metric back to ComplexDouble array */
            double real_arr[2], imag_arr[2];
            vst1q_f64(real_arr, kahler_real);
            vst1q_f64(imag_arr, kahler_imag);
            for (size_t n = 0; n < 2 && (idx + n) < dim * dim; n++) {
                tensor->geometry.kahler_metric[idx + n].real = real_arr[n];
                tensor->geometry.kahler_metric[idx + n].imag = imag_arr[n];
            }
        }
    }

    return QGT_SUCCESS;
}
#endif

/* Scalar implementation of Kähler flow evolution */
static qgt_error_t evolve_kahler_flow_scalar(qgt_advanced_geometry_t* tensor, uint32_t flags) {
    const size_t dim = tensor->dimension;

    #pragma omp parallel for collapse(2) if(flags & QGT_OP_PARALLEL)
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            size_t idx = i * dim + j;

            double kahler_real = tensor->geometry.kahler_metric[idx].real;
            double kahler_imag = tensor->geometry.kahler_metric[idx].imag;
            double ricci = tensor->geometry.ricci_tensor[idx];

            /* Evolve metric through holomorphic terms */
            for (size_t k = 0; k < dim; k++) {
                double holo_real = tensor->geometry.calabi_yau[k].real;
                double holo_imag = tensor->geometry.calabi_yau[k].imag;

                /* Complex multiplication for flow terms */
                double flow_real = kahler_real * holo_real - kahler_imag * holo_imag;
                double flow_imag = kahler_real * holo_imag + kahler_imag * holo_real;

                /* Update metric with Ricci flow contribution */
                kahler_real += flow_real * ricci;
                kahler_imag += flow_imag * ricci;
            }

            tensor->geometry.kahler_metric[idx].real = kahler_real;
            tensor->geometry.kahler_metric[idx].imag = kahler_imag;
        }
    }

    return QGT_SUCCESS;
}

/* Public Kähler flow evolution function - dispatches to platform-specific implementation */
QGT_PUBLIC QGT_HOT QGT_VECTORIZE qgt_error_t
evolve_kahler_flow(qgt_advanced_geometry_t* tensor, uint32_t flags) {
    if (!tensor) return QGT_ERROR_INVALID_ARGUMENT;

    /* Acquire write lock */
    qgt_mutex_t* mutex = tensor->mutex;
    if (pthread_rwlock_wrlock(&mutex->rwlock) != 0) {
        return QGT_ERROR_THREAD_ERROR;
    }

    qgt_error_t result;

#if QGT_USE_AVX
    result = evolve_kahler_flow_avx(tensor, flags);
#elif QGT_USE_NEON
    result = evolve_kahler_flow_neon(tensor, flags);
#else
    result = evolve_kahler_flow_scalar(tensor, flags);
#endif

    pthread_rwlock_unlock(&mutex->rwlock);
    return result;
}

/*
 * =============================================================================
 * G2 Structure Computation
 * =============================================================================
 */

#if QGT_USE_AVX
/* AVX implementation of G2 structure computation */
static qgt_error_t compute_g2_structure_avx(qgt_advanced_geometry_t* tensor, uint32_t flags) {
    const size_t dim = tensor->dimension;
    const size_t dim2 = dim * dim;
    const size_t dim3 = dim * dim * dim;

    #pragma omp parallel for collapse(2) if(flags & QGT_OP_PARALLEL)
    for (size_t i = 0; i < dim; i += 4) {
        for (size_t j = 0; j < dim; j += 4) {
            size_t idx = i * dim + j;

            /* Load metric components */
            __m256d metric = _mm256_loadu_pd(&tensor->geometry.metric_tensor[idx]);

            /* Compute associative 3-form (phi) */
            __m256d phi = _mm256_setzero_pd();
            for (size_t k = 0; k < dim; k += 4) {
                /* Include cross product terms from connection coefficients */
                size_t e1_idx = i * dim2 + j * dim + k;
                size_t e2_idx = j * dim2 + k * dim + i;
                size_t e3_idx = k * dim2 + i * dim + j;

                __m256d e1 = _mm256_loadu_pd(&tensor->geometry.connection_coeffs[e1_idx]);
                __m256d e2 = _mm256_loadu_pd(&tensor->geometry.connection_coeffs[e2_idx]);
                __m256d e3 = _mm256_loadu_pd(&tensor->geometry.connection_coeffs[e3_idx]);

                phi = _mm256_add_pd(phi,
                    _mm256_mul_pd(e1, _mm256_mul_pd(e2, e3)));
            }

            /* Compute coassociative 4-form (psi) */
            __m256d psi = _mm256_setzero_pd();
            for (size_t k = 0; k < dim; k += 4) {
                for (size_t l = 0; l < dim; l += 4) {
                    __m256d vol = _mm256_loadu_pd(&tensor->geometry.metric_tensor[k * dim + l]);
                    psi = _mm256_add_pd(psi,
                        _mm256_mul_pd(phi, _mm256_mul_pd(metric, vol)));
                }
            }

            /* Store G2 structure (phi + psi) */
            _mm256_storeu_pd(&tensor->geometry.g2_structure[idx],
                           _mm256_add_pd(phi, psi));
        }
    }

    return QGT_SUCCESS;
}
#endif

#if QGT_USE_NEON
/* NEON implementation of G2 structure computation */
static qgt_error_t compute_g2_structure_neon(qgt_advanced_geometry_t* tensor, uint32_t flags) {
    const size_t dim = tensor->dimension;
    const size_t dim2 = dim * dim;

    #pragma omp parallel for collapse(2) if(flags & QGT_OP_PARALLEL)
    for (size_t i = 0; i < dim; i += 2) {
        for (size_t j = 0; j < dim; j += 2) {
            size_t idx = i * dim + j;

            /* Load metric components */
            float64x2_t metric = vld1q_f64(&tensor->geometry.metric_tensor[idx]);

            /* Compute associative 3-form (phi) */
            float64x2_t phi = vdupq_n_f64(0.0);
            for (size_t k = 0; k < dim; k += 2) {
                /* Include cross product terms from connection coefficients */
                size_t e1_idx = i * dim2 + j * dim + k;
                size_t e2_idx = j * dim2 + k * dim + i;
                size_t e3_idx = k * dim2 + i * dim + j;

                float64x2_t e1 = vld1q_f64(&tensor->geometry.connection_coeffs[e1_idx]);
                float64x2_t e2 = vld1q_f64(&tensor->geometry.connection_coeffs[e2_idx]);
                float64x2_t e3 = vld1q_f64(&tensor->geometry.connection_coeffs[e3_idx]);

                phi = vaddq_f64(phi, vmulq_f64(e1, vmulq_f64(e2, e3)));
            }

            /* Compute coassociative 4-form (psi) */
            float64x2_t psi = vdupq_n_f64(0.0);
            for (size_t k = 0; k < dim; k += 2) {
                for (size_t l = 0; l < dim; l += 2) {
                    float64x2_t vol = vld1q_f64(&tensor->geometry.metric_tensor[k * dim + l]);
                    psi = vaddq_f64(psi, vmulq_f64(phi, vmulq_f64(metric, vol)));
                }
            }

            /* Store G2 structure (phi + psi) */
            vst1q_f64(&tensor->geometry.g2_structure[idx], vaddq_f64(phi, psi));
        }
    }

    return QGT_SUCCESS;
}
#endif

/* Scalar implementation of G2 structure computation */
static qgt_error_t compute_g2_structure_scalar(qgt_advanced_geometry_t* tensor, uint32_t flags) {
    const size_t dim = tensor->dimension;
    const size_t dim2 = dim * dim;

    #pragma omp parallel for collapse(2) if(flags & QGT_OP_PARALLEL)
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            size_t idx = i * dim + j;
            double metric = tensor->geometry.metric_tensor[idx];

            /* Compute associative 3-form (phi) */
            double phi = 0.0;
            for (size_t k = 0; k < dim; k++) {
                size_t e1_idx = i * dim2 + j * dim + k;
                size_t e2_idx = j * dim2 + k * dim + i;
                size_t e3_idx = k * dim2 + i * dim + j;

                double e1 = tensor->geometry.connection_coeffs[e1_idx];
                double e2 = tensor->geometry.connection_coeffs[e2_idx];
                double e3 = tensor->geometry.connection_coeffs[e3_idx];

                phi += e1 * e2 * e3;
            }

            /* Compute coassociative 4-form (psi) */
            double psi = 0.0;
            for (size_t k = 0; k < dim; k++) {
                for (size_t l = 0; l < dim; l++) {
                    double vol = tensor->geometry.metric_tensor[k * dim + l];
                    psi += phi * metric * vol;
                }
            }

            /* Store G2 structure (phi + psi) */
            tensor->geometry.g2_structure[idx] = phi + psi;
        }
    }

    return QGT_SUCCESS;
}

/* Public G2 structure computation function - dispatches to platform-specific implementation */
QGT_PUBLIC QGT_HOT QGT_VECTORIZE qgt_error_t
compute_g2_structure(qgt_advanced_geometry_t* tensor, uint32_t flags) {
    if (!tensor) return QGT_ERROR_INVALID_ARGUMENT;

    /* Acquire write lock */
    qgt_mutex_t* mutex = tensor->mutex;
    if (pthread_rwlock_wrlock(&mutex->rwlock) != 0) {
        return QGT_ERROR_THREAD_ERROR;
    }

    qgt_error_t result;

#if QGT_USE_AVX
    result = compute_g2_structure_avx(tensor, flags);
#elif QGT_USE_NEON
    result = compute_g2_structure_neon(tensor, flags);
#else
    result = compute_g2_structure_scalar(tensor, flags);
#endif

    pthread_rwlock_unlock(&mutex->rwlock);
    return result;
}

/*
 * =============================================================================
 * Twistor Space Analysis
 * =============================================================================
 */

#if QGT_USE_AVX
/* AVX implementation of twistor space analysis */
static qgt_error_t analyze_twistor_space_avx(qgt_advanced_geometry_t* tensor, uint32_t flags) {
    const size_t dim = tensor->dimension;

    #pragma omp parallel for collapse(2) if(flags & QGT_OP_PARALLEL)
    for (size_t i = 0; i < dim; i += 4) {
        for (size_t j = 0; j < dim; j += 4) {
            size_t idx = i * dim + j;

            /* Load spinor components from spin system */
            __m256d spinor_real = _mm256_set_pd(
                tensor->spin_system.spin_states[i + 3].real,
                tensor->spin_system.spin_states[i + 2].real,
                tensor->spin_system.spin_states[i + 1].real,
                tensor->spin_system.spin_states[i + 0].real
            );
            __m256d spinor_imag = _mm256_set_pd(
                tensor->spin_system.spin_states[i + 3].imag,
                tensor->spin_system.spin_states[i + 2].imag,
                tensor->spin_system.spin_states[i + 1].imag,
                tensor->spin_system.spin_states[i + 0].imag
            );

            /* Compute twistor transform */
            __m256d twistor_real = _mm256_setzero_pd();
            __m256d twistor_imag = _mm256_setzero_pd();

            for (size_t k = 0; k < dim; k += 4) {
                /* Include conformal structure from Kähler metric */
                __m256d conf_real = _mm256_set_pd(
                    tensor->geometry.kahler_metric[k + 3].real,
                    tensor->geometry.kahler_metric[k + 2].real,
                    tensor->geometry.kahler_metric[k + 1].real,
                    tensor->geometry.kahler_metric[k + 0].real
                );
                __m256d conf_imag = _mm256_set_pd(
                    tensor->geometry.kahler_metric[k + 3].imag,
                    tensor->geometry.kahler_metric[k + 2].imag,
                    tensor->geometry.kahler_metric[k + 1].imag,
                    tensor->geometry.kahler_metric[k + 0].imag
                );

                /* Apply twistor correspondence via complex multiplication */
                __m256d corr_real, corr_imag;
                qgt_kahler_multiply_avx(&corr_real, &corr_imag,
                                       &spinor_real, &spinor_imag,
                                       &conf_real, &conf_imag);

                twistor_real = _mm256_add_pd(twistor_real, corr_real);
                twistor_imag = _mm256_add_pd(twistor_imag, corr_imag);
            }

            /* Store twistor space coordinates */
            double real_arr[4], imag_arr[4];
            _mm256_storeu_pd(real_arr, twistor_real);
            _mm256_storeu_pd(imag_arr, twistor_imag);
            for (size_t n = 0; n < 4 && (idx + n) < dim * dim; n++) {
                tensor->geometry.twistor_space[idx + n].real = real_arr[n];
                tensor->geometry.twistor_space[idx + n].imag = imag_arr[n];
            }
        }
    }

    return QGT_SUCCESS;
}
#endif

#if QGT_USE_NEON
/* NEON implementation of twistor space analysis */
static qgt_error_t analyze_twistor_space_neon(qgt_advanced_geometry_t* tensor, uint32_t flags) {
    const size_t dim = tensor->dimension;

    #pragma omp parallel for collapse(2) if(flags & QGT_OP_PARALLEL)
    for (size_t i = 0; i < dim; i += 2) {
        for (size_t j = 0; j < dim; j += 2) {
            size_t idx = i * dim + j;

            /* Load spinor components from spin system */
            double spinor_real_tmp[2] = {
                tensor->spin_system.spin_states[i + 0].real,
                tensor->spin_system.spin_states[i + 1].real
            };
            double spinor_imag_tmp[2] = {
                tensor->spin_system.spin_states[i + 0].imag,
                tensor->spin_system.spin_states[i + 1].imag
            };
            float64x2_t spinor_real = vld1q_f64(spinor_real_tmp);
            float64x2_t spinor_imag = vld1q_f64(spinor_imag_tmp);

            /* Compute twistor transform */
            float64x2_t twistor_real = vdupq_n_f64(0.0);
            float64x2_t twistor_imag = vdupq_n_f64(0.0);

            for (size_t k = 0; k < dim; k += 2) {
                /* Include conformal structure from Kähler metric */
                double conf_real_tmp[2] = {
                    tensor->geometry.kahler_metric[k + 0].real,
                    tensor->geometry.kahler_metric[k + 1].real
                };
                double conf_imag_tmp[2] = {
                    tensor->geometry.kahler_metric[k + 0].imag,
                    tensor->geometry.kahler_metric[k + 1].imag
                };
                float64x2_t conf_real = vld1q_f64(conf_real_tmp);
                float64x2_t conf_imag = vld1q_f64(conf_imag_tmp);

                /* Apply twistor correspondence via complex multiplication */
                float64x2_t corr_real, corr_imag;
                qgt_kahler_multiply_neon(&corr_real, &corr_imag,
                                        &spinor_real, &spinor_imag,
                                        &conf_real, &conf_imag);

                twistor_real = vaddq_f64(twistor_real, corr_real);
                twistor_imag = vaddq_f64(twistor_imag, corr_imag);
            }

            /* Store twistor space coordinates */
            double real_arr[2], imag_arr[2];
            vst1q_f64(real_arr, twistor_real);
            vst1q_f64(imag_arr, twistor_imag);
            for (size_t n = 0; n < 2 && (idx + n) < dim * dim; n++) {
                tensor->geometry.twistor_space[idx + n].real = real_arr[n];
                tensor->geometry.twistor_space[idx + n].imag = imag_arr[n];
            }
        }
    }

    return QGT_SUCCESS;
}
#endif

/* Scalar implementation of twistor space analysis */
static qgt_error_t analyze_twistor_space_scalar(qgt_advanced_geometry_t* tensor, uint32_t flags) {
    const size_t dim = tensor->dimension;

    #pragma omp parallel for collapse(2) if(flags & QGT_OP_PARALLEL)
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            size_t idx = i * dim + j;

            double spinor_real = tensor->spin_system.spin_states[i].real;
            double spinor_imag = tensor->spin_system.spin_states[i].imag;

            /* Compute twistor transform */
            double twistor_real = 0.0;
            double twistor_imag = 0.0;

            for (size_t k = 0; k < dim; k++) {
                double conf_real = tensor->geometry.kahler_metric[k].real;
                double conf_imag = tensor->geometry.kahler_metric[k].imag;

                /* Apply twistor correspondence via complex multiplication */
                double corr_real = spinor_real * conf_real - spinor_imag * conf_imag;
                double corr_imag = spinor_real * conf_imag + spinor_imag * conf_real;

                twistor_real += corr_real;
                twistor_imag += corr_imag;
            }

            tensor->geometry.twistor_space[idx].real = twistor_real;
            tensor->geometry.twistor_space[idx].imag = twistor_imag;
        }
    }

    return QGT_SUCCESS;
}

/* Public twistor space analysis function - dispatches to platform-specific implementation */
QGT_PUBLIC QGT_HOT QGT_VECTORIZE qgt_error_t
analyze_twistor_space(qgt_advanced_geometry_t* tensor, uint32_t flags) {
    if (!tensor) return QGT_ERROR_INVALID_ARGUMENT;

    /* Acquire write lock */
    qgt_mutex_t* mutex = tensor->mutex;
    if (pthread_rwlock_wrlock(&mutex->rwlock) != 0) {
        return QGT_ERROR_THREAD_ERROR;
    }

    qgt_error_t result;

#if QGT_USE_AVX
    result = analyze_twistor_space_avx(tensor, flags);
#elif QGT_USE_NEON
    result = analyze_twistor_space_neon(tensor, flags);
#else
    result = analyze_twistor_space_scalar(tensor, flags);
#endif

    pthread_rwlock_unlock(&mutex->rwlock);
    return result;
}
