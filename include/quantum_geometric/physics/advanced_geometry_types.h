/**
 * @file advanced_geometry_types.h
 * @brief Types for advanced geometric operations (Kähler, Calabi-Yau, G2 structures)
 */

#ifndef ADVANCED_GEOMETRY_TYPES_H
#define ADVANCED_GEOMETRY_TYPES_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include <pthread.h>
#include "quantum_geometric/core/error_codes.h"
#include "quantum_geometric/core/platform_intrinsics.h"
#include "quantum_geometric/core/quantum_complex.h"

#ifdef __cplusplus
extern "C" {
#endif

// Operation flags
#define QGT_OP_PARALLEL    0x01
#define QGT_OP_SIMD        0x02
#define QGT_OP_GPU         0x04
#define QGT_OP_ASYNC       0x08

// Mutex wrapper for thread synchronization
typedef struct qgt_mutex {
    pthread_rwlock_t rwlock;
    pthread_mutex_t mutex;
    bool initialized;
} qgt_mutex_t;

// Spin system for twistor space analysis
#ifdef QGT_SPIN_SYSTEM_DEFINED
#undef QGT_SPIN_SYSTEM_DEFINED
#endif
#define QGT_SPIN_SYSTEM_DEFINED
typedef struct qgt_spin_system {
    ComplexDouble* spin_states;         // Spinor field components
    ComplexDouble* spin_operators;      // Spin operator matrices
    double* spin_foam_metric;           // Spin-foam correlation metric tensor
    size_t num_states;                  // Number of spin states
    size_t spin_dim;                    // Dimension of spin space
} qgt_spin_system_t;

// Geometry data for advanced manifold operations
#ifdef QGT_GEOMETRY_DEFINED
#undef QGT_GEOMETRY_DEFINED
#endif
#define QGT_GEOMETRY_DEFINED
typedef struct qgt_geometry {
    ComplexDouble* kahler_metric;       // Kähler metric tensor
    double* ricci_tensor;               // Ricci curvature tensor
    ComplexDouble* calabi_yau;          // Calabi-Yau holomorphic form
    double* g2_form;                    // G2 holonomy 3-form
    double* spin7_form;                 // Spin(7) holonomy 4-form
    ComplexDouble* twistor_space;       // Twistor space coordinates
    double* metric_tensor;              // Base metric tensor
    double* connection_coeffs;          // Christoffel symbols / connection coefficients
    double* g2_structure;               // G2 structure form storage
    size_t metric_size;                 // Size of metric tensors
} qgt_geometry_t;

// Advanced geometry context for Kähler/Calabi-Yau/G2 operations
// Note: This is distinct from quantum_geometric_tensor_t which is the unified tensor type
typedef struct qgt_advanced_geometry {
    size_t dimension;                   // Manifold dimension
    qgt_geometry_t geometry;            // Geometric data
    qgt_spin_system_t spin_system;      // Spin system for twistor analysis
    qgt_mutex_t* mutex;                 // Thread synchronization
    bool is_initialized;                // Initialization flag
    uint32_t flags;                     // Operation flags
} qgt_advanced_geometry_t;

// Initialize mutex
static inline int qgt_mutex_init(qgt_mutex_t* mutex) {
    if (!mutex) return -1;
    if (pthread_rwlock_init(&mutex->rwlock, NULL) != 0) return -1;
    if (pthread_mutex_init(&mutex->mutex, NULL) != 0) {
        pthread_rwlock_destroy(&mutex->rwlock);
        return -1;
    }
    mutex->initialized = true;
    return 0;
}

// Destroy mutex
static inline void qgt_mutex_destroy(qgt_mutex_t* mutex) {
    if (mutex && mutex->initialized) {
        pthread_rwlock_destroy(&mutex->rwlock);
        pthread_mutex_destroy(&mutex->mutex);
        mutex->initialized = false;
    }
}

// Cross-platform SIMD helpers for geometry operations
#if QGT_ARCH_X86 && QGT_SIMD_AVX

// x86 AVX implementation
static inline void qgt_kahler_multiply_pd(
    double* result_real, double* result_imag,
    const double* omega_real, const double* omega_imag,
    const double* metric_real, const double* metric_imag,
    size_t count) {
    for (size_t i = 0; i + 4 <= count; i += 4) {
        __m256d or_v = _mm256_loadu_pd(omega_real + i);
        __m256d oi_v = _mm256_loadu_pd(omega_imag + i);
        __m256d mr_v = _mm256_loadu_pd(metric_real + i);
        __m256d mi_v = _mm256_loadu_pd(metric_imag + i);

        // Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        __m256d rr = _mm256_sub_pd(_mm256_mul_pd(or_v, mr_v), _mm256_mul_pd(oi_v, mi_v));
        __m256d ri = _mm256_add_pd(_mm256_mul_pd(or_v, mi_v), _mm256_mul_pd(oi_v, mr_v));

        _mm256_storeu_pd(result_real + i, rr);
        _mm256_storeu_pd(result_imag + i, ri);
    }
    // Handle remainder
    for (size_t i = (count / 4) * 4; i < count; i++) {
        result_real[i] = omega_real[i] * metric_real[i] - omega_imag[i] * metric_imag[i];
        result_imag[i] = omega_real[i] * metric_imag[i] + omega_imag[i] * metric_real[i];
    }
}

#elif QGT_ARCH_ARM && QGT_SIMD_NEON

// ARM NEON implementation
static inline void qgt_kahler_multiply_pd(
    double* result_real, double* result_imag,
    const double* omega_real, const double* omega_imag,
    const double* metric_real, const double* metric_imag,
    size_t count) {
    for (size_t i = 0; i + 2 <= count; i += 2) {
        float64x2_t or_v = vld1q_f64(omega_real + i);
        float64x2_t oi_v = vld1q_f64(omega_imag + i);
        float64x2_t mr_v = vld1q_f64(metric_real + i);
        float64x2_t mi_v = vld1q_f64(metric_imag + i);

        // Complex multiplication
        float64x2_t rr = vsubq_f64(vmulq_f64(or_v, mr_v), vmulq_f64(oi_v, mi_v));
        float64x2_t ri = vaddq_f64(vmulq_f64(or_v, mi_v), vmulq_f64(oi_v, mr_v));

        vst1q_f64(result_real + i, rr);
        vst1q_f64(result_imag + i, ri);
    }
    // Handle remainder
    for (size_t i = (count / 2) * 2; i < count; i++) {
        result_real[i] = omega_real[i] * metric_real[i] - omega_imag[i] * metric_imag[i];
        result_imag[i] = omega_real[i] * metric_imag[i] + omega_imag[i] * metric_real[i];
    }
}

#else

// Scalar fallback
static inline void qgt_kahler_multiply_pd(
    double* result_real, double* result_imag,
    const double* omega_real, const double* omega_imag,
    const double* metric_real, const double* metric_imag,
    size_t count) {
    for (size_t i = 0; i < count; i++) {
        result_real[i] = omega_real[i] * metric_real[i] - omega_imag[i] * metric_imag[i];
        result_imag[i] = omega_real[i] * metric_imag[i] + omega_imag[i] * metric_real[i];
    }
}

#endif

#ifdef __cplusplus
}
#endif

#endif // ADVANCED_GEOMETRY_TYPES_H
