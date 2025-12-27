/**
 * @file platform_intrinsics.h
 * @brief Platform-specific SIMD intrinsics abstraction
 *
 * This header provides cross-platform SIMD support by detecting the target
 * architecture and including the appropriate intrinsics headers.
 */

#ifndef PLATFORM_INTRINSICS_H
#define PLATFORM_INTRINSICS_H

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>

// Platform detection
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #define QGT_ARCH_X86 1
    #define QGT_ARCH_ARM 0
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
    #define QGT_ARCH_X86 0
    #define QGT_ARCH_ARM 1
#else
    #define QGT_ARCH_X86 0
    #define QGT_ARCH_ARM 0
#endif

// SIMD feature detection
#if QGT_ARCH_X86
    // x86/x64 architecture
    #if defined(__AVX512F__)
        #define QGT_SIMD_AVX512 1
        #define QGT_SIMD_AVX2 1
        #define QGT_SIMD_AVX 1
        #define QGT_SIMD_SSE4 1
        #define QGT_SIMD_LEVEL 512
    #elif defined(__AVX2__)
        #define QGT_SIMD_AVX512 0
        #define QGT_SIMD_AVX2 1
        #define QGT_SIMD_AVX 1
        #define QGT_SIMD_SSE4 1
        #define QGT_SIMD_LEVEL 256
    #elif defined(__AVX__)
        #define QGT_SIMD_AVX512 0
        #define QGT_SIMD_AVX2 0
        #define QGT_SIMD_AVX 1
        #define QGT_SIMD_SSE4 1
        #define QGT_SIMD_LEVEL 256
    #elif defined(__SSE4_2__) || defined(__SSE4_1__)
        #define QGT_SIMD_AVX512 0
        #define QGT_SIMD_AVX2 0
        #define QGT_SIMD_AVX 0
        #define QGT_SIMD_SSE4 1
        #define QGT_SIMD_LEVEL 128
    #else
        #define QGT_SIMD_AVX512 0
        #define QGT_SIMD_AVX2 0
        #define QGT_SIMD_AVX 0
        #define QGT_SIMD_SSE4 0
        #define QGT_SIMD_LEVEL 0
    #endif

    // Include x86 intrinsics
    #include <immintrin.h>

    // FMA detection
    #if defined(__FMA__)
        #define QGT_SIMD_FMA 1
    #else
        #define QGT_SIMD_FMA 0
    #endif

#elif QGT_ARCH_ARM
    // ARM architecture
    #define QGT_SIMD_AVX512 0
    #define QGT_SIMD_AVX2 0
    #define QGT_SIMD_AVX 0
    #define QGT_SIMD_SSE4 0
    #define QGT_SIMD_FMA 0

    #if defined(__ARM_NEON) || defined(__ARM_NEON__)
        #define QGT_SIMD_NEON 1
        #define QGT_SIMD_LEVEL 128
        #include <arm_neon.h>
    #else
        #define QGT_SIMD_NEON 0
        #define QGT_SIMD_LEVEL 0
    #endif

    #if defined(__ARM_FEATURE_SVE)
        #define QGT_SIMD_SVE 1
        #include <arm_sve.h>
    #else
        #define QGT_SIMD_SVE 0
    #endif

#else
    // Unknown or unsupported architecture - no SIMD
    #define QGT_SIMD_AVX512 0
    #define QGT_SIMD_AVX2 0
    #define QGT_SIMD_AVX 0
    #define QGT_SIMD_SSE4 0
    #define QGT_SIMD_NEON 0
    #define QGT_SIMD_SVE 0
    #define QGT_SIMD_FMA 0
    #define QGT_SIMD_LEVEL 0
#endif

// Check if any SIMD is available
#define QGT_HAS_SIMD (QGT_SIMD_LEVEL > 0)

// Memory alignment requirements based on SIMD level
#if QGT_SIMD_LEVEL >= 512
    #define QGT_SIMD_ALIGNMENT 64
#elif QGT_SIMD_LEVEL >= 256
    #define QGT_SIMD_ALIGNMENT 32
#elif QGT_SIMD_LEVEL >= 128
    #define QGT_SIMD_ALIGNMENT 16
#else
    #define QGT_SIMD_ALIGNMENT 8
#endif

// Alignment macros
#if defined(__GNUC__) || defined(__clang__)
    #define QGT_ALIGN(n) __attribute__((aligned(n)))
    #define QGT_SIMD_ALIGNED QGT_ALIGN(QGT_SIMD_ALIGNMENT)
#elif defined(_MSC_VER)
    #define QGT_ALIGN(n) __declspec(align(n))
    #define QGT_SIMD_ALIGNED QGT_ALIGN(QGT_SIMD_ALIGNMENT)
#else
    #define QGT_ALIGN(n)
    #define QGT_SIMD_ALIGNED
#endif

// Visibility and optimization attributes
#if defined(__GNUC__) && !defined(__clang__)
    // GCC supports optimize attribute
    #define QGT_PUBLIC __attribute__((visibility("default")))
    #define QGT_HIDDEN __attribute__((visibility("hidden")))
    #define QGT_HOT __attribute__((hot))
    #define QGT_COLD __attribute__((cold))
    #define QGT_VECTORIZE __attribute__((optimize("tree-vectorize")))
    #define QGT_NOINLINE __attribute__((noinline))
#elif defined(__clang__)
    // Clang doesn't support optimize attribute, use pragma instead
    #define QGT_PUBLIC __attribute__((visibility("default")))
    #define QGT_HIDDEN __attribute__((visibility("hidden")))
    #define QGT_HOT __attribute__((hot))
    #define QGT_COLD __attribute__((cold))
    #define QGT_VECTORIZE  // Vectorization handled by compiler flags
    #define QGT_NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
    #define QGT_PUBLIC __declspec(dllexport)
    #define QGT_HIDDEN
    #define QGT_HOT
    #define QGT_COLD
    #define QGT_VECTORIZE
    #define QGT_NOINLINE __declspec(noinline)
#else
    #define QGT_PUBLIC
    #define QGT_HIDDEN
    #define QGT_HOT
    #define QGT_COLD
    #define QGT_VECTORIZE
    #define QGT_NOINLINE
#endif

// Prefetch hints
#if defined(__GNUC__) || defined(__clang__)
    #define QGT_PREFETCH_READ(addr) __builtin_prefetch((addr), 0, 3)
    #define QGT_PREFETCH_WRITE(addr) __builtin_prefetch((addr), 1, 3)
    #define QGT_PREFETCH_READ_L2(addr) __builtin_prefetch((addr), 0, 2)
    #define QGT_PREFETCH_WRITE_L2(addr) __builtin_prefetch((addr), 1, 2)
#elif QGT_ARCH_X86 && defined(_MSC_VER)
    #include <xmmintrin.h>
    #define QGT_PREFETCH_READ(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T0)
    #define QGT_PREFETCH_WRITE(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T0)
    #define QGT_PREFETCH_READ_L2(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T1)
    #define QGT_PREFETCH_WRITE_L2(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T1)
#else
    #define QGT_PREFETCH_READ(addr) ((void)(addr))
    #define QGT_PREFETCH_WRITE(addr) ((void)(addr))
    #define QGT_PREFETCH_READ_L2(addr) ((void)(addr))
    #define QGT_PREFETCH_WRITE_L2(addr) ((void)(addr))
#endif

// ============================================================================
// Cross-platform SIMD vector types
// ============================================================================

#if QGT_ARCH_X86 && QGT_SIMD_AVX

// 256-bit double vector (4 doubles)
typedef __m256d qgt_vec4d;
// 256-bit float vector (8 floats)
typedef __m256 qgt_vec8f;
// 128-bit double vector (2 doubles)
typedef __m128d qgt_vec2d;
// 128-bit float vector (4 floats)
typedef __m128 qgt_vec4f;

// Load operations
#define qgt_load4d(ptr) _mm256_loadu_pd(ptr)
#define qgt_load8f(ptr) _mm256_loadu_ps(ptr)
#define qgt_load2d(ptr) _mm_loadu_pd(ptr)
#define qgt_load4f(ptr) _mm_loadu_ps(ptr)

#define qgt_load4d_aligned(ptr) _mm256_load_pd(ptr)
#define qgt_load8f_aligned(ptr) _mm256_load_ps(ptr)

// Store operations
#define qgt_store4d(ptr, v) _mm256_storeu_pd(ptr, v)
#define qgt_store8f(ptr, v) _mm256_storeu_ps(ptr, v)
#define qgt_store2d(ptr, v) _mm_storeu_pd(ptr, v)
#define qgt_store4f(ptr, v) _mm_storeu_ps(ptr, v)

#define qgt_store4d_aligned(ptr, v) _mm256_store_pd(ptr, v)
#define qgt_store8f_aligned(ptr, v) _mm256_store_ps(ptr, v)

// Arithmetic operations - doubles (4-wide)
#define qgt_add4d(a, b) _mm256_add_pd(a, b)
#define qgt_sub4d(a, b) _mm256_sub_pd(a, b)
#define qgt_mul4d(a, b) _mm256_mul_pd(a, b)
#define qgt_div4d(a, b) _mm256_div_pd(a, b)
#define qgt_sqrt4d(a) _mm256_sqrt_pd(a)

// Arithmetic operations - floats (8-wide)
#define qgt_add8f(a, b) _mm256_add_ps(a, b)
#define qgt_sub8f(a, b) _mm256_sub_ps(a, b)
#define qgt_mul8f(a, b) _mm256_mul_ps(a, b)
#define qgt_div8f(a, b) _mm256_div_ps(a, b)
#define qgt_sqrt8f(a) _mm256_sqrt_ps(a)

// Set operations
#define qgt_set1_4d(v) _mm256_set1_pd(v)
#define qgt_set1_8f(v) _mm256_set1_ps(v)
#define qgt_setzero_4d() _mm256_setzero_pd()
#define qgt_setzero_8f() _mm256_setzero_ps()

// FMA operations (if available)
#if QGT_SIMD_FMA
#define qgt_fmadd4d(a, b, c) _mm256_fmadd_pd(a, b, c)
#define qgt_fmadd8f(a, b, c) _mm256_fmadd_ps(a, b, c)
#define qgt_fmsub4d(a, b, c) _mm256_fmsub_pd(a, b, c)
#define qgt_fmsub8f(a, b, c) _mm256_fmsub_ps(a, b, c)
#else
#define qgt_fmadd4d(a, b, c) _mm256_add_pd(_mm256_mul_pd(a, b), c)
#define qgt_fmadd8f(a, b, c) _mm256_add_ps(_mm256_mul_ps(a, b), c)
#define qgt_fmsub4d(a, b, c) _mm256_sub_pd(_mm256_mul_pd(a, b), c)
#define qgt_fmsub8f(a, b, c) _mm256_sub_ps(_mm256_mul_ps(a, b), c)
#endif

// Horizontal operations
static inline double qgt_hadd4d(qgt_vec4d v) {
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    lo = _mm_add_pd(lo, hi);
    __m128d shuf = _mm_shuffle_pd(lo, lo, 1);
    lo = _mm_add_pd(lo, shuf);
    return _mm_cvtsd_f64(lo);
}

static inline float qgt_hadd8f(qgt_vec8f v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

#elif QGT_ARCH_ARM && QGT_SIMD_NEON

// NEON vector types
typedef float64x2_t qgt_vec2d;
typedef float32x4_t qgt_vec4f;

// For 4-wide double operations, we need to use two 2-wide vectors
typedef struct { float64x2_t lo; float64x2_t hi; } qgt_vec4d;
// For 8-wide float operations, we need to use two 4-wide vectors
typedef struct { float32x4_t lo; float32x4_t hi; } qgt_vec8f;

// Load operations
static inline qgt_vec4d qgt_load4d(const double* ptr) {
    qgt_vec4d v;
    v.lo = vld1q_f64(ptr);
    v.hi = vld1q_f64(ptr + 2);
    return v;
}

static inline qgt_vec8f qgt_load8f(const float* ptr) {
    qgt_vec8f v;
    v.lo = vld1q_f32(ptr);
    v.hi = vld1q_f32(ptr + 4);
    return v;
}

#define qgt_load2d(ptr) vld1q_f64(ptr)
#define qgt_load4f(ptr) vld1q_f32(ptr)

#define qgt_load4d_aligned(ptr) qgt_load4d(ptr)
#define qgt_load8f_aligned(ptr) qgt_load8f(ptr)

// Store operations
static inline void qgt_store4d(double* ptr, qgt_vec4d v) {
    vst1q_f64(ptr, v.lo);
    vst1q_f64(ptr + 2, v.hi);
}

static inline void qgt_store8f(float* ptr, qgt_vec8f v) {
    vst1q_f32(ptr, v.lo);
    vst1q_f32(ptr + 4, v.hi);
}

#define qgt_store2d(ptr, v) vst1q_f64(ptr, v)
#define qgt_store4f(ptr, v) vst1q_f32(ptr, v)

#define qgt_store4d_aligned(ptr, v) qgt_store4d(ptr, v)
#define qgt_store8f_aligned(ptr, v) qgt_store8f(ptr, v)

// Arithmetic operations - doubles (4-wide)
static inline qgt_vec4d qgt_add4d(qgt_vec4d a, qgt_vec4d b) {
    qgt_vec4d r;
    r.lo = vaddq_f64(a.lo, b.lo);
    r.hi = vaddq_f64(a.hi, b.hi);
    return r;
}

static inline qgt_vec4d qgt_sub4d(qgt_vec4d a, qgt_vec4d b) {
    qgt_vec4d r;
    r.lo = vsubq_f64(a.lo, b.lo);
    r.hi = vsubq_f64(a.hi, b.hi);
    return r;
}

static inline qgt_vec4d qgt_mul4d(qgt_vec4d a, qgt_vec4d b) {
    qgt_vec4d r;
    r.lo = vmulq_f64(a.lo, b.lo);
    r.hi = vmulq_f64(a.hi, b.hi);
    return r;
}

static inline qgt_vec4d qgt_div4d(qgt_vec4d a, qgt_vec4d b) {
    qgt_vec4d r;
    r.lo = vdivq_f64(a.lo, b.lo);
    r.hi = vdivq_f64(a.hi, b.hi);
    return r;
}

static inline qgt_vec4d qgt_sqrt4d(qgt_vec4d a) {
    qgt_vec4d r;
    r.lo = vsqrtq_f64(a.lo);
    r.hi = vsqrtq_f64(a.hi);
    return r;
}

// Arithmetic operations - floats (8-wide)
static inline qgt_vec8f qgt_add8f(qgt_vec8f a, qgt_vec8f b) {
    qgt_vec8f r;
    r.lo = vaddq_f32(a.lo, b.lo);
    r.hi = vaddq_f32(a.hi, b.hi);
    return r;
}

static inline qgt_vec8f qgt_sub8f(qgt_vec8f a, qgt_vec8f b) {
    qgt_vec8f r;
    r.lo = vsubq_f32(a.lo, b.lo);
    r.hi = vsubq_f32(a.hi, b.hi);
    return r;
}

static inline qgt_vec8f qgt_mul8f(qgt_vec8f a, qgt_vec8f b) {
    qgt_vec8f r;
    r.lo = vmulq_f32(a.lo, b.lo);
    r.hi = vmulq_f32(a.hi, b.hi);
    return r;
}

static inline qgt_vec8f qgt_div8f(qgt_vec8f a, qgt_vec8f b) {
    qgt_vec8f r;
    r.lo = vdivq_f32(a.lo, b.lo);
    r.hi = vdivq_f32(a.hi, b.hi);
    return r;
}

static inline qgt_vec8f qgt_sqrt8f(qgt_vec8f a) {
    qgt_vec8f r;
    r.lo = vsqrtq_f32(a.lo);
    r.hi = vsqrtq_f32(a.hi);
    return r;
}

// Set operations
static inline qgt_vec4d qgt_set1_4d(double v) {
    qgt_vec4d r;
    r.lo = vdupq_n_f64(v);
    r.hi = vdupq_n_f64(v);
    return r;
}

static inline qgt_vec8f qgt_set1_8f(float v) {
    qgt_vec8f r;
    r.lo = vdupq_n_f32(v);
    r.hi = vdupq_n_f32(v);
    return r;
}

static inline qgt_vec4d qgt_setzero_4d(void) {
    qgt_vec4d r;
    r.lo = vdupq_n_f64(0.0);
    r.hi = vdupq_n_f64(0.0);
    return r;
}

static inline qgt_vec8f qgt_setzero_8f(void) {
    qgt_vec8f r;
    r.lo = vdupq_n_f32(0.0f);
    r.hi = vdupq_n_f32(0.0f);
    return r;
}

// FMA operations (NEON has native FMA on ARMv8)
static inline qgt_vec4d qgt_fmadd4d(qgt_vec4d a, qgt_vec4d b, qgt_vec4d c) {
    qgt_vec4d r;
    r.lo = vfmaq_f64(c.lo, a.lo, b.lo);
    r.hi = vfmaq_f64(c.hi, a.hi, b.hi);
    return r;
}

static inline qgt_vec8f qgt_fmadd8f(qgt_vec8f a, qgt_vec8f b, qgt_vec8f c) {
    qgt_vec8f r;
    r.lo = vfmaq_f32(c.lo, a.lo, b.lo);
    r.hi = vfmaq_f32(c.hi, a.hi, b.hi);
    return r;
}

static inline qgt_vec4d qgt_fmsub4d(qgt_vec4d a, qgt_vec4d b, qgt_vec4d c) {
    qgt_vec4d r;
    r.lo = vfmsq_f64(c.lo, a.lo, b.lo);
    r.hi = vfmsq_f64(c.hi, a.hi, b.hi);
    return r;
}

static inline qgt_vec8f qgt_fmsub8f(qgt_vec8f a, qgt_vec8f b, qgt_vec8f c) {
    qgt_vec8f r;
    r.lo = vfmsq_f32(c.lo, a.lo, b.lo);
    r.hi = vfmsq_f32(c.hi, a.hi, b.hi);
    return r;
}

// Horizontal operations
static inline double qgt_hadd4d(qgt_vec4d v) {
    float64x2_t sum = vaddq_f64(v.lo, v.hi);
    return vgetq_lane_f64(sum, 0) + vgetq_lane_f64(sum, 1);
}

static inline float qgt_hadd8f(qgt_vec8f v) {
    float32x4_t sum = vaddq_f32(v.lo, v.hi);
    float32x2_t sum2 = vadd_f32(vget_low_f32(sum), vget_high_f32(sum));
    return vget_lane_f32(vpadd_f32(sum2, sum2), 0);
}

#else
// Scalar fallback for unsupported architectures

typedef struct { double v[4]; } qgt_vec4d;
typedef struct { float v[8]; } qgt_vec8f;
typedef struct { double v[2]; } qgt_vec2d;
typedef struct { float v[4]; } qgt_vec4f;

static inline qgt_vec4d qgt_load4d(const double* ptr) {
    qgt_vec4d r;
    for (int i = 0; i < 4; i++) r.v[i] = ptr[i];
    return r;
}

static inline qgt_vec8f qgt_load8f(const float* ptr) {
    qgt_vec8f r;
    for (int i = 0; i < 8; i++) r.v[i] = ptr[i];
    return r;
}

#define qgt_load4d_aligned(ptr) qgt_load4d(ptr)
#define qgt_load8f_aligned(ptr) qgt_load8f(ptr)

static inline void qgt_store4d(double* ptr, qgt_vec4d v) {
    for (int i = 0; i < 4; i++) ptr[i] = v.v[i];
}

static inline void qgt_store8f(float* ptr, qgt_vec8f v) {
    for (int i = 0; i < 8; i++) ptr[i] = v.v[i];
}

#define qgt_store4d_aligned(ptr, v) qgt_store4d(ptr, v)
#define qgt_store8f_aligned(ptr, v) qgt_store8f(ptr, v)

static inline qgt_vec4d qgt_add4d(qgt_vec4d a, qgt_vec4d b) {
    qgt_vec4d r;
    for (int i = 0; i < 4; i++) r.v[i] = a.v[i] + b.v[i];
    return r;
}

static inline qgt_vec4d qgt_sub4d(qgt_vec4d a, qgt_vec4d b) {
    qgt_vec4d r;
    for (int i = 0; i < 4; i++) r.v[i] = a.v[i] - b.v[i];
    return r;
}

static inline qgt_vec4d qgt_mul4d(qgt_vec4d a, qgt_vec4d b) {
    qgt_vec4d r;
    for (int i = 0; i < 4; i++) r.v[i] = a.v[i] * b.v[i];
    return r;
}

static inline qgt_vec4d qgt_div4d(qgt_vec4d a, qgt_vec4d b) {
    qgt_vec4d r;
    for (int i = 0; i < 4; i++) r.v[i] = a.v[i] / b.v[i];
    return r;
}

static inline qgt_vec4d qgt_sqrt4d(qgt_vec4d a) {
    qgt_vec4d r;
    for (int i = 0; i < 4; i++) r.v[i] = sqrt(a.v[i]);
    return r;
}

static inline qgt_vec8f qgt_add8f(qgt_vec8f a, qgt_vec8f b) {
    qgt_vec8f r;
    for (int i = 0; i < 8; i++) r.v[i] = a.v[i] + b.v[i];
    return r;
}

static inline qgt_vec8f qgt_sub8f(qgt_vec8f a, qgt_vec8f b) {
    qgt_vec8f r;
    for (int i = 0; i < 8; i++) r.v[i] = a.v[i] - b.v[i];
    return r;
}

static inline qgt_vec8f qgt_mul8f(qgt_vec8f a, qgt_vec8f b) {
    qgt_vec8f r;
    for (int i = 0; i < 8; i++) r.v[i] = a.v[i] * b.v[i];
    return r;
}

static inline qgt_vec8f qgt_div8f(qgt_vec8f a, qgt_vec8f b) {
    qgt_vec8f r;
    for (int i = 0; i < 8; i++) r.v[i] = a.v[i] / b.v[i];
    return r;
}

static inline qgt_vec8f qgt_sqrt8f(qgt_vec8f a) {
    qgt_vec8f r;
    for (int i = 0; i < 8; i++) r.v[i] = sqrtf(a.v[i]);
    return r;
}

static inline qgt_vec4d qgt_set1_4d(double v) {
    qgt_vec4d r;
    for (int i = 0; i < 4; i++) r.v[i] = v;
    return r;
}

static inline qgt_vec8f qgt_set1_8f(float v) {
    qgt_vec8f r;
    for (int i = 0; i < 8; i++) r.v[i] = v;
    return r;
}

static inline qgt_vec4d qgt_setzero_4d(void) {
    return qgt_set1_4d(0.0);
}

static inline qgt_vec8f qgt_setzero_8f(void) {
    return qgt_set1_8f(0.0f);
}

static inline qgt_vec4d qgt_fmadd4d(qgt_vec4d a, qgt_vec4d b, qgt_vec4d c) {
    qgt_vec4d r;
    for (int i = 0; i < 4; i++) r.v[i] = a.v[i] * b.v[i] + c.v[i];
    return r;
}

static inline qgt_vec8f qgt_fmadd8f(qgt_vec8f a, qgt_vec8f b, qgt_vec8f c) {
    qgt_vec8f r;
    for (int i = 0; i < 8; i++) r.v[i] = a.v[i] * b.v[i] + c.v[i];
    return r;
}

static inline qgt_vec4d qgt_fmsub4d(qgt_vec4d a, qgt_vec4d b, qgt_vec4d c) {
    qgt_vec4d r;
    for (int i = 0; i < 4; i++) r.v[i] = a.v[i] * b.v[i] - c.v[i];
    return r;
}

static inline qgt_vec8f qgt_fmsub8f(qgt_vec8f a, qgt_vec8f b, qgt_vec8f c) {
    qgt_vec8f r;
    for (int i = 0; i < 8; i++) r.v[i] = a.v[i] * b.v[i] - c.v[i];
    return r;
}

static inline double qgt_hadd4d(qgt_vec4d v) {
    return v.v[0] + v.v[1] + v.v[2] + v.v[3];
}

static inline float qgt_hadd8f(qgt_vec8f v) {
    return v.v[0] + v.v[1] + v.v[2] + v.v[3] + v.v[4] + v.v[5] + v.v[6] + v.v[7];
}

#endif // Architecture selection

// ============================================================================
// Common utility functions
// ============================================================================

// Get optimal SIMD vector width for current platform
static inline size_t qgt_get_simd_width_double(void) {
    return QGT_SIMD_LEVEL / 64;  // bytes to doubles
}

static inline size_t qgt_get_simd_width_float(void) {
    return QGT_SIMD_LEVEL / 32;  // bytes to floats
}

// Check if pointer is aligned for SIMD operations
static inline int qgt_is_aligned(const void* ptr) {
    return ((uintptr_t)ptr % QGT_SIMD_ALIGNMENT) == 0;
}

// Aligned allocation
static inline void* qgt_aligned_alloc(size_t size) {
#if defined(_MSC_VER)
    return _aligned_malloc(size, QGT_SIMD_ALIGNMENT);
#elif defined(__APPLE__)
    void* ptr = NULL;
    posix_memalign(&ptr, QGT_SIMD_ALIGNMENT, size);
    return ptr;
#else
    return aligned_alloc(QGT_SIMD_ALIGNMENT, size);
#endif
}

static inline void qgt_aligned_free(void* ptr) {
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

#endif // PLATFORM_INTRINSICS_H
