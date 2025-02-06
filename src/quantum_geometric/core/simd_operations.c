#include "quantum_geometric/core/simd_operations.h"
#include "quantum_geometric/core/quantum_geometric_logging.h"
#include "quantum_geometric/core/numeric_utils.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#elif defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

void simd_complex_copy(ComplexFloat* dest,
                      const ComplexFloat* src,
                      size_t count) {
    if (!dest || !src) {
        geometric_log_error("Invalid parameters passed to simd_complex_copy");
        return;
    }

#ifdef __AVX512F__
    // Process 8 complex numbers at a time using AVX-512
    size_t simd_count = count / 8 * 8;
    for (size_t i = 0; i < simd_count; i += 8) {
        __m512 vsrc = _mm512_loadu_ps((float*)&src[i]);
        _mm512_storeu_ps((float*)&dest[i], vsrc);
    }
    
    // Handle remaining elements
    for (size_t i = simd_count; i < count; i++) {
        dest[i] = src[i];
    }
#elif defined(__ARM_NEON)
    // Process 4 complex numbers at a time using NEON
    size_t simd_count = count / 4 * 4;
    for (size_t i = 0; i < simd_count; i += 4) {
        float32x4x2_t vsrc = vld2q_f32((const float*)&src[i]);
        vst2q_f32((float*)&dest[i], vsrc);
    }
    
    // Handle remaining elements
    for (size_t i = simd_count; i < count; i++) {
        dest[i] = src[i];
    }
#else
    // Scalar fallback
    memcpy(dest, src, count * sizeof(ComplexFloat));
#endif
}

void simd_complex_multiply_accumulate(ComplexFloat* result,
                                    const ComplexFloat* a,
                                    const ComplexFloat* b,
                                    size_t count) {
    if (!result || !a || !b) {
        geometric_log_error("Invalid parameters passed to simd_complex_multiply_accumulate");
        return;
    }

#ifdef __AVX512F__
    // Process 8 complex numbers at a time using AVX-512
    size_t simd_count = count / 8 * 8;
    for (size_t i = 0; i < simd_count; i += 8) {
        __m512 va = _mm512_loadu_ps((float*)&a[i]);
        __m512 vb = _mm512_loadu_ps((float*)&b[i]);
        __m512 vr = _mm512_loadu_ps((float*)&result[i]);
        
        // Extract real and imaginary parts
        __m512 va_real = _mm512_moveldup_ps(va);
        __m512 va_imag = _mm512_movehdup_ps(va);
        __m512 vb_real = _mm512_moveldup_ps(vb);
        __m512 vb_imag = _mm512_movehdup_ps(vb);
        
        // Complex multiplication
        __m512 prod_real = _mm512_fmsub_ps(va_real, vb_real, _mm512_mul_ps(va_imag, vb_imag));
        __m512 prod_imag = _mm512_fmadd_ps(va_real, vb_imag, _mm512_mul_ps(va_imag, vb_real));
        
        // Add to result
        __m512 sum = _mm512_add_ps(vr, _mm512_unpacklo_ps(prod_real, prod_imag));
        
        _mm512_storeu_ps((float*)&result[i], sum);
    }
    
    // Handle remaining elements
    for (size_t i = simd_count; i < count; i++) {
        ComplexFloat prod = complex_float_multiply(a[i], b[i]);
        result[i] = complex_float_add(result[i], prod);
    }
#elif defined(__ARM_NEON)
    // Process 4 complex numbers at a time using NEON
    size_t simd_count = count / 4 * 4;
    for (size_t i = 0; i < simd_count; i += 4) {
        float32x4x2_t va = vld2q_f32((const float*)&a[i]);
        float32x4x2_t vb = vld2q_f32((const float*)&b[i]);
        float32x4x2_t vr = vld2q_f32((const float*)&result[i]);
        
        // Complex multiplication
        float32x4_t prod_real = vmulq_f32(va.val[0], vb.val[0]);
        prod_real = vmlsq_f32(prod_real, va.val[1], vb.val[1]);
        
        float32x4_t prod_imag = vmulq_f32(va.val[0], vb.val[1]);
        prod_imag = vmlaq_f32(prod_imag, va.val[1], vb.val[0]);
        
        // Add to result
        float32x4x2_t sum;
        sum.val[0] = vaddq_f32(vr.val[0], prod_real);
        sum.val[1] = vaddq_f32(vr.val[1], prod_imag);
        
        vst2q_f32((float*)&result[i], sum);
    }
    
    // Handle remaining elements
    for (size_t i = simd_count; i < count; i++) {
        ComplexFloat prod = complex_float_multiply(a[i], b[i]);
        result[i] = complex_float_add(result[i], prod);
    }
#else
    // Scalar fallback
    for (size_t i = 0; i < count; i++) {
        ComplexFloat prod = complex_float_multiply(a[i], b[i]);
        result[i] = complex_float_add(result[i], prod);
    }
#endif
}

void simd_complex_scale(ComplexFloat* result,
                       const ComplexFloat* input,
                       ComplexFloat scalar,
                       size_t count) {
    if (!result || !input) {
        geometric_log_error("Invalid parameters passed to simd_complex_scale");
        return;
    }

#ifdef __AVX512F__
    // Process 8 complex numbers at a time using AVX-512
    size_t simd_count = count / 8 * 8;
    __m512 vscalar = _mm512_set_ps(scalar.imag, scalar.real,
                                  scalar.imag, scalar.real,
                                  scalar.imag, scalar.real,
                                  scalar.imag, scalar.real);
    
    for (size_t i = 0; i < simd_count; i += 8) {
        __m512 vin = _mm512_loadu_ps((float*)&input[i]);
        
        // Extract real and imaginary parts
        __m512 vin_real = _mm512_moveldup_ps(vin);
        __m512 vin_imag = _mm512_movehdup_ps(vin);
        __m512 vscalar_real = _mm512_moveldup_ps(vscalar);
        __m512 vscalar_imag = _mm512_movehdup_ps(vscalar);
        
        // Complex multiplication
        __m512 prod_real = _mm512_fmsub_ps(vin_real, vscalar_real, _mm512_mul_ps(vin_imag, vscalar_imag));
        __m512 prod_imag = _mm512_fmadd_ps(vin_real, vscalar_imag, _mm512_mul_ps(vin_imag, vscalar_real));
        
        __m512 prod = _mm512_unpacklo_ps(prod_real, prod_imag);
        
        _mm512_storeu_ps((float*)&result[i], prod);
    }
    
    // Handle remaining elements
    for (size_t i = simd_count; i < count; i++) {
        result[i] = complex_float_multiply(input[i], scalar);
    }
#elif defined(__ARM_NEON)
    // Process 4 complex numbers at a time using NEON
    size_t simd_count = count / 4 * 4;
    float32x4_t vscalar_real = vdupq_n_f32(scalar.real);
    float32x4_t vscalar_imag = vdupq_n_f32(scalar.imag);
    
    for (size_t i = 0; i < simd_count; i += 4) {
        float32x4x2_t vin = vld2q_f32((const float*)&input[i]);
        
        // Complex multiplication
        float32x4_t prod_real = vmulq_f32(vin.val[0], vscalar_real);
        prod_real = vmlsq_f32(prod_real, vin.val[1], vscalar_imag);
        
        float32x4_t prod_imag = vmulq_f32(vin.val[0], vscalar_imag);
        prod_imag = vmlaq_f32(prod_imag, vin.val[1], vscalar_real);
        
        float32x4x2_t vout;
        vout.val[0] = prod_real;
        vout.val[1] = prod_imag;
        
        vst2q_f32((float*)&result[i], vout);
    }
    
    // Handle remaining elements
    for (size_t i = simd_count; i < count; i++) {
        result[i] = complex_float_multiply(input[i], scalar);
    }
#else
    // Scalar fallback
    for (size_t i = 0; i < count; i++) {
        result[i] = complex_float_multiply(input[i], scalar);
    }
#endif
}

double simd_complex_norm(const ComplexFloat* input,
                        size_t count) {
    if (!input) {
        geometric_log_error("Invalid parameters passed to simd_complex_norm");
        return 0.0;
    }

#ifdef __AVX512F__
    // Process 8 complex numbers at a time using AVX-512
    __m512 sum = _mm512_setzero_ps();
    size_t simd_count = count / 8 * 8;
    
    for (size_t i = 0; i < simd_count; i += 8) {
        __m512 vin = _mm512_loadu_ps((float*)&input[i]);
        
        // Compute squares and add to sum
        sum = _mm512_fmadd_ps(vin, vin, sum);
    }
    
    // Horizontal sum of all elements
    float total = _mm512_reduce_add_ps(sum);
    
    // Handle remaining elements
    for (size_t i = simd_count; i < count; i++) {
        total += complex_float_abs_squared(input[i]);
    }
    
    return sqrtf(total);
#elif defined(__ARM_NEON)
    float32x4_t sum = vdupq_n_f32(0.0f);
    size_t simd_count = count / 4 * 4;
    
    for (size_t i = 0; i < simd_count; i += 4) {
        float32x4x2_t vin = vld2q_f32((const float*)&input[i]);
        
        // Compute squares and accumulate
        sum = vmlaq_f32(sum, vin.val[0], vin.val[0]);
        sum = vmlaq_f32(sum, vin.val[1], vin.val[1]);
    }
    
    // Horizontal sum
    float32x2_t sum2 = vadd_f32(vget_low_f32(sum), vget_high_f32(sum));
    float total = vget_lane_f32(vpadd_f32(sum2, sum2), 0);
    
    // Handle remaining elements
    for (size_t i = simd_count; i < count; i++) {
        total += complex_float_abs_squared(input[i]);
    }
    
    return sqrtf(total);
#else
    // Scalar fallback
    float sum = 0.0f;
    for (size_t i = 0; i < count; i++) {
        sum += complex_float_abs_squared(input[i]);
    }
    return sqrtf(sum);
#endif
}

void simd_complex_add(ComplexFloat* result,
                     const ComplexFloat* a,
                     const ComplexFloat* b,
                     size_t count) {
    if (!result || !a || !b) {
        geometric_log_error("Invalid parameters passed to simd_complex_add");
        return;
    }

#ifdef __AVX512F__
    // Process 8 complex numbers at a time using AVX-512
    size_t simd_count = count / 8 * 8;
    for (size_t i = 0; i < simd_count; i += 8) {
        __m512 va = _mm512_loadu_ps((float*)&a[i]);
        __m512 vb = _mm512_loadu_ps((float*)&b[i]);
        
        __m512 sum = _mm512_add_ps(va, vb);
        
        _mm512_storeu_ps((float*)&result[i], sum);
    }
    
    // Handle remaining elements
    for (size_t i = simd_count; i < count; i++) {
        result[i] = complex_float_add(a[i], b[i]);
    }
#elif defined(__ARM_NEON)
    size_t simd_count = count / 4 * 4;
    for (size_t i = 0; i < simd_count; i += 4) {
        float32x4x2_t va = vld2q_f32((const float*)&a[i]);
        float32x4x2_t vb = vld2q_f32((const float*)&b[i]);
        
        float32x4x2_t sum;
        sum.val[0] = vaddq_f32(va.val[0], vb.val[0]);
        sum.val[1] = vaddq_f32(va.val[1], vb.val[1]);
        
        vst2q_f32((float*)&result[i], sum);
    }
    
    // Handle remaining elements
    for (size_t i = simd_count; i < count; i++) {
        result[i] = complex_float_add(a[i], b[i]);
    }
#else
    // Scalar fallback
    for (size_t i = 0; i < count; i++) {
        result[i] = complex_float_add(a[i], b[i]);
    }
#endif
}

void simd_complex_subtract(ComplexFloat* result,
                         const ComplexFloat* a,
                         const ComplexFloat* b,
                         size_t count) {
    if (!result || !a || !b) {
        geometric_log_error("Invalid parameters passed to simd_complex_subtract");
        return;
    }

#ifdef __AVX512F__
    // Process 8 complex numbers at a time using AVX-512
    size_t simd_count = count / 8 * 8;
    for (size_t i = 0; i < simd_count; i += 8) {
        __m512 va = _mm512_loadu_ps((float*)&a[i]);
        __m512 vb = _mm512_loadu_ps((float*)&b[i]);
        
        __m512 diff = _mm512_sub_ps(va, vb);
        
        _mm512_storeu_ps((float*)&result[i], diff);
    }
    
    // Handle remaining elements
    for (size_t i = simd_count; i < count; i++) {
        result[i] = complex_float_subtract(a[i], b[i]);
    }
#elif defined(__ARM_NEON)
    size_t simd_count = count / 4 * 4;
    for (size_t i = 0; i < simd_count; i += 4) {
        float32x4x2_t va = vld2q_f32((const float*)&a[i]);
        float32x4x2_t vb = vld2q_f32((const float*)&b[i]);
        
        float32x4x2_t diff;
        diff.val[0] = vsubq_f32(va.val[0], vb.val[0]);
        diff.val[1] = vsubq_f32(va.val[1], vb.val[1]);
        
        vst2q_f32((float*)&result[i], diff);
    }
    
    // Handle remaining elements
    for (size_t i = simd_count; i < count; i++) {
        result[i] = complex_float_subtract(a[i], b[i]);
    }
#else
    // Scalar fallback
    for (size_t i = 0; i < count; i++) {
        result[i] = complex_float_subtract(a[i], b[i]);
    }
#endif
}

void simd_complex_multiply(ComplexFloat* result,
                         const ComplexFloat* a,
                         const ComplexFloat* b,
                         size_t count) {
    if (!result || !a || !b) {
        geometric_log_error("Invalid parameters passed to simd_complex_multiply");
        return;
    }

#ifdef __AVX512F__
    // Process 8 complex numbers at a time using AVX-512
    size_t simd_count = count / 8 * 8;
    for (size_t i = 0; i < simd_count; i += 8) {
        __m512 va = _mm512_loadu_ps((float*)&a[i]);
        __m512 vb = _mm512_loadu_ps((float*)&b[i]);
        
        // Extract real and imaginary parts
        __m512 va_real = _mm512_moveldup_ps(va);
        __m512 va_imag = _mm512_movehdup_ps(va);
        __m512 vb_real = _mm512_moveldup_ps(vb);
        __m512 vb_imag = _mm512_movehdup_ps(vb);
        
        // Complex multiplication
        __m512 prod_real = _mm512_fmsub_ps(va_real, vb_real, _mm512_mul_ps(va_imag, vb_imag));
        __m512 prod_imag = _mm512_fmadd_ps(va_real, vb_imag, _mm512_mul_ps(va_imag, vb_real));
        
        __m512 prod = _mm512_unpacklo_ps(prod_real, prod_imag);
        
        _mm512_storeu_ps((float*)&result[i], prod);
    }
    
    // Handle remaining elements
    for (size_t i = simd_count; i < count; i++) {
        result[i] = complex_float_multiply(a[i], b[i]);
    }
#elif defined(__ARM_NEON)
    size_t simd_count = count / 4 * 4;
    for (size_t i = 0; i < simd_count; i += 4) {
        float32x4x2_t va = vld2q_f32((const float*)&a[i]);
        float32x4x2_t vb = vld2q_f32((const float*)&b[i]);
        
        // Complex multiplication
        float32x4_t prod_real = vmulq_f32(va.val[0], vb.val[0]);
        prod_real = vmlsq_f32(prod_real, va.val[1], vb.val[1]);
        
        float32x4_t prod_imag = vmulq_f32(va.val[0], vb.val[1]);
        prod_imag = vmlaq_f32(prod_imag, va.val[1], vb.val[0]);
        
        float32x4x2_t vout;
        vout.val[0] = prod_real;
        vout.val[1] = prod_imag;
        
        vst2q_f32((float*)&result[i], vout);
    }
    
    // Handle remaining elements
    for (size_t i = simd_count; i < count; i++) {
        result[i] = complex_float_multiply(a[i], b[i]);
    }
#else
    // Scalar fallback
    for (size_t i = 0; i < count; i++) {
        result[i] = complex_float_multiply(a[i], b[i]);
    }
#endif
}

void simd_complex_divide(ComplexFloat* result,
                        const ComplexFloat* a,
                        const ComplexFloat* b,
                        size_t count) {
    if (!result || !a || !b) {
        geometric_log_error("Invalid parameters passed to simd_complex_divide");
        return;
    }

#ifdef __AVX512F__
    // Process 8 complex numbers at a time using AVX-512
    size_t simd_count = count / 8 * 8;
    for (size_t i = 0; i < simd_count; i += 8) {
        __m512 va = _mm512_loadu_ps((float*)&a[i]);
        __m512 vb = _mm512_loadu_ps((float*)&b[i]);
        
        // Extract real and imaginary parts
        __m512 va_real = _mm512_moveldup_ps(va);
        __m512 va_imag = _mm512_movehdup_ps(va);
        __m512 vb_real = _mm512_moveldup_ps(vb);
        __m512 vb_imag = _mm512_movehdup_ps(vb);
        
        // Calculate denominator (b.real^2 + b.imag^2)
        __m512 denom = _mm512_fmadd_ps(vb_real, vb_real, _mm512_mul_ps(vb_imag, vb_imag));
        
        // Calculate numerator real part (a.real*b.real + a.imag*b.imag)
        __m512 num_real = _mm512_fmadd_ps(va_real, vb_real, _mm512_mul_ps(va_imag, vb_imag));
        
        // Calculate numerator imaginary part (a.imag*b.real - a.real*b.imag)
        __m512 num_imag = _mm512_fmsub_ps(va_imag, vb_real, _mm512_mul_ps(va_real, vb_imag));
        
        // Divide by denominator
        __m512 quot_real = _mm512_div_ps(num_real, denom);
        __m512 quot_imag = _mm512_div_ps(num_imag, denom);
        
        __m512 quot = _mm512_unpacklo_ps(quot_real, quot_imag);
        
        _mm512_storeu_ps((float*)&result[i], quot);
    }
    
    // Handle remaining elements
    for (size_t i = simd_count; i < count; i++) {
        result[i] = complex_float_divide(a[i], b[i]);
    }
#elif defined(__ARM_NEON)
    size_t simd_count = count / 4 * 4;
    for (size_t i = 0; i < simd_count; i += 4) {
        float32x4x2_t va = vld2q_f32((const float*)&a[i]);
        float32x4x2_t vb = vld2q_f32((const float*)&b[i]);
        
        // Calculate denominator (b.real^2 + b.imag^2)
        float32x4_t denom = vmulq_f32(vb.val[0], vb.val[0]);
        denom = vmlaq_f32(denom, vb.val[1], vb.val[1]);
        
        // Calculate numerator real part (a.real*b.real + a.imag*b.imag)
        float32x4_t num_real = vmulq_f32(va.val[0], vb.val[0]);
        num_real = vmlaq_f32(num_real, va.val[1], vb.val[1]);
        
        // Calculate numerator imaginary part (a.imag*b.real - a.real*b.imag)
        float32x4_t num_imag = vmulq_f32(va.val[1], vb.val[0]);
        num_imag = vmlsq_f32(num_imag, va.val[0], vb.val[1]);
        
        // Divide by denominator
        float32x4x2_t vout;
        vout.val[0] = vdivq_f32(num_real, denom);
        vout.val[1] = vdivq_f32(num_imag, denom);
        
        vst2q_f32((float*)&result[i], vout);
    }
    
    // Handle remaining elements
    for (size_t i = simd_count; i < count; i++) {
        result[i] = complex_float_divide(a[i], b[i]);
    }
#else
    // Scalar fallback
    for (size_t i = 0; i < count; i++) {
        result[i] = complex_float_divide(a[i], b[i]);
    }
#endif
}

void simd_tensor_add(ComplexFloat* result,
                    const ComplexFloat* a,
                    const ComplexFloat* b,
                    size_t total_elements) {
    simd_complex_add(result, a, b, total_elements);
}

void simd_tensor_subtract(ComplexFloat* result,
                         const ComplexFloat* a,
                         const ComplexFloat* b,
                         size_t total_elements) {
    simd_complex_subtract(result, a, b, total_elements);
}

void simd_tensor_multiply(ComplexFloat* result,
                         const ComplexFloat* a,
                         const ComplexFloat* b,
                         const size_t* dimensions,
                         size_t rank) {
    if (!result || !a || !b || !dimensions || rank == 0) {
        geometric_log_error("Invalid parameters passed to simd_tensor_multiply");
        return;
    }

    size_t total_elements = 1;
    for (size_t i = 0; i < rank; i++) {
        total_elements *= dimensions[i];
    }

    simd_complex_multiply(result, a, b, total_elements);
}

void simd_tensor_scale(ComplexFloat* result,
                      const ComplexFloat* input,
                      ComplexFloat scalar,
                      size_t total_elements) {
    simd_complex_scale(result, input, scalar, total_elements);
}

double simd_tensor_norm(const ComplexFloat* input,
                       size_t total_elements) {
    return simd_complex_norm(input, total_elements);
}

void simd_tensor_conjugate(ComplexFloat* result,
                          const ComplexFloat* input,
                          size_t total_elements) {
    if (!result || !input) {
        geometric_log_error("Invalid parameters passed to simd_tensor_conjugate");
        return;
    }

#ifdef __AVX512F__
    // Process 8 complex numbers at a time using AVX-512
    size_t simd_count = total_elements / 8 * 8;
    const __m512 sign_mask = _mm512_set1_ps(-0.0f);
    
    for (size_t i = 0; i < simd_count; i += 8) {
        __m512 vin = _mm512_loadu_ps((float*)&input[i]);
        
        // Extract real and imaginary parts
        __m512 vin_real = _mm512_moveldup_ps(vin);
        __m512 vin_imag = _mm512_movehdup_ps(vin);
        
        // Negate imaginary part using sign bit
        __m512 conj_imag = _mm512_xor_ps(vin_imag, sign_mask);
        
        __m512 conj = _mm512_unpacklo_ps(vin_real, conj_imag);
        
        _mm512_storeu_ps((float*)&result[i], conj);
    }
    
    // Handle remaining elements
    for (size_t i = simd_count; i < total_elements; i++) {
        result[i] = complex_float_conjugate(input[i]);
    }
#elif defined(__ARM_NEON)
    size_t simd_count = total_elements / 4 * 4;
    
    for (size_t i = 0; i < simd_count; i += 4) {
        float32x4x2_t vin = vld2q_f32((const float*)&input[i]);
        
        float32x4x2_t vout;
        vout.val[0] = vin.val[0];
        vout.val[1] = vnegq_f32(vin.val[1]);
        
        vst2q_f32((float*)&result[i], vout);
    }
    
    // Handle remaining elements
    for (size_t i = simd_count; i < total_elements; i++) {
        result[i] = complex_float_conjugate(input[i]);
    }
#else
    // Scalar fallback
    for (size_t i = 0; i < total_elements; i++) {
        result[i] = complex_float_conjugate(input[i]);
    }
#endif
}

void simd_tensor_contract(ComplexFloat* result,
                         const ComplexFloat* a,
                         const ComplexFloat* b,
                         const size_t* dimensions_a,
                         const size_t* dimensions_b,
                         const size_t* contract_indices,
                         size_t num_indices,
                         size_t rank_a,
                         size_t rank_b) {
    if (!result || !a || !b || !dimensions_a || !dimensions_b || !contract_indices || 
        num_indices == 0 || rank_a == 0 || rank_b == 0) {
        geometric_log_error("Invalid parameters passed to simd_tensor_contract");
        return;
    }

    // Calculate contracted dimension size
    size_t contract_size = 1;
    for (size_t i = 0; i < num_indices; i++) {
        contract_size *= dimensions_a[contract_indices[i]];
    }

    // Calculate output dimensions
    size_t free_dims_a = rank_a - num_indices;
    size_t free_dims_b = rank_b - num_indices;

    // Calculate block size for efficient SIMD processing
    size_t block_size = 1;
#ifdef __AVX512F__
    block_size = 8;  // Process 8 complex numbers at a time
#elif defined(__ARM_NEON)
    block_size = 4;  // Process 4 complex numbers at a time
#endif
    // Use contract_size for block processing
    block_size = min(block_size, contract_size);

    // Perform blocked contraction
    simd_tensor_contract_block(result, a, b, block_size, free_dims_a, free_dims_b);
}

void simd_tensor_contract_block(ComplexFloat* result,
                              const ComplexFloat* a,
                              const ComplexFloat* b,
                              size_t block_size,
                              size_t free_dims_a,
                              size_t free_dims_b) {
    if (!result || !a || !b || block_size == 0) {
        geometric_log_error("Invalid parameters passed to simd_tensor_contract_block");
        return;
    }

#ifdef __AVX512F__
    // Process 8 complex numbers at a time using AVX-512
    size_t simd_count = block_size / 8 * 8;
    
    for (size_t i = 0; i < free_dims_a; i += 8) {
        for (size_t j = 0; j < free_dims_b; j++) {
            __m512 sum_real = _mm512_setzero_ps();
            __m512 sum_imag = _mm512_setzero_ps();
            
            for (size_t k = 0; k < simd_count; k += 8) {
                __m512 va = _mm512_loadu_ps((float*)&a[i * block_size + k]);
                __m512 vb = _mm512_loadu_ps((float*)&b[j * block_size + k]);
                
                // Complex multiplication accumulation
                __m512 va_real = _mm512_moveldup_ps(va);
                __m512 va_imag = _mm512_movehdup_ps(va);
                __m512 vb_real = _mm512_moveldup_ps(vb);
                __m512 vb_imag = _mm512_movehdup_ps(vb);
                
                sum_real = _mm512_fmadd_ps(va_real, vb_real, sum_real);
                sum_real = _mm512_fnmadd_ps(va_imag, vb_imag, sum_real);
                sum_imag = _mm512_fmadd_ps(va_real, vb_imag, sum_imag);
                sum_imag = _mm512_fmadd_ps(va_imag, vb_real, sum_imag);
            }
            
            // Store result
            _mm512_storeu_ps((float*)&result[i * free_dims_b + j], 
                            _mm512_unpacklo_ps(sum_real, sum_imag));
        }
    }
    
#elif defined(__ARM_NEON)
    // Process 4 complex numbers at a time using NEON
    size_t simd_count = block_size / 4 * 4;
    
    for (size_t i = 0; i < free_dims_a; i += 4) {
        for (size_t j = 0; j < free_dims_b; j++) {
            float32x4_t sum_real = vdupq_n_f32(0.0f);
            float32x4_t sum_imag = vdupq_n_f32(0.0f);
            
            for (size_t k = 0; k < simd_count; k += 4) {
                float32x4x2_t va = vld2q_f32((const float*)&a[i * block_size + k]);
                float32x4x2_t vb = vld2q_f32((const float*)&b[j * block_size + k]);
                
                // Complex multiplication accumulation
                sum_real = vmlaq_f32(sum_real, va.val[0], vb.val[0]);
                sum_real = vmlsq_f32(sum_real, va.val[1], vb.val[1]);
                sum_imag = vmlaq_f32(sum_imag, va.val[0], vb.val[1]);
                sum_imag = vmlaq_f32(sum_imag, va.val[1], vb.val[0]);
            }
            
            // Store result
            float32x4x2_t result_vec;
            result_vec.val[0] = sum_real;
            result_vec.val[1] = sum_imag;
            vst2q_f32((float*)&result[i * free_dims_b + j], result_vec);
        }
    }
    
#else
    // Scalar fallback
    for (size_t i = 0; i < free_dims_a; i++) {
        for (size_t j = 0; j < free_dims_b; j++) {
            ComplexFloat sum = COMPLEX_FLOAT_ZERO;
            
            for (size_t k = 0; k < block_size; k++) {
                ComplexFloat prod = complex_float_multiply(
                    a[i * block_size + k],
                    b[j * block_size + k]
                );
                sum = complex_float_add(sum, prod);
            }
            
            result[i * free_dims_b + j] = sum;
        }
    }
#endif

    // Handle remaining elements
    for (size_t i = 0; i < free_dims_a; i++) {
        for (size_t j = 0; j < free_dims_b; j++) {
            for (size_t k = (block_size / 8 * 8); k < block_size; k++) {
                ComplexFloat prod = complex_float_multiply(
                    a[i * block_size + k],
                    b[j * block_size + k]
                );
                result[i * free_dims_b + j] = complex_float_add(
                    result[i * free_dims_b + j],
                    prod
                );
            }
        }
    }
}

void simd_tensor_transpose(ComplexFloat* result,
                          const ComplexFloat* input,
                          const size_t* dimensions,
                          const size_t* permutation,
                          size_t rank) {
    if (!result || !input || !dimensions || !permutation || rank == 0) {
        geometric_log_error("Invalid parameters passed to simd_tensor_transpose");
        return;
    }

    // Calculate total elements and strides
    size_t total_elements = 1;
    size_t* old_strides = (size_t*)malloc(rank * sizeof(size_t));
    size_t* new_strides = (size_t*)malloc(rank * sizeof(size_t));
    
    if (!old_strides || !new_strides) {
        geometric_log_error("Memory allocation failed in simd_tensor_transpose");
        free(old_strides);
        free(new_strides);
        return;
    }

    // Calculate original strides
    old_strides[rank - 1] = 1;
    for (int i = rank - 2; i >= 0; i--) {
        old_strides[i] = old_strides[i + 1] * dimensions[i + 1];
    }

    // Calculate new strides based on permutation
    for (size_t i = 0; i < rank; i++) {
        new_strides[i] = old_strides[permutation[i]];
    }

    // Calculate total elements
    for (size_t i = 0; i < rank; i++) {
        total_elements *= dimensions[i];
    }

    // Perform transpose with SIMD where possible
    size_t* coords = (size_t*)calloc(rank, sizeof(size_t));
    if (!coords) {
        geometric_log_error("Memory allocation failed in simd_tensor_transpose");
        free(old_strides);
        free(new_strides);
        return;
    }

#ifdef __AVX512F__
    // Process 8 complex numbers at a time if they are contiguous
    size_t simd_count = total_elements / 8 * 8;
    for (size_t i = 0; i < simd_count; i += 8) {
        // Calculate source and destination indices
        size_t src_idx = 0;
        size_t dst_idx = 0;
        
        for (size_t j = 0; j < rank; j++) {
            src_idx += coords[j] * old_strides[j];
            dst_idx += coords[j] * new_strides[j];
        }

            // Check if next 8 elements are contiguous in both source and destination
            bool contiguous = true;
            for (size_t k = 1; k < 8; k++) {
                // Update coordinates
                size_t tmp = i + k;
                for (int j = rank - 1; j >= 0; j--) {
                size_t next_coord = (coords[j] + tmp % dimensions[j]) % dimensions[j];
                tmp /= dimensions[j];
                
                if (next_coord != coords[j]) {
                    contiguous = false;
                    break;
                }
            }
            
            if (!contiguous) break;
        }

        if (contiguous) {
            // Use SIMD for contiguous elements
            __m512 vin = _mm512_loadu_ps((float*)&input[src_idx]);
            _mm512_storeu_ps((float*)&result[dst_idx], vin);
        } else {
            // Handle non-contiguous elements scalar way
            result[dst_idx] = input[src_idx];
            
            // Update coordinates
            for (int j = rank - 1; j >= 0; j--) {
                coords[j]++;
                if (coords[j] < dimensions[j]) break;
                coords[j] = 0;
            }
        }
    }
#elif defined(__ARM_NEON)
    // Process 4 complex numbers at a time if they are contiguous
    size_t simd_count = total_elements / 4 * 4;
    for (size_t i = 0; i < simd_count; i += 4) {
        // Calculate source and destination indices
        size_t src_idx = 0;
        size_t dst_idx = 0;
        
        for (size_t j = 0; j < rank; j++) {
            src_idx += coords[j] * old_strides[j];
            dst_idx += coords[j] * new_strides[j];
        }

        // Check if next 4 elements are contiguous in both source and destination
        bool contiguous = true;
        for (size_t k = 1; k < 4; k++) {
            // Update coordinates
            size_t tmp = i + k;
            for (int j = rank - 1; j >= 0; j--) {
                size_t next_coord = (coords[j] + tmp % dimensions[j]) % dimensions[j];
                tmp /= dimensions[j];
                
                if (next_coord != coords[j]) {
                    contiguous = false;
                    break;
                }
            }
            
            if (!contiguous) break;
        }

        if (contiguous) {
            // Use SIMD for contiguous elements
            float32x4x2_t vin = vld2q_f32((const float*)&input[src_idx]);
            vst2q_f32((float*)&result[dst_idx], vin);
        } else {
            // Handle non-contiguous elements scalar way
            result[dst_idx] = input[src_idx];
            
            // Update coordinates
            for (int j = rank - 1; j >= 0; j--) {
                coords[j]++;
                if (coords[j] < dimensions[j]) break;
                coords[j] = 0;
            }
        }
    }
#else
    // Scalar implementation
    for (size_t i = 0; i < total_elements; i++) {
        // Calculate source and destination indices
        size_t src_idx = 0;
        size_t dst_idx = 0;
        
        for (size_t j = 0; j < rank; j++) {
            src_idx += coords[j] * old_strides[j];
            dst_idx += coords[j] * new_strides[j];
        }

        // Copy element
        result[dst_idx] = input[src_idx];

        // Update coordinates
        for (int j = rank - 1; j >= 0; j--) {
            coords[j]++;
            if (coords[j] < dimensions[j]) break;
            coords[j] = 0;
        }
    }
#endif

    // Handle remaining elements
    for (size_t i = (total_elements / 8 * 8); i < total_elements; i++) {
        // Calculate source and destination indices
        size_t src_idx = 0;
        size_t dst_idx = 0;
        
        for (size_t j = 0; j < rank; j++) {
            src_idx += coords[j] * old_strides[j];
            dst_idx += coords[j] * new_strides[j];
        }

        // Copy element
        result[dst_idx] = input[src_idx];

        // Update coordinates
        for (int j = rank - 1; j >= 0; j--) {
            coords[j]++;
            if (coords[j] < dimensions[j]) break;
            coords[j] = 0;
        }
    }

    // Clean up
    free(old_strides);
    free(new_strides);
    free(coords);
}
