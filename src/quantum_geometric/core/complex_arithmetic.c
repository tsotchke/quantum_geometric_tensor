#include "quantum_geometric/core/complex_arithmetic.h"
#include <math.h>
#include <string.h>

ComplexFloat complex_add(ComplexFloat a, ComplexFloat b) {
    ComplexFloat result;
    result.real = a.real + b.real;
    result.imag = a.imag + b.imag;
    return result;
}

ComplexFloat complex_subtract(ComplexFloat a, ComplexFloat b) {
    ComplexFloat result;
    result.real = a.real - b.real;
    result.imag = a.imag - b.imag;
    return result;
}

ComplexFloat complex_multiply(ComplexFloat a, ComplexFloat b) {
    ComplexFloat result;
    result.real = a.real * b.real - a.imag * b.imag;
    result.imag = a.real * b.imag + a.imag * b.real;
    return result;
}

ComplexFloat complex_divide(ComplexFloat a, ComplexFloat b) {
    float denom = b.real * b.real + b.imag * b.imag;
    ComplexFloat result;
    result.real = (a.real * b.real + a.imag * b.imag) / denom;
    result.imag = (a.imag * b.real - a.real * b.imag) / denom;
    return result;
}

ComplexFloat complex_conjugate(ComplexFloat a) {
    ComplexFloat result;
    result.real = a.real;
    result.imag = -a.imag;
    return result;
}

float complex_abs(ComplexFloat a) {
    return sqrtf(a.real * a.real + a.imag * a.imag);
}

float complex_arg(ComplexFloat a) {
    return atan2f(a.imag, a.real);
}

float complex_abs_squared(ComplexFloat a) {
    return a.real * a.real + a.imag * a.imag;
}

void complex_vector_add(const ComplexFloat* a, const ComplexFloat* b,
                       ComplexFloat* result, size_t length) {
    for (size_t i = 0; i < length; i++) {
        result[i] = complex_add(a[i], b[i]);
    }
}

void complex_vector_subtract(const ComplexFloat* a, const ComplexFloat* b,
                           ComplexFloat* result, size_t length) {
    for (size_t i = 0; i < length; i++) {
        result[i] = complex_subtract(a[i], b[i]);
    }
}

void complex_vector_scale(const ComplexFloat* a, ComplexFloat scale,
                         ComplexFloat* result, size_t length) {
    for (size_t i = 0; i < length; i++) {
        result[i] = complex_multiply(a[i], scale);
    }
}

ComplexFloat complex_vector_dot(const ComplexFloat* a, const ComplexFloat* b,
                              size_t length) {
    ComplexFloat sum = {0.0f, 0.0f};
    for (size_t i = 0; i < length; i++) {
        sum = complex_add(sum, complex_multiply(a[i], complex_conjugate(b[i])));
    }
    return sum;
}

void complex_matrix_transpose(const ComplexFloat* a,
                            ComplexFloat* result,
                            size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result[j * rows + i] = a[i * cols + j];
        }
    }
}

void complex_matrix_conjugate_transpose(const ComplexFloat* a,
                                      ComplexFloat* result,
                                      size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result[j * rows + i] = complex_conjugate(a[i * cols + j]);
        }
    }
}

bool complex_is_zero(ComplexFloat a) {
    const float epsilon = 1e-6f;
    return fabsf(a.real) < epsilon && fabsf(a.imag) < epsilon;
}

bool complex_is_real(ComplexFloat a) {
    const float epsilon = 1e-6f;
    return fabsf(a.imag) < epsilon;
}

bool complex_is_equal(ComplexFloat a, ComplexFloat b) {
    const float epsilon = 1e-6f;
    return fabsf(a.real - b.real) < epsilon && fabsf(a.imag - b.imag) < epsilon;
}

ComplexFloat complex_from_polar(float r, float theta) {
    ComplexFloat result;
    result.real = r * cosf(theta);
    result.imag = r * sinf(theta);
    return result;
}

void complex_to_polar(ComplexFloat a, float* r, float* theta) {
    *r = complex_abs(a);
    *theta = complex_arg(a);
}

#ifdef __APPLE__
DSPComplex to_dsp_complex(ComplexFloat a) {
    DSPComplex result;
    result.real = a.real;
    result.imag = a.imag;
    return result;
}

ComplexFloat from_dsp_complex(DSPComplex a) {
    ComplexFloat result;
    result.real = a.real;
    result.imag = a.imag;
    return result;
}

__CLPK_complex to_lapack_complex(ComplexFloat a) {
    __CLPK_complex result;
    result.real = a.real;
    result.imag = a.imag;
    return result;
}

ComplexFloat from_lapack_complex(__CLPK_complex a) {
    ComplexFloat result;
    result.real = a.real;
    result.imag = a.imag;
    return result;
}
#endif

#ifdef HAVE_OPENBLAS
openblas_complex_float to_openblas_complex(ComplexFloat a) {
    openblas_complex_float result;
    result.real = a.real;
    result.imag = a.imag;
    return result;
}

ComplexFloat from_openblas_complex(openblas_complex_float a) {
    ComplexFloat result;
    result.real = a.real;
    result.imag = a.imag;
    return result;
}
#endif

#ifdef HAVE_MKL
MKL_Complex8 to_mkl_complex(ComplexFloat a) {
    MKL_Complex8 result;
    result.real = a.real;
    result.imag = a.imag;
    return result;
}

ComplexFloat from_mkl_complex(MKL_Complex8 a) {
    ComplexFloat result;
    result.real = a.real;
    result.imag = a.imag;
    return result;
}
#endif

// Advanced complex functions
ComplexFloat complex_sqrt(ComplexFloat a) {
    // Handle special cases
    if (a.imag == 0.0f) {
        if (a.real >= 0.0f) {
            ComplexFloat result = {sqrtf(a.real), 0.0f};
            return result;
        } else {
            ComplexFloat result = {0.0f, sqrtf(-a.real)};
            return result;
        }
    }

    // General case using polar form
    float r = complex_abs(a);
    float theta = complex_arg(a);
    float sqrt_r = sqrtf(r);
    float half_theta = theta * 0.5f;
    
    ComplexFloat result;
    result.real = sqrt_r * cosf(half_theta);
    result.imag = sqrt_r * sinf(half_theta);
    return result;
}

ComplexFloat complex_exp(ComplexFloat a) {
    float exp_real = expf(a.real);
    ComplexFloat result;
    result.real = exp_real * cosf(a.imag);
    result.imag = exp_real * sinf(a.imag);
    return result;
}

ComplexFloat complex_log(ComplexFloat a) {
    ComplexFloat result;
    result.real = logf(complex_abs(a));
    result.imag = complex_arg(a);
    return result;
}

ComplexFloat complex_pow(ComplexFloat a, ComplexFloat b) {
    // Handle special cases
    if (complex_is_zero(a)) {
        if (complex_is_zero(b)) {
            // 0^0 is undefined, return 1 by convention
            ComplexFloat result = {1.0f, 0.0f};
            return result;
        }
        ComplexFloat result = {0.0f, 0.0f};
        return result;
    }

    // General case using exp(b * log(a))
    ComplexFloat log_a = complex_log(a);
    ComplexFloat b_log_a = complex_multiply(b, log_a);
    return complex_exp(b_log_a);
}
