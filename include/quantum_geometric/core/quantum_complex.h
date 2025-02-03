#ifndef QUANTUM_COMPLEX_H
#define QUANTUM_COMPLEX_H

#include <stdbool.h>
#include <math.h>

// Complex number type using single precision
typedef struct {
    float real;
    float imag;
} ComplexFloat;

// Complex number type using double precision
typedef struct {
    double real;
    double imag;
} ComplexDouble;

// Complex number constants
static const ComplexFloat COMPLEX_FLOAT_ZERO = {0.0f, 0.0f};
static const ComplexFloat COMPLEX_FLOAT_ONE = {1.0f, 0.0f};
static const ComplexFloat COMPLEX_FLOAT_I = {0.0f, 1.0f};

static const ComplexDouble COMPLEX_DOUBLE_ZERO = {0.0, 0.0};
static const ComplexDouble COMPLEX_DOUBLE_ONE = {1.0, 0.0};
static const ComplexDouble COMPLEX_DOUBLE_I = {0.0, 1.0};

// Complex number operations (single precision)
static inline ComplexFloat complex_float_create(float real, float imag) {
    ComplexFloat c = {real, imag};
    return c;
}

static inline ComplexFloat complex_float_add(ComplexFloat a, ComplexFloat b) {
    ComplexFloat c = {a.real + b.real, a.imag + b.imag};
    return c;
}

static inline ComplexFloat complex_float_subtract(ComplexFloat a, ComplexFloat b) {
    ComplexFloat c = {a.real - b.real, a.imag - b.imag};
    return c;
}

static inline ComplexFloat complex_float_multiply(ComplexFloat a, ComplexFloat b) {
    ComplexFloat c = {
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real
    };
    return c;
}

static inline ComplexFloat complex_float_multiply_real(ComplexFloat a, float r) {
    ComplexFloat c = {a.real * r, a.imag * r};
    return c;
}

static inline ComplexFloat complex_float_divide(ComplexFloat a, ComplexFloat b) {
    float denom = b.real * b.real + b.imag * b.imag;
    ComplexFloat c = {
        (a.real * b.real + a.imag * b.imag) / denom,
        (a.imag * b.real - a.real * b.imag) / denom
    };
    return c;
}

static inline ComplexFloat complex_float_conjugate(ComplexFloat a) {
    ComplexFloat c = {a.real, -a.imag};
    return c;
}

static inline float complex_float_abs(ComplexFloat a) {
    return sqrtf(a.real * a.real + a.imag * a.imag);
}

static inline float complex_float_abs_squared(ComplexFloat a) {
    return a.real * a.real + a.imag * a.imag;
}

static inline float complex_float_arg(ComplexFloat a) {
    return atan2f(a.imag, a.real);
}

static inline ComplexFloat complex_float_negate(ComplexFloat a) {
    ComplexFloat c = {-a.real, -a.imag};
    return c;
}

static inline ComplexFloat complex_float_exp(ComplexFloat a) {
    float r = expf(a.real);
    ComplexFloat c = {r * cosf(a.imag), r * sinf(a.imag)};
    return c;
}

static inline ComplexFloat complex_float_sqrt(ComplexFloat a) {
    float r = sqrtf(complex_float_abs(a));
    float theta = complex_float_arg(a) / 2.0f;
    ComplexFloat c = {r * cosf(theta), r * sinf(theta)};
    return c;
}

// Comparison operations
static inline bool complex_float_equals(ComplexFloat a, ComplexFloat b) {
    return a.real == b.real && a.imag == b.imag;
}

static inline bool complex_float_less_than(ComplexFloat a, ComplexFloat b) {
    return complex_float_abs_squared(a) < complex_float_abs_squared(b);
}

static inline bool complex_float_greater_than(ComplexFloat a, ComplexFloat b) {
    return complex_float_abs_squared(a) > complex_float_abs_squared(b);
}

static inline bool complex_float_less_equal(ComplexFloat a, ComplexFloat b) {
    return complex_float_abs_squared(a) <= complex_float_abs_squared(b);
}

static inline bool complex_float_greater_equal(ComplexFloat a, ComplexFloat b) {
    return complex_float_abs_squared(a) >= complex_float_abs_squared(b);
}

// Complex number operations (double precision)
static inline ComplexDouble complex_double_create(double real, double imag) {
    ComplexDouble c = {real, imag};
    return c;
}

static inline ComplexDouble complex_double_add(ComplexDouble a, ComplexDouble b) {
    ComplexDouble c = {a.real + b.real, a.imag + b.imag};
    return c;
}

static inline ComplexDouble complex_double_subtract(ComplexDouble a, ComplexDouble b) {
    ComplexDouble c = {a.real - b.real, a.imag - b.imag};
    return c;
}

static inline ComplexDouble complex_double_multiply(ComplexDouble a, ComplexDouble b) {
    ComplexDouble c = {
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real
    };
    return c;
}

static inline ComplexDouble complex_double_divide(ComplexDouble a, ComplexDouble b) {
    double denom = b.real * b.real + b.imag * b.imag;
    ComplexDouble c = {
        (a.real * b.real + a.imag * b.imag) / denom,
        (a.imag * b.real - a.real * b.imag) / denom
    };
    return c;
}

static inline ComplexDouble complex_double_conjugate(ComplexDouble a) {
    ComplexDouble c = {a.real, -a.imag};
    return c;
}

static inline double complex_double_abs(ComplexDouble a) {
    return sqrt(a.real * a.real + a.imag * a.imag);
}

static inline double complex_double_abs_squared(ComplexDouble a) {
    return a.real * a.real + a.imag * a.imag;
}

static inline double complex_double_arg(ComplexDouble a) {
    return atan2(a.imag, a.real);
}

static inline ComplexDouble complex_double_exp(ComplexDouble a) {
    double r = exp(a.real);
    ComplexDouble c = {r * cos(a.imag), r * sin(a.imag)};
    return c;
}

static inline ComplexDouble complex_double_sqrt(ComplexDouble a) {
    double r = sqrt(complex_double_abs(a));
    double theta = complex_double_arg(a) / 2.0;
    ComplexDouble c = {r * cos(theta), r * sin(theta)};
    return c;
}

// Comparison operations
static inline bool complex_double_equals(ComplexDouble a, ComplexDouble b) {
    return a.real == b.real && a.imag == b.imag;
}

static inline bool complex_double_less_than(ComplexDouble a, ComplexDouble b) {
    return complex_double_abs_squared(a) < complex_double_abs_squared(b);
}

static inline bool complex_double_greater_than(ComplexDouble a, ComplexDouble b) {
    return complex_double_abs_squared(a) > complex_double_abs_squared(b);
}

static inline bool complex_double_less_equal(ComplexDouble a, ComplexDouble b) {
    return complex_double_abs_squared(a) <= complex_double_abs_squared(b);
}

static inline bool complex_double_greater_equal(ComplexDouble a, ComplexDouble b) {
    return complex_double_abs_squared(a) >= complex_double_abs_squared(b);
}

// Conversion operations
static inline ComplexDouble complex_float_to_double(ComplexFloat a) {
    ComplexDouble c = {(double)a.real, (double)a.imag};
    return c;
}

static inline ComplexFloat complex_double_to_float(ComplexDouble a) {
    ComplexFloat c = {(float)a.real, (float)a.imag};
    return c;
}

#endif // QUANTUM_COMPLEX_H
