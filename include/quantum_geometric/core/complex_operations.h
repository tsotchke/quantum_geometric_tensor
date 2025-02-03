#ifndef COMPLEX_OPERATIONS_H
#define COMPLEX_OPERATIONS_H

#include "quantum_geometric/core/quantum_geometric_types.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

// Complex number operations
static inline ComplexFloat complex_add(ComplexFloat a, ComplexFloat b) {
    return (ComplexFloat){
        a.real + b.real,
        a.imag + b.imag
    };
}

static inline ComplexFloat complex_sub(ComplexFloat a, ComplexFloat b) {
    return (ComplexFloat){
        a.real - b.real,
        a.imag - b.imag
    };
}

static inline ComplexFloat complex_mul(ComplexFloat a, ComplexFloat b) {
    return (ComplexFloat){
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real
    };
}

static inline ComplexFloat complex_div(ComplexFloat a, ComplexFloat b) {
    double denom = b.real * b.real + b.imag * b.imag;
    return (ComplexFloat){
        (a.real * b.real + a.imag * b.imag) / denom,
        (a.imag * b.real - a.real * b.imag) / denom
    };
}

static inline ComplexFloat complex_conj(ComplexFloat a) {
    return (ComplexFloat){a.real, -a.imag};
}

static inline double complex_abs_sq(ComplexFloat a) {
    return a.real * a.real + a.imag * a.imag;
}

static inline double complex_abs(ComplexFloat a) {
    return sqrt(complex_abs_sq(a));
}

static inline ComplexFloat complex_exp(ComplexFloat a) {
    double exp_real = exp(a.real);
    return (ComplexFloat){
        exp_real * cos(a.imag),
        exp_real * sin(a.imag)
    };
}

static inline ComplexFloat complex_sqrt(ComplexFloat a) {
    double r = complex_abs(a);
    double theta = atan2(a.imag, a.real) / 2.0;
    double sqrt_r = sqrt(r);
    return (ComplexFloat){
        sqrt_r * cos(theta),
        sqrt_r * sin(theta)
    };
}

static inline ComplexFloat complex_scale(ComplexFloat a, double scale) {
    return (ComplexFloat){
        a.real * scale,
        a.imag * scale
    };
}

static inline ComplexFloat complex_scale_complex(ComplexFloat a, ComplexFloat scale) {
    return complex_mul(a, scale);
}

// Complex vector operations
static inline ComplexFloat complex_dot(
    const ComplexFloat* a,
    const ComplexFloat* b,
    size_t n) {
    
    ComplexFloat sum = {0, 0};
    for (size_t i = 0; i < n; i++) {
        sum = complex_add(sum, complex_mul(a[i], complex_conj(b[i])));
    }
    return sum;
}

static inline void complex_vec_scale(
    ComplexFloat* a,
    ComplexFloat scale,
    size_t n) {
    
    for (size_t i = 0; i < n; i++) {
        a[i] = complex_mul(a[i], scale);
    }
}

static inline double complex_norm(
    const ComplexFloat* a,
    size_t n) {
    
    return sqrt(complex_abs_sq(complex_dot(a, a, n)));
}

static inline void complex_normalize(
    ComplexFloat* a,
    size_t n) {
    
    double norm = complex_norm(a, n);
    if (norm > 0) {
        ComplexFloat scale = {1.0 / norm, 0};
        complex_vec_scale(a, scale, n);
    }
}

#ifdef __cplusplus
}
#endif

#endif // COMPLEX_OPERATIONS_H
