/**
 * @file numeric_utils.h
 * @brief Numeric utility functions for quantum geometric operations
 */

#ifndef QUANTUM_GEOMETRIC_NUMERIC_UTILS_H
#define QUANTUM_GEOMETRIC_NUMERIC_UTILS_H

#include <stddef.h>
#include <stdbool.h>
#include <math.h>
#include <complex.h>
#include <float.h>
#include "quantum_geometric/core/quantum_complex.h"

// Constants
#define QG_PI 3.14159265358979323846
#define QG_E  2.71828182845904523536
#define QG_EPSILON 1e-10
#define QG_MAX_ITERATIONS 1000

// Basic comparison functions
static inline size_t min(size_t a, size_t b) {
    return (a < b) ? a : b;
}

static inline size_t max(size_t a, size_t b) {
    return (a > b) ? a : b;
}

static inline double min_double(double a, double b) {
    return (a < b) ? a : b;
}

static inline double max_double(double a, double b) {
    return (a > b) ? a : b;
}

// Numeric checks
static inline bool is_zero(double x) {
    return fabs(x) < QG_EPSILON;
}

static inline bool is_equal(double a, double b) {
    return fabs(a - b) < QG_EPSILON;
}

static inline bool is_positive(double x) {
    return x > QG_EPSILON;
}

static inline bool is_negative(double x) {
    return x < -QG_EPSILON;
}

// Complex number operations
static inline double complex make_complex(double real, double imag) {
    return real + imag * I;
}

static inline double get_real(double complex z) {
    return creal(z);
}

static inline double get_imag(double complex z) {
    return cimag(z);
}

static inline double complex conjugate(double complex z) {
    return conj(z);
}

static inline double magnitude(double complex z) {
    return cabs(z);
}

static inline double phase(double complex z) {
    return carg(z);
}

// Vector operations
static inline double dot_product(const double* a, const double* b, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

static inline double complex dot_product_complex(const double complex* a,
                                               const double complex* b,
                                               size_t n) {
    double complex sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += a[i] * conjugate(b[i]);
    }
    return sum;
}

static inline double vector_norm(const double* v, size_t n) {
    return sqrt(dot_product(v, v, n));
}

static inline double vector_norm_complex(const double complex* v, size_t n) {
    return sqrt(creal(dot_product_complex(v, v, n)));
}

// Matrix operations
static bool __attribute__((unused)) matrix_multiply(const ComplexFloat* a,
                          const ComplexFloat* b,
                          ComplexFloat* result,
                          size_t m,
                          size_t n,
                          size_t p) {
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < p; j++) {
            ComplexFloat sum = complex_float_create(0.0f, 0.0f);
            for (size_t k = 0; k < n; k++) {
                sum = complex_float_add(sum, 
                      complex_float_multiply(a[i * n + k], b[k * p + j]));
            }
            result[i * p + j] = sum;
        }
    }
    return true;
}

// Quantum-specific operations
static inline double quantum_phase(double complex z) {
    return atan2(cimag(z), creal(z));
}

static inline double quantum_probability(double complex z) {
    return creal(z * conjugate(z));
}

static inline bool is_unitary(const double complex* matrix,
                            size_t n,
                            double tolerance) {
    // Check if matrix * conjugate_transpose = identity
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            double complex sum = 0.0;
            for (size_t k = 0; k < n; k++) {
                sum += matrix[i * n + k] * conjugate(matrix[j * n + k]);
            }
            if (i == j) {
                if (cabs(sum - 1.0) > tolerance) return false;
            } else {
                if (cabs(sum) > tolerance) return false;
            }
        }
    }
    return true;
}

// Numerical methods
static inline double newton_raphson(double (*f)(double),
                                  double (*df)(double),
                                  double x0,
                                  double tolerance,
                                  int max_iterations) {
    double x = x0;
    for (int i = 0; i < max_iterations; i++) {
        double fx = f(x);
        if (fabs(fx) < tolerance) {
            return x;
        }
        double dfx = df(x);
        if (is_zero(dfx)) {
            return x;
        }
        x = x - fx / dfx;
    }
    return x;
}

// Statistical functions
static inline double mean(const double* data, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum / n;
}

static inline double variance(const double* data, size_t n) {
    double m = mean(data, n);
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = data[i] - m;
        sum += diff * diff;
    }
    return sum / n;
}

static inline double standard_deviation(const double* data, size_t n) {
    return sqrt(variance(data, n));
}

// Utility functions
static inline size_t next_power_of_two(size_t n) {
    size_t power = 1;
    while (power < n) {
        power *= 2;
    }
    return power;
}

static inline bool is_power_of_two(size_t n) {
    return n && !(n & (n - 1));
}

static inline size_t gcd(size_t a, size_t b) {
    while (b != 0) {
        size_t t = b;
        b = a % b;
        a = t;
    }
    return a;
}

static inline size_t lcm(size_t a, size_t b) {
    return (a * b) / gcd(a, b);
}

#endif // QUANTUM_GEOMETRIC_NUMERIC_UTILS_H
