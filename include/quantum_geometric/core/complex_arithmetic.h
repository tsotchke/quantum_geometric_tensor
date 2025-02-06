#ifndef COMPLEX_ARITHMETIC_H
#define COMPLEX_ARITHMETIC_H

#include <stddef.h>
#include "quantum_geometric/core/quantum_complex.h"
#include <stdbool.h>

// Basic complex arithmetic operations
ComplexFloat complex_add(ComplexFloat a, ComplexFloat b);
ComplexFloat complex_subtract(ComplexFloat a, ComplexFloat b);
ComplexFloat complex_multiply(ComplexFloat a, ComplexFloat b);
ComplexFloat complex_divide(ComplexFloat a, ComplexFloat b);
ComplexFloat complex_conjugate(ComplexFloat a);

// Complex number properties
float complex_abs(ComplexFloat a);
float complex_arg(ComplexFloat a);
float complex_abs_squared(ComplexFloat a);

// Complex vector operations
void complex_vector_add(const ComplexFloat* a, const ComplexFloat* b,
                       ComplexFloat* result, size_t length);
void complex_vector_subtract(const ComplexFloat* a, const ComplexFloat* b,
                           ComplexFloat* result, size_t length);
void complex_vector_scale(const ComplexFloat* a, ComplexFloat scale,
                         ComplexFloat* result, size_t length);
ComplexFloat complex_vector_dot(const ComplexFloat* a, const ComplexFloat* b,
                              size_t length);

// Complex matrix operations
void complex_matrix_transpose(const ComplexFloat* a,
                            ComplexFloat* result,
                            size_t rows, size_t cols);
void complex_matrix_conjugate_transpose(const ComplexFloat* a,
                                      ComplexFloat* result,
                                      size_t rows, size_t cols);

// Type conversion utilities
#ifdef __APPLE__
typedef struct { float real; float imag; } DSPComplex;
typedef struct { double real; double imag; } DSPDoubleComplex;
typedef struct { float real; float imag; } __CLPK_complex;

DSPComplex to_dsp_complex(ComplexFloat a);
ComplexFloat from_dsp_complex(DSPComplex a);
__CLPK_complex to_lapack_complex(ComplexFloat a);
ComplexFloat from_lapack_complex(__CLPK_complex a);
#endif

// OpenBLAS complex type conversions
#ifdef HAVE_OPENBLAS
typedef struct { float real; float imag; } openblas_complex_float;
openblas_complex_float to_openblas_complex(ComplexFloat a);
ComplexFloat from_openblas_complex(openblas_complex_float a);
#endif

// MKL complex type conversions
#ifdef HAVE_MKL
typedef struct { float real; float imag; } MKL_Complex8;
MKL_Complex8 to_mkl_complex(ComplexFloat a);
ComplexFloat from_mkl_complex(MKL_Complex8 a);
#endif

// Advanced complex functions
ComplexFloat complex_sqrt(ComplexFloat a);
ComplexFloat complex_exp(ComplexFloat a);
ComplexFloat complex_log(ComplexFloat a);
ComplexFloat complex_pow(ComplexFloat a, ComplexFloat b);

// Utility functions
bool complex_is_zero(ComplexFloat a);
bool complex_is_real(ComplexFloat a);
bool complex_is_equal(ComplexFloat a, ComplexFloat b);
ComplexFloat complex_from_polar(float r, float theta);
void complex_to_polar(ComplexFloat a, float* r, float* theta);

#endif // COMPLEX_ARITHMETIC_H
