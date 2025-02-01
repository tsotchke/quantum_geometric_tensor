#ifndef QGT_SIMD_OPERATIONS_H
#define QGT_SIMD_OPERATIONS_H

#include "quantum_geometric/core/quantum_complex.h"
#include <stddef.h>

// SIMD operations for complex numbers
void simd_complex_multiply_accumulate(ComplexFloat* result, 
                                    const ComplexFloat* a,
                                    const ComplexFloat* b,
                                    size_t length);

double simd_complex_norm(const ComplexFloat* vec, size_t length);

void simd_complex_scale(ComplexFloat* result,
                       const ComplexFloat* vec,
                       ComplexFloat scale,
                       size_t length);

#endif // QGT_SIMD_OPERATIONS_H
