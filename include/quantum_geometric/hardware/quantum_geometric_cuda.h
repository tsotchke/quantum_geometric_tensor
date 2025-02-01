#ifndef QUANTUM_GEOMETRIC_CUDA_H
#define QUANTUM_GEOMETRIC_CUDA_H

#include "quantum_geometric/core/quantum_complex.h"
#include "quantum_geometric/core/quantum_geometric_constants.h"

// Quantum amplitude type using our ComplexFloat
struct QuantumAmplitude {
    ComplexFloat amplitude;
};

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include <cuComplex.h>

// Convert ComplexFloat to cuDoubleComplex
__device__ __host__ inline cuDoubleComplex to_cuda_complex(ComplexFloat c) {
    return make_cuDoubleComplex((double)c.real, (double)c.imag);
}

// Convert cuDoubleComplex to ComplexFloat
__device__ __host__ inline ComplexFloat from_cuda_complex(cuDoubleComplex c) {
    return (ComplexFloat){(float)cuCreal(c), (float)cuCimag(c)};
}

#else
// Stub implementations for non-CUDA builds
inline ComplexFloat to_cuda_complex(ComplexFloat c) {
    return c;  // No-op when CUDA is disabled
}

inline ComplexFloat from_cuda_complex(ComplexFloat c) {
    return c;  // No-op when CUDA is disabled
}
#endif // ENABLE_CUDA

#endif // QUANTUM_GEOMETRIC_CUDA_H
