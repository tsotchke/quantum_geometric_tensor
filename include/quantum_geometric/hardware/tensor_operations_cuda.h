#ifndef TENSOR_OPERATIONS_CUDA_H
#define TENSOR_OPERATIONS_CUDA_H

#include "quantum_geometric/core/quantum_types.h"

// Transformation types for tensor operations
typedef enum {
    TRANSFORM_QUANTUM,    // Quantum-inspired non-linear transformation
    TRANSFORM_GEOMETRIC,  // Geometric transformation
    TRANSFORM_ATTENTION  // Attention-based transformation
} TransformType;

// Initialize and cleanup CUDA resources
void cleanup_cuda_resources(void);

// Optimized tensor multiplication with automatic algorithm selection
void cuda_tensor_multiply(QuantumAmplitude* C, 
                         const QuantumAmplitude* A, 
                         const QuantumAmplitude* B, 
                         int size);

// Optimized tensor transformation
void cuda_tensor_transform(QuantumAmplitude* data, 
                         int size, 
                         TransformType type);

#endif // TENSOR_OPERATIONS_CUDA_H
