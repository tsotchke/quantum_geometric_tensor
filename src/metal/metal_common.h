#ifndef METAL_COMMON_H
#define METAL_COMMON_H

#include <metal_stdlib>
using namespace metal;

// Complex number type for quantum operations
struct ComplexFloat {
    float real;
    float imag;
};

// Common geometric tensor types
struct GeometricTensor {
    float4x4 metric;
    float4x4 connection;
    float4x4 curvature;
    uint dimensions;
    uint rank;
    bool is_symmetric;
};

// AMX configuration for Apple Silicon
struct AMXConfig {
    uint block_size;
    uint num_blocks;
    uint precision;
    bool use_fp16;
    bool use_simd;
};

// Batch processing configuration
struct BatchConfig {
    uint batch_size;
    uint input_size;
    uint output_size;
    uint hidden_size;
};

// Common math functions
inline float4x4 matrix_multiply(float4x4 a, float4x4 b) {
    float4x4 result;
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            result[i][j] = 0;
            for (uint k = 0; k < 4; k++) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return result;
}

// Bit counting for performance metrics
inline uint count_ones_uint(uint x) {
    uint count = 0;
    while (x) {
        count += x & 1;
        x >>= 1;
    }
    return count;
}

inline uint count_ones_float4(float4 v) {
    uint4 bits = as_type<uint4>(v);
    return count_ones_uint(bits.x) + 
           count_ones_uint(bits.y) + 
           count_ones_uint(bits.z) + 
           count_ones_uint(bits.w);
}

inline uint count_ones_float4x4(float4x4 m) {
    return count_ones_float4(m[0]) +
           count_ones_float4(m[1]) +
           count_ones_float4(m[2]) +
           count_ones_float4(m[3]);
}

#endif // METAL_COMMON_H
