#include "quantum_geometric/core/numerical_operations.h"
#include "quantum_geometric/core/simd_operations.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

void qgt_vector_add(double complex* dst, const double complex* a, const double complex* b, size_t n) {
    simd_complex_add(dst, a, b, n);
}

void qgt_vector_scale(double complex* dst, const double complex* src, double scale, size_t n) {
    simd_complex_scale(dst, src, scale, n);
}

double complex qgt_vector_dot(const double complex* a, const double complex* b, size_t n) {
    double complex result = 0;
    
    // Use AVX-512 for complex dot product
    __m512d sum = _mm512_setzero_pd();
    for (size_t i = 0; i + 4 <= n; i += 4) {
        __m512d va = _mm512_load_pd((double*)&a[i]);
        __m512d vb = _mm512_load_pd((double*)&b[i]);
        __m512d prod = _mm512_complex_mul_pd(va, vb);
        sum = _mm512_add_pd(sum, prod);
    }
    
    // Reduce sum across vector
    double complex partial;
    _mm512_store_pd((double*)&partial, sum);
    result += partial;
    
    // Handle remaining elements
    for (size_t i = (n/4)*4; i < n; i++) {
        result += a[i] * b[i];
    }
    
    return result;
}

double qgt_vector_norm(const double complex* vec, size_t n) {
    double norm = 0.0;
    
    // Use AVX-512 for squared norm calculation
    __m512d sum = _mm512_setzero_pd();
    for (size_t i = 0; i + 4 <= n; i += 4) {
        __m512d v = _mm512_load_pd((double*)&vec[i]);
        __m512d squared = _mm512_mul_pd(v, v);
        sum = _mm512_add_pd(sum, squared);
    }
    
    // Reduce sum across vector
    norm += _mm512_reduce_add_pd(sum);
    
    // Handle remaining elements
    for (size_t i = (n/4)*4; i < n; i++) {
        norm += qgt_abs_squared(vec[i]);
    }
    
    return sqrt(norm);
}

void qgt_vector_normalize(double complex* vec, size_t n) {
    double norm = qgt_vector_norm(vec, n);
    if (norm > 1e-10) {
        qgt_vector_scale(vec, vec, 1.0/norm, n);
    }
}

void qgt_tensor_contract(double complex* dst, const double complex* a, const double complex* b,
                        size_t* a_dims, size_t* b_dims, size_t* contract_dims,
                        size_t a_rank, size_t b_rank, size_t num_contract) {
    // Calculate output dimensions
    size_t out_rank = a_rank + b_rank - 2*num_contract;
    size_t* out_dims = malloc(out_rank * sizeof(size_t));
    size_t out_size = 1;
    
    // Copy non-contracted dimensions to output
    size_t out_idx = 0;
    for (size_t i = 0; i < a_rank; i++) {
        int is_contracted = 0;
        for (size_t j = 0; j < num_contract; j++) {
            if (i == contract_dims[j]) {
                is_contracted = 1;
                break;
            }
        }
        if (!is_contracted) {
            out_dims[out_idx++] = a_dims[i];
            out_size *= a_dims[i];
        }
    }
    for (size_t i = 0; i < b_rank; i++) {
        int is_contracted = 0;
        for (size_t j = 0; j < num_contract; j++) {
            if (i == contract_dims[num_contract + j]) {
                is_contracted = 1;
                break;
            }
        }
        if (!is_contracted) {
            out_dims[out_idx++] = b_dims[i];
            out_size *= b_dims[i];
        }
    }
    
    // Initialize output tensor
    memset(dst, 0, out_size * sizeof(double complex));
    
    // Calculate strides
    size_t* a_strides = malloc(a_rank * sizeof(size_t));
    size_t* b_strides = malloc(b_rank * sizeof(size_t));
    size_t stride = 1;
    for (size_t i = a_rank; i > 0; i--) {
        a_strides[i-1] = stride;
        stride *= a_dims[i-1];
    }
    stride = 1;
    for (size_t i = b_rank; i > 0; i--) {
        b_strides[i-1] = stride;
        stride *= b_dims[i-1];
    }
    
    // Perform contraction
    // This is a naive implementation - could be optimized further with SIMD
    size_t contract_size = 1;
    for (size_t i = 0; i < num_contract; i++) {
        contract_size *= a_dims[contract_dims[i]];
    }
    
    for (size_t i = 0; i < out_size; i++) {
        size_t a_idx = 0;
        size_t b_idx = 0;
        size_t tmp = i;
        
        // Calculate indices for output -> input mapping
        for (size_t j = 0; j < out_rank; j++) {
            size_t dim_idx = tmp % out_dims[j];
            tmp /= out_dims[j];
            if (j < a_rank - num_contract) {
                a_idx += dim_idx * a_strides[j];
            } else {
                b_idx += dim_idx * b_strides[j - (a_rank - num_contract)];
            }
        }
        
        // Contract along specified dimensions
        for (size_t j = 0; j < contract_size; j++) {
            size_t contract_a_idx = a_idx;
            size_t contract_b_idx = b_idx;
            size_t tmp = j;
            
            for (size_t k = 0; k < num_contract; k++) {
                size_t dim_idx = tmp % a_dims[contract_dims[k]];
                tmp /= a_dims[contract_dims[k]];
                contract_a_idx += dim_idx * a_strides[contract_dims[k]];
                contract_b_idx += dim_idx * b_strides[contract_dims[num_contract + k]];
            }
            
            dst[i] += a[contract_a_idx] * b[contract_b_idx];
        }
    }
    
    free(out_dims);
    free(a_strides);
    free(b_strides);
}

double qgt_abs_squared(double complex z) {
    return creal(z) * creal(z) + cimag(z) * cimag(z);
}

double qgt_phase(double complex z) {
    return atan2(cimag(z), creal(z));
}

double complex qgt_polar(double r, double theta) {
    return r * (cos(theta) + I * sin(theta));
}

void qgt_sincos_table(double* sin_table, double* cos_table, size_t n) {
    double angle_step = 2.0 * M_PI / n;
    for (size_t i = 0; i < n; i++) {
        double angle = i * angle_step;
        sin_table[i] = sin(angle);
        cos_table[i] = cos(angle);
    }
}
