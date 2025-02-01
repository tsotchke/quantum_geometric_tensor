#include "quantum_geometric/core/tensor_operations.h"
#include "quantum_geometric/core/quantum_geometric_logging.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Basic tensor operations
bool qg_tensor_add(tensor_t* result, const tensor_t* a, const tensor_t* b) {
    if (!result || !a || !b || !a->data || !b->data) {
        geometric_log_error("Invalid parameters in qg_tensor_add");
        return false;
    }

    if (a->rank != b->rank) {
        geometric_log_error("Tensor ranks do not match");
        return false;
    }

    for (size_t i = 0; i < a->rank; i++) {
        if (a->dimensions[i] != b->dimensions[i]) {
            geometric_log_error("Tensor dimensions do not match");
            return false;
        }
    }

    #pragma omp parallel for
    for (size_t i = 0; i < a->total_size; i++) {
        result->data[i].real = a->data[i].real + b->data[i].real;
        result->data[i].imag = a->data[i].imag + b->data[i].imag;
    }

    return true;
}

bool qg_tensor_scale(tensor_t* tensor, float scalar) {
    if (!tensor || !tensor->data) {
        geometric_log_error("Invalid parameters in qg_tensor_scale");
        return false;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < tensor->total_size; i++) {
        tensor->data[i].real *= scalar;
        tensor->data[i].imag *= scalar;
    }

    return true;
}

bool qg_tensor_reshape(ComplexFloat* result, const ComplexFloat* data, 
                      const size_t* new_dimensions, size_t rank) {
    if (!result || !data || !new_dimensions || rank == 0) {
        geometric_log_error("Invalid parameters in qg_tensor_reshape");
        return false;
    }

    // Copy data directly since memory layout doesn't change
    size_t total_size = 1;
    for (size_t i = 0; i < rank; i++) {
        total_size *= new_dimensions[i];
    }
    memcpy(result, data, total_size * sizeof(ComplexFloat));
    return true;
}

size_t qg_tensor_get_size(const size_t* dimensions, size_t rank) {
    if (!dimensions || rank == 0) {
        return 0;
    }
    
    size_t total_size = 1;
    for (size_t i = 0; i < rank; i++) {
        total_size *= dimensions[i];
    }
    return total_size;
}

bool qg_tensor_transpose(ComplexFloat* result, const ComplexFloat* data,
                        const size_t* dimensions, size_t rank,
                        const size_t* permutation) {
    if (!result || !data || !dimensions || !permutation || rank == 0) {
        geometric_log_error("Invalid parameters in qg_tensor_transpose");
        return false;
    }

    // Calculate strides for input and output
    size_t* in_strides = (size_t*)malloc(rank * sizeof(size_t));
    size_t* out_strides = (size_t*)malloc(rank * sizeof(size_t));
    size_t* out_dims = (size_t*)malloc(rank * sizeof(size_t));
    
    if (!in_strides || !out_strides || !out_dims) {
        free(in_strides);
        free(out_strides);
        free(out_dims);
        geometric_log_error("Memory allocation failed");
        return false;
    }

    // Calculate input strides
    in_strides[rank - 1] = 1;
    for (size_t i = rank - 1; i > 0; i--) {
        in_strides[i - 1] = in_strides[i] * dimensions[i];
    }

    // Calculate output dimensions and strides
    for (size_t i = 0; i < rank; i++) {
        out_dims[i] = dimensions[permutation[i]];
    }
    out_strides[rank - 1] = 1;
    for (size_t i = rank - 1; i > 0; i--) {
        out_strides[i - 1] = out_strides[i] * out_dims[i];
    }

    // Calculate total size
    size_t total_size = 1;
    for (size_t i = 0; i < rank; i++) {
        total_size *= dimensions[i];
    }

    // Perform transpose
    #pragma omp parallel for
    for (size_t i = 0; i < total_size; i++) {
        size_t old_idx = 0;
        size_t idx = i;
        
        // Convert linear index to multidimensional indices
        for (size_t j = 0; j < rank; j++) {
            size_t dim_idx = idx / out_strides[j];
            idx %= out_strides[j];
            old_idx += dim_idx * in_strides[permutation[j]];
        }
        
        result[i] = data[old_idx];
    }

    free(in_strides);
    free(out_strides);
    free(out_dims);
    return true;
}

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __AVX512F__
#include <immintrin.h>
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

// SIMD configuration
#if defined(__AVX512F__)
#define SIMD_WIDTH 16  // 512 bits = 16 floats
#elif defined(__ARM_NEON)
#define SIMD_WIDTH 4   // 128 bits = 4 floats
#else
#define SIMD_WIDTH 1   // No SIMD
#endif

// Helper function declarations
static inline size_t min(size_t a, size_t b) {
    return (a < b) ? a : b;
}

static inline size_t max(size_t a, size_t b) {
    return (a > b) ? a : b;
}

// Forward declarations for matrix operations
static void matrix_add(ComplexFloat* result, const ComplexFloat* a, const ComplexFloat* b, size_t rows, size_t cols, size_t stride);
static void matrix_subtract(ComplexFloat* result, const ComplexFloat* a, const ComplexFloat* b, size_t rows, size_t cols, size_t stride);

// Matrix operation declarations
static void matrix_add(ComplexFloat* result, const ComplexFloat* a, const ComplexFloat* b, 
                      size_t rows, size_t cols, size_t stride) {
    #ifdef __AVX512F__
    const size_t simd_width = 8;  // Process 8 complex numbers at a time
    
    for (size_t i = 0; i < rows; i++) {
        size_t j;
        for (j = 0; j + simd_width <= cols; j += simd_width) {
            __m512 va_real = _mm512_loadu_ps((const float*)&a[i * stride + j].real);
            __m512 va_imag = _mm512_loadu_ps((const float*)&a[i * stride + j].imag);
            __m512 vb_real = _mm512_loadu_ps((const float*)&b[i * stride + j].real);
            __m512 vb_imag = _mm512_loadu_ps((const float*)&b[i * stride + j].imag);
            
            __m512 vr_real = _mm512_add_ps(va_real, vb_real);
            __m512 vr_imag = _mm512_add_ps(va_imag, vb_imag);
            
            _mm512_storeu_ps((float*)&result[i * cols + j].real, vr_real);
            _mm512_storeu_ps((float*)&result[i * cols + j].imag, vr_imag);
        }
        
        // Handle remaining elements
        for (; j < cols; j++) {
            result[i * cols + j].real = a[i * stride + j].real + b[i * stride + j].real;
            result[i * cols + j].imag = a[i * stride + j].imag + b[i * stride + j].imag;
        }
    }
    
    #elif defined(__ARM_NEON)
    const size_t simd_width = 2;  // Process 2 complex numbers at a time
    
    for (size_t i = 0; i < rows; i++) {
        size_t j;
        for (j = 0; j + simd_width <= cols; j += simd_width) {
            // Load real and imaginary parts separately
            float32x2_t va_real = vld1_f32((const float*)&a[i * stride + j].real);
            float32x2_t va_imag = vld1_f32((const float*)&a[i * stride + j].imag);
            float32x2_t vb_real = vld1_f32((const float*)&b[i * stride + j].real);
            float32x2_t vb_imag = vld1_f32((const float*)&b[i * stride + j].imag);
            
            // Add real and imaginary parts separately
            float32x2_t vr_real = vadd_f32(va_real, vb_real);
            float32x2_t vr_imag = vadd_f32(va_imag, vb_imag);
            
            // Store results
            vst1_f32((float*)&result[i * cols + j].real, vr_real);
            vst1_f32((float*)&result[i * cols + j].imag, vr_imag);
        }
        
        // Handle remaining elements
        for (; j < cols; j++) {
            result[i * cols + j].real = a[i * stride + j].real + b[i * stride + j].real;
            result[i * cols + j].imag = a[i * stride + j].imag + b[i * stride + j].imag;
        }
    }
    
    #else
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result[i * cols + j].real = a[i * stride + j].real + b[i * stride + j].real;
            result[i * cols + j].imag = a[i * stride + j].imag + b[i * stride + j].imag;
        }
    }
    #endif
}

static void matrix_subtract(ComplexFloat* result, const ComplexFloat* a, const ComplexFloat* b,
                          size_t rows, size_t cols, size_t stride) {
    #ifdef __AVX512F__
    const size_t simd_width = 8;  // Process 8 complex numbers at a time
    
    for (size_t i = 0; i < rows; i++) {
        size_t j;
        for (j = 0; j + simd_width <= cols; j += simd_width) {
            __m512 va_real = _mm512_loadu_ps((const float*)&a[i * stride + j].real);
            __m512 va_imag = _mm512_loadu_ps((const float*)&a[i * stride + j].imag);
            __m512 vb_real = _mm512_loadu_ps((const float*)&b[i * stride + j].real);
            __m512 vb_imag = _mm512_loadu_ps((const float*)&b[i * stride + j].imag);
            
            __m512 vr_real = _mm512_sub_ps(va_real, vb_real);
            __m512 vr_imag = _mm512_sub_ps(va_imag, vb_imag);
            
            _mm512_storeu_ps((float*)&result[i * cols + j].real, vr_real);
            _mm512_storeu_ps((float*)&result[i * cols + j].imag, vr_imag);
        }
        
        // Handle remaining elements
        for (; j < cols; j++) {
            result[i * cols + j].real = a[i * stride + j].real - b[i * stride + j].real;
            result[i * cols + j].imag = a[i * stride + j].imag - b[i * stride + j].imag;
        }
    }
    
    #elif defined(__ARM_NEON)
    const size_t simd_width = 2;  // Process 2 complex numbers at a time
    
    for (size_t i = 0; i < rows; i++) {
        size_t j;
        for (j = 0; j + simd_width <= cols; j += simd_width) {
            // Load real and imaginary parts separately
            float32x2_t va_real = vld1_f32((const float*)&a[i * stride + j].real);
            float32x2_t va_imag = vld1_f32((const float*)&a[i * stride + j].imag);
            float32x2_t vb_real = vld1_f32((const float*)&b[i * stride + j].real);
            float32x2_t vb_imag = vld1_f32((const float*)&b[i * stride + j].imag);
            
            // Subtract real and imaginary parts separately
            float32x2_t vr_real = vsub_f32(va_real, vb_real);
            float32x2_t vr_imag = vsub_f32(va_imag, vb_imag);
            
            // Store results
            vst1_f32((float*)&result[i * cols + j].real, vr_real);
            vst1_f32((float*)&result[i * cols + j].imag, vr_imag);
        }
        
        // Handle remaining elements
        for (; j < cols; j++) {
            result[i * cols + j].real = a[i * stride + j].real - b[i * stride + j].real;
            result[i * cols + j].imag = a[i * stride + j].imag - b[i * stride + j].imag;
        }
    }
    
    #else
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result[i * cols + j].real = a[i * stride + j].real - b[i * stride + j].real;
            result[i * cols + j].imag = a[i * stride + j].imag - b[i * stride + j].imag;
        }
    }
    #endif
}

bool qg_tensor_init(tensor_t* tensor, const size_t* dimensions, size_t rank) {
    if (!tensor || !dimensions || rank == 0) {
        geometric_log_error("Invalid parameters in qg_tensor_init");
        return false;
    }
    
    // Calculate total size first to avoid overflow
    size_t total_size = 1;
    for (size_t i = 0; i < rank; i++) {
        if (dimensions[i] == 0) {
            geometric_log_error("Invalid dimension size 0");
            return false;
        }
        // Check for overflow
        if (total_size > SIZE_MAX / dimensions[i]) {
            geometric_log_error("Tensor size too large - would overflow");
            return false;
        }
        total_size *= dimensions[i];
    }

    // Allocate dimensions array
    size_t* new_dimensions = (size_t*)malloc(rank * sizeof(size_t));
    if (!new_dimensions) {
        geometric_log_error("Memory allocation failed for tensor dimensions");
        return false;
    }

    // Copy dimensions
    memcpy(new_dimensions, dimensions, rank * sizeof(size_t));

    // Initialize the tensor struct
    tensor->dimensions = new_dimensions;
    tensor->data = (ComplexFloat*)malloc(total_size * sizeof(ComplexFloat));
    if (!tensor->data) {
        geometric_log_error("Memory allocation failed for tensor data");
        free(new_dimensions);
        return false;
    }

    // Initialize all elements to zero using SIMD when available
    #if defined(__AVX512F__)
    const size_t simd_width = 8;  // Process 8 complex numbers at a time
    size_t simd_count = total_size / simd_width * simd_width;
    __m512 vzero = _mm512_setzero_ps();
    
    for (size_t i = 0; i < simd_count; i += simd_width) {
        _mm512_storeu_ps((float*)&tensor->data[i].real, vzero);
        _mm512_storeu_ps((float*)&tensor->data[i].imag, vzero);
    }
    
    // Handle remaining elements
    for (size_t i = simd_count; i < total_size; i++) {
        tensor->data[i] = COMPLEX_FLOAT_ZERO;
    }
    
    #elif defined(__ARM_NEON)
    const size_t simd_width = 2;  // Process 2 complex numbers at a time
    size_t simd_count = total_size / simd_width * simd_width;
    float32x2_t vzero = vdup_n_f32(0.0f);
    
    for (size_t i = 0; i < simd_count; i += simd_width) {
        vst1_f32(&tensor->data[i].real, vzero);
        vst1_f32(&tensor->data[i].imag, vzero);
    }
    
    // Handle remaining elements
    for (size_t i = simd_count; i < total_size; i++) {
        tensor->data[i] = COMPLEX_FLOAT_ZERO;
    }
    
    #else
    // Scalar initialization with prefetching
    const size_t prefetch_distance = 8;
    for (size_t i = 0; i < total_size; i++) {
        if (i + prefetch_distance < total_size) {
            __builtin_prefetch(&tensor->data[i + prefetch_distance], 1, 3);
        }
        tensor->data[i] = COMPLEX_FLOAT_ZERO;
    }
    #endif

    tensor->rank = rank;
    tensor->total_size = total_size;
    tensor->is_contiguous = true;
    tensor->strides = NULL;  // Will be initialized if needed
    tensor->owns_data = true;
    tensor->device = NULL;   // CPU tensor by default
    tensor->auxiliary_data = NULL;

    return true;
}

void qg_tensor_cleanup(tensor_t* tensor) {
    if (!tensor) {
        return;
    }
    if (tensor->data && tensor->owns_data) {
        free(tensor->data);
        tensor->data = NULL;
    }
    if (tensor->dimensions) {
        free(tensor->dimensions);
        tensor->dimensions = NULL;
    }
    if (tensor->strides) {
        free(tensor->strides);
        tensor->strides = NULL;
    }
    if (tensor->auxiliary_data) {
        free(tensor->auxiliary_data);
        tensor->auxiliary_data = NULL;
    }
    tensor->rank = 0;
    tensor->total_size = 0;
    tensor->is_contiguous = true;
    tensor->owns_data = false;
    tensor->device = NULL;
}

// Cache-aligned memory allocation
static inline void* aligned_malloc(size_t size) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, 64, size) != 0) {
        return NULL;
    }
    return ptr;
}

// Cache-friendly block sizes for different cache levels
#define L1_BLOCK_SIZE 32
#define L2_BLOCK_SIZE 128
#define L3_BLOCK_SIZE 512

// Optimized matrix multiplication with multi-level blocking
static void matrix_multiply_blocked(ComplexFloat* C, const ComplexFloat* A, const ComplexFloat* B,
                                  size_t M, size_t N, size_t K) {
    // Initialize result to zero
    for (size_t i = 0; i < M * N; i++) {
        C[i] = COMPLEX_FLOAT_ZERO;
    }

    // Allocate aligned temporary buffers for blocks
    ComplexFloat* block_A = (ComplexFloat*)aligned_malloc(L1_BLOCK_SIZE * L1_BLOCK_SIZE * sizeof(ComplexFloat));
    ComplexFloat* block_B = (ComplexFloat*)aligned_malloc(L1_BLOCK_SIZE * L1_BLOCK_SIZE * sizeof(ComplexFloat));
    ComplexFloat* block_C = (ComplexFloat*)aligned_malloc(L1_BLOCK_SIZE * L1_BLOCK_SIZE * sizeof(ComplexFloat));

    if (!block_A || !block_B || !block_C) {
        geometric_log_error("Memory allocation failed for matrix blocks");
        free(block_A);
        free(block_B);
        free(block_C);
        return;
    }

    // Multi-level blocking for better cache utilization
    #pragma omp parallel for collapse(3) schedule(guided)
    for (size_t i0 = 0; i0 < M; i0 += L2_BLOCK_SIZE) {
        for (size_t j0 = 0; j0 < N; j0 += L2_BLOCK_SIZE) {
            for (size_t k0 = 0; k0 < K; k0 += L2_BLOCK_SIZE) {
                // L2 cache blocks
                size_t max_i2 = min(i0 + L2_BLOCK_SIZE, M);
                size_t max_j2 = min(j0 + L2_BLOCK_SIZE, N);
                size_t max_k2 = min(k0 + L2_BLOCK_SIZE, K);

                // L1 cache blocks
                for (size_t i1 = i0; i1 < max_i2; i1 += L1_BLOCK_SIZE) {
                    for (size_t j1 = j0; j1 < max_j2; j1 += L1_BLOCK_SIZE) {
                        for (size_t k1 = k0; k1 < max_k2; k1 += L1_BLOCK_SIZE) {
                            size_t max_i = min(i1 + L1_BLOCK_SIZE, max_i2);
                            size_t max_j = min(j1 + L1_BLOCK_SIZE, max_j2);
                            size_t max_k = min(k1 + L1_BLOCK_SIZE, max_k2);

                            // Load blocks with aggressive prefetching
                            for (size_t i = i1; i < max_i; i++) {
                                for (size_t k = k1; k < max_k; k++) {
                                    block_A[(i-i1) * L1_BLOCK_SIZE + (k-k1)] = A[i * K + k];
                                    if (k + L1_BLOCK_SIZE < max_k) {
                                        __builtin_prefetch(&A[i * K + k + L1_BLOCK_SIZE], 0, 3);
                                    }
                                }
                            }

                            for (size_t k = k1; k < max_k; k++) {
                                for (size_t j = j1; j < max_j; j++) {
                                    block_B[(k-k1) * L1_BLOCK_SIZE + (j-j1)] = B[k * N + j];
                                    if (j + L1_BLOCK_SIZE < max_j) {
                                        __builtin_prefetch(&B[k * N + j + L1_BLOCK_SIZE], 0, 3);
                                    }
                                }
                            }

                            // Process block with vectorization
                            for (size_t i = 0; i < max_i-i1; i++) {
                                for (size_t j = 0; j < max_j-j1; j++) {
                                    ComplexFloat sum = COMPLEX_FLOAT_ZERO;
                                    for (size_t k = 0; k < max_k-k1; k++) {
                                        ComplexFloat a = block_A[i * L1_BLOCK_SIZE + k];
                                        ComplexFloat b = block_B[k * L1_BLOCK_SIZE + j];
                                        // Complex multiplication: (a.r + a.i*i)(b.r + b.i*i)
                                        sum.real += a.real * b.real - a.imag * b.imag;
                                        sum.imag += a.real * b.imag + a.imag * b.real;
                                    }
                                    size_t idx = (i+i1) * N + (j+j1);
                                    C[idx].real += sum.real;
                                    C[idx].imag += sum.imag;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    free(block_A);
    free(block_B);
    free(block_C);
}

// Strassen algorithm threshold
#define STRASSEN_THRESHOLD 512

// Numerical stability threshold
#define STABILITY_THRESHOLD 1e-7

static bool is_unitary(const ComplexFloat* tensor, size_t dim) {
    // Check if tensor is unitary by verifying T^† T = I
    ComplexFloat* temp = (ComplexFloat*)aligned_malloc(dim * dim * sizeof(ComplexFloat));
    if (!temp) return false;
    
    bool is_unit = true;
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            ComplexFloat sum = COMPLEX_FLOAT_ZERO;
            for (size_t k = 0; k < dim; k++) {
                // For conjugate transpose, we conjugate the first term
                ComplexFloat conj = {tensor[k * dim + i].real, -tensor[k * dim + i].imag};
                ComplexFloat b = tensor[k * dim + j];
                // Complex multiplication
                sum.real += conj.real * b.real - conj.imag * b.imag;
                sum.imag += conj.real * b.imag + conj.imag * b.real;
            }
            temp[i * dim + j] = sum;
            if (i == j) {
                // Diagonal elements should be 1
                if (fabsf(sum.real - 1.0f) > STABILITY_THRESHOLD || 
                    fabsf(sum.imag) > STABILITY_THRESHOLD) {
                    is_unit = false;
                    break;
                }
            } else {
                // Off-diagonal elements should be 0
                if (fabsf(sum.real) > STABILITY_THRESHOLD || 
                    fabsf(sum.imag) > STABILITY_THRESHOLD) {
                    is_unit = false;
                    break;
                }
            }
        }
    }
    
    free(temp);
    return is_unit;
}

static void strassen_multiply(ComplexFloat* C, const ComplexFloat* A, const ComplexFloat* B, 
                            size_t n, size_t depth) {
    if (n <= STRASSEN_THRESHOLD || depth >= 3) {
        matrix_multiply_blocked(C, A, B, n, n, n);
        return;
    }
    
    size_t k = n/2;
    size_t size = k * k * sizeof(ComplexFloat);
    
    ComplexFloat *M1 = (ComplexFloat*)aligned_malloc(size);
    ComplexFloat *M2 = (ComplexFloat*)aligned_malloc(size);
    ComplexFloat *M3 = (ComplexFloat*)aligned_malloc(size);
    ComplexFloat *M4 = (ComplexFloat*)aligned_malloc(size);
    ComplexFloat *M5 = (ComplexFloat*)aligned_malloc(size);
    ComplexFloat *M6 = (ComplexFloat*)aligned_malloc(size);
    ComplexFloat *M7 = (ComplexFloat*)aligned_malloc(size);
    
    ComplexFloat *temp1 = (ComplexFloat*)aligned_malloc(size);
    ComplexFloat *temp2 = (ComplexFloat*)aligned_malloc(size);
    
    if (!M1 || !M2 || !M3 || !M4 || !M5 || !M6 || !M7 || !temp1 || !temp2) {
        // Handle allocation failure
        goto cleanup;
    }
    
    // M1 = (A11 + A22)(B11 + B22)
    matrix_add(temp1, &A[0], &A[k*n + k], k, k, n);
    matrix_add(temp2, &B[0], &B[k*n + k], k, k, n);
    strassen_multiply(M1, temp1, temp2, k, depth+1);
    
    // M2 = (A21 + A22)B11
    matrix_add(temp1, &A[k*n], &A[k*n + k], k, k, n);
    strassen_multiply(M2, temp1, &B[0], k, depth+1);
    
    // M3 = A11(B12 - B22)
    matrix_subtract(temp1, &B[k], &B[k*n + k], k, k, n);
    strassen_multiply(M3, &A[0], temp1, k, depth+1);
    
    // M4 = A22(B21 - B11)
    matrix_subtract(temp1, &B[k*n], &B[0], k, k, n);
    strassen_multiply(M4, &A[k*n + k], temp1, k, depth+1);
    
    // M5 = (A11 + A12)B22
    matrix_add(temp1, &A[0], &A[k], k, k, n);
    strassen_multiply(M5, temp1, &B[k*n + k], k, depth+1);
    
    // M6 = (A21 - A11)(B11 + B12)
    matrix_subtract(temp1, &A[k*n], &A[0], k, k, n);
    matrix_add(temp2, &B[0], &B[k], k, k, n);
    strassen_multiply(M6, temp1, temp2, k, depth+1);
    
    // M7 = (A12 - A22)(B21 + B22)
    matrix_subtract(temp1, &A[k], &A[k*n + k], k, k, n);
    matrix_add(temp2, &B[k*n], &B[k*n + k], k, k, n);
    strassen_multiply(M7, temp1, temp2, k, depth+1);
    
    // C11 = M1 + M4 - M5 + M7
    matrix_add(temp1, M1, M4, k, k, k);
    matrix_subtract(temp2, temp1, M5, k, k, k);
    matrix_add(&C[0], temp2, M7, k, k, n);
    
    // C12 = M3 + M5
    matrix_add(&C[k], M3, M5, k, k, n);
    
    // C21 = M2 + M4
    matrix_add(&C[k*n], M2, M4, k, k, n);
    
    // C22 = M1 - M2 + M3 + M6
    matrix_subtract(temp1, M1, M2, k, k, k);
    matrix_add(temp2, temp1, M3, k, k, k);
    matrix_add(&C[k*n + k], temp2, M6, k, k, n);
    
cleanup:
    free(M1); free(M2); free(M3); free(M4);
    free(M5); free(M6); free(M7);
    free(temp1); free(temp2);
}

bool qg_tensor_contract(ComplexFloat* result,
                       const ComplexFloat* a,
                       const ComplexFloat* b,
                       const size_t* dimensions_a,
                       const size_t* dimensions_b,
                       size_t rank_a,
                       size_t rank_b,
                       const size_t* contract_a,
                       const size_t* contract_b,
                       size_t num_contract) {
    // Validate inputs
    if (!result || !a || !b || !dimensions_a || !dimensions_b || !contract_a || !contract_b ||
        rank_a == 0 || rank_b == 0 || num_contract == 0) {
        geometric_log_error("Invalid parameters in qg_tensor_contract");
        return false;
    }
    
    // Validate contraction indices
    for (size_t i = 0; i < num_contract; i++) {
        if (contract_a[i] >= rank_a || contract_b[i] >= rank_b) {
            geometric_log_error("Invalid contraction indices");
            return false;
        }
        if (dimensions_a[contract_a[i]] != dimensions_b[contract_b[i]]) {
            geometric_log_error("Mismatched contraction dimensions");
            return false;
        }
    }

    // Calculate output dimensions
    size_t num_out_dimensions = rank_a + rank_b - 2 * num_contract;
    size_t* out_dimensions = (size_t*)aligned_malloc(num_out_dimensions * sizeof(size_t));
    if (!out_dimensions) {
        geometric_log_error("Memory allocation failed");
        return false;
    }
    
    // Calculate output shape and strides
    size_t out_idx = 0;
    for (size_t i = 0; i < rank_a; i++) {
        bool is_contracted = false;
        for (size_t j = 0; j < num_contract; j++) {
            if (i == contract_a[j]) {
                is_contracted = true;
                break;
            }
        }
        if (!is_contracted) {
            out_dimensions[out_idx++] = dimensions_a[i];
        }
    }
    for (size_t i = 0; i < rank_b; i++) {
        bool is_contracted = false;
        for (size_t j = 0; j < num_contract; j++) {
            if (i == contract_b[j]) {
                is_contracted = true;
                break;
            }
        }
        if (!is_contracted) {
            out_dimensions[out_idx++] = dimensions_b[i];
        }
    }
    
    // Special case: Matrix multiplication (2D tensors)
    if (rank_a == 2 && rank_b == 2 && num_contract == 1) {
        size_t M = dimensions_a[0];
        size_t K = dimensions_a[1];
        size_t N = dimensions_b[1];
        
        // Use Strassen for large matrices
        if (M >= STRASSEN_THRESHOLD && K >= STRASSEN_THRESHOLD && N >= STRASSEN_THRESHOLD) {
            size_t max_dim = max(max(M, K), N);
            size_t padded_size = 1;
            while (padded_size < max_dim) padded_size *= 2;
            
            ComplexFloat* padded_a = (ComplexFloat*)aligned_malloc(padded_size * padded_size * sizeof(ComplexFloat));
            ComplexFloat* padded_b = (ComplexFloat*)aligned_malloc(padded_size * padded_size * sizeof(ComplexFloat));
            ComplexFloat* padded_c = (ComplexFloat*)aligned_malloc(padded_size * padded_size * sizeof(ComplexFloat));
            
            if (!padded_a || !padded_b || !padded_c) {
                free(padded_a); free(padded_b); free(padded_c);
                free(out_dimensions);
                return false;
            }
            
            // Zero pad matrices
            for (size_t i = 0; i < padded_size * padded_size; i++) {
                padded_a[i] = COMPLEX_FLOAT_ZERO;
                padded_b[i] = COMPLEX_FLOAT_ZERO;
            }
            
            // Copy data
            for (size_t i = 0; i < M; i++) {
                for (size_t j = 0; j < K; j++) {
                    padded_a[i * padded_size + j] = a[i * K + j];
                }
            }
            for (size_t i = 0; i < K; i++) {
                for (size_t j = 0; j < N; j++) {
                    padded_b[i * padded_size + j] = b[i * N + j];
                }
            }
            
            // Perform Strassen multiplication
            strassen_multiply(padded_c, padded_a, padded_b, padded_size, 0);
            
            // Copy result back
            for (size_t i = 0; i < M; i++) {
                for (size_t j = 0; j < N; j++) {
                    result[i * N + j] = padded_c[i * padded_size + j];
                }
            }
            
            free(padded_a);
            free(padded_b);
            free(padded_c);
            free(out_dimensions);
            return true;
        }
        
        // Use blocked matrix multiplication for smaller matrices
        matrix_multiply_blocked(result, a, b, M, N, K);
        free(out_dimensions);
        return true;
    }
    
    // General case: Tensor contraction
    size_t total_size = 1;
    for (size_t i = 0; i < num_out_dimensions; i++) {
        total_size *= out_dimensions[i];
    }
    
    // Initialize result tensor to zero
    for (size_t i = 0; i < total_size; i++) {
        result[i] = COMPLEX_FLOAT_ZERO;
    }
    
    // Calculate strides for input tensors
    size_t* strides_a = (size_t*)aligned_malloc(rank_a * sizeof(size_t));
    size_t* strides_b = (size_t*)aligned_malloc(rank_b * sizeof(size_t));
    if (!strides_a || !strides_b) {
        free(strides_a);
        free(strides_b);
        free(out_dimensions);
        return false;
    }
    
    strides_a[rank_a - 1] = 1;
    strides_b[rank_b - 1] = 1;
    for (size_t i = rank_a - 1; i > 0; i--) {
        strides_a[i - 1] = strides_a[i] * dimensions_a[i];
    }
    for (size_t i = rank_b - 1; i > 0; i--) {
        strides_b[i - 1] = strides_b[i] * dimensions_b[i];
    }
    
    // Calculate contraction size
    size_t contract_size = 1;
    for (size_t i = 0; i < num_contract; i++) {
        contract_size *= dimensions_a[contract_a[i]];
    }
    
    // Perform contraction
    #pragma omp parallel for collapse(2) schedule(guided)
    for (size_t i = 0; i < total_size; i++) {
        for (size_t k = 0; k < contract_size; k++) {
            // Calculate indices for tensors a and b
            size_t idx_a = 0;
            size_t idx_b = 0;
            size_t temp_i = i;
            size_t temp_k = k;
            
            // Map output index to tensor a indices
            size_t out_idx = 0;
            for (size_t j = 0; j < rank_a; j++) {
                bool is_contracted = false;
                for (size_t c = 0; c < num_contract; c++) {
                    if (j == contract_a[c]) {
                        is_contracted = true;
                        size_t contract_idx = temp_k % dimensions_a[j];
                        temp_k /= dimensions_a[j];
                        idx_a += contract_idx * strides_a[j];
                        break;
                    }
                }
                if (!is_contracted) {
                    size_t dim_idx = temp_i % out_dimensions[out_idx];
                    temp_i /= out_dimensions[out_idx];
                    idx_a += dim_idx * strides_a[j];
                    out_idx++;
                }
            }
            
            // Map output and contraction indices to tensor b indices
            out_idx = rank_a - num_contract;
            temp_i = i;
            temp_k = k;
            for (size_t j = 0; j < rank_b; j++) {
                bool is_contracted = false;
                for (size_t c = 0; c < num_contract; c++) {
                    if (j == contract_b[c]) {
                        is_contracted = true;
                        size_t contract_idx = temp_k % dimensions_b[j];
                        temp_k /= dimensions_b[j];
                        idx_b += contract_idx * strides_b[j];
                        break;
                    }
                }
                if (!is_contracted) {
                    size_t dim_idx = (temp_i / out_dimensions[out_idx]) % out_dimensions[out_idx];
                    idx_b += dim_idx * strides_b[j];
                    out_idx++;
                }
            }
            
            // Multiply and accumulate
            ComplexFloat prod;
            prod.real = a[idx_a].real * b[idx_b].real - a[idx_a].imag * b[idx_b].imag;
            prod.imag = a[idx_a].real * b[idx_b].imag + a[idx_a].imag * b[idx_b].real;
            
            #pragma omp atomic
            result[i].real += prod.real;
            #pragma omp atomic
            result[i].imag += prod.imag;
        }
    }
    
    free(strides_a);
    free(strides_b);
    free(out_dimensions);
    return true;
}
