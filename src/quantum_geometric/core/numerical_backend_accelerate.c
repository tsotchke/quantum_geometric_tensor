#include "quantum_geometric/core/numerical_backend.h"
#include "quantum_geometric/core/complex_arithmetic.h"
#include "quantum_geometric/core/lapack_wrapper.h"
#include "quantum_geometric/core/hierarchical_matrix.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

// Global state
#ifdef __APPLE__
// DSPComplex alignment requirement - must be a power of 2 and multiple of sizeof(DSPComplex)
#define DSP_ALIGNMENT (sizeof(DSPComplex) * 2)

static struct {
    numerical_config_t config;
    numerical_metrics_t metrics;
    numerical_error_t last_error;
    bool initialized;
    bool has_lapack;
} backend_state = {0};

// Additional vector operations
numerical_error_t numerical_vector_scale(const ComplexFloat* a,
                                       ComplexFloat scale,
                                       ComplexFloat* c,
                                       size_t length) {
    if (!backend_state.initialized) {
        return NUMERICAL_ERROR_INVALID_STATE;
    }
    
    if (!a || !c) {
        return NUMERICAL_ERROR_INVALID_ARGUMENT;
    }
    
    DSPComplex dsp_scale = to_dsp_complex(scale);
    DSPComplex* dsp_a = NULL;
    if (posix_memalign((void**)&dsp_a, DSP_ALIGNMENT, length * sizeof(DSPComplex)) != 0) {
        return NUMERICAL_ERROR_MEMORY;
    }
    
    // Convert input using our conversion utilities
    for (size_t i = 0; i < length; i++) {
        dsp_a[i] = to_dsp_complex(a[i]);
    }
    
    // Allocate aligned output buffer
    DSPComplex* dsp_c = NULL;
    if (posix_memalign((void**)&dsp_c, DSP_ALIGNMENT, length * sizeof(DSPComplex)) != 0) {
        free(dsp_a);
        return NUMERICAL_ERROR_MEMORY;
    }
    
    // Scale using vDSP
    float scale_real = dsp_scale.real;
    vDSP_zrvmul(&scale_real, 1, dsp_a, 1, dsp_c, 1, length);
    
    // Copy result back
    for (size_t i = 0; i < length; i++) {
        c[i] = from_dsp_complex(dsp_c[i]);
    }
    
    free(dsp_c);
    free(dsp_a);
    return NUMERICAL_SUCCESS;
}

numerical_error_t numerical_vector_add(const ComplexFloat* a,
                                     const ComplexFloat* b,
                                     ComplexFloat* c,
                                     size_t length) {
    if (!backend_state.initialized) {
        return NUMERICAL_ERROR_INVALID_STATE;
    }
    
    if (!a || !b || !c) {
        return NUMERICAL_ERROR_INVALID_ARGUMENT;
    }
    
    DSPComplex *dsp_a = NULL, *dsp_b = NULL;
    if (posix_memalign((void**)&dsp_a, DSP_ALIGNMENT, length * sizeof(DSPComplex)) != 0 ||
        posix_memalign((void**)&dsp_b, DSP_ALIGNMENT, length * sizeof(DSPComplex)) != 0) {
        free(dsp_a);
        free(dsp_b);
        return NUMERICAL_ERROR_MEMORY;
    }
    
    // Convert inputs using our conversion utilities
    for (size_t i = 0; i < length; i++) {
        dsp_a[i] = to_dsp_complex(a[i]);
        dsp_b[i] = to_dsp_complex(b[i]);
    }
    
    // Allocate aligned output buffer
    DSPComplex* dsp_c = NULL;
    if (posix_memalign((void**)&dsp_c, DSP_ALIGNMENT, length * sizeof(DSPComplex)) != 0) {
        free(dsp_a);
        free(dsp_b);
        return NUMERICAL_ERROR_MEMORY;
    }
    
    // Add using vDSP
    vDSP_zvadd(dsp_a, 1, dsp_b, 1, dsp_c, 1, length);
    
    // Copy result back
    for (size_t i = 0; i < length; i++) {
        c[i] = from_dsp_complex(dsp_c[i]);
    }
    
    free(dsp_c);
    free(dsp_a);
    free(dsp_b);
    return NUMERICAL_SUCCESS;
}

numerical_error_t numerical_matrix_subtract(const ComplexFloat* a,
                                          const ComplexFloat* b,
                                          ComplexFloat* c,
                                          size_t rows,
                                          size_t cols) {
    if (!backend_state.initialized) {
        return NUMERICAL_ERROR_INVALID_STATE;
    }
    
    if (!a || !b || !c) {
        return NUMERICAL_ERROR_INVALID_ARGUMENT;
    }
    
    printf("DEBUG: numerical_matrix_subtract: rows=%zu, cols=%zu\n", rows, cols);
    
    size_t total = rows * cols;
    
    // Use posix_memalign for proper alignment
    DSPComplex *dsp_a = NULL, *dsp_b = NULL;
    if (posix_memalign((void**)&dsp_a, DSP_ALIGNMENT, total * sizeof(DSPComplex)) != 0 ||
        posix_memalign((void**)&dsp_b, DSP_ALIGNMENT, total * sizeof(DSPComplex)) != 0) {
        printf("DEBUG: Memory allocation failed\n");
        free(dsp_a);
        free(dsp_b);
        return NUMERICAL_ERROR_MEMORY;
    }
    
    printf("DEBUG: Converting input matrices to DSPComplex format\n");
    // Convert inputs using our conversion utilities
    for (size_t i = 0; i < total; i++) {
        dsp_a[i] = to_dsp_complex(a[i]);
        dsp_b[i] = to_dsp_complex(b[i]);
    }
    
    printf("DEBUG: First few elements of matrices:\n");
    printf("Matrix A: ");
    for (size_t i = 0; i < (total < 3 ? total : 3); i++) {
        printf("(%.3f,%.3f) ", dsp_a[i].real, dsp_a[i].imag);
    }
    printf("\nMatrix B: ");
    for (size_t i = 0; i < (total < 3 ? total : 3); i++) {
        printf("(%.3f,%.3f) ", dsp_b[i].real, dsp_b[i].imag);
    }
    printf("\n");
    
    // Allocate aligned output buffer
    DSPComplex* dsp_c = NULL;
    if (posix_memalign((void**)&dsp_c, DSP_ALIGNMENT, total * sizeof(DSPComplex)) != 0) {
        printf("DEBUG: Failed to allocate aligned output buffer\n");
        free(dsp_a);
        free(dsp_b);
        return NUMERICAL_ERROR_MEMORY;
    }
    
    // Initialize output buffer to zero
    memset(dsp_c, 0, total * sizeof(DSPComplex));
    
    // Normalize inputs to prevent numerical instability
    float max_a = 0.0f, max_b = 0.0f;
    for (size_t i = 0; i < total; i++) {
        float mag_a = sqrtf(dsp_a[i].real * dsp_a[i].real + dsp_a[i].imag * dsp_a[i].imag);
        float mag_b = sqrtf(dsp_b[i].real * dsp_b[i].real + dsp_b[i].imag * dsp_b[i].imag);
        if (mag_a > max_a) max_a = mag_a;
        if (mag_b > max_b) max_b = mag_b;
    }
    
    // Normalize inputs in-place to prevent numerical instability
    if (max_a > 1e-6f) {
        float scale_a = 1.0f/max_a;
        vDSP_zrvmul(&scale_a, 1, dsp_a, 1, dsp_a, 1, total);
    }
    if (max_b > 1e-6f) {
        float scale_b = 1.0f/max_b;
        vDSP_zrvmul(&scale_b, 1, dsp_b, 1, dsp_b, 1, total);
    }
    
    // Subtract b from a
    float scale = -1.0f;
    vDSP_zrvmul(&scale, 1, dsp_b, 1, dsp_c, 1, total);
    vDSP_zvadd(dsp_a, 1, dsp_c, 1, dsp_c, 1, total);
    
    // Scale result back
    float scale_factor = (max_a > max_b ? max_a : max_b);
    if (scale_factor > 1e-6f) {
        vDSP_zrvmul(&scale_factor, 1, dsp_c, 1, dsp_c, 1, total);
    }
    
    // Copy result back to output
    for (size_t i = 0; i < total; i++) {
        c[i] = from_dsp_complex(dsp_c[i]);
    }
    
    printf("DEBUG: Result conversion completed\n");
    free(dsp_c);
    
    printf("DEBUG: Result (first few elements): ");
    for (size_t i = 0; i < (total < 3 ? total : 3); i++) {
        printf("(%.3f,%.3f) ", c[i].real, c[i].imag);
    }
    printf("\n");
    
    // Check for invalid results
    for (size_t i = 0; i < total; i++) {
        if (isnan(c[i].real) || isnan(c[i].imag) ||
            isinf(c[i].real) || isinf(c[i].imag)) {
            printf("DEBUG: Invalid result detected, using direct computation\n");
            // Compute subtraction directly
            for (size_t j = 0; j < total; j++) {
                c[j] = complex_subtract(a[j], b[j]);
            }
            free(dsp_a);
            free(dsp_b);
            return NUMERICAL_SUCCESS;
        }
    }
    
    free(dsp_a);
    free(dsp_b);
    return NUMERICAL_SUCCESS;
}

bool initialize_numerical_backend_accelerate(const numerical_config_t* config) {
    if (!config) {
        backend_state.last_error = NUMERICAL_ERROR_INVALID_ARGUMENT;
        return false;
    }
    
    backend_state.config = *config;
    backend_state.initialized = true;
    
    // Reset metrics
    memset(&backend_state.metrics, 0, sizeof(numerical_metrics_t));
    
    // Check for LAPACK availability
    backend_state.has_lapack = lapack_has_capability("svd");
    
    backend_state.last_error = NUMERICAL_SUCCESS;
    return true;
}

void shutdown_numerical_backend_accelerate(void) {
    backend_state.initialized = false;
}

numerical_error_t numerical_matrix_multiply_accelerate(const ComplexFloat* a,
                                                const ComplexFloat* b,
                                                ComplexFloat* c,
                                                size_t m, size_t k, size_t n,
                                                bool transpose_a,
                                                bool transpose_b) {
    if (!backend_state.initialized) {
        return NUMERICAL_ERROR_INVALID_STATE;
    }
    
    if (!a || !b || !c) {
        return NUMERICAL_ERROR_INVALID_ARGUMENT;
    }
    
    printf("DEBUG: Accelerate: Initial matrix dimensions: A[%zu x %zu], B[%zu x %zu], C[%zu x %zu]\n", 
           m, k, k, n, m, n);
    printf("DEBUG: Accelerate: Allocating initial matrices\n");
    
    // Calculate sizes
    size_t dsp_a_size = m * k * sizeof(DSPComplex);
    size_t dsp_b_size = k * n * sizeof(DSPComplex);
    size_t dsp_c_size = m * n * sizeof(DSPComplex);
    
    printf("DEBUG: Accelerate: Matrix sizes: A=%zu bytes, B=%zu bytes, C=%zu bytes\n",
           dsp_a_size, dsp_b_size, dsp_c_size);
    
    // The input pointers are already ComplexFloat arrays, no need to cast
    const ComplexFloat* data_a = a;
    const ComplexFloat* data_b = b;
    
    printf("DEBUG: Accelerate: Input data pointers: data_a=%p, data_b=%p\n", 
           (void*)data_a, (void*)data_b);
    
    // Convert to DSPComplex format with proper alignment
    DSPComplex* dsp_a = NULL;
    DSPComplex* dsp_b = NULL;
    DSPComplex* dsp_c = NULL;
    
    // Calculate aligned sizes to ensure proper memory alignment
    size_t aligned_size_a = ((m * k * sizeof(DSPComplex) + DSP_ALIGNMENT - 1) / DSP_ALIGNMENT) * DSP_ALIGNMENT;
    size_t aligned_size_b = ((k * n * sizeof(DSPComplex) + DSP_ALIGNMENT - 1) / DSP_ALIGNMENT) * DSP_ALIGNMENT;
    size_t aligned_size_c = ((m * n * sizeof(DSPComplex) + DSP_ALIGNMENT - 1) / DSP_ALIGNMENT) * DSP_ALIGNMENT;

    // Use posix_memalign for proper alignment
    if (posix_memalign((void**)&dsp_a, DSP_ALIGNMENT, aligned_size_a) != 0 ||
        posix_memalign((void**)&dsp_b, DSP_ALIGNMENT, aligned_size_b) != 0 ||
        posix_memalign((void**)&dsp_c, DSP_ALIGNMENT, aligned_size_c) != 0) {
        free(dsp_a);
        free(dsp_b);
        free(dsp_c);
        return NUMERICAL_ERROR_MEMORY;
    }

    // Verify alignment
    if (((uintptr_t)dsp_a % DSP_ALIGNMENT) != 0 ||
        ((uintptr_t)dsp_b % DSP_ALIGNMENT) != 0 ||
        ((uintptr_t)dsp_c % DSP_ALIGNMENT) != 0) {
        printf("ERROR: Memory not properly aligned\n");
        free(dsp_a);
        free(dsp_b);
        free(dsp_c);
        return NUMERICAL_ERROR_MEMORY;
    }
    
    printf("DEBUG: Accelerate: Matrix pointers: dsp_a=%p, dsp_b=%p, dsp_c=%p\n",
           (void*)dsp_a, (void*)dsp_b, (void*)dsp_c);
    
    if (!dsp_a || !dsp_b || !dsp_c) {
        printf("DEBUG: Accelerate: Memory allocation failed\n");
        free(dsp_a);
        free(dsp_b);
        free(dsp_c);
        return NUMERICAL_ERROR_MEMORY;
    }
    
    printf("DEBUG: Accelerate: Using direct matrix data pointers\n");
    
    printf("DEBUG: Accelerate: Converting matrices to DSPComplex format\n");
    printf("DEBUG: Accelerate: Matrix A dimensions: %zu x %zu\n", m, k);
    printf("DEBUG: Accelerate: Matrix B dimensions: %zu x %zu\n", k, n);
    
    // Convert input matrices from row-major to column-major order
    printf("DEBUG: Accelerate: Converting matrix A to column-major\n");
    if (transpose_a) {
        // When transposing A[m x k] -> A[k x m], we need to read in transposed order
        for (size_t j = 0; j < k; j++) {  // Columns in original matrix
            for (size_t i = 0; i < m; i++) {  // Rows in original matrix
                size_t src_idx = i * k + j;  // Row-major index in original matrix
                size_t dst_idx = i * k + j;  // Keep same layout for transposed access
                dsp_a[dst_idx] = to_dsp_complex(data_a[src_idx]);
            }
        }
        // Transpose in-place
        for (size_t i = 0; i < k; i++) {
            for (size_t j = i + 1; j < m; j++) {
                size_t idx1 = i * m + j;
                size_t idx2 = j * k + i;
                DSPComplex temp = dsp_a[idx1];
                dsp_a[idx1] = dsp_a[idx2];
                dsp_a[idx2] = temp;
            }
        }
    } else {
        // Normal column-major conversion for A[m x k]
        for (size_t j = 0; j < k; j++) {  // Columns
            for (size_t i = 0; i < m; i++) {  // Rows
                size_t src_idx = i * k + j;  // Row-major index
                size_t dst_idx = j * m + i;  // Column-major index
                dsp_a[dst_idx] = to_dsp_complex(data_a[src_idx]);
            }
        }
    }
    printf("DEBUG: Accelerate: Matrix A converted\n");
    
    printf("DEBUG: Accelerate: Converting matrix B to column-major\n");
    if (transpose_b) {
        // When transposing B[k x n] -> B[n x k], we need to read in transposed order
        for (size_t j = 0; j < n; j++) {  // Columns in original matrix
            for (size_t i = 0; i < k; i++) {  // Rows in original matrix
                size_t src_idx = i * n + j;  // Row-major index in original matrix
                size_t dst_idx = i * n + j;  // Keep same layout for transposed access
                dsp_b[dst_idx] = to_dsp_complex(data_b[src_idx]);
            }
        }
        // Transpose in-place
        for (size_t i = 0; i < n; i++) {
            for (size_t j = i + 1; j < k; j++) {
                size_t idx1 = i * k + j;
                size_t idx2 = j * n + i;
                DSPComplex temp = dsp_b[idx1];
                dsp_b[idx1] = dsp_b[idx2];
                dsp_b[idx2] = temp;
            }
        }
    } else {
        // Normal column-major conversion for B[k x n]
        for (size_t j = 0; j < n; j++) {  // Columns
            for (size_t i = 0; i < k; i++) {  // Rows
                size_t src_idx = i * n + j;  // Row-major index
                size_t dst_idx = j * k + i;  // Column-major index
                dsp_b[dst_idx] = to_dsp_complex(data_b[src_idx]);
            }
        }
    }
    printf("DEBUG: Accelerate: Matrix B converted\n");

    // Print matrix layouts after conversion
    printf("\nDEBUG: Matrix layouts after conversion:\n");
    printf("Matrix A (first few elements):\n");
    for (size_t i = 0; i < (m < 3 ? m : 3); i++) {
        for (size_t j = 0; j < (k < 3 ? k : 3); j++) {
            size_t idx = transpose_a ? (i * k + j) : (j * m + i);
            printf("A[%zu,%zu] = %.3f + %.3fi  ", i, j, dsp_a[idx].real, dsp_a[idx].imag);
        }
        printf("\n");
    }
    printf("\nMatrix B (first few elements):\n");
    for (size_t i = 0; i < (k < 3 ? k : 3); i++) {
        for (size_t j = 0; j < (n < 3 ? n : 3); j++) {
            size_t idx = transpose_b ? (i * n + j) : (j * k + i);
            printf("B[%zu,%zu] = %.3f + %.3fi  ", i, j, dsp_b[idx].real, dsp_b[idx].imag);
        }
        printf("\n");
    }
    
    // Handle matrix transposition if needed
    DSPComplex* a_use = dsp_a;
    DSPComplex* b_use = dsp_b;
    size_t m_use = m;
    size_t k_use = k;
    size_t n_use = n;
    
    // For vDSP_zmmul, we need to handle the strides carefully
    // For column-major format, stride is the distance between consecutive columns
    // For a MxN matrix, stride should be M (number of elements in a column)
    size_t stride_a = m;  // For A[m x k], stride is m
    size_t stride_b = k;  // For B[k x n], stride is k
    size_t stride_c = m;  // For C[m x n], stride is m

    // For 1xN matrices (row vectors), we need special handling
    // We'll treat them as Nx1 matrices and adjust the dimensions
    if (m == 1) {
        // Transpose A from 1xk to kx1
        size_t temp = m_use;
        m_use = k_use;
        k_use = temp;
        stride_a = 1;  // Now it's a column vector
        
        // Adjust B accordingly
        stride_b = m_use;  // Now B is m_use x n
        
        // C will be nx1
        stride_c = 1;
    }

    // Print detailed stride information for debugging
    printf("\nDEBUG: Adjusted matrix layout:\n");
    printf("A: %zux%zu matrix, stride=%zu (treating as column vector)\n", 
           m_use, k_use, stride_a);
    printf("B: %zux%zu matrix, stride=%zu\n", 
           k_use, n_use, stride_b);
    printf("C: %zux%zu matrix, stride=%zu\n", 
           m_use, n_use, stride_c);

    // Print detailed stride information
    printf("\nDEBUG: Matrix strides and dimensions:\n");
    printf("Matrix A: %zu x %zu, stride=%zu (original layout)\n", m, k, stride_a);
    printf("Matrix B: %zu x %zu, stride=%zu (original layout)\n", k, n, stride_b);
    printf("Matrix C: %zu x %zu, stride=%zu (original layout)\n", m, n, stride_c);
    printf("After transposition:\n");
    printf("Matrix A: %zu x %zu (transposed=%d)\n", m_use, k_use, transpose_a);
    printf("Matrix B: %zu x %zu (transposed=%d)\n", k_use, n_use, transpose_b);

    // Print detailed stride calculations
    printf("\nDEBUG: Stride calculations:\n");
    printf("Matrix A: %zu x %zu -> stride = %zu (rows in %s matrix)\n",
           m, k, stride_a, transpose_a ? "transposed" : "original");
    printf("Matrix B: %zu x %zu -> stride = %zu (rows in %s matrix)\n",
           k, n, stride_b, transpose_b ? "transposed" : "original");
    printf("Matrix C: %zu x %zu -> stride = %zu\n", m, n, stride_c);

    // Print detailed memory layout information
    printf("\nDEBUG: Memory layout details:\n");
    printf("Matrix A: %s format, stride=%zu\n", 
           transpose_a ? "transposed" : "normal", stride_a);
    printf("Matrix B: %s format, stride=%zu\n", 
           transpose_b ? "transposed" : "normal", stride_b);
    printf("Matrix C: normal format, stride=%zu\n", stride_c);

    // Print detailed stride information
    printf("DEBUG: Matrix strides after transpose:\n");
    printf("A: stride=%zu (m_use=%zu, k_use=%zu)\n", stride_a, m_use, k_use);
    printf("B: stride=%zu (k_use=%zu, n_use=%zu)\n", stride_b, k_use, n_use);
    printf("C: stride=%zu (m_use=%zu, n_use=%zu)\n", stride_c, m_use, n_use);
    
    // Perform matrix multiplication
    {
        printf("DEBUG: Accelerate: Matrix dimensions before multiplication:\n");
        printf("DEBUG: Accelerate: A dimensions: %zu x %zu\n", m, k);
        printf("DEBUG: Accelerate: B dimensions: %zu x %zu\n", k, n);
        printf("DEBUG: Accelerate: Expected C dimensions: %zu x %zu\n", m, n);
        
        // Calculate aligned size for temp_c
        size_t aligned_size_temp_c = ((m * n * sizeof(DSPComplex) + DSP_ALIGNMENT - 1) / DSP_ALIGNMENT) * DSP_ALIGNMENT;
        
        // Allocate temporary output matrix with proper alignment
        DSPComplex* temp_c = NULL;
        if (posix_memalign((void**)&temp_c, DSP_ALIGNMENT, aligned_size_temp_c) != 0) {
            free(dsp_a);
            free(dsp_b);
            free(dsp_c);
            return NUMERICAL_ERROR_MEMORY;
        }

        // Verify alignment
        if (((uintptr_t)temp_c % DSP_ALIGNMENT) != 0) {
            printf("ERROR: Temporary matrix not properly aligned\n");
            free(dsp_a);
            free(dsp_b);
            free(dsp_c);
            free(temp_c);
            return NUMERICAL_ERROR_MEMORY;
        }

        // Initialize temp_c to zero
        memset(temp_c, 0, aligned_size_temp_c);

        // Print detailed matrix information before multiplication
        printf("\nDEBUG: Matrix A (first few elements, column-major):\n");
        for (size_t j = 0; j < (k < 3 ? k : 3); j++) {
            printf("Column %zu: ", j);
            for (size_t i = 0; i < (m < 3 ? m : 3); i++) {
                size_t idx = j * m + i;
                printf("(%.3f,%.3f) ", dsp_a[idx].real, dsp_a[idx].imag);
            }
            printf("\n");
        }

        printf("\nDEBUG: Matrix B (first few elements, column-major):\n");
        for (size_t j = 0; j < (n < 3 ? n : 3); j++) {
            printf("Column %zu: ", j);
            for (size_t i = 0; i < (k < 3 ? k : 3); i++) {
                size_t idx = j * k + i;
                printf("(%.3f,%.3f) ", dsp_b[idx].real, dsp_b[idx].imag);
            }
            printf("\n");
        }

        printf("\nDEBUG: Matrix multiplication parameters:\n");
        printf("A: %zu x %zu (stride=%zu)\n", m, k, stride_a);
        printf("B: %zu x %zu (stride=%zu)\n", k, n, stride_b);
        printf("C: %zu x %zu (stride=%zu)\n", m, n, stride_c);
        printf("Used dimensions: M=%zu, N=%zu, P=%zu\n", m_use, n_use, k_use);
        printf("Memory alignment: A=%zu, B=%zu, C=%zu\n",
               (size_t)dsp_a % DSP_ALIGNMENT,
               (size_t)dsp_b % DSP_ALIGNMENT,
               (size_t)temp_c % DSP_ALIGNMENT);
        printf("Memory ranges:\n");
        printf("A: %p to %p (size=%zu)\n", (void*)dsp_a, (void*)(dsp_a + m * k), m * k * sizeof(DSPComplex));
        printf("B: %p to %p (size=%zu)\n", (void*)dsp_b, (void*)(dsp_b + k * n), k * n * sizeof(DSPComplex));
        printf("C: %p to %p (size=%zu)\n", (void*)temp_c, (void*)(temp_c + m * n), m * n * sizeof(DSPComplex));

        printf("\nDEBUG: Calling vDSP_zmmul...\n");
        
        // Verify matrix dimensions
        if (m * n * sizeof(DSPComplex) > dsp_c_size) {
            printf("DEBUG: Accelerate: Output matrix size mismatch: need %zu bytes, have %zu bytes\n",
                   m * n * sizeof(DSPComplex), dsp_c_size);
            free(dsp_a);
            free(dsp_b);
            free(dsp_c);
            return NUMERICAL_ERROR_INVALID_ARGUMENT;
        }
        
        
    // Use LAPACK through Accelerate framework for matrix multiplication
    if (backend_state.has_lapack) {
        bool success = lapack_matrix_multiply(a, b, c, m, k, n,
                                           transpose_a, transpose_b,
                                           LAPACK_ROW_MAJOR);
        if (!success) {
            free(dsp_a);
            free(dsp_b);
            free(dsp_c);
            free(temp_c);
            return NUMERICAL_ERROR_COMPUTATION;
        }
        return NUMERICAL_SUCCESS;
    }

    // Fallback to basic implementation if LAPACK is not available
    memset(temp_c, 0, m * n * sizeof(DSPComplex));
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            DSPComplex sum = {0.0f, 0.0f};
            for (size_t l = 0; l < k; l++) {
                DSPComplex a_val = transpose_a ? dsp_a[l * m + i] : dsp_a[i * k + l];
                DSPComplex b_val = transpose_b ? dsp_b[j * k + l] : dsp_b[l * n + j];
                // Complex multiplication
                float real = a_val.real * b_val.real - a_val.imag * b_val.imag;
                float imag = a_val.real * b_val.imag + a_val.imag * b_val.real;
                sum.real += real;
                sum.imag += imag;
            }
            temp_c[i * n + j] = sum;
        }
    }
        
        // Print detailed memory info
        printf("DEBUG: Memory alignment:\n");
        printf("dsp_a alignment: %zu\n", (size_t)dsp_a % sizeof(DSPComplex));
        printf("dsp_b alignment: %zu\n", (size_t)dsp_b % sizeof(DSPComplex));
        printf("temp_c alignment: %zu\n", (size_t)temp_c % sizeof(DSPComplex));
        
        // Print detailed matrix information
        printf("\nDEBUG: Matrix A Details:\n");
        printf("Original dimensions: %zu x %zu\n", m, k);
        printf("Used dimensions: %zu x %zu\n", m_use, k_use);
        printf("Memory layout (first few elements):\n");
        printf("Row-major indices -> values:\n");
        for (size_t i = 0; i < (m < 3 ? m : 3); i++) {
            for (size_t j = 0; j < (k < 3 ? k : 3); j++) {
                size_t idx = i * k + j;
                printf("[%zu] (%zu,%zu) -> %.3f+%.3fi\n", 
                    idx, i, j, dsp_a[idx].real, dsp_a[idx].imag);
            }
        }
        printf("Column-major indices -> values:\n");
        for (size_t j = 0; j < (k < 3 ? k : 3); j++) {
            for (size_t i = 0; i < (m < 3 ? m : 3); i++) {
                size_t idx = j * m + i;
                printf("[%zu] (%zu,%zu) -> %.3f+%.3fi\n", 
                    idx, i, j, dsp_a[idx].real, dsp_a[idx].imag);
            }
        }

        printf("\nDEBUG: Matrix B Details:\n");
        printf("Original dimensions: %zu x %zu\n", k, n);
        printf("Used dimensions: %zu x %zu\n", k_use, n_use);
        printf("Memory layout (first few elements):\n");
        printf("Row-major indices -> values:\n");
        for (size_t i = 0; i < (k < 3 ? k : 3); i++) {
            for (size_t j = 0; j < (n < 3 ? n : 3); j++) {
                size_t idx = i * n + j;
                printf("[%zu] (%zu,%zu) -> %.3f+%.3fi\n", 
                    idx, i, j, dsp_b[idx].real, dsp_b[idx].imag);
            }
        }
        printf("Column-major indices -> values:\n");
        for (size_t j = 0; j < (n < 3 ? n : 3); j++) {
            for (size_t i = 0; i < (k < 3 ? k : 3); i++) {
                size_t idx = j * k + i;
                printf("[%zu] (%zu,%zu) -> %.3f+%.3fi\n", 
                    idx, i, j, dsp_b[idx].real, dsp_b[idx].imag);
            }
        }

        printf("\nDEBUG: Matrix Multiplication Parameters:\n");
        printf("vDSP_zmmul(a_use=%p, stride_a=%zu, b_use=%p, stride_b=%zu, temp_c=%p, stride_c=%zu, M=%zu, N=%zu, P=%zu)\n",
            (void*)a_use, stride_a, (void*)b_use, stride_b, (void*)temp_c, stride_c, m_use, n_use, k_use);
        
        // Print memory addresses and sizes
        printf("DEBUG: Memory ranges:\n");
        printf("dsp_a: %p to %p (size: %zu)\n", (void*)dsp_a, (void*)(dsp_a + m * k), m * k * sizeof(DSPComplex));
        printf("dsp_b: %p to %p (size: %zu)\n", (void*)dsp_b, (void*)(dsp_b + k * n), k * n * sizeof(DSPComplex));
        printf("temp_c: %p to %p (size: %zu)\n", (void*)temp_c, (void*)(temp_c + m * n), m * n * sizeof(DSPComplex));
        
        // Copy result to output matrix
        memcpy(dsp_c, temp_c, m * n * sizeof(DSPComplex));
        free(temp_c);
        printf("DEBUG: Accelerate: Matrix multiplication completed\n");
        
        printf("DEBUG: Accelerate: Result matrix first few elements:\n");
        for (size_t i = 0; i < (m < 3 ? m : 3); i++) {
            for (size_t j = 0; j < (n < 3 ? n : 3); j++) {
                printf("C[%zu,%zu] = %.3f + %.3fi  ", i, j, dsp_c[j * m + i].real, dsp_c[j * m + i].imag);
            }
            printf("\n");
        }
    }
    
    // Convert result from column-major back to row-major order
    // For column-major matrix C[m x n], element at (i,j) is at index j*m + i
    // For row-major output, element at (i,j) should be at index i*n + j
    printf("DEBUG: Converting result matrix from column-major to row-major\n");
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            size_t src_idx = j * m + i;  // Column-major index
            size_t dst_idx = i * n + j;  // Row-major index
            c[dst_idx] = from_dsp_complex(dsp_c[src_idx]);
            printf("DEBUG: Moving element from [%zu] to [%zu]\n", src_idx, dst_idx);
        }
    }
    
    // Clean up allocated memory
    free(dsp_a);
    free(dsp_b);
    free(dsp_c);
    
    return NUMERICAL_SUCCESS;
}

numerical_error_t numerical_matrix_add_accelerate(const ComplexFloat* a,
                                           const ComplexFloat* b,
                                           ComplexFloat* c,
                                           size_t rows,
                                           size_t cols) {
    if (!backend_state.initialized) {
        return NUMERICAL_ERROR_INVALID_STATE;
    }
    
    if (!a || !b || !c) {
        return NUMERICAL_ERROR_INVALID_ARGUMENT;
    }
    
    size_t total = rows * cols;
    DSPComplex *dsp_a = NULL, *dsp_b = NULL;
    if (posix_memalign((void**)&dsp_a, DSP_ALIGNMENT, total * sizeof(DSPComplex)) != 0 ||
        posix_memalign((void**)&dsp_b, DSP_ALIGNMENT, total * sizeof(DSPComplex)) != 0) {
        free(dsp_a);
        free(dsp_b);
        return NUMERICAL_ERROR_MEMORY;
    }
    
    // Convert inputs using our conversion utilities
    for (size_t i = 0; i < total; i++) {
        dsp_a[i] = to_dsp_complex(a[i]);
        dsp_b[i] = to_dsp_complex(b[i]);
    }
    
    // Allocate aligned output buffer
    DSPComplex* dsp_c = NULL;
    if (posix_memalign((void**)&dsp_c, DSP_ALIGNMENT, total * sizeof(DSPComplex)) != 0) {
        free(dsp_a);
        free(dsp_b);
        return NUMERICAL_ERROR_MEMORY;
    }
    
    // Initialize output buffer to zero
    memset(dsp_c, 0, total * sizeof(DSPComplex));
    
    // Perform addition using vDSP
    vDSP_zvadd(dsp_a, 1, dsp_b, 1, dsp_c, 1, total);
    
    // Copy result back
    for (size_t i = 0; i < total; i++) {
        c[i] = from_dsp_complex(dsp_c[i]);
    }
    
    free(dsp_c);
    free(dsp_a);
    free(dsp_b);
    return NUMERICAL_SUCCESS;
}

numerical_error_t numerical_vector_dot_accelerate(const ComplexFloat* a,
                                           const ComplexFloat* b,
                                           ComplexFloat* result,
                                           size_t length) {
    if (!backend_state.initialized) {
        return NUMERICAL_ERROR_INVALID_STATE;
    }
    
    if (!a || !b || !result) {
        return NUMERICAL_ERROR_INVALID_ARGUMENT;
    }
    
    DSPComplex *dsp_a = NULL, *dsp_b = NULL;
    if (posix_memalign((void**)&dsp_a, DSP_ALIGNMENT, length * sizeof(DSPComplex)) != 0 ||
        posix_memalign((void**)&dsp_b, DSP_ALIGNMENT, length * sizeof(DSPComplex)) != 0) {
        free(dsp_a);
        free(dsp_b);
        return NUMERICAL_ERROR_MEMORY;
    }
    
    // Convert inputs using our conversion utilities
    for (size_t i = 0; i < length; i++) {
        dsp_a[i] = to_dsp_complex(a[i]);
        dsp_b[i] = to_dsp_complex(b[i]);
    }
    
    // Allocate aligned output buffer
    DSPComplex* dot = NULL;
    if (posix_memalign((void**)&dot, DSP_ALIGNMENT, sizeof(DSPComplex)) != 0) {
        free(dsp_a);
        free(dsp_b);
        return NUMERICAL_ERROR_MEMORY;
    }
    
    // Initialize output buffer to zero
    memset(dot, 0, sizeof(DSPComplex));
    
    // Compute dot product using vDSP
    vDSP_zdotpr(dsp_a, 1, dsp_b, 1, dot, length);
    
    // Copy result back
    *result = from_dsp_complex(*dot);
    
    free(dot);
    free(dsp_a);
    free(dsp_b);
    return NUMERICAL_SUCCESS;
}

numerical_error_t numerical_svd_accelerate(const ComplexFloat* a,
                                    ComplexFloat* u,
                                    float* s,
                                    ComplexFloat* vt,
                                    size_t m,
                                    size_t n) {
    if (!backend_state.initialized) {
        return NUMERICAL_ERROR_INVALID_STATE;
    }
    
    if (!a || !u || !s || !vt) {
        return NUMERICAL_ERROR_INVALID_ARGUMENT;
    }

    if (backend_state.has_lapack) {
        bool success = lapack_svd(a, m, n, u, s, vt, LAPACK_ROW_MAJOR);
        if (!success) {
            lapack_status_t status = lapack_get_last_status();
            switch (status) {
                case LAPACK_MEMORY_ERROR:
                    return NUMERICAL_ERROR_MEMORY;
                case LAPACK_NOT_CONVERGENT:
                    return NUMERICAL_ERROR_COMPUTATION;
                default:
                    return NUMERICAL_ERROR_BACKEND;
            }
        }
        return NUMERICAL_SUCCESS;
    }

    // If LAPACK is not available, fall back to CPU implementation
    return numerical_svd_cpu(a, u, s, vt, m, n);
}

numerical_error_t get_numerical_metrics_accelerate(numerical_metrics_t* metrics) {
    if (!backend_state.initialized) {
        return NUMERICAL_ERROR_INVALID_STATE;
    }
    
    if (!metrics) {
        return NUMERICAL_ERROR_INVALID_ARGUMENT;
    }
    
    *metrics = backend_state.metrics;
    return NUMERICAL_SUCCESS;
}

numerical_error_t reset_numerical_metrics_accelerate(void) {
    if (!backend_state.initialized) {
        return NUMERICAL_ERROR_INVALID_STATE;
    }
    
    memset(&backend_state.metrics, 0, sizeof(numerical_metrics_t));
    return NUMERICAL_SUCCESS;
}

numerical_error_t get_last_numerical_error_accelerate(void) {
    return backend_state.last_error;
}

numerical_error_t is_backend_available(numerical_backend_t backend) {
    switch (backend) {
        case NUMERICAL_BACKEND_ACCELERATE:
            #ifdef __APPLE__
            return NUMERICAL_SUCCESS;
            #else
            return NUMERICAL_ERROR_NOT_IMPLEMENTED;
            #endif
        case NUMERICAL_BACKEND_CPU:
            return NUMERICAL_SUCCESS;
        default:
            return NUMERICAL_ERROR_NOT_IMPLEMENTED;
    }
}

#else // !__APPLE__

// Stub implementations for non-Apple platforms
bool initialize_numerical_backend_accelerate(const numerical_config_t* config) {
    (void)config;
    return false;
}

void shutdown_numerical_backend_accelerate(void) {}

numerical_error_t numerical_matrix_multiply_accelerate(const ComplexFloat* a,
                                                const ComplexFloat* b,
                                                ComplexFloat* c,
                                                size_t m, size_t k, size_t n,
                                                bool transpose_a,
                                                bool transpose_b) {
    (void)a; (void)b; (void)c; (void)m; (void)k; (void)n;
    (void)transpose_a; (void)transpose_b;
    return NUMERICAL_ERROR_NOT_IMPLEMENTED;
}

numerical_error_t numerical_matrix_add_accelerate(const ComplexFloat* a,
                                           const ComplexFloat* b,
                                           ComplexFloat* c,
                                           size_t rows,
                                           size_t cols) {
    (void)a; (void)b; (void)c; (void)rows; (void)cols;
    return NUMERICAL_ERROR_NOT_IMPLEMENTED;
}

numerical_error_t numerical_vector_dot_accelerate(const ComplexFloat* a,
                                           const ComplexFloat* b,
                                           ComplexFloat* result,
                                           size_t length) {
    (void)a; (void)b; (void)result; (void)length;
    return NUMERICAL_ERROR_NOT_IMPLEMENTED;
}

numerical_error_t numerical_svd_accelerate(const ComplexFloat* a,
                                    ComplexFloat* u,
                                    float* s,
                                    ComplexFloat* vt,
                                    size_t m,
                                    size_t n) {
    (void)a; (void)u; (void)s; (void)vt; (void)m; (void)n;
    return NUMERICAL_ERROR_NOT_IMPLEMENTED;
}

numerical_error_t get_numerical_metrics_accelerate(numerical_metrics_t* metrics) {
    (void)metrics;
    return NUMERICAL_ERROR_NOT_IMPLEMENTED;
}

numerical_error_t reset_numerical_metrics_accelerate(void) {
    return NUMERICAL_ERROR_NOT_IMPLEMENTED;
}

numerical_error_t get_last_numerical_error_accelerate(void) {
    return NUMERICAL_ERROR_NOT_IMPLEMENTED;
}

numerical_error_t is_backend_available(numerical_backend_t backend) {
    (void)backend;
    return NUMERICAL_ERROR_NOT_IMPLEMENTED;
}

#endif // !__APPLE__
