#include "quantum_geometric/core/quantum_geometric_tensor.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/memory_pool.h"
#include "quantum_geometric/core/simd_operations.h"
#include "quantum_geometric/core/quantum_types.h"
#include "quantum_geometric/core/error_handling.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
typedef int32_t lapack_int;
#else
#include <lapacke.h>
typedef int lapack_int;
#endif

#define LAPACK_ROW_MAJOR 101
#ifdef _OPENMP
#include <omp.h>
#endif

// LAPACK function declarations
#ifdef __APPLE__
// LAPACK functions are already declared in Accelerate framework
#else
extern void LAPACKE_sgesvd(int matrix_layout, char jobu, char jobvt,
                          lapack_int m, lapack_int n, float* a,
                          lapack_int lda, float* s, float* u,
                          lapack_int ldu, float* vt, lapack_int ldvt,
                          float* superb);

extern void LAPACKE_ssyev(int matrix_layout, char jobz, char uplo,
                         lapack_int n, float* a, lapack_int lda,
                         float* w);

extern void LAPACKE_sgeqrf(int matrix_layout, lapack_int m, lapack_int n,
                          float* a, lapack_int lda, float* tau);

extern void LAPACKE_sgetrf(int matrix_layout, lapack_int m, lapack_int n,
                          float* a, lapack_int lda, lapack_int* ipiv);
#endif

#ifdef __APPLE__
#ifdef __arm64__
#include "quantum_geometric/core/amx_operations.h"
// AMX tile configuration for Apple Silicon
#define AMX_TILE_M 32
#define AMX_TILE_N 32  
#define AMX_TILE_K 32
#endif
#endif

// Forward declarations of internal functions
static void neon_matrix_multiply(const ComplexFloat* a, const ComplexFloat* b, ComplexFloat* c,
                               size_t m, size_t n, size_t k);

// BLAS function declarations
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
extern void cblas_sgemm(const enum CBLAS_ORDER Order,
                       const enum CBLAS_TRANSPOSE TransA,
                       const enum CBLAS_TRANSPOSE TransB,
                       const int M, const int N, const int K,
                       const float alpha, const float *A, const int lda,
                       const float *B, const int ldb,
                       const float beta, float *C, const int ldc);
#endif

// NEON SIMD configuration for ARM
#ifdef __ARM_NEON
#include <arm_neon.h>
#define SIMD_WIDTH 4  // 4 floats per NEON vector
#else
#define SIMD_WIDTH 1  // Fallback to scalar operations
#endif

// Cache-friendly block sizes optimized for ARM
#define BLOCK_SIZE 64
#define L2_BLOCK_SIZE 256
#define L3_BLOCK_SIZE 1024

// Optimized matrix multiply for ARM using NEON SIMD
static void neon_matrix_multiply(const ComplexFloat* a, const ComplexFloat* b, ComplexFloat* c,
                               size_t m, size_t n, size_t k) {
#ifdef __ARM_NEON
    // Process 4x4 blocks using NEON SIMD
    for (size_t i = 0; i < m; i += 4) {
        for (size_t j = 0; j < n; j += 4) {
            float32x4x2_t sum[4] = {
                {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)},
                {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)},
                {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)},
                {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)}
            };
            
            for (size_t p = 0; p < k; p++) {
                // Load 4 complex numbers from matrix A (8 floats total)
                float32x4x2_t a_vec = vld2q_f32((const float32_t*)&a[i * k + p]);
                // Load 4 complex numbers from matrix B (8 floats total)
                float32x4x2_t b_vec = vld2q_f32((const float32_t*)&b[p * n + j]);
                
                // Complex multiplication using NEON:
                // (a.real + j*a.imag) * (b.real + j*b.imag) = 
                // (a.real*b.real - a.imag*b.imag) + j*(a.real*b.imag + a.imag*b.real)
                
                // Compute products
                float32x4_t real_prod = vmulq_f32(a_vec.val[0], b_vec.val[0]);
                float32x4_t imag_prod = vmulq_f32(a_vec.val[1], b_vec.val[1]);
                float32x4_t real_imag_prod = vmulq_f32(a_vec.val[0], b_vec.val[1]);
                float32x4_t imag_real_prod = vmulq_f32(a_vec.val[1], b_vec.val[0]);
                
                // Accumulate real and imaginary parts
                sum[0].val[0] = vaddq_f32(sum[0].val[0], vsubq_f32(real_prod, imag_prod));
                sum[0].val[1] = vaddq_f32(sum[0].val[1], vaddq_f32(real_imag_prod, imag_real_prod));
            }
            
            // Store results back to memory
            for (size_t ii = 0; ii < 4; ii++) {
                vst2q_f32((float32_t*)&c[(i + ii) * n + j], sum[ii]);
            }
        }
    }
#else
    // Fallback to scalar operations
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            ComplexFloat sum = {0.0f, 0.0f};
            for (size_t p = 0; p < k; p++) {
                sum = complex_float_add(sum, 
                    complex_float_multiply(a[i * k + p], b[p * n + j]));
            }
            c[i * n + j] = sum;
        }
    }
#endif
}

#ifdef __APPLE__
#ifdef __arm64__
// Convert ComplexFloat matrices to float arrays for AMX
static void convert_to_float_array(float* dst, const ComplexFloat* src, size_t elements) {
    for (size_t i = 0; i < elements; i++) {
        dst[i * 2] = src[i].real;
        dst[i * 2 + 1] = src[i].imag;
    }
}

static void convert_from_float_array(ComplexFloat* dst, const float* src, size_t elements) {
    for (size_t i = 0; i < elements; i++) {
        dst[i].real = src[i * 2];
        dst[i].imag = src[i * 2 + 1];
    }
}

// AMX-accelerated matrix multiplication for complex numbers
static void amx_complex_matrix_multiply(const ComplexFloat* a, const ComplexFloat* b, ComplexFloat* c,
                                      size_t m, size_t n, size_t k) {
    // Convert complex matrices to float arrays
    const size_t a_size = m * k;
    const size_t b_size = k * n;
    const size_t c_size = m * n;
    
    float* a_float = aligned_alloc(AMX_ALIGNMENT, a_size * 2 * sizeof(float));
    float* b_float = aligned_alloc(AMX_ALIGNMENT, b_size * 2 * sizeof(float));
    float* c_float = aligned_alloc(AMX_ALIGNMENT, c_size * 2 * sizeof(float));
    
    if (!a_float || !b_float || !c_float) {
        if (a_float) free(a_float);
        if (b_float) free(b_float);
        if (c_float) free(c_float);
        return;
    }
    
    convert_to_float_array(a_float, a, a_size);
    convert_to_float_array(b_float, b, b_size);
    
    // Initialize AMX
    if (amx_init() != 0) {
        free(a_float);
        free(b_float);
        free(c_float);
        return;
    }
    
    // Process matrix in tiles
    for (size_t i = 0; i < m; i += AMX_TILE_M) {
        for (size_t j = 0; j < n; j += AMX_TILE_N) {
            for (size_t k_idx = 0; k_idx < k; k_idx += AMX_TILE_K) {
                size_t i_end = (i + AMX_TILE_M < m) ? AMX_TILE_M : m - i;
                size_t j_end = (j + AMX_TILE_N < n) ? AMX_TILE_N : n - j;
                size_t k_end = (k_idx + AMX_TILE_K < k) ? AMX_TILE_K : k - k_idx;
                
                // Load tiles
                amx_ldx(&a_float[(i * k + k_idx) * 2], 0);
                amx_ldy(&b_float[(k_idx * n + j) * 2], 0);
                
                // Perform complex multiplication:
                // (a.real + j*a.imag) * (b.real + j*b.imag) = 
                // (a.real*b.real - a.imag*b.imag) + j*(a.real*b.imag + a.imag*b.real)
                
                // Real part: a.real*b.real - a.imag*b.imag
                amx_fma64(0, 0, 0); // a.real*b.real
                amx_fma64(1, 1, 1); // a.imag*b.imag (negative)
                
                // Imaginary part: a.real*b.imag + a.imag*b.real
                amx_fma64(0, 1, 2); // a.real*b.imag
                amx_fma64(1, 0, 2); // a.imag*b.real
                
                // Store result tile
                amx_stz(&c_float[(i * n + j) * 2], 0);
            }
        }
    }
    
    // Cleanup AMX state
    amx_stop();
    
    // Convert result back to complex
    convert_from_float_array(c, c_float, c_size);
    
    free(a_float);
    free(b_float);
    free(c_float);
}
#endif
#endif

// Tensor creation/destruction
qgt_error_t geometric_tensor_create(quantum_geometric_tensor_t** tensor,
                                  geometric_tensor_type_t type,
                                  const size_t* dimensions,
                                  size_t rank) {
    if (!tensor || !dimensions || rank == 0) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // Calculate total size with overflow checking
    size_t total_elements = 1;
    for (size_t i = 0; i < rank; i++) {
        if (dimensions[i] == 0) {
            return QGT_ERROR_INVALID_DIMENSION;
        }
        
        // Check for overflow
        if (total_elements > SIZE_MAX / dimensions[i]) {
            return QGT_ERROR_ALLOCATION_FAILED;
        }
        total_elements *= dimensions[i];
    }

    // Allocate aligned memory for better SIMD performance
    *tensor = (quantum_geometric_tensor_t*)aligned_alloc(64, sizeof(quantum_geometric_tensor_t));
    if (!*tensor) {
        return QGT_ERROR_ALLOCATION_FAILED;
    }

    // Allocate aligned dimensions array
    (*tensor)->dimensions = (size_t*)aligned_alloc(64, rank * sizeof(size_t));
    if (!(*tensor)->dimensions) {
        free(*tensor);
        return QGT_ERROR_ALLOCATION_FAILED;
    }

    // Use SIMD to copy dimensions if rank is large enough
    if (rank >= 16) { // Only use SIMD for larger arrays
        #pragma omp simd
        for (size_t i = 0; i < rank; i++) {
            (*tensor)->dimensions[i] = dimensions[i];
        }
    } else {
        memcpy((*tensor)->dimensions, dimensions, rank * sizeof(size_t));
    }

    // Allocate aligned components array with padding for SIMD operations
    const size_t alignment = 64; // AVX-512 alignment
    const size_t padded_size = ((total_elements * sizeof(ComplexFloat) + alignment - 1) / alignment) * alignment;
    (*tensor)->components = (ComplexFloat*)aligned_alloc(alignment, padded_size);
    if (!(*tensor)->components) {
        free((*tensor)->dimensions);
        free(*tensor);
        return QGT_ERROR_ALLOCATION_FAILED;
    }

    // Initialize components to zero using SIMD
    const size_t block_size = 1024; // Process in blocks for better cache utilization
    const ComplexFloat zero = {0.0f, 0.0f};
    
    #pragma omp parallel
    {
        #pragma omp for schedule(guided)
        for (size_t i = 0; i < total_elements; i += block_size) {
            const size_t current_block = (i + block_size <= total_elements) ? block_size : (total_elements - i);
            
            #pragma omp simd
            for (size_t j = 0; j < current_block; j++) {
                (*tensor)->components[i + j] = zero;
            }
        }
    }

    (*tensor)->rank = rank;
    (*tensor)->type = type;
    (*tensor)->total_elements = total_elements;
    (*tensor)->is_symmetric = false;
    (*tensor)->hardware = HARDWARE_TYPE_SIMULATOR;
    (*tensor)->auxiliary_data = NULL;

    return QGT_SUCCESS;
}

void geometric_tensor_destroy(quantum_geometric_tensor_t* tensor) {
    if (!tensor) {
        return;
    }

    if (tensor->dimensions) {
        free(tensor->dimensions);
    }
    if (tensor->components) {
        free(tensor->components);
    }
    if (tensor->auxiliary_data) {
        free(tensor->auxiliary_data);
    }
    free(tensor);
}

qgt_error_t geometric_tensor_multiply(quantum_geometric_tensor_t* result,
                                    const quantum_geometric_tensor_t* a,
                                    const quantum_geometric_tensor_t* b) {
    if (!result || !a || !b) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // Verify dimensions are compatible for matrix multiplication
    if (a->rank != 2 || b->rank != 2 || result->rank != 2) {
        return QGT_ERROR_INVALID_DIMENSION;
    }

    const size_t m = a->dimensions[0];
    const size_t k = a->dimensions[1];
    const size_t n = b->dimensions[1];

    if (b->dimensions[0] != k || result->dimensions[0] != m || result->dimensions[1] != n) {
        return QGT_ERROR_DIMENSION_MISMATCH;
    }

#ifdef __APPLE__
#ifdef __arm64__
    // Use AMX acceleration on Apple Silicon
    amx_complex_matrix_multiply(a->components, b->components, result->components, m, n, k);
#else
    // Use NEON SIMD on other ARM processors
    neon_matrix_multiply(a->components, b->components, result->components, m, n, k);
#endif
#else
    // Fallback to scalar operations
    neon_matrix_multiply(a->components, b->components, result->components, m, n, k);
#endif

    return QGT_SUCCESS;
}

qgt_error_t geometric_tensor_add(quantum_geometric_tensor_t* result,
                               const quantum_geometric_tensor_t* a,
                               const quantum_geometric_tensor_t* b) {
    if (!result || !a || !b) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // Verify dimensions match
    if (a->rank != b->rank || a->rank != result->rank) {
        return QGT_ERROR_DIMENSION_MISMATCH;
    }

    for (size_t i = 0; i < a->rank; i++) {
        if (a->dimensions[i] != b->dimensions[i] || a->dimensions[i] != result->dimensions[i]) {
            return QGT_ERROR_DIMENSION_MISMATCH;
        }
    }

    // Perform addition using SIMD
    #pragma omp parallel for simd schedule(guided)
    for (size_t i = 0; i < a->total_elements; i++) {
        result->components[i] = complex_float_add(a->components[i], b->components[i]);
    }

    return QGT_SUCCESS;
}

qgt_error_t geometric_tensor_subtract(quantum_geometric_tensor_t* result,
                                    const quantum_geometric_tensor_t* a,
                                    const quantum_geometric_tensor_t* b) {
    if (!result || !a || !b) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // Verify dimensions match
    if (a->rank != b->rank || a->rank != result->rank) {
        return QGT_ERROR_DIMENSION_MISMATCH;
    }

    for (size_t i = 0; i < a->rank; i++) {
        if (a->dimensions[i] != b->dimensions[i] || a->dimensions[i] != result->dimensions[i]) {
            return QGT_ERROR_DIMENSION_MISMATCH;
        }
    }

    // Perform subtraction using SIMD
    #pragma omp parallel for simd schedule(guided)
    for (size_t i = 0; i < a->total_elements; i++) {
        result->components[i] = complex_float_subtract(a->components[i], b->components[i]);
    }

    return QGT_SUCCESS;
}

qgt_error_t geometric_tensor_scale(quantum_geometric_tensor_t* result,
                                  const quantum_geometric_tensor_t* tensor,
                                  ComplexFloat scalar) {
    if (!result || !tensor) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // Verify dimensions match
    if (result->rank != tensor->rank) {
        return QGT_ERROR_DIMENSION_MISMATCH;
    }

    for (size_t i = 0; i < tensor->rank; i++) {
        if (result->dimensions[i] != tensor->dimensions[i]) {
            return QGT_ERROR_DIMENSION_MISMATCH;
        }
    }

    // Scale components using SIMD
    #pragma omp parallel for simd schedule(guided)
    for (size_t i = 0; i < tensor->total_elements; i++) {
        result->components[i] = complex_float_multiply(tensor->components[i], scalar);
    }

    return QGT_SUCCESS;
}

// Helper function to convert complex tensor to real array for LAPACK
static void convert_complex_to_real_array(float* real_array, const ComplexFloat* complex_array, size_t elements) {
    for (size_t i = 0; i < elements; i++) {
        real_array[2*i] = complex_array[i].real;
        real_array[2*i + 1] = complex_array[i].imag;
    }
}

// Helper function to convert real array back to complex tensor
static void convert_real_to_complex_array(ComplexFloat* complex_array, const float* real_array, size_t elements) {
    for (size_t i = 0; i < elements; i++) {
        complex_array[i].real = real_array[2*i];
        complex_array[i].imag = real_array[2*i + 1];
    }
}

qgt_error_t geometric_tensor_initialize_zero(quantum_geometric_tensor_t* tensor) {
    if (!tensor) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    const ComplexFloat zero = {0.0f, 0.0f};
    
    #pragma omp parallel for simd schedule(guided)
    for (size_t i = 0; i < tensor->total_elements; i++) {
        tensor->components[i] = zero;
    }

    return QGT_SUCCESS;
}

qgt_error_t geometric_tensor_svd(quantum_geometric_tensor_t* u,
                                quantum_geometric_tensor_t* s,
                                quantum_geometric_tensor_t* v,
                                const quantum_geometric_tensor_t* tensor) {
    if (!u || !s || !v || !tensor) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // SVD only works on 2D tensors (matrices)
    if (tensor->rank != 2) {
        return QGT_ERROR_INVALID_DIMENSION;
    }

    const size_t m = tensor->dimensions[0];
    const size_t n = tensor->dimensions[1];
    const size_t min_dim = (m < n) ? m : n;

    // Verify output dimensions
    if (u->dimensions[0] != m || u->dimensions[1] != m ||
        s->dimensions[0] != min_dim || s->dimensions[1] != min_dim ||
        v->dimensions[0] != n || v->dimensions[1] != n) {
        return QGT_ERROR_DIMENSION_MISMATCH;
    }

    // Allocate work arrays for LAPACK
    float* a = aligned_alloc(64, 2 * m * n * sizeof(float));  // Complex matrix as real array
    float* s_values = aligned_alloc(64, min_dim * sizeof(float));  // Singular values
    float* u_mat = aligned_alloc(64, 2 * m * m * sizeof(float));  // Left singular vectors
    float* vt_mat = aligned_alloc(64, 2 * n * n * sizeof(float)); // Right singular vectors
    float* superb = aligned_alloc(64, min_dim * sizeof(float));   // Superdiagonal elements

    if (!a || !s_values || !u_mat || !vt_mat || !superb) {
        if (a) free(a);
        if (s_values) free(s_values);
        if (u_mat) free(u_mat);
        if (vt_mat) free(vt_mat);
        if (superb) free(superb);
        return QGT_ERROR_ALLOCATION_FAILED;
    }

    // Convert input tensor to real array format for LAPACK
    convert_complex_to_real_array(a, tensor->components, m * n);

    // Perform SVD using LAPACK
#ifdef __APPLE__
    char jobu = 'A';
    char jobvt = 'A';
    lapack_int m_int = m;
    lapack_int n_int = n;
    lapack_int lda = n;
    lapack_int ldu = m;
    lapack_int ldvt = n;
    lapack_int info;
    lapack_int lwork = -1;
    float work_query;

    // Query optimal workspace size
    sgesvd_(&jobu, &jobvt, &m_int, &n_int, a, &lda, s_values, u_mat, &ldu, 
            vt_mat, &ldvt, &work_query, &lwork, &info);

    lwork = (int)work_query;
    float* work = (float*)malloc(lwork * sizeof(float));
    if (!work) {
        free(a);
        free(s_values);
        free(u_mat);
        free(vt_mat);
        free(superb);
        return QGT_ERROR_ALLOCATION_FAILED;
    }

    // Compute SVD
    sgesvd_(&jobu, &jobvt, &m_int, &n_int, a, &lda, s_values, u_mat, &ldu,
            vt_mat, &ldvt, work, &lwork, &info);

    free(work);
#else
    lapack_int info = LAPACKE_sgesvd(LAPACK_ROW_MAJOR, 'A', 'A', m, n,
                                    a, n, s_values, u_mat, m,
                                    vt_mat, n, superb);
#endif

    if (info != 0) {
        free(a);
        free(s_values);
        free(u_mat);
        free(vt_mat);
        free(superb);
        return QGT_ERROR_SVD_FAILED;
    }

    // Convert results back to complex format
    convert_real_to_complex_array(u->components, u_mat, m * m);
    convert_real_to_complex_array(v->components, vt_mat, n * n);

    // Build diagonal matrix S from singular values
    geometric_tensor_initialize_zero(s);
    for (size_t i = 0; i < min_dim; i++) {
        s->components[i * min_dim + i].real = s_values[i];
        s->components[i * min_dim + i].imag = 0.0f;
    }

    // Clean up
    free(a);
    free(s_values);
    free(u_mat);
    free(vt_mat);
    free(superb);

    return QGT_SUCCESS;
}

qgt_error_t geometric_tensor_qr(quantum_geometric_tensor_t* q,
                               quantum_geometric_tensor_t* r,
                               const quantum_geometric_tensor_t* tensor) {
    if (!q || !r || !tensor) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // QR decomposition works on matrices (2D tensors)
    if (tensor->rank != 2) {
        return QGT_ERROR_INVALID_DIMENSION;
    }

    const size_t m = tensor->dimensions[0];
    const size_t n = tensor->dimensions[1];

    // Verify output dimensions
    if (q->dimensions[0] != m || q->dimensions[1] != m ||
        r->dimensions[0] != m || r->dimensions[1] != n) {
        return QGT_ERROR_DIMENSION_MISMATCH;
    }

    // Allocate work arrays
    float* a = aligned_alloc(64, 2 * m * n * sizeof(float));  // Matrix copy for LAPACK
    float* tau = aligned_alloc(64, n * sizeof(float));        // Scalar factors for reflectors
    float* work = aligned_alloc(64, n * sizeof(float));       // Work array

    if (!a || !tau || !work) {
        if (a) free(a);
        if (tau) free(tau);
        if (work) free(work);
        return QGT_ERROR_ALLOCATION_FAILED;
    }

    // Convert input tensor to real array format
    convert_complex_to_real_array(a, tensor->components, m * n);

    // Compute QR decomposition using LAPACK
#ifdef __APPLE__
    __CLPK_integer m_int = m;
    __CLPK_integer n_int = n;
    __CLPK_integer lda = n;
    __CLPK_integer info;
    __CLPK_integer lwork = -1;
    float work_query;

    // Query optimal workspace size
    sgeqrf_(&m_int, &n_int, a, &lda, tau, &work_query, &lwork, &info);

    lwork = (int)work_query;
    float* work_array = (float*)malloc(lwork * sizeof(float));
    if (!work_array) {
        free(a);
        free(tau);
        free(work);
        return QGT_ERROR_ALLOCATION_FAILED;
    }

    // Compute QR decomposition
    sgeqrf_(&m_int, &n_int, a, &lda, tau, work_array, &lwork, &info);
    free(work_array);
#else
    lapack_int info = LAPACKE_sgeqrf(LAPACK_ROW_MAJOR, m, n,
                                    a, n, tau, work, n);
#endif

    if (info != 0) {
        free(a);
        free(tau);
        free(work);
        return QGT_ERROR_QR_FAILED;
    }

    // Extract R (upper triangular part)
    geometric_tensor_initialize_zero(r);
    for (size_t i = 0; i < m; i++) {
        for (size_t j = i; j < n; j++) {  // Only upper triangular part
            r->components[i * n + j].real = a[i * n + j];
            r->components[i * n + j].imag = 0.0f;
        }
    }

    // Compute Q explicitly using orgqr
    memcpy(q->components, a, m * n * sizeof(float));
#ifdef __APPLE__
    // Query optimal workspace size for orgqr
    lwork = -1;
    sorgqr_(&m_int, &m_int, &n_int, (float*)q->components, &lda, tau,
            &work_query, &lwork, &info);

    lwork = (int)work_query;
    work_array = (float*)malloc(lwork * sizeof(float));
    if (!work_array) {
        free(a);
        free(tau);
        free(work);
        return QGT_ERROR_ALLOCATION_FAILED;
    }

    // Compute Q matrix
    sorgqr_(&m_int, &m_int, &n_int, (float*)q->components, &lda, tau,
            work_array, &lwork, &info);
    free(work_array);
#else
    info = LAPACKE_sorgqr(LAPACK_ROW_MAJOR, m, m, n,
                         (float*)q->components, n, tau);
#endif

    if (info != 0) {
        free(a);
        free(tau);
        free(work);
        return QGT_ERROR_QR_FAILED;
    }

    // Clean up
    free(a);
    free(tau);
    free(work);

    return QGT_SUCCESS;
}

qgt_error_t geometric_tensor_eigendecomposition(quantum_geometric_tensor_t* eigenvectors,
                                              ComplexFloat* eigenvalues,
                                              const quantum_geometric_tensor_t* tensor) {
    if (!eigenvectors || !eigenvalues || !tensor) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // Only works on square matrices
    if (tensor->rank != 2 || tensor->dimensions[0] != tensor->dimensions[1]) {
        return QGT_ERROR_INVALID_DIMENSION;
    }

    const size_t n = tensor->dimensions[0];

    // Verify output dimensions
    if (eigenvectors->rank != 2 || 
        eigenvectors->dimensions[0] != n || 
        eigenvectors->dimensions[1] != n) {
        return QGT_ERROR_DIMENSION_MISMATCH;
    }

    // Allocate work arrays
    float* a = aligned_alloc(64, 2 * n * n * sizeof(float));  // Matrix copy for LAPACK
    float* w = aligned_alloc(64, n * sizeof(float));          // Eigenvalues
    float* work = aligned_alloc(64, 3 * n * sizeof(float));   // Work array

    if (!a || !w || !work) {
        if (a) free(a);
        if (w) free(w);
        if (work) free(work);
        return QGT_ERROR_ALLOCATION_FAILED;
    }

    // Convert input tensor to real array format
    convert_complex_to_real_array(a, tensor->components, n * n);

    // Compute eigendecomposition using LAPACK
#ifdef __APPLE__
    char jobz = 'V';
    char uplo = 'U';
    __CLPK_integer n_int = n;
    __CLPK_integer lda = n;
    __CLPK_integer info;
    __CLPK_integer lwork = -1;
    float work_query;

    // Query optimal workspace size
    ssyev_(&jobz, &uplo, &n_int, a, &lda, w, &work_query, &lwork, &info);

    lwork = (int)work_query;
    float* work_array = (float*)malloc(lwork * sizeof(float));
    if (!work_array) {
        free(a);
        free(w);
        free(work);
        return QGT_ERROR_ALLOCATION_FAILED;
    }

    // Compute eigendecomposition
    ssyev_(&jobz, &uplo, &n_int, a, &lda, w, work_array, &lwork, &info);
    free(work_array);
#else
    lapack_int info = LAPACKE_ssyev(LAPACK_ROW_MAJOR, 'V', 'U', n,
                                   a, n, w, work, 3 * n);
#endif

    if (info != 0) {
        free(a);
        free(w);
        free(work);
        return QGT_ERROR_EIGENDECOMPOSITION_FAILED;
    }

    // Convert eigenvectors back to complex format
    convert_real_to_complex_array(eigenvectors->components, a, n * n);

    // Copy eigenvalues to output array
    for (size_t i = 0; i < n; i++) {
        eigenvalues[i].real = w[i];
        eigenvalues[i].imag = 0.0f;  // Eigenvalues are real for Hermitian matrices
    }

    // Clean up
    free(a);
    free(w);
    free(work);

    return QGT_SUCCESS;
}

qgt_error_t geometric_tensor_norm(float* norm,
                                const quantum_geometric_tensor_t* tensor) {
    if (!norm || !tensor) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // Compute Frobenius norm: sqrt(sum_i |a_i|^2)
    float sum = 0.0f;
    
    #pragma omp parallel for simd reduction(+:sum) schedule(guided)
    for (size_t i = 0; i < tensor->total_elements; i++) {
        sum += tensor->components[i].real * tensor->components[i].real +
               tensor->components[i].imag * tensor->components[i].imag;
    }

    *norm = sqrtf(sum);
    return QGT_SUCCESS;
}

qgt_error_t geometric_tensor_trace(ComplexFloat* trace,
                                 const quantum_geometric_tensor_t* tensor) {
    if (!trace || !tensor) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // Trace only defined for square matrices
    if (tensor->rank != 2 || tensor->dimensions[0] != tensor->dimensions[1]) {
        return QGT_ERROR_INVALID_DIMENSION;
    }

    const size_t n = tensor->dimensions[0];
    ComplexFloat sum = {0.0f, 0.0f};

    // Sum diagonal elements
    for (size_t i = 0; i < n; i++) {
        sum = complex_float_add(sum, tensor->components[i * n + i]);
    }

    *trace = sum;
    return QGT_SUCCESS;
}

qgt_error_t geometric_tensor_determinant(ComplexFloat* determinant,
                                       const quantum_geometric_tensor_t* tensor) {
    if (!determinant || !tensor) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // Determinant only defined for square matrices
    if (tensor->rank != 2 || tensor->dimensions[0] != tensor->dimensions[1]) {
        return QGT_ERROR_INVALID_DIMENSION;
    }

    const size_t n = tensor->dimensions[0];

    // For 1x1 matrix
    if (n == 1) {
        *determinant = tensor->components[0];
        return QGT_SUCCESS;
    }

    // For larger matrices, use LU decomposition
    float* a = aligned_alloc(64, 2 * n * n * sizeof(float));
    lapack_int* ipiv = aligned_alloc(64, n * sizeof(lapack_int));

    if (!a || !ipiv) {
        if (a) free(a);
        if (ipiv) free(ipiv);
        return QGT_ERROR_ALLOCATION_FAILED;
    }

    // Convert to real array format for LAPACK
    convert_complex_to_real_array(a, tensor->components, n * n);

    // Compute LU decomposition
    lapack_int info = LAPACKE_sgetrf(LAPACK_ROW_MAJOR, n, n, a, n, ipiv);

    if (info != 0) {
        free(a);
        free(ipiv);
        return QGT_ERROR_LU_FAILED;
    }

    // Compute determinant from diagonal elements of U
    ComplexFloat det = {1.0f, 0.0f};
    int sign = 1;
    
    for (size_t i = 0; i < n; i++) {
        // Account for row permutations
        if (ipiv[i] != (lapack_int)(i + 1)) sign = -sign;
        
        // Multiply by diagonal element
        ComplexFloat diag = {a[i * n + i], a[i * n + i + 1]};
        det = complex_float_multiply(det, diag);
    }

    // Apply sign from permutations
    if (sign < 0) {
        det.real = -det.real;
        det.imag = -det.imag;
    }

    *determinant = det;

    free(a);
    free(ipiv);
    return QGT_SUCCESS;
}

bool geometric_tensor_is_hermitian(const quantum_geometric_tensor_t* tensor) {
    if (!tensor || tensor->rank != 2 || 
        tensor->dimensions[0] != tensor->dimensions[1]) {
        return false;
    }

    const size_t n = tensor->dimensions[0];

    for (size_t i = 0; i < n; i++) {
        for (size_t j = i + 1; j < n; j++) {
            const ComplexFloat a = tensor->components[i * n + j];
            const ComplexFloat b = tensor->components[j * n + i];
            if (fabsf(a.real - b.real) > 1e-6f || 
                fabsf(a.imag + b.imag) > 1e-6f) {
                return false;
            }
        }
    }

    return true;
}

bool geometric_tensor_is_unitary(const quantum_geometric_tensor_t* tensor) {
    if (!tensor || tensor->rank != 2 || 
        tensor->dimensions[0] != tensor->dimensions[1]) {
        return false;
    }

    const size_t n = tensor->dimensions[0];
    
    // Compute U * U^†
    quantum_geometric_tensor_t* product;
    size_t dims[2] = {n, n};
    
    if (geometric_tensor_create(&product, GEOMETRIC_TENSOR_UNITARY, dims, 2) != QGT_SUCCESS) {
        return false;
    }

    // Create conjugate transpose of U
    quantum_geometric_tensor_t* adjoint;
    if (geometric_tensor_create(&adjoint, GEOMETRIC_TENSOR_UNITARY, dims, 2) != QGT_SUCCESS) {
        geometric_tensor_destroy(product);
        return false;
    }

    if (geometric_tensor_adjoint(adjoint, tensor) != QGT_SUCCESS ||
        geometric_tensor_multiply(product, tensor, adjoint) != QGT_SUCCESS) {
        geometric_tensor_destroy(product);
        geometric_tensor_destroy(adjoint);
        return false;
    }

    // Check if product is identity matrix
    bool is_unitary = true;
    for (size_t i = 0; i < n && is_unitary; i++) {
        for (size_t j = 0; j < n && is_unitary; j++) {
            const ComplexFloat val = product->components[i * n + j];
            const float expected_real = (i == j) ? 1.0f : 0.0f;
            if (fabsf(val.real - expected_real) > 1e-6f || 
                fabsf(val.imag) > 1e-6f) {
                is_unitary = false;
            }
        }
    }

    geometric_tensor_destroy(product);
    geometric_tensor_destroy(adjoint);
    return is_unitary;
}

bool geometric_tensor_is_positive_definite(const quantum_geometric_tensor_t* tensor) {
    if (!tensor || tensor->rank != 2 || 
        tensor->dimensions[0] != tensor->dimensions[1] ||
        !geometric_tensor_is_hermitian(tensor)) {
        return false;
    }

    const size_t n = tensor->dimensions[0];
    
    // Compute eigenvalues
    float* a = aligned_alloc(64, 2 * n * n * sizeof(float));
    float* w = aligned_alloc(64, n * sizeof(float));
    float* work = aligned_alloc(64, 3 * n * sizeof(float));

    if (!a || !w || !work) {
        if (a) free(a);
        if (w) free(w);
        if (work) free(work);
        return false;
    }

    // Convert to real array format for LAPACK
    convert_complex_to_real_array(a, tensor->components, n * n);

    // Compute eigenvalues
    lapack_int info = LAPACKE_ssyev(LAPACK_ROW_MAJOR, 'N', 'U', n,
                                   a, n, w, work, 3 * n);

    if (info != 0) {
        free(a);
        free(w);
        free(work);
        return false;
    }

    // Check if all eigenvalues are positive
    bool is_positive = true;
    for (size_t i = 0; i < n && is_positive; i++) {
        if (w[i] <= 0.0f) is_positive = false;
    }

    free(a);
    free(w);
    free(work);
    return is_positive;
}

qgt_error_t geometric_tensor_initialize_identity(quantum_geometric_tensor_t* tensor) {
    if (!tensor) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // Identity only defined for square matrices
    if (tensor->rank != 2 || tensor->dimensions[0] != tensor->dimensions[1]) {
        return QGT_ERROR_INVALID_DIMENSION;
    }

    const size_t n = tensor->dimensions[0];
    const ComplexFloat zero = {0.0f, 0.0f};
    const ComplexFloat one = {1.0f, 0.0f};

    // First zero out all elements
    #pragma omp parallel for simd schedule(guided)
    for (size_t i = 0; i < tensor->total_elements; i++) {
        tensor->components[i] = zero;
    }

    // Set diagonal elements to 1
    for (size_t i = 0; i < n; i++) {
        tensor->components[i * n + i] = one;
    }

    return QGT_SUCCESS;
}

qgt_error_t geometric_tensor_initialize_random(quantum_geometric_tensor_t* tensor,
                                             float min_val,
                                             float max_val) {
    if (!tensor) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    if (min_val >= max_val) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    const float range = max_val - min_val;
    
    #pragma omp parallel
    {
        // Each thread needs its own random state
        unsigned int seed = (unsigned int)time(NULL) ^ omp_get_thread_num();
        
        #pragma omp for schedule(guided)
        for (size_t i = 0; i < tensor->total_elements; i++) {
            // Generate random values between 0 and 1
            float real_rand = (float)rand_r(&seed) / RAND_MAX;
            float imag_rand = (float)rand_r(&seed) / RAND_MAX;
            
            // Scale to desired range
            tensor->components[i].real = min_val + real_rand * range;
            tensor->components[i].imag = min_val + imag_rand * range;
        }
    }

    return QGT_SUCCESS;
}

qgt_error_t geometric_tensor_transpose(quantum_geometric_tensor_t* result,
                                     const quantum_geometric_tensor_t* tensor) {
    if (!result || !tensor) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // Only works on 2D tensors
    if (tensor->rank != 2) {
        return QGT_ERROR_INVALID_DIMENSION;
    }

    const size_t m = tensor->dimensions[0];
    const size_t n = tensor->dimensions[1];

    // Verify output dimensions are transposed
    if (result->rank != 2 || 
        result->dimensions[0] != n || 
        result->dimensions[1] != m) {
        return QGT_ERROR_DIMENSION_MISMATCH;
    }

    // Perform transpose with cache-friendly blocking
    const size_t block_size = 32;  // Adjust based on cache size
    
    for (size_t i = 0; i < m; i += block_size) {
        for (size_t j = 0; j < n; j += block_size) {
            const size_t i_end = (i + block_size < m) ? i + block_size : m;
            const size_t j_end = (j + block_size < n) ? j + block_size : n;
            
            for (size_t ii = i; ii < i_end; ii++) {
                for (size_t jj = j; jj < j_end; jj++) {
                    result->components[jj * m + ii] = tensor->components[ii * n + jj];
                }
            }
        }
    }

    return QGT_SUCCESS;
}

qgt_error_t geometric_tensor_conjugate(quantum_geometric_tensor_t* result,
                                     const quantum_geometric_tensor_t* tensor) {
    if (!result || !tensor) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // Verify dimensions match
    if (result->rank != tensor->rank) {
        return QGT_ERROR_DIMENSION_MISMATCH;
    }

    for (size_t i = 0; i < tensor->rank; i++) {
        if (result->dimensions[i] != tensor->dimensions[i]) {
            return QGT_ERROR_DIMENSION_MISMATCH;
        }
    }

    // Compute complex conjugate using SIMD
    #pragma omp parallel for simd schedule(guided)
    for (size_t i = 0; i < tensor->total_elements; i++) {
        result->components[i].real = tensor->components[i].real;
        result->components[i].imag = -tensor->components[i].imag;
    }

    return QGT_SUCCESS;
}

qgt_error_t geometric_tensor_adjoint(quantum_geometric_tensor_t* result,
                                   const quantum_geometric_tensor_t* tensor) {
    if (!result || !tensor) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // Only works on 2D tensors
    if (tensor->rank != 2) {
        return QGT_ERROR_INVALID_DIMENSION;
    }

    const size_t m = tensor->dimensions[0];
    const size_t n = tensor->dimensions[1];

    // Verify output dimensions are transposed
    if (result->rank != 2 || 
        result->dimensions[0] != n || 
        result->dimensions[1] != m) {
        return QGT_ERROR_DIMENSION_MISMATCH;
    }

    // Perform conjugate transpose with cache-friendly blocking
    const size_t block_size = 32;  // Adjust based on cache size
    
    for (size_t i = 0; i < m; i += block_size) {
        for (size_t j = 0; j < n; j += block_size) {
            const size_t i_end = (i + block_size < m) ? i + block_size : m;
            const size_t j_end = (j + block_size < n) ? j + block_size : n;
            
            for (size_t ii = i; ii < i_end; ii++) {
                for (size_t jj = j; jj < j_end; jj++) {
                    result->components[jj * m + ii].real = tensor->components[ii * n + jj].real;
                    result->components[jj * m + ii].imag = -tensor->components[ii * n + jj].imag;
                }
            }
        }
    }

    return QGT_SUCCESS;
}

qgt_error_t geometric_tensor_copy(quantum_geometric_tensor_t* dst,
                                const quantum_geometric_tensor_t* src) {
    if (!dst || !src) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // Verify dimensions match
    if (dst->rank != src->rank) {
        return QGT_ERROR_DIMENSION_MISMATCH;
    }

    for (size_t i = 0; i < src->rank; i++) {
        if (dst->dimensions[i] != src->dimensions[i]) {
            return QGT_ERROR_DIMENSION_MISMATCH;
        }
    }

    // Copy components using SIMD
    #pragma omp parallel for simd schedule(guided)
    for (size_t i = 0; i < src->total_elements; i++) {
        dst->components[i] = src->components[i];
    }

    dst->type = src->type;
    dst->is_symmetric = src->is_symmetric;
    dst->hardware = src->hardware;

    return QGT_SUCCESS;
}
