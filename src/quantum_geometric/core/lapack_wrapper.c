#include "quantum_geometric/core/lapack_wrapper.h"
#include "quantum_geometric/core/lapack_internal.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Global state
static struct {
    lapack_status_t last_status;
    bool initialized;
    size_t max_workspace_size;
    void* workspace;
} lapack_state = {0};

// Initialize LAPACK wrapper
static bool initialize_lapack(void) {
    if (lapack_state.initialized) {
        return true;
    }
    
    // Allocate a reasonable default workspace
    lapack_state.max_workspace_size = 1024 * 1024;  // 1MB
    lapack_state.workspace = malloc(lapack_state.max_workspace_size);
    if (!lapack_state.workspace) {
        lapack_state.last_status = LAPACK_MEMORY_ERROR;
        return false;
    }
    
    lapack_state.initialized = true;
    lapack_state.last_status = LAPACK_SUCCESS;
    return true;
}

// Cleanup LAPACK wrapper
static void cleanup_lapack(void) {
    if (lapack_state.workspace) {
        free(lapack_state.workspace);
        lapack_state.workspace = NULL;
    }
    lapack_state.initialized = false;
}

// Ensure workspace is large enough
static bool ensure_workspace(size_t size) {
    if (size <= lapack_state.max_workspace_size) {
        return true;
    }
    
    void* new_workspace = realloc(lapack_state.workspace, size);
    if (!new_workspace) {
        lapack_state.last_status = LAPACK_MEMORY_ERROR;
        return false;
    }
    
    lapack_state.workspace = new_workspace;
    lapack_state.max_workspace_size = size;
    return true;
}


// Helper functions
static void convert_layout(ComplexFloat* dst,
                         const ComplexFloat* src,
                         size_t m, size_t n,
                         lapack_layout_t from_layout,
                         lapack_layout_t to_layout) {
    if (from_layout == to_layout) {
        memcpy(dst, src, m * n * sizeof(ComplexFloat));
        return;
    }
    
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            if (from_layout == LAPACK_ROW_MAJOR) {
                dst[j * m + i] = src[i * n + j];
            } else {
                dst[i * n + j] = src[j * m + i];
            }
        }
    }
}

// Implementation of LAPACK operations
// Register cleanup on program exit
static void __attribute__((constructor)) register_cleanup(void) {
    atexit(cleanup_lapack);
}

bool lapack_matrix_multiply(const ComplexFloat* a,
                          const ComplexFloat* b,
                          ComplexFloat* c,
                          size_t m, size_t k, size_t n,
                          bool transpose_a,
                          bool transpose_b,
                          lapack_layout_t layout) {
    if (!initialize_lapack() || !a || !b || !c || m == 0 || k == 0 || n == 0) {
        lapack_state.last_status = LAPACK_INVALID_ARGUMENT;
        return false;
    }
    
    // Convert to column-major if needed
    ComplexFloat* a_col = malloc(m * k * sizeof(ComplexFloat));
    ComplexFloat* b_col = malloc(k * n * sizeof(ComplexFloat));
    if (!a_col || !b_col) {
        free(a_col);
        free(b_col);
        lapack_state.last_status = LAPACK_MEMORY_ERROR;
        return false;
    }
    convert_layout(a_col, a, m, k, layout, LAPACK_COL_MAJOR);
    convert_layout(b_col, b, k, n, layout, LAPACK_COL_MAJOR);
    
    // Prepare LAPACK call
    char transa = transpose_a ? 'T' : 'N';
    char transb = transpose_b ? 'T' : 'N';
    int im = (int)m;
    int in = (int)n;
    int ik = (int)k;
    int lda = transpose_a ? ik : im;
    int ldb = transpose_b ? in : ik;
    int ldc = im;
    
    // Set alpha and beta
    ComplexFloat alpha = {1.0f, 0.0f};
    ComplexFloat beta = {0.0f, 0.0f};
    
    // Perform matrix multiplication
    cgemm_(&transa, &transb, &im, &in, &ik,
           &alpha, a_col, &lda, b_col, &ldb,
           &beta, c, &ldc);
    
    free(b_col);
    free(a_col);
    
    lapack_state.last_status = LAPACK_SUCCESS;
    return true;
}

// Helper function to check matrix properties
static bool is_hermitian(const ComplexFloat* a, size_t n) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < i; j++) {
            ComplexFloat aij = a[i * n + j];
            ComplexFloat aji = a[j * n + i];
            if (fabs(aij.real - aji.real) > 1e-6f || 
                fabs(aij.imag + aji.imag) > 1e-6f) {
                return false;
            }
        }
    }
    return true;
}

static bool is_positive_definite(const ComplexFloat* a, size_t n) {
    // Simple check: diagonal elements must be positive
    for (size_t i = 0; i < n; i++) {
        if (a[i * n + i].real <= 0 || fabs(a[i * n + i].imag) > 1e-6f) {
            return false;
        }
    }
    return true;
}

bool lapack_svd(const ComplexFloat* a, size_t m, size_t n,
                ComplexFloat* u, float* s, ComplexFloat* vt,
                lapack_layout_t layout) {
    if (!initialize_lapack() || !a || !u || !s || !vt || m == 0 || n == 0) {
        lapack_state.last_status = LAPACK_INVALID_ARGUMENT;
        return false;
    }
    
    // Convert to column-major if needed
    ComplexFloat* a_col = malloc(m * n * sizeof(ComplexFloat));
    if (!a_col) {
        lapack_state.last_status = LAPACK_MEMORY_ERROR;
        return false;
    }
    convert_layout(a_col, a, m, n, layout, LAPACK_COL_MAJOR);
    
    // Prepare LAPACK call
    char jobu = 'A', jobvt = 'A';
    int im = (int)m, in = (int)n;
    int lda = im, ldu = im, ldvt = in;
    int lwork = -1;
    int info;
    
    // Query optimal workspace
    ComplexFloat work_query;
    float* rwork = malloc(5 * (m < n ? m : n) * sizeof(float));
    if (!rwork) {
        free(a_col);
        lapack_state.last_status = LAPACK_MEMORY_ERROR;
        return false;
    }
    
    cgesvd_(&jobu, &jobvt, &im, &in, a_col, &lda, s,
            u, &ldu, vt, &ldvt, &work_query, &lwork,
            rwork, &info);
    
    // Allocate workspace
    lwork = (int)work_query.real;
    ComplexFloat* work = malloc(lwork * sizeof(ComplexFloat));
    if (!work) {
        free(rwork);
        free(a_col);
        lapack_state.last_status = LAPACK_MEMORY_ERROR;
        return false;
    }
    
    // Compute SVD
    cgesvd_(&jobu, &jobvt, &im, &in, a_col, &lda, s,
            u, &ldu, vt, &ldvt, work, &lwork,
            rwork, &info);
    
    free(work);
    free(rwork);
    free(a_col);
    
    if (info != 0) {
        lapack_state.last_status = info > 0 ? LAPACK_NOT_CONVERGENT : LAPACK_INVALID_ARGUMENT;
        return false;
    }
    
    lapack_state.last_status = LAPACK_SUCCESS;
    return true;
}

bool lapack_qr(const ComplexFloat* a, size_t m, size_t n,
               ComplexFloat* q, ComplexFloat* r,
               lapack_layout_t layout) {
    if (!initialize_lapack() || !a || !q || !r || m == 0 || n == 0) {
        lapack_state.last_status = LAPACK_INVALID_ARGUMENT;
        return false;
    }
    
    // Convert to column-major if needed
    ComplexFloat* a_col = malloc(m * n * sizeof(ComplexFloat));
    if (!a_col) {
        lapack_state.last_status = LAPACK_MEMORY_ERROR;
        return false;
    }
    convert_layout(a_col, a, m, n, layout, LAPACK_COL_MAJOR);
    
    // Prepare LAPACK call
    int im = (int)m, in = (int)n;
    int lda = im;
    int info;
    
    // Allocate tau
    ComplexFloat* tau = malloc(n * sizeof(ComplexFloat));
    if (!tau) {
        free(a_col);
        lapack_state.last_status = LAPACK_MEMORY_ERROR;
        return false;
    }
    
    // Query optimal workspace
    ComplexFloat work_query;
    int lwork = -1;
    
    cgeqrf_(&im, &in, a_col, &lda, tau, &work_query, &lwork, &info);
    
    // Allocate workspace
    lwork = (int)work_query.real;
    ComplexFloat* work = malloc(lwork * sizeof(ComplexFloat));
    if (!work) {
        free(tau);
        free(a_col);
        lapack_state.last_status = LAPACK_MEMORY_ERROR;
        return false;
    }
    
    // Compute QR factorization
    cgeqrf_(&im, &in, a_col, &lda, tau, work, &lwork, &info);
    
    if (info != 0) {
        free(work);
        free(tau);
        free(a_col);
        lapack_state.last_status = LAPACK_INVALID_ARGUMENT;
        return false;
    }
    
    // Copy R (upper triangular part of a_col)
    memset(r, 0, m * n * sizeof(ComplexFloat));
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j <= i && j < m; j++) {
            r[i * m + j] = a_col[i * m + j];
        }
    }
    
    // Generate Q
    memcpy(q, a_col, m * n * sizeof(ComplexFloat));
    cungqr_(&im, &in, &in, q, &lda, tau, work, &lwork, &info);
    
    free(work);
    free(tau);
    free(a_col);
    
    if (info != 0) {
        lapack_state.last_status = LAPACK_INVALID_ARGUMENT;
        return false;
    }
    
    lapack_state.last_status = LAPACK_SUCCESS;
    return true;
}

bool lapack_eigendecomposition(const ComplexFloat* a, size_t n,
                             ComplexFloat* eigenvectors,
                             ComplexFloat* eigenvalues,
                             lapack_layout_t layout) {
    if (!initialize_lapack() || !a || !eigenvectors || !eigenvalues || n == 0) {
        lapack_state.last_status = LAPACK_INVALID_ARGUMENT;
        return false;
    }
    
    // Check if matrix is Hermitian
    if (!is_hermitian(a, n)) {
        lapack_state.last_status = LAPACK_INVALID_ARGUMENT;
        return false;
    }
    
    // Convert to column-major if needed
    ComplexFloat* a_col = malloc(n * n * sizeof(ComplexFloat));
    if (!a_col) {
        lapack_state.last_status = LAPACK_MEMORY_ERROR;
        return false;
    }
    convert_layout(a_col, a, n, n, layout, LAPACK_COL_MAJOR);
    
    // Prepare LAPACK call
    char jobz = 'V', uplo = 'U';
    int in = (int)n;
    int lda = in;
    int info;
    
    // Allocate workspace
    float* w = malloc(n * sizeof(float));
    float* rwork = malloc(3 * n * sizeof(float));
    if (!w || !rwork) {
        free(w);
        free(rwork);
        free(a_col);
        lapack_state.last_status = LAPACK_MEMORY_ERROR;
        return false;
    }
    
    // Query optimal workspace
    ComplexFloat work_query;
    int lwork = -1;
    
    cheev_(&jobz, &uplo, &in, a_col, &lda, w,
           &work_query, &lwork, rwork, &info);
    
    // Allocate workspace
    lwork = (int)work_query.real;
    ComplexFloat* work = malloc(lwork * sizeof(ComplexFloat));
    if (!work) {
        free(w);
        free(rwork);
        free(a_col);
        lapack_state.last_status = LAPACK_MEMORY_ERROR;
        return false;
    }
    
    // Compute eigendecomposition
    cheev_(&jobz, &uplo, &in, a_col, &lda, w,
           work, &lwork, rwork, &info);
    
    // Copy results
    memcpy(eigenvectors, a_col, n * n * sizeof(ComplexFloat));
    for (size_t i = 0; i < n; i++) {
        eigenvalues[i].real = w[i];
        eigenvalues[i].imag = 0.0f;
    }
    
    free(work);
    free(rwork);
    free(w);
    free(a_col);
    
    if (info != 0) {
        lapack_state.last_status = info > 0 ? LAPACK_NOT_CONVERGENT : LAPACK_INVALID_ARGUMENT;
        return false;
    }
    
    lapack_state.last_status = LAPACK_SUCCESS;
    return true;
}

bool lapack_cholesky(const ComplexFloat* a, size_t n,
                    ComplexFloat* l, bool lower_triangular,
                    lapack_layout_t layout) {
    if (!initialize_lapack() || !a || !l || n == 0) {
        lapack_state.last_status = LAPACK_INVALID_ARGUMENT;
        return false;
    }
    
    // Check if matrix is Hermitian and positive definite
    if (!is_hermitian(a, n) || !is_positive_definite(a, n)) {
        lapack_state.last_status = LAPACK_INVALID_ARGUMENT;
        return false;
    }
    
    // Convert to column-major if needed
    ComplexFloat* a_col = malloc(n * n * sizeof(ComplexFloat));
    if (!a_col) {
        lapack_state.last_status = LAPACK_MEMORY_ERROR;
        return false;
    }
    convert_layout(a_col, a, n, n, layout, LAPACK_COL_MAJOR);
    
    // Prepare LAPACK call
    char uplo = lower_triangular ? 'L' : 'U';
    int in = (int)n;
    int lda = in;
    int info;
    
    // Compute Cholesky decomposition
    cpotrf_(&uplo, &in, a_col, &lda, &info);
    
    // Copy result
    memcpy(l, a_col, n * n * sizeof(ComplexFloat));
    
    free(a_col);
    
    if (info != 0) {
        lapack_state.last_status = info > 0 ? LAPACK_NOT_CONVERGENT : LAPACK_INVALID_ARGUMENT;
        return false;
    }
    
    lapack_state.last_status = LAPACK_SUCCESS;
    return true;
}

bool lapack_lu(const ComplexFloat* a, size_t m, size_t n,
               ComplexFloat* l, ComplexFloat* u, int* ipiv,
               lapack_layout_t layout) {
    if (!initialize_lapack() || !a || !l || !u || !ipiv || m == 0 || n == 0) {
        lapack_state.last_status = LAPACK_INVALID_ARGUMENT;
        return false;
    }
    
    // Convert to column-major if needed
    ComplexFloat* a_col = malloc(m * n * sizeof(ComplexFloat));
    if (!a_col) {
        lapack_state.last_status = LAPACK_MEMORY_ERROR;
        return false;
    }
    convert_layout(a_col, a, m, n, layout, LAPACK_COL_MAJOR);
    
    // Prepare LAPACK call
    int im = (int)m, in = (int)n;
    int lda = im;
    int info;
    
    // Compute LU decomposition
    cgetrf_(&im, &in, a_col, &lda, ipiv, &info);
    
    // Extract L and U
    memset(l, 0, m * n * sizeof(ComplexFloat));
    memset(u, 0, m * n * sizeof(ComplexFloat));
    
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            if (i > j) {
                l[i * n + j] = a_col[i * n + j];
            } else if (i == j) {
                l[i * n + j].real = 1.0f;
                l[i * n + j].imag = 0.0f;
                u[i * n + j] = a_col[i * n + j];
            } else {
                u[i * n + j] = a_col[i * n + j];
            }
        }
    }
    
    free(a_col);
    
    if (info != 0) {
        lapack_state.last_status = info > 0 ? LAPACK_SINGULAR_MATRIX : LAPACK_INVALID_ARGUMENT;
        return false;
    }
    
    lapack_state.last_status = LAPACK_SUCCESS;
    return true;
}

bool lapack_solve_triangular(const ComplexFloat* a,
                           const ComplexFloat* b,
                           ComplexFloat* x,
                           size_t n, size_t nrhs,
                           bool upper_triangular,
                           bool unit_diagonal,
                           lapack_layout_t layout) {
    if (!initialize_lapack() || !a || !b || !x || n == 0 || nrhs == 0) {
        lapack_state.last_status = LAPACK_INVALID_ARGUMENT;
        return false;
    }
    
    // Convert to column-major if needed
    ComplexFloat* a_col = malloc(n * n * sizeof(ComplexFloat));
    ComplexFloat* b_col = malloc(n * nrhs * sizeof(ComplexFloat));
    if (!a_col || !b_col) {
        free(a_col);
        free(b_col);
        lapack_state.last_status = LAPACK_MEMORY_ERROR;
        return false;
    }
    convert_layout(a_col, a, n, n, layout, LAPACK_COL_MAJOR);
    convert_layout(b_col, b, n, nrhs, layout, LAPACK_COL_MAJOR);
    
    // Prepare LAPACK call
    char uplo = upper_triangular ? 'U' : 'L';
    char trans = 'N';
    char diag = unit_diagonal ? 'U' : 'N';
    int in = (int)n;
    int inrhs = (int)nrhs;
    int lda = in;
    int ldb = in;
    int info;
    
    // Solve system
    ctrtrs_(&uplo, &trans, &diag, &in, &inrhs,
            a_col, &lda, b_col, &ldb, &info);
    
    // Copy solution
    convert_layout(x, b_col, n, nrhs, LAPACK_COL_MAJOR, layout);
    
    free(b_col);
    free(a_col);
    
    if (info != 0) {
        lapack_state.last_status = info > 0 ? LAPACK_SINGULAR_MATRIX : LAPACK_INVALID_ARGUMENT;
        return false;
    }
    
    lapack_state.last_status = LAPACK_SUCCESS;
    return true;
}

bool lapack_solve_symmetric(const ComplexFloat* a,
                          const ComplexFloat* b,
                          ComplexFloat* x,
                          size_t n, size_t nrhs,
                          bool positive_definite,
                          lapack_layout_t layout) {
    if (!initialize_lapack() || !a || !b || !x || n == 0 || nrhs == 0) {
        lapack_state.last_status = LAPACK_INVALID_ARGUMENT;
        return false;
    }
    
    // Check if matrix is Hermitian
    if (!is_hermitian(a, n)) {
        lapack_state.last_status = LAPACK_INVALID_ARGUMENT;
        return false;
    }
    
    // For positive definite solver, check if matrix is positive definite
    if (positive_definite && !is_positive_definite(a, n)) {
        lapack_state.last_status = LAPACK_INVALID_ARGUMENT;
        return false;
    }
    
    // Convert to column-major if needed
    ComplexFloat* a_col = malloc(n * n * sizeof(ComplexFloat));
    ComplexFloat* b_col = malloc(n * nrhs * sizeof(ComplexFloat));
    if (!a_col || !b_col) {
        free(a_col);
        free(b_col);
        lapack_state.last_status = LAPACK_MEMORY_ERROR;
        return false;
    }
    convert_layout(a_col, a, n, n, layout, LAPACK_COL_MAJOR);
    convert_layout(b_col, b, n, nrhs, layout, LAPACK_COL_MAJOR);
    
    // Prepare LAPACK call
    char uplo = 'U';
    int in = (int)n;
    int inrhs = (int)nrhs;
    int lda = in;
    int ldb = in;
    int info;
    
    // Solve system
    if (positive_definite) {
        cposv_(&uplo, &in, &inrhs, a_col, &lda,
               b_col, &ldb, &info);
    } else {
        int* ipiv = malloc(n * sizeof(int));
        if (!ipiv) {
            free(b_col);
            free(a_col);
            lapack_state.last_status = LAPACK_MEMORY_ERROR;
            return false;
        }
        cgesv_(&in, &inrhs, a_col, &lda, ipiv,
               b_col, &ldb, &info);
        free(ipiv);
    }
    
    // Copy solution
    convert_layout(x, b_col, n, nrhs, LAPACK_COL_MAJOR, layout);
    
    free(b_col);
    free(a_col);
    
    if (info != 0) {
        lapack_state.last_status = info > 0 ? LAPACK_SINGULAR_MATRIX : LAPACK_INVALID_ARGUMENT;
        return false;
    }
    
    lapack_state.last_status = LAPACK_SUCCESS;
    return true;
}

bool lapack_solve_general(const ComplexFloat* a,
                         const ComplexFloat* b,
                         ComplexFloat* x,
                         size_t n, size_t nrhs,
                         lapack_layout_t layout) {
    if (!initialize_lapack() || !a || !b || !x || n == 0 || nrhs == 0) {
        lapack_state.last_status = LAPACK_INVALID_ARGUMENT;
        return false;
    }
    
    // Convert to column-major if needed
    ComplexFloat* a_col = malloc(n * n * sizeof(ComplexFloat));
    ComplexFloat* b_col = malloc(n * nrhs * sizeof(ComplexFloat));
    if (!a_col || !b_col) {
        free(a_col);
        free(b_col);
        lapack_state.last_status = LAPACK_MEMORY_ERROR;
        return false;
    }
    convert_layout(a_col, a, n, n, layout, LAPACK_COL_MAJOR);
    convert_layout(b_col, b, n, nrhs, layout, LAPACK_COL_MAJOR);
    
    // Prepare LAPACK call
    int in = (int)n;
    int inrhs = (int)nrhs;
    int lda = in;
    int ldb = in;
    int info;
    
    // Allocate pivot array
    int* ipiv = malloc(n * sizeof(int));
    if (!ipiv) {
        free(b_col);
        free(a_col);
        lapack_state.last_status = LAPACK_MEMORY_ERROR;
        return false;
    }
    
    // Solve system
    cgesv_(&in, &inrhs, a_col, &lda, ipiv,
           b_col, &ldb, &info);
    
    // Copy solution
    convert_layout(x, b_col, n, nrhs, LAPACK_COL_MAJOR, layout);
    
    free(ipiv);
    free(b_col);
    free(a_col);
    
    if (info != 0) {
        lapack_state.last_status = info > 0 ? LAPACK_SINGULAR_MATRIX : LAPACK_INVALID_ARGUMENT;
        return false;
    }
    
    lapack_state.last_status = LAPACK_SUCCESS;
    return true;
}

size_t lapack_get_optimal_workspace(const ComplexFloat* a,
                                  size_t m, size_t n,
                                  const char* operation) {
    if (!initialize_lapack() || !operation) {
        lapack_state.last_status = LAPACK_INVALID_ARGUMENT;
        return 0;
    }
    
    ComplexFloat work_query;
    int lwork = -1;
    int info;
    int im = (int)m;
    int in = (int)n;
    
    if (strcmp(operation, "SVD") == 0) {
        char jobu = 'A', jobvt = 'A';
        int lda = im, ldu = im, ldvt = in;
        float* s = malloc(m < n ? m : n * sizeof(float));
        float* rwork = malloc(5 * (m < n ? m : n) * sizeof(float));
        
        if (!s || !rwork) {
            free(s);
            free(rwork);
            lapack_state.last_status = LAPACK_MEMORY_ERROR;
            return 0;
        }
        
        cgesvd_(&jobu, &jobvt, &im, &in, (ComplexFloat*)a, &lda, s,
                NULL, &ldu, NULL, &ldvt, &work_query, &lwork,
                rwork, &info);
        
        free(rwork);
        free(s);
        
        if (info != 0) {
            lapack_state.last_status = LAPACK_INVALID_ARGUMENT;
            return 0;
        }
        
        return (size_t)work_query.real;
    }
    
    // Add other operations as needed
    
    lapack_state.last_status = LAPACK_NOT_IMPLEMENTED;
    return 0;
}

bool lapack_has_capability(const char* capability) {
    if (!initialize_lapack() || !capability) {
        lapack_state.last_status = LAPACK_INVALID_ARGUMENT;
        return false;
    }
    
    // List of supported capabilities
    const char* capabilities[] = {
        "SVD",
        "QR",
        "EIGEN",
        "CHOLESKY",
        "LU",
        "TRIANGULAR_SOLVE",
        "SYMMETRIC_SOLVE",
        "GENERAL_SOLVE",
        "MATRIX_MULTIPLY",
        "WORKSPACE_MANAGEMENT",
        "ERROR_HANDLING",
        "LAYOUT_CONVERSION",
        "HERMITIAN_CHECK",
        "POSITIVE_DEFINITE_CHECK",
        NULL
    };
    
    for (const char** cap = capabilities; *cap; cap++) {
        if (strcmp(capability, *cap) == 0) {
            lapack_state.last_status = LAPACK_SUCCESS;
            return true;
        }
    }
    
    lapack_state.last_status = LAPACK_SUCCESS;
    return false;
}

lapack_status_t lapack_get_last_status(void) {
    return lapack_state.last_status;
}

const char* lapack_get_status_string(lapack_status_t status) {
    switch (status) {
        case LAPACK_SUCCESS:
            return "Success";
        case LAPACK_INVALID_ARGUMENT:
            return "Invalid argument";
        case LAPACK_MEMORY_ERROR:
            return "Memory allocation failed";
        case LAPACK_SINGULAR_MATRIX:
            return "Matrix is singular";
        case LAPACK_NOT_CONVERGENT:
            return "Algorithm did not converge";
        case LAPACK_INTERNAL_ERROR:
            return "Internal error";
        case LAPACK_NOT_IMPLEMENTED:
            return "Operation not implemented";
        default:
            return "Unknown error";
    }
}
