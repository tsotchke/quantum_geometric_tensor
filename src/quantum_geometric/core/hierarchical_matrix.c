#include "quantum_geometric/core/hierarchical_matrix.h"
#include "quantum_geometric/core/simd_operations.h"
#include "quantum_geometric/core/numerical_backend.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Architecture-specific includes
#if defined(__x86_64__) || defined(_M_X64)
  #include <immintrin.h>
  #define USE_X86_SIMD
#elif defined(__arm__) || defined(__aarch64__) || defined(_M_ARM)
  // ARM-specific includes
  #include <arm_neon.h>
  #define USE_ARM_SIMD
#endif

// Utility macros
#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif

// Original optimization constants
#define MIN_MATRIX_SIZE 64
#define MAX_RANK 32
#define SVD_TOLERANCE 1e-12
#define CACHE_LINE_SIZE 64

// Additional optimization constants for O(log n) paths
#define OPT_MIN_MATRIX_SIZE 32  // Reduced to allow more hierarchical levels
#define OPT_MAX_RANK 16        // Reduced toi favor low-rank approximations
#define OPT_SVD_TOLERANCE 1e-8  // Increased tolerance for more aggressive compression
#define COMPRESSION_THRESHOLD 0.1  // Threshold for when to attempt compression
#define ADAPTIVE_BLOCK_SIZE 1024   // Size threshold for adaptive blocking

// Runtime optimization selection - always enabled for performance
static bool use_optimized_path = true;

// Forward declarations for internal functions
void hmatrix_destroy(HierarchicalMatrix* matrix);
bool hmatrix_is_low_rank(const HierarchicalMatrix* matrix);
void hmatrix_truncate(HierarchicalMatrix* matrix);
void hmatrix_add(HierarchicalMatrix* dst, const HierarchicalMatrix* a, const HierarchicalMatrix* b);
double hmatrix_error_estimate(const HierarchicalMatrix* matrix);
static HierarchicalMatrix* hmatrix_create_internal(size_t rows, size_t cols, double tolerance);

// Initialize numerical backend
static bool init_backend(void) {
    static bool initialized = false;
    if (!initialized) {
        numerical_config_t config = {
            .type = NUMERICAL_BACKEND_ACCELERATE,
            .max_threads = 8,
            .use_fma = true,
            .use_avx = true
        };
        initialized = initialize_numerical_backend(&config);
    }
    return initialized;
}

// Helper function to determine optimal execution path
static bool should_use_optimized_path(size_t rows, size_t cols) {
    // Always use optimized path since we've validated it works well
    return (rows * cols > ADAPTIVE_BLOCK_SIZE);
}

// Enhanced error handling macro with cleanup
#define CHECK_ALLOC(ptr, cleanup) if (!ptr) { \
    printf("Memory allocation failed at %s:%d\n", __FILE__, __LINE__); \
    cleanup; \
    return NULL; \
}

// SVD and low-rank approximation
void compute_svd(double complex* data, size_t rows, size_t cols,
                 double complex* U, double complex* S, double complex* V) {
    // Initialize numerical backend
    if (!init_backend()) {
        printf("Failed to initialize numerical backend\n");
        return;
    }
    // Convert double complex to ComplexFloat
    ComplexFloat* data_f = malloc(rows * cols * sizeof(ComplexFloat));
    ComplexFloat* U_f = malloc(rows * rows * sizeof(ComplexFloat));
    float* S_f = malloc(min(rows, cols) * sizeof(float));
    ComplexFloat* VT_f = malloc(cols * cols * sizeof(ComplexFloat));
    
    if (!data_f || !U_f || !S_f || !VT_f) {
        free(data_f);
        free(U_f);
        free(S_f);
        free(VT_f);
        printf("SVD memory allocation failed\n");
        return;
    }
    
    // Convert input data
    for (size_t i = 0; i < rows * cols; i++) {
        data_f[i].real = creal(data[i]);
        data_f[i].imag = cimag(data[i]);
    }
    
    // Compute SVD using numerical backend
    if (!numerical_svd(data_f, U_f, S_f, VT_f, rows, cols)) {
        printf("SVD computation failed\n");
        free(data_f);
        free(U_f);
        free(S_f);
        free(VT_f);
        return;
    }
    
    // Convert results back
    for (size_t i = 0; i < rows * rows; i++) {
        U[i] = U_f[i].real + I * U_f[i].imag;
    }
    
    for (size_t i = 0; i < min(rows, cols); i++) {
        S[i] = S_f[i];
    }
    
    // VT_f is in row-major order, need to transpose while converting
    for (size_t i = 0; i < cols; i++) {
        for (size_t j = 0; j < cols; j++) {
            V[i * cols + j] = VT_f[j * cols + i].real + I * VT_f[j * cols + i].imag;
        }
    }
    
    free(data_f);
    free(U_f);
    free(S_f);
    free(VT_f);
}

// Original linear search implementation
static void truncate_svd_linear(double complex* U, double complex* S, double complex* V,
                        size_t rows, size_t cols, size_t* rank, double tolerance) {
    double total = 0.0;
    #pragma omp parallel for reduction(+:total)
    for (size_t i = 0; i < rows && i < cols; i++) {
        total += cabs(S[i]);
    }
    
    double running_sum = 0.0;
    size_t k;
    for (k = 0; k < rows && k < cols; k++) {
        running_sum += cabs(S[k]);
        if (running_sum / total >= 1.0 - tolerance) break;
    }
    
    *rank = k + 1;
    if (*rank > MAX_RANK) *rank = MAX_RANK;
    
    // Truncate matrices
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            for (size_t i = *rank; i < rows && i < cols; i++) {
                S[i] = 0.0;
            }
        }
        
        #pragma omp section
        {
            for (size_t i = *rank; i < rows && i < cols; i++) {
                for (size_t j = 0; j < rows; j++) {
                    U[j * cols + i] = 0.0;
                }
            }
        }
        
        #pragma omp section
        {
            for (size_t i = *rank; i < rows && i < cols; i++) {
                for (size_t j = 0; j < cols; j++) {
                    V[i * cols + j] = 0.0;
                }
            }
        }
    }
}

// O(log n) optimized implementation using binary search
static void truncate_svd_optimized(double complex* U, double complex* S, double complex* V,
                        size_t rows, size_t cols, size_t* rank, double tolerance) {
    double total = 0.0;
    #pragma omp parallel for reduction(+:total)
    for (size_t i = 0; i < rows && i < cols; i++) {
        total += cabs(S[i]);
    }
    
    size_t left = 0;
    size_t right = min(rows, cols);
    size_t k = 0;
    
    while (left < right) {
        size_t mid = (left + right) / 2;
        double running_sum = 0.0;
        
        #pragma omp parallel for reduction(+:running_sum)
        for (size_t i = 0; i <= mid; i++) {
            running_sum += cabs(S[i]);
        }
        
        if (running_sum / total >= 1.0 - tolerance) {
            k = mid;
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    
    *rank = k + 1;
    if (*rank > MAX_RANK) *rank = MAX_RANK;
    
        // Architecture-specific SIMD optimizations
        #ifdef USE_X86_SIMD
            // Use SIMD for truncation
            size_t vec_size = 8; // AVX-512 processes 8 doubles at once
            size_t vec_count = (rows * (*rank)) / vec_size;
            
            #pragma omp parallel sections
            {
                #pragma omp section
                {
                    for (size_t i = *rank; i < rows && i < cols; i++) {
                        S[i] = 0.0;
                    }
                }
                
                #pragma omp section
                {
                    #pragma omp parallel for
                    for (size_t i = *rank; i < rows && i < cols; i++) {
                        size_t row_offset = i * cols;
                        
                        // Vector operations for bulk of the data
                        for (size_t j = 0; j < vec_count * vec_size; j += vec_size) {
                            _mm512_store_pd((double*)&U[row_offset + j], _mm512_setzero_pd());
                        }
                        
                        // Handle remaining elements
                        for (size_t j = vec_count * vec_size; j < rows; j++) {
                            U[row_offset + j] = 0.0;
                        }
                    }
                }
                
                #pragma omp section
                {
                    #pragma omp parallel for
                    for (size_t i = *rank; i < rows && i < cols; i++) {
                        size_t row_offset = i * cols;
                        
                        for (size_t j = 0; j < vec_count * vec_size; j += vec_size) {
                            _mm512_store_pd((double*)&V[row_offset + j], _mm512_setzero_pd());
                        }
                        
                        for (size_t j = vec_count * vec_size; j < cols; j++) {
                            V[row_offset + j] = 0.0;
                        }
                    }
                }
            }
        #else
            // Generic implementation for other architectures
            #pragma omp parallel sections
            {
                #pragma omp section
                {
                    for (size_t i = *rank; i < rows && i < cols; i++) {
                        S[i] = 0.0;
                    }
                }
                
                #pragma omp section
                {
                    #pragma omp parallel for
                    for (size_t i = *rank; i < rows && i < cols; i++) {
                        for (size_t j = 0; j < rows; j++) {
                            U[j * cols + i] = 0.0;
                        }
                    }
                }
                
                #pragma omp section
                {
                    #pragma omp parallel for
                    for (size_t i = *rank; i < rows && i < cols; i++) {
                        for (size_t j = 0; j < cols; j++) {
                            V[i * cols + j] = 0.0;
                        }
                    }
                }
            }
        #endif
}

// Function to choose between implementations
void truncate_svd(double complex* U, double complex* S, double complex* V,
                        size_t rows, size_t cols, size_t* rank, double tolerance) {
    if (should_use_optimized_path(rows, cols)) {
        truncate_svd_optimized(U, S, V, rows, cols, rank, tolerance);
    } else {
        truncate_svd_linear(U, S, V, rows, cols, rank, tolerance);
    }
}

// Original implementation
static HierarchicalMatrix* hmatrix_create_standard(size_t n, double tolerance) {
    if (n == 0) {
        printf("Invalid matrix dimension: %zu\n", n);
        return NULL;
    }
    
    HierarchicalMatrix* mat = aligned_alloc(CACHE_LINE_SIZE, sizeof(HierarchicalMatrix));
    CHECK_ALLOC(mat, );
    
    mat->rows = n;
    mat->cols = n;
    mat->tolerance = tolerance;
    mat->rank = 0;
    
    // Standard leaf size criteria
    mat->is_leaf = (n <= MIN_MATRIX_SIZE);
    
    if (mat->is_leaf) {
        mat->data = aligned_alloc(CACHE_LINE_SIZE, n * n * sizeof(double complex));
        if (!mat->data) {
            free(mat);
            printf("Failed to allocate matrix data\n");
            return NULL;
        }
        mat->U = NULL;
        mat->V = NULL;
        memset(mat->data, 0, n * n * sizeof(double complex));
    } else {
        mat->data = NULL;
        mat->U = NULL;
        mat->V = NULL;
        
        // Standard even split
        size_t mid = n / 2;
        
        for (int i = 0; i < 4; i++) {
            size_t sub_n = (i < 2) ? mid : (n - mid);
            mat->children[i] = create_hierarchical_matrix(sub_n, mat->tolerance);
            if (!mat->children[i]) {
                for (int j = 0; j < i; j++) {
                    hmatrix_destroy(mat->children[j]);
                }
                free(mat);
                return NULL;
            }
        }
    }
    
    return mat;
}

// Internal implementation supporting rectangular matrices
static HierarchicalMatrix* hmatrix_create_internal(size_t rows, size_t cols, double tolerance) {
    if (rows == 0 || cols == 0) {
        printf("Invalid matrix dimensions: %zu x %zu\n", rows, cols);
        return NULL;
    }

    HierarchicalMatrix* mat = aligned_alloc(CACHE_LINE_SIZE, sizeof(HierarchicalMatrix));
    CHECK_ALLOC(mat, );

    mat->rows = rows;
    mat->cols = cols;
    mat->tolerance = tolerance;
    mat->rank = 0;

    // Leaf size criteria - either dimension is small enough
    mat->is_leaf = (rows <= MIN_MATRIX_SIZE || cols <= MIN_MATRIX_SIZE);

    if (mat->is_leaf) {
        mat->data = aligned_alloc(CACHE_LINE_SIZE, rows * cols * sizeof(double complex));
        if (!mat->data) {
            free(mat);
            printf("Failed to allocate matrix data\n");
            return NULL;
        }
        mat->U = NULL;
        mat->V = NULL;
        memset(mat->data, 0, rows * cols * sizeof(double complex));
        for (int i = 0; i < 4; i++) {
            mat->children[i] = NULL;
        }
    } else {
        mat->data = NULL;
        mat->U = NULL;
        mat->V = NULL;

        // Split into quadrants
        size_t mid_row = rows / 2;
        size_t mid_col = cols / 2;

        for (int i = 0; i < 4; i++) {
            size_t sub_rows = (i < 2) ? mid_row : (rows - mid_row);
            size_t sub_cols = (i % 2 == 0) ? mid_col : (cols - mid_col);
            mat->children[i] = hmatrix_create_internal(sub_rows, sub_cols, tolerance);
            if (!mat->children[i]) {
                for (int j = 0; j < i; j++) {
                    hmatrix_destroy(mat->children[j]);
                }
                free(mat);
                return NULL;
            }
        }
    }

    return mat;
}

// Wrapper for general matrix creation
HierarchicalMatrix* hmatrix_create(size_t rows, size_t cols, double tolerance) {
    return hmatrix_create_internal(rows, cols, tolerance);
}

// O(log n) optimized implementation
static HierarchicalMatrix* hmatrix_create_optimized(size_t n, double tolerance) {
    if (n == 0) {
        printf("Invalid matrix dimension: %zu\n", n);
        return NULL;
    }
    
    HierarchicalMatrix* mat = aligned_alloc(CACHE_LINE_SIZE, sizeof(HierarchicalMatrix));
    CHECK_ALLOC(mat, );
    
    mat->rows = n;
    mat->cols = n;
    mat->tolerance = tolerance;
    mat->rank = 0;
    
    // Adaptive leaf size based on matrix dimensions
    mat->is_leaf = (n <= OPT_MIN_MATRIX_SIZE || 
                   (n <= OPT_MIN_MATRIX_SIZE * 2 && n <= 128)); // Avoid over-subdivision
    
    if (mat->is_leaf) {
        mat->data = aligned_alloc(CACHE_LINE_SIZE, n * n * sizeof(double complex));
        if (!mat->data) {
            free(mat);
            printf("Failed to allocate matrix data\n");
            return NULL;
        }
        mat->U = NULL;
        mat->V = NULL;
        memset(mat->data, 0, n * n * sizeof(double complex));
    } else {
        mat->data = NULL;
        mat->U = NULL;
        mat->V = NULL;
        
        // Use optimal splitting ratio
        size_t mid = n / 2;
        
        for (int i = 0; i < 4; i++) {
            size_t sub_n = (i < 2) ? mid : (n - mid);
            mat->children[i] = create_hierarchical_matrix(sub_n, mat->tolerance);
            if (!mat->children[i]) {
                for (int j = 0; j < i; j++) {
                    hmatrix_destroy(mat->children[j]);
                }
                free(mat);
                return NULL;
            }
        }
    }
    
    return mat;
}

// Validate hierarchical matrix structure
bool validate_hierarchical_matrix(const HierarchicalMatrix* matrix) {
    if (!matrix) return false;
    
    // Check dimensions
    if (matrix->rows == 0 || matrix->cols == 0) return false;
    
    // Check data consistency
    if (matrix->is_leaf) {
        // Leaf nodes must have data
        if (!matrix->data) return false;
        // Leaf nodes shouldn't have children
        for (int i = 0; i < 4; i++) {
            if (matrix->children[i]) return false;
        }
    } else {
        // Non-leaf nodes must have valid children
        for (int i = 0; i < 4; i++) {
            if (!matrix->children[i]) return false;
        }
    }
    
    // Check tolerance
    if (matrix->tolerance <= 0.0) return false;
    
    // All checks passed
    return true;
}

// Function to choose between implementations
HierarchicalMatrix* create_hierarchical_matrix(size_t n, double tolerance) {
    if (should_use_optimized_path(n, n)) {
        return hmatrix_create_optimized(n, tolerance);
    } else {
        return hmatrix_create_standard(n, tolerance);
    }
}

// Initialize matrix properties
bool init_matrix_properties(HierarchicalMatrix* matrix, const matrix_properties_t* props) {
    if (!matrix || !props) {
        return false;
    }
    
    // Set basic properties
    matrix->n = props->dimension;
    matrix->rows = props->dimension;
    matrix->cols = props->dimension;
    matrix->tolerance = props->tolerance;
    
    // Initialize data if not already allocated
    if (!matrix->data && matrix->is_leaf) {
        matrix->data = aligned_alloc(CACHE_LINE_SIZE, matrix->n * matrix->n * sizeof(double complex));
        if (!matrix->data) {
            return false;
        }
        memset(matrix->data, 0, matrix->n * matrix->n * sizeof(double complex));
    }
    
    // Set matrix type based on properties
    if (props->symmetric) {
        matrix->type = MATRIX_HIERARCHICAL;
    } else {
        matrix->type = MATRIX_DENSE;
    }
    
    // Set storage format
    matrix->format = STORAGE_FULL;
    
    return true;
}

// Compress matrix using specified parameters
bool compress_matrix(HierarchicalMatrix* matrix, const compression_params_t* params) {
    if (!matrix || !params) {
        return false;
    }
    
    // If matrix is not a leaf node, recursively compress children
    if (!matrix->is_leaf) {
        bool success = true;
        for (int i = 0; i < 4; i++) {
            if (matrix->children[i]) {
                success &= compress_matrix(matrix->children[i], params);
            }
        }
        return success;
    }
    
    // For leaf nodes, perform compression based on mode
    switch (params->mode) {
        case COMPRESS_SVD: {
            // Allocate memory for SVD
            size_t max_dim = (matrix->rows > matrix->cols) ? matrix->rows : matrix->cols;
            double complex* U = aligned_alloc(CACHE_LINE_SIZE, matrix->rows * max_dim * sizeof(double complex));
            double complex* S = aligned_alloc(CACHE_LINE_SIZE, max_dim * sizeof(double complex));
            double complex* V = aligned_alloc(CACHE_LINE_SIZE, matrix->cols * max_dim * sizeof(double complex));
            
            if (!U || !S || !V) {
                free(U);
                free(S);
                free(V);
                return false;
            }
            
            // Compute SVD
            compute_svd(matrix->data, matrix->rows, matrix->cols, U, S, V);
            
            // Truncate singular values
            size_t rank = 0;
            truncate_svd(U, S, V, matrix->rows, matrix->cols, &rank, params->tolerance);
            
            // Store rank-truncated matrices
            matrix->rank = rank;
            
            // Allocate memory for low-rank representation
            matrix->U = aligned_alloc(CACHE_LINE_SIZE, matrix->rows * rank * sizeof(double complex));
            matrix->V = aligned_alloc(CACHE_LINE_SIZE, matrix->cols * rank * sizeof(double complex));
            
            if (!matrix->U || !matrix->V) {
                free(U);
                free(S);
                free(V);
                free(matrix->U);
                free(matrix->V);
                matrix->U = NULL;
                matrix->V = NULL;
                return false;
            }
            
            // Copy truncated matrices
            for (size_t i = 0; i < matrix->rows; i++) {
                for (size_t j = 0; j < rank; j++) {
                    matrix->U[i * rank + j] = U[i * max_dim + j] * sqrt(cabs(S[j]));
                }
            }
            
            for (size_t i = 0; i < matrix->cols; i++) {
                for (size_t j = 0; j < rank; j++) {
                    matrix->V[i * rank + j] = V[i * max_dim + j] * sqrt(cabs(S[j]));
                }
            }
            
            // Free original data and temporary arrays
            free(matrix->data);
            matrix->data = NULL;
            free(U);
            free(S);
            free(V);
            
            // If recompression is enabled and rank is still high, apply additional compression
            if (params->recompression && rank > params->max_rank) {
                compression_params_t new_params = *params;
                new_params.tolerance *= 2.0;  // Increase tolerance for more aggressive compression
                return compress_matrix(matrix, &new_params);
            }
            
            return true;
        }
        
        case COMPRESS_QR: {
            // QR decomposition-based compression using Householder reflections
            // Computes A ≈ Q * R where Q is orthogonal and R is upper triangular
            // Then truncate based on R's diagonal values

            size_t m = matrix->rows;
            size_t n = matrix->cols;
            size_t min_dim = (m < n) ? m : n;

            // Allocate workspace
            double complex* Q = aligned_alloc(CACHE_LINE_SIZE, m * m * sizeof(double complex));
            double complex* R = aligned_alloc(CACHE_LINE_SIZE, m * n * sizeof(double complex));
            double complex* work = aligned_alloc(CACHE_LINE_SIZE, n * sizeof(double complex));

            if (!Q || !R || !work) {
                free(Q);
                free(R);
                free(work);
                return false;
            }

            // Copy data to R
            memcpy(R, matrix->data, m * n * sizeof(double complex));

            // Initialize Q as identity
            memset(Q, 0, m * m * sizeof(double complex));
            for (size_t i = 0; i < m; i++) {
                Q[i * m + i] = 1.0;
            }

            // Householder QR decomposition
            for (size_t k = 0; k < min_dim; k++) {
                // Compute Householder vector
                double norm_sq = 0.0;
                for (size_t i = k; i < m; i++) {
                    norm_sq += creal(R[i * n + k] * conj(R[i * n + k]));
                }
                double norm = sqrt(norm_sq);

                if (norm < 1e-14) continue;

                double complex alpha = R[k * n + k];
                double complex sign = (creal(alpha) >= 0) ? 1.0 : -1.0;
                double complex u_k = alpha + sign * norm;

                // Normalize Householder vector
                work[k] = 1.0;
                for (size_t i = k + 1; i < m; i++) {
                    work[i] = R[i * n + k] / u_k;
                }

                double tau = 2.0 / (1.0 + creal(conj(u_k) * u_k) / (norm * norm));

                // Apply Householder to R
                for (size_t j = k; j < n; j++) {
                    double complex dot = 0.0;
                    for (size_t i = k; i < m; i++) {
                        dot += conj(work[i]) * R[i * n + j];
                    }
                    dot *= tau;
                    for (size_t i = k; i < m; i++) {
                        R[i * n + j] -= work[i] * dot;
                    }
                }

                // Apply Householder to Q
                for (size_t j = 0; j < m; j++) {
                    double complex dot = 0.0;
                    for (size_t i = k; i < m; i++) {
                        dot += conj(work[i]) * Q[i * m + j];
                    }
                    dot *= tau;
                    for (size_t i = k; i < m; i++) {
                        Q[i * m + j] -= work[i] * dot;
                    }
                }
            }

            // Determine rank based on R diagonal
            double max_diag = 0.0;
            for (size_t i = 0; i < min_dim; i++) {
                double val = cabs(R[i * n + i]);
                if (val > max_diag) max_diag = val;
            }

            size_t rank = 0;
            for (size_t i = 0; i < min_dim; i++) {
                if (cabs(R[i * n + i]) > params->tolerance * max_diag) {
                    rank = i + 1;
                }
            }

            if (rank == 0) rank = 1;
            if (rank > params->max_rank) rank = params->max_rank;

            // Store low-rank approximation
            matrix->rank = rank;
            matrix->U = aligned_alloc(CACHE_LINE_SIZE, m * rank * sizeof(double complex));
            matrix->V = aligned_alloc(CACHE_LINE_SIZE, n * rank * sizeof(double complex));

            if (!matrix->U || !matrix->V) {
                free(Q);
                free(R);
                free(work);
                free(matrix->U);
                free(matrix->V);
                matrix->U = NULL;
                matrix->V = NULL;
                return false;
            }

            // U = Q[:, :rank]
            for (size_t i = 0; i < m; i++) {
                for (size_t j = 0; j < rank; j++) {
                    matrix->U[i * rank + j] = Q[i * m + j];
                }
            }

            // V = R[:rank, :]^T (transpose of top rows of R)
            for (size_t i = 0; i < n; i++) {
                for (size_t j = 0; j < rank; j++) {
                    matrix->V[i * rank + j] = R[j * n + i];
                }
            }

            free(Q);
            free(R);
            free(work);
            free(matrix->data);
            matrix->data = NULL;

            return true;
        }

        case COMPRESS_ACA: {
            // Adaptive Cross Approximation (ACA) algorithm
            // Builds low-rank approximation A ≈ U * V^T iteratively
            // by selecting pivots adaptively based on residual

            size_t m = matrix->rows;
            size_t n = matrix->cols;
            size_t max_rank = (params->max_rank < m && params->max_rank < n) ?
                             params->max_rank : ((m < n) ? m : n);

            // Allocate working matrices
            double complex* U_work = aligned_alloc(CACHE_LINE_SIZE, m * max_rank * sizeof(double complex));
            double complex* V_work = aligned_alloc(CACHE_LINE_SIZE, n * max_rank * sizeof(double complex));
            double complex* residual = aligned_alloc(CACHE_LINE_SIZE, m * n * sizeof(double complex));
            bool* row_used = calloc(m, sizeof(bool));
            bool* col_used = calloc(n, sizeof(bool));

            if (!U_work || !V_work || !residual || !row_used || !col_used) {
                free(U_work);
                free(V_work);
                free(residual);
                free(row_used);
                free(col_used);
                return false;
            }

            // Copy original matrix to residual
            memcpy(residual, matrix->data, m * n * sizeof(double complex));

            // Compute initial Frobenius norm
            double init_norm = 0.0;
            for (size_t i = 0; i < m * n; i++) {
                init_norm += creal(residual[i] * conj(residual[i]));
            }
            init_norm = sqrt(init_norm);

            size_t rank = 0;
            double rel_error = 1.0;

            // ACA iteration
            while (rank < max_rank && rel_error > params->tolerance) {
                // Find pivot: row with maximum residual norm
                size_t pivot_row = 0;
                double max_row_norm = 0.0;

                for (size_t i = 0; i < m; i++) {
                    if (row_used[i]) continue;
                    double row_norm = 0.0;
                    for (size_t j = 0; j < n; j++) {
                        row_norm += creal(residual[i * n + j] * conj(residual[i * n + j]));
                    }
                    if (row_norm > max_row_norm) {
                        max_row_norm = row_norm;
                        pivot_row = i;
                    }
                }

                if (max_row_norm < 1e-14) break;

                // Find pivot column in selected row
                size_t pivot_col = 0;
                double max_val = 0.0;
                for (size_t j = 0; j < n; j++) {
                    if (col_used[j]) continue;
                    double val = cabs(residual[pivot_row * n + j]);
                    if (val > max_val) {
                        max_val = val;
                        pivot_col = j;
                    }
                }

                if (max_val < 1e-14) break;

                double complex pivot = residual[pivot_row * n + pivot_col];

                // Extract row and column vectors
                for (size_t i = 0; i < m; i++) {
                    U_work[i * max_rank + rank] = residual[i * n + pivot_col];
                }
                for (size_t j = 0; j < n; j++) {
                    V_work[j * max_rank + rank] = residual[pivot_row * n + j] / pivot;
                }

                // Update residual: R = R - u * v^T
                for (size_t i = 0; i < m; i++) {
                    double complex u_i = U_work[i * max_rank + rank];
                    for (size_t j = 0; j < n; j++) {
                        residual[i * n + j] -= u_i * V_work[j * max_rank + rank];
                    }
                }

                row_used[pivot_row] = true;
                col_used[pivot_col] = true;
                rank++;

                // Compute relative error
                double res_norm = 0.0;
                for (size_t i = 0; i < m * n; i++) {
                    res_norm += creal(residual[i] * conj(residual[i]));
                }
                rel_error = sqrt(res_norm) / init_norm;
            }

            if (rank == 0) rank = 1;

            // Store result
            matrix->rank = rank;
            matrix->U = aligned_alloc(CACHE_LINE_SIZE, m * rank * sizeof(double complex));
            matrix->V = aligned_alloc(CACHE_LINE_SIZE, n * rank * sizeof(double complex));

            if (!matrix->U || !matrix->V) {
                free(U_work);
                free(V_work);
                free(residual);
                free(row_used);
                free(col_used);
                free(matrix->U);
                free(matrix->V);
                matrix->U = NULL;
                matrix->V = NULL;
                return false;
            }

            // Copy truncated matrices
            for (size_t i = 0; i < m; i++) {
                for (size_t j = 0; j < rank; j++) {
                    matrix->U[i * rank + j] = U_work[i * max_rank + j];
                }
            }
            for (size_t i = 0; i < n; i++) {
                for (size_t j = 0; j < rank; j++) {
                    matrix->V[i * rank + j] = V_work[i * max_rank + j];
                }
            }

            free(U_work);
            free(V_work);
            free(residual);
            free(row_used);
            free(col_used);
            free(matrix->data);
            matrix->data = NULL;

            return true;
        }

        case COMPRESS_QUANTUM: {
            // Quantum-inspired compression using randomized measurements
            // Based on quantum state tomography principles:
            // Approximate matrix by random projections and reconstruction

            size_t m = matrix->rows;
            size_t n = matrix->cols;
            size_t target_rank = params->max_rank;
            size_t oversampling = 10;  // Oversampling for better approximation
            size_t num_samples = target_rank + oversampling;

            if (num_samples > m) num_samples = m;
            if (num_samples > n) num_samples = n;

            // Allocate random projection matrix (Gaussian)
            double complex* Omega = aligned_alloc(CACHE_LINE_SIZE, n * num_samples * sizeof(double complex));
            double complex* Y = aligned_alloc(CACHE_LINE_SIZE, m * num_samples * sizeof(double complex));
            double complex* Q = aligned_alloc(CACHE_LINE_SIZE, m * num_samples * sizeof(double complex));
            double complex* B = aligned_alloc(CACHE_LINE_SIZE, num_samples * n * sizeof(double complex));

            if (!Omega || !Y || !Q || !B) {
                free(Omega);
                free(Y);
                free(Q);
                free(B);
                return false;
            }

            // Generate random Gaussian matrix (quantum-like random measurements)
            // Using Box-Muller transform for normal distribution
            srand(42);  // Reproducible randomness
            for (size_t i = 0; i < n * num_samples; i++) {
                double u1 = (rand() + 1.0) / (RAND_MAX + 1.0);
                double u2 = (rand() + 1.0) / (RAND_MAX + 1.0);
                double mag = sqrt(-2.0 * log(u1));
                double phase = 2.0 * M_PI * u2;
                Omega[i] = (mag * cos(phase) + I * mag * sin(phase)) / sqrt(2.0 * num_samples);
            }

            // Y = A * Omega (project onto random subspace)
            #pragma omp parallel for collapse(2)
            for (size_t i = 0; i < m; i++) {
                for (size_t j = 0; j < num_samples; j++) {
                    double complex sum = 0.0;
                    for (size_t k = 0; k < n; k++) {
                        sum += matrix->data[i * n + k] * Omega[k * num_samples + j];
                    }
                    Y[i * num_samples + j] = sum;
                }
            }

            // Orthonormalize Y using modified Gram-Schmidt to get Q
            memcpy(Q, Y, m * num_samples * sizeof(double complex));

            for (size_t j = 0; j < num_samples; j++) {
                // Normalize column j
                double norm = 0.0;
                for (size_t i = 0; i < m; i++) {
                    norm += creal(Q[i * num_samples + j] * conj(Q[i * num_samples + j]));
                }
                norm = sqrt(norm);

                if (norm > 1e-14) {
                    for (size_t i = 0; i < m; i++) {
                        Q[i * num_samples + j] /= norm;
                    }
                }

                // Orthogonalize remaining columns
                for (size_t k = j + 1; k < num_samples; k++) {
                    double complex dot = 0.0;
                    for (size_t i = 0; i < m; i++) {
                        dot += conj(Q[i * num_samples + j]) * Q[i * num_samples + k];
                    }
                    for (size_t i = 0; i < m; i++) {
                        Q[i * num_samples + k] -= dot * Q[i * num_samples + j];
                    }
                }
            }

            // B = Q^H * A (project A onto range of Q)
            #pragma omp parallel for collapse(2)
            for (size_t i = 0; i < num_samples; i++) {
                for (size_t j = 0; j < n; j++) {
                    double complex sum = 0.0;
                    for (size_t k = 0; k < m; k++) {
                        sum += conj(Q[k * num_samples + i]) * matrix->data[k * n + j];
                    }
                    B[i * n + j] = sum;
                }
            }

            // Now A ≈ Q * B, truncate to target rank
            size_t rank = (target_rank < num_samples) ? target_rank : num_samples;

            matrix->rank = rank;
            matrix->U = aligned_alloc(CACHE_LINE_SIZE, m * rank * sizeof(double complex));
            matrix->V = aligned_alloc(CACHE_LINE_SIZE, n * rank * sizeof(double complex));

            if (!matrix->U || !matrix->V) {
                free(Omega);
                free(Y);
                free(Q);
                free(B);
                free(matrix->U);
                free(matrix->V);
                matrix->U = NULL;
                matrix->V = NULL;
                return false;
            }

            // U = Q[:, :rank]
            for (size_t i = 0; i < m; i++) {
                for (size_t j = 0; j < rank; j++) {
                    matrix->U[i * rank + j] = Q[i * num_samples + j];
                }
            }

            // V = B[:rank, :]^T
            for (size_t i = 0; i < n; i++) {
                for (size_t j = 0; j < rank; j++) {
                    matrix->V[i * rank + j] = B[j * n + i];
                }
            }

            free(Omega);
            free(Y);
            free(Q);
            free(B);
            free(matrix->data);
            matrix->data = NULL;

            return true;
        }

        case COMPRESS_ADAPTIVE: {
            // Adaptive compression - selects best method based on matrix properties

            size_t m = matrix->rows;
            size_t n = matrix->cols;
            double density = 0.0;
            double symmetry_score = 0.0;
            size_t nnz = 0;

            // Analyze matrix properties
            for (size_t i = 0; i < m; i++) {
                for (size_t j = 0; j < n; j++) {
                    if (cabs(matrix->data[i * n + j]) > 1e-14) {
                        nnz++;
                    }
                }
            }
            density = (double)nnz / (m * n);

            // Check symmetry for square matrices
            if (m == n) {
                double sym_diff = 0.0;
                double total = 0.0;
                for (size_t i = 0; i < m; i++) {
                    for (size_t j = i + 1; j < n; j++) {
                        double complex diff = matrix->data[i * n + j] - conj(matrix->data[j * n + i]);
                        sym_diff += cabs(diff);
                        total += cabs(matrix->data[i * n + j]) + cabs(matrix->data[j * n + i]);
                    }
                }
                symmetry_score = (total > 1e-14) ? 1.0 - sym_diff / total : 1.0;
            }

            // Estimate numerical rank via random sampling
            size_t sample_size = (m < 20) ? m : 20;
            double* sampled_vals = malloc(sample_size * sizeof(double));
            if (sampled_vals) {
                for (size_t s = 0; s < sample_size; s++) {
                    size_t i = (s * m) / sample_size;
                    double row_norm = 0.0;
                    for (size_t j = 0; j < n; j++) {
                        row_norm += creal(matrix->data[i * n + j] * conj(matrix->data[i * n + j]));
                    }
                    sampled_vals[s] = sqrt(row_norm);
                }
                // Sort sampled values
                for (size_t i = 0; i < sample_size - 1; i++) {
                    for (size_t j = i + 1; j < sample_size; j++) {
                        if (sampled_vals[i] < sampled_vals[j]) {
                            double tmp = sampled_vals[i];
                            sampled_vals[i] = sampled_vals[j];
                            sampled_vals[j] = tmp;
                        }
                    }
                }
                free(sampled_vals);
            }

            // Select compression method based on analysis
            compression_params_t adapted_params = *params;

            if (density < 0.1) {
                // Sparse matrix - use ACA which exploits sparsity
                adapted_params.mode = COMPRESS_ACA;
            } else if (symmetry_score > 0.9 && m == n) {
                // Symmetric/Hermitian - SVD gives optimal approximation
                adapted_params.mode = COMPRESS_SVD;
            } else if (m * n > 1000000) {
                // Large matrix - use quantum (randomized) for efficiency
                adapted_params.mode = COMPRESS_QUANTUM;
            } else if (m != n) {
                // Rectangular - QR is efficient
                adapted_params.mode = COMPRESS_QR;
            } else {
                // Default to SVD for best accuracy
                adapted_params.mode = COMPRESS_SVD;
            }

            // Apply selected method
            return compress_matrix(matrix, &adapted_params);
        }
            
        default:
            return false;
    }
}

void hmatrix_destroy(HierarchicalMatrix* mat) {
    if (!mat) return;
    
    if (mat->is_leaf) {
        free(mat->data);
    } else {
        for (int i = 0; i < 4; i++) {
            hmatrix_destroy(mat->children[i]);
        }
    }
    
    free(mat->U);
    free(mat->V);
    free(mat);
}

void hmatrix_multiply(HierarchicalMatrix* dst, const HierarchicalMatrix* a,
                     const HierarchicalMatrix* b) {
    if (!dst || !a || !b || a->cols != b->rows) {
        printf("Invalid matrix multiplication arguments\n");
        return;
    }
    
    if (a->is_leaf && b->is_leaf) {
        // Cache-friendly blocking for matrix multiplication
        size_t block_size = 32; // Chosen to fit in L1 cache
        
        #ifdef USE_X86_SIMD
            // Use SIMD for leaf node multiplication with cache-friendly blocking
            #pragma omp parallel for collapse(2) schedule(static)
            for (size_t i = 0; i < a->rows; i += block_size) {
                for (size_t j = 0; j < b->cols; j += block_size) {
                    for (size_t k = 0; k < a->cols; k += block_size) {
                        size_t i_end = min(i + block_size, a->rows);
                        size_t j_end = min(j + block_size, b->cols);
                        size_t k_end = min(k + block_size, a->cols);
                        
                        for (size_t ii = i; ii < i_end; ii++) {
                            for (size_t jj = j; jj < j_end; jj++) {
                                __m512d sum_real = _mm512_setzero_pd();
                                __m512d sum_imag = _mm512_setzero_pd();
                                
                                for (size_t kk = k; kk < k_end; kk += 4) {
                                    __m512d a_vec = _mm512_load_pd((double*)&a->data[ii * a->cols + kk]);
                                    __m512d b_vec = _mm512_load_pd((double*)&b->data[kk * b->cols + jj]);
                                    
                                    __m512d prod_real = _mm512_mul_pd(a_vec, b_vec);
                                    __m512d prod_imag = _mm512_fmaddsub_pd(a_vec, b_vec, sum_imag);
                                    
                                    sum_real = _mm512_add_pd(sum_real, prod_real);
                                    sum_imag = _mm512_add_pd(sum_imag, prod_imag);
                                }
                                
                                dst->data[ii * b->cols + jj] = 
                                    _mm512_reduce_add_pd(sum_real) + 
                                    _mm512_reduce_add_pd(sum_imag) * I;
                            }
                        }
                    }
                }
            }
        #else
            // Generic implementation for other architectures
            #pragma omp parallel for collapse(2) schedule(static)
            for (size_t i = 0; i < a->rows; i += block_size) {
                for (size_t j = 0; j < b->cols; j += block_size) {
                    for (size_t k = 0; k < a->cols; k += block_size) {
                        size_t i_end = min(i + block_size, a->rows);
                        size_t j_end = min(j + block_size, b->cols);
                        size_t k_end = min(k + block_size, a->cols);
                        
                        for (size_t ii = i; ii < i_end; ii++) {
                            for (size_t jj = j; jj < j_end; jj++) {
                                double complex sum = 0;
                                
                                for (size_t kk = k; kk < k_end; kk++) {
                                    sum += a->data[ii * a->cols + kk] * b->data[kk * b->cols + jj];
                                }
                                
                                dst->data[ii * b->cols + jj] = sum;
                            }
                        }
                    }
                }
            }
        #endif
    } else if (hmatrix_is_low_rank(a) && hmatrix_is_low_rank(b)) {
        // Optimized low-rank multiplication
        size_t new_rank = a->rank * b->rank;
        
        if (new_rank <= MAX_RANK) {
            // Direct low-rank multiplication
            dst->rank = new_rank;
            size_t total_size = dst->rows * dst->rank * sizeof(double complex);
            dst->U = aligned_alloc(CACHE_LINE_SIZE, total_size);
            dst->V = aligned_alloc(CACHE_LINE_SIZE, total_size);
            
            // U = a->U * (a->V^T * b->U)
            // V = b->V
            double complex* temp = aligned_alloc(CACHE_LINE_SIZE, 
                                               a->rank * b->rank * sizeof(double complex));
            
            #pragma omp parallel for collapse(2)
            for (size_t i = 0; i < a->rank; i++) {
                for (size_t j = 0; j < b->rank; j++) {
                    double complex sum = 0;
                    for (size_t k = 0; k < a->cols; k++) {
                        sum += conj(a->V[k * a->rank + i]) * b->U[k * b->rank + j];
                    }
                    temp[i * b->rank + j] = sum;
                }
            }
            
            #pragma omp parallel for collapse(2)
            for (size_t i = 0; i < dst->rows; i++) {
                for (size_t j = 0; j < dst->rank; j++) {
                    double complex sum = 0;
                    for (size_t k = 0; k < a->rank; k++) {
                        sum += a->U[i * a->rank + k] * temp[k * b->rank + j];
                    }
                    dst->U[i * dst->rank + j] = sum;
                }
            }
            
            memcpy(dst->V, b->V, dst->cols * dst->rank * sizeof(double complex));
            free(temp);
        } else {
            // Need to recompress
            hmatrix_truncate(dst);
        }
    } else {
        // Recursive multiplication optimized for cache locality
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                HierarchicalMatrix* sub_result = create_hierarchical_matrix(
                    a->children[i]->rows, dst->tolerance
                );
                
                if (!sub_result) {
                    printf("Failed to allocate sub-result matrix\n");
                    continue;
                }
                
                for (int k = 0; k < 2; k++) {
                    HierarchicalMatrix* temp = create_hierarchical_matrix(
                        a->children[i]->rows, dst->tolerance
                    );
                    
                    if (!temp) {
                        printf("Failed to allocate temporary matrix\n");
                        continue;
                    }
                    
                    hmatrix_multiply(temp, a->children[i * 2 + k], 
                                   b->children[k * 2 + j]);
                    hmatrix_add(sub_result, sub_result, temp);
                    hmatrix_destroy(temp);
                }
                
                dst->children[i * 2 + j] = sub_result;
            }
        }
    }
}

bool hmatrix_is_low_rank(const HierarchicalMatrix* mat) {
    if (!mat) return false;
    if (mat->is_leaf) return false;
    return mat->rank > 0 && mat->rank <= MAX_RANK;
}

void hmatrix_compress(HierarchicalMatrix* mat) {
    if (!mat || mat->is_leaf || hmatrix_is_low_rank(mat)) return;
    
    // Try to compress each child first
    #pragma omp parallel for
    for (int i = 0; i < 4; i++) {
        hmatrix_compress(mat->children[i]);
    }
    
    // Estimate potential compression ratio
    size_t full_size = mat->rows * mat->cols;
    size_t low_rank_size = (mat->rows + mat->cols) * MAX_RANK;
    double compression_ratio = (double)low_rank_size / full_size;
    
    // Check if compression would be beneficial
    if (compression_ratio < COMPRESSION_THRESHOLD) {
        double error = hmatrix_error_estimate(mat);
        if (error < mat->tolerance) {
            // Convert to low-rank representation
            size_t max_dim = (mat->rows > mat->cols) ? mat->rows : mat->cols;
            double complex* S = aligned_alloc(CACHE_LINE_SIZE, max_dim * sizeof(double complex));
            if (!S) {
                printf("Failed to allocate singular values array\n");
                return;
            }
            
            compute_svd(mat->data, mat->rows, mat->cols, mat->U, S, mat->V);
            truncate_svd(mat->U, S, mat->V, mat->rows, mat->cols, &mat->rank, mat->tolerance);
            
            free(S);
            free(mat->data);
            mat->data = NULL;
            
            // Free child nodes
            for (int i = 0; i < 4; i++) {
                hmatrix_destroy(mat->children[i]);
                mat->children[i] = NULL;
            }
        }
    }
}

double hmatrix_error_estimate(const HierarchicalMatrix* mat) {
    if (!mat) return INFINITY;
    if (mat->is_leaf) return 0.0;
    
    double error = 0.0;
    
    if (hmatrix_is_low_rank(mat)) {
        #ifdef USE_X86_SIMD
            // Compute ||UV^T - A||_F efficiently using SIMD
            size_t vec_size = 8;
            size_t vec_count = (mat->rows * mat->cols) / vec_size;
            
            #pragma omp parallel
            {
                __m512d local_error = _mm512_setzero_pd();
                
                #pragma omp for reduction(+:error)
                for (size_t i = 0; i < mat->rows; i++) {
                    for (size_t j = 0; j < mat->cols; j++) {
                        double complex sum = 0;
                        
                        // Use SIMD for rank summation
                        __m512d sum_real = _mm512_setzero_pd();
                        __m512d sum_imag = _mm512_setzero_pd();
                        
                        for (size_t k = 0; k < mat->rank; k += 4) {
                            __m512d u_vec = _mm512_load_pd((double*)&mat->U[i * mat->rank + k]);
                            __m512d v_vec = _mm512_load_pd((double*)&mat->V[j * mat->rank + k]);
                            
                            sum_real = _mm512_fmadd_pd(u_vec, v_vec, sum_real);
                            sum_imag = _mm512_fmadd_pd(u_vec, v_vec, sum_imag);
                        }
                        
                        sum = _mm512_reduce_add_pd(sum_real) + 
                              _mm512_reduce_add_pd(sum_imag) * I;
                        
                        double complex d = sum - mat->data[i * mat->cols + j];
                        error += creal(d * conj(d));
                    }
                }
            }
        #else
            // Generic implementation for other architectures
            #pragma omp parallel for reduction(+:error)
            for (size_t i = 0; i < mat->rows; i++) {
                for (size_t j = 0; j < mat->cols; j++) {
                    double complex sum = 0;
                    
                    for (size_t k = 0; k < mat->rank; k++) {
                        sum += mat->U[i * mat->rank + k] * mat->V[j * mat->rank + k];
                    }
                    
                    double complex d = sum - mat->data[i * mat->cols + j];
                    error += creal(d * conj(d));
                }
            }
        #endif
    } else {
        // Sum errors from children in parallel
        #pragma omp parallel for reduction(+:error)
        for (int i = 0; i < 4; i++) {
            error += hmatrix_error_estimate(mat->children[i]);
        }
    }
    
    return sqrt(error);
}

void hmatrix_print_stats(const HierarchicalMatrix* mat) {
    if (!mat) return;
    
    printf("Matrix Stats:\n");
    printf("Dimensions: %zu x %zu\n", mat->rows, mat->cols);
    printf("Is leaf: %d\n", mat->is_leaf);
    printf("Rank: %zu\n", mat->rank);
    printf("Tolerance: %g\n", mat->tolerance);
    printf("Error estimate: %g\n", hmatrix_error_estimate(mat));
    printf("Memory usage: %zu bytes\n", 
           mat->data ? (mat->rows * mat->cols * sizeof(double complex)) :
           (mat->U ? (mat->rows * mat->rank + mat->cols * mat->rank) * sizeof(double complex) : 0));
}

void hmatrix_add(HierarchicalMatrix* dst, const HierarchicalMatrix* a,
                const HierarchicalMatrix* b) {
    if (!dst || !a || !b) return;
    if (a->rows != b->rows || a->cols != b->cols) {
        printf("Matrix dimensions mismatch in addition\n");
        return;
    }
    
    if (a->is_leaf && b->is_leaf) {
        #ifdef USE_X86_SIMD
            // Direct addition using SIMD
            size_t vec_size = 8;
            size_t vec_count = (a->rows * a->cols) / vec_size;
            
            #pragma omp parallel for
            for (size_t i = 0; i < vec_count; i++) {
                size_t idx = i * vec_size;
                __m512d a_real = _mm512_load_pd((double*)&a->data[idx]);
                __m512d a_imag = _mm512_load_pd((double*)&a->data[idx + vec_size/2]);
                __m512d b_real = _mm512_load_pd((double*)&b->data[idx]);
                __m512d b_imag = _mm512_load_pd((double*)&b->data[idx + vec_size/2]);
                
                __m512d sum_real = _mm512_add_pd(a_real, b_real);
                __m512d sum_imag = _mm512_add_pd(a_imag, b_imag);
                
                _mm512_store_pd((double*)&dst->data[idx], sum_real);
                _mm512_store_pd((double*)&dst->data[idx + vec_size/2], sum_imag);
            }
            
            // Handle remaining elements
            for (size_t i = vec_count * vec_size; i < a->rows * a->cols; i++) {
                dst->data[i] = a->data[i] + b->data[i];
            }
        #else
            // Generic implementation for other architectures
            #pragma omp parallel for
            for (size_t i = 0; i < a->rows * a->cols; i++) {
                dst->data[i] = a->data[i] + b->data[i];
            }
        #endif
    } else if (hmatrix_is_low_rank(a) && hmatrix_is_low_rank(b)) {
        // Low-rank addition with optimized memory layout
        size_t new_rank = a->rank + b->rank;
        if (new_rank <= MAX_RANK) {
            #ifdef USE_X86_SIMD
                // Combine U and V matrices with SIMD operations
                size_t vec_size = 8;
                size_t vec_count = (a->rows * a->rank) / vec_size;
                
                #pragma omp parallel for
                for (size_t i = 0; i < vec_count; i++) {
                    size_t idx = i * vec_size;
                    __m512d u_vec = _mm512_load_pd((double*)&a->U[idx]);
                    _mm512_store_pd((double*)&dst->U[idx], u_vec);
                }
                
                #pragma omp parallel for
                for (size_t i = 0; i < (b->rows * b->rank) / vec_size; i++) {
                    size_t idx = i * vec_size;
                    __m512d u_vec = _mm512_load_pd((double*)&b->U[idx]);
                    _mm512_store_pd((double*)&dst->U[a->rows * a->rank + idx], u_vec);
                }
                
                // Similar SIMD operations for V matrices
                vec_count = (a->cols * a->rank) / vec_size;
                
                #pragma omp parallel for
                for (size_t i = 0; i < vec_count; i++) {
                    size_t idx = i * vec_size;
                    __m512d v_vec = _mm512_load_pd((double*)&a->V[idx]);
                    _mm512_store_pd((double*)&dst->V[idx], v_vec);
                }
                
                #pragma omp parallel for
                for (size_t i = 0; i < (b->cols * b->rank) / vec_size; i++) {
                    size_t idx = i * vec_size;
                    __m512d v_vec = _mm512_load_pd((double*)&b->V[idx]);
                    _mm512_store_pd((double*)&dst->V[a->cols * a->rank + idx], v_vec);
                }
            #else
                // Generic implementation for other architectures
                // Copy U matrices
                for (size_t i = 0; i < a->rows; i++) {
                    for (size_t j = 0; j < a->rank; j++) {
                        dst->U[i * new_rank + j] = a->U[i * a->rank + j];
                    }
                    for (size_t j = 0; j < b->rank; j++) {
                        dst->U[i * new_rank + a->rank + j] = b->U[i * b->rank + j];
                    }
                }
                
                // Copy V matrices
                for (size_t i = 0; i < a->cols; i++) {
                    for (size_t j = 0; j < a->rank; j++) {
                        dst->V[i * new_rank + j] = a->V[i * a->rank + j];
                    }
                    for (size_t j = 0; j < b->rank; j++) {
                        dst->V[i * new_rank + a->rank + j] = b->V[i * b->rank + j];
                    }
                }
            #endif
            
            dst->rank = new_rank;
        } else {
            // Need to recompress
            hmatrix_compress(dst);
        }
    } else {
        // Recursive addition
        #pragma omp parallel for
        for (int i = 0; i < 4; i++) {
            hmatrix_add(dst->children[i], a->children[i], b->children[i]);
        }
    }
}

void hmatrix_truncate(HierarchicalMatrix* mat) {
    if (!mat || mat->is_leaf) return;
    
    if (hmatrix_is_low_rank(mat)) {
        // Recompress using optimized SVD
        size_t max_dim = (mat->rows > mat->cols) ? mat->rows : mat->cols;
        double complex* S = aligned_alloc(CACHE_LINE_SIZE, max_dim * sizeof(double complex));
        double complex* temp = aligned_alloc(CACHE_LINE_SIZE, mat->rows * mat->cols * sizeof(double complex));
        
        if (!S || !temp) {
            free(S);
            free(temp);
            printf("Failed to allocate temporary arrays for truncation\n");
            return;
        }
        
        #ifdef USE_X86_SIMD
            // Form UV^T efficiently using SIMD
            size_t vec_size = 8;
            size_t vec_count = (mat->rows * mat->cols) / vec_size;
            
            #pragma omp parallel for collapse(2)
            for (size_t i = 0; i < mat->rows; i++) {
                for (size_t j = 0; j < mat->cols; j++) {
                    __m512d sum_real = _mm512_setzero_pd();
                    __m512d sum_imag = _mm512_setzero_pd();
                    
                    for (size_t k = 0; k < mat->rank; k += 4) {
                        __m512d u_vec = _mm512_load_pd((double*)&mat->U[i * mat->rank + k]);
                        __m512d v_vec = _mm512_load_pd((double*)&mat->V[j * mat->rank + k]);
                        
                        sum_real = _mm512_fmadd_pd(u_vec, v_vec, sum_real);
                        sum_imag = _mm512_fmadd_pd(u_vec, v_vec, sum_imag);
                    }
                    
                    temp[i * mat->cols + j] = 
                        _mm512_reduce_add_pd(sum_real) + 
                        _mm512_reduce_add_pd(sum_imag) * I;
                }
            }
        #else
            // Generic implementation for other architectures
            #pragma omp parallel for collapse(2)
            for (size_t i = 0; i < mat->rows; i++) {
                for (size_t j = 0; j < mat->cols; j++) {
                    double complex sum = 0;
                    
                    for (size_t k = 0; k < mat->rank; k++) {
                        sum += mat->U[i * mat->rank + k] * mat->V[j * mat->rank + k];
                    }
                    
                    temp[i * mat->cols + j] = sum;
                }
            }
        #endif
        
        // Recompute SVD with lower rank
        compute_svd(temp, mat->rows, mat->cols, mat->U, S, mat->V);
        truncate_svd(mat->U, S, mat->V, mat->rows, mat->cols, &mat->rank, mat->tolerance);
        
        free(S);
        free(temp);
    } else {
        // Recursively truncate children
        #pragma omp parallel for
        for (int i = 0; i < 4; i++) {
            hmatrix_truncate(mat->children[i]);
        }
    }
}

void hmatrix_decompose(HierarchicalMatrix* mat) {
    if (!mat || mat->is_leaf) return;
    
    // Recursively decompose into quadtree structure
    size_t mid_row = mat->rows / 2;
    size_t mid_col = mat->cols / 2;
    
    // Create child nodes if they don't exist
    if (!mat->children[0]) {
        for (int i = 0; i < 4; i++) {
            size_t sub_rows = (i < 2) ? mid_row : (mat->rows - mid_row);
            size_t sub_cols = (i % 2 == 0) ? mid_col : (mat->cols - mid_col);
            mat->children[i] = hmatrix_create(sub_rows, sub_cols, mat->tolerance);
            
            if (!mat->children[i]) {
                printf("Failed to create child node %d\n", i);
                continue;
            }
            
            // Copy data to children
            #pragma omp parallel for collapse(2)
            for (size_t r = 0; r < sub_rows; r++) {
                for (size_t c = 0; c < sub_cols; c++) {
                    size_t global_r = (i < 2) ? r : (r + mid_row);
                    size_t global_c = (i % 2 == 0) ? c : (c + mid_col);
                    mat->children[i]->data[r * sub_cols + c] = 
                        mat->data[global_r * mat->cols + global_c];
                }
            }
        }
        
        // Free original data since it's now stored in children
        free(mat->data);
        mat->data = NULL;
    }
    
    // Recursively decompose children
    #pragma omp parallel for
    for (int i = 0; i < 4; i++) {
        hmatrix_decompose(mat->children[i]);
    }
}

void hmatrix_svd(HierarchicalMatrix* mat) {
    if (!mat || mat->is_leaf) return;
    
    // Allocate space for SVD results
    size_t max_dim = (mat->rows > mat->cols) ? mat->rows : mat->cols;
    mat->U = aligned_alloc(CACHE_LINE_SIZE, mat->rows * max_dim * sizeof(double complex));
    mat->V = aligned_alloc(CACHE_LINE_SIZE, mat->cols * max_dim * sizeof(double complex));
    double complex* S = aligned_alloc(CACHE_LINE_SIZE, max_dim * sizeof(double complex));
    
    if (!mat->U || !mat->V || !S) {
        free(mat->U);
        free(mat->V);
        free(S);
        printf("Failed to allocate SVD arrays\n");
        return;
    }
    
    // Compute SVD
    compute_svd(mat->data, mat->rows, mat->cols, mat->U, S, mat->V);
    
    // Scale U and V by singular values for better numerical stability
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            for (size_t i = 0; i < mat->rows; i++) {
                for (size_t j = 0; j < max_dim; j++) {
                    mat->U[i * max_dim + j] *= sqrt(cabs(S[j]));
                }
            }
        }
        
        #pragma omp section
        {
            for (size_t i = 0; i < mat->cols; i++) {
                for (size_t j = 0; j < max_dim; j++) {
                    mat->V[i * max_dim + j] *= sqrt(cabs(S[j]));
                }
            }
        }
    }
    
    // Determine rank and truncate
    truncate_svd(mat->U, S, mat->V, mat->rows, mat->cols, &mat->rank, mat->tolerance);
    
    // Free original data and temporary storage
    free(mat->data);
    mat->data = NULL;
    free(S);
    
    // Recursively apply SVD to children
    #pragma omp parallel for
    for (int i = 0; i < 4; i++) {
        if (mat->children[i]) {
            hmatrix_svd(mat->children[i]);
        }
    }
}

// Matrix-vector multiplication for tensor networks
void hmatrix_multiply_vector(const HierarchicalMatrix* matrix,
                            const double complex* input,
                            double complex* output,
                            size_t batch_size) {
    if (!matrix || !input || !output) return;

    size_t rows = matrix->rows;
    size_t cols = matrix->cols;

    if (matrix->is_leaf && matrix->data) {
        // Direct matrix-vector multiplication for leaf nodes
        #pragma omp parallel for
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t i = 0; i < rows; i++) {
                double complex sum = 0;
                for (size_t j = 0; j < cols; j++) {
                    sum += matrix->data[i * cols + j] * input[b * cols + j];
                }
                output[b * rows + i] = sum;
            }
        }
    } else if (hmatrix_is_low_rank(matrix)) {
        // Low-rank multiplication: output = U * (V^T * input)
        double complex* temp = aligned_alloc(CACHE_LINE_SIZE,
                                            batch_size * matrix->rank * sizeof(double complex));
        if (!temp) return;

        // Compute V^T * input
        #pragma omp parallel for collapse(2)
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t r = 0; r < matrix->rank; r++) {
                double complex sum = 0;
                for (size_t j = 0; j < cols; j++) {
                    sum += conj(matrix->V[j * matrix->rank + r]) * input[b * cols + j];
                }
                temp[b * matrix->rank + r] = sum;
            }
        }

        // Compute U * temp
        #pragma omp parallel for collapse(2)
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t i = 0; i < rows; i++) {
                double complex sum = 0;
                for (size_t r = 0; r < matrix->rank; r++) {
                    sum += matrix->U[i * matrix->rank + r] * temp[b * matrix->rank + r];
                }
                output[b * rows + i] = sum;
            }
        }

        free(temp);
    } else {
        // Recursive multiplication for hierarchical structure
        size_t mid_row = rows / 2;
        size_t mid_col = cols / 2;

        // Allocate temporary storage for recursive results
        double complex* temp_output = aligned_alloc(CACHE_LINE_SIZE,
                                                   batch_size * rows * sizeof(double complex));
        if (!temp_output) return;
        memset(temp_output, 0, batch_size * rows * sizeof(double complex));

        // Process each quadrant
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                if (!matrix->children[i * 2 + j]) continue;

                size_t sub_rows = matrix->children[i * 2 + j]->rows;
                size_t sub_cols = matrix->children[i * 2 + j]->cols;
                size_t row_offset = (i == 0) ? 0 : mid_row;
                size_t col_offset = (j == 0) ? 0 : mid_col;

                double complex* sub_input = aligned_alloc(CACHE_LINE_SIZE,
                                                         batch_size * sub_cols * sizeof(double complex));
                double complex* sub_output = aligned_alloc(CACHE_LINE_SIZE,
                                                          batch_size * sub_rows * sizeof(double complex));

                if (sub_input && sub_output) {
                    // Extract sub-input
                    #pragma omp parallel for
                    for (size_t b = 0; b < batch_size; b++) {
                        for (size_t c = 0; c < sub_cols; c++) {
                            sub_input[b * sub_cols + c] = input[b * cols + col_offset + c];
                        }
                    }

                    // Recursive call
                    hmatrix_multiply_vector(matrix->children[i * 2 + j],
                                           sub_input, sub_output, batch_size);

                    // Accumulate into temp_output
                    #pragma omp parallel for
                    for (size_t b = 0; b < batch_size; b++) {
                        for (size_t r = 0; r < sub_rows; r++) {
                            temp_output[b * rows + row_offset + r] += sub_output[b * sub_rows + r];
                        }
                    }
                }

                free(sub_input);
                free(sub_output);
            }
        }

        memcpy(output, temp_output, batch_size * rows * sizeof(double complex));
        free(temp_output);
    }
}

// Matrix conjugate transpose multiplication
void hmatrix_multiply_conjugate_transpose(const HierarchicalMatrix* matrix,
                                         const double complex* input,
                                         double complex* output,
                                         size_t batch_size) {
    if (!matrix || !input || !output) return;

    size_t rows = matrix->rows;
    size_t cols = matrix->cols;

    if (matrix->is_leaf && matrix->data) {
        // Direct A^H * x computation for leaf nodes
        #pragma omp parallel for
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t j = 0; j < cols; j++) {
                double complex sum = 0;
                for (size_t i = 0; i < rows; i++) {
                    sum += conj(matrix->data[i * cols + j]) * input[b * rows + i];
                }
                output[b * cols + j] = sum;
            }
        }
    } else if (hmatrix_is_low_rank(matrix)) {
        // (UV^T)^H * x = V * (U^H * x)
        double complex* temp = aligned_alloc(CACHE_LINE_SIZE,
                                            batch_size * matrix->rank * sizeof(double complex));
        if (!temp) return;

        // Compute U^H * input
        #pragma omp parallel for collapse(2)
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t r = 0; r < matrix->rank; r++) {
                double complex sum = 0;
                for (size_t i = 0; i < rows; i++) {
                    sum += conj(matrix->U[i * matrix->rank + r]) * input[b * rows + i];
                }
                temp[b * matrix->rank + r] = sum;
            }
        }

        // Compute V * temp
        #pragma omp parallel for collapse(2)
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t j = 0; j < cols; j++) {
                double complex sum = 0;
                for (size_t r = 0; r < matrix->rank; r++) {
                    sum += matrix->V[j * matrix->rank + r] * temp[b * matrix->rank + r];
                }
                output[b * cols + j] = sum;
            }
        }

        free(temp);
    } else {
        // Recursive conjugate transpose multiplication
        memset(output, 0, batch_size * cols * sizeof(double complex));

        size_t mid_row = rows / 2;
        size_t mid_col = cols / 2;

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                if (!matrix->children[i * 2 + j]) continue;

                size_t sub_rows = matrix->children[i * 2 + j]->rows;
                size_t sub_cols = matrix->children[i * 2 + j]->cols;
                size_t row_offset = (i == 0) ? 0 : mid_row;
                size_t col_offset = (j == 0) ? 0 : mid_col;

                double complex* sub_input = aligned_alloc(CACHE_LINE_SIZE,
                                                         batch_size * sub_rows * sizeof(double complex));
                double complex* sub_output = aligned_alloc(CACHE_LINE_SIZE,
                                                          batch_size * sub_cols * sizeof(double complex));

                if (sub_input && sub_output) {
                    #pragma omp parallel for
                    for (size_t b = 0; b < batch_size; b++) {
                        for (size_t r = 0; r < sub_rows; r++) {
                            sub_input[b * sub_rows + r] = input[b * rows + row_offset + r];
                        }
                    }

                    hmatrix_multiply_conjugate_transpose(matrix->children[i * 2 + j],
                                                        sub_input, sub_output, batch_size);

                    #pragma omp parallel for
                    for (size_t b = 0; b < batch_size; b++) {
                        for (size_t c = 0; c < sub_cols; c++) {
                            output[b * cols + col_offset + c] += sub_output[b * sub_cols + c];
                        }
                    }
                }

                free(sub_input);
                free(sub_output);
            }
        }
    }
}

// Apply gradient update to matrix
void hmatrix_apply_gradient(HierarchicalMatrix* matrix,
                           const double complex* gradient,
                           double learning_rate) {
    if (!matrix || !gradient) return;

    if (matrix->is_leaf && matrix->data) {
        size_t size = matrix->rows * matrix->cols;
        #pragma omp parallel for
        for (size_t i = 0; i < size; i++) {
            matrix->data[i] -= learning_rate * gradient[i];
        }

        // Update gradient storage if present
        if (matrix->grad) {
            memcpy(matrix->grad, gradient, size * sizeof(double complex));
        }
    } else if (hmatrix_is_low_rank(matrix)) {
        // Update U and V factors
        size_t u_size = matrix->rows * matrix->rank;
        size_t v_size = matrix->cols * matrix->rank;

        // Compute gradient for U and V from full gradient
        // This is an approximation for low-rank updates
        #pragma omp parallel for
        for (size_t i = 0; i < u_size; i++) {
            matrix->U[i] -= learning_rate * gradient[i % (matrix->rows * matrix->cols)] * 0.5;
        }

        #pragma omp parallel for
        for (size_t i = 0; i < v_size; i++) {
            matrix->V[i] -= learning_rate * gradient[i % (matrix->rows * matrix->cols)] * 0.5;
        }
    } else {
        // Recursive gradient application
        size_t mid_row = matrix->rows / 2;
        size_t mid_col = matrix->cols / 2;

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                if (!matrix->children[i * 2 + j]) continue;

                size_t sub_rows = matrix->children[i * 2 + j]->rows;
                size_t sub_cols = matrix->children[i * 2 + j]->cols;
                size_t row_offset = (i == 0) ? 0 : mid_row;
                size_t col_offset = (j == 0) ? 0 : mid_col;

                double complex* sub_gradient = aligned_alloc(CACHE_LINE_SIZE,
                                                            sub_rows * sub_cols * sizeof(double complex));
                if (sub_gradient) {
                    #pragma omp parallel for collapse(2)
                    for (size_t r = 0; r < sub_rows; r++) {
                        for (size_t c = 0; c < sub_cols; c++) {
                            sub_gradient[r * sub_cols + c] =
                                gradient[(row_offset + r) * matrix->cols + col_offset + c];
                        }
                    }

                    hmatrix_apply_gradient(matrix->children[i * 2 + j],
                                          sub_gradient, learning_rate);
                    free(sub_gradient);
                }
            }
        }
    }
}

// Matrix transpose
void hmatrix_transpose(HierarchicalMatrix* dst, const HierarchicalMatrix* src) {
    if (!dst || !src) return;

    dst->rows = src->cols;
    dst->cols = src->rows;
    dst->rank = src->rank;
    dst->tolerance = src->tolerance;
    dst->is_leaf = src->is_leaf;

    if (src->is_leaf && src->data) {
        size_t size = src->rows * src->cols;
        if (!dst->data) {
            dst->data = aligned_alloc(CACHE_LINE_SIZE, size * sizeof(double complex));
        }
        if (dst->data) {
            #pragma omp parallel for collapse(2)
            for (size_t i = 0; i < src->rows; i++) {
                for (size_t j = 0; j < src->cols; j++) {
                    dst->data[j * src->rows + i] = src->data[i * src->cols + j];
                }
            }
        }
    } else if (hmatrix_is_low_rank(src)) {
        // Transpose of UV^T is VU^T
        size_t u_size = src->rows * src->rank;
        size_t v_size = src->cols * src->rank;

        if (!dst->U) dst->U = aligned_alloc(CACHE_LINE_SIZE, v_size * sizeof(double complex));
        if (!dst->V) dst->V = aligned_alloc(CACHE_LINE_SIZE, u_size * sizeof(double complex));

        if (dst->U && dst->V) {
            memcpy(dst->U, src->V, v_size * sizeof(double complex));
            memcpy(dst->V, src->U, u_size * sizeof(double complex));
        }
    } else {
        // Recursive transpose with quadrant swapping
        // [A B]^T = [A^T C^T]
        // [C D]     [B^T D^T]
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                int src_idx = i * 2 + j;
                int dst_idx = j * 2 + i;  // Swap indices for transpose

                if (src->children[src_idx]) {
                    if (!dst->children[dst_idx]) {
                        dst->children[dst_idx] = hmatrix_create(
                            src->children[src_idx]->cols,
                            src->children[src_idx]->rows,
                            src->tolerance
                        );
                    }
                    if (dst->children[dst_idx]) {
                        hmatrix_transpose(dst->children[dst_idx], src->children[src_idx]);
                    }
                }
            }
        }
    }
}
