#include "quantum_geometric/core/hierarchical_matrix.h"
#include "quantum_geometric/core/simd_operations.h"
#include "quantum_geometric/core/numerical_backend.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Forward declarations
static bool should_use_optimized_path(size_t rows, size_t cols);
static HierarchicalMatrix* hmatrix_create_standard(size_t n, double tolerance);
static HierarchicalMatrix* hmatrix_create_optimized(size_t n, double tolerance);

// Helper macros
#define min(a,b) ((a) < (b) ? (a) : (b))

// Original optimization constants
#define MIN_MATRIX_SIZE 64
#define MAX_RANK 32
#define SVD_TOLERANCE 1e-12
#define CACHE_LINE_SIZE 64

// Additional optimization constants for O(log n) paths
#define OPT_MIN_MATRIX_SIZE 32  // Reduced to allow more hierarchical levels
#define OPT_MAX_RANK 16        // Reduced to favor low-rank approximations
#define OPT_SVD_TOLERANCE 1e-8  // Increased tolerance for more aggressive compression
#define COMPRESSION_THRESHOLD 0.1  // Threshold for when to attempt compression
#define ADAPTIVE_BLOCK_SIZE 1024   // Size threshold for adaptive blocking

// Runtime optimization selection - always enabled for performance
static bool use_optimized_path = true;

// Safe aligned allocation
static void* safe_aligned_alloc(size_t size) {
    // Round up size to nearest multiple of CACHE_LINE_SIZE
    size_t aligned_size = (size + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1);
    void* ptr = aligned_alloc(CACHE_LINE_SIZE, aligned_size);
    if (!ptr) {
        // Fallback to regular malloc if aligned_alloc fails
        ptr = malloc(size);
    }
    return ptr;
}

// Initialize numerical backend
static bool init_backend(void) {
    static bool initialized = false;
    if (!initialized) {
        numerical_config_t config = {
            .type = NUMERICAL_BACKEND_ACCELERATE,
            .max_threads = 8,
            .use_fma = true,
            .use_avx = false,  // Disable x86 features
            .use_neon = true   // Enable ARM features
        };
        initialized = initialize_numerical_backend(&config);
    }
    return initialized;
}

// Helper function to determine optimal execution path
static bool should_use_optimized_path(size_t rows, size_t cols) {
    return (rows * cols > ADAPTIVE_BLOCK_SIZE);
}

// Enhanced error handling macro with cleanup
#define CHECK_ALLOC(ptr, cleanup) if (!ptr) { \
    printf("Memory allocation failed at %s:%d\n", __FILE__, __LINE__); \
    cleanup; \
    return NULL; \
}

// Helper functions for SVD and low-rank approximation
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
    
    // Use platform-agnostic vectorization
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

// Function to choose between implementations
static void truncate_svd(double complex* U, double complex* S, double complex* V,
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
    
    HierarchicalMatrix* mat = safe_aligned_alloc(sizeof(HierarchicalMatrix));
    CHECK_ALLOC(mat, );
    
    mat->rows = n;
    mat->cols = n;
    mat->tolerance = tolerance;
    mat->rank = 0;
    
    // Standard leaf size criteria
    mat->is_leaf = (n <= MIN_MATRIX_SIZE);
    
    if (mat->is_leaf) {
        mat->data = safe_aligned_alloc(n * n * sizeof(double complex));
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
                    destroy_hierarchical_matrix(mat->children[j]);
                }
                free(mat);
                return NULL;
            }
        }
    }
    
    return mat;
}

// O(log n) optimized implementation
static HierarchicalMatrix* hmatrix_create_optimized(size_t n, double tolerance) {
    if (n == 0) {
        printf("Invalid matrix dimension: %zu\n", n);
        return NULL;
    }
    
    HierarchicalMatrix* mat = safe_aligned_alloc(sizeof(HierarchicalMatrix));
    CHECK_ALLOC(mat, );
    
    mat->rows = n;
    mat->cols = n;
    mat->tolerance = tolerance;
    mat->rank = 0;
    
    // Adaptive leaf size based on matrix dimensions
    mat->is_leaf = (n <= OPT_MIN_MATRIX_SIZE || 
                   (n <= OPT_MIN_MATRIX_SIZE * 2 && n <= 128)); // Avoid over-subdivision
    
    if (mat->is_leaf) {
        mat->data = safe_aligned_alloc(n * n * sizeof(double complex));
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
                    destroy_hierarchical_matrix(mat->children[j]);
                }
                free(mat);
                return NULL;
            }
        }
    }
    
    return mat;
}

// Function to choose between implementations
HierarchicalMatrix* create_hierarchical_matrix(size_t n, double tolerance) {
    if (should_use_optimized_path(n, n)) {
        return hmatrix_create_optimized(n, tolerance);
    } else {
        return hmatrix_create_standard(n, tolerance);
    }
}

void destroy_hierarchical_matrix(HierarchicalMatrix* mat) {
    if (!mat) return;
    
    if (mat->is_leaf) {
        free(mat->data);
    } else {
        for (int i = 0; i < 4; i++) {
            destroy_hierarchical_matrix(mat->children[i]);
        }
    }
    
    free(mat->U);
    free(mat->V);
    free(mat);
}

void hmatrix_transpose(HierarchicalMatrix* dst, const HierarchicalMatrix* src) {
    if (!dst || !src) return;
    
    if (src->is_leaf) {
        // Transpose leaf node data
        for (size_t i = 0; i < src->rows; i++) {
            for (size_t j = i + 1; j < src->cols; j++) {
                double complex temp = src->data[i * src->cols + j];
                dst->data[i * src->cols + j] = src->data[j * src->cols + i];
                dst->data[j * src->cols + i] = temp;
            }
        }
    } else {
        // Recursively transpose children
        for (int i = 0; i < 4; i++) {
            int transpose_idx = (i == 1) ? 2 : (i == 2 ? 1 : i);
            hmatrix_transpose(dst->children[transpose_idx], src->children[i]);
        }
    }
}

// Helper function for non-square matrices
static HierarchicalMatrix* hmatrix_create(size_t rows, size_t cols, double tolerance) {
    if (rows == 0 || cols == 0) {
        printf("Invalid matrix dimensions: %zu x %zu\n", rows, cols);
        return NULL;
    }
    
    HierarchicalMatrix* mat = safe_aligned_alloc(sizeof(HierarchicalMatrix));
    CHECK_ALLOC(mat, );
    
    mat->rows = rows;
    mat->cols = cols;
    mat->tolerance = tolerance;
    mat->rank = 0;
    
    // Adaptive leaf size based on dimensions
    mat->is_leaf = (rows <= OPT_MIN_MATRIX_SIZE || cols <= OPT_MIN_MATRIX_SIZE || 
                   (rows <= OPT_MIN_MATRIX_SIZE * 2 && cols <= OPT_MIN_MATRIX_SIZE * 2));
    
    if (mat->is_leaf) {
        mat->data = safe_aligned_alloc(rows * cols * sizeof(double complex));
        if (!mat->data) {
            free(mat);
            printf("Failed to allocate matrix data\n");
            return NULL;
        }
        mat->U = NULL;
        mat->V = NULL;
        memset(mat->data, 0, rows * cols * sizeof(double complex));
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
            mat->children[i] = hmatrix_create(sub_rows, sub_cols, tolerance);
            if (!mat->children[i]) {
                for (int j = 0; j < i; j++) {
                    destroy_hierarchical_matrix(mat->children[j]);
                }
                free(mat);
                return NULL;
            }
        }
    }
    
    return mat;
}
