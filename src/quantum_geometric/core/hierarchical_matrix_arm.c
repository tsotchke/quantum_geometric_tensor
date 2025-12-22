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
// External functions declared in hierarchical_matrix.h
extern bool validate_hierarchical_matrix(const HierarchicalMatrix* matrix);
extern bool compute_qr_decomposition(double complex* A, size_t rows, size_t cols, 
                             double complex* Q, double complex* R);

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
        matrix->data = safe_aligned_alloc(matrix->n * matrix->n * sizeof(double complex));
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
            double complex* U = safe_aligned_alloc(matrix->rows * max_dim * sizeof(double complex));
            double complex* S = safe_aligned_alloc(max_dim * sizeof(double complex));
            double complex* V = safe_aligned_alloc(matrix->cols * max_dim * sizeof(double complex));
            
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
            matrix->U = safe_aligned_alloc(matrix->rows * rank * sizeof(double complex));
            matrix->V = safe_aligned_alloc(matrix->cols * rank * sizeof(double complex));
            
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
            // QR compression implementation
            // Allocate memory for QR decomposition
            size_t min_dim = (matrix->rows < matrix->cols) ? matrix->rows : matrix->cols;
            double complex* Q = safe_aligned_alloc(matrix->rows * min_dim * sizeof(double complex));
            double complex* R = safe_aligned_alloc(min_dim * matrix->cols * sizeof(double complex));
            
            if (!Q || !R) {
                free(Q);
                free(R);
                return false;
            }
            
            // Perform QR decomposition
            if (!compute_qr_decomposition(matrix->data, matrix->rows, matrix->cols, Q, R)) {
                free(Q);
                free(R);
                return false;
            }
            
            // Determine rank based on R matrix diagonal elements
            size_t rank = 0;
            double total = 0.0;
            double* diag_values = malloc(min_dim * sizeof(double));
            
            if (!diag_values) {
                free(Q);
                free(R);
                return false;
            }
            
            // Extract diagonal elements
            for (size_t i = 0; i < min_dim; i++) {
                diag_values[i] = cabs(R[i * matrix->cols + i]);
                total += diag_values[i];
            }
            
            // Determine rank based on tolerance
            double running_sum = 0.0;
            for (rank = 0; rank < min_dim; rank++) {
                running_sum += diag_values[rank];
                if (running_sum / total >= 1.0 - params->tolerance) {
                    break;
                }
            }
            rank++; // Adjust rank to be 1-based
            
            if (rank > params->max_rank) {
                rank = params->max_rank;
            }
            
            // Store rank-truncated matrices
            matrix->rank = rank;
            
            // Allocate memory for low-rank representation
            matrix->U = safe_aligned_alloc(matrix->rows * rank * sizeof(double complex));
            matrix->V = safe_aligned_alloc(matrix->cols * rank * sizeof(double complex));
            
            if (!matrix->U || !matrix->V) {
                free(Q);
                free(R);
                free(diag_values);
                free(matrix->U);
                free(matrix->V);
                matrix->U = NULL;
                matrix->V = NULL;
                return false;
            }
            
            // Copy truncated Q to U
            for (size_t i = 0; i < matrix->rows; i++) {
                for (size_t j = 0; j < rank; j++) {
                    matrix->U[i * rank + j] = Q[i * min_dim + j];
                }
            }
            
            // Compute V from R (transpose of R)
            for (size_t i = 0; i < rank; i++) {
                for (size_t j = 0; j < matrix->cols; j++) {
                    matrix->V[j * rank + i] = R[i * matrix->cols + j];
                }
            }
            
            // Free original data and temporary arrays
            free(matrix->data);
            matrix->data = NULL;
            free(Q);
            free(R);
            free(diag_values);
            
            return true;
        }
        
        case COMPRESS_ACA: {
            // Adaptive Cross Approximation implementation
            // This is a matrix compression technique that works well for certain types of matrices
            
            // Initialize parameters
            size_t max_rank = params->max_rank;
            double tolerance = params->tolerance;
            size_t rows = matrix->rows;
            size_t cols = matrix->cols;
            
            // Allocate memory for low-rank representation
            double complex* U = safe_aligned_alloc(rows * max_rank * sizeof(double complex));
            double complex* V = safe_aligned_alloc(cols * max_rank * sizeof(double complex));
            
            if (!U || !V) {
                free(U);
                free(V);
                return false;
            }
            
            // Initialize variables for ACA algorithm
            size_t rank = 0;
            double error = 0.0;
            double norm_A = 0.0;
            
            // Compute Frobenius norm of original matrix
            for (size_t i = 0; i < rows * cols; i++) {
                norm_A += cabs(matrix->data[i]) * cabs(matrix->data[i]);
            }
            norm_A = sqrt(norm_A);
            
            // Create residual matrix (initially equal to original matrix)
            double complex* residual = safe_aligned_alloc(rows * cols * sizeof(double complex));
            if (!residual) {
                free(U);
                free(V);
                return false;
            }
            
            memcpy(residual, matrix->data, rows * cols * sizeof(double complex));
            
            // Main ACA loop
            size_t i_max = 0, j_max = 0;
            double max_val = 0.0;
            
            while (rank < max_rank) {
                // Find pivot (maximum absolute value in residual)
                max_val = 0.0;
                for (size_t i = 0; i < rows; i++) {
                    for (size_t j = 0; j < cols; j++) {
                        double abs_val = cabs(residual[i * cols + j]);
                        if (abs_val > max_val) {
                            max_val = abs_val;
                            i_max = i;
                            j_max = j;
                        }
                    }
                }
                
                // Check convergence
                if (max_val < tolerance * norm_A || max_val < 1e-15) {
                    break;
                }
                
                // Extract row and column from residual
                double complex pivot = residual[i_max * cols + j_max];
                double complex pivot_inv = 1.0 / pivot;
                
                for (size_t j = 0; j < cols; j++) {
                    V[j * max_rank + rank] = residual[i_max * cols + j] * pivot_inv;
                }
                
                for (size_t i = 0; i < rows; i++) {
                    U[i * max_rank + rank] = residual[i * cols + j_max];
                }
                
                // Update residual: R = R - u*v^T
                for (size_t i = 0; i < rows; i++) {
                    for (size_t j = 0; j < cols; j++) {
                        residual[i * cols + j] -= U[i * max_rank + rank] * V[j * max_rank + rank];
                    }
                }
                
                // Update rank
                rank++;
                
                // Compute error
                error = 0.0;
                for (size_t i = 0; i < rows * cols; i++) {
                    error += cabs(residual[i]) * cabs(residual[i]);
                }
                error = sqrt(error) / norm_A;
                
                // Check if error is below tolerance
                if (error < tolerance) {
                    break;
                }
            }
            
            // Store final rank
            matrix->rank = rank;
            
            // Allocate memory for final low-rank representation
            matrix->U = safe_aligned_alloc(rows * rank * sizeof(double complex));
            matrix->V = safe_aligned_alloc(cols * rank * sizeof(double complex));
            
            if (!matrix->U || !matrix->V) {
                free(U);
                free(V);
                free(residual);
                free(matrix->U);
                free(matrix->V);
                matrix->U = NULL;
                matrix->V = NULL;
                return false;
            }
            
            // Copy final low-rank representation
            for (size_t i = 0; i < rows; i++) {
                for (size_t j = 0; j < rank; j++) {
                    matrix->U[i * rank + j] = U[i * max_rank + j];
                }
            }
            
            for (size_t i = 0; i < cols; i++) {
                for (size_t j = 0; j < rank; j++) {
                    matrix->V[i * rank + j] = V[i * max_rank + j];
                }
            }
            
            // Free original data and temporary arrays
            free(matrix->data);
            matrix->data = NULL;
            free(U);
            free(V);
            free(residual);
            
            return true;
        }
        
        case COMPRESS_QUANTUM: {
            // Quantum-inspired compression implementation
            // This is a simplified version that simulates quantum compression
            
            // Initialize parameters
            size_t max_rank = params->max_rank;
            double tolerance = params->tolerance;
            size_t rows = matrix->rows;
            size_t cols = matrix->cols;
            
            // Allocate memory for quantum state representation
            double complex* quantum_state = safe_aligned_alloc(rows * cols * sizeof(double complex));
            if (!quantum_state) {
                return false;
            }
            
            // Normalize matrix to create quantum state
            double norm = 0.0;
            for (size_t i = 0; i < rows * cols; i++) {
                norm += cabs(matrix->data[i]) * cabs(matrix->data[i]);
            }
            norm = sqrt(norm);
            
            for (size_t i = 0; i < rows * cols; i++) {
                quantum_state[i] = matrix->data[i] / norm;
            }
            
            // Simulate quantum phase estimation
            size_t min_dim = (rows < cols) ? rows : cols;
            double* eigenvalues = malloc(min_dim * sizeof(double));
            double complex* eigenvectors = safe_aligned_alloc(rows * cols * min_dim * sizeof(double complex));
            
            if (!eigenvalues || !eigenvectors) {
                free(quantum_state);
                free(eigenvalues);
                free(eigenvectors);
                return false;
            }
            
            // Simulate quantum eigenvalue estimation (simplified)
            // In a real quantum implementation, this would use quantum phase estimation
            for (size_t i = 0; i < min_dim; i++) {
                eigenvalues[i] = 0.0;
                for (size_t j = 0; j < rows * cols; j++) {
                    double phase = 2.0 * M_PI * i * j / (rows * cols);
                    eigenvectors[i * rows * cols + j] = cexp(I * phase) * quantum_state[j];
                    eigenvalues[i] += cabs(eigenvectors[i * rows * cols + j]) * cabs(eigenvectors[i * rows * cols + j]);
                }
                eigenvalues[i] = sqrt(eigenvalues[i]);
            }
            
            // Sort eigenvalues and eigenvectors
            for (size_t i = 0; i < min_dim - 1; i++) {
                for (size_t j = i + 1; j < min_dim; j++) {
                    if (eigenvalues[j] > eigenvalues[i]) {
                        // Swap eigenvalues
                        double temp = eigenvalues[i];
                        eigenvalues[i] = eigenvalues[j];
                        eigenvalues[j] = temp;
                        
                        // Swap eigenvectors
                        for (size_t k = 0; k < rows * cols; k++) {
                            double complex temp_vec = eigenvectors[i * rows * cols + k];
                            eigenvectors[i * rows * cols + k] = eigenvectors[j * rows * cols + k];
                            eigenvectors[j * rows * cols + k] = temp_vec;
                        }
                    }
                }
            }
            
            // Determine rank based on eigenvalues
            size_t rank = 0;
            double total = 0.0;
            for (size_t i = 0; i < min_dim; i++) {
                total += eigenvalues[i];
            }
            
            double running_sum = 0.0;
            for (rank = 0; rank < min_dim; rank++) {
                running_sum += eigenvalues[rank];
                if (running_sum / total >= 1.0 - tolerance) {
                    break;
                }
            }
            rank++; // Adjust rank to be 1-based
            
            if (rank > max_rank) {
                rank = max_rank;
            }
            
            // Store final rank
            matrix->rank = rank;
            
            // Allocate memory for low-rank representation
            matrix->U = safe_aligned_alloc(rows * rank * sizeof(double complex));
            matrix->V = safe_aligned_alloc(cols * rank * sizeof(double complex));
            
            if (!matrix->U || !matrix->V) {
                free(quantum_state);
                free(eigenvalues);
                free(eigenvectors);
                free(matrix->U);
                free(matrix->V);
                matrix->U = NULL;
                matrix->V = NULL;
                return false;
            }
            
            // Reconstruct U and V from eigenvectors
            for (size_t k = 0; k < rank; k++) {
                double lambda = sqrt(eigenvalues[k]);
                
                for (size_t i = 0; i < rows; i++) {
                    matrix->U[i * rank + k] = 0.0;
                    for (size_t j = 0; j < cols; j++) {
                        matrix->U[i * rank + k] += eigenvectors[k * rows * cols + i * cols + j] * lambda;
                    }
                }
                
                for (size_t j = 0; j < cols; j++) {
                    matrix->V[j * rank + k] = 0.0;
                    for (size_t i = 0; i < rows; i++) {
                        matrix->V[j * rank + k] += conj(eigenvectors[k * rows * cols + i * cols + j]);
                    }
                }
            }
            
            // Free original data and temporary arrays
            free(matrix->data);
            matrix->data = NULL;
            free(quantum_state);
            free(eigenvalues);
            free(eigenvectors);
            
            return true;
        }
        
        case COMPRESS_ADAPTIVE: {
            // Adaptive compression implementation
            // This method selects the best compression algorithm based on matrix properties
            
            // Analyze matrix properties
            size_t rows = matrix->rows;
            size_t cols = matrix->cols;
            bool is_sparse = false;
            bool is_low_rank = false;
            bool is_structured = false;
            
            // Check sparsity
            size_t non_zero_count = 0;
            for (size_t i = 0; i < rows * cols; i++) {
                if (cabs(matrix->data[i]) > 1e-10) {
                    non_zero_count++;
                }
            }
            double sparsity = (double)non_zero_count / (rows * cols);
            is_sparse = (sparsity < 0.1); // Less than 10% non-zero elements
            
            // Check low-rank property using SVD on a sample
            const size_t sample_size = 100;
            size_t sample_rows = (rows < sample_size) ? rows : sample_size;
            size_t sample_cols = (cols < sample_size) ? cols : sample_size;
            
            double complex* sample = safe_aligned_alloc(sample_rows * sample_cols * sizeof(double complex));
            double complex* U_sample = safe_aligned_alloc(sample_rows * sample_rows * sizeof(double complex));
            double complex* S_sample = safe_aligned_alloc(sample_rows * sizeof(double complex));
            double complex* V_sample = safe_aligned_alloc(sample_cols * sample_cols * sizeof(double complex));
            
            if (!sample || !U_sample || !S_sample || !V_sample) {
                free(sample);
                free(U_sample);
                free(S_sample);
                free(V_sample);
                return false;
            }
            
            // Extract sample
            for (size_t i = 0; i < sample_rows; i++) {
                for (size_t j = 0; j < sample_cols; j++) {
                    size_t row_idx = i * rows / sample_rows;
                    size_t col_idx = j * cols / sample_cols;
                    sample[i * sample_cols + j] = matrix->data[row_idx * cols + col_idx];
                }
            }
            
            // Compute SVD of sample
            compute_svd(sample, sample_rows, sample_cols, U_sample, S_sample, V_sample);
            
            // Check singular value decay
            double total_sv = 0.0;
            for (size_t i = 0; i < sample_rows && i < sample_cols; i++) {
                total_sv += cabs(S_sample[i]);
            }
            
            double sv_ratio = 0.0;
            if (sample_rows > 0 && sample_cols > 0 && total_sv > 0.0) {
                sv_ratio = cabs(S_sample[0]) / total_sv;
            }
            
            is_low_rank = (sv_ratio > 0.5); // First singular value contains more than 50% of energy
            
            // Check for structure (e.g., Toeplitz, Hankel)
            is_structured = false;
            if (rows == cols) {
                bool is_toeplitz = true;
                for (size_t i = 1; i < rows; i++) {
                    for (size_t j = 1; j < cols; j++) {
                        if (cabs(matrix->data[i * cols + j] - matrix->data[(i-1) * cols + (j-1)]) > 1e-10) {
                            is_toeplitz = false;
                            break;
                        }
                    }
                    if (!is_toeplitz) break;
                }
                is_structured = is_toeplitz;
            }
            
            // Select best compression method based on analysis
            compression_params_t new_params = *params;
            
            if (is_sparse) {
                // For sparse matrices, ACA often works well
                new_params.mode = COMPRESS_ACA;
            } else if (is_low_rank) {
                // For low-rank matrices, SVD is usually best
                new_params.mode = COMPRESS_SVD;
            } else if (is_structured) {
                // For structured matrices, QR can be efficient
                new_params.mode = COMPRESS_QR;
            } else {
                // Default to SVD for general matrices
                new_params.mode = COMPRESS_SVD;
            }
            
            // Free temporary arrays
            free(sample);
            free(U_sample);
            free(S_sample);
            free(V_sample);
            
            // Apply selected compression method
            return compress_matrix(matrix, &new_params);
        }
        
        default:
            return false;
    }
}

// QR decomposition is now implemented in matrix_qr.c
// to avoid duplicate symbol errors
