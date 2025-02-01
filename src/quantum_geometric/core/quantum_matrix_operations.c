#include "quantum_geometric/core/quantum_matrix_operations.h"
#include "quantum_geometric/core/memory_pool.h"
#include "quantum_geometric/core/tensor_network.h"
#include "quantum_geometric/core/hierarchical_matrix.h"
#include <math.h>
#include <string.h>
#include <complex.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#include <vecLib/vecLibTypes.h>
#include <vecLib/cblas.h>
#include <vecLib/clapack.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif

// Cache line size for alignment
#define CACHE_LINE_SIZE 64

// Align data to cache line boundary
static inline void* align_ptr(void* ptr) {
    return (void*)(((uintptr_t)ptr + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1));
}

// Helper functions for O(log n) operations
static bool decompose_recursive(const float* matrix, int size, tensor_network_t* network, int depth);
static bool compute_condition_recursive(const HierarchicalMatrix* hmatrix, float* min_sv, float* max_sv);

bool quantum_decompose_matrix(float* matrix, int size, float* U, float* V) {
    // Use LAPACKE SVD for optimal performance
    float* s = malloc(size * sizeof(float));
    if (!s) {
        return false;
    }
    
#ifdef __APPLE__
    // Use Accelerate framework's LAPACK interface
    int m = size;
    int n = size;
    int lda = size;
    int ldu = size;
    int ldvt = size;
    int lwork = -1;
    int info;
    float wkopt;
    
    // Query optimal workspace size
    char jobu = 'S';
    char jobvt = 'S';
    sgesvd_(&jobu, &jobvt, &m, &n, matrix, &lda, s, U, &ldu, V, &ldvt,
            &wkopt, &lwork, &info);
            
    lwork = (int)wkopt;
    float* work = (float*)malloc(lwork * sizeof(float));
    if (!work) {
        free(s);
        return false;
    }
    
    // Compute SVD
    sgesvd_(&jobu, &jobvt, &m, &n, matrix, &lda, s, U, &ldu, V, &ldvt,
            work, &lwork, &info);
            
    free(work);
    free(s);
    return info == 0;
#else
    // Use LAPACKE interface
    int info = LAPACKE_sgesvd(LAPACK_ROW_MAJOR, 'S', 'S', size, size,
                             matrix, size, s, U, size, V, size,
                             NULL); // superb parameter not needed
    
    free(s);
    return info == 0;
#endif
}

bool quantum_compute_condition_number(float** matrix, int size, float* condition_number) {
    // Use LAPACKE SVD to compute singular values directly
    float* a = malloc(size * size * sizeof(float));
    float* s = malloc(size * sizeof(float));
    
    if (!a || !s) {
        free(a);
        free(s);
        return false;
    }
    
    // Copy matrix to contiguous array in row-major order
    for (int i = 0; i < size; i++) {
        memcpy(&a[i*size], matrix[i], size * sizeof(float));
    }
    
#ifdef __APPLE__
    // Use Accelerate framework's LAPACK interface
    int m = size;
    int n = size;
    int lda = size;
    int lwork = -1;
    int info;
    float wkopt;
    
    // Query optimal workspace size
    char jobu = 'N';
    char jobvt = 'N';
    sgesvd_(&jobu, &jobvt, &m, &n, a, &lda, s, NULL, &m, NULL, &n,
            &wkopt, &lwork, &info);
            
    lwork = (int)wkopt;
    float* work = malloc(lwork * sizeof(float));
    if (!work) {
        free(a);
        free(s);
        return false;
    }
    
    // Compute SVD for singular values only
    sgesvd_(&jobu, &jobvt, &m, &n, a, &lda, s, NULL, &m, NULL, &n,
            work, &lwork, &info);
            
    free(work);
#else
    // Use LAPACKE interface
    int info = LAPACKE_sgesvd(LAPACK_ROW_MAJOR, 'N', 'N', size, size,
                             a, size, s, NULL, size, NULL, size,
                             NULL); // superb parameter not needed
#endif
            
    if (info == 0) {
        // Compute condition number as ratio of largest to smallest singular value
        float min_sv = s[size-1];
        float max_sv = s[0];
        
        if (min_sv < 1e-6f) {
            min_sv = 1e-6f; // Avoid division by zero
        }
        *condition_number = max_sv / min_sv;
    }
    
    free(a);
    free(s);
    
    return info == 0;
}

bool quantum_matrix_to_tensor_network(const float* matrix, int size, tensor_network_t* network) {
    // Initialize network
    if (!qg_tensor_network_init(network, 1)) {
        return false;
    }

    // Convert using recursive divide-and-conquer
    return decompose_recursive(matrix, size, network, (int)log2(size));
}

bool quantum_matrix_to_hierarchical(const float* matrix, int size, HierarchicalMatrix* hmatrix) {
    if (!matrix || !hmatrix) {
        return false;
    }

    // Copy data and compress recursively
    memcpy(hmatrix->data, matrix, size * size * sizeof(float));
    hierarchical_matrix_compress(hmatrix);  // Using the correct function name

    return true;
}

bool quantum_optimize_decomposition(float* U, float* V, int size, float tolerance) {
    // Use CBLAS for matrix multiplication
    float* temp = malloc(size * size * sizeof(float));
    float* s = malloc(size * sizeof(float));
    
    if (!temp || !s) {
        free(temp);
        free(s);
        return false;
    }
    
    // Compute U*V using CBLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                size, size, size,
                1.0f, U, size, V, size,
                0.0f, temp, size);
           
#ifdef __APPLE__
    // Use Accelerate framework's LAPACK interface
    int m = size;
    int n = size;
    int lda = size;
    int ldu = size;
    int ldvt = size;
    int lwork = -1;
    int info;
    float wkopt;
    
    // Query optimal workspace size
    char jobu = 'S';
    char jobvt = 'S';
    sgesvd_(&jobu, &jobvt, &m, &n, temp, &lda, s, U, &ldu, V, &ldvt,
            &wkopt, &lwork, &info);
            
    lwork = (int)wkopt;
    float* work = malloc(lwork * sizeof(float));
    if (!work) {
        free(temp);
        free(s);
        return false;
    }
    
    // Compute optimized SVD
    sgesvd_(&jobu, &jobvt, &m, &n, temp, &lda, s, U, &ldu, V, &ldvt,
            work, &lwork, &info);
            
    free(work);
#else
    // Use LAPACKE interface
    int info = LAPACKE_sgesvd(LAPACK_ROW_MAJOR, 'S', 'S', size, size,
                             temp, size, s, U, size, V, size,
                             NULL); // superb parameter not needed
#endif
            
    if (info == 0) {
        // Apply tolerance-based truncation
        int rank = size;
        float total = 0.0f;
        for (int i = 0; i < size; i++) {
            total += s[i] * s[i];
        }
        
        float running = 0.0f;
        for (int i = 0; i < size; i++) {
            running += s[i] * s[i];
            if (running / total >= 1.0f - tolerance) {
                rank = i + 1;
                break;
            }
        }
        
        // Scale U and V by singular values
        for (int i = 0; i < size; i++) {
            float scale = (i < rank) ? sqrtf(s[i]) : 0.0f;
            for (int j = 0; j < size; j++) {
                U[j * size + i] *= scale;
                V[i * size + j] *= scale;
            }
        }
    }
    
    free(temp);
    free(s);
    
    return info == 0;
}

// Helper function implementations
static bool decompose_recursive(const float* matrix, int size, tensor_network_t* network, int depth) {
    if (depth == 0) {
        // Base case: convert block to tensor with geometric encoding
        tensor_t tensor;
        size_t dims[] = {size, size};
        if (!qg_tensor_init(&tensor, dims, 2)) {
            return false;
        }
        
        // Apply geometric encoding
        ComplexFloat* encoded = malloc(size * size * sizeof(ComplexFloat));
        if (!encoded) {
            qg_tensor_cleanup(&tensor);
            return false;
        }
        
        // Compute geometric features
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                // Use relative positions for geometric encoding
                float x = (float)i / size;
                float y = (float)j / size;
                float dist = sqrtf(x*x + y*y);
                float angle = atan2f(y, x);
                
                // Combine geometric and value information
                float val = matrix[i * size + j] * 
                    (1.0f + 0.1f * cosf(2.0f * M_PI * dist) + 
                     0.1f * sinf(angle));
                encoded[i * size + j].real = val;
                encoded[i * size + j].imag = 0.0f;
            }
        }
        
        memcpy(tensor.data, encoded, size * size * sizeof(ComplexFloat));
        free(encoded);
        
        return qg_tensor_network_add_node(network, &tensor, NULL);
    }

    // Divide matrix into blocks
    int block_size = size / 2;
    tensor_network_t sub_networks[4];
    
    for (int i = 0; i < 4; i++) {
        if (!qg_tensor_network_init(&sub_networks[i], 1)) {
            for (int j = 0; j < i; j++) {
                qg_tensor_network_cleanup(&sub_networks[j]);
            }
            return false;
        }
    }

    // Process blocks recursively with geometric awareness
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            float* block = malloc(block_size * block_size * sizeof(float));
            if (!block) {
                return false;
            }

            // Extract block with geometric weighting
            float block_center_x = (i + 0.5f) / 2.0f;
            float block_center_y = (j + 0.5f) / 2.0f;
            
            for (int r = 0; r < block_size; r++) {
                for (int c = 0; c < block_size; c++) {
                    // Apply geometric weighting based on distance from block center
                    float x = ((float)r / block_size + i) / 2.0f;
                    float y = ((float)c / block_size + j) / 2.0f;
                    float dist = sqrtf(powf(x - block_center_x, 2) + 
                                     powf(y - block_center_y, 2));
                    float weight = expf(-2.0f * dist);
                    
                    block[r * block_size + c] = 
                        matrix[(i * block_size + r) * size + (j * block_size + c)] * weight;
                }
            }

            // Process recursively
            if (!decompose_recursive(block, block_size, 
                                  &sub_networks[i * 2 + j], depth - 1)) {
                free(block);
                return false;
            }

            free(block);
        }
    }

    // Combine results
    for (int i = 0; i < 4; i++) {
        size_t edge_index = network->num_nodes;
        if (!qg_tensor_network_add_node(network, 
                                       &sub_networks[i].nodes[0], NULL)) {
            return false;
        }
        if (i > 0) {
            if (!qg_tensor_network_connect_nodes(network, edge_index - 1, 
                                               edge_index, i - 1, i)) {
                return false;
            }
        }
    }

    return true;
}

static bool compute_condition_recursive(const HierarchicalMatrix* hmatrix, 
                                     float* min_sv, float* max_sv) {
    if (hmatrix->is_leaf) {
        // Base case: compute SVD directly for small block
        tensor_t tensor;
        size_t dims[] = {hmatrix->rows, hmatrix->cols};
        if (!qg_tensor_init(&tensor, dims, 2)) {
            return false;
        }
        
        // Convert data to ComplexFloat format
        ComplexFloat* complex_data = malloc(hmatrix->n * sizeof(ComplexFloat));
        if (!complex_data) {
            qg_tensor_cleanup(&tensor);
            return false;
        }
        
        for (size_t i = 0; i < hmatrix->n; i++) {
            complex_data[i].real = creal(hmatrix->data[i]);
            complex_data[i].imag = cimag(hmatrix->data[i]);
        }
        
        memcpy(tensor.data, complex_data, hmatrix->n * sizeof(ComplexFloat));
        free(complex_data);

        tensor_t u, s, v;
        if (!qg_tensor_decompose_svd(&tensor, 0, &u, &s, &v)) {
            qg_tensor_cleanup(&tensor);
            return false;
        }

        // Update min/max singular values
        for (size_t i = 0; i < hmatrix->rank; i++) {
            ComplexFloat val = s.data[i];
            float sv = complex_float_abs(val);
            if (sv < *min_sv) *min_sv = sv;
            if (sv > *max_sv) *max_sv = sv;
        }

        qg_tensor_cleanup(&tensor);
        qg_tensor_cleanup(&u);
        qg_tensor_cleanup(&s);
        qg_tensor_cleanup(&v);
        return true;
    }

    // Recurse on children
    for (int i = 0; i < 4; i++) {
        if (hmatrix->children[i]) {
            if (!compute_condition_recursive(hmatrix->children[i], min_sv, max_sv)) {
                return false;
            }
        }
    }

    return true;
}

// Error string function
const char* quantum_matrix_get_error_string(quantum_matrix_error_t error) {
    switch (error) {
        case QUANTUM_MATRIX_SUCCESS:
            return "Success";
        case QUANTUM_MATRIX_INVALID_INPUT:
            return "Invalid input parameters";
        case QUANTUM_MATRIX_DECOMPOSITION_FAILED:
            return "Matrix decomposition failed";
        case QUANTUM_MATRIX_MEMORY_ERROR:
            return "Memory allocation failed";
        case QUANTUM_MATRIX_NUMERICAL_ERROR:
            return "Numerical computation error";
        default:
            return "Unknown error";
    }
}
