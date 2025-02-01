#ifndef TEST_MATRIX_HELPERS_H
#define TEST_MATRIX_HELPERS_H

#include <math.h>
#include <stdlib.h>

/**
 * Make a matrix decomposable for tensor network operations.
 * Creates a low-rank matrix that can be efficiently decomposed.
 * 
 * @param matrix Pointer to matrix to modify
 * @param size Size of the square matrix
 */
static void make_decomposable_matrix(float* matrix, int size) {
    const int rank = size / 16;  // Use low rank for efficient decomposition
    
    // Create temporary matrices for U and V
    float* U = (float*)malloc(size * rank * sizeof(float));
    float* V = (float*)malloc(rank * size * sizeof(float));
    
    // Initialize U and V with random values
    for (int i = 0; i < size * rank; i++) {
        U[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < rank * size; i++) {
        V[i] = (float)rand() / RAND_MAX;
    }
    
    // Compute matrix = U * V
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            float sum = 0.0f;
            for (int k = 0; k < rank; k++) {
                sum += U[i * rank + k] * V[k * size + j];
            }
            matrix[i * size + j] = sum;
        }
    }
    
    free(U);
    free(V);
}

/**
 * Make a matrix well-conditioned for quantum algorithms.
 * Creates a matrix with condition number < 100 by ensuring
 * singular values are within a controlled range.
 * 
 * @param matrix Pointer to matrix to modify
 * @param size Size of the square matrix
 */
static void make_well_conditioned_matrix(float* matrix, int size) {
    // Create orthogonal matrices Q1 and Q2
    float* Q1 = (float*)malloc(size * size * sizeof(float));
    float* Q2 = (float*)malloc(size * size * sizeof(float));
    
    // Initialize with random values
    for (int i = 0; i < size * size; i++) {
        Q1[i] = (float)rand() / RAND_MAX;
        Q2[i] = (float)rand() / RAND_MAX;
    }
    
    // Orthogonalize Q1 and Q2 using Gram-Schmidt
    for (int i = 0; i < size; i++) {
        // Normalize column i
        float norm = 0.0f;
        for (int j = 0; j < size; j++) {
            norm += Q1[j * size + i] * Q1[j * size + i];
        }
        norm = sqrtf(norm);
        for (int j = 0; j < size; j++) {
            Q1[j * size + i] /= norm;
        }
        
        // Subtract projection onto previous columns
        for (int j = i + 1; j < size; j++) {
            float dot = 0.0f;
            for (int k = 0; k < size; k++) {
                dot += Q1[k * size + i] * Q1[k * size + j];
            }
            for (int k = 0; k < size; k++) {
                Q1[k * size + j] -= dot * Q1[k * size + i];
            }
        }
    }
    
    // Same for Q2
    for (int i = 0; i < size; i++) {
        float norm = 0.0f;
        for (int j = 0; j < size; j++) {
            norm += Q2[j * size + i] * Q2[j * size + i];
        }
        norm = sqrtf(norm);
        for (int j = 0; j < size; j++) {
            Q2[j * size + i] /= norm;
        }
        
        for (int j = i + 1; j < size; j++) {
            float dot = 0.0f;
            for (int k = 0; k < size; k++) {
                dot += Q2[k * size + i] * Q2[k * size + j];
            }
            for (int k = 0; k < size; k++) {
                Q2[k * size + j] -= dot * Q2[k * size + i];
            }
        }
    }
    
    // Create diagonal matrix with controlled singular values
    float* D = (float*)calloc(size * size, sizeof(float));
    for (int i = 0; i < size; i++) {
        // Values between 1 and 100 for good conditioning
        D[i * size + i] = 1.0f + 99.0f * i / (size - 1);
    }
    
    // Compute matrix = Q1 * D * Q2^T
    float* temp = (float*)malloc(size * size * sizeof(float));
    
    // temp = D * Q2^T
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            float sum = 0.0f;
            for (int k = 0; k < size; k++) {
                sum += D[i * size + k] * Q2[j * size + k];
            }
            temp[i * size + j] = sum;
        }
    }
    
    // matrix = Q1 * temp
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            float sum = 0.0f;
            for (int k = 0; k < size; k++) {
                sum += Q1[i * size + k] * temp[k * size + j];
            }
            matrix[i * size + j] = sum;
        }
    }
    
    free(Q1);
    free(Q2);
    free(D);
    free(temp);
}

/**
 * Make a matrix sparse with given density.
 * Sets random elements to zero until desired sparsity is achieved.
 * 
 * @param matrix Pointer to matrix to modify
 * @param size Size of the square matrix
 * @param sparsity Target sparsity (0.0 to 1.0, e.g. 0.95 for 95% zeros)
 */
static void make_sparse_matrix(float* matrix, int size, float sparsity) {
    int total_elements = size * size;
    int num_zeros = (int)(total_elements * sparsity);
    
    // Keep track of which elements are already zeroed
    char* is_zero = (char*)calloc(total_elements, sizeof(char));
    
    // Randomly set elements to zero
    for (int i = 0; i < num_zeros; i++) {
        int pos;
        do {
            pos = rand() % total_elements;
        } while (is_zero[pos]);
        
        matrix[pos] = 0.0f;
        is_zero[pos] = 1;
    }
    
    free(is_zero);
}

#endif // TEST_MATRIX_HELPERS_H
