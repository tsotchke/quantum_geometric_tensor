#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H

#include "quantum_geometric/core/quantum_geometric_types.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Multiply two complex matrices
 * 
 * Computes C = A * B where:
 * A is m x n
 * B is n x p
 * C is m x p
 * 
 * @param a First matrix (m x n)
 * @param b Second matrix (n x p)
 * @param result Result matrix (m x p)
 * @param m Number of rows in A
 * @param n Number of columns in A/rows in B
 * @param p Number of columns in B
 * @return true if successful, false otherwise
 */
bool matrix_multiply(
    const ComplexFloat* a,
    const ComplexFloat* b,
    ComplexFloat* result,
    size_t m,
    size_t n,
    size_t p);

/**
 * @brief Solve linear system Ax = b
 * 
 * Uses LU decomposition to solve the system.
 * Matrix A must be square and non-singular.
 * 
 * @param a System matrix A (n x n)
 * @param b Right-hand side vector b (n)
 * @param x Solution vector x (n)
 * @param n System dimension
 * @return true if successful, false otherwise
 */
bool solve_linear_system(
    const ComplexFloat* a,
    const ComplexFloat* b,
    ComplexFloat* x,
    size_t n);

/**
 * @brief Compute matrix inverse
 * 
 * Uses LU decomposition to compute inverse.
 * Matrix must be square and non-singular.
 * 
 * @param a Input matrix (n x n)
 * @param inverse Result matrix (n x n)
 * @param n Matrix dimension
 * @return true if successful, false otherwise
 */
bool matrix_inverse(
    const ComplexFloat* a,
    ComplexFloat* inverse,
    size_t n);

/**
 * @brief Compute matrix eigenvalues
 * 
 * Uses QR algorithm to compute eigenvalues.
 * Matrix must be square.
 * 
 * @param a Input matrix (n x n)
 * @param eigenvalues Array to store eigenvalues (n)
 * @param n Matrix dimension
 * @param max_iter Maximum number of iterations
 * @return true if successful, false otherwise
 */
bool compute_eigenvalues(
    const ComplexFloat* a,
    ComplexFloat* eigenvalues,
    size_t n,
    size_t max_iter);

/**
 * @brief Compute matrix eigenvectors
 * 
 * Uses inverse iteration to compute eigenvectors.
 * Matrix must be square.
 * 
 * @param a Input matrix (n x n)
 * @param eigenvalues Array of eigenvalues (n)
 * @param eigenvectors Result matrix (n x n), each column is an eigenvector
 * @param n Matrix dimension
 * @return true if successful, false otherwise
 */
bool compute_eigenvectors(
    const ComplexFloat* a,
    const ComplexFloat* eigenvalues,
    ComplexFloat* eigenvectors,
    size_t n);

// Additional matrix operation flags
#define MATRIX_OP_HERMITIAN    (1 << 0)  // Matrix is Hermitian
#define MATRIX_OP_UNITARY      (1 << 1)  // Matrix is unitary
#define MATRIX_OP_POSITIVE     (1 << 2)  // Matrix is positive definite
#define MATRIX_OP_SPARSE       (1 << 3)  // Matrix is sparse
#define MATRIX_OP_SYMMETRIC    (1 << 4)  // Matrix is symmetric
#define MATRIX_OP_DIAGONAL     (1 << 5)  // Matrix is diagonal
#define MATRIX_OP_TRIANGULAR   (1 << 6)  // Matrix is triangular

// Matrix operation configuration
typedef struct {
    uint32_t flags;            // Operation flags
    double tolerance;          // Numerical tolerance
    size_t max_iterations;     // Maximum iterations for iterative methods
    bool use_gpu;             // Use GPU acceleration if available
    void* custom_config;       // Additional custom configuration
} matrix_config_t;

/**
 * @brief Set matrix operation configuration
 * 
 * @param config Configuration structure
 * @return true if successful, false otherwise
 */
bool set_matrix_config(const matrix_config_t* config);

/**
 * @brief Get current matrix operation configuration
 * 
 * @param config Output configuration structure
 * @return true if successful, false otherwise
 */
bool get_matrix_config(matrix_config_t* config);

#ifdef __cplusplus
}
#endif

#endif // MATRIX_OPERATIONS_H
