#ifndef MATRIX_QR_H
#define MATRIX_QR_H

#include "quantum_geometric/core/quantum_geometric_types.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Compute QR decomposition of a matrix
 * 
 * Decomposes matrix A into Q*R where:
 * Q is orthogonal (m x m)
 * R is upper triangular (m x n)
 * 
 * @param a Input matrix A (m x n)
 * @param q Output matrix Q (m x m)
 * @param r Output matrix R (m x n)
 * @param m Number of rows
 * @param n Number of columns
 * @return true if successful, false otherwise
 */
bool compute_qr_decomposition(
    ComplexFloat* a,
    ComplexFloat* q,
    ComplexFloat* r,
    size_t m,
    size_t n);

/**
 * @brief Reduce matrix to upper Hessenberg form
 * 
 * Reduces matrix A to upper Hessenberg form H = Q*A*Q'
 * where Q is unitary and H is upper Hessenberg
 * 
 * @param a Input/output matrix A/H (n x n)
 * @param q Output transformation matrix Q (n x n)
 * @param n Matrix dimension
 * @return true if successful, false otherwise
 */
bool compute_hessenberg_form(
    ComplexFloat* a,
    ComplexFloat* q,
    size_t n);

// QR algorithm configuration
typedef struct {
    double convergence_threshold;  // Convergence threshold for eigenvalues
    size_t max_iterations;        // Maximum number of QR iterations
    bool compute_eigenvectors;    // Whether to compute eigenvectors
    bool use_shifts;             // Whether to use shifts in QR algorithm
    bool balance_matrix;         // Whether to balance matrix first
    void* custom_config;         // Additional custom configuration
} qr_config_t;

/**
 * @brief Set QR algorithm configuration
 * 
 * @param config Configuration structure
 * @return true if successful, false otherwise
 */
bool set_qr_config(const qr_config_t* config);

/**
 * @brief Get current QR algorithm configuration
 * 
 * @param config Output configuration structure
 * @return true if successful, false otherwise
 */
bool get_qr_config(qr_config_t* config);

#ifdef __cplusplus
}
#endif

#endif // MATRIX_QR_H
