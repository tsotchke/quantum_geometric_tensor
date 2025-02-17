#ifndef MATRIX_EIGENVALUES_H
#define MATRIX_EIGENVALUES_H

#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/matrix_qr.h"
#include "quantum_geometric/core/matrix_operations.h"
#include "quantum_geometric/core/quantum_complex.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif


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

/**
 * @brief Find eigenvalues and optionally eigenvectors of a matrix
 * 
 * Uses QR algorithm with shifts to find eigenvalues.
 * If eigenvectors is not NULL and compute_eigenvectors is enabled
 * in QR config, also computes eigenvectors.
 * 
 * @param a Input matrix (n x n)
 * @param eigenvalues Output array for eigenvalues (n)
 * @param eigenvectors Output matrix for eigenvectors (n x n), can be NULL
 * @param n Matrix dimension
 * @return true if successful, false otherwise
 */
bool find_eigenvalues(
    ComplexFloat* a,
    ComplexFloat* eigenvalues,
    ComplexFloat* eigenvectors,
    size_t n);

#ifdef __cplusplus
}
#endif

#endif // MATRIX_EIGENVALUES_H
