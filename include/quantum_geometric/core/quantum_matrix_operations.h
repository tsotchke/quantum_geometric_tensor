#ifndef QUANTUM_MATRIX_OPERATIONS_H
#define QUANTUM_MATRIX_OPERATIONS_H

#include <stdbool.h>
#include <stddef.h>
#include "quantum_geometric/core/quantum_geometric_types.h"

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#include <vecLib/vecLibTypes.h>
#include <vecLib/clapack.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif

// Forward declarations
typedef struct tensor_network_t tensor_network_t;
typedef struct HierarchicalMatrix HierarchicalMatrix;

// Error codes
typedef enum {
    QUANTUM_MATRIX_SUCCESS = 0,
    QUANTUM_MATRIX_INVALID_INPUT = -1,
    QUANTUM_MATRIX_DECOMPOSITION_FAILED = -2,
    QUANTUM_MATRIX_MEMORY_ERROR = -3,
    QUANTUM_MATRIX_NUMERICAL_ERROR = -4
} quantum_matrix_error_t;

/**
 * @brief Decompose a matrix into U and V components using tensor networks
 * 
 * Uses geometric encoding and adaptive rank selection for O(log n) complexity.
 * 
 * @param matrix Input matrix
 * @param size Matrix dimension
 * @param U Output U matrix (size x rank)
 * @param V Output V matrix (rank x size)
 * @return true on success, false on failure
 */
bool quantum_decompose_matrix(float* matrix, int size, float* U, float* V);

/**
 * @brief Compute condition number of a matrix using hierarchical representation
 * 
 * Uses recursive divide-and-conquer for O(log n) complexity.
 * 
 * @param matrix Input matrix
 * @param size Matrix dimension
 * @param condition_number Output condition number
 * @return true on success, false on failure
 */
bool quantum_compute_condition_number(float** matrix, int size, float* condition_number);

/**
 * @brief Convert dense matrix to tensor network representation
 * 
 * Uses geometric encoding for improved numerical stability.
 * 
 * @param matrix Input dense matrix
 * @param size Matrix dimension
 * @param network Output tensor network
 * @return true on success, false on failure
 */
bool quantum_matrix_to_tensor_network(const float* matrix, int size, tensor_network_t* network);

/**
 * @brief Convert dense matrix to hierarchical matrix representation
 * 
 * @param matrix Input dense matrix
 * @param size Matrix dimension
 * @param hmatrix Output hierarchical matrix
 * @return true on success, false on failure
 */
bool quantum_matrix_to_hierarchical(const float* matrix, int size, HierarchicalMatrix* hmatrix);

/**
 * @brief Optimize matrix decomposition using geometric features
 * 
 * Uses SVD analysis and geometric weighting for improved compression.
 * 
 * @param U Input/output U matrix
 * @param V Input/output V matrix
 * @param size Matrix dimension
 * @param tolerance Error tolerance for optimization
 * @return true on success, false on failure
 */
bool quantum_optimize_decomposition(float* U, float* V, int size, float tolerance);

/**
 * @brief Get string description of error code
 * 
 * @param error Error code
 * @return Constant string describing the error
 */
const char* quantum_matrix_get_error_string(quantum_matrix_error_t error);

#endif // QUANTUM_MATRIX_OPERATIONS_H
