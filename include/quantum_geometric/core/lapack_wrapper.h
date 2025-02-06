#ifndef LAPACK_WRAPPER_H
#define LAPACK_WRAPPER_H

#include "quantum_geometric/core/quantum_complex.h"
#include <stdbool.h>
#include <stddef.h>

// LAPACK operation status
typedef enum {
    LAPACK_SUCCESS = 0,
    LAPACK_INVALID_ARGUMENT = -1,
    LAPACK_MEMORY_ERROR = -2,
    LAPACK_SINGULAR_MATRIX = -3,
    LAPACK_NOT_CONVERGENT = -4,
    LAPACK_INTERNAL_ERROR = -5,
    LAPACK_NOT_IMPLEMENTED = -6
} lapack_status_t;

// LAPACK matrix layout
typedef enum {
    LAPACK_ROW_MAJOR = 101,
    LAPACK_COL_MAJOR = 102
} lapack_layout_t;

// LAPACK matrix operations
bool lapack_matrix_multiply(const ComplexFloat* a,
                          const ComplexFloat* b,
                          ComplexFloat* c,
                          size_t m, size_t k, size_t n,
                          bool transpose_a,
                          bool transpose_b,
                          lapack_layout_t layout);

bool lapack_svd(const ComplexFloat* a, size_t m, size_t n,
                ComplexFloat* u, float* s, ComplexFloat* vt,
                lapack_layout_t layout);

bool lapack_qr(const ComplexFloat* a, size_t m, size_t n,
               ComplexFloat* q, ComplexFloat* r,
               lapack_layout_t layout);

bool lapack_eigendecomposition(const ComplexFloat* a, size_t n,
                             ComplexFloat* eigenvectors,
                             ComplexFloat* eigenvalues,
                             lapack_layout_t layout);

bool lapack_cholesky(const ComplexFloat* a, size_t n,
                    ComplexFloat* l, bool lower_triangular,
                    lapack_layout_t layout);

bool lapack_lu(const ComplexFloat* a, size_t m, size_t n,
               ComplexFloat* l, ComplexFloat* u, int* ipiv,
               lapack_layout_t layout);

// LAPACK system solvers
bool lapack_solve_triangular(const ComplexFloat* a,
                           const ComplexFloat* b,
                           ComplexFloat* x,
                           size_t n, size_t nrhs,
                           bool upper_triangular,
                           bool unit_diagonal,
                           lapack_layout_t layout);

bool lapack_solve_symmetric(const ComplexFloat* a,
                          const ComplexFloat* b,
                          ComplexFloat* x,
                          size_t n, size_t nrhs,
                          bool positive_definite,
                          lapack_layout_t layout);

bool lapack_solve_general(const ComplexFloat* a,
                         const ComplexFloat* b,
                         ComplexFloat* x,
                         size_t n, size_t nrhs,
                         lapack_layout_t layout);

// LAPACK workspace management
size_t lapack_get_optimal_workspace(const ComplexFloat* a,
                                  size_t m, size_t n,
                                  const char* operation);

// LAPACK capability queries
bool lapack_has_capability(const char* capability);

// Error handling
lapack_status_t lapack_get_last_status(void);
const char* lapack_get_status_string(lapack_status_t status);

#endif // LAPACK_WRAPPER_H
