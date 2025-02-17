#ifndef QUANTUM_GEOMETRIC_NUMERICAL_BACKEND_H
#define QUANTUM_GEOMETRIC_NUMERICAL_BACKEND_H

#include "quantum_geometric/core/quantum_complex.h"
#include <stddef.h>
#include <stdbool.h>

// Error handling
typedef enum {
    NUMERICAL_SUCCESS,
    NUMERICAL_ERROR_INVALID_ARGUMENT,
    NUMERICAL_ERROR_MEMORY,
    NUMERICAL_ERROR_BACKEND,
    NUMERICAL_ERROR_COMPUTATION,
    NUMERICAL_ERROR_NOT_IMPLEMENTED,
    NUMERICAL_ERROR_INVALID_STATE
} numerical_error_t;

numerical_error_t get_last_numerical_error(void);
const char* get_numerical_error_string(numerical_error_t error);

// Backend types
typedef enum {
    NUMERICAL_BACKEND_CPU,      // CPU-based computations
    NUMERICAL_BACKEND_ACCELERATE, // Apple Accelerate framework
    NUMERICAL_BACKEND_OPENBLAS,  // OpenBLAS
    NUMERICAL_BACKEND_MKL,      // Intel MKL
    NUMERICAL_BACKEND_CUDA,     // NVIDIA CUDA
    NUMERICAL_BACKEND_METAL     // Apple Metal
} numerical_backend_t;

// Backend configuration
typedef struct {
    numerical_backend_t type;
    size_t max_threads;
    bool use_fma;              // Use fused multiply-add
    bool use_avx;              // Use AVX instructions
    bool use_neon;             // Use NEON instructions
    size_t cache_size;         // Cache size in bytes
    void* backend_specific;    // Backend-specific configuration
} numerical_config_t;

// Matrix operations
numerical_error_t numerical_matrix_multiply(const ComplexFloat* a,
                                          const ComplexFloat* b,
                                          ComplexFloat* c,
                                          size_t m, size_t k, size_t n,
                                          bool transpose_a,
                                          bool transpose_b);

numerical_error_t numerical_matrix_add(const ComplexFloat* a,
                                     const ComplexFloat* b,
                                     ComplexFloat* c,
                                     size_t rows,
                                     size_t cols);

numerical_error_t numerical_matrix_subtract(const ComplexFloat* a,
                                          const ComplexFloat* b,
                                          ComplexFloat* c,
                                          size_t rows,
                                          size_t cols);

// Vector operations
numerical_error_t numerical_vector_add(const ComplexFloat* a,
                                     const ComplexFloat* b,
                                     ComplexFloat* c,
                                     size_t length);

numerical_error_t numerical_vector_scale(const ComplexFloat* a,
                                       ComplexFloat scale,
                                       ComplexFloat* c,
                                       size_t length);

numerical_error_t numerical_vector_dot(const ComplexFloat* a,
                                     const ComplexFloat* b,
                                     ComplexFloat* result,
                                     size_t length);

// Matrix decompositions
numerical_error_t numerical_svd(const ComplexFloat* a,
                               ComplexFloat* u,
                               float* s,
                               ComplexFloat* vt,
                               size_t m,
                               size_t n);

numerical_error_t numerical_qr(const ComplexFloat* a,
                              ComplexFloat* q,
                              ComplexFloat* r,
                              size_t m,
                              size_t n);

numerical_error_t numerical_eigendecomposition(const ComplexFloat* a,
                                             ComplexFloat* eigenvectors,
                                             ComplexFloat* eigenvalues,
                                             size_t n);

numerical_error_t numerical_cholesky(const ComplexFloat* a,
                                    ComplexFloat* l,
                                    size_t n,
                                    bool lower_triangular);

numerical_error_t numerical_lu(const ComplexFloat* a,
                              ComplexFloat* l,
                              ComplexFloat* u,
                              int* ipiv,
                              size_t m,
                              size_t n);

numerical_error_t numerical_solve_triangular(const ComplexFloat* a,
                                           const ComplexFloat* b,
                                           ComplexFloat* x,
                                           size_t n,
                                           size_t nrhs,
                                           bool upper_triangular,
                                           bool unit_diagonal);

numerical_error_t numerical_solve_symmetric(const ComplexFloat* a,
                                          const ComplexFloat* b,
                                          ComplexFloat* x,
                                          size_t n,
                                          size_t nrhs,
                                          bool positive_definite);

numerical_error_t numerical_solve_general(const ComplexFloat* a,
                                        const ComplexFloat* b,
                                        ComplexFloat* x,
                                        size_t n,
                                        size_t nrhs);

// Tensor operations
numerical_error_t numerical_tensor_contract(const ComplexFloat* a,
                                          const ComplexFloat* b,
                                          ComplexFloat* c,
                                          const size_t* dims_a,
                                          const size_t* dims_b,
                                          const size_t* dims_c,
                                          const size_t* contract_a,
                                          const size_t* contract_b,
                                          size_t num_dims_a,
                                          size_t num_dims_b,
                                          size_t num_dims_c,
                                          size_t num_contract);

// Backend management
numerical_error_t initialize_numerical_backend(const numerical_config_t* config);
void shutdown_numerical_backend(void);
numerical_error_t get_optimal_backend(numerical_config_t* config);
const char* get_backend_name(numerical_backend_t backend);
numerical_error_t is_backend_available(numerical_backend_t backend);

// Performance monitoring
typedef struct {
    double flops;              // Floating point operations per second
    double memory_bandwidth;   // Memory bandwidth in bytes/second
    double cache_hits;         // Cache hit rate
    double utilization;        // Resource utilization
    size_t peak_memory;       // Peak memory usage
} numerical_metrics_t;

numerical_error_t get_numerical_metrics(numerical_metrics_t* metrics);
numerical_error_t reset_numerical_metrics(void);

// Backend-specific function declarations
#ifdef __APPLE__
numerical_error_t numerical_matrix_multiply_accelerate(const ComplexFloat* a,
                                                const ComplexFloat* b,
                                                ComplexFloat* c,
                                                size_t m, size_t k, size_t n,
                                                bool transpose_a,
                                                bool transpose_b);

numerical_error_t numerical_matrix_add_accelerate(const ComplexFloat* a,
                                           const ComplexFloat* b,
                                           ComplexFloat* c,
                                           size_t rows,
                                           size_t cols);

numerical_error_t numerical_vector_dot_accelerate(const ComplexFloat* a,
                                           const ComplexFloat* b,
                                           ComplexFloat* result,
                                           size_t length);

numerical_error_t numerical_svd_accelerate(const ComplexFloat* a,
                                    ComplexFloat* u,
                                    float* s,
                                    ComplexFloat* vt,
                                    size_t m,
                                    size_t n);

numerical_error_t numerical_qr_accelerate(const ComplexFloat* a,
                                   ComplexFloat* q,
                                   ComplexFloat* r,
                                   size_t m,
                                   size_t n);

numerical_error_t numerical_eigendecomposition_accelerate(const ComplexFloat* a,
                                                   ComplexFloat* eigenvectors,
                                                   ComplexFloat* eigenvalues,
                                                   size_t n);

numerical_error_t numerical_cholesky_accelerate(const ComplexFloat* a,
                                         ComplexFloat* l,
                                         size_t n,
                                         bool lower_triangular);

numerical_error_t numerical_lu_accelerate(const ComplexFloat* a,
                                   ComplexFloat* l,
                                   ComplexFloat* u,
                                   int* ipiv,
                                   size_t m,
                                   size_t n);

numerical_error_t numerical_solve_triangular_accelerate(const ComplexFloat* a,
                                                 const ComplexFloat* b,
                                                 ComplexFloat* x,
                                                 size_t n,
                                                 size_t nrhs,
                                                 bool upper_triangular,
                                                 bool unit_diagonal);

numerical_error_t numerical_solve_symmetric_accelerate(const ComplexFloat* a,
                                               const ComplexFloat* b,
                                               ComplexFloat* x,
                                               size_t n,
                                               size_t nrhs,
                                               bool positive_definite);

numerical_error_t numerical_solve_general_accelerate(const ComplexFloat* a,
                                             const ComplexFloat* b,
                                             ComplexFloat* x,
                                             size_t n,
                                             size_t nrhs);

numerical_error_t get_numerical_metrics_accelerate(numerical_metrics_t* metrics);
numerical_error_t reset_numerical_metrics_accelerate(void);
numerical_error_t get_last_numerical_error_accelerate(void);
#endif // __APPLE__

// CPU backend function declarations
numerical_error_t numerical_matrix_multiply_cpu(const ComplexFloat* a,
                                         const ComplexFloat* b,
                                         ComplexFloat* c,
                                         size_t m, size_t k, size_t n,
                                         bool transpose_a,
                                         bool transpose_b);

numerical_error_t numerical_matrix_add_cpu(const ComplexFloat* a,
                                    const ComplexFloat* b,
                                    ComplexFloat* c,
                                    size_t rows,
                                    size_t cols);

numerical_error_t numerical_vector_dot_cpu(const ComplexFloat* a,
                                    const ComplexFloat* b,
                                    ComplexFloat* result,
                                    size_t length);

numerical_error_t numerical_svd_cpu(const ComplexFloat* a,
                             ComplexFloat* u,
                             float* s,
                             ComplexFloat* vt,
                             size_t m,
                             size_t n);

numerical_error_t numerical_qr_cpu(const ComplexFloat* a,
                            ComplexFloat* q,
                            ComplexFloat* r,
                            size_t m,
                            size_t n);

numerical_error_t numerical_eigendecomposition_cpu(const ComplexFloat* a,
                                           ComplexFloat* eigenvectors,
                                           ComplexFloat* eigenvalues,
                                           size_t n);

numerical_error_t numerical_cholesky_cpu(const ComplexFloat* a,
                                  ComplexFloat* l,
                                  size_t n,
                                  bool lower_triangular);

numerical_error_t numerical_lu_cpu(const ComplexFloat* a,
                            ComplexFloat* l,
                            ComplexFloat* u,
                            int* ipiv,
                            size_t m,
                            size_t n);

numerical_error_t numerical_solve_triangular_cpu(const ComplexFloat* a,
                                         const ComplexFloat* b,
                                         ComplexFloat* x,
                                         size_t n,
                                         size_t nrhs,
                                         bool upper_triangular,
                                         bool unit_diagonal);

numerical_error_t numerical_solve_symmetric_cpu(const ComplexFloat* a,
                                        const ComplexFloat* b,
                                        ComplexFloat* x,
                                        size_t n,
                                        size_t nrhs,
                                        bool positive_definite);

numerical_error_t numerical_solve_general_cpu(const ComplexFloat* a,
                                      const ComplexFloat* b,
                                      ComplexFloat* x,
                                      size_t n,
                                      size_t nrhs);

numerical_error_t get_numerical_metrics_cpu(numerical_metrics_t* metrics);
numerical_error_t reset_numerical_metrics_cpu(void);
numerical_error_t get_last_numerical_error_cpu(void);

#endif // QUANTUM_GEOMETRIC_NUMERICAL_BACKEND_H
