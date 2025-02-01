#ifndef HIERARCHICAL_TENSOR_H
#define HIERARCHICAL_TENSOR_H

#include <stddef.h>
#include <complex.h>
#include <stdbool.h>
#include "quantum_geometric/core/quantum_types.h"
#include "quantum_geometric/core/error_codes.h"
#include "quantum_geometric/core/hierarchical_matrix.h"

// Tensor types
typedef enum {
    TENSOR_DENSE,          // Dense tensor
    TENSOR_SPARSE,         // Sparse tensor
    TENSOR_HIERARCHICAL,   // Hierarchical tensor
    TENSOR_QUANTUM,        // Quantum tensor
    TENSOR_HYBRID         // Hybrid tensor
} tensor_type_t;

// Decomposition types
typedef enum {
    DECOMP_CP,            // CANDECOMP/PARAFAC
    DECOMP_TUCKER,        // Tucker decomposition
    DECOMP_TT,            // Tensor train
    DECOMP_QUANTUM,       // Quantum decomposition
    DECOMP_HYBRID        // Hybrid decomposition
} decomposition_type_t;

// Tensor configuration
typedef struct {
    tensor_type_t type;            // Tensor type
    size_t* dimensions;            // Tensor dimensions
    size_t num_dimensions;         // Number of dimensions
    size_t block_size;             // Block size
    double tolerance;              // Error tolerance
    bool use_compression;          // Enable compression
    void* config_data;           // Additional config
} tensor_config_t;

// Decomposition parameters
typedef struct {
    decomposition_type_t type;     // Decomposition type
    size_t max_rank;               // Maximum rank
    double tolerance;              // Tolerance
    bool truncate;                 // Enable truncation
    size_t max_iterations;         // Maximum iterations
    void* decomp_data;           // Additional parameters
} decomposition_params_t;

// Core functions
HierarchicalMatrix* create_hierarchical_tensor(const tensor_config_t* config);
void destroy_hierarchical_tensor(HierarchicalMatrix* tensor);

// Initialization functions
qgt_error_t init_tensor_structure(HierarchicalMatrix* tensor,
                                const tensor_config_t* config);
qgt_error_t set_tensor_dimensions(HierarchicalMatrix* tensor,
                                const size_t* dimensions,
                                size_t num_dimensions);
qgt_error_t validate_tensor(const HierarchicalMatrix* tensor);

// Data conversion
qgt_error_t convert_to_hierarchical(HierarchicalMatrix* dest,
                                  const void* src,
                                  const tensor_config_t* config);
qgt_error_t convert_from_hierarchical(void* dest,
                                    const HierarchicalMatrix* src,
                                    size_t size);
qgt_error_t convert_between_formats(HierarchicalMatrix* dest,
                                  const HierarchicalMatrix* src,
                                  tensor_type_t target_type);

// Basic operations
qgt_error_t tensor_add(HierarchicalMatrix* dest,
                      const HierarchicalMatrix* a,
                      const HierarchicalMatrix* b);
qgt_error_t tensor_multiply(HierarchicalMatrix* dest,
                          const HierarchicalMatrix* a,
                          const HierarchicalMatrix* b);
qgt_error_t tensor_scale(HierarchicalMatrix* tensor,
                        double complex_scalar);
qgt_error_t tensor_contract(HierarchicalMatrix* dest,
                          const HierarchicalMatrix* a,
                          const HierarchicalMatrix* b,
                          size_t* indices,
                          size_t num_indices);

// Decomposition operations
qgt_error_t compute_cp_decomposition(const HierarchicalMatrix* tensor,
                                   HierarchicalMatrix** factors,
                                   size_t* rank,
                                   const decomposition_params_t* params);
qgt_error_t compute_tucker_decomposition(const HierarchicalMatrix* tensor,
                                       HierarchicalMatrix* core,
                                       HierarchicalMatrix** factors,
                                       const decomposition_params_t* params);
qgt_error_t compute_tt_decomposition(const HierarchicalMatrix* tensor,
                                   HierarchicalMatrix** cores,
                                   const decomposition_params_t* params);

// Advanced operations
qgt_error_t tensor_svd(HierarchicalMatrix* u,
                      double* s,
                      HierarchicalMatrix* vt,
                      const HierarchicalMatrix* tensor);
qgt_error_t tensor_inverse(HierarchicalMatrix* dest,
                          const HierarchicalMatrix* src);
qgt_error_t tensor_transpose(HierarchicalMatrix* dest,
                           const HierarchicalMatrix* src,
                           const size_t* perm);

// Quantum operations
qgt_error_t quantum_tensor_product(HierarchicalMatrix* dest,
                                 const HierarchicalMatrix* a,
                                 const HierarchicalMatrix* b);
qgt_error_t quantum_decomposition(const HierarchicalMatrix* tensor,
                                HierarchicalMatrix** factors,
                                const decomposition_params_t* params);
qgt_error_t quantum_contraction(HierarchicalMatrix* dest,
                              const HierarchicalMatrix* tensor,
                              const size_t* indices);

// Analysis functions
qgt_error_t compute_tensor_norm(const HierarchicalMatrix* tensor,
                              double* norm);
qgt_error_t estimate_tensor_rank(const HierarchicalMatrix* tensor,
                               size_t* rank);
qgt_error_t analyze_sparsity(const HierarchicalMatrix* tensor,
                            double* sparsity);

// Utility functions
qgt_error_t export_tensor(const HierarchicalMatrix* tensor,
                         const char* filename);
qgt_error_t import_tensor(HierarchicalMatrix* tensor,
                         const char* filename);
void print_tensor(const HierarchicalMatrix* tensor);

// Validation functions
bool is_unitary(const HierarchicalMatrix* tensor,
                double tolerance);
bool is_hermitian(const HierarchicalMatrix* tensor,
                 double tolerance);
bool is_positive_definite(const HierarchicalMatrix* tensor,
                         double tolerance);

#endif // HIERARCHICAL_TENSOR_H
