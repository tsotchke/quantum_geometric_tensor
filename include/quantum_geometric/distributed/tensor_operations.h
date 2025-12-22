/**
 * @file tensor_operations.h
 * @brief Distributed tensor operations for quantum geometric computing
 *
 * This module provides efficient tensor operations including contraction,
 * symmetry handling, and storage optimization for large-scale quantum
 * geometric tensor computations.
 */

#ifndef TENSOR_OPERATIONS_H
#define TENSOR_OPERATIONS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Storage Format Types
// ============================================================================

typedef enum {
    DENSE_FORMAT,
    SPARSE_FORMAT,
    SYMMETRIC_FORMAT,
    BLOCK_FORMAT
} StorageFormat;

typedef enum {
    PERMUTATION,
    REFLECTION,
    ROTATION
} SymmetryType;

typedef enum {
    ADDITION,
    MULTIPLICATION,
    CONTRACTION,
    TRANSFORMATION
} TensorOperationType;

typedef enum {
    MATRIX_MULTIPLY,
    BLOCK_SPARSE,
    GENERAL
} ContractionPatternType;

// ============================================================================
// Core Data Structures
// ============================================================================

typedef struct {
    size_t* indices;
    size_t rank;
    size_t* dimensions;
} TensorIndex;

typedef struct {
    double* data;
    size_t size;
    TensorIndex index;
    bool is_zero;
} TensorBlock;

typedef struct {
    double* data;
    size_t size;
    StorageFormat format;
    size_t* dimensions;
    size_t rank;
    size_t num_elements;
    TensorBlock** blocks;
    size_t num_blocks;
    bool* symmetries;
    size_t num_symmetries;
} TensorStorage;

typedef struct {
    SymmetryType type;
    size_t* indices;
    size_t num_indices;
    double phase;
} SymmetryOperation;

typedef struct {
    ContractionPatternType type;
    size_t* contracted_dims;
    size_t num_contracted;
    size_t result_rank;
} ContractionPattern;

typedef struct {
    double sparsity;
    double symmetry_factor;
    bool is_block_sparse;
    size_t optimal_block_size;
} TensorAnalysis;

typedef struct {
    TensorOperationType type;
    TensorStorage** inputs;
    size_t num_inputs;
    TensorStorage* output;
    size_t* contraction_indices;
    size_t num_contraction_indices;
} TensorOperation;

typedef struct {
    size_t max_rank;
    size_t max_dimension;
    size_t block_size;
    bool use_symmetry;
    bool use_simd;
    size_t num_threads;
} TensorConfig;

struct SIMDOperations;
typedef struct SIMDOperations SIMDOperations;

typedef struct {
    TensorStorage* storage;
    double* workspace;
    size_t workspace_size;
    SIMDOperations* simd_ops;
    TensorConfig config;
} TensorOperations;

// ============================================================================
// Function Declarations
// ============================================================================

TensorOperations* init_tensor_operations(const TensorConfig* config);
void cleanup_tensor_operations(TensorOperations* ops);

void create_tensor(TensorOperations* ops, const size_t* dimensions,
                   size_t rank, StorageFormat format);
TensorStorage* create_tensor_storage(size_t max_rank, size_t max_dimension);
void cleanup_tensor_storage(TensorStorage* storage);
void allocate_tensor_storage(TensorStorage* storage, const size_t* dimensions,
                             size_t rank, StorageFormat format);
void initialize_tensor_structure(TensorStorage* storage);
void setup_tensor_blocks(TensorStorage* storage);

void contract_tensors(TensorOperations* ops, const TensorStorage* tensor1,
                      const TensorStorage* tensor2, const size_t* indices1,
                      const size_t* indices2, size_t num_indices,
                      TensorStorage* result);
ContractionPattern analyze_contraction(const TensorStorage* tensor1,
                                       const TensorStorage* tensor2,
                                       const size_t* indices1,
                                       const size_t* indices2,
                                       size_t num_indices);
void contract_as_matrix_multiply(TensorOperations* ops,
                                 const TensorStorage* tensor1,
                                 const TensorStorage* tensor2,
                                 ContractionPattern pattern,
                                 TensorStorage* result);
void contract_block_sparse(TensorOperations* ops, const TensorStorage* tensor1,
                           const TensorStorage* tensor2,
                           ContractionPattern pattern, TensorStorage* result);
void contract_general(TensorOperations* ops, const TensorStorage* tensor1,
                      const TensorStorage* tensor2, ContractionPattern pattern,
                      TensorStorage* result);

void apply_symmetry(TensorOperations* ops, TensorStorage* tensor,
                    const SymmetryOperation* symmetry);
bool check_symmetry(const TensorStorage* tensor,
                    const SymmetryOperation* symmetry);
void apply_permutation_symmetry(TensorOperations* ops, TensorStorage* tensor,
                                const SymmetryOperation* symmetry);
void apply_reflection_symmetry(TensorOperations* ops, TensorStorage* tensor,
                               const SymmetryOperation* symmetry);
void apply_rotation_symmetry(TensorOperations* ops, TensorStorage* tensor,
                             const SymmetryOperation* symmetry);
void update_symmetry_info(TensorStorage* tensor,
                          const SymmetryOperation* symmetry);

void optimize_storage(TensorOperations* ops, TensorStorage* tensor);
TensorAnalysis analyze_tensor_structure(const TensorStorage* tensor);
StorageFormat determine_optimal_format(TensorAnalysis analysis);
void convert_storage_format(TensorOperations* ops, TensorStorage* tensor,
                            StorageFormat new_format);
void optimize_block_structure(TensorOperations* ops, TensorStorage* tensor);

void perform_operation(TensorOperations* ops, const TensorOperation* operation);
void prepare_operands(TensorOperations* ops, const TensorOperation* operation);
void perform_tensor_addition(TensorOperations* ops,
                             const TensorOperation* operation);
void perform_tensor_multiplication(TensorOperations* ops,
                                   const TensorOperation* operation);
void perform_tensor_contraction(TensorOperations* ops,
                                const TensorOperation* operation);
void perform_tensor_transformation(TensorOperations* ops,
                                   const TensorOperation* operation);
void update_operation_result(TensorOperations* ops,
                             const TensorOperation* operation);

size_t compute_workspace_size(size_t max_rank, size_t max_dimension);

#ifdef __cplusplus
}
#endif

#endif // TENSOR_OPERATIONS_H
