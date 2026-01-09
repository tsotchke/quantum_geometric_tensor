#include "quantum_geometric/distributed/tensor_operations.h"
#include "quantum_geometric/core/simd_operations.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

// Internal constants
#define ALIGNMENT 64
#define CACHE_LINE 64
#define MAX_TENSOR_RANK 16

// ============================================================================
// Storage Management Functions
// ============================================================================

TensorStorage* create_tensor_storage(size_t max_rank, size_t max_dimension) {
    TensorStorage* storage = aligned_alloc(ALIGNMENT, sizeof(TensorStorage));
    if (!storage) return NULL;

    memset(storage, 0, sizeof(TensorStorage));
    storage->dimensions = calloc(max_rank, sizeof(size_t));
    storage->format = DENSE_FORMAT;
    storage->rank = 0;
    storage->size = 0;
    storage->num_elements = 0;
    storage->blocks = NULL;
    storage->num_blocks = 0;
    storage->symmetries = NULL;
    storage->num_symmetries = 0;
    (void)max_dimension;

    return storage;
}

void cleanup_tensor_storage(TensorStorage* storage) {
    if (!storage) return;
    if (storage->data) free(storage->data);
    if (storage->dimensions) free(storage->dimensions);
    if (storage->blocks) {
        for (size_t i = 0; i < storage->num_blocks; i++) {
            if (storage->blocks[i]) {
                if (storage->blocks[i]->data) free(storage->blocks[i]->data);
                if (storage->blocks[i]->index.indices) free(storage->blocks[i]->index.indices);
                if (storage->blocks[i]->index.dimensions) free(storage->blocks[i]->index.dimensions);
                free(storage->blocks[i]);
            }
        }
        free(storage->blocks);
    }
    if (storage->symmetries) free(storage->symmetries);
    free(storage);
}

void allocate_tensor_storage(TensorStorage* storage, const size_t* dimensions,
                             size_t rank, StorageFormat format) {
    if (!storage || !dimensions || rank == 0) return;

    storage->rank = rank;
    storage->format = format;

    // Copy dimensions
    if (!storage->dimensions) {
        storage->dimensions = calloc(rank, sizeof(size_t));
    }
    memcpy(storage->dimensions, dimensions, rank * sizeof(size_t));

    // Compute total size
    size_t total_size = 1;
    for (size_t i = 0; i < rank; i++) {
        total_size *= dimensions[i];
    }
    storage->size = total_size;
    storage->num_elements = total_size;

    // Allocate data
    if (storage->data) free(storage->data);
    storage->data = aligned_alloc(ALIGNMENT, total_size * sizeof(double));
    if (storage->data) {
        memset(storage->data, 0, total_size * sizeof(double));
    }
}

void initialize_tensor_structure(TensorStorage* storage) {
    if (!storage || !storage->data) return;
    // Already initialized in allocate_tensor_storage
}

void setup_tensor_blocks(TensorStorage* storage) {
    if (!storage) return;

    // Default block size based on cache line
    size_t block_size = CACHE_LINE / sizeof(double);
    size_t num_blocks = (storage->num_elements + block_size - 1) / block_size;

    if (storage->blocks) {
        for (size_t i = 0; i < storage->num_blocks; i++) {
            if (storage->blocks[i]) {
                if (storage->blocks[i]->data) free(storage->blocks[i]->data);
                free(storage->blocks[i]);
            }
        }
        free(storage->blocks);
    }

    storage->blocks = calloc(num_blocks, sizeof(TensorBlock*));
    storage->num_blocks = num_blocks;

    for (size_t i = 0; i < num_blocks; i++) {
        storage->blocks[i] = calloc(1, sizeof(TensorBlock));
        storage->blocks[i]->size = block_size;
        storage->blocks[i]->is_zero = true;
    }
}

size_t compute_workspace_size(size_t max_rank, size_t max_dimension) {
    // Workspace for intermediate results
    size_t max_size = 1;
    for (size_t i = 0; i < max_rank && i < 4; i++) {
        max_size *= max_dimension;
    }
    return max_size * 2;
}

// ============================================================================
// Contraction Analysis and Execution
// ============================================================================

ContractionPattern analyze_contraction(const TensorStorage* tensor1,
                                        const TensorStorage* tensor2,
                                        const size_t* indices1,
                                        const size_t* indices2,
                                        size_t num_indices) {
    ContractionPattern pattern;
    memset(&pattern, 0, sizeof(pattern));

    if (!tensor1 || !tensor2) {
        pattern.type = GENERAL;
        return pattern;
    }

    // Check if this can be done as matrix multiply
    if (num_indices == 1 && indices1 && indices2 &&
        indices1[0] == tensor1->rank - 1 && indices2[0] == 0) {
        pattern.type = MATRIX_MULTIPLY;
    } else if (tensor1->format == BLOCK_FORMAT && tensor2->format == BLOCK_FORMAT) {
        pattern.type = BLOCK_SPARSE;
    } else {
        pattern.type = GENERAL;
    }

    pattern.num_contracted = num_indices;
    pattern.result_rank = tensor1->rank + tensor2->rank - 2 * num_indices;

    return pattern;
}

void contract_as_matrix_multiply(TensorOperations* ops,
                                  const TensorStorage* tensor1,
                                  const TensorStorage* tensor2,
                                  ContractionPattern pattern,
                                  TensorStorage* result) {
    if (!ops || !tensor1 || !tensor2 || !result) return;
    if (!tensor1->data || !tensor2->data || !result->data) return;

    // Determine dimensions for matrix multiply
    size_t K = tensor1->dimensions[tensor1->rank - 1];
    size_t M = tensor1->num_elements / K;
    size_t N = tensor2->num_elements / K;

    (void)pattern;

    // Use BLAS for matrix multiply
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)M, (int)N, (int)K,
                1.0, tensor1->data, (int)K,
                tensor2->data, (int)N,
                0.0, result->data, (int)N);
}

void contract_block_sparse(TensorOperations* ops,
                           const TensorStorage* tensor1,
                           const TensorStorage* tensor2,
                           ContractionPattern pattern,
                           TensorStorage* result) {
    if (!ops || !tensor1 || !tensor2 || !result) return;
    // Fall back to general for now
    contract_general(ops, tensor1, tensor2, pattern, result);
}

void contract_general(TensorOperations* ops,
                      const TensorStorage* tensor1,
                      const TensorStorage* tensor2,
                      ContractionPattern pattern,
                      TensorStorage* result) {
    if (!ops || !tensor1 || !tensor2 || !result) return;
    if (!tensor1->data || !tensor2->data || !result->data) return;
    (void)pattern;

    size_t n1 = tensor1->num_elements;
    size_t n2 = tensor2->num_elements;

    if (n1 == 0 || n2 == 0) return;

    // Simple contraction for now
    for (size_t i = 0; i < n1 && i < result->num_elements; i++) {
        result->data[i] = tensor1->data[i] * (n2 > 0 ? tensor2->data[0] : 0.0);
    }
}

// ============================================================================
// Symmetry Operations
// ============================================================================

bool check_symmetry(const TensorStorage* tensor, const SymmetryOperation* symmetry) {
    if (!tensor || !symmetry) return false;
    if (!tensor->symmetries || tensor->num_symmetries == 0) return false;
    (void)symmetry;
    return false;
}

void apply_permutation_symmetry(TensorOperations* ops, TensorStorage* tensor,
                                 const SymmetryOperation* symmetry) {
    if (!ops || !tensor || !symmetry) return;
    if (!tensor->data || tensor->num_elements == 0) return;
    if (!symmetry->indices || symmetry->num_indices == 0) return;
    if (symmetry->num_indices > tensor->rank) return;

    // For permutation symmetry, we swap indices according to the permutation
    // The permutation is specified by symmetry->indices
    // If indices = [1, 0, 2], it means swap axes 0 and 1

    // Calculate strides for each dimension
    size_t* strides = calloc(tensor->rank, sizeof(size_t));
    if (!strides) return;

    strides[tensor->rank - 1] = 1;
    for (size_t i = tensor->rank - 1; i > 0; i--) {
        strides[i - 1] = strides[i] * tensor->dimensions[i];
    }

    // Create workspace for permuted data
    double* workspace = ops->workspace;
    size_t workspace_needed = tensor->num_elements * sizeof(double);
    bool allocated_workspace = false;

    if (!workspace || ops->workspace_size < workspace_needed) {
        workspace = malloc(workspace_needed);
        if (!workspace) {
            free(strides);
            return;
        }
        allocated_workspace = true;
    }
    memset(workspace, 0, workspace_needed);

    // Permute the tensor data
    size_t* coords = calloc(tensor->rank, sizeof(size_t));
    size_t* perm_coords = calloc(tensor->rank, sizeof(size_t));
    if (!coords || !perm_coords) {
        free(strides);
        free(coords);
        free(perm_coords);
        if (allocated_workspace) free(workspace);
        return;
    }

    for (size_t i = 0; i < tensor->num_elements; i++) {
        // Convert linear index to multi-index
        size_t temp = i;
        for (size_t d = 0; d < tensor->rank; d++) {
            coords[d] = temp / strides[d];
            temp %= strides[d];
        }

        // Apply permutation
        for (size_t d = 0; d < symmetry->num_indices && d < tensor->rank; d++) {
            size_t perm_idx = symmetry->indices[d];
            if (perm_idx < tensor->rank) {
                perm_coords[d] = coords[perm_idx];
            } else {
                perm_coords[d] = coords[d];
            }
        }
        // Keep remaining dimensions unchanged
        for (size_t d = symmetry->num_indices; d < tensor->rank; d++) {
            perm_coords[d] = coords[d];
        }

        // Convert permuted multi-index back to linear index
        size_t perm_idx = 0;
        for (size_t d = 0; d < tensor->rank; d++) {
            perm_idx += perm_coords[d] * strides[d];
        }

        // Apply phase factor if specified
        double phase_factor = (symmetry->phase != 0.0) ? cos(symmetry->phase) : 1.0;
        workspace[perm_idx] = tensor->data[i] * phase_factor;
    }

    // Copy back to tensor
    memcpy(tensor->data, workspace, tensor->num_elements * sizeof(double));

    free(strides);
    free(coords);
    free(perm_coords);
    if (allocated_workspace) free(workspace);

    update_symmetry_info(tensor, symmetry);
}

void apply_reflection_symmetry(TensorOperations* ops, TensorStorage* tensor,
                                const SymmetryOperation* symmetry) {
    if (!ops || !tensor || !symmetry) return;
    if (!tensor->data || tensor->num_elements == 0) return;
    if (!symmetry->indices || symmetry->num_indices == 0) return;

    // For reflection symmetry, we reverse the order along specified axes
    // The indices array specifies which axes to reflect

    // Calculate strides for each dimension
    size_t* strides = calloc(tensor->rank, sizeof(size_t));
    if (!strides) return;

    strides[tensor->rank - 1] = 1;
    for (size_t i = tensor->rank - 1; i > 0; i--) {
        strides[i - 1] = strides[i] * tensor->dimensions[i];
    }

    // Create a boolean mask for reflected dimensions
    bool* reflect_dim = calloc(tensor->rank, sizeof(bool));
    if (!reflect_dim) {
        free(strides);
        return;
    }
    for (size_t i = 0; i < symmetry->num_indices; i++) {
        if (symmetry->indices[i] < tensor->rank) {
            reflect_dim[symmetry->indices[i]] = true;
        }
    }

    // Create workspace for reflected data
    double* workspace = ops->workspace;
    size_t workspace_needed = tensor->num_elements * sizeof(double);
    bool allocated_workspace = false;

    if (!workspace || ops->workspace_size < workspace_needed) {
        workspace = malloc(workspace_needed);
        if (!workspace) {
            free(strides);
            free(reflect_dim);
            return;
        }
        allocated_workspace = true;
    }

    // Reflect the tensor data
    size_t* coords = calloc(tensor->rank, sizeof(size_t));
    if (!coords) {
        free(strides);
        free(reflect_dim);
        if (allocated_workspace) free(workspace);
        return;
    }

    for (size_t i = 0; i < tensor->num_elements; i++) {
        // Convert linear index to multi-index
        size_t temp = i;
        for (size_t d = 0; d < tensor->rank; d++) {
            coords[d] = temp / strides[d];
            temp %= strides[d];
        }

        // Apply reflection: reverse index along reflected dimensions
        size_t reflected_idx = 0;
        for (size_t d = 0; d < tensor->rank; d++) {
            size_t coord = coords[d];
            if (reflect_dim[d]) {
                coord = tensor->dimensions[d] - 1 - coord;
            }
            reflected_idx += coord * strides[d];
        }

        // Apply phase factor if specified
        double phase_factor = (symmetry->phase != 0.0) ? cos(symmetry->phase) : 1.0;
        workspace[reflected_idx] = tensor->data[i] * phase_factor;
    }

    // Copy back to tensor
    memcpy(tensor->data, workspace, tensor->num_elements * sizeof(double));

    free(strides);
    free(reflect_dim);
    free(coords);
    if (allocated_workspace) free(workspace);

    update_symmetry_info(tensor, symmetry);
}

void apply_rotation_symmetry(TensorOperations* ops, TensorStorage* tensor,
                              const SymmetryOperation* symmetry) {
    if (!ops || !tensor || !symmetry) return;
    if (!tensor->data || tensor->num_elements == 0) return;
    if (symmetry->num_indices < 2) return;  // Need at least 2 axes for rotation

    // For rotation symmetry, we perform a cyclic permutation of specified axes
    // The rotation is in the plane defined by the first two indices in symmetry->indices
    // The phase determines the rotation angle: phase = 2*pi*k/n for n-fold rotation

    // For discrete rotations, we apply a cyclic shift
    // For continuous rotations, we interpolate (not fully supported here)

    // Calculate strides
    size_t* strides = calloc(tensor->rank, sizeof(size_t));
    if (!strides) return;

    strides[tensor->rank - 1] = 1;
    for (size_t i = tensor->rank - 1; i > 0; i--) {
        strides[i - 1] = strides[i] * tensor->dimensions[i];
    }

    // Get rotation axes
    size_t axis1 = symmetry->indices[0];
    size_t axis2 = symmetry->indices[1];
    if (axis1 >= tensor->rank || axis2 >= tensor->rank) {
        free(strides);
        return;
    }

    // Create workspace
    double* workspace = ops->workspace;
    size_t workspace_needed = tensor->num_elements * sizeof(double);
    bool allocated_workspace = false;

    if (!workspace || ops->workspace_size < workspace_needed) {
        workspace = malloc(workspace_needed);
        if (!workspace) {
            free(strides);
            return;
        }
        allocated_workspace = true;
    }
    memset(workspace, 0, workspace_needed);

    // Rotation parameters
    double angle = symmetry->phase;  // Rotation angle in radians
    double cos_a = cos(angle);
    double sin_a = sin(angle);

    // Apply rotation in the (axis1, axis2) plane
    size_t* coords = calloc(tensor->rank, sizeof(size_t));
    if (!coords) {
        free(strides);
        if (allocated_workspace) free(workspace);
        return;
    }

    // For discrete tensors, we implement a 90-degree rotation (swap + negate)
    // For the general case, we'd need to interpolate between grid points
    bool is_90_deg = (fabs(fabs(angle) - M_PI / 2.0) < 0.01);
    bool is_180_deg = (fabs(fabs(angle) - M_PI) < 0.01);

    for (size_t i = 0; i < tensor->num_elements; i++) {
        // Convert linear index to multi-index
        size_t temp = i;
        for (size_t d = 0; d < tensor->rank; d++) {
            coords[d] = temp / strides[d];
            temp %= strides[d];
        }

        size_t new_coord1, new_coord2;
        size_t dim1 = tensor->dimensions[axis1];
        size_t dim2 = tensor->dimensions[axis2];

        if (is_90_deg) {
            // 90-degree rotation: (x, y) -> (-y, x), mapped to grid
            new_coord1 = coords[axis2];
            new_coord2 = dim2 - 1 - coords[axis1];
        } else if (is_180_deg) {
            // 180-degree rotation: (x, y) -> (-x, -y)
            new_coord1 = dim1 - 1 - coords[axis1];
            new_coord2 = dim2 - 1 - coords[axis2];
        } else {
            // General rotation: round to nearest grid point
            double x = (double)coords[axis1] - (double)(dim1 - 1) / 2.0;
            double y = (double)coords[axis2] - (double)(dim2 - 1) / 2.0;
            double new_x = cos_a * x - sin_a * y;
            double new_y = sin_a * x + cos_a * y;
            new_x += (double)(dim1 - 1) / 2.0;
            new_y += (double)(dim2 - 1) / 2.0;

            // Round and clamp to valid range
            new_coord1 = (new_x < 0) ? 0 : ((new_x >= dim1) ? dim1 - 1 : (size_t)(new_x + 0.5));
            new_coord2 = (new_y < 0) ? 0 : ((new_y >= dim2) ? dim2 - 1 : (size_t)(new_y + 0.5));
        }

        // Build rotated index
        size_t rotated_idx = 0;
        for (size_t d = 0; d < tensor->rank; d++) {
            if (d == axis1) {
                rotated_idx += new_coord1 * strides[d];
            } else if (d == axis2) {
                rotated_idx += new_coord2 * strides[d];
            } else {
                rotated_idx += coords[d] * strides[d];
            }
        }

        workspace[rotated_idx] = tensor->data[i];
    }

    // Copy back to tensor
    memcpy(tensor->data, workspace, tensor->num_elements * sizeof(double));

    free(strides);
    free(coords);
    if (allocated_workspace) free(workspace);

    update_symmetry_info(tensor, symmetry);
}

void update_symmetry_info(TensorStorage* tensor, const SymmetryOperation* symmetry) {
    if (!tensor || !symmetry) return;
    // Add symmetry to list
    size_t new_count = tensor->num_symmetries + 1;
    bool* new_symmetries = realloc(tensor->symmetries, new_count * sizeof(bool));
    if (new_symmetries) {
        tensor->symmetries = new_symmetries;
        tensor->symmetries[tensor->num_symmetries] = true;
        tensor->num_symmetries = new_count;
    }
}

// ============================================================================
// Storage Optimization
// ============================================================================

TensorAnalysis analyze_tensor_structure(const TensorStorage* tensor) {
    TensorAnalysis analysis;
    memset(&analysis, 0, sizeof(analysis));

    if (!tensor || !tensor->data) return analysis;

    // Count non-zeros and compute sparsity
    size_t nonzeros = 0;
    for (size_t i = 0; i < tensor->num_elements; i++) {
        if (fabs(tensor->data[i]) > 1e-15) {
            nonzeros++;
        }
    }

    analysis.sparsity = 1.0 - (double)nonzeros / (double)(tensor->num_elements > 0 ? tensor->num_elements : 1);
    analysis.symmetry_factor = tensor->num_symmetries > 0 ? 1.0 / tensor->num_symmetries : 1.0;
    analysis.is_block_sparse = (analysis.sparsity > 0.5);
    analysis.optimal_block_size = CACHE_LINE / sizeof(double);

    return analysis;
}

StorageFormat determine_optimal_format(TensorAnalysis analysis) {
    if (analysis.is_block_sparse) {
        return BLOCK_FORMAT;
    }
    if (analysis.sparsity > 0.8) {
        return SPARSE_FORMAT;
    }
    return DENSE_FORMAT;
}

void convert_storage_format(TensorOperations* ops, TensorStorage* tensor,
                             StorageFormat new_format) {
    if (!ops || !tensor) return;
    if (tensor->format == new_format) return;

    tensor->format = new_format;

    if (new_format == BLOCK_FORMAT) {
        setup_tensor_blocks(tensor);
    }
}

void optimize_block_structure(TensorOperations* ops, TensorStorage* tensor) {
    if (!ops || !tensor) return;
    // Optimize block layout for cache efficiency
}

// ============================================================================
// Tensor Operations
// ============================================================================

void prepare_operands(TensorOperations* ops, const TensorOperation* operation) {
    if (!ops || !operation) return;
    // Prepare operands for the operation
}

void perform_tensor_addition(TensorOperations* ops, const TensorOperation* operation) {
    if (!ops || !operation) return;
    if (!operation->inputs || operation->num_inputs < 2 || !operation->output) return;

    TensorStorage* op1 = operation->inputs[0];
    TensorStorage* op2 = operation->inputs[1];
    TensorStorage* result = operation->output;

    if (!op1 || !op2 || !result) return;
    if (!op1->data || !op2->data || !result->data) return;

    size_t n = op1->num_elements;
    if (n != op2->num_elements) return;

    memcpy(result->data, op2->data, n * sizeof(double));
    cblas_daxpy((int)n, 1.0, op1->data, 1, result->data, 1);
}

void perform_tensor_multiplication(TensorOperations* ops, const TensorOperation* operation) {
    if (!ops || !operation) return;
    if (!operation->inputs || operation->num_inputs < 2 || !operation->output) return;

    TensorStorage* op1 = operation->inputs[0];
    TensorStorage* op2 = operation->inputs[1];
    TensorStorage* result = operation->output;

    if (!op1 || !op2 || !result) return;
    if (!op1->data || !op2->data || !result->data) return;

    size_t n = op1->num_elements;
    for (size_t i = 0; i < n; i++) {
        result->data[i] = op1->data[i] * op2->data[i];
    }
}

void perform_tensor_contraction(TensorOperations* ops, const TensorOperation* operation) {
    if (!ops || !operation) return;
    if (!operation->inputs || operation->num_inputs < 2 || !operation->output) return;

    contract_tensors(ops, operation->inputs[0], operation->inputs[1],
                     operation->contraction_indices, operation->contraction_indices,
                     operation->num_contraction_indices, operation->output);
}

void perform_tensor_transformation(TensorOperations* ops, const TensorOperation* operation) {
    if (!ops || !operation) return;
    if (!operation->inputs || operation->num_inputs < 1 || !operation->output) return;

    TensorStorage* input = operation->inputs[0];
    TensorStorage* output = operation->output;

    if (!input || !output) return;
    if (!input->data || !output->data) return;

    memcpy(output->data, input->data, input->num_elements * sizeof(double));
}

void update_operation_result(TensorOperations* ops, const TensorOperation* operation) {
    if (!ops || !operation) return;
    // Update any metadata after operation
}

// ============================================================================
// Main API Functions
// ============================================================================

TensorOperations* init_tensor_operations(const TensorConfig* config) {
    TensorOperations* ops = aligned_alloc(ALIGNMENT, sizeof(TensorOperations));
    if (!ops) return NULL;

    memset(ops, 0, sizeof(TensorOperations));

    // Initialize storage
    ops->storage = create_tensor_storage(config->max_rank, config->max_dimension);

    // Initialize workspace
    size_t workspace_size = compute_workspace_size(config->max_rank, config->max_dimension);
    ops->workspace = aligned_alloc(ALIGNMENT, workspace_size * sizeof(double));
    ops->workspace_size = workspace_size;

    // SIMD operations are stateless function-based
    ops->simd_ops = NULL;

    // Store configuration
    ops->config = *config;

    return ops;
}

void create_tensor(TensorOperations* ops, const size_t* dimensions,
                   size_t rank, StorageFormat format) {
    if (!ops) return;

    allocate_tensor_storage(ops->storage, dimensions, rank, format);
    initialize_tensor_structure(ops->storage);

    if (format == BLOCK_FORMAT) {
        setup_tensor_blocks(ops->storage);
    }
}

void contract_tensors(TensorOperations* ops,
                      const TensorStorage* tensor1,
                      const TensorStorage* tensor2,
                      const size_t* indices1,
                      const size_t* indices2,
                      size_t num_indices,
                      TensorStorage* result) {
    if (!ops || !tensor1 || !tensor2 || !result) return;

    ContractionPattern pattern = analyze_contraction(tensor1, tensor2, indices1, indices2, num_indices);

    switch (pattern.type) {
        case MATRIX_MULTIPLY:
            contract_as_matrix_multiply(ops, tensor1, tensor2, pattern, result);
            break;
        case BLOCK_SPARSE:
            contract_block_sparse(ops, tensor1, tensor2, pattern, result);
            break;
        case GENERAL:
        default:
            contract_general(ops, tensor1, tensor2, pattern, result);
            break;
    }
}

void apply_symmetry(TensorOperations* ops, TensorStorage* tensor,
                    const SymmetryOperation* symmetry) {
    if (!ops || !tensor || !symmetry) return;

    if (check_symmetry(tensor, symmetry)) return;

    switch (symmetry->type) {
        case PERMUTATION:
            apply_permutation_symmetry(ops, tensor, symmetry);
            break;
        case REFLECTION:
            apply_reflection_symmetry(ops, tensor, symmetry);
            break;
        case ROTATION:
            apply_rotation_symmetry(ops, tensor, symmetry);
            break;
    }

    update_symmetry_info(tensor, symmetry);
}

void optimize_storage(TensorOperations* ops, TensorStorage* tensor) {
    if (!ops || !tensor) return;

    TensorAnalysis analysis = analyze_tensor_structure(tensor);
    StorageFormat optimal_format = determine_optimal_format(analysis);

    if (optimal_format != tensor->format) {
        convert_storage_format(ops, tensor, optimal_format);
    }

    if (optimal_format == BLOCK_FORMAT) {
        optimize_block_structure(ops, tensor);
    }
}

void perform_operation(TensorOperations* ops, const TensorOperation* operation) {
    if (!ops || !operation) return;

    prepare_operands(ops, operation);

    switch (operation->type) {
        case ADDITION:
            perform_tensor_addition(ops, operation);
            break;
        case MULTIPLICATION:
            perform_tensor_multiplication(ops, operation);
            break;
        case CONTRACTION:
            perform_tensor_contraction(ops, operation);
            break;
        case TRANSFORMATION:
            perform_tensor_transformation(ops, operation);
            break;
    }

    update_operation_result(ops, operation);
}

void cleanup_tensor_operations(TensorOperations* ops) {
    if (!ops) return;

    cleanup_tensor_storage(ops->storage);
    if (ops->workspace) free(ops->workspace);

    free(ops);
}
