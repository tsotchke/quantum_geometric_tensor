#include "quantum_geometric/core/tensor_network_operations.h"
#include "quantum_geometric/core/hierarchical_matrix.h"
#include "quantum_geometric/core/simd_operations.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

// Tree tensor network node structure
struct TreeTensorNetwork {
    HierarchicalMatrix* tensor;     // Node tensor data
    TreeTensorNetwork* left;        // Left child
    TreeTensorNetwork* right;       // Right child
    size_t input_dim;              // Input dimension
    size_t output_dim;             // Output dimension
    size_t bond_dim;               // Bond dimension
    double tolerance;              // SVD truncation tolerance
    bool is_leaf;                  // Whether this is a leaf node
    TTNConfig config;              // Configuration parameters
};

// Maximum tensor rank for compression
#define MAX_TENSOR_RANK 1024

TreeTensorNetwork* ttn_create(size_t input_dim, size_t output_dim,
                            size_t bond_dim, double tolerance) {
    TreeTensorNetwork* network = malloc(sizeof(TreeTensorNetwork));
    if (!network) return NULL;

    network->tensor = hmatrix_create(output_dim, input_dim, tolerance);
    if (!network->tensor) {
        free(network);
        return NULL;
    }

    network->left = NULL;
    network->right = NULL;
    network->input_dim = input_dim;
    network->output_dim = output_dim;
    network->bond_dim = bond_dim;
    network->tolerance = tolerance;
    network->is_leaf = true;

    // Default configuration
    network->config.max_bond_dimension = bond_dim;
    network->config.compression_tolerance = tolerance;
    network->config.use_gpu = false;
    network->config.use_metal = false;
    network->config.num_threads = 1;
    network->config.cache_size = 1024 * 1024; // 1MB default cache

    return network;
}

void ttn_destroy(TreeTensorNetwork* network) {
    if (!network) return;

    if (network->tensor) {
        hmatrix_destroy(network->tensor);
    }
    if (network->left) {
        ttn_destroy(network->left);
    }
    if (network->right) {
        ttn_destroy(network->right);
    }
    free(network);
}

void ttn_forward(TreeTensorNetwork* network,
                const double complex* input,
                double complex* output,
                size_t batch_size) {
    if (!network || !input || !output) return;

    // For leaf nodes, directly apply tensor
    if (network->is_leaf) {
        hmatrix_multiply(network->tensor, input, output, batch_size);
        return;
    }

    // Split input between children
    size_t mid = network->input_dim / 2;
    double complex* left_output = malloc(network->bond_dim * batch_size * sizeof(double complex));
    double complex* right_output = malloc(network->bond_dim * batch_size * sizeof(double complex));

    if (!left_output || !right_output) {
        free(left_output);
        free(right_output);
        return;
    }

    // Process left child
    if (network->left) {
        ttn_forward(network->left, input, left_output, batch_size);
    }

    // Process right child
    if (network->right) {
        ttn_forward(network->right, input + mid * batch_size, right_output, batch_size);
    }

    // Combine outputs through this node's tensor
    double complex* combined = malloc((network->bond_dim * 2) * batch_size * sizeof(double complex));
    if (!combined) {
        free(left_output);
        free(right_output);
        return;
    }

    // Concatenate child outputs
    for (size_t b = 0; b < batch_size; b++) {
        memcpy(combined + b * network->bond_dim * 2,
               left_output + b * network->bond_dim,
               network->bond_dim * sizeof(double complex));
        memcpy(combined + b * network->bond_dim * 2 + network->bond_dim,
               right_output + b * network->bond_dim,
               network->bond_dim * sizeof(double complex));
    }

    // Apply this node's tensor
    hmatrix_multiply(network->tensor, combined, output, batch_size);

    free(left_output);
    free(right_output);
    free(combined);
}

void ttn_backward(TreeTensorNetwork* network,
                 const double complex* grad_output,
                 double complex* grad_input,
                 size_t batch_size) {
    if (!network || !grad_output || !grad_input) return;

    // For leaf nodes, directly compute gradient
    if (network->is_leaf) {
        hmatrix_multiply_conjugate_transpose(network->tensor,
                                           grad_output, grad_input,
                                           batch_size);
        return;
    }

    // Split gradients between children
    size_t mid = network->input_dim / 2;
    double complex* left_grad = malloc(network->bond_dim * batch_size * sizeof(double complex));
    double complex* right_grad = malloc(network->bond_dim * batch_size * sizeof(double complex));

    if (!left_grad || !right_grad) {
        free(left_grad);
        free(right_grad);
        return;
    }

    // Compute gradients for this node's tensor
    hmatrix_multiply_conjugate_transpose(network->tensor,
                                       grad_output,
                                       grad_input,
                                       batch_size);

    // Split gradients between children
    for (size_t b = 0; b < batch_size; b++) {
        memcpy(left_grad + b * network->bond_dim,
               grad_input + b * network->bond_dim * 2,
               network->bond_dim * sizeof(double complex));
        memcpy(right_grad + b * network->bond_dim,
               grad_input + b * network->bond_dim * 2 + network->bond_dim,
               network->bond_dim * sizeof(double complex));
    }

    // Backpropagate through children
    if (network->left) {
        ttn_backward(network->left, left_grad,
                    grad_input, batch_size);
    }
    if (network->right) {
        ttn_backward(network->right, right_grad,
                    grad_input + mid * batch_size, batch_size);
    }

    free(left_grad);
    free(right_grad);
}

void ttn_update_parameters(TreeTensorNetwork* network,
                         double learning_rate) {
    if (!network) return;

    // Update tensor parameters
    if (network->tensor) {
        size_t size = network->tensor->rows * network->tensor->cols;
        for (size_t i = 0; i < size; i++) {
            network->tensor->data[i] -= learning_rate * network->tensor->grad[i];
        }
    }

    // Recursively update children
    if (network->left) {
        ttn_update_parameters(network->left, learning_rate);
    }
    if (network->right) {
        ttn_update_parameters(network->right, learning_rate);
    }
}

void ttn_compress(TreeTensorNetwork* network,
                 double new_tolerance) {
    if (!network) return;

    network->tolerance = new_tolerance;

    // Compress tensor if present
    if (network->tensor) {
        hmatrix_compress(network->tensor);
    }

    // Recursively compress children
    if (network->left) {
        ttn_compress(network->left, new_tolerance);
    }
    if (network->right) {
        ttn_compress(network->right, new_tolerance);
    }
}

size_t ttn_count_parameters(const TreeTensorNetwork* network) {
    if (!network) return 0;

    size_t count = 0;
    if (network->tensor) {
        count = network->tensor->rows * network->tensor->cols;
    }

    // Add parameters from children
    if (network->left) {
        count += ttn_count_parameters(network->left);
    }
    if (network->right) {
        count += ttn_count_parameters(network->right);
    }

    return count;
}

void ttn_print_stats(const TreeTensorNetwork* network) {
    if (!network) return;

    printf("Tree Tensor Network Statistics:\n");
    printf("Input dimension: %zu\n", network->input_dim);
    printf("Output dimension: %zu\n", network->output_dim);
    printf("Bond dimension: %zu\n", network->bond_dim);
    printf("Tolerance: %g\n", network->tolerance);
    printf("Total parameters: %zu\n", ttn_count_parameters(network));
    printf("Is leaf node: %s\n", network->is_leaf ? "yes" : "no");

    if (network->tensor) {
        printf("Tensor rank: %zu\n", network->tensor->rank);
        printf("Tensor memory: %.2f MB\n",
               (network->tensor->rows * network->tensor->cols * sizeof(double complex))
               / (1024.0 * 1024.0));
    }
}

double ttn_compute_complexity(const TreeTensorNetwork* network,
                            size_t input_size) {
    if (!network) return 0.0;
    
    // Base complexity for leaf node
    double complexity = input_size * network->output_dim;
    
    // Add complexity from children
    if (network->left) {
        complexity += ttn_compute_complexity(network->left, input_size);
    }
    if (network->right) {
        complexity += ttn_compute_complexity(network->right, input_size);
    }
    
    // Account for hierarchical matrix operations
    if (!network->is_leaf) {
        complexity *= log2(input_size);
    }
    
    return complexity;
}

void ttn_optimize_structure(TreeTensorNetwork* network,
                          double compression_tolerance) {
    if (!network) return;
    
    // Recursively optimize children first
    if (network->left) {
        ttn_optimize_structure(network->left, compression_tolerance);
    }
    if (network->right) {
        ttn_optimize_structure(network->right, compression_tolerance);
    }
    
    // Try to merge low-rank nodes
    if (network->left && network->right) {
        bool can_merge = true;
        
        // Check if children are low rank
        if (network->left->tensor) {
            can_merge &= hmatrix_is_low_rank(network->left->tensor);
        }
        if (network->right->tensor) {
            can_merge &= hmatrix_is_low_rank(network->right->tensor);
        }
        
        if (can_merge) {
            // Merge children into this node
            size_t merged_rank = 0;
            if (network->left->tensor) merged_rank += network->left->tensor->rank;
            if (network->right->tensor) merged_rank += network->right->tensor->rank;
            
            if (merged_rank <= MAX_TENSOR_RANK) {
                ttn_merge_networks(network, network->left, network->right);
                network->left = NULL;
                network->right = NULL;
            }
        }
    }
    
    // Compress tensor if possible
    if (network->tensor) {
        hmatrix_compress(network->tensor);
    }
}

void ttn_truncate_bonds(TreeTensorNetwork* network,
                       size_t max_bond_dim) {
    if (!network) return;
    
    // Update configuration
    network->config.max_bond_dimension = max_bond_dim;
    
    // Truncate tensor if needed
    if (network->tensor && network->tensor->rank > max_bond_dim) {
        HierarchicalMatrix* truncated = hmatrix_create(
            network->tensor->rows,
            network->tensor->cols,
            network->tolerance
        );
        
        // Copy with truncation
        memcpy(truncated->data, network->tensor->data,
               network->tensor->rows * max_bond_dim * sizeof(double complex));
        
        hmatrix_destroy(network->tensor);
        network->tensor = truncated;
    }
    
    // Recursively truncate children
    if (network->left) {
        ttn_truncate_bonds(network->left, max_bond_dim);
    }
    if (network->right) {
        ttn_truncate_bonds(network->right, max_bond_dim);
    }
}

void ttn_merge_networks(TreeTensorNetwork* dst,
                       const TreeTensorNetwork* a,
                       const TreeTensorNetwork* b) {
    if (!dst || !a || !b) return;
    
    // Create merged tensor
    size_t merged_rows = a->output_dim;
    size_t merged_cols = a->input_dim + b->input_dim;
    
    HierarchicalMatrix* merged = hmatrix_create(merged_rows, merged_cols, dst->tolerance);
    if (!merged) return;
    
    // Copy tensors side by side
    if (a->tensor) {
        memcpy(merged->data, a->tensor->data,
               a->output_dim * a->input_dim * sizeof(double complex));
    }
    if (b->tensor) {
        memcpy(merged->data + a->output_dim * a->input_dim,
               b->tensor->data,
               b->output_dim * b->input_dim * sizeof(double complex));
    }
    
    // Update destination
    hmatrix_destroy(dst->tensor);
    dst->tensor = merged;
    dst->input_dim = merged_cols;
    dst->output_dim = merged_rows;
}

void ttn_split_network(const TreeTensorNetwork* network,
                      TreeTensorNetwork** left,
                      TreeTensorNetwork** right) {
    if (!network || !left || !right) return;
    
    size_t mid_col = network->input_dim / 2;
    
    // Create left network
    *left = ttn_create(mid_col, network->output_dim,
                      network->bond_dim, network->tolerance);
    if (!*left) return;
    
    // Create right network
    *right = ttn_create(network->input_dim - mid_col,
                       network->output_dim,
                       network->bond_dim, network->tolerance);
    if (!*right) {
        ttn_destroy(*left);
        *left = NULL;
        return;
    }
    
    // Split tensor data
    if (network->tensor) {
        // Copy left half
        memcpy((*left)->tensor->data, network->tensor->data,
               network->output_dim * mid_col * sizeof(double complex));
        
        // Copy right half
        memcpy((*right)->tensor->data,
               network->tensor->data + network->output_dim * mid_col,
               network->output_dim * (network->input_dim - mid_col) * sizeof(double complex));
    }
    
    // Copy configuration
    (*left)->config = network->config;
    (*right)->config = network->config;
}

bool ttn_is_valid(const TreeTensorNetwork* network) {
    if (!network) return false;
    
    // Check dimensions
    if (network->input_dim == 0 || network->output_dim == 0) return false;
    
    // Check tensor if present
    if (network->tensor) {
        if (network->tensor->rows != network->output_dim ||
            network->tensor->cols != network->input_dim) {
            return false;
        }
    }
    
    // Check children recursively
    if (network->left && !ttn_is_valid(network->left)) return false;
    if (network->right && !ttn_is_valid(network->right)) return false;
    
    return true;
}

void ttn_randomize_parameters(TreeTensorNetwork* network) {
    if (!network) return;
    
    // Randomize tensor if present
    if (network->tensor) {
        size_t size = network->tensor->rows * network->tensor->cols;
        for (size_t i = 0; i < size; i++) {
            double real = (double)rand() / RAND_MAX - 0.5;
            double imag = (double)rand() / RAND_MAX - 0.5;
            network->tensor->data[i] = real + I * imag;
        }
    }
    
    // Recursively randomize children
    if (network->left) ttn_randomize_parameters(network->left);
    if (network->right) ttn_randomize_parameters(network->right);
}

void ttn_zero_grad(TreeTensorNetwork* network) {
    if (!network) return;

    // Zero gradients in tensor
    if (network->tensor) {
        size_t size = network->tensor->rows * network->tensor->cols;
        memset(network->tensor->grad, 0, size * sizeof(double complex));
    }

    // Recursively zero children
    if (network->left) ttn_zero_grad(network->left);
    if (network->right) ttn_zero_grad(network->right);
}

const char* ttn_error_string(TTNError error) {
    switch (error) {
        case TTN_SUCCESS:
            return "Success";
        case TTN_ERROR_INVALID_ARGUMENT:
            return "Invalid argument";
        case TTN_ERROR_OUT_OF_MEMORY:
            return "Out of memory";
        case TTN_ERROR_CUDA_ERROR:
            return "CUDA error";
        case TTN_ERROR_METAL_ERROR:
            return "Metal error";
        case TTN_ERROR_INVALID_OPERATION:
            return "Invalid operation";
        default:
            return "Unknown error";
    }
}

void ttn_set_config(TreeTensorNetwork* network,
                   const TTNConfig* config) {
    if (!network || !config) return;
    
    network->config = *config;
    
    // Apply configuration recursively
    if (network->left) ttn_set_config(network->left, config);
    if (network->right) ttn_set_config(network->right, config);
}

void ttn_get_config(const TreeTensorNetwork* network,
                   TTNConfig* config) {
    if (!network || !config) return;
    
    *config = network->config;
}

#ifdef USE_CUDA
void ttn_to_gpu(TreeTensorNetwork* network) {
    // Implementation for CUDA support
}

void ttn_to_cpu(TreeTensorNetwork* network) {
    // Implementation for CUDA support
}
#endif

#ifdef USE_METAL
void ttn_to_metal(TreeTensorNetwork* network) {
    // Implementation for Metal support
}

void ttn_to_host(TreeTensorNetwork* network) {
    // Implementation for Metal support
}
#endif
