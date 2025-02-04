#ifndef HIERARCHICAL_MATRIX_H
#define HIERARCHICAL_MATRIX_H

#include <complex.h>
#include <stddef.h>
#include <stdbool.h>

// Matrix types
typedef enum {
    MATRIX_DENSE,           // Dense matrix
    MATRIX_SPARSE,          // Sparse matrix
    MATRIX_HIERARCHICAL,    // Hierarchical matrix
    MATRIX_QUANTUM,         // Quantum matrix
    MATRIX_HYBRID          // Hybrid matrix
} matrix_type_t;

// Compression modes
typedef enum {
    COMPRESS_SVD,          // SVD compression
    COMPRESS_QR,           // QR compression
    COMPRESS_ACA,          // Adaptive cross approximation
    COMPRESS_QUANTUM,      // Quantum compression
    COMPRESS_ADAPTIVE     // Adaptive compression
} compression_mode_t;

// Storage formats
typedef enum {
    STORAGE_FULL,          // Full storage
    STORAGE_PACKED,        // Packed storage
    STORAGE_DISTRIBUTED,   // Distributed storage
    STORAGE_QUANTUM,       // Quantum storage
    STORAGE_HYBRID        // Hybrid storage
} storage_format_t;

// Matrix properties
typedef struct {
    size_t dimension;              // Matrix dimension
    size_t num_blocks;             // Number of blocks
    double tolerance;              // Error tolerance
    bool symmetric;                // Symmetry flag
    bool positive_definite;        // Positive definiteness
    double condition_number;       // Condition number
    void* properties_data;        // Additional properties
} matrix_properties_t;

// Block configuration
typedef struct {
    size_t* block_sizes;           // Block sizes
    size_t num_blocks;             // Number of blocks
    size_t min_block_size;         // Minimum block size
    size_t max_block_size;         // Maximum block size
    bool adaptive_blocking;        // Adaptive blocking flag
    void* block_data;            // Additional block data
} block_config_t;

// Compression parameters
typedef struct {
    compression_mode_t mode;       // Compression mode
    double tolerance;              // Compression tolerance
    size_t max_rank;              // Maximum rank
    bool recompression;           // Enable recompression
    double threshold;             // Compression threshold
    void* compression_data;      // Additional parameters
} compression_params_t;

// Basic hierarchical matrix structure
typedef struct HierarchicalMatrix {
    matrix_type_t type;            // Matrix type
    storage_format_t format;       // Storage format
    double complex* data;          // Raw data for leaf nodes
    double complex* U;             // Left singular vectors
    double complex* V;             // Right singular vectors
    size_t n;                     // Matrix dimension
    size_t rows;                  // Number of rows
    size_t cols;                  // Number of columns
    size_t rank;                  // Matrix rank
    bool is_leaf;                 // Leaf node flag
    double tolerance;             // SVD tolerance
    struct HierarchicalMatrix* children[4];  // Child matrices
    void* matrix_data;           // Additional data
} HierarchicalMatrix;

// Core functions
HierarchicalMatrix* create_hierarchical_matrix(size_t n, double tolerance);
void destroy_hierarchical_matrix(HierarchicalMatrix* matrix);

// Initialization functions
bool init_matrix_properties(HierarchicalMatrix* matrix,
                          const matrix_properties_t* props);
bool init_block_structure(HierarchicalMatrix* matrix,
                         const block_config_t* config);
bool validate_initialization(const HierarchicalMatrix* matrix);

// Block operations
bool set_block_sizes(HierarchicalMatrix* matrix,
                    const size_t* sizes,
                    size_t num_blocks);
double complex* get_block(HierarchicalMatrix* matrix,
                         size_t block_idx);
size_t get_block_size(const HierarchicalMatrix* matrix,
                      size_t block_idx);
bool set_block_data(HierarchicalMatrix* matrix,
                    size_t block_idx,
                    const double complex* data);

// Matrix operations
bool multiply_matrices(HierarchicalMatrix* result,
                      const HierarchicalMatrix* a,
                      const HierarchicalMatrix* b);
bool add_matrices(HierarchicalMatrix* result,
                  const HierarchicalMatrix* a,
                  const HierarchicalMatrix* b);
bool subtract_matrices(HierarchicalMatrix* result,
                      const HierarchicalMatrix* a,
                      const HierarchicalMatrix* b);
bool scale_matrix(HierarchicalMatrix* matrix,
                  double complex scalar);

// Compression operations
bool compress_matrix(HierarchicalMatrix* matrix,
                    const compression_params_t* params);
bool recompress_matrix(HierarchicalMatrix* matrix,
                      double tolerance);
bool truncate_singular_values(HierarchicalMatrix* matrix,
                            double threshold);

// Decomposition operations
void compute_svd(double complex* data, size_t rows, size_t cols,
                double complex* U, double complex* S, double complex* V);
bool compute_qr(HierarchicalMatrix* matrix,
               double complex** Q,
               double complex** R);
bool compute_lu(HierarchicalMatrix* matrix,
               double complex** L,
               double complex** U);

// Analysis functions
bool compute_norm(const HierarchicalMatrix* matrix,
                 double* norm);
bool estimate_rank(const HierarchicalMatrix* matrix,
                  size_t* rank);
bool compute_condition_number(const HierarchicalMatrix* matrix,
                            double* condition);

// Matrix operations
void hmatrix_transpose(HierarchicalMatrix* dst, const HierarchicalMatrix* src);

// Quantum operations
bool quantum_compress(HierarchicalMatrix* matrix,
                     const compression_params_t* params);
bool quantum_multiply(HierarchicalMatrix* result,
                     const HierarchicalMatrix* a,
                     const HierarchicalMatrix* b);
bool quantum_decompose(HierarchicalMatrix* matrix,
                      double complex** factors);

// Utility functions
bool export_matrix(const HierarchicalMatrix* matrix,
                  const char* filename);
bool import_matrix(HierarchicalMatrix* matrix,
                  const char* filename);
void print_matrix(const HierarchicalMatrix* matrix);

#endif // HIERARCHICAL_MATRIX_H
