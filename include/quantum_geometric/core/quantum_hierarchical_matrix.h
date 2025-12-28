/**
 * @file quantum_hierarchical_matrix.h
 * @brief Hierarchical matrix structures for quantum tensor networks
 *
 * Implements H-matrix and H2-matrix structures for efficient representation
 * and manipulation of large quantum operators and tensor networks with
 * near-linear complexity.
 */

#ifndef QUANTUM_HIERARCHICAL_MATRIX_H
#define QUANTUM_HIERARCHICAL_MATRIX_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

// Portable complex type definition
#ifdef __cplusplus
typedef double _Complex qgt_complex_t;
#else
typedef double _Complex qgt_complex_t;
#endif

// Forward declarations
struct geometric_tensor;
struct quantum_state;

// =============================================================================
// Hierarchical Matrix Types
// =============================================================================

/**
 * Hierarchical matrix types
 */
typedef enum {
    HMATRIX_STANDARD,                // Standard H-matrix
    HMATRIX_H2,                      // H2-matrix (uniform H-matrix)
    HMATRIX_HSS,                     // Hierarchically Semi-Separable
    HMATRIX_HODLR,                   // Hierarchically Off-Diagonal Low-Rank
    HMATRIX_BLR,                     // Block Low-Rank
    HMATRIX_TENSOR_TRAIN,            // Tensor Train format
    HMATRIX_QUANTUM                  // Quantum hierarchical
} HierarchicalMatrixType;

/**
 * Block types in hierarchical decomposition
 */
typedef enum {
    BLOCK_DENSE,                     // Dense matrix block
    BLOCK_LOW_RANK,                  // Low-rank (UV^T) block
    BLOCK_HIERARCHICAL,              // Recursively partitioned
    BLOCK_ZERO,                      // Zero block
    BLOCK_DIAGONAL,                  // Diagonal block
    BLOCK_IDENTITY                   // Identity block
} BlockType;

/**
 * Admissibility conditions
 */
typedef enum {
    ADMISSIBILITY_WEAK,              // Weak admissibility
    ADMISSIBILITY_STRONG,            // Strong admissibility
    ADMISSIBILITY_PARABOLIC,         // Parabolic admissibility
    ADMISSIBILITY_CUSTOM             // User-defined
} AdmissibilityType;

/**
 * Compression methods
 */
typedef enum {
    COMPRESS_SVD,                    // Truncated SVD
    COMPRESS_ACA,                    // Adaptive Cross Approximation
    COMPRESS_ACA_PLUS,               // ACA with pivoting
    COMPRESS_RANDOM,                 // Randomized low-rank
    COMPRESS_RRQR,                   // Rank-Revealing QR
    COMPRESS_QUANTUM                 // Quantum-assisted compression
} CompressionMethod;

// =============================================================================
// Core Structures
// =============================================================================

/**
 * Index set (cluster)
 */
typedef struct {
    size_t* indices;                 // Array of indices
    size_t size;                     // Number of indices
    double* center;                  // Cluster center (for geometry)
    double radius;                   // Cluster radius
    size_t dimension;                // Spatial dimension
} ClusterIndex;

/**
 * Cluster tree node
 */
typedef struct ClusterNode {
    ClusterIndex* index_set;
    struct ClusterNode* left;
    struct ClusterNode* right;
    struct ClusterNode* parent;
    size_t level;
    size_t node_id;
    bool is_leaf;
} ClusterNode;

/**
 * Cluster tree
 */
typedef struct {
    ClusterNode* root;
    size_t num_nodes;
    size_t num_leaves;
    size_t max_depth;
    size_t min_leaf_size;
    ClusterNode** leaves;            // Array of leaf pointers
    ClusterNode** level_nodes;       // Nodes organized by level
    size_t* level_sizes;
} ClusterTree;

/**
 * Low-rank block (UV^T representation)
 */
typedef struct {
    qgt_complex_t* U;                // Left factor [rows, rank]
    qgt_complex_t* V;                // Right factor [cols, rank]
    size_t rows;
    size_t cols;
    size_t rank;
    size_t max_rank;
    double frobenius_norm;
    double relative_error;
} LowRankBlock;

/**
 * Dense block
 */
typedef struct {
    qgt_complex_t* data;            // Dense matrix data
    size_t rows;
    size_t cols;
    size_t ld;                       // Leading dimension
    bool is_symmetric;
    bool is_hermitian;
} DenseBlock;

/**
 * Hierarchical matrix block
 */
typedef struct HMatrixBlock {
    BlockType type;
    ClusterNode* row_cluster;
    ClusterNode* col_cluster;

    union {
        DenseBlock dense;
        LowRankBlock low_rank;
        struct {
            struct HMatrixBlock* children[4];  // 2x2 subdivision
            size_t num_children;
        } hierarchical;
    } data;

    struct HMatrixBlock* parent;
    size_t level;
    size_t block_id;
    double compression_ratio;
} HMatrixBlock;

/**
 * Hierarchical matrix
 */
typedef struct {
    HierarchicalMatrixType type;
    HMatrixBlock* root;
    ClusterTree* row_tree;
    ClusterTree* col_tree;
    size_t rows;
    size_t cols;
    size_t num_blocks;
    size_t num_dense_blocks;
    size_t num_lowrank_blocks;
    size_t total_rank;
    double compression_ratio;
    double tolerance;
    AdmissibilityType admissibility;
    size_t max_rank;
    size_t min_block_size;
} HierarchicalMatrix;

/**
 * H2-matrix transfer matrices
 */
typedef struct {
    qgt_complex_t** transfer_row;   // Row transfer matrices per level
    qgt_complex_t** transfer_col;   // Column transfer matrices per level
    size_t* transfer_ranks;          // Ranks per level
    size_t num_levels;
} H2TransferMatrices;

/**
 * HSS generator representation
 */
typedef struct {
    qgt_complex_t** generators_U;   // Row generators per level
    qgt_complex_t** generators_V;   // Column generators per level
    qgt_complex_t** coupling;       // Coupling matrices
    size_t* ranks;
    size_t num_levels;
} HSSGenerators;

// =============================================================================
// Configuration
// =============================================================================

/**
 * Hierarchical matrix construction configuration
 */
typedef struct {
    HierarchicalMatrixType type;
    AdmissibilityType admissibility;
    CompressionMethod compression;
    double tolerance;                // Approximation tolerance
    size_t max_rank;                 // Maximum rank for low-rank blocks
    size_t min_block_size;           // Minimum dense block size
    size_t max_leaf_size;            // Maximum cluster leaf size
    double eta;                      // Admissibility parameter
    bool symmetric;
    bool use_adaptive_rank;
    size_t random_oversampling;      // For randomized methods
    size_t power_iterations;         // For randomized SVD
} HMatrixConfig;

// =============================================================================
// Cluster Tree Operations
// =============================================================================

/**
 * Create cluster tree from indices
 */
int cluster_tree_create(ClusterTree** tree,
                        size_t* indices,
                        size_t num_indices,
                        double* points,           // Optional geometric points
                        size_t dimension,
                        size_t max_leaf_size);

/**
 * Create cluster tree with custom clustering
 */
int cluster_tree_create_custom(ClusterTree** tree,
                               size_t* indices,
                               size_t num_indices,
                               int (*split_function)(ClusterIndex*, ClusterIndex**, ClusterIndex**));

/**
 * Destroy cluster tree
 */
void cluster_tree_destroy(ClusterTree* tree);

/**
 * Get cluster at level
 */
int cluster_tree_get_level(ClusterTree* tree, size_t level,
                           ClusterNode*** nodes, size_t* num_nodes);

/**
 * Check admissibility between clusters
 */
bool cluster_admissible(ClusterNode* row_cluster,
                        ClusterNode* col_cluster,
                        AdmissibilityType type,
                        double eta);

// =============================================================================
// Hierarchical Matrix Construction
// =============================================================================

/**
 * Create hierarchical matrix
 */
int hmatrix_create(HierarchicalMatrix** hmat,
                   size_t rows,
                   size_t cols,
                   HMatrixConfig* config);

/**
 * Create from dense matrix
 */
int hmatrix_from_dense(HierarchicalMatrix** hmat,
                       qgt_complex_t* dense,
                       size_t rows,
                       size_t cols,
                       HMatrixConfig* config);

/**
 * Create from matrix-free operator
 */
int hmatrix_from_operator(HierarchicalMatrix** hmat,
                          void (*matvec)(qgt_complex_t*, qgt_complex_t*, void*),
                          void* context,
                          size_t rows,
                          size_t cols,
                          HMatrixConfig* config);

/**
 * Create from kernel function
 */
int hmatrix_from_kernel(HierarchicalMatrix** hmat,
                        qgt_complex_t (*kernel)(size_t, size_t, void*),
                        void* context,
                        double* row_points,
                        double* col_points,
                        size_t num_rows,
                        size_t num_cols,
                        size_t dimension,
                        HMatrixConfig* config);

/**
 * Destroy hierarchical matrix
 */
void hmatrix_destroy(HierarchicalMatrix* hmat);

// =============================================================================
// Matrix Operations
// =============================================================================

/**
 * Matrix-vector multiplication: y = alpha * A * x + beta * y
 */
int hmatrix_matvec(HierarchicalMatrix* hmat,
                   qgt_complex_t alpha,
                   qgt_complex_t* x,
                   qgt_complex_t beta,
                   qgt_complex_t* y);

/**
 * Matrix-matrix multiplication: C = alpha * A * B + beta * C
 */
int hmatrix_matmul(HierarchicalMatrix* A,
                   HierarchicalMatrix* B,
                   qgt_complex_t alpha,
                   qgt_complex_t beta,
                   HierarchicalMatrix** C);

/**
 * Matrix addition: C = alpha * A + beta * B
 */
int hmatrix_add(HierarchicalMatrix* A,
                HierarchicalMatrix* B,
                qgt_complex_t alpha,
                qgt_complex_t beta,
                HierarchicalMatrix** C);

/**
 * Transpose
 */
int hmatrix_transpose(HierarchicalMatrix* hmat,
                      HierarchicalMatrix** transposed);

/**
 * Conjugate transpose
 */
int hmatrix_hermitian(HierarchicalMatrix* hmat,
                      HierarchicalMatrix** hermitian);

/**
 * Scale: A = alpha * A
 */
int hmatrix_scale(HierarchicalMatrix* hmat, qgt_complex_t alpha);

// =============================================================================
// Factorizations and Solvers
// =============================================================================

/**
 * LU factorization
 */
int hmatrix_lu(HierarchicalMatrix* hmat,
               HierarchicalMatrix** L,
               HierarchicalMatrix** U);

/**
 * Cholesky factorization (for SPD matrices)
 */
int hmatrix_cholesky(HierarchicalMatrix* hmat,
                     HierarchicalMatrix** L);

/**
 * Solve Ax = b using LU factorization
 */
int hmatrix_solve_lu(HierarchicalMatrix* L,
                     HierarchicalMatrix* U,
                     qgt_complex_t* b,
                     qgt_complex_t* x);

/**
 * Solve Ax = b using Cholesky factorization
 */
int hmatrix_solve_cholesky(HierarchicalMatrix* L,
                           qgt_complex_t* b,
                           qgt_complex_t* x);

/**
 * Compute inverse
 */
int hmatrix_inverse(HierarchicalMatrix* hmat,
                    HierarchicalMatrix** inverse);

/**
 * Iterative solver with H-matrix preconditioner
 */
int hmatrix_solve_iterative(HierarchicalMatrix* A,
                            qgt_complex_t* b,
                            qgt_complex_t* x,
                            HierarchicalMatrix* preconditioner,
                            double tolerance,
                            size_t max_iterations,
                            size_t* iterations_out);

// =============================================================================
// Compression and Recompression
// =============================================================================

/**
 * Compress to specified tolerance
 */
int hmatrix_compress(HierarchicalMatrix* hmat, double tolerance);

/**
 * Recompress after arithmetic operations
 */
int hmatrix_recompress(HierarchicalMatrix* hmat);

/**
 * Truncate ranks to maximum
 */
int hmatrix_truncate_rank(HierarchicalMatrix* hmat, size_t max_rank);

/**
 * Convert to H2-matrix format
 */
int hmatrix_to_h2(HierarchicalMatrix* hmat,
                  HierarchicalMatrix** h2mat);

// =============================================================================
// Quantum Operations
// =============================================================================

/**
 * Create from quantum operator
 */
int hmatrix_from_quantum_operator(HierarchicalMatrix** hmat,
                                  struct geometric_tensor* op,
                                  HMatrixConfig* config);

/**
 * Apply to quantum state
 */
int hmatrix_apply_quantum(HierarchicalMatrix* hmat,
                          struct quantum_state* state,
                          struct quantum_state** result);

/**
 * Quantum-assisted compression
 */
int hmatrix_quantum_compress(HierarchicalMatrix* hmat,
                             size_t num_qubits,
                             double tolerance);

/**
 * Tensor train to H-matrix conversion
 */
int hmatrix_from_tensor_train(HierarchicalMatrix** hmat,
                              qgt_complex_t** cores,
                              size_t* ranks,
                              size_t num_cores,
                              size_t* mode_sizes);

/**
 * H-matrix to tensor train conversion
 */
int hmatrix_to_tensor_train(HierarchicalMatrix* hmat,
                            qgt_complex_t*** cores,
                            size_t** ranks,
                            size_t* num_cores,
                            double tolerance);

// =============================================================================
// Block Operations
// =============================================================================

/**
 * Get block by row/column cluster
 */
HMatrixBlock* hmatrix_get_block(HierarchicalMatrix* hmat,
                                ClusterNode* row_cluster,
                                ClusterNode* col_cluster);

/**
 * Set block data
 */
int hmatrix_set_block(HierarchicalMatrix* hmat,
                      ClusterNode* row_cluster,
                      ClusterNode* col_cluster,
                      BlockType type,
                      void* data);

/**
 * Convert dense block to low-rank
 */
int block_dense_to_lowrank(DenseBlock* dense,
                           LowRankBlock* lowrank,
                           CompressionMethod method,
                           double tolerance);

/**
 * Convert low-rank block to dense
 */
int block_lowrank_to_dense(LowRankBlock* lowrank,
                           DenseBlock* dense);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Get compression ratio
 */
double hmatrix_compression_ratio(HierarchicalMatrix* hmat);

/**
 * Get memory usage in bytes
 */
size_t hmatrix_memory_usage(HierarchicalMatrix* hmat);

/**
 * Estimate approximation error
 */
int hmatrix_estimate_error(HierarchicalMatrix* hmat,
                           qgt_complex_t (*exact_entry)(size_t, size_t, void*),
                           void* context,
                           size_t num_samples,
                           double* error_out);

/**
 * Validate H-matrix structure
 */
bool hmatrix_validate(HierarchicalMatrix* hmat);

/**
 * Print H-matrix structure
 */
void hmatrix_print_structure(HierarchicalMatrix* hmat);

/**
 * Print statistics
 */
void hmatrix_print_stats(HierarchicalMatrix* hmat);

/**
 * Export to dense matrix
 */
int hmatrix_to_dense(HierarchicalMatrix* hmat,
                     qgt_complex_t** dense,
                     size_t* rows,
                     size_t* cols);

/**
 * Clone H-matrix
 */
int hmatrix_clone(HierarchicalMatrix* src, HierarchicalMatrix** dst);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_HIERARCHICAL_MATRIX_H
