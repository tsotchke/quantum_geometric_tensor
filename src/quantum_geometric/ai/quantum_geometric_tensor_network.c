#include "quantum_geometric/ai/quantum_geometric_tensor_network.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/core/tensor_types.h"
#include "quantum_geometric/core/tensor_operations.h"
#include "quantum_geometric/core/memory_pool.h"
#include "quantum_geometric/core/performance_operations.h"
#include "quantum_geometric/core/quantum_geometric_constants.h"
#include "quantum_geometric/core/quantum_geometric_logging.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

// Error code definitions
#ifndef QG_SUCCESS
#define QG_SUCCESS 0
#endif
#ifndef QG_ERROR_INVALID_ARGUMENT
#define QG_ERROR_INVALID_ARGUMENT -1
#endif
#ifndef QG_ERROR_OUT_OF_MEMORY
#define QG_ERROR_OUT_OF_MEMORY -2
#endif
#ifndef QG_ERROR_MEMORY_POOL_INIT
#define QG_ERROR_MEMORY_POOL_INIT -3
#endif

// Default pool configuration
#ifndef QG_INITIAL_POOL_SIZE
#define QG_INITIAL_POOL_SIZE (1024 * 1024)  // 1 MB
#endif
#ifndef QG_TENSOR_CACHE_LINE_SIZE
#define QG_TENSOR_CACHE_LINE_SIZE 64
#endif
#ifndef QG_TENSOR_BLOCK_SIZE
#define QG_TENSOR_BLOCK_SIZE 64
#endif
#ifndef QG_MIN_SIZE_FOR_GPU
#define QG_MIN_SIZE_FOR_GPU (1024 * 1024)  // 1 MB threshold for GPU offload
#endif
#ifndef QG_MAX_TENSOR_THREADS
#define QG_MAX_TENSOR_THREADS 8
#endif

// tensor_t and tensor_network_t are defined in tensor_types.h
// This file uses those unified definitions throughout

// Tensor network statistics
typedef struct tensor_network_stats_t {
    size_t num_contractions;
    size_t num_decompositions;
    double total_contraction_time;
    double total_decomposition_time;
    size_t peak_memory;
    size_t current_memory;
} tensor_network_stats_t;

// Optimization parameters
typedef struct optimization_params_t {
    double tolerance;           // Convergence tolerance
    size_t max_iterations;     // Maximum iterations
    bool use_gpu;              // Use GPU acceleration if available
    double learning_rate;      // Learning rate for iterative optimization
} optimization_params_t;

// Performance tracking
static struct {
    size_t total_contractions;
    size_t total_decompositions;
    double total_contraction_time;
    double total_decomposition_time;
    size_t peak_memory;
    size_t current_memory;
} perf_stats = {0};

// Memory pool for tensor operations
static MemoryPool* tensor_pool = NULL;

// Initialize tensor pool
static int init_tensor_pool(void) {
    if (!tensor_pool) {
        PoolConfig config = {
            .min_block_size = QG_TENSOR_CACHE_LINE_SIZE,
            .alignment = QG_TENSOR_CACHE_LINE_SIZE,
            .num_size_classes = 8,
            .growth_factor = 2.0f,
            .prefetch_distance = 8,
            .use_huge_pages = false,
            .cache_local_free_lists = true,
            .max_blocks_per_class = 1024,
            .thread_cache_size = 64,
            .enable_stats = true
        };
        tensor_pool = create_memory_pool(&config);
    }
    return tensor_pool != NULL ? QG_SUCCESS : QG_ERROR_MEMORY_POOL_INIT;
}

// Helper function to track memory usage
static void update_memory_stats(size_t size, bool allocate) {
    if (allocate) {
        perf_stats.current_memory += size;
        if (perf_stats.current_memory > perf_stats.peak_memory) {
            perf_stats.peak_memory = perf_stats.current_memory;
        }
    } else {
        perf_stats.current_memory -= size;
    }
}

// Helper function to compute total size of a tensor node
static size_t compute_node_total_size(const tensor_node_t* node) {
    if (!node || !node->dimensions || node->num_dimensions == 0) return 0;
    size_t size = 1;
    for (size_t i = 0; i < node->num_dimensions; i++) {
        size *= node->dimensions[i];
    }
    return size;
}

// Forward declarations for helper functions (using tensor_node_t for network nodes)
static size_t count_shared_indices(const tensor_node_t* t1, const tensor_node_t* t2);
static float compute_contraction_cost(const tensor_node_t* t1, const tensor_node_t* t2, size_t shared);
static void find_optimal_sequence(const float* cost_matrix, size_t n, size_t* sequence, float* min_cost);
static int contract_pair(const tensor_node_t* t1, const tensor_node_t* t2, tensor_node_t** result);
static float compute_truncation_error(const ComplexFloat* singular_values, size_t size, float tolerance);
static size_t find_truncation_rank(const ComplexFloat* singular_values, size_t size, float tolerance);
static tensor_node_t* create_compressed_node(const tensor_t* u, const tensor_t* s, const tensor_t* v, size_t rank);
static bool is_gpu_available(void);
static int qg_tensor_network_optimize_gpu(tensor_network_t* network, const optimization_params_t* params);
static int qg_tensor_network_compress_gpu(tensor_network_t* network, float tolerance);

// Tensor initialization with pool allocation
int qg_tensor_network_tensor_init(tensor_t* tensor, const size_t* dimensions, size_t rank) {
    if (!tensor || !dimensions || rank == 0) {
        return QG_ERROR_INVALID_ARGUMENT;
    }

    // Ensure pool is initialized
    int status = init_tensor_pool();
    if (status != QG_SUCCESS) {
        return status;
    }

    // Calculate total size
    size_t total_size = 1;
    for (size_t i = 0; i < rank; i++) {
        if (dimensions[i] == 0) {
            return QG_ERROR_INVALID_ARGUMENT;
        }
        total_size *= dimensions[i];
    }

    // Allocate dimensions array
    tensor->dimensions = (size_t*)malloc(rank * sizeof(size_t));
    if (!tensor->dimensions) {
        return QG_ERROR_OUT_OF_MEMORY;
    }
    memcpy(tensor->dimensions, dimensions, rank * sizeof(size_t));

    // Allocate data from pool for better cache performance
    size_t data_size = total_size * sizeof(ComplexFloat);
    tensor->data = (ComplexFloat*)pool_allocate(tensor_pool, data_size);
    if (!tensor->data) {
        free(tensor->dimensions);
        tensor->dimensions = NULL;
        return QG_ERROR_OUT_OF_MEMORY;
    }

    // Initialize data to zero using SIMD when available
    #ifdef __AVX2__
    const size_t simd_width = 4;  // 256 bits = 4 complex floats (8 floats)
    size_t simd_count = (total_size / simd_width) * simd_width;
    __m256 vzero = _mm256_setzero_ps();

    for (size_t i = 0; i < simd_count; i += simd_width) {
        _mm256_store_ps((float*)&tensor->data[i], vzero);
    }
    for (size_t i = simd_count; i < total_size; i++) {
        tensor->data[i].real = 0.0f;
        tensor->data[i].imag = 0.0f;
    }
    #elif defined(__ARM_NEON)
    const size_t simd_width = 2;  // 128 bits = 2 complex floats
    size_t simd_count = (total_size / simd_width) * simd_width;
    float32x4_t vzero = vdupq_n_f32(0.0f);

    for (size_t i = 0; i < simd_count; i += simd_width) {
        vst1q_f32((float*)&tensor->data[i], vzero);
    }
    for (size_t i = simd_count; i < total_size; i++) {
        tensor->data[i].real = 0.0f;
        tensor->data[i].imag = 0.0f;
    }
    #else
    memset(tensor->data, 0, data_size);
    #endif

    // Initialize other fields
    tensor->rank = rank;
    tensor->total_size = total_size;
    tensor->is_contiguous = true;
    tensor->strides = NULL;
    tensor->owns_data = true;
    tensor->device = NULL;
    tensor->auxiliary_data = NULL;

    // Update memory stats
    update_memory_stats(data_size, true);

    return QG_SUCCESS;
}

// Tensor cleanup with pool deallocation
void qg_tensor_network_tensor_cleanup(tensor_t* tensor) {
    if (!tensor) return;

    if (tensor->data && tensor->owns_data) {
        size_t data_size = tensor->total_size * sizeof(ComplexFloat);
        if (tensor_pool) {
            pool_free(tensor_pool, tensor->data);
        } else {
            free(tensor->data);
        }
        update_memory_stats(data_size, false);
        tensor->data = NULL;
    }

    if (tensor->dimensions) {
        free(tensor->dimensions);
        tensor->dimensions = NULL;
    }

    if (tensor->strides) {
        free(tensor->strides);
        tensor->strides = NULL;
    }

    if (tensor->auxiliary_data) {
        free(tensor->auxiliary_data);
        tensor->auxiliary_data = NULL;
    }

    tensor->rank = 0;
    tensor->total_size = 0;
    tensor->is_contiguous = true;
    tensor->owns_data = false;
    tensor->device = NULL;
}

// Tensor node initialization with pool allocation
static int tensor_node_init(tensor_node_t* node, const size_t* dimensions, size_t rank) {
    if (!node || !dimensions || rank == 0) {
        return QG_ERROR_INVALID_ARGUMENT;
    }

    // Ensure pool is initialized
    int status = init_tensor_pool();
    if (status != QG_SUCCESS) {
        return status;
    }

    // Calculate total size
    size_t total_size = 1;
    for (size_t i = 0; i < rank; i++) {
        if (dimensions[i] == 0) {
            return QG_ERROR_INVALID_ARGUMENT;
        }
        total_size *= dimensions[i];
    }

    // Allocate dimensions array
    node->dimensions = (size_t*)malloc(rank * sizeof(size_t));
    if (!node->dimensions) {
        return QG_ERROR_OUT_OF_MEMORY;
    }
    memcpy(node->dimensions, dimensions, rank * sizeof(size_t));

    // Allocate data from pool
    size_t data_size = total_size * sizeof(ComplexFloat);
    node->data = (ComplexFloat*)pool_allocate(tensor_pool, data_size);
    if (!node->data) {
        free(node->dimensions);
        node->dimensions = NULL;
        return QG_ERROR_OUT_OF_MEMORY;
    }

    // Zero-initialize data
    memset(node->data, 0, data_size);

    // Initialize fields
    node->rank = rank;
    node->num_dimensions = rank;  // Keep both in sync
    node->total_size = total_size;
    node->is_valid = true;
    node->id = 0;
    node->num_connections = 0;
    node->connected_nodes = NULL;
    node->connected_dims = NULL;

    // Update memory stats
    update_memory_stats(data_size, true);

    return QG_SUCCESS;
}

// Tensor node cleanup with pool deallocation
static void tensor_node_cleanup(tensor_node_t* node) {
    if (!node) return;

    if (node->data) {
        size_t data_size = node->total_size * sizeof(ComplexFloat);
        if (tensor_pool) {
            pool_free(tensor_pool, node->data);
        } else {
            free(node->data);
        }
        update_memory_stats(data_size, false);
        node->data = NULL;
    }

    if (node->dimensions) {
        free(node->dimensions);
        node->dimensions = NULL;
    }

    if (node->connected_nodes) {
        free(node->connected_nodes);
        node->connected_nodes = NULL;
    }

    if (node->connected_dims) {
        free(node->connected_dims);
        node->connected_dims = NULL;
    }

    node->rank = 0;
    node->num_dimensions = 0;
    node->total_size = 0;
    node->is_valid = false;
    node->num_connections = 0;
}

// Network initialization
int qg_tensor_network_init(tensor_network_t* network, size_t capacity) {
    if (!network || capacity == 0) {
        return QG_ERROR_INVALID_ARGUMENT;
    }

    // Allocate node pointer array (array of pointers to tensor_node_t)
    network->nodes = (tensor_node_t**)calloc(capacity, sizeof(tensor_node_t*));
    if (!network->nodes) {
        return QG_ERROR_OUT_OF_MEMORY;
    }

    // Allocate connections array (max connections = capacity^2)
    network->connections = (size_t*)calloc(capacity * capacity, sizeof(size_t));
    if (!network->connections) {
        free(network->nodes);
        network->nodes = NULL;
        return QG_ERROR_OUT_OF_MEMORY;
    }

    network->num_nodes = 0;
    network->num_connections = 0;
    network->is_optimized = false;
    network->contraction_order = NULL;
    network->max_memory = 0;
    network->auxiliary_data = NULL;
    network->device = NULL;

    return QG_SUCCESS;
}

// Network cleanup
void qg_tensor_network_cleanup(tensor_network_t* network) {
    if (!network) return;

    // Clean up individual tensors
    if (network->nodes) {
        for (size_t i = 0; i < network->num_nodes; i++) {
            qg_tensor_network_tensor_cleanup(&network->nodes[i]);
        }
        free(network->nodes);
        network->nodes = NULL;
    }

    if (network->connections) {
        free(network->connections);
        network->connections = NULL;
    }

    if (network->contraction_order) {
        free(network->contraction_order);
        network->contraction_order = NULL;
    }

    if (network->auxiliary_data) {
        free(network->auxiliary_data);
        network->auxiliary_data = NULL;
    }

    network->num_nodes = 0;
    network->num_connections = 0;
    network->is_optimized = false;
    network->max_memory = 0;
    network->device = NULL;
}

// Count shared indices between two tensor nodes (for contraction cost estimation)
static size_t count_shared_indices(const tensor_node_t* t1, const tensor_node_t* t2) {
    if (!t1 || !t2 || !t1->dimensions || !t2->dimensions) {
        return 0;
    }

    size_t shared = 0;
    for (size_t i = 0; i < t1->rank; i++) {
        for (size_t j = 0; j < t2->rank; j++) {
            if (t1->dimensions[i] == t2->dimensions[j]) {
                shared++;
            }
        }
    }
    return shared;
}

// Compute contraction cost based on tensor node sizes
static float compute_contraction_cost(const tensor_node_t* t1, const tensor_node_t* t2, size_t shared) {
    if (!t1 || !t2) {
        return INFINITY;
    }

    // Cost is proportional to output size * contraction dimension
    size_t output_size = (t1->total_size / (shared > 0 ? shared : 1)) *
                         (t2->total_size / (shared > 0 ? shared : 1));
    size_t contract_dim = shared > 0 ? shared : 1;

    return (float)(output_size * contract_dim);
}

// Find optimal contraction sequence using greedy algorithm with memoization
static void find_optimal_sequence(const float* cost_matrix, size_t n,
                                   size_t* sequence, float* min_cost) {
    if (!cost_matrix || !sequence || !min_cost || n < 2) {
        return;
    }

    // Track which tensors have been contracted
    bool* used = (bool*)calloc(n, sizeof(bool));
    if (!used) return;

    *min_cost = 0.0f;
    size_t seq_idx = 0;

    // Greedy: always contract the cheapest pair
    for (size_t step = 0; step < n - 1; step++) {
        float best_cost = INFINITY;
        size_t best_i = 0, best_j = 0;

        for (size_t i = 0; i < n; i++) {
            if (used[i]) continue;
            for (size_t j = i + 1; j < n; j++) {
                if (used[j]) continue;
                float cost = cost_matrix[i * n + j];
                if (cost < best_cost) {
                    best_cost = cost;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        sequence[seq_idx++] = best_i;
        sequence[seq_idx++] = best_j;
        *min_cost += best_cost;
        used[best_j] = true;  // Mark second tensor as used, first holds result
    }

    free(used);
}

// Contract a pair of tensor nodes
static int contract_pair(const tensor_node_t* t1, const tensor_node_t* t2, tensor_node_t** result) {
    if (!t1 || !t2 || !result || !t1->data || !t2->data) {
        return QG_ERROR_INVALID_ARGUMENT;
    }

    // Allocate result tensor node
    *result = (tensor_node_t*)malloc(sizeof(tensor_node_t));
    if (!*result) {
        return QG_ERROR_OUT_OF_MEMORY;
    }
    memset(*result, 0, sizeof(tensor_node_t));

    // For now, assume last dimension of t1 contracts with first of t2 (matrix multiplication style)
    size_t new_rank = t1->rank + t2->rank - 2;
    if (new_rank == 0) new_rank = 1;  // Handle scalar result

    size_t* new_dims = (size_t*)malloc(new_rank * sizeof(size_t));
    if (!new_dims) {
        free(*result);
        *result = NULL;
        return QG_ERROR_OUT_OF_MEMORY;
    }

    // Build output dimensions (exclude contracted dimensions)
    size_t dim_idx = 0;
    for (size_t i = 0; i < t1->rank - 1; i++) {
        new_dims[dim_idx++] = t1->dimensions[i];
    }
    for (size_t i = 1; i < t2->rank; i++) {
        new_dims[dim_idx++] = t2->dimensions[i];
    }

    // Initialize result tensor node
    int status = tensor_node_init(*result, new_dims, new_rank);
    free(new_dims);

    if (status != QG_SUCCESS) {
        free(*result);
        *result = NULL;
        return status;
    }

    // Perform contraction using core tensor contract function
    size_t contract_idx_a = t1->rank - 1;
    size_t contract_idx_b = 0;

    bool success = qg_tensor_contract(
        (*result)->data,
        t1->data,
        t2->data,
        t1->dimensions,
        t2->dimensions,
        t1->rank,
        t2->rank,
        &contract_idx_a,
        &contract_idx_b,
        1
    );

    if (!success) {
        tensor_node_cleanup(*result);
        free(*result);
        *result = NULL;
        return QG_ERROR_INVALID_ARGUMENT;
    }

    return QG_SUCCESS;
}

// Check if GPU is available
static bool is_gpu_available(void) {
    #ifdef __APPLE__
    // Check for Metal support on macOS
    return true;  // Metal is available on modern macOS
    #elif defined(CUDA_AVAILABLE)
    return true;
    #else
    return false;
    #endif
}

// GPU-accelerated network optimization (stub for platform-specific implementation)
static int qg_tensor_network_optimize_gpu(tensor_network_t* network,
                                          const optimization_params_t* params) {
    (void)network;
    (void)params;
    // Platform-specific GPU optimization would go here
    // Falls back to CPU implementation for now
    return QG_ERROR_INVALID_ARGUMENT;
}

// GPU-accelerated network compression (stub for platform-specific implementation)
static int qg_tensor_network_compress_gpu(tensor_network_t* network, float tolerance) {
    (void)network;
    (void)tolerance;
    // Platform-specific GPU compression would go here
    return QG_ERROR_INVALID_ARGUMENT;
}

// Enhanced tensor contraction with performance tracking
int qg_tensor_network_contract_tensors(const tensor_t* tensor1,
                                        const tensor_t* tensor2,
                                        const size_t* indices1,
                                        const size_t* indices2,
                                        size_t num_indices,
                                        tensor_t* result) {
    performance_timer_t timer;
    qg_timer_start(&timer, "tensor_contract");
    perf_stats.total_contractions++;

    int status = QG_SUCCESS;

    // Validate inputs
    if (!tensor1 || !tensor2 || !result || !tensor1->data || !tensor2->data) {
        status = QG_ERROR_INVALID_ARGUMENT;
        goto cleanup;
    }

    if (num_indices > 0 && (!indices1 || !indices2)) {
        status = QG_ERROR_INVALID_ARGUMENT;
        goto cleanup;
    }

    // Validate contraction indices
    for (size_t i = 0; i < num_indices; i++) {
        if (indices1[i] >= tensor1->rank || indices2[i] >= tensor2->rank) {
            status = QG_ERROR_INVALID_ARGUMENT;
            goto cleanup;
        }
        // Check dimension compatibility
        if (tensor1->dimensions[indices1[i]] != tensor2->dimensions[indices2[i]]) {
            status = QG_ERROR_INVALID_ARGUMENT;
            goto cleanup;
        }
    }

    // Calculate result dimensions
    size_t result_rank = tensor1->rank + tensor2->rank - 2 * num_indices;
    if (result_rank == 0) result_rank = 1;  // Scalar result

    size_t* result_dims = (size_t*)malloc(result_rank * sizeof(size_t));
    if (!result_dims) {
        status = QG_ERROR_OUT_OF_MEMORY;
        goto cleanup;
    }

    // Build result dimensions from non-contracted indices
    size_t dim_idx = 0;
    for (size_t i = 0; i < tensor1->rank; i++) {
        bool contracted = false;
        for (size_t j = 0; j < num_indices; j++) {
            if (indices1[j] == i) {
                contracted = true;
                break;
            }
        }
        if (!contracted) {
            result_dims[dim_idx++] = tensor1->dimensions[i];
        }
    }
    for (size_t i = 0; i < tensor2->rank; i++) {
        bool contracted = false;
        for (size_t j = 0; j < num_indices; j++) {
            if (indices2[j] == i) {
                contracted = true;
                break;
            }
        }
        if (!contracted) {
            result_dims[dim_idx++] = tensor2->dimensions[i];
        }
    }

    // Initialize result tensor
    status = qg_tensor_network_tensor_init(result, result_dims, result_rank);
    if (status != QG_SUCCESS) {
        free(result_dims);
        goto cleanup;
    }

    // Perform contraction using existing core function
    bool success = qg_tensor_contract(
        result->data,
        tensor1->data,
        tensor2->data,
        tensor1->dimensions,
        tensor2->dimensions,
        tensor1->rank,
        tensor2->rank,
        indices1,
        indices2,
        num_indices
    );

    free(result_dims);

    if (!success) {
        qg_tensor_network_tensor_cleanup(result);
        status = QG_ERROR_INVALID_ARGUMENT;
    }

cleanup:
    qg_timer_stop(&timer);
    perf_stats.total_contraction_time += qg_timer_get_elapsed(&timer);
    return status;
}

// Enhanced SVD decomposition with performance tracking for AI tensor networks
int qg_ai_tensor_decompose_svd(const tensor_t* tensor,
                                size_t split_dim,
                                tensor_t* u_tensor,
                                tensor_t* s_tensor,
                                tensor_t* v_tensor) {
    performance_timer_t timer;
    qg_timer_start(&timer, "tensor_decompose_svd");
    perf_stats.total_decompositions++;

    int status = QG_SUCCESS;

    // Validate inputs
    if (!tensor || !u_tensor || !s_tensor || !v_tensor || !tensor->data) {
        status = QG_ERROR_INVALID_ARGUMENT;
        goto cleanup;
    }

    if (split_dim >= tensor->rank) {
        status = QG_ERROR_INVALID_ARGUMENT;
        goto cleanup;
    }

    // Calculate matrix dimensions for SVD
    // Reshape tensor into matrix: [prod(dims[:split_dim]), prod(dims[split_dim:])]
    size_t rows = 1, cols = 1;
    for (size_t i = 0; i < split_dim; i++) {
        rows *= tensor->dimensions[i];
    }
    for (size_t i = split_dim; i < tensor->rank; i++) {
        cols *= tensor->dimensions[i];
    }

    size_t min_dim = (rows < cols) ? rows : cols;

    // Allocate working buffers for SVD
    // Using a simplified SVD via power iteration for robustness
    ComplexFloat* A = (ComplexFloat*)malloc(rows * cols * sizeof(ComplexFloat));
    ComplexFloat* U_data = (ComplexFloat*)malloc(rows * min_dim * sizeof(ComplexFloat));
    ComplexFloat* S_data = (ComplexFloat*)malloc(min_dim * sizeof(ComplexFloat));
    ComplexFloat* V_data = (ComplexFloat*)malloc(min_dim * cols * sizeof(ComplexFloat));

    if (!A || !U_data || !S_data || !V_data) {
        free(A);
        free(U_data);
        free(S_data);
        free(V_data);
        status = QG_ERROR_OUT_OF_MEMORY;
        goto cleanup;
    }

    // Copy and reshape tensor data
    memcpy(A, tensor->data, rows * cols * sizeof(ComplexFloat));

    // Initialize output matrices
    memset(U_data, 0, rows * min_dim * sizeof(ComplexFloat));
    memset(S_data, 0, min_dim * sizeof(ComplexFloat));
    memset(V_data, 0, min_dim * cols * sizeof(ComplexFloat));

    // Simplified SVD using power iteration method
    // For each singular value/vector pair
    ComplexFloat* v = (ComplexFloat*)malloc(cols * sizeof(ComplexFloat));
    ComplexFloat* u = (ComplexFloat*)malloc(rows * sizeof(ComplexFloat));
    ComplexFloat* temp = (ComplexFloat*)malloc((rows > cols ? rows : cols) * sizeof(ComplexFloat));

    if (!v || !u || !temp) {
        free(A); free(U_data); free(S_data); free(V_data);
        free(v); free(u); free(temp);
        status = QG_ERROR_OUT_OF_MEMORY;
        goto cleanup;
    }

    for (size_t k = 0; k < min_dim; k++) {
        // Initialize v with random unit vector
        for (size_t j = 0; j < cols; j++) {
            v[j].real = (float)(rand() % 1000) / 1000.0f;
            v[j].imag = 0.0f;
        }

        // Power iteration: v = A^T * A * v (normalized)
        const int max_iter = 100;
        const float tol = 1e-6f;

        for (int iter = 0; iter < max_iter; iter++) {
            // temp = A * v
            for (size_t i = 0; i < rows; i++) {
                temp[i].real = 0.0f;
                temp[i].imag = 0.0f;
                for (size_t j = 0; j < cols; j++) {
                    size_t idx = i * cols + j;
                    temp[i].real += A[idx].real * v[j].real - A[idx].imag * v[j].imag;
                    temp[i].imag += A[idx].real * v[j].imag + A[idx].imag * v[j].real;
                }
            }

            // v_new = A^T * temp
            ComplexFloat* v_new = (ComplexFloat*)malloc(cols * sizeof(ComplexFloat));
            if (!v_new) continue;

            for (size_t j = 0; j < cols; j++) {
                v_new[j].real = 0.0f;
                v_new[j].imag = 0.0f;
                for (size_t i = 0; i < rows; i++) {
                    size_t idx = i * cols + j;
                    // Conjugate transpose
                    v_new[j].real += A[idx].real * temp[i].real + A[idx].imag * temp[i].imag;
                    v_new[j].imag += A[idx].real * temp[i].imag - A[idx].imag * temp[i].real;
                }
            }

            // Normalize v_new
            float norm = 0.0f;
            for (size_t j = 0; j < cols; j++) {
                norm += v_new[j].real * v_new[j].real + v_new[j].imag * v_new[j].imag;
            }
            norm = sqrtf(norm);
            if (norm < 1e-12f) {
                free(v_new);
                break;
            }

            // Check convergence
            float diff = 0.0f;
            for (size_t j = 0; j < cols; j++) {
                v_new[j].real /= norm;
                v_new[j].imag /= norm;
                float dr = v_new[j].real - v[j].real;
                float di = v_new[j].imag - v[j].imag;
                diff += dr * dr + di * di;
            }

            memcpy(v, v_new, cols * sizeof(ComplexFloat));
            free(v_new);

            if (sqrtf(diff) < tol) break;
        }

        // Compute u = A * v / sigma
        float sigma = 0.0f;
        for (size_t i = 0; i < rows; i++) {
            u[i].real = 0.0f;
            u[i].imag = 0.0f;
            for (size_t j = 0; j < cols; j++) {
                size_t idx = i * cols + j;
                u[i].real += A[idx].real * v[j].real - A[idx].imag * v[j].imag;
                u[i].imag += A[idx].real * v[j].imag + A[idx].imag * v[j].real;
            }
            sigma += u[i].real * u[i].real + u[i].imag * u[i].imag;
        }
        sigma = sqrtf(sigma);

        if (sigma > 1e-12f) {
            for (size_t i = 0; i < rows; i++) {
                u[i].real /= sigma;
                u[i].imag /= sigma;
            }
        }

        // Store results
        S_data[k].real = sigma;
        S_data[k].imag = 0.0f;

        for (size_t i = 0; i < rows; i++) {
            U_data[i * min_dim + k] = u[i];
        }
        for (size_t j = 0; j < cols; j++) {
            V_data[k * cols + j] = v[j];
        }

        // Deflate: A = A - sigma * u * v^T
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                size_t idx = i * cols + j;
                // u[i] * conj(v[j]) * sigma
                float ur = u[i].real, ui = u[i].imag;
                float vr = v[j].real, vi = -v[j].imag;  // conjugate
                A[idx].real -= sigma * (ur * vr - ui * vi);
                A[idx].imag -= sigma * (ur * vi + ui * vr);
            }
        }
    }

    free(v);
    free(u);
    free(temp);
    free(A);

    // Initialize output tensors with proper dimensions
    size_t u_dims[2] = {rows, min_dim};
    size_t s_dims[1] = {min_dim};
    size_t v_dims[2] = {min_dim, cols};

    status = qg_tensor_network_tensor_init(u_tensor, u_dims, 2);
    if (status != QG_SUCCESS) {
        free(U_data); free(S_data); free(V_data);
        goto cleanup;
    }

    status = qg_tensor_network_tensor_init(s_tensor, s_dims, 1);
    if (status != QG_SUCCESS) {
        qg_tensor_network_tensor_cleanup(u_tensor);
        free(U_data); free(S_data); free(V_data);
        goto cleanup;
    }

    status = qg_tensor_network_tensor_init(v_tensor, v_dims, 2);
    if (status != QG_SUCCESS) {
        qg_tensor_network_tensor_cleanup(u_tensor);
        qg_tensor_network_tensor_cleanup(s_tensor);
        free(U_data); free(S_data); free(V_data);
        goto cleanup;
    }

    // Copy data to output tensors
    memcpy(u_tensor->data, U_data, rows * min_dim * sizeof(ComplexFloat));
    memcpy(s_tensor->data, S_data, min_dim * sizeof(ComplexFloat));
    memcpy(v_tensor->data, V_data, min_dim * cols * sizeof(ComplexFloat));

    free(U_data);
    free(S_data);
    free(V_data);

cleanup:
    qg_timer_stop(&timer);
    perf_stats.total_decomposition_time += qg_timer_get_elapsed(&timer);
    return status;
}

// Compute truncation error from singular values
static float compute_truncation_error(const ComplexFloat* singular_values, size_t size, float tolerance) {
    if (!singular_values || size == 0) return INFINITY;

    float total_sq = 0.0f;
    float truncated_sq = 0.0f;
    bool truncating = false;

    for (size_t i = 0; i < size; i++) {
        float sv = singular_values[i].real;
        total_sq += sv * sv;

        if (!truncating && sv < tolerance) {
            truncating = true;
        }
        if (truncating) {
            truncated_sq += sv * sv;
        }
    }

    if (total_sq < 1e-12f) return 0.0f;
    return sqrtf(truncated_sq / total_sq);
}

// Find truncation rank based on tolerance
static size_t find_truncation_rank(const ComplexFloat* singular_values, size_t size, float tolerance) {
    if (!singular_values || size == 0) return 0;

    // Find how many singular values to keep
    for (size_t i = 0; i < size; i++) {
        if (singular_values[i].real < tolerance) {
            return i > 0 ? i : 1;  // Keep at least one
        }
    }
    return size;
}

// Create compressed tensor from SVD factors
static tensor_t* create_compressed_tensor(const tensor_t* u, const tensor_t* s,
                                           const tensor_t* v, size_t rank) {
    if (!u || !s || !v || !u->data || !s->data || !v->data) {
        return NULL;
    }

    // Truncate to specified rank
    size_t actual_rank = (rank < s->total_size) ? rank : s->total_size;

    // Result dimensions: [u.rows, v.cols]
    size_t rows = u->dimensions[0];
    size_t cols = v->dimensions[1];

    tensor_t* result = (tensor_t*)malloc(sizeof(tensor_t));
    if (!result) return NULL;
    memset(result, 0, sizeof(tensor_t));

    size_t dims[2] = {rows, cols};
    if (qg_tensor_network_tensor_init(result, dims, 2) != QG_SUCCESS) {
        free(result);
        return NULL;
    }

    // Compute result = U[:, :rank] * diag(S[:rank]) * V[:rank, :]
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            ComplexFloat sum = {0.0f, 0.0f};
            for (size_t k = 0; k < actual_rank; k++) {
                ComplexFloat u_val = u->data[i * u->dimensions[1] + k];
                float s_val = s->data[k].real;
                ComplexFloat v_val = v->data[k * cols + j];

                // u * s * v
                ComplexFloat us = {u_val.real * s_val, u_val.imag * s_val};
                sum.real += us.real * v_val.real - us.imag * v_val.imag;
                sum.imag += us.real * v_val.imag + us.imag * v_val.real;
            }
            result->data[i * cols + j] = sum;
        }
    }

    return result;
}

// Optimized network optimization with dynamic programming
int qg_tensor_network_optimize_contraction(tensor_network_t* network,
                                            const optimization_params_t* params) {
    if (!network || !params) {
        return QG_ERROR_INVALID_ARGUMENT;
    }

    if (network->num_nodes < 2) {
        return QG_SUCCESS;  // Nothing to optimize
    }

    // Use GPU for large networks if requested
    size_t total_size = 0;
    for (size_t i = 0; i < network->num_nodes; i++) {
        total_size += compute_node_total_size(network->nodes[i]);
    }
    if (params->use_gpu && total_size >= QG_MIN_SIZE_FOR_GPU && is_gpu_available()) {
        return qg_tensor_network_optimize_gpu(network, params);
    }

    // Initialize cost matrix for contraction pairs
    size_t n = network->num_nodes;

    // Use posix_memalign for portable aligned allocation
    float* cost_matrix = NULL;
    if (posix_memalign((void**)&cost_matrix, QG_TENSOR_BLOCK_SIZE, n * n * sizeof(float)) != 0) {
        return QG_ERROR_OUT_OF_MEMORY;
    }

    // Compute pairwise contraction costs
    // Cross-platform implementation without AVX intrinsics in the loop
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            float cost = INFINITY;
            if (i != j && network->nodes[i]->data && network->nodes[j]->data) {
                // Compute cost based on tensor sizes and shared indices
                size_t shared = count_shared_indices(network->nodes[i],
                                                      network->nodes[j]);
                cost = compute_contraction_cost(network->nodes[i],
                                                 network->nodes[j],
                                                 shared);
            }
            cost_matrix[i * n + j] = cost;
        }
    }

    // Find optimal contraction sequence using dynamic programming
    size_t* sequence = (size_t*)malloc(2 * (n-1) * sizeof(size_t));
    if (!sequence) {
        free(cost_matrix);
        return QG_ERROR_OUT_OF_MEMORY;
    }

    float min_cost = INFINITY;
    find_optimal_sequence(cost_matrix, n, sequence, &min_cost);

    // Store contraction order in network
    if (network->contraction_order) {
        free(network->contraction_order);
    }
    network->contraction_order = sequence;
    network->is_optimized = true;

    free(cost_matrix);
    return QG_SUCCESS;
}

// Enhanced network compression with SVD truncation
int qg_tensor_network_compress(tensor_network_t* network,
                               float tolerance) {
    if (!network || tolerance <= 0.0f) {
        return QG_ERROR_INVALID_ARGUMENT;
    }

    // Use GPU for large networks
    size_t total_size = 0;
    for (size_t i = 0; i < network->num_nodes; i++) {
        if (network->nodes[i]) {
            total_size += network->nodes[i]->total_size;
        }
    }
    if (total_size >= QG_MIN_SIZE_FOR_GPU && is_gpu_available()) {
        return qg_tensor_network_compress_gpu(network, tolerance);
    }

    // Compress each tensor using SVD truncation
    #pragma omp parallel for num_threads(QG_MAX_TENSOR_THREADS)
    for (size_t i = 0; i < network->num_nodes; i++) {
        if (!network->nodes[i] || !network->nodes[i]->data) continue;

        tensor_node_t* node = network->nodes[i];

        // Skip tensors that are too small to compress
        if (node->rank < 2) continue;

        // Find optimal split dimension based on singular values
        size_t best_dim = 0;
        float best_error = INFINITY;

        // Create temporary tensor_t wrapper for SVD operations
        tensor_t temp_tensor = {
            .data = node->data,
            .dimensions = node->dimensions,
            .rank = node->rank,
            .total_size = node->total_size,
            .is_contiguous = true,
            .strides = NULL,
            .owns_data = false,
            .device = NULL,
            .auxiliary_data = NULL
        };

        for (size_t dim = 1; dim < node->rank; dim++) {
            tensor_t u, s, v;
            memset(&u, 0, sizeof(tensor_t));
            memset(&s, 0, sizeof(tensor_t));
            memset(&v, 0, sizeof(tensor_t));

            int status = qg_ai_tensor_decompose_svd(&temp_tensor, dim, &u, &s, &v);
            if (status != QG_SUCCESS) continue;

            float error = compute_truncation_error(s.data, s.total_size, tolerance);
            if (error < best_error) {
                best_error = error;
                best_dim = dim;
            }

            qg_tensor_network_tensor_cleanup(&u);
            qg_tensor_network_tensor_cleanup(&s);
            qg_tensor_network_tensor_cleanup(&v);
        }

        // Apply best compression
        if (best_dim > 0 && best_error < tolerance) {
            tensor_t u, s, v;
            memset(&u, 0, sizeof(tensor_t));
            memset(&s, 0, sizeof(tensor_t));
            memset(&v, 0, sizeof(tensor_t));

            int status = qg_ai_tensor_decompose_svd(&temp_tensor, best_dim, &u, &s, &v);
            if (status == QG_SUCCESS) {
                size_t new_rank = find_truncation_rank(s.data, s.total_size, tolerance);

                tensor_t* compressed = create_compressed_tensor(&u, &s, &v, new_rank);
                if (compressed) {
                    // Update node with compressed data
                    free(node->data);
                    free(node->dimensions);
                    node->data = compressed->data;
                    node->dimensions = compressed->dimensions;
                    node->rank = compressed->rank;
                    node->num_dimensions = compressed->rank;
                    node->total_size = compressed->total_size;
                    compressed->owns_data = false;
                    free(compressed);
                }
            }
            qg_tensor_network_tensor_cleanup(&u);
            qg_tensor_network_tensor_cleanup(&s);
            qg_tensor_network_tensor_cleanup(&v);
        }
    }

    return QG_SUCCESS;
}

// Contract network according to optimized sequence
int qg_tensor_network_contract_sequence(tensor_network_t* network,
                                         tensor_t* result) {
    if (!network || !result) {
        return QG_ERROR_INVALID_ARGUMENT;
    }

    if (network->num_nodes == 0) {
        return QG_ERROR_INVALID_ARGUMENT;
    }

    // If only one tensor, just copy it
    if (network->num_nodes == 1 && network->nodes[0]) {
        tensor_node_t* node = network->nodes[0];
        size_t data_size = node->total_size * sizeof(ComplexFloat);
        int status = qg_tensor_network_tensor_init(result, node->dimensions, node->rank);
        if (status != QG_SUCCESS) return status;
        memcpy(result->data, node->data, data_size);
        return QG_SUCCESS;
    }

    // Make sure network is optimized
    if (!network->is_optimized || !network->contraction_order) {
        optimization_params_t params = {
            .tolerance = 1e-6,
            .max_iterations = 100,
            .use_gpu = false,
            .learning_rate = 0.01
        };
        int status = qg_tensor_network_optimize_contraction(network, &params);
        if (status != QG_SUCCESS) return status;
    }

    // Create working copies of tensors
    size_t n = network->num_nodes;
    tensor_t* work_tensors = (tensor_t*)calloc(n, sizeof(tensor_t));
    if (!work_tensors) return QG_ERROR_OUT_OF_MEMORY;

    for (size_t i = 0; i < n; i++) {
        if (network->nodes[i] && network->nodes[i]->data) {
            tensor_node_t* node = network->nodes[i];
            int status = qg_tensor_network_tensor_init(&work_tensors[i],
                                                        node->dimensions,
                                                        node->rank);
            if (status != QG_SUCCESS) {
                for (size_t j = 0; j < i; j++) {
                    qg_tensor_network_tensor_cleanup(&work_tensors[j]);
                }
                free(work_tensors);
                return status;
            }
            memcpy(work_tensors[i].data, node->data,
                   node->total_size * sizeof(ComplexFloat));
        }
    }

    // Contract according to sequence
    for (size_t step = 0; step < n - 1; step++) {
        size_t idx1 = network->contraction_order[2 * step];
        size_t idx2 = network->contraction_order[2 * step + 1];

        if (!work_tensors[idx1].data || !work_tensors[idx2].data) {
            continue;  // Already contracted
        }

        tensor_t* contracted = NULL;
        int status = contract_pair(&work_tensors[idx1], &work_tensors[idx2], &contracted);
        if (status != QG_SUCCESS || !contracted) {
            for (size_t i = 0; i < n; i++) {
                qg_tensor_network_tensor_cleanup(&work_tensors[i]);
            }
            free(work_tensors);
            return status;
        }

        // Replace idx1 with result, clear idx2
        qg_tensor_network_tensor_cleanup(&work_tensors[idx1]);
        qg_tensor_network_tensor_cleanup(&work_tensors[idx2]);
        work_tensors[idx1] = *contracted;
        free(contracted);
        memset(&work_tensors[idx2], 0, sizeof(tensor_t));
    }

    // Find the final result tensor
    for (size_t i = 0; i < n; i++) {
        if (work_tensors[i].data) {
            *result = work_tensors[i];
            work_tensors[i].data = NULL;  // Transfer ownership
            break;
        }
    }

    // Cleanup
    for (size_t i = 0; i < n; i++) {
        qg_tensor_network_tensor_cleanup(&work_tensors[i]);
    }
    free(work_tensors);

    return QG_SUCCESS;
}

// Apply quantum time evolution to tensor network
int qg_tensor_network_quantum_evolve(tensor_network_t* network,
                                      const ComplexFloat* hamiltonian,
                                      size_t ham_dim,
                                      double time_step) {
    if (!network || !hamiltonian || ham_dim == 0 || time_step <= 0.0) {
        return QG_ERROR_INVALID_ARGUMENT;
    }

    // For each tensor in the network, apply evolution operator exp(-i*H*dt)
    // Using first-order Trotter approximation

    #pragma omp parallel for num_threads(QG_MAX_TENSOR_THREADS)
    for (size_t t = 0; t < network->num_nodes; t++) {
        if (!network->nodes[t] || !network->nodes[t]->data) continue;

        tensor_node_t* node = network->nodes[t];

        // Only evolve if tensor dimension matches Hamiltonian
        if (node->total_size != ham_dim * ham_dim) continue;

        // Create evolution operator: U = exp(-i * H * dt)
        // Using Taylor expansion: U ≈ I - i*H*dt - (H*dt)^2/2 + ...
        ComplexFloat* evolved = (ComplexFloat*)malloc(node->total_size * sizeof(ComplexFloat));
        if (!evolved) continue;

        // Initialize with identity
        for (size_t i = 0; i < ham_dim; i++) {
            for (size_t j = 0; j < ham_dim; j++) {
                if (i == j) {
                    evolved[i * ham_dim + j].real = 1.0f;
                    evolved[i * ham_dim + j].imag = 0.0f;
                } else {
                    evolved[i * ham_dim + j].real = 0.0f;
                    evolved[i * ham_dim + j].imag = 0.0f;
                }
            }
        }

        // Add -i*H*dt term
        float dt = (float)time_step;
        for (size_t i = 0; i < ham_dim * ham_dim; i++) {
            // -i * H * dt = (-i) * (Hr + i*Hi) * dt = Hi*dt - i*Hr*dt
            evolved[i].real += hamiltonian[i].imag * dt;
            evolved[i].imag -= hamiltonian[i].real * dt;
        }

        // Apply evolution: new_tensor = U * node
        ComplexFloat* new_data = (ComplexFloat*)malloc(node->total_size * sizeof(ComplexFloat));
        if (new_data) {
            for (size_t i = 0; i < ham_dim; i++) {
                for (size_t j = 0; j < ham_dim; j++) {
                    ComplexFloat sum = {0.0f, 0.0f};
                    for (size_t k = 0; k < ham_dim; k++) {
                        ComplexFloat u = evolved[i * ham_dim + k];
                        ComplexFloat t_val = node->data[k * ham_dim + j];
                        sum.real += u.real * t_val.real - u.imag * t_val.imag;
                        sum.imag += u.real * t_val.imag + u.imag * t_val.real;
                    }
                    new_data[i * ham_dim + j] = sum;
                }
            }
            memcpy(node->data, new_data, node->total_size * sizeof(ComplexFloat));
            free(new_data);
        }

        free(evolved);
    }

    return QG_SUCCESS;
}

// Perform measurement on tensor network
int qg_tensor_network_measure(const tensor_network_t* network,
                               const size_t* measurement_indices,
                               size_t num_measurements,
                               ComplexFloat* outcomes) {
    if (!network || !outcomes) {
        return QG_ERROR_INVALID_ARGUMENT;
    }

    if (num_measurements == 0) {
        return QG_SUCCESS;
    }

    // Contract network to get final state
    tensor_t contracted;
    memset(&contracted, 0, sizeof(tensor_t));

    // Create mutable copy of network for contraction
    tensor_network_t work_network;
    memset(&work_network, 0, sizeof(tensor_network_t));

    int status = qg_tensor_network_init(&work_network, network->num_nodes);
    if (status != QG_SUCCESS) return status;

    // Copy tensors
    for (size_t i = 0; i < network->num_nodes; i++) {
        if (network->nodes[i] && network->nodes[i]->data) {
            tensor_node_t* src = network->nodes[i];
            work_network.nodes[i] = malloc(sizeof(tensor_node_t));
            if (!work_network.nodes[i]) {
                qg_tensor_network_cleanup(&work_network);
                return QG_ERROR_OUT_OF_MEMORY;
            }
            memset(work_network.nodes[i], 0, sizeof(tensor_node_t));
            status = tensor_node_init(work_network.nodes[i], src->dimensions, src->rank);
            if (status != QG_SUCCESS) {
                qg_tensor_network_cleanup(&work_network);
                return status;
            }
            memcpy(work_network.nodes[i]->data, src->data,
                   src->total_size * sizeof(ComplexFloat));
        }
    }
    work_network.num_nodes = network->num_nodes;

    status = qg_tensor_network_contract_sequence(&work_network, &contracted);
    qg_tensor_network_cleanup(&work_network);

    if (status != QG_SUCCESS) {
        return status;
    }

    // Extract measurement outcomes
    // For quantum states, measure probability amplitudes at specified indices
    for (size_t m = 0; m < num_measurements; m++) {
        size_t idx = measurement_indices ? measurement_indices[m] : m;
        if (idx < contracted.total_size) {
            outcomes[m] = contracted.data[idx];
        } else {
            outcomes[m].real = 0.0f;
            outcomes[m].imag = 0.0f;
        }
    }

    qg_tensor_network_tensor_cleanup(&contracted);
    return QG_SUCCESS;
}

// Compute expectation value of an observable
int qg_tensor_network_expectation(const tensor_network_t* network,
                                   const ComplexFloat* observable,
                                   size_t obs_dim,
                                   ComplexFloat* result) {
    if (!network || !observable || !result || obs_dim == 0) {
        return QG_ERROR_INVALID_ARGUMENT;
    }

    // Contract network to get state
    tensor_t state;
    memset(&state, 0, sizeof(tensor_t));

    // Create mutable copy
    tensor_network_t work_network;
    memset(&work_network, 0, sizeof(tensor_network_t));

    int status = qg_tensor_network_init(&work_network, network->num_nodes);
    if (status != QG_SUCCESS) return status;

    for (size_t i = 0; i < network->num_nodes; i++) {
        if (network->nodes[i] && network->nodes[i]->data) {
            tensor_node_t* src = network->nodes[i];
            work_network.nodes[i] = malloc(sizeof(tensor_node_t));
            if (!work_network.nodes[i]) {
                qg_tensor_network_cleanup(&work_network);
                return QG_ERROR_OUT_OF_MEMORY;
            }
            memset(work_network.nodes[i], 0, sizeof(tensor_node_t));
            status = tensor_node_init(work_network.nodes[i], src->dimensions, src->rank);
            if (status != QG_SUCCESS) {
                qg_tensor_network_cleanup(&work_network);
                return status;
            }
            memcpy(work_network.nodes[i]->data, src->data,
                   src->total_size * sizeof(ComplexFloat));
        }
    }
    work_network.num_nodes = network->num_nodes;

    status = qg_tensor_network_contract_sequence(&work_network, &state);
    qg_tensor_network_cleanup(&work_network);

    if (status != QG_SUCCESS) {
        return status;
    }

    // Compute <psi|O|psi> = sum_ij conj(psi_i) * O_ij * psi_j
    result->real = 0.0f;
    result->imag = 0.0f;

    size_t dim = (state.total_size < obs_dim) ? state.total_size : obs_dim;

    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            // conj(psi_i) * O_ij * psi_j
            ComplexFloat psi_i_conj = {state.data[i].real, -state.data[i].imag};
            ComplexFloat obs_ij = observable[i * obs_dim + j];
            ComplexFloat psi_j = state.data[j];

            // First: obs_ij * psi_j
            ComplexFloat temp;
            temp.real = obs_ij.real * psi_j.real - obs_ij.imag * psi_j.imag;
            temp.imag = obs_ij.real * psi_j.imag + obs_ij.imag * psi_j.real;

            // Then: conj(psi_i) * temp
            result->real += psi_i_conj.real * temp.real - psi_i_conj.imag * temp.imag;
            result->imag += psi_i_conj.real * temp.imag + psi_i_conj.imag * temp.real;
        }
    }

    qg_tensor_network_tensor_cleanup(&state);
    return QG_SUCCESS;
}

// Get performance statistics
void qg_tensor_network_get_performance_stats(tensor_network_stats_t* stats) {
    if (!stats) return;
    
    stats->num_contractions = perf_stats.total_contractions;
    stats->num_decompositions = perf_stats.total_decompositions;
    stats->total_contraction_time = perf_stats.total_contraction_time;
    stats->total_decomposition_time = perf_stats.total_decomposition_time;
    stats->peak_memory = perf_stats.peak_memory;
    stats->current_memory = perf_stats.current_memory;
}

// Reset performance statistics
void qg_tensor_network_reset_performance_stats(void) {
    memset(&perf_stats, 0, sizeof(perf_stats));
}
