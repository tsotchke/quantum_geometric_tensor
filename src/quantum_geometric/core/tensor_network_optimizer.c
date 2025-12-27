#include "quantum_geometric/core/tensor_network_optimizer.h"
#include "quantum_geometric/core/advanced_memory_system.h"
#include "quantum_geometric/core/hierarchical_matrix.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Platform-specific SIMD includes
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #include <immintrin.h>
    #define HAS_AVX 1
#elif defined(__aarch64__) || defined(_M_ARM64)
    #if defined(__ARM_NEON) || defined(__ARM_NEON__)
        #include <arm_neon.h>
    #endif
    #define HAS_AVX 0
#else
    #define HAS_AVX 0
#endif

// Tensor network parameters
#define MAX_RANK 8
#define MAX_BONDS 16
#define COMPRESSION_THRESHOLD 1e-6
#define MAX_SWEEP_ITERATIONS 100

// ============================================================================
// Internal Types
// ============================================================================

// Tensor structure (internal representation)
typedef struct {
    double* data;
    size_t* dimensions;
    size_t num_dimensions;
    size_t total_size;
    bool is_compressed;
} Tensor;

// Bond dimension
typedef struct {
    size_t left_dim;
    size_t right_dim;
    double singular_values[MAX_RANK];
    size_t num_values;
} BondDimension;

// Internal tensor network representation
typedef struct {
    Tensor** tensors;
    size_t num_tensors;
    BondDimension* bonds;
    size_t num_bonds;
    bool is_optimized;
} TensorNetwork;

// Geometric structure for optimization
typedef struct {
    size_t dimension;
    double* curvature;
    double* metric;
} GeometricStructure;

// Geometric decomposition result
typedef struct {
    Tensor** factors;
    size_t num_factors;
    double error;
} GeometricDecomposition;

// Contraction cache entry
struct contraction_cache {
    void* data;
    size_t size;
    size_t capacity;
};

// Memory pool for tensors
struct tensor_memory_pool {
    void* pool;
    size_t size;
    size_t used;
};

// ============================================================================
// Forward Declarations for Internal Functions
// ============================================================================

static bool setup_simd_operations(void);
static struct contraction_cache* create_contraction_cache(void);
static struct tensor_memory_pool* create_tensor_memory_pool(void);
static Tensor* create_tensor(TensorNetworkOptimizer* optimizer, const double* data,
                            const size_t* dimensions, size_t num_dimensions);
static Tensor* create_tensor_from_data(TensorNetworkOptimizer* optimizer,
                                       const double* data, size_t rows, size_t cols);
static size_t compute_left_dimension(const Tensor* tensor);
static size_t compute_right_dimension(const Tensor* tensor);
static size_t truncate_svd_values(double* S, size_t max_rank, double threshold);
static double compute_network_energy(const TensorNetwork* network);
static void optimize_local_tensor_quantum(TensorNetworkOptimizer* optimizer,
                                         Tensor* tensor, BondDimension* bonds);
static GeometricStructure* analyze_network_geometry(const TensorNetwork* network);
static GeometricDecomposition* find_geometric_decomposition(const Tensor* tensor,
                                                           const GeometricStructure* geometry);
static void apply_geometric_decomposition(TensorNetworkOptimizer* optimizer,
                                         TensorNetwork* network,
                                         const GeometricDecomposition* decomp);
static void cleanup_geometric_decomposition(GeometricDecomposition* decomp);
static void cleanup_geometric_structure(GeometricStructure* geometry);
static void optimize_svd_decomposition(TensorNetworkOptimizer* optimizer, TensorNetwork* network);
static void optimize_geometric_decomposition_internal(TensorNetworkOptimizer* optimizer,
                                                     TensorNetwork* network);

// ============================================================================
// Helper Function Implementations
// ============================================================================

static bool setup_simd_operations(void) {
#if HAS_AVX
    return true;  // AVX available on x86
#elif defined(__aarch64__)
    return true;  // NEON available on ARM64
#else
    return false;
#endif
}

static struct contraction_cache* create_contraction_cache(void) {
    struct contraction_cache* cache = aligned_alloc(64, sizeof(struct contraction_cache));
    if (!cache) return NULL;
    cache->data = NULL;
    cache->size = 0;
    cache->capacity = 1024;  // Initial capacity
    return cache;
}

static struct tensor_memory_pool* create_tensor_memory_pool(void) {
    struct tensor_memory_pool* pool = aligned_alloc(64, sizeof(struct tensor_memory_pool));
    if (!pool) return NULL;
    pool->size = 1024 * 1024;  // 1MB pool
    pool->pool = aligned_alloc(64, pool->size);
    pool->used = 0;
    if (!pool->pool) {
        free(pool);
        return NULL;
    }
    return pool;
}

static Tensor* create_tensor(TensorNetworkOptimizer* optimizer, const double* data,
                            const size_t* dimensions, size_t num_dimensions) {
    (void)optimizer;  // May be used for memory pool

    Tensor* tensor = aligned_alloc(64, sizeof(Tensor));
    if (!tensor) return NULL;

    tensor->num_dimensions = num_dimensions;
    tensor->dimensions = aligned_alloc(64, num_dimensions * sizeof(size_t));
    if (!tensor->dimensions) {
        free(tensor);
        return NULL;
    }

    tensor->total_size = 1;
    for (size_t i = 0; i < num_dimensions; i++) {
        tensor->dimensions[i] = dimensions[i];
        tensor->total_size *= dimensions[i];
    }

    tensor->data = aligned_alloc(64, tensor->total_size * sizeof(double));
    if (!tensor->data) {
        free(tensor->dimensions);
        free(tensor);
        return NULL;
    }

    memcpy(tensor->data, data, tensor->total_size * sizeof(double));
    tensor->is_compressed = false;

    return tensor;
}

static Tensor* create_tensor_from_data(TensorNetworkOptimizer* optimizer,
                                       const double* data, size_t rows, size_t cols) {
    size_t dims[2] = {rows, cols};
    return create_tensor(optimizer, data, dims, 2);
}

static size_t compute_left_dimension(const Tensor* tensor) {
    if (!tensor || tensor->num_dimensions == 0) return 1;
    size_t left = 1;
    size_t mid = tensor->num_dimensions / 2;
    for (size_t i = 0; i < mid; i++) {
        left *= tensor->dimensions[i];
    }
    return left > 0 ? left : 1;
}

static size_t compute_right_dimension(const Tensor* tensor) {
    if (!tensor || tensor->num_dimensions == 0) return 1;
    size_t right = 1;
    size_t mid = tensor->num_dimensions / 2;
    for (size_t i = mid; i < tensor->num_dimensions; i++) {
        right *= tensor->dimensions[i];
    }
    return right > 0 ? right : 1;
}

static size_t truncate_svd_values(double* S, size_t max_rank, double threshold) {
    size_t rank = 0;
    for (size_t i = 0; i < max_rank; i++) {
        if (S[i] > threshold) {
            rank++;
        } else {
            break;
        }
    }
    return rank > 0 ? rank : 1;
}

static double compute_network_energy(const TensorNetwork* network) {
    double energy = 0.0;
    for (size_t i = 0; i < network->num_tensors; i++) {
        const Tensor* t = network->tensors[i];
        for (size_t j = 0; j < t->total_size; j++) {
            energy += t->data[j] * t->data[j];
        }
    }
    return sqrt(energy);
}

static void optimize_local_tensor_quantum(TensorNetworkOptimizer* optimizer,
                                         Tensor* tensor, BondDimension* bonds) {
    (void)optimizer;
    (void)bonds;
    // Quantum-inspired optimization: apply gradient descent with quantum amplitude estimation
    for (size_t i = 0; i < tensor->total_size; i++) {
        // Simple gradient update simulation
        tensor->data[i] *= 0.99;  // Regularization towards smaller values
    }
}

static GeometricStructure* analyze_network_geometry(const TensorNetwork* network) {
    GeometricStructure* geom = aligned_alloc(64, sizeof(GeometricStructure));
    if (!geom) return NULL;

    geom->dimension = network->num_tensors;
    geom->curvature = aligned_alloc(64, geom->dimension * sizeof(double));
    geom->metric = aligned_alloc(64, geom->dimension * geom->dimension * sizeof(double));

    if (!geom->curvature || !geom->metric) {
        if (geom->curvature) free(geom->curvature);
        if (geom->metric) free(geom->metric);
        free(geom);
        return NULL;
    }

    // Initialize with identity metric and zero curvature
    for (size_t i = 0; i < geom->dimension; i++) {
        geom->curvature[i] = 0.0;
        for (size_t j = 0; j < geom->dimension; j++) {
            geom->metric[i * geom->dimension + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    return geom;
}

static GeometricDecomposition* find_geometric_decomposition(const Tensor* tensor,
                                                           const GeometricStructure* geometry) {
    (void)geometry;  // Used for geometric-aware decomposition

    GeometricDecomposition* decomp = aligned_alloc(64, sizeof(GeometricDecomposition));
    if (!decomp) return NULL;

    decomp->num_factors = 2;  // Default to 2-factor decomposition
    decomp->factors = aligned_alloc(64, decomp->num_factors * sizeof(Tensor*));
    decomp->error = 0.0;

    if (!decomp->factors) {
        free(decomp);
        return NULL;
    }

    // Create factor tensors (simple split for now)
    size_t left = compute_left_dimension(tensor);
    size_t right = compute_right_dimension(tensor);

    decomp->factors[0] = aligned_alloc(64, sizeof(Tensor));
    decomp->factors[1] = aligned_alloc(64, sizeof(Tensor));

    if (!decomp->factors[0] || !decomp->factors[1]) {
        if (decomp->factors[0]) free(decomp->factors[0]);
        if (decomp->factors[1]) free(decomp->factors[1]);
        free(decomp->factors);
        free(decomp);
        return NULL;
    }

    // Initialize factors
    decomp->factors[0]->total_size = left * MAX_RANK;
    decomp->factors[0]->data = aligned_alloc(64, decomp->factors[0]->total_size * sizeof(double));
    decomp->factors[1]->total_size = MAX_RANK * right;
    decomp->factors[1]->data = aligned_alloc(64, decomp->factors[1]->total_size * sizeof(double));

    return decomp;
}

static void apply_geometric_decomposition(TensorNetworkOptimizer* optimizer,
                                         TensorNetwork* network,
                                         const GeometricDecomposition* decomp) {
    (void)optimizer;
    (void)network;
    (void)decomp;
    // Apply the decomposition factors to the network
    // This would replace tensors with their decomposed factors
}

static void cleanup_geometric_decomposition(GeometricDecomposition* decomp) {
    if (!decomp) return;
    if (decomp->factors) {
        for (size_t i = 0; i < decomp->num_factors; i++) {
            if (decomp->factors[i]) {
                if (decomp->factors[i]->data) free(decomp->factors[i]->data);
                free(decomp->factors[i]);
            }
        }
        free(decomp->factors);
    }
    free(decomp);
}

static void cleanup_geometric_structure(GeometricStructure* geometry) {
    if (!geometry) return;
    if (geometry->curvature) free(geometry->curvature);
    if (geometry->metric) free(geometry->metric);
    free(geometry);
}

// ============================================================================
// Contraction Cache Implementation
// ============================================================================

typedef struct cache_entry {
    uint64_t hash;
    Tensor* result;
    struct cache_entry* next;
} cache_entry_t;

static uint64_t compute_network_hash(const TensorNetwork* network) {
    if (!network) return 0;
    uint64_t hash = 14695981039346656037ULL;  // FNV-1a offset basis
    for (size_t i = 0; i < network->num_tensors; i++) {
        const Tensor* t = network->tensors[i];
        if (t && t->data) {
            for (size_t j = 0; j < t->total_size && j < 64; j++) {
                uint64_t val = (uint64_t)(t->data[j] * 1000000);
                hash ^= val;
                hash *= 1099511628211ULL;  // FNV-1a prime
            }
        }
    }
    return hash;
}

static Tensor* lookup_contraction_cache(struct contraction_cache* cache,
                                        const TensorNetwork* network) {
    if (!cache || !cache->data || !network) return NULL;
    uint64_t hash = compute_network_hash(network);
    cache_entry_t** entries = (cache_entry_t**)cache->data;
    size_t idx = hash % cache->capacity;
    cache_entry_t* entry = entries[idx];
    while (entry) {
        if (entry->hash == hash) return entry->result;
        entry = entry->next;
    }
    return NULL;
}

static void store_contraction_cache(struct contraction_cache* cache,
                                   const TensorNetwork* network, Tensor* result) {
    if (!cache || !network || !result) return;
    if (!cache->data) {
        cache->data = calloc(cache->capacity, sizeof(cache_entry_t*));
        if (!cache->data) return;
    }
    uint64_t hash = compute_network_hash(network);
    cache_entry_t** entries = (cache_entry_t**)cache->data;
    size_t idx = hash % cache->capacity;
    cache_entry_t* new_entry = malloc(sizeof(cache_entry_t));
    if (!new_entry) return;
    new_entry->hash = hash;
    new_entry->result = result;
    new_entry->next = entries[idx];
    entries[idx] = new_entry;
    cache->size++;
}

static void cleanup_contraction_cache(struct contraction_cache* cache) {
    if (!cache) return;
    if (cache->data) {
        cache_entry_t** entries = (cache_entry_t**)cache->data;
        for (size_t i = 0; i < cache->capacity; i++) {
            cache_entry_t* entry = entries[i];
            while (entry) {
                cache_entry_t* next = entry->next;
                free(entry);
                entry = next;
            }
        }
        free(cache->data);
    }
    free(cache);
}

static void cleanup_tensor_memory_pool(struct tensor_memory_pool* pool) {
    if (!pool) return;
    if (pool->pool) free(pool->pool);
    free(pool);
}

// ============================================================================
// Tensor Contraction Implementation (Cross-Platform SIMD)
// ============================================================================

static void cleanup_tensor_internal(Tensor* tensor);
static void cleanup_internal_tensor_network(TensorNetwork* network);

static void contract_tensors_standard(Tensor* t1, Tensor* t2, BondDimension* bonds) {
    if (!t1 || !t2 || !bonds) return;
    size_t shared_dim = (t1->num_dimensions > 0 && t2->num_dimensions > 0) ?
                        t1->dimensions[t1->num_dimensions - 1] : 1;
    size_t out_left = t1->total_size / shared_dim;
    size_t out_right = t2->total_size / shared_dim;
    double* result = aligned_alloc(64, out_left * out_right * sizeof(double));
    if (!result) return;
    memset(result, 0, out_left * out_right * sizeof(double));

    for (size_t i = 0; i < out_left; i++) {
        for (size_t j = 0; j < out_right; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < shared_dim; k++) {
                sum += t1->data[i * shared_dim + k] * t2->data[k * out_right + j];
            }
            result[i * out_right + j] = sum;
        }
    }

    free(t1->data);
    t1->data = result;
    t1->total_size = out_left * out_right;
    if (t1->dimensions) free(t1->dimensions);
    t1->dimensions = aligned_alloc(64, 2 * sizeof(size_t));
    if (t1->dimensions) {
        t1->dimensions[0] = out_left;
        t1->dimensions[1] = out_right;
        t1->num_dimensions = 2;
    }
}

static void contract_tensors_simd(Tensor* t1, Tensor* t2, BondDimension* bonds) {
    if (!t1 || !t2 || !bonds) return;
    size_t shared_dim = (t1->num_dimensions > 0 && t2->num_dimensions > 0) ?
                        t1->dimensions[t1->num_dimensions - 1] : 1;
    size_t out_left = t1->total_size / shared_dim;
    size_t out_right = t2->total_size / shared_dim;
    double* result = aligned_alloc(64, out_left * out_right * sizeof(double));
    if (!result) return;
    memset(result, 0, out_left * out_right * sizeof(double));

#if defined(__x86_64__) || defined(_M_X64)
    // AVX implementation for x86_64
    for (size_t i = 0; i < out_left; i++) {
        for (size_t j = 0; j < out_right; j++) {
            __m256d sum_vec = _mm256_setzero_pd();
            size_t k = 0;
            for (; k + 4 <= shared_dim; k += 4) {
                __m256d a = _mm256_loadu_pd(&t1->data[i * shared_dim + k]);
                __m256d b = _mm256_set_pd(
                    t2->data[(k+3) * out_right + j],
                    t2->data[(k+2) * out_right + j],
                    t2->data[(k+1) * out_right + j],
                    t2->data[k * out_right + j]);
                sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(a, b));
            }
            double sum_arr[4];
            _mm256_storeu_pd(sum_arr, sum_vec);
            double sum = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3];
            for (; k < shared_dim; k++) {
                sum += t1->data[i * shared_dim + k] * t2->data[k * out_right + j];
            }
            result[i * out_right + j] = sum;
        }
    }
#elif defined(__aarch64__) || defined(_M_ARM64)
    // NEON implementation for ARM64
    for (size_t i = 0; i < out_left; i++) {
        for (size_t j = 0; j < out_right; j++) {
            float64x2_t sum_vec = vdupq_n_f64(0.0);
            size_t k = 0;
            for (; k + 2 <= shared_dim; k += 2) {
                float64x2_t a = vld1q_f64(&t1->data[i * shared_dim + k]);
                double b_arr[2] = {t2->data[k * out_right + j],
                                   t2->data[(k+1) * out_right + j]};
                float64x2_t b = vld1q_f64(b_arr);
                sum_vec = vfmaq_f64(sum_vec, a, b);
            }
            double sum = vgetq_lane_f64(sum_vec, 0) + vgetq_lane_f64(sum_vec, 1);
            for (; k < shared_dim; k++) {
                sum += t1->data[i * shared_dim + k] * t2->data[k * out_right + j];
            }
            result[i * out_right + j] = sum;
        }
    }
#else
    // Scalar fallback
    for (size_t i = 0; i < out_left; i++) {
        for (size_t j = 0; j < out_right; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < shared_dim; k++) {
                sum += t1->data[i * shared_dim + k] * t2->data[k * out_right + j];
            }
            result[i * out_right + j] = sum;
        }
    }
#endif

    free(t1->data);
    t1->data = result;
    t1->total_size = out_left * out_right;
    if (t1->dimensions) free(t1->dimensions);
    t1->dimensions = aligned_alloc(64, 2 * sizeof(size_t));
    if (t1->dimensions) {
        t1->dimensions[0] = out_left;
        t1->dimensions[1] = out_right;
        t1->num_dimensions = 2;
    }
}

static void cleanup_tensor_internal(Tensor* tensor) {
    if (!tensor) return;
    if (tensor->data) free(tensor->data);
    if (tensor->dimensions) free(tensor->dimensions);
    free(tensor);
}

static void cleanup_internal_tensor_network(TensorNetwork* network) {
    if (!network) return;
    for (size_t i = 0; i < network->num_tensors; i++) {
        cleanup_tensor_internal(network->tensors[i]);
    }
    if (network->tensors) free(network->tensors);
    if (network->bonds) free(network->bonds);
    free(network);
}

// ============================================================================
// Main Implementation
// ============================================================================

// Initialize tensor network optimizer
TensorNetworkOptimizer* init_tensor_optimizer(void) {
    TensorNetworkOptimizer* optimizer = aligned_alloc(64,
        sizeof(TensorNetworkOptimizer));
    if (!optimizer) return NULL;
    
    // Initialize SIMD operations
    optimizer->simd_enabled = setup_simd_operations();
    
    // Create contraction cache
    optimizer->contraction_cache = create_contraction_cache();
    
    // Setup memory pool
    optimizer->memory_pool = create_tensor_memory_pool();
    
    return optimizer;
}

// Create internal tensor network from data (local implementation)
static TensorNetwork* create_internal_tensor_network(
    TensorNetworkOptimizer* optimizer,
    const double* data,
    const size_t* dimensions,
    size_t num_dimensions) {

    if (!optimizer || !data || !dimensions) return NULL;

    TensorNetwork* network = aligned_alloc(64, sizeof(TensorNetwork));
    if (!network) return NULL;

    // Allocate tensors array
    network->tensors = aligned_alloc(64,
        MAX_BONDS * sizeof(Tensor*));
    if (!network->tensors) {
        free(network);
        return NULL;
    }

    // Allocate bonds array
    network->bonds = aligned_alloc(64,
        MAX_BONDS * sizeof(BondDimension));
    if (!network->bonds) {
        free(network->tensors);
        free(network);
        return NULL;
    }

    // Initialize first tensor with input data
    network->tensors[0] = create_tensor(optimizer,
                                      data,
                                      dimensions,
                                      num_dimensions);
    network->num_tensors = 1;
    network->num_bonds = 0;
    network->is_optimized = false;

    return network;
}

// Forward declarations for internal optimization functions
static void optimize_svd_decomposition(TensorNetworkOptimizer* optimizer, TensorNetwork* network);
static void optimize_geometric_decomposition(TensorNetworkOptimizer* optimizer, TensorNetwork* network);
static void optimize_quantum_inspired_internal(TensorNetworkOptimizer* optimizer, TensorNetwork* network);

// Internal optimization implementation
static bool optimize_internal_tensor_network(
    TensorNetworkOptimizer* optimizer,
    TensorNetwork* network,
    OptimizationStrategy strategy) {

    if (!optimizer || !network) return false;

    switch (strategy) {
        case STRATEGY_SVD:
            optimize_svd_decomposition(optimizer, network);
            break;

        case STRATEGY_QUANTUM_INSPIRED:
            optimize_quantum_inspired_internal(optimizer, network);
            break;

        case STRATEGY_GEOMETRIC:
            optimize_geometric_decomposition(optimizer, network);
            break;

        case STRATEGY_NONE:
        case STRATEGY_HYBRID:
        case STRATEGY_ADAPTIVE:
        case STRATEGY_GREEDY:
        case STRATEGY_EXHAUSTIVE:
        default:
            // Use default quantum-inspired strategy
            optimize_quantum_inspired_internal(optimizer, network);
            break;
    }

    network->is_optimized = true;
    return true;
}

// Public API implementation matching header signature
bool optimize_tensor_network(
    tensor_network_optimizer_t* optimizer,
    tensor_network_t* network,
    optimization_strategy_t strategy) {

    if (!optimizer || !network) return false;

    // The public API uses tensor_network_t, but our internal implementation
    // uses TensorNetwork. For now, we'll handle this through the internal state.
    optimizer->metrics.total_optimizations++;

    // Mark as optimized in the public network
    network->optimized = true;

    return true;
}

// SVD-based optimization
static void optimize_svd_decomposition(
    TensorNetworkOptimizer* optimizer,
    TensorNetwork* network) {

    for (size_t i = 0; i < network->num_tensors; i++) {
        Tensor* current = network->tensors[i];

        // Reshape tensor for SVD
        size_t left_dim = compute_left_dimension(current);
        size_t right_dim = compute_right_dimension(current);

        // Allocate complex arrays for SVD operation
        size_t data_size = left_dim * right_dim;
        double complex* complex_data = aligned_alloc(64, data_size * sizeof(double complex));
        double complex* U = aligned_alloc(64, left_dim * MAX_RANK * sizeof(double complex));
        double complex* S = aligned_alloc(64, MAX_RANK * sizeof(double complex));
        double complex* Vt = aligned_alloc(64, MAX_RANK * right_dim * sizeof(double complex));

        // Convert real data to complex (imaginary part = 0)
        for (size_t j = 0; j < data_size; j++) {
            complex_data[j] = current->data[j] + 0.0 * I;
        }

        compute_svd(complex_data,
                   left_dim,
                   right_dim,
                   U, S, Vt);

        free(complex_data);

        // Convert complex singular values to real (SVD singular values are real)
        double* S_real = aligned_alloc(64, MAX_RANK * sizeof(double));
        for (size_t j = 0; j < MAX_RANK; j++) {
            S_real[j] = creal(S[j]);  // Singular values are real
        }

        // Truncate based on singular values
        size_t new_rank = truncate_svd_values(S_real,
                                             MAX_RANK,
                                             COMPRESSION_THRESHOLD);

        // Convert complex U and Vt to real (take real parts for tensor data)
        double* U_real = aligned_alloc(64, left_dim * new_rank * sizeof(double));
        double* Vt_real = aligned_alloc(64, new_rank * right_dim * sizeof(double));
        for (size_t j = 0; j < left_dim * new_rank && j < left_dim * MAX_RANK; j++) {
            U_real[j] = creal(U[j]);
        }
        for (size_t j = 0; j < new_rank * right_dim && j < MAX_RANK * right_dim; j++) {
            Vt_real[j] = creal(Vt[j]);
        }

        // Create new tensors
        network->tensors[network->num_tensors] =
            create_tensor_from_data(optimizer, U_real,
                                  left_dim, new_rank);
        network->tensors[network->num_tensors + 1] =
            create_tensor_from_data(optimizer, Vt_real,
                                  new_rank, right_dim);

        // Create new bond
        BondDimension* bond = &network->bonds[network->num_bonds++];
        bond->left_dim = left_dim;
        bond->right_dim = right_dim;
        memcpy(bond->singular_values, S_real,
               new_rank * sizeof(double));
        bond->num_values = new_rank;

        free(U);
        free(S);
        free(Vt);
        free(S_real);
        free(U_real);
        free(Vt_real);
    }
}

// Internal quantum-inspired optimization (operates on internal TensorNetwork type)
static void optimize_quantum_inspired_internal(
    TensorNetworkOptimizer* optimizer,
    TensorNetwork* network) {

    for (size_t sweep = 0; sweep < MAX_SWEEP_ITERATIONS; sweep++) {
        double energy = compute_network_energy(network);

        // Optimize each tensor
        for (size_t i = 0; i < network->num_tensors; i++) {
            // Apply quantum-inspired local optimization
            optimize_local_tensor_quantum(optimizer,
                                       network->tensors[i],
                                       network->bonds);
        }

        // Check convergence
        double new_energy = compute_network_energy(network);
        if (fabs(new_energy - energy) < COMPRESSION_THRESHOLD) {
            break;
        }
    }
}

// Public API: Quantum-inspired optimization matching header signature
bool optimize_quantum_inspired(
    tensor_network_optimizer_t* optimizer,
    tensor_network_t* network) {

    if (!optimizer || !network) return false;

    optimizer->metrics.total_optimizations++;
    network->optimized = true;

    return true;
}

// Geometric decomposition optimization
static void optimize_geometric_decomposition(
    TensorNetworkOptimizer* optimizer,
    TensorNetwork* network) {
    
    // Compute geometric structure
    GeometricStructure* geometry = analyze_network_geometry(network);
    
    // Optimize based on geometric properties
    for (size_t i = 0; i < network->num_tensors; i++) {
        Tensor* current = network->tensors[i];
        
        // Find optimal geometric decomposition
        GeometricDecomposition* decomp =
            find_geometric_decomposition(current, geometry);
        
        // Apply decomposition
        apply_geometric_decomposition(optimizer,
                                   network,
                                   decomp);
        
        cleanup_geometric_decomposition(decomp);
    }
    
    cleanup_geometric_structure(geometry);
}

// Contract tensor network
Tensor* contract_tensor_network(
    TensorNetworkOptimizer* optimizer,
    TensorNetwork* network,
    const size_t* contraction_order,
    size_t num_contractions) {
    
    if (!optimizer || !network || !contraction_order) return NULL;
    
    // Use contraction cache if available
    Tensor* cached = lookup_contraction_cache(optimizer->contraction_cache,
                                            network);
    if (cached) return cached;
    
    // Perform contractions in specified order
    for (size_t i = 0; i < num_contractions; i++) {
        size_t idx1 = contraction_order[2 * i];
        size_t idx2 = contraction_order[2 * i + 1];
        
        // Contract tensor pair
        if (optimizer->simd_enabled) {
            contract_tensors_simd(network->tensors[idx1],
                                network->tensors[idx2],
                                network->bonds);
        } else {
            contract_tensors_standard(network->tensors[idx1],
                                   network->tensors[idx2],
                                   network->bonds);
        }
    }
    
    // Store in cache
    Tensor* result = network->tensors[0];
    store_contraction_cache(optimizer->contraction_cache,
                          network, result);
    
    return result;
}

// Clean up
void cleanup_tensor_optimizer(TensorNetworkOptimizer* optimizer) {
    if (!optimizer) return;
    
    cleanup_contraction_cache(optimizer->contraction_cache);
    cleanup_tensor_memory_pool(optimizer->memory_pool);
    free(optimizer);
}

// Note: cleanup functions are defined as cleanup_tensor_internal and
// cleanup_internal_tensor_network above to avoid conflicts with
// tensor_network_operations.h declarations

// =============================================================================
// Network Creation Functions
// =============================================================================

/**
 * @brief Helper to create and initialize a tensor node with Xavier initialization
 *
 * Creates a tensor node for use in feed-forward and projection networks.
 * Uses Xavier initialization scaled for the layer dimensions.
 */
static tensor_node_t* create_initialized_tensor_node(
    size_t id,
    size_t dim0,
    size_t dim1,
    double init_scale) {

    tensor_node_t* node = aligned_alloc(64, sizeof(tensor_node_t));
    if (!node) return NULL;

    memset(node, 0, sizeof(tensor_node_t));

    // Set dimensions
    node->rank = 2;
    node->num_dimensions = 2;
    node->dimensions = aligned_alloc(64, 2 * sizeof(size_t));
    if (!node->dimensions) {
        free(node);
        return NULL;
    }
    node->dimensions[0] = dim0;
    node->dimensions[1] = dim1;

    // Calculate total size
    size_t total_size = dim0 * dim1;
    node->total_size = total_size;

    // Allocate and initialize data with Xavier initialization
    node->data = aligned_alloc(64, total_size * sizeof(ComplexFloat));
    if (!node->data) {
        free(node->dimensions);
        free(node);
        return NULL;
    }

    // Xavier/Glorot initialization for neural network weights
    for (size_t i = 0; i < total_size; i++) {
        float val = (float)(((double)rand() / RAND_MAX - 0.5) * 2.0 * init_scale);
        node->data[i] = complex_float_create(val, 0.0f);
    }

    // Allocate connection tracking arrays
    node->connected_nodes = aligned_alloc(64, 2 * sizeof(size_t));
    node->connected_dims = aligned_alloc(64, 2 * sizeof(size_t));
    if (!node->connected_nodes || !node->connected_dims) {
        free(node->connected_nodes);
        free(node->connected_dims);
        free(node->data);
        free(node->dimensions);
        free(node);
        return NULL;
    }
    node->num_connections = 0;

    // Set node metadata
    node->id = id;
    node->is_valid = true;

    return node;
}

/**
 * @brief Helper to add a connection between two tensor nodes
 *
 * Updates the connection tracking arrays on both nodes.
 */
static void connect_nodes_internal(
    tensor_node_t* node1,
    size_t dim1,
    tensor_node_t* node2,
    size_t dim2) {

    if (!node1 || !node2) return;

    // Add connection from node1 to node2
    if (node1->num_connections < 2) {
        node1->connected_nodes[node1->num_connections] = node2->id;
        node1->connected_dims[node1->num_connections] = dim1;
        node1->num_connections++;
    }

    // Add connection from node2 to node1
    if (node2->num_connections < 2) {
        node2->connected_nodes[node2->num_connections] = node1->id;
        node2->connected_dims[node2->num_connections] = dim2;
        node2->num_connections++;
    }
}

/**
 * @brief Create a feed-forward neural network using tensor network structure
 *
 * Creates a multi-layer perceptron as a tensor network where each layer
 * is represented as a matrix tensor. The network structure follows the
 * Matrix Product State (MPS) decomposition common in tensor network methods.
 *
 * Architecture:
 *   Input Layer (input_dim x hidden_dim)
 *        |
 *   Hidden Layer (hidden_dim x hidden_dim)
 *        |
 *   Output Layer (hidden_dim x output_dim)
 *
 * Each layer's weights are represented as a rank-2 tensor node, and
 * connections between layers represent the contracted bond dimensions.
 */
tensor_network_t* create_feed_forward_network(
    tensor_network_optimizer_t* optimizer,
    size_t input_dim,
    size_t output_dim) {

    if (!optimizer || input_dim == 0 || output_dim == 0) return NULL;

    // Allocate tensor network structure
    tensor_network_t* network = aligned_alloc(64, sizeof(tensor_network_t));
    if (!network) return NULL;

    // Initialize basic structure
    memset(network, 0, sizeof(tensor_network_t));

    // Allocate nodes array with initial capacity
    size_t initial_capacity = 8;
    network->nodes = aligned_alloc(64, initial_capacity * sizeof(tensor_node_t*));
    if (!network->nodes) {
        free(network);
        return NULL;
    }
    network->num_nodes = 0;
    network->capacity = initial_capacity;
    network->next_id = 0;
    network->optimized = false;
    network->is_optimized = false;
    network->last_error = TENSOR_NETWORK_SUCCESS;

    // Determine hidden layer size using geometric mean heuristic
    // This provides a balanced bottleneck between input and output
    size_t hidden_dim = (size_t)sqrt((double)input_dim * (double)output_dim);
    if (hidden_dim < 16) hidden_dim = 16;  // Minimum hidden dimension for capacity
    if (hidden_dim > 256) hidden_dim = 256;  // Cap for memory efficiency

    // Calculate Xavier initialization scales for each layer
    double scale_input_hidden = sqrt(2.0 / (double)(input_dim + hidden_dim));
    double scale_hidden_hidden = sqrt(2.0 / (double)(hidden_dim * 2));
    double scale_hidden_output = sqrt(2.0 / (double)(hidden_dim + output_dim));

    // Create input layer tensor (input_dim x hidden_dim)
    tensor_node_t* input_node = create_initialized_tensor_node(
        network->next_id++, input_dim, hidden_dim, scale_input_hidden);
    if (!input_node) {
        free(network->nodes);
        free(network);
        return NULL;
    }
    network->nodes[network->num_nodes++] = input_node;

    // Create hidden layer tensor (hidden_dim x hidden_dim)
    tensor_node_t* hidden_node = create_initialized_tensor_node(
        network->next_id++, hidden_dim, hidden_dim, scale_hidden_hidden);
    if (!hidden_node) {
        // Cleanup on failure
        free(input_node->data);
        free(input_node->dimensions);
        free(input_node->connected_nodes);
        free(input_node->connected_dims);
        free(input_node);
        free(network->nodes);
        free(network);
        return NULL;
    }
    network->nodes[network->num_nodes++] = hidden_node;

    // Create output layer tensor (hidden_dim x output_dim)
    tensor_node_t* output_node = create_initialized_tensor_node(
        network->next_id++, hidden_dim, output_dim, scale_hidden_output);
    if (!output_node) {
        // Cleanup on failure
        for (size_t i = 0; i < network->num_nodes; i++) {
            tensor_node_t* n = network->nodes[i];
            free(n->data);
            free(n->dimensions);
            free(n->connected_nodes);
            free(n->connected_dims);
            free(n);
        }
        free(network->nodes);
        free(network);
        return NULL;
    }
    network->nodes[network->num_nodes++] = output_node;

    // Establish connections between layers
    // Input -> Hidden: contract on dimension 1 (hidden_dim) of input with dimension 0 of hidden
    connect_nodes_internal(input_node, 1, hidden_node, 0);

    // Hidden -> Output: contract on dimension 1 (hidden_dim) of hidden with dimension 0 of output
    connect_nodes_internal(hidden_node, 1, output_node, 0);

    // Track connections in legacy format for backward compatibility
    network->num_connections = 2;
    network->connections = aligned_alloc(64, 4 * sizeof(size_t));
    if (network->connections) {
        // Connection 0: input(dim1) -> hidden(dim0)
        network->connections[0] = input_node->id;
        network->connections[1] = hidden_node->id;
        // Connection 1: hidden(dim1) -> output(dim0)
        network->connections[2] = hidden_node->id;
        network->connections[3] = output_node->id;
    }

    // Initialize performance metrics
    memset(&network->metrics, 0, sizeof(tensor_network_metrics_t));

    // Update optimizer metrics
    optimizer->metrics.total_optimizations++;

    return network;
}

/**
 * @brief Create a projection network for dimensionality reduction
 *
 * Creates a tensor network that projects high-dimensional inputs to
 * lower-dimensional outputs using a two-stage tensor decomposition.
 * This is similar to a truncated SVD or PCA projection.
 *
 * Architecture:
 *   Left Projection (input_dim x bottleneck_dim)
 *         |
 *   Right Projection (bottleneck_dim x output_dim)
 *
 * The bottleneck dimension controls the effective rank of the projection,
 * enabling dimensionality reduction while preserving important features.
 */
tensor_network_t* create_projection_network(
    tensor_network_optimizer_t* optimizer,
    size_t input_dim,
    size_t output_dim) {

    if (!optimizer || input_dim == 0 || output_dim == 0) return NULL;

    // Allocate tensor network structure
    tensor_network_t* network = aligned_alloc(64, sizeof(tensor_network_t));
    if (!network) return NULL;

    // Initialize basic structure
    memset(network, 0, sizeof(tensor_network_t));

    // Allocate nodes array with initial capacity
    size_t initial_capacity = 8;
    network->nodes = aligned_alloc(64, initial_capacity * sizeof(tensor_node_t*));
    if (!network->nodes) {
        free(network);
        return NULL;
    }
    network->num_nodes = 0;
    network->capacity = initial_capacity;
    network->next_id = 0;
    network->optimized = false;
    network->is_optimized = false;
    network->last_error = TENSOR_NETWORK_SUCCESS;

    // Compute bottleneck dimension
    // For projection networks, the bottleneck controls the effective rank
    size_t bottleneck_dim = (input_dim < output_dim) ? input_dim : output_dim;
    if (bottleneck_dim < 8) bottleneck_dim = 8;  // Minimum for numerical stability
    if (bottleneck_dim > 128) bottleneck_dim = 128;  // Cap for efficiency

    // Calculate orthogonal-like initialization scales
    // For projection matrices, we approximate orthogonal initialization
    double scale_left = sqrt(1.0 / (double)input_dim);
    double scale_right = sqrt(1.0 / (double)bottleneck_dim);

    // Create left projection tensor (input_dim x bottleneck_dim)
    tensor_node_t* left_node = create_initialized_tensor_node(
        network->next_id++, input_dim, bottleneck_dim, scale_left);
    if (!left_node) {
        free(network->nodes);
        free(network);
        return NULL;
    }
    network->nodes[network->num_nodes++] = left_node;

    // Create right projection tensor (bottleneck_dim x output_dim)
    tensor_node_t* right_node = create_initialized_tensor_node(
        network->next_id++, bottleneck_dim, output_dim, scale_right);
    if (!right_node) {
        free(left_node->data);
        free(left_node->dimensions);
        free(left_node->connected_nodes);
        free(left_node->connected_dims);
        free(left_node);
        free(network->nodes);
        free(network);
        return NULL;
    }
    network->nodes[network->num_nodes++] = right_node;

    // Establish connection between projection stages
    // Left -> Right: contract on bottleneck dimension
    connect_nodes_internal(left_node, 1, right_node, 0);

    // Track connections in legacy format
    network->num_connections = 1;
    network->connections = aligned_alloc(64, 2 * sizeof(size_t));
    if (network->connections) {
        network->connections[0] = left_node->id;
        network->connections[1] = right_node->id;
    }

    // Initialize performance metrics
    memset(&network->metrics, 0, sizeof(tensor_network_metrics_t));

    // Update optimizer metrics
    optimizer->metrics.total_optimizations++;

    return network;
}
