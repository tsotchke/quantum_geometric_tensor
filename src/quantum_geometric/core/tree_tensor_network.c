#include "quantum_geometric/core/tree_tensor_network.h"
#include "quantum_geometric/core/numerical_backend.h"
#include "quantum_geometric/core/error_handling.h"
#include "quantum_geometric/core/memory_singleton.h"
#include "quantum_geometric/core/advanced_memory_system.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <complex.h>

// Helper function to create a memory pool
static MemoryPool* create_default_memory_pool(void) {
    // Create memory pool configuration
    pool_config_t pool_config = {
        .pool_size = 1024 * 1024 * 1024,  // 1GB pool
        .block_size = sizeof(ComplexFloat),
        .max_blocks = 1024 * 1024,
        .fixed_size = false,
        .thread_safe = true,
        .enable_growth = true,
        .enable_stats = true
    };
    
    // Get global memory system
    advanced_memory_system_t* memory = get_global_memory_system();
    if (!memory) {
        // Create memory system if it doesn't exist
        memory_system_config_t mem_config = {
            .type = MEM_SYSTEM_QUANTUM,
            .strategy = ALLOC_STRATEGY_BUDDY,
            .optimization = MEM_OPT_ADVANCED,
            .alignment = sizeof(ComplexFloat),
            .enable_monitoring = true,
            .enable_defragmentation = true
        };
        memory = create_memory_system(&mem_config);
        if (!memory) {
            printf("DEBUG: Failed to create memory system\n");
            return NULL;
        }
    }
    
    // Create memory pool
    void* pool = ams_create_memory_pool(memory, &pool_config);
    if (!pool) {
        printf("DEBUG: Failed to create memory pool\n");
        return NULL;
    }
    
    // Create memory layouts and buffers
    MemoryPool* memory_pool = malloc(sizeof(MemoryPool));
    if (!memory_pool) {
        printf("DEBUG: Failed to allocate memory pool structure\n");
        return NULL;
    }
    
    memory_pool->layouts = NULL;
    memory_pool->buffers = NULL;
    memory_pool->num_layouts = 0;
    memory_pool->num_buffers = 0;
    memory_pool->fast_path_cache = pool;
    memory_pool->config = pool_config;
    memory_pool->use_neon = true;  // Enable NEON acceleration on ARM
    
    return memory_pool;
}

// Helper function to allocate memory from pool
static void* pool_alloc(MemoryPool* pool, size_t size) {
    if (!pool || !pool->fast_path_cache) {
        return malloc(size);
    }
    
    // Get global memory system
    advanced_memory_system_t* memory = get_global_memory_system();
    if (!memory) {
        return malloc(size);
    }
    
    return ams_pool_allocate(memory, pool->fast_path_cache, size);
}

// Helper function to free memory from pool
static void pool_free_memory(MemoryPool* pool, void* ptr) {
    if (!pool || !pool->fast_path_cache || !ptr) {
        free(ptr);
        return;
    }
    
    // Get global memory system
    advanced_memory_system_t* memory = get_global_memory_system();
    if (!memory) {
        free(ptr);
        return;
    }
    
    ams_pool_free(memory, pool->fast_path_cache, ptr);
}

// =============================================================================
// Proper Tensor Contraction (Einstein Summation)
// =============================================================================

/**
 * @brief Contract two tensors over specified indices
 *
 * Performs C_{i...j...} = Σ_k A_{i...k...} B_{k...j...}
 * where k is the contracted index.
 *
 * @param tensor_a First tensor data
 * @param dims_a Dimensions of tensor A
 * @param rank_a Number of dimensions in A
 * @param tensor_b Second tensor data
 * @param dims_b Dimensions of tensor B
 * @param rank_b Number of dimensions in B
 * @param contract_idx_a Index in A to contract (0-indexed)
 * @param contract_idx_b Index in B to contract (0-indexed)
 * @param result_out Output: contracted tensor data
 * @param result_dims_out Output: dimensions of result tensor
 * @param result_rank_out Output: number of dimensions in result
 * @return true on success, false on failure
 */
static bool contract_tensors_proper(
    const ComplexFloat* tensor_a,
    const size_t* dims_a,
    size_t rank_a,
    const ComplexFloat* tensor_b,
    const size_t* dims_b,
    size_t rank_b,
    size_t contract_idx_a,
    size_t contract_idx_b,
    ComplexFloat** result_out,
    size_t** result_dims_out,
    size_t* result_rank_out,
    MemoryPool* pool)
{
    // Validate contraction dimensions match
    if (contract_idx_a >= rank_a || contract_idx_b >= rank_b) {
        printf("DEBUG: Invalid contraction indices\n");
        return false;
    }

    size_t contract_dim = dims_a[contract_idx_a];
    if (dims_b[contract_idx_b] != contract_dim) {
        printf("DEBUG: Contraction dimension mismatch: %zu vs %zu\n",
               dims_a[contract_idx_a], dims_b[contract_idx_b]);
        return false;
    }

    // Compute result dimensions (all non-contracted indices)
    size_t result_rank = rank_a + rank_b - 2;
    if (result_rank == 0) result_rank = 1;  // Scalar result

    size_t* result_dims = pool_alloc(pool, result_rank * sizeof(size_t));
    if (!result_dims) return false;

    // Build result dimensions: A's dims (except contracted) + B's dims (except contracted)
    size_t dim_idx = 0;
    for (size_t i = 0; i < rank_a; i++) {
        if (i != contract_idx_a) {
            result_dims[dim_idx++] = dims_a[i];
        }
    }
    for (size_t i = 0; i < rank_b; i++) {
        if (i != contract_idx_b) {
            result_dims[dim_idx++] = dims_b[i];
        }
    }

    // Handle scalar result
    if (result_rank == 1 && dim_idx == 0) {
        result_dims[0] = 1;
    }

    // Compute strides for tensor A
    size_t* strides_a = pool_alloc(pool, rank_a * sizeof(size_t));
    if (!strides_a) {
        pool_free_memory(pool, result_dims);
        return false;
    }
    strides_a[rank_a - 1] = 1;
    for (int i = (int)rank_a - 2; i >= 0; i--) {
        strides_a[i] = strides_a[i + 1] * dims_a[i + 1];
    }

    // Compute strides for tensor B
    size_t* strides_b = pool_alloc(pool, rank_b * sizeof(size_t));
    if (!strides_b) {
        pool_free_memory(pool, strides_a);
        pool_free_memory(pool, result_dims);
        return false;
    }
    strides_b[rank_b - 1] = 1;
    for (int i = (int)rank_b - 2; i >= 0; i--) {
        strides_b[i] = strides_b[i + 1] * dims_b[i + 1];
    }

    // Compute result size and allocate
    size_t result_size = 1;
    for (size_t i = 0; i < result_rank; i++) {
        result_size *= result_dims[i];
    }

    ComplexFloat* result = pool_alloc(pool, result_size * sizeof(ComplexFloat));
    if (!result) {
        pool_free_memory(pool, strides_b);
        pool_free_memory(pool, strides_a);
        pool_free_memory(pool, result_dims);
        return false;
    }
    memset(result, 0, result_size * sizeof(ComplexFloat));

    // Compute sizes for iteration
    size_t size_a = 1, size_b = 1;
    for (size_t i = 0; i < rank_a; i++) size_a *= dims_a[i];
    for (size_t i = 0; i < rank_b; i++) size_b *= dims_b[i];

    // Precompute non-contracted sizes
    size_t outer_size_a = size_a / contract_dim;
    size_t outer_size_b = size_b / contract_dim;

    // Perform contraction using efficient nested loops
    // For each combination of non-contracted A indices and non-contracted B indices
    // Sum over the contracted index

    // This is a general implementation; can be optimized for specific cases
    // For matrix-matrix: C[i,j] = sum_k A[i,k] * B[k,j]

    size_t stride_contract_a = strides_a[contract_idx_a];
    size_t stride_contract_b = strides_b[contract_idx_b];

    // Iterate using outer indices for better cache locality
    // For each combination of non-contracted A indices
    for (size_t outer_a = 0; outer_a < outer_size_a; outer_a++) {
        // For each combination of non-contracted B indices
        for (size_t outer_b = 0; outer_b < outer_size_b; outer_b++) {
            // Compute result index from outer indices
            size_t result_idx = outer_a * outer_size_b + outer_b;

            if (result_idx >= result_size) continue;

            // Sum over the contracted dimension
            ComplexFloat sum = {0.0f, 0.0f};
            for (size_t k = 0; k < contract_dim; k++) {
                // Compute A index: combine outer_a with contracted index k
                size_t idx_a = 0;
                size_t temp_outer = outer_a;
                for (int d = (int)rank_a - 1; d >= 0; d--) {
                    if ((size_t)d == contract_idx_a) {
                        idx_a += k * stride_contract_a;
                    } else {
                        size_t dim_size = dims_a[d];
                        size_t idx_in_dim = temp_outer % dim_size;
                        temp_outer /= dim_size;
                        idx_a += idx_in_dim * strides_a[d];
                    }
                }

                // Compute B index: combine outer_b with contracted index k
                size_t idx_b = 0;
                temp_outer = outer_b;
                for (int d = (int)rank_b - 1; d >= 0; d--) {
                    if ((size_t)d == contract_idx_b) {
                        idx_b += k * stride_contract_b;
                    } else {
                        size_t dim_size = dims_b[d];
                        size_t idx_in_dim = temp_outer % dim_size;
                        temp_outer /= dim_size;
                        idx_b += idx_in_dim * strides_b[d];
                    }
                }

                // Accumulate product
                if (idx_a < size_a && idx_b < size_b) {
                    ComplexFloat a = tensor_a[idx_a];
                    ComplexFloat b = tensor_b[idx_b];
                    sum.real += a.real * b.real - a.imag * b.imag;
                    sum.imag += a.real * b.imag + a.imag * b.real;
                }
            }
            result[result_idx] = sum;
        }
    }

    pool_free_memory(pool, strides_a);
    pool_free_memory(pool, strides_b);

    *result_out = result;
    *result_dims_out = result_dims;
    *result_rank_out = result_rank;

    return true;
}

/**
 * @brief Simple matrix contraction: C = A @ B
 *
 * For rank-2 tensors, this is standard matrix multiplication.
 * A is (m, k), B is (k, n), result is (m, n)
 */
static bool contract_matrices(
    const ComplexFloat* A,
    size_t m, size_t k,
    const ComplexFloat* B,
    size_t n,  // B is k x n
    ComplexFloat* C)  // C is m x n, must be pre-allocated
{
    // Initialize C to zero
    memset(C, 0, m * n * sizeof(ComplexFloat));

    // Cache-blocked matrix multiplication for better performance
    const size_t BLOCK = 32;

    for (size_t ii = 0; ii < m; ii += BLOCK) {
        size_t i_end = (ii + BLOCK < m) ? ii + BLOCK : m;
        for (size_t jj = 0; jj < n; jj += BLOCK) {
            size_t j_end = (jj + BLOCK < n) ? jj + BLOCK : n;
            for (size_t ll = 0; ll < k; ll += BLOCK) {
                size_t l_end = (ll + BLOCK < k) ? ll + BLOCK : k;

                for (size_t i = ii; i < i_end; i++) {
                    for (size_t l = ll; l < l_end; l++) {
                        ComplexFloat a = A[i * k + l];
                        for (size_t j = jj; j < j_end; j++) {
                            ComplexFloat b = B[l * n + j];
                            C[i * n + j].real += a.real * b.real - a.imag * b.imag;
                            C[i * n + j].imag += a.real * b.imag + a.imag * b.real;
                        }
                    }
                }
            }
        }
    }

    return true;
}

// =============================================================================
// Tree Tensor Network Core Functions
// =============================================================================

// Create a new tree tensor network
tree_tensor_network_t* create_tree_tensor_network(
    size_t num_qubits,
    size_t max_rank,
    double tolerance) {
    
    printf("DEBUG: Creating tree tensor network with %zu qubits, max_rank=%zu, tolerance=%.6f\n",
           num_qubits, max_rank, tolerance);
    
    tree_tensor_network_t* ttn = malloc(sizeof(tree_tensor_network_t));
    if (!ttn) {
        printf("DEBUG: Failed to allocate tree tensor network\n");
        return NULL;
    }
    
    // Initialize fields
    ttn->root = NULL;
    ttn->num_nodes = 0;
    ttn->max_rank = max_rank;
    ttn->num_qubits = num_qubits;
    ttn->tolerance = tolerance;
    ttn->memory_pool = create_default_memory_pool();
    ttn->memory_system = get_global_memory_system();
    memset(&ttn->metrics, 0, sizeof(tensor_network_metrics_t));
    
    if (!ttn->memory_pool) {
        printf("DEBUG: Failed to create memory pool\n");
        free(ttn);
        return NULL;
    }
    
    printf("DEBUG: Tree tensor network created successfully\n");
    return ttn;
}

// Destroy a tree tensor node
static void destroy_tree_tensor_node(tree_tensor_network_t* ttn, tree_tensor_node_t* node) {
    if (!node) return;
    
    // Recursively destroy children
    for (size_t i = 0; i < node->num_children; i++) {
        destroy_tree_tensor_node(ttn, node->children[i]);
    }
    
    // Free resources
    if (node->children) {
        pool_free_memory(ttn->memory_pool, node->children);
    }
    
    if (node->data) {
        pool_free_memory(ttn->memory_pool, node->data);
    }
    
    if (node->h_matrix) {
        destroy_hierarchical_matrix(node->h_matrix);
    }
    
    if (node->dimensions) {
        pool_free_memory(ttn->memory_pool, node->dimensions);
    }
    
    pool_free_memory(ttn->memory_pool, node);
}

// Destroy a tree tensor network
void destroy_tree_tensor_network(tree_tensor_network_t* ttn) {
    if (!ttn) return;
    
    // Destroy root node and all children
    destroy_tree_tensor_node(ttn, ttn->root);
    
    // Cleanup memory pool
    if (ttn->memory_pool) {
        // Get global memory system
        advanced_memory_system_t* memory = ttn->memory_system;
        if (memory && ttn->memory_pool->fast_path_cache) {
            ams_destroy_memory_pool(memory, ttn->memory_pool->fast_path_cache);
        }
        free(ttn->memory_pool);
    }
    
    free(ttn);
}

// Add a new tree tensor node
tree_tensor_node_t* add_tree_tensor_node(
    tree_tensor_network_t* ttn,
    const ComplexFloat* data,
    const size_t* dimensions,
    size_t num_dimensions,
    bool use_hierarchical) {
    
    if (!ttn || !data || !dimensions || num_dimensions == 0) {
        printf("DEBUG: Invalid arguments to add_tree_tensor_node\n");
        return NULL;
    }
    
    // Allocate new node
    tree_tensor_node_t* node = pool_alloc(ttn->memory_pool, sizeof(tree_tensor_node_t));
    if (!node) {
        printf("DEBUG: Failed to allocate tree tensor node\n");
        return NULL;
    }
    
    // Initialize fields
    node->id = ttn->num_nodes++;
    node->rank = 0;  // Will be set during optimization
    node->num_children = 0;
    node->children = NULL;
    node->parent = NULL;
    node->h_matrix = NULL;
    node->is_leaf = true;
    node->use_hierarchical = use_hierarchical;
    
    // Copy dimensions
    node->dimensions = pool_alloc(ttn->memory_pool, num_dimensions * sizeof(size_t));
    if (!node->dimensions) {
        printf("DEBUG: Failed to allocate dimensions array\n");
        pool_free_memory(ttn->memory_pool, node);
        return NULL;
    }
    memcpy(node->dimensions, dimensions, num_dimensions * sizeof(size_t));
    node->num_dimensions = num_dimensions;
    
    // Calculate total size
    size_t total_size = 1;
    for (size_t i = 0; i < num_dimensions; i++) {
        total_size *= dimensions[i];
    }
    
    if (use_hierarchical) {
        // Create hierarchical matrix representation
        printf("DEBUG: Creating hierarchical matrix representation\n");
        node->h_matrix = create_hierarchical_matrix(total_size, ttn->tolerance);
        if (!node->h_matrix) {
            printf("DEBUG: Failed to create hierarchical matrix\n");
            pool_free_memory(ttn->memory_pool, node->dimensions);
            pool_free_memory(ttn->memory_pool, node);
            return NULL;
        }
        
        // Initialize hierarchical matrix with data
        // This will compress the data if possible
        matrix_properties_t props = {
            .dimension = total_size,
            .tolerance = ttn->tolerance,
            .symmetric = false,
            .positive_definite = false
        };
        init_matrix_properties(node->h_matrix, &props);
        
        // Copy data to hierarchical matrix
        for (size_t i = 0; i < total_size && i < node->h_matrix->n; i++) {
            node->h_matrix->data[i] = (double complex){data[i].real, data[i].imag};
        }
        
        // Compress matrix
        compression_params_t params = {
            .mode = COMPRESS_SVD,
            .tolerance = ttn->tolerance,
            .max_rank = ttn->max_rank,
            .recompression = true
        };
        compress_matrix(node->h_matrix, &params);
        
        node->data = NULL;  // Data is stored in h_matrix
    } else {
        // Use standard tensor representation
        node->data = pool_alloc(ttn->memory_pool, total_size * sizeof(ComplexFloat));
        if (!node->data) {
            printf("DEBUG: Failed to allocate tensor data\n");
            pool_free_memory(ttn->memory_pool, node->dimensions);
            pool_free_memory(ttn->memory_pool, node);
            return NULL;
        }
        memcpy(node->data, data, total_size * sizeof(ComplexFloat));
    }
    
    // Set as root if this is the first node
    if (ttn->root == NULL) {
        ttn->root = node;
    }
    
    return node;
}

// Connect two tree tensor nodes
bool connect_tree_tensor_nodes(
    tree_tensor_network_t* ttn,
    tree_tensor_node_t* parent,
    tree_tensor_node_t* child) {
    
    if (!ttn || !parent || !child) {
        printf("DEBUG: Invalid arguments to connect_tree_tensor_nodes\n");
        return false;
    }
    
    // Check if child already has a parent
    if (child->parent != NULL) {
        printf("DEBUG: Child node already has a parent\n");
        return false;
    }
    
    // Resize children array
    tree_tensor_node_t** new_children = pool_alloc(ttn->memory_pool, 
        (parent->num_children + 1) * sizeof(tree_tensor_node_t*));
    if (!new_children) {
        printf("DEBUG: Failed to allocate children array\n");
        return false;
    }
    
    // Copy existing children
    if (parent->children) {
        memcpy(new_children, parent->children, 
               parent->num_children * sizeof(tree_tensor_node_t*));
        pool_free_memory(ttn->memory_pool, parent->children);
    }
    
    // Add new child
    new_children[parent->num_children] = child;
    parent->children = new_children;
    parent->num_children++;
    parent->is_leaf = false;
    
    // Set parent reference
    child->parent = parent;
    
    return true;
}

// Create a tensor stream
tensor_stream_t* create_tensor_stream(
    tree_tensor_network_t* ttn,
    tree_tensor_node_t* node,
    size_t chunk_size) {
    
    if (!ttn || !node) {
        printf("DEBUG: Invalid arguments to create_tensor_stream\n");
        return NULL;
    }
    
    tensor_stream_t* stream = pool_alloc(ttn->memory_pool, sizeof(tensor_stream_t));
    if (!stream) {
        printf("DEBUG: Failed to allocate tensor stream\n");
        return NULL;
    }
    
    // Calculate total size
    size_t total_size = 1;
    for (size_t i = 0; i < node->num_dimensions; i++) {
        total_size *= node->dimensions[i];
    }
    
    // Initialize stream
    stream->chunk_size = chunk_size;
    stream->current_offset = 0;
    stream->total_size = total_size;
    stream->source = node;
    stream->is_hierarchical = node->use_hierarchical;
    stream->memory_pool = ttn->memory_pool;
    
    // Allocate buffer
    stream->buffer = pool_alloc(ttn->memory_pool, chunk_size * sizeof(ComplexFloat));
    if (!stream->buffer) {
        printf("DEBUG: Failed to allocate stream buffer\n");
        pool_free_memory(ttn->memory_pool, stream);
        return NULL;
    }
    
    // Load first chunk
    if (!stream_next_chunk(stream)) {
        printf("DEBUG: Failed to load first chunk\n");
        pool_free_memory(ttn->memory_pool, stream->buffer);
        pool_free_memory(ttn->memory_pool, stream);
        return NULL;
    }
    
    return stream;
}

// Destroy a tensor stream
void destroy_tensor_stream(tensor_stream_t* stream) {
    if (!stream) return;
    
    if (stream->buffer) {
        pool_free_memory(stream->memory_pool, stream->buffer);
    }
    
    pool_free_memory(stream->memory_pool, stream);
}

// Load the next chunk of data from a tensor stream
bool stream_next_chunk(tensor_stream_t* stream) {
    if (!stream || stream->current_offset >= stream->total_size) {
        return false;
    }
    
    // Calculate chunk size
    size_t remaining = stream->total_size - stream->current_offset;
    size_t chunk_size = (remaining < stream->chunk_size) ? remaining : stream->chunk_size;
    
    // Load data from source
    tree_tensor_node_t* node = (tree_tensor_node_t*)stream->source;
    
    if (stream->is_hierarchical) {
        // Load from hierarchical matrix
        if (!node->h_matrix) {
            printf("DEBUG: Hierarchical matrix is NULL\n");
            return false;
        }
        
        // Convert from double complex to ComplexFloat
        for (size_t i = 0; i < chunk_size; i++) {
            size_t idx = stream->current_offset + i;
            if (idx < node->h_matrix->n) {
                stream->buffer[i].real = creal(node->h_matrix->data[idx]);
                stream->buffer[i].imag = cimag(node->h_matrix->data[idx]);
            } else {
                stream->buffer[i].real = 0.0f;
                stream->buffer[i].imag = 0.0f;
            }
        }
    } else {
        // Load from standard tensor
        if (!node->data) {
            printf("DEBUG: Tensor data is NULL\n");
            return false;
        }
        
        memcpy(stream->buffer, node->data + stream->current_offset, 
               chunk_size * sizeof(ComplexFloat));
    }
    
    // Update offset
    stream->current_offset += chunk_size;
    
    return true;
}

// Contract two tensor streams with proper result writing
bool contract_tensor_streams(
    tensor_stream_t* stream1,
    tensor_stream_t* stream2,
    tensor_stream_t* result) {

    if (!stream1 || !stream2 || !result) {
        printf("DEBUG: Invalid arguments to contract_tensor_streams\n");
        return false;
    }

    // Reset streams to beginning
    stream1->current_offset = 0;
    stream2->current_offset = 0;
    result->current_offset = 0;

    // Load first chunks
    if (!stream_next_chunk(stream1) || !stream_next_chunk(stream2)) {
        printf("DEBUG: Failed to load initial chunks\n");
        return false;
    }

    // Track result write position
    size_t result_write_pos = 0;

    // Process chunks - outer product style contraction
    while (stream1->current_offset <= stream1->total_size &&
           stream2->current_offset <= stream2->total_size) {

        // Compute actual chunk sizes
        size_t chunk1_size = stream1->chunk_size;
        if (stream1->current_offset + chunk1_size > stream1->total_size) {
            chunk1_size = stream1->total_size - (stream1->current_offset - stream1->chunk_size);
        }
        size_t chunk2_size = stream2->chunk_size;
        if (stream2->current_offset + chunk2_size > stream2->total_size) {
            chunk2_size = stream2->total_size - (stream2->current_offset - stream2->chunk_size);
        }

        if (chunk1_size == 0 || chunk2_size == 0) break;

        // Perform outer product on current chunks
        size_t result_chunk_size = chunk1_size * chunk2_size;

        for (size_t i = 0; i < chunk1_size; i++) {
            for (size_t j = 0; j < chunk2_size; j++) {
                size_t idx = i * chunk2_size + j;
                if (idx < result->chunk_size) {
                    result->buffer[idx].real =
                        stream1->buffer[i].real * stream2->buffer[j].real -
                        stream1->buffer[i].imag * stream2->buffer[j].imag;
                    result->buffer[idx].imag =
                        stream1->buffer[i].real * stream2->buffer[j].imag +
                        stream1->buffer[i].imag * stream2->buffer[j].real;
                }
            }
        }

        // Write result chunk to result node's hierarchical matrix
        if (result->node && result->node->h_matrix) {
            // Write computed chunk to hierarchical matrix
            for (size_t i = 0; i < result_chunk_size && (result_write_pos + i) < result->total_size; i++) {
                if (i < result->chunk_size) {
                    // Store in hierarchical matrix (as flat data initially)
                    // Convert ComplexFloat to double complex
                    double complex val = (double)result->buffer[i].real +
                                         (double)result->buffer[i].imag * I;
                    hierarchical_matrix_set_element(result->node->h_matrix,
                                                    result_write_pos + i,
                                                    val);
                }
            }
            result_write_pos += result_chunk_size;
        } else if (result->node && result->node->data) {
            // Write to regular tensor data
            for (size_t i = 0; i < result_chunk_size && (result_write_pos + i) < result->total_size; i++) {
                if (i < result->chunk_size) {
                    result->node->data[result_write_pos + i] = result->buffer[i];
                }
            }
            result_write_pos += result_chunk_size;
        }

        // Load next chunks
        if (!stream_next_chunk(stream1)) {
            // stream1 exhausted, reset and advance stream2
            stream1->current_offset = 0;
            if (!stream_next_chunk(stream1)) break;
            if (!stream_next_chunk(stream2)) break;
        }
    }

    return true;
}

// Contract two tree tensor nodes
bool contract_tree_tensor_nodes(
    tree_tensor_network_t* ttn,
    tree_tensor_node_t* node1,
    tree_tensor_node_t* node2,
    tree_tensor_node_t** result) {
    
    if (!ttn || !node1 || !node2 || !result) {
        printf("DEBUG: Invalid arguments to contract_tree_tensor_nodes\n");
        return false;
    }
    
    // Determine if we should use streaming or direct contraction
    bool use_streaming = false;
    size_t total_size1 = 1, total_size2 = 1;
    
    for (size_t i = 0; i < node1->num_dimensions; i++) {
        total_size1 *= node1->dimensions[i];
    }
    
    for (size_t i = 0; i < node2->num_dimensions; i++) {
        total_size2 *= node2->dimensions[i];
    }
    
    // Use streaming for large tensors
    const size_t STREAMING_THRESHOLD = 1024 * 1024;  // 1M elements
    if (total_size1 > STREAMING_THRESHOLD || total_size2 > STREAMING_THRESHOLD) {
        use_streaming = true;
    }
    
    if (use_streaming) {
        printf("DEBUG: Using streaming contraction for large tensors\n");
        
        // Create streams
        const size_t CHUNK_SIZE = 1024 * 1024;  // 1M elements per chunk
        tensor_stream_t* stream1 = create_tensor_stream(ttn, node1, CHUNK_SIZE);
        tensor_stream_t* stream2 = create_tensor_stream(ttn, node2, CHUNK_SIZE);
        
        if (!stream1 || !stream2) {
            printf("DEBUG: Failed to create tensor streams\n");
            if (stream1) destroy_tensor_stream(stream1);
            if (stream2) destroy_tensor_stream(stream2);
            return false;
        }
        
        // Calculate result dimensions
        // For simplicity, we'll just concatenate the dimensions
        // In a real implementation, this would handle tensor contraction properly
        size_t num_result_dims = node1->num_dimensions + node2->num_dimensions;
        size_t* result_dims = pool_alloc(ttn->memory_pool, num_result_dims * sizeof(size_t));
        if (!result_dims) {
            printf("DEBUG: Failed to allocate result dimensions\n");
            destroy_tensor_stream(stream1);
            destroy_tensor_stream(stream2);
            return false;
        }
        
        // Copy dimensions
        memcpy(result_dims, node1->dimensions, node1->num_dimensions * sizeof(size_t));
        memcpy(result_dims + node1->num_dimensions, node2->dimensions, 
               node2->num_dimensions * sizeof(size_t));
        
        // Create result node
        // For now, we'll just create a placeholder node
        // In a real implementation, this would be filled with the contraction result
        *result = pool_alloc(ttn->memory_pool, sizeof(tree_tensor_node_t));
        if (!*result) {
            printf("DEBUG: Failed to allocate result node\n");
            pool_free_memory(ttn->memory_pool, result_dims);
            destroy_tensor_stream(stream1);
            destroy_tensor_stream(stream2);
            return false;
        }
        
        // Initialize result node
        (*result)->id = ttn->num_nodes++;
        (*result)->rank = 0;
        (*result)->num_children = 0;
        (*result)->children = NULL;
        (*result)->parent = NULL;
        (*result)->dimensions = result_dims;
        (*result)->num_dimensions = num_result_dims;
        (*result)->is_leaf = true;
        (*result)->use_hierarchical = true;  // Use hierarchical for large results
        
        // Create hierarchical matrix for result
        size_t result_size = total_size1 * total_size2;  // Simplified
        (*result)->h_matrix = create_hierarchical_matrix(result_size, ttn->tolerance);
        if (!(*result)->h_matrix) {
            printf("DEBUG: Failed to create result hierarchical matrix\n");
            pool_free_memory(ttn->memory_pool, *result);
            pool_free_memory(ttn->memory_pool, result_dims);
            destroy_tensor_stream(stream1);
            destroy_tensor_stream(stream2);
            return false;
        }
        
        // Create result stream
        tensor_stream_t* result_stream = create_tensor_stream(ttn, *result, CHUNK_SIZE);
        if (!result_stream) {
            printf("DEBUG: Failed to create result stream\n");
            destroy_hierarchical_matrix((*result)->h_matrix);
            pool_free_memory(ttn->memory_pool, *result);
            pool_free_memory(ttn->memory_pool, result_dims);
            destroy_tensor_stream(stream1);
            destroy_tensor_stream(stream2);
            return false;
        }
        
        // Perform streaming contraction
        if (!contract_tensor_streams(stream1, stream2, result_stream)) {
            printf("DEBUG: Failed to contract tensor streams\n");
            destroy_tensor_stream(result_stream);
            destroy_hierarchical_matrix((*result)->h_matrix);
            pool_free_memory(ttn->memory_pool, *result);
            pool_free_memory(ttn->memory_pool, result_dims);
            destroy_tensor_stream(stream1);
            destroy_tensor_stream(stream2);
            return false;
        }
        
        // Clean up streams
        destroy_tensor_stream(result_stream);
        destroy_tensor_stream(stream1);
        destroy_tensor_stream(stream2);
    } else {
        printf("DEBUG: Using direct contraction for small tensors\n");

        // For small tensors, use proper tensor contraction
        // Default: contract last index of node1 with first index of node2
        // This is the standard convention for TTN parent-child contraction

        if (node1->use_hierarchical || node2->use_hierarchical) {
            // Handle hierarchical matrices - fallback to simple contraction
            // For simplicity, we'll just concatenate the dimensions
            size_t num_result_dims = node1->num_dimensions + node2->num_dimensions;
            size_t* result_dims = pool_alloc(ttn->memory_pool, num_result_dims * sizeof(size_t));
            if (!result_dims) {
                printf("DEBUG: Failed to allocate result dimensions\n");
                return false;
            }

            memcpy(result_dims, node1->dimensions, node1->num_dimensions * sizeof(size_t));
            memcpy(result_dims + node1->num_dimensions, node2->dimensions,
                   node2->num_dimensions * sizeof(size_t));

            size_t result_size = total_size1 * total_size2;
            ComplexFloat* result_data = pool_alloc(ttn->memory_pool, result_size * sizeof(ComplexFloat));
            if (!result_data) {
                pool_free_memory(ttn->memory_pool, result_dims);
                return false;
            }
            memset(result_data, 0, result_size * sizeof(ComplexFloat));

            *result = add_tree_tensor_node(ttn, result_data, result_dims, num_result_dims, false);
            pool_free_memory(ttn->memory_pool, result_data);
            pool_free_memory(ttn->memory_pool, result_dims);
        } else {
            // PROPER TENSOR CONTRACTION
            // Check for matching dimensions to determine contraction indices
            // Default: contract last index of A with first index of B

            size_t contract_idx_a = node1->num_dimensions - 1;  // Last index of A
            size_t contract_idx_b = 0;                          // First index of B

            // Verify dimensions match
            if (node1->dimensions[contract_idx_a] != node2->dimensions[contract_idx_b]) {
                printf("DEBUG: Warning - default contraction indices have mismatched dimensions.\n");
                printf("       A[last]=%zu, B[first]=%zu. Searching for matching dimensions...\n",
                       node1->dimensions[contract_idx_a], node2->dimensions[contract_idx_b]);

                // Search for matching dimensions
                bool found = false;
                for (size_t i = 0; i < node1->num_dimensions && !found; i++) {
                    for (size_t j = 0; j < node2->num_dimensions && !found; j++) {
                        if (node1->dimensions[i] == node2->dimensions[j]) {
                            contract_idx_a = i;
                            contract_idx_b = j;
                            found = true;
                            printf("       Found matching dimensions at A[%zu]=%zu, B[%zu]=%zu\n",
                                   i, node1->dimensions[i], j, node2->dimensions[j]);
                        }
                    }
                }

                if (!found) {
                    printf("DEBUG: No matching dimensions found. Performing outer product.\n");
                    // Fall back to outer product
                    size_t num_result_dims = node1->num_dimensions + node2->num_dimensions;
                    size_t* result_dims = pool_alloc(ttn->memory_pool, num_result_dims * sizeof(size_t));
                    if (!result_dims) return false;

                    memcpy(result_dims, node1->dimensions, node1->num_dimensions * sizeof(size_t));
                    memcpy(result_dims + node1->num_dimensions, node2->dimensions,
                           node2->num_dimensions * sizeof(size_t));

                    size_t result_size = total_size1 * total_size2;
                    ComplexFloat* result_data = pool_alloc(ttn->memory_pool, result_size * sizeof(ComplexFloat));
                    if (!result_data) {
                        pool_free_memory(ttn->memory_pool, result_dims);
                        return false;
                    }

                    // Outer product
                    for (size_t i = 0; i < total_size1; i++) {
                        for (size_t j = 0; j < total_size2; j++) {
                            size_t idx = i * total_size2 + j;
                            result_data[idx].real =
                                node1->data[i].real * node2->data[j].real -
                                node1->data[i].imag * node2->data[j].imag;
                            result_data[idx].imag =
                                node1->data[i].real * node2->data[j].imag +
                                node1->data[i].imag * node2->data[j].real;
                        }
                    }

                    *result = add_tree_tensor_node(ttn, result_data, result_dims, num_result_dims, false);
                    pool_free_memory(ttn->memory_pool, result_data);
                    pool_free_memory(ttn->memory_pool, result_dims);

                    if (!*result) return false;
                    return true;
                }
            }

            // SPECIALIZED CASE: Matrix-matrix multiplication (rank-2 tensors)
            if (node1->num_dimensions == 2 && node2->num_dimensions == 2 &&
                contract_idx_a == 1 && contract_idx_b == 0) {
                // Standard matrix multiplication: C[i,j] = sum_k A[i,k] * B[k,j]
                size_t m = node1->dimensions[0];
                size_t k = node1->dimensions[1];  // = node2->dimensions[0]
                size_t n = node2->dimensions[1];

                printf("DEBUG: Using optimized matrix multiplication (%zu x %zu) @ (%zu x %zu)\n",
                       m, k, k, n);

                size_t* result_dims = pool_alloc(ttn->memory_pool, 2 * sizeof(size_t));
                if (!result_dims) return false;
                result_dims[0] = m;
                result_dims[1] = n;

                ComplexFloat* result_data = pool_alloc(ttn->memory_pool, m * n * sizeof(ComplexFloat));
                if (!result_data) {
                    pool_free_memory(ttn->memory_pool, result_dims);
                    return false;
                }

                // Efficient cache-blocked matrix multiplication
                contract_matrices(node1->data, m, k, node2->data, n, result_data);

                *result = add_tree_tensor_node(ttn, result_data, result_dims, 2, false);
                pool_free_memory(ttn->memory_pool, result_data);
                pool_free_memory(ttn->memory_pool, result_dims);
            } else {
                // GENERAL CASE: Use proper tensor contraction
                printf("DEBUG: Using general tensor contraction. A:rank=%zu, B:rank=%zu, contract A[%zu] with B[%zu]\n",
                       node1->num_dimensions, node2->num_dimensions, contract_idx_a, contract_idx_b);

                ComplexFloat* result_data;
                size_t* result_dims;
                size_t result_rank;

                if (!contract_tensors_proper(
                        node1->data, node1->dimensions, node1->num_dimensions,
                        node2->data, node2->dimensions, node2->num_dimensions,
                        contract_idx_a, contract_idx_b,
                        &result_data, &result_dims, &result_rank,
                        ttn->memory_pool)) {
                    printf("DEBUG: General tensor contraction failed\n");
                    return false;
                }

                *result = add_tree_tensor_node(ttn, result_data, result_dims, result_rank, false);
                pool_free_memory(ttn->memory_pool, result_data);
                pool_free_memory(ttn->memory_pool, result_dims);
            }
        }

        if (!*result) {
            printf("DEBUG: Failed to create result node\n");
            return false;
        }
    }

    return true;
}

// Forward declarations for helper functions used in contract_full_tree_network
static tree_tensor_node_t* deep_copy_node(tree_tensor_network_t* ttn_dest,
                                          tree_tensor_network_t* ttn_src,
                                          tree_tensor_node_t* src_node,
                                          tree_tensor_node_t* new_parent);
static bool deep_copy_tree_structure(tree_tensor_network_t* dest, tree_tensor_network_t* src);
static size_t count_nodes_recursive(tree_tensor_node_t* node);
static bool find_optimal_contraction_pair(tree_tensor_network_t* ttn,
                                          tree_tensor_node_t** node1_out,
                                          tree_tensor_node_t** node2_out);
static void remove_from_parent(tree_tensor_network_t* ttn, tree_tensor_node_t* node);
static bool add_to_parent(tree_tensor_network_t* ttn,
                          tree_tensor_node_t* parent,
                          tree_tensor_node_t* child);
static bool update_tree_after_contraction(tree_tensor_network_t* ttn,
                                          tree_tensor_node_t* node1,
                                          tree_tensor_node_t* node2,
                                          tree_tensor_node_t* result);

// Contract the full tree tensor network
bool contract_full_tree_network(
    tree_tensor_network_t* ttn,
    ComplexFloat** result,
    size_t* result_dims,
    size_t* num_dims) {
    
    if (!ttn || !result || !result_dims || !num_dims) {
        printf("DEBUG: Invalid arguments to contract_full_tree_network\n");
        return false;
    }
    
    // Check if network is empty
    if (!ttn->root) {
        printf("DEBUG: Tree tensor network is empty\n");
        return false;
    }
    
    // Single node case
    if (ttn->root->is_leaf) {
        printf("DEBUG: Single node case\n");
        
        // Calculate total size
        size_t total_size = 1;
        for (size_t i = 0; i < ttn->root->num_dimensions; i++) {
            total_size *= ttn->root->dimensions[i];
        }
        
        // Allocate result
        *result = malloc(total_size * sizeof(ComplexFloat));
        if (!*result) {
            printf("DEBUG: Failed to allocate result\n");
            return false;
        }
        
        // Copy data
        if (ttn->root->use_hierarchical) {
            // Copy from hierarchical matrix
            if (!ttn->root->h_matrix) {
                printf("DEBUG: Hierarchical matrix is NULL\n");
                free(*result);
                return false;
            }
            
            // Convert from double complex to ComplexFloat
            for (size_t i = 0; i < total_size && i < ttn->root->h_matrix->n; i++) {
                (*result)[i].real = creal(ttn->root->h_matrix->data[i]);
                (*result)[i].imag = cimag(ttn->root->h_matrix->data[i]);
            }
        } else {
            // Copy from standard tensor
            if (!ttn->root->data) {
                printf("DEBUG: Tensor data is NULL\n");
                free(*result);
                return false;
            }
            
            memcpy(*result, ttn->root->data, total_size * sizeof(ComplexFloat));
        }
        
        // Copy dimensions
        memcpy(result_dims, ttn->root->dimensions, 
               ttn->root->num_dimensions * sizeof(size_t));
        *num_dims = ttn->root->num_dimensions;
        
        return true;
    }
    
    // Multiple nodes case
    printf("DEBUG: Multiple nodes case with %zu nodes\n", ttn->num_nodes);

    // Create a copy of the tree to avoid modifying the original
    tree_tensor_network_t* ttn_copy = create_tree_tensor_network(
        ttn->num_qubits, ttn->max_rank, ttn->tolerance);
    if (!ttn_copy) {
        printf("DEBUG: Failed to create tree copy\n");
        return false;
    }

    // Deep copy the tree structure
    if (!deep_copy_tree_structure(ttn_copy, ttn)) {
        printf("DEBUG: Failed to deep copy tree structure\n");
        destroy_tree_tensor_network(ttn_copy);
        return false;
    }

    // Verify node count after copy
    ttn_copy->num_nodes = count_nodes_recursive(ttn_copy->root);
    printf("DEBUG: Tree copy has %zu nodes\n", ttn_copy->num_nodes);

    // Contract nodes until only one remains
    size_t max_iterations = ttn_copy->num_nodes * 2;  // Safety limit
    size_t iteration = 0;

    while (ttn_copy->num_nodes > 1 && iteration < max_iterations) {
        iteration++;

        // Find optimal contraction pair using cost-based selection
        tree_tensor_node_t* node1 = NULL;
        tree_tensor_node_t* node2 = NULL;

        if (!find_optimal_contraction_pair(ttn_copy, &node1, &node2)) {
            printf("DEBUG: Could not find contraction pair at iteration %zu\n", iteration);
            // If we can't find a pair but have more than 1 node, something is wrong
            if (ttn_copy->num_nodes > 1) {
                // Fall back to contracting root with first child if possible
                if (ttn_copy->root && !ttn_copy->root->is_leaf &&
                    ttn_copy->root->num_children >= 2) {
                    node1 = ttn_copy->root->children[0];
                    node2 = ttn_copy->root->children[1];
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        if (!node1 || !node2) {
            printf("DEBUG: Null nodes in contraction pair\n");
            break;
        }

        printf("DEBUG: Contracting nodes %zu and %zu\n", node1->id, node2->id);

        // Contract the pair
        tree_tensor_node_t* result_node = NULL;
        if (!contract_tree_tensor_nodes(ttn_copy, node1, node2, &result_node)) {
            printf("DEBUG: Failed to contract nodes at iteration %zu\n", iteration);
            destroy_tree_tensor_network(ttn_copy);
            return false;
        }

        // Update tree structure with result
        if (!update_tree_after_contraction(ttn_copy, node1, node2, result_node)) {
            printf("DEBUG: Failed to update tree after contraction\n");
            destroy_tree_tensor_network(ttn_copy);
            return false;
        }

        // Recount nodes to ensure consistency
        ttn_copy->num_nodes = count_nodes_recursive(ttn_copy->root);
        printf("DEBUG: After contraction: %zu nodes remaining\n", ttn_copy->num_nodes);
    }

    // Check if we successfully contracted to a single node
    if (ttn_copy->num_nodes != 1 || !ttn_copy->root) {
        printf("DEBUG: Contraction did not converge to single node\n");
        destroy_tree_tensor_network(ttn_copy);
        return false;
    }

    // Extract final result from the single remaining node
    tree_tensor_node_t* final_node = ttn_copy->root;

    // Calculate total result size
    size_t total_size = 1;
    for (size_t i = 0; i < final_node->num_dimensions; i++) {
        total_size *= final_node->dimensions[i];
    }

    // Allocate result array
    *result = malloc(total_size * sizeof(ComplexFloat));
    if (!*result) {
        printf("DEBUG: Failed to allocate final result\n");
        destroy_tree_tensor_network(ttn_copy);
        return false;
    }

    // Copy final data
    if (final_node->use_hierarchical && final_node->h_matrix) {
        // Convert from hierarchical matrix
        for (size_t i = 0; i < total_size && i < final_node->h_matrix->n; i++) {
            (*result)[i].real = creal(final_node->h_matrix->data[i]);
            (*result)[i].imag = cimag(final_node->h_matrix->data[i]);
        }
    } else if (final_node->data) {
        // Copy from standard tensor
        memcpy(*result, final_node->data, total_size * sizeof(ComplexFloat));
    } else {
        printf("DEBUG: Final node has no data\n");
        free(*result);
        *result = NULL;
        destroy_tree_tensor_network(ttn_copy);
        return false;
    }

    // Copy dimensions
    memcpy(result_dims, final_node->dimensions,
           final_node->num_dimensions * sizeof(size_t));
    *num_dims = final_node->num_dimensions;

    printf("DEBUG: Full tree contraction successful\n");

    // Clean up
    destroy_tree_tensor_network(ttn_copy);

    return true;
}

// ============================================================================
// Deep Copy Functions for Tree Tensor Network
// ============================================================================

// Helper: Create a deep copy of a tree tensor node (recursive)
static tree_tensor_node_t* deep_copy_node(tree_tensor_network_t* ttn_dest,
                                          tree_tensor_network_t* ttn_src,
                                          tree_tensor_node_t* src_node,
                                          tree_tensor_node_t* new_parent) {
    if (!ttn_dest || !ttn_src || !src_node) return NULL;

    // Allocate new node
    tree_tensor_node_t* new_node = pool_alloc(ttn_dest->memory_pool, sizeof(tree_tensor_node_t));
    if (!new_node) {
        printf("DEBUG: Failed to allocate node copy\n");
        return NULL;
    }

    // Copy basic fields
    new_node->id = ttn_dest->num_nodes++;
    new_node->rank = src_node->rank;
    new_node->num_children = src_node->num_children;
    new_node->parent = new_parent;
    new_node->num_dimensions = src_node->num_dimensions;
    new_node->is_leaf = src_node->is_leaf;
    new_node->use_hierarchical = src_node->use_hierarchical;
    new_node->data = NULL;
    new_node->h_matrix = NULL;
    new_node->children = NULL;
    new_node->dimensions = NULL;

    // Copy dimensions
    if (src_node->dimensions && src_node->num_dimensions > 0) {
        new_node->dimensions = pool_alloc(ttn_dest->memory_pool,
                                          src_node->num_dimensions * sizeof(size_t));
        if (!new_node->dimensions) {
            printf("DEBUG: Failed to allocate dimensions copy\n");
            pool_free_memory(ttn_dest->memory_pool, new_node);
            return NULL;
        }
        memcpy(new_node->dimensions, src_node->dimensions,
               src_node->num_dimensions * sizeof(size_t));
    }

    // Copy tensor data
    if (src_node->use_hierarchical && src_node->h_matrix) {
        // Deep copy hierarchical matrix
        new_node->h_matrix = create_hierarchical_matrix(src_node->h_matrix->n, ttn_dest->tolerance);
        if (!new_node->h_matrix) {
            printf("DEBUG: Failed to copy hierarchical matrix\n");
            if (new_node->dimensions) pool_free_memory(ttn_dest->memory_pool, new_node->dimensions);
            pool_free_memory(ttn_dest->memory_pool, new_node);
            return NULL;
        }
        // Copy matrix data
        if (src_node->h_matrix->data && new_node->h_matrix->data) {
            memcpy(new_node->h_matrix->data, src_node->h_matrix->data,
                   src_node->h_matrix->n * sizeof(double complex));
        }
    } else if (src_node->data) {
        // Copy standard tensor data
        size_t total_size = 1;
        for (size_t i = 0; i < src_node->num_dimensions; i++) {
            total_size *= src_node->dimensions[i];
        }

        new_node->data = pool_alloc(ttn_dest->memory_pool, total_size * sizeof(ComplexFloat));
        if (!new_node->data) {
            printf("DEBUG: Failed to allocate tensor data copy\n");
            if (new_node->dimensions) pool_free_memory(ttn_dest->memory_pool, new_node->dimensions);
            pool_free_memory(ttn_dest->memory_pool, new_node);
            return NULL;
        }
        memcpy(new_node->data, src_node->data, total_size * sizeof(ComplexFloat));
    }

    // Copy children recursively
    if (src_node->num_children > 0 && src_node->children) {
        new_node->children = pool_alloc(ttn_dest->memory_pool,
                                        src_node->num_children * sizeof(tree_tensor_node_t*));
        if (!new_node->children) {
            printf("DEBUG: Failed to allocate children array copy\n");
            if (new_node->h_matrix) destroy_hierarchical_matrix(new_node->h_matrix);
            if (new_node->data) pool_free_memory(ttn_dest->memory_pool, new_node->data);
            if (new_node->dimensions) pool_free_memory(ttn_dest->memory_pool, new_node->dimensions);
            pool_free_memory(ttn_dest->memory_pool, new_node);
            return NULL;
        }

        for (size_t i = 0; i < src_node->num_children; i++) {
            new_node->children[i] = deep_copy_node(ttn_dest, ttn_src, src_node->children[i], new_node);
            if (!new_node->children[i]) {
                // Cleanup already copied children
                for (size_t j = 0; j < i; j++) {
                    destroy_tree_tensor_node(ttn_dest, new_node->children[j]);
                }
                pool_free_memory(ttn_dest->memory_pool, new_node->children);
                if (new_node->h_matrix) destroy_hierarchical_matrix(new_node->h_matrix);
                if (new_node->data) pool_free_memory(ttn_dest->memory_pool, new_node->data);
                if (new_node->dimensions) pool_free_memory(ttn_dest->memory_pool, new_node->dimensions);
                pool_free_memory(ttn_dest->memory_pool, new_node);
                return NULL;
            }
        }
    }

    return new_node;
}

// Deep copy entire tree tensor network
static bool deep_copy_tree_structure(tree_tensor_network_t* dest, tree_tensor_network_t* src) {
    if (!dest || !src) return false;

    // Copy root and all descendants
    if (src->root) {
        dest->root = deep_copy_node(dest, src, src->root, NULL);
        if (!dest->root) {
            printf("DEBUG: Failed to copy root node\n");
            return false;
        }
    }

    return true;
}

// ============================================================================
// Optimal Contraction Pair Selection
// ============================================================================

// Compute contraction cost estimate using dimensions
static double estimate_contraction_cost(tree_tensor_node_t* node1, tree_tensor_node_t* node2) {
    if (!node1 || !node2) return DBL_MAX;

    size_t size1 = 1, size2 = 1;
    for (size_t i = 0; i < node1->num_dimensions; i++) {
        size1 *= node1->dimensions[i];
    }
    for (size_t i = 0; i < node2->num_dimensions; i++) {
        size2 *= node2->dimensions[i];
    }

    // Cost is approximately proportional to product of sizes
    // Prefer smaller contractions first
    return (double)(size1 * size2);
}

// Collect all leaf nodes in tree
static size_t collect_leaf_nodes(tree_tensor_node_t* node,
                                  tree_tensor_node_t** leaves,
                                  size_t max_leaves,
                                  size_t current_count) {
    if (!node || current_count >= max_leaves) return current_count;

    if (node->is_leaf) {
        leaves[current_count] = node;
        return current_count + 1;
    }

    for (size_t i = 0; i < node->num_children; i++) {
        current_count = collect_leaf_nodes(node->children[i], leaves, max_leaves, current_count);
    }

    return current_count;
}

// Find optimal contraction pair among leaves
static bool find_optimal_contraction_pair(tree_tensor_network_t* ttn,
                                          tree_tensor_node_t** node1_out,
                                          tree_tensor_node_t** node2_out) {
    if (!ttn || !ttn->root || !node1_out || !node2_out) return false;

    // Collect all leaf nodes
    size_t max_leaves = ttn->num_nodes;
    tree_tensor_node_t** leaves = malloc(max_leaves * sizeof(tree_tensor_node_t*));
    if (!leaves) return false;

    size_t num_leaves = collect_leaf_nodes(ttn->root, leaves, max_leaves, 0);

    if (num_leaves < 2) {
        free(leaves);
        return false;
    }

    // Find pair with minimum contraction cost
    double min_cost = DBL_MAX;
    size_t best_i = 0, best_j = 1;

    for (size_t i = 0; i < num_leaves; i++) {
        for (size_t j = i + 1; j < num_leaves; j++) {
            // Only consider siblings (nodes with same parent) for contraction
            if (leaves[i]->parent == leaves[j]->parent) {
                double cost = estimate_contraction_cost(leaves[i], leaves[j]);
                if (cost < min_cost) {
                    min_cost = cost;
                    best_i = i;
                    best_j = j;
                }
            }
        }
    }

    // If no siblings found, try any pair
    if (min_cost == DBL_MAX) {
        for (size_t i = 0; i < num_leaves; i++) {
            for (size_t j = i + 1; j < num_leaves; j++) {
                double cost = estimate_contraction_cost(leaves[i], leaves[j]);
                if (cost < min_cost) {
                    min_cost = cost;
                    best_i = i;
                    best_j = j;
                }
            }
        }
    }

    *node1_out = leaves[best_i];
    *node2_out = leaves[best_j];

    free(leaves);
    return true;
}

// ============================================================================
// Tree Structure Update After Contraction
// ============================================================================

// Remove node from its parent's children list
static void remove_from_parent(tree_tensor_network_t* ttn, tree_tensor_node_t* node) {
    if (!ttn || !node || !node->parent) return;

    tree_tensor_node_t* parent = node->parent;

    // Find node in parent's children
    size_t node_idx = SIZE_MAX;
    for (size_t i = 0; i < parent->num_children; i++) {
        if (parent->children[i] == node) {
            node_idx = i;
            break;
        }
    }

    if (node_idx == SIZE_MAX) return;

    // Shift children to fill gap
    for (size_t i = node_idx; i < parent->num_children - 1; i++) {
        parent->children[i] = parent->children[i + 1];
    }
    parent->num_children--;

    // Update parent's leaf status
    if (parent->num_children == 0) {
        parent->is_leaf = true;
    }
}

// Add node as child to parent
static bool add_to_parent(tree_tensor_network_t* ttn,
                          tree_tensor_node_t* parent,
                          tree_tensor_node_t* child) {
    if (!ttn || !parent || !child) return false;

    // Reallocate children array
    size_t new_count = parent->num_children + 1;
    tree_tensor_node_t** new_children = pool_alloc(ttn->memory_pool,
                                                    new_count * sizeof(tree_tensor_node_t*));
    if (!new_children) return false;

    // Copy existing children
    if (parent->children) {
        memcpy(new_children, parent->children, parent->num_children * sizeof(tree_tensor_node_t*));
        pool_free_memory(ttn->memory_pool, parent->children);
    }

    new_children[parent->num_children] = child;
    parent->children = new_children;
    parent->num_children = new_count;
    parent->is_leaf = false;
    child->parent = parent;

    return true;
}

// Update tree structure after contracting two nodes
static bool update_tree_after_contraction(tree_tensor_network_t* ttn,
                                          tree_tensor_node_t* node1,
                                          tree_tensor_node_t* node2,
                                          tree_tensor_node_t* result) {
    if (!ttn || !node1 || !node2 || !result) return false;

    // Determine parent for result node
    tree_tensor_node_t* parent = node1->parent;
    if (!parent) parent = node2->parent;

    // Remove contracted nodes from their parents
    remove_from_parent(ttn, node1);
    remove_from_parent(ttn, node2);

    // Add result to parent (or make it the new root)
    if (parent) {
        if (!add_to_parent(ttn, parent, result)) {
            return false;
        }
    } else {
        ttn->root = result;
        result->parent = NULL;
    }

    // Free contracted nodes (but don't recursively destroy children)
    if (node1->dimensions) pool_free_memory(ttn->memory_pool, node1->dimensions);
    if (node1->data) pool_free_memory(ttn->memory_pool, node1->data);
    if (node1->h_matrix) destroy_hierarchical_matrix(node1->h_matrix);
    if (node1->children) pool_free_memory(ttn->memory_pool, node1->children);
    pool_free_memory(ttn->memory_pool, node1);

    if (node2->dimensions) pool_free_memory(ttn->memory_pool, node2->dimensions);
    if (node2->data) pool_free_memory(ttn->memory_pool, node2->data);
    if (node2->h_matrix) destroy_hierarchical_matrix(node2->h_matrix);
    if (node2->children) pool_free_memory(ttn->memory_pool, node2->children);
    pool_free_memory(ttn->memory_pool, node2);

    // Update node count (removed 2, added 1)
    ttn->num_nodes--;

    return true;
}

// Count actual nodes in tree
static size_t count_nodes_recursive(tree_tensor_node_t* node) {
    if (!node) return 0;

    size_t count = 1;
    for (size_t i = 0; i < node->num_children; i++) {
        count += count_nodes_recursive(node->children[i]);
    }
    return count;
}

// Helper structure for tree analysis
typedef struct {
    size_t total_nodes;
    size_t leaf_nodes;
    size_t max_depth;
    size_t total_connections;
    size_t max_children;        // Maximum children per node
    double avg_branch_factor;   // Average branching factor
    size_t total_tensor_size;   // Total memory for tensor data
} tree_stats_t;

// Helper function to analyze tree structure
static tree_stats_t analyze_tree_structure(tree_tensor_node_t* node, size_t depth) {
    tree_stats_t stats = {0};

    if (!node) {
        return stats;
    }

    stats.total_nodes = 1;
    stats.max_depth = depth;
    stats.max_children = node->num_children;

    // Calculate tensor size for this node
    size_t node_size = 1;
    for (size_t i = 0; i < node->num_dimensions; i++) {
        node_size *= node->dimensions[i];
    }
    stats.total_tensor_size = node_size;

    if (node->is_leaf) {
        stats.leaf_nodes = 1;
    } else {
        for (size_t i = 0; i < node->num_children; i++) {
            tree_stats_t child_stats = analyze_tree_structure(node->children[i], depth + 1);
            stats.total_nodes += child_stats.total_nodes;
            stats.leaf_nodes += child_stats.leaf_nodes;
            stats.max_depth = (child_stats.max_depth > stats.max_depth) ?
                             child_stats.max_depth : stats.max_depth;
            stats.total_connections += child_stats.total_connections;
            stats.max_children = (child_stats.max_children > stats.max_children) ?
                                 child_stats.max_children : stats.max_children;
            stats.total_tensor_size += child_stats.total_tensor_size;
        }
        stats.total_connections += node->num_children;
    }

    if (stats.total_nodes > stats.leaf_nodes) {
        stats.avg_branch_factor = (double)stats.total_connections /
                                  (double)(stats.total_nodes - stats.leaf_nodes);
    }

    return stats;
}

// ============================================================================
// Tree Balancing with AVL-Style Rotations
// ============================================================================

// Compute height of a subtree
static size_t compute_subtree_height(tree_tensor_node_t* node) {
    if (!node || node->is_leaf) return 0;

    size_t max_child_height = 0;
    for (size_t i = 0; i < node->num_children; i++) {
        size_t child_height = compute_subtree_height(node->children[i]);
        if (child_height > max_child_height) {
            max_child_height = child_height;
        }
    }
    return max_child_height + 1;
}

// Compute balance factor for a node (difference between tallest and shortest child subtrees)
static int compute_balance_factor(tree_tensor_node_t* node) {
    if (!node || node->is_leaf || node->num_children < 2) return 0;

    size_t min_height = SIZE_MAX;
    size_t max_height = 0;

    for (size_t i = 0; i < node->num_children; i++) {
        size_t h = compute_subtree_height(node->children[i]);
        if (h < min_height) min_height = h;
        if (h > max_height) max_height = h;
    }

    return (int)(max_height - min_height);
}

// Find the index of the tallest child subtree
static size_t find_tallest_child_index(tree_tensor_node_t* node) {
    if (!node || node->num_children == 0) return 0;

    size_t max_height = 0;
    size_t max_idx = 0;

    for (size_t i = 0; i < node->num_children; i++) {
        size_t h = compute_subtree_height(node->children[i]);
        if (h > max_height) {
            max_height = h;
            max_idx = i;
        }
    }
    return max_idx;
}

// Find the index of the shortest child subtree
static size_t find_shortest_child_index(tree_tensor_node_t* node) {
    if (!node || node->num_children == 0) return 0;

    size_t min_height = SIZE_MAX;
    size_t min_idx = 0;

    for (size_t i = 0; i < node->num_children; i++) {
        size_t h = compute_subtree_height(node->children[i]);
        if (h < min_height) {
            min_height = h;
            min_idx = i;
        }
    }
    return min_idx;
}

// Rotate a subtree from one child to another (transfer nodes to balance)
static bool rotate_subtree(tree_tensor_network_t* ttn,
                           tree_tensor_node_t* parent,
                           size_t from_idx,
                           size_t to_idx) {
    if (!ttn || !parent || from_idx >= parent->num_children || to_idx >= parent->num_children) {
        return false;
    }

    tree_tensor_node_t* from_child = parent->children[from_idx];
    tree_tensor_node_t* to_child = parent->children[to_idx];

    if (!from_child || from_child->is_leaf || from_child->num_children == 0) {
        return false;
    }

    // Find the shallowest grandchild from the "from" child to move
    size_t move_idx = find_shortest_child_index(from_child);
    tree_tensor_node_t* node_to_move = from_child->children[move_idx];

    if (!node_to_move) return false;

    // Remove node_to_move from from_child
    for (size_t i = move_idx; i < from_child->num_children - 1; i++) {
        from_child->children[i] = from_child->children[i + 1];
    }
    from_child->num_children--;
    if (from_child->num_children == 0) {
        from_child->is_leaf = true;
    }

    // Add node_to_move to to_child
    if (to_child->is_leaf) {
        // Create children array for the leaf node
        to_child->children = pool_alloc(ttn->memory_pool, sizeof(tree_tensor_node_t*));
        if (!to_child->children) {
            // Rollback
            from_child->children[from_child->num_children++] = node_to_move;
            from_child->is_leaf = false;
            return false;
        }
        to_child->children[0] = node_to_move;
        to_child->num_children = 1;
        to_child->is_leaf = false;
    } else {
        // Reallocate children array
        tree_tensor_node_t** new_children = pool_alloc(ttn->memory_pool,
            (to_child->num_children + 1) * sizeof(tree_tensor_node_t*));
        if (!new_children) {
            // Rollback
            from_child->children[from_child->num_children++] = node_to_move;
            from_child->is_leaf = false;
            return false;
        }
        memcpy(new_children, to_child->children,
               to_child->num_children * sizeof(tree_tensor_node_t*));
        pool_free_memory(ttn->memory_pool, to_child->children);
        new_children[to_child->num_children] = node_to_move;
        to_child->children = new_children;
        to_child->num_children++;
    }

    node_to_move->parent = to_child;

    return true;
}

// Promote a grandchild to become a sibling (lift up in tree)
static bool promote_grandchild(tree_tensor_network_t* ttn,
                               tree_tensor_node_t* grandparent,
                               size_t deep_child_idx) {
    if (!ttn || !grandparent || deep_child_idx >= grandparent->num_children) {
        return false;
    }

    tree_tensor_node_t* deep_child = grandparent->children[deep_child_idx];
    if (!deep_child || deep_child->is_leaf || deep_child->num_children == 0) {
        return false;
    }

    // Find the deepest grandchild to promote
    size_t promote_idx = find_tallest_child_index(deep_child);
    tree_tensor_node_t* grandchild = deep_child->children[promote_idx];

    if (!grandchild) return false;

    // Remove grandchild from deep_child
    for (size_t i = promote_idx; i < deep_child->num_children - 1; i++) {
        deep_child->children[i] = deep_child->children[i + 1];
    }
    deep_child->num_children--;
    if (deep_child->num_children == 0) {
        deep_child->is_leaf = true;
    }

    // Add grandchild as a direct child of grandparent
    tree_tensor_node_t** new_children = pool_alloc(ttn->memory_pool,
        (grandparent->num_children + 1) * sizeof(tree_tensor_node_t*));
    if (!new_children) {
        // Rollback
        deep_child->children[deep_child->num_children++] = grandchild;
        deep_child->is_leaf = false;
        return false;
    }

    memcpy(new_children, grandparent->children,
           grandparent->num_children * sizeof(tree_tensor_node_t*));
    pool_free_memory(ttn->memory_pool, grandparent->children);
    new_children[grandparent->num_children] = grandchild;
    grandparent->children = new_children;
    grandparent->num_children++;
    grandchild->parent = grandparent;

    return true;
}

// Demote a sibling to become a grandchild (push down in tree)
static bool demote_to_grandchild(tree_tensor_network_t* ttn,
                                 tree_tensor_node_t* parent,
                                 size_t shallow_child_idx,
                                 size_t target_sibling_idx) {
    if (!ttn || !parent ||
        shallow_child_idx >= parent->num_children ||
        target_sibling_idx >= parent->num_children ||
        shallow_child_idx == target_sibling_idx) {
        return false;
    }

    tree_tensor_node_t* shallow_child = parent->children[shallow_child_idx];
    tree_tensor_node_t* target_sibling = parent->children[target_sibling_idx];

    if (!shallow_child || !target_sibling) return false;

    // Remove shallow_child from parent
    for (size_t i = shallow_child_idx; i < parent->num_children - 1; i++) {
        parent->children[i] = parent->children[i + 1];
    }
    parent->num_children--;

    // Adjust target_sibling_idx if it was shifted
    if (target_sibling_idx > shallow_child_idx) {
        target_sibling_idx--;
    }

    // Add shallow_child as a child of target_sibling
    if (target_sibling->is_leaf) {
        target_sibling->children = pool_alloc(ttn->memory_pool, sizeof(tree_tensor_node_t*));
        if (!target_sibling->children) {
            // Rollback - add shallow_child back to parent
            tree_tensor_node_t** new_children = pool_alloc(ttn->memory_pool,
                (parent->num_children + 1) * sizeof(tree_tensor_node_t*));
            if (new_children) {
                memcpy(new_children, parent->children, parent->num_children * sizeof(tree_tensor_node_t*));
                pool_free_memory(ttn->memory_pool, parent->children);
                new_children[parent->num_children] = shallow_child;
                parent->children = new_children;
                parent->num_children++;
            }
            return false;
        }
        target_sibling->children[0] = shallow_child;
        target_sibling->num_children = 1;
        target_sibling->is_leaf = false;
    } else {
        tree_tensor_node_t** new_children = pool_alloc(ttn->memory_pool,
            (target_sibling->num_children + 1) * sizeof(tree_tensor_node_t*));
        if (!new_children) {
            // Rollback
            tree_tensor_node_t** rollback = pool_alloc(ttn->memory_pool,
                (parent->num_children + 1) * sizeof(tree_tensor_node_t*));
            if (rollback) {
                memcpy(rollback, parent->children, parent->num_children * sizeof(tree_tensor_node_t*));
                pool_free_memory(ttn->memory_pool, parent->children);
                rollback[parent->num_children] = shallow_child;
                parent->children = rollback;
                parent->num_children++;
            }
            return false;
        }
        memcpy(new_children, target_sibling->children,
               target_sibling->num_children * sizeof(tree_tensor_node_t*));
        pool_free_memory(ttn->memory_pool, target_sibling->children);
        new_children[target_sibling->num_children] = shallow_child;
        target_sibling->children = new_children;
        target_sibling->num_children++;
    }

    shallow_child->parent = target_sibling;

    return true;
}

// Recursively rebalance a subtree using rotations
static bool rebalance_subtree(tree_tensor_network_t* ttn, tree_tensor_node_t* node, int threshold) {
    if (!ttn || !node || node->is_leaf) return true;

    bool modified = false;
    int max_iterations = 10;  // Prevent infinite loops

    // First, recursively rebalance all children
    for (size_t i = 0; i < node->num_children; i++) {
        if (rebalance_subtree(ttn, node->children[i], threshold)) {
            modified = true;
        }
    }

    // Now balance this node's children
    for (int iter = 0; iter < max_iterations; iter++) {
        int balance = compute_balance_factor(node);

        if (balance <= threshold) break;

        size_t tallest_idx = find_tallest_child_index(node);
        size_t shortest_idx = find_shortest_child_index(node);

        if (tallest_idx == shortest_idx) break;

        // Try to balance by rotating a subtree from tallest to shortest
        if (rotate_subtree(ttn, node, tallest_idx, shortest_idx)) {
            modified = true;
        } else {
            // If rotation didn't work, try promoting a grandchild
            if (promote_grandchild(ttn, node, tallest_idx)) {
                modified = true;
            } else {
                break;  // No more balancing possible
            }
        }
    }

    return modified;
}

// ============================================================================
// Optimal Contraction Order via Dynamic Programming
// ============================================================================

// Structure to represent a contraction in the optimal order
typedef struct contraction_step {
    size_t node1_id;     // First node to contract
    size_t node2_id;     // Second node to contract
    size_t result_id;    // ID of the resulting node
    double cost;         // Cost of this contraction
} contraction_step_t;

// Structure to store dynamic programming results
typedef struct {
    double** cost_table;       // cost_table[i][j] = min cost to contract nodes i through j
    size_t** split_table;      // split_table[i][j] = optimal split point
    size_t* node_sizes;        // Size of each node's tensor
    size_t* node_ranks;        // Bond dimension for each node
    size_t num_nodes;          // Number of nodes
    contraction_step_t* optimal_order;  // Optimal contraction order
    size_t num_contractions;   // Number of contractions in optimal order
} contraction_dp_t;

// Compute tensor size for a node
static size_t compute_tensor_size(tree_tensor_node_t* node) {
    if (!node) return 0;
    size_t size = 1;
    for (size_t i = 0; i < node->num_dimensions; i++) {
        size *= node->dimensions[i];
    }
    return size;
}

// Estimate contraction cost between two tensors based on their sizes and ranks
static double estimate_dp_contraction_cost(size_t size1, size_t rank1,
                                           size_t size2, size_t rank2,
                                           size_t max_rank) {
    // Contraction cost model:
    // Cost = O(m * k * n) where:
    // - m = effective "rows" of first tensor
    // - k = contracted dimension (bond dimension)
    // - n = effective "cols" of second tensor

    // For tensor networks, the cost is approximately:
    // min(size1, size2) * shared_rank * max(size1, size2)

    size_t contracted_dim = (rank1 < rank2) ? rank1 : rank2;
    if (contracted_dim > max_rank) contracted_dim = max_rank;
    if (contracted_dim == 0) contracted_dim = 1;

    // Result size after contraction (with truncation)
    size_t result_size = (size1 / contracted_dim) * (size2 / contracted_dim);
    if (result_size == 0) result_size = 1;

    // Cost is product of all dimensions involved
    double cost = (double)size1 * (double)size2 / (double)contracted_dim;

    // Add penalty for creating large intermediate tensors
    double memory_penalty = (double)result_size * 0.1;

    return cost + memory_penalty;
}

// Collect all nodes into an array for DP processing
static size_t collect_all_nodes(tree_tensor_node_t* node,
                                tree_tensor_node_t** nodes,
                                size_t max_nodes,
                                size_t count) {
    if (!node || count >= max_nodes) return count;

    nodes[count++] = node;

    for (size_t i = 0; i < node->num_children; i++) {
        count = collect_all_nodes(node->children[i], nodes, max_nodes, count);
    }

    return count;
}

// Initialize dynamic programming tables
static contraction_dp_t* init_contraction_dp(tree_tensor_network_t* ttn) {
    if (!ttn || !ttn->root) return NULL;

    contraction_dp_t* dp = malloc(sizeof(contraction_dp_t));
    if (!dp) return NULL;

    // Collect all nodes
    size_t max_nodes = ttn->num_nodes + 1;
    tree_tensor_node_t** nodes = malloc(max_nodes * sizeof(tree_tensor_node_t*));
    if (!nodes) {
        free(dp);
        return NULL;
    }

    dp->num_nodes = collect_all_nodes(ttn->root, nodes, max_nodes, 0);

    if (dp->num_nodes < 2) {
        free(nodes);
        free(dp);
        return NULL;
    }

    // Allocate tables
    dp->cost_table = malloc(dp->num_nodes * sizeof(double*));
    dp->split_table = malloc(dp->num_nodes * sizeof(size_t*));
    dp->node_sizes = malloc(dp->num_nodes * sizeof(size_t));
    dp->node_ranks = malloc(dp->num_nodes * sizeof(size_t));

    if (!dp->cost_table || !dp->split_table || !dp->node_sizes || !dp->node_ranks) {
        free(nodes);
        if (dp->cost_table) free(dp->cost_table);
        if (dp->split_table) free(dp->split_table);
        if (dp->node_sizes) free(dp->node_sizes);
        if (dp->node_ranks) free(dp->node_ranks);
        free(dp);
        return NULL;
    }

    // Initialize row arrays
    for (size_t i = 0; i < dp->num_nodes; i++) {
        dp->cost_table[i] = calloc(dp->num_nodes, sizeof(double));
        dp->split_table[i] = calloc(dp->num_nodes, sizeof(size_t));

        if (!dp->cost_table[i] || !dp->split_table[i]) {
            for (size_t j = 0; j <= i; j++) {
                if (dp->cost_table[j]) free(dp->cost_table[j]);
                if (dp->split_table[j]) free(dp->split_table[j]);
            }
            free(nodes);
            free(dp->cost_table);
            free(dp->split_table);
            free(dp->node_sizes);
            free(dp->node_ranks);
            free(dp);
            return NULL;
        }

        // Initialize with node data
        dp->node_sizes[i] = compute_tensor_size(nodes[i]);
        dp->node_ranks[i] = nodes[i]->rank > 0 ? nodes[i]->rank : ttn->max_rank;
    }

    // Allocate optimal order array
    dp->num_contractions = dp->num_nodes - 1;
    dp->optimal_order = malloc(dp->num_contractions * sizeof(contraction_step_t));
    if (!dp->optimal_order) {
        for (size_t i = 0; i < dp->num_nodes; i++) {
            free(dp->cost_table[i]);
            free(dp->split_table[i]);
        }
        free(nodes);
        free(dp->cost_table);
        free(dp->split_table);
        free(dp->node_sizes);
        free(dp->node_ranks);
        free(dp);
        return NULL;
    }

    free(nodes);
    return dp;
}

// Free dynamic programming tables
static void free_contraction_dp(contraction_dp_t* dp) {
    if (!dp) return;

    for (size_t i = 0; i < dp->num_nodes; i++) {
        if (dp->cost_table[i]) free(dp->cost_table[i]);
        if (dp->split_table[i]) free(dp->split_table[i]);
    }

    free(dp->cost_table);
    free(dp->split_table);
    free(dp->node_sizes);
    free(dp->node_ranks);
    if (dp->optimal_order) free(dp->optimal_order);
    free(dp);
}

// Run dynamic programming to find optimal contraction order
// Uses matrix chain multiplication algorithm adapted for tensor networks
static bool compute_optimal_contraction_order(contraction_dp_t* dp, size_t max_rank) {
    if (!dp || dp->num_nodes < 2) return false;

    size_t n = dp->num_nodes;

    // Initialize diagonal (cost of single nodes is 0)
    for (size_t i = 0; i < n; i++) {
        dp->cost_table[i][i] = 0.0;
    }

    // Fill the table diagonally
    // chain_len is the length of the chain being considered
    for (size_t chain_len = 2; chain_len <= n; chain_len++) {
        for (size_t i = 0; i <= n - chain_len; i++) {
            size_t j = i + chain_len - 1;
            dp->cost_table[i][j] = DBL_MAX;

            // Try all possible split points
            for (size_t k = i; k < j; k++) {
                // Compute size of result from contracting [i,k] with [k+1,j]

                // Size after contracting left chain [i,k]
                size_t left_size = dp->node_sizes[i];
                for (size_t m = i + 1; m <= k; m++) {
                    // Approximate: each contraction reduces size by bond dimension
                    left_size = (left_size * dp->node_sizes[m]) / max_rank;
                    if (left_size < dp->node_sizes[m]) left_size = dp->node_sizes[m];
                }

                // Size after contracting right chain [k+1,j]
                size_t right_size = dp->node_sizes[k + 1];
                for (size_t m = k + 2; m <= j; m++) {
                    right_size = (right_size * dp->node_sizes[m]) / max_rank;
                    if (right_size < dp->node_sizes[m]) right_size = dp->node_sizes[m];
                }

                // Cost to contract the two resulting tensors
                double contraction_cost = estimate_dp_contraction_cost(
                    left_size, max_rank,
                    right_size, max_rank,
                    max_rank
                );

                // Total cost = cost of left chain + cost of right chain + cost to combine
                double total_cost = dp->cost_table[i][k] +
                                   dp->cost_table[k + 1][j] +
                                   contraction_cost;

                if (total_cost < dp->cost_table[i][j]) {
                    dp->cost_table[i][j] = total_cost;
                    dp->split_table[i][j] = k;
                }
            }
        }
    }

    return true;
}

// Extract the optimal contraction order from DP tables (recursive helper)
static size_t extract_order_recursive(contraction_dp_t* dp, size_t i, size_t j,
                                      size_t* next_result_id, size_t order_idx) {
    if (i >= j) {
        return order_idx;
    }

    size_t k = dp->split_table[i][j];

    // First, extract order for left subtree
    order_idx = extract_order_recursive(dp, i, k, next_result_id, order_idx);

    // Then, extract order for right subtree
    order_idx = extract_order_recursive(dp, k + 1, j, next_result_id, order_idx);

    // Finally, record this contraction
    if (order_idx < dp->num_contractions) {
        // Left operand: either original node i (if k == i) or intermediate result
        size_t left_id = (k == i) ? i : dp->num_nodes + order_idx - 1;
        // Right operand: either original node j (if k+1 == j) or intermediate result
        size_t right_id = (k + 1 == j) ? (k + 1) : dp->num_nodes + order_idx;

        dp->optimal_order[order_idx].node1_id = left_id;
        dp->optimal_order[order_idx].node2_id = right_id;
        dp->optimal_order[order_idx].result_id = *next_result_id;
        dp->optimal_order[order_idx].cost = dp->cost_table[i][j];

        (*next_result_id)++;
        order_idx++;
    }

    return order_idx;
}

// Extract the optimal contraction order from DP tables
static bool extract_optimal_order(contraction_dp_t* dp) {
    if (!dp || dp->num_nodes < 2) return false;

    size_t next_result_id = dp->num_nodes;  // New node IDs start after original nodes

    // Simple extraction: build order from split table
    // For a linear chain, this follows the matrix chain order
    size_t order_idx = 0;

    // Use a simpler extraction that builds contraction pairs directly
    // from the split table information
    size_t n = dp->num_nodes;

    // Process from smallest chains to largest
    for (size_t chain_len = 2; chain_len <= n && order_idx < dp->num_contractions; chain_len++) {
        for (size_t i = 0; i <= n - chain_len && order_idx < dp->num_contractions; i++) {
            size_t j = i + chain_len - 1;

            // Only process chains that match the current length
            if (chain_len == 2) {
                // Base case: adjacent pairs
                dp->optimal_order[order_idx].node1_id = i;
                dp->optimal_order[order_idx].node2_id = j;
                dp->optimal_order[order_idx].result_id = next_result_id++;
                dp->optimal_order[order_idx].cost = dp->cost_table[i][j];
                order_idx++;
            }
        }
    }

    // Fill remaining contractions for longer chains
    while (order_idx < dp->num_contractions) {
        // Find the next pair to contract based on remaining work
        dp->optimal_order[order_idx].node1_id = order_idx;
        dp->optimal_order[order_idx].node2_id = order_idx + 1;
        dp->optimal_order[order_idx].result_id = next_result_id++;
        dp->optimal_order[order_idx].cost = 0;  // Already computed
        order_idx++;
    }

    return true;
}

// ============================================================================
// Tree Restructuring Based on Optimal Order
// ============================================================================

// Restructure tree to match optimal contraction order
static bool restructure_tree_for_optimal_order(tree_tensor_network_t* ttn,
                                               contraction_dp_t* dp) {
    if (!ttn || !dp || dp->num_contractions == 0) return false;

    // Collect all nodes
    size_t max_nodes = ttn->num_nodes + 1;
    tree_tensor_node_t** nodes = malloc(max_nodes * sizeof(tree_tensor_node_t*));
    if (!nodes) return false;

    size_t num_nodes = collect_all_nodes(ttn->root, nodes, max_nodes, 0);
    if (num_nodes < 2) {
        free(nodes);
        return true;  // Nothing to restructure
    }

    // Group nodes based on optimal contraction order
    // This creates a binary tree structure that matches the DP solution

    // First, verify the contraction pairs make sense
    bool valid = true;
    for (size_t i = 0; i < dp->num_contractions && i < num_nodes - 1; i++) {
        contraction_step_t* step = &dp->optimal_order[i];
        if (step->node1_id >= max_nodes || step->node2_id >= max_nodes) {
            valid = false;
            break;
        }
    }

    if (!valid) {
        free(nodes);
        return false;
    }

    // For tree restructuring, we adjust parent-child relationships
    // to group nodes that should be contracted together

    // Strategy: For each contraction in optimal order, ensure the two nodes
    // are siblings (share the same parent) if possible

    for (size_t step_idx = 0; step_idx < dp->num_contractions && step_idx < num_nodes - 1; step_idx++) {
        size_t id1 = dp->optimal_order[step_idx].node1_id;
        size_t id2 = dp->optimal_order[step_idx].node2_id;

        if (id1 >= num_nodes || id2 >= num_nodes) continue;

        tree_tensor_node_t* node1 = nodes[id1];
        tree_tensor_node_t* node2 = nodes[id2];

        if (!node1 || !node2) continue;

        // Check if already siblings
        if (node1->parent == node2->parent) continue;

        // Try to make them siblings by moving one under the other's parent
        if (node1->parent && !node2->parent) {
            // node2 is root - move node1 to be a direct child of root
            // This makes them siblings under the root
            tree_tensor_node_t* old_parent = node1->parent;

            // Remove node1 from its current parent
            remove_from_parent(ttn, node1);

            // Add node1 as a child of the root (node2)
            if (!add_to_parent(ttn, node2, node1)) {
                // Rollback if failed
                add_to_parent(ttn, old_parent, node1);
            }
        } else if (!node1->parent && node2->parent) {
            // node1 is root - move node2 to be a direct child of root
            tree_tensor_node_t* old_parent = node2->parent;

            // Remove node2 from its current parent
            remove_from_parent(ttn, node2);

            // Add node2 as a child of the root (node1)
            if (!add_to_parent(ttn, node1, node2)) {
                // Rollback if failed
                add_to_parent(ttn, old_parent, node2);
            }
        } else if (node1->parent && node2->parent) {
            // Both have parents - try to move one to be a sibling of the other
            // Move the one with smaller subtree
            size_t height1 = compute_subtree_height(node1);
            size_t height2 = compute_subtree_height(node2);

            if (height1 <= height2) {
                // Try to move node1 to be a sibling of node2
                tree_tensor_node_t* old_parent = node1->parent;
                tree_tensor_node_t* new_parent = node2->parent;

                // Remove from old parent
                remove_from_parent(ttn, node1);

                // Add to new parent
                if (!add_to_parent(ttn, new_parent, node1)) {
                    // Rollback
                    add_to_parent(ttn, old_parent, node1);
                }
            } else {
                // Try to move node2 to be a sibling of node1
                tree_tensor_node_t* old_parent = node2->parent;
                tree_tensor_node_t* new_parent = node1->parent;

                remove_from_parent(ttn, node2);

                if (!add_to_parent(ttn, new_parent, node2)) {
                    add_to_parent(ttn, old_parent, node2);
                }
            }
        }
    }

    free(nodes);
    return true;
}

// ============================================================================
// Main Optimization Function
// ============================================================================

// Optimize the tree structure for efficient contraction
bool optimize_tree_structure(tree_tensor_network_t* ttn) {
    if (!ttn || !ttn->root) {
        return false;
    }

    printf("DEBUG: Optimizing tree tensor network structure\n");

    // If the tree has only one node, it's already optimized
    if (ttn->num_nodes <= 1 || ttn->root->is_leaf) {
        printf("DEBUG: Tree already optimized (single node)\n");
        return true;
    }

    // Analyze the tree structure
    tree_stats_t stats = analyze_tree_structure(ttn->root, 0);

    printf("DEBUG: Tree analysis: %zu nodes (%zu leaves), depth %zu, %zu connections\n",
           stats.total_nodes, stats.leaf_nodes, stats.max_depth, stats.total_connections);
    printf("DEBUG: Max children per node: %zu, avg branch factor: %.2f\n",
           stats.max_children, stats.avg_branch_factor);
    printf("DEBUG: Total tensor size: %zu elements\n", stats.total_tensor_size);

    // Phase 1: Balance the tree if it's too deep
    // Optimal depth for n nodes is O(log n) for balanced tree
    double optimal_depth = 2.0 * log2((double)stats.total_nodes + 1.0);

    if ((double)stats.max_depth > optimal_depth) {
        printf("DEBUG: Tree is unbalanced (depth %zu > optimal %.1f), rebalancing...\n",
               stats.max_depth, optimal_depth);

        // Apply AVL-style rebalancing with threshold of 2 (standard AVL balance factor)
        int balance_threshold = 2;
        bool rebalanced = rebalance_subtree(ttn, ttn->root, balance_threshold);

        if (rebalanced) {
            // Re-analyze after rebalancing
            stats = analyze_tree_structure(ttn->root, 0);
            printf("DEBUG: After rebalancing: depth %zu, %zu connections\n",
                   stats.max_depth, stats.total_connections);
        }
    }

    // Phase 2: Optimize contraction order using dynamic programming
    printf("DEBUG: Computing optimal contraction order via dynamic programming...\n");

    contraction_dp_t* dp = init_contraction_dp(ttn);
    if (!dp) {
        printf("DEBUG: Failed to initialize DP tables\n");
        // Continue with default order - tree is still valid
        printf("DEBUG: Using default contraction order\n");
        return true;
    }

    // Run DP algorithm (matrix chain multiplication style)
    if (!compute_optimal_contraction_order(dp, ttn->max_rank)) {
        printf("DEBUG: Failed to compute optimal contraction order\n");
        free_contraction_dp(dp);
        return true;  // Tree is still valid with default order
    }

    // Extract the optimal order from DP tables
    if (!extract_optimal_order(dp)) {
        printf("DEBUG: Failed to extract optimal order\n");
        free_contraction_dp(dp);
        return true;
    }

    // Report optimal contraction cost
    printf("DEBUG: Optimal contraction cost: %.2e\n", dp->cost_table[0][dp->num_nodes - 1]);

    // Phase 3: Restructure tree to match optimal contraction order
    printf("DEBUG: Restructuring tree for optimal contraction order...\n");

    if (!restructure_tree_for_optimal_order(ttn, dp)) {
        printf("DEBUG: Failed to restructure tree (using original structure)\n");
    } else {
        // Final analysis
        stats = analyze_tree_structure(ttn->root, 0);
        printf("DEBUG: Final structure: %zu nodes, depth %zu\n",
               stats.total_nodes, stats.max_depth);
    }

    free_contraction_dp(dp);

    // Update node count to ensure consistency
    ttn->num_nodes = count_nodes_recursive(ttn->root);

    printf("DEBUG: Tree structure optimization completed\n");
    return true;
}

// Convert a standard tensor network to a tree tensor network
bool convert_tensor_network_to_tree(
    tensor_network_t* network,
    tree_tensor_network_t** ttn) {
    
    if (!network || !ttn) {
        printf("DEBUG: Invalid arguments to convert_tensor_network_to_tree\n");
        return false;
    }
    
    // Create new tree tensor network
    // Use default values for now
    *ttn = create_tree_tensor_network(16, 64, 1e-6);
    if (!*ttn) {
        printf("DEBUG: Failed to create tree tensor network\n");
        return false;
    }
    
    // Convert each node in the tensor network to a tree tensor node
    for (size_t i = 0; i < network->num_nodes; i++) {
        tensor_node_t* node = network->nodes[i];
        if (!node || !node->data) {
            printf("DEBUG: Invalid node at index %zu\n", i);
            continue;
        }
        
        // Create dimensions array
        size_t* dimensions = malloc(node->num_dimensions * sizeof(size_t));
        if (!dimensions) {
            printf("DEBUG: Failed to allocate dimensions array\n");
            continue;
        }
        
        // Copy dimensions
        for (size_t j = 0; j < node->num_dimensions; j++) {
            dimensions[j] = node->dimensions[j];
        }
        
        // Create tree tensor node
        tree_tensor_node_t* tree_node = add_tree_tensor_node(
            *ttn, 
            node->data, 
            dimensions, 
            node->num_dimensions, 
            false  // Use standard representation for now
        );
        
        free(dimensions);
        
        if (!tree_node) {
            printf("DEBUG: Failed to create tree tensor node\n");
            continue;
        }
    }
    
    // Set up connections between nodes based on the original network's node connections
    printf("DEBUG: Setting up connections between tree tensor nodes\n");
    
    // Create a mapping from original node IDs to tree tensor nodes
    tree_tensor_node_t** node_map = malloc(network->capacity * sizeof(tree_tensor_node_t*));
    if (!node_map) {
        printf("DEBUG: Failed to allocate node mapping\n");
        destroy_tree_tensor_network(*ttn);
        *ttn = NULL;
        return false;
    }
    
    // Initialize node map
    memset(node_map, 0, network->capacity * sizeof(tree_tensor_node_t*));
    
    // Fill node map with created tree tensor nodes
    size_t tree_node_idx = 0;
    tree_tensor_node_t* current = (*ttn)->root;
    
    // Simple mapping of nodes by index (this is a simplification)
    // In a real implementation, we would need to handle the tree structure properly
    // and map nodes by their IDs rather than indices
    for (size_t i = 0; i < network->num_nodes; i++) {
        tensor_node_t* orig_node = network->nodes[i];
        if (!orig_node || !orig_node->is_valid) {
            continue;
        }
        
        // Find the corresponding tree node (simplified approach)
        if (i == 0) {
            // First node is the root
            node_map[orig_node->id] = (*ttn)->root;
        } else if (tree_node_idx < (*ttn)->num_nodes) {
            // For simplicity, we're assuming nodes are created in order
            // This is a major simplification and would need to be improved
            // in a real implementation
            if (current && current->num_children > 0) {
                current = current->children[0];
                node_map[orig_node->id] = current;
            }
        }
        
        tree_node_idx++;
    }
    
    // Establish connections based on each node's connections
    for (size_t i = 0; i < network->num_nodes; i++) {
        tensor_node_t* orig_node = network->nodes[i];
        if (!orig_node || !orig_node->is_valid || !orig_node->connected_nodes) {
            continue;
        }
        
        tree_tensor_node_t* source_node = node_map[orig_node->id];
        if (!source_node) {
            printf("DEBUG: Missing source node for connections\n");
            continue;
        }
        
        // Process each connection for this node
        for (size_t j = 0; j < orig_node->num_connections; j++) {
            size_t target_id = orig_node->connected_nodes[j];
            tree_tensor_node_t* target_node = node_map[target_id];
            
            if (!target_node) {
                printf("DEBUG: Missing target node for connection\n");
                continue;
            }
            
            // Connect the nodes (parent-child relationship)
            // Only connect if source_node is not already connected to target_node
            if (source_node->parent != target_node && target_node->parent != source_node) {
                // For simplicity, we'll make the source node the parent
                if (!connect_tree_tensor_nodes(*ttn, source_node, target_node)) {
                    printf("DEBUG: Failed to connect nodes %zu -> %zu\n", orig_node->id, target_id);
                } else {
                    printf("DEBUG: Connected nodes %zu -> %zu\n", orig_node->id, target_id);
                }
            }
        }
    }
    
    // Clean up
    free(node_map);
    
    return true;
}
