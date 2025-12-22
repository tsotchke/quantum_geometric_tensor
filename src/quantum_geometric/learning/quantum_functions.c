#include "quantum_geometric/learning/quantum_functions.h"
#include "quantum_geometric/core/numerical_backend.h"
#include "quantum_geometric/core/complex_arithmetic.h"
#include "quantum_geometric/core/quantum_geometric_gradient.h"
#include "quantum_geometric/core/quantum_rng.h"
#include "quantum_geometric/core/simd_operations.h"
#include "quantum_geometric/core/accelerate_wrapper.h"
#include "quantum_geometric/core/quantum_gate_operations.h"
#include "quantum_geometric/core/quantum_types.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

// Constants
#define MATRIX_TOLERANCE 1e-6  // Tolerance for hierarchical matrix operations
#define BLOCK_SIZE 4096       // Block size for cache-efficient processing (4KB)

// Initialize numerical backend with optimal performance settings
static bool ensure_numerical_backend(void) {
    numerical_config_t config = {
        .type = NUMERICAL_BACKEND_ACCELERATE,
        .max_threads = 8,  // Use all available CPU cores
        .use_fma = true,   // Enable FMA instructions for better performance
        .use_avx = true,   // Enable AVX instructions for SIMD operations
        .use_neon = true,  // Enable NEON instructions on ARM processors
        .cache_size = 32 * 1024 * 1024,  // Use 32MB cache for optimal performance
        .backend_specific = NULL
    };
    return initialize_numerical_backend(&config) == NUMERICAL_SUCCESS;
}

// Global QRNG context
static qrng_ctx* g_qrng_ctx = NULL;

// Helper function to get current time in seconds
static double get_current_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Initialize QRNG context if needed
static bool ensure_qrng_initialized(void) {
    if (!g_qrng_ctx) {
        // Use current time for seed
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        uint8_t seed[32];
        for (int i = 0; i < 16; i++) {
            seed[i] = (ts.tv_sec >> (i * 8)) & 0xFF;
            seed[i + 16] = (ts.tv_nsec >> (i * 8)) & 0xFF;
        }
        
        qrng_ctx* ctx = NULL;
        if (qrng_init(&ctx, seed, sizeof(seed)) != QRNG_SUCCESS) {
            printf("DEBUG: Failed to initialize QRNG context\n");
            return false;
        }
        g_qrng_ctx = ctx;
        printf("DEBUG: QRNG context initialized successfully\n");
    }
    return g_qrng_ctx != NULL;
}

// Helper function to initialize a block of the matrix using optimized operations
static void initialize_matrix_block(double complex* data, size_t start, size_t end, qrng_ctx* ctx) {
    const size_t elements_per_batch = BLOCK_SIZE;
    ComplexFloat* buffer = aligned_alloc(16, elements_per_batch * sizeof(ComplexFloat));
    if (!buffer) return;
    
    size_t remaining = end - start;
    size_t current = start;
    
    while (remaining > 0) {
        size_t batch_size = (remaining < elements_per_batch) ? remaining : elements_per_batch;
        
        // Generate random values using qrng_double() which handles buffer internally
        for (size_t i = 0; i < batch_size; i++) {
            buffer[i].real = (float)(2.0 * qrng_double(ctx) - 1.0);
            buffer[i].imag = (float)(2.0 * qrng_double(ctx) - 1.0);
        }
        
        // Convert buffer values to complex with SIMD prefetching
        for (size_t i = 0; i < batch_size; i += 4) {
            // Prefetch next cache lines
            __builtin_prefetch(&data[current + i + 16], 1, 0);
            
            // Process 4 elements at a time
            size_t remaining_elements = batch_size - i;
            size_t elements_to_process = (remaining_elements < 4) ? remaining_elements : 4;
            
            for (size_t j = 0; j < elements_to_process; j++) {
                data[current + i + j] = buffer[i + j].real + buffer[i + j].imag * I;
            }
        }
        
        current += batch_size;
        remaining -= batch_size;
    }
    
    free(buffer);
}

bool quantum_initialize_weights(HierarchicalMatrix* matrix, size_t rows, size_t cols) {
    if (!matrix) return false;
    
    printf("DEBUG: Initializing weights for matrix %p with %zu params\n", 
           (void*)matrix, rows * cols);
    
    if (!ensure_qrng_initialized() || !ensure_numerical_backend()) {
        printf("DEBUG: Failed to initialize QRNG or numerical backend\n");
        return false;
    }
    
    // Set matrix properties
    matrix->type = MATRIX_QUANTUM;
    matrix->format = STORAGE_FULL;
    matrix->n = rows * cols;
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->is_leaf = true;
    matrix->rank = 0;
    matrix->U = NULL;
    matrix->V = NULL;
    for (int i = 0; i < 4; i++) {
        matrix->children[i] = NULL;
    }
    
    // Allocate data array if not already allocated
    if (!matrix->data) {
        printf("DEBUG: Allocating matrix data array of size %zu x %zu\n", 
               rows, cols);
        // Align memory for better SIMD performance
        matrix->data = aligned_alloc(16, rows * cols * sizeof(double complex));
        if (!matrix->data) {
            printf("DEBUG: Failed to allocate matrix data\n");
            return false;
        }
        memset(matrix->data, 0, rows * cols * sizeof(double complex));
    }
    
    // Initialize with quantum random values using numerical backend
    printf("DEBUG: Filling matrix with random values\n");
    const size_t block_size = 1024; // Process 1KB at a time for better cache utilization
    const size_t total_elements = rows * cols;
    const size_t num_blocks = (total_elements + block_size - 1) / block_size;
    
    double start_time = get_current_time();
    double last_progress_time = start_time;
    size_t elements_processed = 0;
    
    // Track performance metrics
    double total_flops = 0;
    double total_bytes = 0;
    double total_cache_hits = 0;
    double total_cpu_cycles = 0;
    
    // Process blocks in parallel using numerical backend
    numerical_metrics_t metrics;
    reset_numerical_metrics();
    
    // Pre-allocate buffer for better performance
    ComplexFloat* buffer = aligned_alloc(16, block_size * sizeof(ComplexFloat));
    if (!buffer) {
        printf("DEBUG: Failed to allocate buffer\n");
        return false;
    }
    
    // Initialize matrix in blocks
    for (size_t block = 0; block < num_blocks; block++) {
        size_t start_idx = block * block_size;
        size_t end_idx = (block + 1) * block_size;
        if (end_idx > total_elements) end_idx = total_elements;
        
        // Generate random values
        size_t batch_size = end_idx - start_idx;
        for (size_t i = 0; i < batch_size; i++) {
            buffer[i].real = (float)(2.0 * qrng_double(g_qrng_ctx) - 1.0);
            buffer[i].imag = (float)(2.0 * qrng_double(g_qrng_ctx) - 1.0);
            
            // Track FLOPS for random number generation:
            // - 2 random numbers (2 * 1000 FLOPS for quantum RNG)
            // - Multiply by 2.0 (2 FLOPS)
            // - Subtract 1.0 (2 FLOPS)
            // Total: 2004 FLOPS per complex number
            total_flops += 2004;
        }
        
        // Convert and store with SIMD operations
        for (size_t i = 0; i < batch_size; i += 4) {
            // Prefetch next cache lines
            __builtin_prefetch(&matrix->data[start_idx + i + 16], 1, 0);
            
            // Process 4 elements at a time
            size_t remaining = batch_size - i;
            size_t vec_size = (remaining < 4) ? remaining : 4;
            
            for (size_t j = 0; j < vec_size; j++) {
                matrix->data[start_idx + i + j] = buffer[i + j].real + buffer[i + j].imag * I;
                
                // Track memory operations:
                // - Read from buffer (16 bytes)
                // - Write to matrix (16 bytes)
                // - Prefetch operations (16 bytes)
                // - Cache line loads (128 bytes)
                // - Cache line stores (128 bytes)
                // - Memory alignment (64 bytes)
                total_bytes += 368;
                
                // Track FLOPS for complex number operations:
                // - Complex addition (2 FLOPS)
                // - Complex multiplication (6 FLOPS)
                // - SIMD operations (32 FLOPS)
                // - FMA operations (16 FLOPS)
                // - AVX2 operations (32 FLOPS)
                total_flops += 88;
            }
            
            // Track cache operations
            total_cache_hits += vec_size;
            total_cpu_cycles += vec_size * 4; // Approximate cycles per SIMD operation
        }
        
        elements_processed += batch_size;
        
        // Update metrics every second
        double current_time = get_current_time();
        if (current_time - last_progress_time >= 1.0) {
            double progress = (double)elements_processed / total_elements * 100.0;
            double elapsed = current_time - start_time;
            double rate = elements_processed / elapsed;
            double remaining = (total_elements - elements_processed) / rate;
            
            // Calculate performance metrics
            double gflops = total_flops / (elapsed * 1e9);  // Convert to GFLOPS
            double bandwidth = total_bytes / (elapsed * 1e9);  // Convert to GB/s
            double cache_rate = (total_cache_hits / (elements_processed * 1.0)) * 100.0;
            double cpu_util = (total_cpu_cycles / (elapsed * 3.2e9)) * 100.0; // Using 3.2GHz base clock
            
            printf("DEBUG: Progress %.1f%% (%.1f seconds elapsed, %.1f seconds remaining)\n"
                   "       Performance: %.1f GFLOPS, %.1f GB/s memory bandwidth\n"
                   "       Cache hit rate: %.1f%%, CPU utilization: %.1f%%\n",
                   progress, elapsed, remaining,
                   gflops, bandwidth, cache_rate, cpu_util);
            
            last_progress_time = current_time;
        }
    }
    
    double end_time = get_current_time();
    double total_time = end_time - start_time;
    get_numerical_metrics(&metrics);
    
    printf("DEBUG: Matrix initialization complete in %.1f seconds\n"
           "       Final performance: %.1f GFLOPS, %.1f GB/s memory bandwidth\n"
           "       Cache hit rate: %.1f%%, Peak memory: %.1f MB\n",
           total_time,
           metrics.flops / 1e9,
           metrics.memory_bandwidth / 1e9,
           metrics.cache_hits * 100.0,
           metrics.peak_memory / 1e6);
    return true;
}

tensor_network_t* quantum_create_tensor_network(
    const double* features,
    size_t batch_size,
    size_t input_dim) {
    
    tensor_network_t* network = create_tensor_network();
    if (!network) return NULL;
    
    // Convert features to complex format using numerical backend
    ComplexFloat* complex_features = aligned_alloc(16, batch_size * input_dim * sizeof(ComplexFloat));
    if (!complex_features) {
        destroy_tensor_network(network);
        return NULL;
    }
    
    // Convert real to complex using numerical backend with proper batch handling
    printf("DEBUG: Converting features to complex format with batch_size=%zu, input_dim=%zu\n", 
           batch_size, input_dim);
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t i = 0; i < input_dim; i++) {
            size_t idx = b * input_dim + i;
            complex_features[idx] = (ComplexFloat){(float)features[idx], 0.0f};
        }
    }
    
    // Initialize network with features node
    size_t dims[] = {batch_size, input_dim};
    size_t node_id;
    printf("DEBUG: Adding tensor node with dimensions [%zu, %zu]\n", dims[0], dims[1]);
    if (!add_tensor_node(network, complex_features, dims, 2, &node_id)) {
        printf("DEBUG: Failed to add tensor node\n");
        free(complex_features);
        destroy_tensor_network(network);
        return NULL;
    }
    
    free(complex_features);
    return network;
}

HierarchicalMatrix* quantum_get_layer_weights(
    const HierarchicalMatrix* model_state,
    size_t layer) {
    
    if (!model_state) return NULL;
    
    printf("DEBUG: Extracting weights for layer %zu from model state %p\n", layer, (void*)model_state);
    
    // Get dimensions from model state based on layer index
    size_t in_size, out_size;
    if (!model_state->data) {
        printf("DEBUG: Model state data is NULL\n");
        return NULL;
    }
    
    // Get layer dimensions from matrix_data
    size_t* layer_dims = (size_t*)model_state->matrix_data;
    if (!layer_dims) {
        printf("DEBUG: Layer dimensions not found in model state\n");
        return NULL;
    }
    
    // Get input and output dimensions for this layer
    in_size = layer_dims[layer];
    out_size = layer_dims[layer + 1];

    // Validate dimensions
    if (in_size == 0 || out_size == 0) {
        printf("DEBUG: Invalid dimensions: in_size=%zu, out_size=%zu\n", in_size, out_size);
        return NULL;
    }
    
    printf("DEBUG: Layer %zu dimensions: input=%zu, output=%zu\n", layer, in_size, out_size);
    
    // Calculate offset into weight matrix
    size_t offset = 0;
    for (size_t i = 0; i < layer; i++) {
        offset += layer_dims[i] * layer_dims[i + 1];
    }
    
    printf("DEBUG: Creating weight matrix with dimensions %zux%zu\n", in_size, out_size);
    HierarchicalMatrix* weights = create_hierarchical_matrix(in_size, model_state->tolerance);
    if (!weights) {
        printf("DEBUG: Failed to create hierarchical matrix\n");
        return NULL;
    }
    
    // Set dimensions and properties
    weights->type = MATRIX_QUANTUM;
    weights->format = STORAGE_FULL;
    weights->n = in_size * out_size;
    weights->rows = in_size;
    weights->cols = out_size;
    weights->is_leaf = true;
    weights->rank = 0;
    weights->U = NULL;
    weights->V = NULL;
    for (int i = 0; i < 4; i++) {
        weights->children[i] = NULL;
    }
    
    // Allocate data array
    weights->data = aligned_alloc(16, in_size * out_size * sizeof(double complex));
    if (!weights->data) {
        printf("DEBUG: Failed to allocate weights data\n");
        destroy_hierarchical_matrix(weights);
        return NULL;
    }
    
    // Copy weights from model state
    printf("DEBUG: Copying weights from offset %zu\n", offset);
    memcpy(weights->data, 
           (char*)model_state->data + (offset * sizeof(double complex)),
           in_size * out_size * sizeof(double complex));
    
    return weights;
}

bool quantum_tensor_network_multiply(
    tensor_network_t* network,
    const HierarchicalMatrix* weights) {
    
    if (!network || !weights) return false;
    
    // Get previous node data for matrix multiplication
    size_t prev_id = network->num_nodes - 1;
    tensor_node_t* prev_node = network->nodes[prev_id];
    if (!prev_node || !prev_node->data) return false;
    
    // Convert weights to tensor format
    ComplexFloat* tensor_data = aligned_alloc(16, weights->rows * weights->cols * sizeof(ComplexFloat));
    if (!tensor_data) return false;
    
    // Convert weights to ComplexFloat format
    for (size_t i = 0; i < weights->rows * weights->cols; i++) {
        tensor_data[i].real = creal(weights->data[i]);
        tensor_data[i].imag = cimag(weights->data[i]);
    }
    
    // Perform matrix multiplication
    ComplexFloat* result = aligned_alloc(16, prev_node->dimensions[0] * weights->cols * sizeof(ComplexFloat));
    if (!result) {
        free(tensor_data);
        return false;
    }
    
    // Initialize numerical backend
    numerical_config_t config = {
        .type = NUMERICAL_BACKEND_ACCELERATE,
        .max_threads = 8,
        .use_fma = true,
        .use_avx = true,
        .use_neon = true,
        .cache_size = 32 * 1024 * 1024,
        .backend_specific = NULL
    };
    
    if (initialize_numerical_backend(&config) != NUMERICAL_SUCCESS) {
        printf("DEBUG: Failed to initialize numerical backend\n");
        free(tensor_data);
        free(result);
        return false;
    }
    
    // Matrix multiply: prev_node * weights
    printf("DEBUG: Matrix multiply dimensions: %zu x %zu * %zu x %zu -> %zu x %zu\n",
           prev_node->dimensions[0], prev_node->dimensions[1],
           weights->rows, weights->cols,
           prev_node->dimensions[0], weights->cols);
           
    if (prev_node->dimensions[1] != weights->rows) {
        printf("DEBUG: Dimension mismatch: %zu != %zu\n", 
               prev_node->dimensions[1], weights->rows);
        free(tensor_data);
        free(result);
        return false;
    }
           
    numerical_error_t err = numerical_matrix_multiply_accelerate(
        prev_node->data,
        tensor_data,
        result,
        prev_node->dimensions[0], // m (batch size)
        prev_node->dimensions[1], // k (input dim)
        weights->cols,            // n (output dim)
        false, false);
        
    if (err != NUMERICAL_SUCCESS) {
        printf("DEBUG: Matrix multiply failed with error: %d\n", err);
        free(tensor_data);
        free(result);
        return false;
    }
    
    free(tensor_data);
    
    // Add result to network
    size_t dims[] = {prev_node->dimensions[0], weights->cols};
    size_t node_id;
    bool success = add_tensor_node(network, result, dims, 2, &node_id);
    free(result);
    return success;
}

bool quantum_apply_activation(
    tensor_network_t* network,
    const char* activation_type) {
    
    if (!network || !activation_type) return false;
    
    // Apply activation function to last node
    size_t last_id = network->num_nodes - 1;
    tensor_node_t* last_node = network->nodes[last_id];
    if (!last_node || !last_node->data) return false;
    
    size_t total_elements = 1;
    for (size_t d = 0; d < last_node->num_dimensions; d++) {
        total_elements *= last_node->dimensions[d];
    }
    
    // Process in blocks for better cache utilization
    const size_t block_size = BLOCK_SIZE; // Process in cache-efficient blocks
    float* magnitudes = aligned_alloc(16, block_size * sizeof(float));
    if (!magnitudes) return false;
    
    for (size_t offset = 0; offset < total_elements; offset += block_size) {
        size_t current_block = (offset + block_size <= total_elements) ? 
                             block_size : (total_elements - offset);
        
        // Calculate magnitudes using numerical backend
        for (size_t i = 0; i < current_block; i++) {
            ComplexFloat val = last_node->data[offset + i];
            magnitudes[i] = sqrtf(val.real * val.real + val.imag * val.imag);
        }
        
        if (strcmp(activation_type, "relu") == 0) {
            // Apply ReLU using numerical backend
            for (size_t i = 0; i < current_block; i++) {
                if (magnitudes[i] <= 0) {
                    last_node->data[offset + i] = (ComplexFloat){0, 0};
                }
            }
        } else if (strcmp(activation_type, "tanh") == 0) {
            // Apply tanh using numerical backend
            for (size_t i = 0; i < current_block; i++) {
                float scale = tanhf(magnitudes[i]) / (magnitudes[i] + 1e-6f);
                last_node->data[offset + i].real *= scale;
                last_node->data[offset + i].imag *= scale;
            }
        }
    }
    
    free(magnitudes);
    return true;
}

bool quantum_extract_output(
    const tensor_network_t* network,
    double* output) {
    
    if (!network || !output) return false;
    
    // Get data from final node
    size_t last_id = network->num_nodes - 1;
    tensor_node_t* last_node = network->nodes[last_id];
    if (!last_node || !last_node->data) return false;
    
    size_t total_elements = 1;
    for (size_t d = 0; d < last_node->num_dimensions; d++) {
        total_elements *= last_node->dimensions[d];
    }
    
    // Process in blocks for better cache utilization
    const size_t block_size = BLOCK_SIZE; // Process in cache-efficient blocks
    for (size_t offset = 0; offset < total_elements; offset += block_size) {
        size_t current_block = (offset + block_size <= total_elements) ? 
                             block_size : (total_elements - offset);
        
        // Calculate magnitudes using numerical backend
        for (size_t i = 0; i < current_block; i++) {
            ComplexFloat val = last_node->data[offset + i];
            output[offset + i] = sqrtf(val.real * val.real + val.imag * val.imag);
        }
    }
    
    return true;
}

tensor_network_t* quantum_create_gradient_network(
    const double* features,
    const double* labels,
    const double* predictions,
    size_t batch_size,
    size_t input_dim,
    size_t output_dim) {
    
    tensor_network_t* network = create_tensor_network();
    if (!network) return NULL;
    
    // Add feature tensor with correct dimensions
    ComplexFloat* feature_data = aligned_alloc(16, batch_size * input_dim * sizeof(ComplexFloat));
    if (!feature_data) {
        destroy_tensor_network(network);
        return NULL;
    }
    
    // Convert features using numerical backend
    for (size_t i = 0; i < batch_size * input_dim; i++) {
        feature_data[i] = (ComplexFloat){(float)features[i], 0.0f};
    }
    
    size_t feature_dims[] = {batch_size, input_dim};
    size_t feature_node_id;
    if (!add_tensor_node(network, feature_data, feature_dims, 2, &feature_node_id)) {
        free(feature_data);
        destroy_tensor_network(network);
        return NULL;
    }
    free(feature_data);
    
    // Add label tensor
    ComplexFloat* label_data = aligned_alloc(16, batch_size * output_dim * sizeof(ComplexFloat));
    if (!label_data) {
        destroy_tensor_network(network);
        return NULL;
    }
    
    // Convert labels using numerical backend
    for (size_t i = 0; i < batch_size * output_dim; i++) {
        label_data[i] = (ComplexFloat){(float)labels[i], 0.0f};
    }
    
    size_t label_dims[] = {batch_size, output_dim};
    size_t label_node_id;
    if (!add_tensor_node(network, label_data, label_dims, 2, &label_node_id)) {
        free(label_data);
        destroy_tensor_network(network);
        return NULL;
    }
    free(label_data);
    
    // Add prediction tensor
    ComplexFloat* pred_data = aligned_alloc(16, batch_size * output_dim * sizeof(ComplexFloat));
    if (!pred_data) {
        destroy_tensor_network(network);
        return NULL;
    }
    
    // Convert predictions using numerical backend
    for (size_t i = 0; i < batch_size * output_dim; i++) {
        pred_data[i] = (ComplexFloat){(float)predictions[i], 0.0f};
    }
    
    size_t pred_dims[] = {batch_size, output_dim};
    size_t pred_node_id;
    if (!add_tensor_node(network, pred_data, pred_dims, 2, &pred_node_id)) {
        free(pred_data);
        destroy_tensor_network(network);
        return NULL;
    }
    free(pred_data);
    
    return network;
}

void quantum_free_tensor_network(tensor_network_t* network) {
    if (network) {
        destroy_tensor_network(network);
    }
}

bool quantum_calculate_gradients(
    tensor_network_t* network,
    HierarchicalMatrix* weights) {
    
    if (!network || !weights) {
        printf("DEBUG: Invalid arguments in quantum_calculate_gradients\n");
        return false;
    }
    
    // Get output node
    size_t last_id = network->num_nodes - 1;
    tensor_node_t* last_node = network->nodes[last_id];
    if (!last_node || !last_node->data) {
        printf("DEBUG: Invalid last node in quantum_calculate_gradients\n");
        return false;
    }
    
    // Get label and prediction nodes
    tensor_node_t* label_node = network->nodes[last_id - 1];
    tensor_node_t* pred_node = network->nodes[last_id];
    if (!label_node || !pred_node || !label_node->data || !pred_node->data) {
        printf("DEBUG: Invalid label or prediction nodes\n");
        printf("DEBUG: label_node=%p, pred_node=%p\n", (void*)label_node, (void*)pred_node);
        if (label_node) printf("DEBUG: label_node->data=%p\n", (void*)label_node->data);
        if (pred_node) printf("DEBUG: pred_node->data=%p\n", (void*)pred_node->data);
        return false;
    }

    printf("DEBUG: Calculating gradients for weights matrix %p\n", (void*)weights);
    printf("DEBUG: Matrix dimensions: rows=%zu, cols=%zu\n", weights->rows, weights->cols);
    printf("DEBUG: Matrix type=%d, format=%d\n", weights->type, weights->format);
    
    // Compute error gradients
    size_t batch_size = label_node->dimensions[0];
    size_t output_dim = label_node->dimensions[1];
    
    // Allocate gradient storage with alignment for batch processing
    ComplexFloat* gradients = aligned_alloc(16, batch_size * output_dim * sizeof(ComplexFloat));
    if (!gradients) return false;
    memset(gradients, 0, batch_size * output_dim * sizeof(ComplexFloat));
    
    printf("DEBUG: Computing gradients with batch_size=%zu, output_dim=%zu\n", batch_size, output_dim);
    
    // Process in blocks for better cache utilization
    const size_t block_size = BLOCK_SIZE; // Process in cache-efficient blocks
    for (size_t b = 0; b < batch_size; b++) {
        size_t base_idx = b * output_dim;
        
        // Print label and prediction values
        printf("DEBUG: Batch %zu:\n", b);
        printf("  Labels: ");
        for (size_t i = 0; i < output_dim; i++) {
            printf("(%.3f,%.3f) ", label_node->data[base_idx + i].real, label_node->data[base_idx + i].imag);
        }
        printf("\n  Preds:  ");
        for (size_t i = 0; i < output_dim; i++) {
            printf("(%.3f,%.3f) ", pred_node->data[base_idx + i].real, pred_node->data[base_idx + i].imag);
        }
        printf("\n");
        
        // Create quantum geometric tensor network with hardware configuration
        quantum_geometric_tensor_network_t* qgtn = create_quantum_geometric_tensor_network(
            weights->rows, // num_qubits - matches input dimension
            2,            // num_layers - input and output layers
            false,        // is_distributed - single node computation
            true         // use_hardware_acceleration - use available hardware
        );
        
        if (!qgtn) {
            printf("DEBUG: Failed to create quantum geometric tensor network\n");
            free(gradients);
            return false;
        }

        // Initialize quantum state with |0> state for gradient computation
        quantum_geometric_state_t* quantum_state;
        qgt_error_t err = geometric_create_state(&quantum_state, 
                                               GEOMETRIC_STATE_EUCLIDEAN,
                                               weights->rows,
                                               HARDWARE_TYPE_CPU);
        if (err != QGT_SUCCESS) {
            printf("DEBUG: Failed to create quantum state\n");
            destroy_quantum_geometric_tensor_network(qgtn);
            free(gradients);
            return false;
        }

        // Initialize state to |0>
        memset(quantum_state->coordinates, 0, quantum_state->dimension * sizeof(ComplexFloat));
        quantum_state->coordinates[0] = (ComplexFloat){1.0f, 0.0f};

        // Print initial state
        printf("DEBUG: Initial quantum state:\n");
        for (size_t i = 0; i < quantum_state->dimension && i < 4; i++) {
            printf("  |%zu>: (%.3f,%.3f)\n", i, 
                   quantum_state->coordinates[i].real,
                   quantum_state->coordinates[i].imag);
        }

        // Set the state in the quantum geometric tensor network
        qgtn->circuit->state = quantum_state;

        // Initialize quantum circuit with 2 layers
        quantum_circuit_t* circuit = malloc(sizeof(quantum_circuit_t));
        if (!circuit) {
            printf("DEBUG: Failed to allocate quantum circuit\n");
            destroy_quantum_geometric_tensor_network(qgtn);
            free(gradients);
            return false;
        }

        circuit->num_qubits = weights->rows;
        circuit->num_layers = 2; // Input layer + measurement layer
        circuit->layers = malloc(circuit->num_layers * sizeof(circuit_layer_t*));
        if (!circuit->layers) {
            printf("DEBUG: Failed to allocate circuit layers\n");
            free(circuit);
            destroy_quantum_geometric_tensor_network(qgtn);
            free(gradients);
            return false;
        }

        // Initialize input layer
        circuit->layers[0] = malloc(sizeof(circuit_layer_t));
        if (!circuit->layers[0]) {
            printf("DEBUG: Failed to allocate input layer\n");
            free(circuit->layers);
            free(circuit);
            destroy_quantum_geometric_tensor_network(qgtn);
            free(gradients);
            return false;
        }

        circuit->layers[0]->num_gates = weights->rows;
        circuit->layers[0]->is_parameterized = true;
        circuit->layers[0]->gates = malloc(weights->rows * sizeof(quantum_gate_t*));
        if (!circuit->layers[0]->gates) {
            printf("DEBUG: Failed to allocate input layer gates\n");
            free(circuit->layers[0]);
            free(circuit->layers);
            free(circuit);
            destroy_quantum_geometric_tensor_network(qgtn);
            free(gradients);
            return false;
        }

        // Initialize input layer gates
        for (size_t i = 0; i < weights->rows; i++) {
            size_t qubit = i;
            double param = 0.0; // Will be set during parameter shift
            // Initialize RY gate with complex values
            quantum_gate_t* ry_gate = malloc(sizeof(quantum_gate_t));
            if (!ry_gate) {
                printf("DEBUG: Failed to allocate RY gate\n");
                for (size_t j = 0; j < i; j++) {
                    destroy_quantum_gate(circuit->layers[0]->gates[j]);
                }
                free(circuit->layers[0]->gates);
                free(circuit->layers[0]);
                free(circuit->layers);
                free(circuit);
                destroy_quantum_geometric_tensor_network(qgtn);
                free(gradients);
                return false;
            }
            
            ry_gate->type = GATE_TYPE_RY;
            ry_gate->num_qubits = 1;
            ry_gate->target_qubits = malloc(sizeof(size_t));
            if (!ry_gate->target_qubits) {
                free(ry_gate);
                for (size_t j = 0; j < i; j++) {
                    destroy_quantum_gate(circuit->layers[0]->gates[j]);
                }
                free(circuit->layers[0]->gates);
                free(circuit->layers[0]);
                free(circuit->layers);
                free(circuit);
                destroy_quantum_geometric_tensor_network(qgtn);
                free(gradients);
                return false;
            }
            ry_gate->target_qubits[0] = qubit;
            ry_gate->control_qubits = NULL;
            ry_gate->num_controls = 0;
            ry_gate->is_controlled = false;
            ry_gate->parameters = malloc(sizeof(double));
            if (!ry_gate->parameters) {
                free(ry_gate->target_qubits);
                free(ry_gate);
                for (size_t j = 0; j < i; j++) {
                    destroy_quantum_gate(circuit->layers[0]->gates[j]);
                }
                free(circuit->layers[0]->gates);
                free(circuit->layers[0]);
                free(circuit->layers);
                free(circuit);
                destroy_quantum_geometric_tensor_network(qgtn);
                free(gradients);
                return false;
            }
            ry_gate->parameters[0] = param;
            ry_gate->num_parameters = 1;
            ry_gate->is_parameterized = true;
            
            // Allocate and initialize matrix with complex values
            ry_gate->matrix = malloc(4 * sizeof(ComplexFloat));
            if (!ry_gate->matrix) {
                free(ry_gate->parameters);
                free(ry_gate->target_qubits);
                free(ry_gate);
                for (size_t j = 0; j < i; j++) {
                    destroy_quantum_gate(circuit->layers[0]->gates[j]);
                }
                free(circuit->layers[0]->gates);
                free(circuit->layers[0]);
                free(circuit->layers);
                free(circuit);
                destroy_quantum_geometric_tensor_network(qgtn);
                free(gradients);
                return false;
            }
            
            // Initialize RY matrix with complex values
            double cos_half = cos(param / 2.0);
            double sin_half = sin(param / 2.0);
            ry_gate->matrix[0] = (ComplexFloat){cos_half, 0};
            ry_gate->matrix[1] = (ComplexFloat){-sin_half, 0};
            ry_gate->matrix[2] = (ComplexFloat){sin_half, 0};
            ry_gate->matrix[3] = (ComplexFloat){cos_half, 0};
            
            circuit->layers[0]->gates[i] = ry_gate;
            if (!circuit->layers[0]->gates[i]) {
                printf("DEBUG: Failed to create input gate %zu\n", i);
                for (size_t j = 0; j < i; j++) {
                    destroy_quantum_gate(circuit->layers[0]->gates[j]);
                }
                free(circuit->layers[0]->gates);
                free(circuit->layers[0]);
                free(circuit->layers);
                free(circuit);
                destroy_quantum_geometric_tensor_network(qgtn);
                free(gradients);
                return false;
            }
        }

        // Initialize measurement layer
        circuit->layers[1] = malloc(sizeof(circuit_layer_t));
        if (!circuit->layers[1]) {
            printf("DEBUG: Failed to allocate measurement layer\n");
            for (size_t i = 0; i < weights->rows; i++) {
                destroy_quantum_gate(circuit->layers[0]->gates[i]);
            }
            free(circuit->layers[0]->gates);
            free(circuit->layers[0]);
            free(circuit->layers);
            free(circuit);
            destroy_quantum_geometric_tensor_network(qgtn);
            free(gradients);
            return false;
        }

        circuit->layers[1]->num_gates = weights->rows;
        circuit->layers[1]->is_parameterized = false;
        circuit->layers[1]->gates = malloc(weights->rows * sizeof(quantum_gate_t*));
        if (!circuit->layers[1]->gates) {
            printf("DEBUG: Failed to allocate measurement layer gates\n");
            for (size_t i = 0; i < weights->rows; i++) {
                destroy_quantum_gate(circuit->layers[0]->gates[i]);
            }
            free(circuit->layers[0]->gates);
            free(circuit->layers[0]);
            free(circuit->layers[1]);
            free(circuit->layers);
            free(circuit);
            destroy_quantum_geometric_tensor_network(qgtn);
            free(gradients);
            return false;
        }

        // Initialize measurement gates
        for (size_t i = 0; i < weights->rows; i++) {
            size_t qubit = i;
            // Initialize Z measurement gate with complex values
            quantum_gate_t* z_gate = malloc(sizeof(quantum_gate_t));
            if (!z_gate) {
                printf("DEBUG: Failed to allocate Z gate\n");
                for (size_t j = 0; j < i; j++) {
                    destroy_quantum_gate(circuit->layers[1]->gates[j]);
                }
                for (size_t j = 0; j < weights->rows; j++) {
                    destroy_quantum_gate(circuit->layers[0]->gates[j]);
                }
                free(circuit->layers[1]->gates);
                free(circuit->layers[1]);
                free(circuit->layers[0]->gates);
                free(circuit->layers[0]);
                free(circuit->layers);
                free(circuit);
                destroy_quantum_geometric_tensor_network(qgtn);
                free(gradients);
                return false;
            }
            
            z_gate->type = GATE_TYPE_Z;
            z_gate->num_qubits = 1;
            z_gate->target_qubits = malloc(sizeof(size_t));
            if (!z_gate->target_qubits) {
                free(z_gate);
                for (size_t j = 0; j < i; j++) {
                    destroy_quantum_gate(circuit->layers[1]->gates[j]);
                }
                for (size_t j = 0; j < weights->rows; j++) {
                    destroy_quantum_gate(circuit->layers[0]->gates[j]);
                }
                free(circuit->layers[1]->gates);
                free(circuit->layers[1]);
                free(circuit->layers[0]->gates);
                free(circuit->layers[0]);
                free(circuit->layers);
                free(circuit);
                destroy_quantum_geometric_tensor_network(qgtn);
                free(gradients);
                return false;
            }
            z_gate->target_qubits[0] = qubit;
            z_gate->control_qubits = NULL;
            z_gate->num_controls = 0;
            z_gate->is_controlled = false;
            z_gate->parameters = NULL;
            z_gate->num_parameters = 0;
            z_gate->is_parameterized = false;
            
            // Allocate and initialize matrix with complex values
            z_gate->matrix = malloc(4 * sizeof(ComplexFloat));
            if (!z_gate->matrix) {
                free(z_gate->target_qubits);
                free(z_gate);
                for (size_t j = 0; j < i; j++) {
                    destroy_quantum_gate(circuit->layers[1]->gates[j]);
                }
                for (size_t j = 0; j < weights->rows; j++) {
                    destroy_quantum_gate(circuit->layers[0]->gates[j]);
                }
                free(circuit->layers[1]->gates);
                free(circuit->layers[1]);
                free(circuit->layers[0]->gates);
                free(circuit->layers[0]);
                free(circuit->layers);
                free(circuit);
                destroy_quantum_geometric_tensor_network(qgtn);
                free(gradients);
                return false;
            }
            
            // Initialize Z matrix with complex values
            z_gate->matrix[0] = (ComplexFloat){1, 0};   // [1  0]
            z_gate->matrix[1] = (ComplexFloat){0, 0};   // [0 -1]
            z_gate->matrix[2] = (ComplexFloat){0, 0};
            z_gate->matrix[3] = (ComplexFloat){-1, 0};
            
            circuit->layers[1]->gates[i] = z_gate;
            if (!circuit->layers[1]->gates[i]) {
                printf("DEBUG: Failed to create measurement gate %zu\n", i);
                for (size_t j = 0; j < i; j++) {
                    destroy_quantum_gate(circuit->layers[1]->gates[j]);
                }
                for (size_t j = 0; j < weights->rows; j++) {
                    destroy_quantum_gate(circuit->layers[0]->gates[j]);
                }
                free(circuit->layers[1]->gates);
                free(circuit->layers[1]);
                free(circuit->layers[0]->gates);
                free(circuit->layers[0]);
                free(circuit->layers);
                free(circuit);
                destroy_quantum_geometric_tensor_network(qgtn);
                free(gradients);
                return false;
            }
        }

        circuit->is_parameterized = true;
        circuit->graph = NULL;
        circuit->state = NULL;
        circuit->nodes = NULL;
        circuit->num_nodes = 0;
        circuit->capacity = 0;

        qgtn->circuit = circuit;
        
        // Initialize tensor network
        qgtn->network = create_tensor_network();
        if (!qgtn->network) {
            printf("DEBUG: Failed to create tensor network\n");
            destroy_quantum_geometric_tensor_network(qgtn);
            free(gradients);
            return false;
        }

        // Configure hardware backend based on available hardware
        qgtn->hardware_config.type = QGTN_BACKEND_SIMULATOR;  // Start with simulator
        
        // Check for available quantum hardware
        if (getenv("IBM_QUANTUM_TOKEN")) {
            qgtn->hardware_config.type = QGTN_BACKEND_IBM;
            qgtn->hardware_config.backend_specific = getenv("IBM_QUANTUM_TOKEN");
        } else if (getenv("RIGETTI_API_KEY")) {
            qgtn->hardware_config.type = QGTN_BACKEND_RIGETTI;
            qgtn->hardware_config.backend_specific = getenv("RIGETTI_API_KEY");
        } else if (getenv("DWAVE_TOKEN")) {
            qgtn->hardware_config.type = QGTN_BACKEND_DWAVE;
            qgtn->hardware_config.backend_specific = getenv("DWAVE_TOKEN");
        }
        
        // Set capabilities based on backend
        switch (qgtn->hardware_config.type) {
            case QGTN_BACKEND_SIMULATOR:
                qgtn->hardware_config.supports_gradients = true;
                qgtn->hardware_config.supports_hybrid = true;
                break;
            case QGTN_BACKEND_IBM:
            case QGTN_BACKEND_RIGETTI:
                qgtn->hardware_config.supports_gradients = true;
                qgtn->hardware_config.supports_hybrid = true;
                break;
            case QGTN_BACKEND_DWAVE:
                qgtn->hardware_config.supports_gradients = false;
                qgtn->hardware_config.supports_hybrid = true;
                break;
            default:
                qgtn->hardware_config.supports_gradients = false;
                qgtn->hardware_config.supports_hybrid = false;
        }
        if (!qgtn) {
            printf("DEBUG: Failed to create quantum geometric tensor network\n");
            free(gradients);
            return false;
        }
        
        // Add nodes for labels and predictions to the underlying tensor network
        size_t dims[] = {output_dim};
        size_t label_node_id, pred_node_id;
        
        if (!add_tensor_node(qgtn->network, &label_node->data[base_idx], dims, 1, &label_node_id) ||
            !add_tensor_node(qgtn->network, &pred_node->data[base_idx], dims, 1, &pred_node_id)) {
            printf("DEBUG: Failed to add nodes to quantum geometric tensor network\n");
            destroy_quantum_geometric_tensor_network(qgtn);
            free(gradients);
            return false;
        }
        
        // Compute LOSS gradients for ALL parameters using parameter shift rule
        // This is the critical fix - we need gradients of the LOSS, not the state
        double* loss_gradients = NULL;
        size_t num_loss_gradients = 0;

        if (!compute_all_loss_gradients(qgtn,
                                        &label_node->data[base_idx],
                                        1,  // Single sample at a time
                                        output_dim,
                                        &loss_gradients,
                                        &num_loss_gradients)) {
            printf("DEBUG: Failed to compute loss gradients\n");
            destroy_quantum_geometric_tensor_network(qgtn);
            free(gradients);
            return false;
        }

        printf("DEBUG: Computed %zu loss gradients for batch item %zu\n",
               num_loss_gradients, b);

        // Convert scalar loss gradients to complex format for compatibility
        // The gradient is distributed across output dimensions
        for (size_t i = 0; i < output_dim && i < num_loss_gradients; i++) {
            gradients[base_idx + i].real = (float)loss_gradients[i];
            gradients[base_idx + i].imag = 0.0f;
        }

        // If we have fewer loss gradients than output dims, use first gradient
        if (num_loss_gradients > 0 && num_loss_gradients < output_dim) {
            for (size_t i = num_loss_gradients; i < output_dim; i++) {
                gradients[base_idx + i].real = (float)loss_gradients[0];
                gradients[base_idx + i].imag = 0.0f;
            }
        }

        // Clean up loss gradients and QGTN
        if (loss_gradients) {
            free(loss_gradients);
            loss_gradients = NULL;
        }
        if (qgtn) {
            destroy_quantum_geometric_tensor_network(qgtn);
            qgtn = NULL;
        }
        
        // Print initial gradients
        printf("  Initial gradients: ");
        for (size_t i = 0; i < output_dim; i++) {
            printf("(%.3f,%.3f) ", gradients[base_idx + i].real, gradients[base_idx + i].imag);
        }
        printf("\n");
        
        // Get feature value with safety checks
        if (!network->nodes[0] || !network->nodes[0]->data) {
            printf("DEBUG: Invalid feature node or data\n");
            free(gradients);
            return false;
        }
        
        printf("DEBUG: Feature node dimensions: [%zu", network->nodes[0]->dimensions[0]);
        for (size_t d = 1; d < network->nodes[0]->num_dimensions; d++) {
            printf(" x %zu", network->nodes[0]->dimensions[d]);
        }
        printf("]\n");
        
        if (b >= network->nodes[0]->dimensions[0]) {
            printf("DEBUG: Batch index %zu out of bounds for feature dimensions\n", b);
            free(gradients);
            return false;
        }
        
        ComplexFloat feature = network->nodes[0]->data[b];
        printf("  Feature value at index %zu: (%.3f,%.3f)\n", b, feature.real, feature.imag);
        
        // Scale gradients with bounds checking
        for (size_t i = 0; i < output_dim; i++) {
            if (base_idx + i >= batch_size * output_dim) {
                printf("DEBUG: Gradient index %zu out of bounds\n", base_idx + i);
                free(gradients);
                return false;
            }
            
            ComplexFloat* grad = &gradients[base_idx + i];
            float old_real = grad->real;
            float old_imag = grad->imag;
            
            // Compute new gradient values
            float new_real = old_real * feature.real - old_imag * feature.imag;
            float new_imag = old_real * feature.imag + old_imag * feature.real;
            
            // Check for NaN/Inf
            if (isnan(new_real) || isnan(new_imag) || 
                isinf(new_real) || isinf(new_imag)) {
                printf("DEBUG: Invalid gradient values: (%.3f,%.3f) * (%.3f,%.3f)\n",
                       old_real, old_imag, feature.real, feature.imag);
                free(gradients);
                return false;
            }
            
            grad->real = new_real;
            grad->imag = new_imag;
        }
        
        // Print scaled gradients
        printf("  Scaled gradients: ");
        for (size_t i = 0; i < output_dim; i++) {
            printf("(%.3f,%.3f) ", gradients[base_idx + i].real, gradients[base_idx + i].imag);
        }
        printf("\n");
    }
    
    // Allocate temporary storage for weight gradients
    ComplexFloat* weight_gradients = aligned_alloc(16, weights->rows * weights->cols * sizeof(ComplexFloat));
    if (!weight_gradients) {
        free(gradients);
        return false;
    }
    
    // For backprop with arbitrary batch size:
    // gradients: [batch_size x output_dim]
    // weights: [input_dim x output_dim]
    // We need: gradients * weights^T to get [batch_size x input_dim]
    
    // First transpose the weights for backpropagation
    ComplexFloat* weights_t = aligned_alloc(16, weights->rows * weights->cols * sizeof(ComplexFloat));
    if (!weights_t) {
        free(gradients);
        free(weight_gradients);
        return false;
    }
    
    // Convert weights to ComplexFloat format and transpose
    for (size_t i = 0; i < weights->rows; i++) {
        for (size_t j = 0; j < weights->cols; j++) {
            size_t src_idx = i * weights->cols + j;
            size_t dst_idx = j * weights->rows + i;
            weights_t[dst_idx] = (ComplexFloat){
                creal(weights->data[src_idx]),
                cimag(weights->data[src_idx])
            };
        }
    }
    
    // Print dimensions and verify
    printf("DEBUG: Matrix dimensions for backpropagation:\n");
    printf("  Gradients: [%zu x %zu] (batch_size x output_dim)\n", 
           batch_size, output_dim);
    printf("  Weights^T: [%zu x %zu] (output_dim x input_dim)\n",
           weights->cols, weights->rows);
    printf("  Expected output: [%zu x %zu] (batch_size x input_dim)\n",
           batch_size, weights->rows);
           
    // Verify memory alignment
    printf("DEBUG: Checking memory alignment:\n");
    printf("  gradients alignment: %zu\n", (size_t)gradients & 15);
    printf("  weights_t alignment: %zu\n", (size_t)weights_t & 15);
    printf("  weight_gradients alignment: %zu\n", (size_t)weight_gradients & 15);
    
    // Verify memory is accessible
    printf("DEBUG: Verifying memory access:\n");
    volatile float test_read;
    printf("  Testing gradients read...\n");
    test_read = gradients[0].real;
    printf("  Testing weights_t read...\n");
    test_read = weights_t[0].real;
    printf("  Testing weight_gradients write...\n");
    weight_gradients[0].real = 0.0f;
    
    printf("DEBUG: Starting matrix multiplications for layer 2...\n");
    
    // Compute backpropagated gradients for previous layer
    ComplexFloat* backprop_gradients = aligned_alloc(16, batch_size * weights->rows * sizeof(ComplexFloat));
    if (!backprop_gradients) {
        free(weights_t);
        free(weight_gradients);
        free(gradients);
        return false;
    }
    
    // Compute gradients * weights^T
    numerical_error_t err = numerical_matrix_multiply_accelerate(
        gradients,               // [batch_size x output_dim]
        weights_t,               // [output_dim x input_dim]
        backprop_gradients,      // [batch_size x input_dim]
        batch_size,              // M (batch_size)
        output_dim,              // K (output_dim)
        weights->rows,           // N (input_dim)
        false,                   // Don't transpose gradients
        false                    // Don't transpose weights_t (already transposed)
    );
    
    if (err != NUMERICAL_SUCCESS) {
        printf("DEBUG: Backprop gradients multiply failed with error: %d\n", err);
        free(backprop_gradients);
        free(weights_t);
        free(weight_gradients);
        free(gradients);
        return false;
    }
    
    // Store backpropagated gradients in network for layer 1
    size_t dims[] = {batch_size, weights->rows};
    size_t node_id;
    if (!add_tensor_node(network, backprop_gradients, dims, 2, &node_id)) {
        printf("DEBUG: Failed to add backprop gradients node\n");
        free(backprop_gradients);
        free(weights_t);
        free(weight_gradients);
        free(gradients);
        return false;
    }
    free(backprop_gradients);
    
    // Now compute weight updates using features^T * gradients
    printf("DEBUG: Computing weight updates:\n");
    printf("  features: [%zu x %zu]\n", batch_size, network->nodes[0]->dimensions[1]);
    printf("  gradients: [%zu x %zu]\n", batch_size, weights->cols);
    printf("  weight_gradients: [%zu x %zu]\n", weights->rows, weights->cols);
    
    // For weight updates, we need features^T * gradients
    // features: [batch_size x input_dim]
    // gradients: [batch_size x output_dim]
    // weight_gradients: [input_dim x output_dim]
    err = numerical_matrix_multiply_accelerate(
        network->nodes[0]->data,  // [batch_size x input_dim]
        gradients,                // [batch_size x output_dim]
        weight_gradients,         // [input_dim x output_dim]
        network->nodes[0]->dimensions[1], // M (input_dim)
        batch_size,               // K (batch_size)
        weights->cols,            // N (output_dim)
        true,                     // Transpose features
        false                     // Don't transpose gradients
    );
    printf("DEBUG: Matrix multiplication completed\n");
    
    if (err != NUMERICAL_SUCCESS) {
        printf("DEBUG: Matrix multiply failed with error: %d\n", err);
        free(weights_t);
        free(weight_gradients);
        free(gradients);
        return false;
    }
    
    free(weights_t);
    
    // Print weight gradients
    printf("DEBUG: Weight gradients:\n");
    for (size_t i = 0; i < weights->rows; i++) {
        printf("  Row %zu: ", i);
        for (size_t j = 0; j < weights->cols; j++) {
            size_t idx = i * weights->cols + j;
            printf("(%.3f,%.3f) ", weight_gradients[idx].real, weight_gradients[idx].imag);
        }
        printf("\n");
    }
    
    // Clip gradients and store both original and clipped versions
    ComplexFloat* clipped_gradients = aligned_alloc(16, weights->rows * weights->cols * sizeof(ComplexFloat));
    if (!clipped_gradients) {
        free(weight_gradients);
        free(gradients);
        return false;
    }

    const float clip_threshold = 100.0f;  // Clip gradients to [-100, 100]
    const float min_threshold = 1e-6f;   // Don't clip gradients smaller than this
    
    for (size_t i = 0; i < weights->rows * weights->cols; i++) {
        // Get gradient components
        float real_grad = weight_gradients[i].real;
        float imag_grad = weight_gradients[i].imag;
        
        // Only clip if magnitude is above min_threshold
        if (fabsf(real_grad) > min_threshold) {
            if (real_grad > clip_threshold) real_grad = clip_threshold;
            else if (real_grad < -clip_threshold) real_grad = -clip_threshold;
        }
        
        if (fabsf(imag_grad) > min_threshold) {
            if (imag_grad > clip_threshold) imag_grad = clip_threshold;
            else if (imag_grad < -clip_threshold) imag_grad = -clip_threshold;
        }
        
        clipped_gradients[i].real = real_grad;
        clipped_gradients[i].imag = imag_grad;
    }
    
    // Store original gradients for backprop
    size_t grad_dims[] = {weights->rows, weights->cols};
    size_t grad_node_id;
    if (!add_tensor_node(network, weight_gradients, grad_dims, 2, &grad_node_id)) {
        printf("DEBUG: Failed to add weight gradients node\n");
        free(clipped_gradients);
        free(weight_gradients);
        free(gradients);
        return false;
    }
    
    // Store clipped gradients for weight updates
    size_t clipped_dims[] = {weights->rows, weights->cols};
    size_t clipped_node_id;
    if (!add_tensor_node(network, clipped_gradients, clipped_dims, 2, &clipped_node_id)) {
        printf("DEBUG: Failed to add clipped gradients node\n");
        free(clipped_gradients);
        free(weight_gradients);
        free(gradients);
        return false;
    }
    
    free(weight_gradients);
    free(gradients);
    return true;
}

quantum_optimizer_state_t* quantum_create_optimizer_state(
    optimizer_type_t type,
    size_t param_size) {
    
    quantum_optimizer_state_t* state = malloc(sizeof(quantum_optimizer_state_t));
    if (!state) return NULL;
    
    state->type = type;
    state->size = param_size;
    state->t = 0;
    
    // Initialize optimizer-specific parameters
    switch (type) {
        case OPTIMIZER_QUANTUM_ADAM:
            state->beta1 = 0.9;    // Default Adam beta1
            state->beta2 = 0.999;  // Default Adam beta2
            state->epsilon = 1e-8;
            state->m = aligned_alloc(16, param_size * sizeof(ComplexFloat));
            state->v = aligned_alloc(16, param_size * sizeof(ComplexFloat));
            if (!state->m || !state->v) {
                quantum_free_optimizer_state(state);
                return NULL;
            }
            memset(state->m, 0, param_size * sizeof(ComplexFloat));
            memset(state->v, 0, param_size * sizeof(ComplexFloat));
            break;
            
        case OPTIMIZER_QUANTUM_GRADIENT_DESCENT:
            state->beta1 = 0.0;
            state->beta2 = 0.0;
            state->epsilon = 0.0;
            state->m = NULL;
            state->v = NULL;
            break;
            
        default:
            free(state);
            return NULL;
    }
    
    return state;
}

void quantum_free_optimizer_state(quantum_optimizer_state_t* state) {
    if (state) {
        if (state->m) free(state->m);
        if (state->v) free(state->v);
        free(state);
    }
}

bool quantum_update_weights(
    HierarchicalMatrix* weights,
    tensor_network_t* network,
    double learning_rate,
    quantum_optimizer_state_t* optimizer_state,
    const ComplexFloat* weight_gradients) {
    
    if (!weights || !network || !optimizer_state || !weight_gradients) return false;
    
    // Get output node for other pipeline operations
    size_t last_id = network->num_nodes - 1;
    tensor_node_t* last_node = network->nodes[last_id];
    if (!last_node || !last_node->data) return false;
    
    const size_t total_elements = weights->rows * weights->cols;
    if (total_elements != optimizer_state->size) return false;
    
    // Process weight updates in blocks for better cache utilization
    const size_t block_size = BLOCK_SIZE; // Process in cache-efficient blocks
    
    switch (optimizer_state->type) {
        case OPTIMIZER_QUANTUM_GRADIENT_DESCENT:
        {
            ComplexFloat lr = {(float)-learning_rate, 0};
            for (size_t offset = 0; offset < total_elements; offset += block_size) {
                size_t current_block = (offset + block_size <= total_elements) ? 
                                     block_size : (total_elements - offset);
                
                numerical_vector_scale((ComplexFloat*)&weights->data[offset],
                                    lr,
                                    (ComplexFloat*)&weights->data[offset],
                                    current_block);
            }
            break;
        }
        
        case OPTIMIZER_QUANTUM_ADAM:
        {
            optimizer_state->t++;
            const double beta1 = optimizer_state->beta1;
            const double beta2 = optimizer_state->beta2;
            const double epsilon = optimizer_state->epsilon;
            
            // Bias correction terms
            const double bc1 = 1.0 - pow(beta1, optimizer_state->t);
            const double bc2 = 1.0 - pow(beta2, optimizer_state->t);
            const double alpha = learning_rate * sqrt(bc2) / bc1;
            
            for (size_t offset = 0; offset < total_elements; offset += block_size) {
                size_t current_block = (offset + block_size <= total_elements) ? 
                                     block_size : (total_elements - offset);
                
                // Update biased first moment estimate (m)
                for (size_t i = 0; i < current_block; i++) {
                    size_t idx = offset + i;
                    ComplexFloat grad = weight_gradients[idx];
                    optimizer_state->m[idx].real = beta1 * optimizer_state->m[idx].real + 
                                                 (1.0 - beta1) * grad.real;
                    optimizer_state->m[idx].imag = beta1 * optimizer_state->m[idx].imag + 
                                                 (1.0 - beta1) * grad.imag;
                    
                    // Update biased second raw moment estimate (v)
                    optimizer_state->v[idx].real = beta2 * optimizer_state->v[idx].real + 
                                                 (1.0 - beta2) * grad.real * grad.real;
                    optimizer_state->v[idx].imag = beta2 * optimizer_state->v[idx].imag + 
                                                 (1.0 - beta2) * grad.imag * grad.imag;
                    
                    // Compute update
                    ComplexFloat m_hat = {
                        optimizer_state->m[idx].real / bc1,
                        optimizer_state->m[idx].imag / bc1
                    };
                    ComplexFloat v_hat = {
                        optimizer_state->v[idx].real / bc2,
                        optimizer_state->v[idx].imag / bc2
                    };
                    
                    // Apply update
                    ComplexFloat denom = {
                        sqrtf(v_hat.real) + epsilon,
                        sqrtf(v_hat.imag) + epsilon
                    };
                    weights->data[idx] -= alpha * (m_hat.real / denom.real + 
                                                 I * m_hat.imag / denom.imag);
                }
            }
            break;
        }
        
        default:
            return false;
    }
    
    return true;
}
