/**
 * @file distributed_training.c
 * @brief Production implementation of distributed training infrastructure
 *
 * Provides high-performance distributed training capabilities that integrate
 * with the existing quantum geometric API:
 * - Parameter server architecture with MPI one-sided communication
 * - Integration with WorkloadManager and DistributionContext
 * - Integration with GradientOptimizer for gradient compression
 * - Integration with CommunicationOptimizer for optimized gradient sync
 * - Ring allreduce for efficient data parallelism
 * - Adam/AdamW optimizers with full state tracking
 */

#include "quantum_geometric/distributed/distributed_training.h"
#include "quantum_geometric/distributed/workload_distribution.h"
#include "quantum_geometric/distributed/gradient_optimizer.h"
#include "quantum_geometric/distributed/communication_optimizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <errno.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================================
// Global State
// ============================================================================

static bool g_distributed_initialized = false;
static int g_world_rank = -1;
static int g_world_size = -1;
static DistributedStats g_stats = {0};
static struct timespec g_start_time;

// Integration with existing components
static WorkloadManager* g_workload_manager = NULL;
static GradientOptimizer* g_gradient_optimizer = NULL;
static CommunicationOptimizer* g_comm_optimizer = NULL;

// ============================================================================
// Internal Helper Functions
// ============================================================================

/**
 * @brief Get current time in seconds with nanosecond precision
 */
static double get_time_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/**
 * @brief Compare doubles for qsort (ascending order)
 */
static int compare_doubles_asc(const void* a, const void* b) {
    double diff = *(const double*)a - *(const double*)b;
    return (diff > 0) - (diff < 0);
}

/**
 * @brief Find the k-th largest absolute value threshold
 */
static double find_topk_threshold(const double* values, size_t size, size_t k) {
    if (k == 0 || k > size) return 0.0;

    // Create array of absolute values
    double* abs_values = (double*)malloc(size * sizeof(double));
    if (!abs_values) return 0.0;

    for (size_t i = 0; i < size; i++) {
        abs_values[i] = fabs(values[i]);
    }

    // Sort in ascending order
    qsort(abs_values, size, sizeof(double), compare_doubles_asc);

    // Get k-th largest (from the end)
    double threshold = abs_values[size - k];
    free(abs_values);

    return threshold;
}

/**
 * @brief Apply Adam optimizer update to single parameter
 */
static inline void apply_adam_update_single(
    double* param,
    double* momentum,
    double* velocity,
    double gradient,
    double lr,
    double beta1,
    double beta2,
    double epsilon,
    double weight_decay,
    size_t step
) {
    // Bias correction factors
    double bias_correction1 = 1.0 - pow(beta1, (double)step);
    double bias_correction2 = 1.0 - pow(beta2, (double)step);

    // Prevent division by zero
    if (bias_correction1 < 1e-10) bias_correction1 = 1e-10;
    if (bias_correction2 < 1e-10) bias_correction2 = 1e-10;

    // Update first moment (momentum)
    *momentum = beta1 * (*momentum) + (1.0 - beta1) * gradient;

    // Update second moment (velocity)
    *velocity = beta2 * (*velocity) + (1.0 - beta2) * gradient * gradient;

    // Bias-corrected estimates
    double m_hat = (*momentum) / bias_correction1;
    double v_hat = (*velocity) / bias_correction2;

    // AdamW style decoupled weight decay
    if (weight_decay > 0.0) {
        *param -= lr * weight_decay * (*param);
    }

    // Parameter update with numerical stability
    double denom = sqrt(v_hat) + epsilon;
    if (denom < epsilon) denom = epsilon;
    *param -= lr * m_hat / denom;
}

/**
 * @brief Allocate aligned memory for SIMD operations
 */
static void* aligned_alloc_safe(size_t alignment, size_t size) {
    void* ptr = NULL;
#ifdef _POSIX_C_SOURCE
    if (posix_memalign(&ptr, alignment, size) != 0) {
        ptr = NULL;
    }
#else
    ptr = malloc(size);
#endif
    return ptr;
}

// ============================================================================
// Initialization and Cleanup
// ============================================================================

NodeContext* init_distributed_training(int* argc, char*** argv) {
    if (g_distributed_initialized) {
        fprintf(stderr, "Warning: Distributed training already initialized\n");
        return NULL;
    }

    NodeContext* ctx = (NodeContext*)calloc(1, sizeof(NodeContext));
    if (!ctx) {
        fprintf(stderr, "Error: Failed to allocate NodeContext\n");
        return NULL;
    }

    // Initialize MPI with thread support
    int provided;
    int result = MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided);
    if (result != MPI_SUCCESS) {
        fprintf(stderr, "Error: MPI_Init_thread failed\n");
        free(ctx);
        return NULL;
    }

#ifndef NO_MPI
    if (provided < MPI_THREAD_MULTIPLE) {
        fprintf(stderr, "Warning: MPI_THREAD_MULTIPLE not fully supported (got level %d)\n", provided);
    }
#endif

    // Get basic MPI info
    MPI_Comm_rank(MPI_COMM_WORLD, &ctx->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ctx->world_size);

    ctx->global_comm = MPI_COMM_WORLD;
    ctx->is_master = (ctx->rank == 0);

    // Role assignment: parameter servers vs workers
    // Use 1 parameter server for every 8 workers, minimum 1
    int num_param_servers = ctx->world_size / 8;
    if (num_param_servers < 1) num_param_servers = 1;
    if (num_param_servers > ctx->world_size / 2) {
        num_param_servers = (ctx->world_size > 1) ? ctx->world_size / 2 : 1;
    }

    // Color: 0 = parameter server, 1 = worker
    int color = (ctx->rank < num_param_servers) ? 0 : 1;
    int key = ctx->rank;

    result = MPI_Comm_split(ctx->global_comm, color, key, &ctx->local_comm);
    if (result != MPI_SUCCESS) {
        fprintf(stderr, "Error: MPI_Comm_split failed\n");
        MPI_Finalize();
        free(ctx);
        return NULL;
    }

    ctx->is_parameter_server = (color == 0);

    // Get local rank within split communicator
    MPI_Comm_rank(ctx->local_comm, &ctx->local_rank);
    MPI_Comm_size(ctx->local_comm, &ctx->local_size);

    // Create worker-only communicator
    if (!ctx->is_parameter_server) {
        ctx->worker_comm = ctx->local_comm;
    } else {
        ctx->worker_comm = MPI_COMM_NULL;
    }

    // Allocate aligned communication buffers
    ctx->buffer_size = DIST_COMM_BUFFER_SIZE;
    ctx->send_buffer = aligned_alloc_safe(64, ctx->buffer_size);
    ctx->recv_buffer = aligned_alloc_safe(64, ctx->buffer_size);

    if (!ctx->send_buffer || !ctx->recv_buffer) {
        fprintf(stderr, "Error: Failed to allocate communication buffers\n");
        cleanup_distributed_training(ctx);
        return NULL;
    }

    // Initialize integrated components
    // 1. Workload Manager - for work distribution
    g_workload_manager = init_workload_manager();
    if (!g_workload_manager) {
        fprintf(stderr, "Warning: Failed to initialize workload manager\n");
    }

    // 2. Gradient Optimizer - for gradient processing
    OptimizationConfig opt_config = {
        .algorithm = ADAM,
        .learning_rate = 0.001,
        .beta1 = 0.9,
        .beta2 = 0.999,
        .epsilon = 1e-8,
        .weight_decay = 0.0,
        .model_size = 0,  // Will be set later
        .use_quantum = false,
        .use_geometric = false,
        .noise_reduction = false,
        .compression = COMPRESSION_NONE,
        .compression_ratio = 1.0
    };
    g_gradient_optimizer = init_gradient_optimizer(&opt_config);
    if (!g_gradient_optimizer) {
        fprintf(stderr, "Warning: Failed to initialize gradient optimizer\n");
    }

    // 3. Communication Optimizer - for gradient synchronization
    CommConfig comm_config = {
        .buffer_size = ctx->buffer_size,
        .min_message_size = 4096,
        .max_concurrent = 4,
        .enable_compression = false,
        .enable_topology_aware = true,
        .use_pinned_memory = false,
        .numa_aware = true,
        .topology_aware = true,
        .numa_policy = 0
    };
    g_comm_optimizer = init_communication_optimizer(&comm_config);
    if (!g_comm_optimizer) {
        fprintf(stderr, "Warning: Failed to initialize communication optimizer\n");
    }

    // Set global state
    g_distributed_initialized = true;
    g_world_rank = ctx->rank;
    g_world_size = ctx->world_size;
    clock_gettime(CLOCK_MONOTONIC, &g_start_time);
    memset(&g_stats, 0, sizeof(g_stats));

    if (ctx->is_master) {
        printf("Distributed training initialized: %d nodes (%d param servers, %d workers)\n",
               ctx->world_size, num_param_servers, ctx->world_size - num_param_servers);
    }

    return ctx;
}

void cleanup_distributed_training(NodeContext* ctx) {
    if (!ctx) return;

    // Cleanup integrated components
    if (g_comm_optimizer) {
        cleanup_communication_optimizer(g_comm_optimizer);
        g_comm_optimizer = NULL;
    }

    if (g_gradient_optimizer) {
        cleanup_gradient_optimizer(g_gradient_optimizer);
        g_gradient_optimizer = NULL;
    }

    if (g_workload_manager) {
        cleanup_workload_manager(g_workload_manager);
        g_workload_manager = NULL;
    }

    // Free buffers
    free(ctx->send_buffer);
    free(ctx->recv_buffer);

    // Cleanup MPI communicators
    if (ctx->local_comm != MPI_COMM_NULL && ctx->local_comm != MPI_COMM_WORLD) {
        MPI_Comm_free(&ctx->local_comm);
    }

    MPI_Finalize();

    g_distributed_initialized = false;
    g_world_rank = -1;
    g_world_size = -1;

    free(ctx);
}

ParameterServer* init_parameter_server(NodeContext* ctx, size_t num_parameters) {
    if (!ctx) {
        fprintf(stderr, "Error: NULL context in init_parameter_server\n");
        return NULL;
    }

    ParameterServer* server = (ParameterServer*)calloc(1, sizeof(ParameterServer));
    if (!server) {
        fprintf(stderr, "Error: Failed to allocate ParameterServer\n");
        return NULL;
    }

    server->num_parameters = num_parameters;
    server->initialized = false;

    // Allocate shared memory window for parameters using MPI
    MPI_Comm target_comm = ctx->is_parameter_server ? ctx->local_comm : MPI_COMM_WORLD;

    int result = MPI_Win_allocate(
        num_parameters * sizeof(double),
        sizeof(double),
        MPI_INFO_NULL,
        target_comm,
        &server->parameters,
        &server->window
    );

    if (result != MPI_SUCCESS || server->parameters == NULL) {
        fprintf(stderr, "Error: MPI_Win_allocate failed\n");
        free(server);
        return NULL;
    }

    // Initialize parameters to small random values for symmetry breaking
    double* params = (double*)server->parameters;
    for (size_t i = 0; i < num_parameters; i++) {
        // Xavier/Glorot initialization approximation
        params[i] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
    }

    // Allocate optimizer state buffers with alignment
    server->gradients = aligned_alloc_safe(64, num_parameters * sizeof(double));
    server->momentum = aligned_alloc_safe(64, num_parameters * sizeof(double));
    server->velocity = aligned_alloc_safe(64, num_parameters * sizeof(double));

    if (!server->gradients || !server->momentum || !server->velocity) {
        fprintf(stderr, "Error: Failed to allocate optimizer buffers\n");
        cleanup_parameter_server(server);
        return NULL;
    }

    // Zero-initialize optimizer state
    memset(server->gradients, 0, num_parameters * sizeof(double));
    memset(server->momentum, 0, num_parameters * sizeof(double));
    memset(server->velocity, 0, num_parameters * sizeof(double));

    // Initialize compression settings (default: no compression)
    memset(&server->compression, 0, sizeof(GradientCompression));
    server->compression.ratio = 1.0;
    server->compression.original_size = num_parameters;

    // Default optimizer settings (Adam)
    server->learning_rate = 0.001;
    server->beta1 = 0.9;
    server->beta2 = 0.999;
    server->epsilon = 1e-8;
    server->step = 0;
    server->update_count = 0;

    server->initialized = true;

    return server;
}

void cleanup_parameter_server(ParameterServer* server) {
    if (!server) return;

    if (server->initialized && server->window != MPI_WIN_NULL) {
        MPI_Win_free(&server->window);
    }

    // Note: server->parameters is freed by MPI_Win_free
    free(server->gradients);
    free(server->momentum);
    free(server->velocity);

    cleanup_gradient_compression(&server->compression);

    free(server);
}

// ============================================================================
// Gradient Operations
// ============================================================================

int push_gradients(NodeContext* ctx,
                   ParameterServer* server,
                   const double* gradients,
                   size_t size) {
    if (!ctx || !server || !gradients || size == 0) return -1;
    if (!server->initialized) return -2;
    if (size > server->num_parameters) return -3;

    double start_time = get_time_seconds();

    // Use gradient optimizer preprocessing if available
    double* processed_gradients = (double*)gradients;
    bool free_processed = false;

    if (g_gradient_optimizer) {
        processed_gradients = (double*)malloc(size * sizeof(double));
        if (processed_gradients) {
            memcpy(processed_gradients, gradients, size * sizeof(double));
            preprocess_gradients(g_gradient_optimizer, processed_gradients, size);
            free_processed = true;
        } else {
            processed_gradients = (double*)gradients;
        }
    }

    // Check if compression is enabled
    if (server->compression.ratio < 1.0 && server->compression.sparsity_map) {
        // Compress gradients using our compression
        size_t compressed_size;
        void* compressed = compress_gradients_topk(processed_gradients, size,
                                                   &server->compression, &compressed_size);

        if (!compressed) {
            if (free_processed) free(processed_gradients);
            return -4;
        }

        // Lock window for atomic update
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, server->window);

        // Decompress and accumulate into server gradient buffer
        double* temp_grads = (double*)malloc(size * sizeof(double));
        if (temp_grads) {
            decompress_gradients_topk(compressed, compressed_size, temp_grads, size, &server->compression);

            // Accumulate with existing gradients
            #ifdef _OPENMP
            #pragma omp parallel for simd
            #endif
            for (size_t i = 0; i < size; i++) {
                ((double*)server->gradients)[i] += temp_grads[i];
            }
            free(temp_grads);
        }

        MPI_Win_unlock(0, server->window);

        free(compressed);

        g_stats.total_bytes_sent += compressed_size;
        g_stats.compression_ratio = (double)compressed_size / (size * sizeof(double));
    } else {
        // No compression - direct accumulation
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, server->window);

        #ifdef _OPENMP
        #pragma omp parallel for simd
        #endif
        for (size_t i = 0; i < size; i++) {
            ((double*)server->gradients)[i] += processed_gradients[i];
        }

        MPI_Win_unlock(0, server->window);

        g_stats.total_bytes_sent += size * sizeof(double);
    }

    if (free_processed) {
        free(processed_gradients);
    }

    server->update_count++;
    g_stats.num_gradient_updates++;
    g_stats.total_comm_time += get_time_seconds() - start_time;

    return 0;
}

int pull_parameters(NodeContext* ctx,
                    ParameterServer* server,
                    double* parameters,
                    size_t size) {
    if (!ctx || !server || !parameters || size == 0) return -1;
    if (!server->initialized) return -2;
    if (size > server->num_parameters) return -3;

    double start_time = get_time_seconds();

    // Lock window for read access
    MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, server->window);

    // Copy parameters
    memcpy(parameters, server->parameters, size * sizeof(double));

    MPI_Win_unlock(0, server->window);

    g_stats.num_parameter_pulls++;
    g_stats.total_bytes_received += size * sizeof(double);
    g_stats.total_comm_time += get_time_seconds() - start_time;

    return 0;
}

int sync_gradients_allreduce(NodeContext* ctx, double* gradients, size_t size) {
    if (!ctx || !gradients || size == 0) return -1;

    double start_time = get_time_seconds();

    // Use communication optimizer if available
    if (g_comm_optimizer) {
        GradientBuffer grad_buf = {
            .data = gradients,
            .size = size * sizeof(double),
            .count = size,
            .dtype = 0,  // double
            .is_compressed = false
        };

        int result = sync_gradients(g_comm_optimizer, &grad_buf, 1);
        if (result == 0) {
            g_stats.total_comm_time += get_time_seconds() - start_time;
            g_stats.total_bytes_sent += size * sizeof(double);
            g_stats.total_bytes_received += size * sizeof(double);
            return 0;
        }
        // Fall through to manual implementation if optimizer fails
    }

    // Use appropriate communicator
    MPI_Comm comm = (ctx->worker_comm != MPI_COMM_NULL) ? ctx->worker_comm : ctx->global_comm;

    // Perform allreduce sum
    int result = MPI_Allreduce(MPI_IN_PLACE, gradients, (int)size,
                               MPI_DOUBLE, MPI_SUM, comm);

    if (result != MPI_SUCCESS) return -2;

    // Average gradients across participants
    int comm_size;
    MPI_Comm_size(comm, &comm_size);

    if (comm_size > 1) {
        double scale = 1.0 / comm_size;
        #ifdef _OPENMP
        #pragma omp parallel for simd
        #endif
        for (size_t i = 0; i < size; i++) {
            gradients[i] *= scale;
        }
    }

    g_stats.total_comm_time += get_time_seconds() - start_time;
    g_stats.total_bytes_sent += size * sizeof(double);
    g_stats.total_bytes_received += size * sizeof(double);

    return 0;
}

int sync_gradients_ring(RingAllreduce* ring, double* gradients, size_t size) {
    if (!ring || !gradients || size == 0) return -1;

    double start_time = get_time_seconds();

    int ring_size = ring->size;

    // Single process - nothing to do
    if (ring_size <= 1) return 0;

    // Calculate chunk boundaries
    size_t base_chunk = size / ring_size;
    size_t remainder = size % ring_size;

    // Phase 1: Scatter-reduce
    // Each node accumulates a portion of the gradient
    for (int step = 0; step < ring_size - 1; step++) {
        int send_chunk_idx = (ring->rank - step + ring_size) % ring_size;
        int recv_chunk_idx = (ring->rank - step - 1 + ring_size) % ring_size;

        // Calculate chunk boundaries
        size_t send_offset = send_chunk_idx * base_chunk +
                            (send_chunk_idx < (int)remainder ? send_chunk_idx : remainder);
        size_t recv_offset = recv_chunk_idx * base_chunk +
                            (recv_chunk_idx < (int)remainder ? recv_chunk_idx : remainder);

        size_t send_count = base_chunk + (send_chunk_idx < (int)remainder ? 1 : 0);
        size_t recv_count = base_chunk + (recv_chunk_idx < (int)remainder ? 1 : 0);

        // Non-blocking communication
        MPI_Request send_req, recv_req;

        MPI_Isend(&gradients[send_offset], (int)send_count, MPI_DOUBLE,
                  ring->right_neighbor, step, ring->comm, &send_req);

        MPI_Irecv(ring->temp_buffer, (int)recv_count, MPI_DOUBLE,
                  ring->left_neighbor, step, ring->comm, &recv_req);

        // Wait for completion
        MPI_Status status;
        MPI_Wait(&recv_req, &status);
        MPI_Wait(&send_req, &status);

        // Accumulate received data
        double* recv_data = (double*)ring->temp_buffer;
        for (size_t i = 0; i < recv_count; i++) {
            gradients[recv_offset + i] += recv_data[i];
        }
    }

    // Phase 2: Allgather
    // Distribute complete chunks to all nodes
    for (int step = 0; step < ring_size - 1; step++) {
        int send_chunk_idx = (ring->rank - step + 1 + ring_size) % ring_size;
        int recv_chunk_idx = (ring->rank - step + ring_size) % ring_size;

        size_t send_offset = send_chunk_idx * base_chunk +
                            (send_chunk_idx < (int)remainder ? send_chunk_idx : remainder);
        size_t recv_offset = recv_chunk_idx * base_chunk +
                            (recv_chunk_idx < (int)remainder ? recv_chunk_idx : remainder);

        size_t send_count = base_chunk + (send_chunk_idx < (int)remainder ? 1 : 0);
        size_t recv_count = base_chunk + (recv_chunk_idx < (int)remainder ? 1 : 0);

        MPI_Request send_req, recv_req;

        MPI_Isend(&gradients[send_offset], (int)send_count, MPI_DOUBLE,
                  ring->right_neighbor, ring_size + step, ring->comm, &send_req);

        MPI_Irecv(&gradients[recv_offset], (int)recv_count, MPI_DOUBLE,
                  ring->left_neighbor, ring_size + step, ring->comm, &recv_req);

        MPI_Status status;
        MPI_Wait(&recv_req, &status);
        MPI_Wait(&send_req, &status);
    }

    // Average gradients by ring size
    double scale = 1.0 / ring_size;
    #ifdef _OPENMP
    #pragma omp parallel for simd
    #endif
    for (size_t i = 0; i < size; i++) {
        gradients[i] *= scale;
    }

    g_stats.total_comm_time += get_time_seconds() - start_time;
    g_stats.total_bytes_sent += 2 * size * sizeof(double);
    g_stats.total_bytes_received += 2 * size * sizeof(double);

    return 0;
}

// ============================================================================
// Compression Functions
// ============================================================================

int init_gradient_compression(GradientCompression* compression, size_t size, double ratio) {
    if (!compression || size == 0) return -1;
    if (ratio <= 0.0 || ratio > 1.0) return -2;

    memset(compression, 0, sizeof(GradientCompression));

    compression->ratio = ratio;
    compression->threshold = 0.0;
    compression->original_size = size;
    compression->map_size = (size + 7) / 8;  // Bitmap size in bytes

    compression->sparsity_map = (uint8_t*)calloc(compression->map_size, 1);
    if (!compression->sparsity_map) return -3;

    compression->use_error_feedback = true;
    compression->error_buffer = (double*)calloc(size, sizeof(double));
    if (!compression->error_buffer) {
        free(compression->sparsity_map);
        compression->sparsity_map = NULL;
        return -4;
    }

    return 0;
}

void cleanup_gradient_compression(GradientCompression* compression) {
    if (!compression) return;

    free(compression->sparsity_map);
    free(compression->error_buffer);

    compression->sparsity_map = NULL;
    compression->error_buffer = NULL;
}

void* compress_gradients_topk(const double* gradients,
                              size_t size,
                              GradientCompression* compression,
                              size_t* compressed_size) {
    if (!gradients || !compression || !compressed_size || size == 0) return NULL;

    // Apply error feedback if enabled
    double* adjusted_grads = (double*)malloc(size * sizeof(double));
    if (!adjusted_grads) return NULL;

    if (compression->use_error_feedback && compression->error_buffer) {
        for (size_t i = 0; i < size; i++) {
            adjusted_grads[i] = gradients[i] + compression->error_buffer[i];
        }
    } else {
        memcpy(adjusted_grads, gradients, size * sizeof(double));
    }

    // Calculate number of elements to keep
    size_t k = (size_t)(size * compression->ratio);
    if (k < 1) k = 1;
    if (k > size) k = size;

    // Find threshold for top-k selection
    compression->threshold = find_topk_threshold(adjusted_grads, size, k);

    // Clear sparsity map
    if (compression->sparsity_map) {
        memset(compression->sparsity_map, 0, compression->map_size);
    }

    // Allocate output buffer: bitmap + indices + values
    // Format: [map_size bytes bitmap][k size_t indices][k double values]
    size_t indices_size = k * sizeof(uint32_t);  // Use 32-bit indices to save space
    size_t values_size = k * sizeof(double);
    size_t buffer_size = compression->map_size + indices_size + values_size;

    void* buffer = malloc(buffer_size);
    if (!buffer) {
        free(adjusted_grads);
        return NULL;
    }

    uint8_t* bitmap = (uint8_t*)buffer;
    uint32_t* indices = (uint32_t*)((char*)buffer + compression->map_size);
    double* values = (double*)((char*)buffer + compression->map_size + indices_size);

    // Pack significant gradients
    size_t value_count = 0;
    for (size_t i = 0; i < size && value_count < k; i++) {
        if (fabs(adjusted_grads[i]) >= compression->threshold) {
            // Set bit in bitmap
            bitmap[i / 8] |= (uint8_t)(1 << (i % 8));
            if (compression->sparsity_map) {
                compression->sparsity_map[i / 8] |= (uint8_t)(1 << (i % 8));
            }

            // Store index and value
            indices[value_count] = (uint32_t)i;
            values[value_count] = adjusted_grads[i];
            value_count++;
        }
    }

    // Update error feedback buffer
    if (compression->use_error_feedback && compression->error_buffer) {
        for (size_t i = 0; i < size; i++) {
            bool was_sent = compression->sparsity_map &&
                           (compression->sparsity_map[i / 8] & (1 << (i % 8)));
            if (was_sent) {
                compression->error_buffer[i] = 0.0;
            } else {
                compression->error_buffer[i] = adjusted_grads[i];
            }
        }
    }

    free(adjusted_grads);

    *compressed_size = compression->map_size + value_count * sizeof(uint32_t) +
                      value_count * sizeof(double);
    return buffer;
}

void decompress_gradients_topk(const void* compressed,
                               size_t compressed_size,
                               double* gradients,
                               size_t size,
                               const GradientCompression* compression) {
    if (!compressed || !gradients || !compression || size == 0) return;

    size_t map_size = compression->map_size;
    if (compressed_size < map_size) return;

    // Clear output
    memset(gradients, 0, size * sizeof(double));

    const uint8_t* bitmap = (const uint8_t*)compressed;

    // Calculate how many values we have
    size_t header_size = map_size;
    size_t remaining = compressed_size - header_size;

    // Each entry is 4 bytes (index) + 8 bytes (value) = 12 bytes
    size_t num_values = remaining / (sizeof(uint32_t) + sizeof(double));

    const uint32_t* indices = (const uint32_t*)((const char*)compressed + map_size);
    const double* values = (const double*)((const char*)compressed + map_size +
                                           num_values * sizeof(uint32_t));

    // Unpack values
    for (size_t i = 0; i < num_values; i++) {
        uint32_t idx = indices[i];
        if (idx < size) {
            gradients[idx] = values[i];
        }
    }
}

// ============================================================================
// Optimizer Functions
// ============================================================================

int configure_optimizer(ParameterServer* server, const OptimizerConfig* config) {
    if (!server || !config) return -1;

    server->learning_rate = config->learning_rate;
    server->beta1 = config->beta1;
    server->beta2 = config->beta2;
    server->epsilon = config->epsilon;

    return 0;
}

int apply_optimizer_step(ParameterServer* server) {
    if (!server || !server->initialized) return -1;

    double start_time = get_time_seconds();

    // Increment step counter
    server->step++;

    // Average gradients by update count
    if (server->update_count > 1) {
        double scale = 1.0 / server->update_count;
        #ifdef _OPENMP
        #pragma omp parallel for simd
        #endif
        for (size_t i = 0; i < server->num_parameters; i++) {
            ((double*)server->gradients)[i] *= scale;
        }
    }

    // Compute gradient norm for statistics
    double grad_norm_sq = 0.0;
    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:grad_norm_sq)
    #endif
    for (size_t i = 0; i < server->num_parameters; i++) {
        double g = ((double*)server->gradients)[i];
        grad_norm_sq += g * g;
    }
    double grad_norm = sqrt(grad_norm_sq);

    // Exponential moving average of gradient norm
    if (g_stats.avg_gradient_norm == 0.0) {
        g_stats.avg_gradient_norm = grad_norm;
    } else {
        g_stats.avg_gradient_norm = 0.9 * g_stats.avg_gradient_norm + 0.1 * grad_norm;
    }

    // Apply Adam optimizer updates
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, server->window);

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (size_t i = 0; i < server->num_parameters; i++) {
        double* param = &((double*)server->parameters)[i];
        double* momentum = &server->momentum[i];
        double* velocity = &server->velocity[i];
        double gradient = ((double*)server->gradients)[i];

        apply_adam_update_single(param, momentum, velocity, gradient,
                                server->learning_rate, server->beta1, server->beta2,
                                server->epsilon, 0.0, server->step);
    }

    MPI_Win_unlock(0, server->window);

    // Clear gradients for next iteration
    memset(server->gradients, 0, server->num_parameters * sizeof(double));
    server->update_count = 0;

    g_stats.total_compute_time += get_time_seconds() - start_time;

    return 0;
}

// ============================================================================
// Data Parallel Functions
// ============================================================================

DataParallelContext* init_data_parallel(NodeContext* node_ctx, size_t global_batch_size) {
    if (!node_ctx) return NULL;

    DataParallelContext* ctx = (DataParallelContext*)calloc(1, sizeof(DataParallelContext));
    if (!ctx) return NULL;

    ctx->node_ctx = node_ctx;
    ctx->global_batch_size = global_batch_size;

    // Calculate number of workers
    if (!node_ctx->is_parameter_server && node_ctx->local_size > 0) {
        ctx->num_workers = (size_t)node_ctx->local_size;
    } else if (node_ctx->world_size > 1) {
        // Estimate workers as non-parameter-server nodes
        ctx->num_workers = (size_t)(node_ctx->world_size - node_ctx->world_size / 8);
        if (ctx->num_workers < 1) ctx->num_workers = 1;
    } else {
        ctx->num_workers = 1;
    }

    // Calculate local batch size
    if (ctx->num_workers > 0) {
        ctx->local_batch_size = global_batch_size / ctx->num_workers;

        // Handle remainder
        size_t remainder = global_batch_size % ctx->num_workers;
        if ((size_t)node_ctx->local_rank < remainder) {
            ctx->local_batch_size++;
        }
    } else {
        ctx->local_batch_size = global_batch_size;
    }

    ctx->gradient_scale = 1.0 / (double)(ctx->num_workers > 0 ? ctx->num_workers : 1);
    ctx->overlap_comm_compute = true;

    // Initialize ring allreduce for efficient gradient sync
    if (!node_ctx->is_parameter_server && node_ctx->worker_comm != MPI_COMM_NULL) {
        ctx->ring = init_ring_allreduce(node_ctx->worker_comm, DIST_COMM_BUFFER_SIZE);
    }

    return ctx;
}

void cleanup_data_parallel(DataParallelContext* ctx) {
    if (!ctx) return;

    if (ctx->ring) {
        cleanup_ring_allreduce(ctx->ring);
    }

    if (ctx->compression) {
        cleanup_gradient_compression(ctx->compression);
        free(ctx->compression);
    }

    free(ctx);
}

void get_local_batch_range(DataParallelContext* ctx,
                           size_t total_samples,
                           size_t* start_idx,
                           size_t* end_idx) {
    if (!ctx || !start_idx || !end_idx) return;

    // Use workload distribution API if available
    if (g_workload_manager) {
        // Distribute using the existing workload distribution system
        size_t local_size = distribute_workload(total_samples);
        size_t local_offset = get_local_offset();

        *start_idx = local_offset;
        *end_idx = local_offset + local_size;
        return;
    }

    // Fallback to manual calculation
    if (ctx->num_workers == 0 || ctx->num_workers == 1) {
        *start_idx = 0;
        *end_idx = total_samples;
        return;
    }

    size_t samples_per_worker = total_samples / ctx->num_workers;
    size_t remainder = total_samples % ctx->num_workers;

    int rank = ctx->node_ctx->local_rank;

    if ((size_t)rank < remainder) {
        *start_idx = (size_t)rank * (samples_per_worker + 1);
        *end_idx = *start_idx + samples_per_worker + 1;
    } else {
        *start_idx = remainder * (samples_per_worker + 1) +
                     ((size_t)rank - remainder) * samples_per_worker;
        *end_idx = *start_idx + samples_per_worker;
    }

    if (*end_idx > total_samples) {
        *end_idx = total_samples;
    }
}

// ============================================================================
// Ring Allreduce Functions
// ============================================================================

RingAllreduce* init_ring_allreduce(MPI_Comm comm, size_t buffer_size) {
    RingAllreduce* ring = (RingAllreduce*)calloc(1, sizeof(RingAllreduce));
    if (!ring) return NULL;

    ring->comm = comm;
    MPI_Comm_rank(comm, &ring->rank);
    MPI_Comm_size(comm, &ring->size);

    // Calculate neighbors in ring topology
    ring->left_neighbor = (ring->rank - 1 + ring->size) % ring->size;
    ring->right_neighbor = (ring->rank + 1) % ring->size;

    // Allocate temporary buffer with alignment
    ring->temp_buffer = aligned_alloc_safe(64, buffer_size);
    if (!ring->temp_buffer) {
        free(ring);
        return NULL;
    }

    ring->chunk_size = 0;
    ring->num_chunks = 0;

    return ring;
}

void cleanup_ring_allreduce(RingAllreduce* ring) {
    if (!ring) return;

    free(ring->temp_buffer);
    free(ring);
}

// ============================================================================
// Statistics Functions
// ============================================================================

void get_distributed_stats(NodeContext* ctx,
                           ParameterServer* server,
                           DistributedStats* stats) {
    if (!stats) return;

    *stats = g_stats;

    // Calculate elapsed time
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    double elapsed = (now.tv_sec - g_start_time.tv_sec) +
                    (now.tv_nsec - g_start_time.tv_nsec) * 1e-9;

    // Calculate throughput
    if (elapsed > 0 && g_stats.num_gradient_updates > 0) {
        stats->throughput_samples_sec = (double)g_stats.num_gradient_updates / elapsed;
    }

    // Get parameter server specific stats
    if (server && server->initialized) {
        if (server->compression.ratio < 1.0) {
            stats->compression_ratio = server->compression.ratio;
        }
    }
}

void reset_distributed_stats(NodeContext* ctx) {
    (void)ctx;

    memset(&g_stats, 0, sizeof(g_stats));
    clock_gettime(CLOCK_MONOTONIC, &g_start_time);
}

// ============================================================================
// Utility Functions
// ============================================================================

int dist_training_barrier(NodeContext* ctx) {
    if (!ctx) return -1;

    double start_time = get_time_seconds();
    int result = MPI_Barrier(ctx->global_comm);
    g_stats.total_sync_time += get_time_seconds() - start_time;

    return result == MPI_SUCCESS ? 0 : -1;
}

int distributed_broadcast(NodeContext* ctx, void* data, size_t size) {
    if (!ctx || !data || size == 0) return -1;

    double start_time = get_time_seconds();

    int result = MPI_Bcast(data, (int)size, MPI_BYTE, 0, ctx->global_comm);

    g_stats.total_comm_time += get_time_seconds() - start_time;

    if (ctx->is_master) {
        g_stats.total_bytes_sent += size * (size_t)(ctx->world_size - 1);
    } else {
        g_stats.total_bytes_received += size;
    }

    return result == MPI_SUCCESS ? 0 : -1;
}

bool is_distributed_initialized(void) {
    return g_distributed_initialized;
}

int get_world_rank(void) {
    return g_world_rank;
}

int get_world_size(void) {
    return g_world_size;
}
