/**
 * @file distributed_training.h
 * @brief Distributed training API for quantum geometric computing
 *
 * Provides infrastructure for distributed parameter servers, gradient
 * compression, and communication optimization for large-scale training.
 */

#ifndef DISTRIBUTED_TRAINING_H
#define DISTRIBUTED_TRAINING_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

// MPI Compatibility Layer
#ifndef HAS_MPI
#ifndef NO_MPI
#define NO_MPI
#endif
#endif

#ifndef NO_MPI
#include <mpi.h>
#else
// MPI type stubs for non-MPI builds
typedef int MPI_Comm;
typedef int MPI_Win;
typedef int MPI_Status;
typedef int MPI_Request;
typedef int MPI_Datatype;
typedef int MPI_Op;
typedef int MPI_Info;

#define MPI_COMM_WORLD 0
#define MPI_COMM_NULL (-1)
#define MPI_WIN_NULL 0
#define MPI_BYTE 0
#define MPI_FLOAT 1
#define MPI_DOUBLE 2
#define MPI_SUM 0
#define MPI_SUCCESS 0
#define MPI_INFO_NULL 0
#define MPI_THREAD_MULTIPLE 3
#define MPI_IN_PLACE ((void*)-1)

// Stub MPI functions
static inline int MPI_Init_thread(int* argc, char*** argv, int required, int* provided) {
    (void)argc; (void)argv; (void)required;
    *provided = MPI_THREAD_MULTIPLE;
    return MPI_SUCCESS;
}
static inline int MPI_Comm_rank(MPI_Comm comm, int* rank) { (void)comm; *rank = 0; return MPI_SUCCESS; }
static inline int MPI_Comm_size(MPI_Comm comm, int* size) { (void)comm; *size = 1; return MPI_SUCCESS; }
static inline int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm* newcomm) {
    (void)comm; (void)color; (void)key; *newcomm = 0; return MPI_SUCCESS;
}
static inline int MPI_Comm_free(MPI_Comm* comm) { *comm = MPI_COMM_NULL; return MPI_SUCCESS; }
static inline int MPI_Allreduce(const void* sb, void* rb, int c, MPI_Datatype dt, MPI_Op op, MPI_Comm comm) {
    (void)dt; (void)op; (void)comm;
    // Handle MPI_IN_PLACE: send buffer is same as receive buffer
    if (sb == MPI_IN_PLACE) {
        // In-place operation - data already in rb, nothing to do for single process
        return MPI_SUCCESS;
    }
    if (sb != rb) memcpy(rb, sb, (size_t)c * sizeof(double));
    return MPI_SUCCESS;
}
static inline int MPI_Reduce(const void* sb, void* rb, int c, MPI_Datatype dt, MPI_Op op, int root, MPI_Comm comm) {
    (void)dt; (void)op; (void)root; (void)comm;
    if (sb != rb && rb) memcpy(rb, sb, (size_t)c * sizeof(double));
    return MPI_SUCCESS;
}
static inline int MPI_Barrier(MPI_Comm comm) { (void)comm; return MPI_SUCCESS; }
static inline int MPI_Win_create(void* b, size_t s, int d, MPI_Info info, MPI_Comm c, MPI_Win* w) {
    (void)b; (void)s; (void)d; (void)info; (void)c; *w = 0; return MPI_SUCCESS;
}
static inline int MPI_Win_allocate(size_t size, int disp_unit, MPI_Info info, MPI_Comm comm, void* baseptr, MPI_Win* win) {
    (void)disp_unit; (void)info; (void)comm;
    void** ptr = (void**)baseptr;
    *ptr = calloc(1, size);
    *win = (*ptr != NULL) ? 1 : 0;
    return (*ptr != NULL) ? MPI_SUCCESS : -1;
}
static inline int MPI_Win_free(MPI_Win* w) { *w = MPI_WIN_NULL; return MPI_SUCCESS; }
static inline int MPI_Win_lock(int lock_type, int rank, int assert, MPI_Win win) {
    (void)lock_type; (void)rank; (void)assert; (void)win; return MPI_SUCCESS;
}
static inline int MPI_Win_unlock(int rank, MPI_Win win) { (void)rank; (void)win; return MPI_SUCCESS; }
static inline int MPI_Bcast(void* b, int c, MPI_Datatype dt, int root, MPI_Comm comm) {
    (void)b; (void)c; (void)dt; (void)root; (void)comm; return MPI_SUCCESS;
}
static inline int MPI_Send(const void* buf, int count, MPI_Datatype dt, int dest, int tag, MPI_Comm comm) {
    (void)buf; (void)count; (void)dt; (void)dest; (void)tag; (void)comm; return MPI_SUCCESS;
}
static inline int MPI_Recv(void* buf, int count, MPI_Datatype dt, int src, int tag, MPI_Comm comm, MPI_Status* status) {
    (void)buf; (void)count; (void)dt; (void)src; (void)tag; (void)comm; (void)status; return MPI_SUCCESS;
}
static inline int MPI_Isend(const void* buf, int count, MPI_Datatype dt, int dest, int tag, MPI_Comm comm, MPI_Request* req) {
    (void)buf; (void)count; (void)dt; (void)dest; (void)tag; (void)comm; *req = 0; return MPI_SUCCESS;
}
static inline int MPI_Irecv(void* buf, int count, MPI_Datatype dt, int src, int tag, MPI_Comm comm, MPI_Request* req) {
    (void)buf; (void)count; (void)dt; (void)src; (void)tag; (void)comm; *req = 0; return MPI_SUCCESS;
}
static inline int MPI_Wait(MPI_Request* req, MPI_Status* status) { (void)req; (void)status; return MPI_SUCCESS; }
static inline int MPI_Finalize(void) { return MPI_SUCCESS; }
#define MPI_LOCK_EXCLUSIVE 0
#define MPI_LOCK_SHARED 1
#endif

// ============================================================================
// Training Configuration Constants
// ============================================================================

#define DIST_MAX_NODES 64
#define DIST_PIPELINE_STAGES 4
#define DIST_COMM_BUFFER_SIZE (16 * 1024 * 1024)  // 16MB
#define DIST_COMPRESSION_RATIO 0.01               // Top 1% gradients
#define DIST_WARMUP_ITERATIONS 100

// ============================================================================
// Gradient Compression Types
// ============================================================================

/**
 * @brief Gradient compression configuration
 *
 * Supports top-k sparsification for reducing communication overhead.
 */
typedef struct GradientCompression {
    double ratio;              /**< Fraction of gradients to keep (0.0-1.0) */
    double threshold;          /**< Computed threshold for selection */
    uint8_t* sparsity_map;     /**< Bitmap indicating non-zero positions */
    size_t map_size;           /**< Size of sparsity map in bytes */
    size_t original_size;      /**< Original number of elements */
    bool use_error_feedback;   /**< Accumulate compression errors */
    double* error_buffer;      /**< Buffer for error feedback */
} GradientCompression;

/**
 * @brief Compression method enumeration
 */
typedef enum {
    COMPRESS_NONE = 0,         /**< No compression */
    COMPRESS_TOPK,             /**< Top-K sparsification */
    COMPRESS_RANDOM_K,         /**< Random-K sparsification */
    COMPRESS_THRESHOLD,        /**< Threshold-based sparsification */
    COMPRESS_QUANTIZE_8BIT,    /**< 8-bit quantization */
    COMPRESS_QUANTIZE_1BIT     /**< Sign-based 1-bit quantization */
} CompressionMethod;

// ============================================================================
// Node and Communication Context
// ============================================================================

/**
 * @brief Node context for distributed training
 *
 * Represents a single node in the distributed training cluster.
 */
typedef struct NodeContext {
    int rank;                  /**< This node's rank in global communicator */
    int world_size;            /**< Total number of nodes */
    MPI_Comm global_comm;      /**< Global communicator */
    MPI_Comm local_comm;       /**< Local (intra-node) communicator */
    MPI_Comm worker_comm;      /**< Worker-only communicator */
    void* send_buffer;         /**< Pre-allocated send buffer */
    void* recv_buffer;         /**< Pre-allocated receive buffer */
    size_t buffer_size;        /**< Size of communication buffers */
    bool is_master;            /**< Whether this is the master node */
    bool is_parameter_server;  /**< Whether this is a parameter server */
    int local_rank;            /**< Rank within local node */
    int local_size;            /**< Number of processes on local node */
} NodeContext;

/**
 * @brief Ring-allreduce communication pattern
 */
typedef struct RingAllreduce {
    MPI_Comm comm;             /**< Communicator for ring operations */
    int rank;                  /**< Rank in ring */
    int size;                  /**< Size of ring */
    int left_neighbor;         /**< Left neighbor rank */
    int right_neighbor;        /**< Right neighbor rank */
    void* temp_buffer;         /**< Temporary buffer for operations */
    size_t chunk_size;         /**< Size of each chunk */
    size_t num_chunks;         /**< Number of chunks */
} RingAllreduce;

// ============================================================================
// Parameter Server Types
// ============================================================================

/**
 * @brief Parameter server for centralized gradient aggregation
 *
 * Manages model parameters and gradient accumulation using
 * MPI one-sided communication for efficient updates.
 */
typedef struct ParameterServer {
    void* parameters;          /**< Current model parameters (MPI window) */
    void* gradients;           /**< Accumulated gradients */
    size_t num_parameters;     /**< Total number of parameters */
    MPI_Win window;            /**< MPI window for one-sided communication */
    bool initialized;          /**< Whether server is initialized */
    double* momentum;          /**< Momentum buffer for optimizer */
    double* velocity;          /**< Velocity buffer (Adam/RMSprop) */
    GradientCompression compression; /**< Gradient compression config */
    int update_count;          /**< Number of updates received */
    double learning_rate;      /**< Current learning rate */
    double beta1;              /**< Momentum coefficient */
    double beta2;              /**< RMSprop coefficient */
    double epsilon;            /**< Numerical stability constant */
    size_t step;               /**< Current optimization step */
} ParameterServer;

/**
 * @brief Optimizer type for parameter updates
 */
typedef enum {
    OPTIMIZER_SGD = 0,         /**< Stochastic gradient descent */
    OPTIMIZER_MOMENTUM,        /**< SGD with momentum */
    OPTIMIZER_ADAM,            /**< Adam optimizer */
    OPTIMIZER_ADAMW,           /**< Adam with weight decay */
    OPTIMIZER_LAMB,            /**< Layer-wise Adaptive Moments */
    OPTIMIZER_LARS              /**< Layer-wise Adaptive Rate Scaling */
} OptimizerType;

/**
 * @brief Optimizer configuration
 */
typedef struct OptimizerConfig {
    OptimizerType type;        /**< Optimizer type */
    double learning_rate;      /**< Initial learning rate */
    double weight_decay;       /**< L2 regularization factor */
    double beta1;              /**< First moment coefficient (default: 0.9) */
    double beta2;              /**< Second moment coefficient (default: 0.999) */
    double epsilon;            /**< Numerical stability (default: 1e-8) */
    double momentum;           /**< Momentum factor for SGD */
    bool use_nesterov;         /**< Use Nesterov momentum */
    double trust_ratio;        /**< Trust ratio for LAMB/LARS */
} OptimizerConfig;

// ============================================================================
// Data Parallelism Types
// ============================================================================

/**
 * @brief Data parallel training context
 */
typedef struct DataParallelContext {
    NodeContext* node_ctx;     /**< Node context */
    size_t global_batch_size;  /**< Total batch size across all workers */
    size_t local_batch_size;   /**< Batch size per worker */
    size_t num_workers;        /**< Number of data parallel workers */
    GradientCompression* compression; /**< Optional gradient compression */
    RingAllreduce* ring;       /**< Ring allreduce for gradients */
    bool overlap_comm_compute; /**< Overlap communication and computation */
    double gradient_scale;     /**< Scaling factor for gradients */
} DataParallelContext;

/**
 * @brief Model parallel sharding strategy
 */
typedef enum {
    SHARD_NONE = 0,            /**< No sharding */
    SHARD_TENSOR,              /**< Tensor parallelism (column/row) */
    SHARD_PIPELINE,            /**< Pipeline parallelism */
    SHARD_EXPERT,              /**< Expert parallelism (MoE) */
    SHARD_ZERO_1,              /**< ZeRO stage 1 (optimizer states) */
    SHARD_ZERO_2,              /**< ZeRO stage 2 (+ gradients) */
    SHARD_ZERO_3               /**< ZeRO stage 3 (+ parameters) */
} ShardingStrategy;

// ============================================================================
// Training Statistics
// ============================================================================

/**
 * @brief Distributed training statistics
 */
typedef struct DistributedStats {
    double total_compute_time;        /**< Total computation time (seconds) */
    double total_comm_time;           /**< Total communication time (seconds) */
    double total_sync_time;           /**< Total synchronization time (seconds) */
    size_t total_bytes_sent;          /**< Total bytes sent */
    size_t total_bytes_received;      /**< Total bytes received */
    size_t num_gradient_updates;      /**< Number of gradient updates */
    size_t num_parameter_pulls;       /**< Number of parameter pulls */
    double compression_ratio;         /**< Achieved compression ratio */
    double throughput_samples_sec;    /**< Training throughput */
    double avg_gradient_norm;         /**< Average gradient norm */
    size_t num_stale_updates;         /**< Number of stale updates (async) */
} DistributedStats;

// ============================================================================
// Initialization and Cleanup
// ============================================================================

/**
 * @brief Initialize distributed training environment
 *
 * @param argc Pointer to argc from main
 * @param argv Pointer to argv from main
 * @return Initialized node context or NULL on failure
 */
NodeContext* init_distributed_training(int* argc, char*** argv);

/**
 * @brief Cleanup distributed training resources
 *
 * @param ctx Node context to cleanup
 */
void cleanup_distributed_training(NodeContext* ctx);

/**
 * @brief Initialize parameter server
 *
 * @param ctx Node context (must be parameter server role)
 * @param num_parameters Number of model parameters
 * @return Initialized parameter server or NULL on failure
 */
ParameterServer* init_parameter_server(NodeContext* ctx, size_t num_parameters);

/**
 * @brief Cleanup parameter server
 *
 * @param server Parameter server to cleanup
 */
void cleanup_parameter_server(ParameterServer* server);

// ============================================================================
// Gradient Operations
// ============================================================================

/**
 * @brief Push gradients to parameter server
 *
 * @param ctx Node context
 * @param server Parameter server
 * @param gradients Local gradient buffer
 * @param size Number of gradient elements
 * @return 0 on success, negative on error
 */
int push_gradients(NodeContext* ctx,
                   ParameterServer* server,
                   const double* gradients,
                   size_t size);

/**
 * @brief Pull parameters from parameter server
 *
 * @param ctx Node context
 * @param server Parameter server
 * @param parameters Output parameter buffer
 * @param size Number of parameters to pull
 * @return 0 on success, negative on error
 */
int pull_parameters(NodeContext* ctx,
                    ParameterServer* server,
                    double* parameters,
                    size_t size);

/**
 * @brief Synchronize gradients across all workers (allreduce)
 *
 * @param ctx Node context
 * @param gradients In/out gradient buffer
 * @param size Number of gradient elements
 * @return 0 on success, negative on error
 */
int sync_gradients_allreduce(NodeContext* ctx,
                             double* gradients,
                             size_t size);

/**
 * @brief Synchronize gradients using ring allreduce
 *
 * @param ring Ring allreduce context
 * @param gradients In/out gradient buffer
 * @param size Number of gradient elements
 * @return 0 on success, negative on error
 */
int sync_gradients_ring(RingAllreduce* ring,
                        double* gradients,
                        size_t size);

// ============================================================================
// Compression Functions
// ============================================================================

/**
 * @brief Initialize gradient compression
 *
 * @param compression Compression config to initialize
 * @param size Number of gradient elements
 * @param ratio Compression ratio (fraction to keep)
 * @return 0 on success, negative on error
 */
int init_gradient_compression(GradientCompression* compression,
                              size_t size,
                              double ratio);

/**
 * @brief Cleanup gradient compression
 *
 * @param compression Compression config to cleanup
 */
void cleanup_gradient_compression(GradientCompression* compression);

/**
 * @brief Compress gradients using top-k sparsification
 *
 * @param gradients Input gradients
 * @param size Number of elements
 * @param compression Compression configuration
 * @param compressed_size Output: size of compressed data
 * @return Compressed buffer (caller must free) or NULL on error
 */
void* compress_gradients_topk(const double* gradients,
                              size_t size,
                              GradientCompression* compression,
                              size_t* compressed_size);

/**
 * @brief Decompress gradients from top-k format
 *
 * @param compressed Compressed gradient data
 * @param compressed_size Size of compressed data
 * @param gradients Output gradient buffer
 * @param size Number of elements in output
 * @param compression Compression configuration
 */
void decompress_gradients_topk(const void* compressed,
                               size_t compressed_size,
                               double* gradients,
                               size_t size,
                               const GradientCompression* compression);

// ============================================================================
// Optimizer Functions
// ============================================================================

/**
 * @brief Configure parameter server optimizer
 *
 * @param server Parameter server
 * @param config Optimizer configuration
 * @return 0 on success, negative on error
 */
int configure_optimizer(ParameterServer* server, const OptimizerConfig* config);

/**
 * @brief Apply optimizer step to parameters
 *
 * @param server Parameter server with accumulated gradients
 * @return 0 on success, negative on error
 */
int apply_optimizer_step(ParameterServer* server);

// ============================================================================
// Data Parallel Functions
// ============================================================================

/**
 * @brief Initialize data parallel context
 *
 * @param node_ctx Node context
 * @param global_batch_size Total batch size
 * @return Data parallel context or NULL on error
 */
DataParallelContext* init_data_parallel(NodeContext* node_ctx,
                                        size_t global_batch_size);

/**
 * @brief Cleanup data parallel context
 *
 * @param ctx Data parallel context
 */
void cleanup_data_parallel(DataParallelContext* ctx);

/**
 * @brief Get local batch range for this worker
 *
 * @param ctx Data parallel context
 * @param total_samples Total number of samples
 * @param start_idx Output: starting index
 * @param end_idx Output: ending index (exclusive)
 */
void get_local_batch_range(DataParallelContext* ctx,
                           size_t total_samples,
                           size_t* start_idx,
                           size_t* end_idx);

// ============================================================================
// Ring Allreduce Functions
// ============================================================================

/**
 * @brief Initialize ring allreduce
 *
 * @param comm MPI communicator
 * @param buffer_size Size of temporary buffer
 * @return Ring allreduce context or NULL on error
 */
RingAllreduce* init_ring_allreduce(MPI_Comm comm, size_t buffer_size);

/**
 * @brief Cleanup ring allreduce
 *
 * @param ring Ring allreduce context
 */
void cleanup_ring_allreduce(RingAllreduce* ring);

// ============================================================================
// Statistics Functions
// ============================================================================

/**
 * @brief Get distributed training statistics
 *
 * @param ctx Node context
 * @param server Parameter server (may be NULL)
 * @param stats Output statistics structure
 */
void get_distributed_stats(NodeContext* ctx,
                           ParameterServer* server,
                           DistributedStats* stats);

/**
 * @brief Reset distributed training statistics
 *
 * @param ctx Node context
 */
void reset_distributed_stats(NodeContext* ctx);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Barrier synchronization across all nodes
 *
 * @param ctx Node context
 * @return 0 on success, negative on error
 */
int dist_training_barrier(NodeContext* ctx);

/**
 * @brief Broadcast data from master to all workers
 *
 * @param ctx Node context
 * @param data Buffer to broadcast (input on master, output on workers)
 * @param size Size of data in bytes
 * @return 0 on success, negative on error
 */
int distributed_broadcast(NodeContext* ctx, void* data, size_t size);

/**
 * @brief Check if distributed training is initialized
 *
 * @return true if initialized, false otherwise
 */
bool is_distributed_initialized(void);

/**
 * @brief Get world rank
 *
 * @return Current process rank or -1 if not initialized
 */
int get_world_rank(void);

/**
 * @brief Get world size
 *
 * @return Number of processes or -1 if not initialized
 */
int get_world_size(void);

#ifdef __cplusplus
}
#endif

#endif // DISTRIBUTED_TRAINING_H
