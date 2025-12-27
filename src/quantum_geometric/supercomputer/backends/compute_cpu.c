/**
 * compute_cpu.c - CPU backend implementation
 *
 * This backend provides CPU-based compute using:
 * - OpenMP for multi-threaded parallelization
 * - SIMD operations (AVX2/NEON) via compute_simd.h
 * - MPI for distributed communication (optional)
 *
 * This is the fallback backend that is always available.
 */

#include "quantum_geometric/supercomputer/compute_backend.h"
#include "quantum_geometric/supercomputer/compute_simd.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if COMPUTE_HAS_MPI
#include <mpi.h>
#endif

// Platform-specific headers for memory detection
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

// ============================================================================
// CPU Backend Context
// ============================================================================

typedef struct CPUBackendContext {
    // Configuration
    int num_threads;
    int node_rank;
    int num_nodes;

    // MPI state (if available)
#if COMPUTE_HAS_MPI
    MPI_Comm comm;
    bool mpi_initialized_by_us;
#endif

    // Memory tracking
    size_t total_allocated;
    size_t peak_allocated;

    // Performance metrics
    ComputeMetrics metrics;

    // Thread-local workspace
    float** thread_workspaces;
    size_t workspace_size;
} CPUBackendContext;

// ============================================================================
// Lifecycle Operations
// ============================================================================

static ComputeBackend* cpu_init(const ComputeDistributedConfig* config) {
    CPUBackendContext* ctx = calloc(1, sizeof(CPUBackendContext));
    if (!ctx) return NULL;

    // Configure threading
    ctx->num_threads = config->num_threads_per_node;
    if (ctx->num_threads <= 0) {
#ifdef _OPENMP
        ctx->num_threads = omp_get_max_threads();
#else
        ctx->num_threads = 1;
#endif
    }

#ifdef _OPENMP
    omp_set_num_threads(ctx->num_threads);
#endif

    // Initialize MPI if available and needed
#if COMPUTE_HAS_MPI
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);

    if (!mpi_initialized && config->num_nodes > 1) {
        int provided;
        MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);
        ctx->mpi_initialized_by_us = true;
    }

    if (mpi_initialized || ctx->mpi_initialized_by_us) {
        MPI_Comm_dup(MPI_COMM_WORLD, &ctx->comm);
        MPI_Comm_rank(ctx->comm, &ctx->node_rank);
        MPI_Comm_size(ctx->comm, &ctx->num_nodes);
    } else {
        ctx->node_rank = 0;
        ctx->num_nodes = 1;
    }
#else
    ctx->node_rank = 0;
    ctx->num_nodes = 1;
#endif

    // Allocate thread-local workspaces
    ctx->workspace_size = config->host_buffer_size > 0 ?
                          config->host_buffer_size : (1024 * 1024);  // 1MB default

    ctx->thread_workspaces = calloc(ctx->num_threads, sizeof(float*));
    if (ctx->thread_workspaces) {
        for (int i = 0; i < ctx->num_threads; i++) {
            ctx->thread_workspaces[i] = aligned_alloc(SIMD_ALIGNMENT, ctx->workspace_size);
        }
    }

    return (ComputeBackend*)ctx;
}

static void cpu_cleanup(ComputeBackend* backend) {
    CPUBackendContext* ctx = (CPUBackendContext*)backend;
    if (!ctx) return;

    // Free thread workspaces
    if (ctx->thread_workspaces) {
        for (int i = 0; i < ctx->num_threads; i++) {
            free(ctx->thread_workspaces[i]);
        }
        free(ctx->thread_workspaces);
    }

#if COMPUTE_HAS_MPI
    if (ctx->comm != MPI_COMM_NULL) {
        MPI_Comm_free(&ctx->comm);
    }
    if (ctx->mpi_initialized_by_us) {
        MPI_Finalize();
    }
#endif

    free(ctx);
}

static bool cpu_probe(void) {
    return true;  // CPU backend is always available
}

static ComputeResult cpu_get_capabilities(ComputeBackend* backend,
                                           int* num_devices,
                                           size_t* total_memory) {
    CPUBackendContext* ctx = (CPUBackendContext*)backend;
    if (!ctx) return COMPUTE_ERROR_INVALID_ARGUMENT;

    if (num_devices) *num_devices = 1;  // CPU counts as 1 "device"

    // Get approximate available memory (platform-specific)
    if (total_memory) {
#if defined(__APPLE__)
        // macOS: use sysctl
        int64_t memsize;
        size_t len = sizeof(memsize);
        if (sysctlbyname("hw.memsize", &memsize, &len, NULL, 0) == 0) {
            *total_memory = (size_t)memsize;
        } else {
            *total_memory = 8ULL * 1024 * 1024 * 1024;  // Default 8GB
        }
#elif defined(__linux__)
        // Linux: read /proc/meminfo
        FILE* f = fopen("/proc/meminfo", "r");
        if (f) {
            char line[256];
            while (fgets(line, sizeof(line), f)) {
                unsigned long mem;
                if (sscanf(line, "MemTotal: %lu kB", &mem) == 1) {
                    *total_memory = mem * 1024;
                    break;
                }
            }
            fclose(f);
        } else {
            *total_memory = 8ULL * 1024 * 1024 * 1024;
        }
#else
        *total_memory = 8ULL * 1024 * 1024 * 1024;  // Default 8GB
#endif
    }

    return COMPUTE_SUCCESS;
}

// ============================================================================
// Memory Management
// ============================================================================

static void* cpu_alloc(ComputeBackend* backend, size_t size, ComputeMemType mem_type) {
    CPUBackendContext* ctx = (CPUBackendContext*)backend;
    (void)mem_type;  // All memory types are host memory for CPU backend

    void* ptr = aligned_alloc(SIMD_ALIGNMENT, size);
    if (ptr && ctx) {
        ctx->total_allocated += size;
        if (ctx->total_allocated > ctx->peak_allocated) {
            ctx->peak_allocated = ctx->total_allocated;
        }
    }
    return ptr;
}

static void cpu_free(ComputeBackend* backend, void* ptr, ComputeMemType mem_type) {
    (void)backend;
    (void)mem_type;
    free(ptr);
}

static ComputeResult cpu_memcpy(ComputeBackend* backend,
                                 void* dst, ComputeMemType dst_type,
                                 const void* src, ComputeMemType src_type,
                                 size_t size, ComputeStream* stream) {
    (void)backend;
    (void)dst_type;
    (void)src_type;
    (void)stream;

    memcpy(dst, src, size);
    return COMPUTE_SUCCESS;
}

static ComputeResult cpu_memset(ComputeBackend* backend,
                                 void* ptr, int value, size_t size,
                                 ComputeStream* stream) {
    (void)backend;
    (void)stream;

    memset(ptr, value, size);
    return COMPUTE_SUCCESS;
}

// ============================================================================
// Stream Management (No-op for CPU)
// ============================================================================

typedef struct CPUStream {
    int dummy;
} CPUStream;

static ComputeStream* cpu_create_stream(ComputeBackend* backend) {
    (void)backend;
    return (ComputeStream*)calloc(1, sizeof(CPUStream));
}

static void cpu_destroy_stream(ComputeBackend* backend, ComputeStream* stream) {
    (void)backend;
    free(stream);
}

static ComputeResult cpu_synchronize_stream(ComputeBackend* backend, ComputeStream* stream) {
    (void)backend;
    (void)stream;
    return COMPUTE_SUCCESS;  // CPU operations are synchronous
}

static ComputeEvent* cpu_create_event(ComputeBackend* backend) {
    (void)backend;
    return calloc(1, sizeof(int));  // Placeholder
}

static void cpu_destroy_event(ComputeBackend* backend, ComputeEvent* event) {
    (void)backend;
    free(event);
}

static ComputeResult cpu_record_event(ComputeBackend* backend,
                                       ComputeEvent* event,
                                       ComputeStream* stream) {
    (void)backend;
    (void)event;
    (void)stream;
    return COMPUTE_SUCCESS;
}

static ComputeResult cpu_wait_event(ComputeBackend* backend,
                                     ComputeStream* stream,
                                     ComputeEvent* event) {
    (void)backend;
    (void)stream;
    (void)event;
    return COMPUTE_SUCCESS;
}

// ============================================================================
// Quantum Operations
// ============================================================================

static ComputeResult cpu_quantum_unitary(ComputeBackend* backend,
                                          float* state, size_t state_size,
                                          const float* unitary, size_t unitary_size,
                                          ComputeStream* stream) {
    (void)backend;
    (void)stream;

    if (!state || !unitary || state_size == 0 || unitary_size == 0) {
        return COMPUTE_ERROR_INVALID_ARGUMENT;
    }

    // Allocate output buffer
    float* output = aligned_alloc(SIMD_ALIGNMENT, 2 * state_size * sizeof(float));
    if (!output) return COMPUTE_ERROR_OUT_OF_MEMORY;

    // Apply unitary: |out> = U|in>
    // state_size is number of complex elements
    // unitary is state_size x state_size complex matrix
    simd_complex_matrix_vector_mul(output, unitary, state, state_size, state_size);

    // Copy result back
    memcpy(state, output, 2 * state_size * sizeof(float));
    free(output);

    return COMPUTE_SUCCESS;
}

static ComputeResult cpu_quantum_normalize(ComputeBackend* backend,
                                            float* state, size_t size,
                                            ComputeStream* stream) {
    (void)backend;
    (void)stream;

    if (!state || size == 0) {
        return COMPUTE_ERROR_INVALID_ARGUMENT;
    }

    float norm = simd_complex_norm_float(state, size);
    if (norm > 1e-10f) {
        simd_complex_scale_float(state, size, 1.0f / norm);
    }

    return COMPUTE_SUCCESS;
}

static ComputeResult cpu_quantum_tensor_contract(ComputeBackend* backend,
                                                  float* result,
                                                  const float* a, const float* b,
                                                  size_t m, size_t n, size_t k,
                                                  ComputeStream* stream) {
    (void)backend;
    (void)stream;

    if (!result || !a || !b) {
        return COMPUTE_ERROR_INVALID_ARGUMENT;
    }

    // Complex matrix multiplication: C = A * B
    simd_complex_matrix_multiply(result, a, b, m, n, k);

    return COMPUTE_SUCCESS;
}

static ComputeResult cpu_quantum_gradient(ComputeBackend* backend,
                                           float* gradients,
                                           const float* forward_state,
                                           const float* backward_state,
                                           size_t size,
                                           ComputeStream* stream) {
    (void)backend;
    (void)stream;

    if (!gradients || !forward_state || !backward_state || size == 0) {
        return COMPUTE_ERROR_INVALID_ARGUMENT;
    }

    // Compute gradient as Re(<backward|forward>)
    // This is a simplified parameter-shift gradient
    float inner[2];
    simd_complex_inner_product(inner, backward_state, forward_state, size);

    // Store real part as gradient
    gradients[0] = inner[0];
    gradients[1] = inner[1];

    return COMPUTE_SUCCESS;
}

static ComputeResult cpu_quantum_inner_product(ComputeBackend* backend,
                                                float* result,
                                                const float* state_a,
                                                const float* state_b,
                                                size_t size,
                                                ComputeStream* stream) {
    (void)backend;
    (void)stream;

    if (!result || !state_a || !state_b || size == 0) {
        return COMPUTE_ERROR_INVALID_ARGUMENT;
    }

    simd_complex_inner_product(result, state_a, state_b, size);
    return COMPUTE_SUCCESS;
}

static ComputeResult cpu_quantum_expectation(ComputeBackend* backend,
                                              float* result,
                                              const float* state,
                                              const float* observable,
                                              size_t size,
                                              ComputeStream* stream) {
    (void)backend;
    (void)stream;

    if (!result || !state || !observable || size == 0) {
        return COMPUTE_ERROR_INVALID_ARGUMENT;
    }

    // Assuming diagonal observable
    *result = simd_expectation_diagonal(state, observable, size);
    return COMPUTE_SUCCESS;
}

// ============================================================================
// Collective Communication
// ============================================================================

static ComputeResult cpu_barrier(ComputeBackend* backend) {
#if COMPUTE_HAS_MPI
    CPUBackendContext* ctx = (CPUBackendContext*)backend;
    if (ctx && ctx->num_nodes > 1) {
        MPI_Barrier(ctx->comm);
    }
#else
    (void)backend;
#endif
    return COMPUTE_SUCCESS;
}

static ComputeResult cpu_broadcast(ComputeBackend* backend,
                                    void* data, size_t size,
                                    ComputeDataType dtype, int root) {
#if COMPUTE_HAS_MPI
    CPUBackendContext* ctx = (CPUBackendContext*)backend;
    if (ctx && ctx->num_nodes > 1) {
        MPI_Datatype mpi_type;
        switch (dtype) {
            case COMPUTE_DTYPE_FLOAT32:   mpi_type = MPI_FLOAT; break;
            case COMPUTE_DTYPE_FLOAT64:   mpi_type = MPI_DOUBLE; break;
            case COMPUTE_DTYPE_INT32:     mpi_type = MPI_INT; break;
            case COMPUTE_DTYPE_INT64:     mpi_type = MPI_LONG_LONG; break;
            default:                      mpi_type = MPI_BYTE; break;
        }
        MPI_Bcast(data, (int)size, mpi_type, root, ctx->comm);
    }
#else
    (void)backend;
    (void)data;
    (void)size;
    (void)dtype;
    (void)root;
#endif
    return COMPUTE_SUCCESS;
}

static ComputeResult cpu_allreduce(ComputeBackend* backend,
                                    const void* send_data, void* recv_data,
                                    size_t count, ComputeDataType dtype,
                                    ComputeReduceOp op) {
#if COMPUTE_HAS_MPI
    CPUBackendContext* ctx = (CPUBackendContext*)backend;
    if (ctx && ctx->num_nodes > 1) {
        MPI_Datatype mpi_type;
        MPI_Op mpi_op;

        switch (dtype) {
            case COMPUTE_DTYPE_FLOAT32:   mpi_type = MPI_FLOAT; break;
            case COMPUTE_DTYPE_FLOAT64:   mpi_type = MPI_DOUBLE; break;
            case COMPUTE_DTYPE_INT32:     mpi_type = MPI_INT; break;
            case COMPUTE_DTYPE_INT64:     mpi_type = MPI_LONG_LONG; break;
            default:                      mpi_type = MPI_BYTE; break;
        }

        switch (op) {
            case COMPUTE_REDUCE_SUM:  mpi_op = MPI_SUM; break;
            case COMPUTE_REDUCE_PROD: mpi_op = MPI_PROD; break;
            case COMPUTE_REDUCE_MIN:  mpi_op = MPI_MIN; break;
            case COMPUTE_REDUCE_MAX:  mpi_op = MPI_MAX; break;
            default:                  mpi_op = MPI_SUM; break;
        }

        MPI_Allreduce(send_data, recv_data, (int)count, mpi_type, mpi_op, ctx->comm);
    } else
#endif
    {
        // Single node: just copy
        size_t elem_size = compute_dtype_size(dtype);
        memcpy(recv_data, send_data, count * elem_size);
        (void)backend;
        (void)op;
    }
    return COMPUTE_SUCCESS;
}

static ComputeResult cpu_scatter(ComputeBackend* backend,
                                  const void* send_data, void* recv_data,
                                  size_t count, ComputeDataType dtype, int root) {
#if COMPUTE_HAS_MPI
    CPUBackendContext* ctx = (CPUBackendContext*)backend;
    if (ctx && ctx->num_nodes > 1) {
        MPI_Datatype mpi_type;
        switch (dtype) {
            case COMPUTE_DTYPE_FLOAT32:   mpi_type = MPI_FLOAT; break;
            case COMPUTE_DTYPE_FLOAT64:   mpi_type = MPI_DOUBLE; break;
            default:                      mpi_type = MPI_BYTE; break;
        }
        MPI_Scatter(send_data, (int)count, mpi_type,
                    recv_data, (int)count, mpi_type, root, ctx->comm);
    } else
#endif
    {
        size_t elem_size = compute_dtype_size(dtype);
        memcpy(recv_data, send_data, count * elem_size);
        (void)backend;
        (void)root;
    }
    return COMPUTE_SUCCESS;
}

static ComputeResult cpu_gather(ComputeBackend* backend,
                                 const void* send_data, void* recv_data,
                                 size_t count, ComputeDataType dtype, int root) {
#if COMPUTE_HAS_MPI
    CPUBackendContext* ctx = (CPUBackendContext*)backend;
    if (ctx && ctx->num_nodes > 1) {
        MPI_Datatype mpi_type;
        switch (dtype) {
            case COMPUTE_DTYPE_FLOAT32:   mpi_type = MPI_FLOAT; break;
            case COMPUTE_DTYPE_FLOAT64:   mpi_type = MPI_DOUBLE; break;
            default:                      mpi_type = MPI_BYTE; break;
        }
        MPI_Gather(send_data, (int)count, mpi_type,
                   recv_data, (int)count, mpi_type, root, ctx->comm);
    } else
#endif
    {
        size_t elem_size = compute_dtype_size(dtype);
        memcpy(recv_data, send_data, count * elem_size);
        (void)backend;
        (void)root;
    }
    return COMPUTE_SUCCESS;
}

static ComputeResult cpu_allgather(ComputeBackend* backend,
                                    const void* send_data, void* recv_data,
                                    size_t count, ComputeDataType dtype) {
#if COMPUTE_HAS_MPI
    CPUBackendContext* ctx = (CPUBackendContext*)backend;
    if (ctx && ctx->num_nodes > 1) {
        MPI_Datatype mpi_type;
        switch (dtype) {
            case COMPUTE_DTYPE_FLOAT32:   mpi_type = MPI_FLOAT; break;
            case COMPUTE_DTYPE_FLOAT64:   mpi_type = MPI_DOUBLE; break;
            default:                      mpi_type = MPI_BYTE; break;
        }
        MPI_Allgather(send_data, (int)count, mpi_type,
                      recv_data, (int)count, mpi_type, ctx->comm);
    } else
#endif
    {
        size_t elem_size = compute_dtype_size(dtype);
        memcpy(recv_data, send_data, count * elem_size);
        (void)backend;
    }
    return COMPUTE_SUCCESS;
}

static ComputeResult cpu_reduce_scatter(ComputeBackend* backend,
                                         const void* send_data, void* recv_data,
                                         size_t count, ComputeDataType dtype,
                                         ComputeReduceOp op) {
#if COMPUTE_HAS_MPI
    CPUBackendContext* ctx = (CPUBackendContext*)backend;
    if (ctx && ctx->num_nodes > 1) {
        MPI_Datatype mpi_type;
        MPI_Op mpi_op;

        switch (dtype) {
            case COMPUTE_DTYPE_FLOAT32:   mpi_type = MPI_FLOAT; break;
            case COMPUTE_DTYPE_FLOAT64:   mpi_type = MPI_DOUBLE; break;
            default:                      mpi_type = MPI_BYTE; break;
        }

        switch (op) {
            case COMPUTE_REDUCE_SUM:  mpi_op = MPI_SUM; break;
            case COMPUTE_REDUCE_PROD: mpi_op = MPI_PROD; break;
            case COMPUTE_REDUCE_MIN:  mpi_op = MPI_MIN; break;
            case COMPUTE_REDUCE_MAX:  mpi_op = MPI_MAX; break;
            default:                  mpi_op = MPI_SUM; break;
        }

        int* recvcounts = calloc(ctx->num_nodes, sizeof(int));
        for (int i = 0; i < ctx->num_nodes; i++) {
            recvcounts[i] = (int)count;
        }
        MPI_Reduce_scatter(send_data, recv_data, recvcounts, mpi_type, mpi_op, ctx->comm);
        free(recvcounts);
    } else
#endif
    {
        size_t elem_size = compute_dtype_size(dtype);
        memcpy(recv_data, send_data, count * elem_size);
        (void)backend;
        (void)op;
    }
    return COMPUTE_SUCCESS;
}

// ============================================================================
// Execution & Scheduling
// ============================================================================

static ComputeResult cpu_execute(ComputeBackend* backend,
                                  const ComputeQuantumOp* op,
                                  const ComputeExecutionPlan* plan,
                                  ComputeStream* stream) {
    (void)stream;

    if (!backend || !op) {
        return COMPUTE_ERROR_INVALID_ARGUMENT;
    }

    CPUBackendContext* ctx = (CPUBackendContext*)backend;
    ComputeResult result = COMPUTE_SUCCESS;

    // Execute based on operation type
    switch (op->type) {
        case QUANTUM_OP_UNITARY:
            result = cpu_quantum_unitary(backend,
                                         op->output_data, op->output_size,
                                         op->parameters, op->param_size,
                                         stream);
            break;

        case QUANTUM_OP_NORMALIZE:
            result = cpu_quantum_normalize(backend,
                                           op->output_data, op->output_size,
                                           stream);
            break;

        case QUANTUM_OP_TENSOR_CONTRACT:
            if (op->num_dims >= 3) {
                result = cpu_quantum_tensor_contract(backend,
                                                     op->output_data,
                                                     op->input_data,
                                                     op->parameters,
                                                     op->dims[0], op->dims[1], op->dims[2],
                                                     stream);
            }
            break;

        case QUANTUM_OP_GRADIENT:
            result = cpu_quantum_gradient(backend,
                                          op->output_data,
                                          op->input_data,
                                          op->parameters,
                                          op->input_size,
                                          stream);
            break;

        case QUANTUM_OP_INNER_PRODUCT:
            result = cpu_quantum_inner_product(backend,
                                               op->output_data,
                                               op->input_data,
                                               op->parameters,
                                               op->input_size,
                                               stream);
            break;

        case QUANTUM_OP_EXPECTATION:
            result = cpu_quantum_expectation(backend,
                                             op->output_data,
                                             op->input_data,
                                             op->parameters,
                                             op->input_size,
                                             stream);
            break;

        default:
            result = COMPUTE_ERROR_NOT_IMPLEMENTED;
            break;
    }

    // Update metrics
    ctx->metrics.operations_per_second += 1.0;
    (void)plan;

    return result;
}

static ComputeExecutionPlan* cpu_create_plan(ComputeBackend* backend,
                                              const ComputeQuantumOp* op) {
    CPUBackendContext* ctx = (CPUBackendContext*)backend;
    if (!ctx || !op) return NULL;

    ComputeExecutionPlan* plan = calloc(1, sizeof(ComputeExecutionPlan));
    if (!plan) return NULL;

    // Simple partitioning across nodes
    plan->num_partitions = ctx->num_nodes;
    plan->partition_size = op->input_size / ctx->num_nodes;

    plan->node_assignments = calloc(plan->num_partitions, sizeof(int));
    plan->offsets = calloc(plan->num_partitions, sizeof(size_t));
    plan->sizes = calloc(plan->num_partitions, sizeof(size_t));

    if (!plan->node_assignments || !plan->offsets || !plan->sizes) {
        free(plan->node_assignments);
        free(plan->offsets);
        free(plan->sizes);
        free(plan);
        return NULL;
    }

    for (size_t i = 0; i < plan->num_partitions; i++) {
        plan->node_assignments[i] = (int)i;
        plan->offsets[i] = i * plan->partition_size;
        plan->sizes[i] = (i == plan->num_partitions - 1) ?
                         (op->input_size - i * plan->partition_size) :
                         plan->partition_size;
    }

    return plan;
}

static void cpu_destroy_plan(ComputeBackend* backend, ComputeExecutionPlan* plan) {
    (void)backend;
    if (!plan) return;

    free(plan->node_assignments);
    free(plan->offsets);
    free(plan->sizes);
    free(plan->send_targets);
    free(plan->recv_sources);
    free(plan->workspace);
    free(plan);
}

// ============================================================================
// Performance Monitoring
// ============================================================================

static ComputeResult cpu_get_metrics(ComputeBackend* backend, ComputeMetrics* metrics) {
    CPUBackendContext* ctx = (CPUBackendContext*)backend;
    if (!ctx || !metrics) return COMPUTE_ERROR_INVALID_ARGUMENT;

    *metrics = ctx->metrics;
    metrics->peak_memory_bytes = ctx->peak_allocated;
    metrics->current_memory_bytes = ctx->total_allocated;

    return COMPUTE_SUCCESS;
}

static ComputeResult cpu_reset_metrics(ComputeBackend* backend) {
    CPUBackendContext* ctx = (CPUBackendContext*)backend;
    if (!ctx) return COMPUTE_ERROR_INVALID_ARGUMENT;

    memset(&ctx->metrics, 0, sizeof(ComputeMetrics));
    return COMPUTE_SUCCESS;
}

// ============================================================================
// Backend Registration
// ============================================================================

static const ComputeBackendOps cpu_ops = {
    // Lifecycle
    .init = cpu_init,
    .cleanup = cpu_cleanup,
    .probe = cpu_probe,
    .get_capabilities = cpu_get_capabilities,

    // Memory
    .alloc = cpu_alloc,
    .free = cpu_free,
    .memcpy = cpu_memcpy,
    .memset = cpu_memset,

    // Streams
    .create_stream = cpu_create_stream,
    .destroy_stream = cpu_destroy_stream,
    .synchronize_stream = cpu_synchronize_stream,
    .create_event = cpu_create_event,
    .destroy_event = cpu_destroy_event,
    .record_event = cpu_record_event,
    .wait_event = cpu_wait_event,

    // Quantum operations
    .quantum_unitary = cpu_quantum_unitary,
    .quantum_normalize = cpu_quantum_normalize,
    .quantum_tensor_contract = cpu_quantum_tensor_contract,
    .quantum_gradient = cpu_quantum_gradient,
    .quantum_inner_product = cpu_quantum_inner_product,
    .quantum_expectation = cpu_quantum_expectation,

    // Collective communication
    .barrier = cpu_barrier,
    .broadcast = cpu_broadcast,
    .allreduce = cpu_allreduce,
    .scatter = cpu_scatter,
    .gather = cpu_gather,
    .allgather = cpu_allgather,
    .reduce_scatter = cpu_reduce_scatter,

    // Execution
    .execute = cpu_execute,
    .create_plan = cpu_create_plan,
    .destroy_plan = cpu_destroy_plan,

    // Metrics
    .get_metrics = cpu_get_metrics,
    .reset_metrics = cpu_reset_metrics,
};

// Register the CPU backend at library load time
COMPUTE_REGISTER_BACKEND(COMPUTE_BACKEND_CPU, "CPU (OpenMP+SIMD)", "1.0.0", 10, cpu_ops)
