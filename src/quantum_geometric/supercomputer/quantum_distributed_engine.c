/**
 * @file quantum_distributed_engine.c
 * @brief Distributed quantum computing engine for multi-node clusters
 *
 * This file provides the public API for distributed quantum computing.
 * Backend implementations are now modular and located in:
 * - backends/compute_cpu.c     (CPU + OpenMP + SIMD)
 * - backends/compute_metal.mm  (Apple Metal + Accelerate)
 * - backends/compute_opencl.c  (Cross-platform OpenCL)
 * - backends/compute_cuda.c    (NVIDIA CUDA + NCCL) [future]
 *
 * The vtable-based backend system in compute_backend.h provides:
 * - Automatic backend detection and selection
 * - Runtime backend switching
 * - Unified memory management
 * - Cross-platform collective communication
 *
 * Legacy backend implementations are preserved below for compatibility
 * but new code should use the ComputeEngine API.
 */

#include "quantum_geometric/supercomputer/quantum_distributed_engine.h"
#include "quantum_geometric/supercomputer/compute_backend.h"
#include "quantum_geometric/supercomputer/compute_simd.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// Platform-specific SIMD
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define USE_AVX2 1
#elif defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#define USE_NEON 1
#endif

// ============================================================================
// Backend Detection and Configuration
// ============================================================================

#if defined(USE_CUDA) && defined(USE_NCCL) && defined(USE_MPI)
#define BACKEND_CUDA_AVAILABLE 1
#else
#define BACKEND_CUDA_AVAILABLE 0
#endif

#if defined(__APPLE__)
#define BACKEND_METAL_AVAILABLE 1
#include <dispatch/dispatch.h>
#include <Accelerate/Accelerate.h>
#else
#define BACKEND_METAL_AVAILABLE 0
#endif

#if defined(USE_OPENCL)
#define BACKEND_OPENCL_AVAILABLE 1
#else
#define BACKEND_OPENCL_AVAILABLE 0
#endif

#if defined(USE_MPI)
#define BACKEND_MPI_AVAILABLE 1
#else
#define BACKEND_MPI_AVAILABLE 0
#endif

// Hardware parameters
#define MAX_NODES 1024
#define MAX_GPUS_PER_NODE 8
#define NETWORK_BANDWIDTH 200  // GB/s
#define MEMORY_PER_NODE (1ULL << 40)  // 1TB
#define DEFAULT_GPU_BUFFER_SIZE (256 * 1024 * 1024)  // 256MB
#define DEFAULT_HOST_BUFFER_SIZE (128 * 1024 * 1024)  // 128MB

// Performance parameters
#define MIN_BATCH_SIZE 1024
#define MAX_BATCH_SIZE 16384
#define CACHE_LINE 128
#define VECTOR_WIDTH 8  // For AVX-256

// Quantum operation types are defined in compute_types.h
// Use the definitions from there for consistency

// ============================================================================
// Generic Context Structure
// ============================================================================

struct DistributedContext {
    DistributedBackendType backend_type;
    void* backend_data;
    int num_nodes;
    int node_rank;
};

// ============================================================================
// Forward Declarations
// ============================================================================

static void quantum_unitary_transform(float* data, size_t n, const float* params, size_t param_size);
static void quantum_normalize_state(float* data, size_t n);
static void quantum_tensor_contract(float* out, const float* a, const float* b, size_t m, size_t n, size_t k);
static void quantum_compute_gradient(float* grad, const float* state, const float* target, size_t n);

// ============================================================================
// SIMD-Optimized Quantum Operations
// ============================================================================

#if USE_AVX2

// AVX2-optimized complex multiply-accumulate
static inline void avx2_complex_macc(float* out, const float* a, const float* b, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va_re = _mm256_loadu_ps(a + i);
        __m256 va_im = _mm256_loadu_ps(a + i + 8);
        __m256 vb_re = _mm256_loadu_ps(b + i);
        __m256 vb_im = _mm256_loadu_ps(b + i + 8);

        // (a_re + i*a_im) * (b_re + i*b_im) = (a_re*b_re - a_im*b_im) + i*(a_re*b_im + a_im*b_re)
        __m256 out_re = _mm256_sub_ps(_mm256_mul_ps(va_re, vb_re), _mm256_mul_ps(va_im, vb_im));
        __m256 out_im = _mm256_add_ps(_mm256_mul_ps(va_re, vb_im), _mm256_mul_ps(va_im, vb_re));

        __m256 prev_re = _mm256_loadu_ps(out + i);
        __m256 prev_im = _mm256_loadu_ps(out + i + 8);

        _mm256_storeu_ps(out + i, _mm256_add_ps(prev_re, out_re));
        _mm256_storeu_ps(out + i + 8, _mm256_add_ps(prev_im, out_im));
    }

    // Scalar fallback for remaining elements
    for (; i < n; i++) {
        float a_re = a[i], a_im = a[i + n];
        float b_re = b[i], b_im = b[i + n];
        out[i] += a_re * b_re - a_im * b_im;
        out[i + n] += a_re * b_im + a_im * b_re;
    }
}

// AVX2-optimized vector norm
static inline float avx2_vector_norm(const float* data, size_t n) {
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;

    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(data + i);
        sum = _mm256_fmadd_ps(v, v, sum);
    }

    // Horizontal sum
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float result = _mm_cvtss_f32(sum128);

    // Scalar fallback
    for (; i < n; i++) {
        result += data[i] * data[i];
    }

    return sqrtf(result);
}

// AVX2-optimized vector scale
static inline void avx2_vector_scale(float* data, size_t n, float scale) {
    __m256 vscale = _mm256_set1_ps(scale);
    size_t i = 0;

    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(data + i);
        _mm256_storeu_ps(data + i, _mm256_mul_ps(v, vscale));
    }

    for (; i < n; i++) {
        data[i] *= scale;
    }
}

#elif USE_NEON

// NEON-optimized complex multiply-accumulate
static inline void neon_complex_macc(float* out, const float* a, const float* b, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va_re = vld1q_f32(a + i);
        float32x4_t va_im = vld1q_f32(a + i + n);
        float32x4_t vb_re = vld1q_f32(b + i);
        float32x4_t vb_im = vld1q_f32(b + i + n);

        float32x4_t out_re = vsubq_f32(vmulq_f32(va_re, vb_re), vmulq_f32(va_im, vb_im));
        float32x4_t out_im = vaddq_f32(vmulq_f32(va_re, vb_im), vmulq_f32(va_im, vb_re));

        float32x4_t prev_re = vld1q_f32(out + i);
        float32x4_t prev_im = vld1q_f32(out + i + n);

        vst1q_f32(out + i, vaddq_f32(prev_re, out_re));
        vst1q_f32(out + i + n, vaddq_f32(prev_im, out_im));
    }

    for (; i < n; i++) {
        float a_re = a[i], a_im = a[i + n];
        float b_re = b[i], b_im = b[i + n];
        out[i] += a_re * b_re - a_im * b_im;
        out[i + n] += a_re * b_im + a_im * b_re;
    }
}

// NEON-optimized vector norm
static inline float neon_vector_norm(const float* data, size_t n) {
    float32x4_t sum = vdupq_n_f32(0.0f);
    size_t i = 0;

    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        sum = vfmaq_f32(sum, v, v);
    }

    float result = vaddvq_f32(sum);

    for (; i < n; i++) {
        result += data[i] * data[i];
    }

    return sqrtf(result);
}

// NEON-optimized vector scale
static inline void neon_vector_scale(float* data, size_t n, float scale) {
    float32x4_t vscale = vdupq_n_f32(scale);
    size_t i = 0;

    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        vst1q_f32(data + i, vmulq_f32(v, vscale));
    }

    for (; i < n; i++) {
        data[i] *= scale;
    }
}

#else

// Scalar fallback implementations
static inline void scalar_complex_macc(float* out, const float* a, const float* b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float a_re = a[i], a_im = a[i + n];
        float b_re = b[i], b_im = b[i + n];
        out[i] += a_re * b_re - a_im * b_im;
        out[i + n] += a_re * b_im + a_im * b_re;
    }
}

static inline float scalar_vector_norm(const float* data, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += data[i] * data[i];
    }
    return sqrtf(sum);
}

static inline void scalar_vector_scale(float* data, size_t n, float scale) {
    for (size_t i = 0; i < n; i++) {
        data[i] *= scale;
    }
}

#endif

// SIMD functions are now provided by compute_simd.h
// The following legacy wrappers use the new modular implementations
// simd_complex_macc, simd_vector_norm, simd_vector_scale are declared in compute_simd.h

// ============================================================================
// Quantum Operations Implementation
// ============================================================================

// Apply unitary transformation to quantum state
static void quantum_unitary_transform(float* data, size_t n, const float* params, size_t param_size) {
    if (!data || !params || n == 0 || param_size == 0) return;

    // Quantum state dimension (n = 2 * state_dim for complex numbers)
    size_t state_dim = n / 2;

    // Extract rotation angles from parameters
    float theta = params[0];
    float phi = (param_size > 1) ? params[1] : 0.0f;
    float lambda = (param_size > 2) ? params[2] : 0.0f;

    // Compute rotation matrix elements (general U3 gate)
    float cos_half = cosf(theta / 2.0f);
    float sin_half = sinf(theta / 2.0f);

    float u00_re = cos_half;
    float u00_im = 0.0f;
    float u01_re = -sin_half * cosf(lambda);
    float u01_im = -sin_half * sinf(lambda);
    float u10_re = sin_half * cosf(phi);
    float u10_im = sin_half * sinf(phi);
    float u11_re = cos_half * cosf(phi + lambda);
    float u11_im = cos_half * sinf(phi + lambda);

    // Apply transformation in parallel
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < state_dim; i += 2) {
        if (i + 1 >= state_dim) continue;

        // Load state amplitudes
        float s0_re = data[i];
        float s0_im = data[i + state_dim];
        float s1_re = data[i + 1];
        float s1_im = data[i + 1 + state_dim];

        // Compute new amplitudes
        float new_s0_re = u00_re * s0_re - u00_im * s0_im + u01_re * s1_re - u01_im * s1_im;
        float new_s0_im = u00_re * s0_im + u00_im * s0_re + u01_re * s1_im + u01_im * s1_re;
        float new_s1_re = u10_re * s0_re - u10_im * s0_im + u11_re * s1_re - u11_im * s1_im;
        float new_s1_im = u10_re * s0_im + u10_im * s0_re + u11_re * s1_im + u11_im * s1_re;

        // Store results
        data[i] = new_s0_re;
        data[i + state_dim] = new_s0_im;
        data[i + 1] = new_s1_re;
        data[i + 1 + state_dim] = new_s1_im;
    }
}

// Normalize quantum state
static void quantum_normalize_state(float* data, size_t n) {
    if (!data || n == 0) return;

    float norm = simd_vector_norm(data, n);
    if (norm > 1e-10f) {
        float inv_norm = 1.0f / norm;
        simd_vector_scale(data, n, inv_norm);
    }
}

// Tensor contraction for quantum operations
static void quantum_tensor_contract(float* out, const float* a, const float* b,
                                     size_t m, size_t n, size_t k) {
    if (!out || !a || !b) return;

#if BACKEND_METAL_AVAILABLE
    // Use Accelerate framework's BLAS for matrix multiply
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)m, (int)n, (int)k,
                1.0f, a, (int)k, b, (int)n,
                0.0f, out, (int)n);
#else
    // Parallel blocked matrix multiplication
    const size_t block_size = 64;

    memset(out, 0, m * n * sizeof(float));

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i0 = 0; i0 < m; i0 += block_size) {
        for (size_t j0 = 0; j0 < n; j0 += block_size) {
            for (size_t k0 = 0; k0 < k; k0 += block_size) {
                size_t i_end = (i0 + block_size < m) ? i0 + block_size : m;
                size_t j_end = (j0 + block_size < n) ? j0 + block_size : n;
                size_t k_end = (k0 + block_size < k) ? k0 + block_size : k;

                for (size_t i = i0; i < i_end; i++) {
                    for (size_t kk = k0; kk < k_end; kk++) {
                        float aik = a[i * k + kk];
                        for (size_t j = j0; j < j_end; j++) {
                            out[i * n + j] += aik * b[kk * n + j];
                        }
                    }
                }
            }
        }
    }
#endif
}

// Compute gradient for quantum optimization
static void quantum_compute_gradient(float* grad, const float* state, const float* target, size_t n) {
    if (!grad || !state || !target) return;

    // Gradient of MSE loss: 2 * (state - target)
    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; i++) {
        grad[i] = 2.0f * (state[i] - target[i]);
    }
}

// ============================================================================
// CUDA + NCCL Backend Implementation
// ============================================================================

#if BACKEND_CUDA_AVAILABLE

#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>

typedef struct CudaNodeContext {
    int node_id;
    int local_rank;
    int global_rank;
    int num_gpus;
    cudaStream_t* streams;
    ncclComm_t* nccl_comms;
    ncclUniqueId nccl_id;
    void** gpu_buffers;
    void** grad_buffers;
    void* host_buffer;
    size_t buffer_size;
    MPI_Comm node_comm;
    MPI_Comm global_comm;
} CudaNodeContext;

typedef struct CudaDistributedContext {
    int num_nodes;
    CudaNodeContext** nodes;
    ncclUniqueId nccl_id;
    MPI_Comm world_comm;
    size_t total_work_items;
    size_t items_per_node;
    PerformanceMonitor* monitor;
} CudaDistributedContext;

// CUDA kernel declarations (implemented in .cu file)
extern void cuda_quantum_unitary_kernel(float* data, size_t n, const float* params,
                                         size_t param_size, cudaStream_t stream);
extern void cuda_quantum_normalize_kernel(float* data, size_t n, cudaStream_t stream);
extern void cuda_tensor_contract_kernel(float* out, const float* a, const float* b,
                                         size_t m, size_t n, size_t k, cudaStream_t stream);

static void cuda_init_node_contexts(CudaDistributedContext* ctx, const DistributedConfig* config);
static void cuda_init_gpu_resources(CudaNodeContext* node, const DistributedConfig* config);
static void cuda_init_memory_resources(CudaNodeContext* node, const DistributedConfig* config);
static bool cuda_distribute_data(CudaDistributedContext* ctx, const QuantumOperation* op, const ExecutionPlan* plan);
static bool cuda_execute_operation(CudaDistributedContext* ctx, const QuantumOperation* op, const ExecutionPlan* plan);
static bool cuda_gather_results(CudaDistributedContext* ctx, const ExecutionPlan* plan);

static DistributedContext* cuda_init_distributed_engine(const DistributedConfig* config) {
    CudaDistributedContext* cuda_ctx = calloc(1, sizeof(CudaDistributedContext));
    if (!cuda_ctx) return NULL;

    int provided;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE) {
        fprintf(stderr, "Warning: MPI does not support MPI_THREAD_MULTIPLE\n");
    }

    MPI_Comm_dup(MPI_COMM_WORLD, &cuda_ctx->world_comm);
    MPI_Comm_size(cuda_ctx->world_comm, &cuda_ctx->num_nodes);

    int rank;
    MPI_Comm_rank(cuda_ctx->world_comm, &rank);
    if (rank == 0) {
        ncclGetUniqueId(&cuda_ctx->nccl_id);
    }
    MPI_Bcast(&cuda_ctx->nccl_id, sizeof(ncclUniqueId), MPI_BYTE, 0, cuda_ctx->world_comm);

    cuda_ctx->nodes = calloc(cuda_ctx->num_nodes, sizeof(CudaNodeContext*));
    if (!cuda_ctx->nodes) {
        MPI_Finalize();
        free(cuda_ctx);
        return NULL;
    }

    cuda_init_node_contexts(cuda_ctx, config);

    DistributedContext* ctx = calloc(1, sizeof(DistributedContext));
    ctx->backend_type = DISTRIBUTED_BACKEND_CUDA;
    ctx->backend_data = cuda_ctx;
    ctx->num_nodes = cuda_ctx->num_nodes;
    ctx->node_rank = rank;

    return ctx;
}

static void cuda_init_node_contexts(CudaDistributedContext* ctx, const DistributedConfig* config) {
    int global_rank;
    MPI_Comm_rank(ctx->world_comm, &global_rank);

    MPI_Comm node_comm;
    MPI_Comm_split_type(ctx->world_comm, MPI_COMM_TYPE_SHARED, global_rank, MPI_INFO_NULL, &node_comm);

    CudaNodeContext* node = calloc(1, sizeof(CudaNodeContext));
    node->global_rank = global_rank;
    node->node_comm = node_comm;
    node->nccl_id = ctx->nccl_id;

    MPI_Comm_rank(node_comm, &node->local_rank);

    cudaGetDeviceCount(&node->num_gpus);
    if (config->gpus_per_node > 0 && node->num_gpus > config->gpus_per_node) {
        node->num_gpus = config->gpus_per_node;
    }

    cuda_init_gpu_resources(node, config);
    cuda_init_memory_resources(node, config);

    ctx->nodes[global_rank] = node;
    MPI_Barrier(ctx->world_comm);
}

static void cuda_init_gpu_resources(CudaNodeContext* node, const DistributedConfig* config) {
    (void)config;
    if (node->num_gpus <= 0) return;

    node->streams = calloc(node->num_gpus, sizeof(cudaStream_t));
    node->nccl_comms = calloc(node->num_gpus, sizeof(ncclComm_t));

    for (int i = 0; i < node->num_gpus; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&node->streams[i]);
        ncclCommInitRank(&node->nccl_comms[i], node->num_gpus, node->nccl_id, node->local_rank);
    }
}

static void cuda_init_memory_resources(CudaNodeContext* node, const DistributedConfig* config) {
    if (node->num_gpus <= 0) return;

    node->gpu_buffers = calloc(node->num_gpus, sizeof(void*));
    node->grad_buffers = calloc(node->num_gpus, sizeof(void*));
    size_t buffer_size = config->gpu_buffer_size > 0 ? config->gpu_buffer_size : DEFAULT_GPU_BUFFER_SIZE;

    for (int i = 0; i < node->num_gpus; i++) {
        cudaSetDevice(i);
        cudaMalloc(&node->gpu_buffers[i], buffer_size);
        cudaMalloc(&node->grad_buffers[i], buffer_size);
        cudaMemset(node->gpu_buffers[i], 0, buffer_size);
        cudaMemset(node->grad_buffers[i], 0, buffer_size);
    }

    size_t host_size = config->host_buffer_size > 0 ? config->host_buffer_size : DEFAULT_HOST_BUFFER_SIZE;
    cudaMallocHost(&node->host_buffer, host_size);
    node->buffer_size = buffer_size;
}

static int cuda_execute_distributed_operation(DistributedContext* ctx, const QuantumOperation* op) {
    CudaDistributedContext* cuda_ctx = (CudaDistributedContext*)ctx->backend_data;

    ExecutionPlan* plan = create_execution_plan(ctx, op);
    if (!plan) return -1;

    if (!cuda_distribute_data(cuda_ctx, op, plan)) {
        cleanup_execution(plan);
        return -1;
    }

    if (!cuda_execute_operation(cuda_ctx, op, plan)) {
        cleanup_execution(plan);
        return -1;
    }

    if (!cuda_gather_results(cuda_ctx, plan)) {
        cleanup_execution(plan);
        return -1;
    }

    if (cuda_ctx->monitor) {
        update_performance_metrics(cuda_ctx->monitor, op, plan);
    }

    cleanup_execution(plan);
    return 0;
}

static bool cuda_distribute_data(CudaDistributedContext* ctx, const QuantumOperation* op, const ExecutionPlan* plan) {
    (void)plan;
    size_t total_size = calculate_data_size(op);
    size_t node_size = total_size / ctx->num_nodes;

    for (int i = 0; i < ctx->num_nodes; i++) {
        CudaNodeContext* node = ctx->nodes[i];
        if (!node) continue;

        ncclGroupStart();
        for (int j = 0; j < node->num_gpus; j++) {
            ncclAllGather((const char*)op->data + i * node_size, node->gpu_buffers[j],
                          node_size, ncclFloat32, node->nccl_comms[j], node->streams[j]);
        }
        ncclGroupEnd();

        for (int j = 0; j < node->num_gpus; j++) {
            cudaStreamSynchronize(node->streams[j]);
        }
    }
    return true;
}

static bool cuda_execute_operation(CudaDistributedContext* ctx, const QuantumOperation* op, const ExecutionPlan* plan) {
    for (int i = 0; i < ctx->num_nodes; i++) {
        CudaNodeContext* node = ctx->nodes[i];
        if (!node) continue;

        for (int j = 0; j < node->num_gpus; j++) {
            cudaSetDevice(j);

            switch (op->operation_type) {
                case QUANTUM_OP_UNITARY:
                    cuda_quantum_unitary_kernel(node->gpu_buffers[j], plan->num_work_items,
                                                op->parameters, op->param_size, node->streams[j]);
                    break;
                case QUANTUM_OP_NORMALIZE:
                    cuda_quantum_normalize_kernel(node->gpu_buffers[j], plan->num_work_items,
                                                   node->streams[j]);
                    break;
                case QUANTUM_OP_TENSOR_CONTRACT:
                    cuda_tensor_contract_kernel(node->grad_buffers[j], node->gpu_buffers[j],
                                                 op->parameters, plan->num_work_items,
                                                 plan->num_work_items, plan->num_work_items,
                                                 node->streams[j]);
                    break;
                default:
                    break;
            }
        }

        for (int j = 0; j < node->num_gpus; j++) {
            cudaStreamSynchronize(node->streams[j]);
        }
    }

    MPI_Barrier(ctx->world_comm);
    return true;
}

static bool cuda_gather_results(CudaDistributedContext* ctx, const ExecutionPlan* plan) {
    (void)plan;

    for (int i = 0; i < ctx->num_nodes; i++) {
        CudaNodeContext* node = ctx->nodes[i];
        if (!node) continue;

        ncclGroupStart();
        for (int j = 0; j < node->num_gpus; j++) {
            ncclAllReduce(node->gpu_buffers[j], node->gpu_buffers[j],
                          plan->num_work_items, ncclFloat32, ncclSum,
                          node->nccl_comms[j], node->streams[j]);
        }
        ncclGroupEnd();

        for (int j = 0; j < node->num_gpus; j++) {
            cudaStreamSynchronize(node->streams[j]);
        }
    }

    MPI_Barrier(ctx->world_comm);
    return true;
}

static void cuda_cleanup_distributed_engine(DistributedContext* ctx) {
    if (!ctx || !ctx->backend_data) return;
    CudaDistributedContext* cuda_ctx = (CudaDistributedContext*)ctx->backend_data;

    for (int i = 0; i < cuda_ctx->num_nodes; i++) {
        CudaNodeContext* node = cuda_ctx->nodes[i];
        if (!node) continue;

        for (int j = 0; j < node->num_gpus; j++) {
            if (node->nccl_comms) ncclCommDestroy(node->nccl_comms[j]);
            if (node->streams) cudaStreamDestroy(node->streams[j]);
            if (node->gpu_buffers) cudaFree(node->gpu_buffers[j]);
            if (node->grad_buffers) cudaFree(node->grad_buffers[j]);
        }

        if (node->host_buffer) cudaFreeHost(node->host_buffer);
        free(node->streams);
        free(node->nccl_comms);
        free(node->gpu_buffers);
        free(node->grad_buffers);
        MPI_Comm_free(&node->node_comm);
        free(node);
    }

    free(cuda_ctx->nodes);
    MPI_Comm_free(&cuda_ctx->world_comm);
    MPI_Finalize();
    free(cuda_ctx);
}

#endif // BACKEND_CUDA_AVAILABLE

// ============================================================================
// Metal Backend Implementation (Apple Silicon)
// ============================================================================

#if BACKEND_METAL_AVAILABLE

typedef struct MetalNodeContext {
    int node_id;
    int local_rank;
    int global_rank;
    dispatch_queue_t compute_queue;
    dispatch_group_t dispatch_group;
    void** device_buffers;
    void** grad_buffers;
    void* host_buffer;
    size_t buffer_size;
    size_t num_buffers;
    int num_threads;
} MetalNodeContext;

typedef struct MetalDistributedContext {
    int num_nodes;
    MetalNodeContext** nodes;
    MetalNodeContext* local_node;
    size_t total_work_items;
    PerformanceMonitor* monitor;
    pthread_mutex_t sync_mutex;
    pthread_cond_t sync_cond;
    volatile int sync_counter;
} MetalDistributedContext;

static DistributedContext* metal_init_distributed_engine(const DistributedConfig* config) {
    MetalDistributedContext* metal_ctx = calloc(1, sizeof(MetalDistributedContext));
    if (!metal_ctx) return NULL;

    pthread_mutex_init(&metal_ctx->sync_mutex, NULL);
    pthread_cond_init(&metal_ctx->sync_cond, NULL);
    metal_ctx->num_nodes = config->num_nodes > 0 ? config->num_nodes : 1;
    metal_ctx->sync_counter = 0;

    metal_ctx->nodes = calloc(metal_ctx->num_nodes, sizeof(MetalNodeContext*));
    if (!metal_ctx->nodes) {
        free(metal_ctx);
        return NULL;
    }

    MetalNodeContext* node = calloc(1, sizeof(MetalNodeContext));
    node->node_id = 0;
    node->local_rank = 0;
    node->global_rank = 0;

    // Create high-priority concurrent dispatch queue for compute operations
    dispatch_queue_attr_t attr = dispatch_queue_attr_make_with_qos_class(
        DISPATCH_QUEUE_CONCURRENT, QOS_CLASS_USER_INITIATED, -1);
    node->compute_queue = dispatch_queue_create("com.qgt.metal.distributed.compute", attr);
    node->dispatch_group = dispatch_group_create();

    // Get number of performance cores
    node->num_threads = (int)sysconf(_SC_NPROCESSORS_ONLN);

    size_t buffer_size = config->gpu_buffer_size > 0 ? config->gpu_buffer_size : DEFAULT_GPU_BUFFER_SIZE;
    node->buffer_size = buffer_size;
    node->num_buffers = 2;  // Double buffering for async operations

    node->device_buffers = calloc(node->num_buffers, sizeof(void*));
    node->grad_buffers = calloc(node->num_buffers, sizeof(void*));

    for (size_t i = 0; i < node->num_buffers; i++) {
        // Page-aligned allocation for better performance
        posix_memalign(&node->device_buffers[i], 16384, buffer_size);
        posix_memalign(&node->grad_buffers[i], 16384, buffer_size);
        memset(node->device_buffers[i], 0, buffer_size);
        memset(node->grad_buffers[i], 0, buffer_size);
    }

    size_t host_size = config->host_buffer_size > 0 ? config->host_buffer_size : DEFAULT_HOST_BUFFER_SIZE;
    posix_memalign(&node->host_buffer, CACHE_LINE, host_size);

    metal_ctx->nodes[0] = node;
    metal_ctx->local_node = node;

    DistributedContext* ctx = calloc(1, sizeof(DistributedContext));
    ctx->backend_type = DISTRIBUTED_BACKEND_METAL;
    ctx->backend_data = metal_ctx;
    ctx->num_nodes = metal_ctx->num_nodes;
    ctx->node_rank = 0;

    return ctx;
}

static int metal_execute_distributed_operation(DistributedContext* ctx, const QuantumOperation* op) {
    MetalDistributedContext* metal_ctx = (MetalDistributedContext*)ctx->backend_data;
    MetalNodeContext* node = metal_ctx->local_node;

    ExecutionPlan* plan = create_execution_plan(ctx, op);
    if (!plan) return -1;

    // Copy input data to device buffer
    size_t data_size = op->data_size;
    if (data_size > node->buffer_size) data_size = node->buffer_size;
    memcpy(node->device_buffers[0], op->data, data_size);

    size_t work_items = plan->num_work_items;
    float* data = (float*)node->device_buffers[0];
    float* grad_data = (float*)node->grad_buffers[0];

    // Execute quantum operation based on type using OpenMP for parallelism
    switch (op->operation_type) {
        case QUANTUM_OP_UNITARY: {
            // Parallel unitary transformation
            quantum_unitary_transform(data, work_items * 2,
                                      (const float*)op->parameters,
                                      op->param_size / sizeof(float));
            break;
        }

        case QUANTUM_OP_NORMALIZE: {
            // Two-pass normalization using SIMD
            quantum_normalize_state(data, work_items);
            break;
        }

        case QUANTUM_OP_TENSOR_CONTRACT: {
            // Use Accelerate framework for tensor contraction (matrix multiply)
            size_t dim = (size_t)sqrtf((float)work_items);
            if (dim * dim == work_items && op->parameters) {
                quantum_tensor_contract(grad_data, data, (const float*)op->parameters, dim, dim, dim);
                memcpy(data, grad_data, work_items * sizeof(float));
            }
            break;
        }

        case QUANTUM_OP_GRADIENT: {
            // Compute gradient in parallel
            if (op->parameters) {
                quantum_compute_gradient(grad_data, data, (const float*)op->parameters, work_items);
            }
            break;
        }

        default:
            break;
    }

    // Copy results back if needed
    if (op->data_size <= node->buffer_size) {
        memcpy(op->data, data, op->data_size);
    }

    if (metal_ctx->monitor) {
        update_performance_metrics(metal_ctx->monitor, op, plan);
    }

    cleanup_execution(plan);
    return 0;
}

static void metal_cleanup_distributed_engine(DistributedContext* ctx) {
    if (!ctx || !ctx->backend_data) return;
    MetalDistributedContext* metal_ctx = (MetalDistributedContext*)ctx->backend_data;

    for (int i = 0; i < metal_ctx->num_nodes; i++) {
        MetalNodeContext* node = metal_ctx->nodes[i];
        if (!node) continue;

        if (node->compute_queue) dispatch_release(node->compute_queue);
        if (node->dispatch_group) dispatch_release(node->dispatch_group);

        for (size_t j = 0; j < node->num_buffers; j++) {
            free(node->device_buffers[j]);
            free(node->grad_buffers[j]);
        }
        free(node->device_buffers);
        free(node->grad_buffers);
        free(node->host_buffer);
        free(node);
    }

    free(metal_ctx->nodes);
    pthread_mutex_destroy(&metal_ctx->sync_mutex);
    pthread_cond_destroy(&metal_ctx->sync_cond);
    free(metal_ctx);
}

#endif // BACKEND_METAL_AVAILABLE

// ============================================================================
// OpenCL Backend Implementation
// ============================================================================

#if BACKEND_OPENCL_AVAILABLE

#include <CL/cl.h>

// OpenCL kernel sources
static const char* opencl_quantum_kernels =
    "__kernel void quantum_unitary(__global float* data, const int n,\n"
    "                              __global const float* params, const int param_size) {\n"
    "    int i = get_global_id(0) * 2;\n"
    "    if (i + 1 >= n) return;\n"
    "    \n"
    "    float theta = params[0];\n"
    "    float phi = (param_size > 1) ? params[1] : 0.0f;\n"
    "    float lambda = (param_size > 2) ? params[2] : 0.0f;\n"
    "    \n"
    "    float cos_half = cos(theta / 2.0f);\n"
    "    float sin_half = sin(theta / 2.0f);\n"
    "    \n"
    "    float s0_re = data[i];\n"
    "    float s0_im = data[i + n];\n"
    "    float s1_re = data[i + 1];\n"
    "    float s1_im = data[i + 1 + n];\n"
    "    \n"
    "    float u00_re = cos_half;\n"
    "    float u01_re = -sin_half * cos(lambda);\n"
    "    float u01_im = -sin_half * sin(lambda);\n"
    "    float u10_re = sin_half * cos(phi);\n"
    "    float u10_im = sin_half * sin(phi);\n"
    "    float u11_re = cos_half * cos(phi + lambda);\n"
    "    float u11_im = cos_half * sin(phi + lambda);\n"
    "    \n"
    "    data[i] = u00_re * s0_re + u01_re * s1_re - u01_im * s1_im;\n"
    "    data[i + n] = u00_re * s0_im + u01_re * s1_im + u01_im * s1_re;\n"
    "    data[i + 1] = u10_re * s0_re - u10_im * s0_im + u11_re * s1_re - u11_im * s1_im;\n"
    "    data[i + 1 + n] = u10_re * s0_im + u10_im * s0_re + u11_re * s1_im + u11_im * s1_re;\n"
    "}\n"
    "\n"
    "__kernel void quantum_normalize(__global float* data, const int n,\n"
    "                                __global float* norm_buffer, __local float* scratch) {\n"
    "    int lid = get_local_id(0);\n"
    "    int gid = get_global_id(0);\n"
    "    int group_size = get_local_size(0);\n"
    "    \n"
    "    float val = (gid < n) ? data[gid] : 0.0f;\n"
    "    scratch[lid] = val * val;\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    \n"
    "    for (int s = group_size / 2; s > 0; s >>= 1) {\n"
    "        if (lid < s) scratch[lid] += scratch[lid + s];\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    }\n"
    "    \n"
    "    if (lid == 0) atomic_add((volatile __global int*)norm_buffer, (int)(scratch[0] * 1000000.0f));\n"
    "}\n"
    "\n"
    "__kernel void vector_scale(__global float* data, const int n, const float scale) {\n"
    "    int gid = get_global_id(0);\n"
    "    if (gid < n) data[gid] *= scale;\n"
    "}\n";

typedef struct OpenCLNodeContext {
    int node_id;
    int local_rank;
    int global_rank;
    cl_context context;
    cl_command_queue* command_queues;
    cl_device_id* devices;
    cl_mem* device_buffers;
    cl_mem* grad_buffers;
    cl_mem* param_buffers;
    cl_program program;
    cl_kernel unitary_kernel;
    cl_kernel normalize_kernel;
    cl_kernel scale_kernel;
    void* host_buffer;
    size_t buffer_size;
    int num_devices;
} OpenCLNodeContext;

typedef struct OpenCLDistributedContext {
    int num_nodes;
    OpenCLNodeContext** nodes;
    OpenCLNodeContext* local_node;
    cl_platform_id platform;
    size_t total_work_items;
    PerformanceMonitor* monitor;
} OpenCLDistributedContext;

static DistributedContext* opencl_init_distributed_engine(const DistributedConfig* config) {
    OpenCLDistributedContext* ocl_ctx = calloc(1, sizeof(OpenCLDistributedContext));
    if (!ocl_ctx) return NULL;

    cl_uint num_platforms;
    clGetPlatformIDs(1, &ocl_ctx->platform, &num_platforms);
    if (num_platforms == 0) {
        free(ocl_ctx);
        return NULL;
    }

    ocl_ctx->num_nodes = config->num_nodes > 0 ? config->num_nodes : 1;
    ocl_ctx->nodes = calloc(ocl_ctx->num_nodes, sizeof(OpenCLNodeContext*));

    OpenCLNodeContext* node = calloc(1, sizeof(OpenCLNodeContext));

    cl_uint num_devices;
    cl_int err = clGetDeviceIDs(ocl_ctx->platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
    if (err != CL_SUCCESS || num_devices == 0) {
        clGetDeviceIDs(ocl_ctx->platform, CL_DEVICE_TYPE_CPU, 0, NULL, &num_devices);
        if (num_devices == 0) {
            free(node);
            free(ocl_ctx->nodes);
            free(ocl_ctx);
            return NULL;
        }
        node->devices = calloc(num_devices, sizeof(cl_device_id));
        clGetDeviceIDs(ocl_ctx->platform, CL_DEVICE_TYPE_CPU, num_devices, node->devices, NULL);
    } else {
        node->devices = calloc(num_devices, sizeof(cl_device_id));
        clGetDeviceIDs(ocl_ctx->platform, CL_DEVICE_TYPE_GPU, num_devices, node->devices, NULL);
    }
    node->num_devices = num_devices;

    node->context = clCreateContext(NULL, num_devices, node->devices, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        free(node->devices);
        free(node);
        free(ocl_ctx->nodes);
        free(ocl_ctx);
        return NULL;
    }

    // Compile kernels
    size_t kernel_len = strlen(opencl_quantum_kernels);
    node->program = clCreateProgramWithSource(node->context, 1, &opencl_quantum_kernels, &kernel_len, &err);
    if (err == CL_SUCCESS) {
        err = clBuildProgram(node->program, num_devices, node->devices, "-cl-fast-relaxed-math", NULL, NULL);
        if (err == CL_SUCCESS) {
            node->unitary_kernel = clCreateKernel(node->program, "quantum_unitary", NULL);
            node->normalize_kernel = clCreateKernel(node->program, "quantum_normalize", NULL);
            node->scale_kernel = clCreateKernel(node->program, "vector_scale", NULL);
        }
    }

    node->command_queues = calloc(num_devices, sizeof(cl_command_queue));
    for (cl_uint i = 0; i < num_devices; i++) {
        node->command_queues[i] = clCreateCommandQueue(node->context, node->devices[i],
                                                        CL_QUEUE_PROFILING_ENABLE, NULL);
    }

    size_t buffer_size = config->gpu_buffer_size > 0 ? config->gpu_buffer_size : DEFAULT_GPU_BUFFER_SIZE;
    node->buffer_size = buffer_size;
    node->device_buffers = calloc(num_devices, sizeof(cl_mem));
    node->grad_buffers = calloc(num_devices, sizeof(cl_mem));
    node->param_buffers = calloc(num_devices, sizeof(cl_mem));

    for (cl_uint i = 0; i < num_devices; i++) {
        node->device_buffers[i] = clCreateBuffer(node->context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
        node->grad_buffers[i] = clCreateBuffer(node->context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
        node->param_buffers[i] = clCreateBuffer(node->context, CL_MEM_READ_ONLY, 64 * sizeof(float), NULL, NULL);
    }

    size_t host_size = config->host_buffer_size > 0 ? config->host_buffer_size : DEFAULT_HOST_BUFFER_SIZE;
    posix_memalign(&node->host_buffer, CACHE_LINE, host_size);

    ocl_ctx->nodes[0] = node;
    ocl_ctx->local_node = node;

    DistributedContext* ctx = calloc(1, sizeof(DistributedContext));
    ctx->backend_type = DISTRIBUTED_BACKEND_OPENCL;
    ctx->backend_data = ocl_ctx;
    ctx->num_nodes = ocl_ctx->num_nodes;
    ctx->node_rank = 0;

    return ctx;
}

static int opencl_execute_distributed_operation(DistributedContext* ctx, const QuantumOperation* op) {
    OpenCLDistributedContext* ocl_ctx = (OpenCLDistributedContext*)ctx->backend_data;
    OpenCLNodeContext* node = ocl_ctx->local_node;

    ExecutionPlan* plan = create_execution_plan(ctx, op);
    if (!plan) return -1;

    size_t work_items = plan->num_work_items;
    size_t global_size = (work_items + 255) / 256 * 256;
    size_t local_size = 256;

    for (int i = 0; i < node->num_devices; i++) {
        // Write data to device
        clEnqueueWriteBuffer(node->command_queues[i], node->device_buffers[i], CL_FALSE,
                             0, op->data_size, op->data, 0, NULL, NULL);

        if (op->parameters && op->param_size > 0) {
            clEnqueueWriteBuffer(node->command_queues[i], node->param_buffers[i], CL_FALSE,
                                 0, op->param_size, op->parameters, 0, NULL, NULL);
        }

        clFinish(node->command_queues[i]);

        // Execute appropriate kernel
        switch (op->operation_type) {
            case QUANTUM_OP_UNITARY:
                if (node->unitary_kernel) {
                    int n = (int)work_items;
                    int param_size = (int)op->param_size / sizeof(float);
                    clSetKernelArg(node->unitary_kernel, 0, sizeof(cl_mem), &node->device_buffers[i]);
                    clSetKernelArg(node->unitary_kernel, 1, sizeof(int), &n);
                    clSetKernelArg(node->unitary_kernel, 2, sizeof(cl_mem), &node->param_buffers[i]);
                    clSetKernelArg(node->unitary_kernel, 3, sizeof(int), &param_size);

                    size_t unitary_global = (work_items / 2 + 255) / 256 * 256;
                    clEnqueueNDRangeKernel(node->command_queues[i], node->unitary_kernel, 1, NULL,
                                           &unitary_global, &local_size, 0, NULL, NULL);
                }
                break;

            case QUANTUM_OP_NORMALIZE:
                if (node->normalize_kernel && node->scale_kernel) {
                    // Two-pass normalization
                    int n = (int)work_items;
                    cl_mem norm_buffer = clCreateBuffer(node->context, CL_MEM_READ_WRITE,
                                                        sizeof(float), NULL, NULL);
                    float zero = 0.0f;
                    clEnqueueWriteBuffer(node->command_queues[i], norm_buffer, CL_TRUE,
                                         0, sizeof(float), &zero, 0, NULL, NULL);

                    clSetKernelArg(node->normalize_kernel, 0, sizeof(cl_mem), &node->device_buffers[i]);
                    clSetKernelArg(node->normalize_kernel, 1, sizeof(int), &n);
                    clSetKernelArg(node->normalize_kernel, 2, sizeof(cl_mem), &norm_buffer);
                    clSetKernelArg(node->normalize_kernel, 3, local_size * sizeof(float), NULL);

                    clEnqueueNDRangeKernel(node->command_queues[i], node->normalize_kernel, 1, NULL,
                                           &global_size, &local_size, 0, NULL, NULL);
                    clFinish(node->command_queues[i]);

                    float norm_sq;
                    clEnqueueReadBuffer(node->command_queues[i], norm_buffer, CL_TRUE,
                                        0, sizeof(float), &norm_sq, 0, NULL, NULL);
                    norm_sq /= 1000000.0f;  // Undo fixed-point scaling

                    if (norm_sq > 1e-20f) {
                        float scale = 1.0f / sqrtf(norm_sq);
                        clSetKernelArg(node->scale_kernel, 0, sizeof(cl_mem), &node->device_buffers[i]);
                        clSetKernelArg(node->scale_kernel, 1, sizeof(int), &n);
                        clSetKernelArg(node->scale_kernel, 2, sizeof(float), &scale);

                        clEnqueueNDRangeKernel(node->command_queues[i], node->scale_kernel, 1, NULL,
                                               &global_size, &local_size, 0, NULL, NULL);
                    }

                    clReleaseMemObject(norm_buffer);
                }
                break;

            default:
                break;
        }

        clFinish(node->command_queues[i]);

        // Read back results
        clEnqueueReadBuffer(node->command_queues[i], node->device_buffers[i], CL_TRUE,
                            0, op->data_size, op->data, 0, NULL, NULL);
    }

    cleanup_execution(plan);
    return 0;
}

static void opencl_cleanup_distributed_engine(DistributedContext* ctx) {
    if (!ctx || !ctx->backend_data) return;
    OpenCLDistributedContext* ocl_ctx = (OpenCLDistributedContext*)ctx->backend_data;

    for (int i = 0; i < ocl_ctx->num_nodes; i++) {
        OpenCLNodeContext* node = ocl_ctx->nodes[i];
        if (!node) continue;

        if (node->unitary_kernel) clReleaseKernel(node->unitary_kernel);
        if (node->normalize_kernel) clReleaseKernel(node->normalize_kernel);
        if (node->scale_kernel) clReleaseKernel(node->scale_kernel);
        if (node->program) clReleaseProgram(node->program);

        for (int j = 0; j < node->num_devices; j++) {
            if (node->device_buffers) clReleaseMemObject(node->device_buffers[j]);
            if (node->grad_buffers) clReleaseMemObject(node->grad_buffers[j]);
            if (node->param_buffers) clReleaseMemObject(node->param_buffers[j]);
            if (node->command_queues) clReleaseCommandQueue(node->command_queues[j]);
        }
        if (node->context) clReleaseContext(node->context);

        free(node->device_buffers);
        free(node->grad_buffers);
        free(node->param_buffers);
        free(node->command_queues);
        free(node->devices);
        free(node->host_buffer);
        free(node);
    }

    free(ocl_ctx->nodes);
    free(ocl_ctx);
}

#endif // BACKEND_OPENCL_AVAILABLE

// ============================================================================
// CPU Backend Implementation
// ============================================================================

typedef struct CPUNodeContext {
    int node_id;
    int local_rank;
    int global_rank;
    void* work_buffer;
    void* grad_buffer;
    size_t buffer_size;
    int num_threads;
} CPUNodeContext;

typedef struct CPUDistributedContext {
    int num_nodes;
    CPUNodeContext** nodes;
    CPUNodeContext* local_node;
    size_t total_work_items;
    PerformanceMonitor* monitor;
    pthread_mutex_t sync_mutex;
} CPUDistributedContext;

static DistributedContext* cpu_init_distributed_engine(const DistributedConfig* config) {
    CPUDistributedContext* cpu_ctx = calloc(1, sizeof(CPUDistributedContext));
    if (!cpu_ctx) return NULL;

    pthread_mutex_init(&cpu_ctx->sync_mutex, NULL);
    cpu_ctx->num_nodes = config->num_nodes > 0 ? config->num_nodes : 1;
    cpu_ctx->nodes = calloc(cpu_ctx->num_nodes, sizeof(CPUNodeContext*));

    CPUNodeContext* node = calloc(1, sizeof(CPUNodeContext));
    node->node_id = 0;
    node->local_rank = 0;
    node->global_rank = 0;

    size_t buffer_size = config->host_buffer_size > 0 ? config->host_buffer_size : DEFAULT_HOST_BUFFER_SIZE;
    node->buffer_size = buffer_size;
    posix_memalign(&node->work_buffer, CACHE_LINE, buffer_size);
    posix_memalign(&node->grad_buffer, CACHE_LINE, buffer_size);
    memset(node->work_buffer, 0, buffer_size);
    memset(node->grad_buffer, 0, buffer_size);

#ifdef _OPENMP
    #pragma omp parallel
    {
        #pragma omp single
        node->num_threads = omp_get_num_threads();
    }
#else
    node->num_threads = 1;
#endif

    cpu_ctx->nodes[0] = node;
    cpu_ctx->local_node = node;

    DistributedContext* ctx = calloc(1, sizeof(DistributedContext));
    ctx->backend_type = DISTRIBUTED_BACKEND_CPU;
    ctx->backend_data = cpu_ctx;
    ctx->num_nodes = cpu_ctx->num_nodes;
    ctx->node_rank = 0;

    return ctx;
}

static int cpu_execute_distributed_operation(DistributedContext* ctx, const QuantumOperation* op) {
    CPUDistributedContext* cpu_ctx = (CPUDistributedContext*)ctx->backend_data;
    CPUNodeContext* node = cpu_ctx->local_node;

    ExecutionPlan* plan = create_execution_plan(ctx, op);
    if (!plan) return -1;

    size_t data_size = op->data_size;
    if (data_size > node->buffer_size) data_size = node->buffer_size;
    memcpy(node->work_buffer, op->data, data_size);

    size_t work_items = plan->num_work_items;
    float* data = (float*)node->work_buffer;
    float* grad_data = (float*)node->grad_buffer;

    switch (op->operation_type) {
        case QUANTUM_OP_UNITARY:
            quantum_unitary_transform(data, work_items * 2,
                                      (const float*)op->parameters,
                                      op->param_size / sizeof(float));
            break;

        case QUANTUM_OP_NORMALIZE:
            quantum_normalize_state(data, work_items);
            break;

        case QUANTUM_OP_TENSOR_CONTRACT: {
            size_t dim = (size_t)sqrtf((float)work_items);
            if (dim * dim == work_items && op->parameters) {
                quantum_tensor_contract(grad_data, data, (const float*)op->parameters, dim, dim, dim);
                memcpy(data, grad_data, work_items * sizeof(float));
            }
            break;
        }

        case QUANTUM_OP_GRADIENT:
            if (op->parameters) {
                quantum_compute_gradient(grad_data, data, (const float*)op->parameters, work_items);
            }
            break;

        default:
            break;
    }

    memcpy(op->data, data, data_size);

    cleanup_execution(plan);
    return 0;
}

static void cpu_cleanup_distributed_engine(DistributedContext* ctx) {
    if (!ctx || !ctx->backend_data) return;
    CPUDistributedContext* cpu_ctx = (CPUDistributedContext*)ctx->backend_data;

    for (int i = 0; i < cpu_ctx->num_nodes; i++) {
        CPUNodeContext* node = cpu_ctx->nodes[i];
        if (!node) continue;
        free(node->work_buffer);
        free(node->grad_buffer);
        free(node);
    }

    free(cpu_ctx->nodes);
    pthread_mutex_destroy(&cpu_ctx->sync_mutex);
    free(cpu_ctx);
}

// ============================================================================
// Public API Implementation
// ============================================================================

static DistributedBackendType detect_best_backend(const DistributedConfig* config) {
    // Check user preference
    if (config && config->preferred_backend != DISTRIBUTED_BACKEND_CPU) {
        switch (config->preferred_backend) {
#if BACKEND_CUDA_AVAILABLE
            case DISTRIBUTED_BACKEND_CUDA:
                return DISTRIBUTED_BACKEND_CUDA;
#endif
#if BACKEND_METAL_AVAILABLE
            case DISTRIBUTED_BACKEND_METAL:
                return DISTRIBUTED_BACKEND_METAL;
#endif
#if BACKEND_OPENCL_AVAILABLE
            case DISTRIBUTED_BACKEND_OPENCL:
                return DISTRIBUTED_BACKEND_OPENCL;
#endif
            default:
                break;
        }
    }

    // Auto-detect best available
#if BACKEND_CUDA_AVAILABLE
    return DISTRIBUTED_BACKEND_CUDA;
#elif BACKEND_METAL_AVAILABLE
    return DISTRIBUTED_BACKEND_METAL;
#elif BACKEND_OPENCL_AVAILABLE
    return DISTRIBUTED_BACKEND_OPENCL;
#else
    return DISTRIBUTED_BACKEND_CPU;
#endif
}

DistributedContext* init_distributed_engine(const DistributedConfig* config) {
    if (!config) return NULL;

    DistributedBackendType backend = detect_best_backend(config);

    switch (backend) {
#if BACKEND_CUDA_AVAILABLE
        case DISTRIBUTED_BACKEND_CUDA:
            return cuda_init_distributed_engine(config);
#endif
#if BACKEND_METAL_AVAILABLE
        case DISTRIBUTED_BACKEND_METAL:
            return metal_init_distributed_engine(config);
#endif
#if BACKEND_OPENCL_AVAILABLE
        case DISTRIBUTED_BACKEND_OPENCL:
            return opencl_init_distributed_engine(config);
#endif
        case DISTRIBUTED_BACKEND_CPU:
        default:
            return cpu_init_distributed_engine(config);
    }
}

int execute_distributed_operation(DistributedContext* ctx, const QuantumOperation* op) {
    if (!ctx || !op) return -1;

    switch (ctx->backend_type) {
#if BACKEND_CUDA_AVAILABLE
        case DISTRIBUTED_BACKEND_CUDA:
            return cuda_execute_distributed_operation(ctx, op);
#endif
#if BACKEND_METAL_AVAILABLE
        case DISTRIBUTED_BACKEND_METAL:
            return metal_execute_distributed_operation(ctx, op);
#endif
#if BACKEND_OPENCL_AVAILABLE
        case DISTRIBUTED_BACKEND_OPENCL:
            return opencl_execute_distributed_operation(ctx, op);
#endif
        case DISTRIBUTED_BACKEND_CPU:
        default:
            return cpu_execute_distributed_operation(ctx, op);
    }
}

void cleanup_distributed_engine(DistributedContext* ctx) {
    if (!ctx) return;

    switch (ctx->backend_type) {
#if BACKEND_CUDA_AVAILABLE
        case DISTRIBUTED_BACKEND_CUDA:
            cuda_cleanup_distributed_engine(ctx);
            break;
#endif
#if BACKEND_METAL_AVAILABLE
        case DISTRIBUTED_BACKEND_METAL:
            metal_cleanup_distributed_engine(ctx);
            break;
#endif
#if BACKEND_OPENCL_AVAILABLE
        case DISTRIBUTED_BACKEND_OPENCL:
            opencl_cleanup_distributed_engine(ctx);
            break;
#endif
        case DISTRIBUTED_BACKEND_CPU:
        default:
            cpu_cleanup_distributed_engine(ctx);
            break;
    }

    free(ctx);
}

// ============================================================================
// Utility Functions
// ============================================================================

DistributedBackendType get_distributed_backend_type(const DistributedContext* ctx) {
    return ctx ? ctx->backend_type : DISTRIBUTED_BACKEND_CPU;
}

// Renamed to avoid conflict with numerical_backend_accelerate.c
bool is_distributed_backend_available(DistributedBackendType backend) {
    switch (backend) {
        case DISTRIBUTED_BACKEND_CUDA:
#if BACKEND_CUDA_AVAILABLE
            return true;
#else
            return false;
#endif
        case DISTRIBUTED_BACKEND_METAL:
#if BACKEND_METAL_AVAILABLE
            return true;
#else
            return false;
#endif
        case DISTRIBUTED_BACKEND_OPENCL:
#if BACKEND_OPENCL_AVAILABLE
            return true;
#else
            return false;
#endif
        case DISTRIBUTED_BACKEND_CPU:
            return true;
        default:
            return false;
    }
}

size_t calculate_data_size(const QuantumOperation* op) {
    return op ? op->data_size : 0;
}

ExecutionPlan* create_execution_plan(DistributedContext* ctx, const QuantumOperation* op) {
    if (!ctx || !op) return NULL;

    ExecutionPlan* plan = calloc(1, sizeof(ExecutionPlan));
    if (!plan) return NULL;

    plan->num_work_items = calculate_data_size(op) / sizeof(float);
    plan->work_item_size = sizeof(float);

    plan->node_assignments = calloc(ctx->num_nodes, sizeof(int));
    if (!plan->node_assignments) {
        free(plan);
        return NULL;
    }

    size_t items_per_node = plan->num_work_items / ctx->num_nodes;
    for (int i = 0; i < ctx->num_nodes; i++) {
        plan->node_assignments[i] = (int)items_per_node;
    }

    size_t remainder = plan->num_work_items % ctx->num_nodes;
    for (size_t i = 0; i < remainder; i++) {
        plan->node_assignments[i]++;
    }

    return plan;
}

void cleanup_execution(ExecutionPlan* plan) {
    if (!plan) return;
    free(plan->node_assignments);
    free(plan->workspace);
    free(plan);
}

void update_performance_metrics(PerformanceMonitor* monitor, const QuantumOperation* op,
                                const ExecutionPlan* plan) {
    (void)monitor;
    (void)op;
    (void)plan;
    // Performance monitoring implementation would go here
}

#if BACKEND_CUDA_AVAILABLE
void launch_gpu_kernel(void* buffer, const QuantumOperation* op,
                       const ExecutionPlan* plan, cudaStream_t stream) {
    // Forward to CUDA kernel implementations
    switch (op->operation_type) {
        case QUANTUM_OP_UNITARY:
            cuda_quantum_unitary_kernel(buffer, plan->num_work_items,
                                        op->parameters, op->param_size, stream);
            break;
        case QUANTUM_OP_NORMALIZE:
            cuda_quantum_normalize_kernel(buffer, plan->num_work_items, stream);
            break;
        default:
            break;
    }
}
#endif

PerformanceMonitor* init_performance_monitor_distributed(MonitorConfig* config) {
    (void)config;
    // Performance monitor initialization would go here
    return NULL;
}

void cleanup_performance_monitor_distributed(PerformanceMonitor* monitor) {
    (void)monitor;
    // Stub - matches init_performance_monitor_distributed
}
