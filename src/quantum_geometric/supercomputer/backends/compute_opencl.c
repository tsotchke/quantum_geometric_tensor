/**
 * compute_opencl.c - OpenCL backend implementation
 *
 * Cross-platform GPU acceleration using OpenCL. Provides:
 * - Support for NVIDIA, AMD, Intel GPUs
 * - Runtime kernel compilation
 * - MPI integration for distributed clusters
 *
 * This backend is useful for heterogeneous clusters with mixed GPU vendors.
 */

#include "quantum_geometric/supercomputer/compute_backend.h"
#include "quantum_geometric/supercomputer/compute_simd.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if COMPUTE_HAS_OPENCL

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#if COMPUTE_HAS_MPI
#include <mpi.h>
#endif

// ============================================================================
// OpenCL Kernel Source
// ============================================================================

static const char* kQuantumKernelSource = "\n\
// Complex number operations\n\
typedef float2 complex_t;\n\
\n\
inline complex_t complex_mul(complex_t a, complex_t b) {\n\
    return (complex_t)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);\n\
}\n\
\n\
inline complex_t complex_conj_mul(complex_t a, complex_t b) {\n\
    return (complex_t)(a.x * b.x + a.y * b.y, a.x * b.y - a.y * b.x);\n\
}\n\
\n\
// Unitary transformation: out = U * state\n\
__kernel void quantum_unitary_transform(\n\
    __global const complex_t* state,\n\
    __global const complex_t* unitary,\n\
    __global complex_t* output,\n\
    const uint state_size\n\
) {\n\
    uint tid = get_global_id(0);\n\
    if (tid >= state_size) return;\n\
\n\
    complex_t result = (complex_t)(0.0f, 0.0f);\n\
    for (uint j = 0; j < state_size; j++) {\n\
        complex_t u = unitary[tid * state_size + j];\n\
        complex_t s = state[j];\n\
        result += complex_mul(u, s);\n\
    }\n\
    output[tid] = result;\n\
}\n\
\n\
// Compute norm squared (partial reduction)\n\
__kernel void quantum_norm_squared(\n\
    __global const complex_t* state,\n\
    __global float* partial_sums,\n\
    const uint size,\n\
    __local float* local_sum\n\
) {\n\
    uint tid = get_global_id(0);\n\
    uint lid = get_local_id(0);\n\
    uint group_size = get_local_size(0);\n\
\n\
    float sum = 0.0f;\n\
    if (tid < size) {\n\
        complex_t amp = state[tid];\n\
        sum = amp.x * amp.x + amp.y * amp.y;\n\
    }\n\
\n\
    local_sum[lid] = sum;\n\
    barrier(CLK_LOCAL_MEM_FENCE);\n\
\n\
    // Parallel reduction\n\
    for (uint stride = group_size / 2; stride > 0; stride /= 2) {\n\
        if (lid < stride) {\n\
            local_sum[lid] += local_sum[lid + stride];\n\
        }\n\
        barrier(CLK_LOCAL_MEM_FENCE);\n\
    }\n\
\n\
    if (lid == 0) {\n\
        partial_sums[get_group_id(0)] = local_sum[0];\n\
    }\n\
}\n\
\n\
// Scale state vector\n\
__kernel void quantum_scale(\n\
    __global complex_t* state,\n\
    const float scale,\n\
    const uint size\n\
) {\n\
    uint tid = get_global_id(0);\n\
    if (tid >= size) return;\n\
    state[tid] *= scale;\n\
}\n\
\n\
// Complex matrix multiplication for tensor contraction\n\
__kernel void complex_matrix_multiply(\n\
    __global const complex_t* A,\n\
    __global const complex_t* B,\n\
    __global complex_t* C,\n\
    const uint M,\n\
    const uint N,\n\
    const uint K\n\
) {\n\
    uint row = get_global_id(1);\n\
    uint col = get_global_id(0);\n\
\n\
    if (row >= M || col >= K) return;\n\
\n\
    complex_t sum = (complex_t)(0.0f, 0.0f);\n\
    for (uint i = 0; i < N; i++) {\n\
        sum += complex_mul(A[row * N + i], B[i * K + col]);\n\
    }\n\
    C[row * K + col] = sum;\n\
}\n\
\n\
// Inner product: <a|b>\n\
__kernel void quantum_inner_product(\n\
    __global const complex_t* state_a,\n\
    __global const complex_t* state_b,\n\
    __global complex_t* partial_results,\n\
    const uint size,\n\
    __local complex_t* local_sum\n\
) {\n\
    uint tid = get_global_id(0);\n\
    uint lid = get_local_id(0);\n\
    uint group_size = get_local_size(0);\n\
\n\
    complex_t sum = (complex_t)(0.0f, 0.0f);\n\
    if (tid < size) {\n\
        sum = complex_conj_mul(state_a[tid], state_b[tid]);\n\
    }\n\
\n\
    local_sum[lid] = sum;\n\
    barrier(CLK_LOCAL_MEM_FENCE);\n\
\n\
    for (uint stride = group_size / 2; stride > 0; stride /= 2) {\n\
        if (lid < stride) {\n\
            local_sum[lid] += local_sum[lid + stride];\n\
        }\n\
        barrier(CLK_LOCAL_MEM_FENCE);\n\
    }\n\
\n\
    if (lid == 0) {\n\
        partial_results[get_group_id(0)] = local_sum[0];\n\
    }\n\
}\n\
\n\
// Gradient computation\n\
__kernel void quantum_gradient_compute(\n\
    __global const complex_t* forward_state,\n\
    __global const complex_t* backward_state,\n\
    __global complex_t* gradient,\n\
    const uint size\n\
) {\n\
    uint tid = get_global_id(0);\n\
    if (tid >= size) return;\n\
\n\
    complex_t fw = forward_state[tid];\n\
    complex_t bw = backward_state[tid];\n\
    gradient[tid] = complex_conj_mul(bw, fw);\n\
}\n\
\n\
// Expectation value for diagonal observable\n\
__kernel void quantum_expectation_diagonal(\n\
    __global const complex_t* state,\n\
    __global const float* observable,\n\
    __global float* partial_sums,\n\
    const uint size,\n\
    __local float* local_sum\n\
) {\n\
    uint tid = get_global_id(0);\n\
    uint lid = get_local_id(0);\n\
    uint group_size = get_local_size(0);\n\
\n\
    float sum = 0.0f;\n\
    if (tid < size) {\n\
        complex_t amp = state[tid];\n\
        float prob = amp.x * amp.x + amp.y * amp.y;\n\
        sum = prob * observable[tid];\n\
    }\n\
\n\
    local_sum[lid] = sum;\n\
    barrier(CLK_LOCAL_MEM_FENCE);\n\
\n\
    for (uint stride = group_size / 2; stride > 0; stride /= 2) {\n\
        if (lid < stride) {\n\
            local_sum[lid] += local_sum[lid + stride];\n\
        }\n\
        barrier(CLK_LOCAL_MEM_FENCE);\n\
    }\n\
\n\
    if (lid == 0) {\n\
        partial_sums[get_group_id(0)] = local_sum[0];\n\
    }\n\
}\n";

// ============================================================================
// OpenCL Backend Context
// ============================================================================

typedef struct OpenCLBackendContext {
    // OpenCL objects
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;

    // Kernels
    cl_kernel unitaryKernel;
    cl_kernel normKernel;
    cl_kernel scaleKernel;
    cl_kernel matmulKernel;
    cl_kernel innerProductKernel;
    cl_kernel gradientKernel;
    cl_kernel expectationKernel;

    // Configuration
    int node_rank;
    int num_nodes;
    size_t workgroup_size;

    // MPI state
#if COMPUTE_HAS_MPI
    MPI_Comm comm;
    bool mpi_initialized_by_us;
#endif

    // Memory tracking
    size_t total_allocated;
    size_t peak_allocated;

    // Performance metrics
    ComputeMetrics metrics;

    // Error state
    char last_error[256];
} OpenCLBackendContext;

// ============================================================================
// Helper Functions
// ============================================================================

static void set_cl_error(OpenCLBackendContext* ctx, const char* msg, cl_int err) {
    snprintf(ctx->last_error, sizeof(ctx->last_error), "%s: error %d", msg, err);
}

static cl_kernel create_kernel(OpenCLBackendContext* ctx, const char* name) {
    cl_int err;
    cl_kernel kernel = clCreateKernel(ctx->program, name, &err);
    if (err != CL_SUCCESS) {
        set_cl_error(ctx, name, err);
        return NULL;
    }
    return kernel;
}

// ============================================================================
// Lifecycle Operations
// ============================================================================

static ComputeBackend* opencl_init(const ComputeDistributedConfig* config) {
    cl_int err;

    OpenCLBackendContext* ctx = calloc(1, sizeof(OpenCLBackendContext));
    if (!ctx) return NULL;

    // Get platform
    err = clGetPlatformIDs(1, &ctx->platform, NULL);
    if (err != CL_SUCCESS) {
        set_cl_error(ctx, "clGetPlatformIDs", err);
        free(ctx);
        return NULL;
    }

    // Get GPU device (prefer GPU, fall back to CPU)
    err = clGetDeviceIDs(ctx->platform, CL_DEVICE_TYPE_GPU, 1, &ctx->device, NULL);
    if (err != CL_SUCCESS) {
        err = clGetDeviceIDs(ctx->platform, CL_DEVICE_TYPE_CPU, 1, &ctx->device, NULL);
        if (err != CL_SUCCESS) {
            set_cl_error(ctx, "clGetDeviceIDs", err);
            free(ctx);
            return NULL;
        }
    }

    // Create context
    ctx->context = clCreateContext(NULL, 1, &ctx->device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        set_cl_error(ctx, "clCreateContext", err);
        free(ctx);
        return NULL;
    }

    // Create command queue
#ifdef CL_VERSION_2_0
    cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    ctx->queue = clCreateCommandQueueWithProperties(ctx->context, ctx->device, props, &err);
#else
    ctx->queue = clCreateCommandQueue(ctx->context, ctx->device, CL_QUEUE_PROFILING_ENABLE, &err);
#endif
    if (err != CL_SUCCESS) {
        set_cl_error(ctx, "clCreateCommandQueue", err);
        clReleaseContext(ctx->context);
        free(ctx);
        return NULL;
    }

    // Build program
    const char* source = kQuantumKernelSource;
    size_t source_len = strlen(source);
    ctx->program = clCreateProgramWithSource(ctx->context, 1, &source, &source_len, &err);
    if (err != CL_SUCCESS) {
        set_cl_error(ctx, "clCreateProgramWithSource", err);
        clReleaseCommandQueue(ctx->queue);
        clReleaseContext(ctx->context);
        free(ctx);
        return NULL;
    }

    err = clBuildProgram(ctx->program, 1, &ctx->device, "-cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS) {
        // Get build log
        size_t log_size;
        clGetProgramBuildInfo(ctx->program, ctx->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = malloc(log_size);
        if (log) {
            clGetProgramBuildInfo(ctx->program, ctx->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            snprintf(ctx->last_error, sizeof(ctx->last_error), "Build failed: %s", log);
            free(log);
        }
        clReleaseProgram(ctx->program);
        clReleaseCommandQueue(ctx->queue);
        clReleaseContext(ctx->context);
        free(ctx);
        return NULL;
    }

    // Create kernels
    ctx->unitaryKernel = create_kernel(ctx, "quantum_unitary_transform");
    ctx->normKernel = create_kernel(ctx, "quantum_norm_squared");
    ctx->scaleKernel = create_kernel(ctx, "quantum_scale");
    ctx->matmulKernel = create_kernel(ctx, "complex_matrix_multiply");
    ctx->innerProductKernel = create_kernel(ctx, "quantum_inner_product");
    ctx->gradientKernel = create_kernel(ctx, "quantum_gradient_compute");
    ctx->expectationKernel = create_kernel(ctx, "quantum_expectation_diagonal");

    // Get workgroup size
    clGetDeviceInfo(ctx->device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                    sizeof(ctx->workgroup_size), &ctx->workgroup_size, NULL);
    if (ctx->workgroup_size > 256) ctx->workgroup_size = 256;

    // Initialize MPI
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
    (void)config;
#endif

    return (ComputeBackend*)ctx;
}

static void opencl_cleanup(ComputeBackend* backend) {
    OpenCLBackendContext* ctx = (OpenCLBackendContext*)backend;
    if (!ctx) return;

    if (ctx->unitaryKernel) clReleaseKernel(ctx->unitaryKernel);
    if (ctx->normKernel) clReleaseKernel(ctx->normKernel);
    if (ctx->scaleKernel) clReleaseKernel(ctx->scaleKernel);
    if (ctx->matmulKernel) clReleaseKernel(ctx->matmulKernel);
    if (ctx->innerProductKernel) clReleaseKernel(ctx->innerProductKernel);
    if (ctx->gradientKernel) clReleaseKernel(ctx->gradientKernel);
    if (ctx->expectationKernel) clReleaseKernel(ctx->expectationKernel);
    if (ctx->program) clReleaseProgram(ctx->program);
    if (ctx->queue) clReleaseCommandQueue(ctx->queue);
    if (ctx->context) clReleaseContext(ctx->context);

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

static bool opencl_probe(void) {
    cl_platform_id platform;
    cl_int err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) return false;

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
    }

    return err == CL_SUCCESS;
}

static ComputeResult opencl_get_capabilities(ComputeBackend* backend,
                                              int* num_devices,
                                              size_t* total_memory) {
    OpenCLBackendContext* ctx = (OpenCLBackendContext*)backend;
    if (!ctx) return COMPUTE_ERROR_INVALID_ARGUMENT;

    if (num_devices) {
        cl_uint count;
        clGetDeviceIDs(ctx->platform, CL_DEVICE_TYPE_ALL, 0, NULL, &count);
        *num_devices = (int)count;
    }

    if (total_memory) {
        cl_ulong mem;
        clGetDeviceInfo(ctx->device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, NULL);
        *total_memory = (size_t)mem;
    }

    return COMPUTE_SUCCESS;
}

// ============================================================================
// Memory Management
// ============================================================================

static void* opencl_alloc(ComputeBackend* backend, size_t size, ComputeMemType mem_type) {
    OpenCLBackendContext* ctx = (OpenCLBackendContext*)backend;
    if (!ctx) return NULL;

    cl_mem_flags flags;
    switch (mem_type) {
        case COMPUTE_MEM_DEVICE:
            flags = CL_MEM_READ_WRITE;
            break;
        case COMPUTE_MEM_HOST:
        case COMPUTE_MEM_PINNED:
            flags = CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR;
            break;
        case COMPUTE_MEM_UNIFIED:
        default:
            flags = CL_MEM_READ_WRITE;
            break;
    }

    cl_int err;
    cl_mem buffer = clCreateBuffer(ctx->context, flags, size, NULL, &err);
    if (err != CL_SUCCESS) {
        set_cl_error(ctx, "clCreateBuffer", err);
        return NULL;
    }

    ctx->total_allocated += size;
    if (ctx->total_allocated > ctx->peak_allocated) {
        ctx->peak_allocated = ctx->total_allocated;
    }

    return (void*)buffer;
}

static void opencl_free(ComputeBackend* backend, void* ptr, ComputeMemType mem_type) {
    (void)backend;
    (void)mem_type;
    if (ptr) {
        clReleaseMemObject((cl_mem)ptr);
    }
}

static ComputeResult opencl_memcpy(ComputeBackend* backend,
                                    void* dst, ComputeMemType dst_type,
                                    const void* src, ComputeMemType src_type,
                                    size_t size, ComputeStream* stream) {
    OpenCLBackendContext* ctx = (OpenCLBackendContext*)backend;
    if (!ctx) return COMPUTE_ERROR_INVALID_ARGUMENT;
    (void)stream;

    cl_int err;

    if (dst_type == COMPUTE_MEM_DEVICE && src_type == COMPUTE_MEM_HOST) {
        // Host to device
        err = clEnqueueWriteBuffer(ctx->queue, (cl_mem)dst, CL_TRUE, 0, size, src, 0, NULL, NULL);
    } else if (dst_type == COMPUTE_MEM_HOST && src_type == COMPUTE_MEM_DEVICE) {
        // Device to host
        err = clEnqueueReadBuffer(ctx->queue, (cl_mem)src, CL_TRUE, 0, size, dst, 0, NULL, NULL);
    } else if (dst_type == COMPUTE_MEM_DEVICE && src_type == COMPUTE_MEM_DEVICE) {
        // Device to device
        err = clEnqueueCopyBuffer(ctx->queue, (cl_mem)src, (cl_mem)dst, 0, 0, size, 0, NULL, NULL);
    } else {
        // Host to host
        memcpy(dst, src, size);
        return COMPUTE_SUCCESS;
    }

    if (err != CL_SUCCESS) {
        set_cl_error(ctx, "memcpy", err);
        return COMPUTE_ERROR_COMMUNICATION_FAILED;
    }

    return COMPUTE_SUCCESS;
}

static ComputeResult opencl_memset(ComputeBackend* backend,
                                    void* ptr, int value, size_t size,
                                    ComputeStream* stream) {
    OpenCLBackendContext* ctx = (OpenCLBackendContext*)backend;
    if (!ctx) return COMPUTE_ERROR_INVALID_ARGUMENT;
    (void)stream;

#ifdef CL_VERSION_1_2
    cl_int pattern = value;
    cl_int err = clEnqueueFillBuffer(ctx->queue, (cl_mem)ptr, &pattern, 1, 0, size, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        set_cl_error(ctx, "clEnqueueFillBuffer", err);
        return COMPUTE_ERROR_KERNEL_FAILED;
    }
#else
    // Fallback: map, memset, unmap
    void* mapped = clEnqueueMapBuffer(ctx->queue, (cl_mem)ptr, CL_TRUE,
                                       CL_MAP_WRITE, 0, size, 0, NULL, NULL, NULL);
    if (mapped) {
        memset(mapped, value, size);
        clEnqueueUnmapMemObject(ctx->queue, (cl_mem)ptr, mapped, 0, NULL, NULL);
    }
#endif

    return COMPUTE_SUCCESS;
}

// ============================================================================
// Stream Management
// ============================================================================

typedef struct OpenCLStream {
    cl_command_queue queue;
} OpenCLStream;

static ComputeStream* opencl_create_stream(ComputeBackend* backend) {
    OpenCLBackendContext* ctx = (OpenCLBackendContext*)backend;
    if (!ctx) return NULL;

    OpenCLStream* stream = calloc(1, sizeof(OpenCLStream));
    if (!stream) return NULL;

    cl_int err;
#ifdef CL_VERSION_2_0
    cl_queue_properties props[] = { 0 };
    stream->queue = clCreateCommandQueueWithProperties(ctx->context, ctx->device, props, &err);
#else
    stream->queue = clCreateCommandQueue(ctx->context, ctx->device, 0, &err);
#endif

    if (err != CL_SUCCESS) {
        free(stream);
        return NULL;
    }

    return (ComputeStream*)stream;
}

static void opencl_destroy_stream(ComputeBackend* backend, ComputeStream* stream) {
    (void)backend;
    if (!stream) return;

    OpenCLStream* clStream = (OpenCLStream*)stream;
    if (clStream->queue) {
        clReleaseCommandQueue(clStream->queue);
    }
    free(clStream);
}

static ComputeResult opencl_synchronize_stream(ComputeBackend* backend, ComputeStream* stream) {
    OpenCLBackendContext* ctx = (OpenCLBackendContext*)backend;
    if (!ctx) return COMPUTE_ERROR_INVALID_ARGUMENT;

    cl_command_queue queue = stream ?
        ((OpenCLStream*)stream)->queue : ctx->queue;

    cl_int err = clFinish(queue);
    if (err != CL_SUCCESS) {
        return COMPUTE_ERROR_SYNCHRONIZATION_FAILED;
    }

    return COMPUTE_SUCCESS;
}

static ComputeEvent* opencl_create_event(ComputeBackend* backend) {
    (void)backend;
    cl_event* event = calloc(1, sizeof(cl_event));
    return (ComputeEvent*)event;
}

static void opencl_destroy_event(ComputeBackend* backend, ComputeEvent* event) {
    (void)backend;
    if (event) {
        cl_event* clEvent = (cl_event*)event;
        if (*clEvent) clReleaseEvent(*clEvent);
        free(clEvent);
    }
}

static ComputeResult opencl_record_event(ComputeBackend* backend,
                                          ComputeEvent* event,
                                          ComputeStream* stream) {
    OpenCLBackendContext* ctx = (OpenCLBackendContext*)backend;
    if (!ctx || !event) return COMPUTE_ERROR_INVALID_ARGUMENT;

    cl_command_queue queue = stream ?
        ((OpenCLStream*)stream)->queue : ctx->queue;

    cl_event* clEvent = (cl_event*)event;
    clEnqueueMarker(queue, clEvent);

    return COMPUTE_SUCCESS;
}

static ComputeResult opencl_wait_event(ComputeBackend* backend,
                                        ComputeStream* stream,
                                        ComputeEvent* event) {
    (void)backend;
    (void)stream;
    if (!event) return COMPUTE_ERROR_INVALID_ARGUMENT;

    cl_event* clEvent = (cl_event*)event;
    if (*clEvent) {
        clWaitForEvents(1, clEvent);
    }

    return COMPUTE_SUCCESS;
}

// ============================================================================
// Quantum Operations
// ============================================================================

static ComputeResult opencl_quantum_unitary(ComputeBackend* backend,
                                             float* state, size_t state_size,
                                             const float* unitary, size_t unitary_size,
                                             ComputeStream* stream) {
    OpenCLBackendContext* ctx = (OpenCLBackendContext*)backend;
    if (!ctx || !state || !unitary) return COMPUTE_ERROR_INVALID_ARGUMENT;
    (void)stream;
    (void)unitary_size;

    cl_int err;

    // Create buffers
    size_t state_bytes = state_size * 2 * sizeof(float);
    size_t unitary_bytes = state_size * state_size * 2 * sizeof(float);

    cl_mem stateBuf = clCreateBuffer(ctx->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     state_bytes, state, &err);
    cl_mem unitaryBuf = clCreateBuffer(ctx->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       unitary_bytes, (void*)unitary, &err);
    cl_mem outputBuf = clCreateBuffer(ctx->context, CL_MEM_WRITE_ONLY,
                                      state_bytes, NULL, &err);

    cl_uint size = (cl_uint)state_size;

    // Set kernel arguments
    clSetKernelArg(ctx->unitaryKernel, 0, sizeof(cl_mem), &stateBuf);
    clSetKernelArg(ctx->unitaryKernel, 1, sizeof(cl_mem), &unitaryBuf);
    clSetKernelArg(ctx->unitaryKernel, 2, sizeof(cl_mem), &outputBuf);
    clSetKernelArg(ctx->unitaryKernel, 3, sizeof(cl_uint), &size);

    // Execute
    size_t global_size = state_size;
    size_t local_size = ctx->workgroup_size;
    if (local_size > global_size) local_size = global_size;

    err = clEnqueueNDRangeKernel(ctx->queue, ctx->unitaryKernel, 1, NULL,
                                  &global_size, &local_size, 0, NULL, NULL);

    // Read result
    clEnqueueReadBuffer(ctx->queue, outputBuf, CL_TRUE, 0, state_bytes, state, 0, NULL, NULL);

    // Cleanup
    clReleaseMemObject(stateBuf);
    clReleaseMemObject(unitaryBuf);
    clReleaseMemObject(outputBuf);

    return (err == CL_SUCCESS) ? COMPUTE_SUCCESS : COMPUTE_ERROR_KERNEL_FAILED;
}

static ComputeResult opencl_quantum_normalize(ComputeBackend* backend,
                                               float* state, size_t size,
                                               ComputeStream* stream) {
    OpenCLBackendContext* ctx = (OpenCLBackendContext*)backend;
    (void)stream;

    if (!ctx || !state || size == 0) {
        return COMPUTE_ERROR_INVALID_ARGUMENT;
    }

    // For large sizes, use GPU; for small sizes, SIMD is efficient
    if (size >= 1024 && ctx->normKernel && ctx->scaleKernel) {
        cl_int err;
        size_t state_bytes = size * 2 * sizeof(float);
        size_t num_groups = (size + ctx->workgroup_size - 1) / ctx->workgroup_size;

        // Create buffers
        cl_mem stateBuf = clCreateBuffer(ctx->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                         state_bytes, state, &err);
        cl_mem partialBuf = clCreateBuffer(ctx->context, CL_MEM_READ_WRITE,
                                           num_groups * sizeof(float), NULL, &err);

        cl_uint sz = (cl_uint)size;

        // Compute norm squared
        clSetKernelArg(ctx->normKernel, 0, sizeof(cl_mem), &stateBuf);
        clSetKernelArg(ctx->normKernel, 1, sizeof(cl_mem), &partialBuf);
        clSetKernelArg(ctx->normKernel, 2, sizeof(cl_uint), &sz);
        clSetKernelArg(ctx->normKernel, 3, ctx->workgroup_size * sizeof(float), NULL);

        size_t global_size = num_groups * ctx->workgroup_size;
        size_t local_size = ctx->workgroup_size;
        err = clEnqueueNDRangeKernel(ctx->queue, ctx->normKernel, 1, NULL,
                                      &global_size, &local_size, 0, NULL, NULL);

        // Read partial sums and sum on CPU
        float* partials = malloc(num_groups * sizeof(float));
        clEnqueueReadBuffer(ctx->queue, partialBuf, CL_TRUE, 0,
                            num_groups * sizeof(float), partials, 0, NULL, NULL);

        float norm_sq = 0.0f;
        for (size_t i = 0; i < num_groups; i++) {
            norm_sq += partials[i];
        }
        free(partials);

        float norm = sqrtf(norm_sq);
        if (norm > 1e-10f) {
            float scale = 1.0f / norm;

            // Scale state
            clSetKernelArg(ctx->scaleKernel, 0, sizeof(cl_mem), &stateBuf);
            clSetKernelArg(ctx->scaleKernel, 1, sizeof(float), &scale);
            clSetKernelArg(ctx->scaleKernel, 2, sizeof(cl_uint), &sz);

            clEnqueueNDRangeKernel(ctx->queue, ctx->scaleKernel, 1, NULL,
                                    &size, &local_size, 0, NULL, NULL);

            // Copy back
            clEnqueueReadBuffer(ctx->queue, stateBuf, CL_TRUE, 0,
                                state_bytes, state, 0, NULL, NULL);
        }

        clReleaseMemObject(stateBuf);
        clReleaseMemObject(partialBuf);
    } else {
        // Use SIMD for small sizes
        float norm = simd_complex_norm_float(state, size);
        if (norm > 1e-10f) {
            simd_complex_scale_float(state, size, 1.0f / norm);
        }
    }

    return COMPUTE_SUCCESS;
}

static ComputeResult opencl_quantum_tensor_contract(ComputeBackend* backend,
                                                     float* result,
                                                     const float* a, const float* b,
                                                     size_t m, size_t n, size_t k,
                                                     ComputeStream* stream) {
    OpenCLBackendContext* ctx = (OpenCLBackendContext*)backend;
    if (!ctx || !result || !a || !b) return COMPUTE_ERROR_INVALID_ARGUMENT;
    (void)stream;

    cl_int err;

    // Create buffers
    size_t a_bytes = m * n * 2 * sizeof(float);
    size_t b_bytes = n * k * 2 * sizeof(float);
    size_t c_bytes = m * k * 2 * sizeof(float);

    cl_mem aBuf = clCreateBuffer(ctx->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 a_bytes, (void*)a, &err);
    cl_mem bBuf = clCreateBuffer(ctx->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 b_bytes, (void*)b, &err);
    cl_mem cBuf = clCreateBuffer(ctx->context, CL_MEM_WRITE_ONLY, c_bytes, NULL, &err);

    cl_uint M = (cl_uint)m;
    cl_uint N = (cl_uint)n;
    cl_uint K = (cl_uint)k;

    clSetKernelArg(ctx->matmulKernel, 0, sizeof(cl_mem), &aBuf);
    clSetKernelArg(ctx->matmulKernel, 1, sizeof(cl_mem), &bBuf);
    clSetKernelArg(ctx->matmulKernel, 2, sizeof(cl_mem), &cBuf);
    clSetKernelArg(ctx->matmulKernel, 3, sizeof(cl_uint), &M);
    clSetKernelArg(ctx->matmulKernel, 4, sizeof(cl_uint), &N);
    clSetKernelArg(ctx->matmulKernel, 5, sizeof(cl_uint), &K);

    size_t global_size[2] = { k, m };
    size_t local_size[2] = { 16, 16 };

    err = clEnqueueNDRangeKernel(ctx->queue, ctx->matmulKernel, 2, NULL,
                                  global_size, local_size, 0, NULL, NULL);

    clEnqueueReadBuffer(ctx->queue, cBuf, CL_TRUE, 0, c_bytes, result, 0, NULL, NULL);

    clReleaseMemObject(aBuf);
    clReleaseMemObject(bBuf);
    clReleaseMemObject(cBuf);

    return (err == CL_SUCCESS) ? COMPUTE_SUCCESS : COMPUTE_ERROR_KERNEL_FAILED;
}

static ComputeResult opencl_quantum_gradient(ComputeBackend* backend,
                                              float* gradients,
                                              const float* forward_state,
                                              const float* backward_state,
                                              size_t size,
                                              ComputeStream* stream) {
    OpenCLBackendContext* ctx = (OpenCLBackendContext*)backend;
    (void)stream;

    if (!ctx || !gradients || !forward_state || !backward_state) {
        return COMPUTE_ERROR_INVALID_ARGUMENT;
    }

    // For large sizes, use GPU; for small sizes, SIMD is efficient
    if (size >= 1024 && ctx->gradientKernel) {
        cl_int err;
        size_t bytes = size * 2 * sizeof(float);

        cl_mem fwdBuf = clCreateBuffer(ctx->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       bytes, (void*)forward_state, &err);
        cl_mem bwdBuf = clCreateBuffer(ctx->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       bytes, (void*)backward_state, &err);
        cl_mem gradBuf = clCreateBuffer(ctx->context, CL_MEM_WRITE_ONLY, bytes, NULL, &err);

        cl_uint sz = (cl_uint)size;

        clSetKernelArg(ctx->gradientKernel, 0, sizeof(cl_mem), &fwdBuf);
        clSetKernelArg(ctx->gradientKernel, 1, sizeof(cl_mem), &bwdBuf);
        clSetKernelArg(ctx->gradientKernel, 2, sizeof(cl_mem), &gradBuf);
        clSetKernelArg(ctx->gradientKernel, 3, sizeof(cl_uint), &sz);

        size_t local_size = ctx->workgroup_size;
        err = clEnqueueNDRangeKernel(ctx->queue, ctx->gradientKernel, 1, NULL,
                                      &size, &local_size, 0, NULL, NULL);

        clEnqueueReadBuffer(ctx->queue, gradBuf, CL_TRUE, 0, bytes, gradients, 0, NULL, NULL);

        clReleaseMemObject(fwdBuf);
        clReleaseMemObject(bwdBuf);
        clReleaseMemObject(gradBuf);
    } else {
        simd_complex_inner_product(gradients, backward_state, forward_state, size);
    }

    return COMPUTE_SUCCESS;
}

static ComputeResult opencl_quantum_inner_product(ComputeBackend* backend,
                                                   float* result,
                                                   const float* state_a,
                                                   const float* state_b,
                                                   size_t size,
                                                   ComputeStream* stream) {
    OpenCLBackendContext* ctx = (OpenCLBackendContext*)backend;
    (void)stream;

    if (!ctx || !result || !state_a || !state_b) {
        return COMPUTE_ERROR_INVALID_ARGUMENT;
    }

    // For large sizes, use GPU reduction
    if (size >= 1024 && ctx->innerProductKernel) {
        cl_int err;
        size_t bytes = size * 2 * sizeof(float);
        size_t num_groups = (size + ctx->workgroup_size - 1) / ctx->workgroup_size;

        cl_mem aBuf = clCreateBuffer(ctx->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     bytes, (void*)state_a, &err);
        cl_mem bBuf = clCreateBuffer(ctx->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     bytes, (void*)state_b, &err);
        cl_mem partialBuf = clCreateBuffer(ctx->context, CL_MEM_WRITE_ONLY,
                                           num_groups * 2 * sizeof(float), NULL, &err);

        cl_uint sz = (cl_uint)size;

        clSetKernelArg(ctx->innerProductKernel, 0, sizeof(cl_mem), &aBuf);
        clSetKernelArg(ctx->innerProductKernel, 1, sizeof(cl_mem), &bBuf);
        clSetKernelArg(ctx->innerProductKernel, 2, sizeof(cl_mem), &partialBuf);
        clSetKernelArg(ctx->innerProductKernel, 3, sizeof(cl_uint), &sz);
        clSetKernelArg(ctx->innerProductKernel, 4, ctx->workgroup_size * 2 * sizeof(float), NULL);

        size_t global_size = num_groups * ctx->workgroup_size;
        size_t local_size = ctx->workgroup_size;
        err = clEnqueueNDRangeKernel(ctx->queue, ctx->innerProductKernel, 1, NULL,
                                      &global_size, &local_size, 0, NULL, NULL);

        float* partials = malloc(num_groups * 2 * sizeof(float));
        clEnqueueReadBuffer(ctx->queue, partialBuf, CL_TRUE, 0,
                            num_groups * 2 * sizeof(float), partials, 0, NULL, NULL);

        float real_sum = 0.0f, imag_sum = 0.0f;
        for (size_t i = 0; i < num_groups; i++) {
            real_sum += partials[i * 2];
            imag_sum += partials[i * 2 + 1];
        }
        result[0] = real_sum;
        result[1] = imag_sum;
        free(partials);

        clReleaseMemObject(aBuf);
        clReleaseMemObject(bBuf);
        clReleaseMemObject(partialBuf);
    } else {
        simd_complex_inner_product(result, state_a, state_b, size);
    }

    return COMPUTE_SUCCESS;
}

static ComputeResult opencl_quantum_expectation(ComputeBackend* backend,
                                                 float* result,
                                                 const float* state,
                                                 const float* observable,
                                                 size_t size,
                                                 ComputeStream* stream) {
    OpenCLBackendContext* ctx = (OpenCLBackendContext*)backend;
    (void)stream;

    if (!ctx || !result || !state || !observable) {
        return COMPUTE_ERROR_INVALID_ARGUMENT;
    }

    // For large sizes, use GPU reduction
    if (size >= 1024 && ctx->expectationKernel) {
        cl_int err;
        size_t state_bytes = size * 2 * sizeof(float);
        size_t obs_bytes = size * sizeof(float);
        size_t num_groups = (size + ctx->workgroup_size - 1) / ctx->workgroup_size;

        cl_mem stateBuf = clCreateBuffer(ctx->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         state_bytes, (void*)state, &err);
        cl_mem obsBuf = clCreateBuffer(ctx->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       obs_bytes, (void*)observable, &err);
        cl_mem partialBuf = clCreateBuffer(ctx->context, CL_MEM_WRITE_ONLY,
                                           num_groups * sizeof(float), NULL, &err);

        cl_uint sz = (cl_uint)size;

        clSetKernelArg(ctx->expectationKernel, 0, sizeof(cl_mem), &stateBuf);
        clSetKernelArg(ctx->expectationKernel, 1, sizeof(cl_mem), &obsBuf);
        clSetKernelArg(ctx->expectationKernel, 2, sizeof(cl_mem), &partialBuf);
        clSetKernelArg(ctx->expectationKernel, 3, sizeof(cl_uint), &sz);
        clSetKernelArg(ctx->expectationKernel, 4, ctx->workgroup_size * sizeof(float), NULL);

        size_t global_size = num_groups * ctx->workgroup_size;
        size_t local_size = ctx->workgroup_size;
        err = clEnqueueNDRangeKernel(ctx->queue, ctx->expectationKernel, 1, NULL,
                                      &global_size, &local_size, 0, NULL, NULL);

        float* partials = malloc(num_groups * sizeof(float));
        clEnqueueReadBuffer(ctx->queue, partialBuf, CL_TRUE, 0,
                            num_groups * sizeof(float), partials, 0, NULL, NULL);

        float sum = 0.0f;
        for (size_t i = 0; i < num_groups; i++) {
            sum += partials[i];
        }
        *result = sum;
        free(partials);

        clReleaseMemObject(stateBuf);
        clReleaseMemObject(obsBuf);
        clReleaseMemObject(partialBuf);
    } else {
        *result = simd_expectation_diagonal(state, observable, size);
    }

    return COMPUTE_SUCCESS;
}

// ============================================================================
// Collective Communication (MPI)
// ============================================================================

static ComputeResult opencl_barrier(ComputeBackend* backend) {
#if COMPUTE_HAS_MPI
    OpenCLBackendContext* ctx = (OpenCLBackendContext*)backend;
    if (ctx && ctx->num_nodes > 1) {
        MPI_Barrier(ctx->comm);
    }
#else
    (void)backend;
#endif
    return COMPUTE_SUCCESS;
}

static ComputeResult opencl_broadcast(ComputeBackend* backend,
                                       void* data, size_t size,
                                       ComputeDataType dtype, int root) {
#if COMPUTE_HAS_MPI
    OpenCLBackendContext* ctx = (OpenCLBackendContext*)backend;
    if (ctx && ctx->num_nodes > 1) {
        MPI_Bcast(data, (int)size, MPI_FLOAT, root, ctx->comm);
    }
#else
    (void)backend; (void)data; (void)size; (void)dtype; (void)root;
#endif
    return COMPUTE_SUCCESS;
}

static ComputeResult opencl_allreduce(ComputeBackend* backend,
                                       const void* send_data, void* recv_data,
                                       size_t count, ComputeDataType dtype,
                                       ComputeReduceOp op) {
#if COMPUTE_HAS_MPI
    OpenCLBackendContext* ctx = (OpenCLBackendContext*)backend;
    if (ctx && ctx->num_nodes > 1) {
        MPI_Allreduce(send_data, recv_data, (int)count, MPI_FLOAT, MPI_SUM, ctx->comm);
    } else
#endif
    {
        size_t elem_size = compute_dtype_size(dtype);
        memcpy(recv_data, send_data, count * elem_size);
        (void)backend; (void)op;
    }
    return COMPUTE_SUCCESS;
}

static ComputeResult opencl_scatter(ComputeBackend* backend,
                                     const void* send_data, void* recv_data,
                                     size_t count, ComputeDataType dtype, int root) {
#if COMPUTE_HAS_MPI
    OpenCLBackendContext* ctx = (OpenCLBackendContext*)backend;
    if (ctx && ctx->num_nodes > 1) {
        MPI_Datatype mpi_type;
        switch (dtype) {
            case COMPUTE_DTYPE_FLOAT32:   mpi_type = MPI_FLOAT; break;
            case COMPUTE_DTYPE_FLOAT64:   mpi_type = MPI_DOUBLE; break;
            case COMPUTE_DTYPE_INT32:     mpi_type = MPI_INT; break;
            case COMPUTE_DTYPE_INT64:     mpi_type = MPI_LONG_LONG; break;
            default:                      mpi_type = MPI_BYTE; break;
        }
        MPI_Scatter(send_data, (int)count, mpi_type,
                    recv_data, (int)count, mpi_type, root, ctx->comm);
    } else
#endif
    {
        size_t elem_size = compute_dtype_size(dtype);
        memcpy(recv_data, send_data, count * elem_size);
        (void)backend; (void)root;
    }
    return COMPUTE_SUCCESS;
}

static ComputeResult opencl_gather(ComputeBackend* backend,
                                    const void* send_data, void* recv_data,
                                    size_t count, ComputeDataType dtype, int root) {
#if COMPUTE_HAS_MPI
    OpenCLBackendContext* ctx = (OpenCLBackendContext*)backend;
    if (ctx && ctx->num_nodes > 1) {
        MPI_Datatype mpi_type;
        switch (dtype) {
            case COMPUTE_DTYPE_FLOAT32:   mpi_type = MPI_FLOAT; break;
            case COMPUTE_DTYPE_FLOAT64:   mpi_type = MPI_DOUBLE; break;
            case COMPUTE_DTYPE_INT32:     mpi_type = MPI_INT; break;
            case COMPUTE_DTYPE_INT64:     mpi_type = MPI_LONG_LONG; break;
            default:                      mpi_type = MPI_BYTE; break;
        }
        MPI_Gather(send_data, (int)count, mpi_type,
                   recv_data, (int)count, mpi_type, root, ctx->comm);
    } else
#endif
    {
        size_t elem_size = compute_dtype_size(dtype);
        memcpy(recv_data, send_data, count * elem_size);
        (void)backend; (void)root;
    }
    return COMPUTE_SUCCESS;
}

static ComputeResult opencl_allgather(ComputeBackend* backend,
                                       const void* send_data, void* recv_data,
                                       size_t count, ComputeDataType dtype) {
#if COMPUTE_HAS_MPI
    OpenCLBackendContext* ctx = (OpenCLBackendContext*)backend;
    if (ctx && ctx->num_nodes > 1) {
        MPI_Datatype mpi_type;
        switch (dtype) {
            case COMPUTE_DTYPE_FLOAT32:   mpi_type = MPI_FLOAT; break;
            case COMPUTE_DTYPE_FLOAT64:   mpi_type = MPI_DOUBLE; break;
            case COMPUTE_DTYPE_INT32:     mpi_type = MPI_INT; break;
            case COMPUTE_DTYPE_INT64:     mpi_type = MPI_LONG_LONG; break;
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

static ComputeResult opencl_reduce_scatter(ComputeBackend* backend,
                                            const void* send_data, void* recv_data,
                                            size_t count, ComputeDataType dtype,
                                            ComputeReduceOp op) {
#if COMPUTE_HAS_MPI
    OpenCLBackendContext* ctx = (OpenCLBackendContext*)backend;
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
        (void)backend; (void)op;
    }
    return COMPUTE_SUCCESS;
}

// ============================================================================
// Execution & Scheduling
// ============================================================================

static ComputeResult opencl_execute(ComputeBackend* backend,
                                     const ComputeQuantumOp* op,
                                     const ComputeExecutionPlan* plan,
                                     ComputeStream* stream) {
    (void)plan;

    if (!backend || !op) return COMPUTE_ERROR_INVALID_ARGUMENT;

    switch (op->type) {
        case QUANTUM_OP_UNITARY:
            return opencl_quantum_unitary(backend,
                                          op->output_data, op->output_size,
                                          op->parameters, op->param_size,
                                          stream);

        case QUANTUM_OP_NORMALIZE:
            return opencl_quantum_normalize(backend,
                                            op->output_data, op->output_size,
                                            stream);

        case QUANTUM_OP_TENSOR_CONTRACT:
            if (op->num_dims >= 3) {
                return opencl_quantum_tensor_contract(backend,
                                                      op->output_data,
                                                      op->input_data,
                                                      op->parameters,
                                                      op->dims[0], op->dims[1], op->dims[2],
                                                      stream);
            }
            break;

        case QUANTUM_OP_GRADIENT:
            return opencl_quantum_gradient(backend,
                                           op->output_data,
                                           op->input_data,
                                           op->parameters,
                                           op->input_size,
                                           stream);

        case QUANTUM_OP_INNER_PRODUCT:
            return opencl_quantum_inner_product(backend,
                                                op->output_data,
                                                op->input_data,
                                                op->parameters,
                                                op->input_size,
                                                stream);

        case QUANTUM_OP_EXPECTATION:
            return opencl_quantum_expectation(backend,
                                              op->output_data,
                                              op->input_data,
                                              op->parameters,
                                              op->input_size,
                                              stream);

        default:
            return COMPUTE_ERROR_NOT_IMPLEMENTED;
    }

    return COMPUTE_ERROR_NOT_IMPLEMENTED;
}

static ComputeExecutionPlan* opencl_create_plan(ComputeBackend* backend,
                                                 const ComputeQuantumOp* op) {
    OpenCLBackendContext* ctx = (OpenCLBackendContext*)backend;
    if (!ctx || !op) return NULL;

    ComputeExecutionPlan* plan = calloc(1, sizeof(ComputeExecutionPlan));
    if (!plan) return NULL;

    plan->num_partitions = ctx->num_nodes;
    plan->partition_size = op->input_size / ctx->num_nodes;

    return plan;
}

static void opencl_destroy_plan(ComputeBackend* backend, ComputeExecutionPlan* plan) {
    (void)backend;
    if (plan) {
        free(plan->node_assignments);
        free(plan->offsets);
        free(plan->sizes);
        free(plan);
    }
}

// ============================================================================
// Performance Monitoring
// ============================================================================

static ComputeResult opencl_get_metrics(ComputeBackend* backend, ComputeMetrics* metrics) {
    OpenCLBackendContext* ctx = (OpenCLBackendContext*)backend;
    if (!ctx || !metrics) return COMPUTE_ERROR_INVALID_ARGUMENT;

    *metrics = ctx->metrics;
    metrics->peak_memory_bytes = ctx->peak_allocated;
    metrics->current_memory_bytes = ctx->total_allocated;

    return COMPUTE_SUCCESS;
}

static ComputeResult opencl_reset_metrics(ComputeBackend* backend) {
    OpenCLBackendContext* ctx = (OpenCLBackendContext*)backend;
    if (!ctx) return COMPUTE_ERROR_INVALID_ARGUMENT;

    memset(&ctx->metrics, 0, sizeof(ComputeMetrics));
    return COMPUTE_SUCCESS;
}

// ============================================================================
// Backend Registration
// ============================================================================

static const ComputeBackendOps opencl_ops = {
    // Lifecycle
    .init = opencl_init,
    .cleanup = opencl_cleanup,
    .probe = opencl_probe,
    .get_capabilities = opencl_get_capabilities,

    // Memory
    .alloc = opencl_alloc,
    .free = opencl_free,
    .memcpy = opencl_memcpy,
    .memset = opencl_memset,

    // Streams
    .create_stream = opencl_create_stream,
    .destroy_stream = opencl_destroy_stream,
    .synchronize_stream = opencl_synchronize_stream,
    .create_event = opencl_create_event,
    .destroy_event = opencl_destroy_event,
    .record_event = opencl_record_event,
    .wait_event = opencl_wait_event,

    // Quantum operations
    .quantum_unitary = opencl_quantum_unitary,
    .quantum_normalize = opencl_quantum_normalize,
    .quantum_tensor_contract = opencl_quantum_tensor_contract,
    .quantum_gradient = opencl_quantum_gradient,
    .quantum_inner_product = opencl_quantum_inner_product,
    .quantum_expectation = opencl_quantum_expectation,

    // Collective communication
    .barrier = opencl_barrier,
    .broadcast = opencl_broadcast,
    .allreduce = opencl_allreduce,
    .scatter = opencl_scatter,
    .gather = opencl_gather,
    .allgather = opencl_allgather,
    .reduce_scatter = opencl_reduce_scatter,

    // Execution
    .execute = opencl_execute,
    .create_plan = opencl_create_plan,
    .destroy_plan = opencl_destroy_plan,

    // Metrics
    .get_metrics = opencl_get_metrics,
    .reset_metrics = opencl_reset_metrics,
};

// Register the OpenCL backend at library load time
COMPUTE_REGISTER_BACKEND(COMPUTE_BACKEND_OPENCL, "OpenCL", "1.0.0", 50, opencl_ops)

#endif // COMPUTE_HAS_OPENCL
