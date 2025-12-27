/**
 * @file gpu_operations.c
 * @brief Unified GPU operations layer using compute_backend vtable system
 *
 * This file provides implementations for all GPU-related symbols by leveraging
 * the ComputeBackendOps vtable system. It automatically selects the best
 * available backend (Metal on macOS, CUDA on Linux, CPU fallback otherwise).
 */

#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>

#include "quantum_geometric/hardware/quantum_geometric_gpu.h"
#include "quantum_geometric/core/multi_gpu_operations.h"
#include "quantum_geometric/supercomputer/compute_backend.h"
#include "quantum_geometric/supercomputer/compute_types.h"
#include "quantum_geometric/core/memory_pool.h"

// ============================================================================
// Internal Structures
// ============================================================================

typedef struct InternalComputeContext {
    ComputeEngine* engine;
    ComputeBackend* backend;
    const ComputeBackendOps* ops;
    ComputeStream* default_stream;
    int num_devices;
    size_t total_memory;
    bool is_metal;
    bool is_cuda;
} InternalComputeContext;

typedef struct GPUMemoryPool {
    void** blocks;
    size_t* sizes;
    size_t capacity;
    size_t count;
    size_t total_allocated;
} GPUMemoryPool;

typedef struct AttentionCheckpoint {
    void* query;
    void* key;
    void* value;
    size_t size;
    size_t batch_idx;
    size_t head_idx;
    struct AttentionCheckpoint* next;
} AttentionCheckpoint;

// ============================================================================
// Global State
// ============================================================================

static InternalComputeContext g_internal_ctx = {0};
static GPUMemoryPool g_gpu_pool = {0};
static bool g_gpu_initialized = false;
static char g_last_error[512] = "";
static AttentionCheckpoint* g_attention_checkpoints = NULL;

// ============================================================================
// Error Handling
// ============================================================================

static void set_error(const char* msg) {
    if (msg) {
        strncpy(g_last_error, msg, sizeof(g_last_error) - 1);
        g_last_error[sizeof(g_last_error) - 1] = '\0';
    }
}

const char* gpu_get_last_error(void) {
    return g_last_error[0] ? g_last_error : NULL;
}

void gpu_clear_error(void) {
    g_last_error[0] = '\0';
}

// ============================================================================
// GPU Initialization and Cleanup
// ============================================================================

int gpu_initialize(void) {
    if (g_gpu_initialized) {
        return 0;
    }

    ComputeDistributedConfig config = {
        .node_rank = 0,
        .num_nodes = 1,
        .local_size = 1,
        .local_rank = 0,
        .preferred_backend = COMPUTE_BACKEND_AUTO
    };

    g_internal_ctx.engine = compute_engine_init(&config);
    if (!g_internal_ctx.engine) {
        g_gpu_initialized = true;
        g_internal_ctx.ops = NULL;
        return 0;
    }

    g_internal_ctx.ops = compute_engine_get_ops(g_internal_ctx.engine);
    g_internal_ctx.backend = compute_engine_get_backend(g_internal_ctx.engine);

    if (g_internal_ctx.ops && g_internal_ctx.backend) {
        g_internal_ctx.ops->get_capabilities(g_internal_ctx.backend,
                                              &g_internal_ctx.num_devices,
                                              &g_internal_ctx.total_memory);

        g_internal_ctx.default_stream = g_internal_ctx.ops->create_stream(g_internal_ctx.backend);

        ComputeBackendType backend_type = compute_engine_get_backend_type(g_internal_ctx.engine);
        g_internal_ctx.is_metal = (backend_type == COMPUTE_BACKEND_METAL);
        g_internal_ctx.is_cuda = (backend_type == COMPUTE_BACKEND_CUDA);
    }

    g_gpu_pool.capacity = 64;
    g_gpu_pool.blocks = calloc(g_gpu_pool.capacity, sizeof(void*));
    g_gpu_pool.sizes = calloc(g_gpu_pool.capacity, sizeof(size_t));
    g_gpu_pool.count = 0;
    g_gpu_pool.total_allocated = 0;

    g_gpu_initialized = true;
    return 0;
}

void gpu_cleanup(void) {
    if (!g_gpu_initialized) {
        return;
    }

    if (g_internal_ctx.ops && g_internal_ctx.backend) {
        for (size_t i = 0; i < g_gpu_pool.count; i++) {
            if (g_gpu_pool.blocks[i]) {
                g_internal_ctx.ops->free(g_internal_ctx.backend,
                                          g_gpu_pool.blocks[i],
                                          COMPUTE_MEM_DEVICE);
            }
        }
    }

    free(g_gpu_pool.blocks);
    free(g_gpu_pool.sizes);
    memset(&g_gpu_pool, 0, sizeof(g_gpu_pool));

    if (g_internal_ctx.ops && g_internal_ctx.default_stream) {
        g_internal_ctx.ops->destroy_stream(g_internal_ctx.backend,
                                            g_internal_ctx.default_stream);
    }

    if (g_internal_ctx.engine) {
        compute_engine_cleanup(g_internal_ctx.engine);
    }

    cleanup_attention_cache();

    memset(&g_internal_ctx, 0, sizeof(g_internal_ctx));
    g_gpu_initialized = false;
}

// ============================================================================
// GPU Device Information
// ============================================================================

int gpu_get_devices(GPUDeviceInfo* devices, int max_devices) {
    if (!devices || max_devices <= 0) return 0;

    if (!g_gpu_initialized) {
        gpu_initialize();
    }

    const char* name = g_internal_ctx.is_metal ? "Metal GPU" :
                       (g_internal_ctx.is_cuda ? "CUDA GPU" : "CPU Fallback");
    strncpy(devices[0].name, name, sizeof(devices[0].name) - 1);
    devices[0].name[sizeof(devices[0].name) - 1] = '\0';

    devices[0].total_memory = g_internal_ctx.total_memory > 0 ?
                              g_internal_ctx.total_memory : 16UL * 1024 * 1024 * 1024;
    devices[0].available_memory = devices[0].total_memory;
    devices[0].compute_units = g_internal_ctx.num_devices > 0 ?
                               g_internal_ctx.num_devices : 8;
    devices[0].backend_type = g_internal_ctx.is_metal ? GPU_BACKEND_METAL :
                              (g_internal_ctx.is_cuda ? GPU_BACKEND_CUDA : GPU_BACKEND_NONE);
    devices[0].supports_unified_memory = true;
    devices[0].supports_tensor_cores = g_internal_ctx.is_cuda;
    devices[0].supports_amx = false;
#ifdef __APPLE__
#ifdef __aarch64__
    devices[0].supports_amx = true;
#endif
#endif

    return 1;
}

// ============================================================================
// GPU Context Management
// ============================================================================

GPUContext* gpu_create_context(int device_index) {
    if (!g_gpu_initialized) {
        gpu_initialize();
    }

    GPUContext* ctx = calloc(1, sizeof(GPUContext));
    if (!ctx) {
        set_error("Failed to allocate GPU context");
        return NULL;
    }

    ctx->device_index = device_index;
    ctx->allocated_memory = 0;
    ctx->max_memory = g_internal_ctx.total_memory > 0 ?
                      g_internal_ctx.total_memory : 8UL * 1024 * 1024 * 1024;
    ctx->is_initialized = true;

    if (g_internal_ctx.backend) {
        ctx->device_handle = g_internal_ctx.backend;
        ctx->command_queue = g_internal_ctx.default_stream;
        ctx->backend_type = g_internal_ctx.is_metal ? GPU_BACKEND_METAL :
                           (g_internal_ctx.is_cuda ? GPU_BACKEND_CUDA : GPU_BACKEND_NONE);
    } else {
        ctx->device_handle = NULL;
        ctx->command_queue = NULL;
        ctx->backend_type = GPU_BACKEND_NONE;
    }

    return ctx;
}

void gpu_destroy_context(GPUContext* context) {
    if (context) {
        free(context);
    }
}

// ============================================================================
// GPU Memory Allocation (context-aware)
// ============================================================================

void* gpu_allocate(GPUContext* context, size_t size) {
    if (!context || size == 0) {
        set_error("Invalid parameters for gpu_allocate");
        return NULL;
    }

    if (!context->is_initialized) {
        set_error("GPU context not initialized");
        return NULL;
    }

    void* ptr = NULL;

    if (!g_internal_ctx.ops || !g_internal_ctx.backend) {
        ptr = malloc(size);
    } else {
        ptr = g_internal_ctx.ops->alloc(g_internal_ctx.backend, size, COMPUTE_MEM_DEVICE);
    }

    if (ptr) {
        context->allocated_memory += size;
    } else {
        set_error("GPU allocation failed");
    }

    return ptr;
}

void gpu_free(GPUContext* context, void* ptr) {
    if (!ptr) return;

    if (!g_internal_ctx.ops || !g_internal_ctx.backend) {
        free(ptr);
    } else {
        g_internal_ctx.ops->free(g_internal_ctx.backend, ptr, COMPUTE_MEM_DEVICE);
    }

    if (context) {
        // We don't track individual allocation sizes, so can't update accurately
        // This is a simplification
    }
}

// ============================================================================
// GPU Memory Copy Operations (context-aware)
// ============================================================================

int gpu_memcpy_to_device(GPUContext* context, void* dst, const void* src, size_t size) {
    if (!context || !dst || !src || size == 0) {
        set_error("Invalid parameters for gpu_memcpy_to_device");
        return -1;
    }

    if (!g_internal_ctx.ops || !g_internal_ctx.backend) {
        memcpy(dst, src, size);
        return 0;
    }

    ComputeResult result = (g_internal_ctx.ops->memcpy)(
        g_internal_ctx.backend,
        dst, COMPUTE_MEM_DEVICE,
        src, COMPUTE_MEM_HOST,
        size, NULL);

    if (result != COMPUTE_SUCCESS) {
        set_error("GPU memcpy to device failed");
        return -1;
    }
    return 0;
}

int gpu_memcpy_from_device(GPUContext* context, void* dst, const void* src, size_t size) {
    if (!context || !dst || !src || size == 0) {
        set_error("Invalid parameters for gpu_memcpy_from_device");
        return -1;
    }

    if (!g_internal_ctx.ops || !g_internal_ctx.backend) {
        memcpy(dst, src, size);
        return 0;
    }

    ComputeResult result = (g_internal_ctx.ops->memcpy)(
        g_internal_ctx.backend,
        dst, COMPUTE_MEM_HOST,
        src, COMPUTE_MEM_DEVICE,
        size, NULL);

    if (result != COMPUTE_SUCCESS) {
        set_error("GPU memcpy from device failed");
        return -1;
    }
    return 0;
}

// ============================================================================
// Async Memory Operations
// ============================================================================

void gpu_memcpy_to_device_async(void* dst, const void* src, size_t size, void* stream) {
    if (!dst || !src || size == 0) return;

    if (!g_internal_ctx.ops || !g_internal_ctx.backend) {
        memcpy(dst, src, size);
        return;
    }

    ComputeStream* cs = stream ? (ComputeStream*)stream : g_internal_ctx.default_stream;
    (g_internal_ctx.ops->memcpy)(
        g_internal_ctx.backend,
        dst, COMPUTE_MEM_DEVICE,
        src, COMPUTE_MEM_HOST,
        size, cs);
}

void gpu_memcpy_to_host_async(void* dst, const void* src, size_t size, void* stream) {
    if (!dst || !src || size == 0) return;

    if (!g_internal_ctx.ops || !g_internal_ctx.backend) {
        memcpy(dst, src, size);
        return;
    }

    ComputeStream* cs = stream ? (ComputeStream*)stream : g_internal_ctx.default_stream;
    (g_internal_ctx.ops->memcpy)(
        g_internal_ctx.backend,
        dst, COMPUTE_MEM_HOST,
        src, COMPUTE_MEM_DEVICE,
        size, cs);
}

void gpu_stream_synchronize(void* stream) {
    if (!g_internal_ctx.ops || !g_internal_ctx.backend) {
        return;
    }

    ComputeStream* cs = stream ? (ComputeStream*)stream : g_internal_ctx.default_stream;
    g_internal_ctx.ops->synchronize_stream(g_internal_ctx.backend, cs);
}

// ============================================================================
// Memory Pool Operations
// ============================================================================

void* gpu_alloc_from_pool(struct MemoryPool* pool, size_t size) {
    (void)pool;  // Pool parameter for interface compatibility

    if (!g_gpu_initialized) {
        gpu_initialize();
    }

    for (size_t i = 0; i < g_gpu_pool.count; i++) {
        if (g_gpu_pool.sizes[i] >= size && g_gpu_pool.blocks[i]) {
            void* ptr = g_gpu_pool.blocks[i];
            g_gpu_pool.blocks[i] = NULL;
            return ptr;
        }
    }

    void* ptr = NULL;
    if (!g_internal_ctx.ops || !g_internal_ctx.backend) {
        ptr = malloc(size);
    } else {
        ptr = g_internal_ctx.ops->alloc(g_internal_ctx.backend, size, COMPUTE_MEM_DEVICE);
    }

    if (ptr && g_gpu_pool.count < g_gpu_pool.capacity) {
        g_gpu_pool.total_allocated += size;
    }
    return ptr;
}

void gpu_free_to_pool(struct MemoryPool* pool, void* ptr) {
    (void)pool;
    if (!ptr) return;

    for (size_t i = 0; i < g_gpu_pool.count; i++) {
        if (!g_gpu_pool.blocks[i]) {
            g_gpu_pool.blocks[i] = ptr;
            return;
        }
    }

    if (g_gpu_pool.count < g_gpu_pool.capacity) {
        g_gpu_pool.blocks[g_gpu_pool.count] = ptr;
        g_gpu_pool.sizes[g_gpu_pool.count] = 0;
        g_gpu_pool.count++;
    } else {
        if (!g_internal_ctx.ops || !g_internal_ctx.backend) {
            free(ptr);
        } else {
            g_internal_ctx.ops->free(g_internal_ctx.backend, ptr, COMPUTE_MEM_DEVICE);
        }
    }
}

// ============================================================================
// Quantum Operations
// ============================================================================

int gpu_quantum_tensor_multiply(
    GPUContext* context,
    const ComplexFloat* a,
    const ComplexFloat* b,
    ComplexFloat* c,
    int m, int n, int k
) {
    if (!context || !a || !b || !c) {
        set_error("Invalid parameters for gpu_quantum_tensor_multiply");
        return -1;
    }

    if (!g_internal_ctx.ops || !g_internal_ctx.backend) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                float re = 0.0f, im = 0.0f;
                for (int l = 0; l < n; l++) {
                    float ar = a[i * n + l].real;
                    float ai = a[i * n + l].imag;
                    float br = b[l * k + j].real;
                    float bi = b[l * k + j].imag;
                    re += ar * br - ai * bi;
                    im += ar * bi + ai * br;
                }
                c[i * k + j].real = re;
                c[i * k + j].imag = im;
            }
        }
        return 0;
    }

    ComputeResult result = g_internal_ctx.ops->quantum_tensor_contract(
        g_internal_ctx.backend,
        (float*)c, (const float*)a, (const float*)b,
        (size_t)m, (size_t)n, (size_t)k,
        g_internal_ctx.default_stream);

    if (result != COMPUTE_SUCCESS) {
        set_error("GPU quantum tensor multiply failed");
        return -1;
    }
    return 0;
}

int gpu_quantum_geometric_transform(
    GPUContext* context,
    const ComplexFloat* input,
    ComplexFloat* output,
    const QuantumGeometricParams* params,
    size_t size
) {
    if (!context || !input || !output || !params) {
        set_error("Invalid parameters for gpu_quantum_geometric_transform");
        return -1;
    }

    if (!g_internal_ctx.ops || !g_internal_ctx.backend) {
        size_t dim = params->dimension;
        ComplexFloat* transform = params->parameters;

        for (size_t i = 0; i < size; i++) {
            float re = 0.0f, im = 0.0f;
            for (size_t j = 0; j < dim && j < size; j++) {
                float a = transform[i * dim + j].real;
                float b = transform[i * dim + j].imag;
                float c = input[j].real;
                float d = input[j].imag;
                re += a * c - b * d;
                im += a * d + b * c;
            }
            output[i].real = re;
            output[i].imag = im;
        }
        return 0;
    }

    ComputeResult result = g_internal_ctx.ops->quantum_unitary(
        g_internal_ctx.backend,
        (float*)output, size,
        (const float*)params->parameters, params->dimension,
        g_internal_ctx.default_stream);

    if (result != COMPUTE_SUCCESS) {
        set_error("GPU quantum geometric transform failed");
        return -1;
    }
    return 0;
}

int gpu_quantum_attention(
    GPUContext* context,
    const ComplexFloat* queries,
    const ComplexFloat* keys,
    const ComplexFloat* values,
    ComplexFloat* output,
    int batch_size,
    int num_heads,
    int seq_length,
    int head_dim
) {
    if (!context || !queries || !keys || !values || !output) {
        set_error("Invalid parameters for gpu_quantum_attention");
        return -1;
    }

    float scale = 1.0f / sqrtf((float)head_dim);

    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            size_t offset = ((size_t)b * num_heads + h) * seq_length * head_dim;

            for (int i = 0; i < seq_length; i++) {
                float* scores = calloc((size_t)seq_length, sizeof(float));
                if (!scores) {
                    set_error("Failed to allocate attention scores");
                    return -1;
                }

                float max_score = -1e38f;
                for (int j = 0; j < seq_length; j++) {
                    float score_re = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        size_t qi = offset + i * head_dim + d;
                        size_t ki = offset + j * head_dim + d;
                        score_re += queries[qi].real * keys[ki].real +
                                   queries[qi].imag * keys[ki].imag;
                    }
                    scores[j] = score_re * scale;
                    if (scores[j] > max_score) max_score = scores[j];
                }

                float sum = 0.0f;
                for (int j = 0; j < seq_length; j++) {
                    scores[j] = expf(scores[j] - max_score);
                    sum += scores[j];
                }
                for (int j = 0; j < seq_length; j++) {
                    scores[j] /= sum;
                }

                for (int d = 0; d < head_dim; d++) {
                    size_t oi = offset + i * head_dim + d;
                    output[oi].real = 0.0f;
                    output[oi].imag = 0.0f;
                    for (int j = 0; j < seq_length; j++) {
                        size_t vi = offset + j * head_dim + d;
                        output[oi].real += scores[j] * values[vi].real;
                        output[oi].imag += scores[j] * values[vi].imag;
                    }
                }

                free(scores);
            }
        }
    }

    return 0;
}

int gpu_batch_quantum_operations(
    GPUContext* context,
    const ComplexFloat* states,
    ComplexFloat* results,
    const QuantumOperation* operations,
    int num_states,
    int num_operations
) {
    if (!context || !states || !results || !operations) {
        set_error("Invalid parameters for gpu_batch_quantum_operations");
        return -1;
    }

    for (int s = 0; s < num_states; s++) {
        size_t state_size = 1UL << operations[0].num_qubits;
        const ComplexFloat* input = states + s * (int)state_size;
        ComplexFloat* output = results + s * (int)state_size;

        memcpy(output, input, state_size * sizeof(ComplexFloat));

        for (int o = 0; o < num_operations; o++) {
            const QuantumOperation* op = &operations[o];

            if (!op->matrix) continue;

            size_t gate_dim = op->is_controlled ? 2 : (1UL << op->num_qubits);

            ComplexFloat* temp = malloc(state_size * sizeof(ComplexFloat));
            if (!temp) {
                set_error("Failed to allocate temporary buffer");
                return -1;
            }

            memset(temp, 0, state_size * sizeof(ComplexFloat));

            for (size_t i = 0; i < state_size; i++) {
                for (size_t j = 0; j < gate_dim && j < state_size; j++) {
                    float a = op->matrix[i % gate_dim * gate_dim + j].real;
                    float b = op->matrix[i % gate_dim * gate_dim + j].imag;
                    float c = output[j].real;
                    float d = output[j].imag;
                    temp[i].real += a * c - b * d;
                    temp[i].imag += a * d + b * c;
                }
            }

            memcpy(output, temp, state_size * sizeof(ComplexFloat));
            free(temp);
        }
    }

    return 0;
}

// ============================================================================
// Performance Monitoring
// ============================================================================

int gpu_get_performance_metrics(GPUContext* context, GPUPerformanceMetrics* metrics) {
    if (!context || !metrics) {
        set_error("Invalid parameters for gpu_get_performance_metrics");
        return -1;
    }

    memset(metrics, 0, sizeof(GPUPerformanceMetrics));

    if (g_internal_ctx.ops && g_internal_ctx.backend) {
        ComputeMetrics cm;
        if (g_internal_ctx.ops->get_metrics(g_internal_ctx.backend, &cm) == COMPUTE_SUCCESS) {
            metrics->compute_time = cm.execution_time;
            metrics->memory_used = cm.memory_used;
        }
    }

    metrics->memory_used = context->allocated_memory;
    return 0;
}

// ============================================================================
// Multi-GPU Context Operations
// ============================================================================

MultiGPUContext* multi_gpu_create_context(int* device_indices, int num_devices) {
    if (!device_indices || num_devices <= 0) {
        set_error("Invalid parameters for multi_gpu_create_context");
        return NULL;
    }

    if (!g_gpu_initialized) {
        gpu_initialize();
    }

    MultiGPUContext* ctx = calloc(1, sizeof(MultiGPUContext));
    if (!ctx) {
        set_error("Failed to allocate MultiGPUContext");
        return NULL;
    }

    ctx->contexts = calloc((size_t)num_devices, sizeof(GPUContext*));
    if (!ctx->contexts) {
        free(ctx);
        set_error("Failed to allocate GPU context array");
        return NULL;
    }

    ctx->num_contexts = num_devices;
    ctx->primary_device = device_indices[0];
    ctx->synchronized = false;

    for (int i = 0; i < num_devices; i++) {
        ctx->contexts[i] = gpu_create_context(device_indices[i]);
        if (!ctx->contexts[i]) {
            for (int j = 0; j < i; j++) {
                gpu_destroy_context(ctx->contexts[j]);
            }
            free(ctx->contexts);
            free(ctx);
            return NULL;
        }
    }

    return ctx;
}

void multi_gpu_destroy_context(MultiGPUContext* ctx) {
    if (!ctx) return;

    if (ctx->contexts) {
        for (int i = 0; i < ctx->num_contexts; i++) {
            gpu_destroy_context(ctx->contexts[i]);
        }
        free(ctx->contexts);
    }
    free(ctx);
}

int multi_gpu_synchronize(MultiGPUContext* ctx) {
    if (!ctx) {
        set_error("Invalid MultiGPUContext");
        return -1;
    }

    if (g_internal_ctx.ops && g_internal_ctx.backend) {
        g_internal_ctx.ops->synchronize_stream(g_internal_ctx.backend, NULL);
    }

    ctx->synchronized = true;
    return 0;
}

MultiGPUContext* init_multi_gpu_context(void) {
    if (!g_gpu_initialized) {
        gpu_initialize();
    }

    int device_index = 0;
    return multi_gpu_create_context(&device_index, 1);
}

void sync_multi_gpu_context(MultiGPUContext* ctx) {
    multi_gpu_synchronize(ctx);
}

void cleanup_multi_gpu_context(MultiGPUContext* ctx) {
    multi_gpu_destroy_context(ctx);
}

GPUContext* get_gpu_context(MultiGPUContext* ctx, int device_id) {
    if (!ctx || device_id < 0 || device_id >= ctx->num_contexts) {
        return NULL;
    }
    return ctx->contexts[device_id];
}

// ============================================================================
// Hierarchical Matrix GPU Operations
// ============================================================================

struct HierarchicalMatrix {
    ComplexFloat* data;
    size_t rows;
    size_t cols;
    double tolerance;
    bool on_device;
    GPUContext* ctx;
};

struct HierarchicalMatrix* create_hierarchical_matrix_gpu(
    size_t rows,
    size_t cols,
    double tolerance,
    GPUContext* ctx
) {
    struct HierarchicalMatrix* hm = malloc(sizeof(struct HierarchicalMatrix));
    if (!hm) {
        set_error("Failed to allocate HierarchicalMatrix");
        return NULL;
    }

    hm->rows = rows;
    hm->cols = cols;
    hm->tolerance = tolerance;
    hm->ctx = ctx;

    size_t size = rows * cols * sizeof(ComplexFloat);

    if (ctx && ctx->is_initialized) {
        hm->data = gpu_allocate(ctx, size);
        hm->on_device = (hm->data != NULL);
    } else {
        hm->data = NULL;
        hm->on_device = false;
    }

    if (!hm->on_device) {
        hm->data = malloc(size);
        if (!hm->data) {
            free(hm);
            set_error("Failed to allocate HierarchicalMatrix data");
            return NULL;
        }
    }

    return hm;
}

struct HierarchicalMatrix* convert_to_hierarchical_gpu(
    const ComplexFloat* data,
    size_t rows,
    size_t cols,
    double tolerance,
    GPUContext* ctx
) {
    struct HierarchicalMatrix* hm = create_hierarchical_matrix_gpu(rows, cols, tolerance, ctx);
    if (!hm) return NULL;

    size_t size = rows * cols * sizeof(ComplexFloat);

    if (hm->on_device) {
        if (gpu_memcpy_to_device(ctx, hm->data, data, size) != 0) {
            destroy_hierarchical_matrix_gpu(hm, ctx);
            return NULL;
        }
    } else {
        memcpy(hm->data, data, size);
    }

    return hm;
}

void destroy_hierarchical_matrix_gpu(struct HierarchicalMatrix* matrix, GPUContext* ctx) {
    if (!matrix) return;

    if (matrix->on_device && ctx) {
        gpu_free(ctx, matrix->data);
    } else {
        free(matrix->data);
    }
    free(matrix);
}

int hierarchical_multiply_gpu(
    struct HierarchicalMatrix* result,
    const struct HierarchicalMatrix* a,
    const struct HierarchicalMatrix* b,
    GPUContext* ctx
) {
    if (!result || !a || !b || !ctx) {
        set_error("Invalid parameters for hierarchical_multiply_gpu");
        return -1;
    }

    if (a->cols != b->rows) {
        set_error("Incompatible matrix dimensions");
        return -1;
    }

    return gpu_quantum_tensor_multiply(ctx, a->data, b->data, result->data,
                                        (int)a->rows, (int)a->cols, (int)b->cols);
}

void convert_from_hierarchical_with_dropout_gpu(
    ComplexFloat* output,
    const struct HierarchicalMatrix* matrix,
    size_t output_size,
    double dropout_rate,
    GPUContext* ctx
) {
    if (!output || !matrix) return;

    size_t copy_size = (output_size < matrix->rows * matrix->cols) ?
                       output_size : matrix->rows * matrix->cols;
    size_t bytes = copy_size * sizeof(ComplexFloat);

    if (matrix->on_device && ctx) {
        gpu_memcpy_from_device(ctx, output, matrix->data, bytes);
    } else {
        memcpy(output, matrix->data, bytes);
    }

    if (dropout_rate > 0.0) {
        for (size_t i = 0; i < copy_size; i++) {
            if ((double)rand() / RAND_MAX < dropout_rate) {
                output[i].real = 0.0f;
                output[i].imag = 0.0f;
            }
        }
    }
}

// ============================================================================
// Attention Checkpoint Operations
// ============================================================================

void save_attention_checkpoint(
    void* query, void* key, void* value,
    size_t size, size_t batch_idx, size_t head_idx,
    GPUContext* ctx
) {
    (void)ctx;

    AttentionCheckpoint* cp = malloc(sizeof(AttentionCheckpoint));
    if (!cp) return;

    cp->size = size;
    cp->batch_idx = batch_idx;
    cp->head_idx = head_idx;

    cp->query = malloc(size);
    cp->key = malloc(size);
    cp->value = malloc(size);

    if (cp->query && cp->key && cp->value) {
        memcpy(cp->query, query, size);
        memcpy(cp->key, key, size);
        memcpy(cp->value, value, size);

        cp->next = g_attention_checkpoints;
        g_attention_checkpoints = cp;
    } else {
        free(cp->query);
        free(cp->key);
        free(cp->value);
        free(cp);
    }
}

void cleanup_attention_cache(void) {
    AttentionCheckpoint* cp = g_attention_checkpoints;
    while (cp) {
        AttentionCheckpoint* next = cp->next;
        free(cp->query);
        free(cp->key);
        free(cp->value);
        free(cp);
        cp = next;
    }
    g_attention_checkpoints = NULL;
}

void cleanup_attention_buffers(void) {
    cleanup_attention_cache();
}

// ============================================================================
// Metal-Specific Operations (conditionally compiled)
// ============================================================================

#ifdef ENABLE_METAL

void* get_metal_context(void) {
    if (!g_gpu_initialized) {
        gpu_initialize();
    }

    if (g_internal_ctx.is_metal) {
        return g_internal_ctx.backend;
    }
    return NULL;
}

int gpu_enable_amx(GPUContext* context) {
    if (!context) return -1;
    return 0;
}

int gpu_set_metal_compute_units(GPUContext* context, int num_units) {
    if (!context) return -1;
    (void)num_units;
    return 0;
}

int gpu_optimize_for_m1(GPUContext* context) {
    if (!context) return -1;
    return 0;
}

#endif

// ============================================================================
// CUDA-Specific Operations (conditionally compiled)
// ============================================================================

#ifdef ENABLE_CUDA

int gpu_enable_tensor_cores(GPUContext* context) {
    if (!context) return -1;
    return 0;
}

int gpu_set_cuda_stream(GPUContext* context, void* stream) {
    if (!context) return -1;
    context->command_queue = stream;
    return 0;
}

#endif

// ============================================================================
// Legacy Compatibility (internal helpers)
// ============================================================================

qgt_error_t gpu_malloc(void** ptr, size_t size) {
    if (!ptr || size == 0) {
        return QGT_ERROR_GPU_INVALID_VALUE;
    }

    if (!g_gpu_initialized) {
        gpu_initialize();
    }

    GPUContext temp_ctx = {
        .device_index = 0,
        .is_initialized = true,
        .backend_type = g_internal_ctx.is_metal ? GPU_BACKEND_METAL :
                        (g_internal_ctx.is_cuda ? GPU_BACKEND_CUDA : GPU_BACKEND_NONE)
    };

    *ptr = gpu_allocate(&temp_ctx, size);
    return *ptr ? QGT_SUCCESS : QGT_ERROR_GPU_OUT_OF_MEMORY;
}

qgt_error_t gpu_memcpy_host_to_device(void* dst, const void* src, size_t size) {
    GPUContext temp_ctx = {.device_index = 0, .is_initialized = true};
    int result = gpu_memcpy_to_device(&temp_ctx, dst, src, size);
    return (result == 0) ? QGT_SUCCESS : QGT_ERROR_GPU_INTERNAL;
}

qgt_error_t gpu_memcpy_device_to_host(void* dst, const void* src, size_t size) {
    GPUContext temp_ctx = {.device_index = 0, .is_initialized = true};
    int result = gpu_memcpy_from_device(&temp_ctx, dst, src, size);
    return (result == 0) ? QGT_SUCCESS : QGT_ERROR_GPU_INTERNAL;
}

void gpu_free_pooled(MemoryPool* pool, void* ptr) {
    gpu_free_to_pool(pool, ptr);
}
