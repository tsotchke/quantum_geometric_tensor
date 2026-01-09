/**
 * @file gpu_operations.c
 * @brief Extended GPU operations providing multi-GPU, hierarchical matrix,
 *        and attention checkpoint functionality.
 *
 * NOTE: Base GPU API (gpu_initialize, gpu_create_context, gpu_allocate, etc.)
 * is provided by quantum_geometric_gpu.c which directly dispatches to Metal/CUDA
 * backends. This file provides ADDITIONAL higher-level functionality that
 * builds on top of the base API.
 */

#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>

#include "quantum_geometric/hardware/quantum_geometric_gpu.h"
#include "quantum_geometric/core/multi_gpu_operations.h"
#include "quantum_geometric/core/memory_pool.h"
#include "quantum_geometric/core/error_codes.h"

// ============================================================================
// Internal Structures (for extended operations)
// ============================================================================

typedef struct AttentionCheckpoint {
    void* query;
    void* key;
    void* value;
    size_t size;
    size_t batch_idx;
    size_t head_idx;
    struct AttentionCheckpoint* next;
} AttentionCheckpoint;

// Local hierarchical matrix structure for GPU operations
typedef struct LocalHierarchicalMatrix {
    ComplexFloat* data;
    size_t rows;
    size_t cols;
    double tolerance;
    bool on_device;
    GPUContext* ctx;
} LocalHierarchicalMatrix;

// ============================================================================
// Global State (for extended operations)
// ============================================================================

static AttentionCheckpoint* g_attention_checkpoints = NULL;

// ============================================================================
// Async Memory Operations
// ============================================================================

void gpu_memcpy_to_device_async(void* dst, const void* src, size_t size, void* stream) {
    (void)stream;  // Stream parameter for interface compatibility
    if (!dst || !src || size == 0) return;

    // For async operations, we use the synchronous path as fallback
    // Real async would require backend-specific stream handling
    memcpy(dst, src, size);
}

void gpu_memcpy_to_host_async(void* dst, const void* src, size_t size, void* stream) {
    (void)stream;
    if (!dst || !src || size == 0) return;
    memcpy(dst, src, size);
}

void gpu_stream_synchronize(void* stream) {
    (void)stream;
    // Synchronization is implicit in our current implementation
}

// ============================================================================
// Memory Pool Operations (GPU-aware)
// ============================================================================

void* gpu_alloc_from_pool(struct MemoryPool* pool, size_t size) {
    (void)pool;  // Pool parameter for interface compatibility

    // Use standard GPU allocation path
    GPUContext* ctx = gpu_create_context(0);
    if (!ctx) {
        return malloc(size);  // Fallback to CPU
    }

    void* ptr = gpu_allocate(ctx, size);
    gpu_destroy_context(ctx);

    if (!ptr) {
        return malloc(size);  // Fallback to CPU
    }
    return ptr;
}

void gpu_free_to_pool(struct MemoryPool* pool, void* ptr) {
    (void)pool;
    if (!ptr) return;

    GPUContext* ctx = gpu_create_context(0);
    if (ctx) {
        gpu_free(ctx, ptr);
        gpu_destroy_context(ctx);
    } else {
        free(ptr);  // Fallback to CPU free
    }
}

// ============================================================================
// Multi-GPU Context Operations
// ============================================================================

MultiGPUContext* multi_gpu_create_context(int* device_indices, int num_devices) {
    if (!device_indices || num_devices <= 0) {
        return NULL;
    }

    gpu_initialize();

    MultiGPUContext* ctx = calloc(1, sizeof(MultiGPUContext));
    if (!ctx) {
        return NULL;
    }

    ctx->contexts = calloc((size_t)num_devices, sizeof(GPUContext*));
    if (!ctx->contexts) {
        free(ctx);
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
        return -1;
    }

    // Synchronize all contexts
    for (int i = 0; i < ctx->num_contexts; i++) {
        // Individual context synchronization would go here
        // For now, mark as synchronized
    }

    ctx->synchronized = true;
    return 0;
}

MultiGPUContext* init_multi_gpu_context(void) {
    gpu_initialize();
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

struct HierarchicalMatrix* create_hierarchical_matrix_gpu(
    size_t rows,
    size_t cols,
    double tolerance,
    GPUContext* ctx
) {
    LocalHierarchicalMatrix* hm = malloc(sizeof(LocalHierarchicalMatrix));
    if (!hm) {
        return NULL;
    }

    hm->rows = rows;
    hm->cols = cols;
    hm->tolerance = tolerance;
    hm->ctx = ctx;

    size_t size = rows * cols * sizeof(ComplexFloat);

    if (ctx) {
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
            return NULL;
        }
    }

    return (struct HierarchicalMatrix*)hm;
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

    LocalHierarchicalMatrix* lhm = (LocalHierarchicalMatrix*)hm;
    size_t size = rows * cols * sizeof(ComplexFloat);

    if (lhm->on_device) {
        if (gpu_memcpy_to_device(ctx, lhm->data, data, size) != 0) {
            destroy_hierarchical_matrix_gpu(hm, ctx);
            return NULL;
        }
    } else {
        memcpy(lhm->data, data, size);
    }

    return hm;
}

void destroy_hierarchical_matrix_gpu(struct HierarchicalMatrix* matrix, GPUContext* ctx) {
    if (!matrix) return;

    LocalHierarchicalMatrix* lhm = (LocalHierarchicalMatrix*)matrix;

    if (lhm->on_device && ctx) {
        gpu_free(ctx, lhm->data);
    } else {
        free(lhm->data);
    }
    free(lhm);
}

int hierarchical_multiply_gpu(
    struct HierarchicalMatrix* result,
    const struct HierarchicalMatrix* a,
    const struct HierarchicalMatrix* b,
    GPUContext* ctx
) {
    if (!result || !a || !b || !ctx) {
        return -1;
    }

    LocalHierarchicalMatrix* la = (LocalHierarchicalMatrix*)a;
    LocalHierarchicalMatrix* lb = (LocalHierarchicalMatrix*)b;
    LocalHierarchicalMatrix* lr = (LocalHierarchicalMatrix*)result;

    if (la->cols != lb->rows) {
        return -1;
    }

    return gpu_quantum_tensor_multiply(ctx, la->data, lb->data, lr->data,
                                        (int)la->rows, (int)la->cols, (int)lb->cols);
}

void convert_from_hierarchical_with_dropout_gpu(
    ComplexFloat* output,
    const struct HierarchicalMatrix* matrix,
    size_t output_size,
    double dropout_rate,
    GPUContext* ctx
) {
    if (!output || !matrix) return;

    LocalHierarchicalMatrix* lhm = (LocalHierarchicalMatrix*)matrix;

    size_t copy_size = (output_size < lhm->rows * lhm->cols) ?
                       output_size : lhm->rows * lhm->cols;
    size_t bytes = copy_size * sizeof(ComplexFloat);

    if (lhm->on_device && ctx) {
        gpu_memcpy_from_device(ctx, output, lhm->data, bytes);
    } else {
        memcpy(output, lhm->data, bytes);
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
// Legacy Compatibility Functions
// ============================================================================
// NOTE: The following functions are implemented in core/quantum_geometric_gpu.c
// with full Metal/CUDA support:
//   - gpu_malloc(void** ptr, size_t size)
//   - gpu_free(void* ptr)
//   - gpu_free_pooled(MemoryPool* pool, void* ptr)
//   - gpu_memcpy_host_to_device(void* dst, const void* src, size_t size)
//   - gpu_memcpy_device_to_host(void* dst, const void* src, size_t size)
