#ifndef QUANTUM_GEOMETRIC_GPU_H
#define QUANTUM_GEOMETRIC_GPU_H

#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/memory_pool.h"
#include <stddef.h>

// GPU memory management
qgt_error_t gpu_malloc(void** ptr, size_t size);
qgt_error_t gpu_free(void* ptr);
qgt_error_t gpu_memcpy_host_to_device(void* dst, const void* src, size_t size);
qgt_error_t gpu_memcpy_device_to_host(void* dst, const void* src, size_t size);

// GPU memory pool integration
void gpu_free_pooled(MemoryPool* pool, void* ptr);

// GPU error codes
#define QGT_ERROR_GPU_NOT_AVAILABLE 300
#define QGT_ERROR_GPU_OUT_OF_MEMORY 301
#define QGT_ERROR_GPU_INVALID_VALUE 302
#define QGT_ERROR_GPU_LAUNCH_FAILED 303
#define QGT_ERROR_GPU_SYNC_FAILED 304
#define QGT_ERROR_GPU_INTERNAL 305

#endif // QUANTUM_GEOMETRIC_GPU_H
