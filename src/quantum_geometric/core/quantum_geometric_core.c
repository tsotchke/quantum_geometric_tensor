#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_geometric_tensor.h"
#include "quantum_geometric/core/quantum_geometric_operations.h"
#include "quantum_geometric/core/quantum_geometric_constants.h"
#include "quantum_geometric/core/quantum_geometric_error.h"
#include "quantum_geometric/core/quantum_complex.h"
#include "quantum_geometric/core/quantum_geometric_memory.h"
#include "quantum_geometric/core/memory_pool.h"
#include "quantum_geometric/core/quantum_geometric_logging.h"
#include "quantum_geometric/hardware/quantum_hardware_types.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

// Constants
#define BLOCK_MAGIC 0xDEADBEEF

// Forward declarations for static helper functions
static Block* get_block(const void* ptr);

// Global state
static bool is_initialized = false;
static quantum_geometric_config_t global_config = {0};
static MemoryPool* global_pool = NULL;

// Stream and event tracking
typedef struct Stream {
    void* handle;
    bool in_use;
} Stream;

typedef struct Event {
    void* handle;
    bool recorded;
    Stream* stream;
} Event;

#define MAX_STREAMS 16
#define MAX_EVENTS 32

static Stream streams[MAX_STREAMS] = {0};
static Event events[MAX_EVENTS] = {0};

// Profiling state
static bool profiling_enabled = false;
static size_t operation_count = 0;
static size_t total_flops = 0;
static double total_time = 0.0;

// Context for maximize operation
typedef struct {
    void (*gradient)(void*, const void*);
    void (*objective)(void*, const void*);
    size_t size;
} MaximizeContext;

static MaximizeContext maximize_ctx;

// Get block from pointer with validation
static Block* get_block(const void* ptr) {
    if (!ptr) {
        return NULL;
    }
    
    // Check alignment
    if (((uintptr_t)ptr & (QG_POOL_ALIGNMENT - 1)) != 0) {
        return NULL;
    }
    
    // Get block pointer
    Block* block = (Block*)((char*)ptr - sizeof(Block));
    
    // Validate block
    if (block->magic != BLOCK_MAGIC) {
        return NULL;
    }
    
    // Validate data pointer
    if (block->data != ptr) {
        return NULL;
    }
    
    return block;
}



// Static wrapper functions for maximize operation
static void wrapped_gradient(void* grad, const void* x) {
    float* g = (float*)grad;
    maximize_ctx.gradient(grad, x);
    for (size_t i = 0; i < maximize_ctx.size; i++) {
        g[i] = -g[i];
    }
}

static void wrapped_objective(void* obj, const void* x) {
    float* o = (float*)obj;
    maximize_ctx.objective(obj, x);
    *o = -*o;
}

// Core initialization
qgt_error_t geometric_core_initialize(void) {
    // Initialize default configuration
    global_config.num_threads = 1;
    global_config.batch_size = 32;
    global_config.max_iterations = 1000;
    global_config.learning_rate = 0.001;
    global_config.convergence_threshold = 1e-8;
    global_config.use_gpu = false;
    global_config.distributed = false;
    global_config.custom_config = NULL;
    
    // Initialize memory system
    qgt_error_t error = geometric_init_memory();
    if (error != QGT_SUCCESS) {
        return error;
    }
    
    // Get memory pool instance
    global_pool = geometric_get_memory_pool();
    
    // Configure thread count based on hardware
    #ifdef _OPENMP
    global_config.num_threads = omp_get_max_threads();
    #else
    global_config.num_threads = 1;
    #endif
    
    // Initialize streams and events
    memset(streams, 0, sizeof(streams));
    memset(events, 0, sizeof(events));
    
    // Set initialized flag
    is_initialized = true;
    return QGT_SUCCESS;
}

void geometric_core_shutdown(void) {
    if (is_initialized) {
        // Clean up memory system
        geometric_cleanup_memory();
        global_pool = NULL;
        
        // Clean up any allocated resources
        if (global_config.custom_config) {
            geometric_free(global_config.custom_config);
            global_config.custom_config = NULL;
        }
        
        // Clean up streams and events
        for (int i = 0; i < MAX_STREAMS; i++) {
            if (streams[i].handle) {
                geometric_free(streams[i].handle);
                streams[i].handle = NULL;
            }
        }
        
        for (int i = 0; i < MAX_EVENTS; i++) {
            if (events[i].handle) {
                geometric_free(events[i].handle);
                events[i].handle = NULL;
            }
        }
        
        is_initialized = false;
    }
}

qgt_error_t geometric_core_reset(void) {
    geometric_core_shutdown();
    return geometric_core_initialize();
}

// Stream management
qgt_error_t geometric_core_create_stream(void** stream) {
    QGT_CHECK_NULL(stream);
    QGT_CHECK_STATE(is_initialized);
    
    // Find available stream slot
    int slot = -1;
    for (int i = 0; i < MAX_STREAMS; i++) {
        if (!streams[i].handle && !streams[i].in_use) {
            slot = i;
            break;
        }
    }
    
    if (slot == -1) {
        return QGT_ERROR_RESOURCE_EXHAUSTED;
    }
    
    // Allocate stream handle
    streams[slot].handle = geometric_allocate(sizeof(void*));
    if (!streams[slot].handle) {
        return QGT_ERROR_MEMORY_ALLOCATION;
    }
    
    streams[slot].in_use = true;
    *stream = streams[slot].handle;
    
    return QGT_SUCCESS;
}

void geometric_core_destroy_stream(void* stream) {
    if (!stream || !is_initialized) return;
    
    for (int i = 0; i < MAX_STREAMS; i++) {
        if (streams[i].handle == stream) {
            geometric_free(streams[i].handle);
            streams[i].handle = NULL;
            streams[i].in_use = false;
            break;
        }
    }
}

qgt_error_t geometric_core_synchronize_stream(void* stream) {
    QGT_CHECK_NULL(stream);
    QGT_CHECK_STATE(is_initialized);
    
    // Find stream
    bool found = false;
    for (int i = 0; i < MAX_STREAMS; i++) {
        if (streams[i].handle == stream) {
            found = true;
            break;
        }
    }
    
    if (!found) {
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    // CPU implementation is synchronous
    return QGT_SUCCESS;
}

// Event management
qgt_error_t geometric_core_create_event(void** event) {
    QGT_CHECK_NULL(event);
    QGT_CHECK_STATE(is_initialized);
    
    // Find available event slot
    int slot = -1;
    for (int i = 0; i < MAX_EVENTS; i++) {
        if (!events[i].handle) {
            slot = i;
            break;
        }
    }
    
    if (slot == -1) {
        return QGT_ERROR_RESOURCE_EXHAUSTED;
    }
    
    // Allocate event handle
    events[slot].handle = geometric_allocate(sizeof(void*));
    if (!events[slot].handle) {
        return QGT_ERROR_MEMORY_ALLOCATION;
    }
    
    events[slot].recorded = false;
    events[slot].stream = NULL;
    *event = events[slot].handle;
    
    return QGT_SUCCESS;
}

void geometric_core_destroy_event(void* event) {
    if (!event || !is_initialized) return;
    
    for (int i = 0; i < MAX_EVENTS; i++) {
        if (events[i].handle == event) {
            geometric_free(events[i].handle);
            events[i].handle = NULL;
            events[i].recorded = false;
            events[i].stream = NULL;
            break;
        }
    }
}

qgt_error_t geometric_core_record_event(void* event, void* stream) {
    QGT_CHECK_NULL(event);
    QGT_CHECK_NULL(stream);
    QGT_CHECK_STATE(is_initialized);
    
    // Find event and stream
    Event* evt = NULL;
    Stream* str = NULL;
    
    for (int i = 0; i < MAX_EVENTS; i++) {
        if (events[i].handle == event) {
            evt = &events[i];
            break;
        }
    }
    
    for (int i = 0; i < MAX_STREAMS; i++) {
        if (streams[i].handle == stream) {
            str = &streams[i];
            break;
        }
    }
    
    if (!evt || !str) {
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    evt->recorded = true;
    evt->stream = str;
    
    return QGT_SUCCESS;
}

qgt_error_t geometric_core_synchronize_event(void* event) {
    QGT_CHECK_NULL(event);
    QGT_CHECK_STATE(is_initialized);
    
    // Find event
    Event* evt = NULL;
    for (int i = 0; i < MAX_EVENTS; i++) {
        if (events[i].handle == event) {
            evt = &events[i];
            break;
        }
    }
    
    if (!evt) {
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    if (!evt->recorded) {
        return QGT_ERROR_INVALID_STATE;
    }
    
    // CPU implementation is synchronous
    return QGT_SUCCESS;
}

// Core linear algebra
qgt_error_t geometric_core_matrix_multiply(void* result,
                                         const void* a,
                                         const void* b,
                                         size_t m,
                                         size_t n,
                                         size_t k) {
    QGT_CHECK_NULL(result);
    QGT_CHECK_NULL(a);
    QGT_CHECK_NULL(b);
    QGT_CHECK_STATE(is_initialized);
    
    float* C = (float*)result;
    const float* A = (const float*)a;
    const float* B = (const float*)b;
    
    // Initialize result matrix to zero
    memset(C, 0, m * n * sizeof(float));
    
    // Basic matrix multiplication with cache blocking
    const size_t block_size = 32;
    
    for (size_t i0 = 0; i0 < m; i0 += block_size) {
        size_t imax = (i0 + block_size < m) ? i0 + block_size : m;
        for (size_t j0 = 0; j0 < n; j0 += block_size) {
            size_t jmax = (j0 + block_size < n) ? j0 + block_size : n;
            for (size_t k0 = 0; k0 < k; k0 += block_size) {
                size_t kmax = (k0 + block_size < k) ? k0 + block_size : k;
                
                for (size_t i = i0; i < imax; i++) {
                    for (size_t j = j0; j < jmax; j++) {
                        float sum = C[i * n + j];
                        for (size_t l = k0; l < kmax; l++) {
                            sum += A[i * k + l] * B[l * n + j];
                        }
                        C[i * n + j] = sum;
                    }
                }
            }
        }
    }
    
    if (profiling_enabled) {
        operation_count++;
        total_flops += 2 * m * n * k;
    }
    
    return QGT_SUCCESS;
}

qgt_error_t geometric_core_matrix_transpose(void* result,
                                          const void* matrix,
                                          size_t rows,
                                          size_t cols) {
    QGT_CHECK_NULL(result);
    QGT_CHECK_NULL(matrix);
    QGT_CHECK_STATE(is_initialized);
    
    float* out = (float*)result;
    const float* in = (const float*)matrix;
    
    // Cache-friendly blocked transpose
    const size_t block_size = 32;
    
    for (size_t i0 = 0; i0 < rows; i0 += block_size) {
        size_t imax = (i0 + block_size < rows) ? i0 + block_size : rows;
        for (size_t j0 = 0; j0 < cols; j0 += block_size) {
            size_t jmax = (j0 + block_size < cols) ? j0 + block_size : cols;
            
            for (size_t i = i0; i < imax; i++) {
                for (size_t j = j0; j < jmax; j++) {
                    out[j * rows + i] = in[i * cols + j];
                }
            }
        }
    }
    
    if (profiling_enabled) {
        operation_count++;
        total_flops += rows * cols;
    }
    
    return QGT_SUCCESS;
}

qgt_error_t geometric_core_matrix_inverse(void* result,
                                        const void* matrix,
                                        size_t size) {
    QGT_CHECK_NULL(result);
    QGT_CHECK_NULL(matrix);
    QGT_CHECK_STATE(is_initialized);
    
    float* out = (float*)result;
    const float* in = (const float*)matrix;
    
    // Copy input matrix
    memcpy(out, in, size * size * sizeof(float));
    
    // Create identity matrix
    float* identity = (float*)geometric_allocate(size * size * sizeof(float));
    if (!identity) {
        return QGT_ERROR_MEMORY_ALLOCATION;
    }
    
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            identity[i * size + j] = (i == j) ? 1.0f : 0.0f;
        }
    }
    
    // Gaussian elimination with partial pivoting
    for (size_t i = 0; i < size; i++) {
        // Find pivot
        size_t pivot = i;
        float max_val = fabsf(out[i * size + i]);
        
        for (size_t j = i + 1; j < size; j++) {
            float val = fabsf(out[j * size + i]);
            if (val > max_val) {
                max_val = val;
                pivot = j;
            }
        }
        
        if (max_val < 1e-6f) {
            geometric_free(identity);
            return QGT_ERROR_INVALID_PARAMETER;
        }
        
        // Swap rows if needed
        if (pivot != i) {
            for (size_t j = 0; j < size; j++) {
                float temp = out[i * size + j];
                out[i * size + j] = out[pivot * size + j];
                out[pivot * size + j] = temp;
                
                temp = identity[i * size + j];
                identity[i * size + j] = identity[pivot * size + j];
                identity[pivot * size + j] = temp;
            }
        }
        
        // Scale row
        float scale = 1.0f / out[i * size + i];
        for (size_t j = 0; j < size; j++) {
            out[i * size + j] *= scale;
            identity[i * size + j] *= scale;
        }
        
        // Eliminate column
        for (size_t j = 0; j < size; j++) {
            if (j != i) {
                float factor = out[j * size + i];
                for (size_t k = 0; k < size; k++) {
                    out[j * size + k] -= factor * out[i * size + k];
                    identity[j * size + k] -= factor * identity[i * size + k];
                }
            }
        }
    }
    
    // Copy result
    memcpy(out, identity, size * size * sizeof(float));
    geometric_free(identity);
    
    if (profiling_enabled) {
        operation_count++;
        total_flops += size * size * size;
    }
    
    return QGT_SUCCESS;
}

qgt_error_t geometric_core_solve_linear_system(void* result,
                                             const void* a,
                                             const void* b,
                                             size_t size) {
    QGT_CHECK_NULL(result);
    QGT_CHECK_NULL(a);
    QGT_CHECK_NULL(b);
    QGT_CHECK_STATE(is_initialized);
    
    // Solve using LU decomposition
    float* L = (float*)geometric_allocate(size * size * sizeof(float));
    float* U = (float*)geometric_allocate(size * size * sizeof(float));
    float* y = (float*)geometric_allocate(size * sizeof(float));
    
    if (!L || !U || !y) {
        geometric_free(L);
        geometric_free(U);
        geometric_free(y);
        return QGT_ERROR_MEMORY_ALLOCATION;
    }
    
    const float* A = (const float*)a;
    const float* B = (const float*)b;
    float* x = (float*)result;
    
    // LU decomposition
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            if (j < i) {
                L[i * size + j] = A[i * size + j];
                for (size_t k = 0; k < j; k++) {
                    L[i * size + j] -= L[i * size + k] * U[k * size + j];
                }
                U[i * size + j] = 0;
            }
            else if (j == i) {
                L[i * size + j] = 1;
                U[i * size + j] = A[i * size + j];
                for (size_t k = 0; k < i; k++) {
                    U[i * size + j] -= L[i * size + k] * U[k * size + j];
                }
            }
            else {
                U[i * size + j] = A[i * size + j];
                for (size_t k = 0; k < i; k++) {
                    U[i * size + j] -= L[i * size + k] * U[k * size + j];
                }
                L[i * size + j] = 0;
            }
        }
    }
    
    // Forward substitution Ly = b
    for (size_t i = 0; i < size; i++) {
        y[i] = B[i];
        for (size_t j = 0; j < i; j++) {
            y[i] -= L[i * size + j] * y[j];
        }
    }
    
    // Back substitution Ux = y
    for (size_t i = size; i-- > 0;) {
        x[i] = y[i];
        for (size_t j = i + 1; j < size; j++) {
            x[i] -= U[i * size + j] * x[j];
        }
        x[i] /= U[i * size + i];
    }
    
    geometric_free(L);
    geometric_free(U);
    geometric_free(y);
    
    if (profiling_enabled) {
        operation_count++;
        total_flops += size * size * size;
    }
    
    return QGT_SUCCESS;
}

// Core optimization
qgt_error_t geometric_core_minimize(void* result,
                                  void (*objective)(void*, const void*),
                                  void (*gradient)(void*, const void*),
                                  const void* initial,
                                  size_t size,
                                  size_t max_iterations) {
    QGT_CHECK_NULL(result);
    QGT_CHECK_NULL(objective);
    QGT_CHECK_NULL(gradient);
    QGT_CHECK_NULL(initial);
    QGT_CHECK_STATE(is_initialized);
    
    float* x = (float*)result;
    const float* x0 = (const float*)initial;
    
    // Copy initial guess
    memcpy(x, x0, size * sizeof(float));
    
    // Gradient descent with adaptive learning rate
    float learning_rate = global_config.learning_rate;
    float* grad = (float*)geometric_allocate(size * sizeof(float));
    if (!grad) {
        return QGT_ERROR_MEMORY_ALLOCATION;
    }
    
    for (size_t iter = 0; iter < max_iterations; iter++) {
        // Compute gradient
        gradient(grad, x);
        
        // Update parameters
        float max_grad = 0.0f;
        for (size_t i = 0; i < size; i++) {
            float g = grad[i];
            max_grad = fmaxf(max_grad, fabsf(g));
            x[i] -= learning_rate * g;
        }
        
        // Check convergence
        if (max_grad < global_config.convergence_threshold) {
            geometric_free(grad);
            return QGT_SUCCESS;
        }
        
        // Adaptive learning rate
        if (iter % 10 == 0) {
            float old_obj;
            objective(&old_obj, x);
            
            // Try increasing learning rate
            float* temp = (float*)geometric_allocate(size * sizeof(float));
            if (!temp) {
                geometric_free(grad);
                return QGT_ERROR_MEMORY_ALLOCATION;
            }
            
            memcpy(temp, x, size * sizeof(float));
            for (size_t i = 0; i < size; i++) {
                temp[i] -= 2 * learning_rate * grad[i];
            }
            
            float new_obj;
            objective(&new_obj, temp);
            
            if (new_obj < old_obj) {
                learning_rate *= 1.1f;
                memcpy(x, temp, size * sizeof(float));
            } else {
                learning_rate *= 0.9f;
            }
            
            geometric_free(temp);
        }
    }
    
    geometric_free(grad);
    return QGT_ERROR_TIMEOUT;
}


qgt_error_t geometric_core_maximize(void* result,
                                  void (*objective)(void*, const void*),
                                  void (*gradient)(void*, const void*),
                                  const void* initial,
                                  size_t size,
                                  size_t max_iterations) {
    QGT_CHECK_NULL(result);
    QGT_CHECK_NULL(objective);
    QGT_CHECK_NULL(gradient);
    QGT_CHECK_NULL(initial);
    QGT_CHECK_STATE(is_initialized);
    
    // Update maximize context
    maximize_ctx.gradient = gradient;
    maximize_ctx.objective = objective;
    maximize_ctx.size = size;
    
    qgt_error_t error = geometric_core_minimize(result, wrapped_objective, wrapped_gradient, initial, size, max_iterations);
    return error;
}

// Core validation
qgt_error_t geometric_core_validate_memory(const void* ptr, size_t size) {
    QGT_CHECK_NULL(ptr);
    QGT_CHECK_ARGUMENT(size > 0);
    QGT_CHECK_STATE(is_initialized);
    
    // Validate memory block
    Block* block = get_block(ptr);
    if (!block) {
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    // Validate size
    if (block->size < size) {
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    return QGT_SUCCESS;
}

qgt_error_t geometric_core_validate_device(size_t device) {
    QGT_CHECK_STATE(is_initialized);
    
    // For now, only CPU (device 0) is supported
    if (device != 0) {
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    return QGT_SUCCESS;
}

qgt_error_t geometric_core_validate_stream(const void* stream) {
    QGT_CHECK_NULL(stream);
    QGT_CHECK_STATE(is_initialized);
    
    // Check if stream exists
    bool found = false;
    for (int i = 0; i < MAX_STREAMS; i++) {
        if (streams[i].handle == stream && streams[i].in_use) {
            found = true;
            break;
        }
    }
    
    if (!found) {
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    return QGT_SUCCESS;
}

qgt_error_t geometric_core_validate_event(const void* event) {
    QGT_CHECK_NULL(event);
    QGT_CHECK_STATE(is_initialized);
    
    // Check if event exists
    bool found = false;
    for (int i = 0; i < MAX_EVENTS; i++) {
        if (events[i].handle == event) {
            found = true;
            break;
        }
    }
    
    if (!found) {
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    return QGT_SUCCESS;
}

// Core profiling
qgt_error_t geometric_core_start_profiling(void) {
    QGT_CHECK_STATE(is_initialized);
    
    profiling_enabled = true;
    operation_count = 0;
    total_flops = 0;
    total_time = 0.0;
    
    return QGT_SUCCESS;
}

qgt_error_t geometric_core_stop_profiling(void) {
    QGT_CHECK_STATE(is_initialized);
    
    profiling_enabled = false;
    
    return QGT_SUCCESS;
}

qgt_error_t geometric_core_reset_profiling(void) {
    QGT_CHECK_STATE(is_initialized);
    
    operation_count = 0;
    total_flops = 0;
    total_time = 0.0;
    
    return QGT_SUCCESS;
}

qgt_error_t geometric_core_get_profile_data(void* data, size_t* size) {
    QGT_CHECK_NULL(data);
    QGT_CHECK_NULL(size);
    QGT_CHECK_STATE(is_initialized);
    
    // Profile data structure
    typedef struct {
        size_t operations;
        size_t flops;
        double time;
        double flops_per_second;
    } ProfileData;
    
    if (*size < sizeof(ProfileData)) {
        *size = sizeof(ProfileData);
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    ProfileData* profile = (ProfileData*)data;
    profile->operations = operation_count;
    profile->flops = total_flops;
    profile->time = total_time;
    profile->flops_per_second = total_time > 0 ? (double)total_flops / total_time : 0;
    
    *size = sizeof(ProfileData);
    
    return QGT_SUCCESS;
}

// Memory management
qgt_error_t geometric_core_allocate(void** ptr, size_t size) {
    QGT_CHECK_NULL(ptr);
    QGT_CHECK_ARGUMENT(size > 0);
    QGT_CHECK_STATE(is_initialized);
    
    // Ensure size is properly aligned for quantum operations
    size_t aligned_size = (size + QG_POOL_ALIGNMENT - 1) & ~(QG_POOL_ALIGNMENT - 1);
    
    void* allocated = geometric_allocate(aligned_size);
    if (!allocated) {
        geometric_log_error("Failed to allocate %zu bytes", aligned_size);
        return QGT_ERROR_MEMORY_ALLOCATION;
    }
    
    // Validate the allocated block
    Block* block = get_block(allocated);
    if (!block) {
        geometric_log_error("Invalid memory block after allocation");
        geometric_free(allocated);
        return QGT_ERROR_MEMORY_ALLOCATION;
    }
    
    // Prefetch next likely allocation
    __builtin_prefetch((char*)allocated + aligned_size, 1, 3);
    
    *ptr = allocated;
    return QGT_SUCCESS;
}

void geometric_core_free(void* ptr) {
    if (!ptr || !is_initialized) return;
    
    // Validate the block before freeing
    Block* block = get_block(ptr);
    if (!block) {
        geometric_log_error("Invalid memory block in free");
        return;
    }
    
    if (block->is_free) {
        geometric_log_error("Double free detected");
        return;
    }
    
    geometric_free(ptr);
}

qgt_error_t geometric_core_memcpy(void* dest, const void* src, size_t size) {
    QGT_CHECK_NULL(dest);
    QGT_CHECK_NULL(src);
    QGT_CHECK_ARGUMENT(size > 0);
    QGT_CHECK_STATE(is_initialized);
    
    // Validate source and destination memory blocks
    Block* src_block = get_block((void*)src);
    Block* dest_block = get_block(dest);
    
    if (!src_block || !dest_block) {
        geometric_log_error("Invalid memory block in memcpy");
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    // Validate block states
    if (src_block->is_free || dest_block->is_free) {
        geometric_log_error("Attempt to access freed memory");
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    // Validate sizes
    if (src_block->size < size || dest_block->size < size) {
        geometric_log_error("Memory block size too small for requested copy");
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    // Validate alignment for quantum state operations
    if (((uintptr_t)dest & (QG_POOL_ALIGNMENT - 1)) != 0 ||
        ((uintptr_t)src & (QG_POOL_ALIGNMENT - 1)) != 0) {
        geometric_log_error("Memory not properly aligned for quantum operations");
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    // Use SIMD operations for large copies if available
    #ifdef __AVX512F__
    if (size >= 64 && ((uintptr_t)dest & 63) == 0 && ((uintptr_t)src & 63) == 0) {
        size_t simd_size = size & ~63;
        for (size_t i = 0; i < simd_size; i += 64) {
            __m512i data = _mm512_load_si512((__m512i*)((char*)src + i));
            _mm512_store_si512((__m512i*)((char*)dest + i), data);
        }
        // Handle remaining bytes
        memcpy((char*)dest + simd_size, (char*)src + simd_size, size - simd_size);
    } else
    #elif defined(__ARM_NEON)
    if (size >= 16 && ((uintptr_t)dest & 15) == 0 && ((uintptr_t)src & 15) == 0) {
        size_t simd_size = size & ~15;
        for (size_t i = 0; i < simd_size; i += 16) {
            uint8x16_t data = vld1q_u8((uint8_t*)((char*)src + i));
            vst1q_u8((uint8_t*)((char*)dest + i), data);
        }
        // Handle remaining bytes
        memcpy((char*)dest + simd_size, (char*)src + simd_size, size - simd_size);
    } else
    #endif
    {
        memcpy(dest, src, size);
    }
    
    return QGT_SUCCESS;
}

qgt_error_t geometric_core_memset(void* ptr, int value, size_t size) {
    QGT_CHECK_NULL(ptr);
    QGT_CHECK_ARGUMENT(size > 0);
    QGT_CHECK_STATE(is_initialized);
    
    // Validate memory block
    Block* block = get_block(ptr);
    if (!block) {
        geometric_log_error("Invalid memory block in memset");
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    // Validate block state
    if (block->is_free) {
        geometric_log_error("Attempt to access freed memory");
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    // Validate size
    if (block->size < size) {
        geometric_log_error("Memory block size too small for requested set");
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    // Validate alignment for quantum state operations
    if ((uintptr_t)ptr & (QG_POOL_ALIGNMENT - 1)) {
        geometric_log_error("Memory not properly aligned for quantum operations");
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    // Use SIMD operations for large sets if available
    #ifdef __AVX512F__
    if (size >= 64 && ((uintptr_t)ptr & 63) == 0) {
        size_t simd_size = size & ~63;
        __m512i val = _mm512_set1_epi8(value);
        for (size_t i = 0; i < simd_size; i += 64) {
            _mm512_store_si512((__m512i*)((char*)ptr + i), val);
        }
        // Handle remaining bytes
        memset((char*)ptr + simd_size, value, size - simd_size);
    } else
    #elif defined(__ARM_NEON)
    if (size >= 16 && ((uintptr_t)ptr & 15) == 0) {
        size_t simd_size = size & ~15;
        uint8x16_t val = vdupq_n_u8(value);
        for (size_t i = 0; i < simd_size; i += 16) {
            vst1q_u8((uint8_t*)((char*)ptr + i), val);
        }
        // Handle remaining bytes
        memset((char*)ptr + simd_size, value, size - simd_size);
    } else
    #endif
    {
        memset(ptr, value, size);
    }
    
    return QGT_SUCCESS;
}

qgt_error_t geometric_core_get_memory_stats(size_t* total, size_t* peak, size_t* count) {
    QGT_CHECK_NULL(total);
    QGT_CHECK_NULL(peak);
    QGT_CHECK_NULL(count);
    QGT_CHECK_STATE(is_initialized);

    geometric_get_memory_stats(total, peak, count);

    return QGT_SUCCESS;
}

// ============================================================================
// Device Management Implementation
// ============================================================================

// Current device state
static size_t current_device = 0;
static qgt_error_t last_error = QGT_SUCCESS;
static bool debug_mode = false;
static size_t log_level = 0;

qgt_error_t geometric_core_get_device_count(size_t* count) {
    QGT_CHECK_NULL(count);
    QGT_CHECK_STATE(is_initialized);

    // Currently only CPU device is supported (device 0)
    // GPU support would enumerate CUDA/Metal devices here
    *count = 1;

    #ifdef CUDA_AVAILABLE
    int cuda_count = 0;
    if (cudaGetDeviceCount(&cuda_count) == cudaSuccess) {
        *count += cuda_count;
    }
    #endif

    #ifdef __APPLE__
    // Metal device is always available on macOS
    *count = 1;  // Metal GPU
    #endif

    return QGT_SUCCESS;
}

qgt_error_t geometric_core_set_device(size_t device) {
    QGT_CHECK_STATE(is_initialized);

    size_t device_count = 0;
    qgt_error_t err = geometric_core_get_device_count(&device_count);
    if (err != QGT_SUCCESS) {
        return err;
    }

    if (device >= device_count) {
        last_error = QGT_ERROR_INVALID_PARAMETER;
        return QGT_ERROR_INVALID_PARAMETER;
    }

    current_device = device;

    #ifdef CUDA_AVAILABLE
    if (device > 0) {
        cudaSetDevice(device - 1);  // device 0 is CPU
    }
    #endif

    return QGT_SUCCESS;
}

qgt_error_t geometric_core_get_device_properties(size_t device, void* properties) {
    QGT_CHECK_NULL(properties);
    QGT_CHECK_STATE(is_initialized);

    // Device properties structure
    typedef struct {
        char name[256];
        size_t total_memory;
        size_t available_memory;
        size_t compute_units;
        size_t max_threads;
        bool supports_double;
        bool supports_tensor_cores;
    } DeviceProperties;

    DeviceProperties* props = (DeviceProperties*)properties;

    if (device == 0) {
        // CPU device properties
        strncpy(props->name, "CPU (Host)", sizeof(props->name) - 1);

        #ifdef __linux__
        struct sysinfo si;
        if (sysinfo(&si) == 0) {
            props->total_memory = si.totalram * si.mem_unit;
            props->available_memory = si.freeram * si.mem_unit;
        }
        #elif defined(__APPLE__)
        size_t size = sizeof(size_t);
        sysctlbyname("hw.memsize", &props->total_memory, &size, NULL, 0);
        props->available_memory = props->total_memory / 2;  // Estimate
        #else
        props->total_memory = 8ULL * 1024 * 1024 * 1024;  // Default 8GB
        props->available_memory = props->total_memory / 2;
        #endif

        #ifdef _OPENMP
        props->compute_units = omp_get_max_threads();
        props->max_threads = omp_get_max_threads();
        #else
        props->compute_units = 1;
        props->max_threads = 1;
        #endif

        props->supports_double = true;
        props->supports_tensor_cores = false;
    }
    #ifdef CUDA_AVAILABLE
    else {
        cudaDeviceProp cuda_props;
        cudaGetDeviceProperties(&cuda_props, device - 1);

        strncpy(props->name, cuda_props.name, sizeof(props->name) - 1);
        props->total_memory = cuda_props.totalGlobalMem;
        props->compute_units = cuda_props.multiProcessorCount;
        props->max_threads = cuda_props.maxThreadsPerBlock;
        props->supports_double = (cuda_props.major >= 2);
        props->supports_tensor_cores = (cuda_props.major >= 7);

        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        props->available_memory = free_mem;
    }
    #endif

    return QGT_SUCCESS;
}

qgt_error_t geometric_core_synchronize_device(void) {
    QGT_CHECK_STATE(is_initialized);

    // CPU is always synchronized
    if (current_device == 0) {
        return QGT_SUCCESS;
    }

    #ifdef CUDA_AVAILABLE
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        last_error = QGT_ERROR_DEVICE;
        return QGT_ERROR_DEVICE;
    }
    #endif

    return QGT_SUCCESS;
}

// ============================================================================
// Core Element-wise Operations
// ============================================================================

qgt_error_t geometric_core_add(void* result, const void* a, const void* b, size_t size) {
    QGT_CHECK_NULL(result);
    QGT_CHECK_NULL(a);
    QGT_CHECK_NULL(b);
    QGT_CHECK_ARGUMENT(size > 0);
    QGT_CHECK_STATE(is_initialized);

    float* out = (float*)result;
    const float* in_a = (const float*)a;
    const float* in_b = (const float*)b;

    #ifdef __ARM_NEON
    size_t simd_size = size & ~3;
    for (size_t i = 0; i < simd_size; i += 4) {
        float32x4_t va = vld1q_f32(in_a + i);
        float32x4_t vb = vld1q_f32(in_b + i);
        float32x4_t vr = vaddq_f32(va, vb);
        vst1q_f32(out + i, vr);
    }
    for (size_t i = simd_size; i < size; i++) {
        out[i] = in_a[i] + in_b[i];
    }
    #elif defined(__AVX512F__)
    size_t simd_size = size & ~15;
    for (size_t i = 0; i < simd_size; i += 16) {
        __m512 va = _mm512_load_ps(in_a + i);
        __m512 vb = _mm512_load_ps(in_b + i);
        __m512 vr = _mm512_add_ps(va, vb);
        _mm512_store_ps(out + i, vr);
    }
    for (size_t i = simd_size; i < size; i++) {
        out[i] = in_a[i] + in_b[i];
    }
    #else
    #pragma omp parallel for if(size > 1000)
    for (size_t i = 0; i < size; i++) {
        out[i] = in_a[i] + in_b[i];
    }
    #endif

    if (profiling_enabled) {
        operation_count++;
        total_flops += size;
    }

    return QGT_SUCCESS;
}

qgt_error_t geometric_core_subtract(void* result, const void* a, const void* b, size_t size) {
    QGT_CHECK_NULL(result);
    QGT_CHECK_NULL(a);
    QGT_CHECK_NULL(b);
    QGT_CHECK_ARGUMENT(size > 0);
    QGT_CHECK_STATE(is_initialized);

    float* out = (float*)result;
    const float* in_a = (const float*)a;
    const float* in_b = (const float*)b;

    #ifdef __ARM_NEON
    size_t simd_size = size & ~3;
    for (size_t i = 0; i < simd_size; i += 4) {
        float32x4_t va = vld1q_f32(in_a + i);
        float32x4_t vb = vld1q_f32(in_b + i);
        float32x4_t vr = vsubq_f32(va, vb);
        vst1q_f32(out + i, vr);
    }
    for (size_t i = simd_size; i < size; i++) {
        out[i] = in_a[i] - in_b[i];
    }
    #elif defined(__AVX512F__)
    size_t simd_size = size & ~15;
    for (size_t i = 0; i < simd_size; i += 16) {
        __m512 va = _mm512_load_ps(in_a + i);
        __m512 vb = _mm512_load_ps(in_b + i);
        __m512 vr = _mm512_sub_ps(va, vb);
        _mm512_store_ps(out + i, vr);
    }
    for (size_t i = simd_size; i < size; i++) {
        out[i] = in_a[i] - in_b[i];
    }
    #else
    #pragma omp parallel for if(size > 1000)
    for (size_t i = 0; i < size; i++) {
        out[i] = in_a[i] - in_b[i];
    }
    #endif

    if (profiling_enabled) {
        operation_count++;
        total_flops += size;
    }

    return QGT_SUCCESS;
}

qgt_error_t geometric_core_multiply(void* result, const void* a, const void* b, size_t size) {
    QGT_CHECK_NULL(result);
    QGT_CHECK_NULL(a);
    QGT_CHECK_NULL(b);
    QGT_CHECK_ARGUMENT(size > 0);
    QGT_CHECK_STATE(is_initialized);

    float* out = (float*)result;
    const float* in_a = (const float*)a;
    const float* in_b = (const float*)b;

    #ifdef __ARM_NEON
    size_t simd_size = size & ~3;
    for (size_t i = 0; i < simd_size; i += 4) {
        float32x4_t va = vld1q_f32(in_a + i);
        float32x4_t vb = vld1q_f32(in_b + i);
        float32x4_t vr = vmulq_f32(va, vb);
        vst1q_f32(out + i, vr);
    }
    for (size_t i = simd_size; i < size; i++) {
        out[i] = in_a[i] * in_b[i];
    }
    #elif defined(__AVX512F__)
    size_t simd_size = size & ~15;
    for (size_t i = 0; i < simd_size; i += 16) {
        __m512 va = _mm512_load_ps(in_a + i);
        __m512 vb = _mm512_load_ps(in_b + i);
        __m512 vr = _mm512_mul_ps(va, vb);
        _mm512_store_ps(out + i, vr);
    }
    for (size_t i = simd_size; i < size; i++) {
        out[i] = in_a[i] * in_b[i];
    }
    #else
    #pragma omp parallel for if(size > 1000)
    for (size_t i = 0; i < size; i++) {
        out[i] = in_a[i] * in_b[i];
    }
    #endif

    if (profiling_enabled) {
        operation_count++;
        total_flops += size;
    }

    return QGT_SUCCESS;
}

qgt_error_t geometric_core_divide(void* result, const void* a, const void* b, size_t size) {
    QGT_CHECK_NULL(result);
    QGT_CHECK_NULL(a);
    QGT_CHECK_NULL(b);
    QGT_CHECK_ARGUMENT(size > 0);
    QGT_CHECK_STATE(is_initialized);

    float* out = (float*)result;
    const float* in_a = (const float*)a;
    const float* in_b = (const float*)b;

    #ifdef __ARM_NEON
    size_t simd_size = size & ~3;
    for (size_t i = 0; i < simd_size; i += 4) {
        float32x4_t va = vld1q_f32(in_a + i);
        float32x4_t vb = vld1q_f32(in_b + i);
        float32x4_t vr = vdivq_f32(va, vb);
        vst1q_f32(out + i, vr);
    }
    for (size_t i = simd_size; i < size; i++) {
        if (fabsf(in_b[i]) < 1e-10f) {
            out[i] = (in_a[i] >= 0) ? INFINITY : -INFINITY;
        } else {
            out[i] = in_a[i] / in_b[i];
        }
    }
    #elif defined(__AVX512F__)
    size_t simd_size = size & ~15;
    for (size_t i = 0; i < simd_size; i += 16) {
        __m512 va = _mm512_load_ps(in_a + i);
        __m512 vb = _mm512_load_ps(in_b + i);
        __m512 vr = _mm512_div_ps(va, vb);
        _mm512_store_ps(out + i, vr);
    }
    for (size_t i = simd_size; i < size; i++) {
        if (fabsf(in_b[i]) < 1e-10f) {
            out[i] = (in_a[i] >= 0) ? INFINITY : -INFINITY;
        } else {
            out[i] = in_a[i] / in_b[i];
        }
    }
    #else
    #pragma omp parallel for if(size > 1000)
    for (size_t i = 0; i < size; i++) {
        if (fabsf(in_b[i]) < 1e-10f) {
            out[i] = (in_a[i] >= 0) ? INFINITY : -INFINITY;
        } else {
            out[i] = in_a[i] / in_b[i];
        }
    }
    #endif

    if (profiling_enabled) {
        operation_count++;
        total_flops += size;
    }

    return QGT_SUCCESS;
}

// ============================================================================
// Core Tensor Operations
// ============================================================================

qgt_error_t geometric_core_tensor_contract(void* result,
                                         const void* a,
                                         const void* b,
                                         const size_t* dims_a,
                                         const size_t* dims_b,
                                         size_t rank_a,
                                         size_t rank_b,
                                         const size_t* contract_a,
                                         const size_t* contract_b,
                                         size_t num_contractions) {
    QGT_CHECK_NULL(result);
    QGT_CHECK_NULL(a);
    QGT_CHECK_NULL(b);
    QGT_CHECK_NULL(dims_a);
    QGT_CHECK_NULL(dims_b);
    QGT_CHECK_ARGUMENT(rank_a > 0);
    QGT_CHECK_ARGUMENT(rank_b > 0);
    QGT_CHECK_STATE(is_initialized);

    const float* A = (const float*)a;
    const float* B = (const float*)b;
    float* C = (float*)result;

    // Calculate total sizes
    size_t size_a = 1, size_b = 1;
    for (size_t i = 0; i < rank_a; i++) size_a *= dims_a[i];
    for (size_t i = 0; i < rank_b; i++) size_b *= dims_b[i];

    // Calculate contraction dimension size
    size_t contract_size = 1;
    for (size_t i = 0; i < num_contractions; i++) {
        contract_size *= dims_a[contract_a[i]];
    }

    // Calculate output dimensions
    // Result rank = input ranks - 2 * contracted dimensions (each contraction removes one dim from each tensor)
    size_t result_rank = rank_a + rank_b - 2 * num_contractions;
    size_t result_size = 1;

    // Validate result rank - check for underflow due to too many contractions
    // A scalar result (rank 0) is valid when all dimensions are contracted
    if (result_rank > rank_a + rank_b) {  // Underflow detection (size_t wraparound)
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // Compute strides for tensor A
    size_t* strides_a = (size_t*)geometric_allocate(rank_a * sizeof(size_t));
    if (!strides_a) return QGT_ERROR_MEMORY_ALLOCATION;

    strides_a[rank_a - 1] = 1;
    for (size_t i = rank_a - 1; i > 0; i--) {
        strides_a[i - 1] = strides_a[i] * dims_a[i];
    }

    // Compute strides for tensor B
    size_t* strides_b = (size_t*)geometric_allocate(rank_b * sizeof(size_t));
    if (!strides_b) {
        geometric_free(strides_a);
        return QGT_ERROR_MEMORY_ALLOCATION;
    }

    strides_b[rank_b - 1] = 1;
    for (size_t i = rank_b - 1; i > 0; i--) {
        strides_b[i - 1] = strides_b[i] * dims_b[i];
    }

    // Calculate result size by excluding contracted dimensions
    for (size_t i = 0; i < rank_a; i++) {
        bool is_contracted = false;
        for (size_t j = 0; j < num_contractions; j++) {
            if (contract_a[j] == i) {
                is_contracted = true;
                break;
            }
        }
        if (!is_contracted) {
            result_size *= dims_a[i];
        }
    }
    for (size_t i = 0; i < rank_b; i++) {
        bool is_contracted = false;
        for (size_t j = 0; j < num_contractions; j++) {
            if (contract_b[j] == i) {
                is_contracted = true;
                break;
            }
        }
        if (!is_contracted) {
            result_size *= dims_b[i];
        }
    }

    // Initialize result to zero
    memset(C, 0, result_size * sizeof(float));

    // Perform contraction (general case)
    // This is a simplified implementation - production code would use optimal loop ordering
    size_t outer_a = size_a / contract_size;
    size_t outer_b = size_b / contract_size;

    #pragma omp parallel for collapse(2) if(outer_a * outer_b > 1000)
    for (size_t i = 0; i < outer_a; i++) {
        for (size_t j = 0; j < outer_b; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < contract_size; k++) {
                sum += A[i * contract_size + k] * B[k * outer_b + j];
            }
            C[i * outer_b + j] = sum;
        }
    }

    geometric_free(strides_a);
    geometric_free(strides_b);

    if (profiling_enabled) {
        operation_count++;
        total_flops += 2 * result_size * contract_size;
    }

    return QGT_SUCCESS;
}

qgt_error_t geometric_core_tensor_transpose(void* result,
                                          const void* tensor,
                                          const size_t* dims,
                                          size_t rank,
                                          const size_t* perm) {
    QGT_CHECK_NULL(result);
    QGT_CHECK_NULL(tensor);
    QGT_CHECK_NULL(dims);
    QGT_CHECK_NULL(perm);
    QGT_CHECK_ARGUMENT(rank > 0);
    QGT_CHECK_STATE(is_initialized);

    const float* in = (const float*)tensor;
    float* out = (float*)result;

    // Calculate total size and strides
    size_t total_size = 1;
    for (size_t i = 0; i < rank; i++) {
        total_size *= dims[i];
    }

    // Compute input strides
    size_t* in_strides = (size_t*)geometric_allocate(rank * sizeof(size_t));
    if (!in_strides) return QGT_ERROR_MEMORY_ALLOCATION;

    in_strides[rank - 1] = 1;
    for (size_t i = rank - 1; i > 0; i--) {
        in_strides[i - 1] = in_strides[i] * dims[i];
    }

    // Compute output strides based on permutation
    size_t* out_strides = (size_t*)geometric_allocate(rank * sizeof(size_t));
    if (!out_strides) {
        geometric_free(in_strides);
        return QGT_ERROR_MEMORY_ALLOCATION;
    }

    size_t* out_dims = (size_t*)geometric_allocate(rank * sizeof(size_t));
    if (!out_dims) {
        geometric_free(in_strides);
        geometric_free(out_strides);
        return QGT_ERROR_MEMORY_ALLOCATION;
    }

    for (size_t i = 0; i < rank; i++) {
        out_dims[i] = dims[perm[i]];
    }

    out_strides[rank - 1] = 1;
    for (size_t i = rank - 1; i > 0; i--) {
        out_strides[i - 1] = out_strides[i] * out_dims[i];
    }

    // Perform transpose
    #pragma omp parallel for if(total_size > 10000)
    for (size_t i = 0; i < total_size; i++) {
        // Convert linear index to multi-index
        size_t idx = i;
        size_t out_linear = 0;

        for (size_t d = 0; d < rank; d++) {
            size_t coord = idx / in_strides[d];
            idx %= in_strides[d];

            // Find position in output
            for (size_t p = 0; p < rank; p++) {
                if (perm[p] == d) {
                    out_linear += coord * out_strides[p];
                    break;
                }
            }
        }

        out[out_linear] = in[i];
    }

    geometric_free(in_strides);
    geometric_free(out_strides);
    geometric_free(out_dims);

    if (profiling_enabled) {
        operation_count++;
        total_flops += total_size;
    }

    return QGT_SUCCESS;
}

qgt_error_t geometric_core_tensor_decompose(void* u, void* s, void* v,
                                          const void* tensor,
                                          const size_t* dims,
                                          size_t rank) {
    QGT_CHECK_NULL(u);
    QGT_CHECK_NULL(s);
    QGT_CHECK_NULL(v);
    QGT_CHECK_NULL(tensor);
    QGT_CHECK_NULL(dims);
    QGT_CHECK_ARGUMENT(rank >= 2);
    QGT_CHECK_STATE(is_initialized);

    // Reshape tensor to matrix for SVD
    size_t left_size = 1, right_size = 1;
    size_t mid = rank / 2;

    for (size_t i = 0; i < mid; i++) {
        left_size *= dims[i];
    }
    for (size_t i = mid; i < rank; i++) {
        right_size *= dims[i];
    }

    size_t min_dim = (left_size < right_size) ? left_size : right_size;

    const float* A = (const float*)tensor;
    float* U = (float*)u;
    float* S = (float*)s;
    float* V = (float*)v;

    // Initialize U to identity-like matrix
    memset(U, 0, left_size * min_dim * sizeof(float));
    for (size_t i = 0; i < min_dim; i++) {
        U[i * min_dim + i] = 1.0f;
    }

    // Initialize V to identity-like matrix
    memset(V, 0, min_dim * right_size * sizeof(float));
    for (size_t i = 0; i < min_dim; i++) {
        V[i * right_size + i] = 1.0f;
    }

    // Copy input to working matrix
    float* work = (float*)geometric_allocate(left_size * right_size * sizeof(float));
    if (!work) return QGT_ERROR_MEMORY_ALLOCATION;

    memcpy(work, A, left_size * right_size * sizeof(float));

    // Perform Jacobi SVD iteration
    const size_t max_iters = 100;
    const float tolerance = 1e-6f;

    for (size_t iter = 0; iter < max_iters; iter++) {
        float max_off_diag = 0.0f;

        // Sweep through all pairs
        for (size_t p = 0; p < min_dim; p++) {
            for (size_t q = p + 1; q < min_dim; q++) {
                // Compute 2x2 submatrix elements
                float app = 0.0f, aqq = 0.0f, apq = 0.0f;

                for (size_t i = 0; i < left_size; i++) {
                    app += work[i * right_size + p] * work[i * right_size + p];
                    aqq += work[i * right_size + q] * work[i * right_size + q];
                    apq += work[i * right_size + p] * work[i * right_size + q];
                }

                if (fabsf(apq) > max_off_diag) {
                    max_off_diag = fabsf(apq);
                }

                if (fabsf(apq) > tolerance) {
                    // Compute rotation angle
                    float tau = (aqq - app) / (2.0f * apq);
                    float t = (tau >= 0) ?
                        1.0f / (tau + sqrtf(1.0f + tau * tau)) :
                        -1.0f / (-tau + sqrtf(1.0f + tau * tau));

                    float c = 1.0f / sqrtf(1.0f + t * t);
                    float sn = t * c;

                    // Apply rotation to columns of work matrix
                    for (size_t i = 0; i < left_size; i++) {
                        float wp = work[i * right_size + p];
                        float wq = work[i * right_size + q];
                        work[i * right_size + p] = c * wp - sn * wq;
                        work[i * right_size + q] = sn * wp + c * wq;
                    }

                    // Accumulate right rotation in V
                    for (size_t i = 0; i < min_dim; i++) {
                        float vp = V[i * right_size + p];
                        float vq = V[i * right_size + q];
                        V[i * right_size + p] = c * vp - sn * vq;
                        V[i * right_size + q] = sn * vp + c * vq;
                    }
                }
            }
        }

        // Check convergence
        if (max_off_diag < tolerance) {
            break;
        }
    }

    // Extract singular values and U
    for (size_t i = 0; i < min_dim; i++) {
        float sigma = 0.0f;
        for (size_t j = 0; j < left_size; j++) {
            sigma += work[j * right_size + i] * work[j * right_size + i];
        }
        S[i] = sqrtf(sigma);

        // Normalize column to get U
        if (S[i] > tolerance) {
            for (size_t j = 0; j < left_size; j++) {
                U[j * min_dim + i] = work[j * right_size + i] / S[i];
            }
        }
    }

    geometric_free(work);

    if (profiling_enabled) {
        operation_count++;
        total_flops += left_size * right_size * min_dim;
    }

    return QGT_SUCCESS;
}

// ============================================================================
// Core Error Handling and Utility Functions
// ============================================================================

const char* geometric_core_get_error_string(qgt_error_t error) {
    switch (error) {
        case QGT_SUCCESS:
            return "Success";
        case QGT_ERROR_MEMORY_ALLOCATION:
            return "Memory allocation failed";
        case QGT_ERROR_INVALID_PARAMETER:
            return "Invalid parameter";
        case QGT_ERROR_NOT_INITIALIZED:
            return "Not initialized";
        case QGT_ERROR_INVALID_STATE:
            return "Invalid state";
        case QGT_ERROR_RESOURCE_EXHAUSTED:
            return "Resource exhausted";
        case QGT_ERROR_TIMEOUT:
            return "Operation timed out";
        case QGT_ERROR_DEVICE:
            return "Device error";
        default:
            return "Unknown error";
    }
}

qgt_error_t geometric_core_get_last_error(void) {
    return last_error;
}

void geometric_core_clear_error(void) {
    last_error = QGT_SUCCESS;
}

qgt_error_t geometric_core_get_version(size_t* major, size_t* minor, size_t* patch) {
    QGT_CHECK_NULL(major);
    QGT_CHECK_NULL(minor);
    QGT_CHECK_NULL(patch);

    *major = 1;
    *minor = 0;
    *patch = 0;

    return QGT_SUCCESS;
}

qgt_error_t geometric_core_get_capabilities(void* capabilities) {
    QGT_CHECK_NULL(capabilities);
    QGT_CHECK_STATE(is_initialized);

    typedef struct {
        bool has_openmp;
        bool has_cuda;
        bool has_metal;
        bool has_neon;
        bool has_avx;
        bool has_avx512;
        size_t num_threads;
        size_t num_devices;
    } Capabilities;

    Capabilities* caps = (Capabilities*)capabilities;

    #ifdef _OPENMP
    caps->has_openmp = true;
    caps->num_threads = omp_get_max_threads();
    #else
    caps->has_openmp = false;
    caps->num_threads = 1;
    #endif

    #ifdef CUDA_AVAILABLE
    caps->has_cuda = true;
    #else
    caps->has_cuda = false;
    #endif

    #ifdef __APPLE__
    caps->has_metal = true;
    #else
    caps->has_metal = false;
    #endif

    #ifdef __ARM_NEON
    caps->has_neon = true;
    #else
    caps->has_neon = false;
    #endif

    #if defined(__AVX__)
    caps->has_avx = true;
    #else
    caps->has_avx = false;
    #endif

    #if defined(__AVX512F__)
    caps->has_avx512 = true;
    #else
    caps->has_avx512 = false;
    #endif

    geometric_core_get_device_count(&caps->num_devices);

    return QGT_SUCCESS;
}

qgt_error_t geometric_core_set_debug_mode(bool enable) {
    debug_mode = enable;
    return QGT_SUCCESS;
}

qgt_error_t geometric_core_set_log_level(size_t level) {
    log_level = level;
    return QGT_SUCCESS;
}
