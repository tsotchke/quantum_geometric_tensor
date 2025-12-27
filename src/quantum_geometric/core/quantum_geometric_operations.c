#include "quantum_geometric/core/quantum_geometric_operations.h"
#include "quantum_geometric/core/quantum_geometric_hardware.h"
#include "quantum_geometric/hardware/quantum_geometric_gpu.h"
#include "quantum_geometric/hardware/quantum_hardware_types.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_geometric_tensor.h"
#include "quantum_geometric/core/quantum_geometric_constants.h"
#include "quantum_geometric/core/quantum_geometric_gpu.h"
#include "quantum_geometric/core/quantum_state.h"
#include "quantum_geometric/core/quantum_types.h"
#include "quantum_geometric/hardware/quantum_hardware_constants.h"
#include "quantum_geometric/core/memory_pool.h"
#include "quantum_geometric/core/simd_operations.h"
#include "quantum_geometric/core/quantum_geometric_config.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/core/quantum_geometric_validation.h"
#include "quantum_geometric/core/quantum_geometric_error.h"
#include "quantum_geometric/core/quantum_geometric_memory.h"
#include "quantum_geometric/core/quantum_geometric_profiling.h"
#include "quantum_geometric/core/quantum_geometric_logging.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <pthread.h>
#include <stdatomic.h>

// Thread safety
static pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;
static atomic_flag g_init_lock = ATOMIC_FLAG_INIT;

// OpenMP support with proper thread safety
#if defined(_OPENMP)
#include <omp.h>
#define OMP_PARALLEL _Pragma("omp parallel")
#define OMP_FOR _Pragma("omp for")
#define OMP_CRITICAL _Pragma("omp critical")
#define OMP_FOR_COLLAPSE2 _Pragma("omp for collapse(2)")
#define OMP_FOR_COLLAPSE3 _Pragma("omp for collapse(3)")
#define OMP_FOR_COLLAPSE4 _Pragma("omp for collapse(4)")
#define OMP_FOR_GUIDED _Pragma("omp for schedule(guided)")
#define OMP_FOR_GUIDED_NOWAIT _Pragma("omp for schedule(guided) nowait")
#define OMP_PARALLEL_FOR _Pragma("omp parallel for")
#define OMP_PARALLEL_FOR_COLLAPSE2 _Pragma("omp parallel for collapse(2)")
#define OMP_PARALLEL_FOR_COLLAPSE3 _Pragma("omp parallel for collapse(3)")
#define OMP_PARALLEL_FOR_COLLAPSE4 _Pragma("omp parallel for collapse(4)")
#define OMP_PARALLEL_FOR_IF(cond) _Pragma("omp parallel for if(" #cond ")")
#define OMP_BARRIER _Pragma("omp barrier")
#define OMP_FLUSH(x) _Pragma("omp flush(" #x ")")
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#define omp_in_parallel() 0
#define OMP_PARALLEL
#define OMP_FOR
#define OMP_CRITICAL
#define OMP_FOR_COLLAPSE2
#define OMP_FOR_COLLAPSE3
#define OMP_FOR_COLLAPSE4
#define OMP_FOR_GUIDED
#define OMP_FOR_GUIDED_NOWAIT
#define OMP_PARALLEL_FOR
#define OMP_PARALLEL_FOR_COLLAPSE2
#define OMP_PARALLEL_FOR_COLLAPSE3
#define OMP_PARALLEL_FOR_COLLAPSE4
#define OMP_PARALLEL_FOR_IF(cond)
#define OMP_BARRIER
#define OMP_FLUSH(x)
#endif

// Improved hardware validation with error codes
static inline qgt_error_t validate_hardware_type(HardwareType type) {
    if (type != HARDWARE_TYPE_CPU &&
        type != HARDWARE_TYPE_GPU &&
        type != HARDWARE_TYPE_QPU &&
        type != HARDWARE_TYPE_SIMULATOR &&
        type != HARDWARE_TYPE_METAL) {
        return QGT_ERROR_INVALID_HARDWARE;
    }
    return QGT_SUCCESS;
}

// Thread-safe hardware cleanup with error handling
static qgt_error_t cleanup_hardware_resources(quantum_geometric_hardware_t* hardware) {
    if (!hardware) return QGT_SUCCESS;
    
    pthread_mutex_lock(&g_mutex);
    
    qgt_error_t err = QGT_SUCCESS;
    
    if (hardware->context) {
        if (hardware->input_buffer) {
            gpu_free(hardware->context, hardware->input_buffer);
            hardware->input_buffer = NULL;
        }
        if (hardware->output_buffer) {
            gpu_free(hardware->context, hardware->output_buffer);
            hardware->output_buffer = NULL;
        }
        if (hardware->device_handle) {
            hardware->device_handle = NULL;
        }
        hardware->context = NULL;
    }
    
    pthread_mutex_unlock(&g_mutex);
    return err;
}

// Optional MPI support
#if defined(HAVE_MPI) && !defined(NO_MPI)
#include <mpi.h>
#endif

// Optional HWLOC support
#if defined(HAVE_HWLOC) && !defined(NO_HWLOC)
#include <hwloc.h>
#endif

#define QGT_POOL_INITIAL_SIZE (1024 * 1024)

// GlobalState is defined in quantum_geometric_operations.h
// g_state is cache-aligned for optimal performance
GlobalState g_state __attribute__((aligned(64))) = {
    .initialized = ATOMIC_VAR_INIT(false),
    .config = {0},
    .pool = NULL,
    .hardware = NULL
};

// Helper functions
static inline qgt_error_t validate_dimensions(size_t dim1, size_t dim2) {
    if (dim1 == 0 || dim2 == 0 || dim1 > QGT_MAX_DIMENSIONS || dim2 > QGT_MAX_DIMENSIONS) {
        return QGT_ERROR_INVALID_DIMENSION;
    }
    return (dim1 == dim2) ? QGT_SUCCESS : QGT_ERROR_DIMENSION_MISMATCH;
}

// validate_state is now defined in quantum_geometric_operations.h

static inline void init_default_config(quantum_geometric_config_t* config) {
    *config = (quantum_geometric_config_t) {
        .num_threads = 1,
        .batch_size = QGT_MAX_BATCH_SIZE,
        .max_iterations = QGT_MAX_ITERATIONS,
        .learning_rate = QGT_LEARNING_RATE,
        .convergence_threshold = QGT_CONVERGENCE_TOL,
        .use_gpu = false,
        .distributed = false
    };
}

// Forward declarations
void cleanup_memory_pool(MemoryPool* pool);

// Memory management
qgt_error_t geometric_initialize(void) {
    // Use atomic flag for lightweight initialization check
    if (atomic_flag_test_and_set(&g_init_lock)) {
        return QGT_ERROR_ALREADY_INITIALIZED;
    }
    
    // Double-checked locking pattern
    if (atomic_load(&g_state.initialized)) {
        atomic_flag_clear(&g_init_lock);
        return QGT_ERROR_ALREADY_INITIALIZED;
    }
    
    pthread_mutex_lock(&g_mutex);
    
    qgt_error_t err = QGT_SUCCESS;
    
    // Initialize memory pool with optimized configuration
    struct PoolConfig pool_config = {
        .min_block_size = QGT_MIN_POOL_BLOCK,
        .alignment = QGT_CACHE_LINE_SIZE,  // Align to cache line
        .num_size_classes = QGT_NUM_SIZE_CLASSES,
        .growth_factor = 1.5f,
        .prefetch_distance = QGT_POOL_PREFETCH,
        .use_huge_pages = true,  // Enable huge pages for better performance
        .cache_local_free_lists = true,
        .max_blocks_per_class = QGT_MAX_POOL_BLOCK,
        .thread_cache_size = QGT_MAX_POOL_THREADS,
        .enable_stats = true
    };
    
    MemoryPool* temp_pool = create_memory_pool(&pool_config);
    if (!temp_pool) {
        err = QGT_ERROR_MEMORY_ALLOCATION;
        geometric_set_error(err, __FILE__, __LINE__, __func__,
                          "Failed to initialize memory pool");
        goto cleanup;
    }
    g_state.pool = temp_pool;

    // Initialize hardware with cache line alignment
    g_state.hardware = pool_allocate(g_state.pool,
                                 sizeof(quantum_geometric_hardware_t));
    if (!g_state.hardware) {
        err = QGT_ERROR_MEMORY_ALLOCATION;
        geometric_set_error(err, __FILE__, __LINE__, __func__,
                          "Failed to allocate hardware structure");
        goto cleanup;
    }
    
    // Initialize hardware with proper error checking
    memset(g_state.hardware, 0, sizeof(quantum_geometric_hardware_t));
    g_state.hardware->type = HARDWARE_TYPE_CPU;
    
    // Initialize hardware capabilities
    hardware_config_t hw_config = {
        .gpu_device_id = 0,
        .qpu_connection_string = NULL
    };
    err = geometric_initialize_hardware(g_state.hardware, &hw_config);
    if (err != QGT_SUCCESS) {
        geometric_set_error(err, __FILE__, __LINE__, __func__,
                          "Failed to initialize hardware capabilities");
        goto cleanup;
    }
    
    // Initialize configuration
    init_default_config(&g_state.config);
    
    // Set initialization flag last
    atomic_store(&g_state.initialized, true);
    
cleanup:
    if (err != QGT_SUCCESS) {
        if (g_state.hardware) {
            pool_free(g_state.pool, g_state.hardware);
            g_state.hardware = NULL;
        }
    }
    
    // Only cleanup memory pool if initialization failed
    if (err != QGT_SUCCESS && g_state.pool) {
        cleanup_memory_pool(g_state.pool);
        g_state.pool = NULL;
    }
    
    pthread_mutex_unlock(&g_mutex);
    atomic_flag_clear(&g_init_lock);
    
    return err;
}

void geometric_shutdown(void) {
    if (!atomic_load(&g_state.initialized)) {
        return;
    }
    
    pthread_mutex_lock(&g_mutex);
    
    // Prevent new operations during shutdown
    atomic_store(&g_state.initialized, false);
    
    qgt_error_t err = QGT_SUCCESS;
    
    // First cleanup hardware resources if they exist
    if (g_state.hardware) {
        qgt_error_t hw_err = cleanup_hardware_resources(g_state.hardware);
        if (hw_err != QGT_SUCCESS) {
            geometric_set_error(hw_err, __FILE__, __LINE__, __func__,
                              "Failed to cleanup hardware resources");
        }

        // Now safe to free hardware struct
        if (g_state.pool) {
            pool_free(g_state.pool, g_state.hardware);
            g_state.hardware = NULL;
        }
    }

    // Finally cleanup memory pool (void return type)
    if (g_state.pool) {
        cleanup_memory_pool(g_state.pool);
        g_state.pool = NULL;
    }

    // Zero out configuration
    memset(&g_state.config, 0, sizeof(quantum_geometric_config_t));
    g_state.pool = NULL;
    g_state.hardware = NULL;

    pthread_mutex_unlock(&g_mutex);
}

qgt_error_t geometric_reset(void) {
    static pthread_mutex_t reset_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_lock(&reset_mutex);
    
    qgt_error_t err = validate_state();
    if (err != QGT_SUCCESS) {
        pthread_mutex_unlock(&reset_mutex);
        return err;
    }
    
    // First cleanup existing resources
    if (g_state.hardware) {
        err = cleanup_hardware_resources(g_state.hardware);
        if (err != QGT_SUCCESS) {
            geometric_set_error(err, __FILE__, __LINE__, __func__,
                              "Failed to cleanup hardware resources");
            pthread_mutex_unlock(&reset_mutex);
            return err;
        }
        pool_free(g_state.pool, g_state.hardware);
        g_state.hardware = NULL;
    }
    
    // Cleanup memory pool (void return type)
    cleanup_memory_pool(g_state.pool);
    g_state.pool = NULL;
    
    // Reinitialize with optimized configuration
    struct PoolConfig config = {
        .min_block_size = QGT_MIN_POOL_BLOCK,
        .alignment = QGT_POOL_ALIGNMENT,
        .num_size_classes = QGT_NUM_SIZE_CLASSES,
        .growth_factor = 1.5f,
        .prefetch_distance = QGT_POOL_PREFETCH,
        .use_huge_pages = false,
        .cache_local_free_lists = true,
        .max_blocks_per_class = QGT_MAX_POOL_BLOCK,
        .thread_cache_size = QGT_MAX_POOL_THREADS,
        .enable_stats = true
    };
    
    g_state.pool = create_memory_pool(&config);
    if (!g_state.pool) {
        pthread_mutex_unlock(&reset_mutex);
        return QGT_ERROR_MEMORY_ALLOCATION;
    }
    
    // Reinitialize hardware
    g_state.hardware = pool_allocate(g_state.pool, sizeof(quantum_geometric_hardware_t));
    if (!g_state.hardware) {
        cleanup_memory_pool(g_state.pool);
        g_state.pool = NULL;
        pthread_mutex_unlock(&reset_mutex);
        return QGT_ERROR_MEMORY_ALLOCATION;
    }
    
    memset(g_state.hardware, 0, sizeof(quantum_geometric_hardware_t));
    g_state.hardware->type = HARDWARE_TYPE_CPU;
    g_state.hardware->is_initialized = true;
    
    init_default_config(&g_state.config);
    
    pthread_mutex_unlock(&reset_mutex);
    return QGT_SUCCESS;
}

// Note: geometric_create_optimization, geometric_destroy_optimization, geometric_optimize_parameters,
// and geometric_check_convergence are defined in quantum_geometric_optimization.c

qgt_error_t geometric_optimize_step(quantum_geometric_optimization_t* optimization,
                                  const quantum_geometric_metric_t* metric,
                                  const quantum_geometric_connection_t* connection,
                                  const quantum_geometric_curvature_t* curvature) {
    QGT_CHECK_NULL(optimization);
    QGT_CHECK_NULL(metric);
    QGT_CHECK_NULL(connection);
    QGT_CHECK_NULL(curvature);
    QGT_CHECK_STATE(validate_state());
    
    size_t dim = optimization->dimension;
    size_t block_size = QGT_BLOCK_SIZE / sizeof(ComplexFloat);
    
    // Allocate temporary gradient buffer
    ComplexFloat* gradient = pool_allocate(g_state.pool, dim * sizeof(ComplexFloat));
    if (!gradient) return QGT_ERROR_MEMORY_ALLOCATION;
    
    // Zero initialize gradient
    memset(gradient, 0, dim * sizeof(ComplexFloat));
    
    // Block computation for better cache utilization
    #pragma omp parallel if(dim > QGT_PARALLEL_THRESHOLD)
    {
        // Thread-local buffer to avoid false sharing
        ComplexFloat* local_grad = pool_allocate(g_state.pool, dim * sizeof(ComplexFloat));
        if (local_grad) {
            memset(local_grad, 0, dim * sizeof(ComplexFloat));
            
            // Process blocks
            #pragma omp for
            for (size_t i_block = 0; i_block < dim; i_block += block_size) {
                size_t i_end = (i_block + block_size < dim) ? i_block + block_size : dim;
                
                for (size_t i = i_block; i < i_end; i++) {
                    // Metric contribution with tiling
                    for (size_t j = 0; j < dim; j += QGT_TILE_SIZE) {
                        size_t j_end = (j + QGT_TILE_SIZE < dim) ? j + QGT_TILE_SIZE : dim;
                        for (size_t jj = j; jj < j_end; jj++) {
                            local_grad[i] = complex_float_add(
                                local_grad[i],
                                complex_float_multiply(
                                    metric->components[i * dim + jj],
                                    connection->coefficients[i * dim * dim + jj * dim]
                                )
                            );
                        }
                    }
                    
                    // Curvature contribution with tiling
                    for (size_t j = 0; j < dim; j += QGT_TILE_SIZE) {
                        size_t j_end = (j + QGT_TILE_SIZE < dim) ? j + QGT_TILE_SIZE : dim;
                        for (size_t jj = j; jj < j_end; jj++) {
                            for (size_t k = 0; k < dim; k += QGT_TILE_SIZE) {
                                size_t k_end = (k + QGT_TILE_SIZE < dim) ? k + QGT_TILE_SIZE : dim;
                                for (size_t kk = k; kk < k_end; kk++) {
                                    for (size_t l = 0; l < dim; l += QGT_TILE_SIZE) {
                                        size_t l_end = (l + QGT_TILE_SIZE < dim) ? l + QGT_TILE_SIZE : dim;
                                        for (size_t ll = l; ll < l_end; ll++) {
                                            local_grad[i] = complex_float_add(
                                                local_grad[i],
                                                curvature->components[((i * dim + jj) * dim + kk) * dim + ll]
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            // Reduce thread-local results
            #pragma omp critical
            {
                for (size_t i = 0; i < dim; i++) {
                    gradient[i] = complex_float_add(gradient[i], local_grad[i]);
                }
            }
            
            pool_free(g_state.pool, local_grad);
        }
    }
    
    // Update parameters using gradient descent with SIMD
    #pragma omp parallel for if(dim > QGT_PARALLEL_THRESHOLD)
    for (size_t i = 0; i < dim; i++) {
        ComplexFloat scaled_grad = complex_float_multiply_real(gradient[i], optimization->learning_rate);
        optimization->parameters[i] = complex_float_subtract(optimization->parameters[i], scaled_grad);
    }
    
    pool_free(g_state.pool, gradient);
    optimization->iterations++;
    return QGT_SUCCESS;
}

// Note: geometric_compute_metric is defined in quantum_geometric_metric.c
// Note: geometric_compute_connection is defined in quantum_geometric_connection.c
// Note: geometric_compute_curvature is defined in quantum_geometric_curvature.c

// State management
qgt_error_t geometric_create_state(quantum_geometric_state_t** state,
                                 geometric_state_type_t type,
                                 size_t dimension,
                                 HardwareType hardware_type) {
    QGT_CHECK_NULL(state);
    QGT_CHECK_ARGUMENT(dimension > 0 && dimension <= QGT_MAX_DIMENSIONS);
    QGT_CHECK_STATE(validate_state());
    
    *state = pool_allocate(g_state.pool, sizeof(quantum_geometric_state_t));
    if (!*state) return QGT_ERROR_MEMORY_ALLOCATION;
    
    // Initialize basic fields
    **state = (quantum_geometric_state_t) {
        .type = type,
        .dimension = dimension,
        .hardware = hardware_type,
        .coordinates = NULL,
        .metric = NULL,
        .connection = NULL,
        .auxiliary_data = NULL,
        .is_normalized = false,
        .manifold_dim = dimension
    };
    
    // Allocate coordinates array
    (*state)->coordinates = pool_allocate(g_state.pool, dimension * sizeof(ComplexFloat));
    if (!(*state)->coordinates) {
        pool_free(g_state.pool, *state);
        return QGT_ERROR_MEMORY_ALLOCATION;
    }
    memset((*state)->coordinates, 0, dimension * sizeof(ComplexFloat));
    
    // Allocate metric array
    (*state)->metric = pool_allocate(g_state.pool, dimension * dimension * sizeof(ComplexFloat));
    if (!(*state)->metric) {
        pool_free(g_state.pool, (*state)->coordinates);
        pool_free(g_state.pool, *state);
        return QGT_ERROR_MEMORY_ALLOCATION;
    }
    memset((*state)->metric, 0, dimension * dimension * sizeof(ComplexFloat));
    
    // Allocate connection array
    (*state)->connection = pool_allocate(g_state.pool, dimension * dimension * dimension * sizeof(ComplexFloat));
    if (!(*state)->connection) {
        pool_free(g_state.pool, (*state)->metric);
        pool_free(g_state.pool, (*state)->coordinates);
        pool_free(g_state.pool, *state);
        return QGT_ERROR_MEMORY_ALLOCATION;
    }
    memset((*state)->connection, 0, dimension * dimension * dimension * sizeof(ComplexFloat));
    
    return QGT_SUCCESS;
}

void geometric_destroy_state(quantum_geometric_state_t* state) {
    if (!state || !g_state.initialized) return;
    
    quantum_geometric_hardware_t* hw = NULL;
    if (state->hardware == HARDWARE_TYPE_GPU) {
        hw = g_state.hardware;
        if (!hw || !hw->context) {
            // Fallback to pool free if no valid GPU context
            if (state->coordinates) pool_free(g_state.pool, state->coordinates);
            if (state->metric) pool_free(g_state.pool, state->metric);
            if (state->connection) pool_free(g_state.pool, state->connection);
            pool_free(g_state.pool, state);
            return;
        }
    }

    if (state->coordinates) {
        if (state->hardware == HARDWARE_TYPE_GPU) {
            gpu_free(hw->context, state->coordinates);
        } else {
            pool_free(g_state.pool, state->coordinates);
        }
    }
    if (state->metric) {
        if (state->hardware == HARDWARE_TYPE_GPU) {
            gpu_free(hw->context, state->metric);
        } else {
            pool_free(g_state.pool, state->metric);
        }
    }
    if (state->connection) {
        if (state->hardware == HARDWARE_TYPE_GPU) {
            gpu_free(hw->context, state->connection);
        } else {
            pool_free(g_state.pool, state->connection);
        }
    }
    
    pool_free(g_state.pool, state);
}

// Resource management
qgt_error_t geometric_estimate_resources(const quantum_geometric_state_t* state,
                                       size_t* memory,
                                       size_t* operations) {
    QGT_CHECK_NULL(state);
    QGT_CHECK_NULL(memory);
    QGT_CHECK_NULL(operations);
    
    size_t dim = state->dimension;
    *memory = sizeof(quantum_geometric_state_t) +
              dim * sizeof(ComplexFloat) +
              (state->metric ? dim * dim * sizeof(ComplexFloat) : 0) +
              (state->connection ? dim * dim * dim * sizeof(ComplexFloat) : 0);
              
    *operations = dim * dim; // Basic matrix operations
    return QGT_SUCCESS;
}

// Error correction
qgt_error_t geometric_error_correct(quantum_geometric_state_t* state,
                                  const quantum_geometric_metric_t* metric) {
    QGT_CHECK_NULL(state);
    QGT_CHECK_NULL(metric);
    QGT_CHECK_STATE(validate_state());
    QGT_CHECK_STATE(validate_dimensions(state->dimension, metric->dimension));
    
    ComplexFloat* corrected = pool_allocate(g_state.pool, 
        state->dimension * sizeof(ComplexFloat));
    if (!corrected) return QGT_ERROR_MEMORY_ALLOCATION;
    
    #pragma omp parallel for if(state->dimension > QGT_PARALLEL_THRESHOLD)
    for (size_t i = 0; i < state->dimension; i++) {
        corrected[i] = COMPLEX_FLOAT_ZERO;
        simd_complex_multiply_accumulate(
            &corrected[i],
            &metric->components[i * state->dimension],
            state->coordinates,
            state->dimension
        );
    }
    
    double norm = simd_complex_norm(corrected, state->dimension);
    if (norm > QGT_EPSILON) {
        simd_complex_scale(state->coordinates, corrected,
                         complex_float_create(1.0 / sqrt(norm), 0.0f),
                         state->dimension);
        state->is_normalized = true;
    }
    
    pool_free(g_state.pool, corrected);
    return QGT_SUCCESS;
}

// Utility functions
qgt_error_t geometric_print_state(const quantum_geometric_state_t* state) {
    QGT_CHECK_NULL(state);
    
    printf("Geometric State:\n"
           "  Type: %d\n"
           "  Dimension: %zu\n"
           "  Coordinates:\n", state->type, state->dimension);
           
    for (size_t i = 0; i < state->dimension; i++) {
        printf("    [%zu] = %.6f + %.6fi\n", i,
               state->coordinates[i].real,
               state->coordinates[i].imag);
    }
    
    return QGT_SUCCESS;
}

// Additional geometric operations
qgt_error_t geometric_transform(quantum_geometric_state_t* result,
                              const quantum_geometric_state_t* state,
                              const quantum_geometric_tensor_t* transform) {
    QGT_CHECK_NULL(result);
    QGT_CHECK_NULL(state);
    QGT_CHECK_NULL(transform);
    QGT_CHECK_STATE(validate_state());
    QGT_CHECK_STATE(validate_dimensions(result->dimension, state->dimension));
    QGT_CHECK_STATE(validate_dimensions(transform->dimensions[0], state->dimension));
    
    #pragma omp parallel for if(state->dimension > QGT_PARALLEL_THRESHOLD)
    for (size_t i = 0; i < state->dimension; i++) {
        result->coordinates[i] = COMPLEX_FLOAT_ZERO;
        simd_complex_multiply_accumulate(
            &result->coordinates[i],
            &transform->components[i * state->dimension],
            state->coordinates,
            state->dimension
        );
    }
    
    return QGT_SUCCESS;
}

qgt_error_t geometric_parallel_transport(quantum_geometric_state_t* result,
                                       const quantum_geometric_state_t* state,
                                       const quantum_geometric_connection_t* connection) {
    QGT_CHECK_NULL(result);
    QGT_CHECK_NULL(state);
    QGT_CHECK_NULL(connection);
    QGT_CHECK_STATE(validate_state());
    QGT_CHECK_STATE(validate_dimensions(result->dimension, state->dimension));
    QGT_CHECK_STATE(validate_dimensions(connection->dimension, state->dimension));
    
    size_t dim = state->dimension;
    
    #pragma omp parallel for if(dim > QGT_PARALLEL_THRESHOLD)
    for (size_t i = 0; i < dim; i++) {
        result->coordinates[i] = state->coordinates[i];
        for (size_t j = 0; j < dim; j++) {
            ComplexFloat conn = connection->coefficients[i * dim * dim + j * dim];
            result->coordinates[i] = complex_float_add(
                result->coordinates[i],
                complex_float_multiply(conn, state->coordinates[j])
            );
        }
    }
    
    return QGT_SUCCESS;
}

qgt_error_t geometric_project(quantum_geometric_state_t* result,
                            const quantum_geometric_state_t* state,
                            const quantum_geometric_state_t* subspace) {
    QGT_CHECK_NULL(result);
    QGT_CHECK_NULL(state);
    QGT_CHECK_NULL(subspace);
    QGT_CHECK_STATE(validate_state());
    QGT_CHECK_STATE(validate_dimensions(result->dimension, state->dimension));
    QGT_CHECK_STATE(validate_dimensions(subspace->dimension, state->dimension));
    
    // Compute projection using inner product
    ComplexFloat inner_prod = COMPLEX_FLOAT_ZERO;
    ComplexFloat norm = COMPLEX_FLOAT_ZERO;
    
    // OpenMP reduction not supported for ComplexFloat, use thread-local storage
    #pragma omp parallel if(state->dimension > QGT_PARALLEL_THRESHOLD)
    {
        ComplexFloat local_inner = COMPLEX_FLOAT_ZERO;
        ComplexFloat local_norm = COMPLEX_FLOAT_ZERO;
        
        #pragma omp for nowait
        for (size_t i = 0; i < state->dimension; i++) {
            local_inner = complex_float_add(local_inner,
                complex_float_multiply(
                    complex_float_conjugate(subspace->coordinates[i]),
                    state->coordinates[i]
                )
            );
            local_norm = complex_float_add(local_norm,
                complex_float_multiply(
                    complex_float_conjugate(subspace->coordinates[i]),
                    subspace->coordinates[i]
                )
            );
        }
        
        #pragma omp critical
        {
            inner_prod = complex_float_add(inner_prod, local_inner);
            norm = complex_float_add(norm, local_norm);
        }
    }
    
    if (complex_float_abs(norm) < QGT_EPSILON) {
        return QGT_ERROR_INVALID_STATE;
    }
    
    ComplexFloat scale = complex_float_divide(inner_prod, norm);
    
    #pragma omp parallel for if(state->dimension > QGT_PARALLEL_THRESHOLD)
    for (size_t i = 0; i < state->dimension; i++) {
        result->coordinates[i] = complex_float_multiply(
            scale,
            subspace->coordinates[i]
        );
    }
    
    return QGT_SUCCESS;
}

// Note: geometric_create_metric and geometric_destroy_metric are defined in quantum_geometric_metric.c

qgt_error_t geometric_distance(double* distance,
                             const quantum_geometric_state_t* state1,
                             const quantum_geometric_state_t* state2,
                             const quantum_geometric_metric_t* metric) {
    QGT_CHECK_NULL(distance);
    QGT_CHECK_NULL(state1);
    QGT_CHECK_NULL(state2);
    QGT_CHECK_NULL(metric);
    QGT_CHECK_STATE(validate_state());
    QGT_CHECK_STATE(validate_dimensions(state1->dimension, state2->dimension));
    QGT_CHECK_STATE(validate_dimensions(metric->dimension, state1->dimension));
    
    size_t dim = state1->dimension;
    size_t block_size = QGT_BLOCK_SIZE / sizeof(ComplexFloat);
    ComplexFloat sum = COMPLEX_FLOAT_ZERO;
    
    // Pre-compute differences to avoid redundant calculations
    ComplexFloat* differences = pool_allocate(g_state.pool, dim * sizeof(ComplexFloat));
    if (!differences) return QGT_ERROR_MEMORY_ALLOCATION;
    
    #pragma omp parallel for if(dim > QGT_PARALLEL_THRESHOLD)
    for (size_t i = 0; i < dim; i++) {
        differences[i] = complex_float_subtract(
            state1->coordinates[i],
            state2->coordinates[i]
        );
    }
    
    // Block computation for better cache utilization
    #pragma omp parallel if(dim > QGT_PARALLEL_THRESHOLD)
    {
        ComplexFloat local_sum = COMPLEX_FLOAT_ZERO;
        
        #pragma omp for collapse(2) nowait
        for (size_t i_block = 0; i_block < dim; i_block += block_size) {
            for (size_t j_block = 0; j_block < dim; j_block += block_size) {
                size_t i_end = (i_block + block_size < dim) ? i_block + block_size : dim;
                size_t j_end = (j_block + block_size < dim) ? j_block + block_size : dim;
                
                // Process blocks with tiling
                for (size_t i = i_block; i < i_end; i++) {
                    ComplexFloat diff_i_conj = complex_float_conjugate(differences[i]);
                    
                    for (size_t j = j_block; j < j_end; j += QGT_TILE_SIZE) {
                        size_t j_tile_end = (j + QGT_TILE_SIZE < j_end) ? j + QGT_TILE_SIZE : j_end;
                        
                        for (size_t jj = j; jj < j_tile_end; jj++) {
                            ComplexFloat term = complex_float_multiply(
                                complex_float_multiply(
                                    diff_i_conj,
                                    metric->components[i * dim + jj]
                                ),
                                differences[jj]
                            );
                            local_sum = complex_float_add(local_sum, term);
                        }
                    }
                }
            }
        }
        
        #pragma omp critical
        {
            sum = complex_float_add(sum, local_sum);
        }
    }
    
    pool_free(g_state.pool, differences);
    
    *distance = sqrt(complex_float_abs(sum));
    return QGT_SUCCESS;
}

// Note: geometric_create_connection and geometric_destroy_connection are defined in quantum_geometric_connection.c

qgt_error_t geometric_transport(quantum_geometric_state_t* result,
                              const quantum_geometric_state_t* state,
                              const quantum_geometric_connection_t* connection,
                              const quantum_geometric_state_t* path) {
    QGT_CHECK_NULL(result);
    QGT_CHECK_NULL(state);
    QGT_CHECK_NULL(connection);
    QGT_CHECK_NULL(path);
    QGT_CHECK_STATE(validate_state());
    QGT_CHECK_STATE(validate_dimensions(result->dimension, state->dimension));
    QGT_CHECK_STATE(validate_dimensions(connection->dimension, state->dimension));
    QGT_CHECK_STATE(validate_dimensions(path->dimension, state->dimension));
    
    size_t dim = state->dimension;
    
    // First compute parallel transport
    qgt_error_t err = geometric_parallel_transport(result, state, connection);
    if (err != QGT_SUCCESS) return err;
    
    #pragma omp parallel for if(dim > QGT_PARALLEL_THRESHOLD)
    for (size_t i = 0; i < dim; i++) {
        ComplexFloat sum = COMPLEX_FLOAT_ZERO;
        for (size_t j = 0; j < dim; j++) {
            ComplexFloat path_j = path->coordinates[j];
            for (size_t k = 0; k < dim; k++) {
                ComplexFloat conn_ijk = connection->coefficients[i * dim * dim + j * dim + k];
                sum = complex_float_add(sum,
                    complex_float_multiply(
                        complex_float_multiply(conn_ijk, path_j),
                        result->coordinates[k]
                    )
                );
            }
        }
        result->coordinates[i] = complex_float_add(
            result->coordinates[i],
            sum
        );
    }
    
    return QGT_SUCCESS;
}

// Note: geometric_create_curvature and geometric_destroy_curvature are defined in quantum_geometric_curvature.c

// Hardware operations
qgt_error_t geometric_to_device(quantum_geometric_state_t* state,
                              HardwareType hardware) {
    QGT_CHECK_NULL(state);
    QGT_CHECK_STATE(validate_state());
    
    // Validate hardware type
    if (validate_hardware_type(hardware) != QGT_SUCCESS) {
        return QGT_ERROR_INVALID_HARDWARE;
    }

    if (state->hardware == hardware) {
        return QGT_SUCCESS;  // Already on target device
    }
    
    // Get hardware context
    quantum_geometric_hardware_t* hw = g_state.hardware;
    if (!hw || !hw->context) {
        return QGT_ERROR_INVALID_STATE;
    }

    // Allocate device memory and transfer data
    void* device_coordinates = NULL;
    void* device_metric = NULL;
    void* device_connection = NULL;
    qgt_error_t err = QGT_SUCCESS;
    
    switch (hardware) {
        case HARDWARE_TYPE_GPU:
            // Allocate and transfer coordinates
            if (state->coordinates) {
                device_coordinates = gpu_allocate(hw->context,
                    state->dimension * sizeof(ComplexFloat));
                if (!device_coordinates) {
                    err = QGT_ERROR_MEMORY_ALLOCATION;
                    goto cleanup;
                }
                
                err = gpu_memcpy_to_device(hw->context, device_coordinates,
                    state->coordinates, state->dimension * sizeof(ComplexFloat));
                if (err != QGT_SUCCESS) goto cleanup;
            }
            
            // Allocate and transfer metric if present
            if (state->metric) {
                device_metric = gpu_allocate(hw->context,
                    state->dimension * state->dimension * sizeof(ComplexFloat));
                if (!device_metric) {
                    err = QGT_ERROR_MEMORY_ALLOCATION;
                    goto cleanup;
                }
                
                err = gpu_memcpy_to_device(hw->context, device_metric,
                    state->metric, state->dimension * state->dimension * sizeof(ComplexFloat));
                if (err != QGT_SUCCESS) goto cleanup;
            }
            
            // Allocate and transfer connection if present
            if (state->connection) {
                device_connection = gpu_allocate(hw->context,
                    state->dimension * state->dimension * state->dimension * sizeof(ComplexFloat));
                if (!device_connection) {
                    err = QGT_ERROR_MEMORY_ALLOCATION;
                    goto cleanup;
                }
                
                err = gpu_memcpy_to_device(hw->context, device_connection,
                    state->connection, state->dimension * state->dimension * state->dimension * sizeof(ComplexFloat));
                if (err != QGT_SUCCESS) goto cleanup;
            }
            break;
            
        default:
            err = QGT_ERROR_INVALID_HARDWARE;
            goto cleanup;
    }

cleanup:
    if (err != QGT_SUCCESS) {
        if (device_coordinates) gpu_free(hw->context, device_coordinates);
        if (device_metric) gpu_free(hw->context, device_metric);
        if (device_connection) gpu_free(hw->context, device_connection);
        return err;
    }
    
    // Free host memory
    if (state->coordinates) pool_free(g_state.pool, state->coordinates);
    if (state->metric) pool_free(g_state.pool, state->metric);
    if (state->connection) pool_free(g_state.pool, state->connection);
    
    // Update state
    state->coordinates = device_coordinates;
    state->metric = device_metric;
    state->connection = device_connection;
    state->hardware = hardware;
    
    return QGT_SUCCESS;
}

qgt_error_t geometric_from_device(quantum_geometric_state_t* state,
                                HardwareType hardware) {
    QGT_CHECK_NULL(state);
    QGT_CHECK_STATE(validate_state());
    
    // Validate hardware type
    if (validate_hardware_type(hardware) != QGT_SUCCESS) {
        return QGT_ERROR_INVALID_HARDWARE;
    }

    if (state->hardware == hardware) {
        return QGT_SUCCESS;  // Already on target device
    }
    
    // Allocate host memory and transfer data
    void* host_coordinates = NULL;
    void* host_metric = NULL;
    void* host_connection = NULL;
    qgt_error_t err = QGT_SUCCESS;
    
    switch (hardware) {
        case HARDWARE_TYPE_GPU: {
            // Get hardware context
            quantum_geometric_hardware_t* hw = g_state.hardware;
            if (!hw || !hw->context) {
                err = QGT_ERROR_INVALID_STATE;
                goto cleanup;
            }

            // Allocate and transfer coordinates
            if (state->coordinates) {
                host_coordinates = pool_allocate(g_state.pool,
                    state->dimension * sizeof(ComplexFloat));
                if (!host_coordinates) {
                    err = QGT_ERROR_MEMORY_ALLOCATION;
                    goto cleanup;
                }

                err = gpu_memcpy_from_device(hw->context, host_coordinates,
                    state->coordinates, state->dimension * sizeof(ComplexFloat));
                if (err != QGT_SUCCESS) goto cleanup;
            }

            // Allocate and transfer metric if present
            if (state->metric) {
                host_metric = pool_allocate(g_state.pool,
                    state->dimension * state->dimension * sizeof(ComplexFloat));
                if (!host_metric) {
                    err = QGT_ERROR_MEMORY_ALLOCATION;
                    goto cleanup;
                }

                err = gpu_memcpy_from_device(hw->context, host_metric,
                    state->metric, state->dimension * state->dimension * sizeof(ComplexFloat));
                if (err != QGT_SUCCESS) goto cleanup;
            }

            // Allocate and transfer connection if present
            if (state->connection) {
                host_connection = pool_allocate(g_state.pool,
                    state->dimension * state->dimension * state->dimension * sizeof(ComplexFloat));
                if (!host_connection) {
                    err = QGT_ERROR_MEMORY_ALLOCATION;
                    goto cleanup;
                }

                err = gpu_memcpy_from_device(hw->context, host_connection,
                    state->connection, state->dimension * state->dimension * state->dimension * sizeof(ComplexFloat));
                if (err != QGT_SUCCESS) goto cleanup;
            }
            break;
        }

        default:
            err = QGT_ERROR_INVALID_HARDWARE;
            goto cleanup;
    }

cleanup:
    if (err != QGT_SUCCESS) {
        if (host_coordinates) pool_free(g_state.pool, host_coordinates);
        if (host_metric) pool_free(g_state.pool, host_metric);
        if (host_connection) pool_free(g_state.pool, host_connection);
        return err;
    }
    
    // Free device memory
    quantum_geometric_hardware_t* hw = NULL;
    if (state->hardware == HARDWARE_TYPE_GPU) {
        hw = g_state.hardware;
            if (!hw || !hw->context) {
                // Fallback to pool free if no valid GPU context
                if (state->coordinates) pool_free(g_state.pool, state->coordinates);
                if (state->metric) pool_free(g_state.pool, state->metric);
                if (state->connection) pool_free(g_state.pool, state->connection);
                return QGT_ERROR_INVALID_STATE;
            }
    }

    if (state->coordinates) {
        if (state->hardware == HARDWARE_TYPE_GPU) {
            gpu_free(hw->context, state->coordinates);
        } else {
            pool_free(g_state.pool, state->coordinates);
        }
    }
    if (state->metric) {
        if (state->hardware == HARDWARE_TYPE_GPU) {
            gpu_free(hw->context, state->metric);
        } else {
            pool_free(g_state.pool, state->metric);
        }
    }
    if (state->connection) {
        if (state->hardware == HARDWARE_TYPE_GPU) {
            gpu_free(hw->context, state->connection);
        } else {
            pool_free(g_state.pool, state->connection);
        }
    }
    
    // Update state with host memory
    state->coordinates = host_coordinates;
    state->metric = host_metric;
    state->connection = host_connection;
    state->hardware = hardware;
    
    return QGT_SUCCESS;
}

bool geometric_is_on_device(const quantum_geometric_state_t* state,
                          HardwareType hardware) {
    if (!state) return false;
    return state->hardware == hardware;
}

// Validation operations
qgt_error_t geometric_validate_state(const quantum_geometric_state_t* state) {
    QGT_CHECK_NULL(state);
    
    // Validate state properties
    if (state->dimension == 0 || state->dimension > QGT_MAX_DIMENSIONS) {
        return QGT_ERROR_INVALID_DIMENSION;
    }
    
    if (!state->coordinates) {
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    return QGT_SUCCESS;
}

qgt_error_t geometric_validate_curvature(const quantum_geometric_curvature_t* curvature,
                                       geometric_validation_flags_t flags,
                                       validation_result_t* result) {
    QGT_CHECK_NULL(curvature);
    QGT_CHECK_NULL(result);
    
    result->is_valid = true;
    result->error_code = QGT_SUCCESS;
    memset(result->error_message, 0, QGT_MAX_ERROR_MESSAGE_LENGTH);
    
    // Check dimension validity
    if (curvature->dimension == 0 || curvature->dimension > QGT_MAX_DIMENSIONS) {
        result->is_valid = false;
        result->error_code = QGT_ERROR_INVALID_DIMENSION;
        snprintf(result->error_message, QGT_MAX_ERROR_MESSAGE_LENGTH,
                "Invalid curvature dimension: %zu", curvature->dimension);
        return QGT_SUCCESS;
    }
    
    // Check components array
    if (!curvature->components) {
        result->is_valid = false;
        result->error_code = QGT_ERROR_INVALID_PARAMETER;
        snprintf(result->error_message, QGT_MAX_ERROR_MESSAGE_LENGTH,
                "Curvature components array is NULL");
        return QGT_SUCCESS;
    }
    
    // Check Bianchi identity if requested
    if (flags & GEOMETRIC_VALIDATION_CHECK_BIANCHI) {
        for (size_t i = 0; i < curvature->dimension; i++) {
            for (size_t j = 0; j < curvature->dimension; j++) {
                for (size_t k = 0; k < curvature->dimension; k++) {
                    for (size_t l = 0; l < curvature->dimension; l++) {
                        // First Bianchi identity: R^i_jkl + R^i_klj + R^i_ljk = 0
                        ComplexFloat sum = complex_float_add(
                            curvature->components[(((i * curvature->dimension + j) * 
                                                  curvature->dimension + k) * 
                                                  curvature->dimension) + l],
                            complex_float_add(
                                curvature->components[(((i * curvature->dimension + k) * 
                                                      curvature->dimension + l) * 
                                                      curvature->dimension) + j],
                                curvature->components[(((i * curvature->dimension + l) * 
                                                      curvature->dimension + j) * 
                                                      curvature->dimension) + k]
                            )
                        );
                        
                        if (complex_float_abs(sum) > QGT_VALIDATION_TOLERANCE) {
                            result->is_valid = false;
                            result->error_code = QGT_ERROR_VALIDATION_FAILED;
                            snprintf(result->error_message, QGT_MAX_ERROR_MESSAGE_LENGTH,
                                    "Curvature violates first Bianchi identity at indices (%zu,%zu,%zu,%zu)",
                                    i, j, k, l);
                            return QGT_SUCCESS;
                        }
                    }
                }
            }
        }
    }
    
    // Check bounds if requested
    if (flags & GEOMETRIC_VALIDATION_CHECK_BOUNDS) {
        size_t total_elements = curvature->dimension * curvature->dimension * 
                              curvature->dimension * curvature->dimension;
        for (size_t i = 0; i < total_elements; i++) {
            if (complex_float_abs(curvature->components[i]) > QGT_MAX_PARAMETER_MAGNITUDE) {
                result->is_valid = false;
                result->error_code = QGT_ERROR_VALIDATION_FAILED;
                snprintf(result->error_message, QGT_MAX_ERROR_MESSAGE_LENGTH,
                        "Curvature component at index %zu exceeds maximum magnitude", i);
                return QGT_SUCCESS;
            }
        }
    }
    
    return QGT_SUCCESS;
}

// Note: geometric_validate_metric and geometric_validate_connection are defined in quantum_geometric_validation.c

qgt_error_t geometric_clone_state(quantum_geometric_state_t** dest,
                                const quantum_geometric_state_t* src) {
    QGT_CHECK_NULL(dest);
    QGT_CHECK_NULL(src);
    QGT_CHECK_STATE(validate_state());
    
    // Create new state
    qgt_error_t err = geometric_create_state(dest, src->type, src->dimension, src->hardware);
    if (err != QGT_SUCCESS) return err;
    
    // Copy basic fields
    (*dest)->is_normalized = src->is_normalized;
    (*dest)->hardware = src->hardware;
    
    // Allocate and copy coordinates
    if (src->coordinates) {
        (*dest)->coordinates = pool_allocate(g_state.pool,
            src->dimension * sizeof(ComplexFloat));
        if (!(*dest)->coordinates) {
            geometric_destroy_state(*dest);
            return QGT_ERROR_MEMORY_ALLOCATION;
        }
        memcpy((*dest)->coordinates, src->coordinates,
               src->dimension * sizeof(ComplexFloat));
    }
    
    // Allocate and copy metric if present
    if (src->metric) {
        (*dest)->metric = pool_allocate(g_state.pool,
            src->dimension * src->dimension * sizeof(ComplexFloat));
        if (!(*dest)->metric) {
            geometric_destroy_state(*dest);
            return QGT_ERROR_MEMORY_ALLOCATION;
        }
        memcpy((*dest)->metric, src->metric,
               src->dimension * src->dimension * sizeof(ComplexFloat));
    }
    
    // Allocate and copy connection if present
    if (src->connection) {
        (*dest)->connection = pool_allocate(g_state.pool,
            src->dimension * src->dimension * src->dimension * sizeof(ComplexFloat));
        if (!(*dest)->connection) {
            geometric_destroy_state(*dest);
            return QGT_ERROR_MEMORY_ALLOCATION;
        }
        memcpy((*dest)->connection, src->connection,
               src->dimension * src->dimension * src->dimension * sizeof(ComplexFloat));
    }
    
    return QGT_SUCCESS;
}

qgt_error_t geometric_sectional_curvature(double* curvature,
                                        const quantum_geometric_curvature_t* R,
                                        const quantum_geometric_state_t* u,
                                        const quantum_geometric_state_t* v) {
    QGT_CHECK_NULL(curvature);
    QGT_CHECK_NULL(R);
    QGT_CHECK_NULL(u);
    QGT_CHECK_NULL(v);
    QGT_CHECK_STATE(validate_state());
    QGT_CHECK_STATE(validate_dimensions(R->dimension, u->dimension));
    QGT_CHECK_STATE(validate_dimensions(R->dimension, v->dimension));
    
    size_t dim = R->dimension;
    size_t block_size = QGT_BLOCK_SIZE / sizeof(ComplexFloat);
    ComplexFloat sum = COMPLEX_FLOAT_ZERO;
    
    // OpenMP reduction not supported for ComplexFloat, use thread-local storage
    #pragma omp parallel if(dim > QGT_PARALLEL_THRESHOLD)
    {
        ComplexFloat local_sum = COMPLEX_FLOAT_ZERO;
        
        // Block computation for better cache utilization
        #pragma omp for collapse(4) nowait
        for (size_t i_block = 0; i_block < dim; i_block += block_size) {
            for (size_t j_block = 0; j_block < dim; j_block += block_size) {
                for (size_t k_block = 0; k_block < dim; k_block += block_size) {
                    for (size_t l_block = 0; l_block < dim; l_block += block_size) {
                        size_t i_end = (i_block + block_size < dim) ? i_block + block_size : dim;
                        size_t j_end = (j_block + block_size < dim) ? j_block + block_size : dim;
                        size_t k_end = (k_block + block_size < dim) ? k_block + block_size : dim;
                        size_t l_end = (l_block + block_size < dim) ? l_block + block_size : dim;
                        
                        // Process blocks
                        for (size_t i = i_block; i < i_end; i++) {
                            ComplexFloat u_i_conj = complex_float_conjugate(u->coordinates[i]);
                            
                            for (size_t j = j_block; j < j_end; j++) {
                                ComplexFloat v_j = v->coordinates[j];
                                
                                for (size_t k = k_block; k < k_end; k++) {
                                    ComplexFloat u_k = u->coordinates[k];
                                    
                                    // Inner loop tiled for L1 cache
                                    for (size_t l = l_block; l < l_end; l += QGT_TILE_SIZE) {
                                        size_t l_tile_end = (l + QGT_TILE_SIZE < l_end) ? l + QGT_TILE_SIZE : l_end;
                                        
                                        for (size_t ll = l; ll < l_tile_end; ll++) {
                                            ComplexFloat term = complex_float_multiply(
                                                complex_float_multiply(
                                                    u_i_conj,
                                                    R->components[((i * dim + j) * dim + k) * dim + ll]
                                                ),
                                                complex_float_multiply(
                                                    v_j,
                                                    complex_float_multiply(
                                                        u_k,
                                                        v->coordinates[ll]
                                                    )
                                                )
                                            );
                                            local_sum = complex_float_add(local_sum, term);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        #pragma omp critical
        {
            sum = complex_float_add(sum, local_sum);
        }
    }
    
    *curvature = complex_float_abs(sum);
    return QGT_SUCCESS;
}
