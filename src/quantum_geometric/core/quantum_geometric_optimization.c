#include "quantum_geometric/core/quantum_geometric_optimization.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_geometric_constants.h"
#include "quantum_geometric/core/memory_pool.h"
#include "quantum_geometric/core/simd_operations.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/core/system_dependencies.h"
#include "quantum_geometric/core/quantum_geometric_config.h"
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#define omp_get_max_threads() 1
#endif

// Thread-local storage for intermediate results
static THREAD_LOCAL double* thread_local_buffer = NULL;
static THREAD_LOCAL size_t thread_local_buffer_size = 0;

// Initialize thread-local storage
static qgt_error_t init_thread_local_storage(size_t size) {
    if (thread_local_buffer_size < size) {
        if (thread_local_buffer) {
            pool_free(g_state.pool, thread_local_buffer);
        }
        thread_local_buffer = pool_malloc(g_state.pool, size);
        if (!thread_local_buffer) {
            return QGT_ERROR_MEMORY_ALLOCATION;
        }
        thread_local_buffer_size = size;
    }
    return QGT_SUCCESS;
}

// Cleanup thread-local storage
static void cleanup_thread_local_storage(void) {
    if (thread_local_buffer) {
        pool_free(g_state.pool, thread_local_buffer);
        thread_local_buffer = NULL;
        thread_local_buffer_size = 0;
    }
}

// Global state from quantum_geometric_core.h
extern struct {
    bool initialized;
    quantum_geometric_config_t config;
    MemoryPool* pool;
    quantum_geometric_hardware_t* hardware;
} g_state;

// Block sizes and optimization settings
#define QGT_OPT_BLOCK_SIZE QGT_MAX_BLOCK_SIZE       // 1024 - Maximizes SIMD vectorization
#define QGT_OPT_TILE_SIZE QGT_WARP_SIZE             // 32 - Matches GPU warp size and cache line
#define QGT_OPT_PARALLEL_THRESHOLD (QGT_MAX_THREADS * QGT_WARP_SIZE) // Optimal thread workload balance

// Prefetch settings optimized for data structure sizes and cache line boundaries
// Each ComplexFloat is 8 bytes (2 floats), so we can fit 8 ComplexFloat per 64-byte cache line
#define QGT_BASE_PREFETCH_DISTANCE 8  // Number of ComplexFloats per cache line
// Prefetch multiple cache lines ahead based on data access patterns
#define QGT_METRIC_PREFETCH (QGT_BASE_PREFETCH_DISTANCE * 2)      // Prefetch 2 cache lines worth of metric components
#define QGT_CONNECTION_PREFETCH (QGT_BASE_PREFETCH_DISTANCE * 2)  // Prefetch 2 cache lines worth of connection coefficients
#define QGT_CURVATURE_PREFETCH (QGT_BASE_PREFETCH_DISTANCE * 2)   // Prefetch 2 cache lines worth of curvature components

// Create geometric optimization
qgt_error_t geometric_create_optimization(quantum_geometric_optimization_t** optimization,
                                        geometric_optimization_type_t type,
                                        size_t dimension) {
    QGT_CHECK_NULL(optimization);
    QGT_CHECK_ARGUMENT(dimension > 0 && dimension <= QGT_MAX_DIMENSIONS);
    QGT_CHECK_STATE(validate_state());
    
    *optimization = pool_malloc(g_state.pool, sizeof(quantum_geometric_optimization_t));
    if (!*optimization) {
        return QGT_ERROR_MEMORY_ALLOCATION;
    }
    
    // Allocate optimization parameters with alignment for SIMD
    size_t size = dimension * sizeof(ComplexFloat);
    size_t aligned_size = (size + QGT_CACHE_LINE_SIZE - 1) & ~(QGT_CACHE_LINE_SIZE - 1);
    (*optimization)->parameters = pool_malloc(g_state.pool, aligned_size);
    if (!(*optimization)->parameters) {
        pool_free(g_state.pool, *optimization);
        return QGT_ERROR_MEMORY_ALLOCATION;
    }
    
    (*optimization)->type = type;
    (*optimization)->dimension = dimension;
    (*optimization)->iterations = 0;
    (*optimization)->convergence_threshold = QGT_CONVERGENCE_TOL;
    (*optimization)->learning_rate = QGT_LEARNING_RATE;
    
    // Initialize parameters to zero
    memset((*optimization)->parameters, 0, aligned_size);
    
    return QGT_SUCCESS;
}

// Destroy geometric optimization
void geometric_destroy_optimization(quantum_geometric_optimization_t* optimization) {
    if (!optimization || !g_state.initialized) return;
    
    if (optimization->parameters) {
        pool_free(g_state.pool, optimization->parameters);
    }
    pool_free(g_state.pool, optimization);
}

// Clone geometric optimization
qgt_error_t geometric_clone_optimization(quantum_geometric_optimization_t** dest,
                                       const quantum_geometric_optimization_t* src) {
    QGT_CHECK_NULL(dest);
    QGT_CHECK_NULL(src);
    QGT_CHECK_STATE(validate_state());
    
    qgt_error_t err = geometric_create_optimization(dest, src->type, src->dimension);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    size_t size = src->dimension * sizeof(ComplexFloat);
    size_t aligned_size = (size + QGT_CACHE_LINE_SIZE - 1) & ~(QGT_CACHE_LINE_SIZE - 1);
    memcpy((*dest)->parameters, src->parameters, aligned_size);
    (*dest)->iterations = src->iterations;
    (*dest)->convergence_threshold = src->convergence_threshold;
    (*dest)->learning_rate = src->learning_rate;
    
    return QGT_SUCCESS;
}

// Optimize geometric parameters
qgt_error_t geometric_optimize_parameters(quantum_geometric_optimization_t* optimization,
                                        const quantum_geometric_metric_t* metric,
                                        const quantum_geometric_connection_t* connection,
                                        const quantum_geometric_curvature_t* curvature) {
    QGT_CHECK_NULL(optimization);
    QGT_CHECK_NULL(metric);
    QGT_CHECK_NULL(connection);
    QGT_CHECK_NULL(curvature);
    QGT_CHECK_STATE(validate_state());
    
    if (optimization->dimension != metric->dimension ||
        optimization->dimension != connection->dimension ||
        optimization->dimension != curvature->dimension) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    size_t dim = optimization->dimension;
    size_t block_size = QGT_OPT_BLOCK_SIZE / sizeof(ComplexFloat);
    
    // Initialize thread-local storage
    qgt_error_t err = init_thread_local_storage(block_size * sizeof(ComplexFloat));
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    // Perform optimization based on type
    switch (optimization->type) {
        case GEOMETRIC_OPTIMIZATION_GRADIENT:
            #pragma omp parallel if(dim > QGT_OPT_PARALLEL_THRESHOLD)
            {
                // Thread-local gradient accumulator
                ComplexFloat* local_gradients = (ComplexFloat*)thread_local_buffer;
                
                #pragma omp for schedule(dynamic, QGT_WARP_SIZE) nowait
                for (size_t i_block = 0; i_block < dim; i_block += block_size) {
                    size_t i_end = (i_block + block_size < dim) ? i_block + block_size : dim;
                    
                    // Process block with SIMD
                    #ifdef __AVX512F__
                    for (size_t i = i_block; i < i_end - 7; i += 8) {
                        __m512d vgradient_real = _mm512_setzero_pd();
                        __m512d vgradient_imag = _mm512_setzero_pd();
                        
                        for (size_t j = 0; j < dim; j++) {
                            // Prefetch next iteration's data with structure-specific distances
                            _mm_prefetch(&metric->components[i * dim + j + QGT_METRIC_PREFETCH], _MM_HINT_T0);
                            _mm_prefetch(&connection->coefficients[(i * dim + j + QGT_CONNECTION_PREFETCH) * dim], _MM_HINT_T0);
                            _mm_prefetch(&curvature->components[((i * dim) + QGT_CURVATURE_PREFETCH) * dim * dim], _MM_HINT_T0);
                            
                            // Load current iteration's data
                            __m512d vmetric = _mm512_loadu_pd((double*)&metric->components[i * dim + j]);
                            __m512d vconn = _mm512_loadu_pd((double*)&connection->coefficients[(i * dim + j) * dim]);
                            __m512d vcurv = _mm512_loadu_pd((double*)&curvature->components[(i * dim) * dim * dim]);
                            
                            // Load and unpack complex numbers
                            __m512d vmetric_real = _mm512_permutex_pd(vmetric, 0x88);  // Extract even elements (real)
                            __m512d vmetric_imag = _mm512_permutex_pd(vmetric, 0xdd);  // Extract odd elements (imag)
                            __m512d vconn_real = _mm512_permutex_pd(vconn, 0x88);
                            __m512d vconn_imag = _mm512_permutex_pd(vconn, 0xdd);
                            __m512d vcurv_real = _mm512_permutex_pd(vcurv, 0x88);
                            __m512d vcurv_imag = _mm512_permutex_pd(vcurv, 0xdd);
                            
                            // Add connection and curvature terms
                            __m512d vsum_real = _mm512_add_pd(vconn_real, vcurv_real);
                            __m512d vsum_imag = _mm512_add_pd(vconn_imag, vcurv_imag);
                            
                            // Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
                            __m512d ac = _mm512_mul_pd(vmetric_real, vsum_real);
                            __m512d bd = _mm512_mul_pd(vmetric_imag, vsum_imag);
                            __m512d ad = _mm512_mul_pd(vmetric_real, vsum_imag);
                            __m512d bc = _mm512_mul_pd(vmetric_imag, vsum_real);
                            
                            // Accumulate results
                            vgradient_real = _mm512_add_pd(vgradient_real, _mm512_sub_pd(ac, bd));
                            vgradient_imag = _mm512_add_pd(vgradient_imag, _mm512_add_pd(ad, bc));
                        }
                        
                        // Interleave real and imaginary parts for storage
                        __m512d vresult1 = _mm512_permutex2var_pd(vgradient_real, 
                            _mm512_set_epi64(0xE, 0xC, 0xA, 0x8, 0x6, 0x4, 0x2, 0x0), vgradient_imag);
                        __m512d vresult2 = _mm512_permutex2var_pd(vgradient_real,
                            _mm512_set_epi64(0xF, 0xD, 0xB, 0x9, 0x7, 0x5, 0x3, 0x1), vgradient_imag);
                        
                        // Store interleaved results
                        _mm512_storeu_pd((double*)&local_gradients[i - i_block], vresult1);
                        _mm512_storeu_pd((double*)&local_gradients[i - i_block + 4], vresult2);
                    }
                    
                    #elif defined(__ARM_NEON)
                    for (size_t i = i_block; i < i_end - 3; i += 4) {
                        float32x4_t vgradient_real = vdupq_n_f32(0.0f);
                        float32x4_t vgradient_imag = vdupq_n_f32(0.0f);
                        
                        for (size_t j = 0; j < dim; j++) {
                            // Prefetch next iteration's data with structure-specific distances
                            __builtin_prefetch(&metric->components[i * dim + j + QGT_METRIC_PREFETCH]);
                            __builtin_prefetch(&connection->coefficients[(i * dim + j + QGT_CONNECTION_PREFETCH) * dim]);
                            __builtin_prefetch(&curvature->components[((i * dim) + QGT_CURVATURE_PREFETCH) * dim * dim]);
                            
                            // Load current iteration's data
                            float32x4x2_t vmetric = vld2q_f32((float*)&metric->components[i * dim + j]);
                            float32x4x2_t vconn = vld2q_f32((float*)&connection->coefficients[(i * dim + j) * dim]);
                            float32x4x2_t vcurv = vld2q_f32((float*)&curvature->components[(i * dim) * dim * dim]);
                            
                            // Complex multiply-add
                            float32x4_t vconn_plus_curv_real = vaddq_f32(vconn.val[0], vcurv.val[0]);
                            float32x4_t vconn_plus_curv_imag = vaddq_f32(vconn.val[1], vcurv.val[1]);
                            
                            // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
                            float32x4_t ac = vmulq_f32(vmetric.val[0], vconn_plus_curv_real);
                            float32x4_t bd = vmulq_f32(vmetric.val[1], vconn_plus_curv_imag);
                            float32x4_t ad = vmulq_f32(vmetric.val[0], vconn_plus_curv_imag);
                            float32x4_t bc = vmulq_f32(vmetric.val[1], vconn_plus_curv_real);
                            
                            vgradient_real = vaddq_f32(vgradient_real, vsubq_f32(ac, bd));
                            vgradient_imag = vaddq_f32(vgradient_imag, vaddq_f32(ad, bc));
                        }
                        
                        // Store gradients
                        float32x4x2_t vresult;
                        vresult.val[0] = vgradient_real;
                        vresult.val[1] = vgradient_imag;
                        vst2q_f32((float*)&local_gradients[i - i_block], vresult);
                    }
                    
                    #else
                    // Scalar fallback with tiling
                    for (size_t i = i_block; i < i_end; i++) {
                        ComplexFloat gradient = COMPLEX_FLOAT_ZERO;
                        
                        for (size_t j = 0; j < dim; j += QGT_OPT_TILE_SIZE) {
                            size_t j_end = (j + QGT_OPT_TILE_SIZE < dim) ? j + QGT_OPT_TILE_SIZE : dim;
                            for (size_t jj = j; jj < j_end; jj++) {
                                // Prefetch data for next tile with structure-specific distances
                                if (jj + QGT_METRIC_PREFETCH < j_end) {
                                    __builtin_prefetch(&metric->components[i * dim + jj + QGT_METRIC_PREFETCH]);
                                    __builtin_prefetch(&connection->coefficients[(i * dim + jj + QGT_CONNECTION_PREFETCH) * dim]);
                                    __builtin_prefetch(&curvature->components[((i * dim) + QGT_CURVATURE_PREFETCH) * dim * dim]);
                                }
                                
                                ComplexFloat metric_term = metric->components[i * dim + jj];
                                ComplexFloat connection_term = connection->coefficients[(i * dim + jj) * dim];
                                ComplexFloat curvature_term = curvature->components[(i * dim) * dim * dim];
                                
                                gradient = complex_float_add(gradient,
                                    complex_float_multiply(metric_term,
                                        complex_float_add(connection_term, curvature_term)));
                            }
                        }
                        
                        local_gradients[i - i_block] = gradient;
                    }
                    #endif
                    
                    // Update parameters using gradient descent
                    ComplexFloat learning_rate = complex_float_create(optimization->learning_rate, 0.0f);
                    for (size_t i = i_block; i < i_end; i++) {
                        optimization->parameters[i] = complex_float_subtract(
                            optimization->parameters[i],
                            complex_float_multiply(learning_rate, local_gradients[i - i_block])
                        );
                    }
                }
            }
            break;
            
        default:
            cleanup_thread_local_storage();
            return QGT_ERROR_NOT_IMPLEMENTED;
    }
    
    optimization->iterations++;
    cleanup_thread_local_storage();
    
    return QGT_SUCCESS;
}

// Check optimization convergence
qgt_error_t geometric_check_convergence(const quantum_geometric_optimization_t* optimization,
                                      bool* converged) {
    QGT_CHECK_NULL(optimization);
    QGT_CHECK_NULL(converged);
    QGT_CHECK_STATE(validate_state());
    
    size_t dim = optimization->dimension;
    size_t block_size = QGT_OPT_BLOCK_SIZE / sizeof(double);
    
    // Allocate thread-local storage for norm calculation
    double* thread_norms = pool_malloc(g_state.pool, omp_get_max_threads() * sizeof(double));
    if (!thread_norms) return QGT_ERROR_MEMORY_ALLOCATION;
    memset(thread_norms, 0, omp_get_max_threads() * sizeof(double));
    
    // Block computation for better cache utilization
    #pragma omp parallel if(dim > QGT_OPT_PARALLEL_THRESHOLD)
    {
        int thread_id = omp_get_thread_num();
        double local_norm = 0.0;
        
        #pragma omp for schedule(dynamic, QGT_WARP_SIZE) nowait
        for (size_t i_block = 0; i_block < dim; i_block += block_size) {
            size_t i_end = (i_block + block_size < dim) ? i_block + block_size : dim;
            
            // Process blocks with SIMD
            #ifdef __AVX512F__
            __m512d vnorm = _mm512_setzero_pd();
            
            for (size_t i = i_block; i < i_end - 7; i += 8) {
                // Load complex parameters
                __m512d vparams = _mm512_loadu_pd((double*)&optimization->parameters[i]);
                __m512d vparams_real = _mm512_permutex_pd(vparams, 0x88);  // Extract real parts
                __m512d vparams_imag = _mm512_permutex_pd(vparams, 0xdd);  // Extract imaginary parts
                
                // Calculate magnitude: sqrt(real^2 + imag^2)
                __m512d vreal_squared = _mm512_mul_pd(vparams_real, vparams_real);
                __m512d vimag_squared = _mm512_mul_pd(vparams_imag, vparams_imag);
                __m512d vmagnitude = _mm512_sqrt_pd(_mm512_add_pd(vreal_squared, vimag_squared));
                
                vnorm = _mm512_add_pd(vnorm, vmagnitude);
            }
            
            local_norm += _mm512_reduce_add_pd(vnorm);
            
            // Handle remaining elements
            for (size_t i = i_block + ((i_end - i_block) & ~7); i < i_end; i++) {
                local_norm += complex_float_abs(optimization->parameters[i]);
            }
            
            #elif defined(__ARM_NEON)
            float32x4_t vnorm = vdupq_n_f32(0.0f);
            
            for (size_t i = i_block; i < i_end - 3; i += 4) {
                float32x4x2_t vparams = vld2q_f32((float*)&optimization->parameters[i]);
                float32x4_t vreal = vparams.val[0];
                float32x4_t vimag = vparams.val[1];
                float32x4_t vabs = vaddq_f32(vmulq_f32(vreal, vreal), vmulq_f32(vimag, vimag));
                vnorm = vaddq_f32(vnorm, vsqrtq_f32(vabs));
            }
            
            float32x2_t vsum = vadd_f32(vget_low_f32(vnorm), vget_high_f32(vnorm));
            local_norm += vget_lane_f32(vpadd_f32(vsum, vsum), 0);
            
            // Handle remaining elements
            for (size_t i = i_block + ((i_end - i_block) & ~3); i < i_end; i++) {
                local_norm += complex_float_abs(optimization->parameters[i]);
            }
            
            #else
            // Scalar fallback with tiling for cache efficiency
            for (size_t i = i_block; i < i_end; i += QGT_OPT_TILE_SIZE) {
                size_t i_tile_end = (i + QGT_OPT_TILE_SIZE < i_end) ? i + QGT_OPT_TILE_SIZE : i_end;
                for (size_t ii = i; ii < i_tile_end; ii++) {
                    local_norm += complex_float_abs(optimization->parameters[ii]);
                }
            }
            #endif
        }
        
        thread_norms[thread_id] = local_norm;
    }
    
    // Reduce thread-local norms
    double total_norm = 0.0;
    for (int i = 0; i < omp_get_max_threads(); i++) {
        total_norm += thread_norms[i];
    }
    
    pool_free(g_state.pool, thread_norms);
    
    *converged = (total_norm < optimization->convergence_threshold);
    return QGT_SUCCESS;
}
