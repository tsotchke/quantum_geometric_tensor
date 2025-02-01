#include "quantum_geometric/core/quantum_geometric_metric.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_geometric_constants.h"
#include <stdlib.h>
#include <string.h>

// Create geometric metric
qgt_error_t geometric_create_metric(quantum_geometric_metric_t** metric,
                                  geometric_metric_type_t type,
                                  size_t dimension) {
    QGT_CHECK_NULL(metric);
    QGT_CHECK_ARGUMENT(dimension > 0 && dimension <= QGT_MAX_DIMENSIONS);
    
    // Allocate aligned memory for better SIMD performance
    *metric = (quantum_geometric_metric_t*)aligned_alloc(32, sizeof(quantum_geometric_metric_t));
    if (!*metric) {
        return QGT_ERROR_ALLOCATION_FAILED;
    }
    
    // Allocate aligned metric tensor components
    size_t size = dimension * dimension * sizeof(ComplexFloat);
    (*metric)->components = (ComplexFloat*)aligned_alloc(32, size);
    if (!(*metric)->components) {
        free(*metric);
        return QGT_ERROR_ALLOCATION_FAILED;
    }
    
    (*metric)->type = type;
    (*metric)->dimension = dimension;
    
    // Initialize metric based on type
    switch (type) {
        case GEOMETRIC_METRIC_EUCLIDEAN:
            // Initialize Euclidean metric (identity matrix)
            for (size_t i = 0; i < dimension; i++) {
                for (size_t j = 0; j < dimension; j++) {
                    (*metric)->components[i * dimension + j] = 
                        (i == j) ? COMPLEX_FLOAT_ONE : COMPLEX_FLOAT_ZERO;
                }
            }
            break;
            
        case GEOMETRIC_METRIC_MINKOWSKI:
            // Initialize Minkowski metric (diagonal with -1,1,1,1,...)
            for (size_t i = 0; i < dimension; i++) {
                for (size_t j = 0; j < dimension; j++) {
                    (*metric)->components[i * dimension + j] = COMPLEX_FLOAT_ZERO;
                    if (i == j) {
                        (*metric)->components[i * dimension + j].real = (i == 0) ? -1.0f : 1.0f;
                    }
                }
            }
            break;
            
        case GEOMETRIC_METRIC_FUBINI_STUDY:
            // Initialize Fubini-Study metric for quantum state space
            for (size_t i = 0; i < dimension; i++) {
                for (size_t j = 0; j < dimension; j++) {
                    (*metric)->components[i * dimension + j] = 
                        (i == j) ? COMPLEX_FLOAT_ONE : COMPLEX_FLOAT_ZERO;
                }
            }
            break;
            
        default:
            free((*metric)->components);
            free(*metric);
            return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    return QGT_SUCCESS;
}

// Destroy geometric metric
void geometric_destroy_metric(quantum_geometric_metric_t* metric) {
    if (metric) {
        free(metric->components);
        free(metric);
    }
}

// Compute geometric metric
qgt_error_t geometric_compute_metric(quantum_geometric_metric_t* metric,
                                   const quantum_state_t* state) {
    QGT_CHECK_NULL(metric);
    QGT_CHECK_NULL(state);
    
    if (metric->dimension != state->dimension) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    size_t dim = state->dimension;
    const size_t block_size = 32; // Cache line friendly block size
    
    // Compute metric components based on state
    switch (metric->type) {
        case GEOMETRIC_METRIC_EUCLIDEAN:
            // Euclidean metric remains constant
            break;
            
        case GEOMETRIC_METRIC_FUBINI_STUDY:
            // Compute Fubini-Study metric with blocking and SIMD
            #pragma omp parallel for collapse(2) if(dim > QGT_PARALLEL_THRESHOLD)
            for (size_t i_block = 0; i_block < dim; i_block += block_size) {
                for (size_t j_block = 0; j_block < dim; j_block += block_size) {
                    size_t i_end = (i_block + block_size < dim) ? i_block + block_size : dim;
                    size_t j_end = (j_block + block_size < dim) ? j_block + block_size : dim;
                    
                    // Prefetch next blocks
                    if (i_block + block_size < dim) {
                        __builtin_prefetch(&state->coordinates[i_block + block_size], 0, 3);
                    }
                    
                    for (size_t i = i_block; i < i_end; i++) {
                        ComplexFloat psi_i = state->coordinates[i];
                        ComplexFloat psi_i_conj = complex_float_conjugate(psi_i);
                        
                        #ifdef __AVX512F__
                        __m512 vpsi_i_real = _mm512_set1_ps(psi_i_conj.real);
                        __m512 vpsi_i_imag = _mm512_set1_ps(psi_i_conj.imag);
                        
                        for (size_t j = j_block; j < j_end - 7; j += 8) {
                            // Load 8 complex numbers
                            __m512 vpsi_j = _mm512_loadu_ps((float*)&state->coordinates[j]);
                            
                            // Complex multiplication
                            __m512 vreal = _mm512_fmsub_ps(vpsi_i_real,
                                                         _mm512_shuffle_ps(vpsi_j, vpsi_j, 0x50),
                                                         _mm512_mul_ps(vpsi_i_imag,
                                                                     _mm512_shuffle_ps(vpsi_j, vpsi_j, 0xFA)));
                            __m512 vimag = _mm512_fmadd_ps(vpsi_i_real,
                                                         _mm512_shuffle_ps(vpsi_j, vpsi_j, 0xFA),
                                                         _mm512_mul_ps(vpsi_i_imag,
                                                                     _mm512_shuffle_ps(vpsi_j, vpsi_j, 0x50)));
                            
                            // Subtract from identity
                            __m512 videntity = _mm512_set_ps(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                                                           0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
                            vreal = _mm512_sub_ps(videntity, vreal);
                            vimag = _mm512_sub_ps(_mm512_setzero_ps(), vimag);
                            
                            // Store result
                            _mm512_storeu_ps((float*)&metric->components[i * dim + j],
                                           _mm512_unpacklo_ps(vreal, vimag));
                        }
                        
                        #elif defined(__ARM_NEON)
                        float32x4_t vpsi_i_real = vdupq_n_f32(psi_i_conj.real);
                        float32x4_t vpsi_i_imag = vdupq_n_f32(psi_i_conj.imag);
                        
                        for (size_t j = j_block; j < j_end - 3; j += 4) {
                            float32x4_t vpsi_j = vld1q_f32((float*)&state->coordinates[j]);
                            
                            // Complex multiplication
                            float32x4_t vreal = vmulq_f32(vpsi_i_real, vpsi_j);
                            float32x4_t vimag = vmulq_f32(vpsi_i_imag, vpsi_j);
                            
                            // Subtract from identity
                            float32x4_t videntity = vdupq_n_f32(0.0f);
                            videntity = vsetq_lane_f32(1.0f, videntity, 0);
                            vreal = vsubq_f32(videntity, vreal);
                            vimag = vnegq_f32(vimag);
                            
                            vst1q_f32((float*)&metric->components[i * dim + j], vreal);
                            vst1q_f32((float*)&metric->components[i * dim + j + 2], vimag);
                        }
                        #endif
                        
                        // Handle remaining elements
                        for (size_t j = j_block + ((j_end - j_block) & ~7); j < j_end; j++) {
                            ComplexFloat prod = complex_float_multiply(psi_i_conj, state->coordinates[j]);
                            metric->components[i * dim + j] = complex_float_subtract(
                                (i == j) ? COMPLEX_FLOAT_ONE : COMPLEX_FLOAT_ZERO,
                                prod
                            );
                        }
                    }
                }
            }
            break;
            
        default:
            return QGT_ERROR_NOT_IMPLEMENTED;
    }
    
    return QGT_SUCCESS;
}

// Clone geometric metric
qgt_error_t geometric_clone_metric(quantum_geometric_metric_t** dest,
                                 const quantum_geometric_metric_t* src) {
    QGT_CHECK_NULL(dest);
    QGT_CHECK_NULL(src);
    
    qgt_error_t err = geometric_create_metric(dest, src->type, src->dimension);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    size_t size = src->dimension * src->dimension * sizeof(ComplexFloat);
    memcpy((*dest)->components, src->components, size);
    
    return QGT_SUCCESS;
}

// Transform geometric metric
qgt_error_t geometric_transform_metric(quantum_geometric_metric_t* result,
                                     const quantum_geometric_metric_t* metric,
                                     const quantum_geometric_tensor_t* transform) {
    QGT_CHECK_NULL(result);
    QGT_CHECK_NULL(metric);
    QGT_CHECK_NULL(transform);
    
    if (transform->rank != 2 || 
        transform->dimensions[0] != metric->dimension ||
        transform->dimensions[1] != metric->dimension ||
        result->dimension != metric->dimension) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    size_t dim = metric->dimension;
    const size_t block_size = 32; // Cache-friendly block size
    
    // Zero initialize result
    memset(result->components, 0, dim * dim * sizeof(ComplexFloat));
    
    // Block matrix multiplication for g' = T^t g T
    #pragma omp parallel for collapse(2) if(dim > QGT_PARALLEL_THRESHOLD)
    for (size_t i_block = 0; i_block < dim; i_block += block_size) {
        for (size_t j_block = 0; j_block < dim; j_block += block_size) {
            size_t i_end = (i_block + block_size < dim) ? i_block + block_size : dim;
            size_t j_end = (j_block + block_size < dim) ? j_block + block_size : dim;
            
            // Prefetch next blocks
            if (i_block + block_size < dim) {
                __builtin_prefetch(&transform->components[(i_block + block_size) * dim], 0, 3);
                __builtin_prefetch(&metric->components[(i_block + block_size) * dim], 0, 3);
            }
            
            // Process block
            for (size_t i = i_block; i < i_end; i++) {
                for (size_t j = j_block; j < j_end; j++) {
                    ComplexFloat sum = COMPLEX_FLOAT_ZERO;
                    
                    // First matrix multiply: temp = g T
                    #ifdef __AVX512F__
                    __m512 vsum_real = _mm512_setzero_ps();
                    __m512 vsum_imag = _mm512_setzero_ps();
                    
                    for (size_t k = 0; k < dim; k += 8) {
                        // Load metric and transform components
                        __m512 vmetric = _mm512_loadu_ps((float*)&metric->components[i * dim + k]);
                        __m512 vtransform = _mm512_loadu_ps((float*)&transform->components[k * dim + j]);
                        
                        // Complex multiply-accumulate
                        __m512 vreal = _mm512_fmsub_ps(
                            _mm512_shuffle_ps(vmetric, vmetric, 0x50),
                            _mm512_shuffle_ps(vtransform, vtransform, 0x50),
                            _mm512_mul_ps(
                                _mm512_shuffle_ps(vmetric, vmetric, 0xFA),
                                _mm512_shuffle_ps(vtransform, vtransform, 0xFA)
                            )
                        );
                        
                        __m512 vimag = _mm512_fmadd_ps(
                            _mm512_shuffle_ps(vmetric, vmetric, 0x50),
                            _mm512_shuffle_ps(vtransform, vtransform, 0xFA),
                            _mm512_mul_ps(
                                _mm512_shuffle_ps(vmetric, vmetric, 0xFA),
                                _mm512_shuffle_ps(vtransform, vtransform, 0x50)
                            )
                        );
                        
                        vsum_real = _mm512_add_ps(vsum_real, vreal);
                        vsum_imag = _mm512_add_ps(vsum_imag, vimag);
                    }
                    
                    // Reduce sum
                    ComplexFloat temp;
                    temp.real = _mm512_reduce_add_ps(vsum_real);
                    temp.imag = _mm512_reduce_add_ps(vsum_imag);
                    
                    #elif defined(__ARM_NEON)
                    float32x4_t vsum_real = vdupq_n_f32(0.0f);
                    float32x4_t vsum_imag = vdupq_n_f32(0.0f);
                    
                    for (size_t k = 0; k < dim; k += 4) {
                        float32x4_t vmetric = vld1q_f32((float*)&metric->components[i * dim + k]);
                        float32x4_t vtransform = vld1q_f32((float*)&transform->components[k * dim + j]);
                        
                        float32x4_t vreal = vmulq_f32(vmetric, vtransform);
                        float32x4_t vimag = vmulq_f32(
                            vrev64q_f32(vmetric),
                            vrev64q_f32(vtransform)
                        );
                        
                        vsum_real = vaddq_f32(vsum_real, vreal);
                        vsum_imag = vaddq_f32(vsum_imag, vimag);
                    }
                    
                    // Reduce sum
                    ComplexFloat temp;
                    float32x2_t vsum2_real = vadd_f32(vget_low_f32(vsum_real), vget_high_f32(vsum_real));
                    float32x2_t vsum2_imag = vadd_f32(vget_low_f32(vsum_imag), vget_high_f32(vsum_imag));
                    temp.real = vget_lane_f32(vsum2_real, 0) + vget_lane_f32(vsum2_real, 1);
                    temp.imag = vget_lane_f32(vsum2_imag, 0) + vget_lane_f32(vsum2_imag, 1);
                    
                    #else
                    ComplexFloat temp = COMPLEX_FLOAT_ZERO;
                    #endif
                    
                    // Handle remaining elements and second multiply: result = T^t temp
                    for (size_t k = 0; k < dim; k++) {
                        ComplexFloat t = complex_float_multiply(
                            complex_float_conjugate(transform->components[k * dim + i]),
                            temp
                        );
                        sum = complex_float_add(sum, t);
                    }
                    
                    result->components[i * dim + j] = sum;
                }
            }
        }
    }
    
    return QGT_SUCCESS;
}
