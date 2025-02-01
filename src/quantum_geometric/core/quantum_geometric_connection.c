#include "quantum_geometric/core/quantum_geometric_connection.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_geometric_constants.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Helper function to compute metric derivatives using finite differences
static void compute_metric_derivatives(ComplexFloat* d_metric,
                                    const ComplexFloat* metric,
                                    size_t dim,
                                    size_t direction,
                                    float epsilon) {
    const size_t metric_size = dim * dim;
    const float inv_2eps = 0.5f / epsilon;
    
    // Use central difference formula for better accuracy
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            size_t idx = i * dim + j;
            ComplexFloat forward = metric[idx];
            ComplexFloat backward = metric[idx];
            
            // Perturb the appropriate coordinate
            if (i == direction) {
                forward.real += epsilon;
                backward.real -= epsilon;
            }
            
            // Compute derivative
            d_metric[idx].real = (forward.real - backward.real) * inv_2eps;
            d_metric[idx].imag = (forward.imag - backward.imag) * inv_2eps;
        }
    }
}

// Helper function to compute inverse metric using block LU decomposition
static qgt_error_t compute_inverse_metric(ComplexFloat* inv_metric,
                                        const ComplexFloat* metric,
                                        size_t dim) {
    // Allocate temporary storage for LU decomposition
    ComplexFloat* lu = (ComplexFloat*)aligned_alloc(32, dim * dim * sizeof(ComplexFloat));
    if (!lu) {
        return QGT_ERROR_ALLOCATION_FAILED;
    }
    
    // Copy metric to LU buffer
    memcpy(lu, metric, dim * dim * sizeof(ComplexFloat));
    
    // Perform LU decomposition with partial pivoting
    int* pivots = (int*)malloc(dim * sizeof(int));
    if (!pivots) {
        free(lu);
        return QGT_ERROR_ALLOCATION_FAILED;
    }
    
    // LU decomposition
    for (size_t i = 0; i < dim; i++) {
        pivots[i] = i;
        
        // Find pivot
        float max_val = fabsf(lu[i * dim + i].real);
        size_t max_idx = i;
        for (size_t j = i + 1; j < dim; j++) {
            float val = fabsf(lu[j * dim + i].real);
            if (val > max_val) {
                max_val = val;
                max_idx = j;
            }
        }
        
        // Check for singularity
        if (max_val < 1e-6f) {
            free(pivots);
            free(lu);
            return QGT_ERROR_INVALID_ARGUMENT;
        }
        
        // Swap rows if needed
        if (max_idx != i) {
            for (size_t j = 0; j < dim; j++) {
                ComplexFloat temp = lu[i * dim + j];
                lu[i * dim + j] = lu[max_idx * dim + j];
                lu[max_idx * dim + j] = temp;
            }
            pivots[i] = max_idx;
        }
        
        // Compute L and U factors
        for (size_t j = i + 1; j < dim; j++) {
            lu[j * dim + i] = complex_float_divide(lu[j * dim + i], lu[i * dim + i]);
            for (size_t k = i + 1; k < dim; k++) {
                lu[j * dim + k] = complex_float_subtract(
                    lu[j * dim + k],
                    complex_float_multiply(lu[j * dim + i], lu[i * dim + k])
                );
            }
        }
    }
    
    // Initialize inverse to identity
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            inv_metric[i * dim + j] = (i == j) ? COMPLEX_FLOAT_ONE : COMPLEX_FLOAT_ZERO;
        }
    }
    
    // Forward substitution
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < i; j++) {
            ComplexFloat sum = COMPLEX_FLOAT_ZERO;
            for (size_t k = 0; k < j; k++) {
                sum = complex_float_add(sum,
                    complex_float_multiply(lu[i * dim + k], inv_metric[k * dim + j]));
            }
            inv_metric[i * dim + j] = complex_float_subtract(
                inv_metric[i * dim + j], sum);
        }
    }
    
    // Back substitution
    for (size_t i = dim - 1; i < dim; i--) {
        for (size_t j = 0; j < dim; j++) {
            ComplexFloat sum = COMPLEX_FLOAT_ZERO;
            for (size_t k = i + 1; k < dim; k++) {
                sum = complex_float_add(sum,
                    complex_float_multiply(lu[i * dim + k], inv_metric[k * dim + j]));
            }
            inv_metric[i * dim + j] = complex_float_divide(
                complex_float_subtract(inv_metric[i * dim + j], sum),
                lu[i * dim + i]);
        }
    }
    
    // Apply pivots
    for (size_t i = dim - 1; i < dim; i--) {
        if (pivots[i] != i) {
            for (size_t j = 0; j < dim; j++) {
                ComplexFloat temp = inv_metric[i * dim + j];
                inv_metric[i * dim + j] = inv_metric[pivots[i] * dim + j];
                inv_metric[pivots[i] * dim + j] = temp;
            }
        }
    }
    
    free(pivots);
    free(lu);
    return QGT_SUCCESS;
}

// Create geometric connection
qgt_error_t geometric_create_connection(quantum_geometric_connection_t** connection,
                                      geometric_connection_type_t type,
                                      size_t dimension) {
    QGT_CHECK_NULL(connection);
    QGT_CHECK_ARGUMENT(dimension > 0 && dimension <= QGT_MAX_DIMENSIONS);
    
    // Allocate aligned memory for better SIMD performance
    *connection = (quantum_geometric_connection_t*)aligned_alloc(32, sizeof(quantum_geometric_connection_t));
    if (!*connection) {
        return QGT_ERROR_ALLOCATION_FAILED;
    }
    
    // Allocate aligned connection coefficients (Christoffel symbols)
    size_t size = dimension * dimension * dimension * sizeof(ComplexFloat);
    (*connection)->coefficients = (ComplexFloat*)aligned_alloc(32, size);
    if (!(*connection)->coefficients) {
        free(*connection);
        return QGT_ERROR_ALLOCATION_FAILED;
    }
    
    (*connection)->type = type;
    (*connection)->dimension = dimension;
    (*connection)->is_compatible = true;
    
    // Initialize connection based on type
    switch (type) {
        case GEOMETRIC_CONNECTION_LEVI_CIVITA:
            // Initialize all coefficients to zero (flat connection)
            memset((*connection)->coefficients, 0, size);
            break;
            
        default:
            free((*connection)->coefficients);
            free(*connection);
            return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    return QGT_SUCCESS;
}

// Destroy geometric connection
void geometric_destroy_connection(quantum_geometric_connection_t* connection) {
    if (connection) {
        free(connection->coefficients);
        free(connection);
    }
}

// Clone geometric connection
qgt_error_t geometric_clone_connection(quantum_geometric_connection_t** dest,
                                     const quantum_geometric_connection_t* src) {
    QGT_CHECK_NULL(dest);
    QGT_CHECK_NULL(src);
    
    qgt_error_t err = geometric_create_connection(dest, src->type, src->dimension);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    size_t size = src->dimension * src->dimension * src->dimension * sizeof(ComplexFloat);
    memcpy((*dest)->coefficients, src->coefficients, size);
    (*dest)->is_compatible = src->is_compatible;
    
    return QGT_SUCCESS;
}

// Compute geometric connection
qgt_error_t geometric_compute_connection(quantum_geometric_connection_t* connection,
                                       const quantum_geometric_metric_t* metric) {
    QGT_CHECK_NULL(connection);
    QGT_CHECK_NULL(metric);
    
    if (connection->dimension != metric->dimension) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    size_t dim = metric->dimension;
    const size_t block_size = 32; // Cache-friendly block size
    const float epsilon = 1e-6f; // Step size for derivatives
    
    // Allocate temporary storage for derivatives and inverse metric
    ComplexFloat* d_metric = (ComplexFloat*)aligned_alloc(32, dim * dim * sizeof(ComplexFloat));
    ComplexFloat* inv_metric = (ComplexFloat*)aligned_alloc(32, dim * dim * sizeof(ComplexFloat));
    
    if (!d_metric || !inv_metric) {
        if (d_metric) free(d_metric);
        if (inv_metric) free(inv_metric);
        return QGT_ERROR_ALLOCATION_FAILED;
    }
    
    // Compute inverse metric
    qgt_error_t err = compute_inverse_metric(inv_metric, metric->components, dim);
    if (err != QGT_SUCCESS) {
        free(d_metric);
        free(inv_metric);
        return err;
    }
    
    // Compute connection coefficients (Christoffel symbols)
    switch (connection->type) {
        case GEOMETRIC_CONNECTION_LEVI_CIVITA:
            // Compute Levi-Civita connection from metric with blocking
            #pragma omp parallel for collapse(3) if(dim > QGT_PARALLEL_THRESHOLD)
            for (size_t i_block = 0; i_block < dim; i_block += block_size) {
                for (size_t j_block = 0; j_block < dim; j_block += block_size) {
                    for (size_t k_block = 0; k_block < dim; k_block += block_size) {
                        size_t i_end = (i_block + block_size < dim) ? i_block + block_size : dim;
                        size_t j_end = (j_block + block_size < dim) ? j_block + block_size : dim;
                        size_t k_end = (k_block + block_size < dim) ? k_block + block_size : dim;
                        
                        // Process block
                        for (size_t i = i_block; i < i_end; i++) {
                            // Compute metric derivatives in i direction
                            compute_metric_derivatives(d_metric, metric->components, dim, i, epsilon);
                            
                            for (size_t j = j_block; j < j_end; j++) {
                                for (size_t k = k_block; k < k_end; k++) {
                                    ComplexFloat sum = COMPLEX_FLOAT_ZERO;
                                    
                                    #ifdef __AVX512F__
                                    __m512 vsum_real = _mm512_setzero_ps();
                                    __m512 vsum_imag = _mm512_setzero_ps();
                                    
                                    for (size_t l = 0; l < dim; l += 8) {
                                        // Load metric components and derivatives
                                        __m512 vg_kl = _mm512_loadu_ps((float*)&inv_metric[k * dim + l]);
                                        __m512 vdg_i = _mm512_loadu_ps((float*)&d_metric[j * dim + l]);
                                        __m512 vdg_j = _mm512_loadu_ps((float*)&d_metric[i * dim + l]);
                                        __m512 vdg_l = _mm512_loadu_ps((float*)&metric->components[i * dim + j]);
                                        
                                        // Complex arithmetic for Christoffel symbols
                                        __m512 vreal = _mm512_fmadd_ps(
                                            _mm512_add_ps(vdg_i, vdg_j),
                                            vg_kl,
                                            _mm512_mul_ps(vdg_l, _mm512_set1_ps(-1.0f))
                                        );
                                        
                                        vsum_real = _mm512_add_ps(vsum_real, vreal);
                                    }
                                    
                                    // Reduce sum and multiply by 0.5
                                    sum.real = 0.5f * _mm512_reduce_add_ps(vsum_real);
                                    sum.imag = 0.5f * _mm512_reduce_add_ps(vsum_imag);
                                    
                                    #elif defined(__ARM_NEON)
                                    float32x4_t vsum_real = vdupq_n_f32(0.0f);
                                    float32x4_t vsum_imag = vdupq_n_f32(0.0f);
                                    
                                    for (size_t l = 0; l < dim; l += 4) {
                                        float32x4_t vg_kl = vld1q_f32((float*)&inv_metric[k * dim + l]);
                                        float32x4_t vdg_i = vld1q_f32((float*)&d_metric[j * dim + l]);
                                        float32x4_t vdg_j = vld1q_f32((float*)&d_metric[i * dim + l]);
                                        float32x4_t vdg_l = vdupq_n_f32(metric->components[i * dim + j].real);
                                        
                                        vsum_real = vmlaq_f32(vsum_real,
                                            vaddq_f32(vdg_i, vdg_j),
                                            vg_kl);
                                        vsum_real = vmlsq_f32(vsum_real, vdg_l, vg_kl);
                                    }
                                    
                                    float32x2_t vsum2_real = vadd_f32(vget_low_f32(vsum_real), vget_high_f32(vsum_real));
                                    sum.real = 0.5f * (vget_lane_f32(vsum2_real, 0) + vget_lane_f32(vsum2_real, 1));
                                    sum.imag = 0.0f;
                                    
                                    #else
                                    // Scalar fallback
                                    for (size_t l = 0; l < dim; l++) {
                                        ComplexFloat g_kl = inv_metric[k * dim + l];
                                        ComplexFloat dg_i = d_metric[j * dim + l];
                                        ComplexFloat dg_j = d_metric[i * dim + l];
                                        ComplexFloat dg_l = metric->components[i * dim + j];
                                        
                                        sum = complex_float_add(sum,
                                            complex_float_multiply(g_kl,
                                                complex_float_add(
                                                    complex_float_add(dg_i, dg_j),
                                                    complex_float_multiply(dg_l,
                                                        complex_float_create(-1.0f, 0.0f))
                                                )
                                            )
                                        );
                                    }
                                    sum = complex_float_multiply(complex_float_create(0.5f, 0.0f), sum);
                                    #endif
                                    
                                    connection->coefficients[(i * dim + j) * dim + k] = sum;
                                }
                            }
                        }
                    }
                }
            }
            break;
            
        default:
            free(d_metric);
            free(inv_metric);
            return QGT_ERROR_NOT_IMPLEMENTED;
    }
    
    free(d_metric);
    free(inv_metric);
    return QGT_SUCCESS;
}

// Transform geometric connection
qgt_error_t geometric_transform_connection(quantum_geometric_connection_t* result,
                                         const quantum_geometric_connection_t* connection,
                                         const quantum_geometric_tensor_t* transform) {
    QGT_CHECK_NULL(result);
    QGT_CHECK_NULL(connection);
    QGT_CHECK_NULL(transform);
    
    if (transform->rank != 2 || 
        transform->dimensions[0] != connection->dimension ||
        transform->dimensions[1] != connection->dimension ||
        result->dimension != connection->dimension) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    size_t dim = connection->dimension;
    const size_t block_size = 32; // Cache-friendly block size
    
    // Zero initialize result
    memset(result->coefficients, 0, dim * dim * dim * sizeof(ComplexFloat));
    
    // Transform connection coefficients with blocking
    #pragma omp parallel for collapse(3) if(dim > QGT_PARALLEL_THRESHOLD)
    for (size_t i_block = 0; i_block < dim; i_block += block_size) {
        for (size_t j_block = 0; j_block < dim; j_block += block_size) {
            for (size_t k_block = 0; k_block < dim; k_block += block_size) {
                size_t i_end = (i_block + block_size < dim) ? i_block + block_size : dim;
                size_t j_end = (j_block + block_size < dim) ? j_block + block_size : dim;
                size_t k_end = (k_block + block_size < dim) ? k_block + block_size : dim;
                
                // Process block
                for (size_t i = i_block; i < i_end; i++) {
                    for (size_t j = j_block; j < j_end; j++) {
                        for (size_t k = k_block; k < k_end; k++) {
                            ComplexFloat sum = COMPLEX_FLOAT_ZERO;
                            
                            #ifdef __AVX512F__
                            __m512 vsum_real = _mm512_setzero_ps();
                            __m512 vsum_imag = _mm512_setzero_ps();
                            
                            for (size_t l = 0; l < dim; l += 8) {
                                __m512 vt_kl = _mm512_loadu_ps((float*)&transform->components[k * dim + l]);
                                
                                for (size_t m = 0; m < dim; m++) {
                                    __m512 vt_mi = _mm512_set1_ps(transform->components[m * dim + i].real);
                                    __m512 vconn = _mm512_loadu_ps((float*)&connection->coefficients[(l * dim + m) * dim]);
                                    
                                    __m512 vreal = _mm512_fmadd_ps(vt_kl, vt_mi, vconn);
                                    vsum_real = _mm512_add_ps(vsum_real, vreal);
                                }
                            }
                            
                            // Reduce sum
                            sum.real = _mm512_reduce_add_ps(vsum_real);
                            sum.imag = _mm512_reduce_add_ps(vsum_imag);
                            
                            #elif defined(__ARM_NEON)
                            float32x4_t vsum_real = vdupq_n_f32(0.0f);
                            float32x4_t vsum_imag = vdupq_n_f32(0.0f);
                            
                            for (size_t l = 0; l < dim; l += 4) {
                                float32x4_t vt_kl = vld1q_f32((float*)&transform->components[k * dim + l]);
                                
                                for (size_t m = 0; m < dim; m++) {
                                    float32x4_t vt_mi = vdupq_n_f32(transform->components[m * dim + i].real);
                                    float32x4_t vconn = vld1q_f32((float*)&connection->coefficients[(l * dim + m) * dim]);
                                    
                                    vsum_real = vmlaq_f32(vsum_real, vt_kl, vt_mi);
                                    vsum_real = vmlaq_f32(vsum_real, vconn, vdupq_n_f32(1.0f));
                                }
                            }
                            
                            float32x2_t vsum2_real = vadd_f32(vget_low_f32(vsum_real), vget_high_f32(vsum_real));
                            sum.real = vget_lane_f32(vsum2_real, 0) + vget_lane_f32(vsum2_real, 1);
                            sum.imag = 0.0f;
                            
                            #else
                            // Scalar fallback
                            for (size_t l = 0; l < dim; l++) {
                                for (size_t m = 0; m < dim; m++) {
                                    for (size_t n = 0; n < dim; n++) {
                                        ComplexFloat term = complex_float_multiply(
                                            transform->components[k * dim + l],
                                            complex_float_multiply(
                                                connection->coefficients[(l * dim + m) * dim + n],
                                                complex_float_multiply(
                                                    transform->components[m * dim + i],
                                                    transform->components[n * dim + j]
                                                )
                                            )
                                        );
                                        sum = complex_float_add(sum, term);
                                    }
                                }
                            }
                            #endif
                            
                            result->coefficients[(i * dim + j) * dim + k] = sum;
                        }
                    }
                }
            }
        }
    }
    
    return QGT_SUCCESS;
}
