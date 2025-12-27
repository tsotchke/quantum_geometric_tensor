#include "quantum_geometric/core/quantum_geometric_curvature.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_geometric_constants.h"
#include "quantum_geometric/core/quantum_geometric_tensor_network.h"
#include "quantum_geometric/core/quantum_geometric_metric.h"
#include <stdlib.h>
#include <string.h>

// Create geometric curvature
qgt_error_t geometric_create_curvature(quantum_geometric_curvature_t** curvature,
                                     geometric_curvature_type_t type,
                                     size_t dimension) {
    QGT_CHECK_NULL(curvature);
    QGT_CHECK_ARGUMENT(dimension > 0 && dimension <= QGT_MAX_DIMENSIONS);

    *curvature = (quantum_geometric_curvature_t*)calloc(1, sizeof(quantum_geometric_curvature_t));
    if (!*curvature) {
        return QGT_ERROR_ALLOCATION_FAILED;
    }

    // Allocate curvature tensor components
    // Berry curvature is a rank-2 antisymmetric tensor (dimension x dimension)
    // Riemann curvature is a rank-4 tensor (dimension^4 components)
    size_t size;
    if (type == GEOMETRIC_CURVATURE_BERRY) {
        size = dimension * dimension * sizeof(ComplexFloat);
    } else {
        size = dimension * dimension * dimension * dimension * sizeof(ComplexFloat);
    }

    (*curvature)->components = (ComplexFloat*)malloc(size);
    if (!(*curvature)->components) {
        free(*curvature);
        return QGT_ERROR_ALLOCATION_FAILED;
    }

    (*curvature)->type = type;
    (*curvature)->dimension = dimension;

    // Initialize to zero
    memset((*curvature)->components, 0, size);

    return QGT_SUCCESS;
}

// Destroy geometric curvature
void geometric_destroy_curvature(quantum_geometric_curvature_t* curvature) {
    if (curvature) {
        free(curvature->components);
        free(curvature);
    }
}

// Clone geometric curvature
qgt_error_t geometric_clone_curvature(quantum_geometric_curvature_t** dest,
                                    const quantum_geometric_curvature_t* src) {
    QGT_CHECK_NULL(dest);
    QGT_CHECK_NULL(src);

    qgt_error_t err = geometric_create_curvature(dest, src->type, src->dimension);
    if (err != QGT_SUCCESS) {
        return err;
    }

    // Calculate size based on curvature type
    size_t size;
    if (src->type == GEOMETRIC_CURVATURE_BERRY) {
        size = src->dimension * src->dimension * sizeof(ComplexFloat);
    } else {
        size = src->dimension * src->dimension * src->dimension * src->dimension * sizeof(ComplexFloat);
    }
    memcpy((*dest)->components, src->components, size);

    return QGT_SUCCESS;
}

// Compute geometric curvature
qgt_error_t geometric_compute_curvature(quantum_geometric_curvature_t* curvature,
                                      const quantum_geometric_connection_t* connection) {
    QGT_CHECK_NULL(curvature);
    QGT_CHECK_NULL(connection);
    
    if (curvature->dimension != connection->dimension) {
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    size_t dim = curvature->dimension;
    
    // Compute curvature tensor
    // R^i_jkl = ∂_k Γ^i_jl - ∂_l Γ^i_jk + Γ^m_jl Γ^i_mk - Γ^m_jk Γ^i_ml
    switch (curvature->type) {
        case GEOMETRIC_CURVATURE_RIEMANN:
            for (size_t i = 0; i < dim; i++) {
                for (size_t j = 0; j < dim; j++) {
                    for (size_t k = 0; k < dim; k++) {
                        for (size_t l = 0; l < dim; l++) {
                            ComplexFloat sum = COMPLEX_FLOAT_ZERO;
                            
                            // Simplified version - full implementation would compute
                            // actual derivatives and proper tensor contractions
                            for (size_t m = 0; m < dim; m++) {
                                ComplexFloat gamma1 = connection->coefficients[(j * dim + l) * dim + m];
                                ComplexFloat gamma2 = connection->coefficients[(m * dim + k) * dim + i];
                                ComplexFloat term = complex_float_multiply(gamma1, gamma2);
                                
                                sum = complex_float_add(sum, term);
                            }
                            
                            curvature->components[(((i * dim + j) * dim + k) * dim) + l] = sum;
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

// Transform geometric curvature
qgt_error_t geometric_transform_curvature(quantum_geometric_curvature_t* result,
                                        const quantum_geometric_curvature_t* curvature,
                                        const quantum_geometric_tensor_t* transform) {
    QGT_CHECK_NULL(result);
    QGT_CHECK_NULL(curvature);
    QGT_CHECK_NULL(transform);
    
    if (transform->rank != 2 || 
        transform->dimensions[0] != curvature->dimension ||
        transform->dimensions[1] != curvature->dimension ||
        result->dimension != curvature->dimension) {
            return QGT_ERROR_INVALID_PARAMETER;
    }
    
    size_t dim = curvature->dimension;
    
    // Transform curvature tensor
    // R'^a_bcd = T^a_i R^i_jkl (T^(-1))^j_b (T^(-1))^k_c (T^(-1))^l_d
    for (size_t a = 0; a < dim; a++) {
        for (size_t b = 0; b < dim; b++) {
            for (size_t c = 0; c < dim; c++) {
                for (size_t d = 0; d < dim; d++) {
                    ComplexFloat sum = COMPLEX_FLOAT_ZERO;
                    
                    for (size_t i = 0; i < dim; i++) {
                        for (size_t j = 0; j < dim; j++) {
                            for (size_t k = 0; k < dim; k++) {
                                for (size_t l = 0; l < dim; l++) {
                                    ComplexFloat term = complex_float_multiply(
                                        transform->components[a * dim + i],
                                        complex_float_multiply(
                                            curvature->components[(((i * dim + j) * dim + k) * dim) + l],
                                            complex_float_multiply(
                                                transform->components[j * dim + b],
                                                complex_float_multiply(
                                                    transform->components[k * dim + c],
                                                    transform->components[l * dim + d]
                                                )
                                            )
                                        )
                                    );
                                    sum = complex_float_add(sum, term);
                                }
                            }
                        }
                    }
                    
                    result->components[(((a * dim + b) * dim + c) * dim) + d] = sum;
                }
            }
        }
    }

    return QGT_SUCCESS;
}

// Compute a single element of the Berry curvature tensor
qgt_error_t geometric_compute_berry_curvature_element(
    const quantum_geometric_tensor_network_t* qgtn,
    size_t param_mu,
    size_t param_nu,
    float* result) {

    QGT_CHECK_NULL(qgtn);
    QGT_CHECK_NULL(result);

    // Use the QGT computation which now includes the projection term
    double curvature_value;
    if (!compute_berry_curvature(qgtn, param_mu, param_nu, &curvature_value)) {
        return QGT_ERROR_COMPUTATION_FAILED;
    }

    *result = (float)curvature_value;
    return QGT_SUCCESS;
}

// Compute the full Berry curvature tensor for a parameterized circuit
qgt_error_t geometric_compute_berry_curvature(
    quantum_geometric_curvature_t* curvature,
    const quantum_geometric_tensor_network_t* qgtn,
    size_t num_params) {

    QGT_CHECK_NULL(curvature);
    QGT_CHECK_NULL(qgtn);
    QGT_CHECK_ARGUMENT(num_params > 0);

    // Verify curvature is Berry type and has correct dimension
    if (curvature->type != GEOMETRIC_CURVATURE_BERRY) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    if (curvature->dimension != num_params) {
        return QGT_ERROR_DIMENSION_MISMATCH;
    }

    // Compute each element of the Berry curvature tensor
    // Berry curvature is antisymmetric: Ω_μν = -Ω_νμ
    // Diagonal elements are zero: Ω_μμ = 0
    for (size_t mu = 0; mu < num_params; mu++) {
        // Diagonal is zero
        curvature->components[mu * num_params + mu].real = 0.0f;
        curvature->components[mu * num_params + mu].imag = 0.0f;

        for (size_t nu = mu + 1; nu < num_params; nu++) {
            double curvature_value;

            // Use the corrected QGT computation to get Berry curvature
            if (!compute_berry_curvature(qgtn, mu, nu, &curvature_value)) {
                return QGT_ERROR_COMPUTATION_FAILED;
            }

            // Store Ω_μν (stored in real part as it's a real quantity)
            curvature->components[mu * num_params + nu].real = (float)curvature_value;
            curvature->components[mu * num_params + nu].imag = 0.0f;

            // Antisymmetry: Ω_νμ = -Ω_μν
            curvature->components[nu * num_params + mu].real = -(float)curvature_value;
            curvature->components[nu * num_params + mu].imag = 0.0f;
        }
    }

    curvature->is_flat = false;  // Berry curvature is generally non-zero
    return QGT_SUCCESS;
}

// Compose the full QGT from metric and Berry curvature
qgt_error_t geometric_compose_qgt(
    ComplexFloat* qgt,
    const quantum_geometric_metric_t* metric,
    const quantum_geometric_curvature_t* curvature,
    size_t dimension) {

    QGT_CHECK_NULL(qgt);
    QGT_CHECK_NULL(metric);
    QGT_CHECK_NULL(curvature);

    // Verify dimensions match
    if (metric->dimension != dimension || curvature->dimension != dimension) {
        return QGT_ERROR_DIMENSION_MISMATCH;
    }

    // Verify types
    if (metric->type != GEOMETRIC_METRIC_FUBINI_STUDY) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    if (curvature->type != GEOMETRIC_CURVATURE_BERRY) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // Compose QGT: Q_μν = g_μν + i*Ω_μν
    // The metric is symmetric (real part), Berry curvature is antisymmetric (imaginary part)
    for (size_t mu = 0; mu < dimension; mu++) {
        for (size_t nu = 0; nu < dimension; nu++) {
            size_t idx = mu * dimension + nu;
            // Real part from Fubini-Study metric
            qgt[idx].real = metric->components[idx].real;
            // Imaginary part from Berry curvature (stored in real component of curvature)
            qgt[idx].imag = curvature->components[idx].real;
        }
    }

    return QGT_SUCCESS;
}

// Compute the full QGT directly from a parameterized circuit
qgt_error_t geometric_compute_full_qgt(
    ComplexFloat* qgt,
    const quantum_geometric_tensor_network_t* qgtn,
    size_t num_params) {

    QGT_CHECK_NULL(qgt);
    QGT_CHECK_NULL(qgtn);
    QGT_CHECK_ARGUMENT(num_params > 0);

    // Compute each element of the QGT directly
    // Q_μν = g_μν + i*Ω_μν
    for (size_t mu = 0; mu < num_params; mu++) {
        for (size_t nu = mu; nu < num_params; nu++) {
            // Use the QGT computation from tensor_network which computes both parts
            ComplexFloat qgt_element;
            if (!compute_quantum_geometric_tensor(qgtn, mu, nu, &qgt_element)) {
                return QGT_ERROR_COMPUTATION_FAILED;
            }

            // Store Q_μν
            qgt[mu * num_params + nu] = qgt_element;

            // For off-diagonal elements, use Hermitian symmetry: Q_νμ = Q_μν*
            if (nu != mu) {
                qgt[nu * num_params + mu].real = qgt_element.real;
                qgt[nu * num_params + mu].imag = -qgt_element.imag;  // Complex conjugate
            }
        }
    }

    return QGT_SUCCESS;
}
