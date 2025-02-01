#include "quantum_geometric/physics/quantum_field_helpers.h"
#include "quantum_geometric/core/tensor_operations.h"
#include "quantum_geometric/core/quantum_geometric_constants.h"
#include <stdlib.h>
#include <math.h>
#include <complex.h>

// Transform generator at a spacetime point
void transform_generator(
    Tensor* gauge_field,
    const Tensor* transformation,
    size_t t,
    size_t x,
    size_t y,
    size_t z,
    size_t generator) {
    
    size_t n = gauge_field->dims[5];
    
    // Get generator components
    complex double* components = malloc(n * sizeof(complex double));
    
    for (size_t i = 0; i < n; i++) {
        size_t idx = (((t * gauge_field->dims[1] + x) *
                    gauge_field->dims[2] + y) *
                    gauge_field->dims[3] + z) *
                    gauge_field->dims[4] + generator;
        components[i] = gauge_field->data[idx * n + i];
    }
    
    // Apply adjoint transformation
    complex double* transformed = malloc(n * sizeof(complex double));
    
    for (size_t i = 0; i < n; i++) {
        transformed[i] = 0;
        for (size_t j = 0; j < n; j++) {
            for (size_t k = 0; k < n; k++) {
                size_t trans_idx = i * n + j;
                transformed[i] += conj(transformation->data[trans_idx]) *
                                components[k] *
                                transformation->data[k * n + j];
            }
        }
    }
    
    // Update gauge field
    for (size_t i = 0; i < n; i++) {
        size_t idx = (((t * gauge_field->dims[1] + x) *
                    gauge_field->dims[2] + y) *
                    gauge_field->dims[3] + z) *
                    gauge_field->dims[4] + generator;
        gauge_field->data[idx * n + i] = transformed[i];
    }
    
    free(components);
    free(transformed);
}

// Calculate field derivatives
complex double* calculate_derivatives(
    const Tensor* field,
    size_t t,
    size_t x,
    size_t y,
    size_t z) {
    
    size_t n = field->dims[4];
    complex double* derivatives = malloc(
        QG_SPACETIME_DIMS * n * sizeof(complex double)
    );
    
    // Calculate time derivative
    for (size_t i = 0; i < n; i++) {
        size_t idx = ((t * field->dims[1] + x) *
                   field->dims[2] + y) *
                   field->dims[3] + z;
        
        if (t > 0 && t < field->dims[0] - 1) {
            // Central difference
            derivatives[i] = (field->data[(idx + field->dims[1]*field->dims[2]*field->dims[3]) * n + i] -
                            field->data[(idx - field->dims[1]*field->dims[2]*field->dims[3]) * n + i]) / QG_TWO;
        } else if (t == 0) {
            // Forward difference
            derivatives[i] = field->data[(idx + field->dims[1]*field->dims[2]*field->dims[3]) * n + i] -
                           field->data[idx * n + i];
        } else {
            // Backward difference
            derivatives[i] = field->data[idx * n + i] -
                           field->data[(idx - field->dims[1]*field->dims[2]*field->dims[3]) * n + i];
        }
    }
    
    // Calculate spatial derivatives
    for (size_t mu = 1; mu < QG_SPACETIME_DIMS; mu++) {
        size_t stride = 1;
        for (size_t i = mu + 1; i < QG_SPACETIME_DIMS; i++) {
            stride *= field->dims[i];
        }
        
        for (size_t i = 0; i < n; i++) {
            size_t idx = ((t * field->dims[1] + x) *
                       field->dims[2] + y) *
                       field->dims[3] + z;
            
            if (idx % field->dims[mu] > 0 &&
                idx % field->dims[mu] < field->dims[mu] - 1) {
                // Central difference
                derivatives[mu * n + i] = (field->data[(idx + stride) * n + i] -
                                         field->data[(idx - stride) * n + i]) / QG_TWO;
            } else if (idx % field->dims[mu] == 0) {
                // Forward difference
                derivatives[mu * n + i] = field->data[(idx + stride) * n + i] -
                                        field->data[idx * n + i];
            } else {
                // Backward difference
                derivatives[mu * n + i] = field->data[idx * n + i] -
                                        field->data[(idx - stride) * n + i];
            }
        }
    }
    
    return derivatives;
}

// Calculate covariant derivatives
complex double* calculate_covariant_derivatives(
    const QuantumField* field,
    size_t t,
    size_t x,
    size_t y,
    size_t z) {
    
    size_t n = field->field_tensor->dims[4];
    complex double* cov_derivatives = malloc(
        QG_SPACETIME_DIMS * n * sizeof(complex double)
    );
    
    // Calculate ordinary derivatives
    complex double* derivatives = calculate_derivatives(
        field->field_tensor,
        t, x, y, z
    );
    
    // Add gauge connection
    for (size_t mu = 0; mu < QG_SPACETIME_DIMS; mu++) {
        for (size_t i = 0; i < n; i++) {
            cov_derivatives[mu * n + i] = derivatives[mu * n + i];
            
            // Add gauge field contribution
            for (size_t j = 0; j < n; j++) {
                size_t idx = (((t * field->gauge_field->dims[1] + x) *
                           field->gauge_field->dims[2] + y) *
                           field->gauge_field->dims[3] + z) *
                           field->gauge_field->dims[4] + mu;
                
                cov_derivatives[mu * n + i] += I *
                    field->gauge_field->data[idx * n + j] *
                    field->field_tensor->data[idx * n + j];
            }
        }
    }
    
    free(derivatives);
    return cov_derivatives;
}

// Calculate field strength tensor
complex double* calculate_field_strength(
    const QuantumField* field,
    size_t t,
    size_t x,
    size_t y,
    size_t z) {
    
    size_t n = field->gauge_field->dims[5];
    complex double* F_munu = malloc(
        QG_SPACETIME_DIMS * QG_SPACETIME_DIMS * sizeof(complex double)
    );
    
    // Calculate for each spacetime component pair
    for (size_t mu = 0; mu < QG_SPACETIME_DIMS; mu++) {
        for (size_t nu = 0; nu < QG_SPACETIME_DIMS; nu++) {
            if (mu == nu) {
                F_munu[mu * QG_SPACETIME_DIMS + nu] = 0;
                continue;
            }
            
            // Get gauge field components
            complex double* A_mu = malloc(n * sizeof(complex double));
            complex double* A_nu = malloc(n * sizeof(complex double));
            
            size_t idx = (((t * field->gauge_field->dims[1] + x) *
                       field->gauge_field->dims[2] + y) *
                       field->gauge_field->dims[3] + z);
            
            for (size_t i = 0; i < n; i++) {
                A_mu[i] = field->gauge_field->data[(idx * field->gauge_field->dims[4] + mu) * n + i];
                A_nu[i] = field->gauge_field->data[(idx * field->gauge_field->dims[4] + nu) * n + i];
            }
            
            // Calculate derivatives
            complex double* dA_mu = calculate_derivatives(
                field->gauge_field,
                t, x, y, z
            );
            
            complex double* dA_nu = calculate_derivatives(
                field->gauge_field,
                t, x, y, z
            );
            
            // Calculate commutator
            complex double commutator = 0;
            for (size_t i = 0; i < n; i++) {
                for (size_t j = 0; j < n; j++) {
                    commutator += A_mu[i] * A_nu[j] - A_nu[i] * A_mu[j];
                }
            }
            
            // Combine terms
            F_munu[mu * QG_SPACETIME_DIMS + nu] =
                dA_mu[nu] - dA_nu[mu] + I * field->field_strength * commutator;
            
            free(A_mu);
            free(A_nu);
            free(dA_mu);
            free(dA_nu);
        }
    }
    
    return F_munu;
}
