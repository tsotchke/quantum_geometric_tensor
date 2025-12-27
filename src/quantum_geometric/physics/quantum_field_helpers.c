#include "quantum_geometric/physics/quantum_field_helpers.h"
#include "quantum_geometric/core/tensor_operations.h"
#include "quantum_geometric/core/quantum_geometric_constants.h"
#include <stdlib.h>
#include <math.h>
#include <complex.h>

// transform_generator() - Canonical implementation with bounds checking in quantum_field_calculations.c

// calculate_derivatives() - Canonical implementation in quantum_field_calculations.c
// (removed: duplicate of quantum_field_calculations.c version)

// calculate_covariant_derivatives() - Canonical implementation in quantum_field_calculations.c
// (removed: duplicate of quantum_field_calculations.c version)

// Forward declaration of canonical function from quantum_field_calculations.c
extern complex double* calculate_derivatives(
    const Tensor* field,
    size_t t,
    size_t x,
    size_t y,
    size_t z);

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

            // Calculate derivatives (uses canonical implementation from quantum_field_calculations.c)
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
