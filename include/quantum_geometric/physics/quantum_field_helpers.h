/**
 * @file quantum_field_helpers.h
 * @brief Helper functions for quantum field calculations
 *
 * Provides auxiliary functions for:
 * - Gauge field transformations
 * - Derivative calculations
 * - Field strength computations
 */

#ifndef QUANTUM_FIELD_HELPERS_H
#define QUANTUM_FIELD_HELPERS_H

#include "quantum_geometric/physics/quantum_field_calculations.h"

#ifdef __cplusplus
extern "C" {
#endif

// Additional constants
#ifndef QG_TWO
#define QG_TWO 2.0
#endif

/**
 * @brief Transform generator component at spacetime point
 *
 * @param gauge_field Gauge field tensor
 * @param transformation Gauge transformation matrix
 * @param t Time coordinate
 * @param x X coordinate
 * @param y Y coordinate
 * @param z Z coordinate
 * @param generator Generator index
 */
void transform_generator_helper(
    Tensor* gauge_field,
    const Tensor* transformation,
    size_t t, size_t x, size_t y, size_t z,
    size_t generator);

/**
 * @brief Calculate derivatives of tensor field
 *
 * @param field Input tensor field
 * @param t Time coordinate
 * @param x X coordinate
 * @param y Y coordinate
 * @param z Z coordinate
 * @return Array of derivatives (caller must free)
 */
complex double* calculate_field_derivatives(
    const Tensor* field,
    size_t t, size_t x, size_t y, size_t z);

/**
 * @brief Calculate covariant derivatives with gauge coupling
 *
 * @param field Quantum field with gauge field
 * @param t Time coordinate
 * @param x X coordinate
 * @param y Y coordinate
 * @param z Z coordinate
 * @return Array of covariant derivatives (caller must free)
 */
complex double* calculate_field_covariant_derivatives(
    const QuantumField* field,
    size_t t, size_t x, size_t y, size_t z);

/**
 * @brief Calculate field strength tensor at point
 *
 * @param field Quantum field with gauge field
 * @param t Time coordinate
 * @param x X coordinate
 * @param y Y coordinate
 * @param z Z coordinate
 * @return Field strength tensor F_μν (caller must free)
 */
complex double* calculate_field_strength_tensor(
    const QuantumField* field,
    size_t t, size_t x, size_t y, size_t z);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_FIELD_HELPERS_H
