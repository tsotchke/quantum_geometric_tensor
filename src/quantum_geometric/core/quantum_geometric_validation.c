#include "quantum_geometric/core/quantum_geometric_validation.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_geometric_constants.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Validate geometric metric
qgt_error_t geometric_validate_metric(const quantum_geometric_metric_t* metric,
                                    geometric_validation_flags_t flags,
                                    validation_result_t* result) {
    QGT_CHECK_NULL(metric);
    QGT_CHECK_NULL(result);
    
    result->is_valid = true;
    result->error_code = QGT_SUCCESS;
    memset(result->error_message, 0, QGT_MAX_ERROR_MESSAGE_LENGTH);
    
    // Check symmetry if requested
    if (flags & GEOMETRIC_VALIDATION_CHECK_SYMMETRY) {
        for (size_t i = 0; i < metric->dimension; i++) {
            for (size_t j = 0; j < i; j++) {
                ComplexFloat diff = complex_float_subtract(
                    metric->components[i * metric->dimension + j],
                    metric->components[j * metric->dimension + i]
                );
                if (complex_float_abs(diff) > QGT_VALIDATION_TOLERANCE) {
                    result->is_valid = false;
                    result->error_code = QGT_ERROR_VALIDATION_FAILED;
                    snprintf(result->error_message, QGT_MAX_ERROR_MESSAGE_LENGTH,
                            "Metric not symmetric at indices (%zu,%zu)", i, j);
                    return QGT_SUCCESS;
                }
            }
        }
    }
    
    // Check positive definiteness if requested
    if (flags & GEOMETRIC_VALIDATION_CHECK_POSITIVE_DEFINITE) {
        // Simple check using diagonal elements
        for (size_t i = 0; i < metric->dimension; i++) {
            if (metric->components[i * metric->dimension + i].real <= 0.0f) {
                result->is_valid = false;
                result->error_code = QGT_ERROR_VALIDATION_FAILED;
                snprintf(result->error_message, QGT_MAX_ERROR_MESSAGE_LENGTH,
                        "Metric not positive definite at index %zu", i);
                return QGT_SUCCESS;
            }
        }
    }
    
    return QGT_SUCCESS;
}

// Validate geometric connection
qgt_error_t geometric_validate_connection(const quantum_geometric_connection_t* connection,
                                        geometric_validation_flags_t flags,
                                        validation_result_t* result) {
    QGT_CHECK_NULL(connection);
    QGT_CHECK_NULL(result);
    
    result->is_valid = true;
    result->error_code = QGT_SUCCESS;
    memset(result->error_message, 0, QGT_MAX_ERROR_MESSAGE_LENGTH);
    
    // Check torsion-free property if requested
    if (flags & GEOMETRIC_VALIDATION_CHECK_TORSION_FREE) {
        for (size_t i = 0; i < connection->dimension; i++) {
            for (size_t j = 0; j < connection->dimension; j++) {
                for (size_t k = 0; k < connection->dimension; k++) {
                    ComplexFloat diff = complex_float_subtract(
                        connection->coefficients[(i * connection->dimension + j) * connection->dimension + k],
                        connection->coefficients[(i * connection->dimension + k) * connection->dimension + j]
                    );
                    if (complex_float_abs(diff) > QGT_VALIDATION_TOLERANCE) {
                        result->is_valid = false;
                        result->error_code = QGT_ERROR_VALIDATION_FAILED;
                        snprintf(result->error_message, QGT_MAX_ERROR_MESSAGE_LENGTH,
                                "Connection not torsion-free at indices (%zu,%zu,%zu)", i, j, k);
                        return QGT_SUCCESS;
                    }
                }
            }
        }
    }
    
    return QGT_SUCCESS;
}

// Validate geometric curvature
qgt_error_t geometric_validate_curvature(const quantum_geometric_curvature_t* curvature,
                                       geometric_validation_flags_t flags,
                                       validation_result_t* result) {
    QGT_CHECK_NULL(curvature);
    QGT_CHECK_NULL(result);
    
    result->is_valid = true;
    result->error_code = QGT_SUCCESS;
    memset(result->error_message, 0, QGT_MAX_ERROR_MESSAGE_LENGTH);
    
    // Check first Bianchi identity if requested
    if (flags & GEOMETRIC_VALIDATION_CHECK_BIANCHI) {
        for (size_t i = 0; i < curvature->dimension; i++) {
            for (size_t j = 0; j < curvature->dimension; j++) {
                for (size_t k = 0; k < curvature->dimension; k++) {
                    for (size_t l = 0; l < curvature->dimension; l++) {
                        // R^i_jkl + R^i_klj + R^i_ljk = 0
                        ComplexFloat sum = complex_float_add(
                            curvature->components[(((i * curvature->dimension + j) * curvature->dimension + k) * curvature->dimension) + l],
                            complex_float_add(
                                curvature->components[(((i * curvature->dimension + k) * curvature->dimension + l) * curvature->dimension) + j],
                                curvature->components[(((i * curvature->dimension + l) * curvature->dimension + j) * curvature->dimension) + k]
                            )
                        );
                        if (complex_float_abs(sum) > QGT_VALIDATION_TOLERANCE) {
                            result->is_valid = false;
                            result->error_code = QGT_ERROR_VALIDATION_FAILED;
                            snprintf(result->error_message, QGT_MAX_ERROR_MESSAGE_LENGTH,
                                    "Curvature violates first Bianchi identity at indices (%zu,%zu,%zu,%zu)", i, j, k, l);
                            return QGT_SUCCESS;
                        }
                    }
                }
            }
        }
    }
    
    return QGT_SUCCESS;
}

// Validate geometric optimization
qgt_error_t geometric_validate_optimization(const quantum_geometric_optimization_t* optimization,
                                          geometric_validation_flags_t flags,
                                          validation_result_t* result) {
    QGT_CHECK_NULL(optimization);
    QGT_CHECK_NULL(result);
    
    result->is_valid = true;
    result->error_code = QGT_SUCCESS;
    memset(result->error_message, 0, QGT_MAX_ERROR_MESSAGE_LENGTH);
    
    // Check parameter bounds if requested
    if (flags & GEOMETRIC_VALIDATION_CHECK_BOUNDS) {
        for (size_t i = 0; i < optimization->dimension; i++) {
            float magnitude = complex_float_abs(optimization->parameters[i]);
            if (magnitude > QGT_MAX_PARAMETER_MAGNITUDE) {
                result->is_valid = false;
                result->error_code = QGT_ERROR_VALIDATION_FAILED;
                snprintf(result->error_message, QGT_MAX_ERROR_MESSAGE_LENGTH,
                        "Optimization parameter %zu exceeds maximum magnitude", i);
                return QGT_SUCCESS;
            }
        }
    }
    
    // Check convergence properties if requested
    if (flags & GEOMETRIC_VALIDATION_CHECK_CONVERGENCE) {
        if (optimization->iterations > QGT_MAX_ITERATIONS) {
            result->is_valid = false;
            result->error_code = QGT_ERROR_VALIDATION_FAILED;
            snprintf(result->error_message, QGT_MAX_ERROR_MESSAGE_LENGTH,
                    "Optimization exceeded maximum iterations");
            return QGT_SUCCESS;
        }
    }
    
    return QGT_SUCCESS;
}
