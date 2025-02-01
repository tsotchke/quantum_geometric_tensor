#ifndef QUANTUM_GEOMETRIC_VALIDATION_H
#define QUANTUM_GEOMETRIC_VALIDATION_H

#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_geometric_metric.h"
#include "quantum_geometric/core/quantum_geometric_connection.h"
#include "quantum_geometric/core/quantum_geometric_curvature.h"
#include "quantum_geometric/core/quantum_geometric_optimization.h"

// Validate geometric metric
qgt_error_t geometric_validate_metric(const quantum_geometric_metric_t* metric,
                                    geometric_validation_flags_t flags,
                                    validation_result_t* result);

// Validate geometric connection
qgt_error_t geometric_validate_connection(const quantum_geometric_connection_t* connection,
                                        geometric_validation_flags_t flags,
                                        validation_result_t* result);

// Validate geometric curvature
qgt_error_t geometric_validate_curvature(const quantum_geometric_curvature_t* curvature,
                                       geometric_validation_flags_t flags,
                                       validation_result_t* result);

// Validate geometric optimization
qgt_error_t geometric_validate_optimization(const quantum_geometric_optimization_t* optimization,
                                          geometric_validation_flags_t flags,
                                          validation_result_t* result);

#endif // QUANTUM_GEOMETRIC_VALIDATION_H
