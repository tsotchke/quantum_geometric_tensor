#ifndef QUANTUM_GEOMETRIC_OPTIMIZATION_H
#define QUANTUM_GEOMETRIC_OPTIMIZATION_H

#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_geometric_metric.h"
#include "quantum_geometric/core/quantum_geometric_connection.h"
#include "quantum_geometric/core/quantum_geometric_curvature.h"

// Create geometric optimization
qgt_error_t geometric_create_optimization(quantum_geometric_optimization_t** optimization,
                                        geometric_optimization_type_t type,
                                        size_t dimension);

// Destroy geometric optimization
void geometric_destroy_optimization(quantum_geometric_optimization_t* optimization);

// Clone geometric optimization
qgt_error_t geometric_clone_optimization(quantum_geometric_optimization_t** dest,
                                       const quantum_geometric_optimization_t* src);

// Optimize geometric parameters
qgt_error_t geometric_optimize_parameters(quantum_geometric_optimization_t* optimization,
                                        const quantum_geometric_metric_t* metric,
                                        const quantum_geometric_connection_t* connection,
                                        const quantum_geometric_curvature_t* curvature);

// Check optimization convergence
qgt_error_t geometric_check_convergence(const quantum_geometric_optimization_t* optimization,
                                      bool* converged);

#endif // QUANTUM_GEOMETRIC_OPTIMIZATION_H
