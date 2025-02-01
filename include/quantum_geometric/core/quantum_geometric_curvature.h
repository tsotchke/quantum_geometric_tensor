#ifndef QUANTUM_GEOMETRIC_CURVATURE_H
#define QUANTUM_GEOMETRIC_CURVATURE_H

#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_geometric_connection.h"
#include "quantum_geometric/core/quantum_geometric_tensor.h"

// Create geometric curvature
qgt_error_t geometric_create_curvature(quantum_geometric_curvature_t** curvature,
                                     geometric_curvature_type_t type,
                                     size_t dimension);

// Destroy geometric curvature
void geometric_destroy_curvature(quantum_geometric_curvature_t* curvature);

// Clone geometric curvature
qgt_error_t geometric_clone_curvature(quantum_geometric_curvature_t** dest,
                                    const quantum_geometric_curvature_t* src);

// Compute geometric curvature
qgt_error_t geometric_compute_curvature(quantum_geometric_curvature_t* curvature,
                                      const quantum_geometric_connection_t* connection);

// Transform geometric curvature
qgt_error_t geometric_transform_curvature(quantum_geometric_curvature_t* result,
                                        const quantum_geometric_curvature_t* curvature,
                                        const quantum_geometric_tensor_t* transform);

#endif // QUANTUM_GEOMETRIC_CURVATURE_H
