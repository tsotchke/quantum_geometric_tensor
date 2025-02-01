#ifndef QUANTUM_GEOMETRIC_CONNECTION_H
#define QUANTUM_GEOMETRIC_CONNECTION_H

#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_geometric_metric.h"
#include "quantum_geometric/core/quantum_geometric_tensor.h"

// Create geometric connection
qgt_error_t geometric_create_connection(quantum_geometric_connection_t** connection,
                                      geometric_connection_type_t type,
                                      size_t dimension);

// Destroy geometric connection
void geometric_destroy_connection(quantum_geometric_connection_t* connection);

// Clone geometric connection
qgt_error_t geometric_clone_connection(quantum_geometric_connection_t** dest,
                                     const quantum_geometric_connection_t* src);

// Compute geometric connection
qgt_error_t geometric_compute_connection(quantum_geometric_connection_t* connection,
                                       const quantum_geometric_metric_t* metric);

// Transform geometric connection
qgt_error_t geometric_transform_connection(quantum_geometric_connection_t* result,
                                         const quantum_geometric_connection_t* connection,
                                         const quantum_geometric_tensor_t* transform);

#endif // QUANTUM_GEOMETRIC_CONNECTION_H
