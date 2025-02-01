#ifndef QUANTUM_GEOMETRIC_METRIC_H
#define QUANTUM_GEOMETRIC_METRIC_H

#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_state.h"
#include "quantum_geometric/core/quantum_geometric_tensor.h"

// Create geometric metric
qgt_error_t geometric_create_metric(quantum_geometric_metric_t** metric,
                                  geometric_metric_type_t type,
                                  size_t dimension);

// Destroy geometric metric
void geometric_destroy_metric(quantum_geometric_metric_t* metric);

// Compute geometric metric
qgt_error_t geometric_compute_metric(quantum_geometric_metric_t* metric,
                                   const quantum_state_t* state);

// Clone geometric metric
qgt_error_t geometric_clone_metric(quantum_geometric_metric_t** dest,
                                 const quantum_geometric_metric_t* src);

// Transform geometric metric
qgt_error_t geometric_transform_metric(quantum_geometric_metric_t* result,
                                     const quantum_geometric_metric_t* metric,
                                     const quantum_geometric_tensor_t* transform);

#endif // QUANTUM_GEOMETRIC_METRIC_H
