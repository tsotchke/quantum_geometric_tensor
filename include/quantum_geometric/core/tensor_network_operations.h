#ifndef TENSOR_NETWORK_OPERATIONS_H
#define TENSOR_NETWORK_OPERATIONS_H

#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/tensor_types.h"
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// All core types (tensor_t, tensor_node_t, tensor_network_t, tensor_network_error_t,
// tensor_network_metrics_t) are defined in tensor_types.h

// Contraction optimization strategies
typedef enum {
    CONTRACTION_OPTIMIZE_NONE = 0,
    CONTRACTION_OPTIMIZE_GREEDY = 1,
    CONTRACTION_OPTIMIZE_DYNAMIC = 2,
    CONTRACTION_OPTIMIZE_EXHAUSTIVE = 3
} contraction_optimization_t;

// Network creation and destruction
tensor_network_t* create_tensor_network(void);
void destroy_tensor_network(tensor_network_t* network);

// Node operations
bool add_tensor_node(tensor_network_t* network,
                    const ComplexFloat* data,
                    const size_t* dimensions,
                    size_t num_dimensions,
                    size_t* node_id);

bool remove_tensor_node(tensor_network_t* network,
                       size_t node_id);

// Connection operations
bool connect_tensor_nodes(tensor_network_t* network,
                         size_t node1_id,
                         size_t dim1_idx,
                         size_t node2_id,
                         size_t dim2_idx);

bool disconnect_tensor_nodes(tensor_network_t* network,
                           size_t node1_id,
                           size_t node2_id);

// Contraction operations
bool contract_nodes(tensor_network_t* network,
                   size_t node1_id,
                   size_t node2_id,
                   size_t* result_node_id);

bool contract_subnetwork(tensor_network_t* network,
                        const size_t* node_ids,
                        size_t num_nodes,
                        size_t* result_node_id);

bool contract_full_network(tensor_network_t* network,
                         ComplexFloat** result,
                         size_t* result_dims,
                         size_t* num_dims);

// Optimization operations
bool optimize_contraction_order(tensor_network_t* network,
                              contraction_optimization_t method);

// Node properties
bool get_node_dimensions(const tensor_network_t* network,
                        size_t node_id,
                        size_t** dimensions,
                        size_t* num_dimensions);

bool get_node_data(const tensor_network_t* network,
                  size_t node_id,
                  ComplexFloat** data,
                  size_t* data_size);

bool get_node_connections(const tensor_network_t* network,
                         size_t node_id,
                         size_t** connected_nodes,
                         size_t* num_connections);

// Network properties
bool get_network_size(const tensor_network_t* network,
                     size_t* num_nodes,
                     size_t* num_connections);

bool get_network_complexity(const tensor_network_t* network,
                          size_t* space_complexity,
                          size_t* time_complexity);

// Error handling functions
void set_error(tensor_network_t* network, tensor_network_error_t error);
const char* get_tensor_network_error_string(tensor_network_error_t error);
tensor_network_error_t get_last_tensor_network_error(void);

// Performance monitoring functions
bool get_tensor_network_metrics(const tensor_network_t* network,
                              tensor_network_metrics_t* metrics);

bool reset_tensor_network_metrics(tensor_network_t* network);

#ifdef __cplusplus
}
#endif

#endif // TENSOR_NETWORK_OPERATIONS_H
