#ifndef TENSOR_NETWORK_OPERATIONS_H
#define TENSOR_NETWORK_OPERATIONS_H

#include "quantum_geometric/core/tensor_types.h"
#include <stdbool.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#include <vecLib/vecLibTypes.h>
#endif

// Initialize tensor network
bool qg_tensor_network_init(tensor_network_t* network, size_t initial_capacity);

// Clean up tensor network resources
void qg_tensor_network_cleanup(tensor_network_t* network);

// Add node to tensor network
bool qg_tensor_network_add_node(tensor_network_t* network, tensor_t* tensor, void* auxiliary_data);

// Connect nodes in tensor network
bool qg_tensor_network_connect_nodes(tensor_network_t* network, 
                                   size_t node1_idx, size_t node2_idx,
                                   size_t edge1_idx, size_t edge2_idx);

// Initialize tensor
bool qg_tensor_init(tensor_t* tensor, size_t* dimensions, size_t rank);

// Clean up tensor resources
void qg_tensor_cleanup(tensor_t* tensor);

// Decompose tensor using SVD
bool qg_tensor_decompose_svd(tensor_t* tensor, float tolerance,
                            tensor_t* u, tensor_t* s, tensor_t* v);

#endif // TENSOR_NETWORK_OPERATIONS_H
