#ifndef TENSOR_NETWORKS_H
#define TENSOR_NETWORKS_H

#include <stdbool.h>
#include <stddef.h>
#include "quantum_geometric/core/tensor_types.h"

// Node in tensor network
typedef struct tensor_network_node_t {
    tensor_t tensor;
    size_t num_edges;
    struct tensor_network_node_t** connected_nodes;
    size_t* edge_indices;
} tensor_network_node_t;

// Tensor network structure
typedef struct tensor_network_t {
    tensor_network_node_t** nodes;
    size_t num_nodes;
    size_t capacity;
} tensor_network_t;

/**
 * @brief Initialize a tensor network
 * 
 * @param network Network to initialize
 * @param initial_capacity Initial capacity for nodes
 * @return true on success, false on failure
 */
bool qg_tensor_network_init(tensor_network_t* network, size_t initial_capacity);

/**
 * @brief Add a node to the tensor network
 * 
 * @param network Network to add to
 * @param tensor Tensor for the new node
 * @param edges Array of edge indices (can be NULL)
 * @return true on success, false on failure
 */
bool qg_tensor_network_add_node(tensor_network_t* network, 
                              const tensor_t* tensor,
                              const size_t* edges);

/**
 * @brief Connect two nodes in the network
 * 
 * @param network Network containing the nodes
 * @param node1_idx Index of first node
 * @param node2_idx Index of second node
 * @param edge1_idx Edge index on first node
 * @param edge2_idx Edge index on second node
 * @return true on success, false on failure
 */
bool qg_tensor_network_connect_nodes(tensor_network_t* network,
                                   size_t node1_idx,
                                   size_t node2_idx, 
                                   size_t edge1_idx,
                                   size_t edge2_idx);

/**
 * @brief Contract tensor network to reduce bond dimension
 * 
 * @param network Network to optimize
 * @param max_bond_dim Maximum bond dimension to allow
 * @param tolerance Error tolerance for truncation
 * @return true on success, false on failure
 */
bool qg_tensor_network_reduce_bond_dimension(tensor_network_t* network,
                                           size_t max_bond_dim,
                                           float tolerance);

/**
 * @brief Clean up tensor network resources
 * 
 * @param network Network to clean up
 */
void qg_tensor_network_cleanup(tensor_network_t* network);

#endif // TENSOR_NETWORKS_H
