#include "quantum_geometric/core/tensor_network_operations.h"
#include "quantum_geometric/core/numerical_backend.h"
#include "quantum_geometric/core/error_handling.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Helper function to calculate contraction cost
static size_t calculate_contraction_cost(const tensor_node_t* node1,
                                       const tensor_node_t* node2) {
    size_t cost = 1;
    
    // Calculate output dimensions
    for (size_t i = 0; i < node1->num_dimensions; i++) {
        bool is_contracted = false;
        for (size_t j = 0; j < node1->num_connections; j++) {
            if (node1->connected_dims[j] == i &&
                node1->connected_nodes[j] == node2->id) {
                is_contracted = true;
                break;
            }
        }
        if (!is_contracted) {
            cost *= node1->dimensions[i];
        }
    }
    
    for (size_t i = 0; i < node2->num_dimensions; i++) {
        bool is_contracted = false;
        for (size_t j = 0; j < node2->num_connections; j++) {
            if (node2->connected_dims[j] == i &&
                node2->connected_nodes[j] == node1->id) {
                is_contracted = true;
                break;
            }
        }
        if (!is_contracted) {
            cost *= node2->dimensions[i];
        }
    }
    
    return cost;
}

// Helper function to find optimal contraction pair
static bool find_optimal_contraction_pair(const tensor_network_t* network,
                                        size_t* node1_idx,
                                        size_t* node2_idx) {
    size_t min_cost = SIZE_MAX;
    bool found = false;
    
    for (size_t i = 0; i < network->num_nodes; i++) {
        tensor_node_t* node1 = network->nodes[i];
        for (size_t j = 0; j < node1->num_connections; j++) {
            size_t connected_id = node1->connected_nodes[j];
            
            // Find connected node
            for (size_t k = 0; k < network->num_nodes; k++) {
                if (network->nodes[k]->id == connected_id) {
                    size_t cost = calculate_contraction_cost(node1, network->nodes[k]);
                    if (cost < min_cost) {
                        min_cost = cost;
                        *node1_idx = i;
                        *node2_idx = k;
                        found = true;
                    }
                    break;
                }
            }
        }
    }
    
    return found;
}

bool contract_nodes(tensor_network_t* network,
                   size_t node1_id,
                   size_t node2_id,
                   size_t* result_node_id) {
    if (!network || !result_node_id) {
        set_error(network, TENSOR_NETWORK_ERROR_INVALID_ARGUMENT);
        return false;
    }
    
    // Find nodes
    tensor_node_t* node1 = NULL;
    tensor_node_t* node2 = NULL;
    size_t node1_idx = 0, node2_idx = 0;
    
    for (size_t i = 0; i < network->num_nodes; i++) {
        if (network->nodes[i]->id == node1_id) {
            node1 = network->nodes[i];
            node1_idx = i;
        }
        if (network->nodes[i]->id == node2_id) {
            node2 = network->nodes[i];
            node2_idx = i;
        }
    }
    
    if (!node1 || !node2) {
        set_error(network, TENSOR_NETWORK_ERROR_NODE_NOT_FOUND);
        return false;
    }
    
    // Find contracted dimensions
    size_t num_contracted = 0;
    size_t* contracted_dims1 = malloc(node1->num_dimensions * sizeof(size_t));
    size_t* contracted_dims2 = malloc(node2->num_dimensions * sizeof(size_t));
    
    if (!contracted_dims1 || !contracted_dims2) {
        free(contracted_dims1);
        free(contracted_dims2);
        set_error(network, TENSOR_NETWORK_ERROR_MEMORY);
        return false;
    }
    
    for (size_t i = 0; i < node1->num_connections; i++) {
        if (node1->connected_nodes[i] == node2_id) {
            for (size_t j = 0; j < node2->num_connections; j++) {
                if (node2->connected_nodes[j] == node1_id) {
                    contracted_dims1[num_contracted] = node1->connected_dims[i];
                    contracted_dims2[num_contracted] = node2->connected_dims[j];
                    num_contracted++;
                }
            }
        }
    }
    
    if (num_contracted == 0) {
        free(contracted_dims1);
        free(contracted_dims2);
        set_error(network, TENSOR_NETWORK_ERROR_NO_CONNECTION);
        return false;
    }
    
    // Calculate output dimensions
    size_t num_out_dims = node1->num_dimensions + node2->num_dimensions - 2 * num_contracted;
    size_t* out_dims = malloc(num_out_dims * sizeof(size_t));
    if (!out_dims) {
        free(contracted_dims1);
        free(contracted_dims2);
        set_error(network, TENSOR_NETWORK_ERROR_MEMORY);
        return false;
    }
    
    size_t out_idx = 0;
    for (size_t i = 0; i < node1->num_dimensions; i++) {
        bool is_contracted = false;
        for (size_t j = 0; j < num_contracted; j++) {
            if (contracted_dims1[j] == i) {
                is_contracted = true;
                break;
            }
        }
        if (!is_contracted) {
            out_dims[out_idx++] = node1->dimensions[i];
        }
    }
    
    for (size_t i = 0; i < node2->num_dimensions; i++) {
        bool is_contracted = false;
        for (size_t j = 0; j < num_contracted; j++) {
            if (contracted_dims2[j] == i) {
                is_contracted = true;
                break;
            }
        }
        if (!is_contracted) {
            out_dims[out_idx++] = node2->dimensions[i];
        }
    }
    
    // Initialize numerical backend if needed
    numerical_config_t config = {
        .type = NUMERICAL_BACKEND_CPU,
        .max_threads = 8,
        .use_fma = true,
        .use_avx = true,
        .use_neon = true,
        .cache_size = 32 * 1024 * 1024
    };
    
    if (!initialize_numerical_backend(&config)) {
        free(contracted_dims1);
        free(contracted_dims2);
        free(out_dims);
        set_error(network, TENSOR_NETWORK_ERROR_INVALID_STATE);
        return false;
    }
    
    // Perform contraction using numerical backend
    size_t total_size = 1;
    for (size_t i = 0; i < num_out_dims; i++) {
        total_size *= out_dims[i];
    }
    
    ComplexFloat* result_data = malloc(total_size * sizeof(ComplexFloat));
    if (!result_data) {
        free(contracted_dims1);
        free(contracted_dims2);
        free(out_dims);
        set_error(network, TENSOR_NETWORK_ERROR_MEMORY);
        return false;
    }
    
    // Reshape tensors for matrix multiplication
    size_t m = 1, n = 1, k = 1;
    for (size_t i = 0; i < node1->num_dimensions; i++) {
        bool is_contracted = false;
        for (size_t j = 0; j < num_contracted; j++) {
            if (contracted_dims1[j] == i) {
                k *= node1->dimensions[i];
                is_contracted = true;
                break;
            }
        }
        if (!is_contracted) {
            m *= node1->dimensions[i];
        }
    }
    
    for (size_t i = 0; i < node2->num_dimensions; i++) {
        bool is_contracted = false;
        for (size_t j = 0; j < num_contracted; j++) {
            if (contracted_dims2[j] == i) {
                is_contracted = true;
                break;
            }
        }
        if (!is_contracted) {
            n *= node2->dimensions[i];
        }
    }
    
    if (!numerical_matrix_multiply(node1->data,
                                node2->data,
                                result_data,
                                m, k, n,
                                false, false)) {
        free(contracted_dims1);
        free(contracted_dims2);
        free(out_dims);
        free(result_data);
        set_error(network, TENSOR_NETWORK_ERROR_COMPUTATION);
        return false;
    }
    
    // Create result node
    size_t* node_id = malloc(sizeof(size_t));
    if (!add_tensor_node(network, result_data, out_dims, num_out_dims, node_id)) {
        free(contracted_dims1);
        free(contracted_dims2);
        free(out_dims);
        free(result_data);
        free(node_id);
        return false;
    }
    
    *result_node_id = *node_id;
    free(node_id);
    
    // Update connections
    tensor_node_t* result_node = network->nodes[network->num_nodes - 1];
    
    // Copy uncontracted connections from node1
    for (size_t i = 0; i < node1->num_connections; i++) {
        if (node1->connected_nodes[i] != node2_id) {
            size_t connected_id = node1->connected_nodes[i];
            size_t dim = node1->connected_dims[i];
            
            if (!connect_tensor_nodes(network, result_node->id, dim,
                                    connected_id, dim)) {
                remove_tensor_node(network, result_node->id);
                free(contracted_dims1);
                free(contracted_dims2);
                free(out_dims);
                free(result_data);
                return false;
            }
        }
    }
    
    // Copy uncontracted connections from node2
    for (size_t i = 0; i < node2->num_connections; i++) {
        if (node2->connected_nodes[i] != node1_id) {
            size_t connected_id = node2->connected_nodes[i];
            size_t dim = node2->connected_dims[i];
            
            if (!connect_tensor_nodes(network, result_node->id, dim + node1->num_dimensions - num_contracted,
                                    connected_id, dim)) {
                remove_tensor_node(network, result_node->id);
                free(contracted_dims1);
                free(contracted_dims2);
                free(out_dims);
                free(result_data);
                return false;
            }
        }
    }
    
    // Remove original nodes
    remove_tensor_node(network, node2_id);
    remove_tensor_node(network, node1_id);
    
    free(contracted_dims1);
    free(contracted_dims2);
    free(out_dims);
    free(result_data);
    
    network->metrics.num_contractions++;
    network->optimized = false;
    
    return true;
}

bool contract_full_network(tensor_network_t* network,
                         ComplexFloat** result,
                         size_t* result_dims,
                         size_t* num_dims) {
    if (!network || !result || !result_dims || !num_dims) {
        set_error(network, TENSOR_NETWORK_ERROR_INVALID_ARGUMENT);
        return false;
    }
    
    if (network->num_nodes == 0) {
        set_error(network, TENSOR_NETWORK_ERROR_INVALID_STATE);
        return false;
    }
    
    // Single node case
    if (network->num_nodes == 1) {
        tensor_node_t* node = network->nodes[0];
        size_t total_size = 1;
        for (size_t i = 0; i < node->num_dimensions; i++) {
            total_size *= node->dimensions[i];
        }
        
        *result = malloc(total_size * sizeof(ComplexFloat));
        if (!*result) {
            set_error(network, TENSOR_NETWORK_ERROR_MEMORY);
            return false;
        }
        
        memcpy(*result, node->data, total_size * sizeof(ComplexFloat));
        memcpy(result_dims, node->dimensions, node->num_dimensions * sizeof(size_t));
        *num_dims = node->num_dimensions;
        
        return true;
    }
    
    // Contract pairs until only one node remains
    while (network->num_nodes > 1) {
        size_t node1_idx, node2_idx;
        if (!find_optimal_contraction_pair(network, &node1_idx, &node2_idx)) {
            set_error(network, TENSOR_NETWORK_ERROR_INVALID_STATE);
            return false;
        }
        
        size_t result_id;
        if (!contract_nodes(network,
                          network->nodes[node1_idx]->id,
                          network->nodes[node2_idx]->id,
                          &result_id)) {
            return false;
        }
    }
    
    // Copy final result
    tensor_node_t* final_node = network->nodes[0];
    size_t total_size = 1;
    for (size_t i = 0; i < final_node->num_dimensions; i++) {
        total_size *= final_node->dimensions[i];
    }
    
    *result = malloc(total_size * sizeof(ComplexFloat));
    if (!*result) {
        set_error(network, TENSOR_NETWORK_ERROR_MEMORY);
        return false;
    }
    
    memcpy(*result, final_node->data, total_size * sizeof(ComplexFloat));
    memcpy(result_dims, final_node->dimensions, final_node->num_dimensions * sizeof(size_t));
    *num_dims = final_node->num_dimensions;
    
    return true;
}
