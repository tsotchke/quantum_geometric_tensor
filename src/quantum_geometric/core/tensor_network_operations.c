#include "quantum_geometric/core/tensor_network_operations.h"
#include "quantum_geometric/core/numerical_backend.h"
#include "quantum_geometric/core/error_handling.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Error handling
static tensor_network_error_t g_last_error = TENSOR_NETWORK_SUCCESS;

static void set_error(tensor_network_t* network, tensor_network_error_t error) {
    if (network) {
        network->last_error = error;
    }
    g_last_error = error;
}

tensor_network_error_t get_last_tensor_network_error(void) {
    return g_last_error;
}

const char* get_tensor_network_error_string(tensor_network_error_t error) {
    switch (error) {
        case TENSOR_NETWORK_SUCCESS:
            return "Success";
        case TENSOR_NETWORK_ERROR_INVALID_ARGUMENT:
            return "Invalid argument";
        case TENSOR_NETWORK_ERROR_MEMORY:
            return "Memory allocation failed";
        case TENSOR_NETWORK_ERROR_INVALID_STATE:
            return "Invalid network state";
        case TENSOR_NETWORK_ERROR_DIMENSION_MISMATCH:
            return "Dimension mismatch";
        case TENSOR_NETWORK_ERROR_NODE_NOT_FOUND:
            return "Node not found";
        case TENSOR_NETWORK_ERROR_CONNECTION_EXISTS:
            return "Connection already exists";
        case TENSOR_NETWORK_ERROR_NO_CONNECTION:
            return "No connection exists";
        case TENSOR_NETWORK_ERROR_OPTIMIZATION_FAILED:
            return "Optimization failed";
        case TENSOR_NETWORK_ERROR_COMPUTATION:
            return "Computation error";
        case TENSOR_NETWORK_ERROR_NOT_IMPLEMENTED:
            return "Operation not implemented";
        default:
            return "Unknown error";
    }
}

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

// Network creation and destruction
tensor_network_t* create_tensor_network(void) {
    tensor_network_t* network = malloc(sizeof(tensor_network_t));
    if (!network) {
        set_error(NULL, TENSOR_NETWORK_ERROR_MEMORY);
        return NULL;
    }
    
    network->nodes = malloc(16 * sizeof(tensor_node_t*));
    if (!network->nodes) {
        free(network);
        set_error(NULL, TENSOR_NETWORK_ERROR_MEMORY);
        return NULL;
    }
    
    network->num_nodes = 0;
    network->capacity = 16;
    network->next_id = 0;
    network->optimized = false;
    network->last_error = TENSOR_NETWORK_SUCCESS;
    memset(&network->metrics, 0, sizeof(tensor_network_metrics_t));
    
    return network;
}

void destroy_tensor_network(tensor_network_t* network) {
    if (!network) return;
    
    for (size_t i = 0; i < network->num_nodes; i++) {
        tensor_node_t* node = network->nodes[i];
        if (node) {
            free(node->data);
            free(node->dimensions);
            free(node->connected_nodes);
            free(node->connected_dims);
            free(node);
        }
    }
    
    free(network->nodes);
    free(network);
}

// Node operations
bool add_tensor_node(tensor_network_t* network,
                    const ComplexFloat* data,
                    const size_t* dimensions,
                    size_t num_dimensions,
                    size_t* node_id) {
    if (!network || !data || !dimensions || !node_id || num_dimensions == 0) {
        set_error(network, TENSOR_NETWORK_ERROR_INVALID_ARGUMENT);
        return false;
    }
    
    // Resize nodes array if needed
    if (network->num_nodes >= network->capacity) {
        size_t new_capacity = network->capacity * 2;
        tensor_node_t** new_nodes = realloc(network->nodes,
            new_capacity * sizeof(tensor_node_t*));
        if (!new_nodes) {
            set_error(network, TENSOR_NETWORK_ERROR_MEMORY);
            return false;
        }
        network->nodes = new_nodes;
        network->capacity = new_capacity;
    }
    
    // Create new node
    tensor_node_t* node = malloc(sizeof(tensor_node_t));
    if (!node) {
        set_error(network, TENSOR_NETWORK_ERROR_MEMORY);
        return false;
    }
    
    // Calculate total size
    size_t total_size = 1;
    for (size_t i = 0; i < num_dimensions; i++) {
        total_size *= dimensions[i];
    }
    
    // Allocate and copy data
    node->data = malloc(total_size * sizeof(ComplexFloat));
    if (!node->data) {
        free(node);
        set_error(network, TENSOR_NETWORK_ERROR_MEMORY);
        return false;
    }
    memcpy(node->data, data, total_size * sizeof(ComplexFloat));
    
    // Allocate and copy dimensions
    node->dimensions = malloc(num_dimensions * sizeof(size_t));
    if (!node->dimensions) {
        free(node->data);
        free(node);
        set_error(network, TENSOR_NETWORK_ERROR_MEMORY);
        return false;
    }
    memcpy(node->dimensions, dimensions, num_dimensions * sizeof(size_t));
    
    node->num_dimensions = num_dimensions;
    // Initialize connection arrays with some capacity
    node->connected_nodes = malloc(num_dimensions * sizeof(size_t));
    node->connected_dims = malloc(num_dimensions * sizeof(size_t));
    if (!node->connected_nodes || !node->connected_dims) {
        free(node->connected_nodes);
        free(node->connected_dims);
        free(node->dimensions);
        free(node->data);
        free(node);
        set_error(network, TENSOR_NETWORK_ERROR_MEMORY);
        return false;
    }
    node->num_connections = 0;
    node->id = network->next_id++;
    node->is_valid = true;
    
    // Add node to network
    network->nodes[network->num_nodes++] = node;
    *node_id = node->id;
    network->optimized = false;
    
    return true;
}

bool remove_tensor_node(tensor_network_t* network, size_t node_id) {
    if (!network) {
        set_error(network, TENSOR_NETWORK_ERROR_INVALID_ARGUMENT);
        return false;
    }
    
    // Find node
    tensor_node_t* node = NULL;
    size_t index = 0;
    for (size_t i = 0; i < network->num_nodes; i++) {
        if (network->nodes[i]->id == node_id) {
            node = network->nodes[i];
            index = i;
            break;
        }
    }
    
    if (!node) {
        set_error(network, TENSOR_NETWORK_ERROR_NODE_NOT_FOUND);
        return false;
    }
    
    // Remove connections
    for (size_t i = 0; i < node->num_connections; i++) {
        size_t connected_id = node->connected_nodes[i];
        for (size_t j = 0; j < network->num_nodes; j++) {
            tensor_node_t* other = network->nodes[j];
            if (other->id == connected_id) {
                // Remove connection from other node
                for (size_t k = 0; k < other->num_connections; k++) {
                    if (other->connected_nodes[k] == node_id) {
                        memmove(&other->connected_nodes[k],
                               &other->connected_nodes[k + 1],
                               (other->num_connections - k - 1) * sizeof(size_t));
                        memmove(&other->connected_dims[k],
                               &other->connected_dims[k + 1],
                               (other->num_connections - k - 1) * sizeof(size_t));
                        other->num_connections--;
                        break;
                    }
                }
                break;
            }
        }
    }
    
    // Free node resources
    free(node->data);
    free(node->dimensions);
    free(node->connected_nodes);
    free(node->connected_dims);
    free(node);
    
    // Remove from array
    memmove(&network->nodes[index],
            &network->nodes[index + 1],
            (network->num_nodes - index - 1) * sizeof(tensor_node_t*));
    network->num_nodes--;
    network->optimized = false;
    
    return true;
}

// Connection operations
bool connect_tensor_nodes(tensor_network_t* network,
                        size_t node1_id,
                        size_t dim1_idx,
                        size_t node2_id,
                        size_t dim2_idx) {
    if (!network) {
        set_error(network, TENSOR_NETWORK_ERROR_INVALID_ARGUMENT);
        return false;
    }
    
    // Find nodes
    tensor_node_t* node1 = NULL;
    tensor_node_t* node2 = NULL;
    for (size_t i = 0; i < network->num_nodes; i++) {
        if (network->nodes[i]->id == node1_id) {
            node1 = network->nodes[i];
        }
        if (network->nodes[i]->id == node2_id) {
            node2 = network->nodes[i];
        }
    }
    
    if (!node1 || !node2) {
        set_error(network, TENSOR_NETWORK_ERROR_NODE_NOT_FOUND);
        return false;
    }
    
    // Validate dimensions
    if (dim1_idx >= node1->num_dimensions ||
        dim2_idx >= node2->num_dimensions) {
        set_error(network, TENSOR_NETWORK_ERROR_INVALID_ARGUMENT);
        return false;
    }
    
    if (node1->dimensions[dim1_idx] != node2->dimensions[dim2_idx]) {
        set_error(network, TENSOR_NETWORK_ERROR_DIMENSION_MISMATCH);
        return false;
    }
    
    // Check if connection already exists
    for (size_t i = 0; i < node1->num_connections; i++) {
        if (node1->connected_nodes[i] == node2_id) {
            set_error(network, TENSOR_NETWORK_ERROR_CONNECTION_EXISTS);
            return false;
        }
    }
    
    // Add connection to node1
    size_t* new_nodes1 = realloc(node1->connected_nodes,
        (node1->num_connections + 1) * sizeof(size_t));
    size_t* new_dims1 = realloc(node1->connected_dims,
        (node1->num_connections + 1) * sizeof(size_t));
    if (!new_nodes1 || !new_dims1) {
        free(new_nodes1);
        free(new_dims1);
        set_error(network, TENSOR_NETWORK_ERROR_MEMORY);
        return false;
    }
    node1->connected_nodes = new_nodes1;
    node1->connected_dims = new_dims1;
    node1->connected_nodes[node1->num_connections] = node2_id;
    node1->connected_dims[node1->num_connections] = dim1_idx;
    node1->num_connections++;
    
    // Add connection to node2
    size_t* new_nodes2 = realloc(node2->connected_nodes,
        (node2->num_connections + 1) * sizeof(size_t));
    size_t* new_dims2 = realloc(node2->connected_dims,
        (node2->num_connections + 1) * sizeof(size_t));
    if (!new_nodes2 || !new_dims2) {
        free(new_nodes2);
        free(new_dims2);
        set_error(network, TENSOR_NETWORK_ERROR_MEMORY);
        return false;
    }
    node2->connected_nodes = new_nodes2;
    node2->connected_dims = new_dims2;
    node2->connected_nodes[node2->num_connections] = node1_id;
    node2->connected_dims[node2->num_connections] = dim2_idx;
    node2->num_connections++;
    
    network->optimized = false;
    return true;
}

// Network properties
bool get_network_size(const tensor_network_t* network,
                     size_t* num_nodes,
                     size_t* num_connections) {
    if (!network || !num_nodes || !num_connections) {
        set_error((tensor_network_t*)network, TENSOR_NETWORK_ERROR_INVALID_ARGUMENT);
        return false;
    }
    
    *num_nodes = network->num_nodes;
    
    size_t total_connections = 0;
    for (size_t i = 0; i < network->num_nodes; i++) {
        total_connections += network->nodes[i]->num_connections;
    }
    *num_connections = total_connections / 2;  // Each connection is counted twice
    
    return true;
}

// Performance monitoring
bool get_tensor_network_metrics(const tensor_network_t* network,
                              tensor_network_metrics_t* metrics) {
    if (!network || !metrics) {
        set_error((tensor_network_t*)network, TENSOR_NETWORK_ERROR_INVALID_ARGUMENT);
        return false;
    }
    
    *metrics = network->metrics;
    return true;
}

bool reset_tensor_network_metrics(tensor_network_t* network) {
    if (!network) {
        set_error(network, TENSOR_NETWORK_ERROR_INVALID_ARGUMENT);
        return false;
    }
    
    memset(&network->metrics, 0, sizeof(tensor_network_metrics_t));
    return true;
}

// Contraction operations
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
    if (!add_tensor_node(network, result_data, out_dims, num_out_dims, result_node_id)) {
        free(contracted_dims1);
        free(contracted_dims2);
        free(out_dims);
        free(result_data);
        return false;
    }
    
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

bool optimize_contraction_order(tensor_network_t* network,
                              contraction_optimization_t method) {
    if (!network) {
        set_error(network, TENSOR_NETWORK_ERROR_INVALID_ARGUMENT);
        return false;
    }
    
    // Already optimized
    if (network->optimized) {
        return true;
    }
    
    switch (method) {
        case CONTRACTION_OPTIMIZE_NONE:
            network->optimized = true;
            return true;
            
        case CONTRACTION_OPTIMIZE_GREEDY:
            // Current implementation is already greedy
            network->optimized = true;
            return true;
            
        case CONTRACTION_OPTIMIZE_DYNAMIC:
        case CONTRACTION_OPTIMIZE_EXHAUSTIVE:
            // TODO: Implement more sophisticated optimization strategies
            set_error(network, TENSOR_NETWORK_ERROR_NOT_IMPLEMENTED);
            return false;
            
        default:
            set_error(network, TENSOR_NETWORK_ERROR_INVALID_ARGUMENT);
            return false;
    }
}
