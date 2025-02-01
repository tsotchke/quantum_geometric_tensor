#ifndef QUANTUM_GEOMETRIC_TENSOR_NETWORK_H
#define QUANTUM_GEOMETRIC_TENSOR_NETWORK_H

#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_complex.h"
#include "quantum_geometric/core/tensor_types.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct tensor_t;
struct tensor_network_t;

// Node in tensor network
typedef struct tensor_network_node_t {
    struct tensor_t tensor;
    size_t* connections;
    size_t num_connections;
    struct tensor_network_node_t* next;
} tensor_network_node_t;

// Function declarations
bool qg_tensor_init(struct tensor_t* tensor, const size_t* dims, size_t rank);
void qg_tensor_cleanup(struct tensor_t* tensor);
bool qg_tensor_network_init(struct tensor_network_t* network, size_t initial_capacity);
void qg_tensor_network_cleanup(struct tensor_network_t* network);
bool qg_tensor_network_add_node(struct tensor_network_t* network, const struct tensor_t* tensor, const size_t* connections);
bool qg_tensor_network_connect_nodes(struct tensor_network_t* network, size_t node1, size_t node2, size_t edge1, size_t edge2);
bool qg_tensor_decompose_svd(const struct tensor_t* tensor, float tolerance, struct tensor_t* u, struct tensor_t* s, struct tensor_t* v);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_GEOMETRIC_TENSOR_NETWORK_H
