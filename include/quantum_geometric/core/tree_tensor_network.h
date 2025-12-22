#ifndef TREE_TENSOR_NETWORK_H
#define TREE_TENSOR_NETWORK_H

#include "quantum_geometric/core/tensor_network_operations.h"
#include "quantum_geometric/core/advanced_memory_system.h"
#include "quantum_geometric/core/hierarchical_matrix.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct tensor_stream;
struct tree_tensor_node;
struct tree_tensor_network;
typedef struct tensor_stream tensor_stream_t;
typedef struct tree_tensor_node tree_tensor_node_t;
typedef struct tree_tensor_network tree_tensor_network_t;

// Type aliases for CamelCase naming convention compatibility
typedef struct tree_tensor_network TreeTensorNetwork;

// Tensor stream for processing large tensors in chunks
struct tensor_stream {
    size_t chunk_size;           // Size of each chunk
    size_t current_offset;       // Current position in the stream
    size_t total_size;           // Total size of the tensor
    ComplexFloat* buffer;        // Buffer for current chunk
    void* source;                // Source tensor (could be node or matrix)
    bool is_hierarchical;        // Whether source is a hierarchical matrix
    MemoryPool* memory_pool;     // Memory pool for buffer allocation
};

// Tree tensor node structure
struct tree_tensor_node {
    size_t id;                   // Unique node ID
    size_t rank;                 // Bond dimension / rank
    size_t num_children;         // Number of child nodes
    tree_tensor_node_t** children; // Child nodes
    tree_tensor_node_t* parent;  // Parent node
    ComplexFloat* data;          // Tensor data (for leaf nodes)
    HierarchicalMatrix* h_matrix; // Hierarchical matrix representation
    size_t* dimensions;          // Tensor dimensions
    size_t num_dimensions;       // Number of dimensions
    bool is_leaf;                // Whether this is a leaf node
    bool use_hierarchical;       // Whether to use hierarchical representation
};

// Tree tensor network structure
struct tree_tensor_network {
    tree_tensor_node_t* root;    // Root node of the tree
    size_t num_nodes;            // Total number of nodes
    size_t max_rank;             // Maximum bond dimension
    size_t num_qubits;           // Number of qubits (for quantum states)
    double tolerance;            // SVD truncation tolerance
    MemoryPool* memory_pool;     // Memory pool for tensor allocations
    advanced_memory_system_t* memory_system; // Advanced memory system
    tensor_network_metrics_t metrics; // Performance metrics
};

// Creation and destruction
tree_tensor_network_t* create_tree_tensor_network(
    size_t num_qubits,
    size_t max_rank,
    double tolerance
);

void destroy_tree_tensor_network(
    tree_tensor_network_t* ttn
);

// Node operations
tree_tensor_node_t* add_tree_tensor_node(
    tree_tensor_network_t* ttn,
    const ComplexFloat* data,
    const size_t* dimensions,
    size_t num_dimensions,
    bool use_hierarchical
);

bool connect_tree_tensor_nodes(
    tree_tensor_network_t* ttn,
    tree_tensor_node_t* parent,
    tree_tensor_node_t* child
);

// Tensor operations
bool contract_tree_tensor_nodes(
    tree_tensor_network_t* ttn,
    tree_tensor_node_t* node1,
    tree_tensor_node_t* node2,
    tree_tensor_node_t** result
);

bool contract_full_tree_network(
    tree_tensor_network_t* ttn,
    ComplexFloat** result,
    size_t* result_dims,
    size_t* num_dims
);

// Streaming operations
tensor_stream_t* create_tensor_stream(
    tree_tensor_network_t* ttn,
    tree_tensor_node_t* node,
    size_t chunk_size
);

void destroy_tensor_stream(
    tensor_stream_t* stream
);

bool stream_next_chunk(
    tensor_stream_t* stream
);

bool contract_tensor_streams(
    tensor_stream_t* stream1,
    tensor_stream_t* stream2,
    tensor_stream_t* result
);

// Conversion functions
bool convert_tensor_network_to_tree(
    tensor_network_t* network,
    tree_tensor_network_t** ttn
);

bool convert_tree_to_tensor_network(
    tree_tensor_network_t* ttn,
    tensor_network_t** network
);

// Optimization functions
bool optimize_tree_structure(
    tree_tensor_network_t* ttn
);

bool adapt_bond_dimensions(
    tree_tensor_network_t* ttn,
    double target_error
);

// Memory management
bool set_memory_pool(
    tree_tensor_network_t* ttn,
    MemoryPool* pool
);

bool set_memory_system(
    tree_tensor_network_t* ttn,
    advanced_memory_system_t* system
);

// Performance monitoring
bool get_tree_tensor_network_metrics(
    const tree_tensor_network_t* ttn,
    tensor_network_metrics_t* metrics
);

bool reset_tree_tensor_network_metrics(
    tree_tensor_network_t* ttn
);

#ifdef __cplusplus
}
#endif

#endif // TREE_TENSOR_NETWORK_H
