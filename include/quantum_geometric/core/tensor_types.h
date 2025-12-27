#ifndef QUANTUM_GEOMETRIC_TENSOR_TYPES_H
#define QUANTUM_GEOMETRIC_TENSOR_TYPES_H

#include "quantum_geometric/core/quantum_complex.h"
#include <stddef.h>
#include <stdbool.h>
#include "quantum_geometric/hardware/quantum_hardware_types.h"

// Tensor types - basic tensor structure for standalone tensors
typedef struct tensor_t {
    size_t rank;                  // Number of dimensions (alias: num_dimensions)
    size_t* dimensions;           // Array of dimension sizes
    ComplexFloat* data;           // Raw tensor data
    bool is_contiguous;           // Whether data is stored contiguously
    size_t total_size;            // Total number of elements
    size_t* strides;              // Stride for each dimension
    bool owns_data;               // Whether this tensor owns its data buffer
    struct quantum_hardware_t* device;   // Device where tensor is stored
    void* auxiliary_data;         // Additional data for specific implementations
} tensor_t;

// Tensor node - a tensor with graph connectivity information for tensor networks
typedef struct tensor_node {
    // Tensor data (compatible with tensor_t for easy interop)
    ComplexFloat* data;           // Raw tensor data
    size_t* dimensions;           // Array of dimension sizes
    size_t rank;                  // Number of dimensions (primary name for compatibility)
    size_t num_dimensions;        // Alias for rank (set equal to rank)
    size_t total_size;            // Total number of elements (product of dimensions)

    // Graph connectivity
    size_t* connected_nodes;      // IDs of connected nodes
    size_t* connected_dims;       // Dimension indices for each connection
    size_t num_connections;       // Number of connections to other nodes

    // Node metadata
    size_t id;                    // Unique node identifier
    bool is_valid;                // Whether node contains valid data
} tensor_node_t;

// Performance metrics for tensor network operations
typedef struct tensor_network_metrics {
    size_t num_contractions;      // Number of contraction operations performed
    size_t peak_memory_usage;     // Peak memory usage in bytes
    double total_time;            // Total time for all operations
    double optimization_time;     // Time spent on contraction order optimization
    double contraction_time;      // Time spent on tensor contractions
} tensor_network_metrics_t;

// Error codes for tensor network operations
typedef enum {
    TENSOR_NETWORK_SUCCESS = 0,
    TENSOR_NETWORK_ERROR_INVALID_ARGUMENT = -1,
    TENSOR_NETWORK_ERROR_MEMORY = -2,
    TENSOR_NETWORK_ERROR_INVALID_STATE = -3,
    TENSOR_NETWORK_ERROR_DIMENSION_MISMATCH = -4,
    TENSOR_NETWORK_ERROR_NODE_NOT_FOUND = -5,
    TENSOR_NETWORK_ERROR_CONNECTION_EXISTS = -6,
    TENSOR_NETWORK_ERROR_NO_CONNECTION = -7,
    TENSOR_NETWORK_ERROR_OPTIMIZATION_FAILED = -8,
    TENSOR_NETWORK_ERROR_COMPUTATION = -9,
    TENSOR_NETWORK_ERROR_NOT_IMPLEMENTED = -10
} tensor_network_error_t;

// Unified tensor network structure
// Supports both simple array-based operations and graph-based connectivity tracking
typedef struct tensor_network_t {
    // Node storage - using tensor_node_t for full graph capability
    tensor_node_t** nodes;        // Array of pointers to tensor nodes
    size_t num_nodes;             // Current number of nodes
    size_t capacity;              // Allocated capacity for nodes array
    size_t next_id;               // Next available node ID

    // Connection tracking (for simple array-based operations)
    size_t* connections;          // Connection indices between nodes (legacy support)
    size_t num_connections;       // Number of connections (legacy support)

    // Optimization state
    bool is_optimized;            // Whether contraction order has been optimized
    bool optimized;               // Alias for is_optimized (backward compat)
    size_t* contraction_order;    // Optimal contraction order
    size_t max_memory;            // Maximum memory required during contraction

    // Performance tracking
    tensor_network_metrics_t metrics;    // Performance metrics
    tensor_network_error_t last_error;   // Last error code

    // Hardware and extensions
    struct quantum_hardware_t* device;   // Device where network is stored
    void* auxiliary_data;         // Additional data for specific implementations
} tensor_network_t;

#endif // QUANTUM_GEOMETRIC_TENSOR_TYPES_H
