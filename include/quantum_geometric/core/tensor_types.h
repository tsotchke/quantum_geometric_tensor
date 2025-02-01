#ifndef QUANTUM_GEOMETRIC_TENSOR_TYPES_H
#define QUANTUM_GEOMETRIC_TENSOR_TYPES_H

#include "quantum_geometric/core/quantum_complex.h"
#include <stddef.h>
#include <stdbool.h>
#include "quantum_geometric/hardware/quantum_hardware_types.h"

// Tensor types
typedef struct tensor_t {
    size_t rank;                  // Number of dimensions
    size_t* dimensions;           // Array of dimension sizes
    ComplexFloat* data;           // Raw tensor data
    bool is_contiguous;           // Whether data is stored contiguously
    size_t total_size;           // Total number of elements
    size_t* strides;             // Stride for each dimension
    bool owns_data;              // Whether this tensor owns its data buffer
    struct quantum_hardware_t* device;   // Device where tensor is stored
    void* auxiliary_data;        // Additional data for specific implementations
} tensor_t;

typedef struct tensor_network_t {
    size_t num_nodes;              // Number of tensor nodes
    tensor_t* nodes;               // Array of tensor nodes
    size_t* connections;           // Connection indices between nodes
    size_t num_connections;        // Number of connections
    bool is_optimized;             // Whether network has been optimized
    size_t* contraction_order;     // Optimal contraction order
    size_t max_memory;             // Maximum memory required during contraction
    void* auxiliary_data;          // Additional data for specific implementations
    struct quantum_hardware_t* device;    // Device where network is stored
} tensor_network_t;

#endif // QUANTUM_GEOMETRIC_TENSOR_TYPES_H
