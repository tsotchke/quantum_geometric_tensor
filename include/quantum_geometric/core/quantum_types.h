#ifndef QUANTUM_TYPES_H
#define QUANTUM_TYPES_H

#include <stddef.h>
#include <stdbool.h>

// Include base types for gate_type_t, quantum_state_type_t, HardwareType, ComplexFloat
// This is the authoritative source for these fundamental types
#include "quantum_geometric/core/quantum_base_types.h"

// Include quantum_complex for additional complex number operations
#include "quantum_geometric/core/quantum_complex.h"

// gate_type_t and quantum_state_type_t are defined in quantum_base_types.h
// Do not redefine them here to avoid conflicts

// Forward declarations of all structs
struct quantum_state_t;
struct quantum_circuit_t;
struct quantum_register_t;
struct quantum_system_t;
struct quantum_gate_t;
struct computational_graph_t;
struct quantum_geometric_state_t;
struct geometric_processor_t;

// Node types for quantum operations
typedef enum {
    NODE_UNITARY,
    NODE_MEASUREMENT,
    NODE_TENSOR_PRODUCT,
    NODE_PARTIAL_TRACE,
    NODE_QUANTUM_FOURIER,
    NODE_QUANTUM_PHASE,
    NODE_ROTATION      // For parameterized rotation gates (RX, RY, RZ)
} quantum_node_type_t;

// Forward declaration for state buffer
struct quantum_geometric_state_t;

// Node structure for quantum operations
struct quantum_compute_node_t {
    quantum_node_type_t type;
    size_t num_qubits;
    size_t* qubit_indices;
    ComplexFloat* parameters;
    size_t num_parameters;
    void* additional_data;
    // Tree structure for composite operations (tensor product, partial trace)
    struct quantum_compute_node_t** children;
    size_t num_children;
    struct quantum_geometric_state_t* state_buffer;
};

// Forward declarations for geometric operations
typedef struct computational_graph_t computational_graph_t;
typedef struct quantum_compute_node_t quantum_compute_node_t;
typedef struct quantum_geometric_state_t quantum_geometric_state_t;
typedef struct geometric_processor_t geometric_processor_t;


// Type definitions
typedef struct quantum_state_t quantum_state_t;
typedef struct quantum_circuit_t quantum_circuit_t;
typedef struct quantum_register_t quantum_register_t;
typedef struct quantum_system_t quantum_system_t;
typedef struct quantum_gate_t quantum_gate_t;

// Gate structure - supports both high-level circuit building and low-level matrix operations
struct quantum_gate_t {
    // Core gate properties
    gate_type_t type;
    size_t num_qubits;
    bool is_controlled;
    bool is_parameterized;

    // Qubit targeting - dual interface for compatibility
    size_t* target_qubits;          // Target qubit indices
    size_t* control_qubits;         // Control qubit indices (for controlled gates)
    size_t num_controls;            // Number of control qubits
    size_t* qubits;                 // Unified qubit array (for circuit operations interface)

    // Gate parameters (angles for rotation gates, etc.)
    double* parameters;
    size_t num_parameters;

    // Matrix representation (optional - computed on demand)
    ComplexFloat* matrix;

    // Extension data for custom gates
    void* custom_data;
};

// Struct definitions
struct quantum_state_t {
    quantum_state_type_t type;      // Type of quantum state
    size_t num_qubits;              // Number of qubits (convenience field)
    size_t dimension;               // State dimension (2^num_qubits for pure states)
    size_t manifold_dim;           // Manifold dimension
    ComplexFloat* coordinates;      // State coordinates / amplitudes
    ComplexFloat* metric;          // Metric tensor
    ComplexFloat* connection;      // Connection coefficients
    void* auxiliary_data;          // Additional state data
    bool is_normalized;            // Normalization flag
    HardwareType hardware;   // Hardware location
};

// Circuit layer structure
typedef struct circuit_layer_t {
    quantum_gate_t** gates;
    size_t num_gates;
    bool is_parameterized;
} circuit_layer_t;

// Quantum circuit structure - supports both layer-based and flat gate array interfaces
struct quantum_circuit_t {
    // ========== Core circuit properties ==========
    size_t num_qubits;
    bool is_parameterized;

    // ========== Layer-based structure (for optimized execution) ==========
    // Gates are organized into layers where all gates in a layer can execute in parallel
    circuit_layer_t** layers;
    size_t num_layers;
    size_t layers_capacity;           // Allocated capacity for layers array

    // ========== Flat gate array (for circuit building) ==========
    // Sequential gate list for easy circuit construction
    quantum_gate_t** gates;
    size_t num_gates;
    size_t max_gates;                 // Allocated capacity for gates array

    // ========== Circuit optimization and compilation ==========
    int optimization_level;           // 0=none, 1=basic, 2=aggressive
    bool is_compiled;                 // Whether gates have been compiled to layers

    // ========== Quantum geometric operations ==========
    computational_graph_t* graph;
    quantum_geometric_state_t* state;
    quantum_compute_node_t** nodes;
    size_t num_nodes;
    size_t capacity;
};

struct quantum_register_t {
    size_t size;
    ComplexFloat* amplitudes;
    quantum_system_t* system;
};

struct quantum_system_t {
    size_t num_qubits;
    size_t num_classical_bits;
    int flags;
    int device_type;
    void* device_data;
    void* state;
    void* operations;
    void* hardware;
};

#endif // QUANTUM_TYPES_H
