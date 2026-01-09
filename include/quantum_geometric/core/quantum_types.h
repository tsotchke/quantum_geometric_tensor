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
struct TopologicalAnyon;

// ============================================================================
// Topological Types for Error Correction
// ============================================================================

// 2D position on lattice
typedef struct Position {
    double x;
    double y;
} Position;

// Path through lattice for anyon movement
typedef struct Path {
    Position* vertices;      // Array of positions along path
    size_t length;          // Number of vertices in path
    double total_distance;  // Total path length
} Path;

// ============================================================================
// Pauli Operator Types (unified definition)
// ============================================================================

// Pauli operators - single source of truth for entire codebase
#ifndef PAULI_OPERATOR_DEFINED
#define PAULI_OPERATOR_DEFINED
typedef enum {
    PAULI_I = 0,    // Identity operator
    PAULI_X = 1,    // Pauli X (bit flip)
    PAULI_Y = 2,    // Pauli Y
    PAULI_Z = 3     // Pauli Z (phase flip)
} pauli_type_t;

// Aliases for backward compatibility
typedef pauli_type_t PauliOperator;
typedef pauli_type_t pauli_type;

// Rotation axis aliases (for quantum_circuit_operations.h compatibility)
#define ROTATION_AXIS_X PAULI_X
#define ROTATION_AXIS_Y PAULI_Y
#define ROTATION_AXIS_Z PAULI_Z
typedef pauli_type_t rotation_axis_t;
#endif

// ============================================================================
// Stabilizer Types for Topological Codes
// ============================================================================

// Stabilizer types - single source of truth
#ifndef STABILIZER_TYPE_DEFINED
#define STABILIZER_TYPE_DEFINED
typedef enum StabilizerType {
    // Names used in basic_topological operations
    PLAQUETTE_STABILIZER = 0,   // Z-type stabilizer (plaquette operator)
    VERTEX_STABILIZER = 1,      // X-type stabilizer (vertex operator)
    // Aliases matching stabilizer_types.h naming convention
    STABILIZER_PLAQUETTE = 0,
    STABILIZER_VERTEX = 1,
    // Aliases for Floquet code naming convention
    STABILIZER_Z = 0,           // Z stabilizer (same as plaquette)
    STABILIZER_X = 1,           // X stabilizer (same as vertex)
    // Extended stabilizer types for heavy-hex and advanced codes
    STABILIZER_WEIGHT_6 = 2,    // Weight-6 stabilizer (e.g., heavy-hex)
    STABILIZER_BOUNDARY = 3,    // Boundary stabilizer
    STABILIZER_LOGICAL = 4      // Logical stabilizer
} StabilizerType;
#endif

// Error codes for topological operations
typedef enum TopologicalErrorCode {
    TOPO_NO_ERROR = 0,
    TOPO_ERROR_DETECTED,
    TOPO_ERROR_INVALID_STATE,
    TOPO_ERROR_CORRECTION_FAILED,
    TOPO_ERROR_OUT_OF_MEMORY
} TopologicalErrorCode;

// Anyon structure for topological tracking within quantum state
typedef struct TopologicalAnyon {
    Position position;       // Current position on lattice
    int charge;             // Topological charge (+1 or -1)
    bool paired;            // Whether paired with another anyon
    size_t pair_index;      // Index of paired anyon (if paired)
    double creation_time;   // Time of creation
} TopologicalAnyon;

// Anyon pair for correction operations
typedef struct TopologicalAnyonPair {
    size_t anyon1;          // Index of first anyon
    size_t anyon2;          // Index of second anyon
    double distance;        // Distance between anyons
} TopologicalAnyonPair;

// Stabilizer measurement result
typedef struct StabilizerMeasurement {
    size_t index;           // Stabilizer index
    StabilizerType type;    // Type of stabilizer
    double value;           // Measurement outcome
    double confidence;      // Measurement confidence
} StabilizerMeasurement;

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
    // ========== Core Quantum State Properties ==========
    quantum_state_type_t type;      // Type of quantum state
    size_t num_qubits;              // Number of qubits (convenience field)
    size_t dimension;               // State dimension (2^num_qubits for pure states)
    size_t manifold_dim;            // Manifold dimension
    ComplexFloat* coordinates;      // State coordinates / amplitudes
    ComplexFloat* metric;           // Metric tensor
    ComplexFloat* connection;       // Connection coefficients
    void* auxiliary_data;           // Additional state data
    bool is_normalized;             // Normalization flag
    HardwareType hardware;          // Hardware location

    // ========== Topological Error Correction ==========
    // Lattice structure
    size_t lattice_width;           // Width of 2D lattice
    size_t lattice_height;          // Height of 2D lattice

    // Stabilizer measurements
    size_t num_stabilizers;         // Total number of stabilizers
    size_t num_plaquettes;          // Number of plaquette (Z) stabilizers
    size_t num_vertices;            // Number of vertex (X) stabilizers
    StabilizerMeasurement* stabilizers; // Stabilizer measurement results

    // Anyon tracking
    size_t num_anyons;              // Current number of detected anyons
    size_t max_anyons;              // Maximum capacity for anyon array
    TopologicalAnyon* anyons;       // Array of detected anyons

    // Error syndrome
    double* syndrome_values;        // Current syndrome measurement values
    size_t syndrome_size;           // Size of syndrome array

    // Measurement confidence tracking
    double* measurement_confidence; // Per-qubit measurement confidence values
    size_t confidence_size;         // Size of measurement confidence array
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
