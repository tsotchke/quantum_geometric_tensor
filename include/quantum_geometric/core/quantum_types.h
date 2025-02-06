#ifndef QUANTUM_TYPES_H
#define QUANTUM_TYPES_H

#include <stddef.h>
#include <stdbool.h>
#include "quantum_geometric/core/quantum_complex.h"
#include "quantum_geometric/hardware/quantum_hardware_types.h"

// Gate types
typedef enum {
    GATE_TYPE_I = 0,   // Identity
    GATE_TYPE_X = 1,   // Pauli-X
    GATE_TYPE_Y = 2,   // Pauli-Y
    GATE_TYPE_Z = 3,   // Pauli-Z
    GATE_TYPE_H = 4,   // Hadamard
    GATE_TYPE_S = 5,   // Phase
    GATE_TYPE_T = 6,   // T gate
    GATE_TYPE_RX = 7,  // Rotation around X
    GATE_TYPE_RY = 8,  // Rotation around Y
    GATE_TYPE_RZ = 9,  // Rotation around Z
    GATE_TYPE_CNOT = 10, // Controlled-NOT
    GATE_TYPE_CZ = 11,   // Controlled-Z
    GATE_TYPE_SWAP = 12, // SWAP
    GATE_TYPE_CUSTOM = 13 // Custom unitary
} gate_type_t;

// State types
typedef enum {
    QUANTUM_STATE_PURE,          // Pure quantum state
    QUANTUM_STATE_MIXED,         // Mixed quantum state
    QUANTUM_STATE_THERMAL,       // Thermal quantum state
    QUANTUM_STATE_COHERENT,      // Coherent quantum state
    QUANTUM_STATE_SQUEEZED,      // Squeezed quantum state
    QUANTUM_STATE_ENTANGLED,     // Entangled quantum state
    QUANTUM_STATE_CUSTOM         // Custom quantum state
} quantum_state_type_t;

// Forward declarations of all structs
struct quantum_state_t;
struct quantum_circuit_t;
struct quantum_register_t;
struct quantum_system_t;
struct quantum_gate_t;

// Type definitions
typedef struct quantum_state_t quantum_state_t;
typedef struct quantum_circuit_t quantum_circuit_t;
typedef struct quantum_register_t quantum_register_t;
typedef struct quantum_system_t quantum_system_t;
typedef struct quantum_gate_t quantum_gate_t;

// Gate structure
struct quantum_gate_t {
    size_t num_qubits;
    ComplexFloat* matrix;
    size_t* target_qubits;
    size_t* control_qubits;
    size_t num_controls;
    bool is_controlled;
    gate_type_t type;
    double* parameters;
    size_t num_parameters;
    bool is_parameterized;
};

// Struct definitions
struct quantum_state_t {
    quantum_state_type_t type;      // Type of quantum state
    size_t dimension;               // State dimension
    size_t manifold_dim;           // Manifold dimension
    ComplexFloat* coordinates;      // State coordinates
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

// Quantum circuit structure
struct quantum_circuit_t {
    circuit_layer_t** layers;
    size_t num_layers;
    size_t num_qubits;
    bool is_parameterized;
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
