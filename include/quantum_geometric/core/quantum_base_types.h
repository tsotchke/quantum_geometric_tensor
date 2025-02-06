#ifndef QUANTUM_BASE_TYPES_H
#define QUANTUM_BASE_TYPES_H

#include <stddef.h>
#include <stdbool.h>
#include "quantum_geometric/core/quantum_complex.h"
#include "quantum_geometric/hardware/quantum_hardware_types.h"

// Forward declarations
typedef struct quantum_gate_t quantum_gate_t;
typedef struct quantum_circuit_t quantum_circuit_t;
typedef struct quantum_system_t quantum_system_t;

// Gate types
typedef enum {
    GATE_TYPE_X,
    GATE_TYPE_Y,
    GATE_TYPE_Z,
    GATE_TYPE_H,
    GATE_TYPE_CNOT,
    GATE_TYPE_CUSTOM
} gate_type_t;

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

// Quantum system structure
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

#endif // QUANTUM_BASE_TYPES_H
