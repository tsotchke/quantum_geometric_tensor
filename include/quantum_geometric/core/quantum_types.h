#ifndef QUANTUM_TYPES_H
#define QUANTUM_TYPES_H

#include <stddef.h>
#include <stdbool.h>
#include "quantum_geometric/core/quantum_complex.h"
#include "quantum_geometric/hardware/quantum_hardware_types.h"

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

struct quantum_circuit_t {
    size_t num_qubits;
    size_t num_gates;
    size_t max_gates;
    int optimization_level;
    quantum_gate_t* gates;
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
