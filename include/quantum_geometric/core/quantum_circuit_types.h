#ifndef QUANTUM_CIRCUIT_TYPES_H
#define QUANTUM_CIRCUIT_TYPES_H

#include "quantum_geometric/core/quantum_types.h"
#include <stddef.h>
#include <stdbool.h>

// Quantum circuit layer types
typedef enum {
    LAYER_QUANTUM_CONV,
    LAYER_QUANTUM_POOL,
    LAYER_QUANTUM_DENSE,
    LAYER_QUANTUM_GATE,
    LAYER_QUANTUM_MEASURE
} QuantumLayerType;

// Quantum layer configuration
typedef struct {
    QuantumLayerType* types;
    void** params;
    size_t num_layers;
} QuantumLayerConfig;

// Circuit structure
typedef struct quantum_circuit {
    quantum_gate_t** gates;   // Array of gates
    size_t num_gates;        // Number of gates
    size_t capacity;         // Allocated capacity
    size_t num_qubits;       // Total number of qubits
    bool* measured;          // Array tracking measured qubits
    void* optimization_data; // Data for circuit optimization
} quantum_circuit;

// Function declarations
quantum_circuit* init_quantum_circuit(size_t num_qubits);
void cleanup_quantum_circuit(struct quantum_circuit* circuit);

void add_quantum_conv_layer(quantum_circuit* circuit, void* params);
void add_quantum_pool_layer(quantum_circuit* circuit, void* params);
void add_quantum_dense_layer(quantum_circuit* circuit, void* params);

size_t count_quantum_parameters(const quantum_circuit* circuit);
void apply_quantum_layers(const quantum_circuit* circuit, void* state);
void quantum_backward_pass(quantum_circuit* circuit, double* gradients);
void update_quantum_parameters(quantum_circuit* circuit, void* optimizer);

#endif // QUANTUM_CIRCUIT_TYPES_H
