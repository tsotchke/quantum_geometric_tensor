#ifndef TEST_HELPERS_H
#define TEST_HELPERS_H

#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_state_types.h"
#include <stdlib.h>
#include <string.h>

// Create a test quantum state with given number of qubits
static quantum_state_t* create_test_quantum_state(size_t num_qubits) {
    quantum_state_t* state = (quantum_state_t*)malloc(sizeof(quantum_state_t));
    if (!state) return NULL;
    
    // Initialize state
    state->type = QUANTUM_STATE_PURE;
    state->dimension = num_qubits;
    state->coordinates = (ComplexFloat*)calloc(num_qubits * 2, sizeof(ComplexFloat));
    if (!state->coordinates) {
        free(state);
        return NULL;
    }
    
    // Initialize to |0⟩ state
    for (size_t i = 0; i < num_qubits; i++) {
        state->coordinates[i * 2].real = 1.0;     // Real part
        state->coordinates[i * 2].imag = 0.0;     // Imaginary part
    }
    
    state->is_normalized = true;
    state->hardware = 0; // CPU
    
    return state;
}

// Create a test quantum register with given configuration
static quantum_register_t* create_test_quantum_register(size_t num_qubits, size_t num_ancilla) {
    quantum_register_t* reg = (quantum_register_t*)malloc(sizeof(quantum_register_t));
    if (!reg) return NULL;
    
    // Initialize register
    reg->size = num_qubits + num_ancilla;
    reg->amplitudes = (ComplexFloat*)calloc(reg->size * 2, sizeof(ComplexFloat));
    if (!reg->amplitudes) {
        free(reg);
        return NULL;
    }
    
    // Initialize to |0⟩ state
    for (size_t i = 0; i < reg->size; i++) {
        reg->amplitudes[i * 2].real = 1.0;
        reg->amplitudes[i * 2].imag = 0.0;
    }
    
    reg->system = NULL; // Will be initialized by the quantum system
    
    return reg;
}

// Clean up test resources
static void cleanup_test_resources(quantum_state_t* state,
                                 quantum_register_t* reg) {
    if (state) {
        free(state->coordinates);
        free(state);
    }
    
    if (reg) {
        free(reg->amplitudes);
        free(reg);
    }
}

#endif // TEST_HELPERS_H
