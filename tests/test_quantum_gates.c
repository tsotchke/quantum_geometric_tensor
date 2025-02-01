#include "quantum_geometric/core/quantum_operations.h"
#include "quantum_geometric/core/quantum_circuit_operations.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

#define EPSILON 1e-6

static void test_single_qubit_gates() {
    printf("Testing single qubit gates...\n");
    
    // Create 1-qubit circuit
    quantum_circuit_t* circuit = quantum_circuit_create(1);
    assert(circuit != NULL);
    
    // Initialize quantum state |0⟩
    quantum_state* state = init_quantum_state(1);
    assert(state != NULL);
    
    // Test Hadamard
    quantum_circuit_hadamard(circuit, 0);
    quantum_circuit_execute(circuit, state);
    assert(fabs(cabs(state->amplitudes[0]) - M_SQRT1_2) < EPSILON);
    assert(fabs(cabs(state->amplitudes[1]) - M_SQRT1_2) < EPSILON);
    
    // Reset circuit and state
    quantum_circuit_reset(circuit);
    quantum_state_reset(state);
    
    // Test Pauli-X (NOT gate)
    quantum_circuit_pauli_x(circuit, 0);
    quantum_circuit_execute(circuit, state);
    assert(fabs(cabs(state->amplitudes[0])) < EPSILON);
    assert(fabs(cabs(state->amplitudes[1]) - 1.0) < EPSILON);
    
    // Clean up
    quantum_circuit_destroy(circuit);
    quantum_state_destroy(state);
    
    printf("✓ Single qubit gates test passed\n");
}

static void test_two_qubit_gates() {
    printf("Testing two qubit gates...\n");
    
    // Create 2-qubit circuit
    quantum_circuit_t* circuit = quantum_circuit_create(2);
    assert(circuit != NULL);
    
    // Initialize quantum state |00⟩
    quantum_state* state = init_quantum_state(2);
    assert(state != NULL);
    
    // Test CNOT: |00⟩ -> |00⟩
    quantum_circuit_cnot(circuit, 0, 1);
    quantum_circuit_execute(circuit, state);
    assert(fabs(cabs(state->amplitudes[0]) - 1.0) < EPSILON);
    assert(fabs(cabs(state->amplitudes[1])) < EPSILON);
    assert(fabs(cabs(state->amplitudes[2])) < EPSILON);
    assert(fabs(cabs(state->amplitudes[3])) < EPSILON);
    
    // Reset circuit and state
    quantum_circuit_reset(circuit);
    quantum_state_reset(state);
    
    // Prepare |10⟩ with X gate
    quantum_circuit_pauli_x(circuit, 0);
    quantum_circuit_execute(circuit, state);
    
    // Test CNOT: |10⟩ -> |11⟩
    quantum_circuit_cnot(circuit, 0, 1);
    quantum_circuit_execute(circuit, state);
    assert(fabs(cabs(state->amplitudes[0])) < EPSILON);
    assert(fabs(cabs(state->amplitudes[1])) < EPSILON);
    assert(fabs(cabs(state->amplitudes[2])) < EPSILON);
    assert(fabs(cabs(state->amplitudes[3]) - 1.0) < EPSILON);
    
    // Clean up
    quantum_circuit_destroy(circuit);
    quantum_state_destroy(state);
    
    printf("✓ Two qubit gates test passed\n");
}

static void test_quantum_fourier_transform() {
    printf("Testing quantum Fourier transform...\n");
    
    // Create 2-qubit circuit for QFT
    quantum_circuit_t* circuit = quantum_circuit_create(2);
    assert(circuit != NULL);
    
    // Initialize quantum state |00⟩
    quantum_state* state = init_quantum_state(2);
    assert(state != NULL);
    
    // Build 2-qubit QFT circuit
    // H on qubit 0
    quantum_circuit_hadamard(circuit, 0);
    // Controlled phase between qubits
    quantum_circuit_phase(circuit, 1, M_PI/2.0);
    // H on qubit 1
    quantum_circuit_hadamard(circuit, 1);
    // SWAP qubits
    quantum_circuit_swap(circuit, 0, 1);
    
    // Execute QFT
    quantum_circuit_execute(circuit, state);
    
    // Verify output state is uniform superposition
    double expected = 0.5; // 1/sqrt(4)
    for (int i = 0; i < 4; i++) {
        assert(fabs(cabs(state->amplitudes[i]) - expected) < EPSILON);
    }
    
    // Clean up
    quantum_circuit_destroy(circuit);
    quantum_state_destroy(state);
    
    printf("✓ Quantum Fourier transform test passed\n");
}

int main() {
    printf("Running quantum gates tests...\n");
    
    test_single_qubit_gates();
    test_two_qubit_gates();
    test_quantum_fourier_transform();
    
    printf("All quantum gates tests passed!\n");
    return 0;
}
