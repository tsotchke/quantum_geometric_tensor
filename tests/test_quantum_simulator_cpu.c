#include "quantum_geometric/hardware/quantum_simulator.h"
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <time.h>

// Test single qubit operations
static void test_single_qubit_gates(void) {
    printf("Testing single qubit gates...\n");
    
    // Initialize state |0⟩
    size_t n_qubits = 1;
    size_t n = 1ULL << n_qubits;
    double complex* state = malloc(n * sizeof(double complex));
    init_simulator_state(state, n);
    
    // Create Hadamard gate
    QuantumGate h_gate = {
        .type = GATE_SINGLE,
        .target = 0,
        .matrix = {
            1.0/sqrt(2.0), 1.0/sqrt(2.0),
            1.0/sqrt(2.0), -1.0/sqrt(2.0)
        }
    };
    
    // Create circuit
    QuantumCircuit* circuit = init_quantum_circuit(1);
    add_gate_to_circuit(circuit, &h_gate);
    
    // Simulate
    simulate_circuit_cpu(state, circuit, n_qubits);
    
    // Verify |+⟩ state
    assert(fabs(cabs(state[0]) - 1.0/sqrt(2.0)) < 1e-10);
    assert(fabs(cabs(state[1]) - 1.0/sqrt(2.0)) < 1e-10);
    
    // Cleanup
    free(state);
    cleanup_circuit(circuit);
    printf("Single qubit gate test passed\n");
}

// Test two qubit operations
static void test_two_qubit_gates(void) {
    printf("Testing two qubit gates...\n");
    
    // Initialize state |00⟩
    size_t n_qubits = 2;
    size_t n = 1ULL << n_qubits;
    double complex* state = malloc(n * sizeof(double complex));
    init_simulator_state(state, n);
    
    // Create CNOT gate
    QuantumGate cnot_gate = {
        .type = GATE_TWO,
        .control = 0,
        .target = 1,
        .matrix = {
            1.0, 0.0,
            0.0, 1.0,
            0.0, 1.0,
            1.0, 0.0
        }
    };
    
    // Create circuit: H ⊗ I followed by CNOT
    QuantumCircuit* circuit = init_quantum_circuit(2);
    
    // Add Hadamard on first qubit
    QuantumGate h_gate = {
        .type = GATE_SINGLE,
        .target = 0,
        .matrix = {
            1.0/sqrt(2.0), 1.0/sqrt(2.0),
            1.0/sqrt(2.0), -1.0/sqrt(2.0)
        }
    };
    
    add_gate_to_circuit(circuit, &h_gate);
    add_gate_to_circuit(circuit, &cnot_gate);
    
    // Simulate
    simulate_circuit_cpu(state, circuit, n_qubits);
    
    // Verify Bell state (|00⟩ + |11⟩)/√2
    assert(fabs(cabs(state[0]) - 1.0/sqrt(2.0)) < 1e-10);
    assert(fabs(cabs(state[1])) < 1e-10);
    assert(fabs(cabs(state[2])) < 1e-10);
    assert(fabs(cabs(state[3]) - 1.0/sqrt(2.0)) < 1e-10);
    
    // Cleanup
    free(state);
    cleanup_circuit(circuit);
    printf("Two qubit gate test passed\n");
}

// Test measurement operations
static void test_measurements(void) {
    printf("Testing measurements...\n");
    
    // Initialize state |+⟩
    size_t n_qubits = 1;
    size_t n = 1ULL << n_qubits;
    double complex* state = malloc(n * sizeof(double complex));
    init_simulator_state(state, n);
    
    // Apply Hadamard
    QuantumGate h_gate = {
        .type = GATE_SINGLE,
        .target = 0,
        .matrix = {
            1.0/sqrt(2.0), 1.0/sqrt(2.0),
            1.0/sqrt(2.0), -1.0/sqrt(2.0)
        }
    };
    
    QuantumCircuit* circuit = init_quantum_circuit(2);
    add_gate_to_circuit(circuit, &h_gate);
    
    // Add measurement
    QuantumGate measure = {
        .type = GATE_MEASURE,
        .target = 0
    };
    add_gate_to_circuit(circuit, &measure);
    
    // Run multiple times to verify distribution
    int zeros = 0;
    int trials = 1000;
    
    srand(time(NULL));
    for (int i = 0; i < trials; i++) {
        double complex* trial_state = malloc(n * sizeof(double complex));
        memcpy(trial_state, state, n * sizeof(double complex));
        simulate_circuit_cpu(trial_state, circuit, n_qubits);
        
        // Count |0⟩ outcomes
        if (cabs(trial_state[0]) > 0.9) {
            zeros++;
        }
        free(trial_state);
    }
    
    // Verify roughly 50-50 distribution
    double ratio = (double)zeros / trials;
    assert(fabs(ratio - 0.5) < 0.1);
    
    // Cleanup
    free(state);
    cleanup_circuit(circuit);
    printf("Measurement test passed\n");
}

// Test error correction
static void test_error_correction(void) {
    printf("Testing error correction...\n");
    
    // Initialize state |0⟩
    size_t n_qubits = 1;
    size_t n = 1ULL << n_qubits;
    double complex* state = malloc(n * sizeof(double complex));
    init_simulator_state(state, n);
    
    // Create circuit with error detection and correction
    QuantumCircuit* circuit = init_quantum_circuit(3);
    
    // Configure error correction
    configure_circuit_optimization(circuit, true, true, 64);
    
    // Add error detection gate
    QuantumGate detect = {
        .type = GATE_ERROR_DETECT,
        .target = 0,
        .error_threshold = 0.01
    };
    add_gate_to_circuit(circuit, &detect);
    
    // Add error correction gate
    QuantumGate correct = {
        .type = GATE_ERROR_CORRECT,
        .target = 0,
        .error_threshold = 0.01
    };
    add_gate_to_circuit(circuit, &correct);
    
    // Simulate
    simulate_circuit_cpu(state, circuit, n_qubits);
    
    // Get error statistics
    double avg_error, max_error;
    get_error_statistics(circuit, &avg_error, &max_error);
    
    // Verify error rates are within bounds
    assert(avg_error >= 0.0 && avg_error <= 1.0);
    assert(max_error >= 0.0 && max_error <= 1.0);
    assert(max_error >= avg_error);
    
    // Cleanup
    free(state);
    cleanup_circuit(circuit);
    printf("Error correction test passed\n");
}

// Test optimization configuration
static void test_optimization_config(void) {
    printf("Testing optimization configuration...\n");
    
    // Initialize circuit
    QuantumCircuit* circuit = init_quantum_circuit(1);
    
    // Test default configuration
    assert(circuit->use_error_correction == true);
    assert(circuit->use_tensor_networks == true);
    assert(circuit->cache_line_size == 64);
    
    // Test configuration update
    configure_circuit_optimization(circuit, false, false, 128);
    assert(circuit->use_error_correction == false);
    assert(circuit->use_tensor_networks == false);
    assert(circuit->cache_line_size == 128);
    
    // Cleanup
    cleanup_circuit(circuit);
    printf("Optimization configuration test passed\n");
}

int main(void) {
    printf("Running quantum simulator CPU tests...\n\n");
    
    test_single_qubit_gates();
    test_two_qubit_gates();
    test_measurements();
    test_error_correction();
    test_optimization_config();
    
    printf("\nAll quantum simulator CPU tests passed!\n");
    return 0;
}
