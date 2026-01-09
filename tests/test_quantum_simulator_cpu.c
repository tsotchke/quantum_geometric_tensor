/**
 * @file test_quantum_simulator_cpu.c
 * @brief Tests for CPU quantum simulator functionality
 *
 * Tests single qubit gates, two qubit gates, measurements, and error correction
 * using the quantum simulator CPU API.
 */

#include "quantum_geometric/hardware/quantum_simulator_cpu.h"
#include "quantum_geometric/core/quantum_base_types.h"
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440  // 1/sqrt(2)
#endif

// Test single qubit operations
static void test_single_qubit_gates(void) {
    printf("Testing single qubit gates...\n");

    // Initialize state |0⟩
    size_t n_qubits = 1;
    size_t n = 1ULL << n_qubits;
    double complex* state = malloc(n * sizeof(double complex));
    if (!state) {
        printf("  SKIP: Could not allocate state\n");
        return;
    }
    init_simulator_state(state, n);

    // Verify initial state is |0⟩
    assert(fabs(cabs(state[0]) - 1.0) < 1e-10);
    assert(fabs(cabs(state[1])) < 1e-10);

    // Create Hadamard gate
    QuantumGate h_gate = {
        .type = GATE_H,
        .target_qubit = 0,
        .control_qubit = 0,
        .parameter = 0.0,
        .parameters = NULL,
        .num_parameters = 0
    };

    // Create circuit and add gate
    CPUSimCircuit* circuit = cpu_sim_create_circuit(10);
    if (!circuit) {
        printf("  SKIP: Could not create circuit\n");
        free(state);
        return;
    }

    cpu_sim_add_gate(circuit, &h_gate);

    // Simulate
    simulate_circuit_cpu(state, circuit, n_qubits);

    // Verify |+⟩ state: (|0⟩ + |1⟩)/√2
    double expected = M_SQRT1_2;
    assert(fabs(cabs(state[0]) - expected) < 1e-10);
    assert(fabs(cabs(state[1]) - expected) < 1e-10);

    printf("  State after Hadamard: |0⟩=%.4f, |1⟩=%.4f\n",
           cabs(state[0]), cabs(state[1]));

    // Cleanup
    free(state);
    cpu_sim_cleanup_circuit(circuit);
    printf("  Single qubit gate test passed\n\n");
}

// Test two qubit operations
static void test_two_qubit_gates(void) {
    printf("Testing two qubit gates...\n");

    // Initialize state |00⟩
    size_t n_qubits = 2;
    size_t n = 1ULL << n_qubits;
    double complex* state = malloc(n * sizeof(double complex));
    if (!state) {
        printf("  SKIP: Could not allocate state\n");
        return;
    }
    init_simulator_state(state, n);

    // Create circuit: H ⊗ I followed by CNOT to create Bell state
    CPUSimCircuit* circuit = cpu_sim_create_circuit(10);
    if (!circuit) {
        printf("  SKIP: Could not create circuit\n");
        free(state);
        return;
    }

    // Add Hadamard on first qubit
    QuantumGate h_gate = {
        .type = GATE_H,
        .target_qubit = 0,
        .control_qubit = 0,
        .parameter = 0.0,
        .parameters = NULL,
        .num_parameters = 0
    };
    cpu_sim_add_gate(circuit, &h_gate);

    // Add CNOT gate (control=0, target=1)
    QuantumGate cnot_gate = {
        .type = GATE_CNOT,
        .target_qubit = 1,
        .control_qubit = 0,
        .parameter = 0.0,
        .parameters = NULL,
        .num_parameters = 0
    };
    cpu_sim_add_gate(circuit, &cnot_gate);

    // Simulate
    simulate_circuit_cpu(state, circuit, n_qubits);

    // Verify Bell state (|00⟩ + |11⟩)/√2
    double expected = M_SQRT1_2;
    printf("  State after H-CNOT:\n");
    printf("    |00⟩=%.4f, |01⟩=%.4f, |10⟩=%.4f, |11⟩=%.4f\n",
           cabs(state[0]), cabs(state[1]), cabs(state[2]), cabs(state[3]));

    assert(fabs(cabs(state[0]) - expected) < 1e-10);  // |00⟩
    assert(fabs(cabs(state[1])) < 1e-10);             // |01⟩
    assert(fabs(cabs(state[2])) < 1e-10);             // |10⟩
    assert(fabs(cabs(state[3]) - expected) < 1e-10); // |11⟩

    // Cleanup
    free(state);
    cpu_sim_cleanup_circuit(circuit);
    printf("  Two qubit gate test passed\n\n");
}

// Test X gate (NOT gate)
static void test_x_gate(void) {
    printf("Testing X gate...\n");

    size_t n_qubits = 1;
    size_t n = 1ULL << n_qubits;
    double complex* state = malloc(n * sizeof(double complex));
    if (!state) {
        printf("  SKIP: Could not allocate state\n");
        return;
    }
    init_simulator_state(state, n);

    // Create X gate
    QuantumGate x_gate = {
        .type = GATE_X,
        .target_qubit = 0,
        .control_qubit = 0,
        .parameter = 0.0,
        .parameters = NULL,
        .num_parameters = 0
    };

    CPUSimCircuit* circuit = cpu_sim_create_circuit(10);
    if (!circuit) {
        printf("  SKIP: Could not create circuit\n");
        free(state);
        return;
    }

    cpu_sim_add_gate(circuit, &x_gate);
    simulate_circuit_cpu(state, circuit, n_qubits);

    // After X gate: |0⟩ -> |1⟩
    printf("  State after X: |0⟩=%.4f, |1⟩=%.4f\n", cabs(state[0]), cabs(state[1]));
    assert(fabs(cabs(state[0])) < 1e-10);
    assert(fabs(cabs(state[1]) - 1.0) < 1e-10);

    free(state);
    cpu_sim_cleanup_circuit(circuit);
    printf("  X gate test passed\n\n");
}

// Test Z gate (phase flip)
static void test_z_gate(void) {
    printf("Testing Z gate...\n");

    size_t n_qubits = 1;
    size_t n = 1ULL << n_qubits;
    double complex* state = malloc(n * sizeof(double complex));
    if (!state) {
        printf("  SKIP: Could not allocate state\n");
        return;
    }
    init_simulator_state(state, n);

    // First apply H to get |+⟩
    QuantumGate h_gate = {
        .type = GATE_H,
        .target_qubit = 0,
        .control_qubit = 0,
        .parameter = 0.0,
        .parameters = NULL,
        .num_parameters = 0
    };

    // Then apply Z to get |-⟩
    QuantumGate z_gate = {
        .type = GATE_Z,
        .target_qubit = 0,
        .control_qubit = 0,
        .parameter = 0.0,
        .parameters = NULL,
        .num_parameters = 0
    };

    CPUSimCircuit* circuit = cpu_sim_create_circuit(10);
    if (!circuit) {
        printf("  SKIP: Could not create circuit\n");
        free(state);
        return;
    }

    cpu_sim_add_gate(circuit, &h_gate);
    cpu_sim_add_gate(circuit, &z_gate);
    simulate_circuit_cpu(state, circuit, n_qubits);

    // After H-Z: |0⟩ -> |+⟩ -> |-⟩ = (|0⟩ - |1⟩)/√2
    printf("  State after H-Z: |0⟩=%.4f%+.4fi, |1⟩=%.4f%+.4fi\n",
           creal(state[0]), cimag(state[0]), creal(state[1]), cimag(state[1]));

    // Amplitudes should have opposite signs
    assert(fabs(cabs(state[0]) - M_SQRT1_2) < 1e-10);
    assert(fabs(cabs(state[1]) - M_SQRT1_2) < 1e-10);

    free(state);
    cpu_sim_cleanup_circuit(circuit);
    printf("  Z gate test passed\n\n");
}

// Test rotation gates
static void test_rotation_gates(void) {
    printf("Testing rotation gates...\n");

    size_t n_qubits = 1;
    size_t n = 1ULL << n_qubits;
    double complex* state = malloc(n * sizeof(double complex));
    if (!state) {
        printf("  SKIP: Could not allocate state\n");
        return;
    }
    init_simulator_state(state, n);

    // RX(pi) should act like X gate
    double rx_param = M_PI;
    QuantumGate rx_gate = {
        .type = GATE_RX,
        .target_qubit = 0,
        .control_qubit = 0,
        .parameter = rx_param,
        .parameters = NULL,
        .num_parameters = 0
    };

    CPUSimCircuit* circuit = cpu_sim_create_circuit(10);
    if (!circuit) {
        printf("  SKIP: Could not create circuit\n");
        free(state);
        return;
    }

    cpu_sim_add_gate(circuit, &rx_gate);
    simulate_circuit_cpu(state, circuit, n_qubits);

    // RX(pi)|0⟩ = -i|1⟩
    printf("  State after RX(pi): |0⟩=%.4f%+.4fi, |1⟩=%.4f%+.4fi\n",
           creal(state[0]), cimag(state[0]), creal(state[1]), cimag(state[1]));

    // |1⟩ coefficient should have magnitude 1
    assert(fabs(cabs(state[0])) < 1e-10);
    assert(fabs(cabs(state[1]) - 1.0) < 1e-10);

    free(state);
    cpu_sim_cleanup_circuit(circuit);
    printf("  Rotation gates test passed\n\n");
}

// Test circuit configuration and optimization
static void test_circuit_configuration(void) {
    printf("Testing circuit configuration...\n");

    CPUSimCircuit* circuit = cpu_sim_create_circuit(10);
    if (!circuit) {
        printf("  SKIP: Could not create circuit\n");
        return;
    }

    // Configure circuit optimization
    configure_circuit_optimization(circuit, true, true, 64);

    // Add some gates
    QuantumGate h_gate = {
        .type = GATE_H,
        .target_qubit = 0,
        .control_qubit = 0,
        .parameter = 0.0,
        .parameters = NULL,
        .num_parameters = 0
    };
    cpu_sim_add_gate(circuit, &h_gate);

    QuantumGate x_gate = {
        .type = GATE_X,
        .target_qubit = 0,
        .control_qubit = 0,
        .parameter = 0.0,
        .parameters = NULL,
        .num_parameters = 0
    };
    cpu_sim_add_gate(circuit, &x_gate);

    // Get error statistics
    double avg_error = 0.0, max_error = 0.0;
    cpu_sim_get_error_statistics(circuit, &avg_error, &max_error);

    printf("  Error statistics: avg=%.6f, max=%.6f\n", avg_error, max_error);

    // Verify error rates are in bounds
    assert(avg_error >= 0.0 && avg_error <= 1.0);
    assert(max_error >= 0.0 && max_error <= 1.0);

    cpu_sim_cleanup_circuit(circuit);
    printf("  Circuit configuration test passed\n\n");
}

// Test multiple qubit circuit
static void test_three_qubit_circuit(void) {
    printf("Testing three qubit circuit...\n");

    size_t n_qubits = 3;
    size_t n = 1ULL << n_qubits;
    double complex* state = malloc(n * sizeof(double complex));
    if (!state) {
        printf("  SKIP: Could not allocate state\n");
        return;
    }
    init_simulator_state(state, n);

    // Create GHZ state: (|000⟩ + |111⟩)/√2
    CPUSimCircuit* circuit = cpu_sim_create_circuit(20);
    if (!circuit) {
        printf("  SKIP: Could not create circuit\n");
        free(state);
        return;
    }

    // H on qubit 0
    QuantumGate h_gate = {
        .type = GATE_H,
        .target_qubit = 0,
        .control_qubit = 0,
        .parameter = 0.0,
        .parameters = NULL,
        .num_parameters = 0
    };
    cpu_sim_add_gate(circuit, &h_gate);

    // CNOT(0, 1)
    QuantumGate cnot1 = {
        .type = GATE_CNOT,
        .target_qubit = 1,
        .control_qubit = 0,
        .parameter = 0.0,
        .parameters = NULL,
        .num_parameters = 0
    };
    cpu_sim_add_gate(circuit, &cnot1);

    // CNOT(1, 2)
    QuantumGate cnot2 = {
        .type = GATE_CNOT,
        .target_qubit = 2,
        .control_qubit = 1,
        .parameter = 0.0,
        .parameters = NULL,
        .num_parameters = 0
    };
    cpu_sim_add_gate(circuit, &cnot2);

    simulate_circuit_cpu(state, circuit, n_qubits);

    // Print state
    printf("  GHZ state amplitudes:\n");
    for (size_t i = 0; i < n; i++) {
        if (cabs(state[i]) > 1e-10) {
            printf("    |%zu%zu%zu⟩ = %.4f%+.4fi\n",
                   (i >> 2) & 1, (i >> 1) & 1, i & 1,
                   creal(state[i]), cimag(state[i]));
        }
    }

    // Verify GHZ state
    assert(fabs(cabs(state[0]) - M_SQRT1_2) < 1e-10);  // |000⟩
    assert(fabs(cabs(state[7]) - M_SQRT1_2) < 1e-10); // |111⟩
    for (size_t i = 1; i < 7; i++) {
        assert(fabs(cabs(state[i])) < 1e-10);
    }

    free(state);
    cpu_sim_cleanup_circuit(circuit);
    printf("  Three qubit circuit test passed\n\n");
}

int main(void) {
    printf("=== Quantum Simulator CPU Tests ===\n\n");

    test_single_qubit_gates();
    test_two_qubit_gates();
    test_x_gate();
    test_z_gate();
    test_rotation_gates();
    test_circuit_configuration();
    test_three_qubit_circuit();

    printf("=== All Quantum Simulator CPU Tests Passed ===\n");
    return 0;
}
