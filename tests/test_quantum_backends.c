#include "quantum_geometric/hardware/quantum_ibm_backend.h"
#include "quantum_geometric/hardware/quantum_rigetti_backend.h"
#include "quantum_geometric/hardware/quantum_dwave_backend.h"
#include "quantum_geometric/core/quantum_geometric_operations.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>

// Test parameters
#define NUM_QUBITS 5
#define EPSILON 1e-6
#define NUM_TRIALS 3

// Create test circuit
static QuantumCircuit* create_test_circuit(void) {
    QuantumCircuit* circuit = init_quantum_circuit(NUM_QUBITS);
    if (!circuit) return NULL;
    
    // Add some test gates
    add_gate(circuit, (QuantumGate){
        .type = GATE_H,
        .target = 0
    });
    
    add_gate(circuit, (QuantumGate){
        .type = GATE_CNOT,
        .control = 0,
        .target = 1
    });
    
    add_gate(circuit, (QuantumGate){
        .type = GATE_H,
        .target = 2
    });
    
    add_gate(circuit, (QuantumGate){
        .type = GATE_CNOT,
        .control = 2,
        .target = 3
    });
    
    return circuit;
}

// Create test QUBO
static QUBO* create_test_qubo(void) {
    QUBO* qubo = init_qubo(NUM_QUBITS);
    if (!qubo) return NULL;
    
    // Set linear terms
    for (size_t i = 0; i < NUM_QUBITS; i++) {
        qubo->linear[i] = -1.0;  // Minimize number of 1s
    }
    
    // Set quadratic terms (coupling)
    for (size_t i = 0; i < NUM_QUBITS - 1; i++) {
        qubo->quadratic[i * NUM_QUBITS + i + 1] = 2.0;  // Favor alternating bits
    }
    
    return qubo;
}

// Test IBM Quantum backend
static void test_ibm_backend(void) {
    printf("Testing IBM Quantum backend...\n");
    
    // Initialize backend
    const char* token = getenv("IBM_QUANTUM_TOKEN");
    if (!token) {
        printf("Skipping IBM tests (no token)\n");
        return;
    }
    
    IBMQuantumBackend* backend = init_ibm_backend(token, "ibmq_qasm_simulator");
    assert(backend != NULL);
    
    // Create and run test circuit
    QuantumCircuit* circuit = create_test_circuit();
    assert(circuit != NULL);
    
    QuantumResult result = {0};
    double total_time = 0.0;
    
    for (int trial = 0; trial < NUM_TRIALS; trial++) {
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        int status = submit_quantum_circuit(backend, circuit, &result);
        assert(status == 0);
        
        clock_gettime(CLOCK_MONOTONIC, &end);
        total_time += (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) * 1e-9;
        
        // Verify results
        assert(fabs(result.probabilities[0] - 0.25) < EPSILON);
        assert(fabs(result.probabilities[3] - 0.25) < EPSILON);
        assert(fabs(result.probabilities[12] - 0.25) < EPSILON);
        assert(fabs(result.probabilities[15] - 0.25) < EPSILON);
    }
    
    printf("  Average execution time: %.3f seconds\n",
           total_time / NUM_TRIALS);
    
    cleanup_quantum_circuit(circuit);
    cleanup_ibm_backend(backend);
}

// Test Rigetti backend
static void test_rigetti_backend(void) {
    printf("Testing Rigetti backend...\n");
    
    // Initialize backend
    const char* api_key = getenv("RIGETTI_API_KEY");
    if (!api_key) {
        printf("Skipping Rigetti tests (no API key)\n");
        return;
    }
    
    RigettiBackend* backend = init_rigetti_backend(api_key, "Aspen-M-1");
    assert(backend != NULL);
    
    // Create and run test circuit
    QuantumCircuit* circuit = create_test_circuit();
    assert(circuit != NULL);
    
    QuantumResult result = {0};
    double total_time = 0.0;
    
    for (int trial = 0; trial < NUM_TRIALS; trial++) {
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        int status = submit_rigetti_circuit(backend, circuit, &result);
        assert(status == 0);
        
        clock_gettime(CLOCK_MONOTONIC, &end);
        total_time += (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) * 1e-9;
        
        // Verify results (similar to IBM due to same circuit)
        assert(fabs(result.probabilities[0] - 0.25) < EPSILON);
        assert(fabs(result.probabilities[3] - 0.25) < EPSILON);
        assert(fabs(result.probabilities[12] - 0.25) < EPSILON);
        assert(fabs(result.probabilities[15] - 0.25) < EPSILON);
    }
    
    printf("  Average execution time: %.3f seconds\n",
           total_time / NUM_TRIALS);
    
    cleanup_quantum_circuit(circuit);
    cleanup_rigetti_backend(backend);
}

// Test D-Wave backend
static void test_dwave_backend(void) {
    printf("Testing D-Wave backend...\n");
    
    // Initialize backend
    const char* api_key = getenv("DWAVE_API_KEY");
    if (!api_key) {
        printf("Skipping D-Wave tests (no API key)\n");
        return;
    }
    
    DWaveBackend* backend = init_dwave_backend(api_key, "Advantage_system4.1");
    assert(backend != NULL);
    
    // Create and run test QUBO
    QUBO* qubo = create_test_qubo();
    assert(qubo != NULL);
    
    QUBOResult result = {0};
    double total_time = 0.0;
    
    for (int trial = 0; trial < NUM_TRIALS; trial++) {
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        int status = submit_dwave_problem(backend, qubo, &result);
        assert(status == 0);
        
        clock_gettime(CLOCK_MONOTONIC, &end);
        total_time += (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) * 1e-9;
        
        // Verify results
        assert(result.num_solutions > 0);
        
        // Check if best solution has alternating bits
        const char* best_solution = result.solutions[0];
        for (size_t i = 0; i < NUM_QUBITS - 1; i++) {
            assert(best_solution[i] != best_solution[i + 1]);
        }
        
        // Verify energy is negative (minimization)
        assert(result.energies[0] < 0.0);
    }
    
    printf("  Average execution time: %.3f seconds\n",
           total_time / NUM_TRIALS);
    
    cleanup_qubo(qubo);
    cleanup_dwave_backend(backend);
}

// Compare backends
static void compare_backends(void) {
    printf("\nComparing quantum backends...\n");
    
    // Create Bell state circuit
    QuantumCircuit* bell_circuit = init_quantum_circuit(2);
    assert(bell_circuit != NULL);
    
    add_gate(bell_circuit, (QuantumGate){
        .type = GATE_H,
        .target = 0
    });
    
    add_gate(bell_circuit, (QuantumGate){
        .type = GATE_CNOT,
        .control = 0,
        .target = 1
    });
    
    // Test on each backend
    const char* ibm_token = getenv("IBM_QUANTUM_TOKEN");
    const char* rigetti_key = getenv("RIGETTI_API_KEY");
    
    if (ibm_token) {
        printf("\nIBM Quantum results:\n");
        IBMQuantumBackend* ibm = init_ibm_backend(ibm_token,
                                                "ibmq_qasm_simulator");
        assert(ibm != NULL);
        
        QuantumResult ibm_result = {0};
        int status = submit_quantum_circuit(ibm, bell_circuit,
                                         &ibm_result);
        assert(status == 0);
        
        printf("  |00>: %.3f\n", ibm_result.probabilities[0]);
        printf("  |11>: %.3f\n", ibm_result.probabilities[3]);
        
        cleanup_ibm_backend(ibm);
    }
    
    if (rigetti_key) {
        printf("\nRigetti results:\n");
        RigettiBackend* rigetti = init_rigetti_backend(rigetti_key,
                                                     "Aspen-M-1");
        assert(rigetti != NULL);
        
        QuantumResult rigetti_result = {0};
        int status = submit_rigetti_circuit(rigetti, bell_circuit,
                                         &rigetti_result);
        assert(status == 0);
        
        printf("  |00>: %.3f\n", rigetti_result.probabilities[0]);
        printf("  |11>: %.3f\n", rigetti_result.probabilities[3]);
        
        cleanup_rigetti_backend(rigetti);
    }
    
    cleanup_quantum_circuit(bell_circuit);
}

int main(void) {
    printf("Starting quantum backend tests...\n\n");
    
    // Run tests
    test_ibm_backend();
    test_rigetti_backend();
    test_dwave_backend();
    
    // Compare backends
    compare_backends();
    
    printf("\nAll quantum backend tests passed!\n");
    return 0;
}
