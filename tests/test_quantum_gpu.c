#include "quantum_geometric/hardware/quantum_geometric_gpu.h"
#include "quantum_geometric/core/memory_pool.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// Test parameters
#define NUM_QUBITS 10
#define STATE_SIZE (1ULL << NUM_QUBITS)
#define EPSILON 1e-10

// Test helpers
static double complex* create_test_state(size_t size) {
    double complex* state = aligned_alloc(64,
        size * sizeof(double complex));
    if (!state) return NULL;
    
    // Initialize to |0⟩ state
    state[0] = 1.0 + 0.0 * I;
    for (size_t i = 1; i < size; i++) {
        state[i] = 0.0 + 0.0 * I;
    }
    
    return state;
}

static bool compare_states(const double complex* a,
                         const double complex* b,
                         size_t size) {
    for (size_t i = 0; i < size; i++) {
        if (cabs(a[i] - b[i]) > EPSILON) {
            return false;
        }
    }
    return true;
}

// Test single-qubit gates
static void test_single_qubit_gates(void) {
    printf("Testing single-qubit gates...\n");
    
    // Initialize GPU
    GPUContext* ctx = quantum_gpu_init();
    assert(ctx != NULL);
    
    // Create test state
    double complex* state = create_test_state(STATE_SIZE);
    assert(state != NULL);
    
    // Create reference state
    double complex* ref_state = aligned_alloc(64,
        STATE_SIZE * sizeof(double complex));
    assert(ref_state != NULL);
    
    // Test Hadamard gate
    {
        printf("  Testing Hadamard gate...\n");
        
        // Apply H gate on CPU
        memcpy(ref_state, state, STATE_SIZE * sizeof(double complex));
        double inv_sqrt2 = 1.0 / sqrt(2.0);
        ref_state[0] = inv_sqrt2 * (1.0 + 0.0 * I);
        ref_state[1] = inv_sqrt2 * (1.0 + 0.0 * I);
        
        // Apply H gate on GPU
        QuantumGate gate = {
            .type = GATE_H,
            .target = 0
        };
        apply_quantum_gates_gpu(ctx, &gate, 1, state, NUM_QUBITS);
        
        // Compare results
        assert(compare_states(state, ref_state, STATE_SIZE));
    }
    
    // Test Pauli-X gate
    {
        printf("  Testing Pauli-X gate...\n");
        
        // Apply X gate on CPU
        memcpy(ref_state, state, STATE_SIZE * sizeof(double complex));
        ref_state[0] = 0.0 + 0.0 * I;
        ref_state[1] = 1.0 + 0.0 * I;
        
        // Apply X gate on GPU
        QuantumGate gate = {
            .type = GATE_X,
            .target = 0
        };
        apply_quantum_gates_gpu(ctx, &gate, 1, state, NUM_QUBITS);
        
        // Compare results
        assert(compare_states(state, ref_state, STATE_SIZE));
    }
    
    // Test Pauli-Y gate
    {
        printf("  Testing Pauli-Y gate...\n");
        
        // Apply Y gate on CPU
        memcpy(ref_state, state, STATE_SIZE * sizeof(double complex));
        ref_state[0] = 0.0 + 0.0 * I;
        ref_state[1] = 0.0 + 1.0 * I;
        
        // Apply Y gate on GPU
        QuantumGate gate = {
            .type = GATE_Y,
            .target = 0
        };
        apply_quantum_gates_gpu(ctx, &gate, 1, state, NUM_QUBITS);
        
        // Compare results
        assert(compare_states(state, ref_state, STATE_SIZE));
    }
    
    // Test Pauli-Z gate
    {
        printf("  Testing Pauli-Z gate...\n");
        
        // Apply Z gate on CPU
        memcpy(ref_state, state, STATE_SIZE * sizeof(double complex));
        ref_state[1] = -ref_state[1];
        
        // Apply Z gate on GPU
        QuantumGate gate = {
            .type = GATE_Z,
            .target = 0
        };
        apply_quantum_gates_gpu(ctx, &gate, 1, state, NUM_QUBITS);
        
        // Compare results
        assert(compare_states(state, ref_state, STATE_SIZE));
    }
    
    // Clean up
    free(state);
    free(ref_state);
    quantum_gpu_cleanup(ctx);
    
    printf("Single-qubit gate tests passed!\n");
}

// Test two-qubit gates
static void test_two_qubit_gates(void) {
    printf("Testing two-qubit gates...\n");
    
    // Initialize GPU
    GPUContext* ctx = quantum_gpu_init();
    assert(ctx != NULL);
    
    // Create test state
    double complex* state = create_test_state(STATE_SIZE);
    assert(state != NULL);
    
    // Create reference state
    double complex* ref_state = aligned_alloc(64,
        STATE_SIZE * sizeof(double complex));
    assert(ref_state != NULL);
    
    // Test CNOT gate
    {
        printf("  Testing CNOT gate...\n");
        
        // Prepare input state |10⟩
        state[2] = 1.0 + 0.0 * I;  // |10⟩
        state[0] = 0.0 + 0.0 * I;  // Clear |00⟩
        
        // Apply CNOT on CPU
        memcpy(ref_state, state, STATE_SIZE * sizeof(double complex));
        ref_state[2] = 0.0 + 0.0 * I;  // |10⟩ -> 0
        ref_state[3] = 1.0 + 0.0 * I;  // |11⟩ -> 1
        
        // Apply CNOT on GPU
        QuantumGate gate = {
            .type = GATE_CNOT,
            .control = 1,
            .target = 0
        };
        apply_quantum_gates_gpu(ctx, &gate, 1, state, NUM_QUBITS);
        
        // Compare results
        assert(compare_states(state, ref_state, STATE_SIZE));
    }
    
    // Test SWAP gate
    {
        printf("  Testing SWAP gate...\n");
        
        // Prepare input state |10⟩
        state[2] = 1.0 + 0.0 * I;  // |10⟩
        state[3] = 0.0 + 0.0 * I;  // Clear |11⟩
        
        // Apply SWAP on CPU
        memcpy(ref_state, state, STATE_SIZE * sizeof(double complex));
        ref_state[2] = 0.0 + 0.0 * I;  // |10⟩ -> 0
        ref_state[1] = 1.0 + 0.0 * I;  // |01⟩ -> 1
        
        // Apply SWAP on GPU
        QuantumGate gate = {
            .type = GATE_SWAP,
            .qubit1 = 0,
            .qubit2 = 1
        };
        apply_quantum_gates_gpu(ctx, &gate, 1, state, NUM_QUBITS);
        
        // Compare results
        assert(compare_states(state, ref_state, STATE_SIZE));
    }
    
    // Clean up
    free(state);
    free(ref_state);
    quantum_gpu_cleanup(ctx);
    
    printf("Two-qubit gate tests passed!\n");
}

// Test quantum error correction
static void test_error_correction(void) {
    printf("Testing quantum error correction...\n");
    
    // Initialize GPU
    GPUContext* ctx = quantum_gpu_init();
    assert(ctx != NULL);
    
    // Create test state with error
    double complex* state = create_test_state(STATE_SIZE);
    assert(state != NULL);
    
    // Apply bit flip error
    state[0] = 0.0 + 0.0 * I;
    state[1] = 1.0 + 0.0 * I;
    
    // Create reference state
    double complex* ref_state = aligned_alloc(64,
        STATE_SIZE * sizeof(double complex));
    assert(ref_state != NULL);
    memcpy(ref_state, state, STATE_SIZE * sizeof(double complex));
    
    // Apply error correction
    size_t stabilizer_qubits[4] = {0, 1, 2, 3};
    bool syndrome[1];
    
    // Measure syndrome on GPU
    measure_syndrome_gpu(ctx, state, syndrome, stabilizer_qubits, 1, NUM_QUBITS);
    
    // Apply correction if needed
    if (syndrome[0]) {
        QuantumGate correction = {
            .type = GATE_X,
            .target = 0
        };
        apply_quantum_gates_gpu(ctx, &correction, 1, state, NUM_QUBITS);
    }
    
    // Verify correction
    ref_state[0] = 1.0 + 0.0 * I;
    ref_state[1] = 0.0 + 0.0 * I;
    assert(compare_states(state, ref_state, STATE_SIZE));
    
    // Clean up
    free(state);
    free(ref_state);
    quantum_gpu_cleanup(ctx);
    
    printf("Error correction tests passed!\n");
}

int main(void) {
    printf("Starting quantum GPU tests...\n");
    
    // Check if GPU is available
    if (!is_gpu_available()) {
        printf("No GPU available, skipping tests\n");
        return 0;
    }
    
    // Run tests
    test_single_qubit_gates();
    test_two_qubit_gates();
    test_error_correction();
    
    printf("All quantum GPU tests passed!\n");
    return 0;
}
