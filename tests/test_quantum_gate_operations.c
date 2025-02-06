#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include "quantum_geometric/core/quantum_gate_operations.h"
#include "quantum_geometric/core/quantum_types.h"

#define EPSILON 1e-6f

// Helper function to print matrix
static void print_matrix(const ComplexFloat* matrix, size_t dim) {
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            ComplexFloat val = matrix[i * dim + j];
            printf("%.3f%+.3fi ", val.real, val.imag);
        }
        printf("\n");
    }
    printf("\n");
}

// Helper function to measure time
static double get_time_ms(clock_t start) {
    return ((double)(clock() - start)) * 1000.0 / CLOCKS_PER_SEC;
}

static void test_create_fixed_gates(void) {
    printf("\n=== Testing Fixed Gate Creation ===\n");
    clock_t start = clock();
    
    size_t qubits[] = {0};
    
    // Test Identity gate
    quantum_gate_t* i_gate = create_quantum_gate(GATE_TYPE_I, qubits, 1, NULL, 0);
    assert(i_gate != NULL);
    assert(i_gate->type == GATE_TYPE_I);
    assert(i_gate->is_parameterized == false);
    assert(i_gate->num_parameters == 0);
    assert(i_gate->parameters == NULL);
    printf("\nIdentity Gate Matrix:\n");
    print_matrix(i_gate->matrix, 2);
    destroy_quantum_gate(i_gate);
    
    // Test X gate (NOT gate)
    quantum_gate_t* x_gate = create_quantum_gate(GATE_TYPE_X, qubits, 1, NULL, 0);
    assert(x_gate != NULL);
    assert(x_gate->type == GATE_TYPE_X);
    assert(x_gate->is_parameterized == false);
    assert(fabs(x_gate->matrix[0].real) < EPSILON && fabs(x_gate->matrix[0].imag) < EPSILON);
    assert(fabs(x_gate->matrix[1].real - 1.0f) < EPSILON && fabs(x_gate->matrix[1].imag) < EPSILON);
    assert(fabs(x_gate->matrix[2].real - 1.0f) < EPSILON && fabs(x_gate->matrix[2].imag) < EPSILON);
    assert(fabs(x_gate->matrix[3].real) < EPSILON && fabs(x_gate->matrix[3].imag) < EPSILON);
    printf("\nX (NOT) Gate Matrix:\n");
    print_matrix(x_gate->matrix, 2);
    destroy_quantum_gate(x_gate);
    
    // Test Y gate
    quantum_gate_t* y_gate = create_quantum_gate(GATE_TYPE_Y, qubits, 1, NULL, 0);
    assert(y_gate != NULL);
    assert(y_gate->type == GATE_TYPE_Y);
    assert(y_gate->is_parameterized == false);
    assert(fabs(y_gate->matrix[0].real) < EPSILON && fabs(y_gate->matrix[0].imag) < EPSILON);
    assert(fabs(y_gate->matrix[1].real) < EPSILON && fabs(y_gate->matrix[1].imag + 1.0f) < EPSILON);
    assert(fabs(y_gate->matrix[2].real) < EPSILON && fabs(y_gate->matrix[2].imag - 1.0f) < EPSILON);
    assert(fabs(y_gate->matrix[3].real) < EPSILON && fabs(y_gate->matrix[3].imag) < EPSILON);
    printf("\nY Gate Matrix:\n");
    print_matrix(y_gate->matrix, 2);
    destroy_quantum_gate(y_gate);
    
    // Test Z gate
    quantum_gate_t* z_gate = create_quantum_gate(GATE_TYPE_Z, qubits, 1, NULL, 0);
    assert(z_gate != NULL);
    assert(z_gate->type == GATE_TYPE_Z);
    assert(z_gate->is_parameterized == false);
    assert(fabs(z_gate->matrix[0].real - 1.0f) < EPSILON);
    assert(fabs(z_gate->matrix[1].real) < EPSILON);
    assert(fabs(z_gate->matrix[2].real) < EPSILON);
    assert(fabs(z_gate->matrix[3].real + 1.0f) < EPSILON);
    printf("\nZ Gate Matrix:\n");
    print_matrix(z_gate->matrix, 2);
    destroy_quantum_gate(z_gate);
    
    // Test H gate (Hadamard)
    quantum_gate_t* h_gate = create_quantum_gate(GATE_TYPE_H, qubits, 1, NULL, 0);
    assert(h_gate != NULL);
    assert(h_gate->type == GATE_TYPE_H);
    assert(h_gate->is_parameterized == false);
    float inv_sqrt2 = 1.0f / sqrtf(2.0f);
    assert(fabs(h_gate->matrix[0].real - inv_sqrt2) < EPSILON);
    assert(fabs(h_gate->matrix[1].real - inv_sqrt2) < EPSILON);
    assert(fabs(h_gate->matrix[2].real - inv_sqrt2) < EPSILON);
    assert(fabs(h_gate->matrix[3].real + inv_sqrt2) < EPSILON);
    printf("\nH (Hadamard) Gate Matrix:\n");
    print_matrix(h_gate->matrix, 2);
    destroy_quantum_gate(h_gate);
    
    // Test S gate (Phase gate)
    quantum_gate_t* s_gate = create_quantum_gate(GATE_TYPE_S, qubits, 1, NULL, 0);
    assert(s_gate != NULL);
    assert(s_gate->type == GATE_TYPE_S);
    assert(s_gate->is_parameterized == false);
    assert(fabs(s_gate->matrix[0].real - 1.0f) < EPSILON);
    assert(fabs(s_gate->matrix[1].real) < EPSILON);
    assert(fabs(s_gate->matrix[2].real) < EPSILON);
    assert(fabs(s_gate->matrix[3].imag - 1.0f) < EPSILON);
    printf("\nS (Phase) Gate Matrix:\n");
    print_matrix(s_gate->matrix, 2);
    destroy_quantum_gate(s_gate);
    
    // Test T gate (π/8 gate)
    quantum_gate_t* t_gate = create_quantum_gate(GATE_TYPE_T, qubits, 1, NULL, 0);
    assert(t_gate != NULL);
    assert(t_gate->type == GATE_TYPE_T);
    assert(t_gate->is_parameterized == false);
    assert(fabs(t_gate->matrix[0].real - 1.0f) < EPSILON);
    assert(fabs(t_gate->matrix[1].real) < EPSILON);
    assert(fabs(t_gate->matrix[2].real) < EPSILON);
    assert(fabs(t_gate->matrix[3].real - cos(M_PI/4)) < EPSILON);
    assert(fabs(t_gate->matrix[3].imag - sin(M_PI/4)) < EPSILON);
    printf("\nT (π/8) Gate Matrix:\n");
    print_matrix(t_gate->matrix, 2);
    destroy_quantum_gate(t_gate);
    
    printf("Time: %.3f ms\n", get_time_ms(start));
}

static void test_create_parameterized_gates(void) {
    printf("\n=== Testing Parameterized Gate Creation ===\n");
    clock_t start = clock();
    
    size_t qubits[] = {0};
    double params[] = {0.5};
    
    // Test RX gate
    quantum_gate_t* rx_gate = create_quantum_gate(GATE_TYPE_RX, qubits, 1, params, 1);
    assert(rx_gate != NULL);
    assert(rx_gate->type == GATE_TYPE_RX);
    assert(rx_gate->is_parameterized == true);
    assert(rx_gate->num_parameters == 1);
    assert(rx_gate->parameters != NULL);
    assert(fabs(rx_gate->parameters[0] - 0.5) < EPSILON);
    
    printf("\nRX Gate Matrix (theta = 0.5):\n");
    print_matrix(rx_gate->matrix, 2);
    destroy_quantum_gate(rx_gate);
    
    // Test RY gate
    quantum_gate_t* ry_gate = create_quantum_gate(GATE_TYPE_RY, qubits, 1, params, 1);
    assert(ry_gate != NULL);
    assert(ry_gate->type == GATE_TYPE_RY);
    assert(ry_gate->is_parameterized == true);
    
    printf("\nRY Gate Matrix (theta = 0.5):\n");
    print_matrix(ry_gate->matrix, 2);
    destroy_quantum_gate(ry_gate);
    
    // Test RZ gate
    quantum_gate_t* rz_gate = create_quantum_gate(GATE_TYPE_RZ, qubits, 1, params, 1);
    assert(rz_gate != NULL);
    assert(rz_gate->type == GATE_TYPE_RZ);
    assert(rz_gate->is_parameterized == true);
    
    printf("\nRZ Gate Matrix (theta = 0.5):\n");
    print_matrix(rz_gate->matrix, 2);
    destroy_quantum_gate(rz_gate);
    
    // Test invalid parameter count
    quantum_gate_t* invalid_gate = create_quantum_gate(GATE_TYPE_RX, qubits, 1, NULL, 0);
    assert(invalid_gate == NULL);
    printf("Invalid parameter count test passed\n");
    
    printf("Time: %.3f ms\n", get_time_ms(start));
}

static void test_update_parameters(void) {
    printf("\n=== Testing Parameter Updates ===\n");
    clock_t start = clock();
    
    size_t qubits[] = {0};
    double params[] = {0.5};
    quantum_gate_t* rx_gate = create_quantum_gate(GATE_TYPE_RX, qubits, 1, params, 1);
    
    printf("\nRX Gate Matrix before update (theta = 0.5):\n");
    print_matrix(rx_gate->matrix, 2);
    
    // Test valid update
    double new_params[] = {1.0};
    bool success = update_gate_parameters(rx_gate, new_params, 1);
    assert(success);
    assert(fabs(rx_gate->parameters[0] - 1.0) < EPSILON);
    
    printf("\nRX Gate Matrix after update (theta = 1.0):\n");
    print_matrix(rx_gate->matrix, 2);
    
    // Test invalid updates
    success = update_gate_parameters(rx_gate, NULL, 1);
    assert(!success);
    printf("NULL parameters test passed\n");
    
    success = update_gate_parameters(rx_gate, new_params, 2);
    assert(!success);
    printf("Invalid parameter count test passed\n");
    
    destroy_quantum_gate(rx_gate);
    
    // Test update on non-parameterized gate
    quantum_gate_t* x_gate = create_quantum_gate(GATE_TYPE_X, qubits, 1, NULL, 0);
    success = update_gate_parameters(x_gate, new_params, 1);
    assert(!success);
    printf("Update on non-parameterized gate test passed\n");
    destroy_quantum_gate(x_gate);
    
    printf("Time: %.3f ms\n", get_time_ms(start));
}

static void test_shift_parameters(void) {
    printf("\n=== Testing Parameter Shifts ===\n");
    clock_t start = clock();
    
    size_t qubits[] = {0};
    double params[] = {0.5};
    quantum_gate_t* rx_gate = create_quantum_gate(GATE_TYPE_RX, qubits, 1, params, 1);
    
    printf("\nRX Gate Matrix before shift (theta = 0.5):\n");
    print_matrix(rx_gate->matrix, 2);
    
    // Test valid shift
    bool success = shift_gate_parameters(rx_gate, 0, 0.1);
    assert(success);
    assert(fabs(rx_gate->parameters[0] - 0.6) < EPSILON);
    
    printf("\nRX Gate Matrix after shift (theta = 0.6):\n");
    print_matrix(rx_gate->matrix, 2);
    
    // Test invalid shifts
    success = shift_gate_parameters(rx_gate, 1, 0.1);
    assert(!success);
    printf("Invalid parameter index test passed\n");
    
    success = shift_gate_parameters(NULL, 0, 0.1);
    assert(!success);
    printf("NULL gate test passed\n");
    
    destroy_quantum_gate(rx_gate);
    
    // Test shift on non-parameterized gate
    quantum_gate_t* x_gate = create_quantum_gate(GATE_TYPE_X, qubits, 1, NULL, 0);
    success = shift_gate_parameters(x_gate, 0, 0.1);
    assert(!success);
    printf("Shift on non-parameterized gate test passed\n");
    destroy_quantum_gate(x_gate);
    
    printf("Time: %.3f ms\n", get_time_ms(start));
}

static void test_controlled_gates(void) {
    printf("\n=== Testing Controlled Gates ===\n");
    clock_t start = clock();
    
    size_t qubits[] = {0, 1};
    
    // Test CNOT gate
    quantum_gate_t* cnot_gate = create_quantum_gate(GATE_TYPE_CNOT, qubits, 2, NULL, 0);
    assert(cnot_gate != NULL);
    assert(cnot_gate->type == GATE_TYPE_CNOT);
    assert(cnot_gate->is_controlled == true);
    assert(cnot_gate->num_qubits == 2);
    
    printf("\nCNOT Gate Matrix:\n");
    print_matrix(cnot_gate->matrix, 4);
    destroy_quantum_gate(cnot_gate);
    
    // Test CZ gate
    quantum_gate_t* cz_gate = create_quantum_gate(GATE_TYPE_CZ, qubits, 2, NULL, 0);
    assert(cz_gate != NULL);
    assert(cz_gate->type == GATE_TYPE_CZ);
    assert(cz_gate->is_controlled == true);
    assert(cz_gate->num_qubits == 2);
    
    printf("\nCZ Gate Matrix:\n");
    print_matrix(cz_gate->matrix, 4);
    destroy_quantum_gate(cz_gate);
    
    // Test invalid qubit count
    quantum_gate_t* invalid_cnot = create_quantum_gate(GATE_TYPE_CNOT, qubits, 1, NULL, 0);
    assert(invalid_cnot == NULL);
    printf("Invalid CNOT qubit count test passed\n");
    
    quantum_gate_t* invalid_cz = create_quantum_gate(GATE_TYPE_CZ, qubits, 3, NULL, 0);
    assert(invalid_cz == NULL);
    printf("Invalid CZ qubit count test passed\n");
    
    printf("Time: %.3f ms\n", get_time_ms(start));
}

static void test_memory_cleanup(void) {
    printf("\n=== Testing Memory Cleanup ===\n");
    clock_t start = clock();
    
    // Test destroy on NULL gate
    destroy_quantum_gate(NULL);
    printf("NULL gate cleanup test passed\n");
    
    // Create gate with NULL fields
    quantum_gate_t* gate = malloc(sizeof(quantum_gate_t));
    gate->matrix = NULL;
    gate->target_qubits = NULL;
    gate->control_qubits = NULL;
    gate->parameters = NULL;
    destroy_quantum_gate(gate);
    printf("NULL fields cleanup test passed\n");
    
    printf("Time: %.3f ms\n", get_time_ms(start));
}

static void test_copy_gate(void) {
    printf("\n=== Testing Gate Copying ===\n");
    clock_t start = clock();
    
    size_t qubits[] = {0};
    double params[] = {0.5};
    quantum_gate_t* original = create_quantum_gate(GATE_TYPE_RX, qubits, 1, params, 1);
    
    printf("\nOriginal RX Gate Matrix (theta = 0.5):\n");
    print_matrix(original->matrix, 2);
    
    quantum_gate_t* copy = copy_quantum_gate(original);
    assert(copy != NULL);
    assert(copy->type == original->type);
    assert(copy->is_parameterized == original->is_parameterized);
    assert(copy->num_parameters == original->num_parameters);
    assert(fabs(copy->parameters[0] - original->parameters[0]) < EPSILON);
    
    // Modify copy and verify original is unchanged
    double new_params[] = {1.0};
    bool success = update_gate_parameters(copy, new_params, 1);
    assert(success);
    assert(fabs(copy->parameters[0] - 1.0) < EPSILON);
    assert(fabs(original->parameters[0] - 0.5) < EPSILON);
    
    printf("\nCopied RX Gate Matrix after update (theta = 1.0):\n");
    print_matrix(copy->matrix, 2);
    
    destroy_quantum_gate(original);
    destroy_quantum_gate(copy);
    
    printf("Time: %.3f ms\n", get_time_ms(start));
}

int main(void) {
    printf("Running quantum gate operations tests...\n");
    clock_t total_start = clock();
    
    test_create_fixed_gates();
    test_create_parameterized_gates();
    test_update_parameters();
    test_shift_parameters();
    test_controlled_gates();
    test_memory_cleanup();
    test_copy_gate();
    
    printf("\nAll quantum gate operations tests passed!\n");
    printf("Total time: %.3f ms\n", get_time_ms(total_start));
    return 0;
}
