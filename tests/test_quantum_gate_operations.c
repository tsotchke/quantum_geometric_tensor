#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include "quantum_geometric/core/quantum_gate_operations.h"
#include "quantum_geometric/core/quantum_types.h"
#include "quantum_geometric/core/numerical_backend.h"

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
    if (!i_gate) {
        printf("Failed to create Identity gate\n");
        return;
    }
    if (i_gate->type != GATE_TYPE_I || i_gate->is_parameterized || 
        i_gate->num_parameters != 0 || i_gate->parameters != NULL) {
        printf("Identity gate properties incorrect\n");
        destroy_quantum_gate(i_gate);
        return;
    }
    printf("\nIdentity Gate Matrix:\n");
    print_matrix(i_gate->matrix, 2);
    destroy_quantum_gate(i_gate);
    
    // Test X gate (NOT gate)
    quantum_gate_t* x_gate = create_quantum_gate(GATE_TYPE_X, qubits, 1, NULL, 0);
    if (!x_gate) {
        printf("Failed to create X gate\n");
        return;
    }
    if (x_gate->type != GATE_TYPE_X || x_gate->is_parameterized ||
        fabs(x_gate->matrix[0].real) >= EPSILON || fabs(x_gate->matrix[0].imag) >= EPSILON ||
        fabs(x_gate->matrix[1].real - 1.0f) >= EPSILON || fabs(x_gate->matrix[1].imag) >= EPSILON ||
        fabs(x_gate->matrix[2].real - 1.0f) >= EPSILON || fabs(x_gate->matrix[2].imag) >= EPSILON ||
        fabs(x_gate->matrix[3].real) >= EPSILON || fabs(x_gate->matrix[3].imag) >= EPSILON) {
        printf("X gate matrix incorrect\n");
        destroy_quantum_gate(x_gate);
        return;
    }
    printf("\nX (NOT) Gate Matrix:\n");
    print_matrix(x_gate->matrix, 2);
    destroy_quantum_gate(x_gate);
    
    // Test Y gate
    quantum_gate_t* y_gate = create_quantum_gate(GATE_TYPE_Y, qubits, 1, NULL, 0);
    if (!y_gate) {
        printf("Failed to create Y gate\n");
        return;
    }
    if (y_gate->type != GATE_TYPE_Y || y_gate->is_parameterized ||
        fabs(y_gate->matrix[0].real) >= EPSILON || fabs(y_gate->matrix[0].imag) >= EPSILON ||
        fabs(y_gate->matrix[1].real) >= EPSILON || fabs(y_gate->matrix[1].imag + 1.0f) >= EPSILON ||
        fabs(y_gate->matrix[2].real) >= EPSILON || fabs(y_gate->matrix[2].imag - 1.0f) >= EPSILON ||
        fabs(y_gate->matrix[3].real) >= EPSILON || fabs(y_gate->matrix[3].imag) >= EPSILON) {
        printf("Y gate matrix incorrect\n");
        destroy_quantum_gate(y_gate);
        return;
    }
    printf("\nY Gate Matrix:\n");
    print_matrix(y_gate->matrix, 2);
    destroy_quantum_gate(y_gate);
    
    // Test Z gate
    quantum_gate_t* z_gate = create_quantum_gate(GATE_TYPE_Z, qubits, 1, NULL, 0);
    if (!z_gate) {
        printf("Failed to create Z gate\n");
        return;
    }
    if (z_gate->type != GATE_TYPE_Z || z_gate->is_parameterized ||
        fabs(z_gate->matrix[0].real - 1.0f) >= EPSILON ||
        fabs(z_gate->matrix[1].real) >= EPSILON ||
        fabs(z_gate->matrix[2].real) >= EPSILON ||
        fabs(z_gate->matrix[3].real + 1.0f) >= EPSILON) {
        printf("Z gate matrix incorrect\n");
        destroy_quantum_gate(z_gate);
        return;
    }
    printf("\nZ Gate Matrix:\n");
    print_matrix(z_gate->matrix, 2);
    destroy_quantum_gate(z_gate);
    
    // Test H gate (Hadamard)
    quantum_gate_t* h_gate = create_quantum_gate(GATE_TYPE_H, qubits, 1, NULL, 0);
    if (!h_gate) {
        printf("Failed to create H gate\n");
        return;
    }
    float inv_sqrt2 = 1.0f / sqrtf(2.0f);
    if (h_gate->type != GATE_TYPE_H || h_gate->is_parameterized ||
        fabs(h_gate->matrix[0].real - inv_sqrt2) >= EPSILON ||
        fabs(h_gate->matrix[1].real - inv_sqrt2) >= EPSILON ||
        fabs(h_gate->matrix[2].real - inv_sqrt2) >= EPSILON ||
        fabs(h_gate->matrix[3].real + inv_sqrt2) >= EPSILON) {
        printf("H gate matrix incorrect\n");
        destroy_quantum_gate(h_gate);
        return;
    }
    printf("\nH (Hadamard) Gate Matrix:\n");
    print_matrix(h_gate->matrix, 2);
    destroy_quantum_gate(h_gate);
    
    // Test S gate (Phase gate)
    quantum_gate_t* s_gate = create_quantum_gate(GATE_TYPE_S, qubits, 1, NULL, 0);
    if (!s_gate) {
        printf("Failed to create S gate\n");
        return;
    }
    if (s_gate->type != GATE_TYPE_S || s_gate->is_parameterized ||
        fabs(s_gate->matrix[0].real - 1.0f) >= EPSILON ||
        fabs(s_gate->matrix[1].real) >= EPSILON ||
        fabs(s_gate->matrix[2].real) >= EPSILON ||
        fabs(s_gate->matrix[3].imag - 1.0f) >= EPSILON) {
        printf("S gate matrix incorrect\n");
        destroy_quantum_gate(s_gate);
        return;
    }
    printf("\nS (Phase) Gate Matrix:\n");
    print_matrix(s_gate->matrix, 2);
    destroy_quantum_gate(s_gate);
    
    // Test T gate (π/8 gate)
    quantum_gate_t* t_gate = create_quantum_gate(GATE_TYPE_T, qubits, 1, NULL, 0);
    if (!t_gate) {
        printf("Failed to create T gate\n");
        return;
    }
    if (t_gate->type != GATE_TYPE_T || t_gate->is_parameterized ||
        fabs(t_gate->matrix[0].real - 1.0f) >= EPSILON ||
        fabs(t_gate->matrix[1].real) >= EPSILON ||
        fabs(t_gate->matrix[2].real) >= EPSILON ||
        fabs(t_gate->matrix[3].real - cos(M_PI/4)) >= EPSILON ||
        fabs(t_gate->matrix[3].imag - sin(M_PI/4)) >= EPSILON) {
        printf("T gate matrix incorrect\n");
        destroy_quantum_gate(t_gate);
        return;
    }
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
    if (!rx_gate) {
        printf("Failed to create RX gate\n");
        return;
    }
    if (rx_gate->type != GATE_TYPE_RX || !rx_gate->is_parameterized ||
        rx_gate->num_parameters != 1 || !rx_gate->parameters ||
        fabs(rx_gate->parameters[0] - 0.5) >= EPSILON) {
        printf("RX gate properties incorrect\n");
        destroy_quantum_gate(rx_gate);
        return;
    }
    printf("\nRX Gate Matrix (theta = 0.5):\n");
    print_matrix(rx_gate->matrix, 2);
    destroy_quantum_gate(rx_gate);
    
    // Test RY gate
    quantum_gate_t* ry_gate = create_quantum_gate(GATE_TYPE_RY, qubits, 1, params, 1);
    if (!ry_gate) {
        printf("Failed to create RY gate\n");
        return;
    }
    if (ry_gate->type != GATE_TYPE_RY || !ry_gate->is_parameterized) {
        printf("RY gate properties incorrect\n");
        destroy_quantum_gate(ry_gate);
        return;
    }
    printf("\nRY Gate Matrix (theta = 0.5):\n");
    print_matrix(ry_gate->matrix, 2);
    destroy_quantum_gate(ry_gate);
    
    // Test RZ gate
    quantum_gate_t* rz_gate = create_quantum_gate(GATE_TYPE_RZ, qubits, 1, params, 1);
    if (!rz_gate) {
        printf("Failed to create RZ gate\n");
        return;
    }
    if (rz_gate->type != GATE_TYPE_RZ || !rz_gate->is_parameterized) {
        printf("RZ gate properties incorrect\n");
        destroy_quantum_gate(rz_gate);
        return;
    }
    printf("\nRZ Gate Matrix (theta = 0.5):\n");
    print_matrix(rz_gate->matrix, 2);
    destroy_quantum_gate(rz_gate);
    
    // Test invalid parameter count
    quantum_gate_t* invalid_gate = create_quantum_gate(GATE_TYPE_RX, qubits, 1, NULL, 0);
    if (invalid_gate) {
        printf("Invalid parameter count test failed\n");
        destroy_quantum_gate(invalid_gate);
        return;
    }
    printf("Invalid parameter count test passed\n");
    
    printf("Time: %.3f ms\n", get_time_ms(start));
}

static void test_update_parameters(void) {
    printf("\n=== Testing Parameter Updates ===\n");
    clock_t start = clock();
    
    size_t qubits[] = {0};
    double params[] = {0.5};
    quantum_gate_t* rx_gate = create_quantum_gate(GATE_TYPE_RX, qubits, 1, params, 1);
    if (!rx_gate) {
        printf("Failed to create RX gate\n");
        return;
    }
    
    printf("\nRX Gate Matrix before update (theta = 0.5):\n");
    print_matrix(rx_gate->matrix, 2);
    
    // Test valid update
    double new_params[] = {1.0};
    numerical_error_t error = update_gate_parameters(rx_gate, new_params, 1);
    if (error != NUMERICAL_SUCCESS) {
        printf("Failed to update gate parameters: %s\n", 
               get_numerical_error_string(error));
        destroy_quantum_gate(rx_gate);
        return;
    }
    
    if (fabs(rx_gate->parameters[0] - 1.0) >= EPSILON) {
        printf("Parameter update did not set correct value\n");
        destroy_quantum_gate(rx_gate);
        return;
    }
    
    printf("\nRX Gate Matrix after update (theta = 1.0):\n");
    print_matrix(rx_gate->matrix, 2);
    
    // Test invalid updates
    error = update_gate_parameters(rx_gate, NULL, 1);
    if (error != NUMERICAL_ERROR_INVALID_ARGUMENT) {
        printf("NULL parameters test failed\n");
        destroy_quantum_gate(rx_gate);
        return;
    }
    printf("NULL parameters test passed\n");
    
    error = update_gate_parameters(rx_gate, new_params, 2);
    if (error != NUMERICAL_ERROR_INVALID_ARGUMENT) {
        printf("Invalid parameter count test failed\n");
        destroy_quantum_gate(rx_gate);
        return;
    }
    printf("Invalid parameter count test passed\n");
    
    destroy_quantum_gate(rx_gate);
    
    // Test update on non-parameterized gate
    quantum_gate_t* x_gate = create_quantum_gate(GATE_TYPE_X, qubits, 1, NULL, 0);
    if (!x_gate) {
        printf("Failed to create X gate\n");
        return;
    }
    
    error = update_gate_parameters(x_gate, new_params, 1);
    if (error != NUMERICAL_ERROR_INVALID_STATE) {
        printf("Update on non-parameterized gate test failed\n");
        destroy_quantum_gate(x_gate);
        return;
    }
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
    if (!rx_gate) {
        printf("Failed to create RX gate\n");
        return;
    }
    
    printf("\nRX Gate Matrix before shift (theta = 0.5):\n");
    print_matrix(rx_gate->matrix, 2);
    
    // Test valid shift
    numerical_error_t error = shift_gate_parameters(rx_gate, 0, 0.1);
    if (error != NUMERICAL_SUCCESS) {
        printf("Failed to shift gate parameters: %s\n",
               get_numerical_error_string(error));
        destroy_quantum_gate(rx_gate);
        return;
    }
    
    if (fabs(rx_gate->parameters[0] - 0.6) >= EPSILON) {
        printf("Parameter shift did not set correct value\n");
        destroy_quantum_gate(rx_gate);
        return;
    }
    
    printf("\nRX Gate Matrix after shift (theta = 0.6):\n");
    print_matrix(rx_gate->matrix, 2);
    
    // Test invalid shifts
    error = shift_gate_parameters(rx_gate, 1, 0.1);
    if (error != NUMERICAL_ERROR_INVALID_ARGUMENT) {
        printf("Invalid parameter index test failed\n");
        destroy_quantum_gate(rx_gate);
        return;
    }
    printf("Invalid parameter index test passed\n");
    
    error = shift_gate_parameters(NULL, 0, 0.1);
    if (error != NUMERICAL_ERROR_INVALID_ARGUMENT) {
        printf("NULL gate test failed\n");
        destroy_quantum_gate(rx_gate);
        return;
    }
    printf("NULL gate test passed\n");
    
    destroy_quantum_gate(rx_gate);
    
    // Test shift on non-parameterized gate
    quantum_gate_t* x_gate = create_quantum_gate(GATE_TYPE_X, qubits, 1, NULL, 0);
    if (!x_gate) {
        printf("Failed to create X gate\n");
        return;
    }
    
    error = shift_gate_parameters(x_gate, 0, 0.1);
    if (error != NUMERICAL_ERROR_INVALID_STATE) {
        printf("Shift on non-parameterized gate test failed\n");
        destroy_quantum_gate(x_gate);
        return;
    }
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
    if (!cnot_gate) {
        printf("Failed to create CNOT gate\n");
        return;
    }
    if (cnot_gate->type != GATE_TYPE_CNOT || !cnot_gate->is_controlled ||
        cnot_gate->num_qubits != 2) {
        printf("CNOT gate properties incorrect\n");
        destroy_quantum_gate(cnot_gate);
        return;
    }
    printf("\nCNOT Gate Matrix:\n");
    print_matrix(cnot_gate->matrix, 4);
    destroy_quantum_gate(cnot_gate);
    
    // Test CZ gate
    quantum_gate_t* cz_gate = create_quantum_gate(GATE_TYPE_CZ, qubits, 2, NULL, 0);
    if (!cz_gate) {
        printf("Failed to create CZ gate\n");
        return;
    }
    if (cz_gate->type != GATE_TYPE_CZ || !cz_gate->is_controlled ||
        cz_gate->num_qubits != 2) {
        printf("CZ gate properties incorrect\n");
        destroy_quantum_gate(cz_gate);
        return;
    }
    printf("\nCZ Gate Matrix:\n");
    print_matrix(cz_gate->matrix, 4);
    destroy_quantum_gate(cz_gate);
    
    // Test invalid qubit count
    quantum_gate_t* invalid_cnot = create_quantum_gate(GATE_TYPE_CNOT, qubits, 1, NULL, 0);
    if (invalid_cnot) {
        printf("Invalid CNOT qubit count test failed\n");
        destroy_quantum_gate(invalid_cnot);
        return;
    }
    printf("Invalid CNOT qubit count test passed\n");
    
    quantum_gate_t* invalid_cz = create_quantum_gate(GATE_TYPE_CZ, qubits, 3, NULL, 0);
    if (invalid_cz) {
        printf("Invalid CZ qubit count test failed\n");
        destroy_quantum_gate(invalid_cz);
        return;
    }
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
    if (!copy) {
        printf("Failed to copy RX gate\n");
        destroy_quantum_gate(original);
        return;
    }
    if (copy->type != original->type || 
        copy->is_parameterized != original->is_parameterized ||
        copy->num_parameters != original->num_parameters ||
        fabs(copy->parameters[0] - original->parameters[0]) >= EPSILON) {
        printf("Copied gate properties do not match original\n");
        destroy_quantum_gate(original);
        destroy_quantum_gate(copy);
        return;
    }
    
    // Modify copy and verify original is unchanged
    double new_params[] = {1.0};
    numerical_error_t error = update_gate_parameters(copy, new_params, 1);
    if (error != NUMERICAL_SUCCESS) {
        printf("Failed to update copied gate parameters: %s\n",
               get_numerical_error_string(error));
        destroy_quantum_gate(original);
        destroy_quantum_gate(copy);
        return;
    }
    
    if (fabs(copy->parameters[0] - 1.0) >= EPSILON) {
        printf("Parameter update on copy did not set correct value\n");
        destroy_quantum_gate(original);
        destroy_quantum_gate(copy);
        return;
    }
    
    if (fabs(original->parameters[0] - 0.5) >= EPSILON) {
        printf("Original parameters were modified\n");
        destroy_quantum_gate(original);
        destroy_quantum_gate(copy);
        return;
    }
    
    printf("\nCopied RX Gate Matrix after update (theta = 1.0):\n");
    print_matrix(copy->matrix, 2);
    
    destroy_quantum_gate(original);
    destroy_quantum_gate(copy);
    
    printf("Time: %.3f ms\n", get_time_ms(start));
}

int main(void) {
    printf("Running quantum gate operations tests...\n");
    clock_t total_start = clock();

    // Initialize numerical backend
    numerical_config_t config = {
        .type = NUMERICAL_BACKEND_CPU,
        .max_threads = 1,
        .use_fma = false,
        .use_avx = false,
        .use_neon = false,
        .cache_size = 0,
        .backend_specific = NULL
    };
    
    numerical_error_t error = initialize_numerical_backend(&config);
    if (error != NUMERICAL_SUCCESS) {
        printf("Failed to initialize numerical backend: %s\n",
               get_numerical_error_string(error));
        return 1;
    }
    
    test_create_fixed_gates();
    test_create_parameterized_gates();
    test_update_parameters();
    test_shift_parameters();
    test_controlled_gates();
    test_memory_cleanup();
    test_copy_gate();
    
    // Cleanup
    shutdown_numerical_backend();
    
    printf("\nAll quantum gate operations tests passed!\n");
    printf("Total time: %.3f ms\n", get_time_ms(total_start));
    return 0;
}
