/**
 * @file test_quantum_gpu.c
 * @brief GPU tests for quantum operations
 *
 * Tests basic GPU functionality using the portable abstraction layer.
 */

#include "quantum_geometric/core/quantum_geometric_gpu.h"
#include "quantum_geometric/core/quantum_complex.h"
#include "quantum_geometric/core/error_codes.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

// Test parameters
#define NUM_QUBITS 4
#define STATE_SIZE (1ULL << NUM_QUBITS)
#define EPSILON 1e-6f

// Test helpers
static ComplexFloat* create_test_state(size_t size) {
    ComplexFloat* state = (ComplexFloat*)malloc(size * sizeof(ComplexFloat));
    if (!state) return NULL;

    // Initialize to |0> state
    state[0].real = 1.0f;
    state[0].imag = 0.0f;
    for (size_t i = 1; i < size; i++) {
        state[i].real = 0.0f;
        state[i].imag = 0.0f;
    }

    return state;
}

static float state_norm(const ComplexFloat* state, size_t size) {
    float norm_sq = 0.0f;
    for (size_t i = 0; i < size; i++) {
        norm_sq += state[i].real * state[i].real + state[i].imag * state[i].imag;
    }
    return sqrtf(norm_sq);
}

static bool compare_floats(float a, float b, float eps) {
    return fabsf(a - b) < eps;
}

// Test GPU initialization and device info
static void test_gpu_init(void) {
    printf("Testing GPU initialization...\n");

    int result = qg_gpu_init();
    if (result != QG_GPU_SUCCESS) {
        printf("  SKIP: GPU not available (result=%d)\n", result);
        return;
    }

    // Get device count
    int device_count = 0;
    result = qg_gpu_get_device_count(&device_count);
    assert(result == QG_GPU_SUCCESS);
    printf("  Found %d GPU device(s)\n", device_count);

    // Get device info for first device
    if (device_count > 0) {
        gpu_device_info_t info;
        result = qg_gpu_get_device_info(0, &info);
        assert(result == QG_GPU_SUCCESS);

        printf("  Device 0: %s\n", info.name);
        printf("    Total memory: %.2f MB\n", info.total_memory / (1024.0 * 1024.0));
        printf("    Compute units: %d\n", info.compute_units);
        printf("    Max threads/block: %d\n", info.max_threads_per_block);
    }

    qg_gpu_cleanup();
    printf("  GPU initialization test passed\n");
}

// Test GPU memory allocation
static void test_gpu_memory(void) {
    printf("Testing GPU memory operations...\n");

    int result = qg_gpu_init();
    if (result != QG_GPU_SUCCESS) {
        printf("  SKIP: GPU not available\n");
        return;
    }

    // Test buffer allocation
    gpu_buffer_t buffer;
    size_t test_size = 1024 * sizeof(ComplexFloat);

    result = qg_gpu_allocate(&buffer, test_size);
    if (result != QG_GPU_SUCCESS) {
        printf("  SKIP: GPU allocation failed\n");
        qg_gpu_cleanup();
        return;
    }
    assert(buffer.size == test_size);

    // Create test data
    ComplexFloat* host_data = (ComplexFloat*)malloc(1024 * sizeof(ComplexFloat));
    assert(host_data != NULL);

    for (size_t i = 0; i < 1024; i++) {
        host_data[i].real = (float)i;
        host_data[i].imag = (float)(i * 2);
    }

    // Copy to device
    result = qg_gpu_memcpy_to_device(&buffer, host_data, test_size);
    assert(result == QG_GPU_SUCCESS);

    // Clear host data
    memset(host_data, 0, test_size);

    // Copy back from device
    result = qg_gpu_memcpy_to_host(host_data, &buffer, test_size);
    assert(result == QG_GPU_SUCCESS);

    // Verify data
    bool data_ok = true;
    for (size_t i = 0; i < 1024 && data_ok; i++) {
        if (!compare_floats(host_data[i].real, (float)i, EPSILON) ||
            !compare_floats(host_data[i].imag, (float)(i * 2), EPSILON)) {
            data_ok = false;
        }
    }
    assert(data_ok);

    // Free resources
    result = qg_gpu_free(&buffer);
    assert(result == QG_GPU_SUCCESS);

    free(host_data);
    qg_gpu_cleanup();

    printf("  GPU memory test passed\n");
}

// Test GPU streams
static void test_gpu_streams(void) {
    printf("Testing GPU stream operations...\n");

    int result = qg_gpu_init();
    if (result != QG_GPU_SUCCESS) {
        printf("  SKIP: GPU not available\n");
        return;
    }

    // Create multiple streams
    int stream1 = -1, stream2 = -1;

    result = qg_gpu_create_stream(&stream1);
    if (result != QG_GPU_SUCCESS) {
        printf("  SKIP: Stream creation not supported\n");
        qg_gpu_cleanup();
        return;
    }
    assert(stream1 >= 0);

    result = qg_gpu_create_stream(&stream2);
    assert(result == QG_GPU_SUCCESS);
    assert(stream2 >= 0);
    assert(stream1 != stream2);

    printf("  Created streams %d and %d\n", stream1, stream2);

    // Synchronize streams
    result = qg_gpu_synchronize_stream(stream1);
    assert(result == QG_GPU_SUCCESS);

    result = qg_gpu_synchronize_stream(stream2);
    assert(result == QG_GPU_SUCCESS);

    // Global synchronize
    result = qg_gpu_synchronize();
    assert(result == QG_GPU_SUCCESS);

    // Destroy streams
    result = qg_gpu_destroy_stream(stream1);
    assert(result == QG_GPU_SUCCESS);

    result = qg_gpu_destroy_stream(stream2);
    assert(result == QG_GPU_SUCCESS);

    qg_gpu_cleanup();
    printf("  GPU stream test passed\n");
}

// Test quantum state operations on GPU
static void test_quantum_state_gpu(void) {
    printf("Testing quantum state operations on GPU...\n");

    int result = qg_gpu_init();
    if (result != QG_GPU_SUCCESS) {
        printf("  SKIP: GPU not available\n");
        return;
    }

    // Create test state |0>
    ComplexFloat* state = create_test_state(STATE_SIZE);
    assert(state != NULL);

    // Verify initial state is normalized
    float norm = state_norm(state, STATE_SIZE);
    assert(compare_floats(norm, 1.0f, EPSILON));

    // Allocate GPU buffer
    gpu_buffer_t gpu_state;
    size_t state_bytes = STATE_SIZE * sizeof(ComplexFloat);

    result = qg_gpu_allocate(&gpu_state, state_bytes);
    if (result != QG_GPU_SUCCESS) {
        printf("  SKIP: GPU allocation failed\n");
        free(state);
        qg_gpu_cleanup();
        return;
    }

    // Upload state to GPU
    result = qg_gpu_memcpy_to_device(&gpu_state, state, state_bytes);
    assert(result == QG_GPU_SUCCESS);

    // Download state from GPU
    ComplexFloat* result_state = (ComplexFloat*)malloc(state_bytes);
    assert(result_state != NULL);

    result = qg_gpu_memcpy_to_host(result_state, &gpu_state, state_bytes);
    assert(result == QG_GPU_SUCCESS);

    // Verify round-trip
    bool states_match = true;
    for (size_t i = 0; i < STATE_SIZE && states_match; i++) {
        if (!compare_floats(state[i].real, result_state[i].real, EPSILON) ||
            !compare_floats(state[i].imag, result_state[i].imag, EPSILON)) {
            states_match = false;
        }
    }
    assert(states_match);

    // Verify normalization preserved
    norm = state_norm(result_state, STATE_SIZE);
    assert(compare_floats(norm, 1.0f, EPSILON));

    // Cleanup
    qg_gpu_free(&gpu_state);
    free(state);
    free(result_state);
    qg_gpu_cleanup();

    printf("  Quantum state GPU test passed\n");
}

// Test Hadamard-like transformation (CPU reference)
static void apply_hadamard_cpu(ComplexFloat* state, size_t qubit, size_t num_qubits) {
    size_t state_size = 1ULL << num_qubits;
    size_t mask = 1ULL << qubit;
    float inv_sqrt2 = 1.0f / sqrtf(2.0f);

    for (size_t i = 0; i < state_size; i++) {
        if ((i & mask) == 0) {
            size_t j = i | mask;

            ComplexFloat a = state[i];
            ComplexFloat b = state[j];

            // H|0> = (|0> + |1>)/sqrt(2)
            // H|1> = (|0> - |1>)/sqrt(2)
            state[i].real = inv_sqrt2 * (a.real + b.real);
            state[i].imag = inv_sqrt2 * (a.imag + b.imag);
            state[j].real = inv_sqrt2 * (a.real - b.real);
            state[j].imag = inv_sqrt2 * (a.imag - b.imag);
        }
    }
}

// Test gate operations
static void test_gate_operations(void) {
    printf("Testing gate operations...\n");

    // Create test state |0>
    ComplexFloat* state = create_test_state(STATE_SIZE);
    assert(state != NULL);

    // Apply Hadamard on qubit 0 (CPU reference)
    apply_hadamard_cpu(state, 0, NUM_QUBITS);

    // After H|0>, state should be (|0> + |1>)/sqrt(2)
    // In computational basis: state[0] = state[1] = 1/sqrt(2)
    float expected = 1.0f / sqrtf(2.0f);
    assert(compare_floats(state[0].real, expected, EPSILON));
    assert(compare_floats(state[0].imag, 0.0f, EPSILON));
    assert(compare_floats(state[1].real, expected, EPSILON));
    assert(compare_floats(state[1].imag, 0.0f, EPSILON));

    // Verify normalization preserved
    float norm = state_norm(state, STATE_SIZE);
    assert(compare_floats(norm, 1.0f, EPSILON));

    // Apply Hadamard again - should return to |0>
    apply_hadamard_cpu(state, 0, NUM_QUBITS);

    assert(compare_floats(state[0].real, 1.0f, EPSILON));
    assert(compare_floats(state[0].imag, 0.0f, EPSILON));
    assert(compare_floats(state[1].real, 0.0f, EPSILON));
    assert(compare_floats(state[1].imag, 0.0f, EPSILON));

    free(state);
    printf("  Gate operations test passed\n");
}

// Test error handling
static void test_error_handling(void) {
    printf("Testing error handling...\n");

    // Test operations before init
    gpu_buffer_t buffer;
    int result = qg_gpu_allocate(&buffer, 1024);
    // Should fail or return error since not initialized
    // (behavior depends on implementation)

    // Initialize
    result = qg_gpu_init();
    if (result != QG_GPU_SUCCESS) {
        printf("  SKIP: GPU not available\n");
        return;
    }

    // Test invalid parameters
    result = qg_gpu_allocate(NULL, 1024);
    assert(result != QG_GPU_SUCCESS);

    result = qg_gpu_allocate(&buffer, 0);
    assert(result != QG_GPU_SUCCESS);

    // Test invalid stream operations
    result = qg_gpu_synchronize_stream(-1);
    assert(result != QG_GPU_SUCCESS);

    result = qg_gpu_synchronize_stream(9999);
    assert(result != QG_GPU_SUCCESS);

    result = qg_gpu_destroy_stream(-1);
    assert(result != QG_GPU_SUCCESS);

    // Get error string
    const char* error_str = qg_gpu_get_error_string(QG_GPU_ERROR_OUT_OF_MEMORY);
    assert(error_str != NULL);
    printf("  Error string example: %s\n", error_str);

    qg_gpu_cleanup();
    printf("  Error handling test passed\n");
}

// Test pinned memory
static void test_pinned_memory(void) {
    printf("Testing pinned memory...\n");

    int result = qg_gpu_init();
    if (result != QG_GPU_SUCCESS) {
        printf("  SKIP: GPU not available\n");
        return;
    }

    gpu_buffer_t pinned_buffer;
    result = qg_gpu_allocate_pinned(&pinned_buffer, 4096);
    if (result != QG_GPU_SUCCESS) {
        printf("  SKIP: Pinned memory not supported\n");
        qg_gpu_cleanup();
        return;
    }

    assert(pinned_buffer.is_pinned == true);
    assert(pinned_buffer.size == 4096);

    result = qg_gpu_free(&pinned_buffer);
    assert(result == QG_GPU_SUCCESS);

    qg_gpu_cleanup();
    printf("  Pinned memory test passed\n");
}

int main(void) {
    printf("=== Quantum GPU Tests ===\n\n");

    test_gpu_init();
    printf("\n");

    test_gpu_memory();
    printf("\n");

    test_gpu_streams();
    printf("\n");

    test_quantum_state_gpu();
    printf("\n");

    test_gate_operations();
    printf("\n");

    test_error_handling();
    printf("\n");

    test_pinned_memory();
    printf("\n");

    printf("=== All Quantum GPU Tests Completed ===\n");
    return 0;
}
