#ifndef TEST_HELPERS_H
#define TEST_HELPERS_H

#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_state_types.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#ifdef __APPLE__
#include <mach/mach.h>
#endif

// Test assertion macro
#define TEST_ASSERT(condition, message) do { \
    if (!(condition)) { \
        fprintf(stderr, "ASSERTION FAILED: %s\n  at %s:%d\n", \
                message, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// Get current time in seconds (high resolution)
static inline double get_current_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

// Get peak memory usage in bytes
static inline size_t get_peak_memory_usage(void) {
#ifdef __APPLE__
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &count) == KERN_SUCCESS) {
        return info.resident_size_max;
    }
    return 0;
#else
    // Linux: read from /proc/self/status
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return 0;

    char line[256];
    size_t peak_mem = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmPeak:", 7) == 0) {
            sscanf(line, "VmPeak: %zu", &peak_mem);
            peak_mem *= 1024;  // Convert from kB to bytes
            break;
        }
    }
    fclose(f);
    return peak_mem;
#endif
}

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
