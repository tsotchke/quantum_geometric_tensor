#include "quantum_geometric/core/quantum_geometric_operations.h"
#include "quantum_geometric/hardware/quantum_geometric_gpu.h"
#include "quantum_geometric/physics/quantum_state_operations.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>

// Test parameters
#define MIN_QUBITS 4
#define MAX_QUBITS 20
#define NUM_TRIALS 5
#define EPSILON 1e-10

// Performance measurement
typedef struct {
    size_t num_qubits;
    double encode_time;
    double multiply_time;
    double svd_time;
    double compress_time;
} PerformanceMetrics;

// Create test matrix
static HierarchicalMatrix* create_test_matrix(size_t size) {
    HierarchicalMatrix* mat = hmatrix_create(size, size, 1e-6);
    if (!mat) return NULL;
    
    // Initialize with random values
    if (mat->is_leaf) {
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < size; i++) {
            for (size_t j = 0; j < size; j++) {
                double re = (double)rand() / RAND_MAX - 0.5;
                double im = (double)rand() / RAND_MAX - 0.5;
                mat->data[i * size + j] = re + im * I;
            }
        }
    }
    
    return mat;
}

// Measure execution time
static double measure_time(void (*func)(void*), void* arg) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    func(arg);
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    return (end.tv_sec - start.tv_sec) +
           (end.tv_nsec - start.tv_nsec) * 1e-9;
}

// Test quantum encoding/decoding
static void test_quantum_encoding(void) {
    printf("Testing quantum encoding/decoding...\n");
    
    PerformanceMetrics metrics[MAX_QUBITS - MIN_QUBITS + 1] = {0};
    
    for (size_t n = MIN_QUBITS; n <= MAX_QUBITS; n++) {
        size_t size = 1ULL << n;
        printf("  Testing size 2^%zu x 2^%zu...\n", n, n);
        
        // Create test matrix
        HierarchicalMatrix* mat = create_test_matrix(size);
        assert(mat != NULL);
        
        // Initialize quantum state
        QuantumState* state = init_quantum_state(2 * n);
        assert(state != NULL);
        
        // Measure encoding time
        double total_encode = 0.0;
        double total_decode = 0.0;
        
        for (int trial = 0; trial < NUM_TRIALS; trial++) {
            // Time encoding
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            quantum_encode_matrix(state, mat);
            clock_gettime(CLOCK_MONOTONIC, &end);
            total_encode += (end.tv_sec - start.tv_sec) +
                          (end.tv_nsec - start.tv_nsec) * 1e-9;
            
            // Create output matrix
            HierarchicalMatrix* out = hmatrix_create(size, size, 1e-6);
            assert(out != NULL);
            
            // Time decoding
            clock_gettime(CLOCK_MONOTONIC, &start);
            quantum_decode_matrix(out, state);
            clock_gettime(CLOCK_MONOTONIC, &end);
            total_decode += (end.tv_sec - start.tv_sec) +
                          (end.tv_nsec - start.tv_nsec) * 1e-9;
            
            // Verify correctness
            if (mat->is_leaf && out->is_leaf) {
                for (size_t i = 0; i < size * size; i++) {
                    assert(cabs(mat->data[i] - out->data[i]) < EPSILON);
                }
            }
            
            hmatrix_destroy(out);
        }
        
        // Record metrics
        metrics[n - MIN_QUBITS].num_qubits = n;
        metrics[n - MIN_QUBITS].encode_time = total_encode / NUM_TRIALS;
        
        // Clean up
        free(state->amplitudes);
        free(state);
        hmatrix_destroy(mat);
    }
    
    // Print results
    printf("\nEncoding Performance Results:\n");
    printf("Qubits\tTime (s)\tlog2(Time)\n");
    for (size_t i = 0; i < MAX_QUBITS - MIN_QUBITS + 1; i++) {
        printf("%zu\t%.6f\t%.6f\n",
               metrics[i].num_qubits,
               metrics[i].encode_time,
               log2(metrics[i].encode_time));
    }
}

// Test quantum multiplication
static void test_quantum_multiplication(void) {
    printf("\nTesting quantum multiplication...\n");
    
    PerformanceMetrics metrics[MAX_QUBITS - MIN_QUBITS + 1] = {0};
    
    for (size_t n = MIN_QUBITS; n <= MAX_QUBITS; n++) {
        size_t size = 1ULL << n;
        printf("  Testing size 2^%zu x 2^%zu...\n", n, n);
        
        // Create test matrices
        HierarchicalMatrix* a = create_test_matrix(size);
        HierarchicalMatrix* b = create_test_matrix(size);
        assert(a != NULL && b != NULL);
        
        // Initialize quantum states
        QuantumState* state_a = init_quantum_state(2 * n);
        QuantumState* state_b = init_quantum_state(2 * n);
        assert(state_a != NULL && state_b != NULL);
        
        // Encode matrices
        quantum_encode_matrix(state_a, a);
        quantum_encode_matrix(state_b, b);
        
        // Measure multiplication time
        double total_time = 0.0;
        
        for (int trial = 0; trial < NUM_TRIALS; trial++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
            quantum_circuit_multiply(state_a, state_b);
            
            clock_gettime(CLOCK_MONOTONIC, &end);
            total_time += (end.tv_sec - start.tv_sec) +
                         (end.tv_nsec - start.tv_nsec) * 1e-9;
        }
        
        // Record metrics
        metrics[n - MIN_QUBITS].num_qubits = n;
        metrics[n - MIN_QUBITS].multiply_time = total_time / NUM_TRIALS;
        
        // Clean up
        free(state_a->amplitudes);
        free(state_b->amplitudes);
        free(state_a);
        free(state_b);
        hmatrix_destroy(a);
        hmatrix_destroy(b);
    }
    
    // Print results
    printf("\nMultiplication Performance Results:\n");
    printf("Qubits\tTime (s)\tlog2(Time)\n");
    for (size_t i = 0; i < MAX_QUBITS - MIN_QUBITS + 1; i++) {
        printf("%zu\t%.6f\t%.6f\n",
               metrics[i].num_qubits,
               metrics[i].multiply_time,
               log2(metrics[i].multiply_time));
    }
}

// Test quantum compression
static void test_quantum_compression(void) {
    printf("\nTesting quantum compression...\n");
    
    PerformanceMetrics metrics[MAX_QUBITS - MIN_QUBITS + 1] = {0};
    
    for (size_t n = MIN_QUBITS; n <= MAX_QUBITS; n++) {
        size_t size = 1ULL << n;
        printf("  Testing size 2^%zu x 2^%zu...\n", n, n);
        
        // Create test matrix
        HierarchicalMatrix* mat = create_test_matrix(size);
        assert(mat != NULL);
        
        // Initialize quantum state
        QuantumState* state = init_quantum_state(2 * n);
        assert(state != NULL);
        
        // Encode matrix
        quantum_encode_matrix(state, mat);
        
        // Measure compression time
        double total_time = 0.0;
        
        for (int trial = 0; trial < NUM_TRIALS; trial++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
            quantum_compress_circuit(state, n);  // Compress to half size
            
            clock_gettime(CLOCK_MONOTONIC, &end);
            total_time += (end.tv_sec - start.tv_sec) +
                         (end.tv_nsec - start.tv_nsec) * 1e-9;
        }
        
        // Record metrics
        metrics[n - MIN_QUBITS].num_qubits = n;
        metrics[n - MIN_QUBITS].compress_time = total_time / NUM_TRIALS;
        
        // Clean up
        free(state->amplitudes);
        free(state);
        hmatrix_destroy(mat);
    }
    
    // Print results
    printf("\nCompression Performance Results:\n");
    printf("Qubits\tTime (s)\tlog2(Time)\n");
    for (size_t i = 0; i < MAX_QUBITS - MIN_QUBITS + 1; i++) {
        printf("%zu\t%.6f\t%.6f\n",
               metrics[i].num_qubits,
               metrics[i].compress_time,
               log2(metrics[i].compress_time));
    }
}

int main(void) {
    srand(time(NULL));
    
    printf("Starting quantum circuit tests...\n");
    
    // Run tests
    test_quantum_encoding();
    test_quantum_multiplication();
    test_quantum_compression();
    
    printf("\nAll quantum circuit tests passed!\n");
    return 0;
}
