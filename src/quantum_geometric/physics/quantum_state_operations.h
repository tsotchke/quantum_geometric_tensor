#ifndef QUANTUM_STATE_OPERATIONS_H
#define QUANTUM_STATE_OPERATIONS_H

#include <complex.h>
#include <stddef.h>

// Forward declarations
struct HierarchicalMatrix;
struct QuantumGeometricTensor;

// Core quantum state operations
void quantum_state_multiply(double complex* dst,
                          const double complex* a,
                          const double complex* b,
                          size_t rows, size_t cols, size_t inner);

// Hierarchical matrix quantum operations
void update_hmatrix_quantum_state(struct HierarchicalMatrix* mat);
void cleanup_hmatrix_quantum_state(struct HierarchicalMatrix* mat);

// QGT operations
struct QuantumGeometricTensor* compute_qgt(const double complex* data,
                                         size_t size);

// Helper functions
static inline size_t next_power_of_two(size_t n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    n++;
    return n;
}

static inline size_t count_qubits(size_t n) {
    size_t count = 0;
    while (n > 1) {
        n >>= 1;
        count++;
    }
    return count;
}

#endif // QUANTUM_STATE_OPERATIONS_H
