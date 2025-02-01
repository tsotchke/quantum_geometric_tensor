#include "quantum_geometric/core/quantum_circuit_creation.h"
#include "quantum_geometric/core/quantum_geometric_constants.h"
#include "quantum_geometric/core/quantum_phase_estimation.h"

quantum_circuit_t* quantum_create_inversion_circuit(size_t num_qubits, int flags) {
    quantum_circuit_t* circuit = quantum_circuit_create(num_qubits);
    if (!circuit) return NULL;
    
    // Configure circuit based on flags
    if (flags & QUANTUM_OPTIMIZE_AGGRESSIVE) {
        circuit->optimization_level = 2;
    }
    
    // Add quantum Fourier transform gates
    for (size_t i = 0; i < num_qubits; i++) {
        quantum_circuit_add_hadamard(circuit, i);
        for (size_t j = i + 1; j < num_qubits; j++) {
            quantum_circuit_add_controlled_phase(circuit, i, j, M_PI / (1 << (j - i)));
        }
    }
    
    return circuit;
}

quantum_circuit_t* quantum_create_gradient_circuit(size_t num_qubits, int flags) {
    quantum_circuit_t* circuit = quantum_circuit_create(num_qubits);
    if (!circuit) return NULL;
    
    // Configure optimization
    if (flags & QUANTUM_OPTIMIZE_AGGRESSIVE) {
        circuit->optimization_level = 2;
    }
    
    // Add gradient estimation gates
    for (size_t i = 0; i < num_qubits; i++) {
        quantum_circuit_add_hadamard(circuit, i);
    }
    
    // Add controlled operations
    for (size_t i = 0; i < num_qubits - 1; i++) {
        quantum_circuit_add_controlled_not(circuit, i, i + 1);
    }
    
    return circuit;
}

quantum_circuit_t* quantum_create_hessian_circuit(size_t num_qubits, int flags) {
    quantum_circuit_t* circuit = quantum_circuit_create(num_qubits);
    if (!circuit) return NULL;
    
    // Configure optimization
    if (flags & QUANTUM_OPTIMIZE_AGGRESSIVE) {
        circuit->optimization_level = 2;
    }
    
    // Add initial superposition
    for (size_t i = 0; i < num_qubits; i++) {
        quantum_circuit_add_hadamard(circuit, i);
    }
    
    // Add controlled phase operations
    for (size_t i = 0; i < num_qubits; i++) {
        for (size_t j = i + 1; j < num_qubits; j++) {
            quantum_circuit_add_controlled_phase(circuit, i, j, M_PI / 2);
        }
    }
    
    return circuit;
}

void quantum_compute_gradient(quantum_register_t* reg_state,
                            quantum_register_t* reg_observable,
                            quantum_register_t* reg_gradient,
                            quantum_system_t* system,
                            quantum_circuit_t* circuit,
                            const quantum_phase_config_t* config) {
    // Initialize gradient computation
    for (size_t i = 0; i < reg_gradient->size; i++) {
        reg_gradient->amplitudes[i] = 0;
    }
    
    // Apply quantum phase estimation
    quantum_phase_estimation_optimized(reg_state, system, circuit, config);
    
    // Extract gradient information
    for (size_t i = 0; i < reg_gradient->size; i++) {
        reg_gradient->amplitudes[i] = reg_state->amplitudes[i];
    }
}

void quantum_compute_hessian_hierarchical(quantum_register_t* reg_state,
                                        quantum_register_t* reg_observable,
                                        quantum_register_t* reg_gradient,
                                        quantum_register_t* reg_hessian,
                                        quantum_system_t* system,
                                        quantum_circuit_t* circuit,
                                        const quantum_phase_config_t* config) {
    // Initialize Hessian computation
    for (size_t i = 0; i < reg_hessian->size; i++) {
        reg_hessian->amplitudes[i] = 0;
    }
    
    // Apply quantum phase estimation
    quantum_phase_estimation_optimized(reg_state, system, circuit, config);
    
    // Extract Hessian information
    for (size_t i = 0; i < reg_hessian->size; i++) {
        reg_hessian->amplitudes[i] = reg_state->amplitudes[i];
    }
}

void quantum_apply_threshold(double complex* data,
                           size_t size,
                           double threshold,
                           quantum_system_t* system,
                           quantum_circuit_t* circuit,
                           const quantum_phase_config_t* config) {
    for (size_t i = 0; i < size; i++) {
        if (cabs(data[i]) < threshold) {
            data[i] = 0;
        }
    }
}

void quantum_apply_matrix_threshold(double complex* matrix,
                                  size_t dim,
                                  double threshold,
                                  quantum_system_t* system,
                                  quantum_circuit_t* circuit,
                                  const quantum_phase_config_t* config) {
    for (size_t i = 0; i < dim * dim; i++) {
        if (cabs(matrix[i]) < threshold) {
            matrix[i] = 0;
        }
    }
}

void qgt_normalize_state(double complex* state, size_t dim) {
    double norm = 0;
    for (size_t i = 0; i < dim; i++) {
        norm += cabs(state[i]) * cabs(state[i]);
    }
    
    if (norm > 0) {
        norm = sqrt(norm);
        for (size_t i = 0; i < dim; i++) {
            state[i] /= norm;
        }
    }
}

void qgt_complex_matrix_multiply(const double complex* a,
                               const double complex* b,
                               double complex* c,
                               size_t m, size_t n, size_t p) {
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < p; j++) {
            c[i * p + j] = 0;
            for (size_t k = 0; k < n; k++) {
                c[i * p + j] += a[i * n + k] * b[k * p + j];
            }
        }
    }
}
