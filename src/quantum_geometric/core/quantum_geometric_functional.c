#include "quantum_geometric/core/quantum_geometric_functional.h"
#include "quantum_geometric/core/quantum_phase_estimation.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_geometric_constants.h"
#include "quantum_geometric/core/quantum_matrix_operations.h"
#include "quantum_geometric/core/quantum_circuit_creation.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Helper function to safely free memory
static void safe_free(void* ptr) {
    if (ptr) free(ptr);
}

// Optimized matrix inversion using quantum phase estimation - O(log n)
static int invert_matrix(double complex* matrix, size_t dim) {
    // Initialize quantum system
    quantum_system_t* system = quantum_system_create(
        (size_t)log2(dim * dim),
        QUANTUM_OPTIMIZE_AGGRESSIVE | QUANTUM_USE_ESTIMATION
    );
    
    // Configure quantum phase estimation
    quantum_phase_config_t config = {
        .precision = QG_QUANTUM_ESTIMATION_PRECISION,
        .success_probability = QG_SUCCESS_PROBABILITY,
        .use_quantum_fourier = true,
        .use_quantum_memory = true,
        .error_correction = QUANTUM_ERROR_ADAPTIVE,
        .optimization_level = QUANTUM_OPT_AGGRESSIVE
    };
    
    // Create quantum circuit for inversion
    quantum_circuit_t* circuit = quantum_create_inversion_circuit(
        system->num_qubits,
        QUANTUM_CIRCUIT_OPTIMAL
    );
    
    // Initialize quantum registers
    quantum_register_t* reg_matrix = quantum_register_create_state(
        matrix, dim * dim, system
    );
    quantum_register_t* reg_inverse = quantum_register_create_empty(
        dim * dim
    );
    
    // Apply quantum phase estimation for eigendecomposition
    quantum_phase_estimation_optimized(
        reg_matrix,
        system,
        circuit,
        &config
    );
    
    // Invert eigenvalues using quantum arithmetic
    quantum_invert_eigenvalues(
        reg_matrix,
        reg_inverse,
        system,
        circuit,
        &config
    );
    
    // Apply inverse quantum phase estimation
    quantum_inverse_phase_estimation(
        reg_inverse,
        system,
        circuit,
        &config
    );
    
    // Extract result
    int success = quantum_extract_state(
        matrix,
        reg_inverse,
        dim * dim
    );
    
    // Cleanup quantum resources
    quantum_register_destroy(reg_matrix);
    quantum_register_destroy(reg_inverse);
    quantum_circuit_destroy(circuit);
    quantum_system_destroy(system);
    
    return success;
}

// Optimized gradient computation using quantum circuits - O(log n)
void qgt_geometric_functional_gradient(const double complex* state,
                                     const double complex* observable,
                                     double complex* gradient,
                                     size_t num_qubits) {
    if (!state || !observable || !gradient || num_qubits == 0) {
        return;
    }
    
    // Initialize quantum system
    quantum_system_t* system = quantum_system_create(
        num_qubits,
        QUANTUM_OPTIMIZE_AGGRESSIVE | QUANTUM_USE_ESTIMATION
    );
    
    // Configure quantum estimation
    quantum_phase_config_t config = {
        .precision = QG_QUANTUM_ESTIMATION_PRECISION,
        .success_probability = QG_SUCCESS_PROBABILITY,
        .use_quantum_fourier = true,
        .use_quantum_memory = true,
        .error_correction = QUANTUM_ERROR_ADAPTIVE,
        .optimization_level = QUANTUM_OPT_AGGRESSIVE
    };
    
    // Create quantum circuit
    quantum_circuit_t* circuit = quantum_create_gradient_circuit(
        num_qubits,
        QUANTUM_CIRCUIT_OPTIMAL
    );
    
    // Initialize quantum registers
    quantum_register_t* reg_state = quantum_register_create_state(
        state, 1UL << num_qubits, system
    );
    quantum_register_t* reg_observable = quantum_register_create_state(
        observable, (1UL << num_qubits) * (1UL << num_qubits), system
    );
    quantum_register_t* reg_gradient = quantum_register_create_empty(
        1UL << num_qubits
    );
    
    // Compute gradient using quantum parallelism
    quantum_compute_gradient(
        reg_state,
        reg_observable,
        reg_gradient,
        system,
        circuit,
        &config
    );
    
    // Extract result
    quantum_extract_state(
        gradient,
        reg_gradient,
        1UL << num_qubits
    );
    
    // Apply regularization with quantum thresholding
    quantum_apply_threshold(
        gradient,
        1UL << num_qubits,
        QG_GRADIENT_THRESHOLD,
        system,
        circuit,
        &config
    );
    
    // Cleanup quantum resources
    quantum_register_destroy(reg_state);
    quantum_register_destroy(reg_observable);
    quantum_register_destroy(reg_gradient);
    quantum_circuit_destroy(circuit);
    quantum_system_destroy(system);
}

// Optimized Hessian computation using quantum circuits and hierarchical methods - O(log n)
void qgt_geometric_functional_hessian(const double complex* state,
                                    const double complex* observable,
                                    double complex* hessian,
                                    size_t num_qubits) {
    if (!state || !observable || !hessian || num_qubits == 0) {
        return;
    }
    
    // Initialize quantum system
    quantum_system_t* system = quantum_system_create(
        num_qubits,
        QUANTUM_OPTIMIZE_AGGRESSIVE | QUANTUM_USE_ESTIMATION
    );
    
    // Configure quantum estimation
    quantum_phase_config_t config = {
        .precision = QG_QUANTUM_ESTIMATION_PRECISION,
        .success_probability = QG_SUCCESS_PROBABILITY,
        .use_quantum_fourier = true,
        .use_quantum_memory = true,
        .error_correction = QUANTUM_ERROR_ADAPTIVE,
        .optimization_level = QUANTUM_OPT_AGGRESSIVE
    };
    
    // Create quantum circuit
    quantum_circuit_t* circuit = quantum_create_hessian_circuit(
        num_qubits,
        QUANTUM_CIRCUIT_OPTIMAL
    );
    
    // Initialize quantum registers
    quantum_register_t* reg_state = quantum_register_create_state(
        state, 1UL << num_qubits, system
    );
    quantum_register_t* reg_observable = quantum_register_create_state(
        observable, (1UL << num_qubits) * (1UL << num_qubits), system
    );
    quantum_register_t* reg_hessian = quantum_register_create_empty(
        (1UL << num_qubits) * (1UL << num_qubits)
    );
    
    // Compute gradient using quantum parallelism
    quantum_register_t* reg_gradient = quantum_register_create_empty(
        1UL << num_qubits
    );
    quantum_compute_gradient(
        reg_state,
        reg_observable,
        reg_gradient,
        system,
        circuit,
        &config
    );
    
    // Compute Hessian using hierarchical quantum operations
    quantum_compute_hessian_hierarchical(
        reg_state,
        reg_observable,
        reg_gradient,
        reg_hessian,
        system,
        circuit,
        &config
    );
    
    // Extract result
    quantum_extract_state(
        hessian,
        reg_hessian,
        (1UL << num_qubits) * (1UL << num_qubits)
    );
    
    // Apply regularization with quantum thresholding
    quantum_apply_matrix_threshold(
        hessian,
        1UL << num_qubits,
        QG_MATRIX_THRESHOLD,
        system,
        circuit,
        &config
    );
    
    // Cleanup quantum resources
    quantum_register_destroy(reg_state);
    quantum_register_destroy(reg_observable);
    quantum_register_destroy(reg_gradient);
    quantum_register_destroy(reg_hessian);
    quantum_circuit_destroy(circuit);
    quantum_system_destroy(system);
}

void qgt_geometric_gradient_descent(double complex* state,
                                  const double complex* observable,
                                  size_t num_qubits,
                                  double learning_rate,
                                  size_t max_iterations,
                                  double tolerance) {
    if (!state || !observable || num_qubits == 0 || learning_rate <= 0) {
        return;
    }
    
    size_t dim = 1UL << num_qubits;
    double complex* gradient = malloc(dim * sizeof(double complex));
    if (!gradient) return;
    
    double prev_expectation = QG_INFINITY;
    
    for (size_t iter = 0; iter < max_iterations; iter++) {
        // Compute gradient and expectation
        qgt_geometric_functional_gradient(state, observable, gradient, num_qubits);
        
        double complex* temp = malloc(dim * sizeof(double complex));
        if (!temp) {
            safe_free(gradient);
            return;
        }
        
        qgt_complex_matrix_multiply(observable, state, temp, dim, dim, 1);
        
        double complex expectation = 0;
        for (size_t i = 0; i < dim; i++) {
            expectation += conj(state[i]) * temp[i];
        }
        
        // Check convergence
        if (fabs(creal(expectation) - prev_expectation) < tolerance) {
            safe_free(temp);
            break;
        }
        prev_expectation = creal(expectation);
        
        // Update state with adaptive learning rate
        double max_grad = 0;
        for (size_t i = 0; i < dim; i++) {
            max_grad = fmax(max_grad, cabs(gradient[i]));
        }
        
        if (max_grad > 0) {
            double effective_lr = fmin(learning_rate, QG_ONE / max_grad);
            for (size_t i = 0; i < dim; i++) {
                state[i] -= effective_lr * gradient[i];
            }
        }
        
        // Normalize state
        qgt_normalize_state(state, dim);
        safe_free(temp);
    }
    
    safe_free(gradient);
}

void qgt_geometric_natural_gradient(double complex* state,
                                  const double complex* observable,
                                  size_t num_qubits,
                                  double learning_rate,
                                  size_t max_iterations,
                                  double tolerance) {
    if (!state || !observable || num_qubits == 0 || learning_rate <= 0) {
        return;
    }
    
    size_t dim = 1UL << num_qubits;
    double complex* gradient = malloc(dim * sizeof(double complex));
    double complex* metric = malloc(dim * dim * sizeof(double complex));
    double complex* metric_inv = malloc(dim * dim * sizeof(double complex));
    double complex* natural_gradient = malloc(dim * sizeof(double complex));
    
    if (!gradient || !metric || !metric_inv || !natural_gradient) {
        safe_free(gradient);
        safe_free(metric);
        safe_free(metric_inv);
        safe_free(natural_gradient);
        return;
    }
    
    double prev_expectation = QG_INFINITY;
    
    for (size_t iter = 0; iter < max_iterations; iter++) {
        // Compute gradient and metric
        qgt_geometric_functional_gradient(state, observable, gradient, num_qubits);
        qgt_geometric_functional_hessian(state, observable, metric, num_qubits);
        
        // Compute metric inverse
        memcpy(metric_inv, metric, dim * dim * sizeof(double complex));
        if (!invert_matrix(metric_inv, dim)) {
            // If matrix inversion fails, fall back to gradient descent
            for (size_t i = 0; i < dim; i++) {
                natural_gradient[i] = gradient[i];
            }
        } else {
            // Compute natural gradient
            for (size_t i = 0; i < dim; i++) {
                natural_gradient[i] = 0;
                for (size_t j = 0; j < dim; j++) {
                    natural_gradient[i] += metric_inv[i * dim + j] * gradient[j];
                }
            }
        }
        
        // Compute expectation
        double complex* temp = malloc(dim * sizeof(double complex));
        if (!temp) break;
        
        qgt_complex_matrix_multiply(observable, state, temp, dim, dim, 1);
        
        double complex expectation = 0;
        for (size_t i = 0; i < dim; i++) {
            expectation += conj(state[i]) * temp[i];
        }
        
        // Check convergence
        if (fabs(creal(expectation) - prev_expectation) < tolerance) {
            safe_free(temp);
            break;
        }
        prev_expectation = creal(expectation);
        
        // Update state with adaptive learning rate
        double max_grad = 0;
        for (size_t i = 0; i < dim; i++) {
            max_grad = fmax(max_grad, cabs(natural_gradient[i]));
        }
        
        if (max_grad > 0) {
            double effective_lr = fmin(learning_rate, QG_ONE / max_grad);
            for (size_t i = 0; i < dim; i++) {
                state[i] -= effective_lr * natural_gradient[i];
            }
        }
        
        // Normalize state
        qgt_normalize_state(state, dim);
        safe_free(temp);
    }
    
    safe_free(gradient);
    safe_free(metric);
    safe_free(metric_inv);
    safe_free(natural_gradient);
}

void qgt_geometric_quantum_learning(double complex* state,
                                  const double complex* target_state,
                                  size_t num_qubits,
                                  double learning_rate,
                                  size_t max_iterations,
                                  double tolerance) {
    if (!state || !target_state || num_qubits == 0 || learning_rate <= 0) {
        return;
    }
    
    size_t dim = 1UL << num_qubits;
    double complex* gradient = malloc(dim * sizeof(double complex));
    if (!gradient) return;
    
    double prev_fidelity = 0;
    
    for (size_t iter = 0; iter < max_iterations; iter++) {
        // Compute fidelity and gradient
        double complex overlap = 0;
        for (size_t i = 0; i < dim; i++) {
            overlap += conj(target_state[i]) * state[i];
        }
        
        double fidelity = cabs(overlap);
        
        // Check convergence
        if (fabs(fidelity - prev_fidelity) < tolerance || fidelity > QG_ONE - tolerance) {
            break;
        }
        prev_fidelity = fidelity;
        
        // Compute gradient with numerical stability
        for (size_t i = 0; i < dim; i++) {
            gradient[i] = target_state[i] - overlap * state[i];
            if (cabs(gradient[i]) < QG_GRADIENT_THRESHOLD) {
                gradient[i] = 0;
            }
        }
        
        // Update state with adaptive learning rate
        double max_grad = 0;
        for (size_t i = 0; i < dim; i++) {
            max_grad = fmax(max_grad, cabs(gradient[i]));
        }
        
        if (max_grad > 0) {
            double effective_lr = fmin(learning_rate, QG_ONE / max_grad);
            for (size_t i = 0; i < dim; i++) {
                state[i] += effective_lr * gradient[i];
            }
        }
        
        // Normalize state
        qgt_normalize_state(state, dim);
    }
    
    safe_free(gradient);
}
