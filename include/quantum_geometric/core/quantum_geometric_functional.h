#ifndef QUANTUM_GEOMETRIC_FUNCTIONAL_H
#define QUANTUM_GEOMETRIC_FUNCTIONAL_H

#include <complex.h>
#include <stddef.h>
#include "quantum_geometric/core/quantum_phase_estimation.h"
#include "quantum_geometric/core/quantum_geometric_types.h"

// Function declarations
void qgt_geometric_functional_gradient(const double complex* state,
                                     const double complex* observable,
                                     double complex* gradient,
                                     size_t num_qubits);

void qgt_geometric_functional_hessian(const double complex* state,
                                    const double complex* observable,
                                    double complex* hessian,
                                    size_t num_qubits);

void qgt_geometric_gradient_descent(double complex* state,
                                  const double complex* observable,
                                  size_t num_qubits,
                                  double learning_rate,
                                  size_t max_iterations,
                                  double tolerance);

void qgt_geometric_natural_gradient(double complex* state,
                                  const double complex* observable,
                                  size_t num_qubits,
                                  double learning_rate,
                                  size_t max_iterations,
                                  double tolerance);

void qgt_geometric_quantum_learning(double complex* state,
                                  const double complex* target_state,
                                  size_t num_qubits,
                                  double learning_rate,
                                  size_t max_iterations,
                                  double tolerance);

#endif // QUANTUM_GEOMETRIC_FUNCTIONAL_H
