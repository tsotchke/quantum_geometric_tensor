#ifndef QUANTUM_PHASE_ESTIMATION_H
#define QUANTUM_PHASE_ESTIMATION_H

#include <stdbool.h>
#include <complex.h>
#include <stddef.h>
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_system.h"
#include "quantum_geometric/core/quantum_circuit.h"
#include "quantum_geometric/core/quantum_register.h"

// Use existing flags from quantum_system.h
#ifndef QUANTUM_OPTIMIZE_AGGRESSIVE
#define QUANTUM_OPTIMIZE_AGGRESSIVE (1 << 2)
#endif

#ifndef QUANTUM_CIRCUIT_OPTIMAL
#define QUANTUM_CIRCUIT_OPTIMAL (1 << 4)
#endif

// Constants for quantum phase estimation
#define QG_QUANTUM_ESTIMATION_PRECISION 1e-6
#define QG_SUCCESS_PROBABILITY 0.99
#define QG_GRADIENT_THRESHOLD 1e-8
#define QG_MATRIX_THRESHOLD 1e-8
#define QG_ONE 1.0
#define QG_INFINITY 1e38

typedef struct quantum_phase_config_t {
    double precision;
    double success_probability;
    bool use_quantum_fourier;
    bool use_quantum_memory;
    int error_correction;
    int optimization_level;
} quantum_phase_config_t;

// Function declarations
void quantum_phase_estimation_optimized(quantum_register_t* reg_matrix,
                                      quantum_system_t* system,
                                      quantum_circuit_t* circuit,
                                      const quantum_phase_config_t* config);

void quantum_inverse_phase_estimation(quantum_register_t* reg_inverse,
                                    quantum_system_t* system,
                                    quantum_circuit_t* circuit,
                                    const quantum_phase_config_t* config);

void quantum_invert_eigenvalues(quantum_register_t* reg_matrix,
                               quantum_register_t* reg_inverse,
                               quantum_system_t* system,
                               quantum_circuit_t* circuit,
                               const quantum_phase_config_t* config);

int quantum_extract_state(double complex* matrix,
                         quantum_register_t* reg_inverse,
                         size_t size);

#endif // QUANTUM_PHASE_ESTIMATION_H
