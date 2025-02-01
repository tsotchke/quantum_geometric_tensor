#ifndef QUANTUM_STATE_TYPES_H
#define QUANTUM_STATE_TYPES_H

#include <stddef.h>
#include <stdbool.h>
#include "quantum_geometric/core/quantum_complex.h"
#include "quantum_geometric/core/quantum_types.h"
#include "quantum_geometric/hardware/quantum_hardware_types.h"

// Forward declarations
struct quantum_operator_t;
struct quantum_measurement_t;
struct QuantumCircuit;

// Type definitions
typedef struct quantum_operator_t quantum_operator_t;
typedef struct quantum_measurement_t quantum_measurement_t;

// Quantum state structure
typedef struct {
    size_t num_qubits;
    ComplexFloat* amplitudes;
    void* workspace;
    size_t dimension;
    bool is_normalized;
} QuantumState;

// Operator types
typedef enum quantum_operator_type_t {
    QUANTUM_OPERATOR_UNITARY,     // Unitary operator
    QUANTUM_OPERATOR_HERMITIAN,   // Hermitian operator
    QUANTUM_OPERATOR_PROJECTOR,   // Projection operator
    QUANTUM_OPERATOR_KRAUS,       // Kraus operator
    QUANTUM_OPERATOR_LINDBLAD,    // Lindblad operator
    QUANTUM_OPERATOR_CUSTOM       // Custom operator
} quantum_operator_type_t;

// Measurement types
typedef enum quantum_measurement_type_t {
    QUANTUM_MEASUREMENT_PROJECTIVE,  // Projective measurement
    QUANTUM_MEASUREMENT_POVM,        // POVM measurement
    QUANTUM_MEASUREMENT_WEAK,        // Weak measurement
    QUANTUM_MEASUREMENT_CONTINUOUS,  // Continuous measurement
    QUANTUM_MEASUREMENT_CUSTOM       // Custom measurement
} quantum_measurement_type_t;

// Operator structure
struct quantum_operator_t {
    quantum_operator_type_t type;    // Operator type
    size_t dimension;                // Operator dimension
    ComplexFloat* matrix;            // Operator matrix elements
    void* auxiliary_data;            // Additional operator data
    bool is_hermitian;              // Whether operator is Hermitian
};

// Measurement structure
struct quantum_measurement_t {
    quantum_measurement_type_t type;  // Measurement type
    size_t num_outcomes;             // Number of possible outcomes
    quantum_operator_t** effects;     // Measurement effects
    void* auxiliary_data;            // Additional measurement data
};

// State manipulation functions
QuantumState* encode_input(const double* classical_input, const struct QuantumCircuit* circuit);
void cleanup_quantum_state(QuantumState* state);
double* measure_quantum_state(const QuantumState* state);

// State analysis functions
double compute_fidelity(const QuantumState* state1, const QuantumState* state2);
double compute_trace_distance(const QuantumState* state1, const QuantumState* state2);
bool check_orthogonality(const QuantumState* state1, const QuantumState* state2);

#endif // QUANTUM_STATE_TYPES_H
