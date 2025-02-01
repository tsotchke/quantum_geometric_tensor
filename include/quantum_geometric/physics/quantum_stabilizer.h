#ifndef QUANTUM_STABILIZER_H
#define QUANTUM_STABILIZER_H

#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_complex.h"
#include "quantum_geometric/core/error_codes.h"
#include <stdbool.h>
#include <stddef.h>

// Pauli operator types
typedef enum {
    PAULI_I,      // Identity operator
    PAULI_X,      // Pauli X operator
    PAULI_Y,      // Pauli Y operator
    PAULI_Z       // Pauli Z operator
} pauli_type_t;

// Pauli operator structure
typedef struct {
    pauli_type_t type;        // Type of Pauli operator
    size_t qubit_index;       // Index of target qubit
    ComplexFloat coefficient;  // Operator coefficient
} quantum_pauli_t;

// Stabilizer types
typedef enum {
    STABILIZER_X,      // Pauli X stabilizer
    STABILIZER_Y,      // Pauli Y stabilizer
    STABILIZER_Z,      // Pauli Z stabilizer
    STABILIZER_CUSTOM  // Custom stabilizer operator
} stabilizer_type_t;

// Stabilizer structure
typedef struct {
    stabilizer_type_t type;          // Type of stabilizer
    size_t dimension;                // Dimension of stabilizer
    size_t* qubit_indices;          // Indices of qubits involved
    size_t num_qubits;              // Number of qubits involved
    ComplexFloat* coefficients;      // Stabilizer coefficients
    quantum_pauli_t* terms;         // Array of Pauli terms
    size_t num_terms;               // Number of Pauli terms
    bool is_hermitian;              // Whether stabilizer is Hermitian
    void* auxiliary_data;           // Additional stabilizer data
} quantum_stabilizer_t;

// Create stabilizer operator
qgt_error_t stabilizer_create(quantum_stabilizer_t** stabilizer,
                             stabilizer_type_t type,
                             const size_t* qubit_indices,
                             size_t num_qubits);

// Destroy stabilizer operator
void stabilizer_destroy(quantum_stabilizer_t* stabilizer);

// Clone stabilizer operator
qgt_error_t stabilizer_clone(quantum_stabilizer_t** dest,
                            const quantum_stabilizer_t* src);

// Apply stabilizer to state
qgt_error_t stabilizer_apply(const quantum_stabilizer_t* stabilizer,
                            quantum_geometric_state_t* state);

// Measure stabilizer expectation value
qgt_error_t stabilizer_measure(double* expectation,
                              const quantum_stabilizer_t* stabilizer,
                              const quantum_geometric_state_t* state);

// Check if state is stabilizer eigenstate
qgt_error_t stabilizer_is_eigenstate(bool* is_eigenstate,
                                    const quantum_stabilizer_t* stabilizer,
                                    const quantum_geometric_state_t* state);

// Get stabilizer eigenvalue
qgt_error_t stabilizer_eigenvalue(double* eigenvalue,
                                 const quantum_stabilizer_t* stabilizer,
                                 const quantum_geometric_state_t* state);

// Check if two stabilizers commute
qgt_error_t stabilizer_commute(bool* commute,
                              const quantum_stabilizer_t* stabilizer1,
                              const quantum_stabilizer_t* stabilizer2);

// Multiply two stabilizers
qgt_error_t stabilizer_multiply(quantum_stabilizer_t** result,
                               const quantum_stabilizer_t* stabilizer1,
                               const quantum_stabilizer_t* stabilizer2);

// Check if stabilizer is valid
qgt_error_t stabilizer_validate(const quantum_stabilizer_t* stabilizer);

#endif // QUANTUM_STABILIZER_H
