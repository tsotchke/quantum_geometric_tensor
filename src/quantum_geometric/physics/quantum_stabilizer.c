#include "quantum_geometric/physics/quantum_stabilizer.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/core/memory_pool.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Helper functions
static inline bool is_valid_stabilizer_type(stabilizer_type_t type) {
    return type >= STABILIZER_X && type <= STABILIZER_CUSTOM;
}

static inline bool is_valid_qubit_index(size_t index, size_t dimension) {
    return index < dimension;
}

qgt_error_t stabilizer_create(quantum_stabilizer_t** stabilizer,
                             stabilizer_type_t type,
                             const size_t* qubit_indices,
                             size_t num_qubits) {
    if (!stabilizer || !qubit_indices || num_qubits == 0) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    if (!is_valid_stabilizer_type(type)) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    *stabilizer = malloc(sizeof(quantum_stabilizer_t));
    if (!*stabilizer) {
        return QGT_ERROR_ALLOCATION_FAILED;
    }
    
    // Initialize basic fields
    (*stabilizer)->type = type;
    (*stabilizer)->num_qubits = num_qubits;
    (*stabilizer)->dimension = 1 << num_qubits; // 2^n dimension
    (*stabilizer)->is_hermitian = true; // Pauli operators are Hermitian
    (*stabilizer)->auxiliary_data = NULL;
    
    // Allocate and copy qubit indices
    (*stabilizer)->qubit_indices = malloc(num_qubits * sizeof(size_t));
    if (!(*stabilizer)->qubit_indices) {
        free(*stabilizer);
        return QGT_ERROR_ALLOCATION_FAILED;
    }
    memcpy((*stabilizer)->qubit_indices, qubit_indices, num_qubits * sizeof(size_t));
    
    // Allocate coefficients
    (*stabilizer)->coefficients = malloc((*stabilizer)->dimension * sizeof(ComplexFloat));
    if (!(*stabilizer)->coefficients) {
        free((*stabilizer)->qubit_indices);
        free(*stabilizer);
        return QGT_ERROR_ALLOCATION_FAILED;
    }
    
    // Initialize coefficients based on stabilizer type
    for (size_t i = 0; i < (*stabilizer)->dimension; i++) {
        switch (type) {
            case STABILIZER_X:
                (*stabilizer)->coefficients[i] = complex_float_create(1.0f, 0.0f);
                break;
            case STABILIZER_Y:
                (*stabilizer)->coefficients[i] = complex_float_create(0.0f, 1.0f);
                break;
            case STABILIZER_Z:
                (*stabilizer)->coefficients[i] = complex_float_create(1.0f, 0.0f);
                break;
            case STABILIZER_CUSTOM:
                (*stabilizer)->coefficients[i] = COMPLEX_FLOAT_ZERO;
                break;
        }
    }
    
    return QGT_SUCCESS;
}

void stabilizer_destroy(quantum_stabilizer_t* stabilizer) {
    if (!stabilizer) return;
    
    free(stabilizer->qubit_indices);
    free(stabilizer->coefficients);
    free(stabilizer);
}

qgt_error_t stabilizer_clone(quantum_stabilizer_t** dest,
                            const quantum_stabilizer_t* src) {
    if (!dest || !src) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    qgt_error_t err = stabilizer_create(dest, src->type,
                                       src->qubit_indices,
                                       src->num_qubits);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    // Copy coefficients
    memcpy((*dest)->coefficients, src->coefficients,
           src->dimension * sizeof(ComplexFloat));
    
    // Copy other fields
    (*dest)->is_hermitian = src->is_hermitian;
    
    // Clone auxiliary data if present
    if (src->auxiliary_data) {
        (*dest)->auxiliary_data = malloc(sizeof(void*));
        if (!(*dest)->auxiliary_data) {
            stabilizer_destroy(*dest);
            return QGT_ERROR_ALLOCATION_FAILED;
        }
        memcpy((*dest)->auxiliary_data, src->auxiliary_data, sizeof(void*));
    }
    
    return QGT_SUCCESS;
}

qgt_error_t stabilizer_apply(const quantum_stabilizer_t* stabilizer,
                            quantum_geometric_state_t* state) {
    if (!stabilizer || !state) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    if (state->dimension != stabilizer->dimension) {
        return QGT_ERROR_INVALID_DIMENSION;
    }
    
    // Apply stabilizer operation based on type
    switch (stabilizer->type) {
        case STABILIZER_X:
            // Bit flip
            for (size_t i = 0; i < stabilizer->num_qubits; i++) {
                size_t idx = stabilizer->qubit_indices[i];
                if (!is_valid_qubit_index(idx, state->dimension)) {
                    return QGT_ERROR_INVALID_ARGUMENT;
                }
                // Flip bit at index
                state->coordinates[idx] = complex_float_multiply(
                    state->coordinates[idx],
                    complex_float_create(-1.0f, 0.0f)
                );
            }
            break;
            
        case STABILIZER_Y:
            // Phase and bit flip
            for (size_t i = 0; i < stabilizer->num_qubits; i++) {
                size_t idx = stabilizer->qubit_indices[i];
                if (!is_valid_qubit_index(idx, state->dimension)) {
                    return QGT_ERROR_INVALID_ARGUMENT;
                }
                // Apply i * Z * X
                state->coordinates[idx] = complex_float_multiply(
                    state->coordinates[idx],
                    complex_float_create(0.0f, 1.0f)
                );
            }
            break;
            
        case STABILIZER_Z:
            // Phase flip
            for (size_t i = 0; i < stabilizer->num_qubits; i++) {
                size_t idx = stabilizer->qubit_indices[i];
                if (!is_valid_qubit_index(idx, state->dimension)) {
                    return QGT_ERROR_INVALID_ARGUMENT;
                }
                // Flip phase
                state->coordinates[idx] = complex_float_multiply(
                    state->coordinates[idx],
                    complex_float_create(-1.0f, 0.0f)
                );
            }
            break;
            
        case STABILIZER_CUSTOM:
            // Apply custom stabilizer operation
            for (size_t i = 0; i < state->dimension; i++) {
                state->coordinates[i] = complex_float_multiply(
                    state->coordinates[i],
                    stabilizer->coefficients[i]
                );
            }
            break;
    }
    
    return QGT_SUCCESS;
}

qgt_error_t stabilizer_measure(double* expectation,
                              const quantum_stabilizer_t* stabilizer,
                              const quantum_geometric_state_t* state) {
    if (!expectation || !stabilizer || !state) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    if (state->dimension != stabilizer->dimension) {
        return QGT_ERROR_INVALID_DIMENSION;
    }
    
    // Create temporary state for measurement
    quantum_geometric_state_t* temp_state;
    qgt_error_t err = geometric_create_state(&temp_state,
                                           state->type,
                                           state->dimension);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    // Copy state
    memcpy(temp_state->coordinates, state->coordinates,
           state->dimension * sizeof(ComplexFloat));
    
    // Apply stabilizer
    err = stabilizer_apply(stabilizer, temp_state);
    if (err != QGT_SUCCESS) {
        geometric_destroy_state(temp_state);
        return err;
    }
    
    // Calculate expectation value
    ComplexFloat sum = COMPLEX_FLOAT_ZERO;
    for (size_t i = 0; i < state->dimension; i++) {
        sum = complex_float_add(sum,
            complex_float_multiply(
                complex_float_conjugate(state->coordinates[i]),
                temp_state->coordinates[i]
            )
        );
    }
    
    *expectation = complex_float_abs(sum);
    
    geometric_destroy_state(temp_state);
    return QGT_SUCCESS;
}

qgt_error_t stabilizer_is_eigenstate(bool* is_eigenstate,
                                    const quantum_stabilizer_t* stabilizer,
                                    const quantum_geometric_state_t* state) {
    if (!is_eigenstate || !stabilizer || !state) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    double expectation;
    qgt_error_t err = stabilizer_measure(&expectation, stabilizer, state);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    // State is eigenstate if expectation value is +/-1
    *is_eigenstate = fabs(fabs(expectation) - 1.0) < 1e-6;
    
    return QGT_SUCCESS;
}

qgt_error_t stabilizer_eigenvalue(double* eigenvalue,
                                 const quantum_stabilizer_t* stabilizer,
                                 const quantum_geometric_state_t* state) {
    if (!eigenvalue || !stabilizer || !state) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    bool is_eigen;
    qgt_error_t err = stabilizer_is_eigenstate(&is_eigen, stabilizer, state);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    if (!is_eigen) {
        return QGT_ERROR_INVALID_STATE;
    }
    
    double expectation;
    err = stabilizer_measure(&expectation, stabilizer, state);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    *eigenvalue = expectation;
    return QGT_SUCCESS;
}

qgt_error_t stabilizer_commute(bool* commute,
                              const quantum_stabilizer_t* stabilizer1,
                              const quantum_stabilizer_t* stabilizer2) {
    if (!commute || !stabilizer1 || !stabilizer2) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    // Count overlapping qubits
    size_t overlap = 0;
    for (size_t i = 0; i < stabilizer1->num_qubits; i++) {
        for (size_t j = 0; j < stabilizer2->num_qubits; j++) {
            if (stabilizer1->qubit_indices[i] == stabilizer2->qubit_indices[j]) {
                overlap++;
            }
        }
    }
    
    // Stabilizers commute if they have even overlap
    *commute = (overlap % 2) == 0;
    
    return QGT_SUCCESS;
}

qgt_error_t stabilizer_multiply(quantum_stabilizer_t** result,
                               const quantum_stabilizer_t* stabilizer1,
                               const quantum_stabilizer_t* stabilizer2) {
    if (!result || !stabilizer1 || !stabilizer2) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    // Create result stabilizer
    qgt_error_t err = stabilizer_create(result, STABILIZER_CUSTOM,
                                       stabilizer1->qubit_indices,
                                       stabilizer1->num_qubits);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    // Multiply coefficients
    for (size_t i = 0; i < (*result)->dimension; i++) {
        (*result)->coefficients[i] = complex_float_multiply(
            stabilizer1->coefficients[i],
            stabilizer2->coefficients[i]
        );
    }
    
    return QGT_SUCCESS;
}

qgt_error_t stabilizer_validate(const quantum_stabilizer_t* stabilizer) {
    if (!stabilizer) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    // Check type
    if (!is_valid_stabilizer_type(stabilizer->type)) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    // Check qubit indices
    for (size_t i = 0; i < stabilizer->num_qubits; i++) {
        if (!is_valid_qubit_index(stabilizer->qubit_indices[i],
                                 stabilizer->dimension)) {
            return QGT_ERROR_INVALID_ARGUMENT;
        }
    }
    
    // Check coefficients are normalized
    ComplexFloat sum = COMPLEX_FLOAT_ZERO;
    for (size_t i = 0; i < stabilizer->dimension; i++) {
        sum = complex_float_add(sum,
            complex_float_multiply(
                complex_float_conjugate(stabilizer->coefficients[i]),
                stabilizer->coefficients[i]
            )
        );
    }
    
    if (fabs(complex_float_abs(sum) - 1.0) > 1e-6) {
        return QGT_ERROR_INVALID_STATE;
    }
    
    return QGT_SUCCESS;
}
