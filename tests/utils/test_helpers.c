#include "test_helpers.h"
#include "test_config.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

void init_zero_state(qgt_context_t* ctx, qgt_state_t* state) {
    memset(state->amplitudes, 0, state->dim * sizeof(double complex));
    state->amplitudes[0] = 1.0;
}

void init_random_state(qgt_context_t* ctx, qgt_state_t* state) {
    double norm = 0.0;
    for (size_t i = 0; i < state->dim; i++) {
        double real = (double)rand() / RAND_MAX;
        double imag = (double)rand() / RAND_MAX;
        state->amplitudes[i] = real + I * imag;
        norm += creal(conj(state->amplitudes[i]) * state->amplitudes[i]);
    }
    norm = sqrt(norm);
    for (size_t i = 0; i < state->dim; i++) {
        state->amplitudes[i] /= norm;
    }
}

void init_basis_state(qgt_context_t* ctx, qgt_state_t* state, size_t basis) {
    if (basis >= state->dim) return;
    memset(state->amplitudes, 0, state->dim * sizeof(double complex));
    state->amplitudes[basis] = 1.0;
}

void init_bell_state(qgt_context_t* ctx, qgt_state_t* state) {
    if (state->dim < 4) return;
    memset(state->amplitudes, 0, state->dim * sizeof(double complex));
    state->amplitudes[0] = 1.0 / M_SQRT2;
    state->amplitudes[3] = 1.0 / M_SQRT2;
}

void init_ghz_state(qgt_context_t* ctx, qgt_state_t* state) {
    memset(state->amplitudes, 0, state->dim * sizeof(double complex));
    state->amplitudes[0] = 1.0 / M_SQRT2;
    state->amplitudes[state->dim - 1] = 1.0 / M_SQRT2;
}

void init_w_state(qgt_context_t* ctx, qgt_state_t* state) {
    size_t num_qubits = (size_t)log2(state->dim);
    if (num_qubits < 2) return;
    
    memset(state->amplitudes, 0, state->dim * sizeof(double complex));
    double norm = 1.0 / sqrt(num_qubits);
    
    for (size_t i = 0; i < num_qubits; i++) {
        state->amplitudes[1 << i] = norm;
    }
}

bool verify_state_normalization(const qgt_state_t* state, double tolerance) {
    double norm = 0.0;
    for (size_t i = 0; i < state->dim; i++) {
        norm += creal(conj(state->amplitudes[i]) * state->amplitudes[i]);
    }
    return fabs(norm - 1.0) < tolerance;
}

bool verify_state_orthogonality(const qgt_state_t* state1, 
                              const qgt_state_t* state2,
                              double tolerance) {
    if (state1->dim != state2->dim) return false;
    
    double complex inner = 0;
    for (size_t i = 0; i < state1->dim; i++) {
        inner += conj(state1->amplitudes[i]) * state2->amplitudes[i];
    }
    return cabs(inner) < tolerance;
}

bool verify_geometric_phase(qgt_context_t* ctx,
                          const qgt_state_t* state,
                          const double* path,
                          size_t num_points,
                          double tolerance) {
    // Create copy of state for transport
    qgt_state_t* transported;
    if (qgt_create_state(ctx, (size_t)log2(state->dim), &transported) != QGT_SUCCESS) {
        return false;
    }
    memcpy(transported->amplitudes, state->amplitudes, 
           state->dim * sizeof(double complex));
    
    // Transport state along path
    qgt_error_t err = qgt_geometric_parallel_transport(ctx, transported, path, num_points);
    if (err != QGT_SUCCESS) {
        qgt_destroy_state(ctx, transported);
        return false;
    }
    
    // Compute geometric phase
    double complex overlap = 0;
    for (size_t i = 0; i < state->dim; i++) {
        overlap += conj(state->amplitudes[i]) * transported->amplitudes[i];
    }
    
    qgt_destroy_state(ctx, transported);
    
    // Phase should be purely imaginary for geometric transport
    return fabs(creal(overlap)) > 1.0 - tolerance &&
           fabs(cimag(overlap)) < tolerance;
}

bool verify_metric_positivity(const double* metric,
                            size_t dim,
                            double tolerance) {
    // Check positive definiteness using Cholesky decomposition
    double* L = malloc(dim * dim * sizeof(double));
    if (!L) return false;
    
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j <= i; j++) {
            double sum = metric[i * dim + j];
            
            for (size_t k = 0; k < j; k++) {
                sum -= L[i * dim + k] * L[j * dim + k];
            }
            
            if (i == j) {
                if (sum <= 0) {
                    free(L);
                    return false;
                }
                L[i * dim + i] = sqrt(sum);
            } else {
                L[i * dim + j] = sum / L[j * dim + j];
            }
        }
    }
    
    free(L);
    return true;
}

bool verify_connection_compatibility(const double complex* connection,
                                  const double* metric,
                                  size_t dim,
                                  double tolerance) {
    // Check metric compatibility condition
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            double complex sum = 0;
            for (size_t k = 0; k < dim; k++) {
                sum += metric[i * dim + k] * connection[k * dim + j];
            }
            if (cabs(sum + conj(sum)) > tolerance) {
                return false;
            }
        }
    }
    return true;
}
