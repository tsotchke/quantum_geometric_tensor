/**
 * @file mock_quantum_state.c
 * @brief Mock implementations for quantum state operations in tests
 */

#include "mock_quantum_state.h"
#include "quantum_geometric/core/quantum_complex.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ============================================================================
// Context Management
// ============================================================================

qgt_error_t qgt_create_context(qgt_context_t** ctx) {
    if (!ctx) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    *ctx = (qgt_context_t*)malloc(sizeof(qgt_context_t));
    if (!*ctx) {
        return QGT_ERROR_MEMORY_ALLOCATION;
    }

    (*ctx)->internal_state = NULL;
    (*ctx)->max_qubits = 64;
    (*ctx)->initialized = true;

    return QGT_SUCCESS;
}

void qgt_destroy_context(qgt_context_t* ctx) {
    if (ctx) {
        if (ctx->internal_state) {
            free(ctx->internal_state);
        }
        free(ctx);
    }
}

void qgt_destroy_state(qgt_context_t* ctx, qgt_state_t* state) {
    (void)ctx;
    if (state) {
        if (state->coordinates) {
            free(state->coordinates);
        }
        if (state->metric) {
            free(state->metric);
        }
        if (state->connection) {
            free(state->connection);
        }
        if (state->stabilizers) {
            free(state->stabilizers);
        }
        if (state->anyons) {
            free(state->anyons);
        }
        if (state->syndrome_values) {
            free(state->syndrome_values);
        }
        free(state);
    }
}

// ============================================================================
// State Creation
// ============================================================================

qgt_error_t mock_create_state(qgt_context_t* ctx,
                              size_t num_qubits,
                              qgt_state_t** state) {
    if (!ctx || !state) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    *state = (qgt_state_t*)calloc(1, sizeof(qgt_state_t));
    if (!*state) {
        return QGT_ERROR_MEMORY_ALLOCATION;
    }

    (*state)->num_qubits = num_qubits;
    (*state)->dimension = 1ULL << num_qubits;
    (*state)->coordinates = (ComplexFloat*)calloc((*state)->dimension, sizeof(ComplexFloat));

    if (!(*state)->coordinates) {
        free(*state);
        *state = NULL;
        return QGT_ERROR_MEMORY_ALLOCATION;
    }

    // Initialize to |0⟩ state
    (*state)->coordinates[0] = complex_float_create(1.0f, 0.0f);
    (*state)->is_normalized = true;
    (*state)->hardware = HARDWARE_TYPE_CPU;

    return QGT_SUCCESS;
}

// qgt_create_state - Production API wrapper around mock_create_state
// This provides the expected test API using actual state creation
qgt_error_t qgt_create_state(qgt_context_t* ctx,
                             size_t num_qubits,
                             qgt_state_t** state) {
    return mock_create_state(ctx, num_qubits, state);
}

qgt_error_t mock_create_entangled_state(qgt_context_t* ctx,
                                        int type,
                                        size_t num_qubits,
                                        qgt_state_t** state) {
    if (!ctx || !state || num_qubits < 2) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    qgt_error_t err = mock_create_state(ctx, num_qubits, state);
    if (err != QGT_SUCCESS) {
        return err;
    }

    // Create different types of entangled states
    switch ((entangled_state_type_t)type) {
        case ENTANGLED_STATE_BELL:
            if (num_qubits == 2) {
                // |Φ+⟩ = (|00⟩ + |11⟩)/√2
                float norm = 1.0f / sqrtf(2.0f);
                (*state)->coordinates[0] = complex_float_create(norm, 0.0f);
                (*state)->coordinates[3] = complex_float_create(norm, 0.0f);
            }
            break;

        case ENTANGLED_STATE_GHZ:
            // |GHZ⟩ = (|0...0⟩ + |1...1⟩)/√2
            {
                float norm = 1.0f / sqrtf(2.0f);
                (*state)->coordinates[0] = complex_float_create(norm, 0.0f);
                (*state)->coordinates[(*state)->dimension - 1] = complex_float_create(norm, 0.0f);
            }
            break;

        case ENTANGLED_STATE_W:
            // |W⟩ = (|100...0⟩ + |010...0⟩ + ... + |00...01⟩)/√n
            {
                float norm = 1.0f / sqrtf((float)num_qubits);
                for (size_t i = 0; i < num_qubits; i++) {
                    size_t idx = 1ULL << i;
                    (*state)->coordinates[idx] = complex_float_create(norm, 0.0f);
                }
            }
            break;

        default:
            return QGT_ERROR_INVALID_ARGUMENT;
    }

    return QGT_SUCCESS;
}

qgt_error_t mock_create_random_state(qgt_context_t* ctx,
                                     size_t num_qubits,
                                     qgt_state_t** state) {
    qgt_error_t err = mock_create_state(ctx, num_qubits, state);
    if (err != QGT_SUCCESS) {
        return err;
    }

    // Create random state with uniform random complex amplitudes
    double norm = 0.0;
    for (size_t i = 0; i < (*state)->dimension; i++) {
        float real = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        float imag = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        (*state)->coordinates[i] = complex_float_create(real, imag);
        norm += real * real + imag * imag;
    }

    // Normalize
    float norm_factor = 1.0f / sqrtf(norm);
    for (size_t i = 0; i < (*state)->dimension; i++) {
        ComplexFloat c = (*state)->coordinates[i];
        (*state)->coordinates[i] = complex_float_create(
            c.real * norm_factor,
            c.imag * norm_factor
        );
    }

    return QGT_SUCCESS;
}

qgt_error_t mock_create_product_state(qgt_context_t* ctx,
                                      size_t num_qubits,
                                      qgt_state_t** state) {
    // Create |+⟩⊗n state where |+⟩ = (|0⟩ + |1⟩)/√2
    qgt_error_t err = mock_create_state(ctx, num_qubits, state);
    if (err != QGT_SUCCESS) {
        return err;
    }

    float norm = 1.0f / sqrtf((float)(*state)->dimension);
    for (size_t i = 0; i < (*state)->dimension; i++) {
        (*state)->coordinates[i] = complex_float_create(norm, 0.0f);
    }

    return QGT_SUCCESS;
}

void mock_destroy_state(qgt_context_t* ctx, qgt_state_t* state) {
    qgt_destroy_state(ctx, state);
}

// ============================================================================
// Distributed Computing Functions
// ============================================================================

qgt_error_t qgt_create_distributed_context(qgt_context_t* ctx,
                                           size_t num_nodes,
                                           qgt_distributed_context_t** dist_ctx) {
    if (!ctx || !dist_ctx || num_nodes == 0) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    *dist_ctx = (qgt_distributed_context_t*)malloc(sizeof(qgt_distributed_context_t));
    if (!*dist_ctx) {
        return QGT_ERROR_MEMORY_ALLOCATION;
    }

    (*dist_ctx)->num_nodes = num_nodes;
    (*dist_ctx)->rank = 0;
    (*dist_ctx)->initialized = true;
    (*dist_ctx)->internal_state = NULL;

    return QGT_SUCCESS;
}

void qgt_destroy_distributed_context(qgt_distributed_context_t* dist_ctx) {
    if (dist_ctx) {
        if (dist_ctx->internal_state) {
            free(dist_ctx->internal_state);
        }
        free(dist_ctx);
    }
}

qgt_error_t qgt_distribute_state(qgt_distributed_context_t* dist_ctx, qgt_state_t* state) {
    if (!dist_ctx || !state) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    // Mock implementation - just return success
    return QGT_SUCCESS;
}

qgt_error_t qgt_gather_state(qgt_distributed_context_t* dist_ctx, qgt_state_t* state) {
    if (!dist_ctx || !state) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    // Mock implementation - just return success
    return QGT_SUCCESS;
}

qgt_error_t qgt_distributed_geometric_rotate(qgt_distributed_context_t* dist_ctx,
                                              qgt_state_t* state,
                                              double angle,
                                              const double* axis) {
    if (!dist_ctx || !state || !axis) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    // Mock implementation - just return success
    (void)angle;
    return QGT_SUCCESS;
}

qgt_error_t qgt_distributed_geometric_parallel_transport(qgt_distributed_context_t* dist_ctx,
                                                          qgt_state_t* state,
                                                          const double* path,
                                                          size_t num_points) {
    if (!dist_ctx || !state || !path || num_points == 0) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    // Mock implementation - just return success
    return QGT_SUCCESS;
}

qgt_error_t qgt_simulate_node_failure(qgt_distributed_context_t* dist_ctx, size_t node_id) {
    if (!dist_ctx || node_id >= dist_ctx->num_nodes) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    // Mock implementation - just return success
    return QGT_SUCCESS;
}

// ============================================================================
// Additional Operations
// ============================================================================

qgt_error_t qgt_apply_error_channel(qgt_context_t* ctx, qgt_state_t* state, double error_rate) {
    if (!ctx || !state || error_rate < 0.0 || error_rate > 1.0) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    // Mock implementation - apply depolarizing channel
    return QGT_SUCCESS;
}

qgt_error_t qgt_apply_error_correction(qgt_context_t* ctx, qgt_state_t* state) {
    if (!ctx || !state) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    // Mock implementation - just return success
    return QGT_SUCCESS;
}

qgt_error_t qgt_geometric_measure_fidelity(qgt_context_t* ctx,
                                            qgt_state_t* state1,
                                            qgt_state_t* state2,
                                            double* fidelity) {
    if (!ctx || !state1 || !state2 || !fidelity) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    if (state1->dimension != state2->dimension) {
        return QGT_ERROR_DIMENSION_MISMATCH;
    }

    // Compute fidelity as |⟨ψ1|ψ2⟩|²
    ComplexFloat overlap = {0.0f, 0.0f};
    for (size_t i = 0; i < state1->dimension; i++) {
        ComplexFloat c1 = state1->coordinates[i];
        ComplexFloat c2 = state2->coordinates[i];
        // Complex conjugate of c1 times c2
        ComplexFloat prod = {
            c1.real * c2.real + c1.imag * c2.imag,
            c1.real * c2.imag - c1.imag * c2.real
        };
        overlap.real += prod.real;
        overlap.imag += prod.imag;
    }

    *fidelity = overlap.real * overlap.real + overlap.imag * overlap.imag;
    return QGT_SUCCESS;
}

qgt_error_t qgt_create_logical_state(qgt_context_t* ctx, size_t num_qubits, qgt_state_t** state) {
    // For mock, just create a regular state
    return mock_create_state(ctx, num_qubits, state);
}

qgt_error_t qgt_enable_noise_model(qgt_context_t* ctx, double noise_strength) {
    if (!ctx || noise_strength < 0.0) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    // Mock implementation - just return success
    return QGT_SUCCESS;
}

qgt_error_t qgt_geometric_rotate(qgt_context_t* ctx, qgt_state_t* state, double angle, const double* axis) {
    if (!ctx || !state || !axis) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    // Mock implementation - just return success
    (void)angle;
    return QGT_SUCCESS;
}

qgt_error_t qgt_apply_error_correction_cycle(qgt_context_t* ctx, qgt_state_t* state) {
    if (!ctx || !state) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    // Mock implementation - just return success
    return QGT_SUCCESS;
}

qgt_error_t qgt_measure_logical_fidelity(qgt_context_t* ctx, qgt_state_t* state, double* fidelity) {
    if (!ctx || !state || !fidelity) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    // Mock implementation - return perfect fidelity
    *fidelity = 1.0;
    return QGT_SUCCESS;
}

qgt_error_t qgt_geometric_compute_metric(qgt_context_t* ctx, qgt_state_t* state, double* metric) {
    if (!ctx || !state || !metric) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    // Mock implementation - return identity metric
    size_t dim = state->num_qubits;
    for (size_t i = 0; i < dim * dim; i++) {
        metric[i] = (i % (dim + 1) == 0) ? 1.0 : 0.0;
    }
    return QGT_SUCCESS;
}
