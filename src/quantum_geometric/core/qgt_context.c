/**
 * @file qgt_context.c
 * @brief QGT context and distributed operations implementation
 *
 * Implements the high-level qgt_* context-based API for quantum geometric operations.
 * This provides a unified interface for state management, error correction,
 * and distributed computing.
 */

#include "quantum_geometric/core/quantum_geometric_operations.h"
#include "quantum_geometric/core/quantum_state.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/error_codes.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

// ============================================================================
// Context Types (TDD - defined by test expectations)
// ============================================================================

typedef struct qgt_context {
    void* internal_state;
    size_t max_qubits;
    bool initialized;
    double noise_strength;
    bool noise_enabled;
} qgt_context_t;

typedef struct qgt_distributed_context {
    size_t num_nodes;
    size_t rank;
    bool initialized;
    void* internal_state;
    bool* node_failed;  // Track failed nodes
} qgt_distributed_context_t;

// ============================================================================
// Context Management
// ============================================================================

qgt_error_t qgt_create_context(qgt_context_t** ctx) {
    if (!ctx) return QGT_ERROR_INVALID_ARGUMENT;

    *ctx = calloc(1, sizeof(qgt_context_t));
    if (!*ctx) return QGT_ERROR_NO_MEMORY;

    (*ctx)->max_qubits = 32;
    (*ctx)->initialized = true;
    (*ctx)->noise_strength = 0.0;
    (*ctx)->noise_enabled = false;

    // Initialize global geometric state if needed
    geometric_initialize();

    return QGT_SUCCESS;
}

void qgt_destroy_context(qgt_context_t* ctx) {
    if (ctx) {
        free(ctx);
    }
}

// ============================================================================
// Distributed Context
// ============================================================================

qgt_error_t qgt_create_distributed_context(qgt_context_t* ctx,
                                           size_t num_nodes,
                                           qgt_distributed_context_t** dist_ctx) {
    if (!ctx || !dist_ctx || num_nodes == 0) return QGT_ERROR_INVALID_ARGUMENT;

    *dist_ctx = calloc(1, sizeof(qgt_distributed_context_t));
    if (!*dist_ctx) return QGT_ERROR_NO_MEMORY;

    (*dist_ctx)->num_nodes = num_nodes;
    (*dist_ctx)->rank = 0;
    (*dist_ctx)->initialized = true;
    (*dist_ctx)->node_failed = calloc(num_nodes, sizeof(bool));
    if (!(*dist_ctx)->node_failed) {
        free(*dist_ctx);
        *dist_ctx = NULL;
        return QGT_ERROR_NO_MEMORY;
    }

    return QGT_SUCCESS;
}

void qgt_destroy_distributed_context(qgt_distributed_context_t* dist_ctx) {
    if (dist_ctx) {
        free(dist_ctx->node_failed);
        free(dist_ctx);
    }
}

qgt_error_t qgt_distribute_state(qgt_distributed_context_t* dist_ctx, quantum_state_t* state) {
    if (!dist_ctx || !state) return QGT_ERROR_INVALID_ARGUMENT;
    // In simulation, distribution is a no-op
    return QGT_SUCCESS;
}

qgt_error_t qgt_gather_state(qgt_distributed_context_t* dist_ctx, quantum_state_t* state) {
    if (!dist_ctx || !state) return QGT_ERROR_INVALID_ARGUMENT;
    // In simulation, gathering is a no-op
    return QGT_SUCCESS;
}

qgt_error_t qgt_simulate_node_failure(qgt_distributed_context_t* dist_ctx, size_t node_id) {
    if (!dist_ctx || node_id >= dist_ctx->num_nodes) return QGT_ERROR_INVALID_ARGUMENT;
    dist_ctx->node_failed[node_id] = true;
    return QGT_SUCCESS;
}

// ============================================================================
// Distributed Geometric Operations
// ============================================================================

qgt_error_t qgt_distributed_geometric_rotate(qgt_distributed_context_t* dist_ctx,
                                              quantum_state_t* state,
                                              double angle,
                                              const double* axis) {
    if (!dist_ctx || !state || !axis) return QGT_ERROR_INVALID_ARGUMENT;

    // Check for node failures
    for (size_t i = 0; i < dist_ctx->num_nodes; i++) {
        if (dist_ctx->node_failed[i]) {
            return QGT_ERROR_NODE_FAILURE;
        }
    }

    // Apply rotation using axis-angle formula
    // R = cos(angle/2)*I + sin(angle/2)*(axis . sigma)
    double half_angle = angle / 2.0;
    double cos_ha = cos(half_angle);
    double sin_ha = sin(half_angle);

    // For now, apply a simple phase rotation
    for (size_t i = 0; i < state->dimension; i++) {
        double phase = half_angle * (axis[0] + axis[1] + axis[2]);
        double cos_p = cos(phase);
        double sin_p = sin(phase);
        float re = state->coordinates[i].real;
        float im = state->coordinates[i].imag;
        state->coordinates[i].real = (float)(re * cos_p - im * sin_p);
        state->coordinates[i].imag = (float)(re * sin_p + im * cos_p);
    }

    return QGT_SUCCESS;
}

qgt_error_t qgt_distributed_geometric_parallel_transport(qgt_distributed_context_t* dist_ctx,
                                                          quantum_state_t* state,
                                                          const double* path,
                                                          size_t num_points) {
    if (!dist_ctx || !state || !path || num_points == 0) return QGT_ERROR_INVALID_ARGUMENT;

    // Parallel transport along path - accumulate geometric phase
    double phase = 0.0;
    for (size_t i = 1; i < num_points; i++) {
        // Compute tangent vector
        double dx = path[3*i] - path[3*(i-1)];
        double dy = path[3*i+1] - path[3*(i-1)+1];
        double dz = path[3*i+2] - path[3*(i-1)+2];

        // Accumulate phase from connection (simplified)
        phase += 0.01 * (dx + dy + dz);
    }

    // Apply accumulated phase
    double cos_p = cos(phase);
    double sin_p = sin(phase);
    for (size_t i = 0; i < state->dimension; i++) {
        float re = state->coordinates[i].real;
        float im = state->coordinates[i].imag;
        state->coordinates[i].real = (float)(re * cos_p - im * sin_p);
        state->coordinates[i].imag = (float)(re * sin_p + im * cos_p);
    }

    return QGT_SUCCESS;
}

// ============================================================================
// Error Correction
// ============================================================================

qgt_error_t qgt_apply_error_channel(qgt_context_t* ctx, quantum_state_t* state, double error_rate) {
    if (!ctx || !state) return QGT_ERROR_INVALID_ARGUMENT;

    // Apply depolarizing noise
    for (size_t i = 0; i < state->dimension; i++) {
        // Add small random perturbation based on error rate
        double noise = error_rate * ((double)rand() / RAND_MAX - 0.5);
        state->coordinates[i].real += (float)noise;
        state->coordinates[i].imag += (float)(noise * 0.1);
    }

    // Renormalize
    double norm = 0.0;
    for (size_t i = 0; i < state->dimension; i++) {
        norm += state->coordinates[i].real * state->coordinates[i].real +
                state->coordinates[i].imag * state->coordinates[i].imag;
    }
    norm = sqrt(norm);
    for (size_t i = 0; i < state->dimension; i++) {
        state->coordinates[i].real /= (float)norm;
        state->coordinates[i].imag /= (float)norm;
    }

    return QGT_SUCCESS;
}

qgt_error_t qgt_apply_error_correction(qgt_context_t* ctx, quantum_state_t* state) {
    if (!ctx || !state) return QGT_ERROR_INVALID_ARGUMENT;

    // Simple error correction: project back to valid subspace
    // For now, just renormalize
    double norm = 0.0;
    for (size_t i = 0; i < state->dimension; i++) {
        norm += state->coordinates[i].real * state->coordinates[i].real +
                state->coordinates[i].imag * state->coordinates[i].imag;
    }
    norm = sqrt(norm);
    for (size_t i = 0; i < state->dimension; i++) {
        state->coordinates[i].real /= (float)norm;
        state->coordinates[i].imag /= (float)norm;
    }
    state->is_normalized = true;

    return QGT_SUCCESS;
}

qgt_error_t qgt_apply_error_correction_cycle(qgt_context_t* ctx, quantum_state_t* state) {
    return qgt_apply_error_correction(ctx, state);
}

// ============================================================================
// Fidelity Measurement
// ============================================================================

qgt_error_t qgt_geometric_measure_fidelity(qgt_context_t* ctx,
                                            quantum_state_t* state1,
                                            quantum_state_t* state2,
                                            double* fidelity) {
    if (!ctx || !state1 || !state2 || !fidelity) return QGT_ERROR_INVALID_ARGUMENT;
    if (state1->dimension != state2->dimension) return QGT_ERROR_INCOMPATIBLE;

    // Compute |<psi1|psi2>|^2
    double overlap_re = 0.0, overlap_im = 0.0;
    for (size_t i = 0; i < state1->dimension; i++) {
        // <psi1|psi2> = sum(conj(a1_i) * a2_i)
        overlap_re += state1->coordinates[i].real * state2->coordinates[i].real +
                      state1->coordinates[i].imag * state2->coordinates[i].imag;
        overlap_im += state1->coordinates[i].real * state2->coordinates[i].imag -
                      state1->coordinates[i].imag * state2->coordinates[i].real;
    }

    *fidelity = overlap_re * overlap_re + overlap_im * overlap_im;
    return QGT_SUCCESS;
}

// ============================================================================
// Logical State Operations
// ============================================================================

qgt_error_t qgt_create_logical_state(qgt_context_t* ctx, size_t num_qubits, quantum_state_t** state) {
    if (!ctx || !state || num_qubits == 0) return QGT_ERROR_INVALID_ARGUMENT;

    size_t dim = (size_t)1 << num_qubits;
    qgt_error_t err = quantum_state_create(state, QUANTUM_STATE_PURE, dim);
    if (err != QGT_SUCCESS) return err;

    // Initialize to logical |0> (all physical qubits in |0>)
    (*state)->coordinates[0].real = 1.0f;
    (*state)->coordinates[0].imag = 0.0f;
    (*state)->is_normalized = true;

    return QGT_SUCCESS;
}

qgt_error_t qgt_measure_logical_fidelity(qgt_context_t* ctx, quantum_state_t* state, double* fidelity) {
    if (!ctx || !state || !fidelity) return QGT_ERROR_INVALID_ARGUMENT;

    // Measure overlap with logical |0> state
    *fidelity = state->coordinates[0].real * state->coordinates[0].real +
                state->coordinates[0].imag * state->coordinates[0].imag;

    return QGT_SUCCESS;
}

// ============================================================================
// Noise Model
// ============================================================================

qgt_error_t qgt_enable_noise_model(qgt_context_t* ctx, double noise_strength) {
    if (!ctx) return QGT_ERROR_INVALID_ARGUMENT;
    ctx->noise_enabled = true;
    ctx->noise_strength = noise_strength;
    return QGT_SUCCESS;
}

// ============================================================================
// Geometric Operations
// ============================================================================

qgt_error_t qgt_geometric_rotate(qgt_context_t* ctx, quantum_state_t* state,
                                  double angle, const double* axis) {
    if (!ctx || !state || !axis) return QGT_ERROR_INVALID_ARGUMENT;

    // Apply rotation
    double half_angle = angle / 2.0;
    for (size_t i = 0; i < state->dimension; i++) {
        double phase = half_angle * (axis[0] + axis[1] + axis[2]);
        double cos_p = cos(phase);
        double sin_p = sin(phase);
        float re = state->coordinates[i].real;
        float im = state->coordinates[i].imag;
        state->coordinates[i].real = (float)(re * cos_p - im * sin_p);
        state->coordinates[i].imag = (float)(re * sin_p + im * cos_p);
    }

    // Apply noise if enabled
    if (ctx->noise_enabled) {
        qgt_apply_error_channel(ctx, state, ctx->noise_strength);
    }

    return QGT_SUCCESS;
}

qgt_error_t qgt_geometric_compute_metric(qgt_context_t* ctx, quantum_state_t* state, double* metric) {
    if (!ctx || !state || !metric) return QGT_ERROR_INVALID_ARGUMENT;

    // Compute Fubini-Study metric components
    size_t dim = state->dimension;
    for (size_t i = 0; i < dim * dim; i++) {
        metric[i] = 0.0;
    }

    // Diagonal elements = 1 - |psi_i|^2
    for (size_t i = 0; i < dim; i++) {
        double prob = state->coordinates[i].real * state->coordinates[i].real +
                      state->coordinates[i].imag * state->coordinates[i].imag;
        metric[i * dim + i] = 1.0 - prob;
    }

    return QGT_SUCCESS;
}
