/**
 * @file grover.c
 * @brief Grover's Quantum Search Algorithm implementation
 */

#include "quantum_geometric/algorithms/grover.h"
#include "quantum_geometric/core/quantum_geometric_logging.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Oracle Construction
// ============================================================================

grover_oracle_t* grover_create_function_oracle(grover_oracle_func_t func, void* user_data) {
    if (!func) return NULL;

    grover_oracle_t* oracle = calloc(1, sizeof(grover_oracle_t));
    if (!oracle) return NULL;

    oracle->type = GROVER_ORACLE_FUNCTION;
    oracle->func = func;
    oracle->user_data = user_data;

    return oracle;
}

grover_oracle_t* grover_create_bitstring_oracle(size_t target_state) {
    grover_oracle_t* oracle = calloc(1, sizeof(grover_oracle_t));
    if (!oracle) return NULL;

    oracle->type = GROVER_ORACLE_BITSTRING;
    oracle->target_state = target_state;
    oracle->num_targets = 1;

    return oracle;
}

grover_oracle_t* grover_create_set_oracle(const size_t* targets, size_t num_targets) {
    if (!targets || num_targets == 0) return NULL;

    grover_oracle_t* oracle = calloc(1, sizeof(grover_oracle_t));
    if (!oracle) return NULL;

    oracle->type = GROVER_ORACLE_SET;
    oracle->num_targets = num_targets;
    oracle->target_states = malloc(num_targets * sizeof(size_t));

    if (!oracle->target_states) {
        free(oracle);
        return NULL;
    }

    memcpy(oracle->target_states, targets, num_targets * sizeof(size_t));

    return oracle;
}

grover_oracle_t* grover_create_sat_oracle(int** clauses, size_t* clause_sizes,
                                           size_t num_clauses, size_t num_variables) {
    if (!clauses || !clause_sizes || num_clauses == 0 || num_variables == 0) {
        return NULL;
    }

    grover_oracle_t* oracle = calloc(1, sizeof(grover_oracle_t));
    if (!oracle) return NULL;

    oracle->type = GROVER_ORACLE_SAT;
    oracle->num_clauses = num_clauses;
    oracle->num_variables = num_variables;

    // Copy clauses
    oracle->clauses = malloc(num_clauses * sizeof(int*));
    oracle->clause_sizes = malloc(num_clauses * sizeof(size_t));

    if (!oracle->clauses || !oracle->clause_sizes) {
        grover_destroy_oracle(oracle);
        return NULL;
    }

    memcpy(oracle->clause_sizes, clause_sizes, num_clauses * sizeof(size_t));

    for (size_t i = 0; i < num_clauses; i++) {
        oracle->clauses[i] = malloc(clause_sizes[i] * sizeof(int));
        if (!oracle->clauses[i]) {
            grover_destroy_oracle(oracle);
            return NULL;
        }
        memcpy(oracle->clauses[i], clauses[i], clause_sizes[i] * sizeof(int));
    }

    return oracle;
}

void grover_destroy_oracle(grover_oracle_t* oracle) {
    if (!oracle) return;

    if (oracle->type == GROVER_ORACLE_SET) {
        free(oracle->target_states);
    } else if (oracle->type == GROVER_ORACLE_SAT) {
        if (oracle->clauses) {
            for (size_t i = 0; i < oracle->num_clauses; i++) {
                free(oracle->clauses[i]);
            }
            free(oracle->clauses);
        }
        free(oracle->clause_sizes);
    }

    free(oracle);
}

// Check if state satisfies SAT formula
static bool check_sat_assignment(const grover_oracle_t* oracle, size_t state) {
    for (size_t c = 0; c < oracle->num_clauses; c++) {
        bool clause_satisfied = false;

        for (size_t l = 0; l < oracle->clause_sizes[c]; l++) {
            int literal = oracle->clauses[c][l];
            int var_index = abs(literal) - 1;  // Variables are 1-indexed
            bool var_value = (state >> var_index) & 1;

            // Positive literal: var must be true
            // Negative literal: var must be false
            if ((literal > 0 && var_value) || (literal < 0 && !var_value)) {
                clause_satisfied = true;
                break;
            }
        }

        if (!clause_satisfied) {
            return false;  // At least one clause not satisfied
        }
    }

    return true;  // All clauses satisfied
}

bool grover_oracle_check(const grover_oracle_t* oracle, size_t state) {
    if (!oracle) return false;

    switch (oracle->type) {
        case GROVER_ORACLE_FUNCTION:
            return oracle->func(state, oracle->user_data);

        case GROVER_ORACLE_BITSTRING:
            return state == oracle->target_state;

        case GROVER_ORACLE_SET:
            for (size_t i = 0; i < oracle->num_targets; i++) {
                if (state == oracle->target_states[i]) {
                    return true;
                }
            }
            return false;

        case GROVER_ORACLE_SAT:
            return check_sat_assignment(oracle, state);

        default:
            return false;
    }
}

// ============================================================================
// Grover Algorithm Implementation
// ============================================================================

grover_config_t grover_default_config(size_t num_qubits) {
    grover_config_t config = {
        .num_qubits = num_qubits,
        .num_iterations = 0,    // Auto-compute
        .num_targets = 1,       // Assume single target
        .use_exact_count = false,
        .use_gpu = false,
        .backend = NULL
    };
    return config;
}

size_t grover_optimal_iterations(size_t N, size_t M) {
    if (M == 0 || M >= N) return 0;

    // Optimal iterations ≈ (π/4) * sqrt(N/M)
    double ratio = (double)N / (double)M;
    double iterations = (M_PI / 4.0) * sqrt(ratio);

    return (size_t)round(iterations);
}

grover_state_t* grover_init(const grover_oracle_t* oracle, const grover_config_t* config) {
    if (!oracle || !config || config->num_qubits == 0) return NULL;

    grover_state_t* state = calloc(1, sizeof(grover_state_t));
    if (!state) return NULL;

    state->config = *config;
    state->num_qubits = config->num_qubits;

    // Copy oracle
    state->oracle = calloc(1, sizeof(grover_oracle_t));
    if (!state->oracle) {
        free(state);
        return NULL;
    }
    memcpy(state->oracle, oracle, sizeof(grover_oracle_t));

    // Deep copy oracle data
    if (oracle->type == GROVER_ORACLE_SET && oracle->target_states) {
        state->oracle->target_states = malloc(oracle->num_targets * sizeof(size_t));
        if (!state->oracle->target_states) {
            grover_destroy(state);
            return NULL;
        }
        memcpy(state->oracle->target_states, oracle->target_states,
               oracle->num_targets * sizeof(size_t));
    }

    // Create quantum state
    size_t dim = (size_t)1 << state->num_qubits;
    state->qstate = calloc(1, sizeof(QuantumState));
    if (!state->qstate) {
        grover_destroy(state);
        return NULL;
    }

    state->qstate->num_qubits = state->num_qubits;
    state->qstate->dimension = dim;
    state->qstate->amplitudes = calloc(dim, sizeof(ComplexFloat));
    if (!state->qstate->amplitudes) {
        grover_destroy(state);
        return NULL;
    }

    // Compute optimal iterations
    size_t N = dim;
    size_t M = config->num_targets > 0 ? config->num_targets : 1;
    state->optimal_iterations = config->num_iterations > 0 ?
                                config->num_iterations :
                                grover_optimal_iterations(N, M);

    state->current_iteration = 0;

    return state;
}

qgt_error_t grover_prepare_superposition(grover_state_t* state) {
    if (!state || !state->qstate) return QGT_ERROR_INVALID_ARGUMENT;

    size_t dim = (size_t)1 << state->num_qubits;
    double amp = 1.0 / sqrt((double)dim);

    for (size_t i = 0; i < dim; i++) {
        state->qstate->amplitudes[i] = (ComplexFloat){(float)amp, 0.0f};
    }
    state->qstate->is_normalized = true;

    return QGT_SUCCESS;
}

qgt_error_t grover_apply_oracle(grover_state_t* state) {
    if (!state || !state->qstate || !state->oracle) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    size_t dim = (size_t)1 << state->num_qubits;
    ComplexFloat* amps = state->qstate->amplitudes;
    if (!amps) return QGT_ERROR_INVALID_ARGUMENT;

    // Oracle: |x> -> (-1)^f(x) |x>
    // If f(x) = 1 (marked state), flip phase
    for (size_t x = 0; x < dim; x++) {
        if (grover_oracle_check(state->oracle, x)) {
            amps[x].real = -amps[x].real;
            amps[x].imag = -amps[x].imag;
        }
    }

    return QGT_SUCCESS;
}

qgt_error_t grover_apply_diffusion(grover_state_t* state) {
    if (!state || !state->qstate) return QGT_ERROR_INVALID_ARGUMENT;

    size_t dim = (size_t)1 << state->num_qubits;
    ComplexFloat* amps = state->qstate->amplitudes;
    if (!amps) return QGT_ERROR_INVALID_ARGUMENT;

    // Diffusion operator: 2|s><s| - I
    // where |s> = (1/sqrt(N)) * sum_x |x>

    // Step 1: Compute mean amplitude
    double mean_real = 0.0;
    double mean_imag = 0.0;

    for (size_t x = 0; x < dim; x++) {
        mean_real += amps[x].real;
        mean_imag += amps[x].imag;
    }

    mean_real /= (double)dim;
    mean_imag /= (double)dim;

    // Step 2: Inversion about mean: 2*mean - a[x]
    for (size_t x = 0; x < dim; x++) {
        amps[x].real = (float)(2.0 * mean_real - amps[x].real);
        amps[x].imag = (float)(2.0 * mean_imag - amps[x].imag);
    }

    return QGT_SUCCESS;
}

qgt_error_t grover_apply_iteration(grover_state_t* state) {
    if (!state) return QGT_ERROR_INVALID_ARGUMENT;

    // Apply oracle
    qgt_error_t err = grover_apply_oracle(state);
    if (err != QGT_SUCCESS) return err;

    // Apply diffusion
    err = grover_apply_diffusion(state);
    if (err != QGT_SUCCESS) return err;

    state->current_iteration++;

    return QGT_SUCCESS;
}

qgt_error_t grover_success_probability(grover_state_t* state, double* prob) {
    if (!state || !state->qstate || !state->oracle || !prob) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    size_t dim = (size_t)1 << state->num_qubits;
    ComplexFloat* amps = state->qstate->amplitudes;
    if (!amps) return QGT_ERROR_INVALID_ARGUMENT;

    double total_prob = 0.0;

    for (size_t x = 0; x < dim; x++) {
        if (grover_oracle_check(state->oracle, x)) {
            double p = amps[x].real * amps[x].real + amps[x].imag * amps[x].imag;
            total_prob += p;
        }
    }

    *prob = total_prob;
    state->success_probability = total_prob;

    return QGT_SUCCESS;
}

qgt_error_t grover_measure(grover_state_t* state, size_t* result) {
    if (!state || !state->qstate || !result) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    size_t dim = (size_t)1 << state->num_qubits;
    ComplexFloat* amps = state->qstate->amplitudes;
    if (!amps) return QGT_ERROR_INVALID_ARGUMENT;

    // Compute cumulative probabilities
    double* cum_prob = malloc(dim * sizeof(double));
    if (!cum_prob) return QGT_ERROR_MEMORY_ALLOCATION;

    cum_prob[0] = amps[0].real * amps[0].real + amps[0].imag * amps[0].imag;
    for (size_t i = 1; i < dim; i++) {
        double p = amps[i].real * amps[i].real + amps[i].imag * amps[i].imag;
        cum_prob[i] = cum_prob[i-1] + p;
    }

    // Sample
    double r = (double)rand() / RAND_MAX;
    size_t measured = 0;

    for (size_t i = 0; i < dim; i++) {
        if (r <= cum_prob[i]) {
            measured = i;
            break;
        }
    }

    free(cum_prob);

    *result = measured;

    // Collapse state
    for (size_t i = 0; i < dim; i++) {
        if (i == measured) {
            amps[i] = (ComplexFloat){1.0f, 0.0f};
        } else {
            amps[i] = (ComplexFloat){0.0f, 0.0f};
        }
    }

    return QGT_SUCCESS;
}

grover_result_t* grover_search(grover_state_t* state) {
    if (!state) return NULL;

    grover_result_t* result = calloc(1, sizeof(grover_result_t));
    if (!result) return NULL;

    clock_t start_time = clock();

    // Prepare superposition
    qgt_error_t err = grover_prepare_superposition(state);
    if (err != QGT_SUCCESS) {
        free(result);
        return NULL;
    }

    // Apply Grover iterations
    for (size_t iter = 0; iter < state->optimal_iterations; iter++) {
        err = grover_apply_iteration(state);
        if (err != QGT_SUCCESS) {
            free(result);
            return NULL;
        }
    }

    // Get success probability before measurement
    grover_success_probability(state, &result->final_probability);

    // Measure
    size_t measured;
    err = grover_measure(state, &measured);
    if (err != QGT_SUCCESS) {
        free(result);
        return NULL;
    }

    result->found_state = measured;
    result->is_target = grover_oracle_check(state->oracle, measured);
    result->num_iterations = state->optimal_iterations;
    result->num_oracle_calls = state->optimal_iterations;

    clock_t end_time = clock();
    result->execution_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    return result;
}

void grover_destroy(grover_state_t* state) {
    if (!state) return;

    if (state->oracle) {
        if (state->oracle->type == GROVER_ORACLE_SET) {
            free(state->oracle->target_states);
        }
        free(state->oracle);
    }

    if (state->qstate) {
        free(state->qstate->amplitudes);
        free(state->qstate->workspace);
        free(state->qstate);
    }

    free(state->measured_states);
    free(state);
}

void grover_destroy_result(grover_result_t* result) {
    free(result);
}

// ============================================================================
// Amplitude Amplification
// ============================================================================

grover_result_t* amplitude_amplification(
    qgt_error_t (*prepare_func)(QuantumState*),
    const grover_oracle_t* oracle,
    size_t num_qubits,
    size_t num_iterations
) {
    if (!prepare_func || !oracle || num_qubits == 0) return NULL;

    // Create state
    size_t dim = (size_t)1 << num_qubits;

    QuantumState* qstate = calloc(1, sizeof(QuantumState));
    if (!qstate) return NULL;

    qstate->num_qubits = num_qubits;
    qstate->dimension = dim;
    qstate->amplitudes = calloc(dim, sizeof(ComplexFloat));
    if (!qstate->amplitudes) {
        free(qstate);
        return NULL;
    }

    grover_result_t* result = calloc(1, sizeof(grover_result_t));
    if (!result) {
        free(qstate->amplitudes);
        free(qstate);
        return NULL;
    }

    clock_t start_time = clock();

    // Prepare initial state using custom function
    qgt_error_t err = prepare_func(qstate);
    if (err != QGT_SUCCESS) {
        free(qstate->amplitudes);
        free(qstate);
        free(result);
        return NULL;
    }

    // Store initial amplitudes for reflection
    ComplexFloat* initial_amps = malloc(dim * sizeof(ComplexFloat));
    if (!initial_amps) {
        free(qstate->amplitudes);
        free(qstate);
        free(result);
        return NULL;
    }
    memcpy(initial_amps, qstate->amplitudes, dim * sizeof(ComplexFloat));

    ComplexFloat* amps = qstate->amplitudes;

    // Apply amplitude amplification iterations
    for (size_t iter = 0; iter < num_iterations; iter++) {
        // Apply oracle (phase flip on marked states)
        for (size_t x = 0; x < dim; x++) {
            if (grover_oracle_check(oracle, x)) {
                amps[x].real = -amps[x].real;
                amps[x].imag = -amps[x].imag;
            }
        }

        // Apply reflection about initial state: 2|psi><psi| - I
        // This requires: S = 2|psi><psi| - I
        // S|phi> = 2<psi|phi>|psi> - |phi>

        // Compute <psi|current>
        double dot_real = 0.0, dot_imag = 0.0;
        for (size_t x = 0; x < dim; x++) {
            // <psi|current> = sum_x psi_x* current_x
            dot_real += initial_amps[x].real * amps[x].real + initial_amps[x].imag * amps[x].imag;
            dot_imag += initial_amps[x].real * amps[x].imag - initial_amps[x].imag * amps[x].real;
        }

        // Apply reflection
        for (size_t x = 0; x < dim; x++) {
            // new = 2 * <psi|current> * psi - current
            double new_real = 2.0 * (dot_real * initial_amps[x].real - dot_imag * initial_amps[x].imag) - amps[x].real;
            double new_imag = 2.0 * (dot_real * initial_amps[x].imag + dot_imag * initial_amps[x].real) - amps[x].imag;
            amps[x].real = (float)new_real;
            amps[x].imag = (float)new_imag;
        }
    }

    // Compute success probability
    double success_prob = 0.0;
    for (size_t x = 0; x < dim; x++) {
        if (grover_oracle_check(oracle, x)) {
            double p = amps[x].real * amps[x].real + amps[x].imag * amps[x].imag;
            success_prob += p;
        }
    }

    // Measure
    double* cum_prob = malloc(dim * sizeof(double));
    if (cum_prob) {
        cum_prob[0] = amps[0].real * amps[0].real + amps[0].imag * amps[0].imag;
        for (size_t i = 1; i < dim; i++) {
            double p = amps[i].real * amps[i].real + amps[i].imag * amps[i].imag;
            cum_prob[i] = cum_prob[i-1] + p;
        }

        double r = (double)rand() / RAND_MAX;
        result->found_state = 0;
        for (size_t i = 0; i < dim; i++) {
            if (r <= cum_prob[i]) {
                result->found_state = i;
                break;
            }
        }
        free(cum_prob);
    }

    result->is_target = grover_oracle_check(oracle, result->found_state);
    result->num_iterations = num_iterations;
    result->num_oracle_calls = num_iterations;
    result->final_probability = success_prob;

    clock_t end_time = clock();
    result->execution_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    free(initial_amps);
    free(qstate->amplitudes);
    free(qstate);

    return result;
}

// ============================================================================
// Utility Functions
// ============================================================================

void grover_print_state(const grover_state_t* state) {
    if (!state) return;

    geometric_log_info("Grover State:");
    geometric_log_info("  Qubits: %zu", state->num_qubits);
    geometric_log_info("  Search space: %zu", (size_t)1 << state->num_qubits);
    geometric_log_info("  Optimal iterations: %zu", state->optimal_iterations);
    geometric_log_info("  Current iteration: %zu", state->current_iteration);
    geometric_log_info("  Success probability: %.4f", state->success_probability);
}

void grover_print_result(const grover_result_t* result) {
    if (!result) return;

    geometric_log_info("Grover Result:");
    geometric_log_info("  Found state: %zu", result->found_state);
    geometric_log_info("  Is target: %s", result->is_target ? "yes" : "no");
    geometric_log_info("  Iterations: %zu", result->num_iterations);
    geometric_log_info("  Oracle calls: %zu", result->num_oracle_calls);
    geometric_log_info("  Final probability: %.4f", result->final_probability);
    geometric_log_info("  Execution time: %.3f seconds", result->execution_time);
}
