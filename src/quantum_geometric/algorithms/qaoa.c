/**
 * @file qaoa.c
 * @brief QAOA (Quantum Approximate Optimization Algorithm) implementation
 */

#include "quantum_geometric/algorithms/qaoa.h"
#include "quantum_geometric/core/quantum_state.h"
#include "quantum_geometric/core/quantum_gate_operations.h"
#include "quantum_geometric/core/quantum_geometric_logging.h"
#include "quantum_geometric/core/error_codes.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Graph Construction
// ============================================================================

qaoa_graph_t* qaoa_create_graph(size_t num_vertices) {
    if (num_vertices == 0) return NULL;

    qaoa_graph_t* graph = calloc(1, sizeof(qaoa_graph_t));
    if (!graph) return NULL;

    graph->num_vertices = num_vertices;
    graph->num_edges = 0;
    graph->edges = NULL;
    graph->vertex_weights = calloc(num_vertices, sizeof(double));

    if (!graph->vertex_weights) {
        free(graph);
        return NULL;
    }

    return graph;
}

qgt_error_t qaoa_add_edge(qaoa_graph_t* graph, size_t i, size_t j, double weight) {
    if (!graph) return QGT_ERROR_INVALID_ARGUMENT;
    if (i >= graph->num_vertices || j >= graph->num_vertices) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    if (i == j) return QGT_ERROR_INVALID_ARGUMENT;  // No self-loops

    // Resize edge array
    qaoa_edge_t* new_edges = realloc(graph->edges,
                                      (graph->num_edges + 1) * sizeof(qaoa_edge_t));
    if (!new_edges) return QGT_ERROR_MEMORY_ALLOCATION;

    graph->edges = new_edges;
    graph->edges[graph->num_edges].i = i;
    graph->edges[graph->num_edges].j = j;
    graph->edges[graph->num_edges].weight = weight;
    graph->num_edges++;

    return QGT_SUCCESS;
}

qgt_error_t qaoa_set_vertex_weight(qaoa_graph_t* graph, size_t vertex, double weight) {
    if (!graph || vertex >= graph->num_vertices) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    graph->vertex_weights[vertex] = weight;
    return QGT_SUCCESS;
}

qaoa_graph_t* qaoa_create_random_graph(size_t num_vertices, double edge_probability,
                                        double min_weight, double max_weight) {
    qaoa_graph_t* graph = qaoa_create_graph(num_vertices);
    if (!graph) return NULL;

    srand((unsigned int)time(NULL));

    for (size_t i = 0; i < num_vertices; i++) {
        for (size_t j = i + 1; j < num_vertices; j++) {
            double r = (double)rand() / RAND_MAX;
            if (r < edge_probability) {
                double weight = min_weight + (max_weight - min_weight) * ((double)rand() / RAND_MAX);
                qaoa_add_edge(graph, i, j, weight);
            }
        }
    }

    return graph;
}

qaoa_graph_t* qaoa_create_from_adjacency(const double* adjacency, size_t n) {
    if (!adjacency || n == 0) return NULL;

    qaoa_graph_t* graph = qaoa_create_graph(n);
    if (!graph) return NULL;

    for (size_t i = 0; i < n; i++) {
        for (size_t j = i + 1; j < n; j++) {
            double weight = adjacency[i * n + j];
            if (weight != 0.0) {
                qaoa_add_edge(graph, i, j, weight);
            }
        }
    }

    return graph;
}

void qaoa_destroy_graph(qaoa_graph_t* graph) {
    if (!graph) return;
    free(graph->edges);
    free(graph->vertex_weights);
    free(graph);
}

// ============================================================================
// QAOA Initialization
// ============================================================================

qaoa_config_t qaoa_default_config(size_t p) {
    qaoa_config_t config = {
        .p = p,
        .problem_type = QAOA_PROBLEM_MAXCUT,
        .mixer_type = QAOA_MIXER_X,
        .optimizer_type = QAOA_OPTIMIZER_COBYLA,
        .initial_gamma = NULL,
        .initial_beta = NULL,
        .max_iterations = 1000,
        .tolerance = 1e-6,
        .learning_rate = 0.01,
        .num_shots = 1024,
        .use_expectation = true,
        .use_gpu = false,
        .backend = NULL
    };
    return config;
}

qaoa_state_t* qaoa_init(const qaoa_graph_t* graph, const qaoa_config_t* config) {
    if (!graph || !config || config->p == 0) return NULL;

    qaoa_state_t* state = calloc(1, sizeof(qaoa_state_t));
    if (!state) return NULL;

    // Copy configuration
    state->config = *config;
    state->p = config->p;
    state->num_qubits = graph->num_vertices;

    // Copy graph
    state->graph = qaoa_create_graph(graph->num_vertices);
    if (!state->graph) {
        free(state);
        return NULL;
    }
    for (size_t e = 0; e < graph->num_edges; e++) {
        qaoa_add_edge(state->graph, graph->edges[e].i, graph->edges[e].j,
                      graph->edges[e].weight);
    }
    if (graph->vertex_weights) {
        memcpy(state->graph->vertex_weights, graph->vertex_weights,
               graph->num_vertices * sizeof(double));
    }

    // Allocate parameters
    state->gamma = calloc(config->p, sizeof(double));
    state->beta = calloc(config->p, sizeof(double));
    state->best_gamma = calloc(config->p, sizeof(double));
    state->best_beta = calloc(config->p, sizeof(double));

    if (!state->gamma || !state->beta || !state->best_gamma || !state->best_beta) {
        qaoa_destroy(state);
        return NULL;
    }

    // Initialize parameters
    if (config->initial_gamma && config->initial_beta) {
        memcpy(state->gamma, config->initial_gamma, config->p * sizeof(double));
        memcpy(state->beta, config->initial_beta, config->p * sizeof(double));
    } else {
        // Random initialization in [0, 2*pi] for gamma, [0, pi] for beta
        srand((unsigned int)time(NULL));
        for (size_t i = 0; i < config->p; i++) {
            state->gamma[i] = 2.0 * M_PI * ((double)rand() / RAND_MAX);
            state->beta[i] = M_PI * ((double)rand() / RAND_MAX);
        }
    }

    // Create quantum state (|+>^n)
    size_t dim = (size_t)1 << state->num_qubits;
    state->qstate = calloc(1, sizeof(QuantumState));
    if (!state->qstate) {
        qaoa_destroy(state);
        return NULL;
    }
    state->qstate->num_qubits = state->num_qubits;
    state->qstate->dimension = dim;
    state->qstate->amplitudes = calloc(dim, sizeof(ComplexFloat));
    if (!state->qstate->amplitudes) {
        qaoa_destroy(state);
        return NULL;
    }
    state->qstate->is_normalized = false;

    // Allocate solution storage
    state->best_solution = calloc(state->num_qubits, sizeof(int));
    state->solution_probabilities = calloc(dim, sizeof(double));

    if (!state->best_solution || !state->solution_probabilities) {
        qaoa_destroy(state);
        return NULL;
    }

    // Construct Hamiltonians
    if (qaoa_construct_cost_hamiltonian(state) != QGT_SUCCESS) {
        qaoa_destroy(state);
        return NULL;
    }

    if (qaoa_construct_mixer_hamiltonian(state) != QGT_SUCCESS) {
        qaoa_destroy(state);
        return NULL;
    }

    state->best_cost = INFINITY;
    state->iteration = 0;

    return state;
}

// ============================================================================
// Hamiltonian Construction
// ============================================================================

qgt_error_t qaoa_construct_cost_hamiltonian(qaoa_state_t* state) {
    if (!state || !state->graph) return QGT_ERROR_INVALID_ARGUMENT;

    // For MaxCut: H_C = sum_{(i,j)} w_ij * (1 - Z_i Z_j) / 2
    //           = const - sum_{(i,j)} w_ij * Z_i Z_j / 2
    // The constant shifts energy but doesn't affect optimization

    size_t dim = (size_t)1 << state->num_qubits;

    // Allocate diagonal operator for cost Hamiltonian
    // Cost Hamiltonian for MaxCut is diagonal in computational basis
    quantum_operator_t* H_C = calloc(1, sizeof(quantum_operator_t));
    if (!H_C) return QGT_ERROR_MEMORY_ALLOCATION;

    H_C->type = QUANTUM_OPERATOR_HERMITIAN;
    H_C->dimension = dim;
    H_C->is_hermitian = true;
    H_C->matrix = calloc(dim, sizeof(ComplexFloat));  // Diagonal elements only
    if (!H_C->matrix) {
        free(H_C);
        return QGT_ERROR_MEMORY_ALLOCATION;
    }

    // Compute diagonal elements
    // For each computational basis state |z>, compute H_C|z> = E_z|z>
    for (size_t z = 0; z < dim; z++) {
        double energy = 0.0;

        // Sum over edges
        for (size_t e = 0; e < state->graph->num_edges; e++) {
            size_t i = state->graph->edges[e].i;
            size_t j = state->graph->edges[e].j;
            double w = state->graph->edges[e].weight;

            // Get bit values at positions i and j
            int zi = (z >> i) & 1;
            int zj = (z >> j) & 1;

            // Z_i Z_j eigenvalue is +1 if zi == zj, -1 otherwise
            // (1 - Z_i Z_j)/2 = 1 if zi != zj (edge is cut), 0 otherwise
            if (zi != zj) {
                energy += w;  // Edge is cut
            }
        }

        // Add vertex weights (linear terms)
        for (size_t i = 0; i < state->num_qubits; i++) {
            int zi = (z >> i) & 1;
            // Z_i eigenvalue: +1 if z_i = 0, -1 if z_i = 1
            int z_eigenvalue = 1 - 2 * zi;
            energy += state->graph->vertex_weights[i] * z_eigenvalue;
        }

        H_C->matrix[z] = (ComplexFloat){(float)energy, 0.0f};
    }

    state->cost_hamiltonian = H_C;
    return QGT_SUCCESS;
}

qgt_error_t qaoa_construct_mixer_hamiltonian(qaoa_state_t* state) {
    if (!state) return QGT_ERROR_INVALID_ARGUMENT;

    // For X-mixer: H_M = sum_i X_i
    // This is NOT diagonal, but we can represent its action
    // exp(-i*beta*H_M) = prod_i exp(-i*beta*X_i) = prod_i RX(2*beta)

    size_t dim = (size_t)1 << state->num_qubits;

    quantum_operator_t* H_M = calloc(1, sizeof(quantum_operator_t));
    if (!H_M) return QGT_ERROR_MEMORY_ALLOCATION;

    H_M->type = QUANTUM_OPERATOR_HERMITIAN;
    H_M->dimension = dim;
    H_M->is_hermitian = true;
    H_M->matrix = NULL;   // We apply it directly via RX gates

    state->mixer_hamiltonian = H_M;
    return QGT_SUCCESS;
}

// ============================================================================
// QAOA Circuit Operations
// ============================================================================

qgt_error_t qaoa_prepare_initial_state(qaoa_state_t* state) {
    if (!state || !state->qstate) return QGT_ERROR_INVALID_ARGUMENT;

    size_t dim = (size_t)1 << state->num_qubits;

    // Prepare |+>^n = H^n |0>^n = (1/sqrt(2^n)) * sum_z |z>
    double amp = 1.0 / sqrt((double)dim);

    // Initialize amplitudes
    if (!state->qstate->amplitudes) {
        state->qstate->amplitudes = calloc(dim, sizeof(ComplexFloat));
        if (!state->qstate->amplitudes) return QGT_ERROR_MEMORY_ALLOCATION;
    }

    for (size_t i = 0; i < dim; i++) {
        state->qstate->amplitudes[i] = (ComplexFloat){(float)amp, 0.0f};
    }
    state->qstate->is_normalized = true;

    return QGT_SUCCESS;
}

// Apply exp(-i * gamma * H_C) where H_C is diagonal
static qgt_error_t apply_cost_evolution(qaoa_state_t* state, double gamma) {
    if (!state || !state->qstate || !state->cost_hamiltonian) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    size_t dim = (size_t)1 << state->num_qubits;

    // Get state amplitudes
    ComplexFloat* amps = state->qstate->amplitudes;
    if (!amps) return QGT_ERROR_INVALID_ARGUMENT;

    // Apply diagonal unitary: |z> -> exp(-i * gamma * E_z) |z>
    for (size_t z = 0; z < dim; z++) {
        double E_z = state->cost_hamiltonian->matrix[z].real;
        double phase = -gamma * E_z;

        // Multiply amplitude by exp(i * phase)
        double cos_phase = cos(phase);
        double sin_phase = sin(phase);

        float real = amps[z].real;
        float imag = amps[z].imag;

        amps[z].real = (float)(real * cos_phase - imag * sin_phase);
        amps[z].imag = (float)(real * sin_phase + imag * cos_phase);
    }

    return QGT_SUCCESS;
}

// Apply exp(-i * beta * H_M) = prod_i RX(2*beta)
// Optimized in-place implementation - no extra memory allocation needed
static qgt_error_t apply_mixer_evolution(qaoa_state_t* state, double beta) {
    if (!state || !state->qstate) return QGT_ERROR_INVALID_ARGUMENT;

    size_t dim = (size_t)1 << state->num_qubits;
    ComplexFloat* amps = state->qstate->amplitudes;
    if (!amps) return QGT_ERROR_INVALID_ARGUMENT;

    // Apply RX(2*beta) to each qubit
    // RX(theta) = cos(theta/2) I - i*sin(theta/2) X
    double theta = 2.0 * beta;
    float c = (float)cos(theta / 2.0);
    float s = (float)sin(theta / 2.0);

    // Apply RX to each qubit in-place
    // Since RX only mixes pairs |...0_q...⟩ ↔ |...1_q...⟩,
    // we can apply it in-place by processing each pair once
    for (size_t q = 0; q < state->num_qubits; q++) {
        size_t mask = (size_t)1 << q;

        for (size_t z = 0; z < dim; z++) {
            // Only process when qubit q is 0 (to avoid double-counting pairs)
            if ((z & mask) == 0) {
                size_t z1 = z | mask;  // Partner state with qubit q = 1

                ComplexFloat a0 = amps[z];
                ComplexFloat a1 = amps[z1];

                // RX(theta): |0⟩ → cos(θ/2)|0⟩ - i·sin(θ/2)|1⟩
                //            |1⟩ → -i·sin(θ/2)|0⟩ + cos(θ/2)|1⟩
                amps[z].real = c * a0.real + s * a1.imag;
                amps[z].imag = c * a0.imag - s * a1.real;

                amps[z1].real = s * a0.imag + c * a1.real;
                amps[z1].imag = -s * a0.real + c * a1.imag;
            }
        }
    }

    return QGT_SUCCESS;
}

qgt_error_t qaoa_apply_layer(qaoa_state_t* state, size_t layer_idx) {
    if (!state || layer_idx >= state->p) return QGT_ERROR_INVALID_ARGUMENT;

    // Apply cost evolution
    qgt_error_t err = apply_cost_evolution(state, state->gamma[layer_idx]);
    if (err != QGT_SUCCESS) return err;

    // Apply mixer evolution
    err = apply_mixer_evolution(state, state->beta[layer_idx]);
    return err;
}

qgt_error_t qaoa_apply_circuit(qaoa_state_t* state,
                               const double* gamma,
                               const double* beta) {
    if (!state || !gamma || !beta) return QGT_ERROR_INVALID_ARGUMENT;

    // Update parameters
    memcpy(state->gamma, gamma, state->p * sizeof(double));
    memcpy(state->beta, beta, state->p * sizeof(double));

    // Prepare initial state
    qgt_error_t err = qaoa_prepare_initial_state(state);
    if (err != QGT_SUCCESS) return err;

    // Apply all layers
    for (size_t layer = 0; layer < state->p; layer++) {
        err = qaoa_apply_layer(state, layer);
        if (err != QGT_SUCCESS) return err;
    }

    return QGT_SUCCESS;
}

// ============================================================================
// Expectation Value Computation
// ============================================================================

qgt_error_t qaoa_compute_expectation(qaoa_state_t* state, double* expectation) {
    if (!state || !state->qstate || !state->cost_hamiltonian || !expectation) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    size_t dim = (size_t)1 << state->num_qubits;
    ComplexFloat* amps = state->qstate->amplitudes;
    if (!amps) return QGT_ERROR_INVALID_ARGUMENT;

    // <psi|H_C|psi> = sum_z |a_z|^2 * E_z
    double exp_val = 0.0;
    for (size_t z = 0; z < dim; z++) {
        double prob = amps[z].real * amps[z].real + amps[z].imag * amps[z].imag;
        double E_z = state->cost_hamiltonian->matrix[z].real;
        exp_val += prob * E_z;
    }

    *expectation = exp_val;
    state->current_cost = exp_val;

    // Update best if this is better (for maximization, we want higher values)
    if (exp_val > state->best_cost) {
        state->best_cost = exp_val;
        memcpy(state->best_gamma, state->gamma, state->p * sizeof(double));
        memcpy(state->best_beta, state->beta, state->p * sizeof(double));
    }

    return QGT_SUCCESS;
}

// ============================================================================
// Gradient Computation (Parameter Shift Rule)
// ============================================================================

qgt_error_t qaoa_compute_gradient(qaoa_state_t* state,
                                  double* gamma_grad,
                                  double* beta_grad) {
    if (!state || !gamma_grad || !beta_grad) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    const double SHIFT = M_PI / 2.0;
    double* gamma_temp = malloc(state->p * sizeof(double));
    double* beta_temp = malloc(state->p * sizeof(double));

    if (!gamma_temp || !beta_temp) {
        free(gamma_temp);
        free(beta_temp);
        return QGT_ERROR_MEMORY_ALLOCATION;
    }

    memcpy(gamma_temp, state->gamma, state->p * sizeof(double));
    memcpy(beta_temp, state->beta, state->p * sizeof(double));

    // Compute gradient for gamma parameters
    for (size_t i = 0; i < state->p; i++) {
        double exp_plus, exp_minus;

        // f(gamma + pi/2)
        gamma_temp[i] = state->gamma[i] + SHIFT;
        qaoa_apply_circuit(state, gamma_temp, beta_temp);
        qaoa_compute_expectation(state, &exp_plus);

        // f(gamma - pi/2)
        gamma_temp[i] = state->gamma[i] - SHIFT;
        qaoa_apply_circuit(state, gamma_temp, beta_temp);
        qaoa_compute_expectation(state, &exp_minus);

        // Gradient = (f(x+s) - f(x-s)) / 2
        gamma_grad[i] = (exp_plus - exp_minus) / 2.0;

        // Restore
        gamma_temp[i] = state->gamma[i];
    }

    // Compute gradient for beta parameters
    for (size_t i = 0; i < state->p; i++) {
        double exp_plus, exp_minus;

        // f(beta + pi/2)
        beta_temp[i] = state->beta[i] + SHIFT;
        qaoa_apply_circuit(state, gamma_temp, beta_temp);
        qaoa_compute_expectation(state, &exp_plus);

        // f(beta - pi/2)
        beta_temp[i] = state->beta[i] - SHIFT;
        qaoa_apply_circuit(state, gamma_temp, beta_temp);
        qaoa_compute_expectation(state, &exp_minus);

        beta_grad[i] = (exp_plus - exp_minus) / 2.0;

        // Restore
        beta_temp[i] = state->beta[i];
    }

    // Restore original parameters
    memcpy(state->gamma, gamma_temp, state->p * sizeof(double));
    memcpy(state->beta, beta_temp, state->p * sizeof(double));

    free(gamma_temp);
    free(beta_temp);

    return QGT_SUCCESS;
}

// ============================================================================
// Optimization
// ============================================================================

qaoa_result_t* qaoa_optimize(qaoa_state_t* state) {
    if (!state) return NULL;

    qaoa_result_t* result = calloc(1, sizeof(qaoa_result_t));
    if (!result) return NULL;

    result->optimal_gamma = calloc(state->p, sizeof(double));
    result->optimal_beta = calloc(state->p, sizeof(double));
    result->optimal_solution = calloc(state->num_qubits, sizeof(int));
    result->cost_history = calloc(state->config.max_iterations, sizeof(double));

    if (!result->optimal_gamma || !result->optimal_beta ||
        !result->optimal_solution || !result->cost_history) {
        qaoa_destroy_result(result);
        return NULL;
    }

    clock_t start_time = clock();

    double* gamma_grad = malloc(state->p * sizeof(double));
    double* beta_grad = malloc(state->p * sizeof(double));

    if (!gamma_grad || !beta_grad) {
        free(gamma_grad);
        free(beta_grad);
        qaoa_destroy_result(result);
        return NULL;
    }

    double prev_cost = -INFINITY;
    double learning_rate = state->config.learning_rate;

    for (size_t iter = 0; iter < state->config.max_iterations; iter++) {
        state->iteration = iter;

        // Apply circuit with current parameters
        qgt_error_t err = qaoa_apply_circuit(state, state->gamma, state->beta);
        if (err != QGT_SUCCESS) break;

        // Compute expectation
        double cost;
        err = qaoa_compute_expectation(state, &cost);
        if (err != QGT_SUCCESS) break;

        result->cost_history[iter] = cost;
        result->history_length = iter + 1;

        // Check convergence
        if (fabs(cost - prev_cost) < state->config.tolerance) {
            break;
        }
        prev_cost = cost;

        // Compute gradient
        err = qaoa_compute_gradient(state, gamma_grad, beta_grad);
        if (err != QGT_SUCCESS) break;

        // Update parameters (gradient ascent for maximization)
        for (size_t i = 0; i < state->p; i++) {
            state->gamma[i] += learning_rate * gamma_grad[i];
            state->beta[i] += learning_rate * beta_grad[i];
        }
    }

    free(gamma_grad);
    free(beta_grad);

    // Store final results
    memcpy(result->optimal_gamma, state->best_gamma, state->p * sizeof(double));
    memcpy(result->optimal_beta, state->best_beta, state->p * sizeof(double));
    result->optimal_cost = state->best_cost;
    result->num_iterations = state->iteration + 1;

    // Find best solution from final state
    qaoa_apply_circuit(state, state->best_gamma, state->best_beta);

    size_t dim = (size_t)1 << state->num_qubits;
    ComplexFloat* amps = state->qstate->amplitudes;
    double max_prob = 0.0;
    size_t best_z = 0;

    for (size_t z = 0; z < dim; z++) {
        double prob = amps[z].real * amps[z].real + amps[z].imag * amps[z].imag;
        if (prob > max_prob) {
            max_prob = prob;
            best_z = z;
        }
    }

    // Convert best_z to solution bitstring
    for (size_t i = 0; i < state->num_qubits; i++) {
        result->optimal_solution[i] = (best_z >> i) & 1;
        state->best_solution[i] = result->optimal_solution[i];
    }

    clock_t end_time = clock();
    result->execution_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    return result;
}

// ============================================================================
// Sampling
// ============================================================================

qgt_error_t qaoa_sample(qaoa_state_t* state, int** samples, size_t num_samples) {
    if (!state || !samples || num_samples == 0) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    size_t dim = (size_t)1 << state->num_qubits;
    ComplexFloat* amps = state->qstate->amplitudes;
    if (!amps) return QGT_ERROR_INVALID_ARGUMENT;

    // Compute cumulative probabilities
    double* cum_prob = malloc(dim * sizeof(double));
    if (!cum_prob) return QGT_ERROR_MEMORY_ALLOCATION;

    cum_prob[0] = amps[0].real * amps[0].real + amps[0].imag * amps[0].imag;
    for (size_t z = 1; z < dim; z++) {
        double prob = amps[z].real * amps[z].real + amps[z].imag * amps[z].imag;
        cum_prob[z] = cum_prob[z-1] + prob;
    }

    // Sample
    for (size_t s = 0; s < num_samples; s++) {
        samples[s] = malloc(state->num_qubits * sizeof(int));
        if (!samples[s]) {
            for (size_t i = 0; i < s; i++) free(samples[i]);
            free(cum_prob);
            return QGT_ERROR_MEMORY_ALLOCATION;
        }

        double r = (double)rand() / RAND_MAX;

        // Binary search for sampled state
        size_t sampled_z = 0;
        for (size_t z = 0; z < dim; z++) {
            if (r <= cum_prob[z]) {
                sampled_z = z;
                break;
            }
        }

        // Convert to bitstring
        for (size_t i = 0; i < state->num_qubits; i++) {
            samples[s][i] = (sampled_z >> i) & 1;
        }
    }

    free(cum_prob);
    return QGT_SUCCESS;
}

// ============================================================================
// Solution Evaluation
// ============================================================================

double qaoa_evaluate_solution(const qaoa_graph_t* graph, const int* solution) {
    if (!graph || !solution) return 0.0;

    double cost = 0.0;

    // Count cut edges
    for (size_t e = 0; e < graph->num_edges; e++) {
        size_t i = graph->edges[e].i;
        size_t j = graph->edges[e].j;
        double w = graph->edges[e].weight;

        if (solution[i] != solution[j]) {
            cost += w;  // Edge is cut
        }
    }

    return cost;
}

// ============================================================================
// Cleanup
// ============================================================================

void qaoa_destroy(qaoa_state_t* state) {
    if (!state) return;

    qaoa_destroy_graph(state->graph);

    if (state->cost_hamiltonian) {
        free(state->cost_hamiltonian->matrix);
        free(state->cost_hamiltonian);
    }

    if (state->mixer_hamiltonian) {
        free(state->mixer_hamiltonian);
    }

    if (state->qstate) {
        free(state->qstate->amplitudes);
        free(state->qstate->workspace);
        free(state->qstate);
    }

    free(state->gamma);
    free(state->beta);
    free(state->best_gamma);
    free(state->best_beta);
    free(state->best_solution);
    free(state->solution_probabilities);
    free(state);
}

void qaoa_destroy_result(qaoa_result_t* result) {
    if (!result) return;

    free(result->optimal_gamma);
    free(result->optimal_beta);
    free(result->optimal_solution);
    free(result->cost_history);
    free(result);
}

// ============================================================================
// Utility Functions
// ============================================================================

size_t qaoa_estimate_optimal_p(size_t num_vertices, size_t num_edges) {
    // Heuristic: p grows logarithmically with problem size
    // For dense graphs, need more layers
    double density = (double)num_edges / (num_vertices * (num_vertices - 1) / 2);
    size_t base_p = (size_t)(log2((double)num_vertices) + 1);
    return base_p + (size_t)(density * base_p);
}

double qaoa_approximation_ratio(const qaoa_graph_t* graph,
                                const int* solution,
                                double optimal_cost) {
    if (!graph || !solution || optimal_cost == 0.0) return 0.0;

    double achieved = qaoa_evaluate_solution(graph, solution);
    return achieved / optimal_cost;
}

void qaoa_print_state(const qaoa_state_t* state) {
    if (!state) return;

    geometric_log_info("QAOA State:");
    geometric_log_info("  Qubits: %zu", state->num_qubits);
    geometric_log_info("  Layers (p): %zu", state->p);
    geometric_log_info("  Edges: %zu", state->graph ? state->graph->num_edges : 0);
    geometric_log_info("  Current cost: %.6f", state->current_cost);
    geometric_log_info("  Best cost: %.6f", state->best_cost);
    geometric_log_info("  Iteration: %zu", state->iteration);
}

void qaoa_print_result(const qaoa_result_t* result) {
    if (!result) return;

    geometric_log_info("QAOA Result:");
    geometric_log_info("  Optimal cost: %.6f", result->optimal_cost);
    geometric_log_info("  Iterations: %zu", result->num_iterations);
    geometric_log_info("  Execution time: %.3f seconds", result->execution_time);
    geometric_log_info("  Approximation ratio: %.4f", result->approximation_ratio);
}
