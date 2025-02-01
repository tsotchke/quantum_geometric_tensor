/**
 * @file test_dwave_backend_optimized.c
 * @brief Tests for optimized D-Wave quantum backend
 */

#include "quantum_geometric/hardware/quantum_dwave_backend.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

// Test helper functions
static quantum_problem* create_test_problem() {
    quantum_problem* problem = malloc(sizeof(quantum_problem));
    problem->num_variables = 4;
    problem->num_terms = 0;
    problem->max_terms = 100;
    problem->terms = calloc(problem->max_terms, sizeof(quantum_term));
    return problem;
}

static void cleanup_test_problem(quantum_problem* problem) {
    if (problem) {
        free(problem->terms);
        free(problem);
    }
}

static void test_initialization() {
    printf("Testing D-Wave backend initialization...\n");

    // Setup config
    DWaveConfig config = {
        .solver_name = "DW_2000Q_6",
        .num_reads = 1000,
        .annealing_time = 20.0,
        .chain_strength = 2.0,
        .programming_thermalization = 100,
        .readout_thermalization = 10,
        .auto_scale = true
    };

    // Initialize backend
    DWaveState state;
    bool success = init_dwave_backend(&state, &config);
    assert(success && "Failed to initialize backend");

    // Verify initialization
    assert(state.initialized && "Backend not marked as initialized");
    assert(state.num_qubits > 0 && "Invalid number of qubits");
    assert(state.num_couplers > 0 && "Invalid number of couplers");
    assert(state.qubit_biases && "Qubit biases not allocated");
    assert(state.coupler_strengths && "Coupler strengths not allocated");
    assert(state.qubit_availability && "Qubit availability not allocated");
    assert(state.embedding_map && "Embedding map not allocated");
    assert(state.adjacency_matrix && "Adjacency matrix not allocated");

    cleanup_dwave_backend(&state);
    printf("Initialization test passed\n");
}

static void test_minor_embedding() {
    printf("Testing minor embedding optimization...\n");

    // Setup backend
    DWaveConfig config = {
        .solver_name = "DW_2000Q_6",
        .num_reads = 1000,
        .annealing_time = 20.0,
        .chain_strength = 2.0,
        .programming_thermalization = 100,
        .readout_thermalization = 10,
        .auto_scale = true
    };

    DWaveState state;
    bool success = init_dwave_backend(&state, &config);
    assert(success && "Failed to initialize backend");

    // Create test problem with logical graph
    quantum_problem* problem = create_test_problem();
    
    // Add terms to create a complete graph K4
    quantum_term t1 = {.i = 0, .j = 1, .weight = 1.0};
    quantum_term t2 = {.i = 0, .j = 2, .weight = 1.0};
    quantum_term t3 = {.i = 0, .j = 3, .weight = 1.0};
    quantum_term t4 = {.i = 1, .j = 2, .weight = 1.0};
    quantum_term t5 = {.i = 1, .j = 3, .weight = 1.0};
    quantum_term t6 = {.i = 2, .j = 3, .weight = 1.0};

    problem->terms[problem->num_terms++] = t1;
    problem->terms[problem->num_terms++] = t2;
    problem->terms[problem->num_terms++] = t3;
    problem->terms[problem->num_terms++] = t4;
    problem->terms[problem->num_terms++] = t5;
    problem->terms[problem->num_terms++] = t6;

    // Optimize embedding
    success = optimize_minor_embedding(problem,
                                     state.adjacency_matrix,
                                     state.num_qubits);
    assert(success && "Minor embedding optimization failed");

    // Verify embedding properties
    bool valid_embedding = true;
    for (size_t i = 0; i < problem->num_terms; i++) {
        quantum_term* term = &problem->terms[i];
        if (!are_qubits_adjacent(state.adjacency_matrix,
                                term->physical_i,
                                term->physical_j)) {
            valid_embedding = false;
            break;
        }
    }
    assert(valid_embedding && "Invalid minor embedding");

    cleanup_test_problem(problem);
    cleanup_dwave_backend(&state);
    printf("Minor embedding test passed\n");
}

static void test_chain_strength() {
    printf("Testing chain strength optimization...\n");

    // Setup backend
    DWaveConfig config = {
        .solver_name = "DW_2000Q_6",
        .num_reads = 1000,
        .annealing_time = 20.0,
        .chain_strength = 2.0,
        .programming_thermalization = 100,
        .readout_thermalization = 10,
        .auto_scale = true
    };

    DWaveState state;
    bool success = init_dwave_backend(&state, &config);
    assert(success && "Failed to initialize backend");

    // Create test problem
    quantum_problem* problem = create_test_problem();
    
    // Add terms with varying weights
    quantum_term t1 = {.i = 0, .j = 1, .weight = 0.5};
    quantum_term t2 = {.i = 1, .j = 2, .weight = 1.0};
    quantum_term t3 = {.i = 2, .j = 3, .weight = 1.5};
    problem->terms[problem->num_terms++] = t1;
    problem->terms[problem->num_terms++] = t2;
    problem->terms[problem->num_terms++] = t3;

    // Optimize chain strength
    success = optimize_chain_strength(problem,
                                    state.qubit_biases,
                                    state.num_qubits);
    assert(success && "Chain strength optimization failed");

    // Verify chain strength properties
    double max_weight = 0.0;
    for (size_t i = 0; i < problem->num_terms; i++) {
        if (fabs(problem->terms[i].weight) > max_weight) {
            max_weight = fabs(problem->terms[i].weight);
        }
    }
    assert(problem->chain_strength > max_weight && 
           "Chain strength not strong enough");
    assert(problem->chain_strength < 10 * max_weight && 
           "Chain strength too strong");

    cleanup_test_problem(problem);
    cleanup_dwave_backend(&state);
    printf("Chain strength test passed\n");
}

static void test_annealing_schedule() {
    printf("Testing annealing schedule optimization...\n");

    // Setup backend
    DWaveConfig config = {
        .solver_name = "DW_2000Q_6",
        .num_reads = 1000,
        .annealing_time = 20.0,
        .chain_strength = 2.0,
        .programming_thermalization = 100,
        .readout_thermalization = 10,
        .auto_scale = true
    };

    DWaveState state;
    bool success = init_dwave_backend(&state, &config);
    assert(success && "Failed to initialize backend");

    // Create test problem
    quantum_problem* problem = create_test_problem();
    
    // Add terms
    quantum_term t1 = {.i = 0, .j = 1, .weight = 1.0};
    quantum_term t2 = {.i = 1, .j = 2, .weight = 1.0};
    problem->terms[problem->num_terms++] = t1;
    problem->terms[problem->num_terms++] = t2;

    // Optimize annealing schedule
    success = optimize_annealing_schedule(problem,
                                        state.config.annealing_time);
    assert(success && "Annealing schedule optimization failed");

    // Verify schedule properties
    assert(problem->schedule != NULL && 
           "No annealing schedule created");
    assert(problem->num_schedule_points > 0 && 
           "Empty annealing schedule");
    assert(problem->schedule[0].s == 0.0 && 
           "Schedule doesn't start at s=0");
    assert(problem->schedule[problem->num_schedule_points-1].s == 1.0 && 
           "Schedule doesn't end at s=1");

    cleanup_test_problem(problem);
    cleanup_dwave_backend(&state);
    printf("Annealing schedule test passed\n");
}

static void test_ocean_integration() {
    printf("Testing Ocean SDK integration...\n");

    // Setup backend
    DWaveConfig config = {
        .solver_name = "DW_2000Q_6",
        .num_reads = 1000,
        .annealing_time = 20.0,
        .chain_strength = 2.0,
        .programming_thermalization = 100,
        .readout_thermalization = 10,
        .auto_scale = true
    };

    DWaveState state;
    bool success = init_dwave_backend(&state, &config);
    assert(success && "Failed to initialize backend");

    // Create test problem
    quantum_problem* problem = create_test_problem();
    
    // Add terms
    quantum_term t1 = {.i = 0, .j = 1, .weight = 1.0};
    quantum_term t2 = {.i = 1, .j = 2, .weight = 1.0};
    problem->terms[problem->num_terms++] = t1;
    problem->terms[problem->num_terms++] = t2;

    // Convert to Ocean format
    ocean_problem* ocean = convert_to_ocean(problem);
    assert(ocean && "Failed to convert to Ocean format");

    // Verify Ocean problem structure
    assert(ocean->num_variables == problem->num_variables && 
           "Wrong number of variables");
    assert(ocean->num_terms == problem->num_terms && 
           "Wrong number of terms");

    cleanup_ocean_problem(ocean);
    cleanup_test_problem(problem);
    cleanup_dwave_backend(&state);
    printf("Ocean SDK integration test passed\n");
}

static void test_error_mitigation() {
    printf("Testing D-Wave error mitigation...\n");

    // Setup backend
    DWaveConfig config = {
        .solver_name = "DW_2000Q_6",
        .num_reads = 1000,
        .annealing_time = 20.0,
        .chain_strength = 2.0,
        .programming_thermalization = 100,
        .readout_thermalization = 10,
        .auto_scale = true
    };

    DWaveState state;
    bool success = init_dwave_backend(&state, &config);
    assert(success && "Failed to initialize backend");

    // Create test problem
    quantum_problem* problem = create_test_problem();
    
    // Add terms
    quantum_term t1 = {.i = 0, .j = 1, .weight = 1.0};
    quantum_term t2 = {.i = 1, .j = 2, .weight = 1.0};
    problem->terms[problem->num_terms++] = t1;
    problem->terms[problem->num_terms++] = t2;

    // Execute problem multiple times to build statistics
    quantum_result results[10];
    for (size_t i = 0; i < 10; i++) {
        success = execute_problem(&state, problem, &results[i]);
        assert(success && "Problem execution failed");
    }

    // Verify error mitigation
    double raw_error_rate = 0.0;
    double mitigated_error_rate = 0.0;
    double chain_break_rate = 0.0;

    for (size_t i = 0; i < 10; i++) {
        raw_error_rate += results[i].raw_error_rate;
        mitigated_error_rate += results[i].mitigated_error_rate;
        chain_break_rate += results[i].chain_break_rate;
    }
    raw_error_rate /= 10;
    mitigated_error_rate /= 10;
    chain_break_rate /= 10;

    assert(mitigated_error_rate < raw_error_rate && 
           "Error mitigation not effective");
    assert(chain_break_rate < 0.1 && 
           "Chain break rate too high");
    assert(mitigated_error_rate < state.config.error_threshold && 
           "Error rate above threshold after mitigation");

    cleanup_test_problem(problem);
    cleanup_dwave_backend(&state);
    printf("Error mitigation test passed\n");
}

static void test_error_handling() {
    printf("Testing error handling...\n");

    // Test null pointers
    bool success = init_dwave_backend(NULL, NULL);
    assert(!success && "Should fail with null pointers");

    // Test invalid config
    DWaveConfig invalid_config = {
        .solver_name = NULL,
        .num_reads = 0,
        .annealing_time = -1.0,
        .chain_strength = 0.0,
        .programming_thermalization = 0,
        .readout_thermalization = 0,
        .auto_scale = true
    };

    DWaveState state;
    success = init_dwave_backend(&state, &invalid_config);
    assert(!success && "Should fail with invalid config");

    // Test invalid problem
    DWaveConfig valid_config = {
        .solver_name = "DW_2000Q_6",
        .num_reads = 1000,
        .annealing_time = 20.0,
        .chain_strength = 2.0,
        .programming_thermalization = 100,
        .readout_thermalization = 10,
        .auto_scale = true
    };

    success = init_dwave_backend(&state, &valid_config);
    assert(success && "Failed to initialize with valid config");

    quantum_result result;
    success = execute_problem(&state, NULL, &result);
    assert(!success && "Should fail with null problem");

    // Test problem with too many variables
    quantum_problem* large_problem = create_test_problem();
    large_problem->num_variables = 9999;
    
    success = execute_problem(&state, large_problem, &result);
    assert(!success && "Should fail with too many variables");

    cleanup_test_problem(large_problem);
    cleanup_dwave_backend(&state);
    printf("Error handling test passed\n");
}

int main() {
    printf("Running D-Wave backend tests...\n\n");

    test_initialization();
    test_minor_embedding();
    test_chain_strength();
    test_annealing_schedule();
    test_ocean_integration();
    test_error_mitigation();
    test_error_handling();

    printf("\nAll D-Wave backend tests passed!\n");
    return 0;
}
